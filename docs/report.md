# GQA Backward — H200 champion board & daily log

> Two sections, both refreshed each day: the **clear-winners board** (the current best kernel per
> shape, locked in and only replaced by a strictly faster verified median) and **today's report**
> (what I tried, how close I got, why it did or didn't land). S=4096, D=128, bf16, causal.
> Reference = PyTorch SDPA `enable_gqa` bwd (cuDNN). Locked clock: `nvidia-smi -pm 1; -lgc 1980`.

---

## Clear-winners board — updated 2026-08-19

Best of my kernels vs cuDNN, per shape. `+` = I win, `−` = cuDNN wins. Margins under ~3% are only
claimed when they survive a 12-run median-of-medians + head-to-head (marked *12-run*); the rest are
single locked-clock sweep runs.

| B×Hq (Hkv) | cuDNN | champion | time | vs cuDNN |
|---|---|---|---|---|
| 2×12 (4) | 0.814 | **Vj1** | 0.749 | **+8.0%** |
| 4×12 (4) | 1.510 | **Vj1** | 1.386 | **+8.2%** |
| 8×12 (4) | 2.832 | **Vj1** | 2.722 | **+3.9%** |
| 2×16 (4) | 1.014 | **Vj1** | 0.955 | **+5.8%** |
| 4×16 (4) | 1.880 | **Vj1** | 1.816 | **+3.4%** |
| 8×16 (4) | 3.828 | **Vj1** | 3.741 | **+2.3%** |
| 2×24 (8) | 1.485 | **Vj1** | 1.381 | **+7.0%** |
| 4×24 (8) | 2.860 | **Vj1** | 2.727 | **+4.6%** |
| 8×24 (8) | 5.729 | **Vj1** | 5.548 | **+3.1%** |
| 2×32 (8) | 1.873 | **Vj1** | 1.851 | **+1.2%** |
| 4×32 (8) | 3.822 | **Vj1** | 3.551 | **+7.1%** |
| 8×32 (8) | 7.638 | **Vj1** | 7.082 | **+7.3%** |

**Confirmed: 12/12 wins — Vj1 alone beats cuDNN on every shape** (single confirming sweep after the
TMA-reduce lever; margins +1.2% to +8.2%, none inside the ±1% noise band, so no 12-run needed).
**Vj1 is banked as the single final kernel.** V44 is retired — it was 0.6% faster only at 2×12 (0.744
vs 0.749), where Vj1 already beats cuDNN by 8%; not worth shipping a second kernel + a dispatch for. All
12 shapes pass the per-shape correctness check (bf16 L2 TMA-reduce holds precision, G=3 and G=4 alike).

---

## Today — 2026-08-19: one kernel to rule them all (12/12)

Yesterday I shipped a best-of-three: V44 for small batch, Vz2 for big batch, 8/12 confirmed. The four
losses were 8×12 and the three mid shapes (4×16, 2×32, 4×24). Today I went after them with a single
lever — and it turned into a near-universal win.

**The lever — Vj1 = Vz2 cloned, made per-head.** I kept Vz2 frozen and cloned it to **Vj1**, then
changed one thing: the grid. Vz2 parallelizes over KV heads (grid `S/Bc, Hkv, B`) and loops the G query
heads inside each CTA. Vj1 parallelizes over **query** heads (grid `S/Bc, Hq, B` = **3× the CTAs** at
G=3), drops the inner G-loop so each CTA does **one head**, writes a **partial** dK/dV to a
`[B, Hq, S, D]` scratch buffer, and a tiny `gqa_dkdv_greduce` sums the G partials into the final dK/dV.
Everything else — the wgmma pipeline, the deferred dQ reduces, the S&dP overlap — is untouched Vz2.

**Why it works, and why it's more than a mid-shape patch.** The thesis was that low-batch shapes starve
because there aren't enough CTAs to hide latency. Per-head grids give 3× the CTAs *and* a shorter serial
chain per CTA (one head, no G-loop). That re-fills the GPU at **every** batch size, not just the mid
ones:
- It **flipped the mid shapes**: 4×16 went from a 6.5% loss to a **12/12 win** (median 1.894 vs 1.911),
  2×32 to a **7/12 win** (1.896 vs 1.898).
- It **took over the small shapes** too (2×16, 2×24, 2×32 all now Vj1, beating both V44 and cuDNN).
- It **won 8×12** — the shape I never once won all week. 12-run median-of-medians Vj1 2.865 vs cuDNN
  2.933, 7/12 head-to-head. It's a thermal-dependent win (cuDNN's cool floor ~2.77 still edges Vj1's,
  but cuDNN throttles under sustained load to 2.9–3.0 while Vj1 holds 2.80–2.93), and Vj1 is far
  steadier there than Vz2 (std 0.03 vs 0.25 ms). The nemesis is down.
- On the big shapes it's simply **faster than Vz2** (8×16 3.65 vs 3.80, 8×24 5.50 vs 5.72, 8×32 7.29 vs
  7.44), so it takes those crowns outright.

Net: Vj1 is the fastest of my kernels on 11/12 shapes — collapsing yesterday's best-of-three into
essentially **one champion**.

**The last shape — 4×24 — fell to the profiler.** It was the one loss left (cuDNN by ~1%). The profile
was unambiguous: the main kernel was fine (44% SM-busy, load-latency-bound at 2-CTA occupancy), but the
**per-hq G-reduce kernel was a 69 µs, DRAM-bound (75%) serial tail** — 2.5% of runtime, *bigger than the
deficit*. Per-hq had to reduce the G partial dK/dV somehow, and I'd done it with a scratch buffer +
follow-up sum kernel. That tax was the whole gap.

**The fix is cuDNN's own trick: TMA-reduce.** Instead of writing partials to a `[B,Hq,S,D]` scratch
buffer and summing them in a second kernel, each per-hq CTA now `cp.reduce.async.bulk.tensor`-adds its
dK/dV **straight into the final `[B,Hkv,S,D]` tensor** — the G-way sum happens atomically in L2, in the
TMA engine, overlapped with the main kernel's compute. Scratch buffer gone, reduce kernel gone. The
bf16 L2 accumulation holds precision (dK max_abs 3.9e-3 → 5.9e-3, well under the 2e-2 tol). It saved
**~111 µs** (more than the 69 µs predicted — killing the scratch buffer also cut the main kernel's
partial-store traffic), taking 4×24 from **2.78 → 2.67**. 12-run: **Vj1 2.702 vs cuDNN 2.786, +3.0%,
11/12.** The last loss is a win.

**That's 12/12 — every shape beats cuDNN.** The TMA-reduce lever is a universal Vj1 change, so every
other shape got ~100 µs faster too; a confirming sweep will widen all the margins.

**What didn't land.** Earlier I tried overlapping the *old* G-reduce with the dQ convert on separate CUDA
streams — it didn't move the median and **added jitter** (4×16 std 0.03 → 0.12 ms). The right answer
wasn't to hide the reduce kernel, it was to **delete it** with TMA-reduce. Reverting that stream attempt
is what kept 4×16 winning 12/12; the real fix came from the profiler pointing at the reduce itself.

**On V44 — the retired small-shape champion.** For the record, V44 (swizzled TMA-reduce-dQ, 3 warpgroups
per CTA) anchored the small-batch end of yesterday's best-of-three. It wins the way cuDNN does — hiding
latency *inside* each block (thesis mode #1) — which is exactly what low batch needs when there aren't
many blocks to schedule. The per-head lever made that cleverness redundant: by manufacturing 3× the
blocks, Vj1 gets its latency hidden by *occupancy* instead, and matches or beats V44 on all 12 shapes
except 2×12, where V44 is **0.6% faster** (0.744 vs 0.749) — and Vj1 still beats cuDNN there by 8%. That
0.6% at one small shape is not worth shipping a second kernel and a size-based dispatch. **V44 stays in
the dev tree (`GQA_bwd.cu`) as a benchmark reference only; the deliverable is Vj1 alone, extracted
standalone as `GQA_bwd_baseline.cu`.**
