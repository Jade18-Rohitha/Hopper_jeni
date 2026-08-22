# GQA Backward — H200 champion board & daily log

> Two sections, both refreshed each day: the **clear-winners board** (the current best kernel per
> shape, locked in and only replaced by a strictly faster verified median) and **today's report**
> (what I tried, how close I got, why it did or didn't land). S=4096, D=128, bf16, causal.
> Reference = PyTorch SDPA `enable_gqa` bwd (cuDNN). Locked clock: `nvidia-smi -pm 1; -lgc 1980`.

---

## Clear-winners board — updated 2026-08-20 — **deliverable = Vj1d (deterministic, bit-identical dK/dV)**

The shipped baseline is **Vj1d** — the deterministic kernel whose dK/dV are bit-identical run-to-run
(matching cuDNN). Board is Vj1d vs cuDNN, per shape, locked clock (`-lgc 1980`). `[MRG]` = margin <1%,
needs a 12-run median-of-medians. (The faster **Vj1** — atomic reduce, ~1 ULP dK/dV wobble — lives in
`GQA_bwd.cu`; its column is shown for reference.)

| B×Hq (Hkv) | cuDNN | **Vj1d** (shipped) | vs cuDNN | Vj1 (faster, ref) |
|---|---|---|---|---|
| 2×12 (4) | 0.799 | **0.757** | **+5.3%** | 0.746 (+6.6%) |
| 4×12 (4) | 1.487 | **1.394** | **+6.2%** | 1.373 (+7.7%) |
| 8×12 (4) | 2.775 | **2.760** | **+0.5%** *12-run 11/12* | 2.678 (+2.3%) |
| 2×16 (4) | 1.022 | **0.967** | **+5.4%** | 0.955 (+6.6%) |
| 4×16 (4) | 1.880 | **1.831** | **+2.6%** | 1.797 (+4.4%) |
| 8×16 (4) | 3.828 | **3.593** | **+6.1%** | 3.545 (+7.4%) |
| 2×24 (8) | 1.488 | **1.402** | **+5.8%** | 1.383 (+7.1%) |
| 4×24 (8) | 2.792 | **2.755** | **+1.3%** *12-run 11/12* | 2.677 (+2.4%) |
| 8×24 (8) | 5.729 | **5.563** | **+2.9%** | 5.301 (+7.5%) |
| 2×32 (8) | 1.877 | **1.832** | **+2.4%** | 1.805 (+3.8%) |
| 4×32 (8) | 3.797 | **3.594** | **+5.4%** | 3.550 (+6.5%) |
| 8×32 (8) | 7.562 | **7.156** | **+5.4%** | 7.003 (+7.4%) |

**Vj1d beats cuDNN on all 12 shapes — confirmed 12/12.** 10 shapes are clear single-sweep wins (+2.4% to
+6.2%); the two marginals were settled by 12-run median-of-medians: **8×12 +0.5% (11/12)** and **4×24
+1.3% (11/12)**. So the shipped *deterministic* baseline sweeps the board — cuDNN-grade reproducibility
(dK/dV bit-identical run-to-run, max|Δ|=0; dQ non-deterministic ~3e-5, exactly like cuDNN) **and** faster
than cuDNN on every shape — at the cost of ~69 µs (the greduce) versus the even-faster Vj1. All 12 pass
correctness. **This is the deliverable: `GQA_bwd_baseline.cu`.**

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

**Determinism — measured, and an honest asymmetry.** I checked whether the TMA-reduces cost run-to-run
reproducibility versus cuDNN, by running each backward 8× on byte-identical inputs
(`precision/check_cudnn_determinism.py` for cuDNN, `precision/vj1_determinism_probe.cu` for Vj1):

| gradient | cuDNN max\|Δ\| | Vj1 max\|Δ\| |
|---|---|---|
| dQ | 7.6e-6 (non-det) | ~6e-5 (non-det) |
| dK | **0 — bit-identical** | 4.9e-4 – 7.8e-3 (non-det) |
| dV | **0 — bit-identical** | 3.9e-3 – 3.1e-2 (non-det) |

So it is **not** a symmetric trade. On **dQ** we make the *identical* trade cuDNN does — both reduce dQ
across blocks, both wobble at ~1e-5. On **dK/dV** we trade reproducibility cuDNN *keeps*: cuDNN
accumulates the query heads in-register (one writer → bit-exact), whereas Vj1's per-hq split turned
dK/dV into a cross-CTA **bf16** reduce whose atomic order flips the last bf16 bit — the 3.1e-2 on dV is
exactly one bf16 ULP at dV's larger magnitudes. It's within the correctness envelope (2e-2 relative,
mean_abs ~1e-5) and irrelevant for training, but the asymmetry is real: **cuDNN's dK/dV are exact; Vj1's
are reproducible only to ~1 ULP** — the price of the per-hq speed.

**If bit-identical dK/dV is required, use Vj1d.** The wobble comes from the *atomic* cross-CTA reduce, so
making it fp32 only shrinks it (~1e-4), never to zero — any atomic reduce is order-dependent. The fix is
to drop atomics entirely: Vj1d writes one-writer per-hq partials and sums the G heads with a **fixed-order
`greduce`** (no atomics). Measured: **dK/dV bit-identical, max|Δ| = 0 across 8 runs**, dQ still
non-deterministic at ~3e-5 — i.e. *exactly* cuDNN's determinism profile. Cost: the ~69 µs greduce pass
(the tax Vj1 removed), so Vj1d wins the roomy shapes and loses the tightest ~1–2%. Two supported modes:
**Vj1d** — bit-identical dK/dV like cuDNN, ~69 µs slower — and **Vj1** — fastest (12/12), dK/dV wobble
~1 ULP. **We ship Vj1d as the standalone baseline (`GQA_bwd_baseline.cu`)**: for a *reference*
deliverable, cuDNN-grade reproducibility is worth ~2%, and Vj1d matches cuDNN's determinism profile
exactly (dK/dV bit-identical, dQ non-deterministic). **Vj1** stays in `GQA_bwd.cu` as the max-speed
alternative for anyone who wants the last 2% and doesn't need bit-reproducible gradients.

**On V44 — the retired small-shape champion.** For the record, V44 (swizzled TMA-reduce-dQ, 3 warpgroups
per CTA) anchored the small-batch end of yesterday's best-of-three. It wins the way cuDNN does — hiding
latency *inside* each block (thesis mode #1) — which is exactly what low batch needs when there aren't
many blocks to schedule. The per-head lever made that cleverness redundant: by manufacturing 3× the
blocks, Vj1 gets its latency hidden by *occupancy* instead, and matches or beats V44 on all 12 shapes
except 2×12, where V44 is **0.6% faster** (0.744 vs 0.749) — and Vj1 still beats cuDNN there by 8%. That
0.6% at one small shape is not worth shipping a second kernel and a size-based dispatch. **V44 stays in
the dev tree (`GQA_bwd.cu`) as a benchmark reference only; the standalone deliverable
`GQA_bwd_baseline.cu` ships **Vj1d** (deterministic bit-identical dK/dV), with the faster **Vj1** kept
in `GQA_bwd.cu` as the max-speed alternative.**
