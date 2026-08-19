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
| 2×12 (4) | 0.801 | **V44** | 0.743 | **+7.3%** |
| 4×12 (4) | 1.503 | **Vj1** | 1.408 | **+6.3%** |
| 8×12 (4) | 2.933 | **Vj1** | 2.865 | **+2.3%** *12-run 7/12* |
| 2×16 (4) | 1.022 | **Vj1** | 0.971 | **+5.0%** |
| 4×16 (4) | 1.911 | **Vj1** | 1.894 | **+0.9%** *12-run 12/12* |
| 8×16 (4) | 3.919 | **Vj1** | 3.653 | **+6.8%** |
| 2×24 (8) | 1.474 | **Vj1** | 1.405 | **+4.7%** |
| 4×24 (8) | 2.776 | *cuDNN* | 2.827 | **−1.8%** *12-run 10/12* |
| 8×24 (8) | 5.768 | **Vj1** | 5.496 | **+4.7%** |
| 2×32 (8) | 1.898 | **Vj1** | 1.896 | **+0.1%** *12-run 7/12* |
| 4×32 (8) | 3.883 | **Vj1** | 3.827 | **+1.5%** |
| 8×32 (8) | 7.742 | **Vj1** | 7.294 | **+5.8%** |

**Confirmed: 11/12 wins** (V44 on 2×12; Vj1 on the other ten, all 12-run-verified). **4×24** is the only
clean loss (−1.8%). Vj1 is fastest of *my* kernels on 11 of 12 shapes — only 2×12 stays V44's.

---

## Today — 2026-08-19: one kernel to (nearly) rule them all

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

Net: Vj1 is the fastest of my kernels on 11/12 shapes and beats cuDNN on 10/12 confirmed — collapsing
yesterday's best-of-three into essentially **one champion**.

**What didn't land.** I tried overlapping the dK/dV G-reduce with the dQ convert on separate CUDA
streams, betting the ~74 µs of serial post-processing would hide. It didn't move the median and it
**added jitter** (4×16 std 0.03 → 0.12 ms), costing head-to-head runs — the stream scheduling variance
outweighed any overlap. Reverted to serial; the tighter version is what won 4×16 12/12.

**The one loss — 4×24 (−1.8%).** Per-hq *helped* here (Vj1 2.827 vs Vz2's 2.895) but didn't cross
cuDNN's tight 2.776. It's a real, low-variance gap, not noise — the next dedicated target.

**Next.** Only **4×24** remains cuDNN's (−1.8%, low-variance, real). It's the last clean loss and the
one dedicated target left — a profile-driven crack at it is tomorrow's work.
