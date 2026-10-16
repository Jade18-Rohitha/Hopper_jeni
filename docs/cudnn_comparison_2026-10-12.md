# GQA attention vs cuDNN — apples-to-apples results (H200, 2026-10-12)

**Bottom line:** our backward (Vj1d / Vj1) is **0.15–1.70 ms slower than cuDNN on all 12 shapes** (0.59 ms
at 8×12) and lost every one of 144 runs. Forward + backward (V62 + Vj1d/Vj1) is **0.13–1.77 ms slower**. The V62
forward is within ±0.02 ms of cuDNN on small and mid shapes but falls 0.07–0.26 ms behind at large B×Hq.

The earlier "Vj1d beats cuDNN 12/12" result was wrong: it timed cuDNN's **forward + backward** against our
**backward only** (details in [What was wrong before](#what-was-wrong-before)).

---

## Setup

| | |
|---|---|
| GPU | NVIDIA H200 (sm_90), clock locked at 1980 MHz (`nvidia-smi -pm 1; -lgc 1980`) |
| Problem | GQA, causal, bf16, S = 4096, D = 128, 12 shapes (B ∈ {2,4,8} × Hq/Hkv ∈ {12/4, 16/4, 24/8, 32/8}) |
| Ours | `build/bin/gqa_fwd_bwd`: V62 forward → Vj1d or Vj1 backward, using **V62's own O and LSE** |
| cuDNN | `precision/cudnn_time.py`: `F.scaled_dot_product_attention` with `SDPBackend.CUDNN_ATTENTION` forced; the script stops with an error if the result isn't cuDNN's |
| Inputs | Both sides load the same `data/gqa_*.bin` tensors |
| Timing | Same on both sides, `triton.testing.do_bench` style: L2 flushed before every rep, reps queued back to back, one sync at the end |
| Runs | 12 runs per shape, ours and cuDNN alternating, so thermal drift affects both equally. Tables show the median of the 12. "Wins" = runs where ours was faster. |
| Correctness | All checks pass on all 12 shapes: V62 O and LSE; dQ, dK, dV for both Vj1d and Vj1 (tolerance 2e-2) |
| Log | `reports/median12_20261012_095009.log` |

**What each cuDNN number covers:**
- **fwd:** forward only, with no autograd graph.
- **bwd:** `torch.autograd.grad` on a graph built once outside the timer. This covers cuDNN's dO·O preprocess plus its backward kernel. It's the same work as our backward launcher, which runs memset, drowsum, the main kernel, greduce and the dQ convert.
- **fwd + bwd:** `sdpa(...).backward(dO)`, i.e. one training step.

Δ = cuDNN − ours, in ms, so **+ means we are faster** (by that many ms) and − means slower. All times are in ms.

---

## Forward: V62 vs cuDNN

| B×Hq (Hkv) | cuDNN | V62 | Δ | wins |
|---|---|---|---|---|
| 2×12 (4) | 0.197 | 0.182 | **+0.015** | 12/12 |
| 4×12 (4) | 0.348 | 0.351 | −0.003 | 0/12 |
| 8×12 (4) | 0.652 | 0.693 | −0.041 | 3/12 |
| 2×16 (4) | 0.247 | 0.239 | **+0.008** | 12/12 |
| 4×16 (4) | 0.450 | 0.463 | −0.013 | 5/12 |
| 8×16 (4) | 0.868 | 0.942 | −0.074 | 0/12 |
| 2×24 (8) | 0.349 | 0.351 | −0.002 | 2/12 |
| 4×24 (8) | 0.698 | 0.691 | **+0.007** | 6/12 |
| 8×24 (8) | 1.258 | 1.439 | −0.181 | 0/12 |
| 2×32 (8) | 0.480 | 0.463 | **+0.017** | 8/12 |
| 4×32 (8) | 0.853 | 0.919 | −0.066 | 2/12 |
| 8×32 (8) | 1.662 | 1.917 | −0.255 | 0/12 |

## Backward: Vj1d and Vj1 vs cuDNN

Vj1d = deterministic: dK/dV are bit-identical run to run (fixed-order reduce). Vj1 = atomic TMA-reduce for dK/dV, which wobbles by about 1 bf16 ULP run to run.

| B×Hq (Hkv) | cuDNN | Vj1d | Δ | wins | Vj1 | Δ | wins |
|---|---|---|---|---|---|---|---|
| 2×12 (4) | 0.607 | 0.766 | −0.159 | 0/12 | 0.758 | −0.151 | 0/12 |
| 4×12 (4) | 1.104 | 1.411 | −0.307 | 0/12 | 1.383 | −0.279 | 0/12 |
| 8×12 (4) | 2.147 | 2.735 | −0.588 | 0/12 | 2.763 | −0.616 | 0/12 |
| 2×16 (4) | 0.772 | 0.973 | −0.201 | 0/12 | 0.956 | −0.184 | 0/12 |
| 4×16 (4) | 1.439 | 1.842 | −0.403 | 0/12 | 1.815 | −0.376 | 0/12 |
| 8×16 (4) | 2.892 | 3.608 | −0.716 | 0/12 | 3.561 | −0.669 | 0/12 |
| 2×24 (8) | 1.109 | 1.412 | −0.303 | 0/12 | 1.384 | −0.275 | 0/12 |
| 4×24 (8) | 2.163 | 2.734 | −0.571 | 0/12 | 2.767 | −0.604 | 0/12 |
| 8×24 (8) | 4.288 | 5.383 | −1.095 | 0/12 | 5.397 | −1.109 | 0/12 |
| 2×32 (8) | 1.453 | 1.839 | −0.386 | 0/12 | 1.809 | −0.356 | 0/12 |
| 4×32 (8) | 2.878 | 3.603 | −0.725 | 0/12 | 3.551 | −0.673 | 0/12 |
| 8×32 (8) | 5.706 | 7.323 | −1.617 | 0/12 | 7.403 | −1.697 | 0/12 |

## Forward + backward (training step): V62 + Vj1d / V62 + Vj1 vs cuDNN

| B×Hq (Hkv) | cuDNN | V62+Vj1d | Δ | wins | V62+Vj1 | Δ | wins |
|---|---|---|---|---|---|---|---|
| 2×12 (4) | 0.801 | 0.950 | −0.149 | 0/12 | 0.935 | −0.134 | 0/12 |
| 4×12 (4) | 1.492 | 1.770 | −0.278 | 0/12 | 1.758 | −0.266 | 0/12 |
| 8×12 (4) | 2.906 | 3.475 | −0.569 | 0/12 | 3.425 | −0.519 | 0/12 |
| 2×16 (4) | 1.027 | 1.213 | −0.186 | 0/12 | 1.196 | −0.169 | 0/12 |
| 4×16 (4) | 1.956 | 2.338 | −0.382 | 0/12 | 2.456 | −0.500 | 0/12 |
| 8×16 (4) | 3.838 | 4.911 | −1.073 | 0/12 | 4.905 | −1.067 | 0/12 |
| 2×24 (8) | 1.486 | 1.772 | −0.286 | 0/12 | 1.763 | −0.277 | 0/12 |
| 4×24 (8) | 2.903 | 3.470 | −0.567 | 0/12 | 3.420 | −0.517 | 0/12 |
| 8×24 (8) | 5.674 | 6.898 | −1.224 | 0/12 | 7.098 | −1.424 | 0/12 |
| 2×32 (8) | 1.956 | 2.337 | −0.381 | 0/12 | 2.451 | −0.495 | 0/12 |
| 4×32 (8) | 3.815 | 4.906 | −1.091 | 0/12 | 4.846 | −1.031 | 0/12 |
| 8×32 (8) | 7.578 | 9.322 | −1.744 | 0/12 | 9.343 | −1.765 | 0/12 |

---

## Findings

1. **Most of the gap is in the backward.** Vj1d and Vj1 are 0.15–1.70 ms slower than cuDNN on every shape, and
   our backward takes about as long as cuDNN's forward + backward combined.
2. **Vj1 is no longer reliably faster than Vj1d.** It's slower at 8×12, 4×24, 8×24 and 8×32. Vj1d (deterministic) gives
   up nothing.
3. **V62's forward is close to cuDNN on small and mid shapes but scales worse.** It's within +0.017 to −0.013 ms
   up to mid sizes, then −0.066 / −0.074 / −0.181 / −0.255 ms at 4×32 / 8×16 / 8×24 / 8×32. The V62 header's claim of being
   faster than cuDNN at 8×12 doesn't hold up here (0.693 vs 0.652 ms, −0.041 ms, 3/12 runs won).
4. **Unexplained: extra chain overhead when B×Hq = 128.** At 8×16 and 4×32 only, running forward then backward
   costs a steady **+0.37 ms** (0.30–0.46 across all 12 runs) more than forward + backward timed separately.
   Every other shape is ≤ 0.1 ms, and cuDNN's chain overhead is ≤ 0.2 ms everywhere. That's why those two rows
   lose by about 1.07–1.09 ms in the chain, where forward and backward timed separately predict about 0.71 ms.

## Next steps

- nsys trace of the chained 8×16 run to explain the +0.37 ms at B×Hq = 128.
- Take `cudaGetDeviceProperties` out of `launch_gqa_v62`'s per-launch path. It's a slow host-side call.
- ~~Backward: profile Vj1d against cuDNN's backward~~: **done 2026-10-13**, see
  [ncu_vj1d_vs_cudnn_B8_Hq18.md](ncu_vj1d_vs_cudnn_B8_Hq18.md). The gap is entirely the main kernel: the helper
  kernels match cuDNN's one for one. Vj1d's main kernel is L2-bandwidth bound and moves 2× cuDNN's L2 bytes
  per key (Q/dO reloads and dQ reduce-adds), with the same 64×64 tile. cuDNN most likely covers 128 keys per block.
- Forward: look at V62's scaling at large B×Hq. It uses a persistent grid of 132 CTAs.

---

## What was wrong before

- The old "cuDNN" baseline (`_sdpa_bwd` in `precision/baseline_gqa.py`) timed `sdpa(q,k,v).backward(dO)`,
  which is cuDNN's **forward + backward**. Our `gqa_bwd` bench timed **backward only**, with O and LSE precomputed.
  That made the old board compare our backward against cuDNN's forward + backward.
- The cuDNN backend was never forced. Forcing it now gives gradients bit-identical to default SDPA, so the default backend
  was cuDNN all along. The labelling was right, but the timed region was wrong.
- `harness/run_gqa_bwd_v44_bench.py` had the same forward-inside-the-timer bug for every competitor. It also
  syncs after every iteration, which adds about 1 ms of cuDNN host overhead at 8×12 (3.21 ms vs 2.16 ms GPU time).
  Use do_bench-style timing for kernel comparisons.

**Fixes:** `baseline_gqa.py` now prints `cuDNN bwd-only (forced)`. The v44 harness now times only
`autograd.grad`. The new tools are `gqa_fwd_bwd`, `cudnn_time.py`, `run_median12.sh` and `median_summary.py`.

## Reproduce

```bash
cd Hopper
cmake -B build && cmake --build build --target gqa_fwd_bwd -j
bash scripts/run_median12.sh                              # 12 shapes × 12 runs, locked clock
N=2 SHAPES="8 12 4" bash scripts/run_median12.sh          # quick single-shape check
python3 scripts/median_summary.py reports/median12_<stamp>.log
```
