# GQA Backward — H200 optimization log (2026-08-11)

Starting point: **V44** (swizzled TMA-reduce dQ), the banked best. Goal: close the ~0.3 ms (~1.09×) B=8 gap to cuDNN while holding the small-batch win. Dev box is sm_120 — every benchmark/profile below was run on an H200 remote; the assistant only compiles and reads profiles.

## Headline: the B=8 gap is **not** L2 residency — and the raster is **batch-dependent**

The prior thesis (B=8 gap = L2 residency, inferred from B=2 L2≈91% → B=8 L2≈75%) is **disproven**.

- **L2-aware grid raster** — launch `GRID(S/Bc, Hkv, B)` with `k_tile = blockIdx.x` (fastest-varying) so same-`(b,hkv)` blocks run temporally clustered and their shared Q/dO stays L2-resident. Measured: **L2 hit 75% → 96%** (beats cuDNN's 92%), DRAM 28.4% → 3.8%, misses 7× fewer. **Wall-clock at B=8 did not move.** So the misses were already hidden; L2 was a red herring.
- The real B=8 cost is **intra-CTA exposed latency** (barrier 2.73 + long_scoreboard 2.60 = 58% of 9.2 cyc/issue), **not occupancy** — occupancy is 18.5% (1 CTA/SM), but **cuDNN is also 1 CTA/SM**, so occupancy explains nothing.
- **The raster is a B=8-only win.** A full batch sweep showed it *regresses hard* at low batch — the working set already fits L2 there, so clustering buys no residency but creates a memory hotspot / worse SM load-balance:

| batch | V44 (no raster) | raster | winner |
|---|---|---|---|
| B=2 | **0.788 ms** | 0.952 ms | V44 by ~20% |
| B=4 | **1.568 ms** | 1.701 ms | V44 by ~8% |
| B=8 | 3.310 ms | **3.233 ms** | raster by ~2.4% |

**Fix — adaptive grid order:** the launcher picks the schedule from the Q/dO-vs-L2 footprint — default (B-fastest) for small batch, raster (k_tile-fastest) only when `B·Hq·S·D·4 > 2·L2`. One kernel, two launch configs. This is the shipped baseline.

## Experiments (in order)

Each is bit-identical to V44 (dQ ≤ 1.953e-3, dK ≤ 3.906e-3, dV ≤ 3.125e-2) unless noted.

| # | change | result | kept? |
|---|---|---|---|
| dV/dK stage swizzle | 128B-swizzle the dV/dK epilogue stage | flat (stage already 1-way) | folded |
| dV/dK vec-store | pack the two swizzled bf16 into one STS.32 | tied; cuts store insts, epilogue too small to move clock | **kept** |
| L2 raster | k_tile-fastest grid order | L2 75→96%, but clock flat at B=8; **regresses B=2/B=4** | → made **adaptive** |
| pipelined dQ-reduce | PD=3→2 + triple-buffer + reduce 1 iter behind to delete the split barrier | correct, but **slower** (barrier 2.73→2.91) | reverted |
| **fused store→reduce barrier** | move TMA-reduce after the end barrier; delete the 128-thread split barrier (4→3 loop barriers) | **win**: median 3.231→3.213, barrier 2.73→2.64 | **kept** |
| register-P dS (barrier1+2 merge) | wg1 recomputes its own P (redundant QK) so dS needs no `sP` ldmatrix → merge the two publish barriers (3→2) | correct, but **slower** by ~0.2 ms | reverted |
| register-P + dOV∥QK overlap | issue both wg1 gemms in one wgmma burst, one wait | still slower (~3.43) | reverted |
| **adaptive raster** | pick grid order by L2 footprint | **win at every batch** | **shipped baseline** |

### What the barrier exploration taught us
- The barrier stall (2.73 cyc/issue) is **exposed latency parked at a barrier**, not removable sync. Deleting the biggest barrier via pipelining made the barrier stall go *up* (2.73→2.91) — the latency just resurfaced at the `mbar_wait`s (proof).
- Only a **clean** barrier cut helps — the store→reduce fuse merged two adjacent barriers with zero smem/pipeline change and netted +0.02 ms.
- The two P/dS publish barriers are structurally locked by `dS = P∘(dP−D)` needing P from the other warpgroup. The only unlock (recompute P on wg1) **overloads wg1** (2 gemms vs wg0's 1) — even fully overlapped in one wgmma burst, the redundant QK costs ~0.2 ms and can't be hidden. Compute isn't free at 36% util when you *double* a warpgroup's gemm load.

## Also ruled out (measured)
- **fp32 dQ stage STS.128**: impossible — the 4 fp32 in a swizzle chunk are split across 2 lanes; floored at STS.64.
- **Persistent grid**: L2 *worse* (48%) — the static grid-stride schedule fragments access more than the hardware scheduler.
- **Bc=32 re-tile**: wrong lever — the smem hog is query-side `sQ_sw`/`sdO_sw` (locked to Br=64 by wgmma-m64), not KV-side; Bc=32 saves only ~16 KB and can't reach 2 CTA/SM. And occupancy is moot anyway (cuDNN = 1 CTA/SM).

## Full shape sweep — the deciding data

The raster looked like a win at the Hq=12/B=8 training shape, so I swept the broader head/batch space (SDPA bwd `enable_gqa` = best PyTorch competitor; all S=4096, D=128):

| Shape (B×Hq) | G | V44 (ms) | raster (ms) | SDPA (ms) | best vs SDPA |
|---|---|---|---|---|---|
| 2×16 | 4 | **1.040** | 1.250 | 1.031 | +1% |
| 2×24 | 3 | **1.557** | 1.708 | 1.504 | +3% |
| 2×32 | 4 | **2.089** | 2.254 | 1.899 | +10% |
| 4×16 | 4 | **2.090** | 2.266 | 1.898 | +10% |
| 4×24 | 3 | **3.103** | 3.217 | 2.913 | +6% |
| 4×32 | 4 | ~4.14 | 4.289 | 3.862 | +11% |
| 8×16 | 4 | ~4.14 | 4.292 | 3.865 | +11% |
| 8×24 | 3 | — | 6.328 | 5.743 | +10% |
| 8×32 | 4 | — | 8.350 | 7.693 | +8% |

**The raster loses to plain V44 on every shape I can measure both.** Its one clean win was the narrow Hq=12/B=8 case; everywhere else it regresses (up to +19% at 2×32/4×16, where an adaptive footprint gate mispredicted). The batch-independent tweaks (vec-store, barrier-fuse) were tied-to-marginal and did not survive the broader sweep as robust wins either.

## Final verdict — ship **clean V44**

`src/attention/GQA_bwd_v44.cu` (byte-identical to the banked V44, `swizzled TMA-reduce dQ`) is the deliverable. Everything explored today — L2 raster (adaptive or fixed), pipelined dQ-reduce, register-P barrier merge, store/barrier micro-tweaks — was reverted: none robustly beat V44 across the shape space, and several regressed. V44 is the simple, deterministic, proven kernel.

| batch (Hq=12) | V44 | cuDNN | result |
|---|---|---|---|
| B=2 | 0.796 ms | 0.800 ms | **win** ~0.5% |
| B=4 | 1.576 ms | 1.496 ms | lag ~5% |
| B=8 | 3.244 ms | 2.944 ms | lag ~10% |

Across the broader Hq=16/24/32 sweep, V44 lags SDPA **~1–11%** (tied at 2×16). The B=8 gap is measured to be **intra-CTA exposed latency at fixed 1-CTA occupancy** — the same wall cuDNN sits behind (cuDNN is *also* 1 CTA/SM), which it hides marginally better. Every reachable lever on that last ~0.3 ms — **L2 (red herring, proven), stores (floored), barriers (latency-bound, not sync), occupancy (moot), redundant-compute (overloads a warpgroup)** — has been closed with data. Only `sdpa`/cuDNN are valid apple-to-apple competitors; thunderkittens is forward-only (its backward overflows to NaN on this GQA shape).
