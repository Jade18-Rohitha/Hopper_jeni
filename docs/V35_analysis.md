# V35 Analysis — fuse dP+dS (delete the `sdP` round-trip)

**Result: 3.9615 ms median / 834.0 TFLOP/s — bit-identical, −0.5% over V34 (3.9812 ms).** 168 regs,
0 spill, 186,960 B smem — **identical resource profile to V34**. dQ/dK/dV all `Values match!`,
max_abs unchanged (1.953e-3 / 3.906e-3 / 3.125e-2). Cumulative: V13 8.82 → V34 3.98 → **V35 3.96**.

## Why — a *subtraction* (the pattern that works)

V34 column-splits the tile: wg0 computes `S→P→sP`, wg1 computes `dP` and **stored it to `sdP`**, then a
256-thread loop read **`sP + sdP`** to form `dS = P⊙(dP−D) → sDS`. That `sdP` store + read is pure
column-split tax. V35 deletes it: **wg1 keeps its `dP` fragment in registers** (`dPacc[32]`) and computes
`dS` right there via `fuse_dS_from_sP` — reading only `sP` for P, writing `sDS`. `dP` stays fp32 the whole
way (V34's `sdP` round-trip also never rounded it), swizzle indices are identical to V34's dS loop and to
`store_dS_reg_v35` → **bit-identical**. The `sdP` buffer stays (still wg1's dQ-flush stage) but now carries
**zero dP traffic**.

## Profile — the traffic dropped exactly as designed

| metric | V32 | V34≈V32 | **V35** | Δ vs V32 |
|---|---|---|---|---|
| Elapsed cycles | 6,699,952 | ~6.66 M | **6,621,876** | **−1.2%** |
| L1/TEX throughput | 65.79% | — | **59.83%** | **−6.0 pts** |
| Memory throughput | 65.57% | — | 59.63% | −5.9 pts |
| Compute (SM) throughput | 47.99% | — | 47.08% | −0.9 |
| DRAM throughput | 21.78% | — | 21.99% | ~flat |
| **Shared-load requests** | 72,015,872 | — | **67,092,480** | **−6.8% (−4.9 M)** |
| Shared-load wavefronts | 132,388,161 | — | 119,714,523 | **−9.6% (−12.7 M)** |
| Bank conflicts | 34,234,811 | — | 32,822,559 | −4.1% |
| **Executed instructions** | 1,693,939,316 | — | **1,643,784,060** | **−50.2 M** |

The `sdP` store (`store_acc_smem_v6`) and the dS-loop's `float4 sdP` reads are gone: **−4.9 M shared-load
requests, −12.7 M wavefronts, −50 M instructions, and L1/TEX fell 6 points (65.8→59.8%).** This is precisely
the L1TEX subtraction the fusion promised — the profile confirms it cleanly.

## But the wall barely moved — and *that* is the real finding

Cutting 6.8% of shared loads bought only −1.2% cycles. The reason is a **regime shift**: V32 was
L1TEX-bandwidth-bound (L1/TEX 65.8%, the clear #1). V35 cut enough L1TEX traffic that **bandwidth is no
longer the sole wall** — the kernel is now **latency/scheduling-bound**:

| scheduler metric | V32 | **V35** |
|---|---|---|
| No-Eligible | 51.85% | **52.66%** |
| Eligible warps / scheduler | 0.70 | **0.65** |
| Executed IPC | 1.93 | **1.89** |
| Warp-cyc / issued | 6.16 | **6.26** |

Eligibility went *down*, not up. The freed L1TEX port isn't converted to throughput because there aren't
enough eligible warps to hide the wgmma-completion + barrier latency (occupancy is pinned at **1 block/SM,
18.5%** by the 186 KB smem). Worse, the fusion introduced a **load imbalance the profiler explicitly flags**
("avoid possible load imbalances due to highly different execution durations per warp"): the dS elementwise
is now **wg1-only (128 threads)** while **wg0 idles**, shrinking the eligible-warp pool. That imbalance gives
back most of the L1TEX saving — net ~flat.

So V35 is a *correct* subtraction that the profile rewards (−50 M instructions, −6 pts L1TEX) but the wall
does not — because we've crossed out of the bandwidth regime into the latency regime.

## Where next — the lever changed

"Cut L1TEX traffic" has stopped paying (we did, it went flat). The bottleneck is now **warp-latency /
scheduling** (No-Eligible 52.7%, 0.65 eligible warps). The bit-identical levers that help *there*:

- **Scope/kill barriers** (V36 in flight): the dQ-flush's two 256-thread `consumer_sync`s are intra-warpgroup
  (`sS` is wg0-private, `sdP` is wg1-private) → scope to 128-thread (bar 3 / bar 4). Fewer threads per
  barrier + decoupled wgs → the barrier stall drops and eligible-warps can rise. **This profile is exactly
  what motivates V36.**
- **Kill the dS load imbalance**: wg0 idles during wg1's 128-thread dS. Giving wg0 useful work there (or
  rebalancing the elementwise) would raise eligible-warps — but any scheme that hands `dP` back through smem
  re-adds the `sdP` traffic V35 just deleted, so it must stay in-register. Open.
- Occupancy (1 block/SM) is the ceiling on latency-hiding and is pinned by smem; raising it needs the smem
  footprint down, which the design has already minimized.

Roofline unchanged: cuDNN 2.8 ms = 37% MFU; the Hopper column-split wall is ~3 ms; sub-2.8 needs row-split →
Br=128 → smem-infeasible → Blackwell TMEM. V35 stands as the new baseline — the cleanest column-split we have,
and the first version that is provably **past** the L1TEX-bandwidth wall and into the latency regime.
