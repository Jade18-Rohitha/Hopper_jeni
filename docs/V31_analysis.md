# V31 Analysis — remove the dedicated sA_t buffer (reuse idle sQ_sw)

**Result: 4.2247 ms median / 780.9 TFLOP/s — ~1.51× off cuDNN. Bit-identical. −2.2% over V30
(4.3208 ms).** 168 regs, 0 spill, **186,448 B smem (−18,432 vs V30's 204,880)**. Correctness:
dQ/dK/dV all `Values match!`, max_abs identical to V30 (1.953e-3 / 3.906e-3 / 3.125e-2).
**Std Dev collapsed 0.0159 → 0.0041 ms** — the run got dramatically more stable, too.

## Why this, and why it wasn't "flat"
User-spotted: after **V23** transpose-eliminated the dQ path, `sA_t`'s ldmatrix→stmatrix
staging role was gone — its ONLY remaining use is the dV/dK **epilogue** writeback stage
(`stage_acc_bf16_s` → `store_stage_vec_s`, stride-72 since V25). But the epilogue runs *after*
the consumer loop, when **every loop buffer is idle**. So a dedicated 18 KB buffer for it is
pure waste: the epilogue now stages through `sQ_sw` (`reinterpret_cast<bf16*>(&sQ_sw[0][0]) +
wg*(64*72)`) — 49 KB, 128-aligned, free by then. `sA_t` deleted entirely.

I predicted **flat** (a pure smem-capacity cleanup, and we're L1TEX-bound, not
capacity-bound). Wrong — 7th flat-miss of the effort. The −2.2% is real and the profile says
it's a **scheduling** win, not a traffic cut.

## Profile — a scheduling/eligibility win (H200-confirmed)

| metric | V30 | **V31** |
|---|---|---|
| Duration (ncu) | 5.20 ms | **5.11 ms** |
| **Executed IPC** | 1.82 | **1.87** |
| Warp cyc / issued inst | 6.52 | **6.34** |
| No-Eligible | 54.5% | **53.2%** |
| Eligible warps/sched | 0.62 | **0.64** |
| L1/TEX throughput | 61.3% | 63.1% (higher — same work, less time) |
| Memory throughput | 61.0% | 62.8% |
| Achieved occupancy | 18.5% | 18.6% (unchanged — still 1 block/SM) |

The mechanism is **warp eligibility**, not bandwidth: occupancy is unchanged (still smem-pinned
to 1 block/SM at 186 KB > the 113 KB that 2 blocks would need), but IPC rose 1.82 → 1.87 and
warp-cycles/issue fell 6.52 → 6.34. The most likely cause: the 18 KB `sA_t` block was perturbing
the **bank/offset layout of the hot loop buffers** (sP/sDS/sS/sdP); removing it let them land in
a cleaner arrangement, so operand fetches complete a hair faster → more eligible warps per cycle.
L1/TEX % actually *rose* (63.1%) — same total work compressed into less time, i.e. higher
utilization, not more traffic. The Std-Dev collapse (0.0159 → 0.0041) is consistent with a
steadier smem access pattern.

## Stall ranking after V31 (pcsamp) — the wall is unchanged
1. **long_scoreboard 137K** — L1TEX operand-fetch waits (the inherent HGMMA swizzled reads).
2. **barrier 81K** — the `consumer_sync` (`bar.sync 1,256`) chain.
3. **wait 73K** — `wgmma.wait_group`.

The bit-identical **L1TEX** cuts are now mined out: RS is dead (transpose is inter-warp), the
functional split is a proven wash (a 128B-swizzled descriptor can't span the two atom-major
atoms; single m64n128 impossible without a 2nd dO copy that erases the saving), and the 1.8-way
shared-load conflicts are the inherent wgmma operand fetches (0 excessive). So V32 turns to the
**#2/#3 scheduling stalls** — barrier-scope / wgmma-overlap in the V26/V27 vein — the last
bit-identical levers on Hopper. (The one bigger lever left, bf16-`sdP` exchange, is
precision-affecting and deferred.)

Cumulative: V13 8.82 → V22 5.89 → V25 4.9353 → V26 4.6853 → V30 4.3208 → **V31 4.2247 (~1.51×)**.
Roofline: cuDNN 2.8 ms = 37% MFU (not the floor); Hopper column-split wall ~3 ms; sub-2.8 needs
row-split → Br=128 → smem-infeasible → Blackwell TMEM.
