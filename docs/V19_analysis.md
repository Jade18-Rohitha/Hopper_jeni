# V19 Analysis — vectorized P-write (bf16×2) — BROKE 8 ms

**Result: 7.7692 ms / 424.5 TFLOP/s — 2.76× off cuDNN. Bit-identical, 153 regs, 0-spill.**
**−5.89% over V18 (8.2550 ms). Second consecutive ~6% elementwise-vectorization win — clears the 8 ms barrier.**

## What changed
Vectorized the `fused_p` P-write (the wg0 half of the S∥dP phase): the two adjacent P values
per row (cols c, c+1) are packed into one **bf16×2 store**, with the two `__expf`→bf16
conversions done as float2 and the causal masks applied per element. `b0`,`b1` are even ⇒
4-byte aligned. SASS: **`STS.U16` 32 → 0** — *every* scalar bf16 store in the kernel is now
gone (both `dS` from V18 and P here). Total 2200 → 2128. Bit-identical.

## Profile — the bottleneck is moving to memory

| metric | V18 | **V19** |
|---|---|---|
| ALU pipe | 19.5% | 20.1% |
| FMA pipe | 11.1% | 11.9% |
| memory throughput | 57.7% | **60.0%** (cuDNN 62.8) |
| short_scoreboard stall | 2.12 | **2.18** (#1) |
| long_scoreboard stall | 0.90 | **1.53** (jumped) |
| wait stall | 1.64 | **1.30** (dropped) |
| barrier stall | 1.70 | **1.23** (dropped) |

The shape of the profile changed meaningfully:
- **`wait` and `barrier` both dropped hard** (1.64→1.30, 1.70→1.23) — vectorizing the two
  critical-path elementwise steps (`dS`, P) shortened the dependency chains *and* cut per-tile
  work, so warps reach the barriers with far less skew.
- **`long_scoreboard` jumped** (0.90 → 1.53) — global/L2 latency (the dQ atomics, the TMA
  loads). As we strip out shared/compute work, the *global*-memory latency becomes the newly
  exposed cost.
- **memory throughput is up to 60%** — approaching cuDNN's 62.8%. We're crossing from
  compute/latency-bound toward **memory-bound**.

## Lesson + where next
The elementwise-vectorization vein delivered two stacked ~6% wins (dS, P) precisely because
both sit on the critical path — and it's now visibly paying down the `wait`/`barrier` stalls.
But the profile says the *next* frontier is shifting: **short_scoreboard (shared-load) is
still #1, and long_scoreboard (global) is rising.**

**V20 candidates:**
1. Remaining elementwise steps still scalar — `store_acc_smem` (the dP write, wg1's half of
   S∥dP) and the `stage_acc` staging. Vectorizing `store_acc` (float2 to `sdP`) directly
   balances the S∥dP phase (P is now packed on wg0; dP is still scalar on wg1). Diminishing
   vs V18/V19 since `wait`/`barrier` already fell, but still shared-traffic on the #1 stall.
2. The rising `long_scoreboard` points at the **dQ atomic / global writeback** path as the
   next distinct target once the elementwise vein is exhausted.
