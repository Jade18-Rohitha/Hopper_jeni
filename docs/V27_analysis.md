# V27 Analysis — break the sP-reuse chain (separate sDS + dV overlap)

**Result: PENDING H200 run.** 157 regs, 0 spill, 204,880 B smem (+8 KB for `sDS`, under cap).
**Bit-identical by construction.**

## Why — V26 is consumer-bound, not feed-bound
The V26 profile settled it: the #1 "stall" (26.7% `@!P0 BRA`) is the **producer idling on
`empty[]`** — it fills all pipeline buffers and waits for the slower consumer. The data is on
time; deepening the pipeline can't help (PD=4 regressed). The real wall is the **consumer's
serial 6-GEMM chain**: every GEMM's result lives in the single `sP` buffer, so each step
barriers + waits on the previous (barrier 30% + wgmma-wait 21% of stall), and 18.75% occupancy
(reg/smem-locked, 1 block/SM) can't hide it.

## What changed
The chain serialized because `dS` **overwrote `sP` in place** — forcing dV (which reads `sP`)
to fully complete before dS. V27 gives `dS` its **own output buffer `sDS`**:
- `dS = P ⊙ (dP − D)` reads `sP` (P) and writes **`sDS`** (was in-place `sP`).
- `dK`/`dQ` read `sDS` (the swizzle layout is identical to `sP`, so the Major::MN/Major::K
  descriptor reads are unchanged).
- `dV` is split into **issue** (`run_gemm_dVdK_half_te_issue`, no wait) and **wait**
  (`run_gemm_dVdK_half_te_wait`): issue the dV wgmma, run the dS elementwise, *then* wait dV.
  Since dS no longer touches `sP`, the dV read stays valid across the overlap.
- The **dV→dS WAR barrier is removed** (dV and dS now both only *read* `sP`).

SASS confirms the schedule held: `bar → dV ARRIVE+4×HGMMA → dS (F2FP/STS) → dV DEPBAR wait →
bar` — the dV wgmma overlaps the dS elementwise, and there is **no barrier between them**.
256-thread consumer barriers **9 → 8**.

## Why it should win
Two critical-path cuts at once: the dV wgmma latency is hidden under the dS elementwise (the dS
loop is long; the 4-HGMMA dV is short → its wait resolves for free), and one 256-thread barrier
per tile is gone. Both target the consumer serial chain that the profile identified as the real
bottleneck — not the feed (which is already ahead).

## Correctness (bit-identical by construction)
Same computation; `dS` values are written to `sDS` instead of `sP` (identical swizzle layout),
and dK/dQ read them from `sDS`. dV reads the unchanged `sP`. Deferring the dV wait doesn't
change the accumulator result — the wgmma still completes before the next dependent step. No
new hazard: between dV-issue and dV-wait, `sP` is only read (dS reads it, dK/dQ read `sDS`); the
next overwrite of `sP` is the next tile's `fused_p`, a full iteration and several barriers away.

## To confirm on H200
1. `check(── V27 …)` — 0 mismatches.
2. Benchmark **V27 vs V26** (4.6791 ms).
3. `ncu`: barrier stall below 1.29 (one fewer barrier); wgmma-wait share down (dV wait hidden).

Cumulative: V24 4.99 → V25 4.9353 → V26 4.6853 → **V27 (pending)**.
