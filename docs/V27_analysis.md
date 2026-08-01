# V27 Analysis — pipeline depth PD 3 → 4

**Result: PENDING H200 run.** 156 regs, 0 spill, 229,736 B smem (+33 KB vs V26, under the
232,448 cap — 2.7 KB to spare), same 1-block/SM occupancy (18.75%). **Bit-identical by
construction.**

## What changed
V26 clone; the single change is `constexpr int PD = 4` (was 3) — one more pipeline stage of
producer run-ahead for the `sQ_sw`/`sdO_sw`/`sD` buffers. Costs +33 KB smem (two `[PD][Br·D]`
bf16 tile buffers + the PD-deep `sD`/mbarriers); still fits at the same 1-block/SM occupancy.

## Why
The #1 stall after V26 is the **TMA-feed mbarrier spin** (long_scoreboard 1.98, 80.6% a single
`@!P0 BRA` — consumers parked waiting for the producer). The consumers got **20% faster** across
V22–V26 (5.89 → 4.68 ms), so they drain each tile sooner and hit `mbar_wait` earlier — they now
**starve harder on the feed**. Deeper run-ahead (one more buffered tile) is the lever for a
consumer that outruns its producer. We are *not* HBM-bound (DRAM ~18%), so the feed limit is
latency/issue, not bandwidth — the case where more buffering helps.

## Correctness (bit-identical by construction)
PD changes only how many tiles are in flight, not the per-tile computation or the tile
processing order (the consumer loop stays sequential; the accumulators see the same tile
sequence). Identical basis to V20's bit-identical PD 2→3 bump.

## To confirm on H200
1. `check(── V27 …)` — 0 mismatches.
2. Benchmark **V27 vs V26** (4.6791 ms). If deeper run-ahead hides the feed spin, expect a drop;
   if the producer is issue/throughput-limited rather than buffer-limited, flat.
3. `ncu`: long_scoreboard below 1.98 confirms the feed spin shrank.

Cumulative: V24 4.99 → V25 4.9353 → V26 4.6853 → **V27 (pending)**.
