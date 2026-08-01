# V27 Analysis — per-warpgroup flush barriers (named barriers 3 & 4)

**Result: PENDING H200 run.** 157 regs, 0 spill, 196,688 B smem (all = V26). **Bit-identical
by construction.**

## Why — V26 proved the mechanism
V26 (wg0-scoped sLSE barrier) gave **−5.7%** by pure scheduling: instruction count was flat,
but IPC rose **1.77 → 1.87** (warp-cyc/issue 6.70 → 6.33, no-eligible 55.7% → 53.2%). Letting
wg1 do useful work instead of parking at a barrier it has no stake in **keeps the scheduler
fed**. V27 applies the same lever to the next intra-wg barriers.

## What changed
The dQ-store → flush region ends the tile with two 256-thread `consumer_sync` (`bar.sync
1,256`), but it is **fully intra-warpgroup**:
- `store_acc → (wg==0)?sS:sdP` — wg0 writes `sS`, wg1 writes `sdP`
- `atomic_flush → (wg==0)?sS:sdP` — wg0 reads `sS`, wg1 reads `sdP`

Neither wg touches the other's buffer here (the cross-wg `sdP` read in `dS` is a *different*
region, still guarded by the kept full barriers). So both barriers become **per-wg**:
`consumer_sync_perwg(wg)` → wg0 uses `bar.sync 3,128`, wg1 uses `bar.sync 4,128`. **Separate
IDs are required** — a shared ID would release on a mixed wg0+wg1 arrival set. SASS confirms
`@!P3 BAR.SYNC 0x3,0x80` ×2 (wg0) and `@P3 BAR.SYNC 0x4,0x80` ×2 (wg1); 256-thread consumer
barriers **10 (V25) → 9 (V26) → 7 (V27)**.

Now the two warpgroups drive their independent dQ-halves through the O(64-atomic) flush
without waiting on each other — the slower wg's flush latency no longer stalls the faster.

## Correctness (bit-identical by construction)
Computation unchanged; only barrier *scope* changes. Each wg still enforces its own RAW
(store→flush) and WAR (flush→next-iter overwrite) on its own buffer via its own 128-thread
barrier. `sS` is wg0-exclusive; `sdP` in this region is wg1-exclusive. Named IDs 3/4 each get
exactly one warpgroup's 128 threads → complete, no deadlock, no mixed-arrival hazard.

## To confirm on H200
1. `check(── V27 …)` — 0 mismatches (bit-identical).
2. Benchmark **V27 vs V26** (4.6853 ms). Given the mechanism is IPC/eligibility and this is a
   bigger slice (2 barriers around the O(64-atomic) flush vs V26's one LSE load), expect
   continued improvement — but the flush may overlap less latency than the LSE load did.
3. `ncu`: IPC should rise above 1.87; barrier stall below 1.29.

Cumulative: V13 8.82 → V24 4.99 → V25 4.9353 → V26 4.6853 → **V27 (pending)**.
