# V26 Analysis — wg0-scoped sLSE barrier (named barrier id 3)

**Result: PENDING H200 run** (dev box sm_120 — compile-verify only). 157 regs, 0 spill,
196,688 B smem (all = V25). **Bit-identical by construction.**

## Why this — the V25 profile settled the target
After V24/V25 mined out the shared-conflict lever, the V25 profile showed the wall is now
**feed/sync**, tensor only 20% active:
- long_scoreboard **2.07** (#1) — 80.6% is one `@!P0 BRA` = the mbarrier TMA-feed spin.
- **barrier 1.42** (#2) — the 7 `consumer_sync` (`bar.sync 1,256`) per tile.
- wait 1.05 (#3) — `wgmma.wait_group`.

The consumer chain gates the GEMMs on the reused `sP` buffer; the 7 barriers each guard a
real hazard (sP RAW/WAR, the `empty[]` producer-reuse, the `sS`/`sdP` flush) — **none is
free to delete.** But one can be *scope-narrowed*.

## What changed
The very first barrier of the tile — `@8351`, right after the `sLSE` load — was a full
256-thread `bar.sync 1,256`, yet the hazard it guards is **wg0-only**:
- `@8350`: threads `tid<64` write `sLSE[tid] = d_LSE[…]` (a **global load**, ~400-cycle latency)
- `@8358`: `fused_p` (all of wg0) reads `sLSE[row]` for `exp(S−LSE)`

wg1 (`dP = dO·Vᵀ`) never touches `sLSE`. So `@8351` becomes a **wg0-scoped 128-thread barrier**
(`bar.sync 3,128`, id 3 free: 0=syncthreads/1=consumers/2=producer). New `consumer_sync_wg0()`;
the call site is `if (wg == 0) consumer_sync_wg0();`. SASS confirms `@!P0 BAR.SYNC 0x3,0x80`.

**The gain is latency-hiding, not a cheaper barrier:** wg1 no longer waits at `@8351` for wg0's
`sLSE` global-load latency — it starts its independent `dP = dO·Vᵀ` immediately (inputs
`sdO_sw`/`sV_sw` are already mbar-ready) and overlaps the ~400-cycle LSE load. The two
warpgroups re-converge at the next `bar.sync 1,256` (`@8365`), before any `sdP` read.

## Correctness (bit-identical by construction)
The computation is unchanged; only wg1's *timing* changes. wg0 still enforces the `sLSE`
write→read RAW via `bar 3,128` (all 128 wg0 threads arrive). wg1 skips a barrier it has no
stake in and re-synchronizes at `@8365` before the first cross-wg `sP`/`sdP` dependency. No
named-barrier collision (id 3 used by exactly wg0's 128 threads → completes, no deadlock).

## To confirm on H200
1. `check(── V26 …)` — must be 0 mismatches (bit-identical vs reference).
2. Benchmark **V26 vs V25** (4.9353 ms) — expect small: this hides one warpgroup's LSE-load
   latency at one of 7 barriers. Could be a few %, could be flat if wg1's dP-GEMM already
   overlapped enough. If it moves, the same wg-scoping may extend to the flush barriers
   (`@8400`/`@8402`, which are intra-wg: wg0↔`sS`, wg1↔`sdP`) as V27.
3. `ncu` barrier stall — `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`
   should dip below 1.42 if wg1's wait shrank.

Cumulative: V13 8.82 → V22 5.89 → V24 4.99 → V25 4.9353 → **V26 (pending)**.
