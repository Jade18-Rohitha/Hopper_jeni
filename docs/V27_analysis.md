# V27 Analysis — remove the redundant dQ-flush WAR barrier (dedicated wg1 buffer)

**Result: PENDING H200 run.** 156 regs (−1 vs V26), 0 spill, 215,120 B smem (+18 KB, under
232,448 cap). **Bit-identical by construction.**

## The failed first attempt, and why
V27-v1 tried to *scope* the two flush barriers per-warpgroup (wg0→`bar 3`, wg1→`bar 4`). It
**regressed −1.4%** (4.6826 → 4.7492 ms). Lesson: **scoping a *balanced* barrier is pure
overhead.** V26 won because wg1 had an entire independent dP-GEMM to overlap while wg0 waited
on the LSE load. At the flush, both wgs do identical O(64-atomic) work and immediately
reconverge at the next barrier — there is no independent work to hide, so splitting one
256-barrier into two 128-barriers just adds a second barrier ID and lets the warpgroups drift,
pushing the wait to the next full barrier. No overlap, only cost.

## The proper fix — remove the barrier, don't scope it
The dQ epilogue had **two** intra-wg barriers:
- `store_acc → flush` (RAW) — genuinely needed (fragment store, coalesced read cross-thread).
- `flush → next-iter overwrite` (WAR) — **redundant *if* the flush buffer isn't reused.**

wg0's dQ used its own `sS`; wg1's dQ **reused `sdP`** (its dP buffer), which created a WAR with
the *next* tile's dP-store into `sdP` — the only reason the WAR barrier existed. Fix: give wg1
a **dedicated `sS1`** buffer (`float sS1[Br*72]`, +18 KB, budget allows). Now:
- wg0 dQ → `sS`, wg1 dQ → `sS1`; neither is touched again until the *next* tile's dQ-store,
  which is already guarded by the next iteration's full "after fused_p" `bar 1,256`.
- So the WAR barrier is **removed entirely** (both wgs), not scoped.
- The RAW barrier reverts to a plain full `consumer_sync()`.

SASS: 256-thread consumer barriers **9 (V26) → 8 (V27)**; the per-wg `bar 4` is gone; the V26
sLSE `bar 3` scope is kept. Regs 157→**156**.

## Why it should win (unlike v1)
Removing the WAR barrier is a real work cut: one fewer 256-thread rendezvous per tile, *and*
each wg's flush (fire-and-forget global atomics + LDS reads) now flows straight into the next
`mbar_wait` — overlapping the flush with the TMA-feed wait that is the #1 stall
(long_scoreboard 1.98). No predicate, no drift, no added barrier.

## Correctness (bit-identical by construction)
Computation unchanged; wg1's dQ simply lives in `sS1` instead of `sdP`. The removed WAR is
covered by the next tile's existing full barrier (nothing touches `sS`/`sS1` between the flush
and the next dQ-store, which sits after 4 full barriers). Fire-and-forget atomics to
`d_dq_accum` are order-independent (distinct rows).

## To confirm on H200
1. `check(── V27 …)` — 0 mismatches.
2. Benchmark **V27 vs V26** (4.6826 ms) — expect a small improvement (one barrier removed +
   flush/TMA-feed overlap), recovering v1's regression and then some.
3. `ncu`: barrier stall below 1.29; one fewer `bar.sync 1,256` per tile.

Cumulative: V13 8.82 → V24 4.99 → V25 4.9353 → V26 4.6853 → **V27 (pending)**.
