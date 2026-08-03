# V26 Analysis — wg0-scoped sLSE barrier (named barrier id 3)

**Result: 4.6853 ms (median) / 703.6 TFLOP/s — ~1.69× off cuDNN (broke 1.7×). Bit-identical.
−5.71% over V25 (4.9687 ms).** 157 regs, 0 spill, 196,688 B smem (all = V25). Blew past the
honest "couple %, could be flat" estimate — hiding wg1's ~400-cycle LSE-load latency behind
its dP-GEMM was a much bigger lever than expected (4th under-call in a row: D-split, store-
conflict, bf16-stage, this). Correctness: dQ/dK/dV all `Values match!`, max_abs identical.

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

## Profile — a pure scheduling win (H200-confirmed)

| metric | V25 | **V26** |
|---|---|---|
| Elapsed cycles | 8.23 M | **7.83 M (−4.9%)** |
| **Executed IPC** | 1.77 | **1.87** |
| warp-cyc / issued inst | 6.70 | 6.33 |
| No-eligible | 55.7% | 53.2% |
| barrier stall | 1.42 | 1.29 |
| instructions | ~flat | ~flat |

The mechanism is **scheduling, not a cheaper barrier**: instruction count is unchanged, but
letting wg1 do its dP-GEMM instead of parking on wg0's LSE load raised the *eligible*-warp
count, so IPC rose 1.77 → 1.87 (warp-cyc/issue 6.70 → 6.33). The barrier stall itself only dips
1.42 → 1.29 — the win is the extra warp eligibility it buys. That confirmed the lever for V27:
**wg-scope a barrier whose hazard is intra-warpgroup so the freed wg overlaps latency.** (The
flush barriers `@8400`/`@8402` looked like the next target — but they turned out *balanced*, so
scoping them regressed; V27 instead hid the dV wgmma under the dS elementwise.)

Cumulative: V13 8.82 → V22 5.89 → V24 4.99 → V25 4.9353 → **V26 4.6853 (~1.69×)**.
