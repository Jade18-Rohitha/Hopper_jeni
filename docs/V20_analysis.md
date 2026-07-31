# V20 Analysis — swizzled-D-rowsum smem-free + 3-deep pipeline — STRUCTURAL WIN

**Result: 7.4935 ms / 439.5 TFLOP/s — 2.67× off cuDNN. Bit-identical, 153 regs, 0-spill.**
**−3.65% over V19 (7.7777 ms). First win from breaking the 227 KB smem wall.**

## What changed — two phases
**Phase 1 — free 32,768 B (bit-identical by construction):** the producer D-rowsum
`D[r] = Σ_d dO[r,d]·O[r,d]` no longer needs plain copies. It's now computed from the
*already-loaded* swizzled `sdO_sw` plus a new swizzled `sO_sw` (O loaded via the sw128 TMA
instead of plain). Since a row-sum is order-independent and both buffers share the identical
swizzle layout, the per-element pairing is preserved → **D is fp32-bit-identical** (same
columns, same order, same accumulation). The redundant plain `sdO_pl` is **deleted (−32 KB)**;
`sO_pl` → `sO_sw` (same size). The 224/227 KB wall — which killed retile, cross-tile, and
ping-pong all session — is broken, at zero register cost.

**Phase 2 — spend it on a 3-deep pipeline:** `sQ_sw`/`sdO_sw` `[2]`→`[3]`, mbarriers `[3]`,
producer runs **3 tiles ahead instead of 2**. `sO_sw`/`sD` stay `[2]` (safe by program-order +
the `empty[]` gate, no new mbarrier). `constexpr int PD` toggles depth (3 shipped).

## Profile — the pipeline killed the producer throttle

| metric | V19 | **V20** |
|---|---|---|
| **long_scoreboard** stall | 1.53 (#2) | **0.86 (#4)** — −44% |
| short_scoreboard stall | 2.18 (#1) | 2.04 (#1) |
| wait stall | 1.30 | 1.28 |
| barrier stall | 1.23 | 1.20 |
| memory throughput | 60.0% | **61.1%** (cuDNN 62.8) |
| SM throughput | ~36% | 39.3% |

The localization had shown long_scoreboard was 87% consumers **spinning on the `full[]`
mbarrier** (producer-feed throttle). Deepening the pipeline pre-fills one more slot → the spin
collapses (1.53 → 0.86). **This proves it was NOT a pure parking spot — the producer feed was a
real throttle, and the smem-unblock converts to wall-clock.** The structural direction is live.

## Where the bottleneck moved — and V21
With the producer throttle gone, the wall is now cleanly the **consumer RAW chain**:
short_scoreboard 2.04 (#1, the serial softmax→dS→readout FADD/staging cluster) + wait 1.28.
This is the intrinsic per-tile dependency chain — and the one lever that *hides* it rather than
shortens it is **cross-tile overlap**: compute tile N+1's S∥dP GEMM *while* tile N runs its
elementwise readout.

**V21 = cross-tile S/dP overlap.** Two obstacles to solve: (1) it needs ~33 KB (double-buffered
`sS`/`sdP`) — V20 spent the freed 32 KB on the pipeline (9 KB headroom now), so V21 must free
more or rebalance depth-vs-overlap; (2) the V14 attempt hit **WG.DP serialization** (async
wgmma pending across the wg0/wg1 divergence got ptxas-serialized, −11%) — V21 must structure the
overlap to keep the S∥dP warpgroup concurrency intact. It's the real path toward 6 ms, but it's
the hardest lever — worth a careful design pass before building.
