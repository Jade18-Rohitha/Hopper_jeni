# V22 Analysis — D-rowsum split (cuDNN-style multi-kernel) — BIGGEST WIN

**Result: 5.8870 ms / 560 TFLOP/s — 2.09× off cuDNN. Bit-identical. −20.3% over V21 (7.39 ms).**
The single largest win of the session, and the first that closed real ground on cuDNN's
*structural* edge rather than shaving the scalar chain.

## What changed
Lift D = Σ_d dO·O out of the fused kernel into a separate light kernel (cuDNN's
`compute_dot_do_o_specialized` move):
- **`compute_drowsum_v22`** — grid-stride over all `B·Hq·S` rows, warp-per-row, replicating
  `producer_drowsum`'s exact accumulation order → D is **fp32-bit-identical** → dQ/dK/dV match
  the reference to the ULP.
- **Main kernel** — deletes the inline `producer_drowsum`, `sO_sw`, and the O swizzled-TMA
  (O fed *only* the D-rowsum; dV uses dO). Producer now does a coalesced load of `d_Drow`
  into `sD` instead. 162 regs, 0 spill, **191,824 B smem (−32,768)**.
- Launcher runs D-kernel → main → convert; benchmark times the **total** (honest vs cuDNN's 2.7).

## Why it won (both the agent and I predicted flat — wrong by 20%)
We called the D-rowsum a "hidden producer parking spot." It compounded **three** wins we
underweighted — and the third was the big one:
1. Dropped the #1 short_scoreboard stall (the D-rowsum shuffle-tree).
2. Freed 32 KB smem.
3. **Deleted a whole tile's O TMA traffic per iteration** → the producer feeds dramatically
   faster. V20 had already proven the producer feed is a real throttle; this unthrottled it.

## Profile — the bottleneck moved from compute to memory

| metric | V21 | **V22** |
|---|---|---|
| **short_scoreboard** stall | 1.99 (#1) | **1.02** (collapsed — D-rowsum gone) |
| **long_scoreboard** stall | 1.40 | **2.55 (#1, jumped)** |
| barrier | 0.91 | 1.40 |
| wait | 1.22 | 0.99 |
| **memory throughput** | 61.9% | **65.68%** (cuDNN 62.8 — we now *exceed* it) |
| SM throughput | 38.0% | 36.4% |

The scalar-chain stall (short_scoreboard) **collapsed** — removing the D-rowsum did exactly
what it should. In its place, **long_scoreboard (global-memory latency) is now #1**, and
**memory throughput 65.68% now exceeds cuDNN's 62.8%.** We've crossed from
compute/latency-bound to **genuinely memory-bound** — the main kernel is now waiting on
global memory (TMA feed + dQ atomics + the D load).

## V23 — the freed smem points straight at it
long_scoreboard = global-latency the pipeline should hide, and we now have **32 KB free**.
PD=4 fits cleanly (191,824 + 32,768 = 224,592 B < 232,448 cap — the deleted `sO_sw` exactly
covers the 4th pipeline stage). V20 proved PD=3-vs-2 was worth −3.65% by killing producer
throttle; **PD=4** gives one more tile of run-ahead to hide the now-dominant global latency.
It's a near-trivial change (bump the `PD` constant) and the profile points right at it.

Cumulative: V13 8.82 → V19 7.77 → V21 7.39 → **V22 5.89 (2.09×)**. Sub-2× in reach.
