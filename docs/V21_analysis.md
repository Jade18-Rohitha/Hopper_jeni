# V21 Analysis — causal-mask specialization

**Result: 7.3929 ms / 445.4 TFLOP/s — 2.63× off cuDNN. Bit-identical, 159 regs, 0-spill.**
**−1.49% over V20 (7.5050 ms).**

## What changed
Only the diagonal tile (`qc == qc0`) runs the masked `fused_p`; all off-diagonal tiles
(~97%, where every k < q so the causal mask never fires → bit-identical) run a stripped
`fused_p_nomask_v21` — no `gc/gr` global-index math, no `ISETP`/`SEL` on the exp path. Uniform
branch (`qcC == qc0`), no divergence. +6 regs (both variants inlined), static instr up
(both paths present) but the dynamic hot path is shorter.

## Profile — where the win came from, and where the wall is now

| stall | V20 | **V21** |
|---|---|---|
| short_scoreboard | 2.04 (#1) | 1.99 (#1) |
| long_scoreboard | 0.86 | 1.40 (#2, rose) |
| wait | 1.28 | 1.22 |
| **barrier** | 1.20 | **0.91** (dropped) |
| memory throughput | 61.1% | **61.9%** (cuDNN 62.8) |

The win is mostly the **barrier drop (1.20 → 0.91)**: the shorter off-diagonal `fused_p` cuts
wg0's tail, so the warpgroups arrive at the rendezvous with less skew. As we strip consumer
work, `long_scoreboard` (global — dQ atomics / TMA) rose back to #2. And memory throughput is
now **essentially at cuDNN's level (61.9% vs 62.8%)** — we're about as memory-bound as it is.

## Where we stand
- Cumulative: V13 8.82 → V19 7.77 → V20 7.49 → **V21 7.39**, all bit-identical, 0-spill.
- Confirms again: with the tensor cores idle (~0% tensor stall), **only scalar-chain
  shortening pays** — and the causal specialization was exactly that.
- The #1 stall (short_scoreboard 1.99) is now a **serial FADD reduction/accumulation cluster
  + 64-bit address math** (`LEA.HI`/`IMAD.X`). The causal-specialization vein is spent (the
  fused_p mask was the prize; off-diagonal dS/GEMMs are full-valid work).

## V22 — pending precise localization
V22's target is whatever that FADD cluster is. Candidates: the producer D-rowsum's serial
`partial +=` + shuffle-tree (but that's producer-overlapped, may be a parking spot), the `dS`
subtract, or the dQ-flush address arithmetic. If it's on the consumer critical path and
reducible (multi-accumulator ILP to break the serial FADD chain, or hoisting the 64-bit
address math), it's a real lever; if it's a producer parking spot hidden by V20's pipeline,
it's flat. Localize before building — the number picks the target.
