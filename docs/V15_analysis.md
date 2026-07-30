# V15 Analysis — incremental `(g,qc)` address indices

**Result: 8.7142 ms / 378.5 TFLOP/s — 3.10× off cuDNN (2.81 ms), bit-identical, 0-spill.**
~0.66% over V14 (8.7719 ms). First optimization built *directly from a profiler measurement* rather than intuition.

## What changed (one thing)
The per-iteration query-tile indices were recomputed every loop iteration via
`g = it / perG` and `qc = qc0 + it % perG`. Because `perG` (= `nQTiles − qc0`) is a
**runtime** value, not a power of two, ptxas cannot strength-reduce these — it emits a
full **unsigned-divide sequence** (`I2F` → `MUFU.RCP` → `F2I` → correction `IMAD`s) on
*every* iteration, sitting on the address → TMA / `sLSE`-load critical path.

V15 maintains `(g, qc)` **incrementally** in both the producer and consumer loops:
```cpp
int g = 0, qc = qc0;
for (int it = 0; it < nIter; it++) {
    ... use (g, qc) directly ...
    if (++qc == nQTiles) { qc = qc0; ++g; }   // compare + increment, no divide
}
```
Provably identical to `it/perG`, `qc0+it%perG` by construction → **all of dQ/dK/dV stay
bit-identical** (no atomic reorder, unlike the persistent experiment).

## SASS proof
| | V14 | V15 |
|---|---|---|
| div-sequence (`I2F`/`F2I`/`MUFU.RCP`) | 6 | **0** |
| total static instructions | 2168 | **2112** (−56) |
| registers / spills | 152 / 0 | 154 / 0 |
| smem | 224,824 B | 224,824 B |

The +2 registers hold the two incremental counters — a cheap trade for removing a
per-iteration divide sequence from the hot loop.

## Profile (reports/gqa_v15_profile.ncu-rep) — the measurement loop, closed
Pipe utilization, vs V14 and the cuDNN target:

| pipe (% of peak) | V14 | **V15** | cuDNN |
|---|---|---|---|
| **ALU** (vector integer) | ~22% | **18.7%** | 16.6% |
| Uniform (scalar pipe) | 1.9% | 1.8% | 3.6% |
| FMA (floating-point) | 11.6% | 11.1% | 10.7% |

The div lever pulled vector-ALU utilization **22 → 18.7%**, closing ~60% of the 5.4-point
gap to cuDNN — exactly the drop predicted from the pre-change measurement. FMA unchanged
(same math), confirming the gain was purely bookkeeping.

**Overall:** memory throughput 57.6% (cuDNN 62.8%), SM 34.3%, occupancy 18.68%.
**Warp stalls:** short_scoreboard 2.16 > barrier 1.70 > wait 1.64 > long_scoreboard 1.02.

## Lesson
1. **Measurement-driven chain-shortening works.** Read the ALU-pipe gap → found the
   exposed integer work → removed it → measured the predicted drop. The wall-clock win
   (0.66%) is smaller than the ALU-pipe drop (3.3 pts) because ALU isn't saturated — the
   kernel is latency-bound — but the divs *were* partly exposed on the critical path.
2. **The ALU-mix lever is now ~85% spent** (18.7% vs cuDNN's 16.6%, ~2 points left).
3. **The bottleneck did not move** — it reverted to the same latency triad, with
   short_scoreboard (shared-load latency) #1. The real remaining gap is memory/shared
   throughput (57.6 → 62.8), i.e. the `sp_to_sAt` transpose/staging round-trips — the
   V17 (transpose-elimination) target.
