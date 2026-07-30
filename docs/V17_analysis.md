# V17 Analysis — hoisted swizzle-index write

**Result: 8.6947 ms / 379.4 TFLOP/s — new best, 3.09× off cuDNN. Bit-identical, 152 regs, 0-spill.**
+0.14% over V16 (8.7066 ms). The 5th win in the measurement-driven string
(8.82 → 8.77 → 8.714 → 8.707 → **8.695**).

## What changed
Hoisted the swizzle-index invariants on the `sP` write path (both provably bit-identical):
- **`fused_p_from_acc_v17`**: `c = nt*8+cc` with `cc<8` ⇒ `c>>3 == nt` (compile-time),
  `c&7 == cc` (constant) ⇒ index is `base + ((nt^m)<<3)`, `base`/`m` hoisted per thread.
- **`dS`-write loop**: `CONS(256) % Bc(64) == 0` ⇒ `c = i%Bc` is invariant across the
  thread's 16 iterations ⇒ `c`, `c>>3`, `c&7` hoisted out.

SASS: integer-ALU 595 → **574** (−21 static), total 2200 → 2176.

## Profile — partial recovery, and the latency-bound seesaw

| metric | V15 | V16 | **V17** |
|---|---|---|---|
| ALU pipe | 18.7% | 23.4% | **21.3%** |
| uniform | 1.8% | 2.6% | 2.4% |
| FMA | 11.1% | 12.7% | 11.2% |
| memory throughput | 57.6% | 55.2% | 55.3% |
| SM throughput | 34.3% | 37.3% | 36.1% |
| **short_scoreboard** stall | 2.16 | 1.89 | **2.00** |
| wait stall | 1.64 | 1.52 | 1.56 |
| barrier stall | 1.70 | 1.50 | 1.54 |

Two readings:
1. **The ALU trim was partly real:** 23.4 → 21.3% recovered ~2 of the ~5 points V16 added.
   The remaining gap to V15's 18.7% is **relative** — V16 removed the ldmatrix/stmatrix MIO
   ops, so ALU is a bigger fraction of a smaller total; that part can't be "recovered."
2. **The stalls crept back up** (short_scoreboard 1.89 → 2.00, wait/barrier slightly). This
   is the latency-bound seesaw: removing integer work shortens the kernel a hair (+0.14%),
   but it re-exposes the shared-load / dependency latency that actually gates the clock.

## Where we stand — back against the latency wall
The measurement-driven ALU levers (V14-V17) squeezed ~1.4% total and have now bottomed out —
ALU is near its floor and the bottleneck is firmly the **latency triad**:
**short_scoreboard 2.00 (shared-load) > wait 1.56 (RAW fp32 chain) > barrier 1.54.**

## V18 candidate — the `wait` stall (never directly cut)
short_scoreboard has been attacked (LDS-prefetch worse, transpose-elim helped then crept
back). The one latency source we've **never touched** is `wait` = the RAW fp32 dependency
chain in `dS = P⊙(dP−D)`: per element `bf16→float → (×) → float→bf16`, a serial cvt→mul→cvt.
**Vectorize it (bf16×2 / float2)** — process 2 elements per step, halving both the
instruction count and the number of independent dependency chains in flight. Bit-identical
(same per-element fp ops, just packed). It's the first direct attack on the #2 stall.
