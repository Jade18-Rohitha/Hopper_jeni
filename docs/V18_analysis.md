# V18 Analysis — vectorized `dS` (bf16×2 / float2) — BREAKTHROUGH

**Result: 8.2540 ms / 399.6 TFLOP/s — 2.94× off cuDNN (first time below 3×). Bit-identical, 152 regs, 0-spill.**
**−5.06% over V17 (8.6939 ms) — the biggest single win since the early versions.**

## What changed
Vectorized the `dS = P⊙(dP−D)` elementwise loop: each thread now processes **2 adjacent
columns per step** (`cbase = (2·tid)%Bc` is even ⇒ `sP[pidx]`,`sP[pidx+1]` are adjacent in
the swizzle 8-group). The bf16 load/store became a single **bf16×2** (`LDS.32`/`STS.32`), the
`bf16→float → ×(dP−D) → float→bf16` chain is done as **float2**, and the loop runs **8
iterations instead of 16**. SASS confirms the packing: `LDS.U16` 23→16, `STS.U16` 39→32
(scalar bf16 → 32-bit packed, not the 2×16-bit fallback). Bit-identical (each element
computed once with identical ops).

## Profile — where the 5% came from

| metric | V17 | **V18** |
|---|---|---|
| ALU pipe | 21.3% | **19.5%** |
| FMA pipe | 11.2% | 11.1% |
| memory throughput | 55.3% | 57.7% |
| SM throughput | 36.1% | 34.7% |
| short_scoreboard stall | 2.00 | 2.12 |
| wait stall | 1.56 | 1.64 |
| barrier stall | 1.54 | 1.70 |

**The stall *ratios* rose, yet the clock dropped 5%** — that's the key. The stalls are
*per-issued-instruction*; the win came from **issuing far fewer instructions** on a
critical-path loop (8 packed iterations vs 16 scalar, halved LDS/STS). Total time ≈
issued × cycles-per-issue, and the instruction-count drop dominated the slight per-issue
rise. Memory throughput went *up* (57.7%) purely because the same bytes now flow in 5% less
time. So the mechanism is: **fewer, wider ops on the `dS` link + a halved dependency chain,
and because `dS` is *on* the critical path (feeds both dK and dQ), it converted directly.**

## The lesson — elementwise vectorization is a powerful, reachable lever
Every recent lever was sub-percent (ALU trims on a latency-bound kernel = seesaw). This one
was 5% because it cut **instruction count + chain depth** on a critical-path *elementwise*
step, not just shaved ALU. That reopens the endgame: there are more per-element `cvt`/store
steps to hit the same way.

## V19 — extend the lever
Vectorize the next critical-path elementwise write: **`fused_p` (the P write, wg0)**. It
stores `sP[b0]`,`sP[b0+1]`,`sP[b1]`,`sP[b1+1]` as 4 scalar bf16 — but `b0`/`b0+1` (cols c,c+1
of row r0) and `b1`/`b1+1` (row r1) are adjacent and `b0`,`b1` are even ⇒ **two bf16×2 stores
per `nt`** with the two `__expf` values packed as float2 and the two causal masks applied.
P feeds *everything* downstream, so like `dS` it's squarely on the critical path. Stacking
this (and later the `store_acc` dP-write / `stage_acc` staging) should **break 8 ms.**
