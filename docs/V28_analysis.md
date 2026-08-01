# V28 Analysis — dS loop widened bf16×2 → bf16×4

**Result: 4.4927 ms (median) / 734 TFLOP/s — ~1.62× off cuDNN. Bit-identical. −3.26% over V27
(4.6440 ms).** Broke 4.5 ms. 157 regs, 0 spill, 204,880 B smem. First confirmation that the
instruction-count lever has reachable critical-path cuts (I'd called it "structural").

## What changed
The dS elementwise loop (`P ⊙ (dP − D) → sDS`, a hot 256-thread loop) processed **2 columns per
iteration** (bf16×2 read of `sP` + float2 `sdP`). V28 does **4 columns per iteration**:
- read `sP` as two bf16×2 (`sP[pidx]`, `sP[pidx+2]`), `sdP` as one **float4**;
- write `sDS` as two bf16×2.

Iterations halve (4/thread vs 8) — `r` advances by 16/iter so `r&7` (the swizzle term) stays
loop-invariant. Coverage verified: 256 threads × 4 iters × 4 cols = 4096. Bit-identical (same
values, wider access).

## Profile — dynamic instruction count fell

| metric | V27 | **V28** |
|---|---|---|
| **Executed instructions** | 1.985 B | **1.866 B (−6.0%)** |
| Elapsed cycles | 7.74 M | **7.32 M (−5.4%)** |
| IPC | 1.95 | 1.94 |
| wait stall (wgmma) | 0.93 | 0.84 |
| barrier stall | 1.10 | 1.16 |
| Memory SOL | 57.1% | 59.6% |

The **static** count *rose* (+56 — the bf16×4 body is bigger), but the **dynamic** count fell
6% because the per-iteration overhead (index math `r*64`/`r*72`, the loop branch, the `sD[r]`
load) and the `sdP` reads (one float4 vs two float2) are halved while the compute is unchanged.
Fewer instructions at the same IPC → fewer cycles → the −3.26% wall-clock.

## Why it matters
Reverses the "instruction count is structural (cuDNN's TMA-addressing, unreachable)" pessimism:
the address/index math in the **hot elementwise loops** *is* reducible by widening the
vectorization. The dS loop was one such loop; `fused_p` and the store/stage loops are the same
pattern.

Cumulative: V13 8.82 → V24 4.99 → V26 4.6853 → V27 4.6752 → **V28 4.4927 (~1.62×)**.
