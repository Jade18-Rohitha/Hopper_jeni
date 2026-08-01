# V23 Analysis — dQ transpose-staging elimination

**Result: 5.7017 ms / 578.5 TFLOP/s — 2.03× off cuDNN. Bit-identical. −3.78% over V22 (5.925 ms).**
Validates the SASS-analysis reframe: **cutting instruction count is the lever** (not scheduling).

## What changed
The dQ GEMM (`dS·K`) now reads `dS` **directly** from swizzled `sP` via `make_desc_sw128_K`
(Major::K, trans-a=0) instead of the `sp_to_sAt_v16_dq` ldmatrix→stmatrix round-trip. V16 had
kept dQ staged thinking it "needs a different orientation" — but the *single* swizzled `sP`
serves both: the swizzle makes the col(=k) axis contiguous, so dV/dK read Major::MN (contract
over row=q) and dQ reads Major::K (contract over col=k). The trans flag picks the contraction
axis; no transpose, no scramble (structurally identical to the proven S-GEMM A=Q read).

**SASS work-cut:** `LDSM 2→0`, `STSM 2→0`, `BAR.SYNC 16→15`, total 2680→2648 (−32/thread/tile).
Regs **157 (−5)**, 0 spill, smem unchanged (`sA_t` retained for epilogue). Bit-identical.

## Main-kernel profile — still memory-bound, still instruction-heavy

| metric | V22 | **V23** |
|---|---|---|
| long_scoreboard stall | 2.55 (#1) | **2.37 (#1)** |
| short_scoreboard | 1.02 | 0.87 (down — staging gone) |
| barrier | 1.40 | 1.42 |
| memory throughput | 65.7% | **66.4%** (cuDNN 62.8) |

The staging removal dropped short_scoreboard as expected, but **long_scoreboard (the mbarrier/TMA
feed wait) is still #1** — we're memory-bandwidth-bound (66.4%). And the SASS analysis stands:
we still run ~3.45× cuDNN's instructions. The remaining gap is a mix of **instruction count**
(staging, address math) and **memory traffic** (the O(S²) Q/dO reloads + dQ atomics).

## The D-kernel — profiled, and it BEATS cuDNN's
`compute_drowsum`: **63.1 µs** (~1.1% of the 5.70 ms total), **79.7% occupancy**, 68.6% memory
throughput (≈3.2 TB/s, ~66% of HBM peak). Well-optimized bandwidth-bound reduction, already
vectorized (bf16×2, 2 accumulators). Optimizing further would save <0.5% of the total.

**Head-to-head vs cuDNN (nsys per-kernel breakdown of cuDNN's full backward):**

| kernel | cuDNN | ours |
|---|---|---|
| main bprop (`flash_bprop_wgmma`) | 1.942 ms | ~5.5 ms |
| **D-rowsum** (`compute_dot_do_o_specialized`) | **100 µs** | **63 µs ✅ (37% faster)** |
| `convert_dq_to_16bits` | 81 µs | ~similar |
| **`fmha_reduce_head`** (GQA head reduction) | **68 µs (×2)** | **0 — fused in-register ✅** |

**Our D-kernel is 37% FASTER than cuDNN's** (63 vs 100 µs), and we **avoid an entire kernel
cuDNN pays for** — `fmha_reduce_head` (the GQA G-head reduction, 68 µs) — because we accumulate
the head group in-register in the main loop. So on *every* auxiliary kernel we win or match.

**This locates the entire 2.03× gap in the main kernel:** cuDNN's `flash_bprop` = 1.942 ms
doing 556 M instructions (tensor-bound, 59%); ours ≈ 5.5 ms doing 1.9 B instructions (3.45×,
L1/shared-throughput-bound). The D-kernel, convert, and head-reduction are *not* where we lose —
the main backward kernel's instruction count + shared round-trips are the whole story.

Also confirmed from the same breakdown: **cuDNN does NOT use bf16 atomics** — its
`convert_dq_to_16bits` kernel proves it accumulates dQ in **fp32** then narrows (identical to our
scheme). So fp32 dQ accumulation is the correct, cuDNN-matching choice; bf16 atomics is neither
its edge nor a precision compromise worth making.

## V24 — keep cutting instructions / traffic
The proven lever (V18/19 vectorize, V22 D-split, V23 dQ-TE) is work reduction, and it's live.
Remaining candidates:
1. **The `store_acc` (dP-write) / `stage_acc` (dQ-staging) per-element scalar writes** — still
   instruction overhead cuDNN avoids.
2. **The dQ atomic address math** (LEA/IMAD per element) — cuDNN uses TMA hardware addressing;
   we compute addresses explicitly. Biggest remaining instruction+traffic block, O(S²).
3. **Another cuDNN-style kernel split.**

Localize the V23 main-kernel instruction count to pick the biggest remaining source, then cut it.
Cumulative: V13 8.82 → V19 7.77 → V22 5.89 → **V23 5.70 (2.03×)**. Sub-2× is one work-cut away.
