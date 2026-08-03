# V25 Analysis — bf16 dV/dK stage bank-conflict cut (stride 72)

**Result: 4.9353 ms (median) / 666.97 TFLOP/s — ~1.77× off cuDNN. Bit-identical. −0.98% over
V24 (4.9843 ms).** 157 regs, 0 spill, 196,688 B smem. Beat the honest ~0.3–0.5% estimate: the
8-way→1-way (conflict-free) store fix converted to ~1% wall-clock even though it's the O(S)
dV/dK epilogue, not V24's O(S²) hot path. Correctness: dQ/dK/dV all `Values match!`, max_abs
identical to V24 (1.953e-3 / 3.906e-3 / 3.125e-2) — bit-identical (padding only).

## Why this, and why it's the last shared lever
V24's source-page attribution (`ncu --page source`) settled the shared-traffic picture:
- The **33.5 M "load conflicts" are not reducible** — they're `HGMMA gdesc[…].tnspA` wgmma
  operand fetches (360 M wavefronts, **0 excessive**). That's how the tensor core reads smem.
- The **`sdP` float2 idea was a no-op** — nvcc's load-vectorizer already merged the two scalar
  `sdP` loads into `LDS.64` back at V24 (proven: byte-identical SASS, 2635 instr, empty diff).
- The **only reducible shared conflict left** was 32 lines of **8-way `STS.32`** (3.67 M
  excessive) — the **bf16 dV/dK stage stores**. V24 fixed the fp32 dP/dQ stores; it never
  touched these.

## What changed
`stage_acc_bf16<64>` writes the dV/dK fragment to `sA_t` at stride 64; packed as bf16×2
(`STS.32`), word-index `r·32 + c/2` → `r·32 mod 32 = 0` → all 8 rows/warp hit the same 4
banks → **8-way**. Re-pad the stage to **stride 72** (decoupled from the 64 real data columns,
exactly as V24 decoupled `atomic_flush_stage_s`):
- new `stage_acc_bf16_s<NDATA=64, ROWSTRIDE=72>` and `store_stage_vec_s<ROWS,NCOL=64,ROWSTRIDE=72>`
- `sA_t` sized `2·64·72 = 9216 bf16`; `stage = sA_t + wg·(64·72)`.

**Bank math:** packed word-index `r·36 + c/2`; within a warp the store maps to
`(i·4 + j) mod 32`, `i = lane>>2 ∈ 0..7`, `j = lane&3 ∈ 0..3` → `i·4` gives 8 values spaced by
4, `j` fills the gaps → **all 32 banks distinct → 1-way (conflict-free)**. Better than V24's
2-way fp32 floor: bf16 packs 2 cols/word (column stride 1), so the row term fully spreads —
which the fp32 stride-2 column pattern couldn't reach.

`sA_t` in V24 is *only* the epilogue stage (V23 made dQ read `sP` directly), so there is **no
dQ-stage aliasing** — restriding it is safe. Logical (row,col)→value unchanged → dV/dK
bit-identical to the ULP.

## Profile — the store conflicts hit the floor (H200-confirmed)
`ncu` shared-store conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`):
**4,504,677 → 823,934 (−82%)** — the 3.67 M 8-way excessive collapsed to ~0 at 1-way, exactly
as the bank math predicted. Shared-*load* conflicts unchanged at 33.5 M (the inherent HGMMA
operand fetches, 0 excessive — not ours to fix). The remaining 0.82 M store conflicts are the
residual floor. **This is the shared-conflict lever fully mined out** — V24 took the O(S²) fp32
stores, V25 the O(S) bf16 stores, loads are wgmma-inherent, and there is no reducible shared
conflict left. The wall now moves to feed/sync (see V26).

Cumulative: V13 8.82 → V22 5.89 → V23 5.70 → V24 4.99 → **V25 4.9353 (~1.77×)**.
