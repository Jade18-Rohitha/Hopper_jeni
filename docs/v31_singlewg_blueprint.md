# V31+ blueprint — single-warpgroup-per-tile + RS-wgmma (the cuDNN-structure rewrite)

> ## ⛔ VERDICT (after Step 1 + PTX-level review): DEAD END. RS-wgmma cannot free the backward's smem.
> **Step 1 (1-consumer single-wg) built + H200-verified: bit-identical, but 5.33 ms vs V30 4.30 ms
> (the expected 8-warp-vs-12 occupancy tax).** Then the RS operand-layout review killed the premise:
> - **dV=Pᵀ·dO and dK=dSᵀ·Q need a TRANSPOSED A-operand.** The transpose is *inter-warp* (row-owner
>   warp ≠ column-owner warp); `__shfl` is 32-lane-scoped so a 64×64 register transpose across a
>   128-thread warpgroup is IMPOSSIBLE, and `ldmatrix.trans` still reads smem. The `Major::MN`
>   swizzle-aliased descriptor we've used since V23 is ALREADY a zero-movement transpose — RS can't beat it.
> - **So P must stay in sP (for dV) and dS in sDS (for dK) no matter what.** RS can NEVER delete sP/sDS.
> - **RS only helps the non-transposed dQ** — and only saves a cheap *async descriptor read* of sDS while
>   costing pa[16]=16 live registers at the 232-reg cap → wash-to-marginal, likely a spill/regression.
> - **Premise falsified:** RS was supposed to free ~104 KB so 2 consumer wgs fit. It can't → the 2nd wg
>   never fits → occupancy stays at 8 warps → single-wg stays below V30. **V30's cooperative structure
>   is the right answer for the backward.** The forward won from RS-P@V only because P is consumed ONCE
>   there; in the backward P/dS are multi-consumer AND transposed → smem materialization is unavoidable.
> - cuDNN's residual ~1.5× is structural (persistent grid + TMA addressing + accepting spills), NOT
>   RS-reclaimable. **Landing at V30.** (Original blueprint kept below as the record of the exploration.)

---


Goal: cross from memory-bound (V30, 45% compute) to compute-bound like cuDNN (59%). The enabler
is **each consumer warpgroup owning a full kv-tile** — which makes P/dP/dS register-local (no
cross-wg smem), balances the work (symmetric tiles), and lets RS-wgmma eliminate sP/sDS.

## Structure (mirrors GQA_fwd_ref.cu gqa_v62)
- **384 threads = 1 producer (wg2) + 2 consumers (wg0,wg1)**, same as now (keep 12-warp occupancy).
- **Grid halves: `dim3(B, Hkv, (S/Bc)/2)`.** Each block owns a **kv-tile PAIR**; wg0 → tile
  `2*blockIdx.z`, wg1 → tile `2*blockIdx.z + 1`. (If S/Bc is odd, last block does 1 tile.)
- Each consumer wg runs the **full** per-q-tile chain for its OWN kv-tile, in its 128 threads:
  `S=Q·Kᵀ → softmax(P) → dP=dO·Vᵀ → dS=P⊙(dP−D) → dV+=Pᵀ·dO → dK+=dSᵀ·Q → dQ=dS·K → atomic`.
  Full `dv[64]`, `dk[64]` accumulators (was `[32]` column-split).
- **All operands register-local** — P from softmax acc, dP from dP acc, dS computed in regs.
  So `dS = P⊙(dP−D)` needs no smem (both P and dP are in *this* wg's registers). **sP, sdP, sDS
  all DELETED as cross-wg buffers.**

## Smem budget (the reason it fits — must stay < 232,448 B)
Per-wg tile inputs, shared where possible:
- K/V: each wg its own kv-tile → **2 × (Bc·D·2 ×2 for K+V) = 2 × 32,768 = 65,536**.
- Q/dO: the two kv-tiles attend overlapping q-ranges; feed a **shared** Q/dO pipeline. At **PD=2**:
  `2 (Q,dO) × 2 (PD) × Br·D·2 = 65,536`. (Q/dO are the same queries for both tiles where ranges
  overlap; the union is fed once.)
- sLSE, sD (PD-deep, per q-tile D-rowsum): small (~2 KB).
- dQ flush stage (fp32): per-wg, `Br·72·4 = 18,432` ×2 = **36,864** — OR TMA-store to skip
  (defer; keep smem stage first).
- **sP/sdP/sDS: 0** (RS-wgmma keeps P/dP/dS in registers) — this is the ~52 KB×2 = ~104 KB reclaim
  that makes the 2× layout fit.
- Total ≈ 65,536 + 65,536 + 36,864 + ~4 KB mbarriers ≈ **172 KB < 232 KB ✓** (headroom for PD=3
  on Q/dO if it fits: +32,768 → 205 KB, still ok).

## Registers / spill (accept bounded spilling, like cuDNN's 168+832K-local)
- 384 threads → setmaxnreg producer 40, consumers up to ~232 (already scaffolded in V30).
- Consumer holds: `dv[64]`+`dk[64]` (persistent, 128 regs) + S/dP acc (32 each, transient) + the
  RS A-fragments (P: 32, dS: 16) + descriptors. Likely >232 → **allow spilling of COLD state only**
  (loop bookkeeping, descriptors) — NEVER dv/dk or the live acc. Tune with `-maxrregcount`/the
  setmaxnreg value.

## wgmma operand plan (RS = register-source A)
- **dQ = dS·K** (Major::K, A=dS **non-transposed**): dS lives in the softmax/dS C-fragment layout;
  it *is* the RS A-fragment (like the forward's P→pa). **Do this first — no transpose, validates RS.**
- **dV = Pᵀ·dO**, **dK = dSᵀ·Q** (A **transposed**): need Pᵀ/dSᵀ as the A-fragment.
  1. **Validate** with `ldmatrix.trans` from a tiny scratch (correctness first, still 1 smem hop).
  2. **Optimize** to an in-register warp-shuffle transpose (64×64 bf16) to drop the smem hop.
- S-GEMM (A=Q) and dP-GEMM (A=dO) stay smem-source (Q/dO arrive via TMA in smem — fine, they're
  loads not round-trips).

## Build sequence (each step compiles + passes H200 check() before the next)
1. **Skeleton**: grid-halved, 2-tile-pair producer, per-wg full-tile consumer, **smem-source**
   wgmma (port the existing run_gemm), full `dv[64]/dk[64]`, PD=2. Correctness first (may be
   slower than V30 — that's fine). ← *the big one; ~250 lines*
2. **RS dQ**: dS register-resident → dQ RS-wgmma; delete sDS's dQ role.
3. **RS dV/dK**: ldmatrix.trans validate → shuffle-transpose; delete sP/sDS entirely; reclaim smem
   → bump PD/pipeline.
4. **Spill tune** + TMA-store the dV/dK epilogue (frees the fp32 stage).

Target: compute-bound, toward cuDNN's ~1.9 ms main-kernel (from our 4.32 ms total).
Risk: from-scratch kernel, several H200 correctness iterations, can't compile-verify semantics
locally (sm_120). Step 1 is the foundation; expect it to need debugging passes.
