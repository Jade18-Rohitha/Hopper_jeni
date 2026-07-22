# wgmma m64n64k16 (bf16→f32) — verified reference

Authoritative spec for `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16`, from the
deep-research pass (18 claims confirmed by 3-vote adversarial verification). Primary sources:
CUDA PTX ISA §9.7.16, CUTLASS docs, Colfax WGMMA tutorial. Use this to implement/debug V3+.

## 1. Accumulator fragment layout (f32) — CONFIRMED 3-0
Source: Colfax — *"thread 0 holds the values at (0,0),(0,1),(8,0),(8,1) and repeated every 8 columns to the right."*

- One wgmma is issued by the **whole 128-thread warpgroup** (4 warps); accumulator distributed across all 128 threads.
- **Warp `w` (0..3) owns rows `[16w, 16w+16)`.**
- Within a warp: `row = lane/4` and `row = lane/4 + 8`; `col = (lane%4)*2 + {0,1}`.
- The 64 N-columns are held as **8 ascending column-groups spaced by 8** — thread 0 owns cols `{0,1, 8,9, 16,17, …, 56,57}` at rows `{0,8}` — a "replicated Z-pattern", **no interleaving**.
- 32 regs/thread = 8 column-groups × (2 cols × 2 rows).

➡ **Our `store_acc_smem`/`store_acc_global` map (`r0=16w+lane/4`, `r1=r0+8`, `c=nt*8+(lane%4)*2`, `d[nt*4+{0..3}]`) matches this EXACTLY. The readout map is correct — not the bug.**

## 2. Shared-memory matrix descriptor (64-bit) — CONFIRMED 3-0
Fields (each byte-offset field encoded as `value>>4`): base addr (16B-aligned) [13:0], **LBO** [29:16], **SBO** [45:32], matrix-base-offset [51:49], swizzle-mode [63:62].
- **LBO** = byte distance between adjacent core matrices **along K**.
- **SBO** = byte distance between adjacent core matrices **along M/N**.
- **Core matrix** (CONFIRMED 3-0) = 8×8 elements = 8 rows × 16 bytes for bf16, tiled contiguously.
- Colfax no-swizzle example values: **LBO = 128 B, SBO = 256 B** — but these are for that example's tile; SBO scales with the operand's K-width (256 B = 2 core-matrices for a K=16 operand). Our K=64 buffer gives SBO=1024 B, which is self-consistent with our `tiled_off`.

## 3. Transpose immediates — CONFIRMED 3-0
- Transpose is legal **only** for f16/bf16 operands read from **shared memory** via descriptors.
- `trans = GMMA::Major::K (0)` vs `GMMA::Major::MN (1)`. For our K-contiguous staging, **both operands use trans=0**.
- Operand **A** may be in **registers or shared memory**; operand **B** must **always be in shared memory**.
- Instruction operand order: `d, a-operand, b-desc, scale_d, imm_scale_a, imm_scale_b, imm_trans_a, imm_trans_b`.

## 4. Async sequence — CONFIRMED 3-0
- `wgmma.fence.sync.aligned` = a **register-ordering barrier** (makes accumulator/register writes visible to a later wgmma) — **NOT** a completion barrier.
- Order: `fence` → issue wgmma(s) → `commit_group` → `wait_group N`.
- CUTLASS/Colfax pattern keeps the accumulator **live** across `fence → gemm → commit → wait` — i.e. re-accumulating (`scaleD=1`) into the same accumulator IS the sanctioned pattern **when fenced correctly**. (So the earlier "persistent accumulator" fold was not strictly required by the spec; correctness there hinges on the fence placement.)

## Bottom line for V3 debugging
The spec **confirms our readout map and our descriptor/trans semantics are right.** The remaining suspects are therefore narrow: (a) whether S is actually correct beyond col 7 (never verified — old debug only dumped cols 0–7), and (b) the transposed GEMMs `dV=Pᵀ·dO` / `dK=dSᵀ·Q` (fill_trans path). Next diagnostic must dump a FULL S row + dP on the H200 to localize.
