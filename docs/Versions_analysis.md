# GQA Backward — Version Ladder (V1 → V26)

Reference for `Hopper/src/attention/GQA_bwd.cu`. Each version is described as a **delta**
from the previous one — only what changed, plus the reason. All versions compute the same
math (dQ, dK, dV for grouped-query attention backward, bf16 in/out, fp32 accumulate) and are
validated against the PyTorch reference (`precision/baseline_gqa.py`) at 2e-2.

The five GEMMs per tile (fixed across all versions):
- `S  = Q·Kᵀ · scale`   (contract over D)
- `dP = dO·Vᵀ`           (contract over D)
- `dV += Pᵀ·dO`          (contract over rows i)
- `dK += dSᵀ·Q`          (contract over rows i)
- `dQ += dS·K`           (contract over keys j)

with `P = exp(S − LSE)` (causal-masked), `D[r] = rowsum(dO·O)`, `dS = P ⊙ (dP − D)`.

---

## Quick-reference table

| V | Portability | Threads | Tensor core | Global→smem | Overlap | Kernels | dQ reduction | Headline change |
|---|---|---|---|---|---|---|---|---|
| V1 | SM_70+ (bf16: SM_80+) | 32 (1 warp) | wmma 16×16×16 | plain `ld.global` | none | 2 | per-block accum | baseline, portable |
| V2 | SM_70+ | 128 (4 warps) | wmma 16×16×16 | plain | none | 2 | per-block accum | 64×64 tiles, 4-warp split |
| V3 | Hopper (sm_90a) | 128 (1 WG) | **wgmma** | plain | none | 2 | persistent HW accum | wmma→wgmma port |
| V4 | Hopper | 128 (1 WG) | wgmma | **TMA + 128B swizzle** | **double-buffer** | 2 | persistent HW accum | TMA + swizzle + prefetch |
| V5 | Hopper | 128 (1 WG) | wgmma | TMA + swizzle | double-buffer | **1 fused** | **fp32 atomic scratch** | single KV-centric kernel |
| V6 | Hopper | 128 (1 WG) | wgmma | TMA + swizzle | double-buffer | 1 | fp32 atomic | **bank-conflict padding** |
| V7 | Hopper | 128 (1 WG) | wgmma | TMA + swizzle | double-buffer | 1 | fp32 atomic | **transpose-buffer elimination** (`trans-b=1`) |
| V8 | Hopper | 128 (1 WG) | wgmma | TMA + swizzle | double-buffer | 1 | fp32 atomic | warp-shuffle D-rowsum (A-swizzle blocked) |
| V9 | Hopper | **256 (2 WG)** | wgmma | TMA + swizzle | double-buffer | 1 | fp32 atomic | **column bisection** across 2 WGs |
| V10 | Hopper | 256 (2 WG) | wgmma | TMA + swizzle | double-buffer | 1 | fp32 atomic | **coalesced writeback** (smem staging) |
| V11 | Hopper | **384 (3 WG)** | wgmma | TMA + swizzle | **producer/consumer** | 1 | fp32 atomic | **FA3 warp specialization** |
| V12 | Hopper | 384 (3 WG) | wgmma | TMA + swizzle | producer/consumer | 1 | fp32 atomic | **`ldmatrix→stmatrix`** reshuffle |
| V13 | Hopper | 384 (3 WG) | wgmma | TMA + swizzle | producer/consumer | 1 | fp32 atomic | **register-fused softmax + producer-side D-rowsum** |

Tile shape: V1 = Br16/Bc32, V2 = 64×64 (D flexible, %16). **V3–V13 are hard-fixed Br=Bc=64, D=128.**

---

## V1 — portable wmma baseline
**Lines ~24–407.** The reference implementation; correctness-first, runs anywhere.

- **Portability:** no tcgen05/TMA/wgmma → SM_70+ (bf16 wmma realistically needs SM_80+/Ampere).
- **Threads:** 32 (one warp). All barriers are `__syncwarp()`.
- **Tensor core:** `wmma` 16×16×16 fragments — bf16 `matrix_a`/`matrix_b`, fp32 `accumulator`. Sequence per GEMM: declare fragment → `load_matrix_sync` → `mma_sync` → `store_matrix_sync`.
- **Constraint:** `static_assert` Br%16, Bc%16, D%16 all ==0. Any multiple of 16 (64/128 just what's tested).
- **Two kernels:** `gqa_backward_dQ` (grid `B,Hq,S/Br`) + `gqa_backward_dKdV` (grid `B,Hkv,S/Bc`). Split because dQ reduces over keys, dK/dV reduce over queries.
- **Causal:** tile-level `break` when `kc*Bc >= q_row0 + Br` (dQ) / loop start `qc = k_row0/Br` (dKdV); element mask on the diagonal tile.
- **Detail:** `sdP` (the dP intermediate buffer) is **reused as the writeback staging scratch** — `dQ_acc` fragments are `store_matrix_sync`'d into `sdP`, then copied to global `d_dQ`.

## V2 — wmma, 4 warps, 64×64
**Lines ~410–555.** Same math/structure as V1; introduces the warpgroup thread shape.

- **Delta vs V1:**
  - Tile fixed to **Br=Bc=64** (`static_assert Br==64 && Bc==64 && D%16==0`).
  - **128 threads (4 warps)**; barriers become `__syncthreads()`.
  - **Work split:** each warp owns a **16-row band** of the 64-row Q tile (`row0 = warpId*16`), visible in every smem index (`sQ + row0*D`, `sS + row0*Bc`, …).
  - Writeback: `sdP` partitioned per-warp — `warp_tmp = sdP + warpId*16*16`, each warp stores its `dQ_acc` and streams to global independently, no cross-warp sync.
- **Still wmma, not wgmma.** V2 is the bridge: it establishes the 128-thread/4-warp shape that V3 keeps while swapping the mma.

## V3 — Hopper wgmma baseline
**Lines ~746–1286.** Direct 1:1 port of V2 to warpgroup MMA. Correctness-first (no perf work).

- **Delta vs V2:**
  - **`sm_90a` only** (wgmma is Hopper-only; compile-only elsewhere).
  - **Tile hard-fixed Br=Bc=64, D=128** (`==`, not `%16`) → all descriptor constants fold at compile time.
  - Every `wmma::mma_sync` → one warpgroup MMA `wgmma.mma_async.m64n64k16` (or `m64n128k16`). One block = one warpgroup = 128 threads.
  - **Two extra scratch buffers** `sA_t[64*128]`, `sB_t[64*128]` — no-swizzle wgmma tiled staging (largest operand).
  - **Operand staging:** `fill_copy` (K-major) / `fill_trans` (transpose) / `fill_trans_A` (Major::MN, `mn_off`) reshape row-major smem into the core-matrix-tiled layout wgmma requires.
  - **GEMM helpers:** `run_gemm_n64` (m64n64, `acc[32]`, intermediates S/dP) / `run_gemm_n128` (m64n128, `acc[64]`, outputs) / `run_gemm_n128_tA` (transposed-A for dV/dK, `trans-a=1`). `make_desc` builds the 64-bit smem descriptor per operand.
  - **Result accumulators are PERSISTENT & in-hardware:** `dq[64]`/`dk[64]`/`dv[64]` zeroed once, accumulated across the kc loop via wgmma `scaleD=1`, read out once. `acc[32]` intermediates are local scratch, zeroed/consumed per tile.
  - **Store helpers:** `store_acc_smem<Bc>(…, scl)` (intermediate → smem, takes scale) / `store_acc_global<D>` (output → global, fp32→bf16).
- **Three wgmma correctness traps (documented):** (1) operand core-matrix tiling; (2) trans semantics — native bf16 is `trans=0` (K-contiguous) for both; (3) two async fences — `fence_operandN` brackets the accumulator registers, `fence.proxy.async.shared` bridges generic-proxy fill writes → async-proxy wgmma reads. Persistent accumulators need an **extra `fence_operandN` outside the loop** before the final store.

## V4 — TMA + 128B swizzle + double buffering
**Lines ~1288–1745.** Same two-kernel structure and math as V3; rebuilds the memory pipeline.

- **Delta vs V3:**
  - **TMA loads** (`cp.async.bulk.tensor`) replace manual `ld.global`, driven by mbarriers (`mbar_init`/`mbar_expect_tx`/`tma_load_2d`).
  - **128B-swizzled smem** for the operand read directly on the wgmma SW128 path — no `fill_*` needed for that operand. `run_gemm_n64_sw<bool A_swz>` picks which slot (A or B) is swizzled: `if (A_swz) wgmma(dsw,dno); else wgmma(dno,dsw)`.
  - **Which operands are swizzled (role-dependent, NOT all of Q/K/V):**
    - dQ kernel: **Q swizzled** (S-A), **V swizzled** (dP-B). **K plain** (needed transposed for dQ=dS·K).
    - dKdV kernel: **K swizzled** (S-B), **V swizzled** (dP-B). Q/dO/O plain.
    - Rule: the operand read K-major on the direct path is swizzled; anything transposed or elementwise stays plain.
  - **Swizzle only helps the S and dP GEMMs.** The three output GEMMs (dQ/dV/dK) still use V3's `run_gemm_n128`/`_tA` with `fill_*` on both operands.
  - **Double buffering:** `sK_pl[2]` / `sV_sw[2]` + two mbarriers. Each iteration prefetches tile `kc+1` into slot `s^1` while wgmma computes on slot `s`. This overlap (not TMA alone) is the main win.
  - **"Two atoms":** D=128 = two 128-B swizzle atoms (each 64 bf16 wide) → each swizzled operand loads as 2 `tma_load_2d` calls (cols 0–63, 64–127).
  - Persistent (per-tile) operands loaded once; only the loop-variant tile ping-pongs. O loaded plain (elementwise use in D-rowsum only).

## V5 — single fused KV-centric kernel
**Lines ~1746–2068.** Collapses V4's two kernels into one.

- **Delta vs V4:**
  - **One kernel**, grid `(B, Hkv, S/Bc)`, one block per KV-tile.
  - **dK, dV reduce over queries** → block owns them → persistent `dk[64]`/`dv[64]` accumulators, one writeback, **no atomics**.
  - **dQ reduces over keys** → not owned by one block → computed into a **transient** accumulator, flushed via **fp32 `atomicAdd`** into scratch `d_dq_accum[B*Hq*S*D]` (`atomic_acc_global`). A small `convert_dq_accum_to_bf16_v5` kernel rounds fp32→bf16 at the end. Launcher owns the scratch (malloc/memset/convert/free). fp32 (not bf16 atomic) because bf16 atomicAdd is unsupported on sm_90 and O(S/Bc) partials per row would lose precision.
  - **K loaded both ways:** `sK_sw` (swizzled, for S=Q·Kᵀ) + `sK_pl` (plain, for dQ=dS·K via fill_trans). +16 KB persistent smem vs V4 dKdV.
  - **Loop-variant now Q/dO/O** (double-buffered over the G×qc iteration space); K/V persistent.
  - **dQ split into two m64n64 halves (`acc[32]` each), TWO atomic calls:** register pressure — `dv[64]`+`dk[64]` persist (128 regs); a full `dq[64]` would peak at 192 → spills. Two halves reuse one `acc[32]` (peak 32, 0 spills). Half0 → cols[0,64) at `sB_t+0`; half1 → cols[64,128) at `sB_t+4096`, `coloff=64`.

## V6 — bank-conflict-free (padded smem)
**Lines ~2069–2424.** Identical math to V5; **bit-identical** output. Only storage strides change.

- **Why:** V5 Nsight profile = 17.9-way shared-load conflict (94.4% of wavefronts), tensor cores idle at 13.5% Compute SOL. Root cause: strides that are multiples of 128 B (32 banks × 4 B) alias onto the same banks.
- **Delta vs V5 — three padded layouts:**
  1. **`sA_t`/`sB_t`** wgmma tiled operands: MN core-matrix stride (= wgmma **SBO**) is `(K/8)*128 B` (128-B multiple) → all MN core-matrices alias. Pad MN-block-row stride by `PAD_V6 = 8` bf16 (16 B) → SBO mod 128 = 16 ≠ 0. Total `SMEM_TILED_V6 = 8320` (= 65×128). `sbo_pad_v6`, `tiled_off_v6`, `mn_off_v6` thread the padded SBO into `make_desc`.
  2. **`sS`/`sdP`** (float, stride Bc=64): pad to **65** (`SS_STRIDE_V6`) — odd word stride, coprime with 32 → 8 r0 hit distinct banks.
  3. **`sP`** (bf16, stride Bc=64): pad to **66** (`SP_STRIDE_V6`) — kept even (bf16), 66/2=33 coprime with 32.
  - Padding-aware helper clones: `fill_copy_v6`/`fill_trans_v6`/`fill_trans_A_v6` (take explicit src stride), `store_acc_smem_v6`.
- **NOT padded:** TMA-plain buffers (`sQ_pl`/`sdO_pl`/`sO_pl`/`sK_pl`) — TMA writes contiguous; swizzled `sK_sw`/`sV_sw` — swizzle already de-aliases. Only thread-written scratch gets padded.

## V7 — transpose-buffer elimination (`trans-b=1`)
**Lines ~2425–2845.** Removes the last **unpaddable** conflict.

- **Why:** V6 still stuck at 15.1-way shared-load conflict — the three `fill_trans` reads (Kᵀ/dOᵀ/Qᵀ) of the plain operands are strided `src[k*stride+mn]` = one bank line per thread. Padding can't fix a *read pattern*.
- **Delta vs V6:**
  - **No `fill_trans` of loaded operands, and `sB_t` deleted entirely.** Load K/Q/dO via 128B-swizzled TMA into a **single buffer each**, and feed the transposed GEMM from those same bytes via the wgmma **Major::MN transpose immediate `trans-b=1`** (Hopper analogue of Blackwell's idesc `b_major`). The same physical `[rows][D]` swizzled buffer serves both roles: Major::K (`trans=0`, contract over D) for S/dP, Major::MN (`trans-b=1`, contract over rows) for dQ/dV/dK.
  - Deleted vs V6: `sK_pl`, `sQ_pl→sQ_sw` (Q swizzled, no dup), **`sB_t` entirely**. `sA_t` stays (in-kernel dS/Pᵀ/dSᵀ intermediates). S and dP become both-operand-swizzled (Q_sw·K_sw, dO_sw·V_sw).
  - **dO double-load (the "V10 twist"):** `D[r]=rowsum(dO·O)` is a plain elementwise read that can't index swizzled bytes → dO loaded twice: `sdO_sw` (feeds dP-A, dV-B MMAs) + `sdO_pl` (D-rowsum only). O stays plain. +1 tile TMA/iter.
  - **2× m64n64 atom split for ALL transposed GEMMs:** D=128 = two atoms; a single m64n128 Major::MN read would span both (unvalidated multi-atom). Each transposed-B GEMM split into two N=64 halves (one atom = B300-confirmed single-atom recipe). V6 already split dQ; V7 splits dV/dK too. Per-half B base: half0 `buf+0`, half1 `buf+4096`; per-k-step `+1024` elems.

## V8 — warp-shuffle D-rowsum (A-swizzle blocked)
**Lines ~2846–3210.** Ships a small conflict fix; the intended big fix is disabled.

- **Delta vs V7 (what SHIPS, bit-identical to V7):**
  - **`D[r]=rowsum(dO·O)` → Blackwell dKdV_v10 warp-per-row coalesced + `__shfl_down` reduction**, replacing V7's fixed-j/stride-D plain read (a 32-way same-bank aliasing). The one real conflict reduction that shipped.
  - GEMMs unchanged (proven V7 no-swizzle A-operand path).
- **What's BLOCKED (the intended V8):** the residual 11.3-way load conflict is the wgmma read of the **in-kernel computed `sA_t`** (Pᵀ/dSᵀ/dS), staged no-swizzle. Plan: store it swizzled, read Major::K. First attempt **failed precision on all three gradients**.
  - Root cause isolated: `sw128_idx` and `make_desc_sw128_K` k·16 advance both **confirmed correct** on HW. The unvalidated ingredient is the **operand combination**: swizzled-Major::K A + `trans-b=1` Major::MN B (every proven swizzled-K A pairs with a K-major B; every proven `trans-b=1` B pairs with a no-swizzle A).
  - Carried but dormant: `sw128_idx`/`fill_swz_*`/`run_gemm_*_sw` + a `-DV8_DEBUG` self-test (recomputes one dQ half both ways, prints max|Δ|). **Needs H200** (no wgmma on local sm_120).

## V9 — column bisection across 2 warpgroups
**Lines ~3211–3558.** Occupancy via a second warpgroup; genuine parallel tensor work.

- **Delta vs V8:**
  - **256 threads = 2 warpgroups** (`__launch_bounds__(256,1)`, `wg = tid>>7`, `wtid = tid&127`). smem byte-identical → still 1 block/SM; occupancy rises from the extra resident warpgroup.
  - **Persistent accumulators halved to `acc[32]`:** each WG **owns one column-half** — `dv[32]`/`dk[32]`, WG0 owns dV/dK cols[0,64), WG1 cols[64,128). Disjoint global regions → still **no atomics** for dV/dK.
  - **S ∥ dP concurrent:** WG0 computes S=Q·Kᵀ→sS, WG1 computes dP=dO·Vᵀ→sdP, in parallel (V8 ran them serially).
  - **All output GEMMs use the `_half` variant, one call per WG:** `run_gemm_dVdK_half(dv,…,wg*4096)`, `…(dk,…)`, `run_gemm_dQ_half(…,wg*4096)`. The two m64n64 halves V8 did sequentially in one WG are now **one-per-warpgroup, concurrent** — WG0 does cols[0,64), WG1 cols[64,128). `wg*4096` = B-operand atom offset, `wg*64` = output column offset.
  - `store_acc_global_col` (adds a global column offset), `fill_*_v9` (blockDim.x-strided = 256).
  - Barriers stay block-scope `__syncthreads()` (all 256 reach each — no partial-CTA count hazard).

## V10 — coalesced global writeback
**Lines ~3559–3919.** Structurally identical to V9; only the writeback changes. Bit-identical.

- **Why:** the mma m16n8k16 D-fragment map hands consecutive lanes **strided** global addresses → a warp touches only 8/32 B of a bf16 sector (dK/dV) or 16/32 B of an fp32 sector (dQ atomics) → 8/32 per-transaction efficiency.
- **Delta vs V9 (the V2 stage-through-smem trick, vectorized):**
  - Reshuffle each accumulator through a **dead smem buffer into row-major order**, then re-read so consecutive threads emit consecutive global addresses → full-sector (32/32) coalescing. `stage_acc_bf16`/`stage_acc_f32` + `store_stage_vec`.
  - **dK/dV:** staged in dead `sA_t` (last used by loop's dQ GEMM), WG0→`sA_t[0..4096)`, WG1→`[4096..8192)` → **128-bit `uint4` (8 bf16) vector stores** → 32/32.
  - **dQ:** staged in dead `sS` (WG0) / `sdP` (WG1) (dead since P/dS phases) → **contiguous fp32 atomicAdds** → L2 coalesces (16/32→32/32).
  - **Zero net smem** (V9 at the 222,744 B / 1-block-per-SM cap → reuse dead buffers). `__syncthreads()` between register→smem scatter and smem→global read (RAW), and dV read vs dK overwrite (WAR).

## V11 — 3-warpgroup producer/consumer (FA3 warp specialization)
**Lines ~3920–4172.** Dedicates a warpgroup to loads.

- **Delta vs V10:**
  - **384 threads = 3 warpgroups** (`__launch_bounds__(384,1)`). Theoretical occupancy 18.75% (12/64 warps) — the shape cuDNN uses. smem = V10 + a few mbarriers → still 1 block/SM; occupancy rises purely from the extra resident warpgroup.
  - **Roles (`wg = tid>>7`):** wg0,1 = **consumers** (all compute, exactly V10's two WGs). wg2 = **producer** (all TMA loads + drives double buffer). Only the producer **leader** (tid==256) emits TMA/mbarrier; the other 96 producer threads stay resident and spin on `empty` (FA3 producer-warp residency) to add eligible warps that hide consumer wgmma stalls.
  - **Double buffering is KEPT** (2 stages) — but relocated from consumer-inline-prefetch to a **producer/consumer full/empty mbarrier handshake:**
    - `full[s]` (producer→consumer): leader `expect_tx`+6×TMA; TMA `complete_tx` flips phase; consumer `mbar_wait(full[s])`.
    - `empty[s]` (consumer→producer): one consumer (tid==0) arrives **right after the dK GEMM** (last read of the staged Q/dO buffers); producer waits before overwriting.
    - `mbar_kv`: one-shot persistent K/V.
  - **Async between roles**, up to 2 tiles apart (run-ahead bound = buffer depth 2, `empty[it−2]`). Priming: it=0/it=1 fill fresh buffers without an empty-wait → strictly forward chain, no cycle. **Early empty release** is safe: dQ tail reads only `sK_sw`/`sA_t`/`sS`/`sdP`, never the staged buffers, so the producer can refill during the tail.
  - **Compute barriers become `consumer_sync()` = `bar.sync 1, 256`** (named barrier, 256 consumers only) — NOT `__syncthreads()` (id 0, all 384), which would deadlock waiting on the non-participating producer. The single all-384 `__syncthreads()` is only the init publish.
  - `setmaxnreg` NOT used (127 regs → smem-pinned, not reg-pinned; 384×127 < 65,536).

## V12 — `ldmatrix → stmatrix` reshuffle
**Lines ~4173–4533.** Micro-opt on V11; everything structural unchanged. Bit-identical.

- **Why:** V11 memory-latency bound — the three `fill_*_v11` scatters (building the wgmma A-operand from `sP`) do **2-byte bf16 stores at stride 8** into `sA_t` → hit banks {0,4,…,28} → 4-way store conflict; 2-byte granularity inflates wavefronts (~3.8-way store conflict ~33.8%).
- **Delta vs V11:**
  - Replace each scatter with a matched **`ldmatrix.x4 → stmatrix.x4`** pair (`ldst_matrix_x4`, `sp_to_sAt_v12`) — **16-byte** transactions (one 8×8 b16 core-matrix row) on the HW matrix load/store engine.
  - **NOT RS-wgmma:** the 4 ldmatrix regs are transient (consumed immediately by stmatrix); wgmma still reads `sA_t` from shared. Register budget ~unchanged (V11 at 168/170; RS would blow the ceiling). `run_gemm_*` untouched.
  - **Bit-identical by construction:** ldmatrix/stmatrix have inverse per-lane semantics; store the same opaque `{r0..r3}` fragment → only src/dst row addresses chosen (from the same `mn_off_v6`/`tiled_off_v6` formulae). The "transpose" is a block-placement swap (core base differs), not element-level → no `.trans` needed.
  - **`sP` repad 66 → `SP_STRIDE_V12 = 72`:** ldmatrix needs each row 16-B aligned; 66 elems=132 B is only 4-B aligned on odd rows (illegal). 72 elems=144 B (9×16 B) → every row 16-B aligned, word stride 36 → 8 row addrs on 8 distinct banks. Cost +768 B smem (→223,528 B), still 1 block/SM, occupancy unchanged.
  - Fences unchanged (ldmatrix/stmatrix are generic-proxy ops).

## V13 — register-fused softmax + producer-side D-rowsum
**Lines ~4534–end.** Two independent latency cuts on V11/V12. Bit-identical, ≤170 regs.

- **Part B — register-fused softmax:** after `run_gemm_n64_sw2`, wg0's 128 threads already hold the full 64×64 S in `acc[32]`. Compute **P = exp(S·scale − LSE) + causal mask directly on `acc`** → `sP` (`fused_p_from_acc_v13`), skipping the S→`sS` store, the `sS`→P reload, and **one `consumer_sync`** per tile. `acc*scale` (fp32) == old `sS` value exactly → bit-identical. `sS` NOT deleted (reused as wg0's dQ fp32 staging buffer; no free 16 KB region).
- **Part A — producer-side D-rowsum:** `D[r]=rowsum(dO·O)` moves **off consumers onto wg2's 128 (otherwise-idle) producer threads**, overlapping consumer GEMMs (`producer_drowsum_v13`). Computed for tile `it−1` **one tile lagged** (keeps 2 tiles in flight), into **double-buffered `sD[2][Br]`**; producer syncs its own WG with `producer_sync()` = **`bar.sync 2, 128`** (third named barrier, id 2). Leader arrives `d_ready[sp]`; consumers drop their D-rowsum phase+sync and wait `d_ready[s]` before the dS phase.
  - Deadlock-free: producer's D-compute depends only on that tile's TMA (`full[sp]`, producer-issued), never on a consumer → `d_ready[sp]` always reachable. `full[sp]` now waited by both roles (non-consuming `try_wait.parity`, per-role parity). `sD` double-buffered (producer ≤2 ahead, gated by `empty[s]`).
- **Named barriers now: id 0 = `__syncthreads` (384, init only), id 1 = `consumer_sync` (256), id 2 = `producer_sync` (128).**

---

## Performance-tuning ladder (V13 → V26)

V1–V13 built the *structure* (portable → wgmma → TMA → fused → warp-specialized). From V13 on,
the structure is frozen (384-thread 3-warpgroup producer/consumer, single fused KV-centric
kernel + fp32-atomic dQ) and every version is a **measured latency cut**, all **bit-identical**
(same math, padding/scheduling/vectorization only), ≤170 regs, 0 spill. Config: B=8, Hq=12,
Hkv=4, G=3, S=4096, D=128, bf16, causal. cuDNN SDPA backward ≈ 2.78–2.81 ms (the target).

| V | change | median ms | Δ | note |
|---|---|---|---|---|
| V13 | (perf-era baseline) | 8.82 | — | 3.14× off cuDNN |
| V14 | `__expf` fast softmax (SFU exp) | ~ | | strips range-reduce ALU on wg0's crit. path |
| V15 | no-division incremental index math | ~ | | address-math cut |
| V16 | dV/dK transpose-staging elimination | ~ | | read Pᵀ/dSᵀ direct, Major::MN |
| V17 | hoist swizzle-index math | ~ | | strength-reduce `sw128_idx` |
| V18 | vectorize `dS` (bf16×2 + float2) | ~ | | 2 cols/step |
| V19 | vectorize P-write (bf16×2) | 7.77 | | −12% vs V13 band |
| V20 | swizzled D-rowsum + PD=3 pipeline | ~ | −3.65% | one more tile run-ahead |
| V21 | causal-mask specialization | 7.39 | | `fused_p_nomask` off-diagonal |
| V22 | **D-rowsum split** (separate kernel) | 5.89 | **−20.3%** | biggest win; compute→mem-bound |
| V23 | dQ transpose-staging elimination | 5.70 | −3.78% | 2.03× off cuDNN |
| V24 | **shared-store bank cut** (stride 72) | 4.99 | **−12.5%** | broke 5 ms AND 2× (1.78×) |
| V25 | bf16 dV/dK stage stride-72 | 4.9353 | −0.98% | last shared conflict → 1-way |
| V26 | **wg0-scoped sLSE barrier** | 4.6853 | **−5.71%** | 1.69× off cuDNN; IPC 1.77→1.87 |
| V27 | sP-reuse-chain break (sDS + dV overlap) | 4.6752 | −0.77% | hide dV wgmma under dS; −1 barrier |
| V28 | **dS loop bf16×2 → bf16×4** | 4.4927 | **−3.26%** | dynamic instrs −6%; broke 4.5 ms |
| V29 | exp2f softmax (fold scale·log2e) | 4.3990 | −0.52% | FMUL −189; bit-identical; **2.0× total** |
| V30 | **__launch_bounds__(384,1)** (+setmaxnreg scaffold) | 4.3222 | **−1.86%** | 157→168 regs; 1.55× off cuDNN |

Four regime shifts drove which lever worked: **compute/latency-bound** (V13–V21, cut
instructions/staging) → **L1/shared-throughput-bound** (V22–V25, cut shared traffic & bank
conflicts) → **feed/sync-bound** (V26–V27, keep the scheduler fed by scoping barriers / hiding
wgmma waits under ALU) → **memory-bound** (V28–V30, cut dynamic instructions until L1TEX became
the wall). The same class of lever can be inert in one regime and decisive in another — a
store-conflict fix did nothing while latency-bound but won −12.5% once L1/shared-bound (V24);
"instruction count is structural/unreachable" was wrong once the hot elementwise loops were
widened (V28, −3.26%).

## V14 — `__expf` fast softmax
V13 clone; the single change is the softmax exponential in `fused_p`: **`expf` → `__expf`** (the
fast SFU intrinsic), `fused_p_from_acc_v14`. IEEE `expf` wraps the `MUFU.EX2` hardware in
range-reduction + polynomial + reconstruction ALU; `__expf` issues the `EX2` directly, stripping
that scalar ALU from **wg0's critical-path softmax** (the `exp(S·scale − LSE)` over the full
64×64 P tile). Fast-math is *not* globally enabled — the rest of the kernel stays IEEE; only
this one exp is swapped. Bit-identical at the bf16 output: `__expf`'s ~2⁻²² relative error sits
~14 bits below bf16's ~2⁻⁸ ulp, so the rounded bf16 P is unchanged → dQ/dK/dV match.

## V15 — no-division incremental index math
The per-tile group/query-tile indices (`gC`, `qcC` and the derived global row/col bases) were
recomputed with `/` and `%` each iteration. V15 replaces them with **incremental updates**
(`if (++qcC == nQTiles) { qcC = qc0; ++gC; }`) — integer divide/modulo are multi-instruction on
the SM. Pure address-math instruction cut on the critical path; bit-identical.

## V16 — dV/dK transpose-staging elimination
`dV = Pᵀ·dO` and `dK = dSᵀ·Q` need the **transposed** P/dS as the wgmma A-operand. Through V15
that was built with an `ldmatrix→stmatrix` reshuffle into `sA_t`. V16 reads **Pᵀ/dSᵀ directly
from the swizzled `sP`** as a `Major::MN` operand: the SW128 swizzle makes the col(=k) axis the
contiguous 128-B role, so contracting over row=q is exactly a Major::MN read. Removes the
staging round-trip (LDSM/STSM + a barrier) for dV/dK. **dQ stayed staged** (thought to need a
different orientation — a conservative call reversed at V23). Bit-identical (same fragment map).

## V17 — hoist swizzle-index math
The swizzle index `sw128_idx(r,c) = r*64 + ((c>>3 ^ r&7)<<3) + (c&7)` was recomputed per
element in the elementwise loops. V17 hoists/strength-reduces the row-dependent part out of the
column loop (the `r*64` and `r&7` terms are loop-invariant across columns). Address-math cut.

## V18 — vectorize `dS` (bf16×2 + float2)
`dS = P ⊙ (dP − D)` was a scalar loop. V18 processes **2 adjacent columns per step**: read P as
`__nv_bfloat162` (bf16×2), read the fp32 `dP` pair as `float2`, compute, write P back as bf16×2.
Halves the loop-trip count and the load/store instruction count of the elementwise phase.

## V19 — vectorize the P-write
The softmax epilogue `fused_p` wrote P to `sP` one bf16 at a time. V19 writes **bf16×2** (2
columns per `STS`), matching V18's read width. Lands the V13→V19 band at **7.77 ms**.

## V20 — swizzled D-rowsum + 3-deep pipeline
Two changes: (a) the producer-side D-rowsum reads `O` through a **swizzled** buffer (removing a
strided-load conflict on the producer); (b) **pipeline depth 2 → 3** (`PD=3`) — one more tile of
producer run-ahead to hide the TMA-feed latency. Proved the producer feed was a real throttle:
**−3.65%**. (PD=4 was tested later and was flat — the feed isn't purely depth-limited.)

## V21 — causal-mask specialization
`fused_p` applied the causal element-mask (`global_col > global_row → 0`) on every tile, but
only the **diagonal** tile (`qcC == qc0`) actually straddles the causal boundary. V21 splits
into `fused_p_from_acc` (masked, diagonal tile) and **`fused_p_nomask`** (no mask compute, all
fully-unmasked tiles) — dropping the per-element compare on the common case. Lands at **7.39 ms**.

## V22 — D-rowsum split into a separate kernel (BIGGEST early win, −20.3%)
The single largest cut. `D[r] = rowsum(dO·O)` was computed inside the fused kernel (on the
producer warpgroup). cuDNN computes it in a **separate** kernel (`compute_dot_do_o_specialized`);
V22 does the same — `compute_drowsum_v22`, a light grid-stride kernel over all `B·Hq·S` rows,
warp-per-row, replicating the exact fp32 accumulation order → **D is bit-identical**. The main
kernel then **deletes** the inline `producer_drowsum`, the `sO_sw` buffer, and the entire **O
TMA** (O only ever fed the D-rowsum; dV uses dO). Payoff compounded three ways: dropped the #1
short-scoreboard stall (the D shuffle-tree), **freed 32 KB smem**, and **deleted a whole tile's
O-TMA traffic per iteration** → the producer feeds far faster. **7.39 → 5.89 ms (−20.3%)**;
the bottleneck crossed from compute/latency-bound to **memory-bound** (long_scoreboard now #1).
Both the agent and I predicted this flat — wrong by 20%. Launcher now runs D-kernel → main →
convert and times the **total** (honest vs cuDNN's multi-kernel total). *(Our D-kernel = 63 µs
beats cuDNN's 100 µs; we also fuse the GQA head-reduction cuDNN pays a separate kernel for.)*

## V23 — dQ transpose-staging elimination
The last staged A-operand. `dQ = dS·K` still round-tripped `dS` through `sA_t` via
`sp_to_sAt` (ldmatrix→stmatrix). V23 reads **dS directly from the swizzled `sP` as a `Major::K`
operand** (`run_gemm_dQ_half_te`): dQ contracts over col=k, and the swizzle makes col=k the
contiguous axis, so a Major::K read hits the right contraction axis (structurally identical to
the proven S-GEMM `A=Q_sw` read — *not* the V8 failure, which read a transposed operand as
Major::K). The single swizzled `sP` now serves **both** orientations (dV/dK Major::MN, dQ
Major::K) from one buffer. SASS: `LDSM 2→0, STSM 2→0, BAR 16→15`; regs **162→157**. **5.89 →
5.70 ms (−3.78%), 2.03× off cuDNN.** `sA_t` is now used *only* as the dV/dK epilogue stage.

## V24 — shared-store bank-conflict cut (stride 72) — broke 5 ms AND 2×
Now L1/shared-throughput-bound (66% L1TEX), the two per-tile fp32 D-fragment stores were the
wall: `dP→sdP` (stride 65 = 4-way scalar `STS.32`) and the dQ-stage `→sS` (stride 64 = **8-way**).
Both re-padded to **stride 72 = 2-way** (the hard bank floor — even column-pairs reach only 16
banks, so 2-way is the minimum; 1-way impossible for fp32). The dP store also packs
`STS.32→STS.64` (half the store instructions). The dQ-flush read stride is **decoupled** from
the write (`atomic_flush_stage_s<Br,64,72>`) so global coalescing is preserved. **Store conflicts
234.6 M → 4.5 M (−98%); 5.70 → 4.99 ms (−12.5%), 1.78× off cuDNN.** A store-conflict fix only
converts to wall-clock in this L1/shared-throughput-bound regime (it did nothing while the
kernel was latency-bound) — the regime, not the fix, decides. Bit-identical (padding only).

## V25 — bf16 dV/dK stage bank-conflict cut (stride 72 → 1-way)
The last reducible shared conflict. The bf16 dV/dK **epilogue** stage (`stage_acc_bf16`, stride
64) was still **8-way**. Re-padded to **stride 72** via decoupled-stride helpers
(`stage_acc_bf16_s` / `store_stage_vec_s`, data width 64, row stride 72). Because bf16 packs 2
cols/word (`STS.32`), the packed word-index `r·36 + c/2` maps to `(i·4 + j) mod 32` over the warp
→ **all 32 banks distinct → 1-way (conflict-free)**, better than fp32's 2-way floor. `sA_t` is
now dedicated to this stage (dQ reads `sP` directly since V23), so no aliasing. **Store conflicts
4.5 M → 0.82 M; 4.99 → 4.9353 ms (−0.98%).** Source-page attribution here proved the remaining
33.5 M "shared-load conflicts" are **wgmma operand fetches** (`HGMMA gdesc.tnspA`) — inherent,
not reducible — so the shared-conflict lever is fully mined out.

## V26 — wg0-scoped sLSE barrier (feed/sync regime, −5.71%)
With shared traffic exhausted, the wall is **feed/sync** (tensor only 20% active; #1 stall =
mbarrier TMA-feed spin). The consumer chain has 7 `bar.sync 1,256` per tile, each guarding a
real hazard — but the **first** one (right after the `sLSE` load) guards a **wg0-only** hazard:
`tid<64` write `sLSE[tid] = d_LSE[…]` (a **global load, ~400-cycle latency**), and wg0's
`fused_p` reads it. wg1 (`dP = dO·Vᵀ`) never touches `sLSE`. V26 scopes it to a **128-thread
`bar.sync 3,128`** (id 3 free) that only wg0 runs (`if (wg==0) consumer_sync_wg0()`), so **wg1
skips it and overlaps its independent dP-GEMM with the LSE-load latency**, re-converging at the
next `bar 1,256`. Pure scheduling win — instruction count flat, but **IPC 1.77 → 1.87**
(no-eligible 55.7% → 53.2%). **4.9353 → 4.6853 ms (−5.71%), 1.69× off cuDNN.** Introduced the
"wg-scope a barrier whose hazard is intra-warpgroup so the other wg overlaps latency" lever.

## V27 — break the sP-reuse chain (separate `sDS` + dV overlap, −0.77%)
The consumer's 6 GEMMs serialized because `dS` **overwrote `sP` in place**, forcing dV (which
reads `sP`) to complete before dS. V27 gives dS its own **`sDS`** buffer, splits dV into
issue/wait (`run_gemm_dVdK_half_te_issue`/`_wait`), and runs **dV's wgmma in flight across the dS
elementwise** — hiding the wgmma latency under ALU — while the dV→dS WAR barrier drops (256-thread
barriers 9→8). First win after the profile reframe: the kernel is **consumer-bound** (the #1
"stall" is the *producer idling on `empty[]`*, benign), so the lever is shortening the consumer
serial chain. Bit-identical.

## V28 — dS loop widened bf16×2 → bf16×4 (−3.26%, broke 4.5 ms)
The hot 256-thread dS loop went from 2 columns/iter to **4** (two bf16×2 reads of `sP` + one
`float4` `sdP` + two bf16×2 writes to `sDS`). Iterations halve → the per-iteration index math,
loop branch, and `sdP` reads halve while the compute is unchanged. Static count *rose* (bigger
body) but **dynamic executed instructions fell 6%** → −3.26%. Reversed the "instruction count is
structural (cuDNN's TMA-addressing, unreachable)" pessimism: the address math in the hot
elementwise loops *is* reducible by widening. Bit-identical.

## V29 — exp2f softmax (fold scale·log2e), ported from the cuDNN-beating forward
`fused_p`: `__expf(acc*scale − lse)` → **`ex2.approx.ftz(fmaf(acc, scale2, −lse2))`**
(`scale2 = scale·log2e`, `lse2 = lse·log2e`). Since `exp(x)=2^(x·log2e)`, this folds the `*scale`
and `__expf`'s internal `*log2e` into **one FFMA** + one raw SFU `ex2.approx`. **FMUL −189,
static −392.** Bit-identical — `ex2.approx.ftz`'s ~1-ulp fp32 difference rounds to the same bf16
(check max_abs unchanged). **4.3990 ms — crosses 2.0× total speedup.** After V28/V29 the kernel
is **memory-bound** (60% L1TEX SOL vs 45% compute).

## V30 — `__launch_bounds__(384,1)` + setmaxnreg scaffold (−1.86%)
The backward had *no* launch bounds → the compiler self-capped at **157** regs. Adding
`__launch_bounds__(384,1)` let it use up to 170 (took **168**); the extra headroom held more live
values and cut register-shuffle instructions (−2.6%) for **−1.86%** — *more registers = faster
even at the same 1-block/SM occupancy*. The `setmaxnreg 40/232` calls are scaffolding for a
ping-pong that turned out **infeasible** (below). **4.3222 ms, 1.55× off cuDNN.**

## The landing — why V30 is the floor for this architecture
The forward (`GQA_fwd_ref.cu`, gqa_v62) *beats* cuDNN by keeping `P` in **registers** (RS-wgmma)
and running an **FA3 consumer ping-pong** (two warpgroups on different tiles, softmax overlapping
tensor). Neither ports to the backward: (1) its wgmma A-operands are **transposed** (Pᵀ for dV,
dSᵀ for dK), so `P`/`dS` must live in **swizzled smem** — two independent warpgroups would need
2× the staged buffers (~256–307 KB > the 232 KB cap); (2) the **persistent `dv`/`dk`
accumulators** forbid cross-tile tensor overlap (holding a scaleD=1 group live across another
group races — the rule that killed V28's dK∥dQ). The only rule-legal overlap (wgmma ∥ ALU) is
already spent (V27 dV∥dS). And the kernel is memory-bound with **non-reducible** shared loads
(the inherent HGMMA operand fetches) at occupancy hard-locked to 18.75% (157–168 regs × 384 + 200
KB smem both pin 1 block/SM). `--fmad=true` is already the default, so the 104 M "non-fused" FP32
ops are non-contractable. **No reachable lever remains.**

**Final: 8.82 ms → 4.3222 ms — 3.14× → ~1.55× off cuDNN, 2.04× total, every version
bit-identical, 0 spill throughout.** The residual gap is cuDNN's (and the forward's) register-
source-operand structure, which the backward's transposed operands can't fit in Hopper smem.

---

## Recurring concepts & gotchas (cross-version)

- **Two-granularity causal masking:** tile-level skip (`break` in dQ / loop-start in dKdV) removes fully-masked tiles; element-level mask (`global_col > global_row → 0`) handles the diagonal tile. Present in every version.
- **Persistent HW accumulators (V3+):** `dq`/`dk`/`dv` accumulated in-hardware via wgmma `scaleD=1` across the whole tile loop, read out once. Requires an **extra `fence_operandN` outside the loop** before the final store (the accumulator is live across many async GEMMs).
- **Two wgmma proxy fences (V3+):** `fence_operandN<N>` brackets accumulator registers (wgmma writes them async; blocks ptxas from hoisting reads before `wait_group`); `fence.proxy.async.shared::cta` bridges generic-proxy fill/scatter writes → async-proxy wgmma operand reads. Both invisible in single-tile probes, fatal in the loop.
- **2× m64n64 atom split (V5 dQ, V7 all outputs, V9 per-WG):** D=128 = two 128-B swizzle atoms. Splitting a m64n128 output into two N=64 halves keeps each on the validated single-atom recipe AND (V5) caps register pressure at `acc[32]` so persistent `dv`/`dk` don't spill.
- **Swizzle vs plain vs transpose:** the operand read K-major on the direct SW128 path is swizzled; transposed operands use either `fill_trans`+no-swizzle (V3–V6) or `trans-b=1` in-place (V7+); elementwise-only operands (O, dO-for-D) stay plain.
- **Bank-conflict stride rule:** any smem row/core-matrix stride that is a multiple of 128 B (32 banks × 4 B) aliases. Fix by padding off the boundary (V6) or removing the strided access entirely (V7 `trans-b=1`, V12 ldmatrix/stmatrix).
- **Named barriers under warp specialization (V11+):** the producer must never join a barrier the consumers use, or it deadlocks. id 0/384 (init), id 1/256 (consumers), id 2/128 (producer).
- **dQ is always atomic in the fused kernels (V5+):** it reduces over the key axis, so no single KV-block owns it → fp32 scratch + convert. dK/dV are block-owned → plain accumulate, no atomics.

## The arc in one breath
portable wmma (V1/V2) → wgmma correctness (V3) → TMA+swizzle+double-buffer (V4) → fused KV-centric, atomic dQ (V5) → bank-conflict padding (V6) → transpose-buffer elimination via `trans-b=1` (V7) → warp-shuffle D-rowsum (V8) → 2-WG column bisection (V9) → coalesced writeback (V10) → 3-WG producer/consumer (V11) → `ldmatrix→stmatrix` reshuffle (V12) → register-fused softmax + producer-side D-rowsum (V13) → *[structure frozen; perf tuning begins]* → no-div + hoist address math (V15/V17) → dV/dK transpose-staging elimination (V16) → vectorize dS/P-write (V18/V19) → PD=3 pipeline (V20) → causal-mask specialization (V21) → **D-rowsum split kernel, −20% (V22)** → dQ transpose-staging elimination (V23) → **shared-store bank cut, −12.5%, sub-2× (V24)** → bf16-stage 1-way (V25) → **wg0-scoped sLSE barrier, −5.7% (V26)** → sP-reuse-chain break + dV∥dS overlap (V27) → **dS bf16×4, −3.3% (V28)** → exp2f softmax, 2.0× total (V29) → **__launch_bounds__ register headroom, −1.9% (V30)**.

**8.82 ms → 4.3222 ms: 3.14× → ~1.55× off cuDNN, 2.04× total speedup, every version
bit-identical, 0 spill throughout.** Regimes: compute-bound → L1/shared-bound → feed/sync-bound
→ memory-bound. The recurring lesson — a lever dead in one regime revives in another; trust the
current measurement, not the old verdict (D-split, the store-conflict cut, and the "structural"
instruction count were each predicted flat and each landed a headline win). The residual to
cuDNN is structural: register-source wgmma operands the backward's transposed dV/dK can't fit
in smem.

**8.82 ms (3.14×) → 4.6853 ms (1.69× off cuDNN), every version bit-identical.** Three regimes:
compute-bound (cut instructions) → L1/shared-bound (cut bank conflicts) → feed/sync-bound (keep
the scheduler fed). The recurring lesson: a lever dead in one regime revives in another — trust
the current measurement, not the old verdict (D-split and the store-conflict cut were each
predicted flat and each landed a headline win).
