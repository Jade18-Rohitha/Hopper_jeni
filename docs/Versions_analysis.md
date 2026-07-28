# GQA Backward — Version Ladder (V1 → V13)

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
portable wmma (V1/V2) → wgmma correctness (V3) → TMA+swizzle+double-buffer (V4) → fused KV-centric, atomic dQ (V5) → bank-conflict padding (V6) → transpose-buffer elimination via `trans-b=1` (V7) → warp-shuffle D-rowsum (V8) → 2-WG column bisection (V9) → coalesced writeback (V10) → 3-WG producer/consumer (V11) → `ldmatrix→stmatrix` reshuffle (V12) → register-fused softmax + producer-side D-rowsum (V13).
