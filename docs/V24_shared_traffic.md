# V24 — Shared-store bank-conflict cut (L1/shared-traffic reduction)

**Verdict: BUILT — a real, reducible L1TEX cut found in the shared-STORE path.**
Not a NO-GO. The store-side bank conflict (ncu's #1 L1 table finding, *not* the loads)
is genuinely reducible by a pure smem-stride change, at zero register cost and free smem.

**Status: compile-validated on this box (sm_120), analytically bit-identical + conflict
reduction proven from the deterministic bank math. The MEASURED L1TEX / wall-clock delta
must be taken on the H200 — this box has no wgmma/Hopper hardware to run or profile it.**
Do not report a speedup until the H200 ncu run confirms the store-wavefront drop.

---

## Step 1 — LOCALIZE the L1/shared traffic (from reports/gqa_v23_profile.ncu-rep)

Re-imported the rep (raw + source pages). The L1TEX bottleneck (66.4% mem SOL) splits
cleanly into a LOAD side and a STORE side, and they are **wildly asymmetric**:

| shared op | requests | wavefronts | **bank-conflict wavefronts** | avg conflict | ncu est. speedup |
|---|---|---|---|---|---|
| **stores** | 78.0 M | 347.2 M | **234.6 M (67.6% of store wf)** | **4.5-way** | **44.98%** |
| loads | 104.0 M | 164.2 M | 59.8 M (36.4% of load wf) | 1.6-way | 24.26% |

Store conflicts (234.6 M) outweigh load conflicts (59.8 M) by **~4×**. The store side is
THE reducible L1TEX inflation. (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`
= 234,611,194; `_op_ld.sum` = 59,840,510 — confirmed from the binary rep.)

### Attributing the store conflicts to buffers/ops (SASS stride decode)

Source-page SASS (offsets decode the smem row stride directly). The per-tile fp32 mma
**D-fragment** stores to `sS`/`sdP` are the conflict generators:

| store op (per tile) | buffer | stride | SASS | bank conflict |
|---|---|---|---|---|
| `store_acc_smem_v6` (dP write, wg1) | sdP | **65** (`+0x820` = 8×65 fp32) | scalar STS.32 ×4 | **4-way** |
| `stage_acc_f32<64>` (dQ stage, both wg) | sS/sdP | **64** (`+0x800` = 8×64 fp32) | auto-vec **STS.64** | **8-way** |

The other stores are already mitigated and are *not* the conflict source:
- `fused_p` (P→sP) and `dS` in-place (→sP) are **bf16×2** packed (V18/V19), on the
  1024-aligned XOR-swizzled `sP` — low conflict.
- Epilogue `stage_acc_bf16` (→sA_t) is stride-64 but runs **once per KV-block**, not per
  tile — negligible frequency.

### Why the fp32 fragment store conflicts (the bank math — deterministic)

The m16n8k16 D-fragment hands lane `l` the rows `r = w*16 + (l>>2)` (8 rows per warp) and
column pair `c = nt*8 + (l&3)*2` (always **even**). A same-column STS across the warp lands
on fp32 bank `(r·S + c) mod 32`, where `S = row stride mod 32`:

| stride S | S mod 32 | rows collapse? | bank multiplicity |
|---|---|---|---|
| 64 (`stage_acc_f32`) | 0 | all 8 rows → same banks | **8-way** |
| 65 (`store_acc_smem_v6`) | 1 | triangular pileup | **4-way** |
| **72 (V24)** | **8** | rows map to 4 classes × 2 | **2-way** ✅ |

**2-way is the hard floor** for this fragment: `c` is always even ⇒ every store base bank is
even ⇒ only 16 of 32 banks are reachable by a single scalar stride ⇒ 32 lanes over 16 banks
⇒ ≥2-way, always. Reaching 1-way would need `{r·S mod 32 : r=0..7}` = `{0,8,16,24,1,9,17,25}`,
which is not an 8-term arithmetic progression ⇒ **no single stride can be conflict-free.**
`S ≡ 8 (mod 32)` (=72) is the optimum, and being **even it keeps the (c,c+1) STS.64 packing.**

---

## Step 2 — which is REDUCIBLE (the crux)

- **GEMM-operand reads** (wgmma reads sK/sV/sQ/sdO/sP via descriptors) — **necessary,
  irreducible.** cuDNN runs the same 5 GEMMs. The `sP` 3× re-read (dV Major::MN, dK Major::MN,
  dQ Major::K) is *not* redundant: each is a different contraction axis on the same tile, and
  caching P/dS in registers to avoid the re-reads blows the 170-reg / no-TMEM wall (the
  documented V24-KV-block / TMEM ceiling). Leave as-is.
- **cross-wg dP share** (`store_acc_smem_v6` dP → sdP so all threads see wg1's dP for dS) —
  **necessary**, confirmed. But its *stride* is the free lever (below).
- **dS in-place pass** (read sP+sdP → write sP) — already **bf16×2** (V18); read of sdP is
  **2-way regardless of stride** (row is constant per warp), so V24's stride change does not
  regress it. Minimal, leave.
- **dQ staging round-trip** (stage → atomic_flush, the V10 coalescing trick) — the *coalesced
  atomic* is necessary (direct = −17%, V10). But the **STORE half is 8-way** and that is pure
  waste: the stride can move to 72 with the read staying contiguous/coalesced. **REDUCIBLE.**
- **bank conflicts** — the fp32 fragment stores at stride 64/65 are the leftover conflict
  inflation removable by padding at **zero other cost**. **This is the V24 cut.**

---

## Step 3 — the cut (BUILT as `gqa_backward_v24_kv`)

Clone of V23, **bit-identical numerics** (padding-only: the fragment→(row,col)→value map and
the coalesced dQ atomic are untouched; only the physical smem row stride of `sS`/`sdP` moves
65→**72** = `SS_STRIDE_V24`). Two per-tile fp32 fragment stores change:

1. **dP store** `store_acc_smem_v6<Bc, SS_STRIDE_V24>` — stride 65 (4-way, scalar STS.32) →
   72 (**2-way, STS.64**). Conflict halved **and** store-instruction count halved (scalar→vec).
2. **dQ stage** `stage_acc_f32<64>` → `store_acc_smem_v6<64, SS_STRIDE_V24>` (identical body,
   stride 64→72): 8-way STS.64 → **2-way STS.64**. Conflict **quartered**.
   Paired read `atomic_flush_stage_s<Br,64,SS_STRIDE_V24>` walks the same 72-stride,
   64 real cols/row — consecutive-thread → consecutive-global-address coalescing preserved.

Reads verified non-regressing: dS read of sdP = 2-way at any stride; atomic_flush read is
contiguous (conflict-free). Load side (1.6-way) is already near-optimal — not touched.

### Compile validation (sm_90a, this box cross-compiles; run needs H200)

| | V23 | **V24** |
|---|---|---|
| registers | 157 | **157** (unchanged) |
| spill | 0 | **0** |
| smem | 190,800 B | **193,872 B** (+3,072 B) |
| barriers | 3 | 3 |

+3,072 B smem is free: total 193,872 « 227 KB opt-in wall; occupancy is register+smem pinned
at **1 block/SM regardless**, so the extra smem changes nothing. `-gencode
arch=compute_90a,code=sm_90a` (NOT `-arch=sm_90a`, which mis-targets `.target sm_90` and
ptxas rejects wgmma — the known gotcha).

### Expected effect (analytical — CONFIRM on H200)

The two changed stores go from (4-way, 8-way) → **2-way**, and the dP store also halves its
store count. If they are the bulk of the 234.6 M store conflicts (they are the only per-tile
fp32 fragment stores; bf16×2 sP stores are low-conflict, epilogue is per-KV-block), expect the
**shared-store conflict wavefronts to fall roughly 3–4× on those ops**, cutting a large slice
of the 234.6 M and pulling shared-store L1TEX down materially. ncu's 44.98% is the ceiling if
ALL store conflicts vanished; V24 captures the fragment-store portion. **Wall-clock conversion
is not guaranteed** (V23 is partly latency-bound too) — measure `l1tex__data_bank_conflicts_
pipe_lsu_mem_shared_op_st.sum` and duration on the H200 before claiming a win. Do not ship as
the new floor until measured; if it comes back flat, keep V23 as the 2.03× floor.

### How to validate on the H200
```
cmake --build build            # or the project's nvcc line with -gencode compute_90a,sm_90a
./build/bin/gqa_bwd            # main() runs V24's check() (bit-identical PASS expected)
ncu --set full -k gqa_backward_v24_kv -o reports/gqa_v24_profile ./build/bin/gqa_bwd
# compare _op_st.sum conflicts + Duration vs gqa_v23_profile
```
