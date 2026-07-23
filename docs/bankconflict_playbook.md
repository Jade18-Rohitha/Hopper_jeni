# Bank-Conflict Elimination Playbook (ported from Blackwell GQA-bwd V6→V10)

Reference for **Hopper V7+** (`Hopper/src/attention/GQA_bwd.cu`). Source of truth:
`Blackwell/src/attention/GQA_sm103_bwd.cu` (kernels/launchers V6–V10) and
`Blackwell/documents/V6_profile_analysis.md`, `V10_profile_analysis.md` (only V6 and V10
profile docs exist; V7/V8/V9 were validated by precision + wall-clock only — one full ncu
pass was spent on the converged V10 state).

**Transfer caveat:** Blackwell feeds **tcgen05.mma**; we feed **wgmma**. The tensor-core op
differs but the *shared-memory layout / swizzle / transpose-bit* mechanism transfers cleanly —
Hopper wgmma reads SS-operands via a 64-bit matrix descriptor with **swizzle-mode bits [63:62]**
and per-operand **transpose immediates** (`Major::K=0` / `Major::MN=1`), which are the direct
Hopper equivalents of Blackwell's swizzled TMA descriptor + idesc `b_major` bit (bit 16).

---

## The two ceilings we share with Blackwell

| Ceiling | Our V6 (Hopper) | Blackwell fix arc |
|---|---|---|
| Residual bank conflict | 15.1-way LOAD, 93% wavefronts (transposed reads of TMA-plain `sQ_pl`/`sdO_pl`/`sK_pl`) | V6→V10 swizzle arc drove LOAD-conflict to **gone**, STORE to ~1.1-way |
| Occupancy | 6.25% (1 block/SM, smem-limited 223 KB) | V8+V9+V10 smem savings → **2 blocks/SM (~49%)** as a *side effect* of deleting transpose buffers |

The swizzle arc is the one lever that hits **both** ceilings at once.

---

## Version-by-version

### V6 — pad the float scratch (`sS`/`sdP`)  [we already have this]
- **Technique:** pad the row-major `float` scratch buffers the tensor-core readout writes and the
  P/dS loops read, from stride `Bc` → `BcP` (`Bc+CANON_PAD`). Same padded buffer serves both the
  write (`tmem_readout`) and every read (`sS[row*BcP+j]`), so it is a **two-for-one** fix: kills
  the store-side *and* the fixed-column/varying-row read-side conflict simultaneously.
- **Before→after:** STORE 7–9-way → **1.5–1.6-way** (theoretical floor for 4-byte/bank floats);
  LOAD 6.6-way → **gone**; excessive-shared wavefronts 81–86% → 20–23%. **−48.6%** combined
  wall-clock — the single biggest win of the whole arc. Occupancy unchanged (pure conflict fix).
- **Applicability to our residual:** **LOW (already done).** Our V6 already padded `sS/sdP/sP`;
  this is exactly why our paddable conflicts are gone and only the unpaddable transposed-read
  residual survives. Nothing new to port.

### V7 — swizzle ONE operand (Q), plumbing only  [opening move]
- **Technique:** replace the software canonical-repack of Q with a **hardware 128B-swizzled TMA
  descriptor** (`tma_Q_sw128`) that writes swizzled bytes *directly* into a plain `[Br*D]` `sQ`
  buffer — no `repack_canon_pad`, no separate row-major staging buffer (`stgQ` deleted). The MMA
  reads `sQ` via a swizzled descriptor (`make_smem_desc_sw128`), recomputed each k-step from a
  pointer shifted by `k*32` bytes (**not** the canonical `advance_desc_katom` math — that is
  canonical-layout-only and would be silently wrong on swizzled bytes). Mixing one swizzled
  operand with one non-swizzled operand in the same MMA is legal — each descriptor is independent.
- **Before→after:** **~0%** (predicted null, confirmed). Q's orientation wasn't the conflict
  source; this step only lays the swizzle plumbing and validates one operand end-to-end before
  touching the operands that actually have a transpose duplicate.
- **Applicability:** **MEDIUM.** No direct win, but it is the *validation harness* — port the
  swizzled-TMA + swizzled-wgmma-descriptor path on one operand first, confirm bit-identical, then
  apply to the operands that matter (V8/V9/V10 pattern). Skipping straight to V8 risks debugging
  swizzle + transpose-bit + buffer-elimination all at once.

### V8 — swizzle K **and delete its transpose duplicate `sKt`**  [first real win]
- **Technique:** K gets its own swizzled TMA descriptor (`tma_K_sw128`). The **two** canonical K
  buffers — `sKqk` (K-major) and `sKt` (a full physical transpose for `dQ+=dS@K`) — collapse to
  **one** plain `[Bc*D]` `sK`. The alternate MN-major contraction orientation is selected by the
  **MMA instruction descriptor's `b_major` bit (bit 16)** instead of a second physical buffer:
  - `S=Q@Kᵀ` reads `sK` **K-major** (no bit): descriptor from `sK + k*32` bytes.
  - `dQ+=dS@K` reads the **same** `sK` **MN-major** (bit16 set): descriptor from `sK + k*2048`
    bytes (16 rows × 128 B/row), geometry `SBO=8*128`, `LBO=0` (confirmed unused).
  - `repack_canon_pad` **and** `repack_canon_T_pad` both DELETED.
- **Before→after:** **−6.9%** wall-clock. Deleting `sKt` + the two repack passes removes the
  strided transposed shared read entirely (the conflict has no buffer to live in anymore) and
  drops smem/registers.
- **Applicability to our residual:** **HIGH.** This is the exact template for our `sK_pl`
  transposed-read residual. Hopper equivalent of the `b_major` bit is the wgmma **`Major::MN=1`
  transpose immediate** on that operand + the appropriate MN-major SBO/base-advance
  (see `wgmma-noswizzle-sbo` memory: SBO scales with full K-width; base advances per k-step).

### V9 — swizzle Q in dKdV, delete `sQt`  [repeat on next operand]
- **Technique:** identical recipe applied to Q inside the dKdV kernel: swizzled TMA, single plain
  `sQ`, `sQt` (physical transpose for `dK+=dSᵀ@Q`) eliminated via the same `b_major`/MN-major
  descriptor (SBO=8*128, LBO=0, `k*2048`-byte advance). dQ kernel reused unchanged from V8.
- **Before→after:** **−7.0%** — almost identical to V8, strong evidence of one repeatable
  mechanism, not noise.
- **Applicability:** **HIGH.** Same template as V8, applied to our `sQ_pl` transposed read in the
  dK path.

### V10 — swizzle dO, delete `sdOt`; handle the plain-bytes second use  [closing move]
- **Technique:** same recipe for dO (`dP=dO@Vᵀ` K-major / `dV+=Pᵀ@dO` MN-major from one swizzled
  `sdO`, `sdOt` deleted). **The one twist:** dO has a *second* use — the elementwise
  `D[r]=rowsum(dO·O)` reduction needs dO's **plain row-major bytes**, which a 128B-swizzled buffer
  cannot provide (the XOR permutation is opaque to plain element indexing). Solution: **load dO
  twice** — once plain into `stgdO` (feeds `D[r]` unchanged), once swizzled into `sdO` (feeds the
  MMAs). Costs extra TMA bandwidth on dO only; no restructuring of `D[r]`.
- **Before→after:** **−6.7%.** Cumulative **V6→V10 −19.8%**; all three transpose duplicates gone.
  **Occupancy DOUBLED 1→2 blocks/SM (~49%)** as a side effect of the cumulative smem/register
  savings (`Block Limit Shared Mem = 2`). LOAD conflict gone, STORE 1.1–1.2-way (ncu residual
  est-speedup ~7% — floor). Compute throughput 47% → **67–71%**, IPC 1.9 → **2.7–2.9**,
  No-Eligible ~50% → ~30%.
- **Applicability to our residual:** **HIGH + directly relevant caveat.** Our residual explicitly
  includes the **D-rowsum reads of `sdO_pl`/`sO_pl`** — so we hit V10's exact plain-bytes-second-use
  problem. Port the double-load (plain for rowsum, swizzled for MMA) for dO/O.

---

## Applicability summary

| Version | Technique | Hits transposed-read residual? | Hits occupancy? | Port priority |
|---|---|---|---|---|
| V6 | pad float scratch | no (already done) | no | LOW (have it) |
| V7 | swizzle 1 operand (plumbing) | indirectly (harness) | no | MEDIUM (do first) |
| V8 | swizzle K + delete `sKt` via transpose bit | **yes (`sK_pl`)** | **yes (smem freed)** | **HIGH** |
| V9 | swizzle Q + delete `sQt` | **yes (`sQ_pl`)** | **yes** | **HIGH** |
| V10 | swizzle dO + delete `sdOt` + double-load for rowsum | **yes (`sdO_pl` + D-rowsum)** | **yes** | **HIGH** |

---

## Recommended Hopper V7 plan

Port the **V8 pattern first, on K** (our `sK_pl`), preceded by the **V7 single-operand
plumbing validation**:

1. **V7-style plumbing (do first, one operand):** add a 128B-swizzled TMA descriptor for one
   operand, write directly into a plain `[rows*D]` buffer (no `fill_trans`, no separate staging),
   read it in wgmma via a swizzle-mode descriptor. Confirm bit-identical output. This de-risks the
   swizzle + descriptor math before touching a transpose duplicate.
2. **V8-style elimination (the actual win):** collapse each `*_sw`/`*_pl` + transpose-copy pair
   (`sK_sw`/`sK_pl`, then `sQ_pl`, then `sdO_pl`) to **one** swizzled buffer. Serve the transposed
   GEMM from the *same* bytes via the wgmma **`Major::MN=1` transpose immediate** (Hopper analogue
   of Blackwell's idesc `b_major` bit) with MN-major SBO/base-advance, instead of a physically
   transposed `fill_trans` buffer. This deletes the strided shared read that *is* our 15.1-way
   residual — the conflict has no buffer left to live in.
3. **For dO/O (V10 twist):** keep one plain-row-major load feeding the `D[r]=rowsum(dO·O)`
   reduction, plus one swizzled load feeding the MMA — a deliberate double-load, since swizzled
   bytes can't serve plain element indexing.

**Why this order / why it wins twice:** the transposed reads are unpaddable (§2 of `V6_analysis.md`),
so restructuring — not more padding — is the only lever, exactly as on Blackwell. Deleting the
`fill_trans` transpose duplicates simultaneously (a) removes the strided read that causes the
15.1-way conflict and (b) frees the smem that pins us at 6.25%/1-block-per-SM — Blackwell got the
1→2 block/SM occupancy jump *for free* from the same deletions. One technique, both of our ceilings.
Note our D=128 SBO differs from Blackwell's D=64 (`wgmma-noswizzle-sbo`: SBO = (K/8)*128 B, and the
swizzle-mode descriptor geometry must be re-derived for 128B swizzle at D=128, not copied verbatim).
