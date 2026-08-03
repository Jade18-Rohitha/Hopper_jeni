# Research — where the backward can still go (post-V30, 4.32 ms / 1.55× off cuDNN)

## The reframe: we are memory-bound, cuDNN is compute-bound at the SAME occupancy
From `sdpa_bwd_d128.ncu-rep` (cuDNN's `flash_bprop_wgmma`, same B=8 Hq=12 Hkv=4 S=4096 D=128):

| | cuDNN bwd | **our V30** |
|---|---|---|
| occupancy | 18.75% (1 block/SM) | 18.75% (1 block/SM) |
| registers | 168 | 168 |
| **Compute SOL** | **59.4%** (tensor-bound) | 45% (**memory-bound**) |
| grid | **132 (persistent)** | 2048 |
| instructions | ~556 M | ~1.71 B (3.45×) |

**cuDNN is not more occupied and not more register-rich — it does ~1/3 the instructions**, so
it's tensor-limited where we're L1TEX-limited. The lever is therefore NOT more occupancy (2
blocks/SM is a dead end — cuDNN is 1 block/SM too) but **cutting the address-math + LDS
instructions until we cross to compute-bound.** That's the same instruction-count lever V28/V29
converted on — it just has bigger, structural cuts left. Ranked by reward/effort:

## 1. TMA-store the dV/dK (and dQ) outputs — HIGH reward, MEDIUM effort
cuDNN uses **0 explicit output address math** (TMA hardware addressing). We use
`store_stage_vec_s` / `atomic_flush_stage_s` — manual `uint4` writes with per-element
`base + row*D + col` LEA/IMAD chains (the dQ-atomic address math is the #1 *executed*
address-math block). Replacing the epilogue stores with **`cp.async.bulk.tensor.2d` (TMA store)**
would delete that address math *and* the `sA_t` staging buffer (freeing ~18 KB smem). The dQ
atomic is harder (TMA can't atomic-add), but the dV/dK stores are a clean TMA-store target.
→ directly cuts the biggest instruction category (address math, 36% of static).

## 2. RS-wgmma — move P / dS to registers — HIGHEST reward, HIGH effort
This is the forward's real edge (it packs P into `pa[32]` register operands for P@V). It would
(a) delete the `sP`/`sDS` smem round-trips and their LDS traffic → toward compute-bound, and
(b) free the smem that blocks the FA3 ping-pong. **The hard part:** the backward's A-operands are
*transposed* (Pᵀ for dV, dSᵀ for dK), so the register fragment must be the transpose of the
softmax/dS output layout. Research needed: the exact RS-wgmma transposed-A register layout, and
whether a **warp-shuffle transpose** (in-register, cheap) can produce it from the softmax
fragment — replacing the swizzled-smem round-trip. Note dQ = dS·K is *non-transposed* (dS is
Major::K), so **dQ's A could go RS-wgmma first** (easier, isolates the technique) before tackling
the transposed dV/dK.

### ★ Deeper scoping (why RS-wgmma is a ground-up rewrite, not a single V31)
1. **Cross-wg operand flow forces smem.** `dS = P ⊙ (dP − D)` needs P (wg0 softmax) and dP (wg1
   dP-GEMM) — different warpgroups. Registers can't cross warpgroups, so P/dP reach dS via smem
   (`sP`/`sdP`). Any RS scheme must either keep P and dP on the SAME warpgroup (→ single-wg-per-
   tile) or still round-trip smem (no L1TEX win).
2. **Cooperative balance ⊥ RS.** The 20/20 wgmma split that makes us fast *requires* cross-wg smem
   sharing. Single-wg-per-tile (one wg does S+softmax+dP+dS+dV+dK+dQ → all operands in its regs,
   RS-able) is what enables RS — but that's the imbalanced role-split (dead, §2b) or needs 2× smem
   (dead). Mutually exclusive.
3. **Transposed dV/dK.** Even single-wg, dV=Pᵀ·dO / dK=dSᵀ·Q need Pᵀ/dSᵀ as the RS A-fragment.
   `ldmatrix.trans` still reads smem (no win); an in-register warp-shuffle transpose of 64×64 bf16
   may not beat the round-trip. Only dQ=dS·K is non-transposed.
4. **cuDNN's tell: it SPILLS.** `sdpa_bwd` shows cuDNN at 168 regs + ~832 K local requests — it
   accepts spilling cold state to hold the register-heavy single-wg structure. We've guarded
   0-spill throughout. **The honest path to cuDNN's compute-bound regime = single-wg-per-tile +
   RS-wgmma + accept bounded spilling of cold state** — abandoning both the cooperative split and
   the 0-spill invariant.

**Dedicated-session plan:** (a) single-wg-per-tile — each consumer wg owns a *different* kv-tile
(independent, half the grid each), full dv[64]/dk[64], RS-eliminated sP/sDS (RS frees ~24 KB —
that reclaim is what lets the per-wg buffers fit under 232 KB at PD=1/2); (b) validate dV/dK
correctness via `ldmatrix.trans` first, then attempt the shuffle-transpose to drop the smem read;
(c) allow bounded spilling of *cold* state (never the dv/dk accumulators). One path to sub-1.5×,
several days — not tonight's version.

## 3. Persistent kernel (grid 2048 → 132, grid-stride) — MEDIUM reward, MEDIUM effort
cuDNN is persistent (132 = one CTA/SM, grid-stride over the (b,hkv,k_tile) work). Benefits:
amortizes launch/prologue over many tiles, keeps K/V resident, better L2 reuse, and enables the
forward's "run-ahead issuer" pattern. Our grid is 2048 one-shot blocks. Convert to a persistent
grid-stride loop over the 2048 work-units. Lower-risk than #1/#2; likely a few %.

## 4. Audit remaining manual addressing → TMA / uniform datapath — LOW-MEDIUM
- The lagged D-load (`do_dload`) is a manual global load; could it be TMA?
- The `make_desc` wgmma descriptors are rebuilt per-GEMM with cvta+mask+shift; the forward's
  `desc_add_lo` derives sibling descriptors by a compile-time offset (32-bit add, no carry) —
  saving the per-descriptor IMAD.X. Port `desc_add_lo`.
- Promote more address math to the **uniform datapath** (UR registers) — cuDNN leans on it.

## 2b. Role-based consumer specialization — DEAD END (fatal imbalance)
**Scoped and rejected.** The plan was wg0=ALU(softmax+dS), wg1=tensor(dP+dV+dK+dQ) + sole
dv/dk. But counting wgmmas: the 40 wgmma/tile (S8 dP8 dV8 dK8 dQ8) are **balanced 20/20 by the
current column-split**; role-splitting gives wg0=S(8)+ALU=**8 wgmma**, wg1=**32 wgmma** — a 4:1
tensor imbalance, wg0 idles ¾ of wg1's stream. Strictly worse. (And "wg0 pure-ALU" is worse
still — it needs S staged wg1→wg0, the V28-Pdiv +12.6% regression.) **Root cause: the forward's
ping-pong works because its two consumers do *symmetric* work on different tiles; the backward's
phases have wildly unequal tensor loads, so every role split imbalances and every tile split
needs 2× smem. The cooperative column-split is the correct structure.** Superseded original note:

<details><summary>original (flawed) #2b</summary>
The backward already warp-specializes producer/consumer (wg2 TMA, wg0/wg1 consumers, since V11),
and partially inside the consumers (wg0 softmax ∥ wg1 dP-GEMM). The **tile-based** ping-pong
(each consumer a different tile) is smem-dead. But a **role-based** split is different: make
**wg0 the pure ALU engine** (softmax + dS) and **wg1 the pure tensor engine** that takes *sole*
ownership of `dv`/`dk`, then **pipeline across tiles** — wg0 computes tile n's softmax/dS while
wg1 runs tile n−1's S/dP/dV/dK/dQ. Why it's more promising than the tile ping-pong:
- K/V/Q/dO stay **shared** (not doubled); only `sP`/`sDS` need double-buffering (~+32 KB →
  ~236 KB, tight but far under the 256–307 KB the tile-split needed; a small cut like PD=2 on one
  buffer or dropping `sA_t` gets it under 232 KB).
- `dv`/`dk` live only on wg1 (full `dv[64]`/`dk[64]` = 128 regs) — exactly what V30's
  `setmaxnreg`→232 unlocked.
- **Rule-legal:** wg1's persistent `dv`/`dk` group is live across wg0's *ALU* (no intervening
  wgmma group), the one safe overlap pattern (V27's dV∥dS, now cross-warpgroup). This is the
  ALU∥tensor overlap the forward's ping-pong buys, without needing 2× the operand smem.
Risk: the softmax→sP→dV and dS→sDS→dK dependencies still chain within a tile; the win depends on
how much of wg0's ALU fills wg1's tensor waits across the tile boundary (could be partial). But
it's the most promising *specialization* path that respects the smem cap — worth building after
#1/#3 land.

</details>

## 5. Larger tile Br=128 — RESEARCH feasibility
Fewer, larger tiles amortize per-tile overhead (barriers, index math, softmax setup). Needs
`m64n128`/register budget check; the forward runs BLOCK_N=128. Uncertain vs the register wall.

## Dead ends (measured, don't revisit)
- **2 blocks/SM** — cuDNN is 1 block/SM too; occupancy is not the edge.
- **FA3 ping-pong as-is** — 2× staged smem > 232 KB cap (needs #2 RS-wgmma first to free smem).
- **Cross-tile tensor pipeline** — blocked by the persistent-accumulator wgmma rule.
- **FFMA fusion** — `--fmad=true` already default; the 104 M "non-fused" ops are non-contractable.
- **Shared-load conflict reduction** — the conflicts are inherent HGMMA operand fetches (0 excess).

## Suggested order for tomorrow
**#3 persistent** (lowest risk, sets up the structure) → **#1 TMA-store dV/dK** (biggest clean
instruction cut, frees smem) → **#2 RS-wgmma starting with dQ** (the structural prize; if it
lands, the ping-pong + full RS become reachable). Each is measured against the same H200
`check()` + bit-identical bar.
