# V15 Feasibility: 128×128 Retile of the Hopper (sm_90a) GQA-backward Kernel

**Scope:** analysis + resource modeling only. No kernel built (numbers fail decisively).
Baseline = `gqa_backward_v13_kv` (`Hopper/src/attention/GQA_bwd.cu:4619`): Br=Bc=64, D=128,
384 thr (wg0/wg1 consumers + wg2 TMA producer), 154 regs, 0 spill, 223,800 B smem, 18.75% occ, 1 block/SM.
V13 = 8.82 ms / 374 TFLOP/s / 3.14× off cuDNN (2.81 ms).

Model self-check: my line-by-line V13 smem model reproduces the measured **223,800 B exactly** — so the 128×128 projections below use a validated method.

---

## (0) VERDICT: **NO-GO** — decisive on BOTH register and smem axes, independently.

> A 128×128 tile busts the register file by ~2× (persistent dv/dk = 128 regs + a transient 128×128 accumulator = acc[128], vs a 170 ceiling) **and** busts smem by 1.57–2.42× (549 KB naive, still 357 KB after single-buffering + dropping the D-rowsum plain copies, vs a 227 KB cap). The register bust is the killer and it is **structural: Hopper has no TMEM**, so the 128×128 wgmma accumulator (16,384 fp32 elements) lands directly in RMEM — exactly the cost Blackwell's V16 avoided by holding it in TMEM (448/512 columns). The −35% Blackwell win was TMEM-enabled and bundled with a rewrite; it cannot be reproduced on Hopper. **V13/V14 is the final structural ceiling.**

---

## (1) Register-pressure model (the decisive axis)

The wgmma accumulator on Hopper lives in **registers** (RMEM). Fragment size = (output elements) / (threads over that output). Both the transient S/dP accumulator and the persistent dv/dk accumulators scale with the tile.

### Persistent accumulators (pinned live across the entire K-loop)
`dV = Pᵀ·dO`, output = Bc×D. V13 column-splits the D=128 output across the 2 consumer WGs (WG0 owns cols[0,64), WG1 cols[64,128)):

| | V13 (Bc=64) | 128×128 (Bc=128) |
|---|---|---|
| dV output | 64×128 = 8192 | 128×128 = 16384 |
| per-WG (col-half) | 64×64 = 4096 / 128 thr = **dv[32]** | 128×64 = 8192 / 128 thr = **dv[64]** |
| dv + dk persistent | 32+32 = **64 regs** | 64+64 = **128 regs** |

Persistent pressure **doubles** (Bc doubled; the D column-half width is unchanged). This 128-reg floor is **unavoidable** regardless of how threads are partitioned — the full Bc×D=16384-element output over 128 or 256 threads is 128 or 64 per thread either way, and both dv and dk must stay live across all G×q-tile iterations (KV-centric accumulation).

### Transient S/dP accumulator (live during the S/dP GEMM + downstream elementwise)
`S = Q·Kᵀ`, output = Br×Bc. V9's S∥dP split puts the **full** S on WG0's 128 threads, dP on WG1's:

| | V13 (64×64) | 128×128 |
|---|---|---|
| S output | 64×64 = 4096 | 128×128 = 16384 |
| per WG (128 thr) | **acc[32]** | **acc[128]** |

The transient **quadruples** (32→128) because *both* Br and Bc double. (Contrast the task's "32→64": that was 128×**64**, one dim. 128×128 doubles both → 4×.)

### Total
- **V13:** 64 persistent + ~90 non-accumulator (transient acc[32] + addressing + loop + ldmatrix/stmatrix temps) = **154 regs**, 0 spill.
- **128×128:** 128 persistent (pinned) + acc[128] transient (live simultaneously through the S/dP GEMM and the dS=P⊙(dP−D) readout) + ~90 overhead ≈ **~346 regs peak-live**.

**Verdict:** 346 ≫ 170 ceiling (at 384 thr). Even at ptxas's absolute 255-reg cap it spills heavily — destroying the 0-spill property — and 256 thr × 255 = 65,280 barely fits one block at 12.5% occ *with* spills. The persistent 128 alone leaves only **42 regs** under 170 for the transient acc[128] + all addressing + ldmatrix temps: **categorically impossible.**

Even the leanest variant — cooperative S over all 256 consumer threads (acc[64] instead of acc[128], **sacrificing the S∥dP concurrency**) — gives 128 + 64 + ~70 ≈ **262 regs**, still ~1.5× over the ceiling and spilling.

**Root cause:** no TMEM on Hopper. The 128×128 accumulator (16,384 fp32) has nowhere to live but the register file. Blackwell V16 held it in TMEM (448/512 columns) at **zero RMEM cost** — that is the specific reason the retile worked there and cannot here.

---

## (2) Smem budget (the second, independent bust)

Cap = **232,448 B (227 KB)** usable per block on H100/H200. Buffer scaling: row-dim of Q/dO/O/S/P/dS scales with **Br**; K/V and the S/P/dS *column* dim scale with **Bc**; S/dP/P/dS-scratch scale with **Br·Bc**.

| Buffer | V13 (64×64) | 128×128 | scales with |
|---|---|---|---|
| sK_sw | 16,384 | 32,768 | Bc |
| sV_sw | 16,384 | 32,768 | Bc |
| sQ_sw[2] | 32,768 | 65,536 | Br |
| sdO_sw[2] | 32,768 | 65,536 | Br |
| sdO_pl[2] | 32,768 | 65,536 | Br |
| sO_pl[2] | 32,768 | 65,536 | Br |
| sS (fp32 stage) | 16,640 | 66,048 | Br·Bc |
| sdP (fp32) | 16,640 | 66,048 | Br·Bc |
| sP (bf16) | 9,216 | 34,816 | Br·Bc |
| sLSE | 256 | 512 | Br |
| sD[2] | 512 | 1,024 | Br |
| sA_t (transpose scratch) | 16,640 | 66,048 | Br·Bc |
| mbars | 56 | 56 | — |
| **TOTAL** | **223,800** | **562,232 (549 KB)** | |

- **Naive double-buffered: 549 KB = 2.42× over.**
- Single-buffer the 4 staged tiles (kills TMA prefetch overlap): 421 KB = **1.85× over**.
- Also drop sdO_pl + sO_pl (recompute D-rowsum by deswizzle): 357 KB = **1.57× over**.

**What must give to reach 227 KB:** essentially *everything auxiliary* — delete sA_t (33–66 KB) via layout-aliased transpose, delete the fp32 sS/sdP staging (132 KB combined) via full register-fusion, delete sdO_pl/sO_pl (compute D from swizzled buffers), single-buffer the pipeline. That is a **ground-up rewrite, not a retile**, and even then it is borderline — and it does **nothing** for the register wall in (1).

---

## (3) Does 2× tensor concurrency survive the 128-row M-split?

**Mechanically yes, but it's moot — the register squeeze kills it first.** wgmma M=64 is fixed, so a 128-row S-tile = 2 back-to-back M-groups; these still pipeline on one warpgroup (no serialization penalty per se). The V9 win is WG0-on-S ∥ WG1-on-dP. But sustaining two WGs each carrying acc[128] (S/dP transient) + dv[64]+dk[64] persistent is exactly what busts registers. The only register-survivable variant is **cooperative** S over both WGs — which **un-does the S∥dP split**, forfeiting the 2× tensor-issue concurrency. So concurrency does **not** survive the pressure the retile creates.

---

## (4) Expected ALU/MMA at 128×128 — does the mix gap actually close?

**No, negligibly.** The premise (bigger tile amortizes per-tile bookkeeping → fewer ALU/MMA → narrows the 23-vs-14.7 mix gap) does not hold here:

- Total MMA FLOPs and total elementwise ALU are **tile-size-invariant** (same problem). A 128×128 tile has 4× the elements and 4× the elementwise ALU (exp, dS = P⊙(dP−D), the fp32 dQ scatter) per tile, but there are 4× fewer tiles. The elementwise index math is **per-element**, not per-tile-fixed — it does **not** amortize.
- Only genuinely **per-tile-fixed** overhead amortizes over 4× fewer iterations: loop-counter updates, mbarrier arrive/wait, a handful of once-per-iter base addresses. From V13's static SASS mix (IMAD 113+89+46, LOP3 106, LEA 59, VIADD 77), the fixed fraction is small (~10–15% of address ALU). Amortizing that 4× cuts maybe ~10% of ALU → ALU/MMA ~23 → **~20**, nowhere near cuDNN's 14.7.
- **And it wouldn't convert to wall-clock anyway.** The V13 `-lineinfo` diagnosis established the 33.8% INT-ALU stall is warps **parked on the RAW-serial softmax→dS→readout dependency chain**, not ALU-pipe saturation (de-ALU measured **flat**). A retile does not shorten that chain. cuDNN's 14.7 comes from **TMA hardware address generation** (a different instruction stream), not a bigger tile — you can't reach it by re-tiling, only by changing the instructions.

---

## (5) Occupancy / producer-WG tradeoff — does −35% survive WITHOUT TMEM?

**No.** To fit 128×128 at all you must simultaneously:
- **Drop the producer WG** (fold into consumers) → lose V13's Part-A producer D-rowsum overlap (half of the +21% V13 win) and the async TMA prefetch pipeline.
- **Un-fuse S∥dP** (cooperative S) → lose V9's 2× tensor issue.
- **Drop occupancy** 18.75% → 12.5% or lower, and/or **spill** (lose the 0-spill edge).

Each of these is a *measured-positive* V13 feature being destroyed to make room. Blackwell's −35% leaned on two things Hopper lacks: (a) **TMEM** to hold the big accumulator off-register (Hopper: it hits RMEM, see §1), and (b) it was a **rewrite** bundling layout-aliased transpose-elimination — which on Hopper we already measured **FLAT** in isolation (shared is 0.7% of stalls; the bottleneck is the scalar dep chain). Strip away TMEM and the transpose bonus, force the occupancy/structure losses, and the −35% has no mechanism left. Net effect would almost certainly be **negative**.

---

## (6) Minimal config that could "fit" — and why it's still NO-GO

There is **no CONDITIONAL fit.** The smem could *conceivably* be forced under 227 KB only by a full rewrite (delete sA_t via layout-aliasing, delete fp32 sS/sdP via total register-fusion, delete sdO_pl/sO_pl, single-buffer) — but:

1. That rewrite makes the **register wall worse**, not better: register-fusing S/dP means the 128×128 accumulator stays in RMEM *longer*, and there is no TMEM to offload it. The 128 persistent + acc[64–128] transient floor is untouchable by any smem trick.
2. The only register "fit" (255-cap with spills, cooperative S, dropped producer) sacrifices **0-spill, S∥dP concurrency, the producer D-overlap, and occupancy** — every one a validated V13 win — to chase a −35% that (per §4–5) has no mechanism on Hopper.

So the fit that could exist on the smem axis is destroyed on the register axis, and the win that motivates it doesn't survive the structure loss. This is the same "benefit doesn't survive the register/smem constraint" pattern that already killed 128×64 retile, intra-WG ILP, and ping-pong. **The 170-reg / 227-KB / 1-block-per-SM wall — with no TMEM to relieve the accumulator — is fundamental.**

---

## Bottom line

128×128 retile is **NO-GO** on Hopper. Registers bust ~2× (346 vs 170) because the 128×128 accumulator has no TMEM to live in; smem busts 1.57–2.42×; the amortization win is ~10% of ALU that doesn't touch the dependency-chain bottleneck anyway; and any config that fits destroys S∥dP + the producer overlap + 0-spill + occupancy. This was the last untried structural lever below the 3.14× gap. **V13 (8.82 ms / 374 TFLOP/s / 3.14× off cuDNN / 0-spill) stands as the practical CUDA-C ceiling on Hopper.** The residual gap is cuDNN's TMA-hardware-addressed instruction mix + SASS-level scheduling of the RAW-serial softmax→dS→readout chain — not reachable by retiling, occupancy, or overlap.
