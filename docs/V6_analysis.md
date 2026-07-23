# V6 Analysis — Padded (bank-conflict) fused wgmma GQA Backward (Nsight Compute, D=128)

**V6 = V5 + shared-memory padding.** Correct (bit-identical to V1–V5), 64.2 ms boost wall /
51.35 TFLOP-s, 0 spills. This profile answers the question after the padding: *did the 17.9-way
bank conflict die?* **Answer: it dropped, but only to 15.1-way — bank conflicts are NOT solved.**
This confirms the Blackwell experience (bank conflicts took V6–V10 there); V6 is the opening move.

- **Kernel:** `gqa_backward_v6_kv<64,64,128>` — grid `(8,4,64)=2048`, block 128 (1 warpgroup).
- **Source:** `reports/gqa_v6_profile_d128.{ncu-rep,txt}` (`ncu --set full`, 3 launches, <0.1% spread). ncu base clock 1.28 GHz → 84.6 ms (wall 64.2 ms). Read the %s.
- Compare: `V5_analysis.md` (pre-padding) and `gqa_bwd_py_analysis.md` (cuDNN 2.96 ms).

---

## 0. Headline — padding helped modestly, bank conflicts survive

> Shared-load bank conflict **17.9-way → 15.1-way** (still **93.4% of wavefronts**, ncu est.
> **70.1%** remaining). Aggregate uncoalesced-shared 90% → 88% excessive wavefronts. Wall 68.8
> → 64.2 ms (~7%). Compute SOL 13.5% → **15.4%**, Memory SOL 79.9% → 74.9%. Occupancy
> **unchanged at 6.25%**. 0 spills.

Two ceilings remain, both large: **(1) residual bank conflicts** that padding *cannot* reach,
and **(2) occupancy** (1 warp/scheduler) that padding never touched.

---

## 1. V5 → V6 delta (what the padding bought)
| Metric | V5 | V6 | Δ |
|--------|-----|-----|---|
| Shared-LOAD conflict | 17.9-way / 94.4% | **15.1-way / 93.4%** | −2.8-way; still 93% |
| Shared-load conflicts (count) | 9.51 B | **7.92 B** | −17% |
| Shared-STORE conflict | 3.9-way / 72.6% | 3.6-way / 72.6% | ~flat |
| Aggregate uncoalesced-shared | 90% excessive | 88% excessive | −2 pts |
| Compute SOL | 13.5% | **15.4%** | +1.9 |
| Memory SOL | 79.9% | 74.9% | −5.0 |
| No-eligible | 86.5% | 84.5% | −2 |
| IPC | 0.54 | 0.62 | +0.08 |
| Wall-clock | 68.8 ms | **64.2 ms** | −7% |
| Occupancy | 6.25% | **6.25%** | 0 (unchanged) |
| Regs / smem / spills | 254 / 221.7 KB / 0 | 250 / 223 KB / 0 | — |

The padding did exactly what it could — it removed the *paddable* aliasing (SBO, `sS/sdP/sP`),
shaving 17% off the conflict count and lifting compute 13.5→15.4%. But 93% of shared-load
wavefronts still conflict at 15.1-way, so the dominant conflict source is elsewhere.

## 2. Where the residual 15.1-way conflict lives
The remaining conflicts are the ones the agent flagged as **unpaddable**: the **transposed
`fill_trans` reads of the TMA-plain buffers** (`sQ_pl` / `sdO_pl` / `sK_pl`) and the D-rowsum
reads of `sdO_pl`/`sO_pl`. TMA writes those buffers **contiguously** (no destination-stride
control), so their layout can't be padded — and the transposed read pattern (`src[c*64+r]`,
stride 64 bf16 = one 128 B bank line) collides. **Padding is out of levers here; this needs
access-pattern restructuring** (V7+, per the Blackwell V6–V10 playbook — likely: load these into
a *separately-laid-out padded/swizzled staging* before the transposed read, or restructure the
transposed GEMM to avoid the strided shared read).

## 3. Full metrics (launch 1; all 3 within <0.1%)
| Section | Value |
|---------|-------|
| Duration | 84.6 ms (base 1.28 GHz) / 64.2 ms wall |
| Compute SOL | 15.4% (all pipelines under-utilized, est. 86% local) |
| Memory SOL | 74.9% (L1/TEX 75.1%) |
| DRAM | 1.47% (not BW-bound) |
| L1 hit / L2 hit | 3.3% / 81.7% |
| **Shared-load conflict** | **15.1-way, 93.4% of 8.49 B wavefronts** (7.92 B conflicts) — est. 70.1% |
| Shared-store conflict | 3.6-way, 72.6% (1.15 B) — est. 54.5% |
| Uncoalesced local ld/st | 1/32 bytes — est. 72.5% |
| Uncoalesced global stores | 8/32 bytes (dK/dV/dQ writeback) — est. 56% |
| No-eligible | 84.5% (1.00 active warp/scheduler, 0.15 eligible) |
| Dominant stall | MIO short-scoreboard **43.9%** (shared-mem / bank conflicts) |
| Warp cycles/issued | 6.46 |
| **Occupancy** | **6.25%** (4 warps/SM) — **smem-limited** (223 KB → 1 block/SM); registers (250) allow 2 |
| Launch | block 128, grid 2048, 250 regs, 0 spills, 223 KB smem, waves/SM 15.5 |
| Branch eff | 99.7% (1107 divergent, 0.03% ratio) |

## 4. V5/V6 vs cuDNN (D=128)
| | V5 | V6 | cuDNN |
|--|-----|-----|-------|
| Wall bwd | 68.8 ms | 64.2 ms | **2.96 ms** |
| Compute SOL | 13.5% | 15.4% | 59% |
| Shared-load conflict | 17.9-way | 15.1-way | ~2.2-way |
| Occupancy | 6.25% | 6.25% | 15.6% |
| Spills | 0 | 0 | 1.04 M |

We remain ~22× off. The gap is still concentrated in (a) shared-memory conflicts (cuDNN ~2.2-way
vs our 15.1-way) and (b) occupancy (cuDNN 15.6% vs our 6.25%). We keep the spill-free edge.

---

## 5. V7 priorities (measured)

1. **Restructure the transposed reads of the TMA-plain buffers (est. ~70% still on shared loads).**
   This is the surviving 15.1-way conflict and padding can't touch it. Port the Blackwell V6–V10
   technique (they fought this same residual across several versions) — e.g. stage `sQ_pl`/`sdO_pl`/
   `sK_pl` through a padded/swizzled buffer before the `fill_trans` transposed read, or change the
   transposed GEMM so the shared read is contiguous. **The single biggest remaining lever.**
2. **Occupancy (est. ~25% + unlocks latency hiding).** Still 6.25% (1 warp/scheduler), smem-limited
   at 223 KB → 1 block/SM; registers (250) already permit 2. Reclaim smem (drop the `sK_sw`+`sK_pl`
   duplication; tighter staging) to fit ≥2 blocks/SM → more warps to hide the residual latency.
3. **Coalesce the dK/dV/dQ global writeback (est. 43–56%, 8/32 bytes).** Secondary.

**Read for V7:** the Blackwell `V*_profile_analysis.md` (V6–V10) + `GQA_sm103_bwd.cu` V6–V10 —
they solved this exact residual-conflict grind; port, don't reinvent. Bank conflicts are a
multi-version campaign here as they were there; V6 is step one.
