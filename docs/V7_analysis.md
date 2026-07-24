# V7 Analysis — Swizzled-TMA + wgmma Major::MN transpose-free (Nsight Compute, D=128)

**V7 = V6 + swizzled-TMA loads feeding the transposed GEMMs via wgmma `Major::MN=1`**, deleting
the `fill_trans` transposed reads (and the plain-buffer duplication) entirely. Correct
(bit-identical to V1–V6), **33.0 ms boost / 99.96 TFLOP-s**, 0 spills. This is the biggest single
step so far — ~48% over V6. This profile shows *why* (the shared traffic collapsed) and *what's
left* (occupancy is now the dominant ceiling; bank conflicts reduced but not gone).

- **Kernel:** `gqa_backward_v7_kv<64,64,128>` — grid `(8,4,64)=2048`, block 128 (1 warpgroup).
- **Source:** `reports/gqa_v7_profile_d128.{ncu-rep,txt}` (`ncu --set full`, 3 launches, <0.1% spread). ncu base 1.31 GHz → 43.6 ms (wall 33.0). Read %s.
- Compare: `V6_analysis.md`, `V5_analysis.md`, `gqa_bwd_py_analysis.md` (cuDNN 2.96 ms).

---

## 0. Headline — Major::MN worked; shared traffic collapsed; occupancy is now the wall

> Deleting the `fill_trans` reads cut **total shared-load wavefronts 8.49 B → 3.48 B (−59%)** and
> the conflict count **7.92 B → 3.17 B (−60%)**; way-ness dropped 15.1 → **11.3-way**. That drove
> the ~48% wall-clock win. **But occupancy is unchanged at 6.25% (1 block/SM = 1 warp/scheduler)**,
> compute SOL is still **13.8%** (tensor cores idle), and 86% of cycles issue nothing. The
> `Major::MN=1` transpose (avoided since V3) landed correct first try. 0 spills.

Two things remain: **(1) occupancy** — now the dominant structural ceiling, untouched by V7 — and
**(2) residual bank conflicts** (11.3-way, the multi-version grind continues), plus an emerging
**uncoalesced global writeback**.

---

## 1. V6 → V7 delta (what the Major::MN transpose bought)
| Metric | V6 | V7 | Δ |
|--------|-----|-----|---|
| **Wall-clock** | 64.2 ms | **33.0 ms** | **−49%** |
| TFLOP/s | 51.7 | **99.96** | ~2× |
| Total shared-load wavefronts | 8.49 B | **3.48 B** | **−59%** |
| Shared-load conflicts (count) | 7.92 B | **3.17 B** | **−60%** |
| Shared-load conflict way-ness | 15.1-way / 93% | **11.3-way / 91%** | −3.8-way |
| Aggregate uncoalesced-shared excess | 9.07 B (88%) | **3.55 B (83%)** | −61% |
| Memory SOL | 74.9% | 63.0% | −12 |
| Compute SOL | 15.4% | 13.8% | ~flat (still starved) |
| No-eligible | 84.5% | 86.1% | ~flat |
| Occupancy | 6.25% | **6.25%** | 0 (unchanged) |
| Regs / smem / spills | 250 / 223 KB / 0 | 254 / 222.7 KB / 0 | — |

The win is almost entirely **fewer shared accesses** (the transposed reads are gone), not lower
way-ness — the count dropped 60%. Occupancy was explicitly *not* addressed (the `dO` double-load
offset the deleted buffers), so the latency-hiding ceiling is untouched.

## 2. Full metrics (launch 1; 3 launches <0.1%)
| Section | Value |
|---------|-------|
| Duration | 43.6 ms base / 33.0 ms wall |
| Compute SOL | 13.8% (all pipelines under-utilized, est. 90% local) |
| Memory SOL | 63.0% (L1/TEX 63.1%) |
| DRAM | 2.84% (not BW-bound) |
| L1 / L2 hit | 3.3% / 83.0% |
| **Shared-load conflict** | **11.3-way, 91.2% of 3.48 B wavefronts** (3.17 B) — est. 57.6% |
| Shared-store conflict | 3.1-way, 68.1% (0.38 B) — est. 43.0% |
| **Uncoalesced global stores** | **8/32 bytes** (dK/dV/dQ writeback) — est. 47%; 50% excessive sectors → est. 49% |
| Uncoalesced local ld/st | 1/32 bytes — est. 61% |
| **No-eligible** | **86.1%** (1.00 active warp/scheduler, 0.14 eligible) |
| Warp cycles/issued | 7.21 |
| **Occupancy** | **6.25%** (4 warps/SM) — **smem-limited** (222.7 KB → 1 block/SM); regs (254) allow 2 |
| Launch | block 128, grid 2048, 254 regs, 0 spills, 222.7 KB smem |

## 3. Where V7 stands vs the ladder + cuDNN
| | V5 | V6 | V7 | cuDNN |
|--|-----|-----|-----|-------|
| Wall bwd | 68.8 | 64.2 | **33.0 ms** | 2.96 ms |
| TFLOP/s | 48.1 | 51.7 | **99.96** | 1114 |
| Shared-load conflict | 17.9-way | 15.1-way | **11.3-way** | ~2.2-way |
| Compute SOL | 13.5% | 15.4% | 13.8% | 59% |
| Occupancy | 6.25% | 6.25% | **6.25%** | 15.6% |
| Spills | 0 | 0 | **0** | 1.04 M |

Gap to cuDNN: ~22× (V5) → **~11× (V7)**. Halved in one version. We hold the spill-free edge and
now match on primitive (wgmma + swizzled TMA + Major::MN); the deltas left are **occupancy**
(6.25% vs 15.6%) and **compute-feeding** (13.8% vs 59%), which are the same problem: 1 warp/scheduler.

## 4. V8 priorities (measured)

The bottleneck has shifted. Bank conflicts are much reduced (still an 11.3-way residual worth
chasing), but **occupancy is now the structural ceiling** — at 1 warp/scheduler there is nothing
to hide *any* latency behind, which is why compute SOL is stuck at 13.8% and 86% of cycles issue
nothing.

1. **OCCUPANCY — the dominant lever now (est. ~37% + unlocks latency hiding for everything).**
   6.25% = 1 block/SM, smem-limited at 222.7 KB; registers (254) already allow 2 blocks. Get to
   ≥2 blocks/SM by cutting smem below ~113 KB/block: reduce the double-buffer depth, and/or the
   `dO` plain+swizzled double-presence (the ~128 KB of double-buffered operands is the pin, not
   the transpose buffers V7 removed). This is the FLOP-3-style structural change; it's what
   separates us from cuDNN's 15.6%.
2. **Residual bank conflicts (11.3-way, est. 57.6%)** — the multi-version grind continues (as on
   Blackwell). The surviving conflict is now the in-kernel transposed intermediates (`fill_trans_A`
   of Pᵀ/dSᵀ, no-swizzle Major::MN staging) + the `sS/sdP` readout. Next conflict-reduction target.
3. **Uncoalesced global writeback (est. 47–49%).** dK/dV/dQ stores use 8/32 bytes/sector; the dQ
   atomic + strided flush. Reindex for contiguous 128-bit stores.
4. FFMA fusion (~1.7%), L2 compression (~2%) — ignore.

**Recommended V8:** attack **occupancy** — it's the structural wall (1 warp/scheduler gates
latency hiding), and unlike the conflict grind it hasn't been touched since V3. Cut smem
(double-buffer depth / `dO` double-presence) to land ≥2 blocks/SM. Keep the residual-conflict and
writeback-coalescing work as V9+. (If smem can't be cut enough for 2 blocks without hurting the
TMA pipeline, fall back to continuing the conflict grind per the Blackwell V8–V10 playbook.)
