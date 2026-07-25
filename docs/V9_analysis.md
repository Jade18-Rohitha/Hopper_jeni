# V9 Analysis — 2-warpgroup work-split (occupancy) (Nsight Compute, D=128)

**V9 = V8 + a second warpgroup (128→256 threads), same smem.** WG0/WG1 split the independent work
(S∥dP concurrent, dV/dK by output-column-half, dQ split, D-rowsum over 8 warps). Correct
(bit-identical to V1–V8), **15.84 ms / 208.17 TFLOP-s**, 0 spills. A **25% wall-clock win** over V8
(21.16 ms) from doubling occupancy **6.25% → 12.5%** with the pipeline intact.

- **Kernel:** `gqa_backward_v9_kv<64,64,128>` — grid `(8,4,64)=2048`, block **256** (2 warpgroups).
- **Source:** `reports/gqa_v9_profile_d128.{ncu-rep,txt}` (`ncu --set full`, 3 launches, <0.1% spread). ncu base 1.36 GHz → 20.71 ms (wall 15.84). Read %s.
- Compare: `V8_analysis.md`, `gqa_bwd_py_analysis.md` (cuDNN 2.78–2.96 ms, 15.6% occ).

---

## 0. Headline — occupancy doubled, and it fed everything

> The 2nd warpgroup took occupancy to the cuDNN-adjacent range and lifted every latency metric:
> **occupancy 6.25%→12.5%** (2.00 active warps/scheduler, was 1.00), **IPC 0.90→1.25**, **Compute
> SOL 22.5%→31.3%**, **no-eligible 77.4%→68.7%**. Registers even **dropped 254→166** (each WG holds
> `dv[32]/dk[32]` not `[64]`), and dV/dK now write **disjoint column-halves → no atomics**. Still 0
> spills, smem byte-identical.

Route B (more warps/block, same smem) was the right call over Route A (2 blocks via smem cut):
same 12.5% target, pipeline preserved, every V8 GEMM primitive reused — and it landed correct on
the first H200 run.

---

## 1. V8 → V9 delta
| Metric | V8 | V9 | Δ |
|--------|-----|-----|---|
| **Wall-clock** | 21.16 ms | **15.84 ms** | **−25%** |
| TFLOP/s | 155.7 | **208.2** | +34% |
| **Occupancy** | 6.25% | **12.5%** | **2×** |
| Active warps / scheduler | 1.00 | **2.00** | 2× |
| Compute SOL | 22.5% | **31.3%** | +39% |
| Memory SOL | 33.4% | **44.4%** | +33% |
| IPC | 0.90 | **1.25** | +39% |
| No-eligible | 77.4% | **68.7%** | −9 |
| Warp cycles / issued | 4.43 | 6.38 | +44% (more warps contending) |
| Threads / block | 128 | **256** | 2 warpgroups |
| Regs / spills | 254 / 0 | **166 / 0** | −88 regs |
| smem | 222,744 B | 222,744 B | — |

The one metric that got "worse" — warp-cycles-per-issued 4.43→6.38 — is expected: 8 warps now
contend for shared/MIO, so each instruction waits a bit longer, but 2 warps/scheduler more than
compensate (net IPC +39%). The S∥dP concurrent issue is visible in the compute-SOL jump.

## 2. Full metrics (launch 1; 3 launches <0.1%)
| Section | Value |
|---------|-------|
| Duration | 20.71 ms base / 15.84 ms wall |
| Compute SOL | 31.3% (all pipelines still under-utilized, est. 80% local) |
| Memory SOL | 44.4% (L1/TEX 44.5%) |
| DRAM | 6.0% (not BW-bound) |
| L1 / L2 hit | 4.7% / 82.8% |
| Shared-STORE conflict | 3.0-way, 66.3% of 0.579 B — est. 29.5% (deferred residual, unchanged) |
| Shared-LOAD conflict | none flagged (still solved from V8) |
| **Uncoalesced global stores** | **8/32 bytes** (dK/dV/dQ writeback) — est. 33.3%; 50% excessive sectors → est. 50.1% |
| Uncoalesced local ld/st | 1/32 bytes — est. 43% (0 spills; dynamically-indexed reg arrays → stack) |
| **No-eligible** | **68.7%** (2.00 active warp/scheduler, 0.37 eligible) — issues every 3.2 cyc |
| **Occupancy** | **12.5%** (8 warps/SM) — now limited by **BOTH** registers (Block Limit Reg = 1) **and** smem (Block Limit Smem = 1) |
| Launch | block 256, grid 2048, 166 regs, 0 spills, 222.7 KB smem |

## 3. Ladder + cuDNN
| | V7 | V8 | V9 | cuDNN |
|--|-----|-----|-----|-------|
| Wall bwd | 33.0 | 21.16 | **15.84 ms** | 2.78–2.96 ms |
| TFLOP/s | 99.96 | 155.7 | **208.2** | 1114–1187 |
| Compute SOL | 13.8% | 22.5% | **31.3%** | 59% |
| Occupancy | 6.25% | 6.25% | **12.5%** | 15.6% |
| Spills | 0 | 0 | **0** | 1.04 M |

Gap to cuDNN: ~7.6× (V8) → **~5.7× (V9)**. We're now at **80% of cuDNN's occupancy** (12.5% vs
15.6%) and **53% of its compute-SOL** (31.3% vs 59%), still 0-spill. The gap is closing on the same
axis cuDNN wins on.

## 4. V10 — the fork (measured)

Two levers now, and they're a genuine trade of value vs risk:

1. **Occupancy → 18.75% (the 3rd warpgroup; est. 55.6%, the top lever).** Occupancy is *still* #1,
   but note it's now pinned by **both** registers and smem (166 regs × 256 thr fills one block on
   both axes). The next step is cuDNN's exact shape: **384 threads = 3 warpgroups → 18.75%**. It
   *fits by registers* (384 × 166 = 63,744 < 65,536, barely) with smem unchanged. **Catch:** M=64
   is wgmma-native, so the 3rd WG can't take an equal MMA share of the same M-tile — it has to be a
   **pure TMA-producer** (producer/consumer warp specialization, mbarrier handoff). Higher
   correctness risk than V9's symmetric split, and register headroom is thin (may push spills).
2. **Coalesce the global writeback (est. 33–50%; lower risk, mechanical).** dK/dV/dQ stores use
   **8/32 bytes/sector**; 50% of global sectors are excessive. Stage each accumulator to a small
   smem scratch and emit contiguous 128-bit vector stores (or `stmatrix`), and reindex the dQ
   atomic flush. No barrier/specialization risk. Real, bounded win that's independent of occupancy.
3. Uncoalesced "local" (est. 43%, 0 spills — dynamically-indexed `dv[]/dk[]/acc[]` reg arrays
   touching stack), shared-store 3.0-way (est. 29.5%, deferred), FFMA (~3.6%) — secondary.

**Recommendation:** the highest *expected* value is the 3rd warpgroup (occupancy 55.6% est, and
it's the direct path to cuDNN's shape), but it's the **highest-risk** version yet (producer-only
specialization + thin register budget). The **writeback coalescing** is a clean, lower-risk ~33–50%
that stacks with occupancy later. Given V9 already banked the big occupancy jump, **V10 =
writeback coalescing** is the safer next step (lock in a mechanical win, keep 0 spills), with the
**3rd-warpgroup producer specialization as V11** — the bigger, riskier reach toward cuDNN's 15.6%.
