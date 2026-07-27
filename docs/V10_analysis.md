# V10 Analysis — coalesced global writeback (Nsight Compute, D=128)

**V10 = V9 + smem-staged, address-contiguous global stores** for dK/dV (uint4 vector stores) and dQ
(contiguous fp32 atomics), reusing dead smem buffers (zero net smem). Correct (bit-identical to
V1–V9), **14.07 ms / 234.51 TFLOP-s**, 0 spills, **127 regs**. A ~9% win over V9 (15.40 ms). This
profile confirms the writeback inefficiency is **gone** and re-confirms occupancy as the ceiling.

- **Kernel:** `gqa_backward_v10_kv<64,64,128>` — grid `(8,4,64)=2048`, block 256 (2 warpgroups).
- **Source:** `reports/gqa_v10_profile_d128.{ncu-rep,txt}` (`ncu --set full`, 3 launches, <0.1% spread). ncu SM 1.43 GHz (DRAM 2.62 GHz — different clock state this run) → 17.99 ms base (wall 14.07). Read %s, compare deltas not absolute base-ms.
- Compare: `V9_analysis.md`, `gqa_bwd_py_analysis.md` (cuDNN 2.78–2.96 ms).

---

## 0. Headline — the writeback inefficiency is eliminated

> V9's two flagged global-writeback problems — **uncoalesced global stores (8/32 bytes, est. 33%)**
> and **50% excessive global sectors (est. 50%)** — are **both gone from V10's tables**. The
> smem-staged coalescing landed the stores at full-sector width. Result: **Compute SOL 31.3% →
> 38.0%**, **IPC 1.25 → 1.52**, **no-eligible 68.7% → 61.9%**, **DRAM 6.0% → 10.2%** (efficient
> coalesced bursts). 0 spills, occupancy unchanged 12.5%.

The trade V10 made: it removed the expensive global scatter/atomics at the cost of **added
shared-staging traffic** (the register→smem→global reshuffle). Net positive by ~9%, because
scattered global atomics are far costlier than the extra shared ops.

---

## 1. V9 → V10 delta
| Metric | V9 | V10 | Δ |
|--------|-----|-----|---|
| **Wall-clock** | 15.40 ms | **14.07 ms** | **−9%** |
| TFLOP/s | 214.2 | **234.5** | +9% |
| **Uncoalesced global stores** | 8/32 B, est. 33% | **gone (not flagged)** | eliminated |
| **Excessive global sectors** | 50%, est. 50% | **gone (not flagged)** | eliminated |
| Compute SOL | 31.3% | **38.0%** | +21% rel |
| Memory SOL | 44.4% | 46.2% | +1.8 |
| IPC | 1.25 | **1.52** | +22% |
| No-eligible | 68.7% | **61.9%** | −7 |
| DRAM throughput | 6.0% | **10.2%** | coalesced bursts |
| Warp cycles / issued | 6.38 | **5.25** | −18% |
| Registers / spills | 166 / 0 | **127 / 0** | −39 regs |
| Occupancy / smem | 12.5% / 222,744 B | 12.5% / 222,744 B | unchanged |

**Cost of the trade (accepted):** total shared wavefronts 1.14 B → **1.40 B (+23%)** and shared-STORE
conflict 3.0-way → **3.6-way** — the register→smem staging adds shared stores. This is why
"uncoalesced shared" now shows est. 38.6% (was 34%). It's the price of the global win and net-positive.

## 2. Full metrics (launch 1; 3 launches <0.1%)
| Section | Value |
|---------|-------|
| Duration | 17.99 ms base (SM 1.43 GHz) / 14.07 ms wall |
| Compute SOL | 38.0% (ALU highest pipe 25.9%; still under-utilized) |
| Memory SOL | 46.2% (L1/TEX 46.3%) |
| DRAM | 10.2% (up — writes now efficient) |
| L1 / L2 hit | 5.8% / 74.1% |
| **Global stores** | **not flagged** (coalescing succeeded) |
| Shared-STORE conflict | 3.6-way, 68.2% of 0.79 B — est. 31.6% (staging overhead, grew) |
| Uncoalesced shared (aggregate) | 540 M excessive / 1.40 B (39%) — est. 38.6% |
| Uncoalesced local ld/st | 1/32 bytes — est. 44.7% (0 spills; dynamically-indexed reg arrays → stack) |
| **No-eligible** | **61.9%** (2.00 active warp/sched, 0.45 eligible) — issues every 2.6 cyc |
| Fixed-latency-dep stall | 30.6% of 5.3 warp-cycles ("shows up only in already-optimized kernels") |
| **Occupancy** | **12.5%** (8 warps/SM) — smem-limited; Block Limit Reg now **2** (127 regs), Smem **1** |
| Launch | block 256, grid 2048, 127 regs, 0 spills, 222.7 KB smem |

Note: `Block Limit Registers` is back to **2** at 127 regs — so registers no longer co-limit; smem
alone pins us to 1 block. And the top warp-stall is now "fixed-latency execution dependency," which
ncu notes typically appears only in **already-optimized** kernels — a sign we've entered that regime.

## 3. Ladder + cuDNN
| | V8 | V9 | V10 | cuDNN |
|--|-----|-----|-----|-------|
| Wall bwd | 20.5 | 15.40 | **14.07 ms** | 2.78–2.96 ms |
| TFLOP/s | 161 | 214 | **234.5** | 1114–1187 |
| Compute SOL | 22.5% | 31.3% | **38.0%** | 59% |
| Occupancy | 6.25% | 12.5% | **12.5%** | 15.6% |
| Spills | 0 | 0 | **0** | 1.04 M |

Gap to cuDNN: ~5.7× (V9) → **~5.05× (V10)**. We're now at **64% of cuDNN's compute-SOL** (38% vs 59%)
and **80% of its occupancy** (12.5% vs 15.6%), still 0-spill. The remaining gap is still the same
axis: more warps in flight to hide latency.

## 4. V11 priorities (measured)

The bottleneck ranking is unchanged and clear:

1. **OCCUPANCY → 18.75% (3rd warpgroup; est. 53.8%, still #1 by ~1.4×).** 12.5% = 8 warps, smem-pinned
   to 1 block. cuDNN's exact shape is **384 threads = 3 warpgroups → 18.75%**. **Now fits comfortably
   by registers** (384 × 127 = 48,768 < 65,536; at V9's 166 it was marginal — the V10 register drop
   unlocked this). Catch unchanged: M=64 is wgmma-native, so the 3rd WG can't take an equal MMA share
   — it must be a **pure TMA-producer** (producer/consumer warp specialization, mbarrier handoff).
   Highest-risk version yet, but the top lever and the direct path to cuDNN's design.
2. **Uncoalesced shared (est. 38.6%) + shared-STORE conflict 3.6-way (est. 31.6%).** Partly V10's own
   staging traffic (accepted), partly the residual `sA_t`/`fill_*` patterns. Some is intrinsic to the
   fused layout; revisit after occupancy.
3. **Uncoalesced "local" (est. 44.7%, 0 spills).** Dynamically-indexed `dv[32]`/`dk[32]`/`acc[32]`
   register arrays touching stack (1/32 bytes). Worth a look — forcing compile-time indexing or
   restructuring could remove it — but secondary to occupancy.
4. FFMA fusion (~4%), L2 compression (~7%) — ignore.

**Recommendation: V11 = 3rd warpgroup (occupancy → 18.75%).** It's still the largest lever (est.
53.8%), the register headroom now makes it feasible, and it's the last structural step toward
cuDNN's 15.6%-occupancy / 3-warpgroup design. It is the highest-risk version (pure-producer warp
specialization + mbarrier handoff), so it's the right one to hand back to the cuda-agent rather than
the main thread. The shared-traffic and local-access cleanups are lower-risk V12 follow-ups.
