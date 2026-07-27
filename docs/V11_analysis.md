# V11 Analysis — 3-warpgroup producer/consumer (occupancy 18.75%) (Nsight Compute, D=128)

**V11 = V10 + a 3rd warpgroup as a pure TMA-producer** (cuDNN FA3 warp-specialization; full/empty
mbarrier protocol). Correct (bit-identical to V1–V10, validated at full speed on H100), **12.97 ms
/ 254.31 TFLOP-s**, 0 spills, 168 regs. Occupancy **12.5% → 18.75%** (achieved 18.59%). An ~8% win
over V10 — and the profile shows *why it's "only" 8%*: **occupancy has stopped being the
bottleneck**. This is a strategic inflection.

- **Kernel:** `gqa_backward_v11_kv<64,64,128>` — grid `(8,4,64)=2048`, block **384** (3 warpgroups: 2 consumers + 1 producer).
- **Source:** `reports/gqa_v11_profile_d128.{ncu-rep,txt}` (`ncu --set full`, 3 launches, <0.1% spread). H100 SXM (132 SM, ≡ H200 SXM here), SM 1.43 GHz → 16.50 ms base (wall 12.97). Read %s.
- Compare: `V10_analysis.md`, `gqa_bwd_py_analysis.md` (cuDNN 2.78–2.96 ms, 15.6% occ, 59% compute).

---

## 0. Headline — occupancy is solved (we now *exceed* cuDNN); the wall is memory latency

> V11 hits **18.75% occupancy (achieved 18.59%, 2.97 active warps/scheduler)** — **higher than
> cuDNN's 15.6%** — yet we're still 4.66× slower. So more warps is no longer the answer. The
> tell-tale: occupancy went up 1.5× but **no-eligible barely moved (61.9% → 59.9%)** and
> **warp-cycles-per-issued rose 5.25 → 7.42**. The extra warps now *contend*, and the dominant
> stall **shifted from "fixed-latency dependency" (V10) to "long-scoreboard / L1TEX" (V11, 32.7%
> of 7.4 cyc)** — i.e. warps stall waiting on **memory** (shared + local), not on occupancy.

The producer/consumer protocol landed **correct on the first full-speed H100 run** — the highest-risk
version in the project, validated. But it also revealed that our own **shared-staging + local
traffic** (added in V10's coalescing) is now the ceiling.

---

## 1. V10 → V11 delta
| Metric | V10 | V11 | Δ |
|--------|-----|-----|---|
| **Wall-clock** | 14.08 ms | **12.97 ms** | **−8%** |
| TFLOP/s | 234.2 | **254.3** | +9% |
| **Occupancy (achieved)** | 12.5% | **18.59%** | +49% (2.97 warps/sched) |
| Compute SOL | 38.0% | 40.0% | +2 |
| Memory SOL | 46.2% | 50.8% | +4.6 |
| IPC | 1.52 | 1.60 | +5% |
| **No-eligible** | 61.9% | **59.9%** | **−2 only** ← occupancy barely helped eligibility |
| **Warp cycles / issued** | 5.25 | **7.42** | **+41%** ← contention rose |
| Top warp stall | fixed-latency dep | **long-scoreboard / L1TEX (32.7%)** | shifted to memory |
| Regs / spills / smem | 127 / 0 / 222,744 | 168 / 0 / 222,760 | — |

The small wall-clock gain for a large occupancy gain is the signature of **crossing out of the
occupancy-bound regime**. We spent our last occupancy lever and got diminishing returns — exactly
the point at which to stop chasing warps.

## 2. Full metrics (launch 1; 3 launches <0.1%)
| Section | Value |
|---------|-------|
| Duration | 16.50 ms base (SM 1.43 GHz) / 12.97 ms wall |
| Compute SOL | 40.0% (ALU top pipe 28.9%) |
| Memory SOL | 50.8% (L1/TEX 50.9%) |
| DRAM | 11.0% (not BW-bound) |
| L1 / L2 hit | 8.4% / 74.6% |
| **Dominant stall** | **long-scoreboard (L1TEX: local/global/shared) — 2.4 of 7.4 cyc, 32.7%** |
| **Uncoalesced local ld/st** | **1/32 bytes — est. 49.2% (now the TOP table lever; 0 spills)** |
| Uncoalesced shared (aggregate) | 540 M excessive / 1.40 B (39%) — est. 38.6% |
| Shared-STORE conflict | 3.8-way, 66.4% of 0.83 B — est. 33.8% (grew from 3.6) |
| No-eligible | 59.9% (2.97 active warp/sched, 0.52 eligible) — issues every 2.5 cyc |
| **Occupancy** | **18.59% / 18.75% theo** — now limited by **BOTH** registers (Block Limit Reg = 1) **and** smem (= 1). Maxed for this footprint. |
| Launch | block 384, grid 2048, 168 regs, 0 spills, 222.76 KB smem |

**Occupancy is maxed for this footprint.** A 4th warpgroup (512 thr) fails: 512 × 168 = 86,016 >
65,536 registers, and smem still pins 1 block. So 18.75% is the ceiling unless the smem/register
footprint shrinks. And since we already beat cuDNN's 15.6%, there's no reason to push further here.

## 3. Ladder + cuDNN
| | V9 | V10 | V11 | cuDNN |
|--|-----|-----|-----|-------|
| Wall bwd | 15.40 | 14.08 | **12.97 ms** | 2.78–2.96 ms |
| TFLOP/s | 214 | 234 | **254.3** | 1114–1187 |
| Compute SOL | 31.3% | 38.0% | **40.0%** | 59% |
| Occupancy | 12.5% | 12.5% | **18.75%** | **15.6%** |
| Spills | 0 | 0 | **0** | 1.04 M |

Gap to cuDNN: ~5.05× (V10) → **~4.66× (V11)**. **We now exceed cuDNN on occupancy (18.75% vs 15.6%)
but trail badly on compute-SOL (40% vs 59%).** That is the whole story of the remaining gap: cuDNN
keeps its tensor pipe fed with *fewer* warps via deeper async overlap and *less* shared/local
overhead — we stall on L1TEX. More warps won't close it; **less memory traffic per GEMM will.**

## 4. V12 priorities (measured) — the regime has changed to memory-latency

Occupancy is done. The dominant stall is now **long-scoreboard/L1TEX** (32.7%), fed by:

1. **Uncoalesced "local" traffic (est. 49.2% — the top table lever now, and it GREW V10→V11).**
   1/32 bytes/sector, yet **0 register spills** — so this is dynamically-indexed local/stack arrays
   (candidates: the `dv[32]`/`dk[32]`/`acc[32]` fragments, or a staging temp indexed by a runtime
   value). **First V12 action: pull the Source Counters / SASS to find the exact lines**, then force
   compile-time indexing (`#pragma unroll` + constant indices) so they stay in registers. Potentially
   large and mechanical.
2. **Shared-traffic + conflicts (uncoalesced-shared est. 38.6%, shared-STORE 3.8-way est. 33.8%).**
   This is largely the staging traffic V10 added for coalescing (accepted then) plus the `sA_t`/
   `fill_*` reshuffles — now the second-order cost. Reduce staging round-trips (fuse the stage with
   the consuming read, or use `stmatrix`/`ldmatrix` to move fragments without the smem scatter).
3. **Deeper pipelining (>2 buffer stages).** The producer runs at most 2 tiles ahead; a 3–4 stage
   circular buffer would let it hide more of the L1TEX latency the consumers now stall on. Costs smem
   (currently maxed) — would require shrinking a buffer first, so pairs naturally with (2).
4. FFMA fusion (~5%), L2 compression (~8%) — ignore.

**Recommended V12 = kill the "local" traffic** (est. 49.2%, mechanical, lowest-risk, and directly
attacks the new #1 stall). Pull the SASS/source-counters to localize it first. Then V13 = reduce
shared-staging round-trips (`stmatrix`/fused stage). Deeper pipelining is a V14 once smem is freed.

> **The framing for everything past here:** we are no longer occupancy- or conflict-bound — we are
> **memory-latency bound**, and the job is to move *fewer bytes through L1/shared/local per GEMM*,
> which is exactly how cuDNN feeds its pipe to 59% at *lower* occupancy than ours.
