# V13 Analysis — fused-softmax + producer D-rowsum overlap (Nsight Compute, D=128)

**V13 = V12 + two overlaps:** (B) register-fused softmax (wg0 computes P on its wgmma `acc[32]`
fragment, deleting the S→`sS` store + reload + a sync), and (A) the D-rowsum moved onto the idle
producer warpgroup (computed one tile lagged, double-buffered). Correct (bit-identical to V1–V12,
validated full-speed on H200), **8.82 ms / 374.06 TFLOP-s**, 0 spills, **154 regs** (down from 168).
A **~21% win over V12 (11.16 ms)** — the biggest single-version gain yet.

- **Kernel:** `gqa_backward_v13_kv<64,64,128>` — grid `(8,4,64)=2048`, block 384 (3 warpgroups).
- **Source:** `reports/gqa_v13_profile_d128.{ncu-rep,txt}` (`ncu --set full`, 3 launches, <0.1%). Full clock (DRAM 3.20 GHz, SM 1.35). Wall (benchmark): 8.82 ms.
- Compare: `V12_analysis.md`, `gqa_bwd_py_analysis.md` (cuDNN 2.78–2.96 ms, 59% compute / 62.8% mem).

---

## 0. Headline — the overlap worked, and the regime is shifting toward memory-throughput

> V13 removed serial phases from the consumer critical path (a store+reload+sync in the softmax,
> and the whole D-rowsum), cutting **total instructions 3.50 B → 2.93 B (−16%)** and lifting every
> throughput metric: **Memory SOL 50.2% → 57.0%** (now the highest of any version, nearing cuDNN's
> 62.8%), **Compute SOL 34% → 36%**, **IPC 1.37 → 1.45**, **mem throughput 414 → 523 GB/s**, **L1
> hit 8.4% → 15.4%**. And the **shared-LOAD conflict is now essentially solved: 1.6-way** (cuDNN
> ~2.2). Registers even dropped to 154. 0 spills.

The recurring lesson holds: **at low occupancy, deleting serial phases + syncs from the critical
path pays outsized** — same shape as the V8 D-rowsum shock. The producer's idle threads doing the
rowsum concurrently, plus fusing softmax into the GEMM epilogue, removed real critical-path latency.

---

## 1. V12 → V13 delta
| Metric | V12 | V13 | Δ |
|--------|-----|-----|---|
| **Wall-clock** | 11.16 ms | **8.82 ms** | **−21%** |
| TFLOP/s | 295.4 | **374.1** | +27% |
| Total executed instructions | 3.50 B | **2.93 B** | −16% |
| **Memory SOL** | 50.2% | **57.0%** | +6.8 (highest yet) |
| Compute SOL | 34.0% | 36.2% | +2.2 |
| IPC | 1.37 | 1.45 | +6% |
| Mem throughput | 414 GB/s | **523 GB/s** | +26% |
| L1/TEX hit | 8.4% | **15.4%** | +7 |
| No-eligible | 65.9% | 63.7% | −2 |
| **Shared-LOAD conflict** | (n/a) | **1.6-way / 10.4%** | ~solved (est. 5.9%) |
| Shared-STORE conflict | 4.1-way | **3.8-way** | est. 33.3% (top shared lever now) |
| Total shared wavefronts | 1.09 B | **0.94 B** | −14% |
| Registers / spills | 168 / 0 | **154 / 0** | −14 regs (fused softmax freed them) |
| Occupancy | 18.75% | 18.75% | unchanged |

## 2. Full metrics (launch 1; 3 launches <0.1%)
| Section | Value |
|---------|-------|
| Duration | 11.39 ms base / 8.82 ms wall |
| Compute SOL | 36.2% (ALU top pipe 23.5%) |
| Memory SOL | **57.0%** (L1/TEX 57.2%) — now the binding throughput |
| DRAM | 11.0% (not BW-bound) |
| L1 / L2 hit | 15.4% / 75.1% |
| **Shared-STORE conflict** | **3.8-way, 58.2% of 0.45 B (260 M)** — est. 33.3% (top real lever) |
| Shared-LOAD conflict | 1.6-way, 10.4% of 0.40 B — est. 5.9% (**solved**) |
| Uncoalesced shared (aggregate) | 234 M excessive / 0.94 B (25%) — est. 24.8% |
| Uncoalesced local | 1/32 bytes — est. 55.2% (**red herring**: ~0.16% of instr, 99% L1 hit — ignore) |
| Warp state | **no single dominant stall flagged** (V12 had long-scoreboard 33%) — overlap balanced the stalls |
| No-eligible | 63.7% (2.99 active warp/sched, 0.49 eligible) |
| **Occupancy** | **18.75%** (achieved 18.68%) — maxed (Reg + Smem both limit 1 block) |
| Launch | block 384, grid 2048, 154 regs, 0 spills, 223.8 KB smem |

## 3. Ladder + cuDNN
| | V11 | V12 | V13 | cuDNN |
|--|-----|-----|-----|-------|
| Wall bwd | 13.43 | 11.16 | **8.82 ms** | 2.78–2.96 ms |
| TFLOP/s | 246 | 295 | **374** | 1114–1187 |
| Compute SOL | 40% | 34% | 36% | 59% |
| **Memory SOL** | 50.8% | 50.2% | **57.0%** | 62.8% |
| Occupancy | 18.75% | 18.75% | 18.75% | 15.6% |
| Spills | 0 | 0 | **0** | 1.04 M |

Gap to cuDNN: ~4.0× (V12) → **~3.14× (V13)** (8.82 / 2.81). Journey: V3 ~35× → **V13 3.14×** (~11×
total over the naive baseline, all bit-identical + 0-spill).

**Important regime read:** our **Memory SOL (57%) now nearly matches cuDNN's (62.8%)**, but our
**Compute SOL (36%) is far below cuDNN's (59%)**. cuDNN is *compute-bound* (balanced 59/63); we are
**L1/shared-throughput-bound** (36 compute / 57 mem). The ~3× gap is now: *we push too many bytes
through L1/shared per unit of compute.* Closing it means **cutting shared traffic further** so
compute can become the binding resource, as it is for cuDNN.

## 4. V14 priorities (measured)

1. **Shared-STORE conflict / traffic (3.8-way, est. 33.3% — the top real lever).** The surviving
   conflict is the **fp32 accumulator scatters**: `store_acc_smem_v6` (dP → `sdP`) and the dQ
   `stage_acc_f32` staging. `stmatrix` can't help (it's 16-bit) — options: (a) store `sdP` as
   **bf16** so the dP scatter + the dS read become stmatrix/ldmatrix-able (precision risk — validate
   at 2e-2, but everything is bf16 output so likely fine); (b) pad/reindex the fp32 stores to
   de-alias the 3.8-way; (c) reduce the number of accumulator→shared round-trips. This is the direct
   path to lowering Memory SOL below Compute SOL.
2. **Offload more elementwise to the producer.** The producer still has slack after the D-rowsum;
   the `dS = P⊙(dP−D)` pass or a reshuffle could partly move there (bounded by the P/dP dependency).
3. **Free smem for the cross-tile S/dP pipeline** (still-blocked full FA3 overlap). V13 didn't free
   smem (sS reused for dQ staging); a reorg that genuinely frees ~33 KB would unlock computing tile
   N+1's S/dP during tile N — the largest remaining structural lever, highest risk.
4. FFMA fusion (~3.5%), L2 compression (~7.6%) — ignore.

**Recommended V14 = attack the shared-STORE traffic** (bf16 `sdP` + stmatrix, or de-alias the fp32
scatters). It's the measured top lever (est. 33%) and directly shifts us from memory-bound toward
compute-bound — the exact axis of the remaining cuDNN gap. The cross-tile pipeline (item 3) is the
bigger swing but needs an smem-freeing reorg first.

> **Framing:** occupancy, conflicts (load), writeback, and the softmax serial cost are all handled.
> The remaining ~3× is **shared-memory throughput** — fewer bytes through L1/shared per GEMM — which
> is precisely how cuDNN stays compute-bound at lower occupancy than ours.
