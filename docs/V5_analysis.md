# V5 Analysis — Fused KV-centric wgmma GQA Backward (Nsight Compute, D=128)

**Our kernel's own profile** — the measured basis for the next optimization (V6). Companion to
`gqa_bwd_py_analysis.md` (the cuDNN competitor). V5 is correct (matches V1–V4) and the fastest
of our versions (68.8 ms boost wall / 47.96 TFLOP-s), but this profile shows it is running at a
**completely different — and worse — operating point than cuDNN**, with one dominant, fixable bottleneck.

- **Kernel:** `gqa_backward_v5_kv<64,64,128>` — grid `(8,4,64)=2048` blocks, block `128` threads (1 warpgroup), CC 9.0.
- **Source:** `reports/gqa_v5_profile_d128.{ncu-rep,txt}` (`ncu --set full`, 3 launches, agreement <0.1%). ncu locks SM to 1.29 GHz → its **90.8 ms** is base-clock (wall is **68.8 ms** at boost). Read the **%s**, not the ms.
- **Competitor (from `gqa_bwd_py_analysis.md`):** cuDNN `flash_bprop_wgmma`, **2.96 ms**, 59% compute / 63% mem, 15.6% occ, register-spill-bound.

---

## 0. Headline verdict

> **V5 is shared-memory-BANK-CONFLICT bound.** Memory SOL 80% vs Compute SOL **13.5%** — the
> tensor cores sit almost idle. The single dominant cause: **17.9-way bank conflicts on shared
> loads, 94.4% of all shared-load wavefronts** (ncu est. **75.6%** speedup). Occupancy is
> **6.25%** (smem-limited → 1 block/SM → **1 warp per scheduler**), so 86.5% of cycles issue
> nothing. **0 register spills** (our edge over cuDNN, preserved).

This is the *inverse* of cuDNN: cuDNN is balanced/compute-fed and loses to register spilling;
we are memory-stalled with idle tensor cores and lose to **bank conflicts + occupancy**. That is
good news — our bottleneck is a classic, well-understood, high-headroom one.

---

## 1. Speed-of-Light
| Metric | V5 | cuDNN | Read |
|--------|-----|-------|------|
| **Compute (SM)** | **13.5 %** | 59 % | tensor cores starved |
| **Memory (SOL)** | **79.9 %** | 63 % | L1/TEX (80%) is the wall |
| DRAM | 1.4 % | 7.3 % | nowhere near BW-bound |
| L2 cache | 7.5 % | — | |
| Duration | 90.8 ms (base clk) | 2.44 ms (base) | wall 68.8 vs 2.96 ms |
| Roofline (FP32) | ~0 % | 4 % | |

ncu note: *"Memory is more heavily utilized than Compute — L1 bottleneck."*

## 2. Compute workload
| Metric | Value |
|--------|-------|
| Executed IPC (active) | **0.54** (of 4) |
| Issue slots busy | 13.5 % |
| SM busy | 13.5 % |

ncu: *"All compute pipelines under-utilized"* — est. 87% local speedup if fed. The wgmma units are fine; they're starved by stalls (§3, §4).

## 3. Memory workload — THE bottleneck (bank conflicts)
| Metric | Value | ncu est. speedup |
|--------|-------|------------------|
| **Shared-LOAD bank conflicts** | **17.9-way, 94.4% of 10.07 B wavefronts** (9.51 B conflicts) | **75.6 %** |
| **Shared-STORE bank conflicts** | 3.9-way, 72.6% of 1.59 B wavefronts (1.15 B conflicts) | 58.1 % |
| Uncoalesced local ld/st | 1 of 32 bytes/sector used | 77.4 % |
| Uncoalesced global stores | 8 of 32 bytes/sector used | 59.9 % |
| **Uncoalesced shared (aggregate)** | **90% excessive wavefronts** (10.66 B of 11.89 B) | **89.5 %** |
| L1/TEX hit rate | 3.3 % | |
| L2 hit rate | 81.8 % | |
| Local memory spilling | **0** ✅ | — |
| Memory throughput (abs) | 66 GB/s | (tiny; DRAM idle) |

**This is the story of the kernel.** ~90% of all shared-memory wavefronts are wasted on bank
conflicts — 17.9-way on loads. That traffic saturates L1 at 80% while the tensor cores idle.
The source is the operand path: the `fill_copy`/`fill_trans`/`fill_trans_A` repack writes + the
wgmma/readout shared reads land on a layout with pathological bank striding. Secondary: the
`fill_trans` scatter (uncoalesced local, 1/32) and the dK/dV/dQ global stores (8/32).

## 4. Scheduler & warp state — starved
| Metric | Value |
|--------|-------|
| **No eligible warp** | **86.5 %** of cycles |
| Active warps / scheduler | **1.00** (of 16) |
| Eligible warps / scheduler | 0.13 (issues ~1 instr / 7.4 cyc) |
| Warp cycles / issued instr | 7.41 |
| Dominant stall | **MIO short-scoreboard 50.5%** (shared-memory ops / bank conflicts) |
| Avg active threads / warp | 31.9 (no divergence) |

Only **one active warp per scheduler** — there is literally nothing to hide the shared-memory latency behind. Every bank-conflict replay stalls the whole scheduler.

## 5. Occupancy — smem-limited
| Metric | Value |
|--------|-------|
| **Achieved / theoretical occupancy** | **6.25 %** (4 warps/SM) |
| Block limit — **shared mem** | **1 block/SM** (221.7 KB) ← the limiter |
| Block limit — registers | 2 block/SM (254 regs) |
| Block limit — warps/barriers/SM | 16 / 32 / 32 |

Occupancy is pinned by **shared memory (221.7 KB/block → 1 block/SM)**, giving only 4 warps/SM = 1 warp/scheduler. Registers (254) would allow 2 blocks — smem is the binding constraint.

## 6. Launch / instruction
Block 128 (4 warps, 1 warpgroup); grid 2048; **254 regs, 0 spills**; **221.7 KB static smem**; waves/SM 15.5; branch eff 99.7% (avg 1107 divergent branches, 0.03% ratio). 8.31 B instrs; 152.6 M fused + 204.4 M non-fused FP32 (est. 1.3% from FFMA fusion — minor).

---

## 7. V5 vs cuDNN — different bottleneck, more headroom
| Dimension | V5 (ours) | cuDNN | takeaway |
|-----------|-----------|-------|----------|
| Compute SOL | **13.5 %** | 59 % | we don't feed the tensor cores |
| Memory SOL | 79.9 % (L1) | 63 % | we thrash L1 via bank conflicts |
| DRAM | 1.4 % | 7.3 % | neither BW-bound |
| Occupancy | **6.25 %** (4 warps/SM) | 15.6 % | we're even lower — 1 warp/sched |
| No-eligible | **86.5 %** | 66.7 % | we stall far more |
| Shared bank conflicts | **17.9-way / 94%** | 2.2-way / ~11% (D=64 data) | **our defining weakness** |
| Register spills | **0** ✅ | 1.04 M | our edge |
| Wall-clock | 68.8 ms | 2.96 ms | ~23× gap |

We hold the spill-free edge, but our shared-memory access pattern is far worse than cuDNN's
(17.9-way vs ~2.2-way). Closing that + occupancy is the path from 68.8 ms toward 2.96 ms.

---

## 8. V6 optimization priorities (ranked by measured est. speedup)

1. **Kill the shared-memory bank conflicts (est. 75–89%)** — the #1 lever by a wide margin.
   The `fill_*` repack layout and the wgmma/readout shared reads collide on banks (17.9-way).
   Fix: **swizzle/pad the shared operand + `sS`/`sdP`/`sP` layouts** so 32 threads hit 32 banks.
   The `tiled_off`/`mn_off` core-matrix layouts and the `store_acc_smem` scratch are the suspects
   — add padding (e.g. +1 element/row) or a proper XOR-swizzle. This alone should be transformative.
2. **Raise occupancy (est. ~20% + unlocks latency hiding)** — 221.7 KB smem → 1 block/SM → 1
   warp/scheduler. Cut smem so ≥2 blocks/SM fit (e.g. drop the dual `sK_sw`+`sK_pl` duplication,
   shrink double-buffer staging, or reuse buffers), giving more warps to hide the remaining
   shared latency. Registers (254) already permit 2 blocks — smem is the only thing in the way.
3. **Coalesce the writeback (est. 42–60%)** — dK/dV/dQ global stores use 8/32 bytes/sector, and
   the `fill_trans` local scatter uses 1/32. Reindex the flush for contiguous 128-bit stores.
4. **(Minor) FFMA fusion (~1.3%)** and L2 compression (~1%) — not worth it yet.

**Direction for V6:** attack bank conflicts first (swizzled/padded shared layouts) — it's ~80%
of the loss and independent of everything else. Then occupancy (smem reduction). Warp
specialization (producer/consumer, cuDNN-style) becomes worthwhile only after the shared-memory
path is clean and occupancy is up — otherwise more warps just contend harder on the same
conflicted banks.
