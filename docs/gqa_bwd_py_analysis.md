# GQA Backward — PyTorch Competitor Profile (cuDNN flash-bprop wgmma, H200, D=128)

**Long-term reference.** The permanent record of the PyTorch SDPA backward kernel we are
racing on Hopper H200 at the **final target shape D=128**. Section 0 is the durable synthesis
(identity, strengths, weaknesses, strategic read); §1–§8 are the full metric detail. This does
**not** change per custom-kernel version (V4, V5, …) — those get their own docs. Revisit only
if cuDNN's kernel changes (new cuDNN/driver) or the target shape moves off D=128.

- **GPU:** NVIDIA H200 (Hopper, CC 9.0, 132 SMs), SM base 1.36 GHz / boost ~1.98 GHz, HBM3e ≈ 4.8 TB/s
- **Workload:** GQA backward, `B=8, Hq=12, Hkv=4, G=3, S=4096, D=128`, bf16, causal
- **Backend:** PyTorch SDPA → **cuDNN fused flash attention** (`cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wgmma_f16`)
- **Wall-clock (do_bench, boost):** SDPA bwd **2.96 ms / 1114 TFLOP/s** (4×fwd FLOP convention); fwd 0.71 ms / 1168 TFLOP/s
- **Sources:** `reports/sdpa_bwd_d128.{ncu-rep,txt}` (`ncu --set full`, 3 launches, cross-launch agreement <0.1%); `reports/sdpa_nsys.nsys-rep` (kernel breakdown). ncu locks SM to base 1.36 GHz — its **2.44 ms** duration is a base-clock figure (~1.68 ms at boost); use the do_bench 2.96 ms for wall-clock.

> **Cross-box reconfirmation (2026-07-24, box `plain-game-shines-fin-02`, `reports/sdpa_nsys1.nsys-rep`).**
> A **third** H200 instance dispatched the **same** backend — `cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wgmma_f16_knob_26_64x64x128_1x4x1_cga1x1x1` (main kernel **1.93 ms/call**) — so this whole analysis holds unchanged. This box benchmarks **SDPA bwd (eager) 2.78 ms / 1187 TFLOP/s** (compiled 2.93 ms; fwd 0.73 ms / 1128) — a hair faster than the 2.96 ms above; treat **2.78–2.96 ms** as the box-dependent target band. Confirmed cuDNN's backward is a **multi-kernel pipeline** (per-call): `flash_bprop_wgmma` main 1.93 ms + `compute_dot_do_o_specialized` (dO·O rowsum prep) 0.10 ms + `convert_dq_to_16bits` (dQ fp32→bf16) 0.08 ms + `fmha_reduce_head` ~0.03 ms ≈ 2.14 ms GPU + launch gaps → 2.78 ms wall. This validates our fused V5 structure: cuDNN **also** accumulates dQ then converts, and computes the dO·O rowsum in a separate pass. (Two other H200 boxes seen prior: `agile-light-falls` → this same cuDNN family; `quiet-life` → FlashAttention-2 `flash_bwd_dq_dk_dv_loop_seqk_parallel`.)

---

## 0. Competitor Profile — approach, strengths, weaknesses

### Identity & approach
- Kernel: `…flash_bprop_wgmma_f16_knob_26_64x64x128_1x4x1_cga1x1x1` — a single fused cuDNN-generated flash-attention backward. Tile **64×64×128** (the K dim = head-dim D = 128; vs 64×64×64 at D=64). `1x4x1` warpgroup config, `cga1x1x1` (no real clusters).
- Launch: **grid = 132 (= #SMs), block = 384 threads (12 warps = 3 warpgroups), 1 wave per SM** → a **persistent, warp-specialized** kernel (FA3-style: 1 TMA-producer + 2 wgmma-consumer warpgroups), one block per SM, no tail.
- **wgmma tensor-core driven** — Tensor(FP) is the top pipeline at **59.4%** of cycles.
- Footprint: **168 regs/thread**, **216 KB dynamic smem/block** (of 233 KB) — near the per-block ceiling on both axes → 1 block/SM.

### The D=64 → D=128 shift (important)
At D=64 this kernel was **latency-bound** (44% compute, 57% mem, ncu flagged "latency issues"). At **D=128 it is compute/memory-BALANCED**: **59.4% compute, 62.8% memory**, and ncu's own note changes to *"Compute and Memory are well-balanced."* The larger head dim gives more MMA work per byte moved, so the tensor cores are better fed. It is still not saturated (both ~60%) and still stalls 67% of cycles, but the regime is no longer "pure latency."

### Strengths (durable — worth learning from)
1. **Persistent + warp-specialized + async** (TMA-load producer overlapping wgmma-compute consumers) lets it run at only **15.6% occupancy** yet keep the tensor pipe ~59% busy. Low occupancy is a *deliberate design*, not a defect — it hides latency with async, not with many warps.
2. **High cache locality:** L1/TEX hit 79.7%, L2 hit 92.0% — big tiles + smem staging keep the working set on-chip; **DRAM is only 7.3%** utilized (nowhere near bandwidth-bound).
3. **Near-zero control-flow cost:** branch instrs 0.02% of total, 98.65% efficient.
4. **Balanced compute/memory at D=128** — no single lopsided bottleneck; to go faster you'd have to cut both.

### Weaknesses (durable — real headroom)
1. **Register spilling** — 168 regs still isn't enough: **1,038,468 local spill requests** (100% overhead), and the spill traffic is **badly coalesced (1 of 32 bytes/sector used)** → ncu est. **~60.8% speedup** from fixing the local load/store coalescing alone. This is cuDNN's single biggest inefficiency (same as D=64, now larger in absolute count).
2. **67% "no eligible warp"** — even with async, 2/3 of cycles issue nothing; ncu est. ~37% from raising occupancy/reducing the register+smem pressure that pins it to 1 block/SM.
3. **Free FFMA fusion left on table** — 103.8M non-fused FP32 (0 fused-pair capture), ~5% est.
4. **L2 compression configured but 0% success** — ~4% est., unused.
5. **Net: ~59% compute / ~63% memory of peak — a strong but not maxed kernel.** Real, quantified headroom (~37–60% by ncu's own estimators) exists; cuDNN is a serious target, not a hardware ceiling.

### Strategic takeaway for our kernels
cuDNN's weakness is **register spilling** (its warp-specialized big-tile design is register-hungry). **Our V3 is already 0-spill** — a genuine structural edge to *keep*. We do NOT beat this by cranking occupancy (cuDNN wins at 15.6%); we beat/approach it by **async overlap (TMA + wgmma pipelining) + warp specialization** to feed the tensor pipe — which is exactly the V4/V5 plan. At D=128 the bar is higher than D=64 (59% vs 44% compute), because the head dim now keeps cuDNN's tensor cores genuinely busy.

---

## 1. Speed-of-Light (top-level scoreboard)

| Metric | Value | Read |
|--------|-------|------|
| **Duration** | **2.44 ms** | ncu base clock 1.36 GHz (~1.68 ms at boost; do_bench full-bwd 2.96 ms) |
| Elapsed cycles | 3.32 M | |
| **Compute (SM) throughput** | **59.4 %** | Tensor(FP) top pipeline — well-fed at D=128 |
| **Memory throughput (SOL)** | **62.8 %** | set by L1/TEX (65.7%) |
| **DRAM throughput** | **7.3 %** | not bandwidth-bound |
| L2 cache throughput | 49.8 % | |
| SM active cycles | 3.16 M (95% of elapsed) | |
| Roofline | 4% of FP32 peak | |
| ncu verdict | **"Compute and Memory are well-balanced"** | not latency-bound (unlike D=64) |

## 2. Compute workload
| Metric | Value |
|--------|-------|
| Executed IPC (active) | 1.33 (of 4) |
| Issue slots busy | 31.85 % |
| SM busy | 59.4 % |
| Top pipeline | **Tensor (FP)** @ 59.4 % |

## 3. Memory workload
| Metric | Value | Read |
|--------|-------|------|
| **Local spill requests** | **1,038,468** (100% overhead) | register spilling (168 regs insufficient) |
| Uncoalesced local ld/st | 1 of 32 bytes/sector → **est. 60.8% speedup** | the spill traffic thrashing L1 |
| L1/TEX hit rate | 79.7 % | strong on-chip reuse |
| L2 hit rate | 92.0 % | working set L2-resident |
| Memory throughput (abs) | 352.7 GB/s | tiny vs 4.8 TB/s → DRAM idle |
| Mem pipes busy | 35.2 % | |

## 4. Scheduler & warp state
| Metric | Value |
|--------|-------|
| **No eligible warp** | **66.7 %** of cycles |
| One-or-more eligible | 33.3 % |
| Active warps / scheduler | 2.50 |
| Eligible warps / scheduler | 0.44 (issues ~1 instr / 3.0 cyc) |
| Warp cycles / issued instr | 7.51 |
| Avg active threads / warp | 31.73 (minimal divergence) |

## 5. Instruction statistics
556 M executed; **25.95 M fused + 103.8 M non-fused FP32** (est. 5% from FFMA fusion). Branch instrs 0.02%, 98.65% efficient, avg 206.9 divergent branches (up from ~15 at D=64 but still negligible ratio).

## 6. Launch / occupancy
| Metric | Value |
|--------|-------|
| Block size | **384 threads (12 warps = 3 warpgroups)** |
| Grid size | **132 (= #SMs)** — persistent |
| Waves / SM | **1** |
| Registers / thread | **168** |
| Dynamic smem / block | **216.06 KB** (of 233 KB; was 152 KB at D=64) |
| Theoretical occupancy | 18.75 % |
| **Achieved occupancy** | **15.63 %** (10 warps/SM) |
| Block limit — registers / shared mem | **1 / 1** block/SM |

## 7. Cross-check: D=64 vs D=128 (same kernel family)
| Dimension | D=64 | D=128 |
|-----------|------|-------|
| Wall-clock bwd (do_bench) | 1.88 ms / 877 TFLOP/s | **2.96 ms / 1114 TFLOP/s** |
| ncu duration (base clk) | 1.63 ms | 2.44 ms |
| Compute SOL | 44 % | **59 %** |
| Memory SOL | 57 % | 63 % |
| Regime | latency-bound | **balanced** |
| Occupancy | 15.6 % | 15.6 % |
| No-eligible | 58.6 % | 66.7 % |
| Regs / smem | 168 / 152 KB | 168 / **216 KB** |
| Local spills | 832 K | **1.04 M** |
| Tile | 64×64×64 | **64×64×128** |

The design is identical (persistent, 3 warpgroups, warp-specialized); D=128 doubles the K-tile → 2× the compute per block and ~1.4× smem, pushing compute util from 44%→59% and turning a latency-bound kernel into a balanced one. Throughput rises (877→1114 TFLOP/s) because the tensor cores are better fed.

---

## 8. Scoreboard — what V4+ must chase (D=128)

| Dimension | cuDNN reference | our V3 (current) | V4 goal |
|-----------|-----------------|------------------|---------|
| Wall-clock bwd | **2.96 ms** | ~104 ms (wgmma) / 88.7 ms (V2 wmma) | ≪ 104 ms; approach 2.96 ms |
| Whole-op throughput | **1114 TFLOP/s** | 31.7 (V3) / 37.2 (V2) | ↑ |
| Compute (SM) SOL | 59 % | (unprofiled) | feed the tensor pipe |
| DRAM SOL | 7.3 % | — | stay low |
| Occupancy | 15.6 % (1 blk/SM) | — | don't chase occupancy |
| No-eligible | 66.7 % | — | overlap loads with wgmma |
| Threads / tile | 384 (3 warpgroups, warp-spec), 64×64×128 | 128 (1 warpgroup), 64×64×128 | V5: warp-spec |
| Regs / smem | 168 / 216 KB | 159 / 196 KB (V3) | keep ≤ |
| Register spills | **1.04 M (spills)** | **0 (our edge)** | keep 0 |

**Bottom line:** the target is `flash_bprop_wgmma` at **~2.96 ms / 1114 TFLOP/s**, D=128 — a persistent, warp-specialized, async wgmma kernel running balanced at 59% compute / 15.6% occupancy, with register spilling as its one clear weakness. We share the primitive (wgmma) and hold the spill-free edge; the gap is **async overlap + warp specialization** (V4 = TMA-loaded double-buffered wgmma; V5 = warp specialization + persistent grid).
