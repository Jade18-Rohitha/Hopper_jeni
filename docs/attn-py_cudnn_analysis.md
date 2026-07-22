# Competitor Analysis — PyTorch SDPA Backward (cuDNN flash, wgmma) on H200

**What this is:** the fixed reference profile of the PyTorch attention backward on the
`agile-light-falls` H200 instance — **our base competitor**. This box's PyTorch selects a
different, faster backend than the `quiet-life` box. Companion to
[attn-py_analysis.md](attn-py_analysis.md) (FlashAttention-2, 531 TFLOP/s); **this box runs
cuDNN and hits 877 TFLOP/s**. This doc is the number V3+ must chase and does not change per
custom version.

- **GPU:** NVIDIA H200 (Hopper, CC 9.0, 132 SMs), SM base ≈ 1.36 GHz, HBM3e ≈ 4.8 TB/s peak
- **Workload:** GQA backward, `B=8, Hq=12, Hkv=4, G=3, S=4096, D=64`, bf16, causal
- **Backend:** PyTorch SDPA → **cuDNN fused flash attention** (`cudnn_generated_fort_native_sdpa_sm90_flash_*_wgmma_f16`)
- **Wall-clock (do_bench):** SDPA bwd **1.88 ms / 877 TFLOP/s** (4×fwd FLOP convention); fwd 0.48 ms / 857 TFLOP/s
- **Sources:** `reports/sdpa_nsys.nsys-rep` (breakdown) · `reports/sdpa_bwd_main.{ncu-rep,txt}` (main kernel, `--set full`) · `reports/sdpa_bwd_cudnn.{ncu-rep,txt}` (helper kernels)

---

## 0. The backward is FOUR kernels

Per-call timing from nsys (`cuda_gpu_kern_sum`, wall-clock at boost):

| Kernel | avg / call | share | role |
|--------|-----------|-------|------|
| **`…flash_bprop_wgmma_f16…64x64x64_1x4x1`** | **~1.25 ms** | **~90 %** | main flash backward — S, P, dP, dS, dQ, dK, dV via **wgmma** |
| `cudnn::fusion::compute_dot_do_o_specialized<1,64>` | ~51 µs | ~4 % | pre-pass: delta `D = rowsum(dO∘O)` |
| `cudnn::fusion::convert_dq_to_16bits<1>` | ~42 µs | ~3 % | post-pass: fp32 dQ accumulator → bf16 |
| `cudnn::fusion::fmha_reduce_head<1>` (×2) | ~18 µs ea | ~3 % | GQA reduction: sum dK/dV across the G query heads per KV head |

**The kernel to beat is `flash_bprop_wgmma` (~1.25 ms wall / 1.63 ms at ncu base clock).** The
three helpers add only ~0.13 ms. Forward reference: `…flash_fprop_wgmma…64x128x64_4x1x1` @ ~0.48 ms.

---

## 1. Main kernel — `flash_bprop_wgmma` (ncu `--set full`, 4 launches, near-identical)

Decode: `sm90 · flash bprop (backward) · wgmma · f16 · 64×64×64 tile · knob_26 · cga 1×1×1 (no cluster)`.

### 1.1 Launch / resource envelope — the defining choices
| Metric | Value | Read |
|--------|-------|------|
| **Grid size** | **132 blocks** (= #SMs) | **persistent kernel** — exactly one block per SM |
| **Waves / SM** | **1** | no tail effect; each SM streams all its work in one block |
| **Block size** | **384 threads = 12 warps = 3 warpgroups** | **warp-specialized** (FA3-style: TMA-producer + wgmma-consumer warpgroups) — NOT V3's single warpgroup |
| Registers / thread | **168** | heavy; caps occupancy to 1 block/SM |
| Dynamic smem / block | **152.58 KB** | large; also caps to 1 block/SM (233 KB configured) |
| Stack size | 1024 B | spill backing (see §1.4) |

### 1.2 Speed-of-Light — latency-bound, not saturated
| Metric | Value | Read |
|--------|-------|------|
| **Duration** | **1.63 ms** | ncu base clock 1.36 GHz (wall ≈ 1.25 ms at boost) |
| Elapsed cycles | 2.23 M | |
| **Compute (SM) throughput** | **44.3 %** | Tensor(FP) is the top pipeline @ 44 % |
| **Memory throughput (SOL)** | **57.0 %** | set by **L1/TEX (59.7 %)** |
| **DRAM throughput** | **5.3 %** | **not** bandwidth-bound (256 GB/s of 4.8 TB/s) |
| L2 cache throughput | 37.3 % | |
| Roofline | **7 % of FP32 peak** | |

> ncu's own verdict: *"low compute throughput AND memory bandwidth… below 60 % of peak typically indicates **latency issues**."* **Even cuDNN's tuned kernel is latency-bound, not throughput-saturated.**

### 1.3 Occupancy — deliberately low
| Metric | Value |
|--------|-------|
| **Achieved occupancy** | **15.6 %** (10 active warps/SM of 64) |
| Theoretical occupancy | 18.75 % |
| Block limit — registers | **1 block/SM** (168 regs) |
| Block limit — shared mem | **1 block/SM** (152 KB) |

cuDNN wins at **15.6 % occupancy** — it does *not* rely on many warps to hide latency; it uses
async (TMA + wgmma pipelining) and warp specialization instead. This is the crucial lesson:
**cranking occupancy is not how you beat this.**

### 1.4 Scheduler / warp state — where the 1.63 ms goes
| Metric | Value | Read |
|--------|-------|------|
| **No eligible warp** | **58.6 % of cycles** | stalls > half the time |
| One-or-more eligible | 41.4 % | |
| Active warps / scheduler | 2.50 | |
| Eligible warps / scheduler | 0.58 | issues ~1 instr / 2.4 cycles |
| Warp cycles / issued instr | 6.04 | |
| Avg active threads / warp | 31.98 | essentially no divergence (branch eff 99.9 %) |

### 1.5 Memory workload
| Metric | Value | Read |
|--------|-------|------|
| **L1/TEX hit rate** | **74.8 %** | good reuse (vs FA2's 3 %) — big tiles + smem staging |
| L2 hit rate | 92.1 % | working set L2-resident |
| Mem pipes busy | 28.1 % | |
| **Local memory spilling requests** | **832,644** (100 % overhead) | 168 regs still spills — backward is inherently register-heavy |
| Uncoalesced local loads/stores | est. 55 % speedup flag | the spill traffic (1 of 32 bytes/sector used) |

### 1.6 Instruction mix
463 M instructions; **25.95 M fused + 103.8 M non-fused FP32** (softmax/dS elementwise) — ncu flags ~6 % from further FFMA fusion. Tensor(FP) is the dominant pipeline at 44 %.

---

## 2. Helper-kernel metrics (ncu `--set full`, base clock)

All three helpers are **DRAM-bandwidth-bound** (Memory ≈ DRAM throughput) at **high occupancy
(~84–88 %)** — the opposite regime from the main kernel. Lightweight streaming ops, near
bandwidth-optimal; not where the battle is.

| Kernel | Dur (µs) | DRAM % | achieved BW | Compute % | Occupancy | Regs | Smem | Grid×Block |
|--------|---------|--------|-------------|-----------|-----------|------|------|------------|
| `compute_dot_do_o<1,64>` | 52.1 | 74.2 | 3.57 TB/s | 40.9 | 87.5 % | 26 | 0 | (256,8,12)×128 |
| `convert_dq_to_16bits<1>` | 41.7 | 69.1 | 3.32 TB/s | 20.9 | 84.8 % | 28 | 0 | (128,8,12)×128 |
| `fmha_reduce_head<1>` | 18.0 | 67.2 | 3.23 TB/s | 45.2 | 84.4 % | 32 | 0 | (256,8,4)×128 |

cuDNN splits the softmax `delta`, the GQA head-reduction, and the dQ down-convert into these
separate DRAM-bound passes. A monolithic kernel (our V3) folds them inline — so the fair
comparison target is `flash_bprop_wgmma` + these ~0.13 ms of helpers.

---

## 3. Scoreboard — what V3+ must chase on THIS box

| Dimension | cuDNN reference | our V3 (once correct) |
|-----------|-----------------|-----------------------|
| Wall-clock, full bwd | **1.88 ms** | TBD |
| Main-kernel wall-clock | **~1.25 ms** | TBD |
| Whole-op throughput | **877 TFLOP/s** | TBD |
| Compute (SM) throughput | **44 %** (tensor-limited) | — |
| DRAM throughput | 5.3 % (not BW-bound) | — |
| Occupancy | **15.6 %** (1 block/SM) | — |
| "No eligible warp" | **58.6 %** | — |
| Tile / threads | 64×64×64, **384 thr (3 warpgroups, warp-specialized)** | 64×64, 128 thr (1 warpgroup) |
| Regs / smem | 168 / 152 KB | 111–119 / 99 KB |
| Register spills | **832K local requests (spills)** | **0 spills** ✅ (our edge) |

---

## 4. Strategic takeaways for the roadmap

1. **You do NOT beat this with occupancy.** cuDNN runs at 15.6 % occupancy and stalls 59 % of
   cycles, yet wins on wall-clock. It hides latency with **async (TMA + wgmma pipelining) and
   warp specialization**, not with many resident warps.
2. **It's a persistent, warp-specialized, big-tile kernel.** Grid = #SMs (1 block/SM, 1 wave),
   384 threads = 3 warpgroups (producer + consumers). Our V3 is a single synchronous warpgroup
   with no async — expect it to be **many× slower even when correct**. That's fine; V3 is the
   tensor-core floor.
3. **It is compute/latency-bound, not DRAM-bound** (5.3 % DRAM). So the win comes from keeping
   the tensor pipe fed (currently only 44 %) — i.e. overlap loads with wgmma so the MMA units
   don't stall. That is precisely the async pipeline V3 lacks.
4. **Spill-free is an EDGE we already hold — keep it.** cuDNN spills (832K local requests, 168
   regs) and ncu flags ~55 % lost to the uncoalesced local traffic that causes. That is a
   *consequence of cuDNN's design* (huge live state, big warp-specialized tiles), **not a law**.
   Our V3 is already **0-spill** (`dQ_v3` 111 regs, `dKdV_v3` 119 regs, 0 spill bytes) because our
   64×64 tiling keeps live state under the 255-reg ceiling. This is a genuine advantage — a
   spill-free kernel avoids the L1/local-bandwidth tax cuDNN pays. **Design constraint for V4/V5:
   preserve 0 spills** as we add double-buffering / warp specialization (stage extra live state in
   smem, not registers; keep per-thread accumulator footprint bounded). If a feature forces
   spills, prefer a smaller tile or smem staging over accepting them.
5. **Roadmap implied:** V3 (correct, synchronous wgmma floor) → V4 (TMA/cp.async double-buffered
   loads → feed the tensor pipe, attack the 59 % stall) → V5 (warp specialization: dedicated
   TMA-producer + wgmma-consumer warpgroups, persistent grid) → approach cuDNN's 1.25 ms.

**Bottom line:** the target is `flash_bprop_wgmma` at ~1.25 ms — a *latency-bound, warp-
specialized, async* wgmma kernel at only 44 % compute / 15.6 % occupancy. We're using the same
primitive (wgmma 64×64); closing the gap is about **async overlap and warp specialization**, not
occupancy or bandwidth.
