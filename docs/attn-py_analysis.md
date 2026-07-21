# Competitor Analysis — PyTorch SDPA Backward (FlashAttention-2) on H200

**What this is:** a fixed reference profile of the PyTorch attention path we are trying to
beat, so every custom wgmma version can be measured against a stable target. This does **not**
change per version — it's the short-term goal we're chasing.

- **GPU:** NVIDIA H200 (Hopper, CC 9.0, 132 SMs)
- **Workload:** GQA backward, `B=8, Hq=12, Hkv=4, G=3, S=4096, D=64`, bf16, causal
- **Backend forced:** `SDPBackend.FLASH_ATTENTION` → this is bundled **FlashAttention-2**
  (`pytorch_flash::…`), **not** cuDNN. (cuDNN's Blackwell/Hopper kernel may be a tougher
  target — worth a separate profile later.)
- **Source:** `reports/sdpa_bwd_flash.ncu-rep` (`ncu --set full`, 3 launches, backward only)
- **Tool:** `ncu` via `/usr/local/cuda/bin/ncu`

---

## 0. The one kernel that matters

The SDPA backward is effectively a single kernel. From the earlier nsys `cuda_gpu_kern_sum`:

| Kernel | Share of bwd time |
|--------|-------------------|
| **`flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`** | **~93%** ← this analysis |
| `flash_bwd_convert_dq_kernel` | ~4% |
| `flash_bwd_dot_do_o_kernel` | ~3% |

Everything below is that dominant kernel.

**Full signature:**
```
flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
    Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, 0, 0, bfloat16_t, …>, …>
grid (32, 8, 12) = 3072 blocks   block (256,1,1) = 256 threads
```

Decoding the traits: **headdim 64, tile 128×128 (kBlockM×kBlockN), 8 warps = 2 warpgroups**,
256 threads/block. Grid = `(S/128=32, B=8, Hq=12)` → **3072 blocks over 132 SMs ⇒ 23.3 waves**.
So this is a KV-seq-parallel FA2 backward with 128×128 tiles — much larger tiles and 2× the
warpgroups of our planned naive V3 (1 warpgroup, 64×64).

---

## 1. Headline verdict

> **Memory-bound at the L1 level (~61%), NOT DRAM-bound (3.5%), NOT compute-saturated (41%).
> Crippled by 12.5% occupancy (255 regs/thread + 147 KB smem ⇒ 1 block/SM) and register
> spilling, so it stalls with no eligible warp 65% of cycles.**

This is FlashAttention-**2**, and it is leaving a lot on the table on Hopper. It runs at very
low occupancy *without* the async latency-hiding that FA3 uses to make low occupancy work — so
the SMs sit idle two-thirds of the time. That is the seam to attack.

---

## 2. Speed-of-Light (the top-level scoreboard)

| Metric | Value | Read |
|--------|-------|------|
| **Duration** | **2.67 ms** | per launch, **at ncu-locked base clock 1.35 GHz** (see §8) |
| Elapsed cycles | 3.61 M | |
| **Memory throughput (SOL)** | **61.2 %** | ← the binding constraint |
| **Compute (SM) throughput** | **41.0 %** | tensor pipe, not saturated |
| **DRAM throughput** | **3.5 %** | **not** bandwidth-bound at all |
| L1/TEX cache throughput | 64.2 % | the actual hot spot |
| L2 cache throughput | 20.0 % | |
| SM active cycles | 3.43 M (95 % of elapsed) | SMs are busy… stalling |

The 61% "Memory Throughput" SOL is set by **L1/TEX (64%)**, not DRAM (3.5%). The working set is
L2-resident (see §4), so this kernel is limited by moving bytes through L1, not by HBM.

**Roofline:** achieved **3% of FP32 peak**, 0% FP64. (The MMAs show up as the Tensor(FP)
sub-pipeline; even so, only ~41% of the SM compute SOL.)

---

## 3. Compute workload

| Metric | Value |
|--------|-------|
| Executed IPC (active) | 1.39 inst/cycle (of 4 possible) |
| Issue slots busy | 33.0 % |
| SM busy | 41.0 % |
| Highest-utilized pipeline | **Tensor (FP)** @ ~41 % |

Tensor cores are the busiest pipeline but only ~41% utilized — consistent with "not
compute-bound." Low IPC (1.39) is a symptom of the stalls in §5.

---

## 4. Memory workload — where it actually hurts

| Metric | Value | Read |
|--------|-------|------|
| **L1/TEX hit rate** | **3.2 %** | almost nothing is reused in L1 |
| **L2 hit rate** | **91.6 %** | working set lives in L2, not DRAM |
| Memory throughput (abs) | 170 GB/s | tiny vs H200 ~4.8 TB/s → DRAM idle |
| Mem busy / Max bandwidth | 61.2 % / 50.1 % | |
| Mem pipes busy | 23.1 % | |
| **Local memory spilling requests** | **589,824** (100 % overhead) | **register spills → local mem** |
| Shared-store bank conflicts | 1.3-way, 21.5 % of wavefronts (7.32 M conflicts) | |

Two ncu optimization flags, both large:
- **Uncoalesced local loads & stores** — *"only 1.0 of 32 bytes per sector utilized"*, **est.
  speedup 59%** each. This is the register-spill traffic thrashing L1.
- **Shared-store bank conflicts** — 21.5% of shared-store wavefronts, est. speedup ~14%.

The chain is: **255 regs/thread hits the ceiling → compiler spills to local memory → spill
traffic is strided/uncoalesced (1/32 bytes useful) → L1 saturates at 64% with a 3% hit rate →
"Memory Throughput 61%" becomes the wall.** DRAM never breaks a sweat.

---

## 5. Scheduler & warp state — the stall story

| Metric | Value |
|--------|-------|
| **No eligible warp** | **65.4 % of cycles** |
| One-or-more eligible | 34.7 % |
| Active warps / scheduler | **2.00** (of 16 max) |
| Eligible warps / scheduler | 0.45 |
| Issued warp / scheduler | 0.35 (≈ 1 instr every 2.9 cycles) |
| Warp cycles / issued instr | 5.77 |
| Avg active threads / warp | 32 (no divergence) |

Only **2 active warps per scheduler**, and **0.45 eligible** — so 65% of cycles issue nothing.
With just 1 block/SM and no async overlap, there is nothing to hide the memory/MMA latency
behind. This single fact (65% dead cycles) explains most of the gap to peak.

---

## 6. Occupancy — the root limiter

| Metric | Value |
|--------|-------|
| **Theoretical occupancy** | **12.5 %** |
| **Achieved occupancy** | **12.49 %** (7.99 warps/SM) |
| Block limit — **registers** | **1 block/SM** (255 regs/thread) |
| Block limit — **shared mem** | **1 block/SM** (147 KB/block) |
| Block limit — warps | 8 |
| Block limit — barriers | 32 |

Occupancy is pinned at 12.5% by **both** registers (255/thread → 1 block) **and** shared memory
(147 KB/block → 1 block). ncu est. speedup from fixing occupancy: **~39%**.

---

## 7. Launch / resource envelope

| Metric | Value |
|--------|-------|
| Block size | 256 threads (8 warps, 2 warpgroups) |
| Grid size | 3072 blocks |
| Registers / thread | **255** (at the hard cap) |
| Dynamic shared mem / block | **147.46 KB** (near Hopper's 227 KB cap) |
| Stack size | 1024 B (spill backing) |
| Waves / SM | 23.27 |
| Branch efficiency | 100 % (no divergent branches) |
| FP32 instrs | 158.1 M non-fused, 0 fused (est. 4.8% from FFMA fusion) |

---

## 8. Measurement caveats (read before comparing ms)

1. **ncu locks clocks to base (SM 1.35 GHz).** H200 boosts to ~1.98 GHz, so the **2.67 ms here
   is a base-clock figure** — it is *not* directly comparable to the `do_bench` wall-clock
   (~3.1 ms for the *full* 3-kernel backward at boost). **Compare utilization %s (clock-
   independent), not absolute ms, across tools.** For wall-clock, use the `do_bench`/our own
   `benchmarkKernel` numbers.
2. This is **FA2**, not cuDNN and not FA3. It is a legitimate but not the strongest possible
   target.
3. Three launches were profiled and are within <0.1% of each other — the numbers are stable.

---

## 9. The scoreboard — target numbers for our wgmma versions

When we profile V3+ with the same `ncu --set full`, these are the columns to beat:

| Dimension | FA2 backward (target) | What "winning" looks like |
|-----------|----------------------|---------------------------|
| Memory throughput (SOL) | 61 % (L1-bound) | lower L1 pressure via TMA smem-direct loads |
| Compute (SM) throughput | 41 % | higher tensor-pipe utilization |
| L1 hit rate | 3.2 % | irrelevant if we bypass L1 with TMA |
| DRAM throughput | 3.5 % | stay low (we're not BW-bound either) |
| Occupancy | 12.5 % | either raise it, **or** hide latency at low occ (FA3 way) |
| "No eligible warp" | **65 %** | **the number to crush** — async overlap |
| Registers / thread | 255 (spilling) | avoid spills; keep accumulators in regs/TMEM cleanly |
| Shared mem / block | 147 KB | comparable or less |
| Wall-clock (full bwd) | **~3.1 ms** (boost, do_bench) | our headline to beat |

---

## 10. How we beat it (feeds the V4+ roadmap — not V3's job)

V3 (naive 1-warpgroup wgmma) will be *slower* than this — that's fine; it's the tensor-core
floor. The competitor analysis says the real wins come from:

1. **Async loads (TMA):** move K/V/Q/dO tiles global→shared directly, bypassing the L1/register
   traffic that gives FA2 its 3% L1 hit and 61% memory wall. Directly attacks the binding
   constraint.
2. **Latency hiding at low occupancy (warp specialization, FA3-style):** FA2 stalls 65% of
   cycles because it runs low occupancy *without* async overlap. A producer/consumer split
   (TMA-load warps + wgmma-compute warpgroups) hides latency without needing high occupancy —
   this is the single biggest lever.
3. **Clean register/TMEM accumulation:** avoid the 255-reg spill + fp32-shared round-trips.
   wgmma accumulates in registers; keep it there and flush coalesced.
4. **Kill bank conflicts:** pick a swizzle that removes the 21.5% shared-store conflicts.
5. (Minor) FFMA fusion on the elementwise softmax/dS math — ~5% only.

**Bottom line for the team:** the competitor is *not* fast because it saturates the H200 — it
saturates **L1 at 12.5% occupancy and stalls 2/3 of the time**. It wins on wall-clock purely by
being a mature fused kernel. A Hopper-native async design (TMA + wgmma + warp specialization)
has clear headroom to beat it.
```

*Generated from `reports/sdpa_bwd_flash.txt` (ncu `--set full`, 3× `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`).*
