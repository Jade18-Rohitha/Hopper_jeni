# ncu data: Vj1d vs cuDNN backward, B=8 S=4096 Hq=18 D=128, causal bf16 (H200, 2026-10-13)

Data collection only, no optimization yet. GPU 3, no fixed clocks (SM clock about 1.8 GHz for both runs).

> **Summary.** cuDNN's backward pipeline has the same 5 kernels as Vj1d's (dO·O preprocess, main kernel,
> 2× reduce over the G heads for dK/dV, dQ convert), and the helper kernels cost about the same (≈0.3 ms).
> **The whole gap is the main kernel: 3.578 vs 2.770 ms.** Vj1d's main kernel is limited by L2 bandwidth
> (80% of peak) and moves **2× the L2 bytes per key** that cuDNN does (Q/dO reloads 10.4 vs 5.4 GB, dQ fp32
> reduce-adds 9.8 vs 5.0 GB), with the same 64×64×128 tile. Neither kernel's traffic depends on G. **Confirmed from SASS execution counts
> (see `plan_vm1_bwd.md` step 0):** cuDNN's work item is 128 keys (2 compute warpgroups × 64 keys, sharing one
> Q/dO load and one dQ reduce per q-step, 132 persistent blocks); Vj1d covers 64.

Reports in `reports/` (raw profiles removed 2026-10-20 in the reports cleanup; the numbers in this doc are the record):
- `ncu_vj1d_B8_Hq18_Hkv9.ncu-repz`, `ncu_vj1d_B8_Hq18_Hkv6.ncu-repz`: Vj1d main kernel, full metric set
- `ncu_vj1d_all_B8_Hq18_Hkv9.ncu-repz`, `ncu_vj1d_all_B8_Hq18_Hkv6.ncu-repz`: all 5 kernels of one Vj1d backward
- `ncu_cudnn_bwd_B8_Hq18_Hkv9.ncu-repz`, `ncu_cudnn_bwd_B8_Hq18_Hkv6.ncu-repz`: cuDNN backward kernel, full metric set

## 1. Vj1d: kernels in one backward

| Kernel | Hkv=9 (G=2) | Hkv=6 (G=3) |
|---|---|---|
| compute_drowsum_v22 (dO·O) | 0.082 ms | 0.082 ms |
| gqa_bwd_vj1d (main) | 3.578 ms | 3.559 ms |
| gqa_dkdv_greduce ×2 | 0.113 ms | 0.100 ms |
| convert_dq_accum_to_bf16_v6 | 0.105 ms | 0.105 ms |
| **Total** | **3.878 ms** | **3.846 ms** |

Hkv has almost no effect: Vj1d runs one query head per block, so its work depends only on B×Hq.

## 2. Main kernel: Vj1d vs cuDNN (Hkv=9)

| | **Vj1d** `gqa_bwd_vj1d<64,64,128>` | **cuDNN** `..._flash_bprop_wgmma_f16_knob_2_64x64x128_1x4x1_...` |
|---|---|---|
| Duration | 3.578 ms @ 1.81 GHz | **2.770 ms** @ 1.78 GHz |
| Grid × block | (64, 18, 8) = 9,216 blocks × 128 thr | **132 blocks (persistent, 1 wave) × 384 thr** |
| Tile (Br×Bc×D) | 64×64×128 | 64×64×128 (from kernel name) |
| Registers / thread | 220 | 168 |
| Shared memory / block | 115 KB static | 216 KB dynamic |
| Blocks / SM | 2 (register and smem limited) | 1 |
| Active warps / SM | 7.9 (12.4%) | 10 (15.6%) |
| Spills | 0 | 322K local + 152K shared requests |
| **Tensor pipe (SM busy)** | **44.8%** | **60.0%** |
| L2 throughput | **80.0%** (limiting unit) | 64.2% |
| DRAM throughput | 7.8% | 10.1% |
| L2 hit rate | 95.5% | 91.5% |
| L1/TEX hit rate | 0% | 79.5% |
| DRAM read / write | 764 / 586 MB | 766 / 585 MB |
| Issue slots busy | 26.0% | 31.2% |
| Eligible warps / scheduler | 0.32 | 0.42 |
| No-eligible cycles | 73.5% | 67.8% |

### L2 traffic (sectors × 32 B)

| L2 traffic from SMs | Vj1d | cuDNN | cuDNN / Vj1d |
|---|---|---|---|
| Reads | 324.3 M sectors = **10.38 GB** | 167.6 M = **5.36 GB** | **0.52** |
| Reduce-adds | 306.7 M sectors = **9.81 GB** | 155.7 M = **4.98 GB** | **0.51** |
| Reduce requests | 76.7 M (4 sectors each) | 77.9 M (2 sectors each) | 1.01 |
| Writes | 9.4 M = 0.30 GB | 12.0 M = 0.38 GB | 1.27 |
| All L2 sectors | 818.4 M | 432.3 M | **0.53** |

**cuDNN moves about half as many L2 bytes for both reads and reduce-adds**, with the same tile size
(64×64×128). It issues the same number of reduce requests, but each one is half the size (64 B vs 128 B).

Vj1d's numbers match its algorithm exactly. Per causal step (2,080 steps per head × 144 heads), it reads Q and dO
(32 KB) and adds a 64×128 fp32 dQ partial (32 KB), giving 9.8 GB each.
cuDNN works out to about 18 KB read and 16.6 KB reduced per head-step.

**cuDNN at Hkv=6 (G=3) is identical to Hkv=9 (G=2):**

| cuDNN | Hkv=9 (G=2) | Hkv=6 (G=3) |
|---|---|---|
| Duration | 2.770 ms | 2.760 ms |
| L2 read sectors | 167,611,760 | 167,584,124 |
| L2 reduce sectors | 155,713,536 | 155,713,536 |
| L2 reduce requests | 77,856,768 | 77,856,768 |
| L2 throughput | 64.2% | 64.3% |
| Tensor pipe | 60.0% | 60.0% |

So cuDNN does **not** share K/V across the GQA group: its traffic doesn't depend on G, just like Vj1d's.

**Confirmed from SASS execution counts 2026-10-13 (`plan_vm1_bwd.md` step 0):** each cuDNN block covers **128 keys per q-step**,
as two consumer warpgroups of 64 keys each that share one Q/dO load and sum their dQ partials inside the block
before one reduce. That gives:
- reads: one 32 KB Q+dO load per 128 keys = 16 KB per 64 keys (+K/V/LSE): measured about 18 KB ✓
- reduce bytes: one 32 KB fp32 dQ reduce per 128 keys = 16 KB per 64 keys: measured 16.6 KB ✓
- reduce requests: half as many reduces, but 64 B instead of 128 B per request, so the same request count: measured 77.9 M vs 76.7 M ✓

The 384-thread block (3 warpgroups = 1 producer + 2 consumers) is consistent with this. Vj1d instead covers
64 keys per block (one warpgroup), so it reloads Q/dO and reduces dQ twice as often per key.

### Warp stalls (cycles per issued instruction)

| Stall | Vj1d | cuDNN |
|---|---|---|
| long_scoreboard | 1.70 | 1.78 |
| barrier | 1.53 | 2.21 |
| wait | 1.07 | 0.98 |
| selected | 1.00 | 1.00 |
| short_scoreboard | 0.65 | 0.79 |
| gmma | 0.43 | 0.16 |
| mio_throttle | 0.42 | 0.13 |
| not_selected | 0.20 | 0.30 |
| dispatch_stall | 0.25 | 0.18 |
| **Total (cycles per issue)** | **7.45** | **≈ 7.8** |

The stall profiles are similar. cuDNN gets more done per cycle because it has less L2 traffic per
FLOP (tensor pipe 60% vs 45%), not because its warps stall less.

## 3. Full backward pipeline: cuDNN's kernels vs Vj1d's

cuDNN kernel list from nsys (`reports/nsys_cudnn_bwd_B8_Hq18_Hkv9_kernels.txt`, Hkv=9, avg over 59 backward
calls during `cudnn_time.py`, clock not locked). Vj1d times are from ncu (`ncu_vj1d_all_B8_Hq18_Hkv9`, ~1.81 GHz).
Different tools and clocks, so compare roles and rough sizes, not small differences.

| Role | cuDNN kernel | cuDNN avg | Vj1d kernel | Vj1d |
|---|---|---|---|---|
| dO·O row-dot preprocess | `cudnn::fusion::compute_dot_do_o_specialized` | 0.149 ms | `compute_drowsum_v22` | 0.082 ms |
| **Main backward** | `..._flash_bprop_wgmma_f16_knob_2_64x64x128_1x4x1_...` | **2.965 ms** (ncu: 2.770) | `gqa_bwd_vj1d<64,64,128>` | **3.578 ms** |
| dK/dV sum over the G query heads (×2: dK, dV) | `cudnn::fusion::fmha_reduce_head` ×2 | 2 × 0.055 = 0.110 ms | `gqa_dkdv_greduce` ×2 | 2 × 0.057 = 0.113 ms |
| dQ fp32 → bf16 | `cudnn::fusion::convert_dq_to_16bits` | 0.116 ms | `convert_dq_accum_to_bf16_v6` | 0.105 ms |
| Fill (int) | `at::native::...FillFunctor<int>` | 0.058 ms (54 calls) | in-launcher `cudaMemset` of dQ accum | (not a kernel) |

Also seen in the same capture: the cuDNN forward, `..._flash_fprop_wgmma_f16_knob_1_64x128x128_4x1x1_...`, avg 1.067 ms
(41 calls). Its tile is 64×128×128 with 4×1×1 warp layout, different from the backward's.

**What this shows:**
- **cuDNN's backward pipeline has the same shape as Vj1d's:** dO·O preprocess → main kernel writing
  per-query-head partial dK/dV and fp32 dQ → a separate reduce over the G heads for dK and for dV → dQ convert.
  The `fmha_reduce_head` ×2 kernels match our `gqa_dkdv_greduce` ×2 one-to-one, at the same speed (≈0.055 ms each).
  That also explains why cuDNN's dK/dV are bit-identical run to run: one writer per head plus a fixed-order reduce, the same as Vj1d.
- **Helper kernels are not the gap.** Ours total ≈0.30 ms vs cuDNN's ≈0.33 ms (+0.06 ms fill). Our dO·O
  preprocess is faster (0.082 vs 0.149 ms).
- **The whole gap is the main kernel.** ncu vs ncu, same shape: 3.578 vs 2.770 ms (+0.81 ms, at a slightly
  higher clock for ours). Section 2 shows why: Vj1d moves 2× the L2 bytes per key.

## 4. Still to collect

1. ~~cuDNN at Hkv=6~~: done (section 2): traffic doesn't depend on G.
2. ~~cuDNN's other backward kernels~~: done (section 3).
3. **Locked-clock repeat** (`nvidia-smi -i 3 -lgc 1980` + ncu `--clock-control none`) so durations match the
   benchmark tables.
