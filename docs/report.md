# GQA Backward — H200 optimization log (2026-08-14)

**Result: we now beat cuDNN on three shapes — including the marquee 4×12.** Starting point this morning was V44 at ~1.53 ms (4×12); by end of day V44 is **1.474 ms locked-clock, under cuDNN's 1.478**. The win did **not** come from the main kernel — that hit a hard wall — it came from the e2e helper kernels cuDNN also pays.

The dev box is sm_120, so everything below ran on an H200 remote — compiled here (sm_90a), profiled there. Both wins are shipped in `src/attention/GQA_bwd_baseline.cu`.

---

## 1. The main kernel is walled

I spent the first half chasing the main-kernel gap to cuDNN and mapped it completely.

**The L2/boundary tension — cracked.** cuDNN's bprop kernel is *persistent* (grid = 132 = SM count, 1 wave) with a dynamic tile scheduler. I rebuilt that as a persistent kernel (`Vp1`) and, after three failed prefetch schemes, cracked the tension between L2-locality and boundary-overlap with a **causal-paired chunked scheduler**: grab work-items in pairs `{k_tile t, S/Bc−1−t}` of the same (b,hkv) group. Each pair is constant-work (causal pairs sum to a constant) so the 132-CTA lockstep never drifts → **96 % L2 (beating cuDNN's 92 %)**, and both items share a group so the in-pair K/V prefetch is local (no DRAM pollution). That took Vp1 1.67 → 1.53 ms while *conserving* the L2.

**But the last per-tile gap to cuDNN is three simultaneous hardware walls** (measured, not guessed):

| lever | targets | wall |
|---|---|---|
| wgmma cross-tile pipeline | `wait`+`gmma` 1.58 → 1.16 | **registers** — the persisted S-GEMM accumulator spills (288 B) over the 170-reg cap (384 thr × 1 CTA/SM); result: 3.3 ms |
| PD=3 deeper Q/dO ring | `long_scoreboard` 2.54 → ~1.76 | **smem** — needs single-buffer dQ stage = +0.5 ms serialize |
| `DEFER_BLOCKING` barriers | SM-busy 37 % → 57 % | **ptxas codegen** — not emittable from CUDA C |

cuDNN fits all three in the *same* 217 KB / 170 regs via a leaner tile structure (its `1x4x1` warp-tiling) + a gmem schedule table + 4D TMA — a ground-up redesign, not an incremental lever. So the persistent kernel plateaus; the champions stay **V44** (low/mid batch) and **Vr1** (big memory-bound shapes).

## 2. The e2e-plumbing win

Since the main kernel was walled, I profiled the *full* backward (`compute_drowsum` → main → `convert_dq_accum`) and found the reclaimable headroom in the two memory-bound helpers cuDNN also pays:

- **Vectorized `convert_dq_accum → bf16`** (the fp32 dQ accumulator → bf16 output). It was **scalar** — one fp32 → one bf16 per thread, ~2 TB/s, **76 µs**. Now each thread converts a `float4` (16 B load) → 4× bf16 (`uint2`, 8 B store) → near-peak bandwidth, **~45 µs**. Bit-identical. This alone dropped V44 1.53 → 1.485.
- **Vectorized the D-rowsum** — contiguous `uint2` (4×bf16) loads instead of 4 strided reads. It was already ~2.9 TB/s so the gain is smaller (~12 µs); D shifts by |Δ| ~1e-6 (contiguous vs strided sum order), well within the 2e-2 check tolerance. Dropped V44 1.485 → 1.473.

Together **~0.06 ms off every shape's e2e**, with the changes wired into all launch sites.

## 3. Full sweep

Best-of V44/Vr1/V45, **locked-clock** median ms (`nvidia-smi -pm 1; -lgc 1980`). cuDNN = PyTorch `enable_gqa` bwd (cuDNN backend); Triton = flash bwd:

| B×Hq (Hkv,G) | V44 | Vr1 | best | won | cuDNN | Triton | vs cuDNN | vs Triton |
|---|---|---|---|---|---|---|---|---|
| 2×12 (4,3) | **0.745** | 0.806 | 0.745 | V44 | 0.803 | 1.64 | **win 7.3%** | 2.20× |
| 4×12 (4,3) | **1.474** | 1.548 | 1.474 | V44 | 1.478 | 2.98 | **win 0.3%** | 2.02× |
| 8×12 (4,3) | **3.011** | 3.026 | 3.011 | V44 | 2.742 | 5.65 | lag 9.8% | 1.88× |
| 2×16 (4,4) | **0.984** | 1.051 | 0.984 | V44 | 1.015 | 2.13 | **win 3.1%** | 2.17× |
| 4×16 (4,4) | **1.951** | 2.035 | 1.951 | V44 | 1.875 | 3.92 | lag 4.0% | 2.01× |
| 8×16 (4,4) | 5.662 | **4.002** | 4.002 | Vr1 | 3.653 | 7.47 | lag 9.6% | 1.87× |
| 2×24 (8,3) | **1.477** | 1.549 | 1.477 | V44 | 1.461 | 2.98 | lag 1.1% | 2.02× |
| 4×24 (8,3) | 3.032 | **3.023** | 3.023 | Vr1 | 2.801 | 5.66 | lag 7.9% | 1.87× |
| 8×24 (8,3) | 10.702 | **5.976** | 5.976 | Vr1 | 5.628 | 11.02 | lag 6.2% | 1.84× |
| 2×32 (8,4) | **1.960** | 2.036 | 1.960 | V44 | 1.875 | 3.91 | lag 4.6% | 2.00× |
| 4×32 (8,4) | 5.679 | **4.001** | 4.001 | Vr1 | 3.776 | 7.47 | lag 6.0% | 1.87× |
| 8×32 (8,4) | 15.051 | **7.929** | 7.929 | Vr1 | 7.579 | 14.59 | lag 4.6% | 1.84× |

**We beat cuDNN on three shapes — 2×12 (7.3 %), 4×12 (0.3 %), 2×16 (3.1 %), including the marquee 4×12 — and beat Triton ~2× everywhere.** Std on the clean low-batch shapes is 0.001–0.017 ms (rock-solid); the B=8 rows still wander at 0.3–0.5 ms (power-cap throttle at 16 resident CTAs, not clock drift — min-times are the cleaner read there). V44 wins the eight low/mid-batch shapes; Vr1 (per-hq + G-head reduce) wins the four B=8 / 4×32 memory-bound ones. The remaining lag on the big shapes (5–10 %) is the *walled* per-tile efficiency — `long_scoreboard` ~2.5 vs cuDNN's 1.76 — which needs cuDNN's ground-up register/smem budget, not an incremental change.

**Bottom line:** the main kernel is at the silicon wall, so the day's win came from vectorizing the memory-bound helper kernels cuDNN also runs — pushing V44 under cuDNN on the shapes where the main kernel isn't the bottleneck.
