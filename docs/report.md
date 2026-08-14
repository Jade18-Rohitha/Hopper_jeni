# GQA Backward — H200 optimization log (2026-08-13)

I started today at **V44** (swizzled TMA-reduce dQ), ~1.52 ms at 4×12, chasing cuDNN (~1.49 ms end-to-end). The dev box is sm_120, so every benchmark below ran on an H200 remote — I compiled here and read the profiles. I landed two wins, both correct and shipped in `src/attention/GQA_bwd_baseline.cu` (best-of per shape):

- **early-empty** — a barrier-stall cut that makes V44 (and V45) faster on nearly every shape.
- **Vr1** — cuDNN's actual structure (KV-parallel per-hq + a separate G-head reduce kernel), which wins the mid-large memory-bound shapes.

All medians are **locked-clock** (`nvidia-smi -pm 1; nvidia-smi -lgc 1980`) to kill the thermal drift that was inflating the back-to-back sweep.

---

## First half — early-empty (the V44 win)

I noticed V44 signals `empty[s]` — "slot free, producer may refill" — only at the *very end* of each loop iteration, after the dQ store+reduce. But the slot (`sQ_sw`/`sdO_sw`) is fully consumed one step earlier, at the dK/dQ gemm's `te_wait`. I moved the signal there, giving the producer a full store+reduce of extra lead time, and made `empty` a **2-count mbarrier** so each consumer warpgroup's leader signals independently — no added cross-warpgroup sync.

Profile (4×12): **CTA-barrier stall 3.4 → 1.98 cyc/issue** (now at cuDNN's 1.88), **total stall ~10 → 7.9**. This *removed* work rather than relocating it. Output is bit-identical. It shaved **~1–4% off nearly every shape** — reproducible under locked clocks (4×12 V44 = 1.524 ms, std 0.007, vs 1.579 unlocked/thermal).

Best-of V44/V45 (early-empty on both), median ms. SDPA = PyTorch `enable_gqa` bwd (cuDNN-backed); Triton = flash bwd:

| B×Hq (Hkv,G) | V44 | V45 | best | won | SDPA | Triton | vs SDPA | vs Triton |
|---|---|---|---|---|---|---|---|---|
| 2×12 (4,3) | **0.771** | 0.930 | 0.771 | V44 | 0.800 | 1.643 | **win 4%** | 2.13× |
| 4×12 (4,3) | **1.524** | 1.671 | 1.524 | V44 | 1.483 | 2.981 | lag 3% | 1.96× |
| 8×12 (4,3) | **3.135** | 3.166 | 3.135 | V44 | 2.919 | 5.660 | lag 7% | 1.81× |
| 2×16 (4,4) | **1.017** | 1.230 | 1.017 | V44 | 1.013 | 2.135 | ~tie | 2.10× |
| 4×16 (4,4) | **2.031** | 2.215 | 2.031 | V44 | 1.872 | 3.917 | lag 8% | 1.93× |
| 8×16 (4,4) | 5.624 | **4.194** | 4.194 | V45 | 3.777 | 7.480 | lag 11% | 1.78× |
| 2×24 (8,3) | **1.526** | 1.671 | 1.526 | V44 | 1.475 | 2.981 | lag 3% | 1.95× |
| 4×24 (8,3) | **3.139** | 3.162 | 3.139 | V44 | 2.743 | 5.660 | lag 14% | 1.80× |
| 8×24 (8,3) | 10.929 | **6.145** | 6.145 | V45 | 5.768 | 11.022 | lag 7% | 1.79× |
| 2×32 (8,4) | **2.031** | 2.215 | 2.031 | V44 | 1.871 | 3.921 | lag 9% | 1.93× |
| 4×32 (8,4) | 5.629 | **4.196** | 4.196 | V45 | 3.828 | 7.484 | lag 10% | 1.78× |
| 8×32 (8,4) | 15.064 | **8.158** | 8.158 | V45 | 7.615 | 14.600 | lag 7% | 1.79× |

V44 wins the small/mid shapes; V45's L2 raster wins the large ones where the default grid thrashes. (B=8 V44 rows still wander at std 0.2–0.4 ms — that's power-cap throttle at 16 resident CTAs, not clock drift; min-times are the cleaner read there.)

---

## Second half — Vr1 (cuDNN's reduce-kernel structure)

I built cuDNN's actual arrangement: **KV-parallel *per-hq*** (each block owns one query head and streams its q-tiles) + a **separate G-head reduce kernel** (`fmha_reduce_head` equivalent). The main kernel writes per-hq partial dK/dV `[B,Hq,S,D]`; a tiny reduce kernel sums the G heads → `[B,Hkv,S,D]`. dQ stays inline TMA-reduce.

I validated the reduce standalone first: **18.78 µs — identical to cuDNN's `fmha_reduce_head`** (vectorized 128-bit loads; my naive 16-bit version was 50 µs). Then two things I got wrong and then right:
1. The per-hq main kernel needs the V45-style **raster** to keep Q/dO L2-resident. Without it, Vr1 was DRAM-bound (2.4 GB / 2.10 ms); with it, **0.24 GB — 4× *lower* than V44 — and 1.61 ms**.
2. The G-reduce and the dQ convert are independent and neither saturates DRAM, so they should overlap — but my first attempt used a default stream that syncs with the legacy null stream. Fixing it to a **non-blocking stream** overlapped them: ~40% of the reduce hides under the convert, and the saving *scales* with B·Hq (−0.02 ms at 2×12 → −0.11 ms at 8×32).

Best-of-3 V44/Vr1/V45 (early-empty + overlap), median ms:

| B×Hq (Hkv,G) | V44 | Vr1 | V45 | best | won | SDPA | Triton | vs SDPA | vs Triton |
|---|---|---|---|---|---|---|---|---|---|
| 2×12 (4,3) | **0.765** | 0.825 | 0.923 | 0.765 | V44 | 0.802 | 1.645 | **win 5%** | 2.15× |
| 4×12 (4,3) | **1.512** | 1.583 | 1.658 | 1.512 | V44 | 1.461 | 2.979 | lag 3% | 1.97× |
| **8×12 (4,3)** | 3.302\* | **3.095** | 3.134 | **3.095** | **Vr1** | 2.758 | 5.661 | lag 12% | 1.83× |
| 2×16 (4,4) | **1.008** | 1.075 | 1.218 | 1.008 | V44 | 1.018 | 2.134 | **win 1%** | 2.12× |
| 4×16 (4,4) | **2.000** | 2.083 | 2.196 | 2.000 | V44 | 1.898 | 3.915 | lag 5% | 1.96× |
| **8×16 (4,4)** | 5.675\* | **4.097** | 4.158 | **4.097** | **Vr1** | 3.812 | 7.467 | lag 7% | 1.82× |
| 2×24 (8,3) | **1.523** | 1.583 | 1.657 | 1.523 | V44 | 1.487 | 2.981 | lag 2% | 1.96× |
| **4×24 (8,3)** | 3.979\* | **3.096** | 3.135 | **3.096** | **Vr1** | 2.751 | 5.655 | lag 13% | 1.83× |
| 8×24 (8,3) | 10.89\* | 6.123 | **6.088** | 6.088 | V45 | 5.729 | 11.02 | lag 6% | 1.81× |
| 2×32 (8,4) | **2.000** | 2.085 | 2.196 | 2.000 | V44 | 1.885 | 3.913 | lag 6% | 1.96× |
| **4×32 (8,4)** | 5.633\* | **4.098** | 4.157 | **4.098** | **Vr1** | 3.828 | 7.475 | lag 7% | 1.82× |
| 8×32 (8,4) | 15.08\* | 8.118 | **8.078** | 8.078 | V45 | 7.503 | 14.59 | lag 8% | 1.81× |

`*` = V44 thrashing (std 0.3–0.5 ms). **Vr1 wins 4 shapes** (8×12, 8×16, 4×24, 4×32); the 2 it loses (8×24, 8×32) are within **0.04 ms** of V45. Clean regime split: **V44 the 6 low-batch shapes, Vr1 the 4 mid-large, V45 the 2 biggest.** Best-of beats **Triton 1.8–2.2× everywhere** and wins **SDPA at 2×12 (5%) & 2×16 (1%)**.

**What I learned from Vr1:** on small shapes V44 still wins because the reduce's ~37 µs floor can't be beaten when V44's inline G-reduce is free — and crucially, **cuDNN's edge on the small shapes is *not* the reduce structure** (cuDNN pays that floor too). Its advantage is the main kernel's load-latency hiding: **long_scoreboard 1.76 vs my 2.82**. That is the real remaining frontier — pipeline depth, smem-gated — and it's where I'll pick up next.
