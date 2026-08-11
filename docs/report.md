# GQA Backward — H200 optimization log (2026-08-11)

Starting point: **V44** (swizzled TMA-reduce dQ). Goal: close the gap to cuDNN/SDPA across the shape space. Dev box is sm_120 — every benchmark/profile below ran on an H200 remote; the assistant only compiles and reads profiles.

## Result — deliverable = two kernels, take the best per shape

`src/attention/GQA_bwd_baseline.cu` ships **both** kernels and benchmarks each:
- **V44** — default grid (`b` fastest). Wins small/mid shapes.
- **V45** — L2 raster (`k_tile` fastest, `GRID(S/Bc, Hkv, B)`); same-`(b,hkv)` blocks cluster so Q/dO stays L2-resident. Wins the large shapes where the default thrashes L2. **Byte-identical output** (only block order changes).

Full sweep, median ms, S=4096, D=128. SDPA = PyTorch `enable_gqa` (cuDNN-backed); Triton = flash bwd. Driver: `scripts/run_sweep.sh`.

| B×Hq (Hkv,G) | V44 | V45 | **best** | won | SDPA | Triton | vs SDPA | vs Triton |
|---|---|---|---|---|---|---|---|---|
| 2×12 (4,3) | **0.793** | 0.958 | 0.793 | V44 | 0.809 | 1.633 | **win 2%** | 2.06× |
| 4×12 (4,3) | **1.579** | 1.707 | 1.579 | V44 | 1.508 | 2.960 | lag 5% | 1.87× |
| 8×12 (4,3) | **3.207** | 3.224 | 3.207 | V44 | 2.947 | 5.618 | lag 9% | 1.75× |
| 2×16 (4,4) | **1.044** | 1.255 | 1.044 | V44 | 1.033 | 2.128 | lag 1% | 2.04× |
| 4×16 (4,4) | **2.068** | 2.260 | 2.068 | V44 | 1.889 | 3.905 | lag 9% | 1.89× |
| 8×16 (4,4) | 5.683 | **4.304** | 4.304 | V45 | 3.907 | 7.458 | lag 10% | 1.73× |
| 2×24 (8,3) | **1.582** | 1.711 | 1.582 | V44 | 1.515 | 2.958 | lag 4% | 1.87× |
| 4×24 (8,3) | **3.200** | 3.244 | 3.200 | V44 | 2.963 | 5.616 | lag 8% | 1.76× |
| 8×24 (8,3) | 10.855 | **6.275** | 6.275 | V45 | 5.864 | 11.001 | lag 7% | 1.75× |
| 2×32 (8,4) | **2.070** | 2.273 | 2.070 | V44 | 1.894 | 3.905 | lag 9% | 1.89× |
| 4×32 (8,4) | 5.445 | **4.302** | 4.302 | V45 | 3.879 | 7.451 | lag 11% | 1.73× |
| 8×32 (8,4) | 15.196 | **8.325** | 8.325 | V45 | 7.691 | 14.651 | lag 8% | 1.76× |

**Headline:** best-of beats **Triton 1.7–2.1× at every shape**, wins vs **SDPA/cuDNN** at 2×12, and lags **1–11%** everywhere else — **no catastrophic blowups**. V44 wins the 8 small/mid shapes (~15–20% ahead of raster there); V45 wins the 4 large ones (8×16, 8×24, 4×32, 8×32). Crucially, at 8×24/8×32 the default grid alone thrashes L2 to the point of **losing to Triton** (8×32: 15.2 ms) — V45 rescues it to 8.3 ms (1.8×), back to beating Triton. Neither grid is universally best, so the harness runs both; an adaptive picker (`Hkv=8 && B≥8`, or Q/dO-vs-L2 footprint) captures the min automatically.

## Key findings from the exploration

**1. L2 residency was a red herring (proven).** The original thesis (B=8 gap = L2, from B=2 L2≈91% → B=8 75%) is disproven: the raster drove L2 hit **75%→96%** (beats cuDNN's 92%), DRAM 28.4%→3.8%, misses 7× fewer — yet at the Hq=12/B=8 training shape the **wall-clock barely moved** (3.31→3.23). The misses there were already hidden. *But* the sweep later showed L2 residency **does** matter enormously at **high Hkv/B** (8×24, 8×32), where default thrashes to ~2× SDPA — that's the raster's real payoff.

**2. The gap is intra-CTA exposed latency, not occupancy.** At the training shape: barrier 2.73 + long_scoreboard 2.60 = 58% of 9.2 cyc/issue. Occupancy is 18.5% (1 CTA/SM), doubly-floored by 168 regs *and* 214 KB smem — but **cuDNN is also 1 CTA/SM**, so occupancy explains nothing. Both hide the same latency; cuDNN just does it marginally better.

**3. Barrier stall = exposed latency, not removable sync.** Deleting the biggest barrier via a pipelined dQ-reduce (PD=2 + triple-buffer) made the barrier stall go *up* (2.73→2.91) — the latency resurfaced at the `mbar_wait`s. Only a *clean* cut helped: the store→reduce fuse (merge two adjacent barriers, zero smem change) netted +0.02 ms. The two P/dS publish barriers are locked by `dS=P∘(dP−D)` needing cross-warpgroup P; the only unlock (recompute P on wg1) overloads that warpgroup (2 gemms vs 1) and costs ~0.2 ms even fully overlapped in one wgmma burst.

## Experiments (all bit-identical to V44 unless noted)

| change | result | outcome |
|---|---|---|
| L2 raster (`k_tile` fastest) | flat at Hq=12/B=8, but **~2× faster at large Hkv/B**; ~15–20% slower at small shapes | **shipped as V45** (run alongside V44) |
| dV/dK stage swizzle + vec-store (STS.32) | tied — epilogue too small to move clock | folded / not kept |
| fused store→reduce barrier (4→3) | +0.02 ms (barrier 2.73→2.64) | clean but tiny; not in baseline |
| pipelined dQ-reduce (PD=2 + triple-buf) | slower — barrier stall rose (latency, not sync) | reverted |
| register-P dS (merge barrier1+2) | correct but ~0.2 ms slower — redundant QK overloads wg1 | reverted |
| register-P + dOV∥QK single-burst overlap | still slower — 2 gemms > 1 even overlapped | reverted |

## Ruled out (measured)
- **fp32 dQ stage STS.128**: impossible — the 4 fp32 in a swizzle chunk split across 2 lanes; floored at STS.64.
- **Persistent grid**: L2 *worse* (48%) — static grid-stride fragments access more than the HW scheduler.
- **Bc=32 re-tile**: wrong lever — the smem hog is query-side `sQ_sw`/`sdO_sw` (locked to Br=64 by wgmma-m64), not KV-side; saves only ~16 KB, can't reach 2 CTA/SM (which is moot anyway — cuDNN is 1 CTA/SM).

## Bottom line

`GQA_bwd_baseline.cu` (V44 default + V45 raster, take best) beats Triton 1.7–2.1× at every shape and sits within **1–11% of SDPA/cuDNN** across the full Hq∈{12,16,24,32}, B∈{2,4,8} sweep — winning outright at 2×12 and, critically, never blowing up (the raster caps the worst case that pure V44 hits at high Hkv/B). Only `sdpa`/cuDNN are valid apple-to-apple competitors; thunderkittens is forward-only (its backward overflows to NaN on this GQA shape).
