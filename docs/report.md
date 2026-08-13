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

---

# dQ store lever — full investigation (2026-08-12) → **CLOSED**

Target: the **4×12 (B=4)** shape where best-of lags cuDNN ~5% (V44 1.54 ms vs cuDNN 1.28 ms). Hypothesis carried into the day: the gap is cuDNN's **conflict-free dQ scatter store** — V44's swizzled scalar store has **27.6M** shared bank conflicts (5.3-way), cuDNN's has **207K** (from `reports/cudnn_bprop_full.ncu-rep`). This section records the arc and its **definitive negative result: the store conflicts are not the gap.**

## What cuDNN's dQ store actually is (SASS-decoded)
- **Mechanism = TMA-reduce into an L2-resident `dq_accum`** (`cp.reduce.async.bulk.tensor`, `lts__t_sectors_op_red` 51.9M at 97% L2-hit), **not** atomics (1668) and **not** a separate reduce kernel. Identical class to V44's reduce. The *only* difference is the smem store that feeds it.
- The conflict-free store **irreducibly needs a cross-lane rearrange**: the mma-C fp32 fragment gives each lane a 2×2 block (r0,c),(r0,c+1),(r1,c),(r1,c+1) = 2 rows × 2 cols, but a coalesced store wants 4 contiguous cols of one row per lane → each lane must fetch its neighbor's 2 cols. cuDNN's SASS does exactly this (`STS.128` coalesced at `R11=(tid&31)*16` + a small cross-lane exchange + a multi-D descriptor un-map).

## The store experiments (all bit-identical output, H200-measured, reverted)

| version | store change | median | store conflicts | why it lost |
|---|---|---|---|---|
| **V44** (baseline) | swizzled scalar (5.3-way) + 2D SW128B reduce | **1.54 ms** | 27.6M | — (the reference) |
| V4b | STS.128 via shfl-gather to swizzled slots | 1.89 ms | 53M | STS.128 to swizzled fp32 mis-coalesces; 16 lanes idle |
| V4c | scratch-transpose → coalesced STS.128 → NONE reduce | 2.34 ms | 27.4M | scratch = 2nd store pass (98M wf); swizzle kept r0/r1 alias |
| V4d | 3D de-alias permutation | 1.75 ms | 78.5M | de-alias scattered the coalescing |
| V4f | padded-scratch (stride-65) transpose | 2.66 ms | 78.5M + 25.5M ld | stride-65 breaks 16B align → forced scalar LDS |
| V4e | dQ via a **separate** reduce kernel | ~4 ms | — | 6.4 GB partials, bandwidth-dead |
| V5q | full **Q-parallel** rewrite (dQ one-writer) | 3.08 ms | 53.9M | doubles the scatter — dk+dv both reduce (2× work) |
| **V5c** | **all-32-lane shfl store + 5D TMA-reduce** | **2.78 ms** | **2.05M** | **store fixed — but 5D reduce = 14.9 GB L2** |

## V5c — the decisive one
V5c is the *correct* replication of cuDNN's store, validated in isolation first: `precision/tma_frag_probe32.cu` (single 64×64 tile, all 32 lanes active — even→r0 float4→s0, odd→r1→s1, `pk=laneHi*2+colGroup`, coalesced STS.128, 5D reduce) → **bit-exact + 0 shared store conflicts** on H200. Wired into the kernel it stays bit-identical and drops kernel store conflicts to **2.05M — 13× fewer than V44, approaching cuDNN's 207K.** The store fix unambiguously landed.

**And the kernel got 80% slower (2.78 ms).** Profile (`reports/v5c2_raw.csv`): `lts__t_bytes` = **14.9 GB** — the 5D descriptor's 16-byte strided innermost box fragments the global reduce into scattered transactions, vs V44's compact 2D swizzled reduce (SW128B box 32×64, 2×128B atoms).

## Conclusion — store conflicts are NOT the V44↔cuDNN gap
- V44 runs 1.54 ms **carrying 27.6M store conflicts**; a near-zero-conflict store (2.05M) made the kernel **worse, not better.** The conflicts cost essentially nothing.
- The store *rearrange* is cheap (shfl, 0 conflicts — the earlier "shfl-bound" worry was wrong). But a conflict-free layout **forces** a 5D strided descriptor, and that reduce is far costlier than V44's 2D swizzled one.
- **V44's tradeoff — eat the 5.3-way store conflicts, keep the cheap 2D reduce — is the better one.** Every alternative store either (a) needs a 2nd pass (V4c/V4f, 98–150M wf) or (b) needs the expensive 5D reduce (V5c, 14.9 GB L2).
- The prior "dq store *is* the whole gap" claim (from V4a = 1.286 ms, which removed the store **+ reduce + gemm** together) conflated three things. V5c isolates the store alone and clears it.

**Do not revisit any conflict-free-dQ-store idea.** V44/V45 remains the deliverable. If the ~0.26 ms gap is pursued further it is **not** the dq store — it would need a fresh V44-vs-cuDNN head-to-head on compute/stall metrics, or accepting cuDNN hides its reduce cost via whole-pipeline structure we don't replicate.
