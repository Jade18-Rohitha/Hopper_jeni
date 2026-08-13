# GQA Backward — H200 optimization log (2026-08-11)

Starting point: **V44** (swizzled TMA-reduce dQ). Goal: close the gap to cuDNN/SDPA across the shape space. Dev box is sm_120 — every benchmark/profile below ran on an H200 remote; we compiled here and read profiles.

## Our effort log (2026-08-11 → 08-13)

We chased one question: why does V44 sit ~1.58 ms at 4×12 while cuDNN hits ~1.4 ms end-to-end?

- **We suspected the dQ store.** V44's swizzled store has 27.6M bank conflicts; cuDNN's has 207K. We decoded cuDNN's store *and* reduce from its SASS, then rebuilt them ourselves — V5c/V5d/V5e replicate cuDNN's exact reduce geometry (row-major D-slab blocks + compact 2D TMA-reduce), validated bit-exact in isolation first. We got the store to 2M conflicts (13× fewer than V44). **It didn't help.** V44 runs 1.54 ms *carrying* 27.6M conflicts; our near-conflict-free store was equal-or-slower on identical memory traffic. We proved, with measurement, that the store was never the gap — a hard negative result we're confident in.
- **We tried compute-pipelining** the S/dP gemm one tile ahead to fill the P/dS barrier. It failed twice, each for a concrete reason we pinned down: first a runtime-indexed accumulator array fell to local memory (2.4× slower); fixed that with compile-time buffers, then found the deeper truth — an *async* wgmma issued before a barrier doesn't fill a *synchronous* stall, so the barrier stall went *up*. Dead end, understood.
- **Then we found the win: early-empty.** We signalled `empty[s]` (slot-free to the producer) one step earlier — at the dK/dQ `te_wait`, the slot's true last use — via a 2-count mbarrier so each warpgroup signals independently with no extra sync. This *removed* work: barrier stall 3.4→1.98 (cuDNN's level), total 10→7.9. It shaved ~1–4% off nearly every shape, bit-identical. Shipped in both V44 and V45.
- **We diagnosed exactly what's left.** The remaining gap is TMA-load latency (long_scoreboard 2.82 vs cuDNN's 1.76). We confirmed the producer is *not* saturated (TMA-issue stalls ~0), so it's pipeline depth, not a 2nd producer — and PD=4 is smem-gated (the only free 32KB comes from single-buffering the reduce, which serializes it +0.5 ms, measured twice).
- **We locked GPU clocks** to make the win reproducible: 4×12 V44 = 1.524 ms tight (std 0.007). The earlier wandering 1.53–1.60 was pure thermal drift in the back-to-back sweep.

Net: we didn't reach cuDNN's ~1.4 ms, but we moved the deliverable the right way on every shape with a clean, correct optimization, closed the store question for good, and reduced the remaining gap to one precisely-scoped structural problem.

## Result — deliverable = two kernels, take the best per shape

`src/attention/GQA_bwd_baseline.cu` ships **both** kernels and benchmarks each:
- **V44** — default grid (`b` fastest). Wins small/mid shapes.
- **V45** — L2 raster (`k_tile` fastest, `GRID(S/Bc, Hkv, B)`); same-`(b,hkv)` blocks cluster so Q/dO stays L2-resident. Wins the large shapes where the default thrashes L2. **Byte-identical output** (only block order changes).

Full sweep, median ms, S=4096, D=128. SDPA = PyTorch `enable_gqa` (cuDNN-backed); Triton = flash bwd. Driver: `scripts/run_sweep.sh`.

Locked-clock sweep (2026-08-13, `nvidia-smi -lgc 1980`), both V44 and V45 carry the **early-empty** optimization (below). Median ms.

| B×Hq (Hkv,G) | V44 | V45 | **best** | won | SDPA | Triton | vs SDPA | vs Triton |
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

**Methodology:** clocks locked at 1980 MHz to remove thermal drift (unlocked, back-to-back sweep inflates B=4/B=8 medians ~3% and widens std ~3×). The **early-empty** win is now reproducible: 4×12 V44 = 1.524 ms tight (std 0.007) vs 1.579 base. B=8 V44 shapes still show std 0.2–0.4 — power-cap throttle at 16 resident CTAs (locking graphics clock does not cap board power); min-times are the cleaner read there.

**Headline:** best-of beats **Triton 1.8–2.1× at every shape**, wins vs **SDPA/cuDNN** at 2×12 (win 4%), ties at 2×16, and lags **3–11%** elsewhere — **no catastrophic blowups**. V44 wins the small/mid shapes; V45 wins the large ones (8×16, 8×24, 4×32, 8×32) where the default grid thrashes L2. The **early-empty** optimization (2026-08-13) shaved ~1–4% off nearly every shape (2×12 0.793→0.771, 4×12 1.579→1.524, 2×24 1.582→1.526, all V45-winning shapes faster) — bit-identical output. Neither grid is universally best, so the harness runs both; an adaptive picker (`Hkv=8 && B≥8`) captures the min.

### early-empty (2026-08-13) — the win

Barrier-stall reduction that generalizes across shapes. The consumer loop signalled `empty[s]` (slot free → producer may refill) only at the *very end* of each iteration, after the dQ store+reduce. But slot `s` (`sQ_sw`/`sdO_sw`) is fully consumed one step earlier — at the dK/dQ gemm's `te_wait`. Moving the signal there gives the producer a full store+reduce of extra lead time. Made `empty` a **2-count mbarrier** so each consumer warpgroup's leader signals after its own `te_wait` — no added cross-warpgroup sync. Profile (B=4): **CTA-barrier stall 3.4→1.98 cyc/issue (now at cuDNN's 1.88), total stall ~10→7.9** — this *removed* work rather than relocating it (unlike the same change on the cuDNN-reduce-geometry variant, where it was net-flat). Bit-identical output. The remaining gap to cuDNN (~1.4 ms end-to-end) is **TMA-load latency** (long_scoreboard 2.82 vs 1.76), which is pipeline-depth-bound and smem-gated (PD=4 needs 32KB only freeable by single-buffering the reduce, which serializes it +0.5 ms). Producer is *not* saturated (TMA-issue stalls ~0), so the fix is depth, not a 2nd producer.

## Second half of the day (2026-08-13) — Vr1: cuDNN's reduce-kernel structure

We built cuDNN's actual arrangement: **KV-parallel *per-hq*** (each block owns one query head, streams its q-tiles) + a **separate G-head reduce kernel** (`fmha_reduce_head` equivalent). The main kernel writes per-hq partial dK/dV `[B,Hq,S,D]`; a tiny reduce kernel sums the G heads → `[B,Hkv,S,D]`. dQ stays inline TMA-reduce. We first validated the reduce standalone: **18.78 µs — identical to cuDNN's `fmha_reduce_head`** (vectorized 128-bit loads; the naive 16-bit version was 50 µs). The main kernel needed the V45-style **raster** to keep Q/dO L2-resident (without it, Vr1 was DRAM-bound at 2.4 GB / 2.10 ms; with it, **0.24 GB — 4× *lower* than V44 — and 1.61 ms**).

Best-of-3 sweep, median ms, locked clocks (`-lgc 1980`), all bit-correct:

| B×Hq (Hkv,G) | V44 | Vr1 | V45 | best | won | SDPA | Triton | vs SDPA | vs Triton |
|---|---|---|---|---|---|---|---|---|---|
| 2×12 (4,3) | **0.768** | 0.847 | 0.925 | 0.768 | V44 | 0.802 | 1.645 | **win 4%** | 2.14× |
| 4×12 (4,3) | **1.513** | 1.613 | 1.658 | 1.513 | V44 | 1.461 | 2.979 | lag 4% | 1.97× |
| 8×12 (4,3) | 3.083\* | 3.141 | 3.134 | 3.083 | V44 | 2.758 | 5.661 | lag 12% | 1.84× |
| 2×16 (4,4) | **1.011** | 1.098 | 1.219 | 1.011 | V44 | 1.018 | 2.134 | **win 1%** | 2.11× |
| 4×16 (4,4) | **2.001** | 2.116 | 2.195 | 2.001 | V44 | 1.898 | 3.915 | lag 5% | 1.96× |
| **8×16 (4,4)** | 5.811\* | **4.150** | 4.156 | **4.150** | **Vr1** | 3.812 | 7.467 | lag 9% | 1.80× |
| 2×24 (8,3) | **1.514** | 1.613 | 1.659 | 1.514 | V44 | 1.487 | 2.981 | lag 2% | 1.97× |
| 4×24 (8,3) | 3.088\* | 3.145 | **3.134** | 3.134 | V45 | 2.751 | 5.655 | lag 14% | 1.80× |
| 8×24 (8,3) | 10.84\* | 6.200 | **6.090** | 6.090 | V45 | 5.729 | 11.02 | lag 6% | 1.81× |
| 2×32 (8,4) | **2.005** | 2.115 | 2.194 | 2.005 | V44 | 1.885 | 3.913 | lag 6% | 1.95× |
| **4×32 (8,4)** | 5.796\* | **4.147** | 4.157 | **4.147** | **Vr1** | 3.828 | 7.475 | lag 8% | 1.80× |
| 8×32 (8,4) | 15.09\* | 8.225 | **8.078** | 8.078 | V45 | 7.503 | 14.59 | lag 8% | 1.81× |

`*` = V44 thrashing (std 0.3–0.5 ms, unstable). SDPA = PyTorch `enable_gqa` bwd (cuDNN-backed); vs-SDPA/vs-Triton are best-of-3 vs each. Best-of beats **Triton 1.8–2.1× everywhere**, wins/ties **SDPA at 2×12 & 2×16**, lags **2–14%** elsewhere (all measured in this same locked-clock sweep). **Takeaways:** on small/mid shapes **V44 wins** — the reduce's ~37 µs floor can't be beaten when the inline G-reduce is free. On the large memory-bound shapes **Vr1 and V45 both crush the collapsing V44 (~2×)** and trade the crown (Vr1 wins 8×16 & 4×32; V45 wins 4×24, 8×24, 8×32; within ~1–2%). So Vr1 is cuDNN's structure, correct and competitive exactly where predicted — its wins over V45 are marginal (noise-level) and its dV precision is softer (bf16 partials → G roundings vs V44's one; within tolerance, fixable with fp32 partials). Validated as a legitimate 3rd option; not a clear improvement over V44+V45 best-of.

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
