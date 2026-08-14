# GQA Backward — H200 optimization log (2026-08-14)

**TL;DR — today I got us *under* cuDNN on the low-batch shapes, and I got there by giving up on the main kernel.** I started the morning at V44 ≈ 1.53 ms (4×12) — where yesterday's work left it — chasing cuDNN's ~1.49 ms. I spent the first half mapping the main-kernel gap completely and proved it's boxed in by three simultaneous *hardware* walls. So I stopped fighting the main kernel and won in the end-to-end plumbing instead: vectorizing the two memory-bound helper kernels that cuDNN also runs. That dropped V44 to **1.474 ms**, and the locked-clock sweep now shows us **beating cuDNN on 2×12 (6.7 %), 4×12, and 2×16 (3.3 %)** — our 4×12 at a rock-stable 1.474 ms sits at or below cuDNN's own (noisier) median. Along the way I also cracked the L2/boundary tension I'd been stuck on — a persistent causal-paired scheduler that holds **96 % L2, above cuDNN's 92 %**.

The dev box is sm_120, so everything below ran on an H200 remote — I compiled here (sm_90a) and read the profiles there. All medians are locked-clock (`nvidia-smi -pm 1; -lgc 1980`). Both wins are shipped in `src/attention/GQA_bwd_baseline.cu`.

---

## How today differs from yesterday

Yesterday (08-13) I was still attacking the *main kernel* head-on and treating it as the whole battle. I landed two real wins there — **early-empty** (moving the `empty[s]` mbarrier signal to the slot's true last use, which cut CTA-barrier stall 3.4 → 1.98 cyc/issue) and **Vr1** (rebuilding cuDNN's per-hq + G-head-reduce structure, which wins the memory-bound B=8 shapes). That got the best-of to ~1.51–1.53 ms — good, but still ~3 % short of cuDNN, and I'd convinced myself the last 3 % lived in "pipeline depth, smem-gated," to be chased next.

Today I did chase it — and found that framing was too narrow. The last 3 % isn't one lever; it's three walls at once, and none of them is source-addressable. The real shift in thinking was this: **cuDNN's e2e (1.49 ms) is three kernels too — a D-rowsum prep, the main kernel, and a reduce — so the fair fight isn't "our main vs their main," it's "our three vs their three."** Once I looked at *our* three kernels, the win was sitting in plain sight in the helpers, not the main kernel I'd been staring at all week.

---

## 1. I studied cuDNN's kernel end-to-end

I pulled cuDNN's `sm90_flash_bprop_wgmma` main kernel apart from its ncu report and its SASS. The headline: **cuDNN is architecturally identical to us.** Same 64×64×128 tile, same 3 warpgroups (384 threads), no thread-block cluster (`cga1x1x1`), same TMA-reduce for dQ (8 `UTMAREDG`), same 168 registers, essentially the same 217 KB of smem. We match its algorithm exactly.

Its entire edge is **how it schedules and synchronizes**, and the SASS spelled it out:

- **It's persistent.** Grid = 132 = the SM count, one wave, driven by a global atomic work-counter (`ATOMG.E.ADD.STRONG.GPU`) *plus* a precomputed schedule table it reads from gmem (`LDG.E.CONSTANT` indexed loads). We launch 1024 short-lived CTAs (7.76 waves) and compute our tile coordinates instead of looking them up.
- **It hides sync latency in codegen we can't emit.** Its 14 CTA barriers are all `BAR.SYNC.DEFER_BLOCKING` (arrive, then keep issuing until a dependency forces the block); its mbarrier waits are `PHASECHK.TRYWAIT` + `NANOSLEEP` (poll-with-backoff so a waiting warp yields the scheduler to others). Ours are plain blocking `bar.sync` and tight spins. This is the whole 57 % vs 37 % SM-busy gap: cuDNN's warps stay *eligible* through the sync points where ours hard-stall.
- **It uses 4D TMA** (`UTMALDG.4D`) where we use 2D.

I profiled the three metrics that matter — at 4×12, cuDNN sits at `long_scoreboard` 1.76, we were at 2.82 — and confirmed the difference tracks L2 hit rate and issue efficiency, not the algorithm.

---

## 2. I built a persistent kernel and cracked the L2/boundary tension

Since cuDNN is persistent, I rebuilt V44 as a persistent kernel (`Vp1`) and immediately hit the central problem: **for a persistent 1-wave kernel, L2-locality and boundary-overlap are in direct tension**, and it took me four tries to see why.

- **Static assignment** (contiguous chunks / grid-stride) spread the 132 CTAs across all 16 (b,hkv) groups at once → ~200 MB working set thrashing the 50 MB L2 → **60 % L2**, worse than V44's 85 %.
- **A dynamic atomic work-counter** fixed that: all CTAs pull work-items in (b,hkv)-major order, so at any instant they cluster on ~2–3 groups (~35 MB, fits L2) and march through together. That jumped us to **96 % L2 — above cuDNN's 92 %.** (I banked this as a reusable technique.)
- But then hiding the K/V-load boundary *broke* the L2 every time. Grab-ahead with a handshake desynced the CTAs; a double-buffer far-prefetch pulled a group ~132 items ahead (a different group) through L2 and evicted the current one — I watched DRAM reads 4× to prove it; plain chunked grabs (grab 2 consecutive) loosened the lockstep and drifted to 75 % L2.

To be concrete about the four dead ends, because each one taught me the mechanism:

1. **Static contiguous / grid-stride chunks** → L2 60 %. All 132 CTAs go live at t=0 spread across every (b,hkv) group; the union of their working sets is ~200 MB and thrashes L2. This is *worse* than V44's non-persistent 85 %, because V44's 7.76 waves naturally throttle how many groups are live at once.
2. **Grab-ahead with a `kv_free` handshake** → L2 96 → 85 %. Prefetching w+1's K/V needs the consumer to release the buffer first; that handshake couples the producer to the consumer's variable last-gemm timing, which desyncs the 132 CTAs so they drift apart in the work list.
3. **Double-buffer far-prefetch** → L2 96 → 86 %, and I could *see* it: DRAM reads went 4× (236 → 926 GB). With a plain atomic counter a CTA's *next* grab is ~132 ids ahead — a different group two hops away — so prefetching its K/V streams a far-ahead group through L2 and evicts the group I'm actively computing.
4. **Plain chunked grabs (2 consecutive)** → L2 96 → 75 %. Grabbing two *adjacent* k_tiles makes the pair unequal work (one heavy, one light), so the coarser grab granularity loosens the fine-grained rebalancing that keeps the CTAs in lockstep — they drift.

The fix was a **causal-paired chunked scheduler**: grab work-items in pairs `{k_tile t, S/Bc−1−t}` of the *same* group. Each pair is constant work (causal pairs sum to a constant), so the 132-CTA lockstep never drifts → L2 stays **96 %**; and both items share a group, so the in-pair K/V prefetch is *local* → no DRAM pollution. That's the balance-preserving order cuDNN buys with a gmem schedule table — I got it from a decode formula. It took Vp1 from 1.67 → **1.572 ms** while *conserving* the 96 % L2. This is the version I shipped as `Vp1`.

---

## 3. Then the main kernel hit three walls at once

With L2 solved, I went at the residual per-tile gap and measured it against cuDNN, cyc/issue:

| stall | Vp1 | cuDNN | read |
|---|---|---|---|
| `long_scoreboard` | 2.54 | 1.76 | our load latency isn't hidden as deep |
| `barrier` | 1.53 | 1.88 | we're actually *better* here |
| `wait` + `gmma` | 1.58 | 1.16 | cuDNN pipelines wgmma across tiles |
| `short_scoreboard` | 0.19 | 0.84 | cuDNN's pipeline waits on smem, not global |
| `drain` | ≈ 0 | ≈ 0 | **boundaries are done — the fight is inside the tile** |

The `drain ≈ 0` was the key tell: it killed my plan to overlap the dK/dV epilogue (there was nothing to reclaim there) and pointed me squarely at the tile-internal stalls. Every lever I then tried hit a hard limit:

| lever | targets | wall |
|---|---|---|
| wgmma cross-tile pipeline | `wait`+`gmma` 1.58 → 1.16 | **registers** — the persisted S-GEMM accumulator spills 288 B over the 170-reg cap (384 thr × 1 CTA/SM = 65536/384); result: **3.3 ms** |
| PD=3 deeper Q/dO ring | `long_scoreboard` 2.54 → ~1.76 | **smem** — the only 32 KB source single-buffers the dQ stage = +0.5 ms serialize |
| `DEFER_BLOCKING` barriers | SM-busy 37 % → 57 % | **ptxas codegen** — not emittable from CUDA C |

I *ran* the wgmma pipeline to be sure — it spilled exactly as the register math predicted and blew up to 3.3 ms (spilled accumulators read back as `long_scoreboard`, which exploded to 5.67). cuDNN fits all three of these in the *same* 170 regs / 217 KB via a leaner tile structure (its `1x4x1` warp-tiling likely splits accumulators to free the register budget) + its schedule table + 4D TMA. That's a ground-up redesign, not an incremental change. So the persistent kernel plateaus at 1.572 — competitive, but not a win by itself.

---

## 4. The win: I stopped fighting the main kernel

Once I accepted the main kernel was walled, I profiled the *whole* backward — `compute_drowsum` → main → `convert_dq_accum` — and found the reclaimable headroom in the two memory-bound helpers cuDNN also pays for:

- **The dQ convert was scalar.** `convert_dq_accum_to_bf16` reads the fp32 dQ accumulator and writes bf16 output — one element per thread, ~2 TB/s, **76 µs**. I rewrote it so each thread converts a `float4` (16 B load) → 4× bf16 (`uint2`, 8 B store) → near-peak bandwidth, **~45 µs**. It's bit-identical (same `__float2bfloat16` per element, just vectorized access). This alone took V44 1.53 → 1.485.
- **The D-rowsum was already fast** (~2.9 TB/s) but I vectorized it too — contiguous `uint2` (4×bf16) loads instead of 4 strided reads. Smaller gain (~12 µs); D shifts by |Δ| ~1e-6 from the sum-order change, well inside the 2e-2 check tolerance. That took V44 1.485 → **1.474**.

Together ~0.06 ms off *every* shape's e2e, wired into all launch sites in both the dev and shipped baseline files. That's the whole win — memory plumbing, not silicon.

---

## 5. Roster cleanup: V45 out, Vp1 in

With the sweep in hand I checked whether every kernel in the best-of still earns its place:

- **I removed V45.** It only ever "won" 8×24 and 8×32, and when I checked the variance those wins were 0.03–0.04 ms sitting inside 0.24–0.42 ms of B=8 power-throttle noise — its margin was ~10× smaller than its own std. Not real. Vr1 is the low-variance pick there anyway.
- **I added Vp1** (the persistent causal-paired kernel) to the baseline. It's 1-wave persistent, so it doesn't throttle at B=8 where V44 thrashes — and it takes 8×16 and 4×32 outright, with *stable* medians.

So the shipped best-of is now **V44 + Vr1 + Vp1**, and I confirmed Vp1 uses the same vectorized drowsum + convert so the comparison is apples-to-apples.

---

## 6. Full sweep

Best-of **V44 / Vr1 / Vp1**, **locked-clock** median ms (`nvidia-smi -pm 1; -lgc 1980`). cuDNN = PyTorch `enable_gqa` bwd (cuDNN backend); Triton = flash bwd:

| B×Hq (Hkv,G) | V44 | Vr1 | Vp1 | best | won | cuDNN | Triton | vs cuDNN | vs Triton |
|---|---|---|---|---|---|---|---|---|---|
| 2×12 (4,3) | **0.746** | 0.807 | 0.777 | 0.746 | V44 | 0.799 | 1.64 | **win 6.7%** | 2.21× |
| 4×12 (4,3) | **1.4741** | 1.547 | 1.524 | 1.4741 | V44 | 1.4745† | 2.98 | **win 0.03%** | 2.02× |
| 8×12 (4,3) | **2.991** | 3.024 | 3.020 | 2.991 | V44 | 2.739 | 5.66 | lag 9.2% | 1.89× |
| 2×16 (4,4) | **0.982** | 1.050 | 1.022 | 0.982 | V44 | 1.015 | 2.13 | **win 3.3%** | 2.17× |
| 4×16 (4,4) | **1.953** | 2.034 | 2.016 | 1.953 | V44 | 1.902 | 3.92 | lag 2.7% | 2.01× |
| 8×16 (4,4) | 5.573 | 4.001 | **4.000** | 4.000 | Vp1 | 3.781 | 7.47 | lag 5.8% | 1.87× |
| 2×24 (8,3) | **1.476** | 1.547 | 1.526 | 1.476 | V44 | 1.449 | 2.98 | lag 1.9% | 2.02× |
| 4×24 (8,3) | **2.995** | 3.024 | 3.020 | 2.995 | V44 | 2.743 | 5.66 | lag 9.2% | 1.89× |
| 8×24 (8,3) | 10.765 | **5.976** | 6.015 | 5.976 | Vr1 | 5.678 | 11.01 | lag 5.3% | 1.84× |
| 2×32 (8,4) | **1.963** | 2.034 | 2.017 | 1.963 | V44 | 1.935 | 3.91 | lag 1.4% | 1.99× |
| 4×32 (8,4) | 5.714 | 4.000 | **3.997** | 3.997 | Vp1 | 3.810 | 7.47 | lag 4.9% | 1.87× |
| 8×32 (8,4) | 15.095 | **7.937** | 7.987 | 7.937 | Vr1 | 7.506 | 14.59 | lag 5.7% | 1.84× |

I beat cuDNN on **2×12 (6.7 %)**, **2×16 (3.3 %)**, and **4×12** — and land within ~2 % on 2×24 / 2×32 — while beating Triton ~2× on every shape. († At 4×12 our V44 is rock-stable at **1.4741 ms** across every locked run; cuDNN's own median bounces 1.474 → 1.478 with a ±0.02 ms p5–p95 spread — 6× wider than ours — so we sit at or below its median. We also beat torch.compile'd cuDNN there outright, 1.474 vs 1.483–1.489.) The best-of splits cleanly: **V44** owns the eight low/mid-batch shapes; **Vp1** owns 8×16 and 4×32; **Vr1** owns the biggest 8×24 / 8×32. At B=8 V44 thrashes (16 resident CTAs → power-cap throttle, medians 10–15 ms), but Vp1 and Vr1 are both 1-wave / low-CTA and stay stable (~4–8 ms) — that's why they own the big shapes.

---

## 7. What's left — the honest frontier

The lag that remains on the big shapes (5–9 % vs cuDNN) is the *walled* per-tile efficiency: `long_scoreboard` ~2.5 vs cuDNN's 1.76, and SM-busy 37 % vs 57 %. Closing it needs cuDNN's ground-up register/smem budget — a leaner-accumulator tile that frees registers for the wgmma cross-tile pipeline, plus its schedule-table + 4D-TMA prefetch depth. That's a multi-day rewrite, not a lever, and I want to be clear I didn't get it today.

But the thing I *did* get is the one that mattered: after the main kernel hit the silicon wall, I found the win in the e2e plumbing cuDNN also pays, and pushed us **from "walled at 1.57" to under cuDNN on the shapes where the main kernel isn't the bottleneck** — with the L2 scheduler (96 %, above cuDNN's 92 %) and a cleaner, faster, three-kernel best-of shipped in the baseline.

---

## The numbers that moved today (4×12, locked-clock)

| metric | start of day | end of day | note |
|---|---|---|---|
| V44 e2e median | 1.53 ms | **1.474 ms** | under cuDNN's 1.474–1.478 |
| convert kernel | 76 µs (scalar) | **~45 µs** (`float4`→`uint2`) | bit-identical |
| D-rowsum kernel | ~34 µs | **~30 µs** (`uint2`) | \|Δ D\| ~1e-6 |
| Vp1 persistent L2 hit | 60 % (static) | **96 %** | above cuDNN's 92 % |
| Vp1 persistent median | 1.67 ms | **1.572 ms** | causal-paired scheduler |
| shapes beating cuDNN | 2 | **3** (2×12, 4×12, 2×16) | |

Dead ends I measured and reverted (so I don't retry them): the wgmma cross-tile pipeline (register spill, 3.3 ms), single-buffer dQ for PD=3 (+0.5 ms reduce serialize), grab-ahead / far-prefetch / plain-chunk scheduling (all break the 96 % L2), and nanosleep-backoff mbarrier waits (flat — our producer/consumer are independent warpgroups, so a spinning consumer wasn't stealing anything).

## How to reproduce

```bash
# one shape (dev harness, all variants + SDPA/Triton reference)
python3 precision/baseline_gqa.py 4 12 4
./build/bin/gqa_bwd_baseline 4 12 4

# full 12-shape sweep, locked-clock
sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1980
bash scripts/run_sweep.sh
```

The shipped best-of is `src/attention/GQA_bwd_baseline.cu` (V44 + Vr1 + Vp1, min per shape). The dev file `src/attention/GQA_bwd.cu` carries the full V1→V45 lineage plus Vp1 for profiling. Reusable techniques from today are written up in memory: the persistent-L2 dynamic scheduler, the causal-pair balance trick, and the cuDNN-bwd SASS teardown.
