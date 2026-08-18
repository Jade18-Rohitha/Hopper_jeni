# B=8 GQA-bwd — fresh-start plan & learnings (2026-08-18)

**Goal:** a kernel written *specifically* for B=8 (Hq=12, Hkv=4, G=3, S=4096, D=128, bf16, causal)
that hits **≤ 2.7 ms** — beating cuDNN (2.739 ms) at the one shape we've never won. Build naive →
stack optimizations, each one measured. This doc is the carry-forward of everything already paid for.

## RESULT (2026-08-18): Vz2 @ **2.82 ms** — best kernel we've ever built.
Naive → profile-driven → **beats Vp1 (3.05), Vr1 (3.04), V44, V45** and lands **2.9% off cuDNN (2.74)**,
~1170 TFLOPS. `gqa_bwd_vz2_wgmma`: single warpgroup / full k-tile / 2 CTAs / heavy fine-grained overlap.
Full ladder (each step profile-justified, measured, kept only if it moved SM-busy):
| step | change | ms |
|---|---|---|
| Vz1 | naive scalar fp32 | 304.9 |
| Vz2 | +wgmma tensor cores | 3.49 |
| | +defer dQ half1 drain | 3.29 |
| | +single-buffer prefetch (reuse buffer, overlap reduce) | 3.21 |
| | **+sStage[2] double-buffer, defer BOTH reduces** | 2.91 |
| | +delete redundant post-load barrier | 2.90 |
| | +issue S & dP together (overlap on tensor pipe) | 2.87 |
| | +softmax overlaps dP GEMM (wait1 then fused_p) | **2.82** |

**3-warpgroup verdict (probe, conclusive):** all three 3-wg structures regress — Vp1 3.05, Vc8 5.87,
Vz3 4.71. The probe checked Vp1's dQ and found it ALREADY has every overlap we added to Vz2 (deferred
double-buffered reduce via wait1+db, S/dP overlap via D-split, prefetch via producer) — yet it's 3.05.
So the 3-wg coordination cost (cross-wg dS sync + persistent machinery) is inherently ~8% heavier than
a clean independent single warpgroup at 2 CTAs. **3-wg cannot beat 2.82 for us; grafting overlaps is moot.**
The last ~3% to cuDNN is its un-emittable DEFER_BLOCKING/nanosleep sync — a codegen limit, not hardware.

**Method that worked:** strictly profile-driven, one lever at a time, measure SM-busy (not TFLOPs),
revert anything that didn't move it. Biggest single lessons: (1) recompute the occupancy smem budget
EXACTLY (cap/nCTA) — 17 KB of hidden 2-CTA headroom held the biggest lever (sStage[2]); (2) fewer
barriers ≠ faster if they were already hidden by overlap (combining dQ regressed); (3) prefer levers
that ADD overlap over ones that remove syncs.

---

## 0. The one lesson that reframes the whole effort

**We are LATENCY-bound at 1 CTA/SM, not bandwidth-bound.** Every dead end this week traces to
optimizing the wrong axis. The combo (Vc8) *proved* it: it cut Q/dO L2 traffic −35% (562M→363M
sectors) and it made things **worse** (5.87 vs Vp1 3.1), because:

- DRAM read was 3.19 GB over 5.87 ms = **~540 GB/s = 11% of H200's 4.8 TB/s.** Not bandwidth-bound.
- SM-busy **20%**, "no eligible warp" **78%**, only **3 active warps/scheduler**, `long_scoreboard`
  **6.3 cyc = 46%** of the 13.7-cyc issue gap.

So the scoreboard for the fresh kernel is **SM-busy % and eligible-warps/scheduler**, NOT TFLOPs,
NOT bytes moved. A traffic win only cashes in once you're bandwidth-bound — we never were.

**Design rule #1:** budget smem *backwards from occupancy.* Decide "2 CTAs/SM" or "PD≥3 pipeline"
FIRST, then let that cap the tile/staging smem. The combo's fatal move was spending 225 KB (1 CTA/SM,
PD=2) to buy a traffic win we couldn't use. cuDNN and Vp1 are also 1 CTA/SM — so if we stay there,
we must out-hide-latency another way (below); if we can get to 2 CTAs/SM, that alone may do it.

---

## 1. What cuDNN does (the target structure) and its real edge

Architecturally cuDNN ≈ us: 64×64×128 tile, 3 warpgroups (384 thr), no cluster, TMA-reduce dQ, 168
reg, ~217 KB smem, persistent grid=132 (1 wave) off an atomic counter + a gmem schedule table.
**Each consumer warpgroup owns a WHOLE k-tile** (dV[64]+dK[64]=128 accum regs — this FITS, the
"register wall" was a myth from the wrong 256-accum decomposition).

Its edge is **latency hiding we can't emit from CUDA C**: `BAR.SYNC.DEFER_BLOCKING` (arrive, keep
issuing until a dep forces the block) and `PHASECHK.TRYWAIT`+`NANOSLEEP` mbarrier polling. That's
the whole 57% vs 37% SM-busy gap. **The fresh kernel must chase CUDA-C-emittable equivalents:**
deeper async pipelines, more in-flight mbarriers, `cp.async`/TMA everything, so warps stay *eligible*
across sync points instead of hard-stalling on `bar.sync`/tight spins.

---

## 2. Correctness patterns already paid for (don't re-debug these)

1. **Causal diagonal mask** on tile (q-tile, k-tile): apply the diagonal mask when `q_tile <= k_tile`
   the warpgroup OWNS (`qcC <= ktA+wg`), NOT at the loop's qc0. Fully-below tiles are masked to zero
   by the same true-mask; only tiles with `q_tile > k_tile` skip the mask.
2. **`tma_bulk_wait0` (drain), not `wait1`, before reusing a stage buffer.** `wait1` (≤1 pending)
   does NOT wait for the reduce it just issued → the next write races the in-flight read. `wait1` is
   only safe with genuine double-buffering (alternating buffers, Vp1-style).
3. **Intra-CTA same-address async-reduce is NOT atomic.** Two warpgroups of one CTA doing
   `cp.reduce.async.bulk.tensor` to the SAME dQ element → lost updates. Cross-CTA (through L2) IS
   atomic. Fix: split accumulators (per-wg) or split columns (each wg owns disjoint dQ cols).
4. **Only thread-0 issues/waits a reduce — all threads must honor it.** After the wait, a barrier so
   threads 1–127 don't overwrite the stage buffer before thread-0's drain completes.
5. **Epilogue-vs-compute race across drifting warpgroups.** If warpgroups aren't lockstep and the
   epilogue stages dV/dK into a buffer another warpgroup still reads (e.g. a shared Q/dO ring),
   corruption at the LAST tile. Fix: one cross-wg barrier before the epilogue (cheap, per work-item).

---

## 3. Perf learnings — measured, so we don't retry

**Free wins that WORK (keep):**
- Persistent kernel, grid = SM count (132), 1 wave, **dynamic atomic work-counter** in (b,hkv)-major
  order → 96% L2 (beats cuDNN's 92%). This is the L2 mechanism; reuse it.
- **Causal-pair scheduling** for load balance: within a group order k-tiles `{t, WKV-1-t}` so
  consecutive grabs are constant work → no CTA drift. (For single k-tiles per grab. See caveat below.)
- Vectorized helper kernels: `float4→uint2` dQ convert (76→45 µs), `uint2` D-rowsum. Bit-identical.
- Early-empty mbarrier (signal `empty[s]` at slot's true last use) cut CTA-barrier stall 3.4→1.98.

**Measured DEAD ENDS (do NOT retry):**
- **Shared Q/dO one-full-k-tile-per-wg combo** (Vc8): correct, −35% traffic, but LATENCY-bound so it
  regressed to 5.87. Doubling per-consumer work (full k-tile) with the same PD=2 lengthens the serial
  chain (S→P→dP→dS→dV→dK→dQ) and *lowers* eligible warps. Traffic ≠ the gap.
- **Separate dQ accumulators + convert-sum** to decouple wgs: works but +1.6 GB DRAM (2nd accum) and
  only 5.87 (barrier stall traded for drift/L2-loss). Net wash-ish.
- **Chunked grabs (2 pairs/grab)**, even causal-balanced: 5.87→6.25. Coarser granularity loosens the
  132-CTA lockstep → L2 drops. Confirms scheduling is NOT the lever once you're latency-bound.
- **wgmma cross-tile pipeline at PD=3** (from 08-14): register spill (288 B over cap) → 3.3 ms;
  smem for the deeper ring single-buffers the dQ stage → +0.5 ms. Both smem/register-walled at 1 CTA.
- **`DEFER_BLOCKING` barriers / nanosleep-backoff waits**: not emittable from CUDA C.
- Grab-ahead / far-prefetch scheduling: all broke the 96% L2 (drift or far-group eviction).

---

## 4. Method: PROFILE-DRIVEN, no prescribed order (this is the whole point of the fresh start)

We hit the wall last time by applying our *priors* in a fixed order (traffic → warp-spec → persistent
→ combo) regardless of what actually bound the kernel. **New rule: the profile names the next move,
not our habits.** The proven optimizations in §3 are a *toolbox*, pulled ONLY when a profile points
at the stall they fix. Applying them speculatively is how we re-hit the same wall.

**The loop (repeat until ≤2.7 ms):**
1. Profile the current kernel: `--section SpeedOfLight --section WarpStateStats --section SchedulerStats`.
2. Identify the **single top stall / limiter** (SM-busy, the dominant `stalled_*` reason, or the SoL
   compute-vs-memory verdict). Write it down.
3. Apply the **smallest change** that targets *that* limiter — reach into the §3 toolbox only if it's
   the right tool for *this* stall.
4. Re-profile. If SM-busy didn't move, the change addressed the wrong axis → **revert it.**
5. Only after the profile flips from one limiter to another do we pick a different tool.

### Progress log (each step profile-justified)
| ver | change | time | correct | profile said next |
|---|---|---|---|---|
| Vz1 | naive, scalar fp32 matmul, atomicAdd dQ | 304.9 ms | ✓ | smem-LOAD bound (short_scoreboard 9.13, smem-ld pipe 92.68%, DRAM 0.04%, SM 26%) |
| Vz2 | +wgmma tensor cores (TMA as swizzle enabler); same naive 1-CTA/k-tile, no warp-spec/persistent/pipeline | **3.49 ms** (944 TF) | ✓ | *profiling…* |

Context: Vz2's 3.49 ms is already within 15% of tuned Vp1/Vr1 (3.05) and 27% off cuDNN (2.74) —
naive structure, one optimization. Next step is whatever the Vz2 profile names, NOT an assumption.

**Vz2 evolution + MEASURED dead ends (2026-08-18):**
- Vz2 profile: latency-bound, 2 CTAs/SM (98 KB), barrier 2.95 + long_scoreboard 2.17, 0.28 eligible.
- **Small lever — defer dQ half1 drain across q-tile boundary → 3.49→3.29 ms** (KEEP). Confirmed drains
  were a real slice of the barrier stall. Re-profile: barrier 2.95→2.11, long_scoreboard now #1 at 2.51.
- **DEAD END: producer/consumer pipeline (Vz3, 1 CTA, 1 consumer wg) → 4.71 ms.** Dedicated a whole
  warpgroup to producing = idled compute + lost 2nd CTA. Worst of both.
- **DEAD END: self-prefetch 2-slot ring (1 CTA, no idled producer) → 4.60 ms.** Clean test of "1-CTA
  prefetch vs 2-CTA concurrency" — 2-CTA concurrency WINS decisively at this tile size. The +32 KB ring
  drops to 1 CTA and the shallow PD=2 prefetch can't cover the lost concurrency.
- **MEASURED TRUTH: at Br=Bc=64/D=128, stay at 2 CTAs/SM. Do NOT spend smem on rings/producers — the
  2nd resident CTA hides more than any shallow 1-CTA pipeline we can emit.** cuDNN's 1-CTA works only
  via 2 consumer wgs + deep pipeline + un-emittable DEFER_BLOCKING; we can't match that at 1 CTA.
- **Single-BUFFER software prefetch → 3.29→3.21 ms** (KEEP). Issue next tile's Q/dO TMA into the SAME
  buffer right after half1 dK-gemm frees it, overlapping the dQ-reduce. Profile: long_scoreboard
  2.51→1.31 (worked!), barrier rose to 2.72 (now #1).
- **★ sStage[2] double-buffer, BOTH dQ reduces deferred → 3.21→2.91 ms** (KEEP — BIG win). Key smem
  math I'd gotten wrong: cap 232,448 / 2 CTAs = 116,224 per CTA, so +16 KB (sStage[2]) still fits 2
  CTAs (115,216). half0→sStage[0], half1→sStage[1], both committed w/o inline wait0, drained once next
  tile (overlapped a whole tile). Killed the half0 wait0 that blocked behind barriers.
- **★★ Vz2 @ 2.91 ms is now our BEST kernel — beats Vp1 (3.08), Vr1 (3.04), V44, V45.** 6% off cuDNN
  (2.74). Naive-derived, profile-driven, 2 CTAs, single warpgroup. NO ceiling hit — pure stacking.
  LESSON: recompute occupancy smem budget exactly (cap/nCTA), don't eyeball it — 17 KB of 2-CTA
  headroom was hiding the biggest lever.

**Design rule #2:** after every step, read `sm__throughput` + eligible-warps/scheduler. If SM-busy
didn't move, revert. **Design rule #1 (from §0) still governs any smem spend: budget backward from
occupancy** — but only spend it when the profile shows occupancy/latency is the binding limiter.

---

## 5. Reusable assets in the current tree

`src/attention/GQA_bwd.cu` has, working and correct: swizzled TMA load/store/reduce helpers
(`tma_load_2d_v4`, `tma_reduce_add_2d_v43`, `make_desc_sw128_*`), the m64n64 wgmma kernels
(`run_gemm_*`), `fused_p_stsm` (softmax+mask+stmatrix), `fuse_dS_ldstsm`, `store_acc_sw128_f32`,
the vectorized convert/drowsum, the persistent atomic + causal-pair scheduler (in Vp1/Vc8), and the
correctness harness (`check()` vs `data/gqa_*.bin`, tol 2e-2). Lift these into the fresh kernel;
don't rewrite them.
