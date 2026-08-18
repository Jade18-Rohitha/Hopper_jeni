# B=8 GQA-bwd — fresh-start plan & learnings (2026-08-18)

**Goal:** a kernel written *specifically* for B=8 (Hq=12, Hkv=4, G=3, S=4096, D=128, bf16, causal)
that hits **≤ 2.7 ms** — beating cuDNN (2.739 ms) at the one shape we've never won. Build naive →
stack optimizations, each one measured. This doc is the carry-forward of everything already paid for.

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

## 4. The naive → stacked plan (each step MEASURED on SM-busy, not TFLOPs)

**Step 0 — Naive, correct, simple.** One CTA per (b, hkv, k-tile). Each loads its K/V tile, loops
over q-tiles (causal), computes S=QKᵀ, P, dP, dS, dV+=PᵀdO, dK+=dSᵀQ, dQ (atomicAdd or TMA-reduce).
No warp-spec, no TMA, plain loads. Get it green vs the PyTorch ref @2e-2. This is the baseline SM-busy.

**Step 1 — TMA loads + swizzled wgmma.** Replace global loads with 2D TMA into swizzled smem; wgmma
m64n64k16. Measure SM-busy jump.

**Step 2 — Warp-specialized producer/consumer + mbarrier pipeline (PD=2).** One producer warpgroup
streams Q/dO/K/V; consumers compute. This is where latency-hiding starts. **Watch eligible-warps.**

**Step 3 — Persistent + atomic scheduler + causal-pair balance.** Bank the 96% L2 mechanism.

**Step 4 — THE decision that beats cuDNN. Pick ONE, budget smem for it from the start:**
- **(a) 2 CTAs/SM:** shrink the tile/staging to < 114 KB smem (e.g. Br=Bc=32, or D-split staging, or
  bf16 dQ stage saving 32 KB). Doubling resident CTAs doubles warps/scheduler → the direct fix for
  the "3 warps, 0.25 eligible" wall. **This is the most promising untried lever.**
- **(b) Deeper async pipeline at 1 CTA:** PD≥3 Q/dO ring + more in-flight TMA/mbarriers so a consumer
  has multiple q-tiles queued — emittable latency hiding. Needs the smem PD=3 wanted (freed by (a)-style
  staging cuts).

**Step 5 — Only if bandwidth-bound after Step 4:** *then* consider shared-Q/dO to cut traffic. Not before.

**Design rule #2:** after every step, read `sm__throughput` + eligible-warps/scheduler. If SM-busy
didn't move, the step addressed the wrong axis — revert it.

---

## 5. Reusable assets in the current tree

`src/attention/GQA_bwd.cu` has, working and correct: swizzled TMA load/store/reduce helpers
(`tma_load_2d_v4`, `tma_reduce_add_2d_v43`, `make_desc_sw128_*`), the m64n64 wgmma kernels
(`run_gemm_*`), `fused_p_stsm` (softmax+mask+stmatrix), `fuse_dS_ldstsm`, `store_acc_sw128_f32`,
the vectorized convert/drowsum, the persistent atomic + causal-pair scheduler (in Vp1/Vc8), and the
correctness harness (`check()` vs `data/gqa_*.bin`, tol 2e-2). Lift these into the fresh kernel;
don't rewrite them.
