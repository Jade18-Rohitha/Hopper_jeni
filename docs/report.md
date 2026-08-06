# Today

Started the day on V41 (3.70 ms at B=8) and ended somewhere much more interesting than a single number: **V44 at 3.24 ms median / 3.10 min (1016 TFLOP/s)** at B=8, *and* the discovery that at smaller batch we're already **beating cuDNN**. The deficit isn't a flat gap — it's a batch-scaling effect, and that reframes the whole endgame. All wins bit-identical, 0-spill. (Timings are same-run medians unless noted, min in parens.)

## The win ladder — 3.75 → 3.24 (same-run medians)

**V42 — descriptor-hoist (−4%, → 3.75 / 3.66 min).** The ~24 invariant wgmma operand descriptors were rebuilt every tile because the async-fence `"memory"` blocks nvcc's LICM. Hoisted them to the prologue, const-advances fold to immediates: −80 static instructions, converted ~1:1 to wall-clock.

**V43 — TMA-reduce dQ (−5%, → 3.53 / 3.48 min).** This is where the day turned. We swapped the dQ LSU atomics + smem read-back for `cp.reduce.async.bulk.tensor.add.f32` — moving the dQ accumulation off the contended 64% L1TEX pipe onto the idle TMA engine. First cut *regressed* (single-buffered `wait_group 0` serialized what the atomics hid); double-buffering the staging to overlap the reduce fixed it.

**V44 — swizzled TMA-reduce dQ (−11%, → 3.24 / 3.10 min, 1016 TFLOP/s).** The V43 profile caught a self-inflicted wound: packing the dQ stage stride 72→64 for TMA reintroduced the 8-way bank conflict V24 had killed. Fixed by storing the fp32 fragment into a 128B-swizzled tile with a matching swizzled TMA-reduce descriptor. The swizzle-match was the risk (a software-swizzle store is what silently broke V8), so bit-identical `check()` was the validator — it passed.

## The cuDNN reverse-engineering (and a correction I owed)

Two things I had wrong, both fixed by profiling cuDNN's real kernels instead of arguing on paper:

- **`global_red = 0` does NOT mean "no accumulation."** That counter only sees LSU atomics. cuDNN moves accumulation onto the **TMA engine** (`tma_red` = 3.32 GB for dK/dV), off L1TEX. That's the whole trick behind V43.
- **cuDNN is Q-parallel — dQ one-writer, dK/dV the accumulation side — the mirror of our KV-parallel kernel.** I'd claimed "zero-atomic on both"; the mirror read was right. The `convert_dq` profile settled it: reads 202 MB ≈ exactly 1× the fp32 dq_accum → a single-pass cast, dQ written once.

## The dead ends (honest)

- **bf16 dQ round-trip — regressed.** Halved the bytes, but the 63% pipe counts *wavefronts* not bytes; the scalar read loop was unchanged, and conversions added latency. Wrong currency for a wavefront-bound LSU.
- **persistent-CTA address-math — no-go.** The runtime kv-loop it would optimize is already 4 vector ops; the 562-op bulk is lane-dependent swizzle that can't go uniform. Killed before building.
- **V45 STS.64 store-vectorize — compiler no-op.** Byte-identical SASS to V44; ptxas already auto-fused the adjacent stores. Reverted.

## THE TWIST — we're not universally behind cuDNN

Ran the sweep across batch sizes, and it inverts the framing:

| batch | ours V44 | cuDNN SDPA bwd | result |
|---|---:|---:|---|
| **B=2** | **0.793 ms** | 0.828 ms | **we win +4.4%** |
| **B=4** | 1.579 ms | 1.516 ms | on par (−4.2%) |
| **B=8** | 3.24 ms | ~2.8 ms | cuDNN +16% |

We're *faster* at B=2, even at B=4, and behind only at B=8. Scaling B2→B8: ours **4.09×** (≈linear), cuDNN **3.38×** (sub-linear). cuDNN amortizes better at scale — that's the entire remaining problem, and it's specific to B=8.

**Root cause (confirmed, apples-to-apples, both 384thr/168reg/1-CTA-per-SM):** our grid balloons to **2048 blocks / 15.5 waves** at B=8 vs cuDNN's fixed **132 / 1 wave**. That one-shot grid gives poor temporal locality → **L2 hit 75% vs cuDNN's 92%**, DRAM 28% vs 7%, and at 18% occupancy the miss latency can't be hidden → we're latency-bound at 38% compute vs cuDNN's 59%. Not DRAM-bandwidth (28% util, slack), not compute — it's scheduling → L2 residency.

## Two lines opened for tomorrow

**1. Persistent-grid V45 (the cheap, high-EV B=8 fix — pending H200 gate).** Fixed grid (#SMs CTAs) + grid-stride over the work-items, V44's validated body unchanged — just a reschedule (162 regs / 0 spill / 28 HGMMA = V44's math, bit-identical). First build regressed +47% because the naive `+= gridDim.x` stride *scattered* each CTA across (b,hkv) groups, inverting the L2-reuse it was meant to create. Fixed to **contiguous/blocked** assignment (each CTA drains one group's 64 k-tiles → its ~9 MB set stays L2-resident). Ready to gate at B=2/4/8: watch L2 hit → >85%, waves → 1, DRAM → ~10%. **Caveat to watch:** pure blocked lands all-heavy vs all-light causal k-tiles on different CTAs (~6.6× imbalance, ~56% util) — if L2 recovers but speedup stalls, the fix is strided-within-group k-tile assignment (heavy+light mix per CTA while sharing the resident set). Expected realistic B=8 gain 8–12% → near-parity, keeping the B=2/B=4 wins.

**2. D-collapse (Q-parallel full-D) — feasible on Hopper, correctness on hold.** Scoped and confirmed Hopper-feasible (NOT Blackwell): the M1 isolated consumer proved full-D fits at **232 regs / 0 spill** at cuDNN's exact 384-thr occupancy, and dS stays register-resident (0 LDSM/STSM → the 51% column-split exchange goes to zero). Correctness stuck at dQ: three fixes (transB, async-liveness) killed the overflow but left it finite-wrong; a two-path bisect (proven SS-A vs RS-A on identical dS) proved **both wrong → the bug is upstream in the Q-parallel dS** (causal mask / LSE / D / dP mis-indexed for the flipped KV→Q axis), not the wgmma. That's a precise localization banked — on hold, ready to resume without re-treading the GEMM.

## Where it stands

**V44 = 3.24 / 3.10 at B=8, banked, ~1.16× off cuDNN — but winning at B=2, on par at B=4.** The B=8 gap is a scheduling/L2 problem with a cheap fix queued (persistent-grid V45, H200 gate tomorrow) and a bigger structural one in reserve (the D-collapse, feasible, bug localized). Tomorrow: gate V45 at all three batch sizes, and if the causal imbalance bites, take the strided refinement.
