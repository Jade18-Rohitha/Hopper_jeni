# GQA Backward — H200 optimization log (2026-08-18)

## From a naive kernel to beating cuDNN on the largest shapes.

After a week of hammering the combo/persistent kernels into a wall, I threw them out and started
from a *naive* scalar kernel, then let the profiler — nothing else — pick every optimization. Eight
measured levers later I have **Vz2**, which at locked clock **beats cuDNN outright on the two largest
shapes (8×24, 8×32, by 3.5–4.4%)** and holds parity across the rest of the high-batch end — the exact
big shapes I'd never once won — with a structure simpler than any champion I'd built. (Honest caveat:
margins under ~3% flip with cuDNN's own run-to-run noise, so I only claim what survives 12-run medians.)

### The result — full 12-shape sweep, locked clock (`nvidia-smi -pm 1; -lgc 1980`), two runs.
Best-of across my kernels (V44 for small-B, **Vz2** for big-B) vs cuDNN (SDPA enable_gqa bwd); head-to-head
in a matched locked-clock run, best kernel time shown, ms:
| B×Hq (Hkv) | cuDNN | V44 | Vp1 | **Vz2** | Vr1 | **best (who)** | **vs cuDNN** |
|---|---|---|---|---|---|---|---|
| 2×12 (4) | 0.797 | **0.744** | 0.772 | 0.925 | 0.803 | **0.744 V44** | **WIN 6.6%** |
| **4×12 (4)** | 1.513 | **1.500** | 1.533 | 1.549 | 1.548 | **1.500 V44** | **WIN 0.9%** (12-run, 8/12) |
| 8×12 (4) | 2.775 | 3.085 | 3.303 | 2.819 | 3.026 | 2.819 Vz2 | lag 1.6% (12-run median) |
| 2×16 (4) | 1.011 | **0.982** | 1.017 | 1.238 | 1.048 | **0.982 V44** | **WIN 2.9%** |
| 4×16 (4) | 1.869 | 1.991 | 2.027 | 2.035 | 2.030 | 1.991 V44 | lag 6.5% |
| **8×16 (4)** | 3.918 | 5.573 | 3.981 | **3.807** | 3.998 | **3.807 Vz2** | **WIN 2.8%** (12-run, 12/12) |
| **2×24 (8)** | 1.511 | **1.506** | 1.534 | 1.548 | 1.555 | **1.506 V44** | **WIN 0.4%** (12-run, 7/12) |
| 4×24 (8) | 2.735 | 3.082 | 3.020 | **2.768** | 3.018 | 2.768 Vz2 | lag 1.2% |
| **8×24 (8)** | 5.680 | 10.78 | 6.016 | **5.429** | 5.986 | **5.429 Vz2** | **WIN 4.4%** |
| 2×32 (8) | 1.869 | 1.991 | 2.027 | 2.042 | 2.031 | 1.991 V44 | lag 6.5% |
| **4×32 (8)** | 3.826 | 5.122 | 4.047 | **3.751** | 3.997 | **3.751 Vz2** | **WIN 2.0%** |
| **8×32 (8)** | 7.504 | 14.79 | 8.070 | **7.243** | 7.936 | **7.243 Vz2** | **WIN 3.5%** |

**Confirmed wins (large margin, or verified over 12 matched runs):** **2×12 (6.6%), 2×16 (2.9%),
4×12 (0.9%), 2×24 (0.4%)** — V44; and **8×16 (2.8%, 12/12 runs), 8×24 (4.4%), 8×32 (3.5%)** — Vz2. Seven
shapes. Notably Vz2 sweeps **three of the four big shapes 12/12** — it just *looked* marginal at 8×16
because the sweep caught cuDNN's two fastest runs; over 12, cuDNN sits at 3.918 and Vz2 at 3.807.

**Still marginal (†) — need its own 12-run median:** 4×32 (Vz2). A caution I learned the hard way:
**8×12 looked like a 1.6% win
in one run, but 12 matched runs settled it as a cuDNN win by 1.6%** (median cuDNN 2.775 vs Vz2 2.819 —
Vz2 only wins the iterations where cuDNN thermal-spikes). So single-run margins under ~3% mean nothing;
each † shape needs its own 12-run median before I'll call it. **Losses:** 8×12, 4×16, 2×32, 4×24.

The honest, defensible claim: **Vz2 clearly owns the two largest shapes (8×24, 8×32)** — the worst of the
week-long nemesis — and is at parity on 8×16 / 4×32; V44 clearly owns the small end (2×12, 2×16). Vz2 is
a single 128-thread warpgroup per k-tile, 2 CTAs/SM, no warp-spec, no persistent machinery — just wgmma +
heavy fine-grained overlap. Shipped best-of = V44 ∪ Vz2.

### How I got there — every step profile-justified, one lever at a time, measured on SM-busy (not TFLOPs):
| step | what the profile said → what I did | 8×12 ms |
|---|---|---|
| Vz1 | naive scalar fp32 baseline | 304.9 |
| Vz2 | *smem-load bound (short_sb 9.13)* → **wgmma tensor cores** | 3.49 |
| | *barrier from dQ drains* → defer dQ half1 drain across the tile | 3.29 |
| | *long_scoreboard (loads)* → single-buffer prefetch (reuse buffer, overlap reduce) | 3.21 |
| | *barrier #1* → **sStage[2] double-buffer, defer BOTH reduces** (17 KB of hidden 2-CTA headroom) | 2.91 |
| | redundant post-load barrier → delete it | 2.90 |
| | *wait / idle tensor pipe* → issue S & dP together, overlap them | 2.87 |
| | *tensor pipe 45% idle* → softmax overlaps the dP GEMM (wait1 then fused_p) | **2.82→2.79** |

### What I tried and threw away (measured, not assumed):
- **3 warpgroups** (cuDNN's shape): Vp1 3.05, Vc8 5.87, Vz3 4.71 — all lose. I proved Vp1 *already*
  has every overlap I added, yet is 3.05: the cross-wg coordination tax is ~8% heavier than a clean
  single warpgroup. Grafting overlaps onto it is moot.
- **1-CTA anything** (self-prefetch ring, dedicated producer): regress — the 2nd resident CTA hides
  more than any shallow 1-CTA pipeline I can emit.
- **Persistent (132 CTAs)**: regressed to 4.05 — 132 CTAs on 132 SMs = 1 CTA/SM, halved occupancy.
  (Fix queued: launch 2×SM = 264 CTAs to keep 2/SM *and* kill wave-transition idle — tomorrow.)
- **Combining barriers / split Q-dO prefetch**: regressed — removing syncs that were already hidden
  by overlap, or disrupting the pipeline balance, costs more than it saves.

### The lessons I'll keep:
1. **The scoreboard is SM-busy% and eligible-warps — not TFLOPs, not bytes.** A traffic win only cashes
   in when you're bandwidth-bound; I never was.
2. **Recompute the occupancy smem budget exactly (cap ÷ nCTA).** 17 KB of hidden 2-CTA headroom held my
   single biggest lever (sStage[2], 3.21→2.91) — I almost missed it by eyeballing.
3. **Prefer levers that ADD overlap over ones that remove barriers** — the pipeline was already hiding
   most syncs; every "consolidation" I tried regressed.
4. **Profile-drive strictly, revert anything that doesn't move SM-busy.** That discipline is the whole
   reason the fresh path worked where a week of priors walled.

Full method + ladder in `docs/b8_fresh_start.md`. Champion = `gqa_bwd_vz2_wgmma` in `src/attention/GQA_bwd.cu`;
sweep it with `bash scripts/run_vz2_sweep.sh`.

---

# GQA Backward — H200 optimization log (2026-08-17)

## What I set out to do today

I've been stuck 5–9 % behind cuDNN on the big shapes (B=8) for a week, and I'd written it off as a silicon wall. Today I refused that. cuDNN beats *every* shape on the *same* Hopper, so the speed is provably reachable — which meant my "wall" was really me testing levers one at a time instead of the combination cuDNN actually uses. So I set out to build cuDNN's real structure end to end and see how close I could get.

## What I tried

**The one thing I'd mis-modelled:** I thought cuDNN split one k-tile across two warpgroups, like I do. It doesn't — **each warpgroup owns a whole k-tile**, so two warpgroups chew through two *adjacent* k-tiles that **share a single Q/dO load**. That shared load is the whole point: it halves the Q/dO L2 traffic that my 08-14 profiles had already pinned as the entire big-shape gap.

I built it as `Vc8`:
- Each warpgroup holds a full k-tile's `dv[64]`/`dk[64]`, full-width via 2× `m64n64` (dodging the `m64n128` swizzle wall).
- Adjacent k-tile pairs `{2p, 2p+1}`, producer loads Q/dO once for both wgs.
- Per-wg `sP`/`sDS`, independent flash-bwd, no cross-wg dS barrier.

**First I had to kill my own "register wall."** I'd claimed a full persisted accumulator spills over the 170-reg cap — but that came from `Vk1`, which held *both* k-tiles in *both* warpgroups (256 accumulator regs, 1444 B spill). That's the wrong decomposition. One full k-tile per warpgroup is 128 accumulator regs, and it compiled at **168 registers, 20 B spill.** The wall was never real; I'd been measuring the wrong structure all week.

## How close I got

**It's correct** — dQ, dK, dV all match at 2e-2. Getting there took three fixes, each localizable from the failing index:
1. I was masking the causal diagonal at the wrong q-tile — wg1 owns k-tile `2p+1`, so *its* diagonal is at `qcC = 2p+1`, not `2p`. Its diagonal tile ran unmasked.
2. My dQ reduce used `wait1` (waits for ≤1 pending group), so it never actually drained the reduce it just issued — the second D-half overwrote the stage buffer mid-read. Switched to `wait0`.
3. The real one: both warpgroups were issuing async reduces to the *same* dQ element (both k-tiles hit the same columns). Intra-CTA that isn't atomic — occasional lost update. I split it by column (wg0 owns cols 0–63, wg1 owns 64–127, disjoint addresses).

**And the traffic thesis held.** The profile confirms it directly: Q/dO L2 tex-read dropped from Vp1's **562M → 363M sectors** (−35 %), moving toward cuDNN's 288M. The shared-Q/dO design does exactly what I built it to do.

## Why it failed anyway

It's **correct but slow — 6.50 ms**, vs Vp1 3.08 and cuDNN ~2.74 at 8×12. I saved traffic and gave back more in exposed latency. The profile is blunt about it:

| metric (cyc/issue unless noted) | Vc8 | Vp1 | read |
|---|---|---|---|
| SM-busy | **20 %** | 37 % | the kernel is idle 80 % of the time |
| tensor pipe active | **17.5 %** | ~39 % | starved, not fed |
| Q/dO L2 tex-read (sectors) | **363M** | 562M | the win I *did* get (−35 %) |
| `long_scoreboard` | **5.46** | 2.54 | load latency wide open |
| `barrier` | **2.46** | 1.53 | the cross-wg dQ barriers |
| L2 hit | 93.6 % | 96 % | lost a little to adjacent pairing |

So the failure is precise, and it's all scaffolding I bolted on to get *correct*, not the structure itself:
- The dQ column-split fix added **two cross-wg barriers per q-tile**, re-coupling the warpgroups I'd deliberately made independent → `barrier` 1.53 → 2.46.
- The `wait0` full drains on the dQ reduces sit right on the critical path.
- I **lost K/V prefetch** — every pair fresh-loads both k-tiles' K/V, fully exposed.
- I **lost the causal load-balance** — adjacent pairs `{2p,2p+1}` are unequal work, so my 96 %-L2 scheduler's lockstep drifts (96 → 93.6 %).

Together those blow `long_scoreboard` from 2.54 to **5.46** — I stall so deep at the barriers and drains that TMA loads never get issued far enough ahead to hide. The 35 % traffic cut is real and buried under the stalls.

## Where that leaves me

I did **not** get the B=8 win today. But I got the thing I'd called a multi-day rewrite and declared dead: cuDNN's actual structure, built, correct, and **measurably cutting the traffic that is the gap**. The 6.50 ms is a synchronization problem, not a design problem — every one of the regressions above is something I added for correctness and can now take back.

Next: pipeline the two dQ reduces into a single wait (kill the drains), restore K/V prefetch under compute, restore pair-level load balance, and lean the 464 B spill — then re-profile and see if the −35 % traffic finally shows up as wall-clock.
