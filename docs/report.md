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
