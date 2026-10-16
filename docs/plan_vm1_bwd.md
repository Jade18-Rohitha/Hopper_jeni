# Plan: Vm1, closing the backward gap to cuDNN (H200)

**Goal:** the backward main kernel matches or beats cuDNN's on all 12 shapes, measured backward-only with the cuDNN
backend forced, `scripts/run_median12.sh`, median of 12. Today Vj1d is 0.15–1.70 ms slower everywhere.

**Why Vj1d loses** (data: [ncu_vj1d_vs_cudnn_B8_Hq18.md](ncu_vj1d_vs_cudnn_B8_Hq18.md)): the helper kernels match
cuDNN's one for one. The whole gap is the main kernel (3.578 vs 2.770 ms at 8×18). It is **limited by L2 bandwidth**
(80% of peak) and moves **2× cuDNN's L2 bytes per key**: Q/dO reloads 10.4 vs 5.4 GB, fp32 dQ reduce-adds 9.8 vs 5.0 GB.

## Step 0: cuDNN's structure, confirmed from SASS (done)

Executed-instruction counts from the two ncu reports (8×18, Hkv=9):

| | Vj1d | cuDNN |
|---|---|---|
| Matrix-multiply work (64×64×16 units) | 47,923,200 | 48,660,480 (+4,608 × 160: one fully-masked step per 128-key work item) |
| Q/dO loads | 4 per q-step per **64 keys** | 4 per q-step per **128 keys** (`UTMALDG.4D`) |
| dQ reduce | 4 × 8 KB per q-step per 64 keys | 8 × 4 KB per q-step per **128 keys** (`UTMAREDG.4D.ADD`) |
| Scheduling | 9,216 blocks (64 keys each), 2 per SM | 132 persistent blocks, atomic work counter (4,740 = 4,608 items + 132) |

**cuDNN's work item = 128 keys = 2 compute warpgroups × 64 keys.** Per query step, the two warpgroups share
one Q/dO load and add their dQ partials together inside the block, then send **one** 32 KB fp32 reduce to L2.
Same tile, same math, half the L2 bytes per key.

## Step 1: Vm1 = Vj1d with 128-key work items (the main lever)

New kernel `gqa_bwd_vm1` in `src/attention/GQA_bwd.cu`. **Vj1d stays frozen** as the shipped baseline.

**Status 2026-10-13: built, waiting for H200 gates.** ptxas: 226 registers, 0 spills, 164 KB shared memory
(Vj1d: 218 / 0 / 115 KB). Wired into `gqa_bwd` (dev bench), `gqa_fwd_bwd` (correctness + 6-run dK/dV
bit-identical check + timing: `OURS bwd Vm1`, `OURS fwd+bwd V62+Vm1`) and `scripts/median_summary.py`.

**H200 result 2026-10-16 (8×12, `gqa_bwd_vm1_test`):** gate 1 PASS: dQ/dK/dV identical error to Vj1d, dK/dV
bit-identical over 6 runs. **Gate 3 FAIL: Vm1 3.559 ms vs Vj1d 2.761 ms (+0.798 ms slower)**, do_bench median of 3
(benchmarkKernel: 3.559 vs 2.784). Vm1 std 0.003 ms vs Vj1d 0.033. Leading hypothesis (unconfirmed): lost
inter-block overlap. Vj1d's 2 independent blocks/SM hide each other's stalls; Vm1's 2 warpgroups are lock-stepped
by 6 block barriers per q-step at 1 block/SM (the Vc8 failure mode). Next: ncu of both kernels at 8×12
(`reports/ncu_vm1_vs_vj1d_B8_Hq12_Hkv4`) to check (a) L2 traffic halved? (b) which stall grew.

**ncu of Vm1 at 8×12 (2026-10-16; the capture caught 2 Vm1 launches, no Vj1d; Vj1d and cuDNN scaled from their 8×18
profiles by 2/3):**

| 8×12 | Vj1d (scaled) | **Vm1** | cuDNN (scaled) |
|---|---|---|---|
| L2 read sectors | 216.2 M | **113.8 M (0.53×)** | 111.7 M |
| L2 dQ reduce sectors | 204.5 M | **103.8 M (0.51×)** | 103.8 M |
| Warp instructions | 591.9 M | **779.4 M (+32%)** | 539.5 M |
| long_scoreboard (cycles/issue) | 1.70 | **0.39** | 1.78 |
| barrier | 1.53 | **2.24** | 2.21 |
| short_scoreboard | 0.65 | **1.23** | 0.79 |

- **Gate 2 PASSES:** L2 traffic halved, exactly to cuDNN's level. The 128-key structure is right.
- **Cost 1, +188 M instructions (+32%):** R2UR +74 M, IMAD +67 M, LOP3 +21 M, IADD3 +15 M. Vm1 indexes smem and wgmma
  descriptors by the runtime `wg`, so every HGMMA recomputes its descriptor and moves it to uniform registers.
  (Vj1d: all compile-time constants.)
- **Cost 2, serial dQ combine:** ~20% of stall samples sit right after the block barriers around the combine
  (WG1 idle while WG0 stores, WG0 idle while WG1 adds).

**Vm2 / Vm3 built (2026-10-16), one run measures both fixes separately:**
- **Vm2** = Vm1 with the per-warpgroup body templated on the WG index (fix 1). SASS: R2UR 107 → 16 (Vj1d: 6).
  218 regs (= Vj1d), 0 spills, 194 KB smem.
- **Vm3** = Vm2 + parallel dQ combine (fix 2): both WGs store at once into stage A / stage B, then all 256
  threads sum B into A. Same 218 regs / 0 spills / 194 KB.
- `gqa_bwd_vm1_test` now checks and times Vj1d, Vm1, Vm2, Vm3 (round-robin, do_bench median of 3) + dK/dV
  bit-identical check for each Vm.

**H200 result 2026-10-16 (8×12):** all correct, all bit-identical dK/dV.

| Kernel | do_bench ms | vs Vj1d | step effect |
|---|---|---|---|
| Vj1d | 2.789 | — | |
| Vm1 | 3.561 | +0.772 | |
| Vm2 (compile-time WG addressing) | 3.123 | +0.334 | −0.437 ms |
| Vm3 (+ parallel dQ combine) | 3.104 | +0.315 | −0.019 ms |

Fix 1 was real; fix 2 barely mattered, so the barrier stall around the combine wasn't the combine's own cost.
Leading hypothesis (unconfirmed): the two WGs run in **lockstep**: both on the tensor cores at once, then both
in softmax at once, whereas Vj1d's 2 independent blocks/SM drift apart and overlap ALU with tensor work. Next:
ncu Vj1d + Vm3 at locked 1980 MHz in one report (`ncu_vm3_vs_vj1d_B8_Hq12_Hkv4`). If confirmed → **Vm4**:
ping-pong staggered WGs (WG1 half a step behind), double-buffered Q/dO with mbarrier handoffs, and a dQ
handoff where WG0 stores + signals and WG1 adds + issues the single reduce (no block barriers).

**Cleanup 2026-10-16 (user: remove versions that don't work out, keep everything up to Vj1d):** Vm1 and Vm2
removed from the code (`GQA_bwd.cu`, the test, the `gqa_fwd_bwd` chain, `median_summary.py`). Vm3 kept as its
own kernel `gqa_bwd_vm3` (no template flag) only as the base for Vm4; it goes too if it never beats Vj1d.
Nothing new is in the sweep until a standalone win.

**ncu Vj1d vs Vm3, same report, 8×12, ~1.82 GHz (2026-10-16, `ncu_vm3_vs_vj1d_B8_Hq12_Hkv4`):**

| | Vj1d | Vm3 |
|---|---|---|
| Duration | 2.421 ms | 2.816 ms (+0.395) |
| L2 reads / reduce-adds | 216 M / 204 M | 114 M / 104 M (halved) |
| L2 throughput | 79% | 36% |
| Instructions | 615.7 M | 649.9 M (+5.5%) |
| Tensor pipe | 43.9% | 38.4% |
| long_scoreboard / short_scoreboard / mio_throttle / barrier | 1.38 / 0.74 / 0.44 / 1.57 | 0.79 / 1.34 / 0.72 / 2.03 |

The load-wait saving (−0.59) is eaten by shared-memory-pipe stalls (+0.88). SASS attribution: the dQ combine
(`A += B`, LDS.128 → STS.128) = **18.6% of Vm3's stall samples**, the doubled dQ staging stores (STS.64) another 10%.
cuDNN per 128-key step: 32 KB smem stores + 32 KB loads, 18.6 BAR.SYNC, 113 mbarrier ops; Vm3: 96 KB + 64 KB,
65 BAR.SYNC, 10 mbarrier ops, 64 fence + MEMBAR.ALL.CTA.

**Vm4 (= Vm4a) built:** dQ D-half h owned by WG h; the non-owner stores its partial once, the owner adds in place
(read-modify-write) and issues that half's reduce. 96 KB smem/step (Vm3 160 KB), 4 block barriers/step (Vm3 6),
proxy fence only in the owner, no stage B. 218 regs, 0 spills, 162 KB smem. In `gqa_bwd_vm1_test` next to Vm3.
If it still loses and the stalls move to barriers: **Vm4b** = mbarrier handoffs instead of block barriers +
double-buffered Q/dO (uses the freed 32 KB), i.e. cuDNN's sync style.

**H200 result 2026-10-16 (8×12):** Vj1d 2.734, Vm3 3.088, **Vm4 3.157 ms** (all correct, dK/dV bit-identical).
Vm4 moved 40% less smem data but got slower: under lockstep the owner idles during the non-owner's store and
the non-owner has nothing to overlap with the owner's add. The combine's cost is critical-path time, not smem
bandwidth (same conclusion as Vm2→Vm3). **Vm4 removed** (lost to Vm3). Vm3 remains the only new kernel.

| | Vj1d | Vm1 |
|---|---|---|
| Grid | (S/64, Hq, B) | **(S/128, Hq, B)** |
| Block | 128 threads (1 warpgroup) | **256 threads (2 compute warpgroups)**; thread 0 still issues TMA |
| Keys per block | 64 | **128**: WG0 keys [128p, +64), WG1 keys [128p+64, +64) |
| Q/dO per q-step | loaded per 64 keys | **loaded once, used by both warpgroups** |
| dQ per q-step | one 32 KB fp32 reduce per 64 keys | **WG0 + WG1 partials summed in shared memory, one 32 KB reduce per 128 keys** |
| dK/dV | one writer per 64 keys + greduce | unchanged: each warpgroup writes its own 64 keys, so dK/dV stay **bit-identical** |
| Per-warpgroup pipeline | | **exactly Vj1d's** (S/dP issued together, deferred reduces, prefetch) |

Details:
- **Shared memory (~195 KB, 1 block/SM):** K+V for 128 keys 64 KB; Q+dO 32 KB; per-warpgroup sP/sDS 2 × 16 KB;
  dQ staging 2 × 32 KB (double-buffered, as Vj1d); LSE/D rows.
- **Registers:** same per-thread work as Vj1d (220 regs, 0 spills). At 256 threads the cap is 255, so it should fit
  without spills. Warps per SM stay at 8, same as Vj1d today (2 blocks × 4 warps).
- **Causal diagonal:** at q-step 2p, WG1's keys are fully masked. It skips the math but still joins the barriers
  (cuDNN pays the same cost).
- **Synchronization:** block-wide barriers only where the two warpgroups really share something (Q/dO buffer
  free, dQ combine). Everything else uses per-warpgroup named barriers (`bar.sync id, 128`) so the two don't lock-step.
- **dQ combine, simplest correct version first:** WG0 stores its partial to staging → barrier → WG1 adds its
  partial in place → barrier → thread 0 issues the reduce. Optimize only if the profile shows the barrier stall.

**Gates (all four before moving on):**
1. **Correctness:** dQ/dK/dV vs reference (2e-2) on all 12 shapes plus Hq=18 (Hkv 9 and 6); dK/dV bit-identical
   across 8 runs (`precision/vj1d_determinism_probe.cu` pattern).
2. **Traffic (ncu, 8×18 Hkv=9):** L2 read + reduce sectors ≈ **0.5× Vj1d** (≈ cuDNN's 167.6 M / 155.7 M).
   If this doesn't halve, the structure is wrong; stop and fix before tuning anything.
3. **Speed, quick check:** main kernel time vs Vj1d and cuDNN at 8×18 and 8×12 (ncu, locked 1980 MHz).
4. **Speed, the claim:** `run_median12.sh` with Vm1 added, backward-only vs cuDNN forced. No "win" without it.

**Main risk:** Vc8 (2026-08) also had 2 warpgroups sharing Q/dO, and it was latency-bound (5.87 ms). Differences
this time: Vm1 keeps Vj1d's proven per-warpgroup pipeline and register budget (Vc8 was cloned from the 3-warpgroup
Vp1 with a 168-register cap), and it keeps 8 warps/SM. If gate 2 passes but gate 3 shows long_scoreboard or barrier
stalls taking over, go to step 2.

## Step 2: hide latency in Vm1 (only if gate 3 says so)

Pick by ncu evidence, one change at a time, re-measure after each:
- **Double-buffer Q/dO** (full/empty mbarriers per stage) so the two warpgroups don't wait on each other's last
  read before the next load. Shared memory is tight (~195 + 32 KB); free space first if needed (e.g. reuse
  staging between halves).
- **Make the dQ combine cheaper:** split column ownership (WG0 sums cols 0–63, WG1 cols 64–127, each reduces its
  half), or `red.shared.add.f32`, to cut a barrier per q-step.
- **Wgmma pipelining** (`wgmma.wait_group 1` where the dependency allows), as cuDNN does.

## Step 3: persistent scheduling (only if the profile shows a wave tail or imbalance)

cuDNN runs 132 persistent blocks with an atomic counter. Vm1 has S/128 × Hq × B blocks (4,608 at 8×18 = ~35 waves).
Causal work per block varies 64:2, so check the tail first. If it's significant: persistent blocks + a
heaviest-first order (see `reference_persistent_l2_scheduler`). Note Vz3 (persistent) gave no gain earlier, so
this needs evidence first.

## Step 4: small items

- 4D TMA (cuDNN loads both D-halves of a tile with one `UTMALDG.4D`; we issue 2): fewer TMA instructions, low priority.
- Ship: if Vm1 wins the median12, make it the deterministic baseline (`GQA_bwd_baseline.cu`) and update the docs.

## Separate track (later): forward V62

- 0.07–0.26 ms behind cuDNN at large B×Hq (4×32, 8×16, 8×24, 8×32).
- `cudaGetDeviceProperties` on every launch (host cost).
- +0.37 ms forward→backward chain overhead only at B×Hq = 128 (8×16, 4×32): nsys trace needed.

## Measurement rules (from the 2026-10 baseline mistake)

- Every comparison states: forward / backward / both, backend forced, timing method.
- Kernel vs kernel: ncu at locked clock (`nvidia-smi -i 3 -lgc 1980`, ncu `--clock-control none`).
- End to end: `run_median12.sh` (same inputs, do_bench-style timing, interleaved, 12 runs).
- Correctness and determinism checked before any timing is reported.

**Vm5 built 2026-10-16 (option A, user's choice): the two WGs DECOUPLED, cuDNN's sync style.** 256 threads
(a separate loader warpgroup was tried: ptxas caps registers per warpgroup, 384 threads -> 168 regs + 1.4 KB
spills even with setmaxnreg, matching V62's probe4 note). No block barriers in the loop: Q/dO/LSE/D
double-buffered on mbarriers; "second one does it" handoffs for both the stage refill (second WG to release a
stage issues its next load) and the dQ combine (first WG stores + arrives, second adds + issues the single
reduce), 4-slot role counters, double-buffered exchange buffers, one-time half-step stagger. Bounded mbarrier
waits trap instead of hanging (printf removed: a call serialized every wgmma). 246 regs, 0 spills, 231 KB smem.
ptxas still injects one warpgroup.wait (C7517); locate if Vm5's numbers make it worth tuning.

**Next idea (user, 2026-10-16): fuse more, keep intermediates in registers, cut smem traffic.** Concrete form
= FA3's backward layout: compute S^T = K·Q^T (keys = M rows of each WG's tile) so P^T and dS^T are already
wgmma register-A fragments: dV += P^T·dO and dK += dS^T·Q read registers, P never goes to smem, only dS is
stored (for dQ). Evidence: cuDNN does 36 STSM per 128-key step vs our 64, and Vm3 lost its L2 gain to
smem-pipe stalls (short_sb + mio +0.88). Plan: test it on Vj1d first (**Vj2**) to isolate the lever, then
carry it to the Vm line.
