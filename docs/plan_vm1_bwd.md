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
