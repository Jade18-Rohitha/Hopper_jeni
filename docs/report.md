# GQA Backward — H200 champion board & daily log

> Two sections, both refreshed each day: the **clear-winners board** (the current best kernel per
> shape, locked in and only replaced by a strictly faster verified median) and **today's report**
> (what I tried, how close I got, why it did or didn't land). S=4096, D=128, bf16, causal.
> Reference = **cuDNN backward-only, backend forced** (`SDPBackend.CUDNN_ATTENTION`, forward run once outside
> the timer, time `autograd.grad` only). Locked clock: `nvidia-smi -i <gpu> -pm 1; -lgc 1980` — on the H200
> box the user's GPU is **index 1** (GPU 0 runs another job); runs before 2026-10-16 locked index 3, so the
> lock never applied (SM ran ≤ 1830 MHz). From 2026-10-16 the decision metric is **SUSTAINED** timing
> (back-to-back ~3 s, training-like) + energy per backward; do_bench is kept as a secondary number.

---

## Today — 2026-10-20: Vg1 vs cuDNN, all 18 shapes — Vg1 faster on every shape, every run (fwd, bwd, fwd+bwd)

Vg1 = the colleague's Hopper GQA forward + backward (see 2026-10-19). All numbers: causal, bf16, S=4096, D=128, cuDNN
backend forced, same inputs, interleaved runs, **median of 25 runs per shape**, SM clock locked at **1530 MHz**.
All correctness checks passed on all 18 shapes (Vg1 O, LSE, dQ, dK, dV vs the PyTorch reference, tol 2e-2).

Sources (two runs, same clock, combined):
- 11 shapes (2×12 … 4×18:9): `reports/median25_gpu1_1530MHz_20261019_200003.log`, host p032 GPU 1 — clock held
  **100%** (2170/2170 loaded samples, no throttling).
- 7 shapes (8×18:9 … 8×32): `reports/median25_gpu4_1530MHz_20261020_112558.log`, GPU 4 — clock held **99.6%**
  (2565/2576 samples; 11 dipped as low as 1305 MHz under the power cap, so the strict RULE CHECK reports a miss).
- Each row is one run on one GPU; cuDNN and Vg1 in a row were measured together. Rows from the two GPUs should not be
  compared with each other in absolute ms.

**1. Forward — Vg1 DERIVED** (Vg1 fwd = Vg1 fwd+bwd − Vg1 bwd, per run; the measured Vg1 fwd line was lost to the sweep bug, see below)

| B×Hq (Hkv) | run | cuDNN (ms) | **Vg1** (ms) | Δ = cuDNN − Vg1 (ms) | Vg1 vs cuDNN | wins |
|---|---|---|---|---|---|---|
| 2×12 (4) | GPU 1 | 0.247 | **0.161** | +0.086 | -34.7% | 25/25 |
| 4×12 (4) | GPU 1 | 0.440 | **0.321** | +0.118 | -26.9% | 25/25 |
| 8×12 (4) | GPU 1 | 0.828 | **0.636** | +0.192 | -23.2% | 25/25 |
| 2×16 (4) | GPU 1 | 0.311 | **0.215** | +0.096 | -30.7% | 25/25 |
| 4×16 (4) | GPU 1 | 0.531 | **0.426** | +0.105 | -19.7% | 25/25 |
| 8×16 (4) | GPU 1 | 1.088 | **0.850** | +0.239 | -21.9% | 25/25 |
| 2×18 (6) | GPU 1 | 0.320 | **0.242** | +0.078 | -24.3% | 25/25 |
| 4×18 (6) | GPU 1 | 0.591 | **0.479** | +0.111 | -18.9% | 25/25 |
| 8×18 (6) | GPU 1 | 1.216 | **0.954** | +0.262 | -21.5% | 25/25 |
| 2×18 (9) | GPU 1 | 0.342 | **0.261** | +0.081 | -23.7% | 25/25 |
| 4×18 (9) | GPU 1 | 0.633 | **0.479** | +0.154 | -24.4% | 25/25 |
| 8×18 (9) | GPU 4 | 1.214 | **0.955** | +0.260 | -21.4% | 25/25 |
| 2×24 (8) | GPU 4 | 0.439 | **0.321** | +0.118 | -26.9% | 25/25 |
| 4×24 (8) | GPU 4 | 0.829 | **0.637** | +0.192 | -23.1% | 25/25 |
| 8×24 (8) | GPU 4 | 1.495 | **1.267** | +0.228 | -15.3% | 25/25 |
| 2×32 (8) | GPU 4 | 0.568 | **0.426** | +0.141 | -24.9% | 25/25 |
| 4×32 (8) | GPU 4 | 1.088 | **0.845** | +0.243 | -22.3% | 25/25 |
| 8×32 (8) | GPU 4 | 2.122 | **1.694** | +0.428 | -20.2% | 25/25 |

**2. Backward**

| B×Hq (Hkv) | run | cuDNN (ms) | **Vg1** (ms) | Δ = cuDNN − Vg1 (ms) | Vg1 vs cuDNN | wins |
|---|---|---|---|---|---|---|
| 2×12 (4) | GPU 1 | 0.704 | **0.539** | +0.164 | -23.3% | 25/25 |
| 4×12 (4) | GPU 1 | 1.284 | **1.050** | +0.234 | -18.2% | 25/25 |
| 8×12 (4) | GPU 1 | 2.439 | **2.067** | +0.372 | -15.3% | 25/25 |
| 2×16 (4) | GPU 1 | 0.894 | **0.704** | +0.190 | -21.3% | 25/25 |
| 4×16 (4) | GPU 1 | 1.663 | **1.381** | +0.282 | -17.0% | 25/25 |
| 8×16 (4) | GPU 1 | 3.203 | **2.725** | +0.478 | -14.9% | 25/25 |
| 2×18 (6) | GPU 1 | 0.993 | **0.843** | +0.150 | -15.1% | 25/25 |
| 4×18 (6) | GPU 1 | 1.860 | **1.558** | +0.302 | -16.2% | 25/25 |
| 8×18 (6) | GPU 1 | 3.590 | **3.077** | +0.513 | -14.3% | 25/25 |
| 2×18 (9) | GPU 1 | 0.995 | **0.847** | +0.149 | -14.9% | 25/25 |
| 4×18 (9) | GPU 1 | 1.864 | **1.566** | +0.298 | -16.0% | 25/25 |
| 8×18 (9) | GPU 4 | 3.603 | **3.092** | +0.511 | -14.2% | 25/25 |
| 2×24 (8) | GPU 4 | 1.284 | **1.050** | +0.234 | -18.3% | 25/25 |
| 4×24 (8) | GPU 4 | 2.437 | **2.066** | +0.371 | -15.2% | 25/25 |
| 8×24 (8) | GPU 4 | 4.747 | **4.093** | +0.654 | -13.8% | 25/25 |
| 2×32 (8) | GPU 4 | 1.662 | **1.380** | +0.282 | -17.0% | 25/25 |
| 4×32 (8) | GPU 4 | 3.199 | **2.725** | +0.474 | -14.8% | 25/25 |
| 8×32 (8) | GPU 4 | 6.271 | **5.412** | +0.859 | -13.7% | 25/25 |

**3. Forward + backward (training step)**

| B×Hq (Hkv) | run | cuDNN (ms) | **Vg1** (ms) | Δ = cuDNN − Vg1 (ms) | Vg1 vs cuDNN | wins |
|---|---|---|---|---|---|---|
| 2×12 (4) | GPU 1 | 0.929 | **0.700** | +0.229 | -24.6% | 25/25 |
| 4×12 (4) | GPU 1 | 1.690 | **1.371** | +0.319 | -18.9% | 25/25 |
| 8×12 (4) | GPU 1 | 3.212 | **2.703** | +0.510 | -15.9% | 25/25 |
| 2×16 (4) | GPU 1 | 1.181 | **0.920** | +0.262 | -22.1% | 25/25 |
| 4×16 (4) | GPU 1 | 2.195 | **1.807** | +0.387 | -17.7% | 25/25 |
| 8×16 (4) | GPU 1 | 4.220 | **3.575** | +0.644 | -15.3% | 25/25 |
| 2×18 (6) | GPU 1 | 1.310 | **1.085** | +0.224 | -17.1% | 25/25 |
| 4×18 (6) | GPU 1 | 2.450 | **2.038** | +0.412 | -16.8% | 25/25 |
| 8×18 (6) | GPU 1 | 4.732 | **4.031** | +0.701 | -14.8% | 25/25 |
| 2×18 (9) | GPU 1 | 1.312 | **1.108** | +0.205 | -15.6% | 25/25 |
| 4×18 (9) | GPU 1 | 2.456 | **2.045** | +0.411 | -16.7% | 25/25 |
| 8×18 (9) | GPU 4 | 4.740 | **4.046** | +0.694 | -14.6% | 25/25 |
| 2×24 (8) | GPU 4 | 1.689 | **1.372** | +0.317 | -18.8% | 25/25 |
| 4×24 (8) | GPU 4 | 3.209 | **2.704** | +0.506 | -15.8% | 25/25 |
| 8×24 (8) | GPU 4 | 6.245 | **5.359** | +0.886 | -14.2% | 25/25 |
| 2×32 (8) | GPU 4 | 2.191 | **1.807** | +0.384 | -17.5% | 25/25 |
| 4×32 (8) | GPU 4 | 4.215 | **3.570** | +0.644 | -15.3% | 25/25 |
| 8×32 (8) | GPU 4 | 8.267 | **7.106** | +1.161 | -14.0% | 25/25 |

**Summary**

| | Vg1 vs cuDNN, all 18 shapes | runs won |
|---|---|---|
| Forward (derived) | **15–35% faster** (+0.08 to +0.43 ms) | 25/25 on every shape |
| Backward | **14–23% faster** (+0.15 to +0.86 ms) | 25/25 on every shape |
| Forward + backward | **14–25% faster** (+0.21 to +1.16 ms) | 25/25 on every shape |

**The colleague's claim is confirmed: Vg1 beats cuDNN on all 18 shapes, forward, backward and training step, in every
one of 25 runs.**

**How table 1 is derived:** the measured `VG1 fwd` line was dropped from both logs by the sweep bug described on
10-19 (the test's color-reset code started that line, and the sweep's `grep "^VG1"` skipped it; the fix was not yet
pushed when these runs were made). So per run, Vg1 fwd = (Vg1 fwd+bwd) − (Vg1 bwd), both measured in that run.
Checked on our own kernels, where all three are measured: over all 450 runs, (V62+Vj3) − Vj3 matches the measured
V62 forward with a median error of 0.5% (90th percentile 1.4%) — far below Vg1's 15–35% margin. A rerun with the
fixed build will replace it with measured numbers.

---

## Today — 2026-10-19: colleague's Vg1 beats cuDNN on every shape measured so far (11 of 18, sweep continues)

**Vg1** = a colleague's Hopper GQA forward + backward, added to our sweep unchanged:
- **Vg1 fwd** = `src/attention/GQA_fwd_sm90.cu` (a newer V62: persistent 132 CTAs, 3 warpgroups, m64n128 P@V)
- **Vg1 bwd** = `src/attention/GQA_bwd_vj2_sm90.cu` (FA3's head-dim-128 causal structure: 128 keys per block,
  64 queries per step, 384 threads = loader warp + dQ-writer warp + 2 compute warpgroups, setmaxnreg 24/240,
  register-A dV/dK, 1-D bulk reduce of a fragment-order fp32 dQ). **168 registers, 0 spills.**
- Integration: each file compiled as its own translation unit and linked into `gqa_fwd_bwd`; a 20-line shim
  (`third_party/vg1/device/CachingCudaAllocator.h`) stands in for their framework's allocator. Vg1 bwd consumes
  **Vg1 fwd's own O/LSE**. Checked against the same PyTorch reference and timed exactly like ours, interleaved with
  cuDNN (backend forced), same inputs. Its timed region covers the full backward (prep + main + 2 greduce + dQ
  convert), the same work as cuDNN's `autograd.grad`.
- **Correctness:** Vg1 passes every check (O, LSE, dQ, dK, dV) on all 11 shapes run; error equal to or slightly
  lower than ours (8×12: dQ max 1.47e-3 vs 1.95e-3, dK 3.9e-3 vs 7.8e-3 — it computes dS from fp32 P).

### Partial official board: 11 of 18 shapes, 25-run medians, clock HELD at 1530 MHz (host p032, GPU 1)

`reports/median25_gpu1_1530MHz_20261019_200003.log` (+ `_smi.csv`). **RULE CHECK: 2170/2170 loaded samples at
1530 MHz, throttle reasons: none.** All correctness checks passed. Δ = cuDNN − kernel in ms (+ = faster than cuDNN).

| B×Hq (Hkv) | cuDNN bwd | Vj3 (ours) | **Vg1 bwd** | **Vg1 Δ** | cuDNN fwd+bwd | V62+Vj3 | **Vg1+Vg1** | **Vg1 Δ** |
|---|---|---|---|---|---|---|---|---|
| 2×12 (4) | 0.704 | 0.803 | **0.539** | **+0.164** 25/25 | 0.929 | 1.013 | **0.700** | **+0.229** 25/25 |
| 4×12 (4) | 1.284 | 1.481 | **1.050** | **+0.234** 25/25 | 1.690 | 1.891 | **1.371** | **+0.319** 25/25 |
| 8×12 (4) | 2.439 | 2.828 | **2.067** | **+0.372** 25/25 | 3.212 | 3.638 | **2.703** | **+0.510** 25/25 |
| 2×16 (4) | 0.894 | 1.026 | **0.704** | **+0.190** 25/25 | 1.181 | 1.303 | **0.920** | **+0.262** 25/25 |
| 4×16 (4) | 1.663 | 1.924 | **1.381** | **+0.282** 25/25 | 2.195 | 2.469 | **1.807** | **+0.387** 25/25 |
| 8×16 (4) | 3.203 | 3.715 | **2.725** | **+0.478** 25/25 | 4.220 | 4.790 | **3.575** | **+0.644** 25/25 |
| 2×18 (6) | 0.993 | 1.141 | **0.843** | **+0.150** 25/25 | 1.310 | 1.484 | **1.085** | **+0.224** 25/25 |
| 4×18 (6) | 1.860 | 2.153 | **1.558** | **+0.302** 25/25 | 2.450 | 2.763 | **2.038** | **+0.412** 25/25 |
| 8×18 (6) | 3.590 | 4.177 | **3.077** | **+0.513** 25/25 | 4.732 | 5.388 | **4.031** | **+0.701** 25/25 |
| 2×18 (9) | 0.995 | 1.144 | **0.847** | **+0.149** 25/25 | 1.312 | 1.486 | **1.108** | **+0.205** 25/25 |
| 4×18 (9) | 1.864 | 2.159 | **1.566** | **+0.298** 25/25 | 2.456 | 2.769 | **2.045** | **+0.411** 25/25 |

**All kernels, backward (ms, 25-run median, 1530 MHz):**

| B×Hq (Hkv) | cuDNN | Vj1d | Vj1 | Vj2 | Vj3 | **Vg1** | best |
|---|---|---|---|---|---|---|---|
| 2×12 (4) | 0.704 | 0.882 | 0.870 | 0.838 | 0.803 | **0.539** | Vg1 |
| 4×12 (4) | 1.284 | 1.627 | 1.603 | 1.556 | 1.481 | **1.050** | Vg1 |
| 8×12 (4) | 2.439 | 3.110 | 3.068 | 2.980 | 2.828 | **2.067** | Vg1 |
| 2×16 (4) | 0.894 | 1.128 | 1.112 | 1.074 | 1.026 | **0.704** | Vg1 |
| 4×16 (4) | 1.663 | 2.116 | 2.088 | 2.021 | 1.924 | **1.381** | Vg1 |
| 8×16 (4) | 3.203 | 4.088 | 4.040 | 3.912 | 3.715 | **2.725** | Vg1 |
| 2×18 (6) | 0.993 | 1.255 | 1.236 | 1.197 | 1.141 | **0.843** | Vg1 |
| 4×18 (6) | 1.860 | 2.369 | 2.334 | 2.264 | 2.153 | **1.558** | Vg1 |
| 8×18 (6) | 3.590 | 4.595 | 4.532 | 4.401 | 4.177 | **3.077** | Vg1 |
| 2×18 (9) | 0.995 | 1.260 | 1.240 | 1.201 | 1.144 | **0.847** | Vg1 |
| 4×18 (9) | 1.864 | 2.376 | 2.342 | 2.269 | 2.159 | **1.566** | Vg1 |

**All kernels, forward + backward = training step (ms):**

| B×Hq (Hkv) | cuDNN | V62+Vj1d | V62+Vj1 | V62+Vj2 | V62+Vj3 | **Vg1+Vg1** | best |
|---|---|---|---|---|---|---|---|
| 2×12 (4) | 0.929 | 1.093 | 1.081 | 1.049 | 1.013 | **0.700** | Vg1+Vg1 |
| 4×12 (4) | 1.690 | 2.038 | 2.014 | 1.967 | 1.891 | **1.371** | Vg1+Vg1 |
| 8×12 (4) | 3.212 | 3.924 | 3.881 | 3.792 | 3.638 | **2.703** | Vg1+Vg1 |
| 2×16 (4) | 1.181 | 1.407 | 1.390 | 1.351 | 1.303 | **0.920** | Vg1+Vg1 |
| 4×16 (4) | 2.195 | 2.662 | 2.633 | 2.566 | 2.469 | **1.807** | Vg1+Vg1 |
| 8×16 (4) | 4.220 | 5.165 | 5.118 | 4.988 | 4.790 | **3.575** | Vg1+Vg1 |
| 2×18 (6) | 1.310 | 1.599 | 1.580 | 1.540 | 1.484 | **1.085** | Vg1+Vg1 |
| 4×18 (6) | 2.450 | 2.981 | 2.946 | 2.875 | 2.763 | **2.038** | Vg1+Vg1 |
| 8×18 (6) | 4.732 | 5.808 | 5.744 | 5.613 | 5.388 | **4.031** | Vg1+Vg1 |
| 2×18 (9) | 1.312 | 1.603 | 1.584 | 1.544 | 1.486 | **1.108** | Vg1+Vg1 |
| 4×18 (9) | 2.456 | 2.989 | 2.954 | 2.881 | 2.769 | **2.045** | Vg1+Vg1 |

- **Vg1 bwd beats cuDNN on all 11 shapes, in every one of 25 runs: 14–23% faster** (+0.15 to +0.51 ms).
- **Vg1 fwd+bwd (a full training step) beats cuDNN on all 11, 25/25: 15–25% faster** (+0.21 to +0.70 ms).
- **Our Vj3 vs Vg1: Vj3 takes 35–49% longer than Vg1 bwd** (8×12: 2.828 vs 2.067 ms). Ranking on every shape:
  Vg1 < cuDNN < Vj3 < Vj2 < Vj1 < Vj1d. The colleague's claim holds so far.
- **Vg1 fwd numbers are missing from this run (sweep bug, fixed):** the test's colored "Values match!" output ends
  with a terminal reset code that lands at the start of the next printed line; for Vg1 that line was
  `VG1 fwd …`, so the sweep's `grep "^VG1"` dropped it in every run. Fixed twice: the binary prints a blank line after
  Vg1's checks, and the script strips color codes before grepping. Vg1 bwd and fwd+bwd lines were not affected.
- Our V62 forward vs cuDNN at 1530 MHz: within ±0.03 ms on these 11 shapes (wins 2×12, 2×16; loses 2×18 0/25 to 1/25).

### Clock

- **GPU 4 on p032 held no clock ≥1605 MHz** under sweep load (`find_stable_clock.sh`: 1980/1905 0%, 1830 19%,
  1755 15%, 1680 59%, 1605 88%; dips to ~1250 MHz) → moved to **GPU 1 on p032**, where **1530 MHz held 100%**.
- The stable clock is specific to each GPU and host; absolute times here (1530 MHz) are higher than the 10-17 board
  (1605 MHz, different host) and must not be mixed with it. Comparisons within this run are fair.

### Remaining (next session)

1. The other 7 shapes (8×18:9 was started, 0 runs recorded):
   `SHAPES="8 18 9;2 24 8;4 24 8;8 24 8;2 32 8;4 32 8;8 32 8" GPU=1 CLOCK=1530 bash scripts/run_median12.sh`
   (pull first: it includes the Vg1-fwd fix).
2. Vg1 fwd for the 11 shapes above: rerun them with the fixed build if the forward comparison is needed.
3. Study Vg1 bwd: how its compute path fits **168 registers with 0 spills** (cuDNN: 168; our Vj3: 250; our Vm5
   attempt at the same 3-warpgroup shape spilled at 168).

---

## Today — 2026-10-17: Vj3 (cuDNN's wgmma shapes) is our best; official 18-shape board at a held clock

All runs on the H200 box; Δ in ms; every candidate passed dQ/dK/dV vs the PyTorch reference (2e-2) and dK/dV
bit-identical over 6 runs. Details: [plan_vm1_bwd.md](plan_vm1_bwd.md).

### 1. Official board: 25-run medians, 18 shapes, clock HELD at 1605 MHz (GPU 1)

`reports/median25_gpu1_1605MHz_20261017_130940.log` (+ `_smi.csv`). Rule (user, 10-17): clock fixed, cuDNN and ours
interleaved, 25 runs per shape, median. **RULE CHECK: 3614/3614 loaded samples at 1605 MHz, throttle reasons: none.**
All correctness checks passed. Method = do_bench (L2 flush per rep), cuDNN backend forced, same inputs.

**Backward:**

| B×Hq (Hkv) | cuDNN | Vj1d | Vj2 | **Vj3** | **Vj3 − Vj1d** | Vj3 − cuDNN | wins vs cuDNN |
|---|---|---|---|---|---|---|---|
| 2×12 (4) | 0.675 | 0.854 | 0.809 | **0.773** | **−0.081** | +0.098 | 0/25 |
| 4×12 (4) | 1.230 | 1.575 | 1.499 | **1.427** | **−0.148** | +0.197 | 0/25 |
| 8×12 (4) | 2.338 | 3.010 | 2.868 | **2.724** | **−0.286** | +0.386 | 0/25 |
| 2×16 (4) | 0.858 | 1.091 | 1.035 | **0.988** | **−0.103** | +0.130 | 0/25 |
| 4×16 (4) | 1.594 | 2.047 | 1.944 | **1.852** | **−0.195** | +0.258 | 0/25 |
| 8×16 (4) | 3.069 | 3.957 | 3.762 | **3.576** | **−0.381** | +0.507 | 0/25 |
| 2×18 (6) | 0.952 | 1.214 | 1.153 | **1.099** | **−0.115** | +0.147 | 0/25 |
| 4×18 (6) | 1.783 | 2.293 | 2.179 | **2.073** | **−0.220** | +0.290 | 0/25 |
| 8×18 (6) | 3.443 | 4.450 | 4.235 | **4.024** | **−0.426** | +0.582 | 0/25 |
| 2×18 (9) | 0.955 | 1.218 | 1.157 | **1.103** | **−0.115** | +0.148 | 0/25 |
| 4×18 (9) | 1.787 | 2.301 | 2.185 | **2.081** | **−0.220** | +0.294 | 0/25 |
| 8×18 (9) | 3.452 | 4.464 | 4.249 | **4.039** | **−0.425** | +0.587 | 0/25 |
| 2×24 (8) | 1.229 | 1.575 | 1.499 | **1.427** | **−0.148** | +0.198 | 0/25 |
| 4×24 (8) | 2.337 | 3.011 | 2.868 | **2.724** | **−0.287** | +0.387 | 0/25 |
| 8×24 (8) | 4.550 | 5.888 | 5.598 | **5.325** | **−0.563** | +0.775 | 0/25 |
| 2×32 (8) | 1.593 | 2.048 | 1.944 | **1.852** | **−0.196** | +0.259 | 0/25 |
| 4×32 (8) | 3.066 | 3.957 | 3.762 | **3.575** | **−0.382** | +0.510 | 0/25 |
| 8×32 (8) | 6.009 | 7.782 | 7.401 | **7.029** | **−0.753** | +1.020 | 0/25 |

- **Vj3 is our fastest backward on all 18 shapes: −9.5 to −9.7% vs Vj1d** (Vj2: −4.9 to −5.3%), in every one of 25
  runs. The ladder Vj1d → Vj2 → Vj3 is monotonic at every shape.
- **cuDNN still leads everywhere: Vj3 takes 14.5–17% longer** (+0.10 ms at 2×12 to +1.02 ms at 8×32). Down from
  26–30% for Vj1d.
- The 18:6 and 18:9 shapes behave like their siblings (G doesn't matter for these per-head kernels).
- **Forward + backward (V62 + Vj3)** is our best training step on all 18 shapes: +0.08 ms (2×12) to +1.16 ms (8×32)
  behind cuDNN. The V62 forward is even with cuDNN at small/mid shapes, behind at large B×Hq (−0.08 to −0.17 ms at
  8×18 / 8×24 / 8×32), and newly measured behind at 2×18 (−0.025, 0/25).

### 2. Vj3 — what changed and why

cuDNN's backward SASS at 8×12 (ncu, `reports/ncu_cudnn_bwd_B8_Hq12_Hkv4`) showed **Vj2's math is exactly cuDNN's**
(transposed layout: S^T = K·Q^T, P^T and dS^T as register-A wgmma operands for dV and dK, only dS stored for dQ) but
cuDNN issues **m64n128k16** for dV, dK and dQ where Vj2 issues two m64n64k16 halves. Same 8×12 profile: Vj2 553 M
instructions vs cuDNN 540 M at the same ~7.7 cycles each, but cuDNN keeps 10 warps/SM active vs 7.9 (tensor pipe
59% vs 46%) and moves half the L2 bytes (L2 64% vs 81%).

**Vj3 = Vj2 with one m64n128 per k-step for dV, dK and dQ** — cuDNN's exact wgmma mix (8× n128 register-A + 4× n128
smem-A + 16× n64 per step). The N=128 transposed B spans two 128-byte swizzle blocks; the descriptor needs
**LBO = 8192 B** (CUTLASS canonical B128 MN-major layout). August's notes had avoided this "multi-atom" case as
unvalidated — it is now validated on the H200. Also: dQ reads dS from smem once instead of twice, 7 block barriers
instead of 9. 250 registers (exactly fills 2 blocks/SM), 0 spills, 106 KB smem.
Single-shape check (8×12, GPU 1, 1980 MHz, sustained): Vj1d 2.833, Vj2 2.760, **Vj3 2.667 ms**; energy 2.12 / 2.08 /
**1.98 J** per backward.

### 3. Sustained 12-shape board (GPU 2, 1980 MHz held) — before Vj3

`reports/sustained12_gpu2_20261017_120145.log`: back-to-back ~3 s per kernel, 2 windows, NVML energy.

| B×Hq (Hkv) | cuDNN | Vj1d | **Vj2** | **Vj2 − Vj1d** | Vj2 − cuDNN | J/bwd cuDNN / Vj2 |
|---|---|---|---|---|---|---|
| 2×12 (4) | 0.639 | 0.777 | **0.756** | **−0.021** | +0.118 | 0.48 / 0.57 |
| 4×12 (4) | 1.212 | 1.483 | **1.448** | **−0.035** | +0.236 | 0.92 / 1.10 |
| 8×12 (4) | 2.356 | 2.889 | **2.820** | **−0.070** | +0.464 | 1.76 / 2.10 |
| 2×16 (4) | 0.826 | 1.013 | **0.981** | **−0.032** | +0.155 | 0.62 / 0.74 |
| 4×16 (4) | 1.591 | 1.949 | **1.899** | **−0.049** | +0.309 | 1.20 / 1.43 |
| 8×16 (4) | 3.113 | 3.817 | **3.725** | **−0.092** | +0.612 | 2.34 / 2.75 |
| 2×24 (8) | 1.210 | 1.485 | **1.445** | **−0.040** | +0.235 | 0.92 / 1.10 |
| 4×24 (8) | 2.355 | 2.886 | **2.814** | **−0.072** | +0.460 | 1.78 / 2.10 |
| 8×24 (8) | 4.634 | 5.681 | **5.555** | **−0.126** | +0.921 | 3.43 / 4.04 |
| 2×32 (8) | 1.580 | 1.941 | **1.891** | **−0.050** | +0.311 | 1.19 / 1.42 |
| 4×32 (8) | 3.102 | 3.809 | **3.711** | **−0.098** | +0.609 | 2.31 / 2.76 |
| 8×32 (8) | 6.162 | 7.532 | **7.369** | **−0.162** | +1.208 | 4.50 / 5.27 |

- Vj2 beat Vj1d on all 12 shapes (−2.2 to −2.8%); cuDNN led by ~18–20% with ~19% less energy per backward.

### 4. Measurement: the clock rule

- First 25-run sweep (`median25_20261017_093203.log`, log removed 2026-10-20 — summarized here; 12 shapes, GPU not recorded) **failed the 1980 MHz rule**:
  0% of loaded samples at 1980, median 1785 MHz, 700 W power cap active 79% of the time. In that power-capped
  run Vj2 lost to Vj1d on the large shapes (8×12 +0.235, 8×24 +0.191 ms) — the same power-cap interaction seen on
  10-16 at 8×18. Superseded by the held-clock board above.
- Rule amended (user chose option 2): lock the **highest clock that holds under sweep load**, found per GPU with
  `GPU=<n> bash scripts/find_stable_clock.sh` → 1605 MHz on GPU 1. `run_median12.sh` takes `GPU=` (required) and
  `CLOCK=`, logs the SM clock every 500 ms and ends with a RULE CHECK. Absolute times at 1605 MHz are higher than at
  1980; all comparisons are within one run.
- GPU 2 held 1980 MHz in sustained mode; GPU 1 held 1980 in the single-shape test but not under the sweep's
  continuous load.

### Where it stands / next

- **Best of ours: Vj3** (backward) and **V62 + Vj3** (training step). Gap to cuDNN: 14.5–17% backward.
- Remaining structural difference vs cuDNN: its 128-key, 2-compute-warpgroup work items (half the L2 traffic, 10
  warps/SM). The Vm line (10-16) tried that shape on Vj1d's math and lost; with Vj3's math (n128, register-A) it is
  worth revisiting, but register budget (Vj3 = 250 regs/thread) is the obstacle — cuDNN does it in 168.
- Next: profile Vj3 at 8×12 against cuDNN (registers, issue rate, L2), then decide between a register-lean Vj3
  and a cluster/multicast variant that halves Q/dO L2 reads while keeping 2 blocks/SM.

---

## Today — 2026-10-16: the 128-key line closed, register fusion (Vj2) is the first win over Vj1d

All numbers: H200, GPU 1, S=4096 D=128 causal bf16. Δ = candidate − Vj1d in ms (negative = faster). Every
candidate passed dQ/dK/dV vs the PyTorch reference (2e-2) and dK/dV bit-identical over 6 runs.
Full detail: [plan_vm1_bwd.md](plan_vm1_bwd.md), [ncu_vj1d_vs_cudnn_B8_Hq18.md](ncu_vj1d_vs_cudnn_B8_Hq18.md).

### 1. The Vm line: cuDNN's 128-key structure, five tries, none beat Vj1d

ncu + SASS (10-13) showed cuDNN's backward = 132 persistent blocks, each work item **128 keys as 2 compute
warpgroups × 64**, sharing one Q/dO load and one summed dQ reduce per q-step: half Vj1d's L2 bytes per key.
We built that on Vj1d's per-tile pipeline:

| Version | Change | 8×12 (ms) | Δ vs Vj1d |
|---|---|---|---|
| Vj1d | baseline: 64 keys/block, 2 independent blocks/SM | 2.738 | — |
| Vm1 | 128 keys/block, 2 WGs share Q/dO, dQ summed in smem | 3.561 | +0.77 |
| Vm2 | + compile-time per-WG addressing (Vm1 had +32% instructions of descriptor math) | 3.123 | +0.33 |
| Vm3 | + parallel dQ combine | 3.088 | +0.35 |
| Vm4 | owner-split combine (40% less smem data) | 3.157 | +0.42 |
| Vm5 | fully decoupled WGs (mbarrier handoffs, double-buffered Q/dO, "second one does it") | 3.267 | +0.53 |

Every version cut L2 traffic **exactly to cuDNN's level** (ncu: reads 216 → 114 M sectors, dQ reduce-adds
204 → 104 M at 8×12), yet all lost. At 1 block/SM, every way of combining two warpgroups (lockstep,
parallel, cheaper, decoupled) cost more than the L2 saving; Vj1d's two independent blocks per SM hide
latency better. **All Vm kernels removed** (rule: drop what doesn't work, keep everything up to Vj1d);
recoverable from git `fe8089a`. Side lessons: `setmaxnreg` does not lift ptxas's per-warpgroup register cap
(384 threads → 168 regs + spills); a `printf` anywhere in a kernel serializes every wgmma.

### 2. Vj2: keep P^T / dS^T in registers (user's fusion idea)

FA3's transposed backward on Vj1d: compute S^T = K·Q^T, so P^T and dS^T are already wgmma **register-A**
operands. dV += P^T·dO and dK += dS^T·Q read registers; P never touches smem; only dS^T is stored (for dQ).
SASS: 16/40 HGMMA register-A, STSM 12 → 4, LDSM 4 → 0, one barrier fewer per step; 218 regs, 0 spills,
106 KB smem → still 2 blocks/SM.

| Measurement | 8×12 Vj1d → Vj2 | 8×18 Vj1d → Vj2 |
|---|---|---|
| ncu, main kernel only | — | 3.584 → **3.432 ms (−0.152)** |
| do_bench, 6 runs (median) | 2.768 → **2.700 (−0.068), 6/6** | 4.115 → 4.384 (+0.27), 0/6 |
| **SUSTAINED** (2 windows) | 2.888 → **2.812 (−0.077)** | 4.302 → **4.185 (−0.117)** |
| Energy per backward (sustained) | 2.17 → **2.12 J** | 3.20 → **3.13 J** |

**Vj2 is a consistent ~2.7% win in the training-like regime, and uses less energy.** The one loss is
do_bench at 8×18 (L2-flush memset before every launch, short blocks): nsys showed Vj2 starting each timed
block at ~3.50 ms and degrading over ~10 launches while the GPU sat at its 700 W cap. Cause of that
interaction is still open (an energy explanation was tested and ruled out: Vj2 uses less energy).

### 3. Measurement fixes found today

- **Wrong GPU index:** commands said GPU 3; the user runs on GPU 1 → the clock lock never applied. Scripts
  now take `GPU=` (default 1) and touch only that GPU. They previously locked/reset **all** GPUs, including
  GPU 0 running another user's job.
- **Power cap:** sustained backward runs at the 700 W cap; mid-run SM 1830 MHz. 1980 MHz has never been
  observed on GPU 1 (whether 1830 is the board max is still to check).
- **New tooling:** `gqa_bwd_vm1_test` SUSTAINED mode (ms/bwd, W, J/bwd via NVML); `cudnn_time.py` got the
  same sustained protocol for cuDNN; `GQA_fwd_bwd.cu` + `median_summary.py` include Vj2.

### Where it stands / next

All sustained (back-to-back ~3 s, 2 windows), GPU 1, same protocol for cuDNN (`cudnn_time.py`) and ours:

| Sustained | 8×12 ms/bwd | 8×12 J/bwd | 8×18 ms/bwd | 8×18 J/bwd |
|---|---|---|---|---|
| **cuDNN** | **2.330** | **1.77** | **3.473** | **2.60** |
| **Vj2** (best of ours) | 2.812 | 2.12 | 4.185 | 3.13 |
| Vj1d | 2.888 | 2.17 | 4.302 | 3.20 |

- **Real gap: Vj2 is 0.48 ms (8×12) and 0.71 ms (8×18) behind cuDNN** under the same sustained method
  (cuDNN also slows under sustained load: 2.134 → 2.330 and 3.247 → 3.473 ms vs do_bench).
- **cuDNN uses ~17% less energy per backward than Vj2 at both shapes** (1.77 vs 2.12 J; 2.60 vs 3.13 J). At the
  700 W cap energy turns directly into time, so energy per backward is a useful lens for the next round.

Next session: (1) `nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.max.sm` to see
whether 1830 MHz is the board max; (2) 12-shape sweep with Vj2; (3) profile Vj2 for its next limit (it doesn't
reduce L2 traffic, Vj1d's limit at 80%), with energy per backward next to time.

---

## ⚠ Board corrected — 2026-10-12: **we do NOT beat cuDNN. Vj1d is 24–37% slower on all 12 shapes (measured).**

**What was wrong.** Every cuDNN number on the old board (and in every daily log below) came from
`_sdpa_bwd` in `precision/baseline_gqa.py`, which timed `sdpa(q,k,v).backward(do)` — i.e. cuDNN's
**forward + backward** — while our `gqa_bwd` bench times **backward only** (memset + drowsum + main +
greduce + dQ convert; O/LSE precomputed). The backend was also never forced to cuDNN. So the old
"12/12" compared our backward against cuDNN's forward+backward.

**Corrected board — MEASURED on H200, 2026-10-12** (`vz2_sweep_20261012_091718.log`, log no longer kept; locked
1980 MHz, cuDNN backend forced — `native enable_gqa`, its dQ/dK/dV match default SDPA bit-for-bit, so
the old default path *was* cuDNN). cuDNN bwd = `autograd.grad` on a pre-built graph (do_bench).
Negative = slower than cuDNN. Single sweep; 12-run medians: `bash scripts/run_median12.sh`.

| B×Hq (Hkv) | cuDNN fwd | cuDNN fwd+bwd (old "cuDNN") | **cuDNN bwd** | **Vj1d** | Vj1d vs cuDNN bwd | Vj1 | Vj1 vs cuDNN bwd |
|---|---|---|---|---|---|---|---|
| 2×12 (4) | 0.198 | 0.802 | **0.607** | 0.762 | **−25.6%** | 0.747 | −23.1% |
| 4×12 (4) | 0.372 | 1.511 | **1.107** | 1.400 | **−26.5%** | 1.374 | −24.1% |
| 8×12 (4) | 0.711 | 2.912 | **2.161** | 2.714 | **−25.6%** | 2.735 | −26.5% |
| 2×16 (4) | 0.251 | 1.021 | **0.773** | 0.971 | **−25.7%** | 0.953 | −23.4% |
| 4×16 (4) | 0.463 | 1.971 | **1.426** | 1.957 | **−37.3%** | 1.824 | −27.9% |
| 8×16 (4) | 0.917 | 3.809 | **2.757** | 3.711 | **−34.6%** | 3.700 | −34.2% |
| 2×24 (8) | 0.368 | 1.500 | **1.116** | 1.411 | **−26.4%** | 1.383 | −23.9% |
| 4×24 (8) | 0.724 | 2.927 | **2.196** | 2.725 | **−24.1%** | 2.757 | −25.5% |
| 8×24 (8) | 1.361 | 5.787 | **4.367** | 5.762 | **−32.0%** | 5.412 | −23.9% |
| 2×32 (8) | 0.467 | 1.954 | **1.444** | 1.961 | **−35.7%** | 1.807 | −25.1% |
| 4×32 (8) | 0.948 | 3.846 | **2.852** | 3.716 | **−30.3%** | 3.675 | −28.8% |
| 8×32 (8) | 1.862 | 7.608 | **5.728** | 7.338 | **−28.1%** | 7.121 | −24.3% |

**Score: Vj1d 0/12, Vj1 0/12 (24–37% slower).** Our backward ≈ cuDNN's forward + backward.

### 12-run apples-to-apples — H200, 2026-10-12 (`reports/median12_20261012_095009.log`)

`bash scripts/run_median12.sh`: our chain `gqa_fwd_bwd` (V62 fwd → Vj1d|Vj1 bwd on **V62's own O/LSE**)
vs `precision/cudnn_time.py` (cuDNN forced), **same inputs**, same do_bench-style timing (L2 flush per rep,
async queue, one sync), runs interleaved, locked 1980 MHz. Median of 12; wins = runs ours < cuDNN.
All correctness checks pass on all 12 shapes (V62 O/LSE; dQ/dK/dV for Vj1d and Vj1, tol 2e-2).
Δ: + = we are faster.

| B×Hq (Hkv) | cuDNN fwd | V62 | Δ (wins) | cuDNN bwd | Vj1d | Δ | Vj1 | Δ | cuDNN fwd+bwd | V62+Vj1d | Δ | V62+Vj1 | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2×12 (4) | 0.197 | 0.182 | **+7.2%** (12/12) | 0.607 | 0.766 | −26.2% | 0.758 | −24.8% | 0.801 | 0.950 | −18.6% | 0.935 | −16.7% |
| 4×12 (4) | 0.348 | 0.351 | −0.7% (0/12) | 1.104 | 1.411 | −27.8% | 1.383 | −25.3% | 1.492 | 1.770 | −18.7% | 1.758 | −17.8% |
| 8×12 (4) | 0.652 | 0.693 | −6.3% (3/12) | 2.147 | 2.735 | −27.4% | 2.763 | −28.7% | 2.906 | 3.475 | −19.6% | 3.425 | −17.9% |
| 2×16 (4) | 0.247 | 0.239 | **+3.6%** (12/12) | 0.772 | 0.973 | −26.0% | 0.956 | −23.7% | 1.027 | 1.213 | −18.1% | 1.196 | −16.4% |
| 4×16 (4) | 0.450 | 0.463 | −2.8% (5/12) | 1.439 | 1.842 | −28.0% | 1.815 | −26.1% | 1.956 | 2.338 | −19.5% | 2.456 | −25.5% |
| 8×16 (4) | 0.868 | 0.942 | −8.6% (0/12) | 2.892 | 3.608 | −24.8% | 3.561 | −23.1% | 3.838 | 4.911 | −28.0% | 4.905 | −27.8% |
| 2×24 (8) | 0.349 | 0.351 | −0.7% (2/12) | 1.109 | 1.412 | −27.3% | 1.384 | −24.8% | 1.486 | 1.772 | −19.3% | 1.763 | −18.7% |
| 4×24 (8) | 0.698 | 0.691 | +1.0% (6/12) | 2.163 | 2.734 | −26.4% | 2.767 | −27.9% | 2.903 | 3.470 | −19.5% | 3.420 | −17.8% |
| 8×24 (8) | 1.258 | 1.439 | −14.4% (0/12) | 4.288 | 5.383 | −25.5% | 5.397 | −25.8% | 5.674 | 6.898 | −21.6% | 7.098 | −25.1% |
| 2×32 (8) | 0.480 | 0.463 | **+3.5%** (8/12) | 1.453 | 1.839 | −26.6% | 1.809 | −24.5% | 1.956 | 2.337 | −19.4% | 2.451 | −25.3% |
| 4×32 (8) | 0.853 | 0.919 | −7.8% (2/12) | 2.878 | 3.603 | −25.2% | 3.551 | −23.4% | 3.815 | 4.906 | −28.6% | 4.846 | −27.0% |
| 8×32 (8) | 1.662 | 1.917 | −15.3% (0/12) | 5.706 | 7.323 | −28.4% | 7.403 | −29.8% | 7.578 | 9.322 | −23.0% | 9.343 | −23.3% |

Backward wins 0/12 for both Vj1d and Vj1 (every single run lost). fwd+bwd wins 0/12.

**Reading it.**
- **Backward is the gap:** Vj1d and Vj1 are 23–30% slower than cuDNN's backward on every shape.
  Vj1's atomic TMA-reduce is *not* reliably faster than Vj1d any more (8×12, 4×24, 8×24, 8×32 it's slower).
- **Forward (V62) is roughly at parity on small/mid shapes** (+7% to −3%) but **falls behind at large
  B×Hq** (8×24 −14%, 8×32 −15%, 8×16 −9%). V62's header claim "1.6% faster than cuDNN at 8×12" does
  not reproduce here (−6.3%, 3/12).
- **Open anomaly — chain overhead at B×Hq=128:** at 8×16 and 4×32 (only), fwd→bwd chained costs a
  consistent **+0.37 ms** more than fwd + bwd timed separately (every run, 0.30–0.46); all other shapes
  ≤0.1 ms, and cuDNN's chain overhead is ≤0.2 ms everywhere. That's why those two rows are −28% fwd+bwd.
  Needs an nsys trace of the chained 8×16 run. (V62's launcher also calls `cudaGetDeviceProperties` on
  every launch — host-side cost, worth removing regardless.)

**Fix landed:**
`baseline_gqa.py` now prints `cuDNN bwd-only (forced)` (forward once outside the timer, `autograd.grad`
timed, backend forced, gradients checked vs SDPA) and `scripts/run_vz2_sweep.sh` parses that line. The
old fwd+bwd rows are still printed, relabelled `SDPA fwd+bwd`, for reference only. The apples-to-apples
tooling is `gqa_fwd_bwd` + `precision/cudnn_time.py` + `scripts/run_median12.sh` + `scripts/median_summary.py`.
Full write-up with ms deltas: [cudnn_comparison_2026-10-12.md](cudnn_comparison_2026-10-12.md).

**What still stands:** the relative progress among our own kernels (naive 305 ms → Vz2 → Vj1/Vj1d,
per-hq grid, TMA-reduce dK/dV, determinism measurements) — those were all timed the same way.
**What does not:** every "vs cuDNN" margin, "WIN", "12/12", "nemesis", and "beats cuDNN" in the daily
logs below. They are kept unedited as history.

### Profiling — 2026-10-13: where the backward gap is (ncu + nsys, B=8 S=4096 Hq=18, Hkv=9 and 6)

Data only, no optimization yet. Full numbers: [ncu_vj1d_vs_cudnn_B8_Hq18.md](ncu_vj1d_vs_cudnn_B8_Hq18.md).

| | Vj1d | cuDNN |
|---|---|---|
| Backward kernels | drowsum 0.082 · **main 3.578** · greduce 2×0.057 · dQ convert 0.105 ms | dot_do_o 0.149 · **bprop 2.770** (ncu) · reduce_head 2×0.055 · convert_dq 0.116 ms |
| Main kernel layout | 9,216 blocks × 128 thr, 2/SM, 220 regs | 132 persistent blocks × 384 thr, 1/SM, 168 regs |
| Tile | 64×64×128 | 64×64×128 |
| Tensor pipe / L2 throughput | 44.8% / **80% (limit)** | 60.0% / 64% |
| L2 reads / dQ reduce-adds | 10.4 / 9.8 GB | 5.4 / 5.0 GB (**0.5×**) |

- Same pipeline as cuDNN, kernel for kernel (both write per-head partial dK/dV and reduce over G in a separate
  fixed-order kernel, which is why both have bit-identical dK/dV). Helper kernels are not the gap.
- **The gap is the main kernel**, which is L2-bandwidth bound: per 64 keys it reloads Q/dO (32 KB) and
  reduce-adds an fp32 dQ tile (32 KB) to L2, twice cuDNN's bytes per key. G (Hkv=9 vs 6) changes nothing for either side.
- **Confirmed from SASS execution counts** (`plan_vm1_bwd.md`): cuDNN covers 128 keys per work item per q-step, as two
  consumer warpgroups × 64 keys that share one Q/dO load and sum dQ in-block before one reduce.
- **Plan:** [plan_vm1_bwd.md](plan_vm1_bwd.md): Vm1 = Vj1d with 128-key work items.

---

## Today — 2026-08-19: one kernel to rule them all (12/12)

Yesterday I shipped a best-of-three: V44 for small batch, Vz2 for big batch, 8/12 confirmed. The four
losses were 8×12 and the three mid shapes (4×16, 2×32, 4×24). Today I went after them with a single
lever — and it turned into a near-universal win.

**The lever — Vj1 = Vz2 cloned, made per-head.** I kept Vz2 frozen and cloned it to **Vj1**, then
changed one thing: the grid. Vz2 parallelizes over KV heads (grid `S/Bc, Hkv, B`) and loops the G query
heads inside each CTA. Vj1 parallelizes over **query** heads (grid `S/Bc, Hq, B` = **3× the CTAs** at
G=3), drops the inner G-loop so each CTA does **one head**, writes a **partial** dK/dV to a
`[B, Hq, S, D]` scratch buffer, and a tiny `gqa_dkdv_greduce` sums the G partials into the final dK/dV.
Everything else — the wgmma pipeline, the deferred dQ reduces, the S&dP overlap — is untouched Vz2.

**Why it works, and why it's more than a mid-shape patch.** The thesis was that low-batch shapes starve
because there aren't enough CTAs to hide latency. Per-head grids give 3× the CTAs *and* a shorter serial
chain per CTA (one head, no G-loop). That re-fills the GPU at **every** batch size, not just the mid
ones:
- It **flipped the mid shapes**: 4×16 went from a 6.5% loss to a **12/12 win** (median 1.894 vs 1.911),
  2×32 to a **7/12 win** (1.896 vs 1.898).
- It **took over the small shapes** too (2×16, 2×24, 2×32 all now Vj1, beating both V44 and cuDNN).
- It **won 8×12** — the shape I never once won all week. 12-run median-of-medians Vj1 2.865 vs cuDNN
  2.933, 7/12 head-to-head. It's a thermal-dependent win (cuDNN's cool floor ~2.77 still edges Vj1's,
    but cuDNN throttles under sustained load to 2.9–3.0 while Vj1 holds 2.80–2.93), and Vj1 is far
  steadier there than Vz2 (std 0.03 vs 0.25 ms). The nemesis is down.
- On the big shapes it's simply **faster than Vz2** (8×16 3.65 vs 3.80, 8×24 5.50 vs 5.72, 8×32 7.29 vs
  7.44), so it takes those crowns outright.

Net: Vj1 is the fastest of my kernels on 11/12 shapes — collapsing yesterday's best-of-three into
essentially **one champion**.

**The last shape — 4×24 — fell to the profiler.** It was the one loss left (cuDNN by ~1%). The profile
was unambiguous: the main kernel was fine (44% SM-busy, load-latency-bound at 2-CTA occupancy), but the
**per-hq G-reduce kernel was a 69 µs, DRAM-bound (75%) serial tail** — 2.5% of runtime, *bigger than the
deficit*. Per-hq had to reduce the G partial dK/dV somehow, and I'd done it with a scratch buffer +
follow-up sum kernel. That tax was the whole gap.

**The fix is cuDNN's own trick: TMA-reduce.** Instead of writing partials to a `[B,Hq,S,D]` scratch
buffer and summing them in a second kernel, each per-hq CTA now `cp.reduce.async.bulk.tensor`-adds its
dK/dV **straight into the final `[B,Hkv,S,D]` tensor** — the G-way sum happens atomically in L2, in the
TMA engine, overlapped with the main kernel's compute. Scratch buffer gone, reduce kernel gone. The
bf16 L2 accumulation holds precision (dK max_abs 3.9e-3 → 5.9e-3, well under the 2e-2 tol). It saved
**~111 µs** (more than the 69 µs predicted — killing the scratch buffer also cut the main kernel's
partial-store traffic), taking 4×24 from **2.78 → 2.67**. 12-run: **Vj1 2.702 vs cuDNN 2.786, +3.0%,
11/12.** The last loss is a win.

**That's 12/12 — every shape beats cuDNN.** The TMA-reduce lever is a universal Vj1 change, so every
other shape got ~100 µs faster too; a confirming sweep will widen all the margins.

**What didn't land.** Earlier I tried overlapping the *old* G-reduce with the dQ convert on separate CUDA
streams — it didn't move the median and **added jitter** (4×16 std 0.03 → 0.12 ms). The right answer
wasn't to hide the reduce kernel, it was to **delete it** with TMA-reduce. Reverting that stream attempt
is what kept 4×16 winning 12/12; the real fix came from the profiler pointing at the reduce itself.

**Determinism — measured, and an honest asymmetry.** I checked whether the TMA-reduces cost run-to-run
reproducibility versus cuDNN, by running each backward 8× on byte-identical inputs
(`precision/check_cudnn_determinism.py` for cuDNN, `precision/vj1_determinism_probe.cu` for Vj1):

| gradient | cuDNN max\|Δ\| | Vj1 max\|Δ\| |
|---|---|---|
| dQ | 7.6e-6 (non-det) | ~6e-5 (non-det) |
| dK | **0 — bit-identical** | 4.9e-4 – 7.8e-3 (non-det) |
| dV | **0 — bit-identical** | 3.9e-3 – 3.1e-2 (non-det) |

So it is **not** a symmetric trade. On **dQ** we make the *identical* trade cuDNN does — both reduce dQ
across blocks, both wobble at ~1e-5. On **dK/dV** we trade reproducibility cuDNN *keeps*: cuDNN
accumulates the query heads in-register (one writer → bit-exact), whereas Vj1's per-hq split turned
dK/dV into a cross-CTA **bf16** reduce whose atomic order flips the last bf16 bit — the 3.1e-2 on dV is
exactly one bf16 ULP at dV's larger magnitudes. It's within the correctness envelope (2e-2 relative,
mean_abs ~1e-5) and irrelevant for training, but the asymmetry is real: **cuDNN's dK/dV are exact; Vj1's
are reproducible only to ~1 ULP** — the price of the per-hq speed.

**If bit-identical dK/dV is required, use Vj1d.** The wobble comes from the *atomic* cross-CTA reduce, so
making it fp32 only shrinks it (~1e-4), never to zero — any atomic reduce is order-dependent. The fix is
to drop atomics entirely: Vj1d writes one-writer per-hq partials and sums the G heads with a **fixed-order
`greduce`** (no atomics). Measured: **dK/dV bit-identical, max|Δ| = 0 across 8 runs**, dQ still
non-deterministic at ~3e-5 — i.e. *exactly* cuDNN's determinism profile. Cost: the ~69 µs greduce pass
(the tax Vj1 removed), so Vj1d wins the roomy shapes and loses the tightest ~1–2%. Two supported modes:
**Vj1d** — bit-identical dK/dV like cuDNN, ~69 µs slower — and **Vj1** — fastest (12/12), dK/dV wobble
~1 ULP. **We ship Vj1d as the standalone baseline (`GQA_bwd_baseline.cu`)**: for a *reference*
deliverable, cuDNN-grade reproducibility is worth ~2%, and Vj1d matches cuDNN's determinism profile
exactly (dK/dV bit-identical, dQ non-deterministic). **Vj1** stays in `GQA_bwd.cu` as the max-speed
alternative for anyone who wants the last 2% and doesn't need bit-reproducible gradients.

**On V44 — the retired small-shape champion.** For the record, V44 (swizzled TMA-reduce-dQ, 3 warpgroups
per CTA) anchored the small-batch end of yesterday's best-of-three. It wins the way cuDNN does — hiding
latency *inside* each block (thesis mode #1) — which is exactly what low batch needs when there aren't
many blocks to schedule. The per-head lever made that cleverness redundant: by manufacturing 3× the
blocks, Vj1 gets its latency hidden by *occupancy* instead, and matches or beats V44 on all 12 shapes
except 2×12, where V44 is **0.6% faster** (0.744 vs 0.749) — and Vj1 still beats cuDNN there by 8%. That
0.6% at one small shape is not worth shipping a second kernel and a size-based dispatch. **V44 stays in
the dev tree (`GQA_bwd.cu`) as a benchmark reference only; the standalone deliverable
`GQA_bwd_baseline.cu` ships **Vj1d** (deterministic bit-identical dK/dV), with the faster **Vj1** kept
in `GQA_bwd.cu` as the max-speed alternative.**
