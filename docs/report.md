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

| 8×12 | 8×18 | |
|---|---|---|
| cuDNN bwd (do_bench) 2.212 ms | 3.264 ms | sustained cuDNN not yet measured |
| **Vj2 sustained 2.812 ms** | **4.185 ms** | best of ours |

Gap to cuDNN is still ~0.6–0.9 ms. Next session: (1) sustained cuDNN numbers (same protocol);
(2) 12-shape sweep with Vj2; (3) profile Vj2 for its next limit (it doesn't reduce L2 traffic, which was
Vj1d's limit at 80%); (4) check `clocks.max.sm` on GPU 1.

---

## ⚠ Board corrected — 2026-10-12: **we do NOT beat cuDNN. Vj1d is 24–37% slower on all 12 shapes (measured).**

**What was wrong.** Every cuDNN number on the old board (and in every daily log below) came from
`_sdpa_bwd` in `precision/baseline_gqa.py`, which timed `sdpa(q,k,v).backward(do)` — i.e. cuDNN's
**forward + backward** — while our `gqa_bwd` bench times **backward only** (memset + drowsum + main +
greduce + dQ convert; O/LSE precomputed). The backend was also never forced to cuDNN. So the old
"12/12" compared our backward against cuDNN's forward+backward.

**Corrected board — MEASURED on H200, 2026-10-12** (`reports/vz2_sweep_20261012_091718.log`, locked
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
