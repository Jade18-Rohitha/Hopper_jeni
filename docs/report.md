# GQA Backward — H200 champion board & daily log

> Two sections, both refreshed each day: the **clear-winners board** (the current best kernel per
> shape, locked in and only replaced by a strictly faster verified median) and **today's report**
> (what I tried, how close I got, why it did or didn't land). S=4096, D=128, bf16, causal.
> Reference = **cuDNN backward-only, backend forced** (`SDPBackend.CUDNN_ATTENTION`, forward run once outside
> the timer, time `autograd.grad` only). Locked clock: `nvidia-smi -pm 1; -lgc 1980`.

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

**Fix landed:**
`baseline_gqa.py` now prints `cuDNN bwd-only (forced)` (forward once outside the timer, `autograd.grad`
timed, backend forced, gradients checked vs SDPA) and `scripts/run_vz2_sweep.sh` parses that line. The
old fwd+bwd rows are still printed, relabelled `SDPA fwd+bwd`, for reference only. Re-run
`bash scripts/run_vz2_sweep.sh` on the H200 to replace this table with measured numbers.

**What still stands:** the relative progress among our own kernels (naive 305 ms → Vz2 → Vj1/Vj1d,
per-hq grid, TMA-reduce dK/dV, determinism measurements) — those were all timed the same way.
**What does not:** every "vs cuDNN" margin, "WIN", "12/12", "nemesis", and "beats cuDNN" in the daily
logs below. They are kept unedited as history.

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
