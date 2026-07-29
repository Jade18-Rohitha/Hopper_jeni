# RESEARCH — Next structural move to close the ~3.14× gap to cuDNN (Hopper sm_90a GQA-backward)

**Status:** research / roadmap only. No kernel changes. Decides the next version after V13.
**Baseline:** `gqa_backward_v13_kv<64,64,128>` — 8.82 ms / 374 TFLOP/s, 0 spills, 154 regs,
18.75% occupancy, Compute SOL 36% / Mem SOL 57%, all bit-identical.
**Target:** cuDNN `flash_bprop_wgmma_f16_64x64x128` — 2.81 ms / 1173 TFLOP/s, Compute SOL 59% /
Mem SOL 63%, 15.6% occupancy, 168 regs **spilling** (1.04 M local req).
**Author's confidence:** the *diagnosis* is high-confidence (grounded in our ncu profiles + FA3/FA4
literature that independently names the same bottleneck). The *magnitude* of each proposed fix is an
estimate — flagged per item; only an H200 build/profile confirms it.

---

## 0. Executive summary + the one recommended move

We are **shared-memory-throughput bound**, and this is not an accident of our kernel — it is the
*known* characteristic of an attention backward on Hopper. Colfax's FlashAttention-4 write-up states
it plainly: on the backward pass the kernel is *"constrained by shared-memory bandwidth — leaving
matmul units underutilized"* ([Colfax FA4](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)).
That is exactly our profile: Mem SOL 57% (binding) vs Compute SOL 36% (idle). Our own V13 analysis
already reached the same conclusion — *"we push too many bytes through L1/shared per unit of
compute."* The remaining ~3× is that ratio.

**Where our extra shared bytes come from — two sources, both structural:**
1. **Software operand transposes.** `fill_trans` / `fill_trans_A` (GQA_bwd.cu:801, 967) physically
   re-pack every dV/dK/dQ operand into a *second* smem buffer (`sA_t`/`sB_t`) with ordinary stores,
   then wgmma reads it back. That is an **extra full smem write + read per transposed GEMM operand**,
   pure overhead on the binding resource.
2. **fp32 accumulator scatters.** `store_acc_smem` (GQA_bwd.cu:990) scatters fp32 wgmma fragments to
   `sS`/`sdP`/dQ-staging — the surviving **3.8-way shared-STORE conflict (est. 33% of the shared
   store lever)** flagged in V13_analysis §4.

**cuDNN and FA3 do neither of these.** FA3 performs the Pᵀ/dSᵀ transpose by **smem layout aliasing**
— reading the *same physical bytes* through a transposed wgmma descriptor (`SmemLayoutPdSt` aliases
`SmemLayoutPdS` with swapped M/N; `cute::composition(layout, transposed_shape_stride)`), with **no
data movement at all**
([flash-attention `mainloop_bwd_sm90_tma_gmma_ws.hpp`](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp)).
That single design choice is most of the shared-bytes-per-MMA gap.

> **Recommended next move (highest value, most testable): eliminate the software transpose.**
> Replace the `fill_trans` / `fill_trans_A` re-pack for the transposed GEMMs (dV = Pᵀ·dO, dK = dSᵀ·Q,
> dQ's Kᵀ operand) with **direct transposed-descriptor reads** of the resident operand from its
> existing swizzled smem — the wgmma Major::MN / trans-b path. This attacks the *measured* binding
> resource (L1/shared throughput) at its largest structural source, is **testable one GEMM at a time**
> without a rewrite, and reuses a descriptor recipe we have already validated on hardware
> (`reference_hopper_wgmma_mn_swizzle_d128`). Expected: shared wavefronts and Mem SOL fall; Compute
> SOL rises toward the point where the tensor pipe becomes binding — the same regime shift that makes
> cuDNN compute-bound.

The FA3 **ping-pong / symmetric-two-consumer-warpgroup** restructure (§2, §3a) is the bigger swing but
a full rewrite, higher risk, and its headline benefit is **partly already captured** by V13 (our
S∥dP split already runs two concurrent tensor streams, and our occupancy already *exceeds* cuDNN's).
Recommend it as the *follow-on* if transpose-elimination plateaus, not as move #1.

Honest ceiling: transpose-elimination + de-aliased fp32 scatters should be worth a meaningful chunk
(plausibly toward ~2–2.3× off cuDNN — estimate, not a promise). Full parity almost certainly also
needs the schedule rewrite **and** some hand-tuned-SASS-level effects we cannot reach from CUDA C
(cuDNN's exact instruction interleave). See §5 for the honest gap estimate.

---

## 1. The compute-feeding gap (36% → 59%) — mechanism

### 1.1 What "Compute SOL 36%" actually means here
Our tensor pipe issues MMA only 36% of cycles; cuDNN's is at 59%. We are **not** short of warps
(18.75% occ > cuDNN's 15.6%) and **not** DRAM-bound (11% DRAM). The tensor cores idle because during
each tile the warpgroups spend long windows on **non-MMA work that competes for the L1/shared port**:

| Per-tile window (V13) | What runs | Tensor pipe? | Shared traffic |
|---|---|---|---|
| S=Q·Kᵀ ∥ dP=dO·Vᵀ | wgmma on wg0 ∥ wg1 | **busy** (2 streams) | operand reads |
| P=exp (fused on acc) | ALU on wg0 fragment | idle | none (V13's win) |
| dS = P⊙(dP−D) | ALU + `store_acc_smem` scatter | idle | **fp32 store, 3.8-way** |
| transpose Pᵀ, dSᵀ | `fill_trans_A` re-pack | idle | **full smem W+R** |
| dV, dK | wgmma (persistent acc) | busy | transposed-operand reads |
| dQ + fp32 atomic flush | wgmma + `store_acc_global` + atomics | partly | fp32 staging + atomics |

The idle rows are the "elementwise/reshuffle/staging windows." They are **serial** on the consumer
critical path and every one of them **hits the shared port** (or, for exp, at least stalls the pipe).
Because the shared port is already at 57% SOL, these windows cannot be hidden behind more warps — the
resource they need is busy. That is the mechanism: *tensor idle time is pinned to shared-port-busy
time.*

### 1.2 How cuDNN keeps its pipe 59% busy
Four mechanisms, in decreasing order of what we can plausibly borrow:
1. **No software transpose** — layout aliasing means the transpose window (a whole smem W+R for us)
   simply does not exist. Fewer shared bytes per MMA → the pipe is fed sooner. *(borrowable, §3-primary)*
2. **Two symmetric MMA warpgroups, phase-offset** — while WG_A does its softmax/elementwise, WG_B's
   uninterrupted wgmma stream keeps the pipe busy, then they swap. This *overlaps* the idle windows
   with MMA instead of leaving them serial. *(borrowable but a rewrite, §3a)*
3. **TMA-reduce dQ** (`store_dq()` bulk reduce-add to gmem for hdim<256) instead of per-element fp32
   atomics — moves the dQ reduction off the consumer's shared/atomic path onto the async proxy.
   *(borrowable, §3, low risk)*
4. **Bigger register working set (accepts spills)** — keeps fragments in registers across windows
   rather than round-tripping through smem. This is *why cuDNN spills yet stays compute-bound*:
   it trades local (L1-cached) traffic for shared traffic, and shared is the scarce port. *(see §4)*

Note the FA4 feeds-and-speeds framing generalizes this: the backward's binding resource is
shared-memory bandwidth, not the tensor cores
([Colfax FA4](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)).
Every mechanism above is a way to spend fewer shared bytes per MMA.

---

## 2. FA3's actual backward schedule (what the source shows)

Sourced from the FA3 paper ([arXiv 2407.08608](https://arxiv.org/html/2407.08608v1),
[Tri Dao blog](https://tridao.me/blog/2024/flash3/),
[PyTorch FA3](https://pytorch.org/blog/flashattention-3/)) and the actual Hopper backward mainloop
([`mainloop_bwd_sm90_tma_gmma_ws.hpp`](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp)).

### 2.1 The structure
- **`NumMmaWarpGroups` = 2 consumer warpgroups + 1 producer warp** (`NumProducerThreads = 2·32 = 64`
  threads — a warp-pair, *not* a full warpgroup as in our wg2). Producer issues TMA for Q, dO, K, V.
- **GEMM order per m-block**, with the wgmma `wg_wait` template arg controlling how many async groups
  stay pending:
  1. `S = QK^T` — `wg_wait = -1` (**issue async, do not wait**)
  2. `dP = dO V^T` — `wg_wait = -1` (**issue async**)
  3. `dV = dSᵀ·dO` — `wg_wait = -1`
  4. `dQ = dS·Kᵀ` — `wg_wait = 1` (leave 1 group pending)
  5. `dK = dSᵀ·Q` — `wg_wait = 1`
- **Transpose = smem layout aliasing**, no data movement (§0). `SmemLayoutQt`, `SmemLayoutdOt`,
  `SmemLayoutPdSt` are transposed *views* of the same physical smem read by wgmma descriptors.
- **dQ (hdim<256, i.e. our D=128) = smem-staged then TMA reduce-add to gmem** (`dQacc_use_TMA=true`,
  `store_dq()`), *not* atomics. (Atomics `atomicAdd<float4>` are the `hdim=256` fallback.)
- **dS buffering** `kStages_dS ∈ {1, kStages}`; named barrier `BwdNamedBarriers::PdS` gates P/dS
  writes before the dV/dK/dQ reads.

### 2.2 Is the backward a ping-pong between two warpgroups on different iterations? — **Partly, and this is the key answer to your Q2.**
- The **headline "ping-pong"** described in the FA3 paper is a **forward-pass** technique: two
  warpgroups run QKᵀ / PV on *alternating* iterations, and `bar.sync` forces WG1's GEMMs to schedule
  before WG2's so *"the softmax of warpgroup 1 will be scheduled while warpgroup 2 is performing its
  GEMMs"* ([arXiv 2407.08608](https://arxiv.org/html/2407.08608v1)). That specific 2-stage
  intra-warpgroup GEMM↔softmax overlap is **not** the backward's structure — the backward source is a
  **single-stage compute-then-consume** per warpgroup (softmax(dS) is *not* overlapped with the next
  MMA *within* a warpgroup).
- **The backward's overlap instead comes from (a) two symmetric MMA warpgroups running the same code
  path on different work-slices, and (b) issuing the independent GEMMs (S, dP, dV) async
  back-to-back** (`wg_wait=-1`). While WG_A is in its elementwise/softmax window, WG_B — executing
  the *identical* program with no divergence — has its own wgmma stream in flight on the shared
  tensor core.

**Why this AVOIDS the WG.DP serialization we hit in V14 (the crux):**
Our V14 failure (C7518, WG.DP scoreboard) happened because we kept **wg1's pending dP group alive
across a divergent branch** — `if(wg==0){ S wgmma } else { drain dP }`. At that in-program-order
point, wg0's wgmma **coexists** with wg1's pending group *within one divergent region*, so ptxas
inserts a warpgroup-scoreboard dependency it cannot prove unnecessary
(`reference_cross_tile_async_wgmma_pipeline`). FA3's two consumer warpgroups run the **same code path
with no such divergence** — each warpgroup owns its *own* async scoreboard, its wgmma stream is
**uninterrupted within its phase**, and the two streams are time-multiplexed by the hardware
scheduler, never serialized by a compiler-inserted cross-warpgroup wait. **So yes: the symmetric
two-consumer-warpgroup structure structurally sidesteps the exact serialization that killed our
cross-tile pipeline.** That is the strongest argument for adopting it (§3a).

### 2.3 Recompute
FA3 **recomputes S=QKᵀ in the backward** (it does not store P from the forward): *"there are 2 matmuls
in the forward pass and 5 matmuls in the backward pass, due to recomputation."* **We already do this.**
Recompute trades compute for *not* touching HBM — it does not reduce our *shared* traffic. So the
generic "FA3 recomputes" is a non-lever for us (see §3c).

---

## 3. Candidate structural approaches

| # | Approach | Feasible on our D=128 causal-GQA KV-centric kernel? | Expected Compute-SOL payoff | Risk | Reg/smem impact | CUDA-C reachable? |
|---|---|---|---|---|---|---|
| **P (primary)** | **Transpose-free operand staging** — replace `fill_trans_A`/`fill_trans` with direct transposed-descriptor reads (Major::MN / trans-b) of resident swizzled smem | **Yes, incrementally** — one GEMM at a time; dV/dK use Major::MN (B300-confirmed recipe in memory), dQ's Kᵀ operand similarly | **High** — removes a full smem W+R per transposed operand; directly cuts the binding resource; est. Mem SOL ↓, Compute SOL ↑ several points per GEMM | **Medium** — descriptor/swizzle correctness; our V8 swizzled-A attempt failed once (`reference_hopper_wgmma_swizzled_inkernel_A`) so must ramp-probe first | **Frees smem** (drops `sA_t`/`sB_t` staging for those operands); reg-neutral | **Yes** |
| **a** | **FA3 symmetric 2-consumer-WG ping-pong** — both consumer WGs run the *full* backward on different Q-slices/columns, phase-offset so one's MMA covers the other's softmax | Requires reworking the KV-centric same-tile split; dK/dV must be split by output column (each WG owns D/2) or partial-reduced. Sidesteps WG.DP (§2.2) | **High** — overlaps the currently-serial elementwise/transpose windows with MMA | **High** — full schedule rewrite; V13's S∥dP concurrency + maxed occupancy already capture *part* of this, so incremental gain uncertain | Neutral-to-+ regs; smem unchanged if column-split | **Yes** (proven pattern) |
| **b** | **Persistent kernel** (grid=#SMs + in-kernel work queue) | Low value: we are **already 1 block/SM** (smem+reg limited), so there is no tail-wave or launch-overhead problem to solve. cuDNN is persistent mainly to *amortize its huge setup*; ours is cheap | **~0** for the compute-feeding gap | Low-but-pointless | none | Yes |
| **c** | **Recompute S in backward** | **Already done.** Not a lever (§2.3). Recompute cuts HBM, not shared; DRAM is 11% so no gain | **~0** | n/a | n/a | n/a |
| **d** | **2-CTA cluster / DSMEM** — split a tile across an SM pair, share operands via distributed shared memory | Possible (Hopper clusters real on sm_90a) but the KV-centric persistent dV/dK accumulator would have to be reduced across the cluster; DSMEM adds its own traffic. FA4 uses the Blackwell **2-CTA MMA** for this — *not available on Hopper* (no `tcgen05`); the Hopper DSMEM analogue is weaker | **Low–medium**, speculative | **High** — cluster launch, DSMEM barriers, accumulator reduction; unproven for us | +barrier/DSMEM smem | Yes but hard |
| **e** | **TMA reduce-add dQ** (replace fp32 atomics) — `cp.reduce.async.bulk.tensor.add` to gmem, offloaded to producer | **Yes** — FA3 does exactly this for hdim<256; our producer (wg2) has slack | **Low-medium** — removes atomic contention + fp32 scratch + the convert kernel; cuts a shared/atomic window on the consumer path | **Medium** — TMA-reduce correctness + ordering; must stay bit-identical | Frees the fp32 dQ scratch; may free smem | **Yes** |

### On abandoning the KV-centric persistent dV/dK accumulator
- **Approaches P and e do NOT require it.** Transpose-free reads and TMA-reduce-dQ are drop-in on the
  current accumulation scheme. This is *why P is the recommended first move* — maximum payoff on the
  binding resource for minimum structural disruption.
- **Approach a (ping-pong) DOES require reworking it.** Two symmetric warpgroups each accumulating into
  the *same* persistent dV/dK is a conflict; the clean resolution is a **column-split** (each WG owns
  D/2=64 columns of dV/dK, disjoint writeback, no reduction, halved accumulator regs — the exact
  pattern already in `reference_hopper_multiwg_wgmma_occupancy`) plus an **M-slice** split for dQ (as
  FA3 does: `flash::gemm<dQ_swapAB, M_slice>`). So the accumulator is not *abandoned* but *partitioned*.
  This is the real cost of approach a, and the reason to defer it behind P.

---

## 4. The 0-spill question — is our constraint costing us throughput?

**Short answer: plausibly yes, and for a precise reason.** cuDNN spills 1.04 M local requests (168
regs insufficient) yet hits 59% compute. Spilled registers go to **local memory, which is L1-cached**
(cuDNN's L1 hit is 79.7%). Our alternative — keeping 0 spills — is achieved by **round-tripping
fragments through *shared* memory** instead (the `store_acc_smem` scatters, the `fill_trans` repacks).
The two are not equivalent:

- **Spill traffic hits the L1/local path; our smem round-trips hit the shared path.**
- **Shared is our binding resource (57% SOL); we are trading traffic *onto* the scarce port to keep
  the abundant one (local/L1) empty.** That is backwards for our bottleneck.

So the honest read: **0-spill is not intrinsically valuable — it is valuable only because our smem
round-trips happened to be cheaper than spills would be *at the time we measured*.** Now that shared
is the binding resource, a fragment that lives one extra window in registers (even at the cost of a
spill/reload) may be **cheaper than a shared round-trip**. Concretely: if approach P keeps a
transposed operand as a register fragment rather than re-packing it through smem, a few spills there
could *lower* Mem SOL.

**Recommendation on 0-spill:** stop treating it as a hard invariant. Keep it as a *tie-breaker*, but
**allow ptxas to spill if it removes a shared round-trip on the binding resource** — validate by Mem
SOL, not by the spill counter. Caveat: cuDNN's spills are *badly coalesced* (1/32 bytes/sector, its
single biggest inefficiency, ncu est. 60.8% self-speedup left on its table). So the goal is not "spill
like cuDNN" — it is "don't burn the shared port to avoid a cheap, well-coalesced local access."

---

## 5. Recommended experiment (smallest test that validates/refutes before a rewrite)

### The experiment
**Replace the software transpose on the single dV GEMM (`dV = Pᵀ·dO`) with a direct transposed-descriptor
read**, leaving dK/dQ/everything else exactly as V13. Steps:
1. **Ramp-probe first (non-negotiable).** Before touching the kernel, run a standalone probe (like the
   existing `sw128` probe) that writes a known ramp into the resident swizzled `sP` layout and reads it
   back through the candidate Major::MN / trans-b descriptor, asserting the transposed values match.
   This is mandatory because our one prior in-kernel swizzled-A attempt (V8) failed and the failure was
   never fully root-caused (`reference_hopper_wgmma_swizzled_inkernel_A`); the B300 MN recipe
   (`reference_hopper_wgmma_mn_swizzle_d128`) is the starting descriptor but must be re-confirmed on the
   H200 for *this* buffer's swizzle.
2. If the probe passes, swap only the dV operand staging: drop `fill_trans_A(sA_t, sP)` +
   `fill_trans(sB_t, sdO)` for dV, read Pᵀ and dOᵀ directly from their resident swizzled buffers via
   the transposed descriptor.
3. Profile with `ncu --set full` (3 launches). Compare **shared wavefronts, Mem SOL, Compute SOL,
   and wall-clock** against V13. Verify bit-identical (2e-2).

### Why this is the right first test
- It isolates the **single highest-value mechanism** (transpose elimination) on **one GEMM**, so a
  positive result generalizes to dK and dQ, and a negative result costs one GEMM's effort, not a
  rewrite.
- It touches the **measured binding resource** directly — the metric that moves (shared wavefronts /
  Mem SOL) is unambiguous.
- It does **not** disturb the KV-centric accumulator, the S∥dP concurrency, or V13's fused softmax.

### Expected outcome
- **Success signal:** dV's shared-store/load wavefronts drop; Mem SOL falls a few points; Compute SOL
  rises; wall-clock improves or holds. → roll the same transposed-descriptor read into dK and dQ, then
  add approach **e** (TMA-reduce dQ) and the **0-spill relaxation** (§4). Projected combined landing:
  toward **~2–2.3× off cuDNN** (estimate; the transpose+scatter traffic is the majority of our excess
  shared bytes, but the fp32 exp/dS ALU windows remain).
- **Plateau signal (kill criterion for P):** if the probe passes but the kernel shows **no Mem-SOL
  drop** (the transpose was already hidden), *or* the descriptor read cannot be made bit-identical
  after a bounded effort → **stop pursuing transpose-elimination** and pivot to approach **a** (the FA3
  symmetric-2-WG ping-pong with column-split dK/dV), accepting the larger rewrite risk.
- **Hard kill (both P and a):** if neither transpose-elimination nor a prototyped 2-WG ping-pong moves
  Compute SOL above ~45%, that is the signal the last ~2× is dominated by cuDNN's hand-tuned SASS
  instruction interleave (exact wgmma/ALU scheduling, dual-issue, and register allocation we cannot
  express from CUDA C), and **V13 stands as the practical CUDA-C ceiling.**

### Honest gap estimate
- **Reachable from CUDA C: ~2–2.3× off cuDNN is a defensible target** (from ~3.14×), *if* transpose
  elimination lands. That closes roughly the shared-bytes-per-MMA gap that separates a memory-bound
  from a compute-bound regime.
- **Parity (1.0×) is very unlikely from CUDA C.** cuDNN's remaining edge after the structural moves is
  in effects that live below the CUDA-C abstraction: the precise SASS interleave that overlaps the exp
  SFU window with MMA issue, register allocation that tolerates well-coalesced spills, and possibly
  ptxas scheduling we cannot force. The FA4 team reaching 71% on *Blackwell* required algorithm+kernel
  co-design and hardware features (TMEM, 2-CTA MMA) that do not exist on Hopper — a reminder that the
  top few percent are hardware/compiler-assisted, not free.
- **What only the H200 can confirm:** (1) whether the transposed-descriptor read is bit-identical for
  our specific swizzle; (2) the actual Mem-SOL drop; (3) whether relaxing 0-spill helps or ptxas
  spills badly-coalesced like cuDNN; (4) whether a 2-WG ping-pong's overlap survives at runtime or the
  scheduler leaves the same idle windows. Every magnitude in this doc is an estimate pending those.

---

## Sources
- FlashAttention-3 paper — [arXiv 2407.08608 (HTML)](https://arxiv.org/html/2407.08608v1), [PDF](https://arxiv.org/pdf/2407.08608)
- FA3 blog (ping-pong, warp-spec, GEMM/softmax overlap) — [Tri Dao](https://tridao.me/blog/2024/flash3/), [PyTorch](https://pytorch.org/blog/flashattention-3/), [Together.ai](https://www.together.ai/blog/flashattention-3)
- FA3 Hopper backward mainloop (layout-aliased transpose, `wg_wait`, TMA-reduce dQ, `NumMmaWarpGroups`) — [`mainloop_bwd_sm90_tma_gmma_ws.hpp`](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp), [DeepWiki](https://deepwiki.com/Dao-AILab/flash-attention)
- FlashAttention-4 (backward is shared-memory-bandwidth bound; pipelining co-design) — [Colfax Research](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/), [arXiv 2603.05451](https://arxiv.org/html/2603.05451v1)
- Colfax FMHA / WGMMA tutorials — [WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/), [FA2 on Hopper (arXiv 2312.11918)](https://arxiv.org/pdf/2312.11918)
- Software pipelining / warp specialization theory — [arXiv 2512.18134](https://arxiv.org/pdf/2512.18134)
- Internal: `docs/V13_analysis.md`, `docs/gqa_bwd_py_analysis.md`, `docs/V14_pipeline_design.md`; kernel `src/attention/GQA_bwd.cu` (`fill_trans` :801, `fill_trans_A` :967, `store_acc_smem` :990, `gqa_backward_v13_kv` :4619)
