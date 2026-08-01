# V23 — SASS-Level Instruction Scheduling: Reachability Analysis

**Scope:** Analysis + plan only. No kernel changes. Baseline V22 = 5.887 ms total
(D-kernel + main + convert), 2.09× off cuDNN (2.81 ms). Question: is cuDNN's remaining
edge a SASS-level instruction-scheduling gap we can close via (a) source ILP, (b) inline
PTX, or (c) CuAssembler cubin patching?

---

## (0) One-line verdict

**SASS instruction scheduling is NOT a reachable win. The wall is WORK + SYNCHRONIZATION,
not schedule.** V22 already issues at *higher* IPC (1.46) than cuDNN (1.33) and at *higher*
occupancy (18.5% vs 15.6%); it is 3× slower for one reason — it executes **3.45× the
instructions** — and its #1 stall is a hardware mbarrier spin waiting for TMA data
(78% of all long_scoreboard, 24% of *every* stall sample in the kernel), which by
construction has no independent work to interleave. Path (a) helps only the ~12% scalar
chain and is largely exhausted; path (b) gives ordering, not latency-hiding; path (c) is not
even viable for sm_90a. **Pursue traffic/work reduction and deeper pipelining (PD=4),
not scheduling.**

---

## (1) The measured cuDNN-vs-V22 gap — it is not a schedule gap

All numbers from `reports/gqa_v22_profile_d128.txt` (main kernel, isolated) and
`reports/sdpa_bwd_d128.txt` (cuDNN `flash_bprop_wgmma_f16_knob_26_64x64x128`), same H200,
same locked clock (~1.35 GHz), same 384-thread/18.75%-theoretical launch.

| metric | cuDNN | V22 main | read |
|---|---|---|---|
| **Executed instructions** | **556 M** | **1,917 M** | **3.45× more work** |
| Elapsed cycles | 3.32 M | 9.98 M | **3.00× more cycles** |
| Issued IPC (active) | 1.33 | **1.46** | V22 issues *better* |
| Achieved occupancy | 15.6% | **18.5%** | V22 has *more* warps |
| No-eligible | **66.7%** | 63.5% | V22 *less* starved |
| Warp-cycles / issued instr | 7.51 | 8.11 | comparable |
| Top pipeline | **Tensor 59.4%** | ALU 26.3% / **Compute-SM 36.4%** | cuDNN is tensor-bound; ours tensor-**starved** |
| `gmma` stall (tensor wait) | — | **0.02 / 8.11 (0.2%)** | our tensor cores essentially never the stall |
| L1TEX throughput | 65.7% | 65.9% | same pipe pressure... |
| L1 hit rate | 79.7% | **15.4%** | ...but ours misses far more |
| Local spills | 1.04 M (spilling) | **0** | our standing advantage, preserved |

**The decisive identity:** cycle-ratio (3.00×) ≈ instruction-ratio (3.45×) ÷ IPC-ratio
(1.46/1.33 = 1.10). We are slower *almost exactly in proportion to the extra instructions we
execute*, slightly offset by our higher IPC. **Scheduling improves IPC. Our IPC already
beats cuDNN's.** You cannot schedule your way out of running 3.45× the instructions. cuDNN
wins with *lower* occupancy and *worse* no-eligible because it simply does 1/3 the work and
keeps that work on the tensor pipe (59% busy) instead of the scalar/L1TEX pipe.

### Exposed-latency localization (PC sampling, `reports/gqa_v22_profile.ncu-rep`)

Warp-cycles-per-issue breakdown (sum 8.11): **long_scoreboard 2.55 (31%)**, barrier 1.40
(17%), short_scoreboard 1.02 (13%), wait 0.99 (12%), selected/issuing 1.00 (12%),
mio_throttle 0.34, not_selected 0.34, math_pipe 0.18, **gmma 0.02**.

Per-SASS-instruction sampling (640,703 samples total) pins *where* the stalls live:

| PC (rel off) | SASS | dominant stall | share |
|---|---|---|---|
| **0x9a50** | `@!P0 BRA` after `SYNCS.PHASECHK.TRANS64.TRYWAIT` | **long_scoreboard 156,438** | **78% of all long_sb; 24.4% of ALL stall samples** |
| 0x1100 | `BAR.SYNC.DEFER_BLOCKING` (after LDG→STS of `sD`) | barrier 26,953 | #1 barrier site |
| 0x64c0 / 0x6200 / 0x9640 / 0xa870 | `WARPGROUP.DEPBAR.LE gsb0` (wgmma wait_group) | barrier ~14k each | consumer draining its own MMA |
| 0x68b0 / 0x7000 | `LEA.HI` / `LEA` (address-gen on smem operands) | short_scoreboard ~18k | the scalar/staging chain |
| 0xef?0 (several) | `STS.64` (dQ/dP fp32 staging) | mio_throttle ~2.5k each | 4.5-way shared-store conflict |
| 0xf120/0xf140 | `REDG.E.ADD.F32` (dQ global atomic) | wait ~1.6k each | fixed-latency reduce |

The **single hottest instruction in the whole kernel** is the mbarrier spin:

```
/*9a40*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR13], R12 ;   // check mbarrier phase
/*9a50*/  @!P0 BRA 0xa5c0 ;                                   // not ready -> spin
```

This is a warp-specialized **consumer parked waiting for the producer's TMA tile**. The
long_scoreboard charged here is a genuine global/TMA *data-arrival* wait. `not_selected` is
only 4.3% — when a warp stalls, there is usually **no other eligible warp to run**; warps are
genuinely blocked, not losing scheduler arbitration.

---

## (2) The hot regions — and whether fillable independent work exists

1. **mbarrier spin (0x9a50, 31% of stall, #1).** Fillable work? **NO.** The consumer
   warpgroup's *only* next work is the tile the producer hasn't delivered. Warp
   specialization means it has nothing else queued. The sole way to give it independent work
   is to have **tile N+1's S/dP already in flight** — i.e. deeper run-ahead (more pipeline
   depth / a second S·dP buffer). That is a **buffering/algorithm** change, not a schedule
   change, and cross-tile S∥dP overlap is already recorded NO-GO (wgmma in-order group
   completion + register ceiling; see agent memories `project_gqa_bwd_v21_crosstile_nogo`,
   `project_gqa_bwd_pingpong_v20_nogo`). The *tractable* version is simply **more producer
   run-ahead depth (PD)** so the producer stays further ahead and the phase is more often
   already flipped when the consumer checks — exactly V22-analysis's PD=4 lever.

2. **`__syncthreads` / `WARPGROUP.DEPBAR` rendezvous (0x1100 + DEPBAR sites, 17% barrier).**
   Fillable? **NO by ptxas.** Barrier stall = warp-execution-duration *imbalance* (warps and
   warpgroups arriving at the fence at different times: producer vs 2 consumers, D-load vs
   compute). ptxas cannot reorder across a barrier. Reducible only by *balancing* the phases
   or *removing* syncs — structural, and largely mined already (V13→V22 removed several).

3. **Scalar address-gen + softmax chain (LEA/IMAD/FFMA on smem operands, 13% short_sb).**
   Fillable? **PARTIALLY, and this is the ONLY schedule-adjacent headroom.** These are the
   consumer's own dependent chain (dS elementwise, staging address math, exp). More source
   ILP (independent accumulators, hoisting the next GEMM's operand-prep above the current
   dS) could let ptxas overlap more. **But** IPC is already 1.46 > cuDNN's 1.33, so ptxas is
   already interleaving well, and V13's register-fused softmax + V17/V20 chain-shortening
   already harvested most of it. Realistic ceiling here: low single-digit %.

4. **Shared-store conflict + dQ atomics (STS.64 4.5-way, REDG, mio_throttle/wait 4–12%).**
   Fillable? Not by scheduling — it's *traffic*. 66% of shared-store wavefronts are bank
   conflicts; 35% of all shared wavefronts are excessive. This is L1TEX pipe waste. Prior
   store-conflict fixes are recorded DEAD (V14: +1.4% or slower, cost precision/regs). Real
   reductions must come from *fewer/reshaped* stores, not de-aliasing.

---

## (3) Path reachability + concrete first change + effort/payoff

### (a) Source-level ILP / reorder / unroll — MARGINAL
- **What it can touch:** only region 3 (the 13% short_scoreboard scalar chain). It cannot
  touch the 31% mbarrier spin or the 17% barrier — those aren't ILP-limited, they're
  data-arrival and rendezvous waits.
- **Why limited:** V22 IPC 1.46 > cuDNN 1.33 means ptxas is *already* filling scalar gaps
  with the ILP present. `#pragma unroll` deeper on the KV loop would blow the 162→>168 reg
  ceiling (occupancy cliff to 0 blocks) with no eligible-warp gain.
- **Concrete first change if attempted:** software-pipeline the *consumer* dS-elementwise so
  tile N's `F2FP/STSM` of dS overlaps tile N's `LEA/IMAD` operand-prep for the dK/dV GEMM —
  give ptxas two independent chains across the `WARPGROUP.DEPBAR`. Verify via SASS load→use
  distance (`nvdisasm -g`), gate on reg count ≤166 and 0 spill.
- **Effort/payoff:** ~1 kernel iteration; **estimate +1–3%.** Not the 2×.

### (b) Inline PTX with explicit ordering / scheduling barriers — NO
- **What it actually gives on sm_90:** PTX-level `asm volatile` + `bar.warp.sync` /
  membar only *constrain* reordering (they insert ordering fences). They do **not** let you
  place SASS scheduling control words (stall counts, read/write scoreboard barrier indices,
  yield) — those are assigned by **ptxas** at the PTX→SASS stage, and ptxas freely reschedules
  around inline PTX unless a barrier forbids it. Barriers *remove* eligible work; they never
  *create* it.
- **Why it can't help here:** the #1 stall is a hardware mbarrier spin. No ordering primitive
  makes the TMA data land sooner or gives the parked warp something to do.
- **Effort/payoff:** **skip.** Ordering control is orthogonal to the bottleneck.

### (c) CuAssembler cubin patching for sm_90a — NOT VIABLE
- **Arch support (verified):** CuAssembler (cloudcores) officially covers **Pascal→Ampere
  (SM60–SM86)** only. **Hopper SM90 is not supported**; sm_90a is further out. Its instruction
  description DB does not encode the sm_90a async opcodes this kernel is built on —
  `SYNCS.PHASECHK`, `UTMALDG.2D`, `HGMMA.64`, `WARPGROUP.DEPBAR`, `FENCE.VIEW.ASYNC` — so it
  cannot round-trip our cubin, let alone re-assemble it.
- **Even if it could:** the payoff is hand-editing control words / instruction order. Our
  bottleneck is a data-arrival spin (region 1) and a rendezvous (region 2) — **neither is a
  ptxas mis-schedule.** Hand-editing the schedule cannot make TMA data arrive earlier or make
  imbalanced warps reach a barrier together. Workflow (dump → hand-edit → reassemble →
  re-validate correctness on every register-allocation-fragile edit) is high-effort and
  brittle.
- **Effort/payoff:** **not viable, and pointless if it were.**

---

## The actual reachable path (not scheduling)

The profile points away from schedule and toward **run-ahead depth** and **traffic**:

1. **PD=3 → PD=4 (deeper TMA pipeline).** *Directly* attacks region 1: more producer
   run-ahead means the consumer's `SYNCS.PHASECHK` more often finds the phase already
   flipped, shrinking the 31% mbarrier spin. V22 freed 32 KB smem (deleted `sO_sw`);
   191,824 + 32,768 = 224,592 B < 232,448 cap, so PD=4's 4th stage fits. Near-trivial (`PD`
   constant), profile points right at it. **Highest expected payoff, lowest risk. Try first.**
2. **Reduce instruction count per MMA** (the real 3.45× gap): fewer staging round-trips,
   collapse address-gen. This is the long game — every instruction removed is ~1:1 cycles.
3. **Shared-store conflict / atomics:** deprioritized — recorded diminishing-returns/DEAD.

**Do not invest in SASS scheduling (a-marginal / b-no / c-not-viable). The 2× is traffic +
synchronization depth, not schedule.**

---

Sources:
- [CuAssembler (cloudcores) — supported SASS generations](https://github.com/cloudcores/CuAssembler)
- [SM90 Hopper architecture features (CUTLASS/DeepWiki)](https://deepwiki.com/NVIDIA/cutlass/7.1-sm90-hopper-architecture)
