# V15 Feasibility — Hand-scheduled PTX / inline-asm to tighten the inner-loop schedule

**Kernel:** `gqa_backward_v13_kv` in `src/attention/GQA_bwd.cu` (sm_90a, H200).
**Baseline:** V13 = 8.82 ms / 374 TFLOP/s / **3.14× off cuDNN** (2.81 ms) / 154 regs / 0 spills / 18.75% occ / all-bit-identical.
**Scope:** design/feasibility only — NO kernel changes.

---

## (0) Verdict — NO-GO

**One line:** *ptxas owns the SASS schedule and re-schedules our PTX regardless of how we order inline asm; there is no hand-SASS assembler in the CUDA toolchain; and the V13 wall is an intrinsic RAW scalar dependency chain that no interleave can hide because there is no independent work to interleave into it. Hand-PTX scheduling cannot beat V13 from source — V13 is the toolchain ceiling for this schedule.*

Do **not** attempt a hand-scheduled inline-PTX rewrite of the inner loop. It is a large, fragile, sm_90-pinned undertaking whose reachable gain is ≈0, for three independent reasons developed below. The only remaining moves that can touch the 3.14× gap are *structural* (change the instruction mix / algorithm), not micro-scheduling — ranked in §4.

---

## (1) What instruction-scheduling control we actually have on sm_90 from the CUDA toolchain

The premise "cuDNN wins via SASS-level instruction scheduling, so let's hand-schedule too" assumes we can *author* a SASS schedule. **We cannot.** Concretely, here is the full set of scheduling knobs the toolchain exposes, and what each does *not* do:

### 1a. There is no hand-SASS path. This is the load-bearing fact.
- `nvcc` → NVVM/LLVM → **PTX** (a virtual ISA) → **ptxas** → **SASS** (the real machine ISA).
- **ptxas is a closed-source optimizing assembler and instruction scheduler.** It performs its own list-scheduling, dual-issue packing, stall-count (control-code) assignment, scoreboard/barrier-register allocation, register-bank assignment, and reuse-cache (`.reuse`) decisions when it lowers PTX→SASS. PTX instruction *order is an input, not a constraint* — ptxas reorders freely subject only to data dependencies and memory/barrier semantics.
- `nvdisasm` is **read-only** (disassemble). There is **no assemble-SASS-from-text** tool in the CUDA toolkit, no `-S`-in path, no documented SASS injection. (Third-party projects — MaxAs/CuAssembler/TuringAs — exist for older ISAs, are unofficial, break per driver/ptxas revision, and have **no sm_90 support**.) So the terminal artifact — the instruction interleave that determines whether tensor pipes stay fed — is chosen by ptxas, period.

### 1b. Inline PTX (`asm volatile`) controls *content and PTX ordering*, NOT the SASS interleave.
Our kernel already emits wgmma/ldmatrix/stmatrix/mbarrier/fences as inline PTX (20 asm sites: lines 817, 832–834, 866, 894, 913, 1302–1322, 2490, 2509, 3887–3892, 4237–4239, 4614). Writing the *scalar* FFMA/IADD/EX2 chain as inline asm too would give us:
- **`asm volatile` + `"memory"` clobber** = a *scheduling barrier*: ptxas may not move instructions *across* it, and may not delete/duplicate it. This can only **forbid** code motion — it cannot **compel** a specific downstream interleave. Over-constraining with a wall of `volatile` asms strictly *reduces* ptxas's freedom and almost always yields a **worse** schedule (it defeats the dual-issue packing and latency-hiding motion ptxas does for free). This is exactly why the LDS-prefetch V14 (which forced a load→use hoist) came out **10% slower** — forcing an order ptxas didn't choose was net-negative.
- Non-`volatile` `asm` = ptxas treats it as a schedulable pure op and reorders it like any other. So it gives *less* control, not more.
- **Net:** inline asm lets us pick *which* PTX ops exist and their *relative order as a hint/barrier*, but the SASS schedule (the thing that matters) is still ptxas's. We already exercise the only useful form of this control — the `wgmma.fence`/`commit_group`/`wait_group`, `fence_operandN` (empty `"+f"` asm at line 850), and `fence.proxy.async.shared::cta` — which *fence* the async region correctly. That is ordering-for-correctness, not scheduling-for-speed.

### 1c. The indirect knobs — all already at their useful setting, none inject a schedule.
- `#pragma unroll` (used throughout) — exposes ILP for ptxas to schedule. Already applied to the k-loops and epilogues. It *offers* independent instructions; it does not *place* them.
- `__launch_bounds__` / `-maxrregcount` — change the register budget, which indirectly changes scheduling freedom (more regs → more in-flight values ptxas can hoist). We are pinned at the **170-reg ceiling @ 384 thr**; we cannot give ptxas more registers without dropping the producer warpgroup (occ 18.75%→12.5%). So this lever is spent in the wrong direction.
- **`-Xptxas -O3`** — already the default under `-O3` (our `CMAKE_CUDA_FLAGS = -lineinfo -O3`). ptxas's aggressive scheduler is already on.
- **`-Xptxas --allow-expensive-optimizations=true`** — on by default at `-O3`. Worth confirming it isn't being suppressed by `-lineinfo`: `-lineinfo` (not `-G`) does *not* disable optimization, so the scheduler runs at full strength. (`-G`/`--device-debug` *would* serialize everything — we correctly do **not** use it.) There is no `--schedule=...` or "use my order" flag.
- `-Xptxas --opt-level=3`, `--maxrregcount` per-ptxas — same story: budget/effort knobs, not schedule authoring.

**Conclusion of §1:** The only genuine scheduling control from CUDA C or inline PTX is (a) *forbidding* motion via `volatile` barriers — which the evidence shows hurts — and (b) *exposing* ILP via unroll/registers — already maxed. **We can constrain ptxas; we cannot out-schedule it or hand it a schedule.** ptxas re-schedules regardless of inline-asm ordering.

---

## (2) The hottest inner region, the plausible hand-interleave, and the estimated gain

**Hottest region (from V13 `-lineinfo` stall sampling, `reports/gqa_v13_source.txt`):** the per-tile scalar chain
`S-fragment → exp/mask (softmax) → P → dS → reshuffle (ldmatrix/stmatrix) → readout stores`,
with the warp-stall budget INT-ALU(addr/idx) 33.8% + FP-math(elementwise) 20.5% + MOV/S2R 13.8%; **TENSOR(wgmma) 0.0%**, **SMEM 0.7%**. The tensor cores are *starved* — they idle while the ALU/FP scalar chain crowds the issue slots.

**Could a hand interleave hide this latency?** The question reduces to: *is there independent work to schedule into the stall slots?* The accumulated evidence says **no**:
- The chain is **RAW-serial**: exp consumes S, dS consumes P, the readout consumes dS. Each fp32 step waits on the prior. A schedule can only fill a stall with an *independent* instruction; a serial chain offers none within the tile.
- **Cross-tile** independent work exists in principle (tile N+1's S/dP is independent of tile N's readout) — but pulling it into tile N's stalls is **cross-tile pipelining**, already built and **rejected**: it hit C7518 WG.DP wgmma-serialization across the `if(wg==0)` divergence *and* un-fused V13's genuine S∥dP concurrency → **11% slower** (9.78 vs 8.83 ms). Ping-pong (Blackwell V19 twin) = flat/+3.6% abandoned. ILP-doubling = register/smem NO-GO.
- At 18.75% occ (≈1–3 warps/scheduler) there is **almost no other resident warp** to interleave either — the reason occupancy was the earlier ceiling.

So the hottest region has *nothing to interleave into it*. A hand schedule that is bit-identical to V13's instruction set cannot improve it, because ptxas already extracts the intra-tile ILP (baked into the 154-reg baseline) and the cross-tile ILP is structurally blocked.

**The one theoretical exception — the exp.** wg0's softmax `exp` is IEEE `expf` (no `--use_fast_math`): ~15–20 instr of range-reduction + SFU `EX2` + reconstruction, and the `EX2` is **SFU-throughput-bound** (~256 cyc, 32 EX2/thread @ ¼-rate, *fixed*). A hand interleave could try to slide tile-N readout FFMAs into the EX2 shadow — but (a) the readout depends on dS which depends on P which depends on *this* exp (RAW again), and (b) `__expf` was measured **flat** (removes surrounding ALU but the EX2 throughput floor is unchanged). No slack there.

**Estimated realistic gain from hand-scheduling: ~0% (±noise).** Best plausible case, if a hand interleave found *some* micro-slack ptxas missed, is low-single-digit-percent and easily erased by the `volatile`-barrier over-constraint tax (cf. LDS-prefetch −10%). **This does not move 3.14× toward 2.5×.** That earlier "~2.5×?" hope was predicated on scheduling being the gap; the stall data says the gap is *instruction count / mix* (§4), not interleave.

---

## (3) Effort vs payoff — honest accounting

**Effort (very high, fragile):**
- Hand-write PTX for the entire inner loop's scalar chain (softmax + dS + reshuffle staging + readout), managing register allocation and the exact `.reuse`/dual-issue intent by hand — while ptxas will *re-schedule it anyway* (§1a), so the hand-schedule is a *suggestion ptxas overrides*, not a spec it honors.
- Re-verify **bit-identical** dQ/dK/dV @ 2e-2 (the all-versions-identical property the project guards) after every change.
- **Re-verify on every ptxas / CUDA-toolkit revision** — ptxas scheduling is unstable across versions; a hand-tuned inline sequence that helps on one ptxas can regress on the next. This is a permanent maintenance tax on a single kernel.
- sm_90-pinned: none of it transfers to the Blackwell line (sm_103) the project also maintains, which uses `tcgen05`/TMEM, a different tensor path entirely.

**Payoff:** ≈0% (§2). The premise that "cuDNN's edge is scheduling" is **only partly true and not the actionable part**: cuDNN's real edge is a *different instruction mix* — ~14.7 ALU/MMA vs our ~23, because TMA does addressing in hardware and its tiling amortizes bookkeeping — plus interleave. The interleave rides on *having fewer, more-independent instructions to interleave*. We cannot reproduce that by re-ordering *our* (more numerous, more serially-dependent) instruction stream; we would have to *change the instructions*, which is §4, not hand-scheduling.

**Verdict:** effort/payoff is deeply negative. This is genuinely a *"different project / not expressible from the CUDA toolchain"* wall. The CUDA C source language and inline PTX do not expose SASS scheduling, and the residual gap is not a scheduling gap in the first place.

---

## (4) Fundamentally-different-algorithm alternatives (change the RAW-chain / instruction-mix, ranked)

These do **not** try to hide the chain's latency — they change the *structure* so there is less chain / fewer bookkeeping instructions per MMA. This is the only category with a live path to sub-3×.

1. **128×128 retile (highest value — the real untried lever).** Blackwell's own V16 128×128 retile was its **biggest single win (−35%)**: it collapsed per-tile bookkeeping ~1.70× and *flipped the kernel latency→memory-bound*. That directly attacks our #1 stall (INT-ALU address/index 33.8%): per-*tile* work (loop induction, descriptor construction, base-address setup, barrier orchestration) amortizes over 2× the rows, shrinking the ALU/MMA ratio toward cuDNN's. **Caveat / known blocker:** the Hopper 128×64 retile was NO-GO on the **170-reg ceiling + 227 KB smem cap** (transient acc doubles → ~186–250 regs). 128×128 is heavier still and will *not* fit at 384 thr/18.75% occ — it forces dropping the producer WG (occ→12.5%) and/or 2-block schemes the smem cap forbids. So it is not free; it needs its *own* feasibility pass weighing "bookkeeping-collapse win vs occupancy/producer-overlap loss." **It is the single most promising remaining move and the only one that attacks the actual 34% stall class structurally.** Recommend a dedicated feasibility pass (mirrors Blackwell V16's accounting on Hopper's tighter reg/smem budget).

2. **TMA-store / TMA-reduce the dQ path (mechanical, real ALU reduction).** V13 accumulates dQ via fp32 `atomicAdd` into a scratch + convert. cuDNN uses TMA-reduce. Replacing the per-tile staging + scalar atomic address-gen with `cp.reduce.async.bulk.tensor` removes a chunk of the exact INT-ALU/MOV instructions crowding the issue slots — it *reduces the instruction count* (the cuDNN-gap axis) rather than re-ordering it. Lower ceiling than the retile but lower risk and directly on the stall class. Worth a scoped feasibility pass after the retile verdict.

3. **Recompute / alternative (non-KV-centric) decomposition (speculative).** A different loop nest — e.g. recomputing S rather than staging dS through smem, or a q-centric split — could *shorten the RAW chain itself* (the only thing that has ever paid: V13 fused-softmax −21%, Blackwell V20 chain-shortening). High design risk, unclear it fits the reg/smem budget, and may re-introduce redundant compute. Assess only if (1) and (2) plateau.

4. **2-CTA cluster / DSMEM operand sharing (lowest value here).** Hopper supports thread-block clusters + distributed shared memory (`cluster.map`/`mapa`, DSMEM). It could share KV operands across an SM pair and cut redundant global loads — but our profile shows we are **not** memory-BW-bound (SMEM stall 0.7%, DRAM ~10%); the bottleneck is the scalar ALU/FP chain. Cluster/DSMEM attacks a non-bottleneck. Skip unless a retile flips us memory-bound (as it did on Blackwell), at which point revisit.

**Ranking rationale:** attack the measured stall (INT-ALU/FP scalar chain) by *removing or amortizing instructions* (1, 2) or *shortening the dependency chain* (3) — not by hiding latency (hand-schedule, ping-pong, ILP, cross-tile pipeline — all tried, all flat/negative) and not by cutting bandwidth we aren't limited by (4).

---

## Bottom line

Hand-PTX/inline-asm scheduling is **NO-GO**: (1) ptxas owns the SASS schedule and there is no hand-SASS assembler — inline asm can only forbid motion (which hurts) or expose ILP (already maxed); (2) the hottest region is a RAW-serial chain with no independent work to interleave at our footprint; (3) effort is huge/fragile/sm_90-pinned for ≈0 payoff, and the residual gap is an instruction-*mix* gap, not a scheduling gap. **V13 is the practical CUDA-C + inline-PTX toolchain ceiling for this schedule.** The only live path below 3× is *structural*: **128×128 retile** (top — attacks the 34% bookkeeping-ALU stall, needs its own feasibility pass against Hopper's reg/smem wall), then **TMA-reduce dQ**, then speculative alt-decomposition; cluster/DSMEM is off-bottleneck.
