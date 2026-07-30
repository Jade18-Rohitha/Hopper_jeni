# V15 — Intra-warpgroup 2-tile ILP feasibility (design pass, NO kernel changes)

Base: `gqa_backward_v13_kv` (Hopper SM_90a). Br=Bc=64, D=128, KV-centric, 3 warpgroups
(wg0/wg1 consumers = 256 thr, wg2 producer = 128 thr), **154 regs, 0 spills,
223,800 B smem, 384 thr, occ 18.75%, 8.82 ms**. Confirmed bottleneck: latency-bound
on the softmax(exp) → dS(P⊙(dP−D)) → readout(reshuffle) RAW scalar chain. Barriers,
ALU, shared, transpose, occupancy all came back **flat** when isolated (V14 etc.).
ILP = unroll the q-tile loop by 2 and interleave two *independent* q-tiles' chains so
tile N+1's independent instructions issue during tile N's dependency stalls.

---

## (0) GO / NO-GO

**NO-GO.** A 2-tile interleave that actually overlaps needs 2× the transient state
live *simultaneously* in **both** the register file **and** the single-buffered smem
scratch. Registers hit **~186–201 > the 170 ceiling** at 384 thr (spills — the same
wall the 128×64 retile hit), and the smem scratch-double is **+59.1 KB → 283 KB >> 227 KB**
(cannot fit at any thread count without gutting the operand pipeline). Every
granularity that fits the budget does so by serializing the two tiles' shared scratch/
accumulators — which destroys the exact overlap ILP is meant to buy. Same
"envelope/benefit doesn't survive the constraint" failure mode as the retile.
**V13 stands as the CUDA-C ceiling for this schedule.**

---

## (1) PEAK LIVE-REGISTER BUDGET — the primary result

Ceilings (both blocks 1/SM, 65536 regs/SM):
- **384 thr (keep producer):** 65536/384 = **170 regs/thread** for 0 spills. V13 uses 154 (16 spare).
- **256 thr (drop producer):** 65536/256 = **256 regs/thread**. Comfortable — but occ → 12.5% and the async TMA producer + producer-side D-rowsum are gone.

**What doubles vs what is shared under a 2-tile interleave**

| State | Per-tile size | 2 tiles in flight | Note |
|---|---|---|---|
| `dv[32]` persistent | 32 | **32 (shared)** | both q-tiles accumulate into the SAME dv (same KV tile) → NOT doubled. Confirmed: dv/dk are block-owned, associative `+=` from both tiles' MMAs. |
| `dk[32]` persistent | 32 | **32 (shared)** | ditto |
| transient `acc[32]` (S/dP gemm out, then dQ gemm out — one at a time in V13, scoped `{}`) | 32 (peak) | **64 (doubled)** | the overlap *requires* tile N's acc live while tile N+1's wgmma produces its acc. Staging N's acc to smem first to free it = no register overlap = no benefit. |
| control / addr / parity / mbar / misc | 58 (measured residual) | **58 + 4…15** | second tile carries its own q_row0₂, s₂, cpar/dpar₂ parity, smem base ptrs. Retile (one wider logical tile) shares this; ILP does **not** → ILP ≥ retile on regs. |

**Peak live regs/thread by interleave granularity (384 thr):**

| Granularity | dv/dk | transient (peak phase) | control | **peak** | vs 170 | vs 256 |
|---|---|---|---|---|---|---|
| **Baseline V13 (serial)** | 64 | 32 | 58 | **154** | fits (−16) | fits |
| **Minimal — exp-only overlap** (only S-phase acc doubled) | 64 | 64 (S phase only) | +4 | **~190** | **+20 spill** | fits |
| **Fine — phase-paired** (same-phase accs of N,N+1 both live, staged promptly) | 64 | 64 | +8 | **~194** | **+24 spill** | fits |
| **Coarse — whole-chain** (tile N chain, then N+1; -O3 overlaps) | 64 | 64 | +15 | **~201** | **+31 spill** | fits |
| *(reference) 128×64 retile* | 64 | 64 (M-doubled acc) | shared | **~186** | +16 spill | fits |

**Reading:** NO 384-thr granularity fits ≤170. The floor is ~186 (=retile), and ILP is
strictly *above* it because it carries a second independent tile's control state the
retile folds into one. The +32 transient delta is intrinsic — it is the register-level
expression of "two independent chains overlapping." Remove it and the overlap is gone.
Fits only at 256 thr (drop producer, occ 12.5%).

---

## (2) DOES THE LATENCY-HIDING SURVIVE THE REGISTER CONSTRAINT? — the crux

**No.** Two escape routes, both self-defeating:

- **Spill at 384 thr (186–201 regs):** on a *latency-bound* kernel, spills push the hot
  `acc` fragment to local memory (L1/L2). The RAW chain we are trying to shorten now
  routes register→local→register — it gets **longer**, not hidden. Spilling to buy ILP
  on a latency wall is anti-causal. (Consistent with the retile finding: 186 regs → the
  envelope benefit did not survive.)

- **Drop to 256 thr to fit registers:** the register ceiling opens (256), but you delete
  the producer warpgroup. That costs (a) occupancy 18.75%→12.5% — and on a latency-bound
  kernel *occupancy is the warp-level latency-hiding mechanism*; you'd be removing
  warp-level hiding to add instruction-level hiding, a wash at best; (b) the async TMA
  prefetch overlap and the producer-side lagged D-rowsum fold back into the consumers,
  re-exposing memory latency the producer currently hides. You'd trade a proven hiding
  mechanism for an unproven one — the definition of flat.

**Why register-feasible ⇒ no overlap.** The only way to keep peak ≤170 at 384 thr is to
NOT hold both tiles' `acc` live — i.e. drain tile N's acc to smem before issuing tile
N+1's producing wgmma. That is serialization; the sole remaining overlap is the
intra-tile ILP the compiler *already* extracts (baked into the 154-reg / 8.82 ms
baseline). So the register-feasible granularity delivers ~0 incremental overlap. This is
the go/no-go: feasible ⇒ no benefit; benefit ⇒ infeasible.

---

## (3) SMEM BUDGET

V13 = 223,800 B; H100/H200 opt-in cap = 232,448 B → **8,648 B headroom.**

**Single-buffered scratch that two independent tiles collide on** (must double for genuine
overlap — else tile N+1 clobbers tile N mid-chain):

| Scratch | V13 size | ×2 adds |
|---|---|---|
| `sP` (P then dS) | 9,216 | +9,216 |
| `sS` (dP-side + dQ stage) | 16,640 | +16,640 |
| `sdP` | 16,640 | +16,640 |
| `sA_t` (ldmatrix→stmatrix reshuffle) | 16,640 | +16,640 |
| **scratch-double total** | | **+59,136** |

223,800 + 59,136 = **282,936 B >> 232,448 B (over by 50.5 KB).** Does not fit.

Worse, the operand buffers `sQ_sw[2]/sdO_sw[2]/sdO_pl[2]/sO_pl[2]` are the producer's
2-deep pipeline. Consuming two q-tiles at once occupies **both** stages, so the producer
runs 0 tiles ahead (prefetch overlap gone) unless deepened to [4] — **+131 KB**,
categorically impossible. So even the operand side fights the interleave.

Minimal exp-only variant needs only `sP`×2 = +9,216 → 233,016 B, which is **568 B over**
the cap (trimmable via padding), but it also busts registers (~190 > 170) *and* captures
only ~1 of the ~5 RAW links (the exp), not the reshuffle/wgmma/barrier links that the
10-consumer_sync structure spends most time on. Minimal cost, minimal benefit, still
infeasible.

---

## (4) HONEST ESTIMATE + CONCLUSION

**Estimate:** even if forced to build, expected delta ≈ **flat to negative** (−5% … +2%),
matching the retile (~186 regs, benefit did not survive) and the ping-pong NO-GO
(cross-tile MMA overlap flat vs a scalar-chain-bound attn-bwd). Register spills or
occupancy loss cancel any ILP gain.

**Smallest validating experiment (only if empirical proof is wanted):** a *paper*
2-tile-unrolled variant at **256 thr** (drop producer, fold TMA+D into consumers, ILP the
scalar chain) with single-buffered scratch shared serially — measure vs a 256-thr non-ILP
control (NOT vs the 384-thr V13). If ILP were real it would beat its own 256-thr control.
Prediction: within noise. This isolates ILP from the occupancy/producer confound. Do NOT
attempt at 384 thr — it is register-infeasible by construction (table §1).

**Conclusion:** intra-warpgroup 2-tile ILP joins the NO-GO pile. It is register-infeasible
at 384 thr (≥186 > 170, ≥ the retile floor) and smem-infeasible at any thread count
(+59 KB scratch-double >> 8.6 KB headroom); the only budget-feasible framing (256 thr /
serialized scratch) removes the overlap or the occupancy that would justify it. The RAW
scalar-chain latency is intrinsic to this KV-centric, single-buffered-scratch, 10-barrier
schedule and cannot be hidden by ILP within the Hopper register/smem envelope.
**V13 (154 regs, 0 spills, 8.82 ms) is the CUDA-C ceiling for this schedule; further gains
require a different schedule (fewer barriers / shorter chain / tcgen05-class async on
Blackwell), not more ILP.**
