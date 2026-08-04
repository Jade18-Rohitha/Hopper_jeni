# V39–V41 Analysis — the instruction-count reframe (STSM / LDMATRIX)

**Result: V38 3.885 → V41 3.761 ms (min 3.6995), all bit-identical, 0 spill, 168 regs.** The first real
movement since the scheduling plateau, and it came from finally aiming at the right axis.

## The reframe (why the scheduling work stalled)

We profiled cuDNN's `flash_bprop_wgmma` kernel directly (`reports/sdpa_bwd_d128.ncu-rep`) against our
V38 and the comparison inverts everything the prior analysis assumed:

| metric | cuDNN | us (V38) |
|---|---:|---:|
| **total instructions** | **556 M** | **1.64 B (2.95×)** |
| IPC (issued/cyc) | 0.33 | **0.48** |
| long_scoreboard | 1.76 | 1.88 |
| barrier | 1.88 | 1.22 |
| wait | 1.05 | 0.89 |

**cuDNN's per-instruction efficiency is worse than ours** — higher stalls, lower IPC. It wins purely by
executing **⅓ the instructions.** So the scheduling levers (rolling window, barrier cuts) were the wrong
axis; our higher IPC means cutting *instructions* converts ~directly to time.

Verified against this exact H200 + shape via `baseline_gqa.py`: cuDNN/SDPA bwd **2.78–2.96 ms**, eager
2.7 ms — real, not an artifact. (The Python's Triton bwd reference is broken, 17.9 ms / dQ err 3e3 —
ignore it; SDPA=cuDNN is the competitor.)

## Where the 2.95× actually lives (SASS, per-`.text`)

Correction to the first cut: our **static** footprint is *leaner* than cuDNN's (EX2 64 vs 96, addr-INT
878 vs 1590, BRA 56 vs 228). So the 2.95× is entirely **dynamic** — how many times lean code runs:

- **670 M shared `LDS` = 42% of all instructions**; L1TEX/LSU pipe at **63% of peak — the dominant pipe.**
- Load-source ranking (SASS + trip-weighted):
  | source | share |
  |---|---:|
  | dQ store→read-back round-trip (`atomic_flush` reads `sS`/`sdP`, 32 dyn LDS/thread, both wg) | **~76%** |
  | `fuse_dS` cross-wg `sP` read (16 LDS/thread, wg1) | ~19% |
  | `sLSE`/`sD` | ~5% |

This is the column-split L1TEX tax, quantified.

## The levers (all bit-identical, profile-verified)

| ver | change | instr | cycles | L1TEX | median |
|---|---|---:|---:|---:|---:|
| V38 | scalar baseline | 1.64 B | 6.56 M | 59.5% | 3.885 |
| V39 | STSM `fused_p` (P→`sP`) | — | — | — | ~3.90 |
| V40 | + STSM `fuse_dS` (dS→`sDS`) | 1.598 B | 6.20 M | 63.3% | **3.774** |
| V41 | + LDMATRIX `sP` read | 1.561 B (**2.81×**) | 6.17 M | 63.5% | **3.761** |

- **STSM** (`stmatrix.m8n8.x4` → ptxas emits cuDNN's exact `STSM.16.M88.4`) replaces per-element
  swizzle-index INT-ALU + scalar `STS` with one hardware fragment store. Key validation: STSM output is
  readable by our transpose-elimination `make_desc_sw128_MN` (non-`.trans`), and for `sDS`, by **both**
  Major::MN (dK) and Major::K (dQ) — one layout serves both.
- **LDMATRIX** (`ldmatrix.m8n8.x4` → `LDSM.16.M88.4`) is the exact inverse: reads P back into the same
  fragment registers. `fuse_dS` sP read: 16 scalar LDS/thread → 4 LDSM. `LDS` 24→8 static.

## The load-bearing insight from the profiles

**STSM/LDMATRIX cut instructions, not bytes.** `STSM.16.M88.4` still moves 512 B/warp through L1TEX —
so the 63% LSU byte-wall is *unchanged* across V40→V41 (63.3→63.5%). The wins are pure issue-pressure
relief (cycles drop slightly more than instructions), which is why each is only ~0.3%. **To move the
63% wall we have to cut shared-memory bytes, not instructions** — and the byte-heavy target is the dQ
round-trip (76% of loads + its store side).

## Next (V42, in flight)

Kill the dQ round-trip — atomicAdd the dQ fragment direct to global, uncoalesced. It removes LSU bytes
(off the 63% pipe) at the cost of more L2 atomics (L2/DRAM idle at 22%). The direct-atomic lost +17% on
V36 pre-STSM, but LSU wasn't the bottleneck then; now it is, so the trade flipped — an empirical
question. (The coalesced-via-shuffle variant is a NO-GO: ~250 `SHFL`/thread to transpose mma-C→row-major.)
If V42 regresses, the round-trip is confirmed near-optimal and the structural fix is dQ re-parallelization
to drop the atomics entirely.

Roofline: still ~1.34× off cuDNN's 2.8 ms — but the gap is now a *quantified* instruction/byte count with
a ranked worklist, not a wall.
