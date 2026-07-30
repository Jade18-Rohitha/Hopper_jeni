# V16 Analysis — transpose-elimination (direct swizzled-`sP` wgmma read)

**Result: 8.7065 ms / 378.9 TFLOP/s — fastest median, but ~FLAT vs V15 (8.7136 ms, inside noise). Bit-identical, 0-spill.**

The profile explains the flat wall-clock exactly: **transpose-elim traded memory for ALU, and the two nearly cancelled.**

## What changed
V15 built Pᵀ (for dV) and dSᵀ (for dK) via `sp_to_sAt_v12<true>` — an ldmatrix→stmatrix
round-trip that **writes then reads `sA_t`** in shared. V16 deletes that: `sP` is written
in 128-B-swizzled layout, and the dV/dK GEMMs read Pᵀ/dSᵀ **directly** through a transposed
wgmma descriptor (`make_desc_sw128_MN`, trans-a=1, both operands Major::MN — the HW-verified
arrangement). The two `sp_to_sAt<true>` calls and their 2 `consumer_sync` barriers are gone.
The `__align__(1024)` swizzle-atom fix (the root-caused gotcha) landed clean → bit-identical.

- regs 154 → **152**, spills 0, smem 223,800 → 223,288 B (−512 B; `sA_t` retained for the
  dQ stage + epilogue writeback), consumer barriers −2.

## Profile — the memory side won, the ALU side lost

| metric | V15 | **V16** | direction |
|---|---|---|---|
| short_scoreboard stall (#1) | 2.16 | **1.89** | ✅ −12% (shared-load latency) |
| barrier stall | 1.70 | **1.50** | ✅ (2 deleted syncs) |
| wait stall | 1.64 | 1.52 | ✅ |
| long_scoreboard | 1.02 | 0.94 | ✅ |
| memory throughput | 57.6% | **55.2%** | ✅ less shared traffic |
| SM throughput | 34.3% | 37.3% | ✅ |
| **ALU pipe** | **18.7%** | **23.4%** | ❌ **regressed** |
| uniform pipe | 1.8% | 2.6% | — |
| FMA pipe | 11.1% | 12.7% | — |

**The transpose-elim did exactly what it was supposed to on the memory side** — the #1
short_scoreboard stall dropped 2.16 → 1.89, barriers dropped, memory throughput fell (less
shared traffic). **But the ALU pipe jumped 18.7 → 23.4%** — writing `sP` in swizzled layout
requires computing the `sw128_idx` swizzle index (an XOR + shifts) *per element*, on the
same write path that was previously a plain strided store. That extra vector-integer ALU
cancelled the memory-latency saving → net flat wall-clock.

## This finally explains "transpose-elim was flat before"
It was never a no-op — it's a **rebalance**: it moves cost from the shared-memory round-trip
(latency) onto the vector-ALU pipe (swizzle-index math). At V13/V14 the two cancelled and it
looked like a dead lever. The profile now shows *why*, and points at the real V17 target.

## V17 lever (measurement-driven, same method as V15)
The memory win is real and currently masked by the swizzle-write ALU. **Reduce the
`sw128_idx` computation cost** — compute the swizzle index incrementally / hoist its
invariant part (exactly the V15 playbook, applied to the swizzled-`sP` write in
`fused_p_from_acc_v16` and the `dS = P⊙(dP−D)` step). If the ALU pipe comes back down toward
V15's 18.7% while keeping V16's short_scoreboard 1.89, the memory-side win converts to
wall-clock. V16 is the right structural base for that: bit-identical, −2 regs, −2 barriers,
less memory-bound.
