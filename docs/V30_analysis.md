# V30 Analysis — __launch_bounds__(384,1) + setmaxnreg scaffolding

**Result: 4.3222 ms (median) / 761 TFLOP/s — ~1.55× off cuDNN. Bit-identical. −1.86% over V29
(4.4040 ms).** 168 regs (was 157), 0 spill. **2.04× total speedup** (8.82 → 4.322).

## What changed
Added `__launch_bounds__(384, 1)` to the kernel and the `setmaxnreg` scaffolding
(`reg_dec_producer_v30` = `setmaxnreg.dec 40` in the producer, `reg_inc_consumer_v30` =
`setmaxnreg.inc 232` in the consumers). Code otherwise identical to V29 → bit-identical.

**The win was the launch bounds, not setmaxnreg.** The backward kernel had *no* launch bounds,
so the compiler self-capped at **157** regs; `__launch_bounds__(384,1)` lets it use up to 170,
and it took **168** — the extra headroom held more live values and cut register-shuffle
instructions (executed instructions 1.758 B → 1.712 B, −2.6%). **More registers = faster even
at the same 1-block/SM occupancy** (predicted neutral, got −1.86%).

**setmaxnreg is dormant here** — the compiler *drops* `setmaxnreg.inc 232` when the consumer
only needs 168 (verified: 0 `SETMAXNREG` in the SASS). It activates only once the code actually
uses >170 regs, i.e., the V31 ping-pong's doubled accumulators.

## Profile
Executed instructions 1.712 B (−13.7% cumulative since V27). Memory-bound (61% SOL vs 45%
compute). Occupancy locked 18.54%, No-Eligible 54.5%, long_scoreboard back to 2.04 (producer
idle). The wall is scheduling/eligibility — which is what the ping-pong targets.

## ⚠ V31 obstacle — the FA3 ping-pong may not fit the backward's smem
The forward keeps `S`/`O` in **registers** (RS-wgmma: it packs P into `pa[32]` register operands
for P@V), so its smem is only K/V/Q (229 KB). **The backward stages `sP`/`sDS`/`sS`/`sdP`/`sA_t`
in smem** because its wgmma A-operands are *transposed* (Pᵀ for dV, dSᵀ for dK) and read from the
128-B-swizzled smem. Two *independent* consumer warpgroups (the ping-pong premise) need **two
copies** of those working buffers (~143 KB) plus 2× K/V — ~256–307 KB, over the 232 KB Hopper
cap. So the full ping-pong needs a deeper redesign (RS-wgmma to move P/dS into registers — hard
for the transposed dV/dK operands) or a lighter cross-q-tile pipeline that reuses buffers.

Cumulative: V13 8.82 → V28 4.4927 → V29 4.3990 → **V30 4.3222 (~1.55×, 2.04× total)**.
