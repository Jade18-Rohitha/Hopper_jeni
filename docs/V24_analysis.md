# V24 Analysis — shared-store bank-conflict cut (stride 72) — SUB-2×

**Result: 4.9893 ms / 661 TFLOP/s — 1.78× off cuDNN. Bit-identical. −12.5% over V23 (5.70 ms).**
**Broke both the 5 ms and the 2× barriers. Biggest win since the D-split.**

## What changed
The two per-tile fp32 mma D-fragment stores were the L1/shared bottleneck:
- `store_acc_smem` (dP→sdP): stride 65 → **4-way** conflicts, scalar STS.32
- `stage_acc_f32` (dQ→sS): stride 64 → **8-way** conflicts

Both re-padded to **stride 72** → **2-way** (the hard bank floor: even column-pairs reach only
16 banks, so 2-way is the minimum; 1-way is impossible). Stride 72 is even, so the dP store also
packs **STS.32 → STS.64** (half the store instructions). dQ-flush read stride decoupled
(`atomic_flush_stage_s`, coalescing preserved). **Bit-identical** (padding only). 157 regs
(=V23), 0 spill, +3 KB smem.

## Profile — the store wall came down, and it converted

| metric | V23 | **V24** |
|---|---|---|
| shared **store** conflicts | 234.6 M | **4.5 M (−98%)** |
| shared **load** conflicts | 59.8 M | 33.5 M |
| short_scoreboard stall | 0.87 | **0.25** (collapsed) |
| long_scoreboard stall | 2.37 | 2.08 (#1) |
| L1TEX (mem) throughput | 66.4% | **53.5%** |
| SM throughput | 37.5% | **44.0%** |

The store conflicts fell 98% and L1TEX pressure dropped 13 points — because we were genuinely
L1/shared-bound, the cut converted straight to −12.5% wall-clock.

## The lesson — regime-specific verdicts, not permanent ones
Memory had this lever marked **DEAD**: a similar store-conflict fix *regressed −1.8% at V14*.
But that was when the kernel was **latency-bound** — the conflict-% never materialized. At V23
we're **L1/shared-throughput-bound**, so the exact same class of fix *wins big*. Same lesson as
the D-split (called flat twice, gave −20%): **trust the current measurement, not the old
verdict.** The regime changed; the lever came alive.

## V25 — the load side, now unlocked
With stores solved, **loads are the #1 shared conflict (33.5 M)**. And V24 unlocked a clean
follow-on: `sdP` is now **stride 72 (even)**. In V18 the `dS` step read `sdP` as **2 scalar
loads** *only because* the old odd stride 65 blocked float2 alignment. Now it's even → the dS
read of `sdP` can be a single **float2**, halving those loads — the same padding-unlock pattern,
on the load side. Plus any remaining paddable load conflict. The L1/shared vein is live and
converting; sub-1.7× is in reach.

Cumulative: V13 8.82 → V19 7.77 → V22 5.89 → V23 5.70 → **V24 4.99 (1.78×)**.
