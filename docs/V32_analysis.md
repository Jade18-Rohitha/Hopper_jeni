# V32 Analysis — move the LSE load into the producer (prefetch like D)

**Result: 4.0717 ms median / 815.5 TFLOP/s — ~1.45× off cuDNN. Bit-identical. −3.6% over V31
(4.2244 ms), min 3.9409 ms (broke 4.0).** 168 regs, 0 spill, 186,960 B smem. dQ/dK/dV all
`Values match!`, max_abs identical (1.953e-3 / 3.906e-3 / 3.125e-2). Cumulative: 8.82 → **4.07 ms,
2.17× total**.

## Why — a *subtraction*, not a reshuffle (user-spotted)
Since V22, D was precomputed and prefetched by the producer (wg2) into `sD` alongside the TMA
loads. LSE was the odd one out: the **consumer** still did `sLSE[tid] = d_LSE[…]` in-loop (a global
`LDG`+`STS`), guarded by V26's wg0-scoped barrier. V32 gives LSE the same treatment as D:

- `sLSE` → **PD-deep** `sLSE[PD][Br]`; the producer's `do_dload` now loads `d_LSE[dbase+pl]` into
  `sLSE[s][pl]` right next to `sD`, signaled by the **same `d_ready`** mbarrier.
- The consumer waits `d_ready` **once, early** (covers both LSE and D) and reads `sLSE[s]` from smem.
- **Deleted:** the consumer's LSE `LDG`+`STS` **and** the wg0-scoped `consumer_sync` (V26's barrier).

Crucially this adds **no L1TEX contention** — the producer's `LDG` is on a warpgroup that's
otherwise idle, and the consumer just does one extra smem read it was already doing. Pure removal.
This is why it converted where five *reshuffles* (dK∥dQ, PD=4, P×4-shuffle, cross-iter-flush,
sLSE-sync-move) all went flat-or-worse: those moved work *onto* the L1TEX port; this took work
*off* the consumer entirely.

## Profile — cleaner scheduling, same wall pushed harder

| metric | V30 | **V32** |
|---|---|---|
| Elapsed cycles | 7.17 M | **6.70 M (−6.6%)** |
| Executed IPC | 1.82 | **1.93** |
| No-Eligible | 54.5% | **51.9%** |
| Eligible warps/sched | 0.62 | **0.70** |
| Warp-cyc / issue | 6.52 | **6.16** |
| Instructions | 1.712 B | **1.694 B (−18 M)** |
| L1/TEX throughput | 61.3% | **65.8%** |

The −18 M instructions are the deleted LSE `LDG`/`STS`+barrier; IPC and eligibility rose because
the freed wg0 warp no longer parks on the global-load latency. Note **L1/TEX climbed to 65.8%** —
we cut *cycles*, not L1TEX traffic, so the same tensor-feed work is now packed into a busier
kernel. Occupancy unchanged (1 block/SM, 18.5%); DRAM 22%, compute 48% — neither is the bound.

## Stall breakdown (pcsamp) — the barrier deletion is visible

| stall reason | V31 | **V32** | Δ |
|---|---|---|---|
| long_scoreboard (L1TEX feed) | 136.8 K | **122.8 K** | −10.2%  — **#1** |
| wait (`wgmma.wait_group`)    | 73.1 K  | **70.4 K**  | −3.6%   — **#2** |
| barrier (`consumer_sync`)    | 80.8 K  | **67.6 K**  | **−16.3%** — #2→**#3** |
| short_scoreboard             | 23.4 K  | 25.3 K      | +8% |
| mio_throttle                 | 5.9 K   | 4.8 K       | −19% |

The barrier stall fell hardest and dropped a rank — the deleted wg0-scoped LSE barrier, exactly as
intended. #1 remains `long_scoreboard`: the **inherent HGMMA operand fetches** feeding the tensor
core from swizzled smem (33.5 M shared-load requests, 1.8-way, 0 *excessive* — not ours to fix),
plus the dS-loop's `sP`/`sdP` reads.

## Where next
The wall is L1TEX bandwidth (65.8%, #1 long_scoreboard). Bit-identical levers that *reduce* L1TEX
traffic are exhausted (RS dead, functional-split dead, bank conflicts inherent). What's left:
- **More subtractions** — the barrier (#3, 67.6 K) still has ~5 `consumer_sync`s; if any guards an
  intra-warpgroup hazard it can be scoped/deleted (the pattern that's working). Needs source-line
  attribution to target.
- **bf16-`sdP`** — the only lever that *shrinks* L1TEX traffic (halves the dP exchange read), but
  it rounds dP before `dS=P·(dP−D)` → below cuDNN's precision (cuDNN keeps dP fp32 per FA2/3) →
  breaks bit-identical.

Roofline unchanged: cuDNN 2.8 ms = 37% MFU (beatable in principle); the Hopper *column-split* wall
is ~3 ms; sub-2.8 needs row-split → Br=128 → smem-infeasible → Blackwell TMEM.

Cumulative: V13 8.82 → V22 5.89 → V26 4.6853 → V30 4.3208 → V31 4.2244 → **V32 4.0717 (~1.45×)**.
