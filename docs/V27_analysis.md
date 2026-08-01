# V27 Analysis — break the sP-reuse chain (separate sDS + dV overlap)

**Result: 4.6752 ms (median) / 705.5 TFLOP/s — ~1.68× off cuDNN. Bit-identical. −0.77% over
V26 (4.7117 ms).** 157 regs, 0 spill, 204,880 B smem (+8 KB for `sDS`, under cap). First win
after correctly diagnosing the kernel as **consumer-bound**.

## Why — the V26 profile settled the target
V26 is **consumer-bound, not feed-bound**: the #1 "stall" (long_scoreboard 26.7% of all stall,
one `@!P0 BRA`) is the **producer idling on `empty[]`** — it fills all pipeline buffers and
waits for the slower consumer; the data is on time (PD=4 *regressed* +3.6%, proving deeper
buffering can't help). The real wall is the consumer's **serial 6-GEMM chain** (barrier 30% +
wgmma-wait 21%), unhidable because occupancy is hard-locked at 18.75% (157 regs × 384 threads
*and* 196 KB smem both pin 1 block/SM). Local memory was a red herring (99.2% L1-hit, 0.46% of
peak). So the only lever is **shortening the consumer critical path**.

## What changed
The chain serialized because `dS` **overwrote `sP` in place**, forcing dV (which reads `sP`) to
fully complete before dS. V27 gives `dS` its own **`sDS`** buffer (identical `__align__(1024)`
swizzle layout as `sP`):
- `dS = P ⊙ (dP − D)` reads `sP` (P), writes **`sDS`**; `dK`/`dQ` read `sDS`.
- `dV` split into **issue** (`run_gemm_dVdK_half_te_issue`, no wait) + **wait** (`_wait`): issue
  the dV wgmma, run the dS elementwise, *then* wait dV — the wgmma overlaps the elementwise.
- The **dV→dS WAR barrier (`@8645`) is removed** (dV and dS now both only *read* `sP`).

SASS confirmed the schedule held: `bar → dV ARRIVE+4×HGMMA → dS(F2FP/STS) → dV DEPBAR wait →
bar`, no barrier between dV-issue and dS. 256-thread consumer barriers **9 → 8**.

## Profile — both predicted stalls dropped

| metric | V26 | **V27** |
|---|---|---|
| **barrier** stall | 1.29 | **1.10** (the removed `@8645`) |
| **wait** (wgmma) stall | 1.04 | **0.93** (dV wait hidden under dS) |
| long_scoreboard | 1.98 | 1.91 (producer idle, ~unchanged) |
| **Executed IPC** | 1.87 | **1.95** |
| warp-cyc / issued inst | 6.33 | 6.08 |
| Compute SOL | 46.7% | 48.7% |
| No-eligible | 53.2% | 51.2% |
| Elapsed cycles | 7.83 M | 7.74 M |

Notably **instruction count *rose* +3.2%** (1.924 B → 1.985 B — the extra `sDS` stores and the
issue/wait split), yet it still won: the scheduling gain (IPC 1.87→1.95, both target stalls
down, more eligible warps) more than offset the added work. A pure latency/scheduling win —
which is the right kind for a consumer-serial, occupancy-locked kernel.

## Correctness (bit-identical, H200-confirmed)
Same computation; `dS` values written to `sDS` (identical swizzle) instead of `sP`; dK/dQ read
them; dV reads the unchanged `sP`. Deferring the dV wait doesn't change the accumulator. No new
hazard: between dV-issue and dV-wait, `sP` is only read; the next `sP` overwrite is the next
tile's `fused_p`, a full iteration and several barriers away. dQ/dK/dV all `Values match!`.

## V28 — extend the trick to dK
`dK` still issues + waits inline. Same pattern: split it issue/wait and overlap the dK wgmma
with the following dQ GEMM+store (both read `sDS` = RAR, no conflict), deferring the dK wait
past dQ. Occupancy stays locked, so the residual gap is cuDNN's structural edge (tighter
schedule at 15.6% occ), but the serial-chain lever still has links to shorten.

Cumulative: V13 8.82 → V24 4.99 → V26 4.6853 → **V27 4.6752 (~1.68×)**.
