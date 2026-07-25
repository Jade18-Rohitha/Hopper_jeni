# V8 Analysis — warp-shuffle D-rowsum reduction (Nsight Compute, D=128)

**V8 = V7 + Blackwell's warp-per-row `__shfl_down` reduction for `D[r]=rowsum(dO·O)`**, replacing
V7's serial 128-deep fp32 accumulation over strided plain reads. Correct (bit-identical to V1–V7),
**21.16 ms / 155.75 TFLOP-s**, 0 spills. This is a **36% wall-clock win** over V7 (33.0 ms) from a
change I had scoped as a minor "item #3" — the profile explains why it was actually the top lever,
and confirms **occupancy is now unambiguously the ceiling**.

> The swizzled-A conflict-kill path (the *intended* V8) is **confirmed dead**: the `-DV8_DEBUG`
> self-test measured `max|Δ| = 2.074` for the swizzled-Major::K-A + wgmma `trans-b=1` operand
> combo. That combo is invalid on Hopper wgmma; the scaffold remains in-file as documented proof.
> V8 ships the *safe* win (the D-rowsum reduction) instead.

- **Kernel:** `gqa_backward_v8_kv<64,64,128>` — grid `(8,4,64)=2048`, block 128 (1 warpgroup).
- **Source:** `reports/gqa_v8_profile_d128.{ncu-rep,txt}` (`ncu --set full`, 3 launches, <0.1% spread). ncu base 1.33 GHz → 27.88 ms (wall 21.16). Read %s.
- Compare: `V7_analysis.md`, `gqa_bwd_py_analysis.md` (cuDNN 2.78–2.96 ms).

---

## 0. Headline — the D-rowsum was a serial-chain + conflict double-whammy

> V7's D-rowsum ran **every iteration** with only **64 of 128 threads awake**, each doing a
> **128-deep dependent fp32 add-chain** over **32-way-conflicting** strided plain reads of
> `sdO_pl`/`sO_pl`. At **6.25% occupancy (1 warp/scheduler)** there is nothing to hide that serial
> latency behind. Replacing it with a warp-per-row `__shfl_down` reduction (5 shuffle steps, all
> lanes active, coalesced reads) **collapsed both problems at once**:
> - **Shared-LOAD bank conflict: ELIMINATED** — it was V7's dominant table entry (11.3-way, 91% of
>   3.48 B wavefronts); in V8 it is **gone from the flagged tables entirely**. Total shared
>   wavefronts **4.29 B → 1.14 B (−73%)**, excessive **3.55 B → 0.38 B (−89%)**.
> - **Memory SOL 63.0% → 33.4%** (no longer memory-bound), **Compute SOL 13.8% → 22.5%**, **IPC
>   0.56 → 0.90**, warp-cycles/issued **7.21 → 4.43**.

**Lesson (important for the roadmap): at 6.25% occupancy, serial dependent chains cost more than
raw conflict way-ness.** The single biggest V7→V8 win came from removing a *latency chain*, not
from a descriptor/swizzle trick.

---

## 1. V7 → V8 delta
| Metric | V7 | V8 | Δ |
|--------|-----|-----|---|
| **Wall-clock** | 33.0 ms | **21.16 ms** | **−36%** |
| TFLOP/s | 99.96 | **155.75** | +56% |
| Memory SOL | 63.0% | **33.4%** | −30 (no longer mem-bound) |
| Compute SOL | 13.8% | **22.5%** | +63% |
| IPC | 0.56 | **0.90** | +61% |
| No-eligible | 86.1% | **77.4%** | −9 |
| Warp cycles / issued | 7.21 | **4.43** | −39% |
| Shared-LOAD conflict | 11.3-way / 91% of 3.48 B | **eliminated (not flagged)** | gone |
| Total shared wavefronts | 4.29 B | **1.14 B** | −73% |
| Aggregate uncoalesced-shared excess | 3.55 B (83%) | **0.38 B (34%)** | −89% |
| **Occupancy** | 6.25% | **6.25%** | 0 (unchanged) |
| Regs / smem / spills | 254 / 222.7 KB / 0 | 254 / 222.7 KB / 0 | — |

Only the D-rowsum changed (smem byte-identical), so the entire improvement is attributable to it.

## 2. Full metrics (launch 1; 3 launches <0.1%)
| Section | Value |
|---------|-------|
| Duration | 27.88 ms base / 21.16 ms wall |
| Compute SOL | 22.5% (all pipelines still under-utilized, est. 85% local) |
| Memory SOL | 33.4% (L1/TEX 33.5%) |
| DRAM | 4.5% (not BW-bound) |
| L1 / L2 hit | 3.3% / 82.8% |
| **Shared-LOAD conflict** | **none flagged** (was V7's top entry) |
| Shared-STORE conflict | 3.0-way, 66.7% of 0.576 B — est. 22.4% |
| Uncoalesced global stores | 8/32 bytes (dK/dV/dQ writeback) — est. 25%; 50% excessive sectors → est. 49.8% |
| Uncoalesced local ld/st | 1/32 bytes — est. 32.4% (0 spills; dynamically-indexed reg arrays / stack) |
| **No-eligible** | **77.4%** (1.00 active warp/scheduler, 0.23 eligible) — issues every 4.4 cyc |
| **Occupancy** | **6.25%** (4 warps/SM) — **smem-limited** (222.7 KB → 1 block/SM); regs (254) allow 2 |
| Verdict | ncu: "latency issues" (both compute & memory <60%) |
| Launch | block 128, grid 2048, 254 regs, 0 spills, 222.7 KB smem |

## 3. Where V8 stands on the ladder + cuDNN
| | V6 | V7 | V8 | cuDNN |
|--|-----|-----|-----|-------|
| Wall bwd | 64.2 | 33.0 | **21.16 ms** | 2.78–2.96 ms |
| TFLOP/s | 51.7 | 99.96 | **155.75** | 1114–1187 |
| Compute SOL | 15.4% | 13.8% | **22.5%** | 59% |
| Memory SOL | 74.9% | 63.0% | **33.4%** | 62.8% |
| Shared-load conflict | 15.1-way | 11.3-way | **none flagged** | ~2.2-way |
| Occupancy | 6.25% | 6.25% | **6.25%** | 15.6% |
| Spills | 0 | 0 | **0** | 1.04 M |

Gap to cuDNN: ~11.7× (V7) → **~7.6× (V8)**. We've now essentially **solved the shared-memory
conflict problem** (cuDNN ~2.2-way; we no longer flag load conflicts at all) while keeping the
0-spill edge. The remaining gap is almost entirely **occupancy + latency-hiding**: cuDNN feeds its
tensor pipe to 59% at 15.6% occupancy via warp specialization + async; we sit at 22.5% / 6.25%.

## 4. V9 priorities (measured)

The bottleneck is now singular and unambiguous. Both compute (22.5%) and memory (33.4%) are low →
**pure latency-bound**, and ncu attributes it directly to occupancy.

1. **OCCUPANCY — the dominant lever, by far (ncu est. 66.6%).** 6.25% = 1 block/SM, smem-limited at
   222.7 KB; registers (254) already allow 2 blocks. Get to **≥2 blocks/SM** by cutting smem below
   ~113 KB/block. The pin is the double-buffered operands, and the cheapest cut is the **`dO`/`O`
   plain double-presence** (`sdO_pl`/`sO_pl`, ~2×16 KB): now that the D-rowsum is a warp reduction,
   re-derive whether it can read the *swizzled* `sdO_sw` directly (or a single small staging) and
   delete the plain copies. This is the whole ballgame for V9 — nothing else is close.
2. **Uncoalesced global writeback (est. 25–50%).** dK/dV/dQ stores use 8/32 bytes/sector; reindex
   `store_acc_global` / the dQ atomic flush for contiguous 128-bit stores.
3. **Shared-STORE conflict (3.0-way, est. 22.4%)** and **uncoalesced "local" (est. 32.4%, 0 spills
   — likely dynamically-indexed `dv[64]`/`dk[64]` reg arrays spilling to stack)** — secondary.
4. FFMA fusion (~3%), L2 compression (~3%) — ignore.

**Recommended V9 = OCCUPANCY.** Kill the `dO`/`O` plain double-presence to fit 2 blocks/SM. This is
now worth ~2× more than any remaining conflict/coalescing work (est. 66.6% vs ≤50%), and it's the
same structural gap that separates us from cuDNN (15.6% vs 6.25%). The dead swizzle-A machinery
should be stripped or left as a documented dead-end; the A-operand conflict is no longer even the
top shared-memory issue, so it drops in priority behind occupancy.
