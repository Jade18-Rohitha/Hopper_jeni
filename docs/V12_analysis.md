# V12 Analysis — ldmatrix/stmatrix shared-traffic reduction (Nsight Compute, D=128)

**V12 = V11 + `ldmatrix.x4 → stmatrix.x4` register-passthrough** replacing the manual scalar
`fill_trans_A`/`fill_copy` scatters that built the transposed A-operands (Pᵀ/dSᵀ/dS). Correct
(bit-identical to V1–V11, validated full-speed on H200), **11.16 ms / 295.57 TFLOP-s**, 0 spills,
168 regs. A **~17% win over V11 (13.43 ms)** — the biggest single-version gain since V9, and it
landed **correct on the first H200 run** (the per-lane ldmatrix layout was the #1 risk).

- **Kernel:** `gqa_backward_v12_kv<64,64,128>` — grid `(8,4,64)=2048`, block 384 (3 warpgroups).
- **Source:** `reports/gqa_v12_profile_d128.{ncu-rep,txt}` (`ncu --set full`, 3 launches, <0.1% spread). This profile at full clock (DRAM 3.20 GHz) vs V11's throttled 2.62 GHz — so **compare %s and instruction/wavefront counts, not base-ms**. Wall from benchmark: 11.16 ms.
- Compare: `V11_analysis.md`, `gqa_bwd_py_analysis.md` (cuDNN 2.78–2.96 ms).

---

## 0. Headline — the win came from *doing less work*, not higher utilization

> The counterintuitive part: V12's throughput %s went **down** (Compute SOL 40%→34%, IPC 1.60→1.37)
> yet it's **17% faster**. Why: `ldmatrix`/`stmatrix` do the transpose reshuffle in **far fewer
> instructions** than the scalar scatter — **total executed instructions dropped 4.97 B → 3.50 B
> (−30%)** — and cut shared traffic hard. The old higher IPC was partly *overhead* (address-gen +
> strided scatter). Removing overhead lowers the % but speeds up the kernel. **Fewer, cheaper
> instructions beat a higher utilization of wasteful ones.**

Shared-memory traffic collapsed as intended:
- **Shared-store requests 218 M → 141 M (−35%)**, shared-store conflicts **551 M → 335 M (−39%)**.
- **Total shared wavefronts 1.40 B → 1.09 B (−22%)**, uncoalesced-shared excessive **540 M (39%) →
  310 M (28%)**.

---

## 1. V11 → V12 delta
| Metric | V11 | V12 | Δ |
|--------|-----|-----|---|
| **Wall-clock** | 13.43 ms | **11.16 ms** | **−17%** |
| TFLOP/s | 245.7 | **295.6** | +20% |
| **Total executed instructions** | 4.97 B | **3.50 B** | **−30%** (ldmatrix/stmatrix vs scalar scatter) |
| Shared-store requests | 218 M | **141 M** | −35% |
| Shared-store conflicts | 551 M | **335 M** | −39% |
| Total shared wavefronts | 1.40 B | **1.09 B** | −22% |
| Uncoalesced-shared excessive | 540 M (39%) | **310 M (28%)** | est 38.6% → 28.4% |
| Compute SOL | 40.0% | 34.0% | −6 (removed overhead, see §0) |
| IPC | 1.60 | 1.37 | −0.23 (ditto) |
| No-eligible | 59.9% | 65.9% | +6 |
| Warp cycles / issued | 7.42 | 8.71 | +1.3 |
| Regs / spills | 168 / 0 | 168 / 0 | flat (Option B: transient ldmatrix regs) |
| smem | 222,760 B | 223,528 B | +768 (sP stride 66→72, ldmatrix 16-B align) |
| Occupancy | 18.75% | 18.75% | unchanged |

**Option B was the right pick** — `ldmatrix.trans`→regs→`stmatrix` keeps operands in shared for
SS-wgmma, so register pressure stayed flat (168, under the 170 ceiling) and 0 spills held. Option A
(RS-wgmma, A in registers) would have blown 170.

## 2. Full metrics (launch 1; 3 launches <0.1%)
| Section | Value |
|---------|-------|
| Duration | 14.39 ms base (SM 1.35 GHz, full clock) / 11.16 ms wall |
| Compute SOL | 34.0% (ALU top pipe 20.9% — down, less scatter address-gen) |
| Memory SOL | 50.2% (L1/TEX 50.3%) |
| DRAM | 8.6% (not BW-bound) |
| **Dominant stall** | **long-scoreboard / L1TEX — 2.9 of 8.7 cyc, 33.0%** (unchanged as #1) |
| Shared-STORE conflict | 4.1-way, 58.2% of 0.58 B (335 M) — est. 29.3% (way-ness up, but on −35% requests) |
| Uncoalesced shared (aggregate) | 310 M excessive / 1.09 B (28%) — est. 28.4% |
| Uncoalesced local | 1/32 bytes — est. 48.6% (**red herring**: 0.1% of instr, 99% L1 hit — diagnosed V11, ignore) |
| No-eligible | 65.9% (2.97 active warp/sched, 0.43 eligible) — issues every 2.9 cyc |
| **Occupancy** | **18.75%** (achieved 18.59%) — MAXED (Block Limit Reg = 1 AND Smem = 1) |
| Launch | block 384, grid 2048, 168 regs, 0 spills, 223.5 KB smem |

## 3. Ladder + cuDNN
| | V10 | V11 | V12 | cuDNN |
|--|-----|-----|-----|-------|
| Wall bwd | 14.62 | 13.43 | **11.16 ms** | 2.78–2.96 ms |
| TFLOP/s | 225 | 246 | **295.6** | 1114–1187 |
| Compute SOL | 38% | 40% | 34%¹ | 59% |
| Occupancy | 12.5% | 18.75% | **18.75%** | 15.6% |
| Spills | 0 | 0 | **0** | 1.04 M |

¹ V12's lower Compute SOL is *less overhead work*, not more starvation — compare wall-clock/TFLOP-s.

Gap to cuDNN: ~4.66× (V11) → **~4.0× (V12)** (11.16 / 2.81). Journey: V3 ~35× → **V12 4.0×** (~8.8×
total speedup over the naive wgmma baseline, all bit-identical + 0-spill).

## 4. V13 priorities — the L1TEX-scoreboard stall is now the whole game

V12 removed *traffic and instructions*; what remains is **latency**. The dominant stall is unchanged:
**long-scoreboard / L1TEX, 33% of the 8.7 warp-cycles** — i.e. the tensor cores stalling while
warps wait on shared/global during the **elementwise (P, dS, D-rowsum) and staging phases**, which
are still serialized behind `__syncthreads()`. Occupancy is maxed and can't hide it (more warps just
contend — that's why no-eligible even rose).

**This is exactly what V13 (softmax/MMA overlap) targets** — and V12 set it up perfectly: with 30%
fewer instructions and 22% less shared traffic, there's *less latency to hide*, so the overlap has a
cleaner base. The plan holds:

1. **V13 = softmax↔MMA overlap** (deep warp specialization). Keep the tensor cores busy *through* the
   P/dS/staging phases instead of idling at the barriers. Directly attacks the 33% L1TEX-scoreboard
   stall and the Compute-SOL gap (34% → toward cuDNN's 59%). Highest-complexity version; constrained
   by the KV-centric persistent `dv/dk` accumulator (limits cross-tile GEMM overlap) and the 170-reg
   ceiling. Design-first via the cuda-agent.
2. Residual shared-store 4.1-way (est. 29.3%) — smaller now (−35% requests); a later padding/layout
   pass if it still shows after V13.
3. FFMA fusion (~4%), L2 compression (~6%) — ignore.

> **Framing unchanged from V11, now sharper:** occupancy and conflicts are done; the remaining ~4×
> to cuDNN is **latency hiding** — keeping the pipe fed during the non-GEMM phases. V13 is the lever.
