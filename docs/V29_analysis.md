# V29 Analysis — exp2f softmax (fold scale·log2e), the forward's exp path

**Result: 4.3990 ms (median) / 749 TFLOP/s — ~1.58× off cuDNN. Bit-identical (max_abs
unchanged). −0.52% over V28 (4.4222 ms).** Broke 4.4 ms and **2.0× total speedup** (8.82 →
4.399). 157 regs, 0 spill. Ported from the cuDNN-beating forward (`GQA_fwd_ref.cu`).

## What changed
`fused_p`'s softmax: `__expf(acc*scale − lse)` → **`ex2.approx.ftz(fmaf(acc, scale2, −lse2))`**,
with `scale2 = scale·log2e` and `lse2 = lse·log2e`. Since `exp(x) = 2^(x·log2e)`, this folds both
the `*scale` and the internal `*log2e` of `__expf` into a **single FFMA**, then one raw SFU
`ex2.approx`. New `fused_p_from_acc_v29` / `fused_p_nomask_v29` + an `ex2_approx_v29` inline-asm
helper. SASS: **FMUL 284 → 95 (−189)**, total static **2720 → 2328 (−392)**.

**Bit-identical, confirmed:** `ex2.approx.ftz` differs from `__expf` by ~1 ulp in fp32, but that
rounds to the *same* bf16 — the `check()` max_abs values (dQ 1.953e-3 / dK 3.906e-3 / dV
3.125e-2) are unchanged from every prior version.

## Profile — instructions down, but now memory-bound

| metric | V28 | **V29** |
|---|---|---|
| Executed instructions | 1.866 B | **1.758 B (−5.8%)** |
| IPC | 1.94 | 1.84 |
| Compute SOL | 48.3% | 45.8% |
| Memory SOL | 59.6% | **60.1%** |
| barrier / wait stall | 1.16 / 0.84 | 1.23 / 0.93 |

Instruction count fell another 5.8% (−11.4% cumulative since V27), but IPC dropped and the
profile now reports **"Memory is more heavily utilized than Compute"** — the kernel crossed
from compute/balanced to **memory-bound**. The instruction-count vein (V28/V29) has largely
paid out; the remaining wall is **scheduling** — occupancy locked at 18.54%, No-Eligible 54%,
the two consumer warpgroups still colliding on their wgmma waits.

## Next — the phased forward-port
Phase 1 (exp2f) ✓. **Phase 2 = V30 setmaxnreg** (producer→40, consumers→232) to lift the 170
register ceiling — neutral on its own, the foundation for **Phase 3 = V31 FA3 ping-pong**, which
attacks the No-Eligible/scheduling wall the way the forward beats cuDNN.

Cumulative: V13 8.82 → V27 4.6752 → V28 4.4927 → **V29 4.3990 (~1.58×, 2.0× total)**.
