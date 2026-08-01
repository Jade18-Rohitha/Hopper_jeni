# V24 KV-Blocking Feasibility — one CTA owns 2 KV-tiles (share Q/dO loads)

## (0) VERDICT: **NO-GO** — same no-TMEM register wall that killed every retile.

One-liner: doubling the KV-ownership doubles the persistent dV/dK accumulators
(64 → 128 regs), pushing peak-live to **~221–223 regs vs the 170 ceiling** — a
~51-reg bust that spills and kills the 0-spill edge; the only register-neutral
escape (smem-stash the 2nd accumulator) lands its cost on the *already-binding*
L1/shared-throughput path (66.4% mem SOL), netting ~0 against a modest ~5–10%
upside. On Blackwell (tcgen05 accumulators in TMEM) this fits; on Hopper it does not.

Base = `gqa_backward_v23_kv`, 384 thr (2 consumer WG + 1 TMA-producer WG),
**5.7017 ms, 157 regs, 0 spill, 191,824 B smem, 18.75% occ (1 block/SM), 2.03× off cuDNN.**

---

## (1) REGISTER MODEL — the decider

**Ceiling:** BLOCK(384), 1 block/SM (smem-pinned). Full-occupancy reg budget =
`floor(65536 / 384)` = **170 regs/thread**. V23 sits at 157 (13 to spare).

### V23 base decomposition (157 measured)
| component | regs | note |
|---|---|---|
| persistent `dv[32]` + `dk[32]` | 64 | per consumer WG; column-split D-half (each WG owns 64 of the 128 D-cols), m64n64 fragment = 32 fp32/thread each |
| transient `acc[32]` | 32 | live during S / dP / dQ GEMMs *while* dv/dk persist across the loop |
| overhead | ~61 | descriptors, incremental (g,qc), `cpar/dpar[3]` parities, wtid/tid/wg, LSE+base addr math, causal bounds, pointers |
| **peak-live** | **~157** | matches ptxas |

Note dV/dK GEMMs accumulate **into** `dv`/`dk` directly (they *are* the wgmma
accumulator target) — the transient `acc[32]` only serves S, dP, dQ. It cannot
be eliminated; every wgmma needs a register accumulator fragment.

### KV-blocking minimal-fit variant (both KV-tiles' accumulators register-resident)
Each CTA owns KV-tiles (kv, kv+1). dK/dV output is **2×[Bc,D]** — two *independent*
gradient tiles, no algebraic sharing. With the existing D-half column-split, each
consumer WG must now hold **both** tiles' halves:

| component | regs |
|---|---|
| persistent `dv0[32]`+`dv1[32]`+`dk0[32]`+`dk1[32]` | **128** (doubled) |
| transient `acc[32]` (shared — the 2 tiles run their S/dP/dQ GEMMs sequentially, one transient suffices) | 32 |
| overhead (+2 for 2nd KV base offset + 2nd causal boundary) | ~63 |
| **peak-live** | **~223** |

Absolute floor (assume overhead frozen at 61, transient fully shared): **221 regs.**
Either way **~221–223 > 170 by ~51–53 regs → guaranteed spill.**

This is empirically consistent with the V15 intra-WG ILP NO-GO (a 2-tile scheme
that measured 186–201 regs > 170), so ptxas will land *above* the hand-count, not
below. 221 is a confident bust.

### Is there ANY register-neutral scheme? Three checked:
1. **Share the transient acc** — already assumed shared above; buys nothing. Even
   deleting it entirely (impossible) leaves 128 + 61 = 189 > 170.
2. **Re-cut the column split so accumulators don't double** — impossible. Two
   KV-tiles = 2× distinct output data; distinct data cannot occupy the same
   registers. Keeping each WG at 64 accumulator-regs would require splitting D
   **4 ways** → a 3rd/4th consumer warpgroup. But (a) a 4th WG busts the budget
   worse (`512 thr × regs > 65536`, already documented impossible at V11), and
   (b) M=64 is wgmma-native → no clean 4-way MMA split. Dead.
3. **Smem-stash the 2nd KV-tile's accumulator** (only ONE live at a time) — see (4).
   Fits registers, but moves the cost onto the binding shared path. Net wash.

**This is the identical no-TMEM wall that killed 128×64, 128×128, V15 ILP, and
ping-pong:** Hopper has no tensor memory to hold accumulators off the register
file — every wgmma fragment must live in RMEM. Blackwell's tcgen05.mma writes
accumulators to **TMEM** (512 cols/SM), which is exactly what would absorb the
doubled dV/dK here. Not available on sm_90a.

---

## (2) SMEM BUDGET — fits (but moot)

KV-blocking needs a **2nd resident K and V tile** so both pipelines run from one
Q/dO load. Q/dO buffers (`sQ_sw[3]`, `sdO_sw[3]`) are **shared** → unchanged. The
transient staging (`sP`, `sS`, `sdP`, `sA_t`) is per-(kv,qc) and reused when the
two tiles are processed sequentially → unchanged.

| | bytes |
|---|---|
| V23 total | 191,824 |
| + `sK_sw2[Bc*D]` = 64·128·2 | +16,384 |
| + `sV_sw2[Bc*D]` | +16,384 |
| **KV-block total** | **224,592** |
| Hopper per-block cap (static smem OK, no opt-in for non-extern) | 232,448 |
| **headroom** | **+7,856** |

**Smem FITS**, and the 3-deep (PD=3) shared Q/dO pipeline stays intact alongside.
The V22 D-split freed exactly the room the task predicted. But smem is not the
gate — registers are.

---

## (3) TRAFFIC WIN — what we leave on the table

Config B=8, Hq=12, Hkv=4, G=3, S=4096, D=128, bf16. dKdV grid = (8,4,64) = 2048 CTAs.
Per KV-tile k: nIter = G·(nQTiles−k) = 3·(64−k) Q-tiles. Per Q-tile: Q+dO load =
2·(64·128·2) = 32,768 B; dQ-accum atomic write = 64·128·4 = 32,768 B (fp32).

| HBM stream | V23 | KV-block | Δ |
|---|---|---|---|
| **Q/dO reads** (O(S²)) | Σ nIter·32KB = **6.54 GB** | pairs load union-range once = **3.32 GB** | **−49%** |
| **dQ atomics** (O(S²), write) | 6.54 GB | **6.54 GB** | **0** (# of (kv,qc) pairs unchanged) |
| K/V load + dK/dV writeback | ~0.13 GB | ~0.13 GB | 0 |
| **main-kernel HBM total** | **~13.2 GB** | **~10.0 GB** | **−24%** |

Q/dO reads halve as designed (each of the 32 (b,hkv) groups drops 6240 → 3168
load-iterations). But **dQ atomics are untouched** (KV-blocking cuts the read
side only, as the task notes) and equal Q/dO in size — so total HBM only drops 24%.

### Is 24%-less-HBM worth ~50 registers of spill? Realistic speedup ≈ **5–10%**, NOT 24%.
- Main kernel HBM = 13.2 GB / ~5.6 ms ≈ **2.36 TB/s ≈ 49% of H200 HBM3e peak** — the
  kernel is **not DRAM-saturated**. It is **L1/shared-throughput-bound** (V23 mem
  SOL 66.4% is L1TEX, DRAM was ~10% at V10) with **long_scoreboard #1** (TMA-feed /
  consumer mbarrier-park wait).
- KV-blocking changes **nothing** on the binding constraint: total compute (same
  # of (kv,qc) GEMM pairs), total shared traffic (sP swizzle reads, dS elementwise
  RMW, stage_acc scatters, atomic_flush staging), barriers, and dQ atomics are all
  **identical in aggregate** — only regrouped into half as many CTAs. It relieves
  only the Q/dO TMA-feed volume (−49% load transactions).
- Upside comes solely from shorter long_scoreboard TMA-feed stalls (fewer Q/dO
  loads → producer keeps ahead, consumer parks less). Bounded well below the 24%
  HBM figure because the 66.4% shared-throughput ceiling and the equal-sized
  O(S²) dQ-atomic block are unmoved. **Estimate 5–10%.**

So even if registers fit, this is a modest lever — worth ~0.3–0.6 ms — and it is
being bought with a guaranteed spill. Bad trade.

---

## (4) The register-neutral escape, and why it still fails

**Smem-stash variant:** keep only ONE KV-tile's accumulator register-live; hold the
other's `dv[32]+dk[32]` (64 fp32 = 256 B/thread) resident in smem the whole kernel,
and RMW it each Q-tile (`smem→reg` load, wgmma-accumulate, `reg→smem` store —
wgmma *must* accumulate in registers, so the round-trip is unavoidable).

- **Registers:** persistent drops 128 → 64 → peak ~157–160 → **fits ≤170.** Threads
  the needle.
- **Smem:** +64 fp32/thread × 256 consumer thr per WG-set = +64 KB stash. 224,592 +
  ~64 KB **busts the 232,448 cap** → must also drop the K/V duplication or PD, eroding
  the design.
- **Kills the win:** the RMW is **512 B/thread/Q-tile of extra shared load+store**,
  landing directly on the **66.4%-bound L1/shared path** — the *actual* ceiling. It
  converts an off-chip HBM-read saving (a resource at only 49%) into on-chip
  shared-throughput cost (the resource that's already the wall). Adds ~10–15% shared
  traffic on kv-stashed iterations to buy a ~5–10% HBM-feed relief → **net wash, or
  negative.** Does not thread the needle in any way that matters.

No variant delivers the traffic win without either (a) spilling ≥51 registers or
(b) pushing the binding shared-throughput constraint the wrong way.

---

## Bottom line
**NO-GO. Same no-TMEM register wall as every retile NO-GO.** Smem fits (224,592 <
232,448) and the traffic idea is real (−49% Q/dO reads, −24% HBM), but the 2×
persistent dV/dK accumulators are architecturally forced (2 KV-tiles = 2× distinct
output, no algebraic or column-split escape on 2 WGs) → ~221–223 regs ≫ 170 →
spill → 0-spill edge dies. The register-neutral smem-stash fits regs but negates
the win on the binding L1 path. Realistic upside even if it fit is only ~5–10%
(shared-throughput + O(S²) dQ-atomics are untouched). **Do not build.** The lever
this design wants — accumulators off the register file — is Blackwell TMEM
(tcgen05.mma), not available on Hopper sm_90a. Keep pursuing V23's proven lever:
instruction/traffic work-cuts on the dQ-atomic O(S²) block (address math, staging),
which is the larger untouched stream.
```
```
Numbers checked against live V23 source (GQA_bwd.cu:7609–7804, launcher 7872 BLOCK(384))
and V23_analysis.md. Register ceiling 170 = floor(65536/384). No code written.
```
