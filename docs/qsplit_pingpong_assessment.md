# Q-split ping-pong — the real lever (the forward's structure for the backward)

## The insight
The forward beats cuDNN via FA3 ping-pong: 2 consumer warpgroups each own a **full query-tile**,
streaming the shared kv — so P is produced *and* consumed inside one warpgroup (RS-wgmma, no
cross-wg traffic). Our backward instead **column-splits** (2 wgs cooperate on one q-tile, wg0 owns
P, wg1 owns dP), which forces `dS = P⊙(dP−D)` through a **cross-warpgroup smem exchange** — the dS
loop reading sP+sdP. The agent measured that exchange at **~20% of L1TEX**: pure column-split tax.

**What I earlier missed:** when V31 killed "single-warpgroup-per-tile," I analyzed the **kv-split**
(each wg owns a whole kv-tile → 2× sK/sV → smem blows up). The forward's structure is the
**q-split**: each wg owns a full q-tile, *sharing* the one kv-tile. Totally different feasibility.

## What the q-split changes
- **P and dP for a q-tile are in the SAME warpgroup's registers** → `dS` computed in-register →
  **the cross-wg exchange vanishes, and `sdP` is deleted** (−18 KB smem + the ~20% L1TEX dS-loop
  reads of sP+sdP disappear — only the WRITES of sP/sDS remain, for the transposed dV/dK).
- **RS-wgmma for dQ becomes real**: dQ = dS·K is non-transposed, dS is now in-wg registers → dQ's A
  can be register-source (like the forward's P→pa), dropping dQ's sDS read too.
- dV/dK still need sP/sDS in smem (transposed A, inter-warp — unchanged, that part isn't RS-able).

## Smem budget (the reason it FITS, unlike kv-split)
Shared sK/sV (1×, not 2×):
- sK+sV: 32,768 (shared)
- sQ/sdO pipeline (shared, both wgs consume alternate q-tiles): PD=3 → 98,304
- per-wg sP + sDS (2×(8,192+8,192)): 32,768
- **sdP: 0** (dP register-resident)
- dQ flush stages (2×): 36,864  ← or **TMA-store these to free 36 KB → enables PD=4**
- reduction buffer + sLSE/sD: ~10,000
- **Total ≈ 206 KB at PD=3 (fits < 227 KB)**; PD=4 needs the TMA-store-flush to fit.

## The cost — bounded spilling (what cuDNN does, we've refused)
Each wg now holds **dv[64] + dk[64]** (full D, 128 regs, was [32] column-split) + P/dP/dS in-flight
→ ~256 regs → **spills**. This is exactly cuDNN's 168-reg + ~832K-local profile. Accept bounded
spilling of *cold* state (loop bookkeeping, descriptors) — never the dv/dk accumulators or live acc.
This is the invariant we held all session (0-spill) that has to yield.

## dv/dk cross-wg reduction (cheap)
wg0 accumulates dv/dk over its q-tiles (even), wg1 over odd → one **sum** of the two partials at
the epilogue (O(Bc·D), once per block). Negligible.

## Honest win estimate
Cutting the ~20% dS-exchange L1TEX (+ RS-dQ's sDS read) plausibly lands **~3.3–3.5 ms** (from V33
3.98). The inherent wgmma operand fetches (~80% of L1TEX) remain — so this does **not** reach cuDNN's
2.8 alone; that still needs the *transposed* dV/dK fetches cut, which is the atom-spanning /
Blackwell-TMEM wall. But ~3.3–3.5 closes most of the remaining Hopper gap.

## Build sequence (dedicated effort — NOT a single version)
1. **Skeleton**: 2 wgs each a full q-tile, shared sK/sV, shared q-pipeline; each wg does S→P→dP→dS
   (in-register)→dV-partial→dK-partial→dQ. Column-split DELETED. Validate correctness (may spill).
2. **dv/dk cross-wg reduction** at the epilogue (sum the two partials).
3. **RS-dQ** (dS in-register → pa[16], drop dQ's sDS read).
4. **Spill tuning** (bound to cold state) + **TMA-store flush** to reclaim smem for PD=4.
5. Ping-pong barrier coordination (named barriers, FA3-style) to overlap the two wgs' tensor use.

This is the one path that attacks the column-split tax at its root. Several careful steps; the
premise (dS in-registers) *holds* here because it's q-split, not kv-split — the mistake that made
the RS attempt look dead.
