# V14 (new) — Cross-tile S/dP pipeline: design prep

Goal: raise **Compute SOL 36% → toward cuDNN's 59%** by keeping the tensor cores busy *through*
tile N's elementwise/staging phases, overlapping them with tile N+1's independent GEMMs. This is
the last lever with real headroom (occupancy, load-conflicts, writeback, softmax-serial all done;
V13 is 8.82 ms / 3.14× off cuDNN).

## The enabling insight — wgmma is async
We do NOT need a spare warpgroup. `wgmma.mma_async` runs in the background. So the pipeline is:
**issue tile N+1's independent GEMM(s) early (async), do tile N's elementwise while they run,
`wait_group` when we reach N+1.** The tensor cores stay fed during the elementwise gaps that
currently show as the L1TEX-scoreboard stall.

The independent work: **S(N+1)=Q(N+1)·Kᵀ** and **dP(N+1)=dO(N+1)·Vᵀ** depend only on tile N+1's
loaded Q/dO (+ persistent K/V) — NOT on tile N's P/dS/dV/dK/dQ. So they can be in flight during
tile N's tail.

## Three hard tensions (why this is a big version, not a tweak)

1. **Register-fused softmax (V13) conflicts with pipelining S.** V13's win was keeping S in wg0's
   `acc[32]` registers and fusing `P=exp` there — no `sS` store. But to compute S(N+1) while tile N
   is still live, S(N+1) must go somewhere that doesn't clobber tile N's registers → back to
   **smem (double-buffered `sS`)**, i.e. *un-fusing* the softmax for the pipelined path. That
   partially gives back V13's gain. **dP does NOT have this problem** — it already lands in `sdP`
   (smem), so dP(N+1) can be double-buffered + overlapped with no register conflict.

2. **Smem budget is tight.** V13 = 223,800 B of ~227 KB cap (~3 KB headroom). Double-buffering
   costs: `sdP` +16.6 KB, `sS` +16.6 KB. To afford it we must FREE space:
   - The **plain `sdO_pl[2]`+`sO_pl[2]` = 65,536 B** exist only for the D-rowsum. Compute the
     rowsum from the **swizzled** buffers instead (deswizzle-read via the `sw128_idx` formula on
     the producer's idle threads): drop `sdO_pl` (dO already in `sdO_sw`) and `sO_pl`, but ADD
     `sO_sw[2]` (+32.8 KB, O now loaded swizzled). **Net free ≈ 32 KB.**
   - 32 KB freed covers **double-buffering dP alone** (+16.6 KB) comfortably; double-buffering
     *both* S and dP (+33 KB) is right at the edge — feasible only if we also shrink something
     (e.g. drop `sS` padding, or bf16 an intermediate).

3. **KV-centric persistent `dv`/`dk` accumulator** blocks overlapping consecutive tiles' dV/dK
   GEMMs (same registers) — unchanged constraint. Only S/dP (which write smem, not the persistent
   accumulators) are pipelineable.

## Recommended phasing (de-risked)

**Phase 1 — pipeline dP only (tractable, keeps V13's fused softmax).**
- Free the plain dO/O buffers via deswizzled producer rowsum → ~32 KB.
- Double-buffer `sdP` (+16.6 KB).
- Issue `dP(N+1)` async right after tile N+1's TMA completes; consume at tile N+1. Overlaps one
  GEMM with tile N's elementwise+dV+dK+dQ.
- Register-fused softmax (S) stays exactly as V13. Lower risk, partial compute-SOL gain.

**Phase 2 (only if Phase 1 pays) — also pipeline S.**
- Un-fuse the softmax for the pipelined path: S(N+1) → double-buffered `sS`. Needs the last ~16 KB
  (shrink `sS` stride/precision to fit). Bigger compute-SOL gain, but gives back some of V13's
  fusion win and is tighter on smem — do only if Phase 1 shows the overlap is worth it.

Estimated ceiling: this is the only path that moves Compute SOL materially. Realistic target is
closing 3.14× toward ~2–2.5×, not parity (some cuDNN edge is hand-tuned SASS).

## Risks
- Async-group bookkeeping across iterations (commit/wait_group ordering for the in-flight N+1 GEMM
  vs the current-tile GEMMs) + the double-buffer fences. Deadlock/ordering bugs hide at full speed.
- Register pressure: currently 154/170. The pipeline adds live state (N+1 descriptors, extra
  accumulator). Watch the ceiling; `setmaxnreg` if needed.
- The deswizzled rowsum adds producer ALU (fine — producer has idle threads) but must reproduce the
  exact per-row reduction bit-identically.
- Bit-identical is the bar (2e-2). Phase 1 keeps fp32 throughout → should stay exact.

## On `tma_3d_load` (the open question)
Loading a D=128 operand tile as one `cp.async.bulk.tensor.3d` (dims: 2 atoms × rows × 64) instead
of the current **two 2D calls per operand** halves the producer's TMA-issue instruction count.
- **Standalone: not worth it** — we're not TMA/DRAM-bound (11% DRAM) and the producer isn't the
  bottleneck today.
- **In the pipeline: worth an experiment IF the producer becomes the bottleneck.** With deeper
  buffering the producer issues more TMAs and runs further ahead; fewer/wider issues help it keep
  up. Fold it in *there*, measured — not as a standalone version.
- **Risk to validate first:** whether 128B-swizzle composes correctly with a 3D tensor map feeding
  the wgmma 2-atom descriptor. `cuTensorMapEncodeTiled` supports 3D + swizzle, but the swizzle
  applies to the inner 128 B — needs a small ramp-probe (like the sw128 probe) to confirm the 3D
  layout matches what `make_desc_sw128_*` expects before trusting it in the kernel.

## Next step
Implement **Phase 1** via the cuda-agent (design is concrete above). If Phase 1 lands and profiles
show the producer keeping up, Phase 2 + the tma_3d producer experiment follow. If Phase 1's gain is
marginal, that's the signal we've hit the practical ceiling and V13 stands.
