# GQA benchmark & analysis system

Two tools that connect: a **parser** that reads your `.cu` kernels and a
**harness** that benchmarks them against the 2026 SOTA attention field. The
parser's extracted shape auto-configures the harness so every competitor runs
at *your* exact problem size.

```
tools/gqa_parser.py      static analysis: problem shape + capability vector + gap report
tools/competitors_db.json  SOTA capability DB (FA4/cuDNN/SDPA/FlashInfer/Triton/TK/Liger)
harness/bench_core.py    CUDA-events timing, L2 flush, fp32/fp64 reference, correctness gate
harness/competitors.py   one self-detecting plugin per backend
harness/run_gqa_bench.py unified entrypoint + JSON/table report
```

## 1. Parse your kernels (runs anywhere, no GPU needed)

```bash
python3 tools/gqa_parser.py --target fa4          # capability + ranked gap report
python3 tools/gqa_parser.py --md tools/gqa_capability_matrix.md   # matrix table
python3 tools/gqa_parser.py --emit-configs configs/               # per-file bench shapes
```

The gap report ranks the SOTA techniques each kernel is *missing* vs a target
(default FA4) by expected payoff. For your V34 lineage the top gap is
`sw_exp2_emulation` (FA4's cubic-poly exp2 on FMA units) — the single highest
lever, and not even in cuDNN.

## 2. Benchmark against the field

Self-configures per machine — unavailable backends skip with a reason.

```bash
# match V34's shape automatically, sweep the GQA ratio (the axis that stresses KV-reuse)
python3 harness/run_gqa_bench.py \
    --from src/SM103/attention/GQA_sm103_causal_v2.cu \
    --seqlens 2048,4096,8192,16384 --group-ratios 2,4,8,16 --causal \
    --competitors fa4,cudnn,sdpa,flashinfer,triton --baseline yours \
    --your-so build/libgqa_v34.so --json report.json

# QK-norm + RoPE fusion track (your unique win: no SOTA kernel fuses this)
python3 harness/run_gqa_bench.py --from ... --track fused --group-ratios 4
```

### On this dev box (RTX 3060, sm_86)
Available: **sdpa, cudnn, triton, liger**. FA4/FlashInfer skip (need Blackwell).
Use it to validate the harness and iterate on the Triton/SDPA floors.

### On the B300 (sm_103)
Run `base/setup_fa4_b300.sh` first, then the full field lights up: **fa4, cudnn,
flashinfer** join. This is where the real numbers come from.

## Measurement discipline (from `kernel_guidelines.md`)
- CUDA events, measured at the API boundary (dispatch overhead included).
- L2 flushed before every timed rep (256 MB memset); warmup not flushed.
- 25 warmup / 100 iters / 3 runs; **median of per-run medians** is primary.
- **Correctness gate first**: every backend checked vs an fp32 (or `--ref fp64`)
  reference at bf16 tol 2e-2. A kernel that fails is reported `FAIL`, never
  given a speedup.

## Wiring your kernel (`--your-so`)
Compile the V34 kernel (`GQA_sm103_causal_v2.cu`) to a shared lib exposing an
entrypoint that takes canonical `[B,Hq,S,D]` bf16 and returns the same, then
bind it in `competitors.py::YourKernel.forward` (ctypes or
`torch.utils.cpp_extension.load`). Until then, speedups fall back to vs-FA4.

## Adding a competitor
Subclass `Competitor` in `competitors.py`, implement `available()` +
`forward()`, add an instance to `ATTENTION` (or `NORM_ROPE`). Layout conversion
from canonical `[B,Hq,S,D]` happens inside the plugin.
