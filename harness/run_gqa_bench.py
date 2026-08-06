#!/usr/bin/env python3
"""
run_gqa_bench.py -- unified GQA benchmark: your kernel vs the SOTA field.

Self-configures per machine: every competitor self-detects (installed? arch?)
and is skipped with a reason if unavailable. On the RTX-3060 dev box you get
SDPA + Triton + cuDNN + Liger; on the B300 you additionally get FA4 + FlashInfer.

Flow:
  1. shape comes from a parsed .cu (--from) or explicit flags.
  2. for each (seqlen x group-ratio x causal) point:
       build canonical bf16 inputs -> correctness gate every attention backend
       against a fp32/fp64 reference -> time only the ones that pass.
  3. print a table + write JSON/markdown report; speedups are vs your kernel
     (or vs FA4 if yours isn't wired yet).

Examples:
  python3 harness/run_gqa_bench.py --from src/SM103/attention/GQA_sm103_causal_v2.cu
  python3 harness/run_gqa_bench.py --seqlens 4096 --group-ratios 4,8 --causal \
        --competitors fa4,cudnn,sdpa,triton
  python3 harness/run_gqa_bench.py --track fused --group-ratios 4   # QK-norm+RoPE
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))

import bench_core as bc               # noqa: E402
import competitors as comp            # noqa: E402


def shape_from_cu(cu_path: str) -> dict:
    import gqa_parser
    p = Path(cu_path)
    if not p.is_absolute():
        p = ROOT / cu_path
    cfg = gqa_parser.extract_config(p.read_text(errors="ignore"))
    return cfg


def build_shapes(args) -> list[bc.Shape]:
    base = {}
    if args.from_cu:
        base = shape_from_cu(args.from_cu)
        print(f"# shape parsed from {args.from_cu}: {base}")
    B = args.batch or base.get("B", 2)
    Hkv = args.kv_heads or base.get("Hkv", 4)
    D = args.hdim or base.get("D", 64)
    seqlens = [int(x) for x in args.seqlens.split(",")] if args.seqlens \
        else [base.get("S", 4096)]
    ratios = [int(x) for x in args.group_ratios.split(",")] if args.group_ratios \
        else [base.get("G", Hkv and (base.get("Hq", Hkv) // Hkv) or 4)]
    causal = base.get("causal", True) if args.causal is None else args.causal
    if causal == "maybe":
        causal = True
    shapes = []
    for s in seqlens:
        for g in ratios:
            shapes.append(bc.Shape(B=B, Hq=Hkv * g, Hkv=Hkv, S=s, D=D,
                                   causal=bool(causal), dtype=args.dtype))
    return shapes


def run_attention(shape, backends, args) -> list[bc.Result]:
    q, k, v = bc.make_inputs(shape, device=args.device)
    ref = None
    if not args.no_gate:
        ref = bc.reference_output(q, k, v, shape, mode=args.ref)
    results = []
    for cb in backends:
        ok, reason = cb.available(shape)
        if not ok:
            results.append(bc.Result(name=cb.name, ok=False, reason=reason))
            continue
        try:
            if ref is not None:
                out = cb.forward(q, k, v, shape)
                passed, ma, mn, mr = bc.correctness(out, ref, args.atol, args.rtol)
            else:
                passed, ma, mn, mr = None, float("nan"), float("nan"), float("nan")
            rm, allms = bc.time_fn(cb.make(q, k, v, shape), iters=args.iters,
                                   warmup=args.warmup, runs=args.runs,
                                   flush=not args.no_flush, device=args.device)
            r = bc.summarize(cb.name, rm, allms, shape)
            r.correct, r.max_abs, r.mean_abs, r.max_rel = passed, ma, mn, mr
            if passed is False:
                r.reason = "FAILED correctness gate"
            results.append(r)
        except Exception as e:
            results.append(bc.Result(name=cb.name, ok=False,
                                      reason=f"{type(e).__name__}: {e}"))
    return results


def print_table(shape, results, baseline="yours"):
    base = next((r for r in results if r.ok and r.name == baseline), None)
    if base is None:
        base = next((r for r in results if r.ok and r.name == "fa4"), None)
    base_ms = base.median_ms if base else None
    print(f"\n### {shape}")
    hdr = (f"{'backend':<14} {'role':<26} {'median ms':>10} {'TFLOP/s':>9} "
           f"{'GB/s':>8} {'correct':>8} {'vs base':>8}")
    print(hdr)
    print("-" * len(hdr))
    order = sorted(results, key=lambda r: (not r.ok, r.median_ms if r.ok else 9e9))
    for r in order:
        role = comp.ALL[r.name].role if r.name in comp.ALL else ""
        if not r.ok:
            print(f"{r.name:<14} {role:<26} {'--':>10} {'--':>9} {'--':>8} "
                  f"{'skip':>8}   {r.reason}")
            continue
        corr = {True: "pass", False: "FAIL", None: "-"}[r.correct]
        spd = f"{base_ms / r.median_ms:.2f}x" if base_ms else "-"
        print(f"{r.name:<14} {role:<26} {r.median_ms:>10.4f} {r.tflops:>9.1f} "
              f"{r.gbps:>8.1f} {corr:>8} {spd:>8}")


def run_fused(shape, backends, args):
    """QK-norm + RoPE track: time the pre-attention op stack the SOTA kernels
    do NOT fuse. Your GQA_fused kernel folds this into attention -- run it here
    (via 'yours') to show the fusion win vs these separate ops."""
    q, k, v = bc.make_inputs(shape, device=args.device)
    results = []
    for cb in backends:
        if cb.kind != "norm_rope":
            continue
        ok, reason = cb.available(shape)
        if not ok:
            results.append(bc.Result(name=cb.name, ok=False, reason=reason))
            continue
        try:
            rm, allms = bc.time_fn(cb.make(q, k, v, shape), iters=args.iters,
                                   warmup=args.warmup, runs=args.runs,
                                   flush=not args.no_flush, device=args.device)
            # this track's "flops" is dominated by attention downstream; report ms only
            r = bc.summarize(cb.name, rm, allms, shape)
            r.tflops = float("nan"); r.gbps = float("nan")
            results.append(r)
        except Exception as e:
            results.append(bc.Result(name=cb.name, ok=False,
                                      reason=f"{type(e).__name__}: {e}"))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="from_cu", help="parse shape from this .cu file")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--kv-heads", type=int, dest="kv_heads")
    ap.add_argument("--hdim", type=int)
    ap.add_argument("--seqlens", help="comma list, e.g. 2048,4096,8192,16384")
    ap.add_argument("--group-ratios", dest="group_ratios",
                    help="GQA G=Hq/Hkv sweep, e.g. 2,4,8,16")
    ap.add_argument("--causal", dest="causal", action="store_true", default=None)
    ap.add_argument("--no-causal", dest="causal", action="store_false")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--track", default="attention",
                    choices=["attention", "fused", "both"])
    ap.add_argument("--competitors", default="fa4,cudnn,sdpa,flashinfer,triton",
                    help="comma list; unavailable ones are skipped with a reason")
    ap.add_argument("--baseline", default="yours",
                    help="speedups are relative to this backend (falls back to fa4)")
    ap.add_argument("--your-so", dest="your_so", help="path to your compiled kernel .so")
    ap.add_argument("--ref", default="fp32", choices=["fp32", "fp64"],
                    help="correctness reference precision")
    ap.add_argument("--no-gate", action="store_true", help="skip correctness (timing only)")
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no-flush", action="store_true", help="disable L2 flush")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", help="write results JSON here")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr); sys.exit(1)
    cc = torch.cuda.get_device_capability(0)
    print(f"# device: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")

    names = [n.strip() for n in args.competitors.split(",") if n.strip()]
    if args.track in ("fused", "both"):
        names += [n for n in ("liger", "unsloth") if n not in names]
    backends = comp.get(names)
    if args.your_so:
        yk = comp.YourKernel(so_path=args.your_so)
        backends.insert(0, yk)
        comp.ALL["yours"] = yk

    shapes = build_shapes(args)
    all_out = []
    for shape in shapes:
        if args.track in ("attention", "both"):
            res = run_attention(shape, [b for b in backends
                                        if b.kind == "attention"], args)
            print_table(shape, res, baseline=args.baseline)
            all_out.append({"shape": str(shape), "track": "attention",
                            "results": [r.__dict__ for r in res]})
        if args.track in ("fused", "both"):
            res = run_fused(shape, backends, args)
            print("\n### fused track (QK-norm + RoPE, ms only):")
            for r in sorted(res, key=lambda r: (not r.ok, r.median_ms if r.ok else 9e9)):
                if r.ok:
                    print(f"  {r.name:<12} {r.median_ms:>9.4f} ms")
                else:
                    print(f"  {r.name:<12} skip  {r.reason}")
            all_out.append({"shape": str(shape), "track": "fused",
                            "results": [r.__dict__ for r in res]})

    if args.json:
        Path(args.json).write_text(json.dumps(all_out, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
