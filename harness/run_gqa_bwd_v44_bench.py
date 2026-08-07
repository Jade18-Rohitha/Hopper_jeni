#!/usr/bin/env python3
"""
run_gqa_bwd_v44_bench.py — backward-pass GQA benchmark: V44 vs the SOTA field.

Companion to GQA_bwd_v44.cu's built-in C++ benchmark.  This script times every
PyTorch-accessible backward implementation so you can directly compare V44's
standalone numbers against the field.

Flow:
  1. build canonical bf16 inputs (Q,K,V,dO) with GQA shape
  2. for each competitor: run forward+backward, gate correctness (dQ/dK/dV vs fp32 ref)
  3. time only the backward pass (forward recomputes are included — that's what
     the real backward costs)
  4. print table with TFLOP/s, speedups vs baseline

Usage on H200:
  python harness/run_gqa_bwd_v44_bench.py
  python harness/run_gqa_bwd_v44_bench.py --seqlens 2048,4096,8192
  python harness/run_gqa_bwd_v44_bench.py --batch 4 --competitors sdpa,cudnn,fa4
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))


# ── shapes + inputs ──────────────────────────────────────────────────────────

@dataclass
class Shape:
    B: int
    Hq: int
    Hkv: int
    S: int
    D: int
    causal: bool = True
    dtype: str = "bf16"

    @property
    def G(self) -> int:
        return self.Hq // self.Hkv

    def torch_dtype(self):
        return {"bf16": torch.bfloat16, "fp16": torch.float16,
                "fp32": torch.float32}[self.dtype]

    def bwd_flops(self) -> float:
        """Backward FLOPs: ~4x forward (dQ+dK+dV+P-recompute)."""
        fwd = 4.0 * self.B * self.Hq * self.S * self.S * self.D
        if self.causal:
            fwd *= 0.5
        return 4.0 * fwd      # canonical backward estimate

    def bytes(self) -> float:
        el = 2 if self.dtype in ("bf16", "fp16") else 4
        # read: Q,K,V,O,dO; write: dQ,dK,dV; plus LSE
        qo = 5 * self.B * self.Hq * self.S * self.D * el
        kv = 2 * self.B * self.Hkv * self.S * self.D * el
        dq = self.B * self.Hq * self.S * self.D * el
        dkv = 2 * self.B * self.Hkv * self.S * self.D * el
        return qo + kv + dq + dkv

    def __str__(self):
        return (f"B{self.B} Hq{self.Hq} Hkv{self.Hkv}(G{self.G}) "
                f"S{self.S} D{self.D} {self.dtype} causal={self.causal}")


@dataclass
class Result:
    name: str
    ok: bool
    reason: str = ""
    median_ms: float = float("nan")
    mean_ms: float = float("nan")
    std_ms: float = float("nan")
    tflops: float = float("nan")
    max_abs_dq: float = float("nan")
    max_abs_dk: float = float("nan")
    max_abs_dv: float = float("nan")
    correct: bool | None = None


def make_inputs(shape: Shape, seed=42, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    dt = shape.torch_dtype()
    q = torch.rand(shape.B, shape.Hq,  shape.S, shape.D,
                   generator=g, device=device, dtype=torch.float32).to(dt)
    k = torch.rand(shape.B, shape.Hkv, shape.S, shape.D,
                   generator=g, device=device, dtype=torch.float32).to(dt)
    v = torch.rand(shape.B, shape.Hkv, shape.S, shape.D,
                   generator=g, device=device, dtype=torch.float32).to(dt)
    do = torch.rand(shape.B, shape.Hq, shape.S, shape.D,
                    generator=g, device=device, dtype=torch.float32).to(dt)
    return q, k, v, do


def reference_backward(q, k, v, do, shape: Shape):
    """fp32 reference backward via SDPA autograd."""
    scale = 1.0 / math.sqrt(shape.D)
    q32 = q.float().detach().requires_grad_(True)
    k32 = k.float().detach().requires_grad_(True)
    v32 = v.float().detach().requires_grad_(True)
    o = F.scaled_dot_product_attention(
        q32, k32, v32, scale=scale,
        is_causal=shape.causal, enable_gqa=(shape.G > 1))
    o.backward(do.float())
    return q32.grad, k32.grad, v32.grad


def correctness(dq, dk, dv, dq_ref, dk_ref, dv_ref, atol=2e-2, rtol=2e-2):
    """Check dQ, dK, dV against reference."""
    def _check(got, ref):
        diff = (got.float() - ref.float()).abs()
        ma = diff.max().item()
        return ma, bool(ma <= atol + rtol * ref.float().abs().max().item())

    ma_q, ok_q = _check(dq, dq_ref)
    ma_k, ok_k = _check(dk, dk_ref)
    ma_v, ok_v = _check(dv, dv_ref)
    return ok_q and ok_k and ok_v, ma_q, ma_k, ma_v


# ── L2 flush + timing ────────────────────────────────────────────────────────

class L2Flusher:
    def __init__(self, nbytes=256*1024*1024, device="cuda"):
        self.buf = torch.empty(nbytes, dtype=torch.int8, device=device)
    def flush(self):
        self.buf.zero_()


def time_fn(fn, iters=100, warmup=25, runs=3, flush=True, device="cuda"):
    flusher = L2Flusher(device=device) if flush else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    run_medians, all_ms = [], []
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    for _ in range(runs):
        reps = []
        for _ in range(iters):
            if flush:
                flusher.flush()
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            reps.append(start.elapsed_time(end))
        run_medians.append(statistics.median(reps))
        all_ms += reps
    return run_medians, all_ms


# ── backward competitors ─────────────────────────────────────────────────────

def _has(mod):
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _cc():
    return torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)


class BwdCompetitor:
    name = "base"
    role = ""

    def available(self, shape):
        return True, ""

    def backward(self, q, k, v, do, shape):
        """Returns (dq, dk, dv)."""
        raise NotImplementedError

    def make_bwd(self, q, k, v, do, shape):
        """Returns a zero-arg callable that runs the backward."""
        return lambda: self.backward(q, k, v, do, shape)


class SDPABwd(BwdCompetitor):
    name = "sdpa_bwd"
    role = "stock floor (autograd)"

    def backward(self, q, k, v, do, shape):
        scale = 1.0 / math.sqrt(shape.D)
        qa = q.detach().requires_grad_(True)
        ka = k.detach().requires_grad_(True)
        va = v.detach().requires_grad_(True)
        o = F.scaled_dot_product_attention(
            qa, ka, va, scale=scale,
            is_causal=shape.causal, enable_gqa=(shape.G > 1))
        o.backward(do)
        return qa.grad, ka.grad, va.grad

    def make_bwd(self, q, k, v, do, shape):
        scale = 1.0 / math.sqrt(shape.D)
        def _run():
            qa = q.detach().requires_grad_(True)
            ka = k.detach().requires_grad_(True)
            va = v.detach().requires_grad_(True)
            o = F.scaled_dot_product_attention(
                qa, ka, va, scale=scale,
                is_causal=shape.causal, enable_gqa=(shape.G > 1))
            o.backward(do)
        return _run


class CuDNNBwd(BwdCompetitor):
    name = "cudnn_bwd"
    role = "production peak (autograd)"

    def available(self, shape):
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend  # noqa
        except Exception as e:
            return False, f"torch.nn.attention unavailable: {e}"
        return True, ""

    def backward(self, q, k, v, do, shape):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        scale = 1.0 / math.sqrt(shape.D)
        qa = q.detach().requires_grad_(True)
        ka = k.detach().requires_grad_(True)
        va = v.detach().requires_grad_(True)
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            o = F.scaled_dot_product_attention(
                qa, ka, va, scale=scale,
                is_causal=shape.causal, enable_gqa=(shape.G > 1))
        o.backward(do)
        return qa.grad, ka.grad, va.grad

    def make_bwd(self, q, k, v, do, shape):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        scale = 1.0 / math.sqrt(shape.D)
        def _run():
            qa = q.detach().requires_grad_(True)
            ka = k.detach().requires_grad_(True)
            va = v.detach().requires_grad_(True)
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                o = F.scaled_dot_product_attention(
                    qa, ka, va, scale=scale,
                    is_causal=shape.causal, enable_gqa=(shape.G > 1))
            o.backward(do)
        return _run


class FA4Bwd(BwdCompetitor):
    name = "fa4_bwd"
    role = "PEAK TARGET (bwd)"

    def _fn(self):
        import importlib
        for m in ("flash_attn.cute.interface", "flash_attn.cute",
                  "flash_attn_interface", "flash_attn"):
            if _has(m):
                mod = importlib.import_module(m)
                if hasattr(mod, "flash_attn_func"):
                    return getattr(mod, "flash_attn_func")
        return None

    def available(self, shape):
        cc = _cc()
        if cc[0] < 9:
            return False, f"FA4 targets Hopper/Blackwell; this is sm_{cc[0]}{cc[1]}"
        if self._fn() is None:
            return False, "flash_attn_func not importable (run setup_fa4.sh)"
        return True, ""

    def backward(self, q, k, v, do, shape):
        fn = self._fn()
        # FA4 wants [B,S,H,D]
        qa = q.transpose(1,2).contiguous().detach().requires_grad_(True)
        ka = k.transpose(1,2).contiguous().detach().requires_grad_(True)
        va = v.transpose(1,2).contiguous().detach().requires_grad_(True)
        do_t = do.transpose(1,2).contiguous()
        out = fn(qa, ka, va, causal=shape.causal)
        if isinstance(out, tuple):
            out = out[0]
        out.backward(do_t)
        # back to [B,Hq,S,D] / [B,Hkv,S,D]
        return qa.grad.transpose(1,2), ka.grad.transpose(1,2), va.grad.transpose(1,2)

    def make_bwd(self, q, k, v, do, shape):
        fn = self._fn()
        q_t  = q.transpose(1,2).contiguous()
        k_t  = k.transpose(1,2).contiguous()
        v_t  = v.transpose(1,2).contiguous()
        do_t = do.transpose(1,2).contiguous()
        def _run():
            qa = q_t.detach().requires_grad_(True)
            ka = k_t.detach().requires_grad_(True)
            va = v_t.detach().requires_grad_(True)
            out = fn(qa, ka, va, causal=shape.causal)
            if isinstance(out, tuple):
                out = out[0]
            out.backward(do_t)
        return _run


class FlashInferBwd(BwdCompetitor):
    name = "flashinfer_bwd"
    role = "serving kernel (bwd)"

    def available(self, shape):
        if not _has("flashinfer"):
            return False, "flashinfer not installed"
        cc = _cc()
        if cc[0] < 9:
            return False, f"flashinfer targets sm_90+; this is sm_{cc[0]}{cc[1]}"
        return True, ""

    def backward(self, q, k, v, do, shape):
        # flashinfer backward: use PyTorch autograd through single_prefill_with_kv_cache
        import flashinfer
        outs_dq, outs_dk, outs_dv = [], [], []
        for b in range(shape.B):
            qa = q[b].transpose(0,1).contiguous().detach().requires_grad_(True)
            ka = k[b].transpose(0,1).contiguous().detach().requires_grad_(True)
            va = v[b].transpose(0,1).contiguous().detach().requires_grad_(True)
            do_b = do[b].transpose(0,1).contiguous()
            ob = flashinfer.single_prefill_with_kv_cache(qa, ka, va, causal=shape.causal)
            ob.backward(do_b)
            outs_dq.append(qa.grad.transpose(0,1))
            outs_dk.append(ka.grad.transpose(0,1))
            outs_dv.append(va.grad.transpose(0,1))
        return torch.stack(outs_dq), torch.stack(outs_dk), torch.stack(outs_dv)

    def make_bwd(self, q, k, v, do, shape):
        return lambda: self.backward(q, k, v, do, shape)


BWD_ALL = {
    "sdpa_bwd": SDPABwd(),
    "cudnn_bwd": CuDNNBwd(),
    "fa4_bwd": FA4Bwd(),
    "flashinfer_bwd": FlashInferBwd(),
    # thunderkittens removed: mha_h100 is a forward-only MHA kernel; its mha_backward
    # produced invalid GQA grads here (nan dq/dk) — not a valid backward competitor.
}


# ── main ──────────────────────────────────────────────────────────────────────

def run_backward(shape, competitors, args):
    q, k, v, do = make_inputs(shape, device=args.device)
    ref_dq, ref_dk, ref_dv = None, None, None
    if not args.no_gate:
        ref_dq, ref_dk, ref_dv = reference_backward(q, k, v, do, shape)

    results = []
    for cb in competitors:
        ok, reason = cb.available(shape)
        if not ok:
            results.append(Result(name=cb.name, ok=False, reason=reason))
            continue
        try:
            # correctness
            if ref_dq is not None:
                dq, dk, dv = cb.backward(q, k, v, do, shape)
                passed, ma_q, ma_k, ma_v = correctness(
                    dq, dk, dv, ref_dq, ref_dk, ref_dv, args.atol, args.rtol)
            else:
                passed, ma_q, ma_k, ma_v = None, float("nan"), float("nan"), float("nan")

            # timing
            fn = cb.make_bwd(q, k, v, do, shape)
            run_medians, all_ms = time_fn(fn, iters=args.iters,
                                          warmup=args.warmup, runs=args.runs,
                                          flush=not args.no_flush, device=args.device)
            med = statistics.median(run_medians)
            r = Result(name=cb.name, ok=True)
            r.median_ms = med
            r.mean_ms = statistics.fmean(all_ms)
            r.std_ms = statistics.pstdev(all_ms) if len(all_ms) > 1 else 0.0
            r.tflops = shape.bwd_flops() / (med * 1e-3) / 1e12
            r.correct = passed
            r.max_abs_dq = ma_q
            r.max_abs_dk = ma_k
            r.max_abs_dv = ma_v
            if passed is False:
                r.reason = "FAILED correctness gate"
            results.append(r)
        except Exception as e:
            results.append(Result(name=cb.name, ok=False,
                                  reason=f"{type(e).__name__}: {e}"))
    return results


def print_table(shape, results, v44_ms=None):
    print(f"\n### BACKWARD  {shape}")
    hdr = (f"{'backend':<16} {'role':<26} {'median ms':>10} {'TFLOP/s':>9} "
           f"{'correct':>8} {'vs V44':>8}")
    print(hdr)
    print("-" * len(hdr))
    order = sorted(results, key=lambda r: (not r.ok, r.median_ms if r.ok else 9e9))
    for r in order:
        role = BWD_ALL[r.name].role if r.name in BWD_ALL else ""
        if not r.ok:
            print(f"{r.name:<16} {role:<26} {'--':>10} {'--':>9} "
                  f"{'skip':>8}   {r.reason}")
            continue
        corr = {True: "pass", False: "FAIL", None: "-"}[r.correct]
        spd = f"{v44_ms / r.median_ms:.2f}x" if v44_ms else "-"
        print(f"{r.name:<16} {role:<26} {r.median_ms:>10.4f} {r.tflops:>9.1f} "
              f"{corr:>8} {spd:>8}")
        if r.correct is False:
            print(f"    -> max_abs  dq={r.max_abs_dq:.3e}  dk={r.max_abs_dk:.3e}  dv={r.max_abs_dv:.3e}  (tol {'~2e-2'})")
    if v44_ms:
        print(f"\n  (V44 standalone: {v44_ms:.4f} ms — paste from ./build/bin/gqa_bwd_v44 output)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--kv-heads", type=int, dest="kv_heads", default=4)
    ap.add_argument("--hdim", type=int, default=128)
    ap.add_argument("--seqlens", default="4096",
                    help="comma list, e.g. 2048,4096,8192")
    ap.add_argument("--group-ratios", dest="group_ratios", default="3",
                    help="G=Hq/Hkv, e.g. 3 (matches V44 default Hq=12/Hkv=4)")
    ap.add_argument("--causal", action="store_true", default=True)
    ap.add_argument("--no-causal", dest="causal", action="store_false")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--competitors", default="sdpa_bwd,cudnn_bwd,fa4_bwd,flashinfer_bwd",
                    help="comma list of backward competitors")
    ap.add_argument("--v44-ms", type=float, dest="v44_ms", default=None,
                    help="V44 median ms (from C++ standalone) for speedup column")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", help="write results JSON here")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr); sys.exit(1)
    cc = torch.cuda.get_device_capability(0)
    print(f"# device: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")
    print(f"# GQA backward benchmark — comparing against V44")

    names = [n.strip() for n in args.competitors.split(",") if n.strip()]
    competitors = [BWD_ALL[n] for n in names if n in BWD_ALL]

    seqlens = [int(x) for x in args.seqlens.split(",")]
    ratios  = [int(x) for x in args.group_ratios.split(",")]

    all_out = []
    for s in seqlens:
        for g in ratios:
            shape = Shape(B=args.batch, Hq=args.kv_heads*g, Hkv=args.kv_heads,
                          S=s, D=args.hdim, causal=args.causal, dtype=args.dtype)
            res = run_backward(shape, competitors, args)
            print_table(shape, res, v44_ms=args.v44_ms)
            all_out.append({"shape": str(shape), "results": [r.__dict__ for r in res]})

    if args.json:
        Path(args.json).write_text(json.dumps(all_out, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
