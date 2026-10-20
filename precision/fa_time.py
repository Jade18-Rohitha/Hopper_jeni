"""
fa_time.py — FlashAttention-3 and FlashAttention-4 latency on the SAME inputs our C++ benches use (data/gqa_*.bin).

Fair-comparison rules (same as cudnn_time.py):
  * same inputs: data/gqa_{q,k,v,do}.bin from precision/baseline_gqa.py, causal, bf16, S=4096, D=128, GQA
  * FA's native layout is [B, S, H, D]; the transposes happen ONCE, OUTSIDE every timed region
  * correctness first: O, dQ, dK, dV vs the PyTorch reference in data/ (|a-b| <= 2e-2 + 2e-2*|ref|, as checkResult)
  * timing = triton.testing.do_bench (L2 flush per rep, async queue), three numbers per library:
      FA3 fwd / FA4 fwd            forward only (no autograd graph)              <-> VG1 fwd
      FA3 bwd-only / FA4 bwd-only  autograd.grad on a pre-built graph (includes   <-> VG1 bwd
                                   FA's own preprocess + dQ postprocess kernels)
      FA3 fwd+bwd / FA4 fwd+bwd    out = fa(q,k,v); out.backward(dO)              <-> VG1 fwd+bwd

  FA3 = flash_attn_interface (Dao-AILab/flash-attention hopper/ build); FA4 = flash_attn.cute (CuTe DSL).
  Install both on the H200 with: bash harness/setup_sota_h200.sh fa4 fa3
  A library that is missing or rejects this shape prints "# FAx: unavailable (...)" and is skipped.

usage: python3 precision/fa_time.py B Hq Hkv            (after baseline_gqa.py B Hq Hkv)
       FA_CHECK=0 python3 precision/fa_time.py ...      # skip the correctness checks (sweep runs after the first)
"""
import importlib
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import triton.testing

# FA4's CuTe DSL warns on every JIT call that its aux_data argument is not a JitArgument (harmless, floods the logs)
warnings.filterwarnings("ignore", message=r"Argument aux_data .* cannot be converted to a JitArgument")
B, Hq, Hkv = (int(a) for a in sys.argv[1:4])
S, D = 4096, 128
DATA = Path(__file__).parent.parent / "data"
CHECK = os.environ.get("FA_CHECK", "1") != "0"


def load(name, *shape, dtype=torch.bfloat16):
    a = np.fromfile(DATA / f"gqa_{name}.bin", dtype=np.float32)
    assert a.size == math.prod(shape), f"gqa_{name}.bin size mismatch — rerun baseline_gqa.py {B} {Hq} {Hkv}"
    return torch.from_numpy(a).view(*shape).to("cuda", dtype)


# [B, H, S, D] -> FA's [B, S, H, D], once, outside every timer
bshd = lambda t: t.transpose(1, 2).contiguous()
q, do = bshd(load("q", B, Hq, S, D)), bshd(load("do", B, Hq, S, D))
k, v = bshd(load("k", B, Hkv, S, D)), bshd(load("v", B, Hkv, S, D))
scale = 1.0 / math.sqrt(D)


def find(mods):
    for m in mods:
        try:
            mod = importlib.import_module(m)
        except Exception:
            continue
        if hasattr(mod, "flash_attn_func"):
            return getattr(mod, "flash_attn_func"), m
    return None, None


LIBS = {
    "FA3": ("flash_attn_interface", "flash_attn_3.flash_attn_interface"),
    "FA4": ("flash_attn.cute.interface", "flash_attn.cute"),
}


def bench(fn):
    med = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5])
    return med[0] if isinstance(med, (list, tuple)) else med


def check(label, name, got_bshd, ref_name, H):
    ref = bshd(load(ref_name, B, H, S, D, dtype=torch.float32))
    g = got_bshd.float()
    bad = ((g - ref).abs() > 2e-2 + 2e-2 * ref.abs()).sum().item()
    err = (g - ref).abs()
    status = "Values match!" if bad == 0 else f"Values don't match ({bad} elements over tol)"
    print(f"CHECK {label} {name}:  max_abs={err.max().item():.3e} mean_abs={err.mean().item():.3e}  {status}")


print(f"B={B} Hq={Hq} Hkv={Hkv}  (FA layout [B,S,H,D], transposed outside the timers)")
for label, mods in LIBS.items():
    fn, modname = find(mods)
    if fn is None:
        print(f"# {label}: unavailable (none of {mods} importable)")
        continue
    call = lambda q_, k_, v_: (lambda r: r[0] if isinstance(r, (tuple, list)) else r)(
        fn(q_, k_, v_, softmax_scale=scale, causal=True))
    leaves = lambda: [t.detach().requires_grad_(True) for t in (q, k, v)]
    try:
        qa, ka, va = leaves()
        o = call(qa, ka, va)                      # pre-built graph for the bwd-only timing (also JIT-compiles FA4)
        grads = torch.autograd.grad(o, (qa, ka, va), do, retain_graph=True)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"# {label}: unavailable ({type(e).__name__}: {str(e).splitlines()[0][:120]})")
        continue
    print(f"# {label} -> {modname}")
    if CHECK:
        check(label, "O ", o, "o", Hq)
        check(label, "dQ", grads[0], "dq", Hq)
        check(label, "dK", grads[1], "dk", Hkv)
        check(label, "dV", grads[2], "dv", Hkv)

    def fwd_bwd():
        qb, kb, vb = leaves()
        call(qb, kb, vb).backward(do)

    with torch.no_grad():
        t_f = bench(lambda: call(q, k, v))
    t_b = bench(lambda: torch.autograd.grad(o, (qa, ka, va), do, retain_graph=True))
    t_fb = bench(fwd_bwd)
    print(f"{label} fwd       {t_f:.4f} ms")
    print(f"{label} bwd-only  {t_b:.4f} ms")
    print(f"{label} fwd+bwd   {t_fb:.4f} ms")
