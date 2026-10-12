"""
cudnn_time.py — cuDNN SDPA latency on the SAME inputs our C++ benches use (data/gqa_*.bin).

Prints three medians, all with triton.testing.do_bench (L2 flush per rep, async queue),
backend FORCED to cuDNN (sdpa_kernel raises instead of silently falling back):
  cuDNN fwd        forward only (no autograd graph)          <-> OURS fwd V62
  cuDNN bwd-only   autograd.grad on a pre-built graph         <-> OURS bwd Vj1d
  cuDNN fwd+bwd    sdpa(...).backward(dO), the training step  <-> OURS fwd+bwd V62+Vj1d

usage: python precision/cudnn_time.py B Hq Hkv     (after baseline_gqa.py B Hq Hkv)
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import triton.testing
from torch.nn.attention import sdpa_kernel, SDPBackend

B, Hq, Hkv = (int(a) for a in sys.argv[1:4])
S, D = 4096, 128
DATA = Path(__file__).parent.parent / "data"


def load(name, *shape):
    a = np.fromfile(DATA / f"gqa_{name}.bin", dtype=np.float32)
    assert a.size == math.prod(shape), f"gqa_{name}.bin size mismatch — rerun baseline_gqa.py {B} {Hq} {Hkv}"
    return torch.from_numpy(a).view(*shape).to("cuda", torch.bfloat16)


q, do = load("q", B, Hq, S, D), load("do", B, Hq, S, D)
k, v = load("k", B, Hkv, S, D), load("v", B, Hkv, S, D)
scale = 1.0 / math.sqrt(D)


def sdpa(q_, k_, v_):
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(q_, k_, v_, scale=scale, is_causal=True, enable_gqa=True)


def leaves():
    return [t.detach().requires_grad_(True) for t in (q, k, v)]


qa, ka, va = leaves()
o = sdpa(qa, ka, va)
assert "Cudnn" in type(o.grad_fn).__name__, f"not cuDNN: {type(o.grad_fn).__name__}"


def fwd_bwd():
    qb, kb, vb = leaves()
    sdpa(qb, kb, vb).backward(do)


fns = {
    "cuDNN fwd     ": lambda: sdpa(q, k, v),
    "cuDNN bwd-only": lambda: torch.autograd.grad(o, (qa, ka, va), do, retain_graph=True),
    "cuDNN fwd+bwd ": fwd_bwd,
}
print(f"cuDNN forced ({type(o.grad_fn).__name__})  B={B} Hq={Hq} Hkv={Hkv}")
for name, fn in fns.items():
    med = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5])
    med = med[0] if isinstance(med, (list, tuple)) else med
    print(f"{name}  {med:.4f} ms")
