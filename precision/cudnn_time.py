"""
cudnn_time.py — cuDNN SDPA latency on the SAME inputs our C++ benches use (data/gqa_*.bin).

Prints three medians, all with triton.testing.do_bench (L2 flush per rep, async queue),
backend FORCED to cuDNN (sdpa_kernel raises instead of silently falling back):
  cuDNN fwd        forward only (no autograd graph)          <-> OURS fwd V62
  cuDNN bwd-only   autograd.grad on a pre-built graph         <-> OURS bwd Vj1d
  cuDNN fwd+bwd    sdpa(...).backward(dO), the training step  <-> OURS fwd+bwd V62+Vj1d

Plus SUSTAINED cuDNN bwd-only (same protocol as gqa_bwd_vm1_test): 2 s idle, 1 s warm-up, ~3 s of back-to-back
autograd.grad, 2 windows; ms/bwd, avg W and J/bwd from NVML's energy counter of the GPU torch runs on.

usage: python precision/cudnn_time.py B Hq Hkv     (after baseline_gqa.py B Hq Hkv)
"""
import ctypes
import math
import sys
import time
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


BACKENDS = {"cuDNN": SDPBackend.CUDNN_ATTENTION}


def sdpa(q_, k_, v_, backend):
    if backend is None:
        return F.scaled_dot_product_attention(q_, k_, v_, scale=scale, is_causal=True, enable_gqa=True)
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q_, k_, v_, scale=scale, is_causal=True, enable_gqa=True)


def leaves():
    return [t.detach().requires_grad_(True) for t in (q, k, v)]


def bench(fn):
    med = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5])
    return med[0] if isinstance(med, (list, tuple)) else med


print(f"B={B} Hq={Hq} Hkv={Hkv}")
for label, backend in BACKENDS.items():
    qa, ka, va = leaves()
    try:
        o = sdpa(qa, ka, va, backend)
    except RuntimeError as e:
        print(f"# {label}: unavailable ({str(e).splitlines()[0][:100]})")
        continue
    gf = type(o.grad_fn).__name__
    if label == "cuDNN":
        assert "Cudnn" in gf, f"not cuDNN: {gf}"

    def fwd_bwd(backend=backend):
        qb, kb, vb = leaves()
        sdpa(qb, kb, vb, backend).backward(do)

    print(f"# {label} -> {gf}")
    print(f"{label} fwd       {bench(lambda: sdpa(q, k, v, backend)):.4f} ms")
    print(f"{label} bwd-only  {bench(lambda: torch.autograd.grad(o, (qa, ka, va), do, retain_graph=True)):.4f} ms")
    print(f"{label} fwd+bwd   {bench(fwd_bwd):.4f} ms")


# ---- SUSTAINED cuDNN bwd-only (training-like: back-to-back, no L2 flush) ----
class _Nvml:
    def __init__(self):
        self.h = None
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
            if self.lib.nvmlInit_v2() != 0: return
            pr = torch.cuda.get_device_properties(0)
            bus = f"{getattr(pr, 'pci_domain_id', 0):08x}:{pr.pci_bus_id:02x}:{pr.pci_device_id:02x}.0".encode()
            h = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByPciBusId_v2(bus, ctypes.byref(h)) == 0: self.h = h
            idx = ctypes.c_uint()
            if self.h is not None and self.lib.nvmlDeviceGetIndex(self.h, ctypes.byref(idx)) == 0:
                print(f"NVML: measuring GPU {idx.value} (PCI {bus.decode()})")
        except Exception as e:                                # energy is optional; timing still works
            print(f"NVML unavailable ({e}); sustained timing only")
    def energy_mJ(self):
        e = ctypes.c_ulonglong(); self.lib.nvmlDeviceGetTotalEnergyConsumption(self.h, ctypes.byref(e)); return e.value
    def sm_mhz(self):
        c = ctypes.c_uint(); self.lib.nvmlDeviceGetClockInfo(self.h, 1, ctypes.byref(c)); return c.value   # 1 = NVML_CLOCK_SM


def sustained(name, fn, est_ms, nv, warm_s=1.0, secs=3.0):
    for _ in range(max(10, int(warm_s * 1000 / est_ms))): fn()
    torch.cuda.synchronize()
    n = max(20, int(secs * 1000 / est_ms))
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0 = nv.energy_mJ() if nv.h is not None else 0
    a.record()
    for _ in range(n): fn()
    b.record()
    time.sleep(secs * 0.5)                                    # sample the clock mid-run
    smv = nv.sm_mhz() if nv.h is not None else 0
    b.synchronize()
    e1 = nv.energy_mJ() if nv.h is not None else 0
    ms = a.elapsed_time(b); j = (e1 - e0) / 1000.0
    if nv.h is not None:
        print(f"SUSTAINED {name}  {ms/n:.4f} ms/bwd  {j/(ms/1000):7.1f} W avg  {j/n:.3f} J/bwd  ({n} launches, {ms/1000:.2f} s)  mid-run SM {smv} MHz")
    else:
        print(f"SUSTAINED {name}  {ms/n:.4f} ms/bwd  ({n} launches, {ms/1000:.2f} s)")


qa, ka, va = leaves()
o = sdpa(qa, ka, va, SDPBackend.CUDNN_ATTENTION)
bwd_only = lambda: torch.autograd.grad(o, (qa, ka, va), do, retain_graph=True)
est = bench(bwd_only)
nv = _Nvml()
print("\n== SUSTAINED (back-to-back ~3 s, after 1 s warm-up; 2 s idle before each window) ==")
for _ in range(2):
    time.sleep(2)
    sustained("cuDNN bwd-only", bwd_only, est, nv)
