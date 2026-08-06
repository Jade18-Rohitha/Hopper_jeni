"""
bench_core.py -- timing + correctness primitives shared by every competitor.

Honesty rules baked in (from kernel_guidelines.md):
  * CUDA events only (device time), measured at the API boundary.
  * L2 flushed before every timed rep (256 MB scratch memset).
  * Warmup (default 25) is NOT timed and NOT flushed; timed reps (default 100) are.
  * Median is the primary figure; report mean/std/min/max/p5/p95 too.
  * >=3 independent runs; the reported median is the median of per-run medians.
  * A kernel that fails the correctness gate is never assigned a speedup.

FLOPs (algorithmic): 4 * B * Hq * S * S * D, halved for causal.
Bytes (DRAM traffic, bf16): Q+O = 2*B*Hq*S*D*2 ; K+V = 2*B*Hkv*S*D*2.
"""
from __future__ import annotations
import statistics
from dataclasses import dataclass, field
import torch


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

    def flops(self) -> float:
        f = 4.0 * self.B * self.Hq * self.S * self.S * self.D
        return f * 0.5 if self.causal else f

    def bytes(self) -> float:
        el = 2 if self.dtype in ("bf16", "fp16") else 4
        qo = 2 * self.B * self.Hq * self.S * self.D * el     # read Q, write O
        kv = 2 * self.B * self.Hkv * self.S * self.D * el    # read K, V
        return qo + kv

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
    min_ms: float = float("nan")
    p5_ms: float = float("nan")
    p95_ms: float = float("nan")
    tflops: float = float("nan")
    gbps: float = float("nan")
    max_abs: float = float("nan")
    mean_abs: float = float("nan")
    max_rel: float = float("nan")
    correct: bool | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# canonical inputs: q [B,Hq,S,D], k/v [B,Hkv,S,D], bf16-rounded so every
# competitor starts from identical bits regardless of the layout it wants.
# ---------------------------------------------------------------------------
def make_inputs(shape: Shape, seed: int = 42, device: str = "cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    dt = shape.torch_dtype()
    q = torch.rand(shape.B, shape.Hq, shape.S, shape.D, generator=g,
                   device=device, dtype=torch.float32).to(dt)
    k = torch.rand(shape.B, shape.Hkv, shape.S, shape.D, generator=g,
                   device=device, dtype=torch.float32).to(dt)
    v = torch.rand(shape.B, shape.Hkv, shape.S, shape.D, generator=g,
                   device=device, dtype=torch.float32).to(dt)
    return q, k, v


class L2Flusher:
    """Memset a >L2 scratch buffer to evict hot data between timed reps."""
    def __init__(self, nbytes: int = 256 * 1024 * 1024, device: str = "cuda"):
        self.buf = torch.empty(nbytes, dtype=torch.int8, device=device)

    def flush(self):
        self.buf.zero_()


def time_fn(fn, iters=100, warmup=25, runs=3, flush=True, device="cuda"):
    """Return (per-run medians, all timed ms). fn() must run one full API call."""
    flusher = L2Flusher(device=device) if flush else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    run_medians, all_ms = [], []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(runs):
        reps = []
        for _ in range(iters):
            if flush:
                flusher.flush()
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            reps.append(start.elapsed_time(end))   # ms
        run_medians.append(statistics.median(reps))
        all_ms += reps
    return run_medians, all_ms


def summarize(name, run_medians, all_ms, shape: Shape) -> Result:
    med = statistics.median(run_medians)          # median of per-run medians
    all_sorted = sorted(all_ms)
    n = len(all_sorted)
    r = Result(name=name, ok=True)
    r.median_ms = med
    r.mean_ms = statistics.fmean(all_ms)
    r.std_ms = statistics.pstdev(all_ms) if n > 1 else 0.0
    r.min_ms = all_sorted[0]
    r.p5_ms = all_sorted[max(0, int(0.05 * n) - 1)]
    r.p95_ms = all_sorted[min(n - 1, int(0.95 * n))]
    r.tflops = shape.flops() / (med * 1e-3) / 1e12
    r.gbps = shape.bytes() / (med * 1e-3) / 1e9
    return r


# ---------------------------------------------------------------------------
# reference + correctness gate
# ---------------------------------------------------------------------------
def reference_output(q, k, v, shape: Shape, mode="fp32"):
    """Ground-truth attention output in canonical [B,Hq,S,D] layout.

    mode='fp32': upcast inputs, fp32 math via SDPA (cheap, good enough for a
                 bf16 tol=2e-2 gate).  mode='fp64': strict per-(b,hq) fp64 loop
                 (matches baseline_gqa.py; expensive, use to certify).
    """
    import torch.nn.functional as F
    scale = 1.0 / (shape.D ** 0.5)
    if mode == "fp32":
        out = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), scale=scale,
            is_causal=shape.causal, enable_gqa=(shape.G > 1))
        return out
    # fp64, one head at a time to bound memory
    B, Hq, S, D, G = shape.B, shape.Hq, shape.S, shape.D, shape.G
    out = torch.empty(B, Hq, S, D, dtype=torch.float32, device=q.device)
    causal_mask = None
    if shape.causal:
        causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool,
                                            device=q.device), diagonal=1)
    for b in range(B):
        for hq in range(Hq):
            hkv = hq // G
            sc = (q[b, hq].double() @ k[b, hkv].double().T) * scale
            if causal_mask is not None:
                sc = sc.masked_fill(causal_mask, float("-inf"))
            w = torch.softmax(sc, dim=-1)
            out[b, hq] = (w @ v[b, hkv].double()).float()
    return out


def correctness(out, ref, atol=2e-2, rtol=2e-2):
    """Return (passed, max_abs, mean_abs, max_rel) in fp32."""
    o = out.float()
    r = ref.float()
    diff = (o - r).abs()
    rel = diff / r.abs().clamp_min(1e-5)
    max_abs = diff.max().item()
    passed = bool(max_abs <= atol + rtol * r.abs().max().item())
    return passed, max_abs, diff.mean().item(), rel.max().item()
