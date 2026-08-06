"""
competitors.py -- the SOTA attention field, one self-detecting plugin each.

Every competitor:
  * .available(shape) -> (ok: bool, reason: str)   # installed? supports this arch/shape?
  * .forward(q,k,v,shape) -> out [B,Hq,S,D]         # for the correctness gate
  * .make(q,k,v,shape) -> callable                  # a zero-arg closure to time

Canonical tensor layout everywhere: q [B,Hq,S,D], k/v [B,Hkv,S,D].
Each plugin transposes internally to whatever its backend wants.

Kinds:
  'attention'  -> full GQA kernel (SDPA, cuDNN, FA4, FlashInfer, Triton, TK)
  'norm_rope'  -> QK-norm + RoPE reference for the fused track (Liger, Unsloth)

Missing/unsupported backends report a reason and are skipped -- never crash the run.
"""
from __future__ import annotations
import importlib.util
import math
import torch
import torch.nn.functional as F


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _cc() -> tuple[int, int]:
    return torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)


class Competitor:
    name = "base"
    kind = "attention"
    role = ""

    def available(self, shape):
        return True, ""

    def forward(self, q, k, v, shape):
        raise NotImplementedError

    def make(self, q, k, v, shape):
        return lambda: self.forward(q, k, v, shape)


# ---------------------------------------------------------------------------
# PyTorch SDPA -- the stock floor. Always available.
# ---------------------------------------------------------------------------
class SDPA(Competitor):
    name = "sdpa"
    role = "stock floor"

    def forward(self, q, k, v, shape):
        return F.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(shape.D),
            is_causal=shape.causal, enable_gqa=(shape.G > 1))


# ---------------------------------------------------------------------------
# cuDNN fused attention, reached through torch's backend selector.
# (No cudnn-frontend python dep needed -- SDPBackend.CUDNN_ATTENTION forces it.)
# ---------------------------------------------------------------------------
class CuDNN(Competitor):
    name = "cudnn"
    role = "production peak / SASS source"

    def _backend(self):
        from torch.nn.attention import SDPBackend
        return SDPBackend.CUDNN_ATTENTION

    def available(self, shape):
        try:
            from torch.nn.attention import sdpa_kernel  # noqa: F401
            from torch.nn.attention import SDPBackend    # noqa: F401
        except Exception as e:
            return False, f"torch.nn.attention unavailable: {e}"
        return True, ""

    def forward(self, q, k, v, shape):
        from torch.nn.attention import sdpa_kernel
        with sdpa_kernel(self._backend()):
            return F.scaled_dot_product_attention(
                q, k, v, scale=1.0 / math.sqrt(shape.D),
                is_causal=shape.causal, enable_gqa=(shape.G > 1))


# ---------------------------------------------------------------------------
# FlashAttention-4 (Dao-AILab, CuTe-DSL). Native GQA. Wants [B,S,H,D].
# ---------------------------------------------------------------------------
class FA4(Competitor):
    name = "fa4"
    role = "PEAK TARGET"

    def _fn(self):
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
            return False, "flash_attn_func not importable (run setup_fa4_b300.sh)"
        return True, ""

    def forward(self, q, k, v, shape):
        fn = self._fn()
        qs, ks, vs = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        out = fn(qs, ks, vs, causal=shape.causal)
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(1, 2)      # back to [B,Hq,S,D]


# ---------------------------------------------------------------------------
# FlashInfer -- inference-serving GQA/MQA kernels (SM100/SM103).
# ---------------------------------------------------------------------------
class FlashInfer(Competitor):
    name = "flashinfer"
    role = "serving kernel"

    def available(self, shape):
        cc = _cc()
        if not _has("flashinfer"):
            return False, "flashinfer not installed"
        if cc[0] < 9:
            return False, f"flashinfer GQA path targets sm_90+; this is sm_{cc[0]}{cc[1]}"
        return True, ""

    def forward(self, q, k, v, shape):
        import flashinfer
        # single-request prefill; flashinfer wants [S, H, D] per request, no batch dim,
        # so loop the batch (kept simple & correct; batched API varies by version).
        outs = []
        for b in range(shape.B):
            qb = q[b].transpose(0, 1).contiguous()     # [S, Hq, D]
            kb = k[b].transpose(0, 1).contiguous()     # [S, Hkv, D]
            vb = v[b].transpose(0, 1).contiguous()
            ob = flashinfer.single_prefill_with_kv_cache(
                qb, kb, vb, causal=shape.causal)
            outs.append(ob.transpose(0, 1))            # [Hq, S, D]
        return torch.stack(outs, 0)


# ---------------------------------------------------------------------------
# Triton flash (GQA, causal) -- portability floor. Adapted from baseline_gqa.py.
# ---------------------------------------------------------------------------
class TritonFlash(Competitor):
    name = "triton"
    role = "portability floor"
    _kernel = None

    def available(self, shape):
        if not _has("triton"):
            return False, "triton not installed"
        return True, ""

    def _build(self):
        if TritonFlash._kernel is not None:
            return
        import triton
        import triton.language as tl

        @triton.jit
        def _fwd(Q, K, V, O,
                 sqb, sqh, sqs, sqd, skb, skh, sks, skd,
                 svb, svh, svs, svd, sob, soh, sos, sod,
                 Hq, S, G, scale, CAUSAL: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
            start_m = tl.program_id(0)
            off_bh = tl.program_id(1)
            off_b = off_bh // Hq
            off_hq = off_bh % Hq
            off_hkv = off_hq // G
            qb = Q + off_b * sqb + off_hq * sqh
            kb = K + off_b * skb + off_hkv * skh
            vb = V + off_b * svb + off_hkv * svh
            ob = O + off_b * sob + off_hq * soh
            offs_m = start_m * BM + tl.arange(0, BM)
            offs_d = tl.arange(0, D)
            offs_n = tl.arange(0, BN)
            q = tl.load(qb + offs_m[:, None] * sqs + offs_d[None, :] * sqd,
                        mask=offs_m[:, None] < S, other=0.0)
            m_i = tl.full([BM], -float("inf"), tl.float32)
            l_i = tl.zeros([BM], tl.float32)
            acc = tl.zeros([BM, D], tl.float32)
            n_end = S
            if CAUSAL:
                n_end = tl.minimum(S, (start_m + 1) * BM)
            for start_n in range(0, n_end, BN):
                nc = start_n + offs_n
                k = tl.load(kb + nc[None, :] * sks + offs_d[:, None] * skd,
                            mask=nc[None, :] < S, other=0.0)
                qk = tl.dot(q, k) * scale
                qk = tl.where(nc[None, :] < S, qk, -float("inf"))
                if CAUSAL:
                    qk = tl.where(offs_m[:, None] >= nc[None, :], qk, -float("inf"))
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                p = tl.exp(qk - m_ij[:, None])
                alpha = tl.exp(m_i - m_ij)
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None]
                vv = tl.load(vb + nc[:, None] * svs + offs_d[None, :] * svd,
                             mask=nc[:, None] < S, other=0.0)
                acc += tl.dot(p.to(vv.dtype), vv)
                m_i = m_ij
            acc = acc / l_i[:, None]
            tl.store(ob + offs_m[:, None] * sos + offs_d[None, :] * sod,
                     acc.to(O.dtype.element_ty), mask=offs_m[:, None] < S)

        TritonFlash._kernel = _fwd

    def forward(self, q, k, v, shape, BM=128, BN=64):
        import triton
        self._build()
        o = torch.empty_like(q)
        grid = (triton.cdiv(shape.S, BM), shape.B * shape.Hq)
        TritonFlash._kernel[grid](
            q, k, v, o, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            shape.Hq, shape.S, shape.G, 1.0 / math.sqrt(shape.D),
            1 if shape.causal else 0, BM=BM, BN=BN, D=shape.D,
            num_warps=4, num_stages=2)
        return o


# ---------------------------------------------------------------------------
# ThunderKittens -- tile-DSL reference (HazyResearch/ThunderKittens).
#
# NOT a pip package -- there is no "thunderkittens"/"tk" module to pip-install.
# TK is header-only C++/CUDA; each kernel is built individually as its own
# PyTorch extension via a per-kernel Makefile. kernels/attention/mha_h100/
# genuinely supports GQA despite the "mha" name: its forward pass computes
# kv_head_idx = blockIdx.y / hr with hr = qo_heads/kv_heads, and its own
# gentests.py validates against an explicit Llama-3 GQA reference (see the
# verification that led to this class). D must be 64 or 128, S % 64 == 0,
# causal and non-causal both supported. Build it with
# base/harness/setup_thunderkittens_h200.sh, which `make`s this exact target.
#
# The build produces a bare "_C<ext-suffix>.so" in the kernel directory --
# pybind11's PYBIND11_MODULE macro is single-phase init, so the compiled
# module can ONLY be imported under the exact name it was built with (its
# init symbol is literally PyInit__C); loading it from an arbitrary path
# under a different name (importlib.util.spec_from_file_location with a
# private name) fails with "does not define module export function".
# Verified locally: the only thing that actually works is prepending the
# kernel dir to sys.path and `import _C`, same as TK's own benchmark.py /
# test_correctness.py do. "_C" is a generic name, so guard the sys.modules
# cache in case some *other* extension already claimed it in this process.
#
# ***DANGER, verified locally***: TK's CUDA error checker
# (include/common/util.cuh, CHECK_CUDA_ERROR) calls std::exit(EXIT_FAILURE)
# directly on ANY cudaGetLastError() failure -- not a catchable C++/Python
# exception. run_gqa_bench.py's `try/except Exception` around cb.forward()
# and cb.make() CANNOT catch this: a bad launch config, an unsupported shape
# combination TK doesn't validate ahead of time, or any other CUDA runtime
# error from this kernel kills the ENTIRE benchmark process mid-sweep, not
# just this row. Reproduced by hand: running this kernel on a non-sm_90 GPU
# hits exactly this path (confirmed harmless on a real H200, since the .so is
# built for sm_90a and the H200 IS sm_90a -- but nothing rules out some other
# shape hitting it there instead). If you're running a long unattended sweep
# across many shapes, run --competitors thunderkittens by itself first to
# gate it before trusting it inside a big multi-shape sweep.
# ---------------------------------------------------------------------------
class ThunderKittens(Competitor):
    name = "thunderkittens"
    role = "tile-DSL reference"

    _mod = None

    def _root(self):
        import os
        return os.environ.get("THUNDERKITTENS_ROOT", os.path.expanduser("~/ThunderKittens"))

    def _kernel_dir(self):
        import os
        return os.path.join(self._root(), "kernels", "attention", "mha_h100")

    def _so_path(self):
        import glob, os
        matches = glob.glob(os.path.join(self._kernel_dir(), "_C*.so"))
        return matches[0] if matches else None

    def available(self, shape):
        cc = _cc()
        if cc[0] != 9:
            return False, f"TK mha_h100 targets sm_90 (Hopper); this is sm_{cc[0]}{cc[1]}"
        if shape.D not in (64, 128):
            return False, f"TK mha_h100 only supports D in {{64,128}}; shape has D={shape.D}"
        if shape.S % 64 != 0:
            return False, f"TK mha_h100 requires S % 64 == 0; shape has S={shape.S}"
        if self._so_path() is None:
            return False, (f"TK mha_h100 not built -- run "
                            f"base/harness/setup_thunderkittens_h200.sh "
                            f"(looked in {self._kernel_dir()})")
        return True, ""

    def _load(self):
        if ThunderKittens._mod is not None:
            return ThunderKittens._mod
        import sys
        cached = sys.modules.get("_C")
        if cached is not None and hasattr(cached, "mha_forward"):
            ThunderKittens._mod = cached
            return cached
        kdir = self._kernel_dir()
        if kdir not in sys.path:
            sys.path.insert(0, kdir)
        import _C as mod   # noqa: this IS the TK mha_h100 extension, see class docstring
        ThunderKittens._mod = mod
        return mod

    def forward(self, q, k, v, shape):
        tk = self._load()
        # mha_h100's binding is q,k,v,causal -> (o, l_vec); canonical [B,H,S,D]
        # layout already matches TK's (B,H,N,D), no transpose needed.
        o, _l_vec = tk.mha_forward(q.contiguous(), k.contiguous(), v.contiguous(), shape.causal)
        return o


# ---------------------------------------------------------------------------
# Fused-track references: QK-norm (RMSNorm) + RoPE, the ops FA4/cuDNN do NOT fuse.
# These are 'norm_rope' kind -- measured as the pre-attention op stack so your
# GQA_fused kernel's fusion win shows up instead of hiding inside "attention".
# ---------------------------------------------------------------------------
def _rope(x, cos, sin):
    # NeoX half-split rotation. x: [..., D]
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin


class LigerNormRope(Competitor):
    name = "liger"
    kind = "norm_rope"
    role = "QK-norm+RoPE ref (Liger)"

    def available(self, shape):
        if not _has("liger_kernel"):
            return False, "liger_kernel not installed"
        return True, ""

    def forward(self, q, k, v, shape):
        # per-head RMSNorm(Q), RMSNorm(K), then RoPE -- Qwen3 order.
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        wq = torch.ones(shape.D, device=q.device, dtype=torch.float32)
        qn = LigerRMSNormFunction.apply(q.float(), wq, 1e-6, 0.0, "gemma")
        kn = LigerRMSNormFunction.apply(k.float(), wq, 1e-6, 0.0, "gemma")
        pos = torch.arange(shape.S, device=q.device, dtype=torch.float32)
        inv = 1.0 / (10000 ** (torch.arange(0, shape.D, 2, device=q.device).float()
                               / shape.D))
        ang = pos[:, None] * inv[None, :]
        cos = torch.cat([ang.cos(), ang.cos()], -1)[None, None]
        sin = torch.cat([ang.sin(), ang.sin()], -1)[None, None]
        return _rope(qn, cos, sin), _rope(kn, cos, sin)

    def make(self, q, k, v, shape):
        return lambda: self.forward(q, k, v, shape)


class UnslothNormRope(Competitor):
    name = "unsloth"
    kind = "norm_rope"
    role = "QK-norm+RoPE ref (Unsloth)"

    def available(self, shape):
        if not _has("unsloth"):
            return False, "unsloth not installed"
        return True, ""

    def forward(self, q, k, v, shape):
        raise NotImplementedError("wire unsloth.kernels rms_norm + rope when installed")


# ---------------------------------------------------------------------------
# Your kernel: loads a compiled .so and calls its entrypoint. Stub until wired
# on the B300 (compile GQA_sm103_causal_v2 V34 -> shared lib, expose a symbol).
# ---------------------------------------------------------------------------
class YourKernel(Competitor):
    name = "yours"
    role = "YOUR kernel"

    def __init__(self, so_path=None, entry="launch_gqa"):
        self.so_path = so_path
        self.entry = entry

    def available(self, shape):
        if not self.so_path:
            return False, "no --your-so provided (compile V34 to a .so and pass it)"
        import os
        if not os.path.exists(self.so_path):
            return False, f"{self.so_path} not found"
        return True, ""

    def forward(self, q, k, v, shape):
        raise NotImplementedError(
            "bind your compiled kernel here (ctypes/torch.utils.cpp_extension). "
            "It must take canonical [B,Hq,S,D] bf16 and return the same.")


ATTENTION = [FA4(), CuDNN(), SDPA(), FlashInfer(), TritonFlash(), ThunderKittens()]
NORM_ROPE = [LigerNormRope(), UnslothNormRope()]
ALL = {c.name: c for c in ATTENTION + NORM_ROPE}


def get(names):
    out = []
    for n in names:
        if n in ALL:
            out.append(ALL[n])
    return out
