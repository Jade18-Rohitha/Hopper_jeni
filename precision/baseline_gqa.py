"""
Reference generator + precision/latency comparison for the Hopper (H200) GQA kernel.

Grouped-Query Attention: Hq query heads share Hkv key/value heads, with
G = Hq/Hkv query heads per kv head (query head hq reads kv head hq // G).
Inputs/outputs are bf16 with fp32 accumulation, matching V1/V2 in
src/attention/GQA_bwd.cu. This script is device-agnostic (pure PyTorch +
Triton) and produces identical reference .bin files on H200.

Saves (all float32 .bin so the CUDA side can reuse loadBin; the stored values are
the bf16-ROUNDED inputs widened back to float32, so the kernel and the reference
start from identical input bits):
  - gqa_q.bin                 [B, Hq,  S, D]
  - gqa_k.bin / gqa_v.bin     [B, Hkv, S, D]
  - gqa_o.bin                 bf16 SDPA output (apples-to-apples target), [B, Hq, S, D]
  - gqa_o_fp64.bin            fp64 ground-truth output,                   [B, Hq, S, D]
  - gqa_lse.bin               fp64 ground-truth log-sum-exp of the logits,[B, Hq, S]

Dimensions match GQA_bwd.cu: B=8, Hq=12, Hkv=4, S=4096, D=128.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import math
import triton
import triton.language as tl
import triton.testing

import sys
# Shape from argv: baseline_gqa.py [B] [Hq] [Hkv]   (default 8 12 4; S=4096, D=128 fixed)
B, Hq, Hkv, S, D = 4, 24, 8, 4096, 128
if len(sys.argv) >= 4:
    B, Hq, Hkv = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
G = Hq // Hkv

DATA_DIR = Path(__file__).parent.parent / "data"


def fp64_reference(q, k, v, scale):
    """Double-precision ground truth + log-sum-exp, computed one head at a time.

    The full [B, Hq, S, S] score tensor would be ~25 GB in fp64, so we loop over
    (b, hq) and keep only one [S, S] block live at a time.  Inputs are
    [B, Hq, S, D] (q) and [B, Hkv, S, D] (k, v); GQA routing is hkv = hq // G.
    """
    out = torch.empty(B, Hq, S, D, dtype=torch.float32)
    lse = torch.empty(B, Hq, S,    dtype=torch.float32)
    for b in range(B):
        for hq in range(Hq):
            hkv = hq // G
            q64 = q[b, hq ].double()                       # [S, D]
            k64 = k[b, hkv].double()                       # [S, D]
            v64 = v[b, hkv].double()                       # [S, D]
            scores = (q64 @ k64.transpose(-2, -1)) * scale  # [S, S]
            causal_mask = torch.ones(S, S, device=scores.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal_mask, float('-inf'))
            lse[b, hq] = torch.logsumexp(scores, dim=-1).float().cpu()
            w = torch.softmax(scores, dim=-1)
            out[b, hq] = (w @ v64).float().cpu()
    return out, lse


def sdpa_gqa(q, k, v, scale):
    """PyTorch GQA via scaled_dot_product_attention (enable_gqa expands KV heads)."""
    return F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True, is_causal=True)


def sdpa_lse(q, k, scale):
    """Reference log-sum-exp [B, Hq, S] for GQA, computed head-by-head in fp32.

    Processes one [S, S] score tile at a time (same memory-bounding rationale as
    fp64_reference) so this stays independent of the Triton forward kernel.
    """
    B, Hq, S, _ = q.shape
    Hkv = k.shape[1]
    G   = Hq // Hkv
    lse = torch.empty(B, Hq, S, device=q.device, dtype=torch.float32)
    for b in range(B):
        for hq in range(Hq):
            hkv = hq // G
            scores = (q[b, hq].float() @ k[b, hkv].float().t()) * scale
            causal_mask = torch.ones(S, S, device=scores.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal_mask, float('-inf'))
            lse[b, hq] = torch.logsumexp(scores, dim=-1)
    return lse


# ── Triton flash-attention forward (GQA, non-causal) ──────────────────────────
# One program computes a BLOCK_M tile of query rows for one (batch, query-head),
# streaming the keys in BLOCK_N tiles with an ONLINE softmax (running max/sum).
# GQA routing is the same as the CUDA kernel: kv head = query head // G.

@triton.jit
def _gqa_fwd_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    Hq, S, G, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    start_m = tl.program_id(0)        # which BLOCK_M query tile
    off_bh  = tl.program_id(1)        # flattened (batch, query-head)
    off_b   = off_bh // Hq
    off_hq  = off_bh %  Hq
    off_hkv = off_hq // G             # GQA: shared kv head

    q_base = Q + off_b * stride_qb + off_hq  * stride_qh
    k_base = K + off_b * stride_kb + off_hkv * stride_kh
    v_base = V + off_b * stride_vb + off_hkv * stride_vh
    o_base = O + off_b * stride_ob + off_hq  * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

    # load this block of queries once: [BLOCK_M, D]
    q = tl.load(q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < S, other=0.0)

    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # running denom
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)             # running Σ p·V

    for start_n in range(0, S, BLOCK_N):
        offs_n_curr = start_n + offs_n
        # K tile loaded transposed -> [D, BLOCK_N] so tl.dot(q, k) gives Q·Kᵀ
        k = tl.load(k_base + offs_n_curr[None, :] * stride_ks + offs_d[:, None] * stride_kd,
                    mask=offs_n_curr[None, :] < S, other=0.0)
        qk = tl.dot(q, k) * scale                              # [BLOCK_M, BLOCK_N]
        qk = tl.where(offs_n_curr[None, :] < S, qk, -float('inf'))

        m_ij  = tl.maximum(m_i, tl.max(qk, axis=1))            # new running max
        p     = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)                             # rescale old stats
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        acc   = acc * alpha[:, None]

        v = tl.load(v_base + offs_n_curr[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                    mask=offs_n_curr[:, None] < S, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < S)


def triton_gqa(q, k, v, scale, BLOCK_M=128, BLOCK_N=64):
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    o = torch.empty_like(q)
    grid = (triton.cdiv(S, BLOCK_M), B * Hq)
    _gqa_fwd_kernel[grid](
        q, k, v, o,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        Hq, S, G, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=2,
    )
    return o


# ── FP64 backward reference ───────────────────────────────────────────────────

def fp64_backward(q, k, v, do, scale):
    """Double-precision backward reference, one head at a time to bound memory."""
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    G   = Hq // Hkv
    dq = torch.zeros(B, Hq,  S, D, dtype=torch.float32)
    dk = torch.zeros(B, Hkv, S, D, dtype=torch.float32)
    dv = torch.zeros(B, Hkv, S, D, dtype=torch.float32)
    for b in range(B):
        for hq in range(Hq):
            hkv  = hq // G
            q64  = q[b, hq ].double()
            k64  = k[b, hkv].double()
            v64  = v[b, hkv].double()
            do64 = do[b, hq ].double()

            scores = (q64 @ k64.t()) * scale
            causal_mask = torch.ones(S, S, device=scores.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal_mask, float('-inf'))
            p      = torch.softmax(scores, dim=-1)              # [S, S]
            o64    = p @ v64                                    # [S, D]
            Di     = (do64 * o64).sum(-1, keepdim=True)         # [S, 1]
            dp     = do64 @ v64.t()                             # [S, S]
            ds     = p * (dp - Di)                              # [S, S]

            dq[b, hq ]  += (ds @ k64 * scale).float().cpu()
            dk[b, hkv]  += (ds.t() @ q64 * scale).float().cpu()
            dv[b, hkv]  += (p.t() @ do64).float().cpu()
    return dq, dk, dv


# ── Triton GQA Backward ───────────────────────────────────────────────────────
# Two kernels:
#   _gqa_bwd_dq   — one program per (batch, query-head, BLOCK_M query tile);
#                   streams K/V blocks to accumulate dQ.
#   _gqa_bwd_dkv  — one program per (batch, kv-head, BLOCK_N kv tile);
#                   streams all G*S query rows to accumulate dK and dV.
# Both recompute P from saved LSE (no full [S,S] store).

@triton.jit
def _gqa_bwd_dq(
    Q, K, V, DO, DQ, LSE, Delta,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    stride_db, stride_dh, stride_ds,
    Hq, S, G, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh  = tl.program_id(1)
    off_b   = off_bh // Hq
    off_hq  = off_bh %  Hq
    off_hkv = off_hq // G

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < S

    q_base   = Q     + off_b * stride_qb  + off_hq  * stride_qh
    k_base   = K     + off_b * stride_kb  + off_hkv * stride_kh
    v_base   = V     + off_b * stride_vb  + off_hkv * stride_vh
    do_base  = DO    + off_b * stride_dob + off_hq  * stride_doh
    dq_base  = DQ    + off_b * stride_dqb + off_hq  * stride_dqh
    lse_base = LSE   + off_b * stride_lb  + off_hq  * stride_lh
    d_base   = Delta + off_b * stride_db  + off_hq  * stride_dh

    q   = tl.load(q_base  + offs_m[:, None] * stride_qs  + offs_d[None, :] * stride_qd,
                  mask=mask_m[:, None], other=0.0)
    do  = tl.load(do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                  mask=mask_m[:, None], other=0.0)
    lse = tl.load(lse_base + offs_m * stride_ls, mask=mask_m, other=0.0)
    Di  = tl.load(d_base   + offs_m * stride_ds, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M          # causal: only key blocks up to this query tile
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        causal = offs_m[:, None] >= offs_n[None, :]

        k = tl.load(k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                    mask=mask_n[:, None], other=0.0)                     # [BLOCK_N, D]
        v = tl.load(v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                    mask=mask_n[:, None], other=0.0)                     # [BLOCK_N, D]

        qk = tl.dot(q, tl.trans(k)) * scale                             # [BLOCK_M, BLOCK_N]
        qk = tl.where(mask_n[None, :] & causal, qk, float('-inf'))
        p  = tl.exp(qk - lse[:, None])
        p  = tl.where(mask_n[None, :] & causal, p, 0.0)

        dp = tl.dot(do, tl.trans(v))                                     # [BLOCK_M, BLOCK_N]
        dp = tl.where(mask_n[None, :], dp, 0.0)
        ds = p * (dp - Di[:, None])
        ds = tl.where(mask_n[None, :], ds, 0.0)

        dq += tl.dot(ds.to(k.dtype), k) * scale                         # [BLOCK_M, D]

    tl.store(dq_base + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd,
             dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _gqa_bwd_dkv(
    Q, K, V, DO, DK, DV, LSE, Delta,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    stride_db, stride_dh, stride_ds,
    Hkv, S, scale,
    G: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    start_n  = tl.program_id(0)
    off_bhkv = tl.program_id(1)
    off_b    = off_bhkv // Hkv
    off_hkv  = off_bhkv %  Hkv

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < S

    k_base  = K  + off_b * stride_kb  + off_hkv * stride_kh
    v_base  = V  + off_b * stride_vb  + off_hkv * stride_vh
    dk_base = DK + off_b * stride_dkb + off_hkv * stride_dkh
    dv_base = DV + off_b * stride_dvb + off_hkv * stride_dvh

    k = tl.load(k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None], other=0.0)   # [BLOCK_N, D]
    v = tl.load(v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None], other=0.0)   # [BLOCK_N, D]

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    # G is constexpr so tl.static_range unrolls this loop at compile time
    for g in tl.static_range(G):
        off_hq  = off_hkv * G + g
        q_base  = Q     + off_b * stride_qb  + off_hq * stride_qh
        do_base = DO    + off_b * stride_dob + off_hq * stride_doh
        lse_ptr = LSE   + off_b * stride_lb  + off_hq * stride_lh
        d_ptr   = Delta + off_b * stride_db  + off_hq * stride_dh

        lo = ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M    # causal: skip query blocks fully below the diagonal
        for start_m in range(lo, S, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < S
            causal = offs_m[:, None] >= offs_n[None, :]

            q  = tl.load(q_base  + offs_m[:, None] * stride_qs  + offs_d[None, :] * stride_qd,
                         mask=mask_m[:, None], other=0.0)
            do = tl.load(do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                         mask=mask_m[:, None], other=0.0)
            lse = tl.load(lse_ptr + offs_m * stride_ls, mask=mask_m, other=0.0)
            Di  = tl.load(d_ptr   + offs_m * stride_ds, mask=mask_m, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * scale
            qk = tl.where(mask_m[:, None] & mask_n[None, :] & causal, qk, float('-inf'))
            p  = tl.exp(qk - lse[:, None])
            p  = tl.where(mask_m[:, None] & mask_n[None, :] & causal, p, 0.0)  # [BLOCK_M, BLOCK_N]

            dv += tl.dot(tl.trans(p).to(do.dtype), do)                 # [BLOCK_N, D]

            dp = tl.dot(do, tl.trans(v))
            dp = tl.where(mask_m[:, None] & mask_n[None, :], dp, 0.0)
            ds = p * (dp - Di[:, None])
            ds = tl.where(mask_m[:, None] & mask_n[None, :], ds, 0.0)

            dk += tl.dot(tl.trans(ds).to(q.dtype), q) * scale          # [BLOCK_N, D]

    tl.store(dk_base + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd,
             dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_base + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd,
             dv.to(DV.dtype.element_ty), mask=mask_n[:, None])


def triton_gqa_backward(q, k, v, o, lse, do, scale, BLOCK_M=128, BLOCK_N=64):
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    G   = Hq // Hkv
    # delta[b,hq,s] = rowsum(dO * O) — the per-row correction for softmax backward
    delta = (do.float() * o.float()).sum(dim=-1).contiguous()  # [B, Hq, S]

    dq = torch.empty_like(q)
    dk = torch.empty(B, Hkv, S, D, dtype=q.dtype, device=q.device)
    dv = torch.empty(B, Hkv, S, D, dtype=q.dtype, device=q.device)

    grid_dq  = (triton.cdiv(S, BLOCK_M), B * Hq)
    grid_dkv = (triton.cdiv(S, BLOCK_N), B * Hkv)

    _gqa_bwd_dq[grid_dq](
        q, k, v, do, dq, lse, delta,
        *q.stride(), *k.stride(), *v.stride(),
        *do.stride(), *dq.stride(),
        *lse.stride(), *delta.stride(),
        Hq, S, G, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=8, num_stages=3,
    )
    _gqa_bwd_dkv[grid_dkv](
        q, k, v, do, dk, dv, lse, delta,
        *q.stride(), *k.stride(), *v.stride(),
        *do.stride(), *dk.stride(), *dv.stride(),
        *lse.stride(), *delta.stride(),
        Hkv, S, scale,
        G=G, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=8, num_stages=2,
    )
    return dq, dk, dv


def report(name, ref, got):
    diff = (ref.double() - got.double()).abs()
    rel  = diff / ref.double().abs().clamp_min(1e-5)
    print(f"  {name:<40}  max_abs={diff.max().item():.3e}  "
          f"mean_abs={diff.mean().item():.3e}  max_rel={rel.max().item():.3e}")


def save_reference():
    DATA_DIR.mkdir(exist_ok=True)
    torch.manual_seed(42)

    # Generate fp32 in [0,1) to match the CUDA initPtr range, then round to bf16
    # so both sides begin from identical input bits.
    q = torch.rand(B, Hq,  S, D, device="cuda", dtype=torch.float32).bfloat16()
    k = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.float32).bfloat16()
    v = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.float32).bfloat16()
    scale = 1.0 / math.sqrt(D)

    out_bf16 = sdpa_gqa(q, k, v, scale)                  # bf16 reference (matching precision)
    out_fp64, lse_fp64 = fp64_reference(q, k, v, scale)  # fp64 ground truth

    print(f"=== Precision vs FP64 ground truth  ({B}x{Hq}x{S}x{D}, GQA G={G}) ===")
    report("SDPA bf16 (enable_gqa)", out_fp64, out_bf16.float().cpu())

    # Save as float32 (bf16 widened) so the CUDA side reuses loadBin().
    q.float().cpu().numpy().tofile(DATA_DIR / "gqa_q.bin")
    k.float().cpu().numpy().tofile(DATA_DIR / "gqa_k.bin")
    v.float().cpu().numpy().tofile(DATA_DIR / "gqa_v.bin")
    out_bf16.float().cpu().numpy().tofile(DATA_DIR / "gqa_o.bin")
    out_fp64.cpu().numpy().tofile(DATA_DIR / "gqa_o_fp64.bin")
    lse_fp64.cpu().numpy().tofile(DATA_DIR / "gqa_lse.bin")
    print(f"Saved forward reference data (B={B}, Hq={Hq}, Hkv={Hkv}, S={S}, D={D}) to {DATA_DIR}/")

    # ── Backward reference ────────────────────────────────────────────────────
    # Use a fixed random dO that matches the output shape.
    torch.manual_seed(43)
    do = torch.randn(B, Hq, S, D, device="cuda", dtype=torch.bfloat16)

    # fp64 ground truth
    dq_fp64, dk_fp64, dv_fp64 = fp64_backward(q, k, v, do, scale)

    # bf16 reference via PyTorch autograd
    q_ad = q.detach().clone().requires_grad_(True)
    k_ad = k.detach().clone().requires_grad_(True)
    v_ad = v.detach().clone().requires_grad_(True)
    sdpa_gqa(q_ad, k_ad, v_ad, scale).backward(do)
    dq_bf16 = q_ad.grad.float().cpu()
    dk_bf16 = k_ad.grad.float().cpu()
    dv_bf16 = v_ad.grad.float().cpu()

    print(f"\n=== Backward precision vs FP64 ground truth ===")
    report("dQ SDPA bf16", dq_fp64, dq_bf16)
    report("dK SDPA bf16", dk_fp64, dk_bf16)
    report("dV SDPA bf16", dv_fp64, dv_bf16)

    do.float().cpu().numpy().tofile(DATA_DIR / "gqa_do.bin")
    dq_fp64.numpy().tofile(DATA_DIR / "gqa_dq_fp64.bin")
    dk_fp64.numpy().tofile(DATA_DIR / "gqa_dk_fp64.bin")
    dv_fp64.numpy().tofile(DATA_DIR / "gqa_dv_fp64.bin")
    dq_bf16.numpy().tofile(DATA_DIR / "gqa_dq.bin")
    dk_bf16.numpy().tofile(DATA_DIR / "gqa_dk.bin")
    dv_bf16.numpy().tofile(DATA_DIR / "gqa_dv.bin")
    print(f"Saved backward reference data to {DATA_DIR}/\n")


def main():
    torch.manual_seed(42)
    q = torch.rand(B, Hq,  S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(D)

    # compile so Inductor can fuse/pick the best backend
    compiled = torch.compile(lambda q, k, v: sdpa_gqa(q, k, v, scale))
    _ = compiled(q, k, v)
    torch.cuda.synchronize()

    # ── precision: Triton vs PyTorch bf16 SDPA (both bf16, fp32 accumulation) ──
    out_sdpa = sdpa_gqa(q, k, v, scale)
    out_tri  = triton_gqa(q, k, v, scale)
    diff = (out_sdpa.float() - out_tri.float()).abs()
    print(f"Triton vs SDPA bf16   max_abs={diff.max().item():.3e}  "
          f"mean_abs={diff.mean().item():.3e}\n")

    print(f"=== Latency  ({B}x{Hq}x{S}x{D}, GQA G={G}, bf16) ===")
    fns = {
        "SDPA bf16 (enable_gqa)": lambda: sdpa_gqa(q, k, v, scale),
        "SDPA bf16 (compiled)  ": lambda: compiled(q, k, v),
        "Triton flash (GQA)    ": lambda: triton_gqa(q, k, v, scale),
    }
    # attention FLOPs (algorithmic): 4 * B * Hq * S * S * D  (QKᵀ + P·V, factor 2 for MAC)
    flops = 4 * B * Hq * S * S * D
    header = f"{'Kernel':<26} {'median ms':>10}  {'p5 ms':>8}  {'p95 ms':>8}  {'TFLOP/s':>10}"
    print(header)
    print("-" * len(header))
    for name, fn in fns.items():
        med, p5, p95 = triton.testing.do_bench(fn, warmup=25, rep=100,
                                                quantiles=[0.5, 0.05, 0.95])
        tflops = flops / (med * 1e-3) / 1e12
        print(f"{name:<26} {med:>10.4f}  {p5:>8.4f}  {p95:>8.4f}  {tflops:>10.2f}")

    # ── backward ─────────────────────────────────────────────────────────────
    do = torch.randn_like(q)

    # reference O and LSE come from PyTorch — no dependence on our Triton forward
    o_ref   = sdpa_gqa(q, k, v, scale).detach()
    lse_ref = sdpa_lse(q, k, scale)
    torch.cuda.synchronize()

    # precision: Triton flash bwd vs PyTorch SDPA autograd (both bf16)
    dq_tri, dk_tri, dv_tri = triton_gqa_backward(q, k, v, o_ref, lse_ref, do, scale)
    q_ad = q.detach().requires_grad_(True)
    k_ad = k.detach().requires_grad_(True)
    v_ad = v.detach().requires_grad_(True)
    sdpa_gqa(q_ad, k_ad, v_ad, scale).backward(do)

    print(f"\n=== Backward precision (Triton flash vs SDPA bf16) ===")
    for name, tri, ref in [
        ("dQ", dq_tri, q_ad.grad),
        ("dK", dk_tri, k_ad.grad),
        ("dV", dv_tri, v_ad.grad),
    ]:
        report(f"{name} Triton flash vs SDPA", ref.float().cpu(), tri.float().cpu())

    def _sdpa_bwd():
        qb = q.detach().requires_grad_(True)
        kb = k.detach().requires_grad_(True)
        vb = v.detach().requires_grad_(True)
        sdpa_gqa(qb, kb, vb, scale).backward(do)

    def _sdpa_compiled_bwd():
        qb = q.detach().requires_grad_(True)
        kb = k.detach().requires_grad_(True)
        vb = v.detach().requires_grad_(True)
        compiled(qb, kb, vb).backward(do)

    def _triton_bwd():
        triton_gqa_backward(q, k, v, o_ref, lse_ref, do, scale)

    print(f"\n=== Backward latency  ({B}x{Hq}x{S}x{D}, GQA G={G}, bf16) ===")
    bwd_fns = {
        "SDPA bwd (enable_gqa) ": _sdpa_bwd,
        "SDPA bwd (compiled)   ": _sdpa_compiled_bwd,
        "Triton flash bwd (GQA)": _triton_bwd,
    }
    # rough backward FLOPs: dQ + dK + dV + P recompute ≈ 4× forward
    bwd_flops = 4 * flops
    print(header)
    print("-" * len(header))
    for name, fn in bwd_fns.items():
        med, p5, p95 = triton.testing.do_bench(fn, warmup=25, rep=100,
                                                quantiles=[0.5, 0.05, 0.95])
        tflops = bwd_flops / (med * 1e-3) / 1e12
        print(f"{name:<26} {med:>10.4f}  {p5:>8.4f}  {p95:>8.4f}  {tflops:>10.2f}")


if __name__ == "__main__":
    save_reference()
    main()
