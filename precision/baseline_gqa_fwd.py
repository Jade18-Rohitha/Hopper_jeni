"""
baseline_gqa_fwd.py — Generate forward-pass reference data for GQA_fwd_v62.cu.

Saves (all float32 .bin so the CUDA side reuses loadBin; values are bf16-rounded
inputs widened back to float32, so kernel and reference start from identical bits):
  - gqa_fwd_q.bin              [B, Hq,  S, D]
  - gqa_fwd_k.bin / _v.bin     [B, Hkv, S, D]
  - gqa_fwd_o.bin              fp32 SDPA output  [B, Hq, S, D]
  - gqa_fwd_lse.bin            fp32 log-sum-exp  [B, Hq, S]

Usage:
  python precision/baseline_gqa_fwd.py          # default B=2
  python precision/baseline_gqa_fwd.py --B 4    # for B=4
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def fp64_reference(q, k, v, scale, B, Hq, Hkv, S, D, G):
    """Double-precision ground truth (O + LSE), one head at a time."""
    out = torch.empty(B, Hq, S, D, dtype=torch.float32)
    lse = torch.empty(B, Hq, S,    dtype=torch.float32)
    for b in range(B):
        for hq in range(Hq):
            hkv = hq // G
            q64 = q[b, hq ].double()
            k64 = k[b, hkv].double()
            v64 = v[b, hkv].double()
            scores = (q64 @ k64.transpose(-2, -1)) * scale
            causal_mask = torch.ones(S, S, device=scores.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal_mask, float('-inf'))
            lse[b, hq] = torch.logsumexp(scores, dim=-1).float().cpu()
            w = torch.softmax(scores, dim=-1)
            out[b, hq] = (w @ v64).float().cpu()
    return out, lse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2, help="batch size (default 2)")
    args = ap.parse_args()

    B = args.B
    Hq, Hkv, S, D = 12, 4, 4096, 128
    G = Hq // Hkv
    scale = 1.0 / math.sqrt(D)

    DATA_DIR = Path(__file__).parent.parent / "data"
    DATA_DIR.mkdir(exist_ok=True)

    torch.manual_seed(42)
    q = torch.rand(B, Hq,  S, D, device="cuda", dtype=torch.float32).bfloat16()
    k = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.float32).bfloat16()
    v = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.float32).bfloat16()

    print(f"Computing fp64 forward reference  B={B} Hq={Hq} Hkv={Hkv} S={S} D={D} ...")
    out_fp64, lse_fp64 = fp64_reference(q, k, v, scale, B, Hq, Hkv, S, D, G)

    # Also compute SDPA bf16 for a sanity comparison
    out_sdpa = F.scaled_dot_product_attention(
        q, k, v, scale=scale, enable_gqa=True, is_causal=True
    ).float().cpu()

    diff = (out_fp64 - out_sdpa).abs()
    print(f"  SDPA bf16 vs fp64:  max_abs={diff.max().item():.3e}  "
          f"mean_abs={diff.mean().item():.3e}")

    # Save as float32
    q.float().cpu().numpy().tofile(DATA_DIR / "gqa_fwd_q.bin")
    k.float().cpu().numpy().tofile(DATA_DIR / "gqa_fwd_k.bin")
    v.float().cpu().numpy().tofile(DATA_DIR / "gqa_fwd_v.bin")
    out_fp64.cpu().numpy().tofile(DATA_DIR / "gqa_fwd_o.bin")
    lse_fp64.cpu().numpy().tofile(DATA_DIR / "gqa_fwd_lse.bin")

    print(f"Saved forward reference data (B={B}) to {DATA_DIR}/")
    print(f"  gqa_fwd_q.bin    {q.numel()} floats")
    print(f"  gqa_fwd_k.bin    {k.numel()} floats")
    print(f"  gqa_fwd_v.bin    {v.numel()} floats")
    print(f"  gqa_fwd_o.bin    {out_fp64.numel()} floats")
    print(f"  gqa_fwd_lse.bin  {lse_fp64.numel()} floats")


if __name__ == "__main__":
    main()
