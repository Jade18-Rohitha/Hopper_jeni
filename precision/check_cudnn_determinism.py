#!/usr/bin/env python3
"""Empirical determinism probe for the cuDNN / PyTorch-SDPA GQA backward.

Runs the SAME backward (identical inputs + identical grad_output) N times and
checks whether dQ/dK/dV come back bit-for-bit identical across runs. If any run
differs, the SDPA backward is non-deterministic — i.e. cuDNN pays the same
run-to-run-nondeterminism trade our TMA-reduce kernel does.

Config mirrors precision/baseline_gqa.py: enable_gqa, is_causal, bf16, scale=1/sqrt(D).
Usage:  python precision/check_cudnn_determinism.py [B] [Hq] [Hkv] [N]
"""
import sys, torch
import torch.nn.functional as F

B   = int(sys.argv[1]) if len(sys.argv) > 1 else 4
Hq  = int(sys.argv[2]) if len(sys.argv) > 2 else 24
Hkv = int(sys.argv[3]) if len(sys.argv) > 3 else 8
N   = int(sys.argv[4]) if len(sys.argv) > 4 else 8
S, D = 4096, 128
scale = 1.0 / (D ** 0.5)

torch.manual_seed(0)
# fixed inputs + fixed upstream gradient — reused verbatim every run
q0 = torch.rand(B, Hq,  S, D, device="cuda", dtype=torch.bfloat16)
k0 = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
v0 = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
do = torch.randn(B, Hq,  S, D, device="cuda", dtype=torch.bfloat16)

def one_backward():
    q = q0.detach().clone().requires_grad_(True)
    k = k0.detach().clone().requires_grad_(True)
    v = v0.detach().clone().requires_grad_(True)
    out = F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True, is_causal=True)
    out.backward(do)
    return q.grad.clone(), k.grad.clone(), v.grad.clone()

print(f"shape B={B} Hq={Hq} Hkv={Hkv} S={S} D={D} bf16 causal | {N} runs, identical inputs\n")
ref = one_backward()
names = ("dQ", "dK", "dV")
any_diff = False
for r in range(1, N):
    cur = one_backward()
    for nm, a, b in zip(names, ref, cur):
        exact = torch.equal(a, b)
        md = (a.float() - b.float()).abs().max().item()
        if not exact:
            any_diff = True
            print(f"run {r}: {nm} DIFFERS from run 0   max|Δ|={md:.3e}")
        else:
            print(f"run {r}: {nm} bit-identical")
    print()

print("="*60)
if any_diff:
    print("VERDICT: cuDNN/SDPA backward is NON-DETERMINISTIC (bit differs run-to-run).")
    print("         Same trade our TMA-reduce kernel makes — confirmed empirically.")
else:
    print("VERDICT: cuDNN/SDPA backward was BIT-REPRODUCIBLE across all runs here.")
    print("         (Could still vary across launch configs — but no wobble observed.)")
