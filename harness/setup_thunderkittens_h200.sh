#!/usr/bin/env bash
# =====================================================================================
# setup_thunderkittens_h200.sh -- build HazyResearch/ThunderKittens' GQA-capable
# attention kernel (kernels/attention/mha_h100) for base/harness/run_gqa_bench.py's
# "thunderkittens" competitor, on an H200 (sm_90a) instance.
#
# TK is NOT a pip package. It's header-only C++/CUDA; each kernel is its own
# PyTorch extension built via a per-kernel Makefile. mha_h100.cu genuinely
# supports GQA (kv_head_idx = blockIdx.y / (qo_heads/kv_heads)), D in {64,128},
# causal and non-causal -- verified directly against the source, not just the
# paper's claim, before this script was written.
#
# No CUTLASS/git-submodule dependency (unlike the FA3 build) -- a shallow
# clone is enough, so this is much faster than setup_sota_h200.sh's FA3 step.
#
# ***Read this before trusting the "thunderkittens" row in a big sweep***:
# TK's CUDA error checker (include/common/util.cuh, CHECK_CUDA_ERROR) calls
# std::exit(EXIT_FAILURE) directly on ANY cudaGetLastError() failure -- not a
# catchable Python exception. Reproduced locally: run_gqa_bench.py's
# try/except around a competitor's forward()/make() CANNOT catch this: it
# kills the entire benchmark process mid-sweep, not just this one row. This
# script's own verification step below runs its smoke test in a SEPARATE
# subprocess for exactly this reason -- copy that pattern if you sweep many
# shapes and want one bad TK shape to not take down the whole run.
#
# Safe to re-run: skips the clone/build if already present.
#
# Usage:
#   bash base/harness/setup_thunderkittens_h200.sh
#
# Env overrides:
#   THUNDERKITTENS_ROOT  where to clone/build TK (default: $HOME/ThunderKittens)
#                        -- competitors.py's ThunderKittens class reads this
#                        SAME env var, so set it once and both agree.
# =====================================================================================
set -uo pipefail

TK_ROOT="${THUNDERKITTENS_ROOT:-$HOME/ThunderKittens}"
KERNEL_DIR="$TK_ROOT/kernels/attention/mha_h100"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$1"; }
fail() { printf '\033[1;31mXX %s\033[0m\n' "$1"; }
ok()   { printf '\033[1;32mOK %s\033[0m\n' "$1"; }

# ------------------------------------------------------------------------------------
log "Environment check"
python3 - <<'PY'
import torch
cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
print(f"torch:      {torch.__version__}")
print(f"cuda avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device:     {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")
    if cc[0] != 9:
        print("WARNING: not sm_90 (Hopper) -- the build below will succeed (nvcc cross-"
              "compiles fine) but the kernel will refuse to LAUNCH on this GPU.")
PY
if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc: $(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+') ($(command -v nvcc))"
else
    fail "nvcc not found in PATH -- cannot build. Install the CUDA toolkit first."
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    fail "git not found -- cannot clone the repo."
    exit 1
fi

# ------------------------------------------------------------------------------------
log "pybind11 (needed by the kernel's Makefile to find Python/pybind11 headers)"
if python3 -c "import pybind11" 2>/dev/null; then
    ok "pybind11 already importable"
else
    pip install -U pybind11 || { fail "pip install pybind11 failed"; exit 1; }
fi

# ------------------------------------------------------------------------------------
log "Clone HazyResearch/ThunderKittens (no submodules needed, unlike FA3)"
if [ -d "$TK_ROOT/.git" ]; then
    echo "already cloned at $TK_ROOT -- pulling latest"
    git -C "$TK_ROOT" pull --ff-only || warn "pull failed/diverged -- using the existing checkout as-is"
else
    git clone --depth 1 https://github.com/HazyResearch/ThunderKittens.git "$TK_ROOT" || {
        fail "clone failed -- see output above"; exit 1; }
fi
if [ ! -d "$KERNEL_DIR" ]; then
    fail "kernels/attention/mha_h100 not found under $TK_ROOT -- repo layout may have changed"
    exit 1
fi

# ------------------------------------------------------------------------------------
log "Building kernels/attention/mha_h100 (targets sm_90a; single .cu file, much faster"
echo "than the FA3 source build -- typically a few minutes, not tens)"
(cd "$KERNEL_DIR" && make) 2>&1 | tail -40
SO=$(ls "$KERNEL_DIR"/_C*.so 2>/dev/null | head -1)
if [ -z "$SO" ]; then
    fail "build finished but no _C*.so was produced -- scroll up for the actual nvcc/link error"
    exit 1
fi
ok "built: $SO"

# ------------------------------------------------------------------------------------
log "Verifying (subprocess-isolated: see the exit() warning above)"
THUNDERKITTENS_ROOT="$TK_ROOT" python3 - <<'PY'
import os, subprocess, sys

code = r"""
import os, sys, math, torch
sys.path.insert(0, os.path.join(os.environ["THUNDERKITTENS_ROOT"], "kernels", "attention", "mha_h100"))
import _C as tk

# Real GQA shape (Hq != Hkv), D=128, causal -- exactly what run_gqa_bench.py exercises.
B, Hq, Hkv, S, D = 2, 8, 4, 256, 128
torch.manual_seed(0)
q = torch.randn(B, Hq, S, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16, device="cuda")

o, l_vec = tk.mha_forward(q, k, v, True)

ref = torch.nn.functional.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), scale=1.0/math.sqrt(D),
    is_causal=True, enable_gqa=True)
diff = (o.float() - ref).abs()
print(f"max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}")
assert diff.max().item() < 0.1, "output too far from SDPA reference"
print("PASS")
"""
env = dict(os.environ)
r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode != 0:
    print("--- stderr (last 10 lines) ---")
    print("\n".join(r.stderr.strip().splitlines()[-10:]))
    print(f"--- child exit code: {r.returncode} ---")
    sys.exit(1)
PY
VERIFY_STATUS=$?

# ------------------------------------------------------------------------------------
log "Summary"
if [ "$VERIFY_STATUS" -eq 0 ]; then
    ok "ThunderKittens mha_h100 built and verified against SDPA on a real GQA shape"
    echo
    echo "Re-run your benchmark now:"
    echo "  python3 base/harness/run_gqa_bench.py --competitors thunderkittens"
    echo "(add it into a bigger --competitors list only after that passes on its own --"
    echo " see the exit() warning at the top of this script and in competitors.py)"
else
    fail "Verification failed or the kernel couldn't launch here -- see output above."
    echo "If this ISN'T an H200/sm_90 GPU, that's expected: the .so is built for sm_90a"
    echo "only and will refuse to launch anywhere else (confirmed locally: it fails with"
    echo "'no kernel image is available for execution on the device', not a build bug)."
fi
