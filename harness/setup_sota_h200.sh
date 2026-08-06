#!/usr/bin/env bash
# =====================================================================================
# setup_sota_h200.sh -- install/build the SOTA attention competitors that
# base/harness/run_gqa_bench.py's competitors.py knows how to find, on an
# H200 (sm_90a) instance.
#
# Installs, independently and best-effort (one component failing never blocks
# the others -- same "self-detect, skip with a reason" philosophy as the
# harness itself):
#
#   1. FlashInfer       -- pip package `flashinfer-python`, import name `flashinfer`.
#                           Serving kernels; fills the "flashinfer" row.
#   2. FlashAttention-4 -- pip package `flash-attn-4` (CuTeDSL, beta). Docs say
#                           "optimized for Hopper and Blackwell" -- applies to H200.
#                           Imports as `flash_attn.cute`. Fills the "fa4" row.
#   3. FlashAttention-3 -- built from source (Dao-AILab/flash-attention, hopper/
#                           subfolder). Hopper-only beta, needs CUDA toolkit >= 12.3
#                           (12.8 recommended) present locally to compile. Imports
#                           as `flash_attn_interface`. ALSO fills the "fa4" row.
#
# NOTE on the fa3/fa4 collision: competitors.py's FA4._fn() search order is
# ("flash_attn.cute.interface", "flash_attn.cute", "flash_attn_interface",
# "flash_attn") -- it returns the FIRST one it finds. If both FA4 and FA3 end
# up installed, the "fa4" row in run_gqa_bench.py's table will run FA4 and you
# will never see an FA3 number. If you want both as separate rows, add a
# distinct `FA3` competitor class to competitors.py (same _fn() pattern,
# searching only "flash_attn_interface") -- not done here since it's a
# harness-code change, not a setup-script job.
#
# Safe to re-run: each step skips work that's already done.
#
# Usage:
#   bash base/harness/setup_sota_h200.sh              # install everything
#   bash base/harness/setup_sota_h200.sh flashinfer    # just one component
#   bash base/harness/setup_sota_h200.sh fa4 fa3
#
# Env overrides:
#   FA_REPO_DIR   where to clone flash-attention for the FA3 build (default: $HOME/flash-attention)
#   MAX_JOBS      parallel nvcc jobs for the FA3 build (default: auto, see below)
#   NVCC_THREADS  threads per nvcc job (default: 2, matches upstream default)
# =====================================================================================
set -uo pipefail   # deliberately NOT -e: one failing component must not kill the rest

FA_REPO_DIR="${FA_REPO_DIR:-$HOME/flash-attention}"
NVCC_THREADS="${NVCC_THREADS:-2}"

STATUS_FLASHINFER="not attempted"
STATUS_FA4="not attempted"
STATUS_FA3="not attempted"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$1"; }
fail() { printf '\033[1;31mXX %s\033[0m\n' "$1"; }
ok()   { printf '\033[1;32mOK %s\033[0m\n' "$1"; }

# ------------------------------------------------------------------------------------
log "Environment check"
python3 - <<'PY'
import sys, torch
cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
print(f"python:        {sys.version.split()[0]}")
print(f"torch:         {torch.__version__}")
print(f"torch cuda:    {torch.version.cuda}")
print(f"cuda avail:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device:        {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")
    if cc[0] != 9:
        print("WARNING: this is not an sm_90 (Hopper) device -- FA3 build below will not target it.")
PY
if command -v nvcc >/dev/null 2>&1; then
    NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "nvcc:          $NVCC_VER ($(command -v nvcc))"
else
    warn "nvcc not found in PATH -- the FA3 source build below WILL fail without it."
    warn "Install the CUDA toolkit (not just the driver) first, e.g. via your instance's"
    warn "base image, or 'conda install -c nvidia cuda-nvcc' / the NVIDIA CUDA toolkit installer."
fi
NPROC=$(nproc 2>/dev/null || echo 4)
FREE_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
FREE_GB="${FREE_GB:-16}"
# Each parallel nvcc job compiling a CUTLASS-heavy .cu file can eat several GB of RAM --
# this is the single most common flash-attn build failure report (OOM, not a code bug).
DEFAULT_MAX_JOBS=$(( FREE_GB / 4 ))
[ "$DEFAULT_MAX_JOBS" -lt 1 ] && DEFAULT_MAX_JOBS=1
[ "$DEFAULT_MAX_JOBS" -gt "$NPROC" ] && DEFAULT_MAX_JOBS=$NPROC
MAX_JOBS="${MAX_JOBS:-$DEFAULT_MAX_JOBS}"
echo "cores:         $NPROC, ~${FREE_GB}GB RAM -> MAX_JOBS=$MAX_JOBS (override with MAX_JOBS=n)"

WANT=("$@")
want() { [ "${#WANT[@]}" -eq 0 ] && return 0; for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

# ------------------------------------------------------------------------------------
if want flashinfer; then
log "1/3 FlashInfer (pip install flashinfer-python)"
if python3 -c "import flashinfer" 2>/dev/null; then
    ok "flashinfer already importable -- skipping install"
    STATUS_FLASHINFER="already installed"
else
    if pip install -U flashinfer-python; then
        if python3 -c "import flashinfer" 2>/dev/null; then
            ok "flashinfer installed and importable"
            STATUS_FLASHINFER="installed"
        else
            fail "pip install succeeded but 'import flashinfer' still fails -- see traceback above"
            STATUS_FLASHINFER="installed but import failed"
        fi
    else
        fail "pip install flashinfer-python failed -- see output above"
        STATUS_FLASHINFER="pip install failed"
    fi
fi
fi

# ------------------------------------------------------------------------------------
if want fa4; then
log "2/3 FlashAttention-4 / CuTeDSL (pip install flash-attn-4)"
if python3 -c "from flash_attn.cute import flash_attn_func" 2>/dev/null; then
    ok "flash_attn.cute already importable -- skipping install"
    STATUS_FA4="already installed"
else
    CUDA_MAJOR=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0] if torch.version.cuda else 0)" 2>/dev/null || echo 0)
    PKG="flash-attn-4"
    if [ "$CUDA_MAJOR" = "13" ]; then
        PKG="flash-attn-4[cu13]"
        echo "torch reports CUDA 13.x -- installing the cu13 extra for best performance"
    fi
    # This is still a beta (4.0.0bNN) with a fairly exotic dependency chain
    # (nvidia-cutlass-dsl, apache-tvm-ffi, quack-kernels) -- failure here is
    # informative, not fatal to the rest of this script.
    if pip install -U "$PKG"; then
        if python3 -c "from flash_attn.cute import flash_attn_func" 2>/dev/null; then
            ok "flash_attn.cute installed and importable"
            STATUS_FA4="installed"
        else
            fail "pip install succeeded but 'from flash_attn.cute import flash_attn_func' still fails"
            STATUS_FA4="installed but import failed"
        fi
    else
        fail "pip install $PKG failed -- see output above"
        STATUS_FA4="pip install failed"
    fi
fi
fi

# ------------------------------------------------------------------------------------
if want fa3; then
log "3/3 FlashAttention-3 / Hopper beta (build from source)"
if python3 -c "from flash_attn_3 import flash_attn_interface" 2>/dev/null || \
   python3 -c "import flash_attn_interface" 2>/dev/null; then
    ok "flash_attn_interface already importable -- skipping build"
    STATUS_FA3="already installed"
elif ! command -v nvcc >/dev/null 2>&1; then
    fail "nvcc not found -- cannot build FA3 from source. Install the CUDA toolkit first."
    STATUS_FA3="skipped (no nvcc)"
else
    if [ -d "$FA_REPO_DIR/.git" ]; then
        echo "updating existing clone at $FA_REPO_DIR"
        git -C "$FA_REPO_DIR" fetch --depth 1 origin main && \
        git -C "$FA_REPO_DIR" checkout main && \
        git -C "$FA_REPO_DIR" submodule update --init --recursive
    else
        echo "cloning Dao-AILab/flash-attention (with the csrc/cutlass submodule) into $FA_REPO_DIR"
        git clone --recursive https://github.com/Dao-AILab/flash-attention.git "$FA_REPO_DIR"
    fi
    CLONE_OK=$?
    if [ "$CLONE_OK" -ne 0 ] || [ ! -d "$FA_REPO_DIR/hopper" ]; then
        fail "clone/update of flash-attention failed -- see output above"
        STATUS_FA3="git clone failed"
    else
        echo "building hopper/ (targets sm_90a only, CUDA >= 12.3 required, 12.8 recommended)"
        echo "MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS -- this compiles CUTLASS templates"
        echo "and can take a long time (15-45+ min) and needs real RAM; lower MAX_JOBS if it OOMs."
        (
            cd "$FA_REPO_DIR/hopper" && \
            MAX_JOBS="$MAX_JOBS" NVCC_THREADS="$NVCC_THREADS" python3 setup.py install
        )
        if [ $? -eq 0 ] && { python3 -c "from flash_attn_3 import flash_attn_interface" 2>/dev/null || \
                              python3 -c "import flash_attn_interface" 2>/dev/null; }; then
            ok "FlashAttention-3 built and importable"
            STATUS_FA3="installed"
        else
            fail "FA3 build failed or the result isn't importable -- see output above"
            STATUS_FA3="build failed"
        fi
    fi
fi
fi

# ------------------------------------------------------------------------------------
log "Summary"
printf '  %-14s %s\n' "flashinfer:" "$STATUS_FLASHINFER"
printf '  %-14s %s\n' "fa4 (cute):" "$STATUS_FA4"
printf '  %-14s %s\n' "fa3 (hopper):" "$STATUS_FA3"
echo
echo "Re-run your benchmark now:"
echo "  python3 base/harness/run_gqa_bench.py"
echo
echo "If a component shows an 'import failed' or 'build failed' status, scroll up to the"
echo "matching numbered section above for the actual pip/build error -- this script never"
echo "swallows those, it only keeps going so one bad component doesn't block the others."
