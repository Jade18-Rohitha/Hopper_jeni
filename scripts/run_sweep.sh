#!/usr/bin/env bash
# ============================================================================
# GQA-bwd baseline sweep: for each (Hq,Hkv) x B, generate the FP64/SDPA/Triton
# reference (precision/baseline_gqa.py) then run the baseline binary, which
# benchmarks BOTH kernels — V44 (default grid) and V45 (k_tile-fastest raster)
# — and checks each vs the reference. Take the min(V44,V45) per shape.
#
# Reference data (data/gqa_*.bin) is per-shape and overwritten each run, so we
# pair py+binary per (B,Hq,Hkv). Both take the shape from argv, so no rebuild
# per config — the binary is built once.
#
#   usage:  bash scripts/run_sweep.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # -> Hopper/

CONFIGS=("12 4" "16 4" "24 8" "32 8")   # "Hq Hkv"
BATCHES=(2 4 8)
LOG="reports/baseline_sweep_$(date +%Y%m%d_%H%M%S).log"

echo "== building gqa_bwd_baseline =="
cmake --build build --target gqa_bwd_baseline -j

run_all() {
  for cfg in "${CONFIGS[@]}"; do
    read -r HQ HKV <<< "$cfg"
    for B in "${BATCHES[@]}"; do
      echo ""
      echo "############################################################"
      echo "# SHAPE  B=${B}  Hq=${HQ}  Hkv=${HKV}  (G=$((HQ/HKV)))  S=4096 D=128"
      echo "############################################################"
      python precision/baseline_gqa.py "$B" "$HQ" "$HKV"
      ./build/bin/gqa_bwd_baseline "$B" "$HQ" "$HKV"
    done
  done
}

run_all 2>&1 | tee "$LOG"
echo ""
echo "== sweep complete -> $LOG =="
