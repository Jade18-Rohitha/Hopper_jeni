#!/usr/bin/env bash
# ============================================================================
# Vz2 sweep: run the dev binary (gqa_bwd) — which benchmarks Vz2 (2.82ms champ)
# alongside V44/Vp1/Vr1/V45 — across the 12-shape grid at LOCKED clocks.
# gqa_bwd now takes the shape from argv (B Hq Hkv); reference is regenerated
# per shape by baseline_gqa.py.  Grep the log for "Vz2" to read its time/shape.
#   usage:  bash scripts/run_vz2_sweep.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # -> Hopper/

CONFIGS=("12 4" "16 4" "24 8" "32 8")   # "Hq Hkv"
BATCHES=(2 4 8)
LOG="reports/vz2_sweep_$(date +%Y%m%d_%H%M%S).log"

echo "== building gqa_bwd =="
cmake --build build --target gqa_bwd -j

echo "== locking clocks (pm on, gpu clock 1980 MHz) =="
sudo nvidia-smi -pm 1        >/dev/null
sudo nvidia-smi -lgc 1980    >/dev/null

run_all() {
  for cfg in "${CONFIGS[@]}"; do
    read -r HQ HKV <<< "$cfg"
    for B in "${BATCHES[@]}"; do
      echo ""
      echo "############################################################"
      echo "# SHAPE  B=${B}  Hq=${HQ}  Hkv=${HKV}  (G=$((HQ/HKV)))  S=4096 D=128"
      echo "############################################################"
      python3 precision/baseline_gqa.py "$B" "$HQ" "$HKV"
      ./build/bin/gqa_bwd "$B" "$HQ" "$HKV"
    done
  done
}

run_all 2>&1 | tee "$LOG"

echo "== resetting clocks =="
sudo nvidia-smi -rgc >/dev/null || true

echo ""
echo "== sweep complete -> $LOG =="
echo "== Vz2 lines: =="
grep -A6 'Vz2 wgmma' "$LOG" | grep -E 'SHAPE|Median|Vz2' || true
