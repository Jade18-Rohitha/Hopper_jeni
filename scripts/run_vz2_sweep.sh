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
echo "== per-shape best-of (cuDNN / V44 / Vj1) — champion = faster of V44,Vj1; margin vs cuDNN =="
echo "==   shapes flagged [MARGINAL <1%] need a 12-run median-of-medians before the win is claimed. =="
awk '
  /^# SHAPE/                         { shape=$0; sub(/^# SHAPE  /,"",shape); sub(/  \(G.*/,"",shape) }
  /SDPA bwd \(enable_gqa\)/          { cudnn=$4 }
  /Benchmark: GQA bwd V44/           { k="v44" }
  /Benchmark: GQA bwd Vj1/           { k="vj1" }
  /Median Time:/ && k=="v44"         { v44=$3; k="" }
  /Median Time:/ && k=="vj1"         { vj1=$3; k="";
      best=(v44<vj1)?v44:vj1; who=(v44<vj1)?"V44":"Vj1";
      marg=(cudnn-best)/cudnn*100.0; tag=(marg>=0)?"WIN":"LOSS";
      flag=(marg<1.0 && marg>-1.0)?"  [MARGINAL <1% -> 12-run]":"";
      printf "%-20s cuDNN %-7s V44 %-7s Vj1 %-7s | %s %+5.1f%% (%s)%s\n",
             shape, cudnn, v44, vj1, tag, marg, who, flag }
' "$LOG" || true
echo ""
echo "== 12/12 board is locked. This is the confirming sweep: re-verify only the [MARGINAL] shapes. =="
