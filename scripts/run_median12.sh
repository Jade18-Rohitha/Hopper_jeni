#!/usr/bin/env bash
# ============================================================================
# Apples-to-apples vs cuDNN, N-run median-of-medians (default N=12), locked clock.
#
#   OURS  (build/bin/gqa_fwd_bwd):  V62 fwd | Vj1d, Vj1, Vj2 bwd | V62+Vj1d/Vj1/Vj2 chained fwd+bwd
#   cuDNN (precision/cudnn_time.py): fwd    | bwd-only | sdpa(...).backward(dO)
#
# Both sides: SAME inputs (data/gqa_*.bin), triton.do_bench-style timing (L2 flush per
# rep, async queue, one sync), backend forced to cuDNN. Runs are INTERLEAVED (ours,
# cuDNN, ours, cuDNN, ...) so thermal drift hits both equally.
#
#   usage:  bash scripts/run_median12.sh            # all 12 shapes, N=12, on GPU 1
#           N=5 SHAPES="8 12 4;4 24 8" GPU=1 bash scripts/run_median12.sh
#   GPU = nvidia-smi index of the ONE GPU to use (PCI order); only that GPU's clocks are touched.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # -> Hopper/

N=${N:-12}
GPU=${GPU:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
SHAPES=${SHAPES:-"2 12 4;4 12 4;8 12 4;2 16 4;4 16 4;8 16 4;2 24 8;4 24 8;8 24 8;2 32 8;4 32 8;8 32 8"}
LOG="reports/median${N}_$(date +%Y%m%d_%H%M%S).log"

echo "== building gqa_fwd_bwd =="
cmake --build build --target gqa_fwd_bwd -j

echo "== GPU $GPU: locking clocks (pm on, gpu clock 1980 MHz); other GPUs untouched =="
sudo nvidia-smi -i "$GPU" -pm 1     >/dev/null
sudo nvidia-smi -i "$GPU" -lgc 1980 >/dev/null
trap 'sudo nvidia-smi -i "$GPU" -rgc >/dev/null || true' EXIT
nvidia-smi -i "$GPU" --query-gpu=index,name,pci.bus_id,clocks.sm,clocks.max.sm,power.draw --format=csv

IFS=';' read -ra SH <<< "$SHAPES"
{
for s in "${SH[@]}"; do
  read -r B HQ HKV <<< "$s"
  echo "############ SHAPE B=$B Hq=$HQ Hkv=$HKV"
  python3 precision/baseline_gqa.py "$B" "$HQ" "$HKV" > /dev/null   # reference data/ for this shape
  ./build/bin/gqa_fwd_bwd "$B" "$HQ" "$HKV" | grep -aE "CHECK|don't match|Ref:"   # correctness once
  for r in $(seq 1 "$N"); do
    echo "## run $r"
    ./build/bin/gqa_fwd_bwd "$B" "$HQ" "$HKV" | grep -E "^OURS"
    python3 precision/cudnn_time.py "$B" "$HQ" "$HKV" | grep -E "^cuDNN |^# "
  done
done
} 2>&1 | tee "$LOG"

echo ""
python3 scripts/median_summary.py "$LOG"
echo "== log: $LOG =="
