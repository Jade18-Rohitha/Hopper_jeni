#!/usr/bin/env bash
# ============================================================================
# Apples-to-apples vs cuDNN, N-run median (default N=25), SM clock locked at CLOCK MHz.
# RULE (2026-10-17, user): full sweep = clock FIXED, cuDNN and ours interleaved, 25 runs per shape, median.
# 1980 MHz did not hold under sweep load (median25_20261017_093203: median 1785 MHz, 700 W power cap 79% of the
# time), so CLOCK = the highest clock that holds under this load, measured with scripts/find_stable_clock.sh.
# The actual SM clock is logged for the whole sweep (nvidia-smi, 500 ms) and checked against CLOCK at the end.
#
#   OURS  (build/bin/gqa_fwd_bwd):  V62 fwd | Vj1d, Vj1, Vj2, Vj3 bwd | V62+Vj1d/Vj1/Vj2/Vj3 chained fwd+bwd
#   VG1   (colleague, same binary):  Vg1 fwd | Vg1 bwd | Vg1+Vg1 chained fwd+bwd (Vg1 bwd on Vg1 fwd's O/LSE)
#   cuDNN (precision/cudnn_time.py): fwd    | bwd-only | sdpa(...).backward(dO)
#
# Both sides: SAME inputs (data/gqa_*.bin), triton.do_bench-style timing (L2 flush per
# rep, async queue, one sync), backend forced to cuDNN. Runs are INTERLEAVED (ours,
# cuDNN, ours, cuDNN, ...) so thermal drift hits both equally.
#
#   usage:  GPU=1 CLOCK=1755 bash scripts/run_median12.sh       # all 18 shapes, N=25
#           N=5 SHAPES="8 12 4;4 24 8" GPU=1 CLOCK=1755 bash scripts/run_median12.sh
#   GPU = nvidia-smi index of the ONE GPU to use (PCI order); only that GPU's clocks are touched.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # -> Hopper/

N=${N:-25}
CLOCK=${CLOCK:-1980}
GPU=${GPU:?set GPU=<nvidia-smi index>, e.g. GPU=2 bash scripts/run_median12.sh}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
# 18 shapes: the original 12 + Hq:Hkv = 18:6 and 18:9 (added 2026-10-17), B = 2, 4, 8
SHAPES=${SHAPES:-"2 12 4;4 12 4;8 12 4;2 16 4;4 16 4;8 16 4;2 18 6;4 18 6;8 18 6;2 18 9;4 18 9;8 18 9;2 24 8;4 24 8;8 24 8;2 32 8;4 32 8;8 32 8"}
LOG="reports/median${N}_gpu${GPU}_${CLOCK}MHz_$(date +%Y%m%d_%H%M%S).log"

echo "== building gqa_fwd_bwd =="
cmake --build build --target gqa_fwd_bwd -j

echo "== GPU $GPU: locking clocks (pm on, gpu clock $CLOCK MHz); other GPUs untouched =="
sudo nvidia-smi -i "$GPU" -pm 1           >/dev/null
sudo nvidia-smi -i "$GPU" -lgc "$CLOCK"   >/dev/null
nvidia-smi -i "$GPU" --query-gpu=index,name,pci.bus_id,clocks.sm,clocks.max.sm,power.limit,power.draw --format=csv
SMI="${LOG%.log}_smi.csv"
nvidia-smi -i "$GPU" --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader,nounits -lms 500 > "$SMI" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null; sudo nvidia-smi -i "$GPU" -rgc >/dev/null || true' EXIT

IFS=';' read -ra SH <<< "$SHAPES"
{
for s in "${SH[@]}"; do
  read -r B HQ HKV <<< "$s"
  echo "############ SHAPE B=$B Hq=$HQ Hkv=$HKV"
  python3 precision/baseline_gqa.py "$B" "$HQ" "$HKV" > /dev/null   # reference data/ for this shape
  ./build/bin/gqa_fwd_bwd "$B" "$HQ" "$HKV" | grep -aE "CHECK|don't match|Ref:"   # correctness once
  for r in $(seq 1 "$N"); do
    echo "## run $r"
    ./build/bin/gqa_fwd_bwd "$B" "$HQ" "$HKV" | grep -E "^OURS|^VG1"
    CUDNN_SUSTAINED=0 python3 precision/cudnn_time.py "$B" "$HQ" "$HKV" | grep -E "^cuDNN |^# "
  done
done
} 2>&1 | tee "$LOG"

echo ""
python3 scripts/median_summary.py "$LOG"
kill $SMI_PID 2>/dev/null || true
echo ""
echo "== SM clock during the sweep (GPU $GPU, samples with power > 300 W = under load) =="
python3 - "$SMI" "$CLOCK" <<'EOF'
import sys, collections
TARGET = int(sys.argv[2])
R = {0x1: "idle", 0x4: "SW-power-cap", 0x8: "HW-slowdown", 0x20: "SW-thermal", 0x40: "HW-thermal", 0x80: "HW-power-brake"}
clk, why = [], collections.Counter()
for line in open(sys.argv[1]):
    f = [x.strip() for x in line.split(",")]
    if len(f) < 5: continue
    try: c, p, r = int(f[1]), float(f[2]), int(f[4], 16)
    except ValueError: continue
    if p < 300: continue
    clk.append(c); why[" + ".join(n for b, n in R.items() if r & b) or "none"] += 1
if not clk: print("no loaded samples"); sys.exit()
clk.sort(); n = len(clk)
at = sum(c >= TARGET - 15 for c in clk)          # 15 MHz = one clock step of tolerance
print(f"loaded samples: {n}   at {TARGET} MHz: {at} ({100*at/n:.0f}%)   min {clk[0]}  median {clk[n//2]}  max {clk[-1]} MHz")
print("throttle reasons:", ", ".join(f"{k} {100*v/n:.0f}%" for k, v in why.most_common()))
print("RULE CHECK:", f"clock held at {TARGET}" if at == n else f"clock did NOT hold at {TARGET} for all loaded samples -- see above")
EOF
echo "== log: $LOG   clock log: $SMI =="
