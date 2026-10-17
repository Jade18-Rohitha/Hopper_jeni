#!/usr/bin/env bash
# ============================================================================
# find_stable_clock.sh — highest SM clock that HOLDS under the full sweep's load on one GPU.
# For each candidate clock (high -> low): lock it, run the sweep's heaviest shape (8x32, ours + cuDNN,
# same do_bench pattern as run_median12.sh) for ~ROUNDS iterations while logging the SM clock every 200 ms,
# and report the % of loaded samples (power > 300 W) that sat at the lock (±15 MHz).
# The first clock that holds 100% is the one to pass as CLOCK= to run_median12.sh.
#   usage:  GPU=1 bash scripts/find_stable_clock.sh
#           GPU=1 CLOCKS="1980 1830 1755 1680" ROUNDS=4 bash scripts/find_stable_clock.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # -> Hopper/
GPU=${GPU:?set GPU=<nvidia-smi index>}
CLOCKS=${CLOCKS:-"1980 1905 1830 1755 1680 1605"}
ROUNDS=${ROUNDS:-3}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
cmake --build build --target gqa_fwd_bwd -j >/dev/null
python3 precision/baseline_gqa.py 8 32 8 > /dev/null                 # heaviest sweep shape
sudo nvidia-smi -i "$GPU" -pm 1 >/dev/null
trap 'sudo nvidia-smi -i "$GPU" -rgc >/dev/null || true' EXIT
echo "GPU $GPU: $(nvidia-smi -i "$GPU" --query-gpu=name,clocks.max.sm,power.limit --format=csv,noheader)"
best=""
for c in $CLOCKS; do
  sudo nvidia-smi -i "$GPU" -lgc "$c" >/dev/null
  sleep 3                                                             # cool down / settle
  CSV=$(mktemp)
  nvidia-smi -i "$GPU" --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits -lms 200 > "$CSV" &
  P=$!
  for r in $(seq 1 "$ROUNDS"); do
    ./build/bin/gqa_fwd_bwd 8 32 8 > /dev/null
    CUDNN_SUSTAINED=0 python3 precision/cudnn_time.py 8 32 8 > /dev/null
  done
  kill $P; wait $P 2>/dev/null || true
  read -r pct med mn <<< "$(python3 - "$CSV" "$c" <<'PY'
import sys
t = int(sys.argv[2]); cl = []
for line in open(sys.argv[1]):
    f = [x.strip() for x in line.split(",")]
    try:
        if float(f[1]) > 300: cl.append(int(f[0]))
    except (ValueError, IndexError): pass
cl.sort(); n = len(cl) or 1
print(f"{100*sum(x >= t - 15 for x in cl)/n:.0f} {cl[len(cl)//2] if cl else 0} {cl[0] if cl else 0}")
PY
)"
  rm -f "$CSV"
  echo "lock ${c} MHz: held ${pct}% of loaded samples   (median ${med}, min ${mn} MHz)"
  if [ -z "$best" ] && [ "$pct" = "100" ]; then best=$c; fi
done
echo ""
if [ -n "$best" ]; then echo "RECOMMENDED: CLOCK=$best   (highest lock that held 100% under sweep load)"
else echo "No candidate held 100%; try lower CLOCKS=..."; fi
