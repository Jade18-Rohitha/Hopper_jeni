#!/usr/bin/env bash
# ============================================================================
# Vg1 vs FlashAttention-3 vs FlashAttention-4 — fair, fixed-clock sweep (2026-10-20).
#   Vg1 (colleague's fwd+bwd)   : VG1_ONLY=1 build/bin/gqa_fwd_bwd     -> VG1 fwd / bwd / fwd+bwd
#   FA3, FA4                    : precision/fa_time.py                  -> FA3/FA4 fwd / bwd-only / fwd+bwd
# Same inputs (data/gqa_*.bin), same do_bench timing (L2 flush per rep, async queue), interleaved
# (Vg1, FA, Vg1, FA, ...), N runs per shape (default 12), median. FA's [B,S,H,D] transposes are outside its timers;
# FA's bwd-only includes its own preprocess + dQ postprocess (as Vg1's and cuDNN's do).
# Correctness checked once per shape for all three (vs the PyTorch reference). SM clock locked to CLOCK on GPU only,
# logged every 500 ms, RULE CHECK at the end. Install FA3/FA4 first: bash harness/setup_sota_h200.sh fa4 fa3
#   usage:  GPU=1 CLOCK=1530 bash scripts/run_vg1_vs_fa.sh
#           N=3 SHAPES="8 12 4" GPU=1 CLOCK=1530 bash scripts/run_vg1_vs_fa.sh      # quick check
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # -> Hopper/
GPU=${GPU:?set GPU=<nvidia-smi index>}
CLOCK=${CLOCK:?set CLOCK=<MHz that holds under load on this GPU, from scripts/find_stable_clock.sh>}
N=${N:-12}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
SHAPES=${SHAPES:-"2 12 4;4 12 4;8 12 4;2 16 4;4 16 4;8 16 4;2 18 6;4 18 6;8 18 6;2 18 9;4 18 9;8 18 9;2 24 8;4 24 8;8 24 8;2 32 8;4 32 8;8 32 8"}
LOG="reports/vg1_vs_fa_N${N}_gpu${GPU}_${CLOCK}MHz_$(date +%Y%m%d_%H%M%S).log"

echo "== building gqa_fwd_bwd =="
cmake --build build --target gqa_fwd_bwd -j
echo "== FA libraries available =="
python3 - <<'PY'
import importlib
for lab, mods in (("FA3", ("flash_attn_interface", "flash_attn_3.flash_attn_interface")), ("FA4", ("flash_attn.cute.interface", "flash_attn.cute"))):
    ok = [m for m in mods if importlib.util.find_spec(m.split(".")[0]) is not None]
    print(f"  {lab}: {'found ' + ok[0] if ok else 'NOT INSTALLED (bash harness/setup_sota_h200.sh fa4 fa3)'}")
PY

echo "== GPU $GPU: locking clocks (pm on, gpu clock $CLOCK MHz); other GPUs untouched =="
sudo nvidia-smi -i "$GPU" -pm 1         >/dev/null
sudo nvidia-smi -i "$GPU" -lgc "$CLOCK" >/dev/null
nvidia-smi -i "$GPU" --query-gpu=index,name,pci.bus_id,clocks.sm,clocks.max.sm,power.limit,power.draw --format=csv
SMI="${LOG%.log}_smi.csv"
nvidia-smi -i "$GPU" --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader,nounits -lms 500 > "$SMI" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null; sudo nvidia-smi -i "$GPU" -rgc >/dev/null || true' EXIT

IFS=';' read -ra SH <<< "$SHAPES"
{
echo "# host $(hostname)  GPU $GPU  CLOCK $CLOCK MHz  N $N"
for s in "${SH[@]}"; do
  read -r B HQ HKV <<< "$s"
  echo "############ SHAPE B=$B Hq=$HQ Hkv=$HKV"
  python3 precision/baseline_gqa.py "$B" "$HQ" "$HKV" > /dev/null                   # reference data/ for this shape
  VG1_ONLY=1 ./build/bin/gqa_fwd_bwd "$B" "$HQ" "$HKV" | sed 's/\x1b\[[0-9;]*m//g' | grep -aE "^CHECK|don't match"
  FA_CHECK=1 python3 precision/fa_time.py "$B" "$HQ" "$HKV" | grep -E "^CHECK|^# "     # FA correctness, once
  for r in $(seq 1 "$N"); do
    echo "## run $r"
    VG1_ONLY=1 ./build/bin/gqa_fwd_bwd "$B" "$HQ" "$HKV" | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^VG1"
    FA_CHECK=0 python3 precision/fa_time.py "$B" "$HQ" "$HKV" | grep -E "^FA[34] |^# FA[34]: unavailable"
  done
done
} 2>&1 | tee "$LOG"

kill $SMI_PID 2>/dev/null || true
echo ""
python3 scripts/vg1_fa_summary.py "$LOG"
echo ""
echo "== SM clock during the sweep (GPU $GPU, samples with power > 300 W = under load) =="
python3 - "$SMI" "$CLOCK" <<'PY'
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
clk.sort(); n = len(clk); at = sum(c >= TARGET - 15 for c in clk)
print(f"loaded samples: {n}   at {TARGET} MHz: {at} ({100*at/n:.1f}%)   min {clk[0]}  median {clk[n//2]}  max {clk[-1]} MHz")
print("throttle reasons:", ", ".join(f"{k} {100*v/n:.0f}%" for k, v in why.most_common()))
print("RULE CHECK:", f"clock held at {TARGET}" if at == n else f"clock did NOT hold at {TARGET} for all loaded samples -- see above")
PY
echo "== log: $LOG   clock log: $SMI =="
