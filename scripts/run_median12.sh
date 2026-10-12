#!/usr/bin/env bash
# ============================================================================
# Apples-to-apples vs cuDNN, N-run median-of-medians (default N=12), locked clock.
#
#   OURS  (build/bin/gqa_fwd_bwd):  V62 fwd | Vj1d bwd | V62+Vj1d chained fwd+bwd
#   cuDNN (precision/cudnn_time.py): fwd    | bwd-only | sdpa(...).backward(dO)
#
# Both sides: SAME inputs (data/gqa_*.bin), triton.do_bench-style timing (L2 flush per
# rep, async queue, one sync), backend forced to cuDNN. Runs are INTERLEAVED (ours,
# cuDNN, ours, cuDNN, ...) so thermal drift hits both equally.
#
#   usage:  bash scripts/run_median12.sh            # all 12 shapes, N=12
#           N=5 SHAPES="8 12 4;4 24 8" bash scripts/run_median12.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # -> Hopper/

N=${N:-12}
SHAPES=${SHAPES:-"2 12 4;4 12 4;8 12 4;2 16 4;4 16 4;8 16 4;2 24 8;4 24 8;8 24 8;2 32 8;4 32 8;8 32 8"}
LOG="reports/median${N}_$(date +%Y%m%d_%H%M%S).log"

echo "== building gqa_fwd_bwd =="
cmake --build build --target gqa_fwd_bwd -j

echo "== locking clocks (pm on, gpu clock 1980 MHz) =="
sudo nvidia-smi -pm 1     >/dev/null
sudo nvidia-smi -lgc 1980 >/dev/null
trap 'sudo nvidia-smi -rgc >/dev/null || true' EXIT

IFS=';' read -ra SH <<< "$SHAPES"
{
for s in "${SH[@]}"; do
  read -r B HQ HKV <<< "$s"
  echo "############ SHAPE B=$B Hq=$HQ Hkv=$HKV"
  python3 precision/baseline_gqa.py "$B" "$HQ" "$HKV" > /dev/null   # reference data/ for this shape
  ./build/bin/gqa_fwd_bwd "$B" "$HQ" "$HKV" | grep -E "^  (O|LSE|dQ|dK|dV):"   # correctness once
  for r in $(seq 1 "$N"); do
    echo "## run $r"
    ./build/bin/gqa_fwd_bwd "$B" "$HQ" "$HKV" | grep -E "^OURS"
    python3 precision/cudnn_time.py "$B" "$HQ" "$HKV" | grep -E "^cuDNN (fwd|bwd)"
  done
done
} 2>&1 | tee "$LOG"

echo ""
echo "== median of $N runs per shape (ms). +% = we are faster than cuDNN. wins = runs where ours < cuDNN =="
python3 - "$LOG" <<'EOF'
import re, statistics, sys
rows, cur, run = {}, None, {}
keys = {"OURS fwd V62": "of", "OURS bwd Vj1d": "ob", "OURS fwd+bwd V62+Vj1d": "ofb",
        "cuDNN fwd": "cf", "cuDNN bwd-only": "cb", "cuDNN fwd+bwd": "cfb"}
def flush():
    if cur and len(run) == 6: rows[cur].append(dict(run))
for line in open(sys.argv[1]):
    m = re.match(r"#+ SHAPE B=(\d+) Hq=(\d+) Hkv=(\d+)", line)
    if m:
        flush(); run = {}; cur = f"{m[1]}x{m[2]} ({m[3]})"; rows[cur] = []; continue
    if line.startswith("## run"):
        flush(); run = {}; continue
    m = re.match(r"(.+?)\s+([\d.]+) ms", line.strip())
    if m and m[1].strip() in keys: run[keys[m[1].strip()]] = float(m[2])
flush()
def pct(c, o): return (c - o) / c * 100
print(f"{'B×Hq (Hkv)':<11} | {'cuDNN fwd':>9} {'V62':>7} {'Δ':>7} | {'cuDNN bwd':>9} {'Vj1d':>7} {'Δ':>7} {'wins':>5} | "
      f"{'cuDNN f+b':>9} {'V62+Vj1d':>8} {'Δ':>7} {'wins':>5}")
for shape, rs in rows.items():
    if not rs: continue
    med = {k: statistics.median(r[k] for r in rs) for k in rs[0]}
    wb = sum(r["ob"] < r["cb"] for r in rs); wfb = sum(r["ofb"] < r["cfb"] for r in rs)
    print(f"{shape:<11} | {med['cf']:>9.3f} {med['of']:>7.3f} {pct(med['cf'],med['of']):>+6.1f}% | "
          f"{med['cb']:>9.3f} {med['ob']:>7.3f} {pct(med['cb'],med['ob']):>+6.1f}% {wb:>2}/{len(rs):<2} | "
          f"{med['cfb']:>9.3f} {med['ofb']:>8.3f} {pct(med['cfb'],med['ofb']):>+6.1f}% {wfb:>2}/{len(rs):<2}")
EOF
echo "== log: $LOG =="
