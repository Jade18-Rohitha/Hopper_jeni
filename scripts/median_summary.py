#!/usr/bin/env python3
"""median_summary.py — summarize a run_median12.sh log: per-shape median of N runs vs cuDNN.

usage: python3 scripts/median_summary.py reports/median12_<stamp>.log
Δ = cuDNN - ours in ms: + = we are faster.  wins = runs where ours < cuDNN (matched run pairs).
Vj1 columns appear only if the log has them (gqa_fwd_bwd built after 2026-10-12 15:15).
"""
import re
import statistics
import sys

KEYS = {"OURS fwd V62": "of",
        "OURS bwd Vj1d": "ob", "OURS fwd+bwd V62+Vj1d": "ofb",
        "OURS bwd Vj1": "ob1", "OURS fwd+bwd V62+Vj1": "ofb1",
        "OURS bwd Vj2": "ob2", "OURS fwd+bwd V62+Vj2": "ofb2",
        "OURS bwd Vj3": "ob3", "OURS fwd+bwd V62+Vj3": "ofb3",
        "cuDNN fwd": "cf", "cuDNN bwd-only": "cb", "cuDNN fwd+bwd": "cfb"}
REQUIRED = {"of", "ob", "ofb", "cf", "cb", "cfb"}

rows, checks, cur, run = {}, {}, None, {}


def flush():
    if cur and REQUIRED <= run.keys():
        rows[cur].append(dict(run))


for line in open(sys.argv[1], errors="replace"):
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)          # strip ANSI colors from checkResult
    m = re.match(r"#+ SHAPE B=(\d+) Hq=(\d+) Hkv=(\d+)", line)
    if m:
        flush(); run = {}
        cur = f"{m[1]}x{m[2]} ({m[3]})"; rows[cur] = []; checks[cur] = []
        continue
    if line.startswith("## run"):
        flush(); run = {}
        continue
    if cur and ("Values don't match" in line):
        checks[cur].append(line.strip())
    m = re.match(r"(.+?)\s+([\d.]+) ms", line.strip())
    if m and m[1].strip() in KEYS:
        run[KEYS[m[1].strip()]] = float(m[2])
flush()


def pct(c, o):        # Δ in ms: cuDNN - ours (+ = we are faster)
    return c - o


def med(rs, k):
    return statistics.median(r[k] for r in rs)


def wins(rs, ours, theirs):
    return f"{sum(r[ours] < r[theirs] for r in rs):>2}/{len(rs):<2}"


have_vj1 = any("ob1" in r for rs in rows.values() for r in rs)
have_vj2 = any("ob2" in r for rs in rows.values() for r in rs)
have_vj3 = any("ob3" in r for rs in rows.values() for r in rs)
n = max((len(rs) for rs in rows.values()), default=0)
print(f"== median of {n} runs/shape (ms). Δ = cuDNN - ours in ms: + = we are faster. wins = runs ours < cuDNN ==\n")

hdr = f"{'B×Hq (Hkv)':<11} | {'cuDNN':>7} {'V62':>7} {'Δ':>7} {'wins':>5}"
print("FORWARD\n" + hdr)
for s, rs in rows.items():
    if rs:
        print(f"{s:<11} | {med(rs,'cf'):>7.3f} {med(rs,'of'):>7.3f} {pct(med(rs,'cf'),med(rs,'of')):>+7.3f} {wins(rs,'of','cf')}")

for title, c, pairs in [("BACKWARD", "cb", [("Vj1d", "ob"), ("Vj1", "ob1"), ("Vj2", "ob2"), ("Vj3", "ob3")]),
                        ("FORWARD + BACKWARD (training step)", "cfb",
                         [("V62+Vj1d", "ofb"), ("V62+Vj1", "ofb1"), ("V62+Vj2", "ofb2"), ("V62+Vj3", "ofb3")])]:
    pairs = [p for p in pairs if p[1] in REQUIRED or (have_vj1 and p[1] in ("ob1", "ofb1"))
             or (have_vj2 and p[1] in ("ob2", "ofb2")) or (have_vj3 and p[1] in ("ob3", "ofb3"))]
    print(f"\n{title}\n{'B×Hq (Hkv)':<11} | {'cuDNN':>7} | " +
          " | ".join(f"{name:>8} {'Δ':>7} {'wins':>5}" for name, _ in pairs))
    for s, rs in rows.items():
        if not rs:
            continue
        cm = med(rs, c)
        cells = []
        for _, k in pairs:
            sub = [r for r in rs if k in r]
            cells.append(f"{med(sub,k):>8.3f} {pct(cm,med(sub,k)):>+7.3f} {wins(sub,k,c)}" if sub else f"{'--':>8} {'':>7} {'':>5}")
        print(f"{s:<11} | {cm:>7.3f} | " + " | ".join(cells))

bad = {s: c for s, c in checks.items() if c}
print("\nCORRECTNESS: " + ("all printed checks passed" if not bad else ""))
for s, c in bad.items():
    print(f"  {s}: " + "; ".join(c))
