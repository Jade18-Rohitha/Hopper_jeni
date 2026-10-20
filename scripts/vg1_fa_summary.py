#!/usr/bin/env python3
"""vg1_fa_summary.py — three tables (forward, backward, forward+backward) from a run_vg1_vs_fa.sh log:
Vg1 vs FA3 vs FA4, median of N runs per shape (ms). Δ = FA − Vg1 in ms (+ = Vg1 faster);
wins = runs where Vg1 < that FA library (matched runs). Prints markdown, ready for report.md.
usage: python3 scripts/vg1_fa_summary.py reports/vg1_vs_fa_<...>.log
"""
import re, statistics, sys
K = {"VG1 fwd Vg1": ("g", "f"), "VG1 bwd Vg1": ("g", "b"), "VG1 fwd+bwd Vg1+Vg1": ("g", "fb"),
     "FA3 fwd": ("3", "f"), "FA3 bwd-only": ("3", "b"), "FA3 fwd+bwd": ("3", "fb"),
     "FA4 fwd": ("4", "f"), "FA4 bwd-only": ("4", "b"), "FA4 fwd+bwd": ("4", "fb")}
rows, fails, unavail, cur, run = {}, {}, set(), None, {}
def flush():
    if cur and run: rows[cur].append(dict(run))
for line in open(sys.argv[1], errors="replace"):
    line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
    m = re.match(r"#+ SHAPE B=(\d+) Hq=(\d+) Hkv=(\d+)", line)
    if m: flush(); run = {}; cur = f"{m[1]}×{m[2]} ({m[3]})"; rows[cur] = []; fails[cur] = []; continue
    if line.startswith("## run"): flush(); run = {}; continue
    if line.startswith("CHECK") and "match!" not in line and cur: fails[cur].append(line)
    m = re.match(r"# (FA[34]): unavailable", line)
    if m: unavail.add(m[1])
    m = re.match(r"(.+?)\s+([\d.]+) ms$", line)
    if m and m[1].strip() in K: run[K[m[1].strip()]] = float(m[2])
flush()
libs = [l for l in ("3", "4") if any((l, "b") in r for rs in rows.values() for r in rs)]
n = max((len(r) for r in rows.values()), default=0)
print(f"== Vg1 vs " + " vs ".join(f"FA{l}" for l in libs) + f" — median of {n} runs per shape (ms). "
      "Δ = FA − Vg1 (+ = Vg1 faster); wins = runs Vg1 < FA ==")
if unavail: print("unavailable:", ", ".join(sorted(unavail)))
for title, ph in (("Forward", "f"), ("Backward", "b"), ("Forward + backward (training step)", "fb")):
    hdr = "| B×Hq (Hkv) | **Vg1** | " + " | ".join(f"FA{l} | Δ vs FA{l} | wins" for l in libs) + " |"
    print(f"\n**{title}**\n\n{hdr}\n|" + "---|" * (2 + 3 * len(libs)))
    for s, rs in rows.items():
        g = [r[("g", ph)] for r in rs if ("g", ph) in r]
        if not g: print(f"| {s} | — |" + " — | — | — |" * len(libs)); continue
        cells = [f"**{statistics.median(g):.3f}**"]
        for l in libs:
            pr = [r for r in rs if ("g", ph) in r and (l, ph) in r]
            if not pr: cells += ["—", "—", "—"]; continue
            fa = statistics.median(r[(l, ph)] for r in pr)
            cells += [f"{fa:.3f}", f"{fa - statistics.median(r[('g', ph)] for r in pr):+.3f}",
                      f"{sum(r[('g', ph)] < r[(l, ph)] for r in pr)}/{len(pr)}"]
        print(f"| {s} | " + " | ".join(cells) + " |")
bad = {s: f for s, f in fails.items() if f}
print("\nCORRECTNESS: " + ("all checks passed (Vg1 and FA, vs the PyTorch reference)" if not bad else ""))
for s, f in bad.items(): print(f"  {s}: " + "; ".join(f))
