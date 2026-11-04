"""Horizontal bar chart of the grouped framework-frequency table.

Renders keep_frequency.csv (already grouped into ecosystem categories and counted
as distinct applications by keep_frequency.py), so the chart and the table can
never disagree. Single-series magnitude chart: one hue, recessive chrome, direct
value labels, no legend.

Writes pipeline/artifacts/keep_frequency.png.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths  # noqa: E402

FREQ_CSV = paths.ARTIFACTS_DIR / "keep_frequency.csv"
SLIM_CSV = paths.ARTIFACTS_DIR / "applications_slim.csv"
OUT_PNG = paths.ARTIFACTS_DIR / "keep_frequency.png"

# dataviz palette (light mode): single sequential/categorical hue + recessive ink.
BAR = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25, help="show the top N frameworks")
    args = ap.parse_args()

    if not FREQ_CSV.exists():
        sys.exit(f"{FREQ_CSV} not found — run Applications/keep_frequency.py first.")
    rows = list(csv.DictReader(open(FREQ_CSV, encoding="utf-8")))[:args.top]
    names = [r["framework"] for r in rows]
    vals = [int(r["applications"]) for r in rows]
    total = (sum(1 for _ in open(SLIM_CSV, encoding="utf-8")) - 1) if SLIM_CSV.exists() else None

    fig, ax = plt.subplots(figsize=(10, max(4, 0.44 * len(names))))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y = range(len(names))
    ax.barh(y, vals, height=0.72, color=BAR, zorder=3)   # gap between bars
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, color=INK)
    ax.invert_yaxis()                                    # largest on top

    # direct value labels at bar ends, in ink (not the bar color)
    for i, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, i, str(v), va="center", fontsize=8, color=INK)

    # recessive chrome: no box, hairline x-grid only, muted axis
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("distinct applications importing this framework", color=MUTED)

    title = f"Framework frequency — top {len(names)} (ecosystems grouped)"
    if total:
        title += f"\n{total} slimmed applications"
    ax.set_title(title, color=INK)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
