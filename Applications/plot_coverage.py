"""Cumulative-coverage (Pareto) chart over the grouped framework frequency.

Answers "how many top frameworks cover ~90% of applications?". Single y-axis in
% of applications (no dual axis): bars are each framework's own share; the line
is cumulative DISTINCT-app coverage (union, de-duped for multi-framework apps).
The 90% threshold and the resulting top-N cut are marked.

Writes pipeline/artifacts/framework_coverage.png.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths  # noqa: E402
from pipeline import cuts  # noqa: E402
import importlib.util
_spec = importlib.util.spec_from_file_location("kf", Path(__file__).with_name("keep_frequency.py"))
kf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(kf)

SLIM_CSV = paths.ARTIFACTS_DIR / "applications_slim.csv"
OUT_PNG = paths.ARTIFACTS_DIR / "framework_coverage.png"

# dataviz palette (light): two well-separated categorical hues + recessive ink.
BAR = "#2a78d6"        # slot 1 blue — per-framework share
LINE = "#eb6834"       # slot 8 orange — cumulative coverage
KEEP_WASH = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=90.0, help="coverage %% to mark")
    ap.add_argument("--show", type=int, default=30, help="how many top frameworks to plot")
    args = ap.parse_args()

    rows = cuts.drop_cut(list(csv.DictReader(open(SLIM_CSV, encoding="utf-8"))))
    # Coverage denominator = REAL AI APPS (COVERAGE_ANALYSIS.md). drop_cut alone is not
    # enough: the §9 "Excluded" bucket (junk-token-only rows — omnigent/clai/
    # langchain_tests collisions, real_ai_app=0) was enforced as a flag column, never as
    # in_scope=0, so ~107 junk rows survive it. They are only "coverable" by junk
    # categories, which pushed the 90% cut from top-19 to top-21 (found 2026-08-25).
    rows = [r for r in rows if (r.get("real_ai_app") or "").strip() == "1"]
    total = len(rows)
    # Exempt/phantom frameworks cannot COVER an app: omnigent is the §2 phantom and
    # agentops is exempt observability. With the real-AI denominator they no longer
    # change the cut, but excluding them keeps the ranking principled.
    NON_COVERING = {"omnigent", "agentops"}
    app_cats = [{kf.category(n.strip()) for n in (r.get("matched_frameworks") or "").split(",") if n.strip()}
                - NON_COVERING
                for r in rows]
    per = Counter()
    for cats in app_cats:
        for c in cats:
            per[c] += 1
    ranked = [c for c, _ in per.most_common()]

    # cumulative distinct-app coverage as frameworks are added by rank
    covered, cum_pct = set(), []
    for c in ranked:
        for j, cats in enumerate(app_cats):
            if c in cats:
                covered.add(j)
        cum_pct.append(100 * len(covered) / total)
    cut = next(i for i, p in enumerate(cum_pct) if p >= args.threshold)  # 0-indexed
    cut_n = cut + 1

    n = min(args.show, len(ranked))
    x = range(n)
    shares = [100 * per[ranked[i]] / total for i in x]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    # kept region wash (top-N cut)
    ax.axvspan(-0.5, cut + 0.5, color=KEEP_WASH, alpha=0.06, zorder=0)

    ax.bar(list(x), shares, width=0.72, color=BAR, zorder=3, label="share of apps (this framework)")
    ax.plot(list(x), cum_pct[:n], color=LINE, lw=2, marker="o", ms=4, zorder=4,
            label="cumulative coverage")

    # threshold + cut markers
    ax.axhline(args.threshold, color=MUTED, ls="--", lw=1, zorder=2)
    ax.text(n - 0.5, args.threshold + 1.5, f"{args.threshold:.0f}%", color=MUTED,
            ha="right", va="bottom", fontsize=9)
    ax.axvline(cut + 0.5, color=INK, ls=":", lw=1.2, zorder=2)
    ax.annotate(f"top {cut_n} → {cum_pct[cut]:.0f}%",
                xy=(cut, cum_pct[cut]), xytext=(cut + 1.2, cum_pct[cut] - 12),
                color=INK, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))

    ax.set_xticks(list(x))
    ax.set_xticklabels([ranked[i] for i in x], rotation=45, ha="right", color=INK, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of applications", color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"Framework coverage — top {cut_n} frameworks cover "
                 f"{cum_pct[cut]:.0f}% of {total} applications\n"
                 f"(ecosystems grouped; cumulative = distinct apps)", color=INK)
    ax.legend(loc="center right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT_PNG}  (cut: top {cut_n} = {cum_pct[cut]:.1f}%)")


if __name__ == "__main__":
    main()
