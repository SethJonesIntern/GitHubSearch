"""How many determinism knobs does a single LLM call set?

The determinism table (01_determinism_knobs.png) reports each knob's rate
separately — temperature 5.6%, seed 0.1%, and so on. That leaves the joint
question open: a call could set several knobs, or none. This chart answers it by
counting, per call site, how many of the eight determinism-relevant kwargs are
passed, then plotting the frequency of each count.

Counts are recomputed from call_metadata_all.csv through analyze.py's own
filters (FP tiers dropped, since-removed patterns dropped, excluded repos
dropped), so this figure and the report can never disagree.

Single-series magnitude chart over ordered bins: one hue, the headline bin
emphasised and the rest recessive (the story is one number), no legend, direct
value labels, recessive chrome. Palette validated with the dataviz validator
(ordinal, light surface): ALL CHECKS PASS.

Writes pipeline/artifacts/figures/07_knob_count.png + knob_count.csv (table twin).
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "Applications", _ROOT / "Wrapper"):
    sys.path.insert(0, str(_p))
import analyze  # noqa: E402
from pipeline import paths  # noqa: E402

OUT_PNG = paths.ARTIFACTS_DIR / "figures" / "07_knob_count.png"
OUT_CSV = paths.ARTIFACTS_DIR / "figures" / "knob_count.csv"

# dataviz palette (light mode), blue ordinal ramp: step 450 emphasis / 250 context.
EMPHASIS = "#2a78d6"
CONTEXT = "#86b6ef"
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def distribution() -> tuple[Counter, int, int]:
    """{n_knobs_set -> n_calls} over every distinct call site, the call total,
    and the number of repositories those calls come from."""
    meta = analyze.drop_fp(analyze.drop_removed_patterns_calls(
        analyze.load("call_metadata_all.csv")))
    meta = meta[~meta["repo"].isin(analyze.EXCLUDED)]
    knobs = set(analyze.KNOBS)

    all_calls = meta["call_id"].drop_duplicates()
    per_call = (meta[meta["arg_keyword"].isin(knobs)]
                .groupby("call_id")["arg_keyword"].nunique()
                .reindex(all_calls).fillna(0).astype(int))
    return Counter(per_call), len(all_calls), meta["repo"].nunique()


def round_tops(ax, bars, radius_pt=4.0):
    """Replace each bar with a 4pt-rounded-corner patch anchored to the baseline.

    FancyBboxPatch rounds in data units, so the radius is converted from points
    through the drawn axes size and the x/y scale difference is carried by
    mutation_aspect — otherwise the corners smear on whichever axis is longer.
    A bar shorter than twice the radius is left square: rounding it would eat the
    whole mark and render the tail bins as spikes rather than bars.
    """
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    dpi = ax.figure.dpi
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_per_pt = (x1 - x0) / (bbox.width / dpi * 72)
    y_per_pt = (y1 - y0) / (bbox.height / dpi * 72)
    rx = radius_pt * x_per_pt
    for p in bars:
        bb = p.get_bbox()
        if bb.height < 2 * radius_pt * y_per_pt:
            continue                      # too short to round — keep the rectangle
        p.set_visible(False)
        ax.add_patch(FancyBboxPatch(
            (bb.xmin, bb.ymin), bb.width, bb.height,
            boxstyle=f"round,pad=0,rounding_size={rx}",
            mutation_aspect=(radius_pt * y_per_pt) / rx,
            facecolor=p.get_facecolor(), linewidth=0, clip_on=False))


def main():
    dist, total, n_repos = distribution()
    ks = sorted(dist)
    pct = [100 * dist[k] / total for k in ks]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["knobs_set", "calls", "pct_of_calls"])
        for k in ks:
            w.writerow([k, dist[k], f"{100 * dist[k] / total:.2f}"])

    fig, ax = plt.subplots(figsize=(7.4, 4.3), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    colors = [EMPHASIS if k == 0 else CONTEXT for k in ks]
    bars = ax.bar(ks, pct, width=0.68, color=colors, linewidth=0)

    ax.set_ylim(0, max(pct) * 1.18)
    ax.set_xlim(-0.6, max(ks) + 0.6)
    ax.set_xticks(ks)

    # direct labels: every bin is labelled because the tail values (0.1%, 0.01%)
    # are unreadable from the axis alone, and the table twin carries the counts.
    for k, p in zip(ks, pct):
        label = f"{p:.1f}%" if p >= 0.05 else f"{p:.2f}%"
        ax.text(k, p + max(pct) * 0.022, label, ha="center", va="bottom",
                fontsize=9, color=INK if k == 0 else MUTED,
                fontweight="bold" if k == 0 else "normal")

    fig.text(0.055, 0.965, "Most LLM calls set no determinism parameters at all",
             fontsize=13.5, color=INK, fontweight="bold", va="top")
    fig.text(0.055, 0.905,
             "Determinism kwargs per call site: temperature, top_p, top_k, seed, "
             "max_tokens,\nfrequency_penalty, presence_penalty, model",
             fontsize=8.5, color=SECOND, va="top", linespacing=1.5)
    fig.text(0.055, 0.805, f"n = {total:,} calls across {n_repos} repositories",
             fontsize=8.5, color=MUTED, va="top")
    ax.set_xlabel("number of determinism parameters set in the call",
                  fontsize=9.5, color=SECOND, labelpad=8)
    ax.set_ylabel("share of LLM calls", fontsize=9.5, color=SECOND, labelpad=8)

    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels(["0", "20%", "40%", "60%", "80%"])

    fig.subplots_adjust(top=0.74, left=0.11, right=0.97, bottom=0.17)
    round_tops(ax, bars)
    fig.savefig(OUT_PNG, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_CSV}")
    print(f"\n{'knobs':>5}{'calls':>9}{'pct':>8}")
    for k in ks:
        print(f"{k:>5}{dist[k]:>9}{100 * dist[k] / total:>7.2f}%")


if __name__ == "__main__":
    main()
