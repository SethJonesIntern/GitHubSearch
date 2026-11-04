"""Shared chart style for the question-and-table figure set.

Every figure in `make_figures.py` is built through this module so the eight of
them read as one system: same palette, same header grammar, same chrome, same
mark geometry. Change a colour or a margin here and all eight move together.

The header grammar is the point of the whole set — a reader who knows nothing
about the pipeline should be able to answer the question from the picture alone:

    QUESTION   the research question, verbatim, in bold
    ANSWER     the finding in one plain-English sentence
    NOTE       n = ..., and any caveat that changes how the number reads

Palette: the dataviz reference palette, light mode, surface #fcfcfb.
  - single-series magnitude  -> blue ordinal ramp, step 450 emphasis / 250 context
  - 2-3 distinct series      -> categorical slots 1-3 (blue, orange, aqua)
Validated with scripts/validate_palette.js:
  "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all   -> ALL CHECKS PASS
  "#86b6ef,#2a78d6"         --mode light --ordinal     -> ALL CHECKS PASS
Slot 3 (aqua) carries a contrast WARN against the light surface, so the relief
rule applies: every figure here ships direct value labels AND a CSV table twin.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# ── palette ──────────────────────────────────────────────────────────────────
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3
EMPHASIS = "#2a78d6"                          # blue 450 — the bar that answers it
CONTEXT = "#86b6ef"                           # blue 250 — the rest

# Reserved identity pair. Orange means "raw provider SDK" in EVERY figure and never
# anything else, so a reader can scan the set and trust the hue without re-reading
# each legend. Anything needing a second series for a NON-SDK distinction uses the
# blue ordinal pair (EMPHASIS/CONTEXT) instead — see Q4.
FRAMEWORK = SERIES[0]        # blue 450
SDK = SERIES[1]              # orange
# Light steps for a second series WITHIN each identity (Q6: 0 jumps vs 1 jump).
# Hue keeps carrying framework-vs-SDK; lightness carries the depth. Both pairs
# validated --ordinal on the light surface (blue 2.06:1, orange 2.03:1).
FRAMEWORK_LIGHT = CONTEXT    # blue 250
SDK_LIGHT = "#f2a07d"
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# ── layout, in inches ────────────────────────────────────────────────────────
# The header is MEASURED from its own text, not fixed: an answer or note that
# wraps to another line has to push the axes down, or it prints on top of the
# top gridline. Baselines below are inches from the top edge.
TITLE_Y = 0.26            # question baseline
ANSWER_Y = 0.56           # answer block top
ANSWER_LINE = 0.205       # 9.5pt at linespacing 1.55
NOTE_LINE = 0.177         # 8.5pt at linespacing 1.50
NOTE_PAD = 0.26           # gap between the answer block and the note
BOTTOM_PAD = 0.12         # clearance between the header and the axes
FOOTER_IN = 0.62
LEFT_FRAC = 0.055         # header text x, as a figure fraction

matplotlib.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def figure(question: str, subtitle: str = "", note: str = "", *,
           plot_height_in: float, width_in: float = 7.6, legend_rows: int = 0):
    """Build the figure and draw its header.

    The header is the QUESTION and nothing else. The reader is expected to draw
    their own conclusion from the marks, the axis and the key; a prose answer
    printed above the chart tells them what to think and dates badly the moment
    the numbers move. `subtitle` and `note` are accepted and ignored so the nine
    call sites can keep passing their computed strings — those still flow into
    FIGURES.md and the table twins, where prose belongs.

    `legend_rows` reserves footer space for a legend BELOW the plot; it cannot go
    above the axes, because that space is the question.
    """
    header_in = TITLE_Y + 0.34
    footer = FOOTER_IN + 0.30 * legend_rows
    total = header_in + plot_height_in + footer
    fig, ax = plt.subplots(figsize=(width_in, total), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(top=1 - header_in / total, bottom=footer / total,
                        left=0.30, right=0.97)
    fig._legend_anchor = 0.06 / total       # figure fraction, read by legend()
    fig.text(LEFT_FRAC, 1 - TITLE_Y / total, question, fontsize=12.5, color=INK,
             fontweight="bold", va="top")
    return fig, ax


def frame(ax, axis: str = "y") -> None:
    """Recessive chrome: hairline grid on the value axis only, no box, one
    baseline. `axis` is the axis the GRID runs along ('y' for column charts,
    'x' for horizontal bars)."""
    getattr(ax, f"{axis}axis").grid(True, color=GRID, linewidth=0.8)
    getattr(ax, f"{'x' if axis == 'y' else 'y'}axis").grid(False)
    ax.set_axisbelow(True)
    keep = "bottom" if axis == "y" else "left"
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(side == keep)
    ax.spines[keep].set_color(BASELINE)
    ax.spines[keep].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def round_ends(ax, bars, radius_pt: float = 4.0, horizontal: bool = False):
    """Round the DATA end of each bar; leave the baseline end square.

    FancyBboxPatch rounds in data units, so the radius is converted from points
    through the drawn axes size, and the x/y scale difference is carried by
    mutation_aspect — otherwise the corners smear on whichever axis is longer.

    FancyBboxPatch has no per-corner control, so the baseline end is squared by
    extending the patch one radius PAST the baseline and letting the axes clip
    it. That is why `clip_on=True` here is load-bearing, and why the axis limit
    must start at the baseline (xlim/ylim from 0).

    Pass only the OUTERMOST segment of a stacked bar. Rounding an interior
    segment rounds the join too, and the two curved edges meeting produce a
    pinched arrowhead where the chart should read as one continuous bar.

    A bar shorter than twice the radius is left square: rounding it would eat
    the whole mark and render small values as pills.
    """
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    dpi = ax.figure.dpi
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_per_pt = (x1 - x0) / (bbox.width / dpi * 72)
    y_per_pt = (y1 - y0) / (bbox.height / dpi * 72)
    rx, ry = radius_pt * x_per_pt, radius_pt * y_per_pt
    for p in bars:
        bb = p.get_bbox()
        extent = bb.width if horizontal else bb.height
        if extent < 2 * (rx if horizontal else ry):
            continue                       # too short to round — keep the rectangle
        p.set_visible(False)
        # grow backwards toward the baseline; the axes clip removes the overhang
        origin = (bb.xmin - rx, bb.ymin) if horizontal else (bb.xmin, bb.ymin - ry)
        w = bb.width + rx if horizontal else bb.width
        h = bb.height if horizontal else bb.height + ry
        ax.add_patch(FancyBboxPatch(
            origin, w, h,
            boxstyle=f"round,pad=0,rounding_size={rx}",
            mutation_aspect=ry / rx,
            facecolor=p.get_facecolor(), linewidth=0, clip_on=True,
            zorder=p.get_zorder()))


def legend(fig, labels, colors, ncol: int | None = None):
    """Flat legend in the footer, below the plot. Present whenever there are >=2
    series — the accessibility rule is that identity is never carried by colour
    alone. It goes below rather than above because the space above the axes is
    the question/answer header.
    """
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=7,
                          color=c) for c in colors]
    lg = fig.legend(handles, labels, loc="lower left",
                    bbox_to_anchor=(LEFT_FRAC, getattr(fig, "_legend_anchor", 0.02)),
                    ncol=ncol or len(labels), frameon=False, handletextpad=0.5,
                    columnspacing=1.6, fontsize=9)
    for t in lg.get_texts():
        t.set_color(SECOND)
    return lg


def save(fig, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return png


def table(out_dir: Path, stem: str, header_row, rows) -> Path:
    """The table twin. Every figure has one — it is the accessible view of the
    same numbers, and the relief for the sub-3:1 palette slot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header_row)
        w.writerows(rows)
    return path
