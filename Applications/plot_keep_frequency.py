"""Horizontal bar chart of framework frequency over the slimmed applications.

Counts DISTINCT applications per framework category, reading applications_slim.csv
directly so that grouped categories (e.g. the whole langchain_* ecosystem) de-dupe
repos that import several packages from the same family instead of double-counting.
`mcp` and every other name stay as their own category.

Writes pipeline/artifacts/keep_frequency.png.
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

SLIM_CSV = paths.ARTIFACTS_DIR / "applications_slim.csv"
OUT_PNG = paths.ARTIFACTS_DIR / "keep_frequency.png"

# name -> display category. Prefix groups collapse a whole ecosystem into one bar.
PREFIX_GROUPS = {"langchain": "langchain (ecosystem)"}


def category(name: str) -> str:
    for prefix, label in PREFIX_GROUPS.items():
        if name == prefix or name.startswith(prefix + "_"):
            return label
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25, help="show the top N categories")
    args = ap.parse_args()

    if not SLIM_CSV.exists():
        sys.exit(f"{SLIM_CSV} not found — run Applications/slim_applications.py first.")
    rows = list(csv.DictReader(open(SLIM_CSV, encoding="utf-8")))
    total = len(rows)

    counts = Counter()          # distinct apps per display category
    grouped_members = set()
    for r in rows:
        cats = set()
        for n in (r.get("matched_frameworks") or "").split(","):
            n = n.strip()
            if not n:
                continue
            c = category(n)
            cats.add(c)
            if c != n:
                grouped_members.add(n)
        for c in cats:          # each app counts once per category
            counts[c] += 1

    top = counts.most_common(args.top)
    names = [c for c, _ in top]
    vals = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(names))))
    y = range(len(names))
    colors = ["#E45756" if n == "langchain (ecosystem)" else "#4C78A8" for n in names]
    ax.barh(y, vals, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("distinct applications importing this framework")
    ax.set_title(f"Framework frequency (top {len(names)} categories)\n"
                 f"{total} slimmed applications  ·  langchain_* grouped")
    for i, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, i, str(v), va="center", fontsize=8)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)

    print(f"wrote {OUT_PNG}")
    if grouped_members:
        freq_csv = paths.ARTIFACTS_DIR / "keep_frequency.csv"
        naive = 0
        if freq_csv.exists():
            naive = sum(int(r["applications"])
                        for r in csv.DictReader(open(freq_csv, encoding="utf-8"))
                        if category(r["import_name"]) != r["import_name"])
        print(f"langchain (ecosystem): {counts['langchain (ecosystem)']} distinct apps "
              f"(naive per-name sum would be {naive}) across {len(grouped_members)} packages")


if __name__ == "__main__":
    main()
