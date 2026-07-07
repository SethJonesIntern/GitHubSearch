"""Horizontal bar chart of framework frequency over the slimmed applications.

Counts DISTINCT applications per framework category, reading applications_slim.csv
directly so that grouped categories (e.g. the whole langchain_* ecosystem) de-dupe
repos that import several packages from the same family instead of double-counting.

Two views:
  (default)  every import name is its own bar     -> keep_frequency.png
  --group    multi-package ecosystems collapsed   -> keep_frequency_grouped.png
             into one framework bar (langchain,
             agent_framework, autogen)

Writes pipeline/artifacts/keep_frequency[_grouped].png.
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

# Ecosystems that ship as many separately-named packages but are one framework.
# prefix -> display label: matches `prefix` exactly or `prefix_*`.
ECOSYSTEM_PREFIXES = {
    "langchain": "langchain (ecosystem)",
    "agent_framework": "agent_framework (MS)",
    "autogen": "autogen (ecosystem)",
}
# Same-framework packages whose names don't share the prefix (caught explicitly).
ECOSYSTEM_ALIASES = {
    "pyautogen": "autogen (ecosystem)",
    "autogenstudio": "autogen (ecosystem)",
}
GROUP_LABELS = set(ECOSYSTEM_PREFIXES.values()) | set(ECOSYSTEM_ALIASES.values())


def category(name: str, group: bool) -> str:
    """Display category for an import name. With group=False every name is its own
    category; with group=True the ecosystem families collapse to one label."""
    if not group:
        return name
    if name in ECOSYSTEM_ALIASES:
        return ECOSYSTEM_ALIASES[name]
    for prefix, label in ECOSYSTEM_PREFIXES.items():
        if name == prefix or name.startswith(prefix + "_"):
            return label
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25, help="show the top N categories")
    ap.add_argument("--group", action="store_true",
                    help="collapse multi-package ecosystems (langchain, agent_framework, "
                         "autogen) into one framework bar each")
    args = ap.parse_args()

    out_png = paths.ARTIFACTS_DIR / (
        "keep_frequency_grouped.png" if args.group else "keep_frequency.png")

    if not SLIM_CSV.exists():
        sys.exit(f"{SLIM_CSV} not found — run Applications/slim_applications.py first.")
    rows = list(csv.DictReader(open(SLIM_CSV, encoding="utf-8")))
    total = len(rows)

    counts = Counter()                  # distinct apps per display category
    members: dict[str, set] = {}        # distinct import names folded into each label
    for r in rows:
        cats = set()
        for n in (r.get("matched_frameworks") or "").split(","):
            n = n.strip()
            if not n:
                continue
            c = category(n, args.group)
            cats.add(c)
            if c != n:
                members.setdefault(c, set()).add(n)
        for c in cats:          # each app counts once per category
            counts[c] += 1

    top = counts.most_common(args.top)
    names = [c for c, _ in top]
    vals = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(names))))
    y = range(len(names))
    colors = ["#E45756" if n in GROUP_LABELS else "#4C78A8" for n in names]
    ax.barh(y, vals, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("distinct applications importing this framework")
    unit = "frameworks" if args.group else "import names"
    subtitle = f"{total} slimmed applications"
    if args.group:
        subtitle += "  ·  langchain / agent_framework / autogen grouped"
    ax.set_title(f"Framework frequency (top {len(names)} {unit})\n{subtitle}")
    for i, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, i, str(v), va="center", fontsize=8)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)

    print(f"wrote {out_png}")
    for label in sorted(members, key=lambda l: -counts[l]):
        print(f"  {label}: {counts[label]} distinct apps across {len(members[label])} packages")


if __name__ == "__main__":
    main()
