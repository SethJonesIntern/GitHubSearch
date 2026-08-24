"""Framework-frequency table over the trustworthy (KEEP) import names.

Counts DISTINCT slimmed applications per framework. Many frameworks ship as a
fragmented family of packages (langchain + langchain_*, Microsoft's Agent
Framework as ~28 agent_framework_* packages, autogen_* + pyautogen, ...). Those
are ONE framework, so we group them into an ecosystem category and count distinct
apps (a repo importing several packages of one family counts once).

Reads : applications_slim.csv
Writes:
  keep_frequency.csv             grouped by ecosystem category (the headline table)
  keep_frequency_by_package.csv  raw per import_name (detail, ungrouped)
"""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths  # noqa: E402
from pipeline import cuts  # noqa: E402

SLIM_CSV = paths.ARTIFACTS_DIR / "applications_slim.csv"
OUT_CSV = paths.ARTIFACTS_DIR / "keep_frequency.csv"
OUT_BY_PKG = paths.ARTIFACTS_DIR / "keep_frequency_by_package.csv"

# Ecosystem grouping: a package name folds into its family's category when it
# equals the prefix or starts with "<prefix>_". langgraph is kept separate from
# langchain (distinct product, same org) per the project decision.
PREFIX_GROUPS = {
    "langchain": "langchain",
    "langgraph": "langgraph",
    "agent_framework": "agent_framework (MS)",
    "autogen": "autogen",
    "crewai": "crewai",
    "uagents": "uagents",
    "notte": "notte",
}
# Family members that don't share the prefix pattern. `pyautogen`/`autogenstudio`
# fold into autogen; `clai` is pydantic-ai's CLI module (its apps really import
# pydantic_ai). NOTE: cheshire-cat (`cat`/`agui`) and connectonion (`subagents`) were
# cut — collision-prone junk tokens that matched only apps' OWN submodules (e.g.
# deepagents.middleware.subagents), never the real package. See slim_applications.
EXTRA_MEMBERS = {
    "pyautogen": "autogen", "autogenstudio": "autogen",
    "clai": "pydantic_ai",
}


def category(name: str) -> str:
    if name in EXTRA_MEMBERS:
        return EXTRA_MEMBERS[name]
    for prefix, label in PREFIX_GROUPS.items():
        if name == prefix or name.startswith(prefix + "_"):
            return label
    return name


def _write(path, header, rows_iter):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_iter)


def main():
    if not SLIM_CSV.exists():
        sys.exit(f"{SLIM_CSV} not found — run Applications/slim_applications.py first.")
    rows = list(csv.DictReader(open(SLIM_CSV, encoding="utf-8")))
    before = len(rows)
    rows = cuts.drop_cut(rows)          # a cut repo must not inflate any framework
    if before != len(rows):
        print(f"[audit cuts] dropped {before - len(rows)} repos marked in_scope=0")
    total = len(rows)

    grouped = Counter()   # distinct apps per ecosystem category
    by_pkg = Counter()    # distinct apps per raw import name
    for r in rows:
        names = {n.strip() for n in (r.get("matched_frameworks") or "").split(",") if n.strip()}
        for n in names:
            by_pkg[n] += 1
        for c in {category(n) for n in names}:   # dedupe within a family
            grouped[c] += 1

    def pct(c):
        return f"{100 * c / total:.1f}" if total else "0"

    _write(OUT_CSV, ["framework", "applications", "pct_of_apps"],
           ([name, c, pct(c)] for name, c in grouped.most_common()))
    _write(OUT_BY_PKG, ["import_name", "applications", "pct_of_apps"],
           ([name, c, pct(c)] for name, c in by_pkg.most_common()))

    print(f"total slimmed applications: {total}")
    print(f"ecosystem categories: {len(grouped)}  (from {len(by_pkg)} import names)")
    print(f"wrote {OUT_CSV.name} + {OUT_BY_PKG.name}\n")
    print(f"{'applications':>12}  framework")
    for name, c in grouped.most_common(20):
        print(f"{c:>12}  {name}")


if __name__ == "__main__":
    main()
