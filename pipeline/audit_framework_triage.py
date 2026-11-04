"""Builds `pipeline/artifacts/framework_triage.csv` — every population repo ranked by
how much it looks like a FRAMEWORK/LIBRARY rather than an APPLICATION, with the GitHub
link and the evidence behind each signal, so the call can be made by reading rather
than by inference.

Why this exists: the study measures state of practice in APPLICATIONS. A framework's
own repo (or an SDK, or a plugin host) has self-imports, a large test suite and example
code that inflate every metric — EXCLUSIONS.md §7 already cut 9 such repos and the
effect was large (agno alone: 3,963 -> 254 calls). `audit_framework_check.py` flags the
clear cases into `framework_suspect`; this sheet shows ALL 1,055 with every signal
exposed, because the remaining calls are judgement calls.

    py -3.14 -m pipeline.audit_framework_triage            # the sheet
    py -3.14 -m pipeline.audit_framework_triage --pypi     # + a live PyPI existence check

`score` is a RANKING AID, NOT A VERDICT. It only orders the reading queue; every
component is its own column so a row can be judged on evidence. Nothing here changes
any statistic — cutting a repo still means setting `in_scope=0` in
`application_audit.csv` by hand and adding a dated row to EXCLUSIONS.md.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "Wrapper"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import cuts  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.audit_framework_check import (  # noqa: E402  (reuse, never re-implement)
    HITS_CSV,
    declared_names,
    name_relates_to_repo,
    owner_matches_framework,
    stage1_frameworks,
)

AUDIT_CSV = paths.ARTIFACTS_DIR / "application_audit.csv"
SLIM_CSV = paths.ARTIFACTS_DIR / "applications_slim.csv"
OUT_CSV = paths.ARTIFACTS_DIR / "framework_triage.csv"

csv.field_size_limit(10 ** 9)

# Words an author uses to describe a thing OTHER people build on. Deliberately does not
# include "agent"/"ai"/"llm" — those describe applications just as often.
DESC_RE = re.compile(
    r"\b(framework|library|toolkit|sdk|plugin|middleware|wrapper|"
    r"client library|python package|pip install|api client|"
    r"boilerplate|scaffold|starter|template|building blocks?)\b", re.I)

# Repos analyze.py drops on grounds OTHER than the audit sheet's in_scope column, so
# they carry a blank in_scope but are not counted anywhere. Kept in sync with
# analyze.py's QUALITY_EXCLUDED / NOT_LLM_APP (EXCLUSIONS.md §1).
ANALYZE_HARD_EXCLUDED = {
    "rush86999/atom",                      # 31 unparseable files
    "Sumanth077/Hands-On-AI-Engineering",  # 56 unparseable files
    "sunnypilot/sunnypilot",               # agno name-collision, not an LLM app
}

FIELDS = [
    "rank", "score", "band", "full_name", "github_url",
    "imported_by", "imported_as", "importer_examples",
    "declares_package", "stage1_framework", "owner_matches", "desc_keyword",
    "stars", "contributors", "total_functions",
    "llm_calls", "invokers_direct", "nd_tests", "nd_tests_direct",
    "frameworks_imported", "in_scope", "framework_suspect", "pypi",
    "description", "score_breakdown",
]


def _int(v) -> int:
    try:
        return int(float(str(v).strip() or 0))
    except ValueError:
        return 0


def pypi_exists(names: list[str], workers: int = 16) -> dict[str, bool]:
    """Live check: is a declared name actually published on PyPI? The strongest single
    library signal, but opt-in — it is one request per declared name."""
    import urllib.error
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    def probe(name: str) -> tuple[str, bool]:
        try:
            req = urllib.request.Request(
                f"https://pypi.org/pypi/{name}/json", method="HEAD",
                headers={"User-Agent": "GitHubSearch-audit"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return name, resp.status == 200
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            return name, False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(probe, names))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pypi", action="store_true",
                    help="also check PyPI for each declared package name (network)")
    args = ap.parse_args()

    with AUDIT_CSV.open(encoding="utf-8") as fh:
        audit = list(csv.DictReader(fh))
    with SLIM_CSV.open(encoding="utf-8") as fh:
        slim = {r["full_name"]: r for r in csv.DictReader(fh)}

    # who imports what, from the cached scan (no tree walk)
    slug_to_repo = {r["clone_slug"]: r["full_name"] for r in audit}
    importers: dict[str, set[str]] = defaultdict(set)
    if HITS_CSV.exists():
        with HITS_CSV.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                importers[r["name"]].add(
                    slug_to_repo.get(r["importer_slug"], r["importer_slug"]))
    else:
        print(f"# WARNING {HITS_CSV.name} missing — imported_by will be 0 for every row."
              f"\n#         run: py -3.14 -m pipeline.audit_framework_check --scan",
              file=sys.stderr)

    stage1 = stage1_frameworks()

    # This script READS THE CLONES (for packaging metadata), and pipeline/repos is
    # gitignored — so on a machine that only has the git checkout, every declared name
    # comes back empty and the sheet silently becomes all-zeros rather than failing.
    # Refuse instead: a wrong sheet is worse than no sheet.
    on_disk = sum(1 for r in audit
                  if (paths.REPOS_DIR / r["clone_slug"]).is_dir())
    if on_disk < len(audit) // 2:
        sys.exit(
            f"only {on_disk} of {len(audit)} clones are present under "
            f"{paths.REPOS_DIR}.\nThis script needs the clone tree (~107 GB, gitignored) "
            f"to read each repo's packaging metadata.\nRun it on the machine that has "
            f"the clones; elsewhere just read the committed framework_triage.csv.")

    # declared_names touches only pyproject/setup.py/setup.cfg at each repo ROOT, so
    # this is ~3 stats per repo, not a walk of the 107 GB tree.
    declared = {r["full_name"]: declared_names(r["clone_slug"]) for r in audit}

    pypi: dict[str, bool] = {}
    if args.pypi:
        every = sorted({n for ns in declared.values() for n in ns})
        print(f"# checking {len(every)} declared names against PyPI ...")
        pypi = pypi_exists(every)
        print(f"#   {sum(pypi.values())} of {len(every)} exist on PyPI")

    rows = []
    for a in audit:
        repo = a["full_name"]
        names = declared[repo]

        best_name, best = "", set()
        for n in names:
            others = importers.get(n, set()) - {repo, a["clone_slug"]}
            if len(others) > len(best):
                best_name, best = n, others

        owner_hit = owner_matches_framework(repo, a["matched_frameworks"])
        desc = (a.get("description") or "").strip()
        kw = DESC_RE.search(desc)
        on_pypi = [n for n in names if pypi.get(n)]

        # Transparent additive score. Each term is also its own column, so a row can be
        # re-judged without trusting the total.
        pts: list[str] = []
        score = 0
        if len(best) >= 2:
            score += 3
            pts.append("imported_by>=2:+3")
        elif len(best) == 1:
            score += 1
            pts.append("imported_by==1:+1")
        if repo in stage1:
            score += 3
            pts.append("stage1_framework:+3")
        if owner_hit:
            score += 2
            pts.append("owner_matches:+2")
        if on_pypi:
            score += 2
            pts.append("on_pypi:+2")
        # A declared, importable name that plausibly belongs to this repo means the
        # author is publishing something. Weak alone — plenty of apps are packaged.
        if any(name_relates_to_repo(n, repo) for n in names):
            score += 1
            pts.append("declares_own_package:+1")
        if kw:
            score += 1
            pts.append(f"desc:{kw.group(0).lower()}:+1")

        band = ("strong" if score >= 5 else "likely" if score >= 3
                else "weak" if score >= 1 else "-")

        rows.append({
            "score": score, "band": band, "full_name": repo,
            "github_url": (slim.get(repo, {}).get("html_url")
                           or f"https://github.com/{repo}"),
            "imported_by": len(best), "imported_as": best_name,
            "importer_examples": ", ".join(sorted(best)[:3]),
            "declares_package": ", ".join(names),
            "stage1_framework": int(repo in stage1),
            "owner_matches": owner_hit or "",
            "desc_keyword": kw.group(0).lower() if kw else "",
            "stars": _int(a["stars"]), "contributors": _int(a["contributors"]),
            "total_functions": _int(a["total_functions"]),
            "llm_calls": _int(a["llm_calls"]),
            "invokers_direct": _int(a["invokers_direct"]),
            "nd_tests": _int(a["nd_tests"]),
            "nd_tests_direct": _int(a["nd_tests_direct"]),
            "frameworks_imported": a.get("frameworks_imported", ""),
            "in_scope": a.get("in_scope", ""),
            "framework_suspect": a.get("framework_suspect", ""),
            "pypi": ", ".join(on_pypi) if args.pypi else "",
            "description": desc[:300],
            "score_breakdown": "; ".join(pts),
        })

    rows.sort(key=lambda r: (-r["score"], -r["nd_tests"], r["full_name"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

    # Report on the LIVE population only. A blank `in_scope` is NOT sufficient: two
    # repos are dropped by analyze.py's quality filter and one as a known non-LLM app
    # while still carrying a blank in_scope, so scoping on the column alone overstates
    # every total. Mirror exactly what analyze.py excludes.
    excluded = set(cuts.excluded_from_stats()) | ANALYZE_HARD_EXCLUDED
    live = [r for r in rows if r["full_name"] not in excluded]
    print(f"# wrote {OUT_CSV.relative_to(_ROOT)}  ({len(rows)} repos)")
    print(f"#   population {len(rows)} -> {len(live)} LIVE "
          f"({len(rows) - len(live)} already cut / uncovered / quality-excluded)")
    print(f"#   {'band':<8}{'repos':>6}{'ND tests':>11}{'LLM calls':>11}   (live only)")
    for b in ("strong", "likely", "weak", "-"):
        sel = [r for r in live if r["band"] == b]
        print(f"#   {b:<8}{len(sel):>6}{sum(r['nd_tests'] for r in sel):>11,}"
              f"{sum(r['llm_calls'] for r in sel):>11,}")
    sl = [r for r in live if r["band"] in ("strong", "likely")]
    nd, calls = sum(r["nd_tests"] for r in sl), sum(r["llm_calls"] for r in sl)
    tot_nd = sum(r["nd_tests"] for r in live) or 1
    tot_c = sum(r["llm_calls"] for r in live) or 1
    print(f"#\n#   strong+likely still counted: {len(sl)} repos, {nd:,} ND tests, "
          f"{calls:,} LLM calls")
    print(f"#   = {100 * nd / tot_nd:.1f}% of live ND tests and "
          f"{100 * calls / tot_c:.1f}% of live LLM calls")


if __name__ == "__main__":
    main()
