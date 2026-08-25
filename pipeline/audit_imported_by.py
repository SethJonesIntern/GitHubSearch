"""Builds `pipeline/artifacts/imported_by.csv` — every population repo that ANOTHER
population repo imports, and the exact import token used.

The question this answers: which entries in our application list are actually being
consumed as libraries by other entries in the same list? A repo nobody imports is
behaving as a leaf application; a repo several others import is behaving as a framework.
This is the strongest framework-vs-application signal available, because it is
behavioural evidence from our own corpus rather than a guess from a name or a blurb.

One row per (published repo, import token) pair — a repo that publishes two importable
names gets two rows, so the token that carries the evidence is never hidden behind a
"best name" choice.

    py -3.14 -m pipeline.audit_imported_by

Reads the cached scan `audit_import_hits.csv` (importer_slug, name, files), so it does
NOT re-walk the clone tree. If that cache is stale, refresh it first with
`py -3.14 -m pipeline.audit_framework_check --scan` (~13 min).

Nothing here changes a statistic. Cutting a repo still means setting `in_scope=0` in
`application_audit.csv` by hand and adding a dated row to EXCLUSIONS.md.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "Wrapper"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import paths  # noqa: E402
from pipeline.audit_framework_check import (  # noqa: E402
    HITS_CSV,
    declared_names,
    name_relates_to_repo,
)

AUDIT_CSV = paths.ARTIFACTS_DIR / "application_audit.csv"
SLIM_CSV = paths.ARTIFACTS_DIR / "applications_slim.csv"
TRIAGE_CSV = paths.ARTIFACTS_DIR / "framework_triage.csv"
OUT_CSV = paths.ARTIFACTS_DIR / "imported_by.csv"

csv.field_size_limit(10 ** 9)

FIELDS = [
    "full_name", "github_url", "import_token",
    "importer_count", "importer_files", "importers",
    "token_shared_with", "passes_fork_check",
    "in_scope", "framework_suspect", "triage_band", "triage_score",
    "stars", "nd_tests", "llm_calls", "invokers_direct",
    "frameworks_imported", "description",
]


def _int(v) -> int:
    try:
        return int(float(str(v).strip() or 0))
    except ValueError:
        return 0


def main() -> None:
    with AUDIT_CSV.open(encoding="utf-8") as fh:
        audit = list(csv.DictReader(fh))
    with SLIM_CSV.open(encoding="utf-8") as fh:
        slim = {r["full_name"]: r for r in csv.DictReader(fh)}
    triage = {}
    if TRIAGE_CSV.exists():
        with TRIAGE_CSV.open(encoding="utf-8") as fh:
            triage = {r["full_name"]: r for r in csv.DictReader(fh)}

    if not HITS_CSV.exists():
        sys.exit(f"{HITS_CSV} not found — run:\n"
                 f"  py -3.14 -m pipeline.audit_framework_check --scan")

    # Needs the clones to read packaging metadata (see audit_framework_triage).
    on_disk = sum(1 for r in audit if (paths.REPOS_DIR / r["clone_slug"]).is_dir())
    if on_disk < len(audit) // 2:
        sys.exit(f"only {on_disk} of {len(audit)} clones present under "
                 f"{paths.REPOS_DIR}.\nThis needs the clone tree (gitignored) to read "
                 f"each repo's declared package name.")

    slug_to_repo = {r["clone_slug"]: r["full_name"] for r in audit}

    # token -> {importing repo -> files}. The cache records the IMPORTER, so a repo
    # importing its own package appears here too and must be subtracted below.
    hits: dict[str, dict[str, int]] = defaultdict(dict)
    with HITS_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            importer = slug_to_repo.get(r["importer_slug"], r["importer_slug"])
            hits[r["name"]][importer] = _int(r["files"])

    # token -> every repo declaring it. A token two repos claim identifies neither, so
    # it is reported with `token_shared_with` filled rather than silently attributed.
    declared = {r["full_name"]: declared_names(r["clone_slug"]) for r in audit}
    claimed: dict[str, list[str]] = defaultdict(list)
    for repo, names in declared.items():
        for n in names:
            claimed[n].append(repo)

    rows = []
    for a in audit:
        repo = a["full_name"]
        for token in declared[repo]:
            importers = {k: v for k, v in hits.get(token, {}).items() if k != repo}
            if not importers:
                continue                      # only repos something else imports
            t = triage.get(repo, {})
            rows.append({
                "full_name": repo,
                "github_url": (slim.get(repo, {}).get("html_url")
                               or f"https://github.com/{repo}"),
                "import_token": token,
                "importer_count": len(importers),
                "importer_files": sum(importers.values()),
                "importers": "; ".join(sorted(importers)),
                "token_shared_with": "; ".join(r for r in claimed[token] if r != repo),
                "passes_fork_check": int(name_relates_to_repo(token, repo)),
                "in_scope": a.get("in_scope", ""),
                "framework_suspect": a.get("framework_suspect", ""),
                "triage_band": t.get("band", ""),
                "triage_score": t.get("score", ""),
                "stars": _int(a["stars"]),
                "nd_tests": _int(a["nd_tests"]),
                "llm_calls": _int(a["llm_calls"]),
                "invokers_direct": _int(a["invokers_direct"]),
                "frameworks_imported": a.get("frameworks_imported", ""),
                "description": (a.get("description") or "").strip()[:300],
            })

    rows.sort(key=lambda r: (-r["importer_count"], -r["importer_files"], r["full_name"]))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

    repos = {r["full_name"] for r in rows}
    shared = [r for r in rows if r["token_shared_with"]]
    forky = [r for r in rows if not r["passes_fork_check"]]
    counted = [r for r in rows if not (r["in_scope"] or "").strip()]
    print(f"# wrote {OUT_CSV.relative_to(_ROOT)}")
    print(f"#   {len(rows)} (repo, token) pairs across {len(repos)} repos")
    print(f"#   {len(counted)} pairs are on repos still counted (in_scope blank)")
    print(f"#   {len(shared)} pairs use a token another repo ALSO declares "
          f"(identifies neither — check before cutting)")
    print(f"#   {len(forky)} pairs fail the fork check (token looks inherited from an "
          f"upstream project, e.g. a fork keeping its parent's pyproject)")
    print(f"#\n#   top by importer_count:")
    for r in rows[:12]:
        print(f"#     {r['importer_count']:>3} importers  {r['import_token']:<26} "
              f"{r['full_name']}")


if __name__ == "__main__":
    main()
