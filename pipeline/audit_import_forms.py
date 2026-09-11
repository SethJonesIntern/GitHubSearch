"""Per-import-name repo counts and the LLM calls those repos make.

Default output (import_name_calls.csv): for each KEEP name, how many cloned repos
actually import it, how many of those record zero LLM calls, how many record calls,
and how many calls in total. `--forms` instead emits the three-way split of the
Stage-2 query forms (`from X import` / `from X.` / `import X`).

CAVEAT on total_calls: llm_calls is a *repo-level* count, not attributable to one
import name. A repo importing langchain_core and langchain_openai contributes its
full call count to both rows, so total_calls double-counts across names and must
never be summed down the column.

Below, the original note on the form split, which the --forms output still carries.

Per-import-name match counts, split by the three Stage-2 query forms.

Stage 2 searched every one of the 539 Stage-1 import names with three quoted
code-search queries -- `from X import`, `from X.`, `import X` (see
Applications.search_candidates.import_patterns) -- but recorded only the *union*:
`framework_repo_counts[name]` is the distinct repos across all three forms, and the
per-form attribution was never stored. So the form split cannot be recovered from
the search record; it has to be re-measured on the clones.

This pass walks the local clone tree and, for every name, counts the repos and files
matching each form structurally. The two halves answer different questions and must
not be conflated:

  search_*     GitHub code search over all of GitHub, all three forms unioned
               (13,505 repos enriched). No form split exists.
  scan_*       this pass: the cloned population repos only, form-attributed.
               Structural (anchored at line start), so it does not reproduce GitHub's
               substring tokenisation -- it is the *cleaner* measure, not the same one.

Reads : name_classification.csv, import_name_audit.csv, .search_progress.json,
        applications_slim.csv, application_audit.csv, the clone tree
Writes: pipeline/artifacts/import_name_calls.csv (default)
        pipeline/artifacts/import_name_forms.csv (--forms)
"""
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from pipeline import paths  # noqa: E402

CLASSIFICATION_CSV = paths.ARTIFACTS_DIR / "name_classification.csv"
NAME_AUDIT_CSV = paths.ARTIFACTS_DIR / "import_name_audit.csv"
PROGRESS_JSON = paths.ARTIFACTS_DIR / ".search_progress.json"
SLIM_CSV = paths.ARTIFACTS_DIR / "applications_slim.csv"
AUDIT_CSV = paths.ARTIFACTS_DIR / "application_audit.csv"
FORMS_CSV = paths.ARTIFACTS_DIR / "import_name_forms.csv"    # --forms
OUT_CSV = paths.ARTIFACTS_DIR / "import_name_calls.csv"      # default

# Same exclusions as audit_imports.scan -- vendored/installed trees are not the
# application's own imports.
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".tox", "build",
    "dist", "site-packages", ".mypy_cache", ".pytest_cache", ".ruff_cache", "vendor",
    "third_party", "3rdparty", ".idea", ".vscode", "eggs", ".eggs", ".next",
}
MAX_FILE_BYTES = 1_500_000

FORMS = ("from_import", "from_dot", "import")

# KEEP-set names that post-date the Stage-1 derivation, so they have no row in
# name_classification.csv: blind-spot frameworks recovered by targeted re-search
# (slim_applications.EXPLICIT_KEEP). They were never searched by Stage 2, so their
# search_* columns are empty -- but they must still be scanned and reported.
EXTRA_NAMES = ["lavague", "solace_agent_mesh", "assetopsbench_mcp", "assetopsbench"]

_RE = None


def load_names() -> list:
    with open(CLASSIFICATION_CSV, encoding="utf-8") as fh:
        names = [r["import_name"] for r in csv.DictReader(fh)]
    return names + [n for n in EXTRA_NAMES if n not in set(names)]


def build_regex(names):
    """One alternation over all names, longest-first so `langchain_core` wins over
    `langchain`. Three forms, mapped 1:1 onto the three Stage-2 queries:
        from X import ...   ->  from_import
        from X.sub import   ->  from_dot
        import X[.sub][ as] ->  import
    """
    alt = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
    return re.compile(
        r"^[ \t]*(?P<kw>from|import)[ \t]+(?P<name>" + alt +
        r")(?P<tail>[ \t]*\.|[ \t]+import\b|[ \t]+as\b|[ \t]*,|[ \t]*$)", re.M)


def _init(names):
    global _RE
    _RE = build_regex(names)


def scan(slug: str):
    """-> (slug, {(name, form): files}). One walk, never raises."""
    hits = Counter()
    root = paths.REPOS_DIR / slug
    if not root.is_dir():
        return slug, hits
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(p) > MAX_FILE_BYTES:
                        continue
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    continue
                per_file = set()
                for m in _RE.finditer(text):
                    kw, name, tail = m.group("kw"), m.group("name"), m.group("tail")
                    if kw == "import":
                        form = "import"
                    elif tail.strip().startswith("."):
                        form = "from_dot"
                    else:
                        form = "from_import"
                    per_file.add((name, form))
                hits.update(per_file)          # file-level, not line-level
    except Exception:                          # noqa: BLE001 -- one bad clone, not a dead run
        pass
    return slug, hits


def load_keep_set():
    import importlib.util
    sys.path.insert(0, str(_ROOT / "Applications"))
    sys.path.insert(0, str(_ROOT / "Wrapper"))
    spec = importlib.util.spec_from_file_location(
        "slimapps", _ROOT / "Applications" / "slim_applications.py")
    slim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slim)
    return slim.load_keep_set()


def write_calls_csv(names, repo_hits, keep_only=True):
    """import_name -> repos importing it, and what those repos' call counts look like.

    A repo with a blank llm_calls was never analysed (clone_failed / not processed);
    counting it as zero would invent evidence, so it gets its own column.
    """
    audit = {r["clone_slug"]: r for r in
             csv.DictReader(open(AUDIT_CSV, encoding="utf-8"))}
    keep = load_keep_set()

    # Denominator: the clone tree IS the population (1,055 repos), so repos_importing
    # is already "out of the original population" and pct_of_population says so.
    population = len([d for d in paths.REPOS_DIR.iterdir() if d.is_dir()])

    fields = ["import_name", "repos_importing", "pct_of_population", "repos_zero_calls",
              "repos_with_calls", "repos_unanalyzed", "total_calls", "median_calls",
              "max_calls"]
    rows = []
    for name in names:
        if keep_only and name not in keep:
            continue
        slugs = set()
        for f in FORMS:
            slugs |= repo_hits.get((name, f), set())
        zero = with_calls = unanalyzed = 0
        counts = []
        for s in slugs:
            raw = (audit.get(s, {}).get("llm_calls") or "").strip()
            if raw == "":
                unanalyzed += 1
            elif int(raw) == 0:
                zero += 1
            else:
                with_calls += 1
                counts.append(int(raw))
        counts.sort()
        rows.append({
            "import_name": name,
            "repos_importing": len(slugs),
            "pct_of_population": round(100 * len(slugs) / population, 1),
            "repos_zero_calls": zero,
            "repos_with_calls": with_calls,
            "repos_unanalyzed": unanalyzed,
            "total_calls": sum(counts),
            "median_calls": (counts[len(counts) // 2] if len(counts) % 2
                             else (counts[len(counts) // 2 - 1] +
                                   counts[len(counts) // 2]) // 2) if counts else 0,
            "max_calls": counts[-1] if counts else 0,
        })
    rows.sort(key=lambda r: (-r["repos_importing"], -r["total_calls"], r["import_name"]))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_csv(names, repo_hits, file_hits, keep_only=True):
    cls = {r["import_name"]: r for r in
           csv.DictReader(open(CLASSIFICATION_CSV, encoding="utf-8"))}
    audit = {r["import_name"]: r for r in
             csv.DictReader(open(NAME_AUDIT_CSV, encoding="utf-8"))}
    prog = json.load(open(PROGRESS_JSON, encoding="utf-8"))
    files_searched = prog.get("framework_file_matches", {})

    pop = Counter()
    for r in csv.DictReader(open(SLIM_CSV, encoding="utf-8")):
        for n in (r.get("matched_frameworks") or "").split(","):
            n = n.strip()
            if n:
                pop[n] += 1
    keep = load_keep_set()

    fields = ["import_name", "source_frameworks", "auto_bucket", "reason",
              "kept_for_population", "search_repos_matched", "search_files_matched",
              "candidates_kept", "population_repos", "scan_repos_any",
              "scan_repos_from_import", "scan_repos_from_dot", "scan_repos_import",
              "scan_files_from_import", "scan_files_from_dot", "scan_files_import"]
    rows = []
    for name in names:
        if keep_only and name not in keep:
            continue
        c, a = cls.get(name, {}), audit.get(name, {})
        any_repos = set()
        for f in FORMS:
            any_repos |= repo_hits.get((name, f), set())
        rows.append({
            "import_name": name,
            "source_frameworks": c.get("source_frameworks", ""),
            "auto_bucket": c.get("auto_bucket", "KEEP"),
            "reason": c.get("reason", "recovered_by_re_search"),
            "kept_for_population": "1" if name in keep else "0",
            "search_repos_matched": a.get("distinct_repos_matched", c.get("distinct_repos", "")),
            "search_files_matched": files_searched.get(name, ""),
            "candidates_kept": a.get("kept_candidates", c.get("kept_candidates", "")),
            "population_repos": pop.get(name, 0),
            "scan_repos_any": len(any_repos),
            **{"scan_repos_" + f: len(repo_hits.get((name, f), set())) for f in FORMS},
            **{"scan_files_" + f: file_hits.get((name, f), 0) for f in FORMS},
        })
    rows.sort(key=lambda r: (-int(r["population_repos"] or 0), -r["scan_repos_any"],
                             r["import_name"]))
    with open(FORMS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    argv = sys.argv[1:]
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 8
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    # Default: only the KEEP names -- the set that actually moved repos from
    # framework to application. --all also reports the 365 names Stage 3 dropped.
    keep_only = "--all" not in argv
    forms = "--forms" in argv

    names = load_names()
    slugs = sorted(d.name for d in paths.REPOS_DIR.iterdir() if d.is_dir())
    if limit:
        slugs = slugs[:limit]
    print("# scanning %d clones for %d import names x 3 forms, %d workers"
          % (len(slugs), len(names), workers), file=sys.stderr)

    start = time.time()
    repo_hits = defaultdict(set)     # (name, form) -> {slug}
    file_hits = Counter()            # (name, form) -> files
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(names,)) as pool:
        futures = [pool.submit(scan, s) for s in slugs]
        for done, fut in enumerate(as_completed(futures), 1):
            slug, hits = fut.result()
            for key, n in hits.items():
                repo_hits[key].add(slug)
                file_hits[key] += n
            if done % 100 == 0 or done == len(slugs):
                rate = done / max(time.time() - start, 1e-9)
                print("#   %d/%d  (%.1f/s, %.1f min left)"
                      % (done, len(slugs), rate,
                         (len(slugs) - done) / max(rate, 1e-9) / 60), file=sys.stderr)

    n = (write_csv(names, repo_hits, file_hits, keep_only=keep_only) if forms
         else write_calls_csv(names, repo_hits, keep_only=keep_only))
    print("# wrote %s  (%d rows, %ds)" % (OUT_CSV, n, time.time() - start),
          file=sys.stderr)


if __name__ == "__main__":
    main()
