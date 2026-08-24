"""Stage 8 — map each LLM test to the file/callable it appears to test.

Follows the PyMethod2Test / Methods2Test practice: recover the *focal* unit of a
test from naming convention alone, then treat the call graph as an independent
check rather than as input. Java gets a lot of mileage out of `testFoo -> foo`;
Python test names are prose (`test_agent_calls_retrieve_and_returns_answer`), so
the file level carries most of the signal and the method level needs a cascade.

Cascade, recorded per row so any rung can be filtered out downstream:

  FILE   test file -> focal module
    F1_exact    test_foo.py             -> foo.py
    F2_prefix   test_foo_bar_caching.py -> foo_bar.py, else foo.py (longest wins)
    F3_class    test_llm_manager.py     -> file defining class LLMManager
    F4_package  test_foo.py             -> foo/__init__.py
  METHOD test function -> focal callable (only where a focal file was found)
    M1_exact        test_bar            -> bar in the focal file
    M2_prefix       test_bar_on_empty   -> bar in the focal file
    M3_class_*      TestFoo.test_bar    -> Foo.bar
    M4_global_*     test_bar            -> the repo's only `bar`

Ambiguous candidates are ranked by path proximity to the test file; where that
still ties, an import of the candidate by the test file breaks it (flagged in
`import_tiebreak` so the effect stays measurable).

READ-ONLY over existing artifacts. The single file this writes is FOCAL_MAP_CSV,
a new artifact; no existing CSV is ever opened for write.

    python -m pipeline.focal_map                # all repos in llm_tests_all.csv
    python -m pipeline.focal_map --limit 20     # first 20 repos, for a smoke run
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pipeline.paths import ARTIFACTS_DIR, LLM_TESTS_CSV, REPOS_DIR

FOCAL_MAP_CSV = ARTIFACTS_DIR / "focal_map_llm_tests.csv"

FIELDS = [
    "repo", "test_qname", "test_file", "kind",
    "focal_file", "file_strategy", "file_candidates", "import_tiebreak",
    "focal_callable", "method_strategy",
]

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".tox",
             "build", "dist", "site-packages", ".mypy_cache", ".pytest_cache"}


# ── naming helpers ────────────────────────────────────────────────────────────

def is_test_file(name: str) -> bool:
    return name.startswith("test_") or name.endswith("_test.py")


def strip_test_stem(name: str):
    """test_foo.py -> 'foo'; foo_test.py -> 'foo'; anything else -> None."""
    if name.startswith("test_"):
        return name[len("test_"):-3]
    if name.endswith("_test.py"):
        return name[:-len("_test.py")]
    return None


def prefixes(stem: str):
    """'a_b_c' -> ['a_b_c', 'a_b', 'a'] — longest first, so the most specific
    focal name that actually exists wins."""
    parts = stem.split("_")
    return ["_".join(parts[:i]) for i in range(len(parts), 0, -1)]


def normalize(name: str) -> str:
    """Fold a name for case/underscore-insensitive comparison, so the stem
    `llm_manager` can match the class `LLMManager`."""
    return name.replace("_", "").lower()


def test_class_stem(cls: str):
    """TestFoo / FooTest / FooTests / FooTestCase -> 'Foo'."""
    if cls.startswith("Test"):
        return cls[4:]
    if cls.endswith("TestCase"):
        return cls[:-8]
    if cls.endswith("Tests"):
        return cls[:-5]
    if cls.endswith("Test"):
        return cls[:-4]
    return None


# ── per-repo index ────────────────────────────────────────────────────────────

class RepoIndex:
    """Everything the matcher needs from one clone, from a single AST pass."""

    def __init__(self, slug: str):
        self.slug = slug
        self.by_basename = defaultdict(list)      # 'foo.py' -> [rel, ...]
        self.pkg_init = defaultdict(list)         # 'foo'    -> [rel of foo/__init__.py]
        self.classes = defaultdict(list)          # normalized name -> [(rel, cls)]
        self.classes_in_file = defaultdict(list)  # rel -> [cls]
        self.funcs_in_file = defaultdict(list)    # rel -> [(name, cls_or_None)]
        self.func_global = defaultdict(list)      # name -> [(rel, cls_or_None)]
        self.test_imports = {}                    # rel(test) -> {dotted imports}
        self.parse_errors = 0


def _record_defs(tree, rel: str, idx: RepoIndex) -> None:
    """Walk defs, tracking the enclosing class so methods keep their owner."""
    def visit(node, cls):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                idx.classes[normalize(child.name)].append((rel, child.name))
                idx.classes_in_file[rel].append(child.name)
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                idx.funcs_in_file[rel].append((child.name, cls))
                idx.func_global[child.name].append((rel, cls))
                visit(child, cls)   # nested defs keep the same owning class
            else:
                visit(child, cls)
    visit(tree, None)


def build_index(slug: str) -> RepoIndex:
    """Parse every .py file in the clone once. Test files contribute only their
    imports (the tie-break signal); source files contribute the focal universe."""
    idx = RepoIndex(slug)
    for path in (REPOS_DIR / slug).rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            rel = path.relative_to(REPOS_DIR).as_posix()
        except ValueError:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            idx.parse_errors += 1     # one bad file must not sink the repo
            continue

        if is_test_file(path.name):
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imported.add(a.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
                    for a in node.names:
                        imported.add(f"{node.module}.{a.name}")
            idx.test_imports[rel] = imported
            continue

        idx.by_basename[path.name].append(rel)
        if path.name == "__init__.py":
            idx.pkg_init[path.parent.name].append(rel)
        _record_defs(tree, rel, idx)
    return idx


# ── candidate ranking ─────────────────────────────────────────────────────────

def path_score(test_rel: str, cand_rel: str):
    """Rank candidates for a test file: most shared leading directories first,
    then shallowest. Encodes 'the nearest module is the one under test'."""
    a, b = test_rel.split("/")[:-1], cand_rel.split("/")[:-1]
    shared = 0
    for x, y in zip(a, b):
        if x != y:
            break
        shared += 1
    return (shared, -len(b))


def pick(test_rel: str, cands, imports):
    """-> (best, n_candidates, broken_by_import)."""
    if not cands:
        return None, 0, False
    if len(cands) == 1:
        return cands[0], 1, False
    ranked = sorted(cands, key=lambda c: path_score(test_rel, c), reverse=True)
    top = path_score(test_rel, ranked[0])
    tied = [c for c in ranked if path_score(test_rel, c) == top]
    if len(tied) > 1 and imports:
        mods = {i.replace(".", "/") for i in imports}
        hit = [c for c in tied if any(c[:-3].endswith(m) for m in mods)]
        if len(hit) == 1:
            return hit[0], len(cands), True
    return tied[0], len(cands), False


# ── the two matching levels ───────────────────────────────────────────────────

def match_file(test_rel: str, idx: RepoIndex):
    """-> (focal_rel, strategy, n_candidates, import_tiebreak)."""
    stem = strip_test_stem(test_rel.rsplit("/", 1)[-1])
    if not stem:
        return None, "not_conventional", 0, False
    imports = idx.test_imports.get(test_rel, set())

    if idx.by_basename.get(stem + ".py"):
        best, n, tb = pick(test_rel, idx.by_basename[stem + ".py"], imports)
        return best, "F1_exact", n, tb
    for pref in prefixes(stem)[1:]:
        if idx.by_basename.get(pref + ".py"):
            best, n, tb = pick(test_rel, idx.by_basename[pref + ".py"], imports)
            return best, "F2_prefix", n, tb
    for pref in prefixes(stem):
        if idx.classes.get(normalize(pref)):
            rels = [r for r, _ in idx.classes[normalize(pref)]]
            best, n, tb = pick(test_rel, rels, imports)
            return best, "F3_class", n, tb
    for pref in prefixes(stem):
        if idx.pkg_init.get(pref):
            best, n, tb = pick(test_rel, idx.pkg_init[pref], imports)
            return best, "F4_package", n, tb
    return None, "no_match", 0, False


def match_method(test_qname: str, focal_rel, idx: RepoIndex):
    """-> (focal_callable, strategy). Callable is 'rel/path.py::[Class.]name'."""
    leaf = test_qname.rsplit(".", 1)[-1]
    if not leaf.startswith("test_") or leaf == "test_":
        return None, "not_conventional"
    stem = leaf[len("test_"):]
    segs = test_qname.split(".")
    cls_stem = test_class_stem(segs[-2]) if len(segs) >= 2 else None

    if focal_rel:
        by_name = defaultdict(list)
        for name, cls in idx.funcs_in_file.get(focal_rel, []):
            by_name[name].append(cls)

        # M3 — the test class names the focal class, so scope the lookup to it.
        if cls_stem:
            target = next((c for c in idx.classes_in_file.get(focal_rel, [])
                           if normalize(c) == normalize(cls_stem)), None)
            if target:
                for pref in prefixes(stem):
                    if target in by_name.get(pref, []):
                        return (f"{focal_rel}::{target}.{pref}",
                                "M3_class_exact" if pref == stem else "M3_class_prefix")

        # M1/M2 — anything defined in the focal file.
        for i, pref in enumerate(prefixes(stem)):
            if pref in by_name:
                cls = by_name[pref][0]
                owner = f"{cls}." if cls else ""
                return f"{focal_rel}::{owner}{pref}", "M1_exact" if i == 0 else "M2_prefix"

    # M4 — no focal file (or nothing matched in it): accept a repo-wide unique name.
    for i, pref in enumerate(prefixes(stem)):
        hits = idx.func_global.get(pref, [])
        if len(hits) == 1:
            rel, cls = hits[0]
            owner = f"{cls}." if cls else ""
            return (f"{rel}::{owner}{pref}",
                    "M4_global_exact" if i == 0 else "M4_global_prefix")
        if hits:
            break   # name exists but is ambiguous; shorter prefixes only get worse
    return None, "no_match"


# ── driver ────────────────────────────────────────────────────────────────────

def process_repo(job):
    repo, slug, tests = job
    try:
        idx = build_index(slug)
    except Exception as exc:                      # noqa: BLE001 — keep the batch alive
        return repo, [], 0, f"{type(exc).__name__}: {exc}"

    per_file = {}
    rows = []
    for qname, test_rel, kind in tests:
        if test_rel not in per_file:              # file match is per file, not per test
            per_file[test_rel] = match_file(test_rel, idx)
        focal_rel, fstrat, ncand, tiebreak = per_file[test_rel]
        focal_call, mstrat = match_method(qname, focal_rel, idx)
        rows.append({
            "repo": repo, "test_qname": qname, "test_file": test_rel, "kind": kind,
            "focal_file": focal_rel or "", "file_strategy": fstrat,
            "file_candidates": ncand, "import_tiebreak": tiebreak,
            "focal_callable": focal_call or "", "method_strategy": mstrat,
        })
    return repo, rows, idx.parse_errors, None


def load_tests(limit=None):
    csv.field_size_limit(10**9)
    by_repo = defaultdict(list)
    with LLM_TESTS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_repo[row["repo"]].append((row["qname"], row["file"], row["kind"]))
    jobs = [(repo, tests[0][1].split("/", 1)[0], tests)
            for repo, tests in by_repo.items()]
    jobs.sort(key=lambda j: -len(j[2]))           # long poles first
    return jobs[:limit] if limit else jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only the N repos with the most LLM tests")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = ap.parse_args()

    jobs = load_tests(args.limit)
    print(f"# {sum(len(j[2]) for j in jobs)} LLM tests across {len(jobs)} repos")

    fstrat, mstrat = Counter(), Counter()
    files_seen, errors, parse_errors = set(), [], 0

    with FOCAL_MAP_CSV.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_repo, j) for j in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                repo, rows, perr, err = future.result()
                parse_errors += perr
                if err:
                    errors.append((repo, err))
                for row in rows:
                    writer.writerow(row)
                    fstrat[row["file_strategy"]] += 1
                    mstrat[row["method_strategy"]] += 1
                    files_seen.add((repo, row["test_file"]))
                if done % 50 == 0 or done == len(jobs):
                    print(f"#   {done}/{len(jobs)} repos", file=sys.stderr)

    total = sum(fstrat.values())
    print(f"\n# wrote {FOCAL_MAP_CSV}  ({total} rows, {len(files_seen)} test files)")
    for title, counter in (("FILE", fstrat), ("METHOD", mstrat)):
        print(f"\n# {title} strategy, by LLM test:")
        for key, n in counter.most_common():
            print(f"#   {key:17s} {n:7d}  {100 * n / total:5.1f}%")
    print(f"\n# files that failed to parse: {parse_errors}")
    if errors:
        print(f"# repos that errored: {len(errors)}")
        for repo, exc in errors[:5]:
            print(f"#   {repo}: {exc}")


if __name__ == "__main__":
    main()
