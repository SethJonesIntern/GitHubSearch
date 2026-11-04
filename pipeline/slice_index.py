#!/usr/bin/env python
"""Stage 6a: flatten the per-repo Joern slices into a queryable index.

Stage 6 leaves one ``programs.jsonl`` per repo under ``artifacts/slices/``
(4.3 GB over 788 repos). Nothing reads it: ``batch_call_metadata`` writes it,
``audit_apps`` only checks that the directory exists. Every question we want to
ask of the slices ("which subprograms contain a framework call site", "which
slice bodies recur across repos", "how many slices are even parseable") is a
group-by over a table that did not exist. This builds that table.

One row per subprogram -- duplicates included, since the slicer dedupes only
within a source file and the duplicate multiplicity is itself a signal. The
code bodies move to a content store keyed by ``code_sha256`` so the index stays
small, and the ``pdg_lines`` lists go to a long-form table so they can be joined
against ``llm_calls_all.csv`` on (repo, file, line).

Nothing here re-slices or re-runs Joern; it only reads what Stage 6 emitted.

    py -3.14 -m pipeline.slice_index                 # full corpus
    py -3.14 -m pipeline.slice_index --limit 5       # smoke test
    py -3.14 -m pipeline.slice_index --no-code       # skip the content store
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from array import array
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from . import paths
from .engines import is_test_file, is_test_function

# -- outputs -------------------------------------------------------------------
SLICE_INDEX_CSV = paths.ARTIFACTS_DIR / "slice_index.csv"                 # 1 row / subprogram
SLICE_FILES_CSV = paths.ARTIFACTS_DIR / "slice_files.csv"                 # 1 row / sliced source file
SLICE_PDG_LINES_CSV_GZ = paths.ARTIFACTS_DIR / "slice_pdg_lines.csv.gz"   # long form, for the call join
SLICE_CODE_JSONL_GZ = paths.ARTIFACTS_DIR / "slice_code.jsonl.gz"         # content store, 1 row / unique body
SLICE_INDEX_SUMMARY_JSON = paths.ARTIFACTS_DIR / "slice_index_summary.json"

INDEX_FIELDS = [
    "slice_id",        # running int, stable within one build; the join key
    "repo",            # slices/ subdir name == clone dir name == repo slug
    "program",         # the slicer's name for the source file (its stem)
    "rel_path",        # source file, repo-relative, forward slashes
    "code_kind",       # test | test_support | production -- see code_kind()
    "slice_file",      # the slicer's filename for this subprogram
    "method",
    "variable",
    "criterion_mode",
    "is_duplicate",    # duplicate of an earlier subprogram in the SAME source file
    "dedup_file",      # canonical slice_file whose body this shares
    "n_criterion", "criterion_first", "criterion_last",
    "n_pdg", "pdg_first", "pdg_last", "pdg_span",
    "n_standalone",
    "n_inserted",      # standalone - pdg: lines added only to keep the snippet parseable
    "line_count",
    "code_sha256",
    "parses",          # recomputed here with compile(); the refined JSONL keeps only a file-level count
]

FILE_FIELDS = [
    "repo", "program", "rel_path", "file_kind",   # test | test_support | production
    "subprogram_count", "deduplicated_count", "duplicate_count",
    "syntax_error_count",   # as the slicer recorded it, per source file
    "n_methods", "n_variables",
    "n_rows",               # rows this build emitted for the file (sanity vs subprogram_count)
    "schema",
]

TEST_DIR_NAMES = {"test", "tests", "testing", "unittest", "unittests"}

CODE_KINDS = ("test", "test_support", "production")


def _rel_path(source_path: str, repo: str) -> str:
    """Repo-relative, forward slashes. source_path is the absolute Windows path
    into pipeline/repos/<repo>/ as it stood at slice time."""
    norm = str(source_path).replace("\\", "/")
    marker = "/repos/" + repo + "/"
    idx = norm.find(marker)
    if idx != -1:
        return norm[idx + len(marker):]
    # Fall back to anything after the repo name, then to the bare filename.
    marker = "/" + repo + "/"
    idx = norm.rfind(marker)
    if idx != -1:
        return norm[idx + len(marker):]
    return norm.rsplit("/", 1)[-1]


def _in_test_tree(rel_path: str) -> bool:
    parts = rel_path.split("/")
    return parts[-1] == "conftest.py" or any(p.lower() in TEST_DIR_NAMES for p in parts[:-1])


def file_kind(rel_path: str) -> str:
    """test | test_support | production, for a source file."""
    if is_test_file(rel_path):
        return "test"
    return "test_support" if _in_test_tree(rel_path) else "production"


def code_kind(rel_path: str, function: str) -> str:
    """test | test_support | production, for code in `function` of `rel_path`.

    ``test`` is exactly Stage 5's definition of a test -- the engines
    predicates that batch_call_metadata._tests_among applies: a pytest-
    discoverable file (``test_*.py`` / ``*_test.py``) AND a function named
    ``test_*``. Imported, not restated, so the slice split and the ND-test
    count can never drift apart.

    ``test_support`` is test code that is not a test: helpers and fixtures
    inside a test file, ``conftest.py``, and anything else under a test
    directory. Stage 5 would call it production; it is kept apart because a
    fixture that calls a model is test scaffolding, and counting it as
    production would contaminate any claim about application code.
    """
    if is_test_file(rel_path) and is_test_function(function or ""):
        return "test"
    if is_test_file(rel_path) or _in_test_tree(rel_path):
        return "test_support"
    return "production"


def _span(lines: List[int]) -> Tuple[int, str, str, str]:
    """(count, first, last, span), blank rather than 0 when the list is empty."""
    if not lines:
        return 0, "", "", ""
    lo, hi = lines[0], lines[-1]        # the slicer emits them sorted
    if lo > hi:                         # defensive; never observed
        lo, hi = min(lines), max(lines)
    return len(lines), str(lo), str(hi), str(hi - lo + 1)


def _parses(code: str) -> bool:
    try:
        compile(code, "<slice>", "exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return True


def _iter_program_records(jsonl: Path) -> Iterator[Tuple[Optional[Dict[str, Any]], Optional[str]]]:
    with jsonl.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                yield None, "{}:{}: {}".format(jsonl.parent.name, lineno, exc)
                continue
            if isinstance(rec, dict):
                yield rec, None
            else:
                yield None, "{}:{}: not an object".format(jsonl.parent.name, lineno)


def _quantiles(a: array) -> Dict[str, Any]:
    if not len(a):
        return {}
    s = sorted(a)

    def q(p: float) -> int:
        return s[min(len(s) - 1, int(p * len(s)))]

    return {"min": s[0], "p25": q(0.25), "median": q(0.50), "p75": q(0.75),
            "p90": q(0.90), "p99": q(0.99), "max": s[-1],
            "mean": round(sum(s) / len(s), 2)}


def build(slices_dir: Path, limit: Optional[int], only: Optional[str],
          write_code: bool, quiet: bool) -> Dict[str, Any]:
    started = time.time()
    repo_dirs = sorted(e.path for e in os.scandir(slices_dir) if e.is_dir())
    if only:
        needle = only.lower()
        repo_dirs = [d for d in repo_dirs if needle in Path(d).name.lower()]
    if limit:
        repo_dirs = repo_dirs[:limit]

    paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    slice_id = 0
    n_repos_with_jsonl = 0
    repos_without_jsonl: List[str] = []
    n_programs = 0
    n_rows = 0
    n_dup_rows = 0
    n_bodies_seen = 0
    n_parse_ok = 0
    n_parse_fail = 0
    n_no_body = 0
    positional_mismatches = 0   # dedup record out of step with slice_map (expect 0)
    unpaired_canonicals = 0     # more non-duplicate rows than dedup records (expect 0)
    hash_repo_count: Dict[str, int] = {}     # sha256 -> distinct repos emitting it
    hash_parses: Dict[str, bool] = {}
    schemas: Dict[str, int] = {}
    bad_lines: List[str] = []
    file_syntax_errors = 0
    file_subprogram_total = 0
    pdg_sizes = array("i")
    inserted_sizes = array("i")
    per_repo: Dict[str, Dict[str, int]] = {}
    kind_counts: Dict[str, int] = {k: 0 for k in CODE_KINDS}

    code_fh = gzip.open(SLICE_CODE_JSONL_GZ, "wt", encoding="utf-8", newline="\n") if write_code else None
    lines_fh = gzip.open(SLICE_PDG_LINES_CSV_GZ, "wt", encoding="utf-8", newline="")
    index_fh = SLICE_INDEX_CSV.open("w", encoding="utf-8", newline="")
    files_fh = SLICE_FILES_CSV.open("w", encoding="utf-8", newline="")
    try:
        index_w = csv.writer(index_fh)
        index_w.writerow(INDEX_FIELDS)
        files_w = csv.writer(files_fh)
        files_w.writerow(FILE_FIELDS)
        lines_w = csv.writer(lines_fh)
        lines_w.writerow(["slice_id", "line"])

        for i, repo_dir in enumerate(repo_dirs, 1):
            repo = Path(repo_dir).name
            jsonl = Path(repo_dir) / "programs.jsonl"
            if not jsonl.is_file():
                repos_without_jsonl.append(repo)
                continue
            n_repos_with_jsonl += 1
            repo_rows = 0
            repo_programs = 0
            repo_hashes: Set[str] = set()

            for rec, err in _iter_program_records(jsonl):
                if err or rec is None:
                    if err:
                        bad_lines.append(err)
                    continue
                n_programs += 1
                repo_programs += 1
                schema = str(rec.get("schema", "<none>"))
                schemas[schema] = schemas.get(schema, 0) + 1

                rel_path = _rel_path(rec.get("source_path") or rec.get("source_file") or "", repo)
                fkind = file_kind(rel_path)
                program = rec.get("program") or Path(rel_path).stem

                # Bodies live only on the deduplicated records. Pair them to the
                # non-duplicate slice_map rows BY POSITION, not by filename: the
                # slicer names a subprogram <program>_<method>_<variable>.py, so
                # two same-named functions in one source file collide (16 such
                # filenames corpus-wide) and a name lookup silently hands both
                # rows the same body. The two lists are built in the same pass,
                # so position is exact.
                dedup_list = [d for d in (rec.get("deduplicated_subprograms") or [])
                              if isinstance(d, dict)]
                # slice_file -> [(line_count, sha, code)] for canonicals seen so far,
                # so a duplicate can still be resolved through dedup_file.
                bodies_by_name: Dict[str, List[Tuple[Any, str, str]]] = {}
                d_i = 0

                rows = rec.get("slice_map")
                if not rows:   # pre-slice_map schema: the dedup records are all we have
                    rows = [dict(d, dedup_file=d.get("file"), is_duplicate=False)
                            for d in (rec.get("deduplicated_subprograms") or [])
                            if isinstance(d, dict)]

                methods: Set[str] = set()
                variables: Set[str] = set()
                file_rows = 0

                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    slice_file = item.get("file") or ""
                    dedup_file = item.get("dedup_file") or slice_file
                    is_dup = bool(item.get("is_duplicate"))

                    if is_dup:
                        sha, code = "", None
                        cands = bodies_by_name.get(dedup_file)
                        if cands:
                            lc = item.get("line_count")
                            pick = next((c for c in reversed(cands) if c[0] == lc), cands[-1])
                            sha, code = pick[1], pick[2]
                    else:
                        if d_i < len(dedup_list):
                            d = dedup_list[d_i]
                            d_i += 1
                            if d.get("file") != slice_file:
                                positional_mismatches += 1
                            sha, code = d.get("code_sha256") or "", d.get("code") or ""
                        else:
                            unpaired_canonicals += 1
                            sha, code = "", None
                        bodies_by_name.setdefault(slice_file, []).append(
                            (item.get("line_count"), sha, code))

                    if sha:
                        if sha not in hash_parses:
                            ok = _parses(code or "")
                            hash_parses[sha] = ok
                            hash_repo_count[sha] = 1
                            if code_fh is not None:
                                code_fh.write(json.dumps(
                                    {"code_sha256": sha,
                                     "line_count": item.get("line_count", 0),
                                     "parses": ok,
                                     "code": code or ""},
                                    ensure_ascii=False) + "\n")
                        elif sha not in repo_hashes:
                            hash_repo_count[sha] += 1
                        repo_hashes.add(sha)
                        n_bodies_seen += 1
                        if hash_parses[sha]:
                            n_parse_ok += 1
                            parses = "1"
                        else:
                            n_parse_fail += 1
                            parses = "0"
                    else:
                        n_no_body += 1
                        parses = ""

                    crit = item.get("criterion_lines") or []
                    pdg = item.get("pdg_lines") or []
                    standalone = item.get("standalone_lines") or []
                    n_crit, c_first, c_last, _ = _span(crit)
                    n_pdg, p_first, p_last, p_span = _span(pdg)
                    n_inserted = len(set(standalone) - set(pdg))

                    kind = code_kind(rel_path, item.get("method") or "")
                    kind_counts[kind] += 1
                    slice_id += 1
                    index_w.writerow([
                        slice_id, repo, program, rel_path, kind,
                        slice_file, item.get("method") or "", item.get("variable") or "",
                        item.get("criterion_mode") or "",
                        int(is_dup), dedup_file,
                        n_crit, c_first, c_last,
                        n_pdg, p_first, p_last, p_span,
                        len(standalone), n_inserted,
                        item.get("line_count", 0),
                        sha, parses,
                    ])
                    for ln in pdg:
                        lines_w.writerow([slice_id, ln])

                    pdg_sizes.append(n_pdg)
                    inserted_sizes.append(n_inserted)
                    if item.get("method"):
                        methods.add(item["method"])
                    if item.get("variable"):
                        variables.add(item["variable"])
                    if is_dup:
                        n_dup_rows += 1
                    file_rows += 1
                    n_rows += 1
                    repo_rows += 1

                files_w.writerow([
                    repo, program, rel_path, fkind,
                    rec.get("subprogram_count", ""), rec.get("deduplicated_count", ""),
                    rec.get("duplicate_count", ""), rec.get("syntax_error_count", ""),
                    len(methods), len(variables), file_rows, schema,
                ])
                if isinstance(rec.get("syntax_error_count"), int):
                    file_syntax_errors += rec["syntax_error_count"]
                if isinstance(rec.get("subprogram_count"), int):
                    file_subprogram_total += rec["subprogram_count"]

            per_repo[repo] = {"programs": repo_programs, "subprograms": repo_rows,
                              "unique_bodies": len(repo_hashes)}
            if not quiet and (i % 25 == 0 or i == len(repo_dirs)):
                print("  [{}/{}] {}: {} rows ({} total)".format(
                    i, len(repo_dirs), repo, repo_rows, n_rows), file=sys.stderr, flush=True)
    finally:
        index_fh.close()
        files_fh.close()
        lines_fh.close()
        if code_fh is not None:
            code_fh.close()

    def _pct(n: int, d: int) -> float:
        return round(100.0 * n / d, 2) if d else 0.0

    n_unique = len(hash_parses)
    unique_ok = sum(1 for v in hash_parses.values() if v)
    multi_repo_bodies = sum(1 for c in hash_repo_count.values() if c > 1)
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "slices_dir": str(slices_dir),
        "elapsed_seconds": round(time.time() - started, 1),
        "repo_dirs_seen": len(repo_dirs),
        "repos_with_programs_jsonl": n_repos_with_jsonl,
        "repos_without_programs_jsonl": len(repos_without_jsonl),
        "repos_without_programs_jsonl_names": sorted(repos_without_jsonl)[:50],
        "sliced_source_files": n_programs,
        "subprograms": n_rows,
        "subprograms_by_code_kind": kind_counts,
        "pct_subprograms_by_code_kind": {k: _pct(v, n_rows) for k, v in kind_counts.items()},
        "subprograms_duplicate": n_dup_rows,
        "pct_duplicate": _pct(n_dup_rows, n_rows),
        "unique_bodies_corpus_wide": n_unique,
        "unique_bodies_in_more_than_one_repo": multi_repo_bodies,
        "pct_unique_bodies_cross_repo": _pct(multi_repo_bodies, n_unique),
        "rows_with_body": n_bodies_seen,
        "rows_without_body": n_no_body,
        "positional_mismatches": positional_mismatches,
        "unpaired_canonicals": unpaired_canonicals,
        "rows_parseable": n_parse_ok,
        "rows_unparseable": n_parse_fail,
        "pct_rows_parseable": _pct(n_parse_ok, n_parse_ok + n_parse_fail),
        "unique_bodies_parseable": unique_ok,
        "pct_unique_bodies_parseable": _pct(unique_ok, n_unique),
        "slicer_reported_syntax_errors": file_syntax_errors,
        "slicer_reported_subprograms": file_subprogram_total,
        "schemas": schemas,
        "malformed_jsonl_lines": len(bad_lines),
        "malformed_jsonl_examples": bad_lines[:10],
        "pdg_lines_per_slice": _quantiles(pdg_sizes),
        "inserted_lines_per_slice": _quantiles(inserted_sizes),
        "top_repos_by_subprograms": sorted(
            [dict(repo=r, **v) for r, v in per_repo.items()],
            key=lambda d: -d["subprograms"])[:15],
        "outputs": {
            "slice_index_csv": str(SLICE_INDEX_CSV),
            "slice_files_csv": str(SLICE_FILES_CSV),
            "slice_pdg_lines_csv_gz": str(SLICE_PDG_LINES_CSV_GZ),
            "slice_code_jsonl_gz": str(SLICE_CODE_JSONL_GZ) if write_code else None,
        },
    }
    SLICE_INDEX_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slices-dir", type=Path, default=paths.SLICES_DIR)
    ap.add_argument("--limit", type=int, default=None, help="only the first N repo dirs")
    ap.add_argument("--repo", default=None, help="substring filter on the repo dir name")
    ap.add_argument("--no-code", action="store_true", help="skip the code content store")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not args.slices_dir.is_dir():
        print("no slices directory: {}".format(args.slices_dir), file=sys.stderr)
        return 2

    s = build(args.slices_dir, args.limit, args.repo, not args.no_code, args.quiet)

    print("\n{:,} subprograms from {:,} source files in {:,} repos ({}s)".format(
        s["subprograms"], s["sliced_source_files"], s["repos_with_programs_jsonl"],
        s["elapsed_seconds"]))
    print("  duplicates     : {:,} ({}%)".format(s["subprograms_duplicate"], s["pct_duplicate"]))
    print("  unique bodies  : {:,} ({:,} in >1 repo)".format(
        s["unique_bodies_corpus_wide"], s["unique_bodies_in_more_than_one_repo"]))
    print("  parseable rows : {:,} ({}%)".format(s["rows_parseable"], s["pct_rows_parseable"]))
    print("  wrote {}".format(SLICE_INDEX_SUMMARY_JSON.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
