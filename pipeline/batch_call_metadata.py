"""Stage 5 (+7) driver: for every application, clone it, run the invoker and
semantic-evaluation analyses against the *same* parse, write the results, then
delete the clone.

See PIPELINE.md "Stage 5" / "Stage 7". One clone + parse per repo; the two seed
dicts (FRAMEWORK_CALLS for LLM calls, EVAL_CALLS for eval calls) are matched
against that single parse, so the app set is never cloned or parsed twice.

Outputs (all under pipeline/artifacts/, every row tagged with `repo`):
    llm_invokers_all.csv        every direct+transitive LLM invoker function (Stage 5)
    llm_calls_all.csv           one row per LLM call site         (Stage 5a)
    call_metadata_all.csv       one row per LLM call argument     (Stage 5b → JOERN)
    llm_tests_all.csv           pytest tests reaching an LLM call (Stage 5c)
    eval_invokers_all.csv       every direct+transitive eval invoker function (Stage 7)
    eval_calls_all.csv          one row per eval call site        (Stage 7a)
    eval_call_metadata_all.csv  one row per eval call argument    (Stage 7b → JOERN)

The invoker search (transitive_closure over the call graph) is the backbone:
llm_tests is the pytest subset of llm_invokers; calls/metadata come from the
direct seeds. Without pyan3 (Python 3.14) the closure reduces to direct seeds.

Disk: each repo is shallow-cloned into pipeline/repos/, analysed, then removed
(unless --keep-clones), so only one checkout exists at a time.

Robustness: per-repo error isolation (a clone/parse failure logs and continues);
the call graph (pyan3, needs Python 3.14) is best-effort — without it the per-
argument metadata (the JOERN inputs) is still complete from the direct seeds;
only the *transitive* test closure is reduced to direct hits.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from pipeline import paths
from pipeline.eval_calls import EVAL_CALLS
from pipeline import engines as E

# Union of keys so a single parse captures both LLM and eval framework imports;
# each matching pass then uses its own dict (handles keys present in both).
COMBINED_CALLS = {**E.FRAMEWORK_CALLS, **EVAL_CALLS}


# ── output schemas ────────────────────────────────────────────────────────────

CALL_FIELDS = [
    "repo", "call_id", "file", "enclosing_qname", "framework", "pattern",
    "callable", "call_source", "call_line", "call_col", "is_await", "arg_count",
]
METADATA_FIELDS = ["repo"] + list(E.CALL_METADATA_FIELDS)
# Invokers and tests share a shape: a function that (transitively) reaches a call.
INVOKER_FIELDS = ["repo", "qname", "file", "line", "reason", "kind"]
TEST_FIELDS = INVOKER_FIELDS


def _calls_from_metadata(repo: str, meta_rows: list[dict]) -> list[dict]:
    """Collapse per-argument metadata rows to one row per call site (call_id)."""
    seen: dict[str, dict] = {}
    for r in meta_rows:
        cid = r["call_id"]
        if cid not in seen:
            seen[cid] = {
                "repo": repo,
                "call_id": cid,
                "file": r["file"],
                "enclosing_qname": r["enclosing_qname"],
                "framework": r["framework"],
                "pattern": r["pattern"],
                "callable": r["callable"],
                "call_source": r["call_source"],
                "call_line": r["call_line"],
                "call_col": r["call_col"],
                "is_await": r["is_await"],
                "arg_count": r["arg_count"],
            }
    return list(seen.values())


def _invokers_rows(repo: str, invokers: dict, functions: dict) -> list[dict]:
    """One row per invoker function (every function that directly or transitively
    reaches a matched call), tagged direct vs transitive. This is the invoker
    search result. Nodes pyan3 resolved but we didn't index are skipped."""
    rows = []
    for qname, reason in invokers.items():
        fi = functions.get(qname)
        if fi is None:
            continue
        rows.append({
            "repo": repo,
            "qname": qname,
            "file": fi.file_path,
            "line": fi.line,
            "reason": reason,
            "kind": "direct" if reason.startswith("matches") else "transitive",
        })
    return rows


def _tests_among(invoker_rows: list[dict]) -> list[dict]:
    """The pytest tests among the invokers (file + function both match pytest
    conventions) — a filtered view of the invoker search."""
    return [
        r for r in invoker_rows
        if E.is_test_function(r["qname"]) and E.is_test_file(r["file"])
    ]


# ── per-repo analysis ─────────────────────────────────────────────────────────


def safe_call_graph(clone_dir: Path, repo_root: Path) -> dict:
    """build_call_graph, but never fatal: returns {} if pyan3 is missing or
    errors (e.g. wrong Python version), so the run degrades to direct seeds."""
    try:
        return E.build_call_graph(clone_dir, repo_root)
    except SystemExit:        # build_call_graph sys.exit()s when pyan3 is absent
        return {}
    except Exception as e:    # noqa: BLE001 - any pyan failure is non-fatal here
        print(f"  call graph failed ({e}); falling back to direct seeds", file=sys.stderr)
        return {}


def process_repo(repo_full_name: str, clone_dir: Path) -> dict:
    """Index once, then run the LLM and eval passes against the same parse.
    Returns the five repo-tagged row groups."""
    repo_root = clone_dir.parent  # mirrors the single-repo CLI convention

    functions, contexts = E.index_repo(clone_dir, repo_root, COMBINED_CALLS)
    call_graph = safe_call_graph(clone_dir, repo_root)
    index = E.AstIndex(functions, repo_root)

    def run_pass(calls_dict):
        seeds = E.seed_invokers(functions, contexts, calls_dict)
        meta = E.collect_rows(seeds, index, contexts, calls_dict)
        for r in meta:
            r_repo = {"repo": repo_full_name, **r}
            r.clear()
            r.update(r_repo)
        return seeds, meta

    llm_seeds, llm_meta = run_pass(E.FRAMEWORK_CALLS)
    eval_seeds, eval_meta = run_pass(EVAL_CALLS)

    # Invoker search: walk the call graph backwards from the seeds so every
    # function that transitively reaches a matched call is captured (not just the
    # direct seeds). Without a call graph (pyan3/3.14) this reduces to the seeds.
    llm_invoker_rows = _invokers_rows(
        repo_full_name, E.transitive_closure(llm_seeds, call_graph), functions)
    eval_invoker_rows = _invokers_rows(
        repo_full_name, E.transitive_closure(eval_seeds, call_graph), functions)

    return {
        "llm_invokers": llm_invoker_rows,
        "llm_calls": _calls_from_metadata(repo_full_name, llm_meta),
        "call_metadata": llm_meta,
        "llm_tests": _tests_among(llm_invoker_rows),
        "eval_invokers": eval_invoker_rows,
        "eval_calls": _calls_from_metadata(repo_full_name, eval_meta),
        "eval_call_metadata": eval_meta,
    }


# ── IO ────────────────────────────────────────────────────────────────────────


def load_applications() -> list[dict]:
    if not paths.APPLICATIONS_CSV.exists():
        raise FileNotFoundError(
            f"{paths.APPLICATIONS_CSV} not found — run Stage 2 (search_candidates.py) first.")
    with paths.APPLICATIONS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(path: Path, fieldnames: list[str], rows: list[dict]):
    if not rows:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# group key -> (path, fieldnames)
OUTPUTS = {
    "llm_invokers": (paths.LLM_INVOKERS_CSV, INVOKER_FIELDS),
    "llm_calls": (paths.LLM_CALLS_CSV, CALL_FIELDS),
    "call_metadata": (paths.CALL_METADATA_CSV, METADATA_FIELDS),
    "llm_tests": (paths.LLM_TESTS_CSV, TEST_FIELDS),
    "eval_invokers": (paths.EVAL_INVOKERS_CSV, INVOKER_FIELDS),
    "eval_calls": (paths.EVAL_CALLS_CSV, CALL_FIELDS),
    "eval_call_metadata": (paths.EVAL_CALL_METADATA_CSV, METADATA_FIELDS),
}


def load_progress() -> dict:
    if paths.BATCH_PROGRESS_JSON.exists():
        with paths.BATCH_PROGRESS_JSON.open(encoding="utf-8") as f:
            return json.load(f)
    return {"processed": [], "failed": []}


def save_progress(progress: dict):
    with paths.BATCH_PROGRESS_JSON.open("w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", action="store_true",
                    help="Skip repos already in the progress file instead of starting fresh")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N applications (smoke runs)")
    ap.add_argument("--keep-clones", action="store_true",
                    help="Do not delete each repo after analysis (debugging)")
    args = ap.parse_args()

    paths.ensure_dirs()

    if not args.resume:
        for path, _ in OUTPUTS.values():
            if path.exists():
                path.unlink()
        if paths.BATCH_PROGRESS_JSON.exists():
            paths.BATCH_PROGRESS_JSON.unlink()

    progress = load_progress()
    done = set(progress["processed"])

    apps = load_applications()
    if args.limit is not None:
        apps = apps[:args.limit]
    print(f"Processing {len(apps)} applications "
          f"(LLM frameworks: {len(E.FRAMEWORK_CALLS)}, eval frameworks: {len(EVAL_CALLS)})")

    totals = defaultdict(int)
    for i, app in enumerate(apps, 1):
        full_name = app.get("full_name")
        clone_url = app.get("clone_url")
        if not full_name or not clone_url:
            continue
        if full_name in done:
            continue

        clone_dir = paths.REPOS_DIR / E.repo_slug(clone_url)
        print(f"[{i}/{len(apps)}] {full_name}")

        if clone_dir.exists():
            shutil.rmtree(clone_dir, onerror=E._on_rm_error)
        if not E.shallow_clone(clone_url, clone_dir):
            progress["failed"].append({"repo": full_name, "stage": "clone"})
            save_progress(progress)
            continue

        try:
            results = process_repo(full_name, clone_dir)
            for key, rows in results.items():
                path, fields = OUTPUTS[key]
                append_rows(path, fields, rows)
                totals[key] += len(rows)
            print(f"    {len(results['llm_invokers'])} LLM invokers "
                  f"({len(results['llm_calls'])} calls, {len(results['llm_tests'])} tests), "
                  f"{len(results['eval_invokers'])} eval invokers "
                  f"({len(results['eval_calls'])} calls)")
            progress["processed"].append(full_name)
        except Exception as e:  # noqa: BLE001 - isolate per-repo failures
            print(f"    analysis failed: {e}", file=sys.stderr)
            progress["failed"].append({"repo": full_name, "stage": "analysis", "error": str(e)})
        finally:
            if not args.keep_clones and clone_dir.exists():
                shutil.rmtree(clone_dir, onerror=E._on_rm_error)
            save_progress(progress)

    print("\nDone. Totals:")
    for key, (path, _) in OUTPUTS.items():
        print(f"  {path.name:<28}: {totals[key]} rows")
    if progress["failed"]:
        print(f"  failed repos: {len(progress['failed'])}")


if __name__ == "__main__":
    main()
