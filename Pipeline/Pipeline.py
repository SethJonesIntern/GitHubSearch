"""End-to-end pipeline: Frameworks search -> Applications search -> Wrapper
transitive LLM-test analysis.

Each stage's output feeds the next:
  1. frameworks   Frameworks/GithubSearch.py
                  -> Frameworks/github_agent_framework_candidates.csv
                     (used by stage 2 to exclude framework repos)
  2. applications Applications/search_candidates.py
                  -> Applications/application_candidates_v2.csv
                     (used by stage 3 as the list of repos to analyze)
  3. wrappers     Wrapper/transitive_invokers.py (per candidate repo)
                  -> Pipeline/wrapper_test_results.csv
                  -> Pipeline/wrapper_llm_tests.csv

Use --start-stage / --end-stage to run a subset, e.g. to pick up where a
previous run left off without re-running earlier stages.

Examples:
  python Pipeline.py                                   # run all three stages
  python Pipeline.py --start-stage applications        # skip frameworks search
  python Pipeline.py --start-stage wrappers            # only the wrapper analysis
"""
import argparse
import csv
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FRAMEWORKS_DIR = ROOT / "Frameworks"
APPLICATIONS_DIR = ROOT / "Applications"
WRAPPER_DIR = ROOT / "Wrapper"
SEMANTIC_EVAL_DIR = ROOT / "SemanticEvaluators"

STAGES = ["frameworks", "applications", "wrappers"]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _flush_csv(path: Path, rows: list) -> None:
    """Write rows to CSV with the union of all keys as the header, in
    first-seen order. No-op if rows is empty (avoids clobbering an existing
    file with an empty one on the first iteration)."""
    if not rows:
        return
    fieldnames: list = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


# ── Stage 1: Frameworks search ──────────────────────────────────────────────


def run_frameworks_stage() -> None:
    print("=== Stage 1: Frameworks search ===")
    print(f"  (overwrites {FRAMEWORKS_DIR / 'github_agent_framework_candidates.csv'})")
    sys.path.insert(0, str(FRAMEWORKS_DIR))
    try:
        mod = _load_module("frameworks_github_search", FRAMEWORKS_DIR / "GithubSearch.py")
        cwd = Path.cwd()
        os.chdir(FRAMEWORKS_DIR)
        try:
            mod.main()
        finally:
            os.chdir(cwd)
    finally:
        sys.path.remove(str(FRAMEWORKS_DIR))


# ── Stage 2: Applications search ────────────────────────────────────────────


def run_applications_stage(resume: bool = True) -> None:
    print("=== Stage 2: Applications search ===")
    sys.path.insert(0, str(APPLICATIONS_DIR))
    try:
        mod = _load_module("applications_search_candidates", APPLICATIONS_DIR / "search_candidates.py")
        old_argv = sys.argv
        sys.argv = ["search_candidates.py"] + (["--resume"] if resume else [])
        try:
            mod.main()
        finally:
            sys.argv = old_argv
    finally:
        sys.path.remove(str(APPLICATIONS_DIR))


# ── Stage 3: Wrapper transitive-invoker analysis ────────────────────────────


def run_wrappers_stage() -> None:
    print("=== Stage 3: Wrapper transitive-invoker analysis ===")
    if importlib.util.find_spec("pyan.analyzer") is None:
        print("  pyan3 is required for this stage: py -m pip install pyan3==2.6.0")
        return

    sys.path.insert(0, str(WRAPPER_DIR))
    try:
        ti = _load_module("wrapper_transitive_invokers", WRAPPER_DIR / "transitive_invokers.py")
    finally:
        sys.path.remove(str(WRAPPER_DIR))

    sem = _load_module("semantic_eval_imports", SEMANTIC_EVAL_DIR / "semantic_eval_imports.py")

    input_csv = APPLICATIONS_DIR / "application_candidates_v2.csv"
    if not input_csv.exists():
        print(f"  {input_csv} not found — run the applications stage first")
        return

    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))

    summary_csv = HERE / "wrapper_test_results.csv"
    detail_csv = HERE / "wrapper_llm_tests.csv"

    summary_rows: list = []
    done = set()
    if summary_csv.exists():
        with open(summary_csv, "r", newline="", encoding="utf-8") as f:
            summary_rows = list(csv.DictReader(f))
            done = {r["full_name"] for r in summary_rows}

    detail_rows: list = []
    if detail_csv.exists():
        with open(detail_csv, "r", newline="", encoding="utf-8") as f:
            detail_rows = list(csv.DictReader(f))

    cache_dir = HERE / "repo_cache"
    cache_dir.mkdir(exist_ok=True)

    todo = [r for r in candidates if r["full_name"] not in done]
    print(f"  {len(candidates)} candidates, {len(done)} already done, {len(todo)} remaining")

    for i, row in enumerate(todo, 1):
        full_name = row["full_name"]
        clone_url = row["clone_url"]
        safe_name = full_name.replace("/", "_")
        dest = cache_dir / safe_name
        print(f"  [{i}/{len(todo)}] {full_name}")

        if dest.exists():
            shutil.rmtree(dest, onexc=ti._on_rm_error)

        if not ti.shallow_clone(clone_url, dest):
            summary_rows.append({
                **row, "wrapper_status": "clone_failed",
                "functions_indexed": 0, "seed_invokers": 0,
                "total_invokers": 0, "llm_test_count": 0,
            })
        else:
            try:
                functions, contexts = ti.index_repo(dest, cache_dir)

                # Semantic-evaluator detection rides along on the same parse:
                # contexts already carry each file's full import set.
                eval_hits = sem.detect_from_contexts(contexts)

                seeds = ti.seed_invokers(functions, contexts)
                call_graph = ti.build_call_graph(dest, cache_dir)
                invokers = ti.transitive_closure(seeds, call_graph)

                test_hits = []
                for qname, reason in invokers.items():
                    if not qname.rsplit(".", 1)[-1].startswith("test_"):
                        continue
                    fi = functions.get(qname)
                    if fi is None:
                        continue
                    fname = Path(fi.file_path).name
                    if not (fname.startswith("test_") or fname.endswith("_test.py")):
                        continue
                    test_hits.append((qname, fi, reason))

                for qname, fi, reason in test_hits:
                    detail_rows.append({
                        "repo": full_name,
                        "file": fi.file_path,
                        "line": fi.line,
                        "test_function": qname.rsplit(".", 1)[-1],
                        "reason": reason,
                    })

                summary_rows.append({
                    **row,
                    "wrapper_status": "ok",
                    "functions_indexed": len(functions),
                    "seed_invokers": len(seeds),
                    "total_invokers": len(invokers),
                    "llm_test_count": len(test_hits),
                    "semantic_evaluators": ",".join(sorted(eval_hits)),
                    "semantic_eval_files": ",".join(
                        f"{tool}:{f}" for tool in sorted(eval_hits) for f in eval_hits[tool]
                    ),
                })
                print(f"    {len(functions)} functions, {len(seeds)} seeds, "
                      f"{len(invokers)} invokers, {len(test_hits)} LLM-backed tests"
                      + (f", eval: {','.join(sorted(eval_hits))}" if eval_hits else ""))
            except Exception as e:
                print(f"    error: {e}")
                summary_rows.append({
                    **row, "wrapper_status": f"error: {e}",
                    "functions_indexed": 0, "seed_invokers": 0,
                    "total_invokers": 0, "llm_test_count": 0,
                })
            finally:
                shutil.rmtree(dest, onexc=ti._on_rm_error)

        if i % 5 == 0 or i == len(todo):
            _flush_csv(summary_csv, summary_rows)
            _flush_csv(detail_csv, detail_rows)
            print(f"    ---- CSVs flushed ({len(summary_rows)} repos, {len(detail_rows)} LLM tests) ----")

    _flush_csv(summary_csv, summary_rows)
    _flush_csv(detail_csv, detail_rows)
    shutil.rmtree(cache_dir, onexc=ti._on_rm_error)
    print(f"\n  Done. {len(summary_rows)} repos in {summary_csv}, {len(detail_rows)} LLM-backed tests in {detail_csv}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-stage", choices=STAGES, default="frameworks",
                        help="First stage to run (default: frameworks)")
    parser.add_argument("--end-stage", choices=STAGES, default="wrappers",
                        help="Last stage to run (default: wrappers)")
    parser.add_argument("--fresh-applications", action="store_true",
                        help="Ignore Applications/.search_progress.json and start "
                             "the applications stage from scratch")
    args = parser.parse_args()

    start_idx = STAGES.index(args.start_stage)
    end_idx = STAGES.index(args.end_stage)
    if start_idx > end_idx:
        sys.exit("--start-stage must come at or before --end-stage")

    if start_idx <= STAGES.index("frameworks") <= end_idx:
        run_frameworks_stage()
    if start_idx <= STAGES.index("applications") <= end_idx:
        run_applications_stage(resume=not args.fresh_applications)
    if start_idx <= STAGES.index("wrappers") <= end_idx:
        run_wrappers_stage()


if __name__ == "__main__":
    main()
