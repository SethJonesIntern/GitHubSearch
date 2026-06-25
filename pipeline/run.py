"""Single entry point that runs the whole pipeline in order.

Each stage is run as a subprocess so the stage scripts keep their own argparse
and there are no import collisions between same-named modules in different
folders. Artifacts all land in pipeline/artifacts/ (see pipeline.paths).

Usage:
    python -m pipeline run                  # run every stage, in order
    python -m pipeline run --smoke          # tiny run end-to-end (cheap)
    python -m pipeline run --stages frameworks applications
    python -m pipeline run --from analysis  # this stage onward
    python -m pipeline run --list           # show stages, run nothing
    python -m pipeline run --dry-run        # print commands, run nothing

Stages (in order):
    frameworks    Stage 1   -> frameworks.csv (+ filter stats)
    applications  Stage 2   -> applications.csv, application_metadata.csv (+ stats)
    analysis      Stage 5+7 -> per repo, one clone+parse pass that runs the invoker
                              search (transitive_closure) and the LLM/eval call
                              extraction, emitting: llm_invokers_all, llm_calls_all,
                              call_metadata_all, llm_tests_all, eval_invokers_all,
                              eval_calls_all, eval_call_metadata_all (then JOERN)

Stage 6 (per-variable slicing) is opt-in via --slice: it folds into the analysis
stage's per-repo loop (clone -> analyze -> slice -> delete), so it needs Joern on
PATH. Without --slice the analysis stage behaves exactly as before.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from pipeline.paths import REPO_ROOT

PY = sys.executable

# key -> (title, base command, extra args used only under --smoke)
STAGES = {
    "frameworks": (
        "Stage 1 — framework search",
        [PY, str(REPO_ROOT / "Frameworks" / "GithubSearch.py")],
        ["--limit", "3", "--max-pages", "1"],
    ),
    "applications": (
        "Stage 2 — application search",
        [PY, str(REPO_ROOT / "Applications" / "search_candidates.py")],
        ["--max-terms", "1", "--code-pages", "1", "--max-repos", "2"],
    ),
    "framework_frequency": (
        "Stage 3 — framework frequency table (how many apps import each framework)\n"
        "#            -> framework_frequency.csv  (from the Stage 2 search progress)",
        [PY, str(REPO_ROOT / "Applications" / "framework_distribution.py")],
        [],
    ),
    "analysis": (
        "Stage 5+7 — per repo: clone -> invoker search + LLM/eval call extraction -> delete\n"
        "#            (one clone+parse pass; emits llm_invokers/llm_calls/call_metadata/\n"
        "#             llm_tests + eval_invokers/eval_calls/eval_call_metadata)",
        [PY, "-m", "pipeline.batch_call_metadata"],
        ["--limit", "2"],
    ),
    "eval_frequency": (
        "Stage 7 — eval frequency table (how many apps call each evaluator)\n"
        "#            -> eval_frequency.csv  (derived from eval_calls_all.csv)",
        [PY, "-m", "pipeline.eval_frequency"],
        [],
    ),
}
ORDER = list(STAGES)


def _stage_extra_args(key: str, args) -> list[str]:
    """Per-stage args injected from run-level flags (currently: --slice onto analysis)."""
    if key == "analysis" and args.slice:
        extra = ["--slice",
                 "--joern-parse", args.joern_parse,
                 "--joern", args.joern,
                 "--slice-workers", str(args.slice_workers)]
        if args.keep_cpg:
            extra.append("--keep-cpg")
        return extra
    return []


def _select(args) -> list[str]:
    if args.stages:
        unknown = [s for s in args.stages if s not in STAGES]
        if unknown:
            sys.exit(f"unknown stage(s): {', '.join(unknown)}; choose from {', '.join(ORDER)}")
        return [s for s in ORDER if s in args.stages]
    if args.from_stage:
        if args.from_stage not in STAGES:
            sys.exit(f"unknown --from stage: {args.from_stage}; choose from {', '.join(ORDER)}")
        return ORDER[ORDER.index(args.from_stage):]
    return list(ORDER)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", metavar="STAGE",
                    help=f"run only these stages ({', '.join(ORDER)})")
    ap.add_argument("--from", dest="from_stage", metavar="STAGE",
                    help="run from this stage to the end")
    ap.add_argument("--smoke", action="store_true",
                    help="append small-limit flags to each stage for a cheap end-to-end run")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running")
    ap.add_argument("--slice", action="store_true",
                    help="Stage 6: also slice each repo's LLM-invoker functions during the "
                         "analysis stage, before the clone is deleted (needs Joern on PATH)")
    ap.add_argument("--joern-parse", default="joern-parse",
                    help="joern-parse binary, joern-cli dir, or install root (with --slice)")
    ap.add_argument("--joern", default="joern",
                    help="joern binary used for the CPG query (with --slice)")
    ap.add_argument("--slice-workers", type=int, default=1,
                    help="per-file workers inside each repo's slice (with --slice)")
    ap.add_argument("--keep-cpg", action="store_true",
                    help="persist each repo's CPG under its slice dir instead of a temp dir (with --slice)")
    args = ap.parse_args()

    if args.list:
        for key in ORDER:
            print(f"  {key:<13} {STAGES[key][0]}")
        return

    selected = _select(args)
    print(f"Pipeline: {' -> '.join(selected)}"
          f"{'  (smoke)' if args.smoke else ''}\n")

    for key in selected:
        title, base, smoke_args = STAGES[key]
        cmd = base + (smoke_args if args.smoke else []) + _stage_extra_args(key, args)
        print(f"{'='*70}\n# {title}\n# $ {' '.join(cmd)}\n{'='*70}")
        if args.dry_run:
            continue
        start = time.time()
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        dur = time.time() - start
        if result.returncode != 0:
            sys.exit(f"\nStage '{key}' failed (exit {result.returncode}) after {dur:.0f}s — stopping.")
        print(f"# {key} done in {dur:.0f}s\n")

    if not args.dry_run:
        print("Pipeline complete.")


if __name__ == "__main__":
    main()
