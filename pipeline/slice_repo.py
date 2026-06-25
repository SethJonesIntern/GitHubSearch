#!/usr/bin/env python3
"""Stage 6 per-repo slicer: build a Joern CPG for one application checkout, then
emit per-variable SubPDGs for every function that reaches an LLM call.

This glues the two existing engines together for the real-repo case:

  1. ``joern-parse <repo> --language PYTHONSRC -o repo.cpg`` — reusing
     ``create_project_codenet_cpgs.run_joern_parse`` (with its StackOverflow
     retry ladder), pointed at a whole repo checkout instead of a flat chunk.
  2. ``per_variable_pdg_slicer.process_cpg`` over that CPG, with
     ``--recursive`` discovery and a function filter built from Stage 5's
     ``llm_invokers_all.csv`` so only the LLM-invoker closure is sliced.

Unlike the corpus builders this is intentionally one-repo-at-a-time: the caller
(a future orchestrator stage) owns cloning, iteration, and cleanup. The CPG is
built into a temp dir and discarded unless ``--cpg-out`` is given.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from pipeline.create_project_codenet_cpgs import (
    parse_java_stack_sizes,
    resolve_joern_parse_executable,
    run_joern_parse,
)
from pipeline.per_variable_pdg_slicer import (
    DEFAULT_JOERN,
    FunctionFilter,
    load_function_filter,
    process_cpg,
)

DEFAULT_JAVA_STACK_SIZES = "default,32m,64m,128m,256m,512m"


def slice_repo(
    *,
    repo_dir: Path,
    output_dir: Path,
    invokers_csv: Optional[Path] = None,
    invokers_repo: Optional[str] = None,
    function_filter: Optional[FunctionFilter] = None,
    joern_parse: str = "joern-parse",
    joern: str = DEFAULT_JOERN,
    criterion_mode: str = "bidirectional",
    max_data_depth: Optional[int] = None,
    standalone_closure: bool = True,
    output_format: str = "jsonl",
    jsonl_detail: str = "refined",
    workers: int = 1,
    joern_timeout: int = 7200,
    java_stack_sizes: str = DEFAULT_JAVA_STACK_SIZES,
    java_opts: str = "",
    progress_interval: int = 100,
    cpg_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a CPG for ``repo_dir`` and slice its LLM-invoker functions."""
    repo_dir = Path(repo_dir).resolve()
    if not repo_dir.is_dir():
        raise SystemExit(f"Repo directory not found: {repo_dir}")
    output_dir = Path(output_dir).resolve()

    joern_parse_bin = resolve_joern_parse_executable(joern_parse)
    # A prebuilt filter (e.g. the driver's in-memory invoker rows) wins over the
    # CSV path, so the folded-in caller never has to round-trip through disk.
    if function_filter is None and invokers_csv is not None:
        function_filter = load_function_filter(Path(invokers_csv), invokers_repo)

    tmp_context: Optional[tempfile.TemporaryDirectory] = None
    if cpg_out is None:
        tmp_context = tempfile.TemporaryDirectory(prefix="slice_repo_")
        cpg_path = Path(tmp_context.name) / "repo.cpg"
    else:
        cpg_path = Path(cpg_out).resolve()
        cpg_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        elapsed, java_stack_size, attempts = run_joern_parse(
            joern_parse=joern_parse_bin,
            chunk_dir=repo_dir,
            cpg_path=cpg_path,
            timeout=joern_timeout,
            java_stack_sizes=parse_java_stack_sizes(java_stack_sizes),
            java_opts=java_opts,
        )
        print(
            f"Built CPG for {repo_dir.name} in {elapsed / 60:.2f} min "
            f"(java stack {java_stack_size or 'default'}, attempt {attempts}) -> {cpg_path}",
            flush=True,
        )
        return process_cpg(
            input_dir=repo_dir,
            output_dir=output_dir,
            cpg=cpg_path,
            joern=joern,
            program=None,
            criterion_mode=criterion_mode,
            max_data_depth=max_data_depth,
            standalone_closure=standalone_closure,
            joern_timeout=joern_timeout,
            workers=workers,
            output_format=output_format,
            jsonl_detail=jsonl_detail,
            progress_interval=progress_interval,
            recursive=True,
            function_filter=function_filter,
        )
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a CPG for one repo checkout and slice its LLM-invoker functions."
    )
    parser.add_argument("--repo-dir", type=Path, required=True,
                        help="Path to the application checkout to slice.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write programs.jsonl / overall_summary.json.")
    parser.add_argument("--invokers-csv", type=Path, default=None,
                        help="llm_invokers_all.csv; restricts slicing to the invoker closure.")
    parser.add_argument("--invokers-repo", default=None,
                        help="Restrict the invokers CSV to rows whose 'repo' column matches.")
    parser.add_argument("--joern-parse", default="joern-parse",
                        help="joern-parse binary, joern-cli dir, or install root.")
    parser.add_argument("--joern", default=DEFAULT_JOERN,
                        help="joern binary used for the CPG query.")
    parser.add_argument("--criterion-mode",
                        choices=("bidirectional", "last-use", "all-mentions"),
                        default="bidirectional")
    parser.add_argument("--max-data-depth", type=int, default=-1,
                        help="DDG traversal depth. -1 means unbounded.")
    parser.add_argument("--no-standalone-closure", action="store_true")
    parser.add_argument("--output-format", choices=("jsonl", "files"), default="jsonl")
    parser.add_argument("--jsonl-detail", choices=("refined", "full"), default="refined")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--joern-timeout", type=int, default=7200)
    parser.add_argument("--java-stack-sizes", default=DEFAULT_JAVA_STACK_SIZES)
    parser.add_argument("--java-opts", default="")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--cpg-out", type=Path, default=None,
                        help="Persist the built CPG here instead of a temp dir.")
    args = parser.parse_args()

    overall = slice_repo(
        repo_dir=args.repo_dir,
        output_dir=args.output_dir,
        invokers_csv=args.invokers_csv,
        invokers_repo=args.invokers_repo,
        joern_parse=args.joern_parse,
        joern=args.joern,
        criterion_mode=args.criterion_mode,
        max_data_depth=None if args.max_data_depth < 0 else args.max_data_depth,
        standalone_closure=not args.no_standalone_closure,
        output_format=args.output_format,
        jsonl_detail=args.jsonl_detail,
        workers=args.workers,
        joern_timeout=args.joern_timeout,
        java_stack_sizes=args.java_stack_sizes,
        java_opts=args.java_opts,
        progress_interval=args.progress_interval,
        cpg_out=args.cpg_out,
    )
    print(
        f"Sliced {overall['program_count']} files -> "
        f"{overall['total_subprograms']} subprograms "
        f"({overall['total_deduplicated']} deduplicated, "
        f"{overall['total_syntax_errors']} syntax errors)"
    )


if __name__ == "__main__":
    main()
