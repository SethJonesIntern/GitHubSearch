"""Find every pytest test in a repo that ever invokes an LLM.

Picks up where transitive_invokers.py leaves off. That script flags every
function or method that triggers an LLM call (directly or transitively); this
one keeps just the pytest tests among them: files named test_*.py or *_test.py
with functions whose names start with 'test_'.

The result tells you which tests are non-deterministic because they exercise
LLM-invoking code. Each row carries the reason it was flagged, either a direct
hit ("matches '.invoke' from langchain") or a transitive link ("calls some.qname").
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from transitive_invokers import (
    build_call_graph,
    ensure_clone,
    index_repo,
    seed_invokers,
    transitive_closure,
)


# ── pytest convention checks ──────────────────────────────────────────────────


def is_test_file(rel_path: str) -> bool:
    """A file is a pytest test if its basename matches the discovery defaults."""
    name = Path(rel_path).name
    return name.startswith("test_") or name.endswith("_test.py")


def is_test_function(qname: str) -> bool:
    """True if the last segment of the qname starts with 'test_'. Covers both
    plain `def test_foo(...)` and methods like `TestX.test_y`."""
    return qname.rsplit(".", 1)[-1].startswith("test_")


# ── reporting ─────────────────────────────────────────────────────────────────


def report(
    test_invokers: dict[str, tuple[str, int, str]],
) -> None:
    """Print test invokers grouped by file, sorted by line within each file."""
    by_file: dict[str, list[tuple[str, int, str, str]]] = defaultdict(list)
    for qname, (file_path, line, reason) in test_invokers.items():
        name = qname.rsplit(".", 1)[-1]
        by_file[file_path].append((name, line, reason, qname))

    for file_path in sorted(by_file):
        print(f"\n{file_path}")
        for name, line, reason, _qname in sorted(by_file[file_path], key=lambda r: r[1]):
            print(f"  L{line:<5} {name:40}  <- {reason}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("target", type=str,
                        help="Directory to scan, or a git URL to clone into repos/")
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="Parent of the top-level package; defaults to target's parent")
    parser.add_argument("--json", type=Path,
                        help="Also write the filtered test-invoker map to this JSON file")
    parser.add_argument("--include-non-test-files", action="store_true",
                        help="Match any function whose name starts with 'test_', even when "
                             "the file's name doesn't follow pytest conventions.")
    args = parser.parse_args()

    # Resolve target the same way transitive_invokers does.
    if args.target.startswith(("http://", "https://", "git@")):
        target = ensure_clone(args.target).resolve()
    else:
        target = Path(args.target).resolve()
        if not target.is_dir():
            sys.exit(f"not a URL and not a directory: {args.target}")
    repo_root = (args.repo_root or target.parent).resolve()

    # Run the full transitive_invokers pipeline; test files were indexed
    # along with everything else, so the result already has what we need.
    functions, contexts = index_repo(target, repo_root)
    seeds = seed_invokers(functions, contexts)
    call_graph = build_call_graph(target, repo_root)
    invokers = transitive_closure(seeds, call_graph)

    # Keep only pytest tests. By default both conventions must hold: the file
    # name looks like a test file and the function name starts with 'test_'.
    # --include-non-test-files drops the file-name requirement.
    test_invokers: dict[str, tuple[str, int, str]] = {}
    for qname, reason in invokers.items():
        if not is_test_function(qname):
            continue
        fi = functions.get(qname)
        if fi is None:
            continue
        if not args.include_non_test_files and not is_test_file(fi.file_path):
            continue
        test_invokers[qname] = (fi.file_path, fi.line, reason)

    direct = sum(1 for _, (_, _, r) in test_invokers.items() if r.startswith("matches"))
    transitive = len(test_invokers) - direct

    print(f"# Scanned {len(functions)} functions; {len(invokers)} are invokers")
    print(f"# {len(test_invokers)} of those are pytest tests "
          f"({direct} direct, {transitive} transitive)")

    report(test_invokers)

    if args.json:
        # Stable shape: {qname: {file, line, reason}}.
        out = {
            qname: {"file": fp, "line": ln, "reason": reason}
            for qname, (fp, ln, reason) in test_invokers.items()
        }
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
