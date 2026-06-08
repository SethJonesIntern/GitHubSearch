"""Dump per-argument metadata for every direct LLM call site in a repo.

The slicing itself is done by a separate tool; this script's only job is to hand
that tool (and later analysis) as much structured context about each call's
arguments as we can pull statically. It reuses the seed detection from
transitive_invokers — the same "file imports framework X, body has a call
matching one of X's patterns" logic — but needs no call graph, since everything
here is local to the call site.

Output is a flat, tidy CSV with **one row per argument**. A call with no
arguments still gets a single row (arg fields blank) so the site isn't lost.
Rows from the same call share a `call_id` so they regroup trivially in pandas.

The argument columns (`arg_kind`, `arg_source`, `arg_names`, `arg_is_literal`)
plus the precise `call_line`/`call_col` and the per-call `call_arg_vars` give a
generic superset: whether the downstream slicer keys off (file, line) or
(file, line, variable), the seed is already in the row.
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path
from typing import Iterator, Optional

from astWrappers import matcher
from FrameworkDict import FRAMEWORK_CALLS
from transitive_invokers import (
    derive_module,
    ensure_clone,
    index_repo,
    seed_invokers,
    FunctionInfo,
)


# ── AST helpers ───────────────────────────────────────────────────────────────


def loaded_names(node: ast.AST) -> list[str]:
    """Variable names read inside `node`, sorted & de-duped. These are the
    slicing-criterion variables for an argument expression."""
    return sorted({n.id for n in ast.walk(node)
                   if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)})


def is_literal(node: ast.AST) -> bool:
    """True if the expression is a compile-time constant (a number/string, or a
    list/tuple/dict/set built only from such). Driven by ast.literal_eval so we
    don't have to enumerate container shapes ourselves."""
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, SyntaxError, TypeError):
        return False


def src(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparse failed>"


# ── lazy per-file AST index ───────────────────────────────────────────────────


class AstIndex:
    """Parse each file at most once; map a function qname to its ast node."""

    def __init__(self, functions: dict[str, FunctionInfo], repo_root: Path):
        self.functions = functions
        self.repo_root = repo_root
        self._by_file: dict[str, dict[str, ast.AST]] = {}

    def _file_map(self, rel_path: str) -> dict[str, ast.AST]:
        if rel_path in self._by_file:
            return self._by_file[rel_path]
        node_map: dict[str, ast.AST] = {}
        abs_path = (self.repo_root / rel_path).resolve()
        try:
            tree = ast.parse(abs_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            self._by_file[rel_path] = node_map
            return node_map
        _, module = derive_module(abs_path, self.repo_root)
        for top in tree.body:
            if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node_map[f"{module}.{top.name}"] = top
            elif isinstance(top, ast.ClassDef):
                for sub in top.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        node_map[f"{module}.{top.name}.{sub.name}"] = sub
        self._by_file[rel_path] = node_map
        return node_map

    def node(self, qname: str) -> Optional[ast.AST]:
        fi = self.functions.get(qname)
        if fi is None:
            return None
        return self._file_map(fi.file_path).get(qname)


# ── call-site discovery ───────────────────────────────────────────────────────


def active_matchers(module: str, contexts) -> list[tuple[str, object, str]]:
    """(pattern, matcher_fn, framework) for every pattern whose framework this
    module actually imports — exactly the seed_invokers gating, rebuilt here so
    we can attribute each hit to its pattern."""
    ctx = contexts.get(module)
    if not ctx or not ctx.imported_frameworks:
        return []
    return [(pat, matcher(pat), fw)
            for fw in ctx.imported_frameworks
            for pat in FRAMEWORK_CALLS[fw]]


def module_of(fi: FunctionInfo, qname: str) -> str:
    """The module a function lives in (drops class+method for methods)."""
    if fi.is_method and fi.enclosing_class:
        return fi.enclosing_class.rsplit(".", 1)[0]
    return qname.rsplit(".", 1)[0]


def llm_calls_in(node: ast.AST, active) -> Iterator[tuple[ast.Call, str, str, bool]]:
    """Yield (call_node, framework, pattern, is_await) for each LLM call in the
    function body. First matching pattern wins per call; each call yielded once."""
    await_ids = {id(n.value) for n in ast.walk(node)
                 if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)}
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
        try:
            text = ast.unparse(call.func)
        except Exception:
            continue
        for pat, m, fw in active:
            if m(text):
                yield call, fw, pat, id(call) in await_ids
                break


# ── argument flattening ───────────────────────────────────────────────────────


def iter_arguments(call: ast.Call):
    """Yield (position, keyword, expr) for every argument of a call:

      positional   -> (i, "",  expr)
      *args        -> ("", "*", expr)
      keyword=...  -> ("", name, expr)
      **kwargs     -> ("", "**", expr)
    """
    pos = 0
    for a in call.args:
        if isinstance(a, ast.Starred):
            yield "", "*", a.value
        else:
            yield pos, "", a
            pos += 1
    for kw in call.keywords:
        yield "", (kw.arg if kw.arg is not None else "**"), kw.value


# ── row emission ──────────────────────────────────────────────────────────────


FIELDS = [
    "call_id", "file", "enclosing_qname", "framework", "pattern",
    "callable", "call_source", "call_line", "call_col",
    "call_end_line", "call_end_col", "is_await", "arg_count", "call_arg_vars",
    "arg_position", "arg_keyword", "arg_kind", "arg_source",
    "arg_names", "arg_is_literal",
]


def rows_for_call(qname: str, fi: FunctionInfo, call: ast.Call,
                  framework: str, pattern: str, is_await: bool) -> list[dict]:
    args = list(iter_arguments(call))
    call_id = f"{fi.file_path}::{call.lineno}::{call.col_offset}"
    call_arg_vars = sorted({v for _, _, e in args for v in loaded_names(e)})

    base = {
        "call_id": call_id,
        "file": fi.file_path,
        "enclosing_qname": qname,
        "framework": framework,
        "pattern": pattern,
        "callable": src(call.func),
        "call_source": src(call),
        "call_line": call.lineno,
        "call_col": call.col_offset,
        "call_end_line": getattr(call, "end_lineno", ""),
        "call_end_col": getattr(call, "end_col_offset", ""),
        "is_await": is_await,
        "arg_count": len(args),
        "call_arg_vars": ";".join(call_arg_vars),
    }

    if not args:                      # keep argless call sites in the dump
        return [dict(base, arg_position="", arg_keyword="", arg_kind="",
                     arg_source="", arg_names="", arg_is_literal="")]

    rows = []
    for position, keyword, expr in args:
        rows.append(dict(
            base,
            arg_position=position,
            arg_keyword=keyword,
            arg_kind=type(expr).__name__,
            arg_source=src(expr),
            arg_names=";".join(loaded_names(expr)),
            arg_is_literal=is_literal(expr),
        ))
    return rows


def collect_rows(seeds, index: AstIndex, contexts) -> list[dict]:
    rows: list[dict] = []
    for qname in sorted(seeds):
        fi = index.functions.get(qname)
        node = index.node(qname)
        if fi is None or node is None:
            continue
        active = active_matchers(module_of(fi, qname), contexts)
        if not active:
            continue
        for call, fw, pat, is_await in llm_calls_in(node, active):
            rows.extend(rows_for_call(qname, fi, call, fw, pat, is_await))
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="Directory to scan, or a git URL to clone")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="Parent of the top-level package; defaults to target's parent")
    ap.add_argument("--out", type=Path, default=Path("call_metadata.csv"),
                    help="CSV output path (default: ./call_metadata.csv)")
    args = ap.parse_args()

    if args.target.startswith(("http://", "https://", "git@")):
        target = ensure_clone(args.target).resolve()
    else:
        target = Path(args.target).resolve()
        if not target.is_dir():
            sys.exit(f"not a URL and not a directory: {args.target}")
    repo_root = (args.repo_root or target.parent).resolve()

    functions, contexts = index_repo(target, repo_root)
    seeds = seed_invokers(functions, contexts)
    index = AstIndex(functions, repo_root)
    rows = collect_rows(seeds, index, contexts)

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    sites = len({r["call_id"] for r in rows})
    print(f"# {len(functions)} functions, {len(seeds)} direct invokers")
    print(f"# {sites} LLM call sites -> {len(rows)} argument rows")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
