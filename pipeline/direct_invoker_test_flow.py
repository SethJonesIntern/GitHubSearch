"""Independent bounded result-check candidates from saved Stage-5 first edges.

No graph production or application execution. Run with --output-dir NEW_PATH.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import csv
import hashlib
import io
import json
import ntpath
import os
from pathlib import Path
import subprocess
import sys
import tokenize
from collections import Counter, OrderedDict, defaultdict

from . import paths

FIELDS = ("repo", "qname", "file", "line", "reason", "kind")
_OBSERVED_POP = None
EVIDENCE = "saved_pyan_edge_local_return_to_passfail_candidate"
TARGET_EVIDENCE = "saved_pyan_firstedge_plus_syntactic_call"
DISPATCH_LIMIT = "Saved pyan target plus syntactic call match; dispatch and execution are unverified."
LIMITATIONS = [
    DISPATCH_LIMIT,
    "Independent of PyMethods2Test: no PyMethods2Test data or mappings are used.",
    "The saved pyan first-BFS-edge forest contains only one chosen edge per transitive function.",
    "Multi-hop wrappers and unrecorded outgoing edges are unassessed.",
    "Direct-test rows with only framework-match reasons cannot identify their other target calls.",
    "No candidate means no detection, rather than no test or no check.",
    "Static local candidates establish neither actual invocation, mocking status, semantic correctness nor full interprocedural LLM-result flow.",
    "Compound-flow binding merges, arbitrary validators, dynamic receivers, constructor aliases, fixtures and inheritance are unresolved.",
    "Strata are heuristic path/name classifications; this is a saved-edge candidate detection rate, not exhaustive prevalence or coverage.",
    "Controlled failure denotes a result-derived predicate with an immediate failure statement, not proof that the branch executes.",
    "Explicit receiver-owner conflicts abstain as unresolved target identification; inheritance may explain different short class names, and matching names do not verify dispatch.",
]
VALUE_ASSERTS = {
    "assertEqual": 2, "assertNotEqual": 2, "assertTrue": 1, "assertFalse": 1,
    "assertIs": 2, "assertIsNot": 2, "assertIsNone": 1, "assertIsNotNone": 1,
    "assertIn": 2, "assertNotIn": 2, "assertIsInstance": 2, "assertNotIsInstance": 2,
    "assertGreater": 2, "assertGreaterEqual": 2, "assertLess": 2, "assertLessEqual": 2,
    "assertAlmostEqual": 2, "assertNotAlmostEqual": 2, "assertRegex": 2,
    "assertNotRegex": 2, "assertCountEqual": 2,
}


def _validated(rows, label):
    seen = {}
    for number, raw in enumerate(rows, 2):
        if any(k not in raw or raw[k] is None for k in FIELDS) or None in raw:
            raise ValueError(f"{label}:{number}: required schema is {','.join(FIELDS)}")
        row = {k: raw[k] for k in FIELDS}
        try:
            row["line"] = int(row["line"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}:{number}: invalid line {row['line']!r}") from exc
        if (row["line"] < 1 or not all(isinstance(row[k], str) and row[k]
                for k in FIELDS if k != "line") or row["kind"] not in {"direct", "transitive"}):
            raise ValueError(f"{label}:{number}: invalid identity/kind")
        _slug(row["repo"])
        identity = row["repo"], row["file"], row["line"]
        if identity in seen and seen[identity] != row:
            raise ValueError(f"{label}:{number}: conflicting identity {identity}")
        if identity not in seen:
            seen[identity] = row
            yield row


def _sort_key(row):
    return row["repo"], row["file"], row["line"], row["qname"]


def load_direct_invokers() -> list[dict]:
    """Use the authoritative observed population and direct-pattern filter lazily."""
    global _OBSERVED_POP
    from Applications import make_figures as MF
    import pandas as pd
    _OBSERVED_POP = set(MF.POP)
    frame = pd.read_csv(paths.LLM_INVOKERS_CSV, dtype=str, keep_default_na=False)
    missing = set(FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"{paths.LLM_INVOKERS_CSV}: missing columns {sorted(missing)}; fetch Git LFS content if this is a pointer")
    rows = list(_validated(frame.to_dict("records"), str(paths.LLM_INVOKERS_CSV)))
    # Validate conflicts before filtering: corruption must not silently change scope.
    frame = pd.DataFrame(rows, columns=FIELDS)
    frame = MF.AZ.drop_removed_patterns_invokers(frame)
    frame = frame[frame["kind"].eq("direct") & frame["repo"].isin(MF.POP)]
    frame = frame[~frame["reason"].map(MF.AZ.fw_from_reason).isin(MF.RAW_SDKS)]
    return sorted(list(_validated(frame.to_dict("records"), str(paths.LLM_INVOKERS_CSV))), key=_sort_key)


def _diagnostic(repo, file, line, kind, detail):
    return dict(repo=repo, file=file, line=line, kind=kind, detail=detail)


def select_test_edges(invokers: list[dict], tests: list[dict]) -> tuple[list[dict], list[dict]]:
    targets = defaultdict(list)
    for inv in _validated(invokers, "invokers"):
        if inv["kind"] == "direct":
            targets[inv["repo"], inv["qname"]].append(inv)
    edges, diagnostics = {}, []
    for row in _validated(tests, "tests"):
        if row["kind"] != "transitive" or not row["reason"].startswith("calls "):
            continue
        candidates = targets.get((row["repo"], row["reason"][6:]), [])
        if len(candidates) != 1:
            kind = "ambiguous_target" if candidates else "deferred_wrapper_edge"
            diagnostics.append(_diagnostic(row["repo"], row["file"], row["line"], kind, row["reason"]))
            continue
        inv = candidates[0]
        if (row["repo"], row["file"], row["line"]) == (inv["repo"], inv["file"], inv["line"]):
            continue
        edge = dict(repo=row["repo"], invoker_file=inv["file"], invoker_line=inv["line"],
                    invoker_qname=inv["qname"], test_file=row["file"], test_line=row["line"],
                    test_qname=row["qname"], reason=row["reason"])
        edges[tuple(edge.values())] = edge
    return (sorted(edges.values(), key=lambda r: tuple(r.values())),
            sorted(diagnostics, key=lambda r: (r["repo"], r["file"], r["line"], r["kind"], r["detail"])))


def _local_walk(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _local_walk(child)


def _definition(tree, line):
    return next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.lineno == line), None)


def _definition_identity(tree, line, qname, file, repo):
    """Check the lexical tail after the saved source module, without dispatch inference."""
    lexical = None

    def visit(node, parents):
        nonlocal lexical
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            parents = (*parents, node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == line:
                lexical = parents
        for child in ast.iter_child_nodes(node):
            visit(child, parents)

    visit(tree, ())
    if lexical is None:
        return "definition_missing", "Saved definition line not found"
    relative = _relative(file, _slug(repo))
    module = list(relative.with_suffix("").parts)
    if module and module[-1] == "__init__":
        module.pop()
    # Saved pyan namespaces may have a repository prefix containing hyphens.
    expected = tuple(module[-1:]) + lexical
    parts = tuple(qname.split("."))
    if len(parts) < len(expected) or parts[-len(expected):] != expected:
        return "definition_mismatch", f"Saved qname {qname!r} disagrees with source module/lexical tail {'.'.join(expected)!r}"
    return "ok", ""

def _bound_names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Lambda):
        return set()
    names = set()
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        names.add(node.id)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names.update(a.asname or a.name.split(".")[0] for a in node.names)
    if isinstance(node, (ast.MatchAs, ast.MatchStar, ast.ExceptHandler)) and node.name:
        names.add(node.name)
    if isinstance(node, ast.MatchMapping) and node.rest:
        names.add(node.rest)
    for child in ast.iter_child_nodes(node):
        names.update(_bound_names(child))
    return names

def _import_bindings(tree, function):
    """Explicit imports only, with conservative shadowing/conditional rejection."""
    bindings = {}
    for scope in (tree, function):
        for stmt in scope.body:
            if isinstance(stmt, ast.Import):
                for a in stmt.names:
                    bindings[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module:
                for a in stmt.names:
                    if a.name != "*":
                        bindings[a.asname or a.name] = stmt.module + "." + a.name
            elif stmt is not function:
                for name in _bound_names(stmt):
                    bindings.pop(name, None)
    for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
        bindings.pop(arg.arg, None)
    for arg in (function.args.vararg, function.args.kwarg):
        if arg:
            bindings.pop(arg.arg, None)
    return bindings


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return base + "." + node.attr if base else ""
    return ""


def _receiver_owner_conflict(tree, function, seed, target_qname, bindings):
    if not (isinstance(seed.func, ast.Attribute) and isinstance(seed.func.value, ast.Name)):
        return None
    components = target_qname.split(".")
    if len(components) < 2 or not components[-2][0].isupper():
        return None
    owner = components[-2]
    receiver = seed.func.value.id
    constructors = {}
    rebound = {a.arg for a in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)}
    rebound.update(a.arg for a in (function.args.vararg, function.args.kwarg) if a)
    imported = set()
    for scope in (tree, function):
        for stmt in scope.body:
            for node in _local_walk(stmt):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imported.update(a.asname or a.name.split(".")[0] for a in node.names)

    def constructor(call):
        if not isinstance(call, ast.Call):
            return None
        dotted = _dotted(call.func)
        if not dotted:
            return None
        root, _, rest = dotted.partition(".")
        if root in rebound or (root in imported and root not in bindings):
            return None
        resolved = bindings.get(root, root) + ("." + rest if rest else "")
        short = resolved.rsplit(".", 1)[-1]
        return short if short and short[0].isupper() else None

    for stmt in function.body:
        if any(node is seed for node in _local_walk(stmt)):
            observed = constructors.get(receiver)
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Assert, ast.Return, ast.Raise)):
                observed = None
            if observed and observed != owner:
                return (f"Saved target owner {owner!r} conflicts with explicit receiver constructor {observed!r}; "
                        "target identification is unresolved (inheritance may explain the relationship).")
            return None
        writes = _bound_names(stmt)
        if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.TryStar, ast.With, ast.AsyncWith, ast.Match)):
            constructors.clear()
        # Attribute/subscript writes and calls using an existing receiver may mutate it.
        for node in _local_walk(stmt):
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                base = node
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name):
                    writes.add(base.id)
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                        and node.func.value.id in constructors):
                    writes.add(node.func.value.id)
                for arg in (*node.args, *(k.value for k in node.keywords)):
                    writes.update(n.id for n in _local_walk(arg) if isinstance(n, ast.Name) and n.id in constructors)
        for name in writes:
            constructors.pop(name, None)
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            explicit = constructor(stmt.value)
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if explicit:
                for target in targets:
                    if isinstance(target, ast.Name):
                        constructors[target.id] = explicit
        if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
            rebound.update(writes)
        if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            return None
    return None

def analyze_saved_test(source: str, test_line: int, target_qname: str) -> dict:
    if (not isinstance(target_qname, str) or not target_qname
            or not target_qname.rsplit(".", 1)[-1].isidentifier()
            or any(not part or part in {".", ".."} for part in target_qname.split("."))
            or any(c in target_qname for c in "/\\")
            or any(ord(c) < 32 for c in target_qname)):
        raise ValueError("target_qname must be a nonempty dotted Python name")
    result = dict(status="ok", detail="", checks=[], limitations=[DISPATCH_LIMIT])
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError) as exc:
        result.update(status="parse_error", detail=str(exc))
        return result
    function = _definition(tree, test_line)
    if function is None:
        result.update(status="definition_missing", detail=f"No function definition at {test_line}")
        return result
    bindings = _import_bindings(tree, function)
    calls = [n for stmt in function.body for n in _local_walk(stmt) if isinstance(n, ast.Call)]
    final = target_qname.rsplit(".", 1)[-1]
    matching = []
    for call in calls:
        name = call.func.attr if isinstance(call.func, ast.Attribute) else (
            call.func.id if isinstance(call.func, ast.Name) else "")
        alias_target = bindings.get(call.func.id) if isinstance(call.func, ast.Name) else None
        if name == final or alias_target == target_qname:
            matching.append(call)
    if len(matching) != 1:
        result.update(status="ambiguous_call" if matching else "no_matching_call",
                      detail=f"{len(matching)} syntactic calls match {target_qname}")
        return result
    seed = matching[0]
    conflict = _receiver_owner_conflict(tree, function, seed, target_qname, bindings)
    if conflict:
        result.update(status="receiver_owner_mismatch", detail=conflict)
        result["limitations"].append("Explicit receiver-owner conflicts abstain; inheritance and actual dispatch are unresolved.")
        return result
    env = {}

    def merge(parts):
        lines, names = set(), set()
        for flow in parts:
            if flow:
                lines.update(flow[0])
                names.update(flow[1])
        return (lines, names) if lines else None

    def expr(node):
        if node is seed:
            return {seed.lineno}, set()
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"len", "isinstance", "bool", "str"}:
                if len(node.args) == (2 if node.func.id == "isinstance" else 1) and not node.keywords:
                    result["limitations"].append("Builtin names may be shadowed; calls are not resolved.")
                    return expr(node.args[0])
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"startswith", "endswith"}:
                result["limitations"].append("Predicate receiver type and implementation are unresolved.")
                return expr(node.func.value)
            result["limitations"].append(f"Unresolved validator/call expression at line {node.lineno}")
            return None
        if isinstance(node, (ast.Await, ast.Attribute, ast.Subscript, ast.BinOp, ast.UnaryOp,
                             ast.BoolOp, ast.Compare, ast.Tuple, ast.List, ast.Set, ast.Dict,
                             ast.JoinedStr, ast.FormattedValue, ast.Slice)):
            return merge(expr(child) for child in ast.iter_child_nodes(node))
        return None

    def record(stmt, values, kind, failure=None):
        flow = merge(expr(v) for v in values)
        if flow:
            result["checks"].append(dict(call_line=seed.lineno, check_line=stmt.lineno,
                failure_line=failure.lineno if failure else None, check_kind=kind,
                call_source=ast.get_source_segment(source, seed) or "",
                check_source=ast.get_source_segment(source, stmt) or "",
                trace_lines=sorted(flow[0] | {stmt.lineno}), variables=sorted(flow[1]), evidence=EVIDENCE))

    def failure(stmt):
        if isinstance(stmt, ast.Raise):
            return True
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return False
        dotted = _dotted(stmt.value.func)
        if dotted == "self.fail":
            return True
        root, _, rest = dotted.partition(".")
        resolved = bindings.get(root, "") + ("." + rest if rest else "")
        return resolved == "pytest.fail"

    for stmt in function.body:
        if any(isinstance(n, ast.NamedExpr) for n in _local_walk(stmt)):
            for name in _bound_names(stmt):
                env.pop(name, None)
            result["limitations"].append(f"Assignment-expression flow unassessed at line {stmt.lineno}")
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                break
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            flow = expr(stmt.value)
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    env.pop(target.id, None)
                    if flow:
                        env[target.id] = flow[0] | {stmt.lineno}, flow[1] | {target.id}
                else:
                    for n in ast.walk(target):
                        if isinstance(n, ast.Name):
                            env.pop(n.id, None)
        elif isinstance(stmt, ast.Assert):
            record(stmt, [stmt.test], "assert")
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            name = _dotted(call.func)
            method = name.removeprefix("self.") if name.startswith("self.") else ""
            if method in VALUE_ASSERTS:
                operands = {"first", "second", "expr", "obj", "member", "container", "cls",
                            "text", "regex", "expected_regex"}
                record(stmt, list(call.args[:VALUE_ASSERTS[method]]) +
                       [k.value for k in call.keywords if k.arg in operands], "unittest_assert")
        elif isinstance(stmt, ast.If):
            if stmt.body and failure(stmt.body[0]):
                record(stmt, [stmt.test], "controlled_failure", stmt.body[0])
            for name in _bound_names(stmt):
                env.pop(name, None)
            result["limitations"].append(f"Compound if flow unassessed; written bindings killed at line {stmt.lineno}")
        elif isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            break
        else:
            for name in _bound_names(stmt):
                env.pop(name, None)
            if not isinstance(stmt, (ast.Pass, ast.Import, ast.ImportFrom)):
                result["limitations"].append(f"Unsupported {type(stmt).__name__} flow at line {stmt.lineno}")
    result["limitations"] = sorted(set(result["limitations"]))
    return result


def _slug(repo):
    parts = repo.split("/")
    if len(parts) != 2 or any(not p or p in {".", ".."} or any(c in p for c in "\\:") for p in parts):
        raise ValueError(f"Unsafe repository identity: {repo!r}")
    return "_".join(parts)


def _relative(value, slug):
    value = value.replace("\\", "/")
    if not value or value.startswith("/") or ntpath.splitdrive(value)[0] or ".." in value.split("/"):
        raise ValueError(f"Unsafe source path: {value!r}")
    if value.startswith(slug + "/"):
        value = value[len(slug) + 1:]
    parts = value.split("/")
    if any(p in {"", ".", ".."} or ":" in p for p in parts):
        raise ValueError(f"Unsafe source path: {value!r}")
    return Path(*parts)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SourceCache:
    """Only explicit candidate source paths; bounded decoded-source LRU."""
    def __init__(self, root, capacity=32):
        self.root = Path(root).absolute()
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hashes = {}

    def read(self, repo, file):
        key = repo, file
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        try:
            slug = _slug(repo)
            relative = _relative(file, slug)
            clone = self.root / slug
            path = clone / relative
            for part in (self.root, *self.root.parents):
                if part.is_symlink():
                    raise ValueError(f"Symlink source root: {part}")
            current = self.root
            for part in (slug, *relative.parts):
                current = current / part
                if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
                    raise ValueError(f"Symlink/junction source path: {current}")
            path.resolve().relative_to(clone.resolve())
            if not clone.is_dir():
                value = "missing_clone", f"Clone not found: {clone}", None
            elif not path.is_file():
                value = "missing_source", f"Source not found: {path}", None
            else:
                before = _sha(path)
                with tokenize.open(path) as handle:
                    source = handle.read()
                after = _sha(path)
                if before != after or (str(path) in self.hashes and self.hashes[str(path)] != before):
                    raise ValueError(f"Source changed while reading: {path}")
                self.hashes[str(path)] = before
                value = "ok", "", source
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            value = "source_error", str(exc), None
        self.cache[key] = value
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value


class SavedInputs:
    """Audit actual saved-table opens without intercepting application/source execution."""
    def __init__(self):
        self.before = {}
        self.active = False
        self.busy = False
        self.root = paths.REPO_ROOT.resolve()

    def add(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"Required saved input not found: {path}; restore saved artifacts, do not regenerate")
        self.before.setdefault(path, _sha(path))

    def observe(self, event, args):
        if not self.active or self.busy or event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        self.busy = True
        try:
            path = Path(os.fsdecode(args[0])).resolve()
            try:
                path.relative_to(self.root)
            except ValueError:
                return
            if path.suffix.lower() in {".csv", ".json", ".jsonl", ".gz"}:
                if any(token in path.name.lower() for token in ("pym2t", "method_test_baseline", "invoker_result_checks")):
                    raise ValueError(f"Forbidden unrelated artifact read: {path}")
                self.add(path)
        finally:
            self.busy = False

    def finish(self):
        result = {}
        for path, before in sorted(self.before.items()):
            after = _sha(path)
            if before != after:
                raise ValueError(f"Saved input changed during analysis: {path}")
            try:
                key = path.relative_to(self.root).as_posix()
            except ValueError:
                key = path.as_posix()
            result[key] = dict(sha256_before=before, sha256_after=after)
        return result


def _read_tests(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if set(FIELDS) - set(reader.fieldnames or []):
            raise ValueError(f"{path}: missing required CSV columns; fetch Git LFS content if this is a pointer")
        return list(_validated(reader, str(path)))


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list)
                             else int(v) if isinstance(v, bool) else v for k, v in row.items() if k in fields})


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _report(summary, invokers, edges, checks):
    total = summary["invokers"]["denominator"]
    count = summary["invokers"]["with_result_check_candidate"]
    percent = f"{100 * count / total:.3f}%" if total else "unavailable (empty denominator)"
    lines = ["# Saved-edge direct-invoker result-check candidates", "",
             f"Detected {count}/{total} canonical direct invokers ({percent}) with a local returned-result pass/fail candidate.",
             f"{summary['invokers']['with_saved_direct_test_edge']}/{total} invokers have a selected saved direct-target test edge; "
             f"{summary['test_edges']['selected']} unique edges and {summary['checks']['total']} candidate checks.",
             "", "This is an independent firstedge candidate frame. No PyMethods2Test was used.",
             "Only candidate invoker definition files and candidate test files were inspected. "
             "Other invokers remain in the denominator with source_status=not_assessed. "
             "A usable saved target edge, readable/parseable source, exact saved definitions and one syntactic target call "
             "are required for source assessment.", "", "## Strata", "", "| Stratum | Denominator | Candidate invokers |", "|---|---:|---:|"]
    for kind, counts in sorted(summary["by_stratum"].items()):
        lines.append(f"| {kind} | {counts['denominator']} | {counts['with_result_check_candidate']} |")
    lines.extend(["", "## Unresolved and missingness", "", json.dumps(summary["diagnostic_counts"], sort_keys=True),
                  "", f"Direct-test rows without a target edge: {summary['test_edges']['direct_test_rows_without_target_edge']}.",
                  f"Source-assessed edges: {summary['test_edges']['source_assessed']}/{summary['test_edges']['selected']}.",
                  "", "## Deterministic examples", ""])
    for label, values in (("Positive", checks[:3]), ("No detected candidate", [e for e in edges if e["source_status"] == "ok" and not e["candidate_checks"]][:3])):
        lines.append(f"{label}: " + ("; ".join(f"{v['repo']} {v['test_file']}:{v['test_line']} → {v['invoker_qname']}" for v in values) or "none"))
    lines.extend(["", "## Limits", ""] + ["- " + item for item in LIMITATIONS])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repos-dir", type=Path, default=paths.REPOS_DIR)
    parser.add_argument("--repo", action="append", default=[])
    args = parser.parse_args(argv)
    output = args.output_dir.absolute()
    tracker = SavedInputs()
    try:
        if output.exists():
            raise ValueError(f"Output directory already exists: {output}")
        for repo in args.repo:
            _slug(repo)
        for path in (paths.LLM_INVOKERS_CSV, paths.LLM_TESTS_CSV):
            tracker.add(path)
        sys.addaudithook(tracker.observe)
        tracker.active = True
        global _OBSERVED_POP
        _OBSERVED_POP = None
        invokers = load_direct_invokers()
        tests = _read_tests(paths.LLM_TESTS_CSV)
        tracker.active = False
        population = _OBSERVED_POP if _OBSERVED_POP is not None else {i["repo"] for i in invokers}
        tests = [t for t in tests if t["repo"] in population]
        if args.repo:
            wanted = set(args.repo)
            invokers = [i for i in invokers if i["repo"] in wanted]
            tests = [t for t in tests if t["repo"] in wanted]
        edges, diagnostics = select_test_edges(invokers, tests)
        direct_rows = sum(t["kind"] == "direct" for t in tests)
        from .slice_index import code_kind
        source_cache = SourceCache(args.repos_dir)
        inv_by_key = {}
        for inv in invokers:
            rel = _relative(inv["file"], _slug(inv["repo"]))
            inv.update(stratum=code_kind(rel.as_posix(), inv["qname"].rsplit(".", 1)[-1]),
                       source_status="not_assessed", saved_test_edges=0, candidate_checks=0,
                       has_result_check_candidate=0)
            inv_by_key[inv["repo"], inv["file"], inv["line"]] = inv
        checks = []
        definition_status = {}
        # Sorting by source file batches the source cache without walking clone trees.
        edges.sort(key=lambda e: (e["repo"], e["test_file"], e["test_line"], e["invoker_file"], e["invoker_line"]))
        for edge in edges:
            invkey = edge["repo"], edge["invoker_file"], edge["invoker_line"]
            inv = inv_by_key[invkey]
            inv["saved_test_edges"] += 1
            if invkey not in definition_status:
                status, detail, source = source_cache.read(inv["repo"], inv["file"])
                if status == "ok":
                    try:
                        tree = ast.parse(source)
                        status, detail = _definition_identity(tree, inv["line"], inv["qname"], inv["file"], inv["repo"])
                    except (SyntaxError, ValueError) as exc:
                        status, detail = "parse_error", str(exc)
                definition_status[invkey] = status, detail
                inv["source_status"] = status
                if status != "ok":
                    diagnostics.append(_diagnostic(inv["repo"], inv["file"], inv["line"], status, detail))
            status, detail = definition_status[invkey]
            edge.update(source_status=status, candidate_checks=0, target_evidence=TARGET_EVIDENCE)
            if status != "ok":
                continue
            status, detail, source = source_cache.read(edge["repo"], edge["test_file"])
            if status == "ok":
                try:
                    status, detail = _definition_identity(ast.parse(source), edge["test_line"], edge["test_qname"], edge["test_file"], edge["repo"])
                except (SyntaxError, ValueError) as exc:
                    status, detail = "parse_error", str(exc)
            if status == "ok":
                result = analyze_saved_test(source, edge["test_line"], edge["invoker_qname"])
                status, detail = result["status"], result["detail"]
                for item in result["checks"]:
                    checks.append({**edge, **item})
                edge["candidate_checks"] = len(result["checks"])
                inv["candidate_checks"] += edge["candidate_checks"]
                inv["has_result_check_candidate"] = int(inv["candidate_checks"] > 0)
                for limitation in result["limitations"]:
                    if limitation != DISPATCH_LIMIT:
                        diagnostics.append(_diagnostic(edge["repo"], edge["test_file"], edge["test_line"], "local_limitation", limitation))
            edge["source_status"] = status
            if status != "ok":
                diagnostics.append(_diagnostic(edge["repo"], edge["test_file"], edge["test_line"], status, detail))
        diagnostics = [dict(zip(("repo", "file", "line", "kind", "detail"), values)) for values in
                       sorted(set(tuple(d[k] for k in ("repo", "file", "line", "kind", "detail")) for d in diagnostics))]
        diagnostic_counts = dict(sorted(Counter(d["kind"] for d in diagnostics).items()))
        by_stratum = {}
        for kind in ("production", "test", "test_support"):
            rows = [i for i in invokers if i["stratum"] == kind]
            by_stratum[kind] = dict(denominator=len(rows), with_result_check_candidate=sum(i["has_result_check_candidate"] for i in rows))
        n = len(invokers)
        numerator = sum(i["has_result_check_candidate"] for i in invokers)
        summary = dict(subset=bool(args.repo),
            invokers=dict(denominator=n, with_saved_direct_test_edge=sum(i["saved_test_edges"] > 0 for i in invokers),
                          with_result_check_candidate=numerator, result_check_candidate_detection_rate=_rate(numerator, n)),
            test_edges=dict(selected=len(edges), source_assessed=sum(e["source_status"] == "ok" for e in edges),
                            with_result_check_candidate=sum(e["candidate_checks"] > 0 for e in edges),
                            direct_test_rows_without_target_edge=direct_rows,
                            deferred_wrapper_edges=diagnostic_counts.get("deferred_wrapper_edge", 0),
                            receiver_owner_mismatches=diagnostic_counts.get("receiver_owner_mismatch", 0)),
            checks=dict(total=len(checks)), by_stratum=by_stratum, limitations=LIMITATIONS,
            diagnostic_counts=diagnostic_counts,
            source_status_counts=dict(sorted(Counter(e["source_status"] for e in edges).items())))
        input_hashes = tracker.finish()
        output.mkdir(parents=True, exist_ok=False)
        _write_csv(output / "invokers.csv", ["repo", "file", "line", "qname", "stratum", "source_status",
                   "saved_test_edges", "candidate_checks", "has_result_check_candidate"], invokers)
        edge_fields = ["repo", "invoker_file", "invoker_line", "invoker_qname", "test_file", "test_line", "test_qname"]
        _write_csv(output / "test_edges.csv", edge_fields + ["source_status", "candidate_checks", "target_evidence"], edges)
        _write_csv(output / "checks.csv", edge_fields + ["call_line", "check_line", "failure_line", "check_kind", "evidence",
                   "call_source", "check_source", "trace_lines", "variables"], checks)
        _write_csv(output / "diagnostics.csv", ["repo", "file", "line", "kind", "detail"], diagnostics)
        _write_json(output / "summary.json", summary)
        (output / "REPORT.md").write_text(_report(summary, invokers, edges, checks), encoding="utf-8")
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT,
                                      capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            revision = "unavailable"
        manifest = dict(inputs=tracker.finish(), inspected_source_hashes=dict(sorted(source_cache.hashes.items())),
                        python=sys.version, options=dict(output_dir=str(output), repos_dir=str(args.repos_dir), repos=sorted(set(args.repo))),
                        revision=revision, outputs={p.name: _sha(p) for p in sorted(output.iterdir()) if p.name != "manifest.json"})
        _write_json(output / "manifest.json", manifest)
        print(f"Wrote {output}: {numerator}/{n} direct invokers with result-check candidates")
        return 0
    except (OSError, ValueError, ImportError, csv.Error, KeyError, TypeError) as exc:
        print(f"direct_invoker_test_flow: {exc}", file=sys.stderr)
        return 1
    finally:
        tracker.active = False


if __name__ == "__main__":
    raise SystemExit(main())
