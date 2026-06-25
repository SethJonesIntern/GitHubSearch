#!/usr/bin/env python
"""Build per-variable SubPDGs by querying Joern's CPG directly.

This script follows the slicing, but it no longer depends on
separately exported PDG DOT files. Instead it:

1. Uses a Python AST pass to discover per-function variables and their
   source-level defs/uses.
2. Queries Joern's CPG directly for method AST nodes plus DDG/CDG edges.
3. Builds a per-variable SubPDG by following the selected variable's DDG edges
   and then adding incoming CDG predicates.
4. Maps the SubPDG back to source lines and emits standalone Python
   subprograms.

The emitted source is a research artifact: metadata keeps the raw PDG-selected
lines separate from extra lines inserted only so the snippet remains parseable
and self-contained enough for downstream clone detection.
"""

from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures
import csv
import hashlib
import html
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Set, Tuple


DEFAULT_INPUT_DIR = Path("HumanEvalPrograms")
DEFAULT_OUTPUT_DIR = Path("HumanEvalPrograms_PDG_per_variable")
DEFAULT_CPG = Path("HumanEvalPrograms_PDG/.joern_work/program.cpg")
DEFAULT_CPG_DIR = Path("CPGs")
DEFAULT_DATASET_DIR = Path("the_stack_dataset")
DEFAULT_BATCH_OUTPUT_ROOT = Path("subprograms")
DEFAULT_JOERN = "joern"
DEFAULT_FILES_PER_CPG = 1000

# Directories never worth slicing inside a real repo checkout (vs. the flat
# single-file corpora the script was first written for).
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".eggs",
    "build",
    "dist",
    "site-packages",
    ".idea",
    ".vscode",
}

CODE_FIELDS = (
    "text",
    "code",
    "content",
    "program",
    "source",
    "source_code",
    "completion",
    "canonical_solution",
)

IGNORED_VARIABLES = {
    "",
    "self",
    "cls",
    "True",
    "False",
    "None",
    "List",
    "Dict",
    "Set",
    "Tuple",
    "Optional",
    "Union",
    "Any",
}

IGNORED_NODE_LABELS = {
    "METHOD",
    "METHOD_RETURN",
    "LITERAL",
    "FIELD_IDENTIFIER",
    "METHOD_REF",
    "TYPE_REF",
    "UNKNOWN",
}

MUTATING_METHOD_NAMES = {
    "add",
    "append",
    "appendleft",
    "clear",
    "difference_update",
    "discard",
    "extend",
    "insert",
    "intersection_update",
    "pop",
    "popitem",
    "popleft",
    "remove",
    "reverse",
    "rotate",
    "setdefault",
    "sort",
    "symmetric_difference_update",
    "update",
    "write",
    "writelines",
}

_WORKER_GRAPHS_BY_NAME: Dict[str, List[PdgGraph]] = {}


@dataclass(frozen=True)
class PdgNode:
    id: str
    label: str
    line: int
    code: str


@dataclass(frozen=True)
class PdgEdge:
    src: str
    dst: str
    kind: str
    detail: str


@dataclass
class PdgGraph:
    source_file: str
    method_name: str
    method_full_name: str
    cpg_path: Path
    nodes: Dict[str, PdgNode]
    edges: List[PdgEdge]


@dataclass
class StatementInfo:
    line: int
    end_line: int
    defs: Set[str] = field(default_factory=set)
    uses: Set[str] = field(default_factory=set)
    kind: str = ""
    node: Optional[ast.AST] = None


@dataclass
class FunctionInfo:
    source_file: Path
    name: str
    def_line: int
    end_line: int
    import_lines: Set[int]
    statement_by_line: Dict[int, StatementInfo]
    init_lines: Dict[str, Set[int]]
    def_lines: Dict[str, Set[int]]
    use_lines: Dict[str, Set[int]]


@dataclass
class VariableSlice:
    variable: str
    criterion_lines: Set[int]
    seed_node_ids: Set[str]
    node_ids: Set[str]
    edges: List[PdgEdge]
    pdg_lines: Set[int]
    standalone_lines: Set[int]


def _normalize_method_name(raw: str) -> str:
    name = html.unescape(raw).strip()
    if name.startswith("<module>."):
        name = name[len("<module>.") :]
    if ":" in name:
        name = name.split(":", 1)[-1]
    return name


def _b64decode(value: str) -> str:
    return base64.b64decode(value.encode("ascii")).decode("utf-8")


_CPG_GRAPH_EXPORT_SCRIPT = textwrap.dedent(
    """
    import io.shiftleft.semanticcpg.language._
    import scala.util.Try
    import java.io.PrintWriter
    import java.nio.charset.StandardCharsets
    import java.util.Base64

    def b64(value: String): String =
      Base64.getEncoder.encodeToString(Option(value).getOrElse("").getBytes(StandardCharsets.UTF_8))

    def safeCode(node: io.shiftleft.codepropertygraph.generated.nodes.StoredNode): String =
      Try(node.property("CODE").toString).getOrElse("").replace("\\n", " ")

    def safeName(node: io.shiftleft.codepropertygraph.generated.nodes.StoredNode): String =
      Try(node.property("NAME").toString).getOrElse("")

    def safeLine(node: io.shiftleft.codepropertygraph.generated.nodes.StoredNode): Int =
      Try(node.property("LINE_NUMBER").toString.toInt).getOrElse(-1)

    val output = "__OUTPUT_PATH__"
    val fileFilter = "__FILE_FILTER__"
    {
      importCpg("__CPG_PATH__")
      val wantedFile = Option(fileFilter).getOrElse("").trim
      val writer = new PrintWriter(output, "UTF-8")
      val keepEdgeLabels = Set("REACHING_DEF", "CDG")

      val methods = cpg.method.l
        .filter(m => m.name != "<module>" && m.name != "<global>")
        .filter(m => !m.name.startsWith("<operator>"))
        .filter(m => wantedFile.isEmpty || Try(m.filename).getOrElse("") == wantedFile)
        .sortBy(m => (Try(m.filename).getOrElse(""), m.fullName))

      methods.foreach { method =>
        val filename = Try(method.filename).getOrElse("")
        val methodName = method.name
        val fullName = method.fullName
        writer.println(
          List("METHOD", b64(fullName), b64(filename), b64(methodName)).mkString("\\t")
        )

        val nodes = (List(method) ++ method.ast.l)
          .groupBy(_.id)
          .values
          .map(_.head)
          .toList
          .sortBy(_.id)

        val nodeIds = nodes.map(_.id).toSet
        nodes.foreach { node =>
          writer.println(
            List(
              "NODE",
              b64(fullName),
              node.id.toString,
              b64(node.label),
              safeLine(node).toString,
              b64(safeCode(node)),
              b64(safeName(node)),
            ).mkString("\\t")
          )
        }

        val seenEdges = scala.collection.mutable.LinkedHashSet[(Long, Long, String, String)]()
        nodes.foreach { node =>
          node.outE
            .filter(edge => keepEdgeLabels.contains(edge.label))
            .foreach { edge =>
              val srcId = edge.src.id
              val dstId = edge.dst.id
              if (nodeIds.contains(srcId) && nodeIds.contains(dstId)) {
                val detail = Try(edge.propertyMaybe.map(_.toString).getOrElse("")).getOrElse("")
                val label = if (edge.label == "REACHING_DEF") "DDG" else edge.label
                seenEdges += ((srcId, dstId, label, detail))
              }
            }
        }

        seenEdges.toList.sortBy(item => (item._1, item._2, item._3, item._4)).foreach {
          case (srcId, dstId, label, detail) =>
            writer.println(
              List(
                "EDGE",
                b64(fullName),
                srcId.toString,
                dstId.toString,
                b64(label),
                b64(detail),
              ).mkString("\\t")
            )
        }
      }

      writer.close()
    }
    """
).strip()


def load_cpg_graphs(
    cpg_path: Path,
    joern_bin: str,
    program_filter: Optional[str] = None,
    timeout: int = 1800,
) -> List[PdgGraph]:
    if not cpg_path.is_file():
        raise FileNotFoundError(f"CPG not found: {cpg_path}")

    joern_executable = _resolve_joern_executable(joern_bin)
    wanted_file = ""
    if program_filter:
        wanted_file = program_filter if program_filter.endswith(".py") else f"{program_filter}.py"

    with tempfile.TemporaryDirectory(prefix="cpg_graph_export_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        script_path = tmp_dir / "export_cpg_graphs.sc"
        output_path = tmp_dir / "cpg_graphs.tsv"
        # Bake the output path and file filter into the script as Scala vals
        # instead of passing them as `--param key=value`. On Windows the joern
        # launcher is a .bat that re-tokenizes `%*` and splits `key=value` on the
        # `=`, so --param is unusable there; inlining sidesteps it entirely.
        # Forward-slash the path so it is a safe Scala string literal (no `\`
        # escapes) and still valid for java.io on Windows.
        # Load the CPG inside the script too (importCpg with a quoted, forward-
        # slashed path) rather than as a positional arg: Joern injects an
        # unquoted importCpg(<path>) for the positional form, which a Windows
        # backslash path breaks.
        script_text = (
            _CPG_GRAPH_EXPORT_SCRIPT
            .replace("__CPG_PATH__", str(cpg_path).replace("\\", "/"))
            .replace("__OUTPUT_PATH__", str(output_path).replace("\\", "/"))
            .replace("__FILE_FILTER__", wanted_file.replace("\\", "/"))
        )
        script_path.write_text(script_text, encoding="utf-8")

        cmd = [
            str(joern_executable),
            "--script",
            str(script_path),
        ]

        # Joern's importCpg registers a project in its default workspace, a
        # `workspace/` dir created relative to the process CWD (and it piles up
        # repo.cpg/repo.cpg1/... one per query). Run with CWD inside the temp dir
        # so that workspace lands there and is removed with it — otherwise a batch
        # run litters the launch directory and the workspace grows unbounded.
        # All paths handed to Joern (script, CPG, output) are absolute, so the
        # CWD change does not affect them.
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=tmp_dir_name,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"joern CPG export failed: {detail}")
        if not output_path.is_file():
            raise RuntimeError(f"Joern did not create CPG graph export at {output_path}")

        graphs_by_full_name: Dict[str, PdgGraph] = {}
        for raw_line in output_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split("\t")
            if not parts:
                continue
            record_type = parts[0]
            if record_type == "METHOD":
                method_full_name = _b64decode(parts[1])
                source_file = _b64decode(parts[2])
                method_name = _b64decode(parts[3])
                graphs_by_full_name[method_full_name] = PdgGraph(
                    source_file=source_file,
                    method_name=method_name,
                    method_full_name=method_full_name,
                    cpg_path=cpg_path,
                    nodes={},
                    edges=[],
                )
                continue

            if record_type == "NODE":
                method_full_name = _b64decode(parts[1])
                graph = graphs_by_full_name.get(method_full_name)
                if graph is None:
                    continue
                node = PdgNode(
                    id=parts[2],
                    label=_b64decode(parts[3]),
                    line=int(parts[4]),
                    code=_b64decode(parts[5]),
                )
                graph.nodes[node.id] = node
                continue

            if record_type == "EDGE":
                method_full_name = _b64decode(parts[1])
                graph = graphs_by_full_name.get(method_full_name)
                if graph is None:
                    continue
                graph.edges.append(
                    PdgEdge(
                        src=parts[2],
                        dst=parts[3],
                        kind=_b64decode(parts[4]),
                        detail=_b64decode(parts[5]),
                    )
                )

        return list(graphs_by_full_name.values())


def _resolve_joern_executable(joern_value: str) -> Path:
    """Resolve --joern from a binary name, binary path, joern-cli dir, or install root."""
    candidates: List[Path] = []
    raw = Path(joern_value).expanduser()

    if raw.name == "joern" or raw.suffix:
        candidates.append(raw)
    if raw.is_dir():
        candidates.extend([raw / "joern", raw / "joern-cli" / "joern"])
    else:
        candidates.append(raw / "joern-cli" / "joern")

    env_home = os.environ.get("JOERN_HOME")
    if env_home:
        home = Path(env_home).expanduser()
        candidates.extend([home / "joern", home / "joern-cli" / "joern"])

    found_on_path = shutil.which(joern_value)
    if found_on_path:
        candidates.append(Path(found_on_path))
    found_default = shutil.which("joern")
    if found_default:
        candidates.append(Path(found_default))

    seen: Set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find the Joern executable. Pass --joern as the binary path, "
        "the joern-cli directory, or the Joern install root. "
        f"Checked: {checked}"
    )


def _target_names(target: ast.AST) -> Set[str]:
    names: Set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            names.update(_target_names(element))
    elif isinstance(target, ast.Starred):
        names.update(_target_names(target.value))
    elif isinstance(target, ast.Attribute):
        names.update(_target_names(target.value))
    elif isinstance(target, ast.Subscript):
        names.update(_target_names(target.value))
    return {name for name in names if _is_variable_name(name)}


class _NameUseVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.uses: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and _is_variable_name(node.id):
            self.uses.add(node.id)


def _names_loaded(node: ast.AST) -> Set[str]:
    visitor = _NameUseVisitor()
    visitor.visit(node)
    return visitor.uses


def _mutated_receiver_names(node: Optional[ast.AST]) -> Set[str]:
    if node is None:
        return set()

    names: Set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in MUTATING_METHOD_NAMES:
            continue
        names.update(_target_names(func.value))
    return {name for name in names if _is_variable_name(name)}


def _expr_defs_uses(node: Optional[ast.AST]) -> Tuple[Set[str], Set[str]]:
    if node is None:
        return set(), set()
    return _mutated_receiver_names(node), _names_loaded(node)


def _statement_defs_uses(stmt: ast.stmt) -> Tuple[Set[str], Set[str]]:
    defs: Set[str] = set()
    uses: Set[str] = set()

    if isinstance(stmt, ast.FunctionDef):
        uses.update(_names_loaded(stmt.returns) if stmt.returns else set())
        for decorator in stmt.decorator_list:
            uses.update(_names_loaded(decorator))
        return defs, uses

    if isinstance(stmt, ast.AsyncFunctionDef):
        uses.update(_names_loaded(stmt.returns) if stmt.returns else set())
        for decorator in stmt.decorator_list:
            uses.update(_names_loaded(decorator))
        return defs, uses

    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            defs.update(_target_names(target))
            uses.update(_names_loaded(target))
        expr_defs, expr_uses = _expr_defs_uses(stmt.value)
        defs.update(expr_defs)
        uses.update(expr_uses)
        return defs, uses

    if isinstance(stmt, ast.AnnAssign):
        defs.update(_target_names(stmt.target))
        uses.update(_names_loaded(stmt.target))
        annotation_defs, annotation_uses = _expr_defs_uses(stmt.annotation)
        defs.update(annotation_defs)
        uses.update(annotation_uses)
        if stmt.value is not None:
            value_defs, value_uses = _expr_defs_uses(stmt.value)
            defs.update(value_defs)
            uses.update(value_uses)
        return defs, uses

    if isinstance(stmt, ast.AugAssign):
        defs.update(_target_names(stmt.target))
        uses.update(_names_loaded(stmt.target))
        uses.update(_target_names(stmt.target))
        value_defs, value_uses = _expr_defs_uses(stmt.value)
        defs.update(value_defs)
        uses.update(value_uses)
        return defs, uses

    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        defs.update(_target_names(stmt.target))
        uses.update(_names_loaded(stmt.target))
        iter_defs, iter_uses = _expr_defs_uses(stmt.iter)
        defs.update(iter_defs)
        uses.update(iter_uses)
        return defs, uses

    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        for item in stmt.items:
            context_defs, context_uses = _expr_defs_uses(item.context_expr)
            defs.update(context_defs)
            uses.update(context_uses)
            if item.optional_vars is not None:
                defs.update(_target_names(item.optional_vars))
        return defs, uses

    if isinstance(stmt, ast.ExceptHandler):
        if stmt.name and _is_variable_name(stmt.name):
            defs.add(stmt.name)
        if stmt.type is not None:
            type_defs, type_uses = _expr_defs_uses(stmt.type)
            defs.update(type_defs)
            uses.update(type_uses)
        return defs, uses

    if isinstance(stmt, ast.Return):
        if stmt.value is not None:
            value_defs, value_uses = _expr_defs_uses(stmt.value)
            defs.update(value_defs)
            uses.update(value_uses)
        return defs, uses

    if isinstance(stmt, ast.If):
        test_defs, test_uses = _expr_defs_uses(stmt.test)
        defs.update(test_defs)
        uses.update(test_uses)
        return defs, uses

    if isinstance(stmt, ast.While):
        test_defs, test_uses = _expr_defs_uses(stmt.test)
        defs.update(test_defs)
        uses.update(test_uses)
        return defs, uses

    if isinstance(stmt, ast.Expr):
        value_defs, value_uses = _expr_defs_uses(stmt.value)
        defs.update(value_defs)
        uses.update(value_uses)
        return defs, uses

    uses.update(_names_loaded(stmt))
    return defs, uses


def _iter_args(args: ast.arguments) -> Iterator[ast.arg]:
    yield from args.posonlyargs
    yield from args.args
    if args.vararg is not None:
        yield args.vararg
    yield from args.kwonlyargs
    if args.kwarg is not None:
        yield args.kwarg


def _is_variable_name(name: str) -> bool:
    return (
        bool(name)
        and name not in IGNORED_VARIABLES
        and not name.startswith("__")
        and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name) is not None
        and re.match(r"^tmp\d+$", name) is None
    )


def _top_level_import_lines(tree: ast.Module) -> Set[int]:
    lines: Set[int] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            start = getattr(stmt, "lineno", None)
            end = getattr(stmt, "end_lineno", start)
            if isinstance(start, int) and isinstance(end, int):
                lines.update(range(start, end + 1))
    return lines


def _collect_statement_infos(func: ast.AST) -> Dict[int, StatementInfo]:
    statements: Dict[int, StatementInfo] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.stmt):
            continue
        line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", line)
        if not isinstance(line, int) or not isinstance(end_line, int):
            continue
        defs, uses = _statement_defs_uses(node)
        statements[line] = StatementInfo(
            line=line,
            end_line=end_line,
            defs=defs,
            uses=uses,
            kind=type(node).__name__,
            node=node,
        )
    return statements


def analyze_source(source_path: Path) -> List[FunctionInfo]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    import_lines = _top_level_import_lines(tree)
    functions: List[FunctionInfo] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        def_line = int(node.lineno)
        end_line = int(getattr(node, "end_lineno", node.lineno))
        statements = _collect_statement_infos(node)
        def_lines: Dict[str, Set[int]] = defaultdict(set)
        use_lines: Dict[str, Set[int]] = defaultdict(set)

        for arg in _iter_args(node.args):
            if _is_variable_name(arg.arg):
                def_lines[arg.arg].add(def_line)

        for stmt in statements.values():
            for name in stmt.defs:
                def_lines[name].add(stmt.line)
            for name in stmt.uses:
                use_lines[name].add(stmt.line)

        init_lines: Dict[str, Set[int]] = {
            name: {min(lines)}
            for name, lines in def_lines.items()
            if lines
        }

        functions.append(
            FunctionInfo(
                source_file=source_path,
                name=node.name,
                def_line=def_line,
                end_line=end_line,
                import_lines=import_lines,
                statement_by_line=statements,
                init_lines={key: set(value) for key, value in init_lines.items()},
                def_lines={key: set(value) for key, value in def_lines.items()},
                use_lines={key: set(value) for key, value in use_lines.items()},
            )
        )
    return functions


def _score_graph_for_function(graph: PdgGraph, function: FunctionInfo, source_lines: List[str]) -> int:
    score = 0
    if graph.method_name == function.name:
        score += 100
    for node in graph.nodes.values():
        if node.line == function.def_line and node.label == "METHOD":
            score += 50
        if function.def_line <= node.line <= function.end_line:
            score += 5
            if 1 <= node.line <= len(source_lines):
                source = source_lines[node.line - 1].strip()
                code = node.code.strip()
                if code and (source == code or source in code or code in source):
                    score += 5
    return score


def match_pdgs_to_functions(
    graphs: List[PdgGraph],
    functions_by_file: Dict[Path, List[FunctionInfo]],
) -> Dict[Tuple[Path, str], PdgGraph]:
    graphs_by_name = _group_graphs_by_method(graphs)
    matched: Dict[Tuple[Path, str], PdgGraph] = {}
    for source_file, functions in functions_by_file.items():
        matched.update(_match_graphs_for_file(source_file, functions, graphs_by_name))
    return matched


def _group_graphs_by_method(graphs: List[PdgGraph]) -> Dict[str, List[PdgGraph]]:
    graphs_by_name: Dict[str, List[PdgGraph]] = defaultdict(list)
    for graph in graphs:
        graphs_by_name[graph.method_name].append(graph)
    return graphs_by_name


def _match_graphs_for_file(
    source_file: Path,
    functions: List[FunctionInfo],
    graphs_by_name: Dict[str, List[PdgGraph]],
) -> Dict[Tuple[Path, str], PdgGraph]:
    matched: Dict[Tuple[Path, str], PdgGraph] = {}
    source_lines = source_file.read_text(encoding="utf-8").splitlines()
    for function in functions:
        candidates = [
            graph
            for graph in graphs_by_name.get(function.name, [])
            if Path(graph.source_file).name == source_file.name
        ]
        if not candidates:
            candidates = graphs_by_name.get(function.name, [])
        if not candidates:
            continue
        ranked = sorted(
            candidates,
            key=lambda graph: (
                _score_graph_for_function(graph, function, source_lines),
                Path(graph.source_file).name == source_file.name,
                graph.method_full_name,
            ),
            reverse=True,
        )
        matched[(source_file, function.name)] = ranked[0]
    return matched


def _var_pattern(variable: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(variable)}(?![A-Za-z0-9_])")


def _edge_mentions_variable(edge: PdgEdge, variable: str) -> bool:
    return _var_pattern(variable).search(edge.detail) is not None


def _node_is_statement(node: PdgNode) -> bool:
    return node.line > 0 and node.label not in IGNORED_NODE_LABELS


def _node_mentions_variable(node: PdgNode, variable: str) -> bool:
    return _node_is_statement(node) and _var_pattern(variable).search(node.code) is not None


def _bounded_reachable(
    seeds: Set[str],
    adjacency: Dict[str, Set[str]],
    max_depth: Optional[int],
) -> Set[str]:
    visited: Set[str] = set(seeds)
    queue: Deque[Tuple[str, int]] = deque((seed, 0) for seed in sorted(seeds))
    while queue:
        node_id, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for next_id in sorted(adjacency.get(node_id, set())):
            if next_id in visited:
                continue
            visited.add(next_id)
            queue.append((next_id, depth + 1))
    return visited


def _criterion_lines_for_variable(function: FunctionInfo, variable: str, criterion_mode: str) -> Set[int]:
    if criterion_mode == "last-use":
        use_lines = set(function.use_lines.get(variable, set()))
        if use_lines:
            last_use = max(use_lines)
            return {line for line in use_lines if line == last_use}
        fallback_lines = (
            set(function.def_lines.get(variable, set()))
            or set(function.init_lines.get(variable, set()))
        )
        if fallback_lines:
            last_line = max(fallback_lines)
            return {line for line in fallback_lines if line == last_line}
        return set()

    if criterion_mode == "all-mentions":
        return (
            set(function.init_lines.get(variable, set()))
            | set(function.def_lines.get(variable, set()))
            | set(function.use_lines.get(variable, set()))
        )

    raise ValueError(f"Unsupported criterion mode: {criterion_mode}")


def _seed_nodes_for_criterion(
    graph: PdgGraph,
    variable: str,
    criterion_lines: Set[int],
    criterion_mode: str,
) -> Set[str]:
    if not criterion_lines:
        return set()

    if criterion_mode == "all-mentions":
        seed_nodes = {
            node.id
            for node in graph.nodes.values()
            if node.line in criterion_lines or _node_mentions_variable(node, variable)
        }
    else:
        seed_nodes = {
            node.id
            for node in graph.nodes.values()
            if node.line in criterion_lines and (_node_is_statement(node) or _node_mentions_variable(node, variable))
        }

    for edge in graph.edges:
        if edge.kind != "DDG" or not _edge_mentions_variable(edge, variable):
            continue
        src_node = graph.nodes.get(edge.src)
        dst_node = graph.nodes.get(edge.dst)
        if src_node is not None and src_node.line in criterion_lines:
            seed_nodes.add(edge.src)
        if dst_node is not None and dst_node.line in criterion_lines:
            seed_nodes.add(edge.dst)
        if criterion_mode == "all-mentions":
            seed_nodes.add(edge.src)
            seed_nodes.add(edge.dst)

    if seed_nodes:
        return seed_nodes

    return {
        node.id
        for node in graph.nodes.values()
        if node.line in criterion_lines
    }


def _should_keep_special_use(line: int, variable: str, function: FunctionInfo) -> bool:
    stmt = function.statement_by_line.get(line)
    if not stmt:
        return False
    if stmt.kind in {"Return", "Break", "Continue", "Raise", "Yield", "YieldFrom"}:
        if variable in stmt.uses:
            return True
    for node in ast.walk(stmt.node):
        if isinstance(node, ast.Subscript):
            if variable in _names_loaded(node.slice):
                return True
    return False

def build_variable_slice(
    graph: PdgGraph,
    function: FunctionInfo,
    variable: str,
    criterion_mode: str,
    max_data_depth: Optional[int],
    standalone_closure: bool,
) -> Optional[VariableSlice]:
    variable_def_lines = (
        set(function.init_lines.get(variable, set()))
        | set(function.def_lines.get(variable, set()))
    )

    ddg_forward: Dict[str, Set[str]] = defaultdict(set)
    ddg_backward: Dict[str, Set[str]] = defaultdict(set)
    variable_ddg_endpoints: Set[str] = set()
    for edge in graph.edges:
        if edge.kind != "DDG":
            continue
        if _edge_mentions_variable(edge, variable):
            # Joern often routes object-state flow through mutating calls
            # (for example ``result.append(...)``) and through composite use
            # nodes (for example ``''.join(current_string)``). Restricting the
            # DDG to AST-only definitions drops those legitimate dependence
            # chains, so keep all variable-specific DDG edges here.
            ddg_forward[edge.src].add(edge.dst)
            ddg_backward[edge.dst].add(edge.src)
            variable_ddg_endpoints.add(edge.src)
            variable_ddg_endpoints.add(edge.dst)

    if criterion_mode == "all-mentions":
        criterion_lines = _criterion_lines_for_variable(function, variable, "all-mentions")
        seed_nodes = _seed_nodes_for_criterion(graph, variable, criterion_lines, "all-mentions")
        seed_nodes |= variable_ddg_endpoints
        if not seed_nodes:
            return None
        ddg_reachable = (
            _bounded_reachable(seed_nodes, ddg_forward, max_data_depth)
            | _bounded_reachable(seed_nodes, ddg_backward, max_data_depth)
        )
        core_nodes = ddg_reachable
    else:
        # 'last-use' and 'bidirectional' share the precise backward half: seed at
        # the variable's last use and walk DDG backward to everything that
        # *influences* it, keeping only the variable's own def/use/criterion lines.
        criterion_lines = _criterion_lines_for_variable(function, variable, "last-use")
        seed_nodes = _seed_nodes_for_criterion(graph, variable, criterion_lines, "last-use")
        if not seed_nodes:
            return None
        backward_reachable = _bounded_reachable(seed_nodes, ddg_backward, max_data_depth)
        core_nodes = {
            node_id for node_id in backward_reachable
            if node_id in graph.nodes and (
                graph.nodes[node_id].line in criterion_lines or
                graph.nodes[node_id].line in variable_def_lines or
                _should_keep_special_use(graph.nodes[node_id].line, variable, function)
            )
        }
        ddg_reachable = backward_reachable
        if criterion_mode == "bidirectional":
            # Forward half: from the variable's definitions, follow DDG forward to
            # every statement the value flows *into* (its uses and the values
            # derived from it). Kept unfiltered — those are downstream statements,
            # not this variable's own def/use lines, which is the point of the
            # forward slice.
            forward_seeds = {
                node.id
                for node in graph.nodes.values()
                if node.line in variable_def_lines
                and (_node_is_statement(node) or _node_mentions_variable(node, variable))
            } | {
                node_id
                for node_id in variable_ddg_endpoints
                if node_id in graph.nodes and graph.nodes[node_id].line in variable_def_lines
            }
            forward_reachable = _bounded_reachable(forward_seeds, ddg_forward, max_data_depth)
            core_nodes = core_nodes | forward_reachable
            ddg_reachable = ddg_reachable | forward_reachable

    # Add control predicates that control selected nodes.  This is the CDG part
    # of the SubPDG; only incoming CDG ancestors are followed so unrelated body
    # statements do not flood the variable slice.
    cdg_backward: Dict[str, Set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.kind == "CDG":
            cdg_backward[edge.dst].add(edge.src)
    cdg_reachable = _bounded_reachable(core_nodes, cdg_backward, None)
    
    important_nodes = core_nodes | cdg_reachable

    # Joern sometimes routes control dependence through synthetic UNKNOWN
    # iterator nodes. Once any node on a source line is selected, lift the
    # slice to statement-line granularity by including peer nodes on that line.
    selected_lines = {
        graph.nodes[node_id].line
        for node_id in important_nodes
        if node_id in graph.nodes and graph.nodes[node_id].line > 0
    }
    if criterion_mode in ("last-use", "bidirectional") and criterion_lines:
        upper_bound = max(criterion_lines)
        # Joern's DDG occasionally omits a direct edge from a variable's
        # definition to a later composite use of that same variable. Add the
        # variable's own earlier definition lines as a conservative fallback so
        # the backward slice still reflects the expected def-use chain.
        selected_lines |= {
            line
            for line in variable_def_lines
            if line <= upper_bound
        }

    graph_node_ids = ddg_reachable | cdg_reachable
    graph_node_ids |= {
        node.id
        for node in graph.nodes.values()
        if node.line in selected_lines
    }

    selected_edges = [
        edge
        for edge in graph.edges
        if edge.src in graph_node_ids and edge.dst in graph_node_ids and edge.kind in {"DDG", "CDG"}
    ]
    pdg_lines = {
        graph.nodes[node_id].line
        for node_id in graph_node_ids
        if node_id in graph.nodes and _node_is_statement(graph.nodes[node_id]) and graph.nodes[node_id].line in selected_lines
    }
    pdg_lines.add(function.def_line)
    standalone_lines = set(pdg_lines)
    if standalone_closure:
        standalone_lines = add_standalone_data_closure(standalone_lines, function)

    return VariableSlice(
        variable=variable,
        criterion_lines=criterion_lines,
        seed_node_ids=seed_nodes,
        node_ids=graph_node_ids,
        edges=selected_edges,
        pdg_lines=pdg_lines,
        standalone_lines=standalone_lines,
    )


def add_standalone_data_closure(lines: Set[int], function: FunctionInfo) -> Set[int]:
    """Add local definitions needed by selected source lines.

    This does not change the SubPDG identity; metadata records these as
    standalone-support lines.  The purpose is to avoid snippets such as
    ``if current_depth == 0:`` when ``current_depth`` was never initialized.
    """
    result = set(lines)
    all_local_defs = set(function.def_lines)
    changed = True
    while changed:
        changed = False
        selected_uses: Set[str] = set()
        for line in sorted(result):
            stmt = function.statement_by_line.get(line)
            if stmt is None:
                continue
            selected_uses.update(stmt.uses & all_local_defs)

        for name in selected_uses:
            needed = set(function.init_lines.get(name, set()))
            needed.update(line for line in function.def_lines.get(name, set()) if line <= max(result))
            missing = needed - result
            if missing:
                result.update(missing)
                changed = True
    return result


def _is_docstring_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _docstring_lines(source_text: str) -> Set[int]:
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return set()
    lines: Set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
            stmt = body[0]
            if _is_docstring_stmt(stmt):
                start = getattr(stmt, "lineno", None)
                end = getattr(stmt, "end_lineno", start)
                if isinstance(start, int) and isinstance(end, int):
                    lines.update(range(start, end + 1))
    return lines


def _all_import_lines(source_text: str) -> Set[int]:
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return set()
    lines: Set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        if isinstance(start, int) and isinstance(end, int):
            lines.update(range(start, end + 1))
    return lines


def _all_comment_lines(source_text: str) -> Set[int]:
    return {
        line_number
        for line_number, line in enumerate(source_text.splitlines(), start=1)
        if line.lstrip().startswith("#")
    }


def _add_enclosing_headers(lines: Set[int], function: FunctionInfo) -> Set[int]:
    result = set(lines)
    for line in list(lines):
        for stmt in function.statement_by_line.values():
            if stmt.line < line <= stmt.end_line and stmt.kind in {
                "If",
                "For",
                "AsyncFor",
                "While",
                "With",
                "AsyncWith",
                "Try",
                "ExceptHandler",
                "FunctionDef",
                "AsyncFunctionDef",
            }:
                result.add(stmt.line)
    return result


def _add_missing_branch_headers(lines: Set[int], source_lines: List[str]) -> Set[int]:
    result = set(lines)
    changed = True
    while changed:
        changed = False
        for line_number in sorted(result):
            if not 1 <= line_number <= len(source_lines):
                continue
            line = source_lines[line_number - 1]
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = _line_indent(line)
            for prev in range(line_number - 1, 0, -1):
                prev_line = source_lines[prev - 1]
                prev_stripped = prev_line.lstrip()
                if not prev_stripped or prev_stripped.startswith("#"):
                    continue
                prev_indent = _line_indent(prev_line)
                if prev_indent < indent:
                    header = prev_stripped.split(":")[0].strip()
                    if header in ("else", "finally") or prev_stripped.startswith(("else:", "finally:")):
                        if prev not in result:
                            result.add(prev)
                            changed = True
                    break
    return result


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _add_branch_companions(lines: Set[int], source_lines: List[str]) -> Set[int]:
    result = set(lines)
    changed = True
    while changed:
        changed = False
        for line_number in sorted(result):
            if not 1 <= line_number <= len(source_lines):
                continue
            line = source_lines[line_number - 1]
            stripped = line.lstrip()
            indent = _line_indent(line)
            if stripped.startswith(("elif ", "else:")):
                prefixes = ("if ", "elif ")
            elif stripped.startswith(("except ", "except:", "finally:")):
                prefixes = ("try:", "except ")
            else:
                continue
            for prev in range(line_number - 1, 0, -1):
                prev_line = source_lines[prev - 1]
                prev_stripped = prev_line.lstrip()
                if not prev_stripped or prev_stripped.startswith("#"):
                    continue
                prev_indent = _line_indent(prev_line)
                if prev_indent < indent:
                    break
                if prev_indent == indent and prev_stripped.startswith(prefixes):
                    if prev not in result:
                        result.add(prev)
                        changed = True
                    break
    return result


def _add_try_handlers(lines: Set[int], source_lines: List[str]) -> Set[int]:
    result = set(lines)
    for line_number in sorted(lines):
        if not 1 <= line_number <= len(source_lines):
            continue
        line = source_lines[line_number - 1]
        stripped = line.lstrip()
        if not stripped.startswith("try:"):
            continue
        indent = _line_indent(line)
        has_handler = False
        first_handler: Optional[int] = None
        for next_line in range(line_number + 1, len(source_lines) + 1):
            text = source_lines[next_line - 1]
            next_stripped = text.lstrip()
            if not next_stripped or next_stripped.startswith("#"):
                continue
            next_indent = _line_indent(text)
            if next_indent < indent:
                break
            if next_indent != indent:
                continue
            if next_stripped.startswith(("except ", "except:", "finally:")):
                first_handler = next_line
                if next_line in result:
                    has_handler = True
                break
        if not has_handler and first_handler is not None:
            result.add(first_handler)
    return result


def _add_complete_statement_lines(lines: Set[int], function: FunctionInfo) -> Set[int]:
    result = set(lines)
    compound_kinds = {
        "FunctionDef",
        "AsyncFunctionDef",
        "ClassDef",
        "If",
        "For",
        "AsyncFor",
        "While",
        "With",
        "AsyncWith",
        "Try",
        "ExceptHandler",
        "Match",
    }
    for stmt in function.statement_by_line.values():
        if stmt.kind in compound_kinds:
            continue
        if any(stmt.line <= line <= stmt.end_line for line in result):
            result.update(range(stmt.line, stmt.end_line + 1))
    return result


def _has_selected_body(
    line_number: int,
    selected_lines: Set[int],
    source_lines: List[str],
) -> bool:
    header = source_lines[line_number - 1]
    header_indent = _line_indent(header)
    for next_line in range(line_number + 1, len(source_lines) + 1):
        text = source_lines[next_line - 1]
        stripped = text.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _line_indent(text)
        if indent <= header_indent:
            return False
        if next_line in selected_lines:
            return True
    return False


def _needs_body(line: str) -> bool:
    stripped = line.strip()
    return stripped.endswith(":")


def render_subprogram(
    source_text: str,
    function: FunctionInfo,
    selected_lines: Set[int],
) -> Tuple[str, Set[int]]:
    source_lines = source_text.splitlines()
    base_indent = _line_indent(source_lines[function.def_line - 1])
    docstrings = _docstring_lines(source_text)
    import_lines = _all_import_lines(source_text)
    comment_lines = _all_comment_lines(source_text)
    selected = set(selected_lines)
    selected.add(function.def_line)
    selected = _add_complete_statement_lines(selected, function)
    selected = _add_enclosing_headers(selected, function)
    selected = _add_missing_branch_headers(selected, source_lines)
    selected = _add_branch_companions(selected, source_lines)
    selected = _add_try_handlers(selected, source_lines)
    selected -= docstrings
    selected -= import_lines
    selected -= comment_lines

    output_lines: List[str] = []
    previous_line = None
    for line_number in sorted(selected):
        if not function.def_line <= line_number <= function.end_line:
            continue
        line = source_lines[line_number - 1]
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if previous_line is not None and line_number > previous_line + 1:
            pass
        line_indent = _line_indent(line)
        if base_indent and line_indent >= base_indent:
            emitted_line = line[base_indent:]
        else:
            emitted_line = line
        output_lines.append(emitted_line)
        previous_line = line_number
        if _needs_body(line) and not _has_selected_body(line_number, selected, source_lines):
            pass_indent = max(0, _line_indent(line) - base_indent) + 4
            output_lines.append(" " * pass_indent + "pass")

    return "\n".join(output_lines).rstrip() + "\n", selected


def _safe_filename_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _edge_to_json(edge: PdgEdge) -> Dict[str, str]:
    return {"src": edge.src, "dst": edge.dst, "type": edge.kind, "detail": edge.detail}


def _node_to_json(node: PdgNode) -> Dict[str, Any]:
    return {"id": node.id, "label": node.label, "line": node.line, "code": node.code}


def process_program(
    source_file: Path,
    functions: List[FunctionInfo],
    matched_pdgs: Dict[Tuple[Path, str], PdgGraph],
    output_dir: Path,
    criterion_mode: str,
    max_data_depth: Optional[int],
    standalone_closure: bool,
    output_format: str,
    jsonl_detail: str,
    allowed_lines: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    source_text = source_file.read_text(encoding="utf-8")
    program_name = source_file.stem
    program_dir = output_dir / program_name
    subprogram_dir = program_dir / "subprograms"
    dedup_dir = program_dir / "deduplicated_subprograms"
    graph_dir = program_dir / "subpdgs"
    write_file_tree = output_format == "files"
    include_full_pdg = write_file_tree or jsonl_detail == "full"
    if write_file_tree:
        for path in (subprogram_dir, dedup_dir, graph_dir):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

        (program_dir / source_file.name).write_text(source_text, encoding="utf-8")

    seen_hashes: Dict[str, str] = {}
    subprograms: List[Dict[str, Any]] = []
    deduplicated_subprograms: List[Dict[str, Any]] = []
    syntax_errors: List[Dict[str, str]] = []

    for function in sorted(functions, key=lambda item: item.def_line):
        if allowed_lines is not None and function.def_line not in allowed_lines:
            continue
        graph = matched_pdgs.get((source_file, function.name))
        if graph is None:
            continue
        variables = sorted(function.init_lines)
        for variable in variables:
            slice_data = build_variable_slice(
                graph=graph,
                function=function,
                variable=variable,
                criterion_mode=criterion_mode,
                max_data_depth=max_data_depth,
                standalone_closure=standalone_closure,
            )
            if slice_data is None:
                continue
            code, emitted_lines = render_subprogram(
                source_text, function, slice_data.standalone_lines
            )
            safe_method = _safe_filename_part(function.name)
            safe_variable = _safe_filename_part(variable)
            filename = f"{program_name}_{safe_method}_{safe_variable}.py"
            content = code

            try:
                compile(content, filename, "exec")
            except SyntaxError as exc:
                syntax_errors.append({"file": filename, "error": str(exc)})

            subprogram_path = subprogram_dir / filename
            if write_file_tree:
                subprogram_path.write_text(content, encoding="utf-8")

            body_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
            duplicate_of = seen_hashes.get(body_hash)
            is_duplicate = duplicate_of is not None
            if not is_duplicate:
                seen_hashes[body_hash] = filename
                if write_file_tree:
                    (dedup_dir / filename).write_text(content, encoding="utf-8")

            graph_filename = filename.removesuffix(".py") + ".json"
            graph_payload = None
            if include_full_pdg:
                graph_payload = {
                    "source_file": source_file.name,
                    "method": function.name,
                    "method_full_name": graph.method_full_name,
                    "variable": variable,
                    "criterion_mode": criterion_mode,
                    "criterion_lines": sorted(slice_data.criterion_lines),
                    "seed_node_ids": sorted(slice_data.seed_node_ids),
                    "cpg_file": str(graph.cpg_path),
                    "pdg_lines": sorted(slice_data.pdg_lines),
                    "standalone_lines": sorted(emitted_lines),
                    "nodes": [
                        _node_to_json(graph.nodes[node_id])
                        for node_id in sorted(slice_data.node_ids)
                        if node_id in graph.nodes
                    ],
                    "edges": [_edge_to_json(edge) for edge in slice_data.edges],
                }
                if write_file_tree:
                    (graph_dir / graph_filename).write_text(
                        json.dumps(graph_payload, indent=2), encoding="utf-8"
                    )

            subprogram_record = {
                "file": filename,
                "method": function.name,
                "variable": variable,
                "criterion_mode": criterion_mode,
                "criterion_lines": sorted(slice_data.criterion_lines),
                "pdg_lines": sorted(slice_data.pdg_lines),
                "standalone_lines": sorted(emitted_lines),
                "line_count": len(code.splitlines()),
                "is_duplicate": is_duplicate,
                "duplicate_of": duplicate_of,
            }
            if include_full_pdg:
                subprogram_record["subpdg"] = graph_filename if write_file_tree else graph_payload
            if not write_file_tree:
                subprogram_record["code"] = content
            subprograms.append(subprogram_record)
            if not is_duplicate:
                dedup_record = dict(subprogram_record)
                deduplicated_subprograms.append(dedup_record)

    metadata = {
        "program": program_name,
        "original_file": str(program_dir / source_file.name) if write_file_tree else source_file.name,
        "source_file": source_file.name,
        "source_path": str(source_file),
        "subprogram_count": len(subprograms),
        "deduplicated_count": sum(1 for item in subprograms if not item["is_duplicate"]),
        "duplicate_count": sum(1 for item in subprograms if item["is_duplicate"]),
        "syntax_error_count": len(syntax_errors),
        "syntax_errors": syntax_errors,
        "subprograms": subprograms,
    }
    if write_file_tree:
        (program_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        metadata["deduplicated_subprograms"] = deduplicated_subprograms
        if jsonl_detail == "refined":
            return _refine_program_record(metadata)
        metadata["source_code"] = source_text
    return metadata


def _is_skipped_path(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return any(part in SKIP_DIRS for part in rel.parts)


def _select_source_files(
    input_dir: Path,
    program_filter: Optional[str],
    recursive: bool = False,
) -> List[Path]:
    globber = input_dir.rglob if recursive else input_dir.glob
    files = sorted(
        path
        for path in globber("*.py")
        if path.is_file() and not _is_skipped_path(path, input_dir)
    )
    if program_filter is None:
        return files
    wanted = program_filter if program_filter.endswith(".py") else f"{program_filter}.py"
    return [path for path in files if path.name == wanted or path.stem == program_filter]


# ── invoker-CSV function filter ────────────────────────────────────────────────
# When slicing the LLM applications we only want functions that reach an LLM
# call (Stage 5's llm_invokers_all.csv). The CSV records each invoker as
# (file, line) where ``file`` is a posix path relative to the clone's *parent*
# (so it carries the repo-slug segment) and ``line`` is the function's def line —
# the same ``node.lineno`` the slicer stores as ``function.def_line``. We match
# the path by component-wise suffix so the extra slug prefix is harmless.

FunctionFilter = Set[Tuple[Tuple[str, ...], int]]


def _norm_path_parts(path_str: str) -> Tuple[str, ...]:
    posix = path_str.replace("\\", "/")
    return tuple(part for part in posix.split("/") if part not in ("", "."))


def _parts_suffix_match(a: Tuple[str, ...], b: Tuple[str, ...]) -> bool:
    n = min(len(a), len(b))
    if n == 0:
        return False
    return a[-n:] == b[-n:]


def load_function_filter(csv_path: Path, repo: Optional[str] = None) -> FunctionFilter:
    """Load (path_parts, def_line) keys from an invokers CSV.

    ``repo`` restricts to one repo's rows (the CSV is multi-repo); omit it for a
    single-repo CSV. Rows without a usable file/line are skipped.
    """
    entries: FunctionFilter = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if repo is not None and (row.get("repo") or "") != repo:
                continue
            file_value = (row.get("file") or "").strip()
            line_value = (row.get("line") or "").strip()
            if not file_value or not line_value:
                continue
            try:
                line = int(line_value)
            except ValueError:
                continue
            parts = _norm_path_parts(file_value)
            if parts:
                entries.add((parts, line))
    return entries


def function_filter_from_rows(
    rows: Iterable[Dict[str, Any]],
    file_key: str = "file",
    line_key: str = "line",
) -> FunctionFilter:
    """Build a FunctionFilter from already-loaded invoker rows (e.g. the
    in-memory llm_invoker rows from the Stage-5 driver), bypassing the CSV.

    Same (path_parts, def_line) key shape as load_function_filter, so the
    suffix-match logic in _allowed_lines_for_file applies identically.
    """
    entries: FunctionFilter = set()
    for row in rows:
        file_value = str(row.get(file_key) or "").strip()
        raw_line = row.get(line_key)
        if not file_value or raw_line in (None, ""):
            continue
        try:
            line = int(raw_line)
        except (ValueError, TypeError):
            continue
        parts = _norm_path_parts(file_value)
        if parts:
            entries.add((parts, line))
    return entries


def _allowed_lines_for_file(
    function_filter: Optional[FunctionFilter],
    source_file: Path,
    input_dir: Path,
) -> Optional[Set[int]]:
    """def-lines to keep for one source file, or None when no filter is active.

    An empty set means "filter active, but this file has no invoker functions" —
    callers skip the file entirely.
    """
    if function_filter is None:
        return None
    try:
        rel_parts = _norm_path_parts(source_file.relative_to(input_dir).as_posix())
    except ValueError:
        rel_parts = _norm_path_parts(source_file.name)
    return {
        line
        for parts, line in function_filter
        if _parts_suffix_match(parts, rel_parts)
    }


def _cpg_stem_to_chunk_name(stem: str) -> str:
    match = re.match(r"^(\d+)k_(\d+)k$", stem)
    if match:
        return f"{match.group(1)}k-{match.group(2)}k"
    return stem.replace("_", "-")


def _chunk_name(chunk_index: int, files_per_chunk: int) -> str:
    start = chunk_index * files_per_chunk
    end = start + files_per_chunk
    if start % 1000 == 0 and end % 1000 == 0:
        return f"{start // 1000}k-{end // 1000}k"
    return f"{start}-{end}"


def _chunk_sort_key(path: Path) -> Tuple[int, str]:
    chunk_name = _cpg_stem_to_chunk_name(path.stem)
    match = re.match(r"^(\d+)k-(\d+)k$", chunk_name)
    if match:
        return int(match.group(1)), chunk_name
    return 10**12, chunk_name


def _chunk_index_for_cpg(path: Path) -> Optional[int]:
    chunk_name = _cpg_stem_to_chunk_name(path.stem)
    match = re.match(r"^(\d+)k-(\d+)k$", chunk_name)
    if match:
        return int(match.group(1))
    parts = chunk_name.split("-", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return int(parts[0])
    return None


def _default_jsonl_index_path(jsonl_path: Path, files_per_chunk: int) -> Path:
    return jsonl_path.with_name(f"{jsonl_path.name}.chunk_index_{files_per_chunk}.json")


def _infer_jsonl_code(record: Dict[str, Any]) -> str:
    prompt = record.get("prompt")
    canonical_solution = record.get("canonical_solution")
    if isinstance(prompt, str) and isinstance(canonical_solution, str):
        joiner = "" if not prompt or prompt.endswith(("\n", "\r")) else "\n"
        return f"{prompt}{joiner}{canonical_solution}"

    completion = record.get("completion")
    if isinstance(prompt, str) and isinstance(completion, str):
        joiner = "" if not prompt or prompt.endswith(("\n", "\r")) else "\n"
        return f"{prompt}{joiner}{completion}"

    for field_name in CODE_FIELDS:
        value = record.get(field_name)
        if isinstance(value, str) and value:
            return value
    return ""


def _jsonl_record_code(raw_line: bytes) -> Optional[str]:
    try:
        record = json.loads(raw_line.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    code = _infer_jsonl_code(record)
    return code or None


def ensure_jsonl_chunk_index(
    jsonl_path: Path,
    files_per_chunk: int,
    index_path: Optional[Path],
) -> Path:
    if not jsonl_path.is_file():
        raise SystemExit(f"JSONL dataset not found: {jsonl_path}")
    if files_per_chunk <= 0:
        raise SystemExit("--files-per-cpg must be positive")

    resolved_index_path = index_path or _default_jsonl_index_path(jsonl_path, files_per_chunk)
    if resolved_index_path.is_file():
        return resolved_index_path

    started_at = time.monotonic()
    tmp_path = resolved_index_path.with_suffix(resolved_index_path.suffix + ".tmp")
    entries: List[Dict[str, int]] = []
    kept = 0
    line_number = 0
    with jsonl_path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            line_number += 1
            if not raw_line.strip():
                continue
            code = _jsonl_record_code(raw_line)
            if not code:
                continue
            if kept % files_per_chunk == 0:
                entries.append(
                    {
                        "chunk_index": kept // files_per_chunk,
                        "byte_offset": offset,
                        "jsonl_line": line_number,
                        "global_index_start": kept,
                    }
                )
            kept += 1

    payload = {
        "jsonl_path": str(jsonl_path),
        "files_per_chunk": files_per_chunk,
        "valid_record_count": kept,
        "jsonl_line_count": line_number,
        "chunk_count": len(entries),
        "created_seconds": round(time.monotonic() - started_at, 3),
        "entries": entries,
    }
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(resolved_index_path)
    print(
        f"Wrote JSONL chunk index {resolved_index_path} "
        f"({len(entries):,} chunks, {kept:,} valid records)",
        flush=True,
    )
    return resolved_index_path


def _load_jsonl_index_entries(index_path: Path, files_per_chunk: int) -> List[Dict[str, int]]:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read JSONL chunk index {index_path}: {exc}") from exc
    if int(payload.get("files_per_chunk", -1)) != files_per_chunk:
        raise SystemExit(
            f"JSONL index {index_path} was built for files_per_chunk={payload.get('files_per_chunk')}, "
            f"but --files-per-cpg is {files_per_chunk}"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise SystemExit(f"JSONL index {index_path} is missing entries")
    return entries


def materialize_jsonl_chunk(
    *,
    jsonl_path: Path,
    index_entries: List[Dict[str, int]],
    chunk_index: int,
    files_per_chunk: int,
    temp_root: Path,
) -> Path:
    if chunk_index < 0 or chunk_index >= len(index_entries):
        raise SystemExit(f"Chunk index {chunk_index} is not available in JSONL index")

    chunk_dir = temp_root / _chunk_name(chunk_index, files_per_chunk)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    entry = index_entries[chunk_index]
    global_start = chunk_index * files_per_chunk
    written = 0
    with jsonl_path.open("rb") as handle:
        handle.seek(int(entry["byte_offset"]))
        while written < files_per_chunk:
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                continue
            code = _jsonl_record_code(raw_line)
            if not code:
                continue
            if not code.endswith("\n"):
                code += "\n"
            source_path = chunk_dir / f"{global_start + written:012d}.py"
            source_path.write_text(code, encoding="utf-8")
            written += 1

    if written == 0:
        raise SystemExit(f"No records could be materialized for JSONL chunk {chunk_index}")
    return chunk_dir


def _runtime_cpu_summary() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "os_cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST"),
        "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_ntasks": os.environ.get("SLURM_NTASKS"),
    }


def _format_runtime_cpu_summary(summary: Dict[str, Any]) -> str:
    return (
        f"host={summary.get('hostname')} "
        f"os_cpu_count={summary.get('os_cpu_count')} "
        f"SLURM_CPUS_ON_NODE={summary.get('slurm_cpus_on_node')} "
        f"SLURM_CPUS_PER_TASK={summary.get('slurm_cpus_per_task')} "
        f"SLURM_NTASKS={summary.get('slurm_ntasks')}"
    )


def _effective_worker_count(workers: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if workers == 0:
        return min(task_count, os.cpu_count() or 1)
    if workers < 0:
        raise ValueError("--workers must be 0 or a positive integer")
    return max(1, min(workers, task_count))


def _subprogram_code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _refined_subprogram_record(item: Dict[str, Any], include_code: bool) -> Dict[str, Any]:
    refined = {
        "file": item.get("file"),
        "method": item.get("method"),
        "variable": item.get("variable"),
        "criterion_mode": item.get("criterion_mode"),
        "criterion_lines": item.get("criterion_lines", []),
        "pdg_lines": item.get("pdg_lines", []),
        "standalone_lines": item.get("standalone_lines", []),
        "line_count": item.get("line_count", 0),
    }
    if include_code:
        code = item.get("code", "")
        refined["code_sha256"] = _subprogram_code_hash(code) if isinstance(code, str) else ""
        refined["code"] = code if isinstance(code, str) else ""
    return refined


def _refine_program_record(metadata: Dict[str, Any]) -> Dict[str, Any]:
    subprograms = [
        item for item in metadata.get("subprograms", [])
        if isinstance(item, dict)
    ]
    deduplicated = [
        item for item in metadata.get("deduplicated_subprograms", [])
        if isinstance(item, dict)
    ]
    if not deduplicated:
        deduplicated = [item for item in subprograms if not item.get("is_duplicate")]

    slice_map = []
    for item in subprograms:
        file_name = item.get("file")
        duplicate_of = item.get("duplicate_of")
        slice_map.append(
            {
                "file": file_name,
                "method": item.get("method"),
                "variable": item.get("variable"),
                "dedup_file": duplicate_of or file_name,
                "is_duplicate": bool(item.get("is_duplicate")),
                "criterion_mode": item.get("criterion_mode"),
                "criterion_lines": item.get("criterion_lines", []),
                "pdg_lines": item.get("pdg_lines", []),
                "standalone_lines": item.get("standalone_lines", []),
                "line_count": item.get("line_count", 0),
            }
        )

    return {
        "schema": "aegisma.refined_programs.v1",
        "program": metadata["program"],
        "original_file": metadata["original_file"],
        "source_file": metadata["source_file"],
        "source_path": metadata["source_path"],
        "subprogram_count": metadata["subprogram_count"],
        "deduplicated_count": metadata["deduplicated_count"],
        "duplicate_count": metadata["duplicate_count"],
        "syntax_error_count": metadata["syntax_error_count"],
        "deduplicated_subprograms": [
            _refined_subprogram_record(item, include_code=True)
            for item in deduplicated
        ],
        "slice_map": slice_map,
    }


def _compact_program_summary(metadata: Dict[str, Any], output_format: str) -> Dict[str, Any]:
    if output_format == "files":
        return metadata
    return {
        "program": metadata["program"],
        "original_file": metadata["original_file"],
        "source_file": metadata["source_file"],
        "source_path": metadata["source_path"],
        "subprogram_count": metadata["subprogram_count"],
        "deduplicated_count": metadata["deduplicated_count"],
        "duplicate_count": metadata["duplicate_count"],
        "syntax_error_count": metadata["syntax_error_count"],
        "syntax_errors": metadata.get("syntax_errors", []),
    }


def _append_program_jsonl(handle: Any, metadata: Dict[str, Any]) -> None:
    handle.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def _init_process_worker(graphs: List[PdgGraph]) -> None:
    global _WORKER_GRAPHS_BY_NAME
    _WORKER_GRAPHS_BY_NAME = _group_graphs_by_method(graphs)


def _process_source_file_worker(
    task: Tuple[Path, Path, str, Optional[int], bool, str, str, Optional[Set[int]]]
) -> Dict[str, Any]:
    (
        source_file,
        output_dir,
        criterion_mode,
        max_data_depth,
        standalone_closure,
        output_format,
        jsonl_detail,
        allowed_lines,
    ) = task
    try:
        functions = analyze_source(source_file)
    except (SyntaxError, UnicodeDecodeError) as exc:
        return {
            "status": "skipped_source_error",
            "file": str(source_file),
            "error": str(exc),
        }

    matched = _match_graphs_for_file(source_file, functions, _WORKER_GRAPHS_BY_NAME)
    summary = process_program(
        source_file=source_file,
        functions=functions,
        matched_pdgs=matched,
        output_dir=output_dir,
        criterion_mode=criterion_mode,
        max_data_depth=max_data_depth,
        standalone_closure=standalone_closure,
        output_format=output_format,
        jsonl_detail=jsonl_detail,
        allowed_lines=allowed_lines,
    )
    return {"status": "processed", "summary": summary}


def _process_source_file_serial(
    *,
    source_file: Path,
    output_dir: Path,
    graphs_by_name: Dict[str, List[PdgGraph]],
    criterion_mode: str,
    max_data_depth: Optional[int],
    standalone_closure: bool,
    output_format: str,
    jsonl_detail: str,
    allowed_lines: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    try:
        functions = analyze_source(source_file)
    except (SyntaxError, UnicodeDecodeError) as exc:
        return {
            "status": "skipped_source_error",
            "file": str(source_file),
            "error": str(exc),
        }

    matched = _match_graphs_for_file(source_file, functions, graphs_by_name)
    summary = process_program(
        source_file=source_file,
        functions=functions,
        matched_pdgs=matched,
        output_dir=output_dir,
        criterion_mode=criterion_mode,
        max_data_depth=max_data_depth,
        standalone_closure=standalone_closure,
        output_format=output_format,
        jsonl_detail=jsonl_detail,
        allowed_lines=allowed_lines,
    )
    return {"status": "processed", "summary": summary}


def _should_log_file_progress(completed: int, total: int, interval: int) -> bool:
    if interval <= 0:
        return completed == total
    return completed == 1 or completed == total or completed % interval == 0


def process_cpg(
    *,
    input_dir: Path,
    output_dir: Path,
    cpg: Path,
    joern: str,
    program: Optional[str],
    criterion_mode: str,
    max_data_depth: Optional[int],
    standalone_closure: bool,
    joern_timeout: int,
    workers: int,
    output_format: str,
    jsonl_detail: str,
    progress_interval: int,
    recursive: bool = False,
    function_filter: Optional[FunctionFilter] = None,
) -> Dict[str, Any]:
    total_started_at = time.monotonic()
    runtime_cpu_summary = _runtime_cpu_summary()
    source_files = _select_source_files(input_dir, program, recursive)
    if not source_files:
        raise RuntimeError(f"No source files found under {input_dir}")

    # Pair each source file with the def-lines to keep. With no filter that's
    # None (slice everything); with a filter, files holding no invoker function
    # are dropped here so we never parse or slice them.
    file_plan: List[Tuple[Path, Optional[Set[int]]]] = []
    for source_file in source_files:
        allowed_lines = _allowed_lines_for_file(function_filter, source_file, input_dir)
        if allowed_lines is not None and not allowed_lines:
            continue
        file_plan.append((source_file, allowed_lines))
    if function_filter is not None:
        print(
            f"Function filter active: {len(file_plan)}/{len(source_files)} files "
            f"contain at least one targeted function",
            flush=True,
        )

    graph_started_at = time.monotonic()
    graphs = load_cpg_graphs(cpg, joern, program, timeout=joern_timeout)
    cpg_graph_export_seconds = time.monotonic() - graph_started_at
    if not graphs:
        raise RuntimeError(f"No method graphs could be extracted from {cpg}")

    slicing_started_at = time.monotonic()
    skipped_source_errors: List[Dict[str, str]] = []
    file_total = len(file_plan)
    requested_workers = _effective_worker_count(workers, file_total)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "programs.jsonl"
    jsonl_tmp_path = output_dir / "programs.jsonl.tmp"
    jsonl_handle = None
    if output_format == "jsonl":
        jsonl_tmp_path.unlink(missing_ok=True)
        jsonl_handle = jsonl_tmp_path.open("w", encoding="utf-8")

    summaries: List[Dict[str, Any]] = []
    print(
        f"Processing {file_total:,} source files with {requested_workers} worker(s); "
        f"{_format_runtime_cpu_summary(runtime_cpu_summary)}",
        flush=True,
    )

    try:
        if requested_workers == 1:
            graphs_by_name = _group_graphs_by_method(graphs)
            for completed, (source_file, allowed_lines) in enumerate(file_plan, start=1):
                result = _process_source_file_serial(
                    source_file=source_file,
                    output_dir=output_dir,
                    graphs_by_name=graphs_by_name,
                    criterion_mode=criterion_mode,
                    max_data_depth=max_data_depth,
                    standalone_closure=standalone_closure,
                    output_format=output_format,
                    jsonl_detail=jsonl_detail,
                    allowed_lines=allowed_lines,
                )
                if result["status"] == "skipped_source_error":
                    skipped_source_errors.append({"file": result["file"], "error": result["error"]})
                    print(f"{source_file.name}: skipped source parse error")
                    continue
                full_summary = result["summary"]
                if jsonl_handle is not None:
                    _append_program_jsonl(jsonl_handle, full_summary)
                summary = _compact_program_summary(full_summary, output_format)
                summaries.append(summary)
                if _should_log_file_progress(completed, file_total, progress_interval):
                    print(
                        f"[{completed}/{file_total}] {source_file.name}: "
                        f"{summary['subprogram_count']} subprograms, "
                        f"{summary['deduplicated_count']} deduplicated, "
                        f"{summary['syntax_error_count']} syntax errors"
                    )
        else:
            tasks = [
                (source_file, output_dir, criterion_mode, max_data_depth, standalone_closure, output_format, jsonl_detail, allowed_lines)
                for source_file, allowed_lines in file_plan
            ]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=requested_workers,
                initializer=_init_process_worker,
                initargs=(graphs,),
            ) as executor:
                futures = {
                    executor.submit(_process_source_file_worker, task): task[0]
                    for task in tasks
                }
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    source_file = futures[future]
                    completed += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        skipped_source_errors.append({"file": str(source_file), "error": repr(exc)})
                        print(f"[{completed}/{file_total}] {source_file.name}: failed ({exc!r})")
                        continue
                    if result["status"] == "skipped_source_error":
                        skipped_source_errors.append({"file": result["file"], "error": result["error"]})
                        print(f"[{completed}/{file_total}] {source_file.name}: skipped source parse error")
                        continue
                    full_summary = result["summary"]
                    if jsonl_handle is not None:
                        _append_program_jsonl(jsonl_handle, full_summary)
                    summary = _compact_program_summary(full_summary, output_format)
                    summaries.append(summary)
                    if _should_log_file_progress(completed, file_total, progress_interval):
                        print(
                            f"[{completed}/{file_total}] {source_file.name}: "
                            f"{summary['subprogram_count']} subprograms, "
                            f"{summary['deduplicated_count']} deduplicated, "
                            f"{summary['syntax_error_count']} syntax errors"
                        )
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    slicing_seconds = time.monotonic() - slicing_started_at
    total_seconds = time.monotonic() - total_started_at
    overall = {
        "input_dir": str(input_dir),
        "cpg_file": str(cpg),
        "criterion_mode": criterion_mode,
        "output_format": output_format,
        "jsonl_detail": jsonl_detail if output_format == "jsonl" else None,
        "programs_jsonl": str(jsonl_path) if output_format == "jsonl" else None,
        "workers": requested_workers,
        "runtime_cpu_summary": runtime_cpu_summary,
        "cpg_graph_export_seconds": round(cpg_graph_export_seconds, 3),
        "slicing_seconds": round(slicing_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "program_count": len(summaries),
        "total_subprograms": sum(item["subprogram_count"] for item in summaries),
        "total_deduplicated": sum(item["deduplicated_count"] for item in summaries),
        "total_syntax_errors": sum(item["syntax_error_count"] for item in summaries),
        "skipped_source_error_count": len(skipped_source_errors),
        "skipped_source_errors": skipped_source_errors,
        "programs": summaries,
    }
    if output_format == "jsonl":
        jsonl_tmp_path.replace(jsonl_path)
    (output_dir / "overall_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(f"Wrote summary: {output_dir / 'overall_summary.json'}")
    print(
        f"EXPERIMENT_SECONDS per_variable_pdg_slicer cpg={cpg} "
        f"host={runtime_cpu_summary['hostname']} workers={requested_workers} "
        f"graph_export={cpg_graph_export_seconds:.3f} slicing={slicing_seconds:.3f} total={total_seconds:.3f}",
        flush=True,
    )
    return overall


def process_cpg_directory(
    *,
    cpg_dir: Path,
    dataset_dir: Path,
    dataset_jsonl: Optional[Path],
    jsonl_index: Optional[Path],
    files_per_cpg: int,
    output_root: Path,
    joern: str,
    criterion_mode: str,
    max_data_depth: Optional[int],
    standalone_closure: bool,
    start_dir_index: int,
    numofcpgs: Optional[int],
    skip_existing: bool,
    joern_timeout: int,
    workers: int,
    output_format: str,
    jsonl_detail: str,
    progress_interval: int,
    batch_summary_name: str,
) -> None:
    batch_started_at = time.monotonic()
    batch_cpu_summary = _runtime_cpu_summary()
    print(f"Batch runtime CPU summary: {_format_runtime_cpu_summary(batch_cpu_summary)}", flush=True)
    jsonl_index_entries: Optional[List[Dict[str, int]]] = None
    if dataset_jsonl is not None:
        resolved_index = ensure_jsonl_chunk_index(dataset_jsonl, files_per_cpg, jsonl_index)
        jsonl_index_entries = _load_jsonl_index_entries(resolved_index, files_per_cpg)

    range_end_index = None if numofcpgs is None else start_dir_index + numofcpgs
    indexed_cpgs = {
        index: path
        for path in cpg_dir.glob("*.cpg")
        if (index := _chunk_index_for_cpg(path)) is not None
    }
    if numofcpgs is None:
        cpg_files = [
            path
            for index, path in sorted(indexed_cpgs.items())
            if index >= start_dir_index
        ]
    else:
        missing_cpg_indexes = [
            index
            for index in range(start_dir_index, range_end_index)
            if index not in indexed_cpgs
        ]
        if missing_cpg_indexes:
            preview = ", ".join(str(item) for item in missing_cpg_indexes[:20])
            raise SystemExit(
                f"Missing {len(missing_cpg_indexes)} CPG files in requested range "
                f"{start_dir_index}..{range_end_index - 1}. "
                f"First missing chunk index(es): {preview}"
            )
        cpg_files = [
            indexed_cpgs[index]
            for index in range(start_dir_index, range_end_index)
        ]
    if not cpg_files:
        raise SystemExit(f"No .cpg files found under {cpg_dir}")

    batch_summaries: List[Dict[str, Any]] = []
    for index, cpg_path in enumerate(cpg_files, start=1):
        chunk_name = _cpg_stem_to_chunk_name(cpg_path.stem)
        output_dir = output_root / chunk_name
        summary_path = output_dir / "overall_summary.json"
        if skip_existing and summary_path.is_file():
            print(f"[{index}/{len(cpg_files)}] Skipping existing output for {chunk_name}: {summary_path}")
            batch_summaries.append(
                {
                    "chunk": chunk_name,
                    "cpg_file": str(cpg_path),
                    "output_dir": str(output_dir),
                    "status": "skipped",
                    "total_seconds": 0.0,
                }
            )
            continue
        if dataset_jsonl is None:
            input_dir = dataset_dir / chunk_name
        else:
            input_dir = None

        if dataset_jsonl is None and not input_dir.is_dir():
            print(f"[{index}/{len(cpg_files)}] Missing source chunk for {cpg_path}: {input_dir}")
            batch_summaries.append(
                {
                    "chunk": chunk_name,
                    "cpg_file": str(cpg_path),
                    "output_dir": str(output_dir),
                    "status": "missing_input",
                    "total_seconds": 0.0,
                }
            )
            continue

        temp_context: Optional[tempfile.TemporaryDirectory[str]] = None
        materialize_seconds = 0.0
        if dataset_jsonl is not None:
            if jsonl_index_entries is None:
                raise RuntimeError("JSONL index entries were not initialized")
            chunk_index = _chunk_index_for_cpg(cpg_path)
            if chunk_index is None:
                raise RuntimeError(f"Cannot infer chunk index from {cpg_path}")
            temp_context = tempfile.TemporaryDirectory(prefix=f"pvs_{chunk_name}_")
            materialize_started_at = time.monotonic()
            input_dir = materialize_jsonl_chunk(
                jsonl_path=dataset_jsonl,
                index_entries=jsonl_index_entries,
                chunk_index=chunk_index,
                files_per_chunk=files_per_cpg,
                temp_root=Path(temp_context.name),
            )
            materialize_seconds = time.monotonic() - materialize_started_at
            print(
                f"[{index}/{len(cpg_files)}] Hydrated {chunk_name} from JSONL "
                f"in {materialize_seconds:.3f}s at {input_dir}",
                flush=True,
            )

        print(f"[{index}/{len(cpg_files)}] Processing {cpg_path} with sources {input_dir}")
        cpg_started_at = time.monotonic()
        try:
            overall = process_cpg(
                input_dir=input_dir,
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
            )
        finally:
            if temp_context is not None:
                temp_context.cleanup()
        batch_summaries.append(
            {
                "chunk": chunk_name,
                "cpg_file": str(cpg_path),
                "output_dir": str(output_dir),
                "status": "processed",
                "program_count": overall["program_count"],
                "total_subprograms": overall["total_subprograms"],
                "total_syntax_errors": overall["total_syntax_errors"],
                "skipped_source_error_count": overall["skipped_source_error_count"],
                "source_materialize_seconds": round(materialize_seconds, 3),
                "cpg_graph_export_seconds": overall["cpg_graph_export_seconds"],
                "slicing_seconds": overall["slicing_seconds"],
                "total_seconds": overall["total_seconds"],
                "wall_seconds": round(time.monotonic() - cpg_started_at, 3),
            }
        )

    batch_total_seconds = time.monotonic() - batch_started_at
    output_root.mkdir(parents=True, exist_ok=True)
    batch_summary = {
        "cpg_dir": str(cpg_dir),
        "dataset_dir": str(dataset_dir),
        "dataset_jsonl": str(dataset_jsonl) if dataset_jsonl is not None else None,
        "files_per_cpg": files_per_cpg,
        "output_root": str(output_root),
        "criterion_mode": criterion_mode,
        "output_format": output_format,
        "jsonl_detail": jsonl_detail if output_format == "jsonl" else None,
        "workers": workers,
        "runtime_cpu_summary": batch_cpu_summary,
        "start_dir_index": start_dir_index,
        "numofcpgs": numofcpgs,
        "range_end_index_exclusive": range_end_index,
        "cpg_count": len(cpg_files),
        "total_seconds": round(batch_total_seconds, 3),
        "chunks": batch_summaries,
    }
    (output_root / batch_summary_name).write_text(
        json.dumps(batch_summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote batch summary: {output_root / batch_summary_name}")
    print(f"EXPERIMENT_SECONDS per_variable_pdg_slicer_batch={batch_total_seconds:.3f}")


def _slurm_allocated_nodes() -> List[str]:
    node_list = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not node_list:
        raise SystemExit("--slurm-distribute requires an active Slurm allocation with SLURM_JOB_NODELIST")
    completed = subprocess.run(
        ["scontrol", "show", "hostnames", node_list],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    nodes = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not nodes:
        raise SystemExit(f"Could not resolve Slurm node list: {node_list}")
    return nodes


def _split_count(total: int, parts: int) -> List[int]:
    base = total // parts
    remainder = total % parts
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def _split_workers(total_workers: int, node_count: int) -> List[int]:
    if total_workers <= 0:
        per_node_cores = os.cpu_count() or 1
        total_workers = max(1, (per_node_cores * node_count) - 1)
    return [max(1, count) for count in _split_count(total_workers, node_count)]


def _distributed_total_workers(total_workers: int, node_count: int) -> int:
    if total_workers > 0:
        return total_workers
    per_node_cores = os.cpu_count() or 1
    return max(1, (per_node_cores * node_count) - node_count)


def _slurm_allocation_summary(nodes: List[str], requested_total_workers: int) -> Dict[str, Any]:
    return {
        "launcher_host": socket.gethostname(),
        "nodes": nodes,
        "node_count": len(nodes),
        "launcher_os_cpu_count": os.cpu_count(),
        "requested_total_workers": requested_total_workers,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST"),
        "slurm_job_num_nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
        "slurm_job_cpus_per_node": os.environ.get("SLURM_JOB_CPUS_PER_NODE"),
        "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_ntasks": os.environ.get("SLURM_NTASKS"),
    }


def _child_slicer_command(
    *,
    args: argparse.Namespace,
    script_path: Path,
    node: str,
    cpus_per_task: int,
    start_dir_index: int,
    numofcpgs: int,
    workers: int,
    summary_name: str,
) -> List[str]:
    command = [
        "srun",
        "--exclusive",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus_per_task}",
        f"--nodelist={node}",
        sys.executable,
        str(script_path),
        "--cpg-dir",
        str(args.cpg_dir),
        "--dataset-dir",
        str(args.dataset_dir),
        "--output-root",
        str(args.output_root),
        "--joern",
        args.joern,
        "--criterion-mode",
        args.criterion_mode,
        "--max-data-depth",
        str(args.max_data_depth),
        "--start-dir-index",
        str(start_dir_index),
        "--numofcpgs",
        str(numofcpgs),
        "--joern-timeout",
        str(args.joern_timeout),
        "--workers",
        str(workers),
        "--output-format",
        args.output_format,
        "--jsonl-detail",
        args.jsonl_detail,
        "--files-per-cpg",
        str(args.files_per_cpg),
        "--progress-interval",
        str(args.progress_interval),
        "--batch-summary-name",
        summary_name,
    ]
    if args.dataset_jsonl is not None:
        command.extend(["--dataset-jsonl", str(args.dataset_jsonl)])
    if args.jsonl_index is not None:
        command.extend(["--jsonl-index", str(args.jsonl_index)])
    if args.skip_existing:
        command.append("--skip-existing")
    if args.no_standalone_closure:
        command.append("--no-standalone-closure")
    return command


def launch_slurm_cpg_lanes(
    *,
    args: argparse.Namespace,
    nodes: List[str],
    script_path: Path,
) -> None:
    lane_count = min(args.cpg_parallelism, args.numofcpgs)
    if lane_count <= 0:
        raise SystemExit("--cpg-parallelism must be positive when provided")

    total_workers = _distributed_total_workers(args.distributed_workers, len(nodes))
    workers_per_cpg = args.workers_per_cpg if args.workers_per_cpg > 0 else max(1, total_workers // lane_count)
    lane_counts = _split_count(args.numofcpgs, lane_count)

    lanes: List[Dict[str, Any]] = []
    shard_start = args.start_dir_index
    for lane_index, shard_count in enumerate(lane_counts):
        if shard_count <= 0:
            continue
        node = nodes[lane_index % len(nodes)]
        lanes.append(
            {
                "lane_index": lane_index,
                "node": node,
                "start": shard_start,
                "count": shard_count,
                "workers": workers_per_cpg,
            }
        )
        shard_start += shard_count

    node_cpu_totals: Dict[str, int] = defaultdict(int)
    node_lane_totals: Dict[str, int] = defaultdict(int)
    for lane in lanes:
        node_cpu_totals[lane["node"]] += lane["workers"]
        node_lane_totals[lane["node"]] += 1

    print(
        "Launching Slurm CPG lanes: "
        f"lane_count={len(lanes)} workers_per_cpg={workers_per_cpg} "
        f"node_lanes={dict(node_lane_totals)} node_requested_cpus={dict(node_cpu_totals)}",
        flush=True,
    )
    print(
        "CPG lane model: each lane processes a shard of CPGs sequentially, "
        "and many lanes run concurrently across the allocated nodes.",
        flush=True,
    )

    processes: List[Tuple[str, subprocess.Popen[Any]]] = []
    for lane in lanes:
        summary_name = (
            f"overall_summary_lane{lane['lane_index']:03d}_"
            f"{lane['node']}_{lane['start']}_{lane['count']}.json"
        )
        command = _child_slicer_command(
            args=args,
            script_path=script_path,
            node=lane["node"],
            cpus_per_task=lane["workers"],
            start_dir_index=lane["start"],
            numofcpgs=lane["count"],
            workers=lane["workers"],
            summary_name=summary_name,
        )
        print(
            f"[lane {lane['lane_index']:03d} {lane['node']}] "
            f"start={lane['start']} numofcpgs={lane['count']} workers={lane['workers']}",
            flush=True,
        )
        processes.append((f"lane {lane['lane_index']:03d} {lane['node']}", subprocess.Popen(command)))

    failures = []
    try:
        for label, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((label, return_code))
    except KeyboardInterrupt:
        for _, process in processes:
            process.terminate()
        raise

    if failures:
        details = ", ".join(f"{label} exited {code}" for label, code in failures)
        raise SystemExit(f"One or more Slurm CPG lanes failed: {details}")


def launch_slurm_distributed(args: argparse.Namespace, max_depth: Optional[int]) -> None:
    if args.cpg_dir is None:
        raise SystemExit("--slurm-distribute is only supported with --cpg-dir batch mode")
    if args.numofcpgs is None:
        raise SystemExit("--slurm-distribute requires --numofcpgs so the CPG range can be split safely")

    nodes = args.slurm_nodes.split(",") if args.slurm_nodes else _slurm_allocated_nodes()
    nodes = [node.strip() for node in nodes if node.strip()]
    if not nodes:
        raise SystemExit("No Slurm nodes available for --slurm-distribute")

    shard_counts = _split_count(args.numofcpgs, len(nodes))
    worker_counts = _split_workers(args.distributed_workers, len(nodes))
    active_shards = [
        (node, shard_count, worker_count)
        for node, shard_count, worker_count in zip(nodes, shard_counts, worker_counts)
        if shard_count > 0
    ]
    if not active_shards:
        raise SystemExit("Requested distributed range is empty")

    if args.dataset_jsonl is not None:
        ensure_jsonl_chunk_index(args.dataset_jsonl, args.files_per_cpg, args.jsonl_index)

    allocation_summary = _slurm_allocation_summary(nodes, args.distributed_workers)
    print(
        "Slurm allocation summary: "
        + json.dumps(allocation_summary, sort_keys=True),
        flush=True,
    )

    script_path = Path(__file__).resolve()
    if args.cpg_parallelism > 0:
        launch_slurm_cpg_lanes(args=args, nodes=nodes, script_path=script_path)
        return

    print(
        "Launching Slurm node shards: "
        + ", ".join(
            f"{node}: {count} CPGs, {workers} workers"
            for node, count, workers in active_shards
        ),
        flush=True,
    )

    processes: List[Tuple[str, subprocess.Popen[Any]]] = []
    shard_start = args.start_dir_index
    for node, shard_count, worker_count in active_shards:
        command = _child_slicer_command(
            args=args,
            script_path=script_path,
            node=node,
            cpus_per_task=worker_count,
            start_dir_index=shard_start,
            numofcpgs=shard_count,
            workers=worker_count,
            summary_name=f"overall_summary_{node}_{shard_start}_{shard_count}.json",
        )
        print(f"[{node}] start={shard_start} numofcpgs={shard_count} workers={worker_count}", flush=True)
        processes.append((node, subprocess.Popen(command)))
        shard_start += shard_count

    failures = []
    try:
        for node, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((node, return_code))
    except KeyboardInterrupt:
        for _, process in processes:
            process.terminate()
        raise

    if failures:
        details = ", ".join(f"{node} exited {code}" for node, code in failures)
        raise SystemExit(f"One or more Slurm node shards failed: {details}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-variable SubPDGs by querying Joern's CPG directly."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cpg", type=Path, default=DEFAULT_CPG)
    parser.add_argument(
        "--cpg-dir",
        type=Path,
        default=None,
        help="Batch mode: process all .cpg files in this directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Batch mode source root containing chunk directories such as 0k-1k.",
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=None,
        help=(
            "Batch mode source JSONL. When set, chunk source files are hydrated "
            "temporarily from this JSONL instead of reading --dataset-dir."
        ),
    )
    parser.add_argument(
        "--jsonl-index",
        type=Path,
        default=None,
        help="Optional path for the JSONL chunk index. Default: <jsonl>.chunk_index_<files-per-cpg>.json.",
    )
    parser.add_argument(
        "--files-per-cpg",
        type=int,
        default=DEFAULT_FILES_PER_CPG,
        help="Number of source programs represented by each CPG chunk.",
    )
    parser.add_argument(
        "--build-jsonl-index-only",
        action="store_true",
        help="Build the JSONL chunk index and exit.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_BATCH_OUTPUT_ROOT,
        help="Batch mode output root. Each CPG writes to output-root/<chunk-name>/.",
    )
    parser.add_argument(
        "--joern",
        default=DEFAULT_JOERN,
        help="Path to the joern executable used for direct CPG queries.",
    )
    parser.add_argument(
        "--pdg-dir",
        type=Path,
        default=None,
        help="Deprecated and ignored: this script now queries the CPG directly.",
    )
    parser.add_argument(
        "--export-pdgs",
        action="store_true",
        help="Deprecated and ignored: this script now queries the CPG directly.",
    )
    parser.add_argument("--program", help="Optional single program, e.g. HumanEval1.py")
    parser.add_argument(
        "--max-data-depth",
        type=int,
        default=-1,
        help="DDG traversal depth. -1 means unbounded within each variable's SubPDG.",
    )
    parser.add_argument(
        "--criterion-mode",
        choices=("bidirectional", "last-use", "all-mentions"),
        default="bidirectional",
        help=(
            "How to slice each variable. "
            "'bidirectional' (default) = backward slice (what influences the variable) "
            "plus forward slice (what the variable's value flows into); "
            "'last-use' = backward only; "
            "'all-mentions' = the broader legacy data-dependence neighborhood."
        ),
    )
    parser.add_argument(
        "--no-standalone-closure",
        action="store_true",
        help="Do not add extra local definitions used only to make subprograms standalone.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Discover source files recursively (for real repo checkouts with "
            "nested packages). Off by default to preserve the flat-corpus behavior."
        ),
    )
    parser.add_argument(
        "--invokers-csv",
        type=Path,
        default=None,
        help=(
            "Restrict slicing to functions listed in this invokers CSV "
            "(e.g. llm_invokers_all.csv): only functions whose (file, def-line) "
            "appears are sliced. Slices every function when omitted."
        ),
    )
    parser.add_argument(
        "--invokers-repo",
        default=None,
        help=(
            "When --invokers-csv is a multi-repo file, restrict to rows whose "
            "'repo' column equals this value."
        ),
    )
    parser.add_argument(
        "--numofcpgs",
        type=int,
        default=None,
        help="Batch mode limit: process at most this many CPG files.",
    )
    parser.add_argument(
        "--start-dir-index",
        type=int,
        default=0,
        help="Batch mode: start from this global chunk index, e.g. 4295 for 4295k-4296k.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Batch mode: skip chunks whose overall_summary.json already exists.",
    )
    parser.add_argument(
        "--output-format",
        choices=("jsonl", "files"),
        default="jsonl",
        help=(
            "Output layout. 'jsonl' writes one programs.jsonl per CPG chunk; "
            "'files' writes the legacy per-program directory tree."
        ),
    )
    parser.add_argument(
        "--jsonl-detail",
        choices=("refined", "full"),
        default="refined",
        help=(
            "JSONL payload detail. 'refined' keeps only parent/slice mapping and "
            "deduplicated code; 'full' preserves the older SubPDG nodes/edges metadata."
        ),
    )
    parser.add_argument(
        "--batch-summary-name",
        default="overall_summary.json",
        help="Batch mode summary filename under --output-root.",
    )
    parser.add_argument(
        "--joern-timeout",
        type=int,
        default=7200,
        help="Timeout in seconds for each Joern CPG graph export query.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel source-file workers per CPG. "
            "Use 0 for all available CPU cores, or pass a fixed count such as 47."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help=(
            "Print successful per-file progress every N files inside each CPG. "
            "0 only prints the final successful file; errors are always printed."
        ),
    )
    parser.add_argument(
        "--slurm-distribute",
        action="store_true",
        help="Launch one child slicer per allocated Slurm node and split the requested CPG range.",
    )
    parser.add_argument(
        "--slurm-nodes",
        default=None,
        help="Comma-separated node names for --slurm-distribute. Default: resolve SLURM_JOB_NODELIST.",
    )
    parser.add_argument(
        "--distributed-workers",
        type=int,
        default=0,
        help=(
            "Total worker count for --slurm-distribute. "
            "0 means all allocated node cores minus one."
        ),
    )
    parser.add_argument(
        "--cpg-parallelism",
        type=int,
        default=0,
        help=(
            "With --slurm-distribute, run this many CPG lanes concurrently "
            "instead of one large shard per node. Example: 40 lanes across two nodes."
        ),
    )
    parser.add_argument(
        "--workers-per-cpg",
        type=int,
        default=0,
        help=(
            "Worker count for each CPG lane when --cpg-parallelism is set. "
            "0 derives it from --distributed-workers / --cpg-parallelism."
        ),
    )
    args = parser.parse_args()

    if args.start_dir_index < 0:
        raise SystemExit("--start-dir-index must be non-negative")
    if args.files_per_cpg <= 0:
        raise SystemExit("--files-per-cpg must be positive")
    if args.cpg_parallelism < 0:
        raise SystemExit("--cpg-parallelism must be non-negative")
    if args.workers_per_cpg < 0:
        raise SystemExit("--workers-per-cpg must be non-negative")
    if args.progress_interval < 0:
        raise SystemExit("--progress-interval must be non-negative")

    if args.export_pdgs:
        print(
            "Ignoring --export-pdgs because per_variable_pdg_slicer.py now reads DDG/CDG directly from the CPG."
        )

    max_depth = None if args.max_data_depth < 0 else args.max_data_depth

    if args.build_jsonl_index_only:
        if args.dataset_jsonl is None:
            raise SystemExit("--build-jsonl-index-only requires --dataset-jsonl")
        ensure_jsonl_chunk_index(args.dataset_jsonl, args.files_per_cpg, args.jsonl_index)
        return

    if args.slurm_distribute:
        launch_slurm_distributed(args, max_depth)
        return

    if args.cpg_dir is not None:
        process_cpg_directory(
            cpg_dir=args.cpg_dir,
            dataset_dir=args.dataset_dir,
            dataset_jsonl=args.dataset_jsonl,
            jsonl_index=args.jsonl_index,
            files_per_cpg=args.files_per_cpg,
            output_root=args.output_root,
            joern=args.joern,
            criterion_mode=args.criterion_mode,
            max_data_depth=max_depth,
            standalone_closure=not args.no_standalone_closure,
            start_dir_index=args.start_dir_index,
            numofcpgs=args.numofcpgs,
            skip_existing=args.skip_existing,
            joern_timeout=args.joern_timeout,
            workers=args.workers,
            output_format=args.output_format,
            jsonl_detail=args.jsonl_detail,
            progress_interval=args.progress_interval,
            batch_summary_name=args.batch_summary_name,
        )
        return

    function_filter = (
        load_function_filter(args.invokers_csv, args.invokers_repo)
        if args.invokers_csv is not None
        else None
    )

    process_cpg(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cpg=args.cpg,
        joern=args.joern,
        program=args.program,
        criterion_mode=args.criterion_mode,
        max_data_depth=max_depth,
        standalone_closure=not args.no_standalone_closure,
        joern_timeout=args.joern_timeout,
        workers=args.workers,
        output_format=args.output_format,
        jsonl_detail=args.jsonl_detail,
        progress_interval=args.progress_interval,
        recursive=args.recursive,
        function_filter=function_filter,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
