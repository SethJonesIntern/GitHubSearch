"""Find every function in a repo that ever invokes an LLM.

Two passes:
  1. Seed pass. Scan each function body for calls matching a pattern in
     FRAMEWORK_CALLS (".invoke", "chat.completions.create", "messages.create",
     and so on). A match means the LLM call is right there in the body.

  2. Transitive pass. Walk the call graph backwards from the seeds: anything
     that calls a known invoker is itself an invoker. Keep going until nothing
     new turns up.

The transitive pass relies on a static call graph (built by pyan3) to figure
out which function a given call actually refers to. We also keep a per-file
name map and per-function local imports around for the seed pass and reporting.

"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import stat
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from astWrappers import SKIP_DIRS, matcher, file_imports
from FrameworkDict import FRAMEWORK_CALLS


REPOS_DIR = Path(__file__).parent / "repos"
CLONE_TIMEOUT_SEC = 300


# ── repo cloning ──────────────────────────────────────────────────────────────


def _on_rm_error(func, path, _):
    """On Windows, files under .git/objects/ are read-only and rmtree chokes
    on them. Make the file writable and retry the delete."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def shallow_clone(url: str, dest: Path) -> bool:
    """git clone --depth 1 into dest. Returns True on success, or prints the
    error and returns False on timeout/failure."""
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
            check=True,
            timeout=CLONE_TIMEOUT_SEC,
            capture_output=True,
        )
        return True
    except subprocess.TimeoutExpired:
        print(f"clone timed out: {url}", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode(errors="ignore").strip()
        print(f"clone failed: {url} — {stderr[:200]}", file=sys.stderr)
    return False


def repo_slug(url: str) -> str:
    """Turn a git URL into a safe folder name, e.g.
    'https://github.com/foo/bar.git' -> 'foo_bar'."""
    name = url.rstrip("/")
    if name.endswith(".git"):
        name = name[:-4]
    for sep in ("://", "github.com/", "git@github.com:"):
        if sep in name:
            name = name.split(sep, 1)[1]
    return name.replace("/", "_").replace("\\", "_")


def ensure_clone(url: str) -> Path:
    """Clone url into repos/<slug>/ if it isn't there already, and return that
    path. Reuses an existing clone; cleans up a half-finished one on failure."""
    REPOS_DIR.mkdir(exist_ok=True)
    dest = REPOS_DIR / repo_slug(url)
    if dest.exists():
        print(f"# already cloned at {dest}, re-using")
        return dest
    print(f"# cloning {url} -> {dest}")
    if not shallow_clone(url, dest):
        if dest.exists():
            shutil.rmtree(dest, onerror=_on_rm_error)
        sys.exit(1)
    return dest


# ── data shapes ───────────────────────────────────────────────────────────────


@dataclass
class CallSite:
    """A single call expression in a function body. We keep two views of it:
    `text` is the unparsed callable (matched against FRAMEWORK_CALLS), and
    `root` is just the leftmost identifier (used for name-map lookups)."""
    line: int
    text: str
    root: Optional[str]


@dataclass
class FunctionInfo:
    """An indexed function or method.

    `qname` is the fully qualified name, e.g. 'chatchat.server.llm.wrapper' or
    'chatchat.server.llm.AgentClient.chat'. `local_names` holds imports made
    inside the function body, so wrappers that lazy-import still resolve.
    """
    qname: str
    file_path: str
    line: int
    calls: list[CallSite] = field(default_factory=list)
    is_method: bool = False
    enclosing_class: Optional[str] = None
    local_names: dict[str, str] = field(default_factory=dict)


@dataclass
class FileContext:
    """Per-file state, stored in `contexts` keyed by module name (e.g.
    'chatchat.server.llm'). `name_map` resolves a bare name used in the file
    to the fully qualified thing it points to."""
    rel_path: str
    current_pkg: str
    current_module: str
    name_map: dict[str, str]
    imported_frameworks: set[str]
    # Every top-level package imported anywhere in the file (incl. lazy imports
    # inside function bodies). `imported_frameworks` is just this set narrowed to
    # FRAMEWORK_CALLS; the full set is kept so other passes (e.g. the semantic-
    # evaluator scan) can detect their own packages without re-parsing the file.
    imports: set[str] = field(default_factory=set)


# ── path & name plumbing ──────────────────────────────────────────────────────


def resolve_import(level: int, module: Optional[str], current_pkg: str) -> str:
    """Resolve an ImportFrom to an absolute module path, following Python's own
    rules: level 0 is absolute, level 1 is relative to current_pkg, and each
    extra level strips another segment off current_pkg."""
    if level == 0:
        return module or ""
    pkg_parts = current_pkg.split(".") if current_pkg else []
    drop = level - 1
    if drop > 0:
        pkg_parts = pkg_parts[:-drop] if len(pkg_parts) >= drop else []
    if module:
        return ".".join(pkg_parts + [module])
    return ".".join(pkg_parts)


def root_name(node: ast.AST) -> Optional[str]:
    """The leftmost identifier in an attribute/subscript/call chain:
    'chain' for `chain.invoke(...)`, 'foo' for `foo[0].bar()`. Returns None
    when the expression doesn't bottom out in a plain name."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return None


def build_name_map(tree: ast.Module, current_pkg: str, current_module: str) -> dict[str, str]:
    """Map every top-level binding in the file to its fully qualified target:

      `import foo`            -> {'foo': 'foo'}
      `import foo.bar as fb`  -> {'fb':  'foo.bar'}
      `from x.y import z`     -> {'z':   'x.y.z'}
      `def my_func(): ...`    -> {'my_func': '<current_module>.my_func'}
    """
    names: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    names[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    names[top] = top
        elif isinstance(node, ast.ImportFrom):
            mod = resolve_import(node.level, node.module, current_pkg)
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                names[local] = f"{mod}.{alias.name}" if mod else alias.name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = f"{current_module}.{node.name}"
    return names


def derive_module(file_path: Path, repo_root: Path) -> tuple[str, str]:
    """Convert a file path to (package, module) in dotted notation.
    'chatchat/server/llm.py' -> ('chatchat.server', 'chatchat.server.llm').
    An __init__.py is its package, so 'chatchat/__init__.py' -> ('', 'chatchat').
    """
    rel = file_path.resolve().relative_to(repo_root.resolve())
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    current_module = ".".join(parts)
    current_pkg = ".".join(parts[:-1]) if len(parts) > 1 else ""
    return current_pkg, current_module


# ── indexing ──────────────────────────────────────────────────────────────────


def _make_function_info(
    node: ast.AST, qname: str, rel_path: str, current_pkg: str,
    is_method: bool = False, enclosing_class: Optional[str] = None,
) -> FunctionInfo:
    """Index one function or method by walking its body once and collecting
    every call (as a CallSite) and every local import (into fi.local_names)."""
    fi = FunctionInfo(
        qname=qname,
        file_path=rel_path,
        line=node.lineno,
        is_method=is_method,
        enclosing_class=enclosing_class,
    )
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            try:
                text = ast.unparse(sub.func)
            except Exception:
                continue
            fi.calls.append(CallSite(line=sub.lineno, text=text, root=root_name(sub.func)))
        elif isinstance(sub, ast.Import):
            for alias in sub.names:
                if alias.asname:
                    fi.local_names[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    fi.local_names[top] = top
        elif isinstance(sub, ast.ImportFrom):
            mod = resolve_import(sub.level, sub.module, current_pkg)
            for alias in sub.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                fi.local_names[local] = f"{mod}.{alias.name}" if mod else alias.name
    return fi


def collect_functions(
    tree: ast.Module, current_module: str, current_pkg: str, rel_path: str,
) -> list[FunctionInfo]:
    """Index every top-level function and every method on a top-level class.
    We skip nested functions and methods on inner classes: rare enough that
    the extra scope handling isn't worth it for the recall it buys."""
    out: list[FunctionInfo] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qname = f"{current_module}.{node.name}"
            out.append(_make_function_info(node, qname, rel_path, current_pkg))
        elif isinstance(node, ast.ClassDef):
            cls_qname = f"{current_module}.{node.name}"
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(_make_function_info(
                        sub, f"{cls_qname}.{sub.name}", rel_path, current_pkg,
                        is_method=True, enclosing_class=cls_qname,
                    ))
    return out


def index_repo(
    target: Path, repo_root: Path
) -> tuple[dict[str, FunctionInfo], dict[str, FileContext]]:
    """Parse every .py file under `target` and return the two dicts the rest
    of the pipeline runs on: `functions` (qname -> FunctionInfo) and
    `contexts` (module -> FileContext). Files that fail to parse are skipped,
    so one syntax error doesn't sink the whole run."""
    functions: dict[str, FunctionInfo] = {}
    contexts: dict[str, FileContext] = {}
    framework_keys = set(FRAMEWORK_CALLS.keys())

    for path in target.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        current_pkg, current_module = derive_module(path, repo_root)
        rel = path.resolve().relative_to(repo_root.resolve()).as_posix()

        all_imports = file_imports(tree)
        contexts[current_module] = FileContext(
            rel_path=rel,
            current_pkg=current_pkg,
            current_module=current_module,
            name_map=build_name_map(tree, current_pkg, current_module),
            imported_frameworks=all_imports & framework_keys,
            imports=all_imports,
        )
        for fi in collect_functions(tree, current_module, current_pkg, rel):
            functions[fi.qname] = fi
    return functions, contexts


# ── analysis passes ───────────────────────────────────────────────────────────


def seed_invokers(
    functions: dict[str, FunctionInfo],
    contexts: dict[str, FileContext],
) -> dict[str, str]:
    """Find every function whose body directly contains an LLM call.

    Each function's calls are only tested against the patterns for the
    frameworks its file actually imports, so a langchain file never gets
    checked against openai patterns and vice versa.

    Returns {qname -> "matches 'X' from <framework>"} for every direct hit.
    """
    # Compile each pattern's matcher once up front; we run them a lot.
    matchers_by_fw: dict[str, list[tuple[str, callable]]] = {
        fw: [(pat, matcher(pat)) for pat in pats]
        for fw, pats in FRAMEWORK_CALLS.items()
    }
    invokers: dict[str, str] = {}

    # Group by module so we look up imported_frameworks once per file.
    funcs_by_module: dict[str, list[FunctionInfo]] = defaultdict(list)
    for qname, fi in functions.items():
        # Strip down to the module: for a method 'pkg.mod.Cls.method' that
        # means dropping both class and method; for 'pkg.mod.fn' just the name.
        if fi.is_method:
            module = fi.enclosing_class.rsplit(".", 1)[0]
        else:
            module = qname.rsplit(".", 1)[0]
        funcs_by_module[module].append(fi)

    for module, fns in funcs_by_module.items():
        ctx = contexts.get(module)
        if not ctx or not ctx.imported_frameworks:
            continue
        active = [
            (pat, m, fw)
            for fw in ctx.imported_frameworks
            for (pat, m) in matchers_by_fw[fw]
        ]
        if not active:
            continue
        for fi in fns:
            for call in fi.calls:
                hit = next(((pat, fw) for pat, m, fw in active if m(call.text)), None)
                if hit:
                    pat, fw = hit
                    invokers[fi.qname] = f"matches '{pat}' from {fw}"
                    break
    return invokers


def build_call_graph(repo: Path, repo_root: Optional[Path] = None) -> dict[str, set[str]]:
    """Build a static call graph for the repo with pyan3.

    pyan3 handles the cases our own resolver couldn't: aliased imports,
    inheritance, closures, decorators, relative imports, and so on.

    Returns {caller_qname: {callee_qname, ...}}. pyan3 names nodes as
    namespace.name relative to `root` (e.g. "test_repo.collisions.wrapper_a.wrapper"),
    which matches index_repo's dotted qnames, so seed names line up directly.

    Edges with an unresolved endpoint (no namespace, or a wildcard name) are
    dropped since they can't take part in the BFS.
    """
    try:
        from pyan.analyzer import CallGraphVisitor
    except ImportError:
        sys.exit("pyan3 is required for the transitive pass: py -m pip install pyan3==2.6.0")

    entry_points = [
        str(p) for p in repo.rglob("*.py")
        if not any(part in SKIP_DIRS for part in p.parts)
    ]
    if not entry_points:
        return {}

    # Root pyan at the same base index_repo uses so module names match.
    root = repo_root if repo_root is not None else repo
    try:
        v = CallGraphVisitor(entry_points, root=str(root))
    except Exception as e:
        print(f"# Warning: pyan3 analysis failed: {e}", file=sys.stderr)
        return {}

    def node_qname(node) -> Optional[str]:
        if node.namespace is None or "*" in node.name:
            return None
        ns = node.namespace.lstrip(".")
        return f"{ns}.{node.name}" if ns else node.name

    graph: dict[str, set[str]] = defaultdict(set)
    for src, dsts in v.uses_edges.items():
        src_q = node_qname(src)
        if src_q is None:
            continue
        for dst in dsts:
            dst_q = node_qname(dst)
            if dst_q is None:
                continue
            graph[src_q].add(dst_q)

    return dict(graph)


def transitive_closure(
    seeds: dict[str, str],
    call_graph: dict[str, set[str]],
) -> dict[str, str]:
    """BFS out from the seed invokers over the reversed call graph
    (callee -> callers). Each confirmed invoker enqueues its callers, so we
    never rescan the whole function set."""
    callers_of: dict[str, set[str]] = defaultdict(set)
    for caller, callees in call_graph.items():
        for callee in callees:
            callers_of[callee].add(caller)

    invokers = dict(seeds)
    queue: deque[str] = deque(seeds)

    while queue:
        qname = queue.popleft()
        for caller in callers_of.get(qname, ()):
            if caller not in invokers:
                invokers[caller] = f"calls {qname}"
                queue.append(caller)

    return invokers


# ── reporting & CLI ───────────────────────────────────────────────────────────


def report(
    invokers: dict[str, str],
    functions: dict[str, FunctionInfo],
    contexts: dict[str, FileContext],
) -> None:
    """Print the invokers grouped by file and sorted by line. The reason
    string tells direct hits ("matches '.invoke' from langchain") apart from
    transitive ones ("calls some.qname") at a glance."""
    by_file: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for qname, reason in invokers.items():
        fi = functions.get(qname)
        if fi is None:
            continue  # a module-level node pyan3 resolved but we didn't index
        by_file[fi.file_path].append((qname.rsplit(".", 1)[-1], reason, fi.line))

    for file_path in sorted(by_file):
        print(f"\n{file_path}")
        for name, reason, line in sorted(by_file[file_path], key=lambda r: r[2]):
            print(f"  L{line:<5} {name:30}  <- {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("target", type=str,
                        help="Directory to scan, or a git URL/owner-repo to clone into repos/")
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="Parent of the top-level package; defaults to target's parent")
    parser.add_argument("--json", type=Path,
                        help="Also write the invoker map to this JSON file")
    args = parser.parse_args()

    # URLs get cloned; anything else must be an existing directory.
    if args.target.startswith(("http://", "https://", "git@")):
        target = ensure_clone(args.target).resolve()
    else:
        target = Path(args.target).resolve()
        if not target.is_dir():
            sys.exit(f"not a URL and not a directory: {args.target}")
    repo_root = (args.repo_root or target.parent).resolve()

    functions, contexts = index_repo(target, repo_root)
    print(f"# Indexed {len(functions)} top-level functions across {len(contexts)} modules")
    print(f"# repo_root = {repo_root}")
    print(f"# target    = {target}")

    invokers = seed_invokers(functions, contexts)
    seed_count = len(invokers)
    print(f"# Seed: {seed_count} direct invokers")

    print("# Building call graph with pyan3...")
    call_graph = build_call_graph(target, repo_root)
    print(f"# Call graph: {len(call_graph)} nodes")

    invokers = transitive_closure(invokers, call_graph)
    transitive = len(invokers) - seed_count
    print(f"\n# Result: {len(invokers)} invokers ({seed_count} direct, {transitive} transitive)")

    report(invokers, functions, contexts)

    if args.json:
        args.json.write_text(json.dumps(invokers, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
