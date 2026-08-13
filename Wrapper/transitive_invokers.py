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
import re
import time
import warnings
import shutil
import stat
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from astWrappers import SKIP_DIRS, matcher, file_imports
from false_positives import classify_fp
from FrameworkDict import (
    FRAMEWORK_CALLS,
    DSPY_MODULE_CLASSES,
    resolve_framework_imports,
)


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
    error and returns False on timeout/failure.

    core.longpaths=true is required on Windows: many repos have paths over the
    260-char MAX_PATH limit, which otherwise fetch fine but fail at checkout
    ("Clone succeeded, but checkout failed") and leave no working tree to analyze."""
    try:
        subprocess.run(
            ["git", "-c", "core.longpaths=true", "clone", "--depth", "1", "--quiet",
             url, str(dest)],
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
    # Names bound to a dspy module instance here (`pred = dspy.Predict(...)` -> "pred";
    # `self.prog = dspy.ChainOfThought(...)` -> "self.prog"). A later bare call on one
    # of these names is the dspy __call__ invocation; see seed_invokers.
    dspy_binds: set[str] = field(default_factory=set)


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


def _dspy_ctor_class(value: ast.AST) -> Optional[str]:
    """If `value` is a call to a dspy module constructor, return the class name.
    Handles both `dspy.Predict(...)` (Attribute) and imported `Predict(...)` (Name).
    Returns None otherwise (including for non-call values)."""
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    if isinstance(func, ast.Attribute):
        cls = func.attr
    elif isinstance(func, ast.Name):
        cls = func.id
    else:
        return None
    return cls if cls in DSPY_MODULE_CLASSES else None


def _bind_target_names(target: ast.AST) -> list[str]:
    """Assignment-target names we can key a dspy binding on: a plain `pred` (Name)
    or an attribute chain `self.prog` (Attribute, unparsed). Tuple/list targets are
    unpacked; anything else is ignored."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        try:
            return [ast.unparse(target)]
        except Exception:
            return []
    if isinstance(target, (ast.Tuple, ast.List)):
        out: list[str] = []
        for elt in target.elts:
            out.extend(_bind_target_names(elt))
        return out
    return []


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
        elif isinstance(sub, (ast.Assign, ast.AnnAssign)):
            # Bind names assigned from a dspy module constructor so a later bare call
            # on the name resolves to the dspy __call__ invocation (see seed_invokers).
            if _dspy_ctor_class(sub.value) is not None:
                targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                for tgt in targets:
                    fi.dspy_binds.update(_bind_target_names(tgt))
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
    target: Path, repo_root: Path, framework_calls: dict = FRAMEWORK_CALLS,
) -> tuple[dict[str, FunctionInfo], dict[str, FileContext]]:
    """Parse every .py file under `target` and return the two dicts the rest
    of the pipeline runs on: `functions` (qname -> FunctionInfo) and
    `contexts` (module -> FileContext). Files that fail to parse are skipped,
    so one syntax error doesn't sink the whole run.

    `framework_calls` supplies the keys used to decide which imports count as
    "framework imports" per file. Pass a union of dicts (e.g. LLM + eval) to
    capture both in a single parse, then run the matching passes per dict."""
    functions: dict[str, FunctionInfo] = {}
    contexts: dict[str, FileContext] = {}
    framework_keys = set(framework_calls.keys())

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

        contexts[current_module] = FileContext(
            rel_path=rel,
            current_pkg=current_pkg,
            current_module=current_module,
            name_map=build_name_map(tree, current_pkg, current_module),
            imported_frameworks=resolve_framework_imports(file_imports(tree),
                                                          framework_keys),
        )
        for fi in collect_functions(tree, current_module, current_pkg, rel):
            functions[fi.qname] = fi
    return functions, contexts


# ── analysis passes ───────────────────────────────────────────────────────────


def seed_invokers(
    functions: dict[str, FunctionInfo],
    contexts: dict[str, FileContext],
    framework_calls: dict = FRAMEWORK_CALLS,
) -> dict[str, str]:
    """Find every function whose body directly contains an LLM call.

    Each function's calls are only tested against the patterns for the
    frameworks its file actually imports, so a langchain file never gets
    checked against openai patterns and vice versa.

    `framework_calls` selects which pattern set to match (e.g. FRAMEWORK_CALLS
    for LLM calls, EVAL_CALLS for eval calls). A file's imported_frameworks may
    include keys outside this dict (when indexed against a union); those are
    skipped here.

    Returns {qname -> "matches 'X' from <framework>"} for every direct hit.
    """
    # Compile each pattern's matcher once up front; we run them a lot.
    matchers_by_fw: dict[str, list[tuple[str, callable]]] = {
        fw: [(pat, matcher(pat)) for pat in pats]
        for fw, pats in framework_calls.items()
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
            if fw in matchers_by_fw
            for (pat, m) in matchers_by_fw[fw]
        ]
        if not active:
            continue
        # dspy is invoked by calling a bound module instance (`pred(...)`), so its
        # invocation site can't be a text pattern — it's a bare local/attr call whose
        # name was bound from a dspy constructor. Gather the self.<attr> bindings at
        # CLASS scope so a module built in __init__ but called in forward still resolves
        # (local bare-name bindings apply within their own function only).
        dspy_active = "dspy" in ctx.imported_frameworks
        class_self_binds: dict[str, set[str]] = defaultdict(set)
        if dspy_active:
            for fi in fns:
                if fi.enclosing_class:
                    class_self_binds[fi.enclosing_class] |= {
                        b for b in fi.dspy_binds if b.startswith("self.")
                    }
        for fi in fns:
            binds = (
                fi.dspy_binds | class_self_binds.get(fi.enclosing_class, set())
                if dspy_active else set()
            )
            for call in fi.calls:
                if binds and call.text in binds:
                    invokers[fi.qname] = "matches '__call__' from dspy"
                    break
                # Seed only via a genuine match — skip false-positive collisions
                # (asyncio.run, tool.invoke, Mock assertions; see false_positives /
                # EXCLUSIONS.md §6) so a function isn't an invoker off a collision.
                hit = next(((pat, fw) for pat, m, fw in active
                            if m(call.text) and classify_fp(call.text, pat) is None), None)
                if hit:
                    pat, fw = hit
                    invokers[fi.qname] = f"matches '{pat}' from {fw}"
                    break
    return invokers


# pyan3 is all-or-nothing: a single file it can't parse (or its own scope bug on
# nested lambdas/comprehensions) throws and kills the graph for the WHOLE repo. We
# recover by dropping the offending file and retrying — this keeps pyan's (superior)
# resolution on everything else. Most repos need 0; the failing ones need 1-2.
_MAX_CG_EXCLUSIONS = int(os.environ.get("PYAN_MAX_EXCLUSIONS", "40"))
# Each exclusion re-parses the WHOLE repo, so a huge repo with many bad files (e.g.
# litellm: 55k functions + many nested-lambda test files) could churn for an hour.
# Bound the total resilient-retry time; a repo that blows the budget gives up (empty
# graph, later covered by the Joern fallback) rather than stalling the whole run.
# Both overridable via env for targeted "no-limit" reruns: PYAN_MAX_EXCLUSIONS,
# PYAN_TIME_BUDGET_SEC (set to 0 to disable the time-box entirely).
_CG_TIME_BUDGET_SEC = int(os.environ.get("PYAN_TIME_BUDGET_SEC", "480"))


def _module_name(path: str, root: Path) -> Optional[str]:
    try:
        return ".".join(Path(path).relative_to(root).with_suffix("").parts)
    except ValueError:
        return None


def _find_offending_file(err: str, entry_points: list[str], root: Path) -> Optional[str]:
    """From a pyan failure message, identify which entry-point file to drop:
    a parse error names it as '(file.py, line N)'; the scope bug names a qname
    ('Unknown scope 'pkg.mod.func...') we map back to its module file."""
    # Filename must not contain spaces/parens: pyan messages can carry an extra '('
    # (e.g. "'(' was never closed (test_x.py, line 488)"), and a greedy [^)]+ would
    # latch onto the wrong paren and capture garbage instead of the real file.
    m = re.search(r"\(([^()\s]+\.py), line", err)
    if m:
        want = Path(m.group(1)).name
        for ep in entry_points:
            if Path(ep).name == want:
                return ep
    m = re.search(r"Unknown scope '([^']+)'", err)
    if m:
        qn, best, best_len = m.group(1), None, -1
        for ep in entry_points:
            mn = _module_name(ep, root)
            if mn and (qn == mn or qn.startswith(mn + ".")) and len(mn) > best_len:
                best, best_len = ep, len(mn)
        return best
    return None


def build_call_graph(repo: Path, repo_root: Optional[Path] = None,
                     stats: Optional[dict] = None) -> dict[str, set[str]]:
    """Build a static call graph for the repo with pyan3.

    pyan3 handles the cases our own resolver couldn't: aliased imports,
    inheritance, closures, decorators, relative imports, and so on.

    Returns {caller_qname: {callee_qname, ...}}. pyan3 names nodes as
    namespace.name relative to `root` (e.g. "test_repo.collisions.wrapper_a.wrapper"),
    which matches index_repo's dotted qnames, so seed names line up directly.

    Resilient: if pyan chokes on a file, that file is excluded and the analysis is
    retried (up to _MAX_CG_EXCLUSIONS files) so one bad file doesn't zero the graph.
    Pass `stats` to receive {'excluded_files': N, 'cg_source': 'pyan'|'pyan_resilient'|'none'}.
    """
    try:
        from pyan.analyzer import CallGraphVisitor
    except ImportError:
        sys.exit("pyan3 is required for the transitive pass: py -m pip install pyan3==2.6.0")

    entry_points = [
        str(p) for p in repo.rglob("*.py")
        if not any(part in SKIP_DIRS for part in p.parts)
    ]
    root = repo_root if repo_root is not None else repo  # match index_repo's base

    def _record(source: str, excluded: list) -> None:
        if stats is not None:
            stats["excluded_files"] = len(excluded)
            stats["cg_source"] = source

    if not entry_points:
        _record("none", [])
        return {}

    excluded: list[str] = []
    v = None
    started = time.monotonic()
    while entry_points and len(excluded) <= _MAX_CG_EXCLUSIONS:
        if excluded and _CG_TIME_BUDGET_SEC and time.monotonic() - started > _CG_TIME_BUDGET_SEC:
            print(f"# pyan resilient retries exceeded {_CG_TIME_BUDGET_SEC}s after "
                  f"{len(excluded)} exclusions; giving up on {repo.name}", file=sys.stderr)
            _record("none", excluded)
            return {}
        try:
            # pyan re-parses every file each pass; suppress the SyntaxWarning flood
            # from analyzed repos' own code (e.g. unescaped regex strings).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = CallGraphVisitor(entry_points, root=str(root))
            break
        except Exception as e:  # noqa: BLE001 - pyan raises many parse/scope errors
            bad = _find_offending_file(str(e), entry_points, root)
            if bad is None:
                print(f"# Warning: pyan3 analysis failed (unrecoverable): {e}", file=sys.stderr)
                _record("none", excluded)
                return {}
            entry_points.remove(bad)
            excluded.append(Path(bad).name)
            if os.environ.get("PYAN_VERBOSE"):
                # emit each exclusion so a long resilient run is observable (count,
                # elapsed, which file, and the error class that triggered it).
                kind = "scope-bug" if "Unknown scope" in str(e) else type(e).__name__
                print(f"# [{repo.name}] exclusion {len(excluded):>3} "
                      f"({time.monotonic() - started:6.0f}s): dropped {Path(bad).name} "
                      f"[{kind}]", file=sys.stderr, flush=True)
    if v is None:
        print(f"# Warning: pyan3 still failing after {len(excluded)} exclusions",
              file=sys.stderr)
        _record("none", excluded)
        return {}
    if excluded:
        print(f"# pyan recovered after excluding {len(excluded)} file(s): "
              f"{', '.join(excluded[:5])}{' ...' if len(excluded) > 5 else ''}",
              file=sys.stderr)

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

    _record("pyan_resilient" if excluded else "pyan", excluded)
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
