"""Find every function in a repo that ever invokes an LLM.

This answers the question: "if a test calls this function, will an LLM call
happen?" — even when the call is buried under several wrapper layers.

The approach is two passes over the codebase:

  1. Direct (seed) pass.  Look at every function's body for call expressions
     that match a pattern in FRAMEWORK_CALLS (".invoke", "chat.completions.create",
     "messages.create", etc.).  Anything that matches is a direct invoker — the
     LLM call is right there in its body.

  2. Transitive pass.  Iterate: any function whose body calls something we've
     already flagged as an invoker is itself an invoker.  Repeat until a pass
     adds nothing new.  After N rounds, we've reached every function that's
     within N wrapper hops of a real LLM call.

The transitive pass needs to answer "this call to wrapper() — which fully
qualified function is that, exactly?"  We answer it with a per-file name map
(every name used in the file mapped to whatever absolute module path it
points to), built from the file's imports and top-level definitions.  Each
function also carries its own overlay of names imported inside its body, so
wrappers that lazy-import their dependency aren't invisible.

What's handled:

  - Top-level functions and methods on top-level classes are both indexed.
  - self.X(...) and cls.X(...) inside a method resolve to enclosing_class.X.
  - ClassName.X(...) resolves through the file's name map.
  - Function-local imports are tracked per-function (fi.local_names) and
    take precedence over file-level bindings during resolution.

What we can't see (Python's static-analysis ceiling):

  - obj.method() where obj is an instance variable assigned from a factory —
    resolving it would need type inference.  The seed pass still catches the
    call if .method matches a framework pattern; just the iteration link
    through obj is missing.
  - Calls dispatched through registries, callbacks, or dynamic getattr.
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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from astWrappers import SKIP_DIRS, matcher, file_imports
from FrameworkDict import FRAMEWORK_CALLS


REPOS_DIR = Path(__file__).parent / "repos"
CLONE_TIMEOUT_SEC = 300


# ── repo cloning ──────────────────────────────────────────────────────────────


def _on_rm_error(func, path, _):
    """Windows quirk: files inside .git/objects/ are marked read-only, so
    shutil.rmtree can't delete them by default.  Flip the write bit and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def shallow_clone(url: str, dest: Path) -> bool:
    """Run `git clone --depth 1` into dest.  Returns True on success.

    On timeout or git failure, prints the error to stderr and returns False
    so the caller can clean up and exit.
    """
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
    """Turn a git URL into a filesystem-safe folder name.

    'https://github.com/foo/bar.git' becomes 'foo_bar'.  Strips the scheme,
    host, and .git suffix; replaces slashes with underscores so the result
    is usable as a single directory name.
    """
    name = url.rstrip("/")
    if name.endswith(".git"):
        name = name[:-4]
    for sep in ("://", "github.com/", "git@github.com:"):
        if sep in name:
            name = name.split(sep, 1)[1]
    return name.replace("/", "_").replace("\\", "_")


def ensure_clone(url: str) -> Path:
    """Make sure the repo at `url` lives in repos/<slug>/, then return that path.

    Reruns are cheap: if the directory already exists we re-use it.  If a
    fresh clone fails partway through, we tear down the partial directory
    before bailing.
    """
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
    """One call expression found inside a function body.

    We keep two views of the same call because the seed and iteration passes
    ask different questions about it.  `text` is the source-form of the
    callable (used for pattern matching against FRAMEWORK_CALLS).  `root` is
    just the leftmost identifier (used for name-map lookups during iteration).
    """
    line: int
    text: str
    root: Optional[str]


@dataclass
class FunctionInfo:
    """One indexed function or method.

    `qname` is the fully qualified name: 'chatchat.server.llm.wrapper' for a
    top-level function, 'chatchat.server.llm.AgentClient.chat' for a method.

    `local_names` is the per-function name-map overlay — names bound by
    Import or ImportFrom statements inside this function's body.  Lets a
    wrapper that lazy-imports its dependency still resolve correctly.
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
    """Everything the analysis needs to know about a single .py file.

    Lives in the `contexts` dict keyed by `current_module` (e.g.
    'chatchat.server.llm').  `name_map` is what lets us answer "in this
    file, what does `wrapper` actually point to?" — without it, transitive
    resolution across files would be impossible.
    """
    rel_path: str
    current_pkg: str
    current_module: str
    name_map: dict[str, str]
    imported_frameworks: set[str]


# ── path & name plumbing ──────────────────────────────────────────────────────


def resolve_import(level: int, module: Optional[str], current_pkg: str) -> str:
    """Turn a relative or absolute ImportFrom into an absolute module path.

    Mirrors Python's own relative-import resolution rules:
      - level=0 is absolute (`from x import y`) — return module unchanged.
      - level=1 is `from .y import z` — append y to current_pkg.
      - level=2 is `from ..y import z` — strip one segment off current_pkg first.
    """
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
    """The leftmost identifier in an attribute/subscript/call chain.

    For `chain.invoke(...)` returns 'chain'.  For `foo[0].bar()` returns 'foo'.
    Returns None for expressions that don't reduce to a name — like an
    immediately-invoked lambda or a call on a literal.
    """
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
    """For each top-level statement in the file, record what name it binds
    and what fully-qualified thing that name refers to.

    Three import shapes plus local definitions:
      `import foo`               -> {'foo': 'foo'}
      `import foo.bar as fb`     -> {'fb':  'foo.bar'}
      `from x.y import z`        -> {'z':   'x.y.z'}
      `def my_func(): ...`       -> {'my_func': '<current_module>.my_func'}

    The returned dict is what later lets the iteration step ask "in this
    file, the call to `wrapper()` — which exact function is that?"
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
    """Translate a file's filesystem path into Python module notation.

    'chatchat/server/llm.py' relative to the repo root becomes
    ('chatchat.server', 'chatchat.server.llm').  An __init__.py file IS
    the package, so 'chatchat/__init__.py' becomes ('', 'chatchat').
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
    """Index one function or method.

    Walks the body once and pulls out two kinds of nodes:
      - Call expressions, which become CallSite entries.  The seed pass
        will pattern-match against them; the iteration pass will try to
        resolve them.
      - Import / ImportFrom statements, which become fi.local_names entries.
        Without these, a function that lazy-imports its dependency would
        look like it's calling a name that doesn't exist anywhere.
    """
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

    Nested functions and methods on inner (non-top-level) classes are
    intentionally skipped.  They're rare in practice and tracking them adds
    scope-handling complexity that doesn't pay off in measured recall.
    """
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
    """Read every .py file in `target` and return the two dicts that drive
    the rest of the pipeline.

    `functions` is keyed by qualified name and holds one FunctionInfo per
    indexed function/method.  `contexts` is keyed by module qualified name
    and holds one FileContext per file (its name map, which frameworks it
    imports, etc.).

    Files that don't parse cleanly are silently skipped — we don't want one
    syntax error in a giant repo to abort the analysis.
    """
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

        contexts[current_module] = FileContext(
            rel_path=rel,
            current_pkg=current_pkg,
            current_module=current_module,
            name_map=build_name_map(tree, current_pkg, current_module),
            imported_frameworks=file_imports(tree) & framework_keys,
        )
        for fi in collect_functions(tree, current_module, current_pkg, rel):
            functions[fi.qname] = fi
    return functions, contexts


# ── analysis passes ───────────────────────────────────────────────────────────


def seed_invokers(
    functions: dict[str, FunctionInfo],
    contexts: dict[str, FileContext],
) -> dict[str, str]:
    """Find every function whose body literally contains an LLM call.

    For each function, check its call expressions against the patterns of
    whichever frameworks its file imports.  The file's imported_frameworks
    set restricts the search so a file that only uses langchain doesn't get
    its calls tested against openai's patterns (and vice versa).

    Returns {qname -> "matches 'X' from <framework>"} for every direct hit.
    """
    # Pre-compile every pattern's matcher once, grouped by framework.  In a
    # large repo we'll evaluate these matchers many times per function.
    matchers_by_fw: dict[str, list[tuple[str, callable]]] = {
        fw: [(pat, matcher(pat)) for pat in pats]
        for fw, pats in FRAMEWORK_CALLS.items()
    }
    invokers: dict[str, str] = {}

    # Group functions by their containing module so we can look up the
    # file's imported_frameworks once per module instead of per function.
    funcs_by_module: dict[str, list[FunctionInfo]] = defaultdict(list)
    for qname, fi in functions.items():
        # For a method 'pkg.mod.Cls.method' the *module* is 'pkg.mod' — we
        # need to strip both the class and method off.  For a top-level
        # function 'pkg.mod.fn' we only strip the function name.
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


def _resolve_call(call: CallSite, ctx: FileContext, fi: FunctionInfo) -> Optional[str]:
    """Figure out which fully-qualified function a call site is targeting.

    Four shapes we know how to handle:
      - self.X(...) or cls.X(...) inside a method  -> enclosing_class.X
      - ClassName.X(...)                            -> ClassName_qname.X
      - bare_name(...)                              -> name map lookup
      - anything longer or weirder                  -> None (we don't try)

    Name lookups consult fi.local_names first (function-local imports), then
    fall back to ctx.name_map (the file's top-level bindings).  That order
    matches Python's actual scoping rules — an inner import shadows an outer
    binding.

    Returning None doesn't mean the call is invisible to the analysis; it
    just means the iteration pass can't link this call to a known invoker.
    The seed pass might still flag it via pattern matching.
    """
    if call.root is None:
        return None
    parts = call.text.split(".")
    if fi.is_method and call.root in ("self", "cls"):
        if len(parts) == 2:
            return f"{fi.enclosing_class}.{parts[1]}"
        return None
    if len(parts) == 2:
        cls_qname = fi.local_names.get(parts[0]) or ctx.name_map.get(parts[0])
        if cls_qname:
            return f"{cls_qname}.{parts[1]}"
        return None
    if len(parts) == 1:
        return fi.local_names.get(call.root) or ctx.name_map.get(call.root)
    return None


def iterate_once(
    functions: dict[str, FunctionInfo],
    contexts: dict[str, FileContext],
    invokers: dict[str, str],
) -> int:
    """Run one pass of the transitive closure.

    For each function that isn't an invoker yet, look at its body's calls.
    If any of them resolves to a known invoker, this function becomes a
    transitive invoker too.

    The crucial subtlety: we compare against `invokers` as it was at the
    start of this pass, not as it grows during it.  Additions are buffered
    in `new_invokers` and merged only at the end.  That's why each pass
    walks the call graph exactly one hop further — multi-hop chains require
    multiple passes, and the iteration count tells you how deep the wrapper
    layers go.

    Returns how many new invokers were added.  The outer caller loops until
    this returns 0 (fixed-point reached).
    """
    new_invokers: dict[str, str] = {}
    for qname, fi in functions.items():
        if qname in invokers:
            continue
        # Same module-key derivation as seed_invokers: methods strip class+method,
        # top-level functions strip just the function name.
        if fi.is_method:
            module = fi.enclosing_class.rsplit(".", 1)[0]
        else:
            module = qname.rsplit(".", 1)[0]
        ctx = contexts.get(module)
        if not ctx:
            continue
        for call in fi.calls:
            resolved = _resolve_call(call, ctx, fi)
            if not resolved:
                continue
            if resolved in invokers:
                new_invokers[qname] = f"calls {resolved}"
                break
    invokers.update(new_invokers)
    return len(new_invokers)


# ── reporting & CLI ───────────────────────────────────────────────────────────


def report(
    invokers: dict[str, str],
    functions: dict[str, FunctionInfo],
    contexts: dict[str, FileContext],
) -> None:
    """Print the invoker set grouped by file, sorted by line within each file.

    The reason string makes direct ("matches '.invoke' from langchain") and
    transitive ("calls some.qname") cases distinguishable at a glance.
    """
    by_file: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for qname, reason in invokers.items():
        fi = functions[qname]
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

    # URL targets get cloned; everything else has to be an existing directory.
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

    # Run the fixed-point loop.  Each iteration's count is one wrapper layer's
    # worth of new transitive invokers — that distribution is itself a useful
    # measure of how abstracted the repo's LLM use is.
    iteration = 0
    while True:
        iteration += 1
        added = iterate_once(functions, contexts, invokers)
        if added == 0:
            break
        print(f"# Iteration {iteration}: +{added} (total {len(invokers)})")

    transitive = len(invokers) - seed_count
    print(f"\n# Result: {len(invokers)} invokers ({seed_count} direct, {transitive} transitive)")

    report(invokers, functions, contexts)

    if args.json:
        args.json.write_text(json.dumps(invokers, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
