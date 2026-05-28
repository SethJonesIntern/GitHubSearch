"""Find every function in a repo that transitively invokes an LLM.

Algorithm:
  1. Index every top-level function in the target directory; record its
     qualified name and the list of call expressions in its body.
  2. Build a per-file name-resolution map ({name as used in this file
     -> fully-qualified name it points to}) from the file's imports and
     top-level definitions.
  3. Seed pass: for each function, run the existing FRAMEWORK_CALLS
     substring/word-boundary matchers against each call site.  Direct
     pattern matches become the initial invoker set.
  4. Iterate: any function whose body calls a name that resolves to a
     known invoker becomes an invoker itself.  Repeat until no new
     additions (fixed-point).

Resolution coverage:
  - Top-level functions and methods inside top-level classes are both indexed.
  - self.X(...) and cls.X(...) inside a method resolve to enclosing_class.X.
  - ClassName.X(...) resolves via the file's name map to ClassName_qname.X.
  - Function-local imports are not added to the file's name map.
  - obj.method() where obj is an instance variable is not resolved (no type
    inference); the seed pass still catches it if .method matches a pattern.
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


def _on_rm_error(func, path, _):
    """Windows: git pack files are read-only; clear the flag and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def shallow_clone(url: str, dest: Path) -> bool:
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
    """Derive a safe folder name from a git URL or owner/repo slug."""
    name = url.rstrip("/")
    if name.endswith(".git"):
        name = name[:-4]
    for sep in ("://", "github.com/", "git@github.com:"):
        if sep in name:
            name = name.split(sep, 1)[1]
    return name.replace("/", "_").replace("\\", "_")


def ensure_clone(url: str) -> Path:
    """Clone url into repos/<slug>/ if not already present.  Return the path."""
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


@dataclass
class CallSite:
    line: int
    text: str           # ast.unparse(call.func) — for pattern matching
    root: Optional[str] # leftmost Name in the func expression — for resolution


@dataclass
class FunctionInfo:
    qname: str          # 'chatchat.server.llm.wrapper' or '...AgentClient.chat'
    file_path: str      # repo-relative path for display
    line: int
    calls: list[CallSite] = field(default_factory=list)
    is_method: bool = False
    enclosing_class: Optional[str] = None   # e.g. 'chatchat.server.llm.AgentClient'


@dataclass
class FileContext:
    rel_path: str
    current_pkg: str        # e.g. 'chatchat.server'
    current_module: str     # e.g. 'chatchat.server.llm'
    name_map: dict[str, str]
    imported_frameworks: set[str]


def resolve_import(level: int, module: Optional[str], current_pkg: str) -> str:
    """Resolve a relative-or-absolute ImportFrom against the file's package."""
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
    """Leftmost ast.Name in an attribute/subscript/call chain."""
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
    """Map top-level names used in this file to fully-qualified names."""
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
    """Return (current_pkg, current_module) from a file's path relative to repo_root."""
    rel = file_path.resolve().relative_to(repo_root.resolve())
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    current_module = ".".join(parts)
    current_pkg = ".".join(parts[:-1]) if len(parts) > 1 else ""
    return current_pkg, current_module


def _make_function_info(
    node: ast.AST, qname: str, rel_path: str,
    is_method: bool = False, enclosing_class: Optional[str] = None,
) -> FunctionInfo:
    """Build a FunctionInfo from a FunctionDef / AsyncFunctionDef node."""
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
    return fi


def collect_functions(tree: ast.Module, current_module: str, rel_path: str) -> list[FunctionInfo]:
    """Walk top-level functions AND methods inside top-level classes."""
    out: list[FunctionInfo] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qname = f"{current_module}.{node.name}"
            out.append(_make_function_info(node, qname, rel_path))
        elif isinstance(node, ast.ClassDef):
            cls_qname = f"{current_module}.{node.name}"
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(_make_function_info(
                        sub, f"{cls_qname}.{sub.name}", rel_path,
                        is_method=True, enclosing_class=cls_qname,
                    ))
    return out


def index_repo(
    target: Path, repo_root: Path
) -> tuple[dict[str, FunctionInfo], dict[str, FileContext]]:
    """Walk the target dir; return all top-level functions + per-file context."""
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
        for fi in collect_functions(tree, current_module, rel):
            functions[fi.qname] = fi
    return functions, contexts


def seed_invokers(
    functions: dict[str, FunctionInfo],
    contexts: dict[str, FileContext],
) -> dict[str, str]:
    """Return {qname: reason} for functions whose body matches a FRAMEWORK_CALLS pattern."""
    matchers_by_fw: dict[str, list[tuple[str, callable]]] = {
        fw: [(pat, matcher(pat)) for pat in pats]
        for fw, pats in FRAMEWORK_CALLS.items()
    }
    invokers: dict[str, str] = {}

    funcs_by_module: dict[str, list[FunctionInfo]] = defaultdict(list)
    for qname, fi in functions.items():
        # See iterate_once for why methods need a different module-key derivation.
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
    """Try to resolve a call to a fully-qualified target name.

    Handled shapes:
      - self.X(...) / cls.X(...) inside a method  -> enclosing_class.X
      - ClassName.X(...) when ClassName is in the file's name map -> qname.X
      - bare_name(...) -> file name_map lookup
    Anything else (longer chains, instance-attribute calls) returns None;
    those calls can still be picked up by the seed pass via pattern matching.
    """
    if call.root is None:
        return None
    parts = call.text.split(".")
    if fi.is_method and call.root in ("self", "cls"):
        if len(parts) == 2:
            return f"{fi.enclosing_class}.{parts[1]}"
        return None
    if len(parts) == 2:
        cls_qname = ctx.name_map.get(parts[0])
        if cls_qname:
            return f"{cls_qname}.{parts[1]}"
        return None
    if len(parts) == 1:
        return ctx.name_map.get(call.root)
    return None


def iterate_once(
    functions: dict[str, FunctionInfo],
    contexts: dict[str, FileContext],
    invokers: dict[str, str],
) -> int:
    """One fixed-point pass.  Adds any function whose call resolves to a known
    invoker.  Returns the number of new invokers discovered."""
    new_invokers: dict[str, str] = {}
    for qname, fi in functions.items():
        if qname in invokers:
            continue
        # FileContext is keyed by the file's module qualified name.  For a
        # method 'pkg.mod.Cls.method' the module is 'pkg.mod' (strip class +
        # method); for a top-level function 'pkg.mod.fn' the module is 'pkg.mod'
        # (strip function name).
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


def report(
    invokers: dict[str, str],
    functions: dict[str, FunctionInfo],
    contexts: dict[str, FileContext],
) -> None:
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
