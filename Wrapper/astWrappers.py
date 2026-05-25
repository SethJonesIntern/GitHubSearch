"""AST-based scanning of a local repo for target imports and call expressions."""
import ast
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "dist", "build"}


@dataclass
class Match:
    file: str       # path relative to the repo root
    line: int       # 0 for file-level imports, else the call site line
    kind: str       # "import", "call", or "class"
    text: str       # matched package name, unparsed call expression, or class base
    framework: str  # which framework's import triggered this match


def file_imports(tree: ast.Module) -> set[str]:
    """Top-level package names imported anywhere in the module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def scan_source(
    source: str,
    framework_calls: dict[str, list[str]],
) -> list[tuple[int, str, str, str]]:
    """Scan one source string.

    Returns (line, kind, text, framework) tuples.
    Imports from framework_calls.keys() trigger a targeted call search using
    only the patterns listed for that framework.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    hits: list[tuple[int, str, str, str]] = []

    # Detect which frameworks are imported in this file
    imported = file_imports(tree) & framework_calls.keys()

    for name in imported:
        hits.append((0, "import", name, name))

    if not imported:
        return hits

    # Build ordered list of (pattern, framework) pairs for the detected frameworks
    active_patterns: list[tuple[str, str]] = [
        (pat, fw)
        for fw in imported
        for pat in framework_calls[fw]
    ]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                text = ast.unparse(node.func)
            except Exception:
                continue
            for pat, fw in active_patterns:
                if pat in text:
                    hits.append((node.lineno, "call", text, fw))
                    break  # report once per call site
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                try:
                    text = ast.unparse(base)
                except Exception:
                    continue
                for pat, fw in active_patterns:
                    if pat in text:
                        hits.append((node.lineno, "class", text, fw))
                        break

    return hits


def scan_repo(
    repo: Path,
    framework_calls: dict[str, list[str]],
) -> list[Match]:
    """Walk every .py file in `repo` and return all matches."""
    repo = Path(repo)
    matches: list[Match] = []
    for path in repo.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(repo).as_posix()
        for line, kind, text, framework in scan_source(source, framework_calls):
            matches.append(Match(file=rel, line=line, kind=kind, text=text, framework=framework))
    return matches
