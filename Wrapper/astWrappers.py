"""AST-based scanning of a local repo for target imports and call expressions."""
import ast
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "dist", "build"}


@dataclass
class Match:
    file: str   # path relative to the repo root
    line: int   # 0 for file-level imports, else the call site line
    kind: str   # "import" or "call"
    text: str   # matched package name or unparsed call expression


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
    target_imports: set[str],
    target_calls: list[str],
) -> list[tuple[int, str, str]]:
    """Scan one source string. Returns (line, kind, text) tuples."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    hits: list[tuple[int, str, str]] = []

    for name in file_imports(tree) & target_imports:
        hits.append((0, "import", name))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                text = ast.unparse(node.func)
            except Exception:
                continue
            if any(pat in text for pat in target_calls):
                hits.append((node.lineno, "call", text))

    return hits


def scan_repo(
    repo: Path,
    target_imports: set[str],
    target_calls: list[str],
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
        for line, kind, text in scan_source(source, target_imports, target_calls):
            matches.append(Match(file=rel, line=line, kind=kind, text=text))
    return matches
