"""AST utilities used by the transitive-invoker scanner."""
import ast
import re

SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "dist", "build"}


def matcher(pat: str):
    """Return a callable(text) -> bool for matching this pattern.

    Method-style patterns starting with "." use substring matching so that
    ".invoke" catches "chain.invoke", "self.runnable.invoke", etc.
    Identifier patterns use word-boundary matching so that "BaseTool" matches
    "BaseTool(...)" but not "BaseToolOutput" or "_parse_tool".
    """
    if pat.startswith("."):
        return lambda text: pat in text
    rx = re.compile(rf"\b{re.escape(pat)}\b")
    return lambda text: rx.search(text) is not None


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
