"""Small AST helpers used by the transitive-invoker scanner.

Kept in its own module so the main scanner can stay focused on the analysis
logic; this file is the boring pattern-matching and import-walking plumbing.
"""
import ast
import re

# Directories the scanner skips entirely when walking a repo.  These either
# contain machine-generated code we don't care about (build/, dist/) or
# vendored dependencies whose internals we don't want to count as part of
# the target repo (venv/, node_modules/).
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "dist", "build"}


def matcher(pat: str):
    """Build the right kind of text-matcher for a pattern from FRAMEWORK_CALLS.

    Two pattern shapes need different matching strategies:

    Method-style patterns starting with "." use plain substring matching so
    that ".invoke" catches the .invoke in any "something.invoke" expression
    — `chain.invoke`, `self.runnable.invoke`, `client.batch.invoke`, etc.

    Identifier patterns use word-boundary regex matching.  This is the fix
    for collisions like "BaseTool" matching inside "BaseToolOutput" or
    "_parse_tool" — those used to cause false positives because the original
    matcher was substring-only.  The boundary anchor (\\b) keeps the match
    aligned to identifier edges.
    """
    if pat.startswith("."):
        return lambda text: pat in text
    rx = re.compile(rf"\b{re.escape(pat)}\b")
    return lambda text: rx.search(text) is not None


def file_imports(tree: ast.Module) -> set[str]:
    """The top-level package names imported anywhere in this file.

    Crucially, this uses ast.walk and so includes imports made INSIDE
    function bodies — not just at module level.  That matters because a file
    that lazy-imports langchain (a common pattern for avoiding circular
    dependencies) should still count as a langchain-using file when we
    decide which framework's patterns to activate against it.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names
