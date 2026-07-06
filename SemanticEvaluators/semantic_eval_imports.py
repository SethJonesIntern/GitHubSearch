"""Detect semantic-evaluator frameworks by what a repo actually *imports*.

This is the AST-based analogue of find_semantic_eval_tests.py, which only looks
at *declared* dependencies in requirements.txt / pyproject.toml. A declared dep
may be unused; an `import deepeval` in the source is stronger evidence the tool
is actually exercised.

It is designed to ride along on the Wrapper scan: transitive_invokers.index_repo
already parses every .py file and records the full set of top-level imports on
each FileContext (the `imports` field). detect_from_contexts() just intersects
that against the eval-tool import names below — no extra clone, no second parse.

Each value is the set of top-level *import package names* for that tool (what you
would `import`), which is not always the pip/distribution name:
  arize-phoenix (pip)  ->  import phoenix
  promptfoo            ->  (Node.js tool; no Python import, so omitted here)
"""
from __future__ import annotations

from typing import Dict, List

# tool label -> top-level import names that indicate it is used.
SEMANTIC_EVAL_IMPORTS: Dict[str, set] = {
    "deepeval": {"deepeval"},
    "ragas": {"ragas"},
    "giskard": {"giskard"},
    "opik": {"opik"},
    "phoenix": {"phoenix"},  # pip name: arize-phoenix
}


def detect_from_contexts(contexts) -> Dict[str, List[str]]:
    """Given index_repo's `contexts` (module -> FileContext), return
    {tool: [repo-relative file, ...]} for every eval tool imported anywhere in
    the repo. Tools with no import hit are omitted.

    Relies on FileContext.imports (the full top-level import set per file).
    """
    hits: Dict[str, set] = {}
    for ctx in contexts.values():
        imported = getattr(ctx, "imports", None) or set()
        for tool, names in SEMANTIC_EVAL_IMPORTS.items():
            if imported & names:
                hits.setdefault(tool, set()).add(ctx.rel_path)
    return {tool: sorted(files) for tool, files in hits.items()}
