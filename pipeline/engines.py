"""Bridge to the existing AST/call-graph engines that live in Wrapper/.

The Wrapper modules (`transitive_invokers`, `call_metadata`, `FrameworkDict`)
import each other by bare module name, so they expect their own directory on
`sys.path`. Rather than convert them into a package, we add `Wrapper/` to the
path here and re-export the symbols the pipeline drivers need. This is the one
place that knows where the engines live — see the "Imports across folders" risk
in PIPELINE.md.
"""
from __future__ import annotations

import sys

from pipeline.paths import REPO_ROOT

_WRAPPER_DIR = REPO_ROOT / "Wrapper"
if str(_WRAPPER_DIR) not in sys.path:
    sys.path.insert(0, str(_WRAPPER_DIR))

# Re-exported engine API (imported after the path shim above).
from transitive_invokers import (  # noqa: E402
    build_call_graph,
    ensure_clone,
    index_repo,
    seed_invokers,
    transitive_closure,
    shallow_clone,
    repo_slug,
    _on_rm_error,
)
from FrameworkDict import FRAMEWORK_CALLS, SCOPED_FRAMEWORK_CALLS, IN_SCOPE_FRAMEWORKS  # noqa: E402
from call_metadata import AstIndex, collect_rows  # noqa: E402
from call_metadata import FIELDS as CALL_METADATA_FIELDS  # noqa: E402

from pathlib import Path as _Path


def is_test_file(rel_path: str) -> bool:
    """A file is a pytest test if its basename matches discovery defaults.
    Inlined from Wrapper/find_llm_tests.py — kept here so the driver doesn't
    depend on that bare module name (there are two `find_llm_tests.py` in the
    repo, which collide on sys.path)."""
    name = _Path(rel_path).name
    return name.startswith("test_") or name.endswith("_test.py")


def is_test_function(qname: str) -> bool:
    """True if the qname's last segment starts with 'test_' (plain test or method)."""
    return qname.rsplit(".", 1)[-1].startswith("test_")


__all__ = [
    "build_call_graph",
    "ensure_clone",
    "index_repo",
    "seed_invokers",
    "transitive_closure",
    "shallow_clone",
    "repo_slug",
    "_on_rm_error",
    "FRAMEWORK_CALLS",
    "SCOPED_FRAMEWORK_CALLS",
    "IN_SCOPE_FRAMEWORKS",
    "AstIndex",
    "collect_rows",
    "CALL_METADATA_FIELDS",
    "is_test_file",
    "is_test_function",
]
