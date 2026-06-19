"""Per-eval-framework call patterns — the semantic-evaluation analogue of
`FrameworkDict.FRAMEWORK_CALLS`.

Keyed by the framework's top-level **import name** (what appears in `import X` /
`from X import ...`), mapping to the call patterns that constitute *running an
evaluation*. Fed to the same seed/matcher machinery the LLM-call extraction uses
(Stage 5), so the batch driver can evaluate these against each repo's parsed AST
in the same pass.

⚠️ SEEDS ARE PROVISIONAL. These method/function names are best-effort and need
confirmation against real usage (READMEs/examples), the way invoker seeds were
validated in Applications/verify_invokers.py. Keep the framework set in sync with
the dependency detector (SemanticEvaluators/find_semantic_eval_tests.py) and the
LLM-module list memory.
"""
from __future__ import annotations

EVAL_CALLS: dict[str, list[str]] = {
    # ── DeepEval ────────────────────────────────────────────────────────────────
    "deepeval": [
        "assert_test",
        "evaluate",
        ".measure",
        ".a_measure",
    ],

    # ── RAGAs ───────────────────────────────────────────────────────────────────
    "ragas": [
        "evaluate",
        ".evaluate",
        ".score",
        ".ascore",
    ],

    # ── Giskard ─────────────────────────────────────────────────────────────────
    "giskard": [
        "scan",
        ".scan",
        ".evaluate",
        ".run",
    ],

    # ── Opik (Comet) ────────────────────────────────────────────────────────────
    "opik": [
        "evaluate",
        ".evaluate",
        ".score",
    ],

    # ── Arize Phoenix ─────────────────────────────────────────────────────────────
    "phoenix": [
        "run_evals",
        "llm_classify",
        ".evaluate",
    ],

    # ── promptfoo ─────────────────────────────────────────────────────────────────
    # Primarily a JS/CLI tool; rarely imported in Python. Left empty until a real
    # Python invocation surface is confirmed.
    # "promptfoo": [],
}
