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
    # DONE — evaluate() and assert_test() are the two run entry points; .measure/
    # .a_measure are the metric methods they invoke (a_measure = async, the default
    # when assert_test runs with run_async=True). Pilot-confirmed (.a_measure real).
    # `deepeval test run` CLI is AST-invisible but those files still contain assert_test.
    "deepeval": [
        "assert_test",
        "evaluate",
        ".measure",
        ".a_measure",
    ],

    # ── RAGAs ───────────────────────────────────────────────────────────────────
    # DONE — evaluate() is the top-level run; .score/.ascore are the metric methods
    # (heavily used in pilot: .ascore 192, .score 71). Added the v0.2+ single/multi_turn
    # variants — our .score/.ascore patterns don't match them (different token), and they
    # ARE in the pilot (single_turn_ascore 22, single_turn_score 6, multi_turn_ascore 4).
    # ".evaluate" kept: 0 distinct hits (bare `evaluate` is a superset) but harmless — a
    # safety net if the list order ever changes. (NB: pilot counts include ragas's own repo.)
    "ragas": [
        "evaluate",
        ".evaluate",
        ".score",
        ".ascore",
        "single_turn_score",
        "single_turn_ascore",
        "multi_turn_score",
        "multi_turn_ascore",
    ],

    # ── Giskard ─────────────────────────────────────────────────────────────────
    # DONE (docs-confirmed, UNSAMPLED in pilot — 0 giskard files, verify at full-run
    # scale). Eval-RUN surfaces only; construction/test-gen deliberately excluded.
    #   v2:  giskard.scan(model, dataset)            -> "scan"
    #   v3:  await vulnerability_scan(target=, ...)  -> "vulnerability_scan"
    #        await quality_scan(target=, ...)        -> "quality_scan"
    #        generate_suite(...) + suite.run(...)    -> ".run" (the run step)
    #   RAG: giskard.rag.evaluate(...)               -> "evaluate"
    # Excluded: Model/Dataset/KnowledgeBase/Scenario (construction, not a run) and
    # generate_suite/generate_testset (test creation — the run is caught by .run/scan).
    # .scan/.evaluate methods dropped (the real APIs are the module functions above).
    # .run is collision-prone (real receiver = suite); relies on the FP filter
    # (drops asyncio.run/subprocess.run) + import-scoping.
    "giskard": [
        "scan",
        "vulnerability_scan",
        "quality_scan",
        "evaluate",
        ".run",
    ],

    # ── Opik (Comet) ────────────────────────────────────────────────────────────
    # DONE — three run surfaces: evaluate() (task eval, opik 131 pilot), evaluate_prompt()
    # (prompt eval, 136 pilot), evaluate_experiment() (updates an experiment, 56 pilot).
    # .score/.ascore are the metric methods (.score 260 pilot; .ascore docs-recommended
    # for async, unsampled). ".evaluate" kept (0 distinct hits — bare `evaluate` superset —
    # but harmless). (NB: pilot counts include opik's own repo source.)
    "opik": [
        "evaluate",
        "evaluate_prompt",
        "evaluate_experiment",
        ".evaluate",
        ".score",
        ".ascore",
    ],

    # ── Arize Phoenix ─────────────────────────────────────────────────────────────
    # DONE — Phoenix has two structurally distinct run surfaces we were missing:
    #   experiments: run_experiment(...) / async_run_experiment (146 pilot) — dataset-level
    #   evals:       evaluate_dataframe / async_evaluate_dataframe (53 pilot) — dataframe
    # plus llm_generate (8). run_evals (27) kept — still used. llm_classify (0, now legacy
    # phoenix.evals.legacy) kept as harmless. .evaluate kept — the dominant pilot surface
    # (runner.evaluate 58, evaluator.evaluate 45); TODO verify test_func.evaluate (14) isn't
    # a test-harness collision. (NB: pilot counts include Phoenix's own repo source.)
    "phoenix": [
        "run_experiment",
        "async_run_experiment",
        "evaluate_dataframe",
        "async_evaluate_dataframe",
        "run_evals",
        "llm_classify",
        "llm_generate",
        ".evaluate",
    ],

    # ── promptfoo ─────────────────────────────────────────────────────────────────
    # Primarily a JS/CLI tool; rarely imported in Python. Left empty until a real
    # Python invocation surface is confirmed.
    # "promptfoo": [],
}
