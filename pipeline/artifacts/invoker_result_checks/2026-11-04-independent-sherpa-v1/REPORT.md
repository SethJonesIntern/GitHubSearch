# Saved-edge direct-invoker result-check candidates

Detected 4/33 canonical direct invokers (12.121%) with a local returned-result pass/fail candidate.
4/33 invokers have a selected saved direct-target test edge; 10 unique edges and 12 candidate checks.

This is an independent firstedge candidate frame. No PyMethods2Test was used.
Only candidate invoker definition files and candidate test files were inspected. Other invokers remain in the denominator with source_status=not_assessed. A usable saved target edge, readable/parseable source, exact saved definitions and one syntactic target call are required for source assessment.

## Strata

| Stratum | Denominator | Candidate invokers |
|---|---:|---:|
| production | 27 | 4 |
| test | 6 | 0 |
| test_support | 0 | 0 |

## Unresolved and missingness

{"deferred_wrapper_edge": 203, "local_limitation": 31}

Direct-test rows without a target edge: 6.
Source-assessed edges: 10/10.

## Deterministic examples

Positive: Aggregate-Intellect/sherpa Aggregate-Intellect_sherpa/src/tests/unit_tests/actions/test_planning.py:8 → Aggregate-Intellect_sherpa.src.sherpa_ai.actions.planning.TaskPlanning.execute; Aggregate-Intellect/sherpa Aggregate-Intellect_sherpa/src/tests/unit_tests/test_utils.py:861 → Aggregate-Intellect_sherpa.src.sherpa_ai.utils.chunk_and_summarize; Aggregate-Intellect/sherpa Aggregate-Intellect_sherpa/src/tests/unit_tests/test_utils.py:875 → Aggregate-Intellect_sherpa.src.sherpa_ai.utils.chunk_and_summarize
No detected candidate: Aggregate-Intellect/sherpa Aggregate-Intellect_sherpa/src/tests/unit_tests/tools/test_search_tool.py:71 → Aggregate-Intellect_sherpa.src.sherpa_ai.tools.ContextTool._run; Aggregate-Intellect/sherpa Aggregate-Intellect_sherpa/src/tests/unit_tests/tools/test_search_tool.py:108 → Aggregate-Intellect_sherpa.src.sherpa_ai.tools.ContextTool._run; Aggregate-Intellect/sherpa Aggregate-Intellect_sherpa/src/tests/unit_tests/tools/test_search_tool.py:135 → Aggregate-Intellect_sherpa.src.sherpa_ai.tools.ContextTool._run

## Limits

- Saved pyan target plus syntactic call match; dispatch and execution are unverified.
- Independent of PyMethods2Test: no PyMethods2Test data or mappings are used.
- The saved pyan first-BFS-edge forest contains only one chosen edge per transitive function.
- Multi-hop wrappers and unrecorded outgoing edges are unassessed.
- Direct-test rows with only framework-match reasons cannot identify their other target calls.
- No candidate means no detection, rather than no test or no check.
- Static local candidates establish neither actual invocation, mocking status, semantic correctness nor full interprocedural LLM-result flow.
- Compound-flow binding merges, arbitrary validators, dynamic receivers, constructor aliases, fixtures and inheritance are unresolved.
- Strata are heuristic path/name classifications; this is a saved-edge candidate detection rate, not exhaustive prevalence or coverage.
- Controlled failure denotes a result-derived predicate with an immediate failure statement, not proof that the branch executes.
