# Saved-edge direct-invoker result-check candidates

Detected 367/18961 canonical direct invokers (1.936%) with a local returned-result pass/fail candidate.
877/18961 invokers have a selected saved direct-target test edge; 4187 unique edges and 2445 candidate checks.

This is an independent firstedge candidate frame. No PyMethods2Test was used.
Only candidate invoker definition files and candidate test files were inspected. Other invokers remain in the denominator with source_status=not_assessed. A usable saved target edge, readable/parseable source, exact saved definitions and one syntactic target call are required for source assessment.

## Strata

| Stratum | Denominator | Candidate invokers |
|---|---:|---:|
| production | 8840 | 344 |
| test | 9700 | 0 |
| test_support | 421 | 23 |

## Unresolved and missingness

{"ambiguous_call": 191, "deferred_wrapper_edge": 124328, "local_limitation": 15693, "no_matching_call": 203}

Direct-test rows without a target edge: 11614.
Source-assessed edges: 3793/4187.

## Deterministic examples

Positive: 0ca/BoxPwnr 0ca_BoxPwnr/tests/core/test_deadline.py:149 → 0ca_BoxPwnr.src.boxpwnr.core.llm_manager.LLMManager._invoke_with_retry; 0ca/BoxPwnr 0ca_BoxPwnr/tests/tools/test_tools.py:167 → 0ca_BoxPwnr.src.boxpwnr.tools.tools.WebSearchTool._run; 0ca/BoxPwnr 0ca_BoxPwnr/tests/tools/test_tools.py:167 → 0ca_BoxPwnr.src.boxpwnr.tools.tools.WebSearchTool._run
No detected candidate: 0ca/BoxPwnr 0ca_BoxPwnr/tests/core/test_deadline.py:106 → 0ca_BoxPwnr.src.boxpwnr.core.llm_manager.LLMManager._invoke_with_retry; 0ca/BoxPwnr 0ca_BoxPwnr/tests/core/test_deadline.py:167 → 0ca_BoxPwnr.src.boxpwnr.core.llm_manager.LLMManager._invoke_with_retry; 0ca/BoxPwnr 0ca_BoxPwnr/tests/tools/test_tools.py:196 → 0ca_BoxPwnr.src.boxpwnr.tools.tools.WebSearchTool._run

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
