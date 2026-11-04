# Independent direct-invoker result-check measurement

Measured 2026-11-04. Use 2026-11-04-independent-v2 as the refined result. The independent-v1 and independent-sherpa-v1 outputs are preserved preliminary runs; v1's 367-method figure is superseded by 359 after receiver-owner conflict checking.

## Findings

| Measure | Unique direct invokers | Share of all 18,961 |
|---|---:|---:|
| Total canonical observed direct-invoker inventory | 18,961 | 100% |
| Selected saved direct test-call relationship | 877 | 4.63% |
| Returned value reaches a detected test assertion candidate | 359 | 1.89% |

The 359 qualifying methods occur in 156 repositories. They have 1,305 qualifying method/test edges and 2,309 check rows: 2,203 Python assertions and 106 unittest assertions. Multiple tests/checks do not multiply the method numerator. No controlled-failure candidates were detected in this pass.

The call relationship supplies an upper candidate pool within this recorded first-edge analysis; it is not a ceiling on real testing. The result-check count is a stricter detected subset (359/877, 40.94%). It is not a validated lower bound on actual testing either, because saved target identification and runtime dispatch can be wrong.

## Independent inputs and preserved denominator

No PyMethods2Test tables, mappings, eligibility, or production-method baseline were used. The five saved input files are llm_invokers_all.csv, llm_tests_all.csv, llm_calls_all.csv, application_audit.csv, and .batch_progress.json. Existing scope definitions supply the observed population, valid framework patterns and raw-SDK exclusions. All 18,961 direct invokers remain in the output, including zero detections and unresolved cases.

The total includes test/support methods. Heuristic strata are production 8,840 (337 qualifying), test 9,700 (0 qualifying), and test_support 421 (22 qualifying). The known naming-rule issue with unconventional tests.py files means these strata are provisional. The primary all-invoker denominator is unaffected by that classification issue. A test invoking a framework directly is not counted as a test mapping to itself.

## Evidence and unresolved cases

Saved Stage-5 first-BFS edges identify 4,187 unique direct-target test relationships. Source identities and a unique syntactic target call were assessed successfully for 3,688; 191 have ambiguous source calls, 203 have no matching syntactic call, and 105 have explicit receiver-owner disagreement. An additional 124,328 saved test edges point to wrappers or other non-selected targets and are deferred. The 11,614 observed raw direct-test rows have framework-match reasons rather than a saved target edge; this is a frame limitation, not the historical 9,700 direct-test count.

A concrete rejected relationship was BoxPwnr's test_read_missing_file: the saved graph pointed at WebSearchTool._run, while original source constructs ReadFileTool and calls its _run. The refined analyzer retains that edge as receiver_owner_mismatch with zero qualifying checks. Different class names may also reflect inheritance, so disagreement is unresolved rather than an assertion that no relationship exists. Matching class names do not verify dispatch.

A positive inspected example is BoxPwnr's test_retry_works_normally_without_deadline: result = manager._invoke_with_retry(mock_model, []) followed by assert result == "response". This supports local dependence on the method's return value, while the mock explicitly prevents interpreting it as evidence of real-model execution.

## Limits and verification

This pass uses saved pyan first-edge relationships plus original-source static local return-to-check analysis; it is not a new Joern run or an independent full call-graph reconstruction. Only one outgoing target path per transitive function was retained upstream. Multi-hop wrapper flow, unrecorded edges, unsupported control flow, arbitrary validators, fixture/dynamic dispatch, state and exception checks remain unresolved. A method return checked by a test does not establish that the underlying LLM response reaches that return on the tested path, or that the check evaluates semantic correctness.

Analysis revision: a64f5eb. All five saved input before/after hash pairs match. All 1,611 inspected original source hashes and the six generated output hashes were independently rechecked successfully. The focused independent suite has 10 tests; no application tests were executed. Existing research files, prior reports and raw datasets were not overwritten or regenerated.
