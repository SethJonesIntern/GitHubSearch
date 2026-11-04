# Static result-to-assertion candidates

Population: full saved baseline; 7642 unique saved LLM baseline methods.

- with_mapped_test: 220/7642 methods (2.88%).
- with_exact_mapped_test: 59/7642 methods (0.77%).
- with_file_association: 1225/7642 methods (16.03%).
- with_import_file_association: 1026/7642 methods (13.43%).
- with_test_candidate: 105/7642 methods (1.37%).
- with_production_candidate: 38/7642 methods (0.50%).
- with_shared_slice_candidate: 34/7642 methods (0.44%).

Candidate rows: 431; distinct test links: 461.

## Assessment limits

- Static candidates do not establish execution or correctness.
- Test calls match final names only; dispatch, mocks and execution remain unresolved.
- Only supported straight-line local return-value expressions and value assertions are assessed.
- Saved Joern membership is corroboration only, not verified dependence; scaffolding is not evidence.
- Production and test chains are separate; no cross-method value flow is established.
- Absence of detection is not absence of checking; missing/invalid source remains unassessed.

## Deterministic positive sample

- test candidate: 10xHub/Agentflow tests/store/test_mem0_store_async.py call 117 -> check 118; not_assessed; `assert results and results[0].content.startswith("Alice")`
- test candidate: Aggregate-Intellect/sherpa src/tests/unit_tests/actions/test_planning.py call 23 -> check 27; not_assessed; `assert len(plan.steps) > 0`
- test candidate: Aggregate-Intellect/sherpa src/tests/unit_tests/test_utils.py call 866 -> check 872; not_assessed; `assert result == "summary of the page"`
- test candidate: Aggregate-Intellect/sherpa src/tests/unit_tests/test_utils.py call 879 -> check 885; not_assessed; `assert isinstance(result, str)`
- test candidate: Aggregate-Intellect/sherpa src/tests/unit_tests/test_utils.py call 879 -> check 886; not_assessed; `assert result == "chat summary of the page"`

## Missingness and disagreement examples

- unsupported: microsoft/langchain-for-beginners 01-introduction/solution/model_performance.py:22: Builtin names may be shadowed; calls are not resolved.
- unsupported: XAIHT/Tlamatini Tlamatini/agent/mcp_agent.py:1475: Builtin names may be shadowed; calls are not resolved.
- unsupported: XAIHT/Tlamatini Tlamatini/agent/rag/chains/history_aware.py:384: Builtin names may be shadowed; calls are not resolved.
- unsupported: XAIHT/Tlamatini Tlamatini/agent/rag/chains/unified.py:313: Builtin names may be shadowed; calls are not resolved.
- unsupported: XAIHT/Tlamatini Tlamatini/agent/rag/chains/unified.py:697: Builtin names may be shadowed; calls are not resolved.
- unsupported: XAIHT/Tlamatini Tlamatini/agent/rag/interface.py:668: Builtin names may be shadowed; calls are not resolved.
- unsupported: XAIHT/Tlamatini Tlamatini/agent/tests.py:1074: Builtin names may be shadowed; calls are not resolved.
- unsupported: XAIHT/Tlamatini Tlamatini/agent/tests.py:1158: Builtin names may be shadowed; calls are not resolved.
- unsupported: XAIHT/Tlamatini Tlamatini/agent/tests.py:6843: Builtin names may be shadowed; calls are not resolved.
- unsupported: XAIHT/Tlamatini Tlamatini/agent/tests.py:6860: Builtin names may be shadowed; calls are not resolved.

## Provenance

manifest.json records Python/revision/options, input hashes before/after, inspected original source hashes, and all output hashes except its own. Sources use their declared Python encoding. Source caching is bounded to 16 entries and 8 MiB of text; saved tables are streamed. Lists in CSV are JSON arrays. slice_ids lists every relevant saved call-hit slice; shared_slice_ids lists only those containing both original call and check lines. slice_variables and slice_parse_flags correspond positionally to slice_ids.
