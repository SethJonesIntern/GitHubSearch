# Static result-to-assertion candidates

Population: subset; 27 unique saved LLM baseline methods.

- with_mapped_test: 3/27 methods (11.11%).
- with_exact_mapped_test: 0/27 methods (0.00%).
- with_file_association: 9/27 methods (33.33%).
- with_import_file_association: 9/27 methods (33.33%).
- with_test_candidate: 3/27 methods (11.11%).
- with_production_candidate: 0/27 methods (0.00%).
- with_shared_slice_candidate: 0/27 methods (0.00%).

Candidate rows: 6; distinct test links: 4.

## Assessment limits

- Static candidates do not establish execution or correctness.
- Test calls match final names only; dispatch, mocks and execution remain unresolved.
- Only supported straight-line local return-value expressions and value assertions are assessed.
- Saved Joern membership is corroboration only, not verified dependence; scaffolding is not evidence.
- Production and test chains are separate; no cross-method value flow is established.
- Absence of detection is not absence of checking; missing/invalid source remains unassessed.

## Deterministic positive sample

- test candidate: Aggregate-Intellect/sherpa src/tests/unit_tests/actions/test_planning.py call 23 -> check 27; not_assessed; `assert len(plan.steps) > 0`
- test candidate: Aggregate-Intellect/sherpa src/tests/unit_tests/test_utils.py call 866 -> check 872; not_assessed; `assert result == "summary of the page"`
- test candidate: Aggregate-Intellect/sherpa src/tests/unit_tests/test_utils.py call 879 -> check 885; not_assessed; `assert isinstance(result, str)`
- test candidate: Aggregate-Intellect/sherpa src/tests/unit_tests/test_utils.py call 879 -> check 886; not_assessed; `assert result == "chat summary of the page"`
- test candidate: Aggregate-Intellect/sherpa src/tests/unit_tests/test_utils.py call 891 -> check 899; not_assessed; `assert isinstance(result, str)`

## Missingness and disagreement examples

- unsupported: Aggregate-Intellect/sherpa demo/blueprint_parser_creator/blueprint_utils.py:865: Builtin names may be shadowed; calls are not resolved.
- unsupported: Aggregate-Intellect/sherpa src/tests/unit_tests/actions/test_planning.py:8: Builtin names may be shadowed; calls are not resolved.
- unsupported: Aggregate-Intellect/sherpa src/tests/unit_tests/test_utils.py:875: Builtin names may be shadowed; calls are not resolved.
- unsupported: Aggregate-Intellect/sherpa src/tests/unit_tests/test_utils.py:889: Builtin names may be shadowed; calls are not resolved.
- join_disagreement: Aggregate-Intellect/sherpa src/sherpa_ai/actions/utils/refinement.py:49: No matching baseline definition
- join_disagreement: Aggregate-Intellect/sherpa src/sherpa_ai/policies/react_sm_policy.py:69: No matching baseline definition
- join_disagreement: Aggregate-Intellect/sherpa src/sherpa_ai/tools.py:315: No matching baseline definition
- join_disagreement: Aggregate-Intellect/sherpa src/sherpa_ai/utils.py:1016: No matching baseline definition
- join_disagreement: Aggregate-Intellect/sherpa src/sherpa_ai/utils.py:1056: No matching baseline definition
- join_disagreement: Aggregate-Intellect/sherpa src/sherpa_ai/utils.py:1167: No matching baseline definition

## Provenance

manifest.json records Python/revision/options, input hashes before/after, inspected original source hashes, and all output hashes except its own. Sources use their declared Python encoding. Source caching is bounded to 16 entries and 8 MiB of text; saved tables are streamed. Lists in CSV are JSON arrays. slice_ids lists every relevant saved call-hit slice; shared_slice_ids lists only those containing both original call and check lines. slice_variables and slice_parse_flags correspond positionally to slice_ids.
