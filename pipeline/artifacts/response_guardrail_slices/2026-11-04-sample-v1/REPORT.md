# Response-check candidates

Source-check candidates: 7/42 (16.67%) static invocation sites. Response-slice-supported candidates: 6/42 (14.29%).

These are static invocation-site outputs, not the number of runtime responses, validated guardrail prevalence, or verified dependence paths.

| Saved code kind | Invocation sites | Source-check sites | Response-slice-supported sites | Validator-route sites | Runtime-intervention candidate sites |
|---|---:|---:|---:|---:|---:|
| production | 35 | 6/35 (17.14%) | 5/35 (14.29%) | 0/35 (0.00%) | 6/35 (17.14%) |
| test | 5 | 1/5 (20.00%) | 1/5 (20.00%) | 0/5 (0.00%) | 0/5 (0.00%) |
| test_support | 2 | 0/2 (0.00%) | 0/2 (0.00%) | 0/2 (0.00%) | 0/2 (0.00%) |

Saved hit slices are available for 41/42 (97.62%); 145 distinct relevant slices. Original-source assessable outputs: 42/42 (100.00%).
Source-only check sites (no strong slice-supported check): 1.

Mechanism counts (check rows): {"deterministic_assert": 5, "response_predicate": 2}.
Intervention counts (check rows): {"assertion_failure": 5, "raise": 1, "replacement_return": 1}.
Production intervention counts (unique call IDs per kind): {"assertion_failure": 4, "raise": 1, "replacement_return": 1, "retry_continue": 0}.

Validation routes measure source-resolved data routing, separately from observed failure handling. Early replacement-return candidates are weaker than assertion/raise candidates; retry-continue does not prove another invocation executes. Test assertions and test support are separate from production runtime candidates.

Canonical production labels with obvious test context (sensitivity): {"calls": 0, "runtime_intervention_candidates_without_obvious_test_context": 6, "with_response_slice_check": 0, "with_source_check": 0}.
Canonical code_kind is unchanged. Unconventional tests.py files can be misclassified; these flags use the innermost function/class source context.

Saved enclosing-method counts (exclude blank names, no new method-eligibility rule): {"production": {"calls_without_enclosing_name": 0, "with_response_slice_check": 5, "with_saved_enclosing_name": 31, "with_source_check": 6}, "test": {"calls_without_enclosing_name": 0, "with_response_slice_check": 1, "with_saved_enclosing_name": 5, "with_source_check": 1}, "test_support": {"calls_without_enclosing_name": 0, "with_response_slice_check": 0, "with_saved_enclosing_name": 2, "with_source_check": 0}}.

## Unresolved evidence

Source statuses: {"ok": 42}.
Joern statuses: {"no_joint_slice": 1, "same_response_slice": 6}.
Nonqualifying parser/evaluation/unresolved routes: 0 rows.

- Bounded local source flow is incomplete; negative detections do not imply absent guardrails.
- Builtin and receiver dispatch can remain unresolved; candidates require manual precision review.
- Joint saved slice lines are not verified DDG/CDG paths; full per-edge paths were not persisted.
- Original code_kind strata are retained even where original source has obvious test context.
- Static invocation-site outputs, not runtime responses or validated guardrail prevalence.
- Validator routes do not establish observed failure handling or protected/accepted output.
- builtin_dispatch_unresolved
- caller_flow_unassessed
- container_response_flow_unassessed
- nested_expression_flow_bounded
- nonlocal_binding_unassessed
- original_source_test_context:test_named_function:test_callback_records_on_async_invoke_through_binding
- original_source_test_context:test_named_function:test_callback_records_on_sync_invoke
- original_source_test_context:test_named_function:test_gpt52_interleaved_reasoning_during_tool_calls
- original_source_test_context:test_named_function:test_kimi_api_connection
- original_source_test_context:test_named_function:test_kimi_reasoning_enabled
- receiver_dispatch_unresolved
- runnable_chain_flow_unassessed
- unknown_transform_or_callback:AIYapping
- unknown_transform_or_callback:ToolMessage
- unknown_transform_or_callback:_format_results
- unknown_transform_or_callback:append
- unknown_transform_or_callback:parse
- unknown_transform_or_callback:select_pairs_by_indices
- unknown_transform_or_callback:wait_for
- unknown_transform_or_callback:zip
- unsupported_comprehension_scope
- unsupported_expression:FormattedValue
- unsupported_expression:IfExp
- unsupported_merge
- unsupported_statement:Match
- unsupported_with_exit_flow

## Deterministic review sample

Manual review must establish precision and the property being checked before any guardrail-prevalence claim.

- 0ca_BoxPwnr/tests/core/test_llm_manager_kimi.py::83::15 → line 85: assertion / same_response_slice; assert isinstance(response.content, list), (         f"Expected list content (thinking + text blocks) but got {type(response.content)}"     )
- Certora_AIComposer/composer/prover/analysis.py::35::16 → line 36: early_exit_candidate / same_response_slice; if not isinstance(res, AIMessage):         return None
- Certora_AIComposer/composer/spec/refinement.py::97::20 → line 98: assertion / same_response_slice; assert isinstance(res, AIMessage)
- Certora_AIComposer/composer/spec/source/report/grouping.py::92::19 → line 93: assertion / same_response_slice; assert isinstance(result, GroupingResult)
- Certora_AIComposer/composer/workflow/meta.py::31::16 → line 32: assertion / same_response_slice; assert isinstance(res, ResumeCommentary)
- HKUDS_ViMax/agents/camera_image_generator.py::147::45 → line 149: rejection_branch / no_joint_slice; if len(parent_items) != len(cameras):             raise ValueError(f"Camera tree response length mismatch: expected {len(cameras)}, got {len(parent_items)}")
- HKUDS_ViMax/agents/event_extractor.py::149::23 → line 151: assertion / same_response_slice; assert event.index == len(extracted_events), f"Extracted event index {event.index} does not match the expected index {len(extracted_events)}"
