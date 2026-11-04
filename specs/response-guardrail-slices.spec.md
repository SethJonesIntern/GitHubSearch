# Response guardrail candidates with saved Joern slice corroboration

## Purpose

Measure how many **observed framework invocation sites** have source evidence that their returned response reaches a check, and how many of those checks are corroborated by saved response-variable slices. Use saved data and relevant original source only. This is a candidate measurement, not a count of runtime response executions, validated guardrail prevalence, or full dependence paths. Separate production runtime checks, test assertions and test support. No PyMethods2Test inputs or eligibility.

## Public API

New module `pipeline/response_guardrail_slices.py`; focused tests `pipeline/test_response_guardrail_slices.py`.

```python
from pathlib import Path

def load_calls(artifacts_dir: Path | None = None) -> list[dict]: ...
def analyze_source(source: str, calls: list[dict]) -> list[dict]: ...
def corroborate_slices(checks: list[dict], hits: list[dict],
                       pdg_lines: dict[int, set[int]]) -> list[dict]: ...
def main(argv: list[str] | None = None) -> int: ...
```

No import-time corpus IO. `main` returns 0 on successful analysis including empty subsets; nonzero on fatal IO/schema/snapshot/changed-input errors. Parsing original source failures is a retained unassessed result, not an exception escaping the CLI.

CLI:

```text
py -3.14 -B -m pipeline.response_guardrail_slices --output-dir NEW_PATH [--artifacts-dir PATH] [--repos-dir PATH] [--repo OWNER/NAME ...]
```

Defaults resolve at call time from `pipeline.paths.ARTIFACTS_DIR` and `REPOS_DIR`. Output is required and must not exist. Existing output refusal precedes any loader/corpus read; message contains `exists` and path. Repeatable repo filters label summary `subset=true`; keep full selected-call denominator, including missing source/slices and unsupported output forms. Tests can patch public `load_calls` and named paths/constants with synthetic fixtures; main must not independently import canonical population behind a patched loader.

### Canonical loader and exact inputs

`load_calls` reads `artifacts_dir / "slice_calls_by_call.csv"` (default artifacts root). Lazily import `from Applications import make_figures as MF` and compare the saved full call-ID/repo set with `MF.CALLS`, the authoritative observed framework-call scope. Do not run figure functions, analysis main, graph builders, Joern or artifact generators. A different snapshot is fatal rather than silently changing the denominator. The current snapshot has **23,990 call sites**: production 10,922; test 12,519; test_support 549. Compute values; never hardcode them.

Required saved input subsets (other columns permitted):

| File | Required headers |
|---|---|
| `slice_calls_by_call.csv` | `call_id,repo,repo_slug,rel_path,code_kind,framework,pattern,call_line,enclosing_qname,repo_sliced,in_slice` |
| `slice_call_hits.csv` | `slice_id,call_id,call_repo,repo,rel_path,code_kind,call_line,variable,role,parses` |
| `slice_index.csv` | `slice_id,repo,rel_path,code_kind,method,variable,parses` |
| `slice_pdg_lines.csv.gz` | gzip CSV `slice_id,line` |

`repo` on call rows and `call_repo` on hits is OWNER/NAME; slice/hit `repo` is OWNER_NAME. Saved source path is `repos_dir / repo_slug / rel_path`. Call IDs end in `::line::column`; parse original zero-based AST column, verify line agrees with `call_line`. Returned loader rows include all original fields plus integer `call_line,call_col`; preserve saved `code_kind`. Reject conflicting duplicate identities; identical duplicate call rows/hits/PDG memberships do not multiply counts. UTF-8 with optional BOM supported; required header-only tables are valid. Missing LFS content/dependencies fails actionably, never regenerates or installs.

Synthetic CLI success requires only the four tables above and source files when `load_calls` is patched. The fake loader returns saved call dictionaries plus `call_col`. No hidden canonical imports/real-data reads in that path. A real invocation hashes every actual saved table consumed by canonical scope as well, including raw calls, audit and progress. Do not enumerate artifacts merely to find files to hash.

## Source API and evidence contract

`analyze_source` parses each source file once for all supplied call sites. Input call dictionaries require `call_id,call_line,call_col,pattern,code_kind`; extra identity fields are allowed. Select the exact AST call using original line/column and the pattern's terminal callable name (e.g. `.invoke` -> `invoke`). Chained calls sharing coordinates must not attach the response to `.strip()` or an unrelated call; unresolved collisions return `ambiguous_seed`. Missing exact call returns `seed_missing`. Seed inside a lambda or unsupported class execution context may abstain explicitly. Support module and sync/async function bodies, always the innermost executing scope; nested definitions are not executed as part of their outer body.

Return one dictionary per unique supplied call ID, deterministically ordered:

```python
{
 "call_id": "acme_app/app.py::2::8",
 "status": "ok",  # also parse_error, seed_missing, ambiguous_seed, unsupported_scope
 "output_form": "assigned",  # returned_to_caller, discarded, nested_expression, unassessed
 "response_variables": ["r"],
 "checks": [{"call_id": "acme_app/app.py::2::8", "call_line": 2, "sink_line": 3,
             "check_kind": "assertion", "mechanism": "deterministic_assert",
             "response_variables": ["r"], "trace_lines": [2, 3],
             "sink_source": "assert r is not None",
             "intervention": "assertion_failure", "resolution": "source_local"}],
 "routes": [],
 "limitations": []
}
```

Literal names/enums below are required. Each call/check retains its own call ID, even when one assertion checks multiple results. `routes` contains analogous sink rows with `route_kind` instead of `check_kind`, for observations that **do not qualify** as checks. Unsupported source constructs produce limitations/diagnostics, never a conclusion that no guardrail exists.

### Bounded source behavior

Track local response origins through simple/annotated assignments, aliases, attributes/subscripts, comparisons and supported pure-looking transforms (`len`, `str`, `bool`, `isinstance`, string lower/upper/strip and startswith/endswith). Mark builtin/receiver dispatch unresolved where applicable. Reassignment kills prior response association; unsupported assignment expressions, patterns and nested definition bindings cannot leave stale tags. Terminal return/raise stops analysis along that block. Receiver and prompt/argument variables are not response origins just because they share a slice.

Analyze statements within branch/loop/try/with blocks so a seed and later check in the same block can be found. Never merge a response origin across uncertain branch/loop/exception paths as a definite association: retain explicit `unsupported_merge`/unassessed evidence when writes may change a name. This is a bounded local source analysis, not symbolic execution, heap analysis or a general Python resolver.

Qualifying check families:

1. Python `assert` condition; ignore its message. A small documented `self.assertEqual/assertTrue/assertIsNotNone/assertIn/assertIsInstance/assertGreaterEqual` value-assertion subset is sufficient. Only asserted value operands count, not message arguments. `check_kind=assertion`, mechanism `deterministic_assert`, intervention `assertion_failure`. In production these are runtime assertion candidates; in test code they are test assertion candidates, not runtime application guardrails.
2. Explicitly source-resolved schema validation: `jsonschema.validate(instance, schema)` or keyword `instance=`, and Pydantic `KnownBaseModelSubclass.model_validate(response)` / known `TypeAdapter(...).validate_python(response)`. Resolve import aliases and local BaseModel subclass/type-adapter construction conservatively. Only the API's **instance/data port** counts; a response used as schema/config does not. `check_kind=schema_validation`, mechanism `schema_validator`, intervention `validator_exception_contract`, resolution `source_import`. A validation route is measured separately from observed handling of failure; do not claim the application protects/accepts the output merely because a validator was called.
3. A supported response-derived predicate controlling an immediate `raise` is `check_kind=rejection_branch`, mechanism `response_predicate`, intervention `raise`. Immediate `return None`/`False`/empty literal is weaker `check_kind=early_exit_candidate`, intervention `replacement_return`; include separately in totals and manual review. Returning the response itself is not a guardrail. An immediate `continue` inside a loop containing this invocation is `check_kind=retry_branch`, intervention `retry_continue`; it is a candidate retry relationship, not proof a future invocation executes. Do not infer rejection/retry from arbitrary helper names or ordinary branching alone.

Parser-only `json.loads(response)` is `routes.route_kind=parser_only`, not a qualifying guardrail; it may propagate a derived response origin to a later supported check. Unknown `.validate(...)` or arbitrary validator-name matches are `unresolved_validator` routes only, never primary check counts. Known evaluation/judge calls receiving a response without an observed score/pass-fail use are `evaluation_only` routes. Library adoption/call names alone cannot qualify. General evaluator score propagation and callback/interprocedural evaluation (including a LangSmith target lambda) can remain `evaluation_flow_unassessed`; no completeness claim for complex checks. Wrapping parsers/validators in a runnable chain such as model | PydanticOutputParser is unassessed unless the source association is explicitly established; list this limitation.

Direct return `return llm.invoke(...)` is `output_form=returned_to_caller`; caller-side checks remain unassessed, not absent. Bare expression calls are `discarded` for this local analysis, not proof their effects/results are globally unused. Unknown transformations, schema factories, aliases and dynamic dispatch abstain with explicit limitations.

## Joern corroboration API

`corroborate_slices` is pure; it adds `joern_status,slice_ids,shared_slice_ids,response_slice_ids,slice_variables,slice_parse_flags` to each check. IDs are integer JSON-array values when serialized; all arrays deterministic. Input hit dictionaries require `slice_id,call_id,variable,role,parses`; main enriches/validates original index/source identities before calling it. `pdg_lines` uses original source line numbers, never generated snippet/scaffolding lines.

- Same `call_line == sink_line`: `joern_status=same_line_source_only`; no response-specific corroboration, even when the line occurs in a slice.
- Different lines, call and sink both in a saved hit slice, whose variable overlaps the source-traced `response_variables` and role is `binding`: `same_response_slice`.
- Shared lines but variable is receiver/prompt/unmatched, or role is heuristic `receiver/argument`: `shared_slice_unresolved_role`, not the primary slice-supported numerator. A reused input/output variable may have role argument; retain that uncertainty rather than discard the row.
- Call slices exist but no shared call/sink membership: `no_joint_slice`.
- No relevant saved slice: `no_saved_slice`.

`slice_ids` lists all relevant call-hit slices; `shared_slice_ids` only those with both lines; `response_slice_ids` only binding-role, source-response-variable-overlapping shared slices. Parse flags accompany all slice IDs, but original source evidence remains usable even if generated snippets fail to parse. No status claims a verified DDG/CDG path. Full per-edge paths were not persisted and no Joern query may be rerun for this task.

## Exact outputs and summary

Only fresh output directory; CSVs have headers even when empty. Booleans 0/1, counts explicit integers, unavailable identity empty, list cells JSON arrays. Extra fields allowed.

- `calls.csv`: `call_id,repo,rel_path,code_kind,framework,pattern,call_line,call_col,source_status,output_form,response_variables,source_check_count,slice_supported_check_count,route_count` (one row per selected call ID).
- `checks.csv`: `call_id,repo,rel_path,code_kind,call_line,sink_line,check_kind,mechanism,intervention,resolution,sink_source,response_variables,trace_lines,joern_status,slice_ids,shared_slice_ids,response_slice_ids,slice_variables,slice_parse_flags`.
- `routes.csv`: `call_id,repo,rel_path,code_kind,call_line,sink_line,route_kind,sink_source,response_variables`.
- `diagnostics.csv`: `call_id,repo,rel_path,kind,detail`.
- `summary.json` required paths: `subset`; `calls.denominator`, `calls.with_source_check`, `calls.with_response_slice_check`, `calls.with_validator_route`, `calls.with_runtime_intervention`, `calls.with_parser_only_route`, `calls.with_evaluation_only_route`; `checks.total`; `by_code_kind.<production|test|test_support>.denominator/with_source_check/with_response_slice_check/with_validator_route/with_runtime_intervention`; `source_statuses`; `joern_statuses`; `limitations` string list. Count unique call IDs in method-like numerator fields, never check/slice rows. Validator route means source-resolved schema-validation checks; runtime intervention means production code with assertion_failure/raise/retry_continue/replacement_return, reported by intervention kind separately. Neither means verified protected output. Rates name denominator and use null for zero denominator.
- `manifest.json`: actual saved input before/after hashes, inspected source hashes, code revision/options/Python, output hashes excluding manifest itself; changed consumed input fatal. Source files hashed and rechecked. No PyMethods2Test inputs.
- `REPORT.md`: findings first with exact fractions and strata; saved-slice availability; source-assessable counts; source-only versus response-slice-supported candidates; mechanism/intervention counts; unresolved cases and a small deterministic review sample. Explicitly say static invocation-site outputs, not number of runtime responses or full guardrail prevalence. Existing source classification may misclassify unconventional tests.py files; flag original-source test-named/TestCase evidence when noticed, without changing canonical strata/denominator.

## Focused literal acceptance scenarios

Keep roughly 8-10 focused test functions, not an exhaustive API matrix. Use synthetic source/tables; no application/model/graph execution.

1. `def f():\n r = llm.invoke(x)\n assert r is not None\n`, seed line2 col5 (actual literal indentation offsets must match fixture): assigned result r, one assertion on line3, trace [2,3]. A check of llm or prompt x instead yields zero. Reassignment r='fixed' kills association; assertion message only does not count. Distinct seed IDs stay distinct.
2. Source at module scope and async function with a multiline invocation: exact call line/column resolves; same-column chained `.strip()` is not mistaken for `.invoke`. Missing/ambiguous seed retained. Nested definitions do not leak into outer scopes.
3. `json.loads(r)` yields parser_only and zero check until an explicit downstream assertion; `unknown.validate(r)` is unresolved_validator, not a qualifying check. Imported jsonschema.validate(r, schema) qualifies; jsonschema.validate(fixed_instance, r) does not. Local BaseModel subclass / TypeAdapter validation use only response data port.
4. `if not r: raise ValueError(...)` qualifies rejection; `if not r: return None` is separate weaker early_exit; `if r: return r` does not qualify. Seed and `if not r: continue` in same loop yields retry_branch; ordinary iteration/printing is not a guardrail.
5. Response seeds/checks in one try/with/loop body can be found; uncertain merges and overwritten/mutated bindings never generate a definite cross-path result check. Known evaluation with ignored score remains route/unassessed, not a check.
6. Source check call2/sink3 with r; hit slice7 variable r role binding and PDG {2,3} -> same_response_slice and response_slice_ids [7]. Identical hit/PDG duplicates count once. Variable llm/role receiver cannot qualify. Argument-role r is unresolved-role, not strong corroboration.
7. Call2/sink2 plus binding slice7 -> same_line_source_only and zero primary response-slice numerator. Call2/sink3 with membership {2} -> no_joint_slice. No hit -> no_saved_slice. Parse flag0 preserves original-source candidate, not generated-body certainty.
8. Successful synthetic CLI patches load_calls; four headered saved fixtures only. One source candidate and binding slice produce denominator1/source1/slice1; missing source and missing slices remain in denominator. Existing output rejected before loader. Input/output hashes verified; no PyMethods2Test read. Empty repo subset writes valid zero summaries with null rates.
9. Canonical loader fake MF boundary uses supplied fake CALLS/Python DataFrame, checks snapshot equality, includes source code_kind strata and derives column from call ID. Raw SDK/FP scope belongs to MF, not duplicated in this module.

### Minimal successful CLI fixture

Call: `call_id=acme_app/app.py::2::8`, `repo=acme/app,repo_slug=acme_app,rel_path=app.py,code_kind=production,framework=langchain,pattern=.invoke,call_line=2,enclosing_qname=acme_app.app.f,repo_sliced=1,in_slice=1`; source `def f():\n    r = llm.invoke(x)\n    assert r is not None\n`.

Index: slice_id7, repo acme_app, rel_path app.py, code_kind production, method f, variable r, parses1. Hit uses same source/slice identity and call_id, call_repo acme/app, call_line2, variable r, role binding, parses1. PDG rows (7,2),(7,3). Expected summary calls.denominator=1, with_source_check=1, with_response_slice_check=1, with_runtime_intervention=1; checks.total=1. Fake loader returns matching call dictionary plus call_col=8. Tables may contain only required headers listed above.

## Invariants, boundaries and errors

All rows and source remain untouched; no producer imports except explicitly read-only canonical population/filter access. Source paths exact repo-slug normalized, reject absolute/traversal/symlink/junction/escape. Unsafe source rows remain diagnostics or fatal before source access; never read outside configured clones. Conflicting call/slice identities and missing required tables/schema are fatal with path/key in message. Identical duplicates/reordered rows yield the same deduplicated data counts. Call line/column integers must be nonnegative column and positive line; malformed IDs/schema fatal. Empty population valid. Unicode and Python source coding cookies supported. Missing source, parse errors, unsupported scope and absent slices remain unassessed, never cuts. Negative detections do not imply absent guardrails.

No full symbolic/heap/path-sensitive engine, general validator discovery/taxonomy, full interprocedural return flow, real executions, new Joern exports or verified DDG/CDG paths. Precision and checked properties require manual source review; report candidate rates before any validated guardrail-prevalence claim. No blocking open questions for this bounded saved-data pass.

### Obvious constant/dead-condition abstention

A syntactic mention is insufficient when the check's truth outcome cannot depend on the response. Do not qualify `assert True or r`, `assert r or True`, `assert False and r`, `assert r and False`, `assert [r]` or `assert {"value": r}` as response-dependent guards. Likewise an unconditional truth/failure branch such as `if r or True: raise ...` is not a response-specific check, and code under literal `if False` does not supply a candidate. Bounded recognition/abstention for these obvious literal/short-circuit/container truth cases is sufficient; no general symbolic engine is required. Keep limitations on unsupported more complex expressions. Extend the existing response-vs-receiver/assertion scenario, not an exhaustive truth-table test matrix.

Schema-contract reference sources for the external interpretation notes: [jsonschema validate(instance, schema)](https://github.com/python-jsonschema/jsonschema/blob/main/jsonschema/validators.py), [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/), and [Pydantic validation errors](https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md). These document validation/failure contracts; the analysis only resolves source routing and never executes the validators.

### Minimal evaluation-route fixture and manifest shape

For this first pass, the concrete source-resolved evaluation example is `from openevals.llm import create_llm_as_judge` (including stable explicit import aliases), followed by `judge = create_llm_as_judge(...)`. A bare `judge(outputs=r)` receiving a traced response in its **outputs** port produces one `routes` row with `route_kind=evaluation_only` and no qualifying check. If its result is assigned or otherwise consumed, retain `route_kind=evaluation_flow_unassessed` rather than calling the outcome ignored; propagation from judge feedback to a later score check may remain unassessed. Response mentioned only in judge prompt/model/config or other input ports is not output-evaluation evidence. Rebinding/shadowing the factory or judge variable invalidates the source-resolved association. A LangSmith `evaluate` target callback and its result flow may remain unassessed. This is the only required evaluator-factory fixture; do not expand the registry.

Literal fixture: `from openevals.llm import create_llm_as_judge\ndef f():\n    judge = create_llm_as_judge(prompt="rubric", model="provider:model")\n    r = llm.invoke(x)\n    judge(outputs=r)\n`. Seed line4/column8 yields zero checks and one evaluation_only route at line5. Replacing the last statement by `feedback = judge(outputs=r)` yields evaluation_flow_unassessed, not evaluation_only.

`manifest.json` exposes these exact required keys (extra keys allowed):

```json
{
  "inputs": {"slice_calls_by_call.csv": {"sha256_before": "<lowercase hex SHA-256>", "sha256_after": "<same SHA-256>"}},
  "inspected_sources": {"<original source path>": "<lowercase hex SHA-256>"},
  "outputs": {"calls.csv": "<lowercase hex SHA-256>"},
  "revision": "<Git commit or unavailable>",
  "python": "<version>",
  "options": {}
}
```

For the four named slice inputs, `inputs` keys are artifacts-root-relative forward-slash names exactly as listed in the input schema table. Additional actual canonical-scope saved-table reads may use documented repository-relative forward-slash keys. `inspected_sources` records each original source used. `outputs` hashes every generated file except manifest itself; no recursive self hash. Successful input pairs must match; inspected source hashes must be rechecked before success. A patched-loader synthetic CLI run hashes exactly its four required saved tables and no real-corpus source/table outside the fixture.
