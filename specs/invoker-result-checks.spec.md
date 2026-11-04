# Invoker result/check evidence

## Purpose

Read saved production-method baseline, full PyMethods2Test mappings, original relevant source, and saved Joern slice memberships to obtain inspectable **static candidates** connecting invocation results to assertions. Use the same baseline method population for method-level and broader file-level correspondence. No producer reruns or existing artifact changes.

## Public API

New module `pipeline/invoker_result_checks.py`; new tests `pipeline/test_invoker_result_checks.py`.

```python
def normalized_relative(value: str, slug: str) -> str: ...
def analyze_local_checks(source: str, definition_line: int, seed_call_lines: set[int]) -> dict: ...
def main(argv: list[str] | None = None) -> int: ...
```

`normalized_relative` accepts repo-relative paths or paths beginning exactly `slug + "/"`, normalizes backslashes, strips that prefix. Absolute paths and `..` components raise `ValueError("Unsafe source path")`. Empty path raises the same exception. No suffix matching.

`analyze_local_checks` identifies an exact sync/async function definition line. Result shape:

```python
{
 "status": "ok",  # also parse_error, definition_missing, ambiguous_seed
 "detail": "",
 "checks": [{"call_line": 2, "check_line": 3, "check_kind": "assert",
             "check_source": "assert r is not None", "trace_lines": [2, 3],
             "variables": ["r"], "evidence": "syntactic_local_result_to_assert_candidate"}],
 "limitations": []
}
```

Parsing failure returns `parse_error`, not an exception; a nonexistent definition returns `definition_missing`. Multiple candidate calls beginning on a seed line return `ambiguous_seed` and no checks. Otherwise unsupported expressions/flow produce no finding and an explanatory limitation. Empty seeds return `ok` with no checks.

CLI (defaults resolve from repository root):

```text
py -3.14 -B -m pipeline.invoker_result_checks --artifacts-dir PATH --repos-dir PATH --output-dir PATH [--repo OWNER/NAME ...]
```

`--artifacts-dir` defaults to `pipeline/artifacts`, `--repos-dir` to `pipeline/repos`; output is required and must not exist; repeatable repo filters are optional and label results as a subset. CLI returns 0 on success, nonzero on fatal input/output/schema errors. No import-time IO.

## Behavior

| # | Input | Expected output | Notes |
|---|---|---|---|
| 1 | `def f():\n r = agent.invoke(x)\n assert r is not None\n`, definition 1, seeds `{2}` | one check, call 2, assert 3, variables `["r"]`, trace `[2,3]` | Return value only |
| 2 | `def f():\n r = agent.invoke(x)\n assert agent is not None\n` | no checks | Receiver is not result |
| 3 | `def f():\n r = agent.invoke(x)\n s = r.content\n assert s == "yes"\n` | one check at 4; trace `[2,3,4]` | Local aliases/derived expressions |
| 4 | `def f():\n r = agent.invoke(x)\n r = "fixed"\n assert r\n` | no checks | Reassignment kills result association |
| 5 | `def f():\n r = agent.invoke(x)\n assert True, r\n` | no checks | Assertion message excluded |
| 6 | `def f():\n assert agent.invoke(x) is not None\n`, seed `{2}` | one check at 2 | Direct and awaited calls supported |
| 7 | `def f():\n r = agent.invoke(x)\n self.assertEqual(r, "yes")\n` | unittest_assert candidate at 3 | Whitelist below |
| 8 | `def f():\n r = agent.invoke(x)\n validator(r)\n` | no assertion candidate | Arbitrary validators deferred |
| 9 | same mapped test CSV row twice | one test link | Identity includes definition line |
| 10 | test `test_x` at lines 10 and 20 in same file | two test links | Distinct definitions |
| 11 | saved slice lines `[2,3]`, candidate seed 2/check 3 | `same_slice_candidate` | Never verified dependence |
| 12 | slice lines `[2]`, check 3 | `not_in_same_saved_slice` | Not proof check is absent |
| 13 | no relevant saved slice | `no_saved_slice` | Unassessed slice evidence |

The source analyzer supports simple assignment and annotated assignment to local names, straight-line aliases/derived expressions, Python assertions, and a conservative unittest value-assertion whitelist: `assertEqual`, `assertNotEqual`, `assertTrue`, `assertFalse`, `assertIs`, `assertIsNot`, `assertIsNone`, `assertIsNotNone`, `assertIn`, `assertNotIn`, `assertIsInstance`, `assertNotIsInstance`, `assertGreater`, `assertGreaterEqual`, `assertLess`, `assertLessEqual`, `assertAlmostEqual`, `assertNotAlmostEqual`, `assertRegex`, `assertNotRegex`, `assertCountEqual`. Only `self.<whitelist>` calls qualify; exclude message arguments. Arbitrary similarly named methods and raises/warns/logs context managers do not qualify.

Ignore nested functions/classes when analyzing an outer definition. Do not propagate inferred return-value association across compound branch/loop/try/with boundaries. Invalidate potentially assigned names after unsupported compound statements and record a limitation. Conservative abstention is preferred over a general taint engine. Source analysis is syntactic and does not resolve calls or evaluate expressions.

## Inputs and population

Read only these saved tables plus the relevant original `.py` files:

- `method_test_baseline/methods.csv`, `summary.json`, `by_repo.csv`;
- `pym2t_focal_tests.csv`, `pym2t_focal_calls.csv`;
- `slice_index.csv`, `slice_call_hits.csv`, `slice_pdg_lines.csv.gz`.

Use unique `group == "llm"` baseline methods keyed by `(repo,file,line)` and validate qname/end. Current expected snapshot is 7,642 methods, 220 mapped, but compute from inputs and reconcile with saved summary. Do not use raw invoker population or the historical 9,700 direct-test subset as denominator.

Join focal rows on normalized repo/file/focal_line and require focal method name/end agreement. Preserve exact/fuzzy and file-rule labels. Compute file association and import-only file association independently on this same method population from all identified-test rows.

Join observed saved calls to their innermost baseline method range; preserve call IDs. For mapped tests inspect the exact original test definition, find calls whose final called name equals the mapped method's final name, and analyze those seed returns. Label `callee_resolution = "name_only"`; do not claim resolved dispatch. No syntactically matching call is a retained diagnostic, not a dropped mapping.

Analyze production-side saved framework calls separately from test-side calls to the mapped method. For production candidates, use matching saved call-hit slice IDs and PDG line membership to ask whether call and assertion share a slice. Preserve slice IDs/variables and parse flags. Joern status is one of `same_slice_candidate`, `not_in_same_saved_slice`, `no_saved_slice`, `not_assessed` (test-side corroboration can remain not_assessed). Generated snippet scaffolding is not program evidence. Never conflate two local chains into a proven cross-method chain.

Stream large tables and retain only relevant baseline/mapping/slice entries. Read each needed source once or with a bounded cache, using original source encoding. Never scan all clones recursively. Reject symlinks/paths escaping the configured clone root.

## Outputs

Create only a fresh output directory, exclusively. Files:

- `methods.csv`: one row per selected LLM baseline method; identity, mapping/file/import-only association, source status, call/slice availability, test-side and production-side candidate counts, shared-slice candidate counts.
- `test_links.csv`: unique method/test identities/ranges, exact/fuzzy/file rule, source status, syntactic call count, assertion candidate count, name-only resolution caveat.
- `checks.csv`: evidence side; method/test identities; call/check lines; original source snippets; trace lines/variables; Joern status, slice IDs and variables. Deterministic ordering.
- `diagnostics.csv`: typed missing/parse/join/ambiguous/unsupported records and identity/detail.
- `summary.json`: exact numerators/denominators over common method population, assessed eligibility, missingness, exact-only sensitivity, candidate rows versus unique methods, subset label and limitations.
- `manifest.json`: Python/revision/options, saved input SHA-256 before/after, inspected source hashes, output hashes. Fail visibly if any input changes.
- `REPORT.md`: findings first with exact fractions; assessment limits, a small deterministic positive sample and missing/disagreement examples. Call results candidates; absence of detection is not absence of checking.

## Errors and boundaries

| Condition | Contract |
|---|---|
| Existing output directory or file | CLI nonzero before modifying it; message contains `exists` |
| Missing required table | CLI nonzero; message identifies missing path |
| Invalid schema or inconsistent baseline identity | CLI nonzero with table/key detail |
| Missing original source | diagnostic `source_missing`; retain method/test denominator |
| Source parse error | diagnostic `parse_error`; unassessed, never negative finding |
| Changed input hash | CLI nonzero and visible changed-input diagnostic |
| Empty selected population | valid empty tables, zero counts and null rates; label subset |
| Unicode source/path | accepted if safe; source encoding respected |
| Negative/zero definition line | `definition_missing` |
| Duplicate/reordered rows | stable deduplicated keys/output; conflicting identities are errors |

## Invariants

No original files change. No old outputs are replaced. Every method numerator is deduplicated and uses an explicitly named denominator. No producer import/launch, network/cloning, Joern rerun, application execution or broad clone scan. Tests are synthetic isolated fixtures. Required existing synthetic checks and focused new tests must pass; inspect a bounded evidence sample before full read-only run.

## Non-goals

No verified correctness/coverage claims; no interprocedural return/state/exception graph; no runtime guardrail prevalence; no validator taxonomy; no general heap/alias/path-sensitive taint; no assertion-message evidence; no conclusions about mocks or actual execution; no headline floor/ceiling interpretation.

## Open questions

None blocking this bounded candidate analysis. Later validation must determine call resolution, mocking, actual checked property, and whether local production output is the value checked by the associated test.

## Integration schema contract (required subsets)

The following headers/JSON fields are the **minimal accepted fixture schema**. Extra real-corpus fields are allowed and ignored unless needed for diagnostics. CSVs have a header even when empty; read UTF-8 with optional BOM. Numeric CSV fields use decimal strings. Blank focal fields represent no focal method.

| Input relative to artifacts root | Required headers/JSON fields |
|---|---|
| `method_test_baseline/methods.csv` | `repo,file,qname,line,line_end,group,mapped_tests,exact_mapped_tests` |
| `method_test_baseline/by_repo.csv` | `repo,llm_methods,llm_mapped,llm_exact_mapped` |
| `method_test_baseline/summary.json` | `pooled_all_complete_repos.llm.methods`, `pooled_all_complete_repos.llm.with_mapped_test`, `pooled_all_complete_repos.llm.with_exact_test` |
| `pym2t_focal_tests.csv` | `repo,test_file,test_method,test_line,test_line_end,focal_file,file_rule,focal_method,focal_line,focal_line_end,method_rule` |
| `pym2t_focal_calls.csv` | `repo,call_id,file,enclosing_qname,framework,call_line` |
| `slice_index.csv` | `slice_id,repo,rel_path,code_kind,method,variable,parses` |
| `slice_call_hits.csv` | `slice_id,repo,rel_path,code_kind,method,variable,call_id,call_repo,call_code_kind,call_line` |
| `slice_pdg_lines.csv.gz` | gzip CSV `slice_id,line` |

In baseline tables, repo is `OWNER/NAME`, file is repository-relative, and method qname excludes the repository slug. In focal tables, test/focal paths may begin `OWNER_NAME/`; normalize with the exact repo slug. The saved slice index/hits `repo` is the slug `OWNER_NAME`, while call inventory and hits `call_repo` use `OWNER/NAME`. `slice_id` globally identifies a saved slice. Source files are under `repos_dir / OWNER_NAME / relative_file`.

Baseline reconciliation is between all saved LLM method rows, by-repo LLM counts, and the three summary fields above. Compute validation on the full saved baseline before applying a `--repo` subset; compare selected output counts only against selected rows. Fixture summary may contain only those three fields, with synthetic values such as methods=1, with_mapped_test=1, with_exact_test=1. Never hardcode corpus counts. A focal mapping is valid only if its file/line/end/name matches a baseline method. Invalid focal definitions become join diagnostics and do not alter saved baseline mapping flags. The distinction between saved flags and successfully joined evidence must be retained in counts.

Output CSVs must include at least these exact columns (additional columns allowed):

| Output | Required columns |
|---|---|
| `methods.csv` | `repo,file,qname,line,line_end,mapped_test,file_associated,import_file_associated,source_status,test_candidate_count,production_candidate_count,shared_slice_candidate_count` |
| `test_links.csv` | `repo,file,line,test_file,test_method,test_line,method_rule,file_rule,source_status,matched_call_count,candidate_count,callee_resolution` |
| `checks.csv` | `repo,file,line,side,test_file,test_method,test_line,call_line,check_line,check_kind,check_source,trace_lines,variables,evidence,joern_status,slice_ids,slice_variables` |
| `diagnostics.csv` | `repo,file,line,kind,detail` |

CSV conventions: booleans are `0`/`1`; integer counts include explicit `0`; unavailable source/test identity fields are empty strings. `methods.csv.line` is the production method definition line. `checks.csv.line` is also that production method line; `side` is exactly `test` or `production`. Test identity fields on production-side rows are empty. List-valued fields (`trace_lines`, `variables`, `slice_ids`, `slice_variables`) are JSON arrays in CSV cells, not delimiter-separated strings. `slice_ids` are JSON integers sorted ascending. `evidence` is exactly `syntactic_local_result_to_assert_candidate`. `callee_resolution` on test links is exactly `name_only`. Candidate counts count deduplicated checks per side, but summary method counts count each method once. `mapped_test` reflects saved `mapped_tests > 0`; `file_associated`/`import_file_associated` reflect full identified-test file mappings independently of focal-method matching.

`summary.json` must expose these exact required fields; other keys allowed:

```json
{
  "subset": false,
  "methods": {
    "denominator": 1,
    "with_mapped_test": 1,
    "with_exact_mapped_test": 1,
    "with_file_association": 1,
    "with_import_file_association": 1,
    "with_test_candidate": 1,
    "with_production_candidate": 1,
    "with_shared_slice_candidate": 1
  },
  "checks": {"test": 1, "production": 1, "production_same_slice": 1},
  "assessment": {"production_source_ok": 1, "test_links": 1, "test_source_ok_links": 1},
  "limitations": ["Static candidates do not establish execution or correctness."]
}
```

Values shown are a literal synthetic example, not defaults. In `assessment`, source-ok counts mean exact definition located and source parsed, including rows with zero candidate checks; unassessable rows do not count. `methods.with_shared_slice_candidate` counts production methods with at least one production-side candidate marked `same_slice_candidate`. Test-side `joern_status` is `not_assessed` in this first pass. Exact-only sensitivity may use additional clearly named keys; it must not reinterpret fuzzy mappings as exact.

`manifest.json` must include `inputs`, a JSON object keyed by artifacts-root-relative forward-slash path, whose values have `sha256_before` and `sha256_after` lowercase hexadecimal strings. It must include all eight required saved inputs listed above. Each pair must match on successful exit. Other provenance/output/source-hash fields may use implementation-selected names but must be documented in REPORT.md. Do not include the manifest's own hash recursively.

### Literal shared-slice fixture

Use repo `acme/app`, slug `acme_app`, production file `app.py`, qname `f`, lines 1..3 and source:

```python
def f():
    r = agent.invoke("hello")
    assert r is not None
```

Saved call row has call_id `acme_app/app.py::2::8`, file `acme_app/app.py`, call_line `2`, repo `acme/app`, enclosing_qname `acme_app.app.f`, framework `langchain`.

Index row: slice_id `7`, repo `acme_app`, rel_path `app.py`, code_kind `production`, method `f`, variable `r`, parses `1`. Hit row uses the same slice/repo/path/method/variable, call_id above, call_repo `acme/app`, call_code_kind `production`, call_line `2`. Compressed PDG CSV has rows `(7,2)` and `(7,3)`. Expected production check: `joern_status == "same_slice_candidate"`, `slice_ids == [7]`, `slice_variables == ["r"]`, and summary `methods.with_shared_slice_candidate == 1`. Removing `(7,3)` changes status to `not_in_same_saved_slice`; removing hit/index membership changes it to `no_saved_slice`. Neither variant changes the method denominator or AST candidate count. Focal tables can be empty for this production-side fixture and saved baseline mapped flags can be zero, consistently reflected in its by-repo/summary fixture values.

## Derived-expression clarification

For `analyze_local_checks`, result association in expressions includes attribute access, subscripts, comparisons, boolean/unary/binary operations, and the following **syntactic builtin-name call whitelist**: `len`, `isinstance`, `bool`, `str`. Propagate association from their value arguments. This whitelist is name-based, not resolved dispatch; add limitation `Builtin names may be shadowed; calls are not resolved.` when used. For `isinstance(value, type_expression)`, propagate from its first argument only. Do not infer association merely because an unrelated seeded value occurs as its type argument. For the other three builtins, inspect the first positional value argument only; unsupported signatures may abstain with a limitation. An arbitrary call `validator(r)` remains unsupported even when it appears inside an assert.

Also support the string/sequence predicate whitelist `startswith`, `endswith`, `__contains__` when invoked as an attribute on a result-associated expression, e.g. `result.content.startswith("yes")`. Propagate from that receiver, not from unrelated arguments. These are **syntactic predicate candidates**: receiver type and dynamic method implementation are unresolved and must be stated as a limitation. Do not support arbitrary receiver methods or infer arbitrary validator semantics.

Additional literal behavior cases (seed line 2, definition line 1):

| Input source | Expected behavior |
|---|---|
| `def f():\n r = agent.invoke(x)\n assert len(r.steps) > 0\n` | one candidate at check line 3; variables include `r` |
| `def f():\n r = agent.invoke(x)\n assert isinstance(r, str)\n` | one candidate at 3 |
| `def f():\n r = agent.invoke(x)\n assert bool(r)\n` | one candidate at 3 |
| `def f():\n r = agent.invoke(x)\n assert str(r) == "yes"\n` | one candidate at 3 |
| `def f():\n r = agent.invoke(x)\n assert r.content.startswith("yes")\n` | one candidate at 3; unresolved receiver-type limitation |
| `def f():\n r = agent.invoke(x)\n assert validator(r)\n` | no candidate; unsupported-call limitation |
| `def f():\n r = agent.invoke(x)\n assert isinstance("fixed", r)\n` | no candidate; result is not the value being checked |
| `def f():\n r = agent.invoke(x)\n assert "fixed".startswith(r)\n` | no candidate; receiver is not result-associated |
