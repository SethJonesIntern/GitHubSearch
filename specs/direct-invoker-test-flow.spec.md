# Independent direct-invoker test-result flow from saved Stage-5 edges

## Purpose

Independently of **all PyMethods2Test artifacts**, count canonical observed direct invokers with saved Stage-5 test-call evidence whose returned value reaches a test pass/fail check in original source. This bounded pass uses the saved pyan **first-BFS-edge forest**, not a complete call graph: results are a candidate-detection measurement, never exhaustive test prevalence. Keep the complete direct-invoker denominator, including test/support code, missing source and invokers without a saved candidate edge. No full source-tree scan, graph production, Joern rerun, application execution, or changes to saved research outputs.

## Public API

New module `pipeline/direct_invoker_test_flow.py`; new tests `pipeline/test_direct_invoker_test_flow.py`.

```python
def load_direct_invokers() -> list[dict]: ...
def select_test_edges(invokers: list[dict], tests: list[dict]) -> tuple[list[dict], list[dict]]: ...
def analyze_saved_test(source: str, test_line: int, target_qname: str) -> dict: ...
def main(argv: list[str] | None = None) -> int: ...
```

`load_direct_invokers` reads canonical saved population/filter definitions lazily; no import-time corpus IO. Returns unique dictionaries with `repo,qname,file,line,reason,kind`, line as integer, original saved slug-prefixed `file` key preserved.

`select_test_edges` is pure. Inputs use the six fields above; tests may have direct/transitive kinds. Return `(edges, diagnostics)`. Each edge has `repo,invoker_file,invoker_line,invoker_qname,test_file,test_line,test_qname,reason`. Diagnostic keys `repo,file,line,kind,detail`. Select saved test rows with kind `transitive` and reason exactly `calls <qname>` where `(repo,<qname>)` identifies exactly one canonical direct invoker. No target lookup by basename. Deduplicate complete edge identity. Skip identical test/invoker `(repo,file,line)` keys. Ambiguous canonical qname yields `ambiguous_target` diagnostic rather than an arbitrary target. Direct-test rows do not supply self mappings. Saved tests pointing at transitive wrappers are out of this pass; count them separately as unresolved/deferred, not evidence of no test.

`analyze_saved_test` returns:

```python
{
 "status": "ok",  # parse_error, definition_missing, no_matching_call, ambiguous_call
 "detail": "",
 "checks": [{"call_line": 2, "check_line": 3, "failure_line": None,
             "check_kind": "assert", "call_source": "f()", "check_source": "assert r",
             "trace_lines": [2, 3], "variables": ["r"],
             "evidence": "saved_pyan_edge_local_return_to_passfail_candidate"}],
 "limitations": []
}
```

Identify exact sync/async definition at test_line; nested definitions are excluded. Return statuses for syntax failure/missing definition rather than raising. Empty/invalid target_qname raises ValueError. Call selection is a unique syntactic call with final name equal to target_qname's final component, optionally an explicit local import alias to that same target module/name. It is corroborated by the saved qname edge, **not independently resolved dispatch**. Multiple matching source calls abstain as `ambiguous_call`; no matching call yields `no_matching_call`. Constructor aliases, inheritance/dynamic receivers, shadowing, fixtures and arbitrary helper aliases are unresolved; do not guess by unique basename across the repository. Record this limitation on every successful result: `Saved pyan target plus syntactic call match; dispatch and execution are unverified.`

CLI:

```text
py -3.14 -B -m pipeline.direct_invoker_test_flow --output-dir NEW_PATH [--repos-dir PATH] [--repo OWNER/NAME ...]
```

Repos default to repository `pipeline/repos`; output required and must not exist. Repeatable repo filters label results subset. Return 0 on completed analysis including zero findings, nonzero on fatal inputs/schemas/changed inputs/existing outputs. No dependency installation or environment upgrades.

## Canonical denominator and inputs

`load_direct_invokers` lazily imports existing `Applications.make_figures` (MF) read-only observed population and AZ/RAW_SDKS definitions. Read raw `paths.LLM_INVOKERS_CSV`; apply `MF.AZ.drop_removed_patterns_invokers`, kind direct, repo in `MF.POP`, exclude framework extracted by `MF.AZ.fw_from_reason(reason)` in `MF.RAW_SDKS`; deduplicate `(repo,file,line)`. These existing functions/constants are authoritative; do not reimplement their scope. Do not call figure or analysis main/F5/render functions. Current historical count is 18,961 in 643 observed repositories; compute, do not hardcode.

Read `paths.LLM_TESTS_CSV` independently, not focal mappings. Both raw CSV schemas are `repo,qname,file,line,reason,kind` (other fields allowed). Invoker/test file values preserve slug prefixes such as `acme_app/app.py`; normalize only for source access. Current saved firstedges identify 877 direct targets through 4,187 test edges before source validation; compute these values. No `method_test_baseline`, `pym2t_*`, focal-map APIs, or eligibility conditioned on any of them. Pure source-expression helpers from `invoker_result_checks` may be reused only if importing them does not read PyMethods2Test or run its driver.

Original source access is only for the candidate test definitions and canonical invoker definition files. Clone path is `repos_dir/OWNER_NAME/relative_file`. Normalize exact slug prefixes; reject absolute/traversal/symlink paths and root escapes. Cache/batch per-source-file reads, honor source encoding via tokenize.open. Missing clone/file/definition or source parse error is retained diagnostic/unassessed state, not a denominator cut. Classify invoker strata with existing pure `slice_index.code_kind` where practical, retaining test/test_support/production totals. This classification is heuristic.

Hash before/after every saved input consumed by canonical population/filtering and candidate loading, including invoker/test CSVs, calls/progress/audit and any other actual table reads. Missing LFS content or dependencies yields actionable failure; never regenerate. Source hashes recorded for inspected files.

## Return-to-pass/fail behavior

Support direct/awaited return assignment to simple local names, local aliases/derived attributes/subscripts, reassignment kills, Python assert conditions and a documented small unittest value-assertion whitelist on `self`. Support result-derived `len`, `isinstance`, `bool`, `str` and startswith/endswith candidates as in the pure local analyzer. Do not use assertion messages as check evidence. Receiver/argument checks do not become return checks. Arbitrary validator calls abstain. Do not infer alias flow across compound branch/loop/try/with boundaries; record uncertainty and kill potentially reassigned bindings afterward.

Also support a result-derived `if` predicate with an immediate unconditional `raise`, explicitly resolved `pytest.fail(...)`, or `self.fail(...)` in its body as `check_kind=controlled_failure`; check_line is predicate line and failure_line is failure statement. No nested-control inference, arbitrary failure helpers, caught raises, or uncertain/shadowed pytest aliases. It is a static control candidate, not proof a failure branch executes. An assertion reached only conditionally may remain a candidate if local result use itself is established; no unsupported branch merging.

Source-level findings are labelled `saved_pyan_edge_local_return_to_passfail_candidate`. They support that an invoker may affect a test's outcome under the saved target identification; they do not establish semantic correctness, mocking status, actual invocation or full interprocedural LLM-result flow. Do not report just receiver existence as a returned-result check.

## Exact required output subsets

All new files under an exclusively created output directory. Extra columns allowed. CSV booleans `0/1`; counts decimal integers; list fields JSON arrays; blank unavailable identities; deterministic ordering.

- `invokers.csv`: `repo,file,line,qname,stratum,source_status,saved_test_edges,candidate_checks,has_result_check_candidate` (one row per canonical direct invoker, including zero/unassessed).
- `test_edges.csv`: `repo,invoker_file,invoker_line,invoker_qname,test_file,test_line,test_qname,source_status,candidate_checks,target_evidence`; target_evidence exactly `saved_pyan_firstedge_plus_syntactic_call`.
- `checks.csv`: `repo,invoker_file,invoker_line,invoker_qname,test_file,test_line,test_qname,call_line,check_line,failure_line,check_kind,evidence,call_source,check_source,trace_lines,variables`.
- `diagnostics.csv`: `repo,file,line,kind,detail`.
- `summary.json` required paths: `subset` boolean; `invokers.denominator`, `invokers.with_saved_direct_test_edge`, `invokers.with_result_check_candidate`; `test_edges.selected`, `test_edges.source_assessed`, `test_edges.with_result_check_candidate`; `checks.total`; `by_stratum` with denominator/with_result_check_candidate per stratum; `limitations` string array. Also report deferred wrapper edges, ambiguous targets/source calls and missing/parse source counts. Unique invoker numerators, never multiplied by tests/assertions.
- `manifest.json`: `inputs` keyed by consumed saved-input repository-relative path with `sha256_before/sha256_after`; inspected source hashes, Python/options/revision, output hashes except manifest self hash. Matching input hashes required on success.
- `REPORT.md`: findings first exact counts/fractions, independent firstedge candidate frame, source-assessment eligibility, strata, unresolved/missingness, deterministic positive/negative examples. Explicitly state no PyMethods2Test used, saved forest contains only one chosen edge per transitive function, multi-hop wrappers and unrecorded edges are unassessed, and no candidate means no detection rather than no test. No exhaustive prevalence, coverage, runtime guardrail or correctness claim.

No Joern input is required. Optional saved Joern membership may corroborate only local candidates under separate candidate labels; per-method slice memberships cannot prove cross-method flow. Do not rebuild graphs.

## Focused acceptance scenarios

Approximately ten focused synthetic scenarios; no corpus execution:

1. Canonical loader calls existing scope/filter boundary; raw SDK/transitive/out-of-POP exclusions and repo/file/line dedup. Inject/mock boundary rather than real corpus. Verify zero PyMethods2Test reads.
2. Pure edge selection joins exact repo+qname, deduplicates, rejects ambiguous qname and self-target; wrapper edge is deferred, not a direct hit.
3. Plain source test without import pytest, saved target `acme_app.app.f`, `r=f(); assert len(r)>0`: one candidate with target-evidence caveat.
4. Async/awaited return and `self.assertEqual(r,"x")`: supported; distinct same-named tests at different definition lines retained.
5. Alias/derived value supported; overwrite kills association; receiver-only/message-only checks negative (parameterized cases acceptable).
6. Result predicate controlling immediate raise/resolved pytest.fail/self.fail qualifies; unrelated predicate or caught raise abstains.
7. Two source calls with target final name yield ambiguous_call; absent name yields no_matching_call; no global name fallback.
8. Nested-definition exclusion and compound branch uncertainty; arbitrary validators unresolved.
9. Missing clone/source/definition/parse failure preserves every canonical invoker and reports unassessed; test/support strata retained.
10. CLI existing-output refusal, path/symlink safety, unchanged input hashes, deduplicated deterministic counts, empty subset zero counts/null rates.

## Errors/non-goals/open questions

Existing output and missing required saved input fail with a message identifying the path. Invalid row/schema/conflicting identity fails visibly; identical duplicate rows deduplicate. Changed input hash is fatal. Source failures remain diagnostics. Zero findings is successful analysis.

No new discovery/clone/graph/Joern/analysis producer; no full-corpus source walk; no PyMethods2Test reads; no full dispatch resolver or symbolic engine; no transitive wrapper flow; no real-test execution or mock/correctness claim. No blocking open questions for this bounded saved-edge candidate pass. An exhaustive independently discovered test-call analysis is later work, rather than a claim of this report.

### Candidate-frame coverage gap

A Stage-5 test whose own body contains a framework invocation can be saved as `kind=direct` with only a `matches ...` reason even if it also calls another invoker. Such rows cannot contribute target edges in this saved-forest pass. Report their observed-repo count as `test_edges.direct_test_rows_without_target_edge`; do not call them untested invokers or assume their unrelated target calls do not exist. Similarly first-BFS selection discards other outgoing call relationships. This is why the final fraction is labelled a saved-edge result-check candidate detection rate over all direct invokers, not total true test association.

## Synthetic loader and CLI dependency boundary

The CLI imports `from pipeline import paths` and resolves input/default paths at **call time**, not into module-level copies. Tests may monkeypatch these exact existing constants to temporary fixture paths:

- `paths.ARTIFACTS_DIR`: temporary artifact directory;
- `paths.LLM_INVOKERS_CSV`: temporary six-column invoker CSV;
- `paths.LLM_TESTS_CSV`: temporary six-column test CSV;
- `paths.BATCH_PROGRESS_JSON`: temporary progress JSON if canonical population is exercised;
- `paths.REPOS_DIR`: temporary clone root, unless CLI `--repos-dir` is supplied.

A synthetic CLI success fixture may monkeypatch this module's public `load_direct_invokers` to return fixture rows, patch the constants above, write the raw invoker/test CSVs, create source files under the supplied clone root, and call `main(["--repos-dir", str(repos), "--output-dir", str(fresh)])`. With loader replaced, the CLI must not independently import canonical figure modules, recompute population, or read any real artifact path. Mandatory saved inputs for this synthetic path are exactly LLM_INVOKERS_CSV and LLM_TESTS_CSV; other provenance inputs are hash-tracked only when actually read, not required merely because the production loader ordinarily consumes them. A temporary progress JSON may be empty. The output manifest still includes both raw CSV hashes.

For an isolated loader unit test, its canonical import boundary is `from Applications import make_figures as MF` executed **inside `load_direct_invokers`**. A test may install a fake `Applications.make_figures` module (including the package attribute, using monkeypatch) with `POP`, `RAW_SDKS`, and `AZ` fields. `AZ.drop_removed_patterns_invokers` accepts and returns an installed-pandas DataFrame, while `AZ.fw_from_reason` accepts one reason string and returns its framework or None. No new dependency is required beyond existing research pandas. Fake AZ functions can track that the existing filter boundary was called; fixture raw rows verify direct/POP/raw-SDK/dedup logic without real canonical population files. Canonical import failure must not trigger fallback reimplementation or artifact regeneration.

Existing-output refusal is checked before calling the loader or reading corpus inputs. Synthetic fixtures with zero findings may complete successfully. No hidden artifacts directory enumeration for unrelated tables: provenance records actual relevant reads, not every file under ARTIFACTS_DIR.
