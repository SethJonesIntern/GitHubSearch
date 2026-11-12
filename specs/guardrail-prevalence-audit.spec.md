# Source-reviewed guardrail prevalence audit

Status: implementation contract, 2026-11-12. This is a read-only consumer and
aggregation feature, not a new guardrail detector. No blocking open questions.

## Objective and boundaries

Use the same saved production-method frame for test-suite and runtime guardrails.
Generate source-review worklists from existing candidates and aggregate explicit
review annotations under the two-arm codebook. Never promote syntax/API matches,
focal mappings, keyword findings, or saved slice membership to verified guardrails.

No application, model, validator, Joern, focal-mapping, producer, population or audit
job may run. No new scope cuts, original artifact edits, dependency installation,
or source execution. The baseline remains preliminary and complete-case, with
55 original excluded/unresolved repositories; figures describe this frame only.

Implement in Node stdlib ES modules: Python 3.14/`py` is unavailable in this host's
current tool PATH, and the repository forbids substituting Python 3.12. This small
consumer does not change existing Python producers. No import-time corpus IO.

## Files and exact public API

Implementation: `pipeline/guardrail_prevalence_audit.mjs`. Tests:
`pipeline/test_guardrail_prevalence_audit.mjs` using `node:test` and strict asserts.

```js
export function parseCsv(text) // => Array<Record<string,string>>; RFC-style quotes/newlines, BOM
export function normalizeRel(file, repo) // => repo-relative forward-slash path; reject unsafe paths
export function methodKey(repo, relPath, line) // => `${repo}::${normalizeRel(relPath,repo)}::${line}`
export function buildWorklists({ methods, mappings, calls, checks, independentChecks = [] })
// => {runtime: object[], testSuite: object[], diagnostics: object[]}
export function aggregateAnnotations({ methods, mappings, calls, annotations })
// => {methods: object[], summary: object, verified: object[], rejected: object[], unresolved: object[]}
export async function runAudit({frameDir, callsCsv, checksCsv, independentChecksCsv = null,
  annotationsCsv = null, reposDir, outputDir}) // => final summary object
export async function main(argv = process.argv.slice(2)) // => integer exit code
```

Pure public APIs validate schema/identity and never read files or access globals.
`runAudit` performs safe source-evidence verification, provenance, and output IO.
CLI:

```text
node pipeline/guardrail_prevalence_audit.mjs --frame-dir PATH --calls-csv PATH
  --checks-csv PATH --repos-dir PATH --output-dir NEW_PATH
  [--independent-checks-csv PATH] [--annotations-csv PATH]
```

All listed mandatory paths are explicit. Output must not exist; reject before
reading inputs. CLI error prints a concise message and exits nonzero. Running
without annotations is valid and yields zero **source-reviewed identified**
guardrail methods with all methods lacking verified evidence, not 0% true
prevalence or a completed guardrail assessment. `--help` exits zero without IO.

## Inputs and schemas

`frameDir/method_assessment.csv` required columns:
`producer_method_id,repo,rel_path,qname,line,line_end,mapped_tests,exact_mapped_tests`.
Ignore initial placeholder assessment/oracle columns. Methods must have unique
producer IDs, positive inclusive start/end, consistent canonical `methodKey`.
Identical duplicates deduplicate; conflicting identities are fatal. Do not
hardcode 7,642; this is the current saved frame, not a constant.

`frameDir/mapped_test_edges.csv` required:
`repo,test_file,test_method,test_line,test_line_end,focal_file,focal_method,
focal_line,focal_line_end,file_rule,method_rule,method_key`.
Every mapping must resolve to an exact frame method by repo/file/start/end/name
(the method's final qualified component equals focal_method). `method_key`
must agree. Invalid edges are fatal, not silent denominator changes. A test ID is
`repo::normalized_test_file::test_method::test_line`; use this exact string in
annotations. Identical mapping duplicates deduplicate. Mapping uniqueness is
method ID plus test ID. Imported/fuzzy/basename rules remain evidence fields.

`frameDir/readiness.json` and `codebook.json` are consumed provenance/context;
copy explicit frame limitations/status into the report. Do not interpret their
old `current_percentages` values as annotations.

`callsCsv` required:
`call_id,repo,rel_path,code_kind,call_line`.
Only production-labelled rows can seed runtime candidates in frame methods.
Map a call to the innermost containing frame method in its repo/file (smallest
inclusive span; ambiguous equal spans fatal). Calls outside frame remain
diagnostics. Conflicting call IDs are fatal; identical rows deduplicate.

`checksCsv` required:
`call_id,repo,rel_path,code_kind,call_line,sink_line,check_kind,mechanism,
intervention,sink_source,joern_status,slice_ids,response_slice_ids`.
These are candidates only. Preserve source-only, weaker-return/continue and
slice-supported strata separately; do not classify oracle from mechanism.

Optional independent checks required columns:
`repo,invoker_file,invoker_line,test_file,test_line,test_qname,call_line,
check_line,failure_line,check_kind,evidence,call_source,check_source`.
Join exact invoker definition repo/file/start to a frame method, then exact
normalized test file/definition line/final test_qname component to that method's
focal mapping. No name-only or enclosing-definition substitution. Unjoined rows
become diagnostics. These are independent candidate evidence, not verified guards.

Annotation CSV required columns (empty/header-only valid):
`guardrail_id,arm,producer_method_id,test_id,output_call_id,decision,oracle_type,
checked_property,failure_action,source_association,source_evidence,reviewer,rationale`.
Optional fields copied through:
`slice_ids,joern_status,input_mode,oracle_mode,path_limitations,configuration,
duplicate_family,auxiliary,parent_guardrail_id`.

Enums:
- arm: `test_suite` or `runtime`.
- decision: `guardrail`, `not_guardrail`, `unresolved`.
- oracle_type: `simple`, `complex`, `human_only`, `unknown`.
- source_association: `verified`, `unresolved`.
- auxiliary: blank/`0` or `1`.

All annotations must target an existing method. Runtime test_id must be blank.
Test-suite test_id must match that method's saved mapping; unmatched discovered
tests must be reported separately by the analyst, not silently admitted here.
`output_call_id` may be blank if the reviewer explicitly associates a method's
returned output via evidence; when provided it must identify a canonical call
whose innermost frame method is the annotated producer. All guardrail decisions
require verified association, nonblank property/failure_action/reviewer/rationale,
and nonempty source_evidence. Unknown oracle may remain a verified guardrail with
unresolved type; human-only is tracked separately, never made complex.

`source_evidence` is a JSON array with one or more entries:
`{rel_path, line, line_end, sha256, role}`. Roles are `output`, `checker`,
`enforcement`, `acceptance`, `support`. Positive guardrails must provide at least
one output/checker/enforcement role each (entries may refer to the same file/line).
Line/end are positive inclusive integers, existing source line bounds; SHA-256
matches the whole original file. The tool verifies files under
`reposDir / repo.replace('/','_') / rel_path`; reject absolute/traversal/drive,
symlink/junction/resolution escapes before access. Pure aggregation validates
evidence schema; `runAudit` verifies bytes/line bounds. Do not try to prove the
human semantic judgments automatically.

Annotation identity is arm + producer_method_id + guardrail_id. Identical repeats
deduplicate; conflicting reviews are fatal for explicit adjudication. Several
guards can protect one method; one guardrail may have both simple and complex
atomic checks using separate IDs. Schema validation only packaging an evaluator
verdict is `auxiliary=1` and never inflates original-output simple counts.

## Worklists

`runtime_candidates.csv`: deduplicated candidate rows joined via canonical call
ID and innermost frame method. Preserve original check fields and add producer ID,
qname, candidate_id, and `review_status=unassessed`. Candidate ID must be stable
across input order, derived from method/call/sink/check_kind/source identity.
Candidate count must name source-only versus same-response-slice evidence.
Do not count test-labelled calls here. Checks with conflicting call repo/file/line
are fatal. Runtime-call checks outside frame produce diagnostics.

`test_suite_candidates.csv`: at least one placeholder row for every mapped edge,
with stable test ID and producer ID, focal rules, review_status unassessed; attach
joined independent checks as extra candidate rows (one per unique sink/target
call/check-kind evidence), or retain placeholder when none join. No mapped edge
disappears because no local check was detected. Do not count test-body LLM calls
as evidence they target the producer's output without source association.

Worklists are navigation aids. Their candidate counts never enter verified
guardrail numerators. Simple candidate counts and keyword complex-discovery
counts must not be compared as if equally ascertained prevalence.

## Aggregation and reporting

Every method remains in the denominator for each arm. For each arm emit:
- `simple_detected_only`, `complex_detected_only`, `both_detected` for methods
  with source-reviewed automated guardrails of those types.
- `unknown_type_only`, `human_only`, `no_verified_evidence` as mutually exclusive
  remaining classifications, prioritizing unknown_type_only over human_only when
  both are present. Keep independent has_unknown_type/has_human flags even where
  known simple/complex evidence also exists.
- Independent evidence flags/counts: verified annotation count, rejected count,
  unresolved count, auxiliary count, with mapped test, source/slice evidence.
- `no_verified_evidence` is not absence; methods with no annotations remain
  explicitly unreviewed. A rejected candidate does not establish exhaustive method
  assessment. Do not emit `no_guardrail`.

Verified numeric numerator requires decision=guardrail, verified association and
auxiliary!=1. Count methods once within a category. Per-arm summary has shared
`denominator`, category method counts, identified prevalence percentages over N,
any-simple/any-complex rates (overlap explicitly named), verified automated-method
count G (union of methods with known simple or complex guards; unknown-type-only
and human-only methods are reported outside G), composition percentages among G
for the three known categories,
unknown/human/unresolved/unreviewed counts, and static limitations. If N or G is
zero, its percentages are null. Unknown types are not removed from reported
missingness or relabelled simple. No sampling CI or independent validation claims
are calculated from these purposively reviewed annotations.

Per-method output columns include original method identity and both arm categories
and flags. Report original focal correspondence as a separate statistic, never
as output-related check prevalence. Totals/percentages use unique methods, not
tests, annotations, slice rows, or call sites. Empty annotations must clearly
report review pending and avoid a headline "0% have guardrails".

Outputs in fresh directory:
`runtime_candidates.csv`, `test_suite_candidates.csv`, `methods.csv`,
`verified_guardrails.csv`, `rejected_candidates.csv`, `unresolved_candidates.csv`,
`diagnostics.csv`, `summary.json`, `REPORT.md`, `manifest.json`. Headered empty CSVs
required. Stable deterministic ordering, UTF-8 CSV with quoted multiline support.
Report prevalence as **source-reviewed identified guardrail methods in this frame**,
not confirmed population prevalence. State assistant/source review and independent
human/detection validation status from annotations, not assumed two-reviewer proof.
Include checked-property/failure-action distributions in annotation units, clearly
distinguished from primary method units, and selected evidence references.

### Exact machine-readable summary and method fields

`summary.json` has `schema_version: "guardrail-prevalence-audit-v1"`,
`denominator` (unique method count), `mapped_methods`, `mapped_tests` (distinct
test IDs), `mapping_pct`, `review_status: "source_review_only_not_population_validation"`,
`arms: {runtime: ArmSummary, test_suite: ArmSummary}`, and `limitations` string array.
Additional provenance/diagnostic keys are allowed.

Each `ArmSummary` requires:
`denominator`, `category_counts` (all six categories named above, integer values),
`category_prevalence_pct` (same six keys, percentage of N),
`identified_automated_methods` (G), `composition_pct`
(`simple_detected_only,complex_detected_only,both_detected`, percentage of G),
`any_simple_methods`, `any_complex_methods`, `any_simple_prevalence_pct`,
`any_complex_prevalence_pct`, `unknown_type_methods`, `human_methods`,
`unresolved_methods`, `unreviewed_methods`, `verified_annotations`,
`rejected_annotations`, `unresolved_annotations`, `auxiliary_annotations`.
The unknown/human/unresolved flags are independent and may overlap known categories.
Unreviewed means no annotation for that arm/method (auxiliary-only rows do not
establish a reviewed guardrail, but do establish annotation presence). Percentages
are rounded to six decimal places, null when their denominator is zero.

Each output `methods.csv` row retains
`producer_method_id,repo,rel_path,qname,line,line_end` and adds, for each prefix
`runtime_` and `test_suite_`, `category,has_simple,has_complex,has_unknown_type,
has_human,has_unresolved,annotation_count,verified_count,rejected_count,
unresolved_count,auxiliary_count,unreviewed`. Boolean fields serialize 0/1;
counts serialize integers. `mapped_tests` and `exact_mapped_tests` are retained.
An unresolved annotation never increments verified_count. A verified guardrail
with oracle_type unknown increments verified_count and has_unknown_type, but not
G. Auxiliary rows never increment verified_count or has_simple/has_complex.

Manifest hashes each consumed input before/after, inspected source before/after,
all outputs excluding manifest; record Node version/options and script SHA-256.
Changed consumed input/source is fatal. Never overwrite an existing output.
Do not hash unrelated 1GB/corpus artifacts that this consumer does not read.

## Meaningful acceptance scenarios

Use temporary synthetic files; no real application/model/Joern execution.

1. Two-method frame, one mapped test. No annotations => both arm denominators 2,
   correspondence1, no verified evidence2, unreviewed2; composition null and no
   zero-true-prevalence claim. Existing placeholder assessment flags cannot count.
2. Runtime simple guard on method A, two slice rows and identical repeated
   annotation => A counts once. A separate complex guard on A makes both_detected1;
   B remains no_verified_evidence. Any-simple/any-complex overlap documented.
3. Test guardrail annotation must match method A's exact saved test edge. Same
   test name in another file/line/method cannot join. Fuzzy/basename mappings remain
   eligible but labelled; all mappings appear even without independent candidates.
4. A code-kind production call in an inner method is attributed to the inner,
   not enclosing method. Outside-frame calls produce diagnostics. Conflicting
   call identity or check source identity fails. Candidates alone never count.
5. A schema guard on original output is simple. A schema merely packaging a
   semantic verdict has auxiliary1 and does not add simple. A source-verified
   human-only guard remains human_only, not complex; unknown oracle remains unknown.
6. Unresolved association, rejected shell/blockchain candidate, missing/type-unknown
   annotations remain separate. Conflicting duplicate annotation fails rather than
   choosing most recent. Empty N => null percentages, valid headered outputs.
7. Positive source hash mismatch, unsafe source path/link, missing role evidence,
   invalid line bounds, or absent required semantic fields fails actionably. Same
   source path with Unicode, spaces, quotes and multiline CSV rationale works.
8. Valid synthetic run verifies before/after hashes and all output hashes. Input
   order and identical duplicate rows do not change semantic totals/worklist IDs.
   Output exists refusal precedes input IO. Unknown CLI flag fails, help does no IO.

No corpus size constants, no fallback to Python3.12, no background long jobs,
no data deletion. Tests must exercise actual joins/aggregation/evidence boundaries,
not implementation mirrors. Parent will supply source-reviewed annotation data;
unreviewed candidates remain honest unresolved work.
