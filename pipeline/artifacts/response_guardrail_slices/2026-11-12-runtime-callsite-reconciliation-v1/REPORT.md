# Runtime guardrails at call-site level: reconciliation (2026-11-12)

**Status: preliminary.** Every semantic label here is an assistant source classification
(blind review by eight subagents, then adjudication by the primary session). No independent
human validation has occurred. Joins, counts, percentages and hashes are deterministic.
`manifest.json` records input hashes (all unchanged), rules and provenance. `counts.json` is
the machine-readable source for every number below.

**Unit.** One unique frozen `call_id` from `2026-11-04-v1/calls.csv`: a detected direct
framework LLM call site with saved `code_kind == production`.

**Not used.** No pyan evidence was read: no call graph, invoker closure, transitive
reachability or pyan-derived test relationship. No application, model, discovery, batch
analysis, focal mapping or Joern producer was executed. Saved Joern slices serve only as
co-membership corroboration. The Stage-6 slicer had chosen which functions to slice from
the invoker closure; that is provenance, not evidence used here.

## 1. Presentation counts (recommended)

**Denominator:** all 10,922 production-labelled detected call sites. It is **not corrected**
for false-positive invocations, mislabelled test code or vendored framework copies, because
no consistent correction exists across all 10,922. The rate is an *identified rate*, not a
corrected prevalence.

**Numerator:** distinct protected producer call sites with at least one source-confirmed
runtime guardrail. The producer must be a genuine in-scope framework LLM call in production
context. Discovery is uniform: all 205 production-labelled sites with any November 4 static
check candidate were source-reviewed.

### Pie 1: detected runtime guardrail

| Slice | Call sites | % of 10,922 |
|---|---:|---:|
| Source-confirmed runtime guardrail | 93 | 0.851492 |
| Unresolved candidate | 2 | 0.018312 |
| No detected guardrail | 10,827 | 99.130196 |

Two-slice version: 93 confirmed (0.851492%) and 10,829 with no confirmed detection (99.148508%).

### Pie 2: composition of the same 93

| Class | Call sites | % of 93 |
|---|---:|---:|
| Simple only | 86 | 92.473118 |
| Complex only | 6 | 6.451613 |
| Both | 1 | 1.075269 |
| Unresolved oracle | 0 | 0 |

### Sensitivities, clearly labelled

| Variant | Numerator | Denominator | % | Simple / complex / both |
|---|---:|---:|---:|---|
| Recommended (uniform discovery) | 93 | 10,922 | 0.851492 | 86 / 6 / 1 |
| November 4 test-context flag removed from **both** sides | 93 | 10,765 | 0.863911 | 86 / 6 / 1 |
| Plus purposive judge-loop search additions | 105 | 10,922 | 0.961362 | 89 / 14 / 2 |
| **Historical, unchanged:** November 4 static candidate tier | 106 | 10,922 | 0.970518 | not applicable |

- No confirmed producer carries the deterministic test-context flag. That sensitivity
  therefore changes only the denominator.
- The purposive search targeted LLM-judge loops. Its complex share is not comparable with
  uniform discovery and must not be read as population composition.
- The historical 106 counts static check candidates, not guardrails. Its 47 assertion /
  52 rejection-branch / 7 schema split describes syntax, not simple/complex class.

### Flags inside the 93 (numerator-only, not applied to the denominator)

| Flag | Call sites |
|---|---:|
| Vendored framework copy inside an app repo (CAMEL, AutoGen, Agent Framework) | 34 |
| Example, tutorial or demo code | 12 |
| Call and check share an original-line saved Joern slice | 59 |
| Every qualifying check is a Python `assert` (disabled under `-O`) | 22 |
| Configuration-dependent | 6 |
| Retry-budget bypass (accepts output after N failures) | 3 |
| Fail-open on checker error | 1 |

The vendored share is large. The denominator also contains vendored code at an unknown
rate, so these flags cannot be turned into a corrected rate. A consistent sensitivity would
have to exclude whole vendored subtrees from both sides. That sensitivity is **not yet done**.

## 2. Frozen-set reconciliation: the November 4 106

### A. Exact reconstruction (deterministic)

A frozen site is a production-labelled `call_id` with at least one November 4 check row whose
`check_kind` is `assertion`, `rejection_branch` or `schema_validation` and whose
`joern_status == same_response_slice`. All seven columns of `2026-11-04-candidate-tiers.csv`
reproduce exactly:

| Column | Value |
|---|---:|
| Production-labelled call sites | 10,922 |
| Any check candidate | 205 |
| Explicit check candidate | 175 |
| Explicit and slice-supported | 106 |
| Explicit, outside obvious test context | 150 |
| Explicit and slice-supported, outside obvious test context | 84 |
| Weaker-only candidates | 30 |

Every one of the 106 has exactly one qualifying mechanism, so the split needs no precedence
rule: assertion 47, rejection branch 52, schema validation 7. Each site is counted once.

### B. Outcome of source review for all 106

| Outcome | Sites |
|---|---:|
| Confirmed guardrail on this call (53 simple only, 1 both) | 54 |
| Not an LLM call (false-positive invocation) | 23 |
| Test or test-support code despite production label | 15 |
| Check does not qualify (workflow branch, parse only, after acceptance, etc.) | 9 |
| Judge call; its guard protects an earlier generation and is credited there | 4 |
| Unresolved | 1 |

When a site is both test code and a false-positive call, it is counted once, as a false positive.

| Mechanism | Confirmed | False positive | Test context | Nonqualifying | Judge | Unresolved |
|---|---:|---:|---:|---:|---:|---:|
| Assertion (47) | 22 | 7 | 15 | 1 | 2 | 0 |
| Rejection branch (52) | 27 | 16 | 0 | 7 | 1 | 1 |
| Schema validation (7) | 5 | 0 | 0 | 1 | 1 | 0 |

The November 4 test-context flag covered 22 sites. Review found 18 of those were really test
code and 4 were production code. Of the 84 unflagged sites, review found 2 that were test
support. Reviewed context totals: production 86, test 18, test support 2.

## 3. Why "more than 20% were not LLM calls" does not generalise

- **The 106:** 23 of 106 (21.7%) are false-positive invocations. Repos involved: Tlamatini
  `.invoke` 5 (unit tests driving fake models), Weaver sandbox `.run` 5, intentkit web3
  `.call` 4 and `.generate` 1, codemie `.run` 2, fast-langgraph `.invoke` 2, agent_project
  `.call` 2, local-deep-research `.invoke` 1, Agent-Q-Mix `MessageFactory.create` 1.
- **All 205 uniform candidates:** 77 (37.6%) are false positives. Weaver's E2B
  `sandbox.commands.run` accounts for 34 and intentkit's web3 contract `.call` for 15.
  Others include OpenSearch `.search` in vendored mem0 copies, raw provider SDK calls
  (DashScope, an OpenAI-compatible image API) matched by token, and LangGraph runs with no model.
- **Selection bias:** a site became a candidate because a check sits on its result. Shell
  commands and contract reads have their results checked far more often than LLM calls do.
  These rates describe the detector on checked sites, **not** the false-positive rate of the
  10,922 inventory.
- **What is needed:** a probability sample of the 10,922 to estimate the denominator's
  false-positive, test-mislabel and vendored rates. Until then the denominator stays
  uncorrected.

## 4. Why 106 call sites differ from 86 methods

1. **Different unit.** The 2026-11-12 result counts production methods (7,642 in 571 repos).
   This report counts call sites (10,922).
2. **Different discovery.** The 106 are static candidates and about half are not guardrails.
   The 86 methods also drew on the 99 non-106 November 4 candidates (95 "extra" annotations)
   and on 14 purposive judge-loop annotations.
3. **Different frame.** 14 of the 106 lie outside the 7,642-method frame (55 baseline repos
   were excluded), so they were never annotated on 2026-11-12.

The 2026-11-12 percentages must not be applied to the 106.

## 5. Crosswalk with the 2026-11-12 runtime annotations

`crosswalk_nov12_runtime.csv` maps each of the 201 runtime annotation rows to producer call IDs.

- The one annotation without an `output_call_id` (`runtime-extra-044`) resolves from its own
  evidence to two producers: `_code_executor_agent.py::808::31` and `::821::33`.
- That gives **87 producer call IDs for the 86 verified methods.** 44 are in the frozen 106,
  29 are in the other 99 uniform candidates, and 14 lie outside the 205.
- **86 of 87 are still confirmed.** The exception is
  `julienlucas_agentic-rag-multiagent/.../contextual_compression.py::76::23`. It is ruled a
  relevance filter: a code comment defines its 50-character minimum as a relevance threshold,
  and "AUCUN" is the model's own verdict. This is a borderline call.
- **19 producers are newly confirmed.** Most lie outside the 2026-11-12 method frame (the
  dslmodel/dspygen trigger modules, AutoData, Agent-Q-Mix orchestrator copies) or are producers
  reached through judge attribution (the kodo coach, the InternAgent CAMEL CoT generators,
  dspygen `json_module`, gigaevo `base.py`).
- **Judge attribution:** a judge's verdict that triggers intervention protects the generation
  it judges, and is credited to that generation's call ID. A type or schema check only on the
  judge's verdict is auxiliary and is not counted. Examples: site 191 to 219 (temporalio outline
  gate), 188 to 218, 202 to 201 (AnimoCerebro, which becomes "both"), 121 to the kodo coach
  (fail-open), and 24 to three InternAgent CoT generators.
- Of the 93 primary producers, 54 are in the 106, 31 are in the other 99 uniform candidates,
  and 8 were reached by attribution outside the 205.

## 6. Adjudications by the primary session

Recorded in `assistant_review/adjudication.json`, with the blind values preserved per site.

| Site | Ruling |
|---|---|
| 50, 51 (Tlamatini intent classifiers) | Parse only: substring label parsing with a default branch. Not a guardrail. |
| 13 (DALL-E through LangChain) | Unresolved: an image model, not a text LLM; scope not settled. |
| 3 (DashScope) | Out-of-scope raw SDK call matched by token; counted as a false positive, consistent with sites 87, 204 and 205. |
| 124 | Blind "not a guardrail" kept over the 2026-11-12 "simple". Borderline. |
| 113, 212 | Flagged as tautological type asserts; still simple under the codebook. |
| Six producers outside the review set | Identity checked in source: all genuine production LLM calls. The three InternAgent producers are in a vendored CAMEL copy. |

The two remaining unresolved candidates are site 13 and a CAMEL `role_playing.py::513` critic selection.

## 7. Files

| File | Contents |
|---|---|
| `counts.json` | Every count and percentage: historical, reconstruction, 106 outcomes, pies, sensitivities, crosswalk summary |
| `manifest.json` | Input paths with sha256 before and after (all unchanged), original source-file hashes, rules, provenance split, git revision |
| `frozen_106_reconciliation.csv` | 106 rows: mechanism, sink lines, slice IDs, flag, reviewed context, invocation, outcome, class, attributions, 2026-11-12 label, rationale |
| `callsite_audit.csv` | All 219 reviewed sites plus 6 attributed producers: per-site identity, tier, decision, class, counted-in flags |
| `guard_evidence.csv` | One row per examined check: protected producer, property, oracle, verdict, check and enforcement locations, consequence, before-acceptance, auxiliary, Joern support and slice IDs, retry/fail-open/config/assert flags, evidence lines |
| `protected_producers.csv` | One row per producer with a qualifying guard, including ineligible ones, with discovery stratum and class |
| `crosswalk_nov12_runtime.csv` | The 201 runtime annotation rows mapped to producers, with old and new status and agreement |
| `assistant_review/` | The protocol, eight blind review batches, adjudication, batch plan and packet index (assistant classifications) |
| `scripts/` | `build_packets.py` (blind evidence packets; reads saved data and clones) and `consolidate.py` (deterministic consolidation; refuses an existing output directory) |

Source hashes: all 219 reviewed files match the November 4 `inspected_sources` hashes. The
evidence packets themselves are not shipped. Rebuild them on the PC with `build_packets.py`,
which needs the clones and the saved slice files.

## 8. Open items

- No human precision sample exists, and no probability sample of negatives was drawn.
- The denominator is uncorrected for false-positive invocations, test mislabels and vendored
  copies.
- A vendored-subtree sensitivity, excluding whole subtrees from both sides, is not yet done.
- Scope decisions are pending for image-generation models and vendored framework copies.
- Static presence does not establish execution, enablement, all-path protection or efficacy.
  Assert-only guards vanish under `-O`.
