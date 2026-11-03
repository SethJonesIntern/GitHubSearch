# Methodological review — parked

**Scope as of 2026-09-01:** framework calls only (raw provider SDKs retained but not
reported), and the call graph reduced to the blast-radius measure. Items below are marked
where those decisions changed them.

Findings from the 2026-08-31/09-01 code review, held for the review pass. `METHODOLOGY.md`
states what was done; this file is what's wrong with it, what's undocumented, and what the
method cannot see. Nothing here is settled — it's the agenda for that conversation.

## J. Guardrail study: figure review (2026-11-03)

The current question is whether production application methods containing framework
LLM invocations have associated correctness checks in tests or during execution.
The baseline must count methods with at least one mapped test / eligible methods;
identifying a guardrail then requires evidence of the checked property, and runtime
checks additionally require an intervention before acceptance or action.

This is an initial review of figure measures and generator code, not a new analysis
of the missing source/slice corpus. Existing outputs remain historical artifacts.

| Existing figure or output | Proposed place in the new story | Reason / next action |
|---|---|---|
| F1 — evaluation-framework adoption | Supporting detector evidence; redesign before a guardrail claim | A library call anywhere in an app does not establish a correctness check associated with a particular LLM-containing method. Separate test/runtime context, checked property and intervention; include checks implemented without those libraries. |
| F2 — determinism parameters | Retire from the main figure set | Temperature, seed and sampling settings describe generation configuration. Their presence or absence does not establish a correctness guardrail. Retain historical data/code pending dependency review. |
| F3 — frameworks called | Population context, potentially a table or appendix | Describes the sampled ecosystems and detector scope; does not answer the guardrail question. |
| F4 — tests on LLM-containing methods / call sites | Replace with a production-method correspondence figure | Current chart mixes test counts with call-site counts and its title suggests testedness. Use methods with >=1 mapped test / eligible production methods, alongside the other-method baseline. Correspondence is only the first step toward checking correctness assertions. |
| F5 — tool mapping rate versus LLM-method mapping rate | Replace as a main result; retain tool diagnostics separately | Its 14.12% denominator is tests, while 1.38% uses methods; comparing them cannot establish a method-level testing gap. The generator's old claim that the gap is coverage rather than tooling is unsupported. Its method population also includes test/support code. Use the new baseline's paired repositories and report mapping limitations. |
| `07_knob_count.png`, `knob_count.csv` | Retire from the main figure set | Same generation-configuration issue as F2. |
| `framework_coverage.png`, keep-frequency plots | Sampling/methodology context | Explain discovery and exclusions, with the unit named explicitly. Avoid conflating framework-population coverage with test or guardrail coverage. |
| Jump-depth / blast-radius reports | Methodology diagnostics unless needed for guardrail association | Static reachability alone establishes neither a check nor intervention. Historical transitive-prevalence restrictions remain in force. |

Candidate replacement sequence, with results still pending:

1. Population flow: discovered frameworks → import-search candidates → verified
   applications → production methods with observed framework LLM calls. Use separate
   units at each stage; do not turn mixed units into a common percentage.
2. Test correspondence: fraction of LLM-containing production methods with >=1 mapped
   test versus other production methods in the same repos. Report pooled and paired
   repo rates, exact-only sensitivity and excluded/missing data from
   `pipeline.method_test_baseline`. No percentages until the PC run is reviewed.
3. Test-suite guardrails: among eligible LLM-containing methods, which have an associated
   correctness check? Record the property checked and mapping/inspection evidence.
4. Runtime guardrails: which methods have associated checks with intervention, including
   the checked property, placement and failure response? Detection and validation pending.
5. Combined view: test-suite only / runtime only / both / no identified guardrail,
   with unassessed cases reported separately. Use one common method denominator and
   avoid interpreting no identified guardrail as proven absence.

Before deleting older code, inventory entry points, imports, generated outputs and
reproducibility dependencies. Preserve raw artifacts, the vendored focal mapper,
discovery, scope filters, call extraction, the new baseline and slice tooling until
their role in guardrail detection is established. No deletion is made by this review.

## A. Claims in the code that are false

1. **`search_candidates.py:5`** says the code search matches an exact substring "so a repo
   only matches if it genuinely imports the framework." It does not. Quoted queries still
   returned repositories that never import the name (`clai` → claim/disclaimer, `camel` →
   camelCase). Justify from the clone-side evidence, not from the operator.
2. **`graph_coverage_pct`** is `pyan_nodes / total_functions`, median **110.4%** — pyan
   counts nodes `total_functions` never did. Not a coverage fraction.
3. **`batch_call_metadata`** logs `len(FRAMEWORK_CALLS)` (76) while matching
   `SCOPED_FRAMEWORK_CALLS` (44). Cosmetic.

## B. Definitions that don't line up across stages

4. **Test file.** Inclusion filter (Stages 1–2) is `test_*.py` only; the ND-test detector
   accepts `test_*.py` **or** `*_test.py`. Repos using the second convention were dropped as
   untested — 2,043 went out on that gate. Share of false drops unmeasured.
5. **Contributors.** Stage 1 = GitHub contributors API (anon included); Stage 2 = distinct
   author emails from `git log`. Same ≥ 2 threshold, different construct.
6. **`frameworks_imported`** is a name scan, not resolution — a repo's own `agents/` package
   satisfies it. Relevant wherever it was used as adjudication evidence.

## C. Undocumented choices

7. Where `pushed_at > 2025-04-14` came from.
8. Why stars ≥ 10 at Stage 2 and ≥ 1000 at Stage 1.
9. Why 3 result pages at Stage 1.
10. The ≥ 4-character floor in the import-name classifier.
11. Whether the three hand-curated name lists were reviewed by anyone else.

## D. Sampling limits

12. **The framework frame is effectively one phrase** — 51 of 59 frameworks came from
    "AI agent framework"; two phrases contributed one each.
13. **Stars ≥ 1000 and `language:Python` are query qualifiers**, so they shape the frame
    without appearing in any funnel.
14. **Code search caps at 1,000 results**, and unlike Stage 1 it was binding for common
    names — for those, the sample is GitHub's relevance ranking, not a census.
15. **8 of 59 frameworks lost every import name** and are unrecoverable by import-pattern
    search: openai-agents-python, agent-zero, LaVague, giskard-oss, solace-agent-mesh,
    cheshire-cat, redamon, AssetOpsBench.
16. **273 of 539 names were dropped `unclassified`** — a default-drop, not a review.
17. **Stage-2 clone failures are recorded as `dropped_contributors`** — an unknown share of
    2,436 are network failures, not low-quality repos.

## E. Detection limits

18. Nested functions and inner-class methods are not indexed.
19. Text matching on the unparsed callable — a call on a variable receiver matches by method
    name alone; the FP tiers mitigate but cannot resolve types.
20. Out-of-process invocation (CLI subprocess, HTTP) is invisible; handled by exempting
    omnigent/agentops rather than reporting a false zero.
21. **Raw Gemini is in no dictionary** — `google.generativeai` / `google.genai`, ~254 repos
    (24%). **Largely dissolved** by the framework-only scope decision: SDK call sites are no
    longer reported, so a missing SDK cannot affect a headline number. It still undercounts
    the "imports a framework, calls the model directly" finding, since Gemini-only bypassers
    are invisible — size it only if that finding is reported.
22. **114 of 174 trustworthy names have no call patterns**, including 35 `agent_framework_*`
    companion packages against 4 alias entries. Checked: 62 live repos matched only such a
    name, but only 11 have zero calls and those import langchain/agents — so this is an
    attribution gap, not the cause of the zero-call set.

## F. Aggregation

23. **Transitive invokers inflate with graph size** (21,824 direct vs 590,324 transitive)
    and are structurally 0 for the 75 repos without a graph. **Demoted**: the call graph now
    supports only the blast-radius measure, and all headline figures are direct.
24. **`analyze.py` filters direct rows by the current pattern set but leaves transitive rows
    unfiltered** — a removed seed's closure still counts.
25. **`clai` → `pydantic_ai` is still live** in `keep_frequency.EXTRA_MEMBERS`, and reaches
    `analyze.py` through `group_of`, so it contaminates the grouped call and ND-test tables
    too. 16 non-cut repos carry the token; none imports `clai`; only potpie independently
    matches pydantic_ai. Credits pydantic_ai with ~15 apps that don't import it.
26. **Eval seeds are self-declared provisional** — never validated the way invoker seeds
    were.

## G. Adjudication

27. **Single coder, no second rater, no agreement statistic** across 302 adjudicated repos.
    Consider double-coding a random sample.
28. Two places hold cuts: the audit sheet's `in_scope`, and hard-coded sets in `analyze.py`
    (`QUALITY_EXCLUDED`, `NOT_LLM_APP`).

## H. Reproducibility

29. **The import-name classifier is reconstructed at 539/539** (2026-09-01) but not yet
    committed as code, and three of its five rules are hand lists recovered from the sheet
    itself. Commit the classifier; publish the lists.
30. Code search is not reproducible over time — the artifact CSVs are the record. State
    collection dates.

## I. Slice analysis (2026-09-21)

31. **`relation` in `slice_call_hits.csv` cannot separate prompt from response.** The
    slicing criterion is the variable's last use, so a prompt argument's criterion is the
    call line itself. The `role` column (receiver/argument/binding, read off `callable` and
    `call_source`) carries that distinction instead. `role` has face validity from the
    variable names alone — `agent`/`graph`/`llm` as receivers, `prompt`/`messages` as
    arguments, `result`/`response` as bindings — but has had **no hand-checked precision
    sample**.
32. **The slice→call join is per-line, so a multi-line call expression loses its binding.**
    `response = cast("AIMessage", await model.ainvoke(...))` records the call two lines
    below the assignment. 10.6% of no-binding call sites sit on a multi-line statement
    against 0.4% of binding ones; 529 are `Assign`/`AnnAssign`. Fix: join on the enclosing
    statement span rather than the call line.
33. **"No binding-role slice" is not "the LLM result is unused."** Of the 6,673 in-slice
    call sites with none: 61.0% bare expression statement, 17.2% inside an `assert`, 13.1%
    `return`, 7.9% assigned (item 32). At most **4,068 (18.1% of in-slice call sites)** are
    candidates for a discarded result. Split by `code_kind`: **3,248 in test functions**,
    84 in test support, **736 in production** — an LLM called in a test with the result
    neither bound nor asserted, and a smaller production set that calls and drops. (The
    first write-up said 3,320 test / 748 non-test from an ad-hoc path check; superseded.)
    Before either is a finding it needs: a hand-checked sample; deduction of side-effecting patterns
    (`.publish_message` 161, `.add` 127, `.stream`/`.run_stream` 99); and deduction of
    detector false positives — two of four hand-read samples were not LLM calls at all
    (`create_crm_task.invoke`, `send_fn.invoke`, both caught by the bare `.invoke` pattern).
34. **Slice hits inherit the haystack `.run` problem** (see CLAUDE.md): 616 of the
    discarded-result bucket is haystack, and `retriever` (446) and `component` (351) appear
    among the receivers. The slice tables make that contamination countable per receiver
    name for the first time, but do not correct it.
35. **The slice index spans 788 repo directories; the observed scope is 644**, and Stage 6
    was fed the whole invoker closure, transitive included (`EXCLUSIONS.md` §18). Stage 6b
    filters to POP on the call side only; the index does not filter at all. This is the
    same open question as "Stage 6 slicing vs the direct-only rule".
36. **35.1% of the 51,089 slices touching a call have a body that does not parse**
    (17,913). Corrected 2026-09-22: the first write-up divided that slice count by the
    58,478 slice-call *pairs* and said 30.6%, and did the same for test paths (said 38%).
    The summary JSON now names slice and pair counts separately. `parses` is a column, not
    a defect in the join — but anything that reads slice *bodies* rather than their line
    sets has to filter on it first.

