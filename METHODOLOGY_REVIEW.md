# Methodological review — parked

**Scope as of 2026-09-01:** framework calls only (raw provider SDKs retained but not
reported), and the call graph reduced to the blast-radius measure. Items below are marked
where those decisions changed them.

Findings from the 2026-08-31/09-01 code review, held for the review pass. `METHODOLOGY.md`
states what was done; this file is what's wrong with it, what's undocumented, and what the
method cannot see. Nothing here is settled — it's the agenda for that conversation.

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
