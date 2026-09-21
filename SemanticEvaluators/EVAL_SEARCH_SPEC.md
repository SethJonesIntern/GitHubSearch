# Eval framework search — spec

Implements GitHub issue #3, "Find List Of Missing Eval Frameworks".

## Goal

A candidate list of LLM evaluation frameworks we may be missing. **Discovery only.**
Seth triages the list afterward and decides what, if anything, gets adopted.

Context: `EVAL_CALLS` (pipeline/eval_calls.py) seeds five evaluators — deepeval, ragas,
giskard, opik, phoenix — and `analysis/eval_app_prevalence.csv` shows any-eval usage at
40 of 753 apps (5.3%). The question is whether that floor is real or an artifact of a
short seed list.

## Out of scope — do not touch

- No edits to `EVAL_CALLS`.
- No pipeline re-runs; no stage is re-executed.
- No `in_scope` edits to `application_audit.csv`; no statistic moves.
- In-house eval modules shipped by agentic frameworks (`langchain.evaluation`,
  `llama_index.evaluation`, etc.) are a **separate issue**, not this one.

## Deliverables

1. `SemanticEvaluators/eval_framework_search.py`
2. `SemanticEvaluators/test_eval_framework_search.py`
3. Two `pipeline/paths.py` entries: `EVAL_FRAMEWORKS_CSV`,
   `EVAL_FRAMEWORKS_FILTER_STATS_JSON`
4. Waterfall figure + backing CSV in `pipeline/artifacts/figures/`

Reuse `Frameworks/GithubSearch.py` helpers by import — `github_get`,
`search_repositories`, `enrich_repo`, `derive_import_names`, `get_contributor_count`,
`get_test_metrics`. Do not copy them. Phase B must emit the same `FIELDNAMES` as
`frameworks.csv` so the two tables are row-comparable.

## Search phrases (fixed — do not add or reword)

1. LLM evaluation framework
2. LLM eval framework
3. evaluating LLM applications
4. LLM testing framework
5. LLM-as-a-judge
6. LLM benchmarking framework
7. LLM observability evaluation
8. guardrails evaluation LLM

## Phase A — `--search-only`

- **No `stars:` qualifier. No `language:` qualifier.** Both are post-hoc funnel steps.
- Pre-flight: run each phrase at `per_page=1` and report `total_count` before paging
  anything. 8 requests. A wildly broad phrase gets cut by Seth at this point, not after.
- Page to GitHub's 1000-result ceiling (10 x 100). ~80 requests total, ~3 min.
- Record retrieved-count vs reported `total_count` per phrase. A phrase whose
  `total_count` exceeds 1000 is truncated and relevance-sorted — say so in the output,
  do not imply full coverage.
- Dedupe across phrases; keep all `matched_queries` for a repo.
- Write `eval_framework_candidates.csv` from free search metadata only (no enrichment):
  full_name, description, stars, language, topics, archived, fork, pushed_at,
  plus `matched_queries`, `already_seeded`, `in_frameworks_csv` (already a Stage-1
  agentic framework), and an empty `confirmed` column (see triage below).

`already_seeded` must NOT be computed from import names in Phase A. Import names come
from `derive_import_names`, which needs the repo tree - an API call, i.e. enrichment.
In Phase A match on `full_name` against a hand map of the five seeded evaluators'
repos (confident-ai/deepeval, explodinggradients/ragas, Giskard-AI/giskard,
comet-ml/opik, Arize-ai/phoenix). Phase B recomputes it properly from `import_names`.

## Funnel + waterfall

Post-hoc, sequential attribution — the first step that rejects a repo owns the drop,
matching `apply_filters` in `GithubSearch.py`:

```
search hits retrieved
  -> dedupe across phrases
  -> not fork / archived / disabled
  -> stars >= 10
  -> language == Python
  ---- enrichment boundary ----
  -> contributors >= 2
  -> test files >= 1
  -> confirmed eval framework (hand triage)
```

**`confirmed` is not a heuristic.** The script writes the column empty; Seth fills it
(`1` / `0`). The final bar is computed from whatever is in it, and renders as pending
while the column is blank. Do not invent a keyword or topic rule to auto-populate it -
that would be the script answering the question the study is asking.

Stars is deliberately first: it is the filter under interrogation. Everything above the
boundary is free; everything below costs ~4 API calls per surviving repo.

**The waterfall spans both phases.** Contributors and test-file counts sit below the
enrichment boundary, so those bars cannot be drawn until Phase B has run - which happens
after triage. Render the figure at any point: bars above the boundary are final, bars
below it draw as pending until the data exists. Do not block Phase A on them, and do not
fabricate them.

Figure: `eval_search_waterfall.png` + backing `.csv` written by this script into
`pipeline/artifacts/figures/`. Do **not** wire it into `Applications/make_figures.py` —
that reads `cuts.py` and produces the paper's F1-F5. This is not a paper figure yet.

## Phase B — `--enrich`

Full Stage-1 enrichment over triaged survivors only (expect ~50-100 repos), writing
`eval_frameworks.csv`.

## Tests

Mirror `Frameworks/test_github_search.py`. Cover at minimum: query construction (and
that no stars/language qualifier is emitted), dedupe-across-phrases preserving
`matched_queries`, sequential attribution (drop counts sum to input; a repo failing two
criteria is attributed to the earlier one), each filter predicate at its boundary
(stars 9/10/11, contributors 1/2, test files 0/1), and the truncation flag when
`total_count > 1000`.

## Conventions

- `py -3.14` always.
- GitHub token lives in `Frameworks/.env`, loaded via `load_dotenv`.
- Seth runs the search himself; do not background a long job.
- Distinguish measured / projected / assumed in any reported number.

## Acceptance

`py -3.14 -m SemanticEvaluators.eval_framework_search --search-only` produces the
candidate CSV and the per-phrase pre-flight table; the waterfall renders; tests pass.

**Sanity check on first run:** if the 8 phrases do not surface all five currently-seeded
evaluators, the phrase set has a hole. `already_seeded` makes this a one-glance check.
