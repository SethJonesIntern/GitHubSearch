# GitHubSearch Pipeline

End-to-end plan for going from *"which agent frameworks exist on GitHub"* to
*"per-argument metadata for every LLM call site across applications that use
those frameworks"* — the input JOERN consumes to produce forward and backward
slices.

The pipeline is seven stages (Stage 5 LLM-call extraction and Stage 7 semantic
evaluation are parallel branches off the same application list). It is **built
and wired** — run it with `python -m pipeline run`. The **CURRENT STATE** section
just below is the authoritative, up-to-date summary; the per-stage sections after
it describe each stage as-built.

---

## ⏩ CURRENT STATE — resume here (updated 2026-06-19)

**Single entry point (orchestrator):** [pipeline/run.py](pipeline/run.py) via
[pipeline/__main__.py](pipeline/__main__.py). Runs each stage as a subprocess in
order (subprocess, not import, to dodge duplicate module basenames).

```
python -m pipeline run            # whole pipeline, in order
python -m pipeline run --smoke    # cheap end-to-end (small --limit/--max flags per stage)
python -m pipeline run --list     # list stages
python -m pipeline run --from applications   # this stage onward
python -m pipeline run --dry-run  # print commands only
```

**Interpreter:** use **Python 3.14** for real runs — the transitive test closure
(Stage 5c) needs pyan3, which requires 3.14. The dev box here is 3.9, so the 9
`*_is_transitive` Wrapper tests fail locally (expected); everything else passes.
GitHub token is read from [Frameworks/.env](Frameworks/.env).

**What's DONE (implemented, tested, smoke-verified):**
- **Stage 1** [Frameworks/GithubSearch.py](Frameworks/GithubSearch.py) — to Hasan spec
  (phrases + `stars:>=1000 language:Python`; filters not-archived / contributors≥2
  / test-files≥1). Rich metadata (subscribers/network/owner/feature-flags/has_ci,
  full repo GET). **`import_names` column** — importable package name(s) derived
  from the repo's `__init__.py` structure (`derive_import_names`), which Stage 2
  consumes. Filter funnel sidecar. `--limit/--max-pages/--out`.
- **Stage 2** [Applications/search_candidates.py](Applications/search_candidates.py) — to spec
  (code-search by import pattern + dedup; Python *primary* language enforced;
  not fork/archived/disabled; stars≥10; pushed>2025-04-14; lifetime≥30d;
  contributors≥2; commits/mo≥2; tests≥1). **Import patterns are now DERIVED from
  Stage 1's `import_names`** (`import_patterns()` builds `from X import` / `from
  X.` / `import X` per name) — the hand-curated `FRAMEWORK_IMPORT_PATTERNS` /
  `FRAMEWORK_SEARCH_TERMS` / drift notice are GONE; Stage 1 drives Stage 2.
  `matched_frameworks` now records import names. `application_metadata.csv` (rich,
  all enriched repos, `is_candidate`/`drop_reason`). Filter funnel. Writes a
  header-only `applications.csv` up front (chain doesn't crash on 0 candidates).
  `--max-terms/--code-pages/--max-repos`.
- **Stage 5+7** [pipeline/batch_call_metadata.py](pipeline/batch_call_metadata.py) — clone →
  index once → LLM pass (`FRAMEWORK_CALLS`) + eval pass (`EVAL_CALLS`) on the same
  parse → write 7 repo-tagged CSVs → delete clone. Resumable, per-repo error
  isolation, best-effort call graph. `--resume/--limit/--keep-clones`.
  **Invoker search is a first-class output:** `transitive_closure` runs on both
  LLM and eval seeds → `llm_invokers_all.csv` / `eval_invokers_all.csv` (every
  direct+transitive invoker function: `repo,qname,file,line,reason,kind`).
  `llm_tests_all.csv` is the pytest subset of the LLM invokers. Under Python 3.9
  the closure = direct seeds only; transitive rows appear under 3.14 + pyan3.
- **Engines** parameterized for the dual passes:
  [Wrapper/transitive_invokers.py](Wrapper/transitive_invokers.py) (`index_repo`,
  `seed_invokers`) and [Wrapper/call_metadata.py](Wrapper/call_metadata.py)
  (`active_matchers`, `collect_rows`) take an optional `framework_calls` dict
  (default = `FRAMEWORK_CALLS`, so old callers/tests unaffected).
- **Stage 3 framework frequency** [Applications/framework_distribution.py](Applications/framework_distribution.py)
  — repointed; reads the Stage 2 search progress + `applications.csv`, prints the
  three views (imported / candidate / kept) and writes `framework_frequency.csv`
  (`framework, repos_imported, repos_candidate, repos_kept, pct_of_kept`, keyed by
  import name). Wired as the `framework_frequency` stage.
- **Stage 7 eval frequency** [pipeline/eval_frequency.py](pipeline/eval_frequency.py)
  — derived purely from the eval-call piggyback (`eval_calls_all.csv`): per
  evaluator, `repos_with_calls`, `total_call_sites`, `pct_of_apps` →
  `eval_frequency.csv`. No dependency-declaration check needed — the AST "called"
  view *is* the signal. Wired as the `eval_frequency` orchestrator stage.
- **Orchestrator** [pipeline/run.py](pipeline/run.py) — 5 stages now:
  `frameworks → applications → framework_frequency → analysis → eval_frequency`.
- **Decision: dropped the dependency-file check from the pipeline.** The eval
  analysis already piggybacks on the invoker search (`EVAL_CALLS` →
  eval_calls/invokers/metadata in the `analysis` stage), which is the precise
  "called" view. [SemanticEvaluators/find_semantic_eval_tests.py](SemanticEvaluators/find_semantic_eval_tests.py)
  (the coarser "declared"/deps view) is now a **standalone optional** cross-check,
  not an orchestrator stage.
- **pipeline/** hub: [paths.py](pipeline/paths.py) (all artifact paths),
  [engines.py](pipeline/engines.py) (Wrapper shim + inlined test helpers),
  [eval_calls.py](pipeline/eval_calls.py) (`EVAL_CALLS`).

**Decisions locked in:**
- `giskard` moved out of `FRAMEWORK_CALLS` into `EVAL_CALLS` (it's an evaluator,
  not an LLM-invoking framework) — eval-only now. Still a Stage-2 app-search term.
- `languages` API call **dropped** from both stages (primary `language` already
  known/free; the breakdown wasn't worth the call). No net-new Core calls in Stage 2.
- Stage 2 enrichment does **not** short-circuit — all signals computed for every
  enriched repo, so metadata is complete and the decision is one pure function.
- `commits/month` is `>= 2` (spec "at least 2"), not `> 2`.
- Stage 2 import patterns are **derived from Stage 1 `import_names`** (read from
  real `__init__.py` package structure), not a hand-maintained list. Searches
  **all** of a framework's import names (subpackages like `crewai_tools` are
  real ecosystem users, not noise). Remaining precision risk: generic import
  names colliding with unrelated packages (e.g. `agents`) — quality filters
  can't catch that; optional later fix is dep-file (PyPI-name) verification.

**Tests** (run the two groups separately — duplicate basenames collide in one
pytest invocation):
```
python -m pytest Frameworks/test_github_search.py Applications/test_search_candidates.py Applications/test_framework_distribution.py pipeline/test_batch_call_metadata.py pipeline/test_eval_frequency.py -q   # 65 pass
cd Wrapper && python -m pytest tests/ -q   # 20 pass, 9 transitive fail on 3.9 (pass on 3.14)
```

**NOT yet done (next-session backlog):**
1. **No full runs executed** — only smokes. Real artifacts don't exist until the
   stages run for real (needs Python 3.14 for the transitive invoker pass).
2. **Nothing committed** to git yet.
3. Legacy/duplicate scripts not retired: [Applications/GithubSearch.py](Applications/GithubSearch.py)
   (superseded by search_candidates), test-extraction scripts (see Appendix).
4. *(optional)* the dependency-"declared" cross-check
   ([SemanticEvaluators/find_semantic_eval_tests.py](SemanticEvaluators/find_semantic_eval_tests.py)
   + deep_dep_check.py) is standalone, not wired — only needed to audit
   `EVAL_CALLS` pattern completeness ("declares X but no call detected").
5. *(optional precision)* generic import-name collision verification in Stage 2 —
   confirm a matched repo's dependency actually resolves to the framework's PyPI
   name (reuse SemanticEvaluators dep-file logic). Not blocking.

---

> **Everything lives under `pipeline/`.** All **new code** (orchestrator, batch
> driver, eval seed dict, reporters) and all **generated artifacts** (every
> `.csv` and metadata file) live under the top-level `pipeline/` folder. The
> existing stage scripts stay in their folders (`Frameworks/`, `Applications/`,
> `Wrapper/`) but are **repointed to read/write inside `pipeline/artifacts/`**.
> All artifact paths below are relative to `pipeline/artifacts/`.

```
 (1) Framework search ──► frameworks.csv ──┐    (all .csv/metadata below → pipeline/artifacts/)
                                           │  (import_names column drives Stage 2)
 (2) Application search ◄──────────────────┘
        │
        ├──► applications.csv         ──► (3) Framework frequency table
        ├──► application_metadata.csv     (full per-app metadata)
        │
        ├──► applications.csv         ──► (5) Batch invoker + LLM-call extraction
        │                                 │
        │                                 ├──► llm_invokers_all.csv        (invoker search: all direct+transitive)
        │                                 ├──► llm_calls_all.csv           (list of LLM calls)
        │                                 ├──► call_metadata_all.csv ───────┐  (LLM call arg metadata)
        │                                 └──► llm_tests_all.csv           (pytest subset of invokers)
        │                                                                  │
        │                                                                  ▼
        │                                                            (6) JOERN ──► forward slices
        │                                                                  ▲        backward slices
        │                                                                  │
        └──► applications.csv         ──► (7) Semantic evaluation          │
                                          │                                │
                                          ├──► eval_invokers_all.csv       (eval invoker search)
                                          ├──► eval_calls_all.csv          (list of Eval calls)
                                          ├──► eval_call_metadata_all.csv ─┘  (eval call arg metadata)
                                          └──► eval_frequency.csv          (Eval frequency table)
```

### `pipeline/` layout

```
pipeline/
  __init__.py
  __main__.py                   # `python -m pipeline run ...`
  run.py                        # orchestrator: runs each stage as a subprocess in order
  paths.py                      # every artifact path in one place (single source of truth)
  engines.py                    # shim re-exporting Wrapper engines (+ inlined test helpers)
  batch_call_metadata.py        # Stage 5+7 driver (LLM + eval seed dicts, one parse)
  eval_calls.py                 # EVAL_CALLS seed pattern dict
  eval_frequency.py             # Stage 7 eval_frequency.csv reporter
  test_batch_call_metadata.py   # driver tests
  test_eval_frequency.py        # eval-frequency tests
  repos/                        # cloned application checkouts (gitignored)
  artifacts/                    # all generated CSV/metadata (gitignored)
    frameworks.csv              # Stage 1 (was Frameworks/github_agent_framework_candidates.csv)
    frameworks_filter_stats.json
    applications.csv            # Stage 2 (was Applications/application_candidates_v2.csv)
    application_metadata.csv    # Stage 2 (full per-app metadata)
    applications_filter_stats.json
    .search_progress.json
    framework_frequency.csv     # Stage 3
    llm_invokers_all.csv        # Stage 5 (invoker search)
    llm_calls_all.csv           # Stage 5
    call_metadata_all.csv       # Stage 5  → JOERN
    llm_tests_all.csv           # Stage 5
    .batch_progress.json
    eval_invokers_all.csv       # Stage 7
    eval_calls_all.csv          # Stage 7
    eval_call_metadata_all.csv  # Stage 7  → JOERN
    eval_frequency.csv          # Stage 7
```

Notes:
- `pipeline/` modules import the `Wrapper/` engines via a `sys.path` shim in
  [engines.py](pipeline/engines.py); the stage scripts add the repo root to
  `sys.path` to import `pipeline.paths`. Resolved.
- `repos/` and `artifacts/` are large/regenerable → gitignored (kept via
  `.gitkeep`).
- `artifacts/` is flat.

---

## Stage 1 — Framework search (+ max metadata, + framework list)

**Script:** [Frameworks/GithubSearch.py](Frameworks/GithubSearch.py)
**Outputs:** `pipeline/artifacts/frameworks.csv` + `frameworks_filter_stats.json`
(was `Frameworks/github_agent_framework_candidates.csv`)

Searches GitHub repo search across several agent-framework phrases
(`SEARCH_PHRASES` + `stars:>=1000 language:Python`, built by `build_search_queries()`),
dedupes, and enriches each repo with as much metadata as the API cheaply gives:
stars, forks, watchers, subscribers/network counts, owner, homepage, feature
flags, visibility, language, topics, size, timestamps, latest commit date,
license, contributor count, test-file/function counts, `has_ci`, clone URL, and
the **`import_names`** column. Filters (per Hasan spec) drop archived repos,
`contributors < 2`, and repos with no test files. Funnel counts go to
`frameworks_filter_stats.json`.

**`import_names`** (`derive_import_names`) reads the repo's `__init__.py` package
structure to find the importable top-level package name(s) — handling `src/`
layouts and monorepos (`libs/langchain/langchain/` → `langchain`). **Stage 2
consumes this** to build its code-search patterns, so there's no hand-maintained
import-pattern list.

**Status:** ✅ Built + tested ([Frameworks/test_github_search.py](Frameworks/test_github_search.py)).
Source of truth for "the list of frameworks."

---

## Stage 2 — Application search (consumes the framework list)

**Script:** [Applications/search_candidates.py](Applications/search_candidates.py)
**Outputs (all under `pipeline/artifacts/`):**
- `applications.csv` — the filtered application list (the columns downstream
  stages need). *Existing; was `Applications/application_candidates_v2.csv`.*
- `application_metadata.csv` — **as much per-application metadata as is
  acquirable** from the GitHub API, one row per application. *New (see below).*
- resumable `.search_progress.json` (also under `pipeline/artifacts/`)

**Import patterns derived from Stage 1.** `load_frameworks()` reads
`frameworks.csv` and its `import_names`; `import_patterns(name)` turns each into
`from <name> import` / `from <name>.` / `import <name>`; `build_import_index()`
maps each importable name → the framework(s) that ship it. Stage 2 then runs
GitHub **code search** for those patterns to find repos that genuinely import the
framework. There is **no hand-maintained `FRAMEWORK_IMPORT_PATTERNS` /
`FRAMEWORK_SEARCH_TERMS` / drift notice** — Stage 1's output drives Stage 2.
Searches **all** of a framework's import names (subpackages like `crewai_tools`
are real ecosystem users). Framework repos themselves are excluded.

**Search-time filters** (`passes_search_filters`, on the full repo payload):
Python **primary** language, not fork/archived/disabled, `stars >= 10`, pushed
after 2025-04-14. **Enrichment filters** (`evaluate_candidate`, no short-circuit
so metadata is complete): `lifetime >= 30d`, `contributors >= 2`,
`commits/month >= 2`, `test_files >= 1`. Funnel → `applications_filter_stats.json`.

**Two outputs:**
- `applications.csv` — the lean kept-candidate work list (always created
  header-only up front, so Stage 5 never crashes on 0 candidates).
- `application_metadata.csv` — a wide best-effort record for **every enriched
  repo** (passed search-time filters), with `is_candidate` / `drop_reason`:
  identity (incl. `homepage`, `owner_login`, `owner_type`), popularity
  (`watchers`, `subscribers_count`, `network_count`), timeline + commit stats,
  classification, feature flags, `has_ci`, test counts, `matched_frameworks`
  (the import name(s) matched). The `languages` byte-breakdown call was
  **dropped** (primary `language` is free and Python-primary is already enforced).

**Status:** ✅ Built + tested ([Applications/test_search_candidates.py](Applications/test_search_candidates.py)).

---

## Stage 3 — Framework frequency table (apps per framework)

**Script:** [Applications/framework_distribution.py](Applications/framework_distribution.py)
**Outputs:** console tables **and** `framework_frequency.csv`
(reads `.search_progress.json` + `applications.csv`)

Reports three views of "how many applications import which framework", keyed by
import name:
1. **Imported** — distinct repos importing each name at search time, pre-filter
   (popularity signal, from `framework_repo_counts`).
2. **Candidate** — repos that passed search-time filters.
3. **Kept** — applications in `applications.csv` (`matched_frameworks` column).

`framework_frequency.csv` columns: `framework, repos_imported, repos_candidate,
repos_kept, pct_of_kept` (one row per framework, sorted by kept then imported).
`build_frequency_rows()` is the pure aggregator. Wired as the
`framework_frequency` orchestrator stage; can also run mid-Stage-2.

**Status:** ✅ Built + tested ([Applications/test_framework_distribution.py](Applications/test_framework_distribution.py)).

---

## Stage 4 — Application list

The kept rows of `pipeline/artifacts/applications.csv` **are** the application
list. Key columns consumed downstream: `full_name`, `clone_url`,
`default_branch`, `matched_frameworks`.

**Status:** ✅ Implicit output of Stage 2. No separate script needed.

---

## Stage 5 — Batch invoker + LLM-call extraction over ALL applications

**Driver:** [pipeline/batch_call_metadata.py](pipeline/batch_call_metadata.py)
(the `analysis` orchestrator stage). For each app in `applications.csv`:
**shallow-clone → index once → run both passes → write → delete the clone.**

> **One driver, shared with Stage 7.** Stages 5 and 7 are the same operation —
> clone, parse, find call sites — differing only in the **seed dict**
> (`FRAMEWORK_CALLS` vs. `EVAL_CALLS`). The engines
> ([transitive_invokers.py](Wrapper/transitive_invokers.py),
> [call_metadata.py](Wrapper/call_metadata.py)) were parameterized to accept a
> `framework_calls` dict, so a **single clone+parse per repo** emits both Stage 5
> and Stage 7 outputs. The repo is never cloned twice.

Per repo, the driver indexes once with the *union* of seed-dict keys, then for
each seed dict runs `seed_invokers` → `transitive_closure` (the **invoker
search**) and `collect_rows` (per-argument metadata). Every row carries a `repo`
column. **Four LLM outputs:**

- **`llm_invokers_all.csv`** — the invoker search result: every function that
  directly or transitively reaches an LLM call.
  `repo, qname, file, line, reason, kind` (`kind` = direct/transitive).
- **`llm_calls_all.csv`** — one row per call **site** (deduped by `call_id`):
  `repo, call_id, file, enclosing_qname, framework, pattern, callable,
  call_source, call_line, call_col, is_await, arg_count`.
- **`call_metadata_all.csv`** — per-**argument** detail (`call_metadata.FIELDS` +
  `repo`): `arg_kind/arg_source/arg_names/arg_is_literal/call_arg_vars` … the
  **JOERN input**. `llm_calls` is the one-row-per-`call_id` view of this.
- **`llm_tests_all.csv`** — the pytest subset of `llm_invokers` (file + function
  match pytest conventions).

**Robustness (built):** resumable via `.batch_progress.json` (`--resume`);
per-repo error isolation (clone/parse failure logs to a `failed` list and
continues); clone deleted in a `finally` (Windows read-only `.git` handled),
`--keep-clones` to override; `--limit` for smoke runs. The call graph (pyan3) is
**best-effort** — without it (Python 3.9) the closure reduces to direct seeds, so
calls/metadata are complete and only the *transitive* invoker/test rows are
missing; full transitive results need **Python 3.14 + pyan3**.

**Status:** ✅ Built + tested ([pipeline/test_batch_call_metadata.py](pipeline/test_batch_call_metadata.py)).
Direct/seed path validated; transitive path needs 3.14 to exercise.

---

## Stage 6 — JOERN slicing

**Inputs (both per-argument metadata CSVs, same schema):**
- `pipeline/artifacts/call_metadata_all.csv` — LLM call sites (Stage 5).
- `pipeline/artifacts/eval_call_metadata_all.csv` — eval call sites (Stage 7).

**Outputs:** **forward slices** and **backward slices** for both call kinds.

JOERN keys off `(file, line)` or `(file, line, variable)` — both CSVs carry
`call_line`, `call_col`, `call_arg_vars`, and per-arg `arg_names`, so both seed
shapes are present. Because the two files share the identical column contract
(LLM vs. eval differs only by the `framework`/`pattern` values), JOERN runs the
same way over each — slice the LLM seeds, the eval seeds, or both.

**Status:** External (JOERN). Our responsibility ends at producing clean,
complete `call_metadata_all.csv` **and** `eval_call_metadata_all.csv`.
See [memory: param slicing direction] —
`Wrapper/param_slicing.py` is the interprocedural backward-slice follow-up.

**⚠️ Open: clones are deleted.** The `analysis` stage removes each checkout after
extracting metadata (disk hygiene), so the source `(repo, file)` JOERN needs is
**not on disk** by the time it runs. Resolve one of: run JOERN per repo *before*
deletion (fold it into the driver), pass `--keep-clones` for the JOERN run, or
re-clone from the metadata's `repo` on demand. Decide with the JOERN owner, plus
confirm the `(repo + file) → on-disk path` contract.

---

## Stage 7 — Semantic evaluation (piggybacks on the invoker search)

Stage 7 is **not a separate mechanism** — it's the eval seed dict run through the
same invoker-search machinery as Stage 5. The `analysis` stage indexes each repo
once and evaluates **both** `FRAMEWORK_CALLS` (LLM) and `EVAL_CALLS` (eval)
against it, so the eval outputs come from the same clone+parse. `EVAL_CALLS`
lives in [pipeline/eval_calls.py](pipeline/eval_calls.py) (deepeval, ragas,
giskard, opik, phoenix — `giskard` is eval-only, removed from `FRAMEWORK_CALLS`).

**Outputs (from the `analysis` stage, repo-keyed):**
- **`eval_invokers_all.csv`** — eval invoker search (direct+transitive).
- **`eval_calls_all.csv`** — one row per eval call site (same schema as
  `llm_calls_all.csv`).
- **`eval_call_metadata_all.csv`** — per-argument eval metadata, **identical
  schema** to `call_metadata_all.csv` (only `framework`/`pattern` differ), so
  JOERN slices it the same way (Stage 6).

**Eval frequency table** ([pipeline/eval_frequency.py](pipeline/eval_frequency.py),
the `eval_frequency` stage) — derived **purely from the "called" view**
(`eval_calls_all.csv`): `eval_framework, repos_with_calls, total_call_sites,
pct_of_apps`. No dependency-declaration check — the AST-detected calls *are* the
signal.

> **Dropped from the pipeline:** the dependency-file "declared" check
> ([SemanticEvaluators/find_semantic_eval_tests.py](SemanticEvaluators/find_semantic_eval_tests.py),
> deep_dep_check.py). It was a separate, coarser mechanism (greps dep files for a
> package name) made redundant by the invoker-search piggyback. It remains on
> disk as a **standalone optional** cross-check (audit `EVAL_CALLS` completeness:
> "declares X but no call detected"), not an orchestrator stage.

**Status:** ✅ Built + tested ([pipeline/test_eval_frequency.py](pipeline/test_eval_frequency.py);
calls/metadata covered by the driver tests). `EVAL_CALLS` patterns are still
provisional and want a usage double-check.

---

## Status by stage (as built)

| Stage | Script | Status |
|---|---|---|
| 1 Framework search | [Frameworks/GithubSearch.py](Frameworks/GithubSearch.py) | ✅ built + tested |
| 2 Application search | [Applications/search_candidates.py](Applications/search_candidates.py) | ✅ built + tested |
| 3 Framework frequency | [Applications/framework_distribution.py](Applications/framework_distribution.py) | ✅ built + tested |
| 4 Application list | (the kept rows of `applications.csv`) | ✅ implicit |
| 5 Invoker search + LLM calls | [pipeline/batch_call_metadata.py](pipeline/batch_call_metadata.py) | ✅ built + tested (transitive needs 3.14) |
| 6 JOERN slicing | external | inputs ready |
| 7 Eval calls + frequency | driver + [pipeline/eval_frequency.py](pipeline/eval_frequency.py) | ✅ built + tested |
| — Orchestrator | [pipeline/run.py](pipeline/run.py) | ✅ `frameworks → applications → framework_frequency → analysis → eval_frequency` |

Remaining: a real full run on **Python 3.14**, a git commit, and (optional)
retiring the legacy/duplicate scripts below. See **CURRENT STATE → backlog**.

## Open questions / risks
- **Code-search rate limits** (Stage 2) are strict (~30 req/min); handled with
  backoff, but full reruns are slow.
- **Clone volume** (Stage 5): hundreds of repos — mitigated by deleting each
  clone right after analysis (only one checkout on disk at a time).
- **Generic import-name collisions** (Stage 2): a generic derived import name
  (e.g. `agents`) can match unrelated packages; the quality filters can't catch
  that. Optional fix: verify the matched repo's dep resolves to the framework's
  PyPI name. Not blocking.
- **Transitive pass unvalidated**: the call graph (pyan3) needs Python 3.14; under
  3.9 only the direct-seed half of the invoker search has run. First 3.14 run
  validates the transitive rows.
- **Legacy/duplicate scripts** still on disk, not wired (retire when convenient):
  [Applications/GithubSearch.py](Applications/GithubSearch.py) (older repo-search,
  superseded by `search_candidates.py`); test-extraction scripts
  [Applications/analyze_tests.py](Applications/analyze_tests.py),
  [Frameworks/find_llm_tests.py](Frameworks/find_llm_tests.py),
  [Frameworks/extract_llm_tests.py](Frameworks/extract_llm_tests.py),
  [Frameworks/reformat_csv.py](Frameworks/reformat_csv.py) (superseded by
  `llm_tests_all.csv`); `*First` archived CSVs (prior-run snapshots).
- **Duplicate module basenames** (`GithubSearch.py` in Frameworks/ & Applications/;
  `find_llm_tests.py` in Wrapper/ & Frameworks/) collide if all test dirs are
  collected in one pytest invocation — run the two test groups separately.
