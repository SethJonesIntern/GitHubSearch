# GitHubSearch Pipeline

End-to-end plan for going from *"which agent frameworks exist on GitHub"* to
*"per-argument metadata for every LLM call site across applications that use
those frameworks"* — the input JOERN consumes to produce forward and backward
slices.

The pipeline is seven stages (Stage 5 LLM-call extraction and Stage 7 semantic
evaluation are parallel branches off the same application list). Most stages
already exist as standalone scripts;
this document records the **target wired-up flow**, what each stage produces,
and the orchestration work still needed to connect them.

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

> **Everything lives under `pipeline/`.** All **new code** (batch driver, eval
> seed dict, reporters, drift check) and all **generated artifacts** (every
> `.csv` and metadata file produced by any stage) go into a new top-level
> `pipeline/` folder, so the data set can be organized, reformatted, and
> examined in one place. The existing stage scripts stay in their current
> folders (`Frameworks/`, `Applications/`, `Wrapper/`, `SemanticEvaluators/`)
> but are **repointed to read/write inside `pipeline/`**. All artifact paths
> below are relative to `pipeline/` (e.g. `pipeline/frameworks.csv`).

```
 (1) Framework search ──► frameworks.csv ──┐    (all .csv/metadata below → pipeline/)
                                           │  (drift check, non-fatal)
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
  __main__.py                   # ✅ `python -m pipeline run ...`
  run.py                        # ✅ orchestrator: runs each stage as a subprocess in order
  paths.py                      # ✅ every artifact path in one place (single source of truth)
  engines.py                    # ✅ shim re-exporting Wrapper engines (+ inlined test helpers)
  batch_call_metadata.py        # ✅ shared Stage 5+7 driver (LLM + eval seed dicts)
  test_batch_call_metadata.py   # ✅ driver tests
  eval_calls.py                 # ✅ EVAL_CALLS seed pattern dict
  eval_frequency.py             # 🚧 eval_frequency.csv reporter (scaffold)
  repos/                        # cloned application checkouts (gitignored)
  artifacts/                    # all generated CSV/metadata (gitignored)
    frameworks.csv              # Stage 1 (was Frameworks/github_agent_framework_candidates.csv)
    applications.csv            # Stage 2 (was Applications/application_candidates_v2.csv)
    application_metadata.csv    # Stage 2 (full per-app metadata)
    framework_frequency.csv     # Stage 3
    llm_calls_all.csv           # Stage 5
    call_metadata_all.csv       # Stage 5  → JOERN
    llm_tests_all.csv           # Stage 5
    eval_calls_all.csv          # Stage 7
    eval_call_metadata_all.csv  # Stage 7  → JOERN
    eval_frequency.csv          # Stage 7
    semantic_evaluator_repos.csv# Stage 7 dep-detection (was SemanticEvaluators/...)
```

Notes:
- New modules in `pipeline/` import the existing engines from `Wrapper/`
  (`transitive_invokers`, `call_metadata`, `FrameworkDict`); add `pipeline/` as a
  package and resolve those imports via a path shim or by running from repo root.
- `repos/` and `artifacts/` are large/regenerable → add to `.gitignore`.
- Decide whether to keep `artifacts/` flat or split per stage; flat is simplest
  for the reformatting/examination you want.

---

## Stage 1 — Framework search (+ max metadata, + framework list)

**Script:** [Frameworks/GithubSearch.py](Frameworks/GithubSearch.py)
**Output:** `pipeline/artifacts/frameworks.csv`
(repoint `OUTPUT_CSV`; was `Frameworks/github_agent_framework_candidates.csv`)

Searches GitHub repo search across several agent-framework queries
(`SEARCH_QUERIES`), dedupes, and enriches each repo with as much metadata as the
API cheaply gives: stars, forks, language, topics, issues, size, created/updated/
pushed timestamps, latest default-branch commit date, license, contributor
count, test-file count, test-function count, clone URL. Filters out archived/
disabled repos, single-contributor repos, and repos with no tests.

**Status:** ✅ Works as-is. This is the source of truth for "the list of
frameworks."

**Possible enhancements (optional):**
- Emit a derived `package_name` / import-name column so Stage 2's drift check
  (below) is exact rather than heuristic.

---

## Stage 2 — Application search (consumes the framework list)

**Script:** [Applications/search_candidates.py](Applications/search_candidates.py)
**Outputs (all under `pipeline/artifacts/`):**
- `applications.csv` — the filtered application list (the columns downstream
  stages need). *Existing; was `Applications/application_candidates_v2.csv`.*
- `application_metadata.csv` — **as much per-application metadata as is
  acquirable** from the GitHub API, one row per application. *New (see below).*
- resumable `.search_progress.json` (also under `pipeline/artifacts/`)

Uses GitHub **code search** for each framework's exact import-statement
substring (`FRAMEWORK_IMPORT_PATTERNS`) to find repos that genuinely import the
framework, then fetches full repo metadata and applies quality filters (stars,
push date, lifetime, contributors, commits/month, has tests). Excludes the
framework repos themselves (loaded from the Stage 1 CSV).

### Decision: emit a full application-metadata output

The candidates CSV is intentionally lean (it's the downstream work list). In
addition, capture a wide metadata record per application — the analogue of what
Stage 1 pulls for frameworks, plus everything else cheaply acquirable. Target
columns (superset of the candidates CSV):

- **Identity:** `full_name`, `html_url`, `clone_url`, `default_branch`,
  `description`, `homepage`, `owner_login`, `owner_type` (User/Org).
- **Popularity / activity:** `stars`, `forks`, `watchers`, `subscribers_count`,
  `network_count`, `open_issues`, `size_kb`.
- **Timeline:** `created_at`, `updated_at`, `pushed_at`,
  `latest_default_branch_commit_date`, `lifetime_days`, `total_commits`,
  `commits_per_month`.
- **People:** `contributors`.
- **Classification:** `language`, `languages` (full byte breakdown), `topics`,
  `license`, `is_fork`, `archived`, `disabled`.
- **Test surface:** `test_file_count`, `test_function_count` (parse `test_*.py`
  like Stage 1 does), `has_ci` (presence of `.github/workflows`).
- **Linkage:** `matched_frameworks` (which framework import patterns hit).

Rules:
- "As much as acquirable" is **best-effort**: any field whose fetch fails or is
  unavailable is left blank, never aborts the row.
- This output is written for **every repo that passes the search-time filters**,
  before/independent of the quality cuts that gate the candidates list — so we
  retain metadata even on repos that don't become candidates. (Decide: full
  superset, or only kept candidates. Default: write metadata for all enriched
  repos, mark candidacy with a boolean `is_candidate` column.)

**Work needed:**
- Extend enrichment in `search_candidates.py` to fetch the extra fields
  (`languages`, `subscribers_count`/`network_count`/`watchers` from the repo
  details payload, `.github/workflows` presence + `test_function_count` from the
  tree, owner fields) and write `application_metadata.csv` alongside the
  candidates CSV. Keep it resumable and incrementally flushed like the existing
  output.

**Status:** ⚠️ Partially connected. It already loads the Stage 1 CSV — but only
to *exclude* framework repos. The actual search terms and import patterns are a
**hand-curated dict** (`FRAMEWORK_SEARCH_TERMS` + `FRAMEWORK_IMPORT_PATTERNS`),
not derived from Stage 1.

### Decision: curated patterns stay authoritative; warn on drift

The hand-tuned import patterns remain the source of truth (auto-deriving import
names from repo names is unreliable — PyPI/repo/import names diverge). But we
add a **non-fatal drift check**:

> When loading the Stage 1 CSV, for every framework repo that does **not** appear
> to be covered by any entry in `FRAMEWORK_IMPORT_PATTERNS`, print a console
> notice — e.g.
> `NOTE: framework 'owner/newthing' from Stage 1 has no import pattern configured; skipping it in app search` —
> **and keep going.** This surfaces newly discovered frameworks that should be
> added to the pattern dict, without blocking the run.

**Work needed:**
- Add a `report_uncovered_frameworks()` helper in `search_candidates.py` that
  cross-references the Stage 1 CSV `full_name`s against the configured patterns
  (heuristic: does any search term/import token appear in the repo name?) and
  logs the uncovered ones before the search loop starts.

---

## Stage 3 — Framework frequency table (apps per framework)

**Script:** [Applications/framework_distribution.py](Applications/framework_distribution.py)
**Output:** console tables (reads `.search_progress.json` + candidates CSV)

Reports three views of "how many applications import which framework":
1. **Imported** — distinct repos containing each framework's import pattern,
   captured pre-filter (popularity signal).
2. **Candidate** — repos that passed search-time filters.
3. **Kept** — repos in the final CSV (`matched_frameworks` column).

**Status:** ✅ Works as-is, reads Stage 2's progress file. Can run mid-Stage-2.

**Possible enhancement:** also write the frequency table to a CSV
(`pipeline/artifacts/framework_frequency.csv`) so it's a durable artifact, not
just console output.

---

## Stage 4 — Application list

The kept rows of `pipeline/artifacts/applications.csv` **are** the application
list. Key columns consumed downstream: `full_name`, `clone_url`,
`default_branch`, `matched_frameworks`.

**Status:** ✅ Implicit output of Stage 2. No separate script needed.

---

## Stage 5 — Batch invoker + LLM-call extraction over ALL applications

**Existing single-repo engines (in `Wrapper/`):**
- [transitive_invokers.py](Wrapper/transitive_invokers.py) — builds the call
  graph, finds direct LLM-call seeds (via [FrameworkDict.py](Wrapper/FrameworkDict.py)
  `FRAMEWORK_CALLS`), computes the transitive closure of invokers.
- [find_llm_tests.py](Wrapper/find_llm_tests.py) — filters invokers down to
  pytest tests.
- [call_metadata.py](Wrapper/call_metadata.py) — per-**argument** metadata for
  every direct LLM call site (the backslicer's input).

**Status:** ❌ Not wired for batch. Each takes a single `target` (dir or git
URL) on the CLI. "Over all the applications" needs a new driver.

### Decision: one combined dataset, three outputs

> **Shared driver with Stage 7.** Stages 5 and 7 are the same operation —
> clone every app, parse it, find call sites — differing only in the **seed
> pattern dict** (`FRAMEWORK_CALLS` for LLM calls vs. `EVAL_CALLS` for eval
> calls). So the driver is **parameterized over one or more seed dicts** and
> does a **single clone + parse pass per repo**, emitting Stage 5's *and*
> Stage 7's outputs together. We never clone the app set twice.

Build a new batch driver — **`pipeline/batch_call_metadata.py`** (working name) —
that:

1. Reads `pipeline/artifacts/applications.csv`.
2. For each app row: clone (shallow) via `ensure_clone(clone_url)` into
   `pipeline/repos/` (reuse existing clone helper; skip if already present).
3. Run the existing engines per repo **once**:
   `index_repo` → `seed_invokers` → `build_call_graph`/`transitive_closure`
   (from `transitive_invokers.py`) and `collect_rows` (from `call_metadata.py`),
   evaluating each configured seed dict (LLM and eval) against the same parsed
   AST/call graph.
4. Aggregate into **three combined, repo-keyed outputs** for the LLM seeds (each
   a single file spanning all apps, every row/record carrying a `repo` =
   `full_name` column) — plus Stage 7's eval outputs from the same pass:

   **(a) List of LLM calls — `pipeline/artifacts/llm_calls_all.csv`**
   One row per **call site** (deduped by `call_id`). Columns:
   `repo`, `call_id`, `file`, `enclosing_qname`, `framework`, `pattern`,
   `callable`, `call_source`, `call_line`, `call_col`, `is_await`, `arg_count`.
   This is the "what/where are the LLM calls" inventory — derived by collapsing
   the per-argument rows to one per `call_id`.

   **(b) Metadata on those LLM calls — `pipeline/artifacts/call_metadata_all.csv`**
   The full per-**argument** detail from `call_metadata.py` (`FIELDS`) **plus a
   leading `repo` column**. This is the backslicer's primary input: every
   argument of every call site, with `arg_kind`, `arg_source`, `arg_names`,
   `arg_is_literal`, `call_arg_vars`, etc. (a) is a view of this; (b) is the
   superset.

   **(c) List of LLM tests — `pipeline/artifacts/llm_tests_all.csv`**
   The pytest tests that reach an LLM call (direct or transitive), from
   `find_llm_tests.py`'s filtering of the transitive closure. Columns:
   `repo`, `qname`, `file`, `line`, `reason` (e.g. `matches '.invoke' from
   langchain` vs. `calls some.qname`), `kind` (direct/transitive).

5. Optionally also emit a combined invoker JSON
   (`pipeline/artifacts/llm_invokers_all.json`) keyed by repo for traceability.

**Robustness requirements for the batch driver:**
- Resumable: track processed repos (e.g. a progress JSON like Stage 2) so a
  crash/rate-limit doesn't restart from zero.
- Per-repo error isolation: a parse failure or clone failure logs and continues;
  one bad repo never kills the batch.
- Incremental flush: append/rewrite the combined CSV every N repos.
- Clone hygiene: large repo set — decide on keeping vs. deleting clones after
  extraction (default: keep under `pipeline/repos/`, reuse on re-run).

**Work needed:** write `batch_call_metadata.py`. The per-repo logic is a thin
loop over functions already exported by `call_metadata.py` /
`transitive_invokers.py`; the new code is the CSV iteration, cloning, the `repo`
column, aggregation, and resumability.

---

## Stage 6 — JOERN slicing

**Inputs (both per-argument metadata CSVs, same schema):**
- `pipeline/artifacts/call_metadata_all.csv` — LLM call sites (Stage 5).
- `pipeline/artifacts/eval_call_metadata_all.csv` — eval call sites (Stage 7).

**Outputs:** **forward slices** and **backward slices** for both call kinds.

JOERN keys off `(file, line)` or `(file, line, variable)` — both CSVs carry
`call_line`, `call_col`, `call_arg_vars`, and per-arg `arg_names`, so both seed
shapes are present. The `repo` column lets JOERN locate the corresponding clone
under `pipeline/repos/`. Because the two files share the identical column
contract (LLM vs. eval differs only by the `framework`/`pattern` values), JOERN
runs the same way over each — slice the LLM seeds, the eval seeds, or both.

**Status:** External (JOERN). Our responsibility ends at producing clean,
complete `call_metadata_all.csv` **and** `eval_call_metadata_all.csv`.
See [memory: param slicing direction] —
`Wrapper/param_slicing.py` is the interprocedural backward-slice follow-up.

**Work needed (our side):** confirm the combined CSV's column contract with
JOERN (especially how `repo` + `file` resolve to an on-disk path).

---

## Stage 7 — Semantic evaluation (parallel branch off the application list)

**Existing code (in `SemanticEvaluators/`):**
- [find_semantic_eval_tests.py](SemanticEvaluators/find_semantic_eval_tests.py) —
  for each candidate repo, downloads root-level dependency files and flags which
  semantic-eval frameworks it **declares** (giskard, deepeval, opik, ragas,
  promptfoo, arize-phoenix). Output: `pipeline/artifacts/semantic_evaluator_repos.csv`
  (repoint; was `SemanticEvaluators/semantic_evaluator_repos.csv`).
- [deep_dep_check.py](SemanticEvaluators/deep_dep_check.py) — for repos with no
  root-level dep files, walks the full tree to find dep files anywhere.

**Status:** ⚠️ Detection only. Today we know *which apps depend on* an eval
framework — not *where they call it* or *how often each is used*. This stage
adds the two outputs you want, in parallel with Stage 5's LLM-call extraction.

### Decision: three outputs — eval call list + eval call metadata + frequency

> **Shares Stage 5's driver and clone pass.** All eval outputs come from the
> *same* `batch_call_metadata.py` run, by passing the `EVAL_CALLS` seed dict
> alongside `FRAMEWORK_CALLS`. No separate cloning or parsing — Stage 7's call
> extraction is a second seed dict evaluated against the same parsed repos.

Driven off the same `pipeline/artifacts/applications.csv`:

**(a) List of Eval calls — `pipeline/artifacts/eval_calls_all.csv`**
The eval-framework analogue of `llm_calls_all.csv`. Reuse the Stage 5 batch
machinery (`index_repo` → `seed_invokers` → call-site collection), but seed it
with **eval-framework call patterns** instead of `FRAMEWORK_CALLS`. One row per
eval call site, repo-keyed: `repo`, `file`, `enclosing_qname`, `eval_framework`,
`pattern`, `callable`, `call_source`, `call_line`, `call_col`.

**(b) Eval call metadata — `pipeline/artifacts/eval_call_metadata_all.csv`**
The eval analogue of `call_metadata_all.csv`: per-**argument** detail for every
eval call site, **identical column schema** (with `framework` = the eval
framework). This is what makes eval calls sliceable — it feeds Stage 6 (JOERN)
exactly like the LLM metadata does, yielding forward/backward slices for eval
calls. Produced by running `collect_rows` with the `EVAL_CALLS` seeds; (a) is the
one-row-per-`call_id` view of this file.

**(c) Eval frequency table — `pipeline/artifacts/eval_frequency.csv`**
The eval analogue of Stage 3. How many applications use each eval framework,
across views: **declared** (from `semantic_evaluator_repos.csv` dep detection)
and **called** (distinct repos with ≥1 eval call site in output (a)). Columns:
`eval_framework`, `repos_declared`, `repos_with_calls`, `total_call_sites`, `pct`.

**Work needed:**
- Add an `EVAL_CALLS` pattern dict (mirroring `FrameworkDict.FRAMEWORK_CALLS`)
  for giskard/deepeval/opik/ragas/promptfoo/phoenix invocation methods
  (e.g. `.evaluate`, `.measure`, `.scan`, `.run`, `assert_test`, `metric(...)`).
  Confirm against the [LLM module list memory] which already tracks these eval
  frameworks for non-determinism detection — keep the two lists in sync.
- Pass `EVAL_CALLS` as a second seed dict to the shared `batch_call_metadata.py`
  so the existing run also emits `eval_calls_all.csv` **and**
  `eval_call_metadata_all.csv` (no extra clone/parse).
- A small reporter that joins `eval_calls_all.csv` with
  `semantic_evaluator_repos.csv` to write `eval_frequency.csv`.

---

## Summary of work to build (vs. what already runs)

| # | Item | New / existing |
|---|------|----------------|
| 0 | **Create `pipeline/` (package + `repos/` + `artifacts/`); gitignore the latter two** | 🆕 setup |
| 0 | **Repoint every stage script's in/out paths to `pipeline/artifacts/`** (frameworks/applications/metadata/semantic CSVs + progress files) | 🆕 edit existing scripts |
| 1 | Framework search | ✅ existing |
| 2 | App search reads framework list for *exclusion* | ✅ existing |
| 2 | **Drift notice: warn on Stage 1 frameworks with no import pattern, keep going** | 🆕 add to `search_candidates.py` |
| 2 | **`application_metadata.csv`: full per-app metadata (best-effort superset)** | 🆕 add to `search_candidates.py` |
| 3 | Frequency table | ✅ existing (optional: also write CSV) |
| 4 | App list | ✅ implicit |
| 5+7 | **`batch_call_metadata.py`: clone every app once, parse once, evaluate LLM + eval seed dicts, aggregate** | 🆕 shared driver |
| 5 | **3 combined outputs: `llm_calls_all.csv`, `call_metadata_all.csv`, `llm_tests_all.csv`** | 🆕 |
| 5+7 | Resumability + per-repo error isolation in the driver | 🆕 |
| 6 | Confirm contract with JOERN for **both** `call_metadata_all.csv` + `eval_call_metadata_all.csv` | 🤝 coordinate |
| 7 | Semantic-eval dependency detection | ✅ existing (`find_semantic_eval_tests.py`, `deep_dep_check.py`) |
| 7 | **`EVAL_CALLS` pattern dict → `eval_calls_all.csv` + `eval_call_metadata_all.csv` (same driver pass)** | 🆕 |
| 7 | **`eval_frequency.csv` reporter (declared vs. called)** | 🆕 |

## Appendix — file/path migration checklist (Stage 0)

Exact constants to repoint when artifacts move to `pipeline/artifacts/`. Filenames
also get the cleaner names from the layout block (content/columns unchanged).

| File | Constant(s) | Change |
|------|-------------|--------|
| [Frameworks/GithubSearch.py](Frameworks/GithubSearch.py) | `OUTPUT_CSV` (L31) | **relative → absolute** `pipeline/artifacts/frameworks.csv`. ⚠️ currently bare `"...csv"`, written to CWD — must resolve from repo root, not `os.getcwd()`. |
| [Applications/search_candidates.py](Applications/search_candidates.py) | `FRAMEWORKS_CSV` (L31), `OUTPUT_CSV` (L103), `PROGRESS_FILE` (L104) | read `pipeline/artifacts/frameworks.csv`; write `applications.csv` + `.search_progress.json` there |
| [Applications/framework_distribution.py](Applications/framework_distribution.py) | `PROGRESS_FILE` (L26), `OUTPUT_CSV` (L27) | point both at `pipeline/artifacts/`; optionally also emit `framework_frequency.csv` |
| [SemanticEvaluators/find_semantic_eval_tests.py](SemanticEvaluators/find_semantic_eval_tests.py) | `CANDIDATE_CSVS` (L23), `OUT_CSV`/`NO_DEPS_CSV`/`PROGRESS_FILE` (L28-30) | read `applications.csv`/`frameworks.csv`; write outputs into `pipeline/artifacts/` |
| [SemanticEvaluators/deep_dep_check.py](SemanticEvaluators/deep_dep_check.py) | same set (L31-38) | same |
| [Wrapper/call_metadata.py](Wrapper/call_metadata.py) | `--out` default (L243) | default to `pipeline/artifacts/`; batch driver overrides anyway |

### Decisions this surfaced (not just path edits)

1. **Two application-search scripts exist.** [Applications/search_candidates.py](Applications/search_candidates.py)
   (code-search by import pattern — the Stage 2 script) **and**
   [Applications/GithubSearch.py](Applications/GithubSearch.py) (older repo-search
   by keyword, writes `github_agent_application_candidates.csv`, different
   columns). The plan uses `search_candidates.py`. **Retire/ignore `GithubSearch.py`**
   or it'll write a stray artifact with the wrong schema.

2. **Legacy test-extraction scripts not in the 7-stage plan** still read the old
   CSV names and are superseded by Stage 5's `llm_tests_all.csv`:
   [Applications/analyze_tests.py](Applications/analyze_tests.py),
   [Frameworks/find_llm_tests.py](Frameworks/find_llm_tests.py),
   [Frameworks/extract_llm_tests.py](Frameworks/extract_llm_tests.py),
   [Frameworks/reformat_csv.py](Frameworks/reformat_csv.py).
   Decide per script: **migrate paths, or retire.** (They produce
   `extracted_llm_tests/`, `llm_test_functions.csv`, `agent_framework_table.csv` —
   none referenced by the seven stages.)

3. **`*First` archived artifacts** (`application_candidatesFirst.csv`,
   `.search_progressFirst.json`, etc.) are prior-run snapshots. Leave in place or
   move to `pipeline/artifacts/archive/`; not part of the live flow.

4. **No column/schema changes** — only filenames and directories move. Readers key
   on `full_name`, `clone_url`, `default_branch`, `matched_frameworks`, all
   preserved. So this is path-reformatting, **not** a data reformat.

## Open questions / risks
- **Code-search rate limits** (Stage 2) are strict (~30 req/min); already handled
  with backoff, but full reruns are slow.
- **Clone volume** (Stage 5): the app set could be hundreds of repos × repo size.
  Need a disk budget / cleanup policy.
- **Import-name vs. repo-name** mismatch makes the Stage 2 drift check heuristic
  unless Stage 1 emits a real package-name column.
- **Seed coverage**: `FrameworkDict.FRAMEWORK_CALLS` and
  `search_candidates.FRAMEWORK_IMPORT_PATTERNS` are two separate hand-maintained
  maps; a framework added to one must be added to the other. Consider unifying.
- **Imports across folders**: `pipeline/` modules import the `Wrapper/` engines
  (`transitive_invokers`, `call_metadata`, `FrameworkDict`). Decide the
  mechanism — run from repo root with `pipeline/` as a package, a `sys.path`
  shim, or eventually move the engines into `pipeline/` too. Existing scripts
  keep their relative imports; only their I/O paths change.
- **Path migration**: repointing outputs to `pipeline/artifacts/` orphans the
  current CSVs in `Frameworks/`, `Applications/`, `SemanticEvaluators/`. Decide
  whether to move existing artifacts over or regenerate from scratch.
