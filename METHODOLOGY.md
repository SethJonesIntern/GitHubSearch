# Methodology — what we did

Skeleton for the Methods write-up. Each stage lists what was done and why. Where the
reason isn't recorded in the code or the docs, the why is left blank for Seth to fill.

Confirmed against the code on 2026-09-01. Problems, gaps and inconsistencies are **not**
here — they are in `METHODOLOGY_REVIEW.md` for the separate methodological review.

---

## Stage 1 — Framework discovery
`Frameworks/GithubSearch.py` → `frameworks.csv` (59 frameworks)

- **Searched GitHub repository search with five natural-language phrases** — to follow the
  GitHub-search method of the prior work by Mehedi Hasan.
  ```
  AI agent framework · LLM-based agent framework · LLM agent library
  multi-agent orchestration framework · LLM powered agents framework
  ```
- **Constrained every query to `stars:>=1000`** — star count as a proxy for repository
  popularity.
- **Constrained every query to `language:Python`** — the study analyses Python source.
- **Took up to 3 pages of 100 results per phrase** — why: 
- **Deduplicated by repository across phrases** — the same framework surfaces on several
  phrases; 71 unique repositories resulted.
- **Excluded archived repositories** — why: 
- **Required contributor count ≥ 2** — to focus on well-maintained frameworks; contributor
  count is a recognised maintenance indicator in prior research.
- **Required ≥ 1 test file** (`test_*.py`) — the study's object is testing practice.
- **Result: 59 frameworks** (71 → 2 archived, 2 contributors, 8 no tests).
- **Derived each framework's importable package names from its file tree** — a directory
  containing `__init__.py` whose parent does not, excluding test/doc/example directories.
  These names, not the repository names, are what Stage 2 searches for. **539 distinct
  names** across the 59 frameworks.
- **Recorded per-framework metadata** (stars, forks, contributors, license, CI, commit
  dates, test file and test function counts) — why: 

---

## Stage 2 — Application search
`Applications/search_candidates.py` → `applications.csv` (6,446) + `application_metadata.csv` (13,506)

- **Searched GitHub code search for each of the 539 import names** — to find repositories
  that import a discovered framework. Three quoted queries per name, covering the three
  import forms:
  ```
  "from <name> import"    "from <name>."    "import <name>"        (+ language:Python)
  ```
- **Took up to 10 pages of 100 per query** — GitHub's code-search result limit is 1,000.
- **Excluded the framework repositories themselves** — a framework's own repository is not
  an application built on it.
- **Applied filters at search time**: not a fork, not archived, not disabled; primary
  language Python; stars ≥ 10; pushed after 2025-04-14 — why (stars ≥ 10): popularity
  — why (2025-04-14): Recency
- **Applied quality filters after enrichment**, each repository charged to the first
  condition it fails:
  - lifetime (created → last push) ≥ 30 days — not a one-off repository.
  - contributors ≥ 2 — well-maintained.
  - commit frequency ≥ 2 commits/month — actively developed.
  - ≥ 1 test file (`test_*.py`) — the study's object is testing practice.
- **Computed the enrichment metrics from a local blobless clone rather than the REST API** —
  each metric was one API call per repository, which exhausted the hourly rate limit;
  `git rev-list` / `log` / `ls-tree` answer the same questions offline.
- **Kept a metadata row for every enriched repository**, including rejected ones, with
  `is_candidate` and `drop_reason` — so the funnel is recoverable.
- **Result: 6,446 candidates** from 13,505 enriched (1,494 lifetime, 2,436 contributors,
  1,087 commit frequency, 2,043 no tests).

---

## Stage 3 — Population definition
`Applications/slim_applications.py` → `applications_slim.csv` (1,055)

- **Classified all 539 import names into trustworthy and untrustworthy** — Stage 1's
  derivation over-extracted names (internal subpackages, generic words, vendored provider
  SDKs), so many candidates matched on a name that identifies no framework. Rules, in
  priority order:
  1. a **known-framework list** (25 names, hand-curated) → keep
  2. a **generic/internal list** (66 names: `api`, `core`, `utils`, `http`, `src`, …) → drop
  3. a **provider/tool-SDK list** (32 names: `openai`, `anthropic`, `google`, `azure`, …) →
     drop — these are provider SDKs or infrastructure, not agent frameworks
  4. name of ≥ 4 characters that is a substring of, or contains, its source framework's
     repository name → keep (143 names)
  5. anything else → drop (273 names)
- **Kept candidates matching ≥ 1 trustworthy name, and rewrote `matched_frameworks` to
  those names only** — no re-search was needed: every name was already searched, so the
  existing results already contain every repository matching a trustworthy name.
- **Excluded each framework's and eval tool's own repository** — a framework's own
  self-imports, tests and examples are not an application's use of it, and they dominate
  every metric. 9 removed at this step.
- **Result: the population of 1,055 applications.**
- **Tagged every row with two scope flags** rather than deleting rows — so all three
  denominators (all candidates / real AI apps / analysed apps) come from one file:
  - `real_ai_app` — the matched name is a framework or eval tool we have call patterns for.
  - `analyzed` — the matched name is one of the in-scope frameworks we measure.
- **Scoped measurement to the top-20 frameworks by application count** — the top 20 cover
  ~90% of the population, and frameworks below it would yield too few observations to
  characterise. Applications built only on out-of-scope frameworks are marked `uncovered`:
  excluded from the measured statistics, retained in the coverage denominator.
- **Excluded raw provider SDK calls (`openai`, `anthropic`) from the measurement** — the
  study is about how applications call models *through a framework*. SDK call sites are
  still detected and retained in the artifacts, but are not reported. This does not change
  the population: no repository entered the corpus on an SDK match, since both names are on
  the Stage-3 provider drop list.

---

## Stage 4 — Clone and parse
`pipeline/batch_call_metadata.py`

- **Shallow-cloned each application** (`--depth 1`) — only the current state of the code is
  analysed, so history is not needed.
- **Enabled `core.longpaths`** — many repositories have paths over Windows' 260-character
  limit, which otherwise fetch but fail at checkout.
- **Parsed every `.py` file once per repository**, skipping `.git`, virtualenvs,
  `__pycache__`, `node_modules`, `dist`, `build` — vendored and generated code is not the
  application's own.
- **Ran the LLM and evaluation passes against that single parse** — so the corpus is cloned
  and parsed once, not twice.
- **Skipped files that fail to parse** rather than failing the repository — one syntax error
  should not lose a whole application.
- **Indexed top-level functions and methods of top-level classes.** Nested functions and
  inner-class methods are not indexed — rare enough not to justify the extra scope handling.
- **Deleted each clone after analysis** — the corpus does not fit on disk otherwise
  (~107 GB when retained).
- **Recorded per-repository progress and isolated failures** — a clone or parse failure logs
  and the run continues.

---

## Stage 5 — LLM call-site detection
→ `llm_invokers_all.csv`, `llm_calls_all.csv`, `call_metadata_all.csv`, `llm_tests_all.csv`

- **Built a per-framework dictionary of invocation patterns** (`Wrapper/FrameworkDict.py`,
  76 frameworks recorded, 44 in scope, 468 patterns) — each framework has its own
  invocation API, so a single generic pattern set would be both noisy and incomplete.
- **Matched a file's calls only against the patterns of frameworks that file imports** — so
  a LangChain file is never tested against OpenAI patterns.
- **Collected imports with a full AST walk, not just module level** — lazy imports inside
  function bodies are a common pattern and still make the file a user of that framework.
- **Included only generative-text call surfaces** — embeddings, image, audio, moderation and
  tokenizer endpoints were excluded: different modality, and the determinism knobs the study
  measures do not apply to them.
- **Excluded bare class-name patterns for any framework that has a method pattern** —
  constructing an object (`Agent(...)`) is not invoking a model.
- **Matched method patterns as complete tokens** — `.run` matches `agent.run` but not
  `.run_tool` or `.run_in_executor`.
- **Handled dspy structurally rather than by pattern** — a dspy module is invoked by calling
  the instance (`pred(...)`), so there is no method name to match; names bound from a dspy
  constructor are tracked and the later call on such a name is the invocation site.
- **Tagged, rather than deleted, matches that collide with non-model calls** — a five-tier
  filter (`Wrapper/false_positives.py`) marks calls whose receiver or syntax shows they are
  not model invocations: stdlib and mock receivers (`asyncio.run`), tool and sandbox
  execution (`read_file_tool.invoke`), non-model LangChain runnables (templates, retrievers,
  parsers), storage and cache receivers, and matches that are not the invoked terminal
  segment of the expression. Tagged rows stay in the data and are excluded at analysis time,
  so the filter is auditable and reversible.
- **Exempted frameworks whose invocations are not statically visible** — `omnigent` invokes
  the model in a spawned subprocess and `agentops` is an observability layer; both would
  score zero invocations for reasons unrelated to their real behaviour.

---

## Stage 5 — Call graph (supporting measure only)

The call graph supports one reported measure: how far LLM non-determinism spreads through a
codebase. Everything the study reports as a headline is computed from direct call sites.

- **Built a static call graph per repository with pyan3 and walked it backwards from each
  direct call site** — to measure the **LLM blast radius**: the share of a repository's
  functions that can reach a model call.
- **Measured**: median **0.6%** of functions reach a model call directly; **5.0%** reach one
  transitively, over the 700 analysed repositories with a usable graph.
- **Reported direct and transitive separately** — transitive counts grow with graph size and
  are structurally zero for repositories where no graph could be built, so the two are not
  comparable across repositories.
- **Recorded call-graph health per repository** (`call_graph_health.csv`) — so a repository
  without a usable graph is not read as one with no transitive reach. 958 of 1,033 have a
  usable graph.
- **Made the graph best-effort, never fatal**: unparseable files are dropped before pyan
  runs, and a file pyan itself chokes on is excluded and the analysis retried (pyan is
  all-or-nothing, so one bad file would otherwise zero a whole repository). A repository
  with no graph still yields its direct call sites.

---

## Stage 5 — Determinism metadata
→ `call_metadata_all.csv`

- **Emitted one row per argument of every direct call site**, keeping call sites with no
  arguments as a single blank row — so no site is lost from the data.
- **Recorded per argument**: kind, source text, the variable names read, and whether the
  value is a compile-time literal — literal vs variable distinguishes a setting hard-coded
  at the call from one passed in.
- **Recorded exact line, column and the set of argument variables** — to seed the Stage 6
  slicer whichever granularity it keys on.
- **Tracked eight determinism-relevant keyword arguments** — `temperature`, `top_p`,
  `top_k`, `seed`, `max_tokens`, `frequency_penalty`, `presence_penalty`, `model`.
- **Reported both how many calls set each knob and how many knobs each call sets**, counting
  over all calls so that calls setting nothing appear — most calls set none, which is the
  finding.

---

## Stage 5 — Non-deterministic tests
→ `llm_tests_all.csv`

- **Defined a non-deterministic test as a pytest test that reaches an LLM call** — directly
  or through the call graph.
- **Required both pytest naming conventions**: the file is `test_*.py` or `*_test.py` and
  the function name starts with `test_` — pytest's own default discovery rules.
- **Took the tests as a filtered view of the invoker set** rather than a separate analysis —
  a test is non-deterministic for exactly the same reason any other function is.

### Distance from the model
`Applications/jump_depth.py` → `Q9_test_jump_depth.csv`

- **Measured how many calls separate each test from the function that hits the model** — to
  distinguish a test that calls a model itself from one that reaches one several frames
  away.
- **Recovered the hop count from the existing artifact rather than re-running** — the
  closure is a multi-source BFS in which each function is written once, on first discovery,
  so a transitive row's `reason` names its BFS parent and the parent chain gives the minimum
  distance to the nearest direct call site.
- **Reported the depth as a bounded range, not a point estimate** — 5.6% of transitive rows
  name a parent that was never indexed, which orphans 21.6% of nodes. The `strict` variant
  drops orphans and biases deep; the `lower_bound` variant treats every missing parent as
  depth 0, covers all nodes, and makes each reported depth a floor. The lower bound is
  quoted.
- **Result**: 7.5% of graph-reached tests invoke a model directly, 75% are 3 or more calls
  away.

---

## Stage 7 — Evaluation-framework usage
`pipeline/eval_calls.py` → `eval_calls_all.csv`, `eval_invokers_all.csv`

- **Built a second pattern dictionary for LLM evaluation tools** — DeepEval, RAGAs,
  Giskard, Opik, Arize Phoenix.
- **Matched only patterns that run an evaluation**, not construction of metrics, datasets or
  test suites — constructing an evaluator is not evaluating.
- **Ran it against the same parse as the LLM pass** — no second clone or parse.

---

## Stage 6 — Per-variable slicing
`pipeline/slice_repo.py`, `pipeline/per_variable_pdg_slicer.py`

- **Built a Joern CPG per repository and emitted per-variable SubPDGs** for the functions
  that reach an LLM call — to isolate the computation that produces each LLM argument.
- **Sliced only the LLM-invoker closure, not the whole repository** — the rest of the code
  is not what the study is about.
- **Followed the variable's data-dependence edges, then added incoming control-dependence
  predicates**, mapped back to source lines and emitted standalone parseable subprograms —
  metadata keeps PDG-selected lines separate from lines added only to make the snippet
  self-contained.
- **Excluded `.py` files over 1.5 MB from the CPG** — these are generated data embedded as
  literals; they exhaust Joern's heap at any size and never contain an invoker.
- **Escalated the Java heap per repository (8→12→16→24 GB) instead of setting a flat
  ceiling** — almost every repository finishes at 8 GB, and a flat 24 GB would hand that to
  all of them.

---

## Adjudication and counting

- **Built a per-application audit sheet** (`application_audit.csv`, one row per population
  repository, 41 columns) — machine passes fill their own columns; `in_scope` and `notes`
  are hand-edited and preserved by every pass.
- **Scoped repositories by hand where the automated signals could not decide**, recording
  the reason in `notes` — the matched search token is not evidence of an import, so scope
  had to be decided against what the clone actually contains.
- **Required evidence before excluding**: a matched token counts as a collision only when
  the clone does not import it, checked against the Stage-1 discovery record.
- **Made every exclusion a filter, never a deletion** — an excluded repository keeps its row
  and its raw data; only its `in_scope` value changes.
- **Recorded every exclusion criterion, dated, in `EXCLUSIONS.md`** — so each cut is
  attributable and reversible.
- **Made `in_scope` three-valued** — so the deliberately unmeasured tail is not confused
  with junk:
  - blank — in scope; counted everywhere.
  - `uncovered` — a real LLM application on a framework outside the top 20; excluded from
    measured statistics, retained in the coverage denominator, since quantifying that tail
    is what the coverage figure is for.
  - `0` — not an agentic application; excluded from everything.
- **Read the cuts from one module** (`pipeline/cuts.py`) — so an edit to `in_scope` moves
  both sides of every ratio and every script agrees.
- **Excluded false-positive-tagged calls and patterns no longer in the dictionary at
  analysis time** — so results reflect the current pattern set without re-running the corpus.
- **Reported the population as a waterfall** (`pipeline/waterfall.py`), where a repository
  leaves at the first step that drops it — so the drops sum and no repository is counted
  against two reasons.

### Population, as measured 2026-09-01

| step | dropped | remaining |
|---|---:|---:|
| GitHub code-search candidates | | 6,446 |
| no trustworthy framework name in the match | 5,382 | 1,064 |
| framework / eval self-repositories | 9 | **1,055 — population** |
| clone failed | 22 | 1,033 |
| code quality, and one known non-LLM app | 3 | 1,030 |
| not an agentic application (collisions, frameworks, platforms) | 227 | 803 |
| uncovered tail, outside the top-20 frameworks | 50 | **753 — analysed** |
| no LLM call site found | 58 | **695** |

### The observed population

- **Restricted the study to applications in which a framework LLM call site was detected** —
  the object of study is how framework-mediated model calls are written and tested, so an
  application that never makes one has nothing to observe. This is a scoping boundary, not
  a result: prevalence of LLM usage is therefore not a reported figure.
- **Excluded raw provider-SDK call sites from observation entirely** — `openai` and
  `anthropic` calls are detected and retained in the artifacts but are neither measured nor
  reported.
- 109 of the 753 analysed applications fall outside the boundary: 51 make only raw-SDK
  calls, 58 have no detected call site of any kind (of those, 11 import `litellm`, for which
  no patterns exist).
- The boundary is drawn by the detector, so the population is applications in which a
  framework call site **was detected**, not applications that make one.

### Headline figures

| | |
|---|---:|
| **applications observed** | **644** |
| framework call sites | 23,990 |
| call sites setting **no** determinism knob | **98.9%** |
| … setting `temperature` · `model` · `seed` | 0.1% · 0.4% · 0.0% |
| direct non-deterministic tests | 9,700, in 233 apps (36%) |
| applications using an eval framework | 38 (6%) |
| LLM blast radius (median, usable graph) | 0.6% direct · 5.0% transitive |
