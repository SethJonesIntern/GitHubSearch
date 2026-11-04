# GitHubSearch — project instructions

Repository workflow: `main` is the shared integration branch; create feature branches
from current `main` and merge through pull requests. See CONTRIBUTING.md for CI and
the history-preserving consolidation. Do not merge old branch contents over the
authoritative research tree merely to make branch names agree.

Empirical software-mining study of **correctness guardrails in agentic applications**
mined from GitHub (reframed by Seth, 2026-11-03). Discover agentic frameworks → search
GitHub using framework import patterns → locate in-application framework LLM invocations
→ identify associated test-suite correctness checks and runtime checks with intervention.
Primary unit: production application methods containing direct observed framework calls.
Corresponding tests come from all identified tests, not only direct LLM-calling tests.
Mapping establishes correspondence, not a correctness guardrail. Guardrail detection
and prevalence results are pending; the historical analysis below is infrastructure.
README.md describes the current story; METHODOLOGY_REVIEW.md section J reviews figures.

## Contribution claims (confirmed by Seth, 2026-11-04)

Use README.md's "Contributions and their current status" as the canonical public
statement. Preserve all four contributions when describing the project:

1. **Framework LLM invocation inventory**: the curated API/pattern list in
   `Wrapper/FrameworkDict.py` is a separate research artifact, not merely an internal
   helper or the list of discovered frameworks. Full versus scoped registry and
   known false positives/exclusions must stay distinguishable. Do not claim exhaustive
   API coverage or validated precision without evidence.
2. **Prevalence measurements**: corresponding tests and correctness guardrails for
   production methods containing direct observed framework calls. Initial test
   correspondence exists; validated test-suite/runtime guardrail prevalence does not.
3. **Check characterization**: possibly derive categories/definitions from inspected
   check contents. This is exploratory; do not describe candidate categories as findings.
4. **Reproducible pipeline and dataset**: the existing analysis infrastructure and
   artifacts, with guardrail detection still to be developed and validated.

Primary scope: existence/prevalence. Optional deeper scope: contents and taxonomy.
Distinguish the property checked from the evaluation mechanism. An assertion may wrap
an LLM judge; an evaluation library may perform a deterministic check. Neither tests,
library adoption, invocation patterns nor determinism settings alone prove guardrails.
Attribute pyMethods2Test to its authors; our integration/results are contributions,
not the published mapper itself. Keep method denominators separate from call/test counts.

The full Stage-5 run is done (1,033 of 1,055 processed). Current work: data quality, and
test-to-code traceability. This file is the single handoff: keep it current instead of
writing dated `HANDOFF_*.md` files.

## Where the study stands (measured 2026-09-18)

| | |
|---|---:|
| population | 1,055 |
| analyzed (prevalence denominator) | 753 |
| with ≥1 LLM call site | 695 |
| **observed apps** (≥1 *framework* call site; the paper's population) | **644** |
| framework call sites | 23,990 |
| call sites setting **no** determinism knob | 98.9% (`temperature` 0.1%, `seed` 0) |
| **direct ND tests** | **9,700 in 233 apps** |

The paper's numbers use the observed scope: `make_figures.POP` / `.CALLS` and
`focal_map.observed_tests` for historical direct-test counts. For the method-first
baseline use all identified tests; do not restrict it to `observed_tests`.
Import the existing scope definitions; don't restate them. `analyze.py`
covers the wider 753 and gives 9,701 ND tests in 234 apps. The extra one is
`Portkey-AI/portkey-python-sdk`, whose only calls are false positives.

Regenerate: `py -3.14 -m pipeline.waterfall` (population + pending queues),
`py -3.14 Applications/analyze.py` (report + `analysis/*.csv`),
`py -3.14 Applications/make_figures.py` (F1–F3 + `FIGURES.md`).

## Scope rules — settled, do not re-litigate

1. **An agentic application is one that imports an agentic framework** (2026-08-27).
   A repo reaching a model only through a raw SDK (`openai`, `anthropic`, `ollama`,
   `litellm`) or raw HTTP is out. Stage 2 only searched framework names, so such a repo
   can only have arrived by a token collision.
2. **Raw provider SDKs are out of every number** (2026-09-18). `openai`/`anthropic` rows
   stay in the raw CSVs but are never observed, measured or reported: no calls, knobs,
   invokers or ND tests. This overrides the 08-27 note that raw-SDK calls inside agentic
   apps "stay measured". Raw Gemini is undetected, which is moot for the same reason.
3. **Direct only; transitive is out of scope** (2026-09-18, until Seth says otherwise).
   Transitive's sole permitted use is the **blast radius** of LLM use (share of an app's
   functions that can reach an LLM call). See README "Scope rule" and `EXCLUSIONS.md` §18.
   `direct`/`transitive` is our pyan term, not a pyMethods2Test one.
4. **Stage 2 is GitHub code search, not import matching.** `matched_frameworks` is the
   token that surfaced a repo, and it hits substrings: `camel`→camelCase,
   `agno`→agnostic, `notte`→`@nottest`, `clai`→claim, `omnigent`→`OmniGe(nt)ransformer`.
   **Never decide scope from `matched_frameworks`**; use `frameworks_imported`.
5. **A matched token is a collision only when the clone does not import it.** Check
   `frameworks.csv` (the Stage-1 set) before cutting on that basis. It saved `IBM/ares`
   (`deepteam`) and `fetchai/fetchai` (`uagents_core`); `notte_sdk` is *not* notte.

**Trap, hit twice:** `openai` and `anthropic` are keys in `FRAMEWORK_CALLS`, so a
raw-SDK-only repo *does* have invokers. That isn't a contradiction; check the `reason`
column of `llm_invokers_all.csv` for which patterns fired.

## Docs

| doc | what it settles |
|---|---|
| `EXCLUSIONS.md` | **the ledger**: every cut, exemption, pattern cut and scope decision, dated, with the enforcing code. Add a row whenever you exclude anything. §18 = transitive sweep |
| `METHODOLOGY.md` | what was done, stage by stage, with the paper's figures |
| `METHODOLOGY_REVIEW.md` | parked methodological issues; the agenda for the review pass, nothing settled |
| `README.md` | public overview, headline, the direct-only scope rule |
| `COVERAGE_ANALYSIS.md` | the three scope buckets and the 90.1% coverage claim |
| `RUNBOOK.md` | running Stage 5/6, flags, stop/resume without duplicate rows |
| `pipeline/next_runs/README.md` | targeted re-runs and why they need prep |
| `CALL_GRAPH_EXCLUSIONS.md` | why some repos have no pyan graph |
| `PIPELINE.md` | stage design. Its "CURRENT STATE" (2026-06-25) is stale; ignore it |
| `pipeline/third_party/pymethods2test/NOTICE.md` | provenance of the vendored tool |

## The audit sheet

`pipeline/artifacts/application_audit.csv`: one row per population repo (1,055). Each
pass owns its columns. `in_scope` / `notes` are hand-edited and always preserved.

```
py -3.14 -m pipeline.audit_apps                     # the sheet + counts   (PC only, see below)
py -3.14 -m pipeline.audit_imports --scan           # frameworks_imported -> in_scope (~15 min)
py -3.14 -m pipeline.audit_framework_check --scan   # framework-vs-app (~13 min)
py -3.14 -m pipeline.audit_zero_invokers            # why a repo has no invokers
py -3.14 -m pipeline.audit_framework_triage         # framework_triage.csv (reads clones)
```

| `in_scope` | analyzed stats | coverage denominator |
|---|---|---|
| *(blank)*: in scope | counted | counted |
| `uncovered`: real agentic app on a framework outside the top-20 | out | **in** (the tail the 90.1% measures) |
| `0`: cut | out | out |

`pipeline/cuts.py` reads `in_scope`; `analyze.py`, `keep_frequency.py`,
`plot_coverage.py` and `make_figures.py` read `cuts.py`. **Editing `in_scope` moves every
statistic on the next run.**

- **`audit_apps` needs the slices.** It derives `slice_status` from
  `os.scandir(SLICES_DIR)`; on a machine without `pipeline/artifacts/slices/` it blanks
  ~780 rows. Recover with `git checkout -- pipeline/artifacts/application_audit.csv`.
- `nd_tests` in the sheet is direct only. `nd_tests_transitive` is a raw column that
  nothing reads.
- Supporting sheets: `framework_triage.csv` (all 1,055 ranked by framework-likelihood;
  the score is a ranking aid, trust only the top ~58), `imported_by.csv`,
  `analysis/zero_call_triage.csv`, `analysis/no_framework_no_call.csv`.

## Focal mapping — pyMethods2Test

Method-first comparison branch: `codex/method-test-mapping-baseline`.
`py -3.14 -m pipeline.method_test_baseline` compares production methods with a
direct observed framework call against other production methods in the same
repos. Needs full focal CSVs + clones on the PC; see RUNBOOK.md's method-to-test
baseline section. The full PC run is complete; tracked results are in
`artifacts/method_test_baseline/summary.json` and `by_repo.csv` (relative to `pipeline/`).
644 considered, 589 included, 571 with both method groups. Paired pooled rates:
220/7,642 LLM-containing methods (2.88%) versus 72,527/1,463,857 other methods (4.95%).
Equal-weight repo means: 3.83% versus 4.42%; mean paired difference -0.60 percentage
points. 55 exclusions: 3 crashed runs, 48 source-parsing failures, 4 unmatched focal
definitions. Whole-repo exclusions for template/example parse failures need review;
these are preliminary correspondence results, not proven absence of tests/guardrails.
Earlier 262/18,911 counts included test code and must not be presented as
production-method test correspondence.

**Decided 2026-09-15: focal mapping uses pyMethods2Test** (Abdelmadjid & Dyer, MSR 2025,
Zenodo `14264519`, Apache 2.0), not our `pipeline/focal_map.py` (26.5% method-level match,
superseded; its `observed_tests()` is still the ND-test scope definition).

- Vendored **unmodified** in `pipeline/third_party/pymethods2test/`. Byte-identical to
  Zenodo (verified 2026-09-18).
- `py -3.14 -m pipeline.pym2t_focal` runs it over all 644 observed apps, reusing the
  per-repo JSON in `artifacts/pym2t/` (gitignored); `--rerun` re-runs the tool. It replays
  their loop with their own functions to record per-test detail, and checks the replay
  against their CLI output (0 mismatches).
- Outputs: `pym2t_nd_tests.csv` (direct ND tests → focal status) and
  `pym2t_focal_calls.csv` (call site → inside a focal method?) are tracked.
  `pym2t_focal_tests.csv` (1.2M tests, 290 MB) and `pym2t_runs.csv` are gitignored.
- **Environment:** their pinned `python-Levenshtein~=0.26.1` has no cp314 wheel. Install
  `fuzzywuzzy~=0.18.0` + modern `Levenshtein`. The driver refuses to run if `fuzz` has
  fallen back to `difflib`, which changes every ratio against their `> 50` cutoff. This
  deviation belongs in the methodology.

**How it works** (read from their source; the paper's HTML summary is wrong):
- **Focal file:** source files whose module the test imports. If none, a basename match,
  and every `.py` file (tests included) is a candidate. So it never returns "no focal
  file". Only `file_rule == "import"` is real evidence.
- **Focal method:** a name the test body calls. The test name must end with it, otherwise
  any name with `fuzz.ratio > 50`.
- **A test is only seen if** its file has a top-level `import pytest`/`unittest`, and it
  calls ≥1 name defined in the repo.

**Two different questions, two tools.** Our detector decides whether a test *calls* an
LLM (an ND test: a pytest test whose own body matches a framework invocation pattern).
Calling an LLM does not make the LLM the test's focus. **pyMethods2Test decides what the
test targets.** A test is **LLM-focal** only when its pyMethods2Test focal method contains
a framework LLM call site. Their focal methods are always repo-defined, so a test whose
only LLM call goes straight to the framework API can never be LLM-focal.

**Results on the 9,700 direct ND tests:**
- 19.1% are never identified by the tool. Mostly plain-`assert` pytest files with no
  `import pytest`.
- 76.9% get a focal file only; 49.4% via import.
- **4.0% get a focal method** (391 tests, 57 apps). 56% of those are
  `haystack-core-integrations`, 234 are `run`/`run_async`.
- **LLM-focal: 36 tests (0.4%) in 16 apps.** 10 are `haystack-core-integrations`;
  23 exact / 13 fuzzy method matches; 24 via import / 12 basename file matches.
- **Only 25 of 23,990 call sites (0.1%) lie inside the focal method of an ND test.**
- 3 repos crash on non-UTF-8 files (dd-trace-py, artiphishell,
  astrbot_plugin_self_evolution).

Not yet done: hand-check a precision sample of the 391; report with and without
haystack-core-integrations.

## Slice analysis — Stage 6a/6b (2026-09-21, branch `slice-index`)

Stage 6 left 4.3 GB of per-repo `programs.jsonl` that nothing read. Two scripts make it
queryable; neither re-slices nor touches Joern. **PC only** — 6a reads `artifacts/slices/`,
6b reads 6a's gitignored index.

```
py -3.14 -m pipeline.slice_index     # ~5 min  -> slice_index.csv + 3 more
py -3.14 -m pipeline.slice_calls     # ~4 min  -> slice_call_hits.csv + 2 more
```

| output | tracked | what |
|---|---|---|
| `slice_index.csv` (994 MB) | no | 1 row / subprogram, 2,955,122 rows |
| `slice_pdg_lines.csv.gz` | no | long form `(slice_id, line)`; the join key |
| `slice_code.jsonl.gz` | no | content store, 1 row / unique `code_sha256` |
| `slice_files.csv`, `slice_index_summary.json` | yes | 77,901 sliced files; corpus totals |
| `slice_call_hits.csv` (23 MB) | yes | 1 row / slice-call pair, 58,478 rows |
| `slice_calls_by_call.csv`, `slice_calls_summary.json` | yes | 1 row / observed call site |

**Reconciled exactly** against the slicer's own totals: 2,955,122 subprograms, 1,089,429
unparseable. 6b takes its scope from `make_figures` (POP 644, CALLS 23,990).

- **93.8% of the 23,990 observed call sites lie inside ≥1 slice** (96.1% of the 23,401 in
  a sliced repo). Only 1.73% of slices touch a call.
- **63.1% of subprograms parse.** The emitted body is truncated otherwise; the `pdg_lines`
  metadata is unaffected, so the join is unharmed but body-level analysis must filter on
  `parses`.
- **Exact cross-repo clones are thin**: 43,629 of 2,227,979 distinct bodies (1.96%) occur
  in >1 repo. Mining shared idioms off `code_sha256` alone will not carry a result.
- **`relation` is not a prompt/response split.** The criterion is the variable's *last
  use*, so a prompt argument's criterion is the call line. Use `role` (receiver 29.3% /
  argument 33.9% / binding 36.9%), read off `callable` + `call_source`.
- **Split test from production with `code_kind`**, never a path check:
  `test` | `test_support` | `production`, on every slice, hit and call site.
  `test` = Stage 5's own rule (`engines.is_test_file` + `is_test_function`), and matches
  the ND tests (9,702 vs 9,700, overlap 9,693). `test_support` (fixtures, helpers,
  `conftest.py`, the rest of `tests/`) is what Stage 5 calls production; keep it out of
  production claims. Slices 35.0 / 3.4 / 61.6%; call sites 12,519 / 549 / 10,922.
  Tests slice the agent object (receiver 43.3% of pairs); production slices prompt and
  response (argument 40.6%, binding 41.1%).

Unsettled, parked in `METHODOLOGY_REVIEW.md` §I (items 31–36): no precision sample for
`role`; the per-line join loses bindings on multi-line calls; **"no binding" ≠ "result
unused"** (at most 4,068 bare-statement call sites: 3,248 test, 84 test support, 736
production, before deducting side-effecting patterns and `.invoke` false positives);
haystack contamination carries through; the index spans 788 dirs against the observed 644.

## To look at: found 2026-09-18, nothing decided or changed

- **Haystack `.run` matches non-LLM components.** Haystack's patterns are just
  `.run` / `.run_async`, and every Haystack component has `.run()`. By receiver name
  (unverified), ~1,356 of 2,603 haystack call sites (52%, 18 apps) are retrievers (490),
  embedders (333), converters (218), rankers (145), fetchers, cleaners, etc. All 152 calls
  setting `top_k` in F2 are retrieval `top_k` (haystack retriever/ranker 104, mem0
  `.search` 47), not sampling. Also touches haystack ND tests (2,218).
- **Chained calls share a call ID.** `call_id` is `file::line::col`, so in
  `llm.invoke(x).content.strip()` the real call and the chained `.strip()` collide.
  106 IDs (87 in the observed scope). `llm_calls_all.csv` kept the chained row, which is
  flagged `tier2_nonterminal`, so the real call is dropped from the call counts but kept in
  `call_metadata_all.csv`. That's why F2 says 24,077 call sites and F3/METHODOLOGY say
  23,990. Counting them would give 645 apps (+ `Portkey-AI/portkey-python-sdk`) and
  24,079 call sites.
- **F2's committed caption** calls `model` the most often set parameter; its own table has
  `top_k` (152) above `model` (92).

## Open work, in value order

- **H4. Fix `shallow_clone`** (`Wrapper/transitive_invokers.py:58`). 22 repos (20 in
  scope) "failed to clone" over a `:` in a filename, illegal on Windows. Git fetches
  everything, checkout aborts, and the function returns `False`, but the tree is on
  disk (`Azure/plan-search-chatbot`: all 36 `.py` present). Fix: on
  `CalledProcessError`, if `dest` holds `.py` files, warn, record `clone_partial`, and
  return `True`; genuine failures (auth, 404, timeout) still return `False`. Then
  re-queue. ⚠️ `batch_call_metadata` `rmtree`s the checkout before cloning, so **do not
  re-queue before the fix**. Truly gone: `agno-agi/pal`, `holetron/hindsight-mempalace`,
  `openJiuwen-ai/agent-store`. Denominator returns to ~773.
- **`clai` → `pydantic_ai` in `keep_frequency.EXTRA_MEMBERS` (line 46)**: still there
  and wrong. Inflates pydantic_ai 79 → 127 apps, #4 → #2. Remove it.
- **H5. Queue C: 56 repos import a framework but have no call site.** Detector
  questions, not cuts. The waterfall's 100% for C is arithmetic. ~10 import `litellm`
  (not in `IN_SCOPE_FRAMEWORKS`, can't produce a call site), plus the H4b shape.
- **H4b. Split-receiver detector gap.** `storyAura/astrbot_plugin_biliVideo` calls
  `await provider.text_chat(...)` on a receiver from `context.get_using_provider()` held
  in a wrapper. Missed because `.get_using_provider` was cut (EXCLUSIONS §4). Held, not
  cut (§15). Size how many of queue C share it.
- **H4d. Registry gaps.** `uagents_core` and `deepteam` are Stage-1 discovered but not
  `FRAMEWORK_CALLS` keys (only `uagents` is). Audit every `frameworks.csv` `import_names`
  against `FRAMEWORK_CALLS`.
- **H4c. 7 raw-SDK-only `uncovered` repos.** They have no invokers, so only
  `imports_matched_fw` can judge them. `audit_zero_invokers.py:154` limits `targets` to
  `zero_invoker == "1"`; widen it to `uncovered` rows (EXCLUSIONS §17b-ii).
- **Stage 6 slicing vs the direct-only rule.** The slicer is fed the whole invoker
  closure, transitive included (EXCLUSIONS §18). Decide: filter slice analysis to direct
  invoker functions, or rule it blast radius.
- **Joern: 10 repos still fail at 24g** (5 OOM, 4 process-exit). Heap isn't the
  binding constraint; needs a different approach.
- **H7. `audit_framework_triage --pypi`**: the one independent framework signal never
  switched on (desktop, it reads the clones).
- **H8. Second pyan pass with `PYAN_TIME_BUDGET_SEC=0`**, time-boxed subset only. Low
  priority now that transitive only feeds blast radius. PowerShell:
  `$env:PYAN_TIME_BUDGET_SEC = "0"` on its own line.
- Unverified whether done: Queue D (~53 lower-weight framework/library candidates,
  §12); `prepare_rerun` has no guard against `--apply` on the wrong CSV; no EXCLUSIONS row
  for 3 zero-Python repos dropped from the run queue (`wesleysimplicio/hermes-turbo-agent`,
  `erfanzar/Xerxes-Agents`, `memory-graph/memory-graph`).

## Machines

- **PC (desktop)** has the clone tree (`pipeline/repos/`, ~107 GB, 1,056 repos) and
  `artifacts/slices/`. Needed for: reading source to justify a cut, every `--scan`,
  `audit_framework_triage`, `batch_call_metadata`, slicing, `audit_apps`, `pym2t_focal
  --rerun`.
- **Laptop** has neither. The cut loop still works there: hand-edit `in_scope` →
  `analyze.py` / `keep_frequency.py` / `plot_coverage.py`. Never run `audit_apps` there.
- **Git LFS** tracks `llm_invokers_all.csv` (172 MB) and `llm_tests_all.csv` (87 MB).
  Without `git lfs install` you get pointer stubs and scripts fail confusingly. Free tier
  is **1 GB bandwidth/month**, and one fresh clone is ~259 MB of LFS. `git pull` into an
  existing checkout; don't re-clone. `git pull` before hand-editing the sheet on either
  machine; git can't merge two machines' edits to it.

## Conventions

- **`py -3.14` always** (pyan3 requires it).
- **Cuts are filters, never deletions.** A cut repo keeps its row and raw data; only
  `in_scope` changes. Every criterion lives in code with a comment and gets a dated row in
  `EXCLUSIONS.md`.
- **Evidence before cutting.** Read the actual matched code. Inference has been wrong
  repeatedly: omnigent, connectonion, haystack, clai, notte_sdk, deepteam.
- Distinguish **measured / projected / assumed** when reporting numbers.
- Prefer giving Seth a command to run over backgrounding a long job.
- Never run `batch_call_metadata` without `--resume`: it deletes every artifact and
  restarts the whole run. Re-runs: `pipeline/next_runs/README.md`.

## Gotchas

- **The Grep tool silently skips `pipeline/repos`** (gitignored). Use Bash `grep` or
  `rg --no-ignore` for the clones. A recursive `grep -r` from the repo root also walks
  `Applications/repo_cache` and the clones and will time out; scope it.
- Full-corpus scans take ~15 min at 8 workers. Use one alternation regex over the names
  you want, not match-everything-then-filter (90 min vs 15).
- Windows: `ProcessPoolExecutor` needs an `if __name__ == "__main__"` guard. Pass Joern as
  the `.bat`; bare `joern` gives `WinError 193`.
- Set `PYTHONIOENCODING=utf-8` before `analyze.py` if it dies on a non-ASCII description.
- 107 `__pycache__/*.pyc` files are tracked. Running or compiling a script dirties them;
  restore with `git checkout` rather than committing them.
- `core.autocrlf=true`, so LF-written files produce a harmless CRLF warning on commit.
