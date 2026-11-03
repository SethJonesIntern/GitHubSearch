# Runbook: Stage 5 (invoker/call analysis) + Stage 6 (slicing)

How to run the analysis over the application population, what each flag does, and
which gotchas will bite on Windows.

*The full population run completed 2026-08-17 — 1,035 repos processed of 1,055.
Re-running from scratch is not normally what you want; see "Targeted re-runs".*

## The command

```powershell
py -3.14 -m pipeline.batch_call_metadata `
  --input pipeline/artifacts/applications_slim.csv `
  --slice `
  --joern-parse C:/Users/Seth/joern_install/joern-cli `
  --joern C:/Users/Seth/joern_install/joern-cli/joern.bat `
  --keep-clones `
  --resume
```

One line (bash / no backticks):

```
py -3.14 -m pipeline.batch_call_metadata --input pipeline/artifacts/applications_slim.csv --slice --joern-parse C:/Users/Seth/joern_install/joern-cli --joern C:/Users/Seth/joern_install/joern-cli/joern.bat --keep-clones --resume
```

**The input is `applications_slim.csv` (the full 1,055).** Scope is decided *after*
analysis via the `real_ai_app` / `analyzed` flag columns — never by filtering the
run set, because repos matched only to out-of-scope tokens still make in-scope
raw-SDK calls. See `COVERAGE_ANALYSIS.md`.

## Why each flag

| flag | why |
|------|-----|
| `py -3.14` | pyan3 2.6.0 (the transitive call graph) requires Python 3.14. |
| `--input .../applications_slim.csv` | the full population. |
| `--slice` | Stage 6: build a Joern CPG per repo, emit per-variable SubPDGs for its LLM-invoker functions. Without it you get the invoker/call CSVs only, much faster. |
| `--joern-parse C:/Users/Seth/joern_install/joern-cli` | a directory works; the resolver picks the Windows `.bat`. |
| `--joern .../joern.bat` | **must be the `.bat`.** The bare `joern` path is the extension-less Unix script and dies on Windows with `WinError 193`. |
| `--keep-clones` | keep each checkout under `pipeline/repos/` (lets you inspect source and re-analyze without re-downloading). |
| `--resume` | skip repos already in the progress file. Use after any interruption. |
| `--keep-cpg` *(optional)* | also save each repo's `repo.cpg` under its slice dir. Costs disk. |

**A fresh run without `--resume` deletes and regenerates the artifact CSVs** and
re-clones every repo. Almost always you want `--resume`.

## Prerequisites

- Python 3.14 with `pyan3==2.6.0` (`py -3.14 -m pip install pyan3==2.6.0`).
- Joern at `C:/Users/Seth/joern_install/joern-cli` (adjust both `--joern*` paths if it moves).

## Stopping and resuming safely

Per repo the driver writes rows **before** it marks the repo processed, with
slicing in between. Killing the run during slicing leaves rows on disk for a repo
that is not in `processed` — on `--resume` it is re-analyzed and its rows are
**duplicated**. After any hard stop, check:

```
py -3.14 -c "import csv,json,collections; p=json.load(open('pipeline/artifacts/.batch_progress.json')); done=set(p['processed']); c=collections.Counter(r['repo'] for r in csv.DictReader(open('pipeline/artifacts/call_graph_health.csv',encoding='utf-8'))); print('dups:',{k:v for k,v in c.items() if v>1}); print('orphans:',[r for r in c if r not in done])"
```

Both empty → safe to resume. An orphan → add that repo to `processed` in
`.batch_progress.json` (you lose only its slice) or delete its rows first.

## Targeted re-runs

To re-analyze a subset after a bug fix: remove those repos from `processed` in
`.batch_progress.json`, delete their existing rows from every `*_all.csv`, pass a
curated CSV as `--input`, and re-run. Skipping either of the first two steps gives
you silent skips or duplicate rows.

## Outputs

All under `pipeline/artifacts/`:

- **Stage 5:**
  - `llm_invokers_all.csv` — direct + transitive LLM invokers (`kind` column).
  - `llm_calls_all.csv`, `call_metadata_all.csv` — call sites + per-argument metadata (determinism knobs).
  - `llm_tests_all.csv` — pytest-conventioned invokers. **Note:** this is a subset of `llm_invokers_all.csv`, and it measures *static reachability*, not confirmed non-determinism — a transitive row may never execute the call, and a direct row may be mocked.
  - `eval_invokers_all.csv`, `eval_calls_all.csv` — eval-framework usage.
  - `call_graph_health.csv` — per repo `cg_source` (`pyan` / `pyan_resilient` / `none`), `graph_usable`, edge counts, excluded files. Tells you whether a low transitive count is real or a pyan failure.
- **Stage 6:** `pipeline/artifacts/slices/<repo>/` — `programs.jsonl` (per-variable SubPDGs) + a per-repo summary.
- **Progress:** `.batch_progress.json` — `processed` / `failed` / `slice_failed`.

## After the run

```
py -3.14 Applications/analyze.py
```

writes per-question CSVs to `pipeline/artifacts/analysis/` and prints the
population, test, determinism-knob, and call-graph-health sections.

## Gotchas

- **`py -3.14` always** — pyan3 needs it.
- **Windows Joern**: pass the `.bat`, or `WinError 193`.
- **Slicing is the slow part** (a CPG build per repo). Files >1.5 MB are auto-hidden
  before parse to avoid Joern OOM; heap escalates 8→12→16 GB only on OOM. Joern is
  skipped entirely for repos with no LLM invokers.
- **A slice failure is isolated** — recorded under `slice_failed`, and the repo's
  already-written metadata is kept.
- **Cuts are filters, never deletions.** Raw CSVs and the full `FrameworkDict` are
  preserved; criteria live in code comments and `EXCLUSIONS.md`. Keep it that way.
- **Grep tooling**: `pipeline/repos` is gitignored, so ripgrep-based tools skip it
  silently and report a false "not found". Use `rg --no-ignore` on the clones.
- To wipe kept clones: `py -3.14 -m pipeline.clean_clones` (`--dry-run` to preview).

## Method-to-test baseline (branch codex/method-test-mapping-baseline)

Run on the PC holding `pipeline/repos/`, `pipeline/artifacts/pym2t_focal_tests.csv`
and `pipeline/artifacts/pym2t_runs.csv` from the completed focal-mapping run:

```powershell
git fetch origin
git switch codex/method-test-mapping-baseline
git pull --ff-only
git lfs pull
py -3.14 -m pipeline.method_test_baseline
```

The script reads existing clones and mappings; it does not rerun Joern or
PyMethods2Test. It also reads the canonical tracked inputs used by
`Applications.make_figures.POP/CALLS`; use the existing analysis environment.
If the focal outputs are missing, regenerate them with
`py -3.14 -m pipeline.pym2t_focal` on that PC first (see CLAUDE.md).
`--limit 5` is a smoke run only. `--repos-dir`, `--artifacts-dir` (focal inputs),
and `--output-dir` override local paths; observed scope still comes from this
checkout's canonical POP/CALLS inputs.

The denominator is **production functions/methods**, inventoried by AST from the
saved clones (including nested and async defs, excluding synthetic constructors).
The same `slice_index.code_kind` rule excludes tests and test support. A method
is LLM-containing when an observed, direct framework call lies in its innermost
function body; `other` means no such detected call, not necessarily no indirect
LLM behavior. Calls in decorators/default expressions are not function-body calls;
unjoined production calls are flagged for review.

Each method counts once. Its numerator flag is whether at least one test maps to
that exact file/definition line/name/end line, using all identified tests, not
only tests that call an LLM themselves. The report also gives exact-name-only
mapping rates as a sensitivity check alongside the exact+fuzzy rates. These are
heuristic test correspondences, not execution coverage or proof of absent tests.

Outputs under `pipeline/artifacts/method_test_baseline/`:

- `summary.json`: method counts/rates for LLM vs other methods, pooled over all
  complete repos and separately over repos having both groups; equal-weight
  per-repo averages and paired percentage-point differences; exclusion diagnostics.
- `by_repo.csv`: both denominators, mapped counts, rates and difference per repo.
- `methods.csv`: one row per eligible method with mapped-test counts (gitignored).

Missing clones, failed/incomplete focal runs, replay mismatches, inconsistent
mapping counts, source parse failures and unjoined production calls/focals exclude
the whole repo from both groups, with a recorded reason. Successful `no_tests`
runs remain eligible with zero mappings. Inspect exclusions before interpreting
results; the retained subset may differ from the full observed population.
Line/name/end checks detect some snapshot drift but do not verify source hashes.
The raw tool may also have skipped source files; this baseline inventories parseable
clone source independently and does not make its denominator depend on successful
test identification. Existing name/location heuristics may miss unconventional tests.

Return `summary.json` and `by_repo.csv` for review; individual methods can be inspected
on the PC. This replaces the earlier unfiltered 262/18,911 estimate for any
production-only claim. It does not overwrite existing F4 figures or historical results.
