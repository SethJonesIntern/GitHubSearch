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
