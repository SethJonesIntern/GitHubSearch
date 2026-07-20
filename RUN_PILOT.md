# Runbook: pilot analysis + slicing (top-92, 5 repos/framework)

How to re-run Stage 5 (invoker/call analysis) **and** Stage 6 (slicing) over the
92-repo pilot set. Copy the command, run it, inspect the outputs.

## The command

```powershell
py -3.14 -m pipeline.batch_call_metadata `
  --input pipeline/artifacts/pilot_applications.csv `
  --slice `
  --joern-parse C:/Users/Seth/joern_install/joern-cli `
  --joern C:/Users/Seth/joern_install/joern-cli/joern.bat `
  --keep-clones
```

One-line (bash / no backticks):

```
py -3.14 -m pipeline.batch_call_metadata --input pipeline/artifacts/pilot_applications.csv --slice --joern-parse C:/Users/Seth/joern_install/joern-cli --joern C:/Users/Seth/joern_install/joern-cli/joern.bat --keep-clones
```

If it dies partway through, add `--resume` to continue from where it stopped
(keeps rows/slices already written, skips repos already in the progress file):

```
py -3.14 -m pipeline.batch_call_metadata --input pipeline/artifacts/pilot_applications.csv --slice --joern-parse C:/Users/Seth/joern_install/joern-cli --joern C:/Users/Seth/joern_install/joern-cli/joern.bat --keep-clones --resume
```

## Why each flag

| flag | why |
|------|-----|
| `py -3.14` | pyan3 2.6.0 (the transitive call graph) requires Python 3.14. |
| `--input .../pilot_applications.csv` | the curated 92-repo set (5 per framework). Omit to run the full `applications.csv`. |
| `--slice` | turns on Stage 6: build a Joern CPG per repo, emit per-variable SubPDGs for its LLM-invoker functions. Without it you only get the invoker/call CSVs. |
| `--joern-parse C:/Users/Seth/joern_install/joern-cli` | dir works; the resolver picks the Windows `.bat`. |
| `--joern .../joern.bat` | **must be the `.bat`.** The bare `joern` path is the extension-less Unix script and dies on Windows with `WinError 193`. |
| `--keep-clones` | keep each checkout under `pipeline/repos/` after analysis (re-runs don't re-download; lets you inspect source). |
| `--resume` | skip repos already processed; use to continue an interrupted run. |
| `--keep-cpg` *(optional)* | also save each repo's `repo.cpg` under its slice dir, for inspecting the CPG directly. Costs disk. |

A fresh run (no `--resume`) **deletes and regenerates** the artifact CSVs, and
re-clones every repo (the checkout is wiped and re-cloned even if it already exists).

## Prerequisites

- Python 3.14 with `pyan3==2.6.0` installed (`py -3.14 -m pip install pyan3==2.6.0`).
- Joern installed at `C:/Users/Seth/joern_install/joern-cli` (adjust both `--joern*`
  paths if it moves).
- `pipeline/artifacts/pilot_applications.csv` present. To rebuild it:
  `py -3.14 -m Applications.pilot_select --per-framework 5` (top-5 per framework).

## Outputs to look at

All under `pipeline/artifacts/`:

- **Invoker analysis (Stage 5):**
  - `llm_invokers_all.csv` — direct + transitive LLM invokers (`kind` column).
  - `llm_calls_all.csv`, `call_metadata_all.csv` — call sites + per-arg metadata (determinism knobs).
  - `llm_tests_all.csv` — tests that reach a real LLM call (the non-determinism headline).
  - `eval_invokers_all.csv`, `eval_calls_all.csv` — eval-framework usage.
  - `call_graph_health.csv` — per repo: `cg_source` (`pyan` / `pyan_resilient` / `none`),
    `graph_usable`, edge counts, excluded files. Tells you whether a low transitive
    count is real or a pyan failure.
- **Slices (Stage 6):** `pipeline/artifacts/slices/<repo>/` — `programs.jsonl` (the
  per-variable SubPDGs) + a summary per repo.
- **Progress:** `pipeline/artifacts/.batch_progress.json` — `processed` / `failed` /
  `slice_failed` lists (used by `--resume`).

## After the run

Cross-tabs / the standard report:

```
py -3.14 Applications/analyze.py
```

writes per-question CSVs to `pipeline/artifacts/analysis/` and prints the population,
non-deterministic-test, determinism-knob, and call-graph-health sections.

## Notes

- Slicing is the slow part (a CPG build per repo). Oversized generated files
  (>1.5 MB) are auto-hidden before parse to avoid Joern OOM; heap escalates 8→12→16 GB
  only on OOM.
- A slice failure on one repo is isolated — it's recorded in `.batch_progress.json`
  under `slice_failed` and does **not** lose that repo's already-written metadata.
- To wipe kept clones afterward: `py -3.14 -m pipeline.clean_clones` (`--dry-run` to preview).
