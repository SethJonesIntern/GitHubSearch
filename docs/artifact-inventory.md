# Research artifact inventory

Recorded 2026-11-04 from source inspection and the current checkout. No producer was
executed to create this inventory. Paths are relative to the repository root. This
is an ownership/lineage map, not proof that an artifact is reproducible from this
checkout or backed up elsewhere.

## Source artifacts and research contributions

| Artifact | Producer / role | Inputs | Storage / retention |
|---|---|---|---|
| `Wrapper/FrameworkDict.py` | Curated framework invocation inventory and scoped detector patterns | Framework/API investigation and documented pattern reviews | Git. Contribution in its own right; preserve registry provenance and exclusion history. |
| `pipeline/third_party/pymethods2test/` | Unmodified published focal mapper | Vendored upstream source, NOTICE and LICENSE | Git. Preserve attribution; not our original mapper. |
| `pipeline/cuts.py`, `EXCLUSIONS.md` | Scope enforcement and rationale | Audit decisions | Git. Required to interpret results. |
| `pipeline/artifacts/application_audit.csv` | `pipeline.audit_apps` plus audit passes and manual adjudication | Population, progress, call/invoker data, clone/slice presence | Git. Has hand-maintained fields; refreshing it can change derived fields when local data is absent. |

## Discovery and invocation outputs

| Saved artifact(s) | Producer | Main inputs | Storage / retention |
|---|---|---|---|
| `pipeline/artifacts/frameworks.csv`, `frameworks_filter_stats.json` | `Frameworks/GithubSearch.py` | GitHub framework search/enrichment | Git; preserve saved search population. |
| `pipeline/artifacts/applications.csv`, `application_metadata.csv`, `applications_filter_stats.json` | `Applications/search_candidates.py` | Framework import patterns, GitHub results, repo metrics | Git; fresh discovery can change the population and overwrite outputs. |
| `pipeline/artifacts/applications_slim.csv`, `application_metadata_slim.csv`, `slim_report.json` | `Applications/slim_applications.py` | Candidate and metadata tables, scope decisions | Git; saved population shaping. |
| `pipeline/artifacts/.search_progress.json` | Application discovery | Search progress/state | Gitignored; may exist on data PC. Do not assume Git is its backup. |
| `pipeline/artifacts/.batch_progress.json` | `pipeline.batch_call_metadata` | Processed/failed repo state | Currently Git-tracked despite an ignore rule for new files of that name. Preserve actual data-PC progress; do not overwrite it with an older checkout's state. |
| `pipeline/artifacts/llm_calls_all.csv`, `call_metadata_all.csv` | `pipeline.batch_call_metadata` and Wrapper AST extraction | Saved application clones and scoped registry | Git; calls versus per-argument metadata have different units. |
| `pipeline/artifacts/llm_invokers_all.csv`, `llm_tests_all.csv` | `pipeline.batch_call_metadata` | Direct matches and call-graph closure | Git LFS. Tests are a subset of invokers. Raw/transitive rows remain; apply study filters. |
| `pipeline/artifacts/call_graph_health.csv`, `empty_graph_repos.csv` | Batch analysis and graph audits | Graph construction outcomes | Git; distinguish failed/unavailable graphs from zero reachability. |
| `pipeline/artifacts/eval_calls_all.csv`, `eval_invokers_all.csv`, `eval_call_metadata_all.csv` | Batch evaluation pass | Same clone/index, evaluation patterns | Git; library-call evidence does not establish a correctness guardrail. |
| `pipeline/artifacts/framework_frequency.csv`, `eval_frequency.csv` when present | Framework distribution / `pipeline.eval_frequency` | Saved search or call outputs | Derived outputs; do not regenerate as cleanup. |
| `pipeline/artifacts/audit_*.csv`, `framework_triage.csv`, `import_name_*.csv`, `imported_by.csv`, `name_classification.csv` | `pipeline/audit_*.py` and import-name review passes | Raw artifacts, scope ledger, sometimes live clones | Git; passes have different scan/write modes. Inspect each before use. |
| `pipeline/artifacts/analysis/` | `Applications/analyze.py` and reporting passes | Saved artifacts plus scope filters | Git where tracked; derived reports can overwrite saved results. |

## Focal mapping and production-method baseline

| Saved artifact(s) | Producer | Main inputs | Storage / retention |
|---|---|---|---|
| `pipeline/artifacts/pym2t_nd_tests.csv`, `pym2t_focal_calls.csv` | `pipeline.pym2t_focal` | Observed scope, full focal outputs, historical direct-test subset | Git. Saved calls table is the observed-scope snapshot used by the new figures. |
| `pipeline/artifacts/pym2t_focal_tests.csv`, `pym2t_runs.csv` | `pipeline.pym2t_focal` | Saved clones and upstream mapper/replay | Gitignored, absent in this checkout; expected on data PC. Needed for the baseline. |
| `pipeline/artifacts/pym2t/` | Per-repo mapper runs | Saved clones | Gitignored, absent here; retain per-repo JSON/run diagnostics on data PC. |
| `pipeline/artifacts/focal_map_llm_tests.csv` | Superseded custom matcher in `pipeline.focal_map` | Historical direct LLM-calling test subset | Git. Historical result, not current upstream-mapper evidence. Preserve. |
| `pipeline/artifacts/method_test_baseline/summary.json`, `by_repo.csv` | `pipeline.method_test_baseline` | Full focal CSVs, clone AST inventory, canonical POP/CALLS | Git. Initial correspondence baseline, not guardrail prevalence. |
| `pipeline/artifacts/method_test_baseline/methods.csv` | Same baseline | Same inputs | Gitignored, absent here; detailed method rows expected on data PC. Re-running baseline overwrites its own outputs. |

## Clones, Joern and figures

| Saved artifact(s) | Producer | Main inputs | Storage / retention |
|---|---|---|---|
| `pipeline/repos/`, `Wrapper/repos/` | Discovery/analysis clone operations | Remote repositories at saved revisions | Gitignored; application clones absent here. Existing batch code can replace/delete them. Do not execute it for cleanup. |
| `pipeline/artifacts/slices/` | `pipeline.slice_repo` / Joern slicer | Saved clones, CPGs, invoker filters | Gitignored, absent here; raw slice corpus is on data PC. |
| `pipeline/artifacts/slice_index.csv`, `slice_pdg_lines.csv.gz`, `slice_code.jsonl.gz` | `pipeline.slice_index` | Raw `programs.jsonl` slice corpus | Gitignored, absent here; require data-PC slices to rebuild. Do not regenerate during organization. |
| `pipeline/artifacts/slice_files.csv`, `slice_index_summary.json` | Same indexer | Raw slice corpus | Git; saved indexing evidence. |
| `pipeline/artifacts/slice_call_hits.csv`, `slice_calls_by_call.csv`, `slice_calls_summary.json` | `pipeline.slice_calls` | Full slice index, PDG line table, observed calls | Git; joins do not establish correctness checks. |
| `pipeline/artifacts/figures/`, framework/keep-frequency PNGs | Historical `Applications/make_figures.py` and `plot_*.py` | Historical scoped analysis tables | Preserve unchanged. Interpretation is under review; not the new main figure set. |
| `pipeline/artifacts/guardrail_figures/<version>/` | **New** `Applications/guardrail_figures.py` | Five saved inputs listed by SHA-256 in its manifest | New PNG/SVG files, CSV twins, captions, inventory export and provenance. Requires a new output directory; no overwrite mode. |

## Backups and exploratory files

Preserve `pipeline/artifacts/_checkpoint_*`, `_pre_scope_filter`, `_recovery_partial`,
`_framework_repos_removed`, `_rerun_backup_*`, `*.bak`, `pipeline/next_runs/` queues,
root-level candidate CSV/TSV/JSON files, and old run logs. Their names do not establish
that they are disposable. Producers and restoration paths need case-by-case review.

Before any future migration, produce a read-only file-level manifest on the data PC
(relative path, size, SHA-256, source revision where available), then verify an
independent backup of ignored data. Neither task has been performed by this review.
