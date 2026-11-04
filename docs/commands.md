# Command side effects

This guide documents existing behavior. It does not change producer defaults or
authorize execution. Organization and figure work must not regenerate saved research
outputs, replace clones, or refresh audit tables. Do not run a producer merely to
check whether it still works. Importing an old script may also execute top-level IO.

| Command / entry point | Read/write effects | Use during organization |
|---|---|---|
| Git status/log/diff/branch listing; reading source/CSV/JSON as files | Read-only inspection | Allowed. Reading a file is different from importing/running its module. |
| Explicit CI tests listed in CONTRIBUTING.md | Synthetic temp data and generated Python caches | Isolated validation only. Do not run broad discovery over repository test-looking files or extracted corpus tests. |
| `Applications/guardrail_figures.py --output-dir NEW_DIRECTORY` | Reads saved summary/table/calls and source text; writes only new figure outputs | Authorized figure path. Existing directory rejected; inputs hashed before/after. No existing producer is imported. |
| `pipeline.method_test_baseline` | Reads clones and full focal results; writes/overwrites its own summary, by-repo table and method inventory | Not part of cleanup. Re-run only as an intentional new analysis. |
| `python -m pipeline run` | Launches discovery/analysis producers; analysis currently receives neither `--resume` nor `--keep-clones` by default | **Do not run as a setup/smoke/cleanup check.** The orchestrator is not read-only. |
| `pipeline.batch_call_metadata` | Without `--resume`, unlinks the eight configured output files and batch progress. For each selected repo, removes any existing checkout before cloning. Without `--keep-clones`, deletes checkout after analysis. | **Do not execute during organization.** `--resume` does not prevent replacement of a selected existing clone. |
| `pipeline.prepare_rerun` | Default previews row removal; `--apply` rewrites artifacts/progress and writes lifted-row backups | No apply during organization. Queue correctness and restoration require separate review. |
| `pipeline.clean_clones` | Deletes clone directories when applied; has a preview mode | Do not execute cleanup. |
| `pipeline.audit_apps`, audit `--scan` modes | Rewrites audit evidence; some values depend on available clones/slice directories | Do not refresh here. Missing data can change derived fields even when manual fields survive. |
| `pipeline.pym2t_focal` | May run upstream scripts when cached run files are missing; writes focal tables. `--rerun` replaces per-repo cached outputs. | Do not run for organization or plotting. |
| `pipeline.slice_index`, `pipeline.slice_calls`, slicer/CPG entry points | Writes indexes, join tables, summaries or raw slices | Do not run for organization or plotting. |
| Discovery scripts; `Applications/analyze.py`; historical figure scripts | Network/cloning and/or writes to saved CSVs, reports, images and captions | Do not run to tidy the repository. Use saved files as figure inputs. |

The historical `RUNBOOK.md` commands are intentional research operations, not a
recommendation to execute them now. A non-destructive future CLI could require an
explicit output/run directory and fail when outputs exist; implementing that would
be a separate reviewed change with isolated tests, not an implicit cleanup action.

Changes touching existing research paths require explaining the exact paths and
action to the user first. A passing synthetic test suite is not permission to rerun
the real corpus, delete artifacts, or replace data-PC clones.
