# Working on GitHubSearch

`main` is the shared integration branch. Start each change from current `main`, use a
short-lived `codex/<topic>` branch (or another descriptive feature branch), and open a
pull request into `main`. Keep unrelated research changes in separate PRs.

```powershell
git fetch origin
git switch main
git pull --ff-only
git switch -c codex/your-topic
```

Keep local work committed or stashed before switching. If a command reports conflicts
or would overwrite local files, inspect the state rather than forcing it. Push the
feature branch, review its diff and checks, then merge the PR. Use regular merge
commits to retain research history. Do not rewrite shared branch history or force-push
`main`. CI runs automatically on PRs into `main` and pushes to `main`.

## Checks

Use Python 3.14 for the supported research environment. In an isolated environment:

```powershell
python -m pip install -r requirements/ci.txt
python -m pytest pipeline/test_method_test_baseline.py pipeline/test_slice_repo_filter.py pipeline/test_eval_frequency.py Applications/test_guardrail_figures.py -q
```

These are the current Linux/Windows CI checks. They use synthetic data and need no
GitHub token, cloned corpus, Joern, LFS downloads, or full focal outputs. They do not
validate the entire discovery/call-graph pipeline. See RUNBOOK.md and CLAUDE.md for
research commands, additional tests, runtime dependencies and the data-PC workflow.
The root requirements.txt records an older environment; do not use it as the CI lockfile.
See requirements/README.md for the figure dependency pin and draft analysis declaration.
Figure tests validate refusal to overwrite and input consistency using synthetic files;
CI does not render or regenerate the saved research corpus.

## Research artifacts and local data

Read README.md's contribution/status statement and AGENTS.md before changing analyses.
Keep methods, tests, calls and repositories distinct as counting units. Document
population changes in EXCLUSIONS.md and preserve raw rows. Cite the published tools
we use, including pyMethods2Test.

Install Git LFS and run `git lfs pull` when you need the large tracked CSVs. Saved
clones, Joern slices, full focal outputs and generated method inventories may be
gitignored and available only on the data PC. Never use `git clean`, destructive
resets or forced checkouts to synchronize these machines. Normal CI intentionally
skips downloading LFS payloads.
Consult docs/artifact-inventory.md and docs/commands.md before any analysis run.
Legacy candidates in docs/archive-candidates.md require another thorough dependency
and reproduction review, plus owner approval, before any archive or deletion.

## Branch consolidation (2026-11-04)

The authoritative research snapshot was `codex/method-test-mapping-baseline` at
`6d86324`. The consolidation integration commit retains the prior `main` as an
ancestor using Git's `ours` merge strategy, keeping that research tree unchanged.
This records both histories without combining older file contents into the current
implementation. The old `main` is also preserved on
`archive/main-before-consolidation-2026-11-04`. Source histories remain available
even when fully merged feature-branch names are retired.

`PipelineB` and `slice-index` are already ancestors of the authoritative snapshot;
their work requires no additional content merge. The separate evaluation-framework
discovery branch/PR remains available for review and is not implicitly incorporated
by this consolidation. The owner approved removal of fully merged `PipelineB`,
`slice-index`, the method baseline branch and the workflow/cache-cleanup branches
after ancestry verification. The backup and unmerged evaluation work remain. Review
ancestry and open PRs before retiring further branches.

