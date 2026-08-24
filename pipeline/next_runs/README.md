# next_runs — targeted re-run queues

Curated `--input` files for `batch_call_metadata`, one per problem. Each has the same
columns as `applications_slim.csv` plus a **`reason`** column saying why the repo is
queued, so they work as run inputs *and* can be read by the audit/analysis scripts.

Regenerate any time (they are derived, never hand-edited):

```
py -3.14 -m pipeline.make_next_runs
```

| file | repos | what's wrong | needs prep first? |
|---|---:|---|---|
| `reclone.csv` | 20 | clone failed or the checkout is missing — **no data exists** for these | **No** — run it directly |
| `reslice.csv` | 12 | Joern slicing failed (7 Java OOM, 5 joern-parse). Call/invoker data is fine; only the slices are missing | **Yes** |
| `regraph.csv` | 98 | pyan produced no usable call graph — direct invokers only, **no transitive closure**. These are why transitive numbers are an undercount | **Yes**, and see the warning below |

---

## Read this first — why prep exists

`batch_call_metadata` has two modes and, for a repo that has already been processed,
**both are wrong**:

- **with `--resume`** — it skips any repo already in `.batch_progress.json['processed']`.
  All 12 reslice and all 98 regraph repos are in there, so the run does nothing and
  exits looking like a success.
- **without `--resume`** — it **deletes every artifact CSV and the progress file** and
  restarts the whole 1,035-repo run from zero.

And the writers append without deduping, so a repo re-run while its old rows are still
in the artifacts is counted **twice** in every number.

`prepare_rerun.py` is the missing middle: it removes the queued repos from the progress
file and lifts their existing rows out of every artifact, so the re-run writes them back
exactly once.

Nothing is discarded — lifted rows go to `pipeline/artifacts/_rerun_backup_<timestamp>/`
containing **only the removed rows** (not copies of the 189 MB originals), so a queue can
be restored by appending them back.

---

## Running one

**1. See what it would do (safe, changes nothing):**

```
py -3.14 -m pipeline.prepare_rerun pipeline/next_runs/reslice.csv
```

It prints how many repos are stuck in `processed` and how many rows would be lifted from
each artifact. If it reports `0 in processed` and `0 rows` (as `reclone.csv` does), skip
straight to step 3.

**2. Apply it:**

```
py -3.14 -m pipeline.prepare_rerun pipeline/next_runs/reslice.csv --apply
```

**3. Run it.** One line, bash:

```
py -3.14 -m pipeline.batch_call_metadata --input pipeline/next_runs/reslice.csv --slice --joern-parse C:/Users/Seth/joern_install/joern-cli --joern C:/Users/Seth/joern_install/joern-cli/joern.bat --keep-clones --resume
```

PowerShell (backticks continue the line):

```powershell
py -3.14 -m pipeline.batch_call_metadata `
  --input pipeline/next_runs/reslice.csv `
  --slice `
  --joern-parse C:/Users/Seth/joern_install/joern-cli `
  --joern C:/Users/Seth/joern_install/joern-cli/joern.bat `
  --keep-clones --resume
```

**`--resume` is mandatory.** Leaving it off wipes the entire run. Drop `--slice` and the
two `--joern*` flags for `reclone.csv` / `regraph.csv` if you don't need slices — it is
roughly 2× faster without them.

**4. Refresh the audit sheet and the report afterwards:**

```
py -3.14 -m pipeline.audit_apps
py -3.14 Applications/analyze.py
```

---

## Per-queue notes

**`reclone.csv` (20)** — needs no prep; these repos are not in `processed` and have no
rows anywhere. 12 of them already have a working checkout on disk (the clone failure was
transient), so most should succeed immediately. 3 have no checkout at all:
`agno-agi/pal`, `holetron/hindsight-mempalace`, `openJiuwen-ai/agent-store`. If a repo
has since been deleted or renamed on GitHub, the clone will fail again — that is a real
answer, not a bug.

**`reslice.csv` (12)** — lifting these removes 38,287 rows across 8 artifacts, all of
which get rewritten by the run. The failures were 7 Java `OutOfMemoryError` and 5
`joern-parse failed`. The OOM ones are large repos (`apache/airflow`,
`Azure/azure-sdk-for-python`, `PostHog/posthog`); raising the Java heap
(`_JAVA_OPTIONS=-Xmx16g` is what the last run used) is the thing to change before
re-running, otherwise they will fail the same way.

**`regraph.csv` (98)** — ⚠️ **re-running these as-is reproduces the same dead graphs.**
Nothing about pyan changes between runs, so this queue is only useful with a fix behind
it:

- **time-boxed failures** — set `PYAN_TIME_BUDGET_SEC=0` (no timer). This is already
  validated to recover langflow, litellm, phoenix, marimo, hermes.
- **parse failures** — need the `ast.parse` pre-filter (drop unparseable files before
  pyan instead of letting pyan fail all-or-nothing). **Not built yet.**
- `mlflow`, `angr`, `CVlization` need the angr localizer fix (bare-qname `KeyError` in
  `transitive_invokers._find_offending_file`). **Not built yet.**

See `CALL_GRAPH_EXCLUSIONS.md`.

---

## Gotchas

- **`py -3.14` always** — pyan3 needs it.
- On Windows pass the Joern `.bat`: a bare `joern` is a Unix script and dies with
  `WinError 193`.
- The clone tree is ~107 GB. `--keep-clones` keeps it that way; without it each clone is
  deleted after analysis.
- Long runs: launch it and let it finish. `--resume` is safe to re-issue after any
  interruption.
