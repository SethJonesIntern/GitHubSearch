# GitHubSearch — project instructions

Empirical software-mining study of **non-determinism in the testing of LLM / agent-
framework applications** mined from GitHub. Pipeline: discover frameworks → code-search
for apps importing them → clone + static analysis (LLM call sites, invoker closure,
determinism kwargs, tests) → Joern per-variable slicing.

**The full run is DONE** (2026-08-17, 1,035 of 1,055 repos). Current work is data
quality: deciding which of those repos belong in the study and which numbers survive.

## Read these, in this order

| doc | what it settles |
|---|---|
| `AUDIT_HANDOFF.md` | **start here** — current state, totals, and the ordered open list |
| `EXCLUSIONS.md` | the canonical ledger of every exclusion/exemption/pattern-cut. **Add a dated row whenever you exclude anything.** |
| `COVERAGE_ANALYSIS.md` | the three scope buckets and the 90.1% coverage claim |
| `RUNBOOK.md` | how to run Stage 5/6 |
| `pipeline/next_runs/README.md` | targeted re-runs, and why they need prep |
| `CALL_GRAPH_EXCLUSIONS.md` | why some repos have no pyan graph |
| `PIPELINE.md` | stage-by-stage design. **Its "CURRENT STATE" is stale (2026-06-25)** — ignore that section |

## The one thing to understand first

**Stage 2 is GitHub *code search*, not import matching.** `matched_frameworks` records
the token that surfaced a repo — GitHub tokenizes, so it hits substrings and
co-occurrences, not parsed imports. `camel` matched camelCase, `agno` matched
`agnostic`, `notte` matched `nottest`, `omnigent` matched `OmniGe`**`nt`**`ransformer`.
139 repos never import what they matched.

Consequence: **never decide scope from `matched_frameworks`.** Scope now comes from
`frameworks_imported` in the audit sheet — what the clone actually imports. Anything
still keyed off the search token is suspect.

## The audit sheet

`pipeline/artifacts/application_audit.csv` — one row per population repo (1,055), one
column per quality question. Rebuild/refill with:

```
py -3.14 -m pipeline.audit_apps              # the sheet + counts + CUT list
py -3.14 -m pipeline.audit_imports --scan    # what each clone imports -> in_scope (~15 min)
py -3.14 -m pipeline.audit_framework_check --scan   # framework-vs-app (~13 min)
py -3.14 -m pipeline.audit_zero_invokers     # why a repo has no invokers
```

Each pass owns its columns; re-running never clobbers another pass's work or a hand
edit. `in_scope` / `notes` are yours to edit by hand and are always preserved.

**`in_scope` is three-valued, and the difference matters:**

| value | analyzed stats | coverage denominator |
|---|---|---|
| *(blank)* — in scope | counted | counted |
| `uncovered` — real LLM app on a framework outside the top-20 | **out** | **in** (it *is* the tail the 90.1% measures) |
| `0` — not an LLM app | out | out |

`pipeline/cuts.py` is the single source of truth. `analyze.py`, `keep_frequency.py` and
`plot_coverage.py` all read it, so **editing `in_scope` in the CSV changes every
statistic** on the next run.

## Commands

```
py -3.14 Applications/analyze.py             # the report + analysis/*.csv cross-tabs
py -3.14 Applications/keep_frequency.py      # framework ranking
py -3.14 Applications/plot_coverage.py       # framework_coverage.png
py -3.14 -m pipeline.make_next_runs          # regenerate the re-run queues
```

Re-running Stage 5 on selected repos: see `pipeline/next_runs/README.md`. Never run
`batch_call_metadata` without `--resume` — it deletes every artifact and restarts the
1,035-repo run.

## Conventions

- **`py -3.14` always.** pyan3 (the transitive call graph) requires it.
- **Cuts are filters, never deletions.** A cut repo keeps its row and its raw data; only
  its `in_scope` value changes. Every criterion lives in code with a comment, and gets a
  dated row in `EXCLUSIONS.md`.
- **Evidence before cutting.** Read the actual matched code; inference has been wrong
  repeatedly (omnigent, connectonion, haystack, clai all fooled inference).
- Distinguish **measured / projected / assumed** when reporting numbers.
- Prefer giving Seth a command to run over backgrounding a long job.

## Gotchas

- **The Grep tool silently skips `pipeline/repos`** (gitignored, and ripgrep honours
  `.gitignore`). Use Bash `grep` or `rg --no-ignore` for the clones.
- The clone tree is **~107 GB / 1,056 repos**. A full-corpus scan is ~15 min at 8
  workers — use one alternation regex over the names you want, not match-everything-
  then-filter (that difference was 90 min vs 15).
- Windows: `ProcessPoolExecutor` needs an `if __name__ == "__main__"` guard; pass Joern
  as the `.bat` (a bare `joern` is a Unix script → `WinError 193`).
- `core.autocrlf=true`, so LF-written files produce a harmless CRLF warning on commit.

## Known-bad things not yet fixed

- **`keep_frequency.EXTRA_MEMBERS` still maps `clai` → pydantic_ai.** `clai` is a junk
  collision token; the mapping inflates pydantic_ai from 79 to 127 apps and moves it
  from #4 to #2 in the ranking. Both `COVERAGE_ANALYSIS.md` and `EXCLUSIONS.md` §9 say
  it is wrong.
- **Raw Gemini is invisible to the whole pipeline** — `google.generativeai` /
  `google.genai` is in no dict, while 254 repos (24% of the corpus) import it. `openai`
  and `anthropic` are both in scope, so SDK-level numbers silently omit one of the three
  major providers.
- **109 repos are undecided** (no LLM import found anywhere). They contribute 0 calls and
  0 ND tests but sit in the denominator — resolving them moves ND-test prevalence ~12
  points.
- **ND tests are 95% transitive** (272,008 total, 14,848 direct) and transitive inflates
  with graph size, and is missing entirely for the 98 repos with no usable graph. Report
  direct and transitive separately; 14,848 is the defensible figure.
