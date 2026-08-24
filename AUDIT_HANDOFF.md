# Audit Sheet — session handoff (2026-08-20 → 08-22)

What this session built, what it decided, and what is still open. Companion docs:
`RUNBOOK.md` (how to run Stage 5/6), `EXCLUSIONS.md` (the canonical exclusion ledger),
`COVERAGE_ANALYSIS.md` (the three scope buckets).

---

## 1. What was built

A per-application **audit sheet** — `pipeline/artifacts/application_audit.csv`, one row
per population repo (1,055), one column per data-quality question — plus four scripts
that fill it. Each script owns its columns and is safe to re-run; hand edits and other
passes' columns are always carried forward.

| script | fills |
|---|---|
| `pipeline/audit_apps.py` | builds the sheet: calls / invokers / ND tests / eval calls, pyan health, Joern slice status, and the `CUT` / `UNCOVERED` dispositions |
| `pipeline/audit_imports.py` | `frameworks_imported` — what each clone ACTUALLY imports — then sets `in_scope` from it (`--scan` first, ~15 min) |
| `pipeline/audit_framework_check.py` | `framework_suspect` / `framework_evidence` — is this a library, not an app? (`--scan` first, ~13 min) |
| `pipeline/audit_zero_invokers.py` | `zero_invoker_reason` + the `py/ipynb/test/http/cli` evidence columns |
| `pipeline/cuts.py` | the single source of truth every consumer reads |

**Cuts now reach the statistics.** `analyze.py`, `keep_frequency.py` and
`plot_coverage.py` all read `cuts.py`, so marking a row moves both numerator and
denominator. Set `in_scope=0` (or `uncovered`) in the CSV by hand and re-run — the
edit survives a rebuild.

`in_scope` is three-valued, because the study has three dispositions:

| value | analyzed stats | coverage denominator |
|---|---|---|
| *(blank)* = in scope | counted | counted |
| `uncovered` | **out** | **in** — it *is* the unmeasured tail the 90.1% quantifies |
| `0` | out | out — never an LLM app |

## 2. Current numbers

**Population 1,055:** 823 in scope · 86 uncovered · 37 cut · **109 undecided**.
**Analysis base 808** (in scope + processed; 15 in-scope repos never cloned).

| measure | total | repos with ≥1 |
|---|---:|---:|
| LLM call sites | 35,853 | 740 (92%) |
| direct invokers | 28,465 | 741 (92%) |
| transitive invokers | 560,365 | 619 (77%) |
| ND tests | 272,008 | 546 (68%) |
| — of which **direct** | **14,848** | 299 (37%) |
| eval calls | 171 | 42 (5%) |

Quality flags on those 808: **91.6%** usable pyan graph (68 repos have no transitive
data at all) · Joern 90.8% ok / 7.7% absent / 1.5% failed · 8.0% zero-invoker ·
**5.4% flagged framework-not-application, and those 44 repos hold 13% of all ND tests.**

## 3. What was found

- **`haystack` is three unrelated projects.** deepset Haystack (ours), django-haystack
  (Django search) and Project Haystack (building automation). Of 49 repos matched on the
  token, only 24 call deepset. **20 cut** (EXCLUSIONS §10).
- **`clai` never was pydantic-ai's CLI** — it matched the substring in `claim`/`claiming`.
  49 repos matched it; `keep_frequency.EXTRA_MEMBERS` still maps it to pydantic_ai, which
  inflates that framework from **79 to 127 apps** and moves it from #4 to #2 in the
  ranking. **The mapping is still in the code — not yet removed.** 17 of its repos cut.
- **Stage 2 is code search, not import matching.** `matched_frameworks` records the token
  that surfaced a repo, never a parsed import: `camel`→camelCase, `agno`→agnostic,
  `notte`→nottest, `omnigent`→OmniGe**nt**ransformer. 139 repos never import what they
  matched. This is why scope is now decided from `frameworks_imported`.
- **The framework/app check works off our own corpus** — who imports whom. It found
  marimo, wandb/weave, NeMo/Guardrails, cognee, ddtrace, sentry_sdk, logfire, plus the
  first-party langchain integration packages (`langchain-google`, `-aws`, `-ibm`, `-cohere`).
- **The uncovered tail is mostly raw SDK users** — 51 openai, 21 anthropic, 15 litellm —
  not exotic frameworks. Worth stating in the paper that the population is specifically
  *framework* applications.
- **Per-framework tables were fragmented.** `analyze.py` now applies
  `keep_frequency.category()`: 42 import names → 20 frameworks, and langchain goes from an
  apparent 436 calls to **11,221 (30.9%)**. Raw SDKs get a `kind` column. The ungrouped
  view is preserved in `calls_by_import_name.csv` / `nd_tests_by_import_name.csv`.

## 4. Open — in priority order

1. **Decide the 109 undecided** (no LLM import found anywhere). They contribute **0 calls
   and 0 ND tests** — pure denominator — so resolving them moves ND-test prevalence by
   ~12 points (68% vs 60%). Cross-check against `http_llm_files` / `cli_llm_files` first:
   an app that POSTs to `api.openai.com` imports nothing.
2. **Remove the `clai` → pydantic_ai line** in `keep_frequency.EXTRA_MEMBERS`. Both
   COVERAGE_ANALYSIS.md and EXCLUSIONS §9 already say it is wrong.
3. **The 92 `imports_fw_no_call_site` repos** — imports the framework, no call site
   matched. 16 were the haystack collision; check `dspy` and `honcho` (5 each) for the
   same before treating the rest as pattern work.
4. **The 44 framework_suspect repos** carry 13% of ND tests — decide in or out.
5. **Report ND tests direct vs transitive.** 272,008 is 95% transitive, inflates with
   graph size, and is missing entirely for the 68 unusable-graph repos. 14,848 direct is
   the defensible number.
6. **20 clone failures never resumed** — 12 have working clones on disk already.
7. `applications_analyzed.csv` (827) is staged-deleted in git but **still on disk**;
   anything reading it silently uses the abandoned 827-row scope.

## 5. Gotchas

- `py -3.14` always. The Grep tool skips `pipeline/repos` (gitignored) — use Bash `grep`.
- The clone tree is **107 GB** / 1,056 repos; a full import scan is ~15 min at 8 workers.
  Use one alternation regex over the names you want, not a match-everything-then-filter
  loop — that difference was 90 min vs 15.
- Windows `ProcessPoolExecutor` needs an `if __name__ == "__main__"` guard.
- Cuts are **filters, never deletions** — every cut repo keeps its row and its data.
