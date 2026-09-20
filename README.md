# GitHubSearch

Mining study of **non-determinism in agentic applications**: which agent frameworks real
applications are built on, where those applications invoke a model, whether they pin the
parameters that make a call reproducible, and whether their tests exercise that code.

The population is mined from GitHub, cloned, and analyzed statically: AST pattern
matching for framework call sites, a pyan3 call graph for the blast radius of LLM use,
Joern CPGs for per-variable program slices, and pyMethods2Test for test-to-code (focal)
mapping.

---

## Status (2026-09-18)

The full-population run is complete. Current work is data quality and test traceability.

| | |
|---|---:|
| population (`applications_slim.csv`) | 1,055 |
| processed by Stage 5 | 1,033 |
| clone failures (Windows illegal-filename bug, fixable) | 22 |
| analyzed, after documented scope cuts (`EXCLUSIONS.md`) | 753 |
| **observed applications** (≥1 framework LLM call site) | **644** |
| framework LLM call sites | 23,990 |
| direct non-deterministic tests | 9,700, in 233 apps (36%) |

**Headline results** (observed scope):
- **98.9% of framework LLM call sites set none of the eight determinism-relevant
  parameters.** Temperature is set on 0.1% of calls, seed on none.
- **Only 25 of 23,990 call sites (0.1%) lie inside the focal method of a direct
  non-deterministic test**, as mapped by pyMethods2Test. Tests that reach an LLM rarely
  target the function that makes the call.

---

## Scope rules

These apply to every reported number. Excluded rows stay in the raw CSVs: they are
filtered at analysis time, never deleted.

1. **Agentic applications only.** An application is in scope if it imports an agentic
   framework. A repo that reaches a model only through a raw SDK or raw HTTP is out.
2. **Raw provider SDKs are excluded.** `openai` and `anthropic` call sites are detected
   but not observed, measured or reported: no calls, parameters, invokers or tests.
3. **Direct only (2026-09-18).** `direct` / `transitive` is our Stage-5 term: a direct
   row contains an LLM call, and a transitive row only reaches one through the pyan call
   graph. All test analysis (ND tests, focal mapping, prevalence) uses direct rows only.
   **The only permitted use of transitive data is the blast radius of LLM use**: how
   much of an app's code can reach an LLM call. Transitive means a path *exists*, not
   that it runs, and its counts scale with graph size. Ledger: `EXCLUSIONS.md` §18.

---

## Where the documentation lives

Read these before trusting any number in the artifacts.

| Document | What it is |
|---|---|
| **`EXCLUSIONS.md`** | **The ledger, single source of truth.** Every repo dropped, framework exempted, pattern cut and scope decision, each with its reason and the code that enforces it. Start here. |
| `METHODOLOGY.md` | What was done, stage by stage, with the population waterfall and headline figures. |
| `METHODOLOGY_REVIEW.md` | Parked methodological issues, held for review. |
| `PIPELINE.md` | Stage-by-stage design (its "CURRENT STATE" section is stale). |
| `RUNBOOK.md` | How to run Stage 5 + Stage 6: the command, every flag, the Windows/Joern gotchas, and how to stop and resume without duplicating rows. |
| `COVERAGE_ANALYSIS.md` | How much of the population the analyzed frameworks cover, and the units that number is in. |
| `CALL_GRAPH_EXCLUSIONS.md` | Why some repos have an empty call graph, and the holes in the pyan time-box. |
| `CLAUDE.md` | Working notes: current state, open work, machine setup, gotchas. |

---

## Layout

| Path | Role |
|---|---|
| `pipeline/` | The current system. Stage drivers (`batch_call_metadata.py`, `slice_repo.py`, `pym2t_focal.py`), the audit-sheet passes (`audit_*.py`), `cuts.py` (scope), `waterfall.py`, paths. |
| `pipeline/third_party/pymethods2test/` | pyMethods2Test, vendored unmodified (Apache 2.0; see its `NOTICE.md`). |
| `Wrapper/` | The analysis engine. `FrameworkDict.py` (per-framework invocation patterns), `transitive_invokers.py` (AST index, seed matching, pyan call graph, closure), `false_positives.py` (the five FP tiers), `call_metadata.py` (per-argument extraction). |
| `Applications/` | Discovery (`search_candidates.py`), population shaping (`slim_applications.py`, `keep_frequency.py`), and reporting (`analyze.py`, `make_figures.py`, `plot_*.py`). |
| `Frameworks/` | Stage-1 framework discovery. |
| `pipeline/artifacts/` | All outputs. Gitignored clones live in `pipeline/repos/`. |

## The stages

1. **Framework search**: find agent frameworks, derive their import names.
2. **Application search**: GitHub code search for those names. It matches tokens, not
   parsed imports, so scope is decided afterwards from what each clone actually imports.
3. **Frequency table**: apps per framework ecosystem.
4. **Application list**: slim the candidates to trustworthy names; flag scope.
5. **Invoker + call extraction**: clone each app, match invocation patterns, build the
   call graph, extract per-argument metadata.
6. **Joern slicing**: per-variable SubPDGs for each LLM-invoker function.
7. **Semantic evaluation**: eval-framework usage, piggybacked on the invoker search.
8. **Focal mapping**: pyMethods2Test maps each test to the file and method it targets;
   the result is joined to the direct ND tests and the framework call sites.

---

## Requirements

- **Python 3.14**, required: pyan3 2.6.0 (the call graph) does not work on older versions.
- `py -3.14 -m pip install -r requirements.txt`, plus `pyan3==2.6.0`.
- Focal mapping: `fuzzywuzzy~=0.18.0` and `Levenshtein`. pyMethods2Test pins
  `python-Levenshtein~=0.26.1`, which has no Python 3.14 wheel.
- **Joern**: Stage 6 only. On Windows pass the `.bat`; the extension-less script fails
  with `WinError 193`.
- **Git LFS**: `llm_invokers_all.csv` and `llm_tests_all.csv` are LFS-tracked.
- **`GITHUB_TOKEN`** in `.env`: discovery stages only. Analysis runs on local clones and
  needs no token.

## Common commands

```bash
# the population waterfall: every drop, in order, with the pending review queues
py -3.14 -m pipeline.waterfall

# the standard report (population, tests, determinism parameters, graph health)
py -3.14 Applications/analyze.py

# the figure set (F1-F3 + FIGURES.md)
py -3.14 Applications/make_figures.py

# focal mapping over the observed apps (reuses saved per-repo output; --rerun to redo)
py -3.14 -m pipeline.pym2t_focal

# the analysis run: see RUNBOOK.md for flags and the resume hazard
py -3.14 -m pipeline.batch_call_metadata --input pipeline/artifacts/applications_slim.csv --resume --keep-clones
```

---

## Reading the outputs

Things that are easy to misread, all of which have bitten:

- **`llm_tests_all.csv` is a subset of `llm_invokers_all.csv`**, not a parallel measure.
  Every row in it is also an invoker row. The two counts cannot be divided.
- **It measures static reachability, not non-determinism.** A direct row may still be
  mocked. The defensible claim is "a test whose own body calls an LLM API", minus mocks
  (not yet measured).
- **The raw CSVs still contain out-of-scope rows**: transitive rows and raw-SDK rows.
  Count through `analyze.py`, `make_figures.py` or `focal_map.observed_tests`, which
  apply the scope rules, not by summing a raw CSV.
- **Tests are concentrated.** The top 50 repos hold 83% of direct ND tests, and two
  (`deepset-ai/haystack-core-integrations`, `ericjiang18/Agent-Q-Mix`) hold 37%.
  Framework self-repos and framework/library repos are cut (`EXCLUSIONS.md` §7, §11, §12);
  some integration and observability repos remain (e.g. `DataDog/dd-trace-py`,
  `PostHog/posthog`).

## Method note

Exclusions are **documented filters, never deletions**. Raw CSVs and the full
`FrameworkDict` are preserved; every cut is a row in `EXCLUSIONS.md` naming the code that
enforces it. Keep it that way: the ledger is what makes the numbers reproducible.

## License

MIT
