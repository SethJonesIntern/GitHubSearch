# GitHubSearch

Mining study of **non-determinism in LLM-based applications**: which agent frameworks
real applications are built on, where those applications invoke a model, whether they
pin the parameters that make a call reproducible, and whether their tests exercise that
code.

The population is mined from GitHub, cloned, and analyzed statically — AST pattern
matching for call sites, a pyan3 call graph for transitive reach, and Joern CPGs for
per-variable program slices.

---

## Status (2026-08-18)

The full-population run is **complete**.

| | |
|---|---:|
| population (`applications_slim.csv`) | 1,055 |
| processed | 1,035 |
| clone failures (repos deleted/made private mid-run) | 20 |
| analyzed, after 3 name-based exclusions in `analyze.py` | **1,032** |
| LLM call sites with argument metadata | 36,370 |
| repos with a usable call graph | 90.5% |

Headline result: **83.9% of LLM call sites set none of the eight determinism-relevant
parameters.** Temperature is set on 5.6% of calls, seed on 0.13%.

---

## Where the documentation lives

Read these before trusting any number in the artifacts.

| Document | What it is |
|---|---|
| **`EXCLUSIONS.md`** | **The ledger — single source of truth.** Every repo dropped, framework exempted, pattern cut, and scope decision, each with its reason and the code that enforces it. Start here. |
| `PIPELINE.md` | Stage-by-stage design of the whole system (Stages 1–7). |
| `RUNBOOK.md` | How to run Stage 5 + Stage 6: the command, every flag, the Windows/Joern gotchas, and how to stop and resume without duplicating rows. |
| `COVERAGE_ANALYSIS.md` | How much of the population the analyzed frameworks cover, and the units that number is in. |
| `CALL_GRAPH_EXCLUSIONS.md` | Why 9.5% of repos have an empty call graph, and the two holes in the pyan time-box. |

---

## Layout

| Path | Role |
|---|---|
| `pipeline/` | The current system. Stage drivers (`batch_call_metadata.py`, `slice_repo.py`, `run.py`), paths, and the eval-framework dict. |
| `Wrapper/` | The analysis engine. `FrameworkDict.py` (per-framework invocation patterns), `transitive_invokers.py` (AST index, seed matching, pyan call graph, closure), `false_positives.py` (the five FP tiers), `call_metadata.py` (per-argument extraction). |
| `Applications/` | Discovery (`search_candidates.py`), population shaping (`slim_applications.py`, `keep_frequency.py`), and reporting (`analyze.py`, `plot_*.py`). |
| `Frameworks/` | Stage-1 framework discovery. |
| `pipeline/artifacts/` | All outputs. Gitignored clones live in `pipeline/repos/`. |

## The stages

1. **Framework search** — find agent frameworks, derive their import names.
2. **Application search** — find repos importing those names.
3. **Frequency table** — apps per framework ecosystem.
4. **Application list** — slim the candidates to trustworthy names; flag scope.
5. **Invoker + call extraction** — clone each app, match invocation patterns, build the call graph, extract per-argument metadata.
6. **Joern slicing** — per-variable SubPDGs for each LLM-invoker function.
7. **Semantic evaluation** — eval-framework usage, piggybacked on the invoker search.

---

## Requirements

- **Python 3.14** — required; pyan3 2.6.0 (the call graph) does not work on older versions.
- `py -3.14 -m pip install -r requirements.txt`, plus `pyan3==2.6.0`.
- **Joern** — Stage 6 only. On Windows pass the `.bat`; the extension-less script fails with `WinError 193`.
- **`GITHUB_TOKEN`** in `.env` — discovery stages only. Analysis runs on local clones and needs no token.

## Common commands

```bash
# the standard report (population, tests, determinism knobs, graph health)
py -3.14 Applications/analyze.py

# regenerate the population + scope flags
py -3.14 Applications/slim_applications.py

# the analysis run — see RUNBOOK.md for flags and the resume hazard
py -3.14 -m pipeline.batch_call_metadata --input pipeline/artifacts/applications_slim.csv --resume --keep-clones
```

---

## Reading the outputs

Three things are easy to misread, and all three have bitten:

- **`llm_tests_all.csv` is a subset of `llm_invokers_all.csv`**, not a parallel measure.
  Every row in it is also an invoker row. The two counts cannot be divided.
- **It measures static reachability, not non-determinism.** A `transitive` row means a
  path exists in the call graph, not that the test executes it; a `direct` row may be
  mocked. The defensible claim is "a test whose own body calls an LLM API," which is the
  direct subset, minus mocks (not yet measured).
- **Transitive counts scale with repo size.** Among repos with a usable graph, invoker
  count correlates with function count at r ≈ 0.57. Report direct counts; treat
  transitive separately.

Counts are also concentrated in repos that are not applications — framework orgs,
integration packages, and observability vendors. The top 50 repos hold 80% of all direct
LLM-invoking tests. `EXCLUSIONS.md` §7 covers framework self-repos; the vendor and
integration layer is not yet filtered.

## Method note

Exclusions are **documented filters, never deletions**. Raw CSVs and the full
`FrameworkDict` are preserved; every cut is a row in `EXCLUSIONS.md` naming the code that
enforces it. Keep it that way — the ledger is what makes the numbers reproducible.

## License

MIT
