# GitHubSearch

Mining study of **correctness guardrails in agentic applications**: where applications
invoke LLMs through agentic frameworks, and how they check correctness in their test
suites and during execution.

The study follows **agentic frameworks → applications importing those frameworks →
application methods containing framework LLM invocations → associated guardrails**.
GitHub code search uses framework import patterns to discover candidate applications;
clone-side analysis checks their imports and locates uses of framework methods that
invoke LLMs.

`main` is the shared integration branch. Use feature branches and pull requests for
changes. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, CI checks and data
preservation rules, and [AGENTS.md](AGENTS.md) for agent guidance.

## Contributions and their current status

The paper has four contributions. Their status below distinguishes existing artifacts
from measurements and interpretations that still require work (updated 2026-11-04).

| Contribution | Deliverable | Current status |
|---|---|---|
| **1. Framework LLM invocation inventory** | A curated list of framework APIs/invocation patterns for locating LLM-related operations in application code. This is a research artifact in its own right, distinct from the framework discovery list. | Implemented in [`Wrapper/FrameworkDict.py`](Wrapper/FrameworkDict.py): `FRAMEWORK_CALLS` preserves the full registry and `SCOPED_FRAMEWORK_CALLS` selects the study's detection scope. Pattern precision and completeness remain subject to validation and documented exclusions. |
| **2. Empirical prevalence measurements** | Method-level counts of corresponding tests and associated test-suite correctness checks, runtime guardrails, and their overlap. | Initial production-method test correspondence is measured. Correctness-check and runtime-guardrail prevalence are pending; a mapped test does not establish either. |
| **3. Characterization of correctness checks** | Evidence of what checks evaluate and how they operate, potentially supporting definitions or a taxonomy of guardrail kinds. | Exploratory extension. Candidate categories are hypotheses for inspection, not observed findings or a validated taxonomy. |
| **4. Reproducible pipeline and dataset** | Framework-based discovery, application invocation detection, test mapping, and traceable analysis artifacts; guardrail analysis will extend these. | Existing discovery, call inventory, focal mapping and baseline outputs are available. Guardrail detection and validation are not yet implemented as a completed analysis. |

The primary empirical scope is **existence and prevalence**. Characterizing check
contents and deriving a taxonomy is a possible second contribution to the empirical
analysis. Neither the invocation inventory nor the analysis pipeline alone establishes
that an application implements a correctness guardrail. We use the published
pyMethods2Test tool with attribution; we do not claim that tool as our contribution.

## What counts as a guardrail?

We distinguish two categories by when the correctness check operates:

| Category | What we look for |
|---|---|
| **Test-suite guardrails** | Tests associated with an LLM-containing application method, with assertions, validators, or evaluations that check correctness. A mapped test alone establishes correspondence; its checks still need inspection. |
| **Runtime guardrails** | Checks during execution that detect a deviation and intervene before the result is accepted or acted upon, for example by rejecting, correcting, retrying, or requesting review. Logging alone does not establish intervention. |

The primary unit is an **application function or method containing a detected framework
LLM invocation**. Report the share of these methods with associated checks, using methods
as both numerator and denominator. Multiple tests or call sites in one method do not
increase its count. Test code and test-support code are excluded from production-method
denominators.

A guardrail may live inside the method or in associated caller, validator, or downstream
code. Establishing that association and identifying the checked property are part of the
analysis; proximity or a matching library name alone is insufficient evidence.

When inspecting contents, record **what is checked** separately from **how it is
checked**. Candidate properties include response availability, required structure,
expected content, behavioral constraints and semantic correctness. Mechanisms may
include deterministic assertions/validators, metrics and model-based judgments.
An `assert` can enforce a threshold on an LLM judge's score, and an evaluation framework
can perform a simple deterministic check. Library use and code complexity therefore do
not define mutually exclusive guardrail kinds. These are proposed dimensions for review.

---

## Status

The study was reframed around guardrails on **2026-11-03**. The existing discovery and
LLM-call inventory provide the starting population. **Guardrail prevalence has not yet
been measured.** The numbers below describe the saved September run, not a new guardrail
analysis.

| | |
|---|---:|
| population (`applications_slim.csv`) | 1,055 |
| processed by Stage 5 | 1,033 |
| clone failures (Windows illegal-filename bug, fixable) | 22 |
| analyzed, after documented scope cuts (`EXCLUSIONS.md`) | 753 |
| **observed applications** (≥1 framework LLM call site) | **644** |
| framework LLM call sites | 23,990 |
| tests whose own bodies contain detected framework calls | 9,700, in 233 apps |

Current work:

- Review the completed initial production-method test correspondence baseline from
  `pipeline/method_test_baseline.py`. The run considered 644 apps, retained 589, and
  compared both method groups within 571 apps. In those paired apps, **220/7,642
  LLM-containing methods (2.88%)** and **72,527/1,463,857 other methods (4.95%)** have a
  mapped test. Equal-weight per-repo means are 3.83% and 4.42%, respectively. These are
  correspondence rates, not correctness-check or execution coverage rates. Results and
  diagnostics: [`summary.json`](pipeline/artifacts/method_test_baseline/summary.json)
  and [`by_repo.csv`](pipeline/artifacts/method_test_baseline/by_repo.csv).
- Review the 55 baseline exclusions: 3 crashed mapping runs, 48 repositories with source
  parsing errors, and 4 with unmatched focal definitions. The whole-repo parsing rule
  is conservative and includes failures in templates/examples; exclusions may be
  recoverable and may bias the sample. Eighteen retained apps lack both method groups
  and are omitted only from the paired comparison.
- Inspect corresponding tests for correctness checks, then identify runtime checks and
  their interventions. Test correspondence is an input to guardrail detection.
- Review existing figures against the new questions. The initial review is in
  `METHODOLOGY_REVIEW.md`, section J. Existing figures are historical analyses.

---

## Scope rules

These apply to every reported number. Excluded rows stay in the raw CSVs: they are
filtered at analysis time, never deleted.

1. **Agentic applications only.** An application is in scope if it imports an agentic
   framework. A repo that reaches a model only through a raw SDK or raw HTTP is out.
2. **Raw provider SDKs are excluded.** `openai` and `anthropic` call sites are detected
   but not observed, measured or reported: no calls, parameters, invokers or tests.
3. **Direct invocation methods form the denominator.** `direct` / `transitive` is our
   Stage-5 terminology: a direct method contains a detected call; a transitive method
   reaches one through the static call graph. Transitive reachability does not prove
   execution or a correctness check. The existing restriction on transitive prevalence
   counts remains (`EXCLUSIONS.md` §18); tracing associated guardrails is future work.
4. **Use all corresponding tests.** A test need not contain an LLM call itself to check
   an LLM-containing application method. The historical direct-test subset is not the
   denominator or the eligibility filter for guardrail analysis.

---

## Where the documentation lives

Read these before trusting any number in the artifacts.

| Document | What it is |
|---|---|
| **`EXCLUSIONS.md`** | **The ledger, single source of truth.** Every repo dropped, framework exempted, pattern cut and scope decision, each with its reason and the code that enforces it. Start here. |
| `METHODOLOGY.md` | Historical implementation and results; not yet rewritten for the guardrail study. |
| `METHODOLOGY_REVIEW.md` | Methodological issues and the guardrail figure review (section J). |
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
8. **Focal mapping**: pyMethods2Test maps identified tests to the files and methods they
   appear to target. The production-method baseline uses all corresponding tests.

The existing stages supply evidence: AST matching locates invocations, pyMethods2Test
provides heuristic test correspondence, and Joern slices provide variable-dependence
context. **A detector for correctness checks and runtime interventions is still to be
designed and validated.** Framework evaluation-library usage and determinism parameters
do not, by themselves, establish a guardrail.

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

# historical figures (F1-F5); under review for the guardrail study
py -3.14 Applications/make_figures.py

# production-method test correspondence baseline (needs the data PC)
py -3.14 -m pipeline.method_test_baseline

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
- **Detected calls and mapped tests do not establish correctness guardrails.** A call
  in a test may be mocked; a mapped test may check something unrelated. Inspect the
  checked property and, at runtime, the action taken on failure.
- **Keep units consistent.** Guardrail and correspondence rates divide methods by
  methods. Call-site shares, test counts and repository shares answer separate questions.
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
