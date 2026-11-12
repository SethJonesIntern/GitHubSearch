# GitHubSearch — project instructions

Empirical software-mining study of **non-determinism in the testing of agentic
applications** mined from GitHub. Pipeline: discover frameworks → GitHub code search for
apps importing them → clone + static analysis (LLM call sites, invoker closure,
determinism kwargs, tests) → Joern per-variable slicing → focal mapping (pyMethods2Test).

The full Stage-5 run is done (1,033 of 1,055 processed). Current work: data quality, and
test-to-code traceability. This file is the single handoff: keep it current instead of
writing dated `HANDOFF_*.md` files.

## Resume here — guardrail findings and laptop handoff (2026-11-12)

**Read this section before the historical September inventory below.** Work is on
`codex/invoker-result-checks`, remote `origin` =
`https://github.com/SethJonesIntern/GitHubSearch`. Pull this branch on the laptop;
do not re-clone or download the 107 GB clone tree. This handoff and the accompanying
tracked evidence are sufficient to inspect the current findings and prepare human
sampling. Full-corpus source/slice regeneration still needs the PC.

### User instructions and decisions to preserve

- Two arms: **test-suite guardrails** (pyMethods2Test correspondence plus
  source/Joern evidence) and **runtime guardrails** (response slices plus source).
- Use the same production-method frame when comparing arm-level identified rates;
  do not confuse tests, call sites and methods. No guardrail scope cuts were made.
- **Do not edit/remove CSVs that are not tracked by GitHub.** Existing CSVs were
  preserved during handoff packaging; adding their existing bytes to this commit
  makes the data available on the laptop. Prefer new versioned JSON evidence for
  additional work. No producer should overwrite an existing artifact directory.
- A stale queued agent instruction once appended explanatory text to three
  Mixseek rationale cells in `test_annotations.csv`. Seth explicitly authorized
  restoring only those three fields; restored and verified, all other fields and
  all 206 rows unchanged. No cleanup approval remains pending. The related JSON
  summary contains additional explanatory fields; this is not a count change.
- Seth will conduct **human sampling**. Current labels are assistant/LLM source
  interpretations, not independent human ground truth. Deterministic scripts
  discover candidates, validate identities/hashes, join and count; they do not
  independently prove semantic labels. Agents are authorized; spec-harness is not
  required for this work.

### Current measured detections — freeze these units and limitations

Primary frame: **7,642 production-labelled LLM-containing methods in 571 repos**,
from the existing complete-case baseline. The original baseline includes 589 repos
(18 have no eligible methods) and omits 55 unresolved repos: 52 inventory/snapshot
mismatches and 3 incomplete mapping runs. This is **not a corrected census of all
644 observed applications**. Known false-positive framework calls, production/test
mislabels, vendored copies, tutorials and examples remain in this frozen frame.

| Identified method category | Runtime | Mapped test-suite evidence |
|---|---:|---:|
| Simple detected only | 74 (0.968333% of N) | 121 (1.583355%) |
| Complex detected only | 10 (0.130856%) | 0 identified in mapped frame |
| Both detected | 2 (0.026171%) | 0 identified in mapped frame |
| **Any identified automated guardrail** | **86 (1.125360%)** | **121 (1.583355%)** |
| No verified evidence | 7,556 | 7,521 |

Runtime any-simple = 76 methods (0.994504%); any-complex = 12 (0.157027%),
overlapping in two methods. Among the 86 identified runtime methods, composition
is 86.046512% simple-only / 11.627907% complex-only / 2.325581% both. This describes
the discovered set; complex discovery was purposive, so it is **not population
composition**. Likewise do not claim all test-suite guardrails are simple.

Correspondence is **220/7,642 methods = 2.878828%**, with 461 distinct saved mapped
tests. It is not a guardrail rate. Of the 220 methods, 121 have identified output
checks and 99 have no confirmed qualifying check in reviewed evidence; the 99 are
not proven to lack assertions, testing or guardrails. Source-positive test evidence
has 214 primary annotations across 192 distinct tests. **203/214 annotations use
mocks**, spanning 114 methods / 181 tests. Seven annotations are configured-live
(not executed), and four have unknown/unresolved fixture configuration. These
are execution-mode labels, not evidence of live semantic quality.

Mapping sensitivity: 31 source-positive methods have an exact method-name mapping;
21 retain both import-based file and exact-name method mapping. Broader fuzzy and
basename associations are included only with source review and remain labelled.
Unresolved associations flag 7 runtime methods and 46 test methods; these flags
can overlap positives. No annotations exist for 7,445 runtime / 7,422 test methods.
No detection is not absence. No human precision/recall estimates exist yet.

Earlier **about 1% Joern** = 106/10,922 production-labelled **call sites** with
explicit same-response-slice checks (0.9705%). This is a different unit/discovery
rule from 86/7,642 methods. Prior 47 asserts / 52 conditions / 7 schema counts are
syntax/API categories, not simple/complex semantic composition. Do not replace
either old number silently or combine their numerators/denominators.
The 106 is specifically `explicit_slice_supported` in tracked
`2026-11-04-candidate-tiers.csv`, not the old raw summary's 127 slice-supported
sites (1.16%, includes weaker checks). Of the 106, 22 have obvious test contexts;
the sensitivity tier outside those contexts is 84. Canonical labels were not cut.

### Definitions and source-review procedure

A qualifying association is **protected LLM output/proposed action -> acceptability
check -> verdict/predicate -> failure consequence**. Test consequence is failure
of the test; runtime consequence intervenes before protected acceptance/consumption
(reject, raise, stop, retry, rewrite, suppress or fallback). Mapping, slice
membership, parsing, logging or a reported score alone does not qualify.

- **Simple:** explicit rule-based oracle: type, presence, equality, regex, schema,
  range, membership or deterministic domain constraint. Simple describes oracle
  kind, not code size, computational complexity or usefulness.
- **Complex:** a learned/model-based evaluator judges meaning/quality/relevance/
  groundedness/etc., and its verdict actually controls failure/intervention.
  Trace implementation, evaluator input, configured backend, verdict and use;
  names such as `critic`/`validate` are discovery hints only.
- **Unresolved:** backend, configuration, response provenance or consequence cannot
  be established. Human-only control is a separate type, not automated complex.
- **Auxiliary:** a schema/type/presence check solely packaging a semantic judge's
  verdict does not add a simple check on the underlying protected generation.
  Independently assessed properties of the original output can yield both types.
- **Mocks:** fixed expected-value tests of a mocked judge/client return are simple
  contract tests, not proof a semantic judge ran. Record production oracle and
  tested/mock behavior separately.
- `assert` is conditional enforcement (roughly `if not predicate: raise
  AssertionError`); `assert` vs `if` is syntax, simple vs complex is oracle kind.
  `assert judge_result.grade` can be complex. Python optimization can disable
  runtime asserts. Internal `deterministic_assert` mechanism names do not classify
  the oracle as simple.

Source review checks target identity, original response flow, fixtures/helpers/
callbacks, property and enforcement. Joern original-line co-membership corroborates
the association; saved refined slices do not retain full dependence edges, and
inserted support lines/synthetic `pass` are not proof of a dependence path.
Runtime content includes type/schema/presence contracts, allowed IDs, valid camera
trees/moves, quality/groundedness/relevance critics, rewrite/recheck and fallback.
Several semantic loops accept at retry-budget exhaustion or fail open; guardrail
presence does not prove all-path protection, configuration enablement or efficacy.

### Gensee complex test — keep this counterexample visible

`GenseeAI/open_deep_research/src/legacy/tests/test_report_quality.py`,
`test_response_criteria_evaluation` (definition 140): report -> evaluation model
`.invoke` at **240** -> `assert eval_result.grade` at **294**. Local evaluator
construction is at 32-47. This is a source-confirmed **complex test guardrail**.
It is outside the primary saved method mapping, not reclassified as simple.

Failure points and implemented recovery:

1. LLM call detection succeeded (four calls in that file). Saved `eval_result`
   Joern slice includes both 240 and 294. This is not a call or Joern-slice miss.
2. Old `response_guardrail_slices` discarded response provenance at unknown
   `Panel(...)` at 277, before logging at 284 and assertion 294. New code preserves
   explicitly uncertain alias provenance and emits
   `uncertain_flow_assertion_candidate`, with trace `[240,277,284,294]` and reasons
   for `Panel` and `log_outputs`. It is a pending review route, never an automatic
   check/guardrail increment. Explicit overwrite/mutation/deletion still kills
   provenance. Unknown mutators, aliases, container arguments, import shadowing,
   and mutation inside a predicate are covered by regression tests. Unsupported
   branch/loop merges remain conservative misses.
3. pyMethods2Test finds an imported focal file but no focal method: the test calls
   graph compilation/execution, not the contributing production-node names. Do
   not alter the vendored tool or invent a direct mapping.
4. New `pipeline/guardrail_test_review_queue.py` retains source test-context checks
   even without a saved focal mapping. It reads the supplied frozen call snapshot
   and emits fresh **JSON only**, with identity, trace, original text, hashes,
   uncertainty and mapping status. It does not classify semantics or assign a
   protected production method. Tests without saved calls can still be missed.

Supplemental source association documents **composite, branch-conditioned report
contributors**, not four independently tested focal methods: default `multi_agent`
branch uses `src/legacy/multi_agent.py::188` supervisor and `::353` research_agent;
optional `graph` branch uses `src/legacy/graph.py::268` write_section and `::356`
write_final_sections. All four exist in the frame. Keep outside the primary
method numerator until a consistent composite-association rule is agreed.

### Response use — latest discussion, NOT implemented as a scope filter

Seth pointed out an LLM response may not affect a method's return. Relevant tests
can instead check response-dependent tool arguments, decisions, state changes or
other effects. A return-status assertion may not validate the generated plan.
Discarding an agent's final return does not prove responses are unused internally.

Proposed separate use labels: return-value contribution; decision/side-effect
contribution; check-only use; demonstrably no relevant use; unresolved. **Classify
use independently of guardrail presence.** "Keep if a guardrail exists, otherwise
cut" would select on the outcome and inflate prevalence. Retain broad inventory;
a secondary applicability denominator could exclude proven-unused responses only
under a prospective agreed rule. No such labels, exclusions or new rates exist.

### Data index — these are shipped with this branch

All paths below are under `pipeline/artifacts/response_guardrail_slices/`:

| Directory | Contents / how to use |
|---|---|
| `2026-11-04-v1/` (already tracked) | Frozen 23,990 call rows, original checks/routes/diagnostics, summary and manifest. Source of earlier call-site statistic. |
| `2026-11-11-examination-v1/` | 21 purposive call contexts / 19 repos; original excerpts, review CSV and provenance. |
| `2026-11-12-complex-checks-v1/` | 24 semantic-check/control locations / 19 repos; lexical candidates, exact excerpts and hashes. Not 24 protected production methods or a probability sample. |
| `2026-11-12-test-complex-checks-v1/` | Gensee complex pytest, AgentEvals tutorial and negative/mock examples; saved source/slice evidence and detector diagnostics. |
| `2026-11-12-two-arm-methodology-v1/` | `codebook.json`, `readiness.json`, full 7,642-row `method_assessment.csv`, 461-row `mapped_test_edges.csv`, manifest. Original unassessed labels/status are frozen setup history. |
| `2026-11-12-prevalence-source-audit-v1/` | **742 annotation rows**: runtime201 + test541 in `annotations.csv`; separate strict/extra/supplemental runtime and two test annotation sets; review candidates, hashed extra source excerpts, adjudication and sensitivity. This is the annotation input. |
| `2026-11-12-method-prevalence-v1/` | **Current identified results:** `summary.json`, `REPORT.md`, 7,642-row `methods.csv`, `verified_guardrails.csv`, rejected/unresolved/worklist tables, diagnostics, manifest. 88 runtime and 214 test primary annotation records, not necessarily atomic independent checks. |
| `2026-11-12-gap-bridge-v1/` | Real-source `test_review_queue.json` recovers Gensee 240->294 unmapped; `gensee_composite_association.json` records four conditional contributors. No headline rerun. |
| `2026-11-12-laptop-evidence-v1/` | Portable original-source excerpts for all 473 files referenced by annotations; separate Gensee association excerpts; bundle manifest hashes packaged data. **Excerpts, not complete clones/slices.** Original-file hashes identify PC snapshots; excerpt hashes verify portable text. |

`2026-11-12-laptop-evidence-v1/bundle_manifest.json` lists file paths, byte counts
and SHA-256 values for the portable artifacts and required already-tracked inputs,
plus the frozen count summary and validation results. It excludes itself from its
file-hash list. `sha256` identifies the original packaging bytes;
`git_blob_sha256` identifies repository bytes across machines (one older text JSON
input undergoes normal CRLF conversion). Artifact CSV bytes are preserved.
Review source excerpts by original repo/path/line, not as executable
replacement files or complete negative-case source coverage.

Review coverage: all 187 original runtime candidate method groups + 13 supplemental
producer methods (3 overlap) = 197 annotated runtime methods. The supplemental
semantic search was purposive. Test work assessed 206 saved returned-check rows in
126 tests plus 335 remaining mapped test edges. The first 126 were candidate-check
reviews, not an exhaustive review of every assertion. Source hashes/ranges were
verified across 473 files. Keep unknowns, rejected mappings and missed-case strata.

Frozen manifests intentionally describe old code/snapshots: the prevalence results
precede the uncertainty-recovery code, and **were not regenerated after that fix**.
Do not rewrite historical manifests/codebook statuses to imply a newer validated
run. Some manifest inputs (`method_test_baseline/methods.csv`, 290 MB
`pym2t_focal_tests.csv`, saved slice index/PDG files) are gitignored and PC-only;
the reduced method frame/edges and portable evidence above travel with this push.

### Implemented code and validation

- `pipeline/response_guardrail_slices.py`: uncertain provenance retained as routes;
  separate `uncertain_review_candidates` summary/report lane; excluded from checks.
- `pipeline/guardrail_test_review_queue.py`: mapping-independent test candidate
  discovery over frozen calls, JSON-only fresh output, source/input hash checks.
- `pipeline/guardrail_prevalence_audit.mjs`: deterministic stdlib Node consumer of
  frozen frame/checks/mappings and source annotations. Exact identity joins,
  innermost method attribution, source hashes/ranges and auxiliary exclusion.
  Requires original clones to rerun source verification; use saved outputs on laptop.
- `specs/response-guardrail-slices.spec.md` and
  `specs/guardrail-prevalence-audit.spec.md`: bounded implementation contracts;
  do not treat their original environment/status descriptions as current evidence.
- Validation: **17 source-analyzer + 6 review-queue Python tests passed** with
  Python 3.14.3 and `-B`; **11 Node tests passed**, 1 synthetic symlink subtest
  skipped on Windows EPERM. Models, applications, validators and Joern were not run.
- PC's `py` launcher is absent from the tool PATH, but installed 3.14 was found at
  `C:/Users/Seth/AppData/Local/Python/pythoncore-3.14-64/python.exe` (needs normal
  filesystem access). Never substitute MSYS Python 3.12. Use `py -3.14` on laptop
  if its launcher is available; `-B` prevents tracked bytecode churn.

Laptop-safe synthetic checks (fresh temporary fixtures; no study CSV edits):

```text
py -3.14 -B -m unittest pipeline.test_response_guardrail_slices pipeline.test_guardrail_test_review_queue
node --test pipeline/test_guardrail_prevalence_audit.mjs
```

### Tomorrow's priorities

1. Read the current summary, codebook, adjudication, sensitivity and portable
   evidence; preserve the distinction between correspondence, detection and
   population prevalence. Confirm branch and pull status before editing.
2. Agree response-use/applicability labels and a composite E2E association rule
   before any denominator changes. Evaluate output-dependent effects as well as
   return values; unresolved is not unused. Do not apply new CSV cuts on laptop.
3. Prepare **human sampling** over positives AND apparent negatives/unmapped/
   unresolved cases. Freeze units, strata, selection seed/probabilities and coding
   rules before drawing the sample; keep independent human labels separate from
   assistant labels. Positive-only validation cannot estimate missed guardrails.
4. Assess precision, reviewer agreement and missed cases. Mapping and local-flow
   gaps can disproportionately hide complex guards; do not extrapolate purposive
   examples to population composition. Record mocks and budget/error bypasses.
5. Plan a PC-only, fresh-output recovery run and expanded source review after the
   protocol is fixed. Laptop lacks full clones/slices; do not run `audit_apps`,
   `--scan`, batch metadata, Joern, or focal reruns. Missing-source failures are not
   evidence of no guardrails. Version any regenerated results, document scope or
   detection changes, and compare against this frozen baseline rather than
   silently replacing the counts.

## Where the study stands (measured 2026-09-18)

**Current direction (2026-11):** the README reframes the study around correctness
guardrails. The September inventory below is historical, not guardrail prevalence.
On 2026-11-11, a targeted assistant examination inspected 21 call contexts in 19
repos (20 source files, all matching the November 4 source hashes). Evidence and
classifications: `pipeline/artifacts/response_guardrail_slices/2026-11-11-examination-v1/`.
This is a purposive static review, not a random precision sample or human validation.
Confirmed mechanisms include type/presence/schema contracts, domain invariants,
tool-selection checks, LLM judgment with stop/suppress behavior, and fallback or
corrective invocation. Some candidates are non-LLM shell/blockchain/simulation
operations; mocked tests and E2E helpers also require separate interpretation.
**Slicing limitations:** emitted snippets can replace omitted `raise`/`return`/`exit`
branches with synthetic `pass`; aliases split invocation and check across variable
slices (ViMax camera tree, Baserow formula); inserted support lines are not PDG
evidence. Refined JSONL omits dependence edges. Read original source to establish
intervention. Existing detector misses concrete guards (tic-tac-toe move validation,
reviewer helper flow) and underdescribes semantic gates as type assertions. No
canonical scope, raw data, detector, or prevalence count changed in this examination.
Next: preserve intervention and exception/caller behavior in guardrail evidence,
join alias flow with provenance, and validate invocation precision before prevalence.

**Complex checks examination (2026-11-12):** searched all 5,619 production-labelled
canonical call-source files and source-inspected 24 semantic-check/control locations
across 19 repos. Catalog, exact excerpts, hashes, and unclassified lexical candidates:
`pipeline/artifacts/response_guardrail_slices/2026-11-12-complex-checks-v1/`.
20 catalog locations have an exact canonical call and saved call-hit slices; the
other 4 require caller/helper association. This catalog is broader than the old
same-response-slice numerator, not an exhaustive count or a prevalence estimate.
Duplicates, tutorials, samples and vendored code are labelled. Mechanisms include
judge/stop, judge/rewrite/recheck, judge/suppress, and framework output tripwires.
Several loops finalize on budget exhaustion despite failed quality; Kodo fails
open on judge errors. ASTRA reviewer sampler backend attribution remains unaudited.
Schema checks on the protected output are simple deterministic checks; schema
packaging solely of an LLM verdict is auxiliary, not an additional simple guard
on the underlying output. F8 is an authored
illustration, not mined evidence. No existing detector, scope, or counts changed.

**Complex test checks (2026-11-12):** inspected 1,674 saved test/test-support
call-source files for evaluator/score assertions, then verified targeted examples.
`GenseeAI/open_deep_research/src/legacy/tests/test_report_quality.py` is a real
pytest quality gate: report -> evaluation LLM (line 240) -> `assert eval_result.grade`
(294). Saved eval_result PDG lines include both, but November 4 checks.csv has no
check for any of the file's four canonical calls. Source analysis conservatively
invalidates origins through unknown helper/display calls (Panel) before the assert.
Thus this is a source-detector miss despite usable saved slicing evidence.
The 5zjk5 AgentEvals trajectory-judge tutorial also asserts an LLM verdict, but is
labelled production; its test function is manually invoked. Distinguish both from
mocked Ragas score equality tests, DeepEval result-shape integration tests, and
TigerGraph evaluation reports with no semantic-score pass/fail threshold.
Evidence: `pipeline/artifacts/response_guardrail_slices/2026-11-12-test-complex-checks-v1/`.
No test guardrail prevalence estimated; all corresponding tests/evaluation suites,
including tests without direct LLM calls, still need an expanded assessment frame.

**Two-arm assessment setup (2026-11-12):** Seth requested test-suite guardrails
(pyMethods2Test + Joern) and runtime guardrails (Joern), with simple/complex
composition and a defensible methodology. Saved codebook/readiness/worklists:
`pipeline/artifacts/response_guardrail_slices/2026-11-12-two-arm-methodology-v1/`.
Reconciled existing complete-case baseline: 7,642 production LLM-containing methods
in 571 repos, 461 distinct mapped tests targeting 220 methods, no mapping-count
mismatches. Original baseline includes 589 repos (18 without eligible LLM methods)
and leaves 55 unresolved (52 inventory/snapshot mismatches, 3 incomplete mappings).
This is a preliminary frame, not a newly corrected full-population denominator.
Simple = explicit rule-based oracle (schema included); complex = learned semantic
oracle whose verdict controls failure/intervention. Syntax is a separate field.
Aggregate methods per arm as simple-detected-only / complex-detected-only / both,
with unknown/unassessed and other-oracle evidence retained. Compare both arms on
the same method population; call-site, test, and method denominators are distinct.
The frozen setup labels were unassessed; the current handoff above supersedes
that status for identified detections only. Population composition remains unestimated.
Prior 47/52/7 counts describe syntax/API categories, not semantic composition.
Require source-backed output -> checker -> verdict -> consequence association,
original-source intervention recovery, mapping/mock/configuration sensitivity,
and independent positive/negative validation before study-level percentages.

| | |
|---|---:|
| population | 1,055 |
| analyzed (prevalence denominator) | 753 |
| with ≥1 LLM call site | 695 |
| **observed apps** (≥1 *framework* call site; the paper's population) | **644** |
| framework call sites | 23,990 |
| call sites setting **no** determinism knob | 98.9% (`temperature` 0.1%, `seed` 0) |
| **direct ND tests** | **9,700 in 233 apps** |

The paper's numbers use the observed scope: `make_figures.POP` / `.CALLS` and
`focal_map.observed_tests`. Import those definitions; don't restate them. `analyze.py`
covers the wider 753 and gives 9,701 ND tests in 234 apps. The extra one is
`Portkey-AI/portkey-python-sdk`, whose only calls are false positives.

Regenerate: `py -3.14 -m pipeline.waterfall` (population + pending queues),
`py -3.14 Applications/analyze.py` (report + `analysis/*.csv`),
`py -3.14 Applications/make_figures.py` (F1–F3 + `FIGURES.md`).

## Scope rules — settled, do not re-litigate

1. **An agentic application is one that imports an agentic framework** (2026-08-27).
   A repo reaching a model only through a raw SDK (`openai`, `anthropic`, `ollama`,
   `litellm`) or raw HTTP is out. Stage 2 only searched framework names, so such a repo
   can only have arrived by a token collision.
2. **Raw provider SDKs are out of every number** (2026-09-18). `openai`/`anthropic` rows
   stay in the raw CSVs but are never observed, measured or reported: no calls, knobs,
   invokers or ND tests. This overrides the 08-27 note that raw-SDK calls inside agentic
   apps "stay measured". Raw Gemini is undetected, which is moot for the same reason.
3. **Direct only; transitive is out of scope** (2026-09-18, until Seth says otherwise).
   Transitive's sole permitted use is the **blast radius** of LLM use (share of an app's
   functions that can reach an LLM call). See README "Scope rule" and `EXCLUSIONS.md` §18.
   `direct`/`transitive` is our pyan term, not a pyMethods2Test one.
4. **Stage 2 is GitHub code search, not import matching.** `matched_frameworks` is the
   token that surfaced a repo, and it hits substrings: `camel`→camelCase,
   `agno`→agnostic, `notte`→`@nottest`, `clai`→claim, `omnigent`→`OmniGe(nt)ransformer`.
   **Never decide scope from `matched_frameworks`**; use `frameworks_imported`.
5. **A matched token is a collision only when the clone does not import it.** Check
   `frameworks.csv` (the Stage-1 set) before cutting on that basis. It saved `IBM/ares`
   (`deepteam`) and `fetchai/fetchai` (`uagents_core`); `notte_sdk` is *not* notte.

**Trap, hit twice:** `openai` and `anthropic` are keys in `FRAMEWORK_CALLS`, so a
raw-SDK-only repo *does* have invokers. That isn't a contradiction; check the `reason`
column of `llm_invokers_all.csv` for which patterns fired.

## Docs

| doc | what it settles |
|---|---|
| `EXCLUSIONS.md` | **the ledger**: every cut, exemption, pattern cut and scope decision, dated, with the enforcing code. Add a row whenever you exclude anything. §18 = transitive sweep |
| `METHODOLOGY.md` | what was done, stage by stage, with the paper's figures |
| `METHODOLOGY_REVIEW.md` | parked methodological issues; the agenda for the review pass, nothing settled |
| `README.md` | public overview, headline, the direct-only scope rule |
| `COVERAGE_ANALYSIS.md` | the three scope buckets and the 90.1% coverage claim |
| `RUNBOOK.md` | running Stage 5/6, flags, stop/resume without duplicate rows |
| `pipeline/next_runs/README.md` | targeted re-runs and why they need prep |
| `CALL_GRAPH_EXCLUSIONS.md` | why some repos have no pyan graph |
| `PIPELINE.md` | stage design. Its "CURRENT STATE" (2026-06-25) is stale; ignore it |
| `pipeline/third_party/pymethods2test/NOTICE.md` | provenance of the vendored tool |

## The audit sheet

`pipeline/artifacts/application_audit.csv`: one row per population repo (1,055). Each
pass owns its columns. `in_scope` / `notes` are hand-edited and always preserved.

```
py -3.14 -m pipeline.audit_apps                     # the sheet + counts   (PC only, see below)
py -3.14 -m pipeline.audit_imports --scan           # frameworks_imported -> in_scope (~15 min)
py -3.14 -m pipeline.audit_framework_check --scan   # framework-vs-app (~13 min)
py -3.14 -m pipeline.audit_zero_invokers            # why a repo has no invokers
py -3.14 -m pipeline.audit_framework_triage         # framework_triage.csv (reads clones)
```

| `in_scope` | analyzed stats | coverage denominator |
|---|---|---|
| *(blank)*: in scope | counted | counted |
| `uncovered`: real agentic app on a framework outside the top-20 | out | **in** (the tail the 90.1% measures) |
| `0`: cut | out | out |

`pipeline/cuts.py` reads `in_scope`; `analyze.py`, `keep_frequency.py`,
`plot_coverage.py` and `make_figures.py` read `cuts.py`. **Editing `in_scope` moves every
statistic on the next run.**

- **`audit_apps` needs the slices.** It derives `slice_status` from
  `os.scandir(SLICES_DIR)`; on a machine without `pipeline/artifacts/slices/` it blanks
  ~780 rows. Recover with `git checkout -- pipeline/artifacts/application_audit.csv`.
- `nd_tests` in the sheet is direct only. `nd_tests_transitive` is a raw column that
  nothing reads.
- Supporting sheets: `framework_triage.csv` (all 1,055 ranked by framework-likelihood;
  the score is a ranking aid, trust only the top ~58), `imported_by.csv`,
  `analysis/zero_call_triage.csv`, `analysis/no_framework_no_call.csv`.

## Focal mapping — pyMethods2Test

**Decided 2026-09-15: focal mapping uses pyMethods2Test** (Abdelmadjid & Dyer, MSR 2025,
Zenodo `14264519`, Apache 2.0), not our `pipeline/focal_map.py` (26.5% method-level match,
superseded; its `observed_tests()` is still the ND-test scope definition).

- Vendored **unmodified** in `pipeline/third_party/pymethods2test/`. Byte-identical to
  Zenodo (verified 2026-09-18).
- `py -3.14 -m pipeline.pym2t_focal` runs it over all 644 observed apps, reusing the
  per-repo JSON in `artifacts/pym2t/` (gitignored); `--rerun` re-runs the tool. It replays
  their loop with their own functions to record per-test detail, and checks the replay
  against their CLI output (0 mismatches).
- Outputs: `pym2t_nd_tests.csv` (direct ND tests → focal status) and
  `pym2t_focal_calls.csv` (call site → inside a focal method?) are tracked.
  `pym2t_focal_tests.csv` (1.2M tests, 290 MB) and `pym2t_runs.csv` are gitignored.
- **Environment:** their pinned `python-Levenshtein~=0.26.1` has no cp314 wheel. Install
  `fuzzywuzzy~=0.18.0` + modern `Levenshtein`. The driver refuses to run if `fuzz` has
  fallen back to `difflib`, which changes every ratio against their `> 50` cutoff. This
  deviation belongs in the methodology.

**How it works** (read from their source; the paper's HTML summary is wrong):
- **Focal file:** source files whose module the test imports. If none, a basename match,
  and every `.py` file (tests included) is a candidate. So it never returns "no focal
  file". Only `file_rule == "import"` is real evidence.
- **Focal method:** a name the test body calls. The test name must end with it, otherwise
  any name with `fuzz.ratio > 50`.
- **A test is only seen if** its file has a top-level `import pytest`/`unittest`, and it
  calls ≥1 name defined in the repo.

**Two different questions, two tools.** Our detector decides whether a test *calls* an
LLM (an ND test: a pytest test whose own body matches a framework invocation pattern).
Calling an LLM does not make the LLM the test's focus. **pyMethods2Test decides what the
test targets.** A test is **LLM-focal** only when its pyMethods2Test focal method contains
a framework LLM call site. Their focal methods are always repo-defined, so a test whose
only LLM call goes straight to the framework API can never be LLM-focal.

**Results on the 9,700 direct ND tests:**
- 19.1% are never identified by the tool. Mostly plain-`assert` pytest files with no
  `import pytest`.
- 76.9% get a focal file only; 49.4% via import.
- **4.0% get a focal method** (391 tests, 57 apps). 56% of those are
  `haystack-core-integrations`, 234 are `run`/`run_async`.
- **LLM-focal: 36 tests (0.4%) in 16 apps.** 10 are `haystack-core-integrations`;
  23 exact / 13 fuzzy method matches; 24 via import / 12 basename file matches.
- **Only 25 of 23,990 call sites (0.1%) lie inside the focal method of an ND test.**
- 3 repos crash on non-UTF-8 files (dd-trace-py, artiphishell,
  astrbot_plugin_self_evolution).

Not yet done: hand-check a precision sample of the 391; report with and without
haystack-core-integrations.

## To look at: found 2026-09-18, nothing decided or changed

- **Haystack `.run` matches non-LLM components.** Haystack's patterns are just
  `.run` / `.run_async`, and every Haystack component has `.run()`. By receiver name
  (unverified), ~1,356 of 2,603 haystack call sites (52%, 18 apps) are retrievers (490),
  embedders (333), converters (218), rankers (145), fetchers, cleaners, etc. All 152 calls
  setting `top_k` in F2 are retrieval `top_k` (haystack retriever/ranker 104, mem0
  `.search` 47), not sampling. Also touches haystack ND tests (2,218).
- **Chained calls share a call ID.** `call_id` is `file::line::col`, so in
  `llm.invoke(x).content.strip()` the real call and the chained `.strip()` collide.
  106 IDs (87 in the observed scope). `llm_calls_all.csv` kept the chained row, which is
  flagged `tier2_nonterminal`, so the real call is dropped from the call counts but kept in
  `call_metadata_all.csv`. That's why F2 says 24,077 call sites and F3/METHODOLOGY say
  23,990. Counting them would give 645 apps (+ `Portkey-AI/portkey-python-sdk`) and
  24,079 call sites.
- **F2's committed caption** calls `model` the most often set parameter; its own table has
  `top_k` (152) above `model` (92).

## Open work, in value order

- **H4. Fix `shallow_clone`** (`Wrapper/transitive_invokers.py:58`). 22 repos (20 in
  scope) "failed to clone" over a `:` in a filename, illegal on Windows. Git fetches
  everything, checkout aborts, and the function returns `False`, but the tree is on
  disk (`Azure/plan-search-chatbot`: all 36 `.py` present). Fix: on
  `CalledProcessError`, if `dest` holds `.py` files, warn, record `clone_partial`, and
  return `True`; genuine failures (auth, 404, timeout) still return `False`. Then
  re-queue. ⚠️ `batch_call_metadata` `rmtree`s the checkout before cloning, so **do not
  re-queue before the fix**. Truly gone: `agno-agi/pal`, `holetron/hindsight-mempalace`,
  `openJiuwen-ai/agent-store`. Denominator returns to ~773.
- **`clai` → `pydantic_ai` in `keep_frequency.EXTRA_MEMBERS` (line 46)**: still there
  and wrong. Inflates pydantic_ai 79 → 127 apps, #4 → #2. Remove it.
- **H5. Queue C: 56 repos import a framework but have no call site.** Detector
  questions, not cuts. The waterfall's 100% for C is arithmetic. ~10 import `litellm`
  (not in `IN_SCOPE_FRAMEWORKS`, can't produce a call site), plus the H4b shape.
- **H4b. Split-receiver detector gap.** `storyAura/astrbot_plugin_biliVideo` calls
  `await provider.text_chat(...)` on a receiver from `context.get_using_provider()` held
  in a wrapper. Missed because `.get_using_provider` was cut (EXCLUSIONS §4). Held, not
  cut (§15). Size how many of queue C share it.
- **H4d. Registry gaps.** `uagents_core` and `deepteam` are Stage-1 discovered but not
  `FRAMEWORK_CALLS` keys (only `uagents` is). Audit every `frameworks.csv` `import_names`
  against `FRAMEWORK_CALLS`.
- **H4c. 7 raw-SDK-only `uncovered` repos.** They have no invokers, so only
  `imports_matched_fw` can judge them. `audit_zero_invokers.py:154` limits `targets` to
  `zero_invoker == "1"`; widen it to `uncovered` rows (EXCLUSIONS §17b-ii).
- **Stage 6 slicing vs the direct-only rule.** The slicer is fed the whole invoker
  closure, transitive included (EXCLUSIONS §18). Decide: filter slice analysis to direct
  invoker functions, or rule it blast radius.
- **Joern: 10 repos still fail at 24g** (5 OOM, 4 process-exit). Heap isn't the
  binding constraint; needs a different approach.
- **H7. `audit_framework_triage --pypi`**: the one independent framework signal never
  switched on (desktop, it reads the clones).
- **H8. Second pyan pass with `PYAN_TIME_BUDGET_SEC=0`**, time-boxed subset only. Low
  priority now that transitive only feeds blast radius. PowerShell:
  `$env:PYAN_TIME_BUDGET_SEC = "0"` on its own line.
- Unverified whether done: Queue D (~53 lower-weight framework/library candidates,
  §12); `prepare_rerun` has no guard against `--apply` on the wrong CSV; no EXCLUSIONS row
  for 3 zero-Python repos dropped from the run queue (`wesleysimplicio/hermes-turbo-agent`,
  `erfanzar/Xerxes-Agents`, `memory-graph/memory-graph`).

## Machines

- **PC (desktop)** has the clone tree (`pipeline/repos/`, ~107 GB, 1,056 repos) and
  `artifacts/slices/`. Needed for: reading source to justify a cut, every `--scan`,
  `audit_framework_triage`, `batch_call_metadata`, slicing, `audit_apps`, `pym2t_focal
  --rerun`.
- **Laptop** has neither. The cut loop still works there: hand-edit `in_scope` →
  `analyze.py` / `keep_frequency.py` / `plot_coverage.py`. Never run `audit_apps` there.
- **Git LFS** tracks `llm_invokers_all.csv` (172 MB) and `llm_tests_all.csv` (87 MB).
  Without `git lfs install` you get pointer stubs and scripts fail confusingly. Free tier
  is **1 GB bandwidth/month**, and one fresh clone is ~259 MB of LFS. `git pull` into an
  existing checkout; don't re-clone. `git pull` before hand-editing the sheet on either
  machine; git can't merge two machines' edits to it.

## Conventions

- **`py -3.14` always** (pyan3 requires it).
- **Cuts are filters, never deletions.** A cut repo keeps its row and raw data; only
  `in_scope` changes. Every criterion lives in code with a comment and gets a dated row in
  `EXCLUSIONS.md`.
- **Evidence before cutting.** Read the actual matched code. Inference has been wrong
  repeatedly: omnigent, connectonion, haystack, clai, notte_sdk, deepteam.
- Distinguish **measured / projected / assumed** when reporting numbers.
- Prefer giving Seth a command to run over backgrounding a long job.
- Never run `batch_call_metadata` without `--resume`: it deletes every artifact and
  restarts the whole run. Re-runs: `pipeline/next_runs/README.md`.

## Gotchas

- **The Grep tool silently skips `pipeline/repos`** (gitignored). Use Bash `grep` or
  `rg --no-ignore` for the clones. A recursive `grep -r` from the repo root also walks
  `Applications/repo_cache` and the clones and will time out; scope it.
- Full-corpus scans take ~15 min at 8 workers. Use one alternation regex over the names
  you want, not match-everything-then-filter (90 min vs 15).
- Windows: `ProcessPoolExecutor` needs an `if __name__ == "__main__"` guard. Pass Joern as
  the `.bat`; bare `joern` gives `WinError 193`.
- Set `PYTHONIOENCODING=utf-8` before `analyze.py` if it dies on a non-ASCII description.
- 107 `__pycache__/*.pyc` files are tracked. Running or compiling a script dirties them;
  restore with `git checkout` rather than committing them.
- `core.autocrlf=true`, so LF-written files produce a harmless CRLF warning on commit.
