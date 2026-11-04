# Temporary home-PC context handoff

Created 2026-11-04 at the user's explicit request. Read this entire file, retain the
decisions and pending task in the active conversation, then delete **only this
temporary handoff file**. Its deletion is authorized; no other cleanup is authorized.
Do not replace CLAUDE.md with this file or retain it as permanent documentation.

## First: protect the data

Work in the existing checkout containing the saved research data. Inspect local
changes before switching branches or pulling. Fetching refs and reading a remote
file with `git show` can supply branch context without changing the working tree.
Do not force a checkout if local files would be overwritten.

Preserve all tracked, ignored and untracked research files: clones, Joern slices,
full PyMethods2Test outputs, progress files, backups, reports and historical figures.
Do not rerun discovery, batch analysis, cloning, Joern, focal mapping, audit refreshes
or artifact generators. Do not delete, move, overwrite, clean, reset or force-checkout
research files. The immediate task requires reading saved CSVs only. New reports,
if needed, must go to a new explicitly named location without replacing old outputs.

Deleting this temporary handoff is the sole exception to the no-deletion boundary.
If it was read via `git show` and is absent locally, do not switch or modify the
data checkout just to delete it; arrange its tracked removal through a safe feature
branch/PR. Retain the active context before removing the file.

## Repository and current work

Repository: `SethJonesIntern/GitHubSearch`.

`main` is protected and contains the consolidated research. Future changes use
feature branches and PRs with required Linux/Windows checks. The former main is
preserved on `archive/main-before-consolidation-2026-11-04`; the separate
`issue-3-eval-framework-search` branch remains unmerged in PR #10. Fully merged old
branches, including PipelineB, were removed only after ancestry verification.

PR #19: https://github.com/SethJonesIntern/GitHubSearch/pull/19

Branch: `codex/guardrail-research-figures`.
Before adding this handoff, its latest commit was `81a6833`. Both required checks
passed for that commit. PR #19 was OPEN at the last inspection; verify its current
state rather than assuming it has merged or that newer commits have finished CI.

The branch adds six figures, an artifact inventory, command-side-effect guidance,
documentation-role and dependency guides, and comment-only archive markers. No old
code or research artifact was removed, moved or regenerated. Archive-marked source
ASTs were compared to main and are unchanged. All 27 synthetic tests passed locally.

Read README.md, AGENTS.md and CLAUDE.md. On PR #19's branch also read docs/commands.md,
docs/artifact-inventory.md, docs/archive-candidates.md, docs/research-questions.md and
requirements/README.md. Use `git show` when those files are absent on the data PC's
current branch; do not overwrite the working data checkout to obtain documentation.

Root requirements.txt and Wrapper/requirements.txt are preserved historical snapshots.
requirements/analysis.in is a draft declaration, not a validated full-analysis lock.
Do not upgrade the working PC environment as cleanup. The full research environment
targets Python 3.14; figure-only rendering was separately done on Python 3.12.14.

## Research scope and contributions

We study correctness guardrails in agentic applications:

1. Test-suite checks associated with LLM-containing application methods.
2. Runtime checks that intervene before a result is accepted or acted upon.

The primary empirical scope is existence/prevalence. Inspecting check contents and
deriving categories/definitions is an exploratory extension.

Four contributions must remain explicit:

- Framework LLM invocation API/pattern inventory in Wrapper/FrameworkDict.py, distinct
  from the framework discovery list. Generic pattern entries are not independently
  validated API methods; precision/completeness remain subject to review.
- Empirical method-level test/guardrail prevalence measurements.
- Exploratory characterization of correctness-check contents, potentially a taxonomy.
- Reproducible analysis pipeline and dataset. Attribute the published pyMethods2Test
  mapper to its authors rather than claiming it as our original tool.

The primary unit is a production application method containing a detected direct
framework LLM invocation. Multiple tests or call sites do not multiply the method.
Keep tests, methods, files, call sites and repositories distinct as counting units.
Mapped tests do not establish correctness checks or execution coverage. No identified
mapping does not prove no test. Guardrail prevalence has not yet been established.

Record the property checked separately from the mechanism. Candidate properties:
response availability, structure, expected content, behavioral constraints, semantic
correctness. Mechanisms may include deterministic rules/assertions, metrics and LLM
judges. An assert may enforce a judge-score threshold; an evaluation framework may
perform a simple check. These are proposed dimensions, not observed findings.

## Current evidence and denominators

- 644 observed applications across 18 observed framework ecosystems.
- 23,990 saved observed framework invocation sites.
- Current inventory scope covers 19 ecosystem groups, but one has zero observed
  applications; do not report 19 as the observed-framework count.

### All tests identified by PyMethods2Test

- 1,217,210 identified tests.
- 171,902 have focal-method mappings: **14.12%**, combining exact and fuzzy matches.
- The overall number with a focal FILE has **not been verified** from the full CSV.
  It is incorrect to apply the 80.9% figure below to this full population.

Exact naming rule: a test name ends with a called method name. Fuzzy fallback:
choose the most similar called name if string similarity exceeds 50. Neither rule
establishes semantic targeting or measured execution.

### Historical direct LLM-calling test subset

These 9,700 tests contain framework invocations in their own bodies. They are not
all tests that can test LLM-containing application code.

- 7,459 map to a focal file only.
- 391 map to a focal method as well: 265 exact, 126 fuzzy.
- 1,850 were not identified by the tool.
- Any focal file: 7,850/9,700 = **80.9%**, including the 391 method mappings.
- File association may fall back to the test file itself; this is not necessarily
  a mapping to production code under test.

### Initial production-method baseline

Saved in pipeline/artifacts/method_test_baseline/summary.json and by_repo.csv.

- 644 apps considered; 589 retained; 571 have both production-method groups.
- Excluded: 3 crashed mapping runs, 48 apps with source parsing failures, 4 with
  unmatched focal definitions. Whole-repo exclusions are conservative and include
  template/example failures; recoverability and selection bias need review.
- Paired pooled LLM-containing methods: 220/7,642 = **2.88%** have a mapped test.
- Paired pooled other methods: 72,527/1,463,857 = **4.95%**.
- Equal-weight repository means: **3.83% versus 4.42%**; mean paired difference
  approximately -0.60 percentage points.
- Exact-only sensitivity: approximately **0.77% versus 1.72%** on the same paired
  production-method denominators.

### Additional file-level associations computed read-only in this conversation

Using saved per-file test counts from pym2t_focal_calls.csv, counting each file once:

- 21,988 identified tests map to an invocation-containing file = **1.81%** of 1,217,210.
- These mappings cover 920 invocation-containing files in 276 apps.
- Within the 9,700 direct LLM-calling tests, 2,284 map to an invocation-containing
  file = **23.55%** of that subset.

Historical full direct-invoker inventory (includes test/support methods):

- 18,961 direct invoker definitions after historical observed-repo, valid-pattern
  and raw-SDK exclusions; deduplicated by repo/file/definition line.
- 262 have focal-method mappings = **1.38%** (saved historical F5 result).
- 2,536 reside in a file targeted by a mapped test = **13.37%**.

A different count from the cleaned saved call inventory, deduplicated by
repo/file/enclosing_qname, gives 18,911 methods and 2,534 in targeted files. This
identifier/population mismatch needs reconciliation before headline comparisons.
Do not mix these counts with the 7,642-method production baseline.

The user suggested method/file correspondence as a floor/ceiling. We clarified:
they are stricter versus broader association measures, not validated bounds on
actual testing, because mappings can be wrong or absent and tests may target other
methods in the file. Use a common production-method population for paper results.

## Joern discussion

Saved slices may help connect an invocation/result to an assertion or validator.
The current exporter keeps data/control-dependence edges within each method AST,
not a complete flow across separate method bodies. The sliced corpus is selected
from the LLM-invoker closure, not all source methods/tests.

Potential analysis: resolve test-to-method calls, then inspect dependence evidence
connecting returned values/state/exceptions to checks. For runtime guardrails, also
identify rejection/retry/correction/review before acceptance or action. A shared
slice or a nearby assertion alone is not validated correctness evidence.

Use original source lines and pdg_lines when available. Generated snippets include
added scaffolding and some fail to parse. Inspect saved samples first; do not rerun
Joern or regenerate the corpus for this handoff.

PyMethods2Test may provide complementary correspondence evidence for slice-based
findings, provided populations/relationships match. Similar aggregate percentages
do not establish accuracy. Compare overlap and disagreement and manually inspect
samples; do not report only when the tools agree.

## Figures and presentation decisions

New figures: pipeline/artifacts/guardrail_figures/2026-11-04-v1/.
PNG/SVG, CSV twins, captions, inventory export and manifest are saved. Builder reads
only saved inputs/source literals, refuses existing output directories and creates
files exclusively. All five input hashes were unchanged after rendering.

- G1 invocation-pattern inventory: current main evidence figure.
- G3 method-level test correspondence: current main evidence figure.
- G2 baseline eligibility: methods/appendix, not overarching story.
- G4 exact/fuzzy sensitivity: supplementary diagnostic, not main guardrail finding.
- G5 proposed RQs and G6 candidate check dimensions: conceptual, not measured results.

Five-minute presentation: question/scope; discovery and contributions; current
baseline; candidate properties/mechanisms; next steps and RQs. Keep exclusion details
for Q&A. The user is making the presentation; do not create a deck unless asked.

## Immediate next task: verify all-test file mapping

Using only the saved full pym2t_focal_tests.csv and related saved tables on this PC:

1. Verify total identified test rows and unique test identities; explain duplicates.
2. Count any focal file, file-only, file-plus-method, and no focal file.
3. Break file mappings down by file_rule, especially import, basename and self.
4. Distinguish mappings to production files versus test/support files where possible,
   documenting the classification and any uncertainty.
5. Report exact numerators/denominators and keep the 1.2M population separate from
   the 9,700 direct-test subset and method-based comparisons.

Read CSVs only. Do not regenerate or overwrite existing outputs. Explain findings
first. Any new report must have a fresh path. After absorbing this handoff, delete
only this temporary file as requested, preserving all active context and research data.
