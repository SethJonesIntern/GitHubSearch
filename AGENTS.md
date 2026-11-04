# Repository guidance

Read README.md for current scope and **Contributions and their current status**,
then CLAUDE.md for the shared agent handoff and operational details. README.md's
contribution statement supersedes older non-determinism-focused descriptions.
Use CONTRIBUTING.md for the `main` / feature-branch / pull-request workflow and CI.
Use docs/README.md to locate the artifact inventory, command side effects, archive
candidates, proposed RQs and dependency roles. No research producer may run as cleanup.

Preserve these four contribution claims, with their maturity stated accurately:

1. The framework LLM invocation API/pattern inventory in `Wrapper/FrameworkDict.py`
   is a distinct research artifact, separate from framework discovery.
2. Empirical method-level prevalence of corresponding tests and correctness checks.
   Initial test correspondence is measured; guardrail prevalence remains pending.
3. Exploratory characterization of check contents, potentially a taxonomy.
4. A reproducible analysis pipeline and dataset. Attribute the vendored
   pyMethods2Test mapper to its authors.

Use production application methods containing detected direct framework LLM calls
as the primary guardrail denominator. Test correspondence does not prove a correctness
check, execution coverage, or guardrail. Separate checked properties from mechanisms
(assertions, metrics, model judges) and test-suite checks from runtime intervention.
Candidate categories are proposals until supported by inspection and validation.

Saved clones, slices and full focal outputs may exist only on the data PC and be
gitignored. Preserve them. Do not run destructive cleanup/reset/forced checkout to
synchronize this repository. Review reproducibility dependencies before removing
legacy code, especially the invocation registry, which is itself a contribution.
Archive-candidate markers are review notes only: do not delete, move or disable
those files. The owner requires another thorough check before any removal. New
figure generation must use saved inputs, isolated new output directories, and no
execution of discovery, batch analysis, Joern or focal-mapping producers.
