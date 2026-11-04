# Proposed research questions and figures

These are working questions for discussion, not finalized claims. README.md remains
the authority for contribution status. The current evidence supports correspondence
measurements; it does not yet supply validated correctness-guardrail prevalence.

| Question | Role | Evidence / missing work | Figure |
|---|---|---|---|
| **RQ1. How often do LLM-containing production methods have associated correctness checks in tests, at runtime, or both?** | Primary prevalence question | Mapped-test baseline exists. Inspect what tests check; identify runtime check association and intervention. Use a common method denominator and keep unassessed cases separate. | G5 describes proposed scope; actual prevalence figure awaits data. |
| **RQ2. How does test correspondence for LLM-containing methods compare with other production methods in the same apps?** | Supporting context | Initial paired baseline available. Mapping precision and exclusions need review. Descriptive differences do not prove true testing gaps. | G3 main comparison; G2 eligibility and G4 exact-only sensitivity support it. |
| **RQ3. What properties do these checks evaluate, and which mechanisms do they use?** | Exploratory characterization, potentially a taxonomy | Content inspection/coding and reliability validation pending. Availability, structure, expected content, behavioral constraints and semantic correctness are candidate properties. | G6 is a proposed coding framework, not an observed distribution. |

**Contribution 1** also needs a separate artifact figure: **G1** summarizes the
framework invocation-pattern inventory and its observed application use. Pattern
strings, package keys and framework ecosystems are different units. Generic matching
patterns are not independently validated API methods. The full registry retains raw
SDK provenance, while the observed study excludes those SDKs.

**Contribution 4**, the reproducible pipeline/dataset, supports every RQ through
saved inputs, CSV table twins and manifests with input hashes and software versions.
It does not itself establish guardrail presence or effectiveness.

The new builder reads only saved results and source literals. It refuses an existing
output directory. New outputs live under `pipeline/artifacts/guardrail_figures/`;
old figures remain unchanged. G1–G4 report existing evidence; G5–G6 are conceptual.

Reviewed figure set: [2026-11-04-v1](../pipeline/artifacts/guardrail_figures/2026-11-04-v1/FIGURES.md).
Each figure has PNG and editable SVG versions, a CSV table twin, and a caption/status
in the manifest. G1 shows 19 currently scoped ecosystem groups, of which 18 have
observed calls; the zero-observation group must not inflate observed-framework counts.
