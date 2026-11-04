# Documentation map

This index clarifies document roles without moving existing files or research artifacts.

| Topic | Authoritative starting point |
|---|---|
| Scope, contributions and current measured/planned status | [README](../README.md) |
| Proposed research questions and figure roles | [Research questions](research-questions.md) |
| Artifact producers, inputs, storage and retention | [Artifact inventory](artifact-inventory.md) |
| Command side effects and data-PC boundaries | [Command guide](commands.md), then [RUNBOOK](../RUNBOOK.md) |
| Candidate legacy code, with mandatory review before removal | [Archive candidates](archive-candidates.md) |
| Dependencies and historical environment snapshots | [Dependency guide](../requirements/README.md) |
| Branches, PRs and synthetic CI checks | [CONTRIBUTING](../CONTRIBUTING.md) |
| Scope cuts and their enforcement | [EXCLUSIONS](../EXCLUSIONS.md) |
| Implemented historical methods | [METHODOLOGY](../METHODOLOGY.md) |
| Open measurement limitations and figure review | [METHODOLOGY_REVIEW](../METHODOLOGY_REVIEW.md) |
| Agent operating instructions and shared handoff | [AGENTS](../AGENTS.md), [CLAUDE](../CLAUDE.md) |

`PIPELINE.md` is historical architecture material. Its old current-state section is
not operational guidance. `COVERAGE_ANALYSIS.md` records earlier sampling coverage;
it is not guardrail or test coverage. Historical figures under `pipeline/artifacts/figures/`
remain available, while new guardrail figures live in a separate versioned directory.

No document title or candidate marker authorizes deleting code or data. Do not move
or rename existing paths until imports, commands, output paths and reproduction
dependencies have been checked on both machines.
