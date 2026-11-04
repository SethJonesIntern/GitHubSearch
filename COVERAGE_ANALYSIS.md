# Framework Coverage — how much of the population do we measure?

*Last updated: 2026-08-18. Companion to `EXCLUSIONS.md` §9.*

**Read the units carefully: every number on this page counts APPLICATIONS, not
frameworks.** The study analyzes ~20 framework ecosystems (44 keys in
`FrameworkDict.SCOPED_FRAMEWORK_CALLS`, out of 76 discovered). Any three-digit
number here is a repository count.

---

## The headline

> **Of real AI applications in the population, the frameworks we analyze cover
> 827 of 918 = 90.1%.**

That is: 827 *applications* import something we measure. The remaining ~10% are
real AI applications built on long-tail frameworks that are out of scope.

## The three buckets (of 1,055 rows in `applications_slim.csv`)

| bucket | apps | in denominator? | measured? | what it is |
|--------|-----:|:---:|:---:|------------|
| **Covered** | **827** | yes | yes | imports an in-scope framework or eval tool |
| **Uncovered real AI** | **91** | yes | no | real AI apps on out-of-scope frameworks (`metagpt`, `lagent`, `honcho`, `beeai_framework`, `agent_protocol`, `headroom`, `patchwork`, `adalflow`, `agency_swarm`, `superagi`, `dynamiq`…) |
| **Excluded** | **137** | **no** | no | the matched name identifies no framework: collision tokens (`clai`), non-LLM langchain utilities (`langchain_text_splitters`/`_chroma`/`_qdrant`/`_tests`), the `omnigent` phantom |

- Real AI apps (denominator) = 827 + 91 = **918**
- Coverage = 827 / 918 = **90.1%**

## Definitions (derived from the dicts, not hand-listed)

- **Real AI app** — matched name (after aliasing) is a key in
  `FrameworkDict.FRAMEWORK_CALLS` or `eval_calls.EVAL_CALLS` → `real_ai_app=1`.
- **Covered** — matched name is in `FrameworkDict.IN_SCOPE_FRAMEWORKS` or
  `EVAL_CALLS` → `analyzed=1`.
- **Aliases** — companion packages rolled up to their parent:
  `agent_framework_foundry` / `_openai` / `_foundry_hosting` → `agent_framework`;
  `crewai_tools` → `crewai`. (Known incomplete — see the caveat below.)

Both flags are columns on every row of `applications_slim.csv`.

## Reproduce it

```python
import csv, sys; sys.path.insert(0, "Wrapper")
from FrameworkDict import FRAMEWORK_CALLS, IN_SCOPE_FRAMEWORKS
from pipeline.eval_calls import EVAL_CALLS
REAL_AI = set(FRAMEWORK_CALLS) | set(EVAL_CALLS)
COVERED = set(IN_SCOPE_FRAMEWORKS) | set(EVAL_CALLS)
ALIASES = {"agent_framework_foundry":"agent_framework","agent_framework_openai":"agent_framework",
           "agent_framework_foundry_hosting":"agent_framework","crewai_tools":"crewai"}
res = lambda ns: {ALIASES.get(n,n) for n in ns}
real=cov=0
for r in csv.DictReader(open("pipeline/artifacts/applications_slim.csv",encoding="utf-8")):
    ns = res([x.strip() for x in (r["matched_frameworks"] or "").split(",") if x.strip()])
    if ns & REAL_AI: real += 1
    if ns & COVERED: cov += 1
print(cov, real, f"{100*cov/real:.1f}%")   # -> 827 918 90.1%
```

## IMPORTANT CAVEAT — this is a search-metadata estimate

Every number above is computed from `matched_frameworks`, which records **the
GitHub-search token that found the repo**, not what the repo's code imports.
Measured on the completed run, **83% of analyzed repos import a framework that
their matched token never mentions.** The classification is therefore a
hypothesis, not ground truth.

The corpus has since been fully analyzed, so coverage can now be recomputed from
**frameworks actually detected in code** — that number supersedes this one and
should be produced before publication.

## What this does NOT say

- **It is not a run-set filter.** An earlier revision of this document
  recommended restricting the run to the 827 covered apps. **That guidance was
  tested and rejected**: repos matched only to out-of-scope tokens still make
  in-scope raw-SDK calls (`JetAstra/SDAR`, matched on `lagent`, contains 18
  `openai` + 4 `anthropic` calls). The full 1,055 were analyzed. Do not
  reintroduce a pre-run population filter.
- **`applications_analyzed.csv` (827 rows) was deleted** for the same reason. The
  run input is `applications_slim.csv`; scope is decided after analysis, from the
  `real_ai_app` / `analyzed` flags.

## Run status (for anyone reading a stale number elsewhere)

| | |
|---|---:|
| population (`applications_slim.csv`) | 1,055 |
| processed | 1,035 |
| failed (clone — deleted/private repos) | 20 |
| analyzed after the 3 name-based exclusions in `analyze.py` | **1,032** |

Numbers such as **921** appear in older notes; they are mid-run snapshots and
carry no meaning now.
