# Framework Coverage — the honest number (2026-08-11)

**Hand this to the next chat. It explains how we corrected the "top-20 covers ~90% of apps"
figure so the denominator only contains REAL AI applications.**

---

## The headline

> **Of real AI applications, the frameworks we analyze cover 827 / 918 = 90.1%.**
> The remaining ~10% are real AI apps built on long-tail frameworks we don't measure.

This is computed over the current population file
`pipeline/artifacts/applications_slim.csv` (1,055 rows).

## Why the old "90.4%" was not trustworthy

The old figure (from `keep_frequency` / `framework_coverage.png`) was inflated on two axes
that happened to roughly cancel:
1. **Junk in the denominator** — apps that aren't AI apps at all were counted in the total.
   The worst offender is the token **`clai`**: it's supposed to be pydantic-ai's CLI, but as
   a GitHub search token it collided with totally unrelated repos — `binance-connector-python`,
   `py-stellar-base`, `huaweicloud-sdk`, `python-cwt`, etc. Those are **not AI apps** and must
   not be in the total. (This is the [[import_name_pollution]] problem.)
2. **Exempt frameworks in the numerator** — `omnigent` (out-of-process phantom) and `agentops`
   (observability, not an invoker) are in the old top-20 and were counted as "covered," even
   though we exempted them and never measure them.

The correction: **junk leaves the denominator entirely; exempt frameworks are not credited;
out-of-scope real AI apps stay in the denominator as "known uncovered."**

## The three buckets (of the 1,055 rows in applications_slim.csv)

| bucket | count | in denominator? | in numerator (covered)? | what it is |
|--------|------:|:---:|:---:|------------|
| **Covered** | **827** | ✅ yes | ✅ yes | imports a framework/eval tool WE ANALYZE (in-scope) |
| **Uncovered real AI** | **91** | ✅ yes | ❌ no | real AI apps on OUT-OF-SCOPE frameworks (metagpt, lagent, honcho, beeai_framework, agent_protocol, headroom, patchwork, adalflow, agency_swarm, superagi, dynamiq …). Also `agentops` (exempt observability — borderline). |
| **Not an AI app** | **137** | ❌ **removed** | ❌ no | junk collision tokens (`clai`), langchain **non-LLM utilities** (`langchain_text_splitters`/`_chroma`/`_qdrant`/`_tests`), the `omnigent` phantom |

- **Real AI apps (denominator) = 827 + 91 = 918.**
- **Coverage = 827 / 918 = 90.1%.**

## Definitions (so it's reproducible)

- **Real AI app** = imports at least one *discovered real framework* (a key in
  `FrameworkDict.FRAMEWORK_CALLS`, i.e. a framework we have patterns for) OR an eval tool
  (a key in `eval_calls.EVAL_CALLS`). Junk tokens, non-LLM langchain sub-packages, and the
  `omnigent` phantom are **not** keys → excluded.
- **Covered** = imports a framework in `FrameworkDict.IN_SCOPE_FRAMEWORKS` (the top-20 +
  langchain/autogen families + SDKs) or `EVAL_CALLS` — the set we actually match/measure.
- **Aliases** (legit submodule names rolled up to their parent, so real apps aren't lost):
  `agent_framework_foundry` / `_openai` / `_foundry_hosting` → `agent_framework`;
  `crewai_tools` → `crewai`.

## Reproduce it
```python
import csv, sys; sys.path.insert(0, "Wrapper")
from FrameworkDict import FRAMEWORK_CALLS, IN_SCOPE_FRAMEWORKS
from pipeline.eval_calls import EVAL_CALLS
REAL_AI = set(FRAMEWORK_CALLS) | set(EVAL_CALLS)          # denominator membership
COVERED = set(IN_SCOPE_FRAMEWORKS) | set(EVAL_CALLS)      # what we analyze
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

## What this means for the study (action items for next chat)

1. **The real-AI population is 918, not 1,055 / 1,064.** Use 918 as the denominator for
   coverage AND for prevalence stats (% of apps that invoke an LLM, are tested, etc.).
2. **Run only the 827 covered apps.** Restrict `applications_slim.csv` to apps that import an
   in-scope framework (with the aliases above). This drops the 137 junk (don't clone them —
   saves compute) and leaves the 91 out-of-scope out of the run (they're "known uncovered",
   counted only in the denominator). **The slim edit to do this was drafted but NOT yet
   applied** — add an `analyzed_hit()` filter to `slim_applications.slim_csv` keyed off
   `IN_SCOPE_FRAMEWORKS | EVAL_CALLS` + `ALIASES`, regenerate `applications_slim.csv`.
3. **`clai` is junk, NOT pydantic_ai.** An earlier memory note wrongly called it a
   pydantic_ai rollup — ignore that. `clai`-only apps are false positives.
4. **Already done (context):** framework/eval SELF-repos are excluded (EXCLUSIONS.md §7);
   the FP filter + eval dict are done; the pilot is a clean 84-app dataset. See memory
   `reference_handoff_2026-08-11`.

## The one-line claim for the paper
> Our analyzed frameworks cover **90.1%** of real AI applications (827 of 918); the
> uncovered ~10% are AI apps built on long-tail frameworks outside our top-20.
