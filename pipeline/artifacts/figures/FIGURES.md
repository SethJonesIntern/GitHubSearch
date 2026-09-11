# Findings — one question, one figure, one table

Observed population: **644 applications** in which a framework LLM call site was detected. Raw provider-SDK calls are excluded.

Regenerate with `py -3.14 Applications/make_figures.py`.

## F1. Do these applications use an LLM evaluation framework?

**38 of 644 (5.9%) call one. Two tools account for all but one of them; giskard and phoenix appear in none.**

![F1](figures/F1_eval_adoption.png)

Table: [`F1_eval_adoption.csv`](figures/F1_eval_adoption.csv)

## F2. Do LLM calls set the parameters that control non-determinism?

**23,814 of 24,077 call sites (98.9%) set none of the eight. `seed` is set at no call site in the corpus, and `model` — the most often set — is usually passed as a variable rather than pinned.**

![F2](figures/F2_determinism_parameters.png)

Table: [`F2_determinism_parameters.csv`](figures/F2_determinism_parameters.csv)

## F3. Which frameworks do these applications actually call?

**langchain leads at 60% of 644 applications. Measured from detected call sites, not from the search token, so a framework is credited only where it is actually invoked.**

![F3](figures/F3_frameworks_called.png)

Table: [`F3_frameworks_called.csv`](figures/F3_frameworks_called.csv)
