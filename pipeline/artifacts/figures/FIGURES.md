# Findings — one question, one figure, one table

> **PROVISIONAL: llm_tests_all.csv, llm_invokers_all.csv predate call_metadata_all.csv by >1h, so the test-based figures (Q1, Q6) and the call-based figures (Q3, Q4, Q5) may describe different corpus states. Re-run once the batch driver has finished and the artifacts are in sync.**

Regenerate with `py -3.14 Applications/make_figures.py`.

## Q1. Do these applications actually call an LLM, and do their tests?

**81% contain an LLM call site; only 32% have a test that reaches one (n = 838).**

![Q1](figures/Q1_llm_usage.png)

Table: [`Q1_llm_usage.csv`](figures/Q1_llm_usage.csv)

## Q2. Which frameworks do these applications really import?

**langchain leads at 57% of 838 applications, with the raw OpenAI SDK close behind; measured from imports in the cloned source.**

![Q2](figures/Q2_frameworks_imported.png)

Table: [`Q2_frameworks_imported.csv`](figures/Q2_frameworks_imported.csv)

## Q3. Where do the LLM calls go?

**19% of 28,870 call sites bypass every framework and call a provider SDK directly.**

![Q3](figures/Q3_calls_by_framework.png)

Table: [`Q3_calls_by_framework.csv`](figures/Q3_calls_by_framework.csv)

## Q4. Are LLM calls pinned to deterministic settings?

**Almost never — temperature on 4.5% of 28,965 call sites, seed on 0.1%.**

![Q4](figures/Q4_determinism_knobs.png)

Table: [`Q4_determinism_knobs.csv`](figures/Q4_determinism_knobs.csv)

## Q5. How many determinism parameters does a single call set?

**84% of 28,965 call sites set none at all.**

![Q5](figures/Q5_knobs_per_call.png)

Table: [`Q5_knobs_per_call.csv`](figures/Q5_knobs_per_call.csv)

## Q6. How many tests reach a live LLM?

**11,657 distinct tests in 264 repositories (32% of the corpus), counting direct invocations only.**

![Q6](figures/Q6_nd_tests.png)

Table: [`Q6_nd_tests.csv`](figures/Q6_nd_tests.csv)

## Q7. Can the static analysis be trusted?

**95% of 838 applications have a usable call graph; the 41 without one are why test counts are reported direct-only.**

![Q7](figures/Q7_graph_health.png)

Table: [`Q7_graph_health.csv`](figures/Q7_graph_health.csv)

## Q8. Do these projects use an LLM evaluation framework?

**Only 4.7% of 838 applications call one.**

![Q8](figures/Q8_eval_adoption.png)

Table: [`Q8_eval_adoption.csv`](figures/Q8_eval_adoption.csv)
