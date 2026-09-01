# Findings — one question, one figure, one table

Regenerate with `py -3.14 Applications/make_figures.py`.

## Q1. Do these applications actually call an LLM, and do their tests?

**92% contain an LLM call site; only 35% have a test that reaches one (n = 753).**

![Q1](figures/Q1_llm_usage.png)

Table: [`Q1_llm_usage.csv`](figures/Q1_llm_usage.csv)

## Q2. Which frameworks do these applications really import?

**langchain leads at 66% of 753 applications, with the raw OpenAI SDK close behind; measured from imports in the cloned source.**

![Q2](figures/Q2_frameworks_imported.png)

Table: [`Q2_frameworks_imported.csv`](figures/Q2_frameworks_imported.csv)

## Q3. Where do the LLM calls go?

**21% of 30,439 call sites bypass every framework and call a provider SDK directly.**

![Q3](figures/Q3_calls_by_framework.png)

Table: [`Q3_calls_by_framework.csv`](figures/Q3_calls_by_framework.csv)

## Q4. Are LLM calls pinned to deterministic settings?

**Almost never — temperature on 5.7% of 30,542 call sites, seed on 0.1%.**

![Q4](figures/Q4_determinism_knobs.png)

Table: [`Q4_determinism_knobs.csv`](figures/Q4_determinism_knobs.csv)

## Q5. How many determinism parameters does a single call set?

**83% of 30,542 call sites set none at all.**

![Q5](figures/Q5_knobs_per_call.png)

Table: [`Q5_knobs_per_call.csv`](figures/Q5_knobs_per_call.csv)

## Q6. How many tests reach a live LLM?

**11,774 tests call a model directly; a further 13,329 call a function that does.**

![Q6](figures/Q6_nd_tests.png)

Table: [`Q6_nd_tests.csv`](figures/Q6_nd_tests.csv)

## Q7. Can the static analysis be trusted?

**93% of 753 applications have a usable call graph; the 53 without one are why test counts are reported direct-only.**

![Q7](figures/Q7_graph_health.png)

Table: [`Q7_graph_health.csv`](figures/Q7_graph_health.csv)

## Q8. Do these projects use an LLM evaluation framework?

**Only 5.3% of 753 applications call one.**

![Q8](figures/Q8_eval_adoption.png)

Table: [`Q8_eval_adoption.csv`](figures/Q8_eval_adoption.csv)

## Q9. How far is a “non-deterministic test” from the model?

**Only 7.5% of 156,091 graph-reached tests invoke a model themselves; 75% are 3+ calls away.**

![Q9](figures/Q9_test_jump_depth.png)

Table: [`Q9_test_jump_depth.csv`](figures/Q9_test_jump_depth.csv)
