# Why the call graph is empty for some repos (pyan exclusions & the time-box)

Context: in the 92-repo pilot, ~22% of repos (9/41 at mid-run) came back with an
**empty call graph** (`cg_source = none` in `call_graph_health.csv`). This note explains
exactly why, so the transitive-invoker gap can be defended (or fixed) before presenting.

## TL;DR

- We build the transitive call graph with **pyan3 2.6.0**. pyan is **all-or-nothing**:
  it analyzes *all* files in one call, and if any single file fails, the whole repo's
  graph comes back empty.
- To survive that, we run an **exclude-and-retry loop**: catch the error, find the file
  it named, drop it, rebuild. Each retry **re-analyzes the whole repo from scratch**
  (pyan has no incremental API), so big repos get expensive fast.
- A **480 s time-box** stops the loop so one repo can't stall the run. Large repos
  (litellm 56k funcs, hermes-agent 62k, mlflow 29k, langflow 25k, …) blow the time-box
  after a few exclusions → empty graph.
- The empty-graph repos still keep their **direct** invokers (AST-based, no graph
  needed). Only the **transitive** closure is lost on those repos.

## Why a file gets excluded ("unparseable")

pyan has no parser of its own — it feeds each file to Python's built-in parser
(`ast`/`compile`). If a file wouldn't run as Python, pyan raises the same error Python
would. The failures split into two classes:

### Class 1 — the file is genuinely broken Python (repo's bug, not ours)
Real example from `Sumanth077/Hands-On-AI-Engineering`, file `processor.py`:

```python
import logging
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
import os
```

A **committed git merge-conflict marker** (`>>>>>>>`, `=======`, `<<<<<<<`). Python
reads `1d1e9f13…` as a number that starts with a digit then has letters, and throws
`SyntaxError: invalid decimal literal`. The file cannot import or run. Other members of
this class: notebook exports with non-Python lines, `3rd`-style bad literals, stray
version strings. These are bugs *in the analyzed repo*. (This is why one small tutorial
repo needed **41** exclusions — it's ~15 mini-projects, several with broken files.)

### Class 2 — the file is valid Python, but pyan crashes on it (pyan's bug)
Error looks like `Unknown scope '<qname>'`. The file parses and runs fine; pyan 2.6.0's
own scope resolver chokes on certain valid constructs (deeply nested lambdas /
comprehensions, some decorator patterns). This is the class the **large** repos mostly
hit.

## Why one bad file kills the whole repo

pyan builds a single call graph from all files at once:

```python
CallGraphVisitor([file1, file2, ...], root=...)   # parses + resolves + builds, all here
```

Everything happens inside that one constructor: parse every file, resolve names
*across* files, build the whole-program graph. If any file throws, the constructor
throws and **all** the work — including the 186 perfectly good files — is discarded. You
get a completely empty graph, not a partial one.

## Why it "restarts" on every exclusion

pyan exposes **no incremental interface** — no add-file, no remove-file, no skip-on-error.
The only lever we have to drop a bad file is to **call the constructor again with a
shorter file list**. That re-parses and re-resolves *all the good files too*, from
scratch, every time.

- Cheap for a 472-function tutorial repo.
- Brutal for litellm (56k funcs): 3 exclusions ≈ 4 full analyses of 56k functions →
  past the 480 s budget → gives up → empty graph.

So the big repos aren't failing because they have hundreds of broken files — they're
failing because **re-analysis cost × a few exclusions** exceeds the time-box.

## The 9 empty-graph repos (mid-run snapshot)

| repo | functions | reason |
|------|-----------|--------|
| hermes-agent | 61,971 | too big — timed out |
| litellm | 55,986 | too big — timed out (2 exclusions in) |
| mlflow | 28,730 | too big — timed out |
| langflow | 25,474 | too big — timed out |
| marimo | 17,508 | too big — 13 exclusions then timed out |
| angr | 14,763 | hard parse error, couldn't localize the file (0 exclusions) |
| deer-flow | 12,689 | hard parse error, couldn't localize the file (0 exclusions) |
| phoenix | 11,114 | too big — 9 exclusions then timed out |
| Hands-On-AI-Engineering | 472 | hit the 40-file exclusion cap (broken tutorial files) |

7 of 9 are the "too big → timed out" case. Several (angr, marimo, litellm, mlflow) are
frameworks/tooling rather than applications, and some have very few direct invokers
(angr 2, marimo 11), so a few are marginal to the population anyway.

## Fixes (not mutually exclusive)

### Fix A — `ast.parse()` pre-filter (recommended, cheap, general)
Split the cheap step from the expensive one:
1. Before handing files to pyan, run each through `ast.parse()` — fast, per-file, no
   graph work — and drop the ones that fail. This removes the **entire Class-1 bucket in
   one pass**.
2. Call pyan **once** on the clean file set.

This eliminates restart-on-exclusion for every syntax-broken file, so the slow
exclude-and-retry loop only ever fires for the rarer Class-2 `Unknown scope` files. For
the 7 timed-out repos this is very likely the difference between "empty" and "works in
one pass," because their cost is dominated by re-parse, not by many broken files.

### Fix B — Joern fallback for repos still empty
Joern builds one CPG in a single per-file-resilient pass (no re-parsing, no N× blowup),
so a 56k-function repo that times out pyan is normal for it. Use it only when pyan still
returns empty; record `cg_source = joern_fallback`. Joern under-resolves dynamic Python
vs pyan, so its transitive counts are a **lower bound** — hence fallback, not default.
Prototype: `pipeline/joern_call_graph.py`.

## What to tell the professor

- Direct LLM invokers are captured for **every** repo (AST-based). The gap is only in
  **transitive** invokers, and only on repos where the static graph couldn't be built.
- The `graph_usable` / `cg_source` columns in `call_graph_health.csv` make the gap
  explicit and auditable — empty-graph repos can be excluded or flagged in any
  transitive analysis rather than silently counted as zero.
- The failures are **systematic and explained** (repo-side broken files + pyan's
  all-or-nothing + re-analysis cost on huge repos), not random detector error, and the
  `ast.parse` pre-filter is a targeted fix.
