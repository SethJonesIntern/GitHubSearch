# Sprint Handoff — Framework/Invocation Cleanup (as of 2026-08-03)

Read this before the next sprint. It captures where the project is, what the last
sprint changed, the decisions behind those changes, and exactly what's next. Companion
docs: `RUN_PILOT.md` (how to run), `CALL_GRAPH_EXCLUSIONS.md` (empty-graph gap),
**`EXCLUSIONS.md` (the single canonical ledger of every exclusion/exemption/pattern-cut —
log new ones there)**. The auto-memory under `.claude/.../memory/` has one file per
durable fact — start with `MEMORY.md`.

> **UPDATE (2026-08-06):** §5.4 below calls omnigent a "phantom" — that was CORRECTED.
> omnigent is a real CLI-based (out-of-process) agent, now **exempted** (not measured),
> not a phantom. agentops likewise **exempted** (observability layer). See `EXCLUSIONS.md`
> §2 and memory `project_out_of_process_invocation`. The full invocation-pattern pass is
> done; current state + next steps live in memory `project_post_rerun_plan`.

---

## 1. What this project is

Empirical software-mining study of **non-determinism in the testing of LLM / agent-
framework applications** mined from GitHub. Pipeline stages: (1) discover frameworks →
(2) code-search for apps that import them → (5) static analysis (LLM invokers, call
sites, determinism knobs, tests) → (6) Joern program slicing. Real runs need **Python
3.14** (pyan3 for the transitive call graph). Run instructions: `RUN_PILOT.md`.

## 2. Current phase

**Cleaning framework detection + invocation patterns before the full run.** The last
sprint was almost entirely data-quality work on *which frameworks/apps are real* and
*what counts as an LLM invocation*. The 92-repo pilot has already run once, but its
invocation data is **STALE** (generated before all the cleanup below) — it must be
re-run before any analysis is trusted.

## 3. Population & scope (current numbers)

- **Population: 1,064 distinct applications** (was 1,142; phantoms removed — see §5).
- **Top-20 frameworks cover 90.4%** — see `pipeline/artifacts/framework_coverage.png`
  and `keep_frequency.csv`. Current top-20:
  langchain, pydantic_ai, langgraph, crewai, mem0, dspy, haystack, smolagents, autogen,
  agent_framework (MS), agno, camel, astrbot, graphrag, omnigent, agentops,
  semantic_kernel, swarms, swarm, notte.
- **Detection is scoped** to `SCOPED_FRAMEWORK_CALLS` (43 import-name keys = the ~20
  frameworks + langchain/autogen families + raw SDKs openai/anthropic + `agents`).
  The full 75-key `FRAMEWORK_CALLS` is retained as the Stage-1 discovery record but
  NOT matched. See `Wrapper/FrameworkDict.py` (`IN_SCOPE_FRAMEWORKS`).

## 4. Key mechanism to understand first

**Stage-2 matching is loose GitHub *code search* (token-based), not exact imports.**
`search_candidates.py` searches `"import X"`, `"from X import"`, `"from X."`. GitHub
tokenizes, so a match can be a substring/submodule co-occurrence, NOT a real top-level
import. Consequence: **a recorded `matched_frameworks` entry ≠ the app really imports
that package.** The only reliable check is *reading the actual matched code*. This is
the root of every phantom below. (Memory: `project_out_of_process_invocation.md`.)

## 5. What the last sprint changed (all done)

1. **Import-name grouping** (`keep_frequency.py EXTRA_MEMBERS`): `clai → pydantic_ai`
   (clai is pydantic-ai's CLI module; its apps are real pydantic_ai apps).
2. **Scoping**: added `IN_SCOPE_FRAMEWORKS` / `SCOPED_FRAMEWORK_CALLS` to FrameworkDict;
   wired through `engines.py` + `batch_call_metadata.py` (LLM pass + `COMBINED_CALLS`).
3. **Cut 2 phantom frameworks** (verified by reading matched code, then removed from
   `slim_applications.EXPLICIT_KEEP` + `keep_frequency.EXTRA_MEMBERS` +
   `FrameworkDict._IN_SCOPE_EXPLICIT`):
   - **cheshire-cat** — its tokens `cat`/`agui` are generic junk; apps matched their own
     `agui` modules / `cat` collisions, none import cheshire-cat.
   - **connectonion** — token `subagents` matched apps' OWN `X.subagents` submodules
     (e.g. `deepagents.middleware.subagents`), never the connectonion package.
   → population 1,142 → 1,064; clean top-20 = 90.4%.
4. **omnigent** — identified as a phantom too (token `omnigent` substring-matches the
   diffusion model class `OmniGenTransformer` → `omnige`**`nt`**`ransformer`; its 25
   "apps" are image/video-diffusion repos). **LEFT IN, not cut, per Seth's call.** Still
   0 invocations; treat with that knowledge.
5. **Invocation patterns reviewed & marked `# DONE` in FrameworkDict:**
   - **langchain** family — reviewed (kept; noise flagged for §6).
   - **autogen** family — added `autogen_core` `.create` / `.create_stream`
     (ChatCompletionClient — the real model call). `.send_message`/`.publish_message`
     left in (orchestration; candidate to drop).
   - **pydantic_ai** — added `model_request` / `_sync` / `_stream` / `_stream_sync`
     (the `pydantic_ai.direct` low-level model calls). Rejected `Model.request` (generic
     `.request` collides with HTTP; it's framework-internal).
   - **swarms** — `.run`/`.arun` are correct; its 0 invocations = **not sampled** (was
     below the pre-grouping top-20 cut), a sampling gap, not a pattern gap.
   - **notte** — added to `IN_SCOPE` (pattern `.run`); not sampled, so 0 in pilot.

## 6. Key decisions & findings (carry forward)

- **Phantom-detection heuristic:** the only phantom suspects were **0-invocation top-20
  entries**. Frameworks that fired invocations (langchain 2170 … agno 4065 … down to
  agentops 14, semantic_kernel 20) are confirmed real — collisions don't produce
  hundreds of matched call sites. The 0-invocation cluster (cheshire-cat, connectonion,
  omnigent = phantoms; swarms/swarm/notte = real-but-unsampled) is fully triaged.
- **Verify matches by reading code**, never by inference. Both omnigent (substring) and
  connectonion (submodule) fooled inference; only the actual matched lines settled it.
- **Receiver matters for invocation patterns** (bare verbs are per-framework): agno
  `.run` = real agent call; langchain `.run` = *tool execution* (`tool.run`), not LLM;
  langchain `.predict` = MLflow/sklearn model collision; graphrag `.search` = `re.search`;
  agentops `.init`/`.record` = telemetry. (Memory: `project_invoker_pattern_slimming.md`,
  audit at `pipeline/artifacts/invoker_legitimacy_audit.md`.)
- **Out-of-process invocation** (CLI subprocess / HTTP) is a *theoretical* blind spot
  for static method-matching, but **no confirmed in-study example survived** — the two
  candidates (omnigent, cheshire-cat) were both collisions. One-line methodology caveat.
- **Determinism headline (from stale pilot, directional):** temperature set in ~1.7% of
  calls, seed ~0.1%, model ~8.5% — "non-determinism by omission."
- **Call-graph recovery:** ~13 empty-graph repos; 5 (hermes, litellm, langflow, marimo,
  phoenix) recover with no-timer pyan (validated, counts in
  `pipeline/artifacts/timeout_rerun_results.csv`, NOT yet merged into the dataset);
  Group B (CVlization, mlflow) unrecoverable without a localizer fix. ~90%+ achievable.
- **Population exclusions** (documented filters in `analyze.py`): atom + Hands-On
  (>=10 unparseable files, quality); sunnypilot (0-invoker Stage-2 false positive).

## 7. Next sprint — TODO (priority order)

1. **Finish the invocation-pattern pass** for the remaining in-scope frameworks:
   mem0, crewai, langgraph, dspy, haystack, smolagents, agno, camel, astrbot, graphrag,
   agentops, semantic_kernel, notte, agents, openai, anthropic. For each: confirm its
   real invocation API + how pilot repos call it; keep real calls, drop
   orchestration/collision; mark `# DONE` in FrameworkDict. (Seth-led, with assistant
   pulling docs + sample call sites.)
2. **Receiver-noise cleanup** on in-scope patterns: drop langchain `.run`/`.call`/
   `.acall` (tool exec), `.predict` (ML collision), `.generate`/`.transform`; agentops
   `.init`/`.record`; graphrag `.search`; scope autogen_core `.create` to model_client.
   Add a **global receiver-blocklist** (`asyncio`, `subprocess`, `os`, `re`, `nanoid`)
   + a `*_tool` receiver filter for `.invoke` in `seed_invokers` (`transitive_invokers.py`
   ~L380, the single match choke point). `call.text` holds the receiver expression.
3. **analyze.py**: report by real framework (apply `keep_frequency.category()` grouping)
   with a three-bucket split: top-20 frameworks / direct SDK calls (openai, anthropic) /
   excluded incidental tail.
4. **Stage-2 false-positive repos**: decide in/out for the 0-direct-invoker collisions
   (DjangoBlog, mongodb/motor) and LLM-adjacent-but-not-apps (gpt4free gateway,
   graphrag-toolkit library, astrbot plugins, django-haystack).
5. **Re-run Stage 5 on the pilot** with the scoped dict + cleaned patterns; regenerate
   the invoker-by-framework chart for a before/after; confirm cleaner data + 90%+.
6. **angr localizer fix** — handle the bare-qname `KeyError` in `_find_offending_file`
   (`transitive_invokers.py`); may also rescue Group B (CVlization, mlflow).
7. **Record the 5 no-timer-recovered empty-graph repos** — backfill counts vs full
   row-level re-run (note: they're mostly framework libraries with huge transitive
   closures; keep them out of the application-level headline — see
   `project_metrics_transitive.md`).
8. **pct_touched metric** — add "non-determinism blast radius" (invokers / total
   functions) as a first-class `analyze.py` output, framework-vs-app split.
9. **FULL RUN** — `batch_call_metadata` with `--slice` over the 1,064-app in-scope
   population, once frameworks + patterns are clean (this produces per-app invocation
   data for the whole population; the pilot only has ~88).

## 8. Files that matter

| file | role |
|------|------|
| `Wrapper/FrameworkDict.py` | `FRAMEWORK_CALLS` (75, record), `IN_SCOPE_FRAMEWORKS`, `SCOPED_FRAMEWORK_CALLS`; `# DONE` markers on reviewed frameworks |
| `Wrapper/transitive_invokers.py` | `seed_invokers` (direct-invoker detection; where the receiver-blocklist goes), `build_call_graph` (resilient pyan; env `PYAN_TIME_BUDGET_SEC`/`PYAN_MAX_EXCLUSIONS`), `_find_offending_file` (angr fix) |
| `Applications/slim_applications.py` | `EXPLICIT_KEEP` / `EXPLICIT_DROP` — trustworthy import-name allowlist; writes `applications_slim.csv` |
| `Applications/keep_frequency.py` | `category()` / `EXTRA_MEMBERS` grouping; writes `keep_frequency.csv` |
| `Applications/plot_coverage.py` | writes `framework_coverage.png` |
| `Applications/analyze.py` | the report + population exclusions (`QUALITY_EXCLUDED`, `NOT_LLM_APP`) |
| `pipeline/batch_call_metadata.py` | Stage 5+6 driver (`--slice`) |
| `pipeline/artifacts/` | all outputs: `applications_slim.csv`, `keep_frequency*.csv`, `llm_*_all.csv`, `call_graph_health.csv`, `invoker_legitimacy_audit.md`, `timeout_rerun_results.csv` |

## 9. Commands

```
# regenerate ranking after any framework/token change:
py -3.14 Applications/slim_applications.py
py -3.14 Applications/keep_frequency.py
py -3.14 Applications/plot_coverage.py

# the report (reads the artifact CSVs):
py -3.14 Applications/analyze.py

# re-run the pilot (Stage 5 + slicing) — full command + flags in RUN_PILOT.md
```

## 10. Gotchas

- **`py -3.14`** always (pyan3 needs 3.14). Two test groups (duplicate module basenames
  collide) — see `project_pipeline_state.md`.
- **Windows**: pass `--joern C:/Users/Seth/joern_install/joern-cli/joern.bat` (bare
  `joern` = Unix script → WinError 193).
- **Pilot invocation data is STALE** — don't trust `llm_*_all.csv` numbers until Stage 5
  is re-run with the cleaned patterns/scope.
- Cuts are **filters, never deletions** — raw CSVs and the full FrameworkDict are
  preserved; criteria live in code comments. Keep it that way.
