# Exclusions & Exemptions Ledger

**Single source of truth** for every item removed, exempted, or narrowed in the study,
with the reason and the code location that enforces it. Kept for research
reproducibility: *nothing is deleted from the raw data* — exclusions are documented
filters. When you exclude/exempt/cut anything, **add a row here first**, then implement
the filter in code and reference this file.

Categories:
- **Population exclusion** — a repo dropped from the analyzed denominator (`analyze.py`).
- **Framework exemption** — a framework kept in the study conceptually but *not measured*
  (out of `SCOPED_FRAMEWORK_CALLS`), because it can't be reliably measured.
- **Framework cut (phantom)** — a "framework" removed entirely; its matches were token
  collisions, not real usage.
- **Pattern cut** — a specific method/name removed from a framework's invocation patterns
  (collision or non-invocation), framework otherwise kept.
- **Pending** — an exclusion decided but not yet enforced in code.

Verification rule (recurring lesson): **confirm a match by reading the actual matched
code** — Stage-2 uses loose token-based GitHub code search, so a recorded match ≠ real use.

---

## 1. Population exclusions (repo-level) — enforced in `Applications/analyze.py`

| Repo | Reason | Category | Enforced in |
|------|--------|----------|-------------|
| `rush86999/atom` | 31 unparseable files (≥10 threshold) — code-quality outlier | Population | `analyze.py` `QUALITY_EXCLUDED` |
| `Sumanth077/Hands-On-AI-Engineering` | 56 unparseable files | Population | `analyze.py` `QUALITY_EXCLUDED` |
| `sunnypilot/sunnypilot` | 0 invokers; `agno` name-collision (driver-assistance app, not an LLM app) | Population | `analyze.py` `NOT_LLM_APP` |

## 2. Framework exemptions (in study, NOT measured) — enforced in `Wrapper/FrameworkDict.py` (`_IN_SCOPE_EXPLICIT`)

| Framework | Reason | Enforced in |
|-----------|--------|-------------|
| `omnigent` | CLI-based agent → invokes LLM in a spawned subprocess, invisible to static method-matching (0 invocations = out-of-process blind spot, **not** a phantom). Its `omnigent` token *also* collides with the diffusion class `OmniGe(nt)ransformer`. Unmeasurable either way → exempt. See memory `project_out_of_process_invocation`. | removed from `_IN_SCOPE_EXPLICIT` (2026-08-05) |
| `agentops` | Observability layer, not an invoker — instruments *other* frameworks' calls. Its methods (`.init`/`.record`/`.start_session`/`.end_session`) are telemetry lifecycle, 0 real model calls; an agentops app's LLM calls are already counted via its underlying framework. | removed from `_IN_SCOPE_EXPLICIT` (2026-08-05); patterns kept for provenance |

## 3. Framework cuts (phantoms, removed entirely) — prior sprint

| Framework | Reason | Enforced in |
|-----------|--------|-------------|
| `cheshire-cat` | Junk import tokens (`cat`/`agui`) — matched AG-UI-protocol users + `cat` collisions, never cheshire-cat. | `slim_applications.EXPLICIT_KEEP` / `keep_frequency.EXTRA_MEMBERS` / FrameworkDict removal |
| `connectonion` | Token `subagents` matched apps' own `X.subagents` submodules (e.g. `deepagents.middleware.subagents`), never the package. | same as above |

## 4. Pattern cuts (per framework) — enforced in `Wrapper/FrameworkDict.py` (this sprint, see each entry's `# DONE` comment)

| Framework | Cut | Reason |
|-----------|-----|--------|
| `langgraph` | `.run .arun .call .acall .predict* .generate* .transform* .predict_messages*` (12 legacy methods) | Not in LangGraph's Runnable API (langchain-legacy); 0 pilot hits, pure collision risk. Kept the 8 Runnable-core methods. |
| `langchain` family (langchain, _openai, _anthropic, _community, _core, _experimental) | `.predict`, `.apredict` (2026-08-06) | Deprecated `LLM.predict` collides with the **sklearn/MLflow model API**: 62 pilot hits, ~57 were ML predictions (`pyfunc_loaded_model`/`sklearn_knn_model`/`rf.predict` in mlflow, torch cross-encoder in ragas), only ~4 real `llm.predict` (deprecated; those repos also use `.invoke`, kept). Kept `.predict_messages`/`.apredict_messages` (langchain-specific, no collision). |
| `mem0` | `.get_all`, `.update` | Plain vector/DB lookups & writes, no LLM call. Kept `.add`/`.search` (LLM extraction/query). |
| `dspy` | bare class-name *matching* (`Predict`, `ChainOfThought`, …) | Matching the constructor flagged the construction site, not the call. Replaced with structural `__call__` binding in `seed_invokers`; class names moved to `DSPY_MODULE_CLASSES` as a construction *signal*. |
| `camel` | `.chat`, `.achat` | Not CAMEL's API; 14 hits were OpenAI/Cohere/Mistral client collisions (`client.chat.completions.create`). Kept `.step`/`.astep`. |
| `camel` | bare `ChatAgent`, `RolePlaying` | Construction tokens, already stripped by the bare-name loop. |
| `astrbot` | `.get_using_provider` | A provider *getter*, not a model call (its fn is caught by `.text_chat`). |
| `astrbot` | `.chat` | Not AstrBot's API; collides with `client.chat.*`. |
| `astrbot` | `.request_llm` | Unconfirmed in current docs, 0 hits. (Added confirmed `.llm_generate`, `.tool_loop_agent`.) |
| `graphrag` | `.search`, `.asearch` | 100% collision: all 26 hits were `re.search` / `file_pattern.search` / vector-DB `client.search`. Kept module-level `global/local/drift/basic_search`. |
| `semantic_kernel` | `.invoke_async`, `.invoke_prompt_async` | C#-style pre-1.0 naming, not current Python SK. (Added real `.invoke_prompt_stream`.) |
| `semantic_kernel` | *not added*: `invoke_function_call`, `add_embedding_to_object` | Tool-call execution / embedding helper — not text-generative model calls. |
| `notte` | bare `Notte` | Construction token, already stripped by the bare-name loop. |

## 5. SDK method cuts (generative-text scoping) — enforced in `Wrapper/FrameworkDict.py`

Rationale: the study measures **non-determinism of generative text calls**, where the
knobs (temperature/seed/model) apply. Non-generative / non-text-modality calls are cut.

| SDK | Cut | Reason |
|-----|-----|--------|
| `openai` | `embeddings.create` | Embedding, near-deterministic, no gen knobs. |
| `openai` | `images.generate`, `audio.transcriptions.create`, `audio.translations.create`, `audio.speech.create` | Non-text modalities. |
| `openai` | `moderations.create` | Text *safety classifier*, not generative, no knobs. |
| `anthropic` | `messages.count_tokens` | Tokenizer utility, not a model generation. |

## 6. Pending (decided, NOT yet enforced) — Phase 2 false-positive cleanup

Measured on the cleaned pilot (88 repos) via the holistic receiver audit
(`Applications/audit_fp.py`): **~13% of all LLM calls are false positives**
(~1,310 / 9,968) across five receiver-driven tiers. To be removed by four global rules at
the single `seed_invokers` match choke point (see memory `project_post_rerun_plan`). When
implemented, move these to a "Framework-agnostic pattern rules" section above with the
enforcing commit.

| Rule | Removes | Reason |
|------|---------|--------|
| Receiver-root blocklist: `asyncio`, `subprocess`, `os`, `re`, bare `mock`/`Mock`/`MagicMock`/`AsyncMock`, + util libs `nanoid`, `uuid` | ~541 calls (`asyncio.run` 457 dominant; `nanoid.generate` ~3) | Stdlib/util collisions — no framework's method is `asyncio.run` or `nanoid.generate`. **Keep** `self.<attr>.method`, `mock_client.chat…create`, domain receivers (`agent`/`client`/`chain`). |
| Terminal-segment rule: matched method must be the final call segment. Implement as `call.text.endswith("."+method)` (NOT "method appears with a trailing dot" — that wrongly drops a legit `self.run.run`). | ~220 calls | `agent.arun.assert_called_once` / `.reset_mock` — method *accessed* on a Mock, not called. |
| Tool/executor receiver filter: receiver is a tool/sandbox/driver — root `tool`/`tools`/`toolset`/`*_tool`, `sandbox`, `driver`, `*_executor` — on `.invoke`/`.run`/`.stream`/`.call`/`.acall`. | ~473 calls | LangChain **tool execution**, not a model call (`read_file_tool.invoke`, `sandbox.run`, `driver.run`, `tool.call`/`.acall`, `function_tool.acall` — ~23 on `.call`/`.acall`). **Carve-out: KEEP `model_with_tools`/`llm_with_tools`/`*_with_tools`** — a model bound with tools IS a real LLM call. |
| Non-model Runnable receiver filter: root `*template*`, `retriever`, `parser`/`*_parser`, `passthrough`, `splitter`, `embedder`, bare `prompt`/`*_prompt` — on `.invoke`/`.ainvoke`. | ~57 calls | The LangChain Runnable interface is implemented by every LCEL component; `PromptTemplate.invoke(vars)` *formats* (arg is a var dict), `OutputParser.invoke(AIMessage)` *parses the model's existing output*, `retriever.invoke(query)` *vector-searches* — none call a model. **Recall-safe:** the real model call is always a *separate* `model`/`chain` call site, which is kept. **Carve-out: KEEP `chain`/`model`/`llm`/`*_with_tools`.** |
| Non-model infrastructure receiver filter: `.create` on `zep`/`cache_client`/`*_cache`/`Resource`; `.batch`/`.abatch` on `store`/`collection`; `.run_sync` on `conn`. | ~17 calls | Storage / cache / DB-connection objects on generic verbs — `zep.create`/`cache_client.create` write a memory/cache record, `store.batch` is a langgraph state-store op, `conn.run_sync(fn)` is a SQLAlchemy async-connection call. Not model calls. **Carve-out: KEEP model clients** `client`/`model_client`/`openai_client`/`openai_model_client`. |

**Tail candidates (<0.1%) — still pending Seth's re-examination:**
- `<expr>` non-identifier receivers (~14, mostly langchain_core `.ainvoke`) — subscripts /
  call-results (`get_llm().ainvoke`), unclassifiable by receiver root; needs case reading.

**Convergence note:** a broad sweep of ~18 previously-unexamined patterns (kickoff,
initiate_chat, Runner.run, messages.create, run_sync/run_stream, text_chat, step, graphrag
searches…) surfaced only the one 6-call `conn` collision above — everything else hit real
framework receivers. The FP discovery is effectively exhausted at pilot scale; the five
tiers + langchain `.predict` cut cover the material FPs. Re-run `audit_fp.py` on the full
run to re-confirm at scale.

Discovery tooling: `Applications/audit_fp.py` (holistic receiver audit → `pattern_audit.csv`
/ `receiver_pivot.csv`). Re-run after the FULL population run to catch population-specific
collisions (e.g. `session.add` for mem0, `optimizer.step` for camel — clean in the pilot).
Tier sizes shrink each discovery pass (538 → ~450 → ~57), i.e. the big ones are found.

---

*Last updated: 2026-08-06. Add new exclusions as rows above, dated, with the enforcing code location.*
