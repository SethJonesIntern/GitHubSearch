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
| `openai` | **v0 API**: `ChatCompletion.create`/`.acreate`, `Completion.create`/`.acreate` (2026-08-13) | The pre-1.0 module-level API, deprecated Nov 2023 and absent from the v1 client. The study measures **current** state of practice: a call surface three years out of date is out of scope. Unlike the rows above this is *not* a collision — these are real model calls, deliberately outside the measured surface. **Consequence:** a repo still on v0 reports 0 openai calls (confirmed: `liangliangyy/DjangoBlog`, `openai.ChatCompletion.create`), so its zero is a scoping result, not a detection miss. Unmeasured: how many repos this affects — a grep of the clones would size it if the paper needs a footnote. |

## 6. Framework-agnostic false-positive rules — Phase 2 cleanup

**Status (2026-08-06): ALL FIVE TIERS IMPLEMENTED** in `Wrapper/false_positives.py`
(`classify_fp`), wired into `seed_invokers` (skips FP matches), `call_metadata.py` (tags each
call with an `fp_tier` column), and `analyze.py` (`drop_fp` filters at load by default; raw
rows kept for audit). Validated on the pilot: **1,212 flagged (12.2%)**, 0 legit-call
casualties (56/56 unit tests — carve-outs `model_with_tools`/`prompt_model`/`rag_chain`/
`model_client.create`/`mock_client.chat…create` all pass; per-tier: t1 517, t2 227, t3 392,
t4 64, t5 12). **The effect materializes on the next Stage-5 re-run** (current CSVs predate
the column).

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

**`<expr>` non-identifier receivers — RESOLVED (2026-08-06), 0 new FPs.** The ~14 broke
into: 13 × `(await *_template.ainvoke(...)).x` chains (already double-covered — `*_template`
= tier 4, and `.ainvoke` is non-terminal in the outer chained call = tier 2), and 1 ×
`(agent or self.default_rationale_agent()).step(...)` — a REAL camel invocation (receiver
is an agent via an `or`-expression), kept. **Implementation guardrail:** when a receiver
can't be resolved to a root (`<expr>`), the receiver-based rules (tiers 3–5) must
DEFAULT-KEEP — else the real `(agent or …).step` case is mishandled; the template chains
are caught by the terminal-segment rule (tier 2) regardless.

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

## 7. Framework / eval self-repositories (2026-08-11) — enforced in `Applications/slim_applications.py`

A framework's or eval tool's OWN source repository is **not an application built on it** —
its self-imports, tests, and examples inflate every metric (agno's own test suite alone =
>12k "ND tests"; opik/phoenix eval "usage" was 100% their own repos). The study measures
**state of practice in APPLICATIONS**, so the canonical repo of every framework/eval tool
**we analyze** is dropped from the population.

- **Rule:** exclude only frameworks/eval-tools in our dicts. `langflow`/`litellm`/`marimo`
  etc. are NOT discovered frameworks (not in FRAMEWORK_CALLS/EVAL_CALLS), so they STAY as apps.
- **Enforced in** `slim_applications.py` → `framework_self_repos()` / `SELF_REPOS`, filtered
  in `slim_csv` by `full_name`. Base = `frameworks.csv` `full_name` column (59 repos) +
  a supplement for the top-20/eval added after that snapshot (agno, mem0, dspy, haystack,
  smolagents, graphrag, semantic_kernel, deepeval, ragas [×2 orgs], giskard, opik, phoenix).
- **Scale:** 72-repo set; **9** present in the 1,055-app population, **4** in the pilot
  (`agno-agi/agno`, `Arize-ai/phoenix`, `comet-ml/opik`, `vibrantlabsai/ragas`). Removed
  from the pilot CSVs (backup in `artifacts/_framework_repos_removed/`); auto-excluded for
  the full run via the regenerated `applications_slim.csv` (1064 → 1055).
- **Impact (app-only vs framework-inflated):** LLM calls 8,854→4,701; eval calls 828→30;
  determinism knobs rose (temperature 2.9%→5.3%, model 13%→22.5%, seed still ~0.1%);
  agno 3,963→254 calls. The "non-determinism by omission" finding holds and is cleaner.

**RESOLVED (2026-08-11) → see §9.** The open item here was the ~237 apps matched ONLY to
non-analyzed framework names. It is now settled as a three-way partition (junk / known
uncovered / analyzed) enforced in `slim_applications.py`. **Correction:** this section
previously claimed `clai`→pydantic_ai was a rollup gap — it is **not**. `clai` is a junk
collision token and its apps leave the population entirely (§9). Only `crewai_tools` and
the `agent_framework_*` names are genuine rollups.

## 8. Eval-framework dict (EVAL_CALLS) — reviewed & completed (2026-08-11)

All 5 eval frameworks reviewed against pilot receivers + docs, marked `# DONE` in
`pipeline/eval_calls.py`: **deepeval** (unchanged), **ragas** (+single/multi_turn_score/
ascore — version drift, pilot-confirmed), **giskard** (rewritten to v2/v3 run surfaces:
scan/vulnerability_scan/quality_scan/evaluate/.run; docs-only, unsampled), **opik**
(+evaluate_prompt/evaluate_experiment/.ascore), **phoenix** (+run_experiment/
async_run_experiment/evaluate_dataframe/async_evaluate_dataframe/llm_generate). Caveat noted
in-code: pilot eval counts included the eval frameworks' own repos (now excluded, §7).

---

## 9. Analysis scope — real-AI denominator vs. analyzed run set (2026-08-11) — enforced in `Applications/slim_applications.py`

A trustworthy KEEP name means the token identifies *something*; it does **not** mean the
repo is an AI application, nor one we measure. The 1,055 slimmed candidates partition
three ways (full derivation + the reproduce snippet: `COVERAGE_ANALYSIS.md`):

| Bucket | Count | In denominator? | Analyzed? | What it is |
|--------|------:|:---:|:---:|------------|
| **Analyzed** | **827** | yes | yes | imports an in-scope (top-20 / langchain / autogen / SDK) framework or an eval tool |
| **Known uncovered** | **91** | yes | no | real AI apps on out-of-scope long-tail frameworks (`metagpt`, `lagent`, `honcho`, `beeai_framework`, `agent_protocol`, `headroom`, `patchwork`, `adalflow`, `agency_swarm`, `superagi`, `dynamiq`…) plus `agentops` (exempt observability). Below the top-20 cut, so 0 invocations is the *intended* answer — they are not run. |
| **Excluded** | **137** | **no** | no | the matched name identifies no framework: collision tokens, non-LLM langchain utilities, the `omnigent` phantom |

**Headline:** analyzed frameworks cover **827 / 918 = 90.1%** of real AI applications.

**Definitions (both derived from the dicts, not hand-listed):**
- *Real AI app* = matched name (after aliasing) is a key in `FrameworkDict.FRAMEWORK_CALLS`
  or `eval_calls.EVAL_CALLS` → `real_ai_app=1`.
- *Analyzed* = matched name is in `FrameworkDict.IN_SCOPE_FRAMEWORKS` or `EVAL_CALLS`
  → `analyzed=1`.

| Population exclusion | Count | Reason |
|---|---:|---|
| `clai`-only apps | 42 | **Junk collision token, NOT pydantic-ai's CLI.** As a GitHub search token it matched `binance-connector-python`, `py-stellar-base`, `huaweicloud-sdk`, `python-cwt` etc. — not AI apps. Supersedes the earlier `clai`→pydantic_ai rollup assumption in §7 and in `keep_frequency.EXTRA_MEMBERS`. |
| `langchain_text_splitters` / `_chroma` / `_qdrant` / `_tests` / `_exa` / `_classic`-only apps | ~48 | langchain **non-LLM utility** packages (text splitting, vector stores, test helpers). Importing one is not evidence of an LLM call, and they are not `FRAMEWORK_CALLS` keys. |
| `omnigent`-only apps | 24 | The exempt out-of-process phantom (§2); its token also collides with the diffusion class `OmniGe(nt)ransformer` — orphans included `huggingface/diffusers` and `bytedance/Video-As-Prompt`, plainly not LLM apps. |

**Aliases (rollups, NOT exclusions)** — companion/submodule packages of an analyzed
framework, rolled up so their apps aren't lost: `agent_framework_foundry`,
`agent_framework_openai`, `agent_framework_foundry_hosting` → `agent_framework`;
`crewai_tools` → `crewai`. (`slim_applications.ALIASES`.)

**Enforced in** `slim_applications.slim_csv`: nothing is deleted — all 1,055 rows stay in
`applications_slim.csv` carrying `real_ai_app` / `analyzed` flag columns, so every
denominator is recoverable from one file. Pre-change backup: `artifacts/_pre_scope_filter/`.

**These flags are NOT a run-set filter — that was tried and rejected (2026-08-13).** An
earlier revision emitted `applications_analyzed.csv` (827 rows) as the batch input so we
"never clone a repo we cannot measure." That reasoning is wrong: `analyzed` is computed
from the *search token* that found the repo, and a repo matched only to an out-of-scope
token still makes in-scope raw-SDK calls — `JetAstra/SDAR` (matched on `lagent`) contains
18 `openai` + 4 `anthropic` calls, and `lupantech/AgentFlow` (`agentops`) contains 161.
Filtering pre-run would have silently discarded them. **The full 1,055 were analyzed**;
`applications_analyzed.csv` was deleted. Decide scope after analysis, not before.

**These flags are a hypothesis, not ground truth.** They are derived from
`matched_frameworks` (search metadata). On the completed run, **83% of analyzed repos
import a framework their matched token never mentions.** Coverage should be recomputed
from frameworks actually *detected in code* before publication.

**Stage-5 detector rollup — DONE (2026-08-11).** The aliases above decide *which repos we
run*; the matcher needs them too. `transitive_invokers.index_repo` previously intersected a
file's raw import names with `FRAMEWORK_CALLS` keys, so `from agent_framework_foundry
import ...` activated no patterns and its calls were never tested (a false 0 on a repo we
did clone and parse; ~10 of the 827 exposed). Now goes through
`FrameworkDict.resolve_framework_imports`, which mirrors `slim_applications.ALIASES` —
**keep the two tables in sync.** An alias resolves only when its parent is in the *active*
pattern dict, so EVAL_CALLS passes are unaffected. Regression tests:
`Wrapper/tests/test_import_aliases.py` (verified failing before the change).

## 10. Import-name collisions found by the audit sheet (2026-08-20) — enforced in `pipeline/audit_apps.py` (`CUT`)

`pipeline/artifacts/application_audit.csv` (built by `pipeline/audit_apps.py`, one row per
population repo, plus `audit_framework_check.py` / `audit_zero_invokers.py` for the
judgement columns) exists to explain every repo that produced no invokers. Its first
finding: the token **`haystack` names three unrelated projects**, and only one is ours.

| Cut | Count | Reason |
|---|---:|---|
| django-haystack apps | 16 | Django **search indexing**, not deepset Haystack. Confirmed by reading imports: `from haystack import indexes`, `haystack.forms.SearchForm`, `haystack.views.SearchView` — never `haystack.components`. `tendenci`, `gcd-django`, `daisy`, `DjangoBlog`, `drf-haystack`, `linkedevents`, `cosinnus-core`, `aries-vcr`, `sith`, `nablaweb`, `ajapaik-web`, `macports-webapp`, `Telemeta`, `widelands-website`, `reviewboard`, `django-page-cms` |
| Project Haystack apps | 4 | The **building-automation / IoT** data standard. `pyhaystack`, `haystackfs`, `py-brickschema`, `phable` |

All 20 matched the token `haystack` **and nothing else**, produced 0 invokers and 0 LLM
calls. Rows stay in the sheet carrying `in_scope=0` + the reason in `notes`; nothing is
deleted.

**Consequence for the ranking:** of 49 repos matched on `haystack`, only **24** call
deepset Haystack; 5 are LLM apps built on something else; 20 are these collisions. So
`keep_frequency.csv`'s "haystack: 51 apps (4.8%)" — and therefore the top-20 table and
`framework_coverage.png` — is inflated by roughly 40% for this framework. This is the
token-vs-code problem §9 flags; the audit sheet now measures it per repo.

**Not yet checked:** whether `dspy` and `honcho` (next largest contributors to the
`imports_fw_no_call_site` bucket, 5 repos each) have the same collision.

---

## 11. The langchain-named repos: integration packages, not applications (2026-08-25) — enforced via `in_scope=0` in `application_audit.csv`

The population contains 26 repos with `langchain` in the name. Reviewed one by one
(tree layout + README, per the evidence-before-cutting rule): **22 are framework-side
code, 4 are genuine applications.** The langchain ecosystem's own naming convention —
`langchain-<vendor>` — marks an *integration package* (a pip-installable
`libs/*` monorepo other projects import), and the convention held with exactly the
exceptions you would predict: an app named after its stack, a docs tool, a course, and
a first-party app.

The `imported_by` signal (audit_framework_check / imported_by.csv) under-detected this
group: the `langchain-ai/*` monorepos keep their `pyproject.toml` in `libs/*/`, below
the root-level metadata scan, so most showed `imported_by=0` despite being the clearest
libraries in the corpus. Name convention + layout was the decisive evidence here.

| Cut (22) | Category | Evidence |
|---|---|---|
| `langchain-ai/langchain-{postgres, litellm, google, cohere, aws, ibm, datastax, upstage, meta, mongodb, snowflake}`, `langchain-ai/langgraph-swarm-py` | first-party integration packages | `libs/*/pyproject.toml` monorepos; postgres/litellm are imported by 17/15 other corpus repos |
| `oracle/langchain-oracle`, `googleapis/langchain-google-alloydb-pg-python`, `googleapis/langchain-google-cloud-sql-pg-python`, `oceanbase/langchain-oceanbase`, `tavily-ai/langchain-tavily`, `derf974/copilot-langchain` | vendor-owned integration packages | same layout; READMEs: "LangChain integration for …" |
| `UiPath/uipath-langchain-python` | SDK | README first line: "A Python SDK …" |
| `langchain-ai/deepagents` | first-party agent harness (library) | pip-installable; held 7,801 ND tests — largest single item in this cut |
| `ksachdeva/langchain-graphrag` | library | declares `langchain_graphrag`, readthedocs docs site |
| `xt765/LangChain-Chinese-Comment` | **framework source copy** | langchain's own source annotated with Chinese comments; its 1,989 "calls" are framework internals. Same logic as the §7 self-repo rule — a copy of a framework's source is framework code |

| Kept (4) | Why |
|---|---|
| `chatchat-space/Langchain-Chatchat` | README: deployable RAG/agent **application project** (ships docker/); built ON langchain, nobody imports it |
| `lucebert/langchain-doc-graph` | RAG backend application (subject matter happens to be langchain docs) |
| `microsoft/langchain-for-beginners` | tutorial course. Tutorials are not a cut class (precedent: Hands-On-AI-Engineering was cut on *quality*, not for being a tutorial); excluding tutorials would need a systematic criterion applied corpus-wide |
| `langchain-ai/open-swe` | a coding-agent **application** — first-party to the LangChain org, but an app, not a library. Decided 2026-08-25 |

**Impact:** 3,686 LLM calls, 11,287 ND tests, 2,644 direct invokers leave the counted
set. `in_scope=0` set by hand in `application_audit.csv` with a per-repo `notes` reason;
the 4 keeps carry a dated review note so later audits do not re-flag them. Takes effect
via `pipeline/cuts.py` on the next `analyze.py` run.

**Generalization not yet applied:** the same reasoning likely extends to other
ecosystems' integration packages in the population (e.g. `llama-index-*`-style naming,
provider SDKs). See `pipeline/artifacts/framework_triage.csv` /
`imported_by.csv` for the ranked queue; the ~36 remaining strong/likely candidates
(weave, NeMo Guardrails, cognee, marimo, pydantic-ai-harness, …) are still undecided.

---

## 12. Framework/library and platform cuts from the triage review (2026-08-25) — enforced via `in_scope=0` in `application_audit.csv`

Second pass of the framework-vs-application comb (queue: `framework_triage.csv` +
`imported_by.csv`, top 30 still-counted candidates by ND-test weight, each read
tree+README). **23 cut, 7 kept.** Two cut categories:

### 12a. Self-described frameworks/libraries (18) — their own READMEs say so

| Repo | ND tests | Own words / evidence |
|---|---:|---|
| `DeytaHQ/khora` | 8,108 | "A Python **library** for creating knowledge repositories" |
| `nhadaututtheky/neural-memory` | 5,025 | memory component for *your* agent |
| `TheodoreGalanos/aec-bench` | 4,577 | "A Python **platform** for … evaluating benchmarks", published package |
| `jwwelbor/AgentMap` | 4,408 | "declarative orchestration **framework**" |
| `massgen/MassGen` | 3,961 | "multi-agent **framework**" |
| `kdcube/kdcube-ai-app` | 3,879 | "self-hosted production **runtime** for AI applications" |
| `wandb/weave` | 3,758 | observability toolkit — the §2 agentops precedent; imported by **14** corpus repos |
| `NVIDIA-NeMo/Guardrails` | 3,596 | guardrails **library**; imported by 7 |
| `fabceolin/the_edge_agent` | 3,596 | YAML agent runtime |
| `semantica-agi/semantica` | 3,578 | "Graph-Native **Infrastructure**" |
| `DemonDamon/AgenticX` | 3,472 | "agent technology **stack** … Python Agent Runtime" |
| `vladkesler/initrunner` | 3,358 | agents-from-YAML engine |
| `msu-denver/bili-core` | 3,303 | "An Open-Source LLM **Framework**" (title) |
| `nfraxlab/ai-infra` | 3,252 | "One unified **SDK** for LLMs, agents, RAG" |
| `topoteretes/cognee` | 3,193 | "AI memory **platform**"; imported by 8 |
| `pydantic/pydantic-ai-harness` | 2,208 | first-party pydantic-ai add-on (the §11 deepagents precedent); imported by 3 |
| `areal-project/AReaL` | 2,076 | "RL **infrastructure**" |
| `cuga-project/cuga-agent` | 1,954 | "Agent **Harness** for the Enterprise" |

### 12b. Platforms/builders (5) — ruled 2026-08-25

**The test, decided this date: is the repo's product an LLM application itself (keep),
or a thing whose users build/run their own LLM applications (cut)?** Platforms fail it.

| Repo | ND tests | Evidence |
|---|---:|---|
| `databrickslabs/kasal` | 14,334 | drag-and-drop AI-workflow designer — largest single ND holder in the corpus at cut time |
| `langflow-ai/langflow` | 11,236 | "platform for building and deploying AI-powered agents" |
| `marimo-team/marimo` | 8,470 | notebook IDE; hardest case — its LLM code is its own assistant feature, but it is imported as a library by 7 corpus repos |
| `dimagi/open-chat-studio` | 4,222 | "platform for building, deploying, and evaluating AI-powered chat applications" |
| `griptape-ai/griptape-nodes-engine` | 3,362 | visual workflow builder engine (framework vendor) |

**This supersedes the informal §7-era remark that `langflow`/`marimo` "STAY as apps"** —
that was scoping mechanics (they are not `FRAMEWORK_CALLS` keys, so the self-repo rule
could not reach them), not a considered judgment. The platform test above is the
considered judgment. **Consistency check still open:** `astrbot` (a plugin platform,
kept per the earlier platform-skew note) arguably fails the same test; it is a measured
framework in our dicts, so its situation differs — flagged, not yet relitigated.

### Kept as applications (7)

`theexperiencecompany/gaia` (personal assistant) · `gptme/gptme` (terminal assistant) ·
`NorthlandPositronics/Cogtrix` (local agent app) · `bytedance/deer-flow` (deep-research
agent, run from source, no package) · `ginlix-ai/LangAlpha` (financial analysis) ·
`mpfaffenberger/code_puppy` (code-agent CLI) · `open-jarvis/OpenJarvis` (assistant;
thinnest evidence — README is doc links only). All carry dated KEEP notes in the sheet.

**Impact:** 1,809 LLM calls, **108,926 ND tests** (40% of the pre-cut total), 1,540
direct invokers leave the counted set. `in_scope` now 887 counted / 82 cut / 86
uncovered. ~53 lower-weight triage candidates (mostly <2,000 ND each) remain unreviewed.

---

## 13. Denominator-side collision cuts (2026-08-26) — enforced via `in_scope=0` in `application_audit.csv`

The §11/§12 combs ranked candidates by **ND-test weight**, so by construction they could
only surface repos that *contribute* something. Repos that are pure denominator — 0 calls,
0 invokers, 0 ND tests — were invisible to that queue. This is the first pass from the
other side, and it is the reason "81% of analyzed apps contain an LLM call site" was low:
the shortfall was mostly non-applications sitting in the denominator, not undetected LLM use.

**Criterion (in code — `pipeline/waterfall.py`, `queues()` bucket A):** analyzed, **0 LLM
call sites**, `frameworks_imported` **empty** (the clone imports no known LLM library), and
`real_ai_app=0` (the search token is not a key in `FRAMEWORK_CALLS`/`EVAL_CALLS`) — **and**
`http_llm_files == 0` **and** `cli_llm_files == 0`.

Those last two conditions are the guardrail: an import scan cannot see a provider called
over `requests`/`httpx` or a shelled-out CLI, so a zero import count alone is not evidence
of absence. **9 of the 30 candidates carry HTTP/CLI evidence and were deliberately NOT
cut** (`HKUDS/ClawTeam`, `psi-oss/get-physics-done`, `Lyellr88/MARM-Systems`,
`Intelligent-Internet/CommonGround`, `Stage-11-Agentics/lattice`, `web3spreads/quant-flow`,
`noetl/noetl`, `ttlequals0/PixelProbe`, `canvas-medical/canvas-plugins`) — they are held for
source review and remain counted.

| Cut | Count | Reason |
|---|---:|---|
| `omnigent`-token repos | 18 | The token matches the diffusion class `OmniGe`**`nt`**`ransformer`. All 18 are image/video/audio **generation** research code — out of scope for a study of LLM text generation. Includes `huggingface/diffusers` (33,955★, 2,053 `.py`, **0** LLM calls / 0 invokers / 0 ND tests / `llm_calls_raw=0`) and `bytedance/Video-As-Prompt` — **both named in §9 as "plainly not LLM apps" but never enforced.** |
| `sia`-token repos | 2 | `eavanvalkenburg/pysiaalarm` (SIA DC-09 alarm-panel client), `gip-inclusion/les-emplois` (French employment platform). |
| `pyautogen`-token repo | 1 | `memory-graph/memory-graph` — an MCP memory server with **0 Python files**. |

**Impact: 0 LLM calls, 0 invokers, 0 ND tests leave the counted set** — that is the point,
they never contributed any. The denominator moves 838 → 817 and LLM-call prevalence
80.8% → 82.9%. No numerator anywhere changes.

**The §9 enforcement gap this exposes.** §9 assigns 137 repos to a bucket whose column
reads *"in denominator? **no**"*, and `RUNBOOK.md:28` says scope is decided by the
`real_ai_app`/`analyzed` flags — but **no code reads those columns.** `analyze.py` builds
its exclusions from `QUALITY_EXCLUDED` + `NOT_LLM_APP` + `in_scope`, so every §9 "Excluded"
repo has been counted all along. Measured 2026-08-26: **84** `real_ai_app=0` repos sat in
the 838-repo analyzed set.

**Do not "fix" this by honouring the flag.** Of those 84, **54 demonstrably ARE LLM
applications** — `NVIDIA-AI-Blueprints/aiq` (198 calls, 147 direct ND tests),
`hsliuping/TradingAgents-CN` (82), `project-ryoma/ryoma` (48) — flagged 0 only because the
*search token* that found them was junk. That is exactly the failure §9 itself warns about
("83% of analyzed repos import a framework their matched token never mentions") and why
`applications_analyzed.csv` was deleted. **`real_ai_app` is superseded by
`frameworks_imported` for all scope decisions** (CLAUDE.md: *"never decide scope from
`matched_frameworks`"*). It is retained only as provenance. The correct filter is the
code-derived criterion above, applied repo by repo.

---

## 14. Queue A — the `clai` collision survivors (2026-08-27) — enforced via `in_scope=0`

The nine repos §13 held back pending evidence. Each was read at file level (the flagged
lines, not the README). **All nine cut.**

**Why:** eight matched only the token `clai`, one only `langchain_exa`, and **none of the
49 `clai`-matched repos in the corpus imports `clai`** (verified by grep over every
clone). `clai` *is* pydantic-ai's CLI package — the premise of the
`keep_frequency.EXTRA_MEMBERS` mapping was correct — but Stage 2 is code search, so
`"import clai"` matched the substring in **claim / claims / claimed / disclaimer**:
`claim_line_item_id` (medical billing), `claimtoken` (auth), `claimed_index` (job queue).
The matched token names nothing these repos use.

| Repo | What the flagged files contain |
|---|---|
| `noetl/noetl` | `ollama_bridge/server.py` posts to `/api/chat` on `localhost:11434` — a real model call over raw HTTP |
| `HKUDS/ClawTeam` | spawns Claude/Codex CLIs; `openrouter.ai` / `api.deepseek.com` base URLs |
| `web3spreads/quant-flow` | DeepSeek OpenAI-compatible client in `src/llm.py` |
| `psi-oss/get-physics-done` | Codex CLI adapter |
| `Stage-11-Agentics/lattice` | spawns `claude` with kill-on-timeout handling |
| `canvas-medical/canvas-plugins` | plugin **SDK** (`canvas_sdk`), not an application (§12a logic) |
| `ttlequals0/PixelProbe` | media-integrity scanner; HTTP flag was the regex matching `/api/generate-pdf-report` |
| `Intelligent-Internet/CommonGround` | HTTP evidence is a placeholder URL in tests (`example.openai.azure.com`) |
| `Lyellr88/MARM-Systems` | HTTP evidence is a string literal in an assertion + a smoke-test script |

The first five reach a model through `httpx`/`requests` or a spawned CLI; none imports a
tracked framework, `openai`/`anthropic`, or `transformers`. Recorded here so the cut is
not misread as "no LLM present" — the cut criterion is the collision, not the absence.

**Measured alongside, for the record:** of the 42 repos matched by `clai` and nothing
else, 19 import a tracked framework or SDK, and the 5 above use a model by another route.
Stage 2's filters (≥10 stars, ≥2 contributors, ≥2 commits/month, ≥1 test file, pushed
since 2025-04-14) restrict the corpus to actively maintained, tested Python projects.

Impact: 0 LLM calls and 0 ND tests — a **denominator-only** cut.

---

*Last updated: 2026-08-27. Add new exclusions as rows above, dated, with the enforcing code location.*
