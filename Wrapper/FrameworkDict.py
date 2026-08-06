"""Per-framework call patterns used as seeds by transitive_invokers.py."""

# Maps each framework's top-level import name to the call patterns specific to
# that framework.  When a file imports a framework, the scanner searches only
# for that framework's patterns — reducing noise from generic names like .run.
FRAMEWORK_CALLS: dict[str, list[str]] = {
    # ── Direct LLM SDKs ────────────────────────────────────────────────────────
    # Generative-text calls only — the surface where determinism knobs (temperature/
    # seed/model) apply. Cut the non-text modalities (images.generate, audio.*),
    # embeddings.create, and moderations.create (a text safety-classifier, not
    # generative): different modality / near-deterministic / no determinism knobs, out
    # of scope for the non-determinism analysis.
    "openai": [
        "chat.completions.create", "completions.create", "responses.create",
        "chat.completions.parse", "responses.parse", "responses.stream",
    ],
    # Generative-text calls only. messages.count_tokens dropped — a tokenizer utility,
    # not a model generation (no determinism knobs). (Claude has no embeddings/audio/
    # image endpoints, so nothing else to cut.)
    "anthropic": [
        "messages.create",
        "messages.stream",
        "completions.create",
        "messages.batches.create",
        "beta.messages.create",
    ],


    # ── LangChain family ───────────────────────────────────────────────────────
    # DONE
    # Invocation methods only — Runnable interface + legacy LLM/Chain calls.
    # Constructor/template/message/handler classes are intentionally excluded:
    # constructing them does not invoke a model.
    # .predict/.apredict CUT (2026-08-06): the deprecated LLM.predict method collides
    # with the sklearn/MLflow model API — of 62 pilot hits, ~57 were ML predictions
    # (pyfunc_loaded_model/sklearn_knn/rf.predict in mlflow), only ~4 real llm.predict.
    # Modern langchain uses .invoke (kept). .predict_messages/.apredict_messages kept
    # (langchain-specific, no sklearn collision). See EXCLUSIONS.md §4.
    "langchain": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
        ".astream_events", ".stream_events", ".transform", ".atransform",
    ],
    "langchain_openai": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
        ".astream_events", ".stream_events", ".transform", ".atransform",
    ],
    "langchain_anthropic": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
        ".astream_events", ".stream_events", ".transform", ".atransform",
    ],
    "langchain_community": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
        ".astream_events", ".stream_events", ".transform", ".atransform",
    ],
    "langchain_core": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
        ".astream_events", ".stream_events", ".transform", ".atransform",
    ],
    "langchain_experimental": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
        ".astream_events", ".stream_events", ".transform", ".atransform",
    ],
    # DONE — LangGraph's CompiledStateGraph implements only the Runnable interface;
    # it has none of langchain's legacy LLMChain/Chain methods. Pilot confirms this:
    # of 382 langgraph invocations, every hit was a Runnable-core method (.invoke 175,
    # .ainvoke 114, .stream 47, .astream 38, .batch 4, .abatch 3, .astream_events 1);
    # all 12 legacy methods (.run/.arun/.call/.acall/.predict*/.generate*/.transform*)
    # recorded exactly 0 — dropped, since for langgraph they're pure collision risk
    # (.run = tool exec, .predict = sklearn/mlflow).
    "langgraph": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".astream_events", ".stream_events",
    ],

    # ── AutoGen family ─────────────────────────────────────────────────────────
    # DONE
    "autogen": [
        ".initiate_chat",
        ".generate_reply",
        ".a_initiate_chat",
        ".a_generate_reply",
        ".send",
        ".a_send",
    ],
    "autogen_core": [
        ".create",         # ChatCompletionClient.create — the direct model call
        ".create_stream",  # ChatCompletionClient.create_stream — streaming model call
        ".send_message",
        ".publish_message",
    ],
    "autogen_agentchat": [
        ".initiate_chat",
        ".generate_reply",
        ".run",
        ".run_stream",
        ".on_messages",
        ".on_messages_stream",
    ],
    # autogen_ext.models.* — the concrete ChatCompletionClient implementations
    # (OpenAIChatCompletionClient, AnthropicChatCompletionClient, Azure…, etc.). The
    # model call is client.create / .create_stream, same surface as autogen_core. Its
    # own key so a file that imports the client from autogen_ext (but not autogen_core)
    # still gets .create matched. 0 in the pilot (unsampled), real for the full run.
    # Auto-scoped via the "autogen" IN_SCOPE prefix.
    "autogen_ext": [
        ".create",
        ".create_stream",
    ],

    # ── CrewAI ─────────────────────────────────────────────────────────────────
    # DONE — .kickoff* are the Crew orchestration entry points; .call/.acall are the
    # crewai.LLM class direct model calls (llm.call("prompt") / await llm.acall(...)).
    "crewai": [
        ".kickoff",
        ".kickoff_async",
        ".kickoff_for_each",
        ".kickoff_for_each_async",
        ".call",
        ".acall",
    ],

    # ── OpenAI Swarm ───────────────────────────────────────────────────────────
    # DONE — OpenAI's minimal (archived) Swarm: the sole invocation is Swarm().run(...)
    # (pilot & docs agree; sync-only, no async, streaming is run(stream=True)). Both
    # pilot repos use exactly `app = Swarm(); app.run(...)`. .run is the whole surface.
    "swarm": [
        ".run",
    ],

    # ── OpenAI Agents SDK ──────────────────────────────────────────────────────
    "agents": [
        "Runner.run",
        "Runner.run_sync",
        "Runner.stream",
        "Runner.run_streamed",
    ],

    # ── PydanticAI ─────────────────────────────────────────────────────────────
    # DONE
    "pydantic_ai": [
        ".run",
        ".run_sync",
        ".run_stream",
        # pydantic_ai.direct — low-level imperative model requests (no Agent wrapper)
        "model_request",
        "model_request_sync",
        "model_request_stream",
        "model_request_stream_sync",
    ],

    # ── MetaGPT ────────────────────────────────────────────────────────────────
    "metagpt": [
        ".run",
        ".arun",
        ".run_project",
        "Team",
        "Role",
        "Message",
    ],

    # ── CAMEL ──────────────────────────────────────────────────────────────────
    # DONE — CAMEL agents (ChatAgent / RolePlaying / sessions) are invoked via .step /
    # .astep (pilot: .step 117, .astep 2; receivers verified as *_agent.step /
    # session.step). Cut .chat/.achat — not CAMEL's API; their 14 hits were OpenAI/
    # Cohere/Mistral client collisions (.chat inside client.chat.completions.create).
    # The bare ChatAgent/RolePlaying tokens were construction-only and already stripped
    # by the bare-name loop anyway.
    "camel": [
        ".step",
        ".astep",
    ],

    # ── Griptape ───────────────────────────────────────────────────────────────
    "griptape": [
        ".run",
        "Pipeline",
        "Workflow",
        "Agent",
        "PromptTask",
    ],

    # ── AdalFlow ───────────────────────────────────────────────────────────────
    "adalflow": [
        ".call",
        ".acall",
        ".forward",
        "Generator",
        "Runner",
    ],

    # ── Agency Swarm ───────────────────────────────────────────────────────────
    "agency_swarm": [
        ".run_demo",
        ".initiate_chat",
        ".get_completion",
        ".get_completion_stream",
        "Agency",
        "Agent",
    ],

    # ── Swarms ─────────────────────────────────────────────────────────────────
    # DONE — .run/.arun are the correct API (Agent/Workflow.run); 0 in pilot = not
    # sampled (swarms was below the pre-grouping top-20 cut), not a pattern gap.
    "swarms": [
        ".run",
        ".arun",
        "Swarm",
        "Agent",
        "SequentialWorkflow",
    ],

    # ── Parlant ────────────────────────────────────────────────────────────────
    "parlant": [
        ".run",
        ".arun",
        "Agent",
        "Session",
    ],

    # ── Dynamiq ────────────────────────────────────────────────────────────────
    "dynamiq": [
        ".run",
        ".arun",
        "Workflow",
        "Agent",
    ],

    # ── LiveKit Agents ─────────────────────────────────────────────────────────
    "livekit": [
        ".run",
        ".arun",
        ".generate_reply",
        ".say",
        "WorkerOptions",
        "JobProcess",
    ],

    # ── TEN Framework ──────────────────────────────────────────────────────────
    "ten": [
        ".run",
        ".start",
        "TenEnv",
        "Extension",
    ],

    # ── BeeAI Framework ────────────────────────────────────────────────────────
    "beeai_framework": [
        ".run",
        ".arun",
        ".stream",
        "BeeAgent",
        "ReActAgent",
    ],

    # ── PraisonAI ──────────────────────────────────────────────────────────────
    "praisonai": [
        ".start",
        ".run",
        ".chat",
        "PraisonAI",
        "Agent",
    ],

    # ── SuperAGI ───────────────────────────────────────────────────────────────
    "superagi": [
        ".run",
        ".execute_next_action",
        "SuperAgi",
    ],

    # ── RagaAI Catalyst ────────────────────────────────────────────────────────
    "ragaai_catalyst": [
        ".run",
        ".evaluate",
        "RagaAICatalyst",
        "Tracer",
    ],

    # ── AgentUniverse ──────────────────────────────────────────────────────────
    "agentuniverse": [
        ".run",
        ".arun",
        "AgentManager",
    ],

    # ── Agent Squad (AWS) ──────────────────────────────────────────────────────
    "agent_squad": [
        ".route_request",
        ".process_request",
        "Orchestrator",
    ],

    # ── AgentOps ───────────────────────────────────────────────────────────────
    # EXEMPT (not in scope) — agentops is an OBSERVABILITY layer, not an invoker: it
    # instruments whatever real framework the app uses (langchain/openai/crewai/…). Its
    # methods are pure telemetry lifecycle (.init/.record/.start_session/.end_session),
    # not model calls (pilot: 14 "hits", 0 real invocations). An agentops app's LLM
    # calls are already counted via its underlying framework, so agentops adds no
    # independent signal. Removed from _IN_SCOPE_EXPLICIT; patterns kept for provenance.
    "agentops": [
        ".init",
        ".start_session",
        ".end_session",
        ".record",
    ],

    # ── OpenLIT ────────────────────────────────────────────────────────────────
    "openlit": [
        ".init",
        ".trace",
    ],

    # ── Giskard ────────────────────────────────────────────────────────────────
    # Moved to pipeline/eval_calls.py EVAL_CALLS — giskard is a semantic-evaluation
    # / testing tool, not an LLM-invoking agent framework, so it belongs to the
    # eval pass only (Stage 7), not the LLM invoker pass (Stage 5).

    # ── SuperDuper ─────────────────────────────────────────────────────────────
    "superduper": [
        ".predict",
        ".fit",
        ".apply",
    ],

    # ── ii-agent ───────────────────────────────────────────────────────────────
    # Commented out — run-as-app; no confirmable LLM invoker surfaced in the repo.
    # "ii_agent": [
    #     ".run",
    #     ".execute",
    #     "IIAgent",
    # ],

    # ── LaVague ────────────────────────────────────────────────────────────────
    "lavague": [
        ".run",
        ".execute",
        "WebAgent",
        "ActionEngine",
    ],

    # ── Cheshire Cat (distributed package imports as `cat`, not `cheshire_cat`) ──
    "cat": [
        ".run",
        ".send",
        "CatClient",
    ],

    # ── Solace Agent Mesh ──────────────────────────────────────────────────────
    "solace_agent_mesh": [
        ".run",
        ".publish",
        "SolaceAgentMesh",
    ],

    # ── Misc remaining frameworks ──────────────────────────────────────────────
    "lagent": [".run", ".step", ".chat", ".stream_chat", ".forward", "ActionExecutor"],
    "patchwork": [".run", "PatchFlow"],
    "npcpy": [".run", ".chat", "get_llm_response", "NPC"],
    "any_agent": [".run", ".run_async", "AnyAgent"],
    "sage": [".run", ".query", "Sage"],
    "honcho": [".create", ".chat", "Honcho"],  # .get dropped — it matches dict/config .get, not an LLM call
    "uagents": [".run", ".send", "Agent", "Bureau"],
    "agent_protocol": [".run", ".step", "Agent"],
    "infiagent": [".run", "InfiAgent"],
    # DONE — notte (web-browsing agent): agents/sessions are invoked via .run. The old
    # bare "Notte" construction token was stripped by the bare-name loop anyway (notte
    # has a .method and isn't in _KEEP_BARE_NAMES), so this just matches reality.
    "notte": [".run"],
    "redamon": [".run"],
    # DONE — Microsoft Agent Framework (SK + AutoGen successor): agents invoked via
    # .run / .run_stream (pilot: .run 193, .run_stream 5); .execute is the Workflow
    # Executor method (0 in pilot, kept as a real-but-unsampled surface).
    "agent_framework": [".run", ".execute", ".run_stream"],
    "llmstack": [".run", "LLMStack"],
    # Commented out — web framework (Reflex fork); no user-called LLM invoker found.
    # "nextpy": [".run", "App"],

    # ── Added: top-20 frameworks that were missing from the invoker detector ─────
    # DONE — mem0: .add (LLM fact-extraction) and .search (LLM query) invoke an LLM
    # internally, so they count as invokers per project definition. .get_all/.update
    # dropped: those are plain vector/DB lookups & writes, no LLM call. (No class name
    # — a Memory(...) constructor is not itself an invocation.)
    "mem0": [".add", ".search"],
    # DONE — haystack: Pipelines and Generators are invoked via .run / .run_async
    # (pilot: .run 70, .run_async 10).
    "haystack": [".run", ".run_async"],
    # DONE — smolagents: agents (CodeAgent / ToolCallingAgent) are invoked via .run
    # (pilot: .run 45).
    "smolagents": [".run"],
    # DONE — agno: agents are invoked via .run / .print_response (+ async variants)
    # (pilot: .run 2503, .arun 1095, .print_response 257, .aprint_response 210).
    # Deliberately NOT matching the bare `Agent` name — that hits every Agent(...)
    # constructor, which builds an agent but invokes no LLM (see the agno over-count).
    "agno": [".run", ".arun", ".print_response", ".aprint_response"],
    # dspy: LLM calls go through module objects invoked as bare callables
    # (`pred = dspy.Predict(sig); pred(...)`), which a .method list can't see.
    # DONE — the __call__ site is now detected structurally by seed_invokers: it binds
    # any name assigned from a dspy module constructor (DSPY_MODULE_CLASSES below) and
    # flags the later bare call `pred(...)` / `self.prog(...)` as the invoker, localized
    # to the calling function. So the dict only carries the two direct-invoke methods;
    # the class names moved to DSPY_MODULE_CLASSES (a construction *signal*, not a match).
    "dspy": [".forward", ".aforward"],
    # DONE — graphrag: the query API is module-level FUNCTIONS (global/local/drift/
    # basic_search), confirmed real in the pilot (await global_search(...) etc.). These
    # bare names are kept via _KEEP_BARE_NAMES. Cut .search/.asearch — every one of the
    # 26 pilot hits was a collision (re.search, file_pattern.search, and vector/DB
    # client.search / dataStore.search), zero real graphrag engine calls.
    "graphrag": ["global_search", "local_search", "drift_search", "basic_search"],
    # DONE — semantic_kernel (Microsoft SK, Python). Kernel invocation: .invoke /
    # .invoke_prompt / .invoke_stream / .invoke_prompt_stream (all "execute a function/
    # prompt" -> model call). ChatCompletion service: .get_chat_message_content(s) and
    # .get_streaming_chat_message_content (despite the "get" name these ARE the model
    # call — send chat_history, return the AI message). SK Agent: .get_response.
    # Dropped .invoke_async / .invoke_prompt_async — C#-style pre-1.0 naming, not
    # current Python SK (the real stream-prompt method is .invoke_prompt_stream).
    # Excluded: invoke_function_call (tool-call execution, not a completion) and
    # add_embedding_to_object (memory embedding helper; embeddings caught via the SDK).
    # (No `Kernel` class name — Kernel(...) construction is not an invocation.)
    "semantic_kernel": [".invoke", ".invoke_prompt", ".invoke_stream",
                        ".invoke_prompt_stream",
                        ".get_chat_message_content", ".get_chat_message_contents",
                        ".get_streaming_chat_message_content", ".get_response"],
    # DONE — astrbot: multi-platform LLM chatbot framework. Current (v4.5.7+) plugin
    # API on self.context: .llm_generate (primary model call) and .tool_loop_agent
    # (agent + tools). Legacy but still common in pilot plugins: provider.text_chat /
    # .text_chat_stream after context.get_using_provider() (pilot: .text_chat 9). Cut
    # .get_using_provider (a provider getter, not a call — its fn is caught by
    # .text_chat, cf. mem0 .get_all), .chat (not AstrBot's API; collides with
    # client.chat.*), and .request_llm (unconfirmed in current docs, 0 hits).
    # CAVEAT: astrbot is a plugin PLATFORM — real LLM calls often live in core or go
    # through the raw SDK, so these per-plugin counts undercount; don't rank astrbot on
    # raw invocation counts. See memory project_astrbot_platform_skew.
    "astrbot": [".text_chat", ".text_chat_stream", ".llm_generate", ".tool_loop_agent"],
    # headroom: built on the LangChain Runnable interface — LLM calls go through
    # .invoke/.run/.chat/.completion. Patterns derived from headroomlabs-ai/headroom.
    "headroom": [".invoke", ".ainvoke", ".run", ".arun", ".chat",
                 ".completion", ".acompletion", ".get_response"],
}

# dspy module classes: instances of these are callable — invoking one (`pred(...)`)
# runs __call__ → forward → the LLM. seed_invokers binds names assigned from these
# constructors and treats the later call on the bound name as the invocation site.
# This is a construction *signal* used for binding, NOT a call pattern that gets
# matched against call text (that would flag the construction site, not the call).
DSPY_MODULE_CLASSES = {
    "Predict", "ChainOfThought", "ReAct", "ProgramOfThought",
    "MultiChainComparison", "TypedPredictor", "Retrieve", "Module",
}


# LangChain integration packages (chat-model providers) all expose the standard
# Runnable interface, so they share langchain's invoke/stream/batch patterns. A repo
# that imports only e.g. langchain_mistralai still calls .invoke — cover them all so
# provider-only imports aren't missed. setdefault leaves existing keys untouched.
_LANGCHAIN_INTEGRATIONS = [
    "langchain_mistralai", "langchain_ollama", "langchain_deepseek", "langchain_groq",
    "langchain_huggingface", "langchain_xai", "langchain_fireworks",
    "langchain_perplexity", "langchain_openrouter", "langchain_google_genai",
    "langchain_google_vertexai", "langchain_aws", "langchain_cohere",
    "langchain_together", "langchain_nvidia_ai_endpoints",
]
for _pkg in _LANGCHAIN_INTEGRATIONS:
    FRAMEWORK_CALLS.setdefault(_pkg, FRAMEWORK_CALLS["langchain"])

# Strip bare CapWords class-name tokens (e.g. "Agent", "Team", "CatClient") from any
# framework that already has a ".method" invocation pattern — a bare class name counts
# CONSTRUCTION (`Agent(...)`), not an invocation, and inflated agno ~2x before we caught
# it. Kept deliberately: (1) dspy/graphrag, whose bare names ARE the only invocation
# signal (bare-callable modules / module-level functions); (2) `Class.method` tokens
# like `Runner.run`, which are real invocation calls; (3) lowercase function names like
# `get_llm_response`. A framework whose ONLY signal is a class name is left untouched
# (has_method is False), so nothing gets zeroed out.
# dspy no longer relies on bare class names (its __call__ site is bound structurally
# in seed_invokers via DSPY_MODULE_CLASSES); only graphrag's module-level query
# functions remain a bare-name signal.
_KEEP_BARE_NAMES = {"graphrag"}
for _fw, _pats in list(FRAMEWORK_CALLS.items()):
    if _fw in _KEEP_BARE_NAMES or not any(p.startswith(".") for p in _pats):
        continue
    FRAMEWORK_CALLS[_fw] = [
        p for p in _pats
        if p.startswith(".") or "." in p or (p[:1].islower())
    ]


# ── study scope ────────────────────────────────────────────────────────────────
# FRAMEWORK_CALLS above is the FULL Stage-1 discovery record — every framework we
# ever found, kept intact for provenance. But the study examines only the top-20
# frameworks (by app coverage, per keep_frequency) plus the raw LLM SDKs and the
# OpenAI Agents SDK. Detection is scoped to these via SCOPED_FRAMEWORK_CALLS so we
# don't spend each run matching — and collecting unrepresentative data for — the ~31
# frameworks below the top-20 (their invocations would only ever be incidental hits
# inside top-20 repos). Nothing is deleted; out-of-scope patterns simply aren't run.
#
# Import-name variants roll up to their parent framework, so the whole langchain /
# autogen / agent_framework families are in scope.
#
# omnigent is EXEMPT (not in scope): it's a CLI-based agent that invokes the LLM in a
# spawned subprocess, so static method-matching can't see its calls (0 invocations is
# the out-of-process blind spot, not a real count) — and its `omnigent` token also
# collides with the diffusion class OmniGe(nt)ransformer, mixing in non-LLM repos. It
# can't be reliably measured either way, so we exempt it rather than report skewed
# data. Not a phantom — see memory project_out_of_process_invocation. (connectonion was
# already cut as a true phantom.)
_IN_SCOPE_EXPLICIT = {
    "langgraph", "pydantic_ai", "crewai", "mem0", "dspy", "haystack",
    "smolagents", "agno", "camel", "astrbot", "graphrag",
    "semantic_kernel", "swarms", "swarm", "notte",
    "openai", "anthropic",        # raw LLM provider SDKs (the actual call surface)
    "agents",                     # OpenAI Agents SDK (Runner.run — specific, real usage)
    # omnigent + agentops intentionally EXEMPT (out-of-process CLI / observability
    # layer respectively) — see notes at their FRAMEWORK_CALLS entries.
}
_IN_SCOPE_PREFIXES = ("langchain", "autogen", "agent_framework")
IN_SCOPE_FRAMEWORKS = _IN_SCOPE_EXPLICIT | {
    _k for _k in FRAMEWORK_CALLS
    if any(_k == _p or _k.startswith(_p + "_") for _p in _IN_SCOPE_PREFIXES)
}
SCOPED_FRAMEWORK_CALLS = {
    _k: _v for _k, _v in FRAMEWORK_CALLS.items() if _k in IN_SCOPE_FRAMEWORKS
}
