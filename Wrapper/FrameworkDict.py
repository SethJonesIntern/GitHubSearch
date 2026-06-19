"""Per-framework call patterns used as seeds by transitive_invokers.py."""

# Maps each framework's top-level import name to the call patterns specific to
# that framework.  When a file imports a framework, the scanner searches only
# for that framework's patterns — reducing noise from generic names like .run.
FRAMEWORK_CALLS: dict[str, list[str]] = {
    # ── Direct LLM SDKs ────────────────────────────────────────────────────────
    "openai": [
        "chat.completions.create", "completions.create", "responses.create",
        "images.generate", "chat.completions.parse", "responses.parse",
        "responses.stream", "embeddings.create", "moderations.create",
        "audio.transcriptions.create", "audio.translations.create",
        "audio.speech.create",
    ],
    "anthropic": [
        "messages.create",
        "messages.stream",
        "messages.count_tokens",
        "completions.create",
        "messages.batches.create",
        "beta.messages.create",
    ],


    # ── LangChain family ───────────────────────────────────────────────────────
    # Invocation methods only — Runnable interface + legacy LLM/Chain calls.
    # Constructor/template/message/handler classes are intentionally excluded:
    # constructing them does not invoke a model.
    "langchain": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict", ".apredict", ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
        ".astream_events", ".stream_events", ".transform", ".atransform",
    ],
    "langchain_openai": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict", ".apredict", ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
        ".astream_events", ".stream_events", ".transform", ".atransform",
    ],
    "langchain_anthropic": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict", ".apredict", ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
        ".astream_events", ".stream_events", ".transform", ".atransform",
    ],
    "langchain_community": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict", ".apredict", ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
        ".astream_events", ".stream_events", ".transform", ".atransform",
    ],
    "langchain_core": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict", ".apredict", ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
        ".astream_events", ".stream_events", ".transform", ".atransform",
    ],
    "langchain_experimental": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict", ".apredict", ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
        ".astream_events", ".stream_events", ".transform", ".atransform",
    ],
    "langgraph": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".astream_events", ".stream_events",
    ],

    # ── AutoGen family ─────────────────────────────────────────────────────────
    "autogen": [
        ".initiate_chat",
        ".generate_reply",
        ".a_initiate_chat",
        ".a_generate_reply",
        ".send",
        ".a_send",
    ],
    "autogen_core": [
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

    # ── CrewAI ─────────────────────────────────────────────────────────────────
    "crewai": [
        ".kickoff",
        ".kickoff_async",
        ".kickoff_for_each",
        ".kickoff_for_each_async",
    ],

    # ── OpenAI Swarm ───────────────────────────────────────────────────────────
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
    "pydantic_ai": [
        ".run",
        ".run_sync",
        ".run_stream",
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
    "camel": [
        ".step",
        ".astep",
        ".chat",
        ".achat",
        "ChatAgent",
        "RolePlaying",
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

    # ── Cheshire Cat ───────────────────────────────────────────────────────────
    "cheshire_cat": [
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
    "honcho": [".create", ".get", ".chat", "Honcho"],
    "uagents": [".run", ".send", "Agent", "Bureau"],
    "agent_protocol": [".run", ".step", "Agent"],
    "infiagent": [".run", "InfiAgent"],
    "notte": [".run", "Notte"],
    "redamon": [".run"],
    "agent_framework": [".run", ".execute", ".run_stream"],
    "llmstack": [".run", "LLMStack"],
    # Commented out — web framework (Reflex fork); no user-called LLM invoker found.
    # "nextpy": [".run", "App"],
}
