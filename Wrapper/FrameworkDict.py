"""Per-framework call patterns used as seeds by transitive_invokers.py."""

# Maps each framework's top-level import name to the call patterns specific to
# that framework.  When a file imports a framework, the scanner searches only
# for that framework's patterns — reducing noise from generic names like .run.
FRAMEWORK_CALLS: dict[str, list[str]] = {
    # ── Direct LLM SDKs ────────────────────────────────────────────────────────
    "openai": [
        "chat.completions.create", "completions.create", "responses.create",
               "images.generate"
    ],
    "anthropic": [
        "messages.create",
        "messages.stream",
        "messages.count_tokens",
    ],

    #  # ── chatchat ───────────────────────────────────────────────────────────
    # "chatchat": [
    #     "get_ChatOpenAI",
    #     "query_prometheus",
    #     "create_platform_knowledge_agent",
    #     "create_platform_tools_agent",
    #     "create_qwen_chat_agent",
    #     "create_chat_agent"
    # ],


    # ── LangChain family ───────────────────────────────────────────────────────
    # Invocation methods only — Runnable interface + legacy LLM/Chain calls.
    # Constructor/template/message/handler classes are intentionally excluded:
    # constructing them does not invoke a model.
    "langchain": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict", ".apredict", ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
    ],
    "langchain_openai": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict", ".apredict", ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
    ],
    "langchain_anthropic": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict", ".apredict", ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
    ],
    "langchain_community": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict", ".apredict", ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
    ],
    "langchain_core": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict", ".apredict", ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
    ],
    "langchain_experimental": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
        ".predict", ".apredict", ".predict_messages", ".apredict_messages",
        ".generate", ".agenerate", ".run", ".arun", ".call", ".acall",
    ],
    "langgraph": [
        ".invoke", ".ainvoke", ".stream", ".astream", ".batch", ".abatch",
    ],

    # ── AutoGen family ─────────────────────────────────────────────────────────
    "autogen": [
        ".initiate_chat",
        ".generate_reply",
        ".a_initiate_chat",
    ],
    "autogen_core": [
        ".send_message",
        ".publish_message",
    ],
    "autogen_agentchat": [
        ".initiate_chat",
        ".generate_reply",
    ],

    # ── CrewAI ─────────────────────────────────────────────────────────────────
    "crewai": [
        ".kickoff",
        ".kickoff_async",
        ".kickoff_for_each",
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
        "Team",
        "Role",
        "Message",
    ],

    # ── CAMEL ──────────────────────────────────────────────────────────────────
    "camel": [
        ".step",
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
        ".stream",
        "BeeAgent",
        "ReActAgent",
    ],

    # ── PraisonAI ──────────────────────────────────────────────────────────────
    "praisonai": [
        ".start",
        ".run",
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
    "giskard": [
        ".scan",
        ".evaluate",
        "Model",
        "Dataset",
    ],

    # ── SuperDuper ─────────────────────────────────────────────────────────────
    "superduper": [
        ".predict",
        ".fit",
        ".apply",
    ],

    # ── ii-agent ───────────────────────────────────────────────────────────────
    "ii_agent": [
        ".run",
        ".execute",
        "IIAgent",
    ],

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
    "lagent": [".run", ".step", "ActionExecutor"],
    "patchwork": [".run", "PatchFlow"],
    "npcpy": [".run", ".chat", "NPC"],
    "any_agent": [".run", "AnyAgent"],
    "sage": [".run", ".query", "Sage"],
    "honcho": [".create", ".get", "Honcho"],
    "uagents": [".run", ".send", "Agent", "Bureau"],
    "agent_protocol": [".run", ".step", "Agent"],
    "infiagent": [".run", "InfiAgent"],
    "notte": [".run", "Notte"],
    "redamon": [".run"],
    "agent_framework": [".run", ".execute"],
    "llmstack": [".run", "LLMStack"],
    "nextpy": [".run", "App"],
}
