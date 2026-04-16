# bytedance/deer-flow
# 4 LLM-backed test functions across 109 test files
# Source: https://github.com/bytedance/deer-flow

# --- backend/tests/test_create_deerflow_agent.py ---

def test_agent_features_defaults():
    f = RuntimeFeatures()
    assert f.sandbox is True
    assert f.memory is False
    assert f.summarization is False
    assert f.subagent is False
    assert f.vision is False
    assert f.auto_title is False
    assert f.guardrail is False


# --- backend/tests/test_create_deerflow_agent_live.py ---

def test_minimal_agent_responds():
    """create_deerflow_agent(model) produces a graph that returns a response."""
    from deerflow.agents.factory import create_deerflow_agent

    model = _make_model()
    graph = create_deerflow_agent(model, features=None, middleware=[])

    result = graph.invoke(
        {"messages": [("user", "Say exactly: pong")]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )

    messages = result.get("messages", [])
    assert len(messages) >= 2
    last_msg = messages[-1]
    assert hasattr(last_msg, "content")
    assert len(last_msg.content) > 0

def test_agent_with_custom_tool():
    """Agent can invoke a user-provided tool and return the result."""
    from deerflow.agents.factory import create_deerflow_agent

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    model = _make_model()
    graph = create_deerflow_agent(model, tools=[add], middleware=[])

    result = graph.invoke(
        {"messages": [("user", "Use the add tool to compute 3 + 7. Return only the result.")]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )

    messages = result.get("messages", [])
    # Should have: user msg, AI tool_call, tool result, AI final
    assert len(messages) >= 3
    last_content = messages[-1].content
    assert "10" in last_content

def test_features_mode_middleware_chain():
    """RuntimeFeatures assembles a working middleware chain that executes."""
    from deerflow.agents.factory import create_deerflow_agent
    from deerflow.agents.features import RuntimeFeatures

    model = _make_model()
    feat = RuntimeFeatures(sandbox=False, auto_title=False, memory=False)
    graph = create_deerflow_agent(model, features=feat)

    result = graph.invoke(
        {"messages": [("user", "What is 2+2?")]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )

    messages = result.get("messages", [])
    assert len(messages) >= 2
    last_content = messages[-1].content
    assert len(last_content) > 0

