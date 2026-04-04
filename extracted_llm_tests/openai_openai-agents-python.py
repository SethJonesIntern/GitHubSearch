# openai/openai-agents-python
# 409 test functions with real LLM calls
# Source: https://github.com/openai/openai-agents-python


# --- tests/extensions/experiemental/codex/test_codex_tool.py ---

def test_codex_tool_truncates_span_values() -> None:
    value = {"payload": "x" * 200}
    truncated = codex_tool_module._truncate_span_value(value, 40)

    assert isinstance(truncated, dict)
    assert truncated["truncated"] is True
    assert truncated["original_length"] > 40
    preview = truncated["preview"]
    assert isinstance(preview, str)
    assert len(preview) <= 40

def test_codex_tool_rejects_lossy_default_run_context_thread_id_key_suffix() -> None:
    with pytest.raises(UserError, match="run_context_thread_id_key"):
        codex_tool(name="codex_a-b", use_run_context_thread_id=True)

def test_codex_tool_run_context_mode_hides_thread_id_in_default_parameters() -> None:
    tool = codex_tool(use_run_context_thread_id=True)
    assert "thread_id" not in tool.params_json_schema["properties"]

async def test_codex_tool_duplicate_names_fail_fast() -> None:
    agent = Agent(
        name="test",
        tools=[
            codex_tool(),
            codex_tool(),
        ],
    )

    with pytest.raises(UserError, match="Duplicate Codex tool names found"):
        await agent.get_all_tools(RunContextWrapper(context=None))

async def test_codex_tool_name_collision_with_other_tool_fails_fast() -> None:
    @function_tool(name_override="codex")
    def other_tool() -> str:
        return "ok"

    agent = Agent(
        name="test",
        tools=[
            codex_tool(),
            other_tool,
        ],
    )

    with pytest.raises(UserError, match="Duplicate Codex tool names found"):
        await agent.get_all_tools(RunContextWrapper(context=None))

def test_codex_tool_keyword_rejects_empty_run_context_key() -> None:
    with pytest.raises(UserError, match="run_context_thread_id_key"):
        codex_tool(run_context_thread_id_key=" ")

def test_codex_tool_truncate_span_string_limits() -> None:
    assert codex_tool_module._truncate_span_string("hello", 0) == ""
    long_value = "x" * 100
    assert codex_tool_module._truncate_span_string(long_value, 3) == "xxx"

def test_codex_tool_truncate_span_value_handles_circular_reference() -> None:
    value: list[Any] = []
    value.append(value)
    truncated = codex_tool_module._truncate_span_value(value, 1)
    assert isinstance(truncated, dict)
    assert truncated["truncated"] is True

async def test_codex_tool_on_invoke_tool_handles_failure_error_function_sync() -> None:
    def failure_error_function(_ctx: RunContextWrapper[Any], _exc: Exception) -> str:
        return "handled"

    tool = codex_tool(CodexToolOptions(failure_error_function=failure_error_function))
    input_json = "{bad"
    context = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="call-1",
        tool_arguments=input_json,
    )

    result = await tool.on_invoke_tool(context, input_json)
    assert result == "handled"

async def test_codex_tool_on_invoke_tool_handles_failure_error_function_async() -> None:
    async def failure_error_function(_ctx: RunContextWrapper[Any], _exc: Exception) -> str:
        return "handled-async"

    tool = codex_tool(CodexToolOptions(failure_error_function=failure_error_function))
    input_json = "{bad"
    context = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="call-1",
        tool_arguments=input_json,
    )

    result = await tool.on_invoke_tool(context, input_json)
    assert result == "handled-async"

async def test_codex_tool_on_invoke_tool_raises_without_failure_handler() -> None:
    tool = codex_tool(CodexToolOptions(failure_error_function=None))
    input_json = "{bad"
    context = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="call-1",
        tool_arguments=input_json,
    )

    with pytest.raises(ModelBehaviorError):
        await tool.on_invoke_tool(context, input_json)

async def test_replaced_codex_tool_normal_failure_uses_replaced_policy() -> None:
    tool = dataclasses.replace(
        codex_tool(CodexToolOptions()),
        _failure_error_function=None,
        _use_default_failure_error_function=False,
    )
    input_json = "{bad"
    context = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="call-1",
        tool_arguments=input_json,
    )

    with pytest.raises(ModelBehaviorError):
        await tool.on_invoke_tool(context, input_json)

async def test_replaced_codex_tool_preserves_codex_collision_markers() -> None:
    agent = Agent(
        name="test",
        tools=[
            dataclasses.replace(codex_tool(CodexToolOptions()), name="shared_codex_tool"),
            dataclasses.replace(codex_tool(CodexToolOptions()), name="shared_codex_tool"),
        ],
    )

    with pytest.raises(UserError, match="Duplicate Codex tool names found: shared_codex_tool"):
        await agent.get_all_tools(RunContextWrapper(None))

async def test_codex_tool_consume_events_with_on_stream_error() -> None:
    events = [
        {
            "type": "item.started",
            "item": {
                "id": "cmd-1",
                "type": "command_execution",
                "command": "ls",
                "status": "in_progress",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "cmd-1",
                "type": "command_execution",
                "command": "ls",
                "status": "completed",
                "exit_code": 0,
            },
        },
        {
            "type": "item.started",
            "item": {
                "id": "mcp-1",
                "type": "mcp_tool_call",
                "server": "server",
                "tool": "tool",
                "arguments": {"q": "x"},
                "status": "in_progress",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "mcp-1",
                "type": "mcp_tool_call",
                "server": "server",
                "tool": "tool",
                "arguments": {"q": "x"},
                "status": "failed",
                "error": {"message": "boom"},
            },
        },
        {
            "type": "item.completed",
            "item": {"id": "agent-1", "type": "agent_message", "text": "done"},
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1},
        },
    ]

    async def event_stream():
        for event in events:
            yield event

    callbacks: list[str] = []

    def on_stream(payload: CodexToolStreamEvent) -> None:
        callbacks.append(payload.event.type)
        if payload.event.type == "item.started":
            raise RuntimeError("boom")

    context = ToolContext(
        context=None,
        tool_name="codex",
        tool_call_id="call-1",
        tool_arguments="{}",
    )

    with trace("codex-test"):
        response, usage, thread_id = await codex_tool_module._consume_events(
            event_stream(),
            {"inputs": [{"type": "text", "text": "hello"}]},
            context,
            SimpleNamespace(id="thread-1"),
            on_stream,
            64,
        )

    assert response == "done"
    assert usage == Usage(input_tokens=1, cached_input_tokens=0, output_tokens=1)
    assert thread_id == "thread-1"
    assert "item.started" in callbacks

async def test_codex_tool_consume_events_default_response() -> None:
    events = [
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1},
        }
    ]

    async def event_stream():
        for event in events:
            yield event

    context = ToolContext(
        context=None,
        tool_name="codex",
        tool_call_id="call-1",
        tool_arguments="{}",
    )

    response, usage, thread_id = await codex_tool_module._consume_events(
        event_stream(),
        {"inputs": [{"type": "text", "text": "hello"}]},
        context,
        SimpleNamespace(id="thread-1"),
        None,
        None,
    )

    assert response == "Codex task completed with inputs."
    assert usage == Usage(input_tokens=1, cached_input_tokens=0, output_tokens=1)
    assert thread_id == "thread-1"

async def test_codex_tool_consume_events_turn_failed() -> None:
    events = [{"type": "turn.failed", "error": {"message": "boom"}}]

    async def event_stream():
        for event in events:
            yield event

    context = ToolContext(
        context=None,
        tool_name="codex",
        tool_call_id="call-1",
        tool_arguments="{}",
    )

    with pytest.raises(UserError, match="Codex turn failed: boom"):
        await codex_tool_module._consume_events(
            event_stream(),
            {"inputs": [{"type": "text", "text": "hello"}]},
            context,
            SimpleNamespace(id="thread-1"),
            None,
            None,
        )

async def test_codex_tool_consume_events_error_event() -> None:
    events = [{"type": "error", "message": "boom"}]

    async def event_stream():
        for event in events:
            yield event

    context = ToolContext(
        context=None,
        tool_name="codex",
        tool_call_id="call-1",
        tool_arguments="{}",
    )

    with pytest.raises(UserError, match="Codex stream error"):
        await codex_tool_module._consume_events(
            event_stream(),
            {"inputs": [{"type": "text", "text": "hello"}]},
            context,
            SimpleNamespace(id="thread-1"),
            None,
            None,
        )

def test_codex_tool_coerce_options_rejects_empty_run_context_key() -> None:
    with pytest.raises(UserError, match="run_context_thread_id_key"):
        codex_tool_module._coerce_tool_options(
            {
                "use_run_context_thread_id": True,
                "run_context_thread_id_key": " ",
            }
        )


# --- tests/extensions/memory/test_advanced_sqlite_session.py ---

async def test_tool_usage_tracking_preserves_namespaces_and_tool_search(agent: Agent):
    """Tool usage should retain namespaces and count tool_search calls once."""
    session_id = "tools_namespace_test"
    session = AdvancedSQLiteSession(session_id=session_id, create_tables=True)

    items: list[TResponseInputItem] = [
        {"role": "user", "content": "Look up the same account in multiple systems"},
        {
            "type": "function_call",
            "name": "lookup_account",
            "namespace": "crm",
            "arguments": '{"account_id": "acct_123"}',
            "call_id": "crm-call",
        },
        {
            "type": "function_call",
            "name": "lookup_account",
            "namespace": "billing",
            "arguments": '{"account_id": "acct_123"}',
            "call_id": "billing-call",
        },
        {
            "type": "tool_search_call",
            "id": "tsc_memory",
            "arguments": {"paths": ["crm"], "query": "lookup_account"},
            "execution": "server",
            "status": "completed",
        },
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_output",
                "id": "tso_memory",
                "execution": "server",
                "status": "completed",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup_account",
                        "description": "Look up an account.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "account_id": {
                                    "type": "string",
                                }
                            },
                            "required": ["account_id"],
                        },
                        "defer_loading": True,
                    }
                ],
            },
        ),
    ]
    await session.add_items(items)

    usage_by_tool = {tool_name: count for tool_name, count, _turn in await session.get_tool_usage()}

    assert usage_by_tool["crm.lookup_account"] == 1
    assert usage_by_tool["billing.lookup_account"] == 1
    assert usage_by_tool["tool_search"] == 1

    session.close()

async def test_tool_usage_tracking_counts_tool_search_output_without_matching_call(
    agent: Agent,
) -> None:
    """Tool-search output-only histories should still report one tool_search usage."""
    session_id = "tools_tool_search_output_only_test"
    session = AdvancedSQLiteSession(session_id=session_id, create_tables=True)

    items: list[TResponseInputItem] = [
        {"role": "user", "content": "Look up customer_42"},
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_output",
                "id": "tso_memory_only",
                "execution": "server",
                "status": "completed",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup_account",
                        "description": "Look up an account.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "account_id": {
                                    "type": "string",
                                }
                            },
                            "required": ["account_id"],
                        },
                    }
                ],
            },
        ),
    ]
    await session.add_items(items)

    usage_by_tool = {tool_name: count for tool_name, count, _turn in await session.get_tool_usage()}

    assert usage_by_tool["tool_search"] == 1

    session.close()


# --- tests/extensions/memory/test_sqlalchemy_session.py ---

async def test_add_items_concurrent_first_access_across_from_url_sessions_cross_loop(tmp_path):
    """Concurrent first writes should not race or hang across event loops."""
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'concurrent_from_url_cross_loop.db'}"
    barrier = threading.Barrier(2)
    results: list[tuple[str, str, Any]] = []
    results_lock = threading.Lock()

    def worker(session_id: str, content: str) -> None:
        async def run() -> tuple[str, Any]:
            session = SQLAlchemySession.from_url(session_id, url=db_url, create_tables=True)
            barrier.wait()
            try:
                await asyncio.wait_for(
                    session.add_items([{"role": "user", "content": content}]),
                    timeout=5,
                )
                stored = await session.get_items()
                return ("ok", stored)
            finally:
                await session.engine.dispose()

        try:
            status, payload = asyncio.run(run())
        except Exception as exc:
            status, payload = type(exc).__name__, str(exc)

        with results_lock:
            results.append((session_id, status, payload))

    threads = [
        threading.Thread(target=worker, args=("from_url_cross_loop_a", "one")),
        threading.Thread(target=worker, args=("from_url_cross_loop_b", "two")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        await asyncio.to_thread(thread.join)

    assert len(results) == 2
    assert [status for _, status, _ in results] == ["ok", "ok"]

    stored_by_session = {
        session_id: cast(list[TResponseInputItem], payload) for session_id, _, payload in results
    }
    assert stored_by_session["from_url_cross_loop_a"][0].get("content") == "one"
    assert stored_by_session["from_url_cross_loop_b"][0].get("content") == "two"

async def test_add_items_concurrent_first_access_with_shared_session_cross_loop(tmp_path):
    """A shared session instance should not hang when used from two event loops."""
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'shared_session_cross_loop.db'}"
    session = SQLAlchemySession.from_url(
        "shared_session_cross_loop",
        url=db_url,
        create_tables=True,
    )
    barrier = threading.Barrier(2)
    results: list[tuple[str, str]] = []
    results_lock = threading.Lock()

    def worker(content: str) -> None:
        async def run() -> None:
            barrier.wait()
            await asyncio.wait_for(
                session.add_items([{"role": "user", "content": content}]),
                timeout=5,
            )

        try:
            asyncio.run(run())
            status = "ok"
        except Exception as exc:
            status = type(exc).__name__

        with results_lock:
            results.append((content, status))

    threads = [
        threading.Thread(target=worker, args=("one",)),
        threading.Thread(target=worker, args=("two",)),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            await asyncio.to_thread(thread.join)

        assert sorted(results) == [("one", "ok"), ("two", "ok")]

        stored = await session.get_items()
        stored_contents: list[str] = []
        for item in stored:
            content = item.get("content")
            assert isinstance(content, str)
            stored_contents.append(content)
        assert sorted(stored_contents) == ["one", "two"]
    finally:
        await session.engine.dispose()


# --- tests/model_settings/test_serialization.py ---

def test_all_fields_serialization() -> None:
    """Tests whether ModelSettings can be serialized to a JSON string."""

    # First, lets create a ModelSettings instance
    model_settings = ModelSettings(
        temperature=0.5,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        tool_choice="auto",
        parallel_tool_calls=True,
        truncation="auto",
        max_tokens=100,
        reasoning=Reasoning(),
        metadata={"foo": "bar"},
        store=False,
        prompt_cache_retention="24h",
        include_usage=False,
        response_include=["reasoning.encrypted_content"],
        top_logprobs=1,
        verbosity="low",
        extra_query={"foo": "bar"},
        extra_body={"foo": "bar"},
        extra_headers={"foo": "bar"},
        extra_args={"custom_param": "value", "another_param": 42},
        retry=ModelRetrySettings(
            max_retries=2,
            backoff=ModelRetryBackoffSettings(
                initial_delay=0.1,
                max_delay=1.0,
                multiplier=2.0,
                jitter=False,
            ),
        ),
    )

    # Verify that every single field is set to a non-None value
    for field in fields(model_settings):
        assert getattr(model_settings, field.name) is not None, (
            f"You must set the {field.name} field"
        )

    # Now, lets serialize the ModelSettings instance to a JSON string
    verify_serialization(model_settings)

def test_pydantic_serialization() -> None:
    """Tests whether ModelSettings can be serialized with Pydantic."""

    # First, lets create a ModelSettings instance
    model_settings = ModelSettings(
        temperature=0.5,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        tool_choice="auto",
        parallel_tool_calls=True,
        truncation="auto",
        max_tokens=100,
        reasoning=Reasoning(),
        metadata={"foo": "bar"},
        store=False,
        include_usage=False,
        top_logprobs=1,
        extra_query={"foo": "bar"},
        extra_body={"foo": "bar"},
        extra_headers={"foo": "bar"},
        extra_args={"custom_param": "value", "another_param": 42},
    )

    json = to_json(model_settings)
    deserialized = TypeAdapter(ModelSettings).validate_json(json)

    assert model_settings == deserialized


# --- tests/models/test_any_llm_model.py ---

def test_any_llm_provider_passes_api_override() -> None:
    pytest.importorskip(
        "any_llm",
        reason="`any-llm-sdk` is only available when the optional dependency is installed.",
    )
    from agents.extensions.models.any_llm_model import AnyLLMModel
    from agents.extensions.models.any_llm_provider import AnyLLMProvider

    provider = AnyLLMProvider(api="chat_completions")
    model = provider.get_model("openai/gpt-4.1-mini")

    assert isinstance(model, AnyLLMModel)
    assert model.api == "chat_completions"


# --- tests/models/test_kwargs_functionality.py ---

def test_litellm_get_retry_advice_uses_response_headers() -> None:
    """LiteLLM retry advice should expose OpenAI-compatible retry headers."""

    model = LitellmModel(model="test-model")
    error = RateLimitError(
        message="rate limited",
        llm_provider="openai",
        model="gpt-4o-mini",
        response=Response(
            status_code=429,
            headers=Headers({"x-should-retry": "true", "retry-after-ms": "250"}),
        ),
    )

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=False,
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.retry_after == 0.25

def test_litellm_get_retry_advice_keeps_stateful_transport_failures_ambiguous() -> None:
    model = LitellmModel(model="test-model")
    error = APIConnectionError(
        message="connection error",
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=False,
            previous_response_id="resp_prev",
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety is None


# --- tests/realtime/test_conversion_helpers.py ---

    def test_convert_interrupt(self):
        """Test converting interrupt parameters to conversation item truncate event."""
        current_item_id = "item_789"
        current_audio_content_index = 2
        elapsed_time_ms = 1500

        result = _ConversionHelper.convert_interrupt(
            current_item_id, current_audio_content_index, elapsed_time_ms
        )

        assert isinstance(result, ConversationItemTruncateEvent)
        assert result.type == "conversation.item.truncate"
        assert result.item_id == "item_789"
        assert result.content_index == 2
        assert result.audio_end_ms == 1500

    def test_convert_interrupt_zero_time(self):
        """Test converting interrupt with zero elapsed time."""
        result = _ConversionHelper.convert_interrupt("item_1", 0, 0)

        assert isinstance(result, ConversationItemTruncateEvent)
        assert result.type == "conversation.item.truncate"
        assert result.item_id == "item_1"
        assert result.content_index == 0
        assert result.audio_end_ms == 0

    def test_convert_interrupt_large_values(self):
        """Test converting interrupt with large values."""
        result = _ConversionHelper.convert_interrupt("item_xyz", 99, 999999)

        assert isinstance(result, ConversationItemTruncateEvent)
        assert result.type == "conversation.item.truncate"
        assert result.item_id == "item_xyz"
        assert result.content_index == 99
        assert result.audio_end_ms == 999999

    def test_convert_interrupt_empty_item_id(self):
        """Test converting interrupt with empty item ID."""
        result = _ConversionHelper.convert_interrupt("", 1, 100)

        assert isinstance(result, ConversationItemTruncateEvent)
        assert result.type == "conversation.item.truncate"
        assert result.item_id == ""
        assert result.content_index == 1
        assert result.audio_end_ms == 100


# --- tests/realtime/test_realtime_model_settings.py ---

async def test_collect_enabled_handoffs_filters_disabled() -> None:
    parent = RealtimeAgent(name="parent")
    disabled = realtime_handoff(
        RealtimeAgent(name="child_disabled"),
        is_enabled=lambda ctx, agent: False,
    )
    parent.handoffs = [disabled, RealtimeAgent(name="child_enabled")]

    enabled = await _collect_enabled_handoffs(parent, RunContextWrapper(None))

    assert len(enabled) == 1
    assert isinstance(enabled[0], Handoff)
    assert enabled[0].agent_name == "child_enabled"


# --- tests/test_agent_as_tool.py ---

async def test_agent_as_tool_is_enabled_bool():
    """Test that agent.as_tool() respects static boolean is_enabled parameter."""
    # Create a simple agent
    agent = Agent(
        name="test_agent",
        instructions="You are a test agent that says hello.",
    )

    # Create tool with is_enabled=False
    disabled_tool = agent.as_tool(
        tool_name="disabled_agent_tool",
        tool_description="A disabled agent tool",
        is_enabled=False,
    )

    # Create tool with is_enabled=True (default)
    enabled_tool = agent.as_tool(
        tool_name="enabled_agent_tool",
        tool_description="An enabled agent tool",
        is_enabled=True,
    )

    # Create another tool with default is_enabled (should be True)
    default_tool = agent.as_tool(
        tool_name="default_agent_tool",
        tool_description="A default agent tool",
    )

    # Create test agent that uses these tools
    orchestrator = Agent(
        name="orchestrator",
        instructions="You orchestrate other agents.",
        tools=[disabled_tool, enabled_tool, default_tool],
    )

    # Test with any context
    context = RunContextWrapper(BoolCtx(enable_tools=True))

    # Get all tools - should filter out the disabled one
    tools = await orchestrator.get_all_tools(context)
    tool_names = [tool.name for tool in tools]

    assert "enabled_agent_tool" in tool_names
    assert "default_agent_tool" in tool_names
    assert "disabled_agent_tool" not in tool_names

async def test_agent_as_tool_is_enabled_callable():
    """Test that agent.as_tool() respects callable is_enabled parameter."""
    # Create a simple agent
    agent = Agent(
        name="test_agent",
        instructions="You are a test agent that says hello.",
    )

    # Create tool with callable is_enabled
    async def cond_enabled(ctx: RunContextWrapper[BoolCtx], agent: AgentBase) -> bool:
        return ctx.context.enable_tools

    conditional_tool = agent.as_tool(
        tool_name="conditional_agent_tool",
        tool_description="A conditionally enabled agent tool",
        is_enabled=cond_enabled,
    )

    # Create tool with lambda is_enabled
    lambda_tool = agent.as_tool(
        tool_name="lambda_agent_tool",
        tool_description="A lambda enabled agent tool",
        is_enabled=lambda ctx, agent: ctx.context.enable_tools,
    )

    # Create test agent that uses these tools
    orchestrator = Agent(
        name="orchestrator",
        instructions="You orchestrate other agents.",
        tools=[conditional_tool, lambda_tool],
    )

    # Test with enable_tools=False
    context_disabled = RunContextWrapper(BoolCtx(enable_tools=False))
    tools_disabled = await orchestrator.get_all_tools(context_disabled)
    assert len(tools_disabled) == 0

    # Test with enable_tools=True
    context_enabled = RunContextWrapper(BoolCtx(enable_tools=True))
    tools_enabled = await orchestrator.get_all_tools(context_enabled)
    tool_names = [tool.name for tool in tools_enabled]

    assert len(tools_enabled) == 2
    assert "conditional_agent_tool" in tool_names
    assert "lambda_agent_tool" in tool_names

async def test_agent_as_tool_is_enabled_mixed():
    """Test agent.as_tool() with mixed enabled/disabled tools."""
    # Create a simple agent
    agent = Agent(
        name="test_agent",
        instructions="You are a test agent that says hello.",
    )

    # Create various tools with different is_enabled configurations
    always_enabled = agent.as_tool(
        tool_name="always_enabled",
        tool_description="Always enabled tool",
        is_enabled=True,
    )

    always_disabled = agent.as_tool(
        tool_name="always_disabled",
        tool_description="Always disabled tool",
        is_enabled=False,
    )

    conditionally_enabled = agent.as_tool(
        tool_name="conditionally_enabled",
        tool_description="Conditionally enabled tool",
        is_enabled=lambda ctx, agent: ctx.context.enable_tools,
    )

    default_enabled = agent.as_tool(
        tool_name="default_enabled",
        tool_description="Default enabled tool",
    )

    # Create test agent that uses these tools
    orchestrator = Agent(
        name="orchestrator",
        instructions="You orchestrate other agents.",
        tools=[always_enabled, always_disabled, conditionally_enabled, default_enabled],
    )

    # Test with enable_tools=False
    context_disabled = RunContextWrapper(BoolCtx(enable_tools=False))
    tools_disabled = await orchestrator.get_all_tools(context_disabled)
    tool_names_disabled = [tool.name for tool in tools_disabled]

    assert len(tools_disabled) == 2
    assert "always_enabled" in tool_names_disabled
    assert "default_enabled" in tool_names_disabled
    assert "always_disabled" not in tool_names_disabled
    assert "conditionally_enabled" not in tool_names_disabled

    # Test with enable_tools=True
    context_enabled = RunContextWrapper(BoolCtx(enable_tools=True))
    tools_enabled = await orchestrator.get_all_tools(context_enabled)
    tool_names_enabled = [tool.name for tool in tools_enabled]

    assert len(tools_enabled) == 3
    assert "always_enabled" in tool_names_enabled
    assert "default_enabled" in tool_names_enabled
    assert "conditionally_enabled" in tool_names_enabled
    assert "always_disabled" not in tool_names_enabled

async def test_agent_as_tool_is_enabled_preserves_other_params():
    """Test that is_enabled parameter doesn't interfere with other agent.as_tool() parameters."""
    # Create a simple agent
    agent = Agent(
        name="test_agent",
        instructions="You are a test agent that returns a greeting.",
    )

    # Custom output extractor
    async def custom_extractor(result):
        return f"CUSTOM: {result.new_items[-1].text if result.new_items else 'No output'}"

    # Create tool with all parameters including is_enabled
    tool = agent.as_tool(
        tool_name="custom_tool_name",
        tool_description="A custom tool with all parameters",
        custom_output_extractor=custom_extractor,
        is_enabled=True,
    )

    # Verify the tool was created with correct properties
    assert tool.name == "custom_tool_name"
    assert isinstance(tool, FunctionTool)
    assert tool.description == "A custom tool with all parameters"
    assert tool.is_enabled is True

    # Verify tool is included when enabled
    orchestrator = Agent(
        name="orchestrator",
        instructions="You orchestrate other agents.",
        tools=[tool],
    )

    context = RunContextWrapper(BoolCtx(enable_tools=True))
    tools = await orchestrator.get_all_tools(context)
    assert len(tools) == 1
    assert tools[0].name == "custom_tool_name"

async def test_agent_as_tool_rejects_invalid_builder_output() -> None:
    """Invalid builder output should surface as a tool error."""

    agent = Agent(name="invalid_builder_agent")

    def builder(_options):
        return 123

    tool = agent.as_tool(
        tool_name="invalid_builder_tool",
        tool_description="Invalid builder tool",
        input_builder=builder,
    )

    tool_context = ToolContext(
        context=None,
        tool_name="invalid_builder_tool",
        tool_call_id="call_invalid_builder",
        tool_arguments='{"input": "hi"}',
    )
    result = await tool.on_invoke_tool(tool_context, '{"input": "hi"}')

    assert "Agent tool called with invalid input" in result

async def test_replaced_agent_as_tool_invalid_input_uses_replaced_name() -> None:
    nested_agent = Agent(name="nested_agent")
    replaced_tool = dataclasses.replace(
        nested_agent.as_tool(
            tool_name="nested_agent_tool",
            tool_description="Nested agent tool",
            is_enabled=True,
            failure_error_function=None,
        ),
        name="replaced_nested_agent_tool",
    )

    with pytest.raises(
        ModelBehaviorError,
        match="Invalid JSON input for tool replaced_nested_agent_tool",
    ):
        await replaced_tool.on_invoke_tool(
            ToolContext(
                context=None,
                tool_name=replaced_tool.name,
                tool_call_id="call_1",
                tool_arguments="{}",
            ),
            "{}",
        )


# --- tests/test_agent_prompt.py ---

async def test_static_prompt_is_resolved_correctly():
    static_prompt: Prompt = {
        "id": "my_prompt",
        "version": "1",
        "variables": {"some_var": "some_value"},
    }

    agent = Agent(name="test", prompt=static_prompt)
    context_wrapper = RunContextWrapper(context=None)

    resolved = await agent.get_prompt(context_wrapper)

    assert resolved == {
        "id": "my_prompt",
        "version": "1",
        "variables": {"some_var": "some_value"},
    }

async def test_dynamic_prompt_is_resolved_correctly():
    dynamic_prompt_value: Prompt = {"id": "dyn_prompt", "version": "2"}

    def dynamic_prompt_fn(_data):
        return dynamic_prompt_value

    agent = Agent(name="test", prompt=dynamic_prompt_fn)
    context_wrapper = RunContextWrapper(context=None)

    resolved = await agent.get_prompt(context_wrapper)

    assert resolved == {"id": "dyn_prompt", "version": "2", "variables": None}

async def test_agent_prompt_with_default_model_omits_model_and_tools_parameters():
    called_kwargs: dict[str, object] = {}

    class DummyResponses:
        async def create(self, **kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return get_response_obj([get_text_message("done")])

    class DummyResponsesClient:
        def __init__(self):
            self.responses = DummyResponses()

    model = OpenAIResponsesModel(
        model="gpt-4.1",
        openai_client=DummyResponsesClient(),  # type: ignore[arg-type]
        model_is_explicit=False,
    )

    run_config = RunConfig(model_provider=_SingleModelProvider(model))
    agent = Agent(name="prompt-agent", prompt={"id": "pmpt_agent"})

    await Runner.run(agent, input="hi", run_config=run_config)

    expected_prompt = {"id": "pmpt_agent", "version": None, "variables": None}
    assert called_kwargs["prompt"] == expected_prompt
    assert called_kwargs["model"] is omit
    assert called_kwargs["tools"] is omit


# --- tests/test_agent_runner.py ---

def test_set_default_agent_runner_roundtrip():
    runner = AgentRunner()
    set_default_agent_runner(runner)
    assert get_default_agent_runner() is runner

    # Reset to ensure other tests are unaffected.
    set_default_agent_runner(None)
    assert isinstance(get_default_agent_runner(), AgentRunner)

def test_run_streamed_preserves_legacy_positional_previous_response_id():
    captured: dict[str, Any] = {}

    class DummyRunner:
        def run_streamed(self, starting_agent: Any, input: Any, **kwargs: Any):
            captured.update(kwargs)
            return object()

    original_runner = get_default_agent_runner()
    set_default_agent_runner(cast(Any, DummyRunner()))
    try:
        Runner.run_streamed(
            cast(Any, None),
            "hello",
            None,
            10,
            None,
            None,
            "resp-legacy",
        )
    finally:
        set_default_agent_runner(original_runner)

    assert captured["previous_response_id"] == "resp-legacy"
    assert captured["error_handlers"] is None

def test_run_config_defaults_nested_handoff_history_opt_in():
    assert RunConfig().nest_handoff_history is False

def test_normalize_resumed_input_drops_orphan_tool_search_calls():
    raw_input: list[TResponseInputItem] = [
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": "orphan_search",
                "arguments": {"query": "orphan"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": "paired_search",
                "arguments": {"query": "paired"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_output",
                "call_id": "paired_search",
                "execution": "server",
                "status": "completed",
                "tools": [],
            },
        ),
    ]

    normalized = normalize_resumed_input(raw_input)
    assert isinstance(normalized, list)
    call_ids = [
        cast(dict[str, Any], item).get("call_id")
        for item in normalized
        if isinstance(item, dict) and item.get("type") == "tool_search_call"
    ]
    assert "orphan_search" not in call_ids
    assert "paired_search" in call_ids

def test_normalize_resumed_input_preserves_hosted_tool_search_pair_without_call_ids():
    raw_input: list[TResponseInputItem] = [
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": None,
                "arguments": {"query": "paired"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_output",
                "call_id": None,
                "execution": "server",
                "status": "completed",
                "tools": [],
            },
        ),
    ]

    normalized = normalize_resumed_input(raw_input)
    assert isinstance(normalized, list)
    assert [cast(dict[str, Any], item)["type"] for item in normalized] == [
        "tool_search_call",
        "tool_search_output",
    ]

def test_normalize_resumed_input_matches_latest_anonymous_tool_search_call():
    raw_input: list[TResponseInputItem] = [
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": None,
                "arguments": {"query": "orphan"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": None,
                "arguments": {"query": "paired"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_output",
                "call_id": None,
                "execution": "server",
                "status": "completed",
                "tools": [],
            },
        ),
    ]

    normalized = normalize_resumed_input(raw_input)
    assert isinstance(normalized, list)
    assert [cast(dict[str, Any], item)["type"] for item in normalized] == [
        "tool_search_call",
        "tool_search_output",
    ]
    assert cast(dict[str, Any], normalized[0])["arguments"] == {"query": "paired"}

def test_fingerprint_input_item_returns_none_when_model_dump_fails():
    class _BrokenModelDump:
        def model_dump(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("model_dump failed")

    assert fingerprint_input_item(_BrokenModelDump()) is None

async def test_invalid_handoff_input_json_causes_error():
    agent = Agent(name="test")
    h = handoff(agent, input_type=Foo, on_handoff=lambda _ctx, _input: None)

    with pytest.raises(ModelBehaviorError):
        await h.on_invoke_handoff(
            RunContextWrapper(None),
            # Purposely ignoring the type error here to simulate invalid input
            None,  # type: ignore
        )

    with pytest.raises(ModelBehaviorError):
        await h.on_invoke_handoff(RunContextWrapper(None), "invalid")

async def test_save_result_to_session_preserves_function_outputs():
    session = SimpleListSession()
    original_item = cast(
        TResponseInputItem,
        {
            "type": "function_call_output",
            "call_id": "call_original",
            "output": "1",
        },
    )
    run_item_payload = {
        "type": "function_call_output",
        "call_id": "call_result",
        "output": "2",
    }
    dummy_run_item = _DummyRunItem(run_item_payload)

    await save_result_to_session(
        session,
        [original_item],
        [cast(RunItem, dummy_run_item)],
        None,
    )

    assert len(session.saved_items) == 2
    for saved in session.saved_items:
        saved_dict = cast(dict[str, Any], saved)
        assert saved_dict["type"] == "function_call_output"
        assert "output" in saved_dict

async def test_save_result_to_session_prefers_latest_duplicate_function_outputs():
    session = SimpleListSession()
    original_item = cast(
        TResponseInputItem,
        {
            "type": "function_call_output",
            "call_id": "call_duplicate",
            "output": "old-output",
        },
    )
    new_item_payload = {
        "type": "function_call_output",
        "call_id": "call_duplicate",
        "output": "new-output",
    }
    new_item = _DummyRunItem(new_item_payload)

    await save_result_to_session(
        session,
        [original_item],
        [cast(RunItem, new_item)],
        None,
    )

    duplicates = [
        cast(dict[str, Any], item)
        for item in session.saved_items
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") == "call_duplicate"
    ]
    assert len(duplicates) == 1
    assert duplicates[0]["output"] == "new-output"

async def test_execute_approved_tools_with_rejected_tool():
    """Test _execute_approved_tools handles rejected tools."""
    tool_called = False

    async def test_tool() -> str:
        nonlocal tool_called
        tool_called = True
        return "tool_result"

    tool = function_tool(test_tool, name_override="test_tool")
    _, agent = make_model_and_agent(tools=[tool])

    # Create a rejected tool call
    tool_call = get_function_tool_call("test_tool", "{}")
    assert isinstance(tool_call, ResponseFunctionToolCall)
    approval_item = ToolApprovalItem(agent=agent, raw_item=tool_call)

    generated_items = await run_execute_approved_tools(
        agent=agent,
        approval_item=approval_item,
        approve=False,
    )

    # Should add rejection message
    assert len(generated_items) == 1
    assert "not approved" in generated_items[0].output.lower()
    assert not tool_called  # Tool should not have been executed

async def test_execute_approved_tools_with_rejected_tool_uses_run_level_formatter():
    """Rejected tools should prefer RunConfig tool error formatter output."""

    async def test_tool() -> str:
        return "tool_result"

    tool = function_tool(test_tool, name_override="test_tool")
    _, agent = make_model_and_agent(tools=[tool])

    tool_call = get_function_tool_call("test_tool", "{}")
    assert isinstance(tool_call, ResponseFunctionToolCall)
    approval_item = ToolApprovalItem(agent=agent, raw_item=tool_call)

    generated_items = await run_execute_approved_tools(
        agent=agent,
        approval_item=approval_item,
        approve=False,
        run_config=RunConfig(
            tool_error_formatter=lambda args: f"run-level {args.tool_name} denied ({args.call_id})"
        ),
    )

    assert len(generated_items) == 1
    assert generated_items[0].output == "run-level test_tool denied (2)"

async def test_execute_approved_tools_with_rejected_tool_prefers_explicit_message():
    """Rejected tools should prefer explicit rejection messages over the formatter."""

    async def test_tool() -> str:
        return "tool_result"

    tool = function_tool(test_tool, name_override="test_tool")
    _, agent = make_model_and_agent(tools=[tool])

    tool_call = get_function_tool_call("test_tool", "{}")
    assert isinstance(tool_call, ResponseFunctionToolCall)
    approval_item = ToolApprovalItem(agent=agent, raw_item=tool_call)

    generated_items = await run_execute_approved_tools(
        agent=agent,
        approval_item=approval_item,
        approve=False,
        run_config=RunConfig(
            tool_error_formatter=lambda args: f"run-level {args.tool_name} denied ({args.call_id})"
        ),
        mutate_state=lambda state, item: state.reject(
            item, rejection_message="explicit rejection message"
        ),
    )

    assert len(generated_items) == 1
    assert generated_items[0].output == "explicit rejection message"

async def test_execute_approved_tools_with_rejected_deferred_tool_uses_display_name():
    """Rejected deferred tools should collapse synthetic namespaces in formatter output."""

    async def get_weather() -> str:
        return "sunny"

    tool = function_tool(get_weather, name_override="get_weather", defer_loading=True)
    _, agent = make_model_and_agent(tools=[tool])

    tool_call = get_function_tool_call("get_weather", "{}", namespace="get_weather")
    assert isinstance(tool_call, ResponseFunctionToolCall)
    approval_item = ToolApprovalItem(
        agent=agent,
        raw_item=tool_call,
        tool_name="get_weather",
        tool_namespace="get_weather",
    )

    generated_items = await run_execute_approved_tools(
        agent=agent,
        approval_item=approval_item,
        approve=False,
        run_config=RunConfig(
            tool_error_formatter=lambda args: f"run-level {args.tool_name} denied ({args.call_id})"
        ),
    )

    assert len(generated_items) == 1
    assert generated_items[0].output == "run-level get_weather denied (2)"

async def test_execute_approved_tools_with_rejected_tool_formatter_none_uses_default():
    """Rejected tools should use default message when formatter returns None."""

    async def test_tool() -> str:
        return "tool_result"

    tool = function_tool(test_tool, name_override="test_tool")
    _, agent = make_model_and_agent(tools=[tool])

    tool_call = get_function_tool_call("test_tool", "{}")
    assert isinstance(tool_call, ResponseFunctionToolCall)
    approval_item = ToolApprovalItem(agent=agent, raw_item=tool_call)

    generated_items = await run_execute_approved_tools(
        agent=agent,
        approval_item=approval_item,
        approve=False,
        run_config=RunConfig(tool_error_formatter=lambda _args: None),
    )

    assert len(generated_items) == 1
    assert generated_items[0].output == "Tool execution was not approved."

async def test_execute_approved_tools_with_unclear_status():
    """Test _execute_approved_tools handles unclear approval status."""
    tool_called = False

    async def test_tool() -> str:
        nonlocal tool_called
        tool_called = True
        return "tool_result"

    tool = function_tool(test_tool, name_override="test_tool")
    _, agent = make_model_and_agent(tools=[tool])

    # Create a tool call with unclear status (neither approved nor rejected)
    tool_call = get_function_tool_call("test_tool", "{}")
    assert isinstance(tool_call, ResponseFunctionToolCall)
    approval_item = ToolApprovalItem(agent=agent, raw_item=tool_call)

    generated_items = await run_execute_approved_tools(
        agent=agent,
        approval_item=approval_item,
        approve=None,
    )

    # Should add unclear status message
    assert len(generated_items) == 1
    assert "unclear" in generated_items[0].output.lower()
    assert not tool_called  # Tool should not have been executed

async def test_execute_approved_tools_with_missing_tool():
    """Test _execute_approved_tools handles missing tools."""
    _, agent = make_model_and_agent()
    # Agent has no tools

    # Create an approved tool call for a tool that doesn't exist
    tool_call = get_function_tool_call("nonexistent_tool", "{}")
    assert isinstance(tool_call, ResponseFunctionToolCall)
    approval_item = ToolApprovalItem(agent=agent, raw_item=tool_call)

    generated_items = await run_execute_approved_tools(
        agent=agent,
        approval_item=approval_item,
        approve=True,
    )

    # Should add error message about tool not found
    assert len(generated_items) == 1
    assert isinstance(generated_items[0], ToolCallOutputItem)
    assert "not found" in generated_items[0].output.lower()

async def test_execute_approved_tools_with_missing_call_id():
    """Test _execute_approved_tools handles tool approvals without call IDs."""
    _, agent = make_model_and_agent()
    tool_call = {"type": "function_call", "name": "test_tool"}
    approval_item = ToolApprovalItem(agent=agent, raw_item=tool_call)

    generated_items = await run_execute_approved_tools(
        agent=agent,
        approval_item=approval_item,
        approve=True,
    )

    assert len(generated_items) == 1
    assert isinstance(generated_items[0], ToolCallOutputItem)
    assert "missing call id" in generated_items[0].output.lower()

async def test_execute_approved_tools_with_invalid_raw_item_type():
    """Test _execute_approved_tools handles approvals with unsupported raw_item types."""

    async def test_tool() -> str:
        return "tool_result"

    tool = function_tool(test_tool, name_override="test_tool")
    _, agent = make_model_and_agent(tools=[tool])
    tool_call = {"type": "function_call", "name": "test_tool", "call_id": "call-1"}
    approval_item = ToolApprovalItem(agent=agent, raw_item=tool_call)

    generated_items = await run_execute_approved_tools(
        agent=agent,
        approval_item=approval_item,
        approve=True,
    )

    assert len(generated_items) == 1
    assert isinstance(generated_items[0], ToolCallOutputItem)
    assert "invalid raw_item type" in generated_items[0].output.lower()

async def test_execute_approved_tools_instance_method():
    """Ensure execute_approved_tools runs approved tools as expected."""
    tool_called = False

    async def test_tool() -> str:
        nonlocal tool_called
        tool_called = True
        return "tool_result"

    tool = function_tool(test_tool, name_override="test_tool")
    _, agent = make_model_and_agent(tools=[tool])

    tool_call = get_function_tool_call("test_tool", json.dumps({}))
    assert isinstance(tool_call, ResponseFunctionToolCall)

    approval_item = ToolApprovalItem(agent=agent, raw_item=tool_call)

    generated_items = await run_execute_approved_tools(
        agent=agent,
        approval_item=approval_item,
        approve=True,
    )

    # Tool should have been called
    assert tool_called is True
    assert len(generated_items) == 1
    assert isinstance(generated_items[0], ToolCallOutputItem)
    assert generated_items[0].output == "tool_result"

async def test_execute_approved_tools_timeout_returns_error_as_result() -> None:
    async def slow_tool() -> str:
        await asyncio.sleep(0.2)
        return "tool_result"

    tool = function_tool(slow_tool, name_override="test_tool", timeout=0.01)
    _, agent = make_model_and_agent(tools=[tool])

    tool_call = get_function_tool_call("test_tool", json.dumps({}))
    assert isinstance(tool_call, ResponseFunctionToolCall)

    approval_item = ToolApprovalItem(agent=agent, raw_item=tool_call)
    generated_items = await run_execute_approved_tools(
        agent=agent,
        approval_item=approval_item,
        approve=True,
    )

    assert len(generated_items) == 1
    assert isinstance(generated_items[0], ToolCallOutputItem)
    assert "timed out" in generated_items[0].output.lower()

async def test_execute_approved_tools_timeout_can_raise_exception() -> None:
    async def slow_tool() -> str:
        await asyncio.sleep(0.2)
        return "tool_result"

    tool = function_tool(
        slow_tool,
        name_override="test_tool",
        timeout=0.01,
        timeout_behavior="raise_exception",
    )
    _, agent = make_model_and_agent(tools=[tool])

    tool_call = get_function_tool_call("test_tool", json.dumps({}))
    assert isinstance(tool_call, ResponseFunctionToolCall)

    approval_item = ToolApprovalItem(agent=agent, raw_item=tool_call)
    with pytest.raises(ToolTimeoutError, match="timed out"):
        await run_execute_approved_tools(
            agent=agent,
            approval_item=approval_item,
            approve=True,
        )


# --- tests/test_agent_runner_streamed.py ---

async def test_stream_step_items_to_queue_handles_tool_approval_item():
    """Test that stream_step_items_to_queue handles ToolApprovalItem."""
    _, agent = make_model_and_agent(name="test")
    tool_call = get_function_tool_call("test_tool", "{}")
    assert isinstance(tool_call, ResponseFunctionToolCall)
    approval_item = ToolApprovalItem(agent=agent, raw_item=tool_call)

    queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel] = asyncio.Queue()

    # ToolApprovalItem should not be streamed
    run_loop.stream_step_items_to_queue([approval_item], queue)

    # Queue should be empty since ToolApprovalItem is not streamed
    assert queue.empty()

async def test_streaming_hitl_resume_with_approved_tools():
    """Test resuming streaming run from RunState with approved tools executes them."""
    tool_called = False

    async def test_tool() -> str:
        nonlocal tool_called
        tool_called = True
        return "tool_result"

    # Create a tool that requires approval
    tool = function_tool(test_tool, name_override="test_tool", needs_approval=True)
    model, agent = make_model_and_agent(name="test", tools=[tool])

    # First run - tool call that requires approval
    queue_function_call_and_text(
        model,
        get_function_tool_call("test_tool", json.dumps({})),
        followup=[get_text_message("done")],
    )

    first = Runner.run_streamed(agent, input="Use test_tool")
    await consume_stream(first)

    # Resume from state - should execute approved tool
    result2 = await resume_streamed_after_first_approval(agent, first)

    # Tool should have been called
    assert tool_called is True
    assert result2.final_output == "done"

async def test_streaming_resume_with_session_does_not_duplicate_items():
    """Ensure session persistence does not duplicate tool items after streaming resume."""

    async def test_tool() -> str:
        return "tool_result"

    tool = function_tool(test_tool, name_override="test_tool", needs_approval=True)
    model, agent = make_model_and_agent(name="test", tools=[tool])
    session = SimpleListSession()

    queue_function_call_and_text(
        model,
        get_function_tool_call("test_tool", json.dumps({}), call_id="call-resume"),
        followup=[get_text_message("done")],
    )

    first = Runner.run_streamed(agent, input="Use test_tool", session=session)
    await consume_stream(first)
    assert first.interruptions

    state = first.to_state()
    state.approve(first.interruptions[0])

    resumed = Runner.run_streamed(agent, state, session=session)
    await consume_stream(resumed)
    assert resumed.final_output == "done"

    saved_items = await session.get_items()
    call_count = sum(
        1
        for item in saved_items
        if isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("call_id") == "call-resume"
    )
    output_count = sum(
        1
        for item in saved_items
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") == "call-resume"
    )

    assert call_count == 1
    assert output_count == 1

async def test_streaming_resume_persists_tool_outputs_on_run_again():
    """Approved tool outputs should be persisted before streaming resumes the next turn."""

    async def test_tool() -> str:
        return "tool_result"

    tool = function_tool(test_tool, name_override="test_tool", needs_approval=True)
    model, agent = make_model_and_agent(name="test", tools=[tool])
    session = SimpleListSession()

    queue_function_call_and_text(
        model,
        get_function_tool_call("test_tool", json.dumps({}), call_id="call-resume"),
        followup=[get_text_message("done")],
    )

    first = Runner.run_streamed(agent, input="Use test_tool", session=session)
    await consume_stream(first)

    assert first.interruptions
    state = first.to_state()
    state.approve(first.interruptions[0])

    resumed = Runner.run_streamed(agent, state, session=session)
    await consume_stream(resumed)

    saved_items = await session.get_items()
    assert any(
        isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") == "call-resume"
        for item in saved_items
    ), "approved tool outputs should be persisted on resume"

async def test_streaming_hitl_resume_enforces_max_turns():
    """Test that streamed resumes advance turn counts for max_turns enforcement."""

    async def test_tool() -> str:
        return "tool_result"

    tool = function_tool(test_tool, name_override="test_tool", needs_approval=True)
    model, agent = make_model_and_agent(name="test", tools=[tool])

    queue_function_call_and_text(
        model,
        get_function_tool_call("test_tool", json.dumps({})),
        followup=[get_text_message("done")],
    )

    first = Runner.run_streamed(agent, input="Use test_tool", max_turns=1)
    await consume_stream(first)

    assert first.interruptions
    state = first.to_state()
    state.approve(first.interruptions[0])

    resumed = Runner.run_streamed(agent, state)
    with pytest.raises(MaxTurnsExceeded):
        async for _ in resumed.stream_events():
            pass

async def test_streaming_max_turns_emits_pending_tool_output_events() -> None:
    async def test_tool() -> str:
        return "tool_result"

    tool = function_tool(test_tool, name_override="test_tool")
    model, agent = make_model_and_agent(name="test", tools=[tool])

    queue_function_call_and_text(
        model,
        get_function_tool_call("test_tool", json.dumps({})),
        followup=[get_text_message("done")],
    )

    result = Runner.run_streamed(agent, input="Use test_tool", max_turns=1)
    streamed_item_types: list[str] = []

    with pytest.raises(MaxTurnsExceeded):
        async for event in result.stream_events():
            if event.type == "run_item_stream_event":
                streamed_item_types.append(event.item.type)

    assert "tool_call_item" in streamed_item_types
    assert "tool_call_output_item" in streamed_item_types

async def test_streaming_non_max_turns_exception_does_not_emit_queued_events() -> None:
    model, agent = make_model_and_agent(name="test")
    model.set_next_output([get_text_message("done")])

    result = Runner.run_streamed(agent, input="hello")
    result.cancel()
    await asyncio.sleep(0)

    while not result._event_queue.empty():
        result._event_queue.get_nowait()
        result._event_queue.task_done()

    result._stored_exception = RuntimeError("guardrail-triggered")
    result._event_queue.put_nowait(AgentUpdatedStreamEvent(new_agent=agent))

    streamed_events: list[StreamEvent] = []
    with pytest.raises(RuntimeError, match="guardrail-triggered"):
        async for event in result.stream_events():
            streamed_events.append(event)

    assert streamed_events == []

async def test_streaming_hitl_server_conversation_tracker_priming():
    """Test that resuming streaming run from RunState primes server conversation tracker."""
    model, agent = make_model_and_agent(name="test")

    # First run with conversation_id
    model.set_next_output([get_text_message("First response")])
    result1 = Runner.run_streamed(
        agent, input="test", conversation_id="conv123", previous_response_id="resp123"
    )
    await consume_stream(result1)

    # Create state from result
    state = result1.to_state()

    # Resume with same conversation_id - should not duplicate messages
    model.set_next_output([get_text_message("Second response")])
    result2 = Runner.run_streamed(
        agent, state, conversation_id="conv123", previous_response_id="resp123"
    )
    await consume_stream(result2)

    # Should complete successfully without message duplication
    assert result2.final_output == "Second response"
    assert len(result2.new_items) >= 1


# --- tests/test_agent_tool_state.py ---

def test_agent_tool_run_result_supports_signature_fallback_across_instances() -> None:
    original_call = _function_tool_call("lookup_account", "{}", call_id="call-1")
    restored_call = _function_tool_call("lookup_account", "{}", call_id="call-1")
    run_result = cast(Any, object())

    tool_state.record_agent_tool_run_result(original_call, run_result, scope_id="scope-1")

    assert tool_state.peek_agent_tool_run_result(restored_call, scope_id="scope-1") is run_result
    assert tool_state.consume_agent_tool_run_result(restored_call, scope_id="scope-1") is run_result
    assert tool_state.peek_agent_tool_run_result(original_call, scope_id="scope-1") is None
    assert tool_state._agent_tool_run_results_by_signature == {}

def test_agent_tool_run_result_returns_none_for_ambiguous_signature_matches() -> None:
    first_call = _function_tool_call("lookup_account", "{}", call_id="call-1")
    second_call = _function_tool_call("lookup_account", "{}", call_id="call-1")
    restored_call = _function_tool_call("lookup_account", "{}", call_id="call-1")
    first_result = cast(Any, object())
    second_result = cast(Any, object())

    tool_state.record_agent_tool_run_result(first_call, first_result, scope_id="scope-1")
    tool_state.record_agent_tool_run_result(second_call, second_result, scope_id="scope-1")

    assert tool_state.peek_agent_tool_run_result(restored_call, scope_id="scope-1") is None
    assert tool_state.consume_agent_tool_run_result(restored_call, scope_id="scope-1") is None

    tool_state.drop_agent_tool_run_result(restored_call, scope_id="scope-1")

    assert tool_state.peek_agent_tool_run_result(first_call, scope_id="scope-1") is first_result
    assert tool_state.peek_agent_tool_run_result(second_call, scope_id="scope-1") is second_result
    assert tool_state.peek_agent_tool_run_result(restored_call, scope_id="other-scope") is None

def test_agent_tool_run_result_is_dropped_when_tool_call_is_collected() -> None:
    tool_call = _function_tool_call("lookup_account", "{}", call_id="call-1")
    tool_call_ref = weakref.ref(tool_call)
    tool_call_obj_id = id(tool_call)

    tool_state.record_agent_tool_run_result(tool_call, cast(Any, object()), scope_id="scope-1")

    del tool_call
    gc.collect()

    assert tool_call_ref() is None
    assert tool_call_obj_id not in tool_state._agent_tool_run_results_by_obj
    assert tool_call_obj_id not in tool_state._agent_tool_run_result_signature_by_obj
    assert tool_call_obj_id not in tool_state._agent_tool_call_refs_by_obj


# --- tests/test_anthropic_thinking_blocks.py ---

def test_anthropic_thinking_blocks_with_tool_calls():
    """
    Test for models with extended thinking and interleaved thinking with tool calls.

    This test verifies the Anthropic's API's requirements for thinking blocks
    to be the first content in assistant messages when reasoning is enabled and tool
    calls are present.
    """
    # Create a message with reasoning, thinking blocks and tool calls
    message = InternalChatCompletionMessage(
        role="assistant",
        content="I'll check the weather for you.",
        reasoning_content="The user wants weather information, I need to call the weather function",
        thinking_blocks=[
            {
                "type": "thinking",
                "thinking": (
                    "The user is asking about weather. "
                    "Let me use the weather tool to get this information."
                ),
                "signature": "TestSignature123",
            },
            {
                "type": "thinking",
                "thinking": ("We should use the city Tokyo as the city."),
                "signature": "TestSignature456",
            },
        ],
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_123",
                type="function",
                function=Function(name="get_weather", arguments='{"city": "Tokyo"}'),
            )
        ],
    )

    # Step 1: Convert message to output items
    output_items = Converter.message_to_output_items(message)

    # Verify reasoning item exists and contains thinking blocks
    reasoning_items = [
        item for item in output_items if hasattr(item, "type") and item.type == "reasoning"
    ]
    assert len(reasoning_items) == 1, "Should have exactly two reasoning items"

    reasoning_item = reasoning_items[0]

    # Verify thinking text is stored in content
    assert hasattr(reasoning_item, "content") and reasoning_item.content, (
        "Reasoning item should have content"
    )
    assert reasoning_item.content[0].type == "reasoning_text", (
        "Content should be reasoning_text type"
    )

    # Verify signature is stored in encrypted_content
    assert hasattr(reasoning_item, "encrypted_content"), (
        "Reasoning item should have encrypted_content"
    )
    assert reasoning_item.encrypted_content == "TestSignature123\nTestSignature456", (
        "Signature should be preserved"
    )

    # Verify tool calls are present
    tool_call_items = [
        item for item in output_items if hasattr(item, "type") and item.type == "function_call"
    ]
    assert len(tool_call_items) == 1, "Should have exactly one tool call"

    # Step 2: Convert output items back to messages
    # Convert items to dicts for the converter (simulating serialization/deserialization)
    items_as_dicts: list[dict[str, Any]] = []
    for item in output_items:
        if hasattr(item, "model_dump"):
            items_as_dicts.append(item.model_dump())
        else:
            items_as_dicts.append(cast(dict[str, Any], item))

    messages = Converter.items_to_messages(
        items_as_dicts,  # type: ignore[arg-type]
        model="anthropic/claude-4-opus",
        preserve_thinking_blocks=True,
    )

    # Find the assistant message with tool calls
    assistant_messages = [
        msg for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls")
    ]
    assert len(assistant_messages) == 1, "Should have exactly one assistant message with tool calls"

    assistant_msg = assistant_messages[0]

    # Content must start with thinking blocks, not text
    content = assistant_msg.get("content")
    assert content is not None, "Assistant message should have content"

    assert isinstance(content, list) and len(content) > 0, (
        "Assistant message content should be a non-empty list"
    )

    first_content = content[0]
    assert first_content.get("type") == "thinking", (
        f"First content must be 'thinking' type for Anthropic compatibility, "
        f"but got '{first_content.get('type')}'"
    )
    expected_thinking = (
        "The user is asking about weather. Let me use the weather tool to get this information."
    )
    assert first_content.get("thinking") == expected_thinking, (
        "Thinking content should be preserved"
    )
    # Signature should also be preserved
    assert first_content.get("signature") == "TestSignature123", (
        "Signature should be preserved in thinking block"
    )

    second_content = content[1]
    assert second_content.get("type") == "thinking", (
        f"Second content must be 'thinking' type for Anthropic compatibility, "
        f"but got '{second_content.get('type')}'"
    )
    expected_thinking = "We should use the city Tokyo as the city."
    assert second_content.get("thinking") == expected_thinking, (
        "Thinking content should be preserved"
    )
    # Signature should also be preserved
    assert second_content.get("signature") == "TestSignature456", (
        "Signature should be preserved in thinking block"
    )

    last_content = content[2]
    assert last_content.get("type") == "text", (
        f"First content must be 'text' type but got '{last_content.get('type')}'"
    )
    expected_text = "I'll check the weather for you."
    assert last_content.get("text") == expected_text, "Content text should be preserved"

    # Verify tool calls are preserved
    tool_calls = assistant_msg.get("tool_calls", [])
    assert len(cast(list[Any], tool_calls)) == 1, "Tool calls should be preserved"
    assert cast(list[Any], tool_calls)[0]["function"]["name"] == "get_weather"

def test_items_to_messages_preserves_positional_bool_arguments():
    """
    Preserve positional compatibility for the released items_to_messages signature.
    """
    message = InternalChatCompletionMessage(
        role="assistant",
        content="I'll check the weather for you.",
        reasoning_content="The user wants weather information, I need to call the weather function",
        thinking_blocks=[
            {
                "type": "thinking",
                "thinking": (
                    "The user is asking about weather. "
                    "Let me use the weather tool to get this information."
                ),
                "signature": "TestSignature123",
            }
        ],
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_123",
                type="function",
                function=Function(name="get_weather", arguments='{"city": "Tokyo"}'),
            )
        ],
    )

    output_items = Converter.message_to_output_items(message)
    items_as_dicts: list[dict[str, Any]] = []
    for item in output_items:
        if hasattr(item, "model_dump"):
            items_as_dicts.append(item.model_dump())
        else:
            items_as_dicts.append(cast(dict[str, Any], item))

    messages = Converter.items_to_messages(
        items_as_dicts,  # type: ignore[arg-type]
        "anthropic/claude-4-opus",
        True,
        True,
    )

    assistant_messages = [
        msg for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls")
    ]
    assert len(assistant_messages) == 1, "Should have exactly one assistant message with tool calls"

    assistant_msg = assistant_messages[0]
    content = assistant_msg.get("content")
    assert isinstance(content, list) and len(content) > 0, (
        "Positional bool arguments should still preserve thinking blocks"
    )
    assert content[0].get("type") == "thinking", (
        "The third positional argument must continue to map to preserve_thinking_blocks"
    )

def test_anthropic_thinking_blocks_without_tool_calls():
    """
    Test for models with extended thinking WITHOUT tool calls.

    This test verifies that thinking blocks are properly attached to assistant
    messages even when there are no tool calls (fixes issue #2195).
    """
    # Create a message with reasoning and thinking blocks but NO tool calls
    message = InternalChatCompletionMessage(
        role="assistant",
        content="The weather in Paris is sunny with a temperature of 22°C.",
        reasoning_content="The user wants to know about the weather in Paris.",
        thinking_blocks=[
            {
                "type": "thinking",
                "thinking": "Let me think about the weather in Paris.",
                "signature": "TestSignatureNoTools123",
            }
        ],
        tool_calls=None,  # No tool calls
    )

    # Step 1: Convert message to output items
    output_items = Converter.message_to_output_items(message)

    # Verify reasoning item exists and contains thinking blocks
    reasoning_items = [
        item for item in output_items if hasattr(item, "type") and item.type == "reasoning"
    ]
    assert len(reasoning_items) == 1, "Should have exactly one reasoning item"

    reasoning_item = reasoning_items[0]

    # Verify thinking text is stored in content
    assert hasattr(reasoning_item, "content") and reasoning_item.content, (
        "Reasoning item should have content"
    )
    assert reasoning_item.content[0].type == "reasoning_text", (
        "Content should be reasoning_text type"
    )
    assert reasoning_item.content[0].text == "Let me think about the weather in Paris.", (
        "Thinking text should be preserved"
    )

    # Verify signature is stored in encrypted_content
    assert hasattr(reasoning_item, "encrypted_content"), (
        "Reasoning item should have encrypted_content"
    )
    assert reasoning_item.encrypted_content == "TestSignatureNoTools123", (
        "Signature should be preserved"
    )

    # Verify message item exists
    message_items = [
        item for item in output_items if hasattr(item, "type") and item.type == "message"
    ]
    assert len(message_items) == 1, "Should have exactly one message item"

    # Step 2: Convert output items back to messages with preserve_thinking_blocks=True
    items_as_dicts: list[dict[str, Any]] = []
    for item in output_items:
        if hasattr(item, "model_dump"):
            items_as_dicts.append(item.model_dump())
        else:
            items_as_dicts.append(cast(dict[str, Any], item))

    messages = Converter.items_to_messages(
        items_as_dicts,  # type: ignore[arg-type]
        model="anthropic/claude-4-opus",
        preserve_thinking_blocks=True,
    )

    # Should have one assistant message
    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    assert len(assistant_messages) == 1, "Should have exactly one assistant message"

    assistant_msg = assistant_messages[0]

    # Content must start with thinking blocks even WITHOUT tool calls
    content = assistant_msg.get("content")
    assert content is not None, "Assistant message should have content"
    assert isinstance(content, list), (
        f"Assistant message content should be a list when thinking blocks are present, "
        f"but got {type(content)}"
    )
    assert len(content) >= 2, (
        f"Assistant message should have at least 2 content items "
        f"(thinking + text), got {len(content)}"
    )

    # First content should be thinking block
    first_content = content[0]
    assert first_content.get("type") == "thinking", (
        f"First content must be 'thinking' type for Anthropic compatibility, "
        f"but got '{first_content.get('type')}'"
    )
    assert first_content.get("thinking") == "Let me think about the weather in Paris.", (
        "Thinking content should be preserved"
    )
    assert first_content.get("signature") == "TestSignatureNoTools123", (
        "Signature should be preserved in thinking block"
    )

    # Second content should be text
    second_content = content[1]
    assert second_content.get("type") == "text", (
        f"Second content must be 'text' type, but got '{second_content.get('type')}'"
    )
    assert (
        second_content.get("text") == "The weather in Paris is sunny with a temperature of 22°C."
    ), "Text content should be preserved"


# --- tests/test_computer_action.py ---

async def test_get_screenshot_sync_executes_action_and_takes_screenshot(
    action: Any, expected_call: tuple[str, tuple[Any, ...]]
) -> None:
    """For each action type, assert that the corresponding computer method is invoked
    and that a screenshot is taken and returned."""
    computer = LoggingComputer(screenshot_return="synthetic")
    tool_call = ResponseComputerToolCall(
        id="c1",
        type="computer_call",
        action=action,
        call_id="c1",
        pending_safety_checks=[],
        status="completed",
    )
    screenshot_output = await ComputerAction._execute_action_and_capture(computer, tool_call)
    if isinstance(action, ActionScreenshot):
        assert computer.calls == [("screenshot", ())]
    else:
        assert computer.calls == [expected_call, ("screenshot", ())]
    assert screenshot_output == "synthetic"

async def test_get_screenshot_async_executes_action_and_takes_screenshot(
    action: Any, expected_call: tuple[str, tuple[Any, ...]]
) -> None:
    """For each action type on an `AsyncComputer`, the corresponding coroutine should be awaited
    and a screenshot taken."""
    computer = LoggingAsyncComputer(screenshot_return="async_return")
    assert computer.environment == "mac"
    assert computer.dimensions == (800, 600)
    tool_call = ResponseComputerToolCall(
        id="c2",
        type="computer_call",
        action=action,
        call_id="c2",
        pending_safety_checks=[],
        status="completed",
    )
    screenshot_output = await ComputerAction._execute_action_and_capture(computer, tool_call)
    if isinstance(action, ActionScreenshot):
        assert computer.calls == [("screenshot", ())]
    else:
        assert computer.calls == [expected_call, ("screenshot", ())]
    assert screenshot_output == "async_return"

async def test_get_screenshot_executes_batched_actions_in_order() -> None:
    computer = LoggingComputer(screenshot_return="batched")
    tool_call = ResponseComputerToolCall(
        id="c3",
        type="computer_call",
        actions=[
            BatchedClick(type="click", x=11, y=12, button="left"),
            BatchedType(type="type", text="hello"),
        ],
        call_id="c3",
        pending_safety_checks=[],
        status="completed",
    )

    screenshot_output = await ComputerAction._execute_action_and_capture(computer, tool_call)

    assert computer.calls == [
        ("click", (11, 12, "left")),
        ("type", ("hello",)),
        ("screenshot", ()),
    ]
    assert screenshot_output == "batched"

async def test_get_screenshot_reuses_terminal_batched_screenshot() -> None:
    computer = LoggingComputer(screenshot_return="captured")
    tool_call = ResponseComputerToolCall(
        id="c4",
        type="computer_call",
        actions=[BatchedScreenshot(type="screenshot")],
        call_id="c4",
        pending_safety_checks=[],
        status="completed",
    )

    screenshot_output = await ComputerAction._execute_action_and_capture(computer, tool_call)

    assert computer.calls == [("screenshot", ())]
    assert screenshot_output == "captured"

async def test_execute_invokes_hooks_and_returns_tool_call_output() -> None:
    # ComputerAction.execute should invoke lifecycle hooks and return a proper ToolCallOutputItem.
    computer = LoggingComputer(screenshot_return="xyz")
    comptool = ComputerTool(computer=computer)
    # Create a dummy click action to trigger a click and screenshot.
    action = ActionClick(type="click", x=1, y=2, button="left")
    tool_call = ResponseComputerToolCall(
        id="tool123",
        type="computer_call",
        action=action,
        call_id="tool123",
        pending_safety_checks=[],
        status="completed",
    )
    tool_call.call_id = "tool123"

    # Wrap tool call in ToolRunComputerAction
    tool_run = ToolRunComputerAction(tool_call=tool_call, computer_tool=comptool)
    # Setup agent and hooks.
    agent = Agent(name="test_agent", tools=[comptool])
    # Attach per-agent hooks as well as global run hooks.
    agent_hooks = LoggingAgentHooks()
    agent.hooks = agent_hooks
    run_hooks = LoggingRunHooks()
    context_wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)
    # Execute the computer action.
    output_item = await ComputerAction.execute(
        agent=agent,
        action=tool_run,
        hooks=run_hooks,
        context_wrapper=context_wrapper,
        config=RunConfig(),
    )
    # Both global and per-agent hooks should have been called once.
    assert len(run_hooks.started) == 1 and len(agent_hooks.started) == 1
    assert len(run_hooks.ended) == 1 and len(agent_hooks.ended) == 1
    # The hook invocations should refer to our agent and tool.
    assert run_hooks.started[0][0] is agent
    assert run_hooks.ended[0][0] is agent
    assert run_hooks.started[0][1] is comptool
    assert run_hooks.ended[0][1] is comptool
    # The result passed to on_tool_end should be the raw screenshot string.
    assert run_hooks.ended[0][2] == "xyz"
    assert agent_hooks.ended[0][2] == "xyz"
    # The computer should have performed a click then a screenshot.
    assert computer.calls == [("click", (1, 2, "left")), ("screenshot", ())]
    # The returned item should include the agent, output string, and a ComputerCallOutput.
    assert output_item.agent is agent
    assert isinstance(output_item, ToolCallOutputItem)
    assert output_item.output == "data:image/png;base64,xyz"
    raw = cast(dict[str, Any], output_item.raw_item)
    # Raw item is a dict-like mapping with expected output fields.
    assert raw["type"] == "computer_call_output"
    assert raw["output"]["type"] == "computer_screenshot"
    assert "image_url" in raw["output"]
    assert raw["output"]["image_url"].endswith("xyz")

async def test_execute_emits_function_span() -> None:
    computer = LoggingComputer(screenshot_return="trace_img")
    comptool = ComputerTool(computer=computer)
    tool_call = ResponseComputerToolCall(
        id="tool_trace",
        type="computer_call",
        action=ActionScreenshot(type="screenshot"),
        call_id="tool_trace",
        pending_safety_checks=[],
        status="completed",
    )
    tool_run = ToolRunComputerAction(tool_call=tool_call, computer_tool=comptool)
    agent = Agent(name="test_agent_trace", tools=[comptool])

    set_tracing_disabled(False)
    with trace("computer-span-test"):
        result = await ComputerAction.execute(
            agent=agent,
            action=tool_run,
            hooks=RunHooks[Any](),
            context_wrapper=RunContextWrapper(context=None),
            config=RunConfig(),
        )

    assert isinstance(result, ToolCallOutputItem)
    assert ComputerAction.TRACE_TOOL_NAME == "computer"
    function_span = _get_function_span(ComputerAction.TRACE_TOOL_NAME)
    span_data = cast(dict[str, Any], function_span["span_data"])
    assert span_data.get("input") is not None
    assert cast(str, span_data.get("output", "")).startswith("data:image/png;base64,")

async def test_execute_emits_batched_actions_in_function_span() -> None:
    computer = LoggingComputer(screenshot_return="trace_img")
    comptool = ComputerTool(computer=computer)
    tool_call = ResponseComputerToolCall(
        id="tool_trace_batch",
        type="computer_call",
        actions=[
            BatchedClick(type="click", x=5, y=6, button="left"),
            BatchedType(type="type", text="batched"),
        ],
        call_id="tool_trace_batch",
        pending_safety_checks=[],
        status="completed",
    )
    tool_run = ToolRunComputerAction(tool_call=tool_call, computer_tool=comptool)
    agent = Agent(name="test_agent_trace_batch", tools=[comptool])

    set_tracing_disabled(False)
    with trace("computer-batch-span-test"):
        result = await ComputerAction.execute(
            agent=agent,
            action=tool_run,
            hooks=RunHooks[Any](),
            context_wrapper=RunContextWrapper(context=None),
            config=RunConfig(),
        )

    assert isinstance(result, ToolCallOutputItem)
    function_span = _get_function_span(ComputerAction.TRACE_TOOL_NAME)
    span_data = cast(dict[str, Any], function_span["span_data"])
    assert json.loads(cast(str, span_data["input"])) == [
        {"type": "click", "x": 5, "y": 6, "button": "left"},
        {"type": "type", "text": "batched"},
    ]

async def test_execute_redacts_span_error_when_sensitive_data_disabled() -> None:
    secret_error = "computer secret output"

    class FailingComputer(LoggingComputer):
        def screenshot(self) -> str:
            raise RuntimeError(secret_error)

    computer = FailingComputer()
    comptool = ComputerTool(computer=computer)
    tool_call = ResponseComputerToolCall(
        id="tool_trace_error",
        type="computer_call",
        action=ActionScreenshot(type="screenshot"),
        call_id="tool_trace_error",
        pending_safety_checks=[],
        status="completed",
    )
    tool_run = ToolRunComputerAction(tool_call=tool_call, computer_tool=comptool)
    agent = Agent(name="test_agent_trace_error", tools=[comptool])

    set_tracing_disabled(False)
    with trace("computer-span-redaction-test"):
        with pytest.raises(RuntimeError, match=secret_error):
            await ComputerAction.execute(
                agent=agent,
                action=tool_run,
                hooks=RunHooks[Any](),
                context_wrapper=RunContextWrapper(context=None),
                config=RunConfig(trace_include_sensitive_data=False),
            )

    function_span = _get_function_span(ComputerAction.TRACE_TOOL_NAME)
    assert function_span.get("error") == {
        "message": "Error running tool",
        "data": {
            "tool_name": ComputerAction.TRACE_TOOL_NAME,
            "error": "Tool execution failed. Error details are redacted.",
        },
    }
    assert secret_error not in json.dumps(function_span)
    span_data = cast(dict[str, Any], function_span["span_data"])
    assert span_data.get("input") is None
    assert span_data.get("output") is None

async def test_pending_safety_check_acknowledged() -> None:
    """Safety checks should be acknowledged via the callback."""

    computer = LoggingComputer(screenshot_return="img")
    called: list[ComputerToolSafetyCheckData] = []

    def on_sc(data: ComputerToolSafetyCheckData) -> bool:
        called.append(data)
        return True

    tool = ComputerTool(computer=computer, on_safety_check=on_sc)
    safety = PendingSafetyCheck(id="sc", code="c", message="m")
    tool_call = ResponseComputerToolCall(
        id="t1",
        type="computer_call",
        action=ActionClick(type="click", x=1, y=1, button="left"),
        call_id="t1",
        pending_safety_checks=[safety],
        status="completed",
    )
    run_action = ToolRunComputerAction(tool_call=tool_call, computer_tool=tool)
    agent = Agent(name="a", tools=[tool])
    ctx = RunContextWrapper(context=None)

    results = await run_loop.execute_computer_actions(
        agent=agent,
        actions=[run_action],
        hooks=RunHooks[Any](),
        context_wrapper=ctx,
        config=RunConfig(),
    )

    assert len(results) == 1
    raw = results[0].raw_item
    assert isinstance(raw, dict)
    assert raw.get("acknowledged_safety_checks") == [{"id": "sc", "code": "c", "message": "m"}]
    assert len(called) == 1
    assert called[0].safety_check.id == "sc"


# --- tests/test_config.py ---

def test_set_default_openai_api():
    assert isinstance(OpenAIProvider().get_model("gpt-4"), OpenAIResponsesModel), (
        "Default should be responses"
    )

    set_default_openai_api("chat_completions")
    assert isinstance(OpenAIProvider().get_model("gpt-4"), OpenAIChatCompletionsModel), (
        "Should be chat completions model"
    )

    set_default_openai_api("responses")
    assert isinstance(OpenAIProvider().get_model("gpt-4"), OpenAIResponsesModel), (
        "Should be responses model"
    )

def test_openai_provider_scopes_websocket_model_cache_to_running_loop():
    class DummyAsyncOpenAI:
        pass

    provider = OpenAIProvider(
        use_responses=True,
        use_responses_websocket=True,
        openai_client=DummyAsyncOpenAI(),  # type: ignore[arg-type]
    )

    async def get_model():
        return provider.get_model("gpt-4")

    loop1 = asyncio.new_event_loop()
    loop2 = asyncio.new_event_loop()
    try:
        model1 = loop1.run_until_complete(get_model())
        model1_again = loop1.run_until_complete(get_model())
        model2 = loop2.run_until_complete(get_model())
    finally:
        loop1.close()
        loop2.close()
        asyncio.set_event_loop(None)

    assert isinstance(model1, OpenAIResponsesWSModel)
    assert model1 is model1_again
    assert model2 is not model1


# --- tests/test_extension_filters.py ---

def test_str_history_and_list():
    handoff_input_data = handoff_data(
        input_history="Hello",
        new_items=(_get_message_output_run_item("Hello"),),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    assert filtered_data == handoff_input_data

def test_list_history_and_list():
    handoff_input_data = handoff_data(
        input_history=(_get_message_input_item("Hello"),),
        pre_handoff_items=(_get_message_output_run_item("123"),),
        new_items=(_get_message_output_run_item("World"),),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    assert filtered_data == handoff_input_data

def test_removes_tools_from_history():
    handoff_input_data = handoff_data(
        input_history=(
            _get_message_input_item("Hello1"),
            _get_function_result_input_item("World"),
            _get_message_input_item("Hello2"),
        ),
        pre_handoff_items=(
            _get_tool_output_run_item("abc"),
            _get_message_output_run_item("123"),
        ),
        new_items=(_get_message_output_run_item("World"),),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    assert len(filtered_data.input_history) == 2
    assert len(filtered_data.pre_handoff_items) == 1
    assert len(filtered_data.new_items) == 1

def test_removes_tools_from_new_items():
    handoff_input_data = handoff_data(
        new_items=(
            _get_message_output_run_item("Hello"),
            _get_tool_output_run_item("World"),
        ),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    assert len(filtered_data.input_history) == 0
    assert len(filtered_data.pre_handoff_items) == 0
    assert len(filtered_data.new_items) == 1

def test_removes_tools_from_new_items_and_history():
    handoff_input_data = handoff_data(
        input_history=(
            _get_message_input_item("Hello1"),
            _get_reasoning_input_item(),
            _get_function_result_input_item("World"),
            _get_message_input_item("Hello2"),
        ),
        pre_handoff_items=(
            _get_reasoning_output_run_item(),
            _get_message_output_run_item("123"),
            _get_tool_output_run_item("456"),
        ),
        new_items=(
            _get_reasoning_output_run_item(),
            _get_message_output_run_item("Hello"),
            _get_tool_output_run_item("World"),
        ),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    # reasoning items are also removed (they become orphaned after tool calls are stripped)
    assert len(filtered_data.input_history) == 2
    assert len(filtered_data.pre_handoff_items) == 1
    assert len(filtered_data.new_items) == 1

def test_removes_tool_search_from_history_and_items() -> None:
    handoff_input_data = handoff_data(
        input_history=(
            _get_message_input_item("Hello1"),
            cast(TResponseInputItem, _get_tool_search_call_input_item()),
            cast(TResponseInputItem, _get_tool_search_result_input_item()),
            _get_message_input_item("Hello2"),
        ),
        pre_handoff_items=(
            _get_tool_search_call_run_item(),
            _get_message_output_run_item("123"),
        ),
        new_items=(
            _get_tool_search_output_run_item(),
            _get_message_output_run_item("World"),
        ),
    )

    filtered_data = remove_all_tools(handoff_input_data)

    assert len(filtered_data.input_history) == 2
    assert len(filtered_data.pre_handoff_items) == 1
    assert len(filtered_data.new_items) == 1

def test_removes_handoffs_from_history():
    handoff_input_data = handoff_data(
        input_history=(
            _get_message_input_item("Hello1"),
            _get_handoff_input_item("World"),
        ),
        pre_handoff_items=(
            _get_reasoning_output_run_item(),
            _get_message_output_run_item("Hello"),
            _get_tool_output_run_item("World"),
            _get_handoff_output_run_item("World"),
        ),
        new_items=(
            _get_reasoning_output_run_item(),
            _get_message_output_run_item("Hello"),
            _get_tool_output_run_item("World"),
            _get_handoff_output_run_item("World"),
        ),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    assert len(filtered_data.input_history) == 1
    assert len(filtered_data.pre_handoff_items) == 1
    assert len(filtered_data.new_items) == 1

def test_nest_handoff_history_wraps_transcript() -> None:
    data = handoff_data(
        input_history=(_get_user_input_item("Hello"),),
        pre_handoff_items=(_get_message_output_run_item("Assist reply"),),
        new_items=(
            _get_message_output_run_item("Handoff request"),
            _get_handoff_output_run_item("transfer"),
        ),
    )

    nested = nest_handoff_history(data)

    assert isinstance(nested.input_history, tuple)
    assert len(nested.input_history) == 1
    summary = _as_message(nested.input_history[0])
    assert summary["role"] == "assistant"
    summary_content = summary["content"]
    assert isinstance(summary_content, str)
    start_marker, end_marker = get_conversation_history_wrappers()
    assert start_marker in summary_content
    assert end_marker in summary_content
    assert "Assist reply" in summary_content
    assert "Hello" in summary_content
    assert len(nested.pre_handoff_items) == 0
    assert nested.new_items == data.new_items

def test_nest_handoff_history_handles_missing_user() -> None:
    data = handoff_data(
        pre_handoff_items=(_get_reasoning_output_run_item(),),
    )

    nested = nest_handoff_history(data)

    assert isinstance(nested.input_history, tuple)
    assert len(nested.input_history) == 1
    summary = _as_message(nested.input_history[0])
    assert summary["role"] == "assistant"
    summary_content = summary["content"]
    assert isinstance(summary_content, str)
    assert "reasoning" in summary_content.lower()

def test_nest_handoff_history_appends_existing_history() -> None:
    first = handoff_data(
        input_history=(_get_user_input_item("Hello"),),
        pre_handoff_items=(_get_message_output_run_item("First reply"),),
    )

    first_nested = nest_handoff_history(first)
    assert isinstance(first_nested.input_history, tuple)
    summary_message = first_nested.input_history[0]

    follow_up_history: tuple[TResponseInputItem, ...] = (
        summary_message,
        _get_user_input_item("Another question"),
    )

    second = handoff_data(
        input_history=follow_up_history,
        pre_handoff_items=(_get_message_output_run_item("Second reply"),),
        new_items=(_get_handoff_output_run_item("transfer"),),
    )

    second_nested = nest_handoff_history(second)

    assert isinstance(second_nested.input_history, tuple)
    summary = _as_message(second_nested.input_history[0])
    assert summary["role"] == "assistant"
    content = summary["content"]
    assert isinstance(content, str)
    start_marker, end_marker = get_conversation_history_wrappers()
    assert content.count(start_marker) == 1
    assert content.count(end_marker) == 1
    assert "First reply" in content
    assert "Second reply" in content
    assert "Another question" in content

def test_nest_handoff_history_honors_custom_wrappers() -> None:
    data = handoff_data(
        input_history=(_get_user_input_item("Hello"),),
        pre_handoff_items=(_get_message_output_run_item("First reply"),),
        new_items=(_get_message_output_run_item("Second reply"),),
    )

    set_conversation_history_wrappers(start="<<START>>", end="<<END>>")
    try:
        nested = nest_handoff_history(data)
        assert isinstance(nested.input_history, tuple)
        assert len(nested.input_history) == 1
        summary = _as_message(nested.input_history[0])
        summary_content = summary["content"]
        assert isinstance(summary_content, str)
        lines = summary_content.splitlines()
        assert lines[0] == (
            "For context, here is the conversation so far between the user and the previous agent:"
        )
        assert lines[1].startswith("<<START>>")
        assert summary_content.endswith("<<END>>")

        # Ensure the custom markers are parsed correctly when nesting again.
        second_nested = nest_handoff_history(nested)
        assert isinstance(second_nested.input_history, tuple)
        second_summary = _as_message(second_nested.input_history[0])
        content = second_summary["content"]
        assert isinstance(content, str)
        assert content.count("<<START>>") == 1
        assert content.count("<<END>>") == 1
    finally:
        reset_conversation_history_wrappers()

def test_nest_handoff_history_supports_custom_mapper() -> None:
    data = handoff_data(
        input_history=(_get_user_input_item("Hello"),),
        pre_handoff_items=(_get_message_output_run_item("Assist reply"),),
    )

    def map_history(items: list[TResponseInputItem]) -> list[TResponseInputItem]:
        reversed_items = list(reversed(items))
        return [deepcopy(item) for item in reversed_items]

    nested = nest_handoff_history(data, history_mapper=map_history)

    assert isinstance(nested.input_history, tuple)
    assert len(nested.input_history) == 2
    first = _as_message(nested.input_history[0])
    second = _as_message(nested.input_history[1])
    assert first["role"] == "assistant"
    first_content = first.get("content")
    assert isinstance(first_content, list)
    assert any(
        isinstance(chunk, dict)
        and chunk.get("type") == "output_text"
        and chunk.get("text") == "Assist reply"
        for chunk in first_content
    )
    assert second["role"] == "user"
    assert second["content"] == "Hello"

def test_nest_handoff_history_parse_summary_line_edge_cases() -> None:
    """Test edge cases in parsing summary lines."""
    # Create a nested summary that will be parsed
    first_summary = nest_handoff_history(
        handoff_data(
            input_history=(_get_user_input_item("Hello"),),
            pre_handoff_items=(_get_message_output_run_item("Reply"),),
        )
    )

    # Create a second nested summary that includes the first
    # This will trigger parsing of the nested summary lines
    assert isinstance(first_summary.input_history, tuple)
    second_data = handoff_data(
        input_history=(
            first_summary.input_history[0],
            _get_user_input_item("Another question"),
        ),
    )

    nested = nest_handoff_history(second_data)
    # Should successfully parse and include both messages
    assert isinstance(nested.input_history, tuple)
    summary = _as_message(nested.input_history[0])
    assert "Hello" in summary["content"] or "Another question" in summary["content"]

def test_removes_mcp_run_items_from_new_items() -> None:
    """MCP RunItem types should be removed from new_items and pre_handoff_items."""
    handoff_input_data = handoff_data(
        pre_handoff_items=(
            _get_mcp_list_tools_run_item(),
            _get_mcp_approval_request_run_item(),
            _get_message_output_run_item("kept"),
        ),
        new_items=(
            _get_mcp_call_run_item(),
            _get_mcp_approval_response_run_item(),
            _get_message_output_run_item("also kept"),
        ),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    # Only message items should remain
    assert len(filtered_data.pre_handoff_items) == 1
    assert len(filtered_data.new_items) == 1

def test_removes_mixed_mcp_and_function_items() -> None:
    """Both MCP and function tool items should be removed together."""
    handoff_input_data = handoff_data(
        input_history=(
            _get_message_input_item("Start"),
            _get_mcp_call_input_item(),
            _get_function_result_input_item("fn output"),
            _get_reasoning_input_item(),
            _get_mcp_approval_response_input_item(),
            _get_message_input_item("End"),
        ),
        pre_handoff_items=(
            _get_mcp_list_tools_run_item(),
            _get_tool_output_run_item("fn output"),
            _get_reasoning_output_run_item(),
            _get_message_output_run_item("kept"),
        ),
        new_items=(
            _get_mcp_call_run_item(),
            _get_mcp_approval_request_run_item(),
            _get_mcp_approval_response_run_item(),
            _get_message_output_run_item("also kept"),
        ),
    )
    filtered_data = remove_all_tools(handoff_input_data)
    assert len(filtered_data.input_history) == 2
    assert len(filtered_data.pre_handoff_items) == 1
    assert len(filtered_data.new_items) == 1


# --- tests/test_gemini_thought_signatures.py ---

def test_gemini_multiple_tool_calls_with_thought_signatures():
    """Test multiple tool calls each preserve their own thought signatures."""
    tool_call_1 = InternalToolCall(
        id="call_1",
        type="function",
        function=Function(name="func_a", arguments='{"x": 1}'),
        extra_content={"google": {"thought_signature": "sig_aaa"}},
    )
    tool_call_2 = InternalToolCall(
        id="call_2",
        type="function",
        function=Function(name="func_b", arguments='{"y": 2}'),
        extra_content={"google": {"thought_signature": "sig_bbb"}},
    )

    message = InternalChatCompletionMessage(
        role="assistant",
        content="Calling two functions.",
        reasoning_content="",
        tool_calls=[tool_call_1, tool_call_2],
    )

    provider_data = {"model": "gemini/gemini-3-pro"}
    items = Converter.message_to_output_items(message, provider_data=provider_data)

    func_calls = [i for i in items if hasattr(i, "type") and i.type == "function_call"]
    assert len(func_calls) == 2

    assert func_calls[0].model_dump()["provider_data"]["thought_signature"] == "sig_aaa"
    assert func_calls[1].model_dump()["provider_data"]["thought_signature"] == "sig_bbb"


# --- tests/test_handoff_tool.py ---

async def test_single_handoff_setup():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2", handoffs=[agent_1])

    assert not agent_1.handoffs
    assert agent_2.handoffs == [agent_1]

    assert not (await get_handoffs(agent_1, RunContextWrapper(agent_1)))

    handoff_objects = await get_handoffs(agent_2, RunContextWrapper(agent_2))
    assert len(handoff_objects) == 1
    obj = handoff_objects[0]
    assert obj.tool_name == Handoff.default_tool_name(agent_1)
    assert obj.tool_description == Handoff.default_tool_description(agent_1)
    assert obj.agent_name == agent_1.name

async def test_multiple_handoffs_setup():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(name="test_3", handoffs=[agent_1, agent_2])

    assert agent_3.handoffs == [agent_1, agent_2]
    assert not agent_1.handoffs
    assert not agent_2.handoffs

    handoff_objects = await get_handoffs(agent_3, RunContextWrapper(agent_3))
    assert len(handoff_objects) == 2
    assert handoff_objects[0].tool_name == Handoff.default_tool_name(agent_1)
    assert handoff_objects[1].tool_name == Handoff.default_tool_name(agent_2)

    assert handoff_objects[0].tool_description == Handoff.default_tool_description(agent_1)
    assert handoff_objects[1].tool_description == Handoff.default_tool_description(agent_2)

    assert handoff_objects[0].agent_name == agent_1.name
    assert handoff_objects[1].agent_name == agent_2.name

async def test_custom_handoff_setup():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(
        name="test_3",
        handoffs=[
            agent_1,
            handoff(
                agent_2,
                tool_name_override="custom_tool_name",
                tool_description_override="custom tool description",
            ),
        ],
    )

    assert len(agent_3.handoffs) == 2
    assert not agent_1.handoffs
    assert not agent_2.handoffs

    handoff_objects = await get_handoffs(agent_3, RunContextWrapper(agent_3))
    assert len(handoff_objects) == 2

    first_handoff = handoff_objects[0]
    assert isinstance(first_handoff, Handoff)
    assert first_handoff.tool_name == Handoff.default_tool_name(agent_1)
    assert first_handoff.tool_description == Handoff.default_tool_description(agent_1)
    assert first_handoff.agent_name == agent_1.name

    second_handoff = handoff_objects[1]
    assert isinstance(second_handoff, Handoff)
    assert second_handoff.tool_name == "custom_tool_name"
    assert second_handoff.tool_description == "custom tool description"
    assert second_handoff.agent_name == agent_2.name

async def test_handoff_input_type():
    async def _on_handoff(ctx: RunContextWrapper[Any], input: Foo):
        pass

    agent = Agent(name="test")
    obj = handoff(agent, input_type=Foo, on_handoff=_on_handoff)
    for key, value in Foo.model_json_schema().items():
        assert obj.input_json_schema[key] == value

    # Invalid JSON should raise an error
    with pytest.raises(ModelBehaviorError):
        await obj.on_invoke_handoff(RunContextWrapper(agent), "not json")

    # Empty JSON should raise an error
    with pytest.raises(ModelBehaviorError):
        await obj.on_invoke_handoff(RunContextWrapper(agent), "")

    # Valid JSON should call the on_handoff function
    invoked = await obj.on_invoke_handoff(
        RunContextWrapper(agent), Foo(bar="baz").model_dump_json()
    )
    assert invoked == agent

async def test_on_handoff_called():
    was_called = False

    async def _on_handoff(ctx: RunContextWrapper[Any], input: Foo):
        nonlocal was_called
        was_called = True

    agent = Agent(name="test")
    obj = handoff(agent, input_type=Foo, on_handoff=_on_handoff)
    for key, value in Foo.model_json_schema().items():
        assert obj.input_json_schema[key] == value

    invoked = await obj.on_invoke_handoff(
        RunContextWrapper(agent), Foo(bar="baz").model_dump_json()
    )
    assert invoked == agent

    assert was_called, "on_handoff should have been called"

async def test_on_handoff_without_input_called():
    was_called = False

    def _on_handoff(ctx: RunContextWrapper[Any]):
        nonlocal was_called
        was_called = True

    agent = Agent(name="test")
    obj = handoff(agent, on_handoff=_on_handoff)

    invoked = await obj.on_invoke_handoff(RunContextWrapper(agent), "")
    assert invoked == agent

    assert was_called, "on_handoff should have been called"

async def test_async_on_handoff_without_input_called():
    was_called = False

    async def _on_handoff(ctx: RunContextWrapper[Any]):
        nonlocal was_called
        was_called = True

    agent = Agent(name="test")
    obj = handoff(agent, on_handoff=_on_handoff)

    invoked = await obj.on_invoke_handoff(RunContextWrapper(agent), "")
    assert invoked == agent

    assert was_called, "on_handoff should have been called"

def test_handoff_input_data():
    agent = Agent(name="test")

    data = HandoffInputData(
        input_history="",
        pre_handoff_items=(),
        new_items=(),
        run_context=RunContextWrapper(context=()),
    )
    assert get_len(data) == 1

    data = HandoffInputData(
        input_history=({"role": "user", "content": "foo"},),
        pre_handoff_items=(),
        new_items=(),
        run_context=RunContextWrapper(context=()),
    )
    assert get_len(data) == 1

    data = HandoffInputData(
        input_history=(
            {"role": "user", "content": "foo"},
            {"role": "assistant", "content": "bar"},
        ),
        pre_handoff_items=(),
        new_items=(),
        run_context=RunContextWrapper(context=()),
    )
    assert get_len(data) == 2

    data = HandoffInputData(
        input_history=({"role": "user", "content": "foo"},),
        pre_handoff_items=(
            message_item("foo", agent),
            message_item("foo2", agent),
        ),
        new_items=(
            message_item("bar", agent),
            message_item("baz", agent),
        ),
        run_context=RunContextWrapper(context=()),
    )
    assert get_len(data) == 5

    data = HandoffInputData(
        input_history=(
            {"role": "user", "content": "foo"},
            {"role": "assistant", "content": "bar"},
        ),
        pre_handoff_items=(message_item("baz", agent),),
        new_items=(
            message_item("baz", agent),
            message_item("qux", agent),
        ),
        run_context=RunContextWrapper(context=()),
    )

    assert get_len(data) == 5

async def test_handoff_is_enabled_callable():
    """Test that handoff respects is_enabled callable parameter."""
    agent = Agent(name="test")

    # Test callable that returns True
    def always_enabled(ctx: RunContextWrapper[Any], agent: Agent[Any]) -> bool:
        return True

    handoff_callable_enabled = handoff(agent, is_enabled=always_enabled)
    assert callable(handoff_callable_enabled.is_enabled)
    result = handoff_callable_enabled.is_enabled(RunContextWrapper(agent), agent)
    assert inspect.isawaitable(result)
    result = await result
    assert result is True

    # Test callable that returns False
    def always_disabled(ctx: RunContextWrapper[Any], agent: Agent[Any]) -> bool:
        return False

    handoff_callable_disabled = handoff(agent, is_enabled=always_disabled)
    assert callable(handoff_callable_disabled.is_enabled)
    result = handoff_callable_disabled.is_enabled(RunContextWrapper(agent), agent)
    assert inspect.isawaitable(result)
    result = await result
    assert result is False

    # Test async callable
    async def async_enabled(ctx: RunContextWrapper[Any], agent: Agent[Any]) -> bool:
        return True

    handoff_async_enabled = handoff(agent, is_enabled=async_enabled)
    assert callable(handoff_async_enabled.is_enabled)
    result = await handoff_async_enabled.is_enabled(RunContextWrapper(agent), agent)  # type: ignore
    assert result is True

async def test_handoff_is_enabled_filtering_integration():
    """Integration test that disabled handoffs are filtered out by the runner."""

    # Set up agents
    agent_1 = Agent(name="agent_1")
    agent_2 = Agent(name="agent_2")
    agent_3 = Agent(name="agent_3")

    # Create main agent with mixed enabled/disabled handoffs
    main_agent = Agent(
        name="main_agent",
        handoffs=[
            handoff(agent_1, is_enabled=True),  # enabled
            handoff(agent_2, is_enabled=False),  # disabled
            handoff(agent_3, is_enabled=lambda ctx, agent: True),  # enabled callable
        ],
    )

    context_wrapper = RunContextWrapper(main_agent)

    # Get filtered handoffs using the runner's method
    filtered_handoffs = await get_handoffs(main_agent, context_wrapper)

    # Should only have 2 handoffs (agent_1 and agent_3), agent_2 should be filtered out
    assert len(filtered_handoffs) == 2

    # Check that the correct agents are present
    agent_names = {h.agent_name for h in filtered_handoffs}
    assert agent_names == {"agent_1", "agent_3"}
    assert "agent_2" not in agent_names


# --- tests/test_hitl_error_scenarios.py ---

async def test_resumed_hitl_executes_approved_tools(
    setup_fn: Callable[[], ApprovalScenario],
    user_input: str,
) -> None:
    """Approved tools should run once the interrupted turn resumes."""
    scenario = setup_fn()
    model, agent = make_model_and_agent(tools=[scenario.tool])

    result = await run_and_resume_after_approval(
        agent,
        model,
        scenario.raw_call,
        scenario.final_output,
        user_input=user_input,
    )

    scenario.assert_result(result)

async def test_resume_does_not_duplicate_pending_shell_approvals() -> None:
    """Resuming should not duplicate pending shell approvals."""
    tool = ShellTool(executor=lambda _request: "shell_output", needs_approval=True)
    model, agent = make_model_and_agent(tools=[tool])
    raw_call = make_shell_call(
        "call_shell_pending_dup",
        id_value="shell_pending_dup",
        commands=["echo pending"],
    )
    call_id = extract_tool_call_id(raw_call)
    assert call_id, "shell call must have a call_id"

    model.set_next_output([raw_call])
    first = await Runner.run(agent, "run shell")
    assert first.interruptions, "shell tool should require approval"

    resumed = await Runner.run(agent, first.to_state())
    pending_items = [
        item
        for item in resumed.new_items
        if isinstance(item, ToolApprovalItem) and extract_tool_call_id(item.raw_item) == call_id
    ]
    assert len(pending_items) == 1

async def test_route_local_shell_calls_to_remote_shell_tool():
    """Test that local shell calls are routed to the local shell tool.

    When processing model output with LocalShellCall items, they should be handled by
    LocalShellTool (not ShellTool), even when both tools are registered. This ensures
    local shell operations use the correct executor and approval hooks.
    """
    remote_shell_executed = []
    local_shell_executed = []

    def remote_executor(request: Any) -> str:
        remote_shell_executed.append(request)
        return "remote_output"

    def local_executor(request: Any) -> str:
        local_shell_executed.append(request)
        return "local_output"

    shell_tool = ShellTool(executor=remote_executor)
    local_shell_tool = LocalShellTool(executor=local_executor)
    model, agent = make_model_and_agent(tools=[shell_tool, local_shell_tool])

    # Model emits a local_shell_call
    local_shell_call = LocalShellCall(
        id="local_1",
        call_id="call_local_1",
        type="local_shell_call",
        action={"type": "exec", "command": ["echo", "test"], "env": {}},  # type: ignore[arg-type]
        status="in_progress",
    )
    model.set_next_output([local_shell_call])

    await Runner.run(agent, "run local shell")

    # Local shell call should be handled by LocalShellTool, not ShellTool
    # This test will fail because LocalShellCall is routed to shell_tool first
    assert len(local_shell_executed) > 0, "LocalShellTool should have been executed"
    assert len(remote_shell_executed) == 0, (
        "ShellTool should not have been executed for local shell call"
    )

async def test_preserve_max_turns_when_resuming_from_runresult_state():
    """Test that max_turns is preserved when resuming from RunResult state.

    A run configured with max_turns=20 should keep that limit after resuming from
    result.to_state() without re-passing max_turns.
    """

    async def test_tool() -> str:
        return "tool_result"

    # Create the tool with needs_approval directly
    # The tool name will be "test_tool" based on the function name
    tool = function_tool(test_tool, needs_approval=require_approval)
    model, agent = make_model_and_agent(tools=[tool])

    model.add_multiple_turn_outputs([[make_function_tool_call("test_tool", call_id="call-1")]])

    result1 = await Runner.run(agent, "call test_tool", max_turns=20)
    assert result1.interruptions, "should have an interruption"

    state = approve_first_interruption(result1, always_approve=True)

    # Provide 10 more turns (turns 2-11) to ensure we exceed the default 10 but not 20.
    model.add_multiple_turn_outputs(
        [
            [
                get_text_message(f"turn {i + 2}"),  # Text message first (doesn't finish)
                make_function_tool_call("test_tool", call_id=f"call-{i + 2}"),
            ]
            for i in range(10)
        ]
    )

    result2 = await Runner.run(agent, state)
    assert result2 is not None, "Run should complete successfully with max_turns=20 from state"

async def test_current_turn_not_preserved_in_to_state():
    """Test that current turn counter is preserved when converting RunResult to RunState."""

    async def test_tool() -> str:
        return "tool_result"

    tool = function_tool(test_tool, needs_approval=require_approval)
    model, agent = make_model_and_agent(tools=[tool])

    # Model emits a tool call requiring approval
    model.set_next_output([make_function_tool_call("test_tool", call_id="call-1")])

    # First turn with interruption
    result1 = await Runner.run(agent, "call test_tool")
    assert result1.interruptions, "should have interruption on turn 1"

    # Convert to state - this should preserve current_turn=1
    state1 = result1.to_state()

    # Regression guard: to_state should keep the turn counter instead of resetting it.
    assert state1._current_turn == 1, (
        f"Expected current_turn=1 after 1 turn, got {state1._current_turn}. "
        "to_state() should preserve the current turn counter."
    )

async def test_deserialize_interruptions_preserve_tool_calls(
    tool_factory: Callable[[], Any],
    raw_call_factory: Callable[[], TResponseOutputItem],
    expected_tool_name: str,
    user_input: str,
) -> None:
    """Ensure deserialized interruptions preserve tool types instead of forcing function calls."""
    model, agent = make_model_and_agent(tools=[tool_factory()])
    await assert_roundtrip_tool_name(
        agent, model, raw_call_factory(), expected_tool_name, user_input=user_input
    )

async def test_deserialize_interruptions_preserve_mcp_tools(
    include_provider_data: bool,
) -> None:
    """Ensure MCP/hosted tool approvals survive serialization."""
    model, agent = make_model_and_agent(tools=[])

    mcp_approval_item = make_mcp_approval_item(
        agent, call_id="mcp-approval-1", include_provider_data=include_provider_data
    )
    state = make_state_with_interruptions(agent, [mcp_approval_item])

    state_json = state.to_json()

    deserialized_state = await RunStateClass.from_json(agent, state_json)
    interruptions = deserialized_state.get_interruptions()
    assert len(interruptions) > 0, "Interruptions should be preserved after deserialization"
    assert interruptions[0].tool_name == "test_mcp_tool", (
        "MCP tool approval should be preserved, not converted to function"
    )

async def test_preserve_persisted_item_counter_when_resuming_streamed_runs():
    """Preserve the persisted-item counter on streamed resume to avoid losing history."""
    model, agent = make_model_and_agent()

    # Simulate a turn interrupted mid-persistence: 5 items generated, 3 actually saved.
    context_wrapper = make_context_wrapper()
    state = RunState(
        context=context_wrapper,
        original_input="test input",
        starting_agent=agent,
        max_turns=10,
    )

    # Create 5 generated items (simulating multiple outputs before interruption)
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    for i in range(5):
        message_item = MessageOutputItem(
            agent=agent,
            raw_item=ResponseOutputMessage(
                id=f"msg_{i}",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        type="output_text", text=f"Message {i}", annotations=[], logprobs=[]
                    )
                ],
            ),
        )
        state._generated_items.append(message_item)

    # Persisted count reflects what was already written before interruption.
    state._current_turn_persisted_item_count = 3

    # Add a model response so the state is valid for resumption
    state._model_responses = [
        ModelResponse(
            output=[get_text_message("test")],
            usage=Usage(),
            response_id="resp_1",
        )
    ]

    # Set up model to return final output immediately (so the run completes)
    model.set_next_output([get_text_message("done")])

    result = Runner.run_streamed(agent, state)

    assert result._current_turn_persisted_item_count == 3, (
        f"Expected _current_turn_persisted_item_count=3 (the actual persisted count), "
        f"but got {result._current_turn_persisted_item_count}. "
        f"The counter should reflect persisted items, not len(_generated_items)="
        f"{len(state._generated_items)}."
    )

    await consume_stream(result)

async def test_function_needs_approval_invalid_type_raises() -> None:
    """needs_approval must be bool or callable; invalid types should raise UserError."""

    @function_tool(name_override="bad_tool", needs_approval=cast(Any, "always"))
    def bad_tool() -> str:
        return "ok"

    model, agent = make_model_and_agent(tools=[bad_tool])
    model.set_next_output([make_function_tool_call("bad_tool")])

    with pytest.raises(UserError, match="needs_approval"):
        await Runner.run(agent, "run invalid")

async def test_resume_honors_permanent_namespaced_function_approval_with_new_call_id() -> None:
    @function_tool(needs_approval=True, name_override="lookup_account")
    async def lookup_account(customer_id: str) -> str:
        return customer_id

    namespaced_tool = tool_namespace(
        name="billing",
        description="Billing tools",
        tools=[lookup_account],
    )[0]
    context_wrapper = make_context_wrapper()
    approved_item = ToolApprovalItem(
        agent=Agent(name="billing-agent"),
        raw_item=make_function_tool_call(
            "lookup_account",
            call_id="approved-call",
            arguments='{"customer_id":"customer_1"}',
            namespace="billing",
        ),
    )
    context_wrapper.approve_tool(approved_item, always_approve=True)

    resumed_run = ToolRunFunction(
        tool_call=make_function_tool_call(
            "lookup_account",
            call_id="resumed-call",
            arguments='{"customer_id":"customer_2"}',
            namespace="billing",
        ),
        function_tool=namespaced_tool,
    )
    pending: list[ToolApprovalItem] = []
    rejections: list[str | None] = []

    async def _needs_approval_checker(_run: ToolRunFunction) -> bool:
        return True

    async def _record_rejection(
        call_id: str | None,
        _tool_call: ResponseFunctionToolCall,
        _tool: Any,
    ) -> None:
        rejections.append(call_id)

    selected = await _select_function_tool_runs_for_resume(
        [resumed_run],
        approval_items_by_call_id={},
        context_wrapper=context_wrapper,
        needs_approval_checker=_needs_approval_checker,
        output_exists_checker=lambda _run: False,
        record_rejection=_record_rejection,
        pending_interruption_adder=pending.append,
        pending_item_builder=lambda run: ToolApprovalItem(
            agent=Agent(name="billing-agent"),
            raw_item=run.tool_call,
            tool_name=run.function_tool.name,
            tool_namespace="billing",
        ),
    )

    assert selected == [resumed_run]
    assert pending == []
    assert rejections == []


# --- tests/test_hitl_session_scenario.py ---

async def test_memory_session_hitl_scenario() -> None:
    execute_counts.clear()
    session = SimpleListSession(session_id="memory")
    model = ScenarioModel()

    steps = [
        ScenarioStep(
            label="turn 1",
            message=USER_MESSAGES[0],
            tool_name=TOOL_ECHO,
            approval="approve",
            expected_output=f"approved:{USER_MESSAGES[0]}",
        ),
        ScenarioStep(
            label="turn 2 (rehydrated)",
            message=USER_MESSAGES[1],
            tool_name=TOOL_NOTE,
            approval="approve",
            expected_output=f"approved_note:{USER_MESSAGES[1]}",
        ),
        ScenarioStep(
            label="turn 3 (rejected)",
            message=USER_MESSAGES[2],
            tool_name=TOOL_ECHO,
            approval="reject",
            expected_output=HITL_REJECTION_MSG,
        ),
    ]

    rehydrated: SimpleListSession | None = None

    try:
        first = await run_scenario_step(session, model, steps[0])
        assert_counts(first.items, 1)
        assert_step_output(first.items, first.approval_item, steps[0])

        rehydrated = SimpleListSession(
            session_id=session.session_id,
            history=first.items,
        )
        second = await run_scenario_step(rehydrated, model, steps[1])
        assert_counts(second.items, 2)
        assert_step_output(second.items, second.approval_item, steps[1])

        third = await run_scenario_step(rehydrated, model, steps[2])
        assert_counts(third.items, 3)
        assert_step_output(third.items, third.approval_item, steps[2])

        assert execute_counts.get(TOOL_ECHO) == 1
        assert execute_counts.get(TOOL_NOTE) == 1
    finally:
        await (rehydrated or session).clear_session()

async def test_openai_conversations_session_hitl_scenario() -> None:
    execute_counts.clear()
    stored_items: list[dict[str, Any]] = []

    async def create_items(*, conversation_id: str, items: list[Any]) -> None:
        stored_items.extend(items)

    def list_items(*, conversation_id: str, order: str, limit: int | None = None):
        class StoredItem:
            def __init__(self, payload: dict[str, Any]) -> None:
                self._payload = payload

            def model_dump(self, exclude_unset: bool = True) -> dict[str, Any]:
                return self._payload

        async def iterator():
            if order == "desc":
                items_iter = list(reversed(stored_items))
            else:
                items_iter = list(stored_items)
            if limit is not None:
                items_iter = items_iter[:limit]
            for item in items_iter:
                yield StoredItem(item)

        return iterator()

    class ConversationsItems:
        create = staticmethod(create_items)
        list = staticmethod(list_items)

        async def delete(self, *args: Any, **kwargs: Any) -> None:
            return None

    class Conversations:
        items = ConversationsItems()

        async def create(self, *args: Any, **kwargs: Any) -> Any:
            return type("Response", (), {"id": "conv_test"})()

        async def delete(self, *args: Any, **kwargs: Any) -> None:
            return None

    class Client:
        conversations = Conversations()

    client = Client()
    typed_client = cast(Any, client)
    session = OpenAIConversationsSession(conversation_id="conv_test", openai_client=typed_client)
    rehydrated_session = OpenAIConversationsSession(
        conversation_id="conv_test", openai_client=typed_client
    )
    model = ScenarioModel()

    steps = [
        ScenarioStep(
            label="turn 1",
            message=USER_MESSAGES[0],
            tool_name=TOOL_ECHO,
            approval="approve",
            expected_output=f"approved:{USER_MESSAGES[0]}",
        ),
        ScenarioStep(
            label="turn 2 (rehydrated)",
            message=USER_MESSAGES[1],
            tool_name=TOOL_NOTE,
            approval="approve",
            expected_output=f"approved_note:{USER_MESSAGES[1]}",
        ),
        ScenarioStep(
            label="turn 3 (rejected)",
            message=USER_MESSAGES[2],
            tool_name=TOOL_ECHO,
            approval="reject",
            expected_output=HITL_REJECTION_MSG,
        ),
    ]

    offset = 0
    first = await run_scenario_step(session, model, steps[0])
    first_items = stored_items[offset:]
    offset = len(stored_items)
    assert_step_items(first_items, steps[0], first.approval_item)

    second = await run_scenario_step(rehydrated_session, model, steps[1])
    second_items = stored_items[offset:]
    offset = len(stored_items)
    assert_step_items(second_items, steps[1], second.approval_item)

    third = await run_scenario_step(rehydrated_session, model, steps[2])
    third_items = stored_items[offset:]
    assert_step_items(third_items, steps[2], third.approval_item)

    assert execute_counts.get(TOOL_ECHO) == 1
    assert execute_counts.get(TOOL_NOTE) == 1


# --- tests/test_items_helpers.py ---

def test_to_input_items_for_message() -> None:
    """An output message should convert into an input dict matching the message's own structure."""
    content = ResponseOutputText(
        annotations=[], text="hello world", type="output_text", logprobs=[]
    )
    message = ResponseOutputMessage(
        id="m1", content=[content], role="assistant", status="completed", type="message"
    )
    resp = ModelResponse(output=[message], usage=Usage(), response_id=None)
    input_items = resp.to_input_items()
    assert isinstance(input_items, list) and len(input_items) == 1
    # The dict should contain exactly the primitive values of the message
    expected: ResponseOutputMessageParam = {
        "id": "m1",
        "content": [
            {
                "annotations": [],
                "logprobs": [],
                "text": "hello world",
                "type": "output_text",
            }
        ],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }
    assert input_items[0] == expected

def test_to_input_items_for_file_search_call() -> None:
    """A file search tool call output should produce the same dict as a file search input."""
    fs_call = ResponseFileSearchToolCall(
        id="fs1", queries=["query"], status="completed", type="file_search_call"
    )
    resp = ModelResponse(output=[fs_call], usage=Usage(), response_id=None)
    input_items = resp.to_input_items()
    assert isinstance(input_items, list) and len(input_items) == 1
    expected: ResponseFileSearchToolCallParam = {
        "id": "fs1",
        "queries": ["query"],
        "status": "completed",
        "type": "file_search_call",
    }
    assert input_items[0] == expected

def test_to_input_items_for_web_search_call() -> None:
    """A web search tool call output should produce the same dict as a web search input."""
    ws_call = ResponseFunctionWebSearch(
        id="w1",
        action=ActionSearch(type="search", query="query"),
        status="completed",
        type="web_search_call",
    )
    resp = ModelResponse(output=[ws_call], usage=Usage(), response_id=None)
    input_items = resp.to_input_items()
    assert isinstance(input_items, list) and len(input_items) == 1
    expected: ResponseFunctionWebSearchParam = {
        "id": "w1",
        "status": "completed",
        "type": "web_search_call",
        "action": {"type": "search", "query": "query"},
    }
    assert input_items[0] == expected

def test_to_input_items_for_computer_call_click() -> None:
    """A computer call output should yield a dict whose shape matches the computer call input."""
    action = ActionScreenshot(type="screenshot")
    comp_call = ResponseComputerToolCall(
        id="comp1",
        action=action,
        type="computer_call",
        call_id="comp1",
        pending_safety_checks=[],
        status="completed",
    )
    resp = ModelResponse(output=[comp_call], usage=Usage(), response_id=None)
    input_items = resp.to_input_items()
    assert isinstance(input_items, list) and len(input_items) == 1
    converted_dict = input_items[0]
    # Top-level keys should match what we expect for a computer call input
    expected: ResponseComputerToolCallParam = {
        "id": "comp1",
        "type": "computer_call",
        "action": {"type": "screenshot"},
        "call_id": "comp1",
        "pending_safety_checks": [],
        "status": "completed",
    }
    assert converted_dict == expected

def test_to_input_items_for_computer_call_batched_actions() -> None:
    """A batched computer call should preserve its actions list when replayed as input."""
    comp_call = ResponseComputerToolCall(
        id="comp2",
        actions=[
            BatchedClick(type="click", x=3, y=4, button="left"),
            BatchedType(type="type", text="hello"),
        ],
        type="computer_call",
        call_id="comp2",
        pending_safety_checks=[],
        status="completed",
    )
    resp = ModelResponse(output=[comp_call], usage=Usage(), response_id=None)
    input_items = resp.to_input_items()
    assert isinstance(input_items, list) and len(input_items) == 1
    assert input_items[0] == {
        "id": "comp2",
        "type": "computer_call",
        "actions": [
            {"type": "click", "x": 3, "y": 4, "button": "left"},
            {"type": "type", "text": "hello"},
        ],
        "call_id": "comp2",
        "pending_safety_checks": [],
        "status": "completed",
    }

def test_to_input_items_for_tool_search_strips_created_by() -> None:
    """Tool-search output items should reuse the replay sanitizer before round-tripping."""
    tool_search_call = ResponseToolSearchCall(
        id="tsc_123",
        call_id="call_tsc_123",
        arguments={"query": "profile"},
        execution="server",
        status="completed",
        type="tool_search_call",
        created_by="server",
    )
    tool_search_output = ResponseToolSearchOutputItem(
        id="tso_123",
        call_id="call_tsc_123",
        execution="server",
        status="completed",
        tools=[],
        type="tool_search_output",
        created_by="server",
    )

    resp = ModelResponse(
        output=[tool_search_call, tool_search_output], usage=Usage(), response_id=None
    )
    input_items = resp.to_input_items()

    assert input_items == [
        {
            "id": "tsc_123",
            "call_id": "call_tsc_123",
            "arguments": {"query": "profile"},
            "execution": "server",
            "status": "completed",
            "type": "tool_search_call",
        },
        {
            "id": "tso_123",
            "call_id": "call_tsc_123",
            "execution": "server",
            "status": "completed",
            "tools": [],
            "type": "tool_search_output",
        },
    ]

def test_input_to_new_input_list_copies_the_ones_produced_by_pydantic() -> None:
    """Validated input items should be copied and made JSON dump compatible."""
    original = ResponseOutputMessageParam(
        id="a75654dc-7492-4d1c-bce0-89e8312fbdd7",
        content=[
            ResponseOutputTextParam(
                type="output_text",
                text="Hey, what's up?",
                annotations=[],
                logprobs=[],
            )
        ],
        role="assistant",
        status="completed",
        type="message",
    )
    validated = TypeAdapter(list[ResponseInputItemParam]).validate_python([original])

    new_list = ItemHelpers.input_to_new_input_list(validated)
    assert len(new_list) == 1
    assert new_list[0]["id"] == original["id"]  # type: ignore
    assert new_list[0]["role"] == original["role"]  # type: ignore
    assert new_list[0]["status"] == original["status"]  # type: ignore
    assert new_list[0]["type"] == original["type"]
    assert isinstance(new_list[0]["content"], list)

    first_content = cast(dict[str, object], new_list[0]["content"][0])
    assert first_content["type"] == "output_text"
    assert first_content["text"] == "Hey, what's up?"
    assert isinstance(first_content["annotations"], list)
    assert isinstance(first_content["logprobs"], list)

    # This used to fail when validated payloads retained ValidatorIterator fields.
    json.dumps(new_list)

def test_tool_call_item_to_input_item_keeps_payload_api_safe() -> None:
    agent = Agent(name="test", instructions="test")
    raw_item = ResponseFunctionToolCall(
        id="fc_1",
        call_id="call_1",
        name="my_tool",
        arguments="{}",
        type="function_call",
        status="completed",
    )
    item = ToolCallItem(
        agent=agent,
        raw_item=raw_item,
        title="My Tool",
        description="A helpful tool",
    )

    result = item.to_input_item()
    result_dict = cast(dict[str, Any], result)

    assert isinstance(result, dict)
    assert result_dict["type"] == "function_call"
    assert "title" not in result_dict
    assert "description" not in result_dict


# --- tests/test_local_shell_tool.py ---

async def test_local_shell_action_execute_invokes_executor() -> None:
    executor = RecordingLocalShellExecutor(output="test output")
    tool = LocalShellTool(executor=executor)

    action = LocalShellCallAction(
        command=["bash", "-c", "ls"],
        env={"TEST": "value"},
        type="exec",
        timeout_ms=5000,
        working_directory="/tmp",
    )
    tool_call = LocalShellCall(
        id="lsh_123",
        action=action,
        call_id="call_456",
        status="completed",
        type="local_shell_call",
    )

    tool_run = ToolRunLocalShellCall(tool_call=tool_call, local_shell_tool=tool)
    agent = Agent(name="test_agent", tools=[tool])
    context_wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)

    output_item = await LocalShellAction.execute(
        agent=agent,
        call=tool_run,
        hooks=RunHooks[Any](),
        context_wrapper=context_wrapper,
        config=RunConfig(),
    )

    assert len(executor.calls) == 1
    request = executor.calls[0]
    assert isinstance(request, LocalShellCommandRequest)
    assert request.ctx_wrapper is context_wrapper
    assert request.data is tool_call
    assert request.data.action.command == ["bash", "-c", "ls"]
    assert request.data.action.env == {"TEST": "value"}
    assert request.data.action.timeout_ms == 5000
    assert request.data.action.working_directory == "/tmp"

    assert isinstance(output_item, ToolCallOutputItem)
    assert output_item.agent is agent
    assert output_item.output == "test output"

    raw_item = output_item.raw_item
    assert isinstance(raw_item, dict)
    raw = cast(dict[str, Any], raw_item)
    assert raw["type"] == "local_shell_call_output"
    assert raw["call_id"] == "call_456"
    assert raw["output"] == "test output"


# --- tests/test_model_retry.py ---

async def test_retry_policies_any_merges_later_positive_metadata() -> None:
    raw_decision = retry_policies.any(
        retry_policies.network_error(),
        retry_policies.retry_after(),
    )(
        RetryPolicyContext(
            error=_connection_error(),
            attempt=1,
            max_retries=2,
            stream=False,
            normalized=ModelRetryNormalizedError(
                is_network_error=True,
                retry_after=1.75,
            ),
            provider_advice=ModelRetryAdvice(retry_after=1.75),
        )
    )
    decision = await raw_decision if asyncio.iscoroutine(raw_decision) else raw_decision

    assert isinstance(decision, RetryDecision)
    assert decision.retry is True
    assert decision.delay == 1.75

async def test_stream_response_with_retry_rejects_stateful_retry_without_replay_safety() -> None:
    attempts = 0

    async def rewind() -> None:
        raise AssertionError("Stateful streaming retry should not rewind when replay is vetoed")

    def get_stream() -> AsyncIterator[TResponseStreamEvent]:
        nonlocal attempts
        attempts += 1

        async def iterator() -> AsyncIterator[TResponseStreamEvent]:
            raise _connection_error()
            yield  # pragma: no cover

        return iterator()

    with pytest.raises(APIConnectionError):
        async for _event in stream_response_with_retry(
            get_stream=get_stream,
            rewind=rewind,
            retry_settings=ModelRetrySettings(
                max_retries=1,
                policy=retry_policies.provider_suggested(),
            ),
            get_retry_advice=lambda _request: ModelRetryAdvice(suggested=True),
            previous_response_id="resp_prev",
            conversation_id=None,
        ):
            pass

    assert attempts == 1

async def test_stream_response_with_retry_does_not_retry_after_output_event() -> None:
    attempts = 0

    async def rewind() -> None:
        raise AssertionError("Streaming retries should stop after output has been emitted")

    def get_stream() -> AsyncIterator[TResponseStreamEvent]:
        nonlocal attempts
        attempts += 1

        async def iterator() -> AsyncIterator[TResponseStreamEvent]:
            yield cast(TResponseStreamEvent, {"type": "response.output_item.added"})
            raise _connection_error()

        return iterator()

    with pytest.raises(APIConnectionError):
        async for _event in stream_response_with_retry(
            get_stream=get_stream,
            rewind=rewind,
            retry_settings=ModelRetrySettings(
                max_retries=1,
                policy=retry_policies.network_error(),
            ),
            get_retry_advice=lambda _request: None,
            previous_response_id=None,
            conversation_id=None,
        ):
            pass

    assert attempts == 1

async def test_stream_response_with_retry_closes_current_stream_when_consumer_stops_early() -> None:
    stream = _CloseTrackingStream(
        events=[
            cast(TResponseStreamEvent, {"type": "response.created"}),
            cast(TResponseStreamEvent, {"type": "response.in_progress"}),
        ]
    )

    async def rewind() -> None:
        raise AssertionError("Early consumer exit should not rewind state")

    outer_stream = cast(
        Any,
        stream_response_with_retry(
            get_stream=lambda: stream,
            rewind=rewind,
            retry_settings=ModelRetrySettings(
                max_retries=1,
                policy=retry_policies.network_error(),
            ),
            get_retry_advice=lambda _request: None,
            previous_response_id=None,
            conversation_id=None,
        ),
    )

    first_event = await outer_stream.__anext__()
    assert first_event == cast(TResponseStreamEvent, {"type": "response.created"})

    await outer_stream.aclose()

    assert stream.close_calls == 1


# --- tests/test_openai_chatcompletions.py ---

def test_get_client_disables_provider_managed_retries_on_runner_retry() -> None:
    class DummyChatCompletionsClient:
        def __init__(self) -> None:
            self.base_url = httpx.URL("https://api.openai.com/v1/")
            self.chat = type("ChatNamespace", (), {"completions": object()})()
            self.with_options_calls: list[dict[str, Any]] = []

        def with_options(self, **kwargs):
            self.with_options_calls.append(kwargs)
            return self

    client = DummyChatCompletionsClient()
    model = OpenAIChatCompletionsModel(model="gpt-4", openai_client=client)  # type: ignore[arg-type]

    assert cast(object, model._get_client()) is client
    with provider_managed_retries_disabled(True):
        assert cast(object, model._get_client()) is client

    assert client.with_options_calls == [{"max_retries": 0}]

def test_get_retry_advice_uses_openai_headers() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        headers={
            "x-should-retry": "true",
            "retry-after-ms": "500",
            "x-request-id": "req_123",
        },
        json={"error": {"code": "rate_limit"}},
    )
    error = APIStatusError(
        "rate limited", response=response, body={"error": {"code": "rate_limit"}}
    )
    model = OpenAIChatCompletionsModel(model="gpt-4", openai_client=cast(Any, object()))

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=False,
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.retry_after == 0.5
    assert advice.replay_safety == "safe"
    assert advice.normalized is not None
    assert advice.normalized.error_code == "rate_limit"
    assert advice.normalized.status_code == 429
    assert advice.normalized.request_id == "req_123"

def test_get_retry_advice_keeps_stateful_transport_failures_ambiguous() -> None:
    model = OpenAIChatCompletionsModel(model="gpt-4", openai_client=cast(Any, object()))
    error = APIConnectionError(
        message="connection error",
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=False,
            previous_response_id="resp_prev",
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety is None
    assert advice.normalized is not None
    assert advice.normalized.is_network_error is True

def test_get_retry_advice_marks_stateful_http_failures_replay_safe() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        json={"error": {"code": "rate_limit"}},
    )
    error = APIStatusError(
        "rate limited", response=response, body={"error": {"code": "rate_limit"}}
    )
    model = OpenAIChatCompletionsModel(model="gpt-4", openai_client=cast(Any, object()))

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=False,
            previous_response_id="resp_prev",
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety == "safe"
    assert advice.normalized is not None
    assert advice.normalized.status_code == 429

def test_get_client_disables_provider_managed_retries_when_requested() -> None:
    class DummyClient:
        def __init__(self):
            self.calls: list[dict[str, int]] = []

        def with_options(self, **kwargs):
            self.calls.append(kwargs)
            return "retry-client"

    client = DummyClient()
    model = OpenAIChatCompletionsModel(model="gpt-4", openai_client=cast(Any, client))

    assert cast(object, model._get_client()) is client

    with provider_managed_retries_disabled(True):
        assert cast(object, model._get_client()) == "retry-client"

    assert client.calls == [{"max_retries": 0}]


# --- tests/test_openai_chatcompletions_converter.py ---

def test_message_to_output_items_with_refusal():
    """
    Make sure a message with a refusal string produces a ResponseOutputMessage
    with a ResponseOutputRefusal content part.
    """
    msg = ChatCompletionMessage(role="assistant", refusal="I'm sorry")
    items = Converter.message_to_output_items(msg)
    assert len(items) == 1
    message_item = cast(ResponseOutputMessage, items[0])
    assert len(message_item.content) == 1
    refusal_part = cast(ResponseOutputRefusal, message_item.content[0])
    assert refusal_part.type == "refusal"
    assert refusal_part.refusal == "I'm sorry"

def test_items_to_messages_with_output_message_and_function_call():
    """
    Given a sequence of one ResponseOutputMessageParam followed by a
    ResponseFunctionToolCallParam, the converter should produce a single
    ChatCompletionAssistantMessageParam that includes both the assistant's
    textual content and a populated `tool_calls` reflecting the function call.
    """
    # Construct output message param dict with two content parts.
    output_text: ResponseOutputText = ResponseOutputText(
        text="Part 1",
        type="output_text",
        annotations=[],
        logprobs=[],
    )
    refusal: ResponseOutputRefusal = ResponseOutputRefusal(
        refusal="won't do that",
        type="refusal",
    )
    resp_msg: ResponseOutputMessage = ResponseOutputMessage(
        id="42",
        type="message",
        role="assistant",
        status="completed",
        content=[output_text, refusal],
    )
    # Construct a function call item dict (as if returned from model)
    func_item: ResponseFunctionToolCallParam = {
        "id": "99",
        "call_id": "abc",
        "name": "math",
        "arguments": "{}",
        "type": "function_call",
    }
    items: list[TResponseInputItem] = [
        resp_msg.model_dump(),  # type:ignore
        func_item,
    ]
    messages = Converter.items_to_messages(items)
    # Should return a single assistant message
    assert len(messages) == 1
    assistant = messages[0]
    assert assistant["role"] == "assistant"
    # Content combines text portions of the output message
    assert "content" in assistant
    assert assistant["content"] == "Part 1"
    # Refusal in output message should be represented in assistant message
    assert "refusal" in assistant
    assert assistant["refusal"] == refusal.refusal
    # Tool calls list should contain one ChatCompletionMessageFunctionToolCall dict
    tool_calls = assistant.get("tool_calls")
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "math"
    assert tool_call["function"]["arguments"] == "{}"


# --- tests/test_openai_responses.py ---

async def test_user_agent_header_responses(override_ua: str | None):
    called_kwargs: dict[str, Any] = {}
    expected_ua = override_ua or f"Agents/Python {__version__}"

    class DummyStream:
        def __aiter__(self):
            async def gen():
                yield ResponseCompletedEvent(
                    type="response.completed",
                    response=get_response_obj([]),
                    sequence_number=0,
                )

            return gen()

    class DummyResponses:
        async def create(self, **kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return DummyStream()

    class DummyResponsesClient:
        def __init__(self):
            self.responses = DummyResponses()

    model = OpenAIResponsesModel(model="gpt-4", openai_client=DummyResponsesClient())  # type: ignore

    if override_ua is not None:
        token = RESP_HEADERS.set({"User-Agent": override_ua})
    else:
        token = None

    try:
        stream = model.stream_response(
            system_instructions=None,
            input="hi",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
        )
        async for _ in stream:
            pass
    finally:
        if token is not None:
            RESP_HEADERS.reset(token)

    assert "extra_headers" in called_kwargs
    assert called_kwargs["extra_headers"]["User-Agent"] == expected_ua

async def test_fetch_response_stream_attaches_request_id_to_terminal_response():
    class DummyHTTPStream:
        def __init__(self):
            self._yielded = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._yielded:
                raise StopAsyncIteration
            self._yielded = True
            return ResponseCompletedEvent(
                type="response.completed",
                response=get_response_obj([], response_id="resp-stream-request-id"),
                sequence_number=0,
            )

    inner_stream = DummyHTTPStream()

    class DummyAPIResponse:
        def __init__(self):
            self.request_id = "req_stream_123"
            self.close_calls = 0
            self.parse_calls = 0

        async def parse(self):
            self.parse_calls += 1
            return inner_stream

        async def close(self) -> None:
            self.close_calls += 1

    api_response = DummyAPIResponse()
    aexit_calls: list[tuple[Any, Any, Any]] = []

    class DummyStreamingContextManager:
        async def __aenter__(self):
            return api_response

        async def __aexit__(self, exc_type, exc, tb):
            aexit_calls.append((exc_type, exc, tb))
            await api_response.close()
            return False

    class DummyResponses:
        def __init__(self):
            self.with_streaming_response = SimpleNamespace(create=self.create_streaming)

        def create_streaming(self, **kwargs):
            return DummyStreamingContextManager()

    class DummyResponsesClient:
        def __init__(self):
            self.responses = DummyResponses()

    model = OpenAIResponsesModel(model="gpt-4", openai_client=DummyResponsesClient())  # type: ignore[arg-type]

    stream = await model._fetch_response(
        system_instructions=None,
        input="hi",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        previous_response_id=None,
        conversation_id=None,
        stream=True,
    )

    stream_agen = cast(Any, stream)
    event = await stream_agen.__anext__()

    assert getattr(stream, "request_id", None) == "req_stream_123"
    assert getattr(event.response, "_request_id", None) == "req_stream_123"

    with pytest.raises(StopAsyncIteration):
        await stream_agen.__anext__()

    assert api_response.parse_calls == 1
    assert api_response.close_calls == 1
    assert aexit_calls == [(None, None, None)]

async def test_fetch_response_stream_parse_failure_exits_streaming_context():
    parse_error = RuntimeError("parse failed")
    aexit_calls: list[tuple[Any, Any, Any]] = []

    class DummyAPIResponse:
        request_id = "req_stream_123"

        async def parse(self):
            raise parse_error

    api_response = DummyAPIResponse()

    class DummyStreamingContextManager:
        async def __aenter__(self):
            return api_response

        async def __aexit__(self, exc_type, exc, tb):
            aexit_calls.append((exc_type, exc, tb))
            return False

    class DummyResponses:
        def __init__(self):
            self.with_streaming_response = SimpleNamespace(create=self.create_streaming)

        def create_streaming(self, **kwargs):
            return DummyStreamingContextManager()

    class DummyResponsesClient:
        def __init__(self):
            self.responses = DummyResponses()

    model = OpenAIResponsesModel(model="gpt-4", openai_client=DummyResponsesClient())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="parse failed"):
        await model._fetch_response(
            system_instructions=None,
            input="hi",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            previous_response_id=None,
            conversation_id=None,
            stream=True,
        )

    assert len(aexit_calls) == 1
    exc_type, exc, tb = aexit_calls[0]
    assert exc_type is RuntimeError
    assert exc is parse_error
    assert tb is not None

async def test_fetch_response_stream_without_request_id_still_returns_events():
    class DummyHTTPStream:
        def __init__(self):
            self._yielded = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._yielded:
                raise StopAsyncIteration
            self._yielded = True
            return ResponseCompletedEvent(
                type="response.completed",
                response=get_response_obj([], response_id="resp-stream-request-id"),
                sequence_number=0,
            )

    inner_stream = DummyHTTPStream()
    aexit_calls: list[tuple[Any, Any, Any]] = []

    class DummyAPIResponse:
        def __init__(self):
            self.close_calls = 0
            self.parse_calls = 0

        async def parse(self):
            self.parse_calls += 1
            return inner_stream

        async def close(self) -> None:
            self.close_calls += 1

    api_response = DummyAPIResponse()

    class DummyStreamingContextManager:
        async def __aenter__(self):
            return api_response

        async def __aexit__(self, exc_type, exc, tb):
            aexit_calls.append((exc_type, exc, tb))
            await api_response.close()
            return False

    class DummyResponses:
        def __init__(self):
            self.with_streaming_response = SimpleNamespace(create=self.create_streaming)

        def create_streaming(self, **kwargs):
            return DummyStreamingContextManager()

    class DummyResponsesClient:
        def __init__(self):
            self.responses = DummyResponses()

    model = OpenAIResponsesModel(model="gpt-4", openai_client=DummyResponsesClient())  # type: ignore[arg-type]

    stream = await model._fetch_response(
        system_instructions=None,
        input="hi",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        previous_response_id=None,
        conversation_id=None,
        stream=True,
    )

    stream_agen = cast(Any, stream)
    event = await stream_agen.__anext__()

    assert getattr(stream, "request_id", None) is None
    assert getattr(event.response, "_request_id", None) is None

    with pytest.raises(StopAsyncIteration):
        await stream_agen.__anext__()

    assert api_response.parse_calls == 1
    assert api_response.close_calls == 1
    assert aexit_calls == [(None, None, None)]

async def test_stream_response_ignores_streaming_context_exit_failure_after_terminal_event():
    class DummyHTTPStream:
        def __init__(self):
            self._yielded = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._yielded:
                raise StopAsyncIteration
            self._yielded = True
            return ResponseCompletedEvent(
                type="response.completed",
                response=get_response_obj([], response_id="resp-stream-request-id"),
                sequence_number=0,
            )

    inner_stream = DummyHTTPStream()
    aexit_calls: list[tuple[Any, Any, Any]] = []

    class DummyAPIResponse:
        request_id = "req_stream_123"

        async def parse(self):
            return inner_stream

    api_response = DummyAPIResponse()

    class DummyStreamingContextManager:
        async def __aenter__(self):
            return api_response

        async def __aexit__(self, exc_type, exc, tb):
            aexit_calls.append((exc_type, exc, tb))
            raise RuntimeError("stream context exit failed")

    class DummyResponses:
        def __init__(self):
            self.with_streaming_response = SimpleNamespace(create=self.create_streaming)

        def create_streaming(self, **kwargs):
            return DummyStreamingContextManager()

    class DummyResponsesClient:
        def __init__(self):
            self.responses = DummyResponses()

    model = OpenAIResponsesModel(model="gpt-4", openai_client=DummyResponsesClient())  # type: ignore[arg-type]

    events: list[ResponseCompletedEvent] = []
    async for event in model.stream_response(
        system_instructions=None,
        input="hi",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
    ):
        assert isinstance(event, ResponseCompletedEvent)
        events.append(event)

    assert len(events) == 1
    assert aexit_calls == [(None, None, None)]

def test_build_response_create_kwargs_rejects_duplicate_extra_args_keys():
    client = DummyWSClient()
    model = OpenAIResponsesModel(model="gpt-4", openai_client=client)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="multiple values.*stream"):
        model._build_response_create_kwargs(
            system_instructions=None,
            input="hi",
            model_settings=ModelSettings(extra_args={"stream": False}),
            tools=[],
            output_schema=None,
            handoffs=[],
            previous_response_id=None,
            conversation_id=None,
            stream=True,
            prompt=None,
        )

def test_build_response_create_kwargs_preserves_unknown_response_include_values():
    client = DummyWSClient()
    model = OpenAIResponsesModel(model="gpt-4", openai_client=client)  # type: ignore[arg-type]

    kwargs = model._build_response_create_kwargs(
        system_instructions=None,
        input="hi",
        model_settings=ModelSettings(response_include=["response.future_flag"]),
        tools=[],
        output_schema=None,
        handoffs=[],
        previous_response_id=None,
        conversation_id=None,
        stream=False,
        prompt=None,
    )

    assert kwargs["include"] == ["response.future_flag"]

async def test_websocket_model_prepare_websocket_request_filters_omit_from_extra_body():
    client = DummyWSClient()
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=client)  # type: ignore[arg-type]

    frame, _ws_url, _headers = await model._prepare_websocket_request(
        {
            "model": "gpt-4",
            "input": "hi",
            "stream": True,
            "extra_body": {"keep": "value", "drop": omit},
        }
    )

    assert frame["type"] == "response.create"
    assert frame["keep"] == "value"
    assert "drop" not in frame

async def test_websocket_model_prepare_websocket_request_ignores_top_level_extra_body_sentinels(
    extra_body,
):
    client = DummyWSClient()
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=client)  # type: ignore[arg-type]

    frame, _ws_url, _headers = await model._prepare_websocket_request(
        {
            "model": "gpt-4",
            "input": "hi",
            "stream": True,
            "extra_body": extra_body,
        }
    )

    assert frame["type"] == "response.create"
    assert frame["stream"] is True
    assert frame["model"] == "gpt-4"
    assert frame["input"] == "hi"

async def test_websocket_model_prepare_websocket_request_preserves_envelope_fields():
    client = DummyWSClient()
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=client)  # type: ignore[arg-type]

    frame, _ws_url, _headers = await model._prepare_websocket_request(
        {
            "model": "gpt-4",
            "input": "hi",
            "stream": True,
            "extra_body": {
                "type": "not-response-create",
                "stream": False,
                "custom": "value",
            },
        }
    )

    assert frame["type"] == "response.create"
    assert frame["stream"] is True
    assert frame["custom"] == "value"

async def test_websocket_model_prepare_websocket_request_strips_client_timeout_kwarg():
    client = DummyWSClient()
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=client)  # type: ignore[arg-type]

    frame, _ws_url, _headers = await model._prepare_websocket_request(
        {
            "model": "gpt-4",
            "input": "hi",
            "stream": True,
            "timeout": 30.0,
            "metadata": {"request_id": "123"},
        }
    )

    assert frame["type"] == "response.create"
    assert frame["metadata"] == {"request_id": "123"}
    assert "timeout" not in frame

async def test_websocket_model_prepare_websocket_request_skips_not_given_values():
    client = DummyWSClient()
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=client)  # type: ignore[arg-type]

    frame, _ws_url, _headers = await model._prepare_websocket_request(
        {
            "model": "gpt-4",
            "input": "hi",
            "stream": True,
            "user": NOT_GIVEN,
            "stream_options": NOT_GIVEN,
            "extra_body": {
                "metadata": {"request_id": "123"},
                "optional_field": NOT_GIVEN,
            },
        }
    )

    assert frame["type"] == "response.create"
    assert frame["stream"] is True
    assert frame["metadata"] == {"request_id": "123"}
    assert "user" not in frame
    assert "stream_options" not in frame
    assert "optional_field" not in frame
    json.dumps(frame)

async def test_websocket_model_prepare_websocket_request_omit_removes_inherited_header():
    client = DummyWSClient()
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=client)  # type: ignore[arg-type]

    _frame, _ws_url, headers = await model._prepare_websocket_request(
        {
            "model": "gpt-4",
            "input": "hi",
            "stream": True,
            "extra_headers": {"User-Agent": omit},
        }
    )

    assert "Authorization" in headers
    assert "User-Agent" not in headers

async def test_websocket_model_prepare_websocket_request_replaces_header_case_insensitively():
    client = DummyWSClient()
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=client)  # type: ignore[arg-type]

    _frame, _ws_url, headers = await model._prepare_websocket_request(
        {
            "model": "gpt-4",
            "input": "hi",
            "stream": True,
            "extra_headers": {
                "authorization": "Bearer override-key",
                "user-agent": "Custom UA",
            },
        }
    )

    assert headers["authorization"] == "Bearer override-key"
    assert headers["user-agent"] == "Custom UA"
    assert "Authorization" not in headers
    assert "User-Agent" not in headers

async def test_websocket_model_prepare_websocket_request_skips_not_given_header_values():
    client = DummyWSClient()
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=client)  # type: ignore[arg-type]

    _frame, _ws_url, headers = await model._prepare_websocket_request(
        {
            "model": "gpt-4",
            "input": "hi",
            "stream": True,
            "extra_headers": {
                "Authorization": NOT_GIVEN,
                "X-Optional": NOT_GIVEN,
            },
        }
    )

    assert headers["Authorization"] == "Bearer test-key"
    assert "X-Optional" not in headers
    assert "NOT_GIVEN" not in headers.values()

async def test_websocket_model_close_falls_back_to_transport_abort_on_close_error():
    client = DummyWSClient()
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=client)  # type: ignore[arg-type]

    class DummyTransport:
        def __init__(self):
            self.abort_calls = 0

        def abort(self):
            self.abort_calls += 1

    class FailingWSConnection:
        def __init__(self):
            self.transport = DummyTransport()

        async def close(self):
            raise RuntimeError("attached to a different loop")

    ws = FailingWSConnection()
    model._ws_connection = ws
    model._ws_connection_identity = ("wss://example.test", (("authorization", "x"),))

    await model.close()

    assert ws.transport.abort_calls == 1
    assert model._ws_connection is None
    assert model._ws_connection_identity is None

def test_get_retry_advice_uses_openai_headers() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        429,
        request=request,
        headers={
            "x-should-retry": "true",
            "retry-after-ms": "250",
            "x-request-id": "req_456",
        },
        json={"error": {"code": "rate_limit"}},
    )
    error = RateLimitError(
        "rate limited", response=response, body={"error": {"code": "rate_limit"}}
    )
    model = OpenAIResponsesModel(model="gpt-4", openai_client=cast(Any, object()))

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=False,
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.retry_after == 0.25
    assert advice.replay_safety == "safe"
    assert advice.normalized is not None
    assert advice.normalized.error_code == "rate_limit"
    assert advice.normalized.status_code == 429
    assert advice.normalized.request_id == "req_456"

def test_get_retry_advice_keeps_stateful_transport_failures_ambiguous() -> None:
    model = OpenAIResponsesModel(model="gpt-4", openai_client=cast(Any, object()))
    error = APIConnectionError(
        message="connection error",
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=False,
            previous_response_id="resp_prev",
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety is None
    assert advice.normalized is not None
    assert advice.normalized.is_network_error is True

def test_get_retry_advice_marks_stateful_http_failures_replay_safe() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        429,
        request=request,
        json={"error": {"code": "rate_limit"}},
    )
    error = RateLimitError(
        "rate limited", response=response, body={"error": {"code": "rate_limit"}}
    )
    model = OpenAIResponsesModel(model="gpt-4", openai_client=cast(Any, object()))

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=False,
            previous_response_id="resp_prev",
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety == "safe"
    assert advice.normalized is not None
    assert advice.normalized.status_code == 429

def test_get_retry_advice_keeps_stateless_transport_failures_retryable() -> None:
    model = OpenAIResponsesModel(model="gpt-4", openai_client=cast(Any, object()))
    error = APIConnectionError(
        message="connection error",
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=False,
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety is None
    assert advice.normalized is not None
    assert advice.normalized.is_network_error is True

def test_websocket_get_retry_advice_marks_ambiguous_replay_unsafe() -> None:
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=cast(Any, DummyWSClient()))
    error = RuntimeError("Responses websocket connection closed before a terminal response event.")
    error.__cause__ = _connection_closed_error("peer closed after request send")

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=True,
            previous_response_id="resp_prev",
        )
    )

    assert advice is not None
    assert advice.suggested is False
    assert advice.replay_safety == "unsafe"

def test_websocket_get_retry_advice_allows_stateless_ambiguous_disconnect_retry() -> None:
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=cast(Any, DummyWSClient()))
    error = RuntimeError("Responses websocket connection closed before a terminal response event.")
    error.__cause__ = _connection_closed_error("peer closed after request send")

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=True,
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety is None

def test_websocket_get_retry_advice_keeps_wrapped_pre_send_disconnect_safe() -> None:
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=cast(Any, DummyWSClient()))
    error = RuntimeError(
        "Responses websocket connection closed before any response events were received."
    )
    setattr(error, "_openai_agents_ws_replay_safety", "safe")  # noqa: B010
    error.__cause__ = _connection_closed_error("peer closed before request send")

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=True,
            previous_response_id="resp_prev",
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety == "safe"

def test_websocket_get_retry_advice_allows_stateless_wrapped_post_send_disconnect_retry() -> None:
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=cast(Any, DummyWSClient()))
    error = RuntimeError(
        "Responses websocket connection closed before any response events were received."
    )
    setattr(error, "_openai_agents_ws_replay_safety", "unsafe")  # noqa: B010
    error.__cause__ = _connection_closed_error("peer closed after request send")

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=True,
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety is None

def test_websocket_get_retry_advice_allows_stateless_nonstream_post_send_retry() -> None:
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=cast(Any, DummyWSClient()))
    error = RuntimeError(
        "Responses websocket connection closed before any response events were received."
    )
    setattr(error, "_openai_agents_ws_replay_safety", "unsafe")  # noqa: B010
    error.__cause__ = _connection_closed_error("peer closed after request send")

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=False,
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety is None

def test_websocket_get_retry_advice_marks_wrapped_post_send_disconnect_unsafe() -> None:
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=cast(Any, DummyWSClient()))
    error = RuntimeError(
        "Responses websocket connection closed before any response events were received."
    )
    setattr(error, "_openai_agents_ws_replay_safety", "unsafe")  # noqa: B010
    error.__cause__ = _connection_closed_error("peer closed after request send")

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=True,
            previous_response_id="resp_prev",
        )
    )

    assert advice is not None
    assert advice.suggested is False
    assert advice.replay_safety == "unsafe"

def test_websocket_get_retry_advice_marks_partial_nonstream_failure_unsafe() -> None:
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=cast(Any, DummyWSClient()))
    error = TimeoutError("Responses websocket receive timed out after 5.0 seconds.")
    setattr(error, "_openai_agents_ws_replay_safety", "unsafe")  # noqa: B010
    setattr(error, "_openai_agents_ws_response_started", True)  # noqa: B010

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=False,
        )
    )

    assert advice is not None
    assert advice.suggested is False
    assert advice.replay_safety == "unsafe"

def test_websocket_get_retry_advice_marks_connect_timeout_replay_safe() -> None:
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=cast(Any, DummyWSClient()))
    error = TimeoutError("Responses websocket connect timed out after 5.0 seconds.")

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=True,
            previous_response_id="resp_prev",
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety == "safe"

def test_websocket_get_retry_advice_marks_request_lock_timeout_replay_safe() -> None:
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=cast(Any, DummyWSClient()))
    error = TimeoutError("Responses websocket request lock wait timed out after 5.0 seconds.")

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=False,
            previous_response_id="resp_prev",
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety == "safe"

def test_websocket_get_retry_advice_marks_stateful_receive_timeout_unsafe() -> None:
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=cast(Any, DummyWSClient()))
    error = TimeoutError("Responses websocket receive timed out after 5.0 seconds.")

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=True,
            previous_response_id="resp_prev",
        )
    )

    assert advice is not None
    assert advice.suggested is False
    assert advice.replay_safety == "unsafe"

def test_websocket_get_retry_advice_allows_stateless_receive_timeout_retry() -> None:
    model = OpenAIResponsesWSModel(model="gpt-4", openai_client=cast(Any, DummyWSClient()))
    error = TimeoutError("Responses websocket receive timed out after 5.0 seconds.")

    advice = model.get_retry_advice(
        ModelRetryAdviceRequest(
            error=error,
            attempt=1,
            stream=True,
        )
    )

    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety is None


# --- tests/test_result_cast.py ---

def test_run_result_streaming_supports_pydantic_model_rebuild() -> None:
    class StreamingRunContainer(BaseModel):
        query_id: str
        run_stream: RunResultStreaming | None

        model_config = ConfigDict(arbitrary_types_allowed=True)

    StreamingRunContainer.model_rebuild()

def test_result_cast_typechecks():
    """Correct casts should work fine."""
    result = create_run_result(1)
    assert result.final_output_as(int) == 1

    result = create_run_result("test")
    assert result.final_output_as(str) == "test"

    result = create_run_result(Foo(bar=1))
    assert result.final_output_as(Foo) == Foo(bar=1)

def test_bad_cast_doesnt_raise():
    """Bad casts shouldn't error unless we ask for it."""
    result = create_run_result(1)
    result.final_output_as(str)

    result = create_run_result("test")
    result.final_output_as(Foo)

def test_bad_cast_with_param_raises():
    """Bad casts should raise a TypeError when we ask for it."""
    result = create_run_result(1)
    with pytest.raises(TypeError):
        result.final_output_as(str, raise_if_incorrect_type=True)

    result = create_run_result("test")
    with pytest.raises(TypeError):
        result.final_output_as(Foo, raise_if_incorrect_type=True)

    result = create_run_result(Foo(bar=1))
    with pytest.raises(TypeError):
        result.final_output_as(int, raise_if_incorrect_type=True)

def test_run_result_release_agents_breaks_strong_refs() -> None:
    message = _create_message("hello")
    agent = Agent(name="leak-test-agent")
    item = MessageOutputItem(agent=agent, raw_item=message)
    result = create_run_result(None, new_items=[item], last_agent=agent)
    assert item.agent is not None
    assert item.agent.name == "leak-test-agent"

    agent_ref = weakref.ref(agent)
    result.release_agents()
    del agent
    gc.collect()

    assert agent_ref() is None
    assert item.agent is None
    with pytest.raises(AgentsException):
        _ = result.last_agent

def test_run_item_retains_agent_when_result_is_garbage_collected() -> None:
    def build_item() -> tuple[MessageOutputItem, weakref.ReferenceType[RunResult]]:
        message = _create_message("persist")
        agent = Agent(name="persisted-agent")
        item = MessageOutputItem(agent=agent, raw_item=message)
        result = create_run_result(None, new_items=[item], last_agent=agent)
        return item, weakref.ref(result)

    item, result_ref = build_item()
    gc.collect()

    assert result_ref() is None
    assert item.agent is not None
    assert item.agent.name == "persisted-agent"

def test_run_result_repr_and_asdict_after_release_agents() -> None:
    agent = Agent(name="repr-result-agent")
    result = create_run_result(None, last_agent=agent)

    result.release_agents()

    text = repr(result)
    assert "RunResult" in text

    serialized = dataclasses.asdict(result)
    assert serialized["_last_agent"] is None

def test_run_result_release_agents_without_releasing_new_items() -> None:
    message = _create_message("keep")
    item_agent = Agent(name="item-agent")
    last_agent = Agent(name="last-agent")
    item = MessageOutputItem(agent=item_agent, raw_item=message)
    result = create_run_result(None, new_items=[item], last_agent=last_agent)

    result.release_agents(release_new_items=False)

    assert item.agent is item_agent

    last_agent_ref = weakref.ref(last_agent)
    del last_agent
    gc.collect()

    assert last_agent_ref() is None
    with pytest.raises(AgentsException):
        _ = result.last_agent

def test_run_result_release_agents_is_idempotent() -> None:
    message = _create_message("idempotent")
    agent = Agent(name="idempotent-agent")
    item = MessageOutputItem(agent=agent, raw_item=message)
    result = RunResult(
        input="test",
        new_items=[item],
        raw_responses=[],
        final_output=None,
        input_guardrail_results=[],
        output_guardrail_results=[],
        tool_input_guardrail_results=[],
        tool_output_guardrail_results=[],
        _last_agent=agent,
        context_wrapper=RunContextWrapper(context=None),
        interruptions=[],
    )

    result.release_agents()
    result.release_agents()

    assert item.agent is agent

    agent_ref = weakref.ref(agent)
    del agent
    gc.collect()

    assert agent_ref() is None
    assert item.agent is None
    with pytest.raises(AgentsException):
        _ = result.last_agent

def test_run_result_streaming_release_agents_releases_current_agent() -> None:
    agent = Agent(name="streaming-agent")
    streaming_result = RunResultStreaming(
        input="stream",
        new_items=[],
        raw_responses=[],
        final_output=None,
        input_guardrail_results=[],
        output_guardrail_results=[],
        tool_input_guardrail_results=[],
        tool_output_guardrail_results=[],
        context_wrapper=RunContextWrapper(context=None),
        current_agent=agent,
        current_turn=0,
        max_turns=1,
        _current_agent_output_schema=None,
        trace=None,
        interruptions=[],
    )

    streaming_result.release_agents(release_new_items=False)

    agent_ref = weakref.ref(agent)
    del agent
    gc.collect()

    assert agent_ref() is None
    with pytest.raises(AgentsException):
        _ = streaming_result.last_agent

def test_run_result_agent_tool_invocation_returns_none_for_plain_context() -> None:
    result = create_run_result("ok")

    assert result.agent_tool_invocation is None

def test_run_result_agent_tool_invocation_returns_immutable_metadata() -> None:
    tool_ctx = ToolContext(
        context=None,
        tool_name="my_tool",
        tool_call_id="call_xyz",
        tool_arguments="{}",
    )
    result = RunResult(
        input="test",
        new_items=[],
        raw_responses=[],
        final_output="ok",
        input_guardrail_results=[],
        output_guardrail_results=[],
        tool_input_guardrail_results=[],
        tool_output_guardrail_results=[],
        _last_agent=Agent(name="test"),
        context_wrapper=tool_ctx,
        interruptions=[],
    )

    assert result.agent_tool_invocation == AgentToolInvocation(
        tool_name="my_tool",
        tool_call_id="call_xyz",
        tool_arguments="{}",
    )

    invocation = result.agent_tool_invocation
    assert invocation is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast(Any, invocation).tool_name = "other"

def test_run_result_streaming_agent_tool_invocation_returns_metadata() -> None:
    agent = Agent(name="streaming-tool-agent")
    tool_ctx = ToolContext(
        context=None,
        tool_name="stream_tool",
        tool_call_id="call_stream",
        tool_arguments='{"input":"stream"}',
    )
    result = RunResultStreaming(
        input="stream",
        new_items=[],
        raw_responses=[],
        final_output="done",
        input_guardrail_results=[],
        output_guardrail_results=[],
        tool_input_guardrail_results=[],
        tool_output_guardrail_results=[],
        context_wrapper=tool_ctx,
        current_agent=agent,
        current_turn=0,
        max_turns=1,
        _current_agent_output_schema=None,
        trace=None,
        interruptions=[],
    )

    assert result.agent_tool_invocation == AgentToolInvocation(
        tool_name="stream_tool",
        tool_call_id="call_stream",
        tool_arguments='{"input":"stream"}',
    )


# --- tests/test_run_impl_resume_paths.py ---

async def test_resumed_approval_does_not_duplicate_session_items() -> None:
    async def test_tool() -> str:
        return "tool_result"

    tool = function_tool(test_tool, name_override="test_tool", needs_approval=True)
    model, agent = make_model_and_agent(name="test", tools=[tool])
    session = SimpleListSession()

    queue_function_call_and_text(
        model,
        get_function_tool_call("test_tool", json.dumps({}), call_id="call-resume"),
        followup=[get_text_message("done")],
    )

    first = await Runner.run(agent, input="Use test_tool", session=session)
    assert first.interruptions
    state = first.to_state()
    state.approve(first.interruptions[0])

    resumed = await Runner.run(agent, state, session=session)
    assert resumed.final_output == "done"

    saved_items = await session.get_items()
    call_count = sum(
        1
        for item in saved_items
        if isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("call_id") == "call-resume"
    )
    output_count = sum(
        1
        for item in saved_items
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") == "call-resume"
    )

    assert call_count == 1
    assert output_count == 1


# --- tests/test_run_internal_items.py ---

def test_drop_orphan_function_calls_preserves_non_mapping_entries() -> None:
    payload: list[Any] = [
        cast(TResponseInputItem, "plain-text-input"),
        cast(TResponseInputItem, {"type": "message", "role": "user", "content": "hello"}),
        cast(
            TResponseInputItem,
            {
                "type": "function_call",
                "call_id": "orphan_call",
                "name": "orphan",
                "arguments": "{}",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "function_call",
                "call_id": "paired_call",
                "name": "paired",
                "arguments": "{}",
            },
        ),
        cast(
            TResponseInputItem,
            {"type": "function_call_output", "call_id": "paired_call", "output": "ok"},
        ),
        cast(TResponseInputItem, {"call_id": "not-a-tool-call"}),
    ]

    filtered = run_items.drop_orphan_function_calls(cast(list[TResponseInputItem], payload))
    filtered_values = cast(list[Any], filtered)
    assert "plain-text-input" in filtered_values
    assert cast(dict[str, Any], filtered[1])["type"] == "message"
    assert any(
        isinstance(entry, dict)
        and entry.get("type") == "function_call"
        and entry.get("call_id") == "paired_call"
        for entry in filtered
    )
    assert not any(
        isinstance(entry, dict)
        and entry.get("type") == "function_call"
        and entry.get("call_id") == "orphan_call"
        for entry in filtered
    )

def test_drop_orphan_function_calls_handles_tool_search_calls() -> None:
    payload: list[Any] = [
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": "tool_search_orphan",
                "arguments": {"query": "orphan"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": "tool_search_keep",
                "arguments": {"query": "keep"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_output",
                "call_id": "tool_search_keep",
                "execution": "server",
                "status": "completed",
                "tools": [],
            },
        ),
    ]

    filtered = run_items.drop_orphan_function_calls(cast(list[TResponseInputItem], payload))

    assert any(
        isinstance(entry, dict)
        and entry.get("type") == "tool_search_call"
        and entry.get("call_id") == "tool_search_keep"
        for entry in filtered
    )
    assert not any(
        isinstance(entry, dict)
        and entry.get("type") == "tool_search_call"
        and entry.get("call_id") == "tool_search_orphan"
        for entry in filtered
    )

def test_drop_orphan_function_calls_preserves_hosted_tool_search_pairs_without_call_ids() -> None:
    payload: list[Any] = [
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": None,
                "arguments": {"query": "keep"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_output",
                "call_id": None,
                "execution": "server",
                "status": "completed",
                "tools": [],
            },
        ),
    ]

    filtered = run_items.drop_orphan_function_calls(cast(list[TResponseInputItem], payload))

    assert len(filtered) == 2
    assert cast(dict[str, Any], filtered[0])["type"] == "tool_search_call"
    assert cast(dict[str, Any], filtered[1])["type"] == "tool_search_output"

def test_drop_orphan_function_calls_matches_latest_anonymous_tool_search_call() -> None:
    payload: list[Any] = [
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": None,
                "arguments": {"query": "orphan"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": None,
                "arguments": {"query": "paired"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_output",
                "call_id": None,
                "execution": "server",
                "status": "completed",
                "tools": [],
            },
        ),
    ]

    filtered = run_items.drop_orphan_function_calls(cast(list[TResponseInputItem], payload))

    assert [cast(dict[str, Any], item)["type"] for item in filtered] == [
        "tool_search_call",
        "tool_search_output",
    ]
    assert cast(dict[str, Any], filtered[0])["arguments"] == {"query": "paired"}

def test_drop_orphan_function_calls_does_not_pair_named_tool_search_with_anonymous_output() -> None:
    payload: list[Any] = [
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": "orphan_search",
                "arguments": {"query": "keep"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_output",
                "call_id": None,
                "execution": "server",
                "status": "completed",
                "tools": [],
            },
        ),
    ]

    filtered = run_items.drop_orphan_function_calls(cast(list[TResponseInputItem], payload))

    assert [cast(dict[str, Any], item)["type"] for item in filtered] == ["tool_search_output"]

def test_normalize_and_ensure_input_item_format_keep_non_dict_entries() -> None:
    item = cast(TResponseInputItem, "raw-item")
    assert run_items.ensure_input_item_format(item) == item
    assert run_items.normalize_input_items_for_api([item]) == [item]

def test_extract_mcp_request_id_supports_dicts_and_objects() -> None:
    assert (
        run_items.extract_mcp_request_id(
            {"provider_data": {"id": "provider-id"}, "id": "fallback-id"}
        )
        == "provider-id"
    )
    assert run_items.extract_mcp_request_id({"call_id": "call-id"}) == "call-id"

    class _WithProviderData:
        provider_data = {"id": "from-provider"}

    assert run_items.extract_mcp_request_id(_WithProviderData()) == "from-provider"

    class _BrokenObject:
        @property
        def provider_data(self) -> dict[str, Any]:
            raise RuntimeError("boom")

        def __getattr__(self, _name: str) -> Any:
            raise RuntimeError("boom")

    assert run_items.extract_mcp_request_id(_BrokenObject()) is None

def test_extract_mcp_request_id_from_run_variants() -> None:
    class _Run:
        def __init__(self, request_item: Any = None, requestItem: Any = None) -> None:
            self.request_item = request_item
            self.requestItem = requestItem

    class _RequestObject:
        provider_data = {"id": "provider-object"}
        id = "object-id"
        call_id = "object-call-id"

    assert (
        run_items.extract_mcp_request_id_from_run(
            _Run(request_item={"provider_data": {"id": "provider-dict"}, "id": "fallback"})
        )
        == "provider-dict"
    )
    assert (
        run_items.extract_mcp_request_id_from_run(_Run(request_item={"id": "dict-id"})) == "dict-id"
    )
    assert (
        run_items.extract_mcp_request_id_from_run(_Run(request_item=_RequestObject()))
        == "provider-object"
    )
    assert (
        run_items.extract_mcp_request_id_from_run(_Run(requestItem={"call_id": "camel-call"}))
        == "camel-call"
    )

def test_run_item_to_input_item_preserves_reasoning_item_ids_by_default() -> None:
    agent = Agent(name="A")
    reasoning = ReasoningItem(
        agent=agent,
        raw_item=ResponseReasoningItem(
            type="reasoning",
            id="rs_123",
            summary=[],
        ),
    )

    result = run_items.run_item_to_input_item(reasoning)

    assert isinstance(result, dict)
    assert result.get("type") == "reasoning"
    assert result.get("id") == "rs_123"

def test_run_item_to_input_item_omits_reasoning_item_ids_when_configured() -> None:
    agent = Agent(name="A")
    reasoning = ReasoningItem(
        agent=agent,
        raw_item=ResponseReasoningItem(
            type="reasoning",
            id="rs_456",
            summary=[],
        ),
    )

    result = run_items.run_item_to_input_item(reasoning, "omit")

    assert isinstance(result, dict)
    assert result.get("type") == "reasoning"
    assert "id" not in result

def test_run_item_to_input_item_preserves_tool_search_items() -> None:
    agent = Agent(name="A")
    tool_search_call = ToolSearchCallItem(
        agent=agent,
        raw_item={"type": "tool_search_call", "queries": [{"search_term": "profile"}]},
    )
    tool_search_output = ToolSearchOutputItem(
        agent=agent,
        raw_item={"type": "tool_search_output", "results": [{"text": "Customer profile"}]},
    )

    converted_call = run_items.run_item_to_input_item(tool_search_call)
    converted_output = run_items.run_item_to_input_item(tool_search_output)

    assert isinstance(converted_call, dict)
    assert converted_call["type"] == "tool_search_call"
    assert isinstance(converted_output, dict)
    assert converted_output["type"] == "tool_search_output"

def test_run_item_to_input_item_strips_tool_search_created_by() -> None:
    agent = Agent(name="A")
    tool_search_call = ToolSearchCallItem(
        agent=agent,
        raw_item=ResponseToolSearchCall(
            id="tsc_123",
            type="tool_search_call",
            arguments={"query": "profile"},
            execution="client",
            status="completed",
            created_by="server",
        ),
    )
    tool_search_output = ToolSearchOutputItem(
        agent=agent,
        raw_item=ResponseToolSearchOutputItem(
            id="tso_123",
            type="tool_search_output",
            execution="client",
            status="completed",
            tools=[],
            created_by="server",
        ),
    )

    converted_call = run_items.run_item_to_input_item(tool_search_call)
    converted_output = run_items.run_item_to_input_item(tool_search_output)

    assert isinstance(converted_call, dict)
    assert converted_call["type"] == "tool_search_call"
    assert "created_by" not in converted_call
    assert isinstance(converted_output, dict)
    assert converted_output["type"] == "tool_search_output"
    assert "created_by" not in converted_output

def test_run_item_to_input_item_omits_tool_call_metadata() -> None:
    agent = Agent(name="A")
    tool_call = ToolCallItem(
        agent=agent,
        raw_item=ResponseFunctionToolCall(
            id="fc_123",
            call_id="call_123",
            name="lookup_account",
            arguments="{}",
            type="function_call",
            status="completed",
        ),
        description="Lookup customer records.",
        title="Lookup Account",
    )

    result = run_items.run_item_to_input_item(tool_call)
    result_dict = cast(dict[str, Any], result)

    assert isinstance(result, dict)
    assert result_dict["type"] == "function_call"
    assert "description" not in result_dict
    assert "title" not in result_dict

def test_normalize_input_items_for_api_strips_internal_tool_call_metadata() -> None:
    item = cast(
        TResponseInputItem,
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "lookup_account",
            "arguments": "{}",
            run_items.TOOL_CALL_SESSION_DESCRIPTION_KEY: "Lookup customer records.",
            run_items.TOOL_CALL_SESSION_TITLE_KEY: "Lookup Account",
        },
    )

    normalized = run_items.normalize_input_items_for_api([item])
    normalized_item = cast(dict[str, Any], normalized[0])

    assert run_items.TOOL_CALL_SESSION_DESCRIPTION_KEY not in normalized_item
    assert run_items.TOOL_CALL_SESSION_TITLE_KEY not in normalized_item

def test_fingerprint_input_item_ignores_internal_tool_call_metadata() -> None:
    base_item = cast(
        TResponseInputItem,
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "lookup_account",
            "arguments": "{}",
        },
    )
    with_metadata = cast(
        TResponseInputItem,
        {
            **cast(dict[str, Any], base_item),
            run_items.TOOL_CALL_SESSION_DESCRIPTION_KEY: "Lookup customer records.",
            run_items.TOOL_CALL_SESSION_TITLE_KEY: "Lookup Account",
        },
    )

    assert run_items.fingerprint_input_item(base_item) == run_items.fingerprint_input_item(
        with_metadata
    )

def test_run_result_to_input_list_preserves_tool_search_items() -> None:
    agent = Agent(name="A")
    result = RunResult(
        input="Find CRM tools",
        new_items=[
            ToolSearchCallItem(
                agent=agent,
                raw_item={"type": "tool_search_call", "queries": [{"search_term": "profile"}]},
            ),
            ToolSearchOutputItem(
                agent=agent,
                raw_item={"type": "tool_search_output", "results": [{"text": "Customer profile"}]},
            ),
        ],
        raw_responses=[],
        final_output="done",
        input_guardrail_results=[],
        output_guardrail_results=[],
        tool_input_guardrail_results=[],
        tool_output_guardrail_results=[],
        context_wrapper=RunContextWrapper(context=None),
        _last_agent=agent,
    )

    input_items = result.to_input_list()

    assert len(input_items) == 3
    assert cast(dict[str, Any], input_items[1])["type"] == "tool_search_call"
    assert cast(dict[str, Any], input_items[2])["type"] == "tool_search_output"


# --- tests/test_run_state.py ---

async def test_resume_pending_function_approval_reinterrupts() -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    async def needs_ok(text: str) -> str:
        calls.append(text)
        return text

    model, agent = make_model_and_agent(tools=[needs_ok], name="agent")
    turn_outputs = [
        [get_function_tool_call("needs_ok", json.dumps({"text": "one"}), call_id="1")],
        [get_text_message("done")],
    ]

    first, resumed = await run_and_resume_with_mutation(agent, model, turn_outputs, user_input="hi")

    assert first.final_output is None
    assert resumed.final_output is None
    assert resumed.interruptions and isinstance(resumed.interruptions[0], ToolApprovalItem)
    assert calls == []

async def test_resume_rejected_function_approval_emits_output() -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    async def needs_ok(text: str) -> str:
        calls.append(text)
        return text

    model, agent = make_model_and_agent(tools=[needs_ok], name="agent")
    turn_outputs = [
        [get_function_tool_call("needs_ok", json.dumps({"text": "one"}), call_id="1")],
        [get_final_output_message("done")],
    ]

    first, resumed = await run_and_resume_with_mutation(
        agent,
        model,
        turn_outputs,
        user_input="hi",
        mutate_state=lambda state, approval: state.reject(approval),
    )

    assert first.final_output is None
    assert resumed.final_output == "done"
    assert any(
        isinstance(item, ToolCallOutputItem) and item.output == HITL_REJECTION_MSG
        for item in resumed.new_items
    )
    assert calls == []

    def test_initializes_with_default_values(self):
        """Test that RunState initializes with correct default values."""
        context = RunContextWrapper(context={"foo": "bar"})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context)

        assert state._current_turn == 0
        assert state._current_agent == agent
        assert state._original_input == "input"
        assert state._max_turns == 3
        assert state._model_responses == []
        assert state._generated_items == []
        assert state._current_step is None
        assert state._context is not None
        assert state._context.context == {"foo": "bar"}

    def test_set_tool_use_tracker_snapshot_filters_non_strings(self):
        """Test that set_tool_use_tracker_snapshot filters out non-string agent names and tools."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context)

        # Create snapshot with non-string agent names and non-string tools
        # Use Any to allow invalid types for testing the filtering logic
        snapshot: dict[Any, Any] = {
            "agent1": ["tool1", "tool2"],  # Valid
            123: ["tool3"],  # Non-string agent name (should be filtered)
            "agent2": ["tool4", 456, "tool5"],  # Non-string tool (should be filtered)
            None: ["tool6"],  # None agent name (should be filtered)
        }

        state.set_tool_use_tracker_snapshot(cast(Any, snapshot))

        # Verify non-string agent names are filtered out (line 828)
        result = state.get_tool_use_tracker_snapshot()
        assert "agent1" in result
        assert result["agent1"] == ["tool1", "tool2"]
        assert "agent2" in result
        assert result["agent2"] == ["tool4", "tool5"]  # 456 should be filtered
        # Verify non-string keys were filtered out
        assert str(123) not in result
        assert "None" not in result

    def test_to_json_and_to_string_produce_valid_json(self):
        """Test that toJSON and toString produce valid JSON with correct schema."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="Agent1")
        state = make_state(agent, context=context, original_input="input1", max_turns=2)

        json_data = state.to_json()
        assert json_data["$schemaVersion"] == CURRENT_SCHEMA_VERSION
        assert json_data["current_turn"] == 0
        assert json_data["current_agent"] == {"name": "Agent1"}
        assert json_data["original_input"] == "input1"
        assert json_data["max_turns"] == 2
        assert json_data["generated_items"] == []
        assert json_data["model_responses"] == []

        str_data = state.to_string()
        assert isinstance(str_data, str)
        assert json.loads(str_data) == json_data

    async def test_reasoning_item_id_policy_survives_serialization(self):
        """RunState should preserve reasoning item input policy across serialization."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="AgentReasoningPolicy")
        state = make_state(agent, context=context, original_input="input1", max_turns=2)
        state.set_reasoning_item_id_policy("omit")
        state._generated_items = [
            ReasoningItem(
                agent=agent,
                raw_item=ResponseReasoningItem(type="reasoning", id="rs_state", summary=[]),
            )
        ]

        json_data = state.to_json()
        assert json_data["reasoning_item_id_policy"] == "omit"

        restored = await RunState.from_string(agent, state.to_string())
        assert restored._reasoning_item_id_policy == "omit"

        restored_history = run_items_to_input_items(
            restored._generated_items,
            restored._reasoning_item_id_policy,
        )
        assert len(restored_history) == 1
        assert isinstance(restored_history[0], dict)
        assert restored_history[0].get("type") == "reasoning"
        assert "id" not in restored_history[0]

    async def test_tool_input_survives_serialization_round_trip(self):
        """Structured tool input should be preserved through serialization."""
        context = RunContextWrapper(context={"foo": "bar"})
        context.tool_input = {"text": "hola", "target": "en"}
        agent = Agent(name="ToolInputAgent")
        state = make_state(agent, context=context, original_input="input1", max_turns=2)

        restored = await RunState.from_string(agent, state.to_string())
        assert restored._context is not None
        assert restored._context.tool_input == context.tool_input

    async def test_trace_api_key_serialization_is_opt_in(self):
        """Trace API keys are only serialized when explicitly requested."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="Agent1")
        state = make_state(agent, context=context, original_input="input1", max_turns=2)

        with trace(workflow_name="test", tracing={"api_key": "trace-key"}) as tr:
            state.set_trace(tr)

        default_json = state.to_json()
        assert default_json["trace"] is not None
        assert "tracing_api_key" not in default_json["trace"]
        assert default_json["trace"]["tracing_api_key_hash"]
        assert default_json["trace"]["tracing_api_key_hash"] != "trace-key"

        opt_in_json = state.to_json(include_tracing_api_key=True)
        assert opt_in_json["trace"] is not None
        assert opt_in_json["trace"]["tracing_api_key"] == "trace-key"
        assert (
            opt_in_json["trace"]["tracing_api_key_hash"]
            == default_json["trace"]["tracing_api_key_hash"]
        )

        restored_with_key = await RunState.from_string(
            agent, state.to_string(include_tracing_api_key=True)
        )
        assert restored_with_key._trace_state is not None
        assert restored_with_key._trace_state.tracing_api_key == "trace-key"
        assert (
            restored_with_key._trace_state.tracing_api_key_hash
            == default_json["trace"]["tracing_api_key_hash"]
        )

        restored_without_key = await RunState.from_string(agent, state.to_string())
        assert restored_without_key._trace_state is not None
        assert restored_without_key._trace_state.tracing_api_key is None
        assert (
            restored_without_key._trace_state.tracing_api_key_hash
            == default_json["trace"]["tracing_api_key_hash"]
        )

    async def test_throws_error_if_schema_version_is_missing_or_invalid(self):
        """Test that deserialization fails with missing or invalid schema version."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="Agent1")
        state = make_state(agent, context=context, original_input="input1", max_turns=2)

        json_data = state.to_json()
        del json_data["$schemaVersion"]

        str_data = json.dumps(json_data)
        with pytest.raises(Exception, match="Run state is missing schema version"):
            await RunState.from_string(agent, str_data)

        json_data["$schemaVersion"] = "0.1"
        supported_versions = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        with pytest.raises(
            Exception,
            match=(
                f"Run state schema version 0.1 is not supported. "
                f"Supported versions are: {supported_versions}. "
                f"New snapshots are written as version {CURRENT_SCHEMA_VERSION}."
            ),
        ):
            await RunState.from_string(agent, json.dumps(json_data))

    def test_approve_updates_context_approvals_correctly(self):
        """Test that approve() correctly updates context approvals."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="Agent2")
        state = make_state(agent, context=context, original_input="", max_turns=1)

        approval_item = make_tool_approval_item(
            agent, call_id="cid123", name="toolX", arguments="arguments"
        )

        state.approve(approval_item)

        # Check that the tool is approved
        assert state._context is not None
        assert state._context.is_tool_approved(tool_name="toolX", call_id="cid123") is True

    def test_returns_undefined_when_approval_status_is_unknown(self):
        """Test that isToolApproved returns None for unknown tools."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        assert context.is_tool_approved(tool_name="unknownTool", call_id="cid999") is None

    def test_reject_updates_context_approvals_correctly(self):
        """Test that reject() correctly updates context approvals."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="Agent3")
        state = make_state(agent, context=context, original_input="", max_turns=1)

        approval_item = make_tool_approval_item(
            agent, call_id="cid456", name="toolY", arguments="arguments"
        )

        state.reject(approval_item)

        assert state._context is not None
        assert state._context.is_tool_approved(tool_name="toolY", call_id="cid456") is False

    def test_reject_stores_rejection_message(self):
        """Test that reject() stores the explicit rejection message."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="AgentRejectMessage")
        state = make_state(agent, context=context, original_input="", max_turns=1)

        approval_item = make_tool_approval_item(
            agent, call_id="cid456", name="toolY", arguments="arguments"
        )

        state.reject(approval_item, rejection_message="Denied by reviewer")

        assert state._context is not None
        assert state._context.get_rejection_message("toolY", "cid456") == "Denied by reviewer"

    def test_to_json_non_mapping_context_warns_and_omits(self, caplog):
        """Ensure non-mapping contexts are omitted with a warning during serialization."""

        class NonMappingContext:
            pass

        context = RunContextWrapper(context=NonMappingContext())
        agent = Agent(name="AgentMapping")
        state = make_state(agent, context=context, original_input="input", max_turns=1)

        with caplog.at_level(logging.WARNING, logger="openai.agents"):
            json_data = state.to_json()

        assert json_data["context"]["context"] == {}
        context_meta = json_data["context"]["context_meta"]
        assert context_meta["omitted"] is True
        assert context_meta["serialized_via"] == "omitted"
        assert any("not serializable" in record.message for record in caplog.records)

    def test_to_json_strict_context_requires_serializer(self):
        """Ensure strict_context enforces explicit serialization for custom contexts."""

        class NonMappingContext:
            pass

        context = RunContextWrapper(context=NonMappingContext())
        agent = Agent(name="AgentMapping")
        state = make_state(agent, context=context, original_input="input", max_turns=1)

        with pytest.raises(UserError, match="context_serializer"):
            state.to_json(strict_context=True)

    async def test_from_json_with_context_deserializer(self, caplog):
        """Ensure context_deserializer restores non-mapping contexts."""

        @dataclass
        class SampleContext:
            value: str

        context = RunContextWrapper(context=SampleContext(value="hello"))
        agent = Agent(name="AgentMapping")
        state = make_state(agent, context=context, original_input="input", max_turns=1)

        with caplog.at_level(logging.WARNING, logger="openai.agents"):
            json_data = state.to_json()

        def deserialize_context(payload: Mapping[str, Any]) -> SampleContext:
            return SampleContext(**payload)

        new_state = await RunState.from_json(
            agent,
            json_data,
            context_deserializer=deserialize_context,
        )

        assert new_state._context is not None
        assert isinstance(new_state._context.context, SampleContext)
        assert new_state._context.context.value == "hello"

    def test_to_json_with_context_serializer_records_metadata(self):
        """Ensure context_serializer output is stored with metadata."""

        class CustomContext:
            def __init__(self, value: str) -> None:
                self.value = value

        context = RunContextWrapper(context=CustomContext(value="ok"))
        agent = Agent(name="AgentMapping")
        state = make_state(agent, context=context, original_input="input", max_turns=1)

        def serialize_context(value: Any) -> Mapping[str, Any]:
            return {"value": value.value}

        json_data = state.to_json(context_serializer=serialize_context)

        assert json_data["context"]["context"] == {"value": "ok"}
        context_meta = json_data["context"]["context_meta"]
        assert context_meta["serialized_via"] == "context_serializer"
        assert context_meta["requires_deserializer"] is True
        assert context_meta["omitted"] is False

    async def test_from_json_warns_without_deserializer(self, caplog):
        """Ensure deserialization warns when custom context needs help."""

        @dataclass
        class SampleContext:
            value: str

        context = RunContextWrapper(context=SampleContext(value="hello"))
        agent = Agent(name="AgentMapping")
        state = make_state(agent, context=context, original_input="input", max_turns=1)

        json_data = state.to_json()

        with caplog.at_level(logging.WARNING, logger="openai.agents"):
            _ = await RunState.from_json(agent, json_data)

        assert any("context_deserializer" in record.message for record in caplog.records)

    async def test_from_json_strict_context_requires_deserializer(self):
        """Ensure strict_context raises if deserializer is required."""

        @dataclass
        class SampleContext:
            value: str

        context = RunContextWrapper(context=SampleContext(value="hello"))
        agent = Agent(name="AgentMapping")
        state = make_state(agent, context=context, original_input="input", max_turns=1)

        json_data = state.to_json()

        with pytest.raises(UserError, match="context_deserializer"):
            await RunState.from_json(agent, json_data, strict_context=True)

    async def test_from_json_context_deserializer_can_return_wrapper(self):
        """Ensure deserializer can return a RunContextWrapper."""

        @dataclass
        class SampleContext:
            value: str

        context = RunContextWrapper(context=SampleContext(value="hello"))
        agent = Agent(name="AgentMapping")
        state = make_state(agent, context=context, original_input="input", max_turns=1)
        json_data = state.to_json()

        def deserialize_context(payload: Mapping[str, Any]) -> RunContextWrapper[Any]:
            return RunContextWrapper(context=SampleContext(**payload))

        new_state = await RunState.from_json(
            agent,
            json_data,
            context_deserializer=deserialize_context,
        )

        assert new_state._context is not None
        assert isinstance(new_state._context.context, SampleContext)
        assert new_state._context.context.value == "hello"

    def test_to_json_pydantic_context_records_metadata(self, caplog):
        """Ensure Pydantic contexts serialize with metadata and warnings."""

        class SampleModel(BaseModel):
            value: str

        context = RunContextWrapper(context=SampleModel(value="hello"))
        agent = Agent(name="AgentMapping")
        state = make_state(agent, context=context, original_input="input", max_turns=1)

        with caplog.at_level(logging.WARNING, logger="openai.agents"):
            json_data = state.to_json()

        context_meta = json_data["context"]["context_meta"]
        assert context_meta["original_type"] == "pydantic"
        assert context_meta["serialized_via"] == "model_dump"
        assert context_meta["requires_deserializer"] is True
        assert context_meta["omitted"] is False
        assert any("Pydantic model" in record.message for record in caplog.records)

    async def test_guardrail_results_round_trip(self):
        """Guardrail results survive RunState round-trip."""
        context: RunContextWrapper[dict[str, Any]] = RunContextWrapper(context={})
        agent = Agent(name="GuardrailAgent")
        state = make_state(agent, context=context, original_input="input", max_turns=1)

        input_guardrail = InputGuardrail(
            guardrail_function=lambda ctx, ag, inp: GuardrailFunctionOutput(
                output_info={"input": "info"},
                tripwire_triggered=False,
            ),
            name="input_guardrail",
        )
        output_guardrail = OutputGuardrail(
            guardrail_function=lambda ctx, ag, out: GuardrailFunctionOutput(
                output_info={"output": "info"},
                tripwire_triggered=True,
            ),
            name="output_guardrail",
        )

        state._input_guardrail_results = [
            InputGuardrailResult(
                guardrail=input_guardrail,
                output=GuardrailFunctionOutput(
                    output_info={"input": "info"},
                    tripwire_triggered=False,
                ),
            )
        ]
        state._output_guardrail_results = [
            OutputGuardrailResult(
                guardrail=output_guardrail,
                agent_output="final",
                agent=agent,
                output=GuardrailFunctionOutput(
                    output_info={"output": "info"},
                    tripwire_triggered=True,
                ),
            )
        ]

        restored = await roundtrip_state(agent, state)

        assert len(restored._input_guardrail_results) == 1
        restored_input = restored._input_guardrail_results[0]
        assert restored_input.guardrail.get_name() == "input_guardrail"
        assert restored_input.output.tripwire_triggered is False
        assert restored_input.output.output_info == {"input": "info"}

        assert len(restored._output_guardrail_results) == 1
        restored_output = restored._output_guardrail_results[0]
        assert restored_output.guardrail.get_name() == "output_guardrail"
        assert restored_output.output.tripwire_triggered is True
        assert restored_output.output.output_info == {"output": "info"}
        assert restored_output.agent_output == "final"
        assert restored_output.agent.name == agent.name

    async def test_tool_guardrail_results_round_trip(self):
        """Tool guardrail results survive RunState round-trip."""
        context: RunContextWrapper[dict[str, Any]] = RunContextWrapper(context={})
        agent = Agent(name="ToolGuardrailAgent")
        state = make_state(agent, context=context, original_input="input", max_turns=1)

        tool_input_guardrail: ToolInputGuardrail[Any] = ToolInputGuardrail(
            guardrail_function=lambda data: ToolGuardrailFunctionOutput(
                output_info={"input": "info"},
                behavior=AllowBehavior(type="allow"),
            ),
            name="tool_input_guardrail",
        )
        tool_output_guardrail: ToolOutputGuardrail[Any] = ToolOutputGuardrail(
            guardrail_function=lambda data: ToolGuardrailFunctionOutput(
                output_info={"output": "info"},
                behavior=AllowBehavior(type="allow"),
            ),
            name="tool_output_guardrail",
        )

        state._tool_input_guardrail_results = [
            ToolInputGuardrailResult(
                guardrail=tool_input_guardrail,
                output=ToolGuardrailFunctionOutput(
                    output_info={"input": "info"},
                    behavior=AllowBehavior(type="allow"),
                ),
            )
        ]
        state._tool_output_guardrail_results = [
            ToolOutputGuardrailResult(
                guardrail=tool_output_guardrail,
                output=ToolGuardrailFunctionOutput(
                    output_info={"output": "info"},
                    behavior=AllowBehavior(type="allow"),
                ),
            )
        ]

        restored = await roundtrip_state(agent, state)

        assert len(restored._tool_input_guardrail_results) == 1
        restored_tool_input = restored._tool_input_guardrail_results[0]
        assert restored_tool_input.guardrail.get_name() == "tool_input_guardrail"
        assert restored_tool_input.output.behavior["type"] == "allow"
        assert restored_tool_input.output.output_info == {"input": "info"}

        assert len(restored._tool_output_guardrail_results) == 1
        restored_tool_output = restored._tool_output_guardrail_results[0]
        assert restored_tool_output.guardrail.get_name() == "tool_output_guardrail"
        assert restored_tool_output.output.behavior["type"] == "allow"
        assert restored_tool_output.output.output_info == {"output": "info"}

    def test_reject_permanently_when_always_reject_option_is_passed(self):
        """Test that reject with always_reject=True sets permanent rejection."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="Agent4")
        state = make_state(agent, context=context, original_input="", max_turns=1)

        approval_item = make_tool_approval_item(
            agent, call_id="cid789", name="toolZ", arguments="arguments"
        )

        state.reject(approval_item, always_reject=True)

        assert state._context is not None
        assert state._context.is_tool_approved(tool_name="toolZ", call_id="cid789") is False

        # Check that it's permanently rejected
        assert state._context is not None
        approvals = state._context._approvals
        assert "toolZ" in approvals
        assert approvals["toolZ"].approved is False
        assert approvals["toolZ"].rejected is True

    def test_rejection_is_scoped_to_call_ids(self):
        """Test that a rejected tool call does not auto-apply to new call IDs."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="AgentRejectReuse")
        state = make_state(agent, context=context, original_input="", max_turns=1)

        approval_item = make_tool_approval_item(
            agent, call_id="cid789", name="toolZ", arguments="arguments"
        )

        state.reject(approval_item)

        assert state._context is not None
        assert state._context.is_tool_approved(tool_name="toolZ", call_id="cid789") is False
        assert state._context.is_tool_approved(tool_name="toolZ", call_id="cid999") is None
        assert state._context.get_rejection_message("toolZ", "cid999") is None

    def test_always_reject_reuses_rejection_message_for_future_calls(self):
        """Test that always_reject stores a sticky rejection message."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="AgentStickyReject")
        state = make_state(agent, context=context, original_input="", max_turns=1)

        approval_item = make_tool_approval_item(
            agent, call_id="cid789", name="toolZ", arguments="arguments"
        )

        state.reject(approval_item, always_reject=True, rejection_message="")

        assert state._context is not None
        assert state._context.get_rejection_message("toolZ", "cid789") == ""
        assert state._context.get_rejection_message("toolZ", "cid999") == ""

    def test_approve_raises_when_context_is_none(self):
        """Test that approve raises UserError when context is None."""
        agent = Agent(name="Agent5")
        state: RunState[dict[str, str], Agent[Any]] = make_state(
            agent, context=RunContextWrapper(context={}), original_input="", max_turns=1
        )
        state._context = None  # Simulate None context

        approval_item = make_tool_approval_item(agent, call_id="cid", name="tool", arguments="")

        with pytest.raises(Exception, match="Cannot approve tool: RunState has no context"):
            state.approve(approval_item)

    def test_reject_raises_when_context_is_none(self):
        """Test that reject raises UserError when context is None."""
        agent = Agent(name="Agent6")
        state: RunState[dict[str, str], Agent[Any]] = make_state(
            agent, context=RunContextWrapper(context={}), original_input="", max_turns=1
        )
        state._context = None  # Simulate None context

        approval_item = make_tool_approval_item(agent, call_id="cid", name="tool", arguments="")

        with pytest.raises(Exception, match="Cannot reject tool: RunState has no context"):
            state.reject(approval_item)

    async def test_generated_items_not_duplicated_by_last_processed_response(self):
        """Ensure to_json doesn't duplicate tool calls from last_processed_response (parity with JS)."""  # noqa: E501
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="AgentDedup")
        state = make_state(agent, context=context, original_input="input", max_turns=2)

        tool_call = get_function_tool_call(name="get_weather", call_id="call_1")
        tool_call_item = ToolCallItem(raw_item=cast(Any, tool_call), agent=agent)

        # Simulate a turn that produced a tool call and also stored it in last_processed_response
        state._generated_items = [tool_call_item]
        state._last_processed_response = make_processed_response(new_items=[tool_call_item])

        json_data = state.to_json()
        generated_items_json = json_data["generated_items"]

        # Only the original generated_items should be present (no duplicate from last_processed_response)  # noqa: E501
        assert len(generated_items_json) == 1
        assert generated_items_json[0]["raw_item"]["call_id"] == "call_1"

        # Deserialization should also retain a single instance
        restored = await RunState.from_json(agent, json_data)
        assert len(restored._generated_items) == 1
        raw_item = restored._generated_items[0].raw_item
        if isinstance(raw_item, dict):
            call_id = raw_item.get("call_id")
        else:
            call_id = getattr(raw_item, "call_id", None)
        assert call_id == "call_1"

    async def test_anonymous_tool_search_items_keep_later_same_content_snapshot(self):
        """Ensure later anonymous tool_search snapshots survive the generated-item merge."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="AgentToolSearchMerge")
        state = make_state(agent, context=context, original_input="input", max_turns=2)

        first_tool_search_call_item = ToolSearchCallItem(
            raw_item={
                "type": "tool_search_call",
                "arguments": {"query": "account balance"},
                "execution": "server",
                "status": "completed",
            },
            agent=agent,
        )
        first_tool_search_output_item = ToolSearchOutputItem(
            raw_item={
                "type": "tool_search_output",
                "execution": "server",
                "status": "completed",
                "tools": [],
            },
            agent=agent,
        )

        state._generated_items = [
            first_tool_search_call_item,
            first_tool_search_output_item,
        ]
        state._last_processed_response = make_processed_response(
            new_items=[
                ToolSearchCallItem(
                    raw_item=dict(cast(dict[str, Any], first_tool_search_call_item.raw_item)),
                    agent=agent,
                ),
                ToolSearchOutputItem(
                    raw_item=dict(cast(dict[str, Any], first_tool_search_output_item.raw_item)),
                    agent=agent,
                ),
            ]
        )

        json_data = state.to_json()
        assert [item["type"] for item in json_data["generated_items"]] == [
            "tool_search_call_item",
            "tool_search_output_item",
            "tool_search_call_item",
            "tool_search_output_item",
        ]

    async def test_anonymous_tool_search_items_not_duplicated_across_round_trip(self):
        """Ensure already-merged anonymous tool_search items do not grow across round-trips."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="AgentToolSearchDedup")
        state = make_state(agent, context=context, original_input="input", max_turns=2)

        first_tool_search_call_item = ToolSearchCallItem(
            raw_item={
                "type": "tool_search_call",
                "arguments": {"query": "account balance"},
                "execution": "server",
                "status": "completed",
            },
            agent=agent,
        )
        first_tool_search_output_item = ToolSearchOutputItem(
            raw_item={
                "type": "tool_search_output",
                "execution": "server",
                "status": "completed",
                "tools": [],
            },
            agent=agent,
        )
        later_tool_search_call_item = ToolSearchCallItem(
            raw_item=dict(cast(dict[str, Any], first_tool_search_call_item.raw_item)),
            agent=agent,
        )
        later_tool_search_output_item = ToolSearchOutputItem(
            raw_item=dict(cast(dict[str, Any], first_tool_search_output_item.raw_item)),
            agent=agent,
        )

        state._generated_items = [
            first_tool_search_call_item,
            first_tool_search_output_item,
            later_tool_search_call_item,
            later_tool_search_output_item,
        ]
        state._last_processed_response = make_processed_response(
            new_items=[
                ToolSearchCallItem(
                    raw_item=dict(cast(dict[str, Any], later_tool_search_call_item.raw_item)),
                    agent=agent,
                ),
                ToolSearchOutputItem(
                    raw_item=dict(cast(dict[str, Any], later_tool_search_output_item.raw_item)),
                    agent=agent,
                ),
            ]
        )
        state._mark_generated_items_merged_with_last_processed()

        json_data = state.to_json()
        assert [item["type"] for item in json_data["generated_items"]] == [
            "tool_search_call_item",
            "tool_search_output_item",
            "tool_search_call_item",
            "tool_search_output_item",
        ]

        restored = await RunState.from_json(agent, json_data)
        restored_json = restored.to_json()
        assert [item["type"] for item in restored_json["generated_items"]] == [
            "tool_search_call_item",
            "tool_search_output_item",
            "tool_search_call_item",
            "tool_search_output_item",
        ]

    async def test_from_string_reconstructs_state_for_simple_agent(self):
        """Test that fromString correctly reconstructs state for a simple agent."""
        context = RunContextWrapper(context={"a": 1})
        agent = Agent(name="Solo")
        state = make_state(agent, context=context, original_input="orig", max_turns=7)
        state._current_turn = 5

        str_data = state.to_string()
        new_state = await RunState.from_string(agent, str_data)

        assert new_state._max_turns == 7
        assert new_state._current_turn == 5
        assert new_state._current_agent == agent
        assert new_state._context is not None
        assert new_state._context.context == {"a": 1}
        assert new_state._generated_items == []
        assert new_state._model_responses == []

    async def test_from_json_reconstructs_state(self):
        """Test that from_json correctly reconstructs state from dict."""
        context = RunContextWrapper(context={"test": "data"})
        agent = Agent(name="JsonAgent")
        state = make_state(agent, context=context, original_input="test input", max_turns=5)
        state._current_turn = 2

        json_data = state.to_json()
        new_state = await RunState.from_json(agent, json_data)

        assert new_state._max_turns == 5
        assert new_state._current_turn == 2
        assert new_state._current_agent == agent
        assert new_state._context is not None
        assert new_state._context.context == {"test": "data"}

    def test_get_interruptions_returns_empty_when_no_interruptions(self):
        """Test that get_interruptions returns empty list when no interruptions."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="Agent5")
        state = make_state(agent, context=context, original_input="", max_turns=1)

        assert state.get_interruptions() == []

    def test_get_interruptions_returns_interruptions_when_present(self):
        """Test that get_interruptions returns interruptions when present."""
        agent = Agent(name="Agent6")

        raw_item = ResponseFunctionToolCall(
            type="function_call",
            name="toolA",
            call_id="cid111",
            status="completed",
            arguments="args",
        )
        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item)
        state = make_state_with_interruptions(
            agent, [approval_item], original_input="", max_turns=1
        )

        interruptions = state.get_interruptions()
        assert len(interruptions) == 1
        assert interruptions[0] == approval_item

    async def test_serializes_and_restores_approvals(self):
        """Test that approval state is preserved through serialization."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="ApprovalAgent")
        state = make_state(agent, context=context, original_input="test")

        # Approve one tool
        raw_item1 = ResponseFunctionToolCall(
            type="function_call",
            name="tool1",
            call_id="cid1",
            status="completed",
            arguments="",
        )
        approval_item1 = ToolApprovalItem(agent=agent, raw_item=raw_item1)
        state.approve(approval_item1, always_approve=True)

        # Reject another tool
        raw_item2 = ResponseFunctionToolCall(
            type="function_call",
            name="tool2",
            call_id="cid2",
            status="completed",
            arguments="",
        )
        approval_item2 = ToolApprovalItem(agent=agent, raw_item=raw_item2)
        state.reject(approval_item2, always_reject=True)

        # Serialize and deserialize
        str_data = state.to_string()
        new_state = await RunState.from_string(agent, str_data)

        # Check approvals are preserved
        assert new_state._context is not None
        assert new_state._context.is_tool_approved(tool_name="tool1", call_id="cid1") is True
        assert new_state._context.is_tool_approved(tool_name="tool2", call_id="cid2") is False
        assert new_state._context.get_rejection_message("tool2", "cid2") is None

    async def test_serializes_and_restores_rejection_messages(self):
        """Test that rejection messages are preserved through serialization."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="ApprovalMessageAgent")
        state = make_state(agent, context=context, original_input="test")

        raw_item = ResponseFunctionToolCall(
            type="function_call",
            name="tool2",
            call_id="cid2",
            status="completed",
            arguments="",
        )
        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item)
        state.reject(approval_item, always_reject=True, rejection_message="Denied by reviewer")

        new_state = await RunState.from_string(agent, state.to_string())

        assert new_state._context is not None
        assert new_state._context.get_rejection_message("tool2", "cid2") == "Denied by reviewer"
        assert new_state._context.get_rejection_message("tool2", "cid3") == "Denied by reviewer"

    async def test_from_json_accepts_previous_schema_version_without_rejection_messages(self):
        """Test that 1.5 snapshots restore even without rejection message fields."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="ApprovalLegacyAgent")
        state = make_state(agent, context=context, original_input="test")

        raw_item = ResponseFunctionToolCall(
            type="function_call",
            name="tool2",
            call_id="cid2",
            status="completed",
            arguments="",
        )
        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item)
        state.reject(approval_item, rejection_message="Denied by reviewer")

        json_data = state.to_json()
        json_data["$schemaVersion"] = "1.5"
        del json_data["context"]["approvals"]["tool2"]["rejection_messages"]

        restored = await RunState.from_json(agent, json_data)

        assert restored._context is not None
        assert restored._context.is_tool_approved("tool2", "cid2") is False
        assert restored._context.get_rejection_message("tool2", "cid2") is None

    async def test_from_json_with_context_override_uses_serialized_rejection_messages(self):
        """Test that serialized approvals rebuild onto the override context."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={"source": "saved"})
        agent = Agent(name="ApprovalOverrideAgent")
        state = make_state(agent, context=context, original_input="test")

        approval_item = ToolApprovalItem(
            agent=agent,
            raw_item=ResponseFunctionToolCall(
                type="function_call",
                name="tool2",
                call_id="cid2",
                status="completed",
                arguments="",
            ),
        )
        state.reject(approval_item, always_reject=True, rejection_message="Denied by reviewer")

        override_context: RunContextWrapper[dict[str, str]] = RunContextWrapper(
            context={"source": "override"}
        )
        override_context.reject_tool(
            approval_item,
            always_reject=True,
            rejection_message="override denial",
        )

        restored = await RunState.from_json(
            agent,
            state.to_json(),
            context_override=override_context,
        )

        assert restored._context is override_context
        assert restored._context is not None
        assert restored._context.context == {"source": "override"}
        assert restored._context.get_rejection_message("tool2", "cid2") == "Denied by reviewer"
        assert restored._context.get_rejection_message("tool2", "cid3") == "Denied by reviewer"

    def test_build_agent_map_skips_unresolved_handoff_objects(self):
        """Test that buildAgentMap skips custom handoffs without target agent references."""
        agent_a = Agent(name="AgentA")
        agent_b = Agent(name="AgentB")

        async def _invoke_handoff(_ctx: RunContextWrapper[Any], _input: str) -> Agent[Any]:
            return agent_b

        detached_handoff = Handoff(
            tool_name="transfer_to_agent_b",
            tool_description="Transfer to AgentB.",
            input_json_schema={},
            on_invoke_handoff=_invoke_handoff,
            agent_name=agent_b.name,
        )
        agent_a.handoffs = [detached_handoff]

        agent_map = _build_agent_map(agent_a)

        assert sorted(agent_map.keys()) == ["AgentA"]

    async def test_preserves_usage_data(self):
        """Test that usage data is preserved through serialization."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        context.usage.requests = 5
        context.usage.input_tokens = 100
        context.usage.output_tokens = 50
        context.usage.total_tokens = 150

        agent = Agent(name="UsageAgent")
        state = make_state(agent, context=context, original_input="test", max_turns=10)

        str_data = state.to_string()
        new_state = await RunState.from_string(agent, str_data)

        assert new_state._context is not None
        assert new_state._context.usage.requests == 5
        assert new_state._context.usage is not None
        assert new_state._context.usage.input_tokens == 100
        assert new_state._context.usage is not None
        assert new_state._context.usage.output_tokens == 50
        assert new_state._context.usage is not None
        assert new_state._context.usage.total_tokens == 150

    def test_serializes_generated_items(self):
        """Test that generated items are serialized and restored."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="ItemAgent")
        state = make_state(agent, context=context, original_input="test", max_turns=5)

        # Add a message output item with proper ResponseOutputMessage structure
        message_item = MessageOutputItem(agent=agent, raw_item=make_message_output(text="Hello!"))
        state._generated_items.append(message_item)

        # Serialize
        json_data = state.to_json()
        assert len(json_data["generated_items"]) == 1
        assert json_data["generated_items"][0]["type"] == "message_output_item"

    async def test_serializes_current_step_interruption(self):
        """Test that current step interruption is serialized correctly."""
        agent = Agent(name="InterruptAgent")
        raw_item = ResponseFunctionToolCall(
            type="function_call",
            name="myTool",
            call_id="cid_int",
            status="completed",
            arguments='{"arg": "value"}',
        )
        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item)
        state = make_state_with_interruptions(agent, [approval_item], original_input="test")

        json_data = state.to_json()
        assert json_data["current_step"] is not None
        assert json_data["current_step"]["type"] == "next_step_interruption"
        assert len(json_data["current_step"]["data"]["interruptions"]) == 1

        # Deserialize and verify
        new_state = await RunState.from_json(agent, json_data)
        assert isinstance(new_state._current_step, NextStepInterruption)
        assert len(new_state._current_step.interruptions) == 1
        restored_item = new_state._current_step.interruptions[0]
        assert isinstance(restored_item, ToolApprovalItem)
        assert restored_item.name == "myTool"

    async def test_deserializes_various_item_types(self):
        """Test that deserialization handles different item types."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="ItemAgent")
        state = make_state(agent, context=context, original_input="test", max_turns=5)

        # Add various item types
        # 1. Message output item
        msg = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text="Hello", annotations=[])],
        )
        state._generated_items.append(MessageOutputItem(agent=agent, raw_item=msg))

        # 2. Tool call item with description
        tool_call = ResponseFunctionToolCall(
            type="function_call",
            name="my_tool",
            call_id="call_1",
            status="completed",
            arguments='{"arg": "val"}',
        )
        state._generated_items.append(
            ToolCallItem(
                agent=agent,
                raw_item=tool_call,
                description="My tool description",
                title="My tool title",
            )
        )

        # 3. Tool call item without description
        tool_call_no_desc = ResponseFunctionToolCall(
            type="function_call",
            name="other_tool",
            call_id="call_2",
            status="completed",
            arguments="{}",
        )
        state._generated_items.append(ToolCallItem(agent=agent, raw_item=tool_call_no_desc))

        # 4. Tool call output item
        tool_output = {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "result",
        }
        state._generated_items.append(
            ToolCallOutputItem(agent=agent, raw_item=tool_output, output="result")
        )

        # Serialize and deserialize
        json_data = state.to_json()
        new_state = await RunState.from_json(agent, json_data)

        # Verify all items were restored
        assert len(new_state._generated_items) == 4
        assert isinstance(new_state._generated_items[0], MessageOutputItem)
        assert isinstance(new_state._generated_items[1], ToolCallItem)
        assert isinstance(new_state._generated_items[2], ToolCallItem)
        assert isinstance(new_state._generated_items[3], ToolCallOutputItem)

        # Verify display metadata is preserved
        assert new_state._generated_items[1].description == "My tool description"
        assert new_state._generated_items[1].title == "My tool title"
        assert new_state._generated_items[2].description is None
        assert new_state._generated_items[2].title is None

    async def test_serializes_original_input_with_function_call_output(self):
        """Test that original_input with function_call_output items is preserved."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        # Create original_input with function_call_output (API format)
        # This simulates items from session that are in API format
        original_input = [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "test_tool",
                "arguments": '{"arg": "value"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "result",
            },
        ]

        state = make_state(agent, context=context, original_input=original_input, max_turns=5)

        json_data = state.to_json()

        # Verify original_input was kept in API format
        assert isinstance(json_data["original_input"], list)
        assert len(json_data["original_input"]) == 2

        # First item should remain function_call (snake_case)
        assert json_data["original_input"][0]["type"] == "function_call"
        assert json_data["original_input"][0]["call_id"] == "call_123"
        assert json_data["original_input"][0]["name"] == "test_tool"

        # Second item should remain function_call_output without protocol conversion
        assert json_data["original_input"][1]["type"] == "function_call_output"
        assert json_data["original_input"][1]["call_id"] == "call_123"
        assert "name" not in json_data["original_input"][1]
        assert "status" not in json_data["original_input"][1]
        assert json_data["original_input"][1]["output"] == "result"

    async def test_serializes_assistant_messages(
        self, original_input: list[dict[str, Any]], expected_status: str, expected_text: str
    ):
        """Assistant messages should retain status and normalize content."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        state = make_state(agent, context=context, original_input=original_input, max_turns=5)

        json_data = state.to_json()
        assert isinstance(json_data["original_input"], list)
        assert len(json_data["original_input"]) == 1

        assistant_msg = json_data["original_input"][0]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["status"] == expected_status
        assert isinstance(assistant_msg["content"], list)
        assert assistant_msg["content"][0]["type"] == "output_text"
        assert assistant_msg["content"][0]["text"] == expected_text

    async def test_from_string_normalizes_original_input_dict_items(self):
        """Test that from_string normalizes original input dict items.

        Ensures field names are normalized without mutating unrelated fields.
        """
        agent = Agent(name="TestAgent")

        # Create state JSON with original_input containing dict items that should be normalized.
        state_json = {
            "$schemaVersion": CURRENT_SCHEMA_VERSION,
            "current_turn": 0,
            "current_agent": {"name": "TestAgent"},
            "original_input": [
                {
                    "type": "function_call_output",
                    "call_id": "call123",
                    "name": "test_tool",
                    "status": "completed",
                    "output": "result",
                },
                "simple_string",  # Non-dict item should pass through
            ],
            "model_responses": [],
            "context": {
                "usage": {
                    "requests": 0,
                    "input_tokens": 0,
                    "input_tokens_details": [],
                    "output_tokens": 0,
                    "output_tokens_details": [],
                    "total_tokens": 0,
                    "request_usage_entries": [],
                },
                "approvals": {},
                "context": {},
            },
            "tool_use_tracker": {},
            "max_turns": 10,
            "noActiveAgentRun": True,
            "input_guardrail_results": [],
            "output_guardrail_results": [],
            "generated_items": [],
            "current_step": None,
            "last_model_response": None,
            "last_processed_response": None,
            "current_turn_persisted_item_count": 0,
            "trace": None,
        }

        # Deserialize using from_json (which calls the same normalization logic as from_string)
        state = await RunState.from_json(agent, state_json)

        # Verify original_input was normalized
        assert isinstance(state._original_input, list)
        assert len(state._original_input) == 2
        assert state._original_input[1] == "simple_string"

        # First item should remain API format and have provider data removed
        first_item = state._original_input[0]
        assert isinstance(first_item, dict)
        assert first_item["type"] == "function_call_output"
        assert first_item["name"] == "test_tool"
        assert first_item["status"] == "completed"
        assert first_item["call_id"] == "call123"

    async def test_serializes_original_input_with_non_dict_items(self):
        """Test that non-dict items in original_input are preserved."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        # Mix of dict and non-dict items
        # (though in practice original_input is usually dicts or string)
        original_input = [
            {"role": "user", "content": "Hello"},
            "string_item",  # Non-dict item
        ]

        state = make_state(agent, context=context, original_input=original_input, max_turns=5)

        json_data = state.to_json()
        assert isinstance(json_data["original_input"], list)
        assert len(json_data["original_input"]) == 2
        assert json_data["original_input"][0]["role"] == "user"
        assert json_data["original_input"][1] == "string_item"

    async def test_from_json_preserves_function_output_original_input(self):
        """API formatted original_input should be preserved when loading."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context, original_input="placeholder", max_turns=5)

        state_json = state.to_json()
        state_json["original_input"] = [
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "demo_tool",
                "arguments": '{"x":1}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "name": "demo_tool",
                "status": "completed",
                "output": "demo-output",
            },
        ]

        restored_state = await RunState.from_json(agent, state_json)
        assert isinstance(restored_state._original_input, list)
        assert len(restored_state._original_input) == 2

        first_item = restored_state._original_input[0]
        second_item = restored_state._original_input[1]
        assert isinstance(first_item, dict)
        assert isinstance(second_item, dict)
        assert first_item["type"] == "function_call"
        assert second_item["type"] == "function_call_output"
        assert second_item["call_id"] == "call_abc"
        assert second_item["output"] == "demo-output"
        assert second_item["name"] == "demo_tool"
        assert second_item["status"] == "completed"

    def test_serialize_tool_call_output_looks_up_name(self):
        """ToolCallOutputItem serialization should infer name from generated tool calls."""
        agent = Agent(name="TestAgent")
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        state = make_state(agent, context=context, original_input=[], max_turns=5)

        tool_call = ResponseFunctionToolCall(
            id="fc_lookup",
            type="function_call",
            call_id="call_lookup",
            name="lookup_tool",
            arguments="{}",
            status="completed",
        )
        state._generated_items.append(ToolCallItem(agent=agent, raw_item=tool_call))

        output_item = ToolCallOutputItem(
            agent=agent,
            raw_item={"type": "function_call_output", "call_id": "call_lookup", "output": "ok"},
            output="ok",
        )

        serialized = state._serialize_item(output_item)
        raw_item = serialized["raw_item"]
        assert raw_item["type"] == "function_call_output"
        assert raw_item["call_id"] == "call_lookup"
        assert "name" not in raw_item
        assert "status" not in raw_item

    def test_lookup_function_name_sources(
        self,
        setup_state: Callable[[RunState[Any, Agent[Any]], Agent[Any]], None],
        call_id: str,
        expected_name: str,
    ):
        """_lookup_function_name should locate tool names from multiple sources."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context, original_input=[], max_turns=5)

        setup_state(state, agent)
        assert state._lookup_function_name(call_id) == expected_name

    async def test_deserialization_handles_unknown_agent_gracefully(self):
        """Test that deserialization skips items with unknown agents."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="KnownAgent")
        state = make_state(agent, context=context, original_input="test", max_turns=5)

        # Add an item
        msg = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text="Test", annotations=[])],
        )
        state._generated_items.append(MessageOutputItem(agent=agent, raw_item=msg))

        # Serialize
        json_data = state.to_json()

        # Modify the agent name to an unknown one
        json_data["generated_items"][0]["agent"]["name"] = "UnknownAgent"

        # Deserialize - should skip the item with unknown agent
        new_state = await RunState.from_json(agent, json_data)

        # Item should be skipped
        assert len(new_state._generated_items) == 0

    async def test_deserialization_handles_malformed_items_gracefully(self):
        """Test that deserialization handles malformed items without crashing."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context, original_input="test", max_turns=5)

        # Serialize
        json_data = state.to_json()

        # Add a malformed item
        json_data["generated_items"] = [
            {
                "type": "message_output_item",
                "agent": {"name": "TestAgent"},
                "raw_item": {
                    # Missing required fields - will cause deserialization error
                    "type": "message",
                },
            }
        ]

        # Should not crash, just skip the malformed item
        new_state = await RunState.from_json(agent, json_data)

        # Malformed item should be skipped
        assert len(new_state._generated_items) == 0

    def test_approval_takes_precedence_over_rejection_when_both_true(self):
        """Test that approval takes precedence when both approved and rejected are True."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})

        # Manually set both approved and rejected to True (edge case)
        context._approvals["test_tool"] = type(
            "ApprovalEntry", (), {"approved": True, "rejected": True}
        )()

        # Should return True (approval takes precedence)
        result = context.is_tool_approved("test_tool", "call_id")
        assert result is True

    def test_individual_approval_takes_precedence_over_individual_rejection(self):
        """Test individual call_id approval takes precedence over rejection."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})

        # Set both individual approval and rejection lists with same call_id
        context._approvals["test_tool"] = type(
            "ApprovalEntry", (), {"approved": ["call_123"], "rejected": ["call_123"]}
        )()

        # Should return True (approval takes precedence)
        result = context.is_tool_approved("test_tool", "call_123")
        assert result is True

    def test_returns_none_when_no_approval_or_rejection(self):
        """Test that None is returned when no approval/rejection info exists."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})

        # Tool exists but no approval/rejection
        context._approvals["test_tool"] = type(
            "ApprovalEntry", (), {"approved": [], "rejected": []}
        )()

        # Should return None (unknown status)
        result = context.is_tool_approved("test_tool", "call_456")
        assert result is None

    def test_to_json_raises_when_no_current_agent(self):
        """Test that to_json raises when current_agent is None."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context, original_input="test", max_turns=5)
        state._current_agent = None  # Simulate None agent

        with pytest.raises(Exception, match="Cannot serialize RunState: No current agent"):
            state.to_json()

    def test_to_json_raises_when_no_context(self):
        """Test that to_json raises when context is None."""
        agent = Agent(name="TestAgent")
        state: RunState[dict[str, str], Agent[Any]] = make_state(
            agent, context=RunContextWrapper(context={}), original_input="test", max_turns=5
        )
        state._context = None  # Simulate None context

        with pytest.raises(Exception, match="Cannot serialize RunState: No context"):
            state.to_json()

    async def test_serialization_includes_handoff_fields(self):
        """Test that handoff items include source and target agent fields."""

        agent_a = Agent(name="AgentA")
        agent_b = Agent(name="AgentB")
        agent_a.handoffs = [agent_b]

        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        state = make_state(agent_a, context=context, original_input="test handoff", max_turns=2)

        # Create a handoff output item
        handoff_item = HandoffOutputItem(
            agent=agent_b,
            raw_item={"type": "handoff_output", "status": "completed"},  # type: ignore[arg-type]
            source_agent=agent_a,
            target_agent=agent_b,
        )
        state._generated_items.append(handoff_item)

        json_data = state.to_json()
        assert len(json_data["generated_items"]) == 1
        item_data = json_data["generated_items"][0]
        assert "source_agent" in item_data
        assert "target_agent" in item_data
        assert item_data["source_agent"]["name"] == "AgentA"
        assert item_data["target_agent"]["name"] == "AgentB"

        # Test round-trip deserialization
        restored = await RunState.from_string(agent_a, state.to_string())
        assert len(restored._generated_items) == 1
        assert restored._generated_items[0].type == "handoff_output_item"

    async def test_model_response_serialization_roundtrip(self):
        """Test that model responses serialize and deserialize correctly."""

        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context, original_input="test", max_turns=2)

        # Add a model response
        response = ModelResponse(
            usage=Usage(requests=1, input_tokens=10, output_tokens=20, total_tokens=30),
            output=[
                ResponseOutputMessage(
                    type="message",
                    id="msg1",
                    status="completed",
                    role="assistant",
                    content=[ResponseOutputText(text="Hello", type="output_text", annotations=[])],
                )
            ],
            response_id="resp123",
            request_id="req123",
        )
        state._model_responses.append(response)

        # Round trip
        json_str = state.to_string()
        restored = await RunState.from_string(agent, json_str)

        assert len(restored._model_responses) == 1
        assert restored._model_responses[0].response_id == "resp123"
        assert restored._model_responses[0].request_id == "req123"
        assert restored._model_responses[0].usage.requests == 1
        assert restored._model_responses[0].usage.input_tokens == 10

    async def test_interruptions_serialization_roundtrip(self):
        """Test that interruptions serialize and deserialize correctly."""
        agent = Agent(name="InterruptAgent")

        # Create tool approval item for interruption
        raw_item = ResponseFunctionToolCall(
            type="function_call",
            name="sensitive_tool",
            call_id="call789",
            status="completed",
            arguments='{"data": "value"}',
            id="1",
        )
        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item)

        state = make_state_with_interruptions(
            agent, [approval_item], original_input="test", max_turns=2
        )

        # Round trip
        json_str = state.to_string()
        restored = await RunState.from_string(agent, json_str)

        assert restored._current_step is not None
        assert isinstance(restored._current_step, NextStepInterruption)
        assert len(restored._current_step.interruptions) == 1
        assert restored._current_step.interruptions[0].raw_item.name == "sensitive_tool"  # type: ignore[union-attr]

    async def test_nested_agent_tool_interruptions_roundtrip(self):
        """Test that nested agent tool approvals survive serialization."""
        inner_agent = Agent(name="InnerAgent")
        outer_agent = Agent(name="OuterAgent")
        outer_agent.tools = [
            inner_agent.as_tool(
                tool_name="inner_agent_tool",
                tool_description="Inner agent tool",
                needs_approval=True,
            )
        ]

        approval_item = ToolApprovalItem(
            agent=inner_agent,
            raw_item=make_function_tool_call("sensitive_tool", call_id="inner-1"),
        )
        state = make_state_with_interruptions(
            outer_agent, [approval_item], original_input="test", max_turns=2
        )

        json_str = state.to_string()
        restored = await RunState.from_string(outer_agent, json_str)

        interruptions = restored.get_interruptions()
        assert len(interruptions) == 1
        assert interruptions[0].agent.name == "InnerAgent"
        assert interruptions[0].raw_item.name == "sensitive_tool"  # type: ignore[union-attr]

    async def test_nested_agent_tool_hitl_resume_survives_json_round_trip_after_gc(self) -> None:
        """Nested agent-tool resumptions should survive RunState JSON round-trips."""

        def _has_function_call_output(input_data: str | list[TResponseInputItem]) -> bool:
            if not isinstance(input_data, list):
                return False
            for item in input_data:
                if isinstance(item, dict):
                    if item.get("type") == "function_call_output":
                        return True
                    continue
                if getattr(item, "type", None) == "function_call_output":
                    return True
            return False

        class ResumeAwareToolModel(Model):
            def __init__(
                self, *, tool_name: str, tool_arguments: str, final_text: str, call_prefix: str
            ) -> None:
                self.tool_name = tool_name
                self.tool_arguments = tool_arguments
                self.final_text = final_text
                self.call_prefix = call_prefix
                self.call_count = 0

            async def get_response(
                self,
                system_instructions: str | None,
                input: str | list[TResponseInputItem],
                model_settings: ModelSettings,
                tools: list[Any],
                output_schema: Any,
                handoffs: list[Any],
                tracing: Any,
                *,
                previous_response_id: str | None,
                conversation_id: str | None,
                prompt: Any | None,
            ) -> ModelResponse:
                del (
                    system_instructions,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    tracing,
                    previous_response_id,
                    conversation_id,
                    prompt,
                )
                if _has_function_call_output(input):
                    return ModelResponse(
                        output=[get_text_message(self.final_text)],
                        usage=Usage(),
                        response_id=f"{self.call_prefix}-done",
                    )

                self.call_count += 1
                return ModelResponse(
                    output=[
                        ResponseFunctionToolCall(
                            type="function_call",
                            name=self.tool_name,
                            call_id=f"{self.call_prefix}-{id(self)}-{self.call_count}",
                            arguments=self.tool_arguments,
                        )
                    ],
                    usage=Usage(),
                    response_id=f"{self.call_prefix}-call-{self.call_count}",
                )

            async def stream_response(
                self,
                system_instructions: str | None,
                input: str | list[TResponseInputItem],
                model_settings: ModelSettings,
                tools: list[Any],
                output_schema: Any,
                handoffs: list[Any],
                tracing: Any,
                *,
                previous_response_id: str | None,
                conversation_id: str | None,
                prompt: Any | None,
            ) -> AsyncIterator[TResponseStreamEvent]:
                del (
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    tracing,
                    previous_response_id,
                    conversation_id,
                    prompt,
                )
                if False:
                    yield cast(TResponseStreamEvent, {})
                raise RuntimeError("Streaming is not supported in this test.")

        tool_calls: list[str] = []

        @function_tool(name_override="inner_sensitive_tool", needs_approval=True)
        async def inner_sensitive_tool(text: str) -> str:
            tool_calls.append(text)
            return f"approved:{text}"

        inner_model = ResumeAwareToolModel(
            tool_name="inner_sensitive_tool",
            tool_arguments=json.dumps({"text": "hello"}),
            final_text="inner-complete",
            call_prefix="inner",
        )
        inner_agent = Agent(name="InnerAgent", model=inner_model, tools=[inner_sensitive_tool])

        outer_tool = inner_agent.as_tool(
            tool_name="inner_agent_tool",
            tool_description="Inner agent tool",
        )
        outer_model = ResumeAwareToolModel(
            tool_name="inner_agent_tool",
            tool_arguments=json.dumps({"input": "hello"}),
            final_text="outer-complete",
            call_prefix="outer",
        )
        outer_agent = Agent(name="OuterAgent", model=outer_model, tools=[outer_tool])

        first_result = await Runner.run(outer_agent, "start")
        assert first_result.final_output is None
        assert first_result.interruptions

        state_json = first_result.to_state().to_json()
        del first_result
        gc.collect()

        restored_state_one = await RunState.from_json(outer_agent, state_json)
        restored_state_two = await RunState.from_json(outer_agent, state_json)

        restored_interruptions_one = restored_state_one.get_interruptions()
        restored_interruptions_two = restored_state_two.get_interruptions()
        assert len(restored_interruptions_one) == 1
        assert len(restored_interruptions_two) == 1
        restored_state_one.approve(restored_interruptions_one[0])
        restored_state_two.approve(restored_interruptions_two[0])

        resumed_result_one = await Runner.run(outer_agent, restored_state_one)
        resumed_result_two = await Runner.run(outer_agent, restored_state_two)

        assert resumed_result_one.final_output == "outer-complete"
        assert resumed_result_one.interruptions == []
        assert resumed_result_two.final_output == "outer-complete"
        assert resumed_result_two.interruptions == []
        assert tool_calls == ["hello", "hello"]

    async def test_json_decode_error_handling(self):
        """Test that invalid JSON raises appropriate error."""
        agent = Agent(name="TestAgent")

        with pytest.raises(Exception, match="Failed to parse run state JSON"):
            await RunState.from_string(agent, "{ invalid json }")

    async def test_missing_agent_in_map_error(self):
        """Test error when agent not found in agent map."""
        agent_a = Agent(name="AgentA")
        state: RunState[dict[str, str], Agent[Any]] = make_state(
            agent_a, context=RunContextWrapper(context={}), original_input="test", max_turns=2
        )

        # Serialize with AgentA
        json_str = state.to_string()

        # Try to deserialize with a different agent that doesn't have AgentA in handoffs
        agent_b = Agent(name="AgentB")
        with pytest.raises(Exception, match="Agent AgentA not found in agent map"):
            await RunState.from_string(agent_b, json_str)

    async def test_to_json_includes_tool_call_items_from_last_processed_response(self):
        """Test that to_json includes tool_call_items from last_processed_response.new_items."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context)

        # Create a tool call item
        tool_call = ResponseFunctionToolCall(
            type="function_call",
            name="test_tool",
            call_id="call123",
            status="completed",
            arguments="{}",
        )
        tool_call_item = ToolCallItem(agent=agent, raw_item=tool_call)

        # Create a ProcessedResponse with the tool call item in new_items
        processed_response = make_processed_response(new_items=[tool_call_item])

        # Set the last processed response
        state._last_processed_response = processed_response

        # Serialize
        json_data = state.to_json()

        # Verify that the tool_call_item is in generated_items
        generated_items = json_data.get("generated_items", [])
        assert len(generated_items) == 1
        assert generated_items[0]["type"] == "tool_call_item"
        assert generated_items[0]["raw_item"]["name"] == "test_tool"

    async def test_to_json_camelizes_nested_dicts_and_lists(self):
        """Test that to_json camelizes nested dictionaries and lists."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context)

        # Create a message with nested content
        message = ResponseOutputMessage(
            id="msg1",
            type="message",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputText(
                    type="output_text",
                    text="Hello",
                    annotations=[],
                    logprobs=[],
                )
            ],
        )
        state._generated_items.append(MessageOutputItem(agent=agent, raw_item=message))

        # Serialize
        json_data = state.to_json()

        # Verify that nested structures are camelized
        generated_items = json_data.get("generated_items", [])
        assert len(generated_items) == 1
        raw_item = generated_items[0]["raw_item"]
        # Check that snake_case fields are camelized
        assert "response_id" in raw_item or "id" in raw_item

    async def test_to_string_serializes_non_json_outputs(self):
        """Test that to_string handles outputs with non-JSON values."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context)

        tool_call_output = ToolCallOutputItem(
            agent=agent,
            raw_item={
                "type": "function_call_output",
                "call_id": "call123",
                "output": "ok",
            },
            output={"timestamp": datetime(2024, 1, 1, 12, 0, 0)},
        )
        state._generated_items.append(tool_call_output)

        state_string = state.to_string()
        json_data = json.loads(state_string)

        generated_items = json_data.get("generated_items", [])
        assert len(generated_items) == 1
        output_payload = generated_items[0]["output"]
        assert isinstance(output_payload, dict)
        assert isinstance(output_payload["timestamp"], str)

    async def test_from_json_with_last_processed_response(self):
        """Test that from_json correctly deserializes last_processed_response."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context)

        # Create a tool call item
        tool_call = ResponseFunctionToolCall(
            type="function_call",
            name="test_tool",
            call_id="call123",
            status="completed",
            arguments="{}",
        )
        tool_call_item = ToolCallItem(agent=agent, raw_item=tool_call)

        # Create a ProcessedResponse with the tool call item
        processed_response = make_processed_response(new_items=[tool_call_item])

        # Set the last processed response
        state._last_processed_response = processed_response

        # Serialize and deserialize
        json_data = state.to_json()
        new_state = await RunState.from_json(agent, json_data)

        # Verify that last_processed_response was deserialized
        assert new_state._last_processed_response is not None
        assert len(new_state._last_processed_response.new_items) == 1
        assert new_state._last_processed_response.new_items[0].type == "tool_call_item"

    async def test_last_processed_response_serializes_local_shell_actions(self):
        """Ensure local shell actions survive to_json/from_json."""
        local_shell_tool = LocalShellTool(executor=lambda _req: "ok")
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent", tools=[local_shell_tool])
        state = make_state(agent, context=context)

        local_shell_call = cast(
            LocalShellCall,
            {
                "type": "local_shell_call",
                "id": "ls1",
                "call_id": "call_local",
                "status": "completed",
                "action": {"commands": ["echo hi"], "timeout_ms": 1000},
            },
        )

        processed_response = make_processed_response(
            local_shell_calls=[
                ToolRunLocalShellCall(tool_call=local_shell_call, local_shell_tool=local_shell_tool)
            ],
        )

        state._last_processed_response = processed_response

        json_data = state.to_json()
        last_processed = json_data.get("last_processed_response", {})
        assert "local_shell_actions" in last_processed
        assert last_processed["local_shell_actions"][0]["local_shell"]["name"] == "local_shell"

        new_state = await RunState.from_json(agent, json_data, context_override={})
        assert new_state._last_processed_response is not None
        assert len(new_state._last_processed_response.local_shell_calls) == 1
        restored = new_state._last_processed_response.local_shell_calls[0]
        assert restored.local_shell_tool.name == "local_shell"
        call_id = getattr(restored.tool_call, "call_id", None)
        if call_id is None and isinstance(restored.tool_call, dict):
            call_id = restored.tool_call.get("call_id")
        assert call_id == "call_local"

    async def test_serialize_function_with_description_and_schema(self):
        """Test serialization of function with description and params_json_schema."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        async def tool_func(context: ToolContext[Any], arguments: str) -> str:
            return "result"

        tool = FunctionTool(
            on_invoke_tool=tool_func,
            name="test_tool",
            description="Test tool description",
            params_json_schema={"type": "object", "properties": {}},
        )

        tool_call = ResponseFunctionToolCall(
            type="function_call",
            name="test_tool",
            call_id="call123",
            status="completed",
            arguments="{}",
        )

        function_run = ToolRunFunction(tool_call=tool_call, function_tool=tool)

        processed_response = make_processed_response(functions=[function_run])

        state = make_state(agent, context=context)
        state._last_processed_response = processed_response

        json_data = state.to_json()
        last_processed = json_data.get("last_processed_response", {})
        functions = last_processed.get("functions", [])
        assert len(functions) == 1
        assert functions[0]["tool"]["description"] == "Test tool description"
        assert "paramsJsonSchema" in functions[0]["tool"]

    async def test_serialize_shell_action_with_description(self):
        """Test serialization of shell action with description."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        # Create a shell tool with description
        async def shell_executor(request: Any) -> Any:
            return {"output": "test output"}

        shell_tool = ShellTool(executor=shell_executor)
        shell_tool.description = "Shell tool description"  # type: ignore[attr-defined]

        # ToolRunShellCall.tool_call is Any, so we can use a dict
        tool_call = {
            "id": "1",
            "type": "shell_call",
            "call_id": "call123",
            "status": "completed",
            "command": "echo test",
        }

        action_run = ToolRunShellCall(tool_call=tool_call, shell_tool=shell_tool)

        processed_response = make_processed_response(shell_calls=[action_run])

        state = make_state(agent, context=context)
        state._last_processed_response = processed_response

        json_data = state.to_json()
        last_processed = json_data.get("last_processed_response", {})
        shell_actions = last_processed.get("shell_actions", [])
        assert len(shell_actions) == 1
        # The shell action should have a shell field with description
        assert "shell" in shell_actions[0]
        shell_dict = shell_actions[0]["shell"]
        assert "description" in shell_dict
        assert shell_dict["description"] == "Shell tool description"

    async def test_serialize_item_with_non_dict_raw_item(self):
        """Test serialization of item with non-dict raw_item."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context)

        # Create a message item
        message = ResponseOutputMessage(
            id="msg1",
            type="message",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputText(type="output_text", text="Hello", annotations=[], logprobs=[])
            ],
        )
        item = MessageOutputItem(agent=agent, raw_item=message)

        # The raw_item is a Pydantic model, not a dict, so it should use model_dump
        state._generated_items.append(item)

        json_data = state.to_json()
        generated_items = json_data.get("generated_items", [])
        assert len(generated_items) == 1
        assert generated_items[0]["type"] == "message_output_item"

    async def test_deserialize_processed_response_without_get_all_tools(self):
        """Test deserialization of ProcessedResponse when agent doesn't have get_all_tools."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})

        # Create an agent without get_all_tools method
        class AgentWithoutGetAllTools(Agent):
            pass

        agent_no_tools = AgentWithoutGetAllTools(name="TestAgent")

        processed_response_data: dict[str, Any] = {
            "new_items": [],
            "handoffs": [],
            "functions": [],
            "computer_actions": [],
            "local_shell_actions": [],
            "mcp_approval_requests": [],
            "tools_used": [],
            "interruptions": [],
        }

        # This should trigger line 759 (all_tools = [])
        result = await _deserialize_processed_response(
            processed_response_data, agent_no_tools, context, {}
        )
        assert result is not None

    async def test_deserialize_processed_response_handoff_with_tool_name(self):
        """Test deserialization of ProcessedResponse with handoff that has tool_name."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent_a = Agent(name="AgentA")
        agent_b = Agent(name="AgentB")

        # Create a handoff with tool_name
        handoff_obj = handoff(agent_b, tool_name_override="handoff_tool")
        agent_a.handoffs = [handoff_obj]

        processed_response_data = {
            "new_items": [],
            "handoffs": [
                {
                    "tool_call": {
                        "type": "function_call",
                        "name": "handoff_tool",
                        "call_id": "call123",
                        "status": "completed",
                        "arguments": "{}",
                    },
                    "handoff": {"tool_name": "handoff_tool"},
                }
            ],
            "functions": [],
            "computer_actions": [],
            "local_shell_actions": [],
            "mcp_approval_requests": [],
            "tools_used": [],
            "interruptions": [],
        }

        # This should trigger lines 778-782 and 787-796
        result = await _deserialize_processed_response(
            processed_response_data, agent_a, context, {"AgentA": agent_a, "AgentB": agent_b}
        )
        assert result is not None
        assert len(result.handoffs) == 1

    async def test_deserialize_processed_response_function_in_tools_map(self):
        """Test deserialization of ProcessedResponse with function in tools_map."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        async def tool_func(context: ToolContext[Any], arguments: str) -> str:
            return "result"

        tool = FunctionTool(
            on_invoke_tool=tool_func,
            name="test_tool",
            description="Test tool",
            params_json_schema={"type": "object", "properties": {}},
        )
        agent.tools = [tool]

        processed_response_data = {
            "new_items": [],
            "handoffs": [],
            "functions": [
                {
                    "tool_call": {
                        "type": "function_call",
                        "name": "test_tool",
                        "call_id": "call123",
                        "status": "completed",
                        "arguments": "{}",
                    },
                    "tool": {"name": "test_tool"},
                }
            ],
            "computer_actions": [],
            "local_shell_actions": [],
            "mcp_approval_requests": [],
            "tools_used": [],
            "interruptions": [],
        }

        # This should trigger lines 801-808
        result = await _deserialize_processed_response(
            processed_response_data, agent, context, {"TestAgent": agent}
        )
        assert result is not None
        assert len(result.functions) == 1

    async def test_deserialize_processed_response_function_uses_namespace(self):
        """Test deserialization of ProcessedResponse with namespace-qualified function names."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        crm_tool = function_tool(lambda customer_id: customer_id, name_override="lookup_account")
        billing_tool = function_tool(
            lambda customer_id: customer_id,
            name_override="lookup_account",
        )
        crm_namespace = tool_namespace(
            name="crm",
            description="CRM tools",
            tools=[crm_tool],
        )
        billing_namespace = tool_namespace(
            name="billing",
            description="Billing tools",
            tools=[billing_tool],
        )
        agent.tools = [*crm_namespace, *billing_namespace]

        processed_response_data = {
            "new_items": [],
            "handoffs": [],
            "functions": [
                {
                    "tool_call": {
                        "type": "function_call",
                        "name": "lookup_account",
                        "namespace": "billing",
                        "call_id": "call123",
                        "status": "completed",
                        "arguments": "{}",
                    },
                    "tool": {"name": "lookup_account", "namespace": "billing"},
                }
            ],
            "computer_actions": [],
            "local_shell_actions": [],
            "mcp_approval_requests": [],
            "tools_used": [],
            "interruptions": [],
        }

        result = await _deserialize_processed_response(
            processed_response_data, agent, context, {"TestAgent": agent}
        )

        assert result is not None
        assert len(result.functions) == 1
        assert result.functions[0].function_tool is billing_namespace[0]

    async def test_deserialize_processed_response_rejects_qualified_name_collision(self):
        """Reject dotted top-level names that collide with namespace-wrapped functions."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        dotted_top_level_tool = function_tool(
            lambda customer_id: customer_id,
            name_override="crm.lookup_account",
        )
        namespaced_tool = tool_namespace(
            name="crm",
            description="CRM tools",
            tools=[function_tool(lambda customer_id: customer_id, name_override="lookup_account")],
        )[0]
        agent.tools = [dotted_top_level_tool, namespaced_tool]

        processed_response_data = {
            "new_items": [],
            "handoffs": [],
            "functions": [
                {
                    "tool_call": {
                        "type": "function_call",
                        "name": "lookup_account",
                        "namespace": "crm",
                        "call_id": "call123",
                        "status": "completed",
                        "arguments": "{}",
                    },
                    "tool": {"name": "lookup_account", "namespace": "crm"},
                }
            ],
            "computer_actions": [],
            "local_shell_actions": [],
            "mcp_approval_requests": [],
            "tools_used": [],
            "interruptions": [],
        }

        with pytest.raises(UserError, match="qualified name `crm.lookup_account`"):
            await _deserialize_processed_response(
                processed_response_data, agent, context, {"TestAgent": agent}
            )

    async def test_deserialize_processed_response_uses_last_duplicate_top_level_function(self):
        """Test deserialization preserves last-wins behavior for duplicate top-level tools."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        first_tool = function_tool(lambda customer_id: customer_id, name_override="lookup")
        second_tool = function_tool(lambda customer_id: customer_id, name_override="lookup")
        agent.tools = [first_tool, second_tool]

        processed_response_data = {
            "new_items": [],
            "handoffs": [],
            "functions": [
                {
                    "tool_call": {
                        "type": "function_call",
                        "name": "lookup",
                        "call_id": "call123",
                        "status": "completed",
                        "arguments": "{}",
                    },
                    "tool": {"name": "lookup"},
                }
            ],
            "computer_actions": [],
            "local_shell_actions": [],
            "mcp_approval_requests": [],
            "tools_used": [],
            "interruptions": [],
        }

        result = await _deserialize_processed_response(
            processed_response_data, agent, context, {"TestAgent": agent}
        )

        assert result is not None
        assert len(result.functions) == 1
        assert result.functions[0].function_tool is second_tool

    async def test_deserialize_processed_response_uses_tool_call_namespace_for_deferred_top_level(
        self,
    ):
        """Synthetic deferred namespaces should disambiguate resumed same-name top-level tools."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        visible_tool = function_tool(
            lambda customer_id: customer_id, name_override="lookup_account"
        )
        deferred_tool = function_tool(
            lambda customer_id: customer_id,
            name_override="lookup_account",
            defer_loading=True,
        )
        agent.tools = [visible_tool, deferred_tool]

        processed_response_data = {
            "new_items": [],
            "handoffs": [],
            "functions": [
                {
                    "tool_call": {
                        "type": "function_call",
                        "name": "lookup_account",
                        "namespace": "lookup_account",
                        "call_id": "call123",
                        "status": "completed",
                        "arguments": "{}",
                    },
                    "tool": {"name": "lookup_account"},
                }
            ],
            "computer_actions": [],
            "local_shell_actions": [],
            "mcp_approval_requests": [],
            "tools_used": [],
            "interruptions": [],
        }

        result = await _deserialize_processed_response(
            processed_response_data, agent, context, {"TestAgent": agent}
        )

        assert result is not None
        assert len(result.functions) == 1
        assert result.functions[0].function_tool is deferred_tool

    async def test_deserialize_processed_response_uses_serialized_lookup_key_for_deferred_top_level(
        self,
    ) -> None:
        """Serialized lookup metadata should disambiguate deferred tools without raw namespace."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        visible_tool = function_tool(
            lambda customer_id: f"visible:{customer_id}",
            name_override="lookup_account",
        )
        deferred_tool = function_tool(
            lambda customer_id: f"deferred:{customer_id}",
            name_override="lookup_account",
            defer_loading=True,
        )
        agent.tools = [visible_tool, deferred_tool]

        processed_response_data = {
            "new_items": [],
            "handoffs": [],
            "functions": [
                {
                    "tool_call": {
                        "type": "function_call",
                        "name": "lookup_account",
                        "call_id": "call123",
                        "status": "completed",
                        "arguments": "{}",
                    },
                    "tool": {
                        "name": "lookup_account",
                        "lookupKey": {
                            "kind": "deferred_top_level",
                            "name": "lookup_account",
                        },
                    },
                }
            ],
            "computer_actions": [],
            "local_shell_actions": [],
            "mcp_approval_requests": [],
            "tools_used": [],
            "interruptions": [],
        }

        result = await _deserialize_processed_response(
            processed_response_data, agent, context, {"TestAgent": agent}
        )

        assert result is not None
        assert len(result.functions) == 1
        assert result.functions[0].function_tool is deferred_tool

    async def test_from_json_missing_schema_version(self):
        """Test that from_json raises error when schema version is missing."""
        agent = Agent(name="TestAgent")
        state_json = {
            "original_input": "test",
            "current_agent": {"name": "TestAgent"},
            "context": {
                "context": {},
                "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "approvals": {},
            },
            "max_turns": 3,
            "current_turn": 0,
            "model_responses": [],
            "generated_items": [],
        }

        with pytest.raises(UserError, match="Run state is missing schema version"):
            await RunState.from_json(agent, state_json)

    async def test_from_json_unsupported_schema_version(self, schema_version: str):
        """Test that from_json raises error when schema version is unsupported."""
        agent = Agent(name="TestAgent")
        state_json = {
            "$schemaVersion": schema_version,
            "original_input": "test",
            "current_agent": {"name": "TestAgent"},
            "context": {
                "context": {},
                "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "approvals": {},
            },
            "max_turns": 3,
            "current_turn": 0,
            "model_responses": [],
            "generated_items": [],
        }

        with pytest.raises(
            UserError, match=f"Run state schema version {schema_version} is not supported"
        ):
            await RunState.from_json(agent, state_json)

    async def test_from_json_accepts_previous_schema_version(self):
        """Test that from_json accepts a previous, explicitly supported schema version."""
        agent = Agent(name="TestAgent")
        state_json = {
            "$schemaVersion": "1.0",
            "original_input": "test",
            "current_agent": {"name": "TestAgent"},
            "context": {
                "context": {"foo": "bar"},
                "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "approvals": {},
            },
            "max_turns": 3,
            "current_turn": 0,
            "model_responses": [],
            "generated_items": [],
        }

        restored = await RunState.from_json(agent, state_json)
        assert restored._current_agent is not None
        assert restored._current_agent.name == "TestAgent"
        assert restored._context is not None
        assert restored._context.context == {"foo": "bar"}

    async def test_from_json_agent_not_found(self):
        """Test that from_json raises error when agent is not found in agent map."""
        agent = Agent(name="TestAgent")
        state_json = {
            "$schemaVersion": "1.0",
            "original_input": "test",
            "current_agent": {"name": "NonExistentAgent"},
            "context": {
                "context": {},
                "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "approvals": {},
            },
            "max_turns": 3,
            "current_turn": 0,
            "model_responses": [],
            "generated_items": [],
        }

        with pytest.raises(UserError, match="Agent NonExistentAgent not found in agent map"):
            await RunState.from_json(agent, state_json)

    async def test_deserialize_processed_response_with_last_processed_response(self):
        """Test deserializing RunState with last_processed_response."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        # Create a tool call item
        tool_call = ResponseFunctionToolCall(
            type="function_call",
            name="test_tool",
            call_id="call123",
            status="completed",
            arguments="{}",
        )
        tool_call_item = ToolCallItem(agent=agent, raw_item=tool_call)

        # Create a ProcessedResponse
        processed_response = make_processed_response(new_items=[tool_call_item])

        state = make_state(agent, context=context)
        state._last_processed_response = processed_response

        # Serialize and deserialize
        json_data = state.to_json()
        new_state = await RunState.from_json(agent, json_data)

        # Verify last processed response was deserialized
        assert new_state._last_processed_response is not None
        assert len(new_state._last_processed_response.new_items) == 1

    async def test_from_string_with_last_processed_response(self):
        """Test deserializing RunState with last_processed_response using from_string."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        # Create a tool call item
        tool_call = ResponseFunctionToolCall(
            type="function_call",
            name="test_tool",
            call_id="call123",
            status="completed",
            arguments="{}",
        )
        tool_call_item = ToolCallItem(agent=agent, raw_item=tool_call)

        # Create a ProcessedResponse
        processed_response = make_processed_response(new_items=[tool_call_item])

        state = make_state(agent, context=context)
        state._last_processed_response = processed_response

        # Serialize to string and deserialize using from_string
        state_string = state.to_string()
        new_state = await RunState.from_string(agent, state_string)

        # Verify last processed response was deserialized
        assert new_state._last_processed_response is not None
        assert len(new_state._last_processed_response.new_items) == 1

    async def test_run_state_merge_keeps_tool_output_with_same_call_id(self):
        """RunState merge should keep tool outputs even when call IDs already exist."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        tool_call = ResponseFunctionToolCall(
            type="function_call",
            name="test_tool",
            call_id="call-merge-1",
            status="completed",
            arguments="{}",
        )
        tool_call_item = ToolCallItem(agent=agent, raw_item=tool_call)
        tool_output_item = ToolCallOutputItem(
            agent=agent,
            output="ok",
            raw_item=ItemHelpers.tool_call_output_item(tool_call, "ok"),
        )

        processed_response = make_processed_response(new_items=[tool_output_item])
        state = make_state(agent, context=context)
        state._generated_items = [tool_call_item]
        state._last_processed_response = processed_response

        json_data = state.to_json()
        generated_types = [item["type"] for item in json_data["generated_items"]]
        assert "tool_call_item" in generated_types
        assert "tool_call_output_item" in generated_types

    async def test_deserialize_processed_response_agent_without_get_all_tools(self):
        """Test deserializing processed response when agent doesn't have get_all_tools."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})

        # Create an agent without get_all_tools method
        class AgentWithoutGetAllTools:
            name = "TestAgent"
            handoffs = []

        agent = AgentWithoutGetAllTools()

        processed_response_data: dict[str, Any] = {
            "new_items": [],
            "handoffs": [],
            "functions": [],
            "computer_actions": [],
            "tools_used": [],
            "mcp_approval_requests": [],
        }

        # This should not raise an error, just return empty tools
        result = await _deserialize_processed_response(
            processed_response_data,
            agent,  # type: ignore[arg-type]
            context,
            {},
        )
        assert result is not None

    async def test_deserialize_processed_response_empty_mcp_tool_data(self):
        """Test deserializing processed response with empty mcp_tool_data."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        processed_response_data = {
            "new_items": [],
            "handoffs": [],
            "functions": [],
            "computer_actions": [],
            "tools_used": [],
            "mcp_approval_requests": [
                {
                    "request_item": {
                        "raw_item": {
                            "type": "mcp_approval_request",
                            "id": "req1",
                            "server_label": "test_server",
                            "name": "test_tool",
                            "arguments": "{}",
                        }
                    },
                    "mcp_tool": {},  # Empty mcp_tool_data should be skipped
                }
            ],
        }

        result = await _deserialize_processed_response(processed_response_data, agent, context, {})
        # Should skip the empty mcp_tool_data and not add it to mcp_approval_requests
        assert len(result.mcp_approval_requests) == 0

    def test_tool_approval_item_with_explicit_tool_name(self):
        """Test that ToolApprovalItem uses explicit tool_name when provided."""
        agent = Agent(name="TestAgent")
        raw_item = ResponseFunctionToolCall(
            type="function_call",
            name="raw_tool_name",
            call_id="call123",
            status="completed",
            arguments="{}",
        )

        # Create with explicit tool_name
        approval_item = ToolApprovalItem(
            agent=agent, raw_item=raw_item, tool_name="explicit_tool_name"
        )

        assert approval_item.tool_name == "explicit_tool_name"
        assert approval_item.name == "explicit_tool_name"

    def test_tool_approval_item_falls_back_to_raw_item_name(self):
        """Test that ToolApprovalItem falls back to raw_item.name when tool_name not provided."""
        agent = Agent(name="TestAgent")
        raw_item = ResponseFunctionToolCall(
            type="function_call",
            name="raw_tool_name",
            call_id="call123",
            status="completed",
            arguments="{}",
        )

        # Create without explicit tool_name
        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item)

        assert approval_item.tool_name == "raw_tool_name"
        assert approval_item.name == "raw_tool_name"

    def test_approve_tool_with_explicit_tool_name(self):
        """Test that approve_tool works with explicit tool_name."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        raw_item = ResponseFunctionToolCall(
            type="function_call",
            name="raw_name",
            call_id="call123",
            status="completed",
            arguments="{}",
        )

        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item, tool_name="explicit_name")
        context.approve_tool(approval_item)

        assert context.is_tool_approved(tool_name="explicit_name", call_id="call123") is True

    def test_approve_tool_extracts_call_id_from_dict(self):
        """Test that approve_tool extracts call_id from dict raw_item."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        # Dict with hosted tool identifiers (id instead of call_id)
        raw_item = {
            "type": "hosted_tool_call",
            "name": "hosted_tool",
            "id": "hosted_call_123",  # Hosted tools use "id" instead of "call_id"
        }

        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item)
        context.approve_tool(approval_item)

        assert context.is_tool_approved(tool_name="hosted_tool", call_id="hosted_call_123") is True

    def test_reject_tool_with_explicit_tool_name(self):
        """Test that reject_tool works with explicit tool_name."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        raw_item = ResponseFunctionToolCall(
            type="function_call",
            name="raw_name",
            call_id="call789",
            status="completed",
            arguments="{}",
        )

        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item, tool_name="explicit_name")
        context.reject_tool(approval_item)

        assert context.is_tool_approved(tool_name="explicit_name", call_id="call789") is False

    async def test_serialize_tool_approval_item_with_tool_name(self):
        """Test that ToolApprovalItem serializes tool_name field."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context, original_input="test")

        raw_item = ResponseFunctionToolCall(
            type="function_call",
            name="raw_name",
            call_id="call123",
            status="completed",
            arguments="{}",
        )
        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item, tool_name="explicit_name")
        state._generated_items.append(approval_item)

        json_data = state.to_json()
        generated_items = json_data.get("generated_items", [])
        assert len(generated_items) == 1

        approval_item_data = generated_items[0]
        assert approval_item_data["type"] == "tool_approval_item"
        assert approval_item_data["tool_name"] == "explicit_name"

    async def test_round_trip_serialization_with_tool_name(self):
        """Test round-trip serialization preserves tool_name."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context, original_input="test")

        raw_item = ResponseFunctionToolCall(
            type="function_call",
            name="raw_name",
            call_id="call123",
            status="completed",
            arguments="{}",
        )
        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item, tool_name="explicit_name")
        state._generated_items.append(approval_item)

        # Serialize and deserialize
        json_data = state.to_json()
        new_state = await RunState.from_json(agent, json_data)

        assert len(new_state._generated_items) == 1
        restored_item = new_state._generated_items[0]
        assert isinstance(restored_item, ToolApprovalItem)
        assert restored_item.tool_name == "explicit_name"
        assert restored_item.name == "explicit_name"

    async def test_round_trip_serialization_preserves_allow_bare_name_alias(self):
        """Test round-trip serialization preserves bare-name approval alias metadata."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context, original_input="test")

        raw_item = {
            "type": "function_call",
            "name": "get_weather",
            "call_id": "call123",
            "status": "completed",
            "arguments": "{}",
            "namespace": "get_weather",
        }
        approval_item = ToolApprovalItem(
            agent=agent,
            raw_item=raw_item,
            tool_name="get_weather",
            tool_namespace="get_weather",
            _allow_bare_name_alias=True,
        )
        state._generated_items.append(approval_item)

        json_data = state.to_json()
        assert json_data["generated_items"][0]["allow_bare_name_alias"] is True

        new_state = await RunState.from_json(agent, json_data)

        restored_item = new_state._generated_items[0]
        assert isinstance(restored_item, ToolApprovalItem)
        assert restored_item._allow_bare_name_alias is True

    def test_tool_approval_item_arguments_property(self):
        """Test that ToolApprovalItem.arguments property correctly extracts arguments."""
        agent = Agent(name="TestAgent")

        # Test with ResponseFunctionToolCall
        raw_item1 = ResponseFunctionToolCall(
            type="function_call",
            name="tool1",
            call_id="call1",
            status="completed",
            arguments='{"city": "Oakland"}',
        )
        approval_item1 = ToolApprovalItem(agent=agent, raw_item=raw_item1)
        assert approval_item1.arguments == '{"city": "Oakland"}'

        # Test with dict raw_item
        raw_item2 = {
            "type": "function_call",
            "name": "tool2",
            "call_id": "call2",
            "status": "completed",
            "arguments": '{"key": "value"}',
        }
        approval_item2 = ToolApprovalItem(agent=agent, raw_item=raw_item2)
        assert approval_item2.arguments == '{"key": "value"}'

        # Test with dict raw_item without arguments
        raw_item3 = {
            "type": "function_call",
            "name": "tool3",
            "call_id": "call3",
            "status": "completed",
        }
        approval_item3 = ToolApprovalItem(agent=agent, raw_item=raw_item3)
        assert approval_item3.arguments is None

        # Test with raw_item that has no arguments attribute
        raw_item4 = {"type": "unknown", "name": "tool4"}
        approval_item4 = ToolApprovalItem(agent=agent, raw_item=raw_item4)
        assert approval_item4.arguments is None

    def test_tool_approval_item_tracks_namespace(self):
        """Test that ToolApprovalItem keeps namespace metadata from Responses tool calls."""
        agent = Agent(name="TestAgent")
        raw_item = make_tool_call(
            call_id="call-ns-1",
            name="lookup_account",
            namespace="crm",
            status="completed",
            arguments="{}",
        )

        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item)

        assert approval_item.tool_name == "lookup_account"
        assert approval_item.tool_namespace == "crm"
        assert approval_item.qualified_name == "crm.lookup_account"

    def test_tool_approval_item_collapses_synthetic_deferred_namespace_in_qualified_name(self):
        """Synthetic deferred namespaces should display as the bare tool name."""
        agent = Agent(name="TestAgent")
        raw_item = make_tool_call(
            call_id="call-weather-1",
            name="get_weather",
            namespace="get_weather",
            status="completed",
            arguments="{}",
        )

        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item)

        assert approval_item.tool_name == "get_weather"
        assert approval_item.tool_namespace == "get_weather"
        assert approval_item.qualified_name == "get_weather"

    async def test_round_trip_serialization_with_tool_namespace(self):
        """Test round-trip serialization preserves tool namespace metadata."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context, original_input="test")

        raw_item = make_tool_call(
            call_id="call123",
            name="lookup_account",
            namespace="billing",
            status="completed",
            arguments="{}",
        )
        approval_item = ToolApprovalItem(agent=agent, raw_item=raw_item)
        state._generated_items.append(approval_item)

        new_state = await RunState.from_json(agent, state.to_json())

        assert len(new_state._generated_items) == 1
        restored_item = new_state._generated_items[0]
        assert isinstance(restored_item, ToolApprovalItem)
        assert restored_item.tool_name == "lookup_account"
        assert restored_item.tool_namespace == "billing"
        assert restored_item.qualified_name == "billing.lookup_account"

    async def test_round_trip_serialization_preserves_tool_lookup_key(self) -> None:
        """Deferred approval items should keep their explicit lookup key through RunState."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")
        state = make_state(agent, context=context, original_input="test")

        raw_item = make_tool_call(
            call_id="call-weather",
            name="get_weather",
            namespace="get_weather",
            status="completed",
            arguments="{}",
        )
        approval_item = ToolApprovalItem(
            agent=agent,
            raw_item=raw_item,
            tool_lookup_key=("deferred_top_level", "get_weather"),
        )
        state._generated_items.append(approval_item)

        new_state = await RunState.from_json(agent, state.to_json())

        assert len(new_state._generated_items) == 1
        restored_item = new_state._generated_items[0]
        assert isinstance(restored_item, ToolApprovalItem)
        assert restored_item.tool_lookup_key == ("deferred_top_level", "get_weather")

    async def test_deserialize_items_restores_tool_search_items(self):
        """Test that tool search run items survive RunState round-trips."""
        agent = Agent(name="TestAgent")
        items = _deserialize_items(
            [
                {
                    "type": "tool_search_call_item",
                    "agent": {"name": "TestAgent"},
                    "raw_item": {
                        "id": "tsc_state",
                        "type": "tool_search_call",
                        "arguments": {"paths": ["crm"], "query": "profile"},
                        "execution": "server",
                        "status": "completed",
                    },
                },
                {
                    "type": "tool_search_output_item",
                    "agent": {"name": "TestAgent"},
                    "raw_item": {
                        "id": "tso_state",
                        "type": "tool_search_output",
                        "execution": "server",
                        "status": "completed",
                        "tools": [
                            {
                                "type": "function",
                                "name": "get_customer_profile",
                                "description": "Fetch a CRM customer profile.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "customer_id": {
                                            "type": "string",
                                        }
                                    },
                                    "required": ["customer_id"],
                                },
                                "defer_loading": True,
                            }
                        ],
                    },
                },
            ],
            {"TestAgent": agent},
        )

        assert isinstance(items[0], ToolSearchCallItem)
        assert isinstance(items[1], ToolSearchOutputItem)
        assert isinstance(items[0].raw_item, ResponseToolSearchCall)
        assert isinstance(items[1].raw_item, ResponseToolSearchOutputItem)

    async def test_deserialize_items_handles_non_dict_items_in_original_input(self):
        """Test that from_json handles non-dict items in original_input list."""
        agent = Agent(name="TestAgent")

        state_json = {
            "$schemaVersion": CURRENT_SCHEMA_VERSION,
            "current_turn": 0,
            "current_agent": {"name": "TestAgent"},
            "original_input": [
                "string_item",  # Non-dict item - tests line 759
                {"type": "function_call", "call_id": "call1", "name": "tool1", "arguments": "{}"},
            ],
            "max_turns": 5,
            "context": {
                "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "approvals": {},
                "context": {},
            },
            "generated_items": [],
            "model_responses": [],
        }

        state = await RunState.from_json(agent, state_json)
        # Should handle non-dict items in original_input (line 759)
        assert isinstance(state._original_input, list)
        assert len(state._original_input) == 2
        assert state._original_input[0] == "string_item"

    async def test_from_json_handles_string_original_input(self):
        """Test that from_json handles string original_input."""
        agent = Agent(name="TestAgent")

        state_json = {
            "$schemaVersion": CURRENT_SCHEMA_VERSION,
            "current_turn": 0,
            "current_agent": {"name": "TestAgent"},
            "original_input": "string_input",  # String - tests line 762-763
            "max_turns": 5,
            "context": {
                "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "approvals": {},
                "context": {},
            },
            "generated_items": [],
            "model_responses": [],
        }

        state = await RunState.from_json(agent, state_json)
        # Should handle string original_input (line 762-763)
        assert state._original_input == "string_input"

    async def test_from_string_handles_non_dict_items_in_original_input(self):
        """Test that from_string handles non-dict items in original_input list."""
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        agent = Agent(name="TestAgent")

        state = make_state(agent, context=context, original_input=["string_item"], max_turns=5)
        state_string = state.to_string()

        new_state = await RunState.from_string(agent, state_string)
        # Should handle non-dict items in original_input (line 759)
        assert isinstance(new_state._original_input, list)
        assert new_state._original_input[0] == "string_item"

    async def test_lookup_function_name_searches_last_processed_response_new_items(self):
        """Test _lookup_function_name searches last_processed_response.new_items."""
        agent = Agent(name="TestAgent")
        context: RunContextWrapper[dict[str, str]] = RunContextWrapper(context={})
        state = make_state(agent, context=context, original_input=[], max_turns=5)

        # Create tool call items in last_processed_response
        tool_call1 = ResponseFunctionToolCall(
            id="fc1",
            type="function_call",
            call_id="call1",
            name="tool1",
            arguments="{}",
            status="completed",
        )
        tool_call2 = ResponseFunctionToolCall(
            id="fc2",
            type="function_call",
            call_id="call2",
            name="tool2",
            arguments="{}",
            status="completed",
        )
        tool_call_item1 = ToolCallItem(agent=agent, raw_item=tool_call1)
        tool_call_item2 = ToolCallItem(agent=agent, raw_item=tool_call2)

        # Add non-tool_call item to test skipping (line 658-659)
        message_item = MessageOutputItem(
            agent=agent,
            raw_item=ResponseOutputMessage(
                id="msg1",
                type="message",
                role="assistant",
                content=[ResponseOutputText(type="output_text", text="Hello", annotations=[])],
                status="completed",
            ),
        )

        processed_response = make_processed_response(
            new_items=[message_item, tool_call_item1, tool_call_item2],  # Mix of types
        )
        state._last_processed_response = processed_response

        # Should find names from last_processed_response, skipping non-tool_call items
        assert state._lookup_function_name("call1") == "tool1"
        assert state._lookup_function_name("call2") == "tool2"
        assert state._lookup_function_name("missing") == ""

    async def test_from_json_preserves_function_call_output_items(self):
        """Test from_json keeps function_call_output items without protocol conversion."""
        agent = Agent(name="TestAgent")

        state_json = {
            "$schemaVersion": CURRENT_SCHEMA_VERSION,
            "current_turn": 0,
            "current_agent": {"name": "TestAgent"},
            "original_input": [
                {
                    "type": "function_call_output",
                    "call_id": "call123",
                    "name": "test_tool",
                    "status": "completed",
                    "output": "result",
                }
            ],
            "max_turns": 5,
            "context": {
                "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "approvals": {},
                "context": {},
            },
            "generated_items": [],
            "model_responses": [],
        }

        state = await RunState.from_json(agent, state_json)
        # Should preserve function_call_output entries
        assert isinstance(state._original_input, list)
        assert len(state._original_input) == 1
        item = state._original_input[0]
        assert isinstance(item, dict)
        assert item["type"] == "function_call_output"
        assert item["name"] == "test_tool"
        assert item["status"] == "completed"


# --- tests/test_run_step_execution.py ---

async def test_empty_response_is_final_output():
    agent = Agent[None](name="test")
    response = ModelResponse(
        output=[],
        usage=Usage(),
        response_id=None,
    )
    result = await get_execute_result(agent, response)

    assert result.original_input == "hello"
    assert result.generated_items == []
    assert isinstance(result.next_step, NextStepFinalOutput)
    assert result.next_step.output == ""

async def test_plaintext_agent_no_tool_calls_is_final_output():
    agent = Agent(name="test")
    response = ModelResponse(
        output=[get_text_message("hello_world")],
        usage=Usage(),
        response_id=None,
    )
    result = await get_execute_result(agent, response)

    assert result.original_input == "hello"
    assert len(result.generated_items) == 1
    assert_item_is_message(result.generated_items[0], "hello_world")
    assert isinstance(result.next_step, NextStepFinalOutput)
    assert result.next_step.output == "hello_world"

async def test_plaintext_agent_no_tool_calls_multiple_messages_is_final_output():
    agent = Agent(name="test")
    response = ModelResponse(
        output=[
            get_text_message("hello_world"),
            get_text_message("bye"),
        ],
        usage=Usage(),
        response_id=None,
    )
    result = await get_execute_result(
        agent,
        response,
        original_input=[
            get_text_input_item("test"),
            get_text_input_item("test2"),
        ],
    )

    assert len(result.original_input) == 2
    assert len(result.generated_items) == 2
    assert_item_is_message(result.generated_items[0], "hello_world")
    assert_item_is_message(result.generated_items[1], "bye")

    assert isinstance(result.next_step, NextStepFinalOutput)
    assert result.next_step.output == "bye"

async def test_execute_tools_allows_unhashable_tool_call_arguments():
    agent = make_agent()
    response = ModelResponse(output=[], usage=Usage(), response_id="resp")
    raw_tool_call = {
        "type": "function_call",
        "call_id": "call-1",
        "name": "tool",
        "arguments": {"key": "value"},
    }
    pre_step_items: list[RunItem] = [ToolCallItem(agent=agent, raw_item=raw_tool_call)]

    result = await get_execute_result(agent, response, generated_items=pre_step_items)

    assert len(result.generated_items) == 1
    assert isinstance(result.next_step, NextStepFinalOutput)

async def test_plaintext_agent_with_tool_call_is_run_again():
    agent = Agent(name="test", tools=[get_function_tool(name="test", return_value="123")])
    response = ModelResponse(
        output=[get_text_message("hello_world"), get_function_tool_call("test", "")],
        usage=Usage(),
        response_id=None,
    )
    result = await get_execute_result(agent, response)

    assert result.original_input == "hello"

    # 3 items: new message, tool call, tool result
    assert len(result.generated_items) == 3
    assert isinstance(result.next_step, NextStepRunAgain)

    items = result.generated_items
    assert_item_is_message(items[0], "hello_world")
    assert_item_is_function_tool_call(items[1], "test", None)
    assert_item_is_function_tool_call_output(items[2], "123")

    assert isinstance(result.next_step, NextStepRunAgain)

async def test_plaintext_agent_hosted_shell_items_without_message_runs_again():
    shell_tool = ShellTool(environment={"type": "container_auto"})
    agent = Agent(name="test", tools=[shell_tool])
    response = ModelResponse(
        output=[
            make_shell_call(
                "call_shell_hosted", id_value="shell_call_hosted", commands=["echo hi"]
            ),
            cast(
                Any,
                {
                    "type": "shell_call_output",
                    "id": "sh_out_hosted",
                    "call_id": "call_shell_hosted",
                    "status": "completed",
                    "output": [
                        {
                            "stdout": "hi\n",
                            "stderr": "",
                            "outcome": {"type": "exit", "exit_code": 0},
                        }
                    ],
                },
            ),
        ],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)

    assert len(result.generated_items) == 2
    assert isinstance(result.generated_items[0], ToolCallItem)
    assert isinstance(result.generated_items[1], ToolCallOutputItem)
    assert isinstance(result.next_step, NextStepRunAgain)

async def test_plaintext_agent_shell_output_only_without_message_runs_again():
    agent = Agent(name="test")
    response = ModelResponse(
        output=[
            cast(
                Any,
                {
                    "type": "shell_call_output",
                    "id": "sh_out_only",
                    "call_id": "call_shell_only",
                    "status": "completed",
                    "output": [
                        {
                            "stdout": "hi\n",
                            "stderr": "",
                            "outcome": {"type": "exit", "exit_code": 0},
                        }
                    ],
                },
            ),
        ],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)

    assert len(result.generated_items) == 1
    assert isinstance(result.generated_items[0], ToolCallOutputItem)
    assert isinstance(result.next_step, NextStepRunAgain)

async def test_plaintext_agent_tool_search_only_without_message_runs_again():
    agent = Agent(name="test")
    response = ModelResponse(output=[], usage=Usage(), response_id=None)
    response.output = cast(
        Any,
        [
            {
                "type": "tool_search_call",
                "id": "tsc_step",
                "arguments": {"paths": ["crm"], "query": "profile"},
                "execution": "server",
                "status": "completed",
            },
            {
                "type": "tool_search_output",
                "id": "tso_step",
                "execution": "server",
                "status": "completed",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup_account",
                        "description": "Look up a CRM account.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "account_id": {
                                    "type": "string",
                                }
                            },
                            "required": ["account_id"],
                        },
                        "defer_loading": True,
                    }
                ],
            },
        ],
    )

    result = await get_execute_result(agent, response)

    assert len(result.generated_items) == 2
    assert getattr(result.generated_items[0].raw_item, "type", None) == "tool_search_call"
    raw_output = result.generated_items[1].raw_item
    assert getattr(raw_output, "type", None) == "tool_search_output"
    assert isinstance(result.next_step, NextStepRunAgain)

async def test_plaintext_agent_client_tool_search_requires_manual_handling() -> None:
    agent = Agent(name="test")
    response = ModelResponse(output=[], usage=Usage(), response_id=None)
    response.output = cast(
        Any,
        [
            {
                "type": "tool_search_call",
                "id": "tsc_client_step",
                "call_id": "call_tool_search_client",
                "arguments": {"paths": ["crm"], "query": "profile"},
                "execution": "client",
                "status": "completed",
            }
        ],
    )

    with pytest.raises(ModelBehaviorError, match="Client-executed tool_search calls"):
        await get_execute_result(agent, response)

async def test_plaintext_agent_hosted_shell_with_refusal_message_is_final_output():
    shell_tool = ShellTool(environment={"type": "container_auto"})
    agent = Agent(name="test", tools=[shell_tool])
    refusal_message = ResponseOutputMessage(
        id="msg_refusal",
        type="message",
        role="assistant",
        content=[ResponseOutputRefusal(type="refusal", refusal="I cannot help with that.")],
        status="completed",
    )
    response = ModelResponse(
        output=[
            make_shell_call(
                "call_shell_hosted_refusal",
                id_value="shell_call_hosted_refusal",
                commands=["echo hi"],
            ),
            cast(
                Any,
                {
                    "type": "shell_call_output",
                    "id": "sh_out_hosted_refusal",
                    "call_id": "call_shell_hosted_refusal",
                    "status": "completed",
                    "output": [
                        {
                            "stdout": "hi\n",
                            "stderr": "",
                            "outcome": {"type": "exit", "exit_code": 0},
                        }
                    ],
                },
            ),
            refusal_message,
        ],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)

    assert len(result.generated_items) == 3
    assert isinstance(result.generated_items[0], ToolCallItem)
    assert isinstance(result.generated_items[1], ToolCallOutputItem)
    assert isinstance(result.generated_items[2], MessageOutputItem)
    assert isinstance(result.next_step, NextStepFinalOutput)
    assert result.next_step.output == ""

async def test_multiple_tool_calls():
    agent = Agent(
        name="test",
        tools=[
            get_function_tool(name="test_1", return_value="123"),
            get_function_tool(name="test_2", return_value="456"),
            get_function_tool(name="test_3", return_value="789"),
        ],
    )
    response = ModelResponse(
        output=[
            get_text_message("Hello, world!"),
            get_function_tool_call("test_1"),
            get_function_tool_call("test_2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)
    assert result.original_input == "hello"

    # 5 items: new message, 2 tool calls, 2 tool call outputs
    assert len(result.generated_items) == 5
    assert isinstance(result.next_step, NextStepRunAgain)

    items = result.generated_items
    assert_item_is_message(items[0], "Hello, world!")
    assert_item_is_function_tool_call(items[1], "test_1", None)
    assert_item_is_function_tool_call(items[2], "test_2", None)

    assert isinstance(result.next_step, NextStepRunAgain)

async def test_multiple_tool_calls_still_raise_when_sibling_failure_error_function_none():
    async def _ok_tool() -> str:
        return "ok"

    async def _error_tool() -> str:
        raise ValueError("boom")

    ok_tool = function_tool(_ok_tool, name_override="ok_tool", failure_error_function=None)
    error_tool = function_tool(
        _error_tool,
        name_override="error_tool",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[ok_tool, error_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("ok_tool", "{}", call_id="1"),
            get_function_tool_call("error_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    with pytest.raises(UserError, match="Error running tool error_tool: boom"):
        await get_execute_result(agent, response)

async def test_multiple_tool_calls_use_custom_failure_error_function_for_cancelled_tool():
    async def _ok_tool() -> str:
        return "ok"

    async def _cancel_tool() -> str:
        raise asyncio.CancelledError("tool-cancelled")

    seen_error: Exception | None = None

    def _custom_failure_error(_context: RunContextWrapper[Any], _error: Exception) -> str:
        nonlocal seen_error
        assert isinstance(_error, Exception)
        assert not isinstance(_error, asyncio.CancelledError)
        seen_error = _error
        return "custom-cancel-msg"

    ok_tool = function_tool(_ok_tool, name_override="ok_tool", failure_error_function=None)
    cancel_tool = function_tool(
        _cancel_tool,
        name_override="cancel_tool",
        failure_error_function=_custom_failure_error,
    )

    agent = Agent(name="test", tools=[ok_tool, cancel_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("ok_tool", "{}", call_id="1"),
            get_function_tool_call("cancel_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)

    assert len(result.generated_items) == 4
    assert isinstance(result.next_step, NextStepRunAgain)
    assert_item_is_function_tool_call_output(result.generated_items[2], "ok")
    assert_item_is_function_tool_call_output(result.generated_items[3], "custom-cancel-msg")
    assert seen_error is not None
    assert str(seen_error) == "tool-cancelled"

async def test_multiple_tool_calls_use_custom_failure_error_function_for_replaced_cancelled_tool():
    async def _ok_tool() -> str:
        return "ok"

    async def _cancel_tool() -> str:
        raise asyncio.CancelledError("tool-cancelled")

    def _custom_failure_error(_context: RunContextWrapper[Any], _error: Exception) -> str:
        return "custom-cancel-msg"

    ok_tool = function_tool(_ok_tool, name_override="ok_tool", failure_error_function=None)
    cancel_tool = dataclasses.replace(
        function_tool(
            _cancel_tool,
            name_override="cancel_tool",
            failure_error_function=_custom_failure_error,
        ),
        name="cancel_tool",
    )

    agent = Agent(name="test", tools=[ok_tool, cancel_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("ok_tool", "{}", call_id="1"),
            get_function_tool_call("cancel_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)

    assert len(result.generated_items) == 4
    assert isinstance(result.next_step, NextStepRunAgain)
    assert_item_is_function_tool_call_output(result.generated_items[2], "ok")
    assert_item_is_function_tool_call_output(result.generated_items[3], "custom-cancel-msg")

async def test_multiple_tool_calls_use_default_failure_error_function_for_copied_cancelled_tool():
    async def _ok_tool() -> str:
        return "ok"

    async def _cancel_tool() -> str:
        raise asyncio.CancelledError("tool-cancelled")

    ok_tool = function_tool(_ok_tool, name_override="ok_tool", failure_error_function=None)
    cancel_tool = copy.deepcopy(function_tool(_cancel_tool, name_override="cancel_tool"))

    agent = Agent(name="test", tools=[ok_tool, cancel_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("ok_tool", "{}", call_id="1"),
            get_function_tool_call("cancel_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)

    assert len(result.generated_items) == 4
    assert isinstance(result.next_step, NextStepRunAgain)
    assert_item_is_function_tool_call_output(result.generated_items[2], "ok")
    assert_item_is_function_tool_call_output(
        result.generated_items[3],
        "An error occurred while running the tool. Please try again. Error: tool-cancelled",
    )

async def test_multiple_tool_calls_use_default_failure_error_function_for_manual_cancelled_tool():
    async def _ok_tool() -> str:
        return "ok"

    async def _manual_on_invoke_tool(_ctx: ToolContext[Any], _args: str) -> str:
        raise asyncio.CancelledError("manual-tool-cancelled")

    ok_tool = function_tool(_ok_tool, name_override="ok_tool", failure_error_function=None)
    manual_tool = FunctionTool(
        name="manual_cancel_tool",
        description="manual cancel",
        params_json_schema={},
        on_invoke_tool=_manual_on_invoke_tool,
    )

    agent = Agent(name="test", tools=[ok_tool, manual_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("ok_tool", "{}", call_id="1"),
            get_function_tool_call("manual_cancel_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)

    assert len(result.generated_items) == 4
    assert isinstance(result.next_step, NextStepRunAgain)
    assert_item_is_function_tool_call_output(result.generated_items[2], "ok")
    assert_item_is_function_tool_call_output(
        result.generated_items[3],
        "An error occurred while running the tool. Please try again. Error: manual-tool-cancelled",
    )

async def test_single_tool_call_uses_default_failure_error_function_for_cancelled_tool():
    async def _cancel_tool() -> str:
        raise asyncio.CancelledError("tool-cancelled")

    cancel_tool = function_tool(_cancel_tool, name_override="cancel_tool")
    agent = Agent(name="test", tools=[cancel_tool])
    response = ModelResponse(
        output=[get_function_tool_call("cancel_tool", "{}", call_id="1")],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)

    assert len(result.generated_items) == 2
    assert isinstance(result.next_step, NextStepRunAgain)
    assert_item_is_function_tool_call_output(
        result.generated_items[1],
        "An error occurred while running the tool. Please try again. Error: tool-cancelled",
    )

async def test_multiple_tool_calls_surface_hook_failure_over_sibling_cancellation():
    hook_started = asyncio.Event()

    class FailingHooks(RunHooks[Any]):
        async def on_tool_end(
            self,
            context: RunContextWrapper[Any],
            agent: Agent[Any],
            tool,
            result: str,
        ) -> None:
            if tool.name != "ok_tool":
                return

            hook_started.set()
            raise ValueError("hook boom")

    async def _ok_tool() -> str:
        return "ok"

    async def _cancel_tool() -> str:
        await hook_started.wait()
        raise asyncio.CancelledError("tool-cancelled")

    hooks = FailingHooks()
    ok_tool = function_tool(_ok_tool, name_override="ok_tool", failure_error_function=None)
    cancel_tool = function_tool(
        _cancel_tool,
        name_override="cancel_tool",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[ok_tool, cancel_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("ok_tool", "{}", call_id="1"),
            get_function_tool_call("cancel_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    with pytest.raises(UserError, match="Error running tool ok_tool: hook boom"):
        await get_execute_result(agent, response, hooks=hooks)

async def test_function_tool_preserves_contextvar_from_tool_body_to_post_invoke_hooks():
    tool_state: ContextVar[str] = ContextVar("tool_state", default="unset")
    seen_values: list[tuple[str, str]] = []

    @tool_output_guardrail
    async def record_guardrail(_data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
        seen_values.append(("guardrail", tool_state.get()))
        return ToolGuardrailFunctionOutput.allow(output_info="checked")

    class RecordingHooks(RunHooks[Any]):
        async def on_tool_end(
            self,
            context: RunContextWrapper[Any],
            agent: Agent[Any],
            tool,
            result: str,
        ) -> None:
            seen_values.append(("hook", tool_state.get()))

    async def _context_tool() -> str:
        tool_state.set("from-tool")
        return "ok"

    hooks = RecordingHooks()
    context_tool = function_tool(
        _context_tool,
        name_override="context_tool",
        tool_output_guardrails=[record_guardrail],
    )
    agent = Agent(name="test", tools=[context_tool])
    response = ModelResponse(
        output=[get_function_tool_call("context_tool", "{}", call_id="1")],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response, hooks=hooks)

    assert isinstance(result.next_step, NextStepRunAgain)
    assert_item_is_function_tool_call_output(result.generated_items[1], "ok")
    assert seen_values == [("guardrail", "from-tool"), ("hook", "from-tool")]
    assert tool_state.get() == "unset"

async def test_mixed_tool_calls_preserve_shell_output_when_function_tool_cancelled():
    async def _cancel_tool() -> str:
        raise asyncio.CancelledError("tool-cancelled")

    cancel_tool = function_tool(_cancel_tool, name_override="cancel_tool")
    shell_tool = ShellTool(executor=lambda _request: "shell ok")
    agent = Agent(name="test", tools=[cancel_tool, shell_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("cancel_tool", "{}", call_id="fn-1"),
            make_shell_call("shell-1"),
        ],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)

    assert len(result.generated_items) == 4
    assert isinstance(result.next_step, NextStepRunAgain)
    assert_item_is_function_tool_call_output(
        result.generated_items[2],
        "An error occurred while running the tool. Please try again. Error: tool-cancelled",
    )
    shell_output = cast(ToolCallOutputItem, result.generated_items[3])
    assert shell_output.output == "shell ok"
    assert cast(dict[str, Any], shell_output.raw_item)["type"] == "shell_call_output"

async def test_multiple_tool_calls_skip_post_invoke_work_for_cancelled_sibling_teardown():
    waiting_tool_started = asyncio.Event()
    failure_handler_called = asyncio.Event()
    output_guardrail_called = asyncio.Event()
    on_tool_end_called = asyncio.Event()

    @tool_output_guardrail
    async def allow_output_guardrail(
        data: ToolOutputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        output_guardrail_called.set()
        return ToolGuardrailFunctionOutput.allow(output_info={"echo": data.output})

    class RecordingHooks(RunHooks[Any]):
        async def on_tool_end(
            self,
            context: RunContextWrapper[Any],
            agent: Agent[Any],
            tool,
            result: str,
        ) -> None:
            if tool.name == "waiting_tool":
                on_tool_end_called.set()

    async def _waiting_tool() -> str:
        waiting_tool_started.set()
        await asyncio.Future()
        return "unreachable"

    async def _error_tool() -> str:
        await waiting_tool_started.wait()
        raise ValueError("boom")

    def _failure_handler(_ctx: RunContextWrapper[Any], error: Exception) -> str:
        failure_handler_called.set()
        return f"handled:{error}"

    waiting_tool = function_tool(
        _waiting_tool,
        name_override="waiting_tool",
        failure_error_function=_failure_handler,
        tool_output_guardrails=[allow_output_guardrail],
    )
    error_tool = function_tool(
        _error_tool,
        name_override="error_tool",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[waiting_tool, error_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("waiting_tool", "{}", call_id="1"),
            get_function_tool_call("error_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    with pytest.raises(UserError, match="Error running tool error_tool: boom"):
        await get_execute_result(agent, response, hooks=RecordingHooks())

    await asyncio.sleep(0)

    assert not failure_handler_called.is_set()
    assert not output_guardrail_called.is_set()
    assert not on_tool_end_called.is_set()

async def test_execute_function_tool_calls_parent_cancellation_skips_post_invoke_work():
    tool_started = asyncio.Event()
    failure_handler_called = asyncio.Event()
    output_guardrail_called = asyncio.Event()
    on_tool_end_called = asyncio.Event()

    @tool_output_guardrail
    async def allow_output_guardrail(
        data: ToolOutputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        output_guardrail_called.set()
        return ToolGuardrailFunctionOutput.allow(output_info={"echo": data.output})

    class RecordingHooks(RunHooks[Any]):
        async def on_tool_end(
            self,
            context: RunContextWrapper[Any],
            agent: Agent[Any],
            tool,
            result: str,
        ) -> None:
            on_tool_end_called.set()

    async def _waiting_tool() -> str:
        tool_started.set()
        await asyncio.Future()
        return "unreachable"

    def _failure_handler(_ctx: RunContextWrapper[Any], error: Exception) -> str:
        failure_handler_called.set()
        return f"handled:{error}"

    tool = function_tool(
        _waiting_tool,
        name_override="waiting_tool",
        failure_error_function=_failure_handler,
        tool_output_guardrails=[allow_output_guardrail],
    )
    agent = Agent(name="test", tools=[tool])
    tool_runs = [
        ToolRunFunction(
            tool_call=cast(
                ResponseFunctionToolCall,
                get_function_tool_call("waiting_tool", "{}", call_id="1"),
            ),
            function_tool=tool,
        )
    ]

    execution_task = asyncio.create_task(
        execute_function_tool_calls(
            agent=agent,
            tool_runs=tool_runs,
            hooks=RecordingHooks(),
            context_wrapper=RunContextWrapper(None),
            config=RunConfig(),
            isolate_parallel_failures=True,
        )
    )
    await asyncio.wait_for(tool_started.wait(), timeout=0.2)

    execution_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(execution_task, timeout=0.1)

    await asyncio.sleep(0)

    assert not failure_handler_called.is_set()
    assert not output_guardrail_called.is_set()
    assert not on_tool_end_called.is_set()

async def test_execute_function_tool_calls_eager_task_factory_tracks_state_safely():
    async def _first_tool() -> str:
        return "first"

    async def _second_tool() -> str:
        return "second"

    first_tool = function_tool(_first_tool, name_override="first_tool")
    second_tool = function_tool(_second_tool, name_override="second_tool")
    tool_runs = [
        ToolRunFunction(
            tool_call=cast(
                ResponseFunctionToolCall,
                get_function_tool_call("first_tool", "{}", call_id="call-1"),
            ),
            function_tool=first_tool,
        ),
        ToolRunFunction(
            tool_call=cast(
                ResponseFunctionToolCall,
                get_function_tool_call("second_tool", "{}", call_id="call-2"),
            ),
            function_tool=second_tool,
        ),
    ]
    loop = asyncio.get_running_loop()
    previous_task_factory = loop.get_task_factory()
    eager_task_factory = cast(Any, asyncio.eager_task_factory)
    loop.set_task_factory(eager_task_factory)

    try:
        (
            function_results,
            input_guardrail_results,
            output_guardrail_results,
        ) = await execute_function_tool_calls(
            agent=Agent(name="test", tools=[first_tool, second_tool]),
            tool_runs=tool_runs,
            hooks=RunHooks(),
            context_wrapper=RunContextWrapper(None),
            config=RunConfig(),
        )
    finally:
        loop.set_task_factory(previous_task_factory)

    assert [result.output for result in function_results] == ["first", "second"]
    assert input_guardrail_results == []
    assert output_guardrail_results == []

async def test_execute_function_tool_calls_collapse_trace_name_for_top_level_deferred_tools():
    async def _shipping_eta(tracking_number: str) -> str:
        return f"eta:{tracking_number}"

    tool = function_tool(
        _shipping_eta,
        name_override="get_shipping_eta",
        defer_loading=True,
    )
    tool_run = ToolRunFunction(
        tool_call=cast(
            ResponseFunctionToolCall,
            get_function_tool_call(
                "get_shipping_eta",
                '{"tracking_number":"ZX-123"}',
                call_id="call-1",
                namespace="get_shipping_eta",
            ),
        ),
        function_tool=tool,
    )

    with trace("test_execute_function_tool_calls_collapse_trace_name_for_top_level_deferred_tools"):
        await execute_function_tool_calls(
            agent=Agent(name="test", tools=[tool]),
            tool_runs=[tool_run],
            hooks=RunHooks(),
            context_wrapper=RunContextWrapper(None),
            config=RunConfig(),
        )

    assert "get_shipping_eta" in _function_span_names()
    assert "get_shipping_eta.get_shipping_eta" not in _function_span_names()

async def test_execute_function_tool_calls_preserve_trace_name_for_explicit_namespace():
    async def _shipping_eta(tracking_number: str) -> str:
        return f"eta:{tracking_number}"

    tool = tool_namespace(
        name="shipping",
        description="Shipping tools",
        tools=[
            function_tool(
                _shipping_eta,
                name_override="get_shipping_eta",
                defer_loading=True,
            )
        ],
    )[0]
    tool_run = ToolRunFunction(
        tool_call=cast(
            ResponseFunctionToolCall,
            get_function_tool_call(
                "get_shipping_eta",
                '{"tracking_number":"ZX-123"}',
                call_id="call-1",
                namespace="shipping",
            ),
        ),
        function_tool=tool,
    )

    with trace("test_execute_function_tool_calls_preserve_trace_name_for_explicit_namespace"):
        await execute_function_tool_calls(
            agent=Agent(name="test", tools=[tool]),
            tool_runs=[tool_run],
            hooks=RunHooks(),
            context_wrapper=RunContextWrapper(None),
            config=RunConfig(),
        )

    assert "shipping.get_shipping_eta" in _function_span_names()
    assert "get_shipping_eta" not in _function_span_names()

async def test_single_tool_call_still_raises_normal_exception():
    async def _error_tool() -> str:
        raise ValueError("boom")

    error_tool = function_tool(
        _error_tool,
        name_override="error_tool",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[error_tool])
    response = ModelResponse(
        output=[get_function_tool_call("error_tool", "{}", call_id="1")],
        usage=Usage(),
        response_id=None,
    )

    with pytest.raises(UserError, match="Error running tool error_tool: boom"):
        await get_execute_result(agent, response)

async def test_multiple_tool_calls_allow_exception_objects_as_tool_outputs():
    async def _returns_exception() -> ValueError:
        return ValueError("as data")

    async def _ok_tool() -> str:
        return "ok"

    returning_tool = function_tool(
        _returns_exception,
        name_override="returns_exception",
        failure_error_function=None,
    )
    ok_tool = function_tool(_ok_tool, name_override="ok_tool", failure_error_function=None)

    agent = Agent(name="test", tools=[returning_tool, ok_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("returns_exception", "{}", call_id="1"),
            get_function_tool_call("ok_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)

    assert len(result.generated_items) == 4
    assert isinstance(result.next_step, NextStepRunAgain)
    assert_item_is_function_tool_call_output(result.generated_items[2], "as data")
    assert_item_is_function_tool_call_output(result.generated_items[3], "ok")

async def test_multiple_tool_calls_allow_successful_sibling_on_tool_end_to_finish():
    cleanup_started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    cleanup_release = asyncio.Event()

    class RecordingHooks(RunHooks[Any]):
        async def on_tool_end(
            self,
            context: RunContextWrapper[Any],
            agent: Agent[Any],
            tool,
            result: str,
        ) -> None:
            if tool.name != "ok_tool":
                return

            cleanup_started.set()
            await cleanup_release.wait()
            cleanup_finished.set()

    async def _ok_tool() -> str:
        return "ok"

    async def _error_tool() -> str:
        await cleanup_started.wait()
        raise ValueError("boom")

    hooks = RecordingHooks()
    ok_tool = function_tool(_ok_tool, name_override="ok_tool", failure_error_function=None)
    error_tool = function_tool(
        _error_tool,
        name_override="error_tool",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[ok_tool, error_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("ok_tool", "{}", call_id="1"),
            get_function_tool_call("error_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    execution_task = asyncio.create_task(get_execute_result(agent, response, hooks=hooks))
    await asyncio.wait_for(cleanup_started.wait(), timeout=0.2)

    with pytest.raises(UserError, match="Error running tool error_tool: boom"):
        await asyncio.wait_for(execution_task, timeout=0.2)

    assert not cleanup_finished.is_set()
    cleanup_release.set()
    await asyncio.wait_for(cleanup_finished.wait(), timeout=0.2)

async def test_multiple_tool_calls_surface_post_invoke_failure_unblocked_during_settle_turns():
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()
    unhandled_contexts: list[dict[str, Any]] = []
    guardrail_started = asyncio.Event()
    release_guardrail = asyncio.Event()

    def _exception_handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        unhandled_contexts.append(context)

    @tool_output_guardrail
    async def externally_released_tripwire_guardrail(
        _data: ToolOutputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        guardrail_started.set()
        await release_guardrail.wait()
        return ToolGuardrailFunctionOutput.raise_exception(output_info={"status": "late-tripwire"})

    async def _ok_tool() -> str:
        return "ok"

    async def _error_tool() -> str:
        await guardrail_started.wait()

        async def _release_guardrail_later() -> None:
            await asyncio.sleep(0)
            release_guardrail.set()

        asyncio.create_task(_release_guardrail_later())
        raise ValueError("boom")

    ok_tool = function_tool(
        _ok_tool,
        name_override="ok_tool",
        failure_error_function=None,
        tool_output_guardrails=[externally_released_tripwire_guardrail],
    )
    error_tool = function_tool(
        _error_tool,
        name_override="error_tool",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[ok_tool, error_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("ok_tool", "{}", call_id="1"),
            get_function_tool_call("error_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    loop.set_exception_handler(_exception_handler)
    try:
        with pytest.raises(ToolOutputGuardrailTripwireTriggered):
            await asyncio.wait_for(get_execute_result(agent, response), timeout=0.2)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(original_handler)

    assert not any(
        context.get("message")
        == "Background function tool post-invoke task raised after failure propagation."
        for context in unhandled_contexts
    )

async def test_multiple_tool_calls_surface_sleeping_post_invoke_failure_before_sibling_error():
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()
    unhandled_contexts: list[dict[str, Any]] = []

    @tool_output_guardrail
    async def sleeping_tripwire_guardrail(
        _data: ToolOutputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        await asyncio.sleep(0.05)
        return ToolGuardrailFunctionOutput.raise_exception(output_info={"status": "sleep-tripwire"})

    async def _ok_tool() -> str:
        return "ok"

    async def _error_tool() -> str:
        raise ValueError("boom")

    ok_tool = function_tool(
        _ok_tool,
        name_override="ok_tool",
        failure_error_function=None,
        tool_output_guardrails=[sleeping_tripwire_guardrail],
    )
    error_tool = function_tool(
        _error_tool,
        name_override="error_tool",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[ok_tool, error_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("ok_tool", "{}", call_id="1"),
            get_function_tool_call("error_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    def _exception_handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        unhandled_contexts.append(context)

    loop.set_exception_handler(_exception_handler)
    try:
        with pytest.raises(ToolOutputGuardrailTripwireTriggered):
            await asyncio.wait_for(get_execute_result(agent, response), timeout=0.2)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(original_handler)

    assert not any(
        context.get("message")
        == "Background function tool post-invoke task raised after failure propagation."
        for context in unhandled_contexts
    )

async def test_multiple_tool_calls_do_not_wait_indefinitely_for_sleeping_post_invoke_sibling():
    guardrail_finished = asyncio.Event()

    @tool_output_guardrail
    async def long_sleeping_guardrail(
        _data: ToolOutputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        await asyncio.sleep(0.3)
        guardrail_finished.set()
        return ToolGuardrailFunctionOutput.allow(output_info="done")

    async def _ok_tool() -> str:
        return "ok"

    async def _error_tool() -> str:
        raise ValueError("boom")

    ok_tool = function_tool(
        _ok_tool,
        name_override="ok_tool",
        failure_error_function=None,
        tool_output_guardrails=[long_sleeping_guardrail],
    )
    error_tool = function_tool(
        _error_tool,
        name_override="error_tool",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[ok_tool, error_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("ok_tool", "{}", call_id="1"),
            get_function_tool_call("error_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    with pytest.raises(UserError, match="Error running tool error_tool: boom"):
        await asyncio.wait_for(get_execute_result(agent, response), timeout=0.2)

    await asyncio.wait_for(guardrail_finished.wait(), timeout=0.5)

async def test_multiple_tool_calls_do_not_wait_for_cancelled_sibling_tool_before_raising():
    started = asyncio.Event()
    cancellation_started = asyncio.Event()
    cancellation_finished = asyncio.Event()
    allow_cancellation_exit = asyncio.Event()

    async def _ok_tool() -> str:
        started.set()
        try:
            await asyncio.Future()
            return "unreachable"
        except asyncio.CancelledError:
            cancellation_started.set()
            await allow_cancellation_exit.wait()
            cancellation_finished.set()
            raise

    async def _error_tool() -> str:
        await started.wait()
        raise ValueError("boom")

    ok_tool = function_tool(_ok_tool, name_override="ok_tool", failure_error_function=None)
    error_tool = function_tool(
        _error_tool,
        name_override="error_tool",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[ok_tool, error_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("ok_tool", "{}", call_id="1"),
            get_function_tool_call("error_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    execution_task = asyncio.create_task(get_execute_result(agent, response))
    await asyncio.wait_for(started.wait(), timeout=0.2)
    await asyncio.wait_for(cancellation_started.wait(), timeout=0.2)

    with pytest.raises(UserError, match="Error running tool error_tool: boom"):
        await asyncio.wait_for(execution_task, timeout=0.2)

    assert not cancellation_finished.is_set()

    allow_cancellation_exit.set()
    await asyncio.wait_for(cancellation_finished.wait(), timeout=0.2)

async def test_multiple_tool_calls_bound_cancelled_sibling_self_rescheduling_cleanup():
    sibling_ready = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    stop_cleanup = asyncio.Event()

    async def _looping_cleanup_tool() -> str:
        try:
            sibling_ready.set()
            await asyncio.Future()
            return "unreachable"
        except asyncio.CancelledError:
            cleanup_started.set()
            while not stop_cleanup.is_set():
                await asyncio.sleep(0)
            cleanup_finished.set()
            raise

    async def _error_tool() -> str:
        await sibling_ready.wait()
        raise ValueError("boom")

    looping_cleanup_tool = function_tool(
        _looping_cleanup_tool,
        name_override="looping_cleanup_tool",
        failure_error_function=None,
    )
    error_tool = function_tool(
        _error_tool,
        name_override="error_tool",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[looping_cleanup_tool, error_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("looping_cleanup_tool", "{}", call_id="1"),
            get_function_tool_call("error_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    with pytest.raises(UserError, match="Error running tool error_tool: boom"):
        await asyncio.wait_for(get_execute_result(agent, response), timeout=0.2)

    assert cleanup_started.is_set()

    stop_cleanup.set()
    await asyncio.wait_for(cleanup_finished.wait(), timeout=0.2)

async def test_multiple_tool_calls_drain_completed_fatal_failures_before_raising():
    class ToolAborted(BaseException):
        pass

    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()
    unhandled_contexts: list[dict[str, Any]] = []

    def _exception_handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        unhandled_contexts.append(context)

    async def _error_tool_1() -> str:
        raise ToolAborted("boom-1")

    async def _error_tool_2() -> str:
        raise ToolAborted("boom-2")

    tool_1 = function_tool(
        _error_tool_1,
        name_override="error_tool_1",
        failure_error_function=None,
    )
    tool_2 = function_tool(
        _error_tool_2,
        name_override="error_tool_2",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[tool_1, tool_2])
    response = ModelResponse(
        output=[
            get_function_tool_call("error_tool_1", "{}", call_id="1"),
            get_function_tool_call("error_tool_2", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    loop.set_exception_handler(_exception_handler)
    try:
        with pytest.raises(ToolAborted):
            await get_execute_result(agent, response)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(original_handler)

    assert not any(
        context.get("message") == "Task exception was never retrieved"
        for context in unhandled_contexts
    )

async def test_multiple_tool_calls_preserve_triggering_error_over_cancelled_sibling_cleanup_error():
    sibling_ready = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def _cleanup_tool() -> str:
        try:
            sibling_ready.set()
            await asyncio.Future()
            return "unreachable"
        except asyncio.CancelledError as cancel_exc:
            sibling_cancelled.set()
            raise ValueError("cleanup") from cancel_exc

    async def _error_tool() -> str:
        await sibling_ready.wait()
        raise ValueError("boom")

    cleanup_tool = function_tool(
        _cleanup_tool,
        name_override="cleanup_tool",
        failure_error_function=None,
    )
    error_tool = function_tool(
        _error_tool,
        name_override="error_tool",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[cleanup_tool, error_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("cleanup_tool", "{}", call_id="1"),
            get_function_tool_call("error_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    with pytest.raises(UserError, match="Error running tool error_tool: boom"):
        await asyncio.wait_for(get_execute_result(agent, response), timeout=0.2)

    assert sibling_cancelled.is_set()

async def test_multiple_tool_calls_report_late_cleanup_exception_from_cancelled_sibling():
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()
    reported_contexts: list[dict[str, Any]] = []
    late_cleanup_reported = asyncio.Event()
    sibling_ready = asyncio.Event()
    cleanup_blocked = asyncio.Event()
    cleanup_finished = asyncio.Event()
    release_cleanup = asyncio.Event()

    def _exception_handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        reported_contexts.append(context)
        if context.get("message") == (
            "Background function tool task raised during cancellation cleanup after failure "
            "propagation."
        ) and isinstance(context.get("exception"), UserError):
            late_cleanup_reported.set()

    async def _error_tool() -> str:
        await sibling_ready.wait()
        raise ValueError("boom")

    async def _cleanup_tool() -> str:
        try:
            sibling_ready.set()
            await asyncio.Future()
            return "unreachable"
        except asyncio.CancelledError as cancel_exc:
            cleanup_blocked.set()
            try:
                await release_cleanup.wait()
            finally:
                cleanup_finished.set()
            raise RuntimeError("late-cleanup-boom") from cancel_exc

    error_tool = function_tool(
        _error_tool,
        name_override="error_tool",
        failure_error_function=None,
    )
    cleanup_tool = function_tool(
        _cleanup_tool,
        name_override="cleanup_tool",
        failure_error_function=None,
    )

    agent = Agent(name="test", tools=[cleanup_tool, error_tool])
    response = ModelResponse(
        output=[
            get_function_tool_call("cleanup_tool", "{}", call_id="1"),
            get_function_tool_call("error_tool", "{}", call_id="2"),
        ],
        usage=Usage(),
        response_id=None,
    )

    loop.set_exception_handler(_exception_handler)
    try:
        with pytest.raises(UserError, match="Error running tool error_tool: boom"):
            await asyncio.wait_for(get_execute_result(agent, response), timeout=0.2)

        assert cleanup_blocked.is_set()
        release_cleanup.set()
        await asyncio.wait_for(cleanup_finished.wait(), timeout=0.2)
        await asyncio.wait_for(late_cleanup_reported.wait(), timeout=0.5)
    finally:
        loop.set_exception_handler(original_handler)

    matching_contexts = [
        context
        for context in reported_contexts
        if context.get("message")
        == "Background function tool task raised during cancellation cleanup after failure "
        "propagation."
    ]
    assert any(
        isinstance(context.get("exception"), UserError)
        and str(context["exception"]) == "Error running tool cleanup_tool: late-cleanup-boom"
        for context in matching_contexts
    )

async def test_parent_cancellation_does_not_report_tool_failure_as_background_error():
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()
    reported_contexts: list[dict[str, Any]] = []
    tool_started = asyncio.Event()

    def _exception_handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        reported_contexts.append(context)

    async def _failing_tool() -> str:
        tool_started.set()
        await asyncio.sleep(0)
        raise ValueError("boom")

    tool = function_tool(
        _failing_tool,
        name_override="failing_tool",
        failure_error_function=None,
    )
    agent = Agent(name="test", tools=[tool])
    response = ModelResponse(
        output=[get_function_tool_call("failing_tool", "{}", call_id="1")],
        usage=Usage(),
        response_id=None,
    )

    loop.set_exception_handler(_exception_handler)
    try:
        execution_task = asyncio.create_task(get_execute_result(agent, response))
        await asyncio.wait_for(tool_started.wait(), timeout=0.2)

        execution_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution_task

        await asyncio.sleep(0)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(original_handler)

    assert not any(
        context.get("message")
        == "Background function tool task raised during cancellation cleanup after failure "
        "propagation."
        and isinstance(context.get("exception"), UserError)
        and str(context["exception"]) == "Error running tool failing_tool: boom"
        for context in reported_contexts
    )

async def test_function_tool_context_includes_run_config() -> None:
    async def _tool_with_run_config(context: ToolContext[str]) -> str:
        assert context.run_config is not None
        return str(context.run_config.model)

    tool = function_tool(
        _tool_with_run_config,
        name_override="tool_with_run_config",
        failure_error_function=None,
    )
    agent = Agent(name="test", tools=[tool])
    response = ModelResponse(
        output=[get_function_tool_call("tool_with_run_config", "{}", call_id="call-1")],
        usage=Usage(),
        response_id=None,
    )
    run_config = RunConfig(model="gpt-4.1-mini")

    result = await get_execute_result(agent, response, run_config=run_config)

    assert len(result.generated_items) == 2
    assert_item_is_function_tool_call_output(result.generated_items[1], "gpt-4.1-mini")
    assert isinstance(result.next_step, NextStepRunAgain)

async def test_deferred_function_tool_context_preserves_search_loaded_namespace() -> None:
    async def _tool_with_namespace(context: ToolContext[str]) -> str:
        tool_call_namespace = getattr(context.tool_call, "namespace", None)
        return json.dumps(
            {
                "tool_call_namespace": tool_call_namespace,
                "tool_namespace": context.tool_namespace,
            },
            sort_keys=True,
        )

    tool = function_tool(
        _tool_with_namespace,
        name_override="get_weather",
        defer_loading=True,
        failure_error_function=None,
    )
    agent = Agent(name="test", tools=[tool])
    response = ModelResponse(
        output=[
            get_function_tool_call(
                "get_weather",
                "{}",
                call_id="call-1",
                namespace="get_weather",
            )
        ],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)

    assert len(result.generated_items) == 2
    assert_item_is_function_tool_call_output(
        result.generated_items[1],
        '{"tool_call_namespace": "get_weather", "tool_namespace": "get_weather"}',
    )
    assert isinstance(result.next_step, NextStepRunAgain)

async def test_handoff_output_leads_to_handoff_next_step():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(name="test_3", handoffs=[agent_1, agent_2])
    response = ModelResponse(
        output=[get_text_message("Hello, world!"), get_handoff_tool_call(agent_1)],
        usage=Usage(),
        response_id=None,
    )
    result = await get_execute_result(agent_3, response)

    assert isinstance(result.next_step, NextStepHandoff)
    assert result.next_step.new_agent == agent_1

    assert len(result.generated_items) == 3

async def test_final_output_without_tool_runs_again():
    agent = Agent(name="test", output_type=Foo, tools=[get_function_tool("tool_1", "result")])
    response = ModelResponse(
        output=[get_function_tool_call("tool_1")],
        usage=Usage(),
        response_id=None,
    )
    result = await get_execute_result(agent, response)

    assert isinstance(result.next_step, NextStepRunAgain)
    assert len(result.generated_items) == 2, "expected 2 items: tool call, tool call output"

async def test_input_guardrail_runs_on_invalid_json():
    guardrail_calls: list[str] = []

    def guardrail(data) -> ToolGuardrailFunctionOutput:
        guardrail_calls.append(data.context.tool_arguments)
        return ToolGuardrailFunctionOutput.allow(output_info="checked")

    guardrail_obj: ToolInputGuardrail[Any] = ToolInputGuardrail(guardrail_function=guardrail)

    def _echo(value: str) -> str:
        return value

    tool = function_tool(
        _echo,
        name_override="guarded",
        tool_input_guardrails=[guardrail_obj],
    )
    agent = Agent(name="test", tools=[tool])
    response = ModelResponse(
        output=[get_function_tool_call("guarded", "bad_json")],
        usage=Usage(),
        response_id=None,
    )

    result = await get_execute_result(agent, response)

    assert guardrail_calls == ["bad_json"]
    assert result.tool_input_guardrail_results
    assert result.tool_input_guardrail_results[0].output.output_info == "checked"

    output_item = next(
        item for item in result.generated_items if isinstance(item, ToolCallOutputItem)
    )
    assert "An error occurred while parsing tool arguments" in str(output_item.output)

async def test_execute_tools_handles_tool_approval_items(
    setup_fn: Callable[[], ToolApprovalRun],
) -> None:
    """Tool approvals should surface as interruptions across tool types."""
    scenario = setup_fn()
    result = await run_execute_with_processed_response(scenario.agent, scenario.processed_response)

    assert_single_approval_interruption(result, tool_name=scenario.expected_tool_name)

async def test_execute_tools_preserves_synthetic_namespace_for_deferred_top_level_approval() -> (
    None
):
    async def _deferred_weather() -> str:
        return "tool_result"

    tool = function_tool(
        _deferred_weather,
        name_override="get_weather",
        defer_loading=True,
        needs_approval=True,
    )
    agent = make_agent(tools=[tool])
    tool_call = cast(
        ResponseFunctionToolCall,
        get_function_tool_call("get_weather", "{}", namespace="get_weather"),
    )
    tool_run = ToolRunFunction(function_tool=tool, tool_call=tool_call)
    processed_response = make_processed_response(functions=[tool_run])

    result = await run_execute_with_processed_response(agent, processed_response)
    interruption = assert_single_approval_interruption(result, tool_name="get_weather")

    assert interruption.tool_namespace == "get_weather"
    assert getattr(interruption.raw_item, "namespace", None) == "get_weather"

async def test_deferred_tool_approval_allows_bare_alias_when_visible_peer_is_disabled() -> None:
    async def _visible_weather() -> str:
        return "visible"

    async def _deferred_weather() -> str:
        return "deferred"

    visible_tool = function_tool(
        _visible_weather,
        name_override="get_weather",
        needs_approval=True,
        is_enabled=False,
    )
    deferred_tool = function_tool(
        _deferred_weather,
        name_override="get_weather",
        defer_loading=True,
        needs_approval=True,
    )
    agent = make_agent(tools=[visible_tool, deferred_tool])
    tool_call = cast(
        ResponseFunctionToolCall,
        get_function_tool_call("get_weather", "{}", namespace="get_weather"),
    )
    tool_run = ToolRunFunction(function_tool=deferred_tool, tool_call=tool_call)
    processed_response = make_processed_response(functions=[tool_run])

    result = await run_execute_with_processed_response(agent, processed_response)
    interruption = assert_single_approval_interruption(result, tool_name="get_weather")

    assert interruption.tool_namespace == "get_weather"
    assert interruption._allow_bare_name_alias is True

async def test_execute_tools_runs_hosted_mcp_callback_when_present():
    """Hosted MCP approvals should invoke on_approval_request callbacks."""

    mcp_tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "test_mcp_server",
            "server_url": "https://example.com",
            "require_approval": "always",
        },
        on_approval_request=lambda request: {"approve": True},
    )
    agent = make_agent(tools=[mcp_tool])
    request_item = McpApprovalRequest(
        id="mcp-approval-1",
        type="mcp_approval_request",
        server_label="test_mcp_server",
        arguments="{}",
        name="list_repo_languages",
    )
    processed_response = make_processed_response(
        new_items=[MCPApprovalRequestItem(raw_item=request_item, agent=agent)],
        mcp_approval_requests=[
            ToolRunMCPApprovalRequest(
                request_item=request_item,
                mcp_tool=mcp_tool,
            )
        ],
    )

    result = await run_execute_with_processed_response(agent, processed_response)

    assert not isinstance(result.next_step, NextStepInterruption)
    assert any(isinstance(item, MCPApprovalResponseItem) for item in result.new_step_items)
    assert not result.processed_response or not result.processed_response.interruptions

async def test_execute_tools_surfaces_hosted_mcp_interruptions_without_callback():
    """Hosted MCP approvals should surface as interruptions when no callback is provided."""

    mcp_tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "test_mcp_server",
            "server_url": "https://example.com",
            "require_approval": "always",
        },
        on_approval_request=None,
    )
    agent = make_agent(tools=[mcp_tool])
    request_item = McpApprovalRequest(
        id="mcp-approval-2",
        type="mcp_approval_request",
        server_label="test_mcp_server",
        arguments="{}",
        name="list_repo_languages",
    )
    processed_response = make_processed_response(
        new_items=[MCPApprovalRequestItem(raw_item=request_item, agent=agent)],
        mcp_approval_requests=[
            ToolRunMCPApprovalRequest(
                request_item=request_item,
                mcp_tool=mcp_tool,
            )
        ],
    )

    result = await run_execute_with_processed_response(agent, processed_response)

    assert isinstance(result.next_step, NextStepInterruption)
    assert result.next_step.interruptions
    assert any(isinstance(item, ToolApprovalItem) for item in result.next_step.interruptions)
    assert any(
        isinstance(item, ToolApprovalItem)
        and getattr(item.raw_item, "id", None) == "mcp-approval-2"
        for item in result.new_step_items
    )

async def test_execute_tools_emits_hosted_mcp_rejection_response():
    """Hosted MCP rejections without callbacks should emit approval responses."""

    mcp_tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "test_mcp_server",
            "server_url": "https://example.com",
            "require_approval": "always",
        },
        on_approval_request=None,
    )
    agent = make_agent(tools=[mcp_tool])
    request_item = McpApprovalRequest(
        id="mcp-approval-reject",
        type="mcp_approval_request",
        server_label="test_mcp_server",
        arguments="{}",
        name="list_repo_languages",
    )
    processed_response = make_processed_response(
        new_items=[MCPApprovalRequestItem(raw_item=request_item, agent=agent)],
        mcp_approval_requests=[
            ToolRunMCPApprovalRequest(
                request_item=request_item,
                mcp_tool=mcp_tool,
            )
        ],
    )
    context_wrapper = make_context_wrapper()
    reject_tool_call(context_wrapper, agent, request_item, tool_name="list_repo_languages")

    result = await run_loop.execute_tools_and_side_effects(
        agent=agent,
        original_input="test",
        pre_step_items=[],
        new_response=ModelResponse(output=[], usage=Usage(), response_id="resp"),
        processed_response=processed_response,
        output_schema=None,
        hooks=RunHooks(),
        context_wrapper=context_wrapper,
        run_config=RunConfig(),
    )

    responses = [
        item for item in result.new_step_items if isinstance(item, MCPApprovalResponseItem)
    ]
    assert responses, "Rejection should emit an MCP approval response."
    assert responses[0].raw_item["approve"] is False
    assert responses[0].raw_item["approval_request_id"] == "mcp-approval-reject"
    assert "reason" not in responses[0].raw_item
    assert not isinstance(result.next_step, NextStepInterruption)

async def test_execute_tools_emits_hosted_mcp_rejection_reason_from_explicit_message():
    """Hosted MCP rejections should forward explicit rejection messages as reasons."""

    mcp_tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "test_mcp_server",
            "server_url": "https://example.com",
            "require_approval": "always",
        },
        on_approval_request=None,
    )
    agent = make_agent(tools=[mcp_tool])
    request_item = McpApprovalRequest(
        id="mcp-approval-reject-reason",
        type="mcp_approval_request",
        server_label="test_mcp_server",
        arguments="{}",
        name="list_repo_languages",
    )
    processed_response = make_processed_response(
        new_items=[MCPApprovalRequestItem(raw_item=request_item, agent=agent)],
        mcp_approval_requests=[
            ToolRunMCPApprovalRequest(
                request_item=request_item,
                mcp_tool=mcp_tool,
            )
        ],
    )
    context_wrapper = make_context_wrapper()
    reject_tool_call(
        context_wrapper,
        agent,
        request_item,
        tool_name="list_repo_languages",
        rejection_message="Denied by policy",
    )

    result = await run_loop.execute_tools_and_side_effects(
        agent=agent,
        original_input="test",
        pre_step_items=[],
        new_response=ModelResponse(output=[], usage=Usage(), response_id="resp"),
        processed_response=processed_response,
        output_schema=None,
        hooks=RunHooks(),
        context_wrapper=context_wrapper,
        run_config=RunConfig(),
    )

    responses = [
        item for item in result.new_step_items if isinstance(item, MCPApprovalResponseItem)
    ]
    assert responses, "Rejection should emit an MCP approval response."
    assert responses[0].raw_item["approve"] is False
    assert responses[0].raw_item["approval_request_id"] == "mcp-approval-reject-reason"
    assert responses[0].raw_item["reason"] == "Denied by policy"


# --- tests/test_run_step_processing.py ---

def test_empty_response():
    agent = Agent(name="test")
    response = ModelResponse(
        output=[],
        usage=Usage(),
        response_id=None,
    )

    result = run_loop.process_model_response(
        agent=agent,
        response=response,
        output_schema=None,
        handoffs=[],
        all_tools=[],
    )
    assert not result.handoffs
    assert not result.functions

def test_no_tool_calls():
    agent = Agent(name="test")
    response = ModelResponse(
        output=[get_text_message("Hello, world!")],
        usage=Usage(),
        response_id=None,
    )
    result = run_loop.process_model_response(
        agent=agent, response=response, output_schema=None, handoffs=[], all_tools=[]
    )
    assert not result.handoffs
    assert not result.functions

async def test_handoffs_parsed_correctly():
    agent_1 = Agent(name="test_1")
    agent_2 = Agent(name="test_2")
    agent_3 = Agent(name="test_3", handoffs=[agent_1, agent_2])
    response = ModelResponse(
        output=[get_text_message("Hello, world!")],
        usage=Usage(),
        response_id=None,
    )
    result = await process_response(agent=agent_3, response=response)
    assert not result.handoffs, "Shouldn't have a handoff here"

    response = ModelResponse(
        output=[get_text_message("Hello, world!"), get_handoff_tool_call(agent_1)],
        usage=Usage(),
        response_id=None,
    )
    result = await process_response(
        agent=agent_3,
        response=response,
        handoffs=await get_handoffs(agent_3, _dummy_ctx()),
    )
    assert len(result.handoffs) == 1, "Should have a handoff here"
    handoff = result.handoffs[0]
    assert handoff.handoff.tool_name == Handoff.default_tool_name(agent_1)
    assert handoff.handoff.tool_description == Handoff.default_tool_description(agent_1)
    assert handoff.handoff.agent_name == agent_1.name

    handoff_agent = await handoff.handoff.on_invoke_handoff(
        RunContextWrapper(None), handoff.tool_call.arguments
    )
    assert handoff_agent == agent_1

async def test_file_search_tool_call_parsed_correctly():
    # Ensure that a ResponseFileSearchToolCall output is parsed into a ToolCallItem and that no tool
    # runs are scheduled.

    agent = Agent(name="test")
    file_search_call = ResponseFileSearchToolCall(
        id="fs1",
        queries=["query"],
        status="completed",
        type="file_search_call",
    )
    response = ModelResponse(
        output=[get_text_message("hello"), file_search_call],
        usage=Usage(),
        response_id=None,
    )
    result = await process_response(agent=agent, response=response)
    # The final item should be a ToolCallItem for the file search call
    assert any(
        isinstance(item, ToolCallItem) and item.raw_item is file_search_call
        for item in result.new_items
    )
    assert not result.functions
    assert not result.handoffs

async def test_function_web_search_tool_call_parsed_correctly():
    agent = Agent(name="test")
    web_search_call = ResponseFunctionWebSearch(
        id="w1",
        action=ActionSearch(type="search", query="query"),
        status="completed",
        type="web_search_call",
    )
    response = ModelResponse(
        output=[get_text_message("hello"), web_search_call],
        usage=Usage(),
        response_id=None,
    )
    result = await process_response(agent=agent, response=response)
    assert any(
        isinstance(item, ToolCallItem) and item.raw_item is web_search_call
        for item in result.new_items
    )
    assert not result.functions
    assert not result.handoffs

async def test_computer_tool_call_without_computer_tool_raises_error():
    # If the agent has no ComputerTool in its tools, process_model_response should raise a
    # ModelBehaviorError when encountering a ResponseComputerToolCall.
    computer_call = ResponseComputerToolCall(
        id="c1",
        type="computer_call",
        action=ActionClick(type="click", x=1, y=2, button="left"),
        call_id="c1",
        pending_safety_checks=[],
        status="completed",
    )
    response = ModelResponse(
        output=[computer_call],
        usage=Usage(),
        response_id=None,
    )
    with pytest.raises(ModelBehaviorError):
        await process_response(agent=Agent(name="test"), response=response)

async def test_computer_tool_call_with_computer_tool_parsed_correctly():
    # If the agent contains a ComputerTool, ensure that a ResponseComputerToolCall is parsed into a
    # ToolCallItem and scheduled to run in computer_actions.
    dummy_computer = DummyComputer()
    agent = Agent(name="test", tools=[ComputerTool(computer=dummy_computer)])
    computer_call = ResponseComputerToolCall(
        id="c1",
        type="computer_call",
        action=ActionClick(type="click", x=1, y=2, button="left"),
        call_id="c1",
        pending_safety_checks=[],
        status="completed",
    )
    response = ModelResponse(
        output=[computer_call],
        usage=Usage(),
        response_id=None,
    )
    result = await process_response(agent=agent, response=response)
    assert any(
        isinstance(item, ToolCallItem) and item.raw_item is computer_call
        for item in result.new_items
    )
    assert result.computer_actions and result.computer_actions[0].tool_call == computer_call


# --- tests/test_server_conversation_tracker.py ---

def test_prepare_input_filters_items_seen_by_server_and_tool_calls() -> None:
    tracker = OpenAIServerConversationTracker(conversation_id="conv", previous_response_id=None)

    original_input: list[TResponseInputItem] = [
        cast(TResponseInputItem, {"id": "input-1", "type": "message"}),
        cast(TResponseInputItem, {"id": "input-2", "type": "message"}),
    ]
    new_raw_item = {"type": "message", "content": "hello"}
    generated_items = [
        DummyRunItem({"id": "server-echo", "type": "message"}),
        DummyRunItem(new_raw_item),
        DummyRunItem({"call_id": "call-1", "output": "done"}, type="function_call_output_item"),
    ]
    model_response = object.__new__(ModelResponse)
    model_response.output = [
        cast(Any, {"call_id": "call-1", "output": "prior", "type": "function_call_output"})
    ]
    model_response.usage = Usage()
    model_response.response_id = "resp-1"
    session_items: list[TResponseInputItem] = [
        cast(TResponseInputItem, {"id": "session-1", "type": "message"})
    ]

    tracker.hydrate_from_state(
        original_input=original_input,
        generated_items=cast(list[Any], generated_items),
        model_responses=[model_response],
        session_items=session_items,
    )

    prepared = tracker.prepare_input(
        original_input=original_input,
        generated_items=cast(list[Any], generated_items),
    )

    assert prepared == [new_raw_item]
    assert tracker.sent_initial_input is True
    assert tracker.remaining_initial_input is None

def test_hydrate_from_state_does_not_track_string_initial_input_by_object_identity() -> None:
    tracker = OpenAIServerConversationTracker(
        conversation_id="conv-init-string", previous_response_id=None
    )

    tracker.hydrate_from_state(
        original_input="hello",
        generated_items=[],
        model_responses=[],
    )

    assert tracker.sent_items == set()
    assert tracker.sent_initial_input is True
    assert tracker.remaining_initial_input is None
    assert len(tracker.sent_item_fingerprints) == 1

def test_hydrate_from_state_does_not_track_list_initial_input_by_object_identity() -> None:
    tracker = OpenAIServerConversationTracker(
        conversation_id="conv-init-list", previous_response_id=None
    )
    original_input = [cast(TResponseInputItem, {"role": "user", "content": "hello"})]

    tracker.hydrate_from_state(
        original_input=original_input,
        generated_items=[],
        model_responses=[],
    )

    assert tracker.sent_items == set()
    assert tracker.sent_initial_input is True
    assert tracker.remaining_initial_input is None
    assert len(tracker.sent_item_fingerprints) == 1

def test_mark_input_as_sent_uses_raw_generated_source_for_rebuilt_filtered_item() -> None:
    tracker = OpenAIServerConversationTracker(conversation_id="conv2b", previous_response_id=None)
    raw_generated_item = {
        "type": "function_call_output",
        "call_id": "call-2b",
        "output": "done",
    }
    generated_items = [
        DummyRunItem(raw_generated_item, type="function_call_output_item"),
    ]

    prepared = tracker.prepare_input(
        original_input=[],
        generated_items=cast(list[Any], generated_items),
    )
    rebuilt_filtered_item = cast(TResponseInputItem, dict(cast(dict[str, Any], prepared[0])))

    tracker.mark_input_as_sent([rebuilt_filtered_item])

    assert id(raw_generated_item) in tracker.sent_items
    assert id(rebuilt_filtered_item) not in tracker.sent_items

    prepared_again = tracker.prepare_input(
        original_input=[],
        generated_items=cast(list[Any], generated_items),
    )
    assert prepared_again == []

def test_hydrate_from_state_skips_restored_tool_search_items_by_object_identity() -> None:
    tracker = OpenAIServerConversationTracker(conversation_id="conv2c", previous_response_id=None)
    tool_search_call = {
        "type": "tool_search_call",
        "queries": [{"search_term": "account balance"}],
    }
    tool_search_result = {
        "type": "tool_search_output",
        "results": [{"text": "Balance lookup docs"}],
    }
    hydrated_items = [
        DummyRunItem(tool_search_call, type="tool_search_call_item"),
        DummyRunItem(tool_search_result, type="tool_search_output_item"),
    ]

    tracker.hydrate_from_state(
        original_input=[],
        generated_items=cast(list[Any], hydrated_items),
        model_responses=[],
    )

    prepared = tracker.prepare_input(
        original_input=[],
        generated_items=cast(list[Any], hydrated_items),
    )

    assert prepared == []

def test_hydrate_from_state_skips_restored_tool_search_items_by_fingerprint() -> None:
    tracker = OpenAIServerConversationTracker(conversation_id="conv2d", previous_response_id=None)
    tool_search_call = {
        "type": "tool_search_call",
        "queries": [{"search_term": "account balance"}],
    }
    tool_search_result = {
        "type": "tool_search_output",
        "results": [{"text": "Balance lookup docs"}],
    }
    hydrated_items = [
        DummyRunItem(tool_search_call, type="tool_search_call_item"),
        DummyRunItem(tool_search_result, type="tool_search_output_item"),
    ]
    rebuilt_items = [
        DummyRunItem(dict(tool_search_call), type="tool_search_call_item"),
        DummyRunItem(dict(tool_search_result), type="tool_search_output_item"),
    ]

    tracker.hydrate_from_state(
        original_input=[],
        generated_items=cast(list[Any], hydrated_items),
        model_responses=[],
    )

    prepared = tracker.prepare_input(
        original_input=[],
        generated_items=cast(list[Any], rebuilt_items),
    )

    assert prepared == []

def test_hydrate_from_state_skips_restored_tool_search_items_when_created_by_is_stripped() -> None:
    tracker = OpenAIServerConversationTracker(
        conversation_id="conv2d-created-by", previous_response_id=None
    )
    session_items = [
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": "tool_search_call_1",
                "arguments": {"query": "account balance"},
                "execution": "server",
                "status": "completed",
                "created_by": "server",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_output",
                "call_id": "tool_search_call_1",
                "execution": "server",
                "status": "completed",
                "tools": [],
                "created_by": "server",
            },
        ),
    ]

    tracker.hydrate_from_state(
        original_input=[],
        generated_items=[],
        model_responses=[],
        session_items=session_items,
    )

    prepared = tracker.prepare_input(
        original_input=[],
        generated_items=cast(
            list[RunItem],
            [
                DummyRunItem(
                    {
                        "type": "tool_search_call",
                        "call_id": "tool_search_call_1",
                        "arguments": {"query": "account balance"},
                        "execution": "server",
                        "status": "completed",
                    },
                    type="tool_search_call_item",
                ),
                DummyRunItem(
                    {
                        "type": "tool_search_output",
                        "call_id": "tool_search_call_1",
                        "execution": "server",
                        "status": "completed",
                        "tools": [],
                    },
                    type="tool_search_output_item",
                ),
            ],
        ),
    )

    assert prepared == []

def test_hydrate_from_state_skips_restored_tool_search_items_when_only_ids_differ() -> None:
    tracker = OpenAIServerConversationTracker(
        conversation_id="conv2d-ids-only", previous_response_id=None
    )
    session_items = [
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "id": "tool_search_call_saved",
                "arguments": {"query": "account balance"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_output",
                "id": "tool_search_output_saved",
                "execution": "server",
                "status": "completed",
                "tools": [],
            },
        ),
    ]

    tracker.hydrate_from_state(
        original_input=[],
        generated_items=[],
        model_responses=[],
        session_items=session_items,
    )

    prepared = tracker.prepare_input(
        original_input=[],
        generated_items=cast(
            list[RunItem],
            [
                DummyRunItem(
                    {
                        "type": "tool_search_call",
                        "arguments": {"query": "account balance"},
                        "execution": "server",
                        "status": "completed",
                    },
                    type="tool_search_call_item",
                ),
                DummyRunItem(
                    {
                        "type": "tool_search_output",
                        "execution": "server",
                        "status": "completed",
                        "tools": [],
                    },
                    type="tool_search_output_item",
                ),
            ],
        ),
    )

    assert prepared == []

def test_prepare_input_keeps_repeated_tool_search_items_with_new_ids() -> None:
    tracker = OpenAIServerConversationTracker(
        conversation_id="conv2d-repeated-search", previous_response_id=None
    )

    prior_response = object.__new__(ModelResponse)
    prior_response.output = [
        cast(
            Any,
            {
                "type": "tool_search_call",
                "id": "tool_search_call_saved",
                "arguments": {"query": "account balance"},
                "execution": "server",
                "status": "completed",
                "created_by": "server",
            },
        ),
        cast(
            Any,
            {
                "type": "tool_search_output",
                "id": "tool_search_output_saved",
                "execution": "server",
                "status": "completed",
                "tools": [],
                "created_by": "server",
            },
        ),
    ]
    prior_response.usage = Usage()
    prior_response.response_id = "resp-tool-search-repeat-1"

    tracker.track_server_items(prior_response)

    repeated_items = [
        DummyRunItem(
            {
                "type": "tool_search_call",
                "id": "tool_search_call_repeat",
                "arguments": {"query": "account balance"},
                "execution": "server",
                "status": "completed",
            },
            type="tool_search_call_item",
        ),
        DummyRunItem(
            {
                "type": "tool_search_output",
                "id": "tool_search_output_repeat",
                "execution": "server",
                "status": "completed",
                "tools": [],
            },
            type="tool_search_output_item",
        ),
    ]

    prepared = tracker.prepare_input(
        original_input=[],
        generated_items=cast(list[Any], repeated_items),
    )

    assert prepared == [
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "id": "tool_search_call_repeat",
                "arguments": {"query": "account balance"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_output",
                "id": "tool_search_output_repeat",
                "execution": "server",
                "status": "completed",
                "tools": [],
            },
        ),
    ]

def test_track_server_items_skips_live_tool_search_items_on_next_prepare() -> None:
    tracker = OpenAIServerConversationTracker(conversation_id="conv2e", previous_response_id=None)
    tool_search_call = cast(
        Any,
        {
            "type": "tool_search_call",
            "call_id": "tool_search_call_live",
            "arguments": {"query": "account balance"},
            "execution": "server",
            "status": "completed",
            "created_by": "server",
        },
    )
    tool_search_result = cast(
        Any,
        {
            "type": "tool_search_output",
            "call_id": "tool_search_call_live",
            "execution": "server",
            "status": "completed",
            "tools": [],
            "created_by": "server",
        },
    )
    model_response = object.__new__(ModelResponse)
    model_response.output = [tool_search_call, tool_search_result]
    model_response.usage = Usage()
    model_response.response_id = "resp-tool-search"

    tracker.track_server_items(model_response)

    prepared = tracker.prepare_input(
        original_input=[],
        generated_items=cast(
            list[RunItem],
            [
                DummyRunItem(
                    {
                        "type": "tool_search_call",
                        "call_id": "tool_search_call_live",
                        "arguments": {"query": "account balance"},
                        "execution": "server",
                        "status": "completed",
                    },
                    type="tool_search_call_item",
                ),
                DummyRunItem(
                    {
                        "type": "tool_search_output",
                        "call_id": "tool_search_call_live",
                        "execution": "server",
                        "status": "completed",
                        "tools": [],
                    },
                    type="tool_search_output_item",
                ),
            ],
        ),
    )

    assert prepared == []

def test_track_server_items_filters_pending_tool_search_by_sanitized_fingerprint() -> None:
    tracker = OpenAIServerConversationTracker(
        conversation_id="conv2e-pending", previous_response_id=None
    )
    tracker.remaining_initial_input = [
        cast(
            TResponseInputItem,
            {
                "type": "tool_search_call",
                "call_id": "tool_search_pending",
                "arguments": {"query": "account balance"},
                "execution": "server",
                "status": "completed",
            },
        ),
        cast(TResponseInputItem, {"id": "keep-me", "type": "message"}),
    ]

    model_response = object.__new__(ModelResponse)
    model_response.output = [
        cast(
            Any,
            {
                "type": "tool_search_call",
                "call_id": "tool_search_pending",
                "arguments": {"query": "account balance"},
                "execution": "server",
                "status": "completed",
                "created_by": "server",
            },
        )
    ]
    model_response.usage = Usage()
    model_response.response_id = "resp-tool-search-pending"

    tracker.track_server_items(model_response)

    assert tracker.remaining_initial_input == [
        cast(TResponseInputItem, {"id": "keep-me", "type": "message"})
    ]

def test_prepare_input_applies_reasoning_item_id_policy_for_generated_items() -> None:
    tracker = OpenAIServerConversationTracker(
        conversation_id="conv7",
        previous_response_id=None,
        reasoning_item_id_policy="omit",
    )
    generated_items = [
        DummyRunItem(
            {
                "type": "reasoning",
                "id": "rs_turn_input",
                "content": [{"type": "input_text", "text": "reasoning trace"}],
            },
            type="reasoning_item",
        )
    ]

    prepared = tracker.prepare_input(
        original_input=[],
        generated_items=cast(list[Any], generated_items),
    )

    assert prepared == [
        cast(
            TResponseInputItem,
            {"type": "reasoning", "content": [{"type": "input_text", "text": "reasoning trace"}]},
        )
    ]

def test_prepare_input_does_not_resend_reasoning_item_after_marking_omitted_id_as_sent() -> None:
    tracker = OpenAIServerConversationTracker(
        conversation_id="conv8",
        previous_response_id=None,
        reasoning_item_id_policy="omit",
    )
    generated_items = [
        DummyRunItem(
            {
                "type": "reasoning",
                "id": "rs_turn_input",
                "content": [{"type": "input_text", "text": "reasoning trace"}],
            },
            type="reasoning_item",
        )
    ]

    first_prepared = tracker.prepare_input(
        original_input=[],
        generated_items=cast(list[Any], generated_items),
    )
    assert first_prepared == [
        cast(
            TResponseInputItem,
            {"type": "reasoning", "content": [{"type": "input_text", "text": "reasoning trace"}]},
        )
    ]

    tracker.mark_input_as_sent(first_prepared)

    second_prepared = tracker.prepare_input(
        original_input=[],
        generated_items=cast(list[Any], generated_items),
    )
    assert second_prepared == []


# --- tests/test_stream_events.py ---

def test_stream_step_result_to_queue_uses_new_step_items() -> None:
    agent = Agent(name="StreamHelper")
    queue: asyncio.Queue[Any] = asyncio.Queue()

    tool_search_item = ToolSearchCallItem(
        agent=agent,
        raw_item={
            "type": "tool_search_call",
            "queries": [{"search_term": "docs"}],
        },
    )
    step_result = cast(Any, type("StepResult", (), {"new_step_items": [tool_search_item]})())

    stream_step_result_to_queue(step_result, queue)

    event = queue.get_nowait()
    assert event.name == "tool_search_called"


# --- tests/test_tool_context.py ---

def test_tool_context_requires_fields() -> None:
    ctx: RunContextWrapper[dict[str, object]] = RunContextWrapper(context={})
    with pytest.raises(ValueError):
        ToolContext.from_agent_context(ctx, tool_call_id="call-1")

def test_tool_context_missing_defaults_raise() -> None:
    base_ctx: RunContextWrapper[dict[str, object]] = RunContextWrapper(context={})
    with pytest.raises(ValueError):
        ToolContext(context=base_ctx.context, tool_call_id="call-1", tool_arguments="")
    with pytest.raises(ValueError):
        ToolContext(context=base_ctx.context, tool_name="name", tool_arguments="")
    with pytest.raises(ValueError):
        ToolContext(context=base_ctx.context, tool_name="name", tool_call_id="call-1")

def test_tool_context_from_tool_context_inherits_run_config() -> None:
    original_call = ResponseFunctionToolCall(
        type="function_call",
        name="test_tool",
        call_id="call-3",
        arguments="{}",
    )
    derived_call = ResponseFunctionToolCall(
        type="function_call",
        name="test_tool",
        call_id="call-4",
        arguments="{}",
    )
    parent_run_config = RunConfig(model="gpt-4.1-mini")
    parent_context: ToolContext[dict[str, object]] = ToolContext(
        context={},
        tool_name="test_tool",
        tool_call_id="call-3",
        tool_arguments="{}",
        tool_call=original_call,
        run_config=parent_run_config,
    )

    derived_context = ToolContext.from_agent_context(
        parent_context,
        tool_call_id="call-4",
        tool_call=derived_call,
    )

    assert derived_context.run_config is parent_run_config

def test_tool_context_from_agent_context_prefers_explicit_run_config() -> None:
    tool_call = ResponseFunctionToolCall(
        type="function_call",
        name="test_tool",
        call_id="call-1",
        arguments="{}",
    )
    ctx = make_context_wrapper()
    explicit_run_config = RunConfig(model="gpt-4.1")

    tool_ctx = ToolContext.from_agent_context(
        ctx,
        tool_call_id="call-1",
        tool_call=tool_call,
        run_config=explicit_run_config,
    )

    assert tool_ctx.run_config is explicit_run_config

async def test_invoke_function_tool_passes_plain_run_context_when_requested() -> None:
    captured_context: RunContextWrapper[str] | None = None

    async def on_invoke_tool(ctx: RunContextWrapper[str], _input: str) -> str:
        nonlocal captured_context
        captured_context = ctx
        return ctx.context

    function_tool = FunctionTool(
        name="plain_context_tool",
        description="test",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=on_invoke_tool,
    )
    tool_context = ToolContext(
        context="Stormy",
        usage=Usage(),
        tool_name="plain_context_tool",
        tool_call_id="call-1",
        tool_arguments="{}",
        agent=Agent(name="agent"),
        run_config=RunConfig(model="gpt-4.1-mini"),
        tool_input={"city": "Tokyo"},
    )

    result = await invoke_function_tool(
        function_tool=function_tool,
        context=tool_context,
        arguments="{}",
    )

    assert result == "Stormy"
    assert captured_context is not None
    assert not isinstance(captured_context, ToolContext)
    assert captured_context.context == "Stormy"
    assert captured_context.usage is tool_context.usage
    assert captured_context.tool_input == {"city": "Tokyo"}

async def test_invoke_function_tool_preserves_tool_context_when_requested() -> None:
    captured_context: ToolContext[str] | None = None

    async def on_invoke_tool(ctx: ToolContext[str], _input: str) -> str:
        nonlocal captured_context
        captured_context = ctx
        return ctx.tool_name

    function_tool = FunctionTool(
        name="tool_context_tool",
        description="test",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=on_invoke_tool,
    )
    tool_context = ToolContext(
        context="Stormy",
        usage=Usage(),
        tool_name="tool_context_tool",
        tool_call_id="call-2",
        tool_arguments="{}",
        agent=Agent(name="agent"),
        run_config=RunConfig(model="gpt-4.1-mini"),
    )

    result = await invoke_function_tool(
        function_tool=function_tool,
        context=tool_context,
        arguments="{}",
    )

    assert result == "tool_context_tool"
    assert captured_context is tool_context

async def test_invoke_function_tool_ignores_context_name_substrings_in_string_annotations() -> None:
    captured_context: object | None = None

    class MyRunContextWrapper:
        pass

    async def on_invoke_tool(ctx: "MyRunContextWrapper", _input: str) -> str:
        nonlocal captured_context
        captured_context = ctx
        return "ok"

    function_tool = FunctionTool(
        name="substring_context_tool",
        description="test",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=cast(Any, on_invoke_tool),
    )
    tool_context = ToolContext(
        context="Stormy",
        usage=Usage(),
        tool_name="substring_context_tool",
        tool_call_id="call-3",
        tool_arguments="{}",
    )

    result = await invoke_function_tool(
        function_tool=function_tool,
        context=tool_context,
        arguments="{}",
    )

    assert result == "ok"
    assert captured_context is tool_context

async def test_invoke_function_tool_ignores_annotated_string_metadata_when_matching_context() -> (
    None
):
    captured_context: ToolContext[str] | RunContextWrapper[str] | None = None

    async def on_invoke_tool(
        ctx: Annotated[RunContextWrapper[str], "ToolContext note"], _input: str
    ) -> str:
        nonlocal captured_context
        captured_context = ctx
        return ctx.context

    function_tool = FunctionTool(
        name="annotated_string_context_tool",
        description="test",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=on_invoke_tool,
    )
    tool_context = ToolContext(
        context="Stormy",
        usage=Usage(),
        tool_name="annotated_string_context_tool",
        tool_call_id="call-4",
        tool_arguments="{}",
        tool_input={"city": "Tokyo"},
    )

    result = await invoke_function_tool(
        function_tool=function_tool,
        context=tool_context,
        arguments="{}",
    )

    assert result == "Stormy"
    assert captured_context is not None
    assert not isinstance(captured_context, ToolContext)
    assert captured_context.tool_input == {"city": "Tokyo"}


# --- tests/test_tool_metadata.py ---

def test_tool_context_from_agent_context() -> None:
    ctx = RunContextWrapper(context={"foo": "bar"})
    tool_call = ToolContext.from_agent_context(
        ctx,
        tool_call_id="123",
        tool_call=type(
            "Call",
            (),
            {
                "name": "demo",
                "arguments": "{}",
            },
        )(),
    )
    assert tool_call.tool_name == "demo"


# --- tests/test_tool_use_behavior.py ---

async def test_no_tool_results_returns_not_final_output() -> None:
    # If there are no tool results at all, tool_use_behavior should not produce a final output.
    agent = Agent(name="test")
    result = await run_loop.check_for_final_output_from_tools(
        agent=agent,
        tool_results=[],
        context_wrapper=RunContextWrapper(context=None),
    )
    assert result.is_final_output is False
    assert result.final_output is None

async def test_run_llm_again_behavior() -> None:
    # With the default run_llm_again behavior, even with tools we still expect to keep running.
    agent = Agent(name="test", tool_use_behavior="run_llm_again")
    tool_results = [_make_function_tool_result(agent, "ignored")]
    result = await run_loop.check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
    )
    assert result.is_final_output is False
    assert result.final_output is None

async def test_stop_on_first_tool_behavior() -> None:
    # When tool_use_behavior is stop_on_first_tool, we should surface first tool output as final.
    agent = Agent(name="test", tool_use_behavior="stop_on_first_tool")
    tool_results = [
        _make_function_tool_result(agent, "first_tool_output"),
        _make_function_tool_result(agent, "ignored"),
    ]
    result = await run_loop.check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
    )
    assert result.is_final_output is True
    assert result.final_output == "first_tool_output"

async def test_custom_tool_use_behavior_sync() -> None:
    """If tool_use_behavior is a sync function, we should call it and propagate its return."""

    def behavior(
        context: RunContextWrapper, results: list[FunctionToolResult]
    ) -> ToolsToFinalOutputResult:
        assert len(results) == 3
        return ToolsToFinalOutputResult(is_final_output=True, final_output="custom")

    agent = Agent(name="test", tool_use_behavior=behavior)
    tool_results = [
        _make_function_tool_result(agent, "ignored1"),
        _make_function_tool_result(agent, "ignored2"),
        _make_function_tool_result(agent, "ignored3"),
    ]
    result = await run_loop.check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
    )
    assert result.is_final_output is True
    assert result.final_output == "custom"

async def test_custom_tool_use_behavior_async() -> None:
    """If tool_use_behavior is an async function, we should await it and propagate its return."""

    async def behavior(
        context: RunContextWrapper, results: list[FunctionToolResult]
    ) -> ToolsToFinalOutputResult:
        assert len(results) == 3
        return ToolsToFinalOutputResult(is_final_output=True, final_output="async_custom")

    agent = Agent(name="test", tool_use_behavior=behavior)
    tool_results = [
        _make_function_tool_result(agent, "ignored1"),
        _make_function_tool_result(agent, "ignored2"),
        _make_function_tool_result(agent, "ignored3"),
    ]
    result = await run_loop.check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
    )
    assert result.is_final_output is True
    assert result.final_output == "async_custom"

async def test_invalid_tool_use_behavior_raises() -> None:
    """If tool_use_behavior is invalid, we should raise a UserError."""
    agent = Agent(name="test")
    # Force an invalid value; mypy will complain, so ignore the type here.
    agent.tool_use_behavior = "bad_value"  # type: ignore[assignment]
    tool_results = [_make_function_tool_result(agent, "ignored")]
    with pytest.raises(UserError):
        await run_loop.check_for_final_output_from_tools(
            agent=agent,
            tool_results=tool_results,
            context_wrapper=RunContextWrapper(context=None),
        )

async def test_tool_names_to_stop_at_behavior() -> None:
    agent = Agent(
        name="test",
        tools=[
            get_function_tool("tool1", return_value="tool1_output"),
            get_function_tool("tool2", return_value="tool2_output"),
            get_function_tool("tool3", return_value="tool3_output"),
        ],
        tool_use_behavior={"stop_at_tool_names": ["tool1"]},
    )

    tool_results = [
        _make_function_tool_result(agent, "ignored1", "tool2"),
        _make_function_tool_result(agent, "ignored3", "tool3"),
    ]
    result = await run_loop.check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
    )
    assert result.is_final_output is False, "We should not have stopped at tool1"

    # Now test with a tool that matches the list
    tool_results = [
        _make_function_tool_result(agent, "output1", "tool1"),
        _make_function_tool_result(agent, "ignored2", "tool2"),
        _make_function_tool_result(agent, "ignored3", "tool3"),
    ]
    result = await run_loop.check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
    )
    assert result.is_final_output is True, "We should have stopped at tool1"
    assert result.final_output == "output1"

async def test_stop_at_tool_names_supports_public_and_qualified_names_for_namespaced_tools() -> (
    None
):
    namespaced_tool = tool_namespace(
        name="billing",
        description="Billing tools",
        tools=[function_tool(lambda account_id: account_id, name_override="lookup_account")],
    )[0]
    agent = Agent(
        name="test",
        tools=[namespaced_tool],
        tool_use_behavior={"stop_at_tool_names": ["lookup_account"]},
    )

    tool_results = [
        _make_function_tool_result(agent, "billing-output", tool=namespaced_tool),
    ]
    result = await run_loop.check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
    )
    assert result.is_final_output is True
    assert result.final_output == "billing-output"

    agent.tool_use_behavior = {"stop_at_tool_names": ["billing.lookup_account"]}
    result = await run_loop.check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
    )
    assert result.is_final_output is True


# --- tests/test_tool_use_tracker.py ---

def test_tool_use_tracker_as_serializable_uses_agent_map_or_runtime_snapshot() -> None:
    tracker = AgentToolUseTracker()
    tracker.agent_map = {"agent-a": {"tool-b", "tool-a"}}
    assert tracker.as_serializable() == {"agent-a": ["tool-a", "tool-b"]}

    runtime_tracker = AgentToolUseTracker()
    agent = Agent(name="runtime-agent")
    runtime_tracker.add_tool_use(agent, ["beta", "alpha"])
    assert runtime_tracker.as_serializable() == {"runtime-agent": ["alpha", "beta"]}

def test_tool_use_tracker_from_and_serialize_snapshots() -> None:
    hydrated = AgentToolUseTracker.from_serializable({"agent": ["tool-2", "tool-1"]})
    assert hydrated.agent_map == {"agent": {"tool-1", "tool-2"}}

    runtime_tracker = AgentToolUseTracker()
    agent = Agent(name="serialize-agent")
    runtime_tracker.add_tool_use(agent, ["one"])
    runtime_tracker.add_tool_use(agent, ["two"])
    assert serialize_tool_use_tracker(runtime_tracker) == {"serialize-agent": ["one", "two"]}

def test_record_used_tools_uses_trace_names_for_namespaced_and_deferred_functions() -> None:
    agent = Agent(name="tracked-agent")
    tracker = AgentToolUseTracker()

    billing_tool = tool_namespace(
        name="billing",
        description="Billing tools",
        tools=[function_tool(lambda customer_id: customer_id, name_override="lookup_account")],
    )[0]
    deferred_tool = function_tool(
        lambda city: city,
        name_override="get_weather",
        defer_loading=True,
    )

    tracker.record_used_tools(
        agent,
        [
            ToolRunFunction(
                function_tool=billing_tool,
                tool_call=cast(
                    ResponseFunctionToolCall,
                    get_function_tool_call("lookup_account", namespace="billing"),
                ),
            ),
            ToolRunFunction(
                function_tool=deferred_tool,
                tool_call=cast(
                    ResponseFunctionToolCall,
                    get_function_tool_call("get_weather", namespace="get_weather"),
                ),
            ),
        ],
    )

    assert tracker.as_serializable() == {"tracked-agent": ["billing.lookup_account", "get_weather"]}

def test_hydrate_tool_use_tracker_skips_unknown_agents() -> None:
    class _RunState:
        def get_tool_use_tracker_snapshot(self) -> dict[str, list[str]]:
            return {"known-agent": ["known_tool"], "missing-agent": ["missing_tool"]}

    starting_agent = Agent(name="known-agent")
    tracker = AgentToolUseTracker()

    hydrate_tool_use_tracker(
        tool_use_tracker=tracker,
        run_state=_RunState(),
        starting_agent=starting_agent,
    )

    assert tracker.has_used_tools(starting_agent)
    assert tracker.as_serializable() == {"known-agent": ["known_tool"]}
    assert "missing-agent" not in tracker.as_serializable()


# --- tests/test_usage.py ---

def test_usage_normalizes_chat_completions_types():
    # Chat Completions API uses PromptTokensDetails and CompletionTokensDetails,
    # while Usage expects InputTokensDetails and OutputTokensDetails (Responses API).
    # The BeforeValidator should convert between these types.

    prompt_details = PromptTokensDetails(audio_tokens=10, cached_tokens=50)
    completion_details = CompletionTokensDetails(
        accepted_prediction_tokens=5,
        audio_tokens=10,
        reasoning_tokens=100,
        rejected_prediction_tokens=2,
    )

    usage = Usage(
        requests=1,
        input_tokens=200,
        input_tokens_details=prompt_details,  # type: ignore[arg-type]
        output_tokens=150,
        output_tokens_details=completion_details,  # type: ignore[arg-type]
        total_tokens=350,
    )

    # Should convert to Responses API types, extracting the relevant fields
    assert isinstance(usage.input_tokens_details, InputTokensDetails)
    assert usage.input_tokens_details.cached_tokens == 50

    assert isinstance(usage.output_tokens_details, OutputTokensDetails)
    assert usage.output_tokens_details.reasoning_tokens == 100


# --- tests/utils/test_json.py ---

def test_to_dump_compatible():
    # Given a list of message dictionaries, ensure the returned list is a deep copy.
    input_iter = [
        ResponseOutputMessageParam(
            id="a75654dc-7492-4d1c-bce0-89e8312fbdd7",
            content=[
                ResponseOutputTextParam(
                    type="output_text",
                    text="Hey, what's up?",
                    annotations=[],
                    logprobs=[],
                )
            ].__iter__(),
            role="assistant",
            status="completed",
            type="message",
        )
    ].__iter__()
    # this fails if any of the properties are Iterable objects.
    # result = json.dumps(input_iter)
    result = json.dumps(_to_dump_compatible(input_iter))
    assert (
        result
        == """[{"id": "a75654dc-7492-4d1c-bce0-89e8312fbdd7", "content": [{"type": "output_text", "text": "Hey, what's up?", "annotations": [], "logprobs": []}], "role": "assistant", "status": "completed", "type": "message"}]"""  # noqa: E501
    )

