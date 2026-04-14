# microsoft/agent-framework
# 169 test functions with real LLM calls
# Source: https://github.com/microsoft/agent-framework


# --- python/packages/anthropic/tests/test_anthropic_client.py ---

async def test_anthropic_client_integration_streaming_chat() -> None:
    """Integration test for streaming chat completion."""
    client = AnthropicClient()

    messages = [Message(role="user", contents=["Count from 1 to 5."])]

    chunks = []
    async for chunk in client.get_response(messages=messages, stream=True, options={"max_tokens": 50}):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert any(chunk.contents for chunk in chunks)


# --- python/packages/core/tests/core/test_agents.py ---

def test_agent_type(agent: SupportsAgentRun) -> None:
    assert isinstance(agent, SupportsAgentRun)

async def test_agent_run(agent: SupportsAgentRun) -> None:
    response = await agent.run("test")
    assert response.messages[0].role == "assistant"
    assert response.messages[0].text == "Response"

async def test_agent_run_with_content(agent: SupportsAgentRun) -> None:
    response = await agent.run(Content.from_text("test"))
    assert response.messages[0].role == "assistant"
    assert response.messages[0].text == "Response"

async def test_agent_run_streaming(agent: SupportsAgentRun) -> None:
    async def collect_updates(
        updates: AsyncIterable[AgentResponseUpdate],
    ) -> list[AgentResponseUpdate]:
        return [u async for u in updates]

    updates = await collect_updates(agent.run("test", stream=True))
    assert len(updates) == 1
    assert updates[0].text == "Response"

def test_chat_client_agent_type(client: SupportsChatGetResponse) -> None:
    chat_client_agent = Agent(client=client)
    assert isinstance(chat_client_agent, SupportsAgentRun)

def test_agent_run_docstring_surfaces_raw_agent_runtime_docs() -> None:
    docstring = inspect.getdoc(Agent.run)

    assert docstring is not None
    assert "Run the agent with the given messages and options." in docstring
    assert "function_invocation_kwargs: Keyword arguments forwarded to tool invocation." in docstring
    assert "middleware: Optional per-run agent, chat, and function middleware." in docstring

def test_agent_run_is_defined_on_agent_class() -> None:
    signature = inspect.signature(Agent.run)

    assert Agent.run.__qualname__ == "Agent.run"
    assert "middleware" in signature.parameters

async def test_chat_client_agent_run(client: SupportsChatGetResponse) -> None:
    agent = Agent(client=client)

    result = await agent.run("Hello")

    assert result.text == "test response"

async def test_chat_client_agent_run_streaming(client: SupportsChatGetResponse) -> None:
    agent = Agent(client=client)

    result = await AgentResponse.from_update_generator(agent.run("Hello", stream=True))

    assert result.text == "test streaming response another update"

async def test_chat_client_agent_streaming_response_format_from_default_options(
    client: SupportsChatGetResponse,
) -> None:
    """AgentResponse.value must be parsed when response_format is set in default_options and streaming."""
    from pydantic import BaseModel

    class Greeting(BaseModel):
        greeting: str

    json_text = '{"greeting": "Hello"}'
    client.streaming_responses.append(  # type: ignore[attr-defined]
        [
            ChatResponseUpdate(
                contents=[Content.from_text(json_text)],
                role="assistant",
                finish_reason="stop",
            )
        ]
    )

    agent = Agent(client=client, default_options={"response_format": Greeting})
    stream = agent.run("Hello", stream=True)
    async for _ in stream:
        pass
    result = await stream.get_final_response()

    assert result.text == json_text
    assert result.value is not None
    assert isinstance(result.value, Greeting)
    assert result.value.greeting == "Hello"

async def test_chat_client_agent_streaming_response_format_from_run_options(
    client: SupportsChatGetResponse,
) -> None:
    """AgentResponse.value must be parsed when response_format is passed via run() options kwarg."""
    from pydantic import BaseModel

    class Greeting(BaseModel):
        greeting: str

    json_text = '{"greeting": "Hi"}'
    client.streaming_responses.append(  # type: ignore[attr-defined]
        [
            ChatResponseUpdate(
                contents=[Content.from_text(json_text)],
                role="assistant",
                finish_reason="stop",
            )
        ]
    )

    agent = Agent(client=client)
    stream = agent.run("Hello", stream=True, options={"response_format": Greeting})
    async for _ in stream:
        pass
    result = await stream.get_final_response()

    assert result.text == json_text
    assert result.value is not None
    assert isinstance(result.value, Greeting)
    assert result.value.greeting == "Hi"

async def test_chat_client_agent_response_format_dict_from_default_options(
    client: SupportsChatGetResponse,
) -> None:
    """AgentResponse.value should parse JSON dicts from default_options response_format."""
    json_text = json.dumps({"greeting": "Hello"})
    client.responses.append(ChatResponse(messages=Message(role="assistant", contents=[json_text])))  # type: ignore[attr-defined]

    agent = Agent(
        client=client,
        default_options={"response_format": {"type": "object", "properties": {"greeting": {"type": "string"}}}},
    )
    result = await agent.run("Hello")

    assert result.text == json_text
    assert result.value is not None
    assert isinstance(result.value, dict)
    assert result.value["greeting"] == "Hello"

async def test_chat_client_agent_streaming_response_format_dict_from_run_options(
    client: SupportsChatGetResponse,
) -> None:
    """Agent streaming should preserve mapping response_format and parse the final value as a dict."""
    json_text = json.dumps({"greeting": "Hi"})
    client.streaming_responses.append(  # type: ignore[attr-defined]
        [
            ChatResponseUpdate(
                contents=[Content.from_text(json_text)],
                role="assistant",
                finish_reason="stop",
            )
        ]
    )

    agent = Agent(client=client)
    stream = agent.run(
        "Hello",
        stream=True,
        options={"response_format": {"type": "object", "properties": {"greeting": {"type": "string"}}}},
    )
    async for _ in stream:
        pass
    result = await stream.get_final_response()

    assert result.text == json_text
    assert result.value is not None
    assert isinstance(result.value, dict)
    assert result.value["greeting"] == "Hi"

async def test_prepare_run_context_handles_function_kwargs(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    agent = Agent(client=chat_client_base)
    session = agent.create_session()

    ctx = await agent._prepare_run_context(  # type: ignore[reportPrivateUsage]
        messages="Hello",
        session=session,
        tools=None,
        options={
            "temperature": 0.4,
            "additional_function_arguments": {"from_options": "options-value"},
        },
        compaction_strategy=None,
        tokenizer=None,
        function_invocation_kwargs={"runtime_key": "runtime-value"},
        client_kwargs={"client_key": "client-value"},
    )

    assert ctx["chat_options"]["temperature"] == 0.4
    assert "additional_function_arguments" not in ctx["chat_options"]
    assert ctx["function_invocation_kwargs"]["from_options"] == "options-value"
    assert ctx["function_invocation_kwargs"]["runtime_key"] == "runtime-value"
    assert "session" not in ctx["function_invocation_kwargs"]
    assert ctx["client_kwargs"]["client_key"] == "client-value"
    assert ctx["client_kwargs"]["session"] is session

async def test_chat_agent_persists_history_per_service_call(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    provider = _RecordingHistoryProvider()

    @tool(name="lookup_weather", approval_mode="never_require")
    def lookup_weather(location: str) -> str:
        return f"Weather in {location}: sunny"

    session = AgentSession()
    session.state[provider.source_id] = {
        "messages": [
            Message(role="user", contents=["Earlier question"]),
            Message(role="assistant", contents=["Earlier answer"]),
        ]
    }
    chat_client_base.run_responses = [
        ChatResponse(
            messages=Message(
                role="assistant",
                contents=[
                    Content.from_function_call(
                        call_id="call_1",
                        name="lookup_weather",
                        arguments='{"location": "Seattle"}',
                    )
                ],
            ),
            response_id="resp_call_1",
        ),
        ChatResponse(
            messages=Message(role="assistant", contents=["It is sunny in Seattle."]), response_id="resp_call_2"
        ),
    ]

    agent = Agent(
        client=chat_client_base,
        tools=[lookup_weather],
        context_providers=[provider],
        require_per_service_call_history_persistence=True,
    )

    result = await agent.run("What's the weather in Seattle?", session=session)

    provider_state = session.state[provider.source_id]
    stored_messages = cast(list[Message], provider_state["messages"])

    assert result.text == "It is sunny in Seattle."
    assert result.response_id is None
    assert chat_client_base.call_count == 2
    assert provider_state["get_call_count"] == 2
    assert provider_state["save_call_count"] == 2
    assert stored_messages[-1].text == "It is sunny in Seattle."
    assert session.service_session_id is None

async def test_chat_agent_persists_history_per_service_call_streaming(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    provider = _RecordingHistoryProvider()

    @tool(name="lookup_weather", approval_mode="never_require")
    def lookup_weather(location: str) -> str:
        return f"Weather in {location}: sunny"

    session = AgentSession()
    session.state[provider.source_id] = {
        "messages": [
            Message(role="user", contents=["Earlier question"]),
            Message(role="assistant", contents=["Earlier answer"]),
        ]
    }
    chat_client_base.streaming_responses = [
        [
            ChatResponseUpdate(
                contents=[
                    Content.from_function_call(
                        call_id="call_1",
                        name="lookup_weather",
                        arguments='{"location": "Seattle"}',
                    )
                ],
                role="assistant",
                finish_reason="stop",
                response_id="resp_call_1",
            )
        ],
        [
            ChatResponseUpdate(
                contents=[Content.from_text("It is sunny in Seattle.")],
                role="assistant",
                finish_reason="stop",
                response_id="resp_call_2",
            )
        ],
    ]

    agent = Agent(
        client=chat_client_base,
        tools=[lookup_weather],
        context_providers=[provider],
        require_per_service_call_history_persistence=True,
    )

    stream = agent.run("What's the weather in Seattle?", session=session, stream=True)
    async for _ in stream:
        pass
    result = await stream.get_final_response()

    provider_state = session.state[provider.source_id]
    stored_messages = cast(list[Message], provider_state["messages"])

    assert result.text == "It is sunny in Seattle."
    assert result.response_id is None
    assert chat_client_base.call_count == 2
    assert provider_state["get_call_count"] == 2
    assert provider_state["save_call_count"] == 2
    assert stored_messages[-1].text == "It is sunny in Seattle."
    assert session.service_session_id is None

async def test_streaming_per_service_call_persistence_hides_response_id_from_after_run(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    provider = _ResponseIdRecordingHistoryProvider()

    @tool(name="lookup_weather", approval_mode="never_require")
    def lookup_weather(location: str) -> str:
        return f"Weather in {location}: sunny"

    session = AgentSession()
    session.state[provider.source_id] = {"messages": []}
    chat_client_base.streaming_responses = [
        [
            ChatResponseUpdate(
                contents=[
                    Content.from_function_call(
                        call_id="call_1",
                        name="lookup_weather",
                        arguments='{"location": "Seattle"}',
                    )
                ],
                role="assistant",
                finish_reason="stop",
                response_id="resp_call_1",
            )
        ],
        [
            ChatResponseUpdate(
                contents=[Content.from_text("It is sunny in Seattle.")],
                role="assistant",
                finish_reason="stop",
                response_id="resp_call_2",
            )
        ],
    ]

    agent = Agent(
        client=chat_client_base,
        tools=[lookup_weather],
        context_providers=[provider],
        require_per_service_call_history_persistence=True,
    )

    stream = agent.run("What's the weather in Seattle?", session=session, stream=True)
    async for _ in stream:
        pass
    result = await stream.get_final_response()

    provider_state = session.state[provider.source_id]

    assert result.response_id is None
    assert provider_state["response_ids"] == [None, None]

async def test_per_service_call_persistence_uses_real_service_storage_when_client_stores_by_default(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    provider = _RecordingHistoryProvider()

    @tool(name="lookup_weather", approval_mode="never_require")
    def lookup_weather(location: str) -> str:
        return f"Weather in {location}: sunny"

    chat_client_base.STORES_BY_DEFAULT = True  # type: ignore[attr-defined]

    session = AgentSession()
    session.state[provider.source_id] = {"messages": []}
    chat_client_base.run_responses = [
        ChatResponse(
            messages=Message(
                role="assistant",
                contents=[
                    Content.from_function_call(
                        call_id="call_1",
                        name="lookup_weather",
                        arguments='{"location": "Seattle"}',
                    )
                ],
            ),
            conversation_id="resp_service_managed",
            response_id="resp_call_1",
        ),
        ChatResponse(
            messages=Message(role="assistant", contents=["It is sunny in Seattle."]),
            conversation_id="resp_service_managed",
            response_id="resp_call_2",
        ),
    ]

    agent = Agent(
        client=chat_client_base,
        tools=[lookup_weather],
        context_providers=[provider],
        require_per_service_call_history_persistence=True,
    )

    result = await agent.run("What's the weather in Seattle?", session=session)

    provider_state = session.state[provider.source_id]

    assert result.text == "It is sunny in Seattle."
    assert result.response_id == "resp_call_2"
    assert chat_client_base.call_count == 2
    assert "get_call_count" not in provider_state
    assert "save_call_count" not in provider_state
    assert session.service_session_id == "resp_service_managed"

async def test_chat_agent_without_per_service_call_persistence_preserves_response_id(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    chat_client_base.run_responses = [
        ChatResponse(
            messages=Message(role="assistant", contents=["Hello"]),
            response_id="resp_call_1",
        )
    ]

    agent = Agent(
        client=chat_client_base,
        context_providers=[InMemoryHistoryProvider()],
    )

    result = await agent.run("Hello", session=AgentSession(), options={"store": False})

    assert result.response_id == "resp_call_1"

async def test_per_service_call_persistence_rejects_real_service_conversation_id(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    provider = _RecordingHistoryProvider()
    chat_client_base.STORES_BY_DEFAULT = True  # type: ignore[attr-defined]
    session = AgentSession()
    session.state[provider.source_id] = {"messages": []}
    chat_client_base.run_responses = [
        ChatResponse(
            messages=Message(role="assistant", contents=["Hello"]),
            conversation_id="resp_service_managed",
        )
    ]

    agent = Agent(
        client=chat_client_base,
        context_providers=[provider],
        require_per_service_call_history_persistence=True,
    )

    with pytest.raises(
        ChatClientInvalidResponseException,
        match="require_per_service_call_history_persistence cannot be used",
    ):
        await agent.run("Hello", session=session, options={"store": False})

async def test_per_service_call_persistence_rejects_existing_conversation_id_when_service_not_storing_history(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    provider = _RecordingHistoryProvider()
    session = AgentSession()
    session.state[provider.source_id] = {"messages": []}

    agent = Agent(
        client=chat_client_base,
        context_providers=[provider],
        require_per_service_call_history_persistence=True,
    )

    with pytest.raises(
        AgentInvalidRequestException,
        match="require_per_service_call_history_persistence cannot be used",
    ):
        await agent.run("Hello", session=session, options={"store": False, "conversation_id": "existing_conversation"})

async def test_chat_client_agent_updates_existing_session_id_non_streaming(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    chat_client_base.run_responses = [
        ChatResponse(
            messages=[Message(role="assistant", contents=[Content.from_text("test response")])],
            conversation_id="resp_new_123",
        )
    ]

    agent = Agent(client=chat_client_base)
    session = agent.get_session(service_session_id="resp_old_123")

    await agent.run("Hello", session=session)
    assert session.service_session_id == "resp_new_123"

async def test_chat_client_agent_update_session_id_streaming_uses_conversation_id(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    chat_client_base.streaming_responses = [
        [
            ChatResponseUpdate(
                contents=[Content.from_text("stream part 1")],
                role="assistant",
                response_id="resp_stream_123",
                conversation_id="conv_stream_456",
            ),
            ChatResponseUpdate(
                contents=[Content.from_text(" stream part 2")],
                role="assistant",
                response_id="resp_stream_123",
                conversation_id="conv_stream_456",
                finish_reason="stop",
            ),
        ]
    ]

    agent = Agent(client=chat_client_base)
    session = agent.create_session()

    stream = agent.run("Hello", session=session, stream=True)
    async for _ in stream:
        pass
    result = await stream.get_final_response()
    assert result.text == "stream part 1 stream part 2"
    assert session.service_session_id == "conv_stream_456"

async def test_chat_client_agent_updates_existing_session_id_streaming(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    chat_client_base.streaming_responses = [
        [
            ChatResponseUpdate(
                contents=[Content.from_text("stream part 1")],
                role="assistant",
                response_id="resp_stream_123",
                conversation_id="resp_new_456",
            ),
            ChatResponseUpdate(
                contents=[Content.from_text(" stream part 2")],
                role="assistant",
                response_id="resp_stream_123",
                conversation_id="resp_new_456",
                finish_reason="stop",
            ),
        ]
    ]

    agent = Agent(client=chat_client_base)
    session = agent.get_session(service_session_id="resp_old_456")

    stream = agent.run("Hello", session=session, stream=True)
    async for _ in stream:
        pass
    await stream.get_final_response()
    assert session.service_session_id == "resp_new_456"

async def test_chat_client_agent_update_session_id_streaming_does_not_use_response_id(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    chat_client_base.streaming_responses = [
        [
            ChatResponseUpdate(
                contents=[Content.from_text("stream response without conversation id")],
                role="assistant",
                response_id="resp_only_123",
                finish_reason="stop",
            ),
        ]
    ]

    agent = Agent(client=chat_client_base)
    session = agent.create_session()

    stream = agent.run("Hello", session=session, stream=True)
    async for _ in stream:
        pass
    result = await stream.get_final_response()
    assert result.text == "stream response without conversation id"
    assert session.service_session_id is None

async def test_chat_client_agent_streaming_session_id_set_without_get_final_response(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    """Test that session.service_session_id is set during streaming iteration.

    This verifies the eager propagation of conversation_id via transform hook,
    which is needed for multi-turn flows (e.g. hosted MCP approval) where the
    user iterates the stream and then makes a follow-up call without calling
    get_final_response().
    """
    chat_client_base.streaming_responses = [
        [
            ChatResponseUpdate(
                contents=[Content.from_text("part 1")],
                role="assistant",
                response_id="resp_123",
                conversation_id="resp_123",
            ),
            ChatResponseUpdate(
                contents=[Content.from_text(" part 2")],
                role="assistant",
                response_id="resp_123",
                conversation_id="resp_123",
                finish_reason="stop",
            ),
        ]
    ]

    agent = Agent(client=chat_client_base)
    session = agent.create_session()
    assert session.service_session_id is None

    # Only iterate — do NOT call get_final_response()
    async for _ in agent.run("Hello", session=session, stream=True):
        pass

    assert session.service_session_id == "resp_123"

async def test_chat_client_agent_streaming_session_history_saved_without_get_final_response(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    """Test that session history is saved after streaming iteration without get_final_response().

    Auto-finalization on iteration completion should trigger after_run providers,
    persisting conversation history to the session.
    """
    from agent_framework._sessions import InMemoryHistoryProvider

    chat_client_base.streaming_responses = [
        [
            ChatResponseUpdate(
                contents=[Content.from_text("Hello Alice!")],
                role="assistant",
                response_id="resp_1",
                finish_reason="stop",
            ),
        ]
    ]

    agent = Agent(client=chat_client_base)
    session = agent.create_session()

    # Only iterate — do NOT call get_final_response()
    async for _ in agent.run("My name is Alice", session=session, stream=True):
        pass

    chat_messages: list[Message] = session.state.get(InMemoryHistoryProvider.DEFAULT_SOURCE_ID, {}).get("messages", [])
    assert len(chat_messages) == 2
    assert chat_messages[0].text == "My name is Alice"
    assert chat_messages[1].text == "Hello Alice!"

async def test_chat_client_agent_update_session_messages(
    client: SupportsChatGetResponse,
) -> None:
    from agent_framework._sessions import InMemoryHistoryProvider

    agent = Agent(client=client)
    session = agent.create_session()

    result = await agent.run("Hello", session=session)
    assert result.text == "test response"

    assert session.service_session_id is None

    chat_messages: list[Message] = session.state.get(InMemoryHistoryProvider.DEFAULT_SOURCE_ID, {}).get("messages", [])

    assert chat_messages is not None
    assert len(chat_messages) == 2
    assert chat_messages[0].text == "Hello"
    assert chat_messages[1].text == "test response"

async def test_chat_client_agent_default_author_name(
    client: SupportsChatGetResponse,
) -> None:
    # Name is not specified here, so default name should be used
    agent = Agent(client=client)

    result = await agent.run("Hello")
    assert result.text == "test response"
    assert result.messages[0].author_name == "UnnamedAgent"

async def test_chat_client_agent_author_name_as_agent_name(
    client: SupportsChatGetResponse,
) -> None:
    # Name is specified here, so it should be used as author name
    agent = Agent(client=client, name="TestAgent")

    result = await agent.run("Hello")
    assert result.text == "test response"
    assert result.messages[0].author_name == "TestAgent"

async def test_chat_client_agent_author_name_is_used_from_response(
    chat_client_base: SupportsChatGetResponse,
) -> None:
    chat_client_base.run_responses = [
        ChatResponse(
            messages=[
                Message(
                    role="assistant",
                    contents=[Content.from_text("test response")],
                    author_name="TestAuthor",
                )
            ]
        )
    ]

    agent = Agent(client=chat_client_base, tools={"type": "code_interpreter"})

    result = await agent.run("Hello")
    assert result.text == "test response"
    assert result.messages[0].author_name == "TestAuthor"

async def test_chat_agent_as_tool_with_stream_callback(
    client: SupportsChatGetResponse,
) -> None:
    """Test as_tool with stream callback functionality."""
    agent = Agent(client=client, name="StreamingAgent")

    # Collect streaming updates
    collected_updates: list[AgentResponseUpdate] = []

    def stream_callback(update: AgentResponseUpdate) -> None:
        collected_updates.append(update)

    tool = agent.as_tool(stream_callback=stream_callback)

    # Execute the tool
    result = await tool.invoke(arguments={"task": "Hello"})

    # Should have collected streaming updates
    assert len(collected_updates) > 0
    assert isinstance(result, list)
    result_text = result[0].text
    # Result should be concatenation of all streaming updates
    expected_text = "".join(update.text for update in collected_updates)
    assert result_text == expected_text

async def test_chat_agent_as_tool_with_custom_arg_name(
    client: SupportsChatGetResponse,
) -> None:
    """Test as_tool with custom argument name."""
    agent = Agent(client=client, name="CustomArgAgent")

    tool = agent.as_tool(arg_name="prompt", arg_description="Custom prompt input")

    # Test that the custom argument name works
    result = await tool.invoke(arguments={"prompt": "Test prompt"})
    assert isinstance(result, list)
    assert result[0].text == "test streaming response another update"

async def test_chat_agent_as_tool_with_async_stream_callback(
    client: SupportsChatGetResponse,
) -> None:
    """Test as_tool with async stream callback functionality."""
    agent = Agent(client=client, name="AsyncStreamingAgent")

    # Collect streaming updates using an async callback
    collected_updates: list[AgentResponseUpdate] = []

    async def async_stream_callback(update: AgentResponseUpdate) -> None:
        collected_updates.append(update)

    tool = agent.as_tool(stream_callback=async_stream_callback)

    # Execute the tool
    result = await tool.invoke(arguments={"task": "Hello"})

    # Should have collected streaming updates
    assert len(collected_updates) > 0
    assert isinstance(result, list)
    result_text = result[0].text
    # Result should be concatenation of all streaming updates
    expected_text = "".join(update.text for update in collected_updates)
    assert result_text == expected_text

async def test_chat_agent_as_tool_propagate_session_true(client: SupportsChatGetResponse) -> None:
    """Test that propagate_session=True forwards the session to the sub-agent."""
    agent = Agent(client=client, name="SubAgent", description="Sub agent")
    tool = agent.as_tool(propagate_session=True)

    parent_session = AgentSession(session_id="parent-session-123")
    parent_session.state["shared_key"] = "shared_value"

    original_run = agent.run
    captured_session = None

    def capturing_run(*args: Any, **kwargs: Any) -> Any:
        nonlocal captured_session
        captured_session = kwargs.get("session")
        return original_run(*args, **kwargs)

    agent.run = capturing_run  # type: ignore[assignment, method-assign]

    await tool.invoke(
        context=FunctionInvocationContext(
            function=tool,
            arguments={"task": "Hello"},
            session=parent_session,
        )
    )

    assert captured_session is parent_session
    assert captured_session.session_id == "parent-session-123"
    assert captured_session.state["shared_key"] == "shared_value"

async def test_chat_agent_as_tool_propagate_session_false_by_default(client: SupportsChatGetResponse) -> None:
    """Test that propagate_session defaults to False and does not forward the session."""
    agent = Agent(client=client, name="SubAgent", description="Sub agent")
    tool = agent.as_tool()  # default: propagate_session=False

    parent_session = AgentSession(session_id="parent-session-456")

    original_run = agent.run
    captured_session = None

    def capturing_run(*args: Any, **kwargs: Any) -> Any:
        nonlocal captured_session
        captured_session = kwargs.get("session")
        return original_run(*args, **kwargs)

    agent.run = capturing_run  # type: ignore[assignment, method-assign]

    await tool.invoke(
        context=FunctionInvocationContext(
            function=tool,
            arguments={"task": "Hello"},
            session=parent_session,
        )
    )

    assert captured_session is None

async def test_chat_agent_as_tool_propagate_session_shares_state(client: SupportsChatGetResponse) -> None:
    """Test that a propagated session allows the sub-agent to read and write parent state."""
    agent = Agent(client=client, name="SubAgent", description="Sub agent")
    tool = agent.as_tool(propagate_session=True)

    parent_session = AgentSession(session_id="shared-session")
    parent_session.state["counter"] = 0

    original_run = agent.run
    captured_session = None

    def capturing_run(*args: Any, **kwargs: Any) -> Any:
        nonlocal captured_session
        captured_session = kwargs.get("session")
        if captured_session:
            captured_session.state["counter"] += 1
        return original_run(*args, **kwargs)

    agent.run = capturing_run  # type: ignore[assignment, method-assign]

    await tool.invoke(
        context=FunctionInvocationContext(
            function=tool,
            arguments={"task": "Hello"},
            session=parent_session,
        )
    )

    assert parent_session.state["counter"] == 1

async def test_agent_run_raises_on_local_and_agent_mcp_name_conflict(chat_client_base: Any) -> None:
    local_tool = FunctionTool(
        func=lambda: "local",
        name="delete_all_data",
        description="Local protected tool",
        approval_mode="always_require",
    )
    agent = Agent(
        client=chat_client_base,
        name="TestAgent",
        tools=[_ConnectedMCPTool(name="dangerous-mcp", function_names=["delete_all_data"])],
    )

    with raises(ValueError, match="tool_name_prefix"):
        await agent.run("hello", tools=[local_tool])

async def test_agent_run_raises_on_runtime_local_and_runtime_mcp_name_conflict(chat_client_base: Any) -> None:
    local_tool = FunctionTool(
        func=lambda: "local",
        name="delete_all_data",
        description="Local protected tool",
        approval_mode="always_require",
    )
    runtime_mcp = _ConnectedMCPTool(name="dangerous-mcp", function_names=["delete_all_data"])
    agent = Agent(client=chat_client_base, name="TestAgent")

    with raises(ValueError, match="tool_name_prefix"):
        await agent.run("hello", tools=[local_tool, runtime_mcp])

async def test_agent_run_raises_on_duplicate_agent_mcp_names(chat_client_base: Any) -> None:
    agent = Agent(
        client=chat_client_base,
        name="TestAgent",
        tools=[
            _ConnectedMCPTool(name="docs-mcp", function_names=["search"]),
            _ConnectedMCPTool(name="github-mcp", function_names=["search"]),
        ],
    )

    with raises(ValueError, match="tool_name_prefix"):
        await agent.run("hello")

async def test_agent_run_accepts_prefixed_mcp_tools(chat_client_base: Any) -> None:
    captured_options: list[dict[str, Any]] = []

    original_inner = chat_client_base._inner_get_response

    async def capturing_inner(
        *, messages: MutableSequence[Message], options: dict[str, Any], **kwargs: Any
    ) -> ChatResponse:
        captured_options.append(dict(options))
        return await original_inner(messages=messages, options=options, **kwargs)

    chat_client_base._inner_get_response = capturing_inner

    local_tool = FunctionTool(func=lambda: "local", name="search", description="Local search tool")
    agent = Agent(
        client=chat_client_base,
        name="TestAgent",
        tools=[_ConnectedMCPTool(name="docs-mcp", function_names=["search"], tool_name_prefix="docs")],
    )

    await agent.run("hello", tools=[local_tool])

    tool_names = [tool.name for tool in captured_options[0]["tools"]]
    assert tool_names == ["search", "docs_search"]

async def test_agent_tool_without_context_does_not_receive_session(chat_client_base: Any) -> None:
    """Verify tools without FunctionInvocationContext no longer receive injected session kwargs."""

    captured: dict[str, Any] = {}

    @tool(name="echo_session_info", approval_mode="never_require")
    def echo_session_info(text: str, **kwargs: Any) -> str:  # type: ignore[reportUnknownParameterType]
        session = kwargs.get("session")
        captured["has_session"] = session is not None
        captured["has_state"] = session.state is not None if isinstance(session, AgentSession) else False
        return f"echo: {text}"

    chat_client_base.run_responses = [
        ChatResponse(
            messages=Message(
                role="assistant",
                contents=[
                    Content.from_function_call(
                        call_id="1",
                        name="echo_session_info",
                        arguments='{"text": "hello"}',
                    )
                ],
            )
        ),
        ChatResponse(messages=Message(role="assistant", contents=["done"])),
    ]

    agent = Agent(client=chat_client_base, tools=[echo_session_info])
    session = agent.create_session()

    result = await agent.run("hello", session=session)

    assert result.text == "done"
    assert captured.get("has_session") is False
    assert captured.get("has_state") is False

async def test_agent_tool_receives_explicit_session_via_function_invocation_context_kwargs(
    chat_client_base: Any,
) -> None:
    """Verify ctx-based tools receive the session via FunctionInvocationContext.session."""

    captured: dict[str, Any] = {}

    @tool(name="capture_session_context", approval_mode="never_require")
    def capture_session_context(text: str, ctx: FunctionInvocationContext) -> str:
        captured["session"] = ctx.session
        captured["has_state"] = ctx.session.state is not None if isinstance(ctx.session, AgentSession) else False
        return f"echo: {text}"

    chat_client_base.run_responses = [
        ChatResponse(
            messages=Message(
                role="assistant",
                contents=[
                    Content.from_function_call(
                        call_id="1",
                        name="capture_session_context",
                        arguments='{"text": "hello"}',
                    )
                ],
            )
        ),
        ChatResponse(messages=Message(role="assistant", contents=["done"])),
    ]

    agent = Agent(client=chat_client_base, tools=[capture_session_context])
    session = agent.create_session()

    result = await agent.run("hello", session=session)

    assert result.text == "done"
    assert captured["session"] is session
    assert captured["has_state"] is True

async def test_chat_agent_tool_choice_run_level_overrides_agent_level(chat_client_base: Any, tool_tool: Any) -> None:
    """Verify that tool_choice passed to run() overrides agent-level tool_choice."""

    captured_options: list[dict[str, Any]] = []

    # Store the original inner method
    original_inner = chat_client_base._inner_get_response

    async def capturing_inner(
        *, messages: MutableSequence[Message], options: dict[str, Any], **kwargs: Any
    ) -> ChatResponse:
        captured_options.append(options)
        return await original_inner(messages=messages, options=options, **kwargs)

    chat_client_base._inner_get_response = capturing_inner

    # Create agent with agent-level tool_choice="auto" and a tool (tools required for tool_choice to be meaningful)
    agent = Agent(
        client=chat_client_base,
        tools=[tool_tool],
        default_options={"tool_choice": "auto"},
    )

    # Run with run-level tool_choice="required"
    await agent.run("Hello", options={"tool_choice": "required"})

    # Verify the client received tool_choice="required", not "auto"
    assert len(captured_options) >= 1
    assert captured_options[0]["tool_choice"] == "required"

async def test_chat_agent_tool_choice_agent_level_used_when_run_level_not_specified(
    chat_client_base: Any, tool_tool: Any
) -> None:
    """Verify that agent-level tool_choice is used when run() doesn't specify one."""
    captured_options: list[ChatOptions] = []

    original_inner = chat_client_base._inner_get_response

    async def capturing_inner(
        *, messages: MutableSequence[Message], options: dict[str, Any], **kwargs: Any
    ) -> ChatResponse:
        captured_options.append(options)
        return await original_inner(messages=messages, options=options, **kwargs)

    chat_client_base._inner_get_response = capturing_inner

    # Create agent with agent-level tool_choice="required" and a tool
    agent = Agent(
        client=chat_client_base,
        tools=[tool_tool],
        default_options={"tool_choice": "required"},
    )

    # Run without specifying tool_choice
    await agent.run("Hello")

    # Verify the client received tool_choice="required" from agent-level
    assert len(captured_options) >= 1
    assert captured_options[0]["tool_choice"] == "required"
    # older code compared to ToolMode constants; ensure value is 'required'
    assert captured_options[0]["tool_choice"] == "required"

async def test_chat_agent_tool_choice_none_at_run_preserves_agent_level(chat_client_base: Any, tool_tool: Any) -> None:
    """Verify that tool_choice=None at run() uses agent-level default."""
    captured_options: list[ChatOptions] = []

    original_inner = chat_client_base._inner_get_response

    async def capturing_inner(
        *, messages: MutableSequence[Message], options: dict[str, Any], **kwargs: Any
    ) -> ChatResponse:
        captured_options.append(options)
        return await original_inner(messages=messages, options=options, **kwargs)

    chat_client_base._inner_get_response = capturing_inner

    # Create agent with agent-level tool_choice="auto" and a tool
    agent = Agent(
        client=chat_client_base,
        tools=[tool_tool],
        default_options={"tool_choice": "auto"},
    )

    # Run with explicitly passing None (same as not specifying)
    await agent.run("Hello", options={"tool_choice": None})

    # Verify the client received tool_choice="auto" from agent-level
    assert len(captured_options) >= 1
    assert captured_options[0]["tool_choice"] == "auto"

async def test_chat_agent_compaction_overrides_client_defaults(chat_client_base: Any) -> None:
    captured_roles: list[list[str]] = []
    captured_token_counts: list[list[int | None]] = []
    original_inner = chat_client_base._inner_get_response

    async def capturing_inner(
        *, messages: MutableSequence[Message], options: dict[str, Any], **kwargs: Any
    ) -> ChatResponse:
        captured_roles.append([message.role for message in messages])
        captured_token_counts.append([
            group.get(GROUP_TOKEN_COUNT_KEY) if isinstance(group, dict) else None
            for group in (message.additional_properties.get(GROUP_ANNOTATION_KEY) for message in messages)
        ])
        return await original_inner(messages=messages, options=options, **kwargs)

    chat_client_base._inner_get_response = capturing_inner
    chat_client_base.function_invocation_configuration["enabled"] = False
    chat_client_base.compaction_strategy = TruncationStrategy(max_n=1, compact_to=1)
    chat_client_base.tokenizer = _FixedTokenizer(5)

    agent = Agent(
        client=chat_client_base,
        compaction_strategy=SlidingWindowStrategy(keep_last_groups=2),
        tokenizer=_FixedTokenizer(9),
    )

    await agent.run([
        Message(role="user", contents=["Hello"]),
        Message(role="assistant", contents=["Previous response"]),
    ])

    assert captured_roles == [["user", "assistant"]]
    assert captured_token_counts == [[9, 9]]

async def test_chat_agent_uses_client_compaction_defaults_when_agent_unset(chat_client_base: Any) -> None:
    captured_roles: list[list[str]] = []
    original_inner = chat_client_base._inner_get_response

    async def capturing_inner(
        *, messages: MutableSequence[Message], options: dict[str, Any], **kwargs: Any
    ) -> ChatResponse:
        captured_roles.append([message.role for message in messages])
        return await original_inner(messages=messages, options=options, **kwargs)

    chat_client_base._inner_get_response = capturing_inner
    chat_client_base.function_invocation_configuration["enabled"] = False
    chat_client_base.compaction_strategy = TruncationStrategy(max_n=1, compact_to=1)

    agent = Agent(client=chat_client_base)

    await agent.run([
        Message(role="user", contents=["Hello"]),
        Message(role="assistant", contents=["Previous response"]),
    ])

    assert captured_roles == [["assistant"]]

async def test_chat_agent_run_level_compaction_and_tokenizer_override_agent_defaults(chat_client_base: Any) -> None:
    captured_roles: list[list[str]] = []
    captured_token_counts: list[list[int | None]] = []
    original_inner = chat_client_base._inner_get_response

    async def capturing_inner(
        *, messages: MutableSequence[Message], options: dict[str, Any], **kwargs: Any
    ) -> ChatResponse:
        captured_roles.append([message.role for message in messages])
        captured_token_counts.append([
            group.get(GROUP_TOKEN_COUNT_KEY) if isinstance(group, dict) else None
            for group in (message.additional_properties.get(GROUP_ANNOTATION_KEY) for message in messages)
        ])
        return await original_inner(messages=messages, options=options, **kwargs)

    chat_client_base._inner_get_response = capturing_inner
    chat_client_base.function_invocation_configuration["enabled"] = False

    agent = Agent(
        client=chat_client_base,
        compaction_strategy=SlidingWindowStrategy(keep_last_groups=2),
        tokenizer=_FixedTokenizer(9),
    )

    await agent.run(
        [
            Message(role="user", contents=["Hello"]),
            Message(role="assistant", contents=["Previous response"]),
        ],
        compaction_strategy=TruncationStrategy(max_n=1, compact_to=1),
        tokenizer=_FixedTokenizer(23),
    )

    assert captured_roles == [["assistant"]]
    assert captured_token_counts == [[23]]

def test_merge_options_runtime_model_overrides_default_model() -> None:
    """Test _merge_options lets a runtime model override a default model."""
    result = _merge_options({"model": "default-model"}, {"model": "runtime-model"})

    assert result["model"] == "runtime-model"

async def test_stores_by_default_skips_inmemory_injection(
    client: SupportsChatGetResponse,
) -> None:
    """Client with STORES_BY_DEFAULT=True should not auto-inject InMemoryHistoryProvider."""
    from agent_framework._sessions import InMemoryHistoryProvider

    # Simulate a client that stores by default
    client.STORES_BY_DEFAULT = True  # type: ignore[attr-defined]

    agent = Agent(client=client)
    session = agent.create_session()

    await agent.run("Hello", session=session)

    # No InMemoryHistoryProvider should have been injected
    assert not any(isinstance(p, InMemoryHistoryProvider) for p in agent.context_providers)

async def test_stores_by_default_false_injects_inmemory(
    client: SupportsChatGetResponse,
) -> None:
    """Client with STORES_BY_DEFAULT=False (default) should auto-inject InMemoryHistoryProvider."""
    from agent_framework._sessions import InMemoryHistoryProvider

    agent = Agent(client=client)
    session = agent.create_session()

    await agent.run("Hello", session=session)

    # InMemoryHistoryProvider should have been injected
    assert any(isinstance(p, InMemoryHistoryProvider) for p in agent.context_providers)

async def test_stores_by_default_with_store_false_injects_inmemory(
    client: SupportsChatGetResponse,
) -> None:
    """Client with STORES_BY_DEFAULT=True but store=False should still inject InMemoryHistoryProvider."""
    from agent_framework._sessions import InMemoryHistoryProvider

    client.STORES_BY_DEFAULT = True  # type: ignore[attr-defined]

    agent = Agent(client=client)
    session = agent.create_session()

    await agent.run("Hello", session=session, options={"store": False})

    # User explicitly disabled server storage, so InMemoryHistoryProvider should be injected
    assert any(isinstance(p, InMemoryHistoryProvider) for p in agent.context_providers)

async def test_store_true_skips_inmemory_injection(
    client: SupportsChatGetResponse,
) -> None:
    """Explicitly setting store=True should not auto-inject InMemoryHistoryProvider."""
    from agent_framework._sessions import InMemoryHistoryProvider

    agent = Agent(client=client)
    session = agent.create_session()

    await agent.run("Hello", session=session, options={"store": True})

    # User explicitly enabled server storage, so InMemoryHistoryProvider should not be injected
    assert not any(isinstance(p, InMemoryHistoryProvider) for p in agent.context_providers)

async def test_stores_by_default_with_store_false_in_default_options_injects_inmemory(
    client: SupportsChatGetResponse,
) -> None:
    """Client with STORES_BY_DEFAULT=True but store=False in default_options should inject InMemoryHistoryProvider.

    This covers the regression where store=False is set via Agent(..., default_options={"store": False})
    with no per-run override while the client has STORES_BY_DEFAULT=True.
    """
    from agent_framework._sessions import InMemoryHistoryProvider

    client.STORES_BY_DEFAULT = True  # type: ignore[attr-defined]

    # Set store=False at agent initialization via default_options, not at run-time
    agent = Agent(client=client, default_options={"store": False})
    session = agent.create_session()

    # Run without any per-run options override
    await agent.run("Hello", session=session)

    # User explicitly disabled server storage in default_options, so InMemoryHistoryProvider should be injected
    assert any(isinstance(p, InMemoryHistoryProvider) for p in agent.context_providers)

async def test_as_tool_raises_on_user_input_request(client: SupportsChatGetResponse) -> None:
    """Test that as_tool raises when the wrapped sub-agent requests user input."""
    from agent_framework.exceptions import UserInputRequiredException

    consent_content = Content.from_oauth_consent_request(
        consent_link="https://login.microsoftonline.com/consent",
    )
    client.streaming_responses = [  # type: ignore[attr-defined]
        [ChatResponseUpdate(contents=[consent_content], role="assistant")],
    ]

    agent = Agent(client=client, name="OAuthAgent", description="Agent requiring consent")
    agent_tool = agent.as_tool()

    with raises(UserInputRequiredException) as exc_info:
        await agent_tool.invoke(arguments={"task": "Do something"})

    assert len(exc_info.value.contents) == 1
    assert exc_info.value.contents[0].type == "oauth_consent_request"
    assert exc_info.value.contents[0].consent_link == "https://login.microsoftonline.com/consent"


# --- python/packages/devui/tests/devui/test_openai_sdk_integration.py ---

def test_openai_sdk_responses_create_streaming(devui_server: str) -> None:
    """Test using OpenAI SDK with streaming enabled."""
    base_url = devui_server
    client = OpenAI(base_url=f"{base_url}/v1", api_key="not-needed")

    # Get available entities - extract host and port from base_url
    parsed = urlparse(base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    try:
        conn.request("GET", "/v1/entities")
        response = conn.getresponse()
        entities = json.loads(response.read().decode("utf-8"))["entities"]
    finally:
        conn.close()

    assert len(entities) > 0, "No entities discovered"

    # Find an agent entity
    agent = next((e for e in entities if e["type"] == "agent"), None)
    if not agent:
        pytest.skip("No agent entities found")

    agent_id = agent["id"]

    # Test streaming request
    stream = client.responses.create(
        metadata={"entity_id": agent_id},
        input="Count to 3",
        stream=True,
    )

    events = []
    for event in stream:
        events.append(event)
        if len(events) >= 100:  # Limit for safety
            break

    assert len(events) > 0, "No events received from stream"

    # Check that we got various event types
    event_types = {event.type for event in events}
    # Should have at least response.completed or some content events
    assert len(event_types) > 0


# --- python/packages/foundry/tests/foundry/test_foundry_chat_client.py ---

async def test_integration_options(
    option_name: str,
    option_value: Any,
    needs_validation: bool,
) -> None:
    client = FoundryChatClient(credential=AzureCliCredential())
    client.function_invocation_configuration["max_iterations"] = 2

    if option_name.startswith("tools") or option_name.startswith("tool_choice"):
        messages = [Message(role="user", contents=["What is the weather in Seattle?"])]
    elif option_name.startswith("response_format"):
        messages = [Message(role="user", contents=["The weather in Seattle is sunny"])]
        messages.append(Message(role="user", contents=["What is the weather in Seattle?"]))
    else:
        messages = [Message(role="user", contents=["Say 'Hello World' briefly."])]

    options: dict[str, Any] = {option_name: option_value}
    if option_name.startswith("tool_choice"):
        options["tools"] = [get_weather]

    response = await client.get_response(messages=messages, options=options, stream=True).get_final_response()

    assert isinstance(response, ChatResponse)
    assert response.text is not None
    assert len(response.text) > 0

    if needs_validation:
        if option_name.startswith("tools") or option_name.startswith("tool_choice"):
            text = response.text.lower()
            assert "sunny" in text or "seattle" in text
        elif option_name.startswith("response_format"):
            if option_value == OutputStruct:
                assert response.value is not None
                assert isinstance(response.value, OutputStruct)
                assert "seattle" in response.value.location.lower()
            else:
                assert response.value is not None
                assert isinstance(response.value, dict)
                assert "location" in response.value

async def test_integration_web_search() -> None:
    client = FoundryChatClient(credential=AzureCliCredential())

    web_search_tool = FoundryChatClient.get_web_search_tool()
    content = {
        "messages": [
            Message(
                role="user",
                contents=["Who are the main characters of Kpop Demon Hunters? Do a web search to find the answer."],
            )
        ],
        "options": {"tool_choice": "auto", "tools": [web_search_tool]},
    }
    response = await client.get_response(stream=True, **content).get_final_response()

    assert isinstance(response, ChatResponse)
    assert "Rumi" in response.text
    assert "Mira" in response.text
    assert "Zoey" in response.text

async def test_integration_tool_rich_content_image() -> None:
    image_path = Path(__file__).parent.parent / "assets" / "sample_image.jpg"
    image_bytes = image_path.read_bytes()

    @tool(approval_mode="never_require")
    def get_test_image() -> Content:
        return Content.from_data(data=image_bytes, media_type="image/jpeg")

    client = FoundryChatClient(credential=AzureCliCredential())
    client.function_invocation_configuration["max_iterations"] = 2

    messages = [Message(role="user", contents=["Call the get_test_image tool and describe what you see."])]
    options: dict[str, Any] = {"tools": [get_test_image], "tool_choice": "auto"}

    response = await client.get_response(messages=messages, options=options, stream=True).get_final_response()

    assert isinstance(response, ChatResponse)
    assert response.text is not None
    assert len(response.text) > 0
    assert "house" in response.text.lower(), f"Model did not describe the house image. Response: {response.text}"


# --- python/packages/foundry/tests/test_foundry_evals.py ---

    def test_all_passed_true(self) -> None:
        r = EvalResults(
            provider="test",
            eval_id="e",
            run_id="r",
            status="completed",
            result_counts={"passed": 3, "failed": 0, "errored": 0},
        )
        assert r.all_passed
        assert r.passed == 3
        assert r.failed == 0
        assert r.total == 3

    def test_all_passed_false_on_failure(self) -> None:
        r = EvalResults(
            provider="test",
            eval_id="e",
            run_id="r",
            status="completed",
            result_counts={"passed": 2, "failed": 1, "errored": 0},
        )
        assert not r.all_passed
        assert r.failed == 1

    def test_all_passed_false_on_error(self) -> None:
        r = EvalResults(
            provider="test",
            eval_id="e",
            run_id="r",
            status="completed",
            result_counts={"passed": 2, "failed": 0, "errored": 1},
        )
        assert not r.all_passed

    def test_all_passed_false_on_non_completed(self) -> None:
        r = EvalResults(
            provider="test",
            eval_id="e",
            run_id="r",
            status="timeout",
            result_counts={"passed": 2, "failed": 0, "errored": 0},
        )
        assert not r.all_passed

    def test_all_passed_false_on_empty(self) -> None:
        r = EvalResults(
            provider="test",
            eval_id="e",
            run_id="r",
            status="completed",
            result_counts={"passed": 0, "failed": 0, "errored": 0},
        )
        assert not r.all_passed

    def test_raise_for_status_succeeds(self) -> None:
        r = EvalResults(
            provider="test",
            eval_id="e",
            run_id="r",
            status="completed",
            result_counts={"passed": 1, "failed": 0, "errored": 0},
        )
        r.raise_for_status()  # should not raise

    def test_raise_for_status_raises(self) -> None:
        r = EvalResults(
            provider="test",
            eval_id="e",
            run_id="r",
            status="completed",
            result_counts={"passed": 1, "failed": 1, "errored": 0},
        )
        with pytest.raises(EvalNotPassedError, match="1 passed, 1 failed"):
            r.raise_for_status()

    def test_raise_for_status_custom_message(self) -> None:
        r = EvalResults(provider="test", eval_id="e", run_id="r", status="failed")
        with pytest.raises(EvalNotPassedError, match="custom error"):
            r.raise_for_status("custom error")

    def test_none_result_counts(self) -> None:
        r = EvalResults(provider="test", eval_id="e", run_id="r", status="completed")
        assert r.passed == 0
        assert r.failed == 0
        assert r.total == 0
        assert not r.all_passed

    def test_sub_results_default_empty(self) -> None:
        r = EvalResults(
            provider="test",
            eval_id="e1",
            run_id="r1",
            status="completed",
            result_counts={"passed": 1, "failed": 0},
        )
        assert r.sub_results == {}
        assert r.all_passed

    def test_all_passed_checks_sub_results(self) -> None:
        parent = EvalResults(
            provider="test",
            eval_id="e1",
            run_id="r1",
            status="completed",
            result_counts={"passed": 2, "failed": 0},
            sub_results={
                "agent-a": EvalResults(
                    provider="test",
                    eval_id="e2",
                    run_id="r2",
                    status="completed",
                    result_counts={"passed": 1, "failed": 0},
                ),
                "agent-b": EvalResults(
                    provider="test",
                    eval_id="e3",
                    run_id="r3",
                    status="completed",
                    result_counts={"passed": 1, "failed": 1},
                ),
            },
        )
        assert not parent.all_passed  # agent-b has a failure

    def test_all_passed_with_all_sub_passing(self) -> None:
        parent = EvalResults(
            provider="test",
            eval_id="e1",
            run_id="r1",
            status="completed",
            result_counts={"passed": 2, "failed": 0},
            sub_results={
                "agent-a": EvalResults(
                    provider="test",
                    eval_id="e2",
                    run_id="r2",
                    status="completed",
                    result_counts={"passed": 1, "failed": 0},
                ),
            },
        )
        assert parent.all_passed

    def test_raise_for_status_includes_failed_agents(self) -> None:
        parent = EvalResults(
            provider="test",
            eval_id="e1",
            run_id="r1",
            status="completed",
            result_counts={"passed": 2, "failed": 0},
            sub_results={
                "good-agent": EvalResults(
                    provider="test",
                    eval_id="e2",
                    run_id="r2",
                    status="completed",
                    result_counts={"passed": 1, "failed": 0},
                ),
                "bad-agent": EvalResults(
                    provider="test",
                    eval_id="e3",
                    run_id="r3",
                    status="completed",
                    result_counts={"passed": 0, "failed": 1},
                ),
            },
        )
        with pytest.raises(EvalNotPassedError, match="bad-agent"):
            parent.raise_for_status()

    def test_extracts_single_agent(self) -> None:
        aer = _make_agent_exec_response("planner", "Plan is ready", ["Plan a trip"])

        events = [
            WorkflowEvent.executor_invoked("planner", "Plan a trip"),
            WorkflowEvent.executor_completed("planner", [aer]),
        ]
        result = WorkflowRunResult(events, [])

        data = _extract_agent_eval_data(result)
        assert len(data) == 1
        assert data[0]["executor_id"] == "planner"
        assert data[0]["response"].text == "Plan is ready"

    def test_extracts_multiple_agents(self) -> None:
        aer1 = _make_agent_exec_response("planner", "Plan done", ["Plan a trip"])
        aer2 = _make_agent_exec_response("booker", "Booked!", ["Book flight"])

        events = [
            WorkflowEvent.executor_invoked("planner", "Plan a trip"),
            WorkflowEvent.executor_completed("planner", [aer1]),
            WorkflowEvent.executor_invoked("booker", "Book flight"),
            WorkflowEvent.executor_completed("booker", [aer2]),
        ]
        result = WorkflowRunResult(events, [])

        data = _extract_agent_eval_data(result)
        assert len(data) == 2
        assert data[0]["executor_id"] == "planner"
        assert data[1]["executor_id"] == "booker"

    def test_skips_internal_executors(self) -> None:
        aer = _make_agent_exec_response("planner", "Done", ["Go"])

        events = [
            WorkflowEvent.executor_invoked("input-conversation", "hello"),
            WorkflowEvent.executor_completed("input-conversation", ["hello"]),
            WorkflowEvent.executor_invoked("planner", "Go"),
            WorkflowEvent.executor_completed("planner", [aer]),
            WorkflowEvent.executor_invoked("end", []),
            WorkflowEvent.executor_completed("end", None),
        ]
        result = WorkflowRunResult(events, [])

        data = _extract_agent_eval_data(result)
        assert len(data) == 1
        assert data[0]["executor_id"] == "planner"

    def test_extracts_string_query(self) -> None:
        events = [WorkflowEvent.executor_invoked("input", "Plan a trip")]
        result = WorkflowRunResult(events, [])
        assert _extract_overall_query(result) == "Plan a trip"

    def test_extracts_message_query(self) -> None:
        msgs = [Message("user", ["What's the weather?"])]
        events = [WorkflowEvent.executor_invoked("input", msgs)]
        result = WorkflowRunResult(events, [])
        assert "What's the weather?" in (_extract_overall_query(result) or "")

    def test_returns_none_for_empty(self) -> None:
        result = WorkflowRunResult([], [])
        assert _extract_overall_query(result) is None

    def test_with_token_usage(self) -> None:
        from agent_framework._evaluation import EvalItemResult

        item = EvalItemResult(
            item_id="1",
            status="pass",
            token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        assert item.token_usage is not None
        assert item.token_usage["total_tokens"] == 150

    def test_item_status_properties(self) -> None:
        from agent_framework._evaluation import EvalItemResult

        results = EvalResults(
            provider="test",
            eval_id="e1",
            run_id="r1",
            status="completed",
            result_counts={"passed": 2, "failed": 1, "errored": 1},
            items=[
                EvalItemResult(item_id="1", status="pass"),
                EvalItemResult(item_id="2", status="pass"),
                EvalItemResult(item_id="3", status="fail"),
                EvalItemResult(item_id="4", status="error", error_code="QueryExtractionError"),
            ],
        )
        assert sum(1 for i in results.items if i.is_passed) == 2
        assert sum(1 for i in results.items if i.is_failed) == 1
        assert sum(1 for i in results.items if i.is_error) == 1

    def test_raise_for_status_includes_errored_items(self) -> None:
        from agent_framework._evaluation import EvalItemResult

        results = EvalResults(
            provider="test",
            eval_id="e1",
            run_id="r1",
            status="completed",
            result_counts={"passed": 0, "failed": 0, "errored": 2},
            items=[
                EvalItemResult(item_id="i1", status="error", error_code="QueryExtractionError"),
                EvalItemResult(item_id="i2", status="error", error_code="TimeoutError"),
            ],
        )
        with pytest.raises(EvalNotPassedError, match="Errored items: i1: QueryExtractionError"):
            results.raise_for_status()


# --- python/packages/lab/lightning/tests/test_lightning.py ---

async def test_openai_workflow_two_agents(workflow_two_agents: Workflow):
    events = await workflow_two_agents.run("Please analyze the quarterly sales data")

    # Get all output events with AgentResponse
    agent_outputs = [event.data for event in events if event.type == "output" and isinstance(event.data, AgentResponse)]

    # Check that we have outputs from both agents
    assert len(agent_outputs) == 2
    assert any("Analyzed data shows trend upward" in str(output) for output in agent_outputs)
    assert any(
        "Based on the analysis 'Analyzed data shows trend upward', I recommend investing" in str(output)
        for output in agent_outputs
    )

async def test_observability(workflow_two_agents: Workflow):
    r"""Expected trace tree:

                    [workflow.run]
                    /      \
            [analyzer]      [advisor]
            /      \          /    \
    [DataAnalyzer] [send] [Investment] [send]
            |                    |
        [chat gpt-4o]        [chat gpt-4o]
    """
    pytest.importorskip("agentlightning")
    from agent_framework_lab_lightning import AgentFrameworkTracer
    from agentlightning.adapter import TracerTraceToTriplet

    tracer = AgentFrameworkTracer()
    try:
        tracer.init()
        tracer.init_worker(0)

        async with tracer.trace_context():
            await workflow_two_agents.run("Please analyze the quarterly sales data")

        triplets = TracerTraceToTriplet(agent_match=None, llm_call_match="chat").adapt(tracer.get_last_trace())
        assert len(triplets) == 2

        triplets = TracerTraceToTriplet(agent_match="analyzer", llm_call_match="chat").adapt(tracer.get_last_trace())
        assert len(triplets) == 1

        triplets = TracerTraceToTriplet(agent_match="advisor", llm_call_match="chat").adapt(tracer.get_last_trace())
        assert len(triplets) == 1

        # Parent agent is not matched
        triplets = TracerTraceToTriplet(agent_match="DataAnalyzer", llm_call_match="chat").adapt(
            tracer.get_last_trace()
        )
        assert len(triplets) == 0

        triplets = TracerTraceToTriplet(agent_match="InvestmentAdvisor|advisor", llm_call_match="chat").adapt(
            tracer.get_last_trace()
        )
        assert len(triplets) == 1

    finally:
        tracer.teardown_worker(0)
        tracer.teardown()


# --- python/packages/lab/tau2/tests/test_sliding_window.py ---

async def test_save_and_get_messages():
    """Test saving then getting messages with truncation."""
    provider = SlidingWindowHistoryProvider(max_tokens=50)
    state: dict = {}

    # Save many messages
    msgs = [
        Message(role="user", contents=[Content.from_text(text=f"Message {i} with some content")]) for i in range(10)
    ]
    await provider.save_messages(None, msgs, state=state)

    # get_messages returns truncated
    truncated = await provider.get_messages(None, state=state)
    # Full history is in session state
    all_msgs = state["messages"]

    assert len(all_msgs) == 10
    assert len(truncated) < len(all_msgs)

async def test_real_world_scenario():
    """Test a realistic conversation scenario."""
    provider = SlidingWindowHistoryProvider(max_tokens=30, system_message="You are a helpful assistant")
    state: dict = {}

    conversation = [
        Message(role="user", contents=[Content.from_text(text="Hello, how are you?")]),
        Message(
            role="assistant",
            contents=[Content.from_text(text="I'm doing well, thank you! How can I help you today?")],
        ),
        Message(role="user", contents=[Content.from_text(text="Can you tell me about the weather?")]),
        Message(
            role="assistant",
            contents=[
                Content.from_text(
                    text="I'd be happy to help with weather information, "
                    "but I don't have access to current weather data."
                )
            ],
        ),
        Message(role="user", contents=[Content.from_text(text="What about telling me a joke instead?")]),
        Message(
            role="assistant",
            contents=[
                Content.from_text(text="Sure! Why don't scientists trust atoms? Because they make up everything!")
            ],
        ),
    ]

    await provider.save_messages(None, conversation, state=state)

    truncated = await provider.get_messages(None, state=state)
    all_msgs = state["messages"]

    assert len(all_msgs) == 6
    assert len(truncated) <= 6

    token_count = provider._get_token_count(truncated)
    assert token_count <= provider.max_tokens * 1.1


# --- python/packages/ollama/tests/test_ollama_chat_client.py ---

def test_init(ollama_unit_test_env: dict[str, str]) -> None:
    # Test successful initialization
    ollama_chat_client = OllamaChatClient()

    assert ollama_chat_client.client is not None
    assert isinstance(ollama_chat_client.client, AsyncClient)
    assert ollama_chat_client.model == ollama_unit_test_env["OLLAMA_MODEL"]
    assert isinstance(ollama_chat_client, BaseChatClient)

def test_with_invalid_settings(ollama_unit_test_env: dict[str, str]) -> None:
    with pytest.raises(SettingNotFoundError, match="Required setting 'model'"):
        OllamaChatClient(
            host="http://localhost:12345",
            model=None,
        )

def test_serialize(ollama_unit_test_env: dict[str, str]) -> None:
    settings = {
        "host": ollama_unit_test_env["OLLAMA_HOST"],
        "model": ollama_unit_test_env["OLLAMA_MODEL"],
    }

    ollama_chat_client = OllamaChatClient.from_dict(settings)
    serialized = ollama_chat_client.to_dict()

    assert isinstance(serialized, dict)
    assert serialized["host"] == ollama_unit_test_env["OLLAMA_HOST"]
    assert serialized["model"] == ollama_unit_test_env["OLLAMA_MODEL"]

def test_chat_middleware(ollama_unit_test_env: dict[str, str]) -> None:
    @chat_middleware
    async def sample_middleware(context, call_next):
        await call_next()

    ollama_chat_client = OllamaChatClient(middleware=[sample_middleware])
    assert len(ollama_chat_client.middleware) == 1
    assert ollama_chat_client.middleware[0] == sample_middleware

def test_additional_properties(ollama_unit_test_env: dict[str, str]) -> None:
    additional_properties = {
        "user_location": {
            "country": "US",
            "city": "Seattle",
        }
    }
    ollama_chat_client = OllamaChatClient(
        additional_properties=additional_properties,
    )
    assert ollama_chat_client.additional_properties == additional_properties

async def test_empty_messages() -> None:
    ollama_chat_client = OllamaChatClient(
        host="http://localhost:12345",
        model="test-model",
    )
    with pytest.raises(ChatClientInvalidRequestException):
        await ollama_chat_client.get_response(messages=[])

async def test_cmc_integration_with_tool_call(
    chat_history: list[Message],
) -> None:
    chat_history.append(Message(contents=["Call the hello world function and repeat what it says"], role="user"))

    ollama_client = OllamaChatClient()
    result = await ollama_client.get_response(messages=chat_history, options={"tools": [hello_world]})

    assert "hello" in result.text.lower() and "world" in result.text.lower()
    assert result.messages[-2].contents[0].type == "function_result"
    tool_result = result.messages[-2].contents[0]
    assert tool_result.result == "Hello World"

async def test_cmc_integration_with_chat_completion(
    chat_history: list[Message],
) -> None:
    chat_history.append(Message(contents=["Say Hello World"], role="user"))

    ollama_client = OllamaChatClient()
    result = await ollama_client.get_response(messages=chat_history)

    assert "hello" in result.text.lower()

async def test_cmc_streaming_integration_with_tool_call(
    chat_history: list[Message],
) -> None:
    chat_history.append(Message(contents=["Call the hello world function and repeat what it says"], role="user"))

    ollama_client = OllamaChatClient()
    result: AsyncIterable[ChatResponseUpdate] = ollama_client.get_response(
        messages=chat_history, stream=True, options={"tools": [hello_world]}
    )

    chunks: list[ChatResponseUpdate] = []
    async for chunk in result:
        chunks.append(chunk)

    for c in chunks:
        if len(c.contents) > 0:
            if c.contents[0].type == "function_result":
                tool_result = c.contents[0]
                assert tool_result.result == "Hello World"
            if c.contents[0].type == "function_call":
                tool_call = c.contents[0]
                assert tool_call.name == "hello_world"

async def test_cmc_streaming_integration_with_chat_completion(
    chat_history: list[Message],
) -> None:
    chat_history.append(Message(contents=["Say Hello World"], role="user"))

    ollama_client = OllamaChatClient()
    result: AsyncIterable[ChatResponseUpdate] = ollama_client.get_response(messages=chat_history, stream=True)

    full_text = ""
    async for chunk in result:
        full_text += chunk.text

    assert "hello" in full_text.lower() and "world" in full_text.lower()


# --- python/packages/openai/tests/openai/test_openai_chat_client.py ---

async def test_get_response_with_all_parameters() -> None:
    """Test request preparation with a comprehensive parameter set."""
    client = OpenAIChatClient(model="test-model", api_key="test-key")
    _, run_options, _ = await client._prepare_request(
        messages=[Message(role="user", contents=["Test message"])],
        options={
            "include": ["message.output_text.logprobs"],
            "instructions": "You are a helpful assistant",
            "max_tokens": 100,
            "parallel_tool_calls": True,
            "model": "gpt-4",
            "previous_response_id": "prev-123",
            "reasoning": {"chain_of_thought": "enabled"},
            "service_tier": "auto",
            "response_format": OutputStruct,
            "seed": 42,
            "store": True,
            "temperature": 0.7,
            "tool_choice": "auto",
            "tools": [get_weather],
            "top_p": 0.9,
            "user": "test-user",
            "truncation": "auto",
            "timeout": 30.0,
            "additional_properties": {"custom": "value"},
        },
    )

    assert run_options["include"] == ["message.output_text.logprobs"]
    assert run_options["max_output_tokens"] == 100
    assert run_options["parallel_tool_calls"] is True
    assert run_options["model"] == "gpt-4"
    assert run_options["previous_response_id"] == "prev-123"
    assert run_options["reasoning"] == {"chain_of_thought": "enabled"}
    assert run_options["service_tier"] == "auto"
    assert run_options["text_format"] is OutputStruct
    assert run_options["store"] is True
    assert run_options["temperature"] == 0.7
    assert run_options["tool_choice"] == "auto"
    assert run_options["top_p"] == 0.9
    assert run_options["user"] == "test-user"
    assert run_options["truncation"] == "auto"
    assert run_options["timeout"] == 30.0
    assert run_options["additional_properties"] == {"custom": "value"}
    assert len(run_options["tools"]) == 1
    assert run_options["tools"][0]["type"] == "function"
    assert run_options["tools"][0]["name"] == "get_weather"
    assert run_options["input"][0]["role"] == "system"
    assert run_options["input"][0]["content"][0]["text"] == "You are a helpful assistant"
    assert run_options["input"][1]["role"] == "user"
    assert run_options["input"][1]["content"][0]["text"] == "Test message"

async def test_code_interpreter_tool_variations() -> None:
    """Test HostedCodeInterpreterTool with and without file inputs."""
    client = OpenAIChatClient(model="test-model", api_key="test-key")

    # Test code interpreter using static method
    code_tool = OpenAIChatClient.get_code_interpreter_tool()

    _, run_options, _ = await client._prepare_request(
        messages=[Message("user", ["Run some code"])],
        options={"tools": [code_tool]},
    )

    assert run_options["tools"] == [code_tool]

    # Test code interpreter with files using static method
    code_tool_with_files = OpenAIChatClient.get_code_interpreter_tool(file_ids=["file1", "file2"])

    _, run_options, _ = await client._prepare_request(
        messages=[Message(role="user", contents=["Process these files"])],
        options={"tools": [code_tool_with_files]},
    )

    assert run_options["tools"] == [code_tool_with_files]

def test_get_shell_tool_reuses_function_tool_instance() -> None:
    """Passing a FunctionTool should update and return the same tool instance."""

    @tool(name="run_shell", approval_mode="never_require")
    def run_shell(command: str) -> str:
        return command

    shell_tool = OpenAIChatClient.get_shell_tool(
        func=run_shell,
        description="Run local shell command",
        approval_mode="always_require",
    )

    assert shell_tool is run_shell
    assert shell_tool.kind == "shell"
    assert shell_tool.description == "Run local shell command"
    assert shell_tool.approval_mode == "always_require"
    assert (shell_tool.additional_properties or {}).get("openai.responses.shell.environment") == {"type": "local"}

def test_prepare_message_for_openai_includes_reasoning_with_function_call() -> None:
    """Test _prepare_message_for_openai includes reasoning items alongside function_calls.

    Reasoning models require reasoning items to be present in the input when
    function_call items are included. Stripping reasoning causes a 400 error:
    "function_call was provided without its required reasoning item".
    """
    client = OpenAIChatClient(model="test-model", api_key="test-key")

    reasoning = Content.from_text_reasoning(
        id="rs_abc123",
        text="Let me analyze the request",
        additional_properties={"status": "completed"},
    )
    function_call = Content.from_function_call(
        call_id="call_123",
        name="search_hotels",
        arguments='{"city": "Paris"}',
    )

    message = Message(role="assistant", contents=[reasoning, function_call])

    result = client._prepare_message_for_openai(message)

    # Both reasoning and function_call should be present as top-level items
    types = [item["type"] for item in result]
    assert "reasoning" in types, "Reasoning items must be included for reasoning models"
    assert "function_call" in types

    reasoning_item = next(item for item in result if item["type"] == "reasoning")
    assert reasoning_item["summary"][0]["text"] == "Let me analyze the request"
    assert reasoning_item["id"] == "rs_abc123", "Reasoning id must be preserved for the API"

def test_prepare_messages_for_openai_full_conversation_with_reasoning() -> None:
    """Test _prepare_messages_for_openai correctly serializes a full conversation
    that includes reasoning + function_call + function_result + final text.

    This simulates the conversation history passed between agents in a workflow.
    The API requires reasoning items alongside function_calls.
    """
    client = OpenAIChatClient(model="test-model", api_key="test-key")

    messages = [
        Message(role="user", contents=[Content.from_text(text="search for hotels")]),
        Message(
            role="assistant",
            contents=[
                Content.from_text_reasoning(
                    id="rs_test123",
                    text="I need to search for hotels",
                    additional_properties={"status": "completed"},
                ),
                Content.from_function_call(
                    call_id="call_1",
                    name="search_hotels",
                    arguments='{"city": "Paris"}',
                    additional_properties={"fc_id": "fc_test456"},
                ),
            ],
        ),
        Message(
            role="tool",
            contents=[
                Content.from_function_result(
                    call_id="call_1",
                    result="Found 3 hotels in Paris",
                ),
            ],
        ),
        Message(
            role="assistant",
            contents=[Content.from_text(text="I found hotels for you")],
        ),
    ]

    result = client._prepare_messages_for_openai(messages)

    types = [item.get("type") for item in result]
    assert "message" in types, "User/assistant messages should be present"
    assert "reasoning" in types, "Reasoning items must be present"
    assert "function_call" in types, "Function call items must be present"
    assert "function_call_output" in types, "Function call output must be present"

    # Verify reasoning has id
    reasoning_items = [item for item in result if item.get("type") == "reasoning"]
    assert reasoning_items[0]["id"] == "rs_test123"

    # Verify function_call has id
    fc_items = [item for item in result if item.get("type") == "function_call"]
    assert fc_items[0]["id"] == "fc_test456"

    # Verify correct ordering: reasoning before function_call
    reasoning_idx = types.index("reasoning")
    fc_idx = types.index("function_call")
    assert reasoning_idx < fc_idx, "Reasoning must come before function_call"

async def test_get_response_streaming_with_response_format() -> None:
    """Test get_response streaming with response_format."""
    client = OpenAIChatClient(model="test-model", api_key="test-key")
    messages = [Message(role="user", contents=["Test streaming with format"])]

    # It will fail due to invalid API key, but exercises the code path
    with pytest.raises(ChatClientException):

        async def run_streaming():
            async for _ in client.get_response(
                stream=True,
                messages=messages,
                options={"response_format": OutputStruct},
            ):
                pass

        await run_streaming()

async def test_integration_options(
    option_name: str,
    option_value: Any,
    needs_validation: bool,
) -> None:
    """Parametrized test covering all ChatOptions and OpenAIChatOptions.

    Tests both streaming and non-streaming modes for each option to ensure
    they don't cause failures. Options marked with needs_validation also
    check that the feature actually works correctly.
    """
    client = OpenAIChatClient()
    # Need at least 2 iterations for tool_choice tests: one to get function call, one to get final response
    client.function_invocation_configuration["max_iterations"] = 2

    # Prepare test message
    if option_name.startswith("tools") or option_name.startswith("tool_choice"):
        # Use weather-related prompt for tool tests
        messages = [Message(role="user", contents=["What is the weather in Seattle?"])]
    elif option_name.startswith("response_format"):
        # Use prompt that works well with structured output
        messages = [Message(role="user", contents=["The weather in Seattle is sunny"])]
        messages.append(Message(role="user", contents=["What is the weather in Seattle?"]))
    else:
        # Generic prompt for simple options
        messages = [Message(role="user", contents=["Say 'Hello World' briefly."])]

    # Build options dict
    options: dict[str, Any] = {option_name: option_value}

    # Add tools if testing tool_choice to avoid errors
    if option_name.startswith("tool_choice"):
        options["tools"] = [get_weather]

    # Test streaming mode
    response = await client.get_response(stream=True, messages=messages, options=options).get_final_response()

    assert response is not None
    assert isinstance(response, ChatResponse)
    assert response.text is not None, f"No text in response for option '{option_name}'"
    assert len(response.text) > 0, f"Empty response for option '{option_name}'"

    # Validate based on option type
    if needs_validation:
        if option_name.startswith("tools") or option_name.startswith("tool_choice"):
            # Should have called the weather function
            text = response.text.lower()
            assert "sunny" in text or "seattle" in text, f"Tool not invoked for {option_name}"
        elif option_name.startswith("response_format"):
            if option_value == OutputStruct:
                # Should have structured output
                assert response.value is not None, "No structured output"
                assert isinstance(response.value, OutputStruct)
                assert "seattle" in response.value.location.lower()
            else:
                assert response.value is not None
                assert isinstance(response.value, dict)
                assert "location" in response.value
                assert "seattle" in response.value["location"].lower()

async def test_integration_web_search() -> None:
    client = OpenAIChatClient(model="gpt-5")

    # Test that the client will use the web search tool with location
    web_search_tool_with_location = OpenAIChatClient.get_web_search_tool(
        user_location={"country": "US", "city": "Seattle"},
    )
    content = {
        "messages": [
            Message(
                role="user",
                contents=["What is the current weather? Do not ask for my current location."],
            )
        ],
        "options": {
            "tool_choice": "auto",
            "tools": [web_search_tool_with_location],
        },
    }
    response = await client.get_response(stream=True, **content).get_final_response()
    assert response.text is not None

async def test_integration_streaming_file_search() -> None:
    openai_responses_client = OpenAIChatClient()

    assert isinstance(openai_responses_client, SupportsChatGetResponse)

    file_id, vector_store = await create_vector_store(openai_responses_client)
    # Use static method for file search tool
    file_search_tool = OpenAIChatClient.get_file_search_tool(vector_store_ids=[vector_store.vector_store_id])
    # Test that the client will use the web search tool
    response = openai_responses_client.get_streaming_response(
        messages=[
            Message(
                role="user",
                contents=["What is the weather today? Do a file search to find the answer."],
            )
        ],
        options={
            "tool_choice": "auto",
            "tools": [file_search_tool],
        },
    )

    assert response is not None
    full_message: str = ""
    async for chunk in response:
        assert chunk is not None
        assert isinstance(chunk, ChatResponseUpdate)
        for content in chunk.contents:
            if content.type == "text" and content.text:
                full_message += content.text

    await delete_vector_store(openai_responses_client, file_id, vector_store.vector_store_id)

    assert "sunny" in full_message.lower()
    assert "75" in full_message

async def test_integration_tool_rich_content_image() -> None:
    """Integration test: a tool returns an image and the model describes it."""
    image_path = Path(__file__).parent.parent / "assets" / "sample_image.jpg"
    image_bytes = image_path.read_bytes()

    @tool(approval_mode="never_require")
    def get_test_image() -> Content:
        """Return a test image for analysis."""
        return Content.from_data(data=image_bytes, media_type="image/jpeg")

    client = OpenAIChatClient()
    client.function_invocation_configuration["max_iterations"] = 2

    for streaming in [False, True]:
        messages = [
            Message(
                role="user",
                contents=["Call the get_test_image tool and describe what you see."],
            )
        ]
        options: dict[str, Any] = {"tools": [get_test_image], "tool_choice": "auto"}

        if streaming:
            response = await client.get_response(messages=messages, stream=True, options=options).get_final_response()
        else:
            response = await client.get_response(messages=messages, options=options)

        assert response is not None
        assert isinstance(response, ChatResponse)
        assert response.text is not None
        assert len(response.text) > 0
        # sample_image.jpg contains a photo of a house; the model should mention it.
        assert "house" in response.text.lower(), f"Model did not describe the house image. Response: {response.text}"

async def test_integration_agent_replays_local_tool_history_without_stale_fc_id() -> None:
    """Integration test: persisted local Responses tool history can be replayed on a later turn."""
    hotel_code = "HOTEL-PERSIST-4672"

    @tool(name="search_hotels", approval_mode="never_require")
    async def search_hotels(city: Annotated[str, "The city to search for hotels in"]) -> str:
        return f"The only hotel option in {city} is {hotel_code}."

    # override with model that does not do reasoning by default
    client = OpenAIChatClient(model="gpt-5.4")
    client.function_invocation_configuration["max_iterations"] = 2

    agent = Agent(client=client, tools=[search_hotels], default_options={"store": False})
    session = agent.create_session()

    first_response = await agent.run(
        "Call the search_hotels tool for Paris and answer with the hotel code you found.",
        session=session,
        options={"tool_choice": {"mode": "required", "required_function_name": "search_hotels"}},
    )
    assert first_response.text is not None
    assert hotel_code in first_response.text

    shared_messages = session.state[InMemoryHistoryProvider.DEFAULT_SOURCE_ID]["messages"]
    shared_function_call = next(
        content for message in shared_messages for content in message.contents if content.type == "function_call"
    )
    assert shared_function_call.additional_properties is not None
    assert isinstance(shared_function_call.additional_properties.get("fc_id"), str)
    assert shared_function_call.additional_properties["fc_id"]

    second_response = await agent.run(
        "What hotel code did you already find for Paris? Answer with the exact code only.",
        session=session,
        options={"tool_choice": "none"},
    )
    assert second_response.text is not None
    assert hotel_code in second_response.text

def test_agent_response_update_with_continuation_token() -> None:
    """Test that AgentResponseUpdate accepts and stores continuation_token."""
    from agent_framework import AgentResponseUpdate

    from agent_framework_openai import OpenAIContinuationToken

    token = OpenAIContinuationToken(response_id="resp_012")
    update = AgentResponseUpdate(
        contents=[Content.from_text(text="streaming")],
        role="assistant",
        continuation_token=token,
    )
    assert update.continuation_token is not None
    assert update.continuation_token["response_id"] == "resp_012"

async def test_prepare_messages_for_openai_does_not_replay_fc_id_when_loaded_from_history() -> None:
    """Loaded history must not replay provider-ephemeral Responses function call IDs."""
    client = OpenAIChatClient(model="test-model", api_key="test-key")
    provider = InMemoryHistoryProvider()

    session = AgentSession(session_id="thread-1")
    session.state[provider.source_id] = {
        "messages": [
            Message(
                role="assistant",
                contents=[
                    Content.from_function_call(
                        call_id="call_1",
                        name="search_hotels",
                        arguments='{"city": "Paris"}',
                        additional_properties={"fc_id": "fc_provider123", "status": "completed"},
                    ),
                ],
            ),
            Message(
                role="tool",
                contents=[
                    Content.from_function_result(
                        call_id="call_1",
                        result="Found 3 hotels in Paris",
                    ),
                ],
            ),
        ]
    }

    next_turn_input = Message(role="user", contents=[Content.from_text(text="Book the cheapest one")])

    live_result = client._prepare_messages_for_openai([*session.state[provider.source_id]["messages"], next_turn_input])
    live_function_call = next(item for item in live_result if item.get("type") == "function_call")
    assert live_function_call["id"] == "fc_provider123"

    context = SessionContext(session_id=session.session_id, input_messages=[next_turn_input])
    await provider.before_run(
        agent=None,
        session=session,
        context=context,
        state=session.state.setdefault(provider.source_id, {}),
    )  # type: ignore[arg-type]

    loaded_result = client._prepare_messages_for_openai(
        context.get_messages(sources={provider.source_id}, include_input=True)
    )
    loaded_function_call = next(item for item in loaded_result if item.get("type") == "function_call")
    assert loaded_function_call["id"] == "fc_call_1"

    stored_function_call = session.state[provider.source_id]["messages"][0].contents[0]
    assert stored_function_call.additional_properties is not None
    assert stored_function_call.additional_properties.get("fc_id") == "fc_provider123"

    restored = AgentSession.from_dict(json.loads(json.dumps(session.to_dict())))
    restored_context = SessionContext(session_id=restored.session_id, input_messages=[next_turn_input])
    await provider.before_run(
        agent=None,
        session=restored,
        context=restored_context,
        state=restored.state.setdefault(provider.source_id, {}),
    )  # type: ignore[arg-type]

    restored_result = client._prepare_messages_for_openai(
        restored_context.get_messages(sources={provider.source_id}, include_input=True)
    )
    restored_function_call = next(item for item in restored_result if item.get("type") == "function_call")
    assert restored_function_call["id"] == "fc_call_1"


# --- python/packages/openai/tests/openai/test_openai_chat_client_azure.py ---

async def test_integration_options(
    option_name: str,
    option_value: Any,
    needs_validation: bool,
) -> None:
    async with AzureCliCredential() as credential:
        client = OpenAIChatClient(credential=credential)
        client.function_invocation_configuration["max_iterations"] = 2

        for streaming in [False, True]:
            if option_name in {"tools", "tool_choice"}:
                messages = [Message(role="user", contents=["What is the weather in Seattle?"])]
            elif option_name == "response_format":
                messages = [
                    Message(role="user", contents=["The weather in Seattle is sunny"]),
                    Message(role="user", contents=["What is the weather in Seattle?"]),
                ]
            else:
                messages = [Message(role="user", contents=["Say 'Hello World' briefly."])]

            options: dict[str, Any] = {option_name: option_value}
            if option_name == "tool_choice":
                options["tools"] = [get_weather]

            if streaming:
                response = await client.get_response(
                    messages=messages,
                    stream=True,
                    options=options,
                ).get_final_response()
            else:
                response = await client.get_response(messages=messages, options=options)

            assert isinstance(response, ChatResponse)
            assert response.text is not None
            assert len(response.text) > 0

            if needs_validation:
                if option_name in {"tools", "tool_choice"}:
                    text = response.text.lower()
                    assert "sunny" in text or "seattle" in text
                elif option_name == "response_format":
                    if option_value == OutputStruct:
                        assert response.value is not None
                        assert isinstance(response.value, OutputStruct)
                        assert "seattle" in response.value.location.lower()
                    else:
                        assert response.value is not None
                        assert isinstance(response.value, dict)
                        assert "location" in response.value
                        assert "seattle" in response.value["location"].lower()

async def test_integration_web_search() -> None:
    async with AzureCliCredential() as credential:
        client = OpenAIChatClient(credential=credential)

        response = await client.get_response(
            messages=[
                Message(
                    role="user",
                    contents=["What is the current weather? Do not ask for my current location."],
                )
            ],
            options={
                "tools": [OpenAIChatClient.get_web_search_tool(user_location={"country": "US", "city": "Seattle"})],
            },
            stream=True,
        ).get_final_response()
        assert isinstance(response, ChatResponse)
        assert response.text is not None

async def test_integration_client_file_search_streaming() -> None:
    async with AzureCliCredential() as credential:
        client = OpenAIChatClient(credential=credential)
        file_id, vector_store = await create_vector_store(client)
        try:
            response_stream = client.get_response(
                messages=[
                    Message(role="user", contents=["What is the weather today? Do a file search to find the answer."])
                ],
                stream=True,
                options={
                    "tools": [OpenAIChatClient.get_file_search_tool(vector_store_ids=[vector_store.vector_store_id])],
                    "tool_choice": "auto",
                },
            )

            full_response = await response_stream.get_final_response()
            assert "sunny" in full_response.text.lower()
            assert "75" in full_response.text
        finally:
            await delete_vector_store(client, file_id, vector_store.vector_store_id)

async def test_integration_client_agent_existing_session() -> None:
    async with AzureCliCredential() as credential:
        preserved_session = None

        async with Agent(
            client=OpenAIChatClient(credential=credential),
            instructions="You are a helpful assistant with good memory.",
        ) as first_agent:
            session = first_agent.create_session()
            first_response = await first_agent.run(
                "My hobby is photography. Remember this.",
                session=session,
                options={"store": True},
            )

            assert isinstance(first_response, AgentResponse)
            preserved_session = session

        if preserved_session:
            async with Agent(
                client=OpenAIChatClient(credential=credential),
                instructions="You are a helpful assistant with good memory.",
            ) as second_agent:
                second_response = await second_agent.run(
                    "What is my hobby?", session=preserved_session, options={"store": True}
                )

                assert isinstance(second_response, AgentResponse)
                assert second_response.text is not None
                assert "photography" in second_response.text.lower()

async def test_azure_openai_chat_client_tool_rich_content_image() -> None:
    image_path = Path(__file__).parent.parent / "assets" / "sample_image.jpg"
    image_bytes = image_path.read_bytes()

    @tool(approval_mode="never_require")
    def get_test_image() -> Content:
        """Return a test image for analysis."""
        return Content.from_data(data=image_bytes, media_type="image/jpeg")

    async with AzureCliCredential() as credential:
        client = OpenAIChatClient(credential=credential)
        client.function_invocation_configuration["max_iterations"] = 2

        for streaming in [False, True]:
            messages = [Message(role="user", contents=["Call the get_test_image tool and describe what you see."])]
            options: dict[str, Any] = {"tools": [get_test_image], "tool_choice": "auto"}

            if streaming:
                response = await client.get_response(
                    messages=messages,
                    stream=True,
                    options=options,
                ).get_final_response()
            else:
                response = await client.get_response(messages=messages, options=options)

            assert isinstance(response, ChatResponse)
            assert response.text is not None
            assert "house" in response.text.lower(), (
                f"Model did not describe the house image. Response: {response.text}"
            )


# --- python/packages/openai/tests/openai/test_openai_chat_completion_client.py ---

def test_init(openai_unit_test_env: dict[str, str]) -> None:
    # Test successful initialization
    open_ai_chat_completion = OpenAIChatCompletionClient()

    assert open_ai_chat_completion.model == openai_unit_test_env["OPENAI_MODEL"]
    assert isinstance(open_ai_chat_completion, SupportsChatGetResponse)

def test_get_response_docstring_surfaces_layered_runtime_docs() -> None:
    docstring = inspect.getdoc(OpenAIChatCompletionClient.get_response)

    assert docstring is not None
    assert "Get a response from a chat client." in docstring
    assert "function_invocation_kwargs" in docstring
    assert "middleware: Optional per-call chat and function middleware." in docstring
    assert "function_middleware: Optional per-call function middleware." not in docstring

def test_get_response_is_defined_on_openai_class() -> None:
    signature = inspect.signature(OpenAIChatCompletionClient.get_response)

    assert OpenAIChatCompletionClient.get_response.__qualname__ == "OpenAIChatCompletionClient.get_response"
    assert "middleware" in signature.parameters
    assert all(parameter.kind != inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())

def test_init_uses_explicit_parameters() -> None:
    signature = inspect.signature(RawOpenAIChatCompletionClient.__init__)

    assert "additional_properties" in signature.parameters
    assert "compaction_strategy" in signature.parameters
    assert "tokenizer" in signature.parameters
    assert all(parameter.kind != inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())

def test_supports_web_search_only() -> None:
    assert not isinstance(OpenAIChatCompletionClient, SupportsCodeInterpreterTool)
    assert isinstance(OpenAIChatCompletionClient, SupportsWebSearchTool)
    assert not isinstance(OpenAIChatCompletionClient, SupportsImageGenerationTool)
    assert not isinstance(OpenAIChatCompletionClient, SupportsMCPTool)
    assert not isinstance(OpenAIChatCompletionClient, SupportsFileSearchTool)

def test_init_validation_fail() -> None:
    # Test successful initialization
    with pytest.raises(ValueError):
        OpenAIChatCompletionClient(api_key="34523", model={"test": "dict"})  # type: ignore

def test_init_model_constructor(openai_unit_test_env: dict[str, str]) -> None:
    # Test successful initialization
    model = "test_model"
    open_ai_chat_completion = OpenAIChatCompletionClient(model=model)

    assert open_ai_chat_completion.model == model
    assert isinstance(open_ai_chat_completion, SupportsChatGetResponse)

def test_init_with_default_header(openai_unit_test_env: dict[str, str]) -> None:
    default_headers = {"X-Unit-Test": "test-guid"}

    # Test successful initialization
    open_ai_chat_completion = OpenAIChatCompletionClient(
        default_headers=default_headers,
    )

    assert open_ai_chat_completion.model == openai_unit_test_env["OPENAI_MODEL"]
    assert isinstance(open_ai_chat_completion, SupportsChatGetResponse)

    # Assert that the default header we added is present in the client's default headers
    for key, value in default_headers.items():
        assert key in open_ai_chat_completion.client.default_headers
        assert open_ai_chat_completion.client.default_headers[key] == value

def test_init_base_url(openai_unit_test_env: dict[str, str]) -> None:
    # Test successful initialization
    open_ai_chat_completion = OpenAIChatCompletionClient(base_url="http://localhost:1234/v1")
    assert str(open_ai_chat_completion.client.base_url) == "http://localhost:1234/v1/"

def test_init_with_empty_model(openai_unit_test_env: dict[str, str]) -> None:
    with pytest.raises(SettingNotFoundError):
        OpenAIChatCompletionClient()

def test_init_with_empty_api_key(openai_unit_test_env: dict[str, str]) -> None:
    model = "test_model"

    with pytest.raises(SettingNotFoundError):
        OpenAIChatCompletionClient(
            model=model,
        )

def test_serialize(openai_unit_test_env: dict[str, str]) -> None:
    default_headers = {"X-Unit-Test": "test-guid"}

    settings = {
        "model": openai_unit_test_env["OPENAI_MODEL"],
        "api_key": openai_unit_test_env["OPENAI_API_KEY"],
        "default_headers": default_headers,
    }

    open_ai_chat_completion = OpenAIChatCompletionClient.from_dict(settings)
    dumped_settings = open_ai_chat_completion.to_dict()
    assert dumped_settings["model"] == openai_unit_test_env["OPENAI_MODEL"]
    # Assert that the default header we added is present in the dumped_settings default headers
    for key, value in default_headers.items():
        assert key in dumped_settings["default_headers"]
        assert dumped_settings["default_headers"][key] == value
    # Assert that the 'User-Agent' header is not present in the dumped_settings default headers
    assert "User-Agent" not in dumped_settings["default_headers"]

def test_serialize_with_org_id(openai_unit_test_env: dict[str, str]) -> None:
    settings = {
        "model": openai_unit_test_env["OPENAI_MODEL"],
        "api_key": openai_unit_test_env["OPENAI_API_KEY"],
        "org_id": openai_unit_test_env["OPENAI_ORG_ID"],
    }

    open_ai_chat_completion = OpenAIChatCompletionClient.from_dict(settings)
    dumped_settings = open_ai_chat_completion.to_dict()
    assert dumped_settings["model"] == openai_unit_test_env["OPENAI_MODEL"]
    assert dumped_settings["org_id"] == openai_unit_test_env["OPENAI_ORG_ID"]
    # Assert that the 'User-Agent' header is not present in the dumped_settings default headers
    assert "User-Agent" not in dumped_settings.get("default_headers", {})

def test_unsupported_tool_handling(openai_unit_test_env: dict[str, str]) -> None:
    """Test that unsupported tool types are passed through unchanged."""
    client = OpenAIChatCompletionClient()

    # Create a random object that's not a FunctionTool, dict, or callable
    # This simulates an unsupported tool type that gets passed through
    class UnsupportedTool:
        pass

    unsupported_tool = UnsupportedTool()

    # Unsupported tools are passed through for the API to handle/reject
    result = client._prepare_tools_for_openai([unsupported_tool])  # type: ignore
    assert "tools" in result
    assert len(result["tools"]) == 1

    # Also test with a dict-based tool that should be passed through
    dict_tool = {"type": "function", "name": "test"}
    result = client._prepare_tools_for_openai([dict_tool])  # type: ignore
    assert result["tools"] == [dict_tool]

def test_mcp_tool_dict_passed_through_to_chat_api(openai_unit_test_env: dict[str, str]) -> None:
    """Test that MCP tool dicts are passed through unchanged by the chat client.

    The Chat Completions API does not support "type": "mcp" tools. MCP tools
    should be used with the Responses API client instead. This test documents
    that the chat client passes dict-based tools through without filtering,
    so callers must use the correct client for MCP tools.
    """
    client = OpenAIChatCompletionClient()

    mcp_tool = {
        "type": "mcp",
        "server_label": "Microsoft_Learn_MCP",
        "server_url": "https://learn.microsoft.com/api/mcp",
    }

    result = client._prepare_tools_for_openai(mcp_tool)
    assert "tools" in result
    assert len(result["tools"]) == 1
    # The chat client passes dict tools through unchanged, including unsupported types
    assert result["tools"][0]["type"] == "mcp"

def test_prepare_tools_with_single_function_tool(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that a single FunctionTool is accepted for tool preparation."""
    client = OpenAIChatCompletionClient()

    @tool(approval_mode="never_require")
    def test_function(query: str) -> str:
        """A test function."""
        return f"Result for {query}"

    result = client._prepare_tools_for_openai(test_function)
    assert "tools" in result
    assert len(result["tools"]) == 1
    assert result["tools"][0]["type"] == "function"

def test_function_result_falsy_values_handling(openai_unit_test_env: dict[str, str]):
    """Test that falsy values (like empty list) in function result are properly handled.

    Note: In practice, FunctionTool.invoke() always returns a pre-parsed string.
    These tests verify that the OpenAI client correctly passes through string results.
    """
    client = OpenAIChatCompletionClient()

    # Test with empty list serialized as JSON string (pre-serialized result passed to from_function_result)
    message_with_empty_list = Message(
        role="tool",
        contents=[Content.from_function_result(call_id="call-123", result="[]")],
    )

    openai_messages = client._prepare_message_for_openai(message_with_empty_list)
    assert len(openai_messages) == 1
    assert openai_messages[0]["content"] == "[]"  # Empty list JSON string

    # Test with empty string (falsy but not None)
    message_with_empty_string = Message(
        role="tool",
        contents=[Content.from_function_result(call_id="call-456", result="")],
    )

    openai_messages = client._prepare_message_for_openai(message_with_empty_string)
    assert len(openai_messages) == 1
    assert openai_messages[0]["content"] == ""  # Empty string should be preserved

    # Test with False serialized as JSON string (pre-serialized result passed to from_function_result)
    message_with_false = Message(
        role="tool",
        contents=[Content.from_function_result(call_id="call-789", result="false")],
    )

    openai_messages = client._prepare_message_for_openai(message_with_false)
    assert len(openai_messages) == 1
    assert openai_messages[0]["content"] == "false"  # False JSON string

def test_function_result_exception_handling(openai_unit_test_env: dict[str, str]):
    """Test that exceptions in function result are properly handled.

    Feel free to remove this test in case there's another new behavior.
    """
    client = OpenAIChatCompletionClient()

    # Test with exception (no result)
    test_exception = ValueError("Test error message")
    message_with_exception = Message(
        role="tool",
        contents=[
            Content.from_function_result(
                call_id="call-123",
                result="Error: Function failed.",
                exception=test_exception,
            )
        ],
    )

    openai_messages = client._prepare_message_for_openai(message_with_exception)
    assert len(openai_messages) == 1
    assert openai_messages[0]["content"] == "Error: Function failed."
    assert openai_messages[0]["tool_call_id"] == "call-123"

def test_prepare_content_for_openai_data_content_image(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test _prepare_content_for_openai converts DataContent with image media type to OpenAI format."""
    client = OpenAIChatCompletionClient()

    # Test DataContent with image media type
    image_data_content = Content.from_uri(
        uri="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
        media_type="image/png",
    )

    result = client._prepare_content_for_openai(image_data_content)  # type: ignore

    # Should convert to OpenAI image_url format
    assert result["type"] == "image_url"
    assert result["image_url"]["url"] == image_data_content.uri

    # Test DataContent with non-image media type should use default model_dump
    text_data_content = Content.from_uri(uri="data:text/plain;base64,SGVsbG8gV29ybGQ=", media_type="text/plain")

    result = client._prepare_content_for_openai(text_data_content)  # type: ignore

    # Should use default model_dump format
    assert result["type"] == "data"
    assert result["uri"] == text_data_content.uri
    assert result["media_type"] == "text/plain"

    # Test DataContent with audio media type
    audio_data_content = Content.from_uri(
        uri="data:audio/wav;base64,UklGRjBEAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQwEAAAAAAAAAAAA",
        media_type="audio/wav",
    )

    result = client._prepare_content_for_openai(audio_data_content)  # type: ignore

    # Should convert to OpenAI input_audio format
    assert result["type"] == "input_audio"
    # Data should contain just the base64 part, not the full data URI
    assert result["input_audio"]["data"] == "UklGRjBEAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQwEAAAAAAAAAAAA"
    assert result["input_audio"]["format"] == "wav"

    # Test DataContent with MP3 audio
    mp3_data_content = Content.from_uri(
        uri="data:audio/mp3;base64,//uQAAAAWGluZwAAAA8AAAACAAACcQ==",
        media_type="audio/mp3",
    )

    result = client._prepare_content_for_openai(mp3_data_content)  # type: ignore

    # Should convert to OpenAI input_audio format with mp3
    assert result["type"] == "input_audio"
    # Data should contain just the base64 part, not the full data URI
    assert result["input_audio"]["data"] == "//uQAAAAWGluZwAAAA8AAAACAAACcQ=="
    assert result["input_audio"]["format"] == "mp3"

def test_prepare_content_for_openai_image_url_detail(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test _prepare_content_for_openai includes the detail field in image_url when specified."""
    client = OpenAIChatCompletionClient()

    # Test image with detail set to "high"
    image_with_detail = Content.from_uri(
        uri="https://example.com/image.png",
        media_type="image/png",
        additional_properties={"detail": "high"},
    )

    result = client._prepare_content_for_openai(image_with_detail)  # type: ignore

    assert result["type"] == "image_url"
    assert result["image_url"]["url"] == "https://example.com/image.png"
    assert result["image_url"]["detail"] == "high"

    # Test image with detail set to "low"
    image_low_detail = Content.from_uri(
        uri="https://example.com/image.png",
        media_type="image/png",
        additional_properties={"detail": "low"},
    )

    result = client._prepare_content_for_openai(image_low_detail)  # type: ignore

    assert result["image_url"]["detail"] == "low"

    # Test image with detail set to "auto"
    image_auto_detail = Content.from_uri(
        uri="https://example.com/image.png",
        media_type="image/png",
        additional_properties={"detail": "auto"},
    )

    result = client._prepare_content_for_openai(image_auto_detail)  # type: ignore

    assert result["image_url"]["detail"] == "auto"

    # Test image without detail should not include it
    image_no_detail = Content.from_uri(
        uri="https://example.com/image.png",
        media_type="image/png",
    )

    result = client._prepare_content_for_openai(image_no_detail)  # type: ignore

    assert result["type"] == "image_url"
    assert result["image_url"]["url"] == "https://example.com/image.png"
    assert "detail" not in result["image_url"]

    # Test image with a future/unknown string detail value should pass it through
    image_future_detail = Content.from_uri(
        uri="https://example.com/image.png",
        media_type="image/png",
        additional_properties={"detail": "ultra"},
    )

    result = client._prepare_content_for_openai(image_future_detail)  # type: ignore

    assert result["type"] == "image_url"
    assert result["image_url"]["url"] == "https://example.com/image.png"
    assert result["image_url"]["detail"] == "ultra"

    # Test image with data URI should include detail
    image_data_uri = Content.from_uri(
        uri="data:image/png;base64,iVBORw0KGgo",
        media_type="image/png",
        additional_properties={"detail": "high"},
    )

    result = client._prepare_content_for_openai(image_data_uri)  # type: ignore

    assert result["type"] == "image_url"
    assert result["image_url"]["url"] == "data:image/png;base64,iVBORw0KGgo"
    assert result["image_url"]["detail"] == "high"

    # Test image with non-string detail value should not include it
    image_non_string_detail = Content.from_uri(
        uri="https://example.com/image.png",
        media_type="image/png",
        additional_properties={"detail": 123},
    )

    result = client._prepare_content_for_openai(image_non_string_detail)  # type: ignore

    assert result["type"] == "image_url"
    assert result["image_url"]["url"] == "https://example.com/image.png"
    assert "detail" not in result["image_url"]

def test_prepare_content_for_openai_document_file_mapping(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test _prepare_content_for_openai converts document files (PDF, DOCX, etc.) to OpenAI file format."""
    client = OpenAIChatCompletionClient()

    # Test PDF without filename - should omit filename in OpenAI payload
    pdf_data_content = Content.from_uri(
        uri="data:application/pdf;base64,JVBERi0xLjQKJcfsj6IKNSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFI+PgplbmRvYmoKMiAwIG9iago8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PgplbmRvYmoKMyAwIG9iago8PC9UeXBlL1BhZ2UvTWVkaWFCb3ggWzAgMCA2MTIgNzkyXS9QYXJlbnQgMiAwIFIvUmVzb3VyY2VzPDwvRm9udDw8L0YxIDQgMCBSPj4+Pi9Db250ZW50cyA1IDAgUj4+CmVuZG9iago0IDAgb2JqCjw8L1R5cGUvRm9udC9TdWJ0eXBlL1R5cGUxL0Jhc2VGb250L0hlbHZldGljYT4+CmVuZG9iago1IDAgb2JqCjw8L0xlbmd0aCA0ND4+CnN0cmVhbQpCVApxCjcwIDUwIFRECi9GMSA4IFRmCihIZWxsbyBXb3JsZCEpIFRqCkVUCmVuZHN0cmVhbQplbmRvYmoKeHJlZgowIDYKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDA5IDAwMDAwIG4gCjAwMDAwMDAwNTggMDAwMDAgbiAKMDAwMDAwMDExNSAwMDAwMCBuIAowMDAwMDAwMjQ1IDAwMDAwIG4gCjAwMDAwMDAzMDcgMDAwMDAgbiAKdHJhaWxlcgo8PC9TaXplIDYvUm9vdCAxIDAgUj4+CnN0YXJ0eHJlZgo0MDUKJSVFT0Y=",
        media_type="application/pdf",
    )

    result = client._prepare_content_for_openai(pdf_data_content)  # type: ignore

    # Should convert to OpenAI file format without filename
    assert result["type"] == "file"
    assert "filename" not in result["file"]  # No filename provided, so none should be set
    assert "file_data" in result["file"]
    # Base64 data should be the full data URI (OpenAI requirement)
    assert result["file"]["file_data"].startswith("data:application/pdf;base64,")
    assert result["file"]["file_data"] == pdf_data_content.uri

    # Test PDF with custom filename via additional_properties
    pdf_with_filename = Content.from_uri(
        uri="data:application/pdf;base64,JVBERi0xLjQ=",
        media_type="application/pdf",
        additional_properties={"filename": "report.pdf"},
    )

    result = client._prepare_content_for_openai(pdf_with_filename)  # type: ignore

    # Should use custom filename
    assert result["type"] == "file"
    assert result["file"]["filename"] == "report.pdf"
    assert result["file"]["file_data"] == "data:application/pdf;base64,JVBERi0xLjQ="

    # Test different application/* media types - all should now be mapped to file format
    test_cases = [
        {
            "media_type": "application/json",
            "filename": "data.json",
            "base64": "eyJrZXkiOiJ2YWx1ZSJ9",
        },
        {
            "media_type": "application/xml",
            "filename": "config.xml",
            "base64": "PD94bWwgdmVyc2lvbj0iMS4wIj8+",
        },
        {
            "media_type": "application/octet-stream",
            "filename": "binary.bin",
            "base64": "AQIDBAUGBwgJCg==",
        },
    ]

    for case in test_cases:
        # Test without filename
        doc_content = Content.from_uri(
            uri=f"data:{case['media_type']};base64,{case['base64']}",
            media_type=case["media_type"],
        )

        result = client._prepare_content_for_openai(doc_content)  # type: ignore

        # All application/* types should now be mapped to file format
        assert result["type"] == "file"
        assert "filename" not in result["file"]  # Should omit filename when not provided
        assert result["file"]["file_data"] == doc_content.uri

        # Test with filename - should now use file format with filename
        doc_with_filename = Content.from_uri(
            uri=f"data:{case['media_type']};base64,{case['base64']}",
            media_type=case["media_type"],
            additional_properties={"filename": case["filename"]},
        )

        result = client._prepare_content_for_openai(doc_with_filename)  # type: ignore

        # Should now use file format with filename
        assert result["type"] == "file"
        assert result["file"]["filename"] == case["filename"]
        assert result["file"]["file_data"] == doc_with_filename.uri

    # Test edge case: empty additional_properties dict
    pdf_empty_props = Content.from_uri(
        uri="data:application/pdf;base64,JVBERi0xLjQ=",
        media_type="application/pdf",
        additional_properties={},
    )

    result = client._prepare_content_for_openai(pdf_empty_props)  # type: ignore

    assert result["type"] == "file"
    assert "filename" not in result["file"]

    # Test edge case: None filename in additional_properties
    pdf_none_filename = Content.from_uri(
        uri="data:application/pdf;base64,JVBERi0xLjQ=",
        media_type="application/pdf",
        additional_properties={"filename": None},
    )

    result = client._prepare_content_for_openai(pdf_none_filename)  # type: ignore

    assert result["type"] == "file"
    assert "filename" not in result["file"]  # None filename should be omitted

def test_function_approval_content_is_skipped_in_preparation(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that function approval request and response content are skipped."""
    client = OpenAIChatCompletionClient()

    # Create approval request
    function_call = Content.from_function_call(
        call_id="call_123",
        name="dangerous_action",
        arguments='{"confirm": true}',
    )

    approval_request = Content.from_function_approval_request(
        id="approval_001",
        function_call=function_call,
    )

    # Create approval response
    approval_response = Content.from_function_approval_response(
        approved=False,
        id="approval_001",
        function_call=function_call,
    )

    # Test that approval request is skipped
    message_with_request = Message(role="assistant", contents=[approval_request])
    prepared_request = client._prepare_message_for_openai(message_with_request)
    assert len(prepared_request) == 0  # Should be empty - approval content is skipped

    # Test that approval response is skipped
    message_with_response = Message(role="user", contents=[approval_response])
    prepared_response = client._prepare_message_for_openai(message_with_response)
    assert len(prepared_response) == 0  # Should be empty - approval content is skipped

    # Test with mixed content - approval should be skipped, text should remain
    mixed_message = Message(
        role="assistant",
        contents=[
            Content.from_text(text="I need approval for this action."),
            approval_request,
        ],
    )
    prepared_mixed = client._prepare_message_for_openai(mixed_message)
    assert len(prepared_mixed) == 1  # Only text content should remain
    assert prepared_mixed[0]["content"] == "I need approval for this action."

def test_prepare_options_without_model(openai_unit_test_env: dict[str, str]) -> None:
    """Test that prepare_options raises error when model is not set."""
    client = OpenAIChatCompletionClient()
    client.model = None  # Remove model

    messages = [Message(role="user", contents=["test"])]

    with pytest.raises(ValueError, match="model must be a non-empty string"):
        client._prepare_options(messages, {})

def test_prepare_options_without_messages(openai_unit_test_env: dict[str, str]) -> None:
    """Test that prepare_options raises error when messages are missing."""
    from agent_framework.exceptions import ChatClientInvalidRequestException

    client = OpenAIChatCompletionClient()

    with pytest.raises(ChatClientInvalidRequestException, match="Messages are required"):
        client._prepare_options([], {})

def test_prepare_tools_with_web_search_no_location(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test preparing web search tool without user location."""
    client = OpenAIChatCompletionClient()

    # Web search tool using static method
    web_search_tool = OpenAIChatCompletionClient.get_web_search_tool()

    result = client._prepare_tools_for_openai([web_search_tool])

    # Should have empty web_search_options (no location)
    assert "web_search_options" in result
    assert result["web_search_options"] == {}

def test_prepare_options_with_instructions(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that instructions are prepended as system message."""
    client = OpenAIChatCompletionClient()

    messages = [Message(role="user", contents=["Hello"])]
    options = {"instructions": "You are a helpful assistant."}

    prepared_options = client._prepare_options(messages, options)

    # Should have messages with system message prepended
    assert "messages" in prepared_options
    assert len(prepared_options["messages"]) == 2
    assert prepared_options["messages"][0]["role"] == "system"
    assert prepared_options["messages"][0]["content"] == "You are a helpful assistant."

def test_prepare_options_with_instructions_no_duplicate(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that duplicate system message from instructions is not added again.

    Regression test for https://github.com/microsoft/agent-framework/issues/5049
    """
    client = OpenAIChatCompletionClient()

    # Simulate messages that already contain the system instruction
    messages = [
        Message(role="system", contents=["You are a helpful assistant."]),
        Message(role="user", contents=["Hello"]),
    ]
    options = {"instructions": "You are a helpful assistant."}

    prepared_options = client._prepare_options(messages, options)

    # Should NOT duplicate the system message
    assert "messages" in prepared_options
    assert len(prepared_options["messages"]) == 2
    assert prepared_options["messages"][0]["role"] == "system"
    assert prepared_options["messages"][0]["content"] == "You are a helpful assistant."
    assert prepared_options["messages"][1]["role"] == "user"

def test_prepare_message_with_author_name(openai_unit_test_env: dict[str, str]) -> None:
    """Test that author_name is included in prepared message."""
    client = OpenAIChatCompletionClient()

    message = Message(
        role="user",
        author_name="TestUser",
        contents=[Content.from_text(text="Hello")],
    )

    prepared = client._prepare_message_for_openai(message)

    assert len(prepared) == 1
    assert prepared[0]["name"] == "TestUser"

def test_prepare_message_with_tool_result_author_name(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that author_name is not included for TOOL role messages."""
    client = OpenAIChatCompletionClient()

    # Tool messages should not have 'name' field (it's for function name instead)
    message = Message(
        role="tool",
        author_name="ShouldNotAppear",
        contents=[Content.from_function_result(call_id="call_123", result="result")],
    )

    prepared = client._prepare_message_for_openai(message)

    assert len(prepared) == 1
    # Should not have 'name' field for tool messages
    assert "name" not in prepared[0]

def test_prepare_system_message_content_is_string(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that system message content is a plain string, not a list.

    Some OpenAI-compatible endpoints (e.g. NVIDIA NIM) reject system messages
    with list content. See https://github.com/microsoft/agent-framework/issues/1407
    """
    client = OpenAIChatCompletionClient()

    message = Message(role="system", contents=[Content.from_text(text="You are a helpful assistant.")])

    prepared = client._prepare_message_for_openai(message)

    assert len(prepared) == 1
    assert prepared[0]["role"] == "system"
    assert isinstance(prepared[0]["content"], str)
    assert prepared[0]["content"] == "You are a helpful assistant."

def test_prepare_developer_message_content_is_string(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that developer message content is a plain string, not a list."""
    client = OpenAIChatCompletionClient()

    message = Message(role="developer", contents=[Content.from_text(text="Follow these rules.")])

    prepared = client._prepare_message_for_openai(message)

    assert len(prepared) == 1
    assert prepared[0]["role"] == "developer"
    assert isinstance(prepared[0]["content"], str)
    assert prepared[0]["content"] == "Follow these rules."

def test_prepare_system_message_multiple_text_contents_joined(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that system messages with multiple text contents are joined into a single string."""
    client = OpenAIChatCompletionClient()

    message = Message(
        role="system",
        contents=[
            Content.from_text(text="You are a helpful assistant."),
            Content.from_text(text="Be concise."),
        ],
    )

    prepared = client._prepare_message_for_openai(message)

    assert len(prepared) == 1
    assert prepared[0]["role"] == "system"
    assert isinstance(prepared[0]["content"], str)
    assert prepared[0]["content"] == "You are a helpful assistant.\nBe concise."

def test_prepare_user_message_text_content_is_string(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that text-only user message content is flattened to a plain string.

    Some OpenAI-compatible endpoints (e.g. Foundry Local) cannot deserialize
    the list format. See https://github.com/microsoft/agent-framework/issues/4084
    """
    client = OpenAIChatCompletionClient()

    message = Message(role="user", contents=[Content.from_text(text="Hello")])

    prepared = client._prepare_message_for_openai(message)

    assert len(prepared) == 1
    assert prepared[0]["role"] == "user"
    assert isinstance(prepared[0]["content"], str)
    assert prepared[0]["content"] == "Hello"

def test_prepare_user_message_multimodal_content_remains_list(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that multimodal user message content remains a list."""
    client = OpenAIChatCompletionClient()

    message = Message(
        role="user",
        contents=[
            Content.from_text(text="What's in this image?"),
            Content.from_uri(uri="https://example.com/image.png", media_type="image/png"),
        ],
    )

    prepared = client._prepare_message_for_openai(message)

    # Multimodal content must stay as list for the API
    has_list_content = any(isinstance(m.get("content"), list) for m in prepared)
    assert has_list_content

def test_prepare_assistant_message_text_content_is_string(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that text-only assistant message content is flattened to a plain string."""
    client = OpenAIChatCompletionClient()

    message = Message(role="assistant", contents=[Content.from_text(text="Sure, I can help.")])

    prepared = client._prepare_message_for_openai(message)

    assert len(prepared) == 1
    assert prepared[0]["role"] == "assistant"
    assert isinstance(prepared[0]["content"], str)
    assert prepared[0]["content"] == "Sure, I can help."

def test_tool_choice_required_with_function_name(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that tool_choice with required mode and function name is correctly prepared."""
    client = OpenAIChatCompletionClient()

    messages = [Message(role="user", contents=["test"])]
    options = {
        "tools": [get_weather],
        "tool_choice": {"mode": "required", "required_function_name": "get_weather"},
    }

    prepared_options = client._prepare_options(messages, options)

    # Should format tool_choice correctly
    assert "tool_choice" in prepared_options
    assert prepared_options["tool_choice"]["type"] == "function"
    assert prepared_options["tool_choice"]["function"]["name"] == "get_weather"

def test_response_format_dict_passthrough(openai_unit_test_env: dict[str, str]) -> None:
    """Test that response_format as dict is passed through directly."""
    client = OpenAIChatCompletionClient()

    messages = [Message(role="user", contents=["test"])]
    custom_format = {
        "type": "json_schema",
        "json_schema": {"name": "Test", "schema": {"type": "object"}},
    }
    options = {"response_format": custom_format}

    prepared_options = client._prepare_options(messages, options)

    # Should pass through the dict directly
    assert prepared_options["response_format"] == custom_format

def test_parse_response_with_dict_response_format(openai_unit_test_env: dict[str, str]) -> None:
    """Chat completions should parse dict response_format values into response.value."""
    client = OpenAIChatCompletionClient()
    response = client._parse_response_from_openai(
        ChatCompletion(
            id="test-response",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content='{"answer": "Hello"}'),
                    finish_reason="stop",
                )
            ],
        ),
        options={"response_format": {"type": "object", "properties": {"answer": {"type": "string"}}}},
    )

    assert response.value is not None
    assert isinstance(response.value, dict)
    assert response.value["answer"] == "Hello"

def test_multiple_function_calls_in_single_message(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that multiple function calls in a message are correctly prepared."""
    client = OpenAIChatCompletionClient()

    # Create message with multiple function calls
    message = Message(
        role="assistant",
        contents=[
            Content.from_function_call(call_id="call_1", name="func_1", arguments='{"a": 1}'),
            Content.from_function_call(call_id="call_2", name="func_2", arguments='{"b": 2}'),
        ],
    )

    prepared = client._prepare_message_for_openai(message)

    # Should have one message with multiple tool_calls
    assert len(prepared) == 1
    assert "tool_calls" in prepared[0]
    assert len(prepared[0]["tool_calls"]) == 2
    assert prepared[0]["tool_calls"][0]["id"] == "call_1"
    assert prepared[0]["tool_calls"][1]["id"] == "call_2"

def test_prepare_options_removes_parallel_tool_calls_when_no_tools(
    openai_unit_test_env: dict[str, str],
) -> None:
    """Test that parallel_tool_calls is removed when no tools are present."""
    client = OpenAIChatCompletionClient()

    messages = [Message(role="user", contents=["test"])]
    options = {"allow_multiple_tool_calls": True}

    prepared_options = client._prepare_options(messages, options)

    # Should not have parallel_tool_calls when no tools
    assert "parallel_tool_calls" not in prepared_options

def test_prepare_options_excludes_conversation_id(openai_unit_test_env: dict[str, str]) -> None:
    """Test that conversation_id is excluded from prepared options for chat completions."""
    client = OpenAIChatCompletionClient()

    messages = [Message(role="user", contents=["test"])]
    options = {"conversation_id": "12345", "temperature": 0.7}

    prepared_options = client._prepare_options(messages, options)

    # conversation_id is not a valid parameter for AsyncCompletions.create()
    assert "conversation_id" not in prepared_options
    # Other options should still be present
    assert prepared_options["temperature"] == 0.7

async def test_integration_options(
    option_name: str,
    option_value: Any,
    needs_validation: bool,
) -> None:
    """Parametrized test covering all ChatOptions and OpenAIChatCompletionOptions.

    Tests both streaming and non-streaming modes for each option to ensure
    they don't cause failures. Options marked with needs_validation also
    check that the feature actually works correctly.
    """
    client = OpenAIChatCompletionClient()
    # Need at least 2 iterations for tool_choice tests: one to get function call, one to get final response
    client.function_invocation_configuration["max_iterations"] = 2

    # Prepare test message
    if option_name.startswith("tools") or option_name.startswith("tool_choice"):
        # Use weather-related prompt for tool tests
        messages = [Message(role="user", contents=["What is the weather in Seattle?"])]
    elif option_name.startswith("response_format"):
        # Use prompt that works well with structured output
        messages = [Message(role="user", contents=["The weather in Seattle is sunny"])]
        messages.append(Message(role="user", contents=["What is the weather in Seattle?"]))
    else:
        # Generic prompt for simple options
        messages = [Message(role="user", contents=["Say 'Hello World' briefly."])]

    # Build options dict
    options: dict[str, Any] = {option_name: option_value}

    # Add tools if testing tool_choice to avoid errors
    if option_name.startswith("tool_choice"):
        options["tools"] = [get_weather]

    # Test streaming mode
    response = await client.get_response(
        messages=messages,
        stream=True,
        options=options,
    ).get_final_response()

    assert response is not None
    assert isinstance(response, ChatResponse)
    assert response.messages is not None
    if not option_name.startswith("tool_choice") and (
        (isinstance(option_value, str) and option_value != "required")
        or (isinstance(option_value, dict) and option_value.get("mode") != "required")
    ):
        assert response.text is not None, f"No text in response for option '{option_name}'"
        assert len(response.text) > 0, f"Empty response for option '{option_name}'"

    # Validate based on option type
    if needs_validation:
        if option_name.startswith("tools") or option_name.startswith("tool_choice"):
            # Should have called the weather function
            text = response.text.lower()
            assert "sunny" in text or "seattle" in text, f"Tool not invoked for {option_name}"
        elif option_name.startswith("response_format"):
            if option_value == OutputStruct:
                # Should have structured output
                assert response.value is not None, "No structured output"
                assert isinstance(response.value, OutputStruct)
                assert "seattle" in response.value.location.lower()
            else:
                assert response.value is not None
                assert isinstance(response.value, dict)
                assert "location" in response.value
                assert "seattle" in response.value["location"].lower()

async def test_integration_web_search() -> None:
    client = OpenAIChatCompletionClient(model="gpt-4o-search-preview")

    for streaming in [False, True]:
        # Use static method for web search tool
        web_search_tool = OpenAIChatCompletionClient.get_web_search_tool()
        content = {
            "messages": [
                Message(
                    role="user",
                    contents=["Who are the main characters of Kpop Demon Hunters? Do a web search to find the answer."],
                )
            ],
            "options": {
                "tool_choice": "auto",
                "tools": [web_search_tool],
            },
        }
        if streaming:
            response = await client.get_response(stream=True, **content).get_final_response()
        else:
            response = await client.get_response(**content)

        assert response is not None
        assert isinstance(response, ChatResponse)
        assert "Rumi" in response.text
        assert "Mira" in response.text
        assert "Zoey" in response.text

        # Test that the client will use the web search tool with location
        web_search_tool_with_location = OpenAIChatCompletionClient.get_web_search_tool(
            web_search_options={
                "user_location": {
                    "type": "approximate",
                    "approximate": {"country": "US", "city": "Seattle"},
                },
            }
        )
        content = {
            "messages": [
                Message(
                    role="user",
                    contents=["What is the current weather? Do not ask for my current location."],
                )
            ],
            "options": {
                "tool_choice": "auto",
                "tools": [web_search_tool_with_location],
            },
        }
        if streaming:
            response = await client.get_response(stream=True, **content).get_final_response()
        else:
            response = await client.get_response(**content)
        assert response.text is not None


# --- python/packages/openai/tests/openai/test_openai_chat_completion_client_azure.py ---

def test_init_with_azure_endpoint(azure_openai_unit_test_env: dict[str, str]) -> None:
    client = OpenAIChatCompletionClient(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))

    assert client.model == azure_openai_unit_test_env["AZURE_OPENAI_CHAT_COMPLETION_MODEL"]
    assert isinstance(client, SupportsChatGetResponse)
    assert isinstance(client.client, AsyncAzureOpenAI)
    assert client.OTEL_PROVIDER_NAME == "azure.ai.openai"
    assert client.azure_endpoint == azure_openai_unit_test_env["AZURE_OPENAI_ENDPOINT"]
    assert client.api_version == azure_openai_unit_test_env["AZURE_OPENAI_API_VERSION"]

def test_init_auto_detects_azure_env(azure_openai_unit_test_env: dict[str, str]) -> None:
    client = OpenAIChatCompletionClient()

    assert client.model == azure_openai_unit_test_env["AZURE_OPENAI_CHAT_COMPLETION_MODEL"]
    assert isinstance(client.client, AsyncAzureOpenAI)
    assert client.azure_endpoint == azure_openai_unit_test_env["AZURE_OPENAI_ENDPOINT"]

async def test_azure_openai_chat_completion_client_response() -> None:
    async with AzureCliCredential() as credential:
        client = OpenAIChatCompletionClient(credential=credential)
        assert isinstance(client, SupportsChatGetResponse)

        messages = [
            Message(
                role="user",
                contents=[
                    (
                        "Emily and David, two passionate scientists, met during a research expedition to Antarctica. "
                        "Bonded by their love for the natural world and shared curiosity, they uncovered a "
                        "groundbreaking phenomenon in glaciology that could potentially reshape our understanding "
                        "of climate change."
                    )
                ],
            ),
            Message(role="user", contents=["who are Emily and David?"]),
        ]

        response = await client.get_response(messages=messages)

        assert response is not None
        assert isinstance(response, ChatResponse)
        assert any(
            word in response.text.lower() for word in ["scientists", "research", "antarctica", "glaciology", "climate"]
        )

async def test_azure_openai_chat_completion_client_response_tools() -> None:
    async with AzureCliCredential() as credential:
        client = OpenAIChatCompletionClient(credential=credential)

        response = await client.get_response(
            messages=[Message(role="user", contents=["who are Emily and David?"])],
            options={"tools": [get_story_text], "tool_choice": "auto"},
        )

        assert response is not None
        assert isinstance(response, ChatResponse)
        assert "Emily" in response.text or "David" in response.text

async def test_azure_openai_chat_completion_client_streaming() -> None:
    async with AzureCliCredential() as credential:
        client = OpenAIChatCompletionClient(credential=credential)

        response = client.get_response(
            messages=[
                Message(
                    role="user",
                    contents=[
                        (
                            "Emily and David, two passionate scientists, met during a research expedition to "
                            "Antarctica. Bonded by their love for the natural world and shared curiosity, they "
                            "uncovered a groundbreaking phenomenon in glaciology that could potentially reshape our "
                            "understanding of climate change."
                        )
                    ],
                ),
                Message(role="user", contents=["who are Emily and David?"]),
            ],
            stream=True,
        )

        full_message = ""
        async for chunk in response:
            assert isinstance(chunk, ChatResponseUpdate)
            assert chunk.message_id is not None
            assert chunk.response_id is not None
            for content in chunk.contents:
                if content.type == "text" and content.text:
                    full_message += content.text

        assert "Emily" in full_message or "David" in full_message

async def test_azure_openai_chat_completion_client_streaming_tools() -> None:
    async with AzureCliCredential() as credential:
        client = OpenAIChatCompletionClient(credential=credential)

        response = client.get_response(
            messages=[Message(role="user", contents=["who are Emily and David?"])],
            stream=True,
            options={"tools": [get_story_text], "tool_choice": "auto"},
        )

        full_message = ""
        async for chunk in response:
            assert isinstance(chunk, ChatResponseUpdate)
            for content in chunk.contents:
                if content.type == "text" and content.text:
                    full_message += content.text

        assert "Emily" in full_message or "David" in full_message

async def test_azure_openai_chat_completion_client_agent_basic_run() -> None:
    async with (
        AzureCliCredential() as credential,
        Agent(
            client=OpenAIChatCompletionClient(credential=credential),
        ) as agent,
    ):
        response = await agent.run("Please respond with exactly: 'This is a response test.'")

        assert isinstance(response, AgentResponse)
        assert response.text is not None
        assert "response test" in response.text.lower()

async def test_azure_openai_chat_completion_client_agent_basic_run_streaming() -> None:
    async with (
        AzureCliCredential() as credential,
        Agent(client=OpenAIChatCompletionClient(credential=credential)) as agent,
    ):
        full_text = ""
        async for chunk in agent.run("Please respond with exactly: 'This is a streaming response test.'", stream=True):
            assert isinstance(chunk, AgentResponseUpdate)
            if chunk.text:
                full_text += chunk.text

        assert "streaming response test" in full_text.lower()

async def test_azure_openai_chat_completion_client_agent_session_persistence() -> None:
    async with (
        AzureCliCredential() as credential,
        Agent(
            client=OpenAIChatCompletionClient(credential=credential),
            instructions="You are a helpful assistant with good memory.",
        ) as agent,
    ):
        session = agent.create_session()
        response1 = await agent.run("My name is Alice. Remember this.", session=session)
        response2 = await agent.run("What is my name?", session=session)

        assert isinstance(response1, AgentResponse)
        assert isinstance(response2, AgentResponse)
        assert response2.text is not None
        assert "alice" in response2.text.lower()

async def test_azure_openai_chat_completion_client_agent_existing_session() -> None:
    async with AzureCliCredential() as credential:
        preserved_session = None

        async with Agent(
            client=OpenAIChatCompletionClient(credential=credential),
            instructions="You are a helpful assistant with good memory.",
        ) as first_agent:
            session = first_agent.create_session()
            first_response = await first_agent.run("My name is Alice. Remember this.", session=session)

            assert isinstance(first_response, AgentResponse)
            preserved_session = session

        if preserved_session:
            async with Agent(
                client=OpenAIChatCompletionClient(credential=credential),
                instructions="You are a helpful assistant with good memory.",
            ) as second_agent:
                second_response = await second_agent.run("What is my name?", session=preserved_session)

                assert isinstance(second_response, AgentResponse)
                assert second_response.text is not None
                assert "alice" in second_response.text.lower()

async def test_azure_chat_completion_client_agent_level_tool_persistence() -> None:
    async with (
        AzureCliCredential() as credential,
        Agent(
            client=OpenAIChatCompletionClient(credential=credential),
            instructions="You are a helpful assistant that uses available tools.",
            tools=[get_weather],
        ) as agent,
    ):
        first_response = await agent.run("What's the weather like in Chicago?")
        second_response = await agent.run("What's the weather in Miami?")

        assert isinstance(first_response, AgentResponse)
        assert isinstance(second_response, AgentResponse)
        assert first_response.text is not None
        assert second_response.text is not None
        assert any(term in first_response.text.lower() for term in ["chicago", "sunny", "72"])
        assert any(term in second_response.text.lower() for term in ["miami", "sunny", "72"])

