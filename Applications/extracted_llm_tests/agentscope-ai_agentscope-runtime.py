# agentscope-ai/agentscope-runtime
# 11 LLM-backed test functions across 60 test files
# Source: https://github.com/agentscope-ai/agentscope-runtime

# --- tests/integrated/test_agent_app.py ---

async def test_process_endpoint_stream_async(start_app):
    """
    Async test for streaming /process endpoint (SSE, multiple JSON events).
    """
    url = f"http://localhost:{PORT}/process"
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the capital of France?"},
                ],
            },
        ],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )

            found_paris = False

            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue

                line = chunk.decode("utf-8").strip()
                # SSE lines start with "data:"
                if line.startswith("data:"):
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        # Ignore non‑JSON keepalive messages or partial lines
                        continue

                    # Check if this event has "output" from the assistant
                    if "output" in event:
                        try:
                            text_content = event["output"][0]["content"][0][
                                "text"
                            ].lower()
                            if "paris" in text_content:
                                found_paris = True
                        except Exception:
                            # Structure may differ; ignore
                            pass

            # Final assertion — we must have seen "paris" in at least one event
            assert (
                found_paris
            ), "Did not find 'paris' in any streamed output event"

async def test_multi_turn_stream_async(start_app):
    """
    Async test for multi‑turn conversation with streaming output.
    Ensures that the agent remembers the user's name from a previous turn.
    """
    session_id = "123456"

    url = f"http://localhost:{PORT}/process"

    async with aiohttp.ClientSession() as session:
        payload1 = {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "My name is Alice."}],
                },
            ],
            "session_id": session_id,
        }
        async with session.post(url, json=payload1) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )
            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                line = chunk.decode("utf-8").strip()
                if (
                    line.startswith("data:")
                    and line[len("data:") :].strip() == "[DONE]"
                ):
                    break

    payload2 = {
        "input": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is my name?"}],
            },
        ],
        "session_id": session_id,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload2) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )

            found_name = False

            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                line = chunk.decode("utf-8").strip()
                if line.startswith("data:"):
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "output" in event:
                        try:
                            text_content = event["output"][0]["content"][0][
                                "text"
                            ].lower()
                            if "alice" in text_content:
                                found_name = True
                        except Exception:
                            pass

            assert found_name, "Did not find 'Alice' in the second turn output"


# --- tests/integrated/test_agno_agent_app.py ---

async def test_process_endpoint_stream_async(start_app):
    """
    Async test for streaming /process endpoint (SSE, multiple JSON events).
    """
    url = f"http://localhost:{PORT}/process"
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the capital of France?"},
                ],
            },
        ],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )

            found_paris = False

            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue

                line = chunk.decode("utf-8").strip()
                # SSE lines start with "data:"
                if line.startswith("data:"):
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        # Ignore non‑JSON keepalive messages or partial lines
                        continue

                    # Check if this event has "output" from the assistant
                    if "output" in event:
                        try:
                            text_content = event["output"][0]["content"][0][
                                "text"
                            ].lower()
                            if "paris" in text_content:
                                found_paris = True
                        except Exception:
                            # Structure may differ; ignore
                            pass

            # Final assertion — we must have seen "paris" in at least one event
            assert (
                found_paris
            ), "Did not find 'paris' in any streamed output event"

async def test_multi_turn_stream_async(start_app):
    """
    Async test for multi‑turn conversation with streaming output.
    Ensures that the agent remembers the user's name from a previous turn.
    """
    session_id = "123456"

    url = f"http://localhost:{PORT}/process"

    async with aiohttp.ClientSession() as session:
        payload1 = {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "My name is Alice."}],
                },
            ],
            "session_id": session_id,
        }
        async with session.post(url, json=payload1) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )
            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                line = chunk.decode("utf-8").strip()
                if (
                    line.startswith("data:")
                    and line[len("data:") :].strip() == "[DONE]"
                ):
                    break

    payload2 = {
        "input": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is my name?"}],
            },
        ],
        "session_id": session_id,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload2) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )

            found_name = False

            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                line = chunk.decode("utf-8").strip()
                if line.startswith("data:"):
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "output" in event:
                        try:
                            text_content = event["output"][0]["content"][0][
                                "text"
                            ].lower()
                            if "alice" in text_content:
                                found_name = True
                        except Exception:
                            pass

            assert found_name, "Did not find 'Alice' in the second turn output"


# --- tests/integrated/test_agui_integration.py ---

    async def test_simple_text_exchange(
        self,
        app_endpoint: tuple[str, int],
    ):
        """Test simple text exchange through AG-UI protocol with real LLM."""
        host, port = app_endpoint
        url = f"http://{host}:{port}/ag-ui"
        custom_thread_id = "test_thread_1"
        custom_run_id = "test_run_1"
        ag_ui_request = RunAgentInput(
            threadId=custom_thread_id,
            runId=custom_run_id,
            messages=[
                UserMessage(
                    id="msg_1",
                    content="What is 2+2? Answer in one sentence.",
                ),
            ],
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
        )
        events: List[Event] = []
        event_adapter = TypeAdapter(Event)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=ag_ui_request.model_dump(mode="json"),
            ) as resp:
                assert resp.status == 200
                assert (
                    resp.headers["content-type"]
                    == "text/event-stream; charset=utf-8"
                )

                # Parse SSE events
                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        events.append(event_adapter.validate_json(data_str))

        # Verify event sequence
        assert len(events) >= 3, "Should have at least 3 events"

        # Should have run.started
        run_started = [e for e in events if e.type == EventType.RUN_STARTED]
        assert len(run_started) > 0, "Should have RUN_STARTED event"
        assert run_started[0].thread_id == custom_thread_id
        assert run_started[0].run_id == custom_run_id

        # Should have text message events
        text_events = [
            e
            for e in events
            if e.type
            in {
                EventType.TEXT_MESSAGE_START,
                EventType.TEXT_MESSAGE_CONTENT,
                EventType.TEXT_MESSAGE_END,
            }
        ]
        assert len(text_events) > 0, "Should have text message events"

        # Should have run.finished
        run_finished = [e for e in events if e.type == EventType.RUN_FINISHED]
        assert len(run_finished) > 0, "Should have RUN_FINISHED event"

    async def test_conversation_with_history(
        self,
        app_endpoint: tuple[str, int],
    ):
        """Test conversation with message history through AG-UI."""
        host, port = app_endpoint
        url = f"http://{host}:{port}/ag-ui"

        # First turn: tell agent the user's name
        thread_id = "test_thread_history"
        ag_ui_request_1 = RunAgentInput(
            threadId=thread_id,
            runId="test_run_h1",
            messages=[
                SystemMessage(
                    id="msg_1",
                    content="You are a helpful assistant.",
                ),
                UserMessage(
                    id="msg_2",
                    content="My name is Bob. Please remember it.",
                ),
            ],
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
        )
        event_adapter = TypeAdapter(Event)
        events_1: List[Event] = []

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=ag_ui_request_1.model_dump(mode="json"),
            ) as resp:
                assert resp.status == 200
                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        events_1.append(event_adapter.validate_json(data_str))
        assert any(
            e.type == EventType.RUN_FINISHED for e in events_1
        ), "First turn should finish"

        # Second turn: ask agent to recall the name
        ag_ui_request_2 = RunAgentInput(
            threadId=thread_id,
            runId="test_run_h2",
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
            messages=[
                SystemMessage(
                    id="msg_1",
                    content="You are a helpful assistant.",
                ),
                UserMessage(
                    id="msg_2",
                    content="My name is Bob. Please remember it.",
                ),
                AssistantMessage(
                    id="msg_3",
                    content="Nice to meet you, Bob! I'll remember your name.",
                ),
                UserMessage(
                    id="msg_4",
                    content="What is my name?",
                ),
            ],
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=ag_ui_request_2.model_dump(mode="json"),
            ) as resp:
                assert resp.status == 200

                events: List[Event] = []
                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        events.append(event_adapter.validate_json(data_str))

                # Verify response mentions Bob
                content_events = [
                    e
                    for e in events
                    if e.type == EventType.TEXT_MESSAGE_CONTENT
                ]
                response_text = "".join(e.delta for e in content_events)

                assert (
                    "Bob" in response_text or "bob" in response_text.lower()
                ), "Agent should remember and mention Bob"

    async def test_tool_call(
        self,
        app_endpoint: tuple[str, int],
    ):
        """Test tool call through AG-UI."""
        host, port = app_endpoint
        url = f"http://{host}:{port}/ag-ui"
        thread_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())

        ag_ui_request = RunAgentInput(
            threadId=thread_id,
            runId=run_id,
            messages=[
                UserMessage(
                    id=str(uuid.uuid4()),
                    content="北京今天的天气如何?",
                ),
            ],
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
        )
        events = await invoke_api(url, ag_ui_request)

        run_started_event = [
            e for e in events if e.type == EventType.RUN_STARTED
        ]
        assert (
            len(run_started_event) == 1
        ), "Should have exactly one RUN_STARTED event"
        assert run_started_event[0].thread_id == thread_id
        assert run_started_event[0].run_id == run_id

        run_finished_event = [
            e for e in events if e.type == EventType.RUN_FINISHED
        ]
        assert (
            len(run_finished_event) == 1
        ), "Should have exactly one RUN_FINISHED event"
        assert run_finished_event[0].thread_id == thread_id
        assert run_finished_event[0].run_id == run_id

        tool_call_start_events = [
            e for e in events if e.type == EventType.TOOL_CALL_START
        ]
        assert (
            len(tool_call_start_events) > 0
        ), "Should have TOOL_CALL_START event"

        tool_call_args_events = [
            e for e in events if e.type == EventType.TOOL_CALL_ARGS
        ]
        assert (
            len(tool_call_args_events) > 0
        ), "Should have TOOL_CALL_ARGS event"

        tool_call_end_events = [
            e for e in events if e.type == EventType.TOOL_CALL_END
        ]
        assert len(tool_call_end_events) > 0, "Should have TOOL_CALL_END event"

        tool_call_result_events = [
            e for e in events if e.type == EventType.TOOL_CALL_RESULT
        ]
        assert (
            len(tool_call_result_events) >= 1
        ), "Should have exactly one TOOL_CALL_RESULT event"
        tool_call_id = str(uuid.uuid4())

        multi_turn_request = RunAgentInput(
            thread_id=thread_id,
            run_id=str(uuid.uuid4()),
            messages=[
                UserMessage(
                    id=str(uuid.uuid4()),
                    content="北京今天的天气如何?",
                ),
                AssistantMessage(
                    id=str(uuid.uuid4()),
                    content="The weather in Beijing is sunny with a "
                    "temperature of 25°C.",
                    tool_calls=[
                        ToolCall(
                            id=tool_call_id,
                            function=FunctionCall(
                                name="get_weather",
                                arguments='{"location": "Beijing"}',
                            ),
                        ),
                    ],
                ),
                ToolMessage(
                    id=str(uuid.uuid4()),
                    content="The weather in Beijing is sunny with a "
                    "temperature of 25°C.",
                    tool_call_id=tool_call_id,
                ),
                AssistantMessage(
                    id=str(uuid.uuid4()),
                    content="北京的天气是晴朗的，气温为25°C。",
                ),
                UserMessage(
                    id=str(uuid.uuid4()),
                    content="那杭州的呢？",
                ),
            ],
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
        )
        multi_turn_events = await invoke_api(url, multi_turn_request)

        run_started_event = [
            e for e in multi_turn_events if e.type == EventType.RUN_STARTED
        ]
        assert (
            len(run_started_event) == 1
        ), "Should have exactly one RUN_STARTED event"
        assert run_started_event[0].thread_id == thread_id
        assert run_started_event[0].run_id == multi_turn_request.run_id

        run_finished_event = [
            e for e in multi_turn_events if e.type == EventType.RUN_FINISHED
        ]
        assert (
            len(run_finished_event) == 1
        ), "Should have exactly one RUN_FINISHED event"
        assert run_finished_event[0].thread_id == thread_id
        assert run_finished_event[0].run_id == multi_turn_request.run_id

        tool_call_result_events = [
            e
            for e in multi_turn_events
            if e.type == EventType.TOOL_CALL_RESULT
        ]
        assert (
            len(tool_call_result_events) >= 1
        ), "Should have exactly one TOOL_CALL_RESULT event"


# --- tests/integrated/test_langgraph_agent_app.py ---

async def test_langgraph_process_endpoint_stream_async(start_langgraph_app):
    """
    Async test for streaming /process endpoint (SSE, multiple JSON events).
    """
    url = f"http://localhost:{PORT}/process"
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the capital of France?"},
                ],
            },
        ],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )

            found_response = False
            chunks = []

            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                chunks.append(chunk.decode("utf-8").strip())

            line = chunks[-1]
            # SSE lines start with "data:"
            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
                event = json.loads(data_str)

                # Check if this event has "output" from the assistant
                if "output" in event:
                    try:
                        text_content = event["output"][-1]["content"][0][
                            "text"
                        ].lower()
                        if "paris" in text_content:
                            found_response = True
                    except Exception:
                        # Structure may differ; ignore
                        pass

            # Final assertion — we must have seen "paris" in at least one event
            assert (
                found_response
            ), "Did not find 'paris' in any streamed output event"

async def test_langgraph_multi_turn_stream_async(start_langgraph_app):
    """
    Async test for multi-turn conversation with streaming output.
    """
    session_id = "langgraph_test_session"
    url = f"http://localhost:{PORT}/process"

    # First turn
    async with aiohttp.ClientSession() as session:
        payload1 = {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello LangGraph!"}],
                },
            ],
            "session_id": session_id,
        }
        async with session.post(url, json=payload1) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )
            # Simply consume the stream without detailed checking
            chunks = []
            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                chunks.append(chunk.decode("utf-8").strip())

            found_response = False
            line = chunks[-1]
            # SSE lines start with "data:"
            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
                event = json.loads(data_str)

    payload2 = {
        "input": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "How are you?"}],
            },
        ],
        "session_id": session_id,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload2) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )

            chunks = []
            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                chunks.append(chunk.decode("utf-8").strip())

            found_response = False
            line = chunks[-1]
            # SSE lines start with "data:"
            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
                event = json.loads(data_str)

                # Check if this event has "output" from the assistant
                if "output" in event:
                    try:
                        text_content = event["output"][-1]["content"][0][
                            "text"
                        ].lower()
                        if text_content:
                            found_response = True
                    except Exception:
                        # Structure may differ; ignore
                        pass

            assert (
                found_response
            ), "Did not find expected response in the second turn output"


# --- tests/integrated/test_runner_stream_langgraph.py ---

async def test_runner_sample1():
    from dotenv import load_dotenv

    load_dotenv("../../.env")

    request = AgentRequest.model_validate(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's the weather like in Hangzhou?",
                        },
                    ],
                },
                {
                    "type": "function_call",
                    "content": [
                        {
                            "type": "data",
                            "data": {
                                "call_id": "call_eb113ba709d54ab6a4dcbf",
                                "name": "get_current_weather",
                                "arguments": '{"location": "Hangzhou"}',
                            },
                        },
                    ],
                },
                {
                    "type": "function_call_output",
                    "content": [
                        {
                            "type": "data",
                            "data": {
                                "call_id": "call_eb113ba709d54ab6a4dcbf",
                                "output": '{"temperature": 25, "unit": '
                                '"Celsius"}',
                            },
                        },
                    ],
                },
            ],
            "stream": True,
            "session_id": "Test Session",
        },
    )

    print("\n")
    final_text = ""
    async with MyLangGraphRunner() as runner:
        async for message in runner.stream_query(
            request=request,
        ):
            print(message.model_dump_json())
            if message.object == "message":
                if MessageType.MESSAGE == message.type:
                    if RunStatus.Completed == message.status:
                        res = message.content
                        print(res)
                        if res and len(res) > 0:
                            final_text = res[0].text
                            print(final_text)
                if MessageType.FUNCTION_CALL == message.type:
                    if RunStatus.Completed == message.status:
                        res = message.content
                        print(res)

        print("\n")
    assert "Hangzhou" in final_text

async def test_runner_sample2():
    from dotenv import load_dotenv

    load_dotenv("../../.env")

    request = AgentRequest.model_validate(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in https://example.com?",
                        },
                    ],
                },
            ],
            "stream": True,
            "session_id": "Test Session",
        },
    )

    print("\n")
    final_text = ""
    async with MyLangGraphRunner() as runner:
        async for message in runner.stream_query(
            request=request,
        ):
            print(message.model_dump_json())
            if message.object == "message":
                if MessageType.MESSAGE == message.type:
                    if RunStatus.Completed == message.status:
                        res = message.content
                        print(res)
                        if res and len(res) > 0:
                            final_text = res[0].text
                            print(final_text)
                if MessageType.FUNCTION_CALL == message.type:
                    if RunStatus.Completed == message.status:
                        res = message.content
                        print(res)

        print("\n")

    assert "example.com" in final_text

