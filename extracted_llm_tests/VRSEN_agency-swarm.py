# VRSEN/agency-swarm
# 64 test functions with real LLM calls
# Source: https://github.com/VRSEN/agency-swarm


# --- tests/integration/agent/test_additional_instructions.py ---

async def test_agent_get_response_stream_applies_additional_instructions_and_restores_original() -> None:
    original_instructions = "Base agent instructions."
    additional_instructions = "Streaming run instructions."
    agent = Agent(
        name="TestAgent",
        instructions=original_instructions,
        model=SystemInstructionsEchoModel(),
    )

    stream = agent.get_response_stream("hello", additional_instructions=additional_instructions)
    async for _event in stream:
        pass

    assert stream.final_output == f"{original_instructions}\n\n{additional_instructions}"
    assert agent.instructions == original_instructions


# --- tests/integration/communication/test_streaming_order_consistency.py ---

async def test_full_streaming_flow_hardcoded_sequence(
    use_anthropic: bool, expected_flow: list[tuple[str, str, str | None]]
) -> None:
    """Proves canonical streaming order for Main→Sub agent with tool calls is deterministic."""
    if use_anthropic:
        pytest.importorskip("litellm", reason="litellm package is required for Anthropic test")
        import litellm
        from agents.extensions.models.litellm_model import LitellmModel

        litellm.modify_params = True

        main_model = LitellmModel(model=ANTHROPIC_MODEL_NAME)
        helper_model = LitellmModel(model=ANTHROPIC_MODEL_NAME)
    else:
        main_model = "gpt-5.4-mini"
        helper_model = "gpt-5.4-mini"

    main = Agent(
        name="MainAgent",
        description="Coordinator",
        instructions=(
            "First send a standalone 'ACK' message before any tool calls. "
            "Then call get_market_data('AAPL'). "
            "Then use the send_message tool to ask SubAgent to analyze the data and reply. "
            "Finally, respond to the user with a brief conclusion."
        ),
        model=main_model,
        tools=[get_market_data],
    )

    helper = Agent(
        name="SubAgent",
        description="Risk analyzer",
        instructions=("When prompted by MainAgent: call analyze_risk on the provided data, then reply succinctly."),
        model=helper_model,
        tools=[analyze_risk],
    )

    agency = Agency(
        main,
        communication_flows=[main > helper],
        shared_instructions="",
    )

    before = len(agency.thread_manager.get_all_messages())

    # Collect stream as (type, agent, tool_name)
    stream_items: list[tuple[str, str, str | None]] = []
    async for event in agency.get_response_stream(message="Start."):
        if hasattr(event, "item") and event.item is not None:
            item = event.item
            evt_type = getattr(item, "type", None)
            if evt_type == "reasoning_item":
                continue
            agent_name = getattr(event, "agent", None)
            tool_name = None
            if evt_type == "tool_call_item":
                raw = getattr(item, "raw_item", None)
                tool_name = getattr(raw, "name", None)
            if isinstance(evt_type, str) and isinstance(agent_name, str):
                stream_items.append((evt_type, agent_name, tool_name))

    all_messages = agency.thread_manager.get_all_messages()
    new_messages = all_messages[before:]

    # Map saved messages to same triple format
    comparable: list[dict[str, Any]] = []
    for m in new_messages:
        t = m.get("type")
        role = m.get("role")
        if t in {"function_call", "function_call_output"} or role == "assistant":
            comparable.append(m)

    expected_without_main_message = _strip_optional_initial_message_output(expected_flow, "MainAgent")
    assert stream_items in (expected_flow, expected_without_main_message), (
        "Stream flow mismatch:\n"
        f" got={stream_items}\n"
        f" exp={expected_flow}\n"
        f" exp_without_initial_message={expected_without_main_message}"
    )

    _assert_sanitized_history(comparable)

async def test_multiple_sequential_subagent_calls() -> None:
    """Proves repeated send_message to same sub-agent streams in strict canonical order."""
    coordinator = Agent(
        name="Coordinator",
        description="Main coordinator",
        instructions=(
            "First say 'ACK'. Then call get_market_data('TEST'). "
            "Then use send_message to ask Worker to process the data. "
            "After Worker responds, use send_message again to ask Worker to validate the result. "
            "Finally, respond with 'DONE'."
        ),
        model_settings=ModelSettings(temperature=0.0),
        tools=[get_market_data],
    )

    worker = Agent(
        name="Worker",
        description="Data processor",
        instructions=(
            "When asked to process: use process_data tool and respond 'Processed'. "
            "When asked to validate: use validate_result tool and respond 'Validated'."
        ),
        model_settings=ModelSettings(temperature=0.0),
        tools=[process_data, validate_result],
    )

    agency = Agency(
        coordinator,
        communication_flows=[coordinator > worker],
        shared_instructions="",
    )

    before = len(agency.thread_manager.get_all_messages())

    # Collect stream events
    stream_items: list[tuple[str, str, str | None]] = []
    async for event in agency.get_response_stream(message="Execute multiple tasks."):
        if hasattr(event, "item") and event.item is not None:
            item = event.item
            evt_type = getattr(item, "type", None)
            if evt_type == "reasoning_item":
                continue
            agent_name = getattr(event, "agent", None)
            tool_name = None
            if evt_type == "tool_call_item":
                raw = getattr(item, "raw_item", None)
                tool_name = getattr(raw, "name", None)
            if isinstance(evt_type, str) and isinstance(agent_name, str):
                stream_items.append((evt_type, agent_name, tool_name))

    # Verify stream matches expected
    assert stream_items == EXPECTED_FLOW_MULTIPLE_CALLS, (
        f"Multiple calls stream mismatch:\n got={stream_items}\n exp={EXPECTED_FLOW_MULTIPLE_CALLS}"
    )

    # Verify saved messages
    all_messages = agency.thread_manager.get_all_messages()
    new_messages = all_messages[before:]

    comparable: list[dict[str, Any]] = []
    for m in new_messages:
        t = m.get("type")
        role = m.get("role")
        if t in {"function_call", "function_call_output"} or role == "assistant":
            comparable.append(m)

    _assert_sanitized_history(comparable)

async def test_nested_delegation_streaming() -> None:
    """Proves nested A→B→C delegation appears in stream and AgentA completes after sub-chain."""
    agent_a = Agent(
        name="AgentA",
        description="Top-level coordinator",
        instructions=(
            "First say 'ACK'. "
            "Then use send_message to ask AgentB to process and analyze data. "
            "Finally respond with 'Complete'."
        ),
        model="gpt-5.4-mini",
        tools=[],
    )

    agent_b = Agent(
        name="AgentB",
        description="Middle processor",
        instructions=(
            "When asked by AgentA: "
            "First use send_message to ask AgentC to analyze risk. "
            "Then use process_data tool with the response. "
            "Finally respond 'Processed'."
        ),
        model="gpt-5.4-mini",
        model_settings=ModelSettings(tool_choice="required"),
        tools=[process_data],
    )

    agent_c = Agent(
        name="AgentC",
        description="Risk analyzer",
        instructions="When asked: use analyze_risk tool and respond 'Risk analyzed'.",
        model="gpt-5.4-mini",
        model_settings=ModelSettings(tool_choice="required"),
        tools=[analyze_risk],
    )

    agency = Agency(
        agent_a,
        communication_flows=[agent_a > agent_b, agent_b > agent_c],
        shared_instructions="",
    )

    before = len(agency.thread_manager.get_all_messages())

    # Collect stream events
    stream_items: list[tuple[str, str, str | None]] = []
    async for event in agency.get_response_stream(message="Start nested delegation."):
        if hasattr(event, "item") and event.item is not None:
            item = event.item
            evt_type = getattr(item, "type", None)
            if evt_type == "reasoning_item":
                continue
            agent_name = getattr(event, "agent", None)
            tool_name = None
            if evt_type == "tool_call_item":
                raw = getattr(item, "raw_item", None)
                tool_name = getattr(raw, "name", None)
            if isinstance(evt_type, str) and isinstance(agent_name, str):
                stream_items.append((evt_type, agent_name, tool_name))

    # Verify stream contains the required sequence in order and AgentC performs analyze_risk
    required_seq = [
        ("tool_call_item", "AgentA", "send_message"),
        ("tool_call_item", "AgentB", "send_message"),
        ("tool_call_item", "AgentC", "analyze_risk"),
        ("tool_call_output_item", "AgentA", None),
        ("message_output_item", "AgentA", None),
    ]

    def is_subsequence(needles: list[tuple[str, str, str | None]], haystack: list[tuple[str, str, str | None]]) -> bool:
        i = 0
        for item in haystack:
            if i < len(needles) and item == needles[i]:
                i += 1
        return i == len(needles)

    assert is_subsequence(required_seq, stream_items), (
        f"Nested delegation stream mismatch (required subsequence not found):\n got={stream_items}\n req={required_seq}"
    )

    # Verify saved messages
    all_messages = agency.thread_manager.get_all_messages()
    new_messages = all_messages[before:]

    comparable: list[dict[str, Any]] = []
    for m in new_messages:
        t = m.get("type")
        role = m.get("role")
        if t in {"function_call", "function_call_output"} or role == "assistant":
            comparable.append(m)

    _assert_sanitized_history(comparable)

    # Verify stream contains the required sequence in order (for saved messages verification)
    required_seq = [
        ("tool_call_item", "AgentA", "send_message"),
        ("tool_call_item", "AgentB", "send_message"),
        ("tool_call_output_item", "AgentA", None),
        ("message_output_item", "AgentA", None),
    ]
    assert is_subsequence(required_seq, stream_items), (
        f"Nested delegation stream mismatch (required subsequence not found):\n got={stream_items}\n req={required_seq}"
    )

async def test_parallel_subagent_calls() -> None:
    """Proves orchestrator issues two sub-agent calls and completion follows canonical order."""
    orchestrator = Agent(
        name="Orchestrator",
        description="Main orchestrator",
        instructions=(
            "Call get_market_data('DATA'). "
            "Then use send_message to ask ProcessorA to process the data. "
            "After ProcessorA responds, use send_message to ask ProcessorB to validate. "
            "Finally, use combine_results tool and respond 'All done'."
        ),
        model_settings=ModelSettings(temperature=0.0),
        tools=[get_market_data, combine_results],
    )

    processor_a = Agent(
        name="ProcessorA",
        description="Data processor",
        instructions="When asked: use process_data tool and respond 'ProcessorA complete'.",
        model_settings=ModelSettings(temperature=0.0, tool_choice="required"),
        tools=[process_data],
    )

    processor_b = Agent(
        name="ProcessorB",
        description="Result validator",
        instructions="When asked: use validate_result tool and respond 'ProcessorB complete'.",
        model_settings=ModelSettings(temperature=0.0, tool_choice="required"),
        tools=[validate_result],
    )

    agency = Agency(
        orchestrator,
        communication_flows=[orchestrator > processor_a, orchestrator > processor_b],
        shared_instructions="",
    )

    before = len(agency.thread_manager.get_all_messages())

    # Collect stream events
    stream_items: list[tuple[str, str, str | None]] = []
    async for event in agency.get_response_stream(message="Coordinate parallel work."):
        if hasattr(event, "item") and event.item is not None:
            item = event.item
            evt_type = getattr(item, "type", None)
            if evt_type == "reasoning_item":
                continue
            agent_name = getattr(event, "agent", None)
            tool_name = None
            if evt_type == "tool_call_item":
                raw = getattr(item, "raw_item", None)
                tool_name = getattr(raw, "name", None)
            if isinstance(evt_type, str) and isinstance(agent_name, str):
                stream_items.append((evt_type, agent_name, tool_name))

    # Verify stream matches expected (strict assertion)
    if stream_items != EXPECTED_FLOW_PARALLEL:
        logger.error(
            "Parallel sub-agent stream mismatch",
            extra={
                "got": stream_items,
                "expected": EXPECTED_FLOW_PARALLEL,
            },
        )
    assert stream_items == EXPECTED_FLOW_PARALLEL, (
        f"Parallel calls stream mismatch:\n got={stream_items}\n exp={EXPECTED_FLOW_PARALLEL}"
    )

    # Verify saved messages
    all_messages = agency.thread_manager.get_all_messages()
    new_messages = all_messages[before:]

    comparable: list[dict[str, Any]] = []
    for m in new_messages:
        t = m.get("type")
        role = m.get("role")
        if t in {"function_call", "function_call_output"} or role == "assistant":
            comparable.append(m)

    _assert_tool_call_recorded(new_messages, "ProcessorA", "process_data", context="parallel workflow")
    _assert_tool_call_recorded(new_messages, "ProcessorB", "validate_result", context="parallel workflow")
    _assert_sanitized_history(comparable)


# --- tests/integration/fastapi/test_fastapi_metadata.py ---

def test_metadata_includes_agent_capabilities():
    """Verify that metadata includes capabilities for each agent."""

    class CustomTool(BaseTool):
        """Custom tool for testing."""

        def run(self) -> str:
            return "custom"

    @function_tool
    def sample_function() -> str:
        """Sample function tool."""
        return "sample"

    def create_agency(load_threads_callback=None, save_threads_callback=None):
        # Agent with custom tools
        agent1 = Agent(name="ToolAgent", instructions="Test", tools=[CustomTool, sample_function])
        # Agent with hosted tools
        agent2 = Agent(
            name="HostedAgent",
            instructions="Test",
            tools=[
                FileSearchTool(vector_store_ids=["vs_123"]),
                CodeInterpreterTool(tool_config=CodeInterpreter()),
                WebSearchTool(),
            ],
        )
        agent3 = Agent(
            name="ReasoningAgent",
            instructions="Test",
            model="gpt-5.4-mini",
        )
        agent4 = Agent(
            name="FullAgent",
            instructions="Test",
            model="gpt-5.4-mini",
            tools=[CustomTool, FileSearchTool(vector_store_ids=["vs_456"])],
        )
        return Agency(
            agent1,
            communication_flows=[(agent1, agent2), (agent1, agent3), (agent1, agent4)],
            load_threads_callback=load_threads_callback,
            save_threads_callback=save_threads_callback,
        )

    app = run_fastapi(agencies={"test_agency": create_agency}, return_app=True, app_token_env="")
    client = TestClient(app)

    response = client.get("/test_agency/get_metadata")

    assert response.status_code == 200
    payload = response.json()
    assert "allowed_local_file_dirs" in payload

    # Find agents in nodes
    nodes = payload.get("nodes", [])
    assert len(nodes) > 0

    # Find specific agents and verify capabilities
    tool_agent = next((n for n in nodes if n["id"] == "ToolAgent"), None)
    assert tool_agent is not None
    assert "capabilities" in tool_agent["data"]
    assert "tools" in tool_agent["data"]["capabilities"]

    hosted_agent = next((n for n in nodes if n["id"] == "HostedAgent"), None)
    assert hosted_agent is not None
    assert "capabilities" in hosted_agent["data"]
    capabilities = set(hosted_agent["data"]["capabilities"])
    assert capabilities == {"file_search", "code_interpreter", "web_search"}
    assert "tools" not in capabilities

    reasoning_agent = next((n for n in nodes if n["id"] == "ReasoningAgent"), None)
    assert reasoning_agent is not None
    assert "capabilities" in reasoning_agent["data"]
    assert "reasoning" in reasoning_agent["data"]["capabilities"]

    full_agent = next((n for n in nodes if n["id"] == "FullAgent"), None)
    assert full_agent is not None
    assert "capabilities" in full_agent["data"]
    capabilities = set(full_agent["data"]["capabilities"])
    assert "tools" in capabilities
    assert "reasoning" in capabilities
    assert "file_search" in capabilities

def test_metadata_capabilities_empty_for_basic_agent():
    """Agent with no special features has empty capabilities list."""

    def create_agency(load_threads_callback=None, save_threads_callback=None):
        agent = Agent(name="BasicAgent", instructions="Basic agent with no tools")
        return Agency(agent, load_threads_callback=load_threads_callback, save_threads_callback=save_threads_callback)

    app = run_fastapi(agencies={"test_agency": create_agency}, return_app=True, app_token_env="")
    client = TestClient(app)

    response = client.get("/test_agency/get_metadata")

    assert response.status_code == 200
    payload = response.json()
    assert "allowed_local_file_dirs" in payload
    nodes = payload.get("nodes", [])
    basic_agent = next((n for n in nodes if n["id"] == "BasicAgent"), None)
    assert basic_agent is not None
    assert "capabilities" in basic_agent["data"]
    assert basic_agent["data"]["capabilities"] == []

def test_metadata_includes_allowed_local_file_dirs(tmp_path, agency_factory):
    """Metadata should expose the allowed local file directories configuration."""
    allowed_dir = tmp_path / "uploads"
    allowed_dir.mkdir(parents=True, exist_ok=True)

    app = run_fastapi(
        agencies={"test_agency": agency_factory},
        return_app=True,
        app_token_env="",
        allowed_local_file_dirs=[str(allowed_dir)],
    )
    client = TestClient(app)

    response = client.get("/test_agency/get_metadata")
    assert response.status_code == 200
    payload = response.json()

    assert payload["allowed_local_file_dirs"] == [str(allowed_dir.resolve())]

def test_metadata_includes_quick_replies() -> None:
    """Metadata should expose both starters and quick replies for UI clients."""

    def create_agency(load_threads_callback=None, save_threads_callback=None):
        agent = Agent(
            name="QuickRepliesAgent",
            instructions="Use quick replies",
            conversation_starters=["Support: I need help with billing"],
            quick_replies=["hi", "hello"],
        )
        return Agency(
            agent,
            load_threads_callback=load_threads_callback,
            save_threads_callback=save_threads_callback,
        )

    app = run_fastapi(agencies={"test_agency": create_agency}, return_app=True, app_token_env="")
    client = TestClient(app)

    response = client.get("/test_agency/get_metadata")

    assert response.status_code == 200
    payload = response.json()
    nodes = payload.get("nodes", [])
    quick_agent = next((n for n in nodes if n["id"] == "QuickRepliesAgent"), None)
    assert quick_agent is not None
    data = quick_agent["data"]
    assert data["conversationStarters"] == ["Support: I need help with billing"]
    assert data["quickReplies"] == ["hi", "hello"]

def test_metadata_endpoint_reads_live_metadata():
    """Metadata should reflect the current agency factory state on each request."""

    state = {"quick": ["hi"]}

    def create_agency(load_threads_callback=None, save_threads_callback=None):
        agent = Agent(
            name="LiveMetadataAgent",
            instructions="Use quick replies",
            quick_replies=list(state["quick"]),
        )
        return Agency(
            agent,
            load_threads_callback=load_threads_callback,
            save_threads_callback=save_threads_callback,
        )

    app = run_fastapi(agencies={"test_agency": create_agency}, return_app=True, app_token_env="")
    client = TestClient(app)

    first = client.get("/test_agency/get_metadata")
    assert first.status_code == 200
    first_agent = next((n for n in first.json().get("nodes", []) if n["id"] == "LiveMetadataAgent"), None)
    assert first_agent is not None
    assert first_agent["data"]["quickReplies"] == ["hi"]

    state["quick"] = ["bye"]

    second = client.get("/test_agency/get_metadata")
    assert second.status_code == 200
    second_agent = next((n for n in second.json().get("nodes", []) if n["id"] == "LiveMetadataAgent"), None)
    assert second_agent is not None
    assert second_agent["data"]["quickReplies"] == ["bye"]

def test_metadata_includes_tool_input_schema():
    """Metadata should include input schema for function tools when available."""

    @function_tool
    def sample_tool(text: str) -> str:
        return text

    def create_agency(load_threads_callback=None, save_threads_callback=None):
        agent = Agent(name="SchemaAgent", instructions="Test", tools=[sample_tool])
        return Agency(agent, load_threads_callback=load_threads_callback, save_threads_callback=save_threads_callback)

    app = run_fastapi(agencies={"test_agency": create_agency}, return_app=True, app_token_env="")
    client = TestClient(app)

    response = client.get("/test_agency/get_metadata")

    assert response.status_code == 200
    payload = response.json()
    schema_agent = next((n for n in payload.get("nodes", []) if n["id"] == "SchemaAgent"), None)
    assert schema_agent is not None
    tools = schema_agent["data"].get("tools", [])
    assert tools
    input_schema = tools[0].get("inputSchema")
    assert isinstance(input_schema, dict)
    assert "text" in input_schema.get("properties", {})

def test_metadata_includes_missing_allowed_dirs(tmp_path, agency_factory):
    """Missing allowed local file directories should still appear in metadata."""
    missing_dir = tmp_path / "missing-uploads"

    app = run_fastapi(
        agencies={"test_agency": agency_factory},
        return_app=True,
        app_token_env="",
        allowed_local_file_dirs=[str(missing_dir)],
    )
    client = TestClient(app)

    response = client.get("/test_agency/get_metadata")
    assert response.status_code == 200
    payload = response.json()

    assert payload["allowed_local_file_dirs"] == [str(missing_dir)]

def test_metadata_includes_non_directory_allowed_dirs(tmp_path, agency_factory):
    """Non-directory allowlist entries should appear in metadata as configured."""
    file_entry = tmp_path / "not-a-directory.txt"
    file_entry.write_text("x", encoding="utf-8")

    app = run_fastapi(
        agencies={"test_agency": agency_factory},
        return_app=True,
        app_token_env="",
        allowed_local_file_dirs=[str(file_entry)],
    )
    client = TestClient(app)

    response = client.get("/test_agency/get_metadata")
    assert response.status_code == 200
    payload = response.json()

    assert payload["allowed_local_file_dirs"] == [str(file_entry)]

def test_tool_endpoint_handles_nested_schema():
    """Test that tool endpoints work with nested Pydantic models."""

    class Address(BaseModel):
        street: str
        zip_code: int

    class NestedTool(BaseTool):
        address: Address

        def run(self) -> str:
            return self.address.street

    app = run_fastapi(tools=[NestedTool], return_app=True, app_token_env="")
    client = TestClient(app)

    response = client.post("/tool/NestedTool", json={"address": {"street": "Elm", "zip_code": 90210}})

    assert response.status_code == 200
    assert response.json() == {"response": "Elm"}

def test_openapi_json_includes_nested_schemas():
    """Verify /openapi.json contains proper schemas for tools with nested models."""

    class Address(BaseModel):
        street: str
        zip_code: int

    class NestedTool(BaseTool):
        address: Address

        def run(self) -> str:
            return self.address.street

    class SimpleTool(BaseTool):
        name: str
        age: int

        def run(self) -> str:
            return self.name

    app = run_fastapi(tools=[NestedTool, SimpleTool], return_app=True, app_token_env="")
    client = TestClient(app)

    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    assert "/tool/NestedTool" in schema["paths"]
    assert "/tool/SimpleTool" in schema["paths"]

    nested_endpoint = schema["paths"]["/tool/NestedTool"]["post"]
    assert "requestBody" in nested_endpoint
    nested_schema_ref = nested_endpoint["requestBody"]["content"]["application/json"]["schema"]
    assert nested_schema_ref["$ref"] == "#/components/schemas/NestedTool"

    assert "NestedTool" in schema["components"]["schemas"]
    assert "Address" in schema["components"]["schemas"]

    nested_tool_schema = schema["components"]["schemas"]["NestedTool"]
    assert nested_tool_schema["properties"]["address"]["$ref"] == "#/components/schemas/Address"

    address_schema = schema["components"]["schemas"]["Address"]
    assert address_schema["type"] == "object"
    assert "street" in address_schema["properties"]
    assert "zip_code" in address_schema["properties"]
    assert address_schema["required"] == ["street", "zip_code"]

def test_function_tool_with_nested_schema():
    """Verify that FunctionTools with nested models work correctly via adapted BaseTool."""

    class Address(BaseModel):
        street: str
        zip_code: int

    class UserTool(BaseTool):
        """Create a user with address."""

        name: str
        address: Address

        def run(self) -> str:
            return f"{self.name} at {self.address.street}"

    # Adapt the BaseTool to a FunctionTool (simulates what happens in agents)
    function_tool = ToolFactory.adapt_base_tool(UserTool)

    app = run_fastapi(tools=[function_tool], return_app=True, app_token_env="")
    client = TestClient(app)

    # Test that the endpoint works
    response = client.post(
        "/tool/UserTool", json={"name": "Alice", "address": {"street": "123 Main St", "zip_code": 12345}}
    )
    assert response.status_code == 200
    assert "Alice at 123 Main St" in response.json()["response"]

    # Test that OpenAPI schema includes nested model
    schema_response = client.get("/openapi.json")
    assert schema_response.status_code == 200
    openapi_schema = schema_response.json()

    assert "/tool/UserTool" in openapi_schema["paths"]
    endpoint_schema = openapi_schema["paths"]["/tool/UserTool"]["post"]
    assert "requestBody" in endpoint_schema

    # Verify the schema is properly typed (not generic Request)
    request_schema = endpoint_schema["requestBody"]["content"]["application/json"]["schema"]
    assert "$ref" in request_schema
    assert "UserToolRequest" in request_schema["$ref"]

def test_strict_function_tool_rejects_extra_fields():
    """Ensure strict tools exposed via FastAPI still validate unexpected inputs."""

    class StrictTool(BaseTool):
        """Return the given value."""

        class ToolConfig:
            strict = True

        value: int

        def run(self) -> int:
            return self.value

    strict_function_tool = ToolFactory.adapt_base_tool(StrictTool)

    app = run_fastapi(tools=[strict_function_tool], return_app=True, app_token_env="")
    client = TestClient(app)

    response = client.post("/tool/StrictTool", json={"value": 7, "unexpected": "boom"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(item.get("type") == "extra_forbidden" for item in detail)

def test_tool_endpoint_preserves_explicit_nulls():
    """Tools must receive explicit null payloads without them being dropped."""

    class NullableTool(BaseTool):
        note: str | None = None

        def run(self) -> str | None:
            return self.note

    app = run_fastapi(tools=[NullableTool], return_app=True, app_token_env="")
    client = TestClient(app)

    response = client.post("/tool/NullableTool", json={"note": None})

    assert response.status_code == 200
    assert response.json() == {"response": None}

def test_function_tool_nested_list_validation_survives_schema_export():
    """FunctionTools should retain nested list schemas after ToolFactory exports."""

    class Address(BaseModel):
        street: str
        zip_code: int

    class AddressListTool(BaseTool):
        addresses: list[Address]

        def run(self) -> str:
            return ",".join(addr.street for addr in self.addresses)

    function_tool = ToolFactory.adapt_base_tool(AddressListTool)
    ToolFactory.get_openapi_schema([function_tool], "https://api.test.com")

    app = run_fastapi(tools=[function_tool], return_app=True, app_token_env="")
    client = TestClient(app)

    # Missing zip_code inside nested list should raise a FastAPI validation error (422)
    invalid_response = client.post("/tool/AddressListTool", json={"addresses": [{"street": "Elm"}]})
    assert invalid_response.status_code == 422

    valid_response = client.post(
        "/tool/AddressListTool",
        json={"addresses": [{"street": "Elm", "zip_code": 90210}]},
    )

    assert valid_response.status_code == 200
    assert valid_response.json() == {"response": "Elm"}

def test_function_tool_nested_union_falls_back_to_generic_handler():
    """Nested unions should bypass the typed request model to avoid false 422 errors."""

    class Contact(BaseModel):
        identifier: str | int

    class ContainerTool(BaseTool):
        contact: Contact

        def run(self) -> str:
            return str(self.contact.identifier)

    function_tool = ToolFactory.adapt_base_tool(ContainerTool)

    app = run_fastapi(tools=[function_tool], return_app=True, app_token_env="")
    client = TestClient(app)

    response = client.post("/tool/ContainerTool", json={"contact": {"identifier": 99}})

    assert response.status_code == 200
    assert response.json() == {"response": "99"}

    schema = client.get("/openapi.json").json()
    endpoint = schema["paths"]["/tool/ContainerTool"]["post"]
    assert "requestBody" not in endpoint

def test_openapi_schema_includes_custom_server_url():
    """/openapi.json should expose the configured server base URL."""

    class EchoTool(BaseTool):
        message: str

        def run(self) -> str:
            return self.message

    app = run_fastapi(
        tools=[EchoTool],
        return_app=True,
        app_token_env="",
        server_url="https://api.example.com/base",
    )
    client = TestClient(app)

    schema = client.get("/openapi.json").json()

    assert schema["servers"] == [{"url": "https://api.example.com/base"}]
    assert "/tool/EchoTool" in schema["paths"]


# --- tests/integration/fastapi/test_fastapi_user_context.py ---

def test_non_streaming_user_context(recording_agency_factory: RecordingAgencyFactory):
    """Ensure user_context is forwarded to non-streaming endpoint."""
    app = run_fastapi(agencies={"test_agency": recording_agency_factory}, return_app=True, app_token_env="")
    client = TestClient(app)

    response = client.post(
        "/test_agency/get_response",
        json={"message": "Hello", "user_context": {"plan": "pro", "user_id": "123"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "usage" in payload
    usage = payload["usage"]
    assert usage["request_count"] == 1
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 20
    assert usage["total_tokens"] == 30
    assert isinstance(usage["total_cost"], int | float)
    assert recording_agency_factory.tracker.last_response_context == {"plan": "pro", "user_id": "123"}

def test_streaming_user_context(recording_agency_factory: RecordingAgencyFactory):
    """Ensure user_context is forwarded to streaming endpoint."""
    app = run_fastapi(agencies={"test_agency": recording_agency_factory}, return_app=True, app_token_env="")
    client = TestClient(app)

    with client.stream(
        "POST",
        "/test_agency/get_response_stream",
        json={"message": "Hello", "user_context": {"plan": "pro"}},
    ) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())

    stream_context = recording_agency_factory.tracker.last_stream_context
    assert stream_context is not None
    assert {k: v for k, v in stream_context.items() if k != "streaming_context"} == {"plan": "pro"}
    assert "streaming_context" in stream_context

    messages_payload = _extract_last_messages_payload(lines)
    usage = messages_payload["usage"]
    assert usage["request_count"] == 1
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 20
    assert usage["total_tokens"] == 30
    assert isinstance(usage["total_cost"], int | float)

def test_agui_user_context(recording_agency_factory: RecordingAgencyFactory):
    """Ensure AG-UI streaming endpoint forwards user_context."""
    app = run_fastapi(
        agencies={"test_agency": recording_agency_factory},
        return_app=True,
        app_token_env="",
        enable_agui=True,
    )
    client = TestClient(app)

    agui_payload = {
        "thread_id": "test_thread",
        "run_id": "test_run",
        "state": None,
        "messages": [{"id": "msg1", "role": "user", "content": "Hello"}],
        "tools": [],
        "context": [],
        "forwardedProps": None,
        "user_context": {"plan": "pro", "customer_tier": "gold"},
    }

    with client.stream("POST", "/test_agency/get_response_stream", json=agui_payload) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    stream_context = recording_agency_factory.tracker.last_stream_context
    assert stream_context is not None
    assert {k: v for k, v in stream_context.items() if k != "streaming_context"} == {
        "plan": "pro",
        "customer_tier": "gold",
    }
    assert "streaming_context" in stream_context

def test_user_context_defaults_to_none(recording_agency_factory: RecordingAgencyFactory):
    """Requests without user_context should not inject overrides."""
    app = run_fastapi(agencies={"test_agency": recording_agency_factory}, return_app=True, app_token_env="")
    client = TestClient(app)

    response = client.post("/test_agency/get_response", json={"message": "Hello"})

    assert response.status_code == 200
    assert recording_agency_factory.tracker.last_response_context is None


# --- tests/integration/files/test_file_handling.py ---

async def test_file_search_tool(real_openai_client: AsyncOpenAI, tmp_path: Path):
    """Tests that an agent can use FileSearch tool to process files."""
    file_search_agent, folder_path, tmp_file_path, test_txt_path = await _setup_file_search_agent(
        real_openai_client, tmp_path
    )

    try:
        await _wait_for_vector_store(real_openai_client, file_search_agent)

        # Initialize agency and run test
        agency = Agency(file_search_agent, user_context=None)
        question = (
            "Use FileSearch with the query 'hobbit' to find the answer: What is the title of the 4th book in the list?"
        )

        try:
            from agents import RunConfig

            # Single-turn: enforce FileSearch tool usage deterministically
            response_result = await agency.get_response(
                question,
                run_config=RunConfig(model_settings=ModelSettings(tool_choice="file_search")),
            )
            assert response_result is not None
            print(f"Response for {test_txt_path.name}: {response_result.final_output}")

            # Verify FileSearch was used and expected content found
            final_output_lower = response_result.final_output.lower()
            hobbit_found = any(term in final_output_lower for term in ["hobbit", "the hobbit", "j.r.r. tolkien"])

            if not hobbit_found:
                print("Expected content not found, checking if FileSearch was used")
                tool_calls_made = [
                    item for item in response_result.new_items if hasattr(item, "tool_calls") and item.tool_calls
                ]
                file_search_used = any(
                    any(call.type == "file_search" for call in item.tool_calls if hasattr(call, "type"))
                    for item in tool_calls_made
                )
                if not file_search_used:
                    print("FileSearch tool was not used, this may explain why the answer wasn't found")

            assert hobbit_found, f"Expected 'hobbit' or related terms not found in: {response_result.final_output}"

        except Exception as e:
            # Handle 404 errors with retry
            if "404" in str(e) and "Files" in str(e):
                print(f"Files not found error, re-uploading and retrying: {e}")
                uploaded_file_id = file_search_agent.upload_file(str(tmp_file_path), include_in_vector_store=True)
                print(f"Re-uploaded file {tmp_file_path.name} with ID: {uploaded_file_id}")

                response_result = await agency.get_response(question)
                assert response_result is not None
                print(f"Response (retry): {response_result.final_output}")

                final_output_lower = response_result.final_output.lower()
                hobbit_found = any(term in final_output_lower for term in ["hobbit", "the hobbit", "j.r.r. tolkien"])
                assert hobbit_found, f"Expected 'hobbit' terms not found in retry: {response_result.final_output}"
            else:
                raise

    finally:
        await _cleanup_file_search_resources(real_openai_client, folder_path, file_search_agent)

async def test_vector_store_cleanup_on_init(real_openai_client: AsyncOpenAI, tmp_path: Path):
    """Agent initialization synchronizes vector store with local files, removing orphaned files from VS and OpenAI."""
    source_file = Path("tests/data/files/favorite_books.txt")
    assert source_file.exists(), f"Test file not found at {source_file}"

    # Create temp folder with two files
    files_dir = tmp_path / "cleanup_files"
    files_dir.mkdir(exist_ok=True)
    file_a = files_dir / "books_a.txt"
    file_b = files_dir / "books_b.txt"
    file_a.write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")
    file_b.write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")

    agent_kwargs = {
        "name": "CleanupAgent",
        "instructions": "Use FileSearch to answer from documents.",
        "files_folder": str(files_dir),
        "include_search_results": True,
        "model": "gpt-5.4-mini",
        "model_settings": ModelSettings(tool_choice="file_search"),
        "tool_use_behavior": "stop_on_first_tool",
    }

    # First init: uploads both files and creates VS
    agent1 = Agent(**agent_kwargs)
    agent1._openai_client = real_openai_client
    Agency(agent1, user_context=None)
    await _wait_for_vector_store(real_openai_client, agent1)

    # Find VS folder and collect uploaded ids
    candidates = list(files_dir.parent.glob(f"{files_dir.name}_vs_*"))
    folder_path = candidates[0] if candidates else None
    assert folder_path and folder_path.exists(), "No vector store folder found"

    uploaded_ids = []
    for f in folder_path.glob("*"):
        if f.is_file():
            fid = agent1.file_manager.get_id_from_file(f)
            if fid:
                uploaded_ids.append(fid)
    assert len(uploaded_ids) == 2, f"Expected 2 uploaded files, got {len(uploaded_ids)}"

    # Remove one local file
    local_files = [p for p in folder_path.glob("*") if p.is_file()]
    assert len(local_files) >= 2
    removed_local = local_files[0]
    removed_id = agent1.file_manager.get_id_from_file(removed_local)
    os.remove(removed_local)

    # Re-init: should detach removed from VS and delete OpenAI file object
    agent2 = Agent(**agent_kwargs)
    agent2._openai_client = real_openai_client
    Agency(agent2, user_context=None)
    await _wait_for_vector_store(real_openai_client, agent2)

    vs_id = agent2._associated_vector_store_id
    assert isinstance(vs_id, str) and vs_id

    # Vector Store file listings are eventually consistent; do not assert immediate absence here.
    await _assert_openai_file_absent(real_openai_client, removed_id)

    # Cleanup
    try:
        await _cleanup_file_search_resources(real_openai_client, folder_path, agent2)
    except Exception as e:
        print(f"Cleanup failed: {e}")

async def test_file_reupload_on_mtime_update(real_openai_client: AsyncOpenAI, tmp_path: Path):
    """Modifying local file triggers re-upload with a new file_id and VS update."""
    source_file = Path("tests/data/files/favorite_books.txt")
    assert source_file.exists(), f"Test file not found at {source_file}"

    # Create temp folder and copy file
    files_dir = tmp_path / "reupload_files"
    files_dir.mkdir(exist_ok=True)
    local_file = files_dir / "favorite_books.txt"
    shutil.copy2(source_file, local_file)

    agent_kwargs = {
        "name": "ReuploadAgent",
        "instructions": "Use FileSearch to answer from documents.",
        "files_folder": str(files_dir),
        "include_search_results": True,
        "model": "gpt-5.4-mini",
        "model_settings": ModelSettings(tool_choice="file_search"),
        "tool_use_behavior": "stop_on_first_tool",
    }

    # First init: upload original file
    agent1 = Agent(**agent_kwargs)
    agent1._openai_client = real_openai_client
    Agency(agent1, user_context=None)
    await _wait_for_vector_store(real_openai_client, agent1)

    # Locate vector store folder and uploaded file id
    candidates = list(files_dir.parent.glob(f"{files_dir.name}_vs_*"))
    folder_path = candidates[0] if candidates else None
    assert folder_path and folder_path.exists(), "No vector store folder found"

    vs_files_local = [p for p in folder_path.glob("*") if p.is_file()]
    assert len(vs_files_local) == 1
    uploaded_path = vs_files_local[0]
    old_id = agent1.file_manager.get_id_from_file(uploaded_path)
    assert isinstance(old_id, str) and old_id

    # Ensure mtime > created_at by waiting and modifying the file
    time.sleep(2)
    with open(uploaded_path, "a", encoding="utf-8") as f:
        f.write("\nReupload test line.")
    # Bump mtime explicitly to avoid rounding issues
    try:
        st = uploaded_path.stat()
        os.utime(uploaded_path, (st.st_atime, st.st_mtime + 2))
    except Exception:
        pass

    # Re-init agent: should detect newer mtime and re-upload
    agent2 = Agent(**agent_kwargs)
    agent2._openai_client = real_openai_client
    Agency(agent2, user_context=None)
    await _wait_for_vector_store(real_openai_client, agent2)

    vs_id = agent2._associated_vector_store_id
    assert isinstance(vs_id, str) and vs_id

    # Verify that reupload occurred (new file was uploaded)
    # Note: We don't test that the old file was removed from the vector store,
    # as vector store cleanup may be eventually consistent
    vs_files = await real_openai_client.vector_stores.files.list(vector_store_id=vs_id, filter="completed")
    new_ids = {getattr(f, "file_id", None) or getattr(f, "id", None) for f in vs_files.data}
    assert len(new_ids) >= 1, f"Expected at least 1 file in vector store, got {len(new_ids)}"
    # Verify that a new file ID exists (reupload occurred) - either old file is gone or we have multiple files
    assert old_id not in new_ids or len(new_ids) > 1, (
        f"Reupload should create new file, but only found old_id {old_id} in {new_ids}"
    )

    # Cleanup
    try:
        await _cleanup_file_search_resources(real_openai_client, folder_path, agent2)
    except Exception as e:
        print(f"Cleanup failed: {e}")


# --- tests/integration/files/test_vector_store_citation_extraction.py ---

async def test_vector_store_citation_extraction():
    """
    Test that FileSearch tool properly returns citations when include_search_results=True
    is set on an agent with files_folder configuration.

    This tests the vector store citation pathway, not direct file attachment citations.
    """

    # Use existing test data that's known to work with vector stores
    import shutil

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy the existing favorite_books.txt file to our temp directory
        source_file = Path(__file__).resolve().parents[2] / "data" / "files" / "favorite_books.txt"
        test_file = temp_path / "favorite_books.txt"
        shutil.copy2(source_file, test_file)

        # Create agent with FileSearch capability and citations enabled
        search_agent = Agent(
            name="VectorSearchAgent",
            instructions=(
                "You are a research assistant that searches documents for specific information "
                "using your FileSearch tool."
            ),
            files_folder=str(temp_path),
            include_search_results=True,
            model="gpt-5.4-mini",
            model_settings=ModelSettings(tool_choice="file_search"),
            tool_use_behavior="stop_on_first_tool",
        )

        # Create agency
        agency = Agency(
            search_agent,
            shared_instructions="Test vector store citation functionality.",
        )

        # Ensure vector store is fully processed
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        vs_id = getattr(search_agent, "_associated_vector_store_id", None)

        if vs_id:
            for _ in range(60):
                vs = await client.vector_stores.retrieve(vs_id)
                if getattr(vs, "status", "") == "completed":
                    break
                if getattr(vs, "status", "") == "failed":
                    raise RuntimeError(f"Vector store processing failed: {vs}")
                await asyncio.sleep(1)
        else:
            await asyncio.sleep(5)

        # Test search query for the favorite books content
        test_question = "Use FileSearch to search for books by J.R.R. Tolkien. Report what you find."
        from agents import RunConfig

        response = await agency.get_response(
            test_question, run_config=RunConfig(model_settings=ModelSettings(tool_choice="file_search"))
        )

        # Verify the response contains the expected answer
        assert "Hobbit" in response.final_output or "Tolkien" in response.final_output, (
            f"Expected answer not found in: {response.final_output}"
        )

        # Check that FileSearch tool was called (this verifies the include_search_results setup)
        file_search_calls = [
            item
            for item in response.new_items
            if hasattr(item, "raw_item") and hasattr(item.raw_item, "type") and item.raw_item.type == "file_search_call"
        ]

        all_msgs = agency.thread_manager.get_all_messages()
        system_msgs = [m for m in all_msgs if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert "file_search_preservation" in system_msgs[-1].get("message_origin", "")

        assert len(file_search_calls) > 0, "FileSearch tool was not called despite tool_choice='file_search'"

        file_search_call = file_search_calls[0]
        assert hasattr(file_search_call.raw_item, "id"), "FileSearch call missing ID"
        assert hasattr(file_search_call.raw_item, "queries"), "FileSearch call missing queries"

        print(f"✅ Vector store FileSearch test passed - Tool called with ID: {file_search_call.raw_item.id}")
        print(f"   Queries: {getattr(file_search_call.raw_item, 'queries', [])}")
        print(f"   Status: {getattr(file_search_call.raw_item, 'status', 'unknown')}")

        # Extract citations with a short retry loop to ensure stability
        from agents import RunConfig

        citations = []
        for _ in range(3):
            citations = extract_vector_store_citations(response)
            if citations:
                break
            # Retry by re-asking the original question
            response = await agency.get_response(
                test_question,
                run_config=RunConfig(model_settings=ModelSettings(tool_choice="file_search")),
            )

        assert len(citations) > 0, "Expected FileSearch citations but none were returned"

        citation = citations[0]
        assert "file_id" in citation, "Citation missing file_id"
        assert "text" in citation, "Citation missing text"
        assert "tool_call_id" in citation, "Citation missing tool_call_id"
        assert citation["file_id"].startswith("file-"), f"Invalid file_id format: {citation['file_id']}"
        assert len(citation["text"]) > 0, "Citation text is empty"


# --- tests/integration/tools/test_responses_api_tools.py ---

async def test_tool_cycle_with_sdk_and_responses_api():
    """
    Integration test verifying that the openai-agents SDK properly handles tool cycles
    with the OpenAI Responses API.

    This test ensures that:
    1. Tools can be called successfully using the SDK
    2. Tool outputs are processed correctly
    3. The agent can provide a final response incorporating tool results
    4. The SDK's tool use behavior works as expected with the Responses API
    """

    # Explicitly create an AsyncOpenAI client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    forced_responses_model = OpenAIResponsesModel(model="gpt-5.4-mini", openai_client=client)

    agent = Agent(
        name="SDK Responses API Test Agent",
        instructions="You are an agent that uses tools. When asked to process text, use the simple_processor_tool.",
        tools=[simple_processor_tool],
        tool_use_behavior="run_llm_again",  # Send tool output back to LLM for final response
        model=forced_responses_model,
    )

    logger.info("Testing tool cycle with SDK Agent using OpenAIResponsesModel")

    # Test that the agent can successfully use tools and provide a response
    result = await Runner.run(agent, input="Please process the text 'hello world' using your tool.")

    # Verify the run completed successfully
    assert result is not None, "Runner.run should return a result"
    assert result.final_output is not None, "Result should have a final output"

    logger.info(f"Final output: {result.final_output}")
    logger.info(f"Number of new items: {len(result.new_items) if result.new_items else 0}")

    # Verify that the tool was actually called and the output was processed
    final_output_str = str(result.final_output).lower()

    # The tool should have processed "hello world" to "Processed: hello world"
    assert "processed" in final_output_str, f"Tool output should be processed. Got: {result.final_output}"
    assert "hello world" in final_output_str, f"Original input should be referenced. Got: {result.final_output}"

    # Verify that we have the expected items in the result
    assert result.new_items is not None and len(result.new_items) > 0, "Should have new items from the run"

    # Debug: Print the actual items to understand the structure
    logger.info("Actual items returned:")
    for i, item in enumerate(result.new_items):
        logger.info(f"  Item {i + 1}: {type(item).__name__} - {item}")
        if hasattr(item, "raw_item"):
            logger.info(f"    Raw item type: {type(item.raw_item)}")

    # Check that we have meaningful output from the tool
    # The agent should have used the tool and incorporated the result
    assert "processed" in final_output_str, f"Tool should have been used to process text. Got: {result.final_output}"

    # Verify the tool was actually executed by checking for tool-related items
    # Look for any tool-related items (calls or outputs)
    tool_related_items = [
        item
        for item in result.new_items
        if hasattr(item, "raw_item")
        and ("function" in str(type(item.raw_item)).lower() or "tool" in str(type(item.raw_item)).lower())
    ]

    logger.info(f"Found {len(tool_related_items)} tool-related items")

    # The test passes if the tool was used (evidenced by the output) and we got a response
    # The exact structure of items may vary by SDK version, but the functionality should work
    assert len(result.new_items) > 0, "Should have generated some items during execution"

    logger.info("✅ SDK tool cycle with Responses API working correctly")

async def test_hosted_tool_output_preservation_multi_turn():
    """
    Integration test for hosted tool output preservation in multi-turn conversations.

    This test verifies that hosted tools (FileSearch, WebSearch) results are properly
    preserved in conversation history for future reference.

    Test scenario:
    1. First turn: Agent uses FileSearch tool but doesn't reveal specific details
    2. Second turn: Ask agent to provide exact tool output from previous search

    This ensures hosted tool results are preserved and accessible in subsequent turns,
    solving the bug where they were previously lost between conversations.
    """

    # Create test data with specific content for numeric validation
    with tempfile.TemporaryDirectory(prefix="hosted_tool_test_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        test_file = temp_dir / "company_data.txt"
        test_file.write_text("""
COMPANY FINANCIAL REPORT

Revenue Information:
- Q4 Revenue: $7,892,345.67
- Q3 Revenue: $6,234,567.89
- Operating Costs: $2,345,678.90
- Net Profit: $4,123,456.78

Employee Data:
- Total Employees: 1,234
- New Hires: 567
- Contractors: 89

Product Sales:
- Product Alpha: 12,345 units
- Product Beta: 6,789 units
- Product Gamma: 2,345 units
""")

        # Create Agency Swarm agent with FileSearch via files_folder
        agent = AgencySwarmAgent(
            name="DataSearchAgent",
            instructions=(
                "You are a data search assistant. You MUST use the FileSearch tool to find information. "
                "Always search files before answering. Be concise in your initial responses."
            ),
            model="gpt-5.4-mini",
            model_settings=ModelSettings(tool_choice="file_search"),
            files_folder=str(temp_dir),
            include_search_results=True,
        )

        # Create an agency with the agent
        agency = Agency(agent)

        # Wait for file processing and vector store indexing (active polling for stability)
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        vs_id = getattr(agent, "_associated_vector_store_id", None)
        if vs_id:
            for _ in range(60):  # up to 60 seconds
                vs = await client.vector_stores.retrieve(vs_id)
                if getattr(vs, "status", "") == "completed":
                    break
                if getattr(vs, "status", "") == "failed":
                    raise RuntimeError(f"Vector store processing failed: {vs}")
                await asyncio.sleep(1)
        else:
            # fallback to a short delay if no id is exposed
            await asyncio.sleep(5)

        # TURN 1: Agent searches but gives summary only
        logger.info("=== TURN 1: Agent searches with FileSearch ===")

        from agents import RunConfig

        result1 = await agency.get_response(
            message=(
                "Use FileSearch to search the company data for financial information and employee data. "
                "Just confirm you found it, don't give me the specific numbers yet."
            ),
            run_config=RunConfig(model_settings=ModelSettings(tool_choice="file_search")),
        )

        assert result1 is not None
        logger.info(f"Turn 1 result: {result1.final_output}")

        # Get the conversation history from the agency's thread manager
        history_after_turn1 = agency.thread_manager._store.messages

        logger.info(f"=== CONVERSATION HISTORY AFTER TURN 1 ({len(history_after_turn1)} items) ===")
        hosted_tool_outputs_found = 0
        preservation_items = []

        for i, item in enumerate(history_after_turn1):
            item_type = item.get("type", f"role={item.get('role')}")
            logger.info(f"Item {i + 1}: {item_type}")

            # Look for hosted tool search results messages
            if item.get("role") == "system" and "[SEARCH_RESULTS]" in str(item.get("content", "")):
                hosted_tool_outputs_found += 1
                preservation_items.append(item)
                logger.info(f"  Found search results message: {str(item.get('content', ''))}...")

        logger.info(f"Found {hosted_tool_outputs_found} hosted tool preservation items")

        # TURN 2: Ask for exact tool output
        logger.info("=== TURN 2: Requesting exact tool output ===")

        logger.info(f"History at turn 2: {agency.thread_manager._store.messages}")

        result2 = await agency.get_response(
            message=(
                "Now provide me the exact file search results that you found in the previous tool call. "
                "Do not use the tool again. I'm looking for Q3 and Q4 revenue, operating costs, "
                "and total employee count."
            )
        )

        assert result2 is not None
        logger.info(f"Turn 2 result: {result2.final_output}")

        # Verify agent can access specific data from previous tool call
        response_text = str(result2.final_output)

        # Look for specific numbers that should only come from file search results
        has_q4_revenue = "7,892,345.67" in response_text or "7892345.67" in response_text
        has_q3_revenue = "6,234,567.89" in response_text or "6234567.89" in response_text
        has_operating_costs = "2,345,678.90" in response_text or "2345678.90" in response_text
        has_employees = "1,234" in response_text or "1234" in response_text

        logger.info(f"Agent can access Q4 revenue (7,892,345.67): {has_q4_revenue}")
        logger.info(f"Agent can access Q3 revenue (6,234,567.89): {has_q3_revenue}")
        logger.info(f"Agent can access operating costs (2,345,678.90): {has_operating_costs}")
        logger.info(f"Agent can access employee count (1,234): {has_employees}")

        # TEST ASSERTIONS

        # 1. Verify that hosted tool outputs are preserved in conversation history
        assert hosted_tool_outputs_found > 0, (
            "No hosted tool output preservation found in conversation history. "
            "Hosted tool results should be preserved for multi-turn access."
        )

        # 2. Verify that agent can access specific data from previous hosted tool calls
        data_access_score = sum([has_q4_revenue, has_q3_revenue, has_operating_costs, has_employees])
        assert data_access_score >= 2, (
            f"Agent cannot access specific data from previous hosted tool calls. "
            f"Only found {data_access_score}/4 specific data points in response: {response_text}"
        )

        logger.info("✅ Hosted tool output preservation test completed successfully")


# --- tests/test_agent_modules/test_agui_adapter.py ---

def test_openai_events_emit_message_lifecycle():
    adapter = AguiAdapter()
    run_id = "run-1"

    message = ResponseOutputMessage(
        id="m-1",
        content=[ResponseOutputText(annotations=[], text="Hi", type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )

    start_event = make_raw_event(
        ResponseOutputItemAddedEvent(
            item=message,
            output_index=0,
            sequence_number=1,
            type="response.output_item.added",
        )
    )

    delta_event = make_raw_event(
        ResponseTextDeltaEvent(
            content_index=0,
            delta="Hi",
            item_id="m-1",
            logprobs=[],
            output_index=0,
            sequence_number=2,
            type="response.output_text.delta",
        )
    )
    done_event = make_raw_event(
        ResponseOutputItemDoneEvent(
            item=message,
            output_index=0,
            sequence_number=3,
            type="response.output_item.done",
        )
    )

    start = adapter.openai_to_agui_events(start_event, run_id=run_id)
    delta = adapter.openai_to_agui_events(delta_event, run_id=run_id)
    done = adapter.openai_to_agui_events(done_event, run_id=run_id)

    assert isinstance(start, TextMessageStartEvent)
    assert isinstance(delta, TextMessageContentEvent)
    assert isinstance(done, TextMessageEndEvent)
    assert delta.message_id == "m-1"

def test_openai_events_track_tool_calls_and_arguments():
    adapter = AguiAdapter()
    run_id = "run-2"
    raw_tool = ResponseFunctionToolCall(
        arguments="{}",
        call_id="call-1",
        name="search",
        type="function_call",
        id="item-1",
        status="in_progress",
    )

    adapter.openai_to_agui_events(
        make_raw_event(
            ResponseOutputItemAddedEvent(
                item=raw_tool,
                output_index=0,
                sequence_number=1,
                type="response.output_item.added",
            )
        ),
        run_id=run_id,
    )
    args_event = adapter.openai_to_agui_events(
        make_raw_event(
            ResponseFunctionCallArgumentsDeltaEvent(
                item_id="item-1",
                delta='{"q": "',
                output_index=0,
                sequence_number=2,
                type="response.function_call_arguments.delta",
            )
        ),
        run_id=run_id,
    )
    done_events = adapter.openai_to_agui_events(
        make_raw_event(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=ResponseFunctionToolCall(
                    arguments='{"q": "weather"}',
                    call_id="call-1",
                    name="search",
                    type="function_call",
                    id="item-1",
                    status="completed",
                ),
                output_index=0,
                sequence_number=3,
            )
        ),
        run_id=run_id,
    )

    assert isinstance(args_event, ToolCallArgsEvent)
    assert args_event.tool_call_id == "call-1"
    assert isinstance(done_events, list)
    assert isinstance(done_events[0], ToolCallEndEvent)
    assert isinstance(done_events[1], MessagesSnapshotEvent)

def test_openai_typed_events_emit_message_lifecycle():
    adapter = AguiAdapter()
    run_id = "typed-run"

    message = ResponseOutputMessage(
        id="msg-typed",
        content=[ResponseOutputText(annotations=[], text="Hello world", type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )

    start_event = ResponseOutputItemAddedEvent(
        item=message,
        output_index=0,
        sequence_number=1,
        type="response.output_item.added",
    )
    delta_event = ResponseTextDeltaEvent(
        content_index=0,
        delta="!",
        item_id="msg-typed",
        logprobs=[Logprob(token="!", logprob=0.0, top_logprobs=[])],
        output_index=0,
        sequence_number=2,
        type="response.output_text.delta",
    )
    done_event = ResponseOutputItemDoneEvent(
        item=message,
        output_index=0,
        sequence_number=3,
        type="response.output_item.done",
    )

    start = adapter.openai_to_agui_events(make_raw_event(start_event), run_id=run_id)
    delta = adapter.openai_to_agui_events(make_raw_event(delta_event), run_id=run_id)
    done = adapter.openai_to_agui_events(make_raw_event(done_event), run_id=run_id)

    assert isinstance(start, TextMessageStartEvent)
    assert isinstance(delta, TextMessageContentEvent)
    assert isinstance(done, TextMessageEndEvent)
    assert delta.message_id == "msg-typed"

def test_openai_events_ignore_message_without_id():
    adapter = AguiAdapter()
    event = make_raw_event(
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="message", role="assistant", id=None),
        )
    )

    result = adapter.openai_to_agui_events(event, run_id="missing-message")

    assert isinstance(result, RawEvent)
    assert result.type == EventType.RAW
    assert result.event["data"]["type"] == "response.output_item.added"

def test_openai_events_ignore_tool_call_without_call_id():
    adapter = AguiAdapter()
    run_id = "missing-tool"
    tool = SimpleNamespace(type="function_call", id="item-99", call_id=None, name="search", arguments="{}")

    adapter.openai_to_agui_events(
        make_raw_event(SimpleNamespace(type="response.output_item.added", item=tool)),
        run_id=run_id,
    )
    args_event = adapter.openai_to_agui_events(
        make_raw_event(SimpleNamespace(type="response.function_call_arguments.delta", item_id="item-99", delta="{}")),
        run_id=run_id,
    )

    assert isinstance(args_event, RawEvent)
    assert args_event.type == EventType.RAW
    assert args_event.event["data"]["type"] == "response.function_call_arguments.delta"

def test_openai_events_ignore_text_delta_without_item_id():
    adapter = AguiAdapter()
    event = make_raw_event(SimpleNamespace(type="response.output_text.delta", item_id=None, delta="Hi"))

    result = adapter.openai_to_agui_events(event, run_id="missing-delta-id")

    assert isinstance(result, RawEvent)
    assert result.type == EventType.RAW
    assert result.event["data"]["type"] == "response.output_text.delta"

def test_openai_events_ignore_tool_done_without_call_id():
    adapter = AguiAdapter()
    raw_item = SimpleNamespace(type="function_call", id="item-9", call_id=None, name="search", arguments="{}")
    event = make_raw_event(SimpleNamespace(type="response.output_item.done", item=raw_item))

    result = adapter.openai_to_agui_events(event, run_id="tool-done-missing")

    assert isinstance(result, RawEvent)
    assert result.type == EventType.RAW
    assert result.event["data"]["type"] == "response.output_item.done"

def test_run_item_stream_events_emit_snapshots():
    adapter = AguiAdapter()
    run_id = "run-3"
    output_content = ResponseOutputText(annotations=[], text="Answer", type="output_text")
    raw_item = ResponseOutputMessage(
        id="msg-1",
        content=[output_content],
        role="assistant",
        status="completed",
        type="message",
    )
    item = SimpleNamespace(raw_item=raw_item)

    events = adapter.openai_to_agui_events(make_stream_event("message_output_created", item), run_id=run_id)

    assert isinstance(events, list)
    assert all(isinstance(e, MessagesSnapshotEvent | CustomEvent) for e in events)
    assert any(isinstance(e, MessagesSnapshotEvent) for e in events)

def test_run_item_stream_with_annotations_returns_custom_event():
    adapter = AguiAdapter()
    run_id = "annotated"
    annotation = AnnotationFileCitation(file_id="file-annot", filename="doc.pdf", index=1, type="file_citation")
    output_content = ResponseOutputText(annotations=[annotation], text="Answer", type="output_text")
    raw_item = ResponseOutputMessage(
        id="msg-annot",
        content=[output_content],
        role="assistant",
        status="completed",
        type="message",
    )
    item = SimpleNamespace(raw_item=raw_item)

    events = adapter.openai_to_agui_events(make_stream_event("message_output_created", item), run_id=run_id)

    assert isinstance(events, list)
    assert any(isinstance(e, CustomEvent) for e in events)
    custom = next(e for e in events if isinstance(e, CustomEvent))
    assert custom.value["annotations"] == [annotation.model_dump()]

def test_run_item_stream_ignores_message_without_text():
    adapter = AguiAdapter()
    run_id = "missing-text"
    output_content = SimpleNamespace(text=None, annotations=None)
    raw_item = SimpleNamespace(id="msg-empty", content=[output_content])
    item = SimpleNamespace(raw_item=raw_item)

    result = adapter.openai_to_agui_events(make_stream_event("message_output_created", item), run_id=run_id)

    assert isinstance(result, RawEvent)
    assert result.type == EventType.RAW
    assert result.event["name"] == "message_output_created"

def test_tool_output_stream_event_converts_to_tool_message():
    adapter = AguiAdapter()
    run_id = "run-4"
    item = SimpleNamespace(raw_item={"call_id": "call-7"}, call_id="call-7", output="done")

    event = adapter.openai_to_agui_events(make_stream_event("tool_output", item), run_id=run_id)

    assert isinstance(event, MessagesSnapshotEvent)
    message = event.messages[0]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-7"
    assert message.content == "done"

def test_tool_output_without_call_id_is_ignored():
    adapter = AguiAdapter()
    item = SimpleNamespace(raw_item={}, call_id=None, output="done")

    result = adapter.openai_to_agui_events(make_stream_event("tool_output", item), run_id="tool-missing")

    assert isinstance(result, RawEvent)
    assert result.type == EventType.RAW
    assert result.event["name"] == "tool_output"

def test_run_item_stream_unknown_event_is_returned_as_raw_event():
    adapter = AguiAdapter()
    run_id = "unknown-stream"
    unknown_event = make_stream_event("unhandled_event", None)

    result = adapter.openai_to_agui_events(unknown_event, run_id=run_id)

    assert isinstance(result, RawEvent)
    assert result.type == EventType.RAW
    assert result.event["name"] == "unhandled_event"
    assert result.event["type"] == "run_item_stream_event"

def test_tool_meta_handles_non_function_tools():
    adapter = AguiAdapter()

    typed_file_search = ResponseFileSearchToolCall(
        id="file-1",
        queries=["foo"],
        status="completed",
        type="file_search_call",
        results=[FileSearchResult(file_id="doc", text="bar")],
    )
    typed_code_interpreter = ResponseCodeInterpreterToolCall(
        code="print(42)",
        container_id="cont",
        id="ci-7",
        outputs=[{"type": "logs", "logs": "42"}],
        type="code_interpreter_call",
        status="completed",
    )

    @dataclasses.dataclass
    class LegacyFileSearchCall:
        type: str
        id: str
        queries: list[str]
        results: list[dict]

    @dataclasses.dataclass
    class LegacyCodeInterpreterCall:
        type: str
        id: str
        code: str
        container_id: str
        outputs: list[dict]

    file_search = LegacyFileSearchCall(
        type="file_search_call",
        id=typed_file_search.id,
        queries=typed_file_search.queries or [],
        results=json.loads(typed_file_search.model_dump_json())["results"],
    )
    code_interpreter = LegacyCodeInterpreterCall(
        type="code_interpreter_call",
        id=typed_code_interpreter.id,
        code=typed_code_interpreter.code or "",
        container_id=typed_code_interpreter.container_id,
        outputs=[{"type": "logs", "logs": "42"}],
    )

    file_meta = adapter._tool_meta(file_search)
    code_meta = adapter._tool_meta(code_interpreter)

    assert file_meta[0] == "file-1"
    assert file_meta[1] == "FileSearchTool"
    assert json.loads(file_meta[2])["queries"] == ["foo"]

    assert code_meta[0] == "ci-7"
    assert code_meta[1] == "CodeInterpreterTool"
    assert json.loads(code_meta[2])["code"] == "print(42)"


# --- tests/test_agent_modules/test_agui_adapter_fake_id.py ---

    def test_tool_call_with_real_id_still_works(self):
        """Tool calls with real item IDs should continue to work."""
        adapter = AguiAdapter()
        run_id = "real-id-run"

        tool = ResponseFunctionToolCall(
            arguments="{}",
            call_id="call_real",
            name="real_tool",
            type="function_call",
            id="real_item_id",  # Real ID, not placeholder
            status="in_progress",
        )

        adapter.openai_to_agui_events(
            make_raw_event(
                ResponseOutputItemAddedEvent(
                    item=tool,
                    output_index=0,
                    sequence_number=1,
                    type="response.output_item.added",
                )
            ),
            run_id=run_id,
        )

        delta_event = adapter.openai_to_agui_events(
            make_raw_event(
                ResponseFunctionCallArgumentsDeltaEvent(
                    item_id="real_item_id",
                    delta='{"key": "value"}',
                    output_index=0,
                    sequence_number=2,
                    type="response.function_call_arguments.delta",
                )
            ),
            run_id=run_id,
        )

        assert isinstance(delta_event, ToolCallArgsEvent)
        assert delta_event.tool_call_id == "call_real"


# --- tests/test_agent_modules/test_conversation_starters_cache.py ---

def test_starter_cache_fingerprint_changes_for_guardrails_runtime_tools_and_handoffs() -> None:
    agent_with_guardrails = Agent(
        name="GuardrailAgent",
        instructions="You are helpful.",
        model="gpt-5.4-mini",
        input_guardrails=[require_support_prefix],
        output_guardrails=[block_emails],
    )
    agent_without_guardrails = Agent(
        name="BaselineAgent",
        instructions="You are helpful.",
        model="gpt-5.4-mini",
        input_guardrails=[],
        output_guardrails=[],
    )
    assert compute_starter_cache_fingerprint(agent_with_guardrails) != compute_starter_cache_fingerprint(
        agent_without_guardrails
    )

    sender = Agent(
        name="SenderAgent",
        instructions="You are helpful.",
        model="gpt-5.4-mini",
    )
    recipient = Agent(
        name="RecipientAgent",
        instructions="You are helpful.",
        model="gpt-5.4-mini",
    )
    runtime_state = AgentRuntimeState()
    fingerprint_before = compute_starter_cache_fingerprint(sender, runtime_state=runtime_state)
    sender.register_subagent(recipient, runtime_state=runtime_state)
    fingerprint_after = compute_starter_cache_fingerprint(sender, runtime_state=runtime_state)
    assert fingerprint_before != fingerprint_after

    handoff_sender = Agent(
        name="HandoffSender",
        instructions="You are helpful.",
        model="gpt-5.4-mini",
    )
    handoff_recipient = Agent(
        name="HandoffRecipient",
        instructions="You are helpful.",
        model="gpt-5.4-mini",
    )
    handoff_runtime = AgentRuntimeState()
    handoff_before = compute_starter_cache_fingerprint(handoff_sender, runtime_state=handoff_runtime)
    handoff_runtime.handoffs.append(Handoff().create_handoff(handoff_recipient))
    handoff_after = compute_starter_cache_fingerprint(handoff_sender, runtime_state=handoff_runtime)
    assert handoff_before != handoff_after


# --- tests/test_agent_modules/test_hosted_tool_results.py ---

async def test_web_search_results_have_metadata():
    """Verify web search results are returned as user messages with metadata."""
    agent = Agent(name="MetaAgent", instructions="Test")

    web_call = ResponseFunctionWebSearch(
        id="1",
        action=ActionSearch(query="hello", type="search"),
        status="completed",
        type="web_search_call",
    )

    assistant_msg = ResponseOutputMessage(
        id="m1",
        content=[ResponseOutputText(annotations=[], text="result", type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )

    run_items = [
        ToolCallItem(agent, web_call),
        MessageOutputItem(agent, assistant_msg),
    ]

    results = MessageFormatter.extract_hosted_tool_results(
        agent,
        run_items,
        caller_agent="Researcher",
    )

    assert results, "Expected hosted tool result"
    result = results[0]
    assert result.get("agent") == agent.name
    assert result.get("callerAgent") == "Researcher"
    assert "WEB_SEARCH_RESULTS" in result.get("content", "")

def test_web_search_results_deduplicated():
    """Only one synthetic result should be created for multiple assistant messages."""
    agent = Agent(name="MetaAgent", instructions="Test")

    web_call = ResponseFunctionWebSearch(
        id="1",
        action=ActionSearch(query="hello", type="search"),
        status="completed",
        type="web_search_call",
    )

    assistant_msgs = [
        ResponseOutputMessage(
            id="m1",
            content=[ResponseOutputText(annotations=[], text="result1", type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        ),
        ResponseOutputMessage(
            id="m2",
            content=[ResponseOutputText(annotations=[], text="result2", type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        ),
    ]

    run_items = [ToolCallItem(agent, web_call)] + [MessageOutputItem(agent, m) for m in assistant_msgs]

    results = MessageFormatter.extract_hosted_tool_results(agent, run_items)  # type: ignore[arg-type]
    assert len(results) == 1
    assert "result1" in results[0]["content"]
    assert "result2" not in results[0]["content"]

def test_multiple_web_searches_get_distinct_results():
    """Each web search should get its own corresponding assistant message content."""
    agent = Agent(name="SearchAgent", instructions="Test")

    # First web search and its result
    web_call1 = ResponseFunctionWebSearch(
        id="search_1",
        action=ActionSearch(query="python", type="search"),
        status="completed",
        type="web_search_call",
    )
    assistant_msg1 = ResponseOutputMessage(
        id="msg_1",
        content=[ResponseOutputText(annotations=[], text="Python results", type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )

    # Second web search and its result
    web_call2 = ResponseFunctionWebSearch(
        id="search_2",
        action=ActionSearch(query="javascript", type="search"),
        status="completed",
        type="web_search_call",
    )
    assistant_msg2 = ResponseOutputMessage(
        id="msg_2",
        content=[ResponseOutputText(annotations=[], text="JavaScript results", type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )

    # Build run items in order: search1, msg1, search2, msg2
    run_items = [
        ToolCallItem(agent, web_call1),
        MessageOutputItem(agent, assistant_msg1),
        ToolCallItem(agent, web_call2),
        MessageOutputItem(agent, assistant_msg2),
    ]

    results = MessageFormatter.extract_hosted_tool_results(agent, run_items)

    # Should create two synthetic results
    assert len(results) == 2, "Expected two results for two web searches"

    # First result should have Python content
    assert "search_1" in results[0]["content"]
    assert "Python results" in results[0]["content"]
    assert "JavaScript results" not in results[0]["content"]

    # Second result should have JavaScript content
    assert "search_2" in results[1]["content"]
    assert "JavaScript results" in results[1]["content"]
    assert "Python results" not in results[1]["content"]

def test_file_search_results_only_persist_for_executing_agent():
    """Ensure hosted tool preservation is only emitted by the agent that ran the tool."""
    ceo = Agent(name="CEO", instructions="Test")
    worker = Agent(name="Worker", instructions="Test")

    tool_call = ResponseFileSearchToolCall(
        id="fs_unique",
        queries=["favorite books"],
        status="completed",
        type="file_search_call",
        results=[
            FileSearchResult(
                file_id="file-1",
                filename="favorite_books.txt",
                score=0.9,
                text="Books list",
            )
        ],
    )

    hosted_run_items = [ToolCallItem(ceo, tool_call)]

    ceo_results = MessageFormatter.extract_hosted_tool_results(
        ceo,
        hosted_run_items,
        caller_agent="Worker",
    )
    assert ceo_results, "Executing agent should persist hosted tool results"

    worker_results = MessageFormatter.extract_hosted_tool_results(
        worker,
        hosted_run_items,
        caller_agent="Worker",
    )
    assert worker_results == [], "Non-executing agent must not duplicate hosted tool preservation"

def test_web_search_results_only_persist_for_executing_agent():
    """Ensure web search preservation is written only by the executing agent."""
    ceo = Agent(name="CEO", instructions="Test")
    worker = Agent(name="Worker", instructions="Test")

    web_call = ResponseFunctionWebSearch(
        id="web_unique",
        action=ActionSearch(query="web search", type="search"),
        status="completed",
        type="web_search_call",
    )

    assistant_msg = ResponseOutputMessage(
        id="web_msg",
        content=[ResponseOutputText(annotations=[], text="Search content", type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )

    run_items = [
        ToolCallItem(ceo, web_call),
        MessageOutputItem(ceo, assistant_msg),
    ]

    ceo_results = MessageFormatter.extract_hosted_tool_results(
        ceo,
        run_items,
        caller_agent="Worker",
    )
    assert ceo_results, "Executing agent should persist web search results"
    assert "WEB_SEARCH_RESULTS" in ceo_results[0]["content"]

    worker_results = MessageFormatter.extract_hosted_tool_results(
        worker,
        run_items,
        caller_agent="Worker",
    )
    assert worker_results == [], "Non-executing agent must not duplicate web search preservation"


# --- tests/test_messages_modules/test_message_formatter_history_protocol.py ---

def test_prepare_history_for_runner_allows_compatible_histories() -> None:
    """Compatible history formats should be accepted across supported providers."""
    compatible_cases: list[tuple[dict, callable]] = [
        (
            {
                "role": "user",
                "content": "hello",
                "agent": "AgentA",
                "callerAgent": None,
                "history_protocol": MessageFormatter.HISTORY_PROTOCOL_CHAT_COMPLETIONS,
            },
            _make_responses_agent,
        ),
        (
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "send_message",
                "arguments": "{}",
                "agent": "Coordinator",
                "callerAgent": None,
            },
            _make_chat_agent,
        ),
        (
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "send_message",
                "arguments": "{}",
                "agent": "Coordinator",
                "callerAgent": None,
            },
            lambda name: _make_litellm_agent(name, "openai/gpt-5.4-mini"),
        ),
        (
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "send_message",
                "arguments": "{}",
                "agent": "Coordinator",
                "callerAgent": None,
            },
            lambda name: _make_litellm_agent(name, "anthropic/claude-sonnet-4-20250514"),
        ),
    ]

    for history_item, agent_factory in compatible_cases:
        thread_manager = ThreadManager()
        thread_manager._store.messages = [history_item]
        context = _make_context(thread_manager)
        agent_name = str(history_item.get("agent") or "AgentA")
        agent = agent_factory(agent_name)
        MessageFormatter.prepare_history_for_runner([], agent, None, agency_context=context)

def test_prepare_history_for_runner_stores_responses_protocol_and_strips_runner_metadata() -> None:
    thread_manager = ThreadManager()
    context = _make_context(thread_manager)
    chat_agent = _make_chat_agent("AgentA")
    responses_agent = _make_responses_agent("AgentB")

    first_history = MessageFormatter.prepare_history_for_runner(
        [{"role": "user", "content": "hello"}],
        chat_agent,
        None,
        context,
    )
    MessageFormatter.prepare_history_for_runner([{"role": "user", "content": "second"}], responses_agent, None, context)

    all_messages = thread_manager.get_all_messages()
    assert len(all_messages) == 2
    assert all(msg["history_protocol"] == MessageFormatter.HISTORY_PROTOCOL_RESPONSES for msg in all_messages)
    assert all("history_protocol" not in item for item in first_history)

def test_prepare_history_for_runner_rejects_inferred_protocol_mismatch() -> None:
    thread_manager = ThreadManager()
    thread_manager._store.messages = [
        {
            "role": "tool",
            "content": "ok",
            "tool_call_id": "call-1",
            "agent": "AgentA",
            "callerAgent": None,
        }
    ]

    agent = _make_responses_agent("AgentA")
    context = _make_context(thread_manager)

    with pytest.raises(IncompatibleChatHistoryError):
        MessageFormatter.prepare_history_for_runner([], agent, None, agency_context=context)

def test_prepare_history_for_runner_normalizes_legacy_items_for_responses_protocol() -> None:
    thread_manager = ThreadManager()
    thread_manager._store.messages = [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "send_message",
            "arguments": "{}",
            "tool_calls": [{"id": "legacy-tool-call"}],
            "agent": "Coordinator",
            "callerAgent": None,
        },
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "Agency Swarm"},
            "history_protocol": MessageFormatter.HISTORY_PROTOCOL_CHAT_COMPLETIONS,
            "agent": "Coordinator",
            "callerAgent": None,
        },
        {
            "type": "function_call",
            "id": "call-1",
            "call_id": "call-1",
            "name": "send_message",
            "arguments": "{}",
            "agent": "Coordinator",
            "callerAgent": None,
        },
        {
            "type": "function_call",
            "id": "fc_accepted_by_responses",
            "call_id": "call-2",
            "name": "send_message",
            "arguments": "{}",
            "agent": "Coordinator",
            "callerAgent": None,
        },
    ]

    agent = _make_responses_agent("Coordinator")
    context = _make_context(thread_manager)

    history_for_runner = MessageFormatter.prepare_history_for_runner([], agent, None, agency_context=context)
    assert history_for_runner[0]["type"] == "function_call"
    assert history_for_runner[0]["call_id"] == "call-1"
    assert any(msg.get("type") == "web_search_call" for msg in history_for_runner)

    function_calls = [msg for msg in history_for_runner if msg.get("type") == "function_call" and msg.get("call_id")]
    assert len(function_calls) >= 3
    assert any(msg.get("call_id") == "call-1" and "id" not in msg for msg in function_calls)
    preserved_id_call = next(msg for msg in function_calls if msg.get("call_id") == "call-2")
    assert preserved_id_call.get("id") == "fc_accepted_by_responses"


# --- tests/test_utils_modules/test_model_utils.py ---

def test_get_model_name_from_openai_model_objects() -> None:
    """Model-name extraction should work for both Responses and Chat models."""
    client = AsyncOpenAI(api_key="test")
    cases = [
        OpenAIResponsesModel(model="gpt-5.4-mini", openai_client=client),
        OpenAIChatCompletionsModel(model="gpt-5.4-mini", openai_client=client),
    ]
    for model in cases:
        assert get_model_name(model) == "gpt-5.4-mini"


# --- tests/test_utils_modules/test_usage_tracking.py ---

def test_extract_usage_from_run_result_returns_none_without_run_result() -> None:
    assert extract_usage_from_run_result(None) is None

def test_extract_usage_from_run_result_reads_requests_and_tokens() -> None:
    usage = Usage(
        requests=2,
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_tokens_details=InputTokensDetails(cached_tokens=3),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
    run_result = _make_run_result(usage=usage)

    stats = extract_usage_from_run_result(run_result)
    assert stats == UsageStats(
        request_count=2,
        cached_tokens=3,
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        total_cost=0.0,
        reasoning_tokens=None,
        audio_tokens=None,
    )

def test_extract_usage_from_run_result_extracts_reasoning_and_sums_subagent_reasoning() -> None:
    main_usage = Usage(
        requests=1,
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=5),
    )

    sub_usage = Usage(
        requests=1,
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=7),
    )

    run_result = _make_run_result(usage=main_usage)
    typing.cast(_HasSubAgentResponsesWithModel, run_result)._sub_agent_responses_with_model = [
        ("gpt-5.4-mini", ModelResponse(output=[], usage=sub_usage, response_id=None))
    ]

    stats = extract_usage_from_run_result(run_result)
    assert stats is not None
    assert stats.request_count == 2
    assert stats.input_tokens == 11
    assert stats.output_tokens == 22
    assert stats.total_tokens == 33
    assert stats.reasoning_tokens == 12  # 5 main + 7 sub

def test_calculate_usage_with_cost_per_response_costs_all_token_types() -> None:
    """
    Single per-response costing test that verifies:
    - input token pricing
    - cached input token pricing (via input_tokens_details.cached_tokens)
    - output token pricing
    - reasoning token pricing (via output_tokens_details.reasoning_tokens)
    - dict-based usage (sub-agent) uses that sub-agent's model pricing
    """
    pricing_data = {
        "test/all-tokens-model": {
            "input_cost_per_token": 1.0,
            "cache_read_input_token_cost": 0.1,
            "output_cost_per_token": 2.0,
            "output_cost_per_reasoning_token": 0.01,
        },
        "test/sub-agent-model": {
            "input_cost_per_token": 10.0,
            "cache_read_input_token_cost": 1.0,
            "output_cost_per_token": 20.0,
            "output_cost_per_reasoning_token": 0.5,
        },
    }

    response_usage = Usage(
        requests=1,
        input_tokens=10,
        output_tokens=3,
        total_tokens=13,
        input_tokens_details=InputTokensDetails(cached_tokens=4),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=5),
    )
    response = ModelResponse(output=[], usage=response_usage, response_id=None)
    run_result = _make_run_result(usage=Usage(), raw_responses=[response])
    typing.cast(_HasMainAgentModel, run_result)._main_agent_model = "test/all-tokens-model"

    base = UsageStats(
        request_count=1,
        cached_tokens=0,
        input_tokens=10,
        output_tokens=3,
        total_tokens=13,
        total_cost=0.0,
        reasoning_tokens=None,
        audio_tokens=None,
    )

    with_cost = calculate_usage_with_cost(base, pricing_data=pricing_data, run_result=run_result)

    # Main response:
    # (10 - 4)*1.0 + 4*0.1 + 3*2.0 + 5*0.01 = 6 + 0.4 + 6 + 0.05 = 12.45
    assert with_cost.total_cost == pytest.approx(12.45)

def test_extract_usage_from_run_result_skips_malformed_subagent_entries() -> None:
    usage = Usage(
        requests=1,
        input_tokens=5,
        output_tokens=3,
        total_tokens=8,
        input_tokens_details=InputTokensDetails(cached_tokens=1),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
    run_result = _make_run_result(usage=usage)
    typing.cast(_HasSubAgentResponsesWithModel, run_result)._sub_agent_responses_with_model = [
        ("broken", typing.cast(ModelResponse, object())),
    ]

    stats = extract_usage_from_run_result(run_result)
    assert stats is not None
    assert stats.request_count == 1
    assert stats.total_tokens == 8

def test_extract_usage_from_run_result_returns_none_for_unusable_context_wrapper() -> None:
    run_result = SimpleNamespace(context_wrapper=object())
    assert extract_usage_from_run_result(typing.cast(RunResult, run_result)) is None

def test_calculate_usage_with_cost_handles_run_result_without_model_name() -> None:
    usage = Usage(
        requests=1,
        input_tokens=2,
        output_tokens=1,
        total_tokens=3,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
    response = ModelResponse(output=[], usage=usage, response_id=None)
    run_result = SimpleNamespace(raw_responses=[response], _sub_agent_responses_with_model=[])

    usage_stats = UsageStats(
        request_count=1,
        cached_tokens=0,
        input_tokens=2,
        output_tokens=1,
        total_tokens=3,
        total_cost=123.0,
        reasoning_tokens=None,
        audio_tokens=None,
    )
    with_cost = calculate_usage_with_cost(usage_stats, run_result=typing.cast(RunResult, run_result))
    assert with_cost.total_cost == 0.0

