# langchain-ai/deepagents
# 223 LLM-backed test functions across 190 test files
# Source: https://github.com/langchain-ai/deepagents

# --- libs/cli/tests/unit_tests/test_agent.py ---

def test_format_write_file_description_create_new_file(tmp_path: Path) -> None:
    """Test write_file description for creating a new file."""
    new_file = tmp_path / "new_file.py"
    tool_call = cast(
        "ToolCall",
        {
            "name": "write_file",
            "args": {
                "file_path": str(new_file),
                "content": "def hello():\n    return 'world'\n",
            },
            "id": "call-1",
        },
    )

    description = _format_write_file_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Action: Create file" in description
    assert "File:" not in description

def test_format_write_file_description_overwrite_existing_file(tmp_path: Path) -> None:
    """Test write_file description for overwriting an existing file."""
    existing_file = tmp_path / "existing.py"
    existing_file.write_text("old content")

    tool_call = cast(
        "ToolCall",
        {
            "name": "write_file",
            "args": {
                "file_path": str(existing_file),
                "content": "line1\nline2\nline3\n",
            },
            "id": "call-2",
        },
    )

    description = _format_write_file_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Action: Overwrite file" in description
    assert "File:" not in description

def test_format_edit_file_description_single_occurrence():
    """Test edit_file description for single occurrence replacement."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "edit_file",
            "args": {
                "file_path": "/path/to/file.py",
                "old_string": "foo",
                "new_string": "bar",
                "replace_all": False,
            },
            "id": "call-3",
        },
    )

    description = _format_edit_file_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Action: Replace text (single occurrence)" in description
    assert "File:" not in description

def test_format_edit_file_description_all_occurrences():
    """Test edit_file description for replacing all occurrences."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "edit_file",
            "args": {
                "file_path": "/path/to/file.py",
                "old_string": "foo",
                "new_string": "bar",
                "replace_all": True,
            },
            "id": "call-4",
        },
    )

    description = _format_edit_file_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Action: Replace text (all occurrences)" in description
    assert "File:" not in description

def test_format_web_search_description():
    """Test web_search description formatting."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "web_search",
            "args": {
                "query": "python async programming",
                "max_results": 10,
            },
            "id": "call-5",
        },
    )

    description = _format_web_search_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Query: python async programming" in description
    assert "Max results: 10" in description
    assert f"{get_glyphs().warning}  This will use Tavily API credits" in description

def test_format_web_search_description_default_max_results():
    """Test web_search description with default max_results."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "web_search",
            "args": {
                "query": "langchain tutorial",
            },
            "id": "call-6",
        },
    )

    description = _format_web_search_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Query: langchain tutorial" in description
    assert "Max results: 5" in description

def test_format_fetch_url_description():
    """Test fetch_url description formatting."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "fetch_url",
            "args": {
                "url": "https://example.com/docs",
                "timeout": 60,
            },
            "id": "call-7",
        },
    )

    description = _format_fetch_url_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "URL: https://example.com/docs" in description
    assert "Timeout: 60s" in description
    warning = get_glyphs().warning
    assert f"{warning}  Will fetch and convert web content to markdown" in description

def test_format_fetch_url_description_default_timeout():
    """Test fetch_url description with default timeout."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "fetch_url",
            "args": {
                "url": "https://api.example.com",
            },
            "id": "call-8",
        },
    )

    description = _format_fetch_url_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "URL: https://api.example.com" in description
    assert "Timeout: 30s" in description

def test_format_task_description():
    """Test task (subagent) description formatting."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "task",
            "args": {
                "description": "Analyze code structure and identify main components.",
                "subagent_type": "general-purpose",
            },
            "id": "call-9",
        },
    )

    description = _format_task_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Subagent Type: general-purpose" in description
    assert "Task Instructions:" in description
    assert "Analyze code structure and identify main components." in description
    warning = get_glyphs().warning
    msg = "Subagent will have access to file operations and shell commands"
    assert f"{warning} {msg} {warning}" in description
    assert description.index(warning) < description.index("Task Instructions:")

def test_format_task_description_truncates_long_description():
    """Test task description truncates long descriptions."""
    long_description = "x" * 600  # 600 characters
    tool_call = cast(
        "ToolCall",
        {
            "name": "task",
            "args": {
                "description": long_description,
                "subagent_type": "general-purpose",
            },
            "id": "call-10",
        },
    )

    description = _format_task_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Subagent Type: general-purpose" in description
    assert "..." in description
    # Description should be truncated to 500 chars + "..."
    assert len(description) < len(long_description) + 300

def test_format_execute_description():
    """Test execute command description formatting."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "execute",
            "args": {
                "command": "python script.py",
            },
            "id": "call-12",
        },
    )

    description = _format_execute_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Execute Command: python script.py" in description
    assert "Working Directory:" in description

def test_format_execute_description_with_hidden_unicode():
    """Hidden Unicode in command should trigger warning and marker display."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "execute",
            "args": {"command": "echo a\u202eb"},
            "id": "call-13",
        },
    )
    description = _format_execute_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )
    assert "Execute Command: echo ab" in description
    assert "Hidden Unicode detected" in description
    assert "U+202E" in description
    assert "Raw:" in description

def test_format_fetch_url_description_with_suspicious_url():
    """Suspicious URL should trigger warning lines in fetch_url description."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "fetch_url",
            "args": {"url": "https://аpple.com"},
            "id": "call-14",
        },
    )
    description = _format_fetch_url_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )
    assert "URL warning" in description

def test_format_fetch_url_description_with_hidden_unicode_in_url():
    """Hidden Unicode in URL should be stripped from display."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "fetch_url",
            "args": {"url": "https://exa\u200bmple.com"},
            "id": "call-15",
        },
    )
    description = _format_fetch_url_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )
    assert "URL: https://example.com" in description
    assert "\u200b" not in description

    def test_skips_entry_missing_required_fields(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text(
            '[async_subagents.incomplete]\ndescription = "Missing url and graph_id"\n'
        )
        result = load_async_subagents(config)
        assert result == []


# --- libs/cli/tests/unit_tests/test_configurable_model.py ---

    def test_none_runtime(self) -> None:
        request = ModelRequest(
            model=_make_model("claude-sonnet-4-6"),
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=None,
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0].model is request.model

    def test_non_dict_context_ignored(self) -> None:
        runtime = SimpleNamespace(context="not-a-dict")
        request = ModelRequest(
            model=_make_model("claude-sonnet-4-6"),
            messages=[HumanMessage(content="hi")],
            tools=[],
            runtime=cast("Any", runtime),
        )
        captured: list[ModelRequest] = []
        _mw.wrap_model_call(
            request, lambda r: (captured.append(r), _make_response())[1]
        )
        assert captured[0].model is request.model


# --- libs/deepagents/tests/integration_tests/test_deepagents.py ---

    def test_deep_agent_with_subagents(self):
        subagents = [
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "system_prompt": "You are a weather agent.",
                "tools": [get_weather],
                "model": SAMPLE_MODEL,
            }
        ]
        agent = create_deep_agent(tools=[sample_tool], subagents=subagents)
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "weather_agent" for tool_call in tool_calls)

    def test_deep_agent_with_subagents_gen_purpose(self):
        subagents = [
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "system_prompt": "You are a weather agent.",
                "tools": [get_weather],
                "model": SAMPLE_MODEL,
            }
        ]
        agent = create_deep_agent(tools=[sample_tool], subagents=subagents)
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [HumanMessage(content="Use the general purpose subagent to call the sample tool")]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "general-purpose" for tool_call in tool_calls)

    def test_deep_agent_with_subagents_with_middleware(self):
        subagents = [
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "system_prompt": "You are a weather agent.",
                "tools": [],
                "model": SAMPLE_MODEL,
                "middleware": [WeatherToolMiddleware()],
            }
        ]
        agent = create_deep_agent(tools=[sample_tool], subagents=subagents)
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "weather_agent" for tool_call in tool_calls)

    def test_deep_agent_with_custom_subagents(self):
        subagents = [
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "system_prompt": "You are a weather agent.",
                "tools": [get_weather],
                "model": SAMPLE_MODEL,
            },
            {
                "name": "soccer_agent",
                "description": "Use this agent to get the latest soccer scores",
                "runnable": create_agent(
                    model=SAMPLE_MODEL,
                    tools=[get_soccer_scores],
                    system_prompt="You are a soccer agent.",
                ),
            },
        ]
        agent = create_deep_agent(tools=[sample_tool], subagents=subagents)
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [HumanMessage(content="Look up the weather in Tokyo, and the latest scores for Manchester City!")]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "weather_agent" for tool_call in tool_calls)
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "soccer_agent" for tool_call in tool_calls)

    def test_deep_agent_with_extended_state_and_subagents(self):
        subagents = [
            {
                "name": "basketball_info_agent",
                "description": "Use this agent to get surface level info on any basketball topic",
                "system_prompt": "You are a basketball info agent.",
                "middleware": [ResearchMiddlewareWithTools()],
            }
        ]
        agent = create_deep_agent(tools=[sample_tool], subagents=subagents, middleware=[ResearchMiddleware()])
        assert_all_deepagent_qualities(agent)
        assert "research" in agent.stream_channels
        result = agent.invoke({"messages": [HumanMessage(content="Get surface level info on lebron james")]}, config={"recursion_limit": 100})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "basketball_info_agent" for tool_call in tool_calls)
        assert TOY_BASKETBALL_RESEARCH in result["research"]

    def test_deep_agent_with_subagents_no_tools(self):
        subagents = [
            {
                "name": "basketball_info_agent",
                "description": "Use this agent to get surface level info on any basketball topic",
                "system_prompt": "You are a basketball info agent.",
            }
        ]
        agent = create_deep_agent(tools=[sample_tool], subagents=subagents)
        assert_all_deepagent_qualities(agent)
        result = agent.invoke(
            {"messages": [HumanMessage(content="Use the basketball info subagent to call the sample tool")]}, config={"recursion_limit": 100}
        )
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "basketball_info_agent" for tool_call in tool_calls)

    def test_response_format_tool_strategy(self):
        class StructuredOutput(BaseModel):
            pokemon: list[str]

        agent = create_deep_agent(response_format=ToolStrategy(schema=StructuredOutput))
        response = agent.invoke({"messages": [{"role": "user", "content": "Who are all of the Kanto starters?"}]})
        structured_output = response["structured_response"]
        assert len(structured_output.pokemon) == 3


# --- libs/deepagents/tests/integration_tests/test_filesystem_middleware.py ---

    def test_filesystem_system_prompt_override(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=StateBackend(),
                    system_prompt="In every single response, you must say the word 'pokemon'! You love it!",
                )
            ],
        )
        response = agent.invoke({"messages": [HumanMessage(content="What do you like?")]})
        assert "pokemon" in _to_ascii(response["messages"][1].text.lower())

    def test_filesystem_system_prompt_override_with_composite_backend(self):
        def backend(_rt):
            return build_composite_state_backend(routes={"/memories/": StoreBackend()})

        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=backend,
                    system_prompt="In every single response, you must say the word 'pizza'! You love it!",
                )
            ],
            store=InMemoryStore(),
        )
        response = agent.invoke({"messages": [HumanMessage(content="What do you like?")]})
        assert "pizza" in _to_ascii(response["messages"][1].text.lower())

    def test_ls_longterm_without_path(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/charmander.txt",
            {
                "content": ["Ember"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="List your files in root")],
                "files": {
                    "/pizza.txt": FileData(
                        content=["Hello world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/pokemon/squirtle.txt": FileData(
                        content=["Splash"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        ls_message = next(message for message in messages if message.type == "tool" and message.name == "ls")
        assert "/pizza.txt" in ls_message.text
        assert "/pokemon/squirtle.txt" not in ls_message.text
        assert "/memories/test.txt" not in ls_message.text
        assert "/memories/pokemon/charmander.txt" not in ls_message.text
        # Verify directories are listed with trailing /
        assert "/pokemon/" in ls_message.text
        assert "/memories/" in ls_message.text

    def test_ls_longterm_with_path(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/charmander.txt",
            {
                "content": ["Ember"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="List all of your files in the /pokemon directory")],
                "files": {
                    "/pizza.txt": FileData(
                        content=["Hello world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/pokemon/squirtle.txt": FileData(
                        content=["Splash"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        ls_message = next(message for message in messages if message.type == "tool" and message.name == "ls")
        assert "/pokemon/squirtle.txt" in ls_message.text
        assert "/memories/pokemon/charmander.txt" not in ls_message.text

    def test_read_file_longterm_local_file(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Read test.txt from the local filesystem")],
                "files": {
                    "/test.txt": FileData(
                        content=["Goodbye world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    )
                },
            },
            config=config,
        )
        messages = response["messages"]
        read_file_message = next(message for message in messages if message.type == "tool" and message.name == "read_file")
        assert read_file_message is not None
        assert "Goodbye world" in read_file_message.content

    def test_read_file_longterm_store_file(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Read test.txt from the memories directory")],
                "files": {
                    "/test.txt": FileData(
                        content=["Goodbye world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    )
                },
            },
            config=config,
        )
        messages = response["messages"]
        read_file_message = next(message for message in messages if message.type == "tool" and message.name == "read_file")
        assert read_file_message is not None
        assert "Hello world" in read_file_message.content

    def test_read_file_longterm(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/charmander.txt",
            {
                "content": ["Ember"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Read the contents of the file about charmander from the memories directory.")],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        ai_msg_w_toolcall = next(
            message
            for message in messages
            if message.type == "ai"
            and any(tc["name"] == "read_file" and tc["args"]["file_path"] == "/memories/pokemon/charmander.txt" for tc in message.tool_calls)
        )
        assert ai_msg_w_toolcall is not None

    def test_write_file_longterm(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Write a haiku about Charmander to the memories directory in /charmander.txt, use the word 'fiery'")
                ],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        write_file_message = next(message for message in messages if message.type == "tool" and message.name == "write_file")
        assert write_file_message is not None
        file_item = store.get(("filesystem",), "/charmander.txt")
        assert file_item is not None
        content = file_item.value["content"]
        assert isinstance(content, str), f"Expected str content, got {type(content)}"
        assert "fiery" in content or "Fiery" in content

    def test_write_file_fail_already_exists_in_local(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Write a haiku about Charmander to /charmander.txt, use the word 'fiery'")],
                "files": {
                    "/charmander.txt": FileData(
                        content=["Hello world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    )
                },
            },
            config=config,
        )
        messages = response["messages"]
        write_file_message = next(message for message in messages if message.type == "tool" and message.name == "write_file")
        assert write_file_message is not None
        assert "Cannot write" in write_file_message.content

    def test_edit_file_longterm(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/charmander.txt",
            {
                "content": ["The fire burns brightly. The fire burns hot."],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Edit the file about charmander in the memories directory, to replace all instances of the word 'fire' with 'embers'"
                    )
                ],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        edit_file_message = next(message for message in messages if message.type == "tool" and message.name == "edit_file")
        assert edit_file_message is not None
        edited_content = store.get(("filesystem",), "/charmander.txt").value["content"]
        assert isinstance(edited_content, str), f"Expected str content, got {type(edited_content)}"
        assert "embers" in edited_content.lower()
        assert "fire" not in edited_content.lower()

    def test_tool_call_with_tokens_exceeding_limit(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            tools=[get_nba_standings],
            middleware=[
                FilesystemMiddleware(
                    backend=StateBackend(),
                )
            ],
        )
        response = agent.invoke(
            {"messages": [HumanMessage(content="Get the NBA standings using your tool. If the tool returns bad results, tell the user.")]}
        )
        assert response["messages"][2].type == "tool"
        assert len(response["messages"][2].content) < 10000
        assert len(response["files"].keys()) == 1
        assert any("large_tool_results" in key for key in response["files"])

    def test_tool_call_with_tokens_exceeding_custom_limit(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            tools=[get_nfl_standings],
            middleware=[
                FilesystemMiddleware(
                    backend=StateBackend(),
                    tool_token_limit_before_evict=1000,
                )
            ],
        )
        response = agent.invoke(
            {"messages": [HumanMessage(content="Get the NFL standings using your tool. If the tool returns bad results, tell the user.")]}
        )
        assert response["messages"][2].type == "tool"
        assert len(response["messages"][2].content) < 1500
        assert len(response["files"].keys()) == 1
        assert any("large_tool_results" in key for key in response["files"])

    def test_command_with_tool_call(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            tools=[get_la_liga_standings],
            middleware=[
                FilesystemMiddleware(
                    backend=StateBackend(),
                    tool_token_limit_before_evict=1000,
                )
            ],
        )
        response = agent.invoke(
            {"messages": [HumanMessage(content="Get the la liga standings using your tool. If the tool returns bad results, tell the user.")]}
        )
        assert response["messages"][2].type == "tool"
        assert len(response["messages"][2].content) < 1500
        assert len(response["files"].keys()) == 1
        assert any("large_tool_results" in key for key in response["files"])

    def test_command_with_tool_call_existing_state(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            tools=[get_premier_league_standings],
            middleware=[
                FilesystemMiddleware(
                    backend=StateBackend(),
                    tool_token_limit_before_evict=1000,
                ),
                ResearchMiddleware(),
            ],
        )
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Get the premier league standings using your tool. If the tool returns bad results, tell the user.")
                ],
            }
        )
        assert response["messages"][2].type == "tool"
        assert len(response["messages"][2].content) < 1500
        assert len(response["files"].keys()) == 2
        assert any("large_tool_results" in key for key in response["files"])
        assert "/test.txt" in response["files"]
        assert "research" in response

    def test_glob_search_shortterm_only(self):
        checkpointer = MemorySaver()
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=StateBackend(),
                )
            ],
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Use glob to find all Python files")],
                "files": {
                    "/test.py": FileData(
                        content=["import os"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/main.py": FileData(
                        content=["def main(): pass"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/readme.txt": FileData(
                        content=["Documentation"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        glob_message = next(message for message in messages if message.type == "tool" and message.name == "glob")
        assert "/test.py" in glob_message.content
        assert "/main.py" in glob_message.content
        assert "/readme.txt" not in glob_message.content

    def test_glob_search_longterm_only(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/config.py",
            {
                "content": ["DEBUG = True"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/settings.py",
            {
                "content": ["SECRET_KEY = 'abc'"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/notes.txt",
            {
                "content": ["Important notes"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Use glob to find all Python files in /memories")],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        glob_message = next(message for message in messages if message.type == "tool" and message.name == "glob")
        assert "/memories/config.py" in glob_message.content
        assert "/memories/settings.py" in glob_message.content
        assert "/memories/notes.txt" not in glob_message.content

    def test_glob_search_mixed_memory(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/longterm.py",
            {
                "content": ["# Longterm file"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/longterm.txt",
            {
                "content": ["Text file"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Use glob to find all Python files")],
                "files": {
                    "/shortterm.py": FileData(
                        content=["# Shortterm file"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/shortterm.txt": FileData(
                        content=["Another text file"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        glob_message = next(message for message in messages if message.type == "tool" and message.name == "glob")
        assert "/shortterm.py" in glob_message.content
        assert "/memories/longterm.py" in glob_message.content
        assert "/shortterm.txt" not in glob_message.content
        assert "/memories/longterm.txt" not in glob_message.content

    def test_grep_search_shortterm_only(self):
        checkpointer = MemorySaver()
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=StateBackend(),
                )
            ],
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Use grep to find all files containing the word 'import'")],
                "files": {
                    "/test.py": FileData(
                        content=["import os", "import sys"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/main.py": FileData(
                        content=["def main(): pass"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/helper.py": FileData(
                        content=["import json"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        grep_message = next(message for message in messages if message.type == "tool" and message.name == "grep")
        assert "/test.py" in grep_message.content
        assert "/helper.py" in grep_message.content
        assert "/main.py" not in grep_message.content

    def test_grep_search_longterm_only(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/pokemon/charmander.txt",
            {
                "content": ["Charmander is a fire type", "It evolves into Charmeleon"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/squirtle.txt",
            {
                "content": ["Squirtle is a water type", "It evolves into Wartortle"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/bulbasaur.txt",
            {
                "content": ["Bulbasaur is a grass type"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Use grep to find all files in the memories directory containing the word 'fire'")],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        grep_message = next(message for message in messages if message.type == "tool" and message.name == "grep")
        assert "/memories/pokemon/charmander.txt" in grep_message.content
        assert "/memories/pokemon/squirtle.txt" not in grep_message.content
        assert "/memories/pokemon/bulbasaur.txt" not in grep_message.content

    def test_grep_search_mixed_memory(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/longterm_config.py",
            {
                "content": ["DEBUG = True", "TESTING = False"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/longterm_settings.py",
            {
                "content": ["SECRET_KEY = 'abc'"],
                "encoding": "utf-8",
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Use grep to find all files containing 'DEBUG'")],
                "files": {
                    "/shortterm_config.py": FileData(
                        content=["DEBUG = False", "VERBOSE = True"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/shortterm_main.py": FileData(
                        content=["def main(): pass"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        grep_message = next(message for message in messages if message.type == "tool" and message.name == "grep")
        assert "/shortterm_config.py" in grep_message.content
        assert "/memories/longterm_config.py" in grep_message.content
        assert "/shortterm_main.py" not in grep_message.content
        assert "/memories/longterm_settings.py" not in grep_message.content

    def test_default_backend_fallback(self):
        checkpointer = MemorySaver()
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware()  # No backend specified
            ],
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}

        response = agent.invoke(
            {"messages": [HumanMessage(content="Write 'Hello World' to /test.txt")]},
            config=config,
        )

        assert "/test.txt" in response["files"]
        content = response["files"]["/test.txt"]["content"]
        assert isinstance(content, str), f"Expected str content, got {type(content)}"
        assert "Hello World" in content

        response = agent.invoke(
            {"messages": [HumanMessage(content="Read /test.txt")]},
            config=config,
        )
        messages = response["messages"]
        read_message = next(msg for msg in messages if msg.type == "tool" and msg.name == "read_file")
        assert "Hello World" in read_message.content


# --- libs/deepagents/tests/integration_tests/test_subagent_middleware.py ---

    def test_general_purpose_subagent(self):
        agent = create_agent(
            model="claude-sonnet-4-6",
            system_prompt="Use the general-purpose subagent to get the weather in a city.",
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend(),
                    subagents=[
                        {
                            **GENERAL_PURPOSE_SUBAGENT,
                            "model": "claude-sonnet-4-6",
                            "tools": [get_weather],
                        }
                    ],
                )
            ],
        )
        assert "task" in agent.nodes["tools"].bound._tools_by_name
        response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
        assert response["messages"][1].tool_calls[0]["name"] == "task"
        assert response["messages"][1].tool_calls[0]["args"]["subagent_type"] == "general-purpose"

    def test_defined_subagent_custom_runnable(self):
        custom_subagent = create_agent(
            model="gpt-5.4",
            system_prompt="Use the get_weather tool to get the weather in a city.",
            tools=[get_weather],
        )
        agent = create_agent(
            model="claude-sonnet-4-6",
            system_prompt="Use the task tool to call a subagent.",
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend(),
                    subagents=[
                        {
                            "name": "weather",
                            "description": "This subagent can get weather in cities.",
                            "runnable": custom_subagent,
                        }
                    ],
                )
            ],
        )
        expected_tool_calls = [
            {
                "name": "task",
                "args": {"subagent_type": "weather"},
                "model": "claude-sonnet-4-6",
            },
            {"name": "get_weather", "args": {}, "model": "gpt-5.4"},
        ]
        assert_expected_subgraph_actions(
            expected_tool_calls,
            agent,
            {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        )

    def test_subagent_response_format_serialized_as_json(self):
        """Test that subagent responseFormat produces JSON-serialized ToolMessage content.

        Verifies the end-to-end flow when `response_format` is set directly on a
        `SubAgent` spec: the subagent's `structured_response` is JSON-serialized
        into the ToolMessage content returned to the parent agent.
        """

        class SubagentFindings(BaseModel):
            findings: str = Field(description="The findings")
            confidence: float = Field(description="Confidence score")
            summary: str = Field(description="Brief summary")

        agent = create_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="You are an orchestrator. Always delegate tasks to the appropriate subagent via the task tool.",
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend(),
                    subagents=[
                        {
                            "name": "foo",
                            "description": "Call this when the user says 'foo'",
                            "system_prompt": "You are a foo agent",
                            "model": "claude-haiku-4-5",
                            "tools": [],
                            "response_format": ToolStrategy(schema=SubagentFindings),
                        },
                    ],
                )
            ],
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="foo - tell me how confident you are that pineapple belongs on pizza")]},
            {"recursion_limit": 100},
        )

        agent_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        tool_calls = [tc for msg in agent_messages for tc in (msg.tool_calls or [])]
        assert any(tc["name"] == "task" and tc["args"].get("subagent_type") == "foo" for tc in tool_calls)

        task_tool_messages = [msg for msg in result["messages"] if msg.type == "tool" and msg.name == "task"]
        assert len(task_tool_messages) > 0

        task_tool_message = task_tool_messages[0]
        parsed = json.loads(task_tool_message.content)
        assert "findings" in parsed
        assert "confidence" in parsed
        assert "summary" in parsed
        assert isinstance(parsed["findings"], str)
        assert isinstance(parsed["confidence"], (int, float))
        assert isinstance(parsed["summary"], str)


# --- libs/deepagents/tests/unit_tests/test_artifacts_root.py ---

    def test_large_tool_result_eviction_uses_artifacts_root(self) -> None:
        backend = _make_composite_backend(artifacts_root="/workspace")
        mw = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=100)
        runtime = _runtime("evict_123")

        large_content = "x" * 5000
        msg = ToolMessage(content=large_content, tool_call_id="evict_123")
        result = mw._intercept_large_tool_result(msg, runtime)

        assert isinstance(result, ToolMessage)
        assert "/workspace/large_tool_results/evict_123" in result.content
        [resp] = backend.download_files(["/workspace/large_tool_results/evict_123"])
        assert resp.error is None
        assert resp.content is not None
        assert resp.content == b"x" * 5000

    def test_large_tool_result_eviction_default_root(self) -> None:
        backend = _make_store_backend()
        mw = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=100)
        runtime = _runtime("evict_456")

        large_content = "x" * 5000
        msg = ToolMessage(content=large_content, tool_call_id="evict_456")
        result = mw._intercept_large_tool_result(msg, runtime)

        assert isinstance(result, ToolMessage)
        assert "/large_tool_results/evict_456" in result.content
        [resp] = backend.download_files(["/large_tool_results/evict_456"])
        assert resp.error is None
        assert resp.content is not None
        assert resp.content == b"x" * 5000

    def test_large_tool_result_eviction(self) -> None:
        """Large tool result eviction writes to the custom artifacts_root path."""
        backend = _make_composite_backend(artifacts_root="/workspace")
        mw = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=100)
        runtime = _runtime("evict_ws")

        large_content = "x" * 5000
        msg = ToolMessage(content=large_content, tool_call_id="evict_ws")
        result = mw._intercept_large_tool_result(msg, runtime)

        assert isinstance(result, ToolMessage)
        assert "/workspace/large_tool_results/evict_ws" in result.content
        [resp] = backend.download_files(["/workspace/large_tool_results/evict_ws"])
        assert resp.error is None
        assert resp.content is not None
        assert resp.content == b"x" * 5000
        [resp] = backend.download_files(["/large_tool_results/evict_ws"])
        assert resp.content is None

    async def test_async_large_tool_result_eviction_uses_artifacts_root(self) -> None:
        backend = _make_composite_backend(artifacts_root="/workspace")
        mw = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=100)
        runtime = _runtime("async_evict_123")

        large_content = "x" * 5000
        msg = ToolMessage(content=large_content, tool_call_id="async_evict_123")
        result = await mw._aintercept_large_tool_result(msg, runtime)

        assert isinstance(result, ToolMessage)
        assert "/workspace/large_tool_results/async_evict_123" in result.content
        [resp] = await backend.adownload_files(["/workspace/large_tool_results/async_evict_123"])
        assert resp.error is None
        assert resp.content is not None
        assert resp.content == b"x" * 5000


# --- libs/deepagents/tests/unit_tests/test_async_subagents.py ---

    def test_launch_invalid_type_returns_error_string(self) -> None:
        tools = _build_async_subagent_tools([_make_spec("alpha")])
        launch = tools[0]
        result = launch.func(
            description="do something",
            subagent_type="nonexistent",
            runtime=_make_runtime(),
        )
        assert isinstance(result, str)
        assert "Unknown async subagent type" in result
        assert "`alpha`" in result

    def test_empty_state_returns_no_tasks(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        list_tool = tools[4]
        rt = _make_runtime()
        result = list_tool.func(runtime=rt)
        assert "No async subagent tasks tracked" in result

    async def test_async_list_returns_no_tasks(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        list_tool = tools[4]
        rt = _make_runtime()
        result = await list_tool.coroutine(runtime=rt)
        assert "No async subagent tasks tracked" in result

    def test_cancel_unknown_task_returns_error(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        cancel = _get_tool(tools, "cancel_async_task")
        rt = _make_runtime("tc_cancel")
        result = cancel.func(task_id="nonexistent", runtime=rt)
        assert isinstance(result, str)
        assert "No tracked task found" in result

    def test_check_unknown_task_returns_error(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        check = _get_tool(tools, "check_async_task")
        rt = _make_runtime()
        result = check.func(task_id="nonexistent", runtime=rt)
        assert isinstance(result, str)
        assert "No tracked task found" in result

    def test_update_unknown_task_returns_error(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        update = _get_tool(tools, "update_async_task")
        rt = _make_runtime()
        result = update.func(task_id="nonexistent", message="hello", runtime=rt)
        assert isinstance(result, str)
        assert "No tracked task found" in result


# --- libs/deepagents/tests/unit_tests/test_end_to_end.py ---

    def test_deep_agent_with_middleware_with_tool_and_state(self) -> None:
        """Verifies that middleware can inject both tools and extended state channels."""
        agent = create_deep_agent(middleware=[SampleMiddlewareWithToolsAndState()])
        assert_all_deepagent_qualities(agent)
        assert "sample_tool" in agent.nodes["tools"].bound._tools_by_name
        assert "sample_input" in agent.stream_channels

    def test_state_backend_runtime_deprecation(self) -> None:
        """Passing runtime to StateBackend emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StateBackend("ignored_runtime_value")

        deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert "runtime" in str(deprecations[0].message).lower()

    def test_store_backend_runtime_deprecation(self) -> None:
        """Passing runtime to StoreBackend emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StoreBackend("ignored_runtime_value")

        deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert "runtime" in str(deprecations[0].message).lower()


# --- libs/deepagents/tests/unit_tests/test_middleware.py ---

    def test_filesystem_middleware(self):
        middleware = [FilesystemMiddleware()]
        agent = create_agent(model="claude-sonnet-4-6", middleware=middleware, tools=[])
        assert "files" in agent.stream_channels
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        assert "ls" in agent_tools
        assert "read_file" in agent_tools
        assert "write_file" in agent_tools
        assert "edit_file" in agent_tools
        assert "glob" in agent_tools
        assert "grep" in agent_tools

    def test_multiple_middleware(self):
        middleware = [
            FilesystemMiddleware(),
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[{**GENERAL_PURPOSE_SUBAGENT, "model": "claude-sonnet-4-6", "tools": []}],
            ),
        ]
        agent = create_agent(model="claude-sonnet-4-6", middleware=middleware, tools=[])
        assert "files" in agent.stream_channels
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        assert "ls" in agent_tools
        assert "read_file" in agent_tools
        assert "write_file" in agent_tools
        assert "edit_file" in agent_tools
        assert "glob" in agent_tools
        assert "grep" in agent_tools
        assert "task" in agent_tools

    def test_ls_shortterm(self):
        files = {
            "/test.txt": FileData(
                content=["Hello world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/test2.txt": FileData(
                content=["Goodbye world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result = ls_tool.invoke({"runtime": _runtime(), "path": "/"})
        assert result.content == str(["/test.txt", "/test2.txt"])

    def test_ls_shortterm_with_path(self):
        files = {
            "/test.txt": FileData(
                content=["Hello world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/pokemon/test2.txt": FileData(
                content=["Goodbye world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/pokemon/charmander.txt": FileData(
                content=["Ember"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/pokemon/water/squirtle.txt": FileData(
                content=["Water"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result_raw = ls_tool.invoke(
            {
                "path": "/pokemon/",
                "runtime": _runtime(),
            }
        )
        result = result_raw.content
        # ls should only return files directly in /pokemon/, not in subdirectories
        assert "/pokemon/test2.txt" in result
        assert "/pokemon/charmander.txt" in result
        assert "/pokemon/water/squirtle.txt" not in result  # In subdirectory, should NOT be listed
        # ls should also list subdirectories with trailing /
        assert "/pokemon/water/" in result

    def test_ls_shortterm_lists_directories(self):
        """Test that ls lists directories with trailing / for traversal."""
        files = {
            "/test.txt": FileData(
                content=["Hello world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/pokemon/charmander.txt": FileData(
                content=["Ember"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/pokemon/water/squirtle.txt": FileData(
                content=["Water"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/docs/readme.md": FileData(
                content=["Documentation"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result_raw = ls_tool.invoke(
            {
                "path": "/",
                "runtime": _runtime(),
            }
        )
        result = result_raw.content
        # ls should list both files and directories at root level
        assert "/test.txt" in result
        assert "/pokemon/" in result
        assert "/docs/" in result
        # But NOT subdirectory files
        assert "/pokemon/charmander.txt" not in result
        assert "/pokemon/water/squirtle.txt" not in result

    def test_glob_search_shortterm_simple_pattern(self):
        files = {
            "/test.txt": FileData(
                content=["Hello world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/test.py": FileData(
                content=["print('hello')"],
                modified_at="2021-01-02",
                created_at="2021-01-01",
            ),
            "/pokemon/charmander.py": FileData(
                content=["Ember"],
                modified_at="2021-01-03",
                created_at="2021-01-01",
            ),
            "/pokemon/squirtle.txt": FileData(
                content=["Water"],
                modified_at="2021-01-04",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result_raw = glob_search_tool.invoke(
            {
                "pattern": "*.py",
                "runtime": _runtime(),
            }
        )
        result = result_raw.content
        # Standard glob: *.py only matches files in root directory, not subdirectories
        assert result == str(["/test.py"])

    def test_glob_search_shortterm_wildcard_pattern(self):
        files = {
            "/src/main.py": FileData(
                content=["main code"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/src/utils/helper.py": FileData(
                content=["helper code"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/tests/test_main.py": FileData(
                content=["test code"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result_raw = glob_search_tool.invoke(
            {
                "pattern": "**/*.py",
                "runtime": _runtime(),
            }
        )
        result = result_raw.content
        assert "/src/main.py" in result
        assert "/src/utils/helper.py" in result
        assert "/tests/test_main.py" in result

    def test_glob_search_shortterm_with_path(self):
        files = {
            "/src/main.py": FileData(
                content=["main code"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/src/utils/helper.py": FileData(
                content=["helper code"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/tests/test_main.py": FileData(
                content=["test code"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result_raw = glob_search_tool.invoke(
            {
                "pattern": "*.py",
                "path": "/src",
                "runtime": _runtime(),
            }
        )
        result = result_raw.content
        assert "/src/main.py" in result
        assert "/src/utils/helper.py" not in result
        assert "/tests/test_main.py" not in result

    def test_glob_search_shortterm_no_matches(self):
        files = {
            "/test.txt": FileData(
                content=["Hello world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result = glob_search_tool.invoke(
            {
                "pattern": "*.py",
                "runtime": _runtime(),
            }
        )
        assert result.content == str([])

    def test_glob_search_truncates_large_results(self):
        """Test that glob results are truncated when they exceed token limit."""
        # Create a large number of files that will exceed TOOL_RESULT_TOKEN_LIMIT
        # TOOL_RESULT_TOKEN_LIMIT = 20000, * 4 chars/token = 80000 chars
        # Create files with long paths to exceed this limit
        files = {}
        # Create 2000 files with 50-char paths = 100,000 chars total (exceeds 80k limit)
        for i in range(2000):
            path = f"/very_long_file_name_to_increase_size_{i:04d}.txt"
            files[path] = FileData(
                content=["content"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            )

        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result_raw = glob_search_tool.invoke(
            {
                "pattern": "*.txt",
                "runtime": _runtime(),
            }
        )

        # Result should be truncated
        result = result_raw.content
        assert isinstance(result, str)
        assert len(result.split(", ")) < 2000  # Should be truncated to fewer files
        # Last element should be the truncation message
        # Need to do the :-2 to account for the wrapping list characters
        assert result[:-2].endswith(TRUNCATION_GUIDANCE)

    def test_grep_search_shortterm_files_with_matches(self):
        files = {
            "/test.py": FileData(
                content=["import os", "import sys", "print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/main.py": FileData(
                content=["def main():", "    pass"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/helper.txt": FileData(
                content=["import json"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = grep_search_tool.invoke(
            {
                "pattern": "import",
                "runtime": _runtime(),
            }
        )
        assert "/test.py" in result.content
        assert "/helper.txt" in result.content
        assert "/main.py" not in result.content

    def test_grep_search_shortterm_content_mode(self):
        files = {
            "/test.py": FileData(
                content=["import os", "import sys", "print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = grep_search_tool.invoke(
            {
                "pattern": "import",
                "output_mode": "content",
                "runtime": _runtime(),
            }
        )
        assert "1: import os" in result.content
        assert "2: import sys" in result.content
        assert "print" not in result.content

    def test_grep_search_shortterm_count_mode(self):
        files = {
            "/test.py": FileData(
                content=["import os", "import sys", "print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/main.py": FileData(
                content=["import json", "data = {}"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = grep_search_tool.invoke(
            {
                "pattern": "import",
                "output_mode": "count",
                "runtime": _runtime(),
            }
        )
        assert "/test.py:2" in result.content or "/test.py: 2" in result.content
        assert "/main.py:1" in result.content or "/main.py: 1" in result.content

    def test_grep_search_shortterm_with_include(self):
        files = {
            "/test.py": FileData(
                content=["import os"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/test.txt": FileData(
                content=["import nothing"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = grep_search_tool.invoke(
            {
                "pattern": "import",
                "glob": "*.py",
                "runtime": _runtime(),
            }
        )
        assert "/test.py" in result.content
        assert "/test.txt" not in result.content

    def test_grep_search_shortterm_with_path(self):
        files = {
            "/src/main.py": FileData(
                content=["import os"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/tests/test.py": FileData(
                content=["import pytest"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = grep_search_tool.invoke(
            {
                "pattern": "import",
                "path": "/src",
                "runtime": _runtime(),
            }
        )
        assert "/src/main.py" in result.content
        assert "/tests/test.py" not in result.content

    def test_grep_search_shortterm_regex_pattern(self):
        """Test grep with literal pattern (not regex)."""
        files = {
            "/test.py": FileData(
                content=["def hello():", "def world():", "x = 5"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        # Search for literal "def " - literal search, not regex
        result = grep_search_tool.invoke(
            {
                "pattern": "def ",
                "output_mode": "content",
                "runtime": _runtime(),
            }
        )
        assert "1: def hello():" in result.content
        assert "2: def world():" in result.content
        assert "x = 5" not in result.content

    def test_grep_search_shortterm_no_matches(self):
        files = {
            "/test.py": FileData(
                content=["print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = grep_search_tool.invoke(
            {
                "pattern": "import",
                "runtime": _runtime(),
            }
        )
        assert result.content == "No matches found"

    def test_grep_search_shortterm_invalid_regex(self):
        """Test grep with special characters (literal search, not regex)."""
        files = {
            "/test.py": FileData(
                content=["print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        # Special characters are treated literally, so no matches expected
        result = grep_search_tool.invoke(
            {
                "pattern": "[invalid",
                "runtime": _runtime(),
            }
        )
        assert "No matches found" in result.content

    def test_intercept_short_toolmessage(self):
        """Test that small ToolMessages pass through unchanged."""
        backend, _ = _make_backend()
        middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=1000)
        runtime = _runtime("test_123")

        small_content = "x" * 1000
        tool_message = ToolMessage(content=small_content, tool_call_id="test_123")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert result == tool_message

    def test_intercept_long_toolmessage(self):
        """Test that large ToolMessages are intercepted and saved to filesystem."""
        backend, mem_store = _make_backend()
        middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=1000)
        runtime = _runtime("test_123")

        large_content = "x" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test_123")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, ToolMessage)
        assert mem_store.get(("filesystem",), "/large_tool_results/test_123") is not None
        assert "Tool result too large" in result.content

    def test_intercept_long_toolmessage_preserves_name(self):
        """Test that ToolMessage name is preserved after eviction."""
        backend, _ = _make_backend()
        middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=1000)
        runtime = _runtime("test_123")

        large_content = "x" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test_123", name="example_tool")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, ToolMessage)
        assert result.name == "example_tool"

    def test_intercept_command_with_short_toolmessage(self):
        """Test that Commands with small messages pass through unchanged."""
        backend, _ = _make_backend()
        middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=1000)
        runtime = _runtime("test_123")

        small_content = "x" * 1000
        tool_message = ToolMessage(content=small_content, tool_call_id="test_123")
        command = Command(update={"messages": [tool_message], "files": {}})
        result = middleware._intercept_large_tool_result(command, runtime)

        assert isinstance(result, Command)
        assert result.update["messages"][0].content == small_content

    def test_intercept_command_with_long_toolmessage(self):
        """Test that Commands with large messages are intercepted."""
        backend, mem_store = _make_backend()
        middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=1000)
        runtime = _runtime("test_123")

        large_content = "y" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test_123")
        command = Command(update={"messages": [tool_message], "files": {}})
        result = middleware._intercept_large_tool_result(command, runtime)

        assert isinstance(result, Command)
        assert mem_store.get(("filesystem",), "/large_tool_results/test_123") is not None
        assert "Tool result too large" in result.update["messages"][0].content

    def test_intercept_command_with_files_and_long_toolmessage(self):
        """Test that file updates are properly merged with existing files and other keys preserved."""
        backend, mem_store = _make_backend()
        middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=1000)
        runtime = _runtime("test_123")

        large_content = "z" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test_123")
        existing_file = FileData(content=["existing"], created_at="2021-01-01", modified_at="2021-01-01")
        command = Command(update={"messages": [tool_message], "files": {"/existing.txt": existing_file}, "custom_key": "custom_value"})
        result = middleware._intercept_large_tool_result(command, runtime)

        assert isinstance(result, Command)
        assert "/existing.txt" in result.update["files"]
        assert mem_store.get(("filesystem",), "/large_tool_results/test_123") is not None
        assert result.update["custom_key"] == "custom_value"

    def test_intercept_sanitizes_tool_call_id(self):
        """Test that tool_call_id with dangerous characters is sanitized in file path."""
        backend, mem_store = _make_backend()
        middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=1000)
        runtime = _runtime("test_123")

        large_content = "x" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test/call.id")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, ToolMessage)
        assert mem_store.get(("filesystem",), "/large_tool_results/test_call_id") is not None

    def test_intercept_content_block_with_large_text(self):
        """Test that content blocks with large text get evicted and converted to string."""
        backend, mem_store = _make_backend()
        middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=100)
        runtime = _runtime("test_cb")

        # Create list with content block with large text
        content_blocks = [{"type": "text", "text": "x" * 5000}]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_cb")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, ToolMessage)
        assert mem_store.get(("filesystem",), "/large_tool_results/test_cb") is not None
        # After eviction, content is always converted to plain string
        assert isinstance(result.content, str)
        assert "Tool result too large" in result.content

    def test_intercept_content_block_with_small_text(self):
        """Test that content blocks with small text are not evicted."""
        backend, _ = _make_backend()
        middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=1000)
        runtime = _runtime("test_small_cb")

        # Create list with content block with small text
        content_blocks = [{"type": "text", "text": "small text"}]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_small_cb")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        # Should return original message unchanged
        assert result == tool_message
        assert result.content == content_blocks

    def test_intercept_content_block_non_text_type_not_evicted(self):
        """Test that non-text-only content blocks are not evicted regardless of size."""
        backend, _ = _make_backend()
        middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=100)
        runtime = _runtime("test_other")

        content_blocks = [{"type": "image", "base64": "x" * 5000, "mime_type": "image/png"}]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_other")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert result == tool_message

    def test_single_text_block_extracts_text_directly(self, file_format):
        """Test that single text block extracts text content directly, not stringified structure."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format=file_format)
        middleware = FilesystemMiddleware(backend=be, tool_token_limit_before_evict=100)
        runtime = _runtime("test_single")

        # Create single text block with large text
        content_blocks = [{"type": "text", "text": "Hello world! " * 1000}]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_single")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, ToolMessage)
        # Check that the file contains actual text, not stringified dict
        item = mem_store.get(("filesystem",), "/large_tool_results/test_single")
        assert item is not None
        file_content = item.value["content"]
        if file_format == "v1":
            assert isinstance(file_content, list)
            text = "\n".join(file_content)
        else:
            assert isinstance(file_content, str)
            text = file_content
        # Should start with the actual text, not with "[{" which would indicate stringified dict
        assert text.startswith("Hello world!")
        assert not text.startswith("[{")

    def test_multiple_text_blocks_joins_text(self, file_format):
        """Test that multiple text blocks are joined, not stringified."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format=file_format)
        middleware = FilesystemMiddleware(backend=be, tool_token_limit_before_evict=100)
        runtime = _runtime("test_multi")

        content_blocks = [
            {"type": "text", "text": "First block " * 500},
            {"type": "text", "text": "Second block " * 500},
        ]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_multi")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, ToolMessage)
        item = mem_store.get(("filesystem",), "/large_tool_results/test_multi")
        assert item is not None
        file_content = item.value["content"]
        if file_format == "v1":
            assert isinstance(file_content, list)
            text = "\n".join(file_content)
        else:
            assert isinstance(file_content, str)
            text = file_content
        assert text.startswith("First block")
        assert "Second block" in text
        assert not text.startswith("[{")

    def test_mixed_content_blocks_preserves_non_text(self, file_format):
        """Test that mixed content blocks (text + image) evict text but preserve image blocks."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format=file_format)
        middleware = FilesystemMiddleware(backend=be, tool_token_limit_before_evict=100)
        runtime = _runtime("test_mixed")

        image_block = {"type": "image", "url": "https://example.com/image.png"}
        content_blocks = [
            {"type": "text", "text": "Some text " * 200},
            image_block,
        ]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_mixed")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, ToolMessage)
        item = mem_store.get(("filesystem",), "/large_tool_results/test_mixed")
        assert item is not None
        file_content = item.value["content"]
        text = "\n".join(file_content) if file_format == "v1" else file_content
        assert text.startswith("Some text")

        returned_content = result.content
        assert isinstance(returned_content, list)
        assert len(returned_content) == 2
        assert returned_content[0]["type"] == "text"
        assert "Tool result too large" in returned_content[0]["text"]
        assert returned_content[1] == image_block

    def test_mixed_content_small_text_large_image_not_evicted(self):
        """Test that text+image content is not evicted when only the image is large."""
        backend, _ = _make_backend()
        middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=1000)
        runtime = _runtime("test_no_evict")

        content_blocks = [
            {"type": "text", "text": "small text"},
            {"type": "image", "base64": "x" * 50000, "mime_type": "image/png"},
        ]
        tool_message = ToolMessage(content=content_blocks, tool_call_id="test_no_evict")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert result == tool_message

    def test_read_file_image_returns_standard_image_content_block(self):
        """Test image reads return standard image blocks with base64 + mime_type."""

        class ImageBackend(StateBackend):
            def read(self, path, *, offset=0, limit=100):
                return ReadResult(
                    file_data={
                        "content": "<base64_data>",
                        "encoding": "base64",
                    }
                )

        middleware = FilesystemMiddleware(backend=ImageBackend())
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="img-read-1",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )

        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        result = read_file_tool.invoke({"file_path": "/app/screenshot.png", "runtime": runtime})

        assert isinstance(result, ToolMessage)
        assert result.name == "read_file"
        assert result.tool_call_id == "img-read-1"
        assert result.additional_kwargs["read_file_path"] == "/app/screenshot.png"
        assert result.additional_kwargs["read_file_media_type"] == "image/png"
        assert isinstance(result.content, list)
        assert result.content[0]["type"] == "image"
        assert result.content[0]["mime_type"] == "image/png"
        assert result.content[0]["base64"] == "<base64_data>"

    def test_read_file_image_returns_error_when_download_fails(self):
        """Image reads should return a clear backend error string."""

        class ImageBackend(StateBackend):
            def read(self, path, *, offset=0, limit=100):
                return ReadResult(error="file_not_found")

        middleware = FilesystemMiddleware(backend=ImageBackend())
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="img-read-err",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )

        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        result = read_file_tool.invoke({"file_path": "/app/missing.png", "runtime": runtime})

        assert isinstance(result, str)
        assert result == "Error: file_not_found"

    def test_read_file_handles_str_from_backend(self):
        """Test that read_file works when backend.read() returns a plain str."""

        class StrReadBackend(StateBackend):
            def read(self, path, *, offset=0, limit=100):
                return "     1\tline one\n     2\tline two"

        middleware = FilesystemMiddleware(backend=StrReadBackend())
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="str-read",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )

        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        with pytest.warns(DeprecationWarning, match="Returning a plain `str`"):
            result = read_file_tool.invoke({"file_path": "/app/file.txt", "runtime": runtime})

        assert isinstance(result, str)
        assert "line one" in result

    def test_read_file_str_backend_line_limit_truncation(self):
        """Legacy str backend respects the line-count limit."""

        class StrReadBackend(StateBackend):
            def read(self, path, *, offset=0, limit=100):
                return "\n".join(f"{i:6d}\tline {i}" for i in range(1, 201))

        middleware = FilesystemMiddleware(backend=StrReadBackend())
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="str-trunc",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )

        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        with pytest.warns(DeprecationWarning, match="Returning a plain `str`"):
            result = read_file_tool.invoke({"file_path": "/app/big.txt", "limit": 50, "runtime": runtime})

        assert isinstance(result, str)
        output_lines = [ln for ln in result.splitlines() if ln.strip()]
        assert len(output_lines) <= 50

    def test_read_file_str_backend_token_truncation(self):
        """Legacy str backend applies token-based truncation for huge content."""
        token_limit = 500

        class StrReadBackend(StateBackend):
            def read(self, path, *, offset=0, limit=100):
                return "x" * (NUM_CHARS_PER_TOKEN * token_limit + 1000)

        middleware = FilesystemMiddleware(
            backend=StrReadBackend(),
            tool_token_limit_before_evict=token_limit,
        )
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="str-tok",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )

        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        with pytest.warns(DeprecationWarning, match="Returning a plain `str`"):
            result = read_file_tool.invoke({"file_path": "/app/huge.txt", "runtime": runtime})

        assert isinstance(result, str)
        assert "Output was truncated due to size limits" in result
        assert len(result) <= NUM_CHARS_PER_TOKEN * token_limit

    def test_read_file_empty_file_returns_warning(self):
        """ReadResult with empty content returns the empty-content warning."""
        backend, _ = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        runtime = _runtime("empty-read")

        backend.write("/empty.txt", "")

        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        result = read_file_tool.invoke({"file_path": "/empty.txt", "runtime": runtime})

        assert isinstance(result, str)
        assert result == EMPTY_CONTENT_WARNING

    def test_execute_tool_returns_error_when_backend_doesnt_support(self):
        """Test that execute tool returns friendly error instead of raising exception."""
        backend, _ = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)

        # Find the execute tool
        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")

        # Create runtime with StoreBackend
        runtime = ToolRuntime(
            state={},
            context=None,
            tool_call_id="test_exec",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        # Execute should return error message, not raise exception
        result = execute_tool.invoke({"command": "ls -la", "runtime": runtime})

        assert isinstance(result, str)
        assert "Error: Execution not available" in result
        assert "does not support command execution" in result

    def test_intercept_truncates_content_sample_lines(self):
        """Test that content sample shows head and tail with truncation notice and lines limited to 1000 chars."""
        backend, _ = _make_backend()
        middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=1000)
        runtime = _runtime("test_123")

        # Create content with 15 lines (more than head_lines + tail_lines = 10) to trigger truncation
        # Some lines are longer than 1000 chars to test line truncation
        lines_content = [
            "line 0",
            "a" * 2000,  # Long line in head
            "line 2",
            "line 3",
            "line 4",
            "line 5",  # This will be truncated
            "line 6",
            "line 7",
            "line 8",
            "line 9",
            "line 10",
            "line 11",
            "b" * 2000,  # Long line in tail
            "line 13",
            "line 14",
        ]
        large_content = "\n".join(lines_content)

        tool_message = ToolMessage(content=large_content, tool_call_id="test_123")
        result = middleware._intercept_large_tool_result(tool_message, runtime)

        assert isinstance(result, ToolMessage)
        content_sample_section = result.content

        # Verify the message contains the expected structure with head and tail
        assert "Tool result too large" in content_sample_section
        assert "head and tail" in content_sample_section

        # Verify truncation notice is present
        assert "lines truncated" in content_sample_section
        assert "[5 lines truncated]" in content_sample_section

        # Verify head lines are present (lines 0-4)
        assert "line 0" in content_sample_section
        assert "line 4" in content_sample_section

        # Verify tail lines are present (lines 10-14)
        assert "line 10" in content_sample_section
        assert "line 14" in content_sample_section

        # Verify middle lines are NOT present (lines 5-9)
        assert "line 5" not in content_sample_section
        assert "line 9" not in content_sample_section

        # Check each line in the content sample doesn't exceed 1000 chars
        lines = content_sample_section.split("\n")
        for line in lines:
            if line.strip() and "truncated" not in line:  # Skip empty lines and truncation notice
                assert len(line) <= 1010, f"Line exceeds 1000 chars: {len(line)} chars"

    def test_content_preview_edge_cases(self, num_lines, should_truncate):
        """Test _create_content_preview with various line counts."""
        # Create content with specified number of lines
        if num_lines == 0:
            content_str = ""
        else:
            lines = [f"line {i}" for i in range(num_lines)]
            content_str = "\n".join(lines)

        preview = _create_content_preview(content_str)

        if should_truncate:
            # Should have truncation notice
            assert "truncated" in preview
            # Should have head lines (0-4)
            assert "line 0" in preview
            assert "line 4" in preview
            # Should have tail lines
            assert f"line {num_lines - 5}" in preview
            assert f"line {num_lines - 1}" in preview
            # Should NOT have middle lines
            if num_lines > 11:
                assert "line 5" not in preview
                assert f"line {num_lines - 6}" not in preview
        else:
            # Should NOT have truncation notice
            assert "truncated" not in preview
            # Should have all lines
            for i in range(num_lines):
                assert f"line {i}" in preview

    def test_truncate_list_result_no_truncation(self):
        items = ["/file1.py", "/file2.py", "/file3.py"]
        result = truncate_if_too_long(items)
        assert result == items

    def test_truncate_list_result_with_truncation(self):
        # Create a list that exceeds the token limit (20000 tokens * 4 chars = 80000 chars)
        large_items = [f"/very_long_file_path_{'x' * 100}_{i}.py" for i in range(1000)]
        result = truncate_if_too_long(large_items)

        # Should be truncated
        assert len(result) < len(large_items)
        # Last item should be the truncation message
        assert "results truncated" in result[-1]
        assert "try being more specific" in result[-1]

    def test_truncate_string_result_no_truncation(self):
        content = "short content"
        result = truncate_if_too_long(content)
        assert result == content

    def test_truncate_string_result_with_truncation(self):
        # Create string that exceeds the token limit (20000 tokens * 4 chars = 80000 chars)
        large_content = "x" * 100000
        result = truncate_if_too_long(large_content)

        # Should be truncated
        assert len(result) < len(large_content)
        # Should end with truncation message
        assert "results truncated" in result
        assert "try being more specific" in result

    def test_execute_tool_forwards_zero_timeout_to_backend(self):
        """Middleware should forward timeout=0 for backends that support no-timeout."""
        captured_timeout = {}

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                captured_timeout["value"] = timeout
                return ExecuteResponse(output="ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_zero_timeout",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox()
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = execute_tool.invoke({"command": "echo hello", "timeout": 0, "runtime": rt})

        assert isinstance(result, str)
        assert "ok" in result
        assert captured_timeout["value"] == 0

    def test_execute_tool_rejects_negative_timeout(self):
        """Middleware should return a friendly error for negative timeout."""

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(output="ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_neg_timeout",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox()
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = execute_tool.invoke({"command": "echo hello", "timeout": -5, "runtime": rt})

        assert isinstance(result, str)
        assert "error" in result.lower()
        assert "non-negative" in result.lower()

    def test_execute_tool_forwards_valid_timeout_to_backend(self):
        """Middleware should forward a valid timeout to the backend."""
        captured_timeout = {}

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                captured_timeout["value"] = timeout
                return ExecuteResponse(output="ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_fwd_timeout",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox()
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        execute_tool.invoke({"command": "echo hello", "timeout": 300, "runtime": rt})

        assert captured_timeout["value"] == 300

    def test_execute_tool_rejects_timeout_exceeding_max(self):
        """Middleware should return a friendly error when timeout exceeds max_execute_timeout."""

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(output="ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_max_execute_timeout",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox()
        middleware = FilesystemMiddleware(backend=backend, max_execute_timeout=600)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = execute_tool.invoke({"command": "echo hello", "timeout": 601, "runtime": rt})

        assert isinstance(result, str)
        assert "error" in result.lower()
        assert "601" in result
        assert "600" in result

    def test_execute_tool_accepts_timeout_at_max(self):
        """Middleware should accept timeout exactly equal to max_execute_timeout."""
        captured_timeout = {}

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                captured_timeout["value"] = timeout
                return ExecuteResponse(output="ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_at_max_execute_timeout",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox()
        middleware = FilesystemMiddleware(backend=backend, max_execute_timeout=300)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        execute_tool.invoke({"command": "echo hello", "timeout": 300, "runtime": rt})

        assert captured_timeout["value"] == 300

    def test_execute_tool_none_timeout_skips_max_check(self):
        """Middleware should not reject None timeout against max_execute_timeout."""
        captured_timeout = {}

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                captured_timeout["value"] = timeout
                return ExecuteResponse(output="ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_none_timeout",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox()
        middleware = FilesystemMiddleware(backend=backend, max_execute_timeout=10)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        execute_tool.invoke({"command": "echo hello", "runtime": rt})

        # None should be forwarded without max_execute_timeout rejection
        assert captured_timeout["value"] is None


# --- libs/deepagents/tests/unit_tests/test_middleware_async.py ---

    async def test_als_shortterm(self):
        """Test async ls tool with state backend."""
        files = {
            "/test.txt": FileData(
                content=["Hello world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/test2.txt": FileData(
                content=["Goodbye world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result = await ls_tool.ainvoke({"runtime": _runtime(), "path": "/"})
        assert result.content == str(["/test.txt", "/test2.txt"])

    async def test_als_shortterm_with_path(self):
        """Test async ls tool with specific path."""
        files = {
            "/test.txt": FileData(
                content=["Hello world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/pokemon/test2.txt": FileData(
                content=["Goodbye world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/pokemon/charmander.txt": FileData(
                content=["Ember"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/pokemon/water/squirtle.txt": FileData(
                content=["Water"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result = await ls_tool.ainvoke(
            {
                "path": "/pokemon/",
                "runtime": _runtime(),
            }
        )
        # ls should only return files directly in /pokemon/, not in subdirectories
        assert "/pokemon/test2.txt" in result.content
        assert "/pokemon/charmander.txt" in result.content
        assert "/pokemon/water/squirtle.txt" not in result.content  # In subdirectory
        assert "/pokemon/water/" in result.content

    async def test_als_shortterm_lists_directories(self):
        """Test async ls lists directories with trailing /."""
        files = {
            "/test.txt": FileData(
                content=["Hello world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/pokemon/charmander.txt": FileData(
                content=["Ember"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/pokemon/water/squirtle.txt": FileData(
                content=["Water"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/docs/readme.md": FileData(
                content=["Documentation"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result = await ls_tool.ainvoke(
            {
                "path": "/",
                "runtime": _runtime(),
            }
        )
        # ls should list both files and directories at root level
        assert "/test.txt" in result.content
        assert "/pokemon/" in result.content
        assert "/docs/" in result.content
        # But NOT subdirectory files
        assert "/pokemon/charmander.txt" not in result.content
        assert "/pokemon/water/squirtle.txt" not in result.content

    async def test_aglob_search_shortterm_simple_pattern(self):
        """Test async glob with simple pattern."""
        files = {
            "/test.txt": FileData(
                content=["Hello world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/test.py": FileData(
                content=["print('hello')"],
                modified_at="2021-01-02",
                created_at="2021-01-01",
            ),
            "/pokemon/charmander.py": FileData(
                content=["Ember"],
                modified_at="2021-01-03",
                created_at="2021-01-01",
            ),
            "/pokemon/squirtle.txt": FileData(
                content=["Water"],
                modified_at="2021-01-04",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "*.py",
                "runtime": _runtime(),
            }
        )
        # Standard glob: *.py only matches files in root directory, not subdirectories
        assert result.content == str(["/test.py"])

    async def test_aglob_search_shortterm_wildcard_pattern(self):
        """Test async glob with wildcard pattern."""
        files = {
            "/src/main.py": FileData(
                content=["main code"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/src/utils/helper.py": FileData(
                content=["helper code"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/tests/test_main.py": FileData(
                content=["test code"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "**/*.py",
                "runtime": _runtime(),
            }
        )
        assert "/src/main.py" in result.content
        assert "/src/utils/helper.py" in result.content
        assert "/tests/test_main.py" in result.content

    async def test_aglob_search_shortterm_with_path(self):
        """Test async glob with specific path."""
        files = {
            "/src/main.py": FileData(
                content=["main code"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/src/utils/helper.py": FileData(
                content=["helper code"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/tests/test_main.py": FileData(
                content=["test code"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "*.py",
                "path": "/src",
                "runtime": _runtime(),
            }
        )
        assert "/src/main.py" in result.content
        assert "/src/utils/helper.py" not in result.content
        assert "/tests/test_main.py" not in result.content

    async def test_aglob_search_shortterm_no_matches(self):
        """Test async glob with no matches."""
        files = {
            "/test.txt": FileData(
                content=["Hello world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        glob_search_tool = next(tool for tool in middleware.tools if tool.name == "glob")
        result = await glob_search_tool.ainvoke(
            {
                "pattern": "*.py",
                "runtime": _runtime(),
            }
        )
        assert result.content == str([])

    async def test_agrep_search_shortterm_files_with_matches(self):
        """Test async grep with files_with_matches mode."""
        files = {
            "/test.py": FileData(
                content=["import os", "import sys", "print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/main.py": FileData(
                content=["def main():", "    pass"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/helper.txt": FileData(
                content=["import json"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "runtime": _runtime(),
            }
        )
        assert "/test.py" in result.content
        assert "/helper.txt" in result.content
        assert "/main.py" not in result.content

    async def test_agrep_search_shortterm_content_mode(self):
        """Test async grep with content mode."""
        files = {
            "/test.py": FileData(
                content=["import os", "import sys", "print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "output_mode": "content",
                "runtime": _runtime(),
            }
        )
        assert "1: import os" in result.content
        assert "2: import sys" in result.content
        assert "print" not in result.content

    async def test_agrep_search_shortterm_count_mode(self):
        """Test async grep with count mode."""
        files = {
            "/test.py": FileData(
                content=["import os", "import sys", "print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/main.py": FileData(
                content=["import json", "data = {}"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "output_mode": "count",
                "runtime": _runtime(),
            }
        )
        assert "/test.py:2" in result.content or "/test.py: 2" in result.content
        assert "/main.py:1" in result.content or "/main.py: 1" in result.content

    async def test_agrep_search_shortterm_with_include(self):
        """Test async grep with glob filter."""
        files = {
            "/test.py": FileData(
                content=["import os"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/test.txt": FileData(
                content=["import nothing"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "glob": "*.py",
                "runtime": _runtime(),
            }
        )
        assert "/test.py" in result.content
        assert "/test.txt" not in result.content

    async def test_agrep_search_shortterm_with_path(self):
        """Test async grep with specific path."""
        files = {
            "/src/main.py": FileData(
                content=["import os"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
            "/tests/test.py": FileData(
                content=["import pytest"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "path": "/src",
                "runtime": _runtime(),
            }
        )
        assert "/src/main.py" in result.content
        assert "/tests/test.py" not in result.content

    async def test_agrep_search_shortterm_regex_pattern(self):
        """Test async grep with literal pattern (not regex)."""
        files = {
            "/test.py": FileData(
                content=["def hello():", "def world():", "x = 5"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        # Search for literal "def " - literal search, not regex
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "def ",
                "output_mode": "content",
                "runtime": _runtime(),
            }
        )
        assert "1: def hello():" in result.content
        assert "2: def world():" in result.content
        assert "x = 5" not in result.content

    async def test_agrep_search_shortterm_no_matches(self):
        """Test async grep with no matches."""
        files = {
            "/test.py": FileData(
                content=["print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "import",
                "runtime": _runtime(),
            }
        )
        assert result.content == "No matches found"

    async def test_agrep_search_shortterm_invalid_regex(self):
        """Test async grep with special characters (literal search, not regex)."""
        files = {
            "/test.py": FileData(
                content=["print('hello')"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        grep_search_tool = next(tool for tool in middleware.tools if tool.name == "grep")
        # Special characters are treated literally, so no matches expected
        result = await grep_search_tool.ainvoke(
            {
                "pattern": "[invalid",
                "runtime": _runtime(),
            }
        )
        content = result.content if isinstance(result, ToolMessage) else result
        assert "No matches found" in content

    async def test_aread_file(self):
        """Test async read_file tool."""
        files = {
            "/test.txt": FileData(
                content=["Hello world", "Line 2", "Line 3"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        result = await read_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "runtime": _runtime(),
            }
        )
        assert "Hello world" in result
        assert "Line 2" in result
        assert "Line 3" in result

    async def test_aread_file_with_offset(self):
        """Test async read_file tool with offset."""
        files = {
            "/test.txt": FileData(
                content=["Line 1", "Line 2", "Line 3", "Line 4"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, _ = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        read_file_tool = next(tool for tool in middleware.tools if tool.name == "read_file")
        result = await read_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "offset": 1,
                "limit": 2,
                "runtime": _runtime(),
            }
        )
        assert "Line 2" in result
        assert "Line 3" in result
        assert "Line 1" not in result
        assert "Line 4" not in result

    async def test_awrite_file(self):
        """Test async write_file tool."""
        backend, mem_store = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_file_tool = next(tool for tool in middleware.tools if tool.name == "write_file")
        result = await write_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "content": "Hello world",
                "runtime": ToolRuntime(state={}, context=None, tool_call_id="tc1", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        # StoreBackend writes to the store and returns a plain string
        assert isinstance(result, str)
        assert mem_store.get(("filesystem",), "/test.txt") is not None

    async def test_aedit_file(self):
        """Test async edit_file tool."""
        files = {
            "/test.txt": FileData(
                content=["Hello world", "Goodbye world"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, mem_store = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        edit_file_tool = next(tool for tool in middleware.tools if tool.name == "edit_file")
        result = await edit_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "old_string": "Hello",
                "new_string": "Hi",
                "runtime": ToolRuntime(state={}, context=None, tool_call_id="tc2", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        # StoreBackend writes to the store and returns a plain string
        assert isinstance(result, str)
        assert mem_store.get(("filesystem",), "/test.txt") is not None

    async def test_aedit_file_replace_all(self):
        """Test async edit_file tool with replace_all."""
        files = {
            "/test.txt": FileData(
                content=["Hello world", "Hello again"],
                modified_at="2021-01-01",
                created_at="2021-01-01",
            ),
        }
        backend, mem_store = _make_backend(files)
        middleware = FilesystemMiddleware(backend=backend)
        edit_file_tool = next(tool for tool in middleware.tools if tool.name == "edit_file")
        result = await edit_file_tool.ainvoke(
            {
                "file_path": "/test.txt",
                "old_string": "Hello",
                "new_string": "Hi",
                "replace_all": True,
                "runtime": ToolRuntime(state={}, context=None, tool_call_id="tc3", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert isinstance(result, str)
        assert mem_store.get(("filesystem",), "/test.txt") is not None

    async def test_aexecute_tool_returns_error_when_backend_doesnt_support(self):
        """Test async execute tool returns friendly error instead of raising exception."""
        backend, _ = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)

        # Find the execute tool
        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")

        # Create runtime with StoreBackend
        runtime = ToolRuntime(
            state={},
            context=None,
            tool_call_id="test_exec",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        # Execute should return error message, not raise exception
        result = await execute_tool.ainvoke({"command": "ls -la", "runtime": runtime})

        assert isinstance(result, str)
        assert "Error: Execution not available" in result
        assert "does not support command execution" in result

    async def test_aexecute_tool_forwards_zero_timeout_to_backend(self):
        """Async execute tool should forward timeout=0 for no-timeout backends."""
        captured_timeout = {}

        class TimeoutCaptureSandbox(SandboxBackendProtocol, StateBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                return ExecuteResponse(output="sync ok", exit_code=0, truncated=False)

            async def aexecute(
                self,
                command: str,
                *,
                timeout: int | None = None,  # noqa: ASYNC109
            ) -> ExecuteResponse:
                captured_timeout["value"] = timeout
                return ExecuteResponse(output="async ok", exit_code=0, truncated=False)

            @property
            def id(self):
                return "timeout-capture-sandbox-backend"

        state = FilesystemState(messages=[], files={})
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_zero_timeout_async",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        backend = TimeoutCaptureSandbox()
        middleware = FilesystemMiddleware(backend=backend)

        execute_tool = next(tool for tool in middleware.tools if tool.name == "execute")
        result = await execute_tool.ainvoke({"command": "echo hello", "timeout": 0, "runtime": rt})

        assert "async ok" in result
        assert captured_timeout["value"] == 0


# --- libs/deepagents/tests/unit_tests/test_permissions.py ---

    def test_read_denied_on_restricted_path(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(read_tool, {"file_path": "/secrets/key.txt"}, rules)
        assert "permission denied" in result
        assert "read" in result

    def test_read_allowed_on_permitted_path(self):
        backend = _make_backend({"/workspace/file.txt": "hello"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(read_tool, {"file_path": "/workspace/file.txt"}, rules)
        assert "permission denied" not in result

    def test_write_denied_on_restricted_path(self):
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        result = _invoke_with_permissions(write_tool, {"file_path": "/foo.txt", "content": "data"}, rules)
        assert "permission denied" in result
        assert "write" in result

    def test_edit_denied_on_restricted_path(self):
        backend = _make_backend({"/protected/file.txt": "original"})
        middleware = FilesystemMiddleware(backend=backend)
        edit_tool = next(t for t in middleware.tools if t.name == "edit_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/protected/**"], mode="deny")]
        result = _invoke_with_permissions(
            edit_tool,
            {
                "file_path": "/protected/file.txt",
                "old_string": "original",
                "new_string": "changed",
            },
            rules,
        )
        assert "permission denied" in result

    def test_ls_filters_denied_results(self):
        backend = _make_backend(
            {
                "/public/a.txt": "pub",
                "/secrets/b.txt": "priv",
            }
        )
        # Deny the /secrets/ directory entry itself so it's filtered from ls output
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(t for t in middleware.tools if t.name == "ls")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets/", "/secrets"], mode="deny")]
        # ls /secrets directly should be denied (pre-check on the queried path)
        result_secrets = _invoke_with_permissions(ls_tool, {"path": "/secrets"}, rules)
        assert "permission denied" in result_secrets

    def test_ls_no_filter_when_all_allowed(self):
        backend = _make_backend({"/public/a.txt": "pub", "/public/b.txt": "pub2"})
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(t for t in middleware.tools if t.name == "ls")
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        result = _invoke_with_permissions(ls_tool, {"path": "/"}, rules)
        assert "/public" in result

    def test_no_rules_allows_everything(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = read_tool.invoke({"runtime": _runtime(), "file_path": "/secrets/key.txt"})
        assert "permission denied" not in result

    def test_ls_denied_on_restricted_root(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(t for t in middleware.tools if t.name == "ls")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = _invoke_with_permissions(ls_tool, {"path": "/secrets"}, rules)
        assert "permission denied" in result

    def test_ls_post_filters_denied_children(self):
        backend = _make_backend(
            {
                "/public/a.txt": "pub",
                "/secrets/b.txt": "priv",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(t for t in middleware.tools if t.name == "ls")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(ls_tool, {"path": "/"}, rules)
        assert "/secrets" not in result
        assert "/public" in result

    def test_deny_read_allows_write(self):
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/vault/**"], mode="deny")]
        result = _invoke_with_permissions(write_tool, {"file_path": "/vault/file.txt", "content": "data"}, rules)
        assert "permission denied" not in result

    def test_dotdot_traversal_blocked_by_validate_path(self):
        # validate_path rejects .. before permission checking even runs,
        # so traversal is blocked regardless of permission rules.
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = _invoke_with_permissions(read_tool, {"file_path": "/workspace/../secrets/key.txt"}, rules)
        assert "Path traversal not allowed" in result

    def test_dotdot_traversal_blocked_even_without_permission_rules(self):
        # Traversal is rejected by validate_path even when no permission rules are set.
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = read_tool.invoke({"runtime": _runtime(), "file_path": "/workspace/../secrets/key.txt"})
        assert "Path traversal not allowed" in result

    def test_redundant_separators_normalized(self):
        # /secrets//key.txt is normalized by validate_path to /secrets/key.txt
        # and then caught by the permission rule.
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = _invoke_with_permissions(read_tool, {"file_path": "/secrets//key.txt"}, rules)
        assert "permission denied" in result

    def test_dotdot_write_traversal_blocked_by_validate_path(self):
        # validate_path rejects .. on write paths too.
        rules = [FilesystemPermission(operations=["write"], paths=["/restricted/**"], mode="deny")]
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        result = _invoke_with_permissions(write_tool, {"file_path": "/workspace/../restricted/file.txt", "content": "data"}, rules)
        assert "Path traversal not allowed" in result

    def test_non_traversal_path_still_allowed(self):
        # Verify that normal paths are not affected by the canonicalization logic.
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        backend = _make_backend({"/workspace/safe.txt": "safe content"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        result = _invoke_with_permissions(read_tool, {"file_path": "/workspace/safe.txt"}, rules)
        assert "permission denied" not in result
        assert "Path traversal" not in result

    def test_glob_denied_on_restricted_base_path(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = _invoke_with_permissions(glob_tool, {"pattern": "*.txt", "path": "/secrets"}, rules)
        assert "permission denied" in result
        assert "read" in result

    def test_glob_allowed_on_unrestricted_base_path(self):
        backend = _make_backend({"/workspace/file.txt": "hello"})
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(glob_tool, {"pattern": "*.txt", "path": "/workspace"}, rules)
        assert "permission denied" not in result

    def test_glob_filters_denied_results(self):
        backend = _make_backend(
            {
                "/public/a.txt": "pub",
                "/secrets/b.txt": "priv",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(glob_tool, {"pattern": "**/*.txt", "path": "/"}, rules)
        assert "/secrets/b.txt" not in result
        assert "/public/a.txt" in result
        assert "/secrets" not in result

    def test_glob_no_filter_annotation_when_all_allowed(self):
        backend = _make_backend({"/public/a.txt": "pub", "/public/b.txt": "pub2"})
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        result = _invoke_with_permissions(glob_tool, {"pattern": "**/*.txt", "path": "/"}, rules)
        assert "permission denied" not in result

    async def test_glob_denied_on_restricted_base_path_async(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = await _ainvoke_with_permissions(glob_tool, {"pattern": "*.txt", "path": "/secrets"}, rules)
        assert "permission denied" in result
        assert "read" in result

    async def test_glob_filters_denied_results_async(self):
        backend = _make_backend(
            {
                "/public/a.txt": "pub",
                "/secrets/b.txt": "priv",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        glob_tool = next(t for t in middleware.tools if t.name == "glob")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(glob_tool, {"pattern": "**/*.txt", "path": "/"}, rules)
        assert "/secrets/b.txt" not in result
        assert "/public/a.txt" in result
        assert "/secrets" not in result

    def test_grep_denied_on_restricted_path(self):
        backend = _make_backend({"/secrets/key.txt": "top secret data"})
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = _invoke_with_permissions(grep_tool, {"pattern": "secret", "path": "/secrets"}, rules)
        assert "permission denied" in result
        assert "read" in result

    def test_grep_dotdot_traversal_blocked_by_validate_path(self):
        """Grep rejects ../ traversal via validate_path before the permission check runs."""
        backend = _make_backend({"/secrets/key.txt": "top secret data"})
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = _invoke_with_permissions(grep_tool, {"pattern": "secret", "path": "/workspace/../secrets"}, rules)
        assert "Path traversal not allowed" in result

    def test_grep_allowed_on_unrestricted_path(self):
        backend = _make_backend({"/workspace/file.txt": "hello world"})
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(grep_tool, {"pattern": "hello", "path": "/workspace"}, rules)
        assert "permission denied" not in result

    def test_grep_filters_denied_results_from_matches(self):
        backend = _make_backend(
            {
                "/public/a.txt": "keyword here",
                "/secrets/b.txt": "keyword there",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(grep_tool, {"pattern": "keyword"}, rules)
        assert "/secrets/b.txt" not in result
        assert "/public/a.txt" in result
        assert "/secrets" not in result

    def test_grep_no_filter_annotation_when_all_allowed(self):
        backend = _make_backend({"/public/a.txt": "keyword", "/public/b.txt": "keyword"})
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        result = _invoke_with_permissions(grep_tool, {"pattern": "keyword"}, rules)
        assert "permission denied" not in result

    def test_grep_path_none_bypasses_pre_check_but_filters_results(self):
        backend = _make_backend(
            {
                "/public/a.txt": "keyword here",
                "/secrets/b.txt": "keyword there",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = _invoke_with_permissions(grep_tool, {"pattern": "keyword", "path": None}, rules)
        assert "permission denied" not in result
        assert "/secrets/b.txt" not in result
        assert "/public/a.txt" in result
        assert "/secrets" not in result

    async def test_grep_denied_on_restricted_path_async(self):
        backend = _make_backend({"/secrets/key.txt": "top secret data"})
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = await _ainvoke_with_permissions(grep_tool, {"pattern": "secret", "path": "/secrets"}, rules)
        assert "permission denied" in result
        assert "read" in result

    async def test_grep_filters_denied_results_async(self):
        backend = _make_backend(
            {
                "/public/a.txt": "keyword here",
                "/secrets/b.txt": "keyword there",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(grep_tool, {"pattern": "keyword"}, rules)
        assert "/secrets/b.txt" not in result
        assert "/public/a.txt" in result
        assert "/secrets" not in result

    async def test_grep_path_none_bypasses_pre_check_but_filters_results_async(self):
        backend = _make_backend(
            {
                "/public/a.txt": "keyword here",
                "/secrets/b.txt": "keyword there",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        grep_tool = next(t for t in middleware.tools if t.name == "grep")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(grep_tool, {"pattern": "keyword", "path": None}, rules)
        assert "permission denied" not in result
        assert "/secrets/b.txt" not in result
        assert "/public/a.txt" in result
        assert "/secrets" not in result

    async def test_read_denied_async(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(read_tool, {"file_path": "/secrets/key.txt"}, rules)
        assert "permission denied" in result
        assert "read" in result

    async def test_read_allowed_async(self):
        backend = _make_backend({"/workspace/file.txt": "hello"})
        middleware = FilesystemMiddleware(backend=backend)
        read_tool = next(t for t in middleware.tools if t.name == "read_file")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(read_tool, {"file_path": "/workspace/file.txt"}, rules)
        assert "permission denied" not in result

    async def test_write_denied_async(self):
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        result = await _ainvoke_with_permissions(write_tool, {"file_path": "/foo.txt", "content": "data"}, rules)
        assert "permission denied" in result
        assert "write" in result

    async def test_write_allowed_async(self):
        backend = _make_backend()
        middleware = FilesystemMiddleware(backend=backend)
        write_tool = next(t for t in middleware.tools if t.name == "write_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(write_tool, {"file_path": "/workspace/file.txt", "content": "data"}, rules)
        assert "permission denied" not in result

    async def test_edit_denied_async(self):
        backend = _make_backend({"/protected/file.txt": "original"})
        middleware = FilesystemMiddleware(backend=backend)
        edit_tool = next(t for t in middleware.tools if t.name == "edit_file")
        rules = [FilesystemPermission(operations=["write"], paths=["/protected/**"], mode="deny")]
        result = await _ainvoke_with_permissions(
            edit_tool,
            {
                "file_path": "/protected/file.txt",
                "old_string": "original",
                "new_string": "changed",
            },
            rules,
        )
        assert "permission denied" in result

    async def test_ls_denied_async(self):
        backend = _make_backend({"/secrets/key.txt": "top secret"})
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(t for t in middleware.tools if t.name == "ls")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**", "/secrets"], mode="deny")]
        result = await _ainvoke_with_permissions(ls_tool, {"path": "/secrets"}, rules)
        assert "permission denied" in result

    async def test_ls_filters_denied_results_async(self):
        backend = _make_backend(
            {
                "/public/a.txt": "pub",
                "/secrets/b.txt": "priv",
            }
        )
        middleware = FilesystemMiddleware(backend=backend)
        ls_tool = next(t for t in middleware.tools if t.name == "ls")
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny")]
        result = await _ainvoke_with_permissions(ls_tool, {"path": "/"}, rules)
        assert "/secrets/b.txt" not in result


# --- libs/deepagents/tests/unit_tests/backends/test_composite_backend.py ---

def test_composite_backend_intercept_large_tool_result(file_format):
    mem_store = InMemoryStore()
    rt = make_runtime("t10", store=mem_store)

    middleware = FilesystemMiddleware(
        backend=CompositeBackend(
            default=StoreBackend(store=mem_store, namespace=lambda _rt: ("default",), file_format=file_format),
            routes={"/memories/": StoreBackend(store=mem_store, namespace=lambda _rt: ("memories",))},
        ),
        tool_token_limit_before_evict=1000,
    )
    large_content = "z" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_789")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    # Verify the file was written to the default store backend
    stored_item = mem_store.get(("default",), "/large_tool_results/test_789")
    assert stored_item is not None
    expected = [large_content] if file_format == "v1" else large_content
    assert stored_item.value["content"] == expected

def test_composite_backend_intercept_large_tool_result_routed_to_store(file_format):
    """Test that large tool results can be routed to a specific backend like StoreBackend."""
    mem_store = InMemoryStore()
    rt = make_runtime("t11", store=mem_store)

    middleware = FilesystemMiddleware(
        backend=CompositeBackend(
            default=StoreBackend(store=mem_store, namespace=lambda _rt: ("default",), file_format=file_format),
            routes={"/large_tool_results/": StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format=file_format)},
        ),
        tool_token_limit_before_evict=1000,
    )

    large_content = "w" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_routed_123")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_routed_123" in result.content

    stored_item = mem_store.get(("filesystem",), "/test_routed_123")
    assert stored_item is not None
    expected = [large_content] if file_format == "v1" else large_content
    assert stored_item.value["content"] == expected

def test_composite_grep_multiple_routes_aggregation(tmp_path: Path) -> None:
    """Test grep aggregates results from multiple routed backends with expected isolation.

    This test represents the intuitive expected behavior: files written to /memories/
    should only appear in /memories/, and files written to /archive/ should only appear
    in /archive/.
    """
    root = tmp_path

    (root / "default.txt").write_text("default findme")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    store1 = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))
    store2 = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    comp = CompositeBackend(default=fs, routes={"/memories/": store1, "/archive/": store2})

    # Write to each route
    comp.write("/memories/mem.txt", "memory findme")
    comp.write("/archive/arch.txt", "archive findme")

    # Grep across all backends
    matches = comp.grep("findme", path="/").matches
    assert matches is not None
    match_paths = sorted([m["path"] for m in matches])

    # Expected: each file appears only in its own route
    expected_paths = sorted(
        [
            "/archive/arch.txt",
            "/default.txt",
            "/memories/mem.txt",
        ]
    )
    assert match_paths == expected_paths


# --- libs/deepagents/tests/unit_tests/backends/test_filesystem_backend.py ---

def test_filesystem_backend_intercept_large_tool_result(tmp_path: Path):
    """Test that FilesystemBackend properly handles large tool result interception."""
    root = tmp_path
    rt = ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id="test_fs",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )

    middleware = FilesystemMiddleware(backend=FilesystemBackend(root_dir=str(root), virtual_mode=True), tool_token_limit_before_evict=1000)

    large_content = "f" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_fs_123")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_fs_123" in result.content
    saved_file = root / "large_tool_results" / "test_fs_123"
    assert saved_file.exists()
    assert saved_file.read_text() == large_content


# --- libs/deepagents/tests/unit_tests/backends/test_filesystem_backend_async.py ---

async def test_filesystem_backend_intercept_large_tool_result_async(tmp_path: Path):
    """Test that FilesystemBackend properly handles large tool result interception in async context."""
    root = tmp_path
    rt = ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id="test_fs",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )

    middleware = FilesystemMiddleware(backend=FilesystemBackend(root_dir=str(root), virtual_mode=True), tool_token_limit_before_evict=1000)

    large_content = "f" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_fs_123")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_fs_123" in result.content
    saved_file = root / "large_tool_results" / "test_fs_123"
    assert saved_file.exists()
    assert saved_file.read_text() == large_content


# --- libs/deepagents/tests/unit_tests/backends/test_store_backend.py ---

def test_store_backend_intercept_large_tool_result(file_format):
    """Test that StoreBackend properly handles large tool result interception."""
    mem_store = InMemoryStore()
    middleware = FilesystemMiddleware(
        backend=StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format=file_format),
        tool_token_limit_before_evict=1000,
    )

    large_content = "y" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_456")
    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t2",
        store=mem_store,
        stream_writer=lambda _: None,
        config={},
    )
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_456" in result.content

    stored_content = mem_store.get(("filesystem",), "/large_tool_results/test_456")
    assert stored_content is not None
    expected = [large_content] if file_format == "v1" else large_content
    assert stored_content.value["content"] == expected

def test_compat_wrapper_old_style_runtime_access_warns() -> None:
    """Old-style factories accessing .runtime get a deprecation warning."""
    compat = _NamespaceRuntimeCompat(runtime=None)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compat.runtime
        assert result is None
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert ".runtime" in str(w[0].message)
        assert "v0.7" in str(w[0].message)

def test_compat_wrapper_old_style_state_access_warns() -> None:
    """Old-style factories accessing .state get a deprecation warning."""
    compat = _NamespaceRuntimeCompat(runtime=None, state={"messages": []})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compat.state
        assert result == {"messages": []}
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert ".state" in str(w[0].message)
        assert "v0.7" in str(w[0].message)

def test_compat_wrapper_proxies_runtime_attrs() -> None:
    """New-style factories can access Runtime attributes directly through the wrapper."""

    @dataclass
    class Ctx:
        user_id: str

    rt = Runtime(context=Ctx(user_id="alice"))
    compat = _NamespaceRuntimeCompat(runtime=rt)

    # New-style access: no warning, proxied to Runtime
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert compat.context.user_id == "alice"  # type: ignore[union-attr]
        assert compat.store is None  # type: ignore[union-attr]
        # No deprecation warnings for direct Runtime attr access
        assert len(w) == 0

def test_compat_wrapper_old_style_factory_end_to_end() -> None:
    """An old-style namespace factory using ctx.runtime.context still works."""

    @dataclass
    class Ctx:
        user_id: str

    rt = Runtime(context=Ctx(user_id="bob"))
    compat = _NamespaceRuntimeCompat(runtime=rt)

    # Old-style factory
    def old_factory(ctx: BackendContext) -> tuple[str, ...]:  # type: ignore[type-arg]
        return (ctx.runtime.context.user_id, "filesystem")  # type: ignore[union-attr]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = old_factory(compat)  # type: ignore[arg-type]
        assert result == ("bob", "filesystem")
        assert len(w) == 1  # one warning from .runtime access

def test_compat_wrapper_new_style_factory_end_to_end() -> None:
    """A new-style namespace factory using rt.context works without warnings."""
    rt = Runtime(
        context=None,
        server_info=SimpleNamespace(user=SimpleNamespace(identity="carol")),  # type: ignore[arg-type]
    )
    compat = _NamespaceRuntimeCompat(runtime=rt)

    # New-style factory
    def new_factory(rt: Runtime) -> tuple[str, ...]:  # type: ignore[type-arg]
        return (rt.server_info.user.identity, "filesystem")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = new_factory(compat)  # type: ignore[arg-type]
        assert result == ("carol", "filesystem")
        assert len(w) == 0  # no warnings

def test_compat_wrapper_no_runtime_raises_on_attr_access() -> None:
    """Accessing Runtime attrs when runtime is None raises AttributeError."""
    compat = _NamespaceRuntimeCompat(runtime=None)

    with pytest.raises(AttributeError, match="running outside graph execution"):
        _ = compat.context  # type: ignore[union-attr]


# --- libs/deepagents/tests/unit_tests/backends/test_store_backend_async.py ---

async def test_store_backend_intercept_large_tool_result_async(file_format):
    """Test that StoreBackend properly handles large tool result interception in async context."""
    mem_store = InMemoryStore()
    middleware = FilesystemMiddleware(
        backend=StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format=file_format),
        tool_token_limit_before_evict=1000,
    )

    large_content = "y" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_456")
    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t2",
        store=mem_store,
        stream_writer=lambda _: None,
        config={},
    )
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_456" in result.content

    stored_content = mem_store.get(("filesystem",), "/large_tool_results/test_456")
    assert stored_content is not None
    expected = [large_content] if file_format == "v1" else large_content
    assert stored_content.value["content"] == expected


# --- libs/deepagents/tests/unit_tests/middleware/test_memory_middleware.py ---

def test_memory_middleware_with_store_backend_assistant_id() -> None:
    """Test namespace isolation: each assistant_id gets its own memory namespace."""
    # Setup
    store = InMemoryStore()
    middleware = MemoryMiddleware(
        backend=StoreBackend(store=store, namespace=_assistant_id_namespace),
        sources=["/memory/AGENTS.md"],
    )
    runtime = SimpleNamespace(context=None, store=store, stream_writer=lambda _: None)

    # Add memory for assistant-123 with namespace (assistant-123, filesystem)
    assistant_1_content = make_memory_content("Assistant 1", "- Context for assistant 1")
    store.put(
        ("assistant-123", "filesystem"),
        "/memory/AGENTS.md",
        create_store_memory_item(assistant_1_content),
    )

    # Test: assistant-123 can read its own memory
    with _runtime_context("assistant-123"):
        result_1 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_1 is not None
    assert "/memory/AGENTS.md" in result_1["memory_contents"]
    assert "Context for assistant 1" in result_1["memory_contents"]["/memory/AGENTS.md"]

    # Test: assistant-456 cannot see assistant-123's memory (different namespace)
    with _runtime_context("assistant-456"):
        result_2 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]
    assert result_2 is not None
    assert len(result_2["memory_contents"]) == 0

    # Add memory for assistant-456 with namespace (assistant-456, filesystem)
    assistant_2_content = make_memory_content("Assistant 2", "- Context for assistant 2")
    store.put(
        ("assistant-456", "filesystem"),
        "/memory/AGENTS.md",
        create_store_memory_item(assistant_2_content),
    )

    # Test: assistant-456 can read its own memory
    with _runtime_context("assistant-456"):
        result_3 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_3 is not None
    assert "/memory/AGENTS.md" in result_3["memory_contents"]
    assert "Context for assistant 2" in result_3["memory_contents"]["/memory/AGENTS.md"]
    assert "Context for assistant 1" not in result_3["memory_contents"]["/memory/AGENTS.md"]

    # Test: assistant-123 still only sees its own memory (no cross-contamination)
    with _runtime_context("assistant-123"):
        result_4 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_4 is not None
    assert "/memory/AGENTS.md" in result_4["memory_contents"]
    assert "Context for assistant 1" in result_4["memory_contents"]["/memory/AGENTS.md"]
    assert "Context for assistant 2" not in result_4["memory_contents"]["/memory/AGENTS.md"]

def test_memory_middleware_with_store_backend_no_assistant_id() -> None:
    """Test default namespace: when no assistant_id is provided, uses (filesystem,) namespace."""
    # Setup
    store = InMemoryStore()
    middleware = MemoryMiddleware(
        backend=StoreBackend(store=store, namespace=_assistant_id_namespace),
        sources=["/memory/AGENTS.md"],
    )
    runtime = SimpleNamespace(context=None, store=store, stream_writer=lambda _: None)

    # Add memory to default namespace (filesystem,) - no assistant_id
    shared_content = make_memory_content("Shared Memory", "- Default namespace context")
    store.put(
        ("filesystem",),
        "/memory/AGENTS.md",
        create_store_memory_item(shared_content),
    )

    # Test: runtime without server_info accesses default namespace
    with _runtime_context(None):
        result_1 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_1 is not None
    assert "/memory/AGENTS.md" in result_1["memory_contents"]
    assert "Default namespace context" in result_1["memory_contents"]["/memory/AGENTS.md"]

    # Test: runtime with server_info but empty assistant_id also uses default namespace
    with _runtime_context(""):
        result_2 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_2 is not None
    assert "/memory/AGENTS.md" in result_2["memory_contents"]
    assert "Default namespace context" in result_2["memory_contents"]["/memory/AGENTS.md"]


# --- libs/deepagents/tests/unit_tests/middleware/test_skills_middleware.py ---

def test_skills_middleware_with_state_backend() -> None:
    """Test that SkillsMiddleware can be initialized with StateBackend instance."""
    sources = ["/skills/user"]
    middleware = SkillsMiddleware(
        backend=StateBackend(),
        sources=sources,
    )

    # Verify the middleware was created successfully
    assert middleware is not None
    assert isinstance(middleware._backend, StateBackend)
    assert len(middleware.sources) == 1
    assert middleware.sources[0] == "/skills/user"

    runtime = SimpleNamespace(
        context=None,
        store=None,
        stream_writer=lambda _: None,
    )

    backend = middleware._get_backend({"messages": [], "files": {}}, runtime, {})
    assert isinstance(backend, StateBackend)

def test_skills_middleware_with_store_backend_assistant_id() -> None:
    """Test namespace isolation: each assistant_id gets its own skills namespace."""
    store = InMemoryStore()
    middleware = SkillsMiddleware(
        backend=StoreBackend(store=store, namespace=_assistant_id_namespace),
        sources=["/skills/user"],
    )
    runtime = SimpleNamespace(context=None, store=store, stream_writer=lambda _: None)

    # Add skill for assistant-123 with namespace (assistant-123, filesystem)
    assistant_1_skill = make_skill_content("skill-one", "Skill for assistant 1")
    store.put(
        ("assistant-123", "filesystem"),
        "/skills/user/skill-one/SKILL.md",
        create_store_skill_item(assistant_1_skill),
    )

    # Test: assistant-123 can read its own skill
    with _runtime_context("assistant-123"):
        result_1 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_1 is not None
    assert len(result_1["skills_metadata"]) == 1
    assert result_1["skills_metadata"][0]["name"] == "skill-one"
    assert result_1["skills_metadata"][0]["description"] == "Skill for assistant 1"

    # Test: assistant-456 cannot see assistant-123's skill (different namespace)
    with _runtime_context("assistant-456"):
        result_2 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_2 is not None
    assert len(result_2["skills_metadata"]) == 0  # No skills in assistant-456's namespace yet

    # Add skill for assistant-456 with namespace (assistant-456, filesystem)
    assistant_2_skill = make_skill_content("skill-two", "Skill for assistant 2")
    store.put(
        ("assistant-456", "filesystem"),
        "/skills/user/skill-two/SKILL.md",
        create_store_skill_item(assistant_2_skill),
    )

    # Test: assistant-456 can read its own skill
    with _runtime_context("assistant-456"):
        result_3 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_3 is not None
    assert len(result_3["skills_metadata"]) == 1
    assert result_3["skills_metadata"][0]["name"] == "skill-two"
    assert result_3["skills_metadata"][0]["description"] == "Skill for assistant 2"

    # Test: assistant-123 still only sees its own skill (no cross-contamination)
    with _runtime_context("assistant-123"):
        result_4 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_4 is not None
    assert len(result_4["skills_metadata"]) == 1
    assert result_4["skills_metadata"][0]["name"] == "skill-one"
    assert result_4["skills_metadata"][0]["description"] == "Skill for assistant 1"

def test_skills_middleware_with_store_backend_no_assistant_id() -> None:
    """Test default namespace: when no assistant_id is provided, uses (filesystem,) namespace."""
    store = InMemoryStore()
    middleware = SkillsMiddleware(
        backend=StoreBackend(store=store, namespace=_assistant_id_namespace),
        sources=["/skills/user"],
    )
    runtime = SimpleNamespace(context=None, store=store, stream_writer=lambda _: None)

    # Add skill to default namespace (filesystem,) - no assistant_id
    shared_skill = make_skill_content("shared-skill", "Shared namespace skill")
    store.put(
        ("filesystem",),
        "/skills/user/shared-skill/SKILL.md",
        create_store_skill_item(shared_skill),
    )

    # Test: runtime without server_info accesses default namespace
    with _runtime_context(None):
        result_1 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_1 is not None
    assert len(result_1["skills_metadata"]) == 1
    assert result_1["skills_metadata"][0]["name"] == "shared-skill"
    assert result_1["skills_metadata"][0]["description"] == "Shared namespace skill"

    # Test: runtime with server_info but empty assistant_id also uses default namespace
    with _runtime_context(""):
        result_2 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_2 is not None
    assert len(result_2["skills_metadata"]) == 1
    assert result_2["skills_metadata"][0]["name"] == "shared-skill"
    assert result_2["skills_metadata"][0]["description"] == "Shared namespace skill"

async def test_skills_middleware_with_store_backend_assistant_id_async() -> None:
    """Test namespace isolation with async: each assistant_id gets its own skills namespace."""
    store = InMemoryStore()
    middleware = SkillsMiddleware(
        backend=StoreBackend(store=store, namespace=_assistant_id_namespace),
        sources=["/skills/user"],
    )
    runtime = SimpleNamespace(context=None, store=store, stream_writer=lambda _: None)

    # Add skill for assistant-123 with namespace (assistant-123, filesystem)
    assistant_1_skill = make_skill_content("async-skill-one", "Async skill for assistant 1")
    store.put(
        ("assistant-123", "filesystem"),
        "/skills/user/async-skill-one/SKILL.md",
        create_store_skill_item(assistant_1_skill),
    )

    # Test: assistant-123 can read its own skill
    with _runtime_context("assistant-123"):
        result_1 = await middleware.abefore_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_1 is not None
    assert len(result_1["skills_metadata"]) == 1
    assert result_1["skills_metadata"][0]["name"] == "async-skill-one"
    assert result_1["skills_metadata"][0]["description"] == "Async skill for assistant 1"

    # Test: assistant-456 cannot see assistant-123's skill (different namespace)
    with _runtime_context("assistant-456"):
        result_2 = await middleware.abefore_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_2 is not None
    assert len(result_2["skills_metadata"]) == 0  # No skills in assistant-456's namespace yet

    # Add skill for assistant-456 with namespace (assistant-456, filesystem)
    assistant_2_skill = make_skill_content("async-skill-two", "Async skill for assistant 2")
    store.put(
        ("assistant-456", "filesystem"),
        "/skills/user/async-skill-two/SKILL.md",
        create_store_skill_item(assistant_2_skill),
    )

    # Test: assistant-456 can read its own skill
    with _runtime_context("assistant-456"):
        result_3 = await middleware.abefore_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_3 is not None
    assert len(result_3["skills_metadata"]) == 1
    assert result_3["skills_metadata"][0]["name"] == "async-skill-two"
    assert result_3["skills_metadata"][0]["description"] == "Async skill for assistant 2"

    # Test: assistant-123 still only sees its own skill (no cross-contamination)
    with _runtime_context("assistant-123"):
        result_4 = await middleware.abefore_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_4 is not None
    assert len(result_4["skills_metadata"]) == 1
    assert result_4["skills_metadata"][0]["name"] == "async-skill-one"
    assert result_4["skills_metadata"][0]["description"] == "Async skill for assistant 1"


# --- libs/evals/tests/evals/test_summarization.py ---

def test_summarize_continues_task(tmp_path: Path, model: BaseChatModel) -> None:
    """Test that summarization triggers and the agent can continue reading a large file."""
    agent, _, _ = _setup_summarization_test(tmp_path, model, 15_000)
    thread_id = uuid.uuid4().hex[:8]

    trajectory = run_agent(
        agent,
        model=model,
        query="Can you read the entirety of summarization.py, 500 lines at a time, and summarize it?",
        thread_id=thread_id,
    )

    # Check we summarized
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    assert state.values["_summarization_event"]

    # Verify the agent made substantial progress reading the file after summarization.
    # We check the highest line number seen across all tool observations to confirm
    # the agent continued working after context was summarized.
    max_line_seen = 0
    reached_eof = False

    for step in trajectory.steps:
        for obs in step.observations:
            # Check for EOF error (indicates agent tried to read past end)
            if "exceeds file length" in obs.content:
                reached_eof = True
            # Extract line numbers from formatted output (e.g., "4609\t    )")
            line_numbers = re.findall(r"^\s*(\d+)\t", obs.content, re.MULTILINE)
            if line_numbers:
                max_line_seen = max(max_line_seen, *[int(n) for n in line_numbers])

    assert max_line_seen >= 959 or reached_eof, (
        f"Expected agent to make substantial progress reading file. Max line seen: {max_line_seen}, reached EOF: {reached_eof}"
    )

def test_summarization_offloads_to_filesystem(tmp_path: Path, model: BaseChatModel) -> None:
    """Test that conversation history is offloaded to filesystem during summarization.

    This verifies the summarization middleware correctly writes conversation history
    as markdown to the backend at /conversation_history/{thread_id}.md.
    """
    agent, _, root = _setup_summarization_test(tmp_path, model, 15_000)
    thread_id = uuid.uuid4().hex[:8]

    _ = run_agent(
        agent,
        model=model,
        query="Can you read the entirety of summarization.py, 500 lines at a time, and summarize it?",
        thread_id=thread_id,
    )

    # Check we summarized
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    assert state.values["_summarization_event"]

    # Verify conversation history was offloaded to filesystem
    conversation_history_root = root / "conversation_history"
    assert conversation_history_root.exists(), (
        f"Conversation history root directory not found at {conversation_history_root}"
    )

    # Verify the markdown file exists for thread_id
    history_file = conversation_history_root / f"{thread_id}.md"
    assert history_file.exists(), f"Expected markdown file at {history_file}"

    # Read and verify markdown content
    content = history_file.read_text()

    # Should have timestamp header(s) from summarization events
    assert "## Summarized at" in content, "Missing timestamp header in markdown file"

    # Should contain human-readable message content (from get_buffer_string)
    assert "Human:" in content or "AI:" in content, "Missing message content in markdown file"

    # Verify the summary message references the conversation_history path
    summary_message = state.values["_summarization_event"]["summary_message"]
    assert "conversation_history" in summary_message.content
    assert f"{thread_id}.md" in summary_message.content

    # --- Needle in the haystack follow-up ---
    # Ask about a specific detail from the beginning of the file that was read
    # before summarization. The agent should read the conversation history to find it.
    # The first standard library import in summarization.py (after `from __future__`) is `import base64`.
    followup_trajectory = run_agent(
        agent,
        model=model,
        query=(
            "What is the first standard library import in summarization.py? (After "
            "the `from __future__` import.) Check the conversation history if needed."
        ),
        thread_id=thread_id,
    )

    # The agent should retrieve the answer from the conversation history
    final_answer = followup_trajectory.answer

    # Check that the answer mentions "base64" (the first standard library import)
    assert "logging" in final_answer.lower(), (
        f"Expected agent to find 'logging' as the first import. Got: {final_answer}"
    )

def test_compact_tool_new_task(tmp_path: Path, model: BaseChatModel) -> None:
    """Agent calls compact_conversation when switching to an unrelated task after a long conversation."""
    agent, _, _ = _setup_summarization_test(tmp_path, model, 35_000, include_compact_tool=True)

    seed = _load_seed_messages()
    query = "Thanks. Let's move on to a completely new task. To prepare, first spec out how to upgrade a web app to Typescript 5.5"
    trajectory = run_agent(
        agent,
        model=model,
        query=[*seed, HumanMessage(query)],
    )
    assert _called_compact(trajectory)

def test_compact_tool_not_overly_sensitive(tmp_path: Path, model: BaseChatModel) -> None:
    """Agent does NOT call compact_conversation for a follow-up question related to the prior conversation."""
    agent, _, _ = _setup_summarization_test(tmp_path, model, 35_000, include_compact_tool=True)

    seed = _load_seed_messages()
    query = "Moving on, what are the two primary OpenAI APIs supported?"
    trajectory = run_agent(
        agent,
        model=model,
        query=[*seed, HumanMessage(query)],
    )
    assert not _called_compact(trajectory)

def test_compact_tool_large_reads(tmp_path: Path, model: BaseChatModel) -> None:
    """Agent calls compact_conversation when asked to read another large file after a long conversation."""
    another_large_file = "https://raw.githubusercontent.com/langchain-ai/deepagents/5c90376c02754c67d448908e55d1e953f54b8acd/libs/deepagents/deepagents/middleware/filesystem.py"

    response = requests.get(another_large_file, timeout=30)
    response.raise_for_status()

    agent, backend, _ = _setup_summarization_test(
        tmp_path,
        model,
        35_000,
        middleware=[ModelCallLimitMiddleware(run_limit=3)],
        include_compact_tool=True,
    )
    backend.upload_files([("/filesystem.py", response.content)])

    seed = _load_seed_messages()
    query = "OK, done with that. Now do the same for filesystem.py."
    trajectory = run_agent(
        agent,
        model=model,
        query=[*seed, HumanMessage(query)],
    )
    assert _called_compact(trajectory)


# --- libs/evals/tests/evals/test_tool_usage_incident_graph.py ---

async def test_single_tool_list_incident_ids(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="What are all the incident IDs in the system?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("41017"),
            final_text_contains("41029"),
            final_text_contains("41043"),
            final_text_contains("41058"),
        )
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[tool_call(name="list_incident_ids", step=1)],
        ),
    )

async def test_two_tools_current_incident_service_name(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="What service is affected by the current incident?",
        scorer=TrajectoryScorer()
        .success(final_text_contains("payments-api"))
        .expect(
            agent_steps=4,
            tool_call_requests=3,
            tool_calls=[
                tool_call(name="get_current_incident_id", step=1),
                tool_call(
                    name="get_incident_service", step=2, args_contains={"incident_id": 41017}
                ),
                tool_call(name="get_service_name", step=3, args_contains={"service_id": 8401}),
            ],
        ),
    )

async def test_three_tools_find_service_owner_team(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="Which team owns checkout-web?",
        scorer=TrajectoryScorer()
        .success(final_text_contains("Checkout Experience"))
        .expect(
            agent_steps=4,
            tool_call_requests=3,
            tool_calls=[
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "checkout-web"}
                ),
                tool_call(name="get_service_team", step=2, args_contains={"service_id": 8514}),
                tool_call(name="get_team_name", step=3, args_contains={"team_id": 562}),
            ],
        ),
    )

async def test_multi_question_current_incident_service_and_incident_oncall(
    model: BaseChatModel,
) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Answer both questions: 1) What service is affected by the current incident? "
            "2) Who is the on-call engineer for incident 41029?"
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("payments-api"),
            final_text_contains("Cara Singh"),
        )
        .expect(
            agent_steps=6,
            tool_call_requests=7,
            tool_calls=[
                tool_call(name="get_current_incident_id"),
                tool_call(name="get_incident_service", args_contains={"incident_id": 41017}),
                tool_call(name="get_incident_service", args_contains={"incident_id": 41029}),
                tool_call(name="get_service_team", args_contains={"service_id": 8514}),
                tool_call(name="get_team_oncall_engineer", args_contains={"team_id": 562}),
                tool_call(name="get_service_name", args_contains={"service_id": 8401}),
                tool_call(name="get_engineer_name", args_contains={"engineer_id": 7381}),
            ],
        ),
    )

async def test_multi_question_incident_oncall_and_incident_environment(
    model: BaseChatModel,
) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Answer both questions: 1) Who is the on-call engineer for incident 41029? "
            "2) What environment and region is incident 41058 running in?"
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Cara Singh"),
            final_text_contains("staging", case_insensitive=True),
            final_text_contains("us-west-2"),
        )
        .expect(
            agent_steps=5,
            tool_call_requests=8,
            tool_calls=[
                tool_call(name="get_incident_service", args_contains={"incident_id": 41029}),
                tool_call(name="get_service_team", args_contains={"service_id": 8514}),
                tool_call(name="get_team_oncall_engineer", args_contains={"team_id": 562}),
                tool_call(name="get_engineer_name", args_contains={"engineer_id": 7381}),
                tool_call(name="get_incident_service", args_contains={"incident_id": 41058}),
                tool_call(name="get_service_environment", args_contains={"service_id": 8799}),
                tool_call(name="get_environment_name", args_contains={"environment_id": 442}),
                tool_call(name="get_environment_region", args_contains={"environment_id": 442}),
            ],
        ),
    )

async def test_multi_question_incident_oncall_and_service_with_most_firing_alerts(
    model: BaseChatModel,
) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Answer both questions: 1) Who is on call for incident 41029? "
            "2) Which service currently has the most firing alerts?"
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Cara Singh"),
            final_text_contains("payments-api"),
            final_text_contains("2", case_insensitive=False),
        )
        .expect(
            agent_steps=6,
            tool_call_requests=13,
            tool_calls=[
                tool_call(name="get_incident_service", args_contains={"incident_id": 41029}),
                tool_call(name="get_service_team", args_contains={"service_id": 8514}),
                tool_call(name="get_team_oncall_engineer", args_contains={"team_id": 562}),
                tool_call(name="get_engineer_name", args_contains={"engineer_id": 7381}),
                tool_call(name="get_service_name", args_contains={"service_id": 8401}),
            ],
        ),
    )

async def test_multi_question_three_independent_simple_lookups(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Answer all three questions: 1) What is the severity of incident 41017? "
            "2) What is the default branch for repo 9217? "
            "3) What is the region for environment 442?"
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("sev1", case_insensitive=True),
            final_text_contains("main"),
            final_text_contains("us-west-2"),
        )
        .expect(
            agent_steps=2,
            tool_call_requests=3,
            tool_calls=[
                # All three lookups are independent, so the optimal trajectory issues them
                # together in one tool-calling step and then answers in the final step.
                tool_call(
                    name="get_incident_severity", step=1, args_contains={"incident_id": 41017}
                ),
                tool_call(name="get_repo_default_branch", step=1, args_contains={"repo_id": 9217}),
                tool_call(
                    name="get_environment_region", step=1, args_contains={"environment_id": 442}
                ),
            ],
        ),
    )

async def test_four_tools_incident_to_oncall_name(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="Who is the on-call engineer for incident 41029?",
        scorer=TrajectoryScorer()
        .success(final_text_contains("Cara Singh"))
        .expect(
            agent_steps=5,
            tool_call_requests=4,
            tool_calls=[
                tool_call(
                    name="get_incident_service", step=1, args_contains={"incident_id": 41029}
                ),
                tool_call(name="get_service_team", step=2, args_contains={"service_id": 8514}),
                tool_call(name="get_team_oncall_engineer", step=3, args_contains={"team_id": 562}),
                tool_call(name="get_engineer_name", step=4, args_contains={"engineer_id": 7381}),
            ],
        ),
    )

async def test_four_tools_service_runbook_url(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="What is the runbook URL for payments-api?",
        scorer=TrajectoryScorer()
        .success(final_text_contains("https://runbooks.example.com/payments-api-5xx"))
        .expect(
            agent_steps=4,
            tool_call_requests=3,
            tool_calls=[
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "payments-api"}
                ),
                tool_call(name="get_service_runbook", step=2, args_contains={"service_id": 8401}),
                tool_call(name="get_runbook_url", step=3, args_contains={"runbook_id": 12041}),
            ],
        ),
    )

async def test_five_tools_incident_latest_deploy_and_repo(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="For incident 41017, what repo was most recently deployed and what version was it?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("payments-service"),
            final_text_contains("payments-api@2024.08.12.1"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=5,
            tool_calls=[
                tool_call(
                    name="get_incident_service", step=1, args_contains={"incident_id": 41017}
                ),
                tool_call(name="get_service_repo", step=2, args_contains={"service_id": 8401}),
                tool_call(
                    name="get_latest_deploy_for_service", step=2, args_contains={"service_id": 8401}
                ),
                tool_call(name="get_repo_name", step=3, args_contains={"repo_id": 9104}),
                tool_call(name="get_deploy_version", step=3, args_contains={"deploy_id": 66011}),
            ],
        ),
    )

async def test_five_tools_incident_environment_name_and_region(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="What environment and region is incident 41058 running in?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("staging"),
            final_text_contains("us-west-2"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[
                tool_call(
                    name="get_incident_service", step=1, args_contains={"incident_id": 41058}
                ),
                tool_call(
                    name="get_service_environment", step=2, args_contains={"service_id": 8799}
                ),
                tool_call(
                    name="get_environment_name", step=3, args_contains={"environment_id": 442}
                ),
                tool_call(
                    name="get_environment_region", step=3, args_contains={"environment_id": 442}
                ),
            ],
        ),
    )

async def test_five_tools_service_dependency_names_parallel(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="What services does checkout-web depend on? Give me the dependency names.",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("payments-api"),
            final_text_contains("identity-api"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "checkout-web"}
                ),
                tool_call(
                    name="list_service_dependencies", step=2, args_contains={"service_id": 8514}
                ),
                tool_call(name="get_service_name", step=3, args_contains={"service_id": 8401}),
                tool_call(name="get_service_name", step=3, args_contains={"service_id": 8627}),
            ],
        ),
    )

async def test_five_tools_service_alert_names_parallel(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="List the alert names for payments-api.",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("payments-api 5xx rate"),
            final_text_contains("payments-api latency p95"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "payments-api"}
                ),
                tool_call(
                    name="list_service_alert_ids", step=2, args_contains={"service_id": 8401}
                ),
                tool_call(name="get_alert_name", step=3, args_contains={"alert_id": 55101}),
                tool_call(name="get_alert_name", step=3, args_contains={"alert_id": 55114}),
            ],
        ),
    )

async def test_six_tools_current_incident_oncall_name_and_email(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="For the current incident, who is on call and what is their email address?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Ben Ortiz"),
            final_text_contains("ben@ops.example.com"),
        )
        .expect(
            agent_steps=6,
            tool_call_requests=6,
            tool_calls=[
                tool_call(name="get_current_incident_id", step=1),
                tool_call(
                    name="get_incident_service", step=2, args_contains={"incident_id": 41017}
                ),
                tool_call(name="get_service_team", step=3, args_contains={"service_id": 8401}),
                tool_call(name="get_team_oncall_engineer", step=4, args_contains={"team_id": 481}),
                tool_call(name="get_engineer_name", step=5, args_contains={"engineer_id": 7243}),
                tool_call(name="get_engineer_email", step=5, args_contains={"engineer_id": 7243}),
            ],
        ),
    )

async def test_six_tools_service_repo_and_branch(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="What repository backs identity-api and what is its default branch?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("identity-service"),
            final_text_contains("main"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "identity-api"}
                ),
                tool_call(name="get_service_repo", step=2, args_contains={"service_id": 8627}),
                tool_call(name="get_repo_name", step=3, args_contains={"repo_id": 9346}),
                tool_call(name="get_repo_default_branch", step=3, args_contains={"repo_id": 9346}),
            ],
        ),
    )

async def test_six_tools_incident_title_severity_and_status(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="For incident 41043, tell me its title, severity, and status.",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Identity login error burst"),
            final_text_contains("sev2", case_insensitive=True),
            final_text_contains("resolved", case_insensitive=True),
        )
        .expect(
            agent_steps=2,
            tool_call_requests=3,
            tool_calls=[
                tool_call(name="get_incident_title", args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_severity", args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_status", args_contains={"incident_id": 41043}),
            ],
        ),
    )

async def test_six_tools_current_incident_metrics_parallel(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="For the current incident's service, what are the current error_rate and latency_p95 metrics?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("12.4%"),
            final_text_contains("1.8s"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[
                tool_call(name="get_current_incident_id", step=1),
                tool_call(
                    name="get_incident_service", step=2, args_contains={"incident_id": 41017}
                ),
                tool_call(
                    name="get_metric_value",
                    step=3,
                    args_contains={"service_id": 8401, "metric_name": "error_rate"},
                ),
                tool_call(
                    name="get_metric_value",
                    step=3,
                    args_contains={"service_id": 8401, "metric_name": "latency_p95"},
                ),
            ],
        ),
    )

async def test_aggregation_active_incident_count_by_team(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query=(
            "How many active incidents belong to each team, and which team has the most active incidents? "
            "Please include the team names and counts."
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Payments Platform"),
            final_text_contains("Checkout Experience"),
            final_text_contains("1"),
        )
        .expect(
            agent_steps=6,
            tool_call_requests=13,
            tool_calls=[
                tool_call(name="list_incident_ids", step=1),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41017}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41029}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41058}),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41017}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41029}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41058}
                ),
                tool_call(name="get_service_team", step=4, args_contains={"service_id": 8401}),
                tool_call(name="get_service_team", step=4, args_contains={"service_id": 8514}),
                tool_call(name="get_service_team", step=4, args_contains={"service_id": 8799}),
                tool_call(name="get_team_name", step=5, args_contains={"team_id": 481}),
                tool_call(name="get_team_name", step=5, args_contains={"team_id": 562}),
            ],
        ),
    )

async def test_comparison_active_incident_most_dependencies(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Among the active incidents, which incident affects the service with the most dependencies? "
            "Return the incident ID, incident title, service name, and dependency count."
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("41029"),
            final_text_contains("Checkout page latency spike"),
            final_text_contains("checkout-web"),
            final_text_contains("2"),
        )
        .expect(
            agent_steps=5,
            tool_call_requests=14,
            tool_calls=[
                tool_call(name="list_incident_ids", step=1),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41017}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41029}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41058}),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41017}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41029}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41058}
                ),
                tool_call(
                    name="list_service_dependencies", step=4, args_contains={"service_id": 8401}
                ),
                tool_call(
                    name="list_service_dependencies", step=4, args_contains={"service_id": 8514}
                ),
                tool_call(
                    name="list_service_dependencies", step=4, args_contains={"service_id": 8799}
                ),
                tool_call(name="get_incident_title", step=5, args_contains={"incident_id": 41029}),
                tool_call(name="get_service_name", step=5, args_contains={"service_id": 8514}),
            ],
        ),
    )

async def test_latest_selection_active_incident_most_recent_deploy(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Across the services involved in active incidents, which service had the most recent deploy? "
            "Return the service name, repo name, deploy version, and deploy timestamp."
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("checkout-web"),
            final_text_contains("checkout-frontend"),
            final_text_contains("checkout-web@2024.08.12.3"),
            final_text_contains("2024-08-12T09:05:00Z"),
        )
        .expect(
            agent_steps=6,
            tool_call_requests=15,
            tool_calls=[
                tool_call(name="list_incident_ids", step=1),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41017}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41029}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41058}),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41017}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41029}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41058}
                ),
                tool_call(
                    name="get_latest_deploy_for_service", step=4, args_contains={"service_id": 8401}
                ),
                tool_call(
                    name="get_latest_deploy_for_service", step=4, args_contains={"service_id": 8514}
                ),
                tool_call(
                    name="get_latest_deploy_for_service", step=4, args_contains={"service_id": 8799}
                ),
                tool_call(name="get_deploy_timestamp", step=5, args_contains={"deploy_id": 66011}),
                tool_call(name="get_deploy_timestamp", step=5, args_contains={"deploy_id": 66037}),
                tool_call(name="get_deploy_timestamp", step=5, args_contains={"deploy_id": 66059}),
                tool_call(name="get_service_name", step=5, args_contains={"service_id": 8514}),
                tool_call(name="get_service_repo", step=5, args_contains={"service_id": 8514}),
                tool_call(name="get_deploy_version", step=5, args_contains={"deploy_id": 66037}),
            ],
        ),
    )

async def test_metric_ranking_active_incident_highest_latency(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Among the active incidents affecting customer-facing services with a latency_p95 metric, "
            "which incident is tied to the service with the highest latency_p95, and which team owns that service?"
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("41029"),
            final_text_contains("checkout-web"),
            final_text_contains("2.4s"),
            final_text_contains("Checkout Experience"),
        )
        .expect(
            agent_steps=7,
            tool_call_requests=12,
            tool_calls=[
                tool_call(name="list_incident_ids", step=1),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41017}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41029}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41058}),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41017}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41029}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41058}
                ),
                tool_call(
                    name="get_metric_value",
                    step=4,
                    args_contains={"service_id": 8401, "metric_name": "latency_p95"},
                ),
                tool_call(
                    name="get_metric_value",
                    step=4,
                    args_contains={"service_id": 8514, "metric_name": "latency_p95"},
                ),
                tool_call(name="get_service_name", step=5, args_contains={"service_id": 8514}),
                tool_call(name="get_service_team", step=5, args_contains={"service_id": 8514}),
                tool_call(name="get_team_name", step=6, args_contains={"team_id": 562}),
            ],
        ),
    )

async def test_alert_aggregation_service_with_most_firing_alerts(model: BaseChatModel) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query="Which service has the most firing alerts right now, and what are the names of those alerts?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("payments-api"),
            final_text_contains("payments-api 5xx rate"),
            final_text_contains("payments-api latency p95"),
            final_text_contains("2"),
        )
        .expect(
            agent_steps=6,
            tool_call_requests=16,
            tool_calls=[
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "payments-api"}
                ),
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "checkout-web"}
                ),
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "identity-api"}
                ),
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "analytics-worker"}
                ),
                tool_call(
                    name="list_service_alert_ids", step=2, args_contains={"service_id": 8401}
                ),
                tool_call(
                    name="list_service_alert_ids", step=2, args_contains={"service_id": 8514}
                ),
                tool_call(
                    name="list_service_alert_ids", step=2, args_contains={"service_id": 8627}
                ),
                tool_call(
                    name="list_service_alert_ids", step=2, args_contains={"service_id": 8799}
                ),
                tool_call(name="get_alert_status", step=3, args_contains={"alert_id": 55101}),
                tool_call(name="get_alert_status", step=3, args_contains={"alert_id": 55114}),
                tool_call(name="get_alert_status", step=3, args_contains={"alert_id": 55128}),
                tool_call(name="get_alert_status", step=3, args_contains={"alert_id": 55139}),
                tool_call(name="get_alert_status", step=3, args_contains={"alert_id": 55152}),
                tool_call(name="get_service_name", step=4, args_contains={"service_id": 8401}),
                tool_call(name="get_alert_name", step=5, args_contains={"alert_id": 55101}),
                tool_call(name="get_alert_name", step=5, args_contains={"alert_id": 55114}),
            ],
        ),
    )

async def test_dependency_reasoning_active_incident_depending_on_identity_api(
    model: BaseChatModel,
) -> None:
    agent = _create_agent(model)
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Which active incident affects a service that depends on identity-api, and who is the on-call engineer "
            "for the owning team? Include the engineer email too."
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("41029"),
            final_text_contains("Checkout page latency spike"),
            final_text_contains("Cara Singh"),
            final_text_contains("cara@ops.example.com"),
        )
        .expect(
            agent_steps=7,
            tool_call_requests=16,
            tool_calls=[
                tool_call(name="list_incident_ids", step=1),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41017}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41029}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41058}),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41017}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41029}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41058}
                ),
                tool_call(
                    name="list_service_dependencies", step=4, args_contains={"service_id": 8401}
                ),
                tool_call(
                    name="list_service_dependencies", step=4, args_contains={"service_id": 8514}
                ),
                tool_call(
                    name="list_service_dependencies", step=4, args_contains={"service_id": 8799}
                ),
                tool_call(name="get_incident_title", step=5, args_contains={"incident_id": 41029}),
                tool_call(name="get_service_team", step=5, args_contains={"service_id": 8514}),
                tool_call(name="get_team_oncall_engineer", step=6, args_contains={"team_id": 562}),
                tool_call(name="get_engineer_name", step=7, args_contains={"engineer_id": 7381}),
                tool_call(name="get_engineer_email", step=7, args_contains={"engineer_id": 7381}),
            ],
        ),
    )


# --- libs/evals/tests/evals/tau2_airline/test_tau2_airline.py ---

def test_tau2_airline(model: BaseChatModel, task_id: str) -> None:
    """Run a multi-turn tau2 airline task and evaluate the result.

    Args:
        model: The agent's chat model (from --model CLI option).
        task_id: The tau2 task ID to run.
    """
    # Immediately override @pytest.mark.langsmith auto-capture so the dataset
    # example records clean metadata even if run_multi_turn() raises.
    _clean_inputs = {
        "task_id": task_id,
        "model": str(getattr(model, "model", None) or getattr(model, "model_name", "")),
    }
    t.log_inputs(_clean_inputs)
    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.inputs = _clean_inputs
    else:
        logger.warning(
            "get_current_run_tree() returned None in @pytest.mark.langsmith test; "
            "dataset example inputs will not be overridden"
        )

    task = load_task(task_id)
    policy = load_policy()

    db = load_db()
    initial_state = task.get("initial_state")
    if initial_state:
        for key, value in initial_state.items():
            parts = key.split(".")
            obj = db
            for part in parts[:-1]:
                obj = getattr(obj, part) if not isinstance(obj, dict) else obj[part]
            final_key = parts[-1]
            if isinstance(obj, dict):
                obj[final_key] = value
            else:
                setattr(obj, final_key, value)

    tools, tool_log = create_airline_tools(db)

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT.format(domain_policy=policy),
        checkpointer=MemorySaver(),
    )

    user_model = init_chat_model(USER_SIM_MODEL)
    user_sim = UserSimulator(model=user_model, scenario=task.get("user_scenario", {}))

    conversation = run_multi_turn(
        agent,
        user_sim,
        model=model,
        tool_call_log=tool_log,
        max_turns=30,
    )

    # Override per-turn t.log_inputs() calls from run_agent() inside
    # run_multi_turn() with clean test-level metadata.
    t.log_inputs(_clean_inputs)

    reward = evaluate_task(
        actual_db=db,
        tool_log=tool_log,
        messages=conversation.messages,
        task=task,
    )
    episode_score = score_tau2_episode(reward)

    t.log_feedback(key="db_score", value=reward.db_score)
    t.log_feedback(key="communicate_score", value=reward.communicate_score)
    t.log_feedback(key="turn_count", value=conversation.turn_count)
    for key, value in episode_score.expected_metrics.items():
        t.log_feedback(key=key, value=value)

    logger.info(
        "Task %s: success=%s reasons=%s (%s), %d turns, %d tool calls",
        task_id,
        episode_score.success,
        ",".join(episode_score.success_reasons) if episode_score.success_reasons else "none",
        reward.details,
        conversation.turn_count,
        len(tool_log),
    )

    assert episode_score.success, (
        f"Task {task_id} failed: reasons={episode_score.success_reasons} details={reward.details}\n"
        f"Tool calls: {[e.name for e in tool_log]}\n"
        f"Terminated by: {conversation.terminated_by}"
    )


# --- libs/repl/tests/unit_tests/test_interpreter.py ---

def test_tool_payload_ignores_model_supplied_runtime_dict() -> None:
    runtime = ToolRuntime(
        state={},
        context=None,
        config={"configurable": {"user_id": "trusted-user"}},
        stream_writer=lambda _: None,
        store=None,
        tool_call_id="call_1",
    )
    interpreter = Interpreter(functions={"get_user_id": get_user_id}, runtime=runtime)

    result = interpreter.evaluate('get_user_id({"runtime": "attacker"})')

    assert result == "trusted-user"

def test_parallel_propagates_function_errors() -> None:
    def fail() -> None:
        msg = "boom"
        raise RuntimeError(msg)

    interpreter = Interpreter(functions={"fail": fail})

    with pytest.raises(RuntimeError, match="boom"):
        interpreter.evaluate("parallel(fail(), 1)")

def test_try_returns_fallback_on_error() -> None:
    interpreter = Interpreter(
        functions={"fail": lambda: (_ for _ in ()).throw(RuntimeError("boom"))}
    )

    assert interpreter.evaluate('try(fail(), "fallback")') == "fallback"

def test_try_only_catches_primary_expression_errors() -> None:
    interpreter = Interpreter(
        functions={"fail": lambda: (_ for _ in ()).throw(RuntimeError("boom"))}
    )

    with pytest.raises(NameError, match="Unknown name: missing"):
        interpreter.evaluate("try(fail(), missing)")

