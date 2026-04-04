# microsoft/autogen
# 37 test functions with real LLM calls
# Source: https://github.com/microsoft/autogen


# --- python/packages/autogen-ext/tests/agents/test_openai_agent_builtin_tool_validation.py ---

async def test_integration_image_generation_tool() -> None:
    """Test image_generation tool with actual API call."""
    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)
    tools = ["image_generation"]  # type: ignore
    agent = OpenAIAgent(
        name="image_gen_test",
        description="Test agent with image generation capability",
        client=client,
        model="gpt-4o",
        instructions="You are a helpful assistant with image generation capabilities. Generate images when requested.",
        tools=tools,  # type: ignore
    )
    cancellation_token = CancellationToken()

    # Test image generation functionality
    response = await agent.on_messages(
        [TextMessage(source="user", content="Generate an image of a beautiful sunset over mountains")],
        cancellation_token,
    )
    assert hasattr(response, "chat_message")
    assert hasattr(response.chat_message, "content")
    content = getattr(response.chat_message, "content", "")
    assert len(content) > 0

async def test_integration_multiple_builtin_tools() -> None:
    """Test multiple builtin tools together with actual API call."""
    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)
    tools = ["web_search_preview", "image_generation"]  # type: ignore
    agent = OpenAIAgent(
        name="multi_tool_test",
        description="Test agent with multiple builtin tools",
        client=client,
        model="gpt-4o",
        instructions="You are a helpful assistant with web search and image generation capabilities.",
        tools=tools,  # type: ignore
    )
    cancellation_token = CancellationToken()

    # Test multiple tools functionality
    response = await agent.on_messages(
        [
            TextMessage(
                source="user",
                content="Search for information about space exploration and generate an image of a rocket",
            )
        ],
        cancellation_token,
    )
    assert hasattr(response, "chat_message")
    assert hasattr(response.chat_message, "content")
    content = getattr(response.chat_message, "content", "")
    assert len(content) > 0

async def test_integration_streaming_with_builtin_tools() -> None:
    """Test streaming responses with builtin tools."""
    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)
    tools = ["web_search_preview"]  # type: ignore
    agent = OpenAIAgent(
        name="streaming_test",
        description="Test agent with streaming and builtin tools",
        client=client,
        model="gpt-4o",
        instructions="You are a helpful assistant with web search capabilities.",
        tools=tools,  # type: ignore
    )
    cancellation_token = CancellationToken()

    # Test streaming with builtin tools
    messages: list[Any] = []
    async for message in agent.on_messages_stream(
        [TextMessage(source="user", content="What are the latest news about renewable energy?")],
        cancellation_token,
    ):
        messages.append(message)

    # Verify we received some messages
    assert len(messages) > 0
    # Verify at least one message has content
    content_messages = [
        msg
        for msg in messages
        if hasattr(msg, "chat_message")
        and hasattr(msg.chat_message, "content")
        and getattr(msg.chat_message, "content", False)
    ]
    assert len(content_messages) > 0


# --- python/packages/autogen-ext/tests/models/test_ollama_chat_completion_client.py ---

def test_create_args_from_config_drops_unexpected_kwargs() -> None:
    test_config: Mapping[str, Any] = {
        "model": "llama3.1",
        "messages": [],
        "tools": [],
        "stream": False,
        "format": "json",
        "options": {},
        "keep_alive": 100,
        "extra_unexpected_kwarg": "value",
        "another_extra_unexpected_kwarg": "another_value",
    }

    client = OllamaChatCompletionClient(**test_config)

    final_create_args = client.get_create_args()

    for arg in final_create_args.keys():
        assert arg in OLLAMA_VALID_CREATE_KWARGS_KEYS

async def test_ollama_create(model: str, ollama_client: OllamaChatCompletionClient) -> None:
    create_result = await ollama_client.create(
        messages=[
            UserMessage(
                content="Taking two balls from a bag of 10 green balls and 20 red balls, "
                "what is the probability of getting a green and a red balls?",
                source="user",
            ),
        ]
    )
    assert isinstance(create_result.content, str)
    assert len(create_result.content) > 0
    assert create_result.finish_reason == "stop"
    assert create_result.usage is not None

    chunks: List[str | CreateResult] = []
    async for chunk in ollama_client.create_stream(
        messages=[
            UserMessage(
                content="Taking two balls from a bag of 10 green balls and 20 red balls, "
                "what is the probability of getting a green and a red balls?",
                source="user",
            ),
        ]
    ):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert isinstance(chunks[-1], CreateResult)
    assert chunks[-1].finish_reason == "stop"
    assert len(chunks[-1].content) > 0
    assert chunks[-1].usage is not None

async def test_ollama_create_structured_output(model: str, ollama_client: OllamaChatCompletionClient) -> None:
    class ResponseType(BaseModel):
        calculation: str
        result: str

    create_result = await ollama_client.create(
        messages=[
            UserMessage(
                content="Taking two balls from a bag of 10 green balls and 20 red balls, "
                "what is the probability of getting a green and a red balls?",
                source="user",
            ),
        ],
        json_output=ResponseType,
    )
    assert isinstance(create_result.content, str)
    assert len(create_result.content) > 0
    assert create_result.finish_reason == "stop"
    assert create_result.usage is not None
    assert ResponseType.model_validate_json(create_result.content)

    # Test streaming completion with the Ollama deepseek-r1:1.5b model.
    chunks: List[str | CreateResult] = []
    async for chunk in ollama_client.create_stream(
        messages=[
            UserMessage(
                content="Taking two balls from a bag of 10 green balls and 20 red balls, "
                "what is the probability of getting a green and a red balls?",
                source="user",
            ),
        ],
        json_output=ResponseType,
    ):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert isinstance(chunks[-1], CreateResult)
    assert chunks[-1].finish_reason == "stop"
    assert isinstance(chunks[-1].content, str)
    assert len(chunks[-1].content) > 0
    assert chunks[-1].usage is not None
    assert ResponseType.model_validate_json(chunks[-1].content)

async def test_ollama_create_stream_tools(model: str, ollama_client: OllamaChatCompletionClient) -> None:
    def add(x: int, y: int) -> str:
        return str(x + y)

    add_tool = FunctionTool(add, description="Add two numbers")

    stream = ollama_client.create_stream(
        messages=[
            UserMessage(
                content="What is 2 + 2? Use the add tool.",
                source="user",
            ),
        ],
        tools=[add_tool],
    )
    chunks: List[str | CreateResult] = []
    async for chunk in stream:
        chunks.append(chunk)
    assert len(chunks) > 0
    assert isinstance(chunks[-1], CreateResult)
    create_result = chunks[-1]
    assert isinstance(create_result.content, list)
    assert len(create_result.content) > 0
    assert isinstance(create_result.content[0], FunctionCall)
    assert create_result.content[0].name == add_tool.name
    assert create_result.content[0].arguments == json.dumps({"x": 2, "y": 2})
    assert create_result.finish_reason == "stop"
    assert create_result.usage is not None
    assert create_result.usage.prompt_tokens == 10
    assert create_result.usage.completion_tokens == 12

async def test_tool_choice_required_no_tools_error() -> None:
    """Test tool_choice='required' with no tools raises ValueError"""
    model = "llama3.2"
    client = OllamaChatCompletionClient(model=model)

    with pytest.raises(ValueError, match="tool_choice 'required' specified but no tools provided"):
        await client.create(
            messages=[UserMessage(content="What is 2 + 3?", source="user")],
            tools=[],  # No tools provided
            tool_choice="required",
        )

def test_ollama_load_component() -> None:
    """Test that OllamaChatCompletionClient can be loaded via ChatCompletionClient.load_component()."""
    from autogen_core.models import ChatCompletionClient

    # Test the exact configuration from the issue
    config = {
        "provider": "OllamaChatCompletionClient",
        "config": {
            "model": "qwen3",
            "host": "http://1.2.3.4:30130",
        },
    }

    # This should not raise an error anymore
    client = ChatCompletionClient.load_component(config)

    # Verify we got the right type of client
    assert isinstance(client, OllamaChatCompletionClient)
    assert client._model_name == "qwen3"  # type: ignore[reportPrivateUsage]

    # Test that the config was applied correctly
    create_args = client.get_create_args()
    assert create_args["model"] == "qwen3"  # type: ignore[reportPrivateUsage]

def test_ollama_load_component_via_class() -> None:
    """Test that OllamaChatCompletionClient can be loaded via the class directly."""
    config = {
        "provider": "OllamaChatCompletionClient",
        "config": {
            "model": "llama3.2",
            "host": "http://localhost:11434",
        },
    }

    # Load via the specific class
    client = OllamaChatCompletionClient.load_component(config)

    # Verify we got the right type and configuration
    assert isinstance(client, OllamaChatCompletionClient)
    assert client._model_name == "llama3.2"  # type: ignore[reportPrivateUsage]


# --- python/packages/autogen-ext/tests/models/test_openai_model_client.py ---

async def test_openai_chat_completion_client() -> None:
    client = OpenAIChatCompletionClient(model="gpt-4.1-nano", api_key="api_key")
    assert client

async def test_openai_chat_completion_client_with_gemini_model() -> None:
    client = OpenAIChatCompletionClient(model="gemini-1.5-flash", api_key="api_key")
    assert client

async def test_openai_chat_completion_client_serialization() -> None:
    client = OpenAIChatCompletionClient(model="gpt-4.1-nano", api_key="sk-password")
    assert client
    config = client.dump_component()
    assert config
    assert "sk-password" not in str(config)
    serialized_config = config.model_dump_json()
    assert serialized_config
    assert "sk-password" not in serialized_config
    client2 = OpenAIChatCompletionClient.load_component(config)
    assert client2

async def test_openai_chat_completion_client_raise_on_unknown_model() -> None:
    with pytest.raises(ValueError, match="model_info is required"):
        _ = OpenAIChatCompletionClient(model="unknown", api_key="api_key")

async def test_custom_model_with_capabilities() -> None:
    with pytest.raises(ValueError, match="model_info is required"):
        client = OpenAIChatCompletionClient(model="dummy_model", base_url="https://api.dummy.com/v0", api_key="api_key")

    client = OpenAIChatCompletionClient(
        model="dummy_model",
        base_url="https://api.dummy.com/v0",
        api_key="api_key",
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
    )
    assert client

async def test_azure_openai_chat_completion_client() -> None:
    client = AzureOpenAIChatCompletionClient(
        azure_deployment="gpt-4o-1",
        model="gpt-4o",
        api_key="api_key",
        api_version="2020-08-04",
        azure_endpoint="https://dummy.com",
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.GPT_4O,
            "structured_output": True,
        },
    )
    assert client

async def test_openai_structured_output_with_streaming(model: str, openai_client: OpenAIChatCompletionClient) -> None:
    class AgentResponse(BaseModel):
        thoughts: str
        response: Literal["happy", "sad", "neutral"]

    # Test that the openai client was called with the correct response format.
    stream = openai_client.create_stream(
        messages=[UserMessage(content="I am happy.", source="user")], json_output=AgentResponse
    )
    chunks: List[str | CreateResult] = []
    async for chunk in stream:
        chunks.append(chunk)
    assert len(chunks) > 0
    assert isinstance(chunks[-1], CreateResult)
    assert isinstance(chunks[-1].content, str)
    response = AgentResponse.model_validate(json.loads(chunks[-1].content))
    assert response.thoughts
    assert response.response in ["happy", "sad", "neutral"]

async def test_openai_structured_output_with_streaming_tool_calls(
    model: str, openai_client: OpenAIChatCompletionClient
) -> None:
    class AgentResponse(BaseModel):
        thoughts: str
        response: Literal["happy", "sad", "neutral"]

    def sentiment_analysis(text: str) -> str:
        """Given a text, return the sentiment."""
        return "happy" if "happy" in text else "sad" if "sad" in text else "neutral"

    tool = FunctionTool(sentiment_analysis, description="Sentiment Analysis", strict=True)

    extra_create_args = {"tool_choice": "required"}

    chunks1: List[str | CreateResult] = []
    stream1 = openai_client.create_stream(
        messages=[
            SystemMessage(content="Analyze input text sentiment using the tool provided."),
            UserMessage(content="I am happy.", source="user"),
        ],
        tools=[tool],
        extra_create_args=extra_create_args,
        json_output=AgentResponse,
    )
    async for chunk in stream1:
        chunks1.append(chunk)
    assert len(chunks1) > 0
    create_result1 = chunks1[-1]
    assert isinstance(create_result1, CreateResult)
    assert isinstance(create_result1.content, list)
    assert len(create_result1.content) == 1
    assert isinstance(create_result1.content[0], FunctionCall)
    assert create_result1.content[0].name == "sentiment_analysis"
    assert json.loads(create_result1.content[0].arguments) == {"text": "I am happy."}
    assert create_result1.finish_reason == "function_calls"

    stream2 = openai_client.create_stream(
        messages=[
            SystemMessage(content="Analyze input text sentiment using the tool provided."),
            UserMessage(content="I am happy.", source="user"),
            AssistantMessage(content=create_result1.content, source="assistant"),
            FunctionExecutionResultMessage(
                content=[
                    FunctionExecutionResult(
                        content="happy", call_id=create_result1.content[0].id, is_error=False, name=tool.name
                    )
                ]
            ),
        ],
        json_output=AgentResponse,
    )
    chunks2: List[str | CreateResult] = []
    async for chunk in stream2:
        chunks2.append(chunk)
    assert len(chunks2) > 0
    create_result2 = chunks2[-1]
    assert isinstance(create_result2, CreateResult)
    assert isinstance(create_result2.content, str)
    parsed_response = AgentResponse.model_validate(json.loads(create_result2.content))
    assert parsed_response.thoughts
    assert parsed_response.response in ["happy", "sad", "neutral"]

async def test_hugging_face() -> None:
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        pytest.skip("HF_TOKEN not found in environment variables")

    model_client = OpenAIChatCompletionClient(
        model="microsoft/Phi-3.5-mini-instruct",
        api_key=api_key,
        base_url="https://api-inference.huggingface.co/v1/",
        model_info={
            "function_calling": False,
            "json_output": False,
            "vision": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
    )

    # Test basic completion
    create_result = await model_client.create(
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Explain to me how AI works.", source="user"),
        ]
    )
    assert isinstance(create_result.content, str)
    assert len(create_result.content) > 0

async def test_ollama() -> None:
    model = "deepseek-r1:1.5b"
    model_info: ModelInfo = {
        "function_calling": False,
        "json_output": False,
        "vision": False,
        "family": ModelFamily.R1,
        "structured_output": False,
    }
    # Check if the model is running locally.
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:11434/v1/models/{model}")
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        pytest.skip(f"{model} model is not running locally: {e}")
    except httpx.ConnectError as e:
        pytest.skip(f"Ollama is not running locally: {e}")

    model_client = OpenAIChatCompletionClient(
        model=model,
        api_key="placeholder",
        base_url="http://localhost:11434/v1",
        model_info=model_info,
    )

    # Test basic completion with the Ollama deepseek-r1:1.5b model.
    create_result = await model_client.create(
        messages=[
            UserMessage(
                content="Taking two balls from a bag of 10 green balls and 20 red balls, "
                "what is the probability of getting a green and a red balls?",
                source="user",
            ),
        ]
    )
    assert isinstance(create_result.content, str)
    assert len(create_result.content) > 0
    assert create_result.finish_reason == "stop"
    assert create_result.usage is not None
    if model_info["family"] == ModelFamily.R1:
        assert create_result.thought is not None

    # Test streaming completion with the Ollama deepseek-r1:1.5b model.
    chunks: List[str | CreateResult] = []
    async for chunk in model_client.create_stream(
        messages=[
            UserMessage(
                content="Taking two balls from a bag of 10 green balls and 20 red balls, "
                "what is the probability of getting a green and a red balls?",
                source="user",
            ),
        ]
    ):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert isinstance(chunks[-1], CreateResult)
    assert chunks[-1].finish_reason == "stop"
    assert len(chunks[-1].content) > 0
    assert chunks[-1].usage is not None
    if model_info["family"] == ModelFamily.R1:
        assert chunks[-1].thought is not None

def test_openai_model_registry_find_well() -> None:
    model = "gpt-4o"
    client1 = OpenAIChatCompletionClient(model=model, api_key="test")
    client2 = OpenAIChatCompletionClient(
        model=model,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
            "family": ModelFamily.UNKNOWN,
        },
        api_key="test",
    )

    def get_regitered_transformer(client: OpenAIChatCompletionClient) -> TransformerMap:
        model_name = client._create_args["model"]  # pyright: ignore[reportPrivateUsage]
        model_family = client.model_info["family"]
        return get_transformer("openai", model_name, model_family)

    assert get_regitered_transformer(client1) == get_regitered_transformer(client2)

def test_rstrip_railing_whitespace_at_last_assistant_content() -> None:
    messages: list[LLMMessage] = [
        UserMessage(content="foo", source="user"),
        UserMessage(content="bar", source="user"),
        AssistantMessage(content="foobar ", source="assistant"),
    ]

    # This will crash if _rstrip_railing_whitespace_at_last_assistant_content is not applied to "content"
    dummy_client = OpenAIChatCompletionClient(model="claude-3-5-haiku-20241022", api_key="dummy-key")
    result = dummy_client._rstrip_last_assistant_message(messages)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result[-1].content, str)
    assert result[-1].content == "foobar"

async def test_openai_tool_choice_specific_tool_integration() -> None:
    """Test tool_choice parameter with a specific tool using the actual OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment variables")

    def _pass_function(input: str) -> str:
        """Simple passthrough function."""
        return f"Processed: {input}"

    def _add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    model = "gpt-4o-mini"
    client = OpenAIChatCompletionClient(model=model, api_key=api_key)

    # Define tools
    pass_tool = FunctionTool(_pass_function, description="Process input text", name="_pass_function")
    add_tool = FunctionTool(_add_numbers, description="Add two numbers together", name="_add_numbers")

    # Test forcing use of specific tool
    result = await client.create(
        messages=[UserMessage(content="Process the word 'hello'", source="user")],
        tools=[pass_tool, add_tool],
        tool_choice=pass_tool,  # Force use of specific tool
    )

    assert isinstance(result.content, list)
    assert len(result.content) == 1
    assert isinstance(result.content[0], FunctionCall)
    assert result.content[0].name == "_pass_function"
    assert result.finish_reason == "function_calls"
    assert result.usage is not None

async def test_openai_tool_choice_auto_integration() -> None:
    """Test tool_choice parameter with 'auto' setting using the actual OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment variables")

    def _pass_function(input: str) -> str:
        """Simple passthrough function."""
        return f"Processed: {input}"

    def _add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    model = "gpt-4o-mini"
    client = OpenAIChatCompletionClient(model=model, api_key=api_key)

    # Define tools
    pass_tool = FunctionTool(_pass_function, description="Process input text", name="_pass_function")
    add_tool = FunctionTool(_add_numbers, description="Add two numbers together", name="_add_numbers")

    # Test auto tool choice - model should choose to use add_numbers for math
    result = await client.create(
        messages=[UserMessage(content="What is 15 plus 27?", source="user")],
        tools=[pass_tool, add_tool],
        tool_choice="auto",  # Let model choose
    )

    assert isinstance(result.content, list)
    assert len(result.content) == 1
    assert isinstance(result.content[0], FunctionCall)
    assert result.content[0].name == "_add_numbers"
    assert result.finish_reason == "function_calls"
    assert result.usage is not None

    # Parse arguments to verify correct values
    args = json.loads(result.content[0].arguments)
    assert args["a"] == 15
    assert args["b"] == 27

async def test_openai_tool_choice_none_integration() -> None:
    """Test tool_choice parameter with 'none' setting using the actual OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment variables")

    def _pass_function(input: str) -> str:
        """Simple passthrough function."""
        return f"Processed: {input}"

    model = "gpt-4o-mini"
    client = OpenAIChatCompletionClient(model=model, api_key=api_key)

    # Define tools
    pass_tool = FunctionTool(_pass_function, description="Process input text", name="_pass_function")

    # Test none tool choice - model should not use any tools
    result = await client.create(
        messages=[UserMessage(content="Hello there, how are you?", source="user")],
        tools=[pass_tool],
        tool_choice="none",  # Disable tool usage
    )

    assert isinstance(result.content, str)
    assert len(result.content) > 0
    assert result.finish_reason == "stop"
    assert result.usage is not None

async def test_openai_tool_choice_required_integration() -> None:
    """Test tool_choice parameter with 'required' setting using the actual OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment variables")

    def _pass_function(input: str) -> str:
        """Simple passthrough function."""
        return f"Processed: {input}"

    def _add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    model = "gpt-4o-mini"
    client = OpenAIChatCompletionClient(model=model, api_key=api_key)

    # Define tools
    pass_tool = FunctionTool(_pass_function, description="Process input text", name="_pass_function")
    add_tool = FunctionTool(_add_numbers, description="Add two numbers together", name="_add_numbers")

    # Test required tool choice - model must use a tool even for general conversation
    result = await client.create(
        messages=[UserMessage(content="Say hello to me", source="user")],
        tools=[pass_tool, add_tool],
        tool_choice="required",  # Force tool usage
    )

    assert isinstance(result.content, list)
    assert len(result.content) == 1
    assert isinstance(result.content[0], FunctionCall)
    assert result.content[0].name in ["_pass_function", "_add_numbers"]
    assert result.finish_reason == "function_calls"
    assert result.usage is not None

async def test_openai_tool_choice_validation_error_integration() -> None:
    """Test tool_choice validation with invalid tool reference using the actual OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment variables")

    def _pass_function(input: str) -> str:
        """Simple passthrough function."""
        return f"Processed: {input}"

    def _add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    def _different_function(text: str) -> str:
        """Different function."""
        return text

    model = "gpt-4o-mini"
    client = OpenAIChatCompletionClient(model=model, api_key=api_key)

    # Define tools
    pass_tool = FunctionTool(_pass_function, description="Process input text", name="_pass_function")
    add_tool = FunctionTool(_add_numbers, description="Add two numbers together", name="_add_numbers")
    different_tool = FunctionTool(_different_function, description="Different tool", name="_different_function")

    messages = [UserMessage(content="Hello there", source="user")]

    # Test with a tool that's not in the tools list
    with pytest.raises(
        ValueError, match="tool_choice references '_different_function' but it's not in the provided tools"
    ):
        await client.create(
            messages=messages,
            tools=[pass_tool, add_tool],
            tool_choice=different_tool,  # This tool is not in the tools list
        )


# --- python/packages/autogen-ext/tests/models/test_sk_chat_completion_adapter.py ---

async def test_sk_chat_completion_with_tools(sk_client: AzureChatCompletion) -> None:
    # Create adapter
    adapter = SKChatCompletionAdapter(sk_client)

    # Create kernel
    kernel = Kernel(memory=NullMemory())

    # Create calculator tool instance
    tool = CalculatorTool()

    # Test messages
    messages: list[LLMMessage] = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is 2 + 2?", source="user"),
    ]

    # Call create with tool
    result = await adapter.create(messages=messages, tools=[tool], extra_create_args={"kernel": kernel})

    # Verify response
    assert isinstance(result.content, list)
    assert result.finish_reason == "function_calls"
    assert result.usage.prompt_tokens >= 0
    assert result.usage.completion_tokens >= 0
    assert not result.cached

async def test_sk_chat_completion_with_prompt_tools(sk_client: AzureChatCompletion) -> None:
    # Create adapter
    adapter = SKChatCompletionAdapter(sk_client)

    # Create kernel
    kernel = Kernel(memory=NullMemory())

    # Create calculator tool instance
    tool: ToolSchema = ToolSchema(
        name="calculator",
        description="Add two numbers together",
        parameters=ParametersSchema(
            type="object",
            properties={
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            required=["a", "b"],
        ),
    )

    # Test messages
    messages: list[LLMMessage] = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is 2 + 2?", source="user"),
    ]

    # Call create with tool
    result = await adapter.create(messages=messages, tools=[tool], extra_create_args={"kernel": kernel})

    # Verify response
    assert isinstance(result.content, list)
    assert result.finish_reason == "function_calls"
    assert result.usage.prompt_tokens >= 0
    assert result.usage.completion_tokens >= 0
    assert not result.cached

async def test_sk_chat_completion_stream_with_tools(sk_client: AzureChatCompletion) -> None:
    # Create adapter and kernel
    adapter = SKChatCompletionAdapter(sk_client)
    kernel = Kernel(memory=NullMemory())

    # Create calculator tool
    tool = CalculatorTool()

    # Test messages
    messages: list[LLMMessage] = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is 2 + 2?", source="user"),
    ]

    # Call create_stream with tool
    response_chunks: list[CreateResult | str] = []
    async for chunk in adapter.create_stream(messages=messages, tools=[tool], extra_create_args={"kernel": kernel}):
        response_chunks.append(chunk)

    # Verify response
    assert len(response_chunks) > 0
    final_chunk = response_chunks[-1]
    assert isinstance(final_chunk, CreateResult)
    assert isinstance(final_chunk.content, list)  # Function calls
    assert final_chunk.finish_reason == "function_calls"
    assert final_chunk.usage.prompt_tokens >= 0
    assert final_chunk.usage.completion_tokens >= 0
    assert not final_chunk.cached

async def test_sk_chat_completion_default_model_info(sk_client: AzureChatCompletion) -> None:
    # Create adapter with default model_info
    adapter = SKChatCompletionAdapter(sk_client)

    # Verify default model_info values
    assert adapter.model_info["vision"] is False
    assert adapter.model_info["function_calling"] is False
    assert adapter.model_info["json_output"] is False
    assert adapter.model_info["family"] == ModelFamily.UNKNOWN

    # Verify capabilities returns the same ModelInfo
    assert adapter.capabilities == adapter.model_info

async def test_sk_chat_completion_custom_model_info(sk_client: AzureChatCompletion) -> None:
    # Create custom model info
    custom_model_info = ModelInfo(
        vision=True, function_calling=True, json_output=True, family=ModelFamily.GPT_4, structured_output=False
    )

    # Create adapter with custom model_info
    adapter = SKChatCompletionAdapter(sk_client, model_info=custom_model_info)

    # Verify custom model_info values
    assert adapter.model_info["vision"] is True
    assert adapter.model_info["function_calling"] is True
    assert adapter.model_info["json_output"] is True
    assert adapter.model_info["family"] == ModelFamily.GPT_4

    # Verify capabilities returns the same ModelInfo
    assert adapter.capabilities == adapter.model_info


# --- python/packages/autogen-ext/tests/test_filesurfer_agent.py ---

async def test_file_surfer_serialization() -> None:
    """Test that FileSurfer can be serialized and deserialized properly."""
    model = "gpt-4.1-nano-2025-04-14"
    agent = FileSurfer(
        "FileSurfer",
        model_client=OpenAIChatCompletionClient(model=model, api_key=""),
    )

    # Serialize the agent
    serialized_agent = agent.dump_component()

    # Deserialize the agent
    deserialized_agent = FileSurfer.load_component(serialized_agent)

    # Check that the deserialized agent has the same attributes as the original agent
    assert isinstance(deserialized_agent, FileSurfer)


# --- python/packages/autogen-ext/tests/test_openai_agent.py ---

async def test_tool_calling(agent: OpenAIAgent, cancellation_token: CancellationToken) -> None:
    """Test that enabling a built-in tool yields a tool-style JSON response via the Responses API."""
    message = TextMessage(source="user", content="What's the weather in New York?")

    all_messages: List[Any] = []
    async for msg in agent.on_messages_stream([message], cancellation_token):
        all_messages.append(msg)

    final_response = next((msg for msg in all_messages if hasattr(msg, "chat_message")), None)
    assert final_response is not None
    assert hasattr(final_response, "chat_message")
    response_msg = cast(Response, final_response)
    assert isinstance(response_msg.chat_message, TextMessage)
    assert response_msg.chat_message.content == '{"temperature": 72.5, "conditions": "sunny"}'

async def test_error_handling(error_agent: OpenAIAgent, cancellation_token: CancellationToken) -> None:
    """Test that the agent returns an error message if the Responses API fails."""
    message = TextMessage(source="user", content="This will cause an error")

    all_messages: List[Any] = []
    async for msg in error_agent.on_messages_stream([message], cancellation_token):
        all_messages.append(msg)

    final_response = next((msg for msg in all_messages if hasattr(msg, "chat_message")), None)
    assert final_response is not None
    assert isinstance(final_response.chat_message, TextMessage)
    assert "Error generating response:" in final_response.chat_message.content

async def test_build_api_params(agent: OpenAIAgent) -> None:
    agent._last_response_id = None  # type: ignore
    params = agent._build_api_parameters([{"role": "user", "content": "hi"}])  # type: ignore
    assert "previous_response_id" not in params
    agent._last_response_id = "resp-456"  # type: ignore
    params = agent._build_api_parameters([{"role": "user", "content": "hi"}])  # type: ignore
    assert params.get("previous_response_id") == "resp-456"

    assert "max_tokens" not in params
    assert params.get("max_output_tokens") == 1000

    assert params.get("store") is True
    assert params.get("truncation") == "auto"

    agent._json_mode = True  # type: ignore
    params = agent._build_api_parameters([{"role": "user", "content": "hi"}])  # type: ignore
    assert "text.format" not in params
    assert params.get("text") == {"type": "json_object"}


# --- python/packages/autogen-ext/tests/test_openai_assistant_agent.py ---

async def test_on_reset_behavior(client: AsyncOpenAI, cancellation_token: CancellationToken) -> None:
    # Arrange: Use the default behavior for reset.
    thread = await client.beta.threads.create()  # type: ignore[reportDeprecated]
    await client.beta.threads.messages.create(  # type: ignore[reportDeprecated]
        thread_id=thread.id,
        content="Hi, my name is John and I'm a software engineer. Use this information to help me.",
        role="user",
    )

    agent = OpenAIAssistantAgent(
        name="assistant",
        instructions="Help the user with their task.",
        model="gpt-4.1-nano",
        description="OpenAI Assistant Agent",
        client=client,
        thread_id=thread.id,
    )

    message1 = TextMessage(source="user", content="What is my name?")
    response1 = await agent.on_messages([message1], cancellation_token)
    assert isinstance(response1.chat_message, TextMessage)
    assert "john" in response1.chat_message.content.lower()

    await agent.on_reset(cancellation_token)

    message2 = TextMessage(source="user", content="What is my name?")
    response2 = await agent.on_messages([message2], cancellation_token)
    assert isinstance(response2.chat_message, TextMessage)
    assert "john" in response2.chat_message.content.lower()

    await agent.delete_assistant(cancellation_token)

