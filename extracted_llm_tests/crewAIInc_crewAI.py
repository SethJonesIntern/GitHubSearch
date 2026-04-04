# crewAIInc/crewAI
# 70 test functions with real LLM calls
# Source: https://github.com/crewAIInc/crewAI


# --- lib/crewai/tests/llms/anthropic/test_anthropic.py ---

def test_anthropic_completion_is_used_when_claude_provider():
    """
    Test that AnthropicCompletion is used when provider is 'claude'
    """
    llm = LLM(model="claude/claude-3-5-sonnet-20241022")

    from crewai.llms.providers.anthropic.completion import AnthropicCompletion
    assert isinstance(llm, AnthropicCompletion)
    assert llm.provider == "anthropic"
    assert llm.model == "claude-3-5-sonnet-20241022"

def test_anthropic_completion_module_is_imported():
    """
    Test that the completion module is properly imported when using Anthropic provider
    """
    module_name = "crewai.llms.providers.anthropic.completion"

    # Remove module from cache if it exists
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Create LLM instance - this should trigger the import
    LLM(model="anthropic/claude-3-5-sonnet-20241022")

    # Verify the module was imported
    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)

    # Verify the class exists in the module
    assert hasattr(completion_mod, 'AnthropicCompletion')

def test_anthropic_completion_initialization_parameters():
    """
    Test that AnthropicCompletion is initialized with correct parameters
    """
    llm = LLM(
        model="anthropic/claude-3-5-sonnet-20241022",
        temperature=0.7,
        max_tokens=2000,
        top_p=0.9,
        api_key="test-key"
    )

    from crewai.llms.providers.anthropic.completion import AnthropicCompletion
    assert isinstance(llm, AnthropicCompletion)
    assert llm.model == "claude-3-5-sonnet-20241022"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 2000
    assert llm.top_p == 0.9

def test_anthropic_specific_parameters():
    """
    Test Anthropic-specific parameters like stop_sequences and streaming
    """
    llm = LLM(
        model="anthropic/claude-3-5-sonnet-20241022",
        stop_sequences=["Human:", "Assistant:"],
        stream=True,
        max_retries=5,
        timeout=60
    )

    from crewai.llms.providers.anthropic.completion import AnthropicCompletion
    assert isinstance(llm, AnthropicCompletion)
    assert llm.stop_sequences == ["Human:", "Assistant:"]
    assert llm.stream == True
    assert llm._client.max_retries == 5
    assert llm._client.timeout == 60

def test_anthropic_model_detection():
    """
    Test that various Anthropic model formats are properly detected
    """
    # Test Anthropic model naming patterns that actually work with provider detection
    anthropic_test_cases = [
        "anthropic/claude-3-5-sonnet-20241022",
        "claude/claude-3-5-sonnet-20241022"
    ]

    for model_name in anthropic_test_cases:
        llm = LLM(model=model_name)
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion
        assert isinstance(llm, AnthropicCompletion), f"Failed for model: {model_name}"

def test_anthropic_streaming_parameter():
    """
    Test that streaming parameter is properly handled
    """
    # Test non-streaming
    llm_no_stream = LLM(model="anthropic/claude-3-5-sonnet-20241022", stream=False)
    assert llm_no_stream.stream == False

    # Test streaming
    llm_stream = LLM(model="anthropic/claude-3-5-sonnet-20241022", stream=True)
    assert llm_stream.stream == True

def test_anthropic_cached_prompt_tokens():
    """
    Test that Anthropic correctly extracts and tracks cached_prompt_tokens
    from cache_read_input_tokens. Uses cache_control to enable prompt caching
    and sends the same large prompt twice so the second call hits the cache.
    """
    # Anthropic requires cache_control blocks and >=1024 tokens for caching
    padding = "This is padding text to ensure the prompt is large enough for caching. " * 80
    system_msg = f"You are a helpful assistant. {padding}"

    llm = LLM(model="anthropic/claude-sonnet-4-5-20250929")

    def _ephemeral_user(text: str):
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    # First call: creates the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": _ephemeral_user("Say hello in one word.")},
    ])

    # Second call: same system prompt should hit the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": _ephemeral_user("Say goodbye in one word.")},
    ])

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.successful_requests == 2
    # The second call should have cached prompt tokens
    assert usage.cached_prompt_tokens > 0

def test_anthropic_streaming_cached_prompt_tokens():
    """
    Test that Anthropic streaming correctly extracts and tracks cached_prompt_tokens.
    """
    padding = "This is padding text to ensure the prompt is large enough for caching. " * 80
    system_msg = f"You are a helpful assistant. {padding}"

    llm = LLM(model="anthropic/claude-sonnet-4-5-20250929", stream=True)

    def _ephemeral_user(text: str):
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    # First call: creates the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": _ephemeral_user("Say hello in one word.")},
    ])

    # Second call: same system prompt should hit the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": _ephemeral_user("Say goodbye in one word.")},
    ])

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.successful_requests == 2
    # The second call should have cached prompt tokens
    assert usage.cached_prompt_tokens > 0

def test_tool_search_true_injects_bm25_and_defer_loading():
    """tool_search=True should inject bm25 tool search and defer all tools."""
    llm = LLM(model="anthropic/claude-sonnet-4-5", tool_search=True)

    crewai_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform math calculations",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        },
    ]

    formatted_messages, system_message = llm._format_messages_for_anthropic(
        [{"role": "user", "content": "Hello"}]
    )
    params = llm._prepare_completion_params(
        formatted_messages, system_message, crewai_tools
    )

    tools = params["tools"]
    # Should have 3 tools: tool_search + 2 regular
    assert len(tools) == 3

    # First tool should be the bm25 tool search tool
    assert tools[0]["type"] == "tool_search_tool_bm25_20251119"
    assert tools[0]["name"] == "tool_search_tool_bm25"
    assert "input_schema" not in tools[0]

    # All regular tools should have defer_loading=True
    for t in tools[1:]:
        assert t.get("defer_loading") is True, f"Tool {t['name']} missing defer_loading"

def test_tool_search_regex_config():
    """tool_search with regex config should use regex variant."""
    from crewai.llms.providers.anthropic.completion import AnthropicToolSearchConfig

    config = AnthropicToolSearchConfig(type="regex")
    llm = LLM(model="anthropic/claude-sonnet-4-5", tool_search=config)

    crewai_tools = [
        {
            "type": "function",
            "function": {
                "name": "tool_a",
                "description": "First tool",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_b",
                "description": "Second tool",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        },
    ]

    formatted_messages, system_message = llm._format_messages_for_anthropic(
        [{"role": "user", "content": "Hello"}]
    )
    params = llm._prepare_completion_params(
        formatted_messages, system_message, crewai_tools
    )

    tools = params["tools"]
    assert tools[0]["type"] == "tool_search_tool_regex_20251119"
    assert tools[0]["name"] == "tool_search_tool_regex"

def test_tool_search_disabled_by_default():
    """tool_search=None (default) should NOT inject anything."""
    llm = LLM(model="anthropic/claude-sonnet-4-5")

    crewai_tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        },
    ]

    formatted_messages, system_message = llm._format_messages_for_anthropic(
        [{"role": "user", "content": "Hello"}]
    )
    params = llm._prepare_completion_params(
        formatted_messages, system_message, crewai_tools
    )

    tools = params["tools"]
    assert len(tools) == 1
    for t in tools:
        assert t.get("type", "") not in (
            "tool_search_tool_bm25_20251119",
            "tool_search_tool_regex_20251119",
        )
        assert "defer_loading" not in t

def test_tool_search_no_duplicate_when_manually_provided():
    """If user passes a tool search tool manually, don't inject a duplicate."""
    llm = LLM(model="anthropic/claude-sonnet-4-5", tool_search=True)

    # User manually includes a tool search tool
    tools_with_search = [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        },
    ]

    formatted_messages, system_message = llm._format_messages_for_anthropic(
        [{"role": "user", "content": "Hello"}]
    )
    params = llm._prepare_completion_params(
        formatted_messages, system_message, tools_with_search
    )

    tools = params["tools"]
    search_tools = [
        t for t in tools
        if t.get("type", "").startswith("tool_search_tool")
    ]
    # Should only have 1 tool search tool (the user's manual one)
    assert len(search_tools) == 1
    assert search_tools[0]["type"] == "tool_search_tool_regex_20251119"

def test_tool_search_single_tool_skips_search_and_forces_choice():
    """With only 1 tool, tool_search is skipped (nothing to search) and the
    normal forced tool_choice optimisation still applies."""
    llm = LLM(model="anthropic/claude-sonnet-4-5", tool_search=True)

    crewai_tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        },
    ]

    formatted_messages, system_message = llm._format_messages_for_anthropic(
        [{"role": "user", "content": "Hello"}]
    )
    params = llm._prepare_completion_params(
        formatted_messages,
        system_message,
        crewai_tools,
        available_functions={"test_tool": lambda q: "result"},
    )

    # Single tool — tool_search skipped, tool_choice forced as normal
    assert "tool_choice" in params
    assert params["tool_choice"]["name"] == "test_tool"

    # No tool search tool should be injected
    tool_types = [t.get("type", "") for t in params["tools"]]
    for ts_type in ("tool_search_tool_bm25_20251119", "tool_search_tool_regex_20251119"):
        assert ts_type not in tool_types

    # No defer_loading on the single tool
    assert "defer_loading" not in params["tools"][0]

def test_tool_search_via_llm_class():
    """Verify tool_search param passes through LLM class correctly."""
    from crewai.llms.providers.anthropic.completion import (
        AnthropicCompletion,
        AnthropicToolSearchConfig,
    )

    # Test with True
    llm = LLM(model="anthropic/claude-sonnet-4-5", tool_search=True)
    assert isinstance(llm, AnthropicCompletion)
    assert llm.tool_search is not None
    assert llm.tool_search.type == "bm25"

    # Test with config
    llm2 = LLM(
        model="anthropic/claude-sonnet-4-5",
        tool_search=AnthropicToolSearchConfig(type="regex"),
    )
    assert llm2.tool_search is not None
    assert llm2.tool_search.type == "regex"

    # Test without (default)
    llm3 = LLM(model="anthropic/claude-sonnet-4-5")
    assert llm3.tool_search is None


# --- lib/crewai/tests/llms/openai/test_openai.py ---

def test_openai_completion_is_used_when_no_provider_prefix():
    """
    Test that OpenAICompletion is used when no provider prefix is given (defaults to openai)
    """
    llm = LLM(model="gpt-4o")

    from crewai.llms.providers.openai.completion import OpenAICompletion
    assert isinstance(llm, OpenAICompletion)
    assert llm.provider == "openai"
    assert llm.model == "gpt-4o"

def test_openai_completion_module_is_imported():
    """
    Test that the completion module is properly imported when using OpenAI provider
    """
    module_name = "crewai.llms.providers.openai.completion"

    # Remove module from cache if it exists
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Create LLM instance - this should trigger the import
    LLM(model="gpt-4o")

    # Verify the module was imported
    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)

    # Verify the class exists in the module
    assert hasattr(completion_mod, 'OpenAICompletion')

def test_openai_completion_initialization_parameters():
    """
    Test that OpenAICompletion is initialized with correct parameters
    """
    llm = LLM(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000,
        api_key="test-key"
    )

    from crewai.llms.providers.openai.completion import OpenAICompletion
    assert isinstance(llm, OpenAICompletion)
    assert llm.model == "gpt-4o"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 1000

def test_openai_completion_call_returns_usage_metrics():
    """
    Test that OpenAICompletion.call returns usage metrics
    """
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the population of Tokyo",
        backstory="You are a helpful research assistant.",
        llm=LLM(model="gpt-4o"),
        verbose=True,
    )

    task = Task(
        description="Find information about the population of Tokyo",
        expected_output="The population of Tokyo is 10 million",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    assert result.token_usage is not None
    assert result.token_usage.total_tokens == 289
    assert result.token_usage.prompt_tokens == 173
    assert result.token_usage.completion_tokens == 116
    assert result.token_usage.successful_requests == 1
    assert result.token_usage.cached_prompt_tokens == 0

def test_openai_get_client_params_with_api_base():
    """
    Test that _get_client_params correctly converts api_base to base_url
    """
    llm = OpenAICompletion(
        model="gpt-4o",
        api_base="https://custom.openai.com/v1",
    )
    client_params = llm._get_client_params()
    assert client_params["base_url"] == "https://custom.openai.com/v1"

def test_openai_get_client_params_with_base_url_priority():
    """
    Test that base_url takes priority over api_base in _get_client_params
    """
    llm = OpenAICompletion(
        model="gpt-4o",
        base_url="https://priority.openai.com/v1",
        api_base="https://fallback.openai.com/v1",
    )
    client_params = llm._get_client_params()
    assert client_params["base_url"] == "https://priority.openai.com/v1"

def test_openai_streaming_returns_usage_metrics():
    """
    Test that OpenAI streaming calls return proper token usage metrics.
    """
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the capital of France",
        backstory="You are a helpful research assistant.",
        llm=LLM(model="gpt-4o-mini", stream=True),
        verbose=True,
    )

    task = Task(
        description="What is the capital of France?",
        expected_output="The capital of France",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()

    assert result.token_usage is not None
    assert result.token_usage.total_tokens > 0
    assert result.token_usage.prompt_tokens > 0
    assert result.token_usage.completion_tokens > 0
    assert result.token_usage.successful_requests >= 1

def test_openai_responses_api_initialization():
    """Test that OpenAI Responses API can be initialized with api='responses'."""
    llm = OpenAICompletion(
        model="gpt-5",
        api="responses",
        instructions="You are a helpful assistant.",
        store=True,
    )

    assert llm.api == "responses"
    assert llm.instructions == "You are a helpful assistant."
    assert llm.store is True
    assert llm.model == "gpt-5"

def test_openai_responses_api_default_is_completions():
    """Test that the default API is 'completions' for backward compatibility."""
    llm = OpenAICompletion(model="gpt-4o")

    assert llm.api == "completions"

def test_openai_responses_api_prepare_params():
    """Test that Responses API params are prepared correctly."""
    llm = OpenAICompletion(
        model="gpt-5",
        api="responses",
        instructions="Base instructions.",
        store=True,
        temperature=0.7,
    )

    messages = [
        {"role": "system", "content": "System message."},
        {"role": "user", "content": "Hello!"},
    ]

    params = llm._prepare_responses_params(messages)

    assert params["model"] == "gpt-5"
    assert "Base instructions." in params["instructions"]
    assert "System message." in params["instructions"]
    assert params["store"] is True
    assert params["temperature"] == 0.7
    assert params["input"] == [{"role": "user", "content": "Hello!"}]

def test_openai_responses_api_tool_format():
    """Test that tools are converted to Responses API format (internally-tagged)."""
    llm = OpenAICompletion(model="gpt-5", api="responses")

    tools = [
        {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]

    responses_tools = llm._convert_tools_for_responses(tools)

    assert len(responses_tools) == 1
    tool = responses_tools[0]
    assert tool["type"] == "function"
    assert tool["name"] == "get_weather"
    assert tool["description"] == "Get the weather for a location"
    assert "parameters" in tool
    assert "function" not in tool

def test_openai_completions_api_tool_format():
    """Test that tools are converted to Chat Completions API format (externally-tagged)."""
    llm = OpenAICompletion(model="gpt-4o", api="completions")

    tools = [
        {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]

    completions_tools = llm._convert_tools_for_interference(tools)

    assert len(completions_tools) == 1
    tool = completions_tools[0]
    assert tool["type"] == "function"
    assert "function" in tool
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["description"] == "Get the weather for a location"

def test_openai_responses_api_structured_output_format():
    """Test that structured outputs use text.format for Responses API."""
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    llm = OpenAICompletion(model="gpt-5", api="responses")

    messages = [{"role": "user", "content": "Extract: Jane, 25"}]
    params = llm._prepare_responses_params(messages, response_model=Person)

    assert "text" in params
    assert "format" in params["text"]
    assert params["text"]["format"]["type"] == "json_schema"
    assert params["text"]["format"]["name"] == "Person"
    assert params["text"]["format"]["strict"] is True

def test_openai_responses_api_with_previous_response_id():
    """Test that previous_response_id is passed for multi-turn conversations."""
    llm = OpenAICompletion(
        model="gpt-5",
        api="responses",
        previous_response_id="resp_abc123",
        store=True,
    )

    messages = [{"role": "user", "content": "Continue our conversation."}]
    params = llm._prepare_responses_params(messages)

    assert params["previous_response_id"] == "resp_abc123"
    assert params["store"] is True

def test_openai_responses_api_basic_call():
    """Test basic Responses API call with text generation."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        instructions="You are a helpful assistant. Be concise.",
    )

    result = llm.call("What is 2 + 2? Answer with just the number.")

    assert isinstance(result, str)
    assert "4" in result

def test_openai_responses_api_with_structured_output():
    """Test Responses API with structured output using Pydantic model."""
    from pydantic import BaseModel, Field

    class MathAnswer(BaseModel):
        """Structured math answer."""

        result: int = Field(description="The numerical result")
        explanation: str = Field(description="Brief explanation")

    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
    )

    result = llm.call("What is 5 * 7?", response_model=MathAnswer)

    assert isinstance(result, MathAnswer)
    assert result.result == 35

def test_openai_responses_api_with_system_message_extraction():
    """Test that system messages are properly extracted to instructions."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
    )

    messages = [
        {"role": "system", "content": "You always respond in uppercase letters only."},
        {"role": "user", "content": "Say hello"},
    ]

    result = llm.call(messages)

    assert isinstance(result, str)
    assert result.isupper() or "HELLO" in result.upper()

def test_openai_responses_api_streaming():
    """Test Responses API with streaming enabled."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        stream=True,
        instructions="Be very concise.",
    )

    result = llm.call("Count from 1 to 3, separated by commas.")

    assert isinstance(result, str)
    assert "1" in result
    assert "2" in result
    assert "3" in result

def test_openai_responses_api_returns_usage_metrics():
    """Test that Responses API calls return proper token usage metrics."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
    )

    llm.call("Say hello")

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0

def test_openai_responses_api_builtin_tools_param():
    """Test that builtin_tools parameter is properly configured."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        builtin_tools=["web_search", "code_interpreter"],
    )

    assert llm.builtin_tools == ["web_search", "code_interpreter"]

    messages = [{"role": "user", "content": "Test"}]
    params = llm._prepare_responses_params(messages)

    assert "tools" in params
    tool_types = [t["type"] for t in params["tools"]]
    assert "web_search_preview" in tool_types
    assert "code_interpreter" in tool_types

def test_openai_responses_api_builtin_tools_with_custom_tools():
    """Test that builtin_tools can be combined with custom function tools."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        builtin_tools=["web_search"],
    )

    custom_tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    messages = [{"role": "user", "content": "Test"}]
    params = llm._prepare_responses_params(messages, tools=custom_tools)

    assert len(params["tools"]) == 2
    tool_types = [t.get("type") for t in params["tools"]]
    assert "web_search_preview" in tool_types
    assert "function" in tool_types

def test_openai_responses_api_with_web_search():
    """Test Responses API with web_search built-in tool."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        builtin_tools=["web_search"],
    )

    result = llm.call("What is the current population of Tokyo? Be brief.")

    assert isinstance(result, str)
    assert len(result) > 0

def test_responses_api_result_has_tool_outputs():
    """Test ResponsesAPIResult.has_tool_outputs() method."""
    result_with_web = ResponsesAPIResult(
        text="Test",
        web_search_results=[{"id": "ws_1", "status": "completed", "type": "web_search_call"}],
    )
    assert result_with_web.has_tool_outputs()

    result_with_file = ResponsesAPIResult(
        text="Test",
        file_search_results=[{"id": "fs_1", "status": "completed", "type": "file_search_call", "queries": [], "results": []}],
    )
    assert result_with_file.has_tool_outputs()

def test_openai_responses_api_parse_tool_outputs_param():
    """Test that parse_tool_outputs parameter is properly configured."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        parse_tool_outputs=True,
    )

    assert llm.parse_tool_outputs is True

def test_openai_responses_api_parse_tool_outputs_default_false():
    """Test that parse_tool_outputs defaults to False."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
    )

    assert llm.parse_tool_outputs is False

def test_openai_responses_api_with_parse_tool_outputs():
    """Test Responses API with parse_tool_outputs enabled returns ResponsesAPIResult."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        builtin_tools=["web_search"],
        parse_tool_outputs=True,
    )

    result = llm.call("What is the current population of Tokyo? Be very brief.")

    assert isinstance(result, ResponsesAPIResult)
    assert len(result.text) > 0
    assert result.response_id is not None
    # Web search should have been used
    assert len(result.web_search_results) > 0
    assert result.has_tool_outputs()

def test_openai_responses_api_parse_tool_outputs_basic_call():
    """Test Responses API with parse_tool_outputs but no built-in tools."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        parse_tool_outputs=True,
    )

    result = llm.call("Say hello in exactly 3 words.")

    assert isinstance(result, ResponsesAPIResult)
    assert len(result.text) > 0
    assert result.response_id is not None
    # No built-in tools used
    assert not result.has_tool_outputs()

def test_openai_responses_api_auto_chain_param():
    """Test that auto_chain parameter is properly configured."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=True,
    )

    assert llm.auto_chain is True
    assert llm._last_response_id is None

def test_openai_responses_api_auto_chain_default_false():
    """Test that auto_chain defaults to False."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
    )

    assert llm.auto_chain is False

def test_openai_responses_api_last_response_id_property():
    """Test last_response_id property."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=True,
    )

    # Initially None
    assert llm.last_response_id is None

    # Simulate setting the internal value
    llm._last_response_id = "resp_test_123"
    assert llm.last_response_id == "resp_test_123"

def test_openai_responses_api_reset_chain():
    """Test reset_chain() method clears the response ID."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=True,
    )

    # Set a response ID
    llm._last_response_id = "resp_test_123"
    assert llm.last_response_id == "resp_test_123"

    # Reset the chain
    llm.reset_chain()
    assert llm.last_response_id is None

def test_openai_responses_api_auto_chain_prepare_params():
    """Test that _prepare_responses_params uses auto-chained response ID."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=True,
    )

    # No previous response ID yet
    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert "previous_response_id" not in params

    # Set a previous response ID
    llm._last_response_id = "resp_previous_123"
    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert params.get("previous_response_id") == "resp_previous_123"

def test_openai_responses_api_explicit_previous_response_id_takes_precedence():
    """Test that explicit previous_response_id overrides auto-chained ID."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=True,
        previous_response_id="resp_explicit_456",
    )

    # Set an auto-chained response ID
    llm._last_response_id = "resp_auto_123"

    # Explicit should take precedence
    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert params.get("previous_response_id") == "resp_explicit_456"

def test_openai_responses_api_auto_chain_disabled_no_tracking():
    """Test that response ID is not tracked when auto_chain is False."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain=False,
    )

    # Even with a "previous" response ID set internally, params shouldn't use it
    llm._last_response_id = "resp_should_not_use"
    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert "previous_response_id" not in params

def test_openai_responses_api_auto_chain_integration():
    """Test auto-chaining tracks response IDs across calls."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        auto_chain=True,
    )

    # First call - should not have previous_response_id
    assert llm.last_response_id is None
    result1 = llm.call("My name is Alice. Remember this.")

    # After first call, should have a response ID
    assert llm.last_response_id is not None
    first_response_id = llm.last_response_id
    assert first_response_id.startswith("resp_")

    # Second call - should use the first response ID
    result2 = llm.call("What is my name?")

    # Response ID should be updated
    assert llm.last_response_id is not None
    assert llm.last_response_id != first_response_id  # Should be a new ID

    # The response should remember context (Alice)
    assert isinstance(result1, str)
    assert isinstance(result2, str)

def test_openai_responses_api_auto_chain_with_reset():
    """Test that reset_chain() properly starts a new conversation."""
    llm = OpenAICompletion(
        model="gpt-4o-mini",
        api="responses",
        auto_chain=True,
    )

    # First conversation
    llm.call("My favorite color is blue.")
    first_chain_id = llm.last_response_id
    assert first_chain_id is not None

    # Reset and start new conversation
    llm.reset_chain()
    assert llm.last_response_id is None

    # New call should start fresh
    llm.call("Hello!")
    second_chain_id = llm.last_response_id
    assert second_chain_id is not None
    # New conversation, so different response ID
    assert second_chain_id != first_chain_id

def test_openai_responses_api_auto_chain_reasoning_param():
    """Test that auto_chain_reasoning parameter is properly configured."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=True,
    )

    assert llm.auto_chain_reasoning is True
    assert llm._last_reasoning_items is None

def test_openai_responses_api_auto_chain_reasoning_default_false():
    """Test that auto_chain_reasoning defaults to False."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
    )

    assert llm.auto_chain_reasoning is False

def test_openai_responses_api_auto_chain_reasoning_adds_include():
    """Test that auto_chain_reasoning adds reasoning.encrypted_content to include."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=True,
    )

    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert "include" in params
    assert "reasoning.encrypted_content" in params["include"]

def test_openai_responses_api_auto_chain_reasoning_preserves_existing_include():
    """Test that auto_chain_reasoning preserves existing include items."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=True,
        include=["file_search_call.results"],
    )

    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert "include" in params
    assert "reasoning.encrypted_content" in params["include"]
    assert "file_search_call.results" in params["include"]

def test_openai_responses_api_auto_chain_reasoning_no_duplicate_include():
    """Test that reasoning.encrypted_content is not duplicated if already in include."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=True,
        include=["reasoning.encrypted_content"],
    )

    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    assert "include" in params
    # Should only appear once
    assert params["include"].count("reasoning.encrypted_content") == 1

def test_openai_responses_api_auto_chain_reasoning_disabled_no_include():
    """Test that reasoning.encrypted_content is not added when auto_chain_reasoning is False."""
    llm = OpenAICompletion(
        model="gpt-4o",
        api="responses",
        auto_chain_reasoning=False,
    )

    params = llm._prepare_responses_params(messages=[{"role": "user", "content": "test"}])
    # Should not have include at all (unless explicitly set)
    assert "include" not in params or "reasoning.encrypted_content" not in params.get("include", [])

def test_openai_stop_words_not_applied_to_structured_output():
    """
    Test that stop words are NOT applied when response_model is provided.
    This ensures JSON responses containing stop word patterns (like "Observation:")
    are not truncated, which would cause JSON validation to fail.
    """
    from pydantic import BaseModel, Field

    class ResearchResult(BaseModel):
        """Research result that may contain stop word patterns in string fields."""

        finding: str = Field(description="The research finding")
        observation: str = Field(description="Observation about the finding")

    # Create OpenAI completion instance with stop words configured
    llm = OpenAICompletion(
        model="gpt-4o",
        stop=["Observation:", "Final Answer:"],  # Common stop words
    )

    # JSON response that contains a stop word pattern in a string field
    # Without the fix, this would be truncated at "Observation:" breaking the JSON
    json_response = '{"finding": "The data shows growth", "observation": "Observation: This confirms the hypothesis"}'

    # Test the _validate_structured_output method directly with content containing stop words
    # This simulates what happens when the API returns JSON with stop word patterns
    result = llm._validate_structured_output(json_response, ResearchResult)

    # Should successfully parse the full JSON without truncation
    assert isinstance(result, ResearchResult)
    assert result.finding == "The data shows growth"
    # The observation field should contain the full text including "Observation:"
    assert "Observation:" in result.observation

def test_openai_gpt5_models_do_not_support_stop_words():
    """
    Test that GPT-5 family models do not support stop words via the API.
    GPT-5 models reject the 'stop' parameter, so stop words must be
    applied client-side only.
    """
    gpt5_models = [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-pro",
        "gpt-5.1",
        "gpt-5.1-chat",
        "gpt-5.2",
        "gpt-5.2-chat",
    ]

    for model_name in gpt5_models:
        llm = OpenAICompletion(model=model_name)
        assert llm.supports_stop_words() == False, (
            f"Expected {model_name} to NOT support stop words"
        )

def test_openai_non_gpt5_models_support_stop_words():
    """
    Test that non-GPT-5 models still support stop words normally.
    """
    supported_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4-turbo",
    ]

    for model_name in supported_models:
        llm = OpenAICompletion(model=model_name)
        assert llm.supports_stop_words() == True, (
            f"Expected {model_name} to support stop words"
        )

def test_openai_gpt5_still_applies_stop_words_client_side():
    """
    Test that GPT-5 models still truncate responses at stop words client-side
    via _apply_stop_words(), even though they don't send 'stop' to the API.
    """
    llm = OpenAICompletion(
        model="gpt-5.2",
        stop=["Observation:", "Final Answer:"],
    )

    assert llm.supports_stop_words() == False

    response = "I need to search.\n\nAction: search\nObservation: Found results"
    result = llm._apply_stop_words(response)

    assert "Observation:" not in result
    assert "Found results" not in result
    assert "I need to search" in result

def test_openai_stop_words_still_applied_to_regular_responses():
    """
    Test that stop words ARE still applied for regular (non-structured) responses.
    This ensures the fix didn't break normal stop word behavior.
    """
    # Create OpenAI completion instance with stop words configured
    llm = OpenAICompletion(
        model="gpt-4o",
        stop=["Observation:", "Final Answer:"],
    )

    # Response that contains a stop word - should be truncated
    response_with_stop_word = "I need to search for more information.\n\nAction: search\nObservation: Found results"

    # Test the _apply_stop_words method directly
    result = llm._apply_stop_words(response_with_stop_word)

    # Response should be truncated at the stop word
    assert "Observation:" not in result
    assert "Found results" not in result
    assert "I need to search for more information" in result

def test_openai_structured_output_preserves_json_with_stop_word_patterns():
    """
    Test that structured output validation preserves JSON content
    even when string fields contain stop word patterns.
    """
    from pydantic import BaseModel, Field

    class AgentObservation(BaseModel):
        """Model with fields that might contain stop word-like text."""

        action_taken: str = Field(description="What action was taken")
        observation_result: str = Field(description="The observation result")
        final_answer: str = Field(description="The final answer")

    llm = OpenAICompletion(
        model="gpt-4o",
        stop=["Observation:", "Final Answer:", "Action:"],
    )

    # JSON that contains all the stop word patterns as part of the content
    json_with_stop_patterns = '''{
        "action_taken": "Action: Searched the database",
        "observation_result": "Observation: Found 5 relevant results",
        "final_answer": "Final Answer: The data shows positive growth"
    }'''

    # This should NOT be truncated since it's structured output
    result = llm._validate_structured_output(json_with_stop_patterns, AgentObservation)

    assert isinstance(result, AgentObservation)
    assert "Action:" in result.action_taken
    assert "Observation:" in result.observation_result
    assert "Final Answer:" in result.final_answer

def test_openai_completions_cached_prompt_tokens():
    """
    Test that the Chat Completions API correctly extracts and tracks
    cached_prompt_tokens from prompt_tokens_details.cached_tokens.
    Sends the same large prompt twice so the second call hits the cache.
    """
    # Build a large system prompt to trigger prompt caching (>1024 tokens)
    padding = "This is padding text to ensure the prompt is large enough for caching. " * 80
    system_msg = f"You are a helpful assistant. {padding}"

    llm = OpenAICompletion(model="gpt-4.1")

    # First call: creates the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Say hello in one word."},
    ])

    # Second call: same system prompt should hit the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Say goodbye in one word."},
    ])

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.successful_requests == 2
    # The second call should have cached prompt tokens
    assert usage.cached_prompt_tokens > 0

def test_openai_responses_api_cached_prompt_tokens():
    """
    Test that the Responses API correctly extracts and tracks
    cached_prompt_tokens from input_tokens_details.cached_tokens.
    """
    padding = "This is padding text to ensure the prompt is large enough for caching. " * 80
    system_msg = f"You are a helpful assistant. {padding}"

    llm = OpenAICompletion(model="gpt-4.1", api="responses")

    # First call: creates the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Say hello in one word."},
    ])

    # Second call: same system prompt should hit the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Say goodbye in one word."},
    ])

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.successful_requests == 2
    # The second call should have cached prompt tokens
    assert usage.cached_prompt_tokens > 0

def test_openai_streaming_cached_prompt_tokens():
    """
    Test that streaming Chat Completions API correctly extracts and tracks
    cached_prompt_tokens.
    """
    padding = "This is padding text to ensure the prompt is large enough for caching. " * 80
    system_msg = f"You are a helpful assistant. {padding}"

    llm = OpenAICompletion(model="gpt-4.1", stream=True)

    # First call: creates the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Say hello in one word."},
    ])

    # Second call: same system prompt should hit the cache
    llm.call([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Say goodbye in one word."},
    ])

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.successful_requests == 2
    # The second call should have cached prompt tokens
    assert usage.cached_prompt_tokens > 0

def test_openai_completions_cached_prompt_tokens_with_tools():
    """
    Test that the Chat Completions API correctly tracks cached_prompt_tokens
    when tools are used. The large system prompt should be cached across calls.
    """
    padding = "This is padding text to ensure the prompt is large enough for caching. " * 80
    system_msg = f"You are a helpful assistant that uses tools. {padding}"

    def get_weather(location: str) -> str:
        return f"The weather in {location} is sunny and 72°F"

    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        }
    ]

    llm = OpenAICompletion(model="gpt-4.1")

    # First call with tool: creates the cache
    llm.call(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ],
        tools=tools,
        available_functions={"get_weather": get_weather},
    )

    # Second call with same system prompt + tools: should hit the cache
    llm.call(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "What is the weather in Paris?"},
        ],
        tools=tools,
        available_functions={"get_weather": get_weather},
    )

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.prompt_tokens > 0
    assert usage.successful_requests == 2
    # The second call should have cached prompt tokens
    assert usage.cached_prompt_tokens > 0

def test_openai_responses_api_cached_prompt_tokens_with_tools():
    """
    Test that the Responses API correctly tracks cached_prompt_tokens
    when function tools are used.
    """
    padding = "This is padding text to ensure the prompt is large enough for caching. " * 80
    system_msg = f"You are a helpful assistant that uses tools. {padding}"

    def get_weather(location: str) -> str:
        return f"The weather in {location} is sunny and 72°F"

    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"],
            },
        }
    ]

    llm = OpenAICompletion(model="gpt-4.1", api='responses')

    # First call with tool
    llm.call(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ],
        tools=tools,
        available_functions={"get_weather": get_weather},
    )

    # Second call: same system prompt + tools should hit cache
    llm.call(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "What is the weather in Paris?"},
        ],
        tools=tools,
        available_functions={"get_weather": get_weather},
    )

    usage = llm.get_token_usage_summary()
    assert usage.total_tokens > 0
    assert usage.successful_requests == 2
    assert usage.cached_prompt_tokens > 0


# --- lib/crewai/tests/test_llm.py ---

def test_anthropic_message_formatting_edge_cases(anthropic_llm):
    """Test edge cases for Anthropic message formatting."""
    # Test None messages
    anthropic_llm = AnthropicCompletion(model="claude-3-sonnet", is_litellm=False)
    with pytest.raises(TypeError):
        anthropic_llm._format_messages_for_anthropic(None)

    # Test empty message list - Anthropic requires first message to be from user
    formatted, system_message = anthropic_llm._format_messages_for_anthropic([])
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Hello"

    # Test invalid message format
    with pytest.raises(ValueError, match="must have 'role' and 'content' keys"):
        anthropic_llm._format_messages_for_anthropic([{"invalid": "message"}])

def test_litellm_gpt5_does_not_send_stop_in_params():
    """
    Test that the LiteLLM fallback path does not include 'stop' in API params
    for GPT-5.x models, since they reject it at the API level.
    """
    llm = LLM(model="openai/gpt-5.2", stop=["Observation:"], is_litellm=True)

    params = llm._prepare_completion_params(
        messages=[{"role": "user", "content": "Hello"}]
    )

    assert params.get("stop") is None, (
        "GPT-5.x models should not have 'stop' in API params"
    )

def test_litellm_non_gpt5_sends_stop_in_params():
    """
    Test that the LiteLLM fallback path still includes 'stop' in API params
    for models that support it.
    """
    llm = LLM(model="gpt-4o", stop=["Observation:"], is_litellm=True)

    params = llm._prepare_completion_params(
        messages=[{"role": "user", "content": "Hello"}]
    )

    assert params.get("stop") == ["Observation:"], (
        "Non-GPT-5 models should have 'stop' in API params"
    )

