# lfnovo/esperanto
# 11 LLM-backed test functions across 55 test files
# Source: https://github.com/lfnovo/esperanto

# --- tests/integration/test_langchain_integration.py ---

def test_openai_langchain_conversion(openai_model):
    langchain_model = openai_model.to_langchain()
    assert isinstance(langchain_model, ChatOpenAI)
    assert langchain_model.model_name == "gpt-3.5-turbo"
    assert langchain_model.temperature == 0.7
    assert langchain_model.max_tokens == 100
    assert langchain_model.streaming is True
    assert langchain_model.top_p == 0.9

def test_openrouter_langchain_conversion(openrouter_model):
    langchain_model = openrouter_model.to_langchain()
    assert isinstance(langchain_model, ChatOpenAI)
    assert langchain_model.model_name == "gpt-3.5-turbo"
    assert langchain_model.temperature == 0.7
    assert langchain_model.max_tokens == 100
    assert langchain_model.streaming is True
    assert langchain_model.top_p == 0.9
    assert langchain_model.openai_api_base == "https://openrouter.ai/api/v1"

def test_xai_langchain_conversion(xai_model):
    langchain_model = xai_model.to_langchain()
    assert isinstance(langchain_model, ChatOpenAI)
    assert langchain_model.model_name == "gpt-3.5-turbo"
    assert langchain_model.temperature == 0.7
    assert langchain_model.max_tokens == 100
    assert langchain_model.streaming is True
    assert langchain_model.top_p == 0.9
    assert langchain_model.openai_api_base == "https://api.x.ai/v1"


# --- tests/providers/llm/test_perplexity_provider.py ---

def test_perplexity_get_api_kwargs_exclude_stream(perplexity_provider):
    """Test _get_api_kwargs excludes stream when requested."""
    perplexity_provider.streaming = True
    kwargs = perplexity_provider._get_api_kwargs(exclude_stream=True)
    assert "stream" not in kwargs

async def test_perplexity_async_call(perplexity_provider):
    """Test the asynchronous call method."""
    # Pass messages as dicts, not LangChain objects
    messages = [{"role": "user", "content": "Hello"}]
    expected_response_text = "Hello!"

    response = await perplexity_provider.achat_complete(messages)

    assert response.choices[0].message.content == expected_response_text
    assert response.model == perplexity_provider.get_model_name()

def test_perplexity_call(perplexity_provider):
    """Test the synchronous call method."""
    # Pass messages as dicts, not LangChain objects
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    expected_response_text = "Hello!"

    response = perplexity_provider.chat_complete(messages)

    assert response.choices[0].message.content == expected_response_text
    assert response.model == perplexity_provider.get_model_name()

def test_perplexity_call_with_extra_params(perplexity_provider):
    """Test synchronous call with extra Perplexity parameters."""
    perplexity_provider.search_domain_filter = ["test.com"]
    perplexity_provider.return_images = True
    messages = [{"role": "user", "content": "Hi"}]
    expected_response_text = "Hello!"

    response = perplexity_provider.chat_complete(messages)

    assert response.choices[0].message.content == expected_response_text
    assert response.model == perplexity_provider.get_model_name()
    
    # Test that perplexity params are available
    params = perplexity_provider._get_perplexity_params()
    assert params["search_domain_filter"] == ["test.com"]
    assert params["return_images"] is True

    def test_chat_complete_with_tools(self, perplexity_model_with_tool_response, sample_tools):
        """Test chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = perplexity_model_with_tool_response.chat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = perplexity_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload
        assert len(json_payload["tools"]) == 2

        # Check response has tool calls
        assert len(response.choices) == 1
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1

        tool_call = response.choices[0].message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id == "call_abc123"
        assert tool_call.function.name == "get_weather"
        assert '"location": "San Francisco"' in tool_call.function.arguments

    def test_chat_complete_with_tool_choice(self, perplexity_model_with_tool_response, sample_tools):
        """Test chat_complete with tool_choice parameter."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        perplexity_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice="required"
        )

        call_args = perplexity_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["tool_choice"] == "required"

    async def test_achat_complete_with_tools(self, perplexity_model_with_tool_response, sample_tools):
        """Test async chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = await perplexity_model_with_tool_response.achat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = perplexity_model_with_tool_response.async_client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload

        # Check response has tool calls
        assert response.choices[0].message.tool_calls is not None
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"

    def test_validation_passes_for_valid_tool_call(
        self, perplexity_model_with_tool_response, sample_tools
    ):
        """Test that validation passes for valid tool calls."""
        pytest.importorskip("jsonschema")

        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        # Should not raise
        response = perplexity_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, validate_tool_calls=True
        )

        assert response.choices[0].message.tool_calls is not None

