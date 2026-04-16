# cloudflare/langchain-cloudflare
# 18 LLM-backed test functions across 12 test files
# Source: https://github.com/cloudflare/langchain-cloudflare

# --- libs/langchain-cloudflare/tests/integration_tests/test_workersai_models.py ---

    def test_structured_output_invoke(self, model, account_id, api_token, ai_gateway):
        """Test structured output with invoke()."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        structured_llm = llm.with_structured_output(Data)

        result = structured_llm.invoke(
            f"Extract announcements from this text:\n\n{self.SAMPLE_TEXT}"
        )

        print(f"\n[{model}] Structured Output (invoke):")
        print(f"  Result type: {type(result)}")
        print(f"  Result: {result}")

        assert result is not None, f"Result is None for {model}"
        assert isinstance(result, (dict, Data)), (
            f"Unexpected type {type(result)} for {model}"
        )

        # Check structure
        if isinstance(result, dict):
            assert "announcements" in result, f"Missing 'announcements' key for {model}"
        else:
            assert hasattr(result, "announcements"), (
                f"Missing 'announcements' attr for {model}"
            )

    def test_tool_calling_invoke(self, model, account_id, api_token, ai_gateway):
        """Test tool calling with invoke()."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        llm_with_tools = llm.bind_tools([get_weather, get_stock_price])

        result = llm_with_tools.invoke("What's the weather in San Francisco?")

        print(f"\n[{model}] Tool Calling (invoke):")
        print(f"  Result type: {type(result)}")
        print(f"  Content: {result.content}")
        print(f"  Tool calls: {result.tool_calls}")

        # Model should either call the tool or respond with content
        assert result is not None, f"Result is None for {model}"

        # Check if tool was called
        if result.tool_calls:
            assert len(result.tool_calls) > 0, f"Empty tool_calls for {model}"
            tool_call = result.tool_calls[0]
            assert "name" in tool_call, f"Missing 'name' in tool_call for {model}"
            assert tool_call["name"] == "get_weather", f"Wrong tool called for {model}"
            assert "args" in tool_call, f"Missing 'args' in tool_call for {model}"
            print(f"  Tool call successful: {tool_call}")
        else:
            print(
                f"  No tool call made, content: {get_text_content(result.content)[:200] if result.content else 'empty'}"
            )

    def test_tool_calling_multi_turn(self, model, account_id, api_token, ai_gateway):
        """Test multi-turn tool calling conversation.

        This tests the full flow:
        1. User asks a question
        2. Model responds with a tool call
        3. We execute the tool and send the result back
        4. Model responds with final answer

        This exercises the is_llama_model logic in _create_message_dicts()
        which formats tool call history when sending back to the API.
        """
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        from langchain_core.messages import HumanMessage, ToolMessage

        llm = create_llm(model, account_id, api_token, ai_gateway)
        llm_with_tools = llm.bind_tools([get_weather, get_stock_price])

        # Step 1: Initial user message
        messages = [HumanMessage(content="What's the weather in San Francisco?")]

        # Step 2: Get model response (should be a tool call)
        response1 = llm_with_tools.invoke(messages)

        print(f"\n[{model}] Multi-turn Tool Calling:")
        print("  Step 1 - Initial response:")
        print(
            f"    Content: {get_text_content(response1.content)[:100] if response1.content else 'empty'}"
        )
        print(f"    Tool calls: {response1.tool_calls}")

        assert response1 is not None, f"Response 1 is None for {model}"

        if not response1.tool_calls:
            print("  WARN: No tool call made, skipping multi-turn test")
            return

        # Step 3: Execute the tool and add messages to history
        tool_call = response1.tool_calls[0]
        tool_result = get_weather.invoke(tool_call["args"])

        messages.append(response1)  # Add AI message with tool call
        messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        )

        print("  Step 2 - Tool executed:")
        print(f"    Tool: {tool_call['name']}")
        print(f"    Args: {tool_call['args']}")
        print(f"    Result: {tool_result}")

        # Step 4: Get final response from model
        response2 = llm_with_tools.invoke(messages)

        print("  Step 3 - Final response:")
        print(
            f"    Content: {get_text_content(response2.content)[:200] if response2.content else 'empty'}"
        )
        print(f"    Tool calls: {response2.tool_calls}")

        assert response2 is not None, f"Response 2 is None for {model}"
        # Final response should have content (not another tool call)
        assert response2.content, f"Final response has no content for {model}"
        print("  Status: PASS")

    def test_create_agent_structured_output_invoke(
        self, model, account_id, api_token, ai_gateway
    ):
        """Test create_agent with structured output using invoke()."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)

        agent = create_agent(
            model=llm,
            response_format=Data,
            system_prompt=self.SYSTEM_PROMPT,
            tools=[],
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Text: Acme Corp announced a partnership with TechGiant Inc.",
                    }
                ]
            }
        )

        print(f"\n[{model}] create_agent Structured Output (invoke):")
        print(f"  Result: {result}")

        assert result is not None, f"Result is None for {model}"

    def test_create_agent_tools_invoke(self, model, account_id, api_token, ai_gateway):
        """Test create_agent with tools using invoke()."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)

        agent = create_agent(
            model=llm,
            tools=[get_weather, get_stock_price],
        )

        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": "What's the weather in San Francisco?"}
                ]
            }
        )

        print(f"\n[{model}] create_agent Tools (invoke):")
        print(f"  Result: {result}")

        assert result is not None, f"Result is None for {model}"

    def test_tool_strategy_json_schema_invoke(
        self, model, account_id, api_token, ai_gateway
    ):
        """Test create_agent with ToolStrategy(json_schema_dict) via REST API."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)

        # Use JSON schema dict instead of Pydantic model
        json_schema = Data.model_json_schema()

        agent = create_agent(
            model=llm,
            response_format=ToolStrategy(json_schema),
            system_prompt=self.SYSTEM_PROMPT,
            tools=[],
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Text: Acme Corp announced a "
                            "partnership with TechGiant Inc."
                        ),
                    }
                ]
            }
        )

        print(f"\n[{model}] ToolStrategy JSON Schema (invoke):")
        print(f"  Result: {result}")

        assert result is not None, f"Result is None for {model}"
        # ToolStrategy with json_schema kind returns raw dict
        if isinstance(result, dict):
            structured = result.get("structured_response", result)
            assert structured is not None

    def test_basic_invoke(self, model, account_id, api_token, ai_gateway):
        """Test basic invoke returns content."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)

        result = llm.invoke("Say 'Hello World' and nothing else.")

        print(f"\n[{model}] Basic Invoke:")
        print(f"  Content: {result.content}")

        assert result is not None, f"Result is None for {model}"
        assert result.content, f"Empty content for {model}"
        text = get_text_content(result.content)
        assert "hello" in text.lower(), f"Unexpected response for {model}"

    def test_reasoning_content_sync(self, model, account_id, api_token, ai_gateway):
        """Test that reasoning_content appears as content blocks."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        result = llm.invoke("What is 25 * 37? Think step by step.")

        reasoning = self._extract_reasoning(result)

        print(f"\n[{model}] Reasoning Content (sync):")
        print(f"  Content: {str(result.content)[:200]}")
        print(f"  Reasoning: {reasoning[:200]}")
        print(f"  Content type: {type(result.content).__name__}")

        assert isinstance(result.content, list), (
            f"Expected list content blocks for {model}, got {type(result.content)}"
        )
        thinking_blocks = [
            b
            for b in result.content
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) > 0, (
            f"Expected thinking block in content for {model}"
        )
        assert len(reasoning) > 0, f"Expected non-empty reasoning_content for {model}"

    async def test_reasoning_content_async(
        self, model, account_id, api_token, ai_gateway
    ):
        """Test that reasoning_content appears as content blocks (async)."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        result = await llm.ainvoke("What is 25 * 37? Think step by step.")

        reasoning = self._extract_reasoning(result)

        print(f"\n[{model}] Reasoning Content (async):")
        print(f"  Content: {str(result.content)[:200]}")
        print(f"  Reasoning: {reasoning[:200]}")
        print(f"  Content type: {type(result.content).__name__}")

        assert isinstance(result.content, list), (
            f"Expected list content blocks for {model}, got {type(result.content)}"
        )
        thinking_blocks = [
            b
            for b in result.content
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) > 0, (
            f"Expected thinking block in content for {model}"
        )
        assert len(reasoning) > 0, f"Expected non-empty reasoning_content for {model}"

    def test_reasoning_content_with_tool_calls(
        self, model, account_id, api_token, ai_gateway
    ):
        """Test that reasoning_content is preserved when tool calls are also present."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        llm_with_tools = llm.bind_tools([get_weather])

        result = llm_with_tools.invoke("What's the weather in San Francisco?")

        print(f"\n[{model}] Reasoning + Tool Calls:")
        print(f"  Content type: {type(result.content).__name__}")
        print(f"  Content: {str(result.content)[:200]}")
        print(f"  Tool calls: {result.tool_calls}")

        assert result is not None, f"Result is None for {model}"

        # If the model made a tool call AND has reasoning, both should be present
        if result.tool_calls and isinstance(result.content, list):
            thinking_blocks = [
                b
                for b in result.content
                if isinstance(b, dict) and b.get("type") == "thinking"
            ]
            assert len(thinking_blocks) > 0, (
                f"Expected thinking block alongside tool_calls for {model}"
            )
            assert len(thinking_blocks[0]["thinking"]) > 0, (
                f"Expected non-empty reasoning for {model}"
            )
            print(f"  Reasoning: {thinking_blocks[0]['thinking'][:200]}")
            print("  Status: PASS - reasoning preserved with tool calls")
        elif result.tool_calls:
            # Tool call made but no reasoning - content should be empty string
            print("  Status: WARN - tool call without reasoning content")
        else:
            print("  Status: WARN - no tool call made")

    def test_no_reasoning_content_for_llama(self, account_id, api_token, ai_gateway):
        """Test that Llama content is a plain string, not content blocks."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            account_id,
            api_token,
            ai_gateway,
        )
        result = llm.invoke("Say hello.")

        print("\n[llama] Reasoning Content check:")
        print(f"  Content type: {type(result.content).__name__}")

        assert isinstance(result.content, str), (
            "Llama should return plain string content, not content blocks"
        )

    def test_image_base64(self, model, account_id, api_token, ai_gateway):
        """Test image input via base64-encoded PNG."""
        if not account_id or not api_token:
            pytest.skip("Missing CF_ACCOUNT_ID or CF_AI_API_TOKEN")

        llm = create_llm(model, account_id, api_token, ai_gateway)
        image_b64 = create_test_image_base64()

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Describe this image in one sentence. What color is it?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ]
        )

        try:
            result = llm.invoke([message])
            text = get_text_content(result.content)

            print(f"\n[{model}] Multi-Modal Image (base64):")
            print("  Status: PASS")
            print(f"  Response: {text[:200]}")

            assert len(text) > 0, f"Expected non-empty response from {model}"
        except Exception as e:
            error_msg = str(e)
            print(f"\n[{model}] Multi-Modal Image (base64):")
            print("  Status: FAIL")
            print(f"  Error: {error_msg[:200]}")

            # Skip rather than fail — this is a discovery test
            pytest.skip(
                f"Model {model} does not support multi-modal: {error_msg[:100]}"
            )

    def test_session_affinity_basic_invoke(self, model: str):
        """Verify that requests with session_id succeed and produce responses."""
        llm = ChatCloudflareWorkersAI(
            model=model,
            session_id="test-session-integration",
        )
        result = llm.invoke("Say hello in exactly 3 words.")
        text = get_text_content(result)
        assert text, f"Empty response from {model} with session_id"

    def test_session_affinity_cached_tokens(self, model: str):
        """Two calls with same session_id should succeed; cache hits are best-effort.

        Prompt caching depends on Cloudflare routing both requests to the same
        machine, which session affinity makes *likely* but not guaranteed.
        We verify the plumbing works and log cache metrics without asserting
        on cached_tokens, since it's infrastructure-dependent.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Long system prompt to ensure enough tokens to cache
        system_prompt = "You are an expert assistant. " * 50
        msgs = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Say hi in 3 words."),
        ]

        session = f"test-cache-{uuid.uuid4().hex[:8]}"
        llm = ChatCloudflareWorkersAI(model=model, session_id=session)

        # First call primes the cache
        r1 = llm.invoke(msgs)
        assert get_text_content(r1), "First call produced empty response"
        usage1 = r1.response_metadata.get("token_usage", {})
        cached1 = usage1.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print(f"  Call 1 cached_tokens: {cached1}")

        # Second call may hit the cache (best-effort)
        r2 = llm.invoke(msgs)
        assert get_text_content(r2), "Second call produced empty response"
        usage2 = r2.response_metadata.get("token_usage", {})
        cached2 = usage2.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print(f"  Call 2 cached_tokens: {cached2}")
        # Cache hits are best-effort; log but don't assert
        if cached2 > 0:
            print(f"  Cache HIT: {cached2} tokens cached")
        else:
            print("  Cache MISS: caching is best-effort, not guaranteed")

    async def test_session_affinity_cached_tokens_async(self, model: str):
        """Async variant: two calls with same session_id should succeed."""
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = "You are an expert assistant. " * 50
        msgs = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Say hi in 3 words."),
        ]

        session = f"test-cache-async-{uuid.uuid4().hex[:8]}"
        llm = ChatCloudflareWorkersAI(model=model, session_id=session)

        r1 = await llm.ainvoke(msgs)
        assert get_text_content(r1), "First async call produced empty response"

        r2 = await llm.ainvoke(msgs)
        assert get_text_content(r2), "Second async call produced empty response"
        usage2 = r2.response_metadata.get("token_usage", {})
        cached2 = usage2.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print(f"  Async call 2 cached_tokens: {cached2}")
        if cached2 > 0:
            print(f"  Async cache HIT: {cached2} tokens cached")
        else:
            print("  Async cache MISS: caching is best-effort, not guaranteed")

    def test_aig_timeout_invoke(self):
        """Request with AI Gateway timeout header should succeed."""
        llm = ChatCloudflareWorkersAI(
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            ai_gateway=os.environ["AI_GATEWAY"],
            aig_request_timeout=30000,
        )
        result = llm.invoke("Say hello in one word.")
        text = get_text_content(result)
        assert text, "Empty response with aig_request_timeout"

    def test_aig_retries_invoke(self):
        """Request with AI Gateway retry headers should succeed."""
        llm = ChatCloudflareWorkersAI(
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            ai_gateway=os.environ["AI_GATEWAY"],
            aig_max_attempts=2,
            aig_retry_delay=1000,
            aig_backoff="exponential",
        )
        result = llm.invoke("Say hello in one word.")
        text = get_text_content(result)
        assert text, "Empty response with aig retry headers"

    def test_aig_timeout_with_session_id(self):
        """AI Gateway headers and session_id should work together."""
        llm = ChatCloudflareWorkersAI(
            model="@cf/moonshotai/kimi-k2.5",
            ai_gateway=os.environ["AI_GATEWAY"],
            session_id="test-aig-session",
            aig_request_timeout=30000,
        )
        result = llm.invoke("Say hello.")
        text = get_text_content(result)
        assert text, "Empty response with combined headers"

