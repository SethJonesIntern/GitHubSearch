# SolaceLabs/solace-agent-mesh
# 8 test functions with real LLM calls
# Source: https://github.com/SolaceLabs/solace-agent-mesh


# --- tests/unit/agent/adk/models/test_lite_llm_status.py ---

    async def test_rejects_when_initializing(self):
        """generate_content_async should raise BadRequestError when status is 'initializing'."""
        from litellm.exceptions import BadRequestError
        from google.adk.models.llm_request import LlmRequest
        from google.genai.types import Content, Part

        llm = LiteLlm(model=None)
        assert llm.status == "initializing"

        content = Content(role="user", parts=[Part(text="Hello")])
        request = LlmRequest(contents=[content])

        with pytest.raises(BadRequestError, match="not been configured"):
            async for _ in llm.generate_content_async(request):
                pass

    async def test_rejects_when_unconfigured(self):
        """generate_content_async should raise BadRequestError when status is 'none'."""
        from litellm.exceptions import BadRequestError
        from google.adk.models.llm_request import LlmRequest
        from google.genai.types import Content, Part

        llm = LiteLlm(model="test-model")
        llm.unconfigure_model()
        assert llm.status == "none"

        content = Content(role="user", parts=[Part(text="Hello")])
        request = LlmRequest(contents=[content])

        with pytest.raises(BadRequestError, match="not been configured"):
            async for _ in llm.generate_content_async(request):
                pass


# --- tests/unit/agent/adk/models/test_lite_llm_thinking.py ---

    def test_extracts_reasoning_content_to_custom_metadata(self):
        """reasoning_content is placed into custom_metadata['thinking_content']."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "final answer",
                        "reasoning_content": "step by step reasoning",
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        llm_response = _model_response_to_generate_content_response(response)
        assert llm_response.custom_metadata is not None
        assert llm_response.custom_metadata["thinking_content"] == "step by step reasoning"

    def test_extracts_reasoning_from_provider_specific_fields(self):
        """reasoning_content from provider_specific_fields goes to custom_metadata."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "answer",
                        "provider_specific_fields": {
                            "reasoning_content": "provider reasoning"
                        },
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        llm_response = _model_response_to_generate_content_response(response)
        assert llm_response.custom_metadata is not None
        assert llm_response.custom_metadata["thinking_content"] == "provider reasoning"

    def test_no_custom_metadata_when_no_reasoning(self):
        """No thinking_content in custom_metadata when reasoning is absent."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "plain response",
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        llm_response = _model_response_to_generate_content_response(response)
        if llm_response.custom_metadata:
            assert "thinking_content" not in llm_response.custom_metadata


# --- tests/unit/agent/adk/test_auto_summarization_runner.py ---

    def test_always_leaves_at_least_one_user_turn_uncompacted(self):
        """Last user turn should never be compacted, even with high token targets.

        Real-world scenario: Long conversation with increasing token usage.
        - Turn 1: Detailed question and comprehensive response
        - Turn 2: Follow-up questions (current)

        Should never compact Turn 2 (the current turn), even if token budget is high.
        This ensures the most recent context is always available.
        """
        events = [
            # Turn 1: Detailed initial question
            ADKEvent(
                invocation_id="u1",
                author="user",
                content=adk_types.Content(
                    role="user",
                    parts=[adk_types.Part(
                        text="I need comprehensive help with system architecture design. "
                             "We're building a microservices platform for e-commerce with expected "
                             "traffic of 100k requests per day. We need API gateway, authentication, "
                             "payment processing, inventory management, and analytics. What's your recommendation? "
                             "Consider scalability, cost, and maintainability. Also discuss database choices."
                    )]
                )
            ),
            # Turn 1: Comprehensive response (lots of tokens)
            ADKEvent(
                invocation_id="m1",
                author="model",
                content=adk_types.Content(
                    role="model",
                    parts=[adk_types.Part(
                        text="I'd recommend a comprehensive architecture: Use Kong or AWS API Gateway for routing. "
                             "For authentication, implement OAuth 2.0 with JWT tokens. Payment processing: integrate with Stripe or Adyen. "
                             "Inventory: separate microservice with Redis caching for high-frequency queries. "
                             "Analytics: use event streaming (Kafka) with aggregation in BigQuery or Elastic. "
                             "Database: PostgreSQL for transactional data, with read replicas for analytics queries. "
                             "Use CDN for static assets. Implement circuit breakers and retry logic. "
                             "Container orchestration with Kubernetes. This scales to millions of requests. "
                             "Cost optimization: use spot instances, reserved capacity, auto-scaling."
                    )]
                )
            ),
            # Turn 2: Follow-up questions (CURRENT, should not be compacted)
            ADKEvent(
                invocation_id="u2",
                author="user",
                content=adk_types.Content(
                    role="user",
                    parts=[adk_types.Part(
                        text="That's very helpful. Now I need clarification on a few points. "
                             "How do we handle distributed transactions across microservices? "
                             "What about data consistency? And how do we monitor this complex system?"
                    )]
                )
            ),
        ]

        # Even with very high token target (simulating context limit pressure),
        # should stop before last turn (Turn 2)
        cutoff_idx, actual_tokens = _find_compaction_cutoff(events, 5000)

        # Should be at index 2 (boundary before u2, the current turn)
        assert cutoff_idx == 2, \
            f"Should stop at turn boundary before last turn (u2), got {cutoff_idx}"
        # Should have compacted u1 + m1
        assert actual_tokens > 0, "Should have calculated token count for Turn 1"


# --- tests/unit/common/test_error_handlers.py ---

    def test_rejects_runtime_error(self):
        assert is_llm_exception(RuntimeError("runtime")) is False

    def test_unknown_exception_returns_default(self):
        assert _get_user_friendly_error_message(RuntimeError("oops")) == DEFAULT_LLM_ERROR_MESSAGE

