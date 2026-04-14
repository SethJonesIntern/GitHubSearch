# langchain-ai/langchain
# 357 test functions with real LLM calls
# Source: https://github.com/langchain-ai/langchain


# --- libs/langchain/tests/integration_tests/chains/openai_functions/test_openapi.py ---

def test_openai_openapi_chain() -> None:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = get_openapi_chain(json.dumps(api_spec), llm)
    output = chain.invoke({"query": "Fetch the top two posts."})
    assert len(output["response"]) == 2


# --- libs/langchain_v1/tests/integration_tests/agents/middleware/test_shell_tool_integration.py ---

def test_shell_tool_basic_execution(tmp_path: Path, provider: str) -> None:
    """Test basic shell command execution across different models."""
    workspace = tmp_path / "workspace"
    agent: CompiledStateGraph[Any, Any, _InputAgentState, Any] = create_agent(
        model=_get_model(provider),
        middleware=[ShellToolMiddleware(workspace_root=workspace)],
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Run the command 'echo hello' and tell me what it outputs")]}
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) > 0, "Shell tool should have been called"

    tool_outputs = [msg.content for msg in tool_messages]
    assert any("hello" in output.lower() for output in tool_outputs), (
        "Shell output should contain 'hello'"
    )

def test_shell_session_persistence(tmp_path: Path) -> None:
    """Test shell session state persists across multiple tool calls."""
    workspace = tmp_path / "workspace"
    agent: CompiledStateGraph[Any, Any, _InputAgentState, Any] = create_agent(
        model=_get_model("anthropic"),
        middleware=[ShellToolMiddleware(workspace_root=workspace)],
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "First run 'export TEST_VAR=hello'. "
                    "Then run 'echo $TEST_VAR' to verify it persists."
                )
            ]
        }
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) >= 2, "Shell tool should be called multiple times"

    tool_outputs = [msg.content for msg in tool_messages]
    assert any("hello" in output for output in tool_outputs), "Environment variable should persist"

def test_shell_tool_error_handling(tmp_path: Path) -> None:
    """Test shell tool captures command errors."""
    workspace = tmp_path / "workspace"
    agent: CompiledStateGraph[Any, Any, _InputAgentState, Any] = create_agent(
        model=_get_model("anthropic"),
        middleware=[ShellToolMiddleware(workspace_root=workspace)],
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "Run the command 'ls /nonexistent_directory_12345' and show me the result"
                )
            ]
        }
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) > 0, "Shell tool should have been called"

    tool_outputs = " ".join(msg.content for msg in tool_messages)
    assert (
        "no such file" in tool_outputs.lower()
        or "cannot access" in tool_outputs.lower()
        or "not found" in tool_outputs.lower()
        or "exit code" in tool_outputs.lower()
    ), "Error should be captured in tool output"

def test_shell_tool_with_custom_tools(tmp_path: Path) -> None:
    """Test shell tool works alongside custom tools."""
    workspace = tmp_path / "workspace"

    @tool
    def custom_greeting(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    agent: CompiledStateGraph[Any, Any, _InputAgentState, Any] = create_agent(
        model=_get_model("anthropic"),
        tools=[custom_greeting],
        middleware=[ShellToolMiddleware(workspace_root=workspace)],
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "First, use the custom_greeting tool to greet 'Alice'. "
                    "Then run the shell command 'echo world'."
                )
            ]
        }
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) >= 2, "Both tools should have been called"

    tool_outputs = " ".join(msg.content for msg in tool_messages)
    assert "Alice" in tool_outputs, "Custom tool should be used"
    assert "world" in tool_outputs, "Shell tool should be used"


# --- libs/langchain_v1/tests/integration_tests/chat_models/test_base.py ---

async def test_init_chat_model_chain() -> None:
    model = init_chat_model("gpt-4o", configurable_fields="any", config_prefix="bar")
    model_with_tools = model.bind_tools([Multiply])

    model_with_config = model_with_tools.with_config(
        RunnableConfig(tags=["foo"]),
        configurable={"bar_model": "claude-sonnet-4-5-20250929"},
    )
    prompt = ChatPromptTemplate.from_messages([("system", "foo"), ("human", "{input}")])
    chain = prompt | model_with_config
    output = chain.invoke({"input": "bar"})
    assert isinstance(output, AIMessage)
    events = [event async for event in chain.astream_events({"input": "bar"}, version="v2")]
    assert events


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_decorators.py ---

def test_before_model_decorator() -> None:
    """Test before_model decorator with all configuration options."""

    @before_model(
        state_schema=CustomState, tools=[test_tool], can_jump_to=["end"], name="CustomBeforeModel"
    )
    def custom_before_model(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"jump_to": "end"}

    assert isinstance(custom_before_model, AgentMiddleware)
    assert custom_before_model.state_schema == CustomState
    assert custom_before_model.tools == [test_tool]
    assert getattr(custom_before_model.__class__.before_model, "__can_jump_to__", []) == ["end"]
    assert custom_before_model.__class__.__name__ == "CustomBeforeModel"

    result = custom_before_model.before_model({"messages": [HumanMessage("Hello")]}, Runtime())
    assert result == {"jump_to": "end"}

def test_after_model_decorator() -> None:
    """Test after_model decorator with all configuration options."""

    @after_model(
        state_schema=CustomState,
        tools=[test_tool],
        can_jump_to=["model", "end"],
        name="CustomAfterModel",
    )
    def custom_after_model(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"jump_to": "model"}

    # Verify all options were applied
    assert isinstance(custom_after_model, AgentMiddleware)
    assert custom_after_model.state_schema == CustomState
    assert custom_after_model.tools == [test_tool]
    assert getattr(custom_after_model.__class__.after_model, "__can_jump_to__", []) == [
        "model",
        "end",
    ]
    assert custom_after_model.__class__.__name__ == "CustomAfterModel"

    # Verify it works
    result = custom_after_model.after_model(
        {"messages": [HumanMessage("Hello"), AIMessage("Hi!")]}, Runtime()
    )
    assert result == {"jump_to": "model"}


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_framework.py ---

def test_injected_state_in_middleware_agent() -> None:
    """Test that custom state is properly injected into tools when using middleware."""
    result = agent.invoke(
        {
            "custom_state": "I love pizza",
            "messages": [HumanMessage("Call the test state tool")],
        }
    )

    messages = result["messages"]
    assert len(messages) == 4  # Human message, AI message with tool call, tool message, AI message

    # Find the tool message
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 1

    tool_message = tool_messages[0]
    assert tool_message.name == "test_state_tool"
    assert "success" in tool_message.content
    assert tool_message.tool_call_id == "test_call_1"


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_wrap_model_call.py ---

    def test_no_retry_propagates_error(self) -> None:
        """Test that error is propagated when middleware doesn't retry."""

        class FailingModel(BaseChatModel):
            """Model that always fails."""

            @override
            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                msg = "Model error"
                raise ValueError(msg)

            @property
            def _llm_type(self) -> str:
                return "failing"

        class NoRetryMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> ModelCallResult:
                return handler(request)

        agent = create_agent(model=FailingModel(), middleware=[NoRetryMiddleware()])

        with pytest.raises(ValueError, match="Model error"):
            agent.invoke({"messages": [HumanMessage("Test")]})

    def test_max_attempts_limit(self) -> None:
        """Test that middleware controls termination via retry limits."""

        class AlwaysFailingModel(BaseChatModel):
            """Model that always fails."""

            @override
            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                msg = "Always fails"
                raise ValueError(msg)

            @property
            def _llm_type(self) -> str:
                return "always_failing"

        class LimitedRetryMiddleware(AgentMiddleware):
            """Middleware that limits its own retries."""

            def __init__(self, max_retries: int = 10):
                super().__init__()
                self.max_retries = max_retries
                self.attempt_count = 0

            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> ModelCallResult:
                last_exception = None
                for _attempt in range(self.max_retries):
                    self.attempt_count += 1
                    try:
                        return handler(request)
                    except Exception as e:
                        last_exception = e
                        # Continue to retry

                # All retries exhausted, re-raise the last error
                if last_exception:
                    raise last_exception
                pytest.fail("Should have raised an exception")

        model = AlwaysFailingModel()
        middleware = LimitedRetryMiddleware(max_retries=10)

        agent = create_agent(model=model, middleware=[middleware])

        # Should fail with the model's error after middleware stops retrying
        with pytest.raises(ValueError, match="Always fails"):
            agent.invoke({"messages": [HumanMessage("Test")]})

        # Should have attempted exactly 10 times as configured
        assert middleware.attempt_count == 10


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_human_in_the_loop.py ---

def test_human_in_the_loop_middleware_no_interrupts_needed() -> None:
    """Test HumanInTheLoopMiddleware when no interrupts are needed."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
    )

    # Test with no messages
    state = AgentState[Any](messages=[])
    result = middleware.after_model(state, Runtime())
    assert result is None

    # Test with message but no tool calls
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), AIMessage(content="Hi there")])

    result = middleware.after_model(state, Runtime())
    assert result is None

    # Test with tool calls that don't require interrupts
    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "other_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])
    result = middleware.after_model(state, Runtime())
    assert result is None


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_call_limit.py ---

def test_middleware_unit_functionality() -> None:
    """Test that the middleware works as expected in isolation."""
    # Test with end behavior
    middleware = ModelCallLimitMiddleware(thread_limit=2, run_limit=1)

    runtime = Runtime()

    # Test when limits are not exceeded
    state = ModelCallLimitState(messages=[], thread_model_call_count=0, run_model_call_count=0)
    result = middleware.before_model(state, runtime)
    assert result is None

    # Test when thread limit is exceeded
    state = ModelCallLimitState(messages=[], thread_model_call_count=2, run_model_call_count=0)
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "thread limit (2/2)" in result["messages"][0].content

    # Test when run limit is exceeded
    state = ModelCallLimitState(messages=[], thread_model_call_count=1, run_model_call_count=1)
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "run limit (1/1)" in result["messages"][0].content

    # Test with error behavior
    middleware_exception = ModelCallLimitMiddleware(
        thread_limit=2, run_limit=1, exit_behavior="error"
    )

    # Test exception when thread limit exceeded
    state = ModelCallLimitState(messages=[], thread_model_call_count=2, run_model_call_count=0)
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware_exception.before_model(state, runtime)

    assert "thread limit (2/2)" in str(exc_info.value)

    # Test exception when run limit exceeded
    state = ModelCallLimitState(messages=[], thread_model_call_count=1, run_model_call_count=1)
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware_exception.before_model(state, runtime)

    assert "run limit (1/1)" in str(exc_info.value)

def test_middleware_initialization_validation() -> None:
    """Test that middleware initialization validates parameters correctly."""
    # Test that at least one limit must be specified
    with pytest.raises(ValueError, match="At least one limit must be specified"):
        ModelCallLimitMiddleware()

    # Test invalid exit behavior
    with pytest.raises(ValueError, match="Invalid exit_behavior"):
        ModelCallLimitMiddleware(thread_limit=5, exit_behavior="invalid")  # type: ignore[arg-type]

    # Test valid initialization
    middleware = ModelCallLimitMiddleware(thread_limit=5, run_limit=3)
    assert middleware.thread_limit == 5
    assert middleware.run_limit == 3
    assert middleware.exit_behavior == "end"

    # Test with only thread limit
    middleware = ModelCallLimitMiddleware(thread_limit=5)
    assert middleware.thread_limit == 5
    assert middleware.run_limit is None

    # Test with only run limit
    middleware = ModelCallLimitMiddleware(run_limit=3)
    assert middleware.thread_limit is None
    assert middleware.run_limit == 3

def test_exception_error_message() -> None:
    """Test that the exception provides clear error messages."""
    middleware = ModelCallLimitMiddleware(thread_limit=2, run_limit=1, exit_behavior="error")

    # Test thread limit exceeded
    state = ModelCallLimitState(messages=[], thread_model_call_count=2, run_model_call_count=0)
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware.before_model(state, Runtime())

    error_msg = str(exc_info.value)
    assert "Model call limits exceeded" in error_msg
    assert "thread limit (2/2)" in error_msg

    # Test run limit exceeded
    state = ModelCallLimitState(messages=[], thread_model_call_count=0, run_model_call_count=1)
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware.before_model(state, Runtime())

    error_msg = str(exc_info.value)
    assert "Model call limits exceeded" in error_msg
    assert "run limit (1/1)" in error_msg

    # Test both limits exceeded
    state = ModelCallLimitState(messages=[], thread_model_call_count=2, run_model_call_count=1)
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware.before_model(state, Runtime())

    error_msg = str(exc_info.value)
    assert "Model call limits exceeded" in error_msg
    assert "thread limit (2/2)" in error_msg
    assert "run limit (1/1)" in error_msg


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_fallback.py ---

def test_model_fallback_middleware_with_agent() -> None:
    """Test ModelFallbackMiddleware with agent.invoke and fallback models only."""

    class FailingModel(BaseChatModel):
        """Model that always fails."""

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "Primary model failed"
            raise ValueError(msg)

        @property
        def _llm_type(self) -> str:
            return "failing"

    class SuccessModel(BaseChatModel):
        """Model that succeeds."""

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Fallback success"))]
            )

        @property
        def _llm_type(self) -> str:
            return "success"

    primary = FailingModel()
    fallback = SuccessModel()

    # Only pass fallback models to middleware (not the primary)
    fallback_middleware = ModelFallbackMiddleware(fallback)

    agent = create_agent(model=primary, middleware=[fallback_middleware])

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Should have succeeded with fallback model
    assert len(result["messages"]) == 2
    assert result["messages"][1].content == "Fallback success"

def test_model_fallback_middleware_exhausted_with_agent() -> None:
    """Test ModelFallbackMiddleware with agent.invoke when all models fail."""

    class AlwaysFailingModel(BaseChatModel):
        """Model that always fails."""

        def __init__(self, name: str):
            super().__init__()
            self.name = name

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = f"{self.name} failed"
            raise ValueError(msg)

        @property
        def _llm_type(self) -> str:
            return self.name or "always_failing"

    primary = AlwaysFailingModel("primary")
    fallback1 = AlwaysFailingModel("fallback1")
    fallback2 = AlwaysFailingModel("fallback2")

    # Primary fails (attempt 1), then fallback1 (attempt 2), then fallback2 (attempt 3)
    fallback_middleware = ModelFallbackMiddleware(fallback1, fallback2)

    agent = create_agent(model=primary, middleware=[fallback_middleware])

    # Should fail with the last fallback's error
    with pytest.raises(ValueError, match="fallback2 failed"):
        agent.invoke({"messages": [HumanMessage("Test")]})


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_retry.py ---

def test_model_retry_initialization_custom() -> None:
    """Test ModelRetryMiddleware initialization with custom values."""
    retry = ModelRetryMiddleware(
        max_retries=5,
        retry_on=(ValueError, RuntimeError),
        on_failure="error",
        backoff_factor=1.5,
        initial_delay=0.5,
        max_delay=30.0,
        jitter=False,
    )

    assert retry.max_retries == 5
    assert retry.tools == []
    assert retry.retry_on == (ValueError, RuntimeError)
    assert retry.on_failure == "error"
    assert retry.backoff_factor == 1.5
    assert retry.initial_delay == 0.5
    assert retry.max_delay == 30.0
    assert retry.jitter is False

def test_model_retry_failing_model_returns_message() -> None:
    """Test ModelRetryMiddleware with failing model returns error message."""
    model = AlwaysFailingModel(error_message="Model error", error_type=ValueError)

    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    # Should contain error message with attempts
    last_msg = ai_messages[-1].content
    assert "failed after 3 attempts" in last_msg
    assert "ValueError" in last_msg

def test_model_retry_failing_model_raises() -> None:
    """Test ModelRetryMiddleware with on_failure='error' re-raises exception."""
    model = AlwaysFailingModel(error_message="Model error", error_type=ValueError)

    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        on_failure="error",
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    # Should raise the ValueError from the model
    with pytest.raises(ValueError, match="Model error"):
        agent.invoke(
            {"messages": [HumanMessage("Hello")]},
            {"configurable": {"thread_id": "test"}},
        )

def test_model_retry_custom_failure_formatter() -> None:
    """Test ModelRetryMiddleware with custom failure message formatter."""

    def custom_formatter(exc: Exception) -> str:
        return f"Custom error: {type(exc).__name__}"

    model = AlwaysFailingModel(error_message="Model error", error_type=ValueError)

    retry = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=0.01,
        jitter=False,
        on_failure=custom_formatter,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    assert "Custom error: ValueError" in ai_messages[-1].content

def test_model_retry_succeeds_after_retries() -> None:
    """Test ModelRetryMiddleware succeeds after temporary failures."""
    model = TemporaryFailureModel(fail_count=2)

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.01,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    # Should succeed on 3rd attempt
    assert "Success after 3 attempts" in ai_messages[-1].content
    assert model.attempt == 3

def test_model_retry_specific_exceptions() -> None:
    """Test ModelRetryMiddleware only retries specific exception types."""
    # This model will fail with RuntimeError, which we won't retry
    model = AlwaysFailingModel(error_message="Runtime error", error_type=RuntimeError)

    # Only retry ValueError
    retry = ModelRetryMiddleware(
        max_retries=2,
        retry_on=(ValueError,),
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    # RuntimeError should fail immediately (1 attempt only)
    assert "1 attempt" in ai_messages[-1].content

def test_model_retry_backoff_timing() -> None:
    """Test ModelRetryMiddleware applies correct backoff delays."""
    model = TemporaryFailureModel(fail_count=3)

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.1,
        backoff_factor=2.0,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    start_time = time.time()
    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )
    elapsed = time.time() - start_time

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1

    # Expected delays: 0.1 + 0.2 + 0.4 = 0.7 seconds
    # Allow some margin for execution time
    assert elapsed >= 0.6, f"Expected at least 0.6s, got {elapsed}s"

def test_model_retry_constant_backoff() -> None:
    """Test ModelRetryMiddleware with constant backoff (backoff_factor=0)."""
    model = TemporaryFailureModel(fail_count=2)

    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0.1,
        backoff_factor=0.0,  # Constant backoff
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    start_time = time.time()
    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )
    elapsed = time.time() - start_time

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1

    # Expected delays: 0.1 + 0.1 = 0.2 seconds (constant)
    assert elapsed >= 0.15, f"Expected at least 0.15s, got {elapsed}s"
    assert elapsed < 0.5, f"Expected less than 0.5s (exponential would be longer), got {elapsed}s"

async def test_model_retry_async_failing_model() -> None:
    """Test ModelRetryMiddleware with async execution and failing model."""
    model = AlwaysFailingModel(error_message="Model error", error_type=ValueError)

    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    last_msg = ai_messages[-1].content
    assert "failed after 3 attempts" in last_msg
    assert "ValueError" in last_msg

async def test_model_retry_async_succeeds_after_retries() -> None:
    """Test ModelRetryMiddleware async execution succeeds after temporary failures."""
    model = TemporaryFailureModel(fail_count=2)

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.01,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    assert "Success after 3 attempts" in ai_messages[-1].content

async def test_model_retry_async_backoff_timing() -> None:
    """Test ModelRetryMiddleware async applies correct backoff delays."""
    model = TemporaryFailureModel(fail_count=3)

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.1,
        backoff_factor=2.0,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    start_time = time.time()
    result = await agent.ainvoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )
    elapsed = time.time() - start_time

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1

    # Expected delays: 0.1 + 0.2 + 0.4 = 0.7 seconds
    assert elapsed >= 0.6, f"Expected at least 0.6s, got {elapsed}s"

def test_model_retry_zero_retries() -> None:
    """Test ModelRetryMiddleware with max_retries=0 (no retries)."""
    model = AlwaysFailingModel(error_message="Model error", error_type=ValueError)

    retry = ModelRetryMiddleware(
        max_retries=0,  # No retries
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    # Should fail after 1 attempt (no retries)
    assert "1 attempt" in ai_messages[-1].content


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_pii.py ---

    def test_redact_email(self) -> None:
        middleware = PIIMiddleware("email", strategy="redact")
        state = AgentState[Any](messages=[HumanMessage("Email me at test@example.com")])

        result = middleware.before_model(state, Runtime())

        assert result is not None
        assert "[REDACTED_EMAIL]" in result["messages"][0].content
        assert "test@example.com" not in result["messages"][0].content

    def test_redact_multiple_pii(self) -> None:
        middleware = PIIMiddleware("email", strategy="redact")
        state = AgentState[Any](messages=[HumanMessage("Contact alice@test.com or bob@test.com")])

        result = middleware.before_model(state, Runtime())

        assert result is not None
        content = result["messages"][0].content
        assert content.count("[REDACTED_EMAIL]") == 2
        assert "alice@test.com" not in content
        assert "bob@test.com" not in content

    def test_mask_email(self) -> None:
        middleware = PIIMiddleware("email", strategy="mask")
        state = AgentState[Any](messages=[HumanMessage("Email: user@example.com")])

        result = middleware.before_model(state, Runtime())

        assert result is not None
        content = result["messages"][0].content
        assert "user@****.com" in content
        assert "user@example.com" not in content

    def test_mask_credit_card(self) -> None:
        middleware = PIIMiddleware("credit_card", strategy="mask")
        # Valid test card
        state = AgentState[Any](messages=[HumanMessage("Card: 4532015112830366")])

        result = middleware.before_model(state, Runtime())

        assert result is not None
        content = result["messages"][0].content
        assert "0366" in content  # Last 4 digits visible
        assert "4532015112830366" not in content

    def test_mask_ip(self) -> None:
        middleware = PIIMiddleware("ip", strategy="mask")
        state = AgentState[Any](messages=[HumanMessage("IP: 192.168.1.100")])

        result = middleware.before_model(state, Runtime())

        assert result is not None
        content = result["messages"][0].content
        assert "*.*.*.100" in content
        assert "192.168.1.100" not in content

    def test_hash_email(self) -> None:
        middleware = PIIMiddleware("email", strategy="hash")
        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])

        result = middleware.before_model(state, Runtime())

        assert result is not None
        content = result["messages"][0].content
        assert "<email_hash:" in content
        assert ">" in content
        assert "test@example.com" not in content

    def test_hash_is_deterministic(self) -> None:
        middleware = PIIMiddleware("email", strategy="hash")

        # Same email should produce same hash
        state1 = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])
        state2 = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])

        result1 = middleware.before_model(state1, Runtime())
        result2 = middleware.before_model(state2, Runtime())

        assert result1 is not None
        assert result2 is not None
        assert result1["messages"][0].content == result2["messages"][0].content

    def test_block_raises_exception(self) -> None:
        middleware = PIIMiddleware("email", strategy="block")
        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])

        with pytest.raises(PIIDetectionError) as exc_info:
            middleware.before_model(state, Runtime())

        assert exc_info.value.pii_type == "email"
        assert len(exc_info.value.matches) == 1
        assert "test@example.com" in exc_info.value.matches[0]["value"]

    def test_block_with_multiple_matches(self) -> None:
        middleware = PIIMiddleware("email", strategy="block")
        state = AgentState[Any](messages=[HumanMessage("Emails: alice@test.com and bob@test.com")])

        with pytest.raises(PIIDetectionError) as exc_info:
            middleware.before_model(state, Runtime())

        assert len(exc_info.value.matches) == 2

    def test_apply_to_input_only(self) -> None:
        """Test that middleware only processes input when configured."""
        middleware = PIIMiddleware(
            "email", strategy="redact", apply_to_input=True, apply_to_output=False
        )

        # Should process HumanMessage
        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])
        result = middleware.before_model(state, Runtime())
        assert result is not None
        assert "[REDACTED_EMAIL]" in result["messages"][0].content

        # Should not process AIMessage
        state = AgentState[Any](messages=[AIMessage("My email is ai@example.com")])
        result = middleware.after_model(state, Runtime())
        assert result is None

    def test_apply_to_output_only(self) -> None:
        """Test that middleware only processes output when configured."""
        middleware = PIIMiddleware(
            "email", strategy="redact", apply_to_input=False, apply_to_output=True
        )

        # Should not process HumanMessage
        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])
        result = middleware.before_model(state, Runtime())
        assert result is None

        # Should process AIMessage
        state = AgentState[Any](messages=[AIMessage("My email is ai@example.com")])
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "[REDACTED_EMAIL]" in result["messages"][0].content

    def test_apply_to_both(self) -> None:
        """Test that middleware processes both input and output."""
        middleware = PIIMiddleware(
            "email", strategy="redact", apply_to_input=True, apply_to_output=True
        )

        # Should process HumanMessage
        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])
        result = middleware.before_model(state, Runtime())
        assert result is not None

        # Should process AIMessage
        state = AgentState[Any](messages=[AIMessage("My email is ai@example.com")])
        result = middleware.after_model(state, Runtime())
        assert result is not None

    def test_no_pii_returns_none(self) -> None:
        """Test that middleware returns None when no PII detected."""
        middleware = PIIMiddleware("email", strategy="redact")
        state = AgentState[Any](messages=[HumanMessage("No PII here")])

        result = middleware.before_model(state, Runtime())
        assert result is None

    def test_empty_messages(self) -> None:
        """Test that middleware handles empty messages gracefully."""
        middleware = PIIMiddleware("email", strategy="redact")
        state = AgentState[Any](messages=[])

        result = middleware.before_model(state, Runtime())
        assert result is None

    def test_apply_to_tool_results(self) -> None:
        """Test that middleware processes tool results when enabled."""
        middleware = PIIMiddleware(
            "email", strategy="redact", apply_to_input=False, apply_to_tool_results=True
        )

        # Simulate a conversation with tool call and result containing PII
        state = AgentState[Any](
            messages=[
                HumanMessage("Search for John"),
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="search", args={}, id="call_123", type="tool_call")],
                ),
                ToolMessage(content="Found: john@example.com", tool_call_id="call_123"),
            ]
        )

        result = middleware.before_model(state, Runtime())

        assert result is not None
        # Check that the tool message was redacted
        tool_msg = result["messages"][2]
        assert isinstance(tool_msg, ToolMessage)
        assert "[REDACTED_EMAIL]" in tool_msg.content
        assert "john@example.com" not in tool_msg.content

    def test_apply_to_tool_results_mask_strategy(self) -> None:
        """Test that mask strategy works for tool results."""
        middleware = PIIMiddleware(
            "ip", strategy="mask", apply_to_input=False, apply_to_tool_results=True
        )

        state = AgentState[Any](
            messages=[
                HumanMessage("Get server IP"),
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="get_ip", args={}, id="call_456", type="tool_call")],
                ),
                ToolMessage(content="Server IP: 192.168.1.100", tool_call_id="call_456"),
            ]
        )

        result = middleware.before_model(state, Runtime())

        assert result is not None
        tool_msg = result["messages"][2]
        assert "*.*.*.100" in tool_msg.content
        assert "192.168.1.100" not in tool_msg.content

    def test_apply_to_tool_results_block_strategy(self) -> None:
        """Test that block strategy raises error for PII in tool results."""
        middleware = PIIMiddleware(
            "email", strategy="block", apply_to_input=False, apply_to_tool_results=True
        )

        state = AgentState[Any](
            messages=[
                HumanMessage("Search for user"),
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="search", args={}, id="call_789", type="tool_call")],
                ),
                ToolMessage(content="User email: sensitive@example.com", tool_call_id="call_789"),
            ]
        )

        with pytest.raises(PIIDetectionError) as exc_info:
            middleware.before_model(state, Runtime())

        assert exc_info.value.pii_type == "email"
        assert len(exc_info.value.matches) == 1

    def test_custom_regex_detector(self) -> None:
        # Custom regex for API keys
        middleware = PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="redact",
        )

        state = AgentState[Any](messages=[HumanMessage("Key: sk-abcdefghijklmnopqrstuvwxyz123456")])
        result = middleware.before_model(state, Runtime())

        assert result is not None
        assert "[REDACTED_API_KEY]" in result["messages"][0].content

    def test_custom_callable_detector(self) -> None:
        # Custom detector function
        def detect_custom(content: str) -> list[PIIMatch]:
            matches = []
            if "CONFIDENTIAL" in content:
                idx = content.index("CONFIDENTIAL")
                matches.append(
                    PIIMatch(
                        type="confidential",
                        value="CONFIDENTIAL",
                        start=idx,
                        end=idx + 12,
                    )
                )
            return matches

        middleware = PIIMiddleware(
            "confidential",
            detector=detect_custom,
            strategy="redact",
        )

        state = AgentState[Any](messages=[HumanMessage("This is CONFIDENTIAL information")])
        result = middleware.before_model(state, Runtime())

        assert result is not None
        assert "[REDACTED_CONFIDENTIAL]" in result["messages"][0].content

    def test_custom_callable_detector_with_text_key_hash(self) -> None:
        """Custom detectors returning 'text' instead of 'value' must work with hash strategy.

        Regression test for https://github.com/langchain-ai/langchain/issues/35647:
        Custom detectors documented to return {"text", "start", "end"} caused
        KeyError: 'value' when used with hash or mask strategies.
        """

        def detect_phone(content: str) -> list[dict]:  # type: ignore[type-arg]
            return [
                {"text": m.group(), "start": m.start(), "end": m.end()}
                for m in re.finditer(r"\+91[\s.-]?\d{10}", content)
            ]

        middleware = PIIMiddleware(
            "indian_phone",
            detector=detect_phone,
            strategy="hash",
            apply_to_input=True,
        )

        state = AgentState[Any](messages=[HumanMessage("Call +91 9876543210")])
        result = middleware.before_model(state, Runtime())

        assert result is not None
        assert "<indian_phone_hash:" in result["messages"][0].content
        assert "+91 9876543210" not in result["messages"][0].content

    def test_custom_callable_detector_with_text_key_mask(self) -> None:
        """Custom detectors returning 'text' instead of 'value' must work with mask strategy."""

        def detect_phone(content: str) -> list[dict]:  # type: ignore[type-arg]
            return [
                {"text": m.group(), "start": m.start(), "end": m.end()}
                for m in re.finditer(r"\+91[\s.-]?\d{10}", content)
            ]

        middleware = PIIMiddleware(
            "indian_phone",
            detector=detect_phone,
            strategy="mask",
            apply_to_input=True,
        )

        state = AgentState[Any](messages=[HumanMessage("Call +91 9876543210")])
        result = middleware.before_model(state, Runtime())

        assert result is not None
        assert "****" in result["messages"][0].content
        assert "+91 9876543210" not in result["messages"][0].content

    def test_sequential_application(self) -> None:
        """Test that multiple PII types are detected when applied sequentially."""
        # First apply email middleware
        email_middleware = PIIMiddleware("email", strategy="redact")
        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com, IP: 192.168.1.1")])
        result1 = email_middleware.before_model(state, Runtime())

        # Then apply IP middleware to the result
        ip_middleware = PIIMiddleware("ip", strategy="mask")
        assert result1 is not None
        state_with_email_redacted = AgentState[Any](messages=result1["messages"])
        result2 = ip_middleware.before_model(state_with_email_redacted, Runtime())

        assert result2 is not None
        content = result2["messages"][0].content

        # Email should be redacted
        assert "[REDACTED_EMAIL]" in content
        assert "test@example.com" not in content

        # IP should be masked
        assert "*.*.*.1" in content
        assert "192.168.1.1" not in content

    def test_custom_detector_for_multiple_types(self) -> None:
        """Test using a single middleware with custom detector for multiple PII types.

        This is an alternative to using multiple middleware instances,
        useful when you want the same strategy for multiple PII types.
        """

        # Combine multiple detectors into one
        def detect_email_and_ip(content: str) -> list[PIIMatch]:
            return detect_email(content) + detect_ip(content)

        middleware = PIIMiddleware(
            "email_or_ip",
            detector=detect_email_and_ip,
            strategy="redact",
        )

        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com, IP: 10.0.0.1")])
        result = middleware.before_model(state, Runtime())

        assert result is not None
        content = result["messages"][0].content
        assert "test@example.com" not in content
        assert "10.0.0.1" not in content


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_shell_execution_policies.py ---

def test_docker_policy_rejects_cpu_limit() -> None:
    with pytest.raises(RuntimeError):
        DockerExecutionPolicy(cpu_time_seconds=1)


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_shell_tool.py ---

def test_executes_command_and_persists_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    middleware = ShellToolMiddleware(workspace_root=workspace)
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        resources = middleware._get_or_create_resources(state)

        middleware._run_shell_tool(resources, {"command": "cd /"}, tool_call_id=None)
        result = middleware._run_shell_tool(resources, {"command": "pwd"}, tool_call_id=None)
        assert isinstance(result, str)
        assert result.strip() == "/"
        echo_result = middleware._run_shell_tool(
            resources, {"command": "echo ready"}, tool_call_id=None
        )
        assert "ready" in echo_result
    finally:
        middleware.after_agent(state, runtime)

def test_restart_resets_session_environment(tmp_path: Path) -> None:
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        resources = middleware._get_or_create_resources(state)

        middleware._run_shell_tool(resources, {"command": "export FOO=bar"}, tool_call_id=None)
        restart_message = middleware._run_shell_tool(
            resources, {"restart": True}, tool_call_id=None
        )
        assert "restarted" in restart_message.lower()
        resources = middleware._get_or_create_resources(state)  # reacquire after restart
        result = middleware._run_shell_tool(
            resources, {"command": "echo ${FOO:-unset}"}, tool_call_id=None
        )
        assert "unset" in result
    finally:
        middleware.after_agent(state, runtime)

def test_truncation_indicator_present(tmp_path: Path) -> None:
    policy = HostExecutionPolicy(max_output_lines=5, command_timeout=5.0)
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace", execution_policy=policy)
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        resources = middleware._get_or_create_resources(state)
        result = middleware._run_shell_tool(resources, {"command": "seq 1 20"}, tool_call_id=None)
        assert "Output truncated" in result
    finally:
        middleware.after_agent(state, runtime)

def test_timeout_returns_error(tmp_path: Path) -> None:
    policy = HostExecutionPolicy(command_timeout=0.5)
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace", execution_policy=policy)
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        resources = middleware._get_or_create_resources(state)
        start = time.monotonic()
        result = middleware._run_shell_tool(resources, {"command": "sleep 2"}, tool_call_id=None)
        elapsed = time.monotonic() - start
        assert elapsed < policy.command_timeout + 2.0
        assert "timed out" in result.lower()
    finally:
        middleware.after_agent(state, runtime)

def test_redaction_policy_applies(tmp_path: Path) -> None:
    middleware = ShellToolMiddleware(
        workspace_root=tmp_path / "workspace",
        redaction_rules=(RedactionRule(pii_type="email", strategy="redact"),),
    )
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        resources = middleware._get_or_create_resources(state)
        message = middleware._run_shell_tool(
            resources,
            {"command": "printf 'Contact: user@example.com\\n'"},
            tool_call_id=None,
        )
        assert "[REDACTED_EMAIL]" in message
        assert "user@example.com" not in message
    finally:
        middleware.after_agent(state, runtime)

def test_startup_and_shutdown_commands(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    middleware = ShellToolMiddleware(
        workspace_root=workspace,
        startup_commands=("touch startup.txt",),
        shutdown_commands=("touch shutdown.txt",),
    )
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        assert (workspace / "startup.txt").exists()
    finally:
        middleware.after_agent(state, runtime)
    assert (workspace / "shutdown.txt").exists()

def test_normalize_env_coercion(tmp_path: Path) -> None:
    """Test that environment values are coerced to strings."""
    middleware = ShellToolMiddleware(
        workspace_root=tmp_path / "workspace", env={"NUM": 42, "BOOL": True}
    )
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        resources = middleware._get_or_create_resources(state)
        result = middleware._run_shell_tool(
            resources, {"command": "echo $NUM $BOOL"}, tool_call_id=None
        )
        assert "42" in result
        assert "True" in result
    finally:
        middleware.after_agent(state, runtime)

def test_shell_tool_missing_command_string(tmp_path: Path) -> None:
    """Test that shell tool raises an error when command is not a string."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        resources = middleware._get_or_create_resources(state)

        with pytest.raises(ToolException, match="expects a 'command' string"):
            middleware._run_shell_tool(resources, {"command": None}, tool_call_id=None)

        with pytest.raises(ToolException, match="expects a 'command' string"):
            middleware._run_shell_tool(
                resources,
                {"command": 123},
                tool_call_id=None,
            )
    finally:
        middleware.after_agent(state, runtime)

def test_tool_message_formatting_with_id(tmp_path: Path) -> None:
    """Test that tool messages are properly formatted with tool_call_id."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        resources = middleware._get_or_create_resources(state)

        result = middleware._run_shell_tool(
            resources, {"command": "echo test"}, tool_call_id="test-id-123"
        )

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "test-id-123"
        assert result.name == "shell"
        assert result.status == "success"
        assert "test" in result.content
    finally:
        middleware.after_agent(state, runtime)

def test_nonzero_exit_code_returns_error(tmp_path: Path) -> None:
    """Test that non-zero exit codes are marked as errors."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        resources = middleware._get_or_create_resources(state)

        result = middleware._run_shell_tool(
            resources,
            {"command": "false"},  # Command that exits with 1 but doesn't kill shell
            tool_call_id="test-id",
        )

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "Exit code: 1" in result.content
        assert result.artifact["exit_code"] == 1
    finally:
        middleware.after_agent(state, runtime)

def test_truncation_by_bytes(tmp_path: Path) -> None:
    """Test that output is truncated by bytes when max_output_bytes is exceeded."""
    policy = HostExecutionPolicy(max_output_bytes=50, command_timeout=5.0)
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace", execution_policy=policy)
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        resources = middleware._get_or_create_resources(state)

        result = middleware._run_shell_tool(
            resources, {"command": "python3 -c 'print(\"x\" * 100)'"}, tool_call_id=None
        )

        assert "truncated at 50 bytes" in result.lower()
    finally:
        middleware.after_agent(state, runtime)

def test_startup_command_failure(tmp_path: Path) -> None:
    """Test that startup command failure raises an error."""
    policy = HostExecutionPolicy(startup_timeout=1.0)
    middleware = ShellToolMiddleware(
        workspace_root=tmp_path / "workspace", startup_commands=("exit 1",), execution_policy=policy
    )
    runtime = Runtime()
    state = _empty_state()
    with pytest.raises(RuntimeError, match=r"Startup command.*failed"):
        middleware.before_agent(state, runtime)

def test_shutdown_command_failure_logged(tmp_path: Path) -> None:
    """Test that shutdown command failures are logged but don't raise."""
    policy = HostExecutionPolicy(command_timeout=1.0)
    middleware = ShellToolMiddleware(
        workspace_root=tmp_path / "workspace",
        shutdown_commands=("exit 1",),
        execution_policy=policy,
    )
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
    finally:
        # Should not raise despite shutdown command failing
        middleware.after_agent(state, runtime)

def test_shutdown_command_timeout_logged(tmp_path: Path) -> None:
    """Test that shutdown command timeouts are logged but don't raise."""
    policy = HostExecutionPolicy(command_timeout=0.1)
    middleware = ShellToolMiddleware(
        workspace_root=tmp_path / "workspace",
        execution_policy=policy,
        shutdown_commands=("sleep 2",),
    )
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
    finally:
        # Should not raise despite shutdown command timing out
        middleware.after_agent(state, runtime)

def test_empty_output_replaced_with_no_output(tmp_path: Path) -> None:
    """Test that empty command output is replaced with '<no output>'."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        resources = middleware._get_or_create_resources(state)

        result = middleware._run_shell_tool(
            resources,
            {"command": "true"},  # Command that produces no output
            tool_call_id=None,
        )

        assert "<no output>" in result
    finally:
        middleware.after_agent(state, runtime)

def test_stderr_output_labeling(tmp_path: Path) -> None:
    """Test that stderr output is properly labeled."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    runtime = Runtime()
    state = _empty_state()
    try:
        updates = middleware.before_agent(state, runtime)
        if updates:
            state.update(cast("ShellToolState", updates))
        resources = middleware._get_or_create_resources(state)

        result = middleware._run_shell_tool(
            resources, {"command": "echo error >&2"}, tool_call_id=None
        )

        assert "[stderr] error" in result
    finally:
        middleware.after_agent(state, runtime)

async def test_async_methods_delegate_to_sync(tmp_path: Path) -> None:
    """Test that async methods properly delegate to sync methods."""
    middleware = ShellToolMiddleware(workspace_root=tmp_path / "workspace")
    try:
        state = _empty_state()

        # Test abefore_agent
        updates = await middleware.abefore_agent(state, Runtime())
        if updates:
            state.update(cast("ShellToolState", updates))

        # Test aafter_agent
        await middleware.aafter_agent(state, Runtime())
    finally:
        pass

def test_shell_middleware_resumable_after_interrupt(tmp_path: Path) -> None:
    """Test that shell middleware is resumable after an interrupt.

    This test simulates a scenario where:
    1. The middleware creates a shell session
    2. A command is executed
    3. The agent is interrupted (state is preserved)
    4. The agent resumes with the same state
    5. The shell session is reused (not recreated)
    """
    workspace = tmp_path / "workspace"
    middleware = ShellToolMiddleware(workspace_root=workspace)

    # Simulate first execution (before interrupt)
    runtime = Runtime()
    state = _empty_state()
    updates = middleware.before_agent(state, runtime)
    if updates:
        state.update(cast("ShellToolState", updates))

    # Get the resources and verify they exist
    resources = middleware._get_or_create_resources(state)
    initial_session = resources.session
    initial_tempdir = resources.tempdir

    # Execute a command to set state
    middleware._run_shell_tool(resources, {"command": "export TEST_VAR=hello"}, tool_call_id=None)

    # Simulate interrupt - state is preserved, but we don't call after_agent
    # In a real scenario, the state would be checkpointed here

    # Simulate resumption - call before_agent again with same state
    # This should reuse existing resources, not create new ones
    updates = middleware.before_agent(state, runtime)
    if updates:
        state.update(cast("ShellToolState", updates))

    # Get resources again - should be the same session
    resumed_resources = middleware._get_or_create_resources(state)

    # Verify the session was reused (same object reference)
    assert resumed_resources.session is initial_session
    assert resumed_resources.tempdir is initial_tempdir

    # Verify the session state persisted (environment variable still set)
    result = middleware._run_shell_tool(
        resumed_resources, {"command": "echo ${TEST_VAR:-unset}"}, tool_call_id=None
    )
    assert "hello" in result
    assert "unset" not in result

    # Clean up
    middleware.after_agent(state, runtime)


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_summarization.py ---

def test_summarization_middleware_profile_inference_triggers_summary() -> None:
    """Ensure automatic profile inference triggers summarization when limits are exceeded."""

    def token_counter(messages: Iterable[MessageLikeRepresentation]) -> int:
        return len(list(messages)) * 200

    middleware = SummarizationMiddleware(
        model=ProfileChatModel(),
        trigger=("fraction", 0.81),
        keep=("fraction", 0.5),
        token_counter=token_counter,
    )

    state = AgentState[Any](
        messages=[
            HumanMessage(content="Message 1"),
            AIMessage(content="Message 2"),
            HumanMessage(content="Message 3"),
            AIMessage(content="Message 4"),
        ]
    )

    # Test we don't engage summarization
    # we have total_tokens = 4 * 200 = 800
    # and max_input_tokens = 1000
    # since 0.81 * 1000 == 810 > 800 -> summarization not triggered
    result = middleware.before_model(state, Runtime())
    assert result is None

    # Engage summarization
    # since 0.80 * 1000 == 800 <= 800
    middleware = SummarizationMiddleware(
        model=ProfileChatModel(),
        trigger=("fraction", 0.80),
        keep=("fraction", 0.5),
        token_counter=token_counter,
    )
    result = middleware.before_model(state, Runtime())
    assert result is not None
    assert isinstance(result["messages"][0], RemoveMessage)
    summary_message = result["messages"][1]
    assert isinstance(summary_message, HumanMessage)
    assert summary_message.text.startswith("Here is a summary of the conversation")
    assert len(result["messages"][2:]) == 2  # Preserved messages
    assert [message.content for message in result["messages"][2:]] == [
        "Message 3",
        "Message 4",
    ]

    # With keep=("fraction", 0.6) the target token allowance becomes 600,
    # so the cutoff shifts to keep the last three messages instead of two.
    middleware = SummarizationMiddleware(
        model=ProfileChatModel(),
        trigger=("fraction", 0.80),
        keep=("fraction", 0.6),
        token_counter=token_counter,
    )
    result = middleware.before_model(state, Runtime())
    assert result is not None
    assert [message.content for message in result["messages"][2:]] == [
        "Message 2",
        "Message 3",
        "Message 4",
    ]

    # Once keep=("fraction", 0.8) the inferred limit equals the full
    # context (target tokens = 800), so token-based retention keeps everything
    # and summarization is skipped entirely.
    middleware = SummarizationMiddleware(
        model=ProfileChatModel(),
        trigger=("fraction", 0.80),
        keep=("fraction", 0.8),
        token_counter=token_counter,
    )
    assert middleware.before_model(state, Runtime()) is None

    # Test with tokens_to_keep as absolute int value
    middleware_int = SummarizationMiddleware(
        model=ProfileChatModel(),
        trigger=("fraction", 0.80),
        keep=("tokens", 400),  # Keep exactly 400 tokens (2 messages)
        token_counter=token_counter,
    )
    result = middleware_int.before_model(state, Runtime())
    assert result is not None
    assert [message.content for message in result["messages"][2:]] == [
        "Message 3",
        "Message 4",
    ]

    # Test with tokens_to_keep as larger int value
    middleware_int_large = SummarizationMiddleware(
        model=ProfileChatModel(),
        trigger=("fraction", 0.80),
        keep=("tokens", 600),  # Keep 600 tokens (3 messages)
        token_counter=token_counter,
    )
    result = middleware_int_large.before_model(state, Runtime())
    assert result is not None
    assert [message.content for message in result["messages"][2:]] == [
        "Message 2",
        "Message 3",
        "Message 4",
    ]

def test_summarization_middleware_token_retention_preserves_ai_tool_pairs() -> None:
    """Ensure token retention preserves AI/Tool message pairs together."""

    def token_counter(messages: Iterable[MessageLikeRepresentation]) -> int:
        return sum(len(getattr(message, "content", "")) for message in messages)

    middleware = SummarizationMiddleware(
        model=ProfileChatModel(),
        trigger=("fraction", 0.1),
        keep=("fraction", 0.5),
        token_counter=token_counter,
    )

    # Total tokens: 300 + 200 + 50 + 180 + 160 = 890
    # Target keep: 500 tokens (50% of 1000)
    # Binary search finds cutoff around index 2 (ToolMessage)
    # We move back to index 1 to preserve the AIMessage with its ToolMessage
    messages: list[AnyMessage] = [
        HumanMessage(content="H" * 300),
        AIMessage(
            content="A" * 200,
            tool_calls=[{"name": "test", "args": {}, "id": "call-1"}],
        ),
        ToolMessage(content="T" * 50, tool_call_id="call-1"),
        HumanMessage(content="H" * 180),
        HumanMessage(content="H" * 160),
    ]

    state = AgentState[Any](messages=messages)
    result = middleware.before_model(state, Runtime())
    assert result is not None

    preserved_messages = result["messages"][2:]
    # We move the cutoff back to include the AIMessage with its ToolMessage
    # So we preserve messages from index 1 onward (AI + Tool + Human + Human)
    assert preserved_messages == messages[1:]

    # Verify the AI/Tool pair is preserved together
    assert isinstance(preserved_messages[0], AIMessage)
    assert preserved_messages[0].tool_calls
    assert isinstance(preserved_messages[1], ToolMessage)
    assert preserved_messages[1].tool_call_id == preserved_messages[0].tool_calls[0]["id"]


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_todo.py ---

def test_todo_middleware_write_todos_tool_execution(
    todos: list[dict[str, Any]], expected_message: str
) -> None:
    """Test that the write_todos tool executes correctly."""
    tool_call = {
        "args": {"todos": todos},
        "name": "write_todos",
        "type": "tool_call",
        "id": "test_call",
    }
    result = write_todos.invoke(tool_call)
    assert result.update["todos"] == todos
    assert result.update["messages"][0].content == expected_message

def test_todo_middleware_write_todos_tool_validation_errors(
    invalid_todos: list[dict[str, Any]],
) -> None:
    """Test that the write_todos tool rejects invalid input."""
    tool_call = {
        "args": {"todos": invalid_todos},
        "name": "write_todos",
        "type": "tool_call",
        "id": "test_call",
    }
    with pytest.raises(ValueError, match="1 validation error for write_todos"):
        write_todos.invoke(tool_call)


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_call_limit.py ---

def test_middleware_initialization_validation() -> None:
    """Test that middleware initialization validates parameters correctly."""
    # Test that at least one limit must be specified
    with pytest.raises(ValueError, match="At least one limit must be specified"):
        ToolCallLimitMiddleware()

    # Test valid initialization with both limits
    middleware = ToolCallLimitMiddleware(thread_limit=5, run_limit=3)
    assert middleware.thread_limit == 5
    assert middleware.run_limit == 3
    assert middleware.exit_behavior == "continue"
    assert middleware.tool_name is None

    # Test with tool name
    middleware = ToolCallLimitMiddleware(tool_name="search", thread_limit=5)
    assert middleware.tool_name == "search"
    assert middleware.thread_limit == 5
    assert middleware.run_limit is None

    # Test exit behaviors
    for behavior in ["error", "end", "continue"]:
        middleware = ToolCallLimitMiddleware(thread_limit=5, exit_behavior=behavior)
        assert middleware.exit_behavior == behavior

    # Test invalid exit behavior
    with pytest.raises(ValueError, match="Invalid exit_behavior"):
        ToolCallLimitMiddleware(thread_limit=5, exit_behavior="invalid")  # type: ignore[arg-type]

    # Test run_limit exceeding thread_limit
    with pytest.raises(
        ValueError,
        match=r"run_limit .* cannot exceed thread_limit",
    ):
        ToolCallLimitMiddleware(thread_limit=3, run_limit=5)

    # Test run_limit equal to thread_limit (should be valid)
    middleware = ToolCallLimitMiddleware(thread_limit=5, run_limit=5)
    assert middleware.thread_limit == 5
    assert middleware.run_limit == 5

    # Test run_limit less than thread_limit (should be valid)
    middleware = ToolCallLimitMiddleware(thread_limit=5, run_limit=3)
    assert middleware.thread_limit == 5
    assert middleware.run_limit == 3

def test_middleware_unit_functionality() -> None:
    """Test that the middleware works as expected in isolation.

    Tests basic count tracking, thread limit, run limit, and limit-not-exceeded cases.
    """
    middleware = ToolCallLimitMiddleware(thread_limit=3, run_limit=2, exit_behavior="end")
    runtime = None

    # Test when limits are not exceeded - counts should increment normally
    state = ToolCallLimitState(
        messages=[
            HumanMessage("Question"),
            AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ],
        thread_tool_call_count={},
        run_tool_call_count={},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["thread_tool_call_count"] == {"__all__": 1}
    assert result["run_tool_call_count"] == {"__all__": 1}
    assert "jump_to" not in result

    # Test thread limit exceeded (start at thread_limit so next call will exceed)
    state = ToolCallLimitState(
        messages=[
            HumanMessage("Question 2"),
            AIMessage("Response 2", tool_calls=[{"name": "search", "args": {}, "id": "3"}]),
        ],
        thread_tool_call_count={"__all__": 3},  # Already exceeds thread_limit=3
        run_tool_call_count={"__all__": 0},  # No calls yet
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    # Check the ToolMessage (sent to model - no thread/run details)
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.status == "error"
    assert "Tool call limit exceeded" in tool_msg.content
    # Should include "Do not" instruction
    assert "Do not" in tool_msg.content, (
        "Tool message should include 'Do not' instruction when limit exceeded"
    )
    # Check the final AI message (displayed to user - includes thread/run details)
    final_ai_msg = result["messages"][-1]
    assert isinstance(final_ai_msg, AIMessage)
    assert isinstance(final_ai_msg.content, str)
    assert "limit" in final_ai_msg.content.lower()
    assert "thread limit exceeded" in final_ai_msg.content.lower()
    # Thread count stays at 3 (blocked call not counted)
    assert result["thread_tool_call_count"] == {"__all__": 3}
    # Run count goes to 1 (includes blocked call)
    assert result["run_tool_call_count"] == {"__all__": 1}

    # Test run limit exceeded (thread count must be >= run count)
    state = ToolCallLimitState(
        messages=[
            HumanMessage("Question"),
            AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ],
        thread_tool_call_count={"__all__": 2},
        run_tool_call_count={"__all__": 2},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    # Check the final AI message includes run limit details
    final_ai_msg = result["messages"][-1]
    assert "run limit exceeded" in final_ai_msg.content
    assert "3/2 calls" in final_ai_msg.content
    # Check the tool message (sent to model) - should always include "Do not" instruction
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert "Tool call limit exceeded" in tool_msg.content
    assert "Do not" in tool_msg.content, (
        "Tool message should include 'Do not' instruction for both run and thread limits"
    )

def test_middleware_end_behavior_with_unrelated_parallel_tool_calls() -> None:
    """Test middleware 'end' behavior with unrelated parallel tool calls.

    Test that 'end' behavior raises NotImplementedError when there are parallel calls
    to unrelated tools.

    When limiting a specific tool with "end" behavior and the model proposes parallel calls
    to BOTH the limited tool AND other tools, we can't handle this scenario (we'd be stopping
    execution while other tools should run).
    """
    # Limit search tool specifically
    middleware = ToolCallLimitMiddleware(tool_name="search", thread_limit=1, exit_behavior="end")
    runtime = None

    # Test with search + calculator calls when search exceeds limit
    state = ToolCallLimitState(
        messages=[
            AIMessage(
                "Response",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "1"},
                    {"name": "calculator", "args": {}, "id": "2"},
                ],
            ),
        ],
        thread_tool_call_count={"search": 1},
        run_tool_call_count={"search": 1},
    )

    with pytest.raises(
        NotImplementedError, match="Cannot end execution with other tool calls pending"
    ):
        middleware.after_model(state, runtime)  # type: ignore[arg-type]

def test_middleware_with_specific_tool() -> None:
    """Test middleware that limits a specific tool while ignoring others."""
    middleware = ToolCallLimitMiddleware(
        tool_name="search", thread_limit=2, run_limit=1, exit_behavior="end"
    )
    runtime = None

    # Test search tool exceeding run limit
    state = ToolCallLimitState(
        messages=[
            AIMessage("Response 2", tool_calls=[{"name": "search", "args": {}, "id": "3"}]),
        ],
        thread_tool_call_count={"search": 1},
        run_tool_call_count={"search": 1},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "search" in result["messages"][0].content.lower()

    # Test calculator tool - should be ignored by search-specific middleware
    state = ToolCallLimitState(
        messages=[
            AIMessage("Response", tool_calls=[{"name": "calculator", "args": {}, "id": "1"}] * 10),
        ],
        thread_tool_call_count={},
        run_tool_call_count={},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is None, "Calculator calls shouldn't be counted by search-specific middleware"

def test_middleware_error_behavior() -> None:
    """Test middleware error behavior.

    Test that middleware raises ToolCallLimitExceededError when configured with
    exit_behavior='error'.
    """
    middleware = ToolCallLimitMiddleware(thread_limit=2, exit_behavior="error")
    runtime = None

    state = ToolCallLimitState(
        messages=[AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}])],
        thread_tool_call_count={"__all__": 2},
        run_tool_call_count={"__all__": 2},
    )

    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        middleware.after_model(state, runtime)  # type: ignore[arg-type]

    error = exc_info.value
    # Thread count in error message shows hypothetical count (what it would have been)
    assert error.thread_count == 3
    assert error.thread_limit == 2
    # Run count includes the blocked call
    assert error.run_count == 3
    assert error.tool_name is None

def test_exception_error_messages() -> None:
    """Test that error messages include expected information."""
    # Test for specific tool
    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        raise ToolCallLimitExceededError(
            thread_count=5, run_count=3, thread_limit=4, run_limit=2, tool_name="search"
        )
    msg = str(exc_info.value)
    assert "search" in msg.lower()
    assert "5/4" in msg or "thread" in msg.lower()

    # Test for all tools
    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        raise ToolCallLimitExceededError(
            thread_count=10, run_count=5, thread_limit=8, run_limit=None, tool_name=None
        )
    msg = str(exc_info.value)
    assert "10/8" in msg or "thread" in msg.lower()

def test_limit_reached_but_not_exceeded() -> None:
    """Test that limits are only triggered when exceeded (>), not when reached (==)."""
    middleware = ToolCallLimitMiddleware(thread_limit=3, run_limit=2, exit_behavior="end")
    runtime = None

    # Test when limit is reached exactly (count = limit) - should not trigger
    state = ToolCallLimitState(
        messages=[AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}])],
        thread_tool_call_count={"__all__": 2},  # After +1 will be exactly 3
        run_tool_call_count={"__all__": 1},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert "jump_to" not in result
    assert result["thread_tool_call_count"]["__all__"] == 3

    # Test when limit is exceeded (count > limit) - should trigger
    state = ToolCallLimitState(
        messages=[AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}])],
        thread_tool_call_count={"__all__": 3},  # After +1 will be 4 > 3
        run_tool_call_count={"__all__": 1},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert "jump_to" in result
    assert result["jump_to"] == "end"

def test_thread_count_excludes_blocked_run_calls() -> None:
    """Test that thread count only includes allowed calls, not blocked run-scoped calls.

    When run_limit is lower than thread_limit and multiple parallel calls are made,
    only the allowed calls should increment the thread count.

    Example: If run_limit=1 and 3 parallel calls are made, thread count should be 1
    (not 3) because the other 2 were blocked by the run limit.
    """
    # Set run_limit=1, thread_limit=10 (much higher)
    middleware = ToolCallLimitMiddleware(thread_limit=10, run_limit=1, exit_behavior="continue")
    runtime = None

    # Make 3 parallel tool calls - only 1 should be allowed by run_limit
    state = ToolCallLimitState(
        messages=[
            AIMessage(
                "Response",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "1"},
                    {"name": "search", "args": {}, "id": "2"},
                    {"name": "search", "args": {}, "id": "3"},
                ],
            )
        ],
        thread_tool_call_count={},
        run_tool_call_count={},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None

    # Thread count should be 1 (only the allowed call)
    assert result["thread_tool_call_count"]["__all__"] == 1, (
        "Thread count should only include the 1 allowed call, not the 2 blocked calls"
    )
    # Run count should be 3 (all attempted calls)
    assert result["run_tool_call_count"]["__all__"] == 3, (
        "Run count should include all 3 attempted calls"
    )

    # Verify 2 error messages were created for blocked calls
    assert "messages" in result
    error_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(error_messages) == 2, "Should have 2 error messages for the 2 blocked calls"

def test_unified_error_messages() -> None:
    """Test that error messages instruct model not to call again for both run and thread limits.

    Previously, only thread limit messages included 'Do not' instruction.
    Now both run and thread limit messages should include it.
    """
    middleware = ToolCallLimitMiddleware(thread_limit=10, run_limit=1, exit_behavior="continue")
    runtime = None

    # Test with run limit exceeded (thread limit not exceeded)
    state = ToolCallLimitState(
        messages=[AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "2"}])],
        thread_tool_call_count={"__all__": 1},  # Under thread limit
        run_tool_call_count={"__all__": 1},  # At run limit, next call will exceed
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None

    # Check the error message includes "Do not" instruction
    error_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(error_messages) == 1
    error_content = error_messages[0].content
    assert "Do not" in error_content, (
        "Run limit error message should include 'Do not' instruction to guide model behavior"
    )


# --- libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_retry.py ---

def test_tool_retry_initialization_custom() -> None:
    """Test ToolRetryMiddlewareinitialization with custom values."""
    retry = ToolRetryMiddleware(
        max_retries=5,
        tools=["tool1", "tool2"],
        retry_on=(ValueError, RuntimeError),
        on_failure="error",
        backoff_factor=1.5,
        initial_delay=0.5,
        max_delay=30.0,
        jitter=False,
    )

    assert retry.max_retries == 5
    assert retry._tool_filter == ["tool1", "tool2"]
    assert retry.tools == []
    assert retry.retry_on == (ValueError, RuntimeError)
    assert retry.on_failure == "error"
    assert retry.backoff_factor == 1.5
    assert retry.initial_delay == 0.5
    assert retry.max_delay == 30.0
    assert retry.jitter is False


# --- libs/langchain_v1/tests/unit_tests/agents/middleware_typing/test_middleware_typing.py ---

def test_model_request_preserves_context_type() -> None:
    """Test that ModelRequest.override() preserves ContextT."""
    request: ModelRequest[UserContext] = ModelRequest(
        model=None,  # type: ignore[arg-type]
        messages=[HumanMessage(content="test")],
        runtime=None,
    )

    # Override should preserve the type parameter
    new_request: ModelRequest[UserContext] = request.override(
        messages=[HumanMessage(content="updated")]
    )

    assert type(request) is type(new_request)


# --- libs/langchain_v1/tests/unit_tests/agents/test_response_format_integration.py ---

def test_inference_to_native_output(*, use_responses_api: bool) -> None:
    """Test that native output is inferred when a model supports it."""
    model_kwargs: dict[str, Any] = {"model": "gpt-5", "use_responses_api": use_responses_api}

    if "OPENAI_API_KEY" not in os.environ:
        model_kwargs["api_key"] = "foo"

    model = ChatOpenAI(**model_kwargs)

    agent = create_agent(
        model,
        system_prompt=(
            "You are a helpful weather assistant. Please call the get_weather tool "
            "once, then use the WeatherReport tool to generate the final response."
        ),
        tools=[get_weather],
        response_format=WeatherBaseModel,
    )
    response = agent.invoke({"messages": [HumanMessage("What's the weather in Boston?")]})

    assert isinstance(response["structured_response"], WeatherBaseModel)
    assert response["structured_response"].temperature == 75.0
    assert response["structured_response"].condition.lower() == "sunny"
    assert len(response["messages"]) == 4

    assert [m.type for m in response["messages"]] == [
        "human",  # "What's the weather?"
        "ai",  # "What's the weather?"
        "tool",  # "The weather is sunny and 75°F."
        "ai",  # structured response
    ]

def test_inference_to_tool_output(*, use_responses_api: bool) -> None:
    """Test that tool output is inferred when a model supports it."""
    model_kwargs: dict[str, Any] = {"model": "gpt-5", "use_responses_api": use_responses_api}

    if "OPENAI_API_KEY" not in os.environ:
        model_kwargs["api_key"] = "foo"

    model = ChatOpenAI(**model_kwargs)

    agent = create_agent(
        model,
        system_prompt=(
            "You are a helpful weather assistant. Please call the get_weather tool "
            "once, then use the WeatherReport tool to generate the final response."
        ),
        tools=[get_weather],
        response_format=ToolStrategy(WeatherBaseModel),
    )
    response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

    assert isinstance(response["structured_response"], WeatherBaseModel)
    assert response["structured_response"].temperature == 75.0
    assert response["structured_response"].condition.lower() == "sunny"
    assert len(response["messages"]) == 5

    assert [m.type for m in response["messages"]] == [
        "human",  # "What's the weather?"
        "ai",  # "What's the weather?"
        "tool",  # "The weather is sunny and 75°F."
        "ai",  # structured response
        "tool",  # artificial tool message
    ]


# --- libs/langchain_v1/tests/unit_tests/chat_models/test_chat_models.py ---

def test_configurable() -> None:
    """Test configurable chat model behavior without default parameters.

    Verifies that a configurable chat model initialized without default parameters:
    - Has access to all standard runnable methods (`invoke`, `stream`, etc.)
    - Blocks access to non-configurable methods until configuration is provided
    - Supports declarative operations (`bind_tools`) without mutating original model
    - Can chain declarative operations and configuration to access full functionality
    - Properly resolves to the configured model type when parameters are provided

    Example:
    ```python
    # This creates a configurable model without specifying which model
    model = init_chat_model()

    # This will FAIL - no model specified yet
    model.get_num_tokens("hello")  # AttributeError!

    # This works - provides model at runtime
    response = model.invoke("Hello", config={"configurable": {"model": "gpt-4o"}})
    ```
    """
    model = init_chat_model()

    for method in (
        "invoke",
        "ainvoke",
        "batch",
        "abatch",
        "stream",
        "astream",
        "batch_as_completed",
        "abatch_as_completed",
    ):
        assert hasattr(model, method)

    # Doesn't have access non-configurable, non-declarative methods until a config is
    # provided.
    for method in ("get_num_tokens", "get_num_tokens_from_messages"):
        with pytest.raises(AttributeError):
            getattr(model, method)

    # Can call declarative methods even without a default model.
    model_with_tools = model.bind_tools(
        [{"name": "foo", "description": "foo", "parameters": {}}],
    )

    # Check that original model wasn't mutated by declarative operation.
    assert model._queued_declarative_operations == []

    # Can iteratively call declarative methods.
    model_with_config = model_with_tools.with_config(
        RunnableConfig(tags=["foo"]),
        configurable={"model": "gpt-4o"},
    )
    assert model_with_config.model_name == "gpt-4o"  # type: ignore[attr-defined]

    for method in ("get_num_tokens", "get_num_tokens_from_messages"):
        assert hasattr(model_with_config, method)

    assert model_with_config.model_dump() == {  # type: ignore[attr-defined]
        "name": None,
        "bound": {
            "name": None,
            "disable_streaming": False,
            "disabled_params": None,
            "model_name": "gpt-4o",
            "temperature": None,
            "model_kwargs": {},
            "openai_api_key": SecretStr("foo"),
            "openai_api_base": None,
            "openai_organization": None,
            "openai_proxy": None,
            "output_version": None,
            "request_timeout": None,
            "max_retries": None,
            "presence_penalty": None,
            "reasoning": None,
            "reasoning_effort": None,
            "verbosity": None,
            "frequency_penalty": None,
            "context_management": None,
            "include": None,
            "seed": None,
            "service_tier": None,
            "logprobs": None,
            "top_logprobs": None,
            "logit_bias": None,
            "streaming": False,
            "n": None,
            "top_p": None,
            "truncation": None,
            "max_tokens": None,
            "tiktoken_model_name": None,
            "default_headers": None,
            "default_query": None,
            "stop": None,
            "store": None,
            "extra_body": None,
            "include_response_headers": False,
            "stream_usage": True,
            "use_previous_response_id": False,
            "use_responses_api": None,
        },
        "kwargs": {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "foo", "description": "foo", "parameters": {}},
                },
            ],
        },
        "config": {
            "callbacks": None,
            "configurable": {},
            "metadata": {"model": "gpt-4o"},
            "recursion_limit": 25,
            "tags": ["foo"],
        },
        "config_factories": [],
        "custom_input_type": None,
        "custom_output_type": None,
    }

def test_configurable_with_default() -> None:
    """Test configurable chat model behavior with default parameters.

    Verifies that a configurable chat model initialized with default parameters:
    - Has access to all standard runnable methods (`invoke`, `stream`, etc.)
    - Provides immediate access to non-configurable methods (e.g. `get_num_tokens`)
    - Supports model switching through runtime configuration using `config_prefix`
    - Maintains proper model identity and attributes when reconfigured
    - Can be used in chains with different model providers via configuration

    Example:
    ```python
    # This creates a configurable model with default parameters (model)
    model = init_chat_model("gpt-4o", configurable_fields="any", config_prefix="bar")

    # This works immediately - uses default gpt-4o
    tokens = model.get_num_tokens("hello")

    # This also works - switches to Claude at runtime
    response = model.invoke(
        "Hello", config={"configurable": {"my_model_model": "claude-3-sonnet-20240229"}}
    )
    ```
    """
    model = init_chat_model("gpt-4o", configurable_fields="any", config_prefix="bar")
    for method in (
        "invoke",
        "ainvoke",
        "batch",
        "abatch",
        "stream",
        "astream",
        "batch_as_completed",
        "abatch_as_completed",
    ):
        assert hasattr(model, method)

    # Does have access non-configurable, non-declarative methods since default params
    # are provided.
    for method in ("get_num_tokens", "get_num_tokens_from_messages", "dict"):
        assert hasattr(model, method)

    assert model.model_name == "gpt-4o"

    model_with_tools = model.bind_tools(
        [{"name": "foo", "description": "foo", "parameters": {}}],
    )

    model_with_config = model_with_tools.with_config(
        RunnableConfig(tags=["foo"]),
        configurable={"bar_model": "claude-sonnet-4-5-20250929"},
    )

    assert model_with_config.model == "claude-sonnet-4-5-20250929"  # type: ignore[attr-defined]

    assert model_with_config.model_dump() == {  # type: ignore[attr-defined]
        "name": None,
        "bound": {
            "name": None,
            "disable_streaming": False,
            "effort": None,
            "model": "claude-sonnet-4-5-20250929",
            "mcp_servers": None,
            "max_tokens": 64000,
            "temperature": None,
            "thinking": None,
            "top_k": None,
            "top_p": None,
            "default_request_timeout": None,
            "max_retries": 2,
            "stop_sequences": None,
            "anthropic_api_url": "https://api.anthropic.com",
            "anthropic_proxy": None,
            "context_management": None,
            "anthropic_api_key": SecretStr("bar"),
            "betas": None,
            "default_headers": None,
            "model_kwargs": {},
            "reuse_last_container": None,
            "inference_geo": None,
            "streaming": False,
            "stream_usage": True,
            "output_version": None,
        },
        "kwargs": {
            "tools": [{"name": "foo", "description": "foo", "input_schema": {}}],
        },
        "config": {
            "callbacks": None,
            "configurable": {},
            "metadata": {"bar_model": "claude-sonnet-4-5-20250929"},
            "recursion_limit": 25,
            "tags": ["foo"],
        },
        "config_factories": [],
        "custom_input_type": None,
        "custom_output_type": None,
    }
    prompt = ChatPromptTemplate.from_messages([("system", "foo")])
    chain = prompt | model_with_config
    assert isinstance(chain, RunnableSequence)


# --- libs/partners/anthropic/tests/integration_tests/test_chat_models.py ---

def test_stream() -> None:
    """Test streaming tokens from Anthropic."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    full: BaseMessageChunk | None = None
    chunks_with_input_token_counts = 0
    chunks_with_output_token_counts = 0
    chunks_with_model_name = 0
    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)
        full = cast("BaseMessageChunk", token) if full is None else full + token
        assert isinstance(token, AIMessageChunk)
        if token.usage_metadata is not None:
            if token.usage_metadata.get("input_tokens"):
                chunks_with_input_token_counts += 1
            if token.usage_metadata.get("output_tokens"):
                chunks_with_output_token_counts += 1
        chunks_with_model_name += int("model_name" in token.response_metadata)
    if chunks_with_input_token_counts != 1 or chunks_with_output_token_counts != 1:
        msg = (
            "Expected exactly one chunk with input or output token counts. "
            "AIMessageChunk aggregation adds counts. Check that "
            "this is behaving properly."
        )
        raise AssertionError(
            msg,
        )
    assert chunks_with_model_name == 1
    # check token usage is populated
    assert isinstance(full, AIMessageChunk)
    assert len(full.content_blocks) == 1
    assert full.content_blocks[0]["type"] == "text"
    assert full.content_blocks[0]["text"]
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert full.usage_metadata["total_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )
    assert "stop_reason" in full.response_metadata
    assert "stop_sequence" in full.response_metadata
    assert "model_name" in full.response_metadata

async def test_astream() -> None:
    """Test streaming tokens from Anthropic."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    full: BaseMessageChunk | None = None
    chunks_with_input_token_counts = 0
    chunks_with_output_token_counts = 0
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)
        full = cast("BaseMessageChunk", token) if full is None else full + token
        assert isinstance(token, AIMessageChunk)
        if token.usage_metadata is not None:
            if token.usage_metadata.get("input_tokens"):
                chunks_with_input_token_counts += 1
            if token.usage_metadata.get("output_tokens"):
                chunks_with_output_token_counts += 1
    if chunks_with_input_token_counts != 1 or chunks_with_output_token_counts != 1:
        msg = (
            "Expected exactly one chunk with input or output token counts. "
            "AIMessageChunk aggregation adds counts. Check that "
            "this is behaving properly."
        )
        raise AssertionError(
            msg,
        )
    # check token usage is populated
    assert isinstance(full, AIMessageChunk)
    assert len(full.content_blocks) == 1
    assert full.content_blocks[0]["type"] == "text"
    assert full.content_blocks[0]["text"]
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert full.usage_metadata["total_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )
    assert "stop_reason" in full.response_metadata
    assert "stop_sequence" in full.response_metadata

    # Check expected raw API output
    async_client = llm._async_client
    params: dict = {
        "model": MODEL_NAME,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
    }
    stream = await async_client.messages.create(**params, stream=True)
    async for event in stream:
        if event.type == "message_start":
            assert event.message.usage.input_tokens > 1
            # Different models may report different initial output token counts
            # in the message_start event. Ensure it's a positive value.
            assert event.message.usage.output_tokens >= 1
        elif event.type == "message_delta":
            assert event.usage.output_tokens >= 1
        else:
            pass

async def test_stream_usage() -> None:
    """Test usage metadata can be excluded."""
    model = ChatAnthropic(model_name=MODEL_NAME, stream_usage=False)  # type: ignore[call-arg]
    async for token in model.astream("hi"):
        assert isinstance(token, AIMessageChunk)
        assert token.usage_metadata is None

async def test_stream_usage_override() -> None:
    # check we override with kwarg
    model = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg]
    assert model.stream_usage
    async for token in model.astream("hi", stream_usage=False):
        assert isinstance(token, AIMessageChunk)
        assert token.usage_metadata is None

async def test_async_tool_use() -> None:
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
    )

    llm_with_tools = llm.bind_tools(
        [
            {
                "name": "get_weather",
                "description": "Get weather report for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        ],
    )
    response = await llm_with_tools.ainvoke("what's the weather in san francisco, ca")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, list)
    assert isinstance(response.tool_calls, list)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]

    # Test streaming
    first = True
    chunks: list[BaseMessage | BaseMessageChunk] = []
    async for chunk in llm_with_tools.astream(
        "what's the weather in san francisco, ca",
    ):
        chunks = [*chunks, chunk]
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk  # type: ignore[assignment]
    assert len(chunks) > 1
    assert isinstance(gathered, AIMessageChunk)
    assert isinstance(gathered.tool_call_chunks, list)
    assert len(gathered.tool_call_chunks) == 1
    tool_call_chunk = gathered.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "get_weather"
    assert isinstance(tool_call_chunk["args"], str)
    assert "location" in json.loads(tool_call_chunk["args"])

async def test_ainvoke() -> None:
    """Test invoke tokens."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)
    assert "model_name" in result.response_metadata

def test_invoke() -> None:
    """Test invoke tokens."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    result = llm.invoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)

def test_system_invoke() -> None:
    """Test invoke tokens with a system message."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert cartographer. If asked, you are a cartographer. "
                "STAY IN CHARACTER",
            ),
            ("human", "Are you a mathematician?"),
        ],
    )

    chain = prompt | llm

    result = chain.invoke({})
    assert isinstance(result.content, str)

def test_handle_empty_aimessage() -> None:
    # Anthropic can generate empty AIMessages, which are not valid unless in the last
    # message in a sequence.
    llm = ChatAnthropic(model=MODEL_NAME)
    messages = [
        HumanMessage("Hello"),
        AIMessage([]),
        HumanMessage("My name is Bob."),
    ]
    _ = llm.invoke(messages)

    # Test tool call sequence
    llm_with_tools = llm.bind_tools(
        [
            {
                "name": "get_weather",
                "description": "Get weather report for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        ],
    )
    _ = llm_with_tools.invoke(
        [
            HumanMessage("What's the weather in Boston?"),
            AIMessage(
                content=[],
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"location": "Boston"},
                        "id": "toolu_01V6d6W32QGGSmQm4BT98EKk",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(
                content="It's sunny.", tool_call_id="toolu_01V6d6W32QGGSmQm4BT98EKk"
            ),
            AIMessage([]),
            HumanMessage("Thanks!"),
        ]
    )

def test_anthropic_call() -> None:
    """Test valid call to anthropic."""
    chat = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_anthropic_generate() -> None:
    """Test generate method of anthropic."""
    chat = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    chat_messages: list[list[BaseMessage]] = [
        [HumanMessage(content="How many toes do dogs have?")],
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy

def test_anthropic_streaming() -> None:
    """Test streaming tokens from anthropic."""
    chat = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.stream([message])
    for token in response:
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)

def test_anthropic_multimodal() -> None:
    """Test that multimodal inputs are handled correctly."""
    chat = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        # langchain logo
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAMCAggHCQgGCQgICAcICAgICAgICAYICAgHDAgHCAgICAgIBggICAgICAgICBYICAgICwkKCAgNDQoIDggICQgBAwQEBgUGCgYGCBALCg0QCg0NEA0KCg8LDQoKCgoLDgoQDQoLDQoKCg4NDQ0NDgsQDw0OCg4NDQ4NDQoJDg8OCP/AABEIALAAsAMBEQACEQEDEQH/xAAdAAEAAgEFAQAAAAAAAAAAAAAABwgJAQIEBQYD/8QANBAAAgIBAwIDBwQCAgIDAAAAAQIAAwQFERIIEwYhMQcUFyJVldQjQVGBcZEJMzJiFRYk/8QAGwEBAAMAAwEAAAAAAAAAAAAAAAQFBgEDBwL/xAA5EQACAQIDBQQJBAIBBQAAAAAAAQIDEQQhMQVBUWGREhRxgRMVIjJSU8HR8CNyobFCguEGJGKi4v/aAAwDAQACEQMRAD8ApfJplBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBAEAQBANl16qOTEKB6kkAD+z5Tkcj0On+z7Ub1FlOmanejeavj6dqV6kfsQ1OK4IP8AIM6pVYR1kuqJdLCV6qvCnJ/6v66nL+Ems/RNc+y63+BOvvFL411O/wBW4r5T6D4Saz9E1z7Lrf4Ed4pfGuo9W4r5T6D4Saz9E1z7Lrf4Ed4pfGuo9W4r5T6D4Saz9E1z7Lrf4Ed4pfGuo9W4r5T6D4Saz9E1z7Lrf4Ed4pfGuo9W4r5T6D4Saz9E1z7Lrf4Ed4pfGuo9W4r5T6D4Saz9E1z7Lrf4Ed4pfGuo9W4r5T6D4Saz9E1z7Lrf4Ed4pfGuo9W4r5T6HE1D2e6lQpsu0zU6EXzZ8jTtSoUD9yWuxUAA/kmdkasJaSXVHRVwlekrzpyX+r+mh56m9WHJSGU+hUgg/wBjynaRORvnAEAQBAEAQBAEAQCbennpVzfER95LHE0tX4tlsnJr2B2srw6yQLCpBQ3Me1W+4/VZLKlh4jFRo5ay4cPH7f0XWA2XUxft37MONs34ffRcy/Xsu6bdG0UK2Nh1tkAbHMyAt+Wx2HIi11/SDcQe3jrTXv6IJRVcRUqe88uC0Nxhdn0MMv0458XnJ+e7wVlyJPJkYsTSAIAgCAIAgCAIBqDAIx9qHTbo2tBmycOtcgjYZmOBRlqdjxJtQDuhdye3ette/qhkmliKlP3XlwehXYrZ9DEr9SOfFZS6rXwd1yKCdQ3Srm+HT7yGOXpbPxXLVOLUMTtXXmVgkVliQgvU9qx9h+kz11Ne4fFRrZaS4cfD7f2YfH7LqYT279qHHevH76PlvhKTClEAQBAEAQBAJp6WOn0+I80i7mumYnF8x1LIbSSe3iV2DYq13ElnQ8q6gdijWUuIeKxHoY5e89PuXWy8D3qp7S9iOvN/D9+XiZRNN06uiuvHqrSqmpFrqqrVUrrrUBUREUBVVVAAUAAATNNtu7PR4xUUoxVkskloktxyCZwfRj26jetHPtzrMXSM4Uabj7Vrfj10O2ZdsDbb3bqrCKEYmpeyED8Hs53LZVwvsPg4qN6kbt+OS8t5hdobYqOo44edorK6SzfmtFpz14H16f8Arkz6cmrD1e9crBvsFZy3ropvxC2yo7NTXXXbjhtuXcTmisz91hX2yr4KLjemrNbuPXeMDtuoqihiGnF/5ZJx55ZNceF76GQSUJuhAEAQBAEAhb239WWl+H391s7mXnbAnExu2WqUjdWyLHda6Qw2IXdrCCGFZX5pMo4WdXNZLiyoxm1KOFfZl7UuCtdeN2kvzcRB4d/5JMV7OOVpWRRSWAFmPk1ZTKN9uT1PRi+QHnsj2H12DHYGXLZzS9mV3zVvuVFL/qGDlapSaXFST6qyfS/3tb4M8a4up49WoYlyZGLcCUsTf1B2ZGVgHrsRgVNbqrIwIYAjaVc4Sg+zJWZqaVWFWCnB3T0/PodnqOnV312Y9taW02o1dtViq9dlbAq6OjAqyspIKkEEGfKbTuj7lFSTjJXTyaejXAxd9U/T6fDmYBTzbTMvm+G7FnNRBHcxLLDuWankCrueVlRG5dq7nOlwuI9NHP3lr9zzjamA7rU9n3Jacn8P25eBC0mFKIAgCAIBtdwASfQDc/4nIbsZXulr2ZDR9HwsYpxybqxmZe4Xl71cquyMR69hO3jg+fy0r5n1OWxNX0lRvdovBflz1DZuG7vh4xtZtXl+55vpp5EsyKWZ5X2seH783TdRwsZgmVk4OVRQzMUUXPRYle7gEoCxA5gEqDvsdp2U5KM03omv7I+Ig6lKUIuzaaXmigPtb6HNQ0bEytTGXjZeLiKlhWuu6rINPMLbY1bFqkXHQ908b7CyK+wUqFe+pY2FSSjZpvnl+MwmJ2JVw9OVTtqUYq+Sadt+WaVtd9+W+uLLv5HzB8j/AIlgZ8yRdGfUXXq2JXpGTZtquFUE+cnfMxU2Wu9CzEvaicEsG+/MdzYLbsmexmHdOXaS9l/w+H2PQ9kY9V6apyftxVtdUtJc3x58iykrjQCAIAgFdurzqbPh+lMHFKHVspC6FuLLh427Icp0O4d2ZWREb5WZLGbktJrssMJhvSu8vdX8vh9zP7X2i8LBRp27b46Rj8Vt73JebyVnCfSz0jNqh/8AsGsrZZRcxuoxrms7ua7HmcvLYkOaXJ5Ctjvkb8n/AE+K3TcVi+x+nS6rdyX33eJTbL2S636+JTaeaTveTf8AlLlwjv35ZFmfHnSnoWo47Yo0/FxLOBWnJw8ejHuobb5GVqkUOqnY9qwOjDyI9CKyGKqwd+03ybdjS19mYarHs+jSe5pJNdP6KudBPiTIwNYz/D1jA1WJk91AWKLqGJctDWVg+QFlfdQtsGcVY+//AFgSzx0VKmqi5dJK/wCeZm9iVJ0sRPDye6WWdu1BpXWeV78M8uGd/wCURuCJuqX2YjWNHzMYJyyaKzmYm3Hl71SrOqKW8h307mOT5fLc3mPUSsNV9HUT3aPwf5crNpYbvGHlG2azj+5Zrrp5mKFHBAI9CNx/iak8vTubpwBAEAQDtPCekLk5WHiON0yczFx3H8pbkVVMP7VyJ8zfZi3wTfRHdRh26kI8ZRXk5IzREf6mPPXTSAIB1/iPQa8yjIwrVD05NFuPYrAFWrsrat1YHyIKsRsf2nMXZpo+ZR7UXF77rqYW2xHrJqsHG2smu1T6rapKWKf8OCP6mxvfNHj1nH2XqsnfW6yOVpGr241teVRY9ORS4sqtrPF67B6Mp/2NiCGBIIYMQeGlJWaujsp1JU5KcHZrQyZdK/U3X4ipONdwq1fGQNkVL5JkVbhfe8cE/wDgWKq1e5NFjKD8ttLPm8ThnSd17r0+35qej7N2hHFQs8prVfVcv6J4kIuBAKtdWnV8uj89I090fVeP/wCi8hXq05CvIcg26PmMpDCpgVqUrZaCGqrussLhPSe3P3f7/wCOf4s9tTaXd16On77/APXn48EU58OYl+RremrrRyHbJzdPbI9+LvZZjW21vUlgs5FMe4OqmshVrrscca9jtcSaVKXotydrcVr58zH04znioLFXd3G/a17L08E3u5vJEveGeobX/Cuq2YmttbbjX3NflUu7ZC1VW2OTlaZZuzDHrIbbGXZOFbV9qmwfLElh6Venelqsl4rc+fP6FtT2hicHiHDEu8W7u+ii8lKObtHL3fH/AC1tn1AdReJ4exVvJW/MyEJwcVWG9x2G1zkb8MVNwTbt83kqhmYCVVDDyqytot7/ADeanG46GFh2nm37q4/8c/qVr/4/fZ9k5Obm+J7+Xa430V2soVcrNuuW3LtT+RQUNZKjj3L2QHlRYqWOPqJRVJcvJJWRnth4epKpLE1FqnZ8XJ3b8MuG/LQvdKQ2ZqB/qAYXfFmkLjZWZiINkxszKx0H8JVkW1KP6VAJsIPtRT4pPqjyKtDsVJx4SkvJSdjq59HSIAgCAdp4T1dcbKw8tzsmNmYuQ5/hKsiq1j/SoTPma7UWuKa6o7qM+xUhLhKL8lJXM0RP+pjz100gCAIBjA6x/Y9ZpGq35KofcdSssy8ewA8Vvcl8rHJ3OzrazXAeQNVq8d+3Zx0mDrKpTS3rLy3P6HnG18I6FdzS9mWa/c9V9fPkQTJxRnf+AfHeRpOXj6pjHa/GsDhd+K2p6W0WHY/p31lqidiVDchsyqR8VIKpFxlo/wAv5EjD15UKiqw1X8revMy++DfFtOo4uNqNDcsfKprvrJ8iFZQeLD1Dod0KnzVlI/aZKcXCTi9UerUqkasFOLumk14M8T1L+0uzRdHzdRp8skKlGO2wPC+6xKUt2PkezzN3E7g8NtjvO7D01UqKL03+CzIe0MQ8Ph5VI66Lxbsv7Ks9D3ThTqG/iXOBvSvJsGHTae4L8lWDXZ2QzMzXMt7MoWzzNyW2PzPaYWeNxDj+nDLLPw4dPsZ7Y+CVb/ua3tO7tfitZPzyS5XJS6zOlu3XAmrYSh9Rpq7N2OzKozMYF3RUZyEXIqZ325lVtVyrMOFUjYPEql7MtP6f2J+1tmvE2qU/fWWusfo1/P8AVWfbjruoWabpFGrl/wD5Wq/UOyMhO3mV6QFxaU98BCuzW5dNxW2wcraqeZawku1pQjFVJOn7uWmna1y8uhmMdUqOhSjiPfTlr73o0rXfi1k96V7nq/YP0n6lr99OdqgysfS6qqKw2QbK8rKx6kWrHxcdG2toxlrUA3lU+Q71c3ta+rpr4qFJONOzlnpom9/N8vpkTMBsyriZKeITUEla+rSyUbapLyvzeZkT0fR6saqvFprSmilFrqqrUJXXWo2VEUABVUDbYSgbbd3qbyMVFWSskcucH0ag/wCoBhd8WauuTlZmWh3TIzMrIQ/yluRbap/tXBmwguzFLgkuiPIq0+3UnLjKT8nJ2Orn0dIgCAIBtdAQQfQjY/4nIauZXulr2nDWNHw8kvyyaKxh5e/Hl71SqozsF8h307eQB5fLcvkPQZbE0vR1Gt2q8H+WPUNm4nvGHjK92spfuWT66+ZLMilmIAgHm/aL4ExtVxL9PyaVvptRtkb1WwA9uyths1dqNsRYhDKf39Z905uElKLszor0YVoOE1dP86mH7R/DORdi5OeKz2sI4iZZIKtU+Q11dPJSvl+rS1ZBIKsyDY7krrXJKSjxvbyzPKY0ZuMprSNlLim21p4rPh1t6fA9ieq34Ka1RhW5OA7XKbMcC6ypq7DU/doT9cLyBPNK7ECglmT0nW60FLsN2fPnnroSI4KvKl6aMLxz0zeTavbW3hfy3Wq/4+fbVQKbPDd9wW7vWZGnK2wW2l17l9FTehsS0W5PA/M62uV5CqzhV4+i7+kS5Px4/T8z02wcXHsvDyed24+DzaXg7u3PLLSderP2f3arombi0KXyEFWVVWBu1jU2pc1SD93sqWxAP3dlkHC1FCqm9NOuRd7ToOvhpwjrk14xadv4K7dEPU5gYOI2iZ+RXiql1l2Hk2fJjtVae5ZVbaSUrsW42WB7O2jpYqg8k+exxuGnKXbgr8eOWXmUGxtpUqdP0FV9m12m9Gm72/8AFp8dfEmb22dZmlaXjv7nk42pag4K0U49q3U1t5fqZV1LFErTfl2g4st/8VCjnZXDo4Oc37ScVvv9L/iLXG7Xo0IfpyU57kndeLa0X8vRcq59OnsAzPFWY3iTVmezBa3uMbQOWo2qdhSibcUwa+IrPEBSq9pB/wBjV2GIrxoR9HT1/r/6M/s7A1MbU7ziHeN75/5tbuUF/Oml28h0oDfCAIBE/VL7TRo+j5uSr8cm6s4eJtx5e9XKyK6hvJuwncyCPP5aW8j6GVhqXpKiW7V+C/LFZtLE93w8pXzeUf3PJdNfIxQIgAAHoBsP8TUnl6VjdOAIAgCAIBNPSx1BHw5mE3c20zL4JmIoZjUQT28uusblmp5EMiDlZUTsHaulDDxWH9NHL3lp9i62Xj+61Pa9yWvJ/F9+XgZRNN1Ku+uvIqsS2m1FsqtrZXrsrYBkdHUlWVlIIYEggzNNNOzPR4yUkpRd081bRp7zkTg+jUQCH9Q8FeJjnNdVrmImmPx/QfTKXuqAVOXa2ZeTO5tAe29hWq1bpeS8lKdLs2cH2v3Zfn5kVjpYr0t1VXY4djNaaZ+OumWpGh9j2vaVi6pp+NVpep4+ouxQXY9ZzMnKybbGy8rVbNsHENdKMdiot2Raa0pbtjud/pac5RlK6a4PJJaJasivD4inCcIdmSle11m3JttyeStn/RJ/sG8A6no2LgaTaultiY+MwuuxmzUyDlFue4rek1XGxmd3yWspLvuwoTnskevONSTkr58bafm7dxJuDpVaNONOXZsln2b6+evjv4I6jVejTRLMp9TqTLw8xrRkV24eVZT7vkcuZtorKvUjM25KMj1+Z2RdzOxYuoo9l2a5rVcOJGnsnDubqxTjLVOMmrPilnG/k1yJxrXYAbkkADkdtyf5OwA3Pr5AD+APSQi5K7e1zod0nVrnzanu07KtZnuOMK3x7rWO7WPjuNlsY7sWoenmzMzB2YtLCljZ012XmuevUoMVsWhXk5puEnra1m+Nnl0tffmeY8Df8dum49iXZmZkZ4Q79gImJjv/AALQj23Mv/qt6BvRuQJU9lTaE5K0Vb+X9iNQ2BRg71JOfKyUemb/AJ/gtXhYSVIlNaLXVWqpXWiqqIigBURVACqoAAUAAASrbvmzTpJKy0PtByIBx9R1KuiuzItsSqmpGsttsZUrrrUFnd3YhVVVBJYkAATlJt2R8ykopyk7JZtvRJbzF31T9QR8R5gNPNdMxOSYaMGQ2kkdzLsrOxVruICo45V1AbhGsuQaXC4f0Mc/eev2PONqY7vVT2fcjpzfxfbl4kLSYUogCAIAgCAIBNvTz1VZvh0+7FTl6Wz8mxGfi1DE72WYdhBFZYkuaGHasfc/os9lrQ8RhY1s9JcePj9/7LrAbUnhPYt2ocN68Pto+W+/fsv6ktG1oKuNmVrkEbnDyCKMtTsOQFTkd0LuB3KGtr39HMoquHqU/eWXFaG4wu0KGJX6cs+DykvJ6+KuuZJxEjFiaQBAEAQBAEAQBANQIBGHtR6ktG0UMuTmVtkAbjDxyt+Wx2PEGpG/SDcSO5kNTXv6uJJpYepV91ZcXoV2K2hQwy/UlnwWcn5bvF2XMoL1DdVWb4iPuwU4mlq/JcRX5NewO9dmZYABYVIDilR2q32P6rJXat7h8LGjnrLjw8Pv/Rh8ftSpi/Yt2YcL5vx+2i5kJSYUogCAIAgCAIAgCAbLqFYcWAZT6hgCD/R8pyOZ6HT/AGg6lQorp1PU6EXyVMfUdSoUD9gFpykAA/gCdUqUJaxXREuli69JWhUkv9n9Tl/FvWfreufetb/PnX3el8C6Hf6yxXzX1Hxb1n63rn3rW/z47vS+BdB6yxXzX1Hxb1n63rn3rW/z47vS+BdB6yxXzX1Hxb1n63rn3rW/z47vS+BdB6yxXzX1Hxb1n63rn3rW/wA+O70vgXQessV819R8W9Z+t65961v8+O70vgXQessV819R8W9Z+t65961v8+O70vgXQessV819R8W9Z+t65961v8+O70vgXQessV819Tiah7QdRvU13anqd6N5MmRqOpXqR+4K3ZTgg/wROyNKEdIrojoqYuvVVp1JP/Z/TU89TQqjioCgegAAA/oeU7SJzN84AgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgH/9k=",  # noqa: E501
                    },
                },
                {"type": "text", "text": "What is this a logo for?"},
            ],
        ),
    ]
    response = chat.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    num_tokens = chat.get_num_tokens_from_messages(messages)
    assert num_tokens > 0

def test_tool_use() -> None:
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        temperature=0,
    )
    tool_definition = {
        "name": "get_weather",
        "description": "Get weather report for a city",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    }
    llm_with_tools = llm.bind_tools([tool_definition])
    query = "how are you? what's the weather in san francisco, ca"
    response = llm_with_tools.invoke(query)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, list)
    assert isinstance(response.tool_calls, list)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]

    content_blocks = response.content_blocks
    assert len(content_blocks) == 2
    assert content_blocks[0]["type"] == "text"
    assert content_blocks[0]["text"]
    assert content_blocks[1]["type"] == "tool_call"
    assert content_blocks[1]["name"] == "get_weather"
    assert content_blocks[1]["args"] == tool_call["args"]

    # Test streaming
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")  # type: ignore[call-arg]
    llm_with_tools = llm.bind_tools([tool_definition])
    first = True
    chunks: list[BaseMessage | BaseMessageChunk] = []
    for chunk in llm_with_tools.stream(query):
        chunks = [*chunks, chunk]
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk  # type: ignore[assignment]
        for block in chunk.content_blocks:
            assert block["type"] in ("text", "tool_call_chunk")
    assert len(chunks) > 1
    assert isinstance(gathered.content, list)
    assert len(gathered.content) == 2
    tool_use_block = None
    for content_block in gathered.content:
        assert isinstance(content_block, dict)
        if content_block["type"] == "tool_use":
            tool_use_block = content_block
            break
    assert tool_use_block is not None
    assert tool_use_block["name"] == "get_weather"
    assert "location" in json.loads(tool_use_block["partial_json"])
    assert isinstance(gathered, AIMessageChunk)
    assert isinstance(gathered.tool_calls, list)
    assert len(gathered.tool_calls) == 1
    tool_call = gathered.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]
    assert tool_call["id"] is not None

    content_blocks = gathered.content_blocks
    assert len(content_blocks) == 2
    assert content_blocks[0]["type"] == "text"
    assert content_blocks[0]["text"]
    assert content_blocks[1]["type"] == "tool_call"
    assert content_blocks[1]["name"] == "get_weather"
    assert content_blocks[1]["args"]

    # Test passing response back to model
    stream = llm_with_tools.stream(
        [
            query,
            gathered,
            ToolMessage(content="sunny and warm", tool_call_id=tool_call["id"]),
        ],
    )
    chunks = []
    first = True
    for chunk in stream:
        chunks = [*chunks, chunk]
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk  # type: ignore[assignment]
    assert len(chunks) > 1

def test_builtin_tools_text_editor() -> None:
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")  # type: ignore[call-arg]
    tool = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
    llm_with_tools = llm.bind_tools([tool])
    response = llm_with_tools.invoke(
        "There's a syntax error in my primes.py file. Can you help me fix it?",
    )
    assert isinstance(response, AIMessage)
    assert response.tool_calls

    content_blocks = response.content_blocks
    assert len(content_blocks) == 2
    assert content_blocks[0]["type"] == "text"
    assert content_blocks[0]["text"]
    assert content_blocks[1]["type"] == "tool_call"
    assert content_blocks[1]["name"] == "str_replace_based_edit_tool"

def test_builtin_tools_computer_use() -> None:
    """Test computer use tool integration.

    Beta header should be automatically appended based on tool type.

    This test only verifies tool call generation.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
    )
    tool = {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": 1024,
        "display_height_px": 768,
        "display_number": 1,
    }
    llm_with_tools = llm.bind_tools([tool])
    response = llm_with_tools.invoke(
        "Can you take a screenshot to see what's on the screen?",
    )
    assert isinstance(response, AIMessage)
    assert response.tool_calls

    content_blocks = response.content_blocks
    assert len(content_blocks) >= 2
    assert content_blocks[0]["type"] == "text"
    assert content_blocks[0]["text"]

    # Check that we have a tool_call for computer use
    tool_call_blocks = [b for b in content_blocks if b["type"] == "tool_call"]
    assert len(tool_call_blocks) >= 1
    assert tool_call_blocks[0]["name"] == "computer"

    # Verify tool call has expected action (screenshot in this case)
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "computer"
    assert "action" in tool_call["args"]
    assert tool_call["args"]["action"] == "screenshot"

def test_disable_parallel_tool_calling() -> None:
    llm = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    llm_with_tools = llm.bind_tools([GenerateUsername], parallel_tool_calls=False)
    result = llm_with_tools.invoke(
        "Use the GenerateUsername tool to generate user names for:\n\n"
        "Sally with green hair\n"
        "Bob with blue hair",
    )
    assert isinstance(result, AIMessage)
    assert len(result.tool_calls) == 1

def test_anthropic_with_empty_text_block() -> None:
    """Anthropic SDK can return an empty text block."""

    @tool
    def type_letter(letter: str) -> str:
        """Type the given letter."""
        return "OK"

    model = ChatAnthropic(model=MODEL_NAME, temperature=0).bind_tools(  # type: ignore[call-arg]
        [type_letter],
    )

    messages = [
        SystemMessage(
            content="Repeat the given string using the provided tools. Do not write "
            "anything else or provide any explanations. For example, "
            "if the string is 'abc', you must print the "
            "letters 'a', 'b', and 'c' one at a time and in that order. ",
        ),
        HumanMessage(content="dog"),
        AIMessage(
            content=[
                {"text": "", "type": "text"},
                {
                    "id": "toolu_01V6d6W32QGGSmQm4BT98EKk",
                    "input": {"letter": "d"},
                    "name": "type_letter",
                    "type": "tool_use",
                },
            ],
            tool_calls=[
                {
                    "name": "type_letter",
                    "args": {"letter": "d"},
                    "id": "toolu_01V6d6W32QGGSmQm4BT98EKk",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(content="OK", tool_call_id="toolu_01V6d6W32QGGSmQm4BT98EKk"),
    ]

    model.invoke(messages)

def test_with_structured_output() -> None:
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
    )

    structured_llm = llm.with_structured_output(
        {
            "name": "get_weather",
            "description": "Get weather report for a city",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    )
    response = structured_llm.invoke("what's the weather in san francisco, ca")
    assert isinstance(response, dict)
    assert response["location"]

def test_response_format(schema: dict | type) -> None:
    model = ChatAnthropic(
        model="claude-sonnet-4-5",  # type: ignore[call-arg]
    )
    query = "Chester (a.k.a. Chet) is 100 years old."

    response = model.invoke(query, response_format=schema)
    parsed = json.loads(response.text)
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema.model_validate(parsed)
    else:
        assert isinstance(parsed, dict)
        assert parsed["name"]
        assert parsed["age"]

def test_response_format_in_agent() -> None:
    class Weather(BaseModel):
        temperature: float
        units: str

    # no tools
    agent = create_agent(
        "anthropic:claude-sonnet-4-5", response_format=ProviderStrategy(Weather)
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "75 degrees F."}]})
    assert len(result["messages"]) == 2
    parsed = json.loads(result["messages"][-1].text)
    assert Weather(**parsed) == result["structured_response"]

    # with tools
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "75 degrees Fahrenheit."

    agent = create_agent(
        "anthropic:claude-sonnet-4-5",
        tools=[get_weather],
        response_format=ProviderStrategy(Weather),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in SF?"}]},
    )
    assert len(result["messages"]) == 4
    assert result["messages"][1].tool_calls
    parsed = json.loads(result["messages"][-1].text)
    assert Weather(**parsed) == result["structured_response"]

def test_strict_tool_use() -> None:
    model = ChatAnthropic(
        model="claude-sonnet-4-5",  # type: ignore[call-arg]
    )

    def get_weather(location: str, unit: Literal["C", "F"]) -> str:
        """Get the weather at a location."""
        return "75 degrees Fahrenheit."

    model_with_tools = model.bind_tools([get_weather], strict=True)

    response = model_with_tools.invoke("What's the weather in Boston, in Celsius?")
    assert response.tool_calls

def test_anthropic_bind_tools_tool_choice(tool_choice: str) -> None:
    chat_model = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
    )
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice=tool_choice)
    response = chat_model_with_tools.invoke("what's the weather in ny and la")
    assert isinstance(response, AIMessage)

def test_pdf_document_input() -> None:
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    data = b64encode(requests.get(url, timeout=10).content).decode()

    result = ChatAnthropic(model=MODEL_NAME).invoke(  # type: ignore[call-arg]
        [
            HumanMessage(
                [
                    "summarize this document",
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "data": data,
                            "media_type": "application/pdf",
                        },
                    },
                ],
            ),
        ],
    )
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    assert len(result.content) > 0

def test_agent_loop(output_version: Literal["v0", "v1"]) -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm = ChatAnthropic(model=MODEL_NAME, output_version=output_version)  # type: ignore[call-arg]
    llm_with_tools = llm.bind_tools([get_weather])
    input_message = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)
    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    response = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    assert isinstance(response, AIMessage)

def test_agent_loop_streaming(output_version: Literal["v0", "v1"]) -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm = ChatAnthropic(
        model=MODEL_NAME,
        streaming=True,
        output_version=output_version,  # type: ignore[call-arg]
    )
    llm_with_tools = llm.bind_tools([get_weather])
    input_message = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)

    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    response = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    assert isinstance(response, AIMessage)

def test_citations(output_version: Literal["v0", "v1"]) -> None:
    llm = ChatAnthropic(model=MODEL_NAME, output_version=output_version)  # type: ignore[call-arg]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "content",
                        "content": [
                            {"type": "text", "text": "The grass is green"},
                            {"type": "text", "text": "The sky is blue"},
                        ],
                    },
                    "citations": {"enabled": True},
                },
                {"type": "text", "text": "What color is the grass and sky?"},
            ],
        },
    ]
    response = llm.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, list)
    if output_version == "v1":
        assert any("annotations" in block for block in response.content)
    else:
        assert any("citations" in block for block in response.content)

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream(messages):
        full = cast("BaseMessageChunk", chunk) if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    assert not any("citation" in block for block in full.content)
    if output_version == "v1":
        assert any("annotations" in block for block in full.content)
    else:
        assert any("citations" in block for block in full.content)

    # Test pass back in
    next_message = {
        "role": "user",
        "content": "Can you comment on the citations you just made?",
    }
    _ = llm.invoke([*messages, full, next_message])

def test_thinking() -> None:
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        max_tokens=5_000,  # type: ignore[call-arg]
        thinking={"type": "enabled", "budget_tokens": 2_000},
    )

    input_message = {"role": "user", "content": "Hello"}
    response = llm.invoke([input_message])
    assert any("thinking" in block for block in response.content)
    for block in response.content:
        assert isinstance(block, dict)
        if block["type"] == "thinking":
            assert set(block.keys()) == {"type", "thinking", "signature"}
            assert block["thinking"]
            assert isinstance(block["thinking"], str)
            assert block["signature"]
            assert isinstance(block["signature"], str)

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        full = cast("BaseMessageChunk", chunk) if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    assert any("thinking" in block for block in full.content)
    for block in full.content:
        assert isinstance(block, dict)
        if block["type"] == "thinking":
            assert set(block.keys()) == {"type", "thinking", "signature", "index"}
            assert block["thinking"]
            assert isinstance(block["thinking"], str)
            assert block["signature"]
            assert isinstance(block["signature"], str)

    # Test pass back in
    next_message = {"role": "user", "content": "How are you?"}
    _ = llm.invoke([input_message, full, next_message])

def test_thinking_v1() -> None:
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        max_tokens=5_000,  # type: ignore[call-arg]
        thinking={"type": "enabled", "budget_tokens": 2_000},
        output_version="v1",
    )

    input_message = {"role": "user", "content": "Hello"}
    response = llm.invoke([input_message])
    assert any("reasoning" in block for block in response.content)
    for block in response.content:
        assert isinstance(block, dict)
        if block["type"] == "reasoning":
            assert set(block.keys()) == {"type", "reasoning", "extras"}
            assert block["reasoning"]
            assert isinstance(block["reasoning"], str)
            signature = block["extras"]["signature"]
            assert signature
            assert isinstance(signature, str)

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        full = cast(BaseMessageChunk, chunk) if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    assert any("reasoning" in block for block in full.content)
    for block in full.content:
        assert isinstance(block, dict)
        if block["type"] == "reasoning":
            assert set(block.keys()) == {"type", "reasoning", "extras", "index"}
            assert block["reasoning"]
            assert isinstance(block["reasoning"], str)
            signature = block["extras"]["signature"]
            assert signature
            assert isinstance(signature, str)

    # Test pass back in
    next_message = {"role": "user", "content": "How are you?"}
    _ = llm.invoke([input_message, full, next_message])

def test_redacted_thinking(output_version: Literal["v0", "v1"]) -> None:
    llm = ChatAnthropic(
        # It appears that Sonnet 4.5 either: isn't returning redacted thinking blocks,
        # or the magic string is broken? Retry later once 3-7 finally removed
        model="claude-3-7-sonnet-latest",  # type: ignore[call-arg]
        max_tokens=5_000,  # type: ignore[call-arg]
        thinking={"type": "enabled", "budget_tokens": 2_000},
        output_version=output_version,
    )
    query = "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"  # noqa: E501
    input_message = {"role": "user", "content": query}

    response = llm.invoke([input_message])
    value = None
    for block in response.content:
        assert isinstance(block, dict)
        if block["type"] == "redacted_thinking":
            value = block
        elif (
            block["type"] == "non_standard"
            and block["value"]["type"] == "redacted_thinking"
        ):
            value = block["value"]
        else:
            pass
        if value:
            assert set(value.keys()) == {"type", "data"}
            assert value["data"]
            assert isinstance(value["data"], str)
    assert value is not None

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        full = cast("BaseMessageChunk", chunk) if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    value = None
    for block in full.content:
        assert isinstance(block, dict)
        if block["type"] == "redacted_thinking":
            value = block
            assert set(value.keys()) == {"type", "data", "index"}
            assert "index" in block
        elif (
            block["type"] == "non_standard"
            and block["value"]["type"] == "redacted_thinking"
        ):
            value = block["value"]
            assert isinstance(value, dict)
            assert set(value.keys()) == {"type", "data"}
            assert "index" in block
        else:
            pass
        if value:
            assert value["data"]
            assert isinstance(value["data"], str)
    assert value is not None

    # Test pass back in
    next_message = {"role": "user", "content": "What?"}
    _ = llm.invoke([input_message, full, next_message])

def test_structured_output_thinking_enabled() -> None:
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        max_tokens=5_000,  # type: ignore[call-arg]
        thinking={"type": "enabled", "budget_tokens": 2_000},
    )
    with pytest.warns(match="structured output"):
        structured_llm = llm.with_structured_output(GenerateUsername)
    query = "Generate a username for Sally with green hair"
    response = structured_llm.invoke(query)
    assert isinstance(response, GenerateUsername)

    with pytest.raises(OutputParserException):
        structured_llm.invoke("Hello")

    # Test streaming
    for chunk in structured_llm.stream(query):
        assert isinstance(chunk, GenerateUsername)

def test_structured_output_thinking_force_tool_use() -> None:
    # Structured output currently relies on forced tool use, which is not supported
    # when `thinking` is enabled. When this test fails, it means that the feature
    # is supported and the workarounds in `with_structured_output` should be removed.
    client = anthropic.Anthropic()
    with pytest.raises(anthropic.BadRequestError):
        _ = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=5_000,
            thinking={"type": "enabled", "budget_tokens": 2_000},
            tool_choice={"type": "tool", "name": "get_weather"},
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get the weather at a location.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in San Francisco?",
                }
            ],
        )

def test_effort_parameter() -> None:
    """Test that effort parameter can be passed without errors.

    Only Opus 4.5 supports currently.
    """
    llm = ChatAnthropic(
        model="claude-opus-4-5-20251101",
        effort="medium",
        max_tokens=100,
    )

    result = llm.invoke("Say hello in one sentence")

    # Verify we got a response
    assert isinstance(result.content, str)
    assert len(result.content) > 0

    # Verify response metadata is present
    assert "model_name" in result.response_metadata
    assert result.usage_metadata is not None
    assert result.usage_metadata["input_tokens"] > 0
    assert result.usage_metadata["output_tokens"] > 0

def test_image_tool_calling() -> None:
    """Test tool calling with image inputs."""

    class color_picker(BaseModel):  # noqa: N801
        """Input your fav color and get a random fact about it."""

        fav_color: str

    human_content: list[dict] = [
        {
            "type": "text",
            "text": "what's your favorite color in this image",
        },
    ]
    image_url = "https://raw.githubusercontent.com/langchain-ai/docs/4d11d08b6b0e210bd456943f7a22febbd168b543/src/images/agentic-rag-output.png"
    image_data = b64encode(httpx.get(image_url, timeout=10.0).content).decode("utf-8")
    human_content.append(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_data,
            },
        },
    )
    messages = [
        SystemMessage("you're a good assistant"),
        HumanMessage(human_content),  # type: ignore[arg-type]
        AIMessage(
            [
                {"type": "text", "text": "Hmm let me think about that"},
                {
                    "type": "tool_use",
                    "input": {"fav_color": "purple"},
                    "id": "foo",
                    "name": "color_picker",
                },
            ],
        ),
        HumanMessage(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "foo",
                    "content": [
                        {
                            "type": "text",
                            "text": "purple is a great pick! that's my sister's favorite color",  # noqa: E501
                        },
                    ],
                    "is_error": False,
                },
                {"type": "text", "text": "what's my sister's favorite color"},
            ],
        ),
    ]
    llm = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    _ = llm.bind_tools([color_picker]).invoke(messages)

def test_web_search(output_version: Literal["v0", "v1"]) -> None:
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        max_tokens=1024,
        output_version=output_version,
    )

    tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
    llm_with_tools = llm.bind_tools([tool])

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "How do I update a web app to TypeScript 5.5?",
            },
        ],
    }
    response = llm_with_tools.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)
    block_types = {block["type"] for block in response.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "server_tool_use", "web_search_tool_result"}
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "server_tool_use", "web_search_tool_result"}
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test we can pass back in
    next_message = {
        "role": "user",
        "content": "Please repeat the last search, but focus on sources from 2024.",
    }
    _ = llm_with_tools.invoke(
        [input_message, full, next_message],
    )

def test_web_fetch() -> None:
    """Note: this is a beta feature.

    TODO: Update to remove beta once it's generally available.
    """
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        max_tokens=1024,
        betas=["web-fetch-2025-09-10"],
    )
    tool = {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1}
    llm_with_tools = llm.bind_tools([tool])

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Fetch the content at https://docs.langchain.com and analyze",
            },
        ],
    }
    response = llm_with_tools.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)
    block_types = {
        block["type"] for block in response.content if isinstance(block, dict)
    }

    # A successful fetch call should include:
    # 1. text response from the model (e.g. "I'll fetch that for you")
    # 2. server_tool_use block indicating the tool was called (using tool "web_fetch")
    # 3. web_fetch_tool_result block with the results of said fetch
    assert block_types == {"text", "server_tool_use", "web_fetch_tool_result"}

    # Verify web fetch result structure
    web_fetch_results = [
        block
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "web_fetch_tool_result"
    ]
    assert len(web_fetch_results) == 1  # Since max_uses=1
    fetch_result = web_fetch_results[0]
    assert "content" in fetch_result
    assert "url" in fetch_result["content"]
    assert "retrieved_at" in fetch_result["content"]

    # Fetch with citations enabled
    tool_with_citations = tool.copy()
    tool_with_citations["citations"] = {"enabled": True}
    llm_with_citations = llm.bind_tools([tool_with_citations])

    citation_message = {
        "role": "user",
        "content": (
            "Fetch https://docs.langchain.com and provide specific quotes with "
            "citations"
        ),
    }
    citation_response = llm_with_citations.invoke([citation_message])

    citation_results = [
        block
        for block in citation_response.content
        if isinstance(block, dict) and block.get("type") == "web_fetch_tool_result"
    ]
    assert len(citation_results) == 1  # Since max_uses=1
    citation_result = citation_results[0]
    assert citation_result["content"]["content"]["citations"]["enabled"]
    text_blocks = [
        block
        for block in citation_response.content
        if isinstance(block, dict) and block.get("type") == "text"
    ]

    # Check that the response contains actual citations in the content
    has_citations = False
    for block in text_blocks:
        citations = block.get("citations", [])
        for citation in citations:
            if citation.get("type") and citation.get("start_char_index"):
                has_citations = True
                break
    assert has_citations, (
        "Expected inline citation tags in response when citations are enabled for "
        "web fetch"
    )

    # Max content tokens param
    tool_with_limit = tool.copy()
    tool_with_limit["max_content_tokens"] = 1000
    llm_with_limit = llm.bind_tools([tool_with_limit])

    limit_response = llm_with_limit.invoke([input_message])
    # Response should still work even with content limits
    assert any(
        block["type"] == "web_fetch_tool_result"
        for block in limit_response.content
        if isinstance(block, dict)
    )

    # Domains filtering (note: only one can be set at a time)
    tool_with_allowed_domains = tool.copy()
    tool_with_allowed_domains["allowed_domains"] = ["docs.langchain.com"]
    llm_with_allowed = llm.bind_tools([tool_with_allowed_domains])

    allowed_response = llm_with_allowed.invoke([input_message])
    assert any(
        block["type"] == "web_fetch_tool_result"
        for block in allowed_response.content
        if isinstance(block, dict)
    )

    # Test that a disallowed domain doesn't work
    tool_with_disallowed_domains = tool.copy()
    tool_with_disallowed_domains["allowed_domains"] = [
        "example.com"
    ]  # Not docs.langchain.com
    llm_with_disallowed = llm.bind_tools([tool_with_disallowed_domains])

    disallowed_response = llm_with_disallowed.invoke([input_message])

    # We should get an error result since the domain (docs.langchain.com) is not allowed
    disallowed_results = [
        block
        for block in disallowed_response.content
        if isinstance(block, dict) and block.get("type") == "web_fetch_tool_result"
    ]
    if disallowed_results:
        disallowed_result = disallowed_results[0]
        if disallowed_result.get("content", {}).get("type") == "web_fetch_tool_error":
            assert disallowed_result["content"]["error_code"] in [
                "invalid_url",
                "fetch_failed",
            ]

    # Blocked domains filtering
    tool_with_blocked_domains = tool.copy()
    tool_with_blocked_domains["blocked_domains"] = ["example.com"]
    llm_with_blocked = llm.bind_tools([tool_with_blocked_domains])

    blocked_response = llm_with_blocked.invoke([input_message])
    assert any(
        block["type"] == "web_fetch_tool_result"
        for block in blocked_response.content
        if isinstance(block, dict)
    )

    # Test fetching from a blocked domain fails
    blocked_domain_message = {
        "role": "user",
        "content": "Fetch https://example.com and analyze",
    }
    tool_with_blocked_example = tool.copy()
    tool_with_blocked_example["blocked_domains"] = ["example.com"]
    llm_with_blocked_example = llm.bind_tools([tool_with_blocked_example])

    blocked_domain_response = llm_with_blocked_example.invoke([blocked_domain_message])

    # Should get an error when trying to access a blocked domain
    blocked_domain_results = [
        block
        for block in blocked_domain_response.content
        if isinstance(block, dict) and block.get("type") == "web_fetch_tool_result"
    ]
    if blocked_domain_results:
        blocked_result = blocked_domain_results[0]
        if blocked_result.get("content", {}).get("type") == "web_fetch_tool_error":
            assert blocked_result["content"]["error_code"] in [
                "invalid_url",
                "fetch_failed",
            ]

    # Max uses parameter - test exceeding the limit
    multi_fetch_message = {
        "role": "user",
        "content": (
            "Fetch https://docs.langchain.com and then try to fetch "
            "https://langchain.com"
        ),
    }
    max_uses_response = llm_with_tools.invoke([multi_fetch_message])

    # Should contain at least one fetch result and potentially an error for the second
    fetch_results = [
        block
        for block in max_uses_response.content
        if isinstance(block, dict) and block.get("type") == "web_fetch_tool_result"
    ]  # type: ignore[index]
    assert len(fetch_results) >= 1
    error_results = [
        r
        for r in fetch_results
        if r.get("content", {}).get("type") == "web_fetch_tool_error"
    ]
    if error_results:
        assert any(
            r["content"]["error_code"] == "max_uses_exceeded" for r in error_results
        )

    # Streaming
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    block_types = {block["type"] for block in full.content if isinstance(block, dict)}
    assert block_types == {"text", "server_tool_use", "web_fetch_tool_result"}

    # Test that URLs from context can be used in follow-up
    next_message = {
        "role": "user",
        "content": "What does the site you just fetched say about models?",
    }
    follow_up_response = llm_with_tools.invoke(
        [input_message, full, next_message],
    )
    # Should work without issues since URL was already in context
    assert isinstance(follow_up_response.content, (list, str))

    # Error handling - test with an invalid URL format
    error_message = {
        "role": "user",
        "content": "Try to fetch this invalid URL: not-a-valid-url",
    }
    error_response = llm_with_tools.invoke([error_message])

    # Should handle the error gracefully
    assert isinstance(error_response.content, (list, str))

    # PDF document fetching
    pdf_message = {
        "role": "user",
        "content": (
            "Fetch this PDF: "
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf "
            "and summarize its content",
        ),
    }
    pdf_response = llm_with_tools.invoke([pdf_message])

    assert any(
        block["type"] == "web_fetch_tool_result"
        for block in pdf_response.content
        if isinstance(block, dict)
    )

    # Verify PDF content structure (should have base64 data for PDFs)
    pdf_results = [
        block
        for block in pdf_response.content
        if isinstance(block, dict) and block.get("type") == "web_fetch_tool_result"
    ]
    if pdf_results:
        pdf_result = pdf_results[0]
        content = pdf_result.get("content", {})
        if content.get("content", {}).get("source", {}).get("type") == "base64":
            assert content["content"]["source"]["media_type"] == "application/pdf"
            assert "data" in content["content"]["source"]

def test_web_fetch_v1(output_version: Literal["v0", "v1"]) -> None:
    """Test that http calls are unchanged between v0 and v1."""
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["web-fetch-2025-09-10"],
        output_version=output_version,
    )

    if output_version == "v0":
        call_key = "server_tool_use"
        result_key = "web_fetch_tool_result"
    else:
        # v1
        call_key = "server_tool_call"
        result_key = "server_tool_result"

    tool = {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 1,
        "citations": {"enabled": True},
    }
    llm_with_tools = llm.bind_tools([tool])

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Fetch the content at https://docs.langchain.com and analyze",
            },
        ],
    }
    response = llm_with_tools.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)
    block_types = {block["type"] for block in response.content}  # type: ignore[index]
    assert block_types == {"text", call_key, result_key}

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    assert block_types == {"text", call_key, result_key}

    # Test we can pass back in
    next_message = {
        "role": "user",
        "content": "What does the site you just fetched say about models?",
    }
    _ = llm_with_tools.invoke(
        [input_message, full, next_message],
    )

def test_code_execution_old(output_version: Literal["v0", "v1"]) -> None:
    """Note: this tests the `code_execution_20250522` tool, which is now legacy.

    See the `test_code_execution` test below to test the current
    `code_execution_20250825` tool.

    Migration guide: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#upgrade-to-latest-tool-version
    """
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["code-execution-2025-05-22"],
        output_version=output_version,
    )

    tool = {"type": "code_execution_20250522", "name": "code_execution"}
    llm_with_tools = llm.bind_tools([tool])

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Calculate the mean and standard deviation of "
                    "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
                ),
            },
        ],
    }
    response = llm_with_tools.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)
    block_types = {block["type"] for block in response.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "server_tool_use", "code_execution_tool_result"}
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "server_tool_use", "code_execution_tool_result"}
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test we can pass back in
    next_message = {
        "role": "user",
        "content": "Please add more comments to the code.",
    }
    _ = llm_with_tools.invoke(
        [input_message, full, next_message],
    )

def test_code_execution(output_version: Literal["v0", "v1"]) -> None:
    """Note: this is a beta feature.

    TODO: Update to remove beta once generally available.
    """
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["code-execution-2025-08-25"],
        output_version=output_version,
    )

    tool = {"type": "code_execution_20250825", "name": "code_execution"}
    llm_with_tools = llm.bind_tools([tool])

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Calculate the mean and standard deviation of "
                    "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
                ),
            },
        ],
    }
    response = llm_with_tools.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)
    block_types = {block["type"] for block in response.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {
            "text",
            "server_tool_use",
            "bash_code_execution_tool_result",
        }
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {
            "text",
            "server_tool_use",
            "bash_code_execution_tool_result",
        }
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test we can pass back in
    next_message = {
        "role": "user",
        "content": "Please add more comments to the code.",
    }
    _ = llm_with_tools.invoke(
        [input_message, full, next_message],
    )

def test_remote_mcp(output_version: Literal["v0", "v1"]) -> None:
    """Note: this is a beta feature.

    TODO: Update to remove beta once generally available.
    """
    mcp_servers = [
        {
            "type": "url",
            "url": "https://mcp.deepwiki.com/mcp",
            "name": "deepwiki",
            "authorization_token": "PLACEHOLDER",
        },
    ]

    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        mcp_servers=mcp_servers,
        output_version=output_version,
    ).bind_tools([{"type": "mcp_toolset", "mcp_server_name": "deepwiki"}])

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "What transport protocols does the 2025-03-26 version of the MCP "
                    "spec (modelcontextprotocol/modelcontextprotocol) support?"
                ),
            },
        ],
    }
    response = llm.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)
    block_types = {block["type"] for block in response.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "mcp_tool_use", "mcp_tool_result"}
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    assert all(isinstance(block, dict) for block in full.content)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "mcp_tool_use", "mcp_tool_result"}
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test we can pass back in
    next_message = {
        "role": "user",
        "content": "Please query the same tool again, but add 'please' to your query.",
    }
    _ = llm.invoke(
        [input_message, full, next_message],
    )

def test_files_api_image(block_format: str) -> None:
    """Note: this is a beta feature.

    TODO: Update to remove beta once generally available.
    """
    image_file_id = os.getenv("ANTHROPIC_FILES_API_IMAGE_ID")
    if not image_file_id:
        pytest.skip()
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["files-api-2025-04-14"],
    )
    if block_format == "anthropic":
        block = {
            "type": "image",
            "source": {
                "type": "file",
                "file_id": image_file_id,
            },
        }
    else:
        # standard block format
        block = {
            "type": "image",
            "file_id": image_file_id,
        }
    input_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            block,
        ],
    }
    _ = llm.invoke([input_message])

def test_files_api_pdf(block_format: str) -> None:
    """Note: this is a beta feature.

    TODO: Update to remove beta once generally available.
    """
    pdf_file_id = os.getenv("ANTHROPIC_FILES_API_PDF_ID")
    if not pdf_file_id:
        pytest.skip()
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["files-api-2025-04-14"],
    )
    if block_format == "anthropic":
        block = {"type": "document", "source": {"type": "file", "file_id": pdf_file_id}}
    else:
        # standard block format
        block = {
            "type": "file",
            "file_id": pdf_file_id,
        }
    input_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this document."},
            block,
        ],
    }
    _ = llm.invoke([input_message])

def test_search_result_tool_message() -> None:
    """Test that we can pass a search result tool message to the model."""
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
    )

    @tool
    def retrieval_tool(query: str) -> list[dict]:
        """Retrieve information from a knowledge base."""
        return [
            {
                "type": "search_result",
                "title": "Leave policy",
                "source": "HR Leave Policy 2025",
                "citations": {"enabled": True},
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "To request vacation days, submit a leave request form "
                            "through the HR portal. Approval will be sent by email."
                        ),
                    },
                ],
            },
        ]

    tool_call = {
        "type": "tool_call",
        "name": "retrieval_tool",
        "args": {"query": "vacation days request process"},
        "id": "toolu_abc123",
    }

    tool_message = retrieval_tool.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    assert isinstance(tool_message.content, list)

    messages = [
        HumanMessage("How do I request vacation days?"),
        AIMessage(
            [{"type": "text", "text": "Let me look that up for you."}],
            tool_calls=[tool_call],
        ),
        tool_message,
    ]

    result = llm.invoke(messages)
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, list)
    assert any("citations" in block for block in result.content)

    assert (
        _convert_from_v1_to_anthropic(result.content_blocks, [], "anthropic")
        == result.content
    )

def test_search_result_top_level() -> None:
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
    )
    input_message = HumanMessage(
        [
            {
                "type": "search_result",
                "title": "Leave policy",
                "source": "HR Leave Policy 2025 - page 1",
                "citations": {"enabled": True},
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "To request vacation days, submit a leave request form "
                            "through the HR portal. Approval will be sent by email."
                        ),
                    },
                ],
            },
            {
                "type": "search_result",
                "title": "Leave policy",
                "source": "HR Leave Policy 2025 - page 2",
                "citations": {"enabled": True},
                "content": [
                    {
                        "type": "text",
                        "text": "Managers have 3 days to approve a request.",
                    },
                ],
            },
            {
                "type": "text",
                "text": "How do I request vacation days?",
            },
        ],
    )
    result = llm.invoke([input_message])
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, list)
    assert any("citations" in block for block in result.content)

    assert (
        _convert_from_v1_to_anthropic(result.content_blocks, [], "anthropic")
        == result.content
    )

def test_memory_tool() -> None:
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        betas=["context-management-2025-06-27"],
    )
    llm_with_tools = llm.bind_tools([{"type": "memory_20250818", "name": "memory"}])
    response = llm_with_tools.invoke("What are my interests?")
    assert isinstance(response, AIMessage)
    assert response.tool_calls
    assert response.tool_calls[0]["name"] == "memory"

def test_context_management() -> None:
    # TODO: update example to trigger action
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        betas=["context-management-2025-06-27"],
        context_management={
            "edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "input_tokens", "value": 10},
                    "clear_at_least": {"type": "input_tokens", "value": 5},
                }
            ]
        },
        max_tokens=1024,  # type: ignore[call-arg]
    )
    llm_with_tools = llm.bind_tools(
        [{"type": "web_search_20250305", "name": "web_search"}]
    )
    input_message = {"role": "user", "content": "Search for recent developments in AI"}
    response = llm_with_tools.invoke([input_message])
    assert response.response_metadata.get("context_management")

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.response_metadata.get("context_management")

def test_tool_search(output_version: str) -> None:
    """Test tool search with LangChain tools using extras parameter."""

    @tool(parse_docstring=True, extras={"defer_loading": True})
    def get_weather(location: str, unit: str = "fahrenheit") -> str:
        """Get the current weather for a location.

        Args:
            location: City name
            unit: Temperature unit (celsius or fahrenheit)
        """
        return f"The weather in {location} is sunny and 72°{unit[0].upper()}"

    @tool(parse_docstring=True, extras={"defer_loading": True})
    def search_files(query: str) -> str:
        """Search through files in the workspace.

        Args:
            query: Search query
        """
        return f"Found 3 files matching '{query}'"

    model = ChatAnthropic(
        model="claude-opus-4-5-20251101", output_version=output_version
    )

    agent = create_agent(  # type: ignore[var-annotated]
        model,
        tools=[
            {
                "type": "tool_search_tool_regex_20251119",
                "name": "tool_search_tool_regex",
            },
            get_weather,
            search_files,
        ],
    )

    # Test with actual API call
    input_message = {
        "role": "user",
        "content": "What's the weather in San Francisco? Find and use a tool.",
    }
    result = agent.invoke({"messages": [input_message]})
    first_response = result["messages"][1]
    content_types = [block["type"] for block in first_response.content]
    if output_version == "v0":
        assert content_types == [
            "text",
            "server_tool_use",
            "tool_search_tool_result",
            "text",
            "tool_use",
        ]
    else:
        # v1
        assert content_types == [
            "text",
            "server_tool_call",
            "server_tool_result",
            "text",
            "tool_call",
        ]

    answer = result["messages"][-1]
    assert not answer.tool_calls
    assert answer.text

def test_programmatic_tool_use(output_version: str) -> None:
    """Test programmatic tool use.

    Implicitly checks that `allowed_callers` in tool extras works.
    """

    @tool(extras={"allowed_callers": ["code_execution_20250825"]})
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "It's sunny."

    tools: list = [
        {"type": "code_execution_20250825", "name": "code_execution"},
        get_weather,
    ]

    model = ChatAnthropic(
        model="claude-sonnet-4-5",
        betas=["advanced-tool-use-2025-11-20"],
        reuse_last_container=True,
        output_version=output_version,
    )

    agent = create_agent(model, tools=tools)  # type: ignore[var-annotated]

    input_query = {
        "role": "user",
        "content": "What's the weather in Boston?",
    }

    result = agent.invoke({"messages": [input_query]})
    assert len(result["messages"]) == 4
    tool_call_message = result["messages"][1]
    response_message = result["messages"][-1]

    if output_version == "v0":
        server_tool_use_block = next(
            block
            for block in tool_call_message.content
            if block["type"] == "server_tool_use"
        )
        assert server_tool_use_block

        tool_use_block = next(
            block for block in tool_call_message.content if block["type"] == "tool_use"
        )
        assert "caller" in tool_use_block

        code_execution_result = next(
            block
            for block in response_message.content
            if block["type"] == "code_execution_tool_result"
        )
        assert code_execution_result["content"]["return_code"] == 0
    else:
        server_tool_call_block = next(
            block
            for block in tool_call_message.content
            if block["type"] == "server_tool_call"
        )
        assert server_tool_call_block

        tool_call_block = next(
            block for block in tool_call_message.content if block["type"] == "tool_call"
        )
        assert "caller" in tool_call_block["extras"]

        server_tool_result = next(
            block
            for block in response_message.content
            if block["type"] == "server_tool_result"
        )
        assert server_tool_result["output"]["return_code"] == 0

def test_programmatic_tool_use_streaming(output_version: str) -> None:
    @tool(extras={"allowed_callers": ["code_execution_20250825"]})
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "It's sunny."

    tools: list = [
        {"type": "code_execution_20250825", "name": "code_execution"},
        get_weather,
    ]

    model = ChatAnthropic(
        model="claude-sonnet-4-5",
        betas=["advanced-tool-use-2025-11-20"],
        reuse_last_container=True,
        streaming=True,
        output_version=output_version,
    )

    agent = create_agent(model, tools=tools)  # type: ignore[var-annotated]

    input_query = {
        "role": "user",
        "content": "What's the weather in Boston?",
    }

    result = agent.invoke({"messages": [input_query]})
    assert len(result["messages"]) == 4
    tool_call_message = result["messages"][1]
    response_message = result["messages"][-1]

    if output_version == "v0":
        server_tool_use_block = next(
            block
            for block in tool_call_message.content
            if block["type"] == "server_tool_use"
        )
        assert server_tool_use_block

        tool_use_block = next(
            block for block in tool_call_message.content if block["type"] == "tool_use"
        )
        assert "caller" in tool_use_block

        code_execution_result = next(
            block
            for block in response_message.content
            if block["type"] == "code_execution_tool_result"
        )
        assert code_execution_result["content"]["return_code"] == 0
    else:
        server_tool_call_block = next(
            block
            for block in tool_call_message.content
            if block["type"] == "server_tool_call"
        )
        assert server_tool_call_block

        tool_call_block = next(
            block for block in tool_call_message.content if block["type"] == "tool_call"
        )
        assert "caller" in tool_call_block["extras"]

        server_tool_result = next(
            block
            for block in response_message.content
            if block["type"] == "server_tool_result"
        )
        assert server_tool_result["output"]["return_code"] == 0

def test_async_shared_client() -> None:
    llm = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    _ = asyncio.run(llm.ainvoke("Hello"))
    _ = asyncio.run(llm.ainvoke("Hello"))

def test_fine_grained_tool_streaming() -> None:
    """Test fine-grained tool streaming reduces latency for tool parameter streaming.

    Fine-grained tool streaming enables Claude to stream tool parameter values.

    https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming
    """
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        temperature=0,
        betas=["fine-grained-tool-streaming-2025-05-14"],
    )

    # Define a tool that requires a longer text parameter
    tool_definition = {
        "name": "write_document",
        "description": "Write a document with the given content",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title"},
                "content": {
                    "type": "string",
                    "description": "The full document content",
                },
            },
            "required": ["title", "content"],
        },
    }

    llm_with_tools = llm.bind_tools([tool_definition])
    query = (
        "Write a document about the benefits of streaming APIs. "
        "Include at least 3 paragraphs."
    )

    # Test streaming with fine-grained tool streaming
    first = True
    chunks: list[BaseMessage | BaseMessageChunk] = []
    tool_call_chunks = []

    for chunk in llm_with_tools.stream(query):
        chunks.append(chunk)
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk  # type: ignore[assignment]

        # Collect tool call chunks
        tool_call_chunks.extend(
            [
                block
                for block in chunk.content_blocks
                if block["type"] == "tool_call_chunk"
            ]
        )

    # Verify we got chunks
    assert len(chunks) > 1

    # Verify final message has tool call
    assert isinstance(gathered, AIMessageChunk)
    assert isinstance(gathered.tool_calls, list)
    assert len(gathered.tool_calls) >= 1

    # Find the write_document tool call
    write_doc_call = None
    for tool_call in gathered.tool_calls:
        if tool_call["name"] == "write_document":
            write_doc_call = tool_call
            break

    assert write_doc_call is not None, "write_document tool call not found"
    assert isinstance(write_doc_call["args"], dict)
    assert "title" in write_doc_call["args"]
    assert "content" in write_doc_call["args"]
    assert (
        len(write_doc_call["args"]["content"]) > 100
    )  # Should have substantial content

    # Verify tool_call_chunks were received
    # With fine-grained streaming, we should get tool call chunks
    assert len(tool_call_chunks) > 0

    # Verify content_blocks in final message
    content_blocks = gathered.content_blocks
    assert len(content_blocks) >= 1

    # Should have at least one tool_call block
    tool_call_blocks = [b for b in content_blocks if b["type"] == "tool_call"]
    assert len(tool_call_blocks) >= 1

    write_doc_block = None
    for block in tool_call_blocks:
        if block["name"] == "write_document":
            write_doc_block = block
            break

    assert write_doc_block is not None
    assert write_doc_block["name"] == "write_document"
    assert "args" in write_doc_block

def test_compaction() -> None:
    """Test the compaction beta feature."""
    llm = ChatAnthropic(
        model="claude-opus-4-6",  # type: ignore[call-arg]
        betas=["compact-2026-01-12"],
        max_tokens=4096,
        context_management={
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": 50000},
                    "pause_after_compaction": True,
                }
            ]
        },
    )

    input_message = {
        "role": "user",
        "content": f"Generate a one-sentence summary of this:\n\n{'a' * 100000}",
    }
    messages: list = [input_message]

    first_response = llm.invoke(messages)
    messages.append(first_response)

    second_message = {
        "role": "user",
        "content": f"Generate a one-sentence summary of this:\n\n{'b' * 100000}",
    }
    messages.append(second_message)

    second_response = llm.invoke(messages)
    messages.append(second_response)

    content_blocks = second_response.content_blocks
    compaction_block = next(
        (block for block in content_blocks if block["type"] == "non_standard"),
        None,
    )
    assert compaction_block
    assert compaction_block["value"].get("type") == "compaction"

    third_message = {
        "role": "user",
        "content": "What are we talking about?",
    }
    messages.append(third_message)
    third_response = llm.invoke(messages)
    content_blocks = third_response.content_blocks
    assert [block["type"] for block in content_blocks] == ["text"]

def test_compaction_streaming() -> None:
    """Test the compaction beta feature."""
    llm = ChatAnthropic(
        model="claude-opus-4-6",  # type: ignore[call-arg]
        betas=["compact-2026-01-12"],
        max_tokens=4096,
        context_management={
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": 50000},
                    "pause_after_compaction": False,
                }
            ]
        },
        streaming=True,
    )

    input_message = {
        "role": "user",
        "content": f"Generate a one-sentence summary of this:\n\n{'a' * 100000}",
    }
    messages: list = [input_message]

    first_response = llm.invoke(messages)
    messages.append(first_response)

    second_message = {
        "role": "user",
        "content": f"Generate a one-sentence summary of this:\n\n{'b' * 100000}",
    }
    messages.append(second_message)

    second_response = llm.invoke(messages)
    messages.append(second_response)

    content_blocks = second_response.content_blocks
    compaction_block = next(
        (block for block in content_blocks if block["type"] == "non_standard"),
        None,
    )
    assert compaction_block
    assert compaction_block["value"].get("type") == "compaction"

    third_message = {
        "role": "user",
        "content": "What are we talking about?",
    }
    messages.append(third_message)
    third_response = llm.invoke(messages)
    content_blocks = third_response.content_blocks
    assert [block["type"] for block in content_blocks] == ["text"]


# --- libs/partners/anthropic/tests/integration_tests/test_llms.py ---

def test_anthropic_call() -> None:
    """Test valid call to anthropic."""
    llm = AnthropicLLM(model=MODEL)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_anthropic_streaming() -> None:
    """Test streaming tokens from anthropic."""
    llm = AnthropicLLM(model=MODEL)  # type: ignore[call-arg]
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)

async def test_anthropic_async_generate() -> None:
    """Test async generate."""
    llm = AnthropicLLM(model=MODEL)  # type: ignore[call-arg]
    output = await llm.agenerate(["How many toes do dogs have?"])
    assert isinstance(output, LLMResult)


# --- libs/partners/anthropic/tests/unit_tests/middleware/test_prompt_caching.py ---

    def test_tags_last_block_of_string_system_message(self) -> None:
        result = self._run(self._make_request(SystemMessage("Base prompt")))
        blocks = self._get_content_blocks(result)
        assert len(blocks) == 1
        assert blocks[0]["text"] == "Base prompt"
        assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    def test_tags_only_last_block_of_multi_block_system_message(self) -> None:
        msg = SystemMessage(
            content=[
                {"type": "text", "text": "Block 1"},
                {"type": "text", "text": "Block 2"},
                {"type": "text", "text": "Block 3"},
            ]
        )
        blocks = self._get_content_blocks(self._run(self._make_request(msg)))
        assert len(blocks) == 3
        assert "cache_control" not in blocks[0]
        assert "cache_control" not in blocks[1]
        assert blocks[2]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    def test_does_not_mutate_original_system_message(self) -> None:
        original_content: list[str | dict[str, str]] = [
            {"type": "text", "text": "Block 1"},
            {"type": "text", "text": "Block 2"},
        ]
        msg = SystemMessage(content=original_content)
        self._run(self._make_request(msg))
        assert "cache_control" not in original_content[1]

    def test_passes_through_when_no_system_message(self) -> None:
        result = self._run(self._make_request(system_message=None))
        assert result.system_message is None

    def test_passes_through_when_system_message_has_empty_string(self) -> None:
        msg = SystemMessage(content="")
        result = self._run(self._make_request(msg))
        assert result.system_message is not None
        assert result.system_message.content == ""

    def test_passes_through_when_system_message_has_empty_list(self) -> None:
        msg = SystemMessage(content=[])
        result = self._run(self._make_request(msg))
        assert result.system_message is not None
        assert result.system_message.content == []

    def test_preserves_non_text_block_types(self) -> None:
        msg = SystemMessage(
            content=[
                {"type": "text", "text": "Prompt"},
                {"type": "custom_type", "data": "value"},
            ]
        )
        blocks = self._get_content_blocks(self._run(self._make_request(msg)))
        assert blocks[0] == {"type": "text", "text": "Prompt"}
        assert blocks[1]["type"] == "custom_type"
        assert blocks[1]["data"] == "value"
        assert blocks[1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    def test_tags_only_last_tool_with_cache_control(self) -> None:
        @tool
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return "sunny"

        @tool
        def get_time(timezone: str) -> str:
            """Get time in a timezone."""
            return "12:00"

        result = self._run(self._make_request(tools=[get_weather, get_time]))
        assert result.tools is not None
        assert len(result.tools) == 2
        first = result.tools[0]
        assert isinstance(first, BaseTool)
        assert first.extras is None or "cache_control" not in first.extras
        last = result.tools[1]
        assert isinstance(last, BaseTool)
        assert last.extras is not None
        assert last.extras["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    def test_does_not_mutate_original_tools(self) -> None:
        @tool
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        original_extras = my_tool.extras
        self._run(self._make_request(tools=[my_tool]))
        assert my_tool.extras is original_extras

    def test_preserves_existing_extras(self) -> None:
        @tool(extras={"defer_loading": True})
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        result = self._run(self._make_request(tools=[my_tool]))
        assert result.tools is not None
        t = result.tools[0]
        assert isinstance(t, BaseTool)
        assert t.extras is not None
        assert t.extras["defer_loading"] is True
        assert t.extras["cache_control"] == {
            "type": "ephemeral",
            "ttl": "5m",
        }

    def test_passes_through_empty_tools(self) -> None:
        result = self._run(self._make_request(tools=[]))
        assert result.tools == []

    def test_passes_through_none_tools(self) -> None:
        result = self._run(self._make_request(tools=None))
        assert result.tools == []


# --- libs/partners/anthropic/tests/unit_tests/test_chat_models.py ---

def test_streaming_attribute_should_stream(async_api: bool) -> None:  # noqa: FBT001
    llm = ChatAnthropic(model=MODEL_NAME, streaming=True)
    assert llm._should_stream(async_api=async_api)

def test_anthropic_bind_tools_tool_choice() -> None:
    chat_model = ChatAnthropic(  # type: ignore[call-arg, call-arg]
        model=MODEL_NAME,
        anthropic_api_key="secret-api-key",
    )
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather],
        tool_choice={"type": "tool", "name": "GetWeather"},
    )
    assert cast("RunnableBinding", chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "tool",
        "name": "GetWeather",
    }
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather],
        tool_choice="GetWeather",
    )
    assert cast("RunnableBinding", chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "tool",
        "name": "GetWeather",
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="auto")
    assert cast("RunnableBinding", chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "auto",
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="any")
    assert cast("RunnableBinding", chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "any",
    }

def test_extras_with_fine_grained_streaming() -> None:
    @tool(extras={"eager_input_streaming": True})
    def tell_story(story: str) -> None:
        """Tell a story."""

    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    model_with_tools = model.bind_tools([tell_story])

    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )

    tell_story_tool = None
    for tool_def in payload["tools"]:
        if isinstance(tool_def, dict) and tool_def.get("name") == "tell_story":
            tell_story_tool = tool_def
            break

    assert tell_story_tool is not None
    assert tell_story_tool.get("eager_input_streaming") is True

def test_bind_tools_drops_forced_tool_choice_when_thinking_enabled() -> None:
    """Regression test for https://github.com/langchain-ai/langchain/issues/35539.

    Anthropic API rejects forced tool_choice when thinking is enabled:
    "Thinking may not be enabled when tool_choice forces tool use."
    bind_tools should drop forced tool_choice and warn.
    """
    chat_model = ChatAnthropic(
        model=MODEL_NAME,
        anthropic_api_key="secret-api-key",
        thinking={"type": "enabled", "budget_tokens": 5000},
    )

    # tool_choice="any" should be dropped with warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = chat_model.bind_tools([GetWeather], tool_choice="any")
    assert "tool_choice" not in cast("RunnableBinding", result).kwargs
    assert len(w) == 1
    assert "thinking is enabled" in str(w[0].message)

    # tool_choice="auto" should NOT be dropped (auto is allowed)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = chat_model.bind_tools([GetWeather], tool_choice="auto")
    assert cast("RunnableBinding", result).kwargs["tool_choice"] == {"type": "auto"}
    assert len(w) == 0

    # tool_choice=specific tool name should be dropped with warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = chat_model.bind_tools([GetWeather], tool_choice="GetWeather")
    assert "tool_choice" not in cast("RunnableBinding", result).kwargs
    assert len(w) == 1

    # tool_choice=dict with type "tool" should be dropped with warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = chat_model.bind_tools(
            [GetWeather],
            tool_choice={"type": "tool", "name": "GetWeather"},
        )
    assert "tool_choice" not in cast("RunnableBinding", result).kwargs
    assert len(w) == 1

    # tool_choice=dict with type "any" should also be dropped
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = chat_model.bind_tools(
            [GetWeather],
            tool_choice={"type": "any"},
        )
    assert "tool_choice" not in cast("RunnableBinding", result).kwargs
    assert len(w) == 1

def test_bind_tools_drops_forced_tool_choice_when_adaptive_thinking() -> None:
    """Adaptive thinking has the same forced tool_choice restriction as enabled."""
    chat_model = ChatAnthropic(
        model=MODEL_NAME,
        anthropic_api_key="secret-api-key",
        thinking={"type": "adaptive"},
    )

    # tool_choice="any" should be dropped with warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = chat_model.bind_tools([GetWeather], tool_choice="any")
    assert "tool_choice" not in cast("RunnableBinding", result).kwargs
    assert len(w) == 1
    assert "thinking is enabled" in str(w[0].message)

    # tool_choice="auto" should NOT be dropped (auto is allowed)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = chat_model.bind_tools([GetWeather], tool_choice="auto")
    assert cast("RunnableBinding", result).kwargs["tool_choice"] == {"type": "auto"}
    assert len(w) == 0

    # tool_choice=specific tool name should be dropped with warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = chat_model.bind_tools([GetWeather], tool_choice="GetWeather")
    assert "tool_choice" not in cast("RunnableBinding", result).kwargs
    assert len(w) == 1

    # tool_choice=dict with type "tool" should be dropped with warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = chat_model.bind_tools(
            [GetWeather],
            tool_choice={"type": "tool", "name": "GetWeather"},
        )
    assert "tool_choice" not in cast("RunnableBinding", result).kwargs
    assert len(w) == 1

    # tool_choice=dict with type "any" should also be dropped
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = chat_model.bind_tools(
            [GetWeather],
            tool_choice={"type": "any"},
        )
    assert "tool_choice" not in cast("RunnableBinding", result).kwargs
    assert len(w) == 1

def test_bind_tools_keeps_forced_tool_choice_when_thinking_disabled() -> None:
    """When thinking is not enabled, forced tool_choice should pass through."""
    chat_model = ChatAnthropic(
        model=MODEL_NAME,
        anthropic_api_key="secret-api-key",
    )

    # No thinking — tool_choice="any" should pass through
    result = chat_model.bind_tools([GetWeather], tool_choice="any")
    assert cast("RunnableBinding", result).kwargs["tool_choice"] == {"type": "any"}

    # Thinking explicitly None
    chat_model_none = ChatAnthropic(
        model=MODEL_NAME,
        anthropic_api_key="secret-api-key",
        thinking=None,
    )
    result = chat_model_none.bind_tools([GetWeather], tool_choice="any")
    assert cast("RunnableBinding", result).kwargs["tool_choice"] == {"type": "any"}

    # Thinking explicitly disabled — should NOT drop tool_choice
    chat_model_disabled = ChatAnthropic(
        model=MODEL_NAME,
        anthropic_api_key="secret-api-key",
        thinking={"type": "disabled"},
    )
    result = chat_model_disabled.bind_tools([GetWeather], tool_choice="any")
    assert cast("RunnableBinding", result).kwargs["tool_choice"] == {"type": "any"}


# --- libs/partners/anthropic/tests/unit_tests/test_output_parsers.py ---

def test_tools_output_parser_empty_content() -> None:
    class ChartType(BaseModel):
        chart_type: Literal["pie", "line", "bar"]

    output_parser = ToolsOutputParser(
        first_tool_only=True,
        pydantic_schemas=[ChartType],
    )
    message = AIMessage(
        "",
        tool_calls=[
            {
                "name": "ChartType",
                "args": {"chart_type": "pie"},
                "id": "foo",
                "type": "tool_call",
            },
        ],
    )
    actual = output_parser.invoke(message)
    expected = ChartType(chart_type="pie")
    assert expected == actual


# --- libs/partners/groq/tests/integration_tests/test_chat_models.py ---

def test_invoke() -> None:
    """Test Chat wrapper."""
    chat = ChatGroq(
        model=DEFAULT_MODEL_NAME,
        temperature=0.7,
        base_url=None,
        groq_proxy=None,
        timeout=10.0,
        max_retries=3,
        http_client=None,
        n=1,
        max_tokens=10,
        default_headers=None,
        default_query=None,
    )
    message = HumanMessage(content="Welcome to the Groqetship")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

async def test_ainvoke() -> None:
    """Test ainvoke tokens from ChatGroq."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10)

    result = await chat.ainvoke("Welcome to the Groqetship!", config={"tags": ["foo"]})
    assert isinstance(result, BaseMessage)
    assert isinstance(result.content, str)

async def test_stream() -> None:
    """Test streaming tokens from Groq."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10)

    for token in chat.stream("Welcome to the Groqetship!"):
        assert isinstance(token, BaseMessageChunk)
        assert isinstance(token.content, str)

async def test_astream() -> None:
    """Test streaming tokens from Groq."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10)

    full: BaseMessageChunk | None = None
    chunks_with_token_counts = 0
    chunks_with_response_metadata = 0
    async for token in chat.astream("Welcome to the Groqetship!"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        full = token if full is None else full + token
        if token.usage_metadata is not None:
            chunks_with_token_counts += 1
        if token.response_metadata and not set(token.response_metadata.keys()).issubset(
            {"model_provider", "output_version"}
        ):
            chunks_with_response_metadata += 1
    if chunks_with_token_counts != 1 or chunks_with_response_metadata != 1:
        msg = (
            "Expected exactly one chunk with token counts or metadata. "
            "AIMessageChunk aggregation adds / appends these metadata. Check that "
            "this is behaving properly."
        )
        raise AssertionError(msg)
    assert isinstance(full, AIMessageChunk)
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )
    for expected_metadata in ["model_name", "system_fingerprint"]:
        assert full.response_metadata[expected_metadata]

def test_generate() -> None:
    """Test sync generate."""
    n = 1
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10)
    message = HumanMessage(content="Hello", n=1)
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    assert response.llm_output["model_name"] == chat.model_name
    for generations in response.generations:
        assert len(generations) == n
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

async def test_agenerate() -> None:
    """Test async generation."""
    n = 1
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10, n=1)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    assert response.llm_output["model_name"] == chat.model_name
    for generations in response.generations:
        assert len(generations) == n
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

def test_reasoning_output_invoke() -> None:
    """Test reasoning output from ChatGroq with invoke."""
    chat = ChatGroq(
        model=REASONING_MODEL_NAME,
        reasoning_format="parsed",
    )
    message = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(content="I love programming."),
    ]
    response = chat.invoke(message)
    assert isinstance(response, AIMessage)
    assert "reasoning_content" in response.additional_kwargs
    assert isinstance(response.additional_kwargs["reasoning_content"], str)
    assert len(response.additional_kwargs["reasoning_content"]) > 0

def test_reasoning_output_stream() -> None:
    """Test reasoning output from ChatGroq with stream."""
    chat = ChatGroq(
        model=REASONING_MODEL_NAME,
        reasoning_format="parsed",
    )
    message = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(content="I love programming."),
    ]

    full_response: AIMessageChunk | None = None
    for token in chat.stream(message):
        assert isinstance(token, AIMessageChunk)

        if full_response is None:
            full_response = token
        else:
            # Casting since adding results in a type error
            full_response = cast("AIMessageChunk", full_response + token)

    assert full_response is not None
    assert isinstance(full_response, AIMessageChunk)
    assert "reasoning_content" in full_response.additional_kwargs
    assert isinstance(full_response.additional_kwargs["reasoning_content"], str)
    assert len(full_response.additional_kwargs["reasoning_content"]) > 0

def test_reasoning_effort_none() -> None:
    """Test that no reasoning output is returned if effort is set to none."""
    chat = ChatGroq(
        model="qwen/qwen3-32b",  # Only qwen3 currently supports reasoning_effort = none
        reasoning_effort="none",
    )
    message = HumanMessage(content="What is the capital of France?")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert "reasoning_content" not in response.additional_kwargs
    assert "<think>" not in response.content
    assert "<think/>" not in response.content

def test_reasoning_effort_levels(effort: str) -> None:
    """Test reasoning effort options for different levels."""
    # As of now, only the new gpt-oss models support `'low'`, `'medium'`, and `'high'`
    chat = ChatGroq(
        model=DEFAULT_MODEL_NAME,
        reasoning_effort=effort,
    )
    message = HumanMessage(content="What is the capital of France?")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.response_metadata.get("reasoning_effort") == effort

def test_reasoning_effort_invoke_override(effort: str) -> None:
    """Test that reasoning_effort in invoke() overrides class-level setting."""
    # Create chat with no reasoning effort at class level
    chat = ChatGroq(
        model=DEFAULT_MODEL_NAME,
    )
    message = HumanMessage(content="What is the capital of France?")

    # Override reasoning_effort in invoke()
    response = chat.invoke([message], reasoning_effort=effort)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.response_metadata.get("reasoning_effort") == effort

def test_reasoning_effort_invoke_override_different_level() -> None:
    """Test that reasoning_effort in invoke() overrides class-level setting."""
    # Create chat with reasoning effort at class level
    chat = ChatGroq(
        model=DEFAULT_MODEL_NAME,  # openai/gpt-oss-20b supports reasoning_effort
        reasoning_effort="high",
    )
    message = HumanMessage(content="What is the capital of France?")

    # Override reasoning_effort to 'low' in invoke()
    response = chat.invoke([message], reasoning_effort="low")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    # Should reflect the overridden value, not the class-level setting
    assert response.response_metadata.get("reasoning_effort") == "low"

def test_reasoning_effort_streaming() -> None:
    """Test that reasoning_effort is captured in streaming response metadata."""
    chat = ChatGroq(
        model=DEFAULT_MODEL_NAME,
        reasoning_effort="medium",
    )
    message = HumanMessage(content="What is the capital of France?")

    chunks = list(chat.stream([message]))
    assert len(chunks) > 0

    # Find the final chunk with finish_reason
    final_chunk = None
    for chunk in chunks:
        if chunk.response_metadata.get("finish_reason"):
            final_chunk = chunk
            break

    assert final_chunk is not None
    assert final_chunk.response_metadata.get("reasoning_effort") == "medium"

def test_system_message() -> None:
    """Test ChatGroq wrapper with system message."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_tool_choice() -> None:
    """Test that tool choice is respected."""
    llm = ChatGroq(model=DEFAULT_MODEL_NAME)

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice="MyTool")

    resp = with_tool.invoke("Who was the 27 year old named Erick? Use the tool.")
    assert isinstance(resp, AIMessage)
    assert resp.content == ""  # should just be tool call
    tool_calls = resp.additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"

    assert isinstance(resp.tool_calls, list)
    assert len(resp.tool_calls) == 1
    tool_call = resp.tool_calls[0]
    assert tool_call["name"] == "MyTool"
    assert tool_call["args"] == {"name": "Erick", "age": 27}

def test_tool_choice_bool() -> None:
    """Test that tool choice is respected just passing in True."""
    llm = ChatGroq(model=DEFAULT_MODEL_NAME)

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice=True)

    resp = with_tool.invoke("Who was the 27 year old named Erick? Use the tool.")
    assert isinstance(resp, AIMessage)
    assert resp.content == ""  # should just be tool call
    tool_calls = resp.additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"

def test_streaming_tool_call() -> None:
    """Test that tool choice is respected."""
    llm = ChatGroq(model=DEFAULT_MODEL_NAME)

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice="MyTool")

    resp = with_tool.stream("Who was the 27 year old named Erick?")
    additional_kwargs = None
    for chunk in resp:
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.content == ""  # should just be tool call
        additional_kwargs = chunk.additional_kwargs

    assert additional_kwargs is not None
    tool_calls = additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"

    assert isinstance(chunk, AIMessageChunk)
    assert isinstance(chunk.tool_call_chunks, list)
    assert len(chunk.tool_call_chunks) == 1
    tool_call_chunk = chunk.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "MyTool"
    assert isinstance(tool_call_chunk["args"], str)
    assert json.loads(tool_call_chunk["args"]) == {"name": "Erick", "age": 27}

async def test_astreaming_tool_call() -> None:
    """Test that tool choice is respected."""
    llm = ChatGroq(model=DEFAULT_MODEL_NAME)

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice="MyTool")

    resp = with_tool.astream("Who was the 27 year old named Erick?")
    additional_kwargs = None
    async for chunk in resp:
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.content == ""  # should just be tool call
        additional_kwargs = chunk.additional_kwargs

    assert additional_kwargs is not None
    tool_calls = additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"

    assert isinstance(chunk, AIMessageChunk)
    assert isinstance(chunk.tool_call_chunks, list)
    assert len(chunk.tool_call_chunks) == 1
    tool_call_chunk = chunk.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "MyTool"
    assert isinstance(tool_call_chunk["args"], str)
    assert json.loads(tool_call_chunk["args"]) == {"name": "Erick", "age": 27}

def test_json_mode_structured_output() -> None:
    """Test with_structured_output with json."""

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    chat = ChatGroq(model=DEFAULT_MODEL_NAME).with_structured_output(
        Joke, method="json_mode"
    )
    result = chat.invoke(
        "Tell me a joke about cats, respond in JSON with `setup` and `punchline` keys"
    )
    assert type(result) is Joke
    assert len(result.setup) != 0
    assert len(result.punchline) != 0

def test_setting_service_tier_class() -> None:
    """Test setting service tier defined at ChatGroq level."""
    message = HumanMessage(content="Welcome to the Groqetship")

    # Initialization
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="auto")
    assert chat.service_tier == "auto"
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
    assert response.response_metadata.get("service_tier") == "auto"

    chat = ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="flex")
    assert chat.service_tier == "flex"
    response = chat.invoke([message])
    assert response.response_metadata.get("service_tier") == "flex"

    chat = ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="on_demand")
    assert chat.service_tier == "on_demand"
    response = chat.invoke([message])
    assert response.response_metadata.get("service_tier") == "on_demand"

    chat = ChatGroq(model=DEFAULT_MODEL_NAME)
    assert chat.service_tier == "on_demand"
    response = chat.invoke([message])
    assert response.response_metadata.get("service_tier") == "on_demand"

    with pytest.raises(ValueError):
        ChatGroq(model=DEFAULT_MODEL_NAME, service_tier=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="invalid")  # type: ignore[arg-type]

def test_setting_service_tier_request() -> None:
    """Test setting service tier defined at request level."""
    message = HumanMessage(content="Welcome to the Groqetship")
    chat = ChatGroq(model=DEFAULT_MODEL_NAME)

    response = chat.invoke(
        [message],
        service_tier="auto",
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
    assert response.response_metadata.get("service_tier") == "auto"

    response = chat.invoke(
        [message],
        service_tier="flex",
    )
    assert response.response_metadata.get("service_tier") == "flex"

    response = chat.invoke(
        [message],
        service_tier="on_demand",
    )
    assert response.response_metadata.get("service_tier") == "on_demand"

    assert chat.service_tier == "on_demand"
    response = chat.invoke(
        [message],
    )
    assert response.response_metadata.get("service_tier") == "on_demand"

    # If an `invoke` call is made with no service tier, we fall back to the class level
    # setting
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="auto")
    response = chat.invoke(
        [message],
    )
    assert response.response_metadata.get("service_tier") == "auto"

    response = chat.invoke(
        [message],
        service_tier="on_demand",
    )
    assert response.response_metadata.get("service_tier") == "on_demand"

    with pytest.raises(BadRequestError):
        response = chat.invoke(
            [message],
            service_tier="invalid",
        )

    response = chat.invoke(
        [message],
        service_tier=None,
    )
    assert response.response_metadata.get("service_tier") == "auto"

def test_setting_service_tier_streaming() -> None:
    """Test service tier settings for streaming calls."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="flex")
    chunks = list(chat.stream("Why is the sky blue?", service_tier="auto"))

    # Find the final chunk with finish_reason
    final_chunk = None
    for chunk in chunks:
        if chunk.response_metadata.get("finish_reason"):
            final_chunk = chunk
            break

    assert final_chunk is not None
    assert final_chunk.response_metadata.get("service_tier") == "auto"

async def test_setting_service_tier_request_async() -> None:
    """Test async setting of service tier at the request level."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="flex")
    response = await chat.ainvoke("Hello!", service_tier="on_demand")

    assert response.response_metadata.get("service_tier") == "on_demand"

def test_web_search() -> None:
    llm = ChatGroq(model="groq/compound")
    input_message = {
        "role": "user",
        "content": "Search for the weather in Boston today.",
    }
    full: AIMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs["reasoning_content"]
    assert full.additional_kwargs["executed_tools"]
    assert [block["type"] for block in full.content_blocks] == [
        "reasoning",
        "server_tool_call",
        "server_tool_result",
        "text",
    ]

    next_message = {
        "role": "user",
        "content": "Now search for the weather in San Francisco.",
    }
    response = llm.invoke([input_message, full, next_message])
    assert [block["type"] for block in response.content_blocks] == [
        "reasoning",
        "server_tool_call",
        "server_tool_result",
        "text",
    ]

def test_web_search_v1() -> None:
    llm = ChatGroq(model="groq/compound", output_version="v1")
    input_message = {
        "role": "user",
        "content": "Search for the weather in Boston today.",
    }
    full: AIMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs["reasoning_content"]
    assert full.additional_kwargs["executed_tools"]
    assert [block["type"] for block in full.content_blocks] == [
        "reasoning",
        "server_tool_call",
        "server_tool_result",
        "reasoning",
        "text",
    ]

    next_message = {
        "role": "user",
        "content": "Now search for the weather in San Francisco.",
    }
    response = llm.invoke([input_message, full, next_message])
    assert [block["type"] for block in response.content_blocks] == [
        "reasoning",
        "server_tool_call",
        "server_tool_result",
        "text",
    ]

def test_code_interpreter() -> None:
    llm = ChatGroq(model="groq/compound-mini")
    input_message = {
        "role": "user",
        "content": (
            "Calculate the square root of 101 and show me the Python code you used."
        ),
    }
    full: AIMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs["reasoning_content"]
    assert full.additional_kwargs["executed_tools"]
    assert [block["type"] for block in full.content_blocks] == [
        "reasoning",
        "server_tool_call",
        "server_tool_result",
        "text",
    ]

    next_message = {
        "role": "user",
        "content": "Now do the same for 102.",
    }
    response = llm.invoke([input_message, full, next_message])
    assert [block["type"] for block in response.content_blocks] == [
        "reasoning",
        "server_tool_call",
        "server_tool_result",
        "text",
    ]

def test_code_interpreter_v1() -> None:
    llm = ChatGroq(model="groq/compound-mini", output_version="v1")
    input_message = {
        "role": "user",
        "content": (
            "Calculate the square root of 101 and show me the Python code you used."
        ),
    }
    full: AIMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs["reasoning_content"]
    assert full.additional_kwargs["executed_tools"]
    assert [block["type"] for block in full.content_blocks] == [
        "reasoning",
        "server_tool_call",
        "server_tool_result",
        "reasoning",
        "text",
    ]

    next_message = {
        "role": "user",
        "content": "Now do the same for 102.",
    }
    response = llm.invoke([input_message, full, next_message])
    assert [block["type"] for block in response.content_blocks] == [
        "reasoning",
        "server_tool_call",
        "server_tool_result",
        "text",
    ]


# --- libs/partners/huggingface/tests/unit_tests/test_chat_models.py ---

def test_property_inheritance_integration(chat_hugging_face: Any) -> None:
    """Test that ChatHuggingFace inherits params from LLM object."""
    assert getattr(chat_hugging_face, "temperature", None) == 0.7
    assert getattr(chat_hugging_face, "max_tokens", None) == 512
    assert getattr(chat_hugging_face, "top_p", None) == 0.9
    assert getattr(chat_hugging_face, "streaming", None) is True


# --- libs/partners/ollama/tests/integration_tests/chat_models/test_chat_models.py ---

def test_structured_output(method: str) -> None:
    """Test to verify structured output via tool calling and `format` parameter."""

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    llm = ChatOllama(model=DEFAULT_MODEL_NAME, temperature=0)
    query = "Tell me a joke about cats."

    # Pydantic
    if method == "function_calling":
        structured_llm = llm.with_structured_output(Joke, method="function_calling")
        result = structured_llm.invoke(query)
        assert isinstance(result, Joke)

        for chunk in structured_llm.stream(query):
            assert isinstance(chunk, Joke)

    # JSON Schema
    if method == "json_schema":
        structured_llm = llm.with_structured_output(
            Joke.model_json_schema(), method="json_schema"
        )
        result = structured_llm.invoke(query)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        for chunk in structured_llm.stream(query):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)
        assert set(chunk.keys()) == {"setup", "punchline"}

        # Typed Dict
        class JokeSchema(TypedDict):
            """Joke to tell user."""

            setup: Annotated[str, "question to set up a joke"]
            punchline: Annotated[str, "answer to resolve the joke"]

        structured_llm = llm.with_structured_output(JokeSchema, method="json_schema")
        result = structured_llm.invoke(query)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        for chunk in structured_llm.stream(query):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)
        assert set(chunk.keys()) == {"setup", "punchline"}

def test_structured_output_deeply_nested(model: str) -> None:
    """Test to verify structured output with a nested objects."""
    llm = ChatOllama(model=model, temperature=0)

    class Person(BaseModel):
        """Information about a person."""

        name: str | None = Field(default=None, description="The name of the person")
        hair_color: str | None = Field(
            default=None, description="The color of the person's hair if known"
        )
        height_in_meters: str | None = Field(
            default=None, description="Height measured in meters"
        )

    class Data(BaseModel):
        """Extracted data about people."""

        people: list[Person]

    chat = llm.with_structured_output(Data)
    text = (
        "Alan Smith is 6 feet tall and has blond hair."
        "Alan Poe is 3 feet tall and has grey hair."
    )
    result = chat.invoke(text)
    assert isinstance(result, Data)

    for chunk in chat.stream(text):
        assert isinstance(chunk, Data)

def test_tool_streaming(model: str) -> None:
    """Test that the model can stream tool calls."""
    llm = ChatOllama(model=model)
    chat_model_with_tools = llm.bind_tools([get_current_weather])

    prompt = [HumanMessage("What is the weather today in Boston?")]

    # Flags and collectors for validation
    tool_chunk_found = False
    final_tool_calls = []
    collected_tool_chunks: list[ToolCallChunk] = []

    # Stream the response and inspect the chunks
    for chunk in chat_model_with_tools.stream(prompt):
        assert isinstance(chunk, AIMessageChunk), "Expected AIMessageChunk type"

        if chunk.tool_call_chunks:
            tool_chunk_found = True
            collected_tool_chunks.extend(chunk.tool_call_chunks)

        if chunk.tool_calls:
            final_tool_calls.extend(chunk.tool_calls)

    assert tool_chunk_found, "Tool streaming did not produce any tool_call_chunks."
    assert len(final_tool_calls) == 1, (
        f"Expected 1 final tool call, but got {len(final_tool_calls)}"
    )

    final_tool_call = final_tool_calls[0]
    assert final_tool_call["name"] == "get_current_weather"
    assert final_tool_call["args"] == {"location": "Boston"}

    assert len(collected_tool_chunks) > 0
    assert collected_tool_chunks[0]["name"] == "get_current_weather"

    # The ID should be consistent across chunks that have it
    tool_call_id = collected_tool_chunks[0].get("id")
    assert tool_call_id is not None
    assert all(
        chunk.get("id") == tool_call_id
        for chunk in collected_tool_chunks
        if chunk.get("id")
    )
    assert final_tool_call["id"] == tool_call_id

async def test_tool_astreaming(model: str) -> None:
    """Test that the model can stream tool calls."""
    llm = ChatOllama(model=model)
    chat_model_with_tools = llm.bind_tools([get_current_weather])

    prompt = [HumanMessage("What is the weather today in Boston?")]

    # Flags and collectors for validation
    tool_chunk_found = False
    final_tool_calls = []
    collected_tool_chunks: list[ToolCallChunk] = []

    # Stream the response and inspect the chunks
    async for chunk in chat_model_with_tools.astream(prompt):
        assert isinstance(chunk, AIMessageChunk), "Expected AIMessageChunk type"

        if chunk.tool_call_chunks:
            tool_chunk_found = True
            collected_tool_chunks.extend(chunk.tool_call_chunks)

        if chunk.tool_calls:
            final_tool_calls.extend(chunk.tool_calls)

    assert tool_chunk_found, "Tool streaming did not produce any tool_call_chunks."
    assert len(final_tool_calls) == 1, (
        f"Expected 1 final tool call, but got {len(final_tool_calls)}"
    )

    final_tool_call = final_tool_calls[0]
    assert final_tool_call["name"] == "get_current_weather"
    assert final_tool_call["args"] == {"location": "Boston"}

    assert len(collected_tool_chunks) > 0
    assert collected_tool_chunks[0]["name"] == "get_current_weather"

    # The ID should be consistent across chunks that have it
    tool_call_id = collected_tool_chunks[0].get("id")
    assert tool_call_id is not None
    assert all(
        chunk.get("id") == tool_call_id
        for chunk in collected_tool_chunks
        if chunk.get("id")
    )
    assert final_tool_call["id"] == tool_call_id

def test_agent_loop(model: str, output_version: str | None) -> None:
    """Test agent loop with tool calling and message passing."""

    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny and 75 degrees."

    llm = ChatOllama(model=model, output_version=output_version, reasoning="low")
    llm_with_tools = llm.bind_tools([get_weather])

    input_message = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)

    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert "location" in tool_call["args"]

    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content
    assert isinstance(tool_message.content, str)
    assert "sunny" in tool_message.content.lower()

    resp_message = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    follow_up = HumanMessage("Explain why that might be using a reasoning step.")
    assert isinstance(resp_message, AIMessage)
    assert len(resp_message.content) > 0

    response = llm_with_tools.invoke(
        [input_message, tool_call_message, tool_message, resp_message, follow_up]
    )
    assert isinstance(resp_message, AIMessage)
    assert len(resp_message.content) > 0

    if output_version == "v1":
        content_blocks = response.content_blocks
        assert content_blocks is not None
        assert len(content_blocks) > 0
        assert any(block["type"] == "text" for block in content_blocks)
        assert any(block["type"] == "reasoning" for block in content_blocks)


# --- libs/partners/openai/tests/integration_tests/chat_models/test_azure.py ---

def test_chat_openai(llm: AzureChatOpenAI) -> None:
    """Test AzureChatOpenAI wrapper."""
    message = HumanMessage(content="Hello")
    response = llm.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_chat_openai_generate() -> None:
    """Test AzureChatOpenAI wrapper with generate."""
    chat = _get_llm(max_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

def test_chat_openai_multiple_completions() -> None:
    """Test AzureChatOpenAI wrapper with multiple completions."""
    chat = _get_llm(max_tokens=10, n=5)
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)

async def test_async_chat_openai() -> None:
    """Test async generation."""
    chat = _get_llm(max_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

def test_openai_streaming(llm: AzureChatOpenAI) -> None:
    """Test streaming tokens from OpenAI."""
    full: BaseMessageChunk | None = None
    for chunk in llm.stream("I'm Pickle Rick"):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.response_metadata.get("model_name") is not None

async def test_openai_astream(llm: AzureChatOpenAI) -> None:
    """Test streaming tokens from OpenAI."""

    full: BaseMessageChunk | None = None
    async for chunk in llm.astream("I'm Pickle Rick"):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.response_metadata.get("model_name") is not None

async def test_openai_ainvoke(llm: AzureChatOpenAI) -> None:
    """Test invoke tokens from AzureChatOpenAI."""

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)
    assert result.response_metadata.get("model_name") is not None

def test_openai_invoke(llm: AzureChatOpenAI) -> None:
    """Test invoke tokens from AzureChatOpenAI."""

    result = llm.invoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)
    assert result.response_metadata.get("model_name") is not None

def test_json_mode(llm: AzureChatOpenAI) -> None:
    response = llm.invoke(
        "Return this as json: {'a': 1}", response_format={"type": "json_object"}
    )
    assert isinstance(response.content, str)
    assert json.loads(response.content) == {"a": 1}

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream(
        "Return this as json: {'a': 1}", response_format={"type": "json_object"}
    ):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, str)
    assert json.loads(full.content) == {"a": 1}

async def test_json_mode_async(llm: AzureChatOpenAI) -> None:
    response = await llm.ainvoke(
        "Return this as json: {'a': 1}", response_format={"type": "json_object"}
    )
    assert isinstance(response.content, str)
    assert json.loads(response.content) == {"a": 1}

    # Test streaming
    full: BaseMessageChunk | None = None
    async for chunk in llm.astream(
        "Return this as json: {'a': 1}", response_format={"type": "json_object"}
    ):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, str)
    assert json.loads(full.content) == {"a": 1}

def test_stream_response_format(llm: AzureChatOpenAI) -> None:
    full: BaseMessageChunk | None = None
    chunks = []
    for chunk in llm.stream("how are ya", response_format=Foo):
        chunks.append(chunk)
        full = chunk if full is None else full + chunk
    assert len(chunks) > 1
    assert isinstance(full, AIMessageChunk)
    parsed = full.additional_kwargs["parsed"]
    assert isinstance(parsed, Foo)
    assert isinstance(full.content, str)
    parsed_content = json.loads(full.content)
    assert parsed.response == parsed_content["response"]

async def test_astream_response_format(llm: AzureChatOpenAI) -> None:
    full: BaseMessageChunk | None = None
    chunks = []
    async for chunk in llm.astream("how are ya", response_format=Foo):
        chunks.append(chunk)
        full = chunk if full is None else full + chunk
    assert len(chunks) > 1
    assert isinstance(full, AIMessageChunk)
    parsed = full.additional_kwargs["parsed"]
    assert isinstance(parsed, Foo)
    assert isinstance(full.content, str)
    parsed_content = json.loads(full.content)
    assert parsed.response == parsed_content["response"]


# --- libs/partners/openai/tests/integration_tests/chat_models/test_base.py ---

def test_chat_openai() -> None:
    """Test ChatOpenAI wrapper."""
    chat = ChatOpenAI(
        temperature=0.7,
        base_url=None,
        organization=None,
        openai_proxy=None,
        timeout=10.0,
        max_retries=3,
        http_client=None,
        n=1,
        max_tokens=MAX_TOKEN_COUNT,  # type: ignore[call-arg]
        default_headers=None,
        default_query=None,
    )
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_chat_openai_system_message(use_responses_api: bool) -> None:
    """Test ChatOpenAI wrapper with system message."""
    chat = ChatOpenAI(use_responses_api=use_responses_api, max_tokens=MAX_TOKEN_COUNT)  # type: ignore[call-arg]
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.text, str)

def test_chat_openai_generate() -> None:
    """Test ChatOpenAI wrapper with generate."""
    chat = ChatOpenAI(max_tokens=MAX_TOKEN_COUNT, n=2)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

def test_chat_openai_multiple_completions() -> None:
    """Test ChatOpenAI wrapper with multiple completions."""
    chat = ChatOpenAI(max_tokens=MAX_TOKEN_COUNT, n=5)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)

def test_chat_openai_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatOpenAI(max_tokens=MAX_TOKEN_COUNT)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name

def test_chat_openai_streaming_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatOpenAI(max_tokens=MAX_TOKEN_COUNT, streaming=True)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name

def test_chat_openai_invalid_streaming_params() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with pytest.raises(ValueError):
        ChatOpenAI(max_tokens=MAX_TOKEN_COUNT, streaming=True, temperature=0, n=5)  # type: ignore[call-arg]

def test_openai_invoke() -> None:
    """Test invoke tokens from ChatOpenAI."""
    llm = ChatOpenAI(
        model="gpt-5-nano",
        service_tier="flex",  # Also test service_tier
        max_retries=3,  # Add retries for 503 capacity errors
    )

    result = llm.invoke("Hello", config={"tags": ["foo"]})
    assert isinstance(result.content, str)

    usage_metadata = result.usage_metadata  # type: ignore[attr-defined]

    # assert no response headers if include_response_headers is not set
    assert "headers" not in result.response_metadata
    assert usage_metadata is not None
    flex_input = usage_metadata.get("input_token_details", {}).get("flex")
    assert isinstance(flex_input, int)
    assert flex_input > 0
    assert flex_input == usage_metadata.get("input_tokens")
    flex_output = usage_metadata.get("output_token_details", {}).get("flex")
    assert isinstance(flex_output, int)
    assert flex_output > 0
    # GPT-5-nano/reasoning model specific. Remove if model used in test changes.
    flex_reasoning = usage_metadata.get("output_token_details", {}).get(
        "flex_reasoning"
    )
    assert isinstance(flex_reasoning, int)
    assert flex_reasoning > 0
    assert flex_reasoning + flex_output == usage_metadata.get("output_tokens")

def test_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatOpenAI(
        model="gpt-5-nano",
        service_tier="flex",  # Also test service_tier
        max_retries=3,  # Add retries for 503 capacity errors
    )

    full: BaseMessageChunk | None = None
    for chunk in llm.stream("I'm Pickle Rick"):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.response_metadata.get("finish_reason") is not None
    assert full.response_metadata.get("model_name") is not None

    # check token usage
    aggregate: BaseMessageChunk | None = None
    chunks_with_token_counts = 0
    chunks_with_response_metadata = 0
    for chunk in llm.stream("Hello"):
        assert isinstance(chunk.content, str)
        aggregate = chunk if aggregate is None else aggregate + chunk
        assert isinstance(chunk, AIMessageChunk)
        if chunk.usage_metadata is not None:
            chunks_with_token_counts += 1
        if chunk.response_metadata and not set(chunk.response_metadata.keys()).issubset(
            {"model_provider", "output_version"}
        ):
            chunks_with_response_metadata += 1
    if chunks_with_token_counts != 1 or chunks_with_response_metadata != 1:
        msg = (
            "Expected exactly one chunk with metadata. "
            "AIMessageChunk aggregation can add these metadata. Check that "
            "this is behaving properly."
        )
        raise AssertionError(msg)
    assert isinstance(aggregate, AIMessageChunk)
    assert aggregate.usage_metadata is not None
    assert aggregate.usage_metadata["input_tokens"] > 0
    assert aggregate.usage_metadata["output_tokens"] > 0
    assert aggregate.usage_metadata["total_tokens"] > 0
    assert aggregate.usage_metadata.get("input_token_details", {}).get("flex", 0) > 0  # type: ignore[operator]
    assert aggregate.usage_metadata.get("output_token_details", {}).get("flex", 0) > 0  # type: ignore[operator]
    assert (
        aggregate.usage_metadata.get("output_token_details", {}).get(  # type: ignore[operator]
            "flex_reasoning", 0
        )
        > 0
    )
    assert aggregate.usage_metadata.get("output_token_details", {}).get(  # type: ignore[operator]
        "flex_reasoning", 0
    ) + aggregate.usage_metadata.get("output_token_details", {}).get(
        "flex", 0
    ) == aggregate.usage_metadata.get("output_tokens")

async def test_astream() -> None:
    """Test streaming tokens from OpenAI."""

    async def _test_stream(stream: AsyncIterator, expect_usage: bool) -> None:
        full: BaseMessageChunk | None = None
        chunks_with_token_counts = 0
        chunks_with_response_metadata = 0
        async for chunk in stream:
            assert isinstance(chunk.content, str)
            full = chunk if full is None else full + chunk
            assert isinstance(chunk, AIMessageChunk)
            if chunk.usage_metadata is not None:
                chunks_with_token_counts += 1
            if chunk.response_metadata and not set(
                chunk.response_metadata.keys()
            ).issubset({"model_provider", "output_version"}):
                chunks_with_response_metadata += 1
        assert isinstance(full, AIMessageChunk)
        if chunks_with_response_metadata != 1:
            msg = (
                "Expected exactly one chunk with metadata. "
                "AIMessageChunk aggregation can add these metadata. Check that "
                "this is behaving properly."
            )
            raise AssertionError(msg)
        assert full.response_metadata.get("finish_reason") is not None
        assert full.response_metadata.get("model_name") is not None
        if expect_usage:
            if chunks_with_token_counts != 1:
                msg = (
                    "Expected exactly one chunk with token counts. "
                    "AIMessageChunk aggregation adds counts. Check that "
                    "this is behaving properly."
                )
                raise AssertionError(msg)
            assert full.usage_metadata is not None
            assert full.usage_metadata["input_tokens"] > 0
            assert full.usage_metadata["output_tokens"] > 0
            assert full.usage_metadata["total_tokens"] > 0
        else:
            assert chunks_with_token_counts == 0
            assert full.usage_metadata is None

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_tokens=MAX_TOKEN_COUNT)  # type: ignore[call-arg]
    await _test_stream(llm.astream("Hello", stream_usage=False), expect_usage=False)
    await _test_stream(
        llm.astream("Hello", stream_options={"include_usage": True}), expect_usage=True
    )
    await _test_stream(llm.astream("Hello", stream_usage=True), expect_usage=True)
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        max_tokens=MAX_TOKEN_COUNT,  # type: ignore[call-arg]
        model_kwargs={"stream_options": {"include_usage": True}},
    )
    await _test_stream(llm.astream("Hello"), expect_usage=True)
    await _test_stream(
        llm.astream("Hello", stream_options={"include_usage": False}),
        expect_usage=False,
    )
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        max_tokens=MAX_TOKEN_COUNT,  # type: ignore[call-arg]
        stream_usage=True,
    )
    await _test_stream(llm.astream("Hello"), expect_usage=True)
    await _test_stream(llm.astream("Hello", stream_usage=False), expect_usage=False)

def test_flex_usage_responses(streaming: bool) -> None:
    llm = ChatOpenAI(
        model="gpt-5-nano",
        service_tier="flex",
        max_retries=3,
        use_responses_api=True,
        streaming=streaming,
    )
    result = llm.invoke("Hello")
    assert result.usage_metadata
    flex_input = result.usage_metadata.get("input_token_details", {}).get("flex")
    flex_output = result.usage_metadata.get("output_token_details", {}).get("flex")
    flex_reasoning = result.usage_metadata.get("output_token_details", {}).get(
        "flex_reasoning"
    )
    assert isinstance(flex_input, int)
    assert isinstance(flex_output, int)
    assert isinstance(flex_reasoning, int)
    assert flex_output + flex_reasoning == result.usage_metadata.get("output_tokens")

def test_response_metadata() -> None:
    llm = ChatOpenAI()
    result = llm.invoke([HumanMessage(content="I'm PickleRick")], logprobs=True)
    assert result.response_metadata
    assert all(
        k in result.response_metadata
        for k in (
            "token_usage",
            "model_name",
            "logprobs",
            "system_fingerprint",
            "finish_reason",
            "service_tier",
        )
    )
    assert "content" in result.response_metadata["logprobs"]

async def test_async_response_metadata() -> None:
    llm = ChatOpenAI()
    result = await llm.ainvoke([HumanMessage(content="I'm PickleRick")], logprobs=True)
    assert result.response_metadata
    assert all(
        k in result.response_metadata
        for k in (
            "token_usage",
            "model_name",
            "logprobs",
            "system_fingerprint",
            "finish_reason",
            "service_tier",
        )
    )
    assert "content" in result.response_metadata["logprobs"]

def test_response_metadata_streaming() -> None:
    llm = ChatOpenAI()
    full: BaseMessageChunk | None = None
    for chunk in llm.stream("I'm Pickle Rick", logprobs=True):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert all(
        k in cast(BaseMessageChunk, full).response_metadata
        for k in ("logprobs", "finish_reason", "service_tier")
    )
    assert "content" in cast(BaseMessageChunk, full).response_metadata["logprobs"]

async def test_async_response_metadata_streaming() -> None:
    llm = ChatOpenAI()
    full: BaseMessageChunk | None = None
    async for chunk in llm.astream("I'm Pickle Rick", logprobs=True):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert all(
        k in cast(BaseMessageChunk, full).response_metadata
        for k in ("logprobs", "finish_reason", "service_tier")
    )
    assert "content" in cast(BaseMessageChunk, full).response_metadata["logprobs"]

def test_tool_use() -> None:
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    llm_with_tool = llm.bind_tools(tools=[GenerateUsername], tool_choice=True)
    msgs: list = [HumanMessage("Sally has green hair, what would her username be?")]
    ai_msg = llm_with_tool.invoke(msgs)

    assert isinstance(ai_msg, AIMessage)
    assert isinstance(ai_msg.tool_calls, list)
    assert len(ai_msg.tool_calls) == 1
    tool_call = ai_msg.tool_calls[0]
    assert "args" in tool_call

    tool_msg = ToolMessage("sally_green_hair", tool_call_id=ai_msg.tool_calls[0]["id"])
    msgs.extend([ai_msg, tool_msg])
    llm_with_tool.invoke(msgs)

    # Test streaming
    ai_messages = llm_with_tool.stream(msgs)
    first = True
    for message in ai_messages:
        if first:
            gathered = message
            first = False
        else:
            gathered = gathered + message  # type: ignore
    assert isinstance(gathered, AIMessageChunk)
    assert isinstance(gathered.tool_call_chunks, list)
    assert len(gathered.tool_call_chunks) == 1
    tool_call_chunk = gathered.tool_call_chunks[0]
    assert "args" in tool_call_chunk
    assert gathered.content_blocks == gathered.tool_calls

    streaming_tool_msg = ToolMessage(
        "sally_green_hair", tool_call_id=gathered.tool_calls[0]["id"]
    )
    msgs.extend([gathered, streaming_tool_msg])
    llm_with_tool.invoke(msgs)

def test_manual_tool_call_msg(use_responses_api: bool) -> None:
    """Test passing in manually construct tool call message."""
    llm = ChatOpenAI(
        model="gpt-5-nano", temperature=0, use_responses_api=use_responses_api
    )
    llm_with_tool = llm.bind_tools(tools=[GenerateUsername])
    msgs: list = [
        HumanMessage("Sally has green hair, what would her username be?"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="GenerateUsername",
                    args={"name": "Sally", "hair_color": "green"},
                    id="foo",
                    type="tool_call",
                )
            ],
        ),
        ToolMessage("sally_green_hair", tool_call_id="foo"),
    ]
    output: AIMessage = cast(AIMessage, llm_with_tool.invoke(msgs))
    assert output.content
    # Should not have called the tool again.
    assert not output.tool_calls
    assert not output.invalid_tool_calls

    # OpenAI should error when tool call id doesn't match across AIMessage and
    # ToolMessage
    msgs = [
        HumanMessage("Sally has green hair, what would her username be?"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="GenerateUsername",
                    args={"name": "Sally", "hair_color": "green"},
                    id="bar",
                    type="tool_call",
                )
            ],
        ),
        ToolMessage("sally_green_hair", tool_call_id="foo"),
    ]
    with pytest.raises(Exception):
        llm_with_tool.invoke(msgs)

def test_bind_tools_tool_choice(use_responses_api: bool) -> None:
    """Test passing in manually construct tool call message."""
    llm = ChatOpenAI(
        model="gpt-5-nano", temperature=0, use_responses_api=use_responses_api
    )
    for tool_choice in ("any", "required"):
        llm_with_tools = llm.bind_tools(
            tools=[GenerateUsername, MakeASandwich], tool_choice=tool_choice
        )
        msg = cast(AIMessage, llm_with_tools.invoke("how are you"))
        assert msg.tool_calls

    llm_with_tools = llm.bind_tools(tools=[GenerateUsername, MakeASandwich])
    msg = cast(AIMessage, llm_with_tools.invoke("how are you"))
    assert not msg.tool_calls

def test_disable_parallel_tool_calling() -> None:
    llm = ChatOpenAI(model="gpt-5-nano")
    llm_with_tools = llm.bind_tools([GenerateUsername], parallel_tool_calls=False)
    result = llm_with_tools.invoke(
        "Use the GenerateUsername tool to generate user names for:\n\n"
        "Sally with green hair\n"
        "Bob with blue hair"
    )
    assert isinstance(result, AIMessage)
    assert len(result.tool_calls) == 1

def test_openai_structured_output(model: str) -> None:
    class MyModel(BaseModel):
        """A Person"""

        name: str
        age: int

    llm = ChatOpenAI(model=model).with_structured_output(MyModel)
    result = llm.invoke("I'm a 27 year old named Erick")
    assert isinstance(result, MyModel)
    assert result.name == "Erick"
    assert result.age == 27

def test_openai_response_headers(use_responses_api: bool) -> None:
    """Test ChatOpenAI response headers."""
    chat_openai = ChatOpenAI(
        include_response_headers=True, use_responses_api=use_responses_api
    )
    query = "I'm Pickle Rick"
    result = chat_openai.invoke(query, max_tokens=MAX_TOKEN_COUNT)  # type: ignore[call-arg]
    headers = result.response_metadata["headers"]
    assert headers
    assert isinstance(headers, dict)
    assert "content-type" in headers

    # Stream
    full: BaseMessageChunk | None = None
    for chunk in chat_openai.stream(query, max_tokens=MAX_TOKEN_COUNT):  # type: ignore[call-arg]
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessage)
    headers = full.response_metadata["headers"]
    assert headers
    assert isinstance(headers, dict)
    assert "content-type" in headers

async def test_openai_response_headers_async(use_responses_api: bool) -> None:
    """Test ChatOpenAI response headers."""
    chat_openai = ChatOpenAI(
        include_response_headers=True, use_responses_api=use_responses_api
    )
    query = "I'm Pickle Rick"
    result = await chat_openai.ainvoke(query, max_tokens=MAX_TOKEN_COUNT)  # type: ignore[call-arg]
    headers = result.response_metadata["headers"]
    assert headers
    assert isinstance(headers, dict)
    assert "content-type" in headers

    # Stream
    full: BaseMessageChunk | None = None
    async for chunk in chat_openai.astream(query, max_tokens=MAX_TOKEN_COUNT):  # type: ignore[call-arg]
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessage)
    headers = full.response_metadata["headers"]
    assert headers
    assert isinstance(headers, dict)
    assert "content-type" in headers

def test_image_token_counting_jpeg() -> None:
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    image_url = "https://raw.githubusercontent.com/langchain-ai/docs/9f99bb977307a1bd5efeb8dc6b67eb13904c4af1/src/oss/images/checkpoints.jpg"
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe the weather in this image"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )
    expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
        "input_tokens"
    ]
    actual = model.get_num_tokens_from_messages([message])
    assert expected == actual

    image_data = base64.b64encode(httpx.get(image_url, timeout=10.0).content).decode(
        "utf-8"
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe the weather in this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )
    expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
        "input_tokens"
    ]
    actual = model.get_num_tokens_from_messages([message])
    assert expected == actual

def test_image_token_counting_png() -> None:
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    image_url = "https://raw.githubusercontent.com/langchain-ai/docs/4d11d08b6b0e210bd456943f7a22febbd168b543/src/images/agentic-rag-output.png"
    message = HumanMessage(
        content=[
            {"type": "text", "text": "how many dice are in this image"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )
    expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
        "input_tokens"
    ]
    actual = model.get_num_tokens_from_messages([message])
    assert expected == actual

    image_data = base64.b64encode(httpx.get(image_url, timeout=10.0).content).decode(
        "utf-8"
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": "how many dice are in this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ]
    )
    expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
        "input_tokens"
    ]
    actual = model.get_num_tokens_from_messages([message])
    assert expected == actual

def test_structured_output_strict(
    model: str,
    method: Literal["function_calling", "json_schema"],
    use_responses_api: bool,
) -> None:
    """Test to verify structured output with strict=True."""

    from pydantic import BaseModel as BaseModelProper
    from pydantic import Field as FieldProper

    llm = ChatOpenAI(model=model, use_responses_api=use_responses_api)

    class Joke(BaseModelProper):
        """Joke to tell user."""

        setup: str = FieldProper(description="question to set up a joke")
        punchline: str = FieldProper(description="answer to resolve the joke")

    # Pydantic class
    chat = llm.with_structured_output(Joke, method=method, strict=True)
    result = chat.invoke("Tell me a joke about cats.")
    assert isinstance(result, Joke)

    for chunk in chat.stream("Tell me a joke about cats."):
        assert isinstance(chunk, Joke)

    # Schema
    chat = llm.with_structured_output(
        Joke.model_json_schema(), method=method, strict=True
    )
    result = chat.invoke("Tell me a joke about cats.")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"setup", "punchline"}

    for chunk in chat.stream("Tell me a joke about cats."):
        assert isinstance(chunk, dict)
    assert isinstance(chunk, dict)  # for mypy
    assert set(chunk.keys()) == {"setup", "punchline"}

def test_nested_structured_output_strict(
    model: str, method: Literal["json_schema"], use_responses_api: bool
) -> None:
    """Test to verify structured output with strict=True for nested object."""

    from typing import TypedDict

    llm = ChatOpenAI(model=model, temperature=0, use_responses_api=use_responses_api)

    class SelfEvaluation(TypedDict):
        score: int
        text: str

    class JokeWithEvaluation(TypedDict):
        """Joke to tell user."""

        setup: str
        punchline: str
        self_evaluation: SelfEvaluation

    # Schema
    chat = llm.with_structured_output(JokeWithEvaluation, method=method, strict=True)
    result = chat.invoke("Tell me a joke about cats.")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"setup", "punchline", "self_evaluation"}
    assert set(result["self_evaluation"].keys()) == {"score", "text"}

    for chunk in chat.stream("Tell me a joke about cats."):
        assert isinstance(chunk, dict)
    assert isinstance(chunk, dict)  # for mypy
    assert set(chunk.keys()) == {"setup", "punchline", "self_evaluation"}
    assert set(chunk["self_evaluation"].keys()) == {"score", "text"}

def test_json_schema_openai_format(
    strict: bool, method: Literal["json_schema", "function_calling"]
) -> None:
    """Test we can pass in OpenAI schema format specifying strict."""
    llm = ChatOpenAI(model="gpt-5-nano")
    schema = {
        "name": "get_weather",
        "description": "Fetches the weather in the given location",
        "strict": strict,
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the weather for",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to return the temperature in",
                    "enum": ["F", "C"],
                },
            },
            "additionalProperties": False,
            "required": ["location", "unit"],
        },
    }
    chat = llm.with_structured_output(schema, method=method)
    result = chat.invoke("What is the weather in New York?")
    assert isinstance(result, dict)

def test_audio_output_modality() -> None:
    llm = ChatOpenAI(
        model="gpt-4o-audio-preview",
        temperature=0,
        model_kwargs={
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
        },
    )

    history: list[BaseMessage] = [
        HumanMessage("Make me a short audio clip of you yelling")
    ]

    output = llm.invoke(history)

    assert isinstance(output, AIMessage)
    assert "audio" in output.additional_kwargs

    history.append(output)
    history.append(HumanMessage("Make me a short audio clip of you whispering"))

    output = llm.invoke(history)

    assert isinstance(output, AIMessage)
    assert "audio" in output.additional_kwargs

def test_audio_input_modality() -> None:
    llm = ChatOpenAI(
        model="gpt-4o-audio-preview",
        temperature=0,
        model_kwargs={
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
        },
    )
    filepath = Path(__file__).parent / "audio_input.wav"

    audio_data = filepath.read_bytes()
    b64_audio_data = base64.b64encode(audio_data).decode("utf-8")

    history: list[BaseMessage] = [
        HumanMessage(
            [
                {"type": "text", "text": "What is happening in this audio clip"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": b64_audio_data, "format": "wav"},
                },
            ]
        )
    ]

    output = llm.invoke(history)

    assert isinstance(output, AIMessage)
    assert "audio" in output.additional_kwargs

    history.append(output)
    history.append(HumanMessage("Why?"))

    output = llm.invoke(history)

    assert isinstance(output, AIMessage)
    assert "audio" in output.additional_kwargs

def test_prediction_tokens() -> None:
    code = dedent(
        """
    /// <summary>
    /// Represents a user with a first name, last name, and username.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user's first name.
        /// </summary>
        public string FirstName { get; set; }

        /// <summary>
        /// Gets or sets the user's last name.
        /// </summary>
        public string LastName { get; set; }

        /// <summary>
        /// Gets or sets the user's username.
        /// </summary>
        public string Username { get; set; }
    }
    """
    )

    llm = ChatOpenAI(model="gpt-4.1-nano")
    query = (
        "Replace the Username property with an Email property. "
        "Respond only with code, and with no markdown formatting."
    )
    response = llm.invoke(
        [{"role": "user", "content": query}, {"role": "user", "content": code}],
        prediction={"type": "content", "content": code},
    )
    assert isinstance(response, AIMessage)
    assert response.response_metadata is not None
    output_token_details = response.response_metadata["token_usage"][
        "completion_tokens_details"
    ]
    assert output_token_details["accepted_prediction_tokens"] > 0
    assert output_token_details["rejected_prediction_tokens"] > 0

def test_stream_o_series(use_responses_api: bool) -> None:
    list(
        ChatOpenAI(model="o3-mini", use_responses_api=use_responses_api).stream(
            "how are you"
        )
    )

async def test_astream_o_series(use_responses_api: bool) -> None:
    async for _ in ChatOpenAI(
        model="o3-mini", use_responses_api=use_responses_api
    ).astream("how are you"):
        pass

def test_stream_response_format() -> None:
    full: BaseMessageChunk | None = None
    chunks = []
    for chunk in ChatOpenAI(model="gpt-5-nano").stream(
        "how are ya", response_format=Foo
    ):
        chunks.append(chunk)
        full = chunk if full is None else full + chunk
    assert len(chunks) > 1
    assert isinstance(full, AIMessageChunk)
    parsed = full.additional_kwargs["parsed"]
    assert isinstance(parsed, Foo)
    assert isinstance(full.content, str)
    parsed_content = json.loads(full.content)
    assert parsed.response == parsed_content["response"]

async def test_astream_response_format() -> None:
    full: BaseMessageChunk | None = None
    chunks = []
    async for chunk in ChatOpenAI(model="gpt-5-nano").astream(
        "how are ya", response_format=Foo
    ):
        chunks.append(chunk)
        full = chunk if full is None else full + chunk
    assert len(chunks) > 1
    assert isinstance(full, AIMessageChunk)
    parsed = full.additional_kwargs["parsed"]
    assert isinstance(parsed, Foo)
    assert isinstance(full.content, str)
    parsed_content = json.loads(full.content)
    assert parsed.response == parsed_content["response"]

def test_o1(use_max_completion_tokens: bool, use_responses_api: bool) -> None:
    # o1 models need higher token limits for reasoning
    o1_token_limit = 1000
    if use_max_completion_tokens:
        kwargs: dict = {"max_completion_tokens": o1_token_limit}
    else:
        kwargs = {"max_tokens": o1_token_limit}
    response = ChatOpenAI(
        model="o1",
        reasoning_effort="low",
        use_responses_api=use_responses_api,
        **kwargs,
    ).invoke(
        [
            {"role": "developer", "content": "respond in all caps"},
            {"role": "user", "content": "HOW ARE YOU"},
        ]
    )
    assert isinstance(response, AIMessage)
    assert isinstance(response.text, str)
    assert response.text.upper() == response.text

def test_o1_stream_default_works() -> None:
    result = list(ChatOpenAI(model="o1").stream("say 'hi'"))
    assert len(result) > 0

def test_multi_party_conversation() -> None:
    llm = ChatOpenAI(model="gpt-5-nano")
    messages = [
        HumanMessage("Hi, I have black hair.", name="Alice"),
        HumanMessage("Hi, I have brown hair.", name="Bob"),
        HumanMessage("Who just spoke?", name="Charlie"),
    ]
    response = llm.invoke(messages)
    assert "Bob" in response.content

def test_structured_output_and_tools(schema: Any) -> None:
    llm = ChatOpenAI(model="gpt-5-nano", verbosity="low").bind_tools(
        [GenerateUsername], strict=True, response_format=schema
    )

    response = llm.invoke("What weighs more, a pound of feathers or a pound of gold?")
    if schema == ResponseFormat:
        parsed = response.additional_kwargs["parsed"]
        assert isinstance(parsed, ResponseFormat)
    else:
        parsed = json.loads(response.text)
        assert isinstance(parsed, dict)
        assert parsed["response"]
        assert parsed["explanation"]

    # Test streaming tool calls
    full: BaseMessageChunk | None = None
    for chunk in llm.stream(
        "Generate a user name for Alice, black hair. Use the tool."
    ):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert len(full.tool_calls) == 1
    tool_call = full.tool_calls[0]
    assert tool_call["name"] == "GenerateUsername"

def test_tools_and_structured_output() -> None:
    llm = ChatOpenAI(model="gpt-5-nano").with_structured_output(
        ResponseFormat, strict=True, include_raw=True, tools=[GenerateUsername]
    )

    expected_keys = {"raw", "parsing_error", "parsed"}
    query = "Hello"
    tool_query = "Generate a user name for Alice, black hair. Use the tool."
    # Test invoke
    ## Engage structured output
    response = llm.invoke(query)
    assert isinstance(response["parsed"], ResponseFormat)
    ## Engage tool calling
    response_tools = llm.invoke(tool_query)
    ai_msg = response_tools["raw"]
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.tool_calls
    assert response_tools["parsed"] is None

    # Test stream
    aggregated: dict = {}
    for chunk in llm.stream(tool_query):
        assert isinstance(chunk, dict)
        assert all(key in expected_keys for key in chunk)
        aggregated = {**aggregated, **chunk}
    assert all(key in aggregated for key in expected_keys)
    assert isinstance(aggregated["raw"], AIMessage)
    assert aggregated["raw"].tool_calls
    assert aggregated["parsed"] is None

def test_prompt_cache_key_invoke() -> None:
    """Test that `prompt_cache_key` works with invoke calls."""
    chat = ChatOpenAI(model="gpt-5-nano", max_completion_tokens=500)
    messages = [HumanMessage("Say hello")]

    # Test that invoke works with prompt_cache_key parameter
    response = chat.invoke(messages, prompt_cache_key="integration-test-v1")

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content) > 0

    # Test that subsequent call with same cache key also works
    response2 = chat.invoke(messages, prompt_cache_key="integration-test-v1")

    assert isinstance(response2, AIMessage)
    assert isinstance(response2.content, str)
    assert len(response2.content) > 0

def test_prompt_cache_key_usage_methods_integration() -> None:
    """Integration test for `prompt_cache_key` usage methods."""
    messages = [HumanMessage("Say hi")]

    # Test keyword argument method
    chat = ChatOpenAI(model="gpt-5-nano", max_completion_tokens=10)
    response = chat.invoke(messages, prompt_cache_key="integration-test-v1")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

    # Test model-level via model_kwargs
    chat_model_level = ChatOpenAI(
        model="gpt-5-nano",
        max_completion_tokens=10,
        model_kwargs={"prompt_cache_key": "integration-model-level-v1"},
    )
    response_model_level = chat_model_level.invoke(messages)
    assert isinstance(response_model_level, AIMessage)
    assert isinstance(response_model_level.content, str)

def test_schema_parsing_failures() -> None:
    llm = ChatOpenAI(model="gpt-5-nano", use_responses_api=False)
    try:
        llm.invoke("respond with good", response_format=BadModel)
    except Exception as e:
        assert e.response is not None  # type: ignore[attr-defined]
    else:
        raise AssertionError

def test_schema_parsing_failures_responses_api() -> None:
    llm = ChatOpenAI(model="gpt-5-nano", use_responses_api=True)
    try:
        llm.invoke("respond with good", response_format=BadModel)
    except Exception as e:
        assert e.response is not None  # type: ignore[attr-defined]
    else:
        raise AssertionError

async def test_schema_parsing_failures_async() -> None:
    llm = ChatOpenAI(model="gpt-5-nano", use_responses_api=False)
    try:
        await llm.ainvoke("respond with good", response_format=BadModel)
    except Exception as e:
        assert e.response is not None  # type: ignore[attr-defined]
    else:
        raise AssertionError

async def test_schema_parsing_failures_responses_api_async() -> None:
    llm = ChatOpenAI(model="gpt-5-nano", use_responses_api=True)
    try:
        await llm.ainvoke("respond with good", response_format=BadModel)
    except Exception as e:
        assert e.response is not None  # type: ignore[attr-defined]
    else:
        raise AssertionError


# --- libs/partners/openai/tests/integration_tests/chat_models/test_responses_api.py ---

def test_incomplete_response() -> None:
    model = ChatOpenAI(
        model=MODEL_NAME, use_responses_api=True, max_completion_tokens=16
    )
    response = model.invoke("Tell me a 100 word story about a bear.")
    assert response.response_metadata["incomplete_details"]
    assert response.response_metadata["incomplete_details"]["reason"]
    assert response.response_metadata["status"] == "incomplete"

    full: AIMessageChunk | None = None
    for chunk in model.stream("Tell me a 100 word story about a bear."):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.response_metadata["incomplete_details"]
    assert full.response_metadata["incomplete_details"]["reason"]
    assert full.response_metadata["status"] == "incomplete"

def test_web_search(output_version: Literal["responses/v1", "v1"]) -> None:
    llm = ChatOpenAI(model=MODEL_NAME, output_version=output_version)
    first_response = llm.invoke(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    )
    _check_response(first_response)

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    ):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    _check_response(full)

    # Use OpenAI's stateful API
    response = llm.invoke(
        "what about a negative one",
        tools=[{"type": "web_search_preview"}],
        previous_response_id=first_response.response_metadata["id"],
    )
    _check_response(response)

    # Manually pass in chat history
    response = llm.invoke(
        [
            {"role": "user", "content": "What was a positive news story from today?"},
            first_response,
            {"role": "user", "content": "what about a negative one"},
        ],
        tools=[{"type": "web_search_preview"}],
    )
    _check_response(response)

    # Bind tool
    response = llm.bind_tools([{"type": "web_search_preview"}]).invoke(
        "What was a positive news story from today?"
    )
    _check_response(response)

    for msg in [first_response, full, response]:
        assert msg is not None
        block_types = [block["type"] for block in msg.content]  # type: ignore[index]
        if output_version == "responses/v1":
            assert block_types == ["web_search_call", "text"]
        else:
            assert block_types == ["server_tool_call", "server_tool_result", "text"]

async def test_web_search_async() -> None:
    llm = ChatOpenAI(model=MODEL_NAME, output_version="v0")
    response = await llm.ainvoke(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    )
    _check_response(response)
    assert response.response_metadata["status"]

    # Test streaming
    full: BaseMessageChunk | None = None
    async for chunk in llm.astream(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    ):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    _check_response(full)

    for msg in [response, full]:
        assert msg.additional_kwargs["tool_outputs"]
        assert len(msg.additional_kwargs["tool_outputs"]) == 1
        tool_output = msg.additional_kwargs["tool_outputs"][0]
        assert tool_output["type"] == "web_search_call"

def test_function_calling(output_version: Literal["v0", "responses/v1", "v1"]) -> None:
    def multiply(x: int, y: int) -> int:
        """return x * y"""
        return x * y

    llm = ChatOpenAI(model=MODEL_NAME, output_version=output_version)
    bound_llm = llm.bind_tools([multiply, {"type": "web_search_preview"}])
    ai_msg = cast(AIMessage, bound_llm.invoke("whats 5 * 4"))
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "multiply"
    assert set(ai_msg.tool_calls[0]["args"]) == {"x", "y"}

    full: Any = None
    for chunk in bound_llm.stream("whats 5 * 4"):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert len(full.tool_calls) == 1
    assert full.tool_calls[0]["name"] == "multiply"
    assert set(full.tool_calls[0]["args"]) == {"x", "y"}

    for msg in [ai_msg, full]:
        assert len(msg.content_blocks) == 1
        assert msg.content_blocks[0]["type"] == "tool_call"

    response = bound_llm.invoke("What was a positive news story from today?")
    _check_response(response)

def test_agent_loop(output_version: Literal["responses/v1", "v1"]) -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm = ChatOpenAI(
        model="gpt-5.4",
        use_responses_api=True,
        output_version=output_version,
    )
    llm_with_tools = llm.bind_tools([get_weather])
    input_message = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)
    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    response = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    assert isinstance(response, AIMessage)

def test_agent_loop_streaming(output_version: Literal["responses/v1", "v1"]) -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm = ChatOpenAI(
        model="gpt-5.2",
        use_responses_api=True,
        reasoning={"effort": "medium", "summary": "auto"},
        streaming=True,
        output_version=output_version,
    )
    llm_with_tools = llm.bind_tools([get_weather])
    input_message = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)
    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    response = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    assert isinstance(response, AIMessage)

def test_parsed_pydantic_schema(
    output_version: Literal["v0", "responses/v1", "v1"],
) -> None:
    llm = ChatOpenAI(
        model=MODEL_NAME, use_responses_api=True, output_version=output_version
    )
    response = llm.invoke("how are ya", response_format=Foo)
    parsed = Foo(**json.loads(response.text))
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed.response

    # Test stream
    full: BaseMessageChunk | None = None
    for chunk in llm.stream("how are ya", response_format=Foo):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    parsed = Foo(**json.loads(full.text))
    assert parsed == full.additional_kwargs["parsed"]
    assert parsed.response

async def test_parsed_pydantic_schema_async() -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    response = await llm.ainvoke("how are ya", response_format=Foo)
    parsed = Foo(**json.loads(response.text))
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed.response

    # Test stream
    full: BaseMessageChunk | None = None
    async for chunk in llm.astream("how are ya", response_format=Foo):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    parsed = Foo(**json.loads(full.text))
    assert parsed == full.additional_kwargs["parsed"]
    assert parsed.response

def test_parsed_dict_schema(schema: Any) -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    response = llm.invoke("how are ya", response_format=schema)
    parsed = json.loads(response.text)
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed["response"]
    assert isinstance(parsed["response"], str)

    # Test stream
    full: BaseMessageChunk | None = None
    for chunk in llm.stream("how are ya", response_format=schema):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    parsed = json.loads(full.text)
    assert parsed == full.additional_kwargs["parsed"]
    assert parsed["response"]
    assert isinstance(parsed["response"], str)

def test_parsed_strict() -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)

    class Joke(TypedDict):
        setup: Annotated[str, ..., "The setup of the joke"]
        punchline: Annotated[str, None, "The punchline of the joke"]

    schema = _convert_to_openai_response_format(Joke)
    invalid_schema = cast(dict, _convert_to_openai_response_format(Joke, strict=True))
    invalid_schema["json_schema"]["schema"]["required"] = ["setup"]  # make invalid

    # Test not strict
    response = llm.invoke("Tell me a joke", response_format=schema)
    parsed = json.loads(response.text)
    assert parsed == response.additional_kwargs["parsed"]

    # Test strict
    with pytest.raises(openai.BadRequestError):
        llm.invoke(
            "Tell me a joke about cats.", response_format=invalid_schema, strict=True
        )
    with pytest.raises(openai.BadRequestError):
        next(
            llm.stream(
                "Tell me a joke about cats.",
                response_format=invalid_schema,
                strict=True,
            )
        )

async def test_parsed_dict_schema_async(schema: Any) -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    response = await llm.ainvoke("how are ya", response_format=schema)
    parsed = json.loads(response.text)
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed["response"]
    assert isinstance(parsed["response"], str)

    # Test stream
    full: BaseMessageChunk | None = None
    async for chunk in llm.astream("how are ya", response_format=schema):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    parsed = json.loads(full.text)
    assert parsed == full.additional_kwargs["parsed"]
    assert parsed["response"]
    assert isinstance(parsed["response"], str)

def test_function_calling_and_structured_output(schema: Any) -> None:
    def multiply(x: int, y: int) -> int:
        """return x * y"""
        return x * y

    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    bound_llm = llm.bind_tools([multiply], response_format=schema, strict=True)
    # Test structured output
    response = llm.invoke("how are ya", response_format=schema)
    if schema == Foo:
        parsed = schema(**json.loads(response.text))
        assert parsed.response
    else:
        parsed = json.loads(response.text)
        assert parsed["response"]
    assert parsed == response.additional_kwargs["parsed"]

    # Test function calling
    ai_msg = cast(AIMessage, bound_llm.invoke("whats 5 * 4"))
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "multiply"
    assert set(ai_msg.tool_calls[0]["args"]) == {"x", "y"}

def test_reasoning(output_version: Literal["v0", "responses/v1", "v1"]) -> None:
    llm = ChatOpenAI(
        model="o4-mini", use_responses_api=True, output_version=output_version
    )
    response = llm.invoke("Hello", reasoning={"effort": "low"})
    assert isinstance(response, AIMessage)

    # Test init params + streaming
    llm = ChatOpenAI(
        model="o4-mini", reasoning={"effort": "low"}, output_version=output_version
    )
    full: BaseMessageChunk | None = None
    for chunk in llm.stream("Hello"):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessage)

    for msg in [response, full]:
        if output_version == "v0":
            assert msg.additional_kwargs["reasoning"]
        else:
            block_types = [block["type"] for block in msg.content]
            assert block_types == ["reasoning", "text"]

def test_stateful_api() -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    response = llm.invoke("how are you, my name is Bobo")
    assert "id" in response.response_metadata

    second_response = llm.invoke(
        "what's my name", previous_response_id=response.response_metadata["id"]
    )
    assert isinstance(second_response.content, list)
    assert "bobo" in second_response.content[0]["text"].lower()  # type: ignore

def test_route_from_model_kwargs() -> None:
    llm = ChatOpenAI(
        model=MODEL_NAME, model_kwargs={"text": {"format": {"type": "text"}}}
    )
    _ = next(llm.stream("Hello"))

def test_computer_calls() -> None:
    llm = ChatOpenAI(model="gpt-5.4")
    tool = {"type": "computer"}
    llm_with_tools = llm.bind_tools([tool], tool_choice="any")
    response = llm_with_tools.invoke("Please open the browser.")
    assert any(block["type"] == "computer_call" for block in response.content)  # type: ignore[index]

def test_file_search(
    output_version: Literal["responses/v1", "v1"],
) -> None:
    vector_store_id = os.getenv("OPENAI_VECTOR_STORE_ID")
    if not vector_store_id:
        pytest.skip()

    llm = ChatOpenAI(
        model=MODEL_NAME,
        use_responses_api=True,
        output_version=output_version,
    )
    tool = {
        "type": "file_search",
        "vector_store_ids": [vector_store_id],
    }

    input_message = {"role": "user", "content": "What is deep research by OpenAI?"}
    response = llm.invoke([input_message], tools=[tool])
    _check_response(response)

    if output_version == "v1":
        assert [block["type"] for block in response.content] == [  # type: ignore[index]
            "server_tool_call",
            "server_tool_result",
            "text",
        ]
    else:
        assert [block["type"] for block in response.content] == [  # type: ignore[index]
            "file_search_call",
            "text",
        ]

    full: AIMessageChunk | None = None
    for chunk in llm.stream([input_message], tools=[tool]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    _check_response(full)

    if output_version == "v1":
        assert [block["type"] for block in full.content] == [  # type: ignore[index]
            "server_tool_call",
            "server_tool_result",
            "text",
        ]
    else:
        assert [block["type"] for block in full.content] == ["file_search_call", "text"]  # type: ignore[index]

    next_message = {"role": "user", "content": "Thank you."}
    _ = llm.invoke([input_message, full, next_message])

    for message in [response, full]:
        assert [block["type"] for block in message.content_blocks] == [
            "server_tool_call",
            "server_tool_result",
            "text",
        ]

def test_stream_reasoning_summary(
    output_version: Literal["v0", "responses/v1", "v1"],
) -> None:
    llm = ChatOpenAI(
        model="o4-mini",
        # Routes to Responses API if `reasoning` is set.
        reasoning={"effort": "medium", "summary": "auto"},
        output_version=output_version,
    )
    message_1 = {
        "role": "user",
        "content": "What was the third tallest buliding in the year 2000?",
    }
    response_1: BaseMessageChunk | None = None
    for chunk in llm.stream([message_1]):
        assert isinstance(chunk, AIMessageChunk)
        response_1 = chunk if response_1 is None else response_1 + chunk
    assert isinstance(response_1, AIMessageChunk)
    if output_version == "v0":
        reasoning = response_1.additional_kwargs["reasoning"]
        assert set(reasoning.keys()) == {"id", "type", "summary"}
        summary = reasoning["summary"]
        assert isinstance(summary, list)
        for block in summary:
            assert isinstance(block, dict)
            assert isinstance(block["type"], str)
            assert isinstance(block["text"], str)
            assert block["text"]
    elif output_version == "responses/v1":
        reasoning = next(
            block
            for block in response_1.content
            if block["type"] == "reasoning"  # type: ignore[index]
        )
        if isinstance(reasoning, str):
            reasoning = json.loads(reasoning)
        assert set(reasoning.keys()) == {"id", "type", "summary", "index"}
        summary = reasoning["summary"]
        assert isinstance(summary, list)
        for block in summary:
            assert isinstance(block, dict)
            assert isinstance(block["type"], str)
            assert isinstance(block["text"], str)
            assert block["text"]
    else:
        # v1
        total_reasoning_blocks = 0
        for block in response_1.content_blocks:
            if block["type"] == "reasoning":
                total_reasoning_blocks += 1
                assert isinstance(block.get("id"), str)
                assert block.get("id", "").startswith("rs_")
                assert isinstance(block.get("reasoning"), str)
                assert isinstance(block.get("index"), str)
        assert (
            total_reasoning_blocks > 1
        )  # This query typically generates multiple reasoning blocks

    # Check we can pass back summaries
    message_2 = {"role": "user", "content": "Thank you."}
    response_2 = llm.invoke([message_1, response_1, message_2])
    assert isinstance(response_2, AIMessage)

def test_code_interpreter(output_version: Literal["v0", "responses/v1", "v1"]) -> None:
    llm = ChatOpenAI(
        model="o4-mini", use_responses_api=True, output_version=output_version
    )
    llm_with_tools = llm.bind_tools(
        [{"type": "code_interpreter", "container": {"type": "auto"}}]
    )
    input_message = {
        "role": "user",
        "content": "Write and run code to answer the question: what is 3^3?",
    }
    response = llm_with_tools.invoke([input_message])
    assert isinstance(response, AIMessage)
    _check_response(response)
    if output_version == "v0":
        tool_outputs = [
            item
            for item in response.additional_kwargs["tool_outputs"]
            if item["type"] == "code_interpreter_call"
        ]
        assert len(tool_outputs) == 1
    elif output_version == "responses/v1":
        tool_outputs = [
            item
            for item in response.content
            if isinstance(item, dict) and item["type"] == "code_interpreter_call"
        ]
        assert len(tool_outputs) == 1
    else:
        # v1
        tool_outputs = [
            item
            for item in response.content_blocks
            if item["type"] == "server_tool_call" and item["name"] == "code_interpreter"
        ]
        code_interpreter_result = next(
            item
            for item in response.content_blocks
            if item["type"] == "server_tool_result"
        )
        assert tool_outputs
        assert code_interpreter_result
    assert len(tool_outputs) == 1

    # Test streaming
    # Use same container
    container_id = tool_outputs[0].get("container_id") or tool_outputs[0].get(
        "extras", {}
    ).get("container_id")
    llm_with_tools = llm.bind_tools(
        [{"type": "code_interpreter", "container": container_id}]
    )

    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    if output_version == "v0":
        tool_outputs = [
            item
            for item in response.additional_kwargs["tool_outputs"]
            if item["type"] == "code_interpreter_call"
        ]
        assert tool_outputs
    elif output_version == "responses/v1":
        tool_outputs = [
            item
            for item in response.content
            if isinstance(item, dict) and item["type"] == "code_interpreter_call"
        ]
        assert tool_outputs
    else:
        # v1
        code_interpreter_call = next(
            item
            for item in full.content_blocks
            if item["type"] == "server_tool_call" and item["name"] == "code_interpreter"
        )
        code_interpreter_result = next(
            item for item in full.content_blocks if item["type"] == "server_tool_result"
        )
        assert code_interpreter_call
        assert code_interpreter_result

    # Test we can pass back in
    next_message = {"role": "user", "content": "Please add more comments to the code."}
    _ = llm_with_tools.invoke([input_message, full, next_message])

def test_mcp_builtin() -> None:
    llm = ChatOpenAI(model="o4-mini", use_responses_api=True, output_version="v0")

    llm_with_tools = llm.bind_tools(
        [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": {"always": {"tool_names": ["read_wiki_structure"]}},
            }
        ]
    )
    input_message = {
        "role": "user",
        "content": (
            "What transport protocols does the 2025-03-26 version of the MCP spec "
            "support?"
        ),
    }
    response = llm_with_tools.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)

    approval_message = HumanMessage(
        [
            {
                "type": "mcp_approval_response",
                "approve": True,
                "approval_request_id": output["id"],
            }
            for output in response.additional_kwargs["tool_outputs"]
            if output["type"] == "mcp_approval_request"
        ]
    )
    _ = llm_with_tools.invoke(
        [approval_message], previous_response_id=response.response_metadata["id"]
    )

def test_mcp_builtin_zdr() -> None:
    llm = ChatOpenAI(
        model="gpt-5-nano",
        use_responses_api=True,
        store=False,
        include=["reasoning.encrypted_content"],
    )

    llm_with_tools = llm.bind_tools(
        [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "allowed_tools": ["ask_question"],
                "require_approval": "always",
            }
        ]
    )
    input_message = {
        "role": "user",
        "content": (
            "What transport protocols does the 2025-03-26 version of the MCP "
            "spec (modelcontextprotocol/modelcontextprotocol) support?"
        ),
    }
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert all(isinstance(block, dict) for block in full.content)

    approval_message = HumanMessage(
        [
            {
                "type": "mcp_approval_response",
                "approve": True,
                "approval_request_id": block["id"],  # type: ignore[index]
            }
            for block in full.content
            if block["type"] == "mcp_approval_request"  # type: ignore[index]
        ]
    )
    result = llm_with_tools.invoke([input_message, full, approval_message])
    next_message = {"role": "user", "content": "Thanks!"}
    _ = llm_with_tools.invoke(
        [input_message, full, approval_message, result, next_message]
    )

def test_mcp_builtin_zdr_v1() -> None:
    llm = ChatOpenAI(
        model="gpt-5-nano",
        output_version="v1",
        store=False,
        include=["reasoning.encrypted_content"],
    )

    llm_with_tools = llm.bind_tools(
        [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "allowed_tools": ["ask_question"],
                "require_approval": "always",
            }
        ]
    )
    input_message = {
        "role": "user",
        "content": (
            "What transport protocols does the 2025-03-26 version of the MCP "
            "spec (modelcontextprotocol/modelcontextprotocol) support?"
        ),
    }
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert all(isinstance(block, dict) for block in full.content)

    approval_message = HumanMessage(
        [
            {
                "type": "non_standard",
                "value": {
                    "type": "mcp_approval_response",
                    "approve": True,
                    "approval_request_id": block["value"]["id"],  # type: ignore[index]
                },
            }
            for block in full.content_blocks
            if block["type"] == "non_standard"
            and block["value"]["type"] == "mcp_approval_request"  # type: ignore[index]
        ]
    )
    result = llm_with_tools.invoke([input_message, full, approval_message])
    next_message = {"role": "user", "content": "Thanks!"}
    _ = llm_with_tools.invoke(
        [input_message, full, approval_message, result, next_message]
    )

def test_image_generation_streaming(
    output_version: Literal["v0", "responses/v1"],
) -> None:
    """Test image generation streaming."""
    llm = ChatOpenAI(
        model="gpt-4.1", use_responses_api=True, output_version=output_version
    )
    tool = {
        "type": "image_generation",
        # For testing purposes let's keep the quality low, so the test runs faster.
        "quality": "low",
        "output_format": "jpeg",
        "output_compression": 100,
        "size": "1024x1024",
    }

    # Example tool output for an image
    # {
    #     "background": "opaque",
    #     "id": "ig_683716a8ddf0819888572b20621c7ae4029ec8c11f8dacf8",
    #     "output_format": "png",
    #     "quality": "high",
    #     "revised_prompt": "A fluffy, fuzzy cat sitting calmly, with soft fur, bright "
    #     "eyes, and a cute, friendly expression. The background is "
    #     "simple and light to emphasize the cat's texture and "
    #     "fluffiness.",
    #     "size": "1024x1024",
    #     "status": "completed",
    #     "type": "image_generation_call",
    #     "result": # base64 encode image data
    # }

    expected_keys = {
        "id",
        "index",
        "background",
        "output_format",
        "quality",
        "result",
        "revised_prompt",
        "size",
        "status",
        "type",
    }

    full: BaseMessageChunk | None = None
    for chunk in llm.stream("Draw a random short word in green font.", tools=[tool]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    complete_ai_message = cast(AIMessageChunk, full)
    # At the moment, the streaming API does not pick up annotations fully.
    # So the following check is commented out.
    # _check_response(complete_ai_message)
    if output_version == "v0":
        assert complete_ai_message.additional_kwargs["tool_outputs"]
        tool_output = complete_ai_message.additional_kwargs["tool_outputs"][0]
        assert set(tool_output.keys()).issubset(expected_keys)
    else:
        # "responses/v1"
        tool_output = next(
            block
            for block in complete_ai_message.content
            if isinstance(block, dict) and block["type"] == "image_generation_call"
        )
        assert set(tool_output.keys()).issubset(expected_keys)

def test_image_generation_streaming_v1() -> None:
    """Test image generation streaming."""
    llm = ChatOpenAI(model="gpt-4.1", use_responses_api=True, output_version="v1")
    tool = {
        "type": "image_generation",
        "quality": "low",
        "output_format": "jpeg",
        "output_compression": 100,
        "size": "1024x1024",
    }

    standard_keys = {"type", "base64", "mime_type", "id", "index"}
    extra_keys = {
        "background",
        "output_format",
        "quality",
        "revised_prompt",
        "size",
        "status",
    }

    full: BaseMessageChunk | None = None
    for chunk in llm.stream("Draw a random short word in green font.", tools=[tool]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    complete_ai_message = cast(AIMessageChunk, full)

    tool_output = next(
        block
        for block in complete_ai_message.content
        if isinstance(block, dict) and block["type"] == "image"
    )
    assert set(standard_keys).issubset(tool_output.keys())
    assert set(extra_keys).issubset(tool_output["extras"].keys())

def test_image_generation_multi_turn(
    output_version: Literal["v0", "responses/v1"],
) -> None:
    """Test multi-turn editing of image generation by passing in history."""
    # Test multi-turn
    llm = ChatOpenAI(
        model="gpt-4.1", use_responses_api=True, output_version=output_version
    )
    # Test invocation
    tool = {
        "type": "image_generation",
        # For testing purposes let's keep the quality low, so the test runs faster.
        "quality": "low",
        "output_format": "jpeg",
        "output_compression": 100,
        "size": "1024x1024",
    }
    llm_with_tools = llm.bind_tools([tool])

    chat_history: list[MessageLikeRepresentation] = [
        {"role": "user", "content": "Draw a random short word in green font."}
    ]
    ai_message = llm_with_tools.invoke(chat_history)
    assert isinstance(ai_message, AIMessage)
    _check_response(ai_message)

    expected_keys = {
        "id",
        "background",
        "output_format",
        "quality",
        "result",
        "revised_prompt",
        "size",
        "status",
        "type",
    }

    if output_version == "v0":
        tool_output = ai_message.additional_kwargs["tool_outputs"][0]
        assert set(tool_output.keys()).issubset(expected_keys)
    elif output_version == "responses/v1":
        tool_output = next(
            block
            for block in ai_message.content
            if isinstance(block, dict) and block["type"] == "image_generation_call"
        )
        assert set(tool_output.keys()).issubset(expected_keys)
    else:
        standard_keys = {"type", "base64", "id", "status"}
        tool_output = next(
            block
            for block in ai_message.content
            if isinstance(block, dict) and block["type"] == "image"
        )
        assert set(standard_keys).issubset(tool_output.keys())

    # Example tool output for an image (v0)
    # {
    #     "background": "opaque",
    #     "id": "ig_683716a8ddf0819888572b20621c7ae4029ec8c11f8dacf8",
    #     "output_format": "png",
    #     "quality": "high",
    #     "revised_prompt": "A fluffy, fuzzy cat sitting calmly, with soft fur, bright "
    #     "eyes, and a cute, friendly expression. The background is "
    #     "simple and light to emphasize the cat's texture and "
    #     "fluffiness.",
    #     "size": "1024x1024",
    #     "status": "completed",
    #     "type": "image_generation_call",
    #     "result": # base64 encode image data
    # }

    chat_history.extend(
        [
            # AI message with tool output
            ai_message,
            # New request
            {
                "role": "user",
                "content": (
                    "Now, change the font to blue. Keep the word and everything else "
                    "the same."
                ),
            },
        ]
    )

    ai_message2 = llm_with_tools.invoke(chat_history)
    assert isinstance(ai_message2, AIMessage)
    _check_response(ai_message2)

    if output_version == "v0":
        tool_output = ai_message2.additional_kwargs["tool_outputs"][0]
        assert set(tool_output.keys()).issubset(expected_keys)
    else:
        # "responses/v1"
        tool_output = next(
            block
            for block in ai_message2.content
            if isinstance(block, dict) and block["type"] == "image_generation_call"
        )
        assert set(tool_output.keys()).issubset(expected_keys)

def test_image_generation_multi_turn_v1() -> None:
    """Test multi-turn editing of image generation by passing in history."""
    # Test multi-turn
    llm = ChatOpenAI(model="gpt-4.1", use_responses_api=True, output_version="v1")
    # Test invocation
    tool = {
        "type": "image_generation",
        "quality": "low",
        "output_format": "jpeg",
        "output_compression": 100,
        "size": "1024x1024",
    }
    llm_with_tools = llm.bind_tools([tool])

    chat_history: list[MessageLikeRepresentation] = [
        {"role": "user", "content": "Draw a random short word in green font."}
    ]
    ai_message = llm_with_tools.invoke(chat_history)
    assert isinstance(ai_message, AIMessage)
    _check_response(ai_message)

    standard_keys = {"type", "base64", "mime_type", "id"}
    extra_keys = {
        "background",
        "output_format",
        "quality",
        "revised_prompt",
        "size",
        "status",
    }

    tool_output = next(
        block
        for block in ai_message.content
        if isinstance(block, dict) and block["type"] == "image"
    )
    assert set(standard_keys).issubset(tool_output.keys())
    assert set(extra_keys).issubset(tool_output["extras"].keys())

    chat_history.extend(
        [
            # AI message with tool output
            ai_message,
            # New request
            {
                "role": "user",
                "content": (
                    "Now, change the font to blue. Keep the word and everything else "
                    "the same."
                ),
            },
        ]
    )

    ai_message2 = llm_with_tools.invoke(chat_history)
    assert isinstance(ai_message2, AIMessage)
    _check_response(ai_message2)

    tool_output = next(
        block
        for block in ai_message2.content
        if isinstance(block, dict) and block["type"] == "image"
    )
    assert set(standard_keys).issubset(tool_output.keys())
    assert set(extra_keys).issubset(tool_output["extras"].keys())

def test_verbosity_parameter() -> None:
    """Test verbosity parameter with Responses API.

    Tests that the verbosity parameter works correctly with the OpenAI Responses API.

    """
    llm = ChatOpenAI(model=MODEL_NAME, verbosity="medium", use_responses_api=True)
    response = llm.invoke([HumanMessage(content="Hello, explain quantum computing.")])

    assert isinstance(response, AIMessage)
    assert response.content

def test_custom_tool(output_version: Literal["responses/v1", "v1"]) -> None:
    @custom_tool
    def execute_code(code: str) -> str:
        """Execute python code."""
        return "27"

    llm = ChatOpenAI(model="gpt-5", output_version=output_version).bind_tools(
        [execute_code]
    )

    input_message = {"role": "user", "content": "Use the tool to evaluate 3^3."}
    tool_call_message = llm.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)
    assert len(tool_call_message.tool_calls) == 1
    tool_call = tool_call_message.tool_calls[0]
    tool_message = execute_code.invoke(tool_call)
    response = llm.invoke([input_message, tool_call_message, tool_message])
    assert isinstance(response, AIMessage)

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert len(full.tool_calls) == 1

def test_compaction(output_version: Literal["responses/v1", "v1"]) -> None:
    """Test the compaction beta feature."""
    llm = ChatOpenAI(
        model="gpt-5.2",
        context_management=[{"type": "compaction", "compact_threshold": 10_000}],
        output_version=output_version,
    )

    input_message = {
        "role": "user",
        "content": f"Generate a one-sentence summary of this:\n\n{'a' * 50000}",
    }
    messages: list = [input_message]

    first_response = llm.invoke(messages)
    messages.append(first_response)

    second_message = {
        "role": "user",
        "content": f"Generate a one-sentence summary of this:\n\n{'b' * 50000}",
    }
    messages.append(second_message)

    second_response = llm.invoke(messages)
    messages.append(second_response)

    content_blocks = second_response.content_blocks
    compaction_block = next(
        (block for block in content_blocks if block["type"] == "non_standard"),
        None,
    )
    assert compaction_block
    assert compaction_block["value"].get("type") == "compaction"

    third_message = {
        "role": "user",
        "content": "What are we talking about?",
    }
    messages.append(third_message)
    third_response = llm.invoke(messages)
    assert third_response.text

def test_compaction_streaming(output_version: Literal["responses/v1", "v1"]) -> None:
    """Test the compaction beta feature."""
    llm = ChatOpenAI(
        model="gpt-5.2",
        context_management=[{"type": "compaction", "compact_threshold": 10_000}],
        output_version=output_version,
        streaming=True,
    )

    input_message = {
        "role": "user",
        "content": f"Generate a one-sentence summary of this:\n\n{'a' * 50000}",
    }
    messages: list = [input_message]

    first_response = llm.invoke(messages)
    messages.append(first_response)

    second_message = {
        "role": "user",
        "content": f"Generate a one-sentence summary of this:\n\n{'b' * 50000}",
    }
    messages.append(second_message)

    second_response = llm.invoke(messages)
    messages.append(second_response)

    content_blocks = second_response.content_blocks
    compaction_block = next(
        (block for block in content_blocks if block["type"] == "non_standard"),
        None,
    )
    assert compaction_block
    assert compaction_block["value"].get("type") == "compaction"

    third_message = {
        "role": "user",
        "content": "What are we talking about?",
    }
    messages.append(third_message)
    third_response = llm.invoke(messages)
    assert third_response.text

def test_csv_input() -> None:
    """Test CSV file input with both LangChain standard and OpenAI native formats."""
    # Create sample CSV content
    csv_content = (
        "name,age,city\nAlice,30,New York\nBob,25,Los Angeles\nCarol,35,Chicago"
    )
    csv_bytes = csv_content.encode("utf-8")
    base64_string = base64.b64encode(csv_bytes).decode("utf-8")

    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)

    # Test LangChain standard format
    langchain_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "How many people are in this CSV file?",
            },
            {
                "type": "file",
                "base64": base64_string,
                "mime_type": "text/csv",
                "filename": "people.csv",
            },
        ],
    }
    payload = llm._get_request_payload([langchain_message])
    block = payload["input"][0]["content"][1]
    assert block["type"] == "input_file"

    response = llm.invoke([langchain_message])
    assert isinstance(response, AIMessage)
    assert response.content
    assert (
        "3" in str(response.content).lower() or "three" in str(response.content).lower()
    )

    # Test OpenAI native format
    openai_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "How many people are in this CSV file?",
            },
            {
                "type": "input_file",
                "filename": "people.csv",
                "file_data": f"data:text/csv;base64,{base64_string}",
            },
        ],
    }
    payload2 = llm._get_request_payload([openai_message])
    block2 = payload2["input"][0]["content"][1]
    assert block2["type"] == "input_file"

    response2 = llm.invoke([openai_message])
    assert isinstance(response2, AIMessage)
    assert response2.content
    assert (
        "3" in str(response2.content).lower()
        or "three" in str(response2.content).lower()
    )

def test_phase(output_version: str) -> None:
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "It's sunny."

    model = ChatOpenAI(
        model="gpt-5.4",
        use_responses_api=True,
        verbosity="high",
        reasoning={"effort": "medium", "summary": "auto"},
        output_version=output_version,
    )

    agent = create_agent(model, tools=[get_weather])

    input_message = {
        "role": "user",
        "content": (
            "What's the weather in the oldest major city in the US? State your answer "
            "and then generate a tool call this turn."
        ),
    }
    result = agent.invoke({"messages": [input_message]})
    first_response = result["messages"][1]
    text_block = next(
        block for block in first_response.content if block["type"] == "text"
    )
    assert text_block["phase"] == "commentary"

    final_response = result["messages"][-1]
    text_block = next(
        block for block in final_response.content if block["type"] == "text"
    )
    assert text_block["phase"] == "final_answer"

def test_phase_streaming(output_version: str) -> None:
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "It's sunny."

    model = ChatOpenAI(
        model="gpt-5.4",
        use_responses_api=True,
        verbosity="high",
        reasoning={"effort": "medium", "summary": "auto"},
        streaming=True,
        output_version=output_version,
    )

    agent = create_agent(model, tools=[get_weather])

    input_message = {
        "role": "user",
        "content": (
            "What's the weather in the oldest major city in the US? State your answer "
            "and then generate a tool call this turn."
        ),
    }
    result = agent.invoke({"messages": [input_message]})
    first_response = result["messages"][1]
    if output_version == "responses/v1":
        assert [block["type"] for block in first_response.content] == [
            "reasoning",
            "text",
            "function_call",
        ]
    else:
        assert [block["type"] for block in first_response.content] == [
            "reasoning",
            "text",
            "tool_call",
        ]
    text_block = next(
        block for block in first_response.content if block["type"] == "text"
    )
    assert text_block["phase"] == "commentary"

    final_response = result["messages"][-1]
    assert [block["type"] for block in final_response.content] == ["text"]
    text_block = next(
        block for block in final_response.content if block["type"] == "text"
    )
    assert text_block["phase"] == "final_answer"

def test_tool_search(output_version: str) -> None:
    @tool(extras={"defer_loading": True})
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny and 72°F"

    @tool(extras={"defer_loading": True})
    def get_recipe(query: str) -> None:
        """Get a recipe for chicken soup."""

    model = ChatOpenAI(
        model="gpt-5.4",
        use_responses_api=True,
        output_version=output_version,
    )

    agent = create_agent(
        model=model,
        tools=[get_weather, get_recipe, {"type": "tool_search"}],
    )
    input_message = {"role": "user", "content": "What's the weather in San Francisco?"}
    result = agent.invoke({"messages": [input_message]})
    assert len(result["messages"]) == 4
    tool_call_message = result["messages"][1]
    assert isinstance(tool_call_message, AIMessage)
    assert tool_call_message.tool_calls
    if output_version == "v1":
        assert [block["type"] for block in tool_call_message.content] == [  # type: ignore[index]
            "server_tool_call",
            "server_tool_result",
            "tool_call",
        ]
    else:
        assert [block["type"] for block in tool_call_message.content] == [  # type: ignore[index]
            "tool_search_call",
            "tool_search_output",
            "function_call",
        ]

    assert isinstance(result["messages"][2], ToolMessage)

    assert result["messages"][3].text

def test_tool_search_streaming(output_version: str) -> None:
    @tool(extras={"defer_loading": True})
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny and 72°F"

    @tool(extras={"defer_loading": True})
    def get_recipe(query: str) -> None:
        """Get a recipe for chicken soup."""

    model = ChatOpenAI(
        model="gpt-5.4",
        use_responses_api=True,
        streaming=True,
        output_version=output_version,
    )

    agent = create_agent(
        model=model,
        tools=[get_weather, get_recipe, {"type": "tool_search"}],
    )
    input_message = {"role": "user", "content": "What's the weather in San Francisco?"}
    result = agent.invoke({"messages": [input_message]})
    assert len(result["messages"]) == 4
    tool_call_message = result["messages"][1]
    assert isinstance(tool_call_message, AIMessage)
    assert tool_call_message.tool_calls
    if output_version == "v1":
        assert [block["type"] for block in tool_call_message.content] == [  # type: ignore[index]
            "server_tool_call",
            "server_tool_result",
            "tool_call",
        ]
    else:
        assert [block["type"] for block in tool_call_message.content] == [  # type: ignore[index]
            "tool_search_call",
            "tool_search_output",
            "function_call",
        ]

    assert isinstance(result["messages"][2], ToolMessage)

    assert result["messages"][3].text

def test_client_executed_tool_search() -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny and 72°F"

    def search_tools(goal: str) -> list[dict]:
        """Search for available tools to help answer the question."""
        return [
            {
                "type": "function",
                "defer_loading": True,
                **convert_to_openai_tool(get_weather)["function"],
            }
        ]

    tool_search_schema = convert_to_openai_tool(search_tools, strict=True)
    tool_search_config: dict = {
        "type": "tool_search",
        "execution": "client",
        "description": tool_search_schema["function"]["description"],
        "parameters": tool_search_schema["function"]["parameters"],
    }

    class ClientToolSearchMiddleware(AgentMiddleware):
        @hook_config(can_jump_to=["model"])
        def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage):
                return None
            for block in last_message.content:
                if isinstance(block, dict) and block.get("type") == "tool_search_call":
                    call_id = block.get("call_id")
                    args = block.get("arguments", {})
                    goal = args.get("goal", "") if isinstance(args, dict) else ""
                    loaded_tools = search_tools(goal)
                    tool_search_output = {
                        "type": "tool_search_output",
                        "execution": "client",
                        "call_id": call_id,
                        "status": "completed",
                        "tools": loaded_tools,
                    }
                    return {
                        "messages": [HumanMessage(content=[tool_search_output])],
                        "jump_to": "model",
                    }
            return None

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Any,
        ) -> Any:
            if request.tool_call["name"] == "get_weather":
                return handler(request.override(tool=get_weather))
            return handler(request)

    llm = ChatOpenAI(model="gpt-5.4", use_responses_api=True)

    agent = create_agent(
        model=llm,
        tools=[tool_search_config],
        middleware=[ClientToolSearchMiddleware()],
    )

    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather in San Francisco?")]}
    )
    messages = result["messages"]
    search_tool_call = messages[1]
    assert search_tool_call.content[0]["type"] == "tool_search_call"

    search_tool_output = messages[2]
    assert search_tool_output.content[0]["type"] == "tool_search_output"

    tool_call = messages[3]
    assert tool_call.tool_calls

    assert isinstance(messages[4], ToolMessage)

    assert messages[5].text


# --- libs/partners/openai/tests/integration_tests/chat_models/test_responses_standard.py ---

    def test_openai_pdf_inputs(self, model: BaseChatModel) -> None:
        """Test that the model can process PDF inputs."""
        # Responses API additionally supports files via URL
        url = "https://www.berkshirehathaway.com/letters/2024ltr.pdf"

        message = HumanMessage(
            [
                {"type": "text", "text": "What is the document title, verbatim?"},
                {"type": "file", "url": url},
            ]
        )
        _ = model.invoke([message])

        # Test OpenAI Responses format
        message = HumanMessage(
            [
                {"type": "text", "text": "What is the document title, verbatim?"},
                {"type": "input_file", "file_url": url},
            ]
        )
        _ = model.invoke([message])


# --- libs/partners/openai/tests/integration_tests/llms/test_azure.py ---

def test_openai_call(llm: AzureOpenAI) -> None:
    """Test valid call to openai."""
    output = llm.invoke("Say something nice:")
    assert isinstance(output, str)

def test_openai_streaming(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    full_response = ""
    for token in generator:
        assert isinstance(token, str)
        full_response += token
    assert full_response

async def test_openai_astream(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)

async def test_openai_ainvoke(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)

def test_openai_invoke(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    result = llm.invoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)

def test_openai_multiple_prompts(llm: AzureOpenAI) -> None:
    """Test completion with multiple prompts."""
    output = llm.generate(["I'm Pickle Rick", "I'm Pickle Rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2

def test_openai_streaming_best_of_error() -> None:
    """Test validation for streaming fails if best_of is not 1."""
    with pytest.raises(ValueError):
        _get_llm(best_of=2, streaming=True)

def test_openai_streaming_n_error() -> None:
    """Test validation for streaming fails if n is not 1."""
    with pytest.raises(ValueError):
        _get_llm(n=2, streaming=True)

def test_openai_streaming_multiple_prompts_error() -> None:
    """Test validation for streaming fails if multiple prompts are given."""
    with pytest.raises(ValueError):
        _get_llm(streaming=True).generate(["I'm Pickle Rick", "I'm Pickle Rick"])

def test_openai_streaming_call() -> None:
    """Test valid call to openai."""
    llm = _get_llm(max_tokens=10, streaming=True)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

async def test_openai_async_generate() -> None:
    """Test async generation."""
    llm = _get_llm(max_tokens=10)
    output = await llm.agenerate(["Hello, how are you?"])
    assert isinstance(output, LLMResult)


# --- libs/partners/openai/tests/integration_tests/llms/test_base.py ---

def test_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token, str)

async def test_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)

async def test_ainvoke() -> None:
    """Test invoke tokens from OpenAI."""
    llm = OpenAI()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)

def test_invoke() -> None:
    """Test invoke tokens from OpenAI."""
    llm = OpenAI()

    result = llm.invoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)

def test_openai_call() -> None:
    """Test valid call to openai."""
    llm = OpenAI()
    output = llm.invoke("Say something nice:")
    assert isinstance(output, str)

def test_openai_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    llm = OpenAI(max_tokens=10)
    llm_result = llm.generate(["Hello, how are you?"])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == llm.model_name

def test_openai_stop_valid() -> None:
    """Test openai stop logic on valid configuration."""
    query = "write an ordered list of five items"
    first_llm = OpenAI(stop="3", temperature=0)  # type: ignore[call-arg]
    first_output = first_llm.invoke(query)
    second_llm = OpenAI(temperature=0)
    second_output = second_llm.invoke(query, stop=["3"])
    # Because it stops on new lines, shouldn't return anything
    assert first_output == second_output

def test_openai_streaming() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)

async def test_openai_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)

async def test_openai_ainvoke() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)

def test_openai_invoke() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)

    result = llm.invoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)

def test_openai_multiple_prompts() -> None:
    """Test completion with multiple prompts."""
    llm = OpenAI(max_tokens=10)
    output = llm.generate(["I'm Pickle Rick", "I'm Pickle Rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2

def test_openai_streaming_best_of_error() -> None:
    """Test validation for streaming fails if best_of is not 1."""
    with pytest.raises(ValueError):
        OpenAI(best_of=2, streaming=True)

def test_openai_streaming_n_error() -> None:
    """Test validation for streaming fails if n is not 1."""
    with pytest.raises(ValueError):
        OpenAI(n=2, streaming=True)

def test_openai_streaming_multiple_prompts_error() -> None:
    """Test validation for streaming fails if multiple prompts are given."""
    with pytest.raises(ValueError):
        OpenAI(streaming=True).generate(["I'm Pickle Rick", "I'm Pickle Rick"])

def test_openai_streaming_call() -> None:
    """Test valid call to openai."""
    llm = OpenAI(max_tokens=10, streaming=True)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

async def test_openai_async_generate() -> None:
    """Test async generation."""
    llm = OpenAI(max_tokens=10)
    output = await llm.agenerate(["Hello, how are you?"])
    assert isinstance(output, LLMResult)


# --- libs/partners/openai/tests/unit_tests/chat_models/test_azure.py ---

def test_initialize_more() -> None:
    llm = AzureChatOpenAI(  # type: ignore[call-arg]
        api_key="xyz",  # type: ignore[arg-type]
        azure_endpoint="my-base-url",
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
        temperature=0,
        model="gpt-35-turbo",
        model_version="0125",
    )
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "xyz"
    assert llm.azure_endpoint == "my-base-url"
    assert llm.deployment_name == "35-turbo-dev"
    assert llm.openai_api_version == "2023-05-15"
    assert llm.temperature == 0
    assert llm.stream_usage

    ls_params = llm._get_ls_params()
    assert ls_params.get("ls_provider") == "azure"
    assert ls_params.get("ls_model_name") == "gpt-35-turbo-0125"

def test_max_completion_tokens_in_payload() -> None:
    llm = AzureChatOpenAI(
        azure_deployment="o1-mini",
        api_version="2024-12-01-preview",
        azure_endpoint="my-base-url",
        model_kwargs={"max_completion_tokens": 300},
    )
    messages = [HumanMessage("Hello")]
    payload = llm._get_request_payload(messages)
    assert payload == {
        "messages": [{"content": "Hello", "role": "user"}],
        "model": None,
        "stream": False,
        "max_completion_tokens": 300,
    }

def test_max_completion_tokens_parameter() -> None:
    """Test that max_completion_tokens can be used as a direct parameter."""
    llm = AzureChatOpenAI(
        azure_deployment="gpt-5",
        api_version="2024-12-01-preview",
        azure_endpoint="my-base-url",
        max_completion_tokens=1500,
    )
    messages = [HumanMessage("Hello")]
    payload = llm._get_request_payload(messages)

    # Should use max_completion_tokens instead of max_tokens
    assert "max_completion_tokens" in payload
    assert payload["max_completion_tokens"] == 1500
    assert "max_tokens" not in payload


# --- libs/partners/openai/tests/unit_tests/chat_models/test_base.py ---

def test_openai_model_param() -> None:
    llm = ChatOpenAI(model="foo")
    assert llm.model_name == "foo"
    assert llm.model == "foo"
    llm = ChatOpenAI(model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"
    assert llm.model == "foo"

    llm = ChatOpenAI(max_tokens=10)  # type: ignore[call-arg]
    assert llm.max_tokens == 10
    llm = ChatOpenAI(max_completion_tokens=10)
    assert llm.max_tokens == 10

def test_streaming_attribute_should_stream(async_api: bool) -> None:
    llm = ChatOpenAI(model="foo", streaming=True)
    assert llm._should_stream(async_api=async_api)

def test__convert_dict_to_message_tool_call() -> None:
    raw_tool_call = {
        "id": "call_wm0JY6CdwOMZ4eTxHWUThDNz",
        "function": {
            "arguments": '{"name": "Sally", "hair_color": "green"}',
            "name": "GenerateUsername",
        },
        "type": "function",
    }
    message = {"role": "assistant", "content": None, "tool_calls": [raw_tool_call]}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="call_wm0JY6CdwOMZ4eTxHWUThDNz",
                type="tool_call",
            )
        ],
    )
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message

    # Test malformed tool call
    raw_tool_calls: list = [
        {
            "id": "call_wm0JY6CdwOMZ4eTxHWUThDNz",
            "function": {"arguments": "oops", "name": "GenerateUsername"},
            "type": "function",
        },
        {
            "id": "call_abc123",
            "function": {
                "arguments": '{"name": "Sally", "hair_color": "green"}',
                "name": "GenerateUsername",
            },
            "type": "function",
        },
    ]
    raw_tool_calls = sorted(raw_tool_calls, key=lambda x: x["id"])
    message = {"role": "assistant", "content": None, "tool_calls": raw_tool_calls}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="",
        invalid_tool_calls=[
            InvalidToolCall(
                name="GenerateUsername",
                args="oops",
                id="call_wm0JY6CdwOMZ4eTxHWUThDNz",
                error=(
                    "Function GenerateUsername arguments:\n\noops\n\nare not "
                    "valid JSON. Received JSONDecodeError Expecting value: line 1 "
                    "column 1 (char 0)\nFor troubleshooting, visit: https://docs"
                    ".langchain.com/oss/python/langchain/errors/OUTPUT_PARSING_FAILURE "
                ),
                type="invalid_tool_call",
            )
        ],
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="call_abc123",
                type="tool_call",
            )
        ],
    )
    assert result == expected_output
    reverted_message_dict = _convert_message_to_dict(expected_output)
    reverted_message_dict["tool_calls"] = sorted(
        reverted_message_dict["tool_calls"], key=lambda x: x["id"]
    )
    assert reverted_message_dict == message

def test_bind_tools_tool_choice(tool_choice: Any, strict: bool | None) -> None:
    """Test passing in manually construct tool call message."""
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm.bind_tools(
        tools=[GenerateUsername, MakeASandwich], tool_choice=tool_choice, strict=strict
    )

def test_with_structured_output(
    schema: type | dict[str, Any] | None,
    method: Literal["function_calling", "json_mode", "json_schema"],
    include_raw: bool,
    strict: bool | None,
) -> None:
    """Test passing in manually construct tool call message."""
    if method == "json_mode":
        strict = None
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm.with_structured_output(
        schema, method=method, strict=strict, include_raw=include_raw
    )

def test_minimal_reasoning_effort_payload(
    use_max_completion_tokens: bool, use_responses_api: bool
) -> None:
    """Test that minimal reasoning effort is included in request payload."""
    if use_max_completion_tokens:
        kwargs = {"max_completion_tokens": 100}
    else:
        kwargs = {"max_tokens": 100}

    init_kwargs: dict[str, Any] = {
        "model": "gpt-5",
        "reasoning_effort": "minimal",
        "use_responses_api": use_responses_api,
        **kwargs,
    }

    llm = ChatOpenAI(**init_kwargs)

    messages = [
        {"role": "developer", "content": "respond with just 'test'"},
        {"role": "user", "content": "hello"},
    ]

    payload = llm._get_request_payload(messages, stop=None)

    # When using responses API, reasoning_effort becomes reasoning.effort
    if use_responses_api:
        assert "reasoning" in payload
        assert payload["reasoning"]["effort"] == "minimal"
        # For responses API, tokens param becomes max_output_tokens
        assert payload["max_output_tokens"] == 100
    else:
        # For non-responses API, reasoning_effort remains as is
        assert payload["reasoning_effort"] == "minimal"
        if use_max_completion_tokens:
            assert payload["max_completion_tokens"] == 100
        else:
            # max_tokens gets converted to max_completion_tokens in non-responses API
            assert payload["max_completion_tokens"] == 100

def test_structured_outputs_parser() -> None:
    parsed_response = GenerateUsername(name="alice", hair_color="black")
    llm_output = ChatGeneration(
        message=AIMessage(
            content='{"name": "alice", "hair_color": "black"}',
            additional_kwargs={"parsed": parsed_response},
        )
    )
    output_parser = RunnableLambda(
        partial(_oai_structured_outputs_parser, schema=GenerateUsername)
    )
    serialized = dumps(llm_output)
    deserialized = loads(serialized, allowed_objects=[ChatGeneration, AIMessage])
    assert isinstance(deserialized, ChatGeneration)
    result = output_parser.invoke(cast(AIMessage, deserialized.message))
    assert result == parsed_response

def test_structured_outputs_parser_valid_falsy_response() -> None:
    class LunchBox(BaseModel):
        sandwiches: list[str]

        def __len__(self) -> int:
            return len(self.sandwiches)

    # prepare a valid *but falsy* response object, an empty LunchBox
    parsed_response = LunchBox(sandwiches=[])
    assert len(parsed_response) == 0
    llm_output = AIMessage(
        content='{"sandwiches": []}', additional_kwargs={"parsed": parsed_response}
    )
    output_parser = RunnableLambda(
        partial(_oai_structured_outputs_parser, schema=LunchBox)
    )
    result = output_parser.invoke(llm_output)
    assert result == parsed_response

def test__construct_lc_result_from_responses_api_basic_text_response() -> None:
    """Test a basic text response with no tools or special features."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputText(
                        type="output_text", text="Hello, world!", annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
            )
        ],
        usage=ResponseUsage(
            input_tokens=10,
            output_tokens=3,
            total_tokens=13,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        ),
    )

    # v0
    result = _construct_lc_result_from_responses_api(response, output_version="v0")

    assert isinstance(result, ChatResult)
    assert len(result.generations) == 1
    assert isinstance(result.generations[0], ChatGeneration)
    assert isinstance(result.generations[0].message, AIMessage)
    assert result.generations[0].message.content == [
        {"type": "text", "text": "Hello, world!", "annotations": []}
    ]
    assert result.generations[0].message.id == "msg_123"
    assert result.generations[0].message.usage_metadata
    assert result.generations[0].message.usage_metadata["input_tokens"] == 10
    assert result.generations[0].message.usage_metadata["output_tokens"] == 3
    assert result.generations[0].message.usage_metadata["total_tokens"] == 13
    assert result.generations[0].message.response_metadata["id"] == "resp_123"
    assert result.generations[0].message.response_metadata["model_name"] == "gpt-4o"

    # responses/v1
    result = _construct_lc_result_from_responses_api(response)
    assert result.generations[0].message.content == [
        {"type": "text", "text": "Hello, world!", "annotations": [], "id": "msg_123"}
    ]
    assert result.generations[0].message.id == "resp_123"
    assert result.generations[0].message.response_metadata["id"] == "resp_123"

def test__construct_lc_result_from_responses_api_multiple_text_blocks() -> None:
    """Test a response with multiple text blocks."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputText(
                        type="output_text", text="First part", annotations=[]
                    ),
                    ResponseOutputText(
                        type="output_text", text="Second part", annotations=[]
                    ),
                ],
                role="assistant",
                status="completed",
            )
        ],
    )

    result = _construct_lc_result_from_responses_api(response, output_version="v0")

    assert len(result.generations[0].message.content) == 2
    assert result.generations[0].message.content == [
        {"type": "text", "text": "First part", "annotations": []},
        {"type": "text", "text": "Second part", "annotations": []},
    ]

def test__construct_lc_result_from_responses_api_multiple_messages() -> None:
    """Test a response with multiple text blocks."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputText(type="output_text", text="foo", annotations=[])
                ],
                role="assistant",
                status="completed",
            ),
            ResponseReasoningItem(
                type="reasoning",
                id="rs_123",
                summary=[Summary(type="summary_text", text="reasoning foo")],
            ),
            ResponseOutputMessage(
                type="message",
                id="msg_234",
                content=[
                    ResponseOutputText(type="output_text", text="bar", annotations=[])
                ],
                role="assistant",
                status="completed",
            ),
        ],
    )

    # v0
    result = _construct_lc_result_from_responses_api(response, output_version="v0")

    assert result.generations[0].message.content == [
        {"type": "text", "text": "foo", "annotations": []},
        {"type": "text", "text": "bar", "annotations": []},
    ]
    assert result.generations[0].message.additional_kwargs == {
        "reasoning": {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "reasoning foo"}],
            "id": "rs_123",
        }
    }
    assert result.generations[0].message.id == "msg_234"

    # responses/v1
    result = _construct_lc_result_from_responses_api(response)

    assert result.generations[0].message.content == [
        {"type": "text", "text": "foo", "annotations": [], "id": "msg_123"},
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "reasoning foo"}],
            "id": "rs_123",
        },
        {"type": "text", "text": "bar", "annotations": [], "id": "msg_234"},
    ]
    assert result.generations[0].message.id == "resp_123"

def test__construct_lc_result_from_responses_api_refusal_response() -> None:
    """Test a response with a refusal."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputRefusal(
                        type="refusal", refusal="I cannot assist with that request."
                    )
                ],
                role="assistant",
                status="completed",
            )
        ],
    )

    # v0
    result = _construct_lc_result_from_responses_api(response, output_version="v0")

    assert result.generations[0].message.additional_kwargs["refusal"] == (
        "I cannot assist with that request."
    )

    # responses/v1
    result = _construct_lc_result_from_responses_api(response)
    assert result.generations[0].message.content == [
        {
            "type": "refusal",
            "refusal": "I cannot assist with that request.",
            "id": "msg_123",
        }
    ]

def test__construct_lc_result_from_responses_api_complex_response() -> None:
    """Test a complex response with multiple output types."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text="Here's the information you requested:",
                        annotations=[],
                    )
                ],
                role="assistant",
                status="completed",
            ),
            ResponseFunctionToolCall(
                type="function_call",
                id="func_123",
                call_id="call_123",
                name="get_weather",
                arguments='{"location": "New York"}',
            ),
        ],
        metadata={"key1": "value1", "key2": "value2"},
        incomplete_details=IncompleteDetails(reason="max_output_tokens"),
        status="completed",
        user="user_123",
    )

    # v0
    result = _construct_lc_result_from_responses_api(response, output_version="v0")

    # Check message content
    assert result.generations[0].message.content == [
        {
            "type": "text",
            "text": "Here's the information you requested:",
            "annotations": [],
        }
    ]

    # Check tool calls
    msg: AIMessage = cast(AIMessage, result.generations[0].message)
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["name"] == "get_weather"

    # Check metadata
    assert result.generations[0].message.response_metadata["id"] == "resp_123"
    assert result.generations[0].message.response_metadata["metadata"] == {
        "key1": "value1",
        "key2": "value2",
    }
    assert result.generations[0].message.response_metadata["incomplete_details"] == {
        "reason": "max_output_tokens"
    }
    assert result.generations[0].message.response_metadata["status"] == "completed"
    assert result.generations[0].message.response_metadata["user"] == "user_123"

    # responses/v1
    result = _construct_lc_result_from_responses_api(response)
    msg = cast(AIMessage, result.generations[0].message)
    assert msg.response_metadata["metadata"] == {"key1": "value1", "key2": "value2"}
    assert msg.content == [
        {
            "type": "text",
            "text": "Here's the information you requested:",
            "annotations": [],
            "id": "msg_123",
        },
        {
            "type": "function_call",
            "id": "func_123",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": '{"location": "New York"}',
        },
    ]

def test__construct_lc_result_from_responses_api_no_usage_metadata() -> None:
    """Test a response without usage metadata."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputText(
                        type="output_text", text="Hello, world!", annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
            )
        ],
        # No usage field
    )

    result = _construct_lc_result_from_responses_api(response)

    assert cast(AIMessage, result.generations[0].message).usage_metadata is None

def test__construct_lc_result_from_responses_api_web_search_response() -> None:
    """Test a response with web search output."""
    from openai.types.responses.response_function_web_search import (
        ResponseFunctionWebSearch,
    )

    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseFunctionWebSearch(
                id="websearch_123",
                type="web_search_call",
                status="completed",
                action=ActionSearch(type="search", query="search query"),
            )
        ],
    )

    # v0
    result = _construct_lc_result_from_responses_api(response, output_version="v0")

    assert "tool_outputs" in result.generations[0].message.additional_kwargs
    assert len(result.generations[0].message.additional_kwargs["tool_outputs"]) == 1
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["type"]
        == "web_search_call"
    )
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["id"]
        == "websearch_123"
    )
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["status"]
        == "completed"
    )

    # responses/v1
    result = _construct_lc_result_from_responses_api(response)
    assert result.generations[0].message.content == [
        {
            "type": "web_search_call",
            "id": "websearch_123",
            "status": "completed",
            "action": {"query": "search query", "type": "search"},
        }
    ]

def test__construct_lc_result_from_responses_api_file_search_response() -> None:
    """Test a response with file search output."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseFileSearchToolCall(
                id="filesearch_123",
                type="file_search_call",
                status="completed",
                queries=["python code", "langchain"],
                results=[
                    Result(
                        file_id="file_123",
                        filename="example.py",
                        score=0.95,
                        text="def hello_world() -> None:\n    print('Hello, world!')",
                        attributes={"language": "python", "size": 42},
                    )
                ],
            )
        ],
    )

    # v0
    result = _construct_lc_result_from_responses_api(response, output_version="v0")

    assert "tool_outputs" in result.generations[0].message.additional_kwargs
    assert len(result.generations[0].message.additional_kwargs["tool_outputs"]) == 1
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["type"]
        == "file_search_call"
    )
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["id"]
        == "filesearch_123"
    )
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["status"]
        == "completed"
    )
    assert result.generations[0].message.additional_kwargs["tool_outputs"][0][
        "queries"
    ] == ["python code", "langchain"]
    assert (
        len(
            result.generations[0].message.additional_kwargs["tool_outputs"][0][
                "results"
            ]
        )
        == 1
    )
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["results"][
            0
        ]["file_id"]
        == "file_123"
    )
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["results"][
            0
        ]["score"]
        == 0.95
    )

    # responses/v1
    result = _construct_lc_result_from_responses_api(response)
    assert result.generations[0].message.content == [
        {
            "type": "file_search_call",
            "id": "filesearch_123",
            "status": "completed",
            "queries": ["python code", "langchain"],
            "results": [
                {
                    "file_id": "file_123",
                    "filename": "example.py",
                    "score": 0.95,
                    "text": "def hello_world() -> None:\n    print('Hello, world!')",
                    "attributes": {"language": "python", "size": 42},
                }
            ],
        }
    ]

def test__construct_lc_result_from_responses_api_mixed_search_responses() -> None:
    """Test a response with both web search and file search outputs."""

    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputText(
                        type="output_text", text="Here's what I found:", annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
            ),
            ResponseFunctionWebSearch(
                id="websearch_123",
                type="web_search_call",
                status="completed",
                action=ActionSearch(type="search", query="search query"),
            ),
            ResponseFileSearchToolCall(
                id="filesearch_123",
                type="file_search_call",
                status="completed",
                queries=["python code"],
                results=[
                    Result(
                        file_id="file_123",
                        filename="example.py",
                        score=0.95,
                        text="def hello_world() -> None:\n    print('Hello, world!')",
                    )
                ],
            ),
        ],
    )

    # v0
    result = _construct_lc_result_from_responses_api(response, output_version="v0")

    # Check message content
    assert result.generations[0].message.content == [
        {"type": "text", "text": "Here's what I found:", "annotations": []}
    ]

    # Check tool outputs
    assert "tool_outputs" in result.generations[0].message.additional_kwargs
    assert len(result.generations[0].message.additional_kwargs["tool_outputs"]) == 2

    # Check web search output
    web_search = next(
        output
        for output in result.generations[0].message.additional_kwargs["tool_outputs"]
        if output["type"] == "web_search_call"
    )
    assert web_search["id"] == "websearch_123"
    assert web_search["status"] == "completed"

    # Check file search output
    file_search = next(
        output
        for output in result.generations[0].message.additional_kwargs["tool_outputs"]
        if output["type"] == "file_search_call"
    )
    assert file_search["id"] == "filesearch_123"
    assert file_search["queries"] == ["python code"]
    assert file_search["results"][0]["filename"] == "example.py"

    # responses/v1
    result = _construct_lc_result_from_responses_api(response)
    assert result.generations[0].message.content == [
        {
            "type": "text",
            "text": "Here's what I found:",
            "annotations": [],
            "id": "msg_123",
        },
        {
            "type": "web_search_call",
            "id": "websearch_123",
            "status": "completed",
            "action": {"type": "search", "query": "search query"},
        },
        {
            "type": "file_search_call",
            "id": "filesearch_123",
            "queries": ["python code"],
            "results": [
                {
                    "file_id": "file_123",
                    "filename": "example.py",
                    "score": 0.95,
                    "text": "def hello_world() -> None:\n    print('Hello, world!')",
                }
            ],
            "status": "completed",
        },
    ]

def test_compat_responses_v03() -> None:
    # Check compatibility with v0.3 message format
    message_v03 = AIMessage(
        content=[
            {"type": "text", "text": "Hello, world!", "annotations": [{"type": "foo"}]}
        ],
        additional_kwargs={
            "reasoning": {
                "type": "reasoning",
                "id": "rs_123",
                "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
            },
            "tool_outputs": [
                {
                    "type": "web_search_call",
                    "id": "websearch_123",
                    "status": "completed",
                }
            ],
            "refusal": "I cannot assist with that.",
        },
        response_metadata={"id": "resp_123"},
        id="msg_123",
    )

    message = _convert_from_v03_ai_message(message_v03)
    expected = AIMessage(
        content=[
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
                "id": "rs_123",
            },
            {
                "type": "text",
                "text": "Hello, world!",
                "annotations": [{"type": "foo"}],
                "id": "msg_123",
            },
            {"type": "refusal", "refusal": "I cannot assist with that."},
            {"type": "web_search_call", "id": "websearch_123", "status": "completed"},
        ],
        response_metadata={"id": "resp_123"},
        id="resp_123",
    )
    assert message == expected

    ## Check no mutation
    assert message != message_v03
    assert len(message_v03.content) == 1
    assert all(
        item in message_v03.additional_kwargs
        for item in ["reasoning", "tool_outputs", "refusal"]
    )

    # Convert back
    message_v03_output = _convert_to_v03_ai_message(message)
    assert message_v03_output == message_v03
    assert message_v03_output is not message_v03

def test_convert_from_v1_to_chat_completions(
    message_v1: AIMessage, expected: AIMessage
) -> None:
    result = _convert_from_v1_to_chat_completions(message_v1)
    assert result == expected
    assert result.tool_calls == message_v1.tool_calls  # tool calls remain cached

    # Check no mutation
    assert message_v1 != result

def test_convert_from_v1_to_responses(
    message_v1: AIMessage, expected: list[dict[str, Any]]
) -> None:
    tcs: list[types.ToolCall] = [
        {
            "type": "tool_call",
            "name": tool_call["name"],
            "args": tool_call["args"],
            "id": tool_call.get("id"),
        }
        for tool_call in message_v1.tool_calls
    ]
    result = _convert_from_v1_to_responses(message_v1.content_blocks, tcs)
    assert result == expected

    # Check no mutation
    assert message_v1 != result

def test_structured_output_verbosity(
    verbosity_format: str, streaming: bool, schema_format: str
) -> None:
    class MySchema(BaseModel):
        foo: str

    if verbosity_format == "model_kwargs":
        init_params: dict[str, Any] = {"model_kwargs": {"text": {"verbosity": "high"}}}
    else:
        init_params = {"verbosity": "high"}

    if streaming:
        init_params["streaming"] = True

    llm = ChatOpenAI(model="gpt-5", use_responses_api=True, **init_params)

    if schema_format == "pydantic":
        schema: Any = MySchema
    else:
        schema = MySchema.model_json_schema()

    structured_llm = llm.with_structured_output(schema)
    sequence = cast(RunnableSequence, structured_llm)
    binding = cast(RunnableBinding, sequence.first)
    bound_llm = cast(ChatOpenAI, binding.bound)
    bound_kwargs = binding.kwargs

    messages = [HumanMessage(content="Hello")]
    payload = bound_llm._get_request_payload(messages, **bound_kwargs)

    # Verify that verbosity is present in `text` param
    assert "text" in payload
    assert "verbosity" in payload["text"]
    assert payload["text"]["verbosity"] == "high"

    # Verify that schema is passed correctly
    if schema_format == "pydantic" and not streaming:
        assert payload["text_format"] == schema
    else:
        assert "format" in payload["text"]
        assert payload["text"]["format"]["type"] == "json_schema"


# --- libs/partners/openai/tests/unit_tests/chat_models/test_prompt_cache_key.py ---

def test_prompt_cache_key_parameter_inclusion() -> None:
    """Test that prompt_cache_key parameter is properly included in request payload."""
    chat = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=10)
    messages = [HumanMessage("Hello")]

    payload = chat._get_request_payload(messages, prompt_cache_key="test-cache-key")
    assert "prompt_cache_key" in payload
    assert payload["prompt_cache_key"] == "test-cache-key"

def test_prompt_cache_key_parameter_exclusion() -> None:
    """Test that prompt_cache_key parameter behavior matches OpenAI API."""
    chat = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=10)
    messages = [HumanMessage("Hello")]

    # Test with explicit None (OpenAI should accept None values (marked Optional))
    payload = chat._get_request_payload(messages, prompt_cache_key=None)
    assert "prompt_cache_key" in payload
    assert payload["prompt_cache_key"] is None

def test_prompt_cache_key_per_call() -> None:
    """Test that prompt_cache_key can be passed per-call with different values."""
    chat = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=10)
    messages = [HumanMessage("Hello")]

    # Test different cache keys per call
    payload1 = chat._get_request_payload(messages, prompt_cache_key="cache-v1")
    payload2 = chat._get_request_payload(messages, prompt_cache_key="cache-v2")

    assert payload1["prompt_cache_key"] == "cache-v1"
    assert payload2["prompt_cache_key"] == "cache-v2"

    # Test dynamic cache key assignment
    cache_keys = ["customer-v1", "support-v1", "feedback-v1"]

    for cache_key in cache_keys:
        payload = chat._get_request_payload(messages, prompt_cache_key=cache_key)
        assert "prompt_cache_key" in payload
        assert payload["prompt_cache_key"] == cache_key

def test_prompt_cache_key_model_kwargs() -> None:
    """Test prompt_cache_key via model_kwargs and method precedence."""
    messages = [HumanMessage("Hello world")]

    # Test model-level via model_kwargs
    chat = ChatOpenAI(
        model="gpt-4o-mini",
        max_completion_tokens=10,
        model_kwargs={"prompt_cache_key": "model-level-cache"},
    )
    payload = chat._get_request_payload(messages)
    assert "prompt_cache_key" in payload
    assert payload["prompt_cache_key"] == "model-level-cache"

    # Test that per-call cache key overrides model-level
    payload_override = chat._get_request_payload(
        messages, prompt_cache_key="per-call-cache"
    )
    assert payload_override["prompt_cache_key"] == "per-call-cache"

def test_prompt_cache_key_responses_api() -> None:
    """Test that prompt_cache_key works with Responses API."""
    chat = ChatOpenAI(
        model="gpt-4o-mini",
        use_responses_api=True,
        output_version="responses/v1",
        max_completion_tokens=10,
    )

    messages = [HumanMessage("Hello")]
    payload = chat._get_request_payload(
        messages, prompt_cache_key="responses-api-cache-v1"
    )

    # prompt_cache_key should be present regardless of API type
    assert "prompt_cache_key" in payload
    assert payload["prompt_cache_key"] == "responses-api-cache-v1"


# --- libs/partners/openai/tests/unit_tests/llms/test_base.py ---

def test_stream_response_to_generation_chunk() -> None:
    completion = {
        "id": "cmpl-abc123",
        "choices": [
            {"finish_reason": None, "index": 0, "logprobs": None, "text": "foo"}
        ],
        "created": 1749214401,
        "model": "my-model",
        "object": "text_completion",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = _stream_response_to_generation_chunk(completion)
    assert chunk == GenerationChunk(
        text="foo", generation_info={"finish_reason": None, "logprobs": None}
    )

    # Pathological completion with None text (e.g., from other providers)
    completion = {
        "id": "cmpl-abc123",
        "choices": [
            {"finish_reason": None, "index": 0, "logprobs": None, "text": None}
        ],
        "created": 1749214401,
        "model": "my-model",
        "object": "text_completion",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = _stream_response_to_generation_chunk(completion)
    assert chunk == GenerationChunk(
        text="", generation_info={"finish_reason": None, "logprobs": None}
    )

def test_generate_streaming_multiple_prompts_error() -> None:
    """Ensures ValueError when streaming=True and multiple prompts."""
    llm = OpenAI(streaming=True)

    with pytest.raises(
        ValueError, match="Cannot stream results with multiple prompts\\."
    ):
        llm._generate(["foo", "bar"])


# --- libs/partners/openai/tests/unit_tests/test_load.py ---

def test_loads_runnable_sequence_prompt_model() -> None:
    """Test serialization/deserialization of a chain:

    `prompt | model (RunnableSequence)`
    """
    prompt = ChatPromptTemplate.from_messages([("user", "Hello, {name}!")])
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, openai_api_key="hello")  # type: ignore[call-arg]
    chain = prompt | model

    # Verify the chain is a RunnableSequence
    assert isinstance(chain, RunnableSequence)

    # Serialize
    chain_string = dumps(chain)

    # Deserialize
    # (ChatPromptTemplate contains HumanMessagePromptTemplate and PromptTemplate)
    chain2 = loads(
        chain_string,
        secrets_map={"OPENAI_API_KEY": "hello"},
        allowed_objects=[
            RunnableSequence,
            ChatPromptTemplate,
            HumanMessagePromptTemplate,
            PromptTemplate,
            ChatOpenAI,
        ],
    )

    # Verify structure
    assert isinstance(chain2, RunnableSequence)
    assert isinstance(chain2.first, ChatPromptTemplate)
    assert isinstance(chain2.last, ChatOpenAI)

    # Verify round-trip serialization
    assert dumps(chain2) == chain_string

def test_load_runnable_sequence_prompt_model() -> None:
    """Test load() with a chain:

    `prompt | model (RunnableSequence)`.
    """
    prompt = ChatPromptTemplate.from_messages([("user", "Tell me about {topic}")])
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key="hello")  # type: ignore[call-arg]
    chain = prompt | model

    # Serialize
    chain_obj = dumpd(chain)

    # Deserialize
    # (ChatPromptTemplate contains HumanMessagePromptTemplate and PromptTemplate)
    chain2 = load(
        chain_obj,
        secrets_map={"OPENAI_API_KEY": "hello"},
        allowed_objects=[
            RunnableSequence,
            ChatPromptTemplate,
            HumanMessagePromptTemplate,
            PromptTemplate,
            ChatOpenAI,
        ],
    )

    # Verify structure
    assert isinstance(chain2, RunnableSequence)
    assert isinstance(chain2.first, ChatPromptTemplate)
    assert isinstance(chain2.last, ChatOpenAI)

    # Verify round-trip serialization
    assert dumpd(chain2) == chain_obj


# --- libs/partners/openai/tests/unit_tests/test_tools.py ---

def test_custom_tool() -> None:
    @custom_tool
    def my_tool(x: str) -> str:
        """Do thing."""
        return "a" + x

    # Test decorator
    assert isinstance(my_tool, Tool)
    assert my_tool.metadata == {"type": "custom_tool"}
    assert my_tool.description == "Do thing."

    result = my_tool.invoke(
        {
            "type": "tool_call",
            "name": "my_tool",
            "args": {"whatever": "b"},
            "id": "abc",
            "extras": {"type": "custom_tool_call"},
        }
    )
    assert result == ToolMessage(
        [{"type": "custom_tool_call_output", "output": "ab"}],
        name="my_tool",
        tool_call_id="abc",
    )

    # Test tool schema
    ## Test with format
    @custom_tool(format={"type": "grammar", "syntax": "lark", "definition": "..."})
    def another_tool(x: str) -> None:
        """Do thing."""

    llm = ChatOpenAI(
        model="gpt-4.1", use_responses_api=True, output_version="responses/v1"
    ).bind_tools([another_tool])
    assert llm.kwargs == {  # type: ignore[attr-defined]
        "tools": [
            {
                "type": "custom",
                "name": "another_tool",
                "description": "Do thing.",
                "format": {"type": "grammar", "syntax": "lark", "definition": "..."},
            }
        ]
    }

    llm = ChatOpenAI(
        model="gpt-4.1", use_responses_api=True, output_version="responses/v1"
    ).bind_tools([my_tool])
    assert llm.kwargs == {  # type: ignore[attr-defined]
        "tools": [{"type": "custom", "name": "my_tool", "description": "Do thing."}]
    }

    # Test passing messages back
    message_history = [
        HumanMessage("Use the tool"),
        AIMessage(
            [
                {
                    "type": "custom_tool_call",
                    "id": "ctc_abc123",
                    "call_id": "abc",
                    "name": "my_tool",
                    "input": "a",
                }
            ],
            tool_calls=[
                {
                    "type": "tool_call",
                    "name": "my_tool",
                    "args": {"__arg1": "a"},
                    "id": "abc",
                }
            ],
        ),
        result,
    ]
    payload = llm._get_request_payload(message_history)  # type: ignore[attr-defined]
    expected_input = [
        {"content": "Use the tool", "role": "user", "type": "message"},
        {
            "type": "custom_tool_call",
            "id": "ctc_abc123",
            "call_id": "abc",
            "name": "my_tool",
            "input": "a",
        },
        {"type": "custom_tool_call_output", "call_id": "abc", "output": "ab"},
    ]
    assert payload["input"] == expected_input

async def test_async_custom_tool() -> None:
    @custom_tool
    async def my_async_tool(x: str) -> str:
        """Do async thing."""
        return "a" + x

    # Test decorator
    assert isinstance(my_async_tool, Tool)
    assert my_async_tool.metadata == {"type": "custom_tool"}
    assert my_async_tool.description == "Do async thing."

    result = await my_async_tool.ainvoke(
        {
            "type": "tool_call",
            "name": "my_async_tool",
            "args": {"whatever": "b"},
            "id": "abc",
            "extras": {"type": "custom_tool_call"},
        }
    )
    assert result == ToolMessage(
        [{"type": "custom_tool_call_output", "output": "ab"}],
        name="my_async_tool",
        tool_call_id="abc",
    )


# --- libs/partners/xai/tests/unit_tests/test_chat_models.py ---

def test_chat_xai_invalid_streaming_params() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with pytest.raises(ValueError):
        ChatXAI(
            model=MODEL_NAME,
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )

def test_stream_usage_metadata() -> None:
    model = ChatXAI(model=MODEL_NAME)
    assert model.stream_usage is True

    model = ChatXAI(model=MODEL_NAME, stream_usage=False)
    assert model.stream_usage is False

