# Oaklight/ToolRegistry
# 8 LLM-backed test functions across 47 test files
# Source: https://github.com/Oaklight/ToolRegistry

# --- tests/test_tool_registry.py ---

    def test_execute_tool_calls_with_openai_format(self, populated_registry):
        """Test executing tool calls with OpenAI format."""
        tool_calls = [
            ChatCompletionMessageFunctionToolCall(
                id="call_1",
                function=Function(name="add_numbers", arguments='{"a": 5, "b": 3}'),
            )
        ]

        results = populated_registry.execute_tool_calls(tool_calls)

        assert isinstance(results, dict)
        assert "call_1" in results
        assert int(results["call_1"]) == 8

    def test_execute_tool_calls_with_execution_mode_override(self, populated_registry):
        """Test executing tool calls with execution mode override."""
        tool_calls = [
            ChatCompletionMessageFunctionToolCall(
                id="call_3",
                function=Function(name="add_numbers", arguments='{"a": 10, "b": 20}'),
            )
        ]

        results = populated_registry.execute_tool_calls(
            tool_calls, execution_mode="thread"
        )

        assert isinstance(results, dict)
        assert "call_3" in results
        assert int(results["call_3"]) == 30

    def test_build_tool_call_messages(self, populated_registry):
        """Test recovering assistant message from tool calls."""
        tool_calls = [
            ChatCompletionMessageFunctionToolCall(
                id="call_1",
                function=Function(name="add_numbers", arguments='{"a": 5, "b": 3}'),
            )
        ]

        tool_responses = {"call_1": "8"}

        messages = populated_registry.build_tool_call_messages(
            tool_calls, tool_responses
        )

        assert isinstance(messages, list)
        assert len(messages) == 2  # Assistant message + tool response

        # First message should be assistant with tool calls
        assert messages[0]["role"] == "assistant"
        assert "tool_calls" in messages[0]

        # Second message should be tool response
        assert messages[1]["role"] == "tool"
        assert messages[1]["content"] == "8"

    def test_execute_tool_calls_truncates_large_result(self):
        """Test that large results are truncated when max_result_size is set."""
        registry = ToolRegistry(name="trunc_test")

        def big_output(n: int) -> str:
            """Return a large string."""
            return "x" * n

        registry.register(
            Tool.from_function(big_output, metadata=ToolMetadata(max_result_size=50))
        )

        tool_calls = [
            ChatCompletionMessageFunctionToolCall(
                id="call_big",
                function=Function(name="big_output", arguments='{"n": 500}'),
            )
        ]
        results = registry.execute_tool_calls(tool_calls)

        assert "call_big" in results
        assert "Truncated" in results["call_big"]
        assert "500 chars" in results["call_big"]
        # The truncated content should be much shorter than 500
        assert len(results["call_big"]) < 500

    def test_execute_tool_calls_no_truncation_when_under_limit(self):
        """Test that small results are not truncated."""
        registry = ToolRegistry(name="no_trunc_test")

        def small_output() -> str:
            """Return a small string."""
            return "hello"

        registry.register(
            Tool.from_function(
                small_output, metadata=ToolMetadata(max_result_size=1000)
            )
        )

        tool_calls = [
            ChatCompletionMessageFunctionToolCall(
                id="call_small",
                function=Function(name="small_output", arguments="{}"),
            )
        ]
        results = registry.execute_tool_calls(tool_calls)

        assert results["call_small"] == "hello"

    def test_execute_tool_calls_default_max_result_size(self):
        """Test that registry-level default_max_result_size is applied."""
        registry = ToolRegistry(name="default_trunc", default_max_result_size=30)

        def verbose_output() -> str:
            """Return a verbose string."""
            return "a" * 200

        registry.register(verbose_output)

        tool_calls = [
            ChatCompletionMessageFunctionToolCall(
                id="call_verbose",
                function=Function(name="verbose_output", arguments="{}"),
            )
        ]
        results = registry.execute_tool_calls(tool_calls)

        assert "Truncated" in results["call_verbose"]

    def test_execute_tool_calls_tool_level_overrides_default(self):
        """Test that tool-level max_result_size takes precedence over default."""
        registry = ToolRegistry(name="override_test", default_max_result_size=10)

        def output() -> str:
            """Return a medium string."""
            return "b" * 50

        registry.register(
            Tool.from_function(output, metadata=ToolMetadata(max_result_size=1000))
        )

        tool_calls = [
            ChatCompletionMessageFunctionToolCall(
                id="call_override",
                function=Function(name="output", arguments="{}"),
            )
        ]
        results = registry.execute_tool_calls(tool_calls)

        # 50 chars < 1000 limit, so no truncation
        assert "Truncated" not in results["call_override"]

    def test_execute_tool_calls_strips_thought(self):
        """Test that thought is stripped in execute_tool_calls."""
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall as ChatCompletionMessageFunctionToolCall,
            Function,
        )

        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        registry = ToolRegistry()
        registry.register(greet)

        tool_calls = [
            ChatCompletionMessageFunctionToolCall(
                id="call_think",
                type="function",
                function=Function(
                    name="greet",
                    arguments='{"name": "World", "thought": "I should greet the world"}',
                ),
            )
        ]
        results = registry.execute_tool_calls(tool_calls)
        assert results["call_think"] == "Hello, World!"

