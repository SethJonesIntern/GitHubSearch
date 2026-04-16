# UiPath/uipath-langchain-python
# 1 LLM-backed test functions across 80 test files
# Source: https://github.com/UiPath/uipath-langchain-python

# --- tests/agent/messages/test_message_utils.py ---

    def test_replace_tool_calls_updated_args_visible_via_add_messages(self):
        """Test that updated tool call args are visible after add_messages processes them.

        Reproduces the HITL bug: when a human reviews and updates activity input
        during an escalation, the activity must execute with the reviewed args.
        Without id preservation, add_messages appends a duplicate AIMessage
        instead of replacing the original, causing the tool to run with stale args.
        """
        original_tool_calls = [
            ToolCall(name="my_activity", args={"input": "original_value"}, id="call_1")
        ]
        original_ai_message = AIMessage(
            content_blocks=[
                create_text_block("I will invoke the activity"),
                create_tool_call(
                    name="my_activity", args={"input": "original_value"}, id="call_1"
                ),
            ],
            tool_calls=original_tool_calls,
            id="msg-from-llm",
        )

        messages: list[MessageItem] = [
            HumanMessage(content="do something", id="msg-human"),
            original_ai_message,
        ]

        # Simulate HITL review: human changes the input
        reviewed_tool_calls = [
            ToolCall(name="my_activity", args={"input": "reviewed_value"}, id="call_1")
        ]
        updated_ai_message = replace_tool_calls(
            original_ai_message, reviewed_tool_calls
        )

        # Simulate what Command(update={"messages": [updated_ai_message]}) does
        new_messages: list[MessageItem] = [updated_ai_message]
        result_messages = add_messages(messages, new_messages)

        # There must be exactly one AIMessage — not a duplicate
        ai_messages = [m for m in result_messages if isinstance(m, AIMessage)]
        assert len(ai_messages) == 1, (
            f"Expected 1 AIMessage but got {len(ai_messages)}; "
            "add_messages appended instead of replacing (id mismatch)"
        )

        # The surviving AIMessage must carry the reviewed args
        assert ai_messages[0].tool_calls[0]["args"] == {"input": "reviewed_value"}

