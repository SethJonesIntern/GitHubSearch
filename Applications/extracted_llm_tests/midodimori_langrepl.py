# midodimori/langrepl
# 7 LLM-backed test functions across 77 test files
# Source: https://github.com/midodimori/langrepl

# --- tests/sandboxes/test_serialization.py ---

    def test_serialize_excludes_messages(self, agent_context):
        """Messages should be excluded from serialized state."""
        runtime = _create_runtime(agent_context)

        result = serialize_runtime(runtime)

        assert "messages" not in result["state"]

    def test_serialize_run_id_to_string(self, agent_context):
        """UUID run_id should be converted to string."""
        run_id = uuid.uuid4()
        runtime = _create_runtime(agent_context, run_id=run_id)

        result = serialize_runtime(runtime)

        assert result["config"]["run_id"] == str(run_id)

    def test_serialize_preserves_tool_call_id(self, agent_context):
        """Tool call ID should be preserved."""
        runtime = _create_runtime(agent_context)

        result = serialize_runtime(runtime)

        assert result["tool_call_id"] == "test-call-id"

    def test_serialize_context_as_json(self, agent_context):
        """Context should be serialized as JSON dict."""
        runtime = _create_runtime(agent_context)

        result = serialize_runtime(runtime)

        assert isinstance(result["context"], dict)
        assert result["context"]["approval_mode"] == "aggressive"

    def test_deserialize_reconstructs_config(self, agent_context):
        """Config fields should be restored correctly."""
        runtime = _create_runtime(agent_context, run_id=uuid.uuid4())
        serialized = serialize_runtime(runtime)

        result = deserialize_runtime(serialized)

        assert result.config.get("tags") == ["test-tag"]
        assert result.config.get("metadata") == {"key": "value"}
        assert result.config.get("recursion_limit") == 25
        assert result.config.get("configurable") == {"thread_id": "test-thread"}
        assert isinstance(result.config.get("run_id"), uuid.UUID)

    def test_deserialize_restores_context(self, agent_context):
        """AgentContext should be reconstructed."""
        runtime = _create_runtime(agent_context)
        serialized = serialize_runtime(runtime)

        result = deserialize_runtime(serialized)

        assert result.context is not None
        assert result.context.approval_mode == ApprovalMode.AGGRESSIVE

    def test_deserialize_adds_empty_messages(self, agent_context):
        """Deserialized state should have empty messages list."""
        runtime = _create_runtime(agent_context)
        serialized = serialize_runtime(runtime)

        result = deserialize_runtime(serialized)

        assert result.state["messages"] == []

