# googleapis/mcp-toolbox-sdk-python
# 1 LLM-backed test functions across 31 test files
# Source: https://github.com/googleapis/mcp-toolbox-sdk-python

# --- packages/toolbox-llamaindex/tests/test_async_tools.py ---

    async def test_toolbox_tool_call_requires_auth_strict(self, auth_toolbox_tool):
        with pytest.raises(
            PermissionError,
            match="One or more of the following authn services are required to invoke this tool: test-auth-source",
        ):
            await auth_toolbox_tool.acall(param2=123)

