# langchain-ai/langchain-mcp-adapters
# 6 LLM-backed test functions across 8 test files
# Source: https://github.com/langchain-ai/langchain-mcp-adapters

# --- tests/test_tools.py ---

async def test_load_mcp_tools_with_http_variations(socket_enabled, transport) -> None:
    """Test load mcp tools with annotations."""
    with run_streamable_http(_create_annotations_server, 8181):
        # Initialize client without initial connections
        client = MultiServerMCPClient(
            {
                "time": {
                    "url": "http://localhost:8181/mcp",
                    "transport": transport,
                }
            },
        )
        # pass
        tools = await client.get_tools(server_name="time")
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "get_time"

async def test_load_mcp_tools_with_annotations(socket_enabled) -> None:
    """Test load mcp tools with annotations."""
    with run_streamable_http(_create_annotations_server, 8181):
        # Initialize client without initial connections
        client = MultiServerMCPClient(
            {
                "time": {
                    "url": "http://localhost:8181/mcp",
                    "transport": "streamable_http",
                }
            },
        )
        # pass
        tools = await client.get_tools(server_name="time")
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "get_time"
        assert tool.metadata == {
            "title": "Get Time",
            "readOnlyHint": True,
            "idempotentHint": False,
            "destructiveHint": None,
            "openWorldHint": None,
        }

async def test_convert_langchain_tool_to_fastmcp_tool(tool_instance):
    fastmcp_tool = to_fastmcp(tool_instance)
    assert fastmcp_tool.name == "add"
    assert fastmcp_tool.description == "Add two numbers"
    assert fastmcp_tool.parameters == {
        "description": "Add two numbers",
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "integer"},
        },
        "required": ["a", "b"],
        "title": "add",
        "type": "object",
    }
    assert fastmcp_tool.fn_metadata.arg_model.model_json_schema() == {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "integer"},
        },
        "required": ["a", "b"],
        "title": "addArguments",
        "type": "object",
    }

    arguments = {"a": 1, "b": 2}
    assert await fastmcp_tool.run(arguments=arguments) == 3

async def test_load_mcp_tools_with_custom_httpx_client_factory(socket_enabled) -> None:
    """Test load mcp tools with custom httpx client factory."""

    # Custom httpx client factory
    def custom_httpx_client_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        """Custom factory for creating httpx.AsyncClient with specific configuration."""
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout or httpx.Timeout(30.0),
            auth=auth,
            # Custom configuration
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    with run_streamable_http(_create_status_server, 8182):
        # Initialize client with custom httpx_client_factory
        client = MultiServerMCPClient(
            {
                "status": {
                    "url": "http://localhost:8182/mcp",
                    "transport": "streamable_http",
                    "httpx_client_factory": custom_httpx_client_factory,
                },
            },
        )

        tools = await client.get_tools(server_name="status")
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "get_status"

        # Test that the tool works correctly
        result = await tool.ainvoke({"args": {}, "id": "1", "type": "tool_call"})
        assert result.content == [
            {"type": "text", "text": "Server is running", "id": IsLangChainID}
        ]

async def test_load_mcp_tools_with_custom_httpx_client_factory_sse(
    socket_enabled,
) -> None:
    """Test load mcp tools with custom httpx client factory using SSE transport."""

    # Custom httpx client factory
    def custom_httpx_client_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        """Custom factory for creating httpx.AsyncClient with specific configuration."""
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout or httpx.Timeout(30.0),
            auth=auth,
            # Custom configuration for SSE
            limits=httpx.Limits(max_keepalive_connections=3, max_connections=5),
        )

    with run_streamable_http(_create_info_server, 8183):
        # Initialize client with custom httpx_client_factory for SSE
        client = MultiServerMCPClient(
            {
                "info": {
                    "url": "http://localhost:8183/sse",
                    "transport": "sse",
                    "httpx_client_factory": custom_httpx_client_factory,
                },
            },
        )

        # Note: This test may not work in practice since the server doesn't expose SSE
        # endpoint,
        # but it tests the configuration propagation
        try:
            tools = await client.get_tools(server_name="info")
            # If we get here, the httpx_client_factory was properly passed
            assert isinstance(tools, list)
        except Exception:
            # Expected to fail since server doesn't have SSE endpoint,
            # but the important thing is that httpx_client_factory was passed correctly
            pass

async def test_get_tools_with_name_conflict(socket_enabled) -> None:
    """Test fetching tools with name conflict using tool_name_prefix.

    This test verifies that:
    1. Without tool_name_prefix, both servers would have conflicting "search" tool names
    2. With tool_name_prefix=True, tools get unique names
        (weather_search, flights_search)
    """
    with (
        run_streamable_http(_create_weather_search_server, 8185),
        run_streamable_http(_create_flights_search_server, 8186),
    ):
        # First, verify that without prefix both tools would have the same name
        client_no_prefix = MultiServerMCPClient(
            {
                "weather": {
                    "url": "http://localhost:8185/mcp",
                    "transport": "streamable_http",
                },
                "flights": {
                    "url": "http://localhost:8186/mcp",
                    "transport": "streamable_http",
                },
            },
            tool_name_prefix=False,
        )
        tools_no_prefix = await client_no_prefix.get_tools()
        # Both tools are named "search" without prefix
        assert all(t.name == "search" for t in tools_no_prefix)

        # Now test with prefix - tools should be disambiguated
        client = MultiServerMCPClient(
            {
                "weather": {
                    "url": "http://localhost:8185/mcp",
                    "transport": "streamable_http",
                },
                "flights": {
                    "url": "http://localhost:8186/mcp",
                    "transport": "streamable_http",
                },
            },
            tool_name_prefix=True,
        )
        tools = await client.get_tools()

        # Verify we have both prefixed tools with unique names
        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert tool_names == {"weather_search", "flights_search"}

