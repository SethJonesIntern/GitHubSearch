# oracle/langchain-oracle
# 36 LLM-backed test functions across 44 test files
# Source: https://github.com/oracle/langchain-oracle

# --- libs/oci/tests/integration_tests/chat_models/test_tool_calling.py ---

def test_tool_calling_no_infinite_loop(model_id: str, weather_tool: StructuredTool):
    """Test that tool calling works without infinite loops.

    This test verifies that after a tool is called and results are returned,
    the model generates a final response without making additional tool calls,
    preventing infinite loops.

    The fix sets tool_choice='none' when ToolMessages are present in the
    conversation history, which tells the model to stop calling tools.
    """
    agent = create_agent(model_id, weather_tool)

    # Invoke the agent
    system_msg = (
        "You are a helpful assistant. Use the available tools when "
        "needed to answer questions accurately."
    )
    input_messages: list[BaseMessage] = [
        SystemMessage(content=system_msg),
        HumanMessage(content="What's the weather in Chicago?"),
    ]
    result = agent.invoke({"messages": input_messages})

    messages = result["messages"]

    # Verify the conversation structure
    expected = "Should have at least: System, Human, AI (tool call), Tool, AI"
    assert len(messages) >= 4, expected

    # Find tool messages
    tool_messages = [msg for msg in messages if type(msg).__name__ == "ToolMessage"]
    assert len(tool_messages) >= 1, "Should have at least one tool result"

    # Find AI messages with tool calls
    ai_tool_calls = [
        msg
        for msg in messages
        if (
            type(msg).__name__ == "AIMessage"
            and hasattr(msg, "tool_calls")
            and msg.tool_calls
        )
    ]
    # The model should call the tool, but after receiving results,
    # should not call again. Allow flexibility - some models might make
    # 1 call, others might need 2, but should stop
    error_msg = (
        f"Model made too many tool calls ({len(ai_tool_calls)}), possible infinite loop"
    )
    assert len(ai_tool_calls) <= 2, error_msg

    # Verify final message is an AI response without tool calls
    final_message = messages[-1]
    assert type(final_message).__name__ == "AIMessage", (
        "Final message should be AIMessage"
    )
    assert final_message.content, "Final message should have content"
    assert not (hasattr(final_message, "tool_calls") and final_message.tool_calls), (
        "Final message should not have tool_calls (infinite loop prevention)"
    )

def test_meta_llama_tool_calling(weather_tool: StructuredTool):
    """Specific test for Meta Llama models to ensure fix works."""
    model_id = "meta.llama-4-scout-17b-16e-instruct"
    agent = create_agent(model_id, weather_tool)

    input_messages: list[BaseMessage] = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Check the weather in San Francisco."),
    ]
    result = agent.invoke({"messages": input_messages})

    messages = result["messages"]
    final_message = messages[-1]

    # Meta Llama was specifically affected by infinite loops
    # Verify it stops after receiving tool results (most important check!)
    assert type(final_message).__name__ == "AIMessage"
    assert not (hasattr(final_message, "tool_calls") and final_message.tool_calls)
    assert final_message.content, "Should have generated some response"

def test_cohere_tool_calling(weather_tool: StructuredTool):
    """Specific test for Cohere models to ensure they work correctly."""
    model_id = "cohere.command-a-03-2025"
    agent = create_agent(model_id, weather_tool)

    input_messages: list[BaseMessage] = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What's the weather like in New York?"),
    ]
    result = agent.invoke({"messages": input_messages})

    messages = result["messages"]
    final_message = messages[-1]

    # Cohere models should handle tool calling naturally
    assert type(final_message).__name__ == "AIMessage"
    assert not (hasattr(final_message, "tool_calls") and final_message.tool_calls)
    assert "60" in final_message.content or "cloudy" in final_message.content.lower()

def test_multi_step_tool_orchestration(model_id: str):
    """Test multi-step tool orchestration without infinite loops.

    This test simulates a realistic diagnostic workflow where an agent
    needs to call 4-6 tools sequentially (similar to SRE/monitoring
    scenarios). It verifies that:

    1. The agent can call multiple tools in sequence (multi-step)
    2. The agent eventually stops and provides a final answer
    3. No infinite loops occur (respects max_sequential_tool_calls limit)
    4. Tool call count stays within reasonable bounds (4-8 calls)

    This addresses the specific issue where agents need to perform
    multi-step investigations requiring several tool calls before
    providing a final analysis.
    """

    # Create diagnostic tools that simulate a monitoring workflow
    def check_status(resource: str) -> str:
        """Check the status of a resource."""
        status_data = {
            "payment-service": "Status: Running, Memory: 95%, Restarts: 12",
            "web-server": "Status: Running, Memory: 60%, Restarts: 0",
        }
        return status_data.get(resource, f"Resource {resource} status: Unknown")

    def get_events(resource: str) -> str:
        """Get recent events for a resource."""
        events_data = {
            "payment-service": (
                "Events: [OOMKilled at 14:23, BackOff at 14:30, Started at 14:32]"
            ),
            "web-server": "Events: [Started at 10:00, Healthy]",
        }
        return events_data.get(resource, f"No events for {resource}")

    def get_metrics(resource: str) -> str:
        """Get historical metrics for a resource."""
        metrics_data = {
            "payment-service": (
                "Memory trend: 70%→80%→90%→95% (gradual increase over 2h)"
            ),
            "web-server": "Memory trend: 55%→58%→60% (stable)",
        }
        return metrics_data.get(resource, f"No metrics for {resource}")

    def check_changes(resource: str) -> str:
        """Check recent changes to a resource."""
        changes_data = {
            "payment-service": "Recent deployment: v1.2.3 deployed 2h ago",
            "web-server": "No recent changes (last deployment 3 days ago)",
        }
        return changes_data.get(resource, f"No changes for {resource}")

    def create_alert(severity: str, message: str) -> str:
        """Create an alert/incident."""
        return f"Alert created: [{severity.upper()}] {message}"

    def take_action(resource: str, action: str) -> str:
        """Take a remediation action."""
        return f"Action completed: {action} on {resource}"

    # Create tools
    tools = [
        StructuredTool.from_function(
            func=check_status,
            name="check_status",
            description="Check the current status of a resource",
        ),
        StructuredTool.from_function(
            func=get_events,
            name="get_events",
            description="Get recent events for a resource",
        ),
        StructuredTool.from_function(
            func=get_metrics,
            name="get_metrics",
            description="Get historical metrics for a resource",
        ),
        StructuredTool.from_function(
            func=check_changes,
            name="check_changes",
            description="Check recent changes to a resource",
        ),
        StructuredTool.from_function(
            func=create_alert,
            name="create_alert",
            description="Create an alert or incident",
        ),
        StructuredTool.from_function(
            func=take_action,
            name="take_action",
            description="Take a remediation action on a resource",
        ),
    ]

    # Create agent with higher recursion limit to allow multi-step
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    region = os.getenv("OCI_REGION", "us-chicago-1")
    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    chat_model = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=compartment_id,
        model_kwargs={"temperature": 0.2, "max_tokens": 2048, "top_p": 0.9},
        auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_file_location=os.path.expanduser("~/.oci/config"),
        disable_streaming="tool_calling",
        max_sequential_tool_calls=8,  # Allow up to 8 sequential tool calls
    )

    tool_node = ToolNode(tools=tools)
    model_with_tools = chat_model.bind_tools(tools)

    def call_model(state: MessagesState):
        """Call the model with tools bound."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)

        # OCI LIMITATION: Only allow ONE tool call at a time
        if (
            hasattr(response, "tool_calls")
            and response.tool_calls
            and len(response.tool_calls) > 1
        ):
            # Some models try to call multiple tools in parallel
            # Restrict to first tool only to avoid OCI API error
            response.tool_calls = [response.tool_calls[0]]

        return {"messages": [response]}

    def should_continue(state: MessagesState):
        """Check if the model wants to call a tool."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")
    agent = builder.compile()

    # System prompt that encourages multi-step investigation
    system_prompt = """You are a diagnostic assistant. When investigating
    issues, follow this workflow:

    1. Check current status
    2. Review recent events
    3. Analyze historical metrics
    4. Check for recent changes
    5. Create alert if needed
    6. Take remediation action if appropriate
    7. Provide final summary

    Call the necessary tools to gather information, then provide a
    comprehensive analysis."""

    # Invoke agent with a diagnostic scenario
    # Langgraph invoke signature is generic; passing dict is valid at runtime
    input_messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                "Investigate the payment-service resource. "
                "It has high memory usage and restarts. "
                "Determine root cause and recommend actions."
            )
        ),
    ]
    result = agent.invoke(
        {"messages": input_messages},  # type: ignore[arg-type]
        config={"recursion_limit": 25},  # Allow enough recursion for multi-step
    )

    messages = result["messages"]

    # Count tool calls
    tool_call_messages = [
        msg
        for msg in messages
        if (
            type(msg).__name__ == "AIMessage"
            and hasattr(msg, "tool_calls")
            and msg.tool_calls
        )
    ]
    tool_result_messages = [
        msg for msg in messages if type(msg).__name__ == "ToolMessage"
    ]

    # Verify multi-step orchestration worked
    msg = f"Should have made multiple tool calls (got {len(tool_call_messages)})"
    assert len(tool_call_messages) >= 2, msg

    # CRITICAL: Verify max_sequential_tool_calls limit was respected
    # The agent should stop at or before the limit (8 tool calls)
    # This is the key protection against infinite loops
    assert len(tool_call_messages) <= 8, (
        f"Too many tool calls ({len(tool_call_messages)}), "
        "max_sequential_tool_calls limit not enforced"
    )

    # Verify tool results were received
    assert len(tool_result_messages) >= 2, "Should have received multiple tool results"

    # Verify agent eventually stopped (didn't loop infinitely)
    # The final message might still have tool_calls if the agent hit
    # the max_sequential_tool_calls limit, which is expected behavior.
    # The key is that it STOPPED (didn't continue infinitely).
    final_message = messages[-1]
    assert type(final_message).__name__ in [
        "AIMessage",
        "ToolMessage",
    ], "Final message should be AIMessage or ToolMessage"

    # Verify the agent didn't hit infinite loop by checking message count
    # With max_sequential_tool_calls=8, we expect roughly:
    # System + Human + (AI + Tool) * 8 = ~18 messages maximum
    assert len(messages) <= 25, (
        f"Too many messages ({len(messages)}), possible infinite loop. "
        "The max_sequential_tool_calls limit should have stopped the agent."
    )

def test_gemini_parallel_tool_calls_manual(gemini_llm):
    """Direct reproduction of the Gemini parallel tool call bug.

    Without the flattening fix, step 2 fails with 400 INVALID_ARGUMENT:
    "Please ensure that the number of function response parts is equal
    to the number of function call parts of the function call turn."
    """
    llm = gemini_llm.bind_tools([get_weather, get_time])

    response = llm.invoke(
        "What is the weather AND the current time in New York City? Call both tools."
    )

    if not response.tool_calls:
        pytest.skip("Model did not make any tool calls")
    if len(response.tool_calls) < 2:
        pytest.skip(
            f"Model made {len(response.tool_calls)} tool call(s), "
            "need 2+ to test parallel flattening"
        )

    messages = [
        HumanMessage(
            content=(
                "What is the weather AND the current time in "
                "New York City? Call both tools."
            )
        ),
        response,
        *_execute_tool_calls(response),
    ]

    final = llm.invoke(messages)
    assert final.content, "Gemini should return a final text response"

def test_gemini_agent_with_parallel_tools(gemini_llm, weather_tool, time_tool):
    """Full LangGraph agent loop with Gemini parallel tool calls."""
    tools = [weather_tool, time_tool]
    tool_node = ToolNode(tools=tools)
    model_with_tools = gemini_llm.bind_tools(tools)

    def call_model(state: MessagesState):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if hasattr(last, "tool_calls") and last.tool_calls else END

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")
    agent = builder.compile()

    result = agent.invoke(
        {  # type: ignore[arg-type]
            "messages": [
                HumanMessage(
                    content="What is the weather AND the time in New York? Use both."
                )
            ]
        }
    )

    final = result["messages"][-1]
    assert type(final).__name__ == "AIMessage"
    assert final.content
    assert not (hasattr(final, "tool_calls") and final.tool_calls)

def test_gemini_single_tool_call_unaffected(gemini_llm):
    """Single tool calls still work (flattening is a no-op)."""
    llm = gemini_llm.bind_tools([get_weather])

    response = llm.invoke("What is the weather in Chicago?")

    if not response.tool_calls:
        pytest.skip("Model did not make a tool call")

    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert tc["name"] == "get_weather"

    messages = [
        HumanMessage(content="What is the weather in Chicago?"),
        response,
        ToolMessage(content=get_weather(**tc["args"]), tool_call_id=tc["id"]),
    ]
    final = llm.invoke(messages)
    assert final.content

def test_gemini_models_parallel_tool_calls(model_id: str):
    """Verify parallel flattening works on both Gemini models."""
    llm = _make_gemini_llm(model_id)
    llm_with_tools = llm.bind_tools([get_weather, get_time])

    response = llm_with_tools.invoke(
        "What is the weather and time in Chicago? Call both tools."
    )

    if not response.tool_calls:
        pytest.skip(f"{model_id}: Model did not make any tool calls")

    messages = [
        HumanMessage(content="What is the weather and time in Chicago? Call both."),
        response,
        *_execute_tool_calls(response),
    ]

    final = llm_with_tools.invoke(messages)
    assert final.content, f"{model_id}: should return a final response"

def test_gemini_result_correctness(gemini_llm):
    """Verify tool results are correctly paired after flattening."""
    llm = gemini_llm.bind_tools([get_weather])

    messages: List[BaseMessage] = [
        HumanMessage(content="What is the weather in Tokyo and London?"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "tc_tokyo", "name": "get_weather", "args": {"city": "Tokyo"}},
                {"id": "tc_london", "name": "get_weather", "args": {"city": "London"}},
            ],
        ),
        ToolMessage(content="Clear, 68F", tool_call_id="tc_tokyo"),
        ToolMessage(content="Overcast, 50F", tool_call_id="tc_london"),
    ]

    final = llm.invoke(messages)
    assert final.content

    content_lower = final.content.lower()
    assert any(w in content_lower for w in ["68", "clear"]), (
        f"Should mention Tokyo weather: {final.content}"
    )
    assert any(w in content_lower for w in ["50", "overcast"]), (
        f"Should mention London weather: {final.content}"
    )

def test_meta_llama_tool_result_guidance():
    """Test that tool_result_guidance helps Llama incorporate tool results.

    Reproduces Issue #28: without tool_result_guidance, Llama outputs raw JSON
    tool call syntax instead of natural language when using an agent.
    With tool_result_guidance=True, a system message guides the model to
    respond with natural language incorporating the tool results.
    """
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    endpoint = os.environ.get("OCI_GENAI_SERVICE_ENDPOINT")
    if not endpoint:
        region = os.getenv("OCI_REGION", "us-chicago-1")
        endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    chat = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        service_endpoint=endpoint,
        compartment_id=compartment_id,
        model_kwargs={"temperature": 0.0, "max_tokens": 500},
        auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_file_location=os.path.expanduser("~/.oci/config"),
        tool_result_guidance=True,
    )

    def _get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    from typing import Any

    from langchain.agents import create_agent

    agent: Any = create_agent(
        model=chat,
        tools=[_get_weather],
        system_prompt="You are a helpful assistant",
    )

    messages = [
        SystemMessage(content="You are an AI assistant."),
        HumanMessage(content="What is the weather in SF?"),
    ]

    response = agent.invoke({"messages": messages})
    final_message = response["messages"][-1]

    # Verify the model produced a final response
    assert final_message.content, "Should have generated a response"

    # Verify response is natural language, not raw JSON tool call syntax
    content = final_message.content
    # Check for raw JSON tool call syntax anywhere in response
    assert '{"name"' not in content, (
        f"Response contains raw JSON tool call syntax: {content[:200]}"
    )
    # Check for known Llama failure pattern where it re-explains tool calls
    assert "incorrect assumption" not in content.lower(), (
        f"Model failed to incorporate tool results: {content[:200]}"
    )


# --- libs/oracledb/tests/integration_tests/retrievers/test_hybrid.py ---

def test_hybrid_retrieval(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas

    drop_table_purge(connection, resource_names["table"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    preference = OracleVectorizerPreference.create_preference(
        vs, resource_names["pref"]
    )
    create_hybrid_index(connection, resource_names["index"], preference)

    retriever = OracleHybridSearchRetriever(
        vector_store=vs, idx_name=resource_names["index"], k=2
    )

    assert retriever.search_mode == "hybrid"

    query = "database questions"
    documents = retriever.invoke(query)
    assert len(documents) == 2
    assert documents[0].metadata["id"] == "cncpt_15.5.3.2.2_P4"
    assert "score" not in documents[0].metadata

    retriever = OracleHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        k=1,
        search_mode="semantic",
        return_scores=True,
    )
    query = "tablespace"
    documents = retriever.invoke(query)
    assert len(documents) == 1
    assert documents[0].metadata["id"] == "cncpt_15.5.5_P1"
    assert documents[0].metadata["vector_score"] > 0
    assert documents[0].metadata["text_score"] == 0

    retriever = OracleHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        k=1,
        search_mode="keyword",
        return_scores=True,
    )
    query = "preceding questions"
    documents = retriever.invoke(query)
    assert len(documents) == 1
    assert documents[0].metadata["id"] == "cncpt_15.5.3.2.2_P4"
    assert documents[0].metadata["text_score"] > 0
    assert documents[0].metadata["vector_score"] == 0

async def test_hybrid_retrieval_async(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_texts_and_metadatas,
) -> None:
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas

    drop_table_purge(connection, resource_names["table"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    preference = await OracleVectorizerPreference.acreate_preference(
        vs, resource_names["pref"]
    )
    await acreate_hybrid_index(aconnection, resource_names["index"], preference)

    retriever = OracleHybridSearchRetriever(
        vector_store=vs, idx_name=resource_names["index"], k=2
    )

    assert retriever.search_mode == "hybrid"

    query = "database questions"
    documents = await retriever.ainvoke(query)
    assert len(documents) == 2
    assert documents[0].metadata["id"] == "cncpt_15.5.3.2.2_P4"
    assert "score" not in documents[0].metadata

    retriever = OracleHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        k=1,
        search_mode="semantic",
        return_scores=True,
    )
    query = "tablespace"
    documents = await retriever.ainvoke(query)
    assert len(documents) == 1
    assert documents[0].metadata["id"] == "cncpt_15.5.5_P1"
    assert documents[0].metadata["vector_score"] > 0
    assert documents[0].metadata["text_score"] == 0

    retriever = OracleHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        k=1,
        search_mode="keyword",
        return_scores=True,
    )
    query = "preceding questions"
    documents = await retriever.ainvoke(query)
    assert len(documents) == 1
    assert documents[0].metadata["id"] == "cncpt_15.5.3.2.2_P4"
    assert documents[0].metadata["text_score"] > 0
    assert documents[0].metadata["vector_score"] == 0

def test_docstring_example_sync(
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_docstring_texts_and_metadatas,
) -> None:
    """
    Mirrors the synchronous docstring example:
    - Create preference
    - Create hybrid index
    - Build retriever with params including 'return'
    - Run a query
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_docstring_texts_and_metadatas
    drop_table_purge(connection, resource_names["table"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    pref = OracleVectorizerPreference.create_preference(vs, resource_names["pref"])

    # From docstring: provide additional index parameters (matching example)
    create_hybrid_index(
        connection,
        resource_names["index"],
        pref,
        params={
            "parallel": 4,
        },
    )

    # From docstring: include 'return' values in retriever params
    # (allowed; retriever sets defaults internally)
    retriever = OracleHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        search_mode="hybrid",
        k=2,
        return_scores=True,
    )
    docs = retriever.invoke("refund policy for premium plan")
    assert len(docs) >= 1
    # Ensure the refund document ranks for this query
    assert docs[0].metadata["id"] == "doc_refund"
    # Ensure expected score fields exist
    assert (
        "score" in docs[0].metadata
        and "text_score" in docs[0].metadata
        and "vector_score" in docs[0].metadata
    )

def test_retriever_params_validation_errors(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    """
    Validate error raises for invalid search params:
    - Disallow 'search_text' at top-level
    - Disallow 'search_text'/'search_vector'/'contains' inside 'vector'/'text'
    - Disallow bad params provided at call time as well
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    texts, metadatas = sample_texts_and_metadatas

    drop_table_purge(connection, resource_names["table"])
    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    pref = OracleVectorizerPreference.create_preference(vs, resource_names["pref"])
    create_hybrid_index(connection, resource_names["index"], pref)

    # Top-level search_text should raise
    with pytest.raises(
        ValueError, match="Cannot provide search_text as a parameter at the top level"
    ):
        OracleHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"search_text": "bad"},
        )

    # Nested vector.search_text and text.search_text should raise
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['vector'\]",
    ):
        OracleHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"vector": {"search_text": "bad"}},
        )

    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['text'\]",
    ):
        OracleHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"text": {"search_text": "bad"}},
        )

    # 'contains' under text should also raise (message mentions search_text)
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['text'\]",
    ):
        OracleHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"text": {"contains": "bad"}},
        )

    # Per-call invalid params should raise too
    retriever = OracleHybridSearchRetriever(
        vector_store=vs, idx_name=resource_names["index"]
    )
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['vector'\]",
    ):
        retriever.invoke("ok", params={"vector": {"search_text": "bad"}})
    with pytest.raises(
        ValueError, match="Cannot provide search_text as a parameter at the top level"
    ):
        retriever.invoke("ok", params={"search_text": "bad"})
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['text'\]",
    ):
        retriever.invoke("ok", params={"text": {"contains": "x"}})

def test_hybrid_score_weight_effects(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    """
    Validate 'score_weight' inside 'vector'/'text' influences overall score.
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    texts, metadatas = sample_texts_and_metadatas

    drop_table_purge(connection, resource_names["table"])
    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    pref = OracleVectorizerPreference.create_preference(vs, resource_names["pref"])
    create_hybrid_index(connection, resource_names["index"], pref)

    retriever = OracleHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        k=1,
        search_mode="hybrid",
        return_scores=True,
    )

    query = "preceding questions"
    docs_vec0 = retriever.invoke(query, params={"vector": {"score_weight": 1}})
    assert len(docs_vec0) == 1
    md = docs_vec0[0].metadata
    score1 = md["score"]

    docs_txt0 = retriever.invoke(query)
    assert len(docs_txt0) == 1
    md = docs_txt0[0].metadata
    score2 = md["score"]

    assert score1 != score2

def test_create_hybrid_index_with_vector_store(
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_texts_and_metadatas,
) -> None:
    """
    Create the hybrid index by passing the vector_store directly (without an explicit
    OracleVectorizerPreference) and verify basic retrieval works.
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    # Create hybrid index by providing vector_store (preference is created internally)
    create_hybrid_index(connection, resource_names["index"], vector_store=vs)

    # Smoke-test retrieval to confirm the index is usable
    retriever = OracleHybridSearchRetriever(
        vector_store=vs, idx_name=resource_names["index"], k=1
    )
    docs = retriever.invoke("database")
    assert len(docs) >= 1

async def test_create_hybrid_index_async_with_vector_store(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_texts_and_metadatas,
) -> None:
    """
    Async variant: create the hybrid index by passing vector_store directly and
    verify retrieval works.
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    # Create hybrid index by providing vector_store (preference is created internally)
    await acreate_hybrid_index(aconnection, resource_names["index"], vector_store=vs)

    # Smoke-test retrieval to confirm the index is usable
    retriever = OracleHybridSearchRetriever(
        vector_store=vs, idx_name=resource_names["index"], k=1
    )
    docs = await retriever.ainvoke("database")
    assert len(docs) >= 1

async def test_docstring_example_async(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_docstring_texts_and_metadatas,
) -> None:
    """
    Mirrors the async docstring example:
    - Async preference creation
    - Async hybrid index creation
    - Async retrieval
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_docstring_texts_and_metadatas
    drop_table_purge(connection, resource_names["table"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    pref = await OracleVectorizerPreference.acreate_preference(
        vs, resource_names["pref"]
    )
    await acreate_hybrid_index(
        aconnection,
        resource_names["index"],
        pref,
        params={
            "parallel": 4,
        },
    )

    retriever = OracleHybridSearchRetriever(
        vector_store=vs, idx_name=resource_names["index"], k=2, return_scores=True
    )
    results = await retriever.ainvoke("latest SLA")
    assert len(results) >= 1
    # Ensure the SLA document ranks for this query
    assert results[0].metadata["id"] == "doc_sla"
    assert (
        "score" in results[0].metadata
        and "vector_score" in results[0].metadata
        and "text_score" in results[0].metadata
    )

async def test_retriever_params_validation_errors_async(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_texts_and_metadatas,
) -> None:
    """
    Async counterpart for params validation;
        instantiation/checks should raise similarly.
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    pref = await OracleVectorizerPreference.acreate_preference(
        vs, resource_names["pref"]
    )
    await acreate_hybrid_index(aconnection, resource_names["index"], pref)

    with pytest.raises(
        ValueError, match="Cannot provide search_text as a parameter at the top level"
    ):
        OracleHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"search_text": "bad"},
        )
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['vector'\]",
    ):
        OracleHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"vector": {"search_text": "bad"}},
        )
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['text'\]",
    ):
        OracleHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"text": {"contains": "x"}},
        )

    retriever = OracleHybridSearchRetriever(
        vector_store=vs, idx_name=resource_names["index"]
    )
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['text'\]",
    ):
        await retriever.ainvoke("ok", params={"text": {"search_text": "bad"}})

async def test_hybrid_score_weight_effects_async(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_texts_and_metadatas,
) -> None:
    """
    Async validation for score_weight effects on overall score.
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    pref = await OracleVectorizerPreference.acreate_preference(
        vs, resource_names["pref"]
    )
    await acreate_hybrid_index(aconnection, resource_names["index"], pref)

    retriever = OracleHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        k=1,
        search_mode="hybrid",
        return_scores=True,
    )

    query = "preceding questions"
    docs_vec0 = await retriever.ainvoke(query, params={"vector": {"score_weight": 1}})
    assert len(docs_vec0) == 1
    md = docs_vec0[0].metadata
    score1 = md["score"]

    docs_txt0 = await retriever.ainvoke(query)
    assert len(docs_txt0) == 1
    md = docs_txt0[0].metadata
    score2 = md["score"]

    assert score1 != score2


# --- libs/oracledb/tests/integration_tests/retrievers/test_text.py ---

def test_text_vs_sync_exact_and_scores_and_returned_columns_default(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    """
    - Create OracleVS from texts
    - Create Oracle Text SEARCH INDEX on 'text'
    - Run exact search and validate top document, metadata, and scores
    - Validate default returned_columns behavior (other column auto-included)
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    # Create Oracle Text index on text column
    create_text_index(
        connection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs,
    )

    # Build retriever; returned_columns defaults to ["metadata"] for vs+text
    retriever = OracleTextSearchRetriever(
        vector_store=vs,
        k=1,
        return_scores=True,
    )
    docs = retriever.invoke("tablespace")
    assert len(docs) == 1
    # Ensure the 'tablespace' document ranks first
    assert docs[0].metadata["id"] == "cncpt_15.5.5_P1"
    # Score is present
    assert "score" in docs[0].metadata
    # Ensure user metadata fields are preserved
    assert "link" in docs[0].metadata

    # Override k at call time, expect 1 result
    docs2 = retriever.invoke("database", k=1)
    assert len(docs2) == 1

def test_text_vs_sync_fuzzy_on_text(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    """
    - Create index on 'text'
    - Use fuzzy search to match misspelled term
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    create_text_index(
        connection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs,
        column_name="text",
    )

    retriever = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        fuzzy=True,
        k=1,
        return_scores=True,
    )
    # Misspelled "tablespace"
    docs = retriever.invoke("tabespace")
    assert len(docs) == 1
    assert docs[0].metadata["id"] == "cncpt_15.5.5_P1"
    assert docs[0].metadata["score"] > 0

def test_text_raw_table_sync_exact_and_scores_and_returned_columns(
    connection, cleanup, resource_names
) -> None:
    """
    - Create a user table with (title, body)
    - Create Oracle Text SEARCH INDEX on 'body'
    - Build retriever using client+table_name+column_name
    - Validate ranking, returned_columns and score
    """
    _create_raw_table_and_data(connection, resource_names["table_raw"])

    create_text_index(
        connection,
        idx_name=resource_names["index_raw"],
        table_name=resource_names["table_raw"],
        column_name="body",
    )

    retriever = OracleTextSearchRetriever(
        client=connection,
        table_name=resource_names["table_raw"],
        column_name="body",
        k=1,
        return_scores=True,
        returned_columns=["title"],
    )
    docs = retriever.invoke("refund policy")
    assert len(docs) == 1
    assert "refund policy" in docs[0].page_content.lower()
    assert docs[0].metadata.get("title") == "Refund"
    assert docs[0].metadata.get("score", 0) > 0

def test_text_raw_table_sync_fuzzy_search(connection, cleanup, resource_names) -> None:
    """
    - Create raw table and index
    - Use fuzzy search to match misspelled query
    """
    _create_raw_table_and_data(connection, resource_names["table_raw"])

    create_text_index(
        connection,
        idx_name=resource_names["index_raw"],
        table_name=resource_names["table_raw"],
        column_name="body",
    )

    retriever = OracleTextSearchRetriever(
        client=connection,
        table_name=resource_names["table_raw"],
        column_name="body",
        fuzzy=True,
        k=1,
        return_scores=True,
        returned_columns=["title"],
    )
    docs = retriever.invoke("refnd polciy")
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Refund"
    assert docs[0].metadata["score"] > 0

def test_text_returned_columns_dedup_sync(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    """
    Ensure returned_columns does not duplicate the main column:
    - For OracleVS/text, passing returned_columns=['text'] should not duplicate content
    - For raw table/body, passing returned_columns including 'body' should not duplicate
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    create_text_index(
        connection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs,
        column_name="text",
    )

    retriever_vs = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        returned_columns=["text"],  # should be de-duplicated away
        k=1,
    )
    docs_vs = retriever_vs.invoke("tablespace")
    assert len(docs_vs) == 1
    # metadata should not contain a duplicate 'text' key
    assert "text" not in docs_vs[0].metadata

    # Raw table case
    _create_raw_table_and_data(connection, resource_names["table_raw"])
    create_text_index(
        connection,
        idx_name=resource_names["index_raw"],
        table_name=resource_names["table_raw"],
        column_name="body",
    )
    retriever_raw = OracleTextSearchRetriever(
        client=connection,
        table_name=resource_names["table_raw"],
        column_name="body",
        returned_columns=["title", "body"],  # 'body' should be de-duplicated
        k=1,
    )
    docs_raw = retriever_raw.invoke("refund")
    assert len(docs_raw) == 1
    assert "body" not in docs_raw[0].metadata
    assert "title" in docs_raw[0].metadata

async def test_text_vs_async_exact_and_scores(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_texts_and_metadatas,
) -> None:
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs,
    )

    retriever = OracleTextSearchRetriever(
        vector_store=vs,
        k=1,
        return_scores=True,
    )
    docs = await retriever.ainvoke("tablespace")
    assert len(docs) == 1
    assert docs[0].metadata["id"] == "cncpt_15.5.5_P1"
    assert docs[0].metadata["score"] > 0

async def test_text_vs_async_fuzzy_on_text(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_texts_and_metadatas,
) -> None:
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs,
        column_name="text",
    )

    retriever = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        fuzzy=True,
        k=1,
        return_scores=True,
    )
    docs = await retriever.ainvoke("tabespace")
    assert len(docs) == 1
    assert docs[0].metadata["id"] == "cncpt_15.5.5_P1"
    assert docs[0].metadata["score"] > 0

async def test_text_raw_table_async_exact_and_scores(
    aconnection, connection, cleanup, resource_names
) -> None:
    _create_raw_table_and_data(connection, resource_names["table_raw"])

    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_raw"],
        table_name=resource_names["table_raw"],
        column_name="body",
    )

    retriever = OracleTextSearchRetriever(
        client=aconnection,
        table_name=resource_names["table_raw"],
        column_name="body",
        k=1,
        return_scores=True,
        returned_columns=["title"],
    )
    docs = await retriever.ainvoke("SLA")
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "SLA"
    assert docs[0].metadata["score"] > 0

async def test_text_raw_table_async_fuzzy(
    aconnection, connection, cleanup, resource_names
) -> None:
    _create_raw_table_and_data(connection, resource_names["table_raw"])

    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_raw"],
        table_name=resource_names["table_raw"],
        column_name="body",
    )

    retriever = OracleTextSearchRetriever(
        client=aconnection,
        table_name=resource_names["table_raw"],
        column_name="body",
        fuzzy=True,
        k=1,
        return_scores=True,
        returned_columns=["title"],
    )
    docs = await retriever.ainvoke("refnd polciy")
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Refund"
    assert docs[0].metadata["score"] > 0

async def test_text_vs_async_literal_true_vs_false_operator_semantics(
    aconnection, connection, cleanup, resource_names
) -> None:
    """
    Ensure Oracle Text operators are applied only when operator_search=True.
    Using NEAR(...) should:
      - return a match when operator_search=True
      - return no matches when operator_search=False (treated as literal text)
    """
    proxy = ""
    model = OracleEmbeddings(
        conn=connection,
        params={"provider": "database", "model": "allminilm"},
        proxy=proxy,
    )

    texts = [
        "Refund policy for premium plan allows refunds within 30 days",
        "Completely unrelated sentence",
    ]
    metas = [
        {"id": "doc_refund"},
        {"id": "doc_other"},
    ]

    # Build VS and index
    drop_table_purge(connection, resource_names["table_vs"])
    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metas,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs,
        column_name="text",
    )

    query = "NEAR((policy, refund), 2, TRUE)"
    retr_true = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=False,
        k=1,
        return_scores=True,
    )
    docs_false = await retr_true.ainvoke(query)
    assert len(docs_false) == 1
    assert docs_false[0].metadata.get("id") == "doc_refund"

    # operator_search=True -> operator semantics applied, expect no doc
    retr_false = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=True,
        k=1,
        return_scores=True,
    )
    docs_true = await retr_false.ainvoke(query)
    assert len(docs_true) == 0

def test_text_vs_sync_literal_true_vs_false_operator_semantics(
    connection, cleanup, resource_names
) -> None:
    """
    Sync counterpart for operator_search True vs False behavior using NEAR(...).
    """
    texts = [
        "Refund policy for premium plan allows refunds within 30 days",
        "Completely unrelated sentence",
    ]
    metas = [
        {"id": "doc_refund"},
        {"id": "doc_other"},
    ]
    vs = _build_vs_with_texts(
        connection,
        resource_names["table_vs"],
        resource_names["index_vs_text"],
        texts,
        metas,
    )

    query = "NEAR((policy, refund), 2, TRUE)"
    retr_true = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=False,
        k=1,
        return_scores=True,
    )
    docs_false = retr_true.invoke(query)
    assert len(docs_false) == 1
    assert docs_false[0].metadata.get("id") == "doc_refund"

    retr_false = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=True,
        k=1,
        return_scores=True,
    )
    docs_true = retr_false.invoke(query)
    assert len(docs_true) == 0

def test_text_vs_fuzzy_word(connection, cleanup, resource_names) -> None:
    """
    Test fuzzy search
    """
    texts = [
        "Refund policy for premium plan allows refunds within 30 days",
        "Completely unrelated sentence",
    ]
    metas = [
        {"id": "doc_refund"},
        {"id": "doc_other"},
    ]
    vs = _build_vs_with_texts(
        connection,
        resource_names["table_vs"],
        resource_names["index_vs_text"],
        texts,
        metas,
    )

    retr_true = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=False,
        k=1,
        return_scores=True,
        fuzzy=True,
    )

    query = "policy premium plan near"
    docs_true = retr_true.invoke(query)
    assert len(docs_true) == 1

    query = ""
    docs_true = retr_true.invoke(query)
    assert len(docs_true) == 0


# --- libs/oracledb/tests/integration_tests/vectorstores/test_oraclevs.py ---

def test_create_hnsw_index_test() -> None:
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    # 1. Table_name - TB1
    #    New Index
    #    distance_strategy - DistanceStrategy.Dot_product
    # Expectation:Index created
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    vs = OracleVS(connection, model1, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs)

    # 2. Creating same index again
    #    Table_name - TB1
    # Expectation:Nothing happens
    with pytest.raises(RuntimeError, match="such column list already indexed"):
        create_index(connection, vs)
        drop_index_if_exists(connection, "HNSW")
    drop_table_purge(connection, "TB1")

    # 3. Create index with following parameters:
    #    idx_name - hnsw_idx2
    #    idx_type - HNSW
    # Expectation:Index created
    vs = OracleVS(connection, model1, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs, params={"idx_name": "hnsw_idx2", "idx_type": "HNSW"})
    drop_index_if_exists(connection, "hnsw_idx2")
    drop_table_purge(connection, "TB2")

    # 4. Table Name - TB1
    #    idx_name - "हिन्दी"
    #    idx_type - HNSW
    # Expectation:Index created
    vs = OracleVS(connection, model1, "TB3", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs, params={"idx_name": '"हिन्दी"', "idx_type": "HNSW"})
    drop_index_if_exists(connection, '"हिन्दी"')
    drop_table_purge(connection, "TB3")

    # 5. idx_name passed empty
    # Expectation:ORA-01741: illegal zero-length identifier
    with pytest.raises(ValueError):
        vs = OracleVS(connection, model1, "TB4", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(connection, vs, params={"idx_name": '""', "idx_type": "HNSW"})
        drop_index_if_exists(connection, '""')
    drop_table_purge(connection, "TB4")

    # 6. idx_type left empty
    # Expectation:Index created
    vs = OracleVS(connection, model1, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs, params={"idx_name": "Hello", "idx_type": ""})
    drop_index_if_exists(connection, "Hello")
    drop_table_purge(connection, "TB5")

    # 7. efconstruction passed as parameter but not neighbours
    # Expectation:Index created
    vs = OracleVS(connection, model1, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(
        connection,
        vs,
        params={"idx_name": "idx11", "efConstruction": 100, "idx_type": "HNSW"},
    )
    drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB7")

    # 8. efconstruction passed as parameter as well as neighbours
    # (for this idx_type parameter is also necessary)
    # Expectation:Index created
    vs = OracleVS(connection, model1, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 100,
            "neighbors": 80,
            "idx_type": "HNSW",
        },
    )
    drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB8")

    #  9. Limit of Values for(integer values):
    #     parallel
    #     efConstruction
    #     Neighbors
    #     Accuracy
    #     0<Accuracy<=100
    #     0<Neighbour<=2048
    #     0<efConstruction<=65535
    #     0<parallel<=255
    # Expectation:Index created
    drop_table_purge(connection, "TB9")
    vs = OracleVS(connection, model1, "TB9", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 65535,
            "neighbors": 2048,
            "idx_type": "HNSW",
            "parallel": 255,
        },
    )
    drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB9")
    # index not created:
    with pytest.raises(RuntimeError):
        vs = OracleVS(connection, model1, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 0,
                "neighbors": 2048,
                "idx_type": "HNSW",
                "parallel": 255,
            },
        )
        drop_index_if_exists(connection, "idx11")
    # index not created:
    with pytest.raises(RuntimeError):
        vs = OracleVS(connection, model1, "TB11", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 100,
                "neighbors": 0,
                "idx_type": "HNSW",
                "parallel": 255,
            },
        )
        drop_index_if_exists(connection, "idx11")
    # index not created
    with pytest.raises(RuntimeError):
        vs = OracleVS(connection, model1, "TB12", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 100,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 0,
            },
        )
        drop_index_if_exists(connection, "idx11")
    # index not created
    with pytest.raises(RuntimeError):
        vs = OracleVS(connection, model1, "TB13", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 10,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 10,
                "accuracy": 120,
            },
        )
        drop_index_if_exists(connection, "idx11")
    # with negative values/out-of-bound values for all 4 of them, we get the same errors
    # Expectation:Index not created
    with pytest.raises(RuntimeError):
        vs = OracleVS(connection, model1, "TB14", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 200,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": "hello",
                "accuracy": 10,
            },
        )
        drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB10")
    drop_table_purge(connection, "TB11")
    drop_table_purge(connection, "TB12")
    drop_table_purge(connection, "TB13")
    drop_table_purge(connection, "TB14")

    # 10. Table_name as <schema_name.table_name>
    # Expectation:Index created
    vs = OracleVS(connection, model1, "TB15", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 200,
            "neighbors": 100,
            "idx_type": "HNSW",
            "parallel": 8,
            "accuracy": 10,
        },
    )
    drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB15")

    # 11. index_name as <schema_name.index_name>
    # Expectation:U1 not present
    with pytest.raises(RuntimeError):
        vs = OracleVS(
            connection, model1, "U1.TB16", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        create_index(
            connection,
            vs,
            params={
                "idx_name": "U1.idx11",
                "efConstruction": 200,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 8,
                "accuracy": 10,
            },
        )
        drop_index_if_exists(connection, "U1.idx11")
        drop_table_purge(connection, "TB16")

    # 12. Index_name size >129
    # Expectation:Index not created
    with pytest.raises(RuntimeError):
        vs = OracleVS(connection, model1, "TB17", DistanceStrategy.EUCLIDEAN_DISTANCE)
        create_index(connection, vs, params={"idx_name": "x" * 129, "idx_type": "HNSW"})
        drop_index_if_exists(connection, "x" * 129)
    drop_table_purge(connection, "TB17")

    # 13. Index_name size 128
    # Expectation:Index created
    vs = OracleVS(connection, model1, "TB18", DistanceStrategy.EUCLIDEAN_DISTANCE)
    create_index(connection, vs, params={"idx_name": "x" * 128, "idx_type": "HNSW"})
    drop_index_if_exists(connection, "x" * 128)
    drop_table_purge(connection, "TB18")

async def test_create_hnsw_index_test_async() -> None:
    try:
        connection = await oracledb.connect_async(
            user=username, password=password, dsn=dsn
        )
    except Exception:
        sys.exit(1)
    # 1. Table_name - TB1
    #    New Index
    #    distance_strategy - DistanceStrategy.Dot_product
    # Expectation:Index created
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    vs = await OracleVS.acreate(
        connection, model1, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    await acreate_index(connection, vs)

    # 2. Creating same index again
    #    Table_name - TB1
    # Expectation:Without index name, error happens
    with pytest.raises(RuntimeError, match="such column list already indexed"):
        await acreate_index(connection, vs)
        await adrop_index_if_exists(connection, "HNSW")
    await adrop_table_purge(connection, "TB1")

    # 3. Create index with following parameters:
    #    idx_name - hnsw_idx2
    #    idx_type - HNSW
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    await acreate_index(
        connection, vs, params={"idx_name": "hnsw_idx2", "idx_type": "HNSW"}
    )
    await adrop_index_if_exists(connection, "hnsw_idx2")
    await adrop_table_purge(connection, "TB2")

    # 4. Table Name - TB1
    #    idx_name - "हिन्दी"
    #    idx_type - HNSW
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB3", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    await acreate_index(
        connection, vs, params={"idx_name": '"हिन्दी"', "idx_type": "HNSW"}
    )
    await adrop_index_if_exists(connection, '"हिन्दी"')
    await adrop_table_purge(connection, "TB3")

    # 5. idx_name passed empty
    # Expectation:ORA-01741: illegal zero-length identifier
    with pytest.raises(ValueError):
        vs = await OracleVS.acreate(
            connection, model1, "TB4", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        await acreate_index(
            connection, vs, params={"idx_name": '""', "idx_type": "HNSW"}
        )
        await adrop_index_if_exists(connection, '""')
    await adrop_table_purge(connection, "TB4")

    # 6. idx_type left empty
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    await acreate_index(connection, vs, params={"idx_name": "Hello", "idx_type": ""})
    await adrop_index_if_exists(connection, "Hello")
    await adrop_table_purge(connection, "TB5")

    # 7. efconstruction passed as parameter but not neighbours
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    await acreate_index(
        connection,
        vs,
        params={"idx_name": "idx11", "efConstruction": 100, "idx_type": "HNSW"},
    )
    await adrop_index_if_exists(connection, "idx11")
    await adrop_table_purge(connection, "TB7")

    # 8. efconstruction passed as parameter as well as neighbours
    # (for this idx_type parameter is also necessary)
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    await acreate_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 100,
            "neighbors": 80,
            "idx_type": "HNSW",
        },
    )
    await adrop_index_if_exists(connection, "idx11")
    await adrop_table_purge(connection, "TB8")

    #  9. Limit of Values for(integer values):
    #     parallel
    #     efConstruction
    #     Neighbors
    #     Accuracy
    #     0<Accuracy<=100
    #     0<Neighbour<=2048
    #     0<efConstruction<=65535
    #     0<parallel<=255
    # Expectation:Index created
    await adrop_table_purge(connection, "TB9")
    vs = await OracleVS.acreate(
        connection, model1, "TB9", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    await acreate_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 65535,
            "neighbors": 2048,
            "idx_type": "HNSW",
            "parallel": 255,
        },
    )
    await adrop_index_if_exists(connection, "idx11")
    await adrop_table_purge(connection, "TB9")
    # index not created:
    with pytest.raises(RuntimeError):
        vs = await OracleVS.acreate(
            connection, model1, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        await acreate_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 0,
                "neighbors": 2048,
                "idx_type": "HNSW",
                "parallel": 255,
            },
        )
        await adrop_index_if_exists(connection, "idx11")

    # index not created:
    with pytest.raises(RuntimeError):
        vs = await OracleVS.acreate(
            connection, model1, "TB11", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        await acreate_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 100,
                "neighbors": 0,
                "idx_type": "HNSW",
                "parallel": 255,
            },
        )
        await adrop_index_if_exists(connection, "idx11")
    # index not created
    with pytest.raises(RuntimeError):
        vs = await OracleVS.acreate(
            connection, model1, "TB12", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        await acreate_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 100,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 0,
            },
        )
        await adrop_index_if_exists(connection, "idx11")
    # index not created
    with pytest.raises(RuntimeError):
        vs = await OracleVS.acreate(
            connection, model1, "TB13", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        await acreate_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 10,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 10,
                "accuracy": 120,
            },
        )
        await adrop_index_if_exists(connection, "idx11")
    # with negative values/out-of-bound values for all 4 of them, we get the same errors
    # Expectation:Index not created
    with pytest.raises(RuntimeError):
        vs = await OracleVS.acreate(
            connection, model1, "TB14", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        await acreate_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 200,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": "hello",
                "accuracy": 10,
            },
        )
        await adrop_index_if_exists(connection, "idx11")
    await adrop_table_purge(connection, "TB10")
    await adrop_table_purge(connection, "TB11")
    await adrop_table_purge(connection, "TB12")
    await adrop_table_purge(connection, "TB13")
    await adrop_table_purge(connection, "TB14")

    # 10. Table_name as <schema_name.table_name>
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB15", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    await acreate_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 200,
            "neighbors": 100,
            "idx_type": "HNSW",
            "parallel": 8,
            "accuracy": 10,
        },
    )
    await adrop_index_if_exists(connection, "idx11")
    await adrop_table_purge(connection, "TB15")

    # 11. index_name as <schema_name.index_name>
    # Expectation:U1 not present
    with pytest.raises(RuntimeError):
        vs = await OracleVS.acreate(
            connection, model1, "U1.TB16", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        await acreate_index(
            connection,
            vs,
            params={
                "idx_name": "U1.idx11",
                "efConstruction": 200,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 8,
                "accuracy": 10,
            },
        )
        await adrop_index_if_exists(connection, "U1.idx11")
        await adrop_table_purge(connection, "TB16")

    # 12. Index_name size >129
    # Expectation:Index not created
    with pytest.raises(RuntimeError):
        vs = await OracleVS.acreate(
            connection, model1, "TB17", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        await acreate_index(
            connection, vs, params={"idx_name": "x" * 129, "idx_type": "HNSW"}
        )
        await adrop_index_if_exists(connection, "x" * 129)
    await adrop_table_purge(connection, "TB17")

    # 13. Index_name size 128
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB18", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    await acreate_index(
        connection, vs, params={"idx_name": "x" * 128, "idx_type": "HNSW"}
    )
    await adrop_index_if_exists(connection, "x" * 128)
    await adrop_table_purge(connection, "TB18")

def test_add_texts_test() -> None:
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    # 1. Add 2 records to table
    # Expectation:Successful
    texts = ["Rohan", "Shailendra"]
    metadata = [
        {"id": "100", "link": "Document Example Test 1"},
        {"id": "101", "link": "Document Example Test 2"},
    ]
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = OracleVS(connection, model, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_obj.add_texts(texts, metadata)
    drop_table_purge(connection, "TB1")

    # 2. Add record but metadata is not there
    # Expectation:An exception occurred :: Either specify an 'ids' list or
    # 'metadatas' with an 'id' attribute for each element.
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = OracleVS(connection, model, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts2 = ["Sri Ram", "Krishna"]
    vs_obj.add_texts(texts2)
    drop_table_purge(connection, "TB2")

    # 3. Add record with ids option
    #    ids are passed as string
    #    ids are passed as empty string
    #    ids are passed as multi-line string
    #    ids are passed as "<string>"
    # Expectations:
    # Successful
    # Successful
    # Successful
    # Successful

    vs_obj = OracleVS(connection, model, "TB4", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids3 = ["114", "124"]
    vs_obj.add_texts(texts2, ids=ids3)
    drop_table_purge(connection, "TB4")

    vs_obj = OracleVS(connection, model, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids4 = ["", "134"]
    vs_obj.add_texts(texts2, ids=ids4)
    drop_table_purge(connection, "TB5")

    vs_obj = OracleVS(connection, model, "TB6", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids5 = [
        """Good afternoon
    my friends""",
        "India",
    ]
    vs_obj.add_texts(texts2, ids=ids5)
    drop_table_purge(connection, "TB6")

    vs_obj = OracleVS(connection, model, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids6 = ['"Good afternoon"', '"India"']
    vs_obj.add_texts(texts2, ids=ids6)
    assert len(vs_obj.add_texts(texts2, ids=ids6)) == 0
    drop_table_purge(connection, "TB7")

    # 4. Add records with ids and metadatas
    # Expectation:Successful
    vs_obj = OracleVS(connection, model, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts3 = ["Sri Ram 6", "Krishna 6"]
    ids7 = ["1", "2"]
    metadata = [
        {"id": "102", "link": "Document Example", "stream": "Science"},
        {"id": "104", "link": "Document Example 45"},
    ]
    vs_obj.add_texts(texts3, metadata, ids=ids7)
    drop_table_purge(connection, "TB8")

    # 5. Add 10000 records
    # Expectation:Successful
    vs_obj = OracleVS(connection, model, "TB9", DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts4 = ["Sri Ram{0}".format(i) for i in range(1, 10000)]
    ids8 = ["Hello{0}".format(i) for i in range(1, 10000)]
    vs_obj.add_texts(texts4, ids=ids8)
    drop_table_purge(connection, "TB9")

    # 6. Add 2 different record concurrently
    # Expectation:Successful
    def add(val: str) -> None:
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vs_obj = OracleVS(
            connection, model, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        texts5 = [val]
        ids9 = texts5
        vs_obj.add_texts(texts5, ids=ids9)

    thread_1 = threading.Thread(target=add, args=("Sri Ram"))
    thread_2 = threading.Thread(target=add, args=("Sri Krishna"))
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()
    drop_table_purge(connection, "TB10")

    # 8. create object with table name of type <schema_name.table_name>
    # Expectation:U1 does not exist
    with pytest.raises(RuntimeError):
        vs_obj = OracleVS(connection, model, "U1.TB14", DistanceStrategy.DOT_PRODUCT)
        for i in range(1, 10):
            texts7 = ["Yash{0}".format(i)]
            ids13 = ["1234{0}".format(i)]
            vs_obj.add_texts(texts7, ids=ids13)
        drop_table_purge(connection, "TB14")

async def test_add_texts_test_async() -> None:
    try:
        connection = await oracledb.connect_async(
            user=username, password=password, dsn=dsn
        )
    except Exception:
        sys.exit(1)
    # 1. Add 2 records to table
    # Expectation:Successful
    texts = ["Rohan", "Shailendra"]
    metadata = [
        {"id": "100", "link": "Document Example Test 1"},
        {"id": "101", "link": "Document Example Test 2"},
    ]
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = await OracleVS.acreate(
        connection, model, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    await vs_obj.aadd_texts(texts, metadata)
    await adrop_table_purge(connection, "TB1")

    # 2. Add record but metadata is not there
    # Expectation:An exception occurred :: Either specify an 'ids' list or
    # 'metadatas' with an 'id' attribute for each element.
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = await OracleVS.acreate(
        connection, model, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    texts2 = ["Sri Ram", "Krishna"]
    await vs_obj.aadd_texts(texts2)
    await adrop_table_purge(connection, "TB2")

    # 3. Add record with ids option
    #    ids are passed as string
    #    ids are passed as empty string
    #    ids are passed as multi-line string
    #    ids are passed as "<string>"
    # Expectations:
    # Successful
    # Successful
    # Successful
    # Successful

    vs_obj = await OracleVS.acreate(
        connection, model, "TB4", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    ids3 = ["114", "124"]
    await vs_obj.aadd_texts(texts2, ids=ids3)
    await adrop_table_purge(connection, "TB4")

    vs_obj = await OracleVS.acreate(
        connection, model, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    ids4 = ["", "134"]
    await vs_obj.aadd_texts(texts2, ids=ids4)
    await adrop_table_purge(connection, "TB5")

    vs_obj = await OracleVS.acreate(
        connection, model, "TB6", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    ids5 = [
        """Good afternoon
    my friends""",
        "India",
    ]
    await vs_obj.aadd_texts(texts2, ids=ids5)
    await adrop_table_purge(connection, "TB6")

    vs_obj = await OracleVS.acreate(
        connection, model, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    ids6 = ['"Good afternoon"', '"India"']
    await vs_obj.aadd_texts(texts2, ids=ids6)
    assert len(await vs_obj.aadd_texts(texts2, ids=ids6)) == 0
    await adrop_table_purge(connection, "TB7")

    # 4. Add records with ids and metadatas
    # Expectation:Successful
    vs_obj = await OracleVS.acreate(
        connection, model, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    texts3 = ["Sri Ram 6", "Krishna 6"]
    ids7 = ["1", "2"]
    metadata = [
        {"id": "102", "link": "Document Example", "stream": "Science"},
        {"id": "104", "link": "Document Example 45"},
    ]
    await vs_obj.aadd_texts(texts3, metadata, ids=ids7)
    await adrop_table_purge(connection, "TB8")

    # 5. Add 10000 records
    # Expectation:Successful
    vs_obj = await OracleVS.acreate(
        connection, model, "TB9", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    texts4 = ["Sri Ram{0}".format(i) for i in range(1, 10000)]
    ids8 = ["Hello{0}".format(i) for i in range(1, 10000)]
    await vs_obj.aadd_texts(texts4, ids=ids8)
    await adrop_table_purge(connection, "TB9")

    # 6. Add 2 different record concurrently
    # Expectation:Successful
    async def add(val: str) -> None:
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vs_obj = await OracleVS.acreate(
            connection, model, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        texts5 = [val]
        ids9 = texts5
        await vs_obj.aadd_texts(texts5, ids=ids9)

    task_1 = asyncio.create_task(add("Sri Ram"))
    task_2 = asyncio.create_task(add("Sri Krishna"))

    await asyncio.gather(task_1, task_2)
    await adrop_table_purge(connection, "TB10")

    # 8. create object with table name of type <schema_name.table_name>
    # Expectation:U1 does not exist
    with pytest.raises(RuntimeError):
        vs_obj = await OracleVS.acreate(
            connection, model, "U1.TB14", DistanceStrategy.DOT_PRODUCT
        )
        for i in range(1, 10):
            texts7 = ["Yash{0}".format(i)]
            ids13 = ["1234{0}".format(i)]
            await vs_obj.aadd_texts(texts7, ids=ids13)
        await adrop_table_purge(connection, "TB14")

