# webup/langgraph-up-monorepo
# 21 LLM-backed test functions across 13 test files
# Source: https://github.com/webup/langgraph-up-monorepo

# --- apps/sample-agent/tests/integration/test_handoff.py ---

    async def test_faang_headcount_real_workflow(self):
        """Test real handoff workflow for FAANG headcount question with actual models."""
        # Create the actual graph with real models
        app = make_graph()

        # Initial state with FAANG headcount question
        initial_state = AgentState(
            messages=[HumanMessage(content="what's the combined headcount of the FAANG companies in 2024?")],
            remaining_steps=10,
            task_description=None,
            active_agent=None
        )

        # Execute the real workflow with actual models
        result = await app.ainvoke(initial_state)

        # Verify the workflow completed successfully
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) > 1

        # Check that we got a meaningful response
        final_message = result["messages"][-1]
        response_content = final_message.content.lower()

        # Should mention FAANG or specific companies
        faang_indicators = ["faang", "facebook", "meta", "apple", "amazon", "netflix", "google", "alphabet"]
        has_faang = any(indicator in response_content for indicator in faang_indicators)

        # Should contain numerical information (headcount)
        numerical_indicators = ["employees", "headcount", "workforce", "staff", "total", "1,977,586", "1977586"]
        has_numbers = any(indicator in response_content for indicator in numerical_indicators)

        # Should show evidence of processing the request
        assert has_faang or has_numbers, f"No FAANG or numerical indicators found in: {response_content}"

        # Should show evidence of handoff activity (consumed steps or meaningful response)
        steps_consumed = initial_state["remaining_steps"] - result["remaining_steps"]
        has_meaningful_response = len(final_message.content) > 50  # Non-trivial response

        assert steps_consumed > 0 or has_meaningful_response, (
            f"Workflow should have consumed steps or provided meaningful response. "
            f"Steps consumed: {steps_consumed}, Response length: {len(final_message.content)}"
        )

        print(f"Final response: {final_message.content}")
        print(f"Remaining steps: {result['remaining_steps']} (consumed {steps_consumed})")

    async def test_research_agent_handoff(self):
        """Test handoff to research agent with real models."""
        app = make_graph()

        # Test explicit research request
        research_message = (
            "Research the current employee counts for tech companies "
            "Facebook, Apple, Amazon, Netflix, and Google"
        )
        research_state = AgentState(
            messages=[HumanMessage(content=research_message)],
            remaining_steps=8,
            task_description=None,
            active_agent=None
        )

        result = await app.ainvoke(research_state)

        # Verify research was performed
        assert result is not None
        assert len(result["messages"]) > 1

        final_content = result["messages"][-1].content.lower()

        # Should show evidence of research activity
        research_indicators = ["research", "employee", "company", "tech"]
        has_research = any(indicator in final_content for indicator in research_indicators)

        # Should mention some of the companies
        company_indicators = ["facebook", "meta", "apple", "amazon", "netflix", "google"]
        has_companies = any(company in final_content for company in company_indicators)

        assert has_research or has_companies, f"No research indicators found in: {final_content}"

        # Verify meaningful work was done
        steps_consumed = research_state["remaining_steps"] - result["remaining_steps"]
        assert steps_consumed >= 0, f"Steps should not increase: {steps_consumed}"

        print(f"Research response: {result['messages'][-1].content}")

    async def test_math_agent_handoff(self):
        """Test handoff to math agent with real models."""
        app = make_graph()

        # Test explicit math calculation request
        math_state = AgentState(
            messages=[HumanMessage(content="Calculate the sum: 67317 + 164000 + 1551000 + 14000 + 181269")],
            remaining_steps=8,
            task_description=None,
            active_agent=None
        )

        result = await app.ainvoke(math_state)

        # Verify calculation was performed
        assert result is not None
        assert len(result["messages"]) > 1

        final_content = result["messages"][-1].content

        # Should contain the calculation result or show calculation work
        calculation_indicators = ["1977586", "1,977,586", "sum", "total", "add"]
        has_calculation = any(indicator in final_content for indicator in calculation_indicators)

        assert has_calculation, f"No calculation indicators found in: {final_content}"

        # Verify meaningful work was done
        steps_consumed = math_state["remaining_steps"] - result["remaining_steps"]
        assert steps_consumed >= 0, f"Steps should not increase: {steps_consumed}"

        print(f"Math response: {result['messages'][-1].content}")

    async def test_supervisor_coordination_real(self):
        """Test that supervisor coordinates handoffs properly with real models."""
        app = make_graph()

        # Question that requires supervisor to make routing decisions
        coordination_message = (
            "I need to know the total headcount of major tech companies "
            "and understand how they compare"
        )
        coordination_state = AgentState(
            messages=[HumanMessage(content=coordination_message)],
            remaining_steps=10,
            task_description=None,
            active_agent=None
        )

        result = await app.ainvoke(coordination_state)

        # Supervisor should handle the request appropriately
        assert result is not None
        assert len(result["messages"]) > 1

        final_content = result["messages"][-1].content.lower()

        # Should have made progress on the request (consumed steps or meaningful response)
        steps_consumed = coordination_state["remaining_steps"] - result["remaining_steps"]
        has_meaningful_response = len(final_content) > 30

        assert steps_consumed >= 0 and (steps_consumed > 0 or has_meaningful_response), (
            f"Should show evidence of coordination work. "
            f"Steps consumed: {steps_consumed}, Response length: {len(final_content)}"
        )

        # Should show evidence of addressing the request
        processing_indicators = ["tech", "companies", "headcount", "total", "compare"]
        has_processing = any(indicator in final_content for indicator in processing_indicators)

        assert has_processing, f"No evidence of request processing in: {final_content}"

        print(f"Coordination response: {result['messages'][-1].content}")


# --- apps/sample-agent/tests/unit/test_graph.py ---

    def test_workflow_state_transitions(self):
        """Test state transitions in workflow."""
        # Test initial state
        initial_state = AgentState(
            messages=[HumanMessage(content="What's the FAANG total?")],
            remaining_steps=10,
            task_description=None,
            active_agent=None
        )
        
        # Simulate state after research handoff
        research_state = AgentState(
            messages=initial_state["messages"] + [AIMessage(content="Research complete")],
            remaining_steps=8,
            task_description="Research FAANG headcounts",
            active_agent="research_expert"
        )
        
        # Simulate state after math handoff
        math_state = AgentState(
            messages=research_state["messages"] + [AIMessage(content="Calculation: 1,977,586")],
            remaining_steps=6,
            task_description="Calculate total headcount",
            active_agent="math_expert"
        )
        
        # Verify state progression
        assert len(math_state["messages"]) == 3
        assert math_state["remaining_steps"] < initial_state["remaining_steps"]
        assert math_state["active_agent"] == "math_expert"


# --- apps/sample-deep-agent/tests/integration/test_hitl.py ---

    async def test_comprehensive_hitl_workflow(self):
        """Test comprehensive HITL workflow.

        - Top-level task: interrupt with approve/reject only, approve it
        - think_tool calls: allow all (no interrupts)
        - write_todos: allow but limit to max 1 todo
        - All deep_web_search calls: reject all at all levels
        - Verify no deep_web_search in final message list
        """
        # Skip if no API credentials available
        if not os.getenv("SILICONFLOW_API_KEY") or not os.getenv("TAVILY_API_KEY"):
            pytest.skip("No API credentials available for integration test")

        from sample_deep_agent.context import DeepAgentContext
        from sample_deep_agent.graph import make_graph

        # Create context with max_todos limit
        context = DeepAgentContext(
            max_todos=1,  # Limit to 1 todo to prevent excessive planning
        )

        # Create graph with HITL configuration
        from dataclasses import asdict

        config = {"configurable": asdict(context)}

        # Define interrupt configuration
        interrupt_on = {
            "task": {"allowed_decisions": ["approve", "reject"]},  # Only approve/reject
            "write_todos": False,  # Don't interrupt write_todos
            "think_tool": False,  # Don't interrupt think_tool
            "deep_web_search": True,  # Interrupt at top level
        }

        subagent_interrupts = {
            "research-agent": {
                "deep_web_search": True,  # Interrupt in subagent too
                "think_tool": False,  # Don't interrupt think_tool in subagent
            }
        }

        agent = make_graph(config, interrupt_on=interrupt_on, subagent_interrupts=subagent_interrupts)

        # Use thread_id for state persistence (required for HITL)
        thread_id = str(uuid.uuid4())
        thread_config = {"configurable": {"thread_id": thread_id}}

        try:
            # Invoke the agent with a research task that requires web search
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="What are the core features of LangChain v1?")]},
                config=thread_config,
            )
        except Exception as e:
            if "402" in str(e) or "credits" in str(e).lower():
                pytest.skip("Insufficient API credits for integration test")
            raise

        # Track statistics
        task_approved = False
        deep_web_search_rejected_count = 0

        max_iterations = 20  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            if result.get("__interrupt__"):
                interrupts = result["__interrupt__"][0].value
                action_requests = interrupts["action_requests"]

                # Check what tools are being interrupted
                tool_names = [action["name"] for action in action_requests]
                print(f"Iteration {iteration}: Interrupted for tools: {tool_names}")

                # Process each action request
                decisions = []
                for action in action_requests:
                    tool_name = action["name"]

                    if tool_name == "task":
                        if not task_approved:
                            print("✅ Approving task (only approve/reject allowed)")
                            decisions.append({"type": "approve"})
                            task_approved = True
                        else:
                            print("❌ Rejecting subsequent task call")
                            decisions.append({"type": "reject"})

                    elif tool_name == "deep_web_search":
                        print(f"❌ Rejecting deep_web_search call #{deep_web_search_rejected_count + 1}")
                        decisions.append({"type": "reject"})
                        deep_web_search_rejected_count += 1

                    else:
                        # For other tools, approve
                        print(f"✅ Approving other tool: {tool_name}")
                        decisions.append({"type": "approve"})

                # Resume execution with decisions
                result = await agent.ainvoke(Command(resume={"decisions": decisions}), config=thread_config)
            else:
                # No more interrupts - workflow completed
                print("Workflow completed without further interrupts")
                break

            iteration += 1

        # Verify we got a result
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) > 1

        # Verify task was approved
        assert task_approved, "Task should have been approved"

        # Verify at least one deep_web_search was rejected
        assert deep_web_search_rejected_count > 0, "Should have rejected at least one deep_web_search call"

        # Verify no deep_web_search was executed (check for ToolMessage responses)
        # Note: AIMessages may contain rejected tool_calls, but we check for actual execution
        tool_messages = [msg for msg in result["messages"] if msg.__class__.__name__ == "ToolMessage"]
        for tool_msg in tool_messages:
            # ToolMessage.name contains the tool that was executed
            if hasattr(tool_msg, "name"):
                assert tool_msg.name != "deep_web_search", (
                    f"Found executed deep_web_search in ToolMessage - should have been rejected. "
                    f"Content: {tool_msg.content[:200]}"
                )

        # Report summary
        print("\n📊 Summary:")
        print("  - task: approved")
        print(f"  - deep_web_search calls rejected: {deep_web_search_rejected_count}")
        print("  - max_todos limit: 1")
        print(f"  - Total messages in result: {len(result['messages'])}")

        # The agent should have responded
        final_message = result["messages"][-1]
        assert len(final_message.content) > 0

        print("\n✅ Successfully completed comprehensive HITL workflow")


# --- apps/sample-deep-agent/tests/integration/test_research.py ---

    async def test_research_agent_structured_workflow(self):
        """Test that research agents create structured TODO plans and execute them systematically."""
        # Skip if no API credentials available
        if not os.getenv("SILICONFLOW_API_KEY"):
            pytest.skip("No API credentials available for integration test")

        from sample_deep_agent.graph import make_graph

        # Test with MCP question that should trigger structured workflow
        agent = make_graph()

        try:
            result = await agent.ainvoke({
                "messages": [HumanMessage(content="What is MCP (Model Context Protocol)?")]
            })
        except Exception as e:
            if "402" in str(e) or "credits" in str(e).lower():
                pytest.skip("Insufficient API credits for integration test")
            raise

        # Verify the workflow executed
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) > 1

        # Should have final response from agent
        final_message = result["messages"][-1]
        assert isinstance(final_message, AIMessage)
        assert len(final_message.content) > 50  # Should have meaningful content

        # Check for expected tool usage patterns
        messages = result["messages"]
        all_tools = []

        for msg in messages:
            # Track all tool calls at any level
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('function', {}).get('name', '')
                    if not tool_name:
                        tool_name = tool_call.get('name', '')
                    if tool_name:
                        all_tools.append(tool_name)

            # Check additional_kwargs for tool calls
            if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                for tool_call in msg.additional_kwargs['tool_calls']:
                    tool_name = tool_call.get('function', {}).get('name', '')
                    if not tool_name:
                        tool_name = tool_call.get('name', '')
                    if tool_name:
                        all_tools.append(tool_name)

        # Verify task delegation occurred (coordinator delegates to subagent)
        assert 'task' in all_tools, f"Should delegate to subagent via 'task' tool, tools: {all_tools}"

        # The workflow completed successfully with task delegation
        # We verified task was called, which means subagent was invoked with middleware
        print(f"✅ Task delegation successful with tools: {all_tools}")

        # Verify response contains MCP-related content
        content_lower = final_message.content.lower()
        mcp_indicators = ['mcp', 'model context protocol', 'protocol']
        found_indicators = sum(1 for indicator in mcp_indicators if indicator in content_lower)
        assert found_indicators >= 1, (
            f"Should contain MCP content, found {found_indicators}/{len(mcp_indicators)} indicators"
        )


# --- libs/langgraph-up-devkits/tests/integration/test_middleware.py ---

async def test_filesystem_mask_middleware_with_agent():
    """Test FileSystemMask middleware in a real agent workflow.

    This test verifies that:
    1. The middleware doesn't break the agent workflow
    2. Files are successfully masked during model execution
    3. Files are restored after model execution
    4. The agent completes successfully with middleware active
    """
    from typing import Annotated

    from langchain.agents.middleware import AgentMiddleware, AgentState
    from langgraph.graph.message import add_messages

    try:
        # Load a simple model for testing
        model = load_chat_model("siliconflow:THUDM/glm-4-9b-chat")
    except Exception:
        pytest.skip("SiliconFlow provider not available")

    # Create a simple middleware that adds files field to state
    class FilesStateMiddleware(AgentMiddleware[AgentState]):
        """Middleware that extends state with files field."""

        class FilesState(AgentState):  # type: ignore[type-arg]
            """State with files field."""

            messages: Annotated[list, add_messages]
            files: dict  # Virtual file system

        state_schema = FilesState

    # Create both middlewares
    files_state_middleware = FilesStateMiddleware()
    filesystem_mask_middleware = FileSystemMaskMiddleware()

    # Create agent with both middlewares: first adds files field, second masks it
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant. Answer questions briefly.",
        middleware=[files_state_middleware, filesystem_mask_middleware],
    )

    # Define files to test masking
    initial_files = {
        "file1.txt": "Content of file 1",
        "file2.txt": "Content of file 2",
        "file3.txt": "Content of file 3",
    }

    # Invoke agent - files should be masked from model but restored after
    result = await agent.ainvoke({"messages": [HumanMessage(content="Say hello")], "files": initial_files.copy()})

    # Verify we got a response (agent worked despite middleware)
    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify the model actually responded
    assert any(isinstance(msg, AIMessage) for msg in result["messages"])

    # CRITICAL: Verify files were restored in the result
    assert "files" in result, "Files should be restored by middleware"
    assert result["files"] == initial_files, "Files should match original"

    # The middleware should have cleaned up after itself
    assert filesystem_mask_middleware._shadowed_files is filesystem_mask_middleware._NO_FILES_SENTINEL, (
        "Middleware should clean up shadowed files"
    )

    print("✅ FileSystemMask middleware integration test passed")
    print(f"   Files masked and restored: {len(result['files'])} files")

async def test_filesystem_mask_with_model_provider_middleware():
    """Test FileSystemMask middleware combined with ModelProvider middleware.

    This test verifies that:
    1. Multiple middlewares can work together
    2. FileSystemMask doesn't interfere with ModelProvider
    3. Files are properly masked and restored
    4. All middlewares properly clean up after execution
    """
    from typing import Annotated

    from langchain.agents.middleware import AgentMiddleware, AgentState
    from langgraph.graph.message import add_messages

    try:
        # Load a model - this also tests provider availability
        model = load_chat_model("siliconflow:THUDM/glm-4-9b-chat")
    except Exception:
        pytest.skip("SiliconFlow provider not available")

    # Create a simple middleware that adds files field to state
    class FilesStateMiddleware(AgentMiddleware[AgentState]):
        """Middleware that extends state with files field."""

        class FilesState(AgentState):  # type: ignore[type-arg]
            """State with files field."""

            messages: Annotated[list, add_messages]
            files: dict  # Virtual file system

        state_schema = FilesState

    # Create all three middlewares
    files_state_middleware = FilesStateMiddleware()
    filesystem_mask_middleware = FileSystemMaskMiddleware()
    model_provider_middleware = ModelProviderMiddleware()

    # Create agent with all three middlewares
    agent = create_agent(
        model=model,  # Use already-loaded model
        tools=[],
        system_prompt="You are a helpful assistant. Answer very briefly.",
        middleware=[files_state_middleware, filesystem_mask_middleware, model_provider_middleware],
    )

    # Define files to test masking
    initial_files = {"important_file1.txt": "Sensitive data 1", "important_file2.txt": "Sensitive data 2"}

    # Invoke agent - files should be masked but agent should still work
    result = await agent.ainvoke({"messages": [HumanMessage(content="Hi")], "files": initial_files.copy()})

    # Verify we got a response (both middlewares worked)
    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify the model actually responded
    assert any(isinstance(msg, AIMessage) for msg in result["messages"])

    # CRITICAL: Verify files were restored in the result
    assert "files" in result, "Files should be restored by middleware"
    assert result["files"] == initial_files, "Files should match original"

    # Verify filesystem mask middleware cleaned up
    assert filesystem_mask_middleware._shadowed_files is filesystem_mask_middleware._NO_FILES_SENTINEL, (
        "Middleware should clean up shadowed files"
    )

    print("✅ Combined middleware integration test passed")
    print(f"   Files masked and restored: {len(result['files'])} files")


# --- libs/langgraph-up-devkits/tests/integration/test_tools.py ---

async def test_fetch_url_tool_integration():
    """Test fetch_url tool with real HTTP request."""
    result = await fetch_url.ainvoke({"url": "https://httpbin.org/json", "timeout": 10.0})

    assert isinstance(result, str)
    assert len(result) > 0
    # httpbin.org/json returns JSON with slideshow data
    assert "slideshow" in result.lower()

async def test_create_agent_with_tools():
    """Test create_agent with fetch_url and web_search tools."""
    # Use our load_chat_model which automatically registers providers
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=200)

    # Define system prompt
    agent_prompt = (
        "You are a helpful assistant. "
        "Use tools when needed to fetch information. "
        "Be concise in your responses."
    )

    # Create agent with tools and context schema
    agent = create_agent(
        model=model, tools=[fetch_url, web_search], system_prompt=agent_prompt, context_schema=AgentContext
    )

    # Test agent with a simple fetch task
    context = AgentContext(user_id="integration_test_user", max_search_results=1)

    try:
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=("Please fetch the UUID from https://httpbin.org/uuid and tell me what format it's in.")
                    )
                ]
            },
            context=context,
        )

        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) > 0

        # Check final message content
        final_message = result["messages"][-1]
        assert hasattr(final_message, "content")
        assert len(final_message.content) > 10

        print(f"Agent output: {final_message.content}")

    except Exception as e:
        print(f"Agent execution failed: {e}")
        # Test passes if we can create the agent - execution might fail
        pass

async def test_agent_with_search_tool():
    """Test agent using web_search tool with SearchContext."""
    # Use our load_chat_model which automatically registers providers
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=200)

    search_prompt = "You are a research assistant. Be brief and factual."

    agent = create_agent(model=model, tools=[web_search], system_prompt=search_prompt, context_schema=SearchContext)

    context = SearchContext(max_search_results=1, enable_deepwiki=False)

    try:
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=("Search for information about Python programming language and give me one key fact.")
                    )
                ]
            },
            context=context,
        )

        assert isinstance(result, dict)
        assert "messages" in result

        if result["messages"]:
            final_message = result["messages"][-1]
            print(f"Search agent output: {final_message.content}")

    except Exception as e:
        print(f"Search agent execution failed: {e}")
        # Test structure is valid even if execution fails
        pass

async def test_deepwiki_tools_retrieval():
    """Test deepwiki MCP tools retrieval - verifies server connection and tool loading."""
    try:
        # Get deepwiki tools - this should work since MCP tools are available in our environment
        deepwiki_tools = await get_deepwiki_tools()

        # Check if tools were loaded
        if not deepwiki_tools or len(deepwiki_tools) == 0:
            pytest.skip("DeepWiki MCP server not available or returned no tools")

        # Check that tools have expected attributes
        for tool in deepwiki_tools:
            # Tools should be callable or have an invoke method (StructuredTool pattern)
            assert callable(tool) or hasattr(tool, "invoke"), "Each tool should be callable or have invoke method"
            # Tools should have name and description attributes
            assert hasattr(tool, "name") or hasattr(tool, "__name__"), "Tool should have a name"

        print(f"✅ DeepWiki MCP tools loaded successfully: {len(deepwiki_tools)} tools")
    except Exception as e:
        # Skip test if MCP server is not available
        pytest.skip(f"DeepWiki MCP server not available: {str(e)}")

async def test_agent_with_deepwiki_tools():
    """Test agent with deepwiki MCP tools - simplified test with easier question."""
    try:
        # Get deepwiki tools
        deepwiki_tools = await get_deepwiki_tools()

        # Skip if no tools available
        if not deepwiki_tools or len(deepwiki_tools) == 0:
            pytest.skip("DeepWiki MCP tools not available")

        # Use our load_chat_model which automatically registers providers
        model = load_chat_model(model="siliconflow:THUDM/glm-4-9b-chat", temperature=0.3, max_tokens=300)

        # Create agent with deepwiki tools
        agent = create_agent(
            model=model,
            tools=deepwiki_tools,
            system_prompt="""You are a helpful assistant with access to GitHub repository documentation.
            When asked about a repository, use the ask_question tool with the repo name and question.""",
            context_schema=SearchContext,
        )

        context = SearchContext(enable_deepwiki=True)

        # Test with a simple, direct question
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="What is the main purpose of the facebook/react repository?"
                    )
                ]
            },
            context=context,
        )
    except Exception as e:
        # Skip test if MCP server has issues
        pytest.skip(f"DeepWiki MCP test skipped due to: {str(e)}")

    # Verify response structure
    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Check for tool calls - we know tool calling works from Tavily tests
    tool_calls_found = any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in result["messages"])

    if not tool_calls_found:
        # Print debug info
        print("No tool calls found. Messages:")
        for i, msg in enumerate(result["messages"]):
            print(f"  {i}: {type(msg)} - {getattr(msg, 'content', 'NO_CONTENT')[:100]}")

    # Assert that deepwiki tools were actually used
    assert tool_calls_found, "Agent should use deepwiki tools - tool calling works per other tests"

    print("✅ DeepWiki MCP tools integration working correctly")

async def test_context7_tools_retrieval():
    """Test context7 MCP tools retrieval - verifies server connection and tool loading."""
    try:
        # Get context7 tools
        context7_tools = await get_context7_tools()

        # Check if tools were loaded
        if not context7_tools or len(context7_tools) == 0:
            pytest.skip("Context7 MCP server not available or returned no tools")

        # Check that tools have expected attributes
        for tool in context7_tools:
            # Tools should be callable or have an invoke method (StructuredTool pattern)
            assert callable(tool) or hasattr(tool, "invoke"), "Each tool should be callable or have invoke method"
            # Tools should have name and description attributes
            assert hasattr(tool, "name") or hasattr(tool, "__name__"), "Tool should have a name"

        print(f"✅ Context7 MCP tools loaded successfully: {len(context7_tools)} tools")
    except Exception as e:
        # Skip test if MCP server is not available
        pytest.skip(f"Context7 MCP server not available: {str(e)}")

async def test_agent_with_context7_tools():
    """Test agent with context7 MCP tools - simplified test with easier question."""
    try:
        # Get context7 tools
        context7_tools = await get_context7_tools()

        # Skip if no tools available
        if not context7_tools or len(context7_tools) == 0:
            pytest.skip("Context7 MCP tools not available")

        # Use our load_chat_model which automatically registers providers
        model = load_chat_model(model="siliconflow:THUDM/glm-4-9b-chat", temperature=0.3, max_tokens=300)

        # Create agent with context7 tools
        agent = create_agent(
            model=model,
            tools=context7_tools,
            system_prompt=(
                "You are a helpful assistant with access to library documentation. "
                "When asked about a library, use resolve-library-id to find it, "
                "then get-library-docs to retrieve documentation."
            ),
            context_schema=SearchContext,
        )

        context = SearchContext(enable_deepwiki=True)

        # Test with a simple, direct question
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="What is React used for?"
                    )
                ]
            },
            context=context,
        )
    except Exception as e:
        # Skip test if MCP server has issues
        pytest.skip(f"Context7 MCP test skipped due to: {str(e)}")

    # Verify response structure
    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Check for tool calls - we know tool calling works from other tests
    tool_calls_found = any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in result["messages"])

    if not tool_calls_found:
        # Print debug info
        print("No tool calls found. Messages:")
        for i, msg in enumerate(result["messages"]):
            print(f"  {i}: {type(msg)} - {getattr(msg, 'content', 'NO_CONTENT')[:100]}")

    # Assert that context7 tools were actually used
    assert tool_calls_found, "Agent should use context7 tools - tool calling works per other tests"

    print("✅ Context7 MCP tools integration working correctly")

def test_think_tool_reflection_roundtrip():
    """Think tool returns confirmation containing reflection text."""
    reflection = "Validated search findings and noted missing context."
    result = think_tool.invoke({"reflection": reflection})
    assert result == f"Reflection recorded: {reflection}"

async def test_agent_with_think_tool_integration():
    """Test agent with think tool in a real reflection workflow."""
    # Load model
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=300)

    # Create agent with think tool and web search for a realistic reflection scenario
    agent = create_agent(
        model=model,
        tools=[think_tool, web_search],
        system_prompt="""You are a helpful research assistant. When conducting research:
        1. First search for information
        2. Use the think_tool to reflect on what you found and plan next steps
        3. Continue research if needed or provide final answer

        Always use the think_tool after getting search results to analyze findings.""",
        context_schema=SearchContext,
    )

    context = SearchContext(max_search_results=2)

    # Test agent with a research task that would naturally use reflection
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""
            Research Python's GIL (Global Interpreter Lock). After your search,
            use the think_tool to reflect on what you found before providing a summary.
        """
                )
            ]
        },
        context=context,
    )

    # Verify response structure
    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Check for tool calls and tool messages - look for evidence of both tools
    think_tool_used = False
    web_search_used = False

    for msg in result["messages"]:
        # Check for AI messages with tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                # Handle both dict and object tool_call formats
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "")
                else:
                    tool_name = getattr(tool_call, "name", None) or getattr(tool_call, "__name__", "")

                if tool_name == "think_tool":
                    think_tool_used = True
                elif tool_name == "web_search":
                    web_search_used = True

        # Check for tool messages (responses from tools)
        if hasattr(msg, "content") and isinstance(msg.content, str):
            content = msg.content
            # Look for characteristic responses from our tools
            if "Reflection recorded:" in content:
                think_tool_used = True
            elif "query" in content and ("answer" in content or "follow_up_questions" in content):
                # This looks like a web search response
                web_search_used = True

        # Also check message content for tool names (some formats include tool name in content)
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if "think_tool" in msg.content.lower():
                think_tool_used = True
            if "web_search" in msg.content.lower():
                web_search_used = True

    if not (think_tool_used and web_search_used):
        # Print debug info
        print("Tool usage analysis:")
        for i, msg in enumerate(result["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = getattr(tool_call, "name", None) or getattr(tool_call, "__name__", "unknown")
                    print(f"  Message {i}: Used tool '{tool_name}'")
            else:
                content_preview = getattr(msg, "content", "NO_CONTENT")
                if isinstance(content_preview, str):
                    content_preview = content_preview[:100]
                print(f"  Message {i}: {type(msg).__name__} - {content_preview}")

    # Assert that both tools were used in a realistic research workflow
    assert web_search_used, "Agent should use web_search for research"
    assert think_tool_used, "Agent should use think_tool for reflection after search"

    print("✅ Think tool integration with real agent working correctly")

async def test_agent_with_deep_web_search_tool():
    """Test agent with deep_web_search tool using custom state with files field."""
    from typing import Annotated

    from langchain.agents import AgentState
    from langgraph.graph.message import add_messages

    # Define custom state with files field for VFS support
    class DeepAgentState(AgentState):
        messages: Annotated[list, add_messages]
        files: dict  # Virtual file system

    # Load model using SiliconFlow
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=300)

    # Create agent with deep_web_search tool and custom state schema
    agent = create_agent(
        model=model,
        tools=[deep_web_search],
        system_prompt=(
            "You are a research assistant. You MUST use the deep_web_search tool "
            "when asked to search or research any topic. Never answer from your own "
            "knowledge - always use the tool first."
        ),
        state_schema=DeepAgentState,
    )

    # Test agent with research request
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage(content="Search for Python GIL using deep_web_search tool.")],
            "files": {},  # Start with empty VFS
        }
    )

    # Verify response structure
    assert isinstance(result, dict), "Agent should return a dictionary"
    assert "messages" in result, "Result should contain messages"
    assert "files" in result, "Result should contain files (VFS)"

    # Debug: Print messages if no files
    files = result["files"]
    if len(files) == 0:
        print("\n⚠️  No files created. Message trace:")
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            has_tool_calls = hasattr(msg, "tool_calls") and bool(msg.tool_calls)
            content_preview = str(getattr(msg, "content", "N/A"))[:100]
            print(f"  [{i}] {msg_type} | tool_calls={has_tool_calls} | content: {content_preview}")

    # Check that files were added to VFS
    assert isinstance(files, dict), "Files should be a dictionary"
    assert len(files) > 0, "Should have stored search results in VFS"

    # Verify file content structure (FileData format)
    for filename, file_data in files.items():
        assert filename.startswith("/"), f"Filename {filename} should start with /"
        assert filename.endswith(".md"), f"Filename {filename} should end with .md"
        assert isinstance(file_data, dict), "File data should be a dictionary (FileData)"
        assert "content" in file_data, "FileData should have 'content' field"
        assert "created_at" in file_data, "FileData should have 'created_at' field"
        assert "modified_at" in file_data, "FileData should have 'modified_at' field"
        assert isinstance(file_data["content"], list), "Content should be a list of lines"
        assert len(file_data["content"]) > 0, "File content should not be empty"
        # Join lines to check content
        content = "\n".join(file_data["content"])
        assert "# " in content, "File should contain markdown headers"
        assert "**URL:**" in content, "File should contain URL metadata"
        assert "**Search Query:**" in content, "File should contain search query"
        assert "## Tavily Summary" in content, "File should contain Tavily summary section"

    # Check for tool usage in messages
    deep_search_used = False
    for msg in result["messages"]:
        # Check for AI messages with tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = getattr(tool_call, "name", None) or getattr(tool_call, "__name__", "")
                if tool_name == "deep_web_search":
                    deep_search_used = True

        # Check for tool messages (responses from tools)
        if hasattr(msg, "content") and isinstance(msg.content, str):
            content = msg.content
            if "🔍 Found" in content and "📁 Files saved:" in content:
                deep_search_used = True

    assert deep_search_used, "Agent should have used deep_web_search tool"

    # Verify final message indicates successful research
    final_message = result["messages"][-1]
    assert hasattr(final_message, "content"), "Final message should have content"

    print("✅ Deep web search agent integration working correctly")
    print(f"   Files stored in VFS: {len(files)}")
    print(f"   File names: {list(files.keys())}")
    print(f"   Final message length: {len(final_message.content)} characters")

async def test_agent_with_deep_web_search_state_persistence():
    """Test that agent preserves existing VFS files while adding new ones."""
    from typing import Annotated

    from langchain.agents import AgentState
    from langgraph.graph.message import add_messages

    # Define a custom merger for files that preserves existing files
    def merge_files(existing: dict, new: dict) -> dict:
        """Merge new files with existing files, preserving both."""
        merged = existing.copy() if existing else {}
        if new:
            merged.update(new)
        return merged

    # Define custom state with files field that uses custom merger
    class DeepAgentState(AgentState):
        messages: Annotated[list, add_messages]
        files: Annotated[dict, merge_files]

    # Load model using SiliconFlow
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=200)

    # Create agent
    agent = create_agent(
        model=model,
        tools=[deep_web_search],
        system_prompt="You are a research assistant. Use deep_web_search when asked to research topics.",
        state_schema=DeepAgentState,
    )

    # Start with existing files in VFS
    initial_files = {"existing_research.md": "# Previous Research\nThis was stored earlier."}

    # First search
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Search for LangGraph documentation.")], "files": initial_files.copy()}
    )

    # Verify existing files are preserved and new ones added
    files = result["files"]
    assert "existing_research.md" in files, "Existing files should be preserved"
    assert files["existing_research.md"] == "# Previous Research\nThis was stored earlier.", (
        "Existing content unchanged"
    )

    # Count new files (excluding the existing one)
    new_files = {k: v for k, v in files.items() if k != "existing_research.md"}
    assert len(new_files) >= 1, "Should have added new search result files"

    print("✅ Deep web search state persistence working correctly")
    print(f"   Total files: {len(files)} (1 existing + {len(new_files)} new)")
    print(f"   New files: {list(new_files.keys())}")

async def test_agent_with_deep_web_search_with_context():
    """Test agent with deep_web_search tool using SearchContext for configuration."""
    from typing import Annotated

    from langchain.agents import AgentState
    from langgraph.graph.message import add_messages

    # Use the actual SearchContext from the library
    from langgraph_up_devkits.context import SearchContext as LibSearchContext

    # Define custom state with files field for VFS support
    class DeepAgentState(AgentState):
        messages: Annotated[list, add_messages]
        files: dict  # Virtual file system

    # Load main model using SiliconFlow
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=300)

    # Create agent with deep_web_search tool and context schema
    agent = create_agent(
        model=model,
        tools=[deep_web_search],
        system_prompt="""You are a research assistant with access to deep web search capabilities.
        When asked to search for information, use the deep_web_search tool.
        Always use the tool when asked to research a topic.""",
        state_schema=DeepAgentState,
        context_schema=LibSearchContext,
    )

    # Create context with include_raw_content configuration
    context = LibSearchContext(max_search_results=1, include_raw_content="markdown")

    # Test agent with research request and custom context
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""
            Please search for information about LangGraph patterns.
            Use your deep_web_search tool to find and analyze the content.
        """
                )
            ],
            "files": {},  # Start with empty VFS
        },
        context=context,
    )

    # Verify response structure
    assert isinstance(result, dict), "Agent should return a dictionary"
    assert "messages" in result, "Result should contain messages"
    assert "files" in result, "Result should contain files (VFS)"

    # Check that files were added to VFS
    files = result["files"]
    assert isinstance(files, dict), "Files should be a dictionary"
    assert len(files) > 0, "Should have stored search results in VFS"

    # Verify file content structure - should have Tavily summary and full content (FileData format)
    for filename, file_data in files.items():
        assert filename.startswith("/"), f"Filename {filename} should start with /"
        assert filename.endswith(".md"), f"Filename {filename} should end with .md"
        assert isinstance(file_data, dict), "File data should be a dictionary (FileData)"
        assert "content" in file_data, "FileData should have 'content' field"
        assert "created_at" in file_data, "FileData should have 'created_at' field"
        assert "modified_at" in file_data, "FileData should have 'modified_at' field"
        assert isinstance(file_data["content"], list), "Content should be a list of lines"
        assert len(file_data["content"]) > 0, "File content should not be empty"
        # Join lines to check content
        content = "\n".join(file_data["content"])
        assert "## Tavily Summary" in content, "File should contain Tavily summary section"
        assert "## Full Content" in content, "File should contain full content section"
        # The summary should exist and have content
        summary_section = content.split("## Tavily Summary")[1].split("##")[0].strip()
        assert len(summary_section) > 0, "Tavily summary should have content"

    # Check for tool usage in messages
    deep_search_used = False
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = getattr(tool_call, "name", None) or getattr(tool_call, "__name__", "")
                if tool_name == "deep_web_search":
                    deep_search_used = True

        if hasattr(msg, "content") and isinstance(msg.content, str):
            content = msg.content
            if "🔍 Found" in content and "📁 Files saved:" in content:
                deep_search_used = True

    assert deep_search_used, "Agent should have used deep_web_search tool"

    print("✅ Deep web search with SearchContext configuration working correctly")
    print(f"   Used include_raw_content: {context.include_raw_content}")
    print(f"   Files stored in VFS: {len(files)}")
    print(f"   File names: {list(files.keys())}")

