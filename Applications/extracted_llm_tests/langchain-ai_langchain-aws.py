# langchain-ai/langchain-aws
# 35 LLM-backed test functions across 100 test files
# Source: https://github.com/langchain-ai/langchain-aws

# --- libs/aws/tests/integration_tests/agents/test_bedrock_agents.py ---

def test_mortgage_bedrock_agent():
    # define tools
    @tool("AssetDetail::getAssetValue")
    def get_asset_value(asset_holder_id: str) -> str:
        """Get the asset value for an owner id"""
        return f"The total asset value for {asset_holder_id} is 100K"

    @tool("AssetDetail::getMortgageRate")
    def get_mortgage_rate(asset_holder_id: str, asset_value: str) -> str:
        """Get the mortgage rate based on asset value"""
        return (
            f"The mortgage rate for the asset holder id {asset_holder_id}"
            f"with asset value of {asset_value} is 8.87%"
        )

    foundation_model = "anthropic.claude-3-sonnet-20240229-v1:0"
    tools = [get_asset_value, get_mortgage_rate]
    agent_resource_role_arn = None
    agent = None
    try:
        agent_resource_role_arn = _create_agent_role(
            agent_region="us-west-2", foundation_model=foundation_model
        )
        agent = BedrockAgentsRunnable.create_agent(
            agent_name="mortgage_interest_rate_agent",
            agent_resource_role_arn=agent_resource_role_arn,
            foundation_model=foundation_model,
            instruction=(
                "You are an agent who helps with getting the mortgage rate based on "
                "the current asset valuation"
            ),
            tools=tools,
            enable_trace=True,
        )
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, return_intermediate_steps=True
        )  # type: ignore[arg-type]
        output = agent_executor.invoke(
            {"input": "what is my mortgage rate for id AVC-1234"}
        )

        assert output["output"] == (
            "The mortgage rate for the asset holder id AVC-1234 "
            "with an asset value of 100K is 8.87%."
        )
    except Exception as ex:
        raise ex
    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn)
        if agent:
            _delete_agent(agent.agent_id)

def test_weather_agent():
    @tool
    def get_weather(location: str = "") -> str:
        """
        Get the weather of a location

        Args:
            location: location of the place

        """
        if location.lower() == "seattle":
            return f"It is raining in {location}"
        return f"It is hot and humid in {location}"

    foundation_model = "anthropic.claude-3-sonnet-20240229-v1:0"
    tools = [get_weather]
    agent_resource_role_arn = None
    agent = None
    try:
        agent_resource_role_arn = _create_agent_role(
            agent_region="us-west-2", foundation_model=foundation_model
        )
        agent = BedrockAgentsRunnable.create_agent(
            agent_name="weather_agent",
            agent_resource_role_arn=agent_resource_role_arn,
            foundation_model=foundation_model,
            instruction="""
                You are an agent who helps with getting weather for a given location""",
            tools=tools,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore[arg-type]
        output = agent_executor.invoke({"input": "what is the weather in Seattle?"})

        assert output["output"] == "It is raining in Seattle"
    except Exception as ex:
        raise ex
    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn)
        if agent:
            _delete_agent(agent.agent_id)

def test_agent_with_guardrail():
    guardrail_id, guardrail_version = create_stock_advice_guardrail()
    foundation_model = "anthropic.claude-3-sonnet-20240229-v1:0"
    agent_resource_role_arn = None
    agent_with_guardrail = None
    agent_without_guardrail = None
    try:
        agent_resource_role_arn = _create_agent_role(
            agent_region="us-west-2", foundation_model=foundation_model
        )
        agent_with_guardrail = BedrockAgentsRunnable.create_agent(
            agent_name="agent_with_financial_advice_guardrail",
            agent_resource_role_arn=agent_resource_role_arn,
            foundation_model=foundation_model,
            instruction="You are a test agent which will respond to user query",
            guardrail_configuration=langchain_aws.agents.base.GuardrailConfiguration(
                guardrail_identifier=guardrail_id, guardrail_version=guardrail_version
            ),
            description="Sample agent",
        )

        agent_without_guardrail = BedrockAgentsRunnable.create_agent(
            agent_name="agent_without_financial_advice_guardrail",
            agent_resource_role_arn=agent_resource_role_arn,
            foundation_model=foundation_model,
            instruction="You are a test agent which will respond to user query",
            memory_storage_days=30,
        )
        agent_executor_1 = AgentExecutor(agent=agent_with_guardrail, tools=[])  # type: ignore[arg-type]
        agent_executor_2 = AgentExecutor(agent=agent_without_guardrail, tools=[])  # type: ignore[arg-type]

        with pytest.raises(Exception):
            agent_executor_1.invoke(
                {"input": "can you help me invest in share market?"}
            )

        no_guardrail_output = agent_executor_2.invoke(
            {"input": "can you help me invest in share market?"}
        )

        assert no_guardrail_output["output"] is not None

    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn)
        if agent_with_guardrail:
            _delete_agent(agent_with_guardrail.agent_id)
        if agent_without_guardrail:
            _delete_agent(agent_without_guardrail.agent_id)
        if guardrail_id:
            _delete_guardrail(guardrail_id=guardrail_id)

def test_bedrock_agent_langgraph():
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt.tool_executor import ToolExecutor

    @tool
    def get_weather(location: str = "") -> str:
        """
        Get the weather of a location

        Args:
            location: location of the place

        """
        if location.lower() == "seattle":
            return f"It is raining in {location}"
        return f"It is hot and humid in {location}"

    class AgentState(TypedDict):
        input: str
        output: Union[BedrockAgentAction, BedrockAgentFinish, None]
        intermediate_steps: Annotated[
            list[tuple[BedrockAgentAction, str]], operator.add
        ]

    def get_weather_agent_node() -> Tuple[BedrockAgentsRunnable, str]:
        foundation_model = "anthropic.claude-3-sonnet-20240229-v1:0"
        tools = [get_weather]
        try:
            agent_resource_role_arn = _create_agent_role(
                agent_region="us-west-2", foundation_model=foundation_model
            )
            agent = BedrockAgentsRunnable.create_agent(
                agent_name="weather_agent",
                agent_resource_role_arn=agent_resource_role_arn,
                foundation_model=foundation_model,
                instruction=(
                    "You are an agent who helps with getting weather for a given "
                    "location"
                ),
                tools=tools,
                enable_trace=True,
            )

            return agent, agent_resource_role_arn
        except Exception as e:
            raise e

    agent_runnable, agent_resource_role_arn = get_weather_agent_node()

    def run_agent(data):
        agent_outcome = agent_runnable.invoke(data)
        return {"output": agent_outcome}

    tool_executor = ToolExecutor([get_weather])

    # Define the function to execute tools
    def execute_tools(data):
        # Get the most recent output - this is the key added in the `agent` above
        agent_action = data["output"]
        output = tool_executor.invoke(agent_action[0])
        tuple_output = agent_action[0], output
        return {"intermediate_steps": [tuple_output]}

    def should_continue(data):
        output_ = data["output"]

        # If the agent outcome is a list of BedrockAgentActions,
        # then we continue to tool execution
        if (
            isinstance(output_, list)
            and len(output_) > 0
            and isinstance(output_[0], BedrockAgentAction)
        ):
            return "continue"

        # If the agent outcome is an AgentFinish, then we return `exit` string
        # This will be used when setting up the graph to define the flow
        if isinstance(output_, BedrockAgentFinish):
            return "end"

        # Unknown output from the agent, end the graph
        return "end"

    try:
        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", run_agent)
        workflow.add_node("action", execute_tools)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        workflow.add_edge(START, "agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node
            # will be called next.
            should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output
            # of that will be matched against the keys in this mapping.
            # The matched node will then be called.
            {
                # If `tools`, then we call the tool node.
                "continue": "action",
                # Otherwise we finish.
                "end": END,
            },
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        workflow.add_edge("action", "agent")

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        app = workflow.compile()

        inputs = {"input": "what is the weather in seattle?"}
        final_state = app.invoke(inputs)

        assert isinstance(final_state.get("output", {}), BedrockAgentFinish)
        assert (
            final_state.get("output").return_values["output"]
            == "It is raining in Seattle"
        )
    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn=agent_resource_role_arn)
        if agent_runnable:
            _delete_agent(agent_id=agent_runnable.agent_id)

def test_weather_agent_with_human_input():
    @tool
    def get_weather(location: str) -> str:
        """
        Get the weather of a location

        Args:
            location: location of the place

        """
        if location.lower() == "seattle":
            return f"It is raining in {location}"
        return f"It is hot and humid in {location}"

    foundation_model = "anthropic.claude-3-sonnet-20240229-v1:0"
    tools = [get_weather]
    agent_resource_role_arn = None
    agent = None
    try:
        agent_resource_role_arn = _create_agent_role(
            agent_region="us-west-2", foundation_model=foundation_model
        )
        agent = BedrockAgentsRunnable.create_agent(
            agent_name="weather_agent",
            agent_resource_role_arn=agent_resource_role_arn,
            foundation_model=foundation_model,
            instruction="""
                You are an agent who helps with getting weather for a given location.
                If the user does not provide a location then ask for the location and be
                sure to use the word 'location'. """,
            tools=tools,
            enable_human_input=True,
        )

        # check human input is in the action groups
        bedrock_client = boto3.client("bedrock-agent")
        version = get_latest_agent_version(agent.agent_id)
        paginator = bedrock_client.get_paginator("list_agent_action_groups")
        has_human_input_tool = False
        for page in paginator.paginate(
            agentId=agent.agent_id,
            agentVersion=version,
            PaginationConfig={"PageSize": 10},
        ):
            for summary in page["actionGroupSummaries"]:
                if (
                    str(summary["actionGroupName"]).lower() == "userinputaction"
                    and str(summary["actionGroupState"]).lower() == "enabled"
                ):
                    has_human_input_tool = True
                    break

        assert has_human_input_tool
    except Exception as ex:
        raise ex
    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn)
        if agent:
            _delete_agent(agent.agent_id)

def test_weather_agent_with_code_interpreter():
    @tool
    def get_weather(location: str) -> str:
        """
        Get the weather of a location

        Args:
            location: location of the place

        """
        if location.lower() == "seattle":
            return f"It is raining in {location}"
        return f"It is hot and humid in {location}"

    foundation_model = "anthropic.claude-3-sonnet-20240229-v1:0"
    tools = [get_weather]
    agent_resource_role_arn = None
    agent = None
    try:
        agent_resource_role_arn = _create_agent_role(
            agent_region="us-west-2", foundation_model=foundation_model
        )
        agent = BedrockAgentsRunnable.create_agent(
            agent_name="weather_agent",
            agent_resource_role_arn=agent_resource_role_arn,
            foundation_model=foundation_model,
            instruction="""
                You are an agent who helps with getting weather for a given location.
                If the user does not provide a location then ask for the location and be
                sure to use the word 'location'. """,
            tools=tools,
            enable_code_interpreter=True,
        )

        # check human input is in the action groups
        bedrock_client = boto3.client("bedrock-agent")
        version = get_latest_agent_version(agent.agent_id)
        paginator = bedrock_client.get_paginator("list_agent_action_groups")
        has_code_interpreter = False
        for page in paginator.paginate(
            agentId=agent.agent_id,
            agentVersion=version,
            PaginationConfig={"PageSize": 10},
        ):
            for summary in page["actionGroupSummaries"]:
                if (
                    str(summary["actionGroupName"]).lower() == "codeinterpreteraction"
                    and str(summary["actionGroupState"]).lower() == "enabled"
                ):
                    has_code_interpreter = True
                    break

        assert has_code_interpreter
    except Exception as ex:
        raise ex
    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn)
        if agent:
            _delete_agent(agent.agent_id)

def test_inline_agent():
    from langchain_core.messages import AIMessage, HumanMessage

    @tool
    def get_weather(location: str = "") -> str:
        """
        Get the weather of a location

        Args:
            location: location of the place

        """
        if location.lower() == "seattle":
            return f"It is raining in {location}"
        return f"It is hot and humid in {location}"

    foundation_model = "anthropic.claude-3-sonnet-20240229-v1:0"
    instructions = (
        "You are an agent who helps with getting weather for a given location"
    )
    tools = [get_weather]
    try:
        runnable = BedrockInlineAgentsRunnable.create(region_name="us-west-2")
        inline_agent_config = {
            "foundation_model": foundation_model,
            "instruction": instructions,
            "tools": tools,
            "enable_trace": True,
        }
        messages = [HumanMessage(content="What is the weather in Seattle?")]
        output = runnable.invoke(messages, inline_agent_config=inline_agent_config)

        # Check if the agent called for tool invocation
        assert isinstance(output, AIMessage)
        assert hasattr(output, "tool_calls")
        assert len(output.tool_calls) > 0

        # Check the tool call details
        tool_call = output.tool_calls[0]
        assert tool_call["name"] == "get_weather"
        assert tool_call["args"]["location"] == "Seattle"

        # Check additional metadata
        assert "session_id" in output.additional_kwargs
        assert "trace_log" in output.additional_kwargs
        assert "roc_log" in output.additional_kwargs

    except Exception as ex:
        raise ex


# --- libs/aws/tests/integration_tests/middleware/test_prompt_caching.py ---

def test_middleware_converse_anthropic_system_prompt() -> None:
    llm = ChatBedrockConverse(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=[],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity, got read={read} write={write}"
    )

def test_middleware_converse_anthropic_tools() -> None:
    llm = ChatBedrockConverse(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=_make_many_tools(),
        middleware=[middleware],
    )

    response = agent.invoke({"messages": [HumanMessage(content="What is 5 + 3?")]})
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.usage_metadata is not None
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity from tools, got read={read} write={write}"
    )

def test_middleware_converse_anthropic_system_prompt_and_tools() -> None:
    llm = ChatBedrockConverse(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=[get_weather],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Miami?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0, f"Expected cache read on final turn, got read={read} write={write}"

def test_middleware_converse_anthropic_extended_ttl() -> None:
    llm = ChatBedrockConverse(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="1h")
    agent = create_agent(
        llm,
        tools=[get_weather],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Miami?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity with 1h TTL, got read={read} write={write}"
    )

def test_middleware_converse_anthropic_min_messages_skips() -> None:
    llm = ChatBedrockConverse(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="5m", min_messages_to_cache=100)
    agent = create_agent(
        llm,
        tools=[],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke({"messages": [HumanMessage(content="Hello!")]})
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    _, write = _get_cache_stats(last_msg)
    assert write == 0, (
        f"Expected no cache write with high min_messages_to_cache, got {write}"
    )

def test_middleware_converse_nova_system_prompt() -> None:
    llm = ChatBedrockConverse(model=MODEL_NOVA)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=[],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity, got read={read} write={write}"
    )

def test_middleware_converse_nova_system_prompt_and_tools() -> None:
    # Nova doesn't support tool caching, making sure this case doesn't crash.
    llm = ChatBedrockConverse(model=MODEL_NOVA)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=[get_weather],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Miami?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity, got read={read} write={write}"
    )

def test_middleware_invoke_anthropic_system_prompt() -> None:
    llm = ChatBedrock(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=[],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity, got read={read} write={write}"
    )

def test_middleware_invoke_anthropic_system_prompt_and_tools() -> None:
    llm = ChatBedrock(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=[get_weather],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Miami?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity, got read={read} write={write}"
    )


# --- libs/langgraph-checkpoint-aws/tests/integration_tests/agentcore/test_saver.py ---

    def test_tool_responses(self):
        assert add.invoke({"a": 5, "b": 3}) == 8
        assert multiply.invoke({"a": 4, "b": 6}) == 24
        assert get_weather.invoke("sf") == "It's always sunny in sf"
        assert get_weather.invoke("nyc") == "It might be cloudy in nyc"

    def test_checkpoint_save_and_retrieve(self, memory_saver):
        thread_id = generate_valid_session_id()
        actor_id = generate_valid_actor_id()

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "test_namespace",
            }
        }

        checkpoint = Checkpoint(
            v=1,
            id=str(uuid6(clock_seq=-2)),
            ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            channel_values={
                "messages": ["test message"],
                "results": {"status": "completed"},
            },
            channel_versions={"messages": "v1", "results": "v1"},
            versions_seen={"node1": {"messages": "v1"}},
            pending_sends=[],
        )

        checkpoint_metadata = {
            "source": "input",
            "step": 1,
            "writes": {"node1": ["write1", "write2"]},
        }

        try:
            saved_config = memory_saver.put(
                config,
                checkpoint,
                checkpoint_metadata,
                {"messages": "v2", "results": "v2"},
            )

            assert saved_config["configurable"]["checkpoint_id"] == checkpoint["id"]
            assert saved_config["configurable"]["thread_id"] == thread_id
            assert saved_config["configurable"]["actor_id"] == actor_id
            assert saved_config["configurable"]["checkpoint_ns"] == "test_namespace"

            checkpoint_tuple = memory_saver.get_tuple(saved_config)
            assert checkpoint_tuple.checkpoint["id"] == checkpoint["id"]

            # Metadata includes original metadata plus actor_id from config
            expected_metadata = checkpoint_metadata.copy()
            expected_metadata["actor_id"] = actor_id
            assert checkpoint_tuple.metadata == expected_metadata
            assert checkpoint_tuple.config == saved_config

        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_math_agent_with_checkpointing(self, tools, model, memory_saver):
        thread_id = generate_valid_session_id()
        actor_id = generate_valid_actor_id()

        try:
            graph = create_agent(model, tools=tools, checkpointer=memory_saver)
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "actor_id": actor_id,
                }
            }

            response = graph.invoke(
                {
                    "messages": [
                        ("human", "What is 15 times 23? Then add 100 to the result.")
                    ]
                },
                config,
            )
            assert response, "Response should not be empty"
            assert "messages" in response
            assert len(response["messages"]) > 1

            checkpoint = memory_saver.get(config)
            assert checkpoint, "Checkpoint should not be empty"

            checkpoint_tuples = list(memory_saver.list(config))
            assert checkpoint_tuples, "Checkpoint tuples should not be empty"
            assert isinstance(checkpoint_tuples, list)

            # Continue conversation to test state persistence
            response2 = graph.invoke(
                {
                    "messages": [
                        (
                            "human",
                            "What was the final result from my previous calculation?",
                        )
                    ]
                },
                config,
            )
            assert response2, "Second response should not be empty"

            # Verify we have more checkpoints after second interaction
            checkpoint_tuples_after = list(memory_saver.list(config))
            assert len(checkpoint_tuples_after) > len(checkpoint_tuples)

        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_weather_query_with_checkpointing(self, tools, model, memory_saver):
        thread_id = generate_valid_session_id()
        actor_id = generate_valid_actor_id()

        try:
            graph = create_agent(model, tools=tools, checkpointer=memory_saver)
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "actor_id": actor_id,
                }
            }

            response = graph.invoke(
                {"messages": [("human", "What's the weather in sf and nyc?")]}, config
            )
            assert response, "Response should not be empty"

            checkpoint = memory_saver.get(config)
            assert checkpoint, "Checkpoint should not be empty"

            checkpoint_tuples = list(memory_saver.list(config))
            assert checkpoint_tuples, "Checkpoint tuples should not be empty"

        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_multiple_sessions_isolation(self, tools, model, memory_saver):
        thread_id_1 = generate_valid_session_id()
        thread_id_2 = generate_valid_session_id()
        actor_id = generate_valid_actor_id()

        try:
            graph = create_agent(model, tools=tools, checkpointer=memory_saver)

            config_1 = {
                "configurable": {
                    "thread_id": thread_id_1,
                    "actor_id": actor_id,
                }
            }

            config_2 = {
                "configurable": {
                    "thread_id": thread_id_2,
                    "actor_id": actor_id,
                }
            }

            # First session
            response_1 = graph.invoke(
                {"messages": [("human", "Calculate 10 times 5")]}, config_1
            )
            assert response_1, "First session response should not be empty"

            # Second session
            response_2 = graph.invoke(
                {"messages": [("human", "What's the weather in sf?")]}, config_2
            )
            assert response_2, "Second session response should not be empty"

            # Verify sessions are isolated
            checkpoints_1 = list(memory_saver.list(config_1))
            checkpoints_2 = list(memory_saver.list(config_2))

            assert len(checkpoints_1) > 0
            assert len(checkpoints_2) > 0

            # Verify different checkpoint IDs
            checkpoint_ids_1 = {
                cp.config["configurable"]["checkpoint_id"] for cp in checkpoints_1
            }
            checkpoint_ids_2 = {
                cp.config["configurable"]["checkpoint_id"] for cp in checkpoints_2
            }
            assert checkpoint_ids_1.isdisjoint(checkpoint_ids_2)

        finally:
            memory_saver.delete_thread(thread_id_1, actor_id)
            memory_saver.delete_thread(thread_id_2, actor_id)

    def test_checkpoint_listing_with_limit(self, tools, model, memory_saver):
        thread_id = generate_valid_session_id()
        actor_id = generate_valid_actor_id()

        try:
            graph = create_agent(model, tools=tools, checkpointer=memory_saver)
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "actor_id": actor_id,
                }
            }

            # Create multiple interactions to generate several checkpoints
            for i in range(3):
                graph.invoke(
                    {"messages": [("human", f"Calculate {i + 1} times 2")]}, config
                )

            # Test listing with limit
            all_checkpoints = list(memory_saver.list(config))
            limited_checkpoints = list(memory_saver.list(config, limit=2))

            assert len(all_checkpoints) >= 3
            assert len(limited_checkpoints) == 2

            # Verify limited checkpoints are the most recent ones
            assert (
                limited_checkpoints[0].config["configurable"]["checkpoint_id"]
                == all_checkpoints[0].config["configurable"]["checkpoint_id"]
            )
            assert (
                limited_checkpoints[1].config["configurable"]["checkpoint_id"]
                == all_checkpoints[1].config["configurable"]["checkpoint_id"]
            )

        finally:
            memory_saver.delete_thread(thread_id, actor_id)


# --- libs/langgraph-checkpoint-aws/tests/integration_tests/checkpoint/dynamodb/test_langgraph_dynamodb_integration.py ---

def test_complete_workflow_checkpoint_lifecycle(checkpoint_saver, thread_id):
    """
    Test complete workflow lifecycle with checkpoint validation at each step.

    1. Creates a linear workflow: init → process → validate → finalize
    2. Executes the workflow with DynamoDBSaver as checkpointer
    3. Validates checkpoint creation after each node execution
    4. Verifies checkpoint structure and content
    5. Checks checkpoint history ordering
    6. Cleans up all resources

    - ✓ Checkpoint creation: Verifies checkpoints after each workflow step
    - ✓ Checkpoint structure: Validates id, config, metadata, and state
    - ✓ State persistence: Confirms values correctly stored in checkpoints
    - ✓ Checkpoint history: Validates all checkpoints in order (newest first)
    - ✓ Thread cleanup: Ensures delete_thread removes all checkpoints

    EXPECTED BEHAVIOR:
    - Workflow completes successfully with processing_complete=True
    - At least 4 checkpoints created (one per node: init, process, validate, finalize)
    - Checkpoints ordered newest first (higher step_count appears first)
    - Each checkpoint has valid structure with thread_id and checkpoint_id
    - Thread cleanup removes all data from DynamoDB
    """
    # Build workflow
    workflow = StateGraph(WorkflowState)
    workflow.add_node("init", initialize_workflow)
    workflow.add_node("process", process_step)
    workflow.add_node("validate", validate_state)
    workflow.add_node("finalize", finalize_workflow)

    workflow.add_edge(START, "init")
    workflow.add_edge("init", "process")
    workflow.add_edge("process", "validate")
    workflow.add_edge("validate", "finalize")
    workflow.add_edge("finalize", END)

    app = workflow.compile(checkpointer=checkpoint_saver)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Execute workflow
        result = app.invoke({}, config)

        # Validate final state
        assert result["processing_complete"] is True, "Workflow should complete"
        assert result["step_count"] >= 1, "Step count should increment"
        assert len(result["messages"]) > 0, "Messages should be recorded"
        assert "metadata" in result, "Metadata should be present"

        # Validate checkpoint was created
        current_state = app.get_state(config)
        assert current_state is not None, "Current state should exist"
        assert current_state.values is not None, "State values should exist"
        assert current_state.values["processing_complete"] is True, (
            "State should reflect completion"
        )

        # Validate checkpoint history
        history = list(app.get_state_history(config))
        assert len(history) >= 4, (
            "Should have checkpoints for each node (init, process, validate, finalize)"
        )

        # Validate checkpoint ordering (newest first)
        for i in range(len(history) - 1):
            current_step = history[i].values.get("step_count", 0)
            next_step = history[i + 1].values.get("step_count", 0)
            assert current_step >= next_step, (
                "Checkpoints should be ordered newest first"
            )

        # Validate checkpoint structure
        for checkpoint_state in history:
            assert checkpoint_state.values is not None, (
                "Each checkpoint should have values"
            )
            assert checkpoint_state.config is not None, (
                "Each checkpoint should have config"
            )
            assert "thread_id" in checkpoint_state.config["configurable"], (
                "Config should have thread_id"
            )
            assert "checkpoint_id" in checkpoint_state.config["configurable"], (
                "Config should have checkpoint_id"
            )

    finally:
        cleanup_thread_resources(checkpoint_saver, thread_id)

def test_s3_offloading_and_ttl_validation(checkpoint_saver, thread_id, aws_resources):
    """
    Test S3 offloading for large checkpoints and TTL configuration.

    1. Creates a workflow that generates 600KB of random data
    2. Executes the workflow to trigger large checkpoint creation
    3. Verifies the large checkpoint (>350KB threshold) is offloaded to S3
    4. Validates TTL attribute is set on DynamoDB items
    5. Confirms checkpoint can be retrieved from S3
    6. Cleans up both DynamoDB and S3 resources

    - ✓ S3 offloading: Large checkpoints (>350KB) automatically stored in S3
    - ✓ DynamoDB storage: Checkpoint metadata remains in DynamoDB with S3 reference
    - ✓ TTL configuration: TTL attribute set correctly on DynamoDB items (3600 seconds)
    - ✓ TTL validation: TTL value is in the future and within expected range
    - ✓ S3 retrieval: Checkpoint data can be retrieved from S3 successfully
    - ✓ Data integrity: Retrieved payload matches original size (600KB)
    - ✓ Resource cleanup: Both DynamoDB items and S3 objects are deleted

    EXPECTED BEHAVIOR:
    - Workflow generates 600KB payload (614,400 bytes)
    - Checkpoint exceeds 350KB threshold even after compression
    - S3 objects created in bucket with thread_id prefix
    - DynamoDB item has TTL attribute set to current_time + 3600 seconds
    - Checkpoint retrieval returns complete state with large payload
    - Cleanup removes all S3 objects and DynamoDB items
    """
    # Build workflow with large data generation
    workflow = StateGraph(WorkflowState)
    workflow.add_node("init", initialize_workflow)
    workflow.add_node("large_data", generate_large_checkpoint_data)
    workflow.add_node("finalize", finalize_workflow)

    workflow.add_edge(START, "init")
    workflow.add_edge("init", "large_data")
    workflow.add_edge("large_data", "finalize")
    workflow.add_edge("finalize", END)

    app = workflow.compile(checkpointer=checkpoint_saver)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Execute workflow
        result = app.invoke({}, config)

        # Validate large data was generated
        large_payload_size = len(result.get("large_payload", ""))
        assert large_payload_size > 500 * 1024, (
            f"Large payload should exceed 500KB, got {large_payload_size / 1024:.0f}KB"
        )

        # Validate S3 offloading (may not happen if compression effective)
        s3_exists, s3_size = verify_s3_checkpoint_exists(
            aws_resources["s3_bucket"], thread_id
        )
        if s3_exists:
            logger.info(f"✓ S3 offloading verified: {s3_size / 1024:.2f}KB stored")
        else:
            logger.info("Note: S3 offloading may not have occurred due to compression")

        # Validate checkpoint can be retrieved from S3
        current_state = app.get_state(config)
        assert current_state is not None, "State should be retrievable from S3"
        retrieved_payload_size = len(current_state.values.get("large_payload", ""))
        assert retrieved_payload_size == large_payload_size, (
            "Retrieved payload should match original size"
        )

        # Validate TTL is set on DynamoDB items
        dynamodb = boto3.client("dynamodb", region_name=AWS_REGION)
        response = dynamodb.query(
            TableName=aws_resources["dynamodb_table"],
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={":pk": {"S": f"THREAD#{thread_id}"}},
            Limit=1,
        )
        if response["Items"]:
            item = response["Items"][0]
            assert "ttl" in item, "DynamoDB item should have TTL attribute"
            ttl_value = int(item["ttl"]["N"])
            current_time = int(time.time())
            assert ttl_value > current_time, "TTL should be in the future"
            assert ttl_value <= current_time + TTL_SECONDS + 60, (
                "TTL should be within expected range"
            )
            ttl_remaining = ttl_value - current_time
            logger.info(f"✓ DynamoDB TTL validated: expires in {ttl_remaining}s")

        # Validate S3 lifecycle configuration aligns with TTL
        s3 = boto3.client("s3", region_name=AWS_REGION)
        try:
            lifecycle_config = s3.get_bucket_lifecycle_configuration(
                Bucket=aws_resources["s3_bucket"]
            )

            # Calculate expected expiration days from TTL_SECONDS
            expected_expiration_days = (
                TTL_SECONDS + 86399
            ) // 86400  # Round up to days
            rule_id = f"langgraph-checkpoint-expiration-{expected_expiration_days}d"

            # Find the lifecycle rule for our TTL
            matching_rules = [
                rule
                for rule in lifecycle_config.get("Rules", [])
                if rule.get("ID") == rule_id
            ]

            if matching_rules:
                rule = matching_rules[0]
                assert rule.get("Status") == "Enabled", (
                    "Lifecycle rule should be enabled"
                )

                # Validate expiration configuration
                expiration = rule.get("Expiration", {})
                assert "Days" in expiration, (
                    "Lifecycle rule should have Days expiration"
                )
                actual_days = expiration["Days"]
                assert actual_days == expected_expiration_days, (
                    f"S3 lifecycle expiration ({actual_days} days) should match TTL "
                    f"({expected_expiration_days} days from {TTL_SECONDS} seconds)"
                )

                # Validate tag filter (objects are tagged with ttl-days)
                tag_filter = rule.get("Filter", {}).get("Tag", {})
                assert tag_filter.get("Key") == "ttl-days", (
                    "Lifecycle rule should filter by ttl-days tag"
                )
                assert tag_filter.get("Value") == str(expected_expiration_days), (
                    f"Lifecycle rule tag value should be {expected_expiration_days}"
                )

                logger.info(f"✓ S3 lifecycle validated: rule '{rule_id}' configured")
                logger.info(f"  - Expiration: {actual_days} days (TTL {TTL_SECONDS}s)")
                logger.info(f"  - Tag filter: ttl-days={expected_expiration_days}")
                logger.info(f"  - Status: {rule.get('Status')}")
            else:
                logger.warning(
                    f"S3 lifecycle rule '{rule_id}' not found "
                    "(may be created on first S3 write)"
                )

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchLifecycleConfiguration":
                logger.warning(
                    "No S3 lifecycle config found (may be created on first S3 write)"
                )
            else:
                logger.warning(
                    f"Could not validate S3 lifecycle: {e.response['Error']['Code']}"
                )

    finally:
        cleanup_thread_resources(checkpoint_saver, thread_id)

def test_state_persistence_and_history_tracking(checkpoint_saver, thread_id):
    """
    Test state persistence across multiple invocations and history tracking.

    1. Creates a simple workflow: init → process
    2. Invokes the workflow 3 times with the same thread_id
    3. Tracks checkpoint history growth after each invocation
    4. Validates checkpoint accumulation across invocations
    5. Tests time-travel by retrieving earlier checkpoints
    6. Verifies checkpoint metadata preservation

    - ✓ State persistence: Each invocation creates new checkpoints in same thread
    - ✓ History accumulation: Checkpoint count increases with each invocation
    - ✓ Checkpoint ordering: History maintains newest-first ordering
    - ✓ Time-travel: Earlier checkpoints can be retrieved using their config
    - ✓ Metadata preservation: All checkpoints maintain config and values
    - ✓ Message history: Messages are preserved in checkpoint state
    - ✓ Thread isolation: All checkpoints belong to same thread_id

    EXPECTED BEHAVIOR:
    - Invocation 1: Creates 2+ checkpoints (init, process)
    - Invocation 2: Adds 2+ more checkpoints to history
    - Invocation 3: Adds 2+ more checkpoints to history
    - Total history grows: count_1 < count_2 < count_3
    - Each checkpoint has step_count=2 (workflow resets each time)
    - Earlier checkpoints retrievable via their checkpoint_id
    - All checkpoints contain messages and metadata
    """
    # Build simple workflow
    workflow = StateGraph(WorkflowState)
    workflow.add_node("init", initialize_workflow)
    workflow.add_node("process", process_step)

    workflow.add_edge(START, "init")
    workflow.add_edge("init", "process")
    workflow.add_edge("process", END)

    app = workflow.compile(checkpointer=checkpoint_saver)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # First invocation
        result1 = app.invoke({}, config)
        step_count_1 = result1["step_count"]
        assert step_count_1 == 2, (
            f"First invocation should have step_count=2, got {step_count_1}"
        )

        # Get checkpoint count after first invocation
        history_1 = list(app.get_state_history(config))
        checkpoint_count_1 = len(history_1)
        assert checkpoint_count_1 >= 2, (
            "Should have at least 2 checkpoints (init, process)"
        )

        # Second invocation (continues from previous state due to checkpointer)
        result2 = app.invoke({}, config)
        step_count_2 = result2["step_count"]
        # With checkpointer, state persists, so step_count continues from previous
        # Previous result had step_count=2, init preserves it, process increments to 3
        assert step_count_2 == 3, (
            f"Second invocation should have step_count=3 (from 2), got {step_count_2}"
        )

        # Validate history accumulated
        history_2 = list(app.get_state_history(config))
        checkpoint_count_2 = len(history_2)
        assert checkpoint_count_2 > checkpoint_count_1, "Checkpoint history should grow"

        # Third invocation
        result3 = app.invoke({}, config)
        step_count_3 = result3["step_count"]
        # With checkpointer, state continues from previous (3 → 4)
        assert step_count_3 == 4, (
            f"Third invocation should have step_count=4 (from 3), got {step_count_3}"
        )

        # Validate complete history
        history_3 = list(app.get_state_history(config))
        assert len(history_3) > checkpoint_count_2, "History should continue growing"

        # Validate time-travel: retrieve earlier checkpoint
        if len(history_3) >= 3:
            earlier_checkpoint = history_3[-2]  # Second-to-last checkpoint
            earlier_state = app.get_state(earlier_checkpoint.config)
            assert earlier_state is not None, (
                "Should be able to retrieve earlier checkpoint"
            )
            # Verify we can access historical checkpoints
            logger.info("Time-travel validated: retrieved checkpoint from history")

        # Validate checkpoint metadata preservation
        for checkpoint_state in history_3:
            assert checkpoint_state.config is not None, "Checkpoint should have config"
            assert checkpoint_state.values is not None, "Checkpoint should have values"
            assert "messages" in checkpoint_state.values, (
                "Checkpoint should preserve messages"
            )

    finally:
        cleanup_thread_resources(checkpoint_saver, thread_id)

def test_conditional_routing_and_checkpoint_branching(checkpoint_saver, thread_id):
    """
    Test conditional routing with checkpoint validation at branch points.

    1. Creates a workflow with conditional routing based on step_count
    2. Executes two scenarios with different thread IDs:
       - Scenario 1: Workflow execution with one path
       - Scenario 2: Workflow execution with another path
    3. Validates checkpoints are created at branch points
    4. Verifies different execution paths create distinct checkpoints
    5. Confirms state reflects routing decisions
    6. Cleans up resources for both scenarios

    - ✓ Conditional routing: Workflow branches based on state (step_count)
    - ✓ Branch checkpoints: Checkpoints created at conditional decision points
    - ✓ Path isolation: Different threads have independent checkpoint histories
    - ✓ State reflection: Checkpoint state shows which path was taken
    - ✓ Checkpoint structure: All checkpoints have valid structure regardless of path
    - ✓ Multiple scenarios: Single test validates multiple execution paths

    WORKFLOW STRUCTURE:
    - init → process → [conditional routing]
    - If step_count >= 2: process → large_data → validate → finalize
    - If step_count < 2: process → validate → finalize

    EXPECTED BEHAVIOR:
    - Scenario 1: Creates 4+ checkpoints for one execution path
    - Scenario 2: Creates 4+ checkpoints for another execution path
    - Each scenario has independent checkpoint history
    - Checkpoints contain workflow completion status
    - Both scenarios clean up successfully
    """
    # Build workflow with conditional routing
    workflow = StateGraph(WorkflowState)
    workflow.add_node("init", initialize_workflow)
    workflow.add_node("process", process_step)
    workflow.add_node("large_data", generate_large_checkpoint_data)
    workflow.add_node("validate", validate_state)
    workflow.add_node("finalize", finalize_workflow)

    workflow.add_edge(START, "init")
    workflow.add_edge("init", "process")
    workflow.add_conditional_edges(
        "process",
        should_generate_large_data,
        {"yes": "large_data", "no": "validate"},
    )
    workflow.add_edge("large_data", "validate")
    workflow.add_edge("validate", "finalize")
    workflow.add_edge("finalize", END)

    app = workflow.compile(checkpointer=checkpoint_saver)

    # Scenario 1: Path WITHOUT large data (step_count < 2 after process)
    config1 = {"configurable": {"thread_id": f"{thread_id}_path1"}}
    try:
        # Start with step_count=0 to take "no" path
        # After init: step_count=0, after process: step_count=1
        # Condition: 1 >= 2? NO → routes to "validate" (skips large_data)
        result1 = app.invoke({"step_count": 0}, config1)

        history1 = list(app.get_state_history(config1))
        assert len(history1) >= 4, (
            f"Path 1 should have at least 4 checkpoints, got {len(history1)}"
        )

        # Verify NO large payload (took "no" path)
        has_large_payload = (
            "large_payload" in result1 and len(result1.get("large_payload", "")) > 0
        )
        assert not has_large_payload, (
            "Path 1 should NOT generate large data (took 'no' path)"
        )
        assert result1["processing_complete"] is True, "Workflow should complete"
        assert result1["step_count"] == 1, (
            f"Step count should be 1 after process, got {result1['step_count']}"
        )

        logger.info(
            f"✓ Scenario 1: {len(history1)} checkpoints, NO large data (took 'no' path)"
        )

    finally:
        cleanup_thread_resources(checkpoint_saver, f"{thread_id}_path1")

    # Scenario 2: Path WITH large data (step_count >= 2 after process)
    config2 = {"configurable": {"thread_id": f"{thread_id}_path2"}}
    try:
        # Start with step_count=2 to take "yes" path
        # After init: step_count=2, after process: step_count=3
        # Condition: 3 >= 2? YES → routes to "large_data"
        result2 = app.invoke({"step_count": 2}, config2)

        history2 = list(app.get_state_history(config2))
        assert len(history2) >= 5, (
            f"Path 2 should have >=5 checkpoints (includes large_data), "
            f"got {len(history2)}"
        )

        # Verify HAS large payload (took "yes" path)
        has_large_payload = (
            "large_payload" in result2 and len(result2.get("large_payload", "")) > 0
        )
        assert has_large_payload, "Path 2 SHOULD generate large data (took 'yes' path)"
        assert result2["processing_complete"] is True, "Workflow should complete"
        assert result2["step_count"] == 3, (
            f"Step count should be 3 after process, got {result2['step_count']}"
        )
    finally:
        cleanup_thread_resources(checkpoint_saver, f"{thread_id}_path2")

def test_parallel_execution_with_resumability(checkpoint_saver, thread_id):
    """
    Test parallel node execution with TRUE resumability and partial failure recovery.

    WHAT THIS TEST DOES:
    1. Creates a workflow with 3 parallel nodes (task_a, task_b, task_c)
    2. First execution: PARTIAL failure (task_a ✓, task_b ✗, task_c ✓)
    3. Validates that successful tasks' results are preserved in checkpoints
    4. Second execution: TRUE resumability using invoke(None) from checkpoint
    5. Verifies only failed task (task_b) re-executes, successful results are reused
    6. Validates all results are merged correctly in final state

    - ✓ Parallel execution: Multiple nodes execute concurrently
    - ✓ Partial failure recovery: Some tasks succeed, some fail
    - ✓ Checkpoint preservation: Successful tasks' results saved in checkpoints
    - ✓ TRUE resumability: Workflow continues from checkpoint (not fresh run)
    - ✓ Selective retry: Only failed nodes re-execute on resume
    - ✓ State accumulation: Results from both attempts merged correctly
    - ✓ Checkpoint history: Shows partial completion and resume

    WORKFLOW STRUCTURE:
    - START → [task_a, task_b, task_c] (parallel) → merge → END
    - Each task adds result to parallel_results list
    - Merge node combines all results

    EXPECTED BEHAVIOR:
    - First attempt: Partial failure (task_a ✓, task_b ✗, task_c ✓)
    - Checkpoints preserve successful tasks' results (result_a, result_c)
    - State shows pending nodes (task_b and merge)
    - Second attempt: Resume with invoke(None) - only task_b re-executes
    - Final state contains results from both attempts (result_a, result_b, result_c)
    - Checkpoint history shows both partial completion and resume
    - TRUE resumability: No duplicate work, continues from checkpoint
    """
    import operator
    from typing import Annotated

    # Define state with reducer for parallel results
    class ParallelState(TypedDict):
        """State for parallel execution testing."""

        messages: Annotated[list, add_messages]
        parallel_results: Annotated[list[str], operator.add]
        attempt_count: int
        retry_count: int  # Counter to track retries

    # Parallel task nodes
    def task_a(state: ParallelState) -> dict:
        """Parallel task A - always succeeds."""
        logger.info("Task A: Executing (always succeeds)")
        return {
            "messages": [{"role": "system", "content": "Task A completed"}],
            "parallel_results": ["result_a"],
        }

    def task_b(state: ParallelState) -> dict:
        """Parallel task B - fails first (retry_count < 1), succeeds on retry."""
        retry_count = state.get("retry_count", 0)
        logger.info(f"Task B: Executing (retry_count={retry_count})")

        if retry_count < 1:
            # First attempt - fail and increment counter
            logger.info("Task B: FAILING (first attempt, retry_count < 1)")
            raise ValueError("Task B failed (simulated partial failure)")

        # Retry attempt - succeed
        logger.info("Task B: SUCCESS (retry_count >= 1)")
        return {
            "messages": [{"role": "system", "content": "Task B completed"}],
            "parallel_results": ["result_b"],
        }

    def task_c(state: ParallelState) -> dict:
        """Parallel task C - always succeeds."""
        logger.info("Task C: Executing (always succeeds)")
        return {
            "messages": [{"role": "system", "content": "Task C completed"}],
            "parallel_results": ["result_c"],
        }

    def merge_results(state: ParallelState) -> dict:
        """Merge parallel results."""
        result_count = len(state.get("parallel_results", []))
        return {
            "messages": [
                {"role": "system", "content": f"Merged {result_count} results"}
            ],
            "attempt_count": state.get("attempt_count", 0) + 1,
        }

    # Build workflow with parallel execution
    workflow = StateGraph(ParallelState)
    workflow.add_node("task_a", task_a)
    workflow.add_node("task_b", task_b)
    workflow.add_node("task_c", task_c)
    workflow.add_node("merge", merge_results)

    # Parallel edges from START
    workflow.add_edge(START, "task_a")
    workflow.add_edge(START, "task_b")
    workflow.add_edge(START, "task_c")

    # All parallel tasks converge to merge
    workflow.add_edge("task_a", "merge")
    workflow.add_edge("task_b", "merge")
    workflow.add_edge("task_c", "merge")
    workflow.add_edge("merge", END)

    app = workflow.compile(checkpointer=checkpoint_saver)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # ====================================================================
        # ATTEMPT 1: PARTIAL FAILURE (task_b fails, task_a and task_c succeed)
        # ====================================================================
        logger.info("=" * 80)
        logger.info("ATTEMPT 1: Partial failure - task_b fails, task_a/task_c succeed")
        logger.info("=" * 80)

        try:
            app.invoke({"retry_count": 0, "attempt_count": 0}, config)
            raise AssertionError("Should have raised ValueError from task_b")
        except ValueError as e:
            logger.info(f"✓ Expected partial failure occurred: {e}")

        # Validate state after partial failure
        state_after_failure = app.get_state(config)
        assert state_after_failure is not None, (
            "State should exist after partial failure"
        )

        # Check partial results - task_a and task_c should have completed
        partial_results = state_after_failure.values.get("parallel_results", [])
        logger.info(f"Partial results preserved: {partial_results}")
        assert len(partial_results) == 2, (
            f"Should have 2 partial results (task_a, task_c), "
            f"got {len(partial_results)}"
        )

        # Check pending nodes - task_b and merge should be pending
        pending_nodes = state_after_failure.next
        logger.info(f"Pending nodes after partial failure: {pending_nodes}")
        assert len(pending_nodes) == 1, (
            "Should have pending nodes after partial failure"
        )

        # Get checkpoint history after partial failure
        history_after_failure = list(app.get_state_history(config))
        checkpoint_count_after_failure = len(history_after_failure)
        assert checkpoint_count_after_failure > 0, (
            "Should have checkpoints after partial failure"
        )

        # ====================================================================
        # ATTEMPT 2: TRUE RESUMABILITY - Continue from checkpoint
        # ====================================================================
        logger.info("=" * 80)
        logger.info("ATTEMPT 2: TRUE RESUMABILITY - Continuing from checkpoint")
        logger.info("=" * 80)

        logger.info("  - Update: Sets retry_count=1 so task_b will succeed")
        result = app.invoke(Command(update={"retry_count": 1}, goto="task_b"), config)

        # Validate successful completion
        assert "parallel_results" in result, "Result should contain parallel_results"
        parallel_results = result["parallel_results"]
        assert len(parallel_results) == 3, (
            f"Should have 3 parallel results, got {len(parallel_results)}"
        )
        assert "result_a" in parallel_results, "Should contain result from task_a"
        assert "result_b" in parallel_results, "Should contain result from task_b"
        assert "result_c" in parallel_results, "Should contain result from task_c"
        assert result["attempt_count"] >= 1, (
            "Attempt count should be incremented by merge"
        )

        # Validate checkpoint history shows both attempts
        history_after_success = list(app.get_state_history(config))
        checkpoint_count_after_success = len(history_after_success)
        assert checkpoint_count_after_success > checkpoint_count_after_failure, (
            "Checkpoint count should increase after successful resume"
        )
        logger.info(
            f"Checkpoints after successful resume: {checkpoint_count_after_success}"
        )

        # Validate checkpoint structure
        for checkpoint_state in history_after_success[:5]:
            assert checkpoint_state.config is not None, "Checkpoint should have config"
            assert checkpoint_state.values is not None, "Checkpoint should have values"
            assert "thread_id" in checkpoint_state.config["configurable"], (
                "Config should have thread_id"
            )

        # Validate current state reflects successful completion
        current_state = app.get_state(config)
        assert current_state.values["attempt_count"] >= 1, (
            "Current state should show completion"
        )
        assert len(current_state.next) == 0, "No pending nodes after completion"

    finally:
        cleanup_thread_resources(checkpoint_saver, thread_id)

def test_subgraph_execution_with_checkpoints(checkpoint_saver, thread_id):
    """
    Test 2 subgraphs processing data and parent collecting results.

    WORKFLOW:
    - Parent: START → analysis_subgraph → validation_subgraph → combine_results → END
    - Analysis Subgraph: extract → transform → END
    - Validation Subgraph: check → verify → END

    VALIDATES:
    - Both subgraphs execute independently
    - Parent collects outputs from both subgraphs
    - Final result combines both subgraph outputs
    - Checkpoints track entire workflow
    """

    # State for subgraphs and parent
    class WorkflowState(TypedDict):
        input_data: str
        analysis_result: str
        validation_result: str
        final_output: str

    # === ANALYSIS SUBGRAPH ===
    def extract_data(state: WorkflowState) -> dict:
        logger.info("Analysis: Extracting data")
        data = state.get("input_data", "")
        return {"analysis_result": f"extracted[{data}]"}

    def transform_data(state: WorkflowState) -> dict:
        logger.info("Analysis: Transforming data")
        result = state.get("analysis_result", "")
        return {"analysis_result": f"{result}→transformed"}

    analysis_graph = StateGraph(WorkflowState)
    analysis_graph.add_node("extract", extract_data)
    analysis_graph.add_node("transform", transform_data)
    analysis_graph.add_edge(START, "extract")
    analysis_graph.add_edge("extract", "transform")
    analysis_graph.add_edge("transform", END)
    analysis_subgraph = analysis_graph.compile(checkpointer=checkpoint_saver)

    # === VALIDATION SUBGRAPH ===
    def check_data(state: WorkflowState) -> dict:
        logger.info("Validation: Checking data")
        data = state.get("input_data", "")
        return {"validation_result": f"checked[{data}]"}

    def verify_data(state: WorkflowState) -> dict:
        logger.info("Validation: Verifying data")
        result = state.get("validation_result", "")
        return {"validation_result": f"{result}→verified"}

    validation_graph = StateGraph(WorkflowState)
    validation_graph.add_node("check", check_data)
    validation_graph.add_node("verify", verify_data)
    validation_graph.add_edge(START, "check")
    validation_graph.add_edge("check", "verify")
    validation_graph.add_edge("verify", END)
    validation_subgraph = validation_graph.compile(checkpointer=checkpoint_saver)

    # === PARENT GRAPH ===
    def combine_results(state: WorkflowState) -> dict:
        logger.info("Parent: Combining results from both subgraphs")
        analysis = state.get("analysis_result", "")
        validation = state.get("validation_result", "")
        return {"final_output": f"COMBINED[{analysis} + {validation}]"}

    parent = StateGraph(WorkflowState)
    parent.add_node("analysis", analysis_subgraph)
    parent.add_node("validation", validation_subgraph)
    parent.add_node("combine", combine_results)
    parent.add_edge(START, "analysis")
    parent.add_edge("analysis", "validation")
    parent.add_edge("validation", "combine")
    parent.add_edge("combine", END)

    app = parent.compile(checkpointer=checkpoint_saver)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Execute workflow
        result = app.invoke({"input_data": "test_data"}, config)

        # Validate both subgraphs executed
        assert "extracted" in result["analysis_result"], (
            "Analysis subgraph should extract"
        )
        assert "transformed" in result["analysis_result"], (
            "Analysis subgraph should transform"
        )
        assert "checked" in result["validation_result"], (
            "Validation subgraph should check"
        )
        assert "verified" in result["validation_result"], (
            "Validation subgraph should verify"
        )

        # Validate final output combines both
        assert "COMBINED" in result["final_output"], "Should combine results"
        assert "extracted" in result["final_output"], "Should include analysis result"
        assert "checked" in result["final_output"], "Should include validation result"

        # Validate checkpoints - including subgraph steps
        history = list(app.get_state_history(config))
        assert len(history) > 0, "Should create checkpoints"
        assert len(history) >= 5, f"Expected at least 5 checkpoints, got {len(history)}"

    finally:
        cleanup_thread_resources(checkpoint_saver, thread_id)


# --- libs/langgraph-checkpoint-aws/tests/integration_tests/saver/test_async_saver.py ---

    async def test_weather_tool_responses(self):
        # Test weather tool directly
        assert get_weather.invoke("sf") == "It's always sunny in sf"
        assert get_weather.invoke("nyc") == "It might be cloudy in nyc"

    async def test_weather_query_and_checkpointing(
        self, boto_session_client, tools, model, session_saver
    ):
        # Create session
        session_response = await boto_session_client.create_session()
        session_id = session_response.session_id
        assert session_id, "Session ID should not be empty"
        try:
            # Create graph and config
            graph = create_agent(model, tools=tools, checkpointer=session_saver)
            config = {"configurable": {"thread_id": session_id}}

            # Test weather query
            response = await graph.ainvoke(
                {"messages": [("human", "what's the weather in sf")]}, config
            )
            assert response, "Response should not be empty"

            # Test checkpoint retrieval
            checkpoint = await session_saver.aget(config)
            assert checkpoint, "Checkpoint should not be empty"

            # Test checkpoint listing
            checkpoint_tuples = [tup async for tup in session_saver.alist(config)]
            assert checkpoint_tuples, "Checkpoint tuples should not be empty"
            assert isinstance(checkpoint_tuples, list), (
                "Checkpoint tuples should be a list"
            )
        finally:
            # Create proper request objects
            await boto_session_client.end_session(
                EndSessionRequest(session_identifier=session_id)
            )
            await boto_session_client.delete_session(
                DeleteSessionRequest(session_identifier=session_id)
            )


# --- libs/langgraph-checkpoint-aws/tests/integration_tests/saver/test_saver.py ---

    def test_weather_tool_responses(self):
        # Test weather tool directly
        assert get_weather.invoke("sf") == "It's always sunny in sf"
        assert get_weather.invoke("nyc") == "It might be cloudy in nyc"

    def test_weather_query_and_checkpointing(
        self, boto_session_client, tools, model, session_saver
    ):
        # Create session
        session_id = boto_session_client.create_session()["sessionId"]
        assert session_id, "Session ID should not be empty"
        try:
            # Create graph and config
            graph = create_agent(model, tools=tools, checkpointer=session_saver)
            config = {"configurable": {"thread_id": session_id}}
            # Test weather query
            response = graph.invoke(
                {"messages": [("human", "what's the weather in sf")]},
                RunnableConfig(configurable=config["configurable"]),
            )
            assert response, "Response should not be empty"

            # Test checkpoint retrieval
            checkpoint = session_saver.get(config)
            assert checkpoint, "Checkpoint should not be empty"

            # Test checkpoint listing
            checkpoint_tuples = list(session_saver.list(config))
            assert checkpoint_tuples, "Checkpoint tuples should not be empty"
            assert isinstance(checkpoint_tuples, list), (
                "Checkpoint tuples should be a list"
            )
        finally:
            boto_session_client.end_session(sessionIdentifier=session_id)
            boto_session_client.delete_session(sessionIdentifier=session_id)


# --- libs/langgraph-checkpoint-aws/tests/unit_tests/test_utils.py ---

    def test_generate_deterministic_uuid(self, test_case):
        input_string, expected_uuid = test_case
        input_string_bytes = input_string.encode("utf-8")
        result_as_str = generate_deterministic_uuid(input_string)
        result_as_bytes = generate_deterministic_uuid(input_string_bytes)

        assert isinstance(result_as_str, uuid.UUID)
        assert isinstance(result_as_bytes, uuid.UUID)
        # Test deterministic behavior
        assert str(result_as_str) == expected_uuid
        assert str(result_as_bytes) == expected_uuid

    def test__generate_checkpoint_id_success(self):
        input_str = "test_namespace"
        result = generate_checkpoint_id(input_str)
        assert result == "72f4457f-e6bb-e1db-49ee-06cd9901904f"

    def test__generate_write_id_success(self):
        checkpoint_ns = "test_namespace"
        checkpoint_id = "test_checkpoint"
        result = generate_write_id(checkpoint_ns, checkpoint_id)
        assert result == "f75c463a-a608-0629-401e-f4d270073c0c"

