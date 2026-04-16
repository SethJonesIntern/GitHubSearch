# TBice123123/langchain-dev-utils
# 17 LLM-backed test functions across 15 test files
# Source: https://github.com/TBice123123/langchain-dev-utils

# --- tests/test_agent.py ---

def test_prebuilt_agent():
    from langchain_core.tools import tool

    @tool
    def get_current_weather(city: str) -> str:
        """get current weather"""
        return f"in {city}, it is sunny"

    agent = create_agent(model="dashscope:qwen-flash", tools=[get_current_weather])
    response = agent.invoke(
        {"messages": [HumanMessage("What's the weather in New York?")]}
    )

    assert len(response["messages"]) == 4

    assert response["messages"][1].tool_calls[0]["name"] == "get_current_weather"

async def test_prebuilt_agent_async():
    from langchain_core.tools import tool

    @tool
    def get_current_weather(city: str) -> str:
        """get current weather"""
        return f"in {city}, it is sunny"

    agent = create_agent(model="dashscope:qwen-flash", tools=[get_current_weather])
    response = await agent.ainvoke(
        {"messages": [HumanMessage("What's the weather in New York?")]}
    )
    assert len(response["messages"]) == 4

    assert response["messages"][1].tool_calls[0]["name"] == "get_current_weather"

def test_inference_to_tool_output():
    agent = create_agent(
        model="zai:glm-4.6",
        system_prompt=(
            "You are a helpful weather assistant. Please call the get_weather tool, "
            "then use the **WeatherBaseModel** to generate the final response."
        ),
        tools=[get_weather],
        response_format=ToolStrategy(WeatherBaseModel),
    )
    response = agent.invoke(
        {"messages": [HumanMessage("What's the weather in New York? ")]}
    )

    assert isinstance(response["structured_response"], WeatherBaseModel)
    assert response["structured_response"].temperature == 75.0
    assert response["structured_response"].condition.lower() == "sunny"
    assert len(response["messages"]) == 5

    assert [m.type for m in response["messages"]] == [
        "human",
        "ai",
        "tool",
        "ai",
        "tool",
    ]


# --- tests/test_handoffs_middleware.py ---

def test_handoffs_middleware():
    agent = create_agent(
        model="dashscope:qwen3-max",
        middleware=[
            HandoffAgentMiddleware(
                agents_config=agents_config,
                custom_handoffs_tool_descriptions=custom_tool_descriptions,
                handoffs_tool_overrides=handoffs_tool_map,
            )
        ],
        checkpointer=InMemorySaver(),
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="get current time")]},
        config={"configurable": {"thread_id": "123"}},
    )

    assert response
    assert response["messages"][-1].response_metadata.get("model_name") == "glm-4.5"
    assert isinstance(response["messages"][-2], ToolMessage)
    assert "active_agent" in response and response["active_agent"] == "time_agent"

    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content="Implement a simple bubble sort in Python")
            ]
        },
        config={"configurable": {"thread_id": "123"}},
    )
    assert response
    assert (
        response["messages"][-1].response_metadata.get("model_name")
        == "qwen3-coder-plus"
    )
    assert isinstance(response["messages"][-2], ToolMessage)
    assert any(
        message
        for message in response["messages"]
        if isinstance(message, ToolMessage)
        and "Successfully transferred to coding agent" in message.content
        and message.name == "transfer_to_coding_agent"
    )
    assert "active_agent" in response and response["active_agent"] == "code_agent"

async def test_handoffs_middleware_async():
    agent = create_agent(
        model="dashscope:qwen3-max",
        middleware=[
            HandoffAgentMiddleware(
                agents_config=agents_config,
                custom_handoffs_tool_descriptions=custom_tool_descriptions,
                handoffs_tool_overrides=handoffs_tool_map,
            )
        ],
        checkpointer=InMemorySaver(),
    )

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="get current time")]},
        config={"configurable": {"thread_id": "234"}},
    )

    assert response
    assert response["messages"][-1].response_metadata.get("model_name") == "glm-4.5"
    assert isinstance(response["messages"][-2], ToolMessage)
    assert "active_agent" in response and response["active_agent"] == "time_agent"

    response = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Implement a simple bubble sort in Python")
            ]
        },
        config={"configurable": {"thread_id": "234"}},
    )
    assert response
    assert (
        response["messages"][-1].response_metadata.get("model_name")
        == "qwen3-coder-plus"
    )
    assert isinstance(response["messages"][-2], ToolMessage)
    assert any(
        message
        for message in response["messages"]
        if isinstance(message, ToolMessage)
        and "Successfully transferred to coding agent" in message.content
        and message.name == "transfer_to_coding_agent"
    )
    assert "active_agent" in response and response["active_agent"] == "code_agent"


# --- tests/test_human_in_the_loop.py ---

def test_human_in_loop(tool: BaseTool, expected: Any):
    model = load_chat_model("dashscope:qwen-flash")

    agent = create_agent(
        model=model,
        tools=[tool],
        checkpointer=InMemorySaver(),
    )

    response = agent.invoke(
        {"messages": [HumanMessage("what's the time")]},
        config={"configurable": {"thread_id": "1"}},
    )
    assert "__interrupt__" in response
    assert cast(tuple, response.get("__interrupt__"))[0].value == expected

async def test_human_in_loop_async(tool: BaseTool, expected: Any):
    model = load_chat_model("dashscope:qwen-flash")

    agent = create_agent(
        model=model,
        tools=[tool],
        checkpointer=InMemorySaver(),
    )

    response = await agent.ainvoke(
        {"messages": [HumanMessage("what's the time")]},
        config={"configurable": {"thread_id": "1"}},
    )
    assert "__interrupt__" in response
    assert cast(tuple, response.get("__interrupt__"))[0].value == expected


# --- tests/test_model_tool_emulator.py ---

def test_model_tool_emulator():
    middleware = LLMToolEmulator(model="dashscope:qwen-flash")

    @tool
    def get_current_weather(city: str) -> str:
        """get current weather"""
        return "Not implemented"

    agent = create_agent(
        model="dashscope:qwen-flash",
        tools=[get_current_weather],
        middleware=[middleware],
    )
    response = agent.invoke(
        {"messages": [HumanMessage("What's the weather in New York?")]}
    )

    message = response["messages"][-2]
    assert isinstance(message, ToolMessage)
    assert message.content != "Not implemented"

async def test_model_tool_emulator_async():
    middleware = LLMToolEmulator(model="dashscope:qwen-flash")

    @tool
    def get_current_weather(city: str) -> str:
        """get current weather"""
        return "Not implemented"

    agent = create_agent(
        model="dashscope:qwen-flash",
        tools=[get_current_weather],
        middleware=[middleware],
    )
    response = await agent.ainvoke(
        {"messages": [HumanMessage("What's the weather in New York?")]}
    )

    message = response["messages"][-2]
    assert isinstance(message, ToolMessage)
    assert message.content != "Not implemented"


# --- tests/test_router_model.py ---

def test_model_router_middleware():
    agent = create_agent(
        model="dashscope:qwen3-max",
        tools=[run_python_code],
        middleware=[
            ModelRouterMiddleware(
                router_model="dashscope:qwen-flash",
                model_list=[
                    {
                        "model_name": "dashscope:qwen3-max",
                        "model_description": "The most intelligent large model",
                    },
                    {
                        "model_name": "zai:glm-4.5",
                        "model_description": "The model with the strongest coding performance",
                        "tools": [run_python_code],
                        "model_kwargs": {
                            "extra_body": {
                                "thinking": {
                                    "type": "enabled",
                                },
                            }
                        },
                    },
                ],
            )
        ],
    )
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Implement a simple hello world program without thinking, and finally use the **run_python_code** tool to run the code"
                )
            ]
        }
    )
    assert response
    assert response["messages"][-1].response_metadata.get("model_name") == "glm-4.5"
    assert isinstance(response["messages"][-2], ToolMessage)
    assert (
        "router_model_selection" in response
        and response["router_model_selection"] == "zai:glm-4.5"
    )

async def test_model_router_middleware_async():
    agent = create_agent(
        model="dashscope:qwen3-max",
        tools=[run_python_code],
        middleware=[
            ModelRouterMiddleware(
                router_model="dashscope:qwen-flash",
                model_list=[
                    {
                        "model_name": "dashscope:qwen3-max",
                        "model_description": "The most intelligent large model",
                    },
                    {
                        "model_name": "zai:glm-4.5",
                        "model_description": "The model with the strongest coding performance",
                        "tools": [run_python_code],
                        "model_kwargs": {
                            "extra_body": {
                                "thinking": {
                                    "type": "enabled",
                                },
                            }
                        },
                    },
                ],
            )
        ],
    )
    response = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Implement a simple hello world program without thinking, and finally use the **run_python_code** tool to run the code"
                )
            ]
        }
    )
    assert response
    assert response["messages"][-1].response_metadata.get("model_name") == "glm-4.5"
    assert isinstance(response["messages"][-2], ToolMessage)
    assert (
        "router_model_selection" in response
        and response["router_model_selection"] == "zai:glm-4.5"
    )


# --- tests/test_tool_call_repair.py ---

def test_tool_call_repair_with_no_invalid_tool_call():
    agent = create_agent(
        "deepseek-chat",
        tools=[get_weather],
        middleware=[ToolCallRepairMiddleware()],
    )
    result = agent.invoke({"messages": [HumanMessage(content="New York Weather?")]})

    ai_message = result["messages"][1]
    assert (
        isinstance(ai_message, AIMessage)
        and len(ai_message.tool_calls) > 0
        and len(ai_message.invalid_tool_calls) == 0
    )

async def test_tool_call_repair_with_no_invalid_tool_call_async():
    agent = create_agent(
        "deepseek-chat",
        tools=[get_weather],
        middleware=[ToolCallRepairMiddleware()],
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="New York Weather?")]}
    )

    ai_message = result["messages"][1]
    assert (
        isinstance(ai_message, AIMessage)
        and len(ai_message.tool_calls) > 0
        and len(ai_message.invalid_tool_calls) == 0
    )


# --- tests/test_wrap_agent.py ---

def test_wrap_agent():
    agent = create_agent(
        model="dashscope:qwen-flash", tools=[get_time], name="time_agent"
    )
    call_agent_tool = wrap_agent_as_tool(
        agent, "call_time_agent", "call the agent to query the time"
    )
    assert call_agent_tool.name == "call_time_agent"
    assert call_agent_tool.description == "call the agent to query the time"

    supervisor = create_agent(model="dashscope:qwen3-max", tools=[call_agent_tool])
    response = supervisor.invoke(
        {"messages": [HumanMessage(content="What time is it now?")]}
    )

    msg = None
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and message.name == "call_time_agent":
            msg = message
            break
    assert msg is not None

async def test_wrap_agent_async(
    pre_input_hooks: Any,
    post_output_hooks: Any,
):
    agent = create_agent(
        model="dashscope:qwen-flash", tools=[get_time], name="time_agent"
    )
    call_agent_tool = wrap_agent_as_tool(
        agent, pre_input_hooks=pre_input_hooks, post_output_hooks=post_output_hooks
    )
    assert call_agent_tool.name == "transfor_to_time_agent"
    assert call_agent_tool.description

    supervisor = create_agent(model="dashscope:qwen3-max", tools=[call_agent_tool])
    response = await supervisor.ainvoke(
        {"messages": [HumanMessage(content="What time is it now?")]}
    )
    msg = None
    for message in response["messages"]:
        if (
            isinstance(message, ToolMessage)
            and message.name == "transfor_to_time_agent"
        ):
            msg = message
            break
    assert msg is not None

    assert cast(str, msg.content).startswith("<task_response>")
    assert cast(str, msg.content).endswith("</task_response>")

def test_wrap_all_agents():
    time_agent = create_agent(
        model="dashscope:qwen-flash", tools=[get_time], name="time_agent"
    )
    weather_agent = create_agent(
        model="dashscope:qwen-flash", tools=[get_weather], name="weather_agent"
    )
    call_agent_tool = wrap_all_agents_as_tool(
        [time_agent, weather_agent], "call_sub_agents"
    )
    assert call_agent_tool.name == "call_sub_agents"

    main_agent = create_agent(model="dashscope:qwen3-max", tools=[call_agent_tool])
    response = main_agent.invoke(
        {"messages": [HumanMessage(content="What time is it now?")]}
    )

    msg = None
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and message.name == "call_sub_agents":
            msg = message
            break
    assert msg is not None

async def test_wrap_all_agents_async(
    pre_input_hooks: Any,
    post_output_hooks: Any,
):
    time_agent = create_agent(
        model="dashscope:qwen-flash", tools=[get_time], name="time_agent"
    )
    weather_agent = create_agent(
        model="dashscope:qwen-flash", tools=[get_weather], name="weather_agent"
    )
    call_agent_tool = wrap_all_agents_as_tool(
        [time_agent, weather_agent],
        "call_sub_agents",
        pre_input_hooks=pre_input_hooks,
        post_output_hooks=post_output_hooks,
    )
    assert call_agent_tool.name == "call_sub_agents"

    main_agent = create_agent(model="dashscope:qwen3-max", tools=[call_agent_tool])
    response = await main_agent.ainvoke(
        {"messages": [HumanMessage(content="What time is it now?")]}
    )

    msg = None
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and message.name == "call_sub_agents":
            msg = message
            break
    assert msg is not None

    assert cast(str, msg.content).startswith("<task_response>")
    assert cast(str, msg.content).endswith("</task_response>")

