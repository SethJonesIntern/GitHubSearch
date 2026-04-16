# chatchat-space/Langchain-Chatchat
# 9 LLM-backed test functions across 22 test files
# Source: https://github.com/chatchat-space/Langchain-Chatchat

# --- libs/chatchat-server/tests/test_qwen_agent.py ---

async def test_server_chat():
    from chatchat.server.chat.chat import chat

    mc = {
        "preprocess_model": {
            "qwen": {
                "temperature": 0.4,
                "max_tokens": 2048,
                "history_len": 100,
                "prompt_name": "default",
                "callbacks": False,
            }
        },
        "llm_model": {
            "qwen": {
                "temperature": 0.9,
                "max_tokens": 4096,
                "history_len": 3,
                "prompt_name": "default",
                "callbacks": True,
            }
        },
        "action_model": {
            "qwen": {
                "temperature": 0.01,
                "max_tokens": 4096,
                "prompt_name": "qwen",
                "callbacks": True,
            }
        },
        "postprocess_model": {
            "qwen": {
                "temperature": 0.01,
                "max_tokens": 4096,
                "prompt_name": "default",
                "callbacks": True,
            }
        },
    }

    tc = {"weather_check": {"use": False, "api-key": "your key"}}

    async for x in (
        await chat(
            "苏州天气如何",
            {},
            model_config=mc,
            tool_config=tc,
            conversation_id=None,
            history_len=-1,
            history=[],
            stream=True,
        )
    ).body_iterator:
        pprint(x)

async def test_text2image():
    from chatchat.server.chat.chat import chat

    mc = {
        "preprocess_model": {
            "qwen-api": {
                "temperature": 0.4,
                "max_tokens": 2048,
                "history_len": 100,
                "prompt_name": "default",
                "callbacks": False,
            }
        },
        "llm_model": {
            "qwen-api": {
                "temperature": 0.9,
                "max_tokens": 4096,
                "history_len": 3,
                "prompt_name": "default",
                "callbacks": True,
            }
        },
        "action_model": {
            "qwen-api": {
                "temperature": 0.01,
                "max_tokens": 4096,
                "prompt_name": "qwen",
                "callbacks": True,
            }
        },
        "postprocess_model": {
            "qwen-api": {
                "temperature": 0.01,
                "max_tokens": 4096,
                "prompt_name": "default",
                "callbacks": True,
            }
        },
        "image_model": {"sd-turbo": {}},
    }

    tc = {"text2images": {"use": True}}

    async for x in (
        await chat(
            "draw a house",
            {},
            model_config=mc,
            tool_config=tc,
            conversation_id=None,
            history_len=-1,
            history=[],
            stream=False,
        )
    ).body_iterator:
        x = json.loads(x)
        pprint(x)


# --- libs/chatchat-server/tests/api/test_openai_wrap.py ---

def test_chat():
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": "你是谁"}],
        model=get_default_llm(),
    )
    print(resp)
    assert hasattr(resp, "choices") and len(resp.choices) > 0


# --- libs/chatchat-server/tests/integration_tests/platform_tools/test_platform_tools.py ---

async def test_openai_functions_tools(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    llm_params = get_ChatPlatformAIParams(
        model_name="glm-4-plus",
        temperature=0.01,
        max_tokens=100,
    )
    llm = ChatPlatformAI(**llm_params)
    agent_executor = PlatformToolsRunnable.create_agent_executor(
        agent_type="openai-functions",
        agents_registry=agents_registry,
        llm=llm,
        tools=[multiply, exp, add],
    )

    chat_iterator = agent_executor.invoke(chat_input="计算下 2 乘以 5")
    async for item in chat_iterator:
        if isinstance(item, PlatformToolsAction):
            print("PlatformToolsAction:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsFinish):
            print("PlatformToolsFinish:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsActionToolStart):
            print("PlatformToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsActionToolEnd):
            print("PlatformToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, PlatformToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)

async def test_platform_tools(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    llm_params = get_ChatPlatformAIParams(
        model_name="glm-4-plus",
        temperature=0.01,
        max_tokens=100,
    )
    llm = ChatPlatformAI(**llm_params)

    agent_executor = PlatformToolsRunnable.create_agent_executor(
        agent_type="platform-agent",
        agents_registry=agents_registry,
        llm=llm,
        tools=[multiply, exp, add],
    )

    chat_iterator = agent_executor.invoke(chat_input="计算下 2 乘以 5")
    async for item in chat_iterator:
        if isinstance(item, PlatformToolsAction):
            print("PlatformToolsAction:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsFinish):
            print("PlatformToolsFinish:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsActionToolStart):
            print("PlatformToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsActionToolEnd):
            print("PlatformToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, PlatformToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)

async def test_chatglm3_chat_agent_tools(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    llm_params = get_ChatPlatformAIParams(
        model_name="tmp-chatglm3-6b",
        temperature=0.01,
        max_tokens=100,
    )
    llm = ChatPlatformAI(**llm_params)
    agent_executor = PlatformToolsRunnable.create_agent_executor(
        agent_type="glm3",
        agents_registry=agents_registry,
        llm=llm,
        tools=[multiply, exp, add],
    )

    chat_iterator = agent_executor.invoke(chat_input="计算下 2 乘以 5")
    async for item in chat_iterator:
        if isinstance(item, PlatformToolsAction):
            print("PlatformToolsAction:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsFinish):
            print("PlatformToolsFinish:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsActionToolStart):
            print("PlatformToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsActionToolEnd):
            print("PlatformToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, PlatformToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)

async def test_qwen_chat_agent_tools(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    llm_params = get_ChatPlatformAIParams(
        model_name="tmp_Qwen1.5-1.8B-Chat",
        temperature=0.01,
        max_tokens=100,
    )
    llm = ChatPlatformAI(**llm_params)
    agent_executor = PlatformToolsRunnable.create_agent_executor(
        agent_type="qwen",
        agents_registry=agents_registry,
        llm=llm,
        tools=[multiply, exp, add],
    )

    chat_iterator = agent_executor.invoke(chat_input="2 add 5")
    async for item in chat_iterator:
        if isinstance(item, PlatformToolsAction):
            print("PlatformToolsAction:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsFinish):
            print("PlatformToolsFinish:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsActionToolStart):
            print("PlatformToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsActionToolEnd):
            print("PlatformToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, PlatformToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)

async def test_qwen_structured_chat_agent_tools(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    llm_params = get_ChatPlatformAIParams(
        model_name="tmp_Qwen1.5-1.8B-Chat",
        temperature=0.01,
        max_tokens=100,
    )
    llm = ChatPlatformAI(**llm_params)
    agent_executor = PlatformToolsRunnable.create_agent_executor(
        agent_type="structured-chat-agent",
        agents_registry=agents_registry,
        llm=llm,
        tools=[multiply, exp, add],
    )

    chat_iterator = agent_executor.invoke(chat_input="2 add 5")
    async for item in chat_iterator:
        if isinstance(item, PlatformToolsAction):
            print("PlatformToolsAction:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsFinish):
            print("PlatformToolsFinish:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsActionToolStart):
            print("PlatformToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsActionToolEnd):
            print("PlatformToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, PlatformToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)

async def test_human_platform_tools(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    llm_params = get_ChatPlatformAIParams(
        model_name="glm-4-plus",
        temperature=0.01,
        max_tokens=100,
    )
    llm = ChatPlatformAI(**llm_params)
    agent_executor = PlatformToolsRunnable.create_agent_executor(
        agent_type="platform-agent",
        agents_registry=agents_registry,
        llm=llm,
        tools=[multiply, exp, add],
        callbacks=[],
    )

    chat_iterator = agent_executor.invoke(chat_input="计算下 2 乘以 5")
    async for item in chat_iterator:
        if isinstance(item, PlatformToolsAction):
            print("PlatformToolsAction:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsFinish):
            print("PlatformToolsFinish:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsActionToolStart):
            print("PlatformToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, PlatformToolsActionToolEnd):
            print("PlatformToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, PlatformToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)

