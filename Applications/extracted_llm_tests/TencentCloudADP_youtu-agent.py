# TencentCloudADP/youtu-agent
# 3 LLM-backed test functions across 53 test files
# Source: https://github.com/TencentCloudADP/youtu-agent

# --- tests/utils/test_agents_utils.py ---

async def test_print_stream_events():
    with trace(workflow_name="test_print_stream_events"):
        stream = Runner.run_streamed(agent, "tell me a joke. And what is the weather like in Shanghai?")
        await AgentsUtils.print_stream_events(stream.stream_events())

async def test_print_items():
    result = await Runner.run(agent, "tell me a joke. And what is the weather like in Shanghai?")
    AgentsUtils.print_new_items(result.new_items)

async def test_simplified_openai_chat_completions_model():
    model = os.getenv("UTU_LLM_MODEL")
    api_key = os.getenv("UTU_LLM_API_KEY")
    base_url = os.getenv("UTU_LLM_BASE_URL")
    openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    simplified_openai_model = SimplifiedOpenAIChatCompletionsModel(model=model, openai_client=openai_client)
    with trace(workflow_name="test_agent"):
        res = await simplified_openai_model.query_one(messages=messages, tools=tools, model=model)
        print(res)

