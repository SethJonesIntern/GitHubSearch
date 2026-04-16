# JudgmentLabs/judgeval
# 8 LLM-backed test functions across 43 test files
# Source: https://github.com/JudgmentLabs/judgeval

# --- src/e2etests/test_tracer.py ---

def test_openai_streaming_llm_cost():
    trace_id = openai_streaming_llm_call()
    retrieve_streaming_trace_helper(trace_id)

def test_anthropic_streaming_llm_cost():
    trace_id = anthropic_streaming_llm_call()
    retrieve_streaming_trace_helper(trace_id)

def test_together_streaming_llm_cost():
    trace_id = together_streaming_llm_call()
    retrieve_streaming_trace_helper(trace_id)

async def test_openai_async_streaming_llm_cost():
    trace_id = await openai_async_streaming_llm_call()
    retrieve_streaming_trace_helper(trace_id)

async def test_anthropic_async_streaming_llm_cost():
    trace_id = await anthropic_async_streaming_llm_call()
    retrieve_streaming_trace_helper(trace_id)

async def test_together_async_streaming_llm_cost():
    trace_id = await together_async_streaming_llm_call()
    retrieve_streaming_trace_helper(trace_id)


# --- src/tests/instrumentation/test_wrap_provider.py ---

    def test_wraps_openai_sync(self):
        from openai import OpenAI

        client = OpenAI(api_key="test", base_url="http://localhost")
        original_create = client.chat.completions.create
        wrap_provider(client)
        assert client.chat.completions.create is not original_create

    def test_wraps_openai_async(self):
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="test", base_url="http://localhost")
        original_create = client.chat.completions.create
        wrap_provider(client)
        assert client.chat.completions.create is not original_create

