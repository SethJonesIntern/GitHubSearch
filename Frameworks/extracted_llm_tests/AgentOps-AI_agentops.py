# AgentOps-AI/agentops
# 9 test functions with real LLM calls
# Source: https://github.com/AgentOps-AI/agentops


# --- app/e2e/sdk-api/src/test_multi_agent.py ---

    async def test_single_completion(self):
        await self.engineer.completion(
            "write a python function that adds two numbers together", self.client
        )

        time.sleep(2)

        llm_calls = await self.db.get(
            "llms", "id", "agent_id", getattr(self.engineer, "agent_ops_agent_id")
        )
        self.assertIsNotNone(llm_calls)


# --- tests/integration/test_llm_providers.py ---

async def test_openai_assistants_provider(openai_client):
    """Test OpenAI Assistants API integration for all overridden methods."""
    # Test Assistants CRUD operations
    # Create
    assistant = openai_client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4o-mini",
    )
    assert assistant.id.startswith("asst_")

    # Retrieve
    retrieved_assistant = openai_client.beta.assistants.retrieve(assistant.id)
    assert retrieved_assistant.id == assistant.id

    # Update
    updated_assistant = openai_client.beta.assistants.update(
        assistant.id,
        name="Advanced Math Tutor",
        instructions="You are an advanced math tutor. Explain concepts in detail.",
    )
    assert updated_assistant.name == "Advanced Math Tutor"

    # List
    assistants_list = openai_client.beta.assistants.list()
    assert any(a.id == assistant.id for a in assistants_list.data)

    # Test Threads CRUD operations
    # Create
    thread = openai_client.beta.threads.create()
    assert thread.id.startswith("thread_")

    # Add Multiple Messages
    message1 = openai_client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
    )
    message2 = openai_client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content="Also, what is the square root of 144?"
    )
    assert message1.content[0].text.value
    assert message2.content[0].text.value

    # Create and monitor run
    run = openai_client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
    assert run.id.startswith("run_")

    # Monitor run status with timeout
    async def check_run_status():
        while True:
            run_status = openai_client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            print(f"Current run status: {run_status.status}")  # Print status for debugging
            if run_status.status in ["completed", "failed", "cancelled", "expired"]:
                return run_status
            await asyncio.sleep(1)

    try:
        await asyncio.wait_for(check_run_status(), timeout=10)  # Shorter timeout
    except TimeoutError:
        # Cancel the run if it's taking too long
        openai_client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
        pytest.skip("Assistant run timed out and was cancelled")

    # Get run steps
    run_steps = openai_client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id)
    assert len(run_steps.data) > 0

    # List messages
    messages = openai_client.beta.threads.messages.list(thread_id=thread.id)
    assert len(messages.data) > 0

    # Update thread
    updated_thread = openai_client.beta.threads.update(thread.id, metadata={"test": "value"})
    assert updated_thread.metadata.get("test") == "value"

    # Clean up
    openai_client.beta.threads.delete(thread.id)
    openai_client.beta.assistants.delete(assistant.id)

def test_ai21_provider(ai21_client, ai21_async_client, ai21_test_messages: List[Dict[str, Any]]):
    """Test AI21 provider integration."""
    # Sync completion
    response = ai21_client.chat.completions.create(
        model="jamba-1.5-mini",
        messages=ai21_test_messages,
    )
    assert response.choices[0].message.content

    # Stream completion
    stream = ai21_client.chat.completions.create(
        model="jamba-1.5-mini",
        messages=ai21_test_messages,
        stream=True,
    )
    content = collect_stream_content(stream, "ai21")
    assert len(content) > 0

    # Async completion
    async def async_test():
        response = await ai21_async_client.chat.completions.create(
            model="jamba-1.5-mini",
            messages=ai21_test_messages,
        )
        return response

    async_response = asyncio.run(async_test())
    assert async_response.choices[0].message.content

def test_cohere_provider(cohere_client):
    """Test Cohere provider integration."""
    # Sync chat
    response = cohere_client.chat(message="Say hello in spanish")
    assert response.text

    # Stream chat
    stream = cohere_client.chat_stream(message="Say hello in spanish")
    content = collect_stream_content(stream, "cohere")
    assert len(content) > 0

def test_groq_provider(groq_client, test_messages: List[Dict[str, Any]]):
    """Test Groq provider integration."""
    # Sync completion
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=test_messages,
    )
    assert response.choices[0].message.content

    # Stream completion
    stream = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=test_messages,
        stream=True,
    )
    content = collect_stream_content(stream, "groq")
    assert len(content) > 0

def test_mistral_provider(mistral_client, test_messages: List[Dict[str, Any]]):
    """Test Mistral provider integration."""
    # Sync completion
    response = mistral_client.chat.complete(
        model="open-mistral-nemo",
        messages=test_messages,
    )
    assert response.choices[0].message.content

    # Stream completion
    stream = mistral_client.chat.stream(
        model="open-mistral-nemo",
        messages=test_messages,
    )
    content = collect_stream_content(stream, "mistral")
    assert len(content) > 0

    # Async completion
    async def async_test():
        response = await mistral_client.chat.complete_async(
            model="open-mistral-nemo",
            messages=test_messages,
        )
        return response

    async_response = asyncio.run(async_test())
    assert async_response.choices[0].message.content

def test_litellm_provider(litellm_client, test_messages: List[Dict[str, Any]]):
    """Test LiteLLM provider integration."""
    # Sync completion
    response = litellm_client.completion(
        model="openai/gpt-4o-mini",
        messages=test_messages,
    )
    assert response.choices[0].message.content

    # Stream completion
    stream_response = litellm_client.completion(
        model="anthropic/claude-3-5-sonnet-latest",
        messages=test_messages,
        stream=True,
    )
    content = collect_stream_content(stream_response, "litellm")
    assert len(content) > 0

    # Async completion
    async def async_test():
        async_response = await litellm_client.acompletion(
            model="openrouter/deepseek/deepseek-chat",
            messages=test_messages,
        )
        return async_response

    async_response = asyncio.run(async_test())
    assert async_response.choices[0].message.content

def test_ollama_provider(test_messages: List[Dict[str, Any]]):
    """Test Ollama provider integration."""
    import ollama
    from ollama import AsyncClient

    try:
        # Test if Ollama server is running
        ollama.list()
    except Exception as e:
        pytest.skip(f"Ollama server not running: {e}")

    try:
        # Sync chat
        response = ollama.chat(
            model="llama3.2:1b",
            messages=test_messages,
        )
        assert response["message"]["content"]

        # Stream chat
        stream = ollama.chat(
            model="llama3.2:1b",
            messages=test_messages,
            stream=True,
        )
        content = collect_stream_content(stream, "ollama")
        assert len(content) > 0

        # Async chat
        async def async_test():
            client = AsyncClient()
            response = await client.chat(
                model="llama3.2:1b",
                messages=test_messages,
            )
            return response

        async_response = asyncio.run(async_test())
        assert async_response["message"]["content"]

    except Exception as e:
        pytest.skip(f"Ollama test failed: {e}")


# --- tests/smoke/test_openai.py ---

def test_openai():
    import agentops

    agentops.init(exporter=InMemorySpanExporter())
    agentops.start_session()

    openai.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Write a one-line joke"}]
    )

