# openlit/openlit
# 9 test functions with real LLM calls
# Source: https://github.com/openlit/openlit


# --- sdk/python/tests/test_groq.py ---

def test_sync_groq_chat():
    """
    Tests synchronous Chat Completions.

    Raises:
        AssertionError: If the Chat Completions response object is not as expected.
    """

    try:
        chat_completions_resp = sync_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Monitor LLM Applications",
                }
            ],
            model="llama-3.1-8b-instant",
            max_tokens=1,
            stream=False,
        )
        assert chat_completions_resp.object == "chat.completion"

    # pylint: disable=broad-exception-caught
    except Exception as e:
        if "rate limit" in str(e).lower():
            print("Rate Limited:", e)
        else:
            raise

async def test_async_groq_chat():
    """
    Tests synchronous Chat Completions with the 'claude-3-haiku-20240307' model.

    Raises:
        AssertionError: If the Chat Completions response object is not as expected.
    """

    try:
        chat_completions_resp = await async_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "What is LLM Observability?",
                }
            ],
            model="llama-3.1-8b-instant",
            max_tokens=1,
            stream=False,
        )
        assert chat_completions_resp.object == "chat.completion"

    # pylint: disable=broad-exception-caught
    except Exception as e:
        if "rate limit" in str(e).lower():
            print("Rate Limited:", e)
        else:
            raise


# --- sdk/python/tests/test_mistral.py ---

def test_sync_mistral_chat():
    """
    Tests synchronous chat with the 'open-mistral-7b' model.

    Raises:
        AssertionError: If the chat response object is not as expected.
    """

    messages = [
        {
            "role": "user",
            "content": "sync: What is LLM Observability?",
        },
    ]

    message = client.chat.complete(
        model="open-mistral-7b",
        messages=messages,
        max_tokens=1,
    )
    assert message.object == "chat.completion"

async def test_async_mistral():
    """
    Tests asynchronous Mistral.

    Raises:
        AssertionError: If the chat response object is not as expected.
    """

    #  Tests synchronous chat with the 'open-mistral-7b' model.
    messages = [
        {
            "role": "user",
            "content": "sync: What is LLM Observability?",
        },
    ]

    message = await client.chat.complete_async(
        model="open-mistral-7b",
        messages=messages,
        max_tokens=1,
    )
    assert message.object == "chat.completion"

    # Tests asynchronous embedding creation with the 'mistral-embed' model.
    response = await client.embeddings.create_async(
        model="mistral-embed",
        inputs=["Embed this sentence.", "Monitor LLM Applications"],
    )
    assert response.object == "list"


# --- sdk/python/tests/test_together.py ---

def test_sync_together_chat():
    """
    Tests synchronous chat.

    Raises:
        AssertionError: If the response object is not as expected.
    """

    try:
        response = sync_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "user", "content": "Hi"},
            ],
            max_tokens=1,
            stream=False,
        )
        assert response.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    except Exception as e:
        if "credit_limit" in str(e).lower():
            print("Insufficient balance:", e)
        elif "429" in str(e) or "rate limit" in str(e).lower():
            print("Rate limit exceeded:", e)
        else:
            raise

def test_sync_together_image():
    """
    Tests synchronous image generate.

    Raises:
        AssertionError: If the response object is not as expected.
    """

    try:
        response = sync_client.images.generate(
            prompt="AI Observability dashboard",
            model="black-forest-labs/FLUX.1-dev",
            width=768,
            height=768,
            n=1,
        )
        assert response.model == "black-forest-labs/FLUX.1-dev"

    except Exception as e:
        if "credit_limit" in str(e).lower():
            print("Insufficient balance:", e)
        elif "429" in str(e) or "rate limit" in str(e).lower():
            print("Rate limit exceeded:", e)
        else:
            raise

async def test_async_together_chat():
    """
    Tests asynchronous chat.

    Raises:
        AssertionError: If the response object is not as expected.
    """

    try:
        response = await async_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "user", "content": "Hi"},
            ],
            max_tokens=1,
            stream=False,
        )
        assert response.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    except Exception as e:
        if "credit_limit" in str(e).lower():
            print("Insufficient balance:", e)
        elif "429" in str(e) or "rate limit" in str(e).lower():
            print("Rate limit exceeded:", e)
        else:
            raise

async def test_async_together_image():
    """
    Tests asynchronous image generate.

    Raises:
        AssertionError: If the response object is not as expected.
    """

    try:
        response = await async_client.images.generate(
            prompt="AI Observability dashboard",
            model="black-forest-labs/FLUX.1-dev",
            width=768,
            height=768,
            n=1,
        )
        assert response.model == "black-forest-labs/FLUX.1-dev"

    except Exception as e:
        if "credit_limit" in str(e).lower():
            print("Insufficient balance:", e)
        elif "429" in str(e) or "rate limit" in str(e).lower():
            print("Rate limit exceeded:", e)
        else:
            raise


# --- sdk/python/tests/test_transformers.py ---

def test_text_transformers():
    """
    Test text generation capabilities from HuggingFace Transformers library.
    """

    pipe = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
    response = pipe("LLM Observability")
    assert isinstance(response[0]["generated_text"], str)

    chat = [
        {
            "role": "system",
            "content": "You are an OpenTelemetry AI Observability expert",
        },
        {"role": "user", "content": "What is Agent Observability?"},
    ]

    response = pipe(chat, max_new_tokens=100)

    assert isinstance(response[0]["generated_text"][-1]["content"], str)

