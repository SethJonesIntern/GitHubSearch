# david-strejc/sage-mcp
# 1 LLM-backed test functions across 17 test files
# Source: https://github.com/david-strejc/sage-mcp

# --- test_deepseek.py ---

async def test_deepseek_api():
    """Test DeepSeek API connection and response"""
    print("=" * 60)
    print("DeepSeek API Test")
    print("=" * 60)

    # Create client (DeepSeek uses OpenAI-compatible API)
    client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )

    # Test each model
    for model in DEEPSEEK_MODELS:
        print(f"\n📝 Testing model: {model}")
        print("-" * 60)

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'Hello from DeepSeek!' and tell me your model name in one sentence."}
                ],
                max_tokens=100,
                temperature=0.7
            )

            content = response.choices[0].message.content
            print(f"✅ Success!")
            print(f"Response: {content}")
            print(f"Model used: {response.model}")
            print(f"Tokens - Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}")

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)

