# emcie-co/parlant
# 1 test functions with real LLM calls
# Source: https://github.com/emcie-co/parlant


# --- tests/adapters/nlp/test_openrouter_service.py ---

def test_that_openrouter_estimating_tokenizer_works(container: Container) -> None:
    """Test OpenRouterEstimatingTokenizer token estimation."""
    tokenizer = OpenRouterEstimatingTokenizer(model_name="openai/gpt-4o")
    tokens = asyncio.run(tokenizer.estimate_token_count("Hello world"))
    assert tokens > 0

