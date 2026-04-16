# langchain-ai/langchain-litellm
# 2 LLM-backed test functions across 12 test files
# Source: https://github.com/langchain-ai/langchain-litellm

# --- tests/unit_tests/test_litellm.py ---

def test_create_usage_metadata_reads_pydantic_prompt_details() -> None:
    """Cache token details should be extracted from Pydantic prompt_tokens_details."""
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=30,
            cache_creation_tokens=10,
        ),
    )
    meta = _create_usage_metadata(usage)
    assert meta["input_tokens"] == 100
    assert meta["input_token_details"]["cache_read"] == 30
    assert meta["input_token_details"]["cache_creation"] == 10

def test_create_usage_metadata_extracts_reasoning_tokens_pydantic() -> None:
    """Reasoning tokens should be extracted from Pydantic Usage models too."""
    from litellm.types.utils import CompletionTokensDetailsWrapper, Usage

    usage = Usage(
        prompt_tokens=10,
        completion_tokens=50,
        total_tokens=60,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=30,
        ),
    )
    meta = _create_usage_metadata(usage)
    assert meta["output_token_details"]["reasoning"] == 30

