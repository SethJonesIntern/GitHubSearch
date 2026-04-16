# mfmezger/conversational-agent-langchain
# 1 LLM-backed test functions across 13 test files
# Source: https://github.com/mfmezger/conversational-agent-langchain

# --- tests/vcr/test_litellm_requests.py ---

def test_litellm_gemini_chat_vcr() -> None:
    if _requires_live_gemini_key():
        pytest.skip("Set GEMINI_API_KEY to record live Gemini cassettes")

    if not CASSETTE_PATH.exists() and GEMINI_KEY == "dummy_key":
        pytest.skip("Gemini cassette is not recorded yet. Record once with GEMINI_API_KEY set.")

    model = os.getenv("GEMINI_TEST_MODEL", "gemini/gemini-3-flash-preview")
    llm = ChatLiteLLM(model_name=model, temperature=0)
    try:
        response = llm.invoke("Reply with exactly one short greeting.")
    except NotFoundError as exc:
        pytest.skip(f"Gemini model not available for this account/API: {exc}")

    assert isinstance(response.content, str)
    assert response.content.strip()

