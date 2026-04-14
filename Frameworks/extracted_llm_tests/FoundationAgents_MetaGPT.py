# FoundationAgents/MetaGPT
# 1 test functions with real LLM calls
# Source: https://github.com/FoundationAgents/MetaGPT


# --- tests/metagpt/provider/test_general_api_base.py ---

def test_parse_stream():
    assert parse_stream_helper(None) is None
    assert parse_stream_helper(b"data: [DONE]") is None
    assert parse_stream_helper(b"data: test") == "test"
    assert parse_stream_helper(b"test") is None
    for line in parse_stream([b"data: test"]):
        assert line == "test"

