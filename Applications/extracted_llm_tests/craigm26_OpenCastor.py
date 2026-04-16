# craigm26/OpenCastor
# 1 LLM-backed test functions across 438 test files
# Source: https://github.com/craigm26/OpenCastor

# --- tests/test_vla_provider.py ---

def test_think_stream_yields_strings():
    """think_stream() must yield at least one non-empty string."""
    p = _make_provider()
    chunks = list(p.think_stream(b"", "go left"))
    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, str)

