# plastic-labs/honcho
# 4 test functions with real LLM calls
# Source: https://github.com/plastic-labs/honcho


# --- tests/utils/test_clients.py ---

    def test_stream_chunk_creation(self):
        """Test creating HonchoLLMCallStreamChunk"""
        chunk = HonchoLLMCallStreamChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.is_done is False
        assert chunk.finish_reasons == []

    def test_stream_chunk_done(self):
        """Test creating final HonchoLLMCallStreamChunk"""
        chunk = HonchoLLMCallStreamChunk(
            content="", is_done=True, finish_reasons=["stop"]
        )
        assert chunk.content == ""
        assert chunk.is_done is True
        assert chunk.finish_reasons == ["stop"]

    def test_stream_chunk_default_finish_reasons(self):
        """Test that finish_reasons defaults to empty list"""
        chunk = HonchoLLMCallStreamChunk(content="test")
        assert isinstance(chunk.finish_reasons, list)
        assert chunk.finish_reasons == []

    def test_stream_chunk_with_no_finish_reasons(self):
        """Test stream chunk creation without finish reasons"""
        chunk = HonchoLLMCallStreamChunk(content="test")
        # Should use default_factory for empty list
        assert chunk.finish_reasons == []
        # Modifying the list shouldn't affect other instances
        chunk.finish_reasons.append("stop")

        new_chunk = HonchoLLMCallStreamChunk(content="test2")
        assert new_chunk.finish_reasons == []  # Should still be empty

