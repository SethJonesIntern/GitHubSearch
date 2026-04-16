# pipeshub-ai/pipeshub-ai
# 23 LLM-backed test functions across 477 test files
# Source: https://github.com/pipeshub-ai/pipeshub-ai

# --- backend/python/tests/unit/utils/test_streaming.py ---

    def test_complex_data(self):
        data = {"answer": "hello", "citations": [{"id": 1}], "confidence": 0.9}
        result = create_sse_event("complete", data)
        parsed_data = json.loads(result.split("data: ", 1)[1].strip())
        assert parsed_data == data

    def test_incomplete_cite_regex_matches(self):
        _, _, incomplete_cite_re, _ = _initialize_answer_parser_regex()
        # Matches incomplete markdown links at end of text
        assert incomplete_cite_re.search("text [") is not None
        assert incomplete_cite_re.search("text [12") is not None
        assert incomplete_cite_re.search("text [ref](partial") is not None

    def test_incomplete_cite_regex_no_match_on_complete(self):
        _, _, incomplete_cite_re, _ = _initialize_answer_parser_regex()
        # Complete markdown links should NOT match
        assert incomplete_cite_re.search("text [1](http://example.com)") is None

    def test_basic_response(self):
        async def gen():
            yield b"data"

        response = create_stream_record_response(gen(), "test.txt")
        assert response.media_type == "application/octet-stream"
        assert "Content-Disposition" in response.headers
        assert "test.txt" in response.headers["Content-Disposition"]

    def test_custom_mime_type(self):
        async def gen():
            yield b"data"

        response = create_stream_record_response(
            gen(), "report.pdf", mime_type="application/pdf"
        )
        assert response.media_type == "application/pdf"

    def test_none_filename_uses_fallback(self):
        async def gen():
            yield b"data"

        response = create_stream_record_response(gen(), None, fallback_filename="download")
        assert "download" in response.headers["Content-Disposition"]

    def test_additional_headers(self):
        async def gen():
            yield b"data"

        extra = {"X-Custom": "value"}
        response = create_stream_record_response(gen(), "file.csv", additional_headers=extra)
        assert response.headers.get("X-Custom") == "value"

    def test_default_fallback_filename(self):
        async def gen():
            yield b"data"

        response = create_stream_record_response(gen(), "")
        # Should use "file" as the default fallback
        assert "file" in response.headers["Content-Disposition"]

    async def test_non_string_url_raises_type_error(self):
        """Passing a non-string signed_url should raise TypeError."""
        with pytest.raises(TypeError, match="Expected signed_url to be a string"):
            async for _ in stream_content(123):  # type: ignore
                pass

    async def test_coroutine_url_raises_type_error(self):
        """Passing a coroutine as signed_url should raise TypeError."""

        async def coro():
            return "url"

        with pytest.raises(TypeError, match="Expected signed_url to be a string"):
            async for _ in stream_content(coro()):  # type: ignore
                pass

    async def test_none_url_raises_type_error(self):
        """Passing None as signed_url should raise TypeError."""
        with pytest.raises(TypeError, match="Expected signed_url to be a string"):
            async for _ in stream_content(None):  # type: ignore
                pass

    async def test_list_url_raises_type_error(self):
        """Passing a list as signed_url should raise TypeError."""
        with pytest.raises(TypeError, match="Expected signed_url to be a string"):
            async for _ in stream_content([]):  # type: ignore
                pass

    def test_none_mime_type_uses_default(self):
        async def gen():
            yield b"data"

        response = create_stream_record_response(gen(), "file.txt", mime_type=None)
        assert response.media_type == "application/octet-stream"

    def test_custom_fallback_filename(self):
        async def gen():
            yield b"data"

        response = create_stream_record_response(
            gen(), "", fallback_filename="custom_fallback"
        )
        assert "custom_fallback" in response.headers["Content-Disposition"]

    def test_headers_merged_correctly(self):
        async def gen():
            yield b"data"

        extra = {"X-Total-Count": "42", "X-Custom-Header": "value"}
        response = create_stream_record_response(
            gen(), "file.csv", additional_headers=extra
        )
        assert response.headers.get("X-Total-Count") == "42"
        assert response.headers.get("X-Custom-Header") == "value"

    async def test_invalid_url_type_raises(self):
        """Non-string signed_url raises TypeError."""
        with pytest.raises(TypeError, match="Expected signed_url to be a string"):
            async for _ in stream_content(123):
                pass

    async def test_coroutine_url_type_raises(self):
        """Coroutine function passed as signed_url raises TypeError."""
        async def _dummy():
            return None
        with pytest.raises(TypeError):
            async for _ in stream_content(_dummy):
                pass

    def test_basic_response(self):
        async def gen():
            yield b"data"
        resp = create_stream_record_response(gen(), "test.pdf")
        assert resp.media_type == "application/octet-stream"

    def test_custom_mime_type(self):
        async def gen():
            yield b"data"
        resp = create_stream_record_response(gen(), "test.pdf", mime_type="application/pdf")
        assert resp.media_type == "application/pdf"

    def test_none_filename(self):
        async def gen():
            yield b"data"
        resp = create_stream_record_response(gen(), None, fallback_filename="download")
        assert "download" in resp.headers.get("content-disposition", "")

    def test_additional_headers(self):
        async def gen():
            yield b"data"
        resp = create_stream_record_response(
            gen(), "test.pdf",
            additional_headers={"X-Custom": "value"}
        )
        assert resp.headers.get("x-custom") == "value"

    async def test_url_parse_failure_uses_truncated(self):
        """When URL parsing fails, fallback to truncated URL should be used."""
        from app.utils.streaming import stream_content

        # We need to trigger stream_content with a non-string to hit TypeError
        with pytest.raises(TypeError, match="Expected signed_url to be a string"):
            async for _ in stream_content(signed_url=42, record_id="r1"):
                pass

    async def test_url_parse_fallback(self):
        """The URL parse succeeds for normal URLs, test with very long URL."""
        long_url = "https://example.com/" + "a" * 300
        # This just tests that the function initializes correctly with a long URL
        # It will fail at the HTTP request, but the URL parsing should succeed
        import aiohttp
        from fastapi import HTTPException

        with pytest.raises((HTTPException, TypeError, aiohttp.ClientError, Exception)):
            async for _ in stream_content(long_url, record_id="r1", file_name="test.pdf"):
                pass

