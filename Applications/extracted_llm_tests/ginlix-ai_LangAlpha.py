# ginlix-ai/LangAlpha
# 1 LLM-backed test functions across 177 test files
# Source: https://github.com/ginlix-ai/LangAlpha

# --- libs/ptc-cli/tests/unit_tests/test_streaming_errors.py ---

    def test_regular_exception_not_api_error(self):
        """Test that regular exceptions are not detected as API errors."""
        assert is_api_error(ValueError("test")) is False
        assert is_api_error(RuntimeError("test")) is False
        assert is_api_error(Exception("test")) is False

