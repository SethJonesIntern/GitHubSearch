# existential-birds/amelia
# 2 LLM-backed test functions across 235 test files
# Source: https://github.com/existential-birds/amelia

# --- tests/unit/test_api_driver.py ---

    async def test_rejects_empty_prompt(self, api_driver: ApiDriver) -> None:
        """Should reject empty or whitespace-only prompts."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await api_driver.generate("")

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await api_driver.generate("   \n\t  ")

    def test_execute_returns_stdout(self, sandbox: LocalSandbox) -> None:
        """Should capture stdout from command."""
        result = sandbox.execute("echo hello")
        assert "hello" in result.output
        assert result.exit_code == 0
        assert result.truncated is False

