# mozilla-ai/any-agent
# 1 test functions with real LLM calls
# Source: https://github.com/mozilla-ai/any-agent


# --- tests/unit/tools/test_unit_wrappers.py ---

def test_bad_functions(agent_framework: AgentFramework) -> None:
    """Test the verify_callable function with various bad functions."""

    # Test missing return type
    def missing_return_type(foo: str):  # type: ignore[no-untyped-def]
        """Docstring for foo."""
        return foo

    with pytest.raises(ValueError, match="return type"):
        asyncio.run(_wrap_tools([missing_return_type], agent_framework))

    # Test missing docstring
    def missing_docstring(foo: str) -> str:
        return foo

    with pytest.raises(ValueError, match="docstring"):
        asyncio.run(_wrap_tools([missing_docstring], agent_framework))

    # Test missing parameter type
    def missing_param_type(foo) -> str:  # type: ignore[no-untyped-def]
        """Docstring for foo."""
        return foo  # type: ignore[no-any-return]

    with pytest.raises(ValueError, match="typed arguments"):
        asyncio.run(_wrap_tools([missing_param_type], agent_framework))

    # Good function should not raise an error
    def good_function(foo: str) -> str:
        """Docstring for foo.
        Args:
            foo: The foo argument.
        Returns:
            The foo result.
        """
        return foo

    asyncio.run(_wrap_tools([good_function], agent_framework))

