# VirtualPatientEngine/AIAgents4Pharma
# 1 LLM-backed test functions across 94 test files
# Source: https://github.com/VirtualPatientEngine/AIAgents4Pharma

# --- aiagents4pharma/talk2scholars/tests/test_agents_main_agent.py ---

def test_dummy_llm_generate():
    """Test the dummy LLM's generate function through public interface."""
    dummy = DummyLLM(model_name="test-model")
    # Test that the dummy LLM can be used (testing the class works)
    assert dummy.model_name == "test-model"
    # Test through public interface that internally calls _generate (covers lines 26-27)
    # Use invoke which internally calls _generate
    messages = [HumanMessage(content="test prompt")]
    result = dummy.invoke(messages)
    # Verify the internal state was set
    assert hasattr(DummyLLM, "called_prompt")
    assert result is not None
    assert DummyLLM.called_prompt == "test prompt"

