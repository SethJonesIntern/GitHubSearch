# SylphAI-Inc/AdalFlow
# 5 test functions with real LLM calls
# Source: https://github.com/SylphAI-Inc/AdalFlow


# --- adalflow/tests/test_anthropic_client.py ---

    def test_streaming_state_initialization(self):
        """Test StreamingState initialization"""
        state = StreamingState()
        self.assertFalse(state.started)
        self.assertIsNone(state.text_content_index_and_output)
        self.assertIsNone(state.refusal_content_index_and_output)
        self.assertEqual(state.function_calls, {})


# --- adalflow/tests/test_chat_completion_to_response_converter.py ---

    def test_streaming_state_initialization(self):
        """Test StreamingState initialization"""
        state = StreamingState()
        self.assertFalse(state.started)
        self.assertIsNone(state.text_content_index_and_output)
        self.assertIsNone(state.refusal_content_index_and_output)
        self.assertEqual(state.function_calls, {})

    def test_streaming_state_started_flag(self):
        """Test StreamingState started flag"""
        state = StreamingState()
        state.started = True
        self.assertTrue(state.started)


# --- adalflow/tests/test_ollama_client.py ---

    def test_async_streaming_chat_sync(self):
        """Test async streaming from synchronous context"""
        # This is a wrapper to run the async test in a sync context
        asyncio.run(self.test_async_streaming_chat())

    def test_streaming_with_generator_sync(self):
        """Test streaming with Generator component from sync context"""
        asyncio.run(self.test_streaming_with_generator())

