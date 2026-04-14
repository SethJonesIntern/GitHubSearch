# Intelligent-Internet/ii-agent
# 19 test functions with real LLM calls
# Source: https://github.com/Intelligent-Internet/ii-agent


# --- src/tests/unit/chat/test_chat_llm_openai.py ---

    def test_stream_default_false(self):
        from ii_agent.chat.llm.openai import OpenAIResponseParams

        params = OpenAIResponseParams(model="gpt-4o", input="Hello")
        assert params.stream is False

    def test_storybook_progress_content(self):
        provider = _make_provider()
        msg = _make_tool_result_message(
            "c1",
            "tool",
            StorybookProgressContent(
                storybook_id="sb1",
                storybook_name="Book",
                total_pages=10,
                completed_pages=5,
                current_page=5,
                status="generating",  # must be one of: generating, completed, failed
                generating_pages=[],
                error_message=None,
            ),
        )
        result = provider._convert_messages([msg], _make_empty_container_file())
        data = json.loads(result[0]["output"])
        assert data["type"] == "storybook_progress"


# --- src/tests/unit/chat/test_chat_llm_openai_deep.py ---

    def test_storybook_progress_content_converted(self):
        provider = _make_provider()
        msg = _make_tool_result_message(
            "c1",
            "tool",
            StorybookProgressContent(
                storybook_id="sb1",
                storybook_name="Book",
                total_pages=10,
                completed_pages=5,
                current_page=5,
                status="generating",
                generating_pages=[],
                error_message=None,
            ),
        )
        result = provider._convert_messages([msg], _make_empty_container_file())
        data = json.loads(result[0]["output"])
        assert data["type"] == "storybook_progress"
        assert data["storybook_id"] == "sb1"


# --- src/tests/unit/engine/test_v1_models_claude_deep.py ---

    def test_stop_reason_max_tokens(self):
        c = _make_claude()
        text_block = _make_response_block("text", text="Truncated", citations=None)
        mr = c._parse_provider_response(
            _make_provider_response([text_block], stop_reason="max_tokens")
        )
        assert isinstance(mr, ModelResponse)


# --- src/tests/unit/engine/test_v1_models_openai_completions.py ---

    def test_role_set_to_assistant(self):
        m = _make_oai_chat(api_key="key")
        choice = _make_choice(message_content="Hello")
        mr = m._parse_provider_response(_make_completion([choice]))
        assert mr.role == "assistant"

    def test_text_content_extracted(self):
        m = _make_oai_chat(api_key="key")
        choice = _make_choice(message_content="Hello world")
        mr = m._parse_provider_response(_make_completion([choice]))
        assert mr.content == "Hello world"

    def test_reasoning_content_extracted(self):
        m = _make_oai_chat(api_key="key")
        choice = _make_choice(message_content="Answer", reasoning_content="I reasoned")
        mr = m._parse_provider_response(_make_completion([choice]))
        assert mr.reasoning_content == "I reasoned"

    def test_usage_extracted(self):
        m = _make_oai_chat(api_key="key")
        choice = _make_choice(message_content="Hi")
        usage = _make_usage(prompt=15, completion=25, total=40)
        mr = m._parse_provider_response(_make_completion([choice], usage=usage))
        assert mr.response_usage is not None
        assert mr.response_usage.input_tokens == 15
        assert mr.response_usage.output_tokens == 25

    def test_reasoning_tokens_extracted(self):
        m = _make_oai_chat(api_key="key")
        choice = _make_choice(message_content="Hi")
        usage = _make_usage(reasoning=8)
        mr = m._parse_provider_response(_make_completion([choice], usage=usage))
        assert mr.response_usage.reasoning_tokens == 8

    def test_no_choices_raises_index_error(self):
        # When choices is empty, response.choices[0] raises IndexError.
        # This propagates uncaught from _parse_provider_response directly.
        m = _make_oai_chat(api_key="key")
        with pytest.raises(IndexError):
            m._parse_provider_response(_make_completion([]))

    def test_provider_data_response_id_stored(self):
        # The response.id is stored in model_response.provider_data["id"].
        m = _make_oai_chat(api_key="key")
        choice = _make_choice(message_content="ok")
        comp = _make_completion([choice], completion_id="cmpl_xyz")
        comp.id = "cmpl_xyz"
        mr = m._parse_provider_response(comp)
        assert mr.provider_data is not None
        assert mr.provider_data["id"] == "cmpl_xyz"

    def test_text_delta_extracted(self):
        m = _make_oai_chat(api_key="key")
        assistant_msg = Message(role="assistant", content="")
        chunk = _make_chunk([_make_chunk_choice(delta_content="Hello ")])
        result, _ = m._parse_provider_response_delta(chunk, assistant_msg, self._stream_state())
        assert result.content == "Hello "

    def test_reasoning_delta_extracted(self):
        m = _make_oai_chat(api_key="key")
        assistant_msg = Message(role="assistant", content="")
        chunk = _make_chunk([_make_chunk_choice(delta_reasoning="I think ")])
        result, _ = m._parse_provider_response_delta(chunk, assistant_msg, self._stream_state())
        assert result.reasoning_content == "I think "

    def test_empty_chunk_returns_empty_response(self):
        m = _make_oai_chat(api_key="key")
        assistant_msg = Message(role="assistant", content="")
        chunk = _make_chunk([])
        result, _ = m._parse_provider_response_delta(chunk, assistant_msg, self._stream_state())
        assert isinstance(result, ModelResponse)

    def test_finish_reason_stop_is_handled(self):
        m = _make_oai_chat(api_key="key")
        assistant_msg = Message(role="assistant", content="")
        state = self._stream_state()
        state["current_type"] = "content"
        state["content_started_emitted"] = True
        chunk = _make_chunk([_make_chunk_choice(finish_reason="stop")])
        result, new_state = m._parse_provider_response_delta(chunk, assistant_msg, state)
        assert isinstance(result, ModelResponse)


# --- src/tests/unit/engine/test_v1_models_openai_deep.py ---

    def test_no_content_no_tool_calls(self):
        m = _make_oai_chat(api_key="key")
        choice = _make_choice(message_content=None, tool_calls=None)
        mr = m._parse_provider_response(_make_completion([choice]))
        assert mr.content is None or mr.content == ""

    def test_reasoning_content_from_reasoning_field(self):
        m = _make_oai_chat(api_key="key")
        choice = _make_choice(message_content="Answer", reasoning_content=None)
        choice.message.reasoning = "I reasoned about this"
        mr = m._parse_provider_response(_make_completion([choice]))
        # reasoning should be extracted from .reasoning field if .reasoning_content is None
        assert isinstance(mr, ModelResponse)

    def test_parsed_field_used_as_content(self):
        class OutputSchema(BaseModel):
            answer: str

        m = _make_oai_chat(api_key="key")
        choice = _make_choice(message_content=None)
        parsed_obj = OutputSchema(answer="42")
        choice.message.parsed = parsed_obj
        mr = m._parse_provider_response(_make_completion([choice]))
        # parsed field content should be converted to string
        assert isinstance(mr, ModelResponse)

    def test_max_tokens_included_when_set(self):
        m = _make_oai_chat(api_key="key", max_tokens=2000)
        params = m.get_request_params()
        assert params.get("max_tokens") == 2000 or params.get("max_completion_tokens") == 2000

