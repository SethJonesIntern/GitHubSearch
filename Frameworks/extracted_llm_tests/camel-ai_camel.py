# camel-ai/camel
# 16 test functions with real LLM calls
# Source: https://github.com/camel-ai/camel


# --- test/agents/test_chat_agent.py ---

def test_chat_agent_stream_accumulate_mode_accumulated():
    """Verify accumulated streaming behavior (stream_accumulate=True)."""
    chunks = ["Hello", " ", "world"]
    step_usage = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
    }

    agent = ChatAgent(stream_accumulate=True)
    accumulator = StreamContentAccumulator()
    outputs = []
    for c in chunks:
        resp = agent._create_streaming_response_with_accumulator(
            accumulator, c, step_usage, "acc", []
        )
        outputs.append(resp.msg.content)

    assert len(outputs) == 3
    assert outputs[0] == "Hello"
    assert outputs[1] == "Hello "
    assert outputs[2] == "Hello world"
    assert accumulator.get_full_content() == "Hello world"

def test_chat_agent_stream_accumulate_mode_delta():
    """Verify delta streaming behavior (stream_accumulate=False, default)."""
    chunks = ["Hello", " ", "world"]
    step_usage = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
    }

    agent = ChatAgent(stream_accumulate=False)
    accumulator = StreamContentAccumulator()
    outputs = []
    for c in chunks:
        resp = agent._create_streaming_response_with_accumulator(
            accumulator, c, step_usage, "delta", []
        )
        outputs.append(resp.msg.content)

    assert outputs == chunks
    assert accumulator.get_full_content() == "Hello world"

def test_chat_agent_stream_with_structured_output():
    r"""Test streaming with structured output (response_format).

    This is an e2e test for the fix that properly detects and handles
    ChatCompletionStreamManager returned by OpenAI when using
    streaming + structured output together.
    """

    class MathResult(BaseModel):
        answer: int = Field(description="The numerical answer")
        explanation: str = Field(description="Brief explanation")

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1,
        model_config_dict={"stream": True},
    )

    agent = ChatAgent(
        system_message="You are a helpful math assistant.",
        model=model,
    )

    # Stream with structured output - this triggers ChatCompletionStreamManager
    responses = []
    for response in agent.step("What is 2 + 2?", response_format=MathResult):
        responses.append(response)

    assert len(responses) > 1, "Should receive multiple streaming chunks"
    assert responses[-1].msg.parsed.answer == 4
    assert responses[-1].msg.parsed.explanation

async def test_chat_agent_async_stream_with_structured_output():
    r"""Test async streaming with structured output (response_format).

    This is an e2e test for the fix that properly detects and handles
    AsyncChatCompletionStreamManager returned by OpenAI when using
    async streaming + structured output together.
    """

    class MathResult(BaseModel):
        answer: int = Field(description="The numerical answer")
        explanation: str = Field(description="Brief explanation")

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1,
        model_config_dict={"stream": True},
    )

    agent = ChatAgent(
        system_message="You are a helpful math assistant.",
        model=model,
    )

    # Async stream with structured output - triggers
    # AsyncChatCompletionStreamManager
    responses = []
    async for response in await agent.astep(
        "What is 3 + 3?", response_format=MathResult
    ):
        responses.append(response)

    assert len(responses) > 1, "Should receive multiple streaming chunks"
    assert responses[-1].msg.parsed.answer == 6
    assert responses[-1].msg.parsed.explanation


# --- test/embeddings/test_vlm_embeddings.py ---

def test_image_embed_list_with_valid_input(VLM_instance):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    test_images = [image, image]
    embeddings = VLM_instance.embed_list(test_images)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    for e in embeddings:
        assert len(e) == VLM_instance.get_output_dim()

def test_mixed_embed_list_with_valid_input(VLM_instance):
    test_list = ['Hello world', 'Testing sentence embeddings']
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    test_list.append(image)
    embeddings = VLM_instance.embed_list(test_list)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    for e in embeddings:
        assert len(e) == VLM_instance.get_output_dim()


# --- test/models/test_aws_bedrock_model.py ---

async def test_aws_bedrock_async_supported():
    r"""Test AWSBedrockModel async method is now supported.

    This test verifies that async inference is supported by ensuring
    it doesn't raise NotImplementedError. Instead, it should attempt
    to make a connection and fail with APIConnectionError due to
    invalid credentials in the test environment.
    """
    from openai import APIConnectionError

    model = AWSBedrockModel(
        ModelType.AWS_CLAUDE_3_HAIKU,
        api_key="dummy_key",
        url="http://dummy.url",
    )

    # Async should now be supported, so it should attempt to connect
    # and fail with a connection error (not NotImplementedError)
    with pytest.raises(APIConnectionError):
        await model._arun([{"role": "user", "content": "Test message"}])


# --- test/models/test_base_model.py ---

    def test_metaclass_preprocessing(self):
        r"""Test that metaclass automatically preprocesses messages in run
        method."""
        processed_messages = None

        class TestModel(BaseModelBackend):
            @property
            def token_counter(self):
                pass

            def run(self, messages):
                nonlocal processed_messages
                processed_messages = messages
                return None

            def _run(self, messages, response_format=None, tools=None):
                pass

            async def _arun(self, messages, response_format=None, tools=None):
                pass

        model = TestModel(ModelType.GPT_4O_MINI)
        messages = [
            {'role': 'user', 'content': 'Hello <think>hi</think> world'}
        ]

        # Call run method and verify messages were preprocessed
        model.run(messages)
        assert processed_messages is not None
        assert processed_messages[0]['content'] == 'Hello  world'

    def test_postprocess_extracts_think_tags(self):
        r"""Test that postprocess_response extracts <think> tags into
        reasoning_content."""
        model = self._make_model()
        response = self._make_completion('<think>reasoning here</think>\n\n4')

        result = model.postprocess_response(response)
        assert result.choices[0].message.content == '4'
        assert result.choices[0].message.reasoning_content == 'reasoning here'

    def test_postprocess_multiline_think_tags(self):
        r"""Test extraction of multiline think tags."""
        model = self._make_model()
        response = self._make_completion(
            '<think>\nline1\nline2\n</think>\n\nanswer'
        )

        result = model.postprocess_response(response)
        assert result.choices[0].message.content == 'answer'
        assert result.choices[0].message.reasoning_content == 'line1\nline2'

    def test_postprocess_no_think_tags(self):
        r"""Test that content without think tags is unchanged."""
        model = self._make_model()
        response = self._make_completion('just a normal response')

        result = model.postprocess_response(response)
        assert result.choices[0].message.content == 'just a normal response'
        assert (
            getattr(result.choices[0].message, 'reasoning_content', None)
            is None
        )

    def test_postprocess_preserves_existing_reasoning_content(self):
        r"""Test that existing reasoning_content is not overridden."""
        model = self._make_model()
        response = self._make_completion(
            '<think>in content</think>\n\n4',
            reasoning_content='already set',
        )

        result = model.postprocess_response(response)
        # Content should NOT be cleaned since reasoning_content exists
        assert (
            result.choices[0].message.content
            == '<think>in content</think>\n\n4'
        )
        assert result.choices[0].message.reasoning_content == 'already set'

    def test_think_tags(self, extract_thinking_from_response):
        r"""Test that extract_thinking_from_response controls both
        preprocessing (input) and postprocessing (output) of think tags."""
        model = self._make_model(
            extract_thinking_from_response=extract_thinking_from_response
        )

        # Test preprocessing
        messages = [
            {
                'role': 'assistant',
                'content': '<think>thinking</think>Response',
            },
            {
                'role': 'user',
                'content': 'Hello <think>thought</think> world',
            },
        ]
        processed = model.preprocess_messages(messages)

        # Test postprocessing
        response = self._make_completion('<think>reasoning here</think>\n\n4')
        result = model.postprocess_response(response)

        if extract_thinking_from_response:
            # Preprocessing: think tags should be stripped
            assert processed[0]['content'] == 'Response'
            assert processed[1]['content'] == 'Hello  world'
            # Postprocessing: think tags extracted into reasoning_content
            assert result.choices[0].message.content == '4'
            assert (
                result.choices[0].message.reasoning_content == 'reasoning here'
            )
        else:
            # Preprocessing: think tags should be preserved
            assert processed[0]['content'] == '<think>thinking</think>Response'
            assert (
                processed[1]['content'] == 'Hello <think>thought</think> world'
            )
            # Postprocessing: think tags should be preserved
            assert (
                result.choices[0].message.content
                == '<think>reasoning here</think>\n\n4'
            )
            assert (
                getattr(result.choices[0].message, 'reasoning_content', None)
                is None
            )

    def test_postprocess_multiple_think_tags(self):
        r"""Test extraction with multiple think tags in content."""
        model = self._make_model()
        response = self._make_completion(
            '<think>first thought</think> middle '
            '<think>second thought</think> end'
        )

        result = model.postprocess_response(response)
        assert result.choices[0].message.content == 'middle  end'
        assert result.choices[0].message.reasoning_content == 'first thought'

    def test_postprocess_empty_think_tags(self):
        r"""Test that empty think tags are stripped without setting
        reasoning_content."""
        model = self._make_model()
        response = self._make_completion('<think></think>content after')

        result = model.postprocess_response(response)
        assert result.choices[0].message.content == 'content after'

    def test_metaclass_postprocessing(self):
        r"""Test that metaclass wraps run with postprocessing."""

        class TestModel(BaseModelBackend):
            @property
            def token_counter(self):
                pass

            def run(self, messages, response_format=None, tools=None):
                return self._make_response()

            def _run(self, messages, response_format=None, tools=None):
                pass

            async def _arun(self, messages, response_format=None, tools=None):
                pass

            def _make_response(self):
                return ChatCompletion(
                    id='test',
                    model='test',
                    object='chat.completion',
                    created=0,
                    choices=[
                        Choice(
                            index=0,
                            finish_reason='stop',
                            message=ChatCompletionMessage(
                                role='assistant',
                                content='<think>thought</think>answer',
                            ),
                        )
                    ],
                )

        model = TestModel(ModelType.GPT_4O_MINI)
        result = model.run([{'role': 'user', 'content': 'test'}])

        # Metaclass should have applied postprocess_response
        assert result.choices[0].message.content == 'answer'
        assert result.choices[0].message.reasoning_content == 'thought'

