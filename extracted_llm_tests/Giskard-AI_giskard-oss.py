# Giskard-AI/giskard-oss
# 5 test functions with real LLM calls
# Source: https://github.com/Giskard-AI/giskard-oss


# --- libs/giskard-agents/tests/test_generator.py ---

async def test_generator_completion(generator: LiteLLMGenerator):
    response = await generator.complete(
        messages=[
            Message(
                role="system",
                content="You are a helpful assistant, greeting the user with 'Hello I am TestBot'.",
            ),
            Message(role="user", content="Hello, world!"),
        ]
    )

    assert isinstance(response, Response)
    assert response.message.role == "assistant"
    assert isinstance(response.message.content, str)
    assert "I am TestBot" in response.message.content
    assert response.finish_reason == "stop"

async def test_generator_chat(generator: LiteLLMGenerator):
    test_message = "Hello, world!"
    pipeline = generator.chat(test_message)

    assert isinstance(pipeline, ChatWorkflow)
    assert len(pipeline.messages) == 1
    assert isinstance(pipeline.messages[0], Message)
    assert pipeline.messages[0].role == "user"
    assert pipeline.messages[0].content == test_message

    chat = await pipeline.run()

    assert isinstance(chat, Chat)

    chats = await pipeline.run_many(3)

    assert len(chats) == 3
    assert isinstance(chats[0], Chat)
    assert isinstance(chats[1], Chat)
    assert isinstance(chats[2], Chat)

async def test_call_model_receives_internal_types():
    """Verify _call_model receives Message and Tool objects, not dicts."""

    @tool
    def get_weather(city: str) -> str:
        """Get weather.

        Parameters
        ----------
        city : str
            City name.
        """
        return f"Sunny in {city}"

    gen = SpyGenerator(canned_response="All done")
    chat = await (
        ChatWorkflow(generator=gen)
        .chat("What's the weather?", role="user")
        .with_tools(get_weather)
        .run()
    )

    assert len(gen.calls) >= 1
    assert all(isinstance(m, Message) for m in gen.calls[0]["messages"])
    assert isinstance(gen.calls[0]["params"], GenerationParams)
    assert all(isinstance(t, Tool) for t in gen.calls[0]["params"].tools)

    assert chat.last.content == "All done"

    tool_msg = next(m for m in chat.messages if m.role == "tool")
    assert tool_msg.content == "Sunny in Paris"
    assert tool_msg.tool_call_id == "call_spy_1"

async def test_subclass_controls_message_serialization():
    """A subclass can transform messages however it likes inside _call_model."""

    class TaggingGenerator(BaseGenerator):
        @override
        async def _call_model(
            self,
            messages: list[Message],
            params: GenerationParams,
            metadata: dict[str, Any] | None = None,
        ) -> Response:
            last_content = messages[-1].content or ""
            tagged = f"[tagged] {last_content}"
            return Response(
                message=Message(role="assistant", content=tagged), finish_reason="stop"
            )

    gen = TaggingGenerator()
    chat = await ChatWorkflow(generator=gen).chat("hello", role="user").run()

    assert chat.last.content == "[tagged] hello"

async def test_subclass_controls_tool_serialization():
    """A subclass can reshape tool definitions inside _call_model."""

    @tool
    def my_tool(x: str) -> str:
        """A tool.

        Parameters
        ----------
        x : str
            Input.
        """
        return x

    class RenamedToolGenerator(BaseGenerator):
        @override
        async def _call_model(
            self,
            messages: list[Message],
            params: GenerationParams,
            metadata: dict[str, Any] | None = None,
        ) -> Response:
            content = f"custom_{params.tools[0].name}" if params.tools else "none"
            return Response(
                message=Message(role="assistant", content=content), finish_reason="stop"
            )

    gen = RenamedToolGenerator()
    chat = await (
        ChatWorkflow(generator=gen).chat("hi", role="user").with_tools(my_tool).run()
    )

    assert chat.last.content == "custom_my_tool"

