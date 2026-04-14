# livekit/agents
# 12 test functions with real LLM calls
# Source: https://github.com/livekit/agents


# --- tests/test_realtime/test_realtime.py ---

async def test_generate_reply(rt_session: llm.RealtimeSession):
    gen_ev = await asyncio.wait_for(rt_session.generate_reply(), timeout=15)
    assert gen_ev.user_initiated
    text = await asyncio.wait_for(_collect_text(gen_ev), timeout=15)
    assert len(text) > 0

async def test_generate_reply_with_instructions(rt_session: llm.RealtimeSession):
    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(instructions="Say exactly: pineapple"),
        timeout=15,
    )
    text = await asyncio.wait_for(_collect_text(gen_ev), timeout=15)
    assert "pineapple" in text.lower()

async def test_multiple_sequential_replies(rt_session: llm.RealtimeSession):
    for i in range(2):
        gen_ev = await asyncio.wait_for(
            rt_session.generate_reply(instructions=f"Say the number {i}"),
            timeout=20,
        )
        text = await asyncio.wait_for(_collect_text(gen_ev), timeout=15)
        assert len(text) > 0, f"Reply {i} was empty"

async def test_response_includes_audio(rt_session: llm.RealtimeSession):
    gen_ev = await asyncio.wait_for(rt_session.generate_reply(), timeout=15)

    got_audio = False
    async for msg_gen in gen_ev.message_stream:
        modalities = await asyncio.wait_for(msg_gen.modalities, timeout=10)
        assert "audio" in modalities
        async for frame in msg_gen.audio_stream:
            assert frame.sample_rate == SAMPLE_RATE
            assert frame.num_channels == 1
            assert len(frame.data) > 0
            got_audio = True
            break
    assert got_audio

async def test_input_audio_transcription(rt_session: llm.RealtimeSession):
    transcripts: list[str] = []
    transcript_received = asyncio.Event()

    def on_transcript(ev: llm.InputTranscriptionCompleted):
        transcripts.append(ev.transcript)
        transcript_received.set()

    rt_session.on("input_audio_transcription_completed", on_transcript)
    await _push_speech(rt_session, "weather_question")
    rt_session.commit_audio()

    await asyncio.wait_for(transcript_received.wait(), timeout=15)
    full = " ".join(transcripts).lower()
    assert "weather" in full or "paris" in full

async def test_update_chat_ctx(rt_session: llm.RealtimeSession):
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="My favorite number is seven")
    await asyncio.wait_for(rt_session.update_chat_ctx(chat_ctx), timeout=10)

    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(
            instructions="What is the user's favorite number? Reply with just the number."
        ),
        timeout=15,
    )
    text = await asyncio.wait_for(_collect_text(gen_ev), timeout=15)
    assert "seven" in text.lower() or "7" in text

async def test_chat_ctx_populated_after_reply(rt_session: llm.RealtimeSession):
    gen_ev = await asyncio.wait_for(rt_session.generate_reply(), timeout=15)
    await asyncio.wait_for(_collect_text(gen_ev), timeout=15)

    ctx = rt_session.chat_ctx
    assert len(ctx.items) > 0
    assert any(item.type == "message" and item.role == "assistant" for item in ctx.items)

async def test_update_chat_ctx_replaces_history(rt_session: llm.RealtimeSession):
    ctx1 = llm.ChatContext()
    ctx1.add_message(role="user", content="Remember: color is red")
    await asyncio.wait_for(rt_session.update_chat_ctx(ctx1), timeout=10)

    ctx2 = llm.ChatContext()
    ctx2.add_message(role="user", content="Remember: color is blue")
    await asyncio.wait_for(rt_session.update_chat_ctx(ctx2), timeout=10)

    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(instructions="What color did the user mention?"),
        timeout=15,
    )
    text = await asyncio.wait_for(_collect_text(gen_ev), timeout=15)
    assert "blue" in text.lower()

async def test_interrupt(rt_session: llm.RealtimeSession):
    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(
            instructions="Write a very long essay about the history of computing."
        ),
        timeout=15,
    )
    got_chunk = False
    async for msg_gen in gen_ev.message_stream:
        async for _ in msg_gen.text_stream:
            got_chunk = True
            rt_session.interrupt()
            break
        break
    assert got_chunk

async def test_function_tool(rt_session: llm.RealtimeSession):
    @function_tool
    async def get_weather(ctx: RunContext, city: str) -> str:
        """Get the weather for a city.
        Args:
            city: The city name
        """
        return f"Weather in {city}: sunny, 72F"

    await rt_session.update_tools([get_weather])
    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(instructions="What's the weather in Paris? Use the tool."),
        timeout=15,
    )

    got_function_call = False
    async for fn_call in gen_ev.function_stream:
        assert fn_call.name == "get_weather"
        got_function_call = True
    assert got_function_call

async def test_function_tool_reply(rt_session: llm.RealtimeSession):
    @function_tool
    async def get_capital(ctx: RunContext, country: str) -> str:
        """Get the capital of a country.
        Args:
            country: The country name
        """
        return "Paris"

    await rt_session.update_tools([get_capital])
    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(instructions="What is the capital of France? Use the tool."),
        timeout=15,
    )

    fn_call: llm.FunctionCall | None = None
    async for fc in gen_ev.function_stream:
        fn_call = fc
    assert fn_call is not None

    chat_ctx = rt_session.chat_ctx.copy()
    chat_ctx.items.append(
        llm.FunctionCallOutput(
            id=utils.shortuuid(),
            call_id=fn_call.call_id,
            output="Paris",
            is_error=False,
        )
    )
    await asyncio.wait_for(rt_session.update_chat_ctx(chat_ctx), timeout=10)

    gen_ev2 = await asyncio.wait_for(rt_session.generate_reply(), timeout=15)
    text = await asyncio.wait_for(_collect_text(gen_ev2), timeout=15)
    assert "paris" in text.lower()

async def test_remote_item_added_event(rt_session: llm.RealtimeSession):
    items_added: list[llm.RemoteItemAddedEvent] = []
    rt_session.on("remote_item_added", lambda ev: items_added.append(ev))

    gen_ev = await asyncio.wait_for(rt_session.generate_reply(), timeout=15)
    await asyncio.wait_for(_collect_text(gen_ev), timeout=15)

    assert len(items_added) > 0
    assert any(
        item.item.type == "message" and item.item.role == "assistant" for item in items_added
    )

