# Corpus-OS/corpusos
# 18 LLM-backed test functions across 91 test files
# Source: https://github.com/Corpus-OS/corpusos

# --- tests/frameworks/embedding/test_llamaindex_adapter.py ---

async def test_sync_methods_called_in_event_loop_raise(adapter: Any) -> None:
    """
    Sync methods must refuse to run inside an active event loop to prevent deadlocks.
    """
    embeddings = _make_embeddings(adapter)

    with pytest.raises(RuntimeError) as exc:
        embeddings._get_query_embedding("x")
    msg = str(exc.value)
    assert ErrorCodes.SYNC_WRAPPER_CALLED_IN_EVENT_LOOP in msg

def test_shared_embedder_thread_safety(adapter: Any) -> None:
    embedder = configure_llamaindex_embeddings(
        corpus_adapter=adapter,
        model_name="concurrent-model",
        embedding_dimension=(None if hasattr(adapter, "get_embedding_dimension") else 8),
    )

    def embed_query(text: str) -> Any:
        return embedder._get_query_embedding(text)

    texts = [f"query {i}" for i in range(10)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(embed_query, text) for text in texts]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_error_handling_in_llamaindex_workflow_is_actionable(self) -> None:
        _require_llamaindex()

        class FailingAdapter:
            def embed(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
                raise RuntimeError("Rate limit exceeded: Please wait 60 seconds before retrying")

            def get_embedding_dimension(self) -> int:
                return 8

        failing_embedder = CorpusLlamaIndexEmbeddings(
            corpus_adapter=FailingAdapter(),
            model_name="failing-model",
        )

        with pytest.raises(RuntimeError) as exc_info:
            failing_embedder._get_text_embeddings(["test document"])

        s = str(exc_info.value).lower()
        assert "rate limit" in s or "exceeded" in s
        assert "wait" in s or "retry" in s


# --- tests/frameworks/llm/test_contract_shapes_and_batching.py ---

def test_sync_completion_result_type_stable_across_calls(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    For sync completions, the LLM client should return the same *type* on repeated calls.

    This catches frameworks that sometimes return a string and sometimes return
    a framework-specific result object, which would break callers relying on type stability.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare sync completion",
        )

    completion_fn = _get_method(
        llm_client_instance,
        framework_descriptor.completion_method,
    )

    args1, kwargs1 = _build_primary_call_args(framework_descriptor, completion_fn, text=SYNC_COMPLETION_TEXT_1)
    args2, kwargs2 = _build_primary_call_args(framework_descriptor, completion_fn, text=SYNC_COMPLETION_TEXT_2)

    result1 = completion_fn(*args1, **kwargs1)
    result2 = completion_fn(*args2, **kwargs2)

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(result2), (
        "Sync completion returned different types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
    )

async def test_async_completion_result_type_stable_across_calls_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When async completion is supported, it should return a stable result type
    across calls with similar inputs.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.async_completion_method:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare async completion",
        )

    acompletion_fn = _get_method(
        llm_client_instance,
        framework_descriptor.async_completion_method,
    )

    args1, kwargs1 = _build_primary_call_args(framework_descriptor, acompletion_fn, text=ASYNC_COMPLETION_TEXT_1)
    args2, kwargs2 = _build_primary_call_args(framework_descriptor, acompletion_fn, text=ASYNC_COMPLETION_TEXT_2)

    coro1 = acompletion_fn(*args1, **kwargs1)
    coro2 = acompletion_fn(*args2, **kwargs2)

    assert inspect.isawaitable(coro1)
    assert inspect.isawaitable(coro2)

    result1 = await coro1
    result2 = await coro2

    assert result1 is not None
    assert result2 is not None
    assert type(result1) is type(result2), (
        "Async completion returned different types across calls: "
        f"{type(result1).__name__} vs {type(result2).__name__}"
    )

def test_sync_and_async_completion_result_types_match_when_both_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When both sync and async completion are declared, their *result types* should match.

    Rationale:
    - Callers often swap between sync and async surfaces depending on runtime context.
    - Type drift between the two surfaces is surprising and complicates client code.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not (framework_descriptor.completion_method and framework_descriptor.async_completion_method):
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare both sync and async completion",
        )

    completion_fn = _get_method(llm_client_instance, framework_descriptor.completion_method)
    acompletion_fn = _get_method(llm_client_instance, framework_descriptor.async_completion_method)

    args_s, kwargs_s = _build_primary_call_args(framework_descriptor, completion_fn, text=SYNC_COMPLETION_TEXT_1)
    sync_result = completion_fn(*args_s, **kwargs_s)

    args_a, kwargs_a = _build_primary_call_args(framework_descriptor, acompletion_fn, text=SYNC_COMPLETION_TEXT_1)
    coro = acompletion_fn(*args_a, **kwargs_a)
    assert inspect.isawaitable(coro), "async_completion_method must return an awaitable"

    # Execute the awaitable safely from a sync test:
    # - Prefer .send(None)/loop tricks? Not safe.
    # - Use pytest's async test for execution is cleaner; therefore we only compare
    #   types by awaiting via asyncio.run when no loop is running.
    #
    # NOTE: This is intentionally conservative; if an event loop is already running,
    # we skip to avoid nested-loop hazards. The async-only parity is still covered by
    # the async tests in this suite.
    try:
        import asyncio

        asyncio.get_running_loop()
        pytest.skip("Skipping sync/async completion type parity because an event loop is already running")
    except RuntimeError:
        import asyncio

        async_result = asyncio.run(coro)

    assert sync_result is not None
    assert async_result is not None
    assert type(sync_result) is type(async_result), (
        "Sync and async completion returned different result types: "
        f"{type(sync_result).__name__} vs {type(async_result).__name__}"
    )

def test_stream_chunk_type_consistent_within_stream_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When streaming is supported (via dedicated method or streaming kwarg),
    all chunks yielded from a single sync stream should have a consistent type.

    We don't enforce any particular chunk *shape* here, only that a single
    stream doesn't mix, e.g., strings and dicts.

    We also cap consumption to MAX_STREAM_CHUNKS_TO_SAMPLE to avoid hangs.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_streaming:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare streaming support",
        )

    iterator = _invoke_sync_stream(
        framework_descriptor,
        llm_client_instance,
        SYNC_STREAM_TEXT,
    )
    _assert_iterable(iterator)

    first_chunk_type: type[Any] | None = None
    chunks_seen = 0

    for chunk in iterator:
        chunks_seen += 1
        if first_chunk_type is None:
            first_chunk_type = type(chunk)
        else:
            assert type(chunk) is first_chunk_type, (
                "Sync streaming yielded chunks of inconsistent types within a "
                f"single stream: {first_chunk_type.__name__} vs {type(chunk).__name__}"
            )
        if chunks_seen >= MAX_STREAM_CHUNKS_TO_SAMPLE:
            break

async def test_async_stream_chunk_type_consistent_within_stream_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When async streaming is supported (via dedicated async streaming method
    or via async completion + streaming kwarg), all chunks yielded from a
    single async stream should have a consistent type.

    The async streaming surface may be an async iterator directly, or an
    awaitable resolving to one.

    We also cap consumption to MAX_STREAM_CHUNKS_TO_SAMPLE to avoid hangs.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_streaming:
        pytest.skip(
            f"Framework '{framework_descriptor.name}' does not declare streaming support",
        )

    aiter = await _invoke_async_stream(
        framework_descriptor,
        llm_client_instance,
        ASYNC_STREAM_TEXT,
    )
    _assert_async_iterable(aiter)

    first_chunk_type: type[Any] | None = None
    chunks_seen = 0

    async for chunk in aiter:  # noqa: B007
        chunks_seen += 1
        if first_chunk_type is None:
            first_chunk_type = type(chunk)
        else:
            assert type(chunk) is first_chunk_type, (
                "Async streaming yielded chunks of inconsistent types within a "
                f"single stream: {first_chunk_type.__name__} vs {type(chunk).__name__}"
            )
        if chunks_seen >= MAX_STREAM_CHUNKS_TO_SAMPLE:
            break

def test_sync_stream_first_chunk_type_stable_across_calls_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When streaming is supported, the *type of the first chunk* should be stable
    across separate stream invocations.

    Why this matters:
    - Many clients sample early chunks for routing/format decisions.
    - A stream that sometimes starts with a dict "event" and sometimes starts
      with a string chunk is hard to consume generically.

    This test does not require streams to produce any chunks; if both streams are
    empty, we treat that as acceptable for this smoke suite.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_streaming:
        pytest.skip(f"{framework_descriptor.name}: streaming not declared")

    stream1 = _invoke_sync_stream(framework_descriptor, llm_client_instance, SYNC_STREAM_TEXT + "-a")
    stream2 = _invoke_sync_stream(framework_descriptor, llm_client_instance, SYNC_STREAM_TEXT + "-b")

    _assert_iterable(stream1)
    _assert_iterable(stream2)

    first1 = _sync_first_chunk(stream1)
    first2 = _sync_first_chunk(stream2)

    if first1 is None and first2 is None:
        return

    assert first1 is not None and first2 is not None, (
        "One stream produced a first chunk while the other produced none; "
        "this may indicate inconsistent buffering behavior."
    )
    assert type(first1) is type(first2), (
        "First streaming chunk types differed across calls: "
        f"{type(first1).__name__} vs {type(first2).__name__}"
    )

async def test_async_stream_first_chunk_type_stable_across_calls_when_supported(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Async companion to the sync first-chunk type stability test.

    This validates that two async streams start with the same chunk type.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_streaming:
        pytest.skip(f"{framework_descriptor.name}: streaming not declared")

    aiter1 = await _invoke_async_stream(framework_descriptor, llm_client_instance, ASYNC_STREAM_TEXT + "-a")
    aiter2 = await _invoke_async_stream(framework_descriptor, llm_client_instance, ASYNC_STREAM_TEXT + "-b")

    _assert_async_iterable(aiter1)
    _assert_async_iterable(aiter2)

    first1 = await _async_first_chunk(aiter1)
    first2 = await _async_first_chunk(aiter2)

    if first1 is None and first2 is None:
        return

    assert first1 is not None and first2 is not None, (
        "One async stream produced a first chunk while the other produced none; "
        "this may indicate inconsistent buffering behavior."
    )
    assert type(first1) is type(first2), (
        "First async streaming chunk types differed across calls: "
        f"{type(first1).__name__} vs {type(first2).__name__}"
    )

async def test_stream_first_chunk_type_matches_between_sync_and_async_when_both_declared(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    When both sync and async streaming are declared and usable, the *first chunk type*
    should match between the two surfaces.

    This is a best-effort parity check:
    - If either stream yields no chunks, we do not fail (this suite is smoke-style).
    - If both yield a first chunk, we require type parity.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    # Only run this parity test when both streaming surfaces can be invoked.
    # For kwarg-style streaming, both sync and async can still be present via completion methods.
    if not framework_descriptor.supports_streaming:
        pytest.skip(f"{framework_descriptor.name}: streaming not declared")

    # Sync stream
    # Run in a worker thread to avoid calling sync adapters from an event loop.
    first_sync = await asyncio.to_thread(
        _sync_first_chunk_for_descriptor,
        framework_descriptor,
        llm_client_instance,
        SYNC_STREAM_TEXT + "-parity",
    )

    # Async stream
    async_stream = await _invoke_async_stream(framework_descriptor, llm_client_instance, ASYNC_STREAM_TEXT + "-parity")
    _assert_async_iterable(async_stream)
    first_async = await _async_first_chunk(async_stream)

    if first_sync is None or first_async is None:
        return

    assert type(first_sync) is type(first_async), (
        "Sync/async first streaming chunk types differ: "
        f"{type(first_sync).__name__} vs {type(first_async).__name__}"
    )

def test_streaming_surface_is_resolvable_when_supports_streaming_true(
    framework_descriptor: LLMFrameworkDescriptor,
    llm_client_instance: Any,
) -> None:
    """
    Registry / descriptor coherence check (LLM-specific).

    If supports_streaming=True, at least one of the following must be usable:
      - method-style streaming: streaming_method or async_streaming_method (or both)
      - kwarg-style streaming: completion_method/async_completion_method with streaming_kwarg

    This prevents "supports_streaming=True" from silently becoming a no-op.
    """
    if llm_client_instance is None:
        _assert_unavailable_contract(framework_descriptor)
        return

    if not framework_descriptor.supports_streaming:
        # If not supported, the descriptor should not claim streaming methods/kwargs.
        assert framework_descriptor.streaming_method is None
        assert framework_descriptor.async_streaming_method is None
        assert framework_descriptor.streaming_kwarg is None
        return

    if framework_descriptor.streaming_style == "method":
        assert (
            framework_descriptor.streaming_method is not None
            or framework_descriptor.async_streaming_method is not None
        ), (
            f"{framework_descriptor.name}: supports_streaming=True and streaming_style='method' "
            "but no streaming_method/async_streaming_method is declared"
        )
        # If declared, ensure it exists/callable on the instance.
        if framework_descriptor.streaming_method:
            _get_method(llm_client_instance, framework_descriptor.streaming_method)
        if framework_descriptor.async_streaming_method:
            _get_method(llm_client_instance, framework_descriptor.async_streaming_method)

    if framework_descriptor.streaming_style == "kwarg":
        assert framework_descriptor.streaming_kwarg is not None, (
            f"{framework_descriptor.name}: streaming_style='kwarg' but streaming_kwarg is None"
        )
        assert (
            framework_descriptor.completion_method is not None
            or framework_descriptor.async_completion_method is not None
        ), (
            f"{framework_descriptor.name}: streaming_style='kwarg' requires a completion surface, "
            "but neither completion_method nor async_completion_method is declared"
        )
        if framework_descriptor.completion_method:
            _get_method(llm_client_instance, framework_descriptor.completion_method)
        if framework_descriptor.async_completion_method:
            _get_method(llm_client_instance, framework_descriptor.async_completion_method)


# --- tests/frameworks/llm/test_llamaindex_llm_adapter.py ---

async def test_achat_validates_empty_messages(adapter: Any) -> None:
    """achat() should raise BadRequest for empty messages."""
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    
    with pytest.raises(Exception, match="empty"):
        await llm.achat([])

def test_llamaindex_streaming_with_chatmessage_objects(adapter: Any) -> None:
    """Streaming should work with real LlamaIndex ChatMessage objects."""
    _require_llamaindex_available_for_integration()

    from llama_index.core.llms import ChatMessage, MessageRole
    
    llm = CorpusLlamaIndexLLM(llm_adapter=adapter)
    messages = [ChatMessage(role=MessageRole.USER, content="stream test")]
    
    chunks = list(llm.stream_chat(messages))
    
    assert len(chunks) > 0
    for chunk in chunks:
        # Verify ChatResponse structure
        assert hasattr(chunk, "message")
        assert hasattr(chunk, "delta")


# --- tests/frameworks/vector/test_llamaindex_adapter.py ---

def test_add_guards_event_loop(adapter: Any, TextNode: Any) -> None:
    """Should raise RuntimeError in event loop."""

    @pytest.mark.asyncio
    async def test_in_loop():
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        node = TextNode(text="test", id_="node-1", embedding=[0.1, 0.2])

        with pytest.raises(RuntimeError, match="event loop"):
            store.add([node])

    asyncio.run(test_in_loop())

async def test_ensure_not_in_event_loop_raises_in_loop() -> None:
    """Should raise RuntimeError when called in event loop."""
    with pytest.raises(RuntimeError, match="event loop"):
        _ensure_not_in_event_loop("test_api")

def test_query_stream_raises_without_embedding(adapter: Any, VectorStoreQuery: Any) -> None:
    """Should raise if query_embedding is None."""
    store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
    
    query = VectorStoreQuery(query_embedding=None, similarity_top_k=4)
    
    with pytest.raises(NotSupported, match="query_embedding is None"):
        list(store.query_stream(query))

    async def test_in_loop():
        store = CorpusLlamaIndexVectorStore(corpus_adapter=adapter)
        node = TextNode(text="test", id_="node-1", embedding=[0.1, 0.2])

        with pytest.raises(RuntimeError, match="event loop"):
            store.add([node])

