# Arize-ai/phoenix
# 271 LLM-backed test functions across 200 test files
# Source: https://github.com/Arize-ai/phoenix

# --- packages/phoenix-client/tests/canary/sdk/openai/test_chat.py ---

    def test_round_trip(self, obj: CompletionCreateParamsBase) -> None:
        pv: v1.PromptVersionData = create_prompt_version_from_openai(obj)
        messages, kwargs = to_chat_messages_and_kwargs(pv, formatter=NO_OP_FORMATTER)
        new_obj: CompletionCreateParamsBase = {"messages": messages, **kwargs}  # type: ignore[typeddict-item]
        assert not DeepDiff(obj, new_obj)


# --- packages/phoenix-client/tests/client/helpers/atif/test_convert.py ---

    def test_llm_token_counts(self, simple_trajectory: Dict[str, Any]) -> None:
        spans = _convert_atif_trajectory_to_spans(simple_trajectory)
        llm_spans = [s for s in spans if s["span_kind"] == "LLM"]
        attrs = llm_spans[0].get("attributes", {})
        assert attrs.get("llm.token_count.prompt") == 520
        assert attrs.get("llm.token_count.completion") == 80
        assert attrs.get("llm.token_count.total") == 600

    def test_final_metrics_on_root(self, multi_tool_trajectory: Dict[str, Any]) -> None:
        spans = _convert_atif_trajectory_to_spans(multi_tool_trajectory)
        root = spans[0]
        attrs = root.get("attributes", {})
        assert attrs.get("llm.token_count.prompt") == 9150
        assert attrs.get("llm.token_count.completion") == 635


# --- packages/phoenix-client/tests/client/resources/datasets/test_datasets.py ---

    def test_is_valid_dataset_example(self) -> None:
        valid_example = v1.DatasetExample(
            id="ex1",
            input={"text": "hello"},
            output={"response": "hi"},
            metadata={"source": "test"},
            updated_at="2024-01-15T10:00:00",
        )
        assert _is_input_dataset_example(valid_example)
        assert not _is_input_dataset_example({"incomplete": "dict"})
        assert not _is_input_dataset_example("not a dict")

    def test_dataframe_to_csv_preparation(self) -> None:
        # Create a DataFrame with complex data
        df = pd.DataFrame(
            {
                "input_text": ["What is AI?", "Define ML"],
                "input_context": ["technology", "computer science"],
                "output_answer": ["Artificial Intelligence", "Machine Learning"],
                "metadata_source": ["wiki", "textbook"],
            }
        )

        keys = DatasetKeys(
            input_keys=frozenset(["input_text", "input_context"]),
            output_keys=frozenset(["output_answer"]),
            metadata_keys=frozenset(["metadata_source"]),
        )

        name, file_obj, content_type, headers = _prepare_dataframe_as_csv(df, keys)

        assert name == "dataframe.csv"
        assert content_type == "text/csv"
        assert headers == {"Content-Encoding": "gzip"}

        file_obj.seek(0)
        decompressed = gzip.decompress(file_obj.read()).decode()

        generated_df = pd.read_csv(StringIO(decompressed))  # pyright: ignore[reportUnknownMemberType]

        # Verify structure - columns should be sorted alphabetically within each group
        # (input keys sorted, output keys sorted, metadata keys sorted)
        expected_columns = [
            "input_context",
            "input_text",
            "output_answer",
            "metadata_source",
        ]
        assert list(generated_df.columns) == expected_columns
        assert len(generated_df) == 2

        assert generated_df.iloc[0]["input_text"] == "What is AI?"  # pyright: ignore[reportGeneralTypeIssues]
        assert generated_df.iloc[1]["output_answer"] == "Machine Learning"  # pyright: ignore[reportGeneralTypeIssues]


# --- packages/phoenix-client/tests/client/utils/test_annotation_hepers.py ---

    def test_index_based_id_validation(self) -> None:
        """Test validation for ID columns provided via index instead of columns."""
        # Single ID in named index should pass
        df_single_index = pd.DataFrame(
            {
                "name": ["test"],
                "annotator_kind": ["HUMAN"],
                "label": ["positive"],
            },
            index=pd.Index(["span1"], name="span_id"),
        )
        _validate_span_annotations_dataframe(dataframe=df_single_index)  # Should pass

        # Multi-ID in MultiIndex should pass
        multi_index = pd.MultiIndex.from_tuples(  # pyright: ignore[reportUnknownMemberType]
            [("span1", 0)], names=["span_id", "document_position"]
        )
        df_multi_index = pd.DataFrame(
            {
                "name": ["test"],
                "annotator_kind": ["HUMAN"],
                "label": ["positive"],
            },
            index=multi_index,
        )
        _validate_document_annotations_dataframe(dataframe=df_multi_index)  # Should pass

        # Missing ID level in MultiIndex should fail
        incomplete_multi_index = pd.MultiIndex.from_tuples(  # pyright: ignore[reportUnknownMemberType]
            [("span1", "extra")],
            names=["span_id", "extra_level"],  # Missing document_position
        )
        df_incomplete_index = pd.DataFrame(
            {
                "name": ["test"],
                "annotator_kind": ["HUMAN"],
                "label": ["positive"],
            },
            index=incomplete_multi_index,
        )
        with pytest.raises(ValueError, match="DataFrame must have ALL required ID columns"):
            _validate_document_annotations_dataframe(dataframe=df_incomplete_index)

    def test_multiindex_for_document_annotations(self) -> None:
        """Test using MultiIndex for document annotations with both span_id and document_position."""
        # Create DataFrame with MultiIndex containing both required ID columns
        multi_index = pd.MultiIndex.from_tuples(  # pyright: ignore[reportUnknownMemberType]
            [("span1", 0), ("span1", 1), ("span2", 0)], names=["span_id", "document_position"]
        )
        df = pd.DataFrame(
            {
                "name": ["relevance", "accuracy", "completeness"],
                "annotator_kind": ["HUMAN", "LLM", "CODE"],
                "label": ["relevant", "accurate", "complete"],
            },
            index=multi_index,
        )

        chunks = list(
            _chunk_annotations_dataframe(
                dataframe=df,
                id_config=_DOCUMENT_ID_CONFIG,
                annotation_factory=_create_document_annotation,
            )
        )

        # Check all three annotations were created correctly
        assert len(chunks) == 1  # All fit in one chunk
        annotations = chunks[0]
        assert len(annotations) == 3

        # First annotation
        assert annotations[0]["span_id"] == "span1"
        assert annotations[0]["document_position"] == 0
        assert annotations[0]["name"] == "relevance"

        # Second annotation
        assert annotations[1]["span_id"] == "span1"
        assert annotations[1]["document_position"] == 1
        assert annotations[1]["name"] == "accuracy"

        # Third annotation
        assert annotations[2]["span_id"] == "span2"
        assert annotations[2]["document_position"] == 0
        assert annotations[2]["name"] == "completeness"

    def test_span_annotation_creation(self) -> None:
        """Test creating span annotations with different parameter combinations."""
        # Minimal required parameters
        basic_annotation = _create_span_annotation(
            span_id="span1", annotation_name="sentiment", label="positive"
        )
        assert basic_annotation["span_id"] == "span1"
        assert basic_annotation["name"] == "sentiment"
        assert basic_annotation["annotator_kind"] == "HUMAN"  # default
        assert basic_annotation.get("result", {}).get("label") == "positive"

        # Full parameters with all optional fields
        full_annotation = _create_span_annotation(
            span_id="span2",
            annotation_name="quality",
            annotator_kind="LLM",
            label="high_quality",
            score=0.95,
            explanation="Well structured response",
            metadata={"model": "gpt-4", "version": "1.0"},
            identifier="eval_run_1",
        )
        assert full_annotation["span_id"] == "span2"
        assert full_annotation["name"] == "quality"
        assert full_annotation["annotator_kind"] == "LLM"
        result = full_annotation.get("result", {})
        assert result.get("label") == "high_quality"
        assert result.get("score") == 0.95
        assert result.get("explanation") == "Well structured response"
        metadata = full_annotation.get("metadata", {})
        assert metadata.get("model") == "gpt-4"
        assert full_annotation.get("identifier") == "eval_run_1"


# --- packages/phoenix-client/tests/client/utils/test_executors.py ---

def test_async_executor_runs_synchronously() -> None:
    async def dummy_fn(payload: int) -> int:
        return payload - 2

    executor = AsyncExecutor(dummy_fn, concurrency=10, max_retries=0)
    inputs = [1, 2, 3, 4, 5]
    outputs, _ = executor.run(inputs)
    assert outputs == [-1, 0, 1, 2, 3]

def test_async_executor_run_exits_early_on_error() -> None:
    async def dummy_fn(payload: int) -> int:
        if payload == 3:
            raise ValueError("test error")
        return payload - 1

    executor = AsyncExecutor(
        dummy_fn, concurrency=1, max_retries=0, exit_on_error=True, fallback_return_value=52
    )
    inputs = [1, 2, 3, 4, 5]
    outputs, statuses = executor.run(inputs)
    exceptions = [status.exceptions for status in statuses]
    status_types = [status.status for status in statuses]
    assert outputs == [0, 1, 52, 52, 52]
    assert [len(excs) if excs else 0 for excs in exceptions] == [
        0,
        0,
        1,
        0,
        0,
    ], "one exception raised, then exits"
    assert status_types == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.DID_NOT_RUN,
        ExecutionStatus.DID_NOT_RUN,
    ]
    assert all(isinstance(exc, ValueError) for exc in exceptions[2])

async def test_async_executor_can_continue_on_error() -> None:
    async def dummy_fn(payload: int) -> int:
        if payload == 3:
            raise ValueError("test error")
        return payload - 1

    executor = AsyncExecutor(
        dummy_fn, concurrency=1, max_retries=1, exit_on_error=False, fallback_return_value=52
    )
    inputs = [1, 2, 3, 4, 5]
    outputs, statuses = await executor.execute(inputs)
    exceptions = [status.exceptions for status in statuses]
    status_types = [status.status for status in statuses]
    execution_times = [status.execution_seconds for status in statuses]
    assert outputs == [0, 1, 52, 3, 4], "failed tasks use the fallback value"
    assert [len(excs) if excs else 0 for excs in exceptions] == [
        0,
        0,
        2,
        0,
        0,
    ], "two exceptions due to retries"
    assert status_types == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
    ]
    assert len(execution_times) == 5
    assert all(isinstance(runtime, float) for runtime in execution_times)
    assert all(isinstance(exc, ValueError) for exc in exceptions[2])

async def test_async_executor_marks_completed_with_retries_status() -> None:
    retry_counter = 0

    async def dummy_fn(payload: int) -> int:
        if payload == 3:
            nonlocal retry_counter
            if retry_counter < 2:
                retry_counter += 1
                raise ValueError("test error")
        return payload - 1

    executor = AsyncExecutor(
        dummy_fn, concurrency=1, max_retries=3, exit_on_error=False, fallback_return_value=52
    )
    inputs = [1, 2, 3, 4, 5]
    outputs, execution_details = await executor.execute(inputs)
    assert outputs == [0, 1, 2, 3, 4], "input 3 should only fail twice"
    assert [status.status for status in execution_details] == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED_WITH_RETRIES,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
    ]

def test_sync_executor_runs_many_tasks() -> None:
    def dummy_fn(payload: int) -> int:
        return payload

    executor = SyncExecutor(dummy_fn, max_retries=0)
    inputs = [x for x in range(1000)]
    outputs, _ = executor.run(inputs)
    assert outputs == inputs

def test_sync_executor_runs() -> None:
    def dummy_fn(payload: int) -> int:
        return payload - 2

    executor = SyncExecutor(dummy_fn, max_retries=0)
    inputs = [1, 2, 3, 4, 5]
    outputs, _ = executor.run(inputs)
    assert outputs == [-1, 0, 1, 2, 3]

def test_sync_executor_run_exits_early_on_error() -> None:
    def dummy_fn(payload: int) -> int:
        if payload == 3:
            raise ValueError("test error")
        return payload - 1

    executor = SyncExecutor(dummy_fn, exit_on_error=True, fallback_return_value=52, max_retries=0)
    inputs = [1, 2, 3, 4, 5]
    outputs, execution_details = executor.run(inputs)
    exceptions = [status.exceptions for status in execution_details]
    status_types = [status.status for status in execution_details]
    assert outputs == [0, 1, 52, 52, 52]
    assert [len(excs) if excs else 0 for excs in exceptions] == [
        0,
        0,
        1,
        0,
        0,
    ], "one exception raised, then exits"
    assert status_types == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.DID_NOT_RUN,
        ExecutionStatus.DID_NOT_RUN,
    ]
    assert all(isinstance(exc, ValueError) for exc in exceptions[2])

def test_sync_executor_can_continue_on_error() -> None:
    def dummy_fn(payload: int) -> int:
        if payload == 3:
            raise ValueError("test error")
        return payload - 1

    executor = SyncExecutor(dummy_fn, exit_on_error=False, fallback_return_value=52, max_retries=1)
    inputs = [1, 2, 3, 4, 5]
    outputs, execution_details = executor.run(inputs)
    exceptions = [status.exceptions for status in execution_details]
    status_types = [status.status for status in execution_details]
    execution_times = [status.execution_seconds for status in execution_details]
    assert outputs == [0, 1, 52, 3, 4]
    assert [len(excs) if excs else 0 for excs in exceptions] == [
        0,
        0,
        2,
        0,
        0,
    ], "two exceptions due to retries"
    assert status_types == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
    ]
    assert len(execution_times) == 5
    assert all(isinstance(runtime, float) for runtime in execution_times)
    assert all(isinstance(exc, ValueError) for exc in exceptions[2])

def test_sync_executor_marks_completed_with_retries_status() -> None:
    retry_counter = 0

    def dummy_fn(payload: int) -> int:
        if payload == 3:
            nonlocal retry_counter
            if retry_counter < 2:
                retry_counter += 1
                raise ValueError("test error")
        return payload - 1

    executor = SyncExecutor(dummy_fn, max_retries=3, exit_on_error=False, fallback_return_value=52)
    inputs = [1, 2, 3, 4, 5]
    outputs, execution_details = executor.run(inputs)
    assert outputs == [0, 1, 2, 3, 4], "input 3 should only fail twice"
    assert [status.status for status in execution_details] == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED_WITH_RETRIES,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
    ]

def test_sync_executor_sigint_handling() -> None:
    def sync_fn(x: int) -> int:
        time.sleep(0.01)
        return x

    result_length = 1000
    sigint_index = 50
    executor = SyncExecutor(
        sync_fn,
        max_retries=0,
        fallback_return_value="test",
        termination_signal=signal.SIGUSR1,  # type: ignore[attr-defined, unused-ignore]
    )
    results, _ = executor.run(InterruptingIterator(sigint_index, result_length))
    assert len(results) == result_length
    assert results.count("test") > 100, "most inputs should not have been processed"

def test_sync_executor_defaults_sigint_handling() -> None:
    def sync_fn(x: int) -> Any:
        return signal.getsignal(signal.SIGINT)

    executor = SyncExecutor(
        sync_fn,
        max_retries=0,
        fallback_return_value="test",
    )
    res, _ = executor.run(["test"])
    assert res[0] != signal.default_int_handler

def test_sync_executor_bypasses_sigint_handling_if_none() -> None:
    def sync_fn(x: int) -> Any:
        return signal.getsignal(signal.SIGINT)

    executor = SyncExecutor(
        sync_fn,
        max_retries=0,
        fallback_return_value="test",
        termination_signal=None,
    )
    res, _ = executor.run(["test"])
    assert res[0] == signal.default_int_handler

def test_executor_factory_returns_sync_in_sync_context_if_asked() -> None:
    def sync_fn(x: Any) -> Any:
        return x

    async def async_fn(x: Any) -> Any:
        return x

    def executor_in_sync_context() -> Any:
        return get_executor_on_sync_context(
            sync_fn,
            async_fn,
            run_sync=True,  # request a sync_executor
        )

    executor = executor_in_sync_context()
    assert isinstance(executor, SyncExecutor)

def test_executor_factory_returns_sync_in_threads() -> None:
    def sync_fn(x: Any) -> Any:
        return x

    async def async_fn(x: Any) -> Any:
        return x

    exception_log: queue.Queue[Exception] = queue.Queue()

    def run_test() -> None:
        try:
            executor = get_executor_on_sync_context(
                sync_fn,
                async_fn,
                run_sync=True,  # request a sync_executor
            )
            assert isinstance(executor, SyncExecutor)
            assert executor.termination_signal is None
        except Exception as e:
            exception_log.put(e)

    test_thread = threading.Thread(target=run_test)
    test_thread.start()
    test_thread.join()
    if not exception_log.empty():
        raise exception_log.get()

async def test_executor_factory_returns_sync_in_threads_even_if_async_context() -> None:
    def sync_fn(x: Any) -> Any:
        return x

    async def async_fn(x: Any) -> Any:
        return x

    exception_log: queue.Queue[Exception] = queue.Queue()

    async def run_test() -> None:
        nest_asyncio.apply()  # pyright: ignore
        try:
            executor = get_executor_on_sync_context(
                sync_fn,
                async_fn,
            )
            assert isinstance(executor, SyncExecutor)
            assert executor.termination_signal is None
        except Exception as e:
            exception_log.put(e)

    def async_task(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_test())

    loop = asyncio.new_event_loop()
    test_thread = threading.Thread(target=async_task, args=(loop,))
    test_thread.start()
    test_thread.join()

    if not exception_log.empty():
        raise exception_log.get()

def test_executor_factory_returns_async_not_in_thread_if_async_context() -> None:
    def sync_fn(x: Any) -> Any:
        return x

    async def async_fn(x: Any) -> Any:
        return x

    exception_log: queue.Queue[Exception] = queue.Queue()

    async def run_test() -> None:
        nest_asyncio.apply()  # pyright: ignore
        try:
            executor = get_executor_on_sync_context(
                sync_fn,
                async_fn,
            )
            assert isinstance(executor, AsyncExecutor)
            assert executor.termination_signal is not None
        except Exception as e:
            exception_log.put(e)

    def async_task() -> None:
        asyncio.run(run_test())

    async_task()

    if not exception_log.empty():
        raise exception_log.get()

def test_sync_executor_run_works_in_background_thread() -> None:
    def sync_fn(x: int) -> int:
        return x * 2

    outputs = []
    errors: list[Exception] = []

    def run_in_background() -> None:
        try:
            executor = SyncExecutor(sync_fn, termination_signal=signal.SIGUSR1)  # type: ignore[attr-defined, unused-ignore]
            result, _ = executor.run(range(3))
            outputs.extend(result)  # pyright: ignore[reportUnknownMemberType]
        except Exception as e:
            errors.append(e)

    test_thread = threading.Thread(target=run_in_background)
    test_thread.start()
    test_thread.join()

    assert not errors, f"run() failed in background thread: {errors}"
    assert outputs == [0, 2, 4], f"Expected [0, 2, 4], got {outputs}"

def test_async_executor_run_works_in_background_thread() -> None:
    async def async_fn(x: int) -> int:
        return x * 3

    outputs = []
    errors: list[Exception] = []

    def run_in_background() -> None:
        try:
            executor = AsyncExecutor(async_fn, termination_signal=signal.SIGUSR1)  # type: ignore[attr-defined, unused-ignore]
            result, _ = executor.run(range(3))
            outputs.extend(result)  # pyright: ignore[reportUnknownMemberType]
        except Exception as e:
            errors.append(e)

    test_thread = threading.Thread(target=run_in_background)
    test_thread.start()
    test_thread.join()

    assert not errors, f"run() failed in background thread: {errors}"
    assert outputs == [0, 3, 6], f"Expected [0, 3, 6], got {outputs}"


# --- packages/phoenix-evals/tests/phoenix/evals/test_executor.py ---

def test_async_executor_runs_synchronously():
    async def dummy_fn(payload: int) -> int:
        return payload - 2

    executor = AsyncExecutor(
        dummy_fn, concurrency=10, max_retries=0, enable_dynamic_concurrency=False
    )
    inputs = [1, 2, 3, 4, 5]
    outputs, _ = executor.run(inputs)
    assert outputs == [-1, 0, 1, 2, 3]

def test_async_executor_run_exits_early_on_error():
    async def dummy_fn(payload: int) -> int:
        if payload == 3:
            raise ValueError("test error")
        return payload - 1

    executor = AsyncExecutor(
        dummy_fn,
        concurrency=1,
        max_retries=0,
        exit_on_error=True,
        fallback_return_value=52,
        enable_dynamic_concurrency=False,
    )
    inputs = [1, 2, 3, 4, 5]
    outputs, statuses = executor.run(inputs)
    exceptions = [status.exceptions for status in statuses]
    status_types = [status.status for status in statuses]
    assert outputs == [0, 1, 52, 52, 52]
    assert [len(excs) if excs else 0 for excs in exceptions] == [
        0,
        0,
        1,
        0,
        0,
    ], "one exception raised, then exits"
    assert status_types == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.DID_NOT_RUN,
        ExecutionStatus.DID_NOT_RUN,
    ]
    assert all(isinstance(exc, ValueError) for exc in exceptions[2])

async def test_async_executor_can_continue_on_error():
    async def dummy_fn(payload: int) -> int:
        if payload == 3:
            raise ValueError("test error")
        return payload - 1

    executor = AsyncExecutor(
        dummy_fn,
        concurrency=1,
        max_retries=1,
        exit_on_error=False,
        fallback_return_value=52,
        enable_dynamic_concurrency=False,
    )
    inputs = [1, 2, 3, 4, 5]
    outputs, statuses = await executor.execute(inputs)
    exceptions = [status.exceptions for status in statuses]
    status_types = [status.status for status in statuses]
    execution_times = [status.execution_seconds for status in statuses]
    assert outputs == [0, 1, 52, 3, 4], "failed tasks use the fallback value"
    assert [len(excs) if excs else 0 for excs in exceptions] == [
        0,
        0,
        2,
        0,
        0,
    ], "two exceptions due to retries"
    assert status_types == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
    ]
    assert len(execution_times) == 5
    assert all(isinstance(runtime, float) for runtime in execution_times)
    assert all(isinstance(exc, ValueError) for exc in exceptions[2])

async def test_async_executor_marks_completed_with_retries_status():
    retry_counter = 0

    async def dummy_fn(payload: int) -> int:
        if payload == 3:
            nonlocal retry_counter
            if retry_counter < 2:
                retry_counter += 1
                raise ValueError("test error")
        return payload - 1

    executor = AsyncExecutor(
        dummy_fn,
        concurrency=1,
        max_retries=3,
        exit_on_error=False,
        fallback_return_value=52,
        enable_dynamic_concurrency=False,
    )
    inputs = [1, 2, 3, 4, 5]
    outputs, execution_details = await executor.execute(inputs)
    assert outputs == [0, 1, 2, 3, 4], "input 3 should only fail twice"
    assert [status.status for status in execution_details] == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED_WITH_RETRIES,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
    ]

def test_sync_executor_runs_many_tasks():
    def dummy_fn(payload: int) -> int:
        return payload

    executor = SyncExecutor(dummy_fn, max_retries=0)
    inputs = [x for x in range(1000)]
    outputs, _ = executor.run(inputs)
    assert outputs == inputs

def test_sync_executor_runs():
    def dummy_fn(payload: int) -> int:
        return payload - 2

    executor = SyncExecutor(dummy_fn, max_retries=0)
    inputs = [1, 2, 3, 4, 5]
    outputs, _ = executor.run(inputs)
    assert outputs == [-1, 0, 1, 2, 3]

def test_sync_executor_run_exits_early_on_error():
    def dummy_fn(payload: int) -> int:
        if payload == 3:
            raise ValueError("test error")
        return payload - 1

    executor = SyncExecutor(dummy_fn, exit_on_error=True, fallback_return_value=52, max_retries=0)
    inputs = [1, 2, 3, 4, 5]
    outputs, execution_details = executor.run(inputs)
    exceptions = [status.exceptions for status in execution_details]
    status_types = [status.status for status in execution_details]
    assert outputs == [0, 1, 52, 52, 52]
    assert [len(excs) if excs else 0 for excs in exceptions] == [
        0,
        0,
        1,
        0,
        0,
    ], "one exception raised, then exits"
    assert status_types == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.DID_NOT_RUN,
        ExecutionStatus.DID_NOT_RUN,
    ]
    assert all(isinstance(exc, ValueError) for exc in exceptions[2])

def test_sync_executor_can_continue_on_error():
    def dummy_fn(payload: int) -> int:
        if payload == 3:
            raise ValueError("test error")
        return payload - 1

    executor = SyncExecutor(dummy_fn, exit_on_error=False, fallback_return_value=52, max_retries=1)
    inputs = [1, 2, 3, 4, 5]
    outputs, execution_details = executor.run(inputs)
    exceptions = [status.exceptions for status in execution_details]
    status_types = [status.status for status in execution_details]
    execution_times = [status.execution_seconds for status in execution_details]
    assert outputs == [0, 1, 52, 3, 4]
    assert [len(excs) if excs else 0 for excs in exceptions] == [
        0,
        0,
        2,
        0,
        0,
    ], "two exceptions due to retries"
    assert status_types == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
    ]
    assert len(execution_times) == 5
    assert all(isinstance(runtime, float) for runtime in execution_times)
    assert all(isinstance(exc, ValueError) for exc in exceptions[2])

def test_sync_executor_marks_completed_with_retries_status():
    retry_counter = 0

    def dummy_fn(payload: int) -> int:
        if payload == 3:
            nonlocal retry_counter
            if retry_counter < 2:
                retry_counter += 1
                raise ValueError("test error")
        return payload - 1

    executor = SyncExecutor(dummy_fn, max_retries=3, exit_on_error=False, fallback_return_value=52)
    inputs = [1, 2, 3, 4, 5]
    outputs, execution_details = executor.run(inputs)
    assert outputs == [0, 1, 2, 3, 4], "input 3 should only fail twice"
    assert [status.status for status in execution_details] == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED_WITH_RETRIES,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
    ]

def test_sync_executor_sigint_handling():
    class InterruptingIterator:
        def __init__(self, interruption_index: int, max_elements: int):
            self.interruption_index = interruption_index
            self.max_elements = max_elements
            self.current = 0

        def __len__(self):
            return self.max_elements

        def __iter__(self):
            return self

        def __next__(self):
            if self.current < self.max_elements:
                if self.current == self.interruption_index:
                    # Trigger interruption signal
                    os.kill(os.getpid(), signal.SIGUSR1)
                    time.sleep(0.1)

                res = self.current
                self.current += 1
                return res
            else:
                raise StopIteration

    def sync_fn(x):
        time.sleep(0.01)
        return x

    result_length = 1000
    sigint_index = 50
    executor = SyncExecutor(
        sync_fn,
        max_retries=0,
        fallback_return_value="test",
        termination_signal=signal.SIGUSR1,
    )
    results, _ = executor.run(InterruptingIterator(sigint_index, result_length))
    assert len(results) == result_length
    assert results.count("test") > 100, "most inputs should not have been processed"

def test_sync_executor_defaults_sigint_handling():
    def sync_fn(x):
        return signal.getsignal(signal.SIGINT)

    executor = SyncExecutor(
        sync_fn,
        max_retries=0,
        fallback_return_value="test",
    )
    res, _ = executor.run(["test"])
    assert res[0] != signal.default_int_handler

def test_sync_executor_bypasses_sigint_handling_if_none():
    def sync_fn(x):
        return signal.getsignal(signal.SIGINT)

    executor = SyncExecutor(
        sync_fn,
        max_retries=0,
        fallback_return_value="test",
        termination_signal=None,
    )
    res, _ = executor.run(["test"])
    assert res[0] == signal.default_int_handler

def test_executor_factory_returns_sync_in_sync_context_if_asked():
    def sync_fn():
        pass

    async def async_fn():
        pass

    def executor_in_sync_context():
        return get_executor_on_sync_context(
            sync_fn,
            async_fn,
            run_sync=True,  # request a sync_executor
        )

    executor = executor_in_sync_context()
    assert isinstance(executor, SyncExecutor)

def test_executor_factory_returns_sync_in_threads():
    def sync_fn():
        pass

    async def async_fn():
        pass

    exception_log = queue.Queue()

    def run_test():
        try:
            executor = get_executor_on_sync_context(
                sync_fn,
                async_fn,
                run_sync=True,  # request a sync_executor
            )
            assert isinstance(executor, SyncExecutor)
            assert executor.termination_signal is None
        except Exception as e:
            exception_log.put(e)

    test_thread = threading.Thread(target=run_test)
    test_thread.start()
    test_thread.join()
    if not exception_log.empty():
        raise exception_log.get()

async def test_executor_factory_returns_sync_in_threads_even_if_async_context():
    def sync_fn():
        pass

    async def async_fn():
        pass

    exception_log = queue.Queue()

    async def run_test():
        nest_asyncio.apply()
        try:
            executor = get_executor_on_sync_context(
                sync_fn,
                async_fn,
            )
            assert isinstance(executor, SyncExecutor)
            assert executor.termination_signal is None
        except Exception as e:
            exception_log.put(e)

    def async_task(loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_test())

    loop = asyncio.new_event_loop()
    test_thread = threading.Thread(target=async_task, args=(loop,))
    test_thread.start()
    test_thread.join()

    if not exception_log.empty():
        raise exception_log.get()

def test_executor_factory_returns_async_not_in_thread_if_async_context():
    def sync_fn():
        pass

    async def async_fn():
        pass

    exception_log = queue.Queue()

    async def run_test():
        nest_asyncio.apply()
        try:
            executor = get_executor_on_sync_context(
                sync_fn,
                async_fn,
            )
            assert isinstance(executor, AsyncExecutor)
            assert executor.termination_signal is not None
        except Exception as e:
            exception_log.put(e)

    def async_task():
        asyncio.run(run_test())

    async_task()

    if not exception_log.empty():
        raise exception_log.get()


# --- packages/phoenix-evals/tests/phoenix/evals/test_utils.py ---

    def test_output_column_structure(self, score_data, score_name, expected_columns):
        """Test that output has correct column structure."""
        df = pd.DataFrame({"span_id": ["span_1"], f"{score_name}_score": [json.dumps(score_data)]})
        result = to_annotation_dataframe(dataframe=df, score_names=[score_name])
        assert list(result.columns) == expected_columns


# --- packages/phoenix-evals/tests/phoenix/evals/llm/test_wrapper.py ---

    def test_string_labels_generate_enum_schema(self, labels: List[str], expected_enum: List[str]):
        """Test that string labels generate proper enum schema."""
        schema = generate_classification_schema(labels)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        label_schema = schema["properties"]["label"]
        assert label_schema["type"] == "string"
        assert "enum" in label_schema
        assert label_schema["enum"] == expected_enum
        assert "oneOf" not in label_schema

    def test_dict_labels_generate_one_of_schema(
        self, labels: Dict[str, str], expected_one_of: List[Dict[str, str]]
    ):
        """Test that dict labels generate proper oneOf schema."""
        schema = generate_classification_schema(labels)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        label_schema = schema["properties"]["label"]
        assert label_schema["type"] == "string"
        assert "oneOf" in label_schema
        assert label_schema["oneOf"] == expected_one_of
        assert "enum" not in label_schema

    def test_description_is_added_to_label_schema(
        self, description: str, expected_description: str
    ):
        """Test that description is properly added to the label schema."""
        labels = ["yes", "no"]
        schema = generate_classification_schema(labels, description=description)

        label_schema = schema["properties"]["label"]
        if description:
            assert "description" in label_schema
            assert label_schema["description"] == expected_description
        else:
            assert "description" not in label_schema

    def test_explanation_field_handling(
        self,
        include_explanation: bool,
        expected_properties: List[str],
        expected_required: List[str],
    ):
        """
        Test that explanation field is properly handled based on include_explanation parameter.
        """
        labels = ["yes", "no"]
        schema = generate_classification_schema(labels, include_explanation=include_explanation)

        properties = schema["properties"]
        required = schema["required"]

        assert list(properties.keys()) == expected_properties
        assert required == expected_required

        if include_explanation:
            explanation_schema = properties["explanation"]
            assert explanation_schema["type"] == "string"
            assert explanation_schema["description"] == "A brief explanation of your reasoning."

    def test_explanation_field_order(self):
        """Test that explanation field appears before label field in properties."""
        labels = ["yes", "no"]
        schema = generate_classification_schema(labels, include_explanation=True)

        properties = list(schema["properties"].keys())
        assert properties == ["explanation", "label"]

    def test_required_fields_order(self):
        """Test that required fields are in the correct order."""
        labels = ["yes", "no"]
        schema = generate_classification_schema(labels, include_explanation=True)

        required = schema["required"]
        assert required == ["explanation", "label"]

    def test_complete_schema_generation(
        self,
        labels: Union[List[str], Dict[str, str]],
        description: str,
        include_explanation: bool,
        expected_schema: Dict[str, Any],
    ):
        """Test complete schema generation with various combinations of parameters."""
        schema = generate_classification_schema(labels, include_explanation, description)
        assert schema == expected_schema

    def test_invalid_inputs_raise_errors(self, invalid_labels: Any, expected_error: str):
        """Test that invalid inputs raise appropriate ValueError exceptions."""
        with pytest.raises(ValueError, match=expected_error):
            generate_classification_schema(invalid_labels)

    def test_dict_labels_with_optional_description(self):
        """Test that dict labels with optional description field work correctly."""
        labels = {
            "yes": "Positive response",
            "no": "",  # Empty description
            "maybe": "Uncertain response",
        }

        schema = generate_classification_schema(labels)
        label_schema = schema["properties"]["label"]
        one_of = label_schema["oneOf"]

        assert len(one_of) == 3
        assert one_of[0] == {"const": "yes", "description": "Positive response"}
        assert one_of[1] == {"const": "no"}
        assert one_of[2] == {"const": "maybe", "description": "Uncertain response"}

    def test_default_parameters(self):
        """Test that default parameters work correctly."""
        labels = ["yes", "no"]
        schema = generate_classification_schema(labels)

        # Should include explanation by default
        assert "explanation" in schema["properties"]
        assert "explanation" in schema["required"]

        # Should not have description by default
        label_schema = schema["properties"]["label"]
        assert "description" not in label_schema

    def test_schema_structure_consistency(self):
        """Test that the generated schema has consistent structure."""
        labels = ["yes", "no"]
        schema = generate_classification_schema(labels)

        # Check top-level structure
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Check properties structure
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert "label" in properties

        # Check label schema structure
        label_schema = properties["label"]
        assert isinstance(label_schema, dict)
        assert label_schema["type"] == "string"

        # Check required structure
        required = schema["required"]
        assert isinstance(required, list)
        assert "label" in required


# --- tests/integration/auth/test_oidc.py ---

    async def test_sign_in(
        self,
        allow_sign_up: bool,
        _oidc_server: _OIDCServer,
        _app: _AppInfo,
    ) -> None:
        """Test OIDC sign-in with allow_sign_up enabled/disabled.

        When allow_sign_up=True: New users can sign in and are automatically created.
        When allow_sign_up=False: New users are denied until admin creates their account,
        but can sign in after the admin creates them (verifies auth code reuse after denial).
        """
        path_suffix = "" if allow_sign_up else "_no_sign_up"

        # Set persistent user for potential retry scenario
        test_user_id = f"test_user_sign_in_{token_hex(8)}"
        test_email = f"user_{token_hex(8)}@example.com"
        num_logins = 2 if not allow_sign_up else 1
        _oidc_server.set_user(test_user_id, test_email, num_logins=num_logins)

        # Login 1: Initial attempt
        email1, cookies1, callback_url1 = await _complete_flow(_app, _oidc_server, path_suffix)
        assert email1 == sanitize_email(test_email)

        if not allow_sign_up:
            # Verify access denied for new user
            await _verify_user_denied(_app, email1, cookies1, callback_url1)

            # Admin creates the user without password
            case_insensitive_email = _randomize_casing(email1)
            expected_role: UserRoleInput = choice(list(UserRoleInput))
            _create_user(
                _app,
                _app.admin_secret,
                role=expected_role,
                profile=_Profile(case_insensitive_email, "", token_hex(8)),
                local=False,
            )

            # Login 2: Retry after admin created account - should succeed
            email2, cookies2, callback_url2 = await _complete_flow(_app, _oidc_server, path_suffix)
            assert email2 == email1
            await _verify_user_granted_with_role(
                _app, email2, cookies2, callback_url2, expected_role, cleanup=True
            )
        else:
            # Verify auto-creation with VIEWER role
            expected_role = UserRoleInput.VIEWER
            await _verify_user_granted_with_role(
                _app, email1, cookies1, callback_url1, expected_role, cleanup=True
            )

    async def test_sign_in_conflict_for_local_user_with_password(
        self,
        allow_sign_up: bool,
        _oidc_server: _OIDCServer,
        _app: _AppInfo,
    ) -> None:
        """Test that local users with passwords cannot sign in via OIDC.

        Security requirement: Users with passwords (local accounts) are prevented
        from authenticating via OIDC to avoid credential confusion attacks. This
        ensures users cannot bypass password requirements by using SSO for accounts
        that were set up with passwords.
        """
        path_suffix = "" if allow_sign_up else "_no_sign_up"

        # Start flow
        email, cookies, callback_url = await _complete_flow(_app, _oidc_server, path_suffix)

        # Verify user doesn't exist
        await _verify_user_does_not_exist(_app, email)

        # Create user with password
        _create_user(
            _app,
            _app.admin_secret,
            role=UserRoleInput.VIEWER,
            profile=_Profile(email, token_hex(8), token_hex(8)),
            local=True,
        )

        # Verify OIDC sign-in is rejected
        response = _httpx_client(_app, cookies=cookies).get(callback_url)
        _verify_access_denied(
            response.status_code,
            response.cookies.get("phoenix-access-token"),
            response.headers["location"],
        )
        # User SHOULD exist (we just created them), but OIDC sign-in should be rejected
        await _verify_user_exists_with_role(_app, email, UserRoleInput.VIEWER, cleanup=True)

    async def test_oidc_with_groups(
        self,
        access_granted: bool,
        _oidc_server_standard_with_groups: _OIDCServer,
        _app: _AppInfo,
    ) -> None:
        """Test OIDC with group-based access control.

        When group-based access control is configured, Phoenix can restrict access
        to users who are members of specific IDP groups. This test verifies both
        successful authentication (user in allowed group) and denial (user not in
        allowed group).
        """
        path_suffix = "_granted" if access_granted else "_denied"
        email, cookies, callback_url = await _complete_flow(
            _app, _oidc_server_standard_with_groups, path_suffix
        )

        if access_granted:
            await _verify_user_granted_with_role(
                _app, email, cookies, callback_url, UserRoleInput.VIEWER
            )
        else:
            await _verify_user_denied(_app, email, cookies, callback_url)

    async def test_pkce_confidential_client_flow(
        self,
        _oidc_server_pkce_confidential: _OIDCServer,
        _app: _AppInfo,
    ) -> None:
        """Test PKCE flow with confidential client (has client_secret)."""
        email, cookies, callback_url = await _complete_flow(_app, _oidc_server_pkce_confidential)

        # Verify PKCE cookie set
        assert "phoenix-oauth2-code-verifier" in cookies

        # Exchange code
        response = _httpx_client(_app, cookies=cookies).get(callback_url)
        _verify_tokens_issued(
            response.status_code,
            response.cookies.get("phoenix-access-token"),
            response.cookies.get("phoenix-refresh-token"),
        )

        # Verify cookies cleaned
        set_cookie_headers = response.headers.get_list("set-cookie")
        _verify_sensitive_cookies_cleaned(set_cookie_headers)
        _verify_pkce_cookie_cleaned(set_cookie_headers)

        await _verify_user_exists_with_role(_app, email, UserRoleInput.VIEWER)

    async def test_pkce_with_groups(
        self,
        access_granted: bool,
        _oidc_server_pkce_with_groups: _OIDCServer,
        _app: _AppInfo,
    ) -> None:
        """Test PKCE flow combined with group-based access control.

        Verifies that group-based access restrictions work correctly with PKCE flows.
        Users must both satisfy PKCE requirements AND be in an allowed group.
        """
        path_suffix = "_granted" if access_granted else "_denied"
        email, cookies, callback_url = await _complete_flow(
            _app, _oidc_server_pkce_with_groups, path_suffix
        )

        if access_granted:
            await _verify_user_granted_with_role(
                _app, email, cookies, callback_url, UserRoleInput.VIEWER
            )
        else:
            await _verify_user_denied(_app, email, cookies, callback_url)

    async def test_invalid_role_defaults_to_viewer_non_strict(
        self, _oidc_server_with_invalid_role: _OIDCServer, _app: _AppInfo
    ) -> None:
        """Test invalid role defaults to VIEWER in non-strict mode.

        When the IDP provides an unrecognized role and strict mode is disabled,
        Phoenix should default to VIEWER (least privilege) rather than denying access.
        This allows graceful degradation when IDP roles don't match Phoenix's configuration.
        """
        email, cookies, callback_url = await _complete_flow(
            _app, _oidc_server_with_invalid_role, "_invalid"
        )
        await _verify_user_granted_with_role(
            _app, email, cookies, callback_url, UserRoleInput.VIEWER
        )

    async def test_invalid_role_denies_access_strict_mode(
        self, _oidc_server_with_invalid_role: _OIDCServer, _app: _AppInfo
    ) -> None:
        """Test invalid role denies access in strict mode.

        When the IDP provides an unrecognized role and strict mode is enabled,
        Phoenix should deny access entirely. This enforces explicit role mapping
        and prevents users with unmapped roles from accessing the system.
        """
        email, cookies, callback_url = await _complete_flow(
            _app, _oidc_server_with_invalid_role, "_strict"
        )
        await _verify_user_denied(_app, email, cookies, callback_url)

    async def test_missing_role_defaults_to_viewer(
        self, _oidc_server_without_role: _OIDCServer, _app: _AppInfo
    ) -> None:
        """Test missing role defaults to VIEWER when role mapping is not configured.

        When no role claim is provided by the IDP (role mapping is not configured),
        new users should be assigned the default VIEWER role. This is the safest
        default providing minimum privileges.
        """
        email, cookies, callback_url = await _complete_flow(
            _app, _oidc_server_without_role, "_default"
        )
        await _verify_user_granted_with_role(
            _app, email, cookies, callback_url, UserRoleInput.VIEWER
        )

    async def test_system_role_cannot_be_assigned_via_oidc(
        self, _oidc_server_with_role_system: _OIDCServer, _app: _AppInfo
    ) -> None:
        """Test SYSTEM role from IDP defaults to VIEWER.

        The SYSTEM role is reserved for internal use and should never be assigned
        via OIDC. If an IDP attempts to assign the SYSTEM role, Phoenix should
        default to VIEWER to prevent privilege escalation.
        """
        email, cookies, callback_url = await _complete_flow(
            _app, _oidc_server_with_role_system, "_system"
        )
        await _verify_user_granted_with_role(
            _app, email, cookies, callback_url, UserRoleInput.VIEWER
        )


# --- tests/integration/client/test_experiments.py ---

    async def test_run_experiment_basic(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_experiment_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[
                    {"question": "What is 2+2?"},
                    {"question": "What is the capital of France?"},
                    {"question": "Who wrote Python?"},
                ],
                outputs=[
                    {"answer": "4"},
                    {"answer": "Paris"},
                    {"answer": "Guido van Rossum"},
                ],
                metadata=[
                    {"category": "math"},
                    {"category": "geography"},
                    {"category": "programming"},
                ],
            )
        )

        def simple_task(input: Dict[str, Any]) -> str:
            question = input.get("question", "")
            if "2+2" in question:
                return "The answer is 4"
            elif "capital" in question:
                return "The capital is Paris"
            else:
                return "I don't know"

        result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=simple_task,
                experiment_name=f"test_experiment_{token_hex(4)}",
                experiment_description="A simple test experiment",
                print_summary=False,
            )
        )

        assert "experiment_id" in result
        assert "dataset_id" in result
        assert "task_runs" in result
        assert "evaluation_runs" in result
        assert result["dataset_id"] == dataset.id
        assert len(result["task_runs"]) == 3
        assert len(result["evaluation_runs"]) == 0

    async def test_run_experiment_with_evaluators(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_experiment_eval_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[
                    {"text": "Hello world"},
                    {"text": "Python is great"},
                ],
                outputs=[
                    {"expected": "greeting"},
                    {"expected": "programming"},
                ],
            )
        )

        def classification_task(input: Dict[str, Any]) -> str:
            text = input.get("text", "")
            if "Hello" in text:
                return "greeting"
            elif "Python" in text:
                return "programming"
            else:
                return "unknown"

        def accuracy_evaluator(output: str, expected: Dict[str, Any]) -> float:
            return 1.0 if output == expected.get("expected") else 0.0

        def length_evaluator(output: str) -> Dict[str, Any]:
            return {"score": len(output) / 10.0, "label": "length_score"}

        result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=classification_task,
                evaluators=[accuracy_evaluator, length_evaluator],
                experiment_name=f"test_eval_experiment_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result["task_runs"]) == 2
        assert len(result["evaluation_runs"]) > 0

    async def test_run_experiment_with_different_task_signatures(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_signatures_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[{"prompt": "Test prompt"}],
                outputs=[{"response": "Test response"}],
                metadata=[{"source": "test"}],
            )
        )

        def task_with_input_only(input: Dict[str, Any]) -> str:
            return f"Processed: {input.get('prompt', '')}"

        def task_with_multiple_params(
            input: Dict[str, Any], expected: Dict[str, Any], metadata: Dict[str, Any]
        ) -> str:
            return f"Input: {input}, Expected: {expected}, Meta: {metadata}"

        result1 = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=task_with_input_only,
                experiment_name=f"test_input_only_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result1["task_runs"]) == 1

        result2 = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=task_with_multiple_params,
                experiment_name=f"test_multi_params_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result2["task_runs"]) == 1

    async def test_run_experiment_dry_run(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_dry_run_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[
                    {"text": "Sample 1"},
                    {"text": "Sample 2"},
                    {"text": "Sample 3"},
                ],
                outputs=[
                    {"result": "Result 1"},
                    {"result": "Result 2"},
                    {"result": "Result 3"},
                ],
            )
        )

        def simple_task(input: Dict[str, Any]) -> str:
            return f"Processed: {input.get('text', '')}"

        result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=simple_task,
                experiment_name="dry_run_test",
                dry_run=True,
                print_summary=False,
            )
        )

        assert result["experiment_id"] == "DRY_RUN"
        assert len(result["task_runs"]) == 1

        result_sized = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=simple_task,
                experiment_name="dry_run_sized_test",
                dry_run=2,
                print_summary=False,
            )
        )

        assert len(result_sized["task_runs"]) == 2

    async def test_run_experiment_with_metadata(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_metadata_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[{"question": "Test question"}],
                outputs=[{"answer": "Test answer"}],
            )
        )

        def simple_task(input: Dict[str, Any]) -> str:
            return "Test response"

        experiment_metadata = {
            "version": "1.0",
            "model": "test-model",
            "temperature": 0.7,
        }

        result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=simple_task,
                experiment_name=f"test_with_metadata_{token_hex(4)}",
                experiment_description="Experiment with metadata",
                experiment_metadata=experiment_metadata,
                print_summary=False,
            )
        )

        assert len(result["task_runs"]) == 1

    async def test_task_and_evaluator_parameter_isolation(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient
        from phoenix.client.resources.experiments.types import ExampleProxy

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_copying_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[{"text": "Hello"}],
                outputs=[{"expected": "greeting"}],
                metadata=[{"category": "test"}],
            )
        )

        def mutating_task(
            input: Dict[str, Any],
            expected: Dict[str, Any],
            metadata: Dict[str, Any],
            example: Any,
        ) -> str:
            input["added_by_task"] = True
            expected["added_by_task"] = True
            metadata["added_by_task"] = True
            return "ok"

        observations: Dict[str, Any] = {
            "task_example_is_proxy": None,
            "task_example_has_id": None,
            "task_example_input_text": None,
            "ev1_input_had_task": None,
            "ev1_expected_had_task": None,
            "ev1_metadata_had_task": None,
            "ev1_example_is_proxy": None,
            "ev1_example_has_id": None,
            "ev2_input_had_ev1": None,
            "ev2_expected_had_ev1": None,
            "ev2_metadata_had_ev1": None,
            "ev2_example_is_proxy": None,
            "ev2_example_has_id": None,
        }

        def evaluator_one(
            input: Dict[str, Any],
            expected: Dict[str, Any],
            metadata: Dict[str, Any],
            example: Any,
        ) -> float:
            observations["ev1_input_had_task"] = "added_by_task" in input
            observations["ev1_expected_had_task"] = "added_by_task" in expected
            observations["ev1_metadata_had_task"] = "added_by_task" in metadata
            observations["ev1_example_is_proxy"] = isinstance(example, ExampleProxy)
            observations["ev1_example_has_id"] = bool(example.get("id"))
            input["added_by_ev1"] = True
            expected["added_by_ev1"] = True
            metadata["added_by_ev1"] = True
            return 1.0

        def evaluator_two(
            input: Dict[str, Any],
            expected: Dict[str, Any],
            metadata: Dict[str, Any],
            example: Any,
        ) -> float:
            observations["ev2_input_had_ev1"] = "added_by_ev1" in input
            observations["ev2_expected_had_ev1"] = "added_by_ev1" in expected
            observations["ev2_metadata_had_ev1"] = "added_by_ev1" in metadata
            observations["ev2_example_is_proxy"] = isinstance(example, ExampleProxy)
            observations["ev2_example_has_id"] = bool(example.get("id"))
            return 1.0

        result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=mutating_task,
                evaluators=[evaluator_one, evaluator_two],
                experiment_name=f"test_copying_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result["task_runs"]) == 1
        assert len(result["evaluation_runs"]) == 2

        assert observations["ev1_input_had_task"] is False
        assert observations["ev1_expected_had_task"] is False
        assert observations["ev1_metadata_had_task"] is False

        assert observations["ev2_input_had_ev1"] is False
        assert observations["ev2_expected_had_ev1"] is False
        assert observations["ev2_metadata_had_ev1"] is False

        assert observations["ev1_example_is_proxy"] is True
        assert observations["ev1_example_has_id"] is True

        assert observations["ev2_example_is_proxy"] is True
        assert observations["ev2_example_has_id"] is True

        original_example = next(iter(dataset.examples))
        assert "added_by_task" not in original_example["input"]
        assert "added_by_ev1" not in original_example["input"]

    async def test_example_proxy_properties(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient
        from phoenix.client.resources.experiments.types import ExampleProxy

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_example_proxy_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[{"question": "What is 2+2?"}],
                outputs=[{"answer": "4"}],
                metadata=[{"difficulty": "easy"}],
            )
        )

        observations: Dict[str, Any] = {}

        def example_inspector(example: Any) -> str:
            observations["is_example_proxy"] = isinstance(example, ExampleProxy)
            observations["has_id_property"] = hasattr(example, "id")
            observations["has_input_property"] = hasattr(example, "input")
            observations["has_output_property"] = hasattr(example, "output")
            observations["has_metadata_property"] = hasattr(example, "metadata")
            observations["has_updated_at_property"] = hasattr(example, "updated_at")

            observations["id_value"] = example.id if hasattr(example, "id") else None
            observations["input_value"] = dict(example.input) if hasattr(example, "input") else None
            observations["output_value"] = (
                dict(example.output) if hasattr(example, "output") else None
            )
            observations["metadata_value"] = (
                dict(example.metadata) if hasattr(example, "metadata") else None
            )

            observations["dict_access_id"] = example.get("id")
            observations["dict_access_input"] = example.get("input")

            observations["supports_iteration"] = (
                list(example.keys()) if hasattr(example, "keys") else None
            )

            return "ok"

        result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=example_inspector,
                experiment_name=f"test_proxy_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result["task_runs"]) == 1

        assert observations["is_example_proxy"] is True
        assert observations["has_id_property"] is True
        assert observations["has_input_property"] is True
        assert observations["has_output_property"] is True
        assert observations["has_metadata_property"] is True
        assert observations["has_updated_at_property"] is True

        assert observations["id_value"] is not None
        assert observations["input_value"] == {"question": "What is 2+2?"}
        assert observations["output_value"] == {"answer": "4"}
        assert observations["metadata_value"] == {"difficulty": "easy"}

        assert observations["dict_access_id"] is not None
        assert observations["dict_access_input"] == {"question": "What is 2+2?"}

        assert observations["supports_iteration"] is not None
        assert "id" in observations["supports_iteration"]
        assert "input" in observations["supports_iteration"]

    async def test_run_experiment_evaluator_types(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_eval_types_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[{"text": "Hello world"}],
                outputs=[{"expected": "greeting"}],
            )
        )

        def simple_task(input: Dict[str, Any]) -> str:
            return "greeting"

        def bool_evaluator(output: str) -> bool:
            return output == "greeting"

        def float_evaluator(output: str) -> float:
            return 0.95

        def tuple_evaluator(output: str) -> tuple[float, str, str]:
            return (1.0, "correct", "The output matches expectation")

        def dict_evaluator(output: str) -> Dict[str, Any]:
            return {"score": 0.8, "label": "good"}

        result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=simple_task,
                evaluators={
                    "bool_eval": bool_evaluator,
                    "float_eval": float_evaluator,
                    "tuple_eval": tuple_evaluator,
                    "dict_eval": dict_evaluator,
                },
                experiment_name=f"test_eval_types_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result["task_runs"]) == 1
        assert len(result["evaluation_runs"]) > 0

    async def test_run_async_task(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        if not is_async:
            pytest.skip("Async tasks only supported with AsyncClient")

        api_key = _app.admin_secret

        from phoenix.client import AsyncClient

        unique_name = f"test_async_task_{token_hex(4)}"

        dataset: Dataset = await _await_or_return(
            AsyncClient(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[{"text": "Async test"}],
                outputs=[{"expected": "async_result"}],
            )
        )

        async def async_task(input: Dict[str, Any]) -> str:
            await asyncio.sleep(0.1)
            return f"async_processed_{input.get('text', '')}"

        result = await AsyncClient(
            base_url=_app.base_url, api_key=api_key
        ).experiments.run_experiment(
            dataset=dataset,
            task=async_task,
            experiment_name=f"test_async_{token_hex(4)}",
            print_summary=False,
        )

        assert len(result["task_runs"]) == 1
        assert "async_processed_" in result["task_runs"][0]["output"]

    async def test_error_handling(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_error_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[{"text": "test"}],
                outputs=[{"expected": "result"}],
            )
        )

        def failing_task(input: Dict[str, Any]) -> str:
            raise ValueError("Task failed intentionally")

        result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=failing_task,
                experiment_name=f"test_error_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result["task_runs"]) == 1
        assert "error" in result["task_runs"][0] or result["task_runs"][0]["output"] is None

    async def test_experiment_with_empty_dataset(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_empty_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[{"placeholder": "temp"}],
                outputs=[{"placeholder": "temp"}],
            )
        )

        original_dataset_id = dataset.id
        original_version_id = dataset.version_id

        dataset._examples_data = v1.ListDatasetExamplesData(
            dataset_id=original_dataset_id,
            version_id=original_version_id,
            filtered_splits=[],
            examples=[],
        )

        def simple_task(input: Dict[str, Any]) -> str:
            return "test"

        with pytest.raises(ValueError, match="Dataset has no examples"):
            await _await_or_return(
                Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                    dataset=dataset,  # pyright: ignore[reportArgumentType]
                    task=simple_task,
                    experiment_name="test_empty",
                    print_summary=False,
                )
            )

    async def test_evaluator_dynamic_parameter_binding(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_eval_params_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[
                    {"text": "What is 2+2?", "context": "math"},
                    {"text": "What is the capital of France?", "context": "geography"},
                ],
                outputs=[
                    {"answer": "4", "category": "arithmetic"},
                    {"answer": "Paris", "category": "location"},
                ],
                metadata=[
                    {"difficulty": "easy", "topic": "math"},
                    {"difficulty": "medium", "topic": "geography"},
                ],
            )
        )

        def question_answering_task(input: Dict[str, Any]) -> str:
            question = input.get("text", "")
            if "2+2" in question:
                return "The answer is 4"
            elif "capital" in question:
                return "The answer is Paris"
            else:
                return "I don't know"

        def output_only_evaluator(output: str) -> float:
            return 1.0 if "answer" in output.lower() else 0.0

        def accuracy_evaluator(output: str, expected: Dict[str, Any]) -> float:
            expected_answer = expected.get("answer", "")
            return 1.0 if expected_answer in output else 0.0

        def comprehensive_evaluator(
            input: Dict[str, Any],
            output: str,
            expected: Dict[str, Any],
            reference: Dict[str, Any],
            metadata: Dict[str, Any],
        ) -> Dict[str, Any]:
            has_input = bool(input.get("text"))
            has_output = bool(output)
            has_expected = bool(expected.get("answer"))
            has_reference = bool(reference.get("answer"))
            has_metadata = bool(metadata.get("difficulty"))

            reference_matches_expected = reference == expected

            score = (
                1.0
                if all(
                    [
                        has_input,
                        has_output,
                        has_expected,
                        has_reference,
                        has_metadata,
                        reference_matches_expected,
                    ]
                )
                else 0.0
            )

            return {
                "score": score,
                "label": "comprehensive_check",
                "explanation": (
                    f"Input: {has_input}, Output: {has_output}, Expected: {has_expected}, "
                    f"Reference: {has_reference}, Metadata: {has_metadata}, "
                    f"Reference==Expected: {reference_matches_expected}"
                ),
            }

        def reference_evaluator(output: str, reference: Dict[str, Any]) -> float:
            reference_answer = reference.get("answer", "")
            return 1.0 if reference_answer in output else 0.0

        def metadata_evaluator(output: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
            difficulty = metadata.get("difficulty", "unknown")
            topic = metadata.get("topic", "unknown")

            return {
                "score": 0.8 if difficulty == "easy" else 0.6,
                "label": f"{difficulty}_{topic}",
                "explanation": f"Difficulty: {difficulty}, Topic: {topic}",
            }

        def example_evaluator(example: Dict[str, Any]) -> Dict[str, Any]:
            has_id = bool(example.get("id"))
            has_input = isinstance(example.get("input"), dict)
            return {
                "score": 1.0 if has_id and has_input else 0.0,
                "label": "has_example",
            }

        result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=question_answering_task,
                evaluators={
                    "output_only": output_only_evaluator,
                    "relevance": accuracy_evaluator,
                    "comprehensive": comprehensive_evaluator,
                    "reference": reference_evaluator,
                    "metadata": metadata_evaluator,
                    "example": example_evaluator,
                },
                experiment_name=f"test_param_binding_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result["task_runs"]) == 2
        assert len(result["evaluation_runs"]) == 12  # 2 examples * 6 evaluators

        comprehensive_evals = [
            eval_run for eval_run in result["evaluation_runs"] if eval_run.name == "comprehensive"
        ]
        assert len(comprehensive_evals) == 2

        for eval_run in comprehensive_evals:
            assert eval_run.result is not None
            assert isinstance(eval_run.result, dict)
            assert eval_run.result.get("score") == 1.0
            assert "comprehensive_check" in (eval_run.result.get("label") or "")

        reference_evals = [
            eval_run for eval_run in result["evaluation_runs"] if eval_run.name == "reference"
        ]
        assert len(reference_evals) == 2

        for eval_run in reference_evals:
            assert eval_run.result is not None
            assert isinstance(eval_run.result, dict)
            assert eval_run.result.get("score") == 1.0

        metadata_evals = [
            eval_run for eval_run in result["evaluation_runs"] if eval_run.name == "metadata"
        ]
        assert len(metadata_evals) == 2

        for eval_run in metadata_evals:
            assert eval_run.result is not None
            assert isinstance(eval_run.result, dict)
            assert eval_run.result.get("score") is not None
            assert eval_run.result.get("label") is not None
            assert eval_run.result.get("explanation") is not None

        example_evals = [
            eval_run for eval_run in result["evaluation_runs"] if eval_run.name == "example"
        ]
        assert len(example_evals) == 2
        for eval_run in example_evals:
            assert eval_run.result is not None
            assert isinstance(eval_run.result, dict)
            assert eval_run.result.get("score") == 1.0
            assert eval_run.result.get("label") == "has_example"

    async def test_evaluator_receives_task_run_trace_id(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=f"test_trace_id_binding_{token_hex(4)}",
                inputs=[
                    {"text": "first"},
                    {"text": "second"},
                ],
                outputs=[
                    {"expected": "FIRST"},
                    {"expected": "SECOND"},
                ],
            )
        )

        received_trace_ids: list[str] = []

        def task(input: Dict[str, Any]) -> str:
            return cast(str, input["text"].upper())

        def evaluator(output: str, trace_id: Optional[str] = None) -> float:
            assert trace_id is not None
            received_trace_ids.append(trace_id)
            return 1.0 if trace_id else 0.0

        result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=task,
                evaluators=[evaluator],
                experiment_name=f"test_trace_id_eval_{token_hex(4)}",
                print_summary=False,
            )
        )

        task_run_trace_ids = [run["trace_id"] for run in result["task_runs"]]

        assert len(received_trace_ids) == len(task_run_trace_ids) == 2
        assert all(trace_id is not None for trace_id in task_run_trace_ids)
        assert received_trace_ids == task_run_trace_ids

    async def test_task_dynamic_parameter_binding(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_task_params_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[
                    {"question": "What is 2+2?", "type": "math"},
                    {"question": "What is the capital of France?", "type": "geography"},
                ],
                outputs=[
                    {"answer": "4", "explanation": "Basic arithmetic"},
                    {"answer": "Paris", "explanation": "Capital city of France"},
                ],
                metadata=[
                    {"difficulty": "easy", "category": "arithmetic", "source": "textbook"},
                    {"difficulty": "medium", "category": "geography", "source": "atlas"},
                ],
            )
        )

        def input_only_task(input: Dict[str, Any]) -> str:
            question = input.get("question", "")
            return f"Processing: {question}"

        def input_expected_task(input: Dict[str, Any], expected: Dict[str, Any]) -> str:
            question = input.get("question", "")
            expected_answer = expected.get("answer", "")
            return f"Question: {question}, Expected: {expected_answer}"

        def reference_task(input: Dict[str, Any], reference: Dict[str, Any]) -> str:
            question = input.get("question", "")
            ref_answer = reference.get("answer", "")
            return f"Q: {question}, Ref: {ref_answer}"

        def metadata_task(input: Dict[str, Any], metadata: Dict[str, Any]) -> str:
            question = input.get("question", "")
            difficulty = metadata.get("difficulty", "unknown")
            category = metadata.get("category", "unknown")
            return f"Q: {question} [Difficulty: {difficulty}, Category: {category}]"

        def comprehensive_task(
            input: Dict[str, Any],
            expected: Dict[str, Any],
            reference: Dict[str, Any],
            metadata: Dict[str, Any],
            example: Dict[str, Any],
        ) -> Dict[str, Any]:
            has_input = bool(input.get("question"))
            has_expected = bool(expected.get("answer"))
            has_reference = bool(reference.get("answer"))
            has_metadata = bool(metadata.get("difficulty"))
            has_example = bool(example.get("id"))  # Example should have an ID
            reference_matches_expected = reference == expected

            success = all(
                [
                    has_input,
                    has_expected,
                    has_reference,
                    has_metadata,
                    has_example,
                    reference_matches_expected,
                ]
            )

            return {
                "success": success,
                "question": input.get("question", ""),
                "expected_answer": expected.get("answer", ""),
                "metadata_difficulty": metadata.get("difficulty", ""),
                "example_id": example.get("id", ""),
                "reference_matches_expected": reference_matches_expected,
            }

        result1 = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=input_only_task,
                experiment_name=f"test_input_only_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result1["task_runs"]) == 2
        for task_run in result1["task_runs"]:
            assert "Processing:" in task_run["output"]

        result2 = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=input_expected_task,
                experiment_name=f"test_input_expected_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result2["task_runs"]) == 2
        for task_run in result2["task_runs"]:
            assert "Question:" in task_run["output"]
            assert "Expected:" in task_run["output"]

        result3 = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=reference_task,
                experiment_name=f"test_reference_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result3["task_runs"]) == 2
        for task_run in result3["task_runs"]:
            assert "Q:" in task_run["output"]
            assert "Ref:" in task_run["output"]

        result4 = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=metadata_task,
                experiment_name=f"test_metadata_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result4["task_runs"]) == 2
        for task_run in result4["task_runs"]:
            assert "Difficulty:" in task_run["output"]
            assert "Category:" in task_run["output"]

        result5 = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=comprehensive_task,
                experiment_name=f"test_comprehensive_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert len(result5["task_runs"]) == 2
        for task_run in result5["task_runs"]:
            output = task_run["output"]
            assert isinstance(output, dict)
            assert output["success"] is True
            assert output["reference_matches_expected"] is True
            assert output["question"] != ""
            assert output["expected_answer"] != ""
            assert output["metadata_difficulty"] != ""
            assert output["example_id"] != ""

    async def test_get_experiment(
        self,
        is_async: bool,
        _app: _AppInfo,
        _setup_experiment_test: _SetupExperimentTest,
    ) -> None:
        """Test getting a single experiment by ID."""
        client, helper = _setup_experiment_test(is_async)

        # Create a dataset, experiment, and run
        dataset_id, examples = helper.create_dataset(
            inputs=[{"q": "test"}],
            outputs=[{"a": "answer"}],
        )
        exp = helper.create_experiment(dataset_id, repetitions=1)
        helper.create_runs(exp["id"], [(examples[0]["id"], 1, "response", None)])

        # Test get method
        retrieved = await _await_or_return(client.experiments.get(experiment_id=exp["id"]))

        assert retrieved["id"] == exp["id"]
        assert retrieved["dataset_id"] == dataset_id
        assert retrieved["repetitions"] == 1
        assert retrieved["example_count"] == 1
        assert retrieved["successful_run_count"] == 1
        assert "created_at" in retrieved
        assert "updated_at" in retrieved

    async def test_list_experiments(
        self,
        is_async: bool,
        _app: _AppInfo,
        _setup_experiment_test: _SetupExperimentTest,
    ) -> None:
        """Test listing experiments for a dataset."""
        client, helper = _setup_experiment_test(is_async)

        # Create a dataset
        dataset_id, examples = helper.create_dataset(
            inputs=[{"q": f"test{i}"} for i in range(3)],
            outputs=[{"a": f"answer{i}"} for i in range(3)],
        )

        # Create multiple experiments with runs
        exp_ids: list[str] = []
        for i in range(3):
            exp = helper.create_experiment(dataset_id, repetitions=1)
            # Create runs for all examples in one call
            runs: list[tuple[str, int, Optional[str], Optional[str]]] = [
                (ex["id"], 1, f"response_{i}", None) for ex in examples
            ]
            helper.create_runs(exp["id"], runs)
            exp_ids.append(exp["id"])

        # List experiments
        experiments = await _await_or_return(client.experiments.list(dataset_id=dataset_id))

        assert len(experiments) == 3
        for exp in experiments:
            assert exp["dataset_id"] == dataset_id
            assert exp["example_count"] == 3
            assert exp["successful_run_count"] == 3
            assert "id" in exp
            assert "created_at" in exp
            assert "updated_at" in exp

        # Verify all created experiments are in the list
        retrieved_ids = [exp["id"] for exp in experiments]
        for exp_id in exp_ids:
            assert exp_id in retrieved_ids

    async def test_delete_experiment(
        self,
        is_async: bool,
        _app: _AppInfo,
        _setup_experiment_test: _SetupExperimentTest,
    ) -> None:
        """Test deleting an experiment."""
        client, helper = _setup_experiment_test(is_async)

        # Create a dataset, experiment, and run
        dataset_id, examples = helper.create_dataset(
            inputs=[{"q": "test"}],
            outputs=[{"a": "answer"}],
        )
        exp = helper.create_experiment(dataset_id, repetitions=1)
        helper.create_runs(exp["id"], [(examples[0]["id"], 1, "response", None)])

        # Delete the experiment
        await _await_or_return(client.experiments.delete(experiment_id=exp["id"]))

        # Verify experiment no longer exists
        with pytest.raises(ValueError, match="Experiment not found"):
            await _await_or_return(client.experiments.get(experiment_id=exp["id"]))

        # Verify it's not in the list
        experiments = await _await_or_return(client.experiments.list(dataset_id=dataset_id))
        experiment_ids = [exp["id"] for exp in experiments]
        assert exp["id"] not in experiment_ids

    async def test_experiment_run_upsert_protection(
        self,
        _app: _AppInfo,
        _setup_experiment_test: _SetupExperimentTest,
    ) -> None:
        """
        Test that experiment runs with successful results cannot be overwritten.

        Verifies:
        1. Successful runs (error=None) return 409 Conflict when update is attempted
        2. Failed runs (error not None) can be updated with new errors
        3. Failed runs can be updated to successful runs
        4. Once updated to successful, runs become protected from further updates
        """
        _, helper = _setup_experiment_test(False)

        # Create dataset and experiment
        dataset_id, examples = helper.create_dataset(
            inputs=[{"q": "test1"}, {"q": "test2"}],
            outputs=[{"a": "answer1"}, {"a": "answer2"}],
        )
        exp = helper.create_experiment(dataset_id, repetitions=1)

        # Test 1: Create a successful run (error=None)
        successful_run = helper.create_runs(
            exp["id"], [(examples[0]["id"], 1, "original_output", None)]
        )[0]
        assert successful_run["id"] is not None

        # Test 2: Attempt to update the successful run - should return 409
        response = helper.http_client.post(
            f"v1/experiments/{exp['id']}/runs",
            json={
                "dataset_example_id": examples[0]["id"],
                "repetition_number": 1,
                "output": "updated_output",
                "error": None,
                "start_time": helper.now,
                "end_time": helper.now,
            },
        )
        assert response.status_code == 409
        assert "already exists with a successful result" in response.text

        # Test 3: Create a failed run (error not None)
        failed_run = helper.create_runs(
            exp["id"], [(examples[1]["id"], 1, "failed_output", "Some error")]
        )[0]
        assert failed_run["id"] is not None

        # Verify the failed run was created with the error
        runs = helper.get_runs(exp["id"])
        example1_runs = [r for r in runs if r["dataset_example_id"] == examples[1]["id"]]
        assert len(example1_runs) == 1
        assert example1_runs[0]["output"] == "failed_output"
        assert example1_runs[0].get("error") == "Some error"

        # Test 4: Update the failed run with a new error - should succeed
        updated_run = helper.create_runs(
            exp["id"], [(examples[1]["id"], 1, "retried_output", "New error message")]
        )[0]
        assert updated_run["id"] is not None

        # Verify the error was updated
        runs = helper.get_runs(exp["id"])
        example1_runs = [r for r in runs if r["dataset_example_id"] == examples[1]["id"]]
        assert len(example1_runs) == 1
        assert example1_runs[0]["output"] == "retried_output"
        assert example1_runs[0].get("error") == "New error message"

        # Test 5: Update the failed run to successful - should succeed
        successful_retry = helper.create_runs(
            exp["id"], [(examples[1]["id"], 1, "final_output", None)]
        )[0]
        assert successful_retry["id"] is not None

        # Verify the run is now successful (error is None)
        runs = helper.get_runs(exp["id"])
        example1_runs = [r for r in runs if r["dataset_example_id"] == examples[1]["id"]]
        assert len(example1_runs) == 1
        assert example1_runs[0]["output"] == "final_output"
        assert example1_runs[0].get("error") is None

        # Test 6: Now the previously-failed run is successful, it should be protected
        response = helper.http_client.post(
            f"v1/experiments/{exp['id']}/runs",
            json={
                "dataset_example_id": examples[1]["id"],
                "repetition_number": 1,
                "output": "another_attempt",
                "error": None,
                "start_time": helper.now,
                "end_time": helper.now,
            },
        )
        assert response.status_code == 409

    async def test_resume_incomplete_runs_comprehensive(
        self, is_async: bool, _app: _AppInfo, _setup_experiment_test: _SetupExperimentTest
    ) -> None:
        """
        Comprehensive test for resuming incomplete runs.

        Tests all incomplete run scenarios in one test:
        1. Failed runs are resumed (examples 0-1)
        2. Successful runs are preserved (example 2)
        3. Missing runs are created (examples 3-4)
        4. Mixed failed+missing work together (examples 0-4)
        5. Repetition-level granularity (example 5: only failed rep 2)

        This consolidates test_basic_resume, test_missing_runs,
        test_mixed_missing_and_failed, and test_repetition_level_granularity.
        """
        client, helper = _setup_experiment_test(is_async)

        # Create first dataset for comprehensive mixed scenarios (examples 0-4)
        dataset_id_1, examples_1 = helper.create_dataset(
            inputs=[{"idx": i} for i in range(5)],
            outputs=[{"result": i} for i in range(5)],
        )

        # Create second dataset for repetition-level granularity test (example 5)
        dataset_id_2, examples_2 = helper.create_dataset(
            inputs=[{"idx": 5}],
            outputs=[{"result": 5}],
        )

        # Experiment 1: Examples 0-4 with 2 repetitions each
        exp = helper.create_experiment(dataset_id_1, repetitions=2)
        helper.create_runs(
            exp["id"],
            [
                # Example 0: failed runs (both reps)
                (examples_1[0]["id"], 1, None, "Failed 0"),
                (examples_1[0]["id"], 2, None, "Failed 0"),
                # Example 1: failed runs (both reps)
                (examples_1[1]["id"], 1, None, "Failed 1"),
                (examples_1[1]["id"], 2, None, "Failed 1"),
                # Example 2: successful runs (will be preserved)
                (examples_1[2]["id"], 1, "Success 2", None),
                (examples_1[2]["id"], 2, "Success 2", None),
                # Examples 3-4: missing runs (not created, but should exist per repetitions=2)
            ],
        )

        # Experiment 2: Example 5 with 3 repetitions (repetition-level test)
        exp_reps = helper.create_experiment(dataset_id_2, repetitions=3)
        helper.create_runs(
            exp_reps["id"],
            [
                (examples_2[0]["id"], 1, "Success rep 1", None),
                (examples_2[0]["id"], 2, None, "Failed rep 2"),
                (examples_2[0]["id"], 3, "Success rep 3", None),
            ],
        )

        # Track execution
        processed: set[int] = set()
        call_count = [0]

        def tracking_task(input: dict[str, Any]) -> str:
            idx = cast(int, input["idx"])
            processed.add(idx)
            call_count[0] += 1
            return f"Resumed {idx}"

        # Resume experiment 1 (examples 0-4)
        await _await_or_return(
            client.experiments.resume_experiment(
                experiment_id=exp["id"],
                task=tracking_task,
                print_summary=False,
            )
        )

        # Verify experiment has correct counts after resuming
        # 5 examples × 2 repetitions = 10 runs total
        resumed_exp = await _await_or_return(client.experiments.get(experiment_id=exp["id"]))
        assert resumed_exp["successful_run_count"] == 10
        assert {0, 1, 3, 4} <= processed, "Should process failed and missing examples"

        # Verify outputs
        helper.assert_output_by_example(
            exp["id"],
            expected={
                0: "Resumed 0",  # Failed → resumed (both reps)
                1: "Resumed 1",  # Failed → resumed (both reps)
                2: "Success 2",  # Successful → preserved (both reps)
                3: "Resumed 3",  # Missing → created (both reps)
                4: "Resumed 4",  # Missing → created (both reps)
            },
            examples=examples_1,
        )

        # Resume experiment 2 (example 5 with repetitions)
        call_count[0] = 0  # Reset counter
        await _await_or_return(
            client.experiments.resume_experiment(
                experiment_id=exp_reps["id"],
                task=tracking_task,
                print_summary=False,
            )
        )

        # Verify only 1 repetition was re-run
        assert call_count[0] == 1, "Should only re-run 1 failed repetition"
        assert 5 in processed

        # Verify repetition-level precision
        helper.assert_output_by_example(
            exp_reps["id"],
            expected={
                0: ["Success rep 1", "Resumed 5", "Success rep 3"],
            },
            examples=examples_2,
        )

    async def test_early_exit_when_complete(
        self, is_async: bool, _app: _AppInfo, _setup_experiment_test: _SetupExperimentTest
    ) -> None:
        """
        Test early exit optimization when all runs/evaluations are complete.

        Tests:
        1. resume_experiment early exit when all task runs are successful
        2. resume_evaluation early exit when all evaluations are complete

        Verifies that both operations detect when there's nothing to do and
        skip unnecessary function calls (optimization).
        """
        client, helper = _setup_experiment_test(is_async)

        # Scenario 1: resume_experiment early exit
        dataset_id, examples = helper.create_dataset(
            inputs=[{"q": "Q1"}],
            outputs=[{"a": "A1"}],
        )
        exp = helper.create_experiment(dataset_id, repetitions=1)
        helper.create_runs(exp["id"], [(examples[0]["id"], 1, "Complete", None)])

        task_call_count = [0]

        def noop_task(input: dict[str, Any]) -> str:
            task_call_count[0] += 1
            return "Result"

        await _await_or_return(
            client.experiments.resume_experiment(
                experiment_id=exp["id"],
                task=noop_task,
                print_summary=False,
            )
        )

        assert task_call_count[0] == 0, "Task should not be called when no incomplete runs"
        # Verify experiment state after early exit
        resumed_exp = await _await_or_return(client.experiments.get(experiment_id=exp["id"]))
        assert resumed_exp["id"] == exp["id"]
        assert resumed_exp["example_count"] == 1
        assert resumed_exp["successful_run_count"] == 1

        # Scenario 2: resume_evaluation early exit
        # Add successful evaluation for "relevance"
        helper.create_evaluations(exp["id"], ["relevance"], [])

        eval_call_count = [0]

        def accuracy_evaluator(output: Any) -> float:
            eval_call_count[0] += 1
            return 1.0

        # Try to resume with same evaluator (should early exit, no re-run)
        await _await_or_return(
            client.experiments.resume_evaluation(
                experiment_id=exp["id"],
                evaluators={"relevance": accuracy_evaluator},  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                print_summary=False,
            )
        )

        assert eval_call_count[0] == 0, "Evaluator should not be called when evaluation is complete"

        # Verify the existing evaluation was not modified
        data = helper.get_experiment_annotations(exp["id"])
        helper.assert_annotations(
            runs_data=data["runs"]["edges"],
            expected_count=1,
            expected_by_run={"relevance": 1.0},
        )

    async def test_error_scenarios(
        self, is_async: bool, _app: _AppInfo, _setup_experiment_test: _SetupExperimentTest
    ) -> None:
        """
        Comprehensive error handling test covering multiple error scenarios.

        Tests:
        1. Resume task continues to fail - new error is recorded
        2. Resume evaluation continues to fail - new error is recorded
        3. Invalid experiment ID (non-existent ID)
        4. Empty evaluators dict for resume_evaluation

        Verifies that all error conditions are handled gracefully with appropriate
        error messages and the system remains stable.
        """
        client, helper = _setup_experiment_test(is_async)

        # Scenario 1: Resume task continues to fail
        dataset_id, examples = helper.create_dataset(
            inputs=[{"x": i} for i in range(3)],
            outputs=[{"y": i} for i in range(3)],
        )
        exp = helper.create_experiment(dataset_id, repetitions=1)
        helper.create_runs(
            exp["id"], [(examples[i]["id"], 1, None, "Original error") for i in range(3)]
        )

        def failing_task(input: dict[str, Any]) -> str:
            raise ValueError("Task still fails on resume")

        await _await_or_return(
            client.experiments.resume_experiment(
                experiment_id=exp["id"],
                task=failing_task,
                print_summary=False,
            )
        )

        # Verify NEW errors are recorded in the runs (not the original errors)
        runs = helper.get_runs(exp["id"])
        assert len([r for r in runs if r.get("error")]) == 3, "All runs should still have errors"
        for run in runs:
            error_msg = run.get("error") or ""
            assert "Task still fails on resume" in error_msg, (
                f"Expected new error message, got: {error_msg}"
            )
            assert "Original error" not in error_msg, "Should have new error, not original"

        # Scenario 2: Resume evaluation continues to fail
        exp2 = helper.create_experiment(dataset_id, repetitions=1)
        helper.create_runs(
            exp2["id"], [(examples[i]["id"], 1, f"output_{i}", None) for i in range(3)]
        )
        # Add failed evaluations
        helper.create_evaluations(exp2["id"], [], ["quality"])

        def failing_evaluator(output: Any) -> float:
            raise ValueError("Evaluator still fails on resume")

        await _await_or_return(
            client.experiments.resume_evaluation(
                experiment_id=exp2["id"],
                evaluators={"quality": failing_evaluator},  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                print_summary=False,
            )
        )

        # Verify NEW evaluation errors are recorded
        data = helper.get_experiment_annotations(exp2["id"])
        annotations = data["runs"]["edges"]
        assert len(annotations) == 3, "Should have 3 runs"
        for run_edge in annotations:
            run_annotations = run_edge["run"]["annotations"]["edges"]
            quality_annotations = [
                a["annotation"] for a in run_annotations if a["annotation"]["name"] == "quality"
            ]
            assert len(quality_annotations) == 1, "Should have one quality annotation per run"
            annotation = quality_annotations[0]
            assert annotation["error"] is not None, "Annotation should have error"
            assert "Evaluator still fails on resume" in annotation["error"], (
                f"Expected new evaluation error message, got: {annotation['error']}"
            )

        # Scenario 3: Invalid experiment ID
        with pytest.raises(ValueError, match="Experiment not found"):
            await _await_or_return(
                client.experiments.resume_experiment(
                    experiment_id=str(GlobalID("Experiment", "999999")),
                    task=lambda input: "x",  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                    print_summary=False,
                )
            )

        # Scenario 4: Empty evaluators for resume_evaluation
        with pytest.raises(ValueError, match="Must specify at least one evaluator"):
            await _await_or_return(
                client.experiments.resume_evaluation(
                    experiment_id=exp["id"],
                    evaluators={},  # Empty dict - should fail validation
                    print_summary=False,
                )
            )

    async def test_resume_experiment_with_evaluators(
        self, is_async: bool, _app: _AppInfo, _setup_experiment_test: _SetupExperimentTest
    ) -> None:
        """
        Test resume_experiment with evaluators integration.

        Validates that:
        - Task runs are completed first
        - Evaluators are automatically run on completed runs
        - The integration between resume_experiment and resume_evaluation works correctly
        """
        client, helper = _setup_experiment_test(is_async)

        dataset_id, examples = helper.create_dataset(
            inputs=[{"q": f"Q{i}"} for i in range(2)],
            outputs=[{"a": f"A{i}"} for i in range(2)],
        )

        exp = helper.create_experiment(dataset_id, repetitions=1)
        # Create only failed runs
        helper.create_runs(
            exp["id"],
            [
                (examples[0]["id"], 1, None, "Failed 1"),
                (examples[1]["id"], 1, None, "Failed 2"),
            ],
        )

        # Resume with both task and evaluators
        await _await_or_return(
            client.experiments.resume_experiment(
                experiment_id=exp["id"],
                task=lambda input: f"Resumed {cast(str, input['q'])}",  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                evaluators={"quality": lambda output: 0.9},  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                print_summary=False,
            )
        )

        # Verify task runs completed
        resumed_exp = await _await_or_return(client.experiments.get(experiment_id=exp["id"]))
        assert resumed_exp["id"] == exp["id"]
        assert resumed_exp["successful_run_count"] == 2

        # Verify task outputs were persisted
        helper.assert_output_by_example(
            exp["id"],
            expected={
                0: "Resumed Q0",
                1: "Resumed Q1",
            },
            examples=examples,
        )

        # Verify evaluations were run
        data = helper.get_experiment_annotations(exp["id"])
        helper.assert_annotations(
            runs_data=data["runs"]["edges"],
            expected_count=2,
            expected_by_run={"quality": 0.9},
        )

    async def test_resume_evaluation_comprehensive(
        self, is_async: bool, _app: _AppInfo, _setup_experiment_test: _SetupExperimentTest
    ) -> None:
        """
        Comprehensive test for resume_evaluation covering all scenarios.

        Tests:
        1. Successful evaluations are NOT re-run (accuracy, relevance preserved)
        2. Failed evaluations ARE re-run (quality re-executed)
        3. Missing evaluations ARE run (toxicity added)
        4. Selective retry: can resume only specific evaluators
        5. Pagination: large experiments with > 50 runs are handled correctly
        6. Call count tracking verifies correct execution

        This provides complete coverage of resume_evaluation logic.
        """
        client, helper = _setup_experiment_test(is_async)

        # Create experiment with 3 runs
        dataset_id, examples = helper.create_dataset(
            inputs=[{"input": f"input_{i}"} for i in range(3)],
            outputs=[{"output": f"output_{i}"} for i in range(3)],
        )
        exp = helper.create_experiment(dataset_id, repetitions=1)
        helper.create_runs(
            exp["id"],
            [(examples[i]["id"], 1, f"result_{i}", None) for i in range(3)],
        )

        # Add evaluations: accuracy and relevance successful, quality failed
        helper.create_evaluations(exp["id"], ["accuracy", "relevance"], ["quality"])

        # Part 1: Selective retry - resume only the failed "quality" evaluator
        # Other successful evaluators (accuracy, relevance) should be preserved
        quality_call_count = [0]

        def quality_evaluator(output: Any) -> float:
            quality_call_count[0] += 1
            return 0.95

        await _await_or_return(
            client.experiments.resume_evaluation(
                experiment_id=exp["id"],
                evaluators={"quality": quality_evaluator},  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                print_summary=False,
            )
        )

        # Verify quality evaluator was called exactly 3 times (once per run)
        assert quality_call_count[0] == 3, "Quality evaluator should run for all 3 runs"

        # Verify all three evaluators are present with correct scores
        data = helper.get_experiment_annotations(exp["id"])
        helper.assert_annotations(
            runs_data=data["runs"]["edges"],
            expected_count=3,
            expected_by_run={
                "accuracy": 1.0,  # Preserved from original successful run
                "quality": 0.95,  # Updated from failed to successful
                "relevance": 1.0,  # Preserved from original successful run
            },
        )

        # Part 2: Add missing evaluator - resume with new "toxicity" evaluator
        toxicity_call_count = [0]

        def toxicity_evaluator(output: Any) -> float:
            toxicity_call_count[0] += 1
            return 0.1

        await _await_or_return(
            client.experiments.resume_evaluation(
                experiment_id=exp["id"],
                evaluators={"toxicity": toxicity_evaluator},  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                print_summary=False,
            )
        )

        # Verify toxicity evaluator was called 3 times (once per run)
        assert toxicity_call_count[0] == 3, "Toxicity evaluator should run for all 3 runs"

        # Verify all four evaluators are now present
        data = helper.get_experiment_annotations(exp["id"])
        helper.assert_annotations(
            runs_data=data["runs"]["edges"],
            expected_count=3,
            expected_by_run={
                "accuracy": 1.0,  # Still preserved
                "quality": 0.95,  # Still updated
                "relevance": 1.0,  # Still preserved
                "toxicity": 0.1,  # Newly added
            },
        )

        # Part 3: Pagination - test with large experiment (>50 runs)
        num_examples = 75
        dataset_id_large, examples_large = helper.create_dataset(
            inputs=[{"x": i} for i in range(num_examples)],
            outputs=[{"y": i * 2} for i in range(num_examples)],
        )

        exp_large = helper.create_experiment(dataset_id_large, repetitions=1)
        helper.create_runs(
            exp_large["id"],
            [(examples_large[i]["id"], 1, f"output_{i}", None) for i in range(num_examples)],
        )

        # All runs are successful but missing "pagination_test" evaluation
        evaluated_indices: set[int] = set()

        def pagination_evaluator(output: Any) -> float:
            # Extract index from output to track which runs were evaluated
            output_str = str(output)  # pyright: ignore[reportUnknownArgumentType]
            idx = int(output_str.split("_")[1])
            evaluated_indices.add(idx)
            return 0.8

        await _await_or_return(
            client.experiments.resume_evaluation(
                experiment_id=exp_large["id"],
                evaluators={"pagination_test": pagination_evaluator},  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                print_summary=False,
            )
        )

        # Verify all runs were evaluated (no skips due to pagination)
        assert evaluated_indices == set(range(num_examples)), (
            f"All {num_examples} runs should be evaluated across pagination boundaries, "
            f"but only {len(evaluated_indices)} were evaluated"
        )

        # Verify evaluations were persisted to the database
        data_large = helper.get_experiment_annotations(exp_large["id"])
        helper.assert_annotations(
            runs_data=data_large["runs"]["edges"],
            expected_count=num_examples,
            expected_by_run={"pagination_test": 0.8},
        )

    async def test_run_experiment_then_evaluate_experiment_pattern(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        """Test running experiment without evaluators, then adding evaluations separately."""
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_evaluate_pattern_{token_hex(4)}"

        # Create test dataset
        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[
                    {"question": "What is 2+2?"},
                    {"question": "What is the capital of France?"},
                    {"question": "Who wrote Python?"},
                ],
                outputs=[
                    {"answer": "4"},
                    {"answer": "Paris"},
                    {"answer": "Guido van Rossum"},
                ],
                metadata=[
                    {"category": "math"},
                    {"category": "geography"},
                    {"category": "programming"},
                ],
            )
        )

        def simple_task(input: Dict[str, Any]) -> str:
            question = input.get("question", "")
            if "2+2" in question:
                return "The answer is 4"
            elif "capital" in question:
                return "The capital is Paris"
            elif "Python" in question:
                return "Guido van Rossum created Python"
            else:
                return "I don't know"

        def accuracy_evaluator(output: str, expected: Dict[str, Any]) -> float:
            expected_answer = expected.get("answer", "")
            return 1.0 if expected_answer in output else 0.0

        def length_evaluator(output: str) -> Dict[str, Any]:
            return {"score": len(output) / 20.0, "label": "length_score"}

        # Step 1: Run experiment WITHOUT evaluators (task execution only)
        initial_result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=dataset,
                task=simple_task,
                experiment_name=f"test_initial_{token_hex(4)}",
                print_summary=False,
            )
        )

        # Verify initial result has no evaluations but has task runs
        assert "experiment_id" in initial_result
        assert "dataset_id" in initial_result
        assert "task_runs" in initial_result
        assert "evaluation_runs" in initial_result
        assert initial_result["dataset_id"] == dataset.id
        assert len(initial_result["task_runs"]) == 3
        assert len(initial_result["evaluation_runs"]) == 0  # No evaluations yet

        # Step 2: Add evaluations to the completed experiment
        # This will test the new evaluate_experiment method
        eval_result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.evaluate_experiment(
                experiment=initial_result,
                evaluators=[accuracy_evaluator, length_evaluator],
                print_summary=False,
            )
        )

        # Verify evaluation results
        assert "experiment_id" in eval_result
        assert "dataset_id" in eval_result
        assert "task_runs" in eval_result
        assert "evaluation_runs" in eval_result
        assert eval_result["experiment_id"] == initial_result["experiment_id"]
        assert eval_result["dataset_id"] == dataset.id
        assert len(eval_result["task_runs"]) == 3  # Same task runs as before
        assert len(eval_result["evaluation_runs"]) > 0  # Now we have evaluations

        # Verify that we have evaluations for each task run and each evaluator
        expected_eval_runs = len(eval_result["task_runs"]) * 2  # 2 evaluators
        assert len(eval_result["evaluation_runs"]) == expected_eval_runs

    async def test_evaluation_consistency_when_implemented(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        """Test that run_experiment with evaluators produces same results as separate evaluation."""
        # Test is now enabled since evaluate_experiment is implemented
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_consistency_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[{"text": "Hello world"}, {"text": "Python is great"}],
                outputs=[{"expected": "greeting"}, {"expected": "programming"}],
            )
        )

        def simple_task(input: Dict[str, Any]) -> str:
            text = input.get("text", "")
            if "Hello" in text:
                return "greeting"
            elif "Python" in text:
                return "programming"
            else:
                return "unknown"

        def accuracy_evaluator(output: str, expected: Dict[str, Any]) -> float:
            return 1.0 if output == expected.get("expected") else 0.0

        client = Client(base_url=_app.base_url, api_key=api_key)

        # Method 1: Run experiment with evaluators included
        result_with_evals = await _await_or_return(
            client.experiments.run_experiment(
                dataset=dataset,
                task=simple_task,
                evaluators=[accuracy_evaluator],
                experiment_name=f"test_with_evals_{token_hex(4)}",
                print_summary=False,
            )
        )

        # Method 2: Run experiment without evaluators, then evaluate separately
        result_without_evals = await _await_or_return(
            client.experiments.run_experiment(
                dataset=dataset,
                task=simple_task,
                experiment_name=f"test_without_evals_{token_hex(4)}",
                print_summary=False,
            )
        )

        eval_result = await _await_or_return(
            client.experiments.evaluate_experiment(
                experiment=result_without_evals,
                evaluators=[accuracy_evaluator],
                print_summary=False,
            )
        )

        # Both methods should produce equivalent results
        assert len(result_with_evals["evaluation_runs"]) == len(eval_result["evaluation_runs"])
        assert len(result_with_evals["task_runs"]) == len(result_without_evals["task_runs"])

        # Evaluation results should be equivalent
        for eval1, eval2 in zip(
            result_with_evals["evaluation_runs"], eval_result["evaluation_runs"]
        ):
            assert eval1.name == eval2.name
            assert eval1.result == eval2.result

    async def test_get_experiment_and_evaluate(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_get_experiment_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[
                    {"question": "What is 2+2?"},
                    {"question": "What is the capital of France?"},
                ],
                outputs=[
                    {"answer": "4"},
                    {"answer": "Paris"},
                ],
                metadata=[
                    {"category": "math"},
                    {"category": "geography"},
                ],
            )
        )

        def simple_task(input: Dict[str, Any]) -> str:
            question = input.get("question", "")
            if "2+2" in question:
                return "The answer is 4"
            elif "capital" in question:
                return "The capital is Paris"
            else:
                return "I don't know"

        def accuracy_evaluator(output: str, expected: Dict[str, Any]) -> float:
            expected_answer = expected.get("answer", "")
            return 1.0 if expected_answer in output else 0.0

        def length_evaluator(output: str) -> Dict[str, Any]:
            return {"score": len(output) / 20.0, "label": "length_score"}

        client = Client(base_url=_app.base_url, api_key=api_key)

        initial_result = await _await_or_return(
            client.experiments.run_experiment(
                dataset=dataset,
                task=simple_task,
                evaluators=[accuracy_evaluator],  # Start with one evaluator
                experiment_name=f"test_get_exp_{token_hex(4)}",
                print_summary=False,
            )
        )

        assert "experiment_id" in initial_result
        assert "dataset_id" in initial_result
        assert "task_runs" in initial_result
        assert "evaluation_runs" in initial_result
        assert len(initial_result["task_runs"]) == 2
        assert len(initial_result["evaluation_runs"]) == 2  # Should have 2 evals (1 per task run)

        initial_accuracy_evals = [
            eval_run
            for eval_run in initial_result["evaluation_runs"]
            if eval_run.name == "accuracy_evaluator"
        ]
        assert len(initial_accuracy_evals) == 2

        retrieved_experiment = await _await_or_return(
            client.experiments.get_experiment(experiment_id=initial_result["experiment_id"])
        )

        assert retrieved_experiment["experiment_id"] == initial_result["experiment_id"]
        assert retrieved_experiment["dataset_id"] == initial_result["dataset_id"]
        assert len(retrieved_experiment["task_runs"]) == len(initial_result["task_runs"])

        assert len(retrieved_experiment["evaluation_runs"]) == len(
            initial_result["evaluation_runs"]
        )
        assert len(retrieved_experiment["evaluation_runs"]) == 2

        retrieved_accuracy_evals = [
            eval_run
            for eval_run in retrieved_experiment["evaluation_runs"]
            if eval_run.name == "accuracy_evaluator"
        ]
        assert len(retrieved_accuracy_evals) == 2

        task_outputs = [run["output"] for run in retrieved_experiment["task_runs"]]
        assert "The answer is 4" in task_outputs
        assert "The capital is Paris" in task_outputs

        final_result = await _await_or_return(
            client.experiments.evaluate_experiment(
                experiment=retrieved_experiment,
                evaluators=[length_evaluator],  # Add a second evaluator
                print_summary=False,
            )
        )

        assert final_result["experiment_id"] == initial_result["experiment_id"]
        assert final_result["dataset_id"] == initial_result["dataset_id"]
        assert len(final_result["task_runs"]) == 2  # Same task runs

        assert len(final_result["evaluation_runs"]) == 4

        final_accuracy_evals = [
            eval_run
            for eval_run in final_result["evaluation_runs"]
            if eval_run.name == "accuracy_evaluator"
        ]
        assert len(final_accuracy_evals) == 2

        final_length_evals = [
            eval_run
            for eval_run in final_result["evaluation_runs"]
            if eval_run.name == "length_evaluator"
        ]
        assert len(final_length_evals) == 2

        # Verify evaluation results
        for eval_run in final_accuracy_evals:
            assert eval_run.result is not None
            assert isinstance(eval_run.result, dict)
            assert eval_run.result.get("score") == 1.0

        for eval_run in final_length_evals:
            assert eval_run.result is not None
            assert isinstance(eval_run.result, dict)
            assert eval_run.result.get("score") is not None
            assert eval_run.result.get("label") == "length_score"

    async def test_dry_run_with_evaluate_experiment(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_dry_run_eval_{token_hex(4)}"

        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[
                    {"text": "Hello world"},
                    {"text": "Python is great"},
                ],
                outputs=[
                    {"expected": "greeting"},
                    {"expected": "programming"},
                ],
            )
        )

        def simple_task(input: Dict[str, Any]) -> str:
            text = input.get("text", "")
            if "Hello" in text:
                return "greeting"
            elif "Python" in text:
                return "programming"
            else:
                return "unknown"

        def accuracy_evaluator(output: str, expected: Dict[str, Any]) -> float:
            return 1.0 if output == expected.get("expected") else 0.0

        client = Client(base_url=_app.base_url, api_key=api_key)

        dry_run_result = await _await_or_return(
            client.experiments.run_experiment(
                dataset=dataset,
                task=simple_task,
                experiment_name=f"test_dry_run_{token_hex(4)}",
                dry_run=True,
                print_summary=False,
            )
        )

        assert dry_run_result["experiment_id"] == "DRY_RUN"
        assert len(dry_run_result["task_runs"]) == 1
        assert len(dry_run_result["evaluation_runs"]) == 0

        eval_result = await _await_or_return(
            client.experiments.evaluate_experiment(
                experiment=dry_run_result,
                evaluators=[accuracy_evaluator],
                dry_run=True,
                print_summary=False,
            )
        )

        assert eval_result["experiment_id"] == "DRY_RUN"
        assert len(eval_result["task_runs"]) == 1
        assert len(eval_result["evaluation_runs"]) == 1

    async def test_experiment_with_dataset_splits(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        """Test that experiments correctly record split_ids and populate the experiments_dataset_splits junction table."""
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        from .._helpers import _gql

        Client = AsyncClient if is_async else SyncClient

        unique_name = f"test_exp_splits_{token_hex(4)}"

        # Create dataset with examples
        dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.create_dataset(
                name=unique_name,
                inputs=[
                    {"question": "What is 2+2?"},
                    {"question": "What is the capital of France?"},
                    {"question": "Who wrote Python?"},
                    {"question": "What is recursion?"},
                ],
                outputs=[
                    {"answer": "4"},
                    {"answer": "Paris"},
                    {"answer": "Guido van Rossum"},
                    {"answer": "A function calling itself"},
                ],
                metadata=[
                    {"category": "math"},
                    {"category": "geography"},
                    {"category": "programming"},
                    {"category": "computer_science"},
                ],
            )
        )

        assert len(dataset) == 4
        example_ids = [example["id"] for example in dataset.examples]

        # Create splits using GraphQL
        # Split 1: Training set (first 2 examples)
        split_mutation = """
            mutation($input: CreateDatasetSplitWithExamplesInput!) {
                createDatasetSplitWithExamples(input: $input) {
                    datasetSplit {
                        id
                        name
                    }
                }
            }
        """

        train_split_result, _ = _gql(
            _app,
            _app.admin_secret,
            query=split_mutation,
            variables={
                "input": {
                    "name": f"{unique_name}_train",
                    "color": "#FF0000",
                    "exampleIds": [example_ids[0], example_ids[1]],
                }
            },
        )
        train_split_id = train_split_result["data"]["createDatasetSplitWithExamples"][
            "datasetSplit"
        ]["id"]
        train_split_name = train_split_result["data"]["createDatasetSplitWithExamples"][
            "datasetSplit"
        ]["name"]

        # Split 2: Test set (last 2 examples)
        test_split_result, _ = _gql(
            _app,
            _app.admin_secret,
            query=split_mutation,
            variables={
                "input": {
                    "name": f"{unique_name}_test",
                    "color": "#00FF00",
                    "exampleIds": [example_ids[2], example_ids[3]],
                }
            },
        )
        test_split_id = test_split_result["data"]["createDatasetSplitWithExamples"]["datasetSplit"][
            "id"
        ]
        test_split_name = test_split_result["data"]["createDatasetSplitWithExamples"][
            "datasetSplit"
        ]["name"]

        # First, verify that getting dataset with no splits filter returns ALL examples
        full_dataset_no_filter = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.get_dataset(dataset=dataset.id)
        )
        assert len(full_dataset_no_filter) == 4, (
            f"Expected all 4 examples with no filter, got {len(full_dataset_no_filter)}"
        )
        assert full_dataset_no_filter._filtered_split_names == [], (
            f"Expected empty split_names with no filter, got {full_dataset_no_filter._filtered_split_names}"
        )

        # Verify all original example IDs are present
        full_dataset_example_ids = {example["id"] for example in full_dataset_no_filter.examples}
        assert full_dataset_example_ids == set(example_ids), (
            "Full dataset should contain all original example IDs"
        )

        # Define GraphQL query for verifying experiment splits (used multiple times below)
        verify_splits_query = """
            query($experimentId: ID!) {
                node(id: $experimentId) {
                    ... on Experiment {
                        id
                        datasetSplits {
                            edges {
                                node {
                                    id
                                    name
                                }
                            }
                        }
                    }
                }
            }
        """

        # Define a simple task for the experiment (used in multiple tests below)
        def simple_task(input: Dict[str, Any]) -> str:
            question = input.get("question", "")
            if "2+2" in question:
                return "The answer is 4"
            elif "capital" in question:
                return "The capital is Paris"
            elif "Python" in question:
                return "Created by Guido van Rossum"
            elif "recursion" in question:
                return "When a function calls itself"
            else:
                return "I don't know"

        # Run an experiment on the full dataset (no split filter) to verify it processes all examples
        full_dataset_experiment = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=full_dataset_no_filter,
                task=simple_task,
                experiment_name=f"test_no_split_experiment_{token_hex(4)}",
                experiment_description="Test experiment with no split filter",
                print_summary=False,
            )
        )

        assert len(full_dataset_experiment["task_runs"]) == 4, (
            f"Expected 4 task runs on full dataset, got {len(full_dataset_experiment['task_runs'])}"
        )

        # Verify that experiment with no split filter has empty dataset_splits association
        no_split_exp_id = full_dataset_experiment["experiment_id"]
        no_split_verification, _ = _gql(
            _app,
            _app.admin_secret,
            query=verify_splits_query,
            variables={"experimentId": no_split_exp_id},
        )
        no_split_exp_node = no_split_verification["data"]["node"]
        no_split_edges = no_split_exp_node["datasetSplits"]["edges"]
        assert len(no_split_edges) == 0, (
            f"Expected 0 splits for experiment with no filter, got {len(no_split_edges)}"
        )

        # Get dataset filtered by train split only
        train_dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.get_dataset(
                dataset=dataset.id,
                splits=[train_split_name],
            )
        )

        assert len(train_dataset) == 2, (
            f"Expected 2 examples in train split, got {len(train_dataset)}"
        )
        assert train_split_name in train_dataset._filtered_split_names, (
            "Train split name should be in dataset._filtered_split_names"
        )

        # Run experiment on the filtered train dataset
        result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=train_dataset,
                task=simple_task,
                experiment_name=f"test_split_experiment_{token_hex(4)}",
                experiment_description="Test experiment with dataset splits",
                print_summary=False,
            )
        )

        # Verify experiment was created and ran on correct number of examples
        assert "experiment_id" in result
        assert result["experiment_id"] != "DRY_RUN"
        assert "dataset_id" in result
        assert result["dataset_id"] == dataset.id
        assert len(result["task_runs"]) == 2, (
            f"Expected 2 task runs (train split), got {len(result['task_runs'])}"
        )

        experiment_id = result["experiment_id"]

        # Query the database to verify the experiments_dataset_splits junction table is populated
        splits_verification, _ = _gql(
            _app,
            _app.admin_secret,
            query=verify_splits_query,
            variables={"experimentId": experiment_id},
        )

        experiment_node = splits_verification["data"]["node"]
        assert experiment_node is not None, "Experiment should exist"
        assert "datasetSplits" in experiment_node, "Experiment should have datasetSplits field"

        dataset_splits_edges = experiment_node["datasetSplits"]["edges"]
        assert len(dataset_splits_edges) == 1, (
            f"Expected 1 split associated with experiment, got {len(dataset_splits_edges)}"
        )

        associated_split = dataset_splits_edges[0]["node"]
        assert associated_split["id"] == train_split_id, (
            f"Expected train split {train_split_id}, got {associated_split['id']}"
        )
        assert associated_split["name"] == train_split_name, (
            f"Expected train split name {train_split_name}, got {associated_split['name']}"
        )

        # Test retrieving the experiment and verifying it contains split information
        retrieved_experiment = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.get_experiment(
                experiment_id=experiment_id
            )
        )

        assert retrieved_experiment["experiment_id"] == experiment_id
        assert retrieved_experiment["dataset_id"] == dataset.id
        assert len(retrieved_experiment["task_runs"]) == 2

        # Now test running an experiment on multiple splits
        both_splits_dataset = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).datasets.get_dataset(
                dataset=dataset.id,
                splits=[train_split_name, test_split_name],
            )
        )

        assert len(both_splits_dataset) == 4, (
            f"Expected 4 examples with both splits, got {len(both_splits_dataset)}"
        )
        assert train_split_name in both_splits_dataset._filtered_split_names
        assert test_split_name in both_splits_dataset._filtered_split_names

        # Run experiment on dataset with both splits
        multi_split_result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).experiments.run_experiment(
                dataset=both_splits_dataset,
                task=simple_task,
                experiment_name=f"test_multi_split_experiment_{token_hex(4)}",
                experiment_description="Test experiment with multiple dataset splits",
                print_summary=False,
            )
        )

        assert len(multi_split_result["task_runs"]) == 4, (
            f"Expected 4 task runs (both splits), got {len(multi_split_result['task_runs'])}"
        )

        multi_split_exp_id = multi_split_result["experiment_id"]

        # Verify both splits are associated with the experiment
        multi_splits_verification, _ = _gql(
            _app,
            _app.admin_secret,
            query=verify_splits_query,
            variables={"experimentId": multi_split_exp_id},
        )

        multi_exp_node = multi_splits_verification["data"]["node"]
        multi_splits_edges = multi_exp_node["datasetSplits"]["edges"]
        assert len(multi_splits_edges) == 2, (
            f"Expected 2 splits associated with experiment, got {len(multi_splits_edges)}"
        )

        # Verify both split IDs are present
        associated_split_ids = {edge["node"]["id"] for edge in multi_splits_edges}
        assert train_split_id in associated_split_ids, (
            f"Train split {train_split_id} should be in associated splits"
        )
        assert test_split_id in associated_split_ids, (
            f"Test split {test_split_id} should be in associated splits"
        )


# --- tests/integration/client/test_prompts.py ---

    def test_openai(
        self,
        types_: Sequence[type[BaseModel]],
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret
        expected: Mapping[str, ChatCompletionToolParam] = {
            t.__name__: cast(
                ChatCompletionToolParam, json.loads(json.dumps(pydantic_function_tool(t)))
            )
            for t in types_
        }
        tools = PromptToolsInput(
            tools=[
                PromptToolFunctionInput(
                    function=PromptToolFunctionDefinitionInput(
                        name=v["function"]["name"],
                        description=v["function"].get("description"),
                        parameters=v["function"].get("parameters"),
                        strict=v["function"].get("strict"),
                    )
                )
                for v in expected.values()
            ]
        )
        prompt = _create_chat_prompt(_app, api_key, tools=tools)
        kwargs = prompt.format().kwargs
        assert "tools" in kwargs
        actual: dict[str, ChatCompletionToolParam] = {
            t["function"]["name"]: t
            for t in cast(Iterable[ChatCompletionToolParam], kwargs["tools"])
            if t["type"] == "function" and "parameters" in t["function"]
        }
        assert not DeepDiff(expected, actual)
        _can_recreate_via_client(_app, prompt, api_key)

    def test_openai(
        self,
        expected: ChatCompletionToolChoiceOptionParam,
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret
        tools = PromptToolsInput(
            tools=[
                PromptToolFunctionInput(
                    function=PromptToolFunctionDefinitionInput(
                        name=t["function"]["name"],
                        description=t["function"].get("description"),
                        parameters=t["function"].get("parameters"),
                    )
                )
                for t in [
                    json.loads(json.dumps(pydantic_function_tool(cls)))
                    for cls in cast(Iterable[type[BaseModel]], [_GetWeather, _GetPopulation])
                ]
            ],
            toolChoice=_openai_tool_choice_to_canonical(expected),
        )
        prompt = _create_chat_prompt(_app, api_key, tools=tools)
        kwargs = prompt.format().kwargs
        assert "tool_choice" in kwargs
        actual = kwargs["tool_choice"]
        assert not DeepDiff(expected, actual)
        _can_recreate_via_client(_app, prompt, api_key)

    def test_client(
        self,
        _get_user: _GetUser,
        _app: _AppInfo,
    ) -> None:
        u = _get_user(_app, _MEMBER).log_in(_app)
        api_key = str(u.create_api_key(_app))
        prompt = _PhoenixClient(base_url=_app.base_url, api_key=api_key).prompts.create(
            name=token_hex(8),
            version=PromptVersion.from_openai(
                CompletionCreateParamsBase(
                    model=token_hex(8), messages=[{"role": "user", "content": "hello"}]
                )
            ),
        )
        response, _ = u.gql(_app, query=self.QUERY, variables={"versionId": prompt.id})
        assert u.gid == response["data"]["node"]["user"]["id"]

    async def test_create_and_retrieve_metadata(
        self,
        is_async: bool,
        _app: _AppInfo,
    ) -> None:
        """Test that metadata can be created and retrieved for prompts."""
        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient

        # Create prompt with metadata
        prompt_name = token_hex(8)
        prompt_description = token_hex(8)
        prompt_metadata = {"environment": token_hex(8)}
        await _await_or_return(
            Client(base_url=_app.base_url, api_key=_app.admin_secret).prompts.create(
                name=prompt_name,
                version=PromptVersion.from_openai(
                    CompletionCreateParamsBase(
                        model=token_hex(8), messages=[{"role": "user", "content": "hello"}]
                    )
                ),
                prompt_description=prompt_description,
                prompt_metadata=prompt_metadata,
            )
        )

        # Query prompt metadata via GraphQL
        query = """
        query($name: String!) {
            prompts(first: 1, filter: {col: name, value: $name}) {
                edges {
                    node {
                        id
                        metadata
                        description
                    }
                }
            }
        }
        """
        response, _ = _gql(_app, _app.admin_secret, query=query, variables={"name": prompt_name})
        assert response["data"]["prompts"]["edges"]
        retrieved_metadata = response["data"]["prompts"]["edges"][0]["node"]["metadata"]
        assert retrieved_metadata == prompt_metadata
        assert response["data"]["prompts"]["edges"][0]["node"]["description"] == prompt_description

    def test_round_trip(
        self,
        model_providers: str,
        convert: Callable[..., PromptVersion],
        expected: dict[str, Any],
        template_format: Literal["F_STRING", "MUSTACHE", "NONE"],
        _app: _AppInfo,
    ) -> None:
        api_key = _app.admin_secret
        prompt_identifier = token_hex(16)
        from phoenix.client import Client

        client = Client(base_url=_app.base_url, api_key=api_key)
        for model_provider in model_providers.split(","):
            version: PromptVersion = convert(
                expected,
                template_format=template_format,
                model_provider=model_provider,
            )
            client.prompts.create(
                name=prompt_identifier,
                version=version,
            )
            prompt = client.prompts.get(prompt_identifier=prompt_identifier)
            assert prompt._model_provider == model_provider
            assert prompt._template_format == template_format
            params = prompt.format(formatter=NO_OP_FORMATTER)
            assert not DeepDiff(expected, {**params})


# --- tests/integration/client/test_secrets.py ---

    def test_graphql_empty_secrets_list_returns_error(self, _app: _AppInfo) -> None:
        with pytest.raises(RuntimeError, match="At least one secret is required"):
            _gql(
                _app,
                _app.admin_secret,
                query=UPSERT_OR_DELETE_MUTATION,
                variables={"input": {"secrets": []}},
                operation_name="UpsertOrDeleteSecrets",
            )

    def test_graphql_empty_key_returns_error(self, _app: _AppInfo) -> None:
        with pytest.raises(RuntimeError, match="Key cannot be empty"):
            _gql(
                _app,
                _app.admin_secret,
                query=UPSERT_OR_DELETE_MUTATION,
                variables={"input": {"secrets": [{"key": "", "value": "v"}]}},
                operation_name="UpsertOrDeleteSecrets",
            )

    def test_graphql_key_outside_regex_returns_error(self, _app: _AppInfo) -> None:
        with pytest.raises(
            RuntimeError,
            match=(
                "Key must start with a letter or underscore and contain only "
                "letters, digits, and underscores"
            ),
        ):
            _gql(
                _app,
                _app.admin_secret,
                query=UPSERT_OR_DELETE_MUTATION,
                variables={"input": {"secrets": [{"key": "has-dash", "value": "v"}]}},
                operation_name="UpsertOrDeleteSecrets",
            )

    def test_graphql_empty_value_returns_error(self, _app: _AppInfo) -> None:
        with pytest.raises(RuntimeError, match="Value cannot be empty"):
            _gql(
                _app,
                _app.admin_secret,
                query=UPSERT_OR_DELETE_MUTATION,
                variables={"input": {"secrets": [{"key": "k", "value": ""}]}},
                operation_name="UpsertOrDeleteSecrets",
            )


# --- tests/integration/client/test_spans.py ---

    async def test_unflatten_attributes_on_span_creation(
        self,
        is_async: bool,
        _existing_project: _ExistingProject,
        _app: _AppInfo,
    ) -> None:
        """Test that flattened attributes are properly unflattened when creating spans.

        This test verifies that the unflatten() function correctly converts
        flattened dot-separated keys into nested structures when spans are created.
        It uses GraphQL to check the raw database structure (nested) and the REST API
        to verify round-trip behavior (flattened).
        """
        api_key = _app.admin_secret

        from phoenix.client import AsyncClient
        from phoenix.client import Client as SyncClient

        Client = AsyncClient if is_async else SyncClient  # type: ignore[unused-ignore]

        project_name = _existing_project.name
        trace_id = f"trace_{token_hex(16)}"
        span_id = f"span_{token_hex(8)}"

        # Create a span with MIX of flattened and already-nested attributes
        # This tests robustness of unflatten() handling both formats
        test_span = self._create_test_span(
            "test_unflatten",
            context={"trace_id": trace_id, "span_id": span_id},
            attributes={
                # Flattened attributes
                "llm.model": "gpt-4",
                "llm.token_count.prompt": 100,
                "llm.token_count.completion": 50,
                "llm.token_count.total": 150,
                # Already nested attribute (should be preserved as-is)
                "metadata": {
                    "user": {"id": "user123", "name": "Test User"},
                    "session": {"id": "session456"},
                },
                # Array-like structure (numeric keys with nested content)
                "documents.0.content": "First document",
                "documents.0.id": "doc1",
                "documents.1.content": "Second document",
                "documents.1.id": "doc2",
            },
        )

        # Create the span
        result = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).spans.log_spans(  # pyright: ignore[reportAttributeAccessIssue]
                project_identifier=project_name,
                spans=[test_span],
            )
        )
        assert result["total_received"] == 1
        assert result["total_queued"] == 1

        # Wait for span to be processed
        await _until_spans_exist(_app, [span_id])

        # Use GraphQL to check the nested structure in the database
        gql_query = """
        query GetSpanAttributes($spanId: ID!) {
            node(id: $spanId) {
                ... on Span {
                    spanId
                    attributes
                }
            }
        }
        """

        # Get the span's Global ID using GraphQL - query project directly by its Global ID
        trace_query = """
        query GetTraceSpans($projectId: ID!, $traceId: ID!) {
            node(id: $projectId) {
                ... on Project {
                    trace(traceId: $traceId) {
                        spans(first: 1000) {
                            edges {
                                node {
                                    id
                                    spanId
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        trace_result, _ = _gql(
            _app,
            _app.admin_secret,
            query=trace_query,
            variables={"projectId": str(_existing_project.id), "traceId": trace_id},
        )
        assert not trace_result.get("errors"), f"GraphQL errors: {trace_result.get('errors')}"

        # Find our span's Global ID
        trace = trace_result["data"]["node"]["trace"]
        assert trace is not None, f"Could not find trace with traceId {trace_id}"

        span_global_id = None
        for span_edge in trace["spans"]["edges"]:
            if span_edge["node"]["spanId"] == span_id:
                span_global_id = span_edge["node"]["id"]
                break

        assert span_global_id is not None, f"Could not find span with spanId {span_id}"

        # Now query for the attributes using the Global ID
        attr_result, _ = _gql(
            _app, _app.admin_secret, query=gql_query, variables={"spanId": span_global_id}
        )
        assert not attr_result.get("errors"), f"GraphQL errors: {attr_result.get('errors')}"

        # Get the nested attributes from GraphQL (as stored in the database)
        # GraphQL returns the attributes as a JSON string, so we need to parse it
        db_attrs_json = attr_result["data"]["node"]["attributes"]
        db_attrs = json.loads(db_attrs_json)

        # Verify the attributes are stored in nested structure in the database
        assert isinstance(db_attrs, dict)

        # Check OpenInference attributes
        assert "openinference" in db_attrs
        assert isinstance(db_attrs["openinference"], dict)
        assert "span" in db_attrs["openinference"]
        assert isinstance(db_attrs["openinference"]["span"], dict)
        assert "kind" in db_attrs["openinference"]["span"]
        assert db_attrs["openinference"]["span"]["kind"] == "CHAIN"

        # Check LLM attributes
        assert "llm" in db_attrs
        assert isinstance(db_attrs["llm"], dict)
        assert db_attrs["llm"]["model"] == "gpt-4"
        assert "token_count" in db_attrs["llm"]
        assert isinstance(db_attrs["llm"]["token_count"], dict)
        assert db_attrs["llm"]["token_count"]["prompt"] == 100
        assert db_attrs["llm"]["token_count"]["completion"] == 50
        assert db_attrs["llm"]["token_count"]["total"] == 150

        # Check already-nested attributes are preserved
        assert "metadata" in db_attrs
        assert isinstance(db_attrs["metadata"], dict)
        assert "user" in db_attrs["metadata"]
        assert isinstance(db_attrs["metadata"]["user"], dict)
        assert db_attrs["metadata"]["user"]["id"] == "user123"
        assert db_attrs["metadata"]["user"]["name"] == "Test User"
        assert "session" in db_attrs["metadata"]
        assert isinstance(db_attrs["metadata"]["session"], dict)
        assert db_attrs["metadata"]["session"]["id"] == "session456"

        # Check array-like structures are converted to arrays
        assert "documents" in db_attrs
        assert isinstance(db_attrs["documents"], list)
        assert len(db_attrs["documents"]) == 2
        assert db_attrs["documents"][0]["content"] == "First document"
        assert db_attrs["documents"][0]["id"] == "doc1"
        assert db_attrs["documents"][1]["content"] == "Second document"
        assert db_attrs["documents"][1]["id"] == "doc2"

        # Now verify the REST API round-trip (should be flattened on retrieval)
        spans = await _await_or_return(
            Client(base_url=_app.base_url, api_key=api_key).spans.get_spans(
                project_identifier=project_name,
                limit=100,
            )
        )

        # Find our test span
        test_span_retrieved = None
        for span in spans:
            if span["context"]["span_id"] == span_id:
                test_span_retrieved = span
                break

        assert test_span_retrieved is not None, "Test span should be found in retrieved spans"

        # Verify the attributes are flattened when retrieved via REST API
        assert "attributes" in test_span_retrieved
        rest_attrs = test_span_retrieved["attributes"]
        assert isinstance(rest_attrs, dict)

        # These should be flattened (dot-separated keys)
        assert "llm.model" in rest_attrs
        assert rest_attrs["llm.model"] == "gpt-4"
        assert "llm.token_count.prompt" in rest_attrs
        assert rest_attrs["llm.token_count.prompt"] == 100
        assert "llm.token_count.completion" in rest_attrs
        assert rest_attrs["llm.token_count.completion"] == 50

        # Verify originally-nested metadata is flattened in REST response
        assert "metadata.user.id" in rest_attrs
        assert rest_attrs["metadata.user.id"] == "user123"
        assert "metadata.session.id" in rest_attrs
        assert rest_attrs["metadata.session.id"] == "session456"

        # Verify documents array is flattened back to dotted keys
        assert "documents.0.content" in rest_attrs
        assert rest_attrs["documents.0.content"] == "First document"
        assert "documents.0.id" in rest_attrs
        assert rest_attrs["documents.0.id"] == "doc1"
        assert "documents.1.content" in rest_attrs
        assert rest_attrs["documents.1.content"] == "Second document"


# --- tests/integration/client/test_span_query.py ---

def test_backward_compatibility() -> None:
    query = SpanQuery().select(
        "context.span_id",
        "context.trace_id",
        "cumulative_token_count.completion",
    )

    # The query should internally convert to new field names
    query_dict = query.to_dict()
    assert "select" in query_dict
    select_dict = query_dict["select"]
    assert "span_id" in select_dict
    assert "trace_id" in select_dict
    assert "cumulative_llm_token_count_completion" in select_dict


# --- tests/integration/db_migrations/test_data_migration_4ded9e43755f_create_project_sessions_table.py ---

async def test_data_migration_for_project_sessions(
    _engine: AsyncEngine,
    _alembic_config: Config,
    _schema: str,
) -> None:
    with pytest.raises(BaseException, match="alembic_version"):
        await _version_num(_engine, _schema)

    await _up(_engine, _alembic_config, "cd164e83824f", _schema)

    def _reflect_tables(conn: Connection) -> tuple[Table, Table, Table]:
        metadata = MetaData()
        metadata.reflect(bind=conn)
        t_projects = metadata.tables["projects"]
        t_traces = metadata.tables["traces"]
        t_spans = Table(
            "spans",
            MetaData(),
            Column("attributes", JSON_),
            Column("events", JSON_),
            autoload_with=conn,
        )
        return t_projects, t_traces, t_spans

    table_projects, table_traces, table_spans = await _run_async(_engine, _reflect_tables)

    def time_gen(
        t: datetime,
        delta: timedelta = timedelta(seconds=10),
    ) -> Iterator[datetime]:
        while True:
            yield t
            t += delta

    gen_time = time_gen(datetime.now(timezone.utc))

    def rand_id_gen() -> Iterator[Union[str, int]]:
        while True:
            yield token_urlsafe(16)
            yield int.from_bytes(token_bytes(4), "big")

    gen_session_id = rand_id_gen()
    gen_user_id = rand_id_gen()

    def rand_session_attr() -> dict[str, Any]:
        return {"session": {"id": next(gen_session_id)}, "user": {"id": next(gen_user_id)}}

    num_project_sessions = 7
    num_projects = 5
    num_traces_per_project = 11
    num_spans_per_trace = 3

    session_attrs = [rand_session_attr() for _ in range(num_project_sessions)]
    session_attrs_iter = cycle(session_attrs)

    def get_spans(traces: Iterable[tuple[int, datetime]]) -> Iterator[dict[str, Any]]:
        for trace_rowid, start_time in traces:
            t = time_gen(start_time)
            parent_id = None
            for _ in range(num_spans_per_trace):
                # session attributes on non-root spans should be ignored
                attributes = rand_session_attr() if parent_id else next(session_attrs_iter)
                span_id = token_hex(8)
                yield {
                    "span_id": span_id,
                    "parent_id": parent_id,
                    "name": token_urlsafe(16),
                    "span_kind": token_urlsafe(16),
                    "trace_rowid": trace_rowid,
                    "start_time": next(t) if parent_id else start_time,
                    "end_time": next(t),
                    "status_message": token_urlsafe(16),
                    "cumulative_error_count": 0,
                    "cumulative_llm_token_count_prompt": 0,
                    "cumulative_llm_token_count_completion": 0,
                    "events": [],
                    "attributes": attributes,
                }
                parent_id = span_id

    def _insert_data(conn: Connection) -> None:
        project_rowids = conn.scalars(
            insert(table_projects).returning(table_projects.c.id),
            [{"name": token_urlsafe(16)} for _ in range(num_projects)],
        ).all()
        traces = cast(
            Sequence[tuple[int, datetime]],
            conn.execute(
                insert(table_traces).returning(
                    table_traces.c.id,
                    table_traces.c.start_time,
                ),
                [
                    {
                        "trace_id": token_hex(16),
                        "project_rowid": project_rowid,
                        "start_time": next(gen_time),
                        "end_time": next(gen_time),
                    }
                    for _ in range(num_traces_per_project)
                    for project_rowid in project_rowids
                ],
            ).all(),
        )
        conn.execute(insert(table_spans), list(get_spans(traces)))
        conn.commit()

    await _run_async(_engine, _insert_data)

    for _ in range(2):
        await _down(_engine, _alembic_config, "cd164e83824f", _schema)

        def _check_no_project_sessions(conn: Connection) -> None:
            metadata = MetaData()
            metadata.reflect(bind=conn)
            assert metadata.tables.get("project_sessions") is None

        await _run_async(_engine, _check_no_project_sessions)
        await _up(_engine, _alembic_config, "4ded9e43755f", _schema)

        def _populate(conn: Connection) -> None:
            populate_project_sessions(conn)

        await _run_async(_engine, _populate)

        def _read_tables(conn: Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            df_spans = pd.read_sql_table("spans", conn)
            df_traces = pd.read_sql_table("traces", conn)
            df_project_sessions = pd.read_sql_table("project_sessions", conn)
            return df_spans, df_traces, df_project_sessions

        df_spans, df_traces, df_project_sessions = await _run_async(_engine, _read_tables)
        # Set index after reading since read_sql_table with index_col
        # may not work consistently across sync/async
        df_spans = df_spans.set_index("id")
        df_traces = df_traces.set_index("id")
        df_project_sessions = df_project_sessions.set_index("id")

        assert len(df_project_sessions) == num_project_sessions
        assert len(df_traces) == num_projects * num_traces_per_project
        assert len(df_spans) == len(df_traces) * num_spans_per_trace

        assert df_project_sessions.session_id.nunique() == num_project_sessions
        assert df_traces.project_session_rowid.nunique() == num_project_sessions

        df_span_session_attrs = df_spans.apply(
            lambda row: pd.Series(
                {
                    "trace_rowid": row["trace_rowid"],
                    "span_id": row["span_id"],
                    "parent_id": row["parent_id"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "session_id": str(
                        (
                            json.loads(row["attributes"])  # type: ignore[unused-ignore]
                            if _engine.dialect.name == "sqlite"
                            else row["attributes"]
                        )["session"]["id"]
                    ),  # type: ignore[dict-item, unused-ignore]
                },
            ),
            axis=1,
        )
        assert sum(df_span_session_attrs.session_id.isna()) == 0

        df_traces_joined_spans = pd.merge(
            df_traces.loc[:, ["project_session_rowid", "start_time", "end_time", "project_rowid"]],
            df_span_session_attrs.loc[df_span_session_attrs.parent_id.isna()],
            how="left",
            left_index=True,
            right_on="trace_rowid",
            suffixes=("_trace", ""),
        )
        df_project_sessions_joined_spans = pd.merge(
            df_project_sessions,
            df_traces_joined_spans,
            how="left",
            left_index=True,
            right_on="project_session_rowid",
            suffixes=("", "_span"),
        ).sort_values(["session_id", "start_time_trace"])

        assert df_project_sessions_joined_spans.span_id.nunique() == len(
            df_project_sessions_joined_spans
        )
        assert (
            df_project_sessions_joined_spans.session_id
            == df_project_sessions_joined_spans.session_id_span
        ).all()
        assert (
            df_project_sessions_joined_spans.groupby("session_id")
            .apply(lambda s: s.end_time.min() == s.end_time_trace.max())  # type: ignore[attr-defined, unused-ignore]
            .all()
        )

        is_first = df_project_sessions_joined_spans.groupby("session_id").cumcount() == 0

        assert (
            df_project_sessions_joined_spans.loc[is_first]
            .apply(lambda row: row.start_time == row.start_time_trace, axis=1)
            .all()
        )
        assert (
            df_project_sessions_joined_spans.loc[is_first]
            .apply(lambda row: row.project_id == row.project_rowid, axis=1)
            .all()
        )


# --- tests/integration/server/test_launch_app.py ---

    async def test_db_migrate(self, _env_sql_database: dict[str, str]) -> None:
        from pathlib import Path

        import sqlalchemy
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        import phoenix.db as _phoenix_db
        from phoenix.db.engines import get_async_db_url

        raw_url = _env_sql_database["PHOENIX_SQL_DATABASE_URL"]
        schema = _env_sql_database.get("PHOENIX_SQL_DATABASE_SCHEMA") or None
        url = sqlalchemy.make_url(raw_url)
        if url.get_backend_name() == "postgresql":
            async_url = get_async_db_url(url.render_as_string(hide_password=False))
            async_engine = create_async_engine(async_url, poolclass=NullPool)
        else:
            async_engine = create_async_engine(
                url.set(drivername="sqlite+aiosqlite"), poolclass=NullPool
            )

        # Verify the database is fresh: alembic_version must not exist yet.
        def _check_fresh(conn: sqlalchemy.Connection) -> None:
            inspector = sqlalchemy.inspect(conn)
            assert "alembic_version" not in inspector.get_table_names(schema=schema), (
                "alembic_version already exists before migration ran"
            )

        async with async_engine.connect() as conn:
            await conn.run_sync(_check_fresh)
        await async_engine.dispose()

        command = [sys.executable, "-m", "phoenix.server.main", "db", "migrate"]
        env = (
            {**os.environ, **_env_sql_database}
            if sys.platform == "win32"
            else dict(_env_sql_database)
        )
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

        # Confirm alembic_version now matches the current head revision.
        if url.get_backend_name() == "postgresql":
            async_engine = create_async_engine(
                get_async_db_url(url.render_as_string(hide_password=False)), poolclass=NullPool
            )
        else:
            async_engine = create_async_engine(
                url.set(drivername="sqlite+aiosqlite"), poolclass=NullPool
            )

        def _get_version(conn: sqlalchemy.Connection) -> Any:
            if schema:
                conn.execute(sqlalchemy.text(f'SET search_path TO "{schema}"'))
            return conn.execute(sqlalchemy.text("SELECT version_num FROM alembic_version")).scalar()

        async with async_engine.connect() as conn:
            actual = await conn.run_sync(_get_version)
        await async_engine.dispose()

        scripts_dir = str(Path(_phoenix_db.__file__).parent / "migrations")
        cfg = Config()
        cfg.set_main_option("script_location", scripts_dir)
        expected = ScriptDirectory.from_config(cfg).get_current_head()

        assert actual == expected, f"DB is at {actual!r}, expected head {expected!r}"


# --- tests/unit/test_tracers.py ---

    async def test_save_db_traces_persists_nested_spans(
        self, db: DbSessionFactory, project: models.Project, tracer: Tracer
    ) -> None:
        with tracer.start_as_current_span(
            "parent",
            attributes={OPENINFERENCE_SPAN_KIND: "CHAIN"},
        ) as parent_span:
            parent_span.set_attribute("custom_attr", "parent_value")
            with tracer.start_as_current_span(
                "child",
                attributes={OPENINFERENCE_SPAN_KIND: "LLM"},
            ) as child_span:
                child_span.set_attribute("custom_attr", "child_value")
                child_span.add_event("test_event", {"event_key": "event_value"})
                child_span.set_status(Status(StatusCode.OK))
            parent_span.set_status(Status(StatusCode.OK))

        async with db() as session:
            returned_traces = tracer.get_db_traces(project_id=project.id)
            session.add_all(returned_traces)
            await session.flush()
            fetched_traces = (
                (
                    await session.execute(
                        select(models.Trace).options(joinedload(models.Trace.spans))
                    )
                )
                .scalars()
                .unique()
                .all()
            )

        assert len(returned_traces) == 1
        assert len(fetched_traces) == 1

        returned_trace = returned_traces[0]
        fetched_trace = fetched_traces[0]
        assert returned_trace == fetched_trace
        assert returned_trace.project_rowid == project.id
        returned_spans = returned_trace.spans
        fetched_spans = fetched_trace.spans
        assert len(returned_spans) == 2
        assert len(fetched_spans) == 2
        parent_returned_span = next(s for s in returned_spans if s.name == "parent")
        child_returned_span = next(s for s in returned_spans if s.name == "child")
        parent_db_span = next(s for s in fetched_spans if s.name == "parent")
        child_db_span = next(s for s in fetched_spans if s.name == "child")
        assert parent_returned_span == parent_db_span
        assert child_returned_span == child_db_span

        # spans have the correct trace ID
        for returned_span in returned_spans:
            assert returned_span.trace_rowid == returned_trace.id

        # check parent span
        assert parent_db_span.parent_id is None
        assert parent_db_span.name == "parent"
        assert parent_db_span.span_kind == "CHAIN"
        assert parent_db_span.start_time <= parent_db_span.end_time
        assert parent_db_span.status_code == "OK"
        assert parent_db_span.status_message == ""
        assert not parent_db_span.events
        assert parent_db_span.cumulative_error_count == 0
        assert parent_db_span.cumulative_llm_token_count_prompt == 0
        assert parent_db_span.cumulative_llm_token_count_completion == 0
        assert parent_db_span.llm_token_count_prompt is None
        assert parent_db_span.llm_token_count_completion is None
        assert parent_db_span.attributes == {
            "openinference": {
                "span": {
                    "kind": "CHAIN",
                }
            },
            "custom_attr": "parent_value",
        }

        # check child span
        assert child_db_span.parent_id == parent_db_span.span_id
        assert child_db_span.name == "child"
        assert child_db_span.span_kind == "LLM"
        assert child_db_span.start_time <= child_db_span.end_time
        assert child_db_span.status_code == "OK"
        assert child_db_span.status_message == ""
        assert len(child_db_span.events) == 1
        event = child_db_span.events[0]

        event = child_db_span.events[0]
        assert event.pop("name") == "test_event"
        timestamp = event.pop("timestamp")
        assert isinstance(timestamp, str)
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"timestamp {timestamp!r} is not a valid ISO 8601 string")
        assert event.pop("attributes") == {"event_key": "event_value"}
        assert not event
        assert child_db_span.cumulative_error_count == 0
        assert child_db_span.cumulative_llm_token_count_prompt == 0
        assert child_db_span.cumulative_llm_token_count_completion == 0
        assert child_db_span.llm_token_count_prompt is None
        assert child_db_span.llm_token_count_completion is None
        assert child_db_span.attributes == {
            "openinference": {
                "span": {
                    "kind": "LLM",
                }
            },
            "custom_attr": "child_value",
        }

    async def test_save_db_traces_persists_events_and_exceptions(
        self, db: DbSessionFactory, project: models.Project, tracer: Tracer
    ) -> None:
        with pytest.raises(ValueError, match="Test error message"):
            with tracer.start_as_current_span(
                "span",
                attributes={OPENINFERENCE_SPAN_KIND: "CHAIN"},
            ):
                raise ValueError("Test error message")

        async with db() as session:
            returned_traces = tracer.get_db_traces(project_id=project.id)
            session.add_all(returned_traces)
            await session.flush()
            db_spans = (await session.execute(select(models.Span))).scalars().all()

        returned_spans = returned_traces[0].spans
        assert len(returned_spans) == 1
        assert len(db_spans) == 1
        returned_span = returned_spans[0]
        db_span = db_spans[0]
        assert returned_span == db_span

        # check span fields
        assert returned_span.parent_id is None
        assert returned_span.name == "span"
        assert returned_span.span_kind == "CHAIN"
        assert returned_span.start_time <= returned_span.end_time
        assert returned_span.status_code == "ERROR"
        assert returned_span.status_message == "ValueError: Test error message"
        assert returned_span.cumulative_error_count == 1
        assert returned_span.cumulative_llm_token_count_prompt == 0
        assert returned_span.cumulative_llm_token_count_completion == 0
        assert returned_span.llm_token_count_prompt is None
        assert returned_span.llm_token_count_completion is None
        assert returned_span.attributes == {
            "openinference": {
                "span": {
                    "kind": "CHAIN",
                }
            },
        }

        # check events
        events = returned_span.events
        assert len(events) == 1
        event = dict(events[0])
        assert event.pop("name") == "exception"
        timestamp = event.pop("timestamp")
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"timestamp {timestamp!r} is not a valid ISO 8601 string")
        event_attributes = event.pop("attributes")
        assert event_attributes.pop("exception.type") == "ValueError"
        assert event_attributes.pop("exception.message") == "Test error message"
        assert "Traceback" in event_attributes.pop("exception.stacktrace")
        assert event_attributes.pop("exception.escaped") == "False"
        assert not event_attributes
        assert not event

    async def test_save_db_traces_populates_llm_token_count_fields(
        self, db: DbSessionFactory, project: models.Project, tracer: Tracer
    ) -> None:
        prompt_tokens = 150
        completion_tokens = 75

        with tracer.start_as_current_span(
            "llm_call",
            attributes={
                OPENINFERENCE_SPAN_KIND: "LLM",
                LLM_TOKEN_COUNT_PROMPT: prompt_tokens,
                LLM_TOKEN_COUNT_COMPLETION: completion_tokens,
            },
        ) as span:
            span.set_status(Status(StatusCode.OK))

        async with db() as session:
            returned_traces = tracer.get_db_traces(project_id=project.id)
            session.add_all(returned_traces)
            await session.flush()
            fetched_spans = (await session.execute(select(models.Span))).scalars().all()

        returned_spans = returned_traces[0].spans
        assert len(returned_spans) == 1
        assert len(fetched_spans) == 1

        returned_span = returned_spans[0]
        fetched_span = fetched_spans[0]
        assert returned_span == fetched_span

        # Verify token count fields
        assert fetched_span.llm_token_count_prompt == prompt_tokens
        assert fetched_span.llm_token_count_completion == completion_tokens
        assert fetched_span.llm_token_count_total == prompt_tokens + completion_tokens
        assert fetched_span.cumulative_llm_token_count_prompt == prompt_tokens
        assert fetched_span.cumulative_llm_token_count_completion == completion_tokens
        assert fetched_span.cumulative_llm_token_count_total == prompt_tokens + completion_tokens

    async def test_save_db_traces_handles_llm_spans_without_token_counts(
        self, db: DbSessionFactory, project: models.Project, tracer: Tracer
    ) -> None:
        with tracer.start_as_current_span(
            "llm_call_no_tokens",
            attributes={
                OPENINFERENCE_SPAN_KIND: "LLM",
            },
        ) as span:
            span.set_status(Status(StatusCode.OK))

        async with db() as session:
            returned_traces = tracer.get_db_traces(project_id=project.id)
            session.add_all(returned_traces)
            await session.flush()
            fetched_spans = (await session.execute(select(models.Span))).scalars().all()

        returned_spans = returned_traces[0].spans
        assert len(returned_spans) == 1
        fetched_span = fetched_spans[0]

        assert fetched_span.llm_token_count_prompt is None
        assert fetched_span.llm_token_count_completion is None
        assert fetched_span.cumulative_llm_token_count_prompt == 0
        assert fetched_span.cumulative_llm_token_count_completion == 0
        assert fetched_span.llm_token_count_total == 0
        assert fetched_span.cumulative_llm_token_count_total == 0

    async def test_save_db_traces_correctly_computes_cumulative_counts(
        self, db: DbSessionFactory, project: models.Project, tracer: Tracer
    ) -> None:
        # Create a hierarchy:
        #   parent (no tokens, OK)
        #   ├── child1 (100 prompt + 50 completion, ERROR)
        #   └── child2 (no tokens, OK)
        #       └── grandchild (200 prompt + 75 completion, ERROR)
        #
        # --- Expected cumulative prompt token counts ---
        #   parent:     300 (sum of all descendants)
        #   child1:     100 (own tokens only)
        #   child2:     200 (grandchild's tokens)
        #   grandchild: 200 (own tokens only)
        #
        # --- Expected cumulative completion token counts ---
        #   parent:     125 (sum of all descendants)
        #   child1:     50  (own tokens only)
        #   child2:     75  (grandchild's tokens)
        #   grandchild: 75  (own tokens only)
        #
        # Expected cumulative error counts:
        #   parent: 2 errors (child1 + grandchild)
        #   child1: 1 error (own error)
        #   child2: 1 error (grandchild's error)
        #   grandchild: 1 error (own error)

        with tracer.start_as_current_span(
            "parent",
            attributes={OPENINFERENCE_SPAN_KIND: "CHAIN"},
        ) as parent:
            with tracer.start_as_current_span(
                "child1",
                attributes={
                    OPENINFERENCE_SPAN_KIND: "LLM",
                    LLM_TOKEN_COUNT_PROMPT: 100,
                    LLM_TOKEN_COUNT_COMPLETION: 50,
                },
            ) as child1:
                child1.set_status(Status(StatusCode.ERROR, "child1 failed"))

            with tracer.start_as_current_span(
                "child2",
                attributes={OPENINFERENCE_SPAN_KIND: "CHAIN"},
            ) as child2:
                with tracer.start_as_current_span(
                    "grandchild",
                    attributes={
                        OPENINFERENCE_SPAN_KIND: "LLM",
                        LLM_TOKEN_COUNT_PROMPT: 200,
                        LLM_TOKEN_COUNT_COMPLETION: 75,
                    },
                ) as grandchild:
                    grandchild.set_status(Status(StatusCode.ERROR, "grandchild failed"))
                child2.set_status(Status(StatusCode.OK))

            parent.set_status(Status(StatusCode.OK))

        async with db() as session:
            returned_traces = tracer.get_db_traces(project_id=project.id)
            session.add_all(returned_traces)
            await session.flush()
            fetched_spans = (await session.execute(select(models.Span))).scalars().all()

        returned_spans = returned_traces[0].spans
        assert len(returned_spans) == 4
        assert len(fetched_spans) == 4

        # Get spans by name
        parent_span = next(s for s in fetched_spans if s.name == "parent")
        child1_span = next(s for s in fetched_spans if s.name == "child1")
        child2_span = next(s for s in fetched_spans if s.name == "child2")
        grandchild_span = next(s for s in fetched_spans if s.name == "grandchild")

        # Verify parent cumulative includes all descendants
        assert parent_span.cumulative_error_count == 2
        assert parent_span.cumulative_llm_token_count_prompt == 300
        assert parent_span.cumulative_llm_token_count_completion == 125
        assert parent_span.cumulative_llm_token_count_total == 425
        assert parent_span.llm_token_count_prompt is None
        assert parent_span.llm_token_count_completion is None

        # Verify child1 has only its own counts
        assert child1_span.cumulative_error_count == 1
        assert child1_span.cumulative_llm_token_count_prompt == 100
        assert child1_span.cumulative_llm_token_count_completion == 50
        assert child1_span.cumulative_llm_token_count_total == 150
        assert child1_span.llm_token_count_prompt == 100
        assert child1_span.llm_token_count_completion == 50

        # Verify child2 cumulative includes grandchild
        assert child2_span.cumulative_error_count == 1
        assert child2_span.cumulative_llm_token_count_prompt == 200
        assert child2_span.cumulative_llm_token_count_completion == 75
        assert child2_span.cumulative_llm_token_count_total == 275
        assert child2_span.llm_token_count_prompt is None
        assert child2_span.llm_token_count_completion is None

        # Verify grandchild has only its own counts
        assert grandchild_span.cumulative_error_count == 1
        assert grandchild_span.cumulative_llm_token_count_prompt == 200
        assert grandchild_span.cumulative_llm_token_count_completion == 75
        assert grandchild_span.cumulative_llm_token_count_total == 275
        assert grandchild_span.llm_token_count_prompt == 200
        assert grandchild_span.llm_token_count_completion == 75

    async def test_save_db_traces_calculates_costs_for_llm_spans(
        self,
        db: DbSessionFactory,
        project: models.Project,
        tracer: Tracer,
        gpt_4o_mini_generative_model: models.GenerativeModel,
    ) -> None:
        prompt_tokens = 1000
        completion_tokens = 500

        with tracer.start_as_current_span(
            "llm_call",
            attributes={
                OPENINFERENCE_SPAN_KIND: "LLM",
                LLM_MODEL_NAME: "gpt-4o-mini",
                LLM_PROVIDER: "openai",
                LLM_TOKEN_COUNT_PROMPT: prompt_tokens,
                LLM_TOKEN_COUNT_COMPLETION: completion_tokens,
            },
        ) as span:
            span.set_status(Status(StatusCode.OK))

        async with db() as session:
            returned_traces = tracer.get_db_traces(project_id=project.id)
            session.add_all(returned_traces)
            await session.flush()
            fetched_traces: Sequence[models.Trace] = (
                (
                    await session.scalars(
                        select(models.Trace).options(
                            joinedload(models.Trace.spans)
                            .joinedload(models.Span.span_cost)
                            .joinedload(models.SpanCost.span_cost_details)
                        )
                    )
                )
                .unique()
                .all()
            )

        # Ensure:
        # (1) orm relationships are properly set
        # (2) returned and fetched orms match
        assert len(returned_traces) == 1
        assert len(fetched_traces) == 1
        returned_trace = returned_traces[0]
        fetched_trace = fetched_traces[0]
        assert returned_trace == fetched_trace
        returned_spans = returned_trace.spans
        fetched_spans = fetched_trace.spans
        assert len(returned_spans) == 1
        assert len(fetched_spans) == 1
        returned_span = returned_spans[0]
        fetched_span = fetched_spans[0]
        assert returned_span == fetched_span
        returned_span_cost = returned_span.span_cost
        fetched_span_cost = fetched_span.span_cost
        assert returned_span_cost is not None
        assert fetched_span_cost is not None
        assert returned_span_cost == fetched_span_cost
        returned_span_cost_details = returned_span_cost.span_cost_details
        fetched_span_cost_details = fetched_span_cost.span_cost_details
        assert len(returned_span_cost_details) == 2
        assert len(fetched_span_cost_details) == 2
        returned_input_detail = next(d for d in returned_span_cost_details if d.is_prompt)
        returned_output_detail = next(d for d in returned_span_cost_details if not d.is_prompt)
        fetched_input_detail = next(d for d in fetched_span_cost_details if d.is_prompt)
        fetched_output_detail = next(d for d in fetched_span_cost_details if not d.is_prompt)
        assert returned_input_detail is not None
        assert fetched_input_detail is not None
        assert returned_output_detail is not None
        assert fetched_output_detail is not None
        assert returned_input_detail == fetched_input_detail
        assert returned_output_detail == fetched_output_detail

        # Verify span costs
        assert returned_span_cost.span_rowid == returned_span.id
        assert returned_span_cost.trace_rowid == returned_span.trace_rowid
        assert returned_span_cost.model_id == gpt_4o_mini_generative_model.id
        assert returned_span_cost.span_start_time == returned_span.start_time
        prompt_token_prices = next(
            p for p in gpt_4o_mini_generative_model.token_prices if p.is_prompt
        )
        completion_token_prices = next(
            p for p in gpt_4o_mini_generative_model.token_prices if not p.is_prompt
        )
        prompt_base_rate = prompt_token_prices.base_rate
        completion_base_rate = completion_token_prices.base_rate
        expected_prompt_cost = prompt_tokens * prompt_base_rate
        expected_completion_cost = completion_tokens * completion_base_rate
        expected_total_cost = expected_prompt_cost + expected_completion_cost
        assert expected_prompt_cost == pytest.approx(0.00015)  # (1000 * $0.15/1M) = $0.00015
        assert expected_completion_cost == pytest.approx(0.0003)  # (500 * $0.60/1M) = $0.0003
        assert expected_total_cost == pytest.approx(0.00045)  # $0.00015 + $0.0003 = $0.00045
        assert returned_span_cost.total_cost == pytest.approx(expected_total_cost)
        assert returned_span_cost.total_tokens == prompt_tokens + completion_tokens
        assert returned_span_cost.prompt_tokens == prompt_tokens
        assert returned_span_cost.prompt_cost == pytest.approx(expected_prompt_cost)
        assert returned_span_cost.completion_tokens == completion_tokens
        assert returned_span_cost.completion_cost == pytest.approx(expected_completion_cost)

        # Verify span cost details

        assert returned_input_detail.span_cost_id == returned_span_cost.id
        assert returned_input_detail.token_type == "input"
        assert returned_input_detail.is_prompt is True
        assert returned_input_detail.tokens == prompt_tokens
        assert returned_input_detail.cost == pytest.approx(expected_prompt_cost)
        assert returned_input_detail.cost_per_token == prompt_base_rate

        assert returned_output_detail.span_cost_id == returned_span_cost.id
        assert returned_output_detail.token_type == "output"
        assert returned_output_detail.is_prompt is False
        assert returned_output_detail.tokens == completion_tokens
        assert returned_output_detail.cost == pytest.approx(expected_completion_cost)
        assert returned_output_detail.cost_per_token == completion_base_rate

    async def test_save_db_traces_handles_missing_pricing_model(
        self, db: DbSessionFactory, project: models.Project, tracer: Tracer
    ) -> None:
        prompt_tokens = 100
        completion_tokens = 50

        with tracer.start_as_current_span(
            "llm_call",
            attributes={
                OPENINFERENCE_SPAN_KIND: "LLM",
                LLM_MODEL_NAME: "unknown-model",
                LLM_TOKEN_COUNT_PROMPT: prompt_tokens,
                LLM_TOKEN_COUNT_COMPLETION: completion_tokens,
            },
        ) as span:
            span.set_status(Status(StatusCode.OK))

        async with db() as session:
            returned_traces = tracer.get_db_traces(project_id=project.id)
            session.add_all(returned_traces)
            await session.flush()
            fetched_traces = (
                (
                    await session.execute(
                        select(models.Trace).options(
                            joinedload(models.Trace.spans)
                            .joinedload(models.Span.span_cost)
                            .joinedload(models.SpanCost.span_cost_details)
                        )
                    )
                )
                .scalars()
                .unique()
                .all()
            )

        # Ensure:
        # (1) orm relationships are properly set
        # (2) returned and fetched orms match
        assert len(returned_traces) == 1
        assert len(fetched_traces) == 1
        returned_trace = returned_traces[0]
        fetched_trace = fetched_traces[0]
        assert returned_trace == fetched_trace
        returned_spans = returned_trace.spans
        fetched_spans = fetched_trace.spans
        assert len(returned_spans) == 1
        assert len(fetched_spans) == 1
        returned_span = returned_spans[0]
        fetched_span = fetched_spans[0]
        assert returned_span == fetched_span
        returned_span_cost = returned_span.span_cost
        fetched_span_cost = fetched_span.span_cost
        assert returned_span_cost is not None
        assert fetched_span_cost is not None
        assert returned_span_cost == fetched_span_cost
        returned_span_cost_details = returned_span_cost.span_cost_details
        fetched_span_cost_details = fetched_span_cost.span_cost_details
        assert len(returned_span_cost_details) == 2
        assert len(fetched_span_cost_details) == 2
        returned_input_detail = next(d for d in returned_span_cost_details if d.is_prompt)
        returned_output_detail = next(d for d in returned_span_cost_details if not d.is_prompt)
        fetched_input_detail = next(d for d in fetched_span_cost_details if d.is_prompt)
        fetched_output_detail = next(d for d in fetched_span_cost_details if not d.is_prompt)
        assert returned_input_detail is not None
        assert fetched_input_detail is not None
        assert returned_output_detail is not None
        assert fetched_output_detail is not None
        assert returned_input_detail == fetched_input_detail
        assert returned_output_detail == fetched_output_detail

        # Verify span costs
        assert returned_span_cost.span_rowid == returned_span.id
        assert returned_span_cost.trace_rowid == returned_span.trace_rowid
        assert returned_span_cost.model_id is None  # no pricing model found
        assert returned_span_cost.span_start_time == returned_span.start_time
        assert returned_span_cost.total_cost is None  # no pricing model found
        assert returned_span_cost.total_tokens == prompt_tokens + completion_tokens
        assert returned_span_cost.prompt_tokens == prompt_tokens
        assert returned_span_cost.prompt_cost is None  # no pricing model found
        assert returned_span_cost.completion_tokens == completion_tokens
        assert returned_span_cost.completion_cost is None  # no pricing model found

        # Verify span cost details
        assert returned_input_detail.span_cost_id == returned_span_cost.id
        assert returned_input_detail.token_type == "input"
        assert returned_input_detail.is_prompt is True
        assert returned_input_detail.tokens == prompt_tokens
        assert returned_input_detail.cost is None  # no pricing model found
        assert returned_input_detail.cost_per_token is None  # no pricing model found

        assert returned_output_detail.span_cost_id == returned_span_cost.id
        assert returned_output_detail.token_type == "output"
        assert returned_output_detail.is_prompt is False
        assert returned_output_detail.tokens == completion_tokens
        assert returned_output_detail.cost is None  # no pricing model found
        assert returned_output_detail.cost_per_token is None  # no pricing model found


# --- tests/unit/db/test_helpers.py ---

    async def test_group_by(
        self,
        db: DbSessionFactory,
        _projects: list[models.Project],
    ) -> None:
        df = pd.DataFrame({"timestamp": [p.created_at for p in _projects]}).sort_values("timestamp")
        for field, utc_offset_minutes in cast(
            Iterable[tuple[Literal["minute", "hour", "day", "week", "month", "year"], int]],
            itertools.product(
                ["minute", "hour", "day", "week", "month", "year"],
                [-720, -60, -45, -30, -15, 0, 15, 30, 45, 60, 720],
            ),
        ):
            # Calculate expected buckets using pandas (same logic as SQL function)
            expected_summary = self._count_rows(df, field, utc_offset_minutes)

            # Generate SQL expressions and execute query
            start = date_trunc(
                dialect=db.dialect,
                field=field,
                source=models.Project.created_at,
                utc_offset_minutes=utc_offset_minutes,
            )

            stmt = sa.select(start, func.count(models.Project.id)).group_by(start).order_by(start)

            async with db() as session:
                rows = (await session.execute(stmt)).all()

            actual_summary = pd.DataFrame(rows, columns=["timestamp", "count"])

            if db.dialect is SupportedSQLDialect.SQLITE:
                # SQLite returns timestamps as strings, convert to datetime
                actual_summary["timestamp"] = pd.to_datetime(
                    actual_summary["timestamp"]
                ).dt.tz_localize(timezone.utc)

            # Verify SQL results match pandas calculation
            try:
                pd.testing.assert_frame_equal(actual_summary, expected_summary, check_dtype=False)
            except AssertionError:
                test_desc = (
                    f"Failed for field={field}, utc_offset_minutes={utc_offset_minutes}, "
                    f"dialect={db.dialect}"
                )
                raise AssertionError(f"Failed {test_desc}")

    async def test_select_date_trunc(
        self,
        db: DbSessionFactory,
        timestamp: str,
        expected: str,
        field: Literal["minute", "hour", "day", "week", "month", "year"],
        utc_offset_minutes: int,
    ) -> None:
        # Convert string inputs to datetime objects
        timestamp_dt = datetime.fromisoformat(timestamp)
        expected_dt = datetime.fromisoformat(expected)

        stmt = sa.select(
            date_trunc(
                db.dialect,
                field,
                sa.text(":dt").bindparams(dt=timestamp_dt),
                utc_offset_minutes,
            )
        )

        async with db() as session:
            actual = await session.scalar(stmt)
        assert actual is not None
        if db.dialect is SupportedSQLDialect.SQLITE:
            # SQLite returns timestamps as strings, convert to datetime
            assert isinstance(actual, str)
            actual = normalize_datetime(datetime.fromisoformat(actual), timezone.utc)
        assert actual == expected_dt


# --- tests/unit/db/test_models.py ---

    async def test_comprehensive_orjson_serialization(
        self,
        db: DbSessionFactory,
    ) -> None:
        """Test comprehensive orjson serialization of all special object types across JSON columns.

        This single comprehensive test validates all serialization scenarios:
        1. Special objects via _default function (numpy, datetime, enum)
        2. NaN/Inf sanitization to null values
        3. Cross-dialect compatibility (SQLite vs PostgreSQL)
        4. Both raw SQL reads and ORM reads show correct conversions

        Covers all JsonDict/JsonList columns: span attributes/events + all metadata_ fields.
        More efficient than separate tests since all scenarios use the same serialization pipeline.
        """
        from datetime import timezone
        from enum import Enum

        import numpy as np

        # Define test enums
        class Status(Enum):
            PENDING = "pending"
            ACTIVE = "active"
            INACTIVE = "inactive"

        class Priority(Enum):
            LOW = 1
            MEDIUM = 2
            HIGH = 3

        # Single comprehensive payload testing ALL serialization scenarios:
        # 1. Numpy arrays/scalars (including edge case dtypes) 2. Datetime objects 3. Enum objects 4. NaN/Inf values
        test_datetime = datetime(2023, 12, 25, 10, 30, 45, 123456, timezone.utc)
        comprehensive_attrs = {
            # Numpy arrays and scalars (consolidated edge case dtypes)
            "numpy_array_1d": np.array([1, 2, 3, 4]),
            "numpy_array_2d": np.array([[1, 2], [3, 4]]),
            "numpy_array_empty": np.array([]),
            "numpy_scalar_int": np.int64(42),
            "numpy_scalar_float": np.float64(3.14159),
            "numpy_scalar_bool": np.bool_(True),
            # Edge case numpy dtypes (consolidated from test_numpy_edge_cases)
            "numpy_dtypes": {
                "int8": np.int8(127),
                "int32": np.int32(2147483647),
                "uint16": np.uint16(65535),
                "float16": np.float16(3.14),
                "float32": np.float32(2.718),
                "bool_array": np.array([True, False]),
                "mixed_array": np.array([1, 2.5, 3]),  # Will be converted to float array
            },
            # Datetime and enum objects
            "status_enum": Status.ACTIVE,
            "priority_enum": Priority.HIGH,
            "timestamp_dt": test_datetime,
            # NaN/Inf values (consolidated comprehensive coverage from test_json_sanitization)
            "nan_value": float("nan"),
            "inf_value": float("inf"),
            "neg_inf_value": float("-inf"),
            # Complex nested structures combining ALL types
            "complex_nested": {
                "numpy_arrays": [np.array([5, 6]), {"nested_array": np.array([[7, 8], [9, 10]])}],
                "enums_datetimes": [Status.PENDING, test_datetime, Priority.LOW],
                "nan_inf_mixed": [1, float("nan"), {"inf_nested": float("inf")}, [float("-inf")]],
                "regular_python": {"normal": [1, 2, 3], "string": "test", "bool": False},
                # Deep NaN/Inf nesting (consolidated edge cases)
                "deep_nan_structure": {
                    "level1": [float("nan"), {"level2": [float("inf"), float("-inf")]}],
                    "mixed_arrays": [1, None, float("nan"), "NaN_string", True],
                },
            },
            # Edge cases
            "mixed_types_array": [
                np.int32(100),
                Status.INACTIVE,
                test_datetime,
                float("nan"),
                np.array([20, 30]),
                {"enum_key": Priority.MEDIUM, "nan_key": float("inf")},
            ],
        }

        # Expected converted values after all serialization transformations
        expected_converted = {
            # Numpy → Python native types
            "numpy_array_1d": [1, 2, 3, 4],
            "numpy_array_2d": [[1, 2], [3, 4]],
            "numpy_array_empty": [],
            "numpy_scalar_int": 42,
            "numpy_scalar_float": 3.14159,
            "numpy_scalar_bool": True,
            # Edge case dtypes → Python native types
            "numpy_dtypes": {
                "int8": 127,
                "int32": 2147483647,
                "uint16": 65535,
                "float16": float(np.float16(3.14)),
                "float32": float(np.float32(2.718)),
                "bool_array": [True, False],
                "mixed_array": [1.0, 2.5, 3.0],
            },
            # Datetime → ISO string, Enum → .value
            "status_enum": "active",
            "priority_enum": 3,
            "timestamp_dt": test_datetime.isoformat(),
            # NaN/Inf → null
            "nan_value": None,
            "inf_value": None,
            "neg_inf_value": None,
            # Complex nested with all conversions applied
            "complex_nested": {
                "numpy_arrays": [[5, 6], {"nested_array": [[7, 8], [9, 10]]}],
                "enums_datetimes": ["pending", test_datetime.isoformat(), 1],
                "nan_inf_mixed": [1, None, {"inf_nested": None}, [None]],
                "regular_python": {"normal": [1, 2, 3], "string": "test", "bool": False},
                # Deep NaN/Inf → null conversions
                "deep_nan_structure": {
                    "level1": [None, {"level2": [None, None]}],
                    "mixed_arrays": [1, None, None, "NaN_string", True],
                },
            },
            # Mixed edge cases with all conversions
            "mixed_types_array": [
                100,
                "inactive",
                test_datetime.isoformat(),
                None,
                [20, 30],
                {"enum_key": 2, "nan_key": None},
            ],
        }

        EVENT_NAME = "Comprehensive Serialization Test"
        EVENT_TS = "2022-04-29T18:52:58.114561Z"
        test_event = {"name": EVENT_NAME, "timestamp": EVENT_TS, "attributes": comprehensive_attrs}
        expected_event = {**test_event, "attributes": expected_converted}

        # === SINGLE EFFICIENT DATABASE SETUP ===
        # Create all entities once and test all serialization scenarios together
        async with db() as session:
            project = models.Project(name=token_hex(8))
            session.add(project)
            await session.flush()

            start_time = datetime.fromisoformat("2021-01-01T00:00:00.000+00:00")
            end_time = datetime.fromisoformat("2021-01-01T00:00:30.000+00:00")

            trace = models.Trace(
                project_rowid=project.id,
                trace_id=token_hex(8),
                start_time=start_time,
                end_time=end_time,
            )
            session.add(trace)
            await session.flush()

            span = models.Span(
                trace_rowid=trace.id,
                span_id=token_hex(8),
                name="comprehensive_serialization_test",
                span_kind="LLM",
                start_time=start_time,
                end_time=end_time,
                attributes=comprehensive_attrs,
                events=[test_event],
                status_code="OK",
                status_message="okay",
                cumulative_error_count=0,
                cumulative_llm_token_count_prompt=0,
                cumulative_llm_token_count_completion=0,
            )
            session.add(span)
            await session.flush()  # Flush span to get its ID

            # Create all metadata entities with the comprehensive payload
            session.add(
                models.SpanAnnotation(
                    span_rowid=span.id,
                    name="comprehensive_test",
                    annotator_kind="HUMAN",
                    source="APP",
                    metadata_=comprehensive_attrs,
                )
            )
            session.add(
                models.TraceAnnotation(
                    trace_rowid=trace.id,
                    name="comprehensive_test",
                    annotator_kind="HUMAN",
                    source="APP",
                    metadata_=comprehensive_attrs,
                )
            )
            session.add(
                models.DocumentAnnotation(
                    span_rowid=span.id,
                    document_position=0,
                    name="comprehensive_test",
                    annotator_kind="CODE",
                    source="APP",
                    metadata_=comprehensive_attrs,
                )
            )

            dataset = models.Dataset(
                name=f"comprehensive_ds_{token_hex(6)}", metadata_=comprehensive_attrs
            )
            session.add(dataset)
            await session.flush()

            version = models.DatasetVersion(
                dataset_id=dataset.id, description=None, metadata_=comprehensive_attrs
            )
            session.add(version)
            await session.flush()

            session.add(
                models.Experiment(
                    dataset_id=dataset.id,
                    dataset_version_id=version.id,
                    name=f"comprehensive_exp_{token_hex(6)}",
                    repetitions=1,
                    metadata_=comprehensive_attrs,
                )
            )

        # === COMPREHENSIVE VERIFICATION ===
        # Test all JSON columns with single set of queries (more efficient than separate tests)

        # Raw SQL verification
        async with db() as session:
            # Verify span attributes & events
            attributes_result = (
                await session.scalars(sa.text("SELECT attributes FROM spans"))
            ).first()
            attributes_result = _decode_if_sqlite([attributes_result], db.dialect)[0]
            assert attributes_result == expected_converted

            events_result = (await session.scalars(sa.text("SELECT events FROM spans"))).first()
            events_result = _decode_if_sqlite([events_result], db.dialect)[0]
            assert events_result == [expected_event]

            # Verify all metadata fields
            metadata_tables = [
                "span_annotations",
                "trace_annotations",
                "document_annotations",
                "datasets",
                "dataset_versions",
                "experiments",
            ]
            for table in metadata_tables:
                result = (await session.scalars(sa.text(f"SELECT metadata FROM {table}"))).first()
                result = _decode_if_sqlite([result], db.dialect)[0]
                assert result == expected_converted, f"Failed for table: {table}"

        # ORM verification (ensures type adapters work correctly)
        async with db() as session:
            # Span attributes & events
            assert (
                await session.scalars(select(models.Span.attributes))
            ).first() == expected_converted
            assert (await session.scalars(select(models.Span.events))).first() == [expected_event]

            # All metadata fields
            assert (
                await session.scalars(select(models.SpanAnnotation.metadata_))
            ).first() == expected_converted
            assert (
                await session.scalars(select(models.TraceAnnotation.metadata_))
            ).first() == expected_converted
            assert (
                await session.scalars(select(models.DocumentAnnotation.metadata_))
            ).first() == expected_converted
            assert (
                await session.scalars(select(models.Dataset.metadata_))
            ).first() == expected_converted
            assert (
                await session.scalars(select(models.DatasetVersion.metadata_))
            ).first() == expected_converted
            assert (
                await session.scalars(select(models.Experiment.metadata_))
            ).first() == expected_converted

        # === COMPREHENSIVE SQLite NaN/Inf EDGE CASE TESTING ===
        # SQLite allows raw NaN/Inf in JSON TEXT storage, but ORM should sanitize on read
        # This tests the critical edge case where raw NaN/Inf bypasses our type adapters
        if db.dialect is SupportedSQLDialect.SQLITE:
            # Create NaN/Inf-only payload for raw JSON insertion (json.dumps can handle NaN/Inf)
            raw_attrs_with_nan = {
                "a": float("nan"),
                "b": [1, float("nan"), {"c": float("nan"), "ci": float("inf")}, [float("-inf")]],
                "d": {
                    "e": float("nan"),
                    "f": [float("nan"), {"g": float("nan"), "gi": float("inf")}],
                },
                "h": None,
                "i": True,
                "j": "NaN",
                "k": float("inf"),
                "l": float("-inf"),
                "x": {
                    "y": [
                        float("nan"),
                        {"z": float("nan"), "zi": float("inf")},
                        [float("inf"), float("-inf"), float("nan")],
                    ]
                },
            }
            # Expected sanitized version
            sanitized_nan_attrs = {
                "a": None,
                "b": [1, None, {"c": None, "ci": None}, [None]],
                "d": {"e": None, "f": [None, {"g": None, "gi": None}]},
                "h": None,
                "i": True,
                "j": "NaN",
                "k": None,
                "l": None,
                "x": {"y": [None, {"z": None, "zi": None}, [None, None, None]]},
            }
            # Simple NaN/Inf event for edge case testing
            raw_event_with_nan = {
                "name": "NaN Test Event",
                "timestamp": EVENT_TS,
                "attributes": raw_attrs_with_nan,
            }
            sanitized_event = {**raw_event_with_nan, "attributes": sanitized_nan_attrs}

            # Force raw NaN/Inf JSON directly into database (bypassing type adapters)
            async with db() as session:
                # Update all JSON columns with raw NaN/Inf values
                await session.execute(
                    sa.text("UPDATE spans SET attributes = :attrs").bindparams(
                        attrs=json.dumps(raw_attrs_with_nan)
                    )
                )
                await session.execute(
                    sa.text("UPDATE spans SET events = :events").bindparams(
                        events=json.dumps([raw_event_with_nan])
                    )
                )
                # Update all metadata tables
                await session.execute(
                    sa.text("UPDATE span_annotations SET metadata = :m").bindparams(
                        m=json.dumps(raw_attrs_with_nan)
                    )
                )
                await session.execute(
                    sa.text("UPDATE trace_annotations SET metadata = :m").bindparams(
                        m=json.dumps(raw_attrs_with_nan)
                    )
                )
                await session.execute(
                    sa.text("UPDATE document_annotations SET metadata = :m").bindparams(
                        m=json.dumps(raw_attrs_with_nan)
                    )
                )
                await session.execute(
                    sa.text("UPDATE datasets SET metadata = :m").bindparams(
                        m=json.dumps(raw_attrs_with_nan)
                    )
                )
                await session.execute(
                    sa.text("UPDATE dataset_versions SET metadata = :m").bindparams(
                        m=json.dumps(raw_attrs_with_nan)
                    )
                )
                await session.execute(
                    sa.text("UPDATE experiments SET metadata = :m").bindparams(
                        m=json.dumps(raw_attrs_with_nan)
                    )
                )

            # Verify raw storage: SQLite should contain unsanitized NaN/Inf values
            # Use DeepDiff because NaN != NaN in direct equality comparisons
            async with db() as session:
                # Check span attributes
                raw_attrs_result = (
                    await session.scalars(sa.text("SELECT attributes FROM spans"))
                ).first()
                raw_attrs_result = _decode_if_sqlite([raw_attrs_result], db.dialect)[0]
                assert not DeepDiff(
                    [raw_attrs_result], [raw_attrs_with_nan], ignore_nan_inequality=True
                )

                # Check span events
                raw_events_result = (
                    await session.scalars(sa.text("SELECT events FROM spans"))
                ).first()
                raw_events_result = _decode_if_sqlite([raw_events_result], db.dialect)[0]
                assert not DeepDiff(
                    [raw_events_result], [[raw_event_with_nan]], ignore_nan_inequality=True
                )

                # Check all metadata tables have raw NaN/Inf
                metadata_tables = [
                    "span_annotations",
                    "trace_annotations",
                    "document_annotations",
                    "datasets",
                    "dataset_versions",
                    "experiments",
                ]
                for table in metadata_tables:
                    raw_metadata = (
                        await session.scalars(sa.text(f"SELECT metadata FROM {table}"))
                    ).first()
                    raw_metadata = _decode_if_sqlite([raw_metadata], db.dialect)[0]
                    assert not DeepDiff(
                        [raw_metadata], [raw_attrs_with_nan], ignore_nan_inequality=True
                    )

            # CRITICAL TEST: Even with raw NaN/Inf in DB storage, ORM reads must return sanitized values
            async with db() as session:
                # Span attributes & events should be sanitized by ORM
                assert (
                    await session.scalars(select(models.Span.attributes))
                ).first() == sanitized_nan_attrs
                assert (await session.scalars(select(models.Span.events))).first() == [
                    sanitized_event
                ]

                # All metadata fields should be sanitized by ORM
                assert (
                    await session.scalars(select(models.SpanAnnotation.metadata_))
                ).first() == sanitized_nan_attrs
                assert (
                    await session.scalars(select(models.TraceAnnotation.metadata_))
                ).first() == sanitized_nan_attrs
                assert (
                    await session.scalars(select(models.DocumentAnnotation.metadata_))
                ).first() == sanitized_nan_attrs
                assert (
                    await session.scalars(select(models.Dataset.metadata_))
                ).first() == sanitized_nan_attrs
                assert (
                    await session.scalars(select(models.DatasetVersion.metadata_))
                ).first() == sanitized_nan_attrs
                assert (
                    await session.scalars(select(models.Experiment.metadata_))
                ).first() == sanitized_nan_attrs

    async def test_num_documents_handles_non_array_values(
        self,
        db: DbSessionFactory,
    ) -> None:
        """num_documents returns array length for arrays and 0 for non-array values."""
        start_time = datetime.fromisoformat("2024-01-01T00:00:00+00:00")
        end_time = datetime.fromisoformat("2024-01-01T00:01:00+00:00")

        # Test cases: (span_kind, attributes, expected_num_documents)
        test_cases: list[tuple[str, dict[str, Any], int]] = [
            # RETRIEVER: Valid array - should return count
            (
                "RETRIEVER",
                {
                    "retrieval": {
                        "documents": [
                            {"document": {"content": "doc1"}},
                            {"document": {"content": "doc2"}},
                            {"document": {"content": "doc3"}},
                        ]
                    }
                },
                3,
            ),
            # RETRIEVER: Scalar string - should return 0
            ("RETRIEVER", {"retrieval": {"documents": "not an array"}}, 0),
            # RETRIEVER: Object - should return 0
            ("RETRIEVER", {"retrieval": {"documents": {"key": "value"}}}, 0),
            # RETRIEVER: Number - should return 0
            ("RETRIEVER", {"retrieval": {"documents": 42}}, 0),
            # RETRIEVER: Null/missing - should return 0
            ("RETRIEVER", {}, 0),
            # RETRIEVER: Parent key exists but documents missing - should return 0
            ("RETRIEVER", {"retrieval": {}}, 0),
            # RETRIEVER: Parent key is a scalar - should return 0
            ("RETRIEVER", {"retrieval": 42}, 0),
            # RERANKER: Valid array - should return count
            (
                "RERANKER",
                {
                    "reranker": {
                        "output_documents": [
                            {"document": {"content": "doc1"}},
                            {"document": {"content": "doc2"}},
                        ]
                    }
                },
                2,
            ),
            # RERANKER: Scalar string - should return 0
            ("RERANKER", {"reranker": {"output_documents": "not an array"}}, 0),
            # RERANKER: Object - should return 0
            ("RERANKER", {"reranker": {"output_documents": {"key": "value"}}}, 0),
            # RERANKER: Null/missing - should return 0
            ("RERANKER", {}, 0),
            # RERANKER: Parent key exists but output_documents missing - should return 0
            ("RERANKER", {"reranker": {}}, 0),
            # RERANKER: Parent key is a scalar - should return 0
            ("RERANKER", {"reranker": 42}, 0),
        ]

        async with db() as session:
            project = models.Project(name=f"test_project_{token_hex(4)}")
            session.add(project)
            await session.flush()

            trace = models.Trace(
                project_rowid=project.id,
                trace_id=f"test-trace-{token_hex(4)}",
                start_time=start_time,
                end_time=end_time,
            )
            session.add(trace)
            await session.flush()

            for i, (span_kind, attributes, expected) in enumerate(test_cases):
                span = models.Span(
                    trace_rowid=trace.id,
                    span_id=f"span-{i}-{token_hex(4)}",
                    name="test-span",
                    span_kind=span_kind,
                    start_time=start_time,
                    end_time=end_time,
                    attributes=attributes,
                    events=[],
                    status_code="OK",
                    status_message="",
                    cumulative_error_count=0,
                    cumulative_llm_token_count_prompt=0,
                    cumulative_llm_token_count_completion=0,
                )
                session.add(span)
                await session.flush()

                stmt = select(models.Span.num_documents).where(models.Span.id == span.id)
                result = await session.scalar(stmt)
                assert result == expected, (
                    f"Case {i} ({span_kind}): expected {expected}, got {result}"
                )


# --- tests/unit/db/insertion/test_dataset.py ---

async def test_create_dataset_with_span_links(
    db: DbSessionFactory,
) -> None:
    """Test that dataset examples can be linked to spans via span_id."""
    # First, create a trace and span to link to
    async with db() as session:
        # Create a project
        project_id = await session.scalar(
            insert(models.Project).values(name="test-project").returning(models.Project.id)
        )

        # Create a trace
        trace_id = await session.scalar(
            insert(models.Trace)
            .values(
                project_rowid=project_id,
                trace_id="test-trace-123",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
            )
            .returning(models.Trace.id)
        )

        # Create spans
        span_rowid_1 = await session.scalar(
            insert(models.Span)
            .values(
                trace_rowid=trace_id,
                span_id="span-abc-123",
                name="test_span_1",
                span_kind="INTERNAL",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                attributes={},
                events=[],
                status_code="OK",
                status_message="",
                cumulative_error_count=0,
                cumulative_llm_token_count_prompt=0,
                cumulative_llm_token_count_completion=0,
            )
            .returning(models.Span.id)
        )

        span_rowid_2 = await session.scalar(
            insert(models.Span)
            .values(
                trace_rowid=trace_id,
                span_id="span-def-456",
                name="test_span_2",
                span_kind="INTERNAL",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                attributes={},
                events=[],
                status_code="OK",
                status_message="",
                cumulative_error_count=0,
                cumulative_llm_token_count_prompt=0,
                cumulative_llm_token_count_completion=0,
            )
            .returning(models.Span.id)
        )

        await session.commit()

    # Now create dataset with span links
    async with db() as session:
        await add_dataset_examples(
            session=session,
            examples=[
                ExampleContent(
                    input={"x": 1},
                    output={"z": 3},
                    span_id="span-abc-123",
                ),
                ExampleContent(
                    input={"x": 2},
                    output={"z": 6},
                    span_id="span-def-456",
                ),
                ExampleContent(
                    input={"x": 3},
                    output={"z": 9},
                    span_id="nonexistent-span",  # This span doesn't exist
                ),
                ExampleContent(
                    input={"x": 4},
                    output={"z": 12},
                    span_id=None,  # No span link
                ),
            ],
            name="dataset-with-spans",
        )

    # Verify the dataset examples are linked correctly
    async with db() as session:
        examples = await session.scalars(
            select(models.DatasetExample)
            .join(models.Dataset)
            .where(models.Dataset.name == "dataset-with-spans")
            .order_by(models.DatasetExample.id)
        )

        examples_list = list(examples)
        assert len(examples_list) == 4

        # First example should be linked to span 1
        assert examples_list[0].span_rowid == span_rowid_1

        # Second example should be linked to span 2
        assert examples_list[1].span_rowid == span_rowid_2

        # Third example should have no link (span doesn't exist)
        assert examples_list[2].span_rowid is None

        # Fourth example should have no link (no span_id provided)
        assert examples_list[3].span_rowid is None

async def test_resolve_span_ids_to_rowids_deduplicates_input(
    db: DbSessionFactory,
) -> None:
    """Test that resolve_span_ids_to_rowids deduplicates span IDs before querying.

    This is critical for performance: 10,000 examples referencing the same 5 span IDs
    should only consume 5 query parameters, not 10,000.
    """
    # Create a trace and span
    async with db() as session:
        project_id = await session.scalar(
            insert(models.Project).values(name="dedup-test-project").returning(models.Project.id)
        )
        trace_id = await session.scalar(
            insert(models.Trace)
            .values(
                project_rowid=project_id,
                trace_id="dedup-test-trace",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
            )
            .returning(models.Trace.id)
        )
        span_rowid = await session.scalar(
            insert(models.Span)
            .values(
                trace_rowid=trace_id,
                span_id="duplicate-span-id",
                name="test_span",
                span_kind="INTERNAL",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                attributes={},
                events=[],
                status_code="OK",
                status_message="",
                cumulative_error_count=0,
                cumulative_llm_token_count_prompt=0,
                cumulative_llm_token_count_completion=0,
            )
            .returning(models.Span.id)
        )
        await session.commit()

    # Pass many duplicates of the same span ID
    span_ids_with_duplicates: list[str | None] = (
        ["duplicate-span-id"] * 100 + [None] * 50 + [""] * 25
    )

    async with db() as session:
        result = await resolve_span_ids_to_rowids(session, span_ids_with_duplicates)

    # Should still resolve correctly
    assert len(result) == 1
    assert result["duplicate-span-id"] == span_rowid

async def test_resolve_span_ids_to_rowids_batches_large_inputs(
    db: DbSessionFactory,
) -> None:
    """Test that resolve_span_ids_to_rowids processes large inputs in batches.

    With a small batch_size, we can verify batching works without creating
    thousands of spans.
    """
    # Create multiple spans
    async with db() as session:
        project_id = await session.scalar(
            insert(models.Project).values(name="batch-test-project").returning(models.Project.id)
        )
        trace_id = await session.scalar(
            insert(models.Trace)
            .values(
                project_rowid=project_id,
                trace_id="batch-test-trace",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
            )
            .returning(models.Trace.id)
        )

        # Create 10 spans
        span_id_strs = [f"span-batch-{i}" for i in range(10)]
        span_rowids: dict[str, int] = {}
        for span_id in span_id_strs:
            rowid = await session.scalar(
                insert(models.Span)
                .values(
                    trace_rowid=trace_id,
                    span_id=span_id,
                    name=f"test_span_{span_id}",
                    span_kind="INTERNAL",
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_message="",
                    cumulative_error_count=0,
                    cumulative_llm_token_count_prompt=0,
                    cumulative_llm_token_count_completion=0,
                )
                .returning(models.Span.id)
            )
            assert rowid is not None
            span_rowids[span_id] = rowid
        await session.commit()

    # Resolve with a small batch size to force multiple batches
    # Cast to list[str | None] as required by the function signature
    span_ids_input: list[str | None] = list(span_id_strs)
    async with db() as session:
        result = await resolve_span_ids_to_rowids(session, span_ids_input, batch_size=3)

    # All 10 spans should be resolved correctly
    assert len(result) == 10
    for span_id in span_id_strs:
        assert result[span_id] == span_rowids[span_id]

async def test_resolve_span_ids_to_rowids_various_batch_sizes(
    db: DbSessionFactory,
    batch_size: int,
) -> None:
    """Test that resolve_span_ids_to_rowids works correctly with various batch sizes."""
    # Create spans
    async with db() as session:
        project_id = await session.scalar(
            insert(models.Project)
            .values(name=f"batch-size-{batch_size}-project")
            .returning(models.Project.id)
        )
        trace_id = await session.scalar(
            insert(models.Trace)
            .values(
                project_rowid=project_id,
                trace_id=f"batch-size-{batch_size}-trace",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
            )
            .returning(models.Trace.id)
        )

        # Create 10 spans
        expected_mappings: dict[str, int] = {}
        for i in range(10):
            span_id = f"span-size-{batch_size}-{i}"
            rowid = await session.scalar(
                insert(models.Span)
                .values(
                    trace_rowid=trace_id,
                    span_id=span_id,
                    name=f"test_span_{i}",
                    span_kind="INTERNAL",
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_message="",
                    cumulative_error_count=0,
                    cumulative_llm_token_count_prompt=0,
                    cumulative_llm_token_count_completion=0,
                )
                .returning(models.Span.id)
            )
            assert rowid is not None
            expected_mappings[span_id] = rowid
        await session.commit()

    # Resolve with parameterized batch size
    async with db() as session:
        span_ids: list[str | None] = list(expected_mappings.keys())
        result = await resolve_span_ids_to_rowids(session, span_ids, batch_size=batch_size)

    assert result == expected_mappings


# --- tests/unit/db/types/test_trace_retention.py ---

def test_time_of_next_run(
    cron_expression: str,
    frozen_time: str,
    expected_time: str,
    comment: str,
) -> None:
    """
    Test the time_of_next_run function with various cron expressions.

    Args:
        cron_expression: The cron expression to test
        frozen_time: The time to freeze at for testing
        expected_time: The expected next run time
        comment: Description of the test case
    """
    with freeze_time(frozen_time):
        actual = _time_of_next_run(cron_expression)
        expected = datetime.fromisoformat(expected_time)
        assert actual == expected

def test_invalid_cron_expressions(cron_expression: str, expected_error_msg: str) -> None:
    """
    Test that the time_of_next_run function correctly raises ValueErrors
    for invalid cron expressions.

    Args:
        cron_expression: An invalid cron expression
        expected_error_msg: The expected error message prefix
    """
    with pytest.raises(ValueError) as exc_info:
        _time_of_next_run(cron_expression)
    assert str(exc_info.value).startswith(expected_error_msg)


# --- tests/unit/server/api/test_cancellation.py ---

    async def test_empty_in_progress_no_errors(self) -> None:
        """
        Verify cleanup handles empty in_progress list gracefully.
        """
        in_progress: list[tuple[int, ChatStream, asyncio.Task[Any]]] = []
        not_started: deque[tuple[int, ChatStream]] = deque()

        # Should not raise any exceptions
        await _cleanup_chat_completion_resources(
            in_progress=in_progress,
            not_started=not_started,
        )


# --- tests/unit/server/api/test_queries.py ---

async def test_compare_experiments_returns_expected_comparisons(
    gql_client: AsyncGraphQLClient,
    comparison_experiments: Any,
) -> None:
    query = """
      query ($baseExperimentId: ID!, $compareExperimentIds: [ID!]!, $first: Int, $after: String) {
        compareExperiments(
          baseExperimentId: $baseExperimentId
          compareExperimentIds: $compareExperimentIds
          first: $first
          after: $after
        ) {
          edges {
            node {
              example {
                id
                revision {
                  input
                  output
                  metadata
                }
              }
              repeatedRunGroups {
                experimentId
                runs {
                  id
                  output
                }
              }
            }
          }
        }
      }
    """
    response = await gql_client.execute(
        query=query,
        variables={
            "baseExperimentId": str(GlobalID("Experiment", str(2))),
            "compareExperimentIds": [
                str(GlobalID("Experiment", str(1))),
                str(GlobalID("Experiment", str(3))),
            ],
            "first": 50,
            "after": None,
        },
    )
    assert not response.errors
    assert response.data == {
        "compareExperiments": {
            "edges": [
                {
                    "node": {
                        "example": {
                            "id": str(GlobalID("DatasetExample", str(1))),
                            "revision": {
                                "input": {"revision-2-input-key": "revision-2-input-value"},
                                "output": {"revision-2-output-key": "revision-2-output-value"},
                                "metadata": {
                                    "revision-2-metadata-key": "revision-2-metadata-value"
                                },
                            },
                        },
                        "repeatedRunGroups": [
                            {
                                "experimentId": str(GlobalID("Experiment", str(2))),
                                "runs": [
                                    {
                                        "id": str(GlobalID("ExperimentRun", str(3))),
                                        "output": 3,
                                    },
                                ],
                            },
                            {
                                "experimentId": str(GlobalID("Experiment", str(1))),
                                "runs": [
                                    {
                                        "id": str(GlobalID("ExperimentRun", str(1))),
                                        "output": {"output": "run-1-output-value"},
                                    },
                                ],
                            },
                            {
                                "experimentId": str(GlobalID("Experiment", str(3))),
                                "runs": [
                                    {
                                        "id": str(GlobalID("ExperimentRun", str(5))),
                                        "output": None,
                                    },
                                    {
                                        "id": str(GlobalID("ExperimentRun", str(6))),
                                        "output": {"output": "run-6-output-value"},
                                    },
                                ],
                            },
                        ],
                    },
                },
                {
                    "node": {
                        "example": {
                            "id": str(GlobalID("DatasetExample", str(2))),
                            "revision": {
                                "input": {"revision-4-input-key": "revision-4-input-value"},
                                "output": {"revision-4-output-key": "revision-4-output-value"},
                                "metadata": {
                                    "revision-4-metadata-key": "revision-4-metadata-value"
                                },
                            },
                        },
                        "repeatedRunGroups": [
                            {
                                "experimentId": str(GlobalID("Experiment", str(2))),
                                "runs": [
                                    {
                                        "id": str(GlobalID("ExperimentRun", str(4))),
                                        "output": "",
                                    },
                                ],
                            },
                            {
                                "experimentId": str(GlobalID("Experiment", str(1))),
                                "runs": [],
                            },
                            {
                                "experimentId": str(GlobalID("Experiment", str(3))),
                                "runs": [
                                    {
                                        "id": str(GlobalID("ExperimentRun", str(7))),
                                        "output": "run-7-output-value",
                                    },
                                    {
                                        "id": str(GlobalID("ExperimentRun", str(8))),
                                        "output": 8,
                                    },
                                ],
                            },
                        ],
                    },
                },
            ],
        }
    }

    async def test_openai_chat_completions_returns_standard_params(
        self,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        """OpenAI with CHAT_COMPLETIONS should return standard parameters like temperature."""
        response = await gql_client.execute(
            query=self._QUERY,
            variables={
                "input": {
                    "providerKey": "OPENAI",
                    "modelName": "gpt-4o",
                    "openaiApiType": "CHAT_COMPLETIONS",
                }
            },
        )
        assert not response.errors
        assert response.data is not None
        param_names = [p["invocationName"] for p in response.data["modelInvocationParameters"]]
        assert "temperature" in param_names
        assert "top_p" in param_names
        assert "frequency_penalty" in param_names
        assert "reasoning_effort" not in param_names

    async def test_openai_chat_completions_custom_model_returns_standard_params(
        self,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        """Custom model names with CHAT_COMPLETIONS should return standard parameters."""
        response = await gql_client.execute(
            query=self._QUERY,
            variables={
                "input": {
                    "providerKey": "OPENAI",
                    "modelName": "my-custom-fine-tuned-model",
                    "openaiApiType": "CHAT_COMPLETIONS",
                }
            },
        )
        assert not response.errors
        assert response.data is not None
        param_names = [p["invocationName"] for p in response.data["modelInvocationParameters"]]
        assert "temperature" in param_names
        assert "reasoning_effort" not in param_names

    async def test_openai_reasoning_model_with_chat_completions_returns_reasoning_params(
        self,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        """Reasoning models (o1, o3) with CHAT_COMPLETIONS should return reasoning parameters."""
        response = await gql_client.execute(
            query=self._QUERY,
            variables={
                "input": {
                    "providerKey": "OPENAI",
                    "modelName": "o1",
                    "openaiApiType": "CHAT_COMPLETIONS",
                }
            },
        )
        assert not response.errors
        assert response.data is not None
        param_names = [p["invocationName"] for p in response.data["modelInvocationParameters"]]
        assert "reasoning_effort" in param_names
        assert "temperature" not in param_names

    async def test_azure_chat_completions_returns_standard_params(
        self,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        """Azure OpenAI with CHAT_COMPLETIONS should return standard parameters."""
        response = await gql_client.execute(
            query=self._QUERY,
            variables={
                "input": {
                    "providerKey": "AZURE_OPENAI",
                    "modelName": "my-deployment",
                    "openaiApiType": "CHAT_COMPLETIONS",
                }
            },
        )
        assert not response.errors
        assert response.data is not None
        param_names = [p["invocationName"] for p in response.data["modelInvocationParameters"]]
        assert "temperature" in param_names
        assert "reasoning_effort" not in param_names

    async def test_anthropic_ignores_openai_api_type(
        self,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        """Non-OpenAI providers should ignore openaiApiType and use registry."""
        response = await gql_client.execute(
            query=self._QUERY,
            variables={
                "input": {
                    "providerKey": "ANTHROPIC",
                    "modelName": "claude-3-5-sonnet-latest",
                    "openaiApiType": "CHAT_COMPLETIONS",  # Should be ignored
                }
            },
        )
        assert not response.errors
        assert response.data is not None
        param_names = [p["invocationName"] for p in response.data["modelInvocationParameters"]]
        # Anthropic has its own parameters
        assert "temperature" in param_names
        assert "max_tokens" in param_names
        # Should not have OpenAI-specific reasoning params
        assert "reasoning_effort" not in param_names


# --- tests/unit/server/api/test_subscriptions.py ---

    async def test_openai_text_response_emits_expected_payloads_and_records_expected_span(
        self,
        gql_client: AsyncGraphQLClient,
        openai_api_key: str,
        custom_vcr: CustomVCR,
    ) -> None:
        variables = {
            "input": {
                "promptVersion": {
                    "templateFormat": "NONE",
                    "template": {
                        "messages": [
                            {
                                "role": "USER",
                                "content": [
                                    {
                                        "text": {
                                            "text": "Who won the World Cup in 2018? Answer in one word"
                                        }
                                    }
                                ],
                            }
                        ]
                    },
                    "modelProvider": "OPENAI",
                    "modelName": "gpt-4",
                    "invocationParameters": {"temperature": 0.1},
                    "tools": None,
                },
                "repetitions": 1,
            },
        }
        with custom_vcr.use_cassette():
            payloads = [
                payload
                async for payload in gql_client.subscription(
                    query=self.QUERY,
                    variables=variables,
                    operation_name="ChatCompletionSubscription",
                )
            ]

        # check subscription payloads
        assert payloads
        assert (last_payload := payloads.pop())["chatCompletion"][
            "__typename"
        ] == ChatCompletionSubscriptionResult.__name__
        assert all(
            payload["chatCompletion"]["__typename"] == TextChunk.__name__ for payload in payloads
        )
        response_text = "".join(payload["chatCompletion"]["content"] for payload in payloads)
        assert "france" in response_text.lower()
        subscription_span = last_payload["chatCompletion"]["span"]
        span_id = subscription_span["id"]

        # query for the span via the node interface to ensure that the span
        # recorded in the db contains identical information as the span emitted
        # by the subscription
        response = await gql_client.execute(
            query=self.QUERY, variables={"spanId": span_id}, operation_name="SpanQuery"
        )
        assert (data := response.data) is not None
        span = data["span"]
        assert json.loads(attributes := span.pop("attributes")) == json.loads(
            subscription_span.pop("attributes")
        )
        attributes = dict(flatten(json.loads(attributes)))
        _assert_spans_equal(span, subscription_span)

        # check attributes
        assert span.pop("id") == span_id
        assert span.pop("name") == "ChatCompletion"
        assert span.pop("statusCode") == "OK"
        assert not span.pop("statusMessage")
        assert span.pop("startTime")
        assert span.pop("endTime")
        assert isinstance(span.pop("latencyMs"), float)
        assert span.pop("parentId") is None
        assert span.pop("spanKind") == "llm"
        assert (context := span.pop("context")).pop("spanId")
        assert context.pop("traceId")
        assert not context
        assert span.pop("metadata") is None
        assert span.pop("numDocuments") == 0
        assert isinstance(token_count_total := span.pop("tokenCountTotal"), int)
        assert isinstance(token_count_prompt := span.pop("tokenCountPrompt"), int)
        assert isinstance(token_count_completion := span.pop("tokenCountCompletion"), int)
        assert token_count_prompt > 0
        assert token_count_completion > 0
        assert token_count_total == token_count_prompt + token_count_completion
        assert (input := span.pop("input")).pop("mimeType") == "json"
        assert (input_value := input.pop("value"))
        assert not input
        assert "api_key" not in input_value
        assert "apiKey" not in input_value
        assert (output := span.pop("output")).pop("mimeType") == "text"
        assert output.pop("value")
        assert not output
        assert not span.pop("events")
        assert isinstance(
            cumulative_token_count_total := span.pop("cumulativeTokenCountTotal"), float
        )
        assert isinstance(
            cumulative_token_count_prompt := span.pop("cumulativeTokenCountPrompt"), float
        )
        assert isinstance(
            cumulative_token_count_completion := span.pop("cumulativeTokenCountCompletion"), float
        )
        assert cumulative_token_count_total == token_count_total
        assert cumulative_token_count_prompt == token_count_prompt
        assert cumulative_token_count_completion == token_count_completion
        assert span.pop("propagatedStatusCode") == "OK"
        assert not span

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4"
        assert isinstance(invocation_parameters := attributes.pop("llm.invocation_parameters"), str)
        assert json.loads(invocation_parameters) == {
            "temperature": 0.1,
        }
        assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == token_count_total
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == token_count_prompt
        assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == token_count_completion
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == 0
        assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING) == 0
        assert attributes.pop(INPUT_VALUE)
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE)
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(LLM_INPUT_MESSAGES) == [
            {
                "message": {
                    "role": "user",
                    "content": "Who won the World Cup in 2018? Answer in one word",
                }
            }
        ]
        assert attributes.pop(LLM_OUTPUT_MESSAGES) == [
            {
                "message": {
                    "role": "assistant",
                    "content": response_text,
                }
            }
        ]
        assert attributes.pop(LLM_PROVIDER) == "openai"
        assert attributes.pop(LLM_SYSTEM) == "openai"
        assert attributes.pop(URL_FULL) == "https://api.openai.com/v1/chat/completions"
        assert attributes.pop(URL_PATH) == "chat/completions"
        assert not attributes

    async def test_openai_emits_expected_payloads_and_records_expected_span_on_error(
        self,
        gql_client: AsyncGraphQLClient,
        openai_api_key: str,
        custom_vcr: CustomVCR,
    ) -> None:
        variables = {
            "input": {
                "promptVersion": {
                    "templateFormat": "NONE",
                    "template": {
                        "messages": [
                            {
                                "role": "USER",
                                "content": [
                                    {
                                        "text": {
                                            "text": "Who won the World Cup in 2018? Answer in one word"
                                        }
                                    }
                                ],
                            }
                        ]
                    },
                    "modelProvider": "OPENAI",
                    "modelName": "gpt-4",
                    "invocationParameters": {"temperature": 0.1},
                    "tools": None,
                },
                "repetitions": 1,
            },
        }
        with custom_vcr.use_cassette():
            payloads = [
                payload
                async for payload in gql_client.subscription(
                    query=self.QUERY,
                    variables=variables,
                    operation_name="ChatCompletionSubscription",
                )
            ]

        # check subscription payloads
        assert len(payloads) == 2
        assert (error_payload := payloads[0])["chatCompletion"][
            "__typename"
        ] == ChatCompletionSubscriptionError.__name__
        assert "401" in (status_message := error_payload["chatCompletion"]["message"])
        assert "api key" in status_message.lower()
        assert (last_payload := payloads.pop())["chatCompletion"][
            "__typename"
        ] == ChatCompletionSubscriptionResult.__name__
        subscription_span = last_payload["chatCompletion"]["span"]
        span_id = subscription_span["id"]

        # query for the span via the node interface to ensure that the span
        # recorded in the db contains identical information as the span emitted
        # by the subscription
        response = await gql_client.execute(
            query=self.QUERY, variables={"spanId": span_id}, operation_name="SpanQuery"
        )
        assert (data := response.data) is not None
        span = data["span"]
        assert json.loads(attributes := span.pop("attributes")) == json.loads(
            subscription_span.pop("attributes")
        )
        attributes = dict(flatten(json.loads(attributes)))
        _assert_spans_equal(span, subscription_span)

        # check attributes
        assert span.pop("id") == span_id
        assert span.pop("name") == "ChatCompletion"
        assert span.pop("statusCode") == "ERROR"
        assert span.pop("statusMessage") == status_message
        assert span.pop("startTime")
        assert span.pop("endTime")
        assert isinstance(span.pop("latencyMs"), float)
        assert span.pop("parentId") is None
        assert span.pop("spanKind") == "llm"
        assert (context := span.pop("context")).pop("spanId")
        assert context.pop("traceId")
        assert not context
        assert span.pop("metadata") is None
        assert span.pop("numDocuments") == 0
        assert span.pop("tokenCountTotal") == 0
        assert span.pop("tokenCountPrompt") is None
        assert span.pop("tokenCountCompletion") is None
        assert (input := span.pop("input")).pop("mimeType") == "json"
        assert (input_value := input.pop("value"))
        assert not input
        assert "api_key" not in input_value
        assert "apiKey" not in input_value
        assert span.pop("output") is None
        assert (events := span.pop("events"))
        assert len(events) == 1
        assert (event := events[0])
        assert event.pop("name") == "exception"
        assert event.pop("message") == status_message
        assert datetime.fromisoformat(event.pop("timestamp"))
        assert not event
        assert isinstance(
            cumulative_token_count_total := span.pop("cumulativeTokenCountTotal"), float
        )
        assert isinstance(
            cumulative_token_count_prompt := span.pop("cumulativeTokenCountPrompt"), float
        )
        assert isinstance(
            cumulative_token_count_completion := span.pop("cumulativeTokenCountCompletion"), float
        )
        assert cumulative_token_count_total == 0
        assert cumulative_token_count_prompt == 0
        assert cumulative_token_count_completion == 0
        assert span.pop("propagatedStatusCode") == "ERROR"
        assert not span

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4"
        assert isinstance(invocation_parameters := attributes.pop("llm.invocation_parameters"), str)
        assert json.loads(invocation_parameters) == {
            "temperature": 0.1,
        }
        assert attributes.pop(INPUT_VALUE)
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert attributes.pop(LLM_INPUT_MESSAGES) == [
            {
                "message": {
                    "role": "user",
                    "content": "Who won the World Cup in 2018? Answer in one word",
                }
            }
        ]
        assert attributes.pop(LLM_PROVIDER) == "openai"
        assert attributes.pop(LLM_SYSTEM) == "openai"
        assert attributes.pop(URL_FULL) == "https://api.openai.com/v1/chat/completions"
        assert attributes.pop(URL_PATH) == "chat/completions"
        assert not attributes

    async def test_openai_tool_call_response_emits_expected_payloads_and_records_expected_span(
        self,
        gql_client: AsyncGraphQLClient,
        openai_api_key: str,
        custom_vcr: CustomVCR,
    ) -> None:
        get_current_weather_tool_schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["location"],
                },
                "strict": None,
            },
        }
        variables = {
            "input": {
                "promptVersion": {
                    "templateFormat": "NONE",
                    "template": {
                        "messages": [
                            {
                                "role": "USER",
                                "content": [
                                    {"text": {"text": "How's the weather in San Francisco?"}}
                                ],
                            }
                        ]
                    },
                    "modelProvider": "OPENAI",
                    "modelName": "gpt-4",
                    "invocationParameters": {},
                    "tools": {
                        "tools": [
                            {
                                "function": {
                                    "name": get_current_weather_tool_schema["function"]["name"],
                                    "description": get_current_weather_tool_schema["function"][
                                        "description"
                                    ],
                                    "parameters": get_current_weather_tool_schema["function"][
                                        "parameters"
                                    ],
                                },
                            }
                        ],
                        "toolChoice": {"zeroOrMore": True},
                    },
                },
                "repetitions": 1,
            },
        }
        with custom_vcr.use_cassette():
            payloads = [
                payload
                async for payload in gql_client.subscription(
                    query=self.QUERY,
                    variables=variables,
                    operation_name="ChatCompletionSubscription",
                )
            ]

        # check subscription payloads
        assert payloads
        assert (last_payload := payloads.pop())["chatCompletion"][
            "__typename"
        ] == ChatCompletionSubscriptionResult.__name__
        assert all(
            payload["chatCompletion"]["__typename"] == ToolCallChunk.__name__
            for payload in payloads
        )
        json.loads(
            "".join(payload["chatCompletion"]["function"]["arguments"] for payload in payloads)
        ) == {"location": "San Francisco"}
        subscription_span = last_payload["chatCompletion"]["span"]
        span_id = subscription_span["id"]

        # query for the span via the node interface to ensure that the span
        # recorded in the db contains identical information as the span emitted
        # by the subscription
        response = await gql_client.execute(
            query=self.QUERY, variables={"spanId": span_id}, operation_name="SpanQuery"
        )
        assert (data := response.data) is not None
        span = data["span"]
        assert json.loads(attributes := span.pop("attributes")) == json.loads(
            subscription_span.pop("attributes")
        )
        attributes = dict(flatten(json.loads(attributes)))
        _assert_spans_equal(span, subscription_span)

        # check attributes
        assert span.pop("id") == span_id
        assert span.pop("name") == "ChatCompletion"
        assert span.pop("statusCode") == "OK"
        assert not span.pop("statusMessage")
        assert span.pop("startTime")
        assert span.pop("endTime")
        assert isinstance(span.pop("latencyMs"), float)
        assert span.pop("parentId") is None
        assert span.pop("spanKind") == "llm"
        assert (context := span.pop("context")).pop("spanId")
        assert context.pop("traceId")
        assert not context
        assert span.pop("metadata") is None
        assert span.pop("numDocuments") == 0
        assert isinstance(token_count_total := span.pop("tokenCountTotal"), int)
        assert isinstance(token_count_prompt := span.pop("tokenCountPrompt"), int)
        assert isinstance(token_count_completion := span.pop("tokenCountCompletion"), int)
        assert token_count_prompt > 0
        assert token_count_completion > 0
        assert token_count_total == token_count_prompt + token_count_completion
        assert (input := span.pop("input")).pop("mimeType") == "json"
        assert (input_value := input.pop("value"))
        assert not input
        assert "api_key" not in input_value
        assert "apiKey" not in input_value
        assert (output := span.pop("output")).pop("mimeType") == "json"
        assert output.pop("value")
        assert not output
        assert not span.pop("events")
        assert isinstance(
            cumulative_token_count_total := span.pop("cumulativeTokenCountTotal"), float
        )
        assert isinstance(
            cumulative_token_count_prompt := span.pop("cumulativeTokenCountPrompt"), float
        )
        assert isinstance(
            cumulative_token_count_completion := span.pop("cumulativeTokenCountCompletion"), float
        )
        assert cumulative_token_count_total == token_count_total
        assert cumulative_token_count_prompt == token_count_prompt
        assert cumulative_token_count_completion == token_count_completion
        assert span.pop("propagatedStatusCode") == "OK"
        assert not span

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4"
        assert isinstance(invocation_paramaters := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(invocation_paramaters) == {
            "tool_choice": "auto",
        }
        assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == token_count_total
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == token_count_prompt
        assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == token_count_completion
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == 0
        assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING) == 0
        assert attributes.pop(INPUT_VALUE)
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE)
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(LLM_INPUT_MESSAGES) == [
            {
                "message": {
                    "role": "user",
                    "content": "How's the weather in San Francisco?",
                }
            }
        ]
        assert (output_messages := attributes.pop(LLM_OUTPUT_MESSAGES))
        assert len(output_messages) == 1
        assert (output_message := output_messages[0]["message"])["role"] == "assistant"
        assert "content" not in output_message
        assert (tool_calls := output_message["tool_calls"])
        assert len(tool_calls) == 1
        assert (tool_call := tool_calls[0]["tool_call"])
        assert (function := tool_call["function"])
        assert function["name"] == "get_current_weather"
        assert json.loads(function["arguments"]) == {"location": "San Francisco"}
        assert (llm_tools := attributes.pop(LLM_TOOLS))
        assert len(llm_tools) == 1
        assert json.loads(llm_tools[0]["tool"]["json_schema"]) == get_current_weather_tool_schema
        assert attributes.pop(LLM_PROVIDER) == "openai"
        assert attributes.pop(LLM_SYSTEM) == "openai"
        assert attributes.pop(URL_FULL) == "https://api.openai.com/v1/chat/completions"
        assert attributes.pop(URL_PATH) == "chat/completions"
        assert not attributes

    async def test_openai_tool_call_messages_emits_expected_payloads_and_records_expected_span(
        self,
        gql_client: AsyncGraphQLClient,
        openai_api_key: str,
        custom_vcr: CustomVCR,
    ) -> None:
        tool_call_id = "call_zz1hkqH3IakqnHfVhrrUemlQ"
        variables = {
            "input": {
                "promptVersion": {
                    "templateFormat": "NONE",
                    "template": {
                        "messages": [
                            {
                                "role": "USER",
                                "content": [
                                    {"text": {"text": "How's the weather in San Francisco?"}}
                                ],
                            },
                            {
                                "role": "AI",
                                "content": [
                                    {
                                        "toolCall": {
                                            "toolCallId": tool_call_id,
                                            "toolCall": {
                                                "type": "function",
                                                "name": "get_weather",
                                                "arguments": json.dumps(
                                                    {"city": "San Francisco"}, indent=4
                                                ),
                                            },
                                        }
                                    }
                                ],
                            },
                            {
                                "role": "TOOL",
                                "content": [
                                    {
                                        "toolResult": {
                                            "toolCallId": tool_call_id,
                                            "result": "sunny",
                                        }
                                    }
                                ],
                            },
                        ]
                    },
                    "modelProvider": "OPENAI",
                    "modelName": "gpt-4",
                    "invocationParameters": {},
                    "tools": None,
                },
                "repetitions": 1,
            }
        }
        with custom_vcr.use_cassette():
            payloads = [
                payload
                async for payload in gql_client.subscription(
                    query=self.QUERY,
                    variables=variables,
                    operation_name="ChatCompletionSubscription",
                )
            ]

        # check subscription payloads
        assert payloads
        assert (last_payload := payloads.pop())["chatCompletion"][
            "__typename"
        ] == ChatCompletionSubscriptionResult.__name__
        assert all(
            payload["chatCompletion"]["__typename"] == TextChunk.__name__ for payload in payloads
        )
        response_text = "".join(payload["chatCompletion"]["content"] for payload in payloads)
        assert "sunny" in response_text.lower()
        subscription_span = last_payload["chatCompletion"]["span"]
        span_id = subscription_span["id"]

        # query for the span via the node interface to ensure that the span
        # recorded in the db contains identical information as the span emitted
        # by the subscription
        response = await gql_client.execute(
            query=self.QUERY, variables={"spanId": span_id}, operation_name="SpanQuery"
        )
        assert (data := response.data) is not None
        span = data["span"]
        assert json.loads(attributes := span.pop("attributes")) == json.loads(
            subscription_span.pop("attributes")
        )
        attributes = dict(flatten(json.loads(attributes)))
        _assert_spans_equal(span, subscription_span)

        # check attributes
        assert span.pop("id") == span_id
        assert span.pop("name") == "ChatCompletion"
        assert span.pop("statusCode") == "OK"
        assert not span.pop("statusMessage")
        assert span.pop("startTime")
        assert span.pop("endTime")
        assert isinstance(span.pop("latencyMs"), float)
        assert span.pop("parentId") is None
        assert span.pop("spanKind") == "llm"
        assert (context := span.pop("context")).pop("spanId")
        assert context.pop("traceId")
        assert not context
        assert span.pop("metadata") is None
        assert span.pop("numDocuments") == 0
        assert isinstance(token_count_total := span.pop("tokenCountTotal"), int)
        assert isinstance(token_count_prompt := span.pop("tokenCountPrompt"), int)
        assert isinstance(token_count_completion := span.pop("tokenCountCompletion"), int)
        assert token_count_prompt > 0
        assert token_count_completion > 0
        assert token_count_total == token_count_prompt + token_count_completion
        assert (input := span.pop("input")).pop("mimeType") == "json"
        assert (input_value := input.pop("value"))
        assert not input
        assert "api_key" not in input_value
        assert "apiKey" not in input_value
        assert (output := span.pop("output")).pop("mimeType") == "text"
        assert output.pop("value")
        assert not output
        assert not span.pop("events")
        assert isinstance(
            cumulative_token_count_total := span.pop("cumulativeTokenCountTotal"), float
        )
        assert isinstance(
            cumulative_token_count_prompt := span.pop("cumulativeTokenCountPrompt"), float
        )
        assert isinstance(
            cumulative_token_count_completion := span.pop("cumulativeTokenCountCompletion"), float
        )
        assert cumulative_token_count_total == token_count_total
        assert cumulative_token_count_prompt == token_count_prompt
        assert cumulative_token_count_completion == token_count_completion
        assert span.pop("propagatedStatusCode") == "OK"
        assert not span

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert isinstance(invocation_paramaters := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(invocation_paramaters) == {}
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4"
        assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == token_count_total
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == token_count_prompt
        assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == token_count_completion
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == 0
        assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING) == 0
        assert attributes.pop(INPUT_VALUE)
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE)
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert (llm_input_messages := attributes.pop(LLM_INPUT_MESSAGES))
        assert len(llm_input_messages) == 3
        llm_input_message = llm_input_messages[0]["message"]
        assert llm_input_message == {
            "content": "How's the weather in San Francisco?",
            "role": "user",
        }
        llm_input_message = llm_input_messages[1]["message"]
        assert llm_input_message["content"] == ""
        assert llm_input_message["role"] == "ai"
        assert llm_input_message["tool_calls"] == [
            {
                "tool_call": {
                    "id": tool_call_id,
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "San Francisco"}',
                    },
                }
            }
        ]
        llm_input_message = llm_input_messages[2]["message"]
        assert llm_input_message == {
            "content": "sunny",
            "role": "tool",
            "tool_call_id": tool_call_id,
        }
        assert attributes.pop(LLM_OUTPUT_MESSAGES) == [
            {
                "message": {
                    "role": "assistant",
                    "content": response_text,
                }
            }
        ]
        assert attributes.pop(LLM_PROVIDER) == "openai"
        assert attributes.pop(LLM_SYSTEM) == "openai"
        assert attributes.pop(URL_FULL) == "https://api.openai.com/v1/chat/completions"
        assert attributes.pop(URL_PATH) == "chat/completions"
        assert not attributes

    async def test_anthropic_text_response_emits_expected_payloads_and_records_expected_span(
        self,
        gql_client: AsyncGraphQLClient,
        anthropic_api_key: str,
        custom_vcr: CustomVCR,
    ) -> None:
        variables = {
            "input": {
                "promptVersion": {
                    "templateFormat": "NONE",
                    "template": {
                        "messages": [
                            {
                                "role": "USER",
                                "content": [
                                    {
                                        "text": {
                                            "text": "Who won the World Cup in 2018? Answer in one word"
                                        }
                                    }
                                ],
                            }
                        ]
                    },
                    "modelProvider": "ANTHROPIC",
                    "modelName": "claude-3-5-sonnet-20240620",
                    "invocationParameters": {"temperature": 0.1, "max_tokens": 1024},
                    "tools": None,
                },
                "repetitions": 1,
            },
        }
        with custom_vcr.use_cassette():
            payloads = [
                payload
                async for payload in gql_client.subscription(
                    query=self.QUERY,
                    variables=variables,
                    operation_name="ChatCompletionSubscription",
                )
            ]

        # check subscription payloads
        assert payloads
        assert (last_payload := payloads.pop())["chatCompletion"][
            "__typename"
        ] == ChatCompletionSubscriptionResult.__name__
        assert all(
            payload["chatCompletion"]["__typename"] == TextChunk.__name__ for payload in payloads
        )
        response_text = "".join(payload["chatCompletion"]["content"] for payload in payloads)
        assert "france" in response_text.lower()
        subscription_span = last_payload["chatCompletion"]["span"]
        span_id = subscription_span["id"]

        # query for the span via the node interface to ensure that the span
        # recorded in the db contains identical information as the span emitted
        # by the subscription
        response = await gql_client.execute(
            query=self.QUERY, variables={"spanId": span_id}, operation_name="SpanQuery"
        )
        assert (data := response.data) is not None
        span = data["span"]
        assert json.loads(attributes := span.pop("attributes")) == json.loads(
            subscription_span.pop("attributes")
        )
        attributes = dict(flatten(json.loads(attributes)))
        _assert_spans_equal(span, subscription_span)

        # check attributes
        assert span.pop("id") == span_id
        assert span.pop("name") == "ChatCompletion"
        assert span.pop("statusCode") == "OK"
        assert not span.pop("statusMessage")
        assert span.pop("startTime")
        assert span.pop("endTime")
        assert isinstance(span.pop("latencyMs"), float)
        assert span.pop("parentId") is None
        assert span.pop("spanKind") == "llm"
        assert (context := span.pop("context")).pop("spanId")
        assert context.pop("traceId")
        assert not context
        assert span.pop("metadata") is None
        assert span.pop("numDocuments") == 0
        assert isinstance(token_count_total := span.pop("tokenCountTotal"), int)
        assert isinstance(token_count_prompt := span.pop("tokenCountPrompt"), int)
        assert isinstance(token_count_completion := span.pop("tokenCountCompletion"), int)
        assert token_count_prompt > 0
        assert token_count_completion > 0
        assert token_count_total == token_count_prompt + token_count_completion
        assert (input := span.pop("input")).pop("mimeType") == "json"
        assert (input_value := input.pop("value"))
        assert not input
        assert "api_key" not in input_value
        assert "apiKey" not in input_value
        assert (output := span.pop("output")).pop("mimeType") == "json"
        assert output.pop("value")
        assert not output
        assert not span.pop("events")
        assert isinstance(
            cumulative_token_count_total := span.pop("cumulativeTokenCountTotal"), float
        )
        assert isinstance(
            cumulative_token_count_prompt := span.pop("cumulativeTokenCountPrompt"), float
        )
        assert isinstance(
            cumulative_token_count_completion := span.pop("cumulativeTokenCountCompletion"), float
        )
        assert cumulative_token_count_total == token_count_total
        assert cumulative_token_count_prompt == token_count_prompt
        assert cumulative_token_count_completion == token_count_completion
        assert span.pop("propagatedStatusCode") == "OK"
        assert not span

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(LLM_MODEL_NAME) == "claude-3-5-sonnet-20240620"
        assert isinstance(invocation_paramaters := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(invocation_paramaters) == {"temperature": 0.1, "max_tokens": 1024}
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == token_count_prompt
        assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == token_count_completion
        assert attributes.pop(INPUT_VALUE)
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE)
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(LLM_INPUT_MESSAGES) == [
            {
                "message": {
                    "role": "user",
                    "content": "Who won the World Cup in 2018? Answer in one word",
                }
            }
        ]
        assert attributes.pop(LLM_OUTPUT_MESSAGES) == [
            {
                "message": {
                    "role": "assistant",
                    "content": response_text,
                }
            }
        ]
        assert attributes.pop(LLM_PROVIDER) == "anthropic"
        assert attributes.pop(LLM_SYSTEM) == "anthropic"
        assert attributes.pop(URL_FULL) == "https://api.anthropic.com/v1/messages"
        assert attributes.pop(URL_PATH) == "v1/messages"
        assert not attributes

    async def test_all_spans_yielded_when_number_of_examples_exceeds_batch_size(
        self,
        gql_client: AsyncGraphQLClient,
        openai_api_key: str,
        cities_and_countries: list[tuple[str, str]],
        playground_city_and_country_dataset: None,
        custom_vcr: CustomVCR,
        db: DbSessionFactory,
    ) -> None:
        dataset_id = str(GlobalID(type_name=Dataset.__name__, node_id=str(1)))
        version_id = str(GlobalID(type_name=DatasetVersion.__name__, node_id=str(1)))
        variables = {
            "input": {
                "datasetId": dataset_id,
                "datasetVersionId": version_id,
                "promptVersion": {
                    "templateFormat": "F_STRING",
                    "template": {
                        "messages": [
                            {
                                "role": "USER",
                                "content": [
                                    {
                                        "text": {
                                            "text": (
                                                "What country is {city} in? "
                                                "Answer with the country name only without punctuation."
                                            )
                                        }
                                    }
                                ],
                            }
                        ]
                    },
                    "modelProvider": "OPENAI",
                    "modelName": "gpt-4",
                    "invocationParameters": {},
                    "tools": None,
                },
                "repetitions": 1,
            }
        }
        payloads: dict[Optional[str], list[Any]] = {}
        custom_vcr.register_matcher(
            _request_bodies_contain_same_city.__name__, _request_bodies_contain_same_city
        )  # a custom request matcher is needed since the requests are concurrent
        with custom_vcr.use_cassette(match_on=[_request_bodies_contain_same_city.__name__]):
            async for payload in gql_client.subscription(
                query=self.QUERY,
                variables=variables,
                operation_name="ChatCompletionOverDatasetSubscription",
            ):
                if (
                    dataset_example_id := payload["chatCompletionOverDataset"]["datasetExampleId"]
                ) not in payloads:
                    payloads[dataset_example_id] = []
                payloads[dataset_example_id].append(payload)

        # check subscription payloads
        cities_to_countries = dict(cities_and_countries)
        num_examples = len(cities_to_countries)
        example_ids = [
            str(GlobalID(type_name=DatasetExample.__name__, node_id=str(index)))
            for index in range(1, num_examples + 1)
        ]
        assert set(payloads.keys()) == set(example_ids) | {None}

        # check span payloads
        for example_id in example_ids:
            assert (span_payload := payloads[example_id].pop()["chatCompletionOverDataset"])[
                "__typename"
            ] == ChatCompletionSubscriptionResult.__name__
            assert all(
                payload["chatCompletionOverDataset"]["__typename"] == TextChunk.__name__
                for payload in payloads[example_id]
            )
            assert (span := span_payload["span"])
            assert isinstance(span["attributes"], str)
            attributes = json.loads(span["attributes"])
            assert isinstance(
                input_messages := get_attribute_value(attributes, LLM_INPUT_MESSAGES),
                list,
            )
            assert len(input_messages) == 1
            assert isinstance(input_message_content := input_messages[0]["message"]["content"], str)
            assert (city := _extract_city(input_message_content)) in cities_to_countries
            assert isinstance(
                output_messages := get_attribute_value(attributes, LLM_OUTPUT_MESSAGES),
                list,
            )
            assert len(output_messages) == 1
            assert isinstance(
                output_message_content := output_messages[0]["message"]["content"], str
            )
            assert output_message_content == cities_to_countries[city]
            response_text = "".join(
                payload["chatCompletionOverDataset"]["content"] for payload in payloads[example_id]
            )
            assert response_text == output_message_content

        # check experiment payload
        assert len(payloads[None]) == 1
        assert (experiment := payloads[None].pop()["chatCompletionOverDataset"]["experiment"])
        experiment_id = experiment["id"]
        assert isinstance(experiment_id, str)

        async with db() as session:
            await verify_experiment_examples_junction_table(session, experiment_id)

    async def test_experiment_with_single_split_filters_examples(
        self,
        gql_client: AsyncGraphQLClient,
        openai_api_key: str,
        playground_dataset_with_splits: None,
        custom_vcr: CustomVCR,
        db: DbSessionFactory,
    ) -> None:
        """Test that providing a single split ID filters examples correctly."""
        dataset_id = str(GlobalID(type_name=Dataset.__name__, node_id=str(1)))
        version_id = str(GlobalID(type_name=DatasetVersion.__name__, node_id=str(1)))
        train_split_id = str(GlobalID(type_name="DatasetSplit", node_id=str(1)))

        variables = {
            "input": {
                "datasetId": dataset_id,
                "datasetVersionId": version_id,
                "promptVersion": {
                    "templateFormat": "F_STRING",
                    "template": {
                        "messages": [
                            {
                                "role": "USER",
                                "content": [
                                    {
                                        "text": {
                                            "text": "What country is {city} in? Answer in one word, no punctuation."
                                        }
                                    }
                                ],
                            }
                        ]
                    },
                    "modelProvider": "OPENAI",
                    "modelName": "gpt-4",
                    "invocationParameters": {},
                    "tools": None,
                },
                "repetitions": 1,
                "splitIds": [train_split_id],  # Only train split
            }
        }

        payloads: dict[Optional[str], list[Any]] = {}
        custom_vcr.register_matcher(
            _request_bodies_contain_same_city.__name__, _request_bodies_contain_same_city
        )
        with custom_vcr.use_cassette(match_on=[_request_bodies_contain_same_city.__name__]):
            async for payload in gql_client.subscription(
                query=self.QUERY,
                variables=variables,
                operation_name="ChatCompletionOverDatasetSubscription",
            ):
                if (
                    dataset_example_id := payload["chatCompletionOverDataset"]["datasetExampleId"]
                ) not in payloads:
                    payloads[dataset_example_id] = []
                payloads[dataset_example_id].append(payload)

        # Should only have examples 1, 2, 3 (train split) + experiment payload
        # Examples 4 and 5 (test split) should NOT be present
        train_example_ids = [
            str(GlobalID(type_name=DatasetExample.__name__, node_id=str(i))) for i in range(1, 4)
        ]
        test_example_ids = [
            str(GlobalID(type_name=DatasetExample.__name__, node_id=str(i))) for i in range(4, 6)
        ]

        assert set(payloads.keys()) == set(train_example_ids) | {None}
        for test_id in test_example_ids:
            assert test_id not in payloads, f"Test example {test_id} should not be in results"

        # Verify experiment payload exists
        assert len(payloads[None]) == 1
        assert (experiment_payload := payloads[None][0]["chatCompletionOverDataset"])[
            "__typename"
        ] == ChatCompletionSubscriptionExperiment.__name__
        experiment_id = experiment_payload["experiment"]["id"]

        # Verify experiment has the correct split association in DB
        async with db() as session:
            _, exp_id = from_global_id(GlobalID.from_id(experiment_id))
            result = await session.execute(
                select(models.ExperimentDatasetSplit).where(
                    models.ExperimentDatasetSplit.experiment_id == exp_id
                )
            )
            split_links = result.scalars().all()
            assert len(split_links) == 1
            assert split_links[0].dataset_split_id == 1  # train split

    async def test_experiment_with_multiple_splits(
        self,
        gql_client: AsyncGraphQLClient,
        openai_api_key: str,
        playground_dataset_with_splits: None,
        custom_vcr: CustomVCR,
        db: DbSessionFactory,
    ) -> None:
        """Test that providing multiple split IDs includes examples from all specified splits."""
        dataset_id = str(GlobalID(type_name=Dataset.__name__, node_id=str(1)))
        version_id = str(GlobalID(type_name=DatasetVersion.__name__, node_id=str(1)))
        train_split_id = str(GlobalID(type_name="DatasetSplit", node_id=str(1)))
        test_split_id = str(GlobalID(type_name="DatasetSplit", node_id=str(2)))

        variables = {
            "input": {
                "datasetId": dataset_id,
                "datasetVersionId": version_id,
                "promptVersion": {
                    "templateFormat": "F_STRING",
                    "template": {
                        "messages": [
                            {
                                "role": "USER",
                                "content": [
                                    {
                                        "text": {
                                            "text": "What country is {city} in? Answer in one word, no punctuation."
                                        }
                                    }
                                ],
                            }
                        ]
                    },
                    "modelProvider": "OPENAI",
                    "modelName": "gpt-4",
                    "invocationParameters": {},
                    "tools": None,
                },
                "repetitions": 1,
                "splitIds": [train_split_id, test_split_id],  # Both splits
            }
        }

        payloads: dict[Optional[str], list[Any]] = {}
        custom_vcr.register_matcher(
            _request_bodies_contain_same_city.__name__, _request_bodies_contain_same_city
        )
        with custom_vcr.use_cassette(match_on=[_request_bodies_contain_same_city.__name__]):
            async for payload in gql_client.subscription(
                query=self.QUERY,
                variables=variables,
                operation_name="ChatCompletionOverDatasetSubscription",
            ):
                if (
                    dataset_example_id := payload["chatCompletionOverDataset"]["datasetExampleId"]
                ) not in payloads:
                    payloads[dataset_example_id] = []
                payloads[dataset_example_id].append(payload)

        # Should have all examples 1-5 + experiment payload
        all_example_ids = [
            str(GlobalID(type_name=DatasetExample.__name__, node_id=str(i))) for i in range(1, 6)
        ]
        assert set(payloads.keys()) == set(all_example_ids) | {None}

        # Verify experiment has both split associations in DB
        assert len(payloads[None]) == 1
        experiment_id = payloads[None][0]["chatCompletionOverDataset"]["experiment"]["id"]

        async with db() as session:
            _, exp_id = from_global_id(GlobalID.from_id(experiment_id))
            result = await session.execute(
                select(models.ExperimentDatasetSplit)
                .where(models.ExperimentDatasetSplit.experiment_id == exp_id)
                .order_by(models.ExperimentDatasetSplit.dataset_split_id)
            )
            split_links = result.scalars().all()
            assert len(split_links) == 2
            assert split_links[0].dataset_split_id == 1  # train split
            assert split_links[1].dataset_split_id == 2  # test split

    async def test_experiment_without_splits_includes_all_examples(
        self,
        gql_client: AsyncGraphQLClient,
        openai_api_key: str,
        playground_dataset_with_splits: None,
        custom_vcr: CustomVCR,
        db: DbSessionFactory,
    ) -> None:
        """Test backward compatibility: when no splits are specified, all examples are included."""
        dataset_id = str(GlobalID(type_name=Dataset.__name__, node_id=str(1)))
        version_id = str(GlobalID(type_name=DatasetVersion.__name__, node_id=str(1)))

        variables = {
            "input": {
                "datasetId": dataset_id,
                "datasetVersionId": version_id,
                "promptVersion": {
                    "templateFormat": "F_STRING",
                    "template": {
                        "messages": [
                            {
                                "role": "USER",
                                "content": [
                                    {
                                        "text": {
                                            "text": "What country is {city} in? Answer in one word, no punctuation."
                                        }
                                    }
                                ],
                            }
                        ]
                    },
                    "modelProvider": "OPENAI",
                    "modelName": "gpt-4",
                    "invocationParameters": {},
                    "tools": None,
                },
                "repetitions": 1,
                # No splitIds provided
            }
        }

        payloads: dict[Optional[str], list[Any]] = {}
        custom_vcr.register_matcher(
            _request_bodies_contain_same_city.__name__, _request_bodies_contain_same_city
        )
        with custom_vcr.use_cassette(match_on=[_request_bodies_contain_same_city.__name__]):
            async for payload in gql_client.subscription(
                query=self.QUERY,
                variables=variables,
                operation_name="ChatCompletionOverDatasetSubscription",
            ):
                if (
                    dataset_example_id := payload["chatCompletionOverDataset"]["datasetExampleId"]
                ) not in payloads:
                    payloads[dataset_example_id] = []
                payloads[dataset_example_id].append(payload)

        # Should have all examples 1-5 + experiment payload
        all_example_ids = [
            str(GlobalID(type_name=DatasetExample.__name__, node_id=str(i))) for i in range(1, 6)
        ]
        assert set(payloads.keys()) == set(all_example_ids) | {None}

        # Verify experiment has NO split associations in DB
        assert len(payloads[None]) == 1
        experiment_id = payloads[None][0]["chatCompletionOverDataset"]["experiment"]["id"]

        async with db() as session:
            _, exp_id = from_global_id(GlobalID.from_id(experiment_id))
            result = await session.execute(
                select(models.ExperimentDatasetSplit).where(
                    models.ExperimentDatasetSplit.experiment_id == exp_id
                )
            )
            split_links = result.scalars().all()
            assert len(split_links) == 0  # No splits associated

    async def test_evaluator_emits_evaluation_chunk_and_persists_annotation(
        self,
        gql_client: AsyncGraphQLClient,
        openai_api_key: str,
        single_example_dataset: models.Dataset,
        assign_correctness_llm_evaluator_to_dataset: Callable[
            [int], Awaitable[models.DatasetEvaluators]
        ],
        assign_exact_match_builtin_evaluator_to_dataset: Callable[
            [int], Awaitable[models.DatasetEvaluators]
        ],
        custom_vcr: CustomVCR,
        db: DbSessionFactory,
    ) -> None:
        llm_dataset_evaluator = await assign_correctness_llm_evaluator_to_dataset(
            single_example_dataset.id
        )
        llm_evaluator_gid = str(
            GlobalID(type_name=DatasetEvaluator.__name__, node_id=str(llm_dataset_evaluator.id))
        )
        builtin_dataset_evaluator = await assign_exact_match_builtin_evaluator_to_dataset(
            single_example_dataset.id
        )
        builtin_evaluator_gid = str(
            GlobalID(
                type_name=DatasetEvaluator.__name__,
                node_id=str(builtin_dataset_evaluator.id),
            )
        )

        dataset_gid = str(
            GlobalID(type_name=Dataset.__name__, node_id=str(single_example_dataset.id))
        )
        version_gid = str(GlobalID(type_name=DatasetVersion.__name__, node_id=str(1)))
        variables = {
            "input": {
                "datasetId": dataset_gid,
                "datasetVersionId": version_gid,
                "promptVersion": {
                    "templateFormat": "F_STRING",
                    "template": {
                        "messages": [
                            {
                                "role": "USER",
                                "content": [
                                    {
                                        "text": {
                                            "text": "What country is {city} in? Answer in one word, no punctuation."
                                        }
                                    }
                                ],
                            }
                        ]
                    },
                    "modelProvider": "OPENAI",
                    "modelName": "gpt-4",
                    "invocationParameters": {},
                    "tools": None,
                },
                "repetitions": 1,
                "tracingEnabled": True,
                "evaluators": [
                    {
                        "id": llm_evaluator_gid,
                        "name": "correctness",
                        "inputMapping": {"literalMapping": {}, "pathMapping": {}},
                    },
                    {
                        "id": builtin_evaluator_gid,
                        "name": "exact-match",
                        "inputMapping": {"literalMapping": {}, "pathMapping": {}},
                    },
                ],
            }
        }

        payloads: dict[Optional[str], list[Any]] = {}
        evaluation_chunks: list[Any] = []

        with custom_vcr.use_cassette():
            async for payload in gql_client.subscription(
                query=self.QUERY,
                variables=variables,
                operation_name="ChatCompletionOverDatasetSubscription",
            ):
                typename = payload["chatCompletionOverDataset"]["__typename"]
                if typename == EvaluationChunk.__name__:
                    evaluation_chunks.append(payload["chatCompletionOverDataset"])
                else:
                    dataset_example_id = payload["chatCompletionOverDataset"]["datasetExampleId"]
                    if dataset_example_id not in payloads:
                        payloads[dataset_example_id] = []
                    payloads[dataset_example_id].append(payload)

        assert len(evaluation_chunks) == 2
        llm_chunk = next(
            chunk
            for chunk in evaluation_chunks
            if chunk["experimentRunEvaluation"]["name"] == "correctness"
        )
        assert llm_chunk["__typename"] == EvaluationChunk.__name__
        llm_annotation = llm_chunk["experimentRunEvaluation"]
        assert llm_annotation is not None
        assert llm_annotation["annotatorKind"] == "LLM"
        builtin_chunk = next(
            chunk
            for chunk in evaluation_chunks
            if chunk["experimentRunEvaluation"]["name"] == "exact-match"
        )
        assert builtin_chunk["__typename"] == EvaluationChunk.__name__
        builtin_annotation = builtin_chunk["experimentRunEvaluation"]
        assert builtin_annotation is not None
        assert builtin_annotation["annotatorKind"] == "CODE"

        async with db() as session:
            result = await session.execute(select(models.ExperimentRunAnnotation))
            annotations = result.scalars().all()
            assert len(annotations) == 2

            llm_annotation_orm = next(
                annotation for annotation in annotations if annotation.name == "correctness"
            )
            assert llm_annotation_orm.annotator_kind == "LLM"
            assert llm_annotation_orm.experiment_run_id is not None

            builtin_annotation_orm = next(
                annotation for annotation in annotations if annotation.name == "exact-match"
            )
            assert builtin_annotation_orm.annotator_kind == "CODE"
            assert builtin_annotation_orm.experiment_run_id is not None

            llm_traces_result = await session.scalars(
                select(models.Trace).where(
                    models.Trace.project_rowid == llm_dataset_evaluator.project_id,
                )
            )
            llm_traces = llm_traces_result.all()
            assert len(llm_traces) == 1
            llm_evaluator_trace = llm_traces[0]

            llm_spans_result = await session.execute(
                select(models.Span).where(
                    models.Span.trace_rowid == llm_evaluator_trace.id,
                )
            )
            llm_spans = llm_spans_result.scalars().all()
            assert len(llm_spans) == 5

            builtin_traces_result = await session.scalars(
                select(models.Trace).where(
                    models.Trace.project_rowid == builtin_dataset_evaluator.project_id,
                )
            )
            builtin_traces = builtin_traces_result.all()
            assert len(builtin_traces) == 1
            builtin_evaluator_trace = builtin_traces[0]

            builtin_spans_result = await session.execute(
                select(models.Span).where(
                    models.Span.trace_rowid == builtin_evaluator_trace.id,
                )
            )
            builtin_spans = builtin_spans_result.scalars().all()
            assert len(builtin_spans) == 4

            # Parse LLM evaluator spans
            llm_evaluator_span = None
            llm_input_mapping_span = None
            llm_prompt_span = None
            llm_llm_span = None
            llm_parse_span = None
            for span in llm_spans:
                if span.span_kind == "EVALUATOR":
                    llm_evaluator_span = span
                elif span.span_kind == "CHAIN" and span.name == "Input Mapping":
                    llm_input_mapping_span = span
                elif span.span_kind == "PROMPT" and span.name.startswith("Prompt:"):
                    llm_prompt_span = span
                elif span.span_kind == "LLM":
                    llm_llm_span = span
                elif span.span_kind == "CHAIN" and span.name == "Parse Eval Result":
                    llm_parse_span = span

            assert llm_evaluator_span is not None
            assert llm_evaluator_span.parent_id is None
            assert llm_input_mapping_span is not None
            assert llm_input_mapping_span.parent_id == llm_evaluator_span.span_id
            assert llm_prompt_span is not None
            assert llm_prompt_span.parent_id == llm_evaluator_span.span_id
            assert llm_llm_span is not None
            assert llm_llm_span.parent_id == llm_evaluator_span.span_id
            assert llm_parse_span is not None
            assert llm_parse_span.parent_id == llm_evaluator_span.span_id

            # LLM evaluator span
            assert llm_evaluator_span.name == "Evaluator: correctness"
            assert llm_evaluator_span.span_kind == "EVALUATOR"
            attributes = dict(flatten(llm_evaluator_span.attributes, recurse_on_sequence=True))
            assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "EVALUATOR"
            raw_input_value = attributes.pop(INPUT_VALUE)
            assert raw_input_value is not None
            input_value = json.loads(raw_input_value)
            assert set(input_value.keys()) == {"input", "output", "reference", "metadata"}
            assert attributes.pop(INPUT_MIME_TYPE) == JSON
            raw_output_value = attributes.pop(OUTPUT_VALUE)
            assert raw_output_value is not None
            output_value = json.loads(raw_output_value)
            assert set(output_value.keys()) == {"results"}
            assert len(output_value["results"]) == 1
            assert set(output_value["results"][0].keys()) == {
                "name",
                "label",
                "score",
                "explanation",
            }
            assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
            assert not attributes
            assert not llm_evaluator_span.events
            assert llm_evaluator_span.status_code == "OK"
            assert not llm_evaluator_span.status_message

            # input mapping span
            assert llm_input_mapping_span.name == "Input Mapping"
            assert llm_input_mapping_span.span_kind == "CHAIN"
            assert llm_input_mapping_span.status_code == "OK"
            assert not llm_input_mapping_span.status_message
            assert not llm_input_mapping_span.events
            attributes = dict(flatten(llm_input_mapping_span.attributes, recurse_on_sequence=True))
            assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "CHAIN"
            input_value = json.loads(attributes.pop(INPUT_VALUE))
            assert input_value == {
                "input_mapping": {
                    "path_mapping": {"input": "$.input", "output": "$.output"},
                    "literal_mapping": {},
                },
                "template_variables": {
                    "input": {"city": "Paris"},
                    "output": {
                        "available_tools": [],
                        "messages": [{"content": "France", "role": "assistant"}],
                    },
                    "reference": {"country": "France"},
                    "metadata": {},
                },
            }
            assert attributes.pop(INPUT_MIME_TYPE) == JSON
            output_value = json.loads(attributes.pop(OUTPUT_VALUE))
            assert output_value == {
                "input": "{'city': 'Paris'}",
                "output": "{'messages': [{'role': 'assistant', 'content': 'France'}], 'available_tools': []}",
            }
            assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
            assert not attributes

            # Prompt span
            assert llm_prompt_span.name == "Prompt: correctness-prompt"
            assert llm_prompt_span.span_kind == "PROMPT"
            assert llm_prompt_span.status_code == "OK"
            assert not llm_prompt_span.status_message
            assert not llm_prompt_span.events
            attributes = dict(flatten(llm_prompt_span.attributes, recurse_on_sequence=True))
            assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "PROMPT"
            input_value = json.loads(attributes.pop(INPUT_VALUE))
            assert input_value == {
                "input": "{'city': 'Paris'}",
                "output": "{'messages': [{'role': 'assistant', 'content': 'France'}], 'available_tools': []}",
            }
            assert attributes.pop(INPUT_MIME_TYPE) == JSON
            output_value = json.loads(attributes.pop(OUTPUT_VALUE))
            assert output_value == {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an evaluator that assesses the correctness of outputs.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Input: {'city': 'Paris'}\n\n"
                            "Output: {'messages': [{'role': 'assistant', 'content': 'France'}], "
                            "'available_tools': []}\n\n"
                            "Is this output correct?"
                        ),
                    },
                ]
            }
            assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
            assert not attributes

            # llm span
            assert llm_llm_span.name == "ChatCompletion"
            assert llm_llm_span.span_kind == "LLM"
            assert llm_llm_span.status_code == "OK"
            assert not llm_llm_span.status_message
            assert llm_llm_span.llm_token_count_prompt is not None
            assert llm_llm_span.llm_token_count_prompt > 0
            assert llm_llm_span.llm_token_count_completion is not None
            assert llm_llm_span.llm_token_count_completion > 0
            assert llm_llm_span.cumulative_llm_token_count_prompt > 0
            assert llm_llm_span.cumulative_llm_token_count_completion > 0
            attributes = dict(flatten(llm_llm_span.attributes, recurse_on_sequence=True))
            assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
            assert attributes.pop(LLM_MODEL_NAME) == "gpt-4"
            assert attributes.pop(LLM_PROVIDER) == "openai"
            assert attributes.pop(LLM_SYSTEM) == "openai"
            assert attributes.pop(URL_FULL) == "https://api.openai.com/v1/chat/completions"
            assert attributes.pop(URL_PATH) == "chat/completions"
            assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
            assert (
                "evaluator" in attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}").lower()
            )
            assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
            assert "Paris" in attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}")
            token_count_attribute_keys = [
                attribute_key
                for attribute_key in attributes
                if attribute_key.startswith("llm.token_count.")
            ]
            for key in token_count_attribute_keys:
                assert isinstance(attributes.pop(key), int)
            assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
            raw_output_value = attributes.pop(OUTPUT_VALUE)
            output_value = json.loads(raw_output_value)
            assert output_value == {
                "id": "chatcmpl-DQenkImJhoauMjqwwwqLeZGXyTXmE",
                "object": "chat.completion",
                "created": 1775245740,
                "model": "gpt-4-0613",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_4q0aw0YpXJlp7wsm38rZs30l",
                                    "type": "function",
                                    "function": {
                                        "name": "correctness",
                                        "arguments": '{\n"label": "incorrect"\n}',
                                    },
                                }
                            ],
                            "annotations": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 104,
                    "completion_tokens": 11,
                    "total_tokens": 115,
                    "prompt_tokens_details": {
                        "cached_tokens": 0,
                        "audio_tokens": 0,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                        "audio_tokens": 0,
                        "accepted_prediction_tokens": 0,
                        "rejected_prediction_tokens": 0,
                    },
                },
                "service_tier": "default",
            }
            assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
            assert isinstance(
                attributes.pop(
                    f"{LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_ID}"
                ),
                str,
            )
            assert (
                attributes.pop(
                    f"{LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}"
                )
                == "correctness"
            )
            arguments = attributes.pop(
                f"{LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS}"
            )
            assert arguments is not None
            assert json.loads(arguments) == {"label": "incorrect"}
            assert attributes.pop(INPUT_MIME_TYPE) == JSON
            assert isinstance(attributes.pop(INPUT_VALUE), str)
            assert isinstance(attributes.pop(LLM_INVOCATION_PARAMETERS), str)
            tool_json_schema = json.loads(attributes.pop(f"{LLM_TOOLS}.0.{TOOL_JSON_SCHEMA}"))
            assert tool_json_schema["type"] == "function"
            assert tool_json_schema["function"]["name"] == "correctness"
            assert not attributes

            # span costs for evaluator trace
            span_costs_result = await session.execute(
                select(models.SpanCost).where(models.SpanCost.trace_rowid == llm_evaluator_trace.id)
            )
            span_costs = span_costs_result.scalars().all()
            assert len(span_costs) == 1
            span_cost = span_costs[0]
            assert span_cost.span_rowid == llm_llm_span.id
            assert span_cost.trace_rowid == llm_llm_span.trace_rowid
            assert span_cost.model_id is not None
            assert span_cost.span_start_time == llm_llm_span.start_time
            assert span_cost.total_cost is not None
            assert span_cost.total_cost > 0
            assert span_cost.total_tokens == (
                llm_llm_span.llm_token_count_prompt + llm_llm_span.llm_token_count_completion
            )
            assert span_cost.prompt_tokens == llm_llm_span.llm_token_count_prompt
            assert span_cost.prompt_cost is not None
            assert span_cost.prompt_cost > 0
            assert span_cost.completion_tokens == llm_llm_span.llm_token_count_completion
            assert span_cost.completion_cost is not None
            assert span_cost.completion_cost > 0

            # span cost details for evaluator trace
            span_cost_details_result = await session.execute(
                select(models.SpanCostDetail).where(
                    models.SpanCostDetail.span_cost_id == span_cost.id
                )
            )
            span_cost_details = span_cost_details_result.scalars().all()
            assert len(span_cost_details) >= 2
            input_detail = next(
                d for d in span_cost_details if d.is_prompt and d.token_type == "input"
            )
            output_detail = next(
                d for d in span_cost_details if not d.is_prompt and d.token_type == "output"
            )
            assert input_detail.span_cost_id == span_cost.id
            assert input_detail.token_type == "input"
            assert input_detail.is_prompt is True
            assert input_detail.tokens == llm_llm_span.llm_token_count_prompt
            assert input_detail.cost is not None
            assert input_detail.cost > 0
            assert input_detail.cost_per_token is not None
            assert output_detail.span_cost_id == span_cost.id
            assert output_detail.token_type == "output"
            assert output_detail.is_prompt is False
            assert output_detail.tokens == llm_llm_span.llm_token_count_completion
            assert output_detail.cost is not None
            assert output_detail.cost > 0
            assert output_detail.cost_per_token is not None

            # chain span
            assert llm_parse_span.name == "Parse Eval Result"
            assert llm_parse_span.span_kind == "CHAIN"
            assert llm_parse_span.status_code == "OK"
            assert not llm_parse_span.status_message
            assert not llm_parse_span.events
            attributes = dict(flatten(llm_parse_span.attributes, recurse_on_sequence=True))
            assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "CHAIN"
            input_value = json.loads(attributes.pop(INPUT_VALUE))
            assert set(input_value.keys()) == {"tool_calls", "output_configs"}
            tool_calls = input_value["tool_calls"]
            assert len(tool_calls) == 1
            tool_call = next(iter(tool_calls.values()))
            assert tool_call["name"] == "correctness"
            assert input_value["output_configs"] == {
                "correctness": {
                    "values": [
                        {"label": "correct", "score": 1.0},
                        {"label": "incorrect", "score": 0.0},
                    ]
                }
            }
            assert attributes.pop(INPUT_MIME_TYPE) == JSON
            output_value = json.loads(attributes.pop(OUTPUT_VALUE))
            assert output_value == {
                "results": [
                    {
                        "name": "correctness",
                        "label": "incorrect",
                        "score": 0.0,
                        "explanation": None,
                    }
                ]
            }
            assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
            assert not attributes

            # built-in evaluator spans
            builtin_evaluator_span = None
            builtin_input_mapping_span = None
            builtin_execution_span = None
            builtin_parse_span = None
            for span in builtin_spans:
                if span.span_kind == "EVALUATOR":
                    builtin_evaluator_span = span
                elif span.span_kind == "CHAIN":
                    if span.name == "Input Mapping":
                        builtin_input_mapping_span = span
                    elif span.name == "exact_match":
                        builtin_execution_span = span
                    elif span.name == "Parse Eval Result":
                        builtin_parse_span = span

            assert builtin_evaluator_span is not None
            assert builtin_input_mapping_span is not None
            assert builtin_execution_span is not None
            assert builtin_parse_span is not None

            # Verify span hierarchy
            assert builtin_evaluator_span.parent_id is None
            assert builtin_input_mapping_span.parent_id == builtin_evaluator_span.span_id
            assert builtin_execution_span.parent_id == builtin_evaluator_span.span_id
            assert builtin_parse_span.parent_id == builtin_evaluator_span.span_id

            # Built-in evaluator span
            assert builtin_evaluator_span.name == "Evaluator: exact-match"
            assert builtin_evaluator_span.span_kind == "EVALUATOR"
            assert builtin_evaluator_span.status_code == "OK"
            assert not builtin_evaluator_span.status_message
            assert not builtin_evaluator_span.events
            attributes = dict(flatten(builtin_evaluator_span.attributes, recurse_on_sequence=True))
            assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "EVALUATOR"
            assert json.loads(attributes.pop(INPUT_VALUE)) == {
                "input": {"city": "Paris"},
                "output": {
                    "messages": [{"role": "assistant", "content": "France"}],
                    "available_tools": [],
                },
                "reference": {"country": "France"},
                "metadata": {},
            }
            assert attributes.pop(INPUT_MIME_TYPE) == JSON
            assert json.loads(attributes.pop(OUTPUT_VALUE)) == {
                "label": "true",
                "score": 1.0,
                "explanation": "expected matches actual",
            }
            assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
            assert not attributes

            # Built-in input mapping span
            assert builtin_input_mapping_span.name == "Input Mapping"
            assert builtin_input_mapping_span.span_kind == "CHAIN"
            assert builtin_input_mapping_span.status_code == "OK"
            assert not builtin_input_mapping_span.status_message
            assert not builtin_input_mapping_span.events
            attributes = dict(
                flatten(builtin_input_mapping_span.attributes, recurse_on_sequence=True)
            )
            assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "CHAIN"
            assert json.loads(attributes.pop(INPUT_VALUE)) == {
                "input_mapping": {
                    "path_mapping": {"actual": "$.output.messages[0].content"},
                    "literal_mapping": {"expected": "France"},
                },
                "template_variables": {
                    "input": {"city": "Paris"},
                    "output": {
                        "messages": [{"role": "assistant", "content": "France"}],
                        "available_tools": [],
                    },
                    "reference": {"country": "France"},
                    "metadata": {},
                },
            }
            assert attributes.pop(INPUT_MIME_TYPE) == JSON
            assert json.loads(attributes.pop(OUTPUT_VALUE)) == {
                "expected": "France",
                "actual": "France",
            }
            assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
            assert not attributes

            assert builtin_execution_span.name == "exact_match"
            assert builtin_execution_span.span_kind == "CHAIN"
            assert builtin_execution_span.status_code == "OK"
            assert not builtin_execution_span.status_message
            assert not builtin_execution_span.events
            attributes = dict(flatten(builtin_execution_span.attributes, recurse_on_sequence=True))
            assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "CHAIN"
            assert json.loads(attributes.pop(INPUT_VALUE)) == {
                "expected": "France",
                "actual": "France",
                "case_sensitive": True,
            }
            assert attributes.pop(INPUT_MIME_TYPE) == JSON
            assert json.loads(attributes.pop(OUTPUT_VALUE)) is True
            assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
            assert not attributes

            # Built-in parse span (Parse Eval Result)
            assert builtin_parse_span.name == "Parse Eval Result"
            assert builtin_parse_span.span_kind == "CHAIN"
            assert not builtin_parse_span.status_message
            assert not builtin_parse_span.events
            attributes = dict(flatten(builtin_parse_span.attributes, recurse_on_sequence=True))
            assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "CHAIN"
            assert json.loads(attributes.pop(INPUT_VALUE)) is True
            assert attributes.pop(INPUT_MIME_TYPE) == JSON
            output_value = json.loads(attributes.pop(OUTPUT_VALUE))
            assert output_value == {
                "label": "true",
                "score": 1.0,
                "explanation": "expected matches actual",
            }
            assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
            assert not attributes

    async def test_evaluator_not_emitted_when_task_errors(
        self,
        gql_client: AsyncGraphQLClient,
        openai_api_key: str,
        single_example_dataset: models.Dataset,
        assign_correctness_llm_evaluator_to_dataset: Callable[
            [int], Awaitable[models.DatasetEvaluators]
        ],
        custom_vcr: CustomVCR,
        db: DbSessionFactory,
    ) -> None:
        dataset_evaluator = await assign_correctness_llm_evaluator_to_dataset(
            single_example_dataset.id
        )
        evaluator_gid = str(
            GlobalID(type_name=DatasetEvaluator.__name__, node_id=str(dataset_evaluator.id))
        )
        dataset_gid = str(
            GlobalID(type_name=Dataset.__name__, node_id=str(single_example_dataset.id))
        )
        version_gid = str(GlobalID(type_name=DatasetVersion.__name__, node_id=str(1)))
        variables = {
            "input": {
                "datasetId": dataset_gid,
                "datasetVersionId": version_gid,
                "promptVersion": {
                    "templateFormat": "F_STRING",
                    "template": {
                        "messages": [
                            {
                                "role": "USER",
                                "content": [
                                    {
                                        "text": {
                                            "text": "What country is {city} in? Answer in one word, no punctuation."
                                        }
                                    }
                                ],
                            }
                        ]
                    },
                    "modelProvider": "OPENAI",
                    "modelName": "gpt-nonexistent-model",
                    "invocationParameters": {},
                    "tools": None,
                },
                "repetitions": 1,
                "evaluators": [
                    {
                        "id": evaluator_gid,
                        "name": "correctness",
                        "inputMapping": {
                            "pathMapping": {
                                "input": "$.input",
                                "output": "$.output",
                            },
                        },
                    }
                ],
            }
        }

        error_chunks: list[Any] = []
        evaluation_chunks: list[Any] = []

        with custom_vcr.use_cassette():
            async for payload in gql_client.subscription(
                query=self.QUERY,
                variables=variables,
                operation_name="ChatCompletionOverDatasetSubscription",
            ):
                typename = payload["chatCompletionOverDataset"]["__typename"]
                if typename == ChatCompletionSubscriptionError.__name__:
                    error_chunks.append(payload["chatCompletionOverDataset"])
                elif typename == EvaluationChunk.__name__:
                    evaluation_chunks.append(payload["chatCompletionOverDataset"])

        # Verify we got an error chunk (message may mention model or be a connection/API error)
        assert len(error_chunks) == 1
        assert len(error_chunks[0]["message"]) > 0

        # Verify no evaluation chunks were emitted
        assert len(evaluation_chunks) == 0

        # Verify no experiment run annotations were persisted
        async with db() as session:
            result = await session.execute(select(models.ExperimentRunAnnotation))
            annotations = result.scalars().all()
            assert len(annotations) == 0

    async def test_builtin_evaluator_uses_name(
        self,
        gql_client: AsyncGraphQLClient,
        openai_api_key: str,
        single_example_dataset: models.Dataset,
        assign_exact_match_builtin_evaluator_to_dataset: Callable[
            ..., Awaitable[models.DatasetEvaluators]
        ],
        custom_vcr: CustomVCR,
        db: DbSessionFactory,
    ) -> None:
        """Test that builtin evaluators use name for annotation names in dataset runs."""
        builtin_dataset_evaluator = await assign_exact_match_builtin_evaluator_to_dataset(
            single_example_dataset.id,
            input_mapping=InputMapping(
                literal_mapping={"expected": "test", "actual": "test"},
                path_mapping={},
            ),
            output_configs=[
                CategoricalOutputConfig(
                    type=AnnotationType.CATEGORICAL.value,
                    name="my-dataset-exact-match",
                    optimization_direction=OptimizationDirection.MAXIMIZE,
                    values=[
                        CategoricalAnnotationValue(label="true", score=1.0),
                        CategoricalAnnotationValue(label="false", score=0.0),
                    ],
                )
            ],
        )
        evaluator_gid = str(
            GlobalID(
                type_name=DatasetEvaluator.__name__,
                node_id=str(builtin_dataset_evaluator.id),
            )
        )
        custom_name = "my-dataset-exact-match"
        dataset_gid = str(
            GlobalID(type_name=Dataset.__name__, node_id=str(single_example_dataset.id))
        )
        version_gid = str(GlobalID(type_name=DatasetVersion.__name__, node_id=str(1)))
        variables = {
            "input": {
                "datasetId": dataset_gid,
                "datasetVersionId": version_gid,
                "promptVersion": {
                    "templateFormat": "F_STRING",
                    "template": {
                        "messages": [
                            {
                                "role": "USER",
                                "content": [
                                    {
                                        "text": {
                                            "text": "What country is {city} in? Answer in one word, no punctuation."
                                        }
                                    }
                                ],
                            }
                        ]
                    },
                    "modelProvider": "OPENAI",
                    "modelName": "gpt-4",
                    "invocationParameters": {},
                    "tools": None,
                },
                "repetitions": 1,
                "evaluators": [
                    {
                        "id": evaluator_gid,
                        "name": custom_name,
                        "inputMapping": {"literalMapping": {}, "pathMapping": {}},
                    }
                ],
            }
        }

        evaluation_chunks: list[Any] = []

        custom_vcr.register_matcher(
            _request_bodies_contain_same_city.__name__, _request_bodies_contain_same_city
        )
        with custom_vcr.use_cassette():
            async for payload in gql_client.subscription(
                query=self.QUERY,
                variables=variables,
                operation_name="ChatCompletionOverDatasetSubscription",
            ):
                typename = payload["chatCompletionOverDataset"]["__typename"]
                if typename == EvaluationChunk.__name__:
                    evaluation_chunks.append(payload["chatCompletionOverDataset"])

        # Verify we got exactly 1 evaluation chunk with custom display name
        assert len(evaluation_chunks) == 1
        eval_chunk = evaluation_chunks[0]
        eval_annotation = eval_chunk["experimentRunEvaluation"]
        assert eval_annotation["name"] == custom_name
        assert eval_annotation["annotatorKind"] == "CODE"

        # Verify experiment run annotation was persisted with name
        async with db() as session:
            result = await session.execute(select(models.ExperimentRunAnnotation))
            annotations = result.scalars().all()
            assert len(annotations) == 1

            annotation = annotations[0]
            assert annotation.name == custom_name
            assert annotation.annotator_kind == "CODE"


# --- tests/unit/server/api/dataloaders/test_annotation_summaries.py ---

async def test_evaluation_summaries(
    db: DbSessionFactory,
    data_for_testing_dataloaders: None,
) -> None:
    start_time = datetime.fromisoformat("2021-01-01T00:00:10.000+00:00")
    end_time = datetime.fromisoformat("2021-01-01T00:10:00.000+00:00")
    pid = models.Trace.project_rowid
    async with db() as session:
        span_df = await session.run_sync(
            lambda s: pd.read_sql_query(
                select(
                    pid,
                    models.SpanAnnotation.name,
                    func.avg(models.SpanAnnotation.score).label("mean_score"),
                )
                .group_by(pid, models.SpanAnnotation.name)
                .order_by(pid, models.SpanAnnotation.name)
                .join_from(models.Trace, models.Span)
                .join_from(models.Span, models.SpanAnnotation)
                .where(models.Span.name.contains("_trace4_"))
                .where(models.SpanAnnotation.name.in_(("A", "C")))
                .where(start_time <= models.Span.start_time)
                .where(models.Span.start_time < end_time),
                s.connection(),
            )
        )
        trace_df = await session.run_sync(
            lambda s: pd.read_sql_query(
                select(
                    pid,
                    models.TraceAnnotation.name,
                    func.avg(models.TraceAnnotation.score).label("mean_score"),
                )
                .group_by(pid, models.TraceAnnotation.name)
                .order_by(pid, models.TraceAnnotation.name)
                .join_from(models.Trace, models.TraceAnnotation)
                .where(models.TraceAnnotation.name.in_(("B", "D")))
                .where(start_time <= models.Trace.start_time)
                .where(models.Trace.start_time < end_time),
                s.connection(),
            )
        )
    expected = trace_df.loc[:, "mean_score"].to_list() + span_df.loc[:, "mean_score"].to_list()
    kinds: list[Literal["span", "trace"]] = ["trace", "span"]
    session_filter_condition = None
    keys: list[Key] = [
        (
            kind,
            id_ + 1,
            TimeRange(start=start_time, end=end_time),
            "'_trace4_' in name" if kind == "span" else None,
            session_filter_condition,
            eval_name,
        )
        for kind in kinds
        for id_ in range(10)
        for eval_name in (("B", "D") if kind == "trace" else ("A", "C"))
    ]

    summaries = [summary for summary in await AnnotationSummaryDataLoader(db)._load_fn(keys)]
    actual = []
    for summary in summaries:
        assert summary is not None
        actual.append(
            summary.mean_score(),  # type: ignore[call-arg]
        )
    assert actual == pytest.approx(expected, 1e-7)


# --- tests/unit/server/api/dataloaders/test_latency_ms_quantiles.py ---

async def test_latency_ms_quantiles_p25_p50_p75(
    db: DbSessionFactory,
    data_for_testing_dataloaders: None,
) -> None:
    start_time = datetime.fromisoformat("2021-01-01T00:00:10.000+00:00")
    end_time = datetime.fromisoformat("2021-01-01T00:10:00.000+00:00")
    pid = models.Trace.project_rowid
    async with db() as session:
        span_df = await session.run_sync(
            lambda s: pd.read_sql_query(
                select(pid, models.Span.latency_ms.label("latency_ms"))
                .join_from(models.Trace, models.Span)
                .where(models.Span.name.contains("_trace4_"))
                .where(start_time <= models.Span.start_time)
                .where(models.Span.start_time < end_time),
                s.connection(),
            )
        )
        trace_df = await session.run_sync(
            lambda s: pd.read_sql_query(
                select(pid, models.Trace.latency_ms.label("latency_ms"))
                .where(start_time <= models.Trace.start_time)
                .where(models.Trace.start_time < end_time),
                s.connection(),
            )
        )
    expected = (
        trace_df.groupby("project_rowid")["latency_ms"]
        .quantile(np.array([0.25, 0.50, 0.75]))
        .sort_index()
        .to_list()
        + span_df.groupby("project_rowid")["latency_ms"]
        .quantile(np.array([0.25, 0.50, 0.75]))
        .sort_index()
        .to_list()
    )
    kinds: list[Literal["span", "trace"]] = ["trace", "span"]
    session_filter_condition = None
    keys: list[Key] = [
        (
            kind,
            id_ + 1,
            TimeRange(start=start_time, end=end_time),
            "'_trace4_' in name" if kind == "span" else None,
            session_filter_condition,
            probability,
        )
        for kind in kinds
        for id_ in range(10)
        for probability in (0.25, 0.50, 0.75)
    ]
    actual = await LatencyMsQuantileDataLoader(db)._load_fn(keys)
    assert actual == pytest.approx(expected, 1e-7)


# --- tests/unit/server/api/dataloaders/test_record_counts.py ---

async def test_record_counts(
    db: DbSessionFactory,
    data_for_testing_dataloaders: None,
) -> None:
    start_time = datetime.fromisoformat("2021-01-01T00:00:10.000+00:00")
    end_time = datetime.fromisoformat("2021-01-01T00:10:00.000+00:00")
    pid = models.Trace.project_rowid
    async with db() as session:
        span_df = await session.run_sync(
            lambda s: pd.read_sql_query(
                select(pid, func.count().label("count"))
                .join_from(models.Trace, models.Span)
                .group_by(pid)
                .order_by(pid)
                .where(models.Span.name.contains("_trace4_"))
                .where(start_time <= models.Span.start_time)
                .where(models.Span.start_time < end_time),
                s.connection(),
            )
        )
        trace_df = await session.run_sync(
            lambda s: pd.read_sql_query(
                select(pid, func.count().label("count"))
                .group_by(pid)
                .order_by(pid)
                .where(start_time <= models.Trace.start_time)
                .where(models.Trace.start_time < end_time),
                s.connection(),
            )
        )
    expected = trace_df.loc[:, "count"].to_list() + span_df.loc[:, "count"].to_list()
    kinds: list[Literal["span", "trace"]] = ["trace", "span"]
    session_filter_condition = None
    keys: list[Key] = [
        (
            kind,
            id_ + 1,
            TimeRange(start=start_time, end=end_time),
            "'_trace4_' in name" if kind == "span" else None,
            session_filter_condition,
        )
        for kind in kinds
        for id_ in range(10)
    ]

    actual = await RecordCountDataLoader(db)._load_fn(keys)
    assert actual == expected


# --- tests/unit/server/api/dataloaders/test_session_trace_latency_ms_quantile.py ---

async def test_session_trace_latency_ms_quantile(
    db: DbSessionFactory,
    data_for_testing_dataloaders: None,
) -> None:
    psid = models.Trace.project_session_rowid
    async with db() as session:
        trace_df = await session.run_sync(
            lambda s: pd.read_sql_query(
                select(psid, models.Trace.latency_ms.label("latency_ms")),
                s.connection(),
            )
        )
    expected = (
        trace_df.groupby("project_session_rowid")["latency_ms"]
        .quantile(np.array([0.25, 0.50, 0.75]))
        .sort_index()
        .to_list()
    )
    keys: list[Key] = [
        (
            id_ + 1,
            probability,
        )
        for id_ in range(20)
        for probability in (0.25, 0.50, 0.75)
    ]
    actual = await SessionTraceLatencyMsQuantileDataLoader(db)._load_fn(keys)
    assert actual == expected


# --- tests/unit/server/api/dataloaders/test_token_counts.py ---

async def test_token_counts(
    db: DbSessionFactory,
    data_for_testing_dataloaders: None,
) -> None:
    start_time = datetime.fromisoformat("2021-01-01T00:00:10.000+00:00")
    end_time = datetime.fromisoformat("2021-01-01T00:10:00.000+00:00")
    async with db() as session:
        prompt = models.Span.attributes[["llm", "token_count", "prompt"]].as_float()
        completion = models.Span.attributes[["llm", "token_count", "completion"]].as_float()
        pid = models.Trace.project_rowid
        span_df = await session.run_sync(
            lambda s: pd.read_sql_query(
                select(
                    pid,
                    func.sum(prompt).label("prompt"),
                    func.sum(completion).label("completion"),
                )
                .join(models.Span)
                .group_by(pid)
                .order_by(pid)
                .where(models.Span.name.contains("_trace4_"))
                .where(start_time <= models.Span.start_time)
                .where(models.Span.start_time < end_time),
                s.connection(),
            )
        )
    expected = (
        span_df.loc[:, "prompt"].to_list()
        + span_df.loc[:, "completion"].to_list()
        + (span_df.loc[:, "prompt"] + span_df.loc[:, "completion"]).to_list()
    )
    kinds: list[Literal["prompt", "completion", "total"]] = ["prompt", "completion", "total"]
    keys: list[Key] = [
        (
            kind,
            id_ + 1,
            TimeRange(start=start_time, end=end_time),
            "'_trace4_' in name",
        )
        for kind in kinds
        for id_ in range(10)
    ]
    actual = await TokenCountDataLoader(db)._load_fn(keys)
    assert actual == expected


# --- tests/unit/server/api/helpers/test_dataset_helpers.py ---

def test_get_message(message: dict[str, Any], expected: dict[str, Any]) -> None:
    assert _get_message(message) == expected

def test_merge_assistant_output_items(
    raw: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> None:
    assert _merge_assistant_output_items(raw) == expected  # type: ignore[arg-type]


# --- tests/unit/server/api/helpers/test_evaluators.py ---

    async def test_evaluate_returns_correct_result(
        self,
        db: DbSessionFactory,
        project: models.Project,
        tracer: Tracer,
        llm_evaluator: LLMEvaluator,
        output_config: CategoricalOutputConfig,
        input_mapping: EvaluatorInputMappingInput,
        custom_vcr: CustomVCR,
        gpt_4o_mini_generative_model: models.GenerativeModel,
    ) -> None:
        with custom_vcr.use_cassette():
            evaluation_result = (
                await llm_evaluator.evaluate(
                    context={"input": "What is 2 + 2?", "output": "4"},
                    input_mapping=input_mapping.to_orm(),
                    name="correctness",
                    output_configs=[output_config],
                    tracer=tracer,
                )
            )[0]

        result = dict(evaluation_result)
        assert result.pop("error") is None
        assert result.pop("label") == "correct"
        assert result.pop("score") == 1.0
        assert result.pop("explanation") is not None
        assert result.pop("annotator_kind") == "LLM"
        assert result.pop("name") == "correctness"
        trace_id = result.pop("trace_id")
        assert isinstance(trace_id, str)
        assert isinstance(result.pop("start_time"), datetime)
        assert isinstance(result.pop("end_time"), datetime)
        assert result.pop("metadata") == {}
        result.pop("error_exc", None)
        assert not result

        async with db() as session:
            db_traces = tracer.get_db_traces(project_id=project.id)
            session.add_all(db_traces)
            await session.flush()

        assert len(db_traces) == 1
        db_trace = db_traces[0]
        assert db_trace.trace_id == trace_id
        db_spans = db_trace.spans
        span_costs = db_trace.span_costs
        assert len(db_spans) == 5

        evaluator_span = None
        input_mapping_span = None
        prompt_span = None
        llm_span = None
        parse_eval_result_span = None
        for span in db_spans:
            if span.span_kind == "EVALUATOR":
                evaluator_span = span
            elif span.span_kind == "LLM":
                llm_span = span
            elif span.span_kind == "PROMPT":
                prompt_span = span
            elif span.span_kind == "CHAIN":
                if span.name == "Input Mapping":
                    input_mapping_span = span
                elif span.name == "Parse Eval Result":
                    parse_eval_result_span = span

        assert evaluator_span is not None
        assert input_mapping_span is not None
        assert prompt_span is not None
        assert llm_span is not None
        assert parse_eval_result_span is not None
        assert evaluator_span.parent_id is None
        assert input_mapping_span.parent_id == evaluator_span.span_id
        assert prompt_span.parent_id == evaluator_span.span_id
        assert llm_span.parent_id == evaluator_span.span_id
        assert parse_eval_result_span.parent_id == evaluator_span.span_id

        # evaluator span
        assert evaluator_span.name == "Evaluator: correctness"
        assert evaluator_span.status_code == "OK"
        assert not evaluator_span.events
        attributes = dict(flatten(evaluator_span.attributes, recurse_on_sequence=True))
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "EVALUATOR"
        raw_input_value = attributes.pop(INPUT_VALUE)
        assert raw_input_value is not None
        input_value = json.loads(raw_input_value)
        assert set(input_value.keys()) == {"input", "output"}
        assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
        raw_output_value = attributes.pop(OUTPUT_VALUE)
        assert raw_output_value is not None
        output_value = json.loads(raw_output_value)
        assert set(output_value.keys()) == {"results"}
        assert len(output_value["results"]) == 1
        assert output_value["results"][0]["name"] == "correctness"
        assert output_value["results"][0]["label"] == "correct"
        assert output_value["results"][0]["score"] == 1.0
        assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
        assert not attributes

        # Input Mapping span
        assert input_mapping_span.name == "Input Mapping"
        assert input_mapping_span.status_code == "OK"
        assert not input_mapping_span.events
        attributes = dict(flatten(input_mapping_span.attributes, recurse_on_sequence=True))
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "CHAIN"
        raw_input_value = attributes.pop(INPUT_VALUE)
        input_value = json.loads(raw_input_value)
        assert input_value == {
            "input_mapping": {"path_mapping": {}, "literal_mapping": {}},
            "template_variables": {"input": "What is 2 + 2?", "output": "4"},
        }
        assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
        raw_output_value = attributes.pop(OUTPUT_VALUE)
        output_value = json.loads(raw_output_value)
        assert output_value == {"input": "What is 2 + 2?", "output": "4"}
        assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
        assert not attributes

        # Prompt span
        assert prompt_span.name == "Prompt: test-prompt"
        assert prompt_span.status_code == "OK"
        assert not prompt_span.events
        attributes = dict(flatten(prompt_span.attributes, recurse_on_sequence=True))
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "PROMPT"
        raw_input_value = attributes.pop(INPUT_VALUE)
        input_value = json.loads(raw_input_value)
        assert input_value == {"input": "What is 2 + 2?", "output": "4"}
        assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
        raw_output_value = attributes.pop(OUTPUT_VALUE)
        output_value = json.loads(raw_output_value)
        assert output_value == {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an evaluator. Assess whether the output correctly answers the input question.",
                },
                {
                    "role": "user",
                    "content": "Input: What is 2 + 2?\n\nOutput: 4\n\nIs this output correct?",
                },
            ]
        }
        assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
        assert not attributes

        # llm span
        assert llm_span.name == "ChatCompletion"
        assert llm_span.status_code == "OK"
        assert not llm_span.events
        assert llm_span.llm_token_count_prompt is not None
        assert llm_span.llm_token_count_prompt > 0
        assert llm_span.llm_token_count_completion is not None
        assert llm_span.llm_token_count_completion > 0
        assert llm_span.cumulative_llm_token_count_prompt > 0
        assert llm_span.cumulative_llm_token_count_completion > 0
        attributes = dict(flatten(llm_span.attributes, recurse_on_sequence=True))
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4o-mini"
        assert attributes.pop(LLM_PROVIDER) == "openai"
        assert attributes.pop(LLM_SYSTEM) == "openai"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
        assert "evaluator" in attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}").lower()
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
        user_message = attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}")
        assert "What is 2 + 2?" in user_message
        assert "4" in user_message
        # Check token count attributes exist and are integers
        token_count_attribute_keys = [
            key for key in attributes if key.startswith("llm.token_count.")
        ]
        for key in token_count_attribute_keys:
            assert isinstance(attributes.pop(key), int)
        assert attributes.pop(URL_FULL) == "https://api.openai.com/v1/chat/completions"
        assert attributes.pop(URL_PATH) == "chat/completions"
        assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
        raw_output_value = attributes.pop(OUTPUT_VALUE)
        output_value = json.loads(raw_output_value)
        assert output_value == {
            "id": "chatcmpl-DQeuwqU8iYEn0jmW2dXui8MlYo5sn",
            "object": "chat.completion",
            "created": 1775246186,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_yUjsmpi4tVtBJrk9lDCaeHXm",
                                "type": "function",
                                "function": {
                                    "name": "correctness",
                                    "arguments": (
                                        '{"label":"correct","explanation":"The output correctly '
                                        'states that 2 + 2 equals 4."}'
                                    ),
                                },
                            }
                        ],
                        "annotations": [],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 98,
                "completion_tokens": 32,
                "total_tokens": 130,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "audio_tokens": 0,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
            "service_tier": "default",
            "system_fingerprint": "fp_ebf4e532f9",
        }
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert isinstance(
            attributes.pop(
                f"{LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_ID}"
            ),
            str,
        )
        assert (
            attributes.pop(
                f"{LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}"
            )
            == "correctness"
        )
        raw_arguments = attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS}"
        )
        assert isinstance(raw_arguments, str)
        arguments = json.loads(raw_arguments)
        assert arguments.pop("label") == "correct"
        assert isinstance(arguments.pop("explanation"), str)
        assert not arguments
        assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
        input_value = json.loads(attributes.pop(INPUT_VALUE))
        assert input_value.pop("messages") == [
            {
                "role": "system",
                "content": "You are an evaluator. Assess whether the output correctly answers the input question.",
            },
            {
                "role": "user",
                "content": "Input: What is 2 + 2?\n\nOutput: 4\n\nIs this output correct?",
            },
        ]
        assert isinstance(invocation_parameters := attributes.pop("llm.invocation_parameters"), str)
        assert json.loads(invocation_parameters) == {
            "temperature": 0.0,
            "tool_choice": "required",
        }
        expected_tool = {
            "type": "function",
            "function": {
                "name": "correctness",
                "description": "Evaluates the correctness of the output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["correct", "incorrect"],
                            "description": "correctness",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief explanation for the label",
                        },
                    },
                    "required": ["label", "explanation"],
                },
                "strict": None,
            },
        }
        assert isinstance(tool_schtemas := attributes.pop("llm.tools.0.tool.json_schema"), str)
        assert json.loads(tool_schtemas) == expected_tool
        assert not attributes

        # Parse Eval Result span
        assert parse_eval_result_span.name == "Parse Eval Result"
        assert parse_eval_result_span.status_code == "OK"
        assert not parse_eval_result_span.events
        attributes = dict(flatten(parse_eval_result_span.attributes, recurse_on_sequence=True))
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "CHAIN"
        raw_input_value = attributes.pop(INPUT_VALUE)
        assert raw_input_value is not None
        input_value = json.loads(raw_input_value)
        tool_calls = input_value.pop("tool_calls")
        assert len(tool_calls) == 1
        tool_call = next(
            iter(tool_calls.values())
        )  # the key is a random tool call ID from the LLM response
        assert tool_call == {
            "name": "correctness",
            "arguments": '{"label":"correct","explanation":"The output correctly states that 2 + 2 equals 4."}',
        }
        assert input_value == {
            "output_configs": {
                "correctness": {
                    "values": [
                        {"label": "correct", "score": 1.0},
                        {"label": "incorrect", "score": 0.0},
                    ],
                }
            },
        }
        assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
        assert json.loads(attributes.pop(OUTPUT_VALUE)) == {
            "results": [
                {
                    "name": "correctness",
                    "label": "correct",
                    "score": 1.0,
                    "explanation": "The output correctly states that 2 + 2 equals 4.",
                }
            ],
        }
        assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
        assert not attributes

        # Verify span costs for LLM span
        assert len(span_costs) == 1
        span_cost = span_costs[0]

        assert span_cost.span_rowid == llm_span.id
        assert span_cost.trace_rowid == llm_span.trace_rowid
        assert span_cost.model_id == gpt_4o_mini_generative_model.id
        assert span_cost.span_start_time == llm_span.start_time
        prompt_token_prices = next(
            p for p in gpt_4o_mini_generative_model.token_prices if p.is_prompt
        )
        completion_token_prices = next(
            p for p in gpt_4o_mini_generative_model.token_prices if not p.is_prompt
        )
        prompt_base_rate = prompt_token_prices.base_rate
        completion_base_rate = completion_token_prices.base_rate
        expected_prompt_cost = llm_span.llm_token_count_prompt * prompt_base_rate
        expected_completion_cost = llm_span.llm_token_count_completion * completion_base_rate
        expected_total_cost = expected_prompt_cost + expected_completion_cost
        assert span_cost.total_cost == pytest.approx(expected_total_cost)
        assert span_cost.total_tokens == llm_span.llm_token_count_total
        assert span_cost.prompt_tokens == llm_span.llm_token_count_prompt
        assert span_cost.prompt_cost == pytest.approx(expected_prompt_cost)
        assert span_cost.completion_tokens == llm_span.llm_token_count_completion
        assert span_cost.completion_cost == pytest.approx(expected_completion_cost)

        # Verify span cost details
        assert len(span_cost.span_cost_details) >= 2
        input_detail = next(
            d for d in span_cost.span_cost_details if d.is_prompt and d.token_type == "input"
        )
        output_detail = next(
            d for d in span_cost.span_cost_details if not d.is_prompt and d.token_type == "output"
        )

        assert input_detail.span_cost_id == span_cost.id
        assert input_detail.token_type == "input"
        assert input_detail.is_prompt is True
        assert input_detail.tokens == llm_span.llm_token_count_prompt
        assert input_detail.cost == pytest.approx(expected_prompt_cost)
        assert input_detail.cost_per_token == prompt_base_rate

        assert output_detail.span_cost_id == span_cost.id
        assert output_detail.token_type == "output"
        assert output_detail.is_prompt is False
        assert output_detail.tokens == llm_span.llm_token_count_completion
        assert output_detail.cost == pytest.approx(expected_completion_cost)
        assert output_detail.cost_per_token == completion_base_rate


# --- tests/unit/server/api/helpers/test_experiment_run_filters.py ---

def test_compile_sqlalchemy_filter_condition_raises_appropriate_error_message(
    filter_condition: str,
    expected_error_prefix: str,
) -> None:
    with pytest.raises(ExperimentRunFilterConditionSyntaxError) as exc_info:
        compile_sqlalchemy_filter_condition(
            filter_condition=filter_condition,
            experiment_ids=[0, 1, 2],
        )

    error = exc_info.value
    assert str(error).startswith(expected_error_prefix)


# --- tests/unit/server/api/helpers/test_message_helpers.py ---

    def test_role_conversion(
        self, openai_message: dict[str, Any], expected: PlaygroundMessage
    ) -> None:
        result = convert_openai_message_to_internal(openai_message)
        assert result == expected

    def test_content_handling(
        self, openai_message: dict[str, Any], expected: PlaygroundMessage
    ) -> None:
        result = convert_openai_message_to_internal(openai_message)
        assert result == expected

    def test_tool_message_with_tool_call_id(self) -> None:
        openai_message = {
            "role": "tool",
            "content": '{"temperature": 72}',
            "tool_call_id": "call_abc123",
        }
        result = convert_openai_message_to_internal(openai_message)
        assert result == create_playground_message(
            ChatCompletionMessageRole.TOOL,
            '{"temperature": 72}',
            tool_call_id="call_abc123",
        )

    def test_assistant_message_with_tool_calls(self) -> None:
        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco"}',
                },
            }
        ]
        openai_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
        result = convert_openai_message_to_internal(openai_message)
        assert result["role"] == ChatCompletionMessageRole.AI
        assert result["content"] == ""
        assert result.get("tool_call_id") is None
        # Tool calls are passed through directly
        assert result.get("tool_calls") == tool_calls

    def test_assistant_message_with_multiple_tool_calls(self) -> None:
        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
            },
            {
                "id": "call_def456",
                "type": "function",
                "function": {"name": "get_time", "arguments": '{"timezone": "PST"}'},
            },
        ]
        openai_message = {
            "role": "assistant",
            "content": "Let me check both for you.",
            "tool_calls": tool_calls,
        }
        result = convert_openai_message_to_internal(openai_message)
        assert result["role"] == ChatCompletionMessageRole.AI
        assert result["content"] == "Let me check both for you."
        # Tool calls are passed through directly
        assert result.get("tool_calls") == tool_calls

    def test_simple_conversation(self) -> None:
        data = {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        result = extract_and_convert_example_messages(data, "messages")
        assert len(result) == 2
        assert result[0] == create_playground_message(ChatCompletionMessageRole.USER, "Hello!")
        assert result[1] == create_playground_message(ChatCompletionMessageRole.AI, "Hi there!")

    def test_nested_messages_path(self) -> None:
        data = {
            "input": {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "What is 2+2?"},
                ]
            }
        }
        result = extract_and_convert_example_messages(data, "input.messages")
        assert len(result) == 2
        assert result[0] == create_playground_message(
            ChatCompletionMessageRole.SYSTEM, "You are helpful."
        )
        assert result[1] == create_playground_message(
            ChatCompletionMessageRole.USER, "What is 2+2?"
        )

    def test_openai_fine_tuning_format(self) -> None:
        """Test the OpenAI fine-tuning format as described in the feature spec."""
        data = {
            "messages": [
                {"role": "user", "content": "What is the weather in San Francisco?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_id",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location": "San Francisco, USA", "format": "celsius"}',
                            },
                        }
                    ],
                },
            ],
            "parallel_tool_calls": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather",
                    },
                }
            ],
        }
        result = extract_and_convert_example_messages(data, "messages")
        assert len(result) == 2
        assert result[0] == create_playground_message(
            ChatCompletionMessageRole.USER,
            "What is the weather in San Francisco?",
        )
        assert result[1]["role"] == ChatCompletionMessageRole.AI
        assert result[1]["content"] == ""  # No content, just tool calls
        assert result[1].get("tool_calls") is not None
        assert len(result[1]["tool_calls"]) == 1

    def test_tool_response_in_conversation(self) -> None:
        data = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"temp": 72}',
                    "tool_call_id": "call_123",
                },
                {"role": "assistant", "content": "The temperature is 72°F."},
            ]
        }
        result = extract_and_convert_example_messages(data, "messages")
        assert len(result) == 4
        # Check the tool response message
        assert result[2] == create_playground_message(
            ChatCompletionMessageRole.TOOL,
            '{"temp": 72}',
            tool_call_id="call_123",
        )

    def test_complex_multi_turn_conversation(self) -> None:
        """Test a realistic multi-turn conversation with various message types."""
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Can you help me with the weather?"},
                {"role": "assistant", "content": "Of course! Which city would you like?"},
                {"role": "user", "content": "San Francisco"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "San Francisco"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather_1",
                    "content": '{"temperature": 65, "condition": "sunny"}',
                },
                {
                    "role": "assistant",
                    "content": "It's 65°F and sunny in San Francisco!",
                },
            ]
        }
        result = extract_and_convert_example_messages(data, "messages")
        assert len(result) == 7

        # Verify each message type is correctly converted
        assert result[0]["role"] == ChatCompletionMessageRole.SYSTEM
        assert result[1]["role"] == ChatCompletionMessageRole.USER
        assert result[2]["role"] == ChatCompletionMessageRole.AI
        assert result[3]["role"] == ChatCompletionMessageRole.USER
        assert result[4]["role"] == ChatCompletionMessageRole.AI
        assert result[4].get("tool_calls") is not None  # Has tool calls
        assert result[5]["role"] == ChatCompletionMessageRole.TOOL
        assert result[5].get("tool_call_id") == "call_weather_1"
        assert result[6]["role"] == ChatCompletionMessageRole.AI
        assert "65°F" in result[6]["content"]


# --- tests/unit/server/api/helpers/test_playground_clients.py ---

    async def test_text_response_records_expected_attributes(
        self,
        openai_client_factory: Any,
        custom_vcr: CustomVCR,
        tracer: Tracer,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        client = OpenAIBaseStreamingClient(
            client_factory=openai_client_factory,
            model_name="gpt-4o-mini",
            provider="openai",
        )

        messages: list[PlaygroundMessage] = [
            create_playground_message(
                ChatCompletionMessageRole.USER,
                "Who won the World Cup in 2018? Answer in one word",
            )
        ]

        invocation_parameters: Mapping[str, Any] = {"temperature": 0.1}

        with custom_vcr.use_cassette():
            text_chunks = []
            async for chunk in client.chat_completion_create(
                messages=messages,
                tools=None,
                invocation_parameters=invocation_parameters,
                tracer=tracer,
            ):
                if isinstance(chunk, TextChunk):
                    text_chunks.append(chunk.content)

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span: ReadableSpan = spans[0]

        assert span.name == "ChatCompletion"
        assert span.status.is_ok
        assert not span.events

        assert span.attributes is not None
        attributes = dict(span.attributes)

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4o-mini"

        invocation_params = attributes.pop(LLM_INVOCATION_PARAMETERS)
        assert isinstance(invocation_params, str)
        assert json.loads(invocation_params) == {
            "temperature": 0.1,
        }

        input_messages_role = attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}")
        assert input_messages_role == "user"
        input_messages_content = attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        assert input_messages_content == "Who won the World Cup in 2018? Answer in one word"

        output_messages_role = attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}")
        assert output_messages_role == "assistant"
        output_messages_content = attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        response_text = "".join(text_chunks)
        assert output_messages_content == response_text
        assert "france" in response_text.lower()

        token_count_total = attributes.pop(LLM_TOKEN_COUNT_TOTAL)
        assert isinstance(token_count_total, int)
        assert token_count_total > 0

        token_count_prompt = attributes.pop(LLM_TOKEN_COUNT_PROMPT)
        assert isinstance(token_count_prompt, int)
        assert token_count_prompt > 0

        token_count_completion = attributes.pop(LLM_TOKEN_COUNT_COMPLETION)
        assert isinstance(token_count_completion, int)
        assert token_count_completion > 0

        assert token_count_total == token_count_prompt + token_count_completion

        cache_read = attributes.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ)
        assert cache_read == 0

        reasoning_tokens = attributes.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING)
        assert reasoning_tokens == 0

        audio_prompt_tokens = attributes.pop("llm.token_count.prompt_details.audio")
        assert audio_prompt_tokens == 0

        audio_completion_tokens = attributes.pop("llm.token_count.completion_details.audio")
        assert audio_completion_tokens == 0

        url_full = attributes.pop("url.full")
        assert url_full == "https://api.openai.com/v1/chat/completions"

        url_path = attributes.pop("url.path")
        assert url_path == "chat/completions"

        llm_provider = attributes.pop(LLM_PROVIDER)
        assert llm_provider == "openai"

        llm_system = attributes.pop(LLM_SYSTEM)
        assert llm_system == "openai"

        assert attributes.pop(INPUT_VALUE)
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE)
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT

        assert not attributes

    async def test_tool_call_response_records_expected_attributes(
        self,
        openai_client_factory: Any,
        custom_vcr: CustomVCR,
        tracer: Tracer,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        client = OpenAIBaseStreamingClient(
            client_factory=openai_client_factory,
            model_name="gpt-4o-mini",
            provider="openai",
        )

        get_current_weather_tools = PromptTools(
            type="tools",
            tool_choice=PromptToolChoiceZeroOrMore(type="zero_or_more"),
            tools=[
                PromptToolFunction(
                    type="function",
                    function=PromptToolFunctionDefinition(
                        name="get_current_weather",
                        description="Get the current weather in a given location",
                        parameters={
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city name, e.g. San Francisco",
                                },
                            },
                            "required": ["location"],
                        },
                    ),
                )
            ],
        )

        messages: list[PlaygroundMessage] = [
            create_playground_message(
                ChatCompletionMessageRole.USER,
                "How's the weather in San Francisco?",
            )
        ]

        invocation_parameters: Mapping[str, Any] = {}

        with custom_vcr.use_cassette():
            tool_call_chunks = []
            async for chunk in client.chat_completion_create(
                messages=messages,
                tools=get_current_weather_tools,
                tracer=tracer,
                invocation_parameters=invocation_parameters,
            ):
                tool_call_chunks.append(chunk)

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span: ReadableSpan = spans[0]

        assert span.name == "ChatCompletion"
        assert span.status.is_ok
        assert not span.events

        assert span.attributes is not None
        attributes = dict(span.attributes)

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4o-mini"

        invocation_params = attributes.pop(LLM_INVOCATION_PARAMETERS)
        assert isinstance(invocation_params, str)
        assert json.loads(invocation_params) == {
            "tool_choice": "auto",
        }

        input_messages_role = attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}")
        assert input_messages_role == "user"
        input_messages_content = attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        assert input_messages_content == "How's the weather in San Francisco?"

        output_messages_role = attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}")
        assert output_messages_role == "assistant"

        tool_call_id = attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_ID}"
        )
        assert isinstance(tool_call_id, str)

        tool_call_function_name = attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}"
        )
        assert tool_call_function_name == "get_current_weather"

        tool_call_function_arguments = attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        )
        assert isinstance(tool_call_function_arguments, str)
        arguments = json.loads(tool_call_function_arguments)
        assert arguments == {"location": "San Francisco"}

        token_count_total = attributes.pop(LLM_TOKEN_COUNT_TOTAL)
        assert isinstance(token_count_total, int)
        assert token_count_total > 0

        token_count_prompt = attributes.pop(LLM_TOKEN_COUNT_PROMPT)
        assert isinstance(token_count_prompt, int)
        assert token_count_prompt > 0

        token_count_completion = attributes.pop(LLM_TOKEN_COUNT_COMPLETION)
        assert isinstance(token_count_completion, int)
        assert token_count_completion > 0

        assert token_count_total == token_count_prompt + token_count_completion

        cache_read = attributes.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ)
        assert cache_read == 0

        reasoning_tokens = attributes.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING)
        assert reasoning_tokens == 0

        audio_prompt_tokens = attributes.pop("llm.token_count.prompt_details.audio")
        assert audio_prompt_tokens == 0

        audio_completion_tokens = attributes.pop("llm.token_count.completion_details.audio")
        assert audio_completion_tokens == 0

        url_full = attributes.pop("url.full")
        assert url_full == "https://api.openai.com/v1/chat/completions"

        url_path = attributes.pop("url.path")
        assert url_path == "chat/completions"

        llm_provider = attributes.pop(LLM_PROVIDER)
        assert llm_provider == "openai"

        llm_system = attributes.pop(LLM_SYSTEM)
        assert llm_system == "openai"

        assert isinstance(
            llm_tool_schema := attributes.pop(f"{LLM_TOOLS}.0.{TOOL_JSON_SCHEMA}"), str
        )
        assert json.loads(llm_tool_schema) == {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        }
                    },
                    "required": ["location"],
                },
                "strict": None,
                "description": "Get the current weather in a given location",
            },
        }

        assert attributes.pop(INPUT_VALUE)
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE)
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

        assert not attributes

    async def test_authentication_error_records_error_status_on_span(
        self,
        openai_client_factory: Any,
        custom_vcr: CustomVCR,
        tracer: Tracer,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        client = OpenAIBaseStreamingClient(
            client_factory=openai_client_factory,
            model_name="gpt-4o-mini",
            provider="openai",
        )

        messages: list[PlaygroundMessage] = [
            create_playground_message(
                ChatCompletionMessageRole.USER,
                "Say hello",
            )
        ]

        invocation_parameters: Mapping[str, Any] = {"temperature": 0.1}

        with custom_vcr.use_cassette():
            with pytest.raises(AuthenticationError) as exc_info:
                async for _ in client.chat_completion_create(
                    messages=messages,
                    tools=None,
                    tracer=tracer,
                    invocation_parameters=invocation_parameters,
                ):
                    pass

        assert exc_info.value.status_code == 401

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "ChatCompletion"
        assert span.status.status_code is StatusCode.ERROR
        status_description = span.status.description
        assert status_description is not None
        assert isinstance(status_description, str)
        assert status_description.startswith("Error code: 401")
        assert "invalid_api_key" in status_description

        events = span.events
        assert len(events) == 1
        event = events[0]
        assert event.name == "exception"
        assert event.attributes is not None
        event_attrs = dict(event.attributes)
        assert event_attrs.pop("exception.type") == "openai.AuthenticationError"
        exception_message = event_attrs.pop("exception.message")
        assert isinstance(exception_message, str)
        assert exception_message.startswith("Error code: 401")
        assert event_attrs.pop("exception.escaped") == "False"
        exception_stacktrace = event_attrs.pop("exception.stacktrace")
        assert isinstance(exception_stacktrace, str)
        assert "AuthenticationError" in exception_stacktrace
        assert not event_attrs

        assert span.attributes is not None
        attributes = dict(span.attributes)

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4o-mini"

        invocation_params = attributes.pop(LLM_INVOCATION_PARAMETERS)
        assert isinstance(invocation_params, str)
        assert json.loads(invocation_params) == {
            "temperature": 0.1,
        }

        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "Say hello"

        assert attributes.pop(LLM_PROVIDER) == "openai"
        assert attributes.pop(LLM_SYSTEM) == "openai"

        url_full = attributes.pop("url.full")
        assert url_full == "https://api.openai.com/v1/chat/completions"

        url_path = attributes.pop("url.path")
        assert url_path == "chat/completions"

        input_value = attributes.pop(INPUT_VALUE)
        assert isinstance(input_value, str)
        input_data = json.loads(input_value)
        assert input_data == {
            "messages": [{"role": "user", "content": "Say hello"}],
            "model": "gpt-4o-mini",
            "temperature": 0.1,
        }
        assert attributes.pop(INPUT_MIME_TYPE) == JSON

        assert not attributes

    def test_openai_chat_completions_returns_streaming_client(self) -> None:
        """Standard models with CHAT_COMPLETIONS should return OpenAIStreamingClient."""
        client_class = get_openai_client_class(
            GenerativeProviderKey.OPENAI,
            "gpt-4o",
            OpenAIApiType.CHAT_COMPLETIONS,
        )
        assert client_class is OpenAIStreamingClient

    def test_openai_chat_completions_custom_model_returns_streaming_client(self) -> None:
        """Custom/unknown models with CHAT_COMPLETIONS should return OpenAIStreamingClient."""
        client_class = get_openai_client_class(
            GenerativeProviderKey.OPENAI,
            "my-custom-fine-tuned-model",
            OpenAIApiType.CHAT_COMPLETIONS,
        )
        assert client_class is OpenAIStreamingClient

    def test_openai_chat_completions_reasoning_model_returns_reasoning_client(self) -> None:
        """Reasoning models (o1, o3) with CHAT_COMPLETIONS should return reasoning client."""
        for model_name in ["o1", "o3", "o3-mini"]:
            client_class = get_openai_client_class(
                GenerativeProviderKey.OPENAI,
                model_name,
                OpenAIApiType.CHAT_COMPLETIONS,
            )
            assert client_class is OpenAIReasoningNonStreamingClient, f"Failed for {model_name}"

    def test_azure_chat_completions_returns_azure_streaming_client(self) -> None:
        """Azure with CHAT_COMPLETIONS should return AzureOpenAIStreamingClient."""
        client_class = get_openai_client_class(
            GenerativeProviderKey.AZURE_OPENAI,
            "gpt-4o",
            OpenAIApiType.CHAT_COMPLETIONS,
        )
        assert client_class is AzureOpenAIStreamingClient

    def test_azure_chat_completions_custom_model_returns_azure_streaming_client(self) -> None:
        """Azure custom models with CHAT_COMPLETIONS should return AzureOpenAIStreamingClient."""
        client_class = get_openai_client_class(
            GenerativeProviderKey.AZURE_OPENAI,
            "my-azure-deployment",
            OpenAIApiType.CHAT_COMPLETIONS,
        )
        assert client_class is AzureOpenAIStreamingClient

    def test_azure_chat_completions_reasoning_model_returns_reasoning_client(self) -> None:
        """Azure reasoning models with CHAT_COMPLETIONS should return reasoning client."""
        client_class = get_openai_client_class(
            GenerativeProviderKey.AZURE_OPENAI,
            "o1",
            OpenAIApiType.CHAT_COMPLETIONS,
        )
        assert client_class is AzureOpenAIReasoningNonStreamingClient

    def test_anthropic_returns_none(self) -> None:
        """Non-OpenAI providers should return None (caller uses registry)."""
        client_class = get_openai_client_class(
            GenerativeProviderKey.ANTHROPIC,
            "claude-3-opus",
            OpenAIApiType.CHAT_COMPLETIONS,
        )
        assert client_class is None

    def test_google_returns_none(self) -> None:
        """Google provider should return None."""
        client_class = get_openai_client_class(
            GenerativeProviderKey.GOOGLE,
            "gemini-pro",
            OpenAIApiType.CHAT_COMPLETIONS,
        )
        assert client_class is None

    def test_chat_completions_has_temperature_parameter(self) -> None:
        """CHAT_COMPLETIONS client should have temperature parameter."""
        client_class = get_openai_client_class(
            GenerativeProviderKey.OPENAI,
            "my-custom-model",
            OpenAIApiType.CHAT_COMPLETIONS,
        )
        assert client_class is not None
        params = client_class.supported_invocation_parameters()
        param_names = [p.invocation_name for p in params]
        assert "temperature" in param_names
        assert "top_p" in param_names
        assert "frequency_penalty" in param_names
        assert "reasoning_effort" not in param_names

    def test_reasoning_model_has_reasoning_effort_parameter(self) -> None:
        """Reasoning models should have reasoning_effort parameter."""
        client_class = get_openai_client_class(
            GenerativeProviderKey.OPENAI,
            "o1",
            OpenAIApiType.CHAT_COMPLETIONS,
        )
        assert client_class is not None
        params = client_class.supported_invocation_parameters()
        param_names = [p.invocation_name for p in params]
        assert "reasoning_effort" in param_names
        assert "temperature" not in param_names


# --- tests/unit/server/api/mutations/test_model_mutations.py ---

    async def test_create_model_with_invalid_input_raises_expected_error(
        self,
        gql_client: AsyncGraphQLClient,
        variables: dict[str, Any],
        expected_error_message: str,
        custom_model: models.GenerativeModel,
    ) -> None:
        result = await gql_client.execute(
            query=self.QUERY,
            variables=variables,
            operation_name="CreateModelMutation",
        )
        assert len(result.errors) == 1
        assert result.errors[0].message == expected_error_message
        assert result.data is None

    async def test_updating_model_with_invalid_input_fails_with_expected_error(
        self,
        gql_client: AsyncGraphQLClient,
        variables: dict[str, Any],
        expected_error_message: str,
    ) -> None:
        result = await gql_client.execute(
            query=self.QUERY,
            variables=variables,
            operation_name="UpdateModelMutation",
        )
        assert len(result.errors) == 1
        assert result.errors[0].message == expected_error_message
        assert result.data is None


# --- tests/unit/server/api/routers/test_chat_tracing.py ---

    def test_error_propagates_to_both_spans(self, db: DbSessionFactory) -> None:
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        tracer = _make_tracer(db)
        messages = [ModelRequest(parts=[UserPromptPart(content="hi")])]

        agent_span = create_agent_span(tracer, input_messages=messages)
        llm_span = create_llm_span(tracer, parent_span=agent_span, input_messages=messages)

        error = RuntimeError("API timeout")
        finalize_llm_span(llm_span, error=error)
        finalize_agent_span(agent_span, error=error)

        db_traces = tracer.get_db_traces(project_id=1)
        for db_span in db_traces[0].spans:
            assert db_span.status_code == "ERROR"
            assert db_span.status_message == "API timeout"
            assert any(e["name"] == "exception" for e in db_span.events)


# --- tests/unit/server/api/routers/test_data_stream_protocol.py ---

    def test_accumulates_text(self) -> None:
        acc = StreamAccumulator()
        acc.text_parts.append("Hello ")
        acc.text_parts.append("world!")
        assert acc.accumulated_text == "Hello world!"

    def test_accumulates_tool_calls(self) -> None:
        acc = StreamAccumulator()
        # Simulate tool call start.
        acc._current_tool_meta[0] = {"id": "tc-1", "name": "search"}
        acc._current_tool_args[0] = ['{"q":', '"test"}']
        # Simulate tool call end.
        meta = acc._current_tool_meta.pop(0)
        args = acc._current_tool_args.pop(0)
        acc.tool_calls.append(
            {
                "id": meta["id"],
                "name": meta["name"],
                "arguments": "".join(args),
            }
        )
        assert len(acc.tool_calls) == 1
        assert acc.tool_calls[0]["id"] == "tc-1"
        assert acc.tool_calls[0]["name"] == "search"
        assert acc.tool_calls[0]["arguments"] == '{"q":"test"}'

    def test_empty_accumulator(self) -> None:
        acc = StreamAccumulator()
        assert acc.accumulated_text == ""
        assert acc.tool_calls == []


# --- tests/unit/server/api/routers/v1/test_annotations.py ---

async def test_list_span_annotations_empty_result_when_all_excluded(
    httpx_client: httpx.AsyncClient,
    db: DbSessionFactory,
) -> None:
    async with db() as session:
        project_row_id = await session.scalar(
            insert(models.Project).values(name="filtered-project").returning(models.Project.id)
        )

        trace_id = await session.scalar(
            insert(models.Trace)
            .values(
                trace_id="filtered-trace-id",
                project_rowid=project_row_id,
                start_time=datetime.fromisoformat("2021-01-01T00:00:00.000+00:00"),
                end_time=datetime.fromisoformat("2021-01-01T00:01:00.000+00:00"),
            )
            .returning(models.Trace.id)
        )

        span_id = await session.scalar(
            insert(models.Span)
            .values(
                trace_rowid=trace_id,
                span_id="filtered-span",
                parent_id=None,
                name="test span with filtered annotations",
                span_kind="CHAIN",
                start_time=datetime.fromisoformat("2021-01-01T00:00:00.000+00:00"),
                end_time=datetime.fromisoformat("2021-01-01T00:00:30.000+00:00"),
                attributes={},
                events=[],
                status_code="OK",
                status_message="",
                cumulative_error_count=0,
                cumulative_llm_token_count_prompt=0,
                cumulative_llm_token_count_completion=0,
            )
            .returning(models.Span.id)
        )

        await session.execute(
            insert(models.SpanAnnotation).values(
                span_rowid=span_id,
                name="test-annotation",
                label=None,
                score=None,
                explanation="This annotation will be excluded",
                metadata_={},
                annotator_kind="HUMAN",
                source="APP",
                identifier="test-identifier",
            )
        )

        await session.commit()

    response = await httpx_client.get(
        "v1/projects/filtered-project/span_annotations",
        params={"span_ids": ["filtered-span"], "exclude_annotation_names": ["test-annotation"]},
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["data"]) == 0
    assert data["next_cursor"] is None

async def test_list_span_annotations_pagination_with_filters(
    httpx_client: httpx.AsyncClient,
    db: DbSessionFactory,
) -> None:
    async with db() as session:
        project_row_id = await session.scalar(
            insert(models.Project).values(name="pagination-project").returning(models.Project.id)
        )

        trace_id = await session.scalar(
            insert(models.Trace)
            .values(
                trace_id="pagination-trace-id",
                project_rowid=project_row_id,
                start_time=datetime.fromisoformat("2021-01-01T00:00:00.000+00:00"),
                end_time=datetime.fromisoformat("2021-01-01T00:01:00.000+00:00"),
            )
            .returning(models.Trace.id)
        )

        span_id = await session.scalar(
            insert(models.Span)
            .values(
                trace_rowid=trace_id,
                span_id="pagination-span",
                parent_id=None,
                name="test span for pagination",
                span_kind="CHAIN",
                start_time=datetime.fromisoformat("2021-01-01T00:00:00.000+00:00"),
                end_time=datetime.fromisoformat("2021-01-01T00:00:30.000+00:00"),
                attributes={},
                events=[],
                status_code="OK",
                status_message="",
                cumulative_error_count=0,
                cumulative_llm_token_count_prompt=0,
                cumulative_llm_token_count_completion=0,
            )
            .returning(models.Span.id)
        )

        for i in range(5):
            await session.execute(
                insert(models.SpanAnnotation).values(
                    span_rowid=span_id,
                    name=f"annotation-{i}",
                    label=f"label-{i}",
                    score=0.1 * i,
                    explanation=f"Explanation {i}",
                    metadata_={},
                    annotator_kind="HUMAN",
                    source="API",
                    identifier=f"identifier-{i}",
                )
            )

        for i in range(3):
            await session.execute(
                insert(models.SpanAnnotation).values(
                    span_rowid=span_id,
                    name="excluded-annotation",
                    label=None,
                    score=None,
                    explanation=f"Excluded annotation {i}",
                    metadata_={},
                    annotator_kind="HUMAN",
                    source="APP",
                    identifier=f"excluded-identifier-{i}",
                )
            )

        await session.commit()

    response = await httpx_client.get(
        "v1/projects/pagination-project/span_annotations",
        params={
            "span_ids": ["pagination-span"],
            "limit": 3,
            "exclude_annotation_names": ["excluded-annotation"],
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["data"]) == 3

    for annotation in data["data"]:
        assert annotation["name"] != "excluded-annotation"
        assert annotation["name"].startswith("annotation-")

    assert data["next_cursor"] is not None


# --- tests/unit/server/api/routers/v1/test_datasets.py ---

async def test_post_dataset_upload_pyarrow_create_then_append(
    httpx_client: httpx.AsyncClient,
    db: DbSessionFactory,
) -> None:
    name = inspect.stack()[0][3]
    df = pd.read_csv(StringIO("a,b,c,d,e,f\n1,2,3,4,5,6\n"))
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    file = BytesIO(sink.getvalue().to_pybytes())
    response = await httpx_client.post(
        url="v1/datasets/upload?sync=true",
        files={"file": (" ", file, "application/x-pandas-pyarrow", {})},
        data={
            "action": "create",
            "name": name,
            "input_keys[]": ["a", "b", "c"],
            "output_keys[]": ["b", "c", "d"],
            "metadata_keys[]": ["c", "d", "e"],
        },
    )
    assert response.status_code == 200
    assert (data := response.json().get("data"))
    assert (dataset_id := data.get("dataset_id"))
    assert "version_id" in data
    version_id_str = data["version_id"]
    version_global_id = GlobalID.from_id(version_id_str)
    assert version_global_id.type_name == "DatasetVersion"
    del response, file, data, df, table, sink
    df = pd.read_csv(StringIO("a,b,c,d,e,f\n11,22,33,44,55,66\n"))
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    file = BytesIO(sink.getvalue().to_pybytes())
    response = await httpx_client.post(
        url="v1/datasets/upload?sync=true",
        files={"file": (" ", file, "application/x-pandas-pyarrow", {})},
        data={
            "action": "append",
            "name": name,
            "input_keys[]": ["a", "b", "c"],
            "output_keys[]": ["b", "c", "d"],
            "metadata_keys[]": ["c", "d", "e"],
        },
    )
    assert response.status_code == 200
    assert (data := response.json().get("data"))
    assert dataset_id == data.get("dataset_id")
    assert "version_id" in data
    version_id_str = data["version_id"]
    version_global_id = GlobalID.from_id(version_id_str)
    assert version_global_id.type_name == "DatasetVersion"
    async with db() as session:
        revisions = list(
            await session.scalars(
                select(models.DatasetExampleRevision)
                .join(models.DatasetExample)
                .join_from(models.DatasetExample, models.Dataset)
                .where(models.Dataset.name == name)
                .order_by(models.DatasetExample.id)
            )
        )
    assert len(revisions) == 2
    assert revisions[0].input == {"a": 1, "b": 2, "c": 3}
    assert revisions[0].output == {"b": 2, "c": 3, "d": 4}
    assert revisions[0].metadata_ == {"c": 3, "d": 4, "e": 5}
    assert revisions[1].input == {"a": 11, "b": 22, "c": 33}
    assert revisions[1].output == {"b": 22, "c": 33, "d": 44}
    assert revisions[1].metadata_ == {"c": 33, "d": 44, "e": 55}

    # Verify the DatasetVersion from the response
    db_dataset_version = await session.get(models.DatasetVersion, int(version_global_id.node_id))
    assert db_dataset_version is not None
    assert db_dataset_version.dataset_id == int(GlobalID.from_id(dataset_id).node_id)

async def test_post_dataset_upload_pyarrow_with_splits(
    httpx_client: httpx.AsyncClient,
    db: DbSessionFactory,
) -> None:
    """Test PyArrow upload with split_keys."""
    name = inspect.stack()[0][3]
    df = pd.read_csv(StringIO("question,answer,data_split\nQ1,A1,train\nQ2,A2,test\n"))
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    file = BytesIO(sink.getvalue().to_pybytes())

    response = await httpx_client.post(
        url="v1/datasets/upload?sync=true",
        files={"file": (" ", file, "application/x-pandas-pyarrow", {})},
        data={
            "action": "create",
            "name": name,
            "input_keys[]": ["question"],
            "output_keys[]": ["answer"],
            "split_keys[]": ["data_split"],
        },
    )
    assert response.status_code == 200
    assert (data := response.json().get("data"))
    assert (dataset_id := data.get("dataset_id"))

    async with db() as session:
        splits = list(
            await session.scalars(select(models.DatasetSplit).order_by(models.DatasetSplit.name))
        )
        assert set(s.name for s in splits) == {"train", "test"}

        # Verify example assignments
        dataset_db_id = int(GlobalID.from_id(dataset_id).node_id)
        examples = list(
            await session.scalars(
                select(models.DatasetExample)
                .where(models.DatasetExample.dataset_id == dataset_db_id)
                .order_by(models.DatasetExample.id)
            )
        )
        assert len(examples) == 2

        async def get_example_splits(example_id: int) -> set[str]:
            result = await session.scalars(
                select(models.DatasetSplit)
                .join(models.DatasetSplitDatasetExample)
                .where(models.DatasetSplitDatasetExample.dataset_example_id == example_id)
            )
            return {s.name for s in result}

        assert await get_example_splits(examples[0].id) == {"train"}
        assert await get_example_splits(examples[1].id) == {"test"}


# --- tests/unit/server/api/routers/v1/test_experiments.py ---

async def test_experiments_api(
    httpx_client: httpx.AsyncClient,
    simple_dataset: Any,
    db: DbSessionFactory,
) -> None:
    """
    A simple test of the expected flow for the experiments API flow
    """

    dataset_gid = GlobalID("Dataset", "0")

    # first, create an experiment associated with a dataset
    created_experiment = (
        await httpx_client.post(
            f"v1/datasets/{dataset_gid}/experiments",
            json={"version_id": None, "repetitions": 1},
        )
    ).json()["data"]

    experiment_gid = created_experiment["id"]
    version_gid = created_experiment["dataset_version_id"]
    assert created_experiment["repetitions"] == 1

    dataset_examples = (
        await httpx_client.get(
            f"v1/datasets/{dataset_gid}/examples",
            params={"version_id": str(version_gid)},
        )
    ).json()["data"]["examples"]

    # Verify that the experiment examples snapshot was created in the junction table
    async with db() as session:
        await verify_experiment_examples_junction_table(session, experiment_gid)

    # experiments can be read using the GET /experiments route
    experiment = (await httpx_client.get(f"v1/experiments/{experiment_gid}")).json()["data"]
    assert experiment
    assert created_experiment["repetitions"] == 1

    # get experiment JSON before any runs - should return 404
    response = await httpx_client.get(f"v1/experiments/{experiment_gid}/json")
    assert response.status_code == 404
    assert "has no runs" in response.text

    # create experiment runs for each dataset example
    run_payload = {
        "dataset_example_id": str(dataset_examples[0]["id"]),
        "trace_id": "placeholder-id",
        "output": "some LLM application output",
        "repetition_number": 1,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "error": "an error message, if applicable",
    }
    run_payload["id"] = (
        await httpx_client.post(
            f"v1/experiments/{experiment_gid}/runs",
            json=run_payload,
        )
    ).json()["data"]["id"]

    # get experiment JSON after runs but before evaluations
    response = await httpx_client.get(f"v1/experiments/{experiment_gid}/json")
    assert response.status_code == 200
    runs = json.loads(response.text)
    assert len(runs) == 1
    run = runs[0]
    assert isinstance(run.pop("example_id"), str)
    assert run.pop("repetition_number") == 1
    assert run.pop("input") == {"in": "foo"}
    assert run.pop("reference_output") == {"out": "bar"}
    assert run.pop("output") == "some LLM application output"
    assert run.pop("error") == "an error message, if applicable"
    assert isinstance(run.pop("latency_ms"), float)
    assert isinstance(run.pop("start_time"), str)
    assert isinstance(run.pop("end_time"), str)
    assert run.pop("trace_id") == "placeholder-id"
    assert run.pop("prompt_token_count") is None
    assert run.pop("completion_token_count") is None
    assert run.pop("annotations") == []
    assert not run

    # get experiment CSV after runs but before evaluations
    response = await httpx_client.get(f"v1/experiments/{experiment_gid}/csv")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv"
    assert response.headers["content-disposition"].startswith('attachment; filename="')

    # Parse CSV content and verify the data
    csv_content = response.text
    df = pd.read_csv(StringIO(csv_content))
    assert len(df) == 1

    # Convert first row to dictionary and verify all fields
    row = df.iloc[0].to_dict()
    assert isinstance(row.pop("example_id"), str)
    assert row.pop("repetition_number") == 1
    assert json.loads(row.pop("input")) == {"in": "foo"}
    assert json.loads(row.pop("reference_output")) == {"out": "bar"}
    assert row.pop("output") == "some LLM application output"
    assert row.pop("error") == "an error message, if applicable"
    assert isinstance(row.pop("latency_ms"), float)
    assert isinstance(row.pop("start_time"), str)
    assert isinstance(row.pop("end_time"), str)
    assert row.pop("trace_id") == "placeholder-id"
    assert pd.isna(row.pop("prompt_token_count"))
    assert pd.isna(row.pop("completion_token_count"))
    assert not row

    # experiment runs can be listed for evaluations
    experiment_runs = (await httpx_client.get(f"v1/experiments/{experiment_gid}/runs")).json()[
        "data"
    ]
    assert experiment_runs
    assert len(experiment_runs) == 1

    # each experiment run can be evaluated
    evaluation_payload = {
        "experiment_run_id": run_payload["id"],
        "trace_id": "placeholder-id",
        "name": "some_evaluation_name",
        "annotator_kind": "LLM",
        "result": {
            "label": "some label",
            "score": 0.5,
            "explanation": "some explanation",
            "metadata": {"some": "metadata"},
        },
        "error": "an error message, if applicable",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
    }
    experiment_evaluation = (
        await httpx_client.post("v1/experiment_evaluations", json=evaluation_payload)
    ).json()
    assert experiment_evaluation

    # get experiment JSON after adding evaluations
    response = await httpx_client.get(f"v1/experiments/{experiment_gid}/json")
    assert response.status_code == 200
    runs = json.loads(response.text)
    assert len(runs) == 1
    assert len(runs[0]["annotations"]) == 1
    annotation = runs[0]["annotations"][0]
    assert annotation.pop("name") == "some_evaluation_name"
    assert annotation.pop("label") == "some label"
    assert annotation.pop("score") == 0.5
    assert annotation.pop("explanation") == "some explanation"
    assert annotation.pop("metadata") == {}
    assert annotation.pop("annotator_kind") == "LLM"
    assert annotation.pop("trace_id") == "placeholder-id"
    assert annotation.pop("error") == "an error message, if applicable"
    assert isinstance(annotation.pop("start_time"), str)
    assert isinstance(annotation.pop("end_time"), str)
    assert not annotation

    # get experiment CSV after evaluations
    response = await httpx_client.get(f"v1/experiments/{experiment_gid}/csv")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv"
    assert response.headers["content-disposition"].startswith('attachment; filename="')

    # Parse CSV content and verify the data with annotations
    csv_content = response.text
    df = pd.read_csv(StringIO(csv_content))
    assert len(df) == 1

    # Verify base fields
    row = df.iloc[0].to_dict()
    assert isinstance(row.pop("example_id"), str)
    assert row.pop("repetition_number") == 1
    assert json.loads(row.pop("input")) == {"in": "foo"}
    assert json.loads(row.pop("reference_output")) == {"out": "bar"}
    assert row.pop("output") == "some LLM application output"
    assert row.pop("error") == "an error message, if applicable"
    assert isinstance(row.pop("latency_ms"), float)
    assert isinstance(row.pop("start_time"), str)
    assert isinstance(row.pop("end_time"), str)
    assert row.pop("trace_id") == "placeholder-id"
    assert pd.isna(row.pop("prompt_token_count"))
    assert pd.isna(row.pop("completion_token_count"))

    # Verify annotation fields
    annotation_prefix = "annotation_some_evaluation_name"
    assert row.pop(f"{annotation_prefix}_label") == "some label"
    assert row.pop(f"{annotation_prefix}_score") == 0.5
    assert row.pop(f"{annotation_prefix}_explanation") == "some explanation"
    assert json.loads(row.pop(f"{annotation_prefix}_metadata")) == {}
    assert row.pop(f"{annotation_prefix}_annotator_kind") == "LLM"
    assert row.pop(f"{annotation_prefix}_trace_id") == "placeholder-id"
    assert row.pop(f"{annotation_prefix}_error") == "an error message, if applicable"
    assert isinstance(row.pop(f"{annotation_prefix}_start_time"), str)
    assert isinstance(row.pop(f"{annotation_prefix}_end_time"), str)
    assert not row

async def test_deleting_dataset_also_deletes_experiments(
    httpx_client: httpx.AsyncClient,
    dataset_with_experiments_runs_and_evals: Any,
) -> None:
    ds_url = f"v1/datasets/{GlobalID('Dataset', str(1))}"
    exp_url = f"v1/experiments/{GlobalID('Experiment', str(1))}"
    runs_url = f"{exp_url}/runs"
    (await httpx_client.get(exp_url)).raise_for_status()
    assert len((await httpx_client.get(runs_url)).json()["data"]) > 0
    (await httpx_client.delete(ds_url)).raise_for_status()
    assert len((await httpx_client.get(runs_url)).json()["data"]) == 0
    with pytest.raises(HTTPStatusError):
        (await httpx_client.get(exp_url)).raise_for_status()

async def test_experiment_runs_pagination(
    httpx_client: httpx.AsyncClient,
    simple_dataset: Any,
) -> None:
    """Test pagination functionality for experiment runs endpoint."""
    dataset_gid = GlobalID("Dataset", "0")

    # Create experiment and runs
    experiment = (
        await httpx_client.post(
            f"v1/datasets/{dataset_gid}/experiments",
            json={"version_id": None, "repetitions": 1},
        )
    ).json()["data"]

    dataset_examples = (
        await httpx_client.get(
            f"v1/datasets/{dataset_gid}/examples",
            params={"version_id": str(experiment["dataset_version_id"])},
        )
    ).json()["data"]["examples"]

    # Create 5 runs for pagination testing
    created_runs = []
    for i in range(5):
        run = (
            await httpx_client.post(
                f"v1/experiments/{experiment['id']}/runs",
                json={
                    "dataset_example_id": str(dataset_examples[0]["id"]),
                    "trace_id": f"trace-{i}",
                    "output": f"output-{i}",
                    "repetition_number": i + 1,
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "end_time": datetime.now(timezone.utc).isoformat(),
                },
            )
        ).json()["data"]
        created_runs.append(run["id"])

    def get_numeric_ids(run_ids: list[str]) -> list[int]:
        """Helper to extract numeric IDs for comparison."""
        return [int(GlobalID.from_id(run_id).node_id) for run_id in run_ids]

    # Expected order: descending by numeric ID
    expected_ids = sorted(get_numeric_ids(created_runs), reverse=True)  # [5, 4, 3, 2, 1]

    async def get_runs(limit: Optional[int] = None, cursor: Optional[str] = None) -> dict[str, Any]:
        """Helper to fetch runs with optional pagination."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        response = await httpx_client.get(f"v1/experiments/{experiment['id']}/runs", params=params)
        assert response.status_code == 200
        return response.json()  # type: ignore[no-any-return]

    # Test: No pagination (backward compatibility)
    all_runs = await get_runs()
    assert len(all_runs["data"]) == 5
    assert all_runs["next_cursor"] is None
    all_runs_ids = [run["id"] for run in all_runs["data"]]
    assert get_numeric_ids(all_runs_ids) == expected_ids

    # Test: Page-by-page pagination with exact content validation
    page1 = await get_runs(limit=2)
    assert len(page1["data"]) == 2
    assert page1["next_cursor"] is not None
    page1_ids = get_numeric_ids([run["id"] for run in page1["data"]])
    assert page1_ids == expected_ids[:2]  # [5, 4]
    assert GlobalID.from_id(page1["next_cursor"]).node_id == str(expected_ids[2])  # "3"

    page2 = await get_runs(limit=2, cursor=page1["next_cursor"])
    assert len(page2["data"]) == 2
    assert page2["next_cursor"] is not None
    page2_ids = get_numeric_ids([run["id"] for run in page2["data"]])
    assert page2_ids == expected_ids[2:4]  # [3, 2]
    assert GlobalID.from_id(page2["next_cursor"]).node_id == str(expected_ids[4])  # "1"

    page3 = await get_runs(limit=2, cursor=page2["next_cursor"])
    assert len(page3["data"]) == 1
    assert page3["next_cursor"] is None
    page3_ids = get_numeric_ids([run["id"] for run in page3["data"]])
    assert page3_ids == expected_ids[4:5]  # [1]

    # Test: Aggregated pagination equals non-paginated
    paginated_ids = page1_ids + page2_ids + page3_ids
    assert paginated_ids == expected_ids
    paginated_run_ids = [run["id"] for run in page1["data"] + page2["data"] + page3["data"]]
    assert paginated_run_ids == all_runs_ids

    # Test: Large limit (no pagination)
    large_limit = await get_runs(limit=100)
    assert len(large_limit["data"]) == 5
    assert large_limit["next_cursor"] is None
    assert get_numeric_ids([run["id"] for run in large_limit["data"]]) == expected_ids

    # Test: Invalid cursor
    response = await httpx_client.get(
        f"v1/experiments/{experiment['id']}/runs", params={"limit": 2, "cursor": "invalid-cursor"}
    )
    assert response.status_code == 422

    async def test_comprehensive_count_scenarios(
        self,
        httpx_client: httpx.AsyncClient,
        experiments_with_incomplete_runs: ExperimentsWithIncompleteRuns,
    ) -> None:
        """
        Comprehensive test for example_count, successful_run_count, failed_run_count, and missing_run_count fields.

        Scenarios tested:
        1. Mixed runs (v1) - some successful, some failed, some missing
        2. No runs at all (v1) - zero successful and failed runs, all missing
        3. Deleted examples (v2) - handles dataset versioning with deletions
        4. Incremental additions (v2) - successful -> failed -> successful progression
        5. List endpoint - multiple experiments with correct counts
        6. Create endpoint - returns correct initial counts
        7. All runs failed - edge case where runs exist but successful_run_count = 0
        8. Simple boundary - minimal viable case (1 repetition, 1 successful run)

        missing_run_count is calculated as: (example_count × repetitions) - successful_run_count - failed_run_count
        """
        dataset = experiments_with_incomplete_runs.dataset
        exp_v1_mixed = experiments_with_incomplete_runs.experiment_v1_mixed
        exp_v1_empty = experiments_with_incomplete_runs.experiment_v1_empty
        exp_v2_deletion = experiments_with_incomplete_runs.experiment_v2_with_deletion
        exp_v2_incremental = experiments_with_incomplete_runs.experiment_v2_incremental
        examples = experiments_with_incomplete_runs.examples_in_v1

        # Convert to GlobalIDs
        dataset_gid = GlobalID("Dataset", str(dataset.id))
        exp_v1_mixed_gid = GlobalID("Experiment", str(exp_v1_mixed.id))
        exp_v1_empty_gid = GlobalID("Experiment", str(exp_v1_empty.id))
        exp_v2_deletion_gid = GlobalID("Experiment", str(exp_v2_deletion.id))
        exp_v2_incremental_gid = GlobalID("Experiment", str(exp_v2_incremental.id))

        # ===== Test 1: Experiment with mixed successful and failed runs (v1) =====
        # exp_v1_mixed: has 5 examples, 7 successful runs, 3 failed runs, 3 repetitions
        # Total expected: 5 × 3 = 15 runs
        # (ex0: 3 successful, ex1: 1 successful + 1 failed,
        #  ex2: 0 runs, ex3: 2 successful + 1 failed, ex4: 1 successful + 1 failed)
        exp1_data = await self._get_experiment(httpx_client, exp_v1_mixed_gid)
        assert exp1_data["example_count"] == 5, "exp_v1_mixed should have 5 examples"
        assert exp1_data["successful_run_count"] == 7, (
            "exp_v1_mixed should have 7 successful runs (3+1+0+2+1)"
        )
        assert exp1_data["failed_run_count"] == 3, (
            "exp_v1_mixed should have 3 failed runs (0+1+0+1+1)"
        )
        assert exp1_data["missing_run_count"] == 5, (
            "exp_v1_mixed should have 5 missing runs (15 total - 7 successful - 3 failed)"
        )

        # ===== Test 2: Experiment with no runs at all (v1) =====
        # exp_v1_empty: 5 examples, 2 repetitions = 10 total expected runs
        exp2_data = await self._get_experiment(httpx_client, exp_v1_empty_gid)
        assert exp2_data["example_count"] == 5, "exp_v1_empty should have 5 examples"
        assert exp2_data["successful_run_count"] == 0, "exp_v1_empty should have 0 successful runs"
        assert exp2_data["failed_run_count"] == 0, "exp_v1_empty should have 0 failed runs"
        assert exp2_data["missing_run_count"] == 10, (
            "exp_v1_empty should have 10 missing runs (5 examples × 2 repetitions)"
        )

        # ===== Test 3: Experiment with deleted example in v2 =====
        # exp_v2_deletion: has 4 examples (ex2 deleted from v2), 4 successful runs, 1 failed, 2 repetitions
        # Total expected: 4 × 2 = 8 runs
        exp3_data = await self._get_experiment(httpx_client, exp_v2_deletion_gid)
        assert exp3_data["example_count"] == 4, (
            "exp_v2_deletion should have 4 examples (ex2 deleted)"
        )
        assert exp3_data["successful_run_count"] == 4, (
            "exp_v2_deletion should have 4 successful runs (2+1+0+1)"
        )
        assert exp3_data["failed_run_count"] == 1, (
            "exp_v2_deletion should have 1 failed run (0+1+0+0)"
        )
        assert exp3_data["missing_run_count"] == 3, (
            "exp_v2_deletion should have 3 missing runs (8 total - 4 successful - 1 failed)"
        )

        # ===== Test 4: Fresh experiment (v2), then incrementally add runs =====
        # exp_v2_incremental: has 2 examples, 3 repetitions = 6 total expected runs
        exp4_data = await self._get_experiment(httpx_client, exp_v2_incremental_gid)
        assert exp4_data["example_count"] == 2, "exp_v2_incremental should have 2 examples"
        assert exp4_data["successful_run_count"] == 0, (
            "exp_v2_incremental should start with 0 successful runs"
        )
        assert exp4_data["failed_run_count"] == 0, (
            "exp_v2_incremental should start with 0 failed runs"
        )
        assert exp4_data["missing_run_count"] == 6, (
            "exp_v2_incremental should start with 6 missing runs (2 × 3)"
        )

        # Add a successful run for the first example
        example_gid_0 = GlobalID("DatasetExample", str(examples[0].id))
        await self._create_run(
            httpx_client, exp_v2_incremental_gid, example_gid_0, 1, "test-trace-1", "success output"
        )

        # Verify count increased after successful run
        exp4_data = await self._get_experiment(httpx_client, exp_v2_incremental_gid)
        assert exp4_data["example_count"] == 2
        assert exp4_data["successful_run_count"] == 1, (
            "Should have 1 successful run after adding one"
        )
        assert exp4_data["failed_run_count"] == 0, "Should still have 0 failed runs"
        assert exp4_data["missing_run_count"] == 5, "Should have 5 missing runs (6 - 1)"

        # Add a failed run for the first example (different repetition)
        await self._create_run(
            httpx_client,
            exp_v2_incremental_gid,
            example_gid_0,
            2,
            "test-trace-2",
            "error output",
            error="Test error occurred",
        )

        # Verify failed run doesn't increment successful_run_count but decrements missing_run_count
        exp4_data = await self._get_experiment(httpx_client, exp_v2_incremental_gid)
        assert exp4_data["example_count"] == 2
        assert exp4_data["successful_run_count"] == 1, (
            "Failed run should not increment successful count"
        )
        assert exp4_data["failed_run_count"] == 1, "Should have 1 failed run after adding one"
        assert exp4_data["missing_run_count"] == 4, "Should have 4 missing runs (6 - 1 - 1)"

        # Add another successful run
        await self._create_run(
            httpx_client, exp_v2_incremental_gid, example_gid_0, 3, "test-trace-3", "success output"
        )

        # Verify count increased again
        exp4_data = await self._get_experiment(httpx_client, exp_v2_incremental_gid)
        assert exp4_data["example_count"] == 2
        assert exp4_data["successful_run_count"] == 2, "Should have 2 successful runs now"
        assert exp4_data["failed_run_count"] == 1, "Should still have 1 failed run"
        assert exp4_data["missing_run_count"] == 3, "Should have 3 missing runs (6 - 2 - 1)"

        # ===== Test 5: List experiments endpoint returns all with correct counts =====
        list_response = await httpx_client.get(f"v1/datasets/{dataset_gid}/experiments")
        assert list_response.status_code == 200
        experiments_list = list_response.json()["data"]

        assert len(experiments_list) == 4, "Should have 4 experiments"

        # Find the experiments in the list (order might vary)
        exp1_in_list = next(e for e in experiments_list if e["id"] == str(exp_v1_mixed_gid))
        exp2_in_list = next(e for e in experiments_list if e["id"] == str(exp_v1_empty_gid))
        exp3_in_list = next(e for e in experiments_list if e["id"] == str(exp_v2_deletion_gid))
        exp4_in_list = next(e for e in experiments_list if e["id"] == str(exp_v2_incremental_gid))

        # Verify counts in list endpoint match individual GET requests
        assert exp1_in_list["example_count"] == 5
        assert exp1_in_list["successful_run_count"] == 7
        assert exp1_in_list["failed_run_count"] == 3
        assert exp1_in_list["missing_run_count"] == 5
        assert exp2_in_list["example_count"] == 5
        assert exp2_in_list["successful_run_count"] == 0
        assert exp2_in_list["failed_run_count"] == 0
        assert exp2_in_list["missing_run_count"] == 10
        assert exp3_in_list["example_count"] == 4  # ex2 deleted in v2
        assert exp3_in_list["successful_run_count"] == 4
        assert exp3_in_list["failed_run_count"] == 1
        assert exp3_in_list["missing_run_count"] == 3
        assert exp4_in_list["example_count"] == 2
        assert exp4_in_list["successful_run_count"] == 2
        assert exp4_in_list["failed_run_count"] == 1
        assert exp4_in_list["missing_run_count"] == 3

        # ===== Test 6: Create endpoint returns correct initial counts =====
        # Create a fresh experiment and verify the create response has correct counts
        new_exp_response = await httpx_client.post(
            f"v1/datasets/{dataset_gid}/experiments",
            json={"version_id": None, "repetitions": 1},
        )
        assert new_exp_response.status_code == 200
        new_exp_data = new_exp_response.json()["data"]

        # Verify counts in create response (not just GET)
        assert new_exp_data["example_count"] == 5, "Create response should have example_count"
        assert new_exp_data["successful_run_count"] == 0, (
            "Create response should start with 0 successful runs"
        )
        assert new_exp_data["failed_run_count"] == 0, (
            "Create response should start with 0 failed runs"
        )
        assert new_exp_data["missing_run_count"] == 5, (
            "Create response should start with 5 missing runs (5 examples × 1 repetition)"
        )

        # ===== Test 7: Edge case - All runs failed =====
        new_exp_gid = new_exp_data["id"]

        # Add only failed runs for all examples
        for i, example in enumerate(examples):
            example_gid = GlobalID("DatasetExample", str(example.id))
            await self._create_run(
                httpx_client,
                new_exp_gid,
                example_gid,
                1,
                f"all-failed-trace-{i}",
                "failed output",
                error=f"All runs failed - example {i}",
            )

        # Verify that with all runs failed, successful_run_count is still 0 but failed_run_count is 5
        all_failed_data = await self._get_experiment(httpx_client, new_exp_gid)
        assert all_failed_data["example_count"] == 5
        assert all_failed_data["successful_run_count"] == 0, (
            "All failed runs should result in 0 successful count"
        )
        assert all_failed_data["failed_run_count"] == 5, (
            "All failed runs should result in 5 failed count"
        )
        assert all_failed_data["missing_run_count"] == 0, (
            "All failed runs should result in 0 missing count"
        )

        # ===== Test 8: Simple boundary case - 1 example, 1 repetition =====
        # This is the simplest possible experiment
        simple_exp_response = await httpx_client.post(
            f"v1/datasets/{dataset_gid}/experiments",
            json={"version_id": None, "repetitions": 1},
        )
        simple_exp_data = simple_exp_response.json()["data"]
        simple_exp_gid = simple_exp_data["id"]

        # Verify simple case starts correctly
        assert simple_exp_data["example_count"] == 5
        assert simple_exp_data["successful_run_count"] == 0
        assert simple_exp_data["failed_run_count"] == 0
        assert simple_exp_data["missing_run_count"] == 5

        # Add exactly 1 successful run
        await self._create_run(
            httpx_client,
            simple_exp_gid,
            GlobalID("DatasetExample", str(examples[0].id)),
            1,
            "simple-success",
            "simple output",
        )

        # Verify count is exactly 1
        simple_data = await self._get_experiment(httpx_client, simple_exp_gid)
        assert simple_data["example_count"] == 5
        assert simple_data["successful_run_count"] == 1, (
            "Simple 1-run case should have exactly 1 successful"
        )
        assert simple_data["failed_run_count"] == 0, "Simple 1-run case should have 0 failed runs"
        assert simple_data["missing_run_count"] == 4, (
            "Simple 1-run case should have 4 missing runs (5 - 1)"
        )


# --- tests/unit/server/api/routers/v1/test_spans.py ---

async def test_delete_span_cumulative_metrics_propagation(
    httpx_client: httpx.AsyncClient,
    span_hierarchy_with_metrics: dict[str, Any],
    db: DbSessionFactory,
) -> None:
    """Test that cumulative metrics are properly updated when deleting a span with a parent."""
    hierarchy = span_hierarchy_with_metrics

    # Get initial metrics for root span
    async with db() as session:
        initial_root = await session.scalar(
            select(models.Span).where(models.Span.id == hierarchy["root"].id)
        )
        assert initial_root is not None
        initial_root_errors = initial_root.cumulative_error_count
        initial_root_prompt = initial_root.cumulative_llm_token_count_prompt
        initial_root_completion = initial_root.cumulative_llm_token_count_completion

    # Delete child span (should subtract child's cumulative values from root)
    child_errors = hierarchy["child"].cumulative_error_count
    child_prompt = hierarchy["child"].cumulative_llm_token_count_prompt
    child_completion = hierarchy["child"].cumulative_llm_token_count_completion

    response = await httpx_client.delete("v1/spans/child-span")
    assert response.status_code == 204

    # Verify metrics propagation
    async with db() as session:
        # Child should be deleted
        remaining_child = await session.scalar(
            select(models.Span).where(models.Span.id == hierarchy["child"].id)
        )
        assert remaining_child is None

        # Root metrics should be reduced by child's cumulative values
        updated_root = await session.scalar(
            select(models.Span).where(models.Span.id == hierarchy["root"].id)
        )
        assert updated_root is not None

        expected_root_errors = initial_root_errors - child_errors
        expected_root_prompt = initial_root_prompt - child_prompt
        expected_root_completion = initial_root_completion - child_completion

        assert updated_root.cumulative_error_count == expected_root_errors
        assert updated_root.cumulative_llm_token_count_prompt == expected_root_prompt
        assert updated_root.cumulative_llm_token_count_completion == expected_root_completion

        # Grandchild should still exist (orphaned but not deleted)
        remaining_grandchild = await session.scalar(
            select(models.Span).where(models.Span.id == hierarchy["grandchild"].id)
        )
        assert remaining_grandchild is not None
        # Grandchild metrics should be unchanged
        assert remaining_grandchild.cumulative_error_count == 2
        assert remaining_grandchild.cumulative_llm_token_count_prompt == 20
        assert remaining_grandchild.cumulative_llm_token_count_completion == 40

async def test_delete_span_no_metrics_propagation_when_no_parent(
    httpx_client: httpx.AsyncClient,
    span_hierarchy_with_metrics: dict[str, Any],
    db: DbSessionFactory,
) -> None:
    """Test that no metrics propagation occurs when deleting a root span (no parent)."""
    hierarchy = span_hierarchy_with_metrics

    # Delete root span (no parent, so no propagation should occur)
    response = await httpx_client.delete("v1/spans/root-span")
    assert response.status_code == 204

    # Verify root is deleted and children remain with unchanged metrics
    async with db() as session:
        # Root should be deleted
        remaining_root = await session.scalar(
            select(models.Span).where(models.Span.id == hierarchy["root"].id)
        )
        assert remaining_root is None

        # Child and grandchild should still exist with original metrics
        remaining_child = await session.scalar(
            select(models.Span).where(models.Span.id == hierarchy["child"].id)
        )
        assert remaining_child is not None
        assert remaining_child.cumulative_error_count == 5
        assert remaining_child.cumulative_llm_token_count_prompt == 50
        assert remaining_child.cumulative_llm_token_count_completion == 100

        remaining_grandchild = await session.scalar(
            select(models.Span).where(models.Span.id == hierarchy["grandchild"].id)
        )
        assert remaining_grandchild is not None
        assert remaining_grandchild.cumulative_error_count == 2
        assert remaining_grandchild.cumulative_llm_token_count_prompt == 20
        assert remaining_grandchild.cumulative_llm_token_count_completion == 40


# --- tests/unit/server/api/routers/v1/test_traces.py ---

async def test_delete_trace_with_multiple_spans(
    httpx_client: httpx.AsyncClient,
    db: DbSessionFactory,
) -> None:
    """
    Test deleting a trace that contains multiple spans.

    This test verifies that:
    1. All spans in the trace are deleted via CASCADE
    2. Parent-child span relationships don't prevent deletion
    """
    # Create a trace with multiple spans (parent and child)
    async with db() as session:
        project_row_id = await session.scalar(
            insert(models.Project).values(name="multi-span-project").returning(models.Project.id)
        )
        trace_row_id = await session.scalar(
            insert(models.Trace)
            .values(
                trace_id="multispantrace123456789abcdef",
                project_rowid=project_row_id,
                start_time=datetime.fromisoformat("2021-01-01T00:00:00.000+00:00"),
                end_time=datetime.fromisoformat("2021-01-01T00:01:00.000+00:00"),
            )
            .returning(models.Trace.id)
        )

        # Create parent span
        parent_span_id = await session.scalar(
            insert(models.Span)
            .values(
                trace_rowid=trace_row_id,
                span_id="parentspan123456",
                parent_id=None,
                name="parent span",
                span_kind="CHAIN",
                start_time=datetime.fromisoformat("2021-01-01T00:00:00.000+00:00"),
                end_time=datetime.fromisoformat("2021-01-01T00:00:30.000+00:00"),
                attributes={"type": "parent"},
                events=[],
                status_code="OK",
                status_message="okay",
                cumulative_error_count=0,
                cumulative_llm_token_count_prompt=0,
                cumulative_llm_token_count_completion=0,
            )
            .returning(models.Span.id)
        )

        # Create child span
        child_span_id = await session.scalar(
            insert(models.Span)
            .values(
                trace_rowid=trace_row_id,
                span_id="childspan1234567",
                parent_id="parentspan123456",
                name="child span",
                span_kind="LLM",
                start_time=datetime.fromisoformat("2021-01-01T00:00:05.000+00:00"),
                end_time=datetime.fromisoformat("2021-01-01T00:00:25.000+00:00"),
                attributes={"type": "child"},
                events=[],
                status_code="OK",
                status_message="okay",
                cumulative_error_count=0,
                cumulative_llm_token_count_prompt=0,
                cumulative_llm_token_count_completion=0,
            )
            .returning(models.Span.id)
        )

        await session.commit()

    # Delete the trace
    url = "v1/traces/multispantrace123456789abcdef"
    response = await httpx_client.delete(url)
    assert response.status_code == 204

    # Verify trace and all spans are deleted
    async with db() as session:
        deleted_trace = await session.get(models.Trace, trace_row_id)
        assert deleted_trace is None, "Trace should be deleted"

        deleted_parent = await session.get(models.Span, parent_span_id)
        assert deleted_parent is None, "Parent span should be deleted via CASCADE"

        deleted_child = await session.get(models.Span, child_span_id)
        assert deleted_child is None, "Child span should be deleted via CASCADE"


# --- tests/unit/server/api/types/test_DatasetExample.py ---

async def test_dataset_example_span_resolver(
    example_id: str,
    expected_span: Mapping[str, Any],
    gql_client: AsyncGraphQLClient,
    dataset_with_span_and_nonspan_examples: Any,
) -> None:
    query = """
      query ($exampleId: ID!) {
        example: node(id: $exampleId) {
          ... on DatasetExample {
            id
            span {
              context {
                spanId
                traceId
              }
              name
              spanKind
              startTime
              endTime
              attributes
              events {
                name
              }
              statusCode
              statusMessage
              cumulativeTokenCountPrompt
              cumulativeTokenCountCompletion
              cumulativeTokenCountTotal
            }
          }
        }
      }
    """
    response = await gql_client.execute(
        query=query,
        variables={"exampleId": example_id},
    )
    assert not response.errors
    assert (data := response.data) is not None
    assert data["example"] == {
        "id": example_id,
        "span": expected_span,
    }

async def test_dataset_example_experiment_runs_resolver_returns_relevant_runs(
    gql_client: AsyncGraphQLClient,
    example_with_experiment_runs: Any,
) -> None:
    query = """
      query ($exampleId: ID!) {
        example: node(id: $exampleId) {
          ... on DatasetExample {
            experimentRuns {
              edges {
                run: node {
                  id
                  traceId
                  output
                  startTime
                  endTime
                  error
                }
              }
            }
          }
        }
      }
    """
    response = await gql_client.execute(
        query=query,
        variables={"exampleId": str(GlobalID("DatasetExample", str(1)))},
    )
    assert not response.errors
    assert response.data == {
        "example": {
            "experimentRuns": {
                "edges": [
                    {
                        "run": {
                            "id": str(GlobalID("ExperimentRun", str(1))),
                            "traceId": None,
                            "output": "experiment-1-run-1-output",
                            "startTime": "2020-01-01T00:00:00+00:00",
                            "endTime": "2020-01-01T00:01:00+00:00",
                            "error": None,
                        }
                    },
                    {
                        "run": {
                            "id": str(GlobalID("ExperimentRun", str(2))),
                            "traceId": None,
                            "output": {"output": "experiment-2-run-1-output"},
                            "startTime": "2020-01-01T00:00:00+00:00",
                            "endTime": "2020-01-01T00:01:00+00:00",
                            "error": None,
                        }
                    },
                ]
            }
        }
    }

async def test_dataset_example_experiment_runs_resolver_filters_by_experiment_ids(
    gql_client: AsyncGraphQLClient,
    example_with_experiment_runs: Any,
) -> None:
    query = """
      query ($exampleId: ID!, $experimentIds: [ID!]) {
        example: node(id: $exampleId) {
          ... on DatasetExample {
            experimentRuns(experimentIds: $experimentIds) {
              edges {
                run: node {
                  id
                  traceId
                  output
                  startTime
                  endTime
                  error
                }
              }
            }
          }
        }
      }
    """
    response = await gql_client.execute(
        query=query,
        variables={
            "exampleId": str(GlobalID("DatasetExample", str(1))),
            "experimentIds": [str(GlobalID("Experiment", str(1)))],
        },
    )
    assert not response.errors
    assert response.data == {
        "example": {
            "experimentRuns": {
                "edges": [
                    {
                        "run": {
                            "id": str(GlobalID("ExperimentRun", str(1))),
                            "traceId": None,
                            "output": "experiment-1-run-1-output",
                            "startTime": "2020-01-01T00:00:00+00:00",
                            "endTime": "2020-01-01T00:01:00+00:00",
                            "error": None,
                        }
                    },
                ]
            }
        }
    }


# --- tests/unit/server/api/types/test_Experiment.py ---

async def test_runs_resolver_returns_runs_for_experiment_in_expected_order(
    gql_client: AsyncGraphQLClient,
    variables: dict[str, Any],
    expected_run_ids: list[int],
    expected_has_next_page: bool,
    expected_end_cursor: str,
    dataset_with_experiment_runs: Any,
    db: DbSessionFactory,
) -> None:
    query = """
      query ($experimentId: ID!, $first: Int, $after: String, $sort: ExperimentRunSort) {
        experiment: node(id: $experimentId) {
          ... on Experiment {
            runs(first: $first, after: $after, sort: $sort) {
              edges {
                run: node {
                  id
                }
              }
              pageInfo {
                hasNextPage
                endCursor
              }
            }
          }
        }
      }
    """
    response = await gql_client.execute(
        query=query,
        variables=variables,
    )
    assert not response.errors
    assert response.data
    actual_run_ids = [
        int(GlobalID.from_id(edge["run"]["id"]).node_id)
        for edge in response.data["experiment"]["runs"]["edges"]
    ]
    assert actual_run_ids == expected_run_ids
    page_info = response.data["experiment"]["runs"]["pageInfo"]
    assert page_info["hasNextPage"] == expected_has_next_page
    assert page_info["endCursor"] == expected_end_cursor

async def test_expected_run_count_resolver(
    gql_client: AsyncGraphQLClient,
    experiment_with_expected_run_count: int,
) -> None:
    query = """
      query ($experimentId: ID!) {
        experiment: node(id: $experimentId) {
          ... on Experiment {
            expectedRunCount
          }
        }
      }
    """
    response = await gql_client.execute(
        query=query,
        variables={
            "experimentId": str(
                GlobalID(type_name="Experiment", node_id=str(experiment_with_expected_run_count))
            ),
        },
    )
    assert not response.errors
    assert response.data == {"experiment": {"expectedRunCount": 6}}


# --- tests/unit/server/api/types/test_ExperimentRun.py ---

async def test_annotations_resolver_returns_annotations_for_run(
    gql_client: AsyncGraphQLClient,
    experiment_run_with_annotations: Any,
) -> None:
    query = """
      query ($runId: ID!) {
        run: node(id: $runId) {
          ... on ExperimentRun {
            annotations {
              edges {
                annotation: node {
                  id
                  name
                  annotatorKind
                  label
                  score
                  explanation
                  error
                  metadata
                  startTime
                  endTime
                }
              }
            }
          }
        }
      }
    """
    response = await gql_client.execute(
        query=query,
        variables={
            "runId": str(GlobalID(type_name="ExperimentRun", node_id=str(1))),
        },
    )
    assert not response.errors
    assert response.data == {
        "run": {
            "annotations": {
                "edges": [
                    {
                        "annotation": {
                            "id": str(
                                GlobalID(type_name="ExperimentRunAnnotation", node_id=str(2))
                            ),
                            "name": "annotation-2-name",
                            "annotatorKind": "LLM",
                            "label": "annotation-2-label",
                            "score": 0.2,
                            "explanation": "annotation-2-explanation",
                            "error": "annotation-2-error",
                            "metadata": {
                                "annotation-2-metadata-key": "annotation-2-metadata-value"
                            },
                            "startTime": "2020-01-01T00:00:00+00:00",
                            "endTime": "2020-01-01T00:01:00+00:00",
                        }
                    },
                    {
                        "annotation": {
                            "id": str(
                                GlobalID(type_name="ExperimentRunAnnotation", node_id=str(1))
                            ),
                            "name": "annotation-1-name",
                            "annotatorKind": "LLM",
                            "label": "annotation-1-label",
                            "score": 0.2,
                            "explanation": "annotation-1-explanation",
                            "error": "annotation-1-error",
                            "metadata": {
                                "annotation-1-metadata-key": "annotation-1-metadata-value"
                            },
                            "startTime": "2020-01-01T00:00:00+00:00",
                            "endTime": "2020-01-01T00:01:00+00:00",
                        }
                    },
                ]
            }
        }
    }


# --- tests/unit/server/api/types/test_Project.py ---

async def test_paginate_spans_by_trace_start_time(
    db: DbSessionFactory,
    gql_client: AsyncGraphQLClient,
) -> None:
    """Test the _paginate_span_by_trace_start_time optimization function.

    This function is triggered when:
    - rootSpansOnly: true
    - No filter_condition
    - sort.col is SpanColumn.startTime

    Key behaviors tested:
    - Returns one representative span per trace (not all spans)
    - Orders by trace start time (not span start time)
    - Uses cursors based on trace rowids + start times (unusual!)
    - Handles orphan spans based on orphan_span_as_root_span parameter
    - Supports time range filtering on trace start times
    - May return empty edges while has_next_page=True when traces have no matching spans
    - **RETRY LOGIC**: When insufficient edges are found (len(edges) < first) but has_next_page=True,
      the function automatically retries pagination with larger batch sizes (max(first, 1000))
      up to 10 times (retries=10) to collect enough spans. This handles cases where many traces
      exist but lack matching root spans.

    Implementation Details:
    - Uses CTEs (Common Table Expressions) for efficient trace-based pagination
    - PostgreSQL: Uses DISTINCT ON for deduplication
    - SQLite: Uses Python groupby() for deduplication (too complex for SQLite DISTINCT)
    - SQL ordering: trace start_time -> trace id -> span start_time (ASC for earliest) -> span id (DESC)
    - Cursors contain trace rowid + trace start_time, NOT span data
    - Over-fetches by 1 trace to determine has_next_page efficiently

    Test Data Setup:
    ================
    Creates 5 traces with start times at hours 1, 2, 3, 4, 5:

    Trace Index | Hour | Real Root Span | Orphan Span  | Additional Spans | Expected Name
    ------------|------|----------------|--------------|------------------|---------------
    0 (even)    |  1   |      ✓         |      ✗       | +2nd root span   | root-span-1
    1 (odd)     |  2   |      ✗         |      ✓       | +2nd orphan span | orphan-span-2
    2 (even)    |  3   |      ✓         |      ✗       | +2nd root span   | root-span-3
    3 (odd)     |  4   |      ✗         |      ✓       | +2nd orphan span | orphan-span-4
    4 (even)    |  5   |      ✓         |      ✗       | +2nd root span   | root-span-5

    Key Testing Points:
    - ALL traces have multiple candidate spans to test "earliest span per trace" selection
    - Trace 1: 2 root spans → Returns earliest (root-span-1, not second-root-span-1)
    - Trace 2: 2 orphan spans → Returns earliest (orphan-span-2, not second-orphan-span-2)
    - Trace 3: 2 root spans → Returns earliest (root-span-3, not second-root-span-3)
    - Trace 4: 2 orphan spans → Returns earliest (orphan-span-4, not second-orphan-span-4)
    - Trace 5: 2 root spans → Returns earliest (root-span-5, not second-root-span-5)
    - Comprehensive test of SQL ordering: ORDER BY span.start_time ASC, span.id DESC

    With orphan_span_as_root_span=false: Only returns real root spans 1, 3, 5 (3 total)
    With orphan_span_as_root_span=true:  Returns all spans 1, 2, 3, 4, 5 (5 total)
    """
    # ========================================
    # SETUP: Create test data
    # ========================================
    async with db() as session:
        project = models.Project(name=token_hex(8))
        session.add(project)
        await session.flush()

        # Create 5 traces with start times at hours 1, 2, 3, 4, 5
        base_time = datetime.fromisoformat("2024-01-01T00:00:00+00:00")
        traces = []
        spans = []

        for i in range(5):
            # Trace start times: 01:00, 02:00, 03:00, 04:00, 05:00
            trace = models.Trace(
                trace_id=token_hex(16),
                project_rowid=project.id,
                start_time=base_time + timedelta(hours=i + 1),
                end_time=base_time + timedelta(hours=i + 2),
            )
            session.add(trace)
            await session.flush()
            traces.append(trace)

            if i % 2 == 0:
                # EVEN indices (0, 2, 4) → traces at hours 1, 3, 5 → CREATE REAL ROOT SPANS
                # These spans have parent_id=None (true root spans)
                root_span = models.Span(
                    trace_rowid=trace.id,
                    span_id=token_hex(8),
                    parent_id=None,  # ← This makes it a real root span
                    name=f"root-span-{i + 1}",
                    span_kind="CHAIN",
                    start_time=trace.start_time + timedelta(minutes=10),
                    end_time=trace.start_time + timedelta(minutes=20),
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_message="",
                    cumulative_error_count=0,
                    cumulative_llm_token_count_prompt=0,
                    cumulative_llm_token_count_completion=0,
                )
                session.add(root_span)
                spans.append(root_span)

                # Also create a child span to verify only root span is returned per trace
                child_span = models.Span(
                    trace_rowid=trace.id,
                    span_id=token_hex(8),
                    parent_id=root_span.span_id,  # ← Child of the root span
                    name=f"child-span-{i + 1}",
                    span_kind="CHAIN",
                    start_time=trace.start_time + timedelta(minutes=15),
                    end_time=trace.start_time + timedelta(minutes=25),
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_message="",
                    cumulative_error_count=0,
                    cumulative_llm_token_count_prompt=0,
                    cumulative_llm_token_count_completion=0,
                )
                session.add(child_span)

                # Add a SECOND root span with later start time to test "earliest span" selection
                # This span should NOT be returned (only the earliest root span per trace)
                second_root_span = models.Span(
                    trace_rowid=trace.id,
                    span_id=token_hex(8),
                    parent_id=None,  # ← Also a root span
                    name=f"second-root-span-{i + 1}",
                    span_kind="CHAIN",
                    start_time=trace.start_time + timedelta(minutes=30),  # ← Later start time
                    end_time=trace.start_time + timedelta(minutes=40),
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_message="",
                    cumulative_error_count=0,
                    cumulative_llm_token_count_prompt=0,
                    cumulative_llm_token_count_completion=0,
                )
                session.add(second_root_span)
            else:
                # ODD indices (1, 3) → traces at hours 2, 4 → CREATE ORPHAN SPANS
                # These spans have parent_id pointing to non-existent spans (orphans)
                orphan_span = models.Span(
                    trace_rowid=trace.id,
                    span_id=token_hex(8),
                    parent_id=token_hex(8),  # ← Points to non-existent span (orphan)
                    name=f"orphan-span-{i + 1}",
                    span_kind="CHAIN",
                    start_time=trace.start_time + timedelta(minutes=10),
                    end_time=trace.start_time + timedelta(minutes=20),
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_message="",
                    cumulative_error_count=0,
                    cumulative_llm_token_count_prompt=0,
                    cumulative_llm_token_count_completion=0,
                )
                session.add(orphan_span)
                spans.append(orphan_span)

                # Add a SECOND orphan span with later start time to test "earliest span" selection
                # This span should NOT be returned (only the earliest orphan span per trace)
                second_orphan_span = models.Span(
                    trace_rowid=trace.id,
                    span_id=token_hex(8),
                    parent_id=token_hex(8),  # ← Also an orphan span (different parent_id)
                    name=f"second-orphan-span-{i + 1}",
                    span_kind="CHAIN",
                    start_time=trace.start_time + timedelta(minutes=30),  # ← Later start time
                    end_time=trace.start_time + timedelta(minutes=40),
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_message="",
                    cumulative_error_count=0,
                    cumulative_llm_token_count_prompt=0,
                    cumulative_llm_token_count_completion=0,
                )
                session.add(second_orphan_span)

        project_gid = str(GlobalID(type_name="Project", node_id=str(project.id)))

    # ========================================
    # TEST 1: Basic pagination with orphan_span_as_root_span=false
    # Expected: Only real root spans (1, 3, 5) returned, NOT orphan spans (2, 4)
    # ========================================
    query = """
        query ($projectId: ID!, $first: Int!, $after: String) {
            node(id: $projectId) {
                ... on Project {
                    spans(
                        rootSpansOnly: true,
                        orphanSpanAsRootSpan: false,  # ← Exclude orphan spans
                        sort: {col: startTime, dir: desc},
                        first: $first,
                        after: $after
                    ) {
                        edges {
                            node {
                                id
                                name
                            }
                            cursor
                        }
                        pageInfo {
                            hasNextPage
                            hasPreviousPage
                            startCursor
                            endCursor
                        }
                    }
                }
            }
        }
    """

    # Page 1: Request first 2 spans in descending order (by trace start time)
    # Expected: Only root-span-5 (trace 5 is latest, and only that trace has a real root span)
    # Note: trace 4 has an orphan span, but it's excluded by orphanSpanAsRootSpan=false
    response = await gql_client.execute(
        query=query,
        variables={
            "projectId": project_gid,
            "first": 2,
        },
    )

    assert not response.errors
    assert (data := response.data) is not None

    page = data["node"]["spans"]
    edges = page["edges"]
    page_info = page["pageInfo"]

    assert len(edges) == 2
    assert edges[0]["node"]["name"] == "root-span-5"
    assert edges[1]["node"]["name"] == "root-span-3"
    assert page_info["hasNextPage"] is True  # More traces to check
    assert page_info["hasPreviousPage"] is False

    # Verify cursor contains trace rowid (5) and trace start time (05:00:00)
    # This demonstrates the unusual "trace-based cursors" behavior
    assert (
        base64.b64decode(page_info["startCursor"].encode())
        == b"5:DATETIME:2024-01-01T05:00:00+00:00"
    )
    assert (
        base64.b64decode(page_info["endCursor"].encode()) == b"3:DATETIME:2024-01-01T03:00:00+00:00"
    )

    # Page 2: Continue pagination after trace 5
    # Expected: root-span-3 (trace 3 is next latest with real root span)
    # Note: trace 4 is skipped because it only has orphan span (excluded)
    response = await gql_client.execute(
        query=query,
        variables={
            "projectId": project_gid,
            "first": 3,
            "after": base64.b64encode(b"5:DATETIME:2024-01-01T05:00:00+00:00").decode(),
        },
    )

    assert not response.errors
    assert (data := response.data) is not None

    page = data["node"]["spans"]
    edges = page["edges"]
    page_info = page["pageInfo"]

    assert len(edges) == 2
    assert edges[0]["node"]["name"] == "root-span-3"
    assert edges[1]["node"]["name"] == "root-span-1"
    assert page_info["hasNextPage"] is False
    assert page_info["hasPreviousPage"] is False
    assert (
        base64.b64decode(page_info["startCursor"].encode())
        == b"4:DATETIME:2024-01-01T04:00:00+00:00"  # Trace 3 yielded the span
    )
    assert (
        base64.b64decode(page_info["endCursor"].encode()) == b"1:DATETIME:2024-01-01T01:00:00+00:00"
    )

    # Page 3: Continue pagination after trace 3
    # Expected: root-span-1 (trace 1 is oldest with real root span)
    # Note: trace 2 is skipped because it only has orphan span (excluded)
    response = await gql_client.execute(
        query=query,
        variables={
            "projectId": project_gid,
            "first": 4,
            "after": base64.b64encode(b"3:DATETIME:2024-01-01T03:00:00+00:00").decode(),
        },
    )

    assert not response.errors
    assert (data := response.data) is not None

    page = data["node"]["spans"]
    edges = page["edges"]
    page_info = page["pageInfo"]

    # Should return root-span-1 (oldest real root span)
    assert len(edges) == 1
    assert edges[0]["node"]["name"] == "root-span-1"
    assert page_info["hasNextPage"] is False  # No more traces
    assert page_info["hasPreviousPage"] is False
    assert (
        base64.b64decode(page_info["startCursor"].encode())
        == b"2:DATETIME:2024-01-01T02:00:00+00:00"
    )
    assert (
        base64.b64decode(page_info["endCursor"].encode()) == b"1:DATETIME:2024-01-01T01:00:00+00:00"
    )

    # ========================================
    # TEST 2: Ascending order (orphan_span_as_root_span=false)
    # Expected: Same spans but in reverse order: root-span-1, root-span-3, root-span-5
    # ========================================
    response = await gql_client.execute(
        query=query.replace("dir: desc", "dir: asc"),
        variables={
            "projectId": project_gid,
            "first": 2,
        },
    )

    assert not response.errors
    assert (data := response.data) is not None

    asc_page = data["node"]["spans"]
    edges = asc_page["edges"]
    page_info = asc_page["pageInfo"]

    # Should return first span in ascending order (oldest trace with real root span)
    assert len(edges) == 2
    assert edges[0]["node"]["name"] == "root-span-1"
    assert edges[1]["node"]["name"] == "root-span-3"
    assert page_info["hasNextPage"] is True

    # ========================================
    # TEST 3: Bulk query (orphan_span_as_root_span=false)
    # Expected: All 3 real root spans at once
    # ========================================
    response = await gql_client.execute(
        query=query,
        variables={
            "projectId": project_gid,
            "first": 10,
        },
    )

    assert not response.errors
    assert (data := response.data) is not None

    all_spans = data["node"]["spans"]
    edges = all_spans["edges"]
    page_info = all_spans["pageInfo"]
    span_names = [edge["node"]["name"] for edge in edges]

    # Should return all 3 real root spans (excluding orphan spans 2, 4)
    # IMPORTANT: Returns earliest root span per trace (ALL traces have multiple candidates):
    # - Trace 1: root-span-1 (NOT second-root-span-1 which has later start time)
    # - Trace 3: root-span-3 (NOT second-root-span-3 which has later start time)
    # - Trace 5: root-span-5 (NOT second-root-span-5 which has later start time)
    assert len(edges) == 3
    assert span_names == [
        "root-span-5",
        "root-span-3",
        "root-span-1",
    ]
    assert page_info["hasNextPage"] is False

    # ========================================
    # TEST 4: Time range filtering (orphan_span_as_root_span=false)
    # Filter: hours 2-4 (includes traces 2, 3, 4)
    # Expected: Only root-span-3 (trace 3 has real root span, traces 2&4 have orphans)
    # ========================================
    time_range_query = """
        query ($projectId: ID!, $first: Int!, $timeRange: TimeRange) {
            node(id: $projectId) {
                ... on Project {
                    spans(
                        rootSpansOnly: true,
                        orphanSpanAsRootSpan: false,  # ← Exclude orphan spans
                        sort: {col: startTime, dir: desc},
                        first: $first,
                        timeRange: $timeRange
                    ) {
                        edges {
                            node {
                                id
                                name
                            }
                        }
                        pageInfo {
                            hasNextPage
                        }
                    }
                }
            }
        }
    """

    response = await gql_client.execute(
        query=time_range_query,
        variables={
            "projectId": project_gid,
            "first": 10,
            "timeRange": {
                "start": (base_time + timedelta(hours=2)).isoformat(),  # 02:00:00
                "end": (base_time + timedelta(hours=4)).isoformat(),  # 04:00:00
            },
        },
    )

    assert not response.errors
    assert (data := response.data) is not None

    filtered_spans = data["node"]["spans"]
    edges = filtered_spans["edges"]

    # Time range includes traces 2, 3, 4:
    # - Trace 2 (hour 2): has orphan span → excluded by orphanSpanAsRootSpan=false
    # - Trace 3 (hour 3): has real root span → included
    # - Trace 4 (hour 4): has orphan span → excluded by orphanSpanAsRootSpan=false
    assert len(edges) == 1
    assert edges[0]["node"]["name"] == "root-span-3"

    # ========================================
    # TEST 5: Include orphan spans (orphanSpanAsRootSpan=true)
    # Expected: All 5 spans returned (3 real roots + 2 orphans)
    # ========================================
    orphan_query = """
        query ($projectId: ID!, $first: Int!, $after: String) {
            node(id: $projectId) {
                ... on Project {
                    spans(
                        rootSpansOnly: true,
                        orphanSpanAsRootSpan: true,
                        sort: {col: startTime, dir: desc},
                        first: $first,
                        after: $after
                    ) {
                        edges {
                            node {
                                id
                                name
                            }
                            cursor
                        }
                        pageInfo {
                            hasNextPage
                            hasPreviousPage
                            startCursor
                            endCursor
                        }
                    }
                }
            }
        }
    """

    # Test 5a: Basic pagination with orphans included
    # Expected: Now returns 2 spans per page instead of 1 (includes orphan spans)
    response = await gql_client.execute(
        query=orphan_query,
        variables={
            "projectId": project_gid,
            "first": 2,
        },
    )

    assert not response.errors
    assert (data := response.data) is not None

    page = data["node"]["spans"]
    edges = page["edges"]
    page_info = page["pageInfo"]

    # Should return 2 spans: both real root and orphan spans
    assert len(edges) == 2
    assert edges[0]["node"]["name"] == "root-span-5"  # Real root span from trace 5 (latest)
    assert edges[1]["node"]["name"] == "orphan-span-4"  # Orphan span from trace 4 (2nd latest)
    assert page_info["hasNextPage"] is True

    # Test 5b: Bulk query with orphans included
    # Expected: All 5 spans (3 real + 2 orphan) vs 3 spans when orphans excluded
    response = await gql_client.execute(
        query=orphan_query,
        variables={
            "projectId": project_gid,
            "first": 10,
        },
    )

    assert not response.errors
    assert (data := response.data) is not None

    all_spans = data["node"]["spans"]
    edges = all_spans["edges"]
    page_info = all_spans["pageInfo"]
    span_names = [edge["node"]["name"] for edge in edges]

    # Should return ALL 5 spans (3 real root spans + 2 orphan spans) in descending order
    # IMPORTANT: Returns earliest span per trace (ALL traces have multiple candidates):
    # - Trace 1: root-span-1 (NOT second-root-span-1)
    # - Trace 2: orphan-span-2 (NOT second-orphan-span-2)
    # - Trace 3: root-span-3 (NOT second-root-span-3)
    # - Trace 4: orphan-span-4 (NOT second-orphan-span-4)
    # - Trace 5: root-span-5 (NOT second-root-span-5)
    assert len(edges) == 5
    assert span_names == [
        "root-span-5",
        "orphan-span-4",
        "root-span-3",
        "orphan-span-2",
        "root-span-1",
    ]
    assert page_info["hasNextPage"] is False

    # Test 5c: Ascending order with orphans included
    # Expected: Same 5 spans but in reverse order
    response = await gql_client.execute(
        query=orphan_query.replace("dir: desc", "dir: asc"),
        variables={"projectId": project_gid, "first": 3},
    )

    assert not response.errors
    assert (data := response.data) is not None

    asc_page = data["node"]["spans"]
    edges = asc_page["edges"]
    span_names = [edge["node"]["name"] for edge in edges]

    # Should return first 3 spans in ascending order (includes orphan span 2)
    assert len(edges) == 3
    assert span_names == [
        "root-span-1",
        "orphan-span-2",
        "root-span-3",
    ]

    # Test 5d: Time range filtering with orphans included
    orphan_time_range_query = """
        query ($projectId: ID!, $first: Int!, $timeRange: TimeRange) {
            node(id: $projectId) {
                ... on Project {
                    spans(
                        rootSpansOnly: true,
                        orphanSpanAsRootSpan: true,
                        sort: {col: startTime, dir: desc},
                        first: $first,
                        timeRange: $timeRange
                    ) {
                        edges {
                            node {
                                id
                                name
                            }
                        }
                        pageInfo {
                            hasNextPage
                        }
                    }
                }
            }
        }
    """

    # Expected: Now returns 2 spans (includes orphan span 2) vs 1 span when orphans excluded
    response = await gql_client.execute(
        query=orphan_time_range_query,
        variables={
            "projectId": project_gid,
            "first": 10,
            "timeRange": {
                "start": (base_time + timedelta(hours=2)).isoformat(),  # 02:00:00
                "end": (base_time + timedelta(hours=4)).isoformat(),  # 04:00:00
            },
        },
    )

    assert not response.errors
    assert (data := response.data) is not None

    filtered_spans = data["node"]["spans"]
    edges = filtered_spans["edges"]
    span_names = [edge["node"]["name"] for edge in edges]

    # Time range includes traces 2, 3, 4 - with orphans included:
    # - Trace 2 (hour 2): has orphan span → NOW INCLUDED
    # - Trace 3 (hour 3): has real root span → included
    # - Trace 4 (hour 4): has orphan span → NOW INCLUDED
    # But trace 4 is excluded by time range end=04:00:00 (exclusive), so only traces 2 & 3
    assert len(edges) == 2
    assert span_names == [
        "root-span-3",
        "orphan-span-2",
    ]  # Descending order: trace 3, then trace 2

    async def test_top_models_comprehensive(
        self,
        _cost_data: _CostTestData,
        gql_client: AsyncGraphQLClient,
    ) -> None:
        """Comprehensive test for both top_models_by_token_count and top_models_by_cost fields."""
        project = _cost_data.project
        base_time = _cost_data.base_time
        project_gid = str(GlobalID(type_name="Project", node_id=str(project.id)))

        # Full time range for comprehensive testing
        full_time_range = {
            "start": base_time.isoformat(),
            "end": (base_time + timedelta(days=1)).isoformat(),
        }

        # --- TEST 1: Basic ordering for both token count and cost ---

        token_query = """
            query ($projectId: ID!, $timeRange: TimeRange!) {
                node(id: $projectId) {
                    ... on Project {
                        topModelsByTokenCount(timeRange: $timeRange) {
                            name
                            costSummary(projectId: $projectId, timeRange: $timeRange) {
                                total { tokens cost }
                                prompt { tokens cost }
                                completion { tokens cost }
                            }
                        }
                    }
                }
            }
        """

        cost_query = """
            query ($projectId: ID!, $timeRange: TimeRange!) {
                node(id: $projectId) {
                    ... on Project {
                        topModelsByCost(timeRange: $timeRange) {
                            name
                            costSummary(projectId: $projectId, timeRange: $timeRange) {
                                total { tokens cost }
                                prompt { tokens cost }
                                completion { tokens cost }
                            }
                        }
                    }
                }
            }
        """

        # Test token-based ordering
        token_response = await gql_client.execute(
            query=token_query,
            variables={"projectId": project_gid, "timeRange": full_time_range},
        )
        assert not token_response.errors
        assert (token_data := token_response.data) is not None

        token_models = token_data["node"]["topModelsByTokenCount"]
        assert len(token_models) == 3

        # Expected token totals: gpt-3.5-turbo (11K), gpt-4 (3.6K), claude (3.3K)
        assert token_models[0]["name"] == "gpt-3.5-turbo"
        assert token_models[1]["name"] == "gpt-4"
        assert token_models[2]["name"] == "claude-3-sonnet"

        token_counts = [m["costSummary"]["total"]["tokens"] for m in token_models]
        assert token_counts == sorted(token_counts, reverse=True)
        assert token_counts[0] == 11000  # gpt-3.5-turbo
        assert token_counts[1] == 3600  # gpt-4
        assert token_counts[2] == 3300  # claude

        # Test cost-based ordering
        cost_response = await gql_client.execute(
            query=cost_query,
            variables={"projectId": project_gid, "timeRange": full_time_range},
        )
        assert not cost_response.errors
        assert (cost_data := cost_response.data) is not None

        cost_models = cost_data["node"]["topModelsByCost"]
        assert len(cost_models) == 3

        # Expected cost totals: gpt-4 ($5.40), gpt-3.5-turbo ($2.80), claude ($2.50)
        assert cost_models[0]["name"] == "gpt-4"
        assert cost_models[1]["name"] == "gpt-3.5-turbo"
        assert cost_models[2]["name"] == "claude-3-sonnet"

        costs = [m["costSummary"]["total"]["cost"] for m in cost_models]
        assert costs == sorted(costs, reverse=True)
        assert abs(costs[0] - 5.40) < 0.01  # gpt-4
        assert abs(costs[1] - 2.80) < 0.01  # gpt-3.5-turbo
        assert abs(costs[2] - 2.50) < 0.01  # claude

        # Verify ordering is different between token and cost
        token_order = [m["name"] for m in token_models]
        cost_order = [m["name"] for m in cost_models]
        assert token_order != cost_order

        # --- TEST 2: Cost summary calculations accuracy ---

        # Find gpt-4 model and verify its calculations
        gpt4_model = next(m for m in cost_models if m["name"] == "gpt-4")
        cost_summary = gpt4_model["costSummary"]

        # Verify totals match sum of prompt + completion
        total_cost = cost_summary["total"]["cost"]
        prompt_cost = cost_summary["prompt"]["cost"]
        completion_cost = cost_summary["completion"]["cost"]
        assert abs(total_cost - (prompt_cost + completion_cost)) < 0.01

        total_tokens = cost_summary["total"]["tokens"]
        prompt_tokens = cost_summary["prompt"]["tokens"]
        completion_tokens = cost_summary["completion"]["tokens"]
        assert abs(total_tokens - (prompt_tokens + completion_tokens)) < 1

        # --- TEST 3: Time range filtering ---

        # Test filtering to only hour 2-3 (gpt-3.5-turbo data only)
        filtered_time_range = {
            "start": (base_time + timedelta(hours=2)).isoformat(),
            "end": (base_time + timedelta(hours=4)).isoformat(),
        }

        token_filtered_response = await gql_client.execute(
            query=token_query,
            variables={"projectId": project_gid, "timeRange": filtered_time_range},
        )
        assert not token_filtered_response.errors
        assert (token_filtered_data := token_filtered_response.data) is not None

        filtered_models = token_filtered_data["node"]["topModelsByTokenCount"]
        assert len(filtered_models) == 1
        assert filtered_models[0]["name"] == "gpt-3.5-turbo"
        assert filtered_models[0]["costSummary"]["total"]["tokens"] == 11000

        # Test filtering to only hour 0-1 (gpt-4 data only)
        gpt4_time_range = {
            "start": base_time.isoformat(),
            "end": (base_time + timedelta(hours=2)).isoformat(),
        }

        cost_filtered_response = await gql_client.execute(
            query=cost_query,
            variables={"projectId": project_gid, "timeRange": gpt4_time_range},
        )
        assert not cost_filtered_response.errors
        assert (cost_filtered_data := cost_filtered_response.data) is not None

        gpt4_only_models = cost_filtered_data["node"]["topModelsByCost"]
        assert len(gpt4_only_models) == 1
        assert gpt4_only_models[0]["name"] == "gpt-4"
        assert abs(gpt4_only_models[0]["costSummary"]["total"]["cost"] - 5.40) < 0.01

        # --- TEST 4: Partial time ranges ---

        # Query that includes gpt-4 (hours 0-1) and gpt-3.5 (hours 2-3) but excludes claude (hours 4-5)
        partial_time_range = {
            "start": base_time.isoformat(),
            "end": (base_time + timedelta(hours=4)).isoformat(),
        }

        partial_response = await gql_client.execute(
            query=cost_query,
            variables={"projectId": project_gid, "timeRange": partial_time_range},
        )
        assert not partial_response.errors
        assert (partial_data := partial_response.data) is not None

        partial_models = partial_data["node"]["topModelsByCost"]
        assert len(partial_models) == 2  # Should include gpt-4 and gpt-3.5, exclude claude

        model_names = [m["name"] for m in partial_models]
        assert "gpt-4" in model_names
        assert "gpt-3.5-turbo" in model_names
        assert "claude-3-sonnet" not in model_names

        # gpt-4 should still rank higher by cost than gpt-3.5-turbo
        assert partial_models[0]["name"] == "gpt-4"
        assert partial_models[1]["name"] == "gpt-3.5-turbo"

        # --- TEST 5: No data scenarios ---

        empty_time_range = {
            "start": "2023-01-01T00:00:00+00:00",
            "end": "2024-01-01T00:00:00+00:00",
        }

        # Test token count with no data
        empty_token_response = await gql_client.execute(
            query=token_query,
            variables={"projectId": project_gid, "timeRange": empty_time_range},
        )
        assert not empty_token_response.errors
        assert (empty_token_data := empty_token_response.data) is not None
        assert len(empty_token_data["node"]["topModelsByTokenCount"]) == 0

        # Test cost with no data
        empty_cost_response = await gql_client.execute(
            query=cost_query,
            variables={"projectId": project_gid, "timeRange": empty_time_range},
        )
        assert not empty_cost_response.errors
        assert (empty_cost_data := empty_cost_response.data) is not None
        assert len(empty_cost_data["node"]["topModelsByCost"]) == 0

    async def test_sessions_sort_cost_total(
        self,
        db: DbSessionFactory,
        httpx_client: httpx.AsyncClient,
    ) -> None:
        """Test sorting project sessions by total cost.

        Note: Sessions without cost data are filtered out (inner join behavior).
        """
        async with db() as session:
            project = await _add_project(session, name="cost-sort-test")
            model = await _add_generative_model(session, name="gpt-4", provider="openai")

            # Create 5 sessions: 4 with costs, 1 without
            sessions = []
            costs = [100.0, 50.0, 200.0, 75.0, None]

            for i, cost in enumerate(costs):
                ps = await _add_project_session(session, project)
                sessions.append(ps)
                trace = await _add_trace(session, project, ps)
                span = await _add_span(session, trace)

                # Only add span cost if cost is not None
                if cost is not None:
                    await _add_span_cost(
                        session,
                        span=span,
                        trace=trace,
                        model=model,
                        total_cost=cost,
                        total_tokens=1000,
                        prompt_cost=cost * 0.75,
                        prompt_tokens=800,
                        completion_cost=cost * 0.25,
                        completion_tokens=200,
                        span_start_time=span.start_time,
                    )

        column = "costTotal"
        # Expected order desc: 200, 100, 75, 50 (session without cost is filtered out)
        result_desc = [
            _gid(sessions[2]),  # 200.0
            _gid(sessions[0]),  # 100.0
            _gid(sessions[3]),  # 75.0
            _gid(sessions[1]),  # 50.0
        ]

        # Test descending order
        field = f"sessions(sort:{{col:{column},dir:desc}}){{edges{{node{{id}}}}}}"
        res = await self._node(field, project, httpx_client)
        assert [e["node"]["id"] for e in res["edges"]] == result_desc

        # Test ascending order: 50, 75, 100, 200
        field = f"sessions(sort:{{col:{column},dir:asc}}){{edges{{node{{id}}}}}}"
        res = await self._node(field, project, httpx_client)
        assert [e["node"]["id"] for e in res["edges"]] == result_desc[::-1]

        # Test pagination
        first = 2
        field = f"sessions(sort:{{col:{column},dir:desc}},first:{first}){{edges{{node{{id}}}}pageInfo{{hasNextPage}}}}"
        res = await self._node(field, project, httpx_client)
        assert [e["node"]["id"] for e in res["edges"]] == result_desc[:2]
        assert res["pageInfo"]["hasNextPage"] is True


# --- tests/unit/server/api/types/test_Span.py ---

async def test_span_fields(
    gql_client: AsyncGraphQLClient,
    _span_data: tuple[models.Project, Mapping[int, models.Trace], Mapping[int, models.Span]],
) -> None:
    query = """
      query SpanBySpanNodeId($id: ID!) {
        node(id: $id) {
          ... on Span {
            ...SpanFragment
          }
        }
      }
      query SpansByTraceNodeId($traceId: ID!) {
        node(id: $traceId) {
          ... on Trace {
            spans(first: 1000) {
              edges {
                node {
                  ...SpanFragment
                }
              }
            }
          }
        }
      }
      query SpansByProjectNodeId($projectId: ID!) {
        node(id: $projectId) {
          ... on Project {
            spans(first: 1000) {
              edges {
                node {
                  ...SpanFragment
                }
              }
            }
          }
        }
      }
      fragment SpanFragment on Span {
        id
        name
        statusCode
        statusMessage
        startTime
        endTime
        latencyMs
        parentId
        spanKind
        context {
          spanId
          traceId
        }
        trace {
          id
          numSpans
        }
        attributes
        tokenCountTotal
        tokenCountPrompt
        tokenCountCompletion
        cumulativeTokenCountTotal
        cumulativeTokenCountPrompt
        cumulativeTokenCountCompletion
        propagatedStatusCode
        input {
          mimeType
          value
        }
        output {
          mimeType
          value
        }
        events {
          name
          message
          timestamp
        }
        metadata
        numDocuments
        numChildSpans
        descendants(maxDepth: 3) {
          edges {
            node {
              id
            }
          }
        }
      }
    """
    db_project, db_traces, db_spans = _span_data
    db_num_spans_per_trace = _get_num_spans_per_trace(db_spans)
    db_descendent_rowids = _get_descendant_rowids(db_spans, 3)
    db_num_child_spans = _get_num_child_spans(db_spans)
    project_id = str(GlobalID(Project.__name__, str(db_project.id)))
    response = await gql_client.execute(
        query=query,
        variables={"projectId": project_id},
        operation_name="SpansByProjectNodeId",
    )
    assert not response.errors
    assert (data := response.data) is not None
    spans = [e["node"] for e in data["node"]["spans"]["edges"]]
    assert len(spans) == len(db_spans)
    for db_trace in db_traces.values():
        trace_gid = str(GlobalID(Trace.__name__, str(db_trace.id)))
        response = await gql_client.execute(
            query=query,
            variables={"traceId": trace_gid},
            operation_name="SpansByTraceNodeId",
        )
        assert not response.errors
        assert (data := response.data) is not None
        spans.extend(e["node"] for e in data["node"]["spans"]["edges"])
    assert len(spans) == len(db_spans) * 2
    for db_span in db_spans.values():
        id_ = str(GlobalID(Span.__name__, str(db_span.id)))
        response = await gql_client.execute(
            query=query,
            variables={"id": id_},
            operation_name="SpanBySpanNodeId",
        )
        assert not response.errors
        assert (data := response.data) is not None
        spans.append(data["node"])
    assert len(spans) == len(db_spans) * 3
    for span in spans:
        span_rowid = from_global_id_with_expected_type(GlobalID.from_id(span["id"]), Span.__name__)
        db_span = db_spans[span_rowid]
        assert span["id"] == str(GlobalID(Span.__name__, str(db_span.id)))
        assert span["name"] == db_span.name
        assert span["statusCode"] == db_span.status_code
        assert span["statusMessage"] == db_span.status_message
        assert span["startTime"] == db_span.start_time.isoformat()
        assert span["endTime"] == db_span.end_time.isoformat()
        assert span["parentId"] == db_span.parent_id
        assert span["spanKind"] == db_span.span_kind.lower()
        assert span["context"]["spanId"] == db_span.span_id
        assert span["context"]["traceId"] == db_traces[db_span.trace_rowid].trace_id
        assert isinstance(span["attributes"], str) and span["attributes"]
        assert json.loads(span["attributes"]) == db_span.attributes
        assert span["tokenCountPrompt"] == db_span.llm_token_count_prompt
        assert span["tokenCountCompletion"] == db_span.llm_token_count_completion
        assert span["tokenCountTotal"] == (db_span.llm_token_count_completion or 0) + (
            db_span.llm_token_count_prompt or 0
        )
        assert span["cumulativeTokenCountPrompt"] == (
            db_span.cumulative_llm_token_count_prompt or 0
        )
        assert span["cumulativeTokenCountCompletion"] == (
            db_span.cumulative_llm_token_count_completion or 0
        )
        assert span["cumulativeTokenCountTotal"] == (
            db_span.cumulative_llm_token_count_completion or 0
        ) + (db_span.cumulative_llm_token_count_prompt or 0)
        assert span["propagatedStatusCode"] == "ERROR" if db_span.cumulative_error_count else "OK"
        if db_span.input_value:
            assert span["input"]["value"] == db_span.input_value
            if db_span.input_mime_type:
                assert span["input"]["mimeType"] in db_span.input_mime_type
            else:
                assert span["input"]["mimeType"] == "text"
        else:
            assert not span["input"]
        if db_span.output_value:
            assert span["output"]["value"] == db_span.output_value
            if db_span.output_mime_type:
                assert span["output"]["mimeType"] in db_span.output_mime_type
            else:
                assert span["output"]["mimeType"] == "text"
        else:
            assert not span["output"]
        if db_span.events:
            for event, db_event in zip(span["events"], db_span.events):
                assert event["name"] == db_event["name"]
                assert event["timestamp"] == db_event["timestamp"].isoformat()
        else:
            assert not span["events"]
        if db_span.metadata_:
            assert isinstance(span["metadata"], str) and span["metadata"]
            assert json.loads(span["metadata"]) == db_span.metadata_
        else:
            assert not span["metadata"]
        assert span["numDocuments"] == db_span.num_documents
        if num_child_spans := db_num_child_spans.get(span_rowid):
            assert span["numChildSpans"] == num_child_spans
        else:
            assert not span["numChildSpans"]
        if descendants := db_descendent_rowids.get(db_span.id):
            assert {e["node"]["id"] for e in span["descendants"]["edges"]} == {
                str(GlobalID(Span.__name__, str(id_))) for id_ in descendants
            }
        else:
            assert not span["descendants"]["edges"]
        assert span["trace"]["id"] == str(GlobalID(Trace.__name__, str(db_span.trace_rowid)))
        assert span["trace"]["numSpans"] == db_num_spans_per_trace[db_span.trace_rowid]


# --- tests/unit/server/cli/commands/test_serve.py ---

async def test_create_db_session_factory_routes_reads_to_replica_for_postgres(
    dialect: str,
    postgresql_primary_and_replica_urls: PostgresPrimaryAndReplicaUrls,
) -> None:
    factory, shutdown_callbacks = _create_db_session_factory(
        db_connection_str=postgresql_primary_and_replica_urls.primary_url,
        read_replica_connection_str=postgresql_primary_and_replica_urls.replica_url,
        migrate=False,
        log_to_stdout=False,
        log_migrations=False,
    )
    try:
        async with factory() as session:
            assert (
                str(await session.scalar(text("SELECT current_database()")))
                == postgresql_primary_and_replica_urls.primary_db_name
            )
        async with factory.read() as session:
            assert (
                str(await session.scalar(text("SELECT current_database()")))
                == postgresql_primary_and_replica_urls.replica_db_name
            )
    finally:
        await _run_shutdown_callbacks(shutdown_callbacks)

async def test_create_db_session_factory_uses_primary_when_replica_not_configured_for_postgres(
    dialect: str,
    postgresql_primary_and_replica_urls: PostgresPrimaryAndReplicaUrls,
) -> None:
    factory, shutdown_callbacks = _create_db_session_factory(
        db_connection_str=postgresql_primary_and_replica_urls.primary_url,
        read_replica_connection_str=None,
        migrate=False,
        log_to_stdout=False,
        log_migrations=False,
    )
    try:
        async with factory() as session:
            assert (
                str(await session.scalar(text("SELECT current_database()")))
                == postgresql_primary_and_replica_urls.primary_db_name
            )
        async with factory.read() as session:
            assert (
                str(await session.scalar(text("SELECT current_database()")))
                == postgresql_primary_and_replica_urls.primary_db_name
            )
    finally:
        await _run_shutdown_callbacks(shutdown_callbacks)


# --- tests/unit/server/cost_tracking/test_cost_details_calculator.py ---

    def test_cost_per_token_edge_cases(self) -> None:
        """
        Test edge cases for cost_per_token calculation.

        This test verifies proper handling of:
        - Cost per token when cost is None (no calculator available)
        - Cost per token when cost is 0 but tokens > 0
        - Cost per token when both cost and tokens are 0
        """
        # Create calculator without specific calculators for image/audio
        calculator = SpanCostDetailsCalculator(
            [
                models.TokenPrice(token_type="input", is_prompt=True, base_rate=0.001),
                models.TokenPrice(token_type="output", is_prompt=False, base_rate=0.002),
            ]
        )

        # Test case where we have tokens but cost is calculated (fallback behavior)
        result = calculator.calculate_details(
            {
                "llm": {
                    "token_count": {
                        "prompt": 100,
                        "completion": 50,
                        "prompt_details": {"image": 50},
                        "completion_details": {"reasoning": 25},
                    }
                }
            }
        )

        # Verify that all details have proper cost_per_token calculations
        for detail in result:
            if detail.tokens and detail.tokens > 0:
                if detail.cost is not None and detail.cost > 0:
                    assert detail.cost_per_token is not None
                    assert detail.cost_per_token == detail.cost / detail.tokens
                elif detail.cost == 0.0:
                    assert detail.cost_per_token == 0.0
            else:
                assert detail.cost_per_token is None


# --- tests/unit/server/cost_tracking/test_helpers.py ---

    def test_get_aggregated_tokens_success(
        self, attributes: dict[str, Any], expected: tuple[int, int, int]
    ) -> None:
        """Test successful token aggregation with various input scenarios."""
        result: tuple[Optional[int], Optional[int], Optional[int]] = get_aggregated_tokens(
            attributes
        )
        assert result == expected


# --- tests/unit/server/cost_tracking/test_regex_specificity.py ---

def test_score_invalid_patterns(pattern: str) -> None:
    with pytest.raises(ValueError):
        score(pattern)


# --- tests/unit/server/daemons/test_experiment_runner.py ---

    def test_record_success_resets_counter(self) -> None:
        cb = CircuitBreaker(threshold=3)
        cb.record_failure(RuntimeError("e1"))
        cb.record_failure(RuntimeError("e2"))
        assert cb._consecutive_failures == 2
        cb.record_success()
        assert cb._consecutive_failures == 0
        assert not cb.is_tripped

    def test_trips_at_threshold_and_stays_tripped(self) -> None:
        cb = CircuitBreaker(threshold=3)
        cb.record_failure(RuntimeError("e1"))
        cb.record_failure(RuntimeError("e2"))
        tripped = cb.record_failure(RuntimeError("e3"))
        assert tripped is True
        assert cb.is_tripped
        assert cb.trip_reason == "RuntimeError"
        # Success after trip does NOT un-trip
        cb.record_success()
        assert cb.is_tripped

    def test_already_tripped_ignores_further_failures(self) -> None:
        cb = CircuitBreaker(threshold=2)
        cb.record_failure(RuntimeError("e1"))
        assert cb.record_failure(RuntimeError("e2")) is True
        assert cb.is_tripped
        # Further failures return False (already tripped)
        assert cb.record_failure(RuntimeError("e3")) is False

    def test_task_batch_size_scales_with_max_concurrency(self) -> None:
        exp = _make_running_experiment(max_concurrency=20)
        assert exp._task_batch_size == 40

    def test_task_batch_size_is_bounded(self) -> None:
        exp_low = _make_running_experiment(max_concurrency=1)
        assert exp_low._task_batch_size == 10

        exp_zero = _make_running_experiment(max_concurrency=0)
        assert exp_zero._task_batch_size == 10

        exp_high = _make_running_experiment(max_concurrency=500)
        assert exp_high._task_batch_size == 200

    def test_backpressure_hysteresis_toggles_only_at_watermarks(self) -> None:
        exp = _make_running_experiment(max_concurrency=1)
        exp._work_item_high_watermark = 4
        exp._work_item_low_watermark = 2

        # Below high watermark -> remains off.
        for i in range(4):
            exp._task_queue.append(_make_task_work_item(exp, dataset_example_id=100 + i))
        exp._task_queue.pop()  # resident=3
        exp._update_backpressure_state()
        assert exp._backpressure_active is False

        # At high watermark -> turns on.
        exp._task_queue.append(_make_task_work_item(exp, dataset_example_id=104))  # resident=4
        exp._update_backpressure_state()
        assert exp._backpressure_active is True

        # Above low watermark -> stays on.
        exp._task_queue.popleft()  # resident=3
        exp._update_backpressure_state()
        assert exp._backpressure_active is True

        # At/below low watermark -> turns off.
        exp._task_queue.popleft()  # resident=2
        exp._update_backpressure_state()
        assert exp._backpressure_active is False

    def test_has_work_when_eval_db_not_exhausted(self) -> None:
        exp = _make_running_experiment()
        # Default: _eval_db_exhausted is True (no evaluators), _task_db_exhausted is False
        assert exp.has_work() is True
        exp._task_db_exhausted = True
        assert exp.has_work() is False

    async def test_try_get_ready_work_item_priority_order(self) -> None:
        """Evals > ready retries > tasks."""
        exp = _make_running_experiment()
        exp._task_db_exhausted = True
        exp._eval_db_exhausted = True

        task = _make_task_work_item(exp, dataset_example_id=1)
        eval_work_item = _make_eval_work_item(exp)
        retry_task = _make_task_work_item(exp, dataset_example_id=2)
        retry_item = RetryItem(
            ready_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            work_item=retry_task,
        )

        # Add all three types
        exp._task_queue.append(task)
        exp._eval_queue.append(eval_work_item)
        heapq.heappush(exp._retry_heap, retry_item)

        # First: eval (highest priority)
        work_item1 = await exp.try_get_ready_work_item()
        assert work_item1 is eval_work_item

        # Second: ready retry
        work_item2 = await exp.try_get_ready_work_item()
        assert work_item2 is retry_task

        # Third: task
        work_item3 = await exp.try_get_ready_work_item()
        assert work_item3 is task

    async def test_try_get_ready_work_item_respects_max_concurrency(self) -> None:
        """Returns None when in_flight >= max_concurrency."""
        exp = _make_running_experiment(max_concurrency=1)
        exp._task_db_exhausted = True
        exp._eval_db_exhausted = True

        task1 = _make_task_work_item(exp, dataset_example_id=1)
        task2 = _make_task_work_item(exp, dataset_example_id=2)
        exp._task_queue.append(task1)
        exp._task_queue.append(task2)

        # Simulate one in-flight work item
        exp._in_flight.add(task1)
        exp._task_queue.popleft()

        work_item = await exp.try_get_ready_work_item()
        assert work_item is None

    async def test_on_rate_limit_requeues_with_backoff(self) -> None:
        """Work item lands in retry heap with correct ready_at."""
        exp = _make_running_experiment(base_backoff_seconds=1.0)
        task = _make_task_work_item(exp)
        exp._task_db_exhausted = True
        exp._eval_db_exhausted = True

        before = datetime.now(timezone.utc)
        await exp.on_rate_limit(task)
        after = datetime.now(timezone.utc)

        assert len(exp._retry_heap) == 1
        retry = exp._retry_heap[0]
        assert retry.work_item is task
        assert task.retry_count == 1
        # Backoff = 1.0 * 2^(1-1) = 1.0s
        assert retry.ready_at >= before + timedelta(seconds=1.0)
        assert retry.ready_at <= after + timedelta(seconds=1.0)

    async def test_unregister_cancel_scope_cleans_in_flight_state(self) -> None:
        """Unregister always removes work item from in-flight and scope maps."""
        exp = _make_running_experiment()
        eval_item = _make_eval_work_item(exp, run_id=313, dataset_evaluator_id=1)
        scope = anyio.CancelScope()

        exp.register_cancel_scope(eval_item, scope)
        assert eval_item in exp._in_flight
        assert eval_item in exp._cancel_scopes

        await exp.unregister_cancel_scope(eval_item)

        assert eval_item not in exp._in_flight
        assert eval_item not in exp._cancel_scopes

    async def test_check_completion_fires_on_done(self) -> None:
        """When has_work() returns False, _on_done callback invoked."""
        on_done = _make_on_done()
        exp = _make_running_experiment(on_done=on_done)
        exp._task_db_exhausted = True
        exp._eval_db_exhausted = True

        await exp._check_completion()

        assert not exp._active
        on_done.assert_called_once_with(exp._experiment.id)

    async def test_round_robin_picks_least_recently_served(self) -> None:
        """_try_get_ready_work_item in ExperimentRunner picks least-recently-served experiment."""
        from phoenix.server.daemons.experiment_runner import ExperimentRunner

        runner = object.__new__(ExperimentRunner)
        runner._experiments = {}

        exp_a = _make_running_experiment(experiment_id=1)
        exp_a._task_db_exhausted = True
        exp_a._eval_db_exhausted = True
        exp_a.last_served_at = datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

        exp_b = _make_running_experiment(experiment_id=2)
        exp_b._task_db_exhausted = True
        exp_b._eval_db_exhausted = True
        exp_b.last_served_at = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # older

        task_a = _make_task_work_item(exp_a, dataset_example_id=1)
        task_b = _make_task_work_item(exp_b, dataset_example_id=2)
        exp_a._task_queue.append(task_a)
        exp_b._task_queue.append(task_b)

        runner._experiments = {1: exp_a, 2: exp_b}

        work_item = await runner._try_get_ready_work_item()
        # exp_b was served less recently, so it should be picked first
        assert work_item is task_b


# --- tests/unit/trace/test_attributes.py ---

def test_unflatten(key_value_pairs: tuple[tuple[str, Any], ...], desired: dict[str, Any]) -> None:
    actual = dict(unflatten(key_value_pairs))
    assert actual == desired
    actual = dict(unflatten(reversed(key_value_pairs)))
    assert actual == desired


# --- tests/unit/trace/test_span_json_decoder.py ---

def test_span_json_decoder_document_retrieval() -> None:
    span = json_to_span(
        {
            "name": "retrieve",
            "context": {
                "trace_id": "9241913b-eadf-4891-9e17-24686ccc3ed3",
                "span_id": "89ff67d5-1818-41b7-ab09-17bcc450491c",
            },
            "span_kind": "RETRIEVER",
            "parent_id": "d493e9ab-321f-41ac-a1b2-60d80c50b2cb",
            "start_time": "2023-09-15T14:04:07.167267",
            "end_time": "2023-09-15T14:04:07.812851",
            "status_code": "OK",
            "status_message": "",
            "attributes": {
                "input.value": "How do I use the SDK to upload a ranking model?",
                "input.mime_type": "text/plain",
                "retrieval.documents": [
                    {
                        "document.id": "883e74ee-691a-46e0-acd7-f58bd565dad4",
                        "document.score": 0.8024018669959406,
                        "document.content": """\nRanking models are used by
                                            search engines to display query
                                            results ranked in the order of the
                                            highest relevance. These predictions
                                            seek to maximize user actions that
                                            are then used to evaluate model
                                            performance.&#x20;\n\nThe complexity
                                            within a ranking model makes
                                            failures challenging to pinpoint as
                                            a model\u2019s dimensions expand per
                                            recommendation. Notable challenges
                                            within ranking models include
                                            upstream data quality issues,
                                            poor-performing segments, the cold
                                            start problem, and more.
                                            &#x20;\n\n\n\n""",
                        "document.metadata": {"category": "ranking"},
                    },
                    {
                        "document.id": "d169f0ce-b5ea-4e88-9653-f8bb2fb1d105",
                        "document.score": 0.7964861566463088,
                        "document.content": """\n**Use the
                                            'arize-demo-hotel-ranking' model,
                                            available in all free accounts, to
                                            follow along.**&#x20;\n\n""",
                        "document.metadata": {},
                    },
                ],
            },
            "events": [],
            "conversation": None,
        }
    )
    assert span.name == "retrieve"
    assert len(span.attributes["retrieval.documents"]) == 2


# --- tests/unit/trace/dsl/test_filter.py ---

def test_get_attribute_keys_list(expression: str, expected: Optional[list[str]]) -> None:
    actual = _get_attribute_keys_list(
        ast.parse(expression, mode="eval").body,
    )
    if expected is None:
        assert actual is None
    else:
        assert isinstance(actual, list)
        assert [c.value for c in actual] == expected


# --- tests/unit/trace/dsl/test_query.py ---

async def test_select_all(
    db: DbSessionFactory,
    abc_project: Any,
) -> None:
    # i.e. `get_spans_dataframe`
    sq = SpanQuery()
    expected = pd.DataFrame(
        {
            "context.span_id": ["234", "345", "456", "567"],
            "context.trace_id": ["012", "012", "012", "012"],
            "parent_id": ["123", "234", "234", "234"],
            "name": ["root span", "embedding span", "retriever span", "llm span"],
            "span_kind": ["UNKNOWN", "EMBEDDING", "RETRIEVER", "LLM"],
            "status_code": ["OK", "OK", "OK", "ERROR"],
            "status_message": ["okay", "no problemo", "okay", "uh-oh"],
            "start_time": [
                datetime.fromisoformat("2021-01-01T00:00:00.000+00:00"),
                datetime.fromisoformat("2021-01-01T00:00:00.000+00:00"),
                datetime.fromisoformat("2021-01-01T00:00:05.000+00:00"),
                datetime.fromisoformat("2021-01-01T00:00:20.000+00:00"),
            ],
            "end_time": [
                datetime.fromisoformat("2021-01-01T00:00:30.000+00:00"),
                datetime.fromisoformat("2021-01-01T00:00:05.000+00:00"),
                datetime.fromisoformat("2021-01-01T00:00:20.000+00:00"),
                datetime.fromisoformat("2021-01-01T00:00:30.000+00:00"),
            ],
            "attributes.input.value": ["xy%z*", "XY%*Z", "xy%*z", None],
            "attributes.output.value": ["321", None, None, None],
            "attributes.llm.token_count.prompt": [None, None, None, 100.0],
            "attributes.llm.token_count.completion": [None, None, None, 200.0],
            "attributes.metadata": [
                None,
                {"a.b.c": 123, "1.2.3": "abc", "x.y": {"z.a": {"b.c": 321}}},
                None,
                None,
            ],
            "attributes.embedding.model_name": [None, "xyz", None, None],
            "attributes.embedding.embeddings": [
                None,
                [
                    {"embedding.vector": [1, 2, 3], "embedding.text": "123"},
                    {"embedding.vector": [2, 3, 4], "embedding.text": "234"},
                ],
                None,
                None,
            ],
            "attributes.retrieval.documents": [
                None,
                None,
                [
                    {"document.content": "A", "document.score": 1.0},
                    {"document.content": "B", "document.score": 2.0},
                    {"document.content": "C", "document.score": 3.0},
                ],
                None,
            ],
            "attributes.attributes": [None, None, "attributes", {"attributes": "attributes"}],
            "events": [[], [], [], []],
        }
    ).set_index("context.span_id", drop=False)
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_select_all_with_no_data(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery()
    expected = pd.DataFrame(
        columns=[
            "context.span_id",
            "context.trace_id",
            "parent_id",
            "name",
            "span_kind",
            "status_code",
            "status_message",
            "start_time",
            "end_time",
            "events",
        ]
    ).set_index("context.span_id", drop=False)
    async with db() as session:
        actual = await session.run_sync(sq, project_name="opq")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_select(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().select("name", tcp="llm.token_count.prompt")
    expected = pd.DataFrame(
        {
            "context.span_id": ["234", "345", "456", "567"],
            "name": ["root span", "embedding span", "retriever span", "llm span"],
            "tcp": [None, None, None, 100.0],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_select_parent_id_as_span_id(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().select("name", span_id="parent_id")
    expected = pd.DataFrame(
        {
            "context.span_id": ["123", "234", "234", "234"],
            "name": ["root span", "embedding span", "retriever span", "llm span"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_select_trace_id_as_index(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().select("span_id").with_index("trace_id")
    expected = pd.DataFrame(
        {
            "context.trace_id": ["012", "012", "012", "012"],
            "context.span_id": ["234", "345", "456", "567"],
        }
    ).set_index("context.trace_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1).sort_values("context.span_id"),
        expected.sort_index().sort_index(axis=1).sort_values("context.span_id"),
    )

async def test_select_nonexistent(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().select("name", "opq", "opq.rst")
    expected = pd.DataFrame(
        {
            "context.span_id": ["234", "345", "456", "567"],
            "name": ["root span", "embedding span", "retriever span", "llm span"],
            "opq": [None, None, None, None],
            "opq.rst": [None, None, None, None],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_default_project(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().select(
        "name",
        **{"Latency (milliseconds)": "latency_ms"},
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["2345"],
            "name": ["root span"],
            "Latency (milliseconds)": [30000.0],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, root_spans_only=True)
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_root_spans_only(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().select(
        "name",
        **{"Latency (milliseconds)": "latency_ms"},
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["234"],
            "name": ["root span"],
            "Latency (milliseconds)": [30000.0],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc", root_spans_only=True)
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_start_time(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().select("name")
    expected = pd.DataFrame(
        {
            "context.span_id": ["567"],
            "name": ["llm span"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(
            sq,
            project_name="abc",
            start_time=datetime.fromisoformat(
                "2021-01-01T00:00:20.000+00:00",
            ),
        )
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_end_time(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().select("name")
    expected = pd.DataFrame(
        {
            "context.span_id": ["234", "345"],
            "name": ["root span", "embedding span"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(
            sq,
            project_name="abc",
            end_time=datetime.fromisoformat(
                "2021-01-01T00:00:01.000+00:00",
            ),
        )
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_limit(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery()
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc", limit=2)
    # Newest-first ordering
    assert actual.index.tolist() == ["567", "456"]

async def test_limit_with_select_statement(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().select("context.span_id")
    expected = pd.DataFrame(
        {
            "context.span_id": ["234", "345"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc", limit=2)
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_for_none(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("name")
        .where(
            "parent_id is None",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": [],
            "name": [],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
        check_dtype=False,
        check_column_type=False,
        check_frame_type=False,
        check_index_type=False,
    )

async def test_filter_for_not_none(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("name")
        .where(
            "output.value is not None",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["234"],
            "name": ["root span"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_for_substring_case_sensitive_not_glob_not_like(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("input.value")
        .where(
            "'y%*' in input.value",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["456"],
            "input.value": ["xy%*z"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_for_not_substring_case_sensitive_not_glob_not_like(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("input.value")
        .where(
            "'y%*' not in input.value",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["234", "345"],
            "input.value": ["xy%z*", "XY%*Z"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_nonexistent_is_not_none(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("name")
        .where(
            "opq is not None or opq.rst is not None",
        )
    )
    expected = pd.DataFrame(
        columns=["context.span_id", "name"],
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_nonexistent_is_none(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("name")
        .where(
            "opq is None or opq.rst is None",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["234", "345", "456", "567"],
            "name": ["root span", "embedding span", "retriever span", "llm span"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_latency(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select(
            "name",
            **{"Latency (milliseconds)": "latency_ms"},
        )
        .where("9_000 < latency_ms < 11_000")
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["567"],
            "name": ["llm span"],
            "Latency (milliseconds)": [10000.0],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_cumulative_token_count(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("name")
        .where("290 < cumulative_token_count.total < 310 and llm.token_count.prompt is None")
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["234"],
            "name": ["root span"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_metadata_with_arithmetic(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("metadata['a.b.c']")
        .where(
            "12 - metadata['a.b.c'] == -111",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["345"],
            "metadata['a.b.c']": [123],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_metadata_cast_as_int(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("metadata['a.b.c']")
        .where(
            "12 - int(metadata['a.b.c']) == -111",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["345"],
            "metadata['a.b.c']": [123],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_metadata_substring_search(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("metadata['1.2.3']")
        .where(
            "'b' in metadata['1.2.3']",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["345"],
            "metadata['1.2.3']": ["abc"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_metadata_cast_as_str(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("metadata['1.2.3']")
        .where(
            "'b' in str(metadata['1.2.3'])",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["345"],
            "metadata['1.2.3']": ["abc"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_metadata_using_subscript_key(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("metadata['1.2.3']")
        .where(
            "metadata['1.2.3'] == 'abc'",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["345"],
            "metadata['1.2.3']": ["abc"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_metadata_using_subscript_keys_list_with_single_key(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("metadata[['1.2.3']]")
        .where(
            "metadata[['1.2.3']] == 'abc'",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["345"],
            "metadata[['1.2.3']]": ["abc"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_metadata_using_subscript_keys_list_with_multiple_keys(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("metadata[['x.y', 'z.a']]")
        .where(
            "metadata[['x.y', 'z.a', 'b.c']] == 321",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["345"],
            "metadata[['x.y', 'z.a']]": [{"b.c": 321}],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_attribute_using_subscript_key(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("attributes['attributes']")
        .where(
            "attributes['attributes'] == 'attributes'",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["456"],
            "attributes['attributes']": ["attributes"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_attribute_using_subscript_keys_list_with_single_key(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("attributes[['attributes']]")
        .where(
            "attributes[['attributes']] == 'attributes'",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["456"],
            "attributes[['attributes']]": ["attributes"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_attribute_using_subscript_keys_list_with_multiple_keys(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("attributes[['attributes', 'attributes']]")
        .where(
            "attributes[['attributes', 'attributes']] == 'attributes'",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["567"],
            "attributes[['attributes', 'attributes']]": ["attributes"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_span_id_single(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("embedding.model_name")
        .where(
            "span_id == '345'",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["345"],
            "embedding.model_name": ["xyz"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_span_id_multiple(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("embedding.model_name")
        .where(
            "span_id in ['345', '567']",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["345", "567"],
            "embedding.model_name": ["xyz", None],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_trace_id_single(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("trace_id")
        .where(
            "trace_id == '012'",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["234", "345", "456", "567"],
            "context.trace_id": ["012", "012", "012", "012"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_trace_id_multiple(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("trace_id")
        .where(
            "trace_id in ('012',)",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["234", "345", "456", "567"],
            "context.trace_id": ["012", "012", "012", "012"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_filter_on_span_annotation(
    db: DbSessionFactory,
    abc_project: Any,
    condition: str,
    expected: list[str],
) -> None:
    sq = SpanQuery().select("span_id").where(condition)
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert sorted(actual.index) == expected

async def test_explode_embeddings_no_select(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().explode("embedding.embeddings")
    expected = pd.DataFrame(
        {
            "context.span_id": ["345", "345"],
            "position": [0, 1],
            "embedding.text": ["123", "234"],
            "embedding.vector": [[1, 2, 3], [2, 3, 4]],
        }
    ).set_index(["context.span_id", "position"])
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_explode_embeddings_with_select_and_no_kwargs(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("embedding.model_name")
        .explode(
            "embedding.embeddings",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["345", "345"],
            "position": [0, 1],
            "embedding.model_name": ["xyz", "xyz"],
            "embedding.text": ["123", "234"],
            "embedding.vector": [[1, 2, 3], [2, 3, 4]],
        }
    ).set_index(["context.span_id", "position"])
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_explode_documents_no_select(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().explode(
        "retrieval.documents",
        content="document.content",
        score="document.score",
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["456", "456", "456"],
            "document_position": [0, 1, 2],
            "content": ["A", "B", "C"],
            "score": [1, 2, 3],
        }
    ).set_index(["context.span_id", "document_position"])
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_explode_documents_with_select_and_non_ascii_kwargs(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("trace_id")
        .explode(
            "retrieval.documents",
            **{
                "콘텐츠": "document.content",
                "スコア": "document.score",
            },
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["456", "456", "456"],
            "document_position": [0, 1, 2],
            "context.trace_id": ["012", "012", "012"],
            "콘텐츠": ["A", "B", "C"],
            "スコア": [1, 2, 3],
        }
    ).set_index(["context.span_id", "document_position"])
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_concat_documents_no_select(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().concat(
        "retrieval.documents",
        content="document.content",
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["456"],
            "content": ["A\n\nB\n\nC"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_concat_documents_no_select_but_no_data(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = SpanQuery().concat(
        "retrieval.documents",
        content="document.content",
    )
    expected = pd.DataFrame(
        columns=["context.span_id", "content"],
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="opq")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_concat_documents_with_select(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("trace_id")
        .concat(
            "retrieval.documents",
            content="document.content",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["456"],
            "context.trace_id": ["012"],
            "content": ["A\n\nB\n\nC"],
        }
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_concat_documents_with_select_but_no_data(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("trace_id")
        .concat(
            "retrieval.documents",
            content="document.content",
        )
    )
    expected = pd.DataFrame(
        columns=["context.span_id", "content", "context.trace_id"],
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="opq")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_concat_documents_with_select_but_with_typo_in_array_name(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("trace_id")
        .concat(
            "retriever.documents",
            content="document.content",
        )
    )
    expected = pd.DataFrame(
        columns=["context.span_id", "content", "context.trace_id"],
    ).set_index("context.span_id")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_concat_documents_with_select_and_non_default_separator(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .with_index("name")
        .with_concat_separator(",")
        .concat(
            "embedding.embeddings",
            text="embedding.text",
        )
    )
    expected = pd.DataFrame(
        {
            "name": ["embedding span"],
            "text": ["123,234"],
        }
    ).set_index("name")
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_explode_and_concat_on_same_array(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .concat(
            "retrieval.documents",
            content="document.content",
        )
        .explode(
            "retrieval.documents",
            score="document.score",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["456", "456", "456"],
            "document_position": [0, 1, 2],
            "content": ["A\n\nB\n\nC", "A\n\nB\n\nC", "A\n\nB\n\nC"],
            "score": [1, 2, 3],
        }
    ).set_index(["context.span_id", "document_position"])
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_explode_and_concat_on_same_array_but_no_data(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .concat(
            "retrieval.documents",
            content="document.content",
        )
        .explode(
            "retrieval.documents",
            score="document.score",
        )
    )
    expected = pd.DataFrame(
        columns=[
            "context.span_id",
            "document_position",
            "content",
            "score",
        ]
    ).set_index(["context.span_id", "document_position"])
    async with db() as session:
        actual = await session.run_sync(sq, project_name="opq")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_explode_and_concat_on_same_array_with_same_label(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    async with db() as session:
        bind = session.get_bind()
        if isinstance(bind, Engine) and "asyncpg" in str(bind.url):
            pytest.xfail("FIX THIS: this test does not currently pass for postgres")
    sq = (
        SpanQuery()
        .concat(
            "retrieval.documents",
            content="document.content",
        )
        .explode(
            "retrieval.documents",
            content="document.content",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["456", "456", "456"],
            "document_position": [0, 1, 2],
            "content": ["A\n\nB\n\nC", "A\n\nB\n\nC", "A\n\nB\n\nC"],
        }
    ).set_index(["context.span_id", "document_position"])
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_explode_and_concat_on_same_array_but_with_typo_in_concat_array_name(
    db: DbSessionFactory,
    default_project: None,
    abc_project: None,
) -> None:
    sq = (
        SpanQuery()
        .concat(
            "retriever.documents",
            content="document.content",
        )
        .explode(
            "retrieval.documents",
            score="document.score",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["456", "456", "456"],
            "document_position": [0, 1, 2],
            "content": [None, None, None],
            "score": [1, 2, 3],
        }
    ).set_index(["context.span_id", "document_position"])
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_explode_and_concat_on_same_array_but_with_typo_in_explode_array_name(
    db: DbSessionFactory,
    default_project: None,
    abc_project: None,
) -> None:
    async with db() as session:
        bind = session.get_bind()
        if isinstance(bind, Engine) and "asyncpg" in str(bind.url):
            pytest.xfail("FIX THIS: this test does not currently pass for postgres")
    sq = (
        SpanQuery()
        .concat(
            "retrieval.documents",
            content="document.content",
        )
        .explode(
            "retriever.documents",
            score="document.score",
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["456"],
            "content": ["A\n\nB\n\nC"],
        }
    ).set_index(["context.span_id"])
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

async def test_explode_and_concat_on_same_array_with_non_ascii_kwargs(
    db: DbSessionFactory,
    default_project: Any,
    abc_project: Any,
) -> None:
    sq = (
        SpanQuery()
        .select("name")
        .concat(
            "retrieval.documents",
            **{"콘텐츠": "document.content"},
        )
        .explode(
            "retrieval.documents",
            **{"スコア": "document.score"},
        )
    )
    expected = pd.DataFrame(
        {
            "context.span_id": ["456", "456", "456"],
            "document_position": [0, 1, 2],
            "name": ["retriever span", "retriever span", "retriever span"],
            "콘텐츠": ["A\n\nB\n\nC", "A\n\nB\n\nC", "A\n\nB\n\nC"],
            "スコア": [1, 2, 3],
        }
    ).set_index(["context.span_id", "document_position"])
    async with db() as session:
        actual = await session.run_sync(sq, project_name="abc")
    assert_frame_equal(
        actual.sort_index().sort_index(axis=1),
        expected.sort_index().sort_index(axis=1),
    )

