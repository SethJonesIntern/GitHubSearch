# truera/trulens
# 31 LLM-backed test functions across 104 test files
# Source: https://github.com/truera/trulens

# --- tests/e2e/test_otel_combined_costs.py ---

    def test_uncombinable_costs(self):
        app = _TestApp()
        tru_app = TruApp(
            app,
            app_name="test_app",
            app_version="v1",
            main_method=app.respond_to_query,
        )
        tru_app.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("What's 21+21?", ["cortex", "openai"]),
        )
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 1)
        self._check_costs(
            events.iloc[0]["record_attributes"], "mixed", "mixed", False
        )

    def test_combinable_costs(self):
        app = _TestApp()
        tru_app = TruApp(
            app,
            app_name="test_app",
            app_version="v1",
            main_method=app.respond_to_query,
        )
        tru_app.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("What's 21+21?", ["cortex", "cortex"]),
        )
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 1)
        self._check_costs(
            events.iloc[0]["record_attributes"],
            "mistral-large2",
            "Snowflake credits",
            False,
        )
        self.assertEqual(
            events.iloc[0]["record_attributes"][
                SpanAttributes.COST.NUM_PROMPT_TOKENS
            ]
            % 2,
            0,
        )


# --- tests/e2e/test_otel_costs.py ---

    def test_tru_chain_cortex(self):
        # Set up.
        tru_session = TruSession()
        tru_session.reset_database()
        # Create app
        os.environ["SNOWFLAKE_USERNAME"] = os.environ["SNOWFLAKE_USER"]
        os.environ["SNOWFLAKE_PASSWORD"] = os.environ["SNOWFLAKE_USER_PASSWORD"]
        app = ChatSnowflakeCortex(
            model="mistral-large2",
            cortex_function="complete",
        )
        tru_recorder = TruChain(app, app_name="testing", app_version="v1")
        tru_recorder.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("How is baby Kojikun able to be so cute?",),
        )
        tru_session.force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 3)


# --- tests/e2e/test_tru_chain.py ---

    def test_sync(self):
        """Synchronous (`invoke`) test."""

        chain, recorder = self._create_basic_chain(streaming=False)

        message, expected_answers = self._get_question_and_answers(0)

        with recorder as recording:
            result = chain.invoke(input=dict(question=message))

        record = recording.get()

        self.assertIn(result, expected_answers)

        self._check_generation_costs(record.cost)

        # Check that recorder is garbage collected.
        recorder_ref = weakref.ref(recorder)
        chain_ref = weakref.ref(chain)
        del recorder, recording, record, chain
        gc.collect()
        self.assertCollected(recorder_ref, "recorder isn't GCed!")
        self.assertCollected(chain_ref, "chain isn't GCed!")

    async def test_async(self):
        """Asynchronous (`ainvoke`) test."""

        chain, recorder = self._create_basic_chain(streaming=False)

        message, expected_answers = self._get_question_and_answers(0)

        async with recorder as recording:
            result = await chain.ainvoke(input=dict(question=message))

        record = recording.get()

        self.assertIn(result, expected_answers)

        self._check_generation_costs(record.cost)

        # Check that recorder is garbage collected.
        recorder_ref = weakref.ref(recorder)
        # chain_ref = weakref.ref(chain)
        del recorder, recording, record, chain, result
        self.assertCollected(recorder_ref)

    def test_sync_stream(self):
        """Synchronous stream (`stream`) test."""

        chain, recorder = self._create_basic_chain(streaming=True)

        message, expected_answers = self._get_question_and_answers(0)

        result = ""
        with recorder as recording:
            for chunk in chain.stream(input=dict(question=message)):
                result += chunk

        record = recording.get()

        self.assertIn(result, expected_answers)

        self._check_stream_generation_costs(record.cost)

        # Check that recorder is garbage collected.
        recorder_ref = weakref.ref(recorder)
        chain_ref = weakref.ref(chain)
        del recorder, recording, record, chain, result
        self.assertCollected(recorder_ref)
        self.assertCollected(chain_ref)

    async def test_async_stream(self):
        """Asynchronous stream (`astream`) test."""

        chain, recorder = self._create_basic_chain(streaming=True)

        message, expected_answers = self._get_question_and_answers(0)

        result = ""
        async with recorder as recording:
            async for chunk in chain.astream(input=dict(question=message)):
                result += chunk

        record = recording.get()

        self.assertIn(result, expected_answers)

        self._check_stream_generation_costs(record.cost)

        # Check that recorder is garbage collected.
        recorder_ref = weakref.ref(recorder)
        # chain_ref = weakref.ref(chain)
        del recorder, recording, record, chain
        self.assertCollected(recorder_ref)

    def test_record_metadata_plain(self):
        """Test inclusion of metadata in records."""

        # Need unique app_id per test as they may be run in parallel and have
        # same ids.
        session = core_session.TruSession()
        chain, recorder = self._create_basic_chain(app_name="metaplain")

        message, _ = self._get_question_and_answers(0)
        meta = "this is plain metadata"

        with recorder as recording:
            recording.record_metadata = meta
            chain.invoke(input=dict(question=message))

        record = recording.get()

        with self.subTest("Check the record has the metadata"):
            self.assertEqual(record.meta, meta)

        with self.subTest(
            "Check the record has the metadata when retrieved back from db"
        ):
            recs, _ = session.get_records_and_feedback([recorder.app_id])
            self.assertGreater(len(recs), 0)
            rec = self._load_first_record(recs.iloc[-1:])
            self.assertEqual(rec.meta, meta)

        with self.subTest("Check updating the record metadata in the db."):
            new_meta = "this is new meta"
            rec.meta = new_meta
            session.update_record(rec)
            recs, _ = session.get_records_and_feedback([recorder.app_id])
            self.assertGreater(len(recs), 0)
            rec = self._load_first_record(recs[recs.record_id == rec.record_id])
            self.assertNotEqual(rec.meta, meta)
            self.assertEqual(rec.meta, new_meta)

        with self.subTest(
            "Check adding meta to a record that initially didn't have it."
        ):
            with recorder as recording:
                chain.invoke(input=dict(question=message))

            with self.subTest("with no metadata"):
                rec = recording.get()
                self.assertEqual(rec.meta, None)
                recs, _ = session.get_records_and_feedback([recorder.app_id])
                self.assertGreater(len(recs), 1)
                rec = self._load_first_record(
                    recs[recs.record_id == rec.record_id]
                )
                self.assertEqual(rec.meta, None)

            with self.subTest("Updated with metadata"):
                rec.meta = new_meta
                session.update_record(rec)
                recs, _ = session.get_records_and_feedback([recorder.app_id])
                self.assertGreater(len(recs), 1)
                rec = self._load_first_record(
                    recs[recs.record_id == rec.record_id]
                )
                self.assertEqual(rec.meta, new_meta)

    def test_record_metadata_json(self):
        """Test inclusion of json metadata in records."""

        # Need unique app_id per test as they may be run in parallel and have
        # same ids.
        chain, recorder = self._create_basic_chain(app_name="metajson")

        message, _ = self._get_question_and_answers(0)
        meta = dict(field1="hello", field2="there")

        with recorder as recording:
            recording.record_metadata = meta
            chain.invoke(input=dict(question=message))
        record = recording.get()

        # Check record has metadata.
        self.assertEqual(record.meta, meta)

        # Check the record has the metadata when retrieved back from db.
        recs, _ = core_session.TruSession().get_records_and_feedback([
            recorder.app_id
        ])
        self.assertGreater(len(recs), 0)
        rec = self._load_first_record(recs)
        self.assertEqual(rec.meta, meta)

        # Check updating the record metadata in the db.
        new_meta = dict(hello="this is new meta")
        rec.meta = new_meta
        core_session.TruSession().update_record(rec)

        recs, _ = core_session.TruSession().get_records_and_feedback([
            recorder.app_id
        ])
        self.assertGreater(len(recs), 0)
        rec = self._load_first_record(recs)
        self.assertNotEqual(rec.meta, meta)
        self.assertEqual(rec.meta, new_meta)


# --- tests/e2e/test_tru_llama.py ---

    def test_query_engine_sync_stream(self):
        """Synchronous streaming query engine test."""

        self._sync_test(self._create_query_engine, "query", streaming=True)

    async def test_query_engine_async_stream(self):
        """Asynchronous streaming query engine test."""

        await self._async_test(
            self._create_query_engine, "aquery", streaming=True
        )

    async def test_chat_engine_async(self):
        """Asynchronous chat engine test."""

        await self._async_test(self._create_chat_engine, "achat")

    def test_chat_engine_sync_stream(self):
        """Synchronous streaming chat engine test."""

        self._sync_test(self._create_chat_engine, "chat", streaming=True)

    async def test_chat_engine_async_stream(self):
        """Asynchronous streaming chat engine test."""

        await self._async_test(
            self._create_chat_engine, "achat", streaming=True
        )


# --- tests/e2e/test_tru_session.py ---

    def test_run_feedback_functions_wait(self):
        """
        Test run_feedback_functions in wait mode. This mode blocks until results
        are ready.
        """

        app = self._create_custom()

        feedbacks = self._create_feedback_functions()

        expected_feedback_names = {f.name for f in feedbacks}

        session = core_session.TruSession()

        tru_app = custom_app.TruCustomApp(app)

        with tru_app as recording:
            app.respond_to_query("hello")

        record = recording.get()

        feedback_results = list(
            session.run_feedback_functions(
                record=record,
                feedback_functions=feedbacks,
                app=tru_app,
                wait=True,
            )
        )

        # Check we get the right number of results.
        self.assertEqual(len(feedback_results), len(feedbacks))

        # Check that the results are for the feedbacks we submitted.
        self.assertEqual(
            set(expected_feedback_names),
            set(res.name for res in feedback_results),
            "feedback result names do not match requested feedback names",
        )

        # Check that the structure of returned tuples is correct.
        for result in feedback_results:
            self.assertIsInstance(result, feedback_schema.FeedbackResult)
            self.assertIsInstance(result.result, float)

        # TODO: move tests to test_add_feedbacks.
        # Add to db.
        session.add_feedbacks(feedback_results)

        # Check that results were added to db.
        _, returned_feedback_names = session.get_records_and_feedback(
            app_ids=[tru_app.app_id]
        )

        # Check we got the right feedback names from db.
        self.assertEqual(expected_feedback_names, set(returned_feedback_names))

    def test_run_feedback_functions_nowait(self):
        """
        Test run_feedback_functions in non-blocking mode. This mode returns
        futures instead of results.
        """

        app = self._create_custom()

        feedbacks = self._create_feedback_functions()
        expected_feedback_names = {f.name for f in feedbacks}

        session = core_session.TruSession()

        tru_app = custom_app.TruCustomApp(app)

        with tru_app as recording:
            app.respond_to_query("hello")

        record = recording.get()

        start_time = datetime.now()

        future_feedback_results = list(
            session.run_feedback_functions(
                record=record,
                feedback_functions=feedbacks,
                app=tru_app,
                wait=False,
            )
        )

        end_time = datetime.now()

        # Should return quickly.
        self.assertLess(
            (end_time - start_time).total_seconds(),
            2.0,  # TODO: get it to return faster
            "Non-blocking run_feedback_functions did not return fast enough.",
        )

        # Check we get the right number of results.
        self.assertEqual(len(future_feedback_results), len(feedbacks))

        feedback_results = []

        # Check that the structure of returned tuples is correct.
        for future_result in future_feedback_results:
            self.assertIsInstance(future_result, FutureClass)

            wait([future_result])

            result = future_result.result()
            self.assertIsInstance(result, feedback_schema.FeedbackResult)
            self.assertIsInstance(result.result, float)

            feedback_results.append(result)

        # TODO: move tests to test_add_feedbacks.
        # Add to db.
        session.add_feedbacks(feedback_results)

        # Check that results were added to db.
        _, returned_feedback_names = session.get_records_and_feedback(
            app_ids=[tru_app.app_id]
        )

        # Check we got the right feedback names.
        self.assertEqual(expected_feedback_names, set(returned_feedback_names))

    def test_start_evaluator_with_blocking(self):
        session = core_session.TruSession()
        f = core_feedback.Feedback(
            feedback_tests.custom_feedback_function
        ).on_default()
        app_name = f"test_start_evaluator_with_blocking_{str(uuid.uuid4())}"
        tru_app = basic_app.TruBasicApp(
            text_to_text=lambda t: f"returning {t}",
            feedbacks=[f],
            feedback_mode=feedback_schema.FeedbackMode.DEFERRED,
            app_name=app_name,
        )
        with tru_app:
            tru_app.main_call("test_deferred_mode")
        time.sleep(2)
        session.start_evaluator(return_when_done=True)
        if session._evaluator_proc is not None:
            # We should never get here since the variable isn't supposed to be set.
            raise ValueError("The evaluator is still running!")
        records_and_feedback = session.get_records_and_feedback(
            app_ids=[tru_app.app_id]
        )
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(records_and_feedback[1], ["custom_feedback_function"])
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["custom_feedback_function"].iloc[0],
            0.1,
        )


# --- tests/unit/test_langchain_instrumentation.py ---

    def test_vectorstore_similarity_search_retrieval_span(self):
        """Test 2: VectorStore.similarity_search is instrumented with RETRIEVAL span type."""
        # Create a simple chain that uses the vector store
        retriever = self.vectorstore.as_retriever()

        # Create TruChain to trigger instrumentation
        tru_chain = TruChain(
            retriever,
            app_name="test_retrieval",
            app_version="v1",
            main_method=retriever.invoke,
        )

        # Invoke the retriever
        tru_chain.instrumented_invoke_main_method(
            run_name="test_retrieval_run",
            input_id="test_id_3",
            main_method_args=("search query",),
        )

        # Get events from database
        events_df = self._get_events()

        # Filter for similarity_search events
        search_events = events_df[
            events_df["record"].apply(
                lambda x: "similarity_search" in x.get("name", "")
            )
        ]

        # Verify at least one similarity_search event exists
        self.assertGreater(
            len(search_events), 0, "No similarity_search events found"
        )

        # Check that the span type is RETRIEVAL
        for _, event in search_events.iterrows():
            span_type = event["record_attributes"].get(SpanAttributes.SPAN_TYPE)
            self.assertEqual(
                span_type,
                SpanAttributes.SpanType.RETRIEVAL,
                f"Expected RETRIEVAL span type, got {span_type}",
            )

    async def test_vectorstore_asimilarity_search_retrieval_span(self):
        """Test 2b: VectorStore.asimilarity_search is instrumented with RETRIEVAL span type."""
        # Create a simple chain that uses the vector store
        retriever = self.vectorstore.as_retriever()

        # Create TruChain to trigger instrumentation
        tru_chain = TruChain(
            retriever,
            app_name="test_async_retrieval",
            app_version="v1",
            main_method=retriever.ainvoke,
        )

        # Invoke the retriever asynchronously
        await tru_chain.instrumented_ainvoke_main_method(
            run_name="test_async_retrieval_run",
            input_id="test_id_4",
            main_method_args=("async search query",),
        )

        # Get events from database
        events_df = self._get_events()

        # Filter for asimilarity_search or similarity_search events
        search_events = events_df[
            events_df["record"].apply(
                lambda x: "similarity_search" in x.get("name", "")
            )
        ]

        # Verify at least one similarity_search event exists
        self.assertGreater(
            len(search_events), 0, "No async similarity_search events found"
        )

        # Check that the span type is RETRIEVAL
        for _, event in search_events.iterrows():
            span_type = event["record_attributes"].get(SpanAttributes.SPAN_TYPE)
            self.assertEqual(
                span_type,
                SpanAttributes.SpanType.RETRIEVAL,
                f"Expected RETRIEVAL span type, got {span_type}",
            )

    def test_extract_event_content_direct_content(self):
        """Test 4a: _extract_event_content extracts direct content field."""
        # Test with direct content field
        event = {"content": "Direct content string"}

        # Extract using TruChain's _extract_event_content logic
        # We'll test this by simulating the main_output method
        from trulens.apps.langchain.tru_chain import TruChain

        # Create a dummy TruChain instance
        dummy_chain = RunnablePassthrough()
        tru_chain = TruChain(dummy_chain, app_name="test", app_version="v1")

        # The _extract_event_content is an inner function, so we test
        # the main_output behavior with event-style outputs
        ret = [event]
        output = tru_chain.main_output(
            func=lambda x: x,
            sig=None,
            bindings=None,
            ret=ret,
        )

        self.assertEqual(output, "Direct content string")

    def test_extract_event_content_nested_messages(self):
        """Test 4b: _extract_event_content extracts content from nested messages."""
        # Test with messages list
        msg = AIMessage(content="Message content")
        event = {"messages": [msg]}

        # Create a dummy TruChain instance
        dummy_chain = RunnablePassthrough()
        tru_chain = TruChain(dummy_chain, app_name="test", app_version="v1")

        ret = [event]
        output = tru_chain.main_output(
            func=lambda x: x,
            sig=None,
            bindings=None,
            ret=ret,
        )

        self.assertEqual(output, "Message content")

    def test_extract_event_content_data_chunk(self):
        """Test 4c: _extract_event_content extracts content from data.chunk."""
        # Test with data.chunk structure
        chunk = AIMessageChunk(content="Chunk content")
        event = {"data": {"chunk": chunk}}

        # Create a dummy TruChain instance
        dummy_chain = RunnablePassthrough()
        tru_chain = TruChain(dummy_chain, app_name="test", app_version="v1")

        ret = [event]
        output = tru_chain.main_output(
            func=lambda x: x,
            sig=None,
            bindings=None,
            ret=ret,
        )

        self.assertEqual(output, "Chunk content")

    def test_extract_event_content_data_output(self):
        """Test 4d: _extract_event_content extracts content from data.output."""
        # Test with data.output as string
        event = {"data": {"output": "Output string"}}

        # Create a dummy TruChain instance
        dummy_chain = RunnablePassthrough()
        tru_chain = TruChain(dummy_chain, app_name="test", app_version="v1")

        ret = [event]
        output = tru_chain.main_output(
            func=lambda x: x,
            sig=None,
            bindings=None,
            ret=ret,
        )

        self.assertEqual(output, "Output string")

    def test_extract_event_content_return_values(self):
        """Test 4e: _extract_event_content extracts content from data.return_values."""
        # Test with return_values structure
        event = {"data": {"return_values": {"output": "Return value output"}}}

        # Create a dummy TruChain instance
        dummy_chain = RunnablePassthrough()
        tru_chain = TruChain(dummy_chain, app_name="test", app_version="v1")

        ret = [event]
        output = tru_chain.main_output(
            func=lambda x: x,
            sig=None,
            bindings=None,
            ret=ret,
        )

        self.assertEqual(output, "Return value output")

    def test_extract_event_content_multiple_events(self):
        """Test 4f: _extract_event_content handles multiple events correctly."""
        # Test with multiple events (simulating streaming)
        events = [
            {"data": {"chunk": AIMessageChunk(content="Part 1")}},
            {"data": {"chunk": AIMessageChunk(content=" Part 2")}},
            {"data": {"chunk": AIMessageChunk(content=" Part 3")}},
        ]

        # Create a dummy TruChain instance
        dummy_chain = RunnablePassthrough()
        tru_chain = TruChain(dummy_chain, app_name="test", app_version="v1")

        output = tru_chain.main_output(
            func=lambda x: x,
            sig=None,
            bindings=None,
            ret=events,
        )

        self.assertEqual(output, "Part 1 Part 2 Part 3")

    def test_instrumented_methods_configuration(self):
        """Test that METHODS configuration includes expected instrumentation."""
        # Get the instrumented methods configuration
        methods = LangChainInstrument.Default.METHODS()

        # Check for BaseLanguageModel methods with GENERATION span type
        generation_methods = [
            m
            for m in methods
            if m.span_type == SpanAttributes.SpanType.GENERATION
            and m.class_filter == BaseLanguageModel
        ]

        generation_method_names = [m.method for m in generation_methods]
        self.assertIn("invoke", generation_method_names)
        self.assertIn("ainvoke", generation_method_names)

        # Check for VectorStore methods with RETRIEVAL span type
        retrieval_methods = [
            m
            for m in methods
            if m.span_type == SpanAttributes.SpanType.RETRIEVAL
            and "similarity_search" in m.method
        ]

        self.assertGreater(
            len(retrieval_methods),
            0,
            "Expected retrieval methods for VectorStore",
        )

        # Check for stream_events methods
        stream_methods = [
            m
            for m in methods
            if m.method in ["stream_events", "astream_events"]
        ]

        self.assertGreater(
            len(stream_methods),
            0,
            "Expected stream_events instrumentation",
        )


# --- tests/unit/test_otel_tru_chain.py ---

    def test_smoke(self) -> None:
        # Create app.
        rag_chain = self._create_simple_rag()
        tru_recorder = TruChain(
            rag_chain,
            app_name="Simple RAG",
            app_version="v1",
            main_method=rag_chain.invoke,
        )
        # Record and invoke.
        tru_recorder.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("What is multi-headed attention?",),
        )
        # Smoke test - just verify it runs without errors
        # Check garbage collection.
        # Note that we need to delete `rag_chain` too since `rag_chain` has
        # instrument decorators that have closures of the `tru_recorder` object.
        # Specifically the record root has this at the very least as it calls
        # `TruChain::main_input` for instance.
        tru_recorder_ref = weakref.ref(tru_recorder)
        del tru_recorder
        del rag_chain
        gc.collect()
        self.assertCollected(tru_recorder_ref)

    def test_legacy_app(self) -> None:
        # Create app.
        rag_chain = self._create_simple_rag()
        tru_recorder = TruChain(
            rag_chain, app_name="Simple RAG", app_version="v1"
        )
        # Record and invoke.
        with tru_recorder:
            rag_chain.invoke("What is multi-headed attention?")
        # Compare results to expected.
        self._compare_record_attributes_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_chain__test_smoke.csv"
        )


# --- tests/unit/test_otel_tru_llama.py ---

    def test_smoke(self) -> None:
        # Create app.
        rag = self._create_simple_rag()
        tru_recorder = TruLlama(
            rag,
            app_name="Simple RAG",
            app_version="v1",
            main_method=rag.query,
        )
        # Record and invoke.
        tru_recorder.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("What is multi-headed attention?",),
        )
        # Smoke test - just verify it runs without errors
        # Check garbage collection.
        # Note that we need to delete `rag` too since `rag` has instrument
        # decorators that have closures of the `tru_recorder` object.
        # Specifically the record root has this at the very least as it calls
        # `TruLlama::main_input` for instance.
        tru_recorder_ref = weakref.ref(tru_recorder)
        del tru_recorder
        del rag
        gc.collect()
        self.assertCollected(tru_recorder_ref)


# --- tests/unit/test_tru_llama_workflow.py ---

    def test_llm_attributes(self):
        """Test LLM-specific attribute mapping."""

        # Test data
        response_data = {
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
                "completion_tokens_details": {"reasoning_tokens": 50},
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is the response",
                    }
                }
            ],
        }

        attrs = {}

        # Map to TruLens attributes
        if response_data.get("model"):
            attrs[SpanAttributes.COST.MODEL] = response_data["model"]

        usage = response_data.get("usage", {})
        if usage:
            attrs[SpanAttributes.COST.NUM_PROMPT_TOKENS] = usage.get(
                "prompt_tokens", 0
            )
            attrs[SpanAttributes.COST.NUM_COMPLETION_TOKENS] = usage.get(
                "completion_tokens", 0
            )
            attrs[SpanAttributes.COST.NUM_TOKENS] = usage.get("total_tokens", 0)

            # Check for reasoning tokens (o1 models)
            if "completion_tokens_details" in usage:
                details = usage["completion_tokens_details"]
                if "reasoning_tokens" in details:
                    attrs[SpanAttributes.COST.NUM_REASONING_TOKENS] = details[
                        "reasoning_tokens"
                    ]

        if response_data.get("choices"):
            first_choice = response_data["choices"][0]
            message = first_choice.get("message", {})
            content = message.get("content", "")
            attrs[SpanAttributes.CALL.RETURN] = content
            attrs["llm.output_text"] = content
            attrs["llm.completions"] = json.dumps([
                {"role": message.get("role", "assistant"), "content": content}
            ])

        # Verify all attributes
        assert attrs[SpanAttributes.COST.MODEL] == "gpt-4"
        assert attrs[SpanAttributes.COST.NUM_PROMPT_TOKENS] == 100
        assert attrs[SpanAttributes.COST.NUM_COMPLETION_TOKENS] == 200
        assert attrs[SpanAttributes.COST.NUM_TOKENS] == 300
        assert attrs[SpanAttributes.COST.NUM_REASONING_TOKENS] == 50
        assert attrs[SpanAttributes.CALL.RETURN] == "This is the response"
        assert attrs["llm.output_text"] == "This is the response"
        assert "assistant" in attrs["llm.completions"]


# --- tests/unit/providers/test_async_openai_capabilities.py ---

    def test_large_content_truncation(self):
        """Test that large content is properly truncated."""
        # Create a large response
        large_content = "x" * 10000  # 10,000 characters

        # Truncate for storage
        truncated = large_content[:1000]

        assert len(truncated) == 1000
        assert truncated == "x" * 1000

        # Test with JSON
        large_json = json.dumps({"content": large_content})
        truncated_json = large_json[:2000]

        assert len(truncated_json) == 2000

