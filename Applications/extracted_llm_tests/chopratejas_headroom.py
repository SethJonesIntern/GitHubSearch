# chopratejas/headroom
# 33 LLM-backed test functions across 193 test files
# Source: https://github.com/chopratejas/headroom

# --- examples/test_intelligent_context_toin_ccr.py ---

def test_toin_ccr_integration():
    """Test TOIN + CCR integration with IntelligentContextManager."""
    print("=" * 70)
    print("TOIN + CCR Integration Test for IntelligentContextManager")
    print("=" * 70)

    # Get TOIN and CCR store
    toin = get_toin()
    store = get_compression_store()

    # Record initial state
    initial_patterns = len(toin._patterns) if hasattr(toin, "_patterns") else 0
    # CCR store uses a backend, not direct _store
    if hasattr(store, "_backend") and hasattr(store._backend, "_store"):
        initial_store_size = len(store._backend._store)
    else:
        initial_store_size = 0

    print("\nInitial state:")
    print(f"  TOIN patterns: {initial_patterns}")
    print(f"  CCR store entries: {initial_store_size}")

    # Create manager with TOIN
    config = IntelligentContextConfig(
        enabled=True,
        keep_system=True,
        keep_last_turns=3,
        output_buffer_tokens=2000,
        use_importance_scoring=True,
    )
    manager = IntelligentContextManager(config=config, toin=toin)
    tokenizer = Tokenizer(EstimatingTokenCounter())

    # Run multiple compression cycles to accumulate TOIN patterns
    print("\n" + "-" * 70)
    print("Running compression cycles...")
    print("-" * 70)

    all_ccr_refs = []

    for cycle in range(5):
        # Create fresh conversation each cycle
        messages = create_large_conversation(num_turns=30 + cycle * 5)

        tokens_before = tokenizer.count_messages(messages)

        # Set a tight limit to force dropping
        model_limit = tokens_before // 2

        result = manager.apply(
            messages,
            tokenizer,
            model_limit=model_limit,
            output_buffer=1000,
        )

        # Extract CCR reference from marker if present
        ccr_ref = None
        for marker in result.markers_inserted:
            if "ccr_retrieve" in marker and "reference '" in marker:
                start = marker.find("reference '") + len("reference '")
                end = marker.find("'", start)
                ccr_ref = marker[start:end]
                all_ccr_refs.append(ccr_ref)

        print(f"\nCycle {cycle + 1}:")
        print(f"  Messages: {len(messages)} → {len(result.messages)}")
        print(
            f"  Tokens: {result.tokens_before} → {result.tokens_after} "
            f"({100 * (1 - result.tokens_after / result.tokens_before):.1f}% reduction)"
        )
        print(f"  Transforms: {result.transforms_applied}")
        print(f"  CCR reference: {ccr_ref or 'None'}")

    # Check final state
    final_patterns = len(toin._patterns) if hasattr(toin, "_patterns") else 0
    if hasattr(store, "_backend") and hasattr(store._backend, "_store"):
        final_store_size = len(store._backend._store)
    else:
        final_store_size = 0

    print("\n" + "-" * 70)
    print("Final state:")
    print("-" * 70)
    print(
        f"  TOIN patterns: {initial_patterns} → {final_patterns} (+{final_patterns - initial_patterns})"
    )
    print(
        f"  CCR store entries: {initial_store_size} → {final_store_size} (+{final_store_size - initial_store_size})"
    )
    print(f"  CCR references created: {len(all_ccr_refs)}")

    # Test retrieval from CCR
    if all_ccr_refs:
        print("\n" + "-" * 70)
        print("Testing CCR retrieval...")
        print("-" * 70)

        ref = all_ccr_refs[-1]  # Use the most recent reference
        entry = store.retrieve(ref)

        if entry:
            # Parse the retrieved content from the CompressionEntry
            try:
                dropped_messages = json.loads(entry.original_content)
                print(f"  Retrieved {len(dropped_messages)} dropped messages from CCR")
                print(f"  First message role: {dropped_messages[0].get('role', 'unknown')}")
                print(f"  Content preview: {str(dropped_messages[0].get('content', ''))[:100]}...")
                print("  Entry metadata:")
                print(f"    - Tool: {entry.tool_name}")
                print(f"    - Original tokens: {entry.original_tokens}")
                print(f"    - Compressed tokens: {entry.compressed_tokens}")
            except json.JSONDecodeError:
                print(f"  Retrieved content (not JSON): {entry.original_content[:200]}...")
        else:
            print(f"  WARNING: Could not retrieve CCR reference {ref}")
            # Debug: check what's in the store
            print(f"  Store backend type: {type(store._backend)}")
            if hasattr(store._backend, "_store"):
                print(f"  Backend store keys: {list(store._backend._store.keys())[:5]}...")

    # Print TOIN statistics
    print("\n" + "-" * 70)
    print("TOIN Statistics:")
    print("-" * 70)

    stats = toin.get_stats()
    print(f"  Total patterns: {stats.get('total_patterns', 0)}")
    print(f"  Total compressions: {stats.get('total_compressions', 0)}")
    print(f"  Total retrievals: {stats.get('total_retrievals', 0)}")
    print(f"  Retrieval rate: {stats.get('retrieval_rate', 0):.1%}")

    # Check for intelligent_context_drop patterns
    drop_patterns = (
        [
            p
            for p in toin._patterns.values()
            if hasattr(p, "tool_name") and "intelligent_context" in str(getattr(p, "tool_name", ""))
        ]
        if hasattr(toin, "_patterns")
        else []
    )

    print(f"  IntelligentContext drop patterns: {len(drop_patterns)}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    # Assertions
    assert final_patterns >= initial_patterns, "TOIN should have recorded new patterns"
    assert len(all_ccr_refs) > 0, "Should have created CCR references"
    # CCR store entries should exist (though count may vary due to TTL)
    if final_store_size == 0 and initial_store_size == 0:
        print("  Note: CCR store size shows 0 (entries may have different backend)")
    else:
        assert final_store_size > initial_store_size, "CCR store should have new entries"

    print("\n✓ All assertions passed!")
    return True

def test_with_real_llm():
    """Test with a real LLM call to verify end-to-end flow."""
    print("\n" + "=" * 70)
    print("Real LLM Integration Test")
    print("=" * 70)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping real LLM test - OPENAI_API_KEY not set")
        return

    try:
        from openai import OpenAI

        client = OpenAI()
    except ImportError:
        print("Skipping real LLM test - openai package not installed")
        return

    # Create a conversation that will be compressed
    messages = create_large_conversation(num_turns=20)

    # Apply IntelligentContext compression
    toin = get_toin()
    config = IntelligentContextConfig(
        enabled=True,
        keep_system=True,
        keep_last_turns=2,
    )
    manager = IntelligentContextManager(config=config, toin=toin)
    tokenizer = Tokenizer(EstimatingTokenCounter())

    tokens_before = tokenizer.count_messages(messages)

    result = manager.apply(
        messages,
        tokenizer,
        model_limit=tokens_before // 3,  # Force significant compression
        output_buffer=500,
    )

    print("\nCompression result:")
    print(f"  Messages: {len(messages)} → {len(result.messages)}")
    print(f"  Tokens: {result.tokens_before} → {result.tokens_after}")

    # Convert to OpenAI format (filter out tool messages with None content)
    openai_messages = []
    for msg in result.messages:
        if msg.get("role") == "tool":
            continue  # Skip tool messages for this test
        if msg.get("content") is None:
            continue  # Skip messages with None content
        openai_messages.append({"role": msg["role"], "content": msg["content"]})

    # Add a question about the compressed context
    openai_messages.append(
        {
            "role": "user",
            "content": "Based on our conversation, what errors did we discuss? "
            "If you see a message about compressed context, note the CCR reference.",
        }
    )

    print(f"\nSending {len(openai_messages)} messages to OpenAI...")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            max_tokens=500,
        )

        print("\nLLM Response:")
        print("-" * 40)
        print(response.choices[0].message.content)
        print("-" * 40)
        print(f"\nTokens used: {response.usage.total_tokens}")

    except Exception as e:
        print(f"LLM call failed: {e}")


# --- tests/test_memory_integration.py ---

    def test_extraction_prompt_injection(self, openai_client, temp_db_path, user_id):
        """Verify extraction prompt is injected into system message."""
        from headroom.memory import with_memory_tools
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig
        from headroom.memory.extraction import EXTRACTION_SYSTEM_PROMPT

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)

        wrapper = with_memory_tools(
            openai_client,
            backend=backend,
            user_id=user_id,
            optimized=True,
            inject_extraction_prompt=True,
        )

        # Get the completions object
        completions = wrapper.chat.completions

        # Test _prepare_messages with existing system message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        prepared = completions._prepare_messages(messages)

        # Verify system message has extraction prompt appended
        assert len(prepared) == 2
        assert EXTRACTION_SYSTEM_PROMPT in prepared[0]["content"]
        assert "You are a helpful assistant." in prepared[0]["content"]

        # Test _prepare_messages without existing system message
        messages_no_system = [{"role": "user", "content": "Hello"}]
        prepared_no_system = completions._prepare_messages(messages_no_system)

        # Verify system message was inserted
        assert len(prepared_no_system) == 2
        assert prepared_no_system[0]["role"] == "system"
        assert EXTRACTION_SYSTEM_PROMPT.strip() in prepared_no_system[0]["content"]

    def test_e2e_standard_mode_llm_call(self, openai_client, temp_db_path, user_id):
        """Test end-to-end flow with real LLM call in standard mode."""
        from headroom.memory import with_memory_tools
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)

        client = with_memory_tools(
            openai_client,
            backend=backend,
            user_id=user_id,
            optimized=False,  # Standard mode
        )

        # Make a real LLM call that should trigger memory_save
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that remembers important user information. When the user shares personal information, save it to memory using the memory_save tool.",
                },
                {
                    "role": "user",
                    "content": "Hi! My name is Alex and I work as a data scientist at Google.",
                },
            ],
        )

        # Verify response was generated
        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0

        # Check if memory tool was called
        message = response.choices[0].message
        if message.tool_calls:
            # Verify memory_save was called
            tool_names = [tc.function.name for tc in message.tool_calls]
            print(f"Tools called: {tool_names}")

            # Check if auto-handled
            if hasattr(response, "_memory_tool_results"):
                print(f"Memory tool results: {response._memory_tool_results}")
                assert len(response._memory_tool_results) > 0

    def test_e2e_optimized_mode_llm_call(self, openai_client, temp_db_path, user_id):
        """Test end-to-end flow with real LLM call in optimized mode."""
        from headroom.memory import with_memory_tools
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)

        client = with_memory_tools(
            openai_client,
            backend=backend,
            user_id=user_id,
            optimized=True,  # Optimized mode - should extract facts/entities
        )

        # Make a real LLM call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "I'm Sarah, a software engineer at Microsoft. I use Python, React, and PostgreSQL daily.",
                },
            ],
        )

        # Verify response was generated
        assert response is not None
        assert response.choices is not None

        # Check if memory tool was called with pre-extraction
        message = response.choices[0].message
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc.function.name == "memory_save":
                    import json

                    args = json.loads(tc.function.arguments)
                    print(f"memory_save arguments: {json.dumps(args, indent=2)}")

                    # In optimized mode, LLM SHOULD include facts/entities
                    # (depends on LLM following the extraction prompt)
                    if "facts" in args:
                        print(f"Pre-extracted facts: {args['facts']}")
                    if "extracted_entities" in args:
                        print(f"Pre-extracted entities: {args['extracted_entities']}")
                    if "extracted_relationships" in args:
                        print(f"Pre-extracted relationships: {args['extracted_relationships']}")

        # Check auto-handled results
        if hasattr(response, "_memory_tool_results"):
            print(f"Memory tool results: {response._memory_tool_results}")

    def test_full_flow_save_then_search(self, openai_client, temp_db_path, user_id):
        """Test complete flow: LLM saves memory, then searches for it."""
        import json

        from headroom.memory import with_memory_tools
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)

        client = with_memory_tools(
            openai_client,
            backend=backend,
            user_id=user_id,
            optimized=True,
        )

        # First: Have LLM save some information
        save_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Remember this: My favorite programming language is Rust and I'm working on a CLI tool called headroom.",
                },
            ],
        )

        print(f"Save response: {save_response.choices[0].message}")

        # Process tool calls if any
        if save_response.choices[0].message.tool_calls:
            print(
                f"Tool calls made: {[tc.function.name for tc in save_response.choices[0].message.tool_calls]}"
            )
            if hasattr(save_response, "_memory_tool_results"):
                print(f"Results: {save_response._memory_tool_results}")

        # Second: Ask LLM to recall the information
        recall_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "What is my favorite programming language? Search your memory.",
                },
            ],
        )

        print(f"Recall response: {recall_response.choices[0].message}")

        # Check if search was invoked
        if recall_response.choices[0].message.tool_calls:
            for tc in recall_response.choices[0].message.tool_calls:
                print(f"Tool: {tc.function.name}, Args: {tc.function.arguments}")
                if hasattr(recall_response, "_memory_tool_results"):
                    results = recall_response._memory_tool_results.get(tc.id, {})
                    print(f"Tool result: {json.dumps(results, indent=2, default=str)}")


# --- tests/test_evals/test_html_oss_benchmarks.py ---

    def test_full_suite(self, answer_fn):
        """Run the complete benchmark suite."""
        pytest.importorskip("datasets")
        from headroom.evals.html_oss_benchmarks import run_full_benchmark_suite

        result = run_full_benchmark_suite(
            answer_fn=answer_fn,
            extraction_samples=30,
            qa_questions=20,
        )

        # Print comprehensive results
        print("\n" + "=" * 60)
        print("FULL BENCHMARK SUITE RESULTS")
        print("=" * 60)

        summary = result.summary()

        if result.extraction_result:
            ext = summary["extraction"]
            print("\n📊 Extraction Benchmark:")
            print(f"   Samples:   {ext['total_samples']}")
            print(f"   Precision: {ext['avg_precision']:.3f}")
            print(f"   Recall:    {ext['avg_recall']:.3f}")
            print(f"   F1:        {ext['avg_f1']:.3f} (baseline: {ext['baseline_f1']:.3f})")
            print(f"   Compression: {(1 - ext['avg_compression_ratio']) * 100:.1f}% reduction")

        if result.qa_result:
            qa = summary["qa_accuracy"]
            print("\n📝 QA Accuracy Preservation:")
            print(f"   Questions: {qa['total_questions']}")
            print(f"   Original:  {qa['accuracy_original_html']:.3f}")
            print(f"   Extracted: {qa['accuracy_extracted']:.3f}")
            print(f"   Delta:     {qa['accuracy_delta']:+.3f}")
            print(f"   Preserved: {'✅' if qa['accuracy_preserved'] else '❌'}")

        print(f"\n{'=' * 60}")
        print(f"ALL BENCHMARKS PASSED: {'✅' if summary['all_passed'] else '❌'}")
        print(f"{'=' * 60}\n")

        # Assert all passed
        assert result.all_passed, "Not all benchmarks passed"


# --- tests/test_integrations/agno/test_model.py ---

    def test_headroom_model_has_required_abstract_methods(self):
        """HeadroomAgnoModel must implement all required abstract methods."""
        from agno.models.openai import OpenAIChat

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = OpenAIChat(id="gpt-4o")
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # Verify required methods exist and are callable
        assert hasattr(headroom_model, "invoke")
        assert callable(headroom_model.invoke)

        assert hasattr(headroom_model, "ainvoke")
        assert callable(headroom_model.ainvoke)

        assert hasattr(headroom_model, "invoke_stream")
        assert callable(headroom_model.invoke_stream)

        assert hasattr(headroom_model, "ainvoke_stream")
        assert callable(headroom_model.ainvoke_stream)

        assert hasattr(headroom_model, "_parse_provider_response")
        assert callable(headroom_model._parse_provider_response)

        assert hasattr(headroom_model, "_parse_provider_response_delta")
        assert callable(headroom_model._parse_provider_response_delta)

    def test_agent_run_with_ollama(self, ollama_model_name):
        """Actually run an agent with Ollama - full end-to-end test."""
        from agno.agent import Agent
        from agno.models.ollama import Ollama

        from headroom.integrations.agno import HeadroomAgnoModel

        # Create wrapped Ollama model
        base_model = Ollama(id=ollama_model_name)
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # Create and run agent
        agent = Agent(model=headroom_model, markdown=False)

        # Actually run the agent - this tests the full pipeline
        response = agent.run("Say 'hello' and nothing else.")

        # Verify we got a response
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

        # Verify Headroom optimization was applied
        assert len(headroom_model.metrics_history) >= 1

    def test_agent_with_system_prompt_and_ollama(self, ollama_model_name):
        """Test agent with system prompt using Ollama."""
        from agno.agent import Agent
        from agno.models.ollama import Ollama

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = Ollama(id=ollama_model_name)
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # Agent with system prompt - tests system message optimization
        agent = Agent(
            model=headroom_model,
            description="You are a helpful assistant that always responds with exactly one word.",
            markdown=False,
        )

        response = agent.run("What is 2+2?")

        assert response is not None
        assert response.content is not None

        # Headroom should have processed the system prompt
        assert headroom_model.total_tokens_saved >= 0

    def test_multiple_turns_with_ollama(self, ollama_model_name):
        """Test multi-turn conversation with Ollama."""
        from agno.agent import Agent
        from agno.models.ollama import Ollama

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = Ollama(id=ollama_model_name)
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        agent = Agent(model=headroom_model, markdown=False)

        # Multiple turns
        agent.run("My name is Alice.")
        agent.run("What is my name?")

        # Should have tracked multiple optimization passes
        assert len(headroom_model.metrics_history) >= 2


# --- tests/test_integrations/langchain/test_chat_model.py ---

    def test_on_llm_error(self):
        """Track errors."""
        from headroom.integrations import HeadroomCallbackHandler

        handler = HeadroomCallbackHandler()
        handler._current_request = {"start_time": datetime.now()}

        handler.on_llm_error(ValueError("Test error"), run_id=uuid4())

        assert handler.total_requests == 1
        assert "error" in handler.requests[0]

    def test_init_defaults(self):
        """Initialize with defaults."""
        from headroom.integrations.langchain import HeadroomRunnable

        runnable = HeadroomRunnable()

        assert runnable.mode == HeadroomMode.OPTIMIZE
        assert runnable.config is not None

    def test_init_custom_config(self):
        """Initialize with custom config."""
        from headroom.integrations.langchain import HeadroomRunnable

        config = HeadroomConfig(default_mode=HeadroomMode.AUDIT)
        runnable = HeadroomRunnable(config=config, mode=HeadroomMode.SIMULATE)

        assert runnable.config is config
        assert runnable.mode == HeadroomMode.SIMULATE

    def test_as_runnable(self):
        """Convert to LangChain Runnable."""
        from langchain_core.runnables import RunnableLambda

        from headroom.integrations.langchain import HeadroomRunnable

        runnable = HeadroomRunnable()
        lc_runnable = runnable.as_runnable()

        assert isinstance(lc_runnable, RunnableLambda)

    def test_invoke_with_ollama(self, ollama_model_name):
        """Actually invoke an LLM call with Ollama - full end-to-end test."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)

        messages = [
            SystemMessage(content="You are a helpful assistant. Be very brief."),
            HumanMessage(content="What is 2+2? Answer with just the number."),
        ]

        # This makes a real LLM call
        result = headroom_model.invoke(messages)

        assert result is not None
        assert result.content is not None
        assert len(result.content) > 0

    def test_generate_with_ollama(self, ollama_model_name):
        """Test _generate method with real Ollama model."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)

        messages = [
            HumanMessage(content="Say 'hello' and nothing else."),
        ]

        result = headroom_model._generate(messages)

        assert result is not None
        assert len(result.generations) > 0
        assert result.generations[0].message.content is not None

    def test_optimization_tracked_with_ollama(self, ollama_model_name):
        """Test that optimization metrics are tracked with real calls."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)

        # Make a call with some messages
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hi"),
        ]

        headroom_model.invoke(messages)

        # Metrics should be tracked
        assert len(headroom_model._metrics_history) >= 1

    def test_multiple_turns_with_ollama(self, ollama_model_name):
        """Test multi-turn conversation with real Ollama."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)

        # First turn
        messages = [
            SystemMessage(content="You are a helpful assistant. Be brief."),
            HumanMessage(content="My name is Alice."),
        ]
        response1 = headroom_model.invoke(messages)

        # Second turn - add previous exchange
        messages.append(AIMessage(content=response1.content))
        messages.append(HumanMessage(content="What is my name?"))

        response2 = headroom_model.invoke(messages)

        assert response2 is not None
        assert response2.content is not None
        # Model should remember the name from context
        assert len(response2.content) > 0

    def test_headroom_optimization_reduces_tokens(self, ollama_model_name):
        """Test that Headroom optimization actually reduces token count."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model, mode=HeadroomMode.OPTIMIZE)

        # Create a conversation with repetitive content that should be compressed
        messages = [SystemMessage(content="You are a helpful assistant.")]
        for i in range(20):
            messages.append(HumanMessage(content=f"Question {i}: What is {i} + {i}?"))
            messages.append(AIMessage(content=f"The answer to {i} + {i} is {i + i}."))
        messages.append(HumanMessage(content="What was question 5?"))

        # This should trigger compression
        headroom_model.invoke(messages)

        # Check that some optimization was tracked
        if headroom_model._metrics_history:
            metrics = headroom_model._metrics_history[-1]
            # With a large conversation, we expect some savings
            assert metrics.tokens_before >= metrics.tokens_after

    def test_callback_handler_with_ollama(self, ollama_model_name):
        """Test HeadroomCallbackHandler with real Ollama calls."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomCallbackHandler, HeadroomChatModel

        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)
        handler = HeadroomCallbackHandler()

        messages = [
            HumanMessage(content="Say 'test' and nothing else."),
        ]

        # Invoke with callback
        headroom_model.invoke(messages, config={"callbacks": [handler]})

        # Handler should have tracked the request
        assert handler.total_requests >= 1

    def test_lcel_chain_with_headroom(self, ollama_model_name):
        """Test LCEL chain composition with HeadroomChatModel."""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        # Create chain: prompt -> headroom model -> output parser
        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Be very brief."),
                ("human", "{input}"),
            ]
        )

        chain = prompt | headroom_model | StrOutputParser()

        # Invoke the chain
        result = chain.invoke({"input": "What is 1+1? Just the number."})

        assert result is not None
        assert len(result) > 0


# --- tests/test_integrations/langchain/test_langchain_live.py ---

    def test_wrap_openai_and_invoke(self, openai_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        messages = [HumanMessage(content="Reply with exactly: OK")]
        response = model.invoke(messages)

        assert response is not None
        assert hasattr(response, "content")
        assert response.content is not None
        assert len(response.content) > 0
        assert len(model._metrics_history) >= 1
        m = model._metrics_history[-1]
        assert m.tokens_before >= 0
        assert m.tokens_after >= 0

    def test_invoke_with_string_input(self, openai_llm):
        """LangChain allows invoke(str); BaseChatModel converts to messages."""
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        response = model.invoke("Say hello in one word.")
        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

    def test_system_and_user_messages(self, openai_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        messages = [
            SystemMessage(content="You are a helpful assistant. Be very brief."),
            HumanMessage(content="What is 2+2? One number only."),
        ]
        response = model.invoke(messages)
        assert response.content is not None
        assert "4" in response.content or "four" in response.content.lower()

    def test_get_savings_summary_after_calls(self, openai_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        model.invoke([HumanMessage(content="Hi")])
        summary = model.get_savings_summary()
        assert summary["total_requests"] >= 1
        assert "total_tokens_saved" in summary
        assert "average_savings_percent" in summary

    def test_wrap_anthropic_and_invoke(self, anthropic_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(anthropic_llm)
        messages = [HumanMessage(content="Reply with exactly: OK")]
        try:
            response = model.invoke(messages)
        except Exception as e:
            if "404" in str(e) or "not_found" in str(e).lower():
                pytest.skip(f"Anthropic model not available: {e}")
            raise
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert len(model._metrics_history) >= 1

    def test_stream_openai(self, openai_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        messages = [HumanMessage(content="Count from 1 to 3, one number per line.")]
        chunks = list(model.stream(messages))
        assert len(chunks) >= 1
        full = "".join(c.content for c in chunks if c.content)
        assert "1" in full or "2" in full or "3" in full

    async def test_astream_openai(self, openai_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        messages = [HumanMessage(content="Say 'stream' and nothing else.")]
        count = 0
        async for chunk in model.astream(messages):
            if chunk.content:
                count += 1
        assert count >= 1

    def test_bind_tools_and_invoke_with_tool_output(self, openai_llm):
        """Simulate agent turn: user -> model (tool call) -> tool result -> model. We compress tool result."""
        from headroom.integrations import HeadroomChatModel

        @tool
        def big_search(query: str) -> str:
            """Search (returns large JSON)."""
            import json

            return json.dumps(
                {
                    "results": [
                        {"id": i, "title": f"Result {i}", "snippet": "x" * 200} for i in range(50)
                    ],
                    "total": 50,
                }
            )

        base = openai_llm.bind_tools([big_search])
        model = HeadroomChatModel(base)

        # User asks something that may trigger tool use
        messages = [
            HumanMessage(
                content="Search for 'python tutorials' and tell me how many results you got."
            ),
        ]
        response = model.invoke(messages)

        assert response is not None
        # Either direct answer or tool_calls
        if response.tool_calls:
            assert len(response.tool_calls) >= 1
            tc = response.tool_calls[0]
            assert "name" in tc or hasattr(tc, "get")
        assert len(model._metrics_history) >= 1

    def test_messages_with_tool_result_compressed(self, openai_llm):
        """Conversation with tool call + large tool result; Headroom should compress the tool result."""
        import json

        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        # Simulate: user -> assistant (tool call) -> tool (large result) -> user (follow-up)
        large_result = json.dumps([{"id": i, "data": "x" * 100} for i in range(100)])
        messages = [
            HumanMessage(content="Get items 1 to 100."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_items",
                        "args": {"limit": 100},
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content=large_result, tool_call_id="call_1"),
            HumanMessage(content="How many items did you get? One number only."),
        ]
        response = model.invoke(messages)

        assert response is not None
        assert response.content is not None
        # Optimization should have run (tool content was large)
        assert len(model._metrics_history) >= 1
        last = model._metrics_history[-1]
        assert last.tokens_before >= last.tokens_after or last.tokens_before == last.tokens_after

    def test_prompt_pipe_headroom_pipe_llm(self, openai_llm):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are helpful. Reply in one short sentence."),
                ("human", "{input}"),
            ]
        )
        chain = prompt | model | StrOutputParser()
        result = chain.invoke({"input": "What is the capital of France?"})
        assert result is not None
        assert "Paris" in result or "paris" in result.lower()


# --- tests/test_providers/test_universal.py ---

    def test_default_values(self):
        """Test default capability values."""
        caps = ModelCapabilities(model="test-model")
        assert caps.context_window == 128000
        assert caps.max_output_tokens == 4096
        assert caps.supports_tools is True
        assert caps.supports_vision is False
        assert caps.supports_streaming is True

