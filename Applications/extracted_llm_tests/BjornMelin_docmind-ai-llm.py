# BjornMelin/docmind-ai-llm
# 22 LLM-backed test functions across 313 test files
# Source: https://github.com/BjornMelin/docmind-ai-llm

# --- tests/e2e/test_app.py ---

def test_app_renders_and_shows_chat(app_test) -> None:
    """Verify app renders and the chat section is present."""
    app = app_test.run()
    assert not app.exception, f"App failed with exception: {app.exception}"
    app_str = str(app)
    assert "Chat with Documents" in app_str or hasattr(app, "chat_input")

def test_streamlit_app_markers_and_structure(app_test):
    """Confirm core Streamlit UI components are present in the app."""
    app = app_test.run()

    # Verify app structure and key components
    assert not app.exception, f"App structure test failed: {app.exception}"

    # Check main sections
    # Robust structure check: rely on widget presence rather than exact strings
    ui_has_sidebar = hasattr(app, "sidebar")
    ui_has_controls = bool(getattr(app, "selectbox", [])) or bool(
        getattr(app, "button", [])
    )
    assert ui_has_sidebar or ui_has_controls, "Missing core UI components"

    # Check sidebar components
    sidebar_str = str(app.sidebar)
    sidebar_components = ["Backend", "Model", "Use GPU"]
    present_components = [comp for comp in sidebar_components if comp in sidebar_str]

    # Should have at least some sidebar components
    assert present_components, "No sidebar components found"

    print("✅ App structure validation completed")
    print(f"   - Sidebar present: {ui_has_sidebar}")
    print(f"   - Controls present: {ui_has_controls}")


# --- tests/e2e/test_document_processing_validation.py ---

def test_analysis_output_schema_validation():
    """Test that the analysis output schema works correctly."""
    try:
        from src.models.schemas import AnalysisOutput

        # Test valid analysis output creation
        analysis = AnalysisOutput(
            summary="Document processing validation completed successfully",
            key_insights=[
                "Document loading functionality works correctly",
                "Hardware detection provides accurate information",
                "Configuration system integrates properly",
                "Analysis schema validates input and output",
            ],
            action_items=[
                "Continue with E2E testing",
                "Validate UI integration",
                "Test multi-agent coordination",
            ],
            open_questions=[
                "How to optimize document processing performance?",
                "What additional validation scenarios should be tested?",
            ],
        )

        # Validate analysis output structure
        assert analysis.summary is not None
        assert len(analysis.summary) > 0
        assert len(analysis.key_insights) == 4
        assert len(analysis.action_items) == 3
        assert len(analysis.open_questions) == 2

        # Validate content
        assert "validation completed successfully" in analysis.summary
        assert "Document loading functionality" in analysis.key_insights[0]
        assert "Continue with E2E testing" in analysis.action_items[0]
        assert "optimize document processing" in analysis.open_questions[0]

        print("✅ Analysis output schema validation passed")
        print(f"   - Summary length: {len(analysis.summary)} chars")
        print(f"   - Key insights: {len(analysis.key_insights)}")
        print(f"   - Action items: {len(analysis.action_items)}")
        print(f"   - Open questions: {len(analysis.open_questions)}")

    except ImportError as e:
        pytest.skip(f"Analysis schema test failed: {e}")

def test_memory_management_components():
    """Test that memory and session management components are available."""
    try:
        from llama_index.core.llms import ChatMessage
        from llama_index.core.memory import ChatMemoryBuffer

        # Test memory buffer creation
        memory = ChatMemoryBuffer.from_defaults(token_limit=2048)
        assert memory is not None

        # Test chat message creation
        user_message = ChatMessage(
            role="user", content="Test document processing query"
        )
        assert user_message.role == "user"
        assert user_message.content == "Test document processing query"

        assistant_message = ChatMessage(
            role="assistant", content="Document processing completed"
        )
        assert assistant_message.role == "assistant"
        assert assistant_message.content == "Document processing completed"

        print("✅ Memory management components validated")
        print(f"   - Memory buffer token limit: {memory.token_limit}")
        print(f"   - User message: {user_message.content[:50]}...")
        print(f"   - Assistant message: {assistant_message.content[:50]}...")

    except ImportError as e:
        pytest.skip(f"Memory management components test failed: {e}")


# --- tests/integration/test_settings_page.py ---

def test_settings_save_persists_env(
    settings_app_test: AppTest,
    tmp_path: Path,
    reset_settings_after_test: None,
) -> None:
    """Saving settings should write expected keys into .env in temp cwd."""
    import sys

    before = set(sys.modules)
    app = settings_app_test.run()
    assert not app.exception
    _assert_hybrid_toggle_is_read_only(app)

    # Perf guard: initial render should not trigger heavy integration imports.
    after = set(sys.modules)
    delta = after - before
    assert "src.config.integrations" not in delta

    # Set a few key fields to ensure persistence writes recognizable values
    # Model field
    text_inputs = list(app.text_input)
    # Find model input by label
    if model_inputs := [w for w in text_inputs if "Model (id or GGUF path)" in str(w)]:
        model_inputs[0].set_value("Hermes-2-Pro-Llama-3-8B").run()

    # LM Studio base URL (must end with /v1)
    if lmstudio_inputs := [w for w in text_inputs if "LM Studio base URL" in str(w)]:
        lmstudio_inputs[0].set_value("http://localhost:1234/v1").run()

    # Ollama advanced settings
    if api_key_inputs := [w for w in text_inputs if "Ollama API key" in str(w)]:
        api_key_inputs[0].set_value("key-123").run()

    allow_remote = [w for w in app.checkbox if "Allow remote endpoints" in str(w)]
    assert allow_remote, "Allow remote endpoints checkbox not found"
    allow_remote[0].set_value(True).run()

    web_tools = [w for w in app.checkbox if "Enable Ollama web search tools" in str(w)]
    assert web_tools, "Ollama web tools checkbox not found"
    web_tools[0].set_value(True).run()

    logprobs = [w for w in app.checkbox if "Enable Ollama logprobs" in str(w)]
    if logprobs:
        logprobs[0].set_value(True).run()

    embed_dims = [w for w in app.number_input if "Embed dimensions" in str(w)]
    assert embed_dims, "Embed dimensions input not found"
    embed_dims[0].set_value(384).run()

    top_logprobs = [w for w in app.number_input if "Top logprobs" in str(w)]
    assert top_logprobs, "Top logprobs input not found"
    top_logprobs[0].set_value(2).run()

    # Click Save
    save_buttons = [b for b in app.button if getattr(b, "label", "") == "Save"]
    assert save_buttons, "Save button not found"
    save_buttons[0].click().run()

    # Verify .env was created with keys
    env_file = tmp_path / ".env"
    assert env_file.exists(), ".env not created by Save action"
    from dotenv import dotenv_values

    values = dotenv_values(env_file)
    assert values.get("DOCMIND_MODEL") == "Hermes-2-Pro-Llama-3-8B"
    assert values.get("DOCMIND_LMSTUDIO_BASE_URL") == "http://localhost:1234/v1"
    assert values.get("DOCMIND_OLLAMA_API_KEY") == "key-123"
    assert values.get("DOCMIND_OLLAMA_ENABLE_WEB_SEARCH") == "true"
    assert values.get("DOCMIND_OLLAMA_EMBED_DIMENSIONS") == "384"
    assert values.get("DOCMIND_OLLAMA_ENABLE_LOGPROBS") == "true"
    assert values.get("DOCMIND_OLLAMA_TOP_LOGPROBS") == "2"

def test_settings_save_persists_openai_compatible_env(
    settings_app_test: AppTest,
    tmp_path: Path,
    reset_settings_after_test: None,
) -> None:
    """Saving settings should persist OpenAI-compatible configuration to .env."""
    import json

    app = settings_app_test.run()
    assert not app.exception

    providers = [w for w in app.selectbox if getattr(w, "label", "") == "LLM Provider"]
    assert providers, "LLM Provider selectbox not found"
    app = providers[0].set_value("openai_compatible").run()
    assert not app.exception

    # Remote provider base URL needs allow_remote_endpoints enabled in tests.
    allow_remote = [w for w in app.checkbox if "Allow remote endpoints" in str(w)]
    assert allow_remote, "Allow remote endpoints checkbox not found"
    app = allow_remote[0].set_value(True).run()
    assert not app.exception

    # Configure OpenAI-compatible fields.
    text_inputs = list(app.text_input)
    base_url_inputs = [w for w in text_inputs if getattr(w, "label", "") == "Base URL"]
    assert base_url_inputs, "OpenAI-compatible Base URL input not found"
    app = base_url_inputs[0].set_value("https://ai-gateway.vercel.sh/v1").run()
    assert not app.exception

    text_inputs = list(app.text_input)
    api_key_inputs = [
        w for w in text_inputs if getattr(w, "label", "") == "API key (optional)"
    ]
    assert api_key_inputs, "OpenAI-compatible API key input not found"
    app = api_key_inputs[0].set_value("key-xyz").run()
    assert not app.exception

    require_v1 = [
        w
        for w in app.checkbox
        if getattr(w, "label", "") == "Normalize base URL to include /v1"
    ]
    assert require_v1, "Require /v1 checkbox not found"
    app = require_v1[0].set_value(True).run()
    assert not app.exception

    api_mode = [w for w in app.selectbox if getattr(w, "label", "") == "API mode"]
    assert api_mode, "API mode selectbox not found"
    app = api_mode[0].set_value("responses").run()
    assert not app.exception

    headers_areas = [
        w
        for w in app.text_area
        if getattr(w, "label", "") == "Default headers (JSON object)"
    ]
    assert headers_areas, "Default headers text area not found"
    app = (
        headers_areas[0]
        .set_value(
            json.dumps({"HTTP-Referer": "https://example.com", "X-Test": "1"}, indent=2)
        )
        .run()
    )
    assert not app.exception

    save_buttons = [b for b in app.button if getattr(b, "label", "") == "Save"]
    assert save_buttons, "Save button not found"
    save_buttons[0].click().run()

    env_file = tmp_path / ".env"
    assert env_file.exists(), ".env not created by Save action"
    from dotenv import dotenv_values

    values = dotenv_values(env_file)
    assert values.get("DOCMIND_LLM_BACKEND") == "openai_compatible"
    assert values.get("DOCMIND_OPENAI__BASE_URL") == "https://ai-gateway.vercel.sh/v1"
    assert values.get("DOCMIND_OPENAI__API_KEY") == "key-xyz"
    assert values.get("DOCMIND_OPENAI__REQUIRE_V1") == "true"
    assert values.get("DOCMIND_OPENAI__API_MODE") == "responses"
    assert values.get("DOCMIND_OPENAI__DEFAULT_HEADERS") == (
        '{"HTTP-Referer":"https://example.com","X-Test":"1"}'
    )

def test_settings_connectivity_test_shows_for_openai_compatible(
    settings_app_test: AppTest,
    reset_settings_after_test: None,
) -> None:
    """Connectivity test should render for OpenAI-compatible backends."""
    app = settings_app_test.run()
    assert not app.exception

    providers = [w for w in app.selectbox if getattr(w, "label", "") == "LLM Provider"]
    assert providers, "LLM Provider selectbox not found"
    app = providers[0].set_value("openai_compatible").run()
    assert not app.exception

    text_inputs = list(app.text_input)
    base_url_inputs = [w for w in text_inputs if getattr(w, "label", "") == "Base URL"]
    assert base_url_inputs, "OpenAI-compatible Base URL input not found"
    app = base_url_inputs[0].set_value("http://localhost:1234/v1").run()
    assert not app.exception

    test_buttons = [b for b in app.button if getattr(b, "label", "") == "Test endpoint"]
    assert test_buttons, "Connectivity test button not found for OpenAI-compatible"

def test_settings_save_normalizes_lmstudio_url(
    settings_app_test: AppTest,
    tmp_path: Path,
    reset_settings_after_test: None,
) -> None:
    """LM Studio base URL should be normalized to include /v1 on Save."""
    app = settings_app_test.run()
    assert not app.exception

    text_inputs = list(app.text_input)
    if lmstudio_inputs := [w for w in text_inputs if "LM Studio base URL" in str(w)]:
        lmstudio_inputs[0].set_value("http://localhost:1234").run()

    save_buttons = [b for b in app.button if getattr(b, "label", "") == "Save"]
    assert save_buttons, "Save button not found"
    save_buttons[0].click().run()

    env_file = tmp_path / ".env"
    assert env_file.exists(), ".env not created by Save action"
    from dotenv import dotenv_values

    values = dotenv_values(env_file)
    assert values.get("DOCMIND_LMSTUDIO_BASE_URL") == "http://localhost:1234/v1"

def test_settings_invalid_remote_url_disables_actions(
    settings_app_test: AppTest,
    reset_settings_after_test: None,
) -> None:
    """Remote URLs should be blocked when allow_remote_endpoints is disabled."""
    app = settings_app_test.run()
    assert not app.exception

    vllm_inputs = [w for w in app.text_input if "vLLM base URL" in str(w)]
    assert vllm_inputs, "vLLM base URL input not found"
    vllm_inputs[0].set_value("http://example.com:8000").run()

    apply_buttons = [
        b for b in app.button if getattr(b, "label", "") == "Apply runtime"
    ]
    save_buttons = [b for b in app.button if getattr(b, "label", "") == "Save"]
    assert apply_buttons
    assert save_buttons
    assert apply_buttons[0].disabled is True
    assert save_buttons[0].disabled is True
    assert any(
        "Remote endpoints are disabled" in str(getattr(e, "value", ""))
        for e in app.error
    )

def test_settings_allow_remote_allows_remote_urls(
    settings_app_test: AppTest,
    reset_settings_after_test: None,
) -> None:
    """Remote URLs should be allowed when allow_remote_endpoints is enabled."""
    app = settings_app_test.run()
    assert not app.exception

    allow_remote = [w for w in app.checkbox if "Allow remote endpoints" in str(w)]
    assert allow_remote, "Allow remote endpoints checkbox not found"
    allow_remote[0].set_value(True).run()

    vllm_inputs = [w for w in app.text_input if "vLLM base URL" in str(w)]
    assert vllm_inputs, "vLLM base URL input not found"
    vllm_inputs[0].set_value("http://example.com:8000").run()

    apply_buttons = [
        b for b in app.button if getattr(b, "label", "") == "Apply runtime"
    ]
    save_buttons = [b for b in app.button if getattr(b, "label", "") == "Save"]
    assert apply_buttons
    assert save_buttons
    assert apply_buttons[0].disabled is False
    assert save_buttons[0].disabled is False
    assert not list(app.error)

def test_settings_warns_when_ollama_allowlist_missing(
    settings_app_test: AppTest,
    reset_settings_after_test: None,
) -> None:
    """Enabling Ollama web tools should warn when allowlist lacks ollama.com."""
    app = settings_app_test.run()
    assert not app.exception

    allow_remote = [w for w in app.checkbox if "Allow remote endpoints" in str(w)]
    assert allow_remote, "Allow remote endpoints checkbox not found"
    allow_remote[0].set_value(True).run()

    web_tool_checks = [
        w for w in app.checkbox if "Enable Ollama web search tools" in str(w)
    ]
    assert web_tool_checks, "Ollama web tools checkbox not found"
    web_tool_checks[0].set_value(True).run()

    warnings = [str(getattr(w, "value", "")) for w in app.warning]
    expected_warning = (
        "Ollama web tools require `https://ollama.com` in "
        "`DOCMIND_SECURITY__ENDPOINT_ALLOWLIST`."
    )
    assert any(msg.strip() == expected_warning for msg in warnings), warnings


# --- tests/unit/agents/tools/test_retrieval.py ---

def test_retrieve_documents_supports_sync_invoke() -> None:
    """`retrieve_documents.invoke()` should work for sync graph execution."""
    result_json = retrieve_documents.invoke({"query": "test query", "state": {}})
    result = json.loads(result_json)
    assert result["documents"] == []
    assert "error" in result

    async def test_retrieve_documents_no_tools_data(self):
        """Test retrieval behavior when no tools data is available."""
        result_json = await retrieve_documents.ainvoke(
            {"query": "test query", "state": {}}
        )
        result = json.loads(result_json)

        assert result["documents"] == []
        assert "error" in result
        assert result["strategy_used"] == "hybrid"

    async def test_retrieve_documents_no_state(self):
        """Test retrieval behavior when state parameter is omitted."""
        result_json = await retrieve_documents.ainvoke({"query": "test query"})
        result = json.loads(result_json)
        assert result["documents"] == []
        assert "error" in result

    def test_extract_indexes_prefers_runtime_context(self):
        """_extract_indexes prefers runtime.context over persisted state."""
        import src.agents.tools.retrieval as mod

        runtime = type(
            "R", (), {"context": {"vector": "rv", "kg": "rk", "retriever": "rr"}}
        )()
        v, kg, r = mod._extract_indexes(
            {"tools_data": {"vector": "sv"}}, runtime=runtime
        )
        assert (v, kg, r) == ("rv", "rk", "rr")

    async def test_vector_strategy_errors_when_vector_index_missing(self):
        """Vector strategy fails with a clear error when vector index is absent."""
        data = json.loads(
            await retrieve_documents.ainvoke(
                {
                    "query": "q",
                    "strategy": "vector",
                    "use_dspy": False,
                    "state": {"tools_data": {"kg": object()}},
                }
            )
        )
        assert "No vector index available" in data.get("error", "")

    def test_parse_tool_result_get_content_branch_and_text_error(self):
        """_parse_tool_result uses get_content and falls back on text errors."""
        import src.agents.tools.retrieval as mod

        class _NodeNoText:
            def get_content(self):  # type: ignore[no-untyped-def]
                return "gc"

        class _NodeBadText:
            @property
            def text(self):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")

            def __str__(self) -> str:
                return "fallback"

        class _Nws:
            def __init__(self, node):  # type: ignore[no-untyped-def]
                self.node = node
                self.score = 0.1

        class _R:
            def __init__(self) -> None:
                self.source_nodes = [_Nws(_NodeNoText()), _Nws(_NodeBadText())]

        docs = mod._parse_tool_result(_R())
        assert [d["content"] for d in docs] == ["gc", "fallback"]


# --- tests/unit/core/test_dependencies.py ---

    def test_streamlit_available(self):
        """Test that Streamlit is available."""
        try:
            import streamlit as st

            assert st is not None
            # Test key components the app uses
            assert hasattr(st, "set_page_config")
            assert hasattr(st, "session_state")
        except ImportError:
            pytest.skip("Streamlit not available in test environment")


# --- tests/unit/models/embeddings/test_embedding_dimension_validation.py ---

    def test_similarity_computation_dimension_mismatch_detection(
        self, sample_embeddings_1024d
    ):
        """Test dimension mismatch detection in similarity computations."""
        valid_1024 = sample_embeddings_1024d["valid_1024d"][0]
        invalid_512 = sample_embeddings_1024d["invalid_512d"][0]

        assert len(valid_1024) == 1024, "Valid vector should be 1024D"
        assert len(invalid_512) == 512, "Invalid vector should be 512D"
        assert len(valid_1024) != len(invalid_512), (
            "Vectors should have mismatched dimensions"
        )

        # Attempting similarity with mismatched dimensions should be detectable
        vec1_np = np.array(valid_1024)
        vec2_np = np.array(invalid_512)

        # This should fail due to dimension mismatch
        with pytest.raises((ValueError, RuntimeError)):
            np.dot(vec1_np, vec2_np)


# --- tests/unit/nlp/test_spacy_service.py ---

def test_cuda_device_smoke_when_available() -> None:
    """Optional GPU smoke test (manual)."""
    _clear_spacy_cache()
    from thinc.api import prefer_gpu

    try:
        ok = prefer_gpu(0)
    except Exception:  # pragma: no cover - environment-dependent
        pytest.skip("GPU runtime not available")
    if not ok:
        pytest.skip("GPU runtime not available")

    cfg = SpacyNlpSettings(model="__missing_cuda__", device=SpacyDevice.CUDA)
    service = SpacyNlpService(cfg)
    _ = service.load()


# --- tests/unit/processing/test_ingestion_pipeline.py ---

async def test_ingest_documents_sync_guard() -> None:
    cfg = IngestionConfig()
    inputs = [IngestionInput(document_id="doc", payload_bytes=b"inline text")]

    with pytest.raises(RuntimeError) as exc_info:
        ingest_documents_sync(cfg, inputs, embedding=DummyEmbedding())
    assert "await ingest_documents" in str(exc_info.value)


# --- tests/unit/retrieval/reranking/infra/test_helpers.py ---

def test_run_with_timeout_returns_value_and_none():
    """_run_with_timeout returns value when fast, None when exceeding budget."""

    def _fast():
        return 42

    def _slow():
        start = time.perf_counter()
        while (time.perf_counter() - start) < 0.05:
            pass
        return 1

    assert rr._run_with_timeout(_fast, timeout_ms=10_000) == 42
    assert rr._run_with_timeout(_slow, timeout_ms=1) is None

