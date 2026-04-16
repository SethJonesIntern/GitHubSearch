# comet-ml/opik
# 322 LLM-backed test functions across 458 test files
# Source: https://github.com/comet-ml/opik

# --- sdks/opik_optimizer/tests/e2e/optimizers/multimodal/test_multimodal_prompt.py ---

def test_multimodal_prompt(
    optimizer_class: type,
    setup_driving_hazard_dataset: opik.Dataset,
) -> None:
    """
    Test that optimizers can handle multimodal prompts with text and images.

    This test verifies:
    1. Optimization completes with multimodal content
    2. Multimodal structure (content parts) is preserved
    3. Image URL placeholders are preserved
    4. Text content can be optimized

    """
    # Create multimodal prompt
    original_prompt = create_multimodal_prompt()

    # Get multimodal dataset (created once per session via conftest)
    dataset = setup_driving_hazard_dataset

    # Create optimizer with minimal config
    config = create_optimizer_config(optimizer_class, verbose=0)
    optimizer = optimizer_class(**config)
    results = run_optimizer(
        optimizer_class=optimizer_class,
        optimizer=optimizer,
        prompt=original_prompt,
        dataset=dataset,
        metric=hazard_metric,
        parameter_space=get_parameter_space(),
        n_samples=1,
        max_trials=1,
    )

    # Validate results structure
    assert results.optimizer == optimizer_class.__name__, (
        f"Expected {optimizer_class.__name__}, got {results.optimizer}"
    )

    # Get optimized prompt - handle both ChatPrompt and list returns
    optimized_prompt = results.prompt

    if isinstance(optimized_prompt, ChatPrompt):
        assert_multimodal_structure_preserved(original_prompt, optimized_prompt)
    elif isinstance(optimized_prompt, list):
        # Some algorithms may return a list[dict] for prompts.
        # Re-wrap to use shared assertions and keep behavior identical.
        wrapped = ChatPrompt(messages=optimized_prompt)
        assert_multimodal_structure_preserved(original_prompt, wrapped)
    else:
        pytest.fail(f"Unexpected prompt type: {type(optimized_prompt)}")

    print(f"✅ {optimizer_class.__name__}: Multimodal prompt - PASSED")


# --- sdks/opik_optimizer/tests/e2e/optimizers/multi_prompt/test_multi_prompt_with_agent.py ---

def test_multi_prompt_with_agent(
    optimizer_class: type,
    tiny_dataset: Dataset,
    multi_prompt_agent: MultiPromptTestAgent,
) -> None:
    """
    Test that optimizers can optimize multiple prompts using a custom agent.

    This test verifies:
    1. Optimization completes with dict of prompts
    2. All prompts in the dict are returned
    3. Prompts maintain valid structure
    4. Agent can still execute with optimized prompts
    """
    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable required")

    # Skip GEPA if not installed
    if optimizer_class == GepaOptimizer:
        import importlib.util

        if importlib.util.find_spec("gepa") is None:
            pytest.skip("gepa package not installed")

    # Create multi-prompt dict
    original_prompts = create_multi_prompt_dict()

    agent = multi_prompt_agent
    dataset = tiny_dataset

    # Create optimizer with minimal config
    config = create_optimizer_config(
        optimizer_class,
        verbose=0,
    )
    optimizer = optimizer_class(**config)

    results = run_optimizer(
        optimizer_class=optimizer_class,
        optimizer=optimizer,
        prompt=original_prompts,
        dataset=dataset,
        metric=levenshtein_metric,
        agent=agent,
        parameter_space=get_parameter_space(),
        n_samples=1,
        max_trials=1,
    )

    # Validate results structure
    assert results.optimizer == optimizer_class.__name__, (
        f"Expected {optimizer_class.__name__}, got {results.optimizer}"
    )

    # Get optimized prompts
    optimized_prompts = results.prompt

    # Handle both single ChatPrompt and dict returns
    if isinstance(optimized_prompts, dict):
        # Verify all original prompt keys are present
        for name in original_prompts:
            assert name in optimized_prompts, (
                f"Prompt '{name}' missing from optimized results"
            )

            optimized = optimized_prompts[name]
            assert isinstance(optimized, ChatPrompt), (
                f"Optimized prompt '{name}' should be ChatPrompt, got {type(optimized)}"
            )

            # Verify prompt has valid messages
            messages = optimized.get_messages()
            assert len(messages) > 0, f"Optimized prompt '{name}' should have messages"
            for msg in messages:
                assert "role" in msg, f"Message in '{name}' should have 'role' field"
                assert "content" in msg, (
                    f"Message in '{name}' should have 'content' field"
                )
    else:
        # Single prompt returned (some optimizers may do this)
        assert isinstance(optimized_prompts, ChatPrompt), (
            f"Optimized result should be ChatPrompt or dict, got {type(optimized_prompts)}"
        )
        messages = optimized_prompts.get_messages()
        assert len(messages) > 0, "Optimized prompt should have messages"

    print(f"✅ {optimizer_class.__name__}: Multi-prompt with agent - PASSED")


# --- sdks/opik_optimizer/tests/unit/metrics/test_total_span_cost.py ---

    def test_uses_existing_total_cost_when_available(self) -> None:
        """Test that existing total_cost is used instead of calculating"""
        metric = SpanCost()
        span = make_span(
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            total_cost=0.05,  # Pre-calculated cost
        )

        result = metric.score(task_span=span)

        assert_cost_result(
            result=result,
            expected_cost=0.05,
            processed_span_count=1,
            total_prompt_tokens=100,
            total_completion_tokens=50,
        )

    def test_skips_spans_with_zero_tokens(self) -> None:
        """Spans with 0 prompt and 0 completion tokens should be ignored."""
        metric = SpanCost()
        span = make_span(
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )

        result = metric.score(task_span=span)

        assert_cost_result(result=result, expected_cost=0.0, processed_span_count=0)


# --- sdks/python/tests/e2e/test_agent_config.py ---

def test_prompt_field_and_trace_metadata__happyflow(
    opik_client: opik.Opik,
    project_name: str,
):
    """Prompt-typed and ChatPrompt-typed fields survive the roundtrip with the correct
    class; field access inside a tracked function injects agent_configuration into
    trace and span metadata."""

    prompt_name = f"e2e-prompt-{uuid.uuid4().hex[:8]}"
    chat_prompt_name = f"e2e-chat-prompt-{uuid.uuid4().hex[:8]}"

    prompt_v1 = opik_client.create_prompt(
        name=prompt_name, prompt="Hello v1", project_name=project_name
    )
    chat_prompt_v1 = opik_client.create_chat_prompt(
        name=chat_prompt_name,
        messages=[{"role": "user", "content": "Hi v1"}],
        project_name=project_name,
    )

    class PromptConfig(opik.Config):
        system_prompt: Prompt
        chat_template: ChatPrompt
        temperature: float

    opik_client.create_config(
        PromptConfig(
            system_prompt=prompt_v1,
            chat_template=chat_prompt_v1,
            temperature=0.3,
        ),
        project_name=project_name,
    )

    get_global_registry().clear()

    id_storage = {}

    @opik.track(project_name=project_name)
    def run():
        cfg = opik_client.get_or_create_config(
            fallback=PromptConfig(
                system_prompt=prompt_v1,
                chat_template=chat_prompt_v1,
                temperature=0.0,
            ),
            project_name=project_name,
            version="latest",
        )
        id_storage["trace_id"] = opik_context.get_current_trace_data().id
        id_storage["span_id"] = opik_context.get_current_span_data().id
        id_storage["system_prompt"] = cfg.system_prompt
        id_storage["system_prompt_version_id"] = cfg.system_prompt.version_id
        id_storage["chat_template"] = cfg.chat_template
        id_storage["chat_template_version_id"] = cfg.chat_template.version_id
        _ = cfg.temperature
        return cfg

    run()
    opik.flush_tracker()

    # Prompt field roundtrip — must come back as Prompt, not ChatPrompt.
    assert isinstance(id_storage["system_prompt"], Prompt)
    assert not isinstance(id_storage["system_prompt"], ChatPrompt)
    assert id_storage["system_prompt_version_id"] == prompt_v1.version_id

    # ChatPrompt field roundtrip — must come back as ChatPrompt, not plain Prompt.
    assert isinstance(id_storage["chat_template"], ChatPrompt)
    assert id_storage["chat_template_version_id"] == chat_prompt_v1.version_id

    expected_meta = {
        "_blueprint_id": ANY_BUT_NONE,
        "blueprint_version": ANY_BUT_NONE,
        "values": ANY_DICT,
    }
    verifiers.verify_trace(
        opik_client=opik_client,
        trace_id=id_storage["trace_id"],
        metadata={"agent_configuration": ANY_DICT.containing(expected_meta)},
    )
    verifiers.verify_span(
        opik_client=opik_client,
        span_id=id_storage["span_id"],
        trace_id=id_storage["trace_id"],
        parent_span_id=None,
        metadata={"agent_configuration": ANY_DICT.containing(expected_meta)},
    )


# --- sdks/python/tests/e2e/test_attachments_client.py ---

def test_attachments_client__get_attachment_list_for_trace__happyflow(
    opik_client: opik.Opik, attachment_data_file
):
    trace_id = id_helpers.generate_id()

    file_name = os.path.basename(attachment_data_file.name)
    attachment = Attachment(
        data=attachment_data_file.name,
        file_name=file_name,
        content_type="application/octet-stream",
    )

    opik_client.trace(
        id=trace_id,
        name="test-trace-with-attachment",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        attachments=[attachment],
    )

    opik_client.flush()

    attachments_client = opik_client.get_attachment_client()

    synchronization.wait_for_done(
        lambda: len(
            attachments_client.get_attachment_list(
                project_name=OPIK_E2E_TESTS_PROJECT_NAME,
                entity_id=trace_id,
                entity_type="trace",
            )
        )
        > 0,
        timeout=30,
    )

    attachments_list = attachments_client.get_attachment_list(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        entity_id=trace_id,
        entity_type="trace",
    )
    assert len(attachments_list) == 1
    assert attachments_list[0].file_name == file_name
    assert attachments_list[0].mime_type == "application/octet-stream"

def test_attachments_client__download_attachment_for_trace__happyflow(
    opik_client: opik.Opik, attachment_data_file
):
    trace_id = id_helpers.generate_id()

    file_name = os.path.basename(attachment_data_file.name)
    attachment = Attachment(
        data=attachment_data_file.name,
        file_name=file_name,
        content_type="text/plain",
    )

    opik_client.trace(
        id=trace_id,
        name="test-trace-download-attachment",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        attachments=[attachment],
    )

    opik_client.flush()
    attachments_client = opik_client.get_attachment_client()

    synchronization.wait_for_done(
        lambda: len(
            attachments_client.get_attachment_list(
                project_name=OPIK_E2E_TESTS_PROJECT_NAME,
                entity_id=trace_id,
                entity_type="trace",
            )
        )
        > 0,
        timeout=30,
    )

    attachment_data = attachments_client.download_attachment(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        entity_type="trace",
        entity_id=trace_id,
        file_name=file_name,
        mime_type="text/plain",
    )
    downloaded_content = b"".join(attachment_data)

    # Read the original file content to compare
    attachment_data_file.seek(0)
    expected_content = attachment_data_file.read()
    assert downloaded_content == expected_content

def test_attachments_client__get_attachment_list_for_span__happyflow(
    opik_client: opik.Opik, attachment_data_file
):
    span_id = id_helpers.generate_id()
    trace_id = id_helpers.generate_id()

    file_name = os.path.basename(attachment_data_file.name)
    attachment = Attachment(
        data=attachment_data_file.name,
        file_name=file_name,
        content_type="application/octet-stream",
    )

    opik_client.trace(
        id=trace_id,
        name="test-trace-for-span",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )
    opik_client.span(
        id=span_id,
        trace_id=trace_id,
        name="test-span-with-attachment",
        attachments=[attachment],
    )

    opik_client.flush()

    attachments_client = opik_client.get_attachment_client()

    synchronization.wait_for_done(
        lambda: len(
            attachments_client.get_attachment_list(
                project_name=OPIK_E2E_TESTS_PROJECT_NAME,
                entity_id=span_id,
                entity_type="span",
            )
        )
        > 0,
        timeout=30,
    )

    attachments_list = attachments_client.get_attachment_list(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        entity_id=span_id,
        entity_type="span",
    )
    assert len(attachments_list) == 1
    assert attachments_list[0].file_name == file_name
    assert attachments_list[0].mime_type == "application/octet-stream"

def test_attachments_client__upload_attachment_for_trace__happyflow(
    opik_client: opik.Opik, attachment_data_file
):
    """Test uploading an attachment for a trace."""
    trace_id = id_helpers.generate_id()

    opik_client.trace(
        id=trace_id,
        name="test-trace-for-upload",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )

    opik_client.flush()

    attachments_client = opik_client.get_attachment_client()

    file_name = os.path.basename(attachment_data_file.name)
    attachments_client.upload_attachment(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        entity_type="trace",
        entity_id=trace_id,
        file_path=attachment_data_file.name,
        file_name=file_name,
        mime_type="application/octet-stream",
    )

    verifiers.verify_attachments(
        opik_client=opik_client,
        entity_type="trace",
        entity_id=trace_id,
        attachments={
            file_name: Attachment(
                data=attachment_data_file.name,
                file_name=file_name,
                content_type="application/octet-stream",
            )
        },
        data_sizes={file_name: ATTACHMENT_FILE_SIZE},
    )

def test_attachments_client__upload_attachment_for_span__happyflow(
    opik_client: opik.Opik, attachment_data_file
):
    """Test uploading an attachment for a span."""
    span_id = id_helpers.generate_id()
    trace_id = id_helpers.generate_id()

    opik_client.trace(
        id=trace_id,
        name="test-trace-for-span-upload",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )
    opik_client.span(
        id=span_id,
        trace_id=trace_id,
        name="test-span-for-upload",
    )

    opik_client.flush()

    attachments_client = opik_client.get_attachment_client()

    file_name = os.path.basename(attachment_data_file.name)
    attachments_client.upload_attachment(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        entity_type="span",
        entity_id=span_id,
        file_path=attachment_data_file.name,
        file_name=file_name,
        mime_type="application/octet-stream",
    )

    verifiers.verify_attachments(
        opik_client=opik_client,
        entity_type="span",
        entity_id=span_id,
        attachments={
            file_name: Attachment(
                data=attachment_data_file.name,
                file_name=file_name,
                content_type="application/octet-stream",
            )
        },
        data_sizes={file_name: ATTACHMENT_FILE_SIZE},
    )


# --- sdks/python/tests/e2e/test_attachments_extraction.py ---

def test_extraction__trace_with_end_time__extracts_attachments_from_input(
    opik_client: opik.Opik,
):
    """Test that traces with end_time has attachments extracted from the input field."""
    trace_id = id_helpers.generate_id()

    # Create a trace with end_time and base64-encoded images in input
    opik_client.trace(
        id=trace_id,
        name="test-trace-extraction-input",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        input={
            "image1": _create_base64_url("image/png", constants.PNG_BASE64),
            "image2": _create_base64_url("image/jpeg", constants.JPEG_BASE64),
            "text": "regular text field",
        },
        end_time=datetime_helpers.local_timestamp(),
    )

    opik_client.flush()

    # Verify attachments were extracted and uploaded
    expected_sizes = [
        len(base64.b64decode(constants.PNG_BASE64)),
        len(base64.b64decode(constants.JPEG_BASE64)),
    ]
    verifiers.verify_auto_extracted_attachments(
        opik_client=opik_client,
        entity_type="trace",
        entity_id=trace_id,
        expected_sizes=expected_sizes,
    )

def test_extraction__trace_without_end_time__does_not_extract_attachments(
    opik_client: opik.Opik,
):
    """Test that traces without end_time does NOT have attachments extracted."""
    trace_id = id_helpers.generate_id()

    # Create a trace WITHOUT calling end() - no end_time set
    opik_client.trace(
        id=trace_id,
        name="test-trace-no-extraction",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        input={
            "image": _create_base64_url("image/png", constants.PNG_BASE64),
        },
    )
    # Note: NOT calling trace.end()

    opik_client.flush()

    # Wait a bit to ensure processing has completed
    time.sleep(2)

    # Verify NO attachments were extracted
    attachments_client = opik_client.get_attachment_client()

    attachment_list = attachments_client.get_attachment_list(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        entity_id=trace_id,
        entity_type="trace",
    )

    assert len(attachment_list) == 0, (
        f"Expected no attachments, but found {len(attachment_list)}"
    )

def test_extraction__trace_with_end_time__extracts_attachments_from_output(
    opik_client: opik.Opik,
):
    """Test that traces with end_time has attachments extracted from the output field."""
    trace_id = id_helpers.generate_id()

    opik_client.trace(
        id=trace_id,
        name="test-trace-extraction-output",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        input={"prompt": "generate an image"},
        output={
            "result_image": _create_base64_url("image/png", constants.PNG_BASE64),
            "result_pdf": _create_base64_url("application/pdf", constants.PDF_BASE64),
        },
        end_time=datetime_helpers.local_timestamp(),
    )

    opik_client.flush()

    # Verify attachments were extracted and uploaded
    expected_sizes = [
        len(base64.b64decode(constants.PNG_BASE64)),
        len(base64.b64decode(constants.PDF_BASE64)),
    ]
    verifiers.verify_auto_extracted_attachments(
        opik_client=opik_client,
        entity_type="trace",
        entity_id=trace_id,
        expected_sizes=expected_sizes,
    )

def test_extraction__trace_with_end_time__extracts_attachments_from_metadata(
    opik_client: opik.Opik,
):
    """Test that traces with end_time has attachments extracted from the metadata field."""
    trace_id = id_helpers.generate_id()

    opik_client.trace(
        id=trace_id,
        name="test-trace-extraction-metadata",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        metadata={
            "screenshot": _create_base64_url("image/png", constants.PNG_BASE64),
            "version": "1.0",
        },
        end_time=datetime_helpers.local_timestamp(),
    )

    opik_client.flush()

    # Verify attachments were extracted and uploaded
    expected_sizes = [len(base64.b64decode(constants.PNG_BASE64))]
    verifiers.verify_auto_extracted_attachments(
        opik_client=opik_client,
        entity_type="trace",
        entity_id=trace_id,
        expected_sizes=expected_sizes,
    )

def test_extraction__trace_with_end_time__extracts_from_all_fields(
    opik_client: opik.Opik,
):
    """Test extraction from input, output, and metadata simultaneously."""
    trace_id = id_helpers.generate_id()

    opik_client.trace(
        id=trace_id,
        name="test-trace-extraction-all-fields",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        input={
            "input_img": _create_base64_url("image/png", constants.PNG_BASE64),
        },
        output={
            "output_img": _create_base64_url("image/jpeg", constants.JPEG_BASE64),
        },
        metadata={
            "meta_gif": _create_base64_url("image/gif", constants.GIF89_BASE64),
        },
        end_time=datetime_helpers.local_timestamp(),
    )

    opik_client.flush()

    # Verify attachments were extracted and uploaded
    expected_sizes = [
        len(base64.b64decode(constants.PNG_BASE64)),
        len(base64.b64decode(constants.JPEG_BASE64)),
        len(base64.b64decode(constants.GIF89_BASE64)),
    ]
    verifiers.verify_auto_extracted_attachments(
        opik_client=opik_client,
        entity_type="trace",
        entity_id=trace_id,
        expected_sizes=expected_sizes,
    )

def test_extraction__span_with_end_time__extracts_attachments(
    opik_client: opik.Opik,
):
    """Test that spans with end_time has attachments extracted."""
    trace_id = id_helpers.generate_id()
    span_id = id_helpers.generate_id()

    # Create trace first
    trace = opik_client.trace(
        id=trace_id,
        name="test-trace-for-span-extraction",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )

    # Create a span with end_time and attachments
    trace.span(
        id=span_id,
        name="test-span-extraction",
        input={
            "image": _create_base64_url("image/png", constants.PNG_BASE64),
        },
        output={
            "result": _create_base64_url("image/webp", constants.WEBP_BASE64),
        },
        metadata={
            "meta_gif": _create_base64_url("image/gif", constants.GIF89_BASE64),
        },
        end_time=datetime_helpers.local_timestamp(),
    )

    opik_client.flush()

    # Verify attachments were extracted and uploaded
    expected_sizes = [
        len(base64.b64decode(constants.PNG_BASE64)),
        len(base64.b64decode(constants.WEBP_BASE64)),
        len(base64.b64decode(constants.GIF89_BASE64)),
    ]
    verifiers.verify_auto_extracted_attachments(
        opik_client=opik_client,
        entity_type="span",
        entity_id=span_id,
        expected_sizes=expected_sizes,
    )

def test_extraction__span_without_end_time__does_not_extract_attachments(
    opik_client: opik.Opik,
):
    """Test that spans without end_time does NOT have attachments extracted."""
    trace_id = id_helpers.generate_id()
    span_id = id_helpers.generate_id()

    # Create trace first
    trace = opik_client.trace(
        id=trace_id,
        name="test-trace-for-span-no-extraction",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )

    # Create a span WITHOUT an end_time set
    trace.span(
        id=span_id,
        name="test-span-no-extraction",
        input={
            "image": _create_base64_url("image/png", constants.PNG_BASE64),
        },
    )

    opik_client.flush()

    # Wait a bit to ensure processing has completed
    time.sleep(2)

    # Verify NO attachments were extracted
    attachments_client = opik_client.get_attachment_client()

    attachment_list = attachments_client.get_attachment_list(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        entity_id=span_id,
        entity_type="span",
    )

    assert len(attachment_list) == 0, (
        f"Expected no attachments, but found {len(attachment_list)}"
    )

def test_extraction__trace_update__extracts_attachments(
    opik_client: opik.Opik,
):
    """Test that trace updates have attachments extracted (updates are always processed)."""
    trace_id = id_helpers.generate_id()

    # Create an initial trace without attachments
    trace = opik_client.trace(
        id=trace_id,
        name="test-trace-update-extraction",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        input={"prompt": "initial input"},
    )

    opik_client.flush()

    # Update the trace with attachment data
    trace.update(
        output={
            "result_image": _create_base64_url("image/png", constants.PNG_BASE64),
        }
    )

    opik_client.flush()

    # Verify attachments were extracted and uploaded
    expected_sizes = [len(base64.b64decode(constants.PNG_BASE64))]
    verifiers.verify_auto_extracted_attachments(
        opik_client=opik_client,
        entity_type="trace",
        entity_id=trace_id,
        expected_sizes=expected_sizes,
    )

def test_extraction__span_update__extracts_attachments(
    opik_client: opik.Opik,
):
    """Test that span updates have attachments extracted (updates are always processed)."""
    trace_id = id_helpers.generate_id()
    span_id = id_helpers.generate_id()

    # Create trace
    trace = opik_client.trace(
        id=trace_id,
        name="test-trace-for-span-update",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )

    # Create an initial span without attachments
    span = trace.span(
        id=span_id,
        name="test-span-update-extraction",
        input={"data": "initial"},
    )

    opik_client.flush()

    # Update the span with attachment data
    span.update(
        output={
            "chart": _create_base64_url("image/svg+xml", constants.SVG_BASE64),
        }
    )

    opik_client.flush()

    # Verify attachments were extracted and uploaded
    expected_sizes = [len(base64.b64decode(constants.SVG_BASE64))]
    verifiers.verify_auto_extracted_attachments(
        opik_client=opik_client,
        entity_type="span",
        entity_id=span_id,
        expected_sizes=expected_sizes,
    )

def test_extraction__various_file_types__all_extracted(
    opik_client: opik.Opik,
):
    """Test extraction of various file types (PNG, JPEG, PDF, GIF, WebP, SVG, JSON)."""
    trace_id = id_helpers.generate_id()

    opik_client.trace(
        id=trace_id,
        name="test-trace-various-types",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        input={
            "png": _create_base64_url("image/png", constants.PNG_BASE64),
            "jpeg": _create_base64_url("image/jpeg", constants.JPEG_BASE64),
            "pdf": _create_base64_url("application/pdf", constants.PDF_BASE64),
            "gif": _create_base64_url("image/gif", constants.GIF89_BASE64),
            "webp": _create_base64_url("image/webp", constants.WEBP_BASE64),
            "svg": _create_base64_url("image/svg+xml", constants.SVG_BASE64),
            "json": _create_base64_url("application/json", constants.JSON_BASE64),
        },
        end_time=datetime_helpers.local_timestamp(),
    )

    opik_client.flush()

    # Verify attachments were extracted and uploaded
    expected_sizes = [
        len(base64.b64decode(constants.PNG_BASE64)),
        len(base64.b64decode(constants.JPEG_BASE64)),
        len(base64.b64decode(constants.PDF_BASE64)),
        len(base64.b64decode(constants.GIF89_BASE64)),
        len(base64.b64decode(constants.WEBP_BASE64)),
        len(base64.b64decode(constants.SVG_BASE64)),
        len(base64.b64decode(constants.JSON_BASE64)),
    ]
    verifiers.verify_auto_extracted_attachments(
        opik_client=opik_client,
        entity_type="trace",
        entity_id=trace_id,
        expected_sizes=expected_sizes,
    )

def test_extraction__backend_reinjects_extracted_attachments(
    opik_client: opik.Opik,
):
    """Test that backend reinjects extracted attachments."""
    trace_id = id_helpers.generate_id()
    span_id = id_helpers.generate_id()

    # Create a trace with end_time and base64-encoded images in input
    trace_input = {
        "image1": _create_base64_url("image/png", constants.PNG_BASE64),
        "text": "regular text field",
    }
    trace = opik_client.trace(
        id=trace_id,
        name="test-trace-backend_reinjects_extracted_attachments",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        input=trace_input,
        end_time=datetime_helpers.local_timestamp(),
    )

    # Create a span with end_time and attachments
    span_input = {
        "image": _create_base64_url("image/png", constants.PNG_BASE64),
    }
    trace.span(
        id=span_id,
        name="test-span--backend_reinjects_extracted_attachments",
        input=span_input,
        end_time=datetime_helpers.local_timestamp(),
    )

    opik_client.flush()

    #
    # Verify attachments were extracted and uploaded for trace and span
    #
    expected_sizes = [
        len(base64.b64decode(constants.PNG_BASE64)),
    ]
    verifiers.verify_auto_extracted_attachments(
        opik_client=opik_client,
        entity_type="trace",
        entity_id=trace_id,
        expected_sizes=expected_sizes,
    )

    expected_sizes = [
        len(base64.b64decode(constants.PNG_BASE64)),
    ]
    verifiers.verify_auto_extracted_attachments(
        opik_client=opik_client,
        entity_type="span",
        entity_id=span_id,
        expected_sizes=expected_sizes,
    )

    #
    # Verify trace and span returned by backend has extracted attachments injected back into
    #

    # Verify trace
    verifiers.verify_trace(
        opik_client=opik_client,
        trace_id=trace.id,
        name="test-trace-backend_reinjects_extracted_attachments",
        input=trace_input,
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )

    # Verify span
    verifiers.verify_span(
        opik_client=opik_client,
        span_id=span_id,
        parent_span_id=None,
        trace_id=trace_id,
        name="test-span--backend_reinjects_extracted_attachments",
        input=span_input,
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )

def test_extraction__input_as_top_level_list(
    opik_client: opik.Opik,
):
    """Test that top-level lists are extracted and uploaded as separate attachments."""
    trace_id = id_helpers.generate_id()
    span_id = id_helpers.generate_id()

    # Create trace first
    trace = opik_client.trace(
        id=trace_id,
        name="test-trace-for-attachment-list-extraction",
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )

    # Create a span with end_time and a top-level list in the input with attachments
    data = [
        {"image": constants.JPEG_BASE64},
        {"pdf": constants.PDF_BASE64},
    ]

    trace.span(
        id=span_id,
        name="test-span-extraction",
        input=data,
        end_time=datetime_helpers.local_timestamp(),
    )

    opik_client.flush()

    # Verify attachments were extracted and uploaded
    expected_sizes = [
        len(base64.b64decode(constants.JPEG_BASE64)),
        len(base64.b64decode(constants.PDF_BASE64)),
    ]
    verifiers.verify_auto_extracted_attachments(
        opik_client=opik_client,
        entity_type="span",
        entity_id=span_id,
        expected_sizes=expected_sizes,
    )


# --- sdks/python/tests/e2e/test_cli_import_export.py ---

    def test_export_import_traces_happy_flow(
        self,
        opik_client: opik.Opik,
        source_project_name: str,
        test_data_dir: Path,
    ) -> None:
        """Test the complete export/import flow for traces."""
        # Step 1: Prepare test data
        self._create_test_traces(opik_client, source_project_name)

        # Verify traces were created
        traces = opik_client.search_traces(project_name=source_project_name)
        print(f"Found {len(traces)} traces in project {source_project_name}")
        for trace in traces:
            print(f"Trace: {trace.id} - {trace.name}")
        assert len(traces) >= 1, "Expected at least 1 trace to be created"

        # Step 2: Export traces using direct function call
        export_project_by_name(
            name=source_project_name,
            workspace="default",
            output_path=str(test_data_dir),
            max_results=None,
            filter_string=None,
            force=False,
            debug=False,
            format="json",
            api_key=None,
        )

        # Check if the directory was created
        # New CLI structure: default/projects/{project_name}/ for traces
        project_dir = test_data_dir / "default" / "projects" / source_project_name
        assert project_dir.exists(), f"Export directory not found: {project_dir}"

        print(f"Export directory created: {project_dir}")
        print(f"Directory contents: {list(project_dir.iterdir())}")

        trace_files = list(project_dir.glob("trace_*.json"))
        assert len(trace_files) >= 1, (
            f"Expected trace files, found: {list(project_dir.glob('*'))}"
        )

        # Verify trace file content
        with open(trace_files[0], "r") as f:
            trace_data = json.load(f)

        assert "trace" in trace_data
        assert "spans" in trace_data
        assert "downloaded_at" in trace_data
        assert trace_data["project_name"] == source_project_name

        # Step 3: Import traces using direct function call
        source_dir = test_data_dir / "default" / "projects"
        stats = import_projects_from_directory(
            client=opik_client,
            source_dir=source_dir,
            dry_run=False,
            name_pattern=None,
            debug=False,
            recreate_experiments_flag=False,
        )

        # Verify import succeeded
        assert stats.get("projects", 0) >= 1, (
            "Expected at least 1 project to be imported"
        )

    def test_export_import_datasets_happy_flow(
        self,
        opik_client: opik.Opik,
        source_project_name: str,
        test_data_dir: Path,
    ) -> None:
        """Test the complete export/import flow for datasets."""
        # Step 1: Prepare test data
        dataset_name = self._create_test_dataset(opik_client)

        # Verify dataset was created
        datasets = opik_client.get_datasets(max_results=100)
        assert len(datasets) >= 1, "Expected at least 1 dataset to be created"

        # Step 2: Export datasets using direct function call
        export_dataset_by_name(
            name=dataset_name,
            workspace="default",
            output_path=str(test_data_dir),
            max_results=None,
            force=False,
            debug=False,
            format="json",
            api_key=None,
        )

        # Verify export files were created
        # New CLI structure: default/datasets/ for datasets
        datasets_dir = test_data_dir / "default" / "datasets"
        assert datasets_dir.exists(), f"Export directory not found: {datasets_dir}"

        dataset_files = list(datasets_dir.glob("dataset_*.json"))
        assert len(dataset_files) >= 1, (
            f"Expected dataset files, found: {list(datasets_dir.glob('*'))}"
        )

        # Verify dataset file content
        with open(dataset_files[0], "r") as f:
            dataset_data = json.load(f)

        assert "name" in dataset_data
        assert "items" in dataset_data
        assert "downloaded_at" in dataset_data

        # Step 3: Import datasets using direct function call
        source_dir = test_data_dir / "default" / "datasets"
        stats = import_datasets_from_directory(
            client=opik_client,
            source_dir=source_dir,
            dry_run=False,
            name_pattern=None,
            debug=False,
        )

        # Verify import succeeded
        assert stats.get("datasets", 0) >= 1, (
            "Expected at least 1 dataset to be imported"
        )

    def test_export_import_prompts_happy_flow(
        self,
        opik_client: opik.Opik,
        source_project_name: str,
        test_data_dir: Path,
    ) -> None:
        """Test the complete export/import flow for prompts."""
        # Step 1: Prepare test data
        prompt_name = self._create_test_prompt(opik_client)

        # Verify prompt was created
        prompts = opik_client.search_prompts()
        prompt_names = [p.name for p in prompts]
        assert prompt_name in prompt_names, (
            f"Expected prompt {prompt_name} to be created"
        )

        # Step 2: Export prompts using direct function call
        export_prompt_by_name(
            name=prompt_name,
            workspace="default",
            output_path=str(test_data_dir),
            max_results=None,
            force=False,
            debug=False,
            format="json",
            api_key=None,
        )

        # Verify export files were created
        # New CLI structure: default/prompts/ for prompts
        prompts_dir = test_data_dir / "default" / "prompts"
        assert prompts_dir.exists(), f"Export directory not found: {prompts_dir}"

        prompt_files = list(prompts_dir.glob("prompt_*.json"))
        assert len(prompt_files) >= 1, (
            f"Expected prompt files, found: {list(prompts_dir.glob('*'))}"
        )

        # Verify prompt file content
        with open(prompt_files[0], "r") as f:
            prompt_data = json.load(f)

        assert "name" in prompt_data
        assert "current_version" in prompt_data
        assert "history" in prompt_data
        assert "downloaded_at" in prompt_data

        # Step 3: Import prompts using direct function call
        source_dir = test_data_dir / "default" / "prompts"
        stats = import_prompts_from_directory(
            client=opik_client,
            source_dir=source_dir,
            dry_run=False,
            name_pattern=None,
            debug=False,
        )

        # Verify import succeeded
        assert stats.get("prompts", 0) >= 1, "Expected at least 1 prompt to be imported"

        # Verify prompt was correctly imported to backend
        imported_prompts = opik_client.search_prompts()
        imported_prompt_names = [p.name for p in imported_prompts]
        assert prompt_name in imported_prompt_names, (
            f"Expected prompt {prompt_name} to be imported"
        )

        # Get the imported prompt and verify its content
        imported_prompt = next(p for p in imported_prompts if p.name == prompt_name)
        verifiers.verify_prompt_version(
            imported_prompt,
            name=prompt_name,
        )

    def test_export_import_all_data_types_happy_flow(
        self,
        opik_client: opik.Opik,
        source_project_name: str,
        test_data_dir: Path,
    ) -> None:
        """Test the complete export/import flow for all data types."""
        # Step 1: Prepare test data (minimal)
        self._create_test_traces(opik_client, source_project_name)
        dataset_name = self._create_test_dataset(opik_client)
        prompt_name = self._create_test_prompt(opik_client)

        # Step 2: Export all data types with limited results using direct function calls
        # Export projects (traces)
        export_project_by_name(
            name=source_project_name,
            workspace="default",
            output_path=str(test_data_dir),
            max_results=10,  # Limit to 10 traces
            filter_string=None,
            force=False,
            debug=False,
            format="json",
            api_key=None,
        )

        # Export datasets with limit
        export_dataset_by_name(
            name=dataset_name,
            workspace="default",
            output_path=str(test_data_dir),
            max_results=5,  # Limit to 5 datasets
            force=False,
            debug=False,
            format="json",
            api_key=None,
        )

        # Export prompts with limit
        export_prompt_by_name(
            name=prompt_name,
            workspace="default",
            output_path=str(test_data_dir),
            max_results=5,  # Limit to 5 prompt versions
            force=False,
            debug=False,
            format="json",
            api_key=None,
        )

        # Verify export files were created
        project_dir = test_data_dir / "default" / "projects" / source_project_name
        datasets_dir = test_data_dir / "default" / "datasets"
        prompts_dir = test_data_dir / "default" / "prompts"

        assert project_dir.exists(), f"Export directory not found: {project_dir}"
        assert datasets_dir.exists(), f"Export directory not found: {datasets_dir}"
        assert prompts_dir.exists(), f"Export directory not found: {prompts_dir}"

        # Check for all file types
        trace_files = list(project_dir.glob("trace_*.json"))
        dataset_files = list(datasets_dir.glob("dataset_*.json"))
        prompt_files = list(prompts_dir.glob("prompt_*.json"))

        assert len(trace_files) >= 1, "Expected trace files"
        assert len(dataset_files) >= 1, "Expected dataset files"
        assert len(prompt_files) >= 1, "Expected prompt files"

        # Step 3: Import all data types using direct function calls
        # Import projects (traces)
        projects_stats = import_projects_from_directory(
            client=opik_client,
            source_dir=test_data_dir / "default" / "projects",
            dry_run=False,
            name_pattern=None,
            debug=False,
            recreate_experiments_flag=False,
        )
        assert projects_stats.get("projects", 0) >= 1, (
            "Expected projects to be imported"
        )

        # Import datasets
        datasets_stats = import_datasets_from_directory(
            client=opik_client,
            source_dir=test_data_dir / "default" / "datasets",
            dry_run=False,
            name_pattern=None,
            debug=False,
        )
        assert datasets_stats.get("datasets", 0) >= 1, (
            "Expected datasets to be imported"
        )

        # Import prompts
        prompts_stats = import_prompts_from_directory(
            client=opik_client,
            source_dir=test_data_dir / "default" / "prompts",
            dry_run=False,
            name_pattern=None,
            debug=False,
        )
        assert prompts_stats.get("prompts", 0) >= 1, "Expected prompts to be imported"

    def test_import_dry_run(
        self,
        opik_client: opik.Opik,
        source_project_name: str,
        test_data_dir: Path,
    ) -> None:
        """Test import with dry run option."""
        # Create test data and export it using direct function call
        self._create_test_traces(opik_client, source_project_name)

        export_project_by_name(
            name=source_project_name,
            workspace="default",
            output_path=str(test_data_dir),
            max_results=None,
            filter_string=None,
            force=False,
            debug=False,
            format="json",
            api_key=None,
        )

        # Test dry run import using direct function call
        source_dir = test_data_dir / "default" / "projects"

        # Count traces before dry run
        traces_before = opik_client.search_traces(project_name=source_project_name)
        count_before = len(traces_before)

        _ = import_projects_from_directory(
            client=opik_client,
            source_dir=source_dir,
            dry_run=True,  # Dry run mode
            name_pattern=None,
            debug=False,
            recreate_experiments_flag=False,
        )

        # Verify dry run reported it would import but didn't actually import
        # (In dry run, stats may show what WOULD be imported, but no actual import occurs)

        # Count traces after dry run - should be the same
        traces_after = opik_client.search_traces(project_name=source_project_name)
        count_after = len(traces_after)

        # Dry run should not create any new traces
        assert count_after == count_before, (
            f"Dry run should not modify data: had {count_before} traces, now have {count_after}"
        )

    def test_cli_subprocess_validation(
        self, opik_client: opik.Opik, source_project_name: str, test_data_dir: Path
    ) -> None:
        """Test that CLI commands work via subprocess (validates CLI interface)."""
        # This test validates the actual CLI interface by using subprocess
        # All other tests call Python functions directly for speed

        # Create test data
        self._create_test_traces(opik_client, source_project_name)

        # Test export via CLI subprocess
        export_cmd = [
            "export",
            "default",
            "project",
            source_project_name,
            "--path",
            str(test_data_dir),
        ]

        result = self._run_cli_command(export_cmd)
        assert result.returncode == 0, f"CLI export failed: {result.stderr}"

        # Verify export worked
        project_dir = test_data_dir / "default" / "projects" / source_project_name
        assert project_dir.exists(), "CLI export did not create directory"
        trace_files = list(project_dir.glob("trace_*.json"))
        assert len(trace_files) >= 1, "CLI export did not create trace files"

        # Test import via CLI subprocess
        import_cmd = [
            "import",
            "default",
            "project",
            ".*",
            "--path",
            str(test_data_dir / "default"),
        ]

        result = self._run_cli_command(import_cmd)
        assert result.returncode == 0, f"CLI import failed: {result.stderr}"

    def test_export_import_error_handling(
        self, opik_client: opik.Opik, test_data_dir: Path
    ) -> None:
        """Test error handling for invalid commands."""
        # Test export with non-existent project - should fail gracefully
        try:
            export_project_by_name(
                name="non-existent-project",
                workspace="default",
                output_path=str(test_data_dir),
                max_results=None,
                filter_string=None,
                force=False,
                debug=False,
                format="json",
                api_key=None,
            )
            # If we get here, the function didn't raise an error as expected
            # Check if it at least didn't create any files
            project_dir = (
                test_data_dir / "default" / "projects" / "non-existent-project"
            )
            assert not project_dir.exists(), (
                "Should not create directory for non-existent project"
            )
        except SystemExit:
            # Expected - function calls sys.exit(1) on error
            pass

        # Test import with non-existent directory - should fail gracefully
        try:
            import_projects_from_directory(
                client=opik_client,
                source_dir=test_data_dir / "non-existent",
                dry_run=False,
                name_pattern=None,
                debug=False,
                recreate_experiments_flag=False,
            )
            # If we get here, check that stats show no imports
            # (function may return empty stats instead of raising)
        except (FileNotFoundError, SystemExit):
            # Expected - function may raise error or exit
            pass

    def test_export_import_chat_prompts_happy_flow(
        self,
        opik_client: opik.Opik,
        source_project_name: str,
        test_data_dir: Path,
    ) -> None:
        """Test the complete export/import flow for chat prompts."""
        # Step 1: Create a test chat prompt
        prompt_name = self._create_test_chat_prompt(opik_client)

        # Verify chat prompt was created
        prompts = opik_client.search_prompts()
        prompt_names = [p.name for p in prompts]
        assert prompt_name in prompt_names, (
            f"Expected chat prompt {prompt_name} to be created"
        )

        # Step 2: Export chat prompt using direct function call
        export_prompt_by_name(
            name=prompt_name,
            workspace="default",
            output_path=str(test_data_dir),
            max_results=None,
            force=False,
            debug=False,
            format="json",
            api_key=None,
        )

        # Verify export files were created
        prompts_dir = test_data_dir / "default" / "prompts"
        assert prompts_dir.exists(), f"Export directory not found: {prompts_dir}"

        prompt_files = list(prompts_dir.glob("prompt_*.json"))
        assert len(prompt_files) >= 1, "Expected at least 1 prompt file"

        # Verify chat-specific structure in exported file
        with open(prompt_files[0], "r") as f:
            prompt_data = json.load(f)

        assert "name" in prompt_data
        assert prompt_data["name"] == prompt_name
        assert "current_version" in prompt_data
        current_version = prompt_data["current_version"]

        # Verify it's a chat prompt (messages should be a list)
        assert "prompt" in current_version
        assert isinstance(current_version["prompt"], list), (
            "Chat prompt should have messages as a list"
        )
        assert "template_structure" in current_version
        assert current_version["template_structure"] == "chat"

        # Step 3: Import chat prompt using direct function call
        source_dir = test_data_dir / "default" / "prompts"
        stats = import_prompts_from_directory(
            client=opik_client,
            source_dir=source_dir,
            dry_run=False,
            name_pattern=None,
            debug=False,
        )

        # Verify import succeeded
        assert stats.get("prompts", 0) >= 1, "Expected at least 1 prompt to be imported"

        # Verify chat prompt was correctly imported to backend
        imported_prompts = opik_client.search_prompts()
        imported_prompt_names = [p.name for p in imported_prompts]
        assert prompt_name in imported_prompt_names, (
            f"Expected chat prompt {prompt_name} to be imported"
        )

        # Get the imported chat prompt and verify its content
        imported_chat_prompt = opik_client.get_chat_prompt(name=prompt_name)
        verifiers.verify_chat_prompt_version(
            imported_chat_prompt,
            name=prompt_name,
        )

    def test_import_projects_automatically_recreates_experiments(
        self,
        opik_client: opik.Opik,
        source_project_name: str,
        test_data_dir: Path,
    ) -> None:
        """Test import projects automatically recreates experiments."""
        # Step 1: Prepare test data with experiments
        dataset_name = self._create_test_dataset(opik_client)
        self._create_test_traces(opik_client, source_project_name)
        self._create_test_experiment(opik_client, source_project_name, dataset_name)

        # Step 2: Export the project data (traces) using direct function call
        export_project_by_name(
            name=source_project_name,
            workspace="default",
            output_path=str(test_data_dir),
            max_results=None,
            filter_string=None,
            force=False,
            debug=False,
            format="json",
            api_key=None,
        )

        # Verify export files were created
        project_dir = test_data_dir / "default" / "projects" / source_project_name
        assert project_dir.exists(), f"Export directory not found: {project_dir}"

        # Step 3: Test import (experiments are automatically recreated) using direct function call
        source_dir = test_data_dir / "default" / "projects"
        stats = import_projects_from_directory(
            client=opik_client,
            source_dir=source_dir,
            dry_run=False,
            name_pattern=None,
            debug=False,
            recreate_experiments_flag=True,  # Automatically recreate experiments
        )

        # Verify import succeeded
        assert stats.get("projects", 0) >= 1, (
            "Expected at least 1 project to be imported"
        )

    def test_export_import_llm_span_preserves_type_and_usage(
        self,
        opik_client: opik.Opik,
        test_data_dir: Path,
    ) -> None:
        """Test that LLM span type and usage data (including non-standard tokens) are preserved during import/export.

        This test verifies that:
        1. Span type="llm" is preserved after import
        2. Usage data with non-standard tokens (e.g., cached_tokens, reasoning_tokens) is preserved
        3. Cost is calculated by the backend based on usage
        """
        # Create a unique project name for this test
        project_name = f"cli-test-llm-span-{random_chars()}"

        # Create a trace with an LLM span that has usage with non-standard tokens
        # Using OpenAI format with detailed token breakdown that affects cost calculation
        llm_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            # Non-standard tokens that some providers track and use for cost calculation
            "completion_tokens_details": {
                "reasoning_tokens": 20,  # Used in o1 models
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {
                "cached_tokens": 25,  # Cached tokens have different pricing
                "audio_tokens": 0,
            },
        }

        # Create trace
        trace = opik_client.trace(
            name="test-llm-trace",
            input={"prompt": "What is the capital of France?"},
            output={"response": "Paris"},
            project_name=project_name,
        )

        # Create LLM span with type="llm", usage, model, and provider
        span = opik_client.span(
            trace_id=trace.id,
            name="llm-call",
            type="llm",
            input={
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ]
            },
            output={"content": "Paris"},
            usage=llm_usage,
            model="gpt-4o-mini",
            provider="openai",
            project_name=project_name,
        )

        original_span_id = span.id
        opik_client.flush()

        # Wait for data to be available and get original span data
        import time

        time.sleep(2)  # Give backend time to process and calculate cost

        original_span_data = opik_client.get_span_content(id=original_span_id)
        assert original_span_data is not None, "Failed to get original span data"
        assert original_span_data.type == "llm", (
            f"Original span type should be 'llm', got {original_span_data.type}"
        )
        assert original_span_data.usage is not None, (
            "Original span should have usage data"
        )

        # Store original values for comparison
        original_type = original_span_data.type
        original_usage = original_span_data.usage
        original_model = original_span_data.model
        original_provider = original_span_data.provider
        original_cost = original_span_data.total_estimated_cost

        print(
            f"Original span - type: {original_type}, model: {original_model}, provider: {original_provider}"
        )
        print(f"Original usage: {original_usage}")
        print(f"Original cost: {original_cost}")

        # Export the project
        export_project_by_name(
            name=project_name,
            workspace="default",
            output_path=str(test_data_dir),
            max_results=None,
            filter_string=None,
            force=False,
            debug=False,
            format="json",
            api_key=None,
        )

        # Verify export was created
        project_dir = test_data_dir / "default" / "projects" / project_name
        assert project_dir.exists(), f"Export directory not found: {project_dir}"

        trace_files = list(project_dir.glob("trace_*.json"))
        assert len(trace_files) == 1, (
            f"Expected 1 trace file, found: {len(trace_files)}"
        )

        # Verify exported JSON contains correct span data
        with open(trace_files[0], "r") as f:
            exported_data = json.load(f)

        exported_spans = exported_data.get("spans", [])
        assert len(exported_spans) == 1, (
            f"Expected 1 span in export, found: {len(exported_spans)}"
        )

        exported_span = exported_spans[0]
        assert exported_span.get("type") == "llm", (
            f"Exported span type should be 'llm', got {exported_span.get('type')}"
        )
        assert exported_span.get("model") == "gpt-4o-mini", "Exported model mismatch"
        assert exported_span.get("provider") == "openai", "Exported provider mismatch"
        assert exported_span.get("usage") is not None, (
            "Exported span should have usage data"
        )

        # Import the project to a new project (by modifying the project directory name)
        imported_project_name = f"{project_name}-imported"
        imported_project_dir = (
            test_data_dir / "default" / "projects" / imported_project_name
        )
        project_dir.rename(imported_project_dir)

        # Also update the project_name in the trace file
        trace_file = list(imported_project_dir.glob("trace_*.json"))[0]
        with open(trace_file, "r") as f:
            trace_data = json.load(f)
        trace_data["project_name"] = imported_project_name
        with open(trace_file, "w") as f:
            json.dump(trace_data, f)

        # Import the project
        source_dir = test_data_dir / "default" / "projects"
        stats = import_projects_from_directory(
            client=opik_client,
            source_dir=source_dir,
            dry_run=False,
            name_pattern=imported_project_name,
            debug=True,
            recreate_experiments_flag=False,
        )

        assert stats.get("projects", 0) >= 1, (
            "Expected at least 1 project to be imported"
        )
        assert stats.get("traces", 0) >= 1, "Expected at least 1 trace to be imported"

        opik_client.flush()
        time.sleep(2)  # Give backend time to process

        # Find the imported trace and span
        imported_traces = opik_client.search_traces(project_name=imported_project_name)
        assert len(imported_traces) >= 1, (
            f"Expected at least 1 imported trace, found: {len(imported_traces)}"
        )

        imported_trace = imported_traces[0]
        imported_spans = opik_client.search_spans(
            project_name=imported_project_name,
            trace_id=imported_trace.id,
        )
        assert len(imported_spans) >= 1, (
            f"Expected at least 1 imported span, found: {len(imported_spans)}"
        )

        imported_span = imported_spans[0]
        imported_span_data = opik_client.get_span_content(id=imported_span.id)

        print(
            f"Imported span - type: {imported_span_data.type}, model: {imported_span_data.model}, provider: {imported_span_data.provider}"
        )
        print(f"Imported usage: {imported_span_data.usage}")
        print(f"Imported cost: {imported_span_data.total_estimated_cost}")

        # Verify span type is preserved
        assert imported_span_data.type == "llm", (
            f"Imported span type should be 'llm', got '{imported_span_data.type}'"
        )

        # Verify model and provider are preserved
        assert imported_span_data.model == original_model, (
            f"Model mismatch: expected '{original_model}', got '{imported_span_data.model}'"
        )
        assert imported_span_data.provider == original_provider, (
            f"Provider mismatch: expected '{original_provider}', got '{imported_span_data.provider}'"
        )

        # Verify usage data is preserved (key token counts should match)
        assert imported_span_data.usage is not None, (
            "Imported span should have usage data"
        )

        # Check standard token counts
        assert imported_span_data.usage.get("prompt_tokens") == original_usage.get(
            "prompt_tokens"
        ), (
            f"prompt_tokens mismatch: expected {original_usage.get('prompt_tokens')}, "
            f"got {imported_span_data.usage.get('prompt_tokens')}"
        )
        assert imported_span_data.usage.get("completion_tokens") == original_usage.get(
            "completion_tokens"
        ), (
            f"completion_tokens mismatch: expected {original_usage.get('completion_tokens')}, "
            f"got {imported_span_data.usage.get('completion_tokens')}"
        )
        assert imported_span_data.usage.get("total_tokens") == original_usage.get(
            "total_tokens"
        ), (
            f"total_tokens mismatch: expected {original_usage.get('total_tokens')}, "
            f"got {imported_span_data.usage.get('total_tokens')}"
        )

        # Check non-standard token fields are preserved
        # The backend stores these with 'original_usage.' prefix for provider-specific token details
        assert (
            imported_span_data.usage.get(
                "original_usage.completion_tokens_details.reasoning_tokens"
            )
            == llm_usage["completion_tokens_details"]["reasoning_tokens"]
        ), (
            f"reasoning_tokens mismatch: expected {llm_usage['completion_tokens_details']['reasoning_tokens']}, "
            f"got {imported_span_data.usage.get('original_usage.completion_tokens_details.reasoning_tokens')}"
        )
        assert (
            imported_span_data.usage.get(
                "original_usage.prompt_tokens_details.cached_tokens"
            )
            == llm_usage["prompt_tokens_details"]["cached_tokens"]
        ), (
            f"cached_tokens mismatch: expected {llm_usage['prompt_tokens_details']['cached_tokens']}, "
            f"got {imported_span_data.usage.get('original_usage.prompt_tokens_details.cached_tokens')}"
        )

        # Verify cost is calculated (should be non-None if backend supports cost calculation for this model)
        # Note: Cost calculation depends on backend having pricing info for the model
        if original_cost is not None:
            assert imported_span_data.total_estimated_cost is not None, (
                "Expected imported span to have cost calculated by backend"
            )
            # Cost should be the same since usage is the same
            assert imported_span_data.total_estimated_cost == original_cost, (
                f"Cost mismatch: expected {original_cost}, got {imported_span_data.total_estimated_cost}"
            )


# --- sdks/python/tests/e2e/test_dataset.py ---

def test_insert_and_update_item__dataset_size_should_be_the_same__an_item_with_the_same_id_should_have_new_content(
    opik_client: opik.Opik, dataset_name: str
):
    DESCRIPTION = "E2E test dataset"
    project_name = opik_client.project_name

    dataset = opik_client.create_dataset(
        dataset_name, description=DESCRIPTION, project_name=project_name
    )

    ITEM_ID = helpers.generate_id()
    dataset.insert(
        [
            {
                "id": ITEM_ID,
                "input": {"question": "What is the of capital of France?"},
            },
        ]
    )
    dataset.update(
        [
            {
                "id": ITEM_ID,
                "input": {"question": "What is the of capital of Belarus?"},
            },
        ]
    )
    EXPECTED_DATASET_ITEMS = [
        dataset_item.DatasetItem(
            input={"question": "What is the of capital of Belarus?"},
        ),
    ]

    verifiers.verify_dataset(
        opik_client=opik_client,
        name=dataset_name,
        description=DESCRIPTION,
        dataset_items=EXPECTED_DATASET_ITEMS,
        project_name=project_name,
    )


# --- sdks/python/tests/e2e/test_failed_messages_replay.py ---

def test_failed_message_replay__create_attachment__replays_successfully(
    not_batching_opik_client: opik.Opik,
    project_name: str,
    attachment_data_file,
):
    """CreateAttachmentMessage stored while offline is delivered after replay.

    CreateAttachmentMessage bypasses the batch-manager and is stored in SQLite
    as-is regardless of the batching mode.  The non-batching client is used here
    so that CreateTraceMessage and CreateSpanMessage are stored in SQLite
    *before* their respective CreateAttachmentMessage — guaranteeing that the
    entities exist on the server when the upload is attempted during replay.

    In batching mode the order would be reversed: CreateAttachmentMessage lands
    in SQLite immediately (bypasses the batcher), while the corresponding
    CreateTraceBatchMessage / CreateSpansBatchMessage only arrives after an
    explicit flush — introducing a race between the async upload and the entity
    creation REST call.
    """
    file_name = "replay-attachment.bin"

    with offline_mode(not_batching_opik_client):
        # CreateTraceMessage → SQLite first, then CreateAttachmentMessage → SQLite second.
        trace = not_batching_opik_client.trace(
            name="replay-attachment-trace",
            project_name=project_name,
            attachments=[
                attachment.Attachment(
                    data=attachment_data_file.name,
                    file_name=file_name,
                    content_type="application/octet-stream",
                )
            ],
        )
        # CreateSpanMessage → SQLite third, then CreateAttachmentMessage → SQLite fourth.
        span = trace.span(
            name="replay-attachment-span",
            attachments=[
                attachment.Attachment(
                    data=attachment_data_file.name,
                    file_name=file_name,
                    content_type="application/octet-stream",
                )
            ],
        )
        not_batching_opik_client.flush()

    expected_attachment = {
        file_name: attachment.Attachment(
            data=attachment_data_file.name,
            file_name=file_name,
            content_type="application/octet-stream",
        )
    }

    verifiers.verify_attachments(
        opik_client=not_batching_opik_client,
        entity_type="trace",
        entity_id=trace.id,
        attachments=expected_attachment,
        data_sizes={file_name: ATTACHMENT_FILE_SIZE},
    )
    verifiers.verify_attachments(
        opik_client=not_batching_opik_client,
        entity_type="span",
        entity_id=span.id,
        attachments=expected_attachment,
        data_sizes={file_name: ATTACHMENT_FILE_SIZE},
    )


# --- sdks/python/tests/e2e/test_local_recording.py ---

def test_prevents_nested_usage():
    with opik.record_traces_locally():
        with pytest.raises(RuntimeError):
            with opik.record_traces_locally():
                pass

def test_cleanup_and_reuse_after_exit__should_save_new_data():
    client = opik_client.get_global_client()

    # First run: record and ensure the local processor becomes active
    with opik.record_traces_locally() as storage:
        _ = _sample_tracked_function("first run")
        assert len(storage.span_trees) == 1
        assert len(storage.trace_trees) == 1

        trace_trees = storage.trace_trees

        local = message_processors_chain.get_local_emulator_message_processor(
            chain=client.__internal_api__message_processor__
        )
        assert local is not None and local.is_active()

    EXPECTED_TRACE_TREE = TraceModel(
        id=ANY_BUT_NONE,
        start_time=ANY_BUT_NONE,
        name="_sample_tracked_function",
        project_name="Default Project",
        input={"x": "first run"},
        output={"output": "out:first run"},
        end_time=ANY_BUT_NONE,
        spans=[
            SpanModel(
                id=ANY_BUT_NONE,
                start_time=ANY_BUT_NONE,
                name="_sample_tracked_function",
                input={"x": "first run"},
                output={"output": "out:first run"},
                type="general",
                end_time=ANY_BUT_NONE,
                project_name="Default Project",
                last_updated_at=ANY_BUT_NONE,
                source="sdk",
            )
        ],
        last_updated_at=ANY_BUT_NONE,
        source="sdk",
    )

    assert_equal(expected=EXPECTED_TRACE_TREE, actual=trace_trees[0])

    # After context exit: local processor should be inactive and reset on the next activation
    local = message_processors_chain.get_local_emulator_message_processor(
        chain=client.__internal_api__message_processor__
    )
    assert local is not None and not local.is_active()

    # The second run should work independently
    with opik.record_traces_locally() as storage:
        _ = _sample_tracked_function("second run")

        assert len(storage.span_trees) == 1
        assert len(storage.trace_trees) == 1

        trace_trees = storage.trace_trees

    EXPECTED_TRACE_TREE = TraceModel(
        id=ANY_BUT_NONE,
        start_time=ANY_BUT_NONE,
        name="_sample_tracked_function",
        project_name="Default Project",
        input={"x": "second run"},
        output={"output": "out:second run"},
        end_time=ANY_BUT_NONE,
        spans=[
            SpanModel(
                id=ANY_BUT_NONE,
                start_time=ANY_BUT_NONE,
                name="_sample_tracked_function",
                input={"x": "second run"},
                output={"output": "out:second run"},
                type="general",
                end_time=ANY_BUT_NONE,
                project_name="Default Project",
                last_updated_at=ANY_BUT_NONE,
                source="sdk",
            )
        ],
        last_updated_at=ANY_BUT_NONE,
        source="sdk",
    )

    assert_equal(expected=EXPECTED_TRACE_TREE, actual=trace_trees[0])


# --- sdks/python/tests/e2e/test_optimization.py ---

def test_optimization_lifecycle__happyflow(opik_client: opik.Opik, dataset_name: str):
    dataset = opik_client.create_dataset(dataset_name)

    project_name = f"test_optimization_{dataset_name}"

    # Create optimization
    optimization = opik_client.create_optimization(
        objective_name="some-objective-name",
        dataset_name=dataset.name,
        name="some-optimization-name",
        project_name=project_name,
    )

    assert optimization.project_name == project_name

    verifiers.verify_optimization(
        opik_client=opik_client,
        optimization_id=optimization.id,
        name="some-optimization-name",
        dataset_name=dataset.name,
        status="running",
        objective_name="some-objective-name",
        project_name=project_name,
    )

    # Update optimization name and status
    optimization.update(name="new-optimization-name", status="completed")
    verifiers.verify_optimization(
        opik_client=opik_client,
        optimization_id=optimization.id,
        name="new-optimization-name",
        dataset_name=dataset.name,
        status="completed",
        objective_name="some-objective-name",
        project_name=project_name,
    )

    # Check project_name propagation
    optimization = opik_client.get_optimization_by_id(optimization.id)
    assert optimization.project_name == project_name

    opik_client.delete_optimizations([optimization.id])

    with pytest.raises(rest_api_core.ApiError):
        opik_client.get_optimization_by_id(optimization.id)


# --- sdks/python/tests/e2e/test_prompt.py ---

def test_prompt__filter_versions(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    shared_tag = _generate_random_tag()
    project_name = f"project-prompt__filter_versions-{_generate_random_suffix()}"

    v1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v1-{_generate_random_suffix()}",
        project_name=project_name,
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v1.version_id],
        tags=[shared_tag, _generate_random_tag()],
    )
    v2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v2-{_generate_random_suffix()}",
        project_name=project_name,
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v2.version_id],
        tags=_generate_random_tags(),
    )
    v3 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v3-{_generate_random_suffix()}",
        project_name=project_name,
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v3.version_id],
        tags=[_generate_random_tag(), shared_tag],
    )

    filtered_versions = opik_client.get_prompt_history(
        name=prompt_name,
        filter_string=f'tags contains "{shared_tag}"',
        project_name=project_name,
    )

    assert len(filtered_versions) == 2
    version_ids = {v.version_id for v in filtered_versions}
    assert v1.version_id in version_ids
    assert v3.version_id in version_ids
    assert v2.version_id not in version_ids

def test_prompt__search_versions(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    search_term = f"unique-search-term-{_generate_random_suffix()}"
    project_name = f"project-prompt__search_versions-{_generate_random_suffix()}"

    v1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"This template contains {search_term} for testing",
        project_name=project_name,
    )
    v2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"This template has different content {_generate_random_suffix()} for testing",
        project_name=project_name,
    )
    v3 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Another template with {search_term} included",
        project_name=project_name,
    )

    search_results = opik_client.get_prompt_history(
        name=prompt_name, search=search_term, project_name=project_name
    )

    assert len(search_results) == 2
    version_ids = {v.version_id for v in search_results}
    assert v1.version_id in version_ids
    assert v3.version_id in version_ids
    assert v2.version_id not in version_ids

def test_chat_prompt__filter_versions(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    shared_tag = _generate_random_tag()
    project_name = f"project-chat_prompt__filter_versions-{_generate_random_suffix()}"

    v1 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {"role": "user", "content": f"Message v1-{_generate_random_suffix()}"}
        ],
        project_name=project_name,
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v1.version_id],
        tags=[shared_tag, _generate_random_tag()],
    )
    v2 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {"role": "user", "content": f"Message v2-{_generate_random_suffix()}"}
        ],
        project_name=project_name,
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v2.version_id],
        tags=_generate_random_tags(),
    )
    v3 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {"role": "user", "content": f"Message v3-{_generate_random_suffix()}"}
        ],
        project_name=project_name,
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v3.version_id],
        tags=[_generate_random_tag(), shared_tag],
    )

    filtered_versions = opik_client.get_chat_prompt_history(
        name=prompt_name,
        filter_string=f'tags contains "{shared_tag}"',
        project_name=project_name,
    )

    assert len(filtered_versions) == 2
    version_ids = {v.version_id for v in filtered_versions}
    assert v1.version_id in version_ids
    assert v3.version_id in version_ids
    assert v2.version_id not in version_ids

def test_chat_prompt__search_versions(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    search_term = f"unique-search-term-{_generate_random_suffix()}"
    project_name = f"project-chat_prompt__search_versions-{_generate_random_suffix()}"

    v1 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {
                "role": "user",
                "content": f"This message contains {search_term} for testing",
            }
        ],
        project_name=project_name,
    )
    v2 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {
                "role": "user",
                "content": f"This message has different content {_generate_random_suffix()} for testing",
            }
        ],
        project_name=project_name,
    )
    v3 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {"role": "user", "content": f"Another message with {search_term} included"}
        ],
        project_name=project_name,
    )

    search_results = opik_client.get_chat_prompt_history(
        name=prompt_name, search=search_term, project_name=project_name
    )

    assert len(search_results) == 2
    version_ids = {v.version_id for v in search_results}
    assert v1.version_id in version_ids
    assert v3.version_id in version_ids
    assert v2.version_id not in version_ids

def test_prompt__update_version_tags__replace_mode(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    project_name = (
        f"project-prompt__update_version_tags__replace_mode-{_generate_random_suffix()}"
    )

    version1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v1-{_generate_random_suffix()}",
        project_name=project_name,
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id],
        tags=_generate_random_tags(),
        merge=False,
    )
    version2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v2-{_generate_random_suffix()}",
        project_name=project_name,
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version2.version_id],
        tags=_generate_random_tags(),
        merge=False,
    )

    new_tags = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id, version2.version_id],
        tags=new_tags,
        merge=False,
    )

    history = opik_client.get_prompt_history(
        name=prompt_name, project_name=project_name
    )
    assert len(history) == 2
    v1_in_history = next(
        (v for v in history if v.version_id == version1.version_id), None
    )
    v2_in_history = next(
        (v for v in history if v.version_id == version2.version_id), None
    )
    assert set(v1_in_history.tags) == set(new_tags)
    assert set(v2_in_history.tags) == set(new_tags)

def test_prompt__update_version_tags__default_replace_mode(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    project_name = f"project-prompt__update_version_tags__default_replace_mode-{_generate_random_suffix()}"

    version1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v1-{_generate_random_suffix()}",
        project_name=project_name,
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id],
        tags=_generate_random_tags(),
    )
    version2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v2-{_generate_random_suffix()}",
        project_name=project_name,
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version2.version_id],
        tags=_generate_random_tags(),
    )

    new_tags = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id, version2.version_id],
        tags=new_tags,
    )

    history = opik_client.get_prompt_history(
        name=prompt_name, project_name=project_name
    )
    assert len(history) == 2
    v1_in_history = next(
        (v for v in history if v.version_id == version1.version_id), None
    )
    v2_in_history = next(
        (v for v in history if v.version_id == version2.version_id), None
    )
    assert set(v1_in_history.tags) == set(new_tags)
    assert set(v2_in_history.tags) == set(new_tags)

def test_prompt__update_version_tags__clear_with_empty_array(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    version1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v1-{_generate_random_suffix()}",
    )
    version2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v2-{_generate_random_suffix()}",
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id, version2.version_id],
        tags=_generate_random_tags(),
    )

    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id, version2.version_id],
        tags=[],
    )

    history = opik_client.get_prompt_history(name=prompt_name)
    assert len(history) == 2
    v1_in_history = next(
        (v for v in history if v.version_id == version1.version_id), None
    )
    v2_in_history = next(
        (v for v in history if v.version_id == version2.version_id), None
    )
    assert v1_in_history.tags == []
    assert v2_in_history.tags == []

def test_prompt__update_version_tags__preserve_with_none(
    opik_client: opik.Opik, merge_param
):
    prompt_name = _generate_random_prompt_name()
    version1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v1-{_generate_random_suffix()}",
    )
    initial_tags_v1 = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id],
        tags=initial_tags_v1,
    )
    version2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v2-{_generate_random_suffix()}",
    )
    initial_tags_v2 = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version2.version_id],
        tags=initial_tags_v2,
    )

    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id, version2.version_id],
        tags=None,
        merge=merge_param,
    )

    history = opik_client.get_prompt_history(name=prompt_name)
    assert len(history) == 2
    v1_in_history = next(
        (v for v in history if v.version_id == version1.version_id), None
    )
    v2_in_history = next(
        (v for v in history if v.version_id == version2.version_id), None
    )
    assert set(v1_in_history.tags) == set(initial_tags_v1)
    assert set(v2_in_history.tags) == set(initial_tags_v2)

def test_prompt__update_version_tags__merge_mode(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    version1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v1-{_generate_random_suffix()}",
    )
    initial_tags_v1 = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id],
        tags=initial_tags_v1,
        merge=False,
    )
    version2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v2-{_generate_random_suffix()}",
    )
    initial_tags_v2 = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version2.version_id],
        tags=initial_tags_v2,
        merge=False,
    )

    additional_tags = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id, version2.version_id],
        tags=additional_tags,
        merge=True,
    )

    history = opik_client.get_prompt_history(name=prompt_name)
    assert len(history) == 2
    v1_in_history = next(
        (v for v in history if v.version_id == version1.version_id), None
    )
    v2_in_history = next(
        (v for v in history if v.version_id == version2.version_id), None
    )
    assert set(v1_in_history.tags) == set(initial_tags_v1 + additional_tags)
    assert set(v2_in_history.tags) == set(initial_tags_v2 + additional_tags)

def test_chat_prompt__update_version_tags(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    v1 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {"role": "user", "content": f"Message v1 {_generate_random_suffix()}"}
        ],
    )
    v2 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {"role": "user", "content": f"Message v2 {_generate_random_suffix()}"}
        ],
    )

    new_tags = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v1.version_id, v2.version_id],
        tags=new_tags,
        merge=False,
    )

    history = opik_client.get_chat_prompt_history(name=prompt_name)
    assert len(history) == 2
    v1_in_history = next((v for v in history if v.version_id == v1.version_id), None)
    v2_in_history = next((v for v in history if v.version_id == v2.version_id), None)
    assert set(v1_in_history.tags) == set(new_tags)
    assert set(v2_in_history.tags) == set(new_tags)


# --- sdks/python/tests/e2e/test_tracing.py ---

def test_search_spans__happyflow(opik_client: opik.Opik):
    # To define a unique search query, we will create a unique identifier that will be part of the trace input
    trace_id = helpers.generate_id()
    unique_identifier = str(uuid.uuid4())[-6:]

    filter_string = f'input contains "{unique_identifier}"'

    # Send a trace that matches the input filter
    trace = opik_client.trace(
        id=trace_id,
        name="trace-name",
        input={"input": "Some random input"},
        output={"output": "trace-output"},
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )
    matching_span = trace.span(
        name="span-name",
        input={"input": f"Some random input - {unique_identifier}"},
        output={"output": "span-output"},
    )
    trace.span(
        name="span-name",
        input={"input": "Some random input 1"},
        output={"output": "span-output"},
    )
    trace.span(
        name="span-name",
        input={"input": "Some random input 2"},
        output={"output": "span-output"},
    )

    opik_client.flush()

    # Search for the spans
    spans = opik_client.search_spans(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        trace_id=trace_id,
        filter_string=filter_string,
    )

    # Verify that the matching trace is returned
    assert len(spans) == 1, "Expected to find 1 matching span"
    assert spans[0].id == matching_span.id, "Expected to find the matching span"

def test_search_spans__wait_for_at_least__happy_flow(opik_client: opik.Opik):
    # check that synchronized searching for spans is working
    trace_id = helpers.generate_id()
    unique_identifier = str(uuid.uuid4())[-6:]

    # Send a trace that matches the input filter
    trace = opik_client.trace(
        id=trace_id,
        name="trace-name",
        input={"input": "Some random input"},
        output={"output": "trace-output"},
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )
    matching_count = 1000
    matching_span_ids = []
    for i in range(matching_count):
        matching_span = trace.span(
            name=f"span-name-{i}",
            input={"input": f"Some random input - {unique_identifier}"},
            output={"output": "span-output"},
        )
        matching_span_ids.append(matching_span.id)

    # adding two not matching spans
    trace.span(
        name="span-name",
        input={"input": "Some random input 1"},
        output={"output": "span-output"},
    )
    trace.span(
        name="span-name",
        input={"input": "Some random input 2"},
        output={"output": "span-output"},
    )

    opik_client.flush()

    filter_string = f'input contains "{unique_identifier}"'

    # Search for the spans with synchronization
    spans = opik_client.search_spans(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        trace_id=trace_id,
        filter_string=filter_string,
        wait_for_at_least=matching_count,
        wait_for_timeout=10,
    )

    # Verify that the matching trace is returned
    assert len(spans) == matching_count, (
        f"Expected to find {matching_count} matching spans"
    )
    for span in spans:
        assert span.id in matching_span_ids, (
            f"Expected to find the matching span id {span.id}"
        )

def test_search_spans__wait_for_at_least__timeout__exception_raised(
    opik_client: opik.Opik,
):
    trace_id = helpers.generate_id()
    unique_identifier = str(uuid.uuid4())[-6:]

    # Send a trace that matches the input filter
    trace = opik_client.trace(
        id=trace_id,
        name="trace-name",
        input={"input": "Some random input"},
        output={"output": "trace-output"},
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )
    trace.span(
        name="span-name",
        input={"input": f"Some random input - {unique_identifier}"},
        output={"output": "span-output"},
    )
    trace.span(
        name="span-name",
        input={"input": "Some random input 1"},
        output={"output": "span-output"},
    )
    trace.span(
        name="span-name",
        input={"input": "Some random input 2"},
        output={"output": "span-output"},
    )

    opik_client.flush()

    # Search for the spans
    unmatchable_count = 1000
    filter_string = f'input contains "{unique_identifier}"'
    with pytest.raises(exceptions.SearchTimeoutError):
        opik_client.search_spans(
            project_name=OPIK_E2E_TESTS_PROJECT_NAME,
            trace_id=trace_id,
            filter_string=filter_string,
            wait_for_at_least=unmatchable_count,
            wait_for_timeout=1,
        )

def test_tracked_function__update_current_span_and_trace_called__happyflow(
    opik_client,
):
    # Setup
    ID_STORAGE = {}
    THREAD_ID = id_helpers.generate_id()

    @opik.track
    def f():
        opik_context.update_current_span(
            name="span-name",
            input={"span-input": "span-input-value"},
            output={"span-output": "span-output-value"},
            metadata={"span-metadata-key": "span-metadata-value"},
            total_cost=0.42,
        )
        opik_context.update_current_trace(
            name="trace-name",
            input={"trace-input": "trace-input-value"},
            output={"trace-output": "trace-output-value"},
            metadata={"trace-metadata-key": "trace-metadata-value"},
            thread_id=THREAD_ID,
        )
        ID_STORAGE["f_span-id"] = opik_context.get_current_span_data().id
        ID_STORAGE["f_trace-id"] = opik_context.get_current_trace_data().id

    # Call
    f()
    opik.flush_tracker()

    # Verify top level span
    verifiers.verify_span(
        opik_client=opik_client,
        span_id=ID_STORAGE["f_span-id"],
        parent_span_id=None,
        trace_id=ID_STORAGE["f_trace-id"],
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        name="span-name",
        input={"span-input": "span-input-value"},
        output={"span-output": "span-output-value"},
        metadata={"span-metadata-key": "span-metadata-value"},
        total_cost=0.42,
        source="sdk",
    )

    verifiers.verify_trace(
        opik_client=opik_client,
        trace_id=ID_STORAGE["f_trace-id"],
        name="trace-name",
        input={"trace-input": "trace-input-value"},
        output={"trace-output": "trace-output-value"},
        metadata={"trace-metadata-key": "trace-metadata-value"},
        thread_id=THREAD_ID,
        source="sdk",
    )

def test_opik_trace__attachments(opik_client, attachment_data_file):
    trace_id = helpers.generate_id()
    file_name = os.path.basename(attachment_data_file.name)
    names = [file_name + "_first", file_name + "_second"]
    attachments = {
        names[0]: Attachment(
            data=attachment_data_file.name,
            file_name=names[0],
            content_type="application/octet-stream",
        ),
        names[1]: Attachment(
            data=attachment_data_file.name,
            file_name=names[1],
            content_type="application/octet-stream",
        ),
    }
    data_sizes = {
        names[0]: ATTACHMENT_FILE_SIZE,
        names[1]: ATTACHMENT_FILE_SIZE,
    }

    # Send a trace that matches the input filter
    opik_client.trace(
        id=trace_id,
        name="trace-name",
        input={"input": "Some random input"},
        output={"output": "trace-output"},
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        attachments=attachments.values(),
    )

    opik_client.flush()

    # check that the attachment was uploaded
    verifiers.verify_attachments(
        opik_client=opik_client,
        entity_type="trace",
        entity_id=trace_id,
        attachments=attachments,
        data_sizes=data_sizes,
    )

def test_tracked_function__update_current_trace__with_attachments(
    opik_client, attachment_data_file
):
    # Setup
    ID_STORAGE = {}
    THREAD_ID = id_helpers.generate_id()

    file_name = os.path.basename(attachment_data_file.name)
    attachments = {
        file_name: Attachment(
            data=attachment_data_file.name,
            file_name=file_name,
            content_type="application/octet-stream",
        )
    }
    data_sizes = {
        file_name: ATTACHMENT_FILE_SIZE,
    }

    @opik.track
    def f():
        opik_context.update_current_trace(
            name="trace-name",
            input={"trace-input": "trace-input-value"},
            output={"trace-output": "trace-output-value"},
            metadata={"trace-metadata-key": "trace-metadata-value"},
            thread_id=THREAD_ID,
            attachments=attachments.values(),
        )
        ID_STORAGE["f_trace-id"] = opik_context.get_current_trace_data().id

    # Call
    f()
    opik.flush_tracker()

    # check that the attachment was uploaded
    verifiers.verify_attachments(
        opik_client=opik_client,
        entity_type="trace",
        entity_id=ID_STORAGE["f_trace-id"],
        attachments=attachments,
        data_sizes=data_sizes,
    )

def test_opik_client_span__attachments(opik_client, attachment_data_file):
    trace_id = helpers.generate_id()
    file_name = os.path.basename(attachment_data_file.name)
    names = [file_name + "_first", file_name + "_second"]
    attachments = {
        names[0]: Attachment(
            data=attachment_data_file.name,
            file_name=names[0],
            content_type="application/octet-stream",
        ),
        names[1]: Attachment(
            data=attachment_data_file.name,
            file_name=names[1],
            content_type="application/octet-stream",
        ),
    }
    data_sizes = {
        names[0]: ATTACHMENT_FILE_SIZE,
        names[1]: ATTACHMENT_FILE_SIZE,
    }

    # Send a trace that matches the input filter
    opik_client.trace(
        id=trace_id,
        name="trace-name",
        input={"input": "Some random input"},
        output={"output": "trace-output"},
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )
    span = opik_client.span(
        trace_id=trace_id,
        name="span-name",
        input={"input": "Some random input 2"},
        output={"output": "span-output"},
        attachments=attachments.values(),
    )

    opik_client.flush()

    # check that the attachment was uploaded
    verifiers.verify_attachments(
        opik_client=opik_client,
        entity_type="span",
        entity_id=span.id,
        attachments=attachments,
        data_sizes=data_sizes,
    )

def test_opik_client_span__attachment_with_file_like_data(
    opik_client, attachment_data_file
):
    """
    Test that a span can be created with an attachment that has file-like data.
    """
    trace_id = helpers.generate_id()
    file_name = os.path.basename(attachment_data_file.name)
    names = [file_name + "_first", file_name + "_without_mime_type"]
    # read file bytes into memory
    attachment_data_file.seek(0)
    attachment_data = attachment_data_file.read()

    attachments = {
        names[0]: Attachment(
            data=attachment_data,
            file_name=names[0],
            content_type="application/octet-stream",
        ),
        names[1]: Attachment(
            data=attachment_data,
            file_name=names[1],
        ),
    }
    data_sizes = {
        names[0]: ATTACHMENT_FILE_SIZE,
        names[1]: ATTACHMENT_FILE_SIZE,
    }

    # Send a trace that matches the input filter
    opik_client.trace(
        id=trace_id,
        name="trace-name",
        input={"input": "Some random input"},
        output={"output": "trace-output"},
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )
    span = opik_client.span(
        trace_id=trace_id,
        name="span-name",
        input={"input": "Some random input 2"},
        output={"output": "span-output"},
        attachments=attachments.values(),
    )

    opik_client.flush()

    expected_attachments = {
        names[0]: attachments[names[0]],
        names[1]: Attachment(
            data=attachment_data,
            file_name=names[1],
            content_type="application/octet-stream",  # should be inferred from data
        ),
    }

    # check that the attachment was uploaded
    verifiers.verify_attachments(
        opik_client=opik_client,
        entity_type="span",
        entity_id=span.id,
        attachments=expected_attachments,
        data_sizes=data_sizes,
    )

def test_span_span__attachments(opik_client, attachment_data_file):
    trace_id = helpers.generate_id()
    file_name = os.path.basename(attachment_data_file.name)
    names = [file_name + "_first", file_name + "_second"]
    attachments = {
        names[0]: Attachment(
            data=attachment_data_file.name,
            file_name=names[0],
            content_type="application/octet-stream",
        ),
        names[1]: Attachment(
            data=attachment_data_file.name,
            file_name=names[1],
            content_type="application/octet-stream",
        ),
    }
    data_sizes = {
        names[0]: ATTACHMENT_FILE_SIZE,
        names[1]: ATTACHMENT_FILE_SIZE,
    }

    # Send a trace that matches the input filter
    opik_client.trace(
        id=trace_id,
        name="trace-name",
        input={"input": "Some random input"},
        output={"output": "trace-output"},
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )
    span = opik_client.span(
        trace_id=trace_id,
        name="span-name",
        input={"input": "Some random input 2"},
        output={"output": "span-output"},
    )
    last_span = span.span(
        name="span-name",
        input={"input": "Some random input 2"},
        output={"output": "span-output"},
        attachments=attachments.values(),
    )

    opik_client.flush()

    # check that the attachment was uploaded
    verifiers.verify_attachments(
        opik_client=opik_client,
        entity_type="span",
        entity_id=last_span.id,
        attachments=attachments,
        data_sizes=data_sizes,
    )

def test_trace_span__attachments(opik_client, attachment_data_file):
    trace_id = helpers.generate_id()
    file_name = os.path.basename(attachment_data_file.name)
    names = [file_name + "_first", file_name + "_second"]
    attachments = {
        names[0]: Attachment(
            data=attachment_data_file.name,
            file_name=names[0],
            content_type="application/octet-stream",
        ),
        names[1]: Attachment(
            data=attachment_data_file.name,
            file_name=names[1],
            content_type="application/octet-stream",
        ),
    }
    data_sizes = {
        names[0]: ATTACHMENT_FILE_SIZE,
        names[1]: ATTACHMENT_FILE_SIZE,
    }

    # Send a trace that matches the input filter
    trace = opik_client.trace(
        id=trace_id,
        name="trace-name",
        input={"input": "Some random input"},
        output={"output": "trace-output"},
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )
    span = trace.span(
        name="span-name",
        input={"input": "Some random input 2"},
        output={"output": "span-output"},
        attachments=attachments.values(),
    )

    opik_client.flush()

    # check that the attachment was uploaded
    verifiers.verify_attachments(
        opik_client=opik_client,
        entity_type="span",
        entity_id=span.id,
        attachments=attachments,
        data_sizes=data_sizes,
    )

def test_tracked_function__update_current_span__with_attachments(
    opik_client, attachment_data_file
):
    # Setup
    ID_STORAGE = {}
    THREAD_ID = id_helpers.generate_id()

    file_name = os.path.basename(attachment_data_file.name)
    attachments = {
        file_name: Attachment(
            data=attachment_data_file.name,
            file_name=file_name,
            content_type="application/octet-stream",
        )
    }
    data_sizes = {
        file_name: ATTACHMENT_FILE_SIZE,
    }

    @opik.track
    def f():
        opik_context.update_current_span(
            name="span-name",
            input={"span-input": "span-input-value"},
            output={"span-output": "span-output-value"},
            metadata={"span-metadata-key": "span-metadata-value"},
            total_cost=0.42,
            attachments=attachments.values(),
        )
        opik_context.update_current_trace(
            name="trace-name",
            input={"trace-input": "trace-input-value"},
            output={"trace-output": "trace-output-value"},
            metadata={"trace-metadata-key": "trace-metadata-value"},
            thread_id=THREAD_ID,
        )
        ID_STORAGE["f_span-id"] = opik_context.get_current_span_data().id

    # Call
    f()
    opik.flush_tracker()

    # check that the attachment was uploaded
    verifiers.verify_attachments(
        opik_client=opik_client,
        entity_type="span",
        entity_id=ID_STORAGE["f_span-id"],
        attachments=attachments,
        data_sizes=data_sizes,
    )

def test_opik_client__update_trace__happy_flow(
    new_input, new_output, new_tags, new_metadata, new_thread_id, opik_client: opik.Opik
):
    # test that the trace update works by updating only one field at a time
    project_name = "update_trace_happy_flow"
    trace_name = "trace_name"
    input = {"input": "trace-input-value"}
    output = {"output": "trace-output-value"}
    tags = ["trace-tag"]
    metadata = {"trace-metadata-key": "trace-metadata-value"}
    thread_id = id_helpers.generate_id()
    trace = opik_client.trace(
        name=trace_name,
        input=input,
        output=output,
        tags=tags,
        metadata=metadata,
        project_name=project_name,
        thread_id=thread_id,
    )

    opik_client.flush()

    # verify that the trace was saved
    verifiers.verify_trace(
        opik_client=opik_client,
        trace_id=trace.id,
        name=trace_name,
        project_name=project_name,
        input=input,
        output=output,
        metadata=metadata,
        tags=tags,
        thread_id=thread_id,
        source="sdk",
    )

    #
    # Do partial update
    #
    opik_client.update_trace(
        trace_id=trace.id,
        project_name=project_name,
        input=new_input,
        output=new_output,
        tags=new_tags,
        metadata=new_metadata,
        thread_id=new_thread_id,
    )

    # flush to make sure the update was logged to server
    opik_client.flush()

    input = new_input or input
    output = new_output or output
    tags = new_tags or tags
    metadata = new_metadata or metadata
    thread_id = new_thread_id or thread_id

    verifiers.verify_trace(
        opik_client=opik_client,
        trace_id=trace.id,
        name=trace_name,
        project_name=project_name,
        input=input,
        output=output,
        tags=tags,
        metadata=metadata,
        thread_id=thread_id,
        source="sdk",
    )

def test_search_spans__filter_by_feedback_score__is_empty_and_equals(
    opik_client: opik.Opik,
):
    # Create a unique metric name to avoid conflicts with other tests
    unique_metric = f"test_metric_{str(uuid.uuid4()).replace('-', '_')[-8:]}"
    trace_id = helpers.generate_id()

    # Create a trace with two spans
    trace = opik_client.trace(
        id=trace_id,
        name="trace-name",
        input={"input": "Some random input"},
        output={"output": "trace-output"},
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
    )

    # Create span with the feedback score
    span_with_score = trace.span(
        name="span-with-score",
        input={"input": "span-input-1"},
        output={"output": "span-output-1"},
    )
    span_with_score.log_feedback_score(
        unique_metric, value=0.85, category_name="test-category", reason="test-reason"
    )

    # Create span without the feedback score
    span_without_score = trace.span(
        name="span-without-score",
        input={"input": "span-input-2"},
        output={"output": "span-output-2"},
    )

    opik_client.flush()

    # Test filtering with is_empty - should find span without the score
    spans_empty = opik_client.search_spans(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        trace_id=trace_id,
        filter_string=f"feedback_scores.{unique_metric} is_empty",
    )
    span_ids_empty = {span.id for span in spans_empty}
    assert span_without_score.id in span_ids_empty, (
        "Span without score should be found with is_empty filter"
    )
    assert span_with_score.id not in span_ids_empty, (
        "Span with score should not be found with is_empty filter"
    )

    # Test filtering with is_not_empty - should find span with the score
    spans_not_empty = opik_client.search_spans(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        trace_id=trace_id,
        filter_string=f"feedback_scores.{unique_metric} is_not_empty",
    )
    span_ids_not_empty = {span.id for span in spans_not_empty}
    assert span_with_score.id in span_ids_not_empty, (
        "Span with score should be found with is_not_empty filter"
    )
    assert span_without_score.id not in span_ids_not_empty, (
        "Span without score should not be found with is_not_empty filter"
    )

    # Test filtering with = operator - should find span with the specific score value
    spans_with_value = opik_client.search_spans(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        trace_id=trace_id,
        filter_string=f"feedback_scores.{unique_metric} = 0.85",
    )
    span_ids_with_value = {span.id for span in spans_with_value}
    assert span_with_score.id in span_ids_with_value, (
        "Span with score value 0.85 should be found"
    )
    assert span_without_score.id not in span_ids_with_value, (
        "Span without score should not be found"
    )

    # Verify is_not_empty and = return the same span
    assert span_ids_not_empty == span_ids_with_value, (
        "is_not_empty and = filters should return the same spans for this test case"
    )


# --- sdks/python/tests/e2e/compatibility_v1/test_dataset.py ---

def test_insert_and_update_item__dataset_size_should_be_the_same__an_item_with_the_same_id_should_have_new_content(
    opik_client: opik.Opik, dataset_name: str
):
    DESCRIPTION = "E2E test dataset"

    dataset = opik_client.create_dataset(dataset_name, description=DESCRIPTION)

    ITEM_ID = helpers.generate_id()
    dataset.insert(
        [
            {
                "id": ITEM_ID,
                "input": {"question": "What is the of capital of France?"},
            },
        ]
    )
    dataset.update(
        [
            {
                "id": ITEM_ID,
                "input": {"question": "What is the of capital of Belarus?"},
            },
        ]
    )
    EXPECTED_DATASET_ITEMS = [
        dataset_item.DatasetItem(
            input={"question": "What is the of capital of Belarus?"},
        ),
    ]

    verifiers.verify_dataset(
        opik_client=opik_client,
        name=dataset_name,
        description=DESCRIPTION,
        dataset_items=EXPECTED_DATASET_ITEMS,
    )


# --- sdks/python/tests/e2e/compatibility_v1/test_optimization.py ---

def test_optimization_lifecycle__happyflow(opik_client: opik.Opik, dataset_name: str):
    dataset = opik_client.create_dataset(dataset_name)

    # Create optimization
    optimization = opik_client.create_optimization(
        objective_name="some-objective-name",
        dataset_name=dataset.name,
        name="some-optimization-name",
    )

    verifiers.verify_optimization(
        opik_client=opik_client,
        optimization_id=optimization.id,
        name="some-optimization-name",
        dataset_name=dataset.name,
        status="running",
        objective_name="some-objective-name",
    )

    # Update optimization name and status
    optimization.update(name="new-optimization-name", status="completed")
    verifiers.verify_optimization(
        opik_client=opik_client,
        optimization_id=optimization.id,
        name="new-optimization-name",
        dataset_name=dataset.name,
        status="completed",
        objective_name="some-objective-name",
    )

    opik_client.delete_optimizations([optimization.id])

    with pytest.raises(rest_api_core.ApiError):
        opik_client.get_optimization_by_id(optimization.id)


# --- sdks/python/tests/e2e/compatibility_v1/test_prompt.py ---

def test_prompt__filter_versions(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    shared_tag = _generate_random_tag()
    v1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v1-{_generate_random_suffix()}",
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v1.version_id],
        tags=[shared_tag, _generate_random_tag()],
    )
    v2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v2-{_generate_random_suffix()}",
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v2.version_id],
        tags=_generate_random_tags(),
    )
    v3 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v3-{_generate_random_suffix()}",
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v3.version_id],
        tags=[_generate_random_tag(), shared_tag],
    )

    filtered_versions = opik_client.get_prompt_history(
        name=prompt_name,
        filter_string=f'tags contains "{shared_tag}"',
    )

    assert len(filtered_versions) == 2
    version_ids = {v.version_id for v in filtered_versions}
    assert v1.version_id in version_ids
    assert v3.version_id in version_ids
    assert v2.version_id not in version_ids

def test_prompt__search_versions(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    search_term = f"unique-search-term-{_generate_random_suffix()}"
    v1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"This template contains {search_term} for testing",
    )
    v2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"This template has different content {_generate_random_suffix()} for testing",
    )
    v3 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Another template with {search_term} included",
    )

    search_results = opik_client.get_prompt_history(
        name=prompt_name, search=search_term
    )

    assert len(search_results) == 2
    version_ids = {v.version_id for v in search_results}
    assert v1.version_id in version_ids
    assert v3.version_id in version_ids
    assert v2.version_id not in version_ids

def test_chat_prompt__filter_versions(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    shared_tag = _generate_random_tag()
    v1 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {"role": "user", "content": f"Message v1-{_generate_random_suffix()}"}
        ],
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v1.version_id],
        tags=[shared_tag, _generate_random_tag()],
    )
    v2 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {"role": "user", "content": f"Message v2-{_generate_random_suffix()}"}
        ],
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v2.version_id],
        tags=_generate_random_tags(),
    )
    v3 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {"role": "user", "content": f"Message v3-{_generate_random_suffix()}"}
        ],
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v3.version_id],
        tags=[_generate_random_tag(), shared_tag],
    )

    filtered_versions = opik_client.get_chat_prompt_history(
        name=prompt_name,
        filter_string=f'tags contains "{shared_tag}"',
    )

    assert len(filtered_versions) == 2
    version_ids = {v.version_id for v in filtered_versions}
    assert v1.version_id in version_ids
    assert v3.version_id in version_ids
    assert v2.version_id not in version_ids

def test_chat_prompt__search_versions(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    search_term = f"unique-search-term-{_generate_random_suffix()}"
    v1 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {
                "role": "user",
                "content": f"This message contains {search_term} for testing",
            }
        ],
    )
    v2 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {
                "role": "user",
                "content": f"This message has different content {_generate_random_suffix()} for testing",
            }
        ],
    )
    v3 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {"role": "user", "content": f"Another message with {search_term} included"}
        ],
    )

    search_results = opik_client.get_chat_prompt_history(
        name=prompt_name, search=search_term
    )

    assert len(search_results) == 2
    version_ids = {v.version_id for v in search_results}
    assert v1.version_id in version_ids
    assert v3.version_id in version_ids
    assert v2.version_id not in version_ids

def test_prompt__update_version_tags__replace_mode(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    version1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v1-{_generate_random_suffix()}",
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id],
        tags=_generate_random_tags(),
        merge=False,
    )
    version2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v2-{_generate_random_suffix()}",
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version2.version_id],
        tags=_generate_random_tags(),
        merge=False,
    )

    new_tags = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id, version2.version_id],
        tags=new_tags,
        merge=False,
    )

    history = opik_client.get_prompt_history(name=prompt_name)
    assert len(history) == 2
    v1_in_history = next(
        (v for v in history if v.version_id == version1.version_id), None
    )
    v2_in_history = next(
        (v for v in history if v.version_id == version2.version_id), None
    )
    assert set(v1_in_history.tags) == set(new_tags)
    assert set(v2_in_history.tags) == set(new_tags)

def test_prompt__update_version_tags__default_replace_mode(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    version1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v1-{_generate_random_suffix()}",
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id],
        tags=_generate_random_tags(),
    )
    version2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v2-{_generate_random_suffix()}",
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version2.version_id],
        tags=_generate_random_tags(),
    )

    new_tags = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id, version2.version_id],
        tags=new_tags,
    )

    history = opik_client.get_prompt_history(name=prompt_name)
    v1_in_history = next(
        (v for v in history if v.version_id == version1.version_id), None
    )
    v2_in_history = next(
        (v for v in history if v.version_id == version2.version_id), None
    )
    assert set(v1_in_history.tags) == set(new_tags)
    assert set(v2_in_history.tags) == set(new_tags)

def test_prompt__update_version_tags__clear_with_empty_array(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    version1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v1-{_generate_random_suffix()}",
    )
    version2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v2-{_generate_random_suffix()}",
    )
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id, version2.version_id],
        tags=_generate_random_tags(),
    )

    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id, version2.version_id],
        tags=[],
    )

    history = opik_client.get_prompt_history(name=prompt_name)
    assert len(history) == 2
    v1_in_history = next(
        (v for v in history if v.version_id == version1.version_id), None
    )
    v2_in_history = next(
        (v for v in history if v.version_id == version2.version_id), None
    )
    assert v1_in_history.tags == []
    assert v2_in_history.tags == []

def test_prompt__update_version_tags__preserve_with_none(
    opik_client: opik.Opik, merge_param
):
    prompt_name = _generate_random_prompt_name()
    version1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v1-{_generate_random_suffix()}",
    )
    initial_tags_v1 = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id],
        tags=initial_tags_v1,
    )
    version2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v2-{_generate_random_suffix()}",
    )
    initial_tags_v2 = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version2.version_id],
        tags=initial_tags_v2,
    )

    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id, version2.version_id],
        tags=None,
        merge=merge_param,
    )

    history = opik_client.get_prompt_history(name=prompt_name)
    assert len(history) == 2
    v1_in_history = next(
        (v for v in history if v.version_id == version1.version_id), None
    )
    v2_in_history = next(
        (v for v in history if v.version_id == version2.version_id), None
    )
    assert set(v1_in_history.tags) == set(initial_tags_v1)
    assert set(v2_in_history.tags) == set(initial_tags_v2)

def test_prompt__update_version_tags__merge_mode(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    version1 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v1-{_generate_random_suffix()}",
    )
    initial_tags_v1 = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id],
        tags=initial_tags_v1,
        merge=False,
    )
    version2 = opik_client.create_prompt(
        name=prompt_name,
        prompt=f"Template v2-{_generate_random_suffix()}",
    )
    initial_tags_v2 = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version2.version_id],
        tags=initial_tags_v2,
        merge=False,
    )

    additional_tags = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[version1.version_id, version2.version_id],
        tags=additional_tags,
        merge=True,
    )

    history = opik_client.get_prompt_history(name=prompt_name)
    assert len(history) == 2
    v1_in_history = next(
        (v for v in history if v.version_id == version1.version_id), None
    )
    v2_in_history = next(
        (v for v in history if v.version_id == version2.version_id), None
    )
    assert set(v1_in_history.tags) == set(initial_tags_v1 + additional_tags)
    assert set(v2_in_history.tags) == set(initial_tags_v2 + additional_tags)

def test_chat_prompt__update_version_tags(opik_client: opik.Opik):
    prompt_name = _generate_random_prompt_name()
    v1 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {"role": "user", "content": f"Message v1 {_generate_random_suffix()}"}
        ],
    )
    v2 = opik_client.create_chat_prompt(
        name=prompt_name,
        messages=[
            {"role": "user", "content": f"Message v2 {_generate_random_suffix()}"}
        ],
    )

    new_tags = _generate_random_tags()
    opik_client.get_prompts_client().batch_update_prompt_version_tags(
        version_ids=[v1.version_id, v2.version_id],
        tags=new_tags,
        merge=False,
    )

    history = opik_client.get_chat_prompt_history(name=prompt_name)
    assert len(history) == 2
    v1_in_history = next((v for v in history if v.version_id == v1.version_id), None)
    v2_in_history = next((v for v in history if v.version_id == v2.version_id), None)
    assert set(v1_in_history.tags) == set(new_tags)
    assert set(v2_in_history.tags) == set(new_tags)


# --- sdks/python/tests/e2e/compatibility_v1/evaluation/test_evaluate_filter_string.py ---

def test_evaluate__with_filter_string__filters_dataset_items(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """Test that evaluate correctly filters dataset items using filter_string."""
    dataset = opik_client.create_dataset(dataset_name)

    dataset_items = [
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the capital of France?"},
            "expected_model_output": {"output": "Paris"},
            "category": "geography",
        },
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is 2+2?"},
            "expected_model_output": {"output": "4"},
            "category": "math",
        },
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the capital of Germany?"},
            "expected_model_output": {"output": "Berlin"},
            "category": "geography",
        },
    ]

    dataset.insert(dataset_items)

    def task(item: Dict[str, Any]):
        if item["input"] == {"question": "What is the capital of France?"}:
            return {"output": "Paris"}
        if item["input"] == {"question": "What is the capital of Germany?"}:
            return {"output": "Berlin"}
        if item["input"] == {"question": "What is 2+2?"}:
            return {"output": "4"}

        raise AssertionError(
            f"Task received dataset item with an unexpected input: {item['input']}"
        )

    equals_metric = metrics.Equals()
    opik.evaluate(
        dataset=dataset,
        task=task,
        scoring_metrics=[equals_metric],
        experiment_name=experiment_name,
        scoring_key_mapping={
            "reference": lambda x: x["expected_model_output"]["output"],
        },
        dataset_filter_string='data.category = "geography"',
    )

    opik.flush_tracker()

    retrieved_experiment = opik_client.get_experiment_by_name(experiment_name)
    experiment_items_contents = retrieved_experiment.get_items()
    assert len(experiment_items_contents) == 2, (
        f"Expected 2 experiment items (filtered by geography category), but got {len(experiment_items_contents)}. "
        f"Experiment items: {experiment_items_contents}"
    )

def test_evaluate_optimization_trial__with_filter_string__filters_dataset_items(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """Test that evaluate_optimization_trial correctly filters dataset items using filter_string."""
    dataset = opik_client.create_dataset(dataset_name)

    dataset_items = [
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the capital of France?"},
            "expected_model_output": {"output": "Paris"},
            "category": "geography",
        },
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is 2+2?"},
            "expected_model_output": {"output": "4"},
            "category": "math",
        },
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the capital of Germany?"},
            "expected_model_output": {"output": "Berlin"},
            "category": "geography",
        },
    ]

    dataset.insert(dataset_items)

    def task(item: Dict[str, Any]):
        if item["input"] == {"question": "What is the capital of France?"}:
            return {"output": "Paris"}
        if item["input"] == {"question": "What is the capital of Germany?"}:
            return {"output": "Berlin"}
        if item["input"] == {"question": "What is 2+2?"}:
            return {"output": "4"}

        raise AssertionError(
            f"Task received dataset item with an unexpected input: {item['input']}"
        )

    equals_metric = metrics.Equals()
    evaluator_module.evaluate_optimization_trial(
        optimization_id=id_helpers.generate_id(),
        dataset=dataset,
        task=task,
        scoring_metrics=[equals_metric],
        experiment_name=experiment_name,
        scoring_key_mapping={
            "reference": lambda x: x["expected_model_output"]["output"],
        },
        dataset_filter_string='data.category = "math"',
    )

    opik.flush_tracker()

    retrieved_experiment = opik_client.get_experiment_by_name(experiment_name)
    experiment_items_contents = retrieved_experiment.get_items()
    assert len(experiment_items_contents) == 1, (
        f"Expected 1 experiment item (filtered by math category), but got {len(experiment_items_contents)}. "
        f"Experiment items: {experiment_items_contents}"
    )


# --- sdks/python/tests/e2e/compatibility_v1/evaluation/test_experiment_evaluate.py ---

def test_experiment_creation_via_evaluate_function__single_prompt_arg_used__filter_dataset_items_by_id(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    dataset = opik_client.create_dataset(dataset_name)

    dataset_items = [
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the of capital of France?"},
            "expected_model_output": {"output": "Paris"},
        },
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the of capital of Germany?"},
            "expected_model_output": {"output": "Berlin"},
        },
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the of capital of Poland?"},
            "expected_model_output": {"output": "Warsaw"},
        },
    ]

    dataset.insert(dataset_items)

    def task(item: Dict[str, Any]):
        if item["input"] == {"question": "What is the of capital of France?"}:
            return {"output": "Paris"}
        if item["input"] == {"question": "What is the of capital of Germany?"}:
            return {"output": "Berlin"}
        if item["input"] == {"question": "What is the of capital of Poland?"}:
            return {"output": "Krakow"}

        raise AssertionError(
            f"Task received dataset item with an unexpected input: {item['input']}"
        )

    prompt = Prompt(
        name=f"test-experiment-prompt-{random_chars()}",
        prompt=f"test-experiment-prompt-template-{random_chars()}",
    )

    dataset_item_ids = [item["id"] for item in dataset_items]
    dataset_item_ids.pop(2)
    # add non existing id
    dataset_item_ids.append(id_helpers.generate_id())

    equals_metric = metrics.Equals()
    evaluation_result = opik.evaluate(
        dataset=dataset,
        task=task,
        scoring_metrics=[equals_metric],
        experiment_name=experiment_name,
        experiment_config={
            "model_name": "gpt-3.5",
        },
        scoring_key_mapping={
            "reference": lambda x: x["expected_model_output"]["output"],
        },
        prompt=prompt,
        dataset_item_ids=dataset_item_ids,
    )

    opik.flush_tracker()

    verifiers.verify_experiment(
        opik_client=opik_client,
        id=evaluation_result.experiment_id,
        experiment_name=evaluation_result.experiment_name,
        experiment_metadata={"model_name": "gpt-3.5"},
        traces_amount=2,  # one trace per dataset item
        feedback_scores_amount=1,
        prompts=[prompt],
    )

    assert evaluation_result.dataset_id == dataset.id, (
        f"Expected evaluation result dataset_id '{dataset.id}', but got '{evaluation_result.dataset_id}'"
    )

    retrieved_experiment = opik_client.get_experiment_by_name(experiment_name)
    experiment_items_contents = retrieved_experiment.get_items()
    assert len(experiment_items_contents) == 2, (
        f"Expected 2 experiment items, but got {len(experiment_items_contents)}. "
        f"Experiment items: {experiment_items_contents}"
    )

    EXPECTED_EXPERIMENT_ITEMS_CONTENT = [
        experiment_item.ExperimentItemContent(
            id=ANY_BUT_NONE,
            dataset_item_id=ANY_BUT_NONE,
            trace_id=ANY_BUT_NONE,
            dataset_item_data={
                "input": {"question": "What is the of capital of France?"},
                "expected_model_output": {"output": "Paris"},
                "id": ANY_BUT_NONE,
            },
            evaluation_task_output={"output": "Paris"},
            feedback_scores=[
                {
                    "category_name": None,
                    "name": "equals_metric",
                    "reason": None,
                    "value": 1.0,
                }
            ],
        ),
        experiment_item.ExperimentItemContent(
            id=ANY_BUT_NONE,
            dataset_item_id=ANY_BUT_NONE,
            trace_id=ANY_BUT_NONE,
            dataset_item_data={
                "input": {"question": "What is the of capital of Germany?"},
                "expected_model_output": {"output": "Berlin"},
                "id": ANY_BUT_NONE,
            },
            evaluation_task_output={"output": "Berlin"},
            feedback_scores=[
                {
                    "category_name": None,
                    "name": "equals_metric",
                    "reason": None,
                    "value": 1.0,
                }
            ],
        ),
    ]
    assert_equal(
        sorted(
            EXPECTED_EXPERIMENT_ITEMS_CONTENT,
            key=lambda item: str(item.dataset_item_data),
        ),
        sorted(experiment_items_contents, key=lambda item: str(item.dataset_item_data)),
    )


# --- sdks/python/tests/e2e/evaluation/test_evaluate_filter_string.py ---

def test_evaluate__with_filter_string__filters_dataset_items(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """Test that evaluate correctly filters dataset items using filter_string."""
    project_name = "test_project_evaluate_filter_string"
    dataset = opik_client.create_dataset(dataset_name, project_name=project_name)

    dataset_items = [
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the capital of France?"},
            "expected_model_output": {"output": "Paris"},
            "category": "geography",
        },
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is 2+2?"},
            "expected_model_output": {"output": "4"},
            "category": "math",
        },
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the capital of Germany?"},
            "expected_model_output": {"output": "Berlin"},
            "category": "geography",
        },
    ]

    dataset.insert(dataset_items)

    def task(item: Dict[str, Any]):
        if item["input"] == {"question": "What is the capital of France?"}:
            return {"output": "Paris"}
        if item["input"] == {"question": "What is the capital of Germany?"}:
            return {"output": "Berlin"}
        if item["input"] == {"question": "What is 2+2?"}:
            return {"output": "4"}

        raise AssertionError(
            f"Task received dataset item with an unexpected input: {item['input']}"
        )

    equals_metric = metrics.Equals()
    opik.evaluate(
        dataset=dataset,
        task=task,
        scoring_metrics=[equals_metric],
        experiment_name=experiment_name,
        scoring_key_mapping={
            "reference": lambda x: x["expected_model_output"]["output"],
        },
        dataset_filter_string='data.category = "geography"',
        project_name=project_name,
    )

    opik.flush_tracker()

    retrieved_experiment = opik_client.get_experiment_by_name(
        experiment_name, project_name=project_name
    )
    experiment_items_contents = retrieved_experiment.get_items()
    assert len(experiment_items_contents) == 2, (
        f"Expected 2 experiment items (filtered by geography category), but got {len(experiment_items_contents)}. "
        f"Experiment items: {experiment_items_contents}"
    )
    assert retrieved_experiment.project_name == project_name

def test_evaluate_optimization_trial__with_filter_string__filters_dataset_items(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """Test that evaluate_optimization_trial correctly filters dataset items using filter_string."""
    project_name = "test_project_evaluate_optimization_trial_filter_string"
    dataset = opik_client.create_dataset(dataset_name, project_name=project_name)

    dataset_items = [
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the capital of France?"},
            "expected_model_output": {"output": "Paris"},
            "category": "geography",
        },
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is 2+2?"},
            "expected_model_output": {"output": "4"},
            "category": "math",
        },
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the capital of Germany?"},
            "expected_model_output": {"output": "Berlin"},
            "category": "geography",
        },
    ]

    dataset.insert(dataset_items)

    def task(item: Dict[str, Any]):
        if item["input"] == {"question": "What is the capital of France?"}:
            return {"output": "Paris"}
        if item["input"] == {"question": "What is the capital of Germany?"}:
            return {"output": "Berlin"}
        if item["input"] == {"question": "What is 2+2?"}:
            return {"output": "4"}

        raise AssertionError(
            f"Task received dataset item with an unexpected input: {item['input']}"
        )

    equals_metric = metrics.Equals()
    evaluator_module.evaluate_optimization_trial(
        optimization_id=id_helpers.generate_id(),
        dataset=dataset,
        task=task,
        scoring_metrics=[equals_metric],
        experiment_name=experiment_name,
        scoring_key_mapping={
            "reference": lambda x: x["expected_model_output"]["output"],
        },
        dataset_filter_string='data.category = "math"',
        project_name=project_name,
    )

    opik.flush_tracker()

    retrieved_experiment = opik_client.get_experiment_by_name(
        experiment_name, project_name=project_name
    )
    experiment_items_contents = retrieved_experiment.get_items()
    assert len(experiment_items_contents) == 1, (
        f"Expected 1 experiment item (filtered by math category), but got {len(experiment_items_contents)}. "
        f"Experiment items: {experiment_items_contents}"
    )
    assert retrieved_experiment.project_name == project_name


# --- sdks/python/tests/e2e/evaluation/test_experiment_evaluate.py ---

def test_experiment_creation_via_evaluate_function__single_prompt_arg_used__filter_dataset_items_by_id(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    project_name = "test-project-experiment_creation_via_evaluate_function"
    dataset = opik_client.create_dataset(dataset_name, project_name=project_name)

    dataset_items = [
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the of capital of France?"},
            "expected_model_output": {"output": "Paris"},
        },
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the of capital of Germany?"},
            "expected_model_output": {"output": "Berlin"},
        },
        {
            "id": id_helpers.generate_id(),
            "input": {"question": "What is the of capital of Poland?"},
            "expected_model_output": {"output": "Warsaw"},
        },
    ]

    dataset.insert(dataset_items)

    def task(item: Dict[str, Any]):
        if item["input"] == {"question": "What is the of capital of France?"}:
            return {"output": "Paris"}
        if item["input"] == {"question": "What is the of capital of Germany?"}:
            return {"output": "Berlin"}
        if item["input"] == {"question": "What is the of capital of Poland?"}:
            return {"output": "Krakow"}

        raise AssertionError(
            f"Task received dataset item with an unexpected input: {item['input']}"
        )

    prompt = Prompt(
        name=f"test-experiment-prompt-{random_chars()}",
        prompt=f"test-experiment-prompt-template-{random_chars()}",
    )

    dataset_item_ids = [item["id"] for item in dataset_items]
    dataset_item_ids.pop(2)
    # add non existing id
    dataset_item_ids.append(id_helpers.generate_id())

    equals_metric = metrics.Equals()
    evaluation_result = opik.evaluate(
        dataset=dataset,
        task=task,
        scoring_metrics=[equals_metric],
        experiment_name=experiment_name,
        experiment_config={
            "model_name": "gpt-3.5",
        },
        scoring_key_mapping={
            "reference": lambda x: x["expected_model_output"]["output"],
        },
        prompt=prompt,
        dataset_item_ids=dataset_item_ids,
        project_name=project_name,
    )

    opik.flush_tracker()

    verifiers.verify_experiment(
        opik_client=opik_client,
        id=evaluation_result.experiment_id,
        experiment_name=evaluation_result.experiment_name,
        experiment_metadata={"model_name": "gpt-3.5"},
        traces_amount=2,  # one trace per dataset item
        feedback_scores_amount=1,
        prompts=[prompt],
        project_name=project_name,
    )

    assert evaluation_result.dataset_id == dataset.id, (
        f"Expected evaluation result dataset_id '{dataset.id}', but got '{evaluation_result.dataset_id}'"
    )

    retrieved_experiments = opik_client.get_experiments_by_name(
        experiment_name, project_name=project_name
    )
    assert len(retrieved_experiments) == 1, (
        f"Expected 1 experiment, but got {len(retrieved_experiments)}. "
        f"Experiments: {retrieved_experiments}"
    )
    retrieved_experiment = retrieved_experiments[0]
    experiment_items_contents = retrieved_experiment.get_items()
    assert len(experiment_items_contents) == 2, (
        f"Expected 2 experiment items, but got {len(experiment_items_contents)}. "
        f"Experiment items: {experiment_items_contents}"
    )

    EXPECTED_EXPERIMENT_ITEMS_CONTENT = [
        experiment_item.ExperimentItemContent(
            id=ANY_BUT_NONE,
            dataset_item_id=ANY_BUT_NONE,
            trace_id=ANY_BUT_NONE,
            dataset_item_data={
                "input": {"question": "What is the of capital of France?"},
                "expected_model_output": {"output": "Paris"},
                "id": ANY_BUT_NONE,
            },
            evaluation_task_output={"output": "Paris"},
            feedback_scores=[
                {
                    "category_name": None,
                    "name": "equals_metric",
                    "reason": None,
                    "value": 1.0,
                }
            ],
        ),
        experiment_item.ExperimentItemContent(
            id=ANY_BUT_NONE,
            dataset_item_id=ANY_BUT_NONE,
            trace_id=ANY_BUT_NONE,
            dataset_item_data={
                "input": {"question": "What is the of capital of Germany?"},
                "expected_model_output": {"output": "Berlin"},
                "id": ANY_BUT_NONE,
            },
            evaluation_task_output={"output": "Berlin"},
            feedback_scores=[
                {
                    "category_name": None,
                    "name": "equals_metric",
                    "reason": None,
                    "value": 1.0,
                }
            ],
        ),
    ]
    assert_equal(
        sorted(
            EXPECTED_EXPERIMENT_ITEMS_CONTENT,
            key=lambda item: str(item.dataset_item_data),
        ),
        sorted(experiment_items_contents, key=lambda item: str(item.dataset_item_data)),
    )


# --- sdks/python/tests/e2e/evaluation/test_test_suite.py ---

def test_test_suite__item_level_assertions__feedback_scores_created(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    Main flow: Items have their own assertions.

    Each item can have different assertions to verify.

    Expected behavior:
    - Each item is evaluated using its own assertions
    - Feedback scores are created with assertion text as the score name
    - Score values are boolean (True=1.0, False=0.0)
    """
    geography_assertion = (
        "The response correctly identifies Paris as the capital of France"
    )
    math_assertion = "The response correctly states that 2 + 2 equals 4"

    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test item-level assertions",
    )

    suite.insert(
        [
            {
                "data": {"input": {"question": "What is the capital of France?"}},
                "assertions": [geography_assertion],
            },
            {
                "data": {"input": {"question": "What is 2 + 2?"}},
                "assertions": [math_assertion],
            },
        ]
    )

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        question = item["input"]["question"]
        if "France" in question:
            return {"input": item["input"], "output": "The capital of France is Paris."}
        if "2 + 2" in question:
            return {"input": item["input"], "output": "2 + 2 equals 4."}
        return {"input": item["input"], "output": "Unknown"}

    suite_result = opik.run_tests(
        test_suite=suite,
        task=task,
        experiment_name=experiment_name,
        verbose=0,
    )
    opik.flush_tracker()

    verifiers.verify_test_suite_result(
        opik_client=opik_client,
        suite_result=suite_result,
        items_total=2,
        items_passed=2,
        experiment_items_count=2,
        total_feedback_scores=2,  # 1 assertion per item * 2 items
        expected_score_names={geography_assertion, math_assertion},
    )

    # Verify score values are boolean (True=1.0, False=0.0)
    retrieved_experiment = opik_client.get_experiment_by_name(experiment_name)
    for exp_item in retrieved_experiment.get_items():
        for score in exp_item.feedback_scores:
            assert score["value"] in [0.0, 1.0, True, False], (
                f"Score value should be boolean, got {score['value']}"
            )

def test_test_suite__multiple_assertions_per_item__all_scores_created(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    Test that multiple assertions on a single item create multiple
    feedback scores, each evaluated independently.
    """
    assertion_1 = "The response is factually correct"
    assertion_2 = "The response is concise and clear"

    project_name = "project_test_test_suite__multiple_assertions_per_item"
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test multiple assertions per item",
        project_name=project_name,
    )

    suite.insert(
        [
            {
                "data": {"input": {"question": "What is the capital of France?"}},
                "assertions": [assertion_1, assertion_2],
            }
        ]
    )

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        return {"input": item["input"], "output": "Paris is the capital of France."}

    suite_result = opik.run_tests(
        test_suite=suite,
        task=task,
        experiment_name=experiment_name,
        verbose=0,
    )
    opik.flush_tracker()

    verifiers.verify_test_suite_result(
        opik_client=opik_client,
        suite_result=suite_result,
        items_total=1,
        items_passed=1,
        experiment_items_count=1,
        total_feedback_scores=2,  # 2 assertions on 1 experiment item
        expected_score_names={assertion_1, assertion_2},
        project_name=project_name,
    )

def test_test_suite__suite_level_assertions__applied_to_all_items(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    Test that suite-level assertions are applied to every item.
    """
    suite_assertion = "The response is helpful and informative"

    project_name = "project_test_test_suite__suite_level_assertions"
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test suite-level assertions",
        global_assertions=[suite_assertion],
        project_name=project_name,
    )

    suite.insert(
        [
            {"data": {"input": {"question": "What is the capital of France?"}}},
            {"data": {"input": {"question": "What is 2 + 2?"}}},
        ]
    )

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        question = item["input"]["question"]
        if "France" in question:
            return {"input": item["input"], "output": "The capital of France is Paris."}
        return {"input": item["input"], "output": "2 + 2 equals 4."}

    suite_result = opik.run_tests(
        test_suite=suite,
        task=task,
        experiment_name=experiment_name,
        verbose=0,
    )
    opik.flush_tracker()

    verifiers.verify_test_suite_result(
        opik_client=opik_client,
        suite_result=suite_result,
        items_total=2,
        experiment_items_count=2,
        total_feedback_scores=2,  # 1 assertion * 2 experiment items
        expected_score_names={suite_assertion},
        project_name=project_name,
    )

def test_test_suite__combined_suite_and_item_level_assertions__all_scores_created(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    Test that suite-level and item-level assertions are combined:
    total feedback scores = suite-level assertions + item-level assertions.
    """
    suite_assertion = "The response is helpful and informative"
    item_assertion = "The response correctly identifies Paris as the capital"

    project_name = "project_test_test_suite__combined_suite_and_item_level_assertions"
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test combined suite and item level assertions",
        global_assertions=[suite_assertion],
        project_name=project_name,
    )

    suite.insert(
        [
            {
                "data": {"input": {"question": "What is the capital of France?"}},
                "assertions": [item_assertion],
            }
        ]
    )

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        return {"input": item["input"], "output": "The capital of France is Paris."}

    suite_result = opik.run_tests(
        test_suite=suite,
        task=task,
        experiment_name=experiment_name,
        verbose=0,
    )
    opik.flush_tracker()

    verifiers.verify_test_suite_result(
        opik_client=opik_client,
        suite_result=suite_result,
        items_total=1,
        experiment_items_count=1,
        total_feedback_scores=2,  # 1 suite + 1 item assertion
        expected_score_names={suite_assertion, item_assertion},
        project_name=project_name,
    )

def test_test_suite__no_assertions_default_policy__items_pass_with_single_run(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    Edge case: Items without assertions pass by default, and the default
    execution policy runs each item exactly once with pass_threshold=1.

    Expected behavior:
    - No assertions to check -> items pass
    - Default runs_per_item=1, pass_threshold=1
    - No feedback scores created
    """
    project_name = "project_test_test_suite__no_assertions_default_policy"
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test items without assertions and default policy",
        project_name=project_name,
    )

    suite.insert(
        [
            {
                "data": {
                    "input": {"question": "What is the capital of France?"},
                    "reference": "Paris",
                },
            },
            {
                "data": {
                    "input": {"question": "What is the capital of Germany?"},
                    "reference": "Berlin",
                },
            },
        ]
    )

    call_count = ThreadSafeCounter()

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        call_count.increment()
        return {"input": item["input"], "output": "Some response"}

    suite_result = opik.run_tests(
        test_suite=suite,
        task=task,
        experiment_name=experiment_name,
        verbose=0,
    )
    opik.flush_tracker()

    # Default: runs_per_item=1, so 2 items = 2 calls
    assert call_count.value == 2

    verifiers.verify_test_suite_result(
        opik_client=opik_client,
        suite_result=suite_result,
        items_total=2,
        items_passed=2,
        experiment_items_count=2,
        total_feedback_scores=0,
        project_name=project_name,
    )

    verifiers.verify_experiment(
        opik_client=opik_client,
        id=suite_result.experiment_id,
        experiment_name=suite_result.experiment_name,
        experiment_metadata=None,
        traces_amount=2,
        feedback_scores_amount=0,
        project_name=project_name,
    )

    # Verify default pass_threshold=1 and runs_total=1
    for item_result in suite_result.item_results.values():
        assert item_result.runs_total == 1
        assert item_result.pass_threshold == 1

def test_test_suite__execution_policy_runs_per_item__task_called_multiple_times(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    Test that runs_per_item causes multiple task executions per item.
    """
    project_name = "project_test_test_suite__execution_policy_runs_per_item"
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test runs_per_item execution policy",
        global_execution_policy={"runs_per_item": 2, "pass_threshold": 1},
        project_name=project_name,
    )

    suite.insert(
        [
            {
                "data": {"input": {"question": "What is the capital of France?"}},
            }
        ]
    )

    call_count = ThreadSafeCounter()

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        call_count.increment()
        return {"input": item["input"], "output": "Paris"}

    suite_result = opik.run_tests(
        test_suite=suite,
        task=task,
        experiment_name=experiment_name,
        verbose=0,
    )
    opik.flush_tracker()

    assert call_count.value == 2

    verifiers.verify_test_suite_result(
        opik_client=opik_client,
        suite_result=suite_result,
        items_total=1,
        items_passed=1,
        experiment_items_count=2,
        total_feedback_scores=0,
        project_name=project_name,
    )

    verifiers.verify_experiment(
        opik_client=opik_client,
        id=suite_result.experiment_id,
        experiment_name=suite_result.experiment_name,
        experiment_metadata=None,
        traces_amount=2,
        feedback_scores_amount=0,
        project_name=project_name,
    )

    item_result = list(suite_result.item_results.values())[0]
    assert item_result.runs_total == 2
    assert item_result.pass_threshold == 1

def test_test_suite__item_level_execution_policy__overrides_suite_policy(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    Test that item-level execution policy overrides suite-level policy.
    """
    project_name = "project_test_test_suite__item_level_execution_policy"
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test item-level execution policy override",
        global_execution_policy={"runs_per_item": 1, "pass_threshold": 1},
        project_name=project_name,
    )

    # Item 1: uses suite-level policy (runs_per_item=1)
    # Item 2: overrides with item-level policy (runs_per_item=3)
    suite.insert(
        [
            {
                "data": {"input": {"question": "What is the capital of France?"}},
            },
            {
                "data": {"input": {"question": "What is the capital of Germany?"}},
                "execution_policy": {"runs_per_item": 3, "pass_threshold": 2},
            },
        ]
    )

    france_count = ThreadSafeCounter()
    germany_count = ThreadSafeCounter()

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        question = item["input"]["question"]
        if "France" in question:
            france_count.increment()
        elif "Germany" in question:
            germany_count.increment()
        return {"input": item["input"], "output": "Answer"}

    suite_result = opik.run_tests(
        test_suite=suite,
        task=task,
        experiment_name=experiment_name,
        verbose=0,
    )
    opik.flush_tracker()

    assert france_count.value == 1
    assert germany_count.value == 3

    verifiers.verify_test_suite_result(
        opik_client=opik_client,
        suite_result=suite_result,
        items_total=2,
        items_passed=2,
        experiment_items_count=4,  # 1 + 3
        total_feedback_scores=0,
        project_name=project_name,
    )

    verifiers.verify_experiment(
        opik_client=opik_client,
        id=suite_result.experiment_id,
        experiment_name=suite_result.experiment_name,
        experiment_metadata=None,
        traces_amount=4,
        feedback_scores_amount=0,
        project_name=project_name,
    )

    # Verify item-level pass_threshold is used
    for item_result in suite_result.item_results.values():
        if item_result.runs_total == 3:
            assert item_result.pass_threshold == 2
        else:
            assert item_result.pass_threshold == 1

def test_test_suite__assertion_fails__item_fails(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    Test that items fail when assertions fail.
    """
    failing_assertion = "The response correctly states that 2 + 2 equals 5"

    project_name = "project_test_test_suite__assertion_fails"
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test assertion failure",
        project_name=project_name,
    )

    suite.insert(
        [
            {
                "data": {"input": {"question": "What is 2 + 2?"}},
                "assertions": [failing_assertion],
            }
        ]
    )

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        return {"input": item["input"], "output": "2 + 2 equals 4."}

    suite_result = opik.run_tests(
        test_suite=suite,
        task=task,
        experiment_name=experiment_name,
        verbose=0,
    )
    opik.flush_tracker()

    verifiers.verify_test_suite_result(
        opik_client=opik_client,
        suite_result=suite_result,
        items_total=1,
        items_passed=0,
        experiment_items_count=1,
        total_feedback_scores=1,
        expected_score_names={failing_assertion},
        project_name=project_name,
    )

    # Additionally verify the assertion result indicates failure
    retrieved_experiment = opik_client.get_experiment_by_name(
        experiment_name, project_name=project_name
    )
    items = retrieved_experiment.get_items()
    assert len(items) > 0, "Expected at least 1 experiment item"
    assert len(items[0].assertion_results) > 0, "Expected at least 1 assertion result"
    assertion = items[0].assertion_results[0]
    assert assertion["passed"] is False, (
        f"Expected failing assertion (passed=False), got {assertion['passed']}"
    )

def test_test_suite__pass_threshold_not_met__item_fails(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    Test that items fail when pass_threshold is not met across multiple runs.

    With runs_per_item=3, pass_threshold=2: only the first run returns a
    correct answer, so at most 1 run passes (< threshold of 2).
    """
    project_name = "project_test_test_suite__pass_threshold_not_met"
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test pass threshold failure",
        global_assertions=["The response correctly states that 2 + 2 equals 4"],
        global_execution_policy={"runs_per_item": 3, "pass_threshold": 2},
        project_name=project_name,
    )

    suite.insert([{"data": {"input": {"question": "What is 2 + 2?"}}}])

    call_count = ThreadSafeCounter()

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        n = call_count.increment()
        # Only first run returns correct answer
        if n == 1:
            return {"input": item["input"], "output": "2 + 2 equals 4."}
        return {"input": item["input"], "output": "I don't know."}

    suite_result = opik.run_tests(
        test_suite=suite,
        task=task,
        experiment_name=experiment_name,
        verbose=0,
    )
    opik.flush_tracker()

    verifiers.verify_test_suite_result(
        opik_client=opik_client,
        suite_result=suite_result,
        items_total=1,
        items_passed=0,
        project_name=project_name,
    )

    item_result = list(suite_result.item_results.values())[0]
    assert item_result.passed is False
    assert item_result.runs_total == 3
    assert item_result.pass_threshold == 2

def test_test_suite__multiple_assertions_multiple_runs__pass_threshold_logic(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    Comprehensive pass/fail logic test:
    - 1 item, 3 assertions, runs_per_item=3, pass_threshold=2
    - Consistent correct answers -> all runs pass -> item passes

    Pass/fail logic:
    1. A RUN passes if ALL assertions in that run pass
    2. An ITEM passes if runs_passed >= pass_threshold
    3. The SUITE passes if all items pass
    """
    assertion_1 = "The response mentions Paris"
    assertion_2 = "The response mentions France"
    assertion_3 = "The response is factually correct"

    project_name = "project_test_test_suite__multiple_assertions_multiple_runs"
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test multiple assertions with multiple runs",
        global_assertions=[assertion_1, assertion_2, assertion_3],
        global_execution_policy={"runs_per_item": 3, "pass_threshold": 2},
        project_name=project_name,
    )

    suite.insert([{"data": {"input": {"question": "What is the capital of France?"}}}])

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        return {"input": item["input"], "output": "The capital of France is Paris."}

    suite_result = opik.run_tests(
        test_suite=suite,
        task=task,
        experiment_name=experiment_name,
        verbose=0,
    )
    opik.flush_tracker()

    assert suite_result.pass_rate == 1.0

    verifiers.verify_test_suite_result(
        opik_client=opik_client,
        suite_result=suite_result,
        items_total=1,
        items_passed=1,
        experiment_items_count=3,  # 1 item * 3 runs
        total_feedback_scores=9,  # 3 assertions * 3 runs
        expected_score_names={assertion_1, assertion_2, assertion_3},
        project_name=project_name,
    )

    item_result = list(suite_result.item_results.values())[0]
    assert item_result.runs_total == 3
    assert item_result.pass_threshold == 2
    assert item_result.runs_passed >= 2
    assert item_result.passed is True

    # Verify each experiment item has exactly 3 assertion results (one per assertion)
    retrieved_experiment = opik_client.get_experiment_by_name(
        experiment_name, project_name=project_name
    )
    assert retrieved_experiment.project_name == project_name
    for exp_item in retrieved_experiment.get_items():
        assert exp_item.assertion_results is not None
        assert len(exp_item.assertion_results) == 3, (
            f"Expected 3 assertion results per run, got {len(exp_item.assertion_results)}"
        )
        assertion_names = {ar["value"] for ar in exp_item.assertion_results}
        assert assertion_names == {assertion_1, assertion_2, assertion_3}, (
            f"Expected all 3 assertion names on each run, got {assertion_names}"
        )

def test_test_suite__create_get_and_run__end_to_end(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    End-to-end test: create a suite, retrieve it via get_test_suite(),
    then run it. Verifies that suite-level config survives the round-trip.
    """
    suite_assertion = "The response correctly identifies Paris as the capital of France"
    item_assertion = "Response is correct"
    project_name = "project_test_test_suite__create_get_and_run__end_to_end"

    # 1. Create suite with assertions + execution_policy
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Persistence test suite",
        global_assertions=[suite_assertion],
        global_execution_policy={"runs_per_item": 2, "pass_threshold": 1},
        project_name=project_name,
    )

    suite.insert(
        [
            {
                "data": {"input": {"question": "What is the capital of France?"}},
                "assertions": [item_assertion],
                "description": "Geography: France capital",
            },
            {
                "data": {"input": {"question": "What is the capital of Germany?"}},
                "description": "Geography: Germany capital",
            },
        ]
    )

    # 2. Retrieve from backend (simulates a fresh client loading existing suite)
    retrieved_suite = opik_client.get_test_suite(
        name=dataset_name, project_name=project_name
    )

    # Verify item descriptions survived the round-trip
    retrieved_items = retrieved_suite.get_items()
    retrieved_descriptions = {i["description"] for i in retrieved_items}
    assert "Geography: France capital" in retrieved_descriptions
    assert "Geography: Germany capital" in retrieved_descriptions

    # Verify item-level assertions survived the round-trip
    items_with_assertions = [i for i in retrieved_items if len(i["assertions"]) > 0]
    assert len(items_with_assertions) == 1
    assert items_with_assertions[0]["assertions"] == [item_assertion]

    # 3. Run the retrieved suite — assertions/execution_policy come from BE
    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        question = item["input"]["question"]
        if "France" in question:
            return {"input": item["input"], "output": "The capital of France is Paris."}
        return {"input": item["input"], "output": "The capital of Germany is Berlin."}

    suite_result = opik.run_tests(
        test_suite=retrieved_suite,
        task=task,
        experiment_name=experiment_name,
        verbose=0,
    )
    opik.flush_tracker()

    # Verify suite ran with persisted execution policy (runs_per_item=2)
    verifiers.verify_test_suite_result(
        opik_client=opik_client,
        suite_result=suite_result,
        items_total=2,
        experiment_items_count=4,  # 2 items * 2 runs
        total_feedback_scores=6,  # France: 2 runs * 2 assertions + Germany: 2 runs * 1 assertion
        expected_score_names={suite_assertion, item_assertion},
        project_name=project_name,
    )

    for item_result in suite_result.item_results.values():
        assert item_result.runs_total == 2
        assert item_result.pass_threshold == 1

def test_test_suite__get_global_execution_policy__returns_persisted_policy(
    opik_client: opik.Opik, dataset_name: str
):
    """
    Test that get_execution_policy() returns the persisted execution policy.
    """
    project_name = "project_test_test_suite__get_execution_policy"
    opik_client.create_test_suite(
        name=dataset_name,
        description="Test get_execution_policy",
        global_execution_policy={"runs_per_item": 5, "pass_threshold": 3},
        project_name=project_name,
    )

    # Retrieve from BE to verify persistence
    retrieved_suite = opik_client.get_test_suite(
        name=dataset_name, project_name=project_name
    )
    assert retrieved_suite.project_name == project_name

    policy = retrieved_suite.get_global_execution_policy()
    assert policy["runs_per_item"] == 5
    assert policy["pass_threshold"] == 3

def test_test_suite__update_test_settings__changes_assertions_and_policy(
    opik_client: opik.Opik, dataset_name: str
):
    """
    Test that update() changes suite-level assertions and execution policy.
    """
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test update",
        global_assertions=["Response is helpful"],
        global_execution_policy={"runs_per_item": 1, "pass_threshold": 1},
    )

    # Verify initial state
    assertions = suite.get_global_assertions()
    assert set(assertions) == {"Response is helpful"}

    policy = suite.get_global_execution_policy()
    assert policy["runs_per_item"] == 1

    # Update with new assertions and policy
    suite.update_test_settings(
        global_assertions=["Response is accurate", "Response is concise"],
        global_execution_policy={"runs_per_item": 3, "pass_threshold": 2},
    )

    # Retrieve from BE to verify persistence
    retrieved_suite = opik_client.get_test_suite(name=dataset_name)

    updated_assertions = retrieved_suite.get_global_assertions()
    assert set(updated_assertions) == {
        "Response is accurate",
        "Response is concise",
    }

    updated_policy = retrieved_suite.get_global_execution_policy()
    assert updated_policy["runs_per_item"] == 3
    assert updated_policy["pass_threshold"] == 2

def test_get_or_create_test_suite__existing_with_different_policy__does_not_modify(
    opik_client: opik.Opik, dataset_name: str
):
    """
    Test that get_or_create_test_suite does not modify an existing suite's
    execution policy even when different values are passed.
    """
    opik_client.create_test_suite(
        name=dataset_name,
        description="Original suite",
        global_assertions=["Response is helpful"],
        global_execution_policy={"runs_per_item": 1, "pass_threshold": 1},
    )

    opik_client.get_or_create_test_suite(
        name=dataset_name,
        global_execution_policy={"runs_per_item": 5, "pass_threshold": 3},
    )

    retrieved = opik_client.get_test_suite(name=dataset_name)

    policy = retrieved.get_global_execution_policy()
    assert policy["runs_per_item"] == 1
    assert policy["pass_threshold"] == 1

    assertions = retrieved.get_global_assertions()
    assert set(assertions) == {"Response is helpful"}

def test_test_suite__update_test_settings_assertions_only__keeps_existing_policy(
    opik_client: opik.Opik, dataset_name: str
):
    """
    Test that update() with only assertions keeps the existing execution policy.
    """
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test partial update",
        global_assertions=["Response is helpful"],
        global_execution_policy={"runs_per_item": 3, "pass_threshold": 2},
    )

    suite.update_test_settings(global_assertions=["Response is accurate"])

    retrieved = opik_client.get_test_suite(name=dataset_name)

    assertions = retrieved.get_global_assertions()
    assert set(assertions) == {"Response is accurate"}

    policy = retrieved.get_global_execution_policy()
    assert policy["runs_per_item"] == 3
    assert policy["pass_threshold"] == 2

def test_test_suite__update_test_settings_policy_only__keeps_existing_assertions(
    opik_client: opik.Opik, dataset_name: str
):
    """
    Test that update() with only execution_policy keeps existing assertions.
    """
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test partial update",
        global_assertions=["Response is helpful", "Response is accurate"],
        global_execution_policy={"runs_per_item": 1, "pass_threshold": 1},
    )

    suite.update_test_settings(
        global_execution_policy={"runs_per_item": 5, "pass_threshold": 3}
    )

    retrieved = opik_client.get_test_suite(name=dataset_name)

    policy = retrieved.get_global_execution_policy()
    assert policy["runs_per_item"] == 5
    assert policy["pass_threshold"] == 3

    assertions = retrieved.get_global_assertions()
    assert set(assertions) == {
        "Response is helpful",
        "Response is accurate",
    }

def test_test_suite__update_test_settings_with_empty_assertions__clears_assertions(
    opik_client: opik.Opik, dataset_name: str
):
    """
    Test that update(assertions=[]) clears all suite-level assertions.
    """
    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test clearing assertions",
        global_assertions=["Response is helpful", "Response is accurate"],
        global_execution_policy={"runs_per_item": 1, "pass_threshold": 1},
    )

    assert len(suite.get_global_assertions()) == 2

    suite.update_test_settings(global_assertions=[])

    retrieved = opik_client.get_test_suite(name=dataset_name)
    assert retrieved.get_global_assertions() == []

    policy = retrieved.get_global_execution_policy()
    assert policy["runs_per_item"] == 1
    assert policy["pass_threshold"] == 1

def test_test_suite__insert_batch__all_items_persisted(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    Test that insert() adds multiple items in a single batch.
    """
    assertion = "The response is factually correct"

    suite = opik_client.create_test_suite(
        name=dataset_name,
        description="Test batch insert",
    )

    suite.insert(
        [
            {
                "data": {"input": {"question": "What is the capital of France?"}},
                "assertions": [assertion],
            },
            {
                "data": {"input": {"question": "What is the capital of Germany?"}},
                "assertions": [assertion],
            },
            {
                "data": {"input": {"question": "What is the capital of Spain?"}},
                "assertions": [assertion],
            },
        ]
    )

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        answers = {
            "What is the capital of France?": "Paris",
            "What is the capital of Germany?": "Berlin",
            "What is the capital of Spain?": "Madrid",
        }
        question = item["input"]["question"]
        return {"input": item["input"], "output": answers.get(question, "Unknown")}

    suite_result = opik.run_tests(
        test_suite=suite,
        task=task,
        experiment_name=experiment_name,
        verbose=0,
    )
    opik.flush_tracker()

    verifiers.verify_test_suite_result(
        opik_client=opik_client,
        suite_result=suite_result,
        items_total=3,
        items_passed=3,
        experiment_items_count=3,
        total_feedback_scores=3,
        expected_score_names={assertion},
    )

def test_get_test_suite_experiments__returns_experiments(
    opik_client: opik.Opik, dataset_name: str, experiment_name: str
):
    """
    Test that get_test_suite_experiments() returns experiments run on the suite.
    """
    suite = opik_client.create_test_suite(name=dataset_name)
    suite.insert([{"data": {"input": {"question": "Hello"}}}])

    def task(item: Dict[str, Any]) -> Dict[str, Any]:
        return {"input": item["input"], "output": "World"}

    opik.run_tests(
        test_suite=suite, task=task, experiment_name=experiment_name, verbose=0
    )
    opik.flush_tracker()

    experiments = opik_client.get_test_suite_experiments(name=dataset_name)
    experiment_names = {e.name for e in experiments}

    assert experiment_name in experiment_names

def test_test_suite__create_without_metadata_then_update__metadata_persisted(
    opik_client: opik.Opik, dataset_name: str
):
    """
    OPIK-5815: create a suite without evaluators or execution_policy,
    then update with assertions and policy. Verifies update works
    without an initial version.
    """
    assertion = "The response is factually correct"

    suite = opik_client.create_test_suite(name=dataset_name)

    assert suite.get_version_info() is None

    suite.update_test_settings(
        global_assertions=[assertion],
        global_execution_policy={"runs_per_item": 2, "pass_threshold": 1},
    )

    reloaded = opik_client.get_test_suite(name=dataset_name)
    assert reloaded.get_global_assertions() == [assertion]
    assert reloaded.get_global_execution_policy() == {
        "runs_per_item": 2,
        "pass_threshold": 1,
    }


# --- sdks/python/tests/e2e/evaluation/test_threads_evaluate.py ---

def test_evaluate_threads__happy_path(
    opik_client, active_thread_and_project_name, eval_project_name
):
    active_thread, project_name = active_thread_and_project_name
    # wait for active threads to propagate
    if not synchronization.until(
        lambda: _one_thread_is_active(project_name, opik_client), max_try_seconds=30
    ):
        raise AssertionError(f"Failed to create threads in project '{project_name}'")

    # evaluate_threads requires closed threads (SDK constraint)
    opik_client.rest_client.traces.close_trace_thread(
        project_name=project_name, thread_id=active_thread
    )
    if not synchronization.until(
        lambda: _all_threads_closed(project_name, opik_client), max_try_seconds=10
    ):
        raise AssertionError(
            f"Failed to get closed threads from project '{project_name}'"
        )

    metrics_ = [
        metrics.ConversationalCoherenceMetric(window_size=2),
        metrics.UserFrustrationMetric(window_size=2),
        metrics.SessionCompletenessQuality(),
    ]

    result = evaluator.evaluate_threads(
        project_name=project_name,
        filter_string=f'id = "{active_thread}"',
        metrics=metrics_,
        eval_project_name=eval_project_name,
        trace_input_transform=lambda x: x["input"],
        trace_output_transform=lambda x: x["output"],
        verbose=1,
    )

    assert result is not None
    assert len(result.results) == 1  # we have only one thread

    thread_result = result.results[0]
    assert thread_result.thread_id == active_thread
    assert len(thread_result.scores) == len(metrics_)

    feedback_scores = [
        FeedbackScoreDict(
            id=active_thread,
            name=score.name,
            value=score.value,
            reason=score.reason.strip(),
            category_name=None,
        )
        for score in thread_result.scores
        if not score.scoring_failed
    ]

    verifiers.verify_thread(
        opik_client=opik_client,
        thread_id=active_thread,
        project_name=project_name,
        feedback_scores=feedback_scores,
    )

def test_evaluate_threads__no_truncation_for_long_traces(
    opik_client, temporary_project_name
):
    """E2E test verifying that long trace content is not truncated during evaluation.

    The test creates a trace with output exceeding 15,000 characters with a unique marker
    at the end, then verifies the marker is present in the transform function, proving
    that truncation did not occur.
    """
    thread_id = str(uuid.uuid4())[-6:]

    # Create a long output that exceeds the truncation threshold (~9935 chars)
    # with a unique marker at the end that would be lost if truncated
    marker = "UNIQUE_END_MARKER_XYZ123"
    long_content = "a" * 15000 + marker

    # Create a trace with very long output
    opik_client.trace(
        name=f"long-trace:{thread_id}",
        input={"input": "test input"},
        output={"output": long_content},
        project_name=temporary_project_name,
        thread_id=thread_id,
    )

    opik_client.flush()

    # Wait for thread to be created
    if not synchronization.until(
        lambda: _one_thread_is_active(temporary_project_name, opik_client),
        max_try_seconds=30,
    ):
        raise AssertionError(
            f"Failed to create thread in project '{temporary_project_name}'"
        )

    # evaluate_threads requires closed threads (SDK constraint)
    opik_client.rest_client.traces.close_trace_thread(
        project_name=temporary_project_name, thread_id=thread_id
    )
    if not synchronization.until(
        lambda: _all_threads_closed(temporary_project_name, opik_client),
        max_try_seconds=10,
    ):
        raise AssertionError(
            f"Failed to close thread in project '{temporary_project_name}'"
        )

    # Track what the transform receives
    received_outputs = []

    def input_transform(x):
        return x.get("input", "")

    def output_transform(x):
        """Transform that captures the output to verify it's not truncated."""
        # When truncated, x might be a string (malformed JSON) instead of dict
        if isinstance(x, str):
            # This is the bug! Truncation causes malformed JSON string
            received_outputs.append(f"TRUNCATED_STRING:{x[:100]}...")
            return "TRUNCATED"
        output = x.get("output", "")
        received_outputs.append(output)
        return output

    # Create a simple metric that just checks the output
    class ContentVerificationMetric(metrics.base_metric.BaseMetric):
        def __init__(self):
            super().__init__(
                name="content_verification",
                track=False,
            )

        def score(self, conversation, **ignored_kwargs):
            # Just return a dummy score - we're really testing the transform
            return metrics.score_result.ScoreResult(
                name=self.name,
                value=1.0,
                reason="Content verification",
            )

    # Run evaluation
    result = evaluator.evaluate_threads(
        project_name=temporary_project_name,
        filter_string=f'id = "{thread_id}"',
        metrics=[ContentVerificationMetric()],
        eval_project_name=temporary_project_name,
        trace_input_transform=input_transform,
        trace_output_transform=output_transform,
        verbose=0,
    )

    assert result is not None
    assert len(result.results) == 1

    # Verify that the transform received the full content with the marker
    assert len(received_outputs) > 0, "Transform should have been called"
    transformed_content = received_outputs[0]

    # This is the critical assertion: if truncation occurred, the marker would be missing
    assert marker in transformed_content, (
        f"Content was truncated! Expected marker '{marker}' not found. "
        f"Content length: {len(transformed_content)}, expected: {len(long_content)}. "
        f"Last 100 chars: {transformed_content[-100:]}"
    )

    # Also verify the full length is preserved
    assert len(transformed_content) == len(long_content), (
        f"Content length mismatch: got {len(transformed_content)}, "
        f"expected {len(long_content)}"
    )


# --- sdks/python/tests/e2e/runner/test_bridge_e2e.py ---

def test_bridge_command__unsigned__fails_with_auth_failed(
    api_client, bridge_runner_process: RunnerInfo
):
    cmd = _submit_and_wait(
        api_client,
        bridge_runner_process.runner_id,
        "Exec",
        {"command": "echo should-not-run"},
    )
    assert cmd.status == "failed"
    assert cmd.error["code"] == "auth_failed"

def test_bridge_command__invalid_hmac__fails_with_auth_failed(
    api_client, bridge_runner_process: RunnerInfo
):
    cmd = _submit_and_wait(
        api_client,
        bridge_runner_process.runner_id,
        "Exec",
        {"command": "echo should-not-run", "_hmac": "dGhpcyBpcyBub3QgYSB2YWxpZCBobWFj"},
    )
    assert cmd.status == "failed"
    assert cmd.error["code"] == "auth_failed"

def test_bridge_exec_echo(api_client, bridge_runner_process: RunnerInfo):
    """Submit a simple echo command and verify the result."""
    marker = f"bridge-e2e-{int(time.time())}"

    cmd = _submit_and_wait(
        api_client,
        bridge_runner_process.runner_id,
        "Exec",
        {"command": f"echo {marker}"},
        bridge_key=bridge_runner_process.bridge_key,
    )

    assert cmd.status == "completed"
    assert marker in cmd.result["stdout"]
    assert cmd.result["exit_code"] == 0

def test_bridge_exec_nonzero_exit(api_client, bridge_runner_process: RunnerInfo):
    """Verify non-zero exit codes are returned correctly."""
    cmd = _submit_and_wait(
        api_client,
        bridge_runner_process.runner_id,
        "Exec",
        {"command": "exit 42"},
        bridge_key=bridge_runner_process.bridge_key,
    )

    assert cmd.status == "completed"
    assert cmd.result["exit_code"] == 42

def test_bridge_exec_background(api_client, bridge_runner_process: RunnerInfo):
    """Submit a background command, verify PID is returned."""
    cmd = _submit_and_wait(
        api_client,
        bridge_runner_process.runner_id,
        "Exec",
        {"command": "sleep 30", "background": True},
        bridge_key=bridge_runner_process.bridge_key,
    )

    assert cmd.status == "completed"
    assert "pid" in cmd.result
    assert cmd.result["status"] == "running"

def test_bridge_file_operations(api_client, bridge_runner_process: RunnerInfo):
    """Write a file, find it with list/search, edit it, and read back."""
    rid = bridge_runner_process.runner_id
    bk = bridge_runner_process.bridge_key
    marker = f"xyzzy_{int(time.time())}"
    filename = f"bridge_e2e_{int(time.time())}.py"
    original_content = f"# {marker}\n"

    # 1. Write
    cmd = _submit_and_wait(
        api_client,
        rid,
        "WriteFile",
        {"path": filename, "content": original_content},
        bridge_key=bk,
    )
    assert cmd.status == "completed"
    assert cmd.result["created"] is True

    # 2. ListFiles — new file should appear in the root
    cmd = _submit_and_wait(
        api_client,
        rid,
        "ListFiles",
        {"pattern": f"{filename}"},
        bridge_key=bk,
    )
    assert cmd.status == "completed"
    assert any(filename in f for f in cmd.result["files"]), (
        f"{filename} not in {cmd.result['files']}"
    )

    # 3. EditFile — replace content
    cmd = _submit_and_wait(
        api_client,
        rid,
        "EditFile",
        {
            "path": filename,
            "edits": [{"old_string": marker, "new_string": f"edited_{marker}"}],
        },
        bridge_key=bk,
    )
    assert cmd.status == "completed"
    assert cmd.result["edits_applied"] == 1

    # 4. ReadFile — verify the edit took effect
    cmd = _submit_and_wait(
        api_client,
        rid,
        "ReadFile",
        {"path": filename},
        bridge_key=bk,
    )
    assert cmd.status == "completed"
    assert f"edited_{marker}" in cmd.result["content"]

    # 5. Cleanup via Exec
    _submit_and_wait(
        api_client,
        rid,
        "Exec",
        {"command": f"rm {filename}"},
        bridge_key=bk,
    )


# --- sdks/python/tests/e2e/runner/test_runner_e2e.py ---

def test_runner_happy_path(api_client, runner_process: RunnerInfo, project_id):
    """Basic: register echo agent, run job, verify job result, trace output, and job logs."""
    message = f"hello-e2e-{int(time.time())}"

    wait_for_agent_registration(api_client, "echo", project_id)

    submit_job(api_client, "echo", message, project_id)

    job = wait_for_completed_job(api_client, runner_process.runner_id, message)
    assert job.result is not None, "Completed job should have a result"
    assert f"echo: {message}" in str(job.result)
    assert job.trace_id is not None, "Completed job should have a trace_id"

    trace = find_trace_by_input(api_client, OPIK_E2E_TESTS_PROJECT_NAME, message)
    assert f"echo: {message}" in str(trace.output)

    logs_result = []

    def _find_logs():
        logs = api_client.runners.get_job_logs(job.id)
        if logs:
            logs_result.clear()
            logs_result.extend(logs)
            return True
        return False

    assert opik.synchronization.until(
        _find_logs,
        max_try_seconds=5,
        allow_errors=True,
    ), f"Expected job logs for job {job.id}, got none"

    log_text = " ".join(entry.text for entry in logs_result)
    assert message in log_text, f"Expected '{message}' in job logs, got: {log_text}"

def test_runner_with_mask(
    opik_client, api_client, runner_process: RunnerInfo, project_id
):
    """Mask: register echo_config agent, create mask, verify mask value in job result and trace."""
    message = f"mask-e2e-{int(time.time())}"
    custom_greeting = f"custom-greeting-{int(time.time())}"

    wait_for_agent_registration(api_client, "echo_config", project_id)

    manager = ConfigManager(
        project_name=OPIK_E2E_TESTS_PROJECT_NAME,
        rest_client_=opik_client.rest_client,
    )
    manager.create_blueprint(
        parameters={"greeting": "default-greeting"},
    )
    mask_id = manager.create_mask(
        parameters={"greeting": custom_greeting},
    )

    submit_job(api_client, "echo_config", message, project_id, mask_id=mask_id)

    job = wait_for_completed_job(api_client, runner_process.runner_id, message)
    assert job.result is not None, "Completed job should have a result"
    assert custom_greeting in str(job.result)
    assert job.trace_id is not None, "Completed job should have a trace_id"

    trace = find_trace_by_input(api_client, OPIK_E2E_TESTS_PROJECT_NAME, message)
    assert custom_greeting in str(trace.output), (
        f"Expected '{custom_greeting}' in trace output, got: {trace.output}"
    )


# --- sdks/python/tests/e2e_library_integration/adk/test_opik_tracer.py ---

def test_opik_tracer_with_sample_agent(
    opik_client_unique_project_name, start_api_server
) -> None:
    base_url = start_api_server

    # send the request to the ADK API server
    json_data = {
        "app_name": "sample_agent",
        "user_id": ADK_USER,
        "session_id": ADK_SESSION,
        "new_message": {
            "role": "user",
            "parts": [{"text": "Hey, whats the weather in New York today?"}],
        },
    }
    result = requests.post(
        f"{base_url}/run",
        json=json_data,
    )
    assert result.status_code == 200, (
        f"ADK /run returned {result.status_code}. Response: {result.text!r}"
    )

    traces = opik_client_unique_project_name.search_traces(
        filter_string='input contains "Hey, whats the weather in New York today?"',
        wait_for_at_least=1,
        wait_for_timeout=30,
    )
    assert len(traces) == 1

    trace = traces[0]
    assert trace.span_count == 3  # two LLM calls and one function call
    assert trace.usage is not None
    assert "adk_invocation_id" in trace.metadata.keys()
    assert trace.metadata["created_from"] == "google-adk"
    testlib.assert_dict_has_keys(trace.usage, EXPECTED_USAGE_KEYS_GOOGLE)

    spans = opik_client_unique_project_name.search_spans(wait_for_at_least=3)
    assert len(spans) == 3
    assert spans[0].provider == adk_helpers.get_adk_provider()
    assert spans[2].provider == adk_helpers.get_adk_provider()
    testlib.assert_dict_has_keys(spans[0].usage, EXPECTED_USAGE_KEYS_GOOGLE)
    testlib.assert_dict_has_keys(spans[2].usage, EXPECTED_USAGE_KEYS_GOOGLE)

def test_opik_tracer_with_sample_agent_sse(
    opik_client_unique_project_name, start_api_server
) -> None:
    """Run the test against the SSE endpoint with streaming enabled using the gemini-2.5-flash model."""
    base_url = start_api_server

    # send the request to the ADK API server
    json_data = {
        "app_name": "sample_agent_sse",
        "user_id": ADK_USER,
        "session_id": ADK_SESSION,
        "new_message": {
            "role": "user",
            "parts": [{"text": "Hey, whats the weather in New York today?"}],
        },
        "streaming": True,
    }

    result = requests.post(
        f"{base_url}/run_sse",
        json=json_data,
    )
    # print("Response: ", result.text)
    assert result.status_code == 200

    traces = opik_client_unique_project_name.search_traces(
        filter_string='input contains "Hey, whats the weather in New York today?"',
        wait_for_at_least=1,
        wait_for_timeout=30,
    )
    assert len(traces) == 1

    trace = traces[0]
    assert trace.span_count == 3  # two LLM calls and one function call
    assert trace.usage is not None
    assert "adk_invocation_id" in trace.metadata.keys()
    assert trace.metadata["created_from"] == "google-adk"
    testlib.assert_dict_keys_in_list(trace.usage, EXPECTED_USAGE_KEYS_GOOGLE_REASONING)

    spans = opik_client_unique_project_name.search_spans()
    assert len(spans) == 3
    assert spans[0].provider == adk_helpers.get_adk_provider()
    assert spans[2].provider == adk_helpers.get_adk_provider()
    testlib.assert_dict_keys_in_list(
        spans[0].usage, EXPECTED_USAGE_KEYS_GOOGLE_REASONING
    )
    testlib.assert_dict_keys_in_list(
        spans[2].usage, EXPECTED_USAGE_KEYS_GOOGLE_REASONING
    )

def test_opik_tracer_with_sample_agent__openai(
    opik_client_unique_project_name, start_api_server
) -> None:
    base_url = start_api_server

    # send the request to the ADK API server
    json_data = {
        "app_name": "sample_agent_openai",
        "user_id": ADK_USER,
        "session_id": ADK_SESSION,
        "new_message": {
            "role": "user",
            "parts": [{"text": "Hey, whats the weather in New York today?"}],
        },
    }
    result = requests.post(
        f"{base_url}/run",
        json=json_data,
    )
    print("Response: ", result.text)
    assert result.status_code == 200

    traces = opik_client_unique_project_name.search_traces(
        filter_string='input contains "Hey, whats the weather in New York today?"',
        wait_for_at_least=1,
        wait_for_timeout=30,
    )
    assert len(traces) == 1

    trace = traces[0]
    assert trace.span_count >= 3  # two LLM calls and one function call + duplicates
    assert trace.usage is not None
    assert "adk_invocation_id" in trace.metadata.keys()
    assert trace.metadata["created_from"] == "google-adk"
    OpenAICompletionsUsage.from_original_usage_dict(trace.usage)

    spans = opik_client_unique_project_name.search_spans()

    assert len(spans) >= 3  # sometimes it duplicates calls to the function
    for span in spans:
        if span.type == "llm":
            assert span.provider == "openai"
            assert span.model.startswith("gpt-4o")
            OpenAICompletionsUsage.from_original_usage_dict(span.usage)
        elif span.type == "tool":
            assert span.name == "get_weather"

def test_opik_tracer_with_sample_agent__anthropic(
    opik_client_unique_project_name, start_api_server
) -> None:
    base_url = start_api_server

    # send the request to the ADK API server
    json_data = {
        "app_name": "sample_agent_anthropic",
        "user_id": ADK_USER,
        "session_id": ADK_SESSION,
        "new_message": {
            "role": "user",
            "parts": [{"text": "Hey, whats the weather in New York today?"}],
        },
    }
    result = requests.post(
        f"{base_url}/run",
        json=json_data,
    )
    print("Response: ", result.text)
    assert result.status_code == 200

    traces = opik_client_unique_project_name.search_traces(
        filter_string='input contains "Hey, whats the weather in New York today?"',
        wait_for_at_least=1,
        wait_for_timeout=30,
    )
    assert len(traces) == 1

    trace = traces[0]
    assert trace.span_count == 3  # two LLM calls and one function call
    assert trace.usage is not None
    assert "adk_invocation_id" in trace.metadata.keys()
    assert trace.metadata["created_from"] == "google-adk"
    OpenAICompletionsUsage.from_original_usage_dict(trace.usage)

    spans = opik_client_unique_project_name.search_spans()

    assert len(spans) == 3
    assert spans[0].type == "llm"
    assert spans[0].provider == "anthropic"
    assert spans[0].model.startswith("claude-sonnet-4")
    OpenAICompletionsUsage.from_original_usage_dict(spans[0].usage)

    assert spans[2].type == "llm"
    assert spans[2].provider == "anthropic"
    assert spans[2].model.startswith("claude-sonnet-4")
    OpenAICompletionsUsage.from_original_usage_dict(spans[2].usage)


# --- sdks/python/tests/e2e_library_integration/harbor/test_harbor_e2e.py ---

    async def test_track_harbor_creates_traces_and_experiment(
        self,
        opik_client: Opik,
        configure_e2e_tests_env_unique_project_name: str,
        temp_jobs_dir: Path,
    ):
        """Test that track_harbor automatically creates traces, dataset, and experiment."""
        agent = AgentConfig(
            name=constants.AGENT_NAME,
            model_name=constants.MODEL_NAME,
            override_timeout_sec=constants.TIMEOUT_SEC,
        )

        dataset = RegistryDatasetConfig(
            registry=RemoteRegistryInfo(),
            name=constants.DATASET_NAME,
            version=constants.DATASET_VERSION,
            task_names=constants.TASK_NAMES,
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"opik-test-{timestamp}"

        job = Job(
            JobConfig(
                job_name=job_name,
                jobs_dir=temp_jobs_dir,
                orchestrator=OrchestratorConfig(n_concurrent_trials=1),
                environment=EnvironmentConfig(delete=True),
                agents=[agent],
                datasets=[dataset],
            )
        )

        tracked_job = track_harbor(job)

        await tracked_job.run()

        if not synchronization.until(
            function=lambda: opik_client.search_traces(
                project_name=configure_e2e_tests_env_unique_project_name
            )[0].output
            is not None,
            allow_errors=True,
            max_try_seconds=60,
        ):
            raise AssertionError("Failed to get traces")

        traces = opik_client.search_traces(
            project_name=configure_e2e_tests_env_unique_project_name, truncate=False
        )
        assert len(traces) == 2, f"Expected 2 traces (one per task), got {len(traces)}"

        for trace in traces:
            assert trace.metadata.get("created_from") == "harbor"
            assert "harbor" in (trace.tags or [])
            assert constants.AGENT_NAME in trace.name, (
                f"Trace name '{trace.name}' should contain agent name '{constants.AGENT_NAME}'"
            )

            spans = opik_client.search_spans(trace_id=trace.id, truncate=False)
            assert len(spans) >= 1

        opik_dataset = opik_client.get_dataset(constants.DATASET_NAME)
        assert opik_dataset is not None, "Dataset should be created automatically"

        assert_harbor_experiment_created(opik_client, temp_jobs_dir, job_name)

    def test_opik_harbor_cli_creates_traces_and_experiment(
        self,
        opik_client: Opik,
        configure_e2e_tests_env_unique_project_name: str,
        temp_jobs_dir: Path,
    ):
        """Test that `opik harbor run` automatically creates traces, dataset, and experiment."""

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"opik-cli-test-{timestamp}"

        # Create config file with proper timeout settings
        # (--ak override_timeout_sec=X doesn't work due to Harbor CLI bug)
        config = {
            "job_name": job_name,
            "jobs_dir": str(temp_jobs_dir),
            "n_attempts": 1,
            "timeout_multiplier": 1.0,
            "orchestrator": {
                "type": "local",
                "n_concurrent_trials": 1,
            },
            "environment": {
                "type": "docker",
                "delete": True,
            },
            "agents": [
                {
                    "name": constants.AGENT_NAME,
                    "model_name": constants.MODEL_NAME,
                    "override_timeout_sec": constants.TIMEOUT_SEC,
                }
            ],
            "datasets": [
                {
                    "registry": {"type": "remote"},
                    "name": constants.DATASET_NAME,
                    "version": constants.DATASET_VERSION,
                    "task_names": constants.TASK_NAMES,
                }
            ],
        }

        config_path = temp_jobs_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "harbor",
                "run",
                "-c",
                str(config_path),
            ],
            catch_exceptions=False,
        )

        # Print output for debugging visibility
        print(f"\n=== Harbor CLI Output ===\n{result.output}")

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        if not synchronization.until(
            function=lambda: opik_client.search_traces(
                project_name=configure_e2e_tests_env_unique_project_name
            )[0].output
            is not None,
            allow_errors=True,
            max_try_seconds=60,
        ):
            raise AssertionError("Failed to get traces after CLI run")

        traces = opik_client.search_traces(
            project_name=configure_e2e_tests_env_unique_project_name, truncate=False
        )
        assert len(traces) == 2, f"Expected 2 traces (one per task), got {len(traces)}"
        for trace in traces:
            assert "harbor" in (trace.tags or [])
            assert constants.AGENT_NAME in trace.name, (
                f"Trace name '{trace.name}' should contain agent name '{constants.AGENT_NAME}'"
            )

        opik_dataset = opik_client.get_dataset(constants.DATASET_NAME)
        assert opik_dataset is not None, "Dataset should be created automatically"

        assert_harbor_experiment_created(opik_client, temp_jobs_dir, job_name)


# --- sdks/python/tests/e2e_library_integration/litellm/test_opik_logging.py ---

def test_litellm_opik_logging__happyflow(
    ensure_openai_configured,
    opik_client: Opik,
    configure_e2e_tests_env_unique_project_name: str,
):
    litellm.callbacks = ["opik"]

    def streaming_function(input):
        messages = [{"role": "user", "content": input}]
        response = litellm.completion(
            model=constants.MODEL_NAME,
            messages=messages,
            metadata={
                "opik": {
                    "tags": ["streaming-test"],
                },
            },
        )
        return response

    _response = streaming_function("Why is tracking and evaluation of LLMs important?")

    if not synchronization.until(
        function=lambda: (len(opik_client.search_traces()) > 0),
        allow_errors=True,
        max_try_seconds=30,
    ):
        raise AssertionError(
            f"Failed to get traces from project '{configure_e2e_tests_env_unique_project_name}'"
        )

    traces = opik_client.search_traces(truncate=False)
    spans = opik_client.search_spans(truncate=False)

    assert len(traces) == 1
    assert len(spans) == 1

    verifiers.verify_trace(
        opik_client=opik_client,
        trace_id=traces[0].id,
        name="chat.completion",
        metadata=ANY_DICT.containing({"created_from": "litellm"}),
        input=[
            {
                "content": "Why is tracking and evaluation of LLMs important?",
                "role": "user",
            }
        ],
        output=ANY_DICT,
        tags=["openai", "streaming-test"],
        project_name=configure_e2e_tests_env_unique_project_name,
        error_info=None,
    )

    verifiers.verify_span(
        opik_client=opik_client,
        trace_id=traces[0].id,
        span_id=spans[0].id,
        parent_span_id=None,
        name=ANY_STRING.starting_with(constants.MODEL_NAME),
        metadata=ANY_DICT.containing({"created_from": "litellm"}),
        input=[
            {
                "content": "Why is tracking and evaluation of LLMs important?",
                "role": "user",
            }
        ],
        output=ANY_DICT,
        tags=["openai", "streaming-test"],
        project_name=configure_e2e_tests_env_unique_project_name,
        error_info=None,
        type="llm",
    )


# --- sdks/python/tests/integration/simulation/test_simulation_integration.py ---

    def test_simulation_with_class_based_app(self):
        """Test simulation with a class-based app that manages state."""

        class WeatherAgent:
            def __init__(self):
                self.histories = {}

            def __call__(self, message, *, thread_id, **kwargs):
                # Initialize history for this thread
                if thread_id not in self.histories:
                    self.histories[thread_id] = []

                # Add user message to history
                self.histories[thread_id].append(message)

                # Generate response based on full history
                response_content = f"Response to turn {len(self.histories[thread_id])}"
                assistant_message = {"role": "assistant", "content": response_content}

                # Add to history
                self.histories[thread_id].append(assistant_message)

                return assistant_message

        agent = WeatherAgent()
        user_simulator = SimulatedUser(
            persona="You are curious about weather",
            fixed_responses=["What's the weather?", "Tell me more", "Thanks!"],
        )

        result = run_simulation(app=agent, user_simulator=user_simulator, max_turns=3)

        # Verify conversation structure
        history = result["conversation_history"]
        assert len(history) == 6  # 3 turns * 2 messages

        # Verify agent maintained state
        assert len(agent.histories[result["thread_id"]]) == 6

        # Verify responses reference turn numbers
        assert "Response to turn 1" in history[1]["content"]
        assert "Response to turn 3" in history[3]["content"]
        assert "Response to turn 5" in history[5]["content"]

    def test_simulation_with_multiple_threads(self):
        """Test that different thread_ids maintain separate state."""

        class StatefulAgent:
            def __init__(self):
                self.histories = {}

            def __call__(self, message, *, thread_id, **kwargs):
                if thread_id not in self.histories:
                    self.histories[thread_id] = []

                self.histories[thread_id].append(message)
                response = {
                    "role": "assistant",
                    "content": f"Thread {thread_id} turn {len(self.histories[thread_id])}",
                }
                self.histories[thread_id].append(response)
                return response

        agent = StatefulAgent()
        user_simulator = SimulatedUser(persona="Test user", fixed_responses=["Message"])

        # Run first simulation
        result1 = run_simulation(
            app=agent, user_simulator=user_simulator, thread_id="thread-1", max_turns=2
        )

        # Run second simulation with different thread
        result2 = run_simulation(
            app=agent, user_simulator=user_simulator, thread_id="thread-2", max_turns=2
        )

        # Verify separate state
        assert result1["thread_id"] == "thread-1"
        assert result2["thread_id"] == "thread-2"
        assert len(agent.histories["thread-1"]) == 4
        assert len(agent.histories["thread-2"]) == 4

        # Verify responses reference correct threads
        assert "Thread thread-1 turn 1" in result1["conversation_history"][1]["content"]
        assert "Thread thread-2 turn 1" in result2["conversation_history"][1]["content"]

    def test_simulation_error_recovery(self):
        """Test simulation continues after app errors."""
        call_count = 0

        def error_prone_app(message, *, thread_id, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second call
                raise Exception("Temporary error")
            return {"role": "assistant", "content": f"Success {call_count}"}

        user_simulator = SimulatedUser(
            persona="Test user", fixed_responses=["Message 1", "Message 2", "Message 3"]
        )

        result = run_simulation(
            app=error_prone_app, user_simulator=user_simulator, max_turns=3
        )

        history = result["conversation_history"]

        # First turn should succeed
        assert "Success 1" in history[1]["content"]

        # Second turn should have error message
        assert "Error processing message: Temporary error" in history[3]["content"]

        # Third turn should succeed again
        assert "Success 3" in history[5]["content"]

    def test_simulation_with_complex_app_kwargs(self):
        """Test simulation with complex app configuration."""

        class ConfigurableAgent:
            def __init__(self):
                self.config = {}
                self.histories = {}

            def __call__(
                self, message, *, thread_id, model=None, temperature=None, **kwargs
            ):
                # Store configuration
                self.config[thread_id] = {
                    "model": model,
                    "temperature": temperature,
                    "other_kwargs": kwargs,
                }

                # Manage history
                if thread_id not in self.histories:
                    self.histories[thread_id] = []

                self.histories[thread_id].append(message)
                response = {"role": "assistant", "content": "Configured response"}
                self.histories[thread_id].append(response)
                return response

        agent = ConfigurableAgent()
        user_simulator = SimulatedUser(persona="Test user", fixed_responses=["Message"])

        result = run_simulation(
            app=agent,
            user_simulator=user_simulator,
            model="gpt-4",
            temperature=0.7,
            custom_param="test",
            max_turns=1,
        )

        # Verify configuration was stored
        config = agent.config[result["thread_id"]]
        assert config["model"] == "gpt-4"
        assert config["temperature"] == 0.7
        assert config["other_kwargs"]["custom_param"] == "test"


# --- sdks/python/tests/library_integration/dspy/test_dspy_llm_router.py ---

    def test_predict_returns_llm(self):
        """dspy.Predict should return 'llm' span type."""
        instance = dspy.Predict("question -> answer")
        assert get_span_type(instance) == "llm"

    def test_all_fields_set(self):
        """LMHistoryInfo should store all fields correctly."""
        from opik.llm_usage import OpikUsage

        usage = OpikUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            provider_usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        )
        info = LMHistoryInfo(
            usage=usage,
            cache_hit=False,
            actual_provider="Novita",
            actual_model="qwen/qwen-2.5-72b-instruct",
            total_cost=1.0e-05,
        )

        assert info.usage == usage
        assert info.cache_hit is False
        assert info.actual_provider == "Novita"
        assert info.actual_model == "qwen/qwen-2.5-72b-instruct"
        assert info.total_cost == 1.0e-05


# --- sdks/python/tests/library_integration/langchain/test_langchain_openai.py ---

def test_langchain__find_token_usage_dict__multi_turn_returns_latest():
    """
    Test that find_token_usage_dict returns the most recent usage_metadata.

    This is a regression test for a bug where the first token usage was always returned
    instead of the most recent one in multi-turn conversations.
    """
    from opik.integrations.langchain.provider_usage_extractors.langchain_run_helpers import (
        helpers,
    )

    multi_turn_run_dict = {
        "id": "run-123",
        "name": "ChatOpenAI",
        "inputs": {
            "messages": [{"role": "user", "content": "what is the weather in sf"}]
        },
        "outputs": {
            "generations": [
                [
                    {
                        "message": {
                            "content": "I'll check the weather for you.",
                            "kwargs": {
                                "usage_metadata": {
                                    "input_tokens": 150,
                                    "output_tokens": 25,
                                    "total_tokens": 175,
                                }
                            },
                        }
                    }
                ]
            ]
        },
        "events": [
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": {
                        "kwargs": {
                            "usage_metadata": {
                                "input_tokens": 150,
                                "output_tokens": 25,
                                "total_tokens": 175,
                            }
                        }
                    }
                },
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": {
                        "kwargs": {
                            "usage_metadata": {
                                "input_tokens": 190,
                                "output_tokens": 13,
                                "total_tokens": 203,
                            }
                        }
                    }
                },
            },
        ],
    }

    candidate_keys = {"input_tokens", "output_tokens", "total_tokens"}
    result = helpers.find_token_usage_dict(
        multi_turn_run_dict, candidate_keys, all_keys_should_match=False
    )

    assert result is not None
    assert result["input_tokens"] == 190
    assert result["output_tokens"] == 13
    assert result["total_tokens"] == 203


# --- sdks/python/tests/library_integration/langchain/test_message_converters.py ---

def test_convert_to_langchain_messages_handles_chat_prompt_template() -> None:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "user",
                [
                    {"type": "text", "text": "Describe the following image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://python.langchain.com/img/phone_handoff.jpeg",
                            "detail": "high",
                        },
                    },
                ],
            ),
        ]
    )

    rendered = prompt.invoke({})
    converted = convert_to_langchain_messages(messages_to_dict(rendered.messages))

    assert len(converted) == 2
    assert converted[1].type == "human"
    human_content = converted[1].content
    assert isinstance(human_content, list)
    assert human_content[1]["image_url"]["detail"] == "high"


# --- sdks/python/tests/library_integration/langchain/test_opik_langchain_chat_model.py ---

def test__langchain_chat_model__happyflow():
    tested = langchain_chat_model.LangchainChatModel(
        chat_model=langchain_openai.ChatOpenAI(
            model_name="gpt-4o",
        ),
        track=False,
    )

    assert isinstance(tested.generate_string("Say hi"), str)
    provider_response = tested.generate_provider_response(
        messages=[
            {
                "content": "Hello, world!",
                "role": "user",
            }
        ]
    )
    assert isinstance(provider_response, langchain_core.messages.AIMessage)
    assert isinstance(provider_response.content, str)

def test__langchain_chat_model__response_format_is_used():
    tested = langchain_chat_model.LangchainChatModel(
        chat_model=langchain_openai.ChatOpenAI(
            model_name="gpt-4o",
        ),
        track=False,
    )

    class Answer(pydantic.BaseModel):
        content: str
        value: str

    response_string = tested.generate_string("What's 2+2?", response_format=Answer)
    structured_response = json.loads(response_string)
    assert isinstance(structured_response, dict)
    assert isinstance(structured_response["content"], str)
    assert isinstance(structured_response["value"], str)


# --- sdks/python/tests/library_integration/langchain/test_opik_tracer.py ---

def test_is_langgraph_parent_command(error_traceback: str, expected: bool):
    """Test is_langgraph_parent_command with various input formats."""
    result = is_langgraph_parent_command(error_traceback)
    assert result == expected, (
        f"Expected {expected!r}, got {result!r} for input: {error_traceback[:120]}"
    )


# --- sdks/python/tests/library_integration/metrics_with_llm_judge/test_evaluation_metrics.py ---

def test__g_eval(model):
    g_eval_metric = metrics.GEval(
        model=model,
        track=False,
        task_introduction="You are an expert judge tasked with evaluating the faithfulness of an AI-generated answer to the given context.",
        evaluation_criteria="In provided text the OUTPUT must not introduce new information beyond what's provided in the CONTEXT.",
    )

    result = g_eval_metric.score(
        output="""
                OUTPUT: Paris is the capital of France.
                CONTEXT: France is a country in Western Europe, Its capital is Paris, which is known for landmarks like the Eiffel Tower.
               """
    )

    assert_helpers.assert_score_result(result)


# --- sdks/python/tests/library_integration/metrics_with_llm_judge/test_session_completeness_metric.py ---

def test__session_completeness_quality__with_real_model__happy_path(
    ensure_openai_configured, real_model_conversation
):
    """Integration test with a real model."""
    metric = session_completeness.SessionCompletenessQuality(
        track=False
    )  # Uses default model
    result = metric.score(real_model_conversation)

    assert_helpers.assert_score_result(result)

async def test__session_completeness_quality__with_real_model_async__happy_path(
    ensure_openai_configured, real_model_conversation
):
    os.environ["SSL_CERT_FILE"] = certifi.where()

    """Integration test with a real model asyncio mode."""
    metric = session_completeness.SessionCompletenessQuality(
        track=False
    )  # Uses default model
    result = await metric.ascore(real_model_conversation)

    assert_helpers.assert_score_result(result)


# --- sdks/python/tests/unit/test_cli_changes.py ---

    def test_valid_include_option_accepted(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["export", "default", "all", "--include", "datasets,prompts", "--help"],
        )
        # --help always exits 0 regardless of option values
        assert result.exit_code == 0

    def test_export_all_help_is_accessible(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["export", "default", "all", "--help"])
        assert result.exit_code == 0
        assert "all" in result.output.lower()
        assert "--include" in result.output

    def test_import_all_help_is_accessible(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["import", "default", "all", "--help"])
        assert result.exit_code == 0
        assert "all" in result.output.lower()
        assert "--include" in result.output

    def test_export_group_help_lists_all(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["export", "--help"])
        assert result.exit_code == 0
        assert "all" in result.output

    def test_import_group_help_lists_all(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["import", "--help"])
        assert result.exit_code == 0
        assert "all" in result.output

    def test_export_missing_subcommand_error_mentions_all(self):
        """When no subcommand is given the error message should list 'all'."""
        runner = CliRunner()
        result = runner.invoke(cli, ["export", "default"])
        # Non-zero exit or the error message includes "all"
        assert "all" in result.output


# --- sdks/python/tests/unit/test_export_import_all.py ---

    def test_fresh_manifest_status_is_not_started(self, tmp_path):
        m = ExportManifest(tmp_path)
        assert m.status == "not_started"
        assert not m.is_in_progress
        assert not m.is_completed

    def test_start_sets_in_progress(self, tmp_path):
        m = ExportManifest(tmp_path)
        m.start("json")
        assert m.status == "in_progress"
        assert m.is_in_progress
        assert not m.is_completed

    def test_complete_sets_completed_and_records_last_exported_at(self, tmp_path):
        m = ExportManifest(tmp_path)
        m.start("json")
        ts = "2024-06-01T00:00:00+00:00"
        m.complete(ts)
        assert m.status == "completed"
        assert m.is_completed
        assert not m.is_in_progress
        assert m.get_last_exported_at() == ts

    def test_reset_clears_all_state(self, tmp_path):
        m = ExportManifest(tmp_path)
        m.start("json")
        m.mark_trace_downloaded("t1")
        m.complete("2024-01-01T00:00:00+00:00")
        m.reset()
        assert m.status == "not_started"
        assert m.downloaded_count() == 0
        assert m.get_last_exported_at() is None
        assert m.get_format() is None

    def test_fresh_manifest_is_not_started(self, tmp_path):
        m = MigrationManifest(tmp_path)
        assert m.status == "not_started"
        assert not m.is_in_progress
        assert not m.is_completed

    def test_start_sets_in_progress(self, tmp_path):
        m = MigrationManifest(tmp_path)
        m.start()
        assert m.status == "in_progress"
        assert m.is_in_progress
        assert not m.is_completed

    def test_complete_sets_completed(self, tmp_path):
        m = MigrationManifest(tmp_path)
        m.start()
        m.complete()
        assert m.status == "completed"
        assert m.is_completed
        assert not m.is_in_progress

    def test_reset_clears_all_state(self, tmp_path):
        m = MigrationManifest(tmp_path)
        m.start()
        f = tmp_path / "f.json"
        f.touch()
        m.mark_file_completed(f)
        m.add_trace_mapping("src1", "dst1")
        m.complete()
        m.reset()
        assert m.status == "not_started"
        assert m.completed_count() == 0
        assert m.get_trace_id_map() == {}

    def test_mark_file_completed_and_check(self, tmp_path):
        m = MigrationManifest(tmp_path, batch_size=1)
        f = tmp_path / "data.json"
        f.touch()
        m.mark_file_completed(f)
        assert m.is_file_completed(f)

    def test_is_file_completed_returns_false_for_unknown_file(self, tmp_path):
        m = MigrationManifest(tmp_path)
        assert not m.is_file_completed(tmp_path / "nope.json")

    def test_completion_supersedes_prior_failure(self, tmp_path):
        """Completing a previously-failed file removes it from failed_files."""
        m = MigrationManifest(tmp_path, batch_size=1)
        f = tmp_path / "retry.json"
        f.touch()
        m.mark_file_failed(f, "first attempt failed")
        assert m.failed_count() == 1
        m.mark_file_completed(f)
        assert m.failed_count() == 0
        assert m.is_file_completed(f)

    def test_completed_count(self, tmp_path):
        m = MigrationManifest(tmp_path, batch_size=1)
        for i in range(3):
            fi = tmp_path / f"f{i}.json"
            fi.touch()
            m.mark_file_completed(fi)
        assert m.completed_count() == 3


# --- sdks/python/tests/unit/test_messages.py ---

def test_messages__all_fields_are_serializable():
    payload_dict = {
        "trace_id": "1234",
        "project_name": "TestProject",
        "name": "TestName",
        "start_time": datetime.now(),
        "end_time": None,
        "input": {"key": "value"},
        "output": None,
        "metadata": None,
        "tags": None,
        "error_info": None,
        "thread_id": None,
        "last_updated_at": datetime.now(),
        "source": "sdk",
    }

    message = messages.CreateTraceMessage(**payload_dict)

    result = message.as_payload_dict()
    payload_dict["id"] = payload_dict.pop("trace_id")

    assert result == payload_dict

def test_messages__not_all_fields_are_serializable():
    """
    Even if not all fields of the message are serializable, as_payload_dict() should still work
    """
    non_serializable_lock = RLock()

    payload_dict = {
        "trace_id": "1234",
        "project_name": "TestProject",
        "name": "TestName",
        "start_time": datetime.now(),
        "end_time": None,
        "input": {"key": "value"},
        "output": {
            "key": "value",
            "lock": non_serializable_lock,
        },
        "metadata": None,
        "tags": None,
        "error_info": None,
        "thread_id": None,
        "last_updated_at": datetime.now(),
        "source": "sdk",
    }

    message = messages.CreateTraceMessage(**payload_dict)

    result = message.as_payload_dict()
    payload_dict["id"] = payload_dict.pop("trace_id")

    assert result == payload_dict


# --- sdks/python/tests/unit/api_objects/test_opik_query_language.py ---

def test_trace_thread_oql__valid_filters(filter_string, expected):
    oql = OpikQueryLanguage.for_threads(filter_string)
    parsed = json.loads(oql.parsed_filters)
    assert len(parsed) == len(expected)

    for i, line in enumerate(expected):
        for key, value in line.items():
            assert parsed[i][key] == value


# --- sdks/python/tests/unit/api_objects/test_rest_stream_parser.py ---

def test_read_and_parse_stream__span(spans_stream_source):
    spans = rest_stream_parser.read_and_parse_stream(
        spans_stream_source, item_class=rest_api_types.SpanPublic
    )
    assert len(spans) == 2
    for i, span in enumerate(spans):
        expected = rest_api_types.SpanPublic.model_validate(SPANS_STREAM_JSON[i])
        assert span == expected

def test_read_and_parse_stream__limit_samples(spans_stream_source):
    spans = rest_stream_parser.read_and_parse_stream(
        spans_stream_source, item_class=rest_api_types.SpanPublic, nb_samples=1
    )
    assert len(spans) == 1
    expected = rest_api_types.SpanPublic.model_validate(SPANS_STREAM_JSON[0])
    assert spans[0] == expected

def test_read_and_parse_full_stream__happy_flow(spans_stream_source):
    spans = rest_stream_parser.read_and_parse_full_stream(
        read_source=lambda current_batch_size, last_retrieved_id: spans_stream_source,
        parsed_item_class=rest_api_types.SpanPublic,
        max_results=10,
    )
    assert len(spans) == 2
    for i, span in enumerate(spans):
        expected = rest_api_types.SpanPublic.model_validate(SPANS_STREAM_JSON[i])
        assert span == expected


# --- sdks/python/tests/unit/api_objects/attachment/test_converters.py ---

def test_attachment_to_message__no_temp_copy(original_file: str):
    url_override = "https://example.com"
    entity_id = "123"
    project_name = "test-project"
    attachment_data = attachment.Attachment(data=original_file, create_temp_copy=False)

    message = converters.attachment_to_message(
        attachment_data=attachment_data,
        entity_type="trace",
        entity_id=entity_id,
        project_name=project_name,
        url_override=url_override,
    )

    assert message == messages.CreateAttachmentMessage(
        file_path=original_file,
        file_name=os.path.basename(original_file),
        mime_type="text/plain",
        entity_type="trace",
        entity_id=entity_id,
        project_name=project_name,
        encoded_url_override="aHR0cHM6Ly9leGFtcGxlLmNvbQ==",
    )

def test_attachment_to_message__file_name(original_file: str):
    url_override = "https://example.com"
    entity_id = "123"
    project_name = "test-project"
    attachment_data = attachment.Attachment(
        data=original_file, file_name="test.jpg", create_temp_copy=False
    )

    message = converters.attachment_to_message(
        attachment_data=attachment_data,
        entity_type="trace",
        entity_id=entity_id,
        project_name=project_name,
        url_override=url_override,
    )

    assert message == messages.CreateAttachmentMessage(
        file_path=original_file,
        file_name="test.jpg",
        mime_type="image/jpeg",
        entity_type="trace",
        entity_id=entity_id,
        project_name=project_name,
        encoded_url_override="aHR0cHM6Ly9leGFtcGxlLmNvbQ==",
    )

def test_attachment_to_message__content_type(original_file: str):
    url_override = "https://example.com"
    entity_id = "123"
    project_name = "test-project"
    attachment_data = attachment.Attachment(
        data=original_file, content_type="image/jpeg", create_temp_copy=False
    )

    message = converters.attachment_to_message(
        attachment_data=attachment_data,
        entity_type="trace",
        entity_id=entity_id,
        project_name=project_name,
        url_override=url_override,
    )

    assert message == messages.CreateAttachmentMessage(
        file_path=original_file,
        file_name=os.path.basename(original_file),
        mime_type="image/jpeg",
        entity_type="trace",
        entity_id=entity_id,
        project_name=project_name,
        encoded_url_override="aHR0cHM6Ly9leGFtcGxlLmNvbQ==",
    )

def test_attachment_to_message__bytes_data():
    """Test that bytes data is written to a temp file and marked for deletion."""
    url_override = "https://example.com"
    entity_id = "123"
    project_name = "test-project"
    data = b"binary content here"

    attachment_data = attachment.Attachment(
        data=data,
        file_name="test.bin",
        content_type="application/octet-stream",
    )

    message = converters.attachment_to_message(
        attachment_data=attachment_data,
        entity_type="trace",
        entity_id=entity_id,
        project_name=project_name,
        url_override=url_override,
    )

    # A temp file should be created
    assert os.path.exists(message.file_path)
    assert message.file_path != "test.bin"

    # delete_after_upload should be True for bytes data
    assert message.delete_after_upload is True

    # Verify file content matches the bytes
    with open(message.file_path, "rb") as f:
        assert f.read() == data

    # Other fields should be preserved
    assert message.file_name == "test.bin"
    assert message.mime_type == "application/octet-stream"
    assert message.entity_type == "trace"
    assert message.entity_id == entity_id
    assert message.project_name == project_name
    assert message.encoded_url_override == "aHR0cHM6Ly9leGFtcGxlLmNvbQ=="

    # Clean up
    os.unlink(message.file_path)


# --- sdks/python/tests/unit/api_objects/dataset/test_suite/test_converters.py ---

    def test_item_with_execution_policy(self):
        item = dataset_item.DatasetItem(
            id="item-3",
            execution_policy=dataset_item.ExecutionPolicyItem(
                runs_per_item=3,
                pass_threshold=2,
            ),
            question="Test",
        )

        result = converters.dataset_item_to_suite_item_dict(item)

        assert result["execution_policy"]["runs_per_item"] == 3
        assert result["execution_policy"]["pass_threshold"] == 2


# --- sdks/python/tests/unit/api_objects/dataset/test_suite/test_report_file.py ---

    def test_to_report_dict__single_item_with_mixed_scores__returns_correct_structure(
        self,
    ):
        test_results_list = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[("Is polite", True), ("Is helpful", False)],
                task_output={"input": "hi", "output": "hello"},
                dataset_item_content={"question": "hi"},
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
                task_execution_time=1.234,
                scoring_time=0.567,
            )
        ]
        suite_result = _make_suite_result(test_results_list, suite_name="My Suite")

        result = suite_result.to_report_dict()

        assert result["suite_passed"] is False
        assert result["items_passed"] == 0
        assert result["items_total"] == 1
        assert result["pass_rate"] == 0.0
        assert result["experiment_id"] == "exp-123"
        assert result["experiment_name"] == "my-experiment"
        assert result["experiment_url"] == "http://example.com/experiment/exp-123"
        assert result["suite_name"] == "My Suite"
        assert "generated_at" in result

        assert len(result["items"]) == 1
        item = result["items"][0]
        assert item["dataset_item_id"] == "item-1"
        assert item["passed"] is False
        assert item["runs_passed"] == 0
        assert item["execution_policy"] == {"runs_per_item": 1, "pass_threshold": 1}

        assert len(item["runs"]) == 1
        run = item["runs"][0]
        assert run["trial_id"] == 0
        assert run["passed"] is False
        assert run["input"] == "hi"
        assert run["output"] == "hello"
        assert run["trace_id"] == "trace-item-1-0"
        assert run["task_execution_time_seconds"] == 1.234
        assert run["scoring_time_seconds"] == 0.567

        assert len(run["assertions"]) == 2
        assert run["assertions"][0]["name"] == "Is polite"
        assert run["assertions"][0]["passed"] is True
        assert run["assertions"][1]["name"] == "Is helpful"
        assert run["assertions"][1]["passed"] is False

    def test_to_report_dict__all_items_pass__suite_passed_true(self):
        test_results_list = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[("A1", True)],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            ),
            _make_test_result(
                dataset_item_id="item-2",
                trial_id=0,
                scores=[("A1", True)],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            ),
        ]
        suite_result = _make_suite_result(test_results_list)

        result = suite_result.to_report_dict()

        assert result["suite_passed"] is True
        assert result["items_passed"] == 2
        assert result["pass_rate"] == 1.0

    def test_to_report_dict__with_total_time__includes_rounded_value(self):
        test_results_list = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[("A1", True)],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            ),
        ]
        suite_result = _make_suite_result(test_results_list, total_time=12.3456)

        result = suite_result.to_report_dict()

        assert result["total_time_seconds"] == 12.346

    def test_to_report_dict__scoring_failed__marks_assertion_failed_with_reason(self):
        test_results_list = [
            test_result.TestResult(
                test_case=test_case.TestCase(
                    trace_id="trace-1",
                    dataset_item_id="item-1",
                    task_output={"output": "test"},
                    dataset_item_content={},
                    dataset_item=dataset_item.DatasetItem(
                        id="item-1",
                        execution_policy=dataset_item.ExecutionPolicyItem(
                            runs_per_item=1,
                            pass_threshold=1,
                        ),
                    ),
                ),
                score_results=[
                    score_result.ScoreResult(
                        name="A1",
                        value=0,
                        scoring_failed=True,
                        reason="Model error",
                    ),
                ],
                trial_id=0,
            )
        ]
        suite_result = _make_suite_result(test_results_list)

        result = suite_result.to_report_dict()

        assertion = result["items"][0]["runs"][0]["assertions"][0]
        assert assertion["passed"] is False
        assert assertion["scoring_failed"] is True
        assert assertion["reason"] == "Model error"

    def test_save_report__valid_input__writes_json_file(self, tmp_path):
        test_results_list = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[("A1", True)],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            ),
        ]
        suite_result = _make_suite_result(
            test_results_list, suite_name="Test Suite", total_time=5.0
        )
        output_path = str(tmp_path / "report.json")

        result_path = file_writer.save_report(suite_result, output_path)

        assert result_path == output_path
        assert os.path.exists(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data["suite_name"] == "Test Suite"
        assert data["suite_passed"] is True
        assert len(data["items"]) == 1

    def test_save_report__nested_path__creates_parent_directories(self, tmp_path):
        output_path = str(tmp_path / "nested" / "dir" / "report.json")

        test_results_list = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[("A1", True)],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            ),
        ]
        suite_result = _make_suite_result(test_results_list)

        result_path = file_writer.save_report(suite_result, output_path)

        assert os.path.exists(result_path)

    def test_to_dict__passing_suite__returns_dict_with_suite_passed_true(self):
        test_results_list = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[("A1", True)],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            ),
        ]
        suite_result = _make_suite_result(test_results_list)

        result = suite_result.to_dict()

        assert isinstance(result, dict)
        assert result["suite_passed"] is True
        assert "items" in result

    def test_to_report_dict__save_to_file__produces_valid_json(self, tmp_path):
        test_results_list = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[("A1", True)],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            ),
        ]
        suite_result = _make_suite_result(test_results_list)
        output_path = str(tmp_path / "result.json")

        path = file_writer.save_report(suite_result, output_path)

        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["suite_passed"] is True


# --- sdks/python/tests/unit/api_objects/dataset/test_suite/test_test_suite.py ---

    def test_item_with_all_optional_fields__passes(self):
        validators.validate_suite_items(
            [
                {
                    "data": {"question": "Hello"},
                    "assertions": ["Is polite"],
                    "description": "Test case",
                    "execution_policy": {"runs_per_item": 3, "pass_threshold": 2},
                }
            ]
        )

    def test_execution_policy_non_int_value__raises_type_error(self):
        with pytest.raises(TypeError, match="must be an int"):
            validators.validate_suite_items(
                [
                    {
                        "data": {"q": "Hello"},
                        "execution_policy": {
                            "runs_per_item": "3",
                            "pass_threshold": 1,
                        },
                    },
                ]
            )

    def test_valid_policy__passes(self):
        validators.validate_execution_policy({"runs_per_item": 3, "pass_threshold": 2})

    def test_partial_policy__raises_value_error(self):
        with pytest.raises(ValueError, match="missing required keys"):
            validators.validate_execution_policy({"runs_per_item": 5})

    def test_unknown_keys__raises_value_error(self):
        with pytest.raises(ValueError, match="unknown keys"):
            validators.validate_execution_policy({"runs_per_item": 3, "retry": True})

    def test_non_int_value__raises_type_error(self):
        with pytest.raises(TypeError, match="must be an int"):
            validators.validate_execution_policy(
                {"runs_per_item": 3.5, "pass_threshold": 1}
            )

    def test_string_value__raises_type_error(self):
        with pytest.raises(TypeError, match="must be an int"):
            validators.validate_execution_policy(
                {"runs_per_item": 1, "pass_threshold": "2"}
            )

    def test_build_suite_result__single_item_all_assertions_pass__item_passes(self):
        """Single item with 3 assertions, all pass -> item passes."""
        test_results = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[
                    ("Assertion 1", True),
                    ("Assertion 2", True),
                    ("Assertion 3", True),
                ],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            )
        ]
        eval_result = _make_evaluation_result(test_results)

        suite_result = suite_result_constructor.build_suite_result(eval_result)

        assert suite_result.all_items_passed is True
        assert suite_result.items_passed == 1
        assert suite_result.items_total == 1
        assert suite_result.pass_rate == 1.0
        assert suite_result.item_results["item-1"].passed is True
        assert suite_result.item_results["item-1"].runs_passed == 1

    def test_build_suite_result__single_item_one_assertion_fails__run_fails(self):
        """Single item with 3 assertions, one fails -> run fails -> item fails."""
        test_results = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[
                    ("Assertion 1", True),
                    ("Assertion 2", False),  # One fails
                    ("Assertion 3", True),
                ],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            )
        ]
        eval_result = _make_evaluation_result(test_results)

        suite_result = suite_result_constructor.build_suite_result(eval_result)

        assert suite_result.all_items_passed is False
        assert suite_result.items_passed == 0
        assert suite_result.items_total == 1
        assert suite_result.pass_rate == 0.0
        assert suite_result.item_results["item-1"].passed is False
        assert suite_result.item_results["item-1"].runs_passed == 0

    def test_build_suite_result__multiple_runs_pass_threshold_met__item_passes(self):
        """
        3 runs, pass_threshold=2.
        Run 1: all pass -> run passes
        Run 2: one fails -> run fails
        Run 3: all pass -> run passes
        Result: 2 runs passed >= pass_threshold(2) -> item passes
        """
        test_results = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[("A1", True), ("A2", True), ("A3", True)],
                execution_policy={"runs_per_item": 3, "pass_threshold": 2},
            ),
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=1,
                scores=[("A1", True), ("A2", False), ("A3", True)],
                execution_policy={"runs_per_item": 3, "pass_threshold": 2},
            ),
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=2,
                scores=[("A1", True), ("A2", True), ("A3", True)],
                execution_policy={"runs_per_item": 3, "pass_threshold": 2},
            ),
        ]
        eval_result = _make_evaluation_result(test_results)

        suite_result = suite_result_constructor.build_suite_result(eval_result)

        assert suite_result.all_items_passed is True
        assert suite_result.items_passed == 1
        item_result = suite_result.item_results["item-1"]
        assert item_result.passed is True
        assert item_result.runs_passed == 2
        assert item_result.runs_total == 3
        assert item_result.pass_threshold == 2

    def test_build_suite_result__multiple_runs_pass_threshold_not_met__item_fails(self):
        """
        3 runs, pass_threshold=2.
        Run 1: one fails -> run fails
        Run 2: one fails -> run fails
        Run 3: all pass -> run passes
        Result: 1 run passed < pass_threshold(2) -> item fails
        """
        test_results = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[("A1", False), ("A2", True), ("A3", True)],
                execution_policy={"runs_per_item": 3, "pass_threshold": 2},
            ),
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=1,
                scores=[("A1", True), ("A2", False), ("A3", True)],
                execution_policy={"runs_per_item": 3, "pass_threshold": 2},
            ),
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=2,
                scores=[("A1", True), ("A2", True), ("A3", True)],
                execution_policy={"runs_per_item": 3, "pass_threshold": 2},
            ),
        ]
        eval_result = _make_evaluation_result(test_results)

        suite_result = suite_result_constructor.build_suite_result(eval_result)

        assert suite_result.all_items_passed is False
        assert suite_result.items_passed == 0
        item_result = suite_result.item_results["item-1"]
        assert item_result.passed is False
        assert item_result.runs_passed == 1
        assert item_result.runs_total == 3
        assert item_result.pass_threshold == 2

    def test_build_suite_result__no_scores__run_passes_by_default(self):
        """Items with no evaluators (no scores) pass by default."""
        test_results = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            )
        ]
        eval_result = _make_evaluation_result(test_results)

        suite_result = suite_result_constructor.build_suite_result(eval_result)

        assert suite_result.all_items_passed is True
        assert suite_result.item_results["item-1"].passed is True
        assert suite_result.item_results["item-1"].runs_passed == 1

    def test_build_suite_result__multiple_items__calculates_pass_rate(self):
        """Suite with 3 items: 2 pass, 1 fails -> pass_rate = 2/3."""
        test_results = [
            _make_test_result(
                dataset_item_id="item-1",
                trial_id=0,
                scores=[("A1", True)],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            ),
            _make_test_result(
                dataset_item_id="item-2",
                trial_id=0,
                scores=[("A1", False)],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            ),
            _make_test_result(
                dataset_item_id="item-3",
                trial_id=0,
                scores=[("A1", True)],
                execution_policy={"runs_per_item": 1, "pass_threshold": 1},
            ),
        ]
        eval_result = _make_evaluation_result(test_results)

        suite_result = suite_result_constructor.build_suite_result(eval_result)

        assert suite_result.all_items_passed is False  # Not all items passed
        assert suite_result.items_passed == 2
        assert suite_result.items_total == 3
        assert suite_result.pass_rate == pytest.approx(2 / 3)

    def test_build_suite_result__integer_scores__treats_1_as_pass_0_as_fail(self):
        """Scores with integer values: 1 = pass, 0 = fail."""
        ds_item = dataset_item.DatasetItem(
            id="item-1",
            execution_policy=dataset_item.ExecutionPolicyItem(
                runs_per_item=1,
                pass_threshold=1,
            ),
        )
        test_results = [
            test_result.TestResult(
                test_case=test_case.TestCase(
                    trace_id="trace-1",
                    dataset_item_id="item-1",
                    task_output={"output": "test"},
                    dataset_item_content={},
                    dataset_item=ds_item,
                ),
                score_results=[
                    score_result.ScoreResult(name="A1", value=1),
                    score_result.ScoreResult(name="A2", value=1),
                ],
                trial_id=0,
            )
        ]
        eval_result = _make_evaluation_result(test_results)

        suite_result = suite_result_constructor.build_suite_result(eval_result)

        assert suite_result.item_results["item-1"].passed is True
        assert suite_result.item_results["item-1"].runs_passed == 1

    def test_build_suite_result__integer_score_zero__run_fails(self):
        """Scores with integer value 0 should fail the run."""
        ds_item = dataset_item.DatasetItem(
            id="item-1",
            execution_policy=dataset_item.ExecutionPolicyItem(
                runs_per_item=1,
                pass_threshold=1,
            ),
        )
        test_results = [
            test_result.TestResult(
                test_case=test_case.TestCase(
                    trace_id="trace-1",
                    dataset_item_id="item-1",
                    task_output={"output": "test"},
                    dataset_item_content={},
                    dataset_item=ds_item,
                ),
                score_results=[
                    score_result.ScoreResult(name="A1", value=1),
                    score_result.ScoreResult(name="A2", value=0),  # Fails
                ],
                trial_id=0,
            )
        ]
        eval_result = _make_evaluation_result(test_results)

        suite_result = suite_result_constructor.build_suite_result(eval_result)

        assert suite_result.item_results["item-1"].passed is False
        assert suite_result.item_results["item-1"].runs_passed == 0


# --- sdks/python/tests/unit/cli/test_attachments.py ---

    def test_export_traces__with_attachment__key_present_in_json(self, tmp_path):
        trace = _make_trace(TRACE_ID)
        span = _make_span(SPAN_ID, TRACE_ID)
        att_client = _make_attachment_client(
            trace_attachments=[
                {"file_name": "img.png", "mime_type": "image/png", "file_size": 100}
            ]
        )
        client = _make_opik_client(
            traces=[trace],
            spans_by_trace_id={TRACE_ID: [span]},
            attachment_client=att_client,
        )

        self._run_export(client, tmp_path, include_attachments=True)

        trace_file = tmp_path / f"trace_{TRACE_ID}.json"
        assert trace_file.exists()
        data = json.loads(trace_file.read_text())
        assert "attachments" in data
        assert len(data["attachments"]) == 1
        att = data["attachments"][0]
        assert att["entity_type"] == "trace"
        assert att["entity_id"] == TRACE_ID
        assert att["file_name"] == "img.png"
        assert att["mime_type"] == "image/png"

    def test_export_traces__no_attachments__list_is_empty(self, tmp_path):
        trace = _make_trace(TRACE_ID)
        att_client = _make_attachment_client()  # no attachments
        client = _make_opik_client(
            traces=[trace],
            spans_by_trace_id={TRACE_ID: []},
            attachment_client=att_client,
        )

        self._run_export(client, tmp_path, include_attachments=True)

        data = json.loads((tmp_path / f"trace_{TRACE_ID}.json").read_text())
        assert data["attachments"] == []

    def test_export_traces__no_attachments_flag__omits_metadata(self, tmp_path):
        trace = _make_trace(TRACE_ID)
        att_client = _make_attachment_client(
            trace_attachments=[{"file_name": "img.png"}]
        )
        client = _make_opik_client(
            traces=[trace],
            spans_by_trace_id={TRACE_ID: []},
            attachment_client=att_client,
        )

        self._run_export(client, tmp_path, include_attachments=False)

        data = json.loads((tmp_path / f"trace_{TRACE_ID}.json").read_text())
        # attachments key present but empty (include_attachments=False → att_client is None)
        assert data["attachments"] == []
        # And get_attachment_client was never called
        client.get_attachment_client.assert_not_called()

    def test_export_traces__attachment_download__writes_to_correct_path(self, tmp_path):
        trace = _make_trace(TRACE_ID)
        att_client = _make_attachment_client(
            trace_attachments=[
                {"file_name": "photo.jpg", "mime_type": "image/jpeg", "file_size": 50}
            ]
        )
        att_client.download_attachment.return_value = iter([b"jpeg bytes"])
        client = _make_opik_client(
            traces=[trace],
            spans_by_trace_id={TRACE_ID: []},
            attachment_client=att_client,
        )

        self._run_export(client, tmp_path, include_attachments=True)

        dest = tmp_path / "attachments" / "trace" / TRACE_ID / "photo.jpg"
        assert dest.exists()
        assert dest.read_bytes() == b"jpeg bytes"

    def test_export_traces__span_attachment__binary_downloaded(self, tmp_path):
        trace = _make_trace(TRACE_ID)
        span = _make_span(SPAN_ID, TRACE_ID)
        att_client = _make_attachment_client(
            span_attachments={
                SPAN_ID: [
                    {
                        "file_name": "span_output.txt",
                        "mime_type": "text/plain",
                        "file_size": 10,
                    }
                ]
            }
        )
        att_client.download_attachment.return_value = iter([b"span text"])
        client = _make_opik_client(
            traces=[trace],
            spans_by_trace_id={TRACE_ID: [span]},
            attachment_client=att_client,
        )

        self._run_export(client, tmp_path, include_attachments=True)

        dest = tmp_path / "attachments" / "span" / SPAN_ID / "span_output.txt"
        assert dest.exists()
        assert dest.read_bytes() == b"span text"

    def test_export_traces__existing_attachment_file_no_force__skips_download(
        self, tmp_path
    ):
        trace = _make_trace(TRACE_ID)
        att_client = _make_attachment_client(
            trace_attachments=[{"file_name": "img.png"}]
        )
        # Pre-create the attachment file
        dest = tmp_path / "attachments" / "trace" / TRACE_ID / "img.png"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"already here")

        client = _make_opik_client(
            traces=[trace],
            spans_by_trace_id={TRACE_ID: []},
            attachment_client=att_client,
        )

        self._run_export(client, tmp_path, include_attachments=True, force=False)

        # File unchanged; download_attachment was never called
        att_client.download_attachment.assert_not_called()
        assert dest.read_bytes() == b"already here"

    def test_import_projects__trace_attachment__uploaded_with_new_trace_id(
        self, tmp_path
    ):
        client = _make_import_client(new_trace_id="new-trace", new_span_id="new-span")
        project_dir = tmp_path / "test-project"
        _write_trace_file(project_dir, _TRACE_WITH_ATTACHMENTS)
        _write_attachment_file(
            project_dir, "trace", "orig-trace-id", "trace_img.png", b"img bytes"
        )
        _write_attachment_file(
            project_dir, "span", "orig-span-id", "span_data.csv", b"csv bytes"
        )

        import_projects_from_directory(
            client=client,
            source_dir=tmp_path,
            dry_run=False,
            name_pattern=None,
            debug=False,
            include_attachments=True,
        )

        upload_calls = client.queue_attachment_upload.call_args_list

        # Verify trace attachment uses new trace ID
        trace_call = next(
            c for c in upload_calls if c.kwargs.get("entity_type") == "trace"
        )
        assert trace_call.kwargs["entity_id"] == "new-trace"
        assert trace_call.kwargs["file_name"] == "trace_img.png"
        assert trace_call.kwargs["mime_type"] == "image/png"

    def test_import_projects__span_attachment__uploaded_with_new_span_id(
        self, tmp_path
    ):
        client = _make_import_client(new_trace_id="new-trace", new_span_id="new-span")
        project_dir = tmp_path / "test-project"
        _write_trace_file(project_dir, _TRACE_WITH_ATTACHMENTS)
        _write_attachment_file(project_dir, "trace", "orig-trace-id", "trace_img.png")
        _write_attachment_file(project_dir, "span", "orig-span-id", "span_data.csv")

        import_projects_from_directory(
            client=client,
            source_dir=tmp_path,
            dry_run=False,
            name_pattern=None,
            debug=False,
            include_attachments=True,
        )

        upload_calls = client.queue_attachment_upload.call_args_list

        span_call = next(
            c for c in upload_calls if c.kwargs.get("entity_type") == "span"
        )
        assert span_call.kwargs["entity_id"] == "new-span"
        assert span_call.kwargs["file_name"] == "span_data.csv"

    def test_import_projects__multiple_attachments__all_uploaded(self, tmp_path):
        client = _make_import_client()
        project_dir = tmp_path / "test-project"
        _write_trace_file(project_dir, _TRACE_WITH_ATTACHMENTS)
        _write_attachment_file(project_dir, "trace", "orig-trace-id", "trace_img.png")
        _write_attachment_file(project_dir, "span", "orig-span-id", "span_data.csv")

        import_projects_from_directory(
            client=client,
            source_dir=tmp_path,
            dry_run=False,
            name_pattern=None,
            debug=False,
            include_attachments=True,
        )

        assert client.queue_attachment_upload.call_count == 2

    def test_import_projects__no_attachments_flag__skips_all_uploads(self, tmp_path):
        client = _make_import_client()
        project_dir = tmp_path / "test-project"
        _write_trace_file(project_dir, _TRACE_WITH_ATTACHMENTS)
        _write_attachment_file(project_dir, "trace", "orig-trace-id", "trace_img.png")

        import_projects_from_directory(
            client=client,
            source_dir=tmp_path,
            dry_run=False,
            name_pattern=None,
            debug=False,
            include_attachments=False,
        )

        # queue_attachment_upload must not be called at all
        client.queue_attachment_upload.assert_not_called()

    def test_import_projects__attachment_files_missing__trace_still_imported(
        self, tmp_path
    ):
        """Missing attachment files emit a warning but don't abort the trace import."""
        client = _make_import_client()
        project_dir = tmp_path / "test-project"
        _write_trace_file(project_dir, _TRACE_WITH_ATTACHMENTS)
        # Do NOT write attachment files — simulate export with --no-attachments

        import_projects_from_directory(
            client=client,
            source_dir=tmp_path,
            dry_run=False,
            name_pattern=None,
            debug=False,
            include_attachments=True,
        )

        # Trace and span must still be created
        assert client.trace.call_count == 1
        assert client.span.call_count == 1
        # No uploads because files don't exist
        client.queue_attachment_upload.assert_not_called()

    def test_import_projects__dry_run__skips_uploads(self, tmp_path):
        client = _make_import_client()
        project_dir = tmp_path / "test-project"
        _write_trace_file(project_dir, _TRACE_WITH_ATTACHMENTS)
        _write_attachment_file(project_dir, "trace", "orig-trace-id", "trace_img.png")

        import_projects_from_directory(
            client=client,
            source_dir=tmp_path,
            dry_run=True,
            name_pattern=None,
            debug=False,
            include_attachments=True,
        )

        client.queue_attachment_upload.assert_not_called()

    def test_import_projects__no_attachments_key__imports_normally(self, tmp_path):
        """Trace files from before attachment support (no 'attachments' key) import fine."""
        trace_data = {
            "trace": {
                "id": "legacy-trace",
                "name": "legacy",
                "start_time": "2026-01-01T00:00:00Z",
                "end_time": "2026-01-01T00:00:01Z",
                "input": {},
                "output": {},
                "metadata": None,
                "tags": None,
                "feedback_scores": None,
                "error_info": None,
                "thread_id": None,
            },
            "spans": [],
            "downloaded_at": "2026-01-01T00:01:00",
            "project_name": "test-project",
            # NOTE: no "attachments" key
        }
        client = _make_import_client()
        project_dir = tmp_path / "test-project"
        _write_trace_file(project_dir, trace_data)

        import_projects_from_directory(
            client=client,
            source_dir=tmp_path,
            dry_run=False,
            name_pattern=None,
            debug=False,
            include_attachments=True,
        )

        assert client.trace.call_count == 1
        att_client = client.get_attachment_client.return_value
        att_client.upload_attachment.assert_not_called()


# --- sdks/python/tests/unit/cli/test_cli.py ---

    def test_export_group_help(self):
        """Test that the export group shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["export", "--help"])
        assert result.exit_code == 0
        assert "Export data from Opik workspace" in result.output
        assert "dataset" in result.output
        assert "project" in result.output
        assert "experiment" in result.output

    def test_export_dataset_help(self):
        """Test that the export dataset command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["export", "default", "dataset", "--help"])
        assert result.exit_code == 0
        assert "Export a dataset by exact name" in result.output
        assert "--force" in result.output

    def test_export_project_help(self):
        """Test that the export project command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["export", "default", "project", "--help"])
        assert result.exit_code == 0
        assert "Export a project by name or ID" in result.output
        assert "NAME" in result.output

    def test_export_experiment_help(self):
        """Test that the export experiment command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["export", "default", "experiment", "--help"])
        assert result.exit_code == 0
        assert "Export an experiment by exact name" in result.output
        assert "NAME" in result.output

    def test_import_group_help(self):
        """Test that the import group shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["import", "--help"])
        assert result.exit_code == 0
        assert "Import data to Opik workspace" in result.output
        assert "dataset" in result.output
        assert "project" in result.output
        assert "experiment" in result.output

    def test_import_dataset_help(self):
        """Test that the import dataset command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["import", "default", "dataset", "--help"])
        assert result.exit_code == 0
        assert "Import datasets from workspace/datasets directory" in result.output
        assert "--dry-run" in result.output

    def test_import_project_help(self):
        """Test that the import project command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["import", "default", "project", "--help"])
        assert result.exit_code == 0
        assert "Import projects from workspace/projects directory" in result.output
        assert "--dry-run" in result.output

    def test_import_experiment_help(self):
        """Test that the import experiment command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["import", "default", "experiment", "--help"])
        assert result.exit_code == 0
        assert (
            "Import experiments from workspace/experiments directory" in result.output
        )
        assert "--dry-run" in result.output

    def test_smoke_test_help(self):
        """Test that the healthcheck --smoke-test command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["healthcheck", "--help"])
        assert result.exit_code == 0
        assert "--smoke-test" in result.output
        assert "--project-name" in result.output
        assert "Project name for the smoke test" in result.output
        assert "WORKSPACE" in result.output
        assert "Run a smoke test to verify Opik integration" in result.output

    def test_smoke_test_minimal_args_parsing(self):
        """Test that healthcheck --smoke-test command requires workspace value."""
        runner = CliRunner()
        # Test that help shows workspace is required
        result = runner.invoke(cli, ["healthcheck", "--help"])
        assert result.exit_code == 0
        assert "WORKSPACE" in result.output
        # Test that missing workspace value causes error
        result = runner.invoke(cli, ["healthcheck", "--smoke-test"])
        assert result.exit_code != 0
        assert "Error" in result.output or "Missing" in result.output

    def test_check_permissions_option__in_healthcheck_help__shown(self):
        """Test that --check-permissions option is shown in healthcheck help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["healthcheck", "--help"])
        assert result.exit_code == 0
        assert "--check-permissions" in result.output
        assert "WORKSPACE" in result.output

    def test_check_permissions__missing_workspace__raises_error(self):
        """Test that --check-permissions without a workspace value raises an error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["healthcheck", "--check-permissions"])
        assert result.exit_code != 0
        assert "Error" in result.output or "Missing" in result.output


# --- sdks/python/tests/unit/cli/test_connect.py ---

    def test_connect__no_project__shows_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["connect"])
        assert result.exit_code == 2
        assert "--project" in result.output


# --- sdks/python/tests/unit/cli/test_endpoint.py ---

    def test_endpoint__no_project__shows_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["endpoint", "--", "echo", "hello"])
        assert result.exit_code == 2
        assert "--project" in result.output

    def test_endpoint__no_command__shows_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["endpoint", "--project", "my-proj"])
        assert result.exit_code == 2

    def test_endpoint__nonexistent_binary__shows_error(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["endpoint", "--project", "my-proj", "--", "nonexistent-binary-xyz-12345"],
        )
        assert result.exit_code == 2
        assert "not found" in result.output.lower()


# --- sdks/python/tests/unit/cli/test_export_project_rate_limiting.py ---

    def test_export_traces__page_fetch_429_with_retry_after_header__sleep_honours_header(
        self, tmp_path
    ):
        exc = ApiError(status_code=429, headers={"retry-after": "45"})
        sleep_calls = self._run_with_first_call_raising(exc, tmp_path)
        # Jitter adds 0–5 s on top of the header value.
        assert any(45.0 <= s <= 50.0 for s in sleep_calls)

    def test_export_traces__page_fetch_429_retry_after_exceeds_cap__sleep_clamped_to_max(
        self, tmp_path
    ):
        # 200 s > _EXPORT_MAX_RETRY_AFTER_SECONDS (120 s) → clamped to 120 s
        exc = ApiError(status_code=429, headers={"retry-after": "200"})
        sleep_calls = self._run_with_first_call_raising(exc, tmp_path)
        # Jitter adds 0–5 s on top of the 120 s cap.
        assert any(120.0 <= s <= 125.0 for s in sleep_calls)

    def test_export_traces__page_fetch_429_retry_after_http_date__sleep_honours_header(
        self, tmp_path
    ):
        import email.utils
        import time

        # Build an HTTP-date Retry-After 60 seconds in the future.
        future = email.utils.formatdate(timeval=time.time() + 60, usegmt=True)
        exc = ApiError(status_code=429, headers={"retry-after": future})
        sleep_calls = self._run_with_first_call_raising(exc, tmp_path)
        # Should sleep approximately 60 s (allow ±2 s for execution) plus 0–5 s jitter.
        assert any(58.0 <= s <= 67.0 for s in sleep_calls)

    def test_export_traces__page_fetch_429_no_retry_after_header__sleep_is_30s(
        self, tmp_path
    ):
        exc = ApiError(status_code=429, headers={})
        sleep_calls = self._run_with_first_call_raising(exc, tmp_path)
        # Jitter adds 0–5 s on top of the 30 s fallback.
        assert any(30.0 <= s <= 35.0 for s in sleep_calls)

    def test_export_traces__page_fetch_non_429_transient_error__sleep_is_exponential_backoff(
        self, tmp_path
    ):
        exc = ApiError(status_code=503)
        sleep_calls = self._run_with_first_call_raising(exc, tmp_path)
        # Backoff is exponential starting at 2 s, capped at 60 s.
        assert any(2.0 <= s <= 60.0 for s in sleep_calls)


# --- sdks/python/tests/unit/cli/test_migration_manifest.py ---

    def test_manifest__fresh_instance__status_is_not_started(
        self, tmp_base: Path
    ) -> None:
        manifest = MigrationManifest(tmp_base)
        assert manifest.status == "not_started"
        assert not manifest.is_in_progress
        assert not manifest.is_completed

    def test_complete__after_start__status_becomes_completed(
        self, tmp_base: Path
    ) -> None:
        manifest = MigrationManifest(tmp_base)
        manifest.start()
        manifest.complete()
        assert manifest.is_completed
        assert not manifest.is_in_progress

    def test_reset__after_start_with_data__all_state_cleared(
        self, tmp_base: Path
    ) -> None:
        manifest = MigrationManifest(tmp_base)
        manifest.start()
        trace_file = tmp_base / "projects" / "p1" / "trace_abc.json"
        trace_file.parent.mkdir(parents=True)
        trace_file.touch()
        manifest.mark_file_completed(trace_file)
        manifest.add_trace_mapping("src-1", "dest-1")

        manifest.reset()

        assert manifest.status == "not_started"
        assert manifest.completed_count() == 0
        assert manifest.get_trace_id_map() == {}

    def test_is_file_completed__fresh_manifest__returns_false(
        self, tmp_base: Path
    ) -> None:
        manifest = MigrationManifest(tmp_base)
        f = self._make_file(tmp_base, "datasets/dataset_foo.json")
        assert not manifest.is_file_completed(f)

    def test_mark_file_completed__new_file__recorded_and_count_incremented(
        self, tmp_base: Path
    ) -> None:
        manifest = MigrationManifest(tmp_base)
        f = self._make_file(tmp_base, "datasets/dataset_foo.json")
        manifest.mark_file_completed(f)
        assert manifest.is_file_completed(f)
        assert manifest.completed_count() == 1

    def test_mark_file_completed__same_file_twice__count_remains_one(
        self, tmp_base: Path
    ) -> None:
        manifest = MigrationManifest(tmp_base)
        f = self._make_file(tmp_base, "datasets/dataset_foo.json")
        manifest.mark_file_completed(f)
        manifest.mark_file_completed(f)
        assert manifest.completed_count() == 1

    def test_mark_file_completed__previously_failed_file__removed_from_failed(
        self, tmp_base: Path
    ) -> None:
        manifest = MigrationManifest(tmp_base)
        f = self._make_file(tmp_base, "projects/p1/trace_abc.json")
        manifest.mark_file_failed(f, "timeout")
        manifest.mark_file_completed(f)
        assert manifest.failed_count() == 0
        assert manifest.is_file_completed(f)

    def test_mark_file_completed__duplicate_write__count_remains_one(
        self, tmp_base: Path
    ) -> None:
        """INSERT OR IGNORE means re-flushing the same path never duplicates rows."""
        manifest = MigrationManifest(tmp_base, batch_size=1)
        f = tmp_base / "projects" / "p" / "trace_x.json"
        f.parent.mkdir(parents=True)
        f.touch()
        manifest.mark_file_completed(f)
        manifest.mark_file_completed(f)
        assert manifest.completed_count() == 1

    def test_mark_file_completed__within_batch__visible_via_api_before_flush(
        self, tmp_base: Path
    ) -> None:
        """Buffered completions are visible through the public API (which flushes
        before querying) even before the batch threshold is reached."""
        manifest = MigrationManifest(tmp_base, batch_size=50)
        f = self._make_trace(tmp_base, "trace_001.json")
        manifest.mark_file_completed(f)
        # completed_count() flushes first, so the buffered write IS visible.
        assert manifest.completed_count() == 1
        assert manifest.is_file_completed(f)

    def test_mark_file_completed__crash_before_flush__unflushed_data_lost(
        self, tmp_base: Path
    ) -> None:
        """Simulates a crash (new instance, no save()) with batch_size > pending count.

        This is the documented trade-off: up to batch_size-1 completions may be
        absent from the manifest after a crash. Those files will simply be
        re-imported on resume.
        """
        f = self._make_trace(tmp_base, "trace_001.json")

        m1 = MigrationManifest(tmp_base, batch_size=50)
        m1.start()  # always flushes immediately
        m1.mark_file_completed(f)  # buffered — NOT yet on disk
        # Simulate crash: m1 is abandoned without save() or complete().
        # __del__ is NOT called here because m1 is still in scope when m2 is created.

        m2 = MigrationManifest(tmp_base, batch_size=50)
        # The completion is not on disk — resume will re-process this file.
        assert not m2.is_file_completed(f)
        assert m2.completed_count() == 0

    def test_mark_file_completed__batch_size_one__each_write_immediately_durable(
        self, tmp_base: Path
    ) -> None:
        """With batch_size=1 every completion is immediately committed to disk."""
        f = self._make_trace(tmp_base, "trace_001.json")

        m1 = MigrationManifest(tmp_base, batch_size=1)
        m1.start()
        m1.mark_file_completed(f)  # auto-flushes (batch_size=1)

        # New instance reads from disk — must see the completion.
        m2 = MigrationManifest(tmp_base, batch_size=1)
        assert m2.is_file_completed(f)
        assert m2.completed_count() == 1

    def test_mark_file_completed__batch_threshold_reached__auto_flushed_to_disk(
        self, tmp_base: Path
    ) -> None:
        """When pending count hits batch_size the buffer is auto-flushed."""
        batch_size = 3
        files = [self._make_trace(tmp_base, f"trace_{i:03d}.json") for i in range(3)]

        m1 = MigrationManifest(tmp_base, batch_size=batch_size)
        m1.start()
        for f in files:
            m1.mark_file_completed(f)  # third call triggers auto-flush

        # New instance must see all three completions.
        m2 = MigrationManifest(tmp_base, batch_size=batch_size)
        assert m2.completed_count() == 3
        for f in files:
            assert m2.is_file_completed(f)

    def test_resume__interrupted_import__completed_files_skipped_and_id_map_available(
        self, tmp_base: Path
    ) -> None:
        """After an interrupted run the second run skips already-flushed files.

        batch_size=1 so every mark_file_completed is immediately durable —
        this models the boundary between two process invocations where only
        flushed data survives.
        """
        trace_files = []
        for i in range(3):
            p = tmp_base / "projects" / "proj" / f"trace_{i:03d}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            trace_files.append(p)

        m1 = MigrationManifest(tmp_base, batch_size=1)
        m1.start()
        m1.add_trace_mapping("src-0", "dest-0")
        m1.mark_file_completed(trace_files[0])
        m1.add_trace_mapping("src-1", "dest-1")
        m1.mark_file_completed(trace_files[1])
        # Process crashes before trace_files[2] — no complete() call.

        m2 = MigrationManifest(tmp_base)
        assert m2.is_in_progress
        assert m2.completed_count() == 2
        assert m2.is_file_completed(trace_files[0])
        assert m2.is_file_completed(trace_files[1])
        assert not m2.is_file_completed(trace_files[2])

        id_map = m2.get_trace_id_map()
        assert id_map["src-0"] == "dest-0"
        assert id_map["src-1"] == "dest-1"
        assert "src-2" not in id_map

    def test_reset__after_completed_files__all_state_cleared(
        self, tmp_base: Path
    ) -> None:
        trace_file = tmp_base / "projects" / "p" / "trace_abc.json"
        trace_file.parent.mkdir(parents=True)
        trace_file.touch()

        m1 = MigrationManifest(tmp_base)
        m1.start()
        m1.mark_file_completed(trace_file)

        m2 = MigrationManifest(tmp_base)
        m2.reset()

        assert m2.completed_count() == 0
        assert not m2.is_file_completed(trace_file)
        assert m2.status == "not_started"


# --- sdks/python/tests/unit/cli/test_pairing.py ---

    def test_build_pairing_link__valid_inputs__correct_payload_layout(self):
        session_id = "550e8400-e29b-41d4-a716-446655440000"
        project_id = "660e8400-e29b-41d4-a716-446655440000"
        activation_key = b"\xaa" * 32
        runner_name = "my-runner"

        link = build_pairing_link(
            base_url="https://www.comet.com/opik/api/",
            session_id=session_id,
            activation_key=activation_key,
            project_id=project_id,
            runner_name=runner_name,
        )

        assert link.startswith("https://www.comet.com/opik/pair/v1#")

        fragment = link.split("#", 1)[1]
        padding_needed = (4 - len(fragment) % 4) % 4
        payload = base64.urlsafe_b64decode(fragment + "=" * padding_needed)

        assert payload[0:16] == uuid.UUID(session_id).bytes
        assert payload[16:48] == activation_key
        assert payload[48:64] == uuid.UUID(project_id).bytes
        assert payload[64] == len(runner_name.encode("utf-8"))
        name_end = 65 + payload[64]
        assert payload[65:name_end] == runner_name.encode("utf-8")
        # Default runner type is CONNECT (0x00)
        assert payload[name_end] == 0x00

    def test_build_pairing_link__endpoint_type__encodes_0x01(self):
        link = build_pairing_link(
            base_url="http://localhost:5173/api/",
            session_id="550e8400-e29b-41d4-a716-446655440000",
            activation_key=b"\x00" * 32,
            project_id="660e8400-e29b-41d4-a716-446655440000",
            runner_name="r",
            runner_type=RunnerType.ENDPOINT,
        )
        fragment = link.split("#", 1)[1]
        padding_needed = (4 - len(fragment) % 4) % 4
        payload = base64.urlsafe_b64decode(fragment + "=" * padding_needed)
        # name_len=1, name="r", then type byte
        assert payload[66] == 0x01

    def test_build_pairing_link__cloud_url__no_double_opik_path(self):
        link = build_pairing_link(
            base_url="https://www.comet.com/opik/api/",
            session_id="550e8400-e29b-41d4-a716-446655440000",
            activation_key=b"\x00" * 32,
            project_id="660e8400-e29b-41d4-a716-446655440000",
            runner_name="r",
        )
        assert "/opik/opik/" not in link

    def test_build_pairing_link__localhost_url__correct_prefix(self):
        link = build_pairing_link(
            base_url="http://localhost:5173/api/",
            session_id="550e8400-e29b-41d4-a716-446655440000",
            activation_key=b"\x00" * 32,
            project_id="660e8400-e29b-41d4-a716-446655440000",
            runner_name="r",
        )
        assert link.startswith("http://localhost:5173/opik/pair/v1#")

    def test_validate_runner_name__valid_name__passes(self):
        validate_runner_name("my-runner")

    def test_validate_runner_name__empty__raises(self):
        with pytest.raises(Exception, match="empty"):
            validate_runner_name("")

    def test_validate_runner_name__whitespace_only__raises(self):
        with pytest.raises(Exception, match="empty"):
            validate_runner_name("   ")

    def test_validate_runner_name__over_128_chars__raises(self):
        with pytest.raises(Exception, match="128 characters"):
            validate_runner_name("x" * 129)

    def test_validate_runner_name__over_255_utf8_bytes__raises(self):
        name = "\U0001f600" * 64
        with pytest.raises(Exception, match="255 UTF-8 bytes"):
            validate_runner_name(name)

    def test_generate_runner_name__explicit_name__returns_it(self):
        assert generate_runner_name("my-name") == "my-name"

    def test_generate_runner_name__none__generates_random_hex(self):
        name = generate_runner_name(None)
        assert "-" in name
        hex_part = name.rsplit("-", 1)[1]
        assert len(hex_part) == 6
        int(hex_part, 16)

    def test_run_headless__creates_and_self_activates(self):
        api = self._make_api()
        result = run_headless(
            api=api,
            project_name="my-proj",
            runner_name="test-runner",
            runner_type=RunnerType.ENDPOINT,
        )

        assert result.runner_id == self.RUNNER_ID
        assert result.project_id == self.PROJECT_ID
        assert result.bridge_key == b""
        api.pairing.create_pairing_session.assert_called_once()
        api.pairing.activate_pairing_session.assert_called_once()

    def test_run_headless__no_polling(self):
        api = self._make_api()
        run_headless(
            api=api,
            project_name="my-proj",
            runner_name="test-runner",
            runner_type=RunnerType.ENDPOINT,
        )
        # No get_runner polling — headless activates immediately
        api.runners.get_runner.assert_not_called()

    def test_run_headless__connect_type__raises(self):
        api = self._make_api()
        with pytest.raises(click.ClickException, match="not supported"):
            run_headless(
                api=api,
                project_name="my-proj",
                runner_name="test-runner",
                runner_type=RunnerType.CONNECT,
            )


# --- sdks/python/tests/unit/error_tracking/test_logger_setup.py ---

def test_singleton_sentry_handlers():
    """Check that we are only adding sentry handlers once, they are using a singleton object in
    sentry so we don't need to add them multiple time
    """

    test_logger = logging.getLogger("test_singleton_sentry_handlers")
    base_handler = logging.StreamHandler()
    test_logger.addHandler(base_handler)

    assert len(test_logger.handlers) == 1

    # Add sentry handlers, there are two of them
    logger_setup.setup_sentry_error_handlers(test_logger)

    assert len(test_logger.handlers) == 3
    # Make sure the existing handler wasn't removed
    assert base_handler in test_logger.handlers

    # Sentry handlers are already present
    logger_setup.setup_sentry_error_handlers(test_logger)

    assert len(test_logger.handlers) == 3
    # Make sure the existing handler wasn't removed
    assert base_handler in test_logger.handlers


# --- sdks/python/tests/unit/evaluation/metrics/test_base_metric.py ---

def test_base_metric_ascore_returns_expected_result():
    metric = DummyMetric()
    actual_result = asyncio.run(metric.ascore())

    expected_result = score_result.ScoreResult(
        name="DummyMetric", value=0.5, reason="Test metric score"
    )
    assert actual_result == expected_result

    def test_opik_lightweight_import_does_not_load_heavy_modules(self):
        """Verify that importing from _opik stays lightweight.

        The _opik package must only use stdlib modules. If this test fails,
        someone added a dependency to _opik that pulls in heavy packages.

        HOW TO FIX: Remove the heavy import from _opik/. The _opik package
        must only depend on stdlib (abc, dataclasses, typing).
        """
        code = """
import sys

from _opik import BaseMetric, ScoreResult

# Verify basic functionality works
class SimpleMetric(BaseMetric):
    def score(self, **kwargs):
        return ScoreResult(name="simple", value=1.0)

metric = SimpleMetric()
result = metric.score()
assert result.name == "simple"
assert result.value == 1.0

# Only these opik-related modules should be loaded
ALLOWED = {"_opik", "_opik._base_metric", "_opik._score_result"}
loaded = {m for m in sys.modules if m.startswith(("opik", "_opik"))}
unexpected = sorted(loaded - ALLOWED)

if unexpected:
    print("FAIL")
    print(
        "Lightweight _opik import loaded unexpected modules.\\n"
        "The _opik package must only use stdlib.\\n"
        "Unexpected modules:\\n  " + "\\n  ".join(unexpected)
    )
    sys.exit(1)

print("LIGHTWEIGHT_OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Lightweight _opik loaded unexpected modules.\n"
            f"See stdout for details:\n{result.stdout}\n{result.stderr}"
        )
        assert "LIGHTWEIGHT_OK" in result.stdout


# --- sdks/python/tests/unit/evaluation/metrics/test_g_eval_presets.py ---

def test_bias_and_agent_wrapper_presets():
    assert (
        DemographicBiasJudge(track=False).task_introduction
        == GEVAL_PRESETS["bias_demographic"].task_introduction
    )
    assert (
        PoliticalBiasJudge(track=False).evaluation_criteria
        == GEVAL_PRESETS["bias_political"].evaluation_criteria
    )
    assert (
        GenderBiasJudge(track=False).evaluation_criteria
        == GEVAL_PRESETS["bias_gender"].evaluation_criteria
    )
    assert (
        ReligiousBiasJudge(track=False).evaluation_criteria
        == GEVAL_PRESETS["bias_religion"].evaluation_criteria
    )
    assert (
        RegionalBiasJudge(track=False).task_introduction
        == GEVAL_PRESETS["bias_regional"].task_introduction
    )
    assert (
        AgentToolCorrectnessJudge(track=False).evaluation_criteria
        == GEVAL_PRESETS["agent_tool_correctness"].evaluation_criteria
    )
    assert (
        AgentTaskCompletionJudge(track=False).task_introduction
        == GEVAL_PRESETS["agent_task_completion"].task_introduction
    )


# --- sdks/python/tests/unit/evaluation/metrics/llm_judges/structure_output_compliance/test_template.py ---

    def test_generate_query_basic(self):
        """Test basic query generation without schema or examples."""
        output = '{"name": "John", "age": 30}'

        query = template.generate_query(output=output)

        assert output in query
        assert "You are an expert in structured data validation" in query
        assert "EXPECTED STRUCTURE" in query
        assert "OUTPUT:" in query
        assert "Respond in the following JSON format:" in query
        assert "(No schema provided — assume valid JSON)" in query
        assert "EXAMPLES:" not in query

    def test_generate_query_with_schema(self):
        """Test query generation with schema."""
        output = '{"name": "John", "age": 30}'
        schema = "User(name: str, age: int)"

        query = template.generate_query(output=output, schema=schema)

        assert output in query
        assert schema in query
        assert "(No schema provided — assume valid JSON)" not in query

    def test_generate_query_with_few_shot_examples(self):
        """Test query generation with few-shot examples."""
        output = '{"name": "John", "age": 30}'
        few_shot_examples = [
            FewShotExampleStructuredOutputCompliance(
                title="Valid JSON",
                output='{"name": "Alice", "age": 25}',
                output_schema="User(name: str, age: int)",
                score=True,
                reason="Valid JSON format",
            ),
            FewShotExampleStructuredOutputCompliance(
                title="Invalid JSON",
                output='{"name": "Bob", age: 30}',
                output_schema="User(name: str, age: int)",
                score=False,
                reason="Missing quotes around age key",
            ),
        ]

        query = template.generate_query(
            output=output, few_shot_examples=few_shot_examples
        )

        assert output in query
        assert "EXAMPLES:" in query
        assert "Valid JSON" in query
        assert "Invalid JSON" in query
        assert "Alice" in query
        assert "Bob" in query
        assert "true" in query
        assert "false" in query
        assert "Valid JSON format" in query
        assert "Missing quotes around age key" in query
        assert "<example>" in query
        assert "</example>" in query

    def test_generate_query_with_schema_and_examples(self):
        """Test query generation with both schema and few-shot examples."""
        output = '{"name": "John", "age": 30}'
        schema = "User(name: str, age: int)"
        few_shot_examples = [
            FewShotExampleStructuredOutputCompliance(
                title="Valid Example",
                output='{"name": "Alice", "age": 25}',
                output_schema="User(name: str, age: int)",
                score=True,
                reason="Valid format",
            )
        ]

        query = template.generate_query(
            output=output, schema=schema, few_shot_examples=few_shot_examples
        )

        assert output in query
        assert schema in query
        assert "EXAMPLES:" in query
        assert "Valid Example" in query

    def test_generate_query_empty_examples_list(self):
        """Test query generation with empty examples list."""
        output = '{"name": "John", "age": 30}'
        few_shot_examples = []

        query = template.generate_query(
            output=output, few_shot_examples=few_shot_examples
        )

        assert output in query
        assert "EXAMPLES:" not in query

    def test_generate_query_example_without_schema(self):
        """Test query generation with examples that don't have schema."""
        output = '{"name": "John", "age": 30}'
        few_shot_examples = [
            FewShotExampleStructuredOutputCompliance(
                title="Valid JSON",
                output='{"name": "Alice"}',
                score=True,
                reason="Valid JSON format",
            )
        ]

        query = template.generate_query(
            output=output, few_shot_examples=few_shot_examples
        )

        assert "Expected Schema: None" in query
        assert "Valid JSON" in query
        assert "true" in query

    def test_generate_query_template_structure(self):
        """Test that the generated query has the correct template structure."""
        output = '{"test": "data"}'

        query = template.generate_query(output=output)

        assert "You are an expert in structured data validation" in query
        assert "Guidelines:" in query
        assert "1. OUTPUT must be a valid JSON object" in query
        assert "2. If a schema is provided" in query
        assert "3. If no schema is provided" in query
        assert "4. Common formatting issues" in query
        assert "5. Partial compliance is considered non-compliant" in query
        assert "6. Respond only in the specified JSON format" in query
        assert (
            "7. Score should be true if output fully complies, false otherwise" in query
        )
        assert "EXPECTED STRUCTURE (optional):" in query
        assert "OUTPUT:" in query
        assert '"score": <true or false>' in query
        assert '"reason": ["list of reasons' in query


# --- sdks/python/tests/unit/llm_usage/test_openai_chat_completions_usage.py ---

def test_openai_completions_usage_creation__happyflow():
    usage_data = {
        "completion_tokens": 100,
        "prompt_tokens": 200,
        "total_tokens": 300,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 50,
            "audio_tokens": 20,
        },
        "prompt_tokens_details": {
            "audio_tokens": 10,
            "cached_tokens": 30,
        },
    }
    usage = OpenAICompletionsUsage.from_original_usage_dict(usage_data)
    assert usage.completion_tokens == 100
    assert usage.prompt_tokens == 200
    assert usage.total_tokens == 300
    assert usage.completion_tokens_details.accepted_prediction_tokens == 50
    assert usage.completion_tokens_details.audio_tokens == 20
    assert usage.prompt_tokens_details.audio_tokens == 10
    assert usage.prompt_tokens_details.cached_tokens == 30

def test_openai_completions_usage_creation__no_details_keys__details_are_None():
    usage_data = {
        "completion_tokens": 100,
        "prompt_tokens": 200,
        "total_tokens": 300,
    }
    usage = OpenAICompletionsUsage.from_original_usage_dict(usage_data)
    assert usage.completion_tokens == 100
    assert usage.prompt_tokens == 200
    assert usage.total_tokens == 300
    assert usage.completion_tokens_details is None
    assert usage.prompt_tokens_details is None

def test_openai_completions_usage__to_backend_compatible_flat_dict__happyflow():
    usage_data = {
        "completion_tokens": 100,
        "prompt_tokens": 200,
        "total_tokens": 300,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 50,
            "audio_tokens": 20,
        },
        "prompt_tokens_details": {
            "audio_tokens": 10,
            "cached_tokens": 30,
        },
    }
    usage = OpenAICompletionsUsage.from_original_usage_dict(usage_data)
    flat_dict = usage.to_backend_compatible_flat_dict("original_usage")
    assert flat_dict == {
        "original_usage.completion_tokens": 100,
        "original_usage.prompt_tokens": 200,
        "original_usage.total_tokens": 300,
        "original_usage.completion_tokens_details.accepted_prediction_tokens": 50,
        "original_usage.completion_tokens_details.audio_tokens": 20,
        "original_usage.prompt_tokens_details.audio_tokens": 10,
        "original_usage.prompt_tokens_details.cached_tokens": 30,
    }

def test_openai_completions_usage__invalid_data_passed__validation_error_is_raised():
    usage_data = {
        "completion_tokens": "invalid",
        "prompt_tokens": None,
        "total_tokens": 300,
        "completion_tokens_details": "not_a_dict",
    }
    with pytest.raises(pydantic.ValidationError):
        OpenAICompletionsUsage.from_original_usage_dict(usage_data)

def test_openai_completions_usage__extra_unknown_keys_are_passed__fields_are_accepted__all_integers_included_to_the_resulting_flat_dict():
    usage_data = {
        "completion_tokens": 100,
        "prompt_tokens": 200,
        "total_tokens": 300,
        "extra_integer": 99,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 40,
            "extra_completion_detail_int": 888,
            "ignored_string": "ignored",
        },
        "prompt_tokens_details": {
            "audio_tokens": 10,
            "cached_tokens": 30,
            "extra_prompt_detail_int": 111,
            "ignored_string": "ignored",
        },
        "extra_details_dict": {
            "extra_detail_int": 0,
            "ignored_string": "ignored",
        },
    }
    usage = OpenAICompletionsUsage.from_original_usage_dict(usage_data)
    assert usage.extra_integer == 99
    assert usage.extra_details_dict == {
        "extra_detail_int": 0,
        "ignored_string": "ignored",
    }
    assert usage.completion_tokens_details.extra_completion_detail_int == 888
    assert usage.prompt_tokens_details.extra_prompt_detail_int == 111

    flat_dict = usage.to_backend_compatible_flat_dict("original_usage")
    assert flat_dict == {
        "original_usage.completion_tokens": 100,
        "original_usage.prompt_tokens": 200,
        "original_usage.total_tokens": 300,
        "original_usage.extra_integer": 99,
        "original_usage.completion_tokens_details.accepted_prediction_tokens": 40,
        "original_usage.completion_tokens_details.extra_completion_detail_int": 888,
        "original_usage.prompt_tokens_details.audio_tokens": 10,
        "original_usage.prompt_tokens_details.cached_tokens": 30,
        "original_usage.prompt_tokens_details.extra_prompt_detail_int": 111,
        "original_usage.extra_details_dict.extra_detail_int": 0,
    }


# --- sdks/python/tests/unit/llm_usage/test_opik_usage.py ---

def test_opik_usage__from_openai_completions_dict__happyflow():
    usage_data = {
        "completion_tokens": 100,
        "prompt_tokens": 200,
        "total_tokens": 300,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 50,
            "audio_tokens": 20,
        },
        "prompt_tokens_details": {
            "audio_tokens": 10,
            "cached_tokens": 30,
        },
        "video_seconds": 10,
    }
    usage = OpikUsage.from_openai_completions_dict(usage_data)
    assert usage.completion_tokens == 100
    assert usage.prompt_tokens == 200
    assert usage.total_tokens == 300
    assert usage.provider_usage.completion_tokens == 100
    assert usage.provider_usage.prompt_tokens == 200
    assert usage.provider_usage.total_tokens == 300
    assert usage.provider_usage.video_seconds == 10

def test_opik_usage__from_google_dict__happyflow():
    usage_data = {
        "candidates_token_count": 100,
        "prompt_token_count": 200,
        "total_token_count": 300,
        "cached_content_token_count": 50,
    }
    usage = OpikUsage.from_google_dict(usage_data)
    assert usage.completion_tokens == 100
    assert usage.prompt_tokens == 200
    assert usage.total_tokens == 300
    assert usage.provider_usage.candidates_token_count == 100
    assert usage.provider_usage.prompt_token_count == 200
    assert usage.provider_usage.total_token_count == 300

def test_opik_usage__to_backend_compatible_full_usage_dict__openai_source():
    usage_data = {
        "completion_tokens": 100,
        "prompt_tokens": 200,
        "total_tokens": 300,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 50,
            "audio_tokens": 20,
        },
        "prompt_tokens_details": {
            "audio_tokens": 10,
            "cached_tokens": 30,
        },
    }
    usage = OpikUsage.from_openai_completions_dict(usage_data)
    full_dict = usage.to_backend_compatible_full_usage_dict()
    assert full_dict == {
        "completion_tokens": 100,
        "prompt_tokens": 200,
        "total_tokens": 300,
        "original_usage.completion_tokens": 100,
        "original_usage.prompt_tokens": 200,
        "original_usage.total_tokens": 300,
        "original_usage.completion_tokens_details.accepted_prediction_tokens": 50,
        "original_usage.completion_tokens_details.audio_tokens": 20,
        "original_usage.prompt_tokens_details.audio_tokens": 10,
        "original_usage.prompt_tokens_details.cached_tokens": 30,
    }

def test_opik_usage__from_unknown_usage_dict__both_tokens_present__total_is_calculated():
    usage_data = {
        "prompt_tokens": 200,
        "completion_tokens": 100,
    }
    usage = OpikUsage.from_unknown_usage_dict(usage_data)
    assert usage.prompt_tokens == 200
    assert usage.completion_tokens == 100
    assert usage.total_tokens == 300

def test_opik_usage__from_unknown_usage_dict__only_prompt_tokens__total_is_none():
    usage_data = {
        "prompt_tokens": 200,
    }
    usage = OpikUsage.from_unknown_usage_dict(usage_data)
    assert usage.prompt_tokens == 200
    assert usage.completion_tokens is None
    assert usage.total_tokens is None

def test_opik_usage__from_unknown_usage_dict__only_completion_tokens__total_is_none():
    usage_data = {
        "completion_tokens": 100,
    }
    usage = OpikUsage.from_unknown_usage_dict(usage_data)
    assert usage.prompt_tokens is None
    assert usage.completion_tokens == 100
    assert usage.total_tokens is None

def test_opik_usage__from_unknown_usage_dict__empty_dict__all_none():
    usage = OpikUsage.from_unknown_usage_dict({})
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None

def test_opik_usage__from_unknown_usage_dict__string_tokens__coerced_to_int():
    usage_data = {
        "prompt_tokens": "200",
        "completion_tokens": "100",
    }
    usage = OpikUsage.from_unknown_usage_dict(usage_data)
    assert usage.prompt_tokens == 200
    assert usage.completion_tokens == 100
    assert usage.total_tokens == 300

def test_opik_usage__from_unknown_usage_dict__invalid_token_values__total_is_none():
    usage_data = {
        "prompt_tokens": "not-a-number",
        "completion_tokens": "also-invalid",
    }
    usage = OpikUsage.from_unknown_usage_dict(usage_data)
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None

def test_opik_usage__from_anthropic_dict__with_compaction_iterations__sums_all_iterations():
    # When compaction fires, top-level input/output_tokens reflect only the non-compaction
    # iterations (i.e. the message iterations). The compaction iteration is excluded from
    # the top-level but IS billed — summing all iterations gives the true billed cost.
    # https://platform.claude.com/docs/en/build-with-claude/compaction#understanding-usage
    usage_data = {
        # top-level = sum of non-compaction ("message") iterations only
        "input_tokens": 23000,
        "output_tokens": 1000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "iterations": [
            {
                "type": "compaction",
                "input_tokens": 180000,
                "output_tokens": 3500,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            {
                "type": "message",
                "input_tokens": 23000,
                "output_tokens": 1000,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        ],
    }
    usage = OpikUsage.from_anthropic_dict(usage_data)
    assert usage.prompt_tokens == 203000  # 180000 + 23000
    assert usage.completion_tokens == 4500  # 3500 + 1000
    assert usage.total_tokens == 207500

def test_opik_usage__from_anthropic_dict__compaction_with_caching__includes_cache_tokens_per_iteration():
    # When both compaction and prompt caching are active, each iteration always carries
    # cache_creation_input_tokens and cache_read_input_tokens (required fields per SDK types).
    # top-level tokens reflect only the non-compaction iterations.
    usage_data = {
        # top-level = message iteration only: input=23000, cache_read=5000
        "input_tokens": 23000,
        "output_tokens": 1000,
        "cache_creation_input_tokens": 500,
        "cache_read_input_tokens": 5000,
        "iterations": [
            {
                "type": "compaction",
                "input_tokens": 180000,
                "output_tokens": 3500,
                "cache_read_input_tokens": 10000,
                "cache_creation_input_tokens": 2000,
            },
            {
                "type": "message",
                "input_tokens": 23000,
                "output_tokens": 1000,
                "cache_read_input_tokens": 5000,
                "cache_creation_input_tokens": 500,
            },
        ],
    }
    usage = OpikUsage.from_anthropic_dict(usage_data)
    assert usage.prompt_tokens == 220500  # (180000+10000+2000) + (23000+5000+500)
    assert usage.completion_tokens == 4500  # 3500 + 1000
    assert usage.total_tokens == 225000

def test_opik_usage__from_anthropic_dict__no_compaction__uses_top_level_tokens():
    usage_data = {
        "input_tokens": 200,
        "output_tokens": 100,
        "cache_creation_input_tokens": 50,
        "cache_read_input_tokens": 30,
    }
    usage = OpikUsage.from_anthropic_dict(usage_data)
    assert usage.prompt_tokens == 280  # 200 + 30 cache_read + 50 cache_creation
    assert usage.completion_tokens == 100
    assert usage.total_tokens == 380

def test_opik_usage__invalid_data_passed__validation_error_is_raised():
    usage_data = {"a": 123}
    with pytest.raises(pydantic.ValidationError):
        OpikUsage.from_openai_completions_dict(usage_data)
    with pytest.raises(pydantic.ValidationError):
        OpikUsage.from_google_dict(usage_data)
    with pytest.raises(pydantic.ValidationError):
        OpikUsage.from_anthropic_dict(usage_data)


# --- sdks/python/tests/unit/llm_usage/test_opik_usage_factory.py ---

def test_opik_usage_factory__openai_happyflow():
    result = llm_usage.build_opik_usage(
        provider=opik.LLMProvider.OPENAI,
        usage={"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
    )

    assert result.completion_tokens == 10
    assert result.prompt_tokens == 20
    assert result.total_tokens == 30

    assert result.provider_usage.completion_tokens == 10
    assert result.provider_usage.prompt_tokens == 20
    assert result.provider_usage.total_tokens == 30

def test_opik_usage_factory__anthropic_happyflow():
    result = llm_usage.build_opik_usage(
        provider=opik.LLMProvider.ANTHROPIC,
        usage={"input_tokens": 10, "output_tokens": 20},
    )

    assert result.completion_tokens == 20
    assert result.prompt_tokens == 10
    assert result.total_tokens == 30

    assert result.provider_usage.input_tokens == 10
    assert result.provider_usage.output_tokens == 20

def test_opik_usage_factory__vertex_ai_none_candidates_token_count__happy_flow():
    result = llm_usage.build_opik_usage(
        provider=opik.LLMProvider.GOOGLE_VERTEXAI,
        usage={
            "cached_content_token_count": None,
            "candidates_token_count": None,
            "prompt_token_count": 7859,
            "thoughts_token_count": None,
            "total_token_count": 7859,
        },
    )

    assert result.completion_tokens == 0
    assert result.prompt_tokens == 7859
    assert result.total_tokens == 7859


# --- sdks/python/tests/unit/message_processing/test_uploads_streaming.py ---

def test_streamer__flush__attachment_uploads__ok(
    streamer_with_file_upload_manager, temp_file_15mb
):
    message_streamer, file_upload_manager = streamer_with_file_upload_manager

    attachment = messages.CreateAttachmentMessage(
        file_path=temp_file_15mb.name,
        file_name="test_file",
        mime_type=None,
        entity_type="span",
        entity_id=NOT_USED,
        project_name=NOT_USED,
        encoded_url_override=NOT_USED,
    )

    message_streamer.put(attachment)
    message_streamer.put(attachment)

    # we have timeout greater than upload time 1 > 0.5, thus all uploads will be completed
    assert message_streamer.flush(timeout=1, upload_sleep_time=1) is True

    assert file_upload_manager.remaining_data().uploads == 0

def test_streamer__flush__attachment_uploads__timeout(
    streamer_with_file_upload_manager, temp_file_15mb
):
    message_streamer, file_upload_manager = streamer_with_file_upload_manager

    attachment = messages.CreateAttachmentMessage(
        file_path=temp_file_15mb.name,
        file_name="test_file",
        mime_type=None,
        entity_type="span",
        entity_id=NOT_USED,
        project_name=NOT_USED,
        encoded_url_override=NOT_USED,
    )

    message_streamer.put(attachment)
    message_streamer.put(attachment)

    # we have timeout less than upload time 0.1 < 0.5, thus not all uploads will be completed
    assert message_streamer.flush(timeout=0.1, upload_sleep_time=0.1) is False

    assert file_upload_manager.remaining_data().uploads == 2


# --- sdks/python/tests/unit/message_processing/emulation/test_local_emulator_message_processor.py ---

    def test_trace_trees_returns_single_trace_without_spans(self):
        trace_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="test_trace",
            start_time=self.test_datetime,
            end_time=self.later_datetime,
            input={"key": "value"},
            output={"result": "success"},
            metadata={"meta": "data"},
            tags=["tag1"],
            error_info=None,
            thread_id="thread_1",
            last_updated_at=self.test_datetime,
            source="sdk",
        )

        self.processor.process(trace_message)
        trace_trees = self.processor.trace_trees

        assert len(trace_trees) == 1
        assert trace_trees[0].id == "trace_1"
        assert trace_trees[0].name == "test_trace"
        assert trace_trees[0].spans == []
        assert trace_trees[0].feedback_scores == []

    def test_trace_trees_returns_trace_with_top_level_spans(self):
        trace_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="test_trace",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=self.later_datetime,
            source="sdk",
        )

        span1_message = messages.CreateSpanMessage(
            span_id="span_1",
            trace_id="trace_1",
            project_name="test_project",
            parent_span_id=None,
            name="first_span",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            type="general",
            usage=None,
            model=None,
            provider=None,
            error_info=None,
            total_cost=None,
            last_updated_at=None,
            source="sdk",
        )

        span2_message = messages.CreateSpanMessage(
            span_id="span_2",
            trace_id="trace_1",
            project_name="test_project",
            parent_span_id=None,
            name="second_span",
            start_time=self.later_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            type="general",
            usage=None,
            model=None,
            provider=None,
            error_info=None,
            total_cost=None,
            last_updated_at=None,
            source="sdk",
        )

        self.processor.process(trace_message)
        self.processor.process(span1_message)
        self.processor.process(span2_message)

        trace_trees = self.processor.trace_trees

        assert len(trace_trees) == 1

        EXCPECTED_TRACE_TREE = models.TraceModel(
            id="trace_1",
            start_time=self.test_datetime,
            name="test_trace",
            project_name="test_project",
            input=None,
            output=None,
            tags=None,
            metadata=None,
            end_time=None,
            spans=[
                models.SpanModel(
                    id="span_1",
                    start_time=self.test_datetime,
                    name="first_span",
                    input=None,
                    output=None,
                    tags=None,
                    metadata=None,
                    type="general",
                    usage=None,
                    end_time=None,
                    project_name="test_project",
                    spans=[],
                    feedback_scores=[],
                    model=None,
                    provider=None,
                    error_info=None,
                    total_cost=None,
                    last_updated_at=None,
                    source="sdk",
                ),
                models.SpanModel(
                    id="span_2",
                    start_time=self.later_datetime,
                    name="second_span",
                    input=None,
                    output=None,
                    tags=None,
                    metadata=None,
                    type="general",
                    usage=None,
                    end_time=None,
                    project_name="test_project",
                    spans=[],
                    feedback_scores=[],
                    model=None,
                    provider=None,
                    error_info=None,
                    total_cost=None,
                    last_updated_at=None,
                    source="sdk",
                ),
            ],
            feedback_scores=[],
            error_info=None,
            thread_id=None,
            last_updated_at=self.later_datetime,
            source="sdk",
        )

        assert_helpers.assert_equal(
            expected=EXCPECTED_TRACE_TREE, actual=trace_trees[0]
        )

    def test_trace_trees__orphan_span_trace_link__is_skipped(self, capture_log):
        trace_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="test_trace",
            start_time=self.test_datetime,
            end_time=self.later_datetime,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=self.later_datetime,
            source="sdk",
        )
        orphan_span_message = messages.CreateSpanMessage(
            span_id="span_orphan",
            trace_id="missing_trace_id",
            project_name="test_project",
            parent_span_id=None,
            name="orphan_span",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            type="general",
            usage=None,
            model=None,
            provider=None,
            error_info=None,
            total_cost=None,
            last_updated_at=None,
            source="sdk",
        )

        self.processor.process(trace_message)
        self.processor.process(orphan_span_message)

        trace_trees = self.processor.trace_trees

        assert len(trace_trees) == 1
        assert trace_trees[0].id == "trace_1"
        assert "orphan span-to-trace link" in capture_log.text

    def test_trace_trees_with_nested_span_hierarchy(self):
        trace_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="test_trace",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=self.later_datetime,
            source="sdk",
        )

        parent_span_message = messages.CreateSpanMessage(
            span_id="parent_span",
            trace_id="trace_1",
            project_name="test_project",
            parent_span_id=None,
            name="parent_span",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            type="general",
            usage=None,
            model=None,
            provider=None,
            error_info=None,
            total_cost=None,
            last_updated_at=None,
            source="sdk",
        )

        child_span_message = messages.CreateSpanMessage(
            span_id="child_span",
            trace_id="trace_1",
            project_name="test_project",
            parent_span_id="parent_span",
            name="child_span",
            start_time=self.later_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            type="general",
            usage=None,
            model=None,
            provider=None,
            error_info=None,
            total_cost=None,
            last_updated_at=None,
            source="sdk",
        )

        self.processor.process(trace_message)
        self.processor.process(parent_span_message)
        self.processor.process(child_span_message)

        trace_trees = self.processor.trace_trees
        assert len(trace_trees) == 1

        EXCPECTED_TRACE_TREE = models.TraceModel(
            id="trace_1",
            start_time=self.test_datetime,
            name="test_trace",
            project_name="test_project",
            input=None,
            output=None,
            tags=None,
            metadata=None,
            end_time=None,
            spans=[
                models.SpanModel(
                    id="parent_span",
                    start_time=self.test_datetime,
                    name="parent_span",
                    input=None,
                    output=None,
                    tags=None,
                    metadata=None,
                    type="general",
                    usage=None,
                    end_time=None,
                    project_name="test_project",
                    spans=[
                        models.SpanModel(
                            id="child_span",
                            start_time=self.later_datetime,
                            name="child_span",
                            input=None,
                            output=None,
                            tags=None,
                            metadata=None,
                            type="general",
                            usage=None,
                            end_time=None,
                            project_name="test_project",
                            spans=[],
                            feedback_scores=[],
                            model=None,
                            provider=None,
                            error_info=None,
                            total_cost=None,
                            last_updated_at=None,
                            source="sdk",
                        )
                    ],
                    feedback_scores=[],
                    model=None,
                    provider=None,
                    error_info=None,
                    total_cost=None,
                    last_updated_at=None,
                    source="sdk",
                )
            ],
            feedback_scores=[],
            error_info=None,
            thread_id=None,
            last_updated_at=self.later_datetime,
            source="sdk",
        )

        assert_helpers.assert_equal(
            expected=EXCPECTED_TRACE_TREE, actual=trace_trees[0]
        )

    def test_trace_trees_multiple_traces_sorted_by_start_time(self):
        trace1_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="trace_1",
            start_time=self.later_datetime,  # Later start time
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=None,
            source="sdk",
        )

        trace2_message = messages.CreateTraceMessage(
            trace_id="trace_2",
            project_name="test_project",
            name="trace_2",
            start_time=self.test_datetime,  # Earlier start time
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=None,
            source="sdk",
        )

        self.processor.process(trace1_message)
        self.processor.process(trace2_message)

        trace_trees = self.processor.trace_trees

        assert len(trace_trees) == 2
        assert trace_trees[0].id == "trace_2"  # Earlier start time first
        assert trace_trees[1].id == "trace_1"  # Later start time second

    def test_trace_trees_with_trace_feedback_scores_batch(self):
        trace_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="test_trace",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=None,
            source="sdk",
        )

        # Create batch feedback score messages for trace
        feedback_score_1 = messages.FeedbackScoreMessage(
            id="trace_1",  # This should match the trace_id
            project_name="test_project",
            name="accuracy",
            value=0.95,
            source="user",
            category_name="evaluation",
            reason="high confidence",
        )

        feedback_score_2 = messages.FeedbackScoreMessage(
            id="trace_1",  # This should match the trace_id
            project_name="test_project",
            name="relevance",
            value=0.88,
            source="automated",
            category_name="quality",
            reason="good content match",
        )

        trace_feedback_batch_message = messages.AddTraceFeedbackScoresBatchMessage(
            batch=[feedback_score_1, feedback_score_2]
        )

        self.processor.process(trace_message)
        self.processor.process(trace_feedback_batch_message)

        trace_trees = self.processor.trace_trees

        assert len(trace_trees) == 1
        trace = trace_trees[0]
        assert trace.id == "trace_1"
        assert len(trace.feedback_scores) == 2

        # Check first feedback score
        feedback_1 = trace.feedback_scores[0]
        assert feedback_1.name == "accuracy"
        assert feedback_1.value == 0.95
        assert feedback_1.category_name == "evaluation"
        assert feedback_1.reason == "high confidence"

        # Check the second feedback score
        feedback_2 = trace.feedback_scores[1]
        assert feedback_2.name == "relevance"
        assert feedback_2.value == 0.88
        assert feedback_2.category_name == "quality"
        assert feedback_2.reason == "good content match"

    def test_trace_trees_with_span_feedback_scores_batch(self):
        trace_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="test_trace",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=None,
            source="sdk",
        )

        span_message = messages.CreateSpanMessage(
            span_id="span_1",
            trace_id="trace_1",
            project_name="test_project",
            parent_span_id=None,
            name="test_span",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            type="general",
            usage=None,
            model=None,
            provider=None,
            error_info=None,
            total_cost=None,
            last_updated_at=None,
            source="sdk",
        )

        # Create batch feedback score messages for span
        span_feedback_score_1 = messages.FeedbackScoreMessage(
            id="span_1",  # This should match the span_id
            project_name="test_project",
            name="quality",
            value=0.92,
            source="user",
            category_name="output_quality",
            reason="well structured response",
        )

        span_feedback_score_2 = messages.FeedbackScoreMessage(
            id="span_1",  # This should match the span_id
            project_name="test_project",
            name="latency_score",
            value=0.75,
            source="automated",
            category_name="performance",
            reason="acceptable response time",
        )

        span_feedback_batch_message = messages.AddSpanFeedbackScoresBatchMessage(
            batch=[span_feedback_score_1, span_feedback_score_2]
        )

        self.processor.process(trace_message)
        self.processor.process(span_message)
        self.processor.process(span_feedback_batch_message)

        trace_trees = self.processor.trace_trees

        assert len(trace_trees) == 1
        trace = trace_trees[0]
        assert trace.id == "trace_1"
        assert len(trace.spans) == 1

        span = trace.spans[0]
        assert span.id == "span_1"
        assert len(span.feedback_scores) == 2

        # Check the first span feedback score
        span_feedback_1 = span.feedback_scores[0]
        assert span_feedback_1.name == "quality"
        assert span_feedback_1.value == 0.92
        assert span_feedback_1.category_name == "output_quality"
        assert span_feedback_1.reason == "well structured response"

        # Check the second span feedback score
        span_feedback_2 = span.feedback_scores[1]
        assert span_feedback_2.name == "latency_score"
        assert span_feedback_2.value == 0.75
        assert span_feedback_2.category_name == "performance"
        assert span_feedback_2.reason == "acceptable response time"

    def test_trace_trees_with_mixed_feedback_scores_batch(self):
        # Test both trace and span feedback scores in the same trace tree
        trace_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="test_trace",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=None,
            source="sdk",
        )

        span_message = messages.CreateSpanMessage(
            span_id="span_1",
            trace_id="trace_1",
            project_name="test_project",
            parent_span_id=None,
            name="test_span",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            type="general",
            usage=None,
            model=None,
            provider=None,
            error_info=None,
            total_cost=None,
            last_updated_at=None,
            source="sdk",
        )

        # Trace feedback scores
        trace_feedback_score = messages.FeedbackScoreMessage(
            id="trace_1",
            project_name="test_project",
            name="overall_quality",
            value=0.90,
            source="user",
            category_name="overall",
            reason="good trace execution",
        )

        trace_feedback_batch_message = messages.AddTraceFeedbackScoresBatchMessage(
            batch=[trace_feedback_score]
        )

        # Span feedback scores
        span_feedback_score = messages.FeedbackScoreMessage(
            id="span_1",
            project_name="test_project",
            name="step_quality",
            value=0.85,
            source="user",
            category_name="step",
            reason="good individual step",
        )

        span_feedback_batch_message = messages.AddSpanFeedbackScoresBatchMessage(
            batch=[span_feedback_score]
        )

        self.processor.process(trace_message)
        self.processor.process(span_message)
        self.processor.process(trace_feedback_batch_message)
        self.processor.process(span_feedback_batch_message)

        trace_trees = self.processor.trace_trees

        assert len(trace_trees) == 1
        trace = trace_trees[0]
        assert trace.id == "trace_1"
        assert len(trace.feedback_scores) == 1
        assert trace.feedback_scores[0].name == "overall_quality"
        assert trace.feedback_scores[0].value == 0.90

        assert len(trace.spans) == 1
        span = trace.spans[0]
        assert span.id == "span_1"
        assert len(span.feedback_scores) == 1
        assert span.feedback_scores[0].name == "step_quality"
        assert span.feedback_scores[0].value == 0.85

    def test_trace_trees_with_empty_feedback_scores_batch(self):
        # Test empty batch messages don't cause issues
        trace_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="test_trace",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=None,
            source="sdk",
        )

        empty_trace_feedback_batch = messages.AddTraceFeedbackScoresBatchMessage(
            batch=[]
        )

        empty_span_feedback_batch = messages.AddSpanFeedbackScoresBatchMessage(batch=[])

        self.processor.process(trace_message)
        self.processor.process(empty_trace_feedback_batch)
        self.processor.process(empty_span_feedback_batch)

        trace_trees = self.processor.trace_trees

        assert len(trace_trees) == 1
        trace = trace_trees[0]
        assert trace.id == "trace_1"
        assert len(trace.feedback_scores) == 0
        assert len(trace.spans) == 0

    def test_trace_trees_with_multiple_feedback_scores_batch_for_same_entity(self):
        # Test multiple batch messages targeting the same trace/span
        trace_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="test_trace",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=None,
            source="sdk",
        )

        # First batch of trace feedback scores
        first_batch = messages.AddTraceFeedbackScoresBatchMessage(
            batch=[
                messages.FeedbackScoreMessage(
                    id="trace_1",
                    project_name="test_project",
                    name="accuracy",
                    value=0.95,
                    source="user",
                    category_name="evaluation",
                )
            ]
        )

        # Second batch of trace feedback scores
        second_batch = messages.AddTraceFeedbackScoresBatchMessage(
            batch=[
                messages.FeedbackScoreMessage(
                    id="trace_1",
                    project_name="test_project",
                    name="relevance",
                    value=0.88,
                    source="automated",
                    category_name="quality",
                ),
                messages.FeedbackScoreMessage(
                    id="trace_1",
                    project_name="test_project",
                    name="completeness",
                    value=0.92,
                    source="user",
                    category_name="evaluation",
                ),
            ]
        )

        self.processor.process(trace_message)
        self.processor.process(first_batch)
        self.processor.process(second_batch)

        trace_trees = self.processor.trace_trees

        assert len(trace_trees) == 1
        trace = trace_trees[0]
        assert trace.id == "trace_1"
        assert len(trace.feedback_scores) == 3  # Should accumulate all feedback scores

        # Check all feedback scores are present
        feedback_names = [fs.name for fs in trace.feedback_scores]
        assert "accuracy" in feedback_names
        assert "relevance" in feedback_names
        assert "completeness" in feedback_names

    def test_trace_trees_with_create_trace_batch_message(self):
        # Test CreateTraceBatchMessage with multiple TraceWrite objects
        trace_write_1 = trace_write.TraceWrite(
            id="trace_1",
            project_name="test_project",
            name="first_trace",
            start_time=self.test_datetime,
            end_time=None,
            input={"input": "test1"},
            output={"output": "result1"},
            metadata={"meta": "data1"},
            tags=["tag1"],
            error_info=None,
            last_updated_at=None,
            thread_id="thread_1",
            source="sdk",
        )

        trace_write_2 = trace_write.TraceWrite(
            id="trace_2",
            project_name="test_project",
            name="second_trace",
            start_time=self.later_datetime,
            end_time=None,
            input={"input": "test2"},
            output={"output": "result2"},
            metadata={"meta": "data2"},
            tags=["tag2"],
            error_info=None,
            last_updated_at=None,
            thread_id="thread_2",
            source="sdk",
        )

        trace_batch_message = messages.CreateTraceBatchMessage(
            batch=[trace_write_1, trace_write_2]
        )

        self.processor.process(trace_batch_message)
        trace_trees = self.processor.trace_trees
        assert len(trace_trees) == 2

        pprint.pprint(trace_trees)

        EXPECTED_TRACE_TREE = [
            models.TraceModel(
                id="trace_1",
                start_time=self.test_datetime,
                name="first_trace",
                project_name="test_project",
                input={"input": "test1"},
                output={"output": "result1"},
                tags=["tag1"],
                metadata={"meta": "data1"},
                end_time=None,
                spans=[],
                feedback_scores=[],
                error_info=None,
                thread_id="thread_1",
                last_updated_at=None,
                source="sdk",
            ),
            models.TraceModel(
                id="trace_2",
                start_time=self.later_datetime,
                name="second_trace",
                project_name="test_project",
                input={"input": "test2"},
                output={"output": "result2"},
                tags=["tag2"],
                metadata={"meta": "data2"},
                end_time=None,
                spans=[],
                feedback_scores=[],
                error_info=None,
                thread_id="thread_2",
                last_updated_at=None,
                source="sdk",
            ),
        ]

        assert_helpers.assert_equal(expected=EXPECTED_TRACE_TREE, actual=trace_trees)

    def test_trace_trees_with_create_spans_batch_message(self):
        # First, create a trace to attach spans to
        trace_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="test_trace",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=self.later_datetime,
            source="sdk",
        )

        # Create batch span write objects
        span_write_1 = span_write.SpanWrite(
            id="span_1",
            project_name="test_project",
            trace_id="trace_1",
            parent_span_id=None,
            name="first_span",
            type="general",
            start_time=self.test_datetime,
            end_time=None,
            input={"input": "span1_input"},
            output={"output": "span1_result"},
            metadata={"meta": "span1_data"},
            model="gpt-3.5",
            provider="openai",
            tags=["span1_tag"],
            usage={"tokens": 50},
            error_info=None,
            last_updated_at=None,
            total_estimated_cost=0.005,
            source="sdk",
        )

        span_write_2 = span_write.SpanWrite(
            id="span_2",
            project_name="test_project",
            trace_id="trace_1",
            parent_span_id=None,
            name="second_span",
            type="tool",
            start_time=self.later_datetime,
            end_time=None,
            input={"input": "span2_input"},
            output={"output": "span2_result"},
            metadata={"meta": "span2_data"},
            model="gpt-4",
            provider="openai",
            tags=["span2_tag"],
            usage={"tokens": 100},
            error_info=None,
            last_updated_at=None,
            total_estimated_cost=0.02,
            source="sdk",
        )

        spans_batch_message = messages.CreateSpansBatchMessage(
            batch=[span_write_1, span_write_2]
        )

        self.processor.process(trace_message)
        self.processor.process(spans_batch_message)

        trace_trees = self.processor.trace_trees
        assert len(trace_trees) == 1

        EXPECTED_TRACE_TREE = models.TraceModel(
            id="trace_1",
            start_time=self.test_datetime,
            name="test_trace",
            project_name="test_project",
            input=None,
            output=None,
            tags=None,
            metadata=None,
            end_time=None,
            spans=[
                models.SpanModel(
                    id="span_1",
                    start_time=self.test_datetime,
                    name="first_span",
                    input={"input": "span1_input"},
                    output={"output": "span1_result"},
                    tags=["span1_tag"],
                    metadata={"meta": "span1_data"},
                    type="general",
                    usage={"tokens": 50},
                    end_time=None,
                    project_name="test_project",
                    spans=[],
                    feedback_scores=[],
                    model="gpt-3.5",
                    provider="openai",
                    error_info=None,
                    total_cost=0.005,
                    last_updated_at=None,
                    source="sdk",
                ),
                models.SpanModel(
                    id="span_2",
                    start_time=self.later_datetime,
                    name="second_span",
                    input={"input": "span2_input"},
                    output={"output": "span2_result"},
                    tags=["span2_tag"],
                    metadata={"meta": "span2_data"},
                    type="tool",
                    usage={"tokens": 100},
                    end_time=None,
                    project_name="test_project",
                    spans=[],
                    feedback_scores=[],
                    model="gpt-4",
                    provider="openai",
                    error_info=None,
                    total_cost=0.02,
                    last_updated_at=None,
                    source="sdk",
                ),
            ],
            feedback_scores=[],
            error_info=None,
            thread_id=None,
            last_updated_at=self.later_datetime,
            source="sdk",
        )

        assert_helpers.assert_equal(expected=EXPECTED_TRACE_TREE, actual=trace_trees[0])

    def test_trace_trees_with_nested_spans_batch_message(self):
        # Test CreateSpansBatchMessage with nested spans (parent-child relationships)
        trace_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="test_trace",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=self.later_datetime,
            source="sdk",
        )

        parent_span_write = span_write.SpanWrite(
            id="parent_span",
            project_name="test_project",
            trace_id="trace_1",
            parent_span_id=None,
            name="parent_span",
            type="general",
            start_time=self.test_datetime,
            end_time=None,
            input={"parent": "input"},
            output={"parent": "output"},
            metadata=None,
            model=None,
            provider=None,
            tags=None,
            usage=None,
            error_info=None,
            last_updated_at=None,
            source="sdk",
        )

        child_span_write = span_write.SpanWrite(
            id="child_span",
            project_name="test_project",
            trace_id="trace_1",
            parent_span_id="parent_span",
            name="child_span",
            type="tool",
            start_time=self.later_datetime,
            end_time=None,
            input={"child": "input"},
            output={"child": "output"},
            metadata=None,
            model=None,
            provider=None,
            tags=None,
            usage=None,
            error_info=None,
            last_updated_at=None,
            source="sdk",
        )

        spans_batch_message = messages.CreateSpansBatchMessage(
            batch=[parent_span_write, child_span_write]
        )

        self.processor.process(trace_message)
        self.processor.process(spans_batch_message)

        trace_trees = self.processor.trace_trees
        assert len(trace_trees) == 1

        EXPECTED_TRACE_TREE = models.TraceModel(
            id="trace_1",
            start_time=self.test_datetime,
            name="test_trace",
            project_name="test_project",
            input=None,
            output=None,
            tags=None,
            metadata=None,
            end_time=None,
            spans=[
                models.SpanModel(
                    id="parent_span",
                    start_time=self.test_datetime,
                    name="parent_span",
                    input={"parent": "input"},
                    output={"parent": "output"},
                    tags=None,
                    metadata=None,
                    type="general",
                    usage=None,
                    end_time=None,
                    project_name="test_project",
                    spans=[
                        models.SpanModel(
                            id="child_span",
                            start_time=self.later_datetime,
                            name="child_span",
                            input={"child": "input"},
                            output={"child": "output"},
                            tags=None,
                            metadata=None,
                            type="tool",
                            usage=None,
                            end_time=None,
                            project_name="test_project",
                            spans=[],
                            feedback_scores=[],
                            model=None,
                            provider=None,
                            error_info=None,
                            total_cost=None,
                            last_updated_at=None,
                            source="sdk",
                        )
                    ],
                    feedback_scores=[],
                    model=None,
                    provider=None,
                    error_info=None,
                    total_cost=None,
                    last_updated_at=None,
                    source="sdk",
                )
            ],
            feedback_scores=[],
            error_info=None,
            thread_id=None,
            last_updated_at=self.later_datetime,
            source="sdk",
        )
        assert_helpers.assert_equal(expected=EXPECTED_TRACE_TREE, actual=trace_trees[0])

    def test_trace_trees_with_empty_batch_messages(self):
        # Test empty batch messages don't cause issues
        trace_message = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="test_trace",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=None,
            source="sdk",
        )

        empty_traces_batch = messages.CreateTraceBatchMessage(batch=[])
        empty_spans_batch = messages.CreateSpansBatchMessage(batch=[])

        self.processor.process(trace_message)
        self.processor.process(empty_traces_batch)
        self.processor.process(empty_spans_batch)

        trace_trees = self.processor.trace_trees

        assert len(trace_trees) == 1
        trace = trace_trees[0]
        assert trace.id == "trace_1"
        assert len(trace.spans) == 0

    def test_trace_trees_with_mixed_batch_and_individual_messages(self):
        # Test a combination of batch messages and individual create messages
        # Individual trace message
        individual_trace = messages.CreateTraceMessage(
            trace_id="trace_1",
            project_name="test_project",
            name="individual_trace",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            thread_id=None,
            last_updated_at=None,
            source="sdk",
        )

        # Batch trace message
        batch_trace_write = trace_write.TraceWrite(
            id="trace_2",
            project_name="test_project",
            name="batch_trace",
            start_time=self.later_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            error_info=None,
            last_updated_at=None,
            thread_id=None,
            source="sdk",
        )

        trace_batch = messages.CreateTraceBatchMessage(batch=[batch_trace_write])

        # Individual span message
        individual_span = messages.CreateSpanMessage(
            span_id="span_1",
            trace_id="trace_1",
            project_name="test_project",
            parent_span_id=None,
            name="individual_span",
            start_time=self.test_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            tags=None,
            type="general",
            usage=None,
            model=None,
            provider=None,
            error_info=None,
            total_cost=None,
            last_updated_at=None,
            source="sdk",
        )

        # Batch span message
        batch_span_write = span_write.SpanWrite(
            id="span_2",
            project_name="test_project",
            trace_id="trace_2",
            parent_span_id=None,
            name="batch_span",
            type="tool",
            start_time=self.later_datetime,
            end_time=None,
            input=None,
            output=None,
            metadata=None,
            model=None,
            provider=None,
            tags=None,
            usage=None,
            error_info=None,
            last_updated_at=None,
            source="sdk",
        )

        spans_batch = messages.CreateSpansBatchMessage(batch=[batch_span_write])

        # Process all messages
        self.processor.process(individual_trace)
        self.processor.process(trace_batch)
        self.processor.process(individual_span)
        self.processor.process(spans_batch)

        trace_trees = self.processor.trace_trees
        assert len(trace_trees) == 2

        EXPECTED_TRACE_TREE = [
            models.TraceModel(
                id="trace_1",
                start_time=self.test_datetime,
                name="individual_trace",
                project_name="test_project",
                input=None,
                output=None,
                tags=None,
                metadata=None,
                end_time=None,
                spans=[
                    models.SpanModel(
                        id="span_1",
                        start_time=self.test_datetime,
                        name="individual_span",
                        input=None,
                        output=None,
                        tags=None,
                        metadata=None,
                        type="general",
                        usage=None,
                        end_time=None,
                        project_name="test_project",
                        spans=[],
                        feedback_scores=[],
                        model=None,
                        provider=None,
                        error_info=None,
                        total_cost=None,
                        last_updated_at=None,
                        source="sdk",
                    )
                ],
                feedback_scores=[],
                error_info=None,
                thread_id=None,
                last_updated_at=None,
                source="sdk",
            ),
            models.TraceModel(
                id="trace_2",
                start_time=self.later_datetime,
                name="batch_trace",
                project_name="test_project",
                input=None,
                output=None,
                tags=None,
                metadata=None,
                end_time=None,
                spans=[
                    models.SpanModel(
                        id="span_2",
                        start_time=self.later_datetime,
                        name="batch_span",
                        input=None,
                        output=None,
                        tags=None,
                        metadata=None,
                        type="tool",
                        usage=None,
                        end_time=None,
                        project_name="test_project",
                        spans=[],
                        feedback_scores=[],
                        model=None,
                        provider=None,
                        error_info=None,
                        total_cost=None,
                        last_updated_at=None,
                        source="sdk",
                    )
                ],
                feedback_scores=[],
                error_info=None,
                thread_id=None,
                last_updated_at=None,
                source="sdk",
            ),
        ]
        assert_helpers.assert_equal(expected=EXPECTED_TRACE_TREE, actual=trace_trees)


# --- sdks/python/tests/unit/message_processing/replay/test_db_manager.py ---

    def test_get_message__existing_message__returns_deserialized_message(
        self, manager: db_manager.DBManager
    ):
        """Test getting an existing message returns the deserialized BaseMessage."""
        original = _create_trace_message(message_id=1)
        manager.register_message(original)

        result = manager.get_message(message_id=1)

        assert result is not None
        assert isinstance(result, messages.CreateTraceMessage)
        assert result.trace_id == "trace-1"
        assert result.project_name == "test-project"

    def test_replay_failed_messages__callback_invocation__receives_deserialized_messages(
        self, manager: db_manager.DBManager
    ):
        """Test that callback receives properly deserialized BaseMessage objects."""
        original = _create_trace_message(message_id=1)
        manager.register_message(original, status=db_manager.MessageStatus.failed)

        received_messages = []

        def callback(msg: messages.BaseMessage) -> None:
            received_messages.append(msg)

        manager.replay_failed_messages(callback)

        assert len(received_messages) == 1
        assert isinstance(received_messages[0], messages.CreateTraceMessage)
        assert received_messages[0].trace_id == "trace-1"

    def test_db_message_to_message__supported_type__returns_correct_message(
        self, manager: db_manager.DBManager
    ):
        """Test conversion of DBMessage to BaseMessage for supported types."""
        original = _create_trace_message(message_id=1)
        manager.register_message(original)

        db_message = manager.get_db_message(1)
        result = db_manager.db_message_to_message(db_message)

        assert isinstance(result, messages.CreateTraceMessage)
        assert result.trace_id == "trace-1"


# --- sdks/python/tests/unit/message_processing/replay/test_message_serialization.py ---

    def test_add_feedback_scores_batch_message__round_trip_serialization__preserves_data(
        self,
    ):
        """Test round-trip serialization/deserialization through JSON."""
        feedback_scores = [
            messages.FeedbackScoreMessage(
                id="score-1",
                project_name="test-project",
                name="accuracy",
                value=0.95,
                source="sdk",
                reason="Good prediction",
                category_name="metrics",
            ),
            messages.FeedbackScoreMessage(
                id="score-2",
                project_name="test-project-2",
                name="latency",
                value=0.5,
                source="api",
                reason=None,
                category_name=None,
            ),
        ]

        original = messages.AddFeedbackScoresBatchMessage(batch=feedback_scores)

        # Serialize to JSON string
        json_str = message_serialization.serialize_message(original)

        # Deserialize to message
        deserialized = message_serialization.deserialize_message(
            message_class=messages.AddFeedbackScoresBatchMessage,
            json_str=json_str,
        )

        # Verify deserialized message
        assert isinstance(deserialized, messages.AddFeedbackScoresBatchMessage)
        assert deserialized.supports_batching is True
        assert len(deserialized.batch) == 2

        # Verify the first batch item - all fields
        item0 = deserialized.batch[0]
        assert isinstance(item0, messages.FeedbackScoreMessage)
        assert item0.id == "score-1"
        assert item0.project_name == "test-project"
        assert item0.name == "accuracy"
        assert item0.value == 0.95
        assert item0.source == "sdk"
        assert item0.reason == "Good prediction"
        assert item0.category_name == "metrics"

        # Verify the second batch item - all fields
        item1 = deserialized.batch[1]
        assert isinstance(item1, messages.FeedbackScoreMessage)
        assert item1.id == "score-2"
        assert item1.project_name == "test-project-2"
        assert item1.name == "latency"
        assert item1.value == 0.5
        assert item1.source == "api"
        assert item1.reason is None
        assert item1.category_name is None

    def test_create_spans_batch_message__round_trip_serialization__preserves_data(self):
        """Test round-trip serialization/deserialization through JSON."""
        start_time = datetime.datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
        )
        end_time = datetime.datetime(2024, 1, 1, 12, 0, 1, tzinfo=datetime.timezone.utc)

        spans = [
            span_write.SpanWrite(
                id="span-1",
                trace_id="trace-1",
                project_name="test-project",
                parent_span_id="parent-span-1",
                name="test-span",
                type="llm",
                start_time=start_time,
                end_time=end_time,
                input={"prompt": "test"},
                output={"response": "result"},
                metadata={"key": "value"},
                model="gpt-4",
                provider="openai",
                tags=["tag1", "tag2"],
                usage={"prompt_tokens": 10, "completion_tokens": 20},
                total_estimated_cost=0.001,
            ),
        ]

        original = messages.CreateSpansBatchMessage(batch=spans)

        # Serialize to JSON string
        json_str = message_serialization.serialize_message(original)

        # Deserialize to message
        deserialized = message_serialization.deserialize_message(
            message_class=messages.CreateSpansBatchMessage,
            json_str=json_str,
        )

        # Verify deserialized message
        assert isinstance(deserialized, messages.CreateSpansBatchMessage)
        assert len(deserialized.batch) == 1

        # Verify batch item - all fields
        item = deserialized.batch[0]
        assert isinstance(item, span_write.SpanWrite)
        assert item.id == "span-1"
        assert item.trace_id == "trace-1"
        assert item.project_name == "test-project"
        assert item.parent_span_id == "parent-span-1"
        assert item.name == "test-span"
        assert item.type == "llm"
        assert item.start_time == start_time
        assert item.end_time == end_time
        assert item.input == {"prompt": "test"}
        assert item.output == {"response": "result"}
        assert item.metadata == {"key": "value"}
        assert item.model == "gpt-4"
        assert item.provider == "openai"
        assert item.tags == ["tag1", "tag2"]
        assert item.usage == {"prompt_tokens": 10, "completion_tokens": 20}
        assert item.total_estimated_cost == 0.001

    def test_create_trace_batch_message__round_trip_serialization__preserves_data(self):
        """Test round-trip serialization/deserialization through JSON."""
        start_time = datetime.datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
        )
        end_time = datetime.datetime(2024, 1, 1, 12, 0, 1, tzinfo=datetime.timezone.utc)

        traces = [
            trace_write.TraceWrite(
                id="trace-1",
                project_name="test-project",
                name="test-trace",
                start_time=start_time,
                end_time=end_time,
                input={"query": "test input"},
                output={"answer": "test output"},
                metadata={"meta_key": "meta_value"},
                tags=["trace-tag1", "trace-tag2"],
            ),
        ]

        original = messages.CreateTraceBatchMessage(batch=traces)

        # Serialize to JSON string
        json_str = message_serialization.serialize_message(original)

        # Deserialize to message
        deserialized = message_serialization.deserialize_message(
            message_class=messages.CreateTraceBatchMessage,
            json_str=json_str,
        )

        # Verify deserialized message
        assert isinstance(deserialized, messages.CreateTraceBatchMessage)
        assert len(deserialized.batch) == 1

        # Verify batch item - all fields
        item = deserialized.batch[0]
        assert isinstance(item, trace_write.TraceWrite)
        assert item.id == "trace-1"
        assert item.project_name == "test-project"
        assert item.name == "test-trace"
        assert item.start_time == start_time
        assert item.end_time == end_time
        assert item.input == {"query": "test input"}
        assert item.output == {"answer": "test output"}
        assert item.metadata == {"meta_key": "meta_value"}
        assert item.tags == ["trace-tag1", "trace-tag2"]

    def test_create_experiment_items_batch_message__round_trip_serialization__preserves_data(
        self,
    ):
        """Test round-trip serialization/deserialization through JSON."""
        experiment_items = [
            messages.ExperimentItemMessage(
                id="item-1",
                experiment_id="exp-1",
                trace_id="trace-1",
                dataset_item_id="dataset-item-1",
            ),
            messages.ExperimentItemMessage(
                id="item-2",
                experiment_id="exp-2",
                trace_id="trace-2",
                dataset_item_id="dataset-item-2",
            ),
        ]

        original = messages.CreateExperimentItemsBatchMessage(batch=experiment_items)

        # Serialize to JSON string
        json_str = message_serialization.serialize_message(original)

        # Deserialize to message
        deserialized = message_serialization.deserialize_message(
            message_class=messages.CreateExperimentItemsBatchMessage,
            json_str=json_str,
        )

        # Verify deserialized message
        assert isinstance(deserialized, messages.CreateExperimentItemsBatchMessage)
        assert deserialized.supports_batching is True
        assert len(deserialized.batch) == 2

        # Verify the first batch item - all fields
        item0 = deserialized.batch[0]
        assert isinstance(item0, messages.ExperimentItemMessage)
        assert item0.id == "item-1"
        assert item0.experiment_id == "exp-1"
        assert item0.trace_id == "trace-1"
        assert item0.dataset_item_id == "dataset-item-1"

        # Verify the second batch item - all fields
        item1 = deserialized.batch[1]
        assert isinstance(item1, messages.ExperimentItemMessage)
        assert item1.id == "item-2"
        assert item1.experiment_id == "exp-2"
        assert item1.trace_id == "trace-2"
        assert item1.dataset_item_id == "dataset-item-2"

    def test_create_attachment_message__round_trip_serialization__preserves_data(self):
        """Test round-trip serialization/deserialization through JSON."""
        original = messages.CreateAttachmentMessage(
            file_path="/path/to/document.pdf",
            file_name="document.pdf",
            mime_type="application/pdf",
            entity_type="span",
            entity_id="span-123",
            project_name="test-project",
            encoded_url_override="https://storage.example.com/uploads/document.pdf",
            delete_after_upload=True,
        )

        # Serialize to JSON string
        json_str = message_serialization.serialize_message(original)

        # Deserialize to message
        deserialized = message_serialization.deserialize_message(
            message_class=messages.CreateAttachmentMessage,
            json_str=json_str,
        )

        # Verify deserialized message - all fields
        assert isinstance(deserialized, messages.CreateAttachmentMessage)
        assert deserialized.file_path == "/path/to/document.pdf"
        assert deserialized.file_name == "document.pdf"
        assert deserialized.mime_type == "application/pdf"
        assert deserialized.entity_type == "span"
        assert deserialized.entity_id == "span-123"
        assert deserialized.project_name == "test-project"
        assert (
            deserialized.encoded_url_override
            == "https://storage.example.com/uploads/document.pdf"
        )
        assert deserialized.delete_after_upload is True

    def test_create_trace_message__round_trip_serialization__preserves_data(self):
        """Test round-trip serialization/deserialization through JSON."""
        start_time = datetime.datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
        )
        end_time = datetime.datetime(2024, 1, 1, 12, 0, 1, tzinfo=datetime.timezone.utc)
        last_updated_at = datetime.datetime(
            2024, 1, 1, 12, 0, 2, tzinfo=datetime.timezone.utc
        )
        error_info = ErrorInfoDict(
            exception_type="ValueError",
            message="test error",
            traceback="Traceback (most recent call last):\n",
        )

        original = messages.CreateTraceMessage(
            trace_id="trace-1",
            project_name="test-project",
            name="test-trace",
            start_time=start_time,
            end_time=end_time,
            input={"query": "test input"},
            output={"answer": "test output"},
            metadata={"meta_key": "meta_value"},
            tags=["tag1", "tag2"],
            error_info=error_info,
            thread_id="thread-1",
            last_updated_at=last_updated_at,
            source="sdk",
        )

        # Serialize to JSON string
        json_str = message_serialization.serialize_message(original)

        # Deserialize to message
        deserialized = message_serialization.deserialize_message(
            message_class=messages.CreateTraceMessage,
            json_str=json_str,
        )

        # Verify deserialized message - all fields
        assert isinstance(deserialized, messages.CreateTraceMessage)
        assert deserialized.trace_id == "trace-1"
        assert deserialized.project_name == "test-project"
        assert deserialized.name == "test-trace"
        # Datetime fields become strings after JSON roundtrip
        assert str(deserialized.start_time) == str(start_time)
        assert str(deserialized.end_time) == str(end_time)
        assert deserialized.input == {"query": "test input"}
        assert deserialized.output == {"answer": "test output"}
        assert deserialized.metadata == {"meta_key": "meta_value"}
        assert deserialized.tags == ["tag1", "tag2"]
        assert deserialized.error_info == error_info
        assert deserialized.thread_id == "thread-1"
        assert str(deserialized.last_updated_at) == str(last_updated_at)
        assert deserialized.source == "sdk"

    def test_create_span_message__round_trip_serialization__preserves_data(self):
        """Test round-trip serialization/deserialization through JSON."""
        start_time = datetime.datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
        )
        end_time = datetime.datetime(2024, 1, 1, 12, 0, 1, tzinfo=datetime.timezone.utc)
        last_updated_at = datetime.datetime(
            2024, 1, 1, 12, 0, 2, tzinfo=datetime.timezone.utc
        )

        error_info = ErrorInfoDict(
            exception_type="ValueError",
            message="test error",
            traceback="Traceback (most recent call last):\n",
        )
        original = messages.CreateSpanMessage(
            span_id="span-1",
            trace_id="trace-1",
            project_name="test-project",
            parent_span_id="parent-span-1",
            name="test-span",
            start_time=start_time,
            end_time=end_time,
            input={"prompt": "test prompt"},
            output={"response": "test response"},
            metadata={"span_meta": "value"},
            tags=["span-tag1", "span-tag2"],
            type="llm",
            usage={"prompt_tokens": 100, "completion_tokens": 200},
            model="gpt-4",
            provider="openai",
            error_info=error_info,
            total_cost=0.05,
            last_updated_at=last_updated_at,
            source="sdk",
        )

        # Serialize to JSON string
        json_str = message_serialization.serialize_message(original)

        # Deserialize to message
        deserialized = message_serialization.deserialize_message(
            message_class=messages.CreateSpanMessage,
            json_str=json_str,
        )

        # Verify deserialized message - all fields
        assert isinstance(deserialized, messages.CreateSpanMessage)
        assert deserialized.span_id == "span-1"
        assert deserialized.trace_id == "trace-1"
        assert deserialized.project_name == "test-project"
        assert deserialized.parent_span_id == "parent-span-1"
        assert deserialized.name == "test-span"
        # Datetime fields become strings after JSON roundtrip
        assert str(deserialized.start_time) == str(start_time)
        assert str(deserialized.end_time) == str(end_time)
        assert deserialized.input == {"prompt": "test prompt"}
        assert deserialized.output == {"response": "test response"}
        assert deserialized.metadata == {"span_meta": "value"}
        assert deserialized.tags == ["span-tag1", "span-tag2"]
        assert deserialized.type == "llm"
        assert deserialized.usage == {"prompt_tokens": 100, "completion_tokens": 200}
        assert deserialized.model == "gpt-4"
        assert deserialized.provider == "openai"
        assert deserialized.error_info == error_info
        assert deserialized.total_cost == 0.05
        assert str(deserialized.last_updated_at) == str(last_updated_at)
        assert deserialized.source == "sdk"

    def test_update_span_message__round_trip_serialization__preserves_data(self):
        """Test round-trip serialization/deserialization through JSON."""
        end_time = datetime.datetime(2024, 1, 1, 12, 0, 1, tzinfo=datetime.timezone.utc)

        original = messages.UpdateSpanMessage(
            span_id="span-1",
            parent_span_id="parent-span-2",
            trace_id="trace-1",
            project_name="test-project",
            end_time=end_time,
            input={"prompt": "updated prompt"},
            output={"response": "updated response"},
            metadata={"updated_meta": "new_value"},
            tags=["updated-span-tag"],
            usage={"prompt_tokens": 150, "completion_tokens": 250},
            model="gpt-4-turbo",
            provider="openai",
            error_info=None,
            total_cost=0.08,
            source="sdk",
        )

        # Serialize to JSON string
        json_str = message_serialization.serialize_message(original)

        # Deserialize to message
        deserialized = message_serialization.deserialize_message(
            message_class=messages.UpdateSpanMessage,
            json_str=json_str,
        )

        # Verify deserialized message - all fields
        assert isinstance(deserialized, messages.UpdateSpanMessage)
        assert deserialized.span_id == "span-1"
        assert deserialized.parent_span_id == "parent-span-2"
        assert deserialized.trace_id == "trace-1"
        assert deserialized.project_name == "test-project"
        # Datetime fields become strings after JSON roundtrip
        assert str(deserialized.end_time) == str(end_time)
        assert deserialized.input == {"prompt": "updated prompt"}
        assert deserialized.output == {"response": "updated response"}
        assert deserialized.metadata == {"updated_meta": "new_value"}
        assert deserialized.tags == ["updated-span-tag"]
        assert deserialized.usage == {"prompt_tokens": 150, "completion_tokens": 250}
        assert deserialized.model == "gpt-4-turbo"
        assert deserialized.provider == "openai"
        assert deserialized.error_info is None
        assert deserialized.total_cost == 0.08
        assert deserialized.source == "sdk"

    def test_create_trace__iso_strings_in_input_output_metadata__preserved_as_strings(
        self,
    ):
        """ISO strings in input/output/metadata dicts survive a round-trip as strings."""
        start_time = datetime.datetime(2024, 1, 1, 12, 0, 0)

        original = messages.CreateTraceMessage(
            trace_id="trace-1",
            project_name="test-project",
            name="test-trace",
            start_time=start_time,
            end_time=None,
            input={"query": "test", "requested_at": "2024-06-15T09:00:00Z"},
            output={"answer": "result", "completed_at": "2024-06-15T09:01:00"},
            metadata={"event_time": "2024-06-15T09:00:30.123456+02:00"},
            tags=[],
            error_info=None,
            thread_id=None,
            last_updated_at=None,
            source="sdk",
        )

        json_str = message_serialization.serialize_message(original)
        deserialized = message_serialization.deserialize_message(
            message_class=messages.CreateTraceMessage,
            json_str=json_str,
        )

        # Datetime fields are properly converted
        assert isinstance(deserialized.start_time, datetime.datetime)

        # ISO strings inside input/output/metadata must remain as strings
        assert deserialized.input["requested_at"] == "2024-06-15T09:00:00Z"
        assert isinstance(deserialized.input["requested_at"], str)

        assert deserialized.output["completed_at"] == "2024-06-15T09:01:00"
        assert isinstance(deserialized.output["completed_at"], str)

        assert deserialized.metadata["event_time"] == "2024-06-15T09:00:30.123456+02:00"
        assert isinstance(deserialized.metadata["event_time"], str)

    def test_create_span__iso_strings_in_input_output_metadata__preserved_as_strings(
        self,
    ):
        """ISO strings in span input/output/metadata dicts survive a round-trip as strings."""
        start_time = datetime.datetime(2024, 1, 1, 12, 0, 0)

        original = messages.CreateSpanMessage(
            span_id="span-1",
            trace_id="trace-1",
            project_name="test-project",
            parent_span_id=None,
            name="test-span",
            start_time=start_time,
            end_time=None,
            input={"timestamp": "2024-06-15T09:00:00"},
            output={"created": "2024-06-15T09:01:00Z"},
            metadata={"logged_at": "2024-06-15T09:00:30"},
            tags=[],
            type="general",
            usage=None,
            model=None,
            provider=None,
            error_info=None,
            total_cost=None,
            last_updated_at=None,
            source="sdk",
        )

        json_str = message_serialization.serialize_message(original)
        deserialized = message_serialization.deserialize_message(
            message_class=messages.CreateSpanMessage,
            json_str=json_str,
        )

        assert isinstance(deserialized.start_time, datetime.datetime)

        assert deserialized.input["timestamp"] == "2024-06-15T09:00:00"
        assert isinstance(deserialized.input["timestamp"], str)

        assert deserialized.output["created"] == "2024-06-15T09:01:00Z"
        assert isinstance(deserialized.output["created"], str)

        assert deserialized.metadata["logged_at"] == "2024-06-15T09:00:30"
        assert isinstance(deserialized.metadata["logged_at"], str)

    def test_create_spans_batch__iso_strings_in_span_data_fields__preserved_as_strings(
        self,
    ):
        """ISO strings in batch span input/output/metadata survive a round-trip as strings."""
        start_time = datetime.datetime(2024, 1, 1, 12, 0, 0)

        from opik.rest_api.types import span_write

        spans = [
            span_write.SpanWrite(
                id="span-1",
                trace_id="trace-1",
                project_name="test-project",
                name="test-span",
                type="llm",
                start_time=start_time,
                input={"user_ts": "2024-03-10T14:30:00Z"},
                output={"llm_ts": "2024-03-10T14:30:05"},
                metadata={"logged": "2024-03-10T14:30:01+05:00"},
            ),
        ]

        original = messages.CreateSpansBatchMessage(batch=spans)

        json_str = message_serialization.serialize_message(original)
        deserialized = message_serialization.deserialize_message(
            message_class=messages.CreateSpansBatchMessage,
            json_str=json_str,
        )

        item = deserialized.batch[0]
        assert isinstance(item.start_time, datetime.datetime)

        assert item.input["user_ts"] == "2024-03-10T14:30:00Z"
        assert isinstance(item.input["user_ts"], str)

        assert item.output["llm_ts"] == "2024-03-10T14:30:05"
        assert isinstance(item.output["llm_ts"], str)

        assert item.metadata["logged"] == "2024-03-10T14:30:01+05:00"
        assert isinstance(item.metadata["logged"], str)


# --- sdks/python/tests/unit/message_processing/replay/test_replay_manager.py ---

    def test_loop__connection_restored__replays_failed_messages(
        self,
        manager_monitor: Tuple[
            replay_manager.ReplayManager, connection_monitor.OpikConnectionMonitor
        ],
    ):
        rm, monitor = manager_monitor
        monitor.tick.return_value = (
            connection_monitor.ConnectionStatus.connection_restored
        )

        # Register a failed message
        msg = _create_trace_message(message_id=1)
        rm.database_manager.register_message(
            msg, status=db_manager.MessageStatus.failed
        )

        replayed: List[messages.BaseMessage] = []

        def callback(m: messages.BaseMessage) -> None:
            replayed.append(m)

        rm.set_replay_callback(callback)
        rm.start()

        # Wait for at least one tick cycle
        deadline = time.time() + 0.5
        while not replayed and time.time() < deadline:
            time.sleep(0.05)

        rm.close()
        rm.join(timeout=2)

        assert len(replayed) == 1
        assert isinstance(replayed[0], messages.CreateTraceMessage)
        assert replayed[0].trace_id == "trace-1"

    def test_loop__replay_callback_raises__thread_continues(
        self,
        manager_monitor: Tuple[
            replay_manager.ReplayManager, connection_monitor.OpikConnectionMonitor
        ],
    ):
        """If replay_failed_messages raises, the loop should continue."""
        rm, monitor = manager_monitor

        msg = _create_trace_message(message_id=1)
        rm.database_manager.register_message(
            msg, status=db_manager.MessageStatus.failed
        )
        # mark connection as restored to enable replay callback
        monitor.tick.return_value = (
            connection_monitor.ConnectionStatus.connection_restored
        )

        call_count = 0

        def failing_callback(m: messages.BaseMessage) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("replay error")

        rm.set_replay_callback(failing_callback)
        rm.start()

        # Wait for a few ticks
        deadline = time.time() + 2.0
        while call_count < 2 and time.time() < deadline:
            time.sleep(0.05)

        rm.close()
        rm.join(timeout=2)

        assert call_count >= 2, "Loop should have continued despite replay errors"

        assert rm.is_alive() is False, "Thread should have exited cleanly"

    def test_flush_and_loop__serialized_by_replay_lock(
        self,
        manager_monitor: Tuple[
            replay_manager.ReplayManager, connection_monitor.OpikConnectionMonitor
        ],
    ):
        """flush() and the loop both acquire _replay_lock, so only one replay
        can happen at a time."""
        rm, monitor = manager_monitor

        for i in range(5):
            msg = _create_trace_message(message_id=i + 1, trace_id=f"trace-{i}")
            rm.database_manager.register_message(
                msg, status=db_manager.MessageStatus.failed
            )

        replayed_ids: List[str] = []
        lock = threading.Lock()

        def callback(m: messages.BaseMessage) -> None:
            with lock:
                assert isinstance(m, messages.CreateTraceMessage)
                replayed_ids.append(m.trace_id)

        rm.set_replay_callback(callback)
        rm.start()

        # Also call flush from the main thread
        rm.flush()

        # Let the loop run too
        time.sleep(0.3)

        rm.close()
        rm.join(timeout=2)

        # Each message should appear at least once (no corruption)
        assert len(replayed_ids) == 5

    def test_full_lifecycle__register_fail_restore_replay(
        self,
        manager_monitor: Tuple[
            replay_manager.ReplayManager, connection_monitor.OpikConnectionMonitor
        ],
    ):
        """Test the complete message lifecycle:
        register → fail → connection restored → replay → delivered."""
        rm, monitor = manager_monitor

        replayed: List[messages.BaseMessage] = []

        def callback(m: messages.BaseMessage) -> None:
            replayed.append(m)

        rm.set_replay_callback(callback)

        # 1. Register a message
        msg = _create_trace_message(message_id=1)
        rm.register_message(msg)

        # 2. Mark it as failed (simulating a connection error during sending)
        rm.message_sent_failed(1, failure_reason="connection timeout")
        db_msg = rm.database_manager.get_db_message(1)
        assert db_msg is not None
        assert db_msg.status == db_manager.MessageStatus.failed

        # 3. Start the thread — connection is still failed, no replay
        rm.start()
        time.sleep(0.2)
        assert len(replayed) == 0

        # 4. Simulate connection restored
        monitor.tick.return_value = (
            connection_monitor.ConnectionStatus.connection_restored
        )

        # 5. Wait for replay
        deadline = time.time() + 2.0
        while not replayed and time.time() < deadline:
            time.sleep(0.05)

        rm.close()
        rm.join(timeout=2)

        assert len(replayed) == 1
        assert isinstance(replayed[0], messages.CreateTraceMessage)
        assert replayed[0].trace_id == "trace-1"
        monitor.reset.assert_called()


# --- sdks/python/tests/unit/runner/test_activate.py ---

def test_install_signal_handlers__from_background_thread__does_not_raise():
    shutdown_event = threading.Event()
    errors = []

    def run():
        try:
            activate_module.install_signal_handlers(shutdown_event)
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=run)
    t.start()
    t.join()

    assert not errors
    assert not shutdown_event.is_set()


# --- sdks/python/tests/unit/runner/test_supervisor.py ---

    def test_stops_all(self) -> None:
        sup = _make_supervisor(watch=False)

        t = threading.Thread(target=sup.run, daemon=True)
        t.start()

        time.sleep(1)
        sup._shutdown_event.set()
        t.join(timeout=10)

        assert sup._child is None

    def test_waits_for_child(self) -> None:
        sup = _make_supervisor(watch=False)

        t = threading.Thread(target=sup.run, daemon=True)
        t.start()

        time.sleep(0.5)
        sup._shutdown_event.set()
        t.join(timeout=15)

        assert not t.is_alive()

    def test_bridge_loop_runs(self) -> None:
        sup = _make_supervisor(watch=False, runner_type=RunnerType.CONNECT)

        t = threading.Thread(target=sup.run, daemon=True)
        t.start()

        time.sleep(1)

        bridge_alive = False
        for thread in threading.enumerate():
            if thread.name == "bridge-poll":
                bridge_alive = True
                break

        sup._shutdown_event.set()
        t.join(timeout=10)

        assert bridge_alive


# --- sdks/python/tests/unit/simulation/test_simulated_user.py ---

    def test_generate_response_with_fixed_responses(self):
        """Test response generation using fixed responses."""
        fixed_responses = ["Response 1", "Response 2", "Response 3"]
        user = SimulatedUser(persona="Test persona", fixed_responses=fixed_responses)

        # First call
        response1 = user.generate_response([])
        assert response1 == "Response 1"
        assert user._response_index == 1

        # Second call
        response2 = user.generate_response([])
        assert response2 == "Response 2"
        assert user._response_index == 2

        # Third call
        response3 = user.generate_response([])
        assert response3 == "Response 3"
        assert user._response_index == 3

        # Fourth call (cycles back)
        response4 = user.generate_response([])
        assert response4 == "Response 1"
        assert user._response_index == 4


# --- tests_load/tests/test_image_inference.py ---

def test_openai_image_generation(prompt: str):
    """Test 1: Generate image with DALL-E using OpenAI integration"""
    print("\n" + "=" * 60)
    print("TEST 1: Simple OpenAI DALL-E 3 (images.generate)")
    print("=" * 60)

    if "openai" not in clients:
        print("❌ OpenAI client not available")
        return None, None

    print(f"Generating image with prompt: {prompt}")

    try:
        # Generate image - automatically tracked by Opik
        response = clients["openai"].images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        revised_prompt = response.data[0].revised_prompt

        print(f"✓ Image generated: {image_url}")
        print(f"✓ Revised prompt: {revised_prompt[:100]}...")
        print(f"✓ Logged to Opik project: {PROJECT_NAME}")

        return image_url, revised_prompt
    except Exception as e:
        print(f"❌ OpenAI image generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_openai_gpt_image_generation(prompt: str):
    """Test 2: Generate image using OpenAI gpt-image-1 (Images API)"""
    print("\n" + "=" * 60)
    print("TEST 2: OpenAI Image Generation (gpt-image-1 via Images API)")
    print("=" * 60)

    if "openai" not in clients:
        print("❌ OpenAI client not available")
        return None, None

    print(f"Generating image with prompt: {prompt}")

    try:
        # Use the Images API with gpt-image-1 (quality: low|medium|high|auto)
        img = clients["openai"].images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            quality="low",
            n=1,
        )
        # Try URL first
        url = None
        try:
            url = img.data[0].url
        except Exception:
            url = None
        if not url:
            # Some SDKs return base64 instead
            b64 = getattr(img.data[0], "b64_json", None)
            if b64:
                url = f"data:image/png;base64,{b64}"
        if url:
            print(f"✓ Image generated: {url[:80]}...")
            print(f"✓ Logged to Opik project: {PROJECT_NAME}")
            return url, prompt
        print("⚠️  No URL or base64 returned by Images API for gpt-image-1. Skipping.")
        return None, None
    except Exception as e:
        print(f"❌ OpenAI gpt-image-1 images.generate failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_openrouter_gemini_image_generation(prompt: str):
    """Test X: Generate image using Gemini via OpenRouter"""
    print("\n" + "=" * 60)
    print("TEST 2: Gemini 2.5 Flash Image Generation (via OpenRouter)")
    print("=" * 60)

    if "openrouter" not in clients:
        print("❌ OpenRouter client not available")
        return None, None

    print(f"Generating image with prompt: {prompt}")

    try:
        # Use Gemini 2.5 Flash Image model through OpenRouter
        # Per docs: send to /chat/completions with modalities ["image","text"]
        # https://openrouter.ai/docs/features/multimodal/image-generation
        response = clients["openrouter"].chat.completions.create(
            model="google/gemini-2.5-flash-image-preview",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            modalities=["image", "text"],
            max_tokens=1000
        )

        # Extract image per docs: assistant message includes images list with image_url.url (base64 data URL)
        image_url = None
        try:
            message = response.choices[0].message
        except Exception:
            message = None
        image_url = _extract_image_url(message) or _extract_image_url(response)
        if not image_url:
            # Fallback: regex scan for data URL in stringified response
            try:
                import re
                blob = json.dumps(response, default=str)
                m = re.search(r"data:image\/(?:png|jpeg|jpg);base64,[A-Za-z0-9+\/=]+", blob)
                if m:
                    image_url = m.group(0)
            except Exception:
                pass
        if not image_url:
            raise Exception(
                "No image found in OpenRouter response; ensure model supports image output and modalities were set")

        print(f"✓ Image generated: {image_url[:50]}...")
        print(f"✓ Logged to Opik project: {PROJECT_NAME}")

        return image_url, prompt
    except Exception as e:
        print(f"❌ Gemini image generation via OpenRouter failed: {e}")
        print(f"   This might mean:")
        print(f"   - The model 'google/gemini-2.5-flash-image-preview' isn't available")
        print(f"   - OpenRouter API structure has changed")
        print(f"   - Check OpenRouter documentation for current image generation API")
        import traceback
        traceback.print_exc()
        return None, None

def test_google_gemini_image_generation(prompt: str):
    """Test X: Generate image using Google Gemini via Google ADK / Generative AI"""
    print("\n" + "=" * 60)
    print("TEST 3: Google Gemini (via Google ADK)")
    print("=" * 60)

    if "google" not in clients:
        print("❌ Google Gemini client not available (ADK or Generative AI)")
        return None, None

    print(f"Generating image with prompt: {prompt}")

    try:
        provider = clients.get("google_provider")
        image_url = None
        revised_prompt = prompt
        if provider == "adk":
            # Prefer generating images via Google GenAI even if ADK is present
            try:
                from google import genai  # type: ignore
                genai_key = clients.get("google_api_key") or os.environ.get("GOOGLE_API_KEY") or os.environ.get(
                    "GEMINI_API_KEY")
                genai_client = genai.Client(api_key=genai_key) if genai_key else genai.Client()
                try:
                    from google.genai import types as genai_types  # type: ignore
                except Exception:
                    genai_types = None
                result = genai_client.models.generate_images(
                    model='imagen-3.0-generate-002',
                    prompt=prompt,
                    config=(genai_types.GenerateImagesConfig(
                        number_of_images=1,
                        output_mime_type='image/jpeg',
                    ) if genai_types else dict(number_of_images=1, output_mime_type='image/jpeg'))
                )
                gi = result.generated_images[0]
                img_bytes = gi.image.image_bytes
                if isinstance(img_bytes, (bytes, bytearray)):
                    b64 = base64.b64encode(img_bytes).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{b64}"
                elif hasattr(gi.image, 'uri') and gi.image.uri:
                    image_url = gi.image.uri
            except Exception as adk_genai_e:
                print(f"⚠️  ADK path using Google GenAI failed: {adk_genai_e}")
                # Last resort: call ADK client if it exposes generate_image
                try:
                    if hasattr(clients["google"], "generate_image"):
                        response = clients["google"].generate_image(
                            prompt=prompt,
                            model="gemini-2.0-flash-exp",
                            size="1024x1024"
                        )
                        image_url = (
                                response.get("image_url") or response.get("url") or response.get("data", {}).get("url")
                        )
                        revised_prompt = response.get("revised_prompt", prompt)
                except Exception as adk_direct_e:
                    print(f"⚠️  ADK direct image generation failed: {adk_direct_e}")
        elif provider == "genai":
            # Google GenAI official client: prefer Gemini native image generation (preview)
            # https://ai.google.dev/gemini-api/docs/image-generation
            client_genai = clients["google"]
            try:
                response = client_genai.models.generate_content(
                    model="gemini-2.5-flash-image-preview",
                    contents=[prompt],
                )
                # Extract inline image bytes
                try:
                    parts = response.candidates[0].content.parts
                except Exception:
                    parts = []
                for part in parts:
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data and getattr(inline_data, "data", None):
                        b64 = inline_data.data if isinstance(inline_data.data, str) else base64.b64encode(
                            inline_data.data).decode("utf-8")
                        image_url = f"data:image/png;base64,{b64}"
                        break
                if not image_url:
                    # Fallback to Imagen generate_images
                    try:
                        from google.genai import types as genai_types  # type: ignore
                    except Exception:
                        genai_types = None
                    result = client_genai.models.generate_images(
                        model='imagen-3.0-generate-002',
                        prompt=prompt,
                        config=(genai_types.GenerateImagesConfig(
                            number_of_images=1,
                            output_mime_type='image/jpeg',
                        ) if genai_types else dict(number_of_images=1, output_mime_type='image/jpeg'))
                    )
                    gi = result.generated_images[0]
                    img_bytes = gi.image.image_bytes
                    if isinstance(img_bytes, (bytes, bytearray)):
                        b64 = base64.b64encode(img_bytes).decode('utf-8')
                        image_url = f"data:image/jpeg;base64,{b64}"
                    elif hasattr(gi.image, 'uri') and gi.image.uri:
                        image_url = gi.image.uri
            except Exception as ge:
                print(f"⚠️  Google GenAI generate_content failed: {ge}")
                image_url = None
        else:
            # Legacy google.generativeai path (kept as last-resort)
            result = clients["google"].generate_content([prompt])
            try:
                parts = getattr(result, "candidates", [])[0].content.parts  # type: ignore
            except Exception:
                parts = []
            for p in parts:
                uri = getattr(p, "file_data", None) or getattr(p, "inline_data", None)
                if uri and getattr(uri, "mime_type", "").startswith("image/"):
                    image_url = getattr(uri, "file_uri", None) or getattr(uri, "data", None)
                    break

        if not image_url:
            print("❌ No image URL found in Gemini response")
            return None, None

        print(f"✓ Image generated: {image_url}")
        print(f"✓ Logged to Opik project: {PROJECT_NAME}")

        return image_url, revised_prompt
    except Exception as e:
        print(f"❌ Google Gemini image generation failed: {e}")
        print(f"   This might mean the model isn't available or the API has changed")
        import traceback
        traceback.print_exc()
        return None, None

def test_openai_agents_multimodal():
    """Test X: OpenAI Agents with multimodal function tools"""
    print("\n" + "=" * 60)
    print("TEST 7: OpenAI Agents Multimodal Operations")
    print("=" * 60)

    if "openai_agents" not in clients:
        print("❌ OpenAI Agents not available")
        return None

    try:
        # Create a multimodal agent with image generation and analysis tools
        multimodal_agent = Agent(
            name="MultimodalAssistant",
            instructions="""You are a multimodal AI assistant with access to image generation and analysis tools. 
            You can:
            1. Generate images using DALL-E 3
            2. Analyze images using GPT-4o Vision
            3. Analyze images using Claude Vision
            
            When asked to create or analyze images, use the appropriate tools and provide detailed responses.
            Always explain what you're doing and provide the results clearly.""",
            model="gpt-4o-mini",
            tools=[generate_image_with_dalle, analyze_image_with_vision, analyze_image_with_claude]
        )

        # Test 1: Generate and analyze an image
        print("🤖 Testing image generation and analysis workflow...")

        result = Runner.run_sync(
            multimodal_agent,
            "Generate an image of a futuristic AI laboratory and then analyze it in detail. Use both GPT-4o and Claude for analysis to compare their perspectives."
        )

        print(f"✅ Agent response: {result.final_output[:200]}...")
        print(f"✅ Logged to Opik project: {PROJECT_NAME}")

        return result.final_output

    except Exception as e:
        print(f"❌ OpenAI Agents multimodal test failed: {e}")
        return None

def test_openai_agents_conversation():
    """Test X: OpenAI Agents multi-turn conversation with image context"""
    print("\n" + "=" * 60)
    print("TEST 8: OpenAI Agents Multi-turn Conversation")
    print("=" * 60)

    if "openai_agents" not in clients:
        print("❌ OpenAI Agents not available")
        return None

    try:
        import uuid
        from agents import trace

        # Create a conversational agent
        conversation_agent = Agent(
            name="ConversationalAssistant",
            instructions="You are a helpful assistant that can generate and analyze images. Be conversational and engaging.",
            model="gpt-4o-mini",
            tools=[generate_image_with_dalle, analyze_image_with_vision]
        )

        # Create a conversation thread
        thread_id = str(uuid.uuid4())
        print(f"🧵 Starting conversation thread: {thread_id}")

        with trace(workflow_name="MultimodalConversation", group_id=thread_id):
            # First turn: Generate an image
            print("📝 Turn 1: Generating an image...")
            result1 = Runner.run_sync(
                conversation_agent,
                "Create an image of a beautiful sunset over mountains"
            )
            print(f"🤖 Response 1: {result1.final_output[:150]}...")

            # Extract image URL from the response (this would need parsing in a real scenario)
            # For now, we'll simulate a follow-up question
            print("📝 Turn 2: Asking about the image...")
            result2 = Runner.run_sync(
                conversation_agent,
                "Can you analyze the image you just created and tell me about the colors and mood?"
            )
            print(f"🤖 Response 2: {result2.final_output[:150]}...")

        print(f"✅ Multi-turn conversation completed")
        print(f"✅ Logged to Opik project: {PROJECT_NAME}")

        return {
            "thread_id": thread_id,
            "turn1": result1.final_output,
            "turn2": result2.final_output
        }

    except Exception as e:
        print(f"❌ OpenAI Agents conversation test failed: {e}")
        return None

def test_openai_agents_gpt5_image_generation(prompt: str):
    """Test X: OpenAI Agent SDK using gpt-5 to directly generate an image"""
    print("\n" + "=" * 60)
    print("TEST X: OpenAI Agent SDK (gpt-5 direct image generation)")
    print("=" * 60)

    if "openai_agents" not in clients:
        print("❌ OpenAI Agents not available")
        return None, None

    try:
        agent = Agent(
            name="GPT5ImageAgent",
            instructions=(
                "You can generate images directly. When asked to create an image, "
                "produce the image and include a link or data reference in your response."
            ),
            model="gpt-5",
            tools=[]
        )

        result = Runner.run_sync(agent, f"Generate an image: {prompt}")

        image_url = None
        # Best-effort extraction from potential result structures
        for attr in ("artifacts", "attachments"):
            if hasattr(result, attr):
                items = getattr(result, attr) or []
                try:
                    for it in items:
                        if isinstance(it, dict):
                            image_url = it.get("image_url") or it.get("url")
                            if image_url:
                                break
                        else:
                            iu = getattr(it, "image_url", None) or getattr(it, "url", None)
                            if iu:
                                image_url = iu
                                break
                except Exception:
                    pass

        # Fallback: try to find a URL in final_output text
        if not image_url and hasattr(result, "final_output") and isinstance(result.final_output, str):
            import re
            m = re.search(r"https?://\S+", result.final_output)
            if m:
                image_url = m.group(0)

        if image_url:
            print(f"✓ Agent generated image: {image_url[:80]}...")
        else:
            print("⚠️  Agent response did not contain a direct image URL; see Opik trace for details")
            if hasattr(result, "final_output"):
                print(f"📝 Agent output (truncated): {str(result.final_output)[:200]}...")

        print(f"✓ Logged to Opik project: {PROJECT_NAME}")
        return image_url, prompt
    except Exception as e:
        print(f"❌ OpenAI Agent gpt-5 image generation failed: {e}")
        return None, None

