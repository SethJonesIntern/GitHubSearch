# comet-ml/opik
# 94 semantic-evaluator test functions
# Source extract: c:/Users/Seth/Documents/papers/GitHubSearch/Applications/extracted_llm_tests/comet-ml_opik.py

# --- sdks/opik_optimizer/tests/e2e/optimizers/multimodal/test_multimodal_prompt.py  [opik] ---

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


# --- sdks/python/tests/e2e/test_agent_config.py  [opik] ---

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


# --- sdks/python/tests/e2e/test_attachments_client.py  [opik] ---

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


# --- sdks/python/tests/e2e/test_attachments_extraction.py  [opik] ---

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


# --- sdks/python/tests/e2e/test_dataset.py  [opik] ---

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


# --- sdks/python/tests/e2e/test_failed_messages_replay.py  [opik] ---

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


# --- sdks/python/tests/e2e/test_local_recording.py  [opik] ---

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


# --- sdks/python/tests/e2e/test_optimization.py  [opik] ---

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


# --- sdks/python/tests/e2e/test_prompt.py  [opik] ---

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


# --- sdks/python/tests/e2e/test_tracing.py  [opik] ---

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


# --- sdks/python/tests/e2e/compatibility_v1/test_dataset.py  [opik] ---

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


# --- sdks/python/tests/e2e/compatibility_v1/test_optimization.py  [opik] ---

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


# --- sdks/python/tests/e2e/compatibility_v1/test_prompt.py  [opik] ---

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


# --- sdks/python/tests/e2e/compatibility_v1/evaluation/test_evaluate_filter_string.py  [opik] ---

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


# --- sdks/python/tests/e2e/compatibility_v1/evaluation/test_experiment_evaluate.py  [opik] ---

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


# --- sdks/python/tests/e2e/evaluation/test_evaluate_filter_string.py  [opik] ---

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


# --- sdks/python/tests/e2e/evaluation/test_experiment_evaluate.py  [opik] ---

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


# --- sdks/python/tests/e2e/evaluation/test_test_suite.py  [opik] ---

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


# --- sdks/python/tests/e2e/runner/test_runner_e2e.py  [opik] ---

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


# --- sdks/python/tests/e2e_library_integration/litellm/test_opik_logging.py  [opik] ---

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


# --- sdks/python/tests/library_integration/langchain/test_langchain_openai.py  [opik] ---

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


# --- sdks/python/tests/unit/evaluation/metrics/test_base_metric.py  [opik] ---

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


# --- sdks/python/tests/unit/llm_usage/test_opik_usage_factory.py  [opik] ---

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


# --- tests_load/tests/test_image_inference.py  [opik] ---

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

