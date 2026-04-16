# langchain-ai/langchain-google
# 57 LLM-backed test functions across 84 test files
# Source: https://github.com/langchain-ai/langchain-google

# --- libs/vertexai/tests/integration_tests/test_chat_models.py ---

def test_init_from_credentials_obj() -> None:
    credentials_dict = json.loads(os.environ["GOOGLE_VERTEX_AI_WEB_CREDENTIALS"])
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict
    )
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME, credentials=credentials)
    llm.invoke("how are you")

def test_vertexai_single_call(model_name: str | None, endpoint_version: str) -> None:
    """Test making a single invoke call."""
    model = ChatVertexAI(
        model=model_name,
        rate_limiter=RATE_LIMITER,
        endpoint_version=endpoint_version,
    )
    message = HumanMessage(content="Hello")
    response = model.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    _check_usage_metadata(response)

def test_candidates() -> None:
    """Test making a single invoke call with `n>1`.

    # TODO: what is chat-bison@001? is it marked for deprecation?
    """
    model = ChatVertexAI(
        model="chat-bison@001", temperature=0.3, n=2, rate_limiter=RATE_LIMITER
    )
    message = HumanMessage(content="Hello")
    response = model.generate(messages=[[message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    assert len(response.generations[0]) == 2

async def test_vertexai_generate() -> None:
    # TODO: parameterize with sync/async generate method
    model = ChatVertexAI(
        temperature=0, model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER
    )
    message = HumanMessage(content="Hello")
    response = await model.agenerate([[message]])
    assert isinstance(response, LLMResult)
    async_generation = cast("ChatGeneration", response.generations[0][0])
    output_message = async_generation.message
    assert isinstance(output_message, AIMessage)
    _check_usage_metadata(output_message)

    sync_response = model.generate([[message]])
    sync_generation = cast("ChatGeneration", sync_response.generations[0][0])

    usage_metadata = sync_generation.generation_info["usage_metadata"]  # type: ignore
    assert int(usage_metadata["prompt_token_count"]) > 0
    assert int(usage_metadata["candidates_token_count"]) > 0
    usage_metadata = async_generation.generation_info["usage_metadata"]  # type: ignore
    assert int(usage_metadata["prompt_token_count"]) > 0
    assert int(usage_metadata["candidates_token_count"]) > 0

def test_vertexai_stream() -> None:
    # TODO: parameterize with astream equivalent
    model = ChatVertexAI(
        temperature=0, model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER
    )
    message = HumanMessage(content="Hello")

    sync_response = model.stream([message])
    full: BaseMessageChunk | None = None
    chunks_with_usage_metadata = 0
    chunks_with_model_name = 0
    for chunk in sync_response:
        assert isinstance(chunk, AIMessageChunk)
        if chunk.usage_metadata:
            chunks_with_usage_metadata += 1
        if chunk.response_metadata.get("model_name"):
            chunks_with_model_name += 1
        full = chunk if full is None else full + chunk
    if chunks_with_usage_metadata == 0 or chunks_with_model_name != 1:
        pytest.fail(
            "Expected >=1 chunk with usage metadata and exactly 1 with model_name."
        )
    assert isinstance(full, AIMessageChunk)
    _check_usage_metadata(full)
    assert full.response_metadata["model_name"] == _DEFAULT_MODEL_NAME

async def test_vertexai_astream() -> None:
    # TODO: parameterize with stream equivalent
    model = ChatVertexAI(
        temperature=0, model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER
    )
    message = HumanMessage(content="Hello")

    full: BaseMessageChunk | None = None
    chunks_with_usage_metadata = 0
    chunks_with_model_name = 0
    async for chunk in model.astream([message]):
        assert isinstance(chunk, AIMessageChunk)
        if chunk.usage_metadata:
            chunks_with_usage_metadata += 1
        if chunk.response_metadata.get("model_name"):
            chunks_with_model_name += 1
        full = chunk if full is None else full + chunk
    if chunks_with_usage_metadata == 0 or chunks_with_model_name != 1:
        pytest.fail(
            "Expected >=1 chunk with usage metadata and exactly 1 with model_name."
        )
    assert isinstance(full, AIMessageChunk)
    _check_usage_metadata(full)
    assert full.response_metadata["model_name"] == _DEFAULT_MODEL_NAME

def test_multimodal() -> None:
    """Test multimodal input with a gcs image URL in chat completions format."""
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER)
    gcs_url = (
        "gs://cloud-samples-data/generative-ai/image/320px-Felis_catus-cat_on_snow.jpg"
    )
    image_message = {
        "type": "image_url",
        "image_url": {"url": gcs_url},
    }
    text_message = {
        "type": "text",
        "text": "What is shown in this image?",
    }
    message = HumanMessage(content=[text_message, image_message])
    output = llm.invoke([message])
    assert isinstance(output.content, str)
    assert isinstance(output, AIMessage)
    _check_usage_metadata(output)

    llm = ChatVertexAI(model="gemini-2.5-pro", rate_limiter=RATE_LIMITER)
    for chunk in llm.stream([message]):
        assert isinstance(chunk, AIMessageChunk)

def test_multimodal_media_file_uri(file_uri, mime_type) -> None:
    """Test multimodal input with gcs file URIs (video, audio, image)."""
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER)
    media_message = {
        "type": "media",
        "file_uri": file_uri,
        "mime_type": mime_type,
    }
    text_message = {
        "type": "text",
        "text": "Describe the attached media in 5 words!",
    }
    message = HumanMessage(content=[text_message, media_message])
    output = llm.invoke([message])
    assert isinstance(output.content, str)

def test_multimodal_media_inline_base64(file_uri, mime_type) -> None:
    """Test multimodal input with base64 encoded media content (video, audio, image)."""
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER)
    storage_client = storage.Client()
    blob = storage.Blob.from_string(file_uri, client=storage_client)
    media_base64 = base64.b64encode(blob.download_as_bytes()).decode()
    media_message = {
        "type": "media",
        "data": media_base64,
        "mime_type": mime_type,
    }
    text_message = {
        "type": "text",
        "text": "Describe the attached media in 5 words!",
    }
    message = HumanMessage(content=[text_message, media_message])
    output = llm.invoke([message])
    assert isinstance(output.content, str)

def test_multimodal_media_inline_base64_template() -> None:
    """Test multimodal input with base64 encoded media content using prompt template."""
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME)
    prompt_template = ChatPromptTemplate(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "media",
                        "data": "{media_base64}",
                        "mime_type": "{mime_type}",
                    },
                    {
                        "type": "text",
                        "text": "Describe the attached media in 5 words!",
                    },
                ],
            },
        ]
    )
    storage_client = storage.Client()
    file_uri = (
        "gs://cloud-samples-data/generative-ai/audio/audio_summary_clean_energy.mp3"
    )
    mime_type = "audio/mp3"
    blob = storage.Blob.from_string(file_uri, client=storage_client)
    media_base64 = base64.b64encode(blob.download_as_bytes()).decode()
    chain = prompt_template | llm
    output = chain.invoke({"media_base64": media_base64, "mime_type": mime_type})
    assert isinstance(output.content, str)

def test_audio_timestamp() -> None:
    storage_client = storage.Client()
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER)

    file_uri = "gs://cloud-samples-data/generative-ai/audio/pixel.mp3"
    mime_type = "audio/mp3"
    blob = storage.Blob.from_string(file_uri, client=storage_client)
    media_base64 = base64.b64encode(blob.download_as_bytes()).decode()
    media_message = {
        "type": "media",
        "data": media_base64,
        "mime_type": mime_type,
    }
    instruction = """
    Transcribe the video.
    """
    text_message = {"type": "text", "text": instruction}

    message = HumanMessage(content=[media_message, text_message])
    output = llm.invoke([message], audio_timestamp=True)

    assert isinstance(output.content, str)
    assert re.search(r"(\d{2}:\d{2}:?|\[\d{2}:\d{2}:\d{2}\])", output.content)

def test_multimodal_video_metadata(file_uri, mime_type) -> None:
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER)
    media_message = {
        "type": "media",
        "file_uri": file_uri,
        "mime_type": mime_type,
        "video_metadata": {
            "start_offset": {"seconds": 22, "nanos": 5000},
            "end_offset": {"seconds": 25, "nanos": 5000},
        },
    }
    text_message = {
        "type": "text",
        "text": "What is shown in the subtitles",
    }

    message = HumanMessage(content=[text_message, media_message])
    output = llm.invoke([message])
    assert isinstance(output.content, str)

def test_vertexai_single_call_with_history(model_name: str | None) -> None:
    model = ChatVertexAI(model=model_name, rate_limiter=RATE_LIMITER)
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    response = model.invoke([message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_vertexai_system_message() -> None:
    model = ChatVertexAI(model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER)
    system_instruction = """CymbalBank is a bank located in London"""
    text_question1 = "Where is Cymbal located? Provide only the name of the city."
    sys_message = SystemMessage(content=system_instruction)
    message1 = HumanMessage(content=text_question1)
    response = model.invoke([sys_message, message1])

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "london" in response.content.lower()

def test_vertexai_single_call_with_no_system_messages() -> None:
    model = ChatVertexAI(model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER)
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    response = model.invoke([message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_vertexai_single_call_previous_blocked_response() -> None:
    """If a previous call was blocked, the AIMessage will have empty content.

    Empty content should be ignored.
    """
    model = ChatVertexAI(model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER)
    text_question2 = "How much is 3+3?"
    # Previous blocked response included in history. This can happen with a LangGraph
    # ReAct agent.
    message1 = AIMessage(
        content="",
        response_metadata={
            "is_blocked": True,
            "safety_ratings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "probability_label": "MEDIUM",
                    "probability_score": 0.33039191365242004,
                    "blocked": True,
                    "severity": "HARM_SEVERITY_MEDIUM",
                    "severity_score": 0.2782268822193146,
                },
            ],
            "finish_reason": "SAFETY",
        },
    )
    message2 = HumanMessage(content=text_question2)
    response = model.invoke([message1, message2])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_vertexai_gemini_function_calling(endpoint_version: str) -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    safety = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    # Test .bind_tools with BaseModel
    message = HumanMessage(content="My name is Erick and I am 27 years old")
    model = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        safety_settings=safety,
        rate_limiter=RATE_LIMITER,
        endpoint_version=endpoint_version,
    ).bind_tools([MyModel])
    response = model.invoke([message])
    _check_tool_calls(response, "MyModel")

    # Test .bind_tools with function
    def my_model(name: str, age: int) -> None:
        """Invoke this with names and ages."""

    model = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        safety_settings=safety,
        rate_limiter=RATE_LIMITER,
    ).bind_tools([my_model])
    response = model.invoke([message])
    _check_tool_calls(response, "my_model")

    # Test .bind_tools with tool
    @tool
    def my_tool(name: str, age: int) -> None:
        """Invoke this with names and ages."""

    model = ChatVertexAI(model=_DEFAULT_MODEL_NAME, safety_settings=safety).bind_tools(
        [my_tool]
    )
    response = model.invoke([message])
    _check_tool_calls(response, "my_tool")

    # Test streaming
    stream = model.stream([message])
    first = True
    for chunk in stream:
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk  # type: ignore
    assert isinstance(gathered, AIMessageChunk)
    assert len(gathered.tool_call_chunks) == 1
    tool_call_chunk = gathered.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "my_tool"
    assert tool_call_chunk["args"]
    assert json.loads(tool_call_chunk["args"]) == {"age": 27.0, "name": "Erick"}

def test_chat_vertexai_gemini_function_calling_tool_config_any() -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    safety = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    model = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        safety_settings=safety,
        rate_limiter=RATE_LIMITER,
    ).bind(
        functions=[MyModel],
        tool_config={
            "function_calling_config": {
                "mode": FunctionCallingConfig.Mode.ANY,
                "allowed_function_names": ["MyModel"],
            }
        },
    )
    message = HumanMessage(content="My name is Erick and I am 27 years old")
    response = model.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == ""
    function_call = response.additional_kwargs.get("function_call")
    assert function_call
    assert function_call["name"] == "MyModel"
    arguments_str = function_call.get("arguments")
    assert arguments_str
    arguments = json.loads(arguments_str)
    assert arguments == {
        "name": "Erick",
        "age": 27.0,
    }

def test_chat_vertexai_gemini_function_calling_tool_config_none() -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    safety = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    model = ChatVertexAI(model=_DEFAULT_MODEL_NAME, safety_settings=safety).bind(
        functions=[MyModel],
        tool_config={
            "function_calling_config": {
                "mode": FunctionCallingConfig.Mode.NONE,
            }
        },
    )
    message = HumanMessage(content="My name is Erick and I am 27 years old")
    response = model.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content != ""
    function_call = response.additional_kwargs.get("function_call")
    assert function_call is None

def test_chat_model_multiple_system_message() -> None:
    model = ChatVertexAI(model=_DEFAULT_MODEL_NAME)
    response = model.invoke(
        [
            SystemMessage("Be helpful"),
            AIMessage("Hi, I'm LeoAI. How can I help?"),
            SystemMessage("Your name is LeoAI"),
        ]
    )
    assert isinstance(response, AIMessage)

def test_chat_vertexai_gemini_with_structured_output(
    method: Literal["json_mode"] | None,
) -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    safety = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        safety_settings=safety,
        rate_limiter=RATE_LIMITER,
    )
    model = llm.with_structured_output(MyModel, method=method)
    message = HumanMessage(content="My name is Erick and I am 27 years old")

    response = model.invoke([message])
    assert isinstance(response, MyModel)
    assert response == MyModel(name="Erick", age=27)

    if method is None:  # This won't work with json_schema as it expects an OpenAPI dict
        model = llm.with_structured_output(
            {
                "name": "MyModel",
                "description": "MyModel",
                "parameters": MyModel.model_json_schema(),
            },
            method=method,
        )
        response = model.invoke([message])
        assert response == {
            "name": "Erick",
            "age": 27,
        }

    model = llm.with_structured_output(
        {
            "title": "MyModel",
            "description": "MyModel",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        },
        method=method,
    )
    response = model.invoke([message])
    assert response == {
        "name": "Erick",
        "age": 27,
    }

def test_chat_vertexai_gemini_with_structured_output_nested_model() -> None:
    class Argument(BaseModel):
        description: str

    class Reason(BaseModel):
        strength: int
        argument: list[Argument]

    class Response(BaseModel):
        response: str
        reasons: list[Reason]

    model = ChatVertexAI(model=_DEFAULT_MODEL_NAME).with_structured_output(
        Response, method="json_mode"
    )

    response = model.invoke("Why is Real Madrid better than Barcelona?")

    assert isinstance(response, Response)

def test_chat_vertexai_gemini_function_calling_with_multiple_parts() -> None:
    @tool
    def search(
        question: str,
    ) -> str:
        """Useful for when you need to answer questions or visit websites.
        You should ask targeted questions.
        """
        return "brown"

    tools = [search]

    safety = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        safety_settings=safety,
        temperature=0,
        rate_limiter=RATE_LIMITER,
        endpoint_version="v1",
        location="global",
    )
    llm_with_search = llm.bind(
        functions=tools,
    )
    llm_with_search_force = llm_with_search.bind(
        tool_config={
            "function_calling_config": {
                "mode": FunctionCallingConfig.Mode.ANY,
                "allowed_function_names": ["search"],
            }
        },
    )
    request = HumanMessage(
        content="Please tell the primary color of following birds: sparrow, hawk, crow",
    )
    response = llm_with_search_force.invoke([request])

    assert isinstance(response, AIMessage)
    tool_calls = response.tool_calls
    assert len(tool_calls) == 3

    tool_response = search.invoke({"question": "sparrow"})
    tool_messages: list[BaseMessage] = []

    for tool_call in tool_calls:
        assert tool_call["name"] == "search"
        tool_message = ToolMessage(
            name=tool_call["name"],
            content=json.dumps(tool_response),
            tool_call_id=(tool_call["id"] or ""),
        )
        tool_messages.append(tool_message)

    result = llm_with_search.invoke([request, response, *tool_messages])

    assert isinstance(result, AIMessage)
    assert "brown" in result.content[0]["text"]  # type: ignore
    assert len(result.tool_calls) == 0

def test_chat_vertexai_gemini_image_output() -> None:
    model = ChatVertexAI(
        model=_DEFAULT_IMAGE_GENERATION_MODEL_NAME,
        response_modalities=[Modality.TEXT, Modality.IMAGE],
    )
    result = model.invoke("Generate an image of a cat. Then, say meow!")

    assert isinstance(result, AIMessage)
    assert isinstance(result.content, list)

    image_element = None
    for item in result.content:
        if isinstance(item, dict) and item.get("type") == "image_url":
            image_element = item
            break
    assert image_element is not None, "Did not find the expected image content"

    text_element = None
    for item in result.content:
        if isinstance(item, str):
            text_element = item
            break
    assert text_element is not None, "Did not find the expected text content"

def test_chat_vertexai_gemini_image_output_with_generation_config() -> None:
    model = ChatVertexAI(model=_DEFAULT_IMAGE_GENERATION_MODEL_NAME)
    result = model.invoke(
        "Generate an image of a cat. Then, say meow!",
        response_modalities=[Modality.TEXT, Modality.IMAGE],
    )

    assert isinstance(result, AIMessage)
    assert isinstance(result.content, list)

    image_element = None
    for item in result.content:
        if isinstance(item, dict) and item.get("type") == "image_url":
            image_element = item
            break
    assert image_element is not None, "Did not find the expected image content"

    text_element = None
    for item in result.content:
        if isinstance(item, str):
            text_element = item
            break
        elif isinstance(item, dict) and item.get("type") == "text":
            text_element = item.get("text")
            break
    assert text_element is not None, "Did not find the expected text content"

def test_chat_vertexai_gemini_thinking_auto() -> None:
    model = ChatVertexAI(model=_DEFAULT_THINKING_MODEL_NAME)
    response = model.invoke("How many O's are in Google? Think before you answer.")
    assert isinstance(response, AIMessage)
    assert response.usage_metadata is not None
    assert response.usage_metadata["output_token_details"]["reasoning"] > 0
    assert (
        response.usage_metadata["total_tokens"]
        > response.usage_metadata["input_tokens"]
        + response.usage_metadata["output_tokens"]
    )

def test_chat_vertexai_gemini_thinking_configured() -> None:
    model = ChatVertexAI(model=_DEFAULT_THINKING_MODEL_NAME, thinking_budget=100)
    response = model.invoke("How many O's are in Google? Think before you answer.")
    assert isinstance(response, AIMessage)
    assert response.usage_metadata is not None
    assert response.usage_metadata["output_token_details"]["reasoning"] > 0
    assert response.usage_metadata["output_token_details"]["reasoning"] <= 100
    assert (
        response.usage_metadata["total_tokens"]
        > response.usage_metadata["input_tokens"]
        + response.usage_metadata["output_tokens"]
    )

def test_chat_vertexai_gemini_thinking_auto_include_thoughts(
    output_version: str,
) -> None:
    model = ChatVertexAI(
        model=_DEFAULT_THINKING_MODEL_NAME,
        include_thoughts=True,
        output_version=output_version,
    )

    input_message = {
        "role": "user",
        "content": "How many O's are in Google? Think before you answer.",
    }

    full: AIMessageChunk | None = None
    for chunk in model.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)

    _check_thinking_output(cast("list", full.content), output_version)

    assert full.usage_metadata is not None
    assert full.usage_metadata["output_token_details"]["reasoning"] > 0
    assert (
        full.usage_metadata["total_tokens"]
        > full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
    )

    # Test we can pass back in
    next_message = {"role": "user", "content": "Thanks!"}
    _ = model.invoke([input_message, full, next_message])

def test_thought_signatures() -> None:
    """Test Gemini thought signatures.

    Verifies that thought signature byte blobs flow correctly through the entire Gemini
    to GAPIC to LangChain parsing and back into subsequent calls, without crashing or
    losing type safety.
    """
    llm = ChatVertexAI(model="gemini-2.5-pro", include_thoughts=True)

    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm_with_tools = llm.bind_tools([get_weather])

    input_message = {
        "role": "user",
        # TODO: a query that does not generate tool calls (e.g., "Hello") will generate
        # thought signatures on text message blocks. Support this when migrating to
        # standard outputs.
        "content": "What's the weather in London?",
    }

    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)

    next_message = {"role": "user", "content": "Thanks!"}
    _ = llm_with_tools.invoke([input_message, full, next_message])

def test_chat_vertexai_gemini_thinking_disabled() -> None:
    model = ChatVertexAI(model=_DEFAULT_THINKING_MODEL_NAME, thinking_budget=0)
    response = model.invoke("How many O's are in Google?")
    assert isinstance(response, AIMessage)
    assert (
        response.usage_metadata["total_tokens"]  # type: ignore
        == response.usage_metadata["input_tokens"]  # type: ignore
        + response.usage_metadata["output_tokens"]  # type: ignore
    )
    assert "output_token_details" not in response.usage_metadata  # type: ignore

def test_chat_vertexai_gemini_thinking_configurable() -> None:
    model = ChatVertexAI(model=_DEFAULT_THINKING_MODEL_NAME)
    configurable_model = model.configurable_fields(
        thinking_budget=ConfigurableField(id="thinking_budget")
    )
    response = configurable_model.invoke(
        "How many O's are in Google?", {"configurable": {"thinking_budget": 0}}
    )
    assert isinstance(response, AIMessage)
    assert response.usage_metadata is not None
    assert (
        response.usage_metadata["total_tokens"]
        == response.usage_metadata["input_tokens"]
        + response.usage_metadata["output_tokens"]
    )
    assert "output_token_details" not in response.usage_metadata

def test_prediction_client_transport() -> None:
    model = ChatVertexAI(model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER)

    assert model.prediction_client.transport.kind == "grpc"
    assert model.async_prediction_client.transport.kind == "grpc_asyncio"

    model = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER, api_transport="rest"
    )

    assert model.prediction_client.transport.kind == "rest"
    assert model.async_prediction_client.transport.kind == "grpc_asyncio"

    vertexai.init(api_transport="grpc")  # Reset global config to "grpc"

def test_structured_output_schema_json() -> None:
    model = ChatVertexAI(
        rate_limiter=RATE_LIMITER,
        model=_DEFAULT_MODEL_NAME,
        response_mime_type="application/json",
        response_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                    },
                },
                "required": ["recipe_name"],
            },
        },
    )

    response = model.invoke("List a few popular cookie recipes")

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    parsed_response = json.loads(response.content)
    assert isinstance(parsed_response, list)
    assert len(parsed_response) > 0
    assert "recipe_name" in parsed_response[0]

    model = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        response_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                    },
                },
                "required": ["recipe_name"],
            },
        },
        rate_limiter=RATE_LIMITER,
    )
    with pytest.raises(ValueError, match="response_mime_type"):
        response = model.invoke("List a few popular cookie recipes")

def test_json_mode_typeddict() -> None:
    class MyModel(TypedDict):
        name: str
        age: int

    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        rate_limiter=RATE_LIMITER,
    )
    model = llm.with_structured_output(MyModel, method="json_mode")
    message = HumanMessage(content="My name is Erick and I am 28 years old")

    response = model.invoke([message])
    assert isinstance(response, dict)
    assert response == {"name": "Erick", "age": 28}

    # Test stream
    last_non_empty: dict[str, object] | None = None
    for chunk in model.stream([message]):
        assert isinstance(chunk, dict)
        assert all(key in ["name", "age"] for key in chunk)
        if chunk:
            last_non_empty = chunk
    assert last_non_empty == {"name": "Erick", "age": 28}

def test_structured_output_schema_enum() -> None:
    model = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        response_schema={"type": "STRING", "enum": ["drama", "comedy", "documentary"]},
        response_mime_type="text/x.enum",
        rate_limiter=RATE_LIMITER,
    )

    response = model.invoke(
        """
        The film aims to educate and inform viewers about real-life subjects, events, or
        people. It offers a factual record of a particular topic by combining interviews
        , historical footage and narration. The primary purpose of a film is to present
        information and provide insights into various aspects of reality.
        """
    )

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

    assert response.content in ("drama", "comedy", "documentary")

def test_context_catching() -> None:
    system_instruction = """

    You are an expert researcher. You always stick to the facts in the sources provided,
    and never make up new facts.

    If asked about it, the secret number is 747.

    Now look at these research papers, and answer the following questions.

    """

    cached_content = create_context_cache(
        ChatVertexAI(
            model=_DEFAULT_MODEL_NAME,
            rate_limiter=RATE_LIMITER,
        ),
        messages=[
            SystemMessage(content=system_instruction),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
                        },
                    },
                ]
            ),
        ],
    )

    # Using cached_content in constructor
    chat = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        cached_content=cached_content,
        rate_limiter=RATE_LIMITER,
    )

    response = chat.invoke("What is the secret number?")

    assert isinstance(response, AIMessage)
    if isinstance(response.content, str):
        content_text = response.content
    else:
        content_text = " ".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    assert isinstance(content_text, str)

    # Using cached content in request
    chat = ChatVertexAI(model=_DEFAULT_MODEL_NAME, rate_limiter=RATE_LIMITER)
    response = chat.invoke("What is the secret number?", cached_content=cached_content)

    assert isinstance(response, AIMessage)
    if isinstance(response.content, str):
        content_text = response.content
    else:
        content_text = " ".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    assert isinstance(content_text, str)

def test_context_catching_tools() -> None:
    from langchain import agents

    @tool
    def get_secret_number() -> int:
        """Gets secret number."""
        return 747

    tools = [get_secret_number]
    system_instruction = """
    You are an expert researcher. You always stick to the facts in the sources
    provided, and never make up new facts.

    You have a get_secret_number function available. Use this tool if someone asks
    for the secret number.
    Now look at these research papers, and answer the following questions.

    """

    cached_content = create_context_cache(
        model=ChatVertexAI(
            model=_DEFAULT_MODEL_NAME,
        ),
        messages=[
            SystemMessage(content=system_instruction),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
                        },
                    },
                ]
            ),
        ],
        tools=tools,
    )

    chat = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        cached_content=cached_content,
    )

    agent: CompiledStateGraph[Any, Any] = agents.create_agent(
        model=chat,
        tools=tools,
    )
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the secret number?"}]}
    )
    assert "messages" in response
    assert len(response["messages"]) > 0
    assert isinstance(response["messages"][-1], AIMessage)

def test_json_serializable() -> None:
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
    )
    # Needed to init self.client and self.async_client
    llm.prediction_client
    llm.async_prediction_client
    json.loads(llm.model_dump_json())

def test_langgraph_example() -> None:
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        max_output_tokens=8192,
        temperature=0.2,
    )

    add_declaration = {
        "name": "add",
        "description": "Adds a and b.",
        "parameters": {
            "properties": {
                "a": {"description": "first int", "type": "integer"},
                "b": {"description": "second int", "type": "integer"},
            },
            "required": ["a", "b"],
            "type": "object",
        },
    }

    multiply_declaration = {
        "name": "multiply",
        "description": "Multiply a and b.",
        "parameters": {
            "properties": {
                "a": {"description": "first int", "type": "integer"},
                "b": {"description": "second int", "type": "integer"},
            },
            "required": ["a", "b"],
            "type": "object",
        },
    }

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant tasked with performing "
                "arithmetic on a set of inputs."
            )
        ),
        HumanMessage(content="Multiply 2 and 3"),
        HumanMessage(content="No, actually multiply 3 and 3!"),
    ]
    step1 = llm.invoke(
        messages,
        tools=[{"function_declarations": [add_declaration, multiply_declaration]}],
    )
    step2 = llm.invoke(
        [
            *messages,
            step1,
            ToolMessage(content="9", tool_call_id=step1.tool_calls[0]["id"]),
        ],
        tools=[{"function_declarations": [add_declaration, multiply_declaration]}],
    )
    assert isinstance(step2, AIMessage)

async def test_astream_events_langgraph_example() -> None:
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        max_output_tokens=8192,
        temperature=0.2,
    )

    add_declaration = {
        "name": "add",
        "description": "Adds a and b.",
        "parameters": {
            "properties": {
                "a": {"description": "first int", "type": "integer"},
                "b": {"description": "second int", "type": "integer"},
            },
            "required": ["a", "b"],
            "type": "object",
        },
    }

    multiply_declaration = {
        "name": "multiply",
        "description": "Multiply a and b.",
        "parameters": {
            "properties": {
                "a": {"description": "first int", "type": "integer"},
                "b": {"description": "second int", "type": "integer"},
            },
            "required": ["a", "b"],
            "type": "object",
        },
    }

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant tasked with performing "
                "arithmetic on a set of inputs."
            )
        ),
        HumanMessage(content="Multiply 2 and 3"),
        HumanMessage(content="No, actually multiply 3 and 3!"),
    ]
    agenerator = llm.astream_events(
        messages,
        tools=[{"function_declarations": [add_declaration, multiply_declaration]}],
        version="v2",
    )
    events = [events async for events in agenerator]
    assert len(events) > 0
    # Check the function call in the output
    output = events[-1]["data"]["output"]
    assert output.additional_kwargs["function_call"]["name"] == "multiply"

def test_label_metadata() -> None:
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        labels={
            "task": "labels_using_declaration",
            "environment": "testing",
        },
    )
    llm.invoke("hey! how are you")

def test_label_metadata_invoke_method() -> None:
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME)
    llm.invoke(
        "hello! invoke method",
        labels={
            "task": "labels_using_invoke",
            "environment": "testing",
        },
    )

def test_multimodal_pdf_input_gcs(multimodal_pdf_chain: RunnableSerializable) -> None:
    # TODO: parallelize with url and b64 tests
    gcs_uri = "gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf"
    # GCS URI
    response = multimodal_pdf_chain.invoke({"image": gcs_uri})
    assert isinstance(response, AIMessage)

def test_multimodal_pdf_input_url(multimodal_pdf_chain: RunnableSerializable) -> None:
    # TODO: parallelize with gcs and b64 tests
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    # URL
    response = multimodal_pdf_chain.invoke({"image": url})
    assert isinstance(response, AIMessage)

def test_multimodal_pdf_input_b64(multimodal_pdf_chain: RunnableSerializable) -> None:
    # TODO: parallelize with gcs and url tests
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    request_response = requests.get(url, allow_redirects=True)
    # B64
    with io.BytesIO() as stream:
        stream.write(request_response.content)
        image_data = base64.b64encode(stream.getbuffer()).decode("utf-8")
        image = f"data:application/pdf;base64,{image_data}"
        response = multimodal_pdf_chain.invoke({"image": image})
        assert isinstance(response, AIMessage)

def test_response_metadata_avg_logprobs() -> None:
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME)
    response = llm.invoke("Hello!")
    probs = response.response_metadata.get("avg_logprobs")
    if probs is not None:
        assert isinstance(probs, float)

def test_logprobs() -> None:
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME, logprobs=2)
    msg = llm.invoke("hey")
    tokenprobs = msg.response_metadata.get("logprobs_result")
    assert tokenprobs is None or isinstance(tokenprobs, list)
    if tokenprobs:
        stack = tokenprobs[:]
        while stack:
            token = stack.pop()
            assert isinstance(token, dict)
            assert "token" in token
            assert "logprob" in token
            assert isinstance(token.get("token"), str)
            assert isinstance(token.get("logprob"), float)
            if "top_logprobs" in token and token.get("top_logprobs") is not None:
                assert isinstance(token.get("top_logprobs"), list)
                stack.extend(token.get("top_logprobs", []))

    llm2 = ChatVertexAI(model=_DEFAULT_MODEL_NAME, logprobs=True)
    msg2 = llm2.invoke("how are you")
    assert msg2.response_metadata["logprobs_result"]

    llm3 = ChatVertexAI(model=_DEFAULT_MODEL_NAME, logprobs=False)
    msg3 = llm3.invoke("howdy")
    assert msg3.response_metadata.get("logprobs_result") is None

def test_logprobs_with_json_schema() -> None:
    """Ensure logprobs are populated when using JSON schema responses.

    This exercises the same logprobs path as `test_logprobs`, but with
    `response_mime_type='application/json'` and `response_schema` set, which
    previously exposed missing tokens in `logprobs_result` (issue #34133).

    The fix ensures:
    1. Zero logprobs (prob=1.0, 100% certainty) are included, not filtered
    2. All logprob values are valid (non-positive, non-NaN)
    """

    output_schema = {
        "title": "Test Schema",
        "type": "object",
        "properties": {
            "fieldA": {"type": "string"},
            "fieldB": {"type": "number"},
        },
        "required": ["fieldA", "fieldB"],
    }

    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        response_mime_type="application/json",
        response_schema=output_schema,
        logprobs=True,
    )

    msg = llm.invoke("Return a JSON object with fieldA='test' and fieldB=42")
    tokenprobs = msg.response_metadata.get("logprobs_result")
    # We don't assert exact content to avoid flakiness, but if present it must
    # be a well-formed list of token/logprob dicts, including zero logprobs.
    assert tokenprobs is None or isinstance(tokenprobs, list)
    if tokenprobs:
        logprob_values = []
        for token in tokenprobs:
            assert isinstance(token, dict)
            assert "token" in token
            assert "logprob" in token
            assert isinstance(token.get("token"), str)
            assert isinstance(token.get("logprob"), (float, int))
            logprob_values.append(token["logprob"])

        # Verify all logprobs are valid: non-positive (zero allowed) and not NaN
        # This validates the fix for issue #34133 where zero logprobs were
        # incorrectly filtered out

        for val in logprob_values:
            assert not math.isnan(val), "logprob should not be NaN"
            assert val <= 0, f"logprob should be <= 0, got {val}"

        # If we have logprobs, we should have at least some tokens
        assert len(logprob_values) > 0, "Expected at least one logprob token"

def test_vertexai_global_location_single_call(
    model_name: str | None, endpoint_version: str
) -> None:
    """Test ChatVertexAI single call with global location."""
    model = ChatVertexAI(
        model=model_name,
        location="global",
        rate_limiter=RATE_LIMITER,
        endpoint_version=endpoint_version,
    )
    assert model.location == "global"
    message = HumanMessage(content="Hello")
    response = model.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    _check_usage_metadata(response)

def test_nested_bind_tools() -> None:
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME)

    class Person(BaseModel):
        name: str = Field(description="The name.")
        hair_color: str | None = Field("Hair color, only if provided.")

    class People(BaseModel):
        data: list[Person] = Field(description="The people.")

    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME)
    llm_with_tools = llm.bind_tools([People], tool_choice="People")

    response = llm_with_tools.invoke("Chester, no hair color provided.")
    assert isinstance(response, AIMessage)
    assert response.tool_calls[0]["name"] == "People"

def test_search_builtin(output_version: str) -> None:
    """Test the built-in search tool."""
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME, output_version=output_version
    ).bind_tools([{"google_search": {}}])
    input_message = {
        "role": "user",
        "content": "What is today's news?",
    }

    # Test streaming
    full: AIMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    _check_web_search_output(full, output_version)

    # Test we can process chat history
    next_message = {
        "role": "user",
        "content": "Tell me more about that last story.",
    }
    response = llm.invoke([input_message, full, next_message])
    _check_web_search_output(response, output_version)

def test_code_execution_builtin(output_version: str) -> None:
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME, output_version=output_version
    ).bind_tools([{"code_execution": {}}])
    input_message = {
        "role": "user",
        "content": "What is 3^3?",
    }

    full: AIMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)

    _check_code_execution_output(full, output_version)

    # Test passing back in chat history without raising errors
    next_message = {
        "role": "user",
        "content": "Can you show me the calculation again with comments?",
    }
    response = llm.invoke([input_message, full, next_message])
    _check_code_execution_output(response, output_version)

def test_chat_vertexai_timeout_non_streaming() -> None:
    """Test timeout parameter in non-streaming mode."""
    vertexai.init(api_transport="grpc")
    model = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        timeout=0.001,
        rate_limiter=RATE_LIMITER,
    )
    with pytest.raises(DeadlineExceeded):
        model.invoke([HumanMessage(content="Hello")])

def test_chat_vertexai_timeout_streaming() -> None:
    """Test timeout parameter in streaming mode."""
    vertexai.init(api_transport="grpc")
    model = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        timeout=0.001,
        streaming=True,
        rate_limiter=RATE_LIMITER,
    )
    with pytest.raises(DeadlineExceeded):
        model.invoke([HumanMessage(content="Hello")])


# --- libs/vertexai/tests/integration_tests/test_vectorstores.py ---

def test_vector_store_update_index(
    vector_store: VectorSearchVectorStore, sample_documents: list[Document]
) -> None:
    vector_store.add_documents(documents=sample_documents, is_complete_overwrite=True)

def test_vector_store_stream_update_index(
    datastore_vector_store: VectorSearchVectorStoreDatastore,
    sample_documents: list[Document],
) -> None:
    datastore_vector_store.add_documents(
        documents=sample_documents, is_complete_overwrite=True
    )


# --- libs/vertexai/tests/unit_tests/test_anthropic_utils.py ---

def test_make_thinking_message_chunk_from_anthropic_event() -> None:
    """Test the conversion of Anthropic event into AIMessageChunk."""
    thinking_chunk = _make_message_chunk_from_anthropic_event(
        event=RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=ThinkingDelta(
                thinking="thoughts of the model...",
                type="thinking_delta",
            ),
        ),
        stream_usage=True,
        coerce_content_to_string=False,
    )
    signature_chunk = _make_message_chunk_from_anthropic_event(
        event=RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=SignatureDelta(
                signature="thoughts-signature",
                type="signature_delta",
            ),
        ),
        stream_usage=True,
        coerce_content_to_string=False,
    )

    assert thinking_chunk == AIMessageChunk(
        content=[
            {
                "index": 1,
                "type": "thinking",
                "thinking": "thoughts of the model...",
            }
        ]
    )
    assert signature_chunk == AIMessageChunk(
        content=[
            {
                "index": 1,
                "type": "thinking",
                "signature": "thoughts-signature",
            }
        ]
    )
    assert isinstance(thinking_chunk, AIMessageChunk)
    assert isinstance(signature_chunk, AIMessageChunk)

