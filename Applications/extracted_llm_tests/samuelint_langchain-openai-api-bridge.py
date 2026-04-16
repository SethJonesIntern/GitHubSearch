# samuelint/langchain-openai-api-bridge
# 46 LLM-backed test functions across 40 test files
# Source: https://github.com/samuelint/langchain-openai-api-bridge

# --- tests/test_functional/fastapi_assistant_agent_anthropic/test_anthropic_multimodal.py ---

    def test_run_stream_message_deltas(
        self, openai_client: OpenAI, base64_pig_image: str
    ):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in the image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_pig_image},
                        },
                    ],
                },
            ]
        )

        stream = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="claude-3-5-sonnet-20240620",
            assistant_id="any",
            stream=True,
            temperature=0,
        )

        str_response = assistant_stream_events_to_str_response(stream)

        assert "pig" in str_response.lower()


# --- tests/test_functional/fastapi_assistant_agent_anthropic/test_assistant_server_anthropic.py ---

    def test_run_stream_message_deltas(
        self,
        openai_client: OpenAI,
    ):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "Hello!",
                },
            ]
        )

        stream = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="claude-3-5-sonnet-20240620",
            assistant_id="any",
            stream=True,
            temperature=0,
        )

        str_response = assistant_stream_events_to_str_response(stream)

        assert len(str_response) > 0


# --- tests/test_functional/fastapi_assistant_agent_groq/test_assistant_server_groq.py ---

    def test_run_stream_message_deltas(
        self,
        openai_client: OpenAI,
    ):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "Hello!",
                },
            ]
        )

        stream = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="llama3-8b-8192",
            assistant_id="any",
            stream=True,
            temperature=0,
        )

        str_response = assistant_stream_events_to_str_response(stream)

        assert len(str_response) > 0


# --- tests/test_functional/fastapi_assistant_agent_llamacpp/test_assistant_server_llamacpp.py ---

    def test_run_stream_response_has_no_undesired_characters(
        self,
        openai_client: OpenAI,
    ):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "Hello!",
                },
            ]
        )

        stream = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="llama3",
            assistant_id="any",
            stream=True,
            temperature=0,
        )

        str_response = assistant_stream_events_to_str_response(stream)

        assert len(str_response) > 0
        assert "&#39;" not in str_response

    def test_function_calling(
        self,
        openai_client: OpenAI,
    ):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "What is the magic number of 125?",
                },
            ]
        )

        stream = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="llama3",
            assistant_id="any",
            stream=True,
            temperature=0,
        )

        str_response = assistant_stream_events_to_str_response(stream)

        assert "127" in str_response


# --- tests/test_functional/fastapi_assistant_agent_openai/test_assistant_server_openai.py ---

    def test_run_stream_starts_with_thread_run_created(
        self, stream_response_events: List[AssistantStreamEvent]
    ):
        assert stream_response_events[0].event == "thread.run.created"

    def test_run_stream_ends_with_thread_run_completed(
        self, stream_response_events: List[AssistantStreamEvent]
    ):
        assert stream_response_events[-1].event == "thread.run.completed"

    def test_run_stream_message_deltas(
        self, stream_response_events: List[AssistantStreamEvent]
    ):
        str_response = assistant_stream_events_to_str_response(stream_response_events)

        assert "This is a test message." in str_response

    def test_message_id_is_same_for_start_delta_ends(
        self, stream_response_events: List[AssistantStreamEvent]
    ):
        first_thread_message_completed = next(
            (
                event
                for event in stream_response_events
                if event.event == "thread.message.completed"
            ),
            None,
        )

        assert stream_response_events[1].event == "thread.message.created"
        assert stream_response_events[2].event == "thread.message.delta"
        assert first_thread_message_completed.event == "thread.message.completed"
        assert stream_response_events[1].data.id == stream_response_events[2].data.id
        assert (
            first_thread_message_completed.data.id == stream_response_events[2].data.id
        )

    def test_run_stream_starts_with_thread_run_created(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "Remember that my favorite fruit is banana. I Like bananas.",
                },
            ]
        )

        openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="gpt-4o-mini",
            assistant_id="any",
            temperature=0,
            stream=True,
        )

        openai_client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content="What is my favority fruit?"
        )

        stream_2 = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            temperature=0,
            model="gpt-4o-mini",
            assistant_id="any",
            stream=True,
        )

        events_2: List[AssistantStreamEvent] = []
        for event in stream_2:
            events_2.append(event)

        followup_response = assistant_stream_events_to_str_response(events_2)

        assert "banana" in followup_response

    def test_run_data_is_retreivable_from_message(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "Remember that my favorite fruit is banana. I Like bananas.",
                },
            ]
        )
        openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="gpt-4o-mini",
            assistant_id="any",
            temperature=0,
            stream=True,
        )
        messages = openai_client.beta.threads.messages.list(
            thread_id=thread.id,
        ).data
        last_message_run_id = messages[-1].run_id
        last_message_run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=last_message_run_id,
        )

        assert last_message_run is not None


# --- tests/test_functional/fastapi_assistant_agent_openai/test_list_runs.py ---

    def test_list_threads(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create()

        run1 = openai_client.beta.threads.runs.create(
            assistant_id="assistant1",
            thread_id=thread.id,
            model="any",
            stream=False,
        )

        run2 = openai_client.beta.threads.runs.create(
            assistant_id="assistant1",
            thread_id=thread.id,
            model="any",
            stream=False,
        )

        thread_runs = openai_client.beta.threads.runs.list(thread_id=thread.id).data

        assert len(thread_runs) == 2
        assert run1.id in [run.id for run in thread_runs]
        assert run2.id in [run.id for run in thread_runs]


# --- tests/test_functional/fastapi_assistant_agent_openai/test_message_crud.py ---

    def test_retreive_message(self, openai_client: OpenAI):
        created_thread = openai_client.beta.threads.create()
        created_message = openai_client.beta.threads.messages.create(
            thread_id=created_thread.id, role="user", content="Hello, what is AI?"
        )

        message = openai_client.beta.threads.messages.retrieve(
            message_id=created_message.id, thread_id=created_thread.id
        )

        assert message.role == "user"
        assert message.content[0].type == "text"
        assert message.content[0].text.value == "Hello, what is AI?"

    def test_delete_message(self, openai_client: OpenAI):
        created_thread = openai_client.beta.threads.create()
        created_message = openai_client.beta.threads.messages.create(
            thread_id=created_thread.id, role="user", content="Hello, what is AI?"
        )

        created_message = openai_client.beta.threads.messages.delete(
            thread_id=created_thread.id, message_id=created_message.id
        )
        message = openai_client.beta.threads.messages.retrieve(
            message_id=created_message.id, thread_id=created_thread.id
        )

        assert message is None


# --- tests/test_functional/fastapi_assistant_agent_openai/test_openai_multimodal.py ---

    def test_run_stream_message_deltas(
        self, openai_client: OpenAI, base64_pig_image: str
    ):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in the image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_pig_image},
                        },
                    ],
                },
            ]
        )

        stream = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="gpt-4o",
            assistant_id="any",
            stream=True,
            temperature=0,
        )

        str_response = assistant_stream_events_to_str_response(stream)

        assert "pig" in str_response.lower()


# --- tests/test_functional/fastapi_assistant_agent_openai/test_thread_delete.py ---

    def test_runs_associated_with_thread_are_deleted(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create()

        run = openai_client.beta.threads.runs.create(
            assistant_id="assistant1",
            thread_id=thread.id,
            model="any",
            stream=False,
        )

        openai_client.beta.threads.delete(thread_id=thread.id)

        retreive_run = openai_client.beta.threads.runs.retrieve(
            run_id=run.id, thread_id=thread.id
        )

        assert retreive_run is None

    def test_messages_associated_with_thread_are_deleted(self, openai_client: OpenAI):

        thread = openai_client.beta.threads.create()
        message = openai_client.beta.threads.messages.create(
            thread_id=thread.id, content="hello", role="user"
        )

        openai_client.beta.threads.delete(thread_id=thread.id)

        retreive_message = openai_client.beta.threads.messages.retrieve(
            message_id=message.id, thread_id=thread.id
        )

        assert retreive_message is None


# --- tests/test_functional/fastapi_assistant_agent_openai/test_tool_calling_server_openai.py ---

    def test_simple_tool_is_called(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "What is the magic number of 45?",
                },
            ]
        )
        stream = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="gpt-4o-mini",
            assistant_id="any",
            temperature=0,
            stream=True,
        )

        events = assistant_stream_events(stream)
        str_response = assistant_stream_events_to_str_response(events)

        assert "47" in str_response

    def test_the_same_answer_is_not_repeated(
        self, result_events: list[AssistantStreamEvent]
    ):
        str_response = assistant_stream_events_to_str_response(result_events)

        is_a_repetition = validate_llm_response(
            question="Is the same message a repeted?",
            str_response=str_response,
        )

        assert "no" in is_a_repetition


# --- tests/test_functional/fastapi_chat_completion_anthropic/test_server_anthropic.py ---

def test_chat_completion_invoke(openai_client):
    chat_completion = openai_client.chat.completions.create(
        model="claude-3-haiku-20240307",
        messages=[
            {
                "role": "user",
                "content": 'Say "This is a test"',
            }
        ],
    )
    assert "This is a test" in chat_completion.choices[0].message.content

def test_chat_completion_stream(openai_client):
    chunks = openai_client.chat.completions.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": 'Say "This is a test"'}],
        stream=True,
    )
    every_content = []
    for chunk in chunks:
        if chunk.choices and isinstance(chunk.choices[0].delta.content, str):
            every_content.append(chunk.choices[0].delta.content)

    stream_output = "".join(every_content)

    assert "This is a test" in stream_output


# --- tests/test_functional/fastapi_chat_completion_multi_agent_openai/test_multi_agent_server_openai.py ---

def test_chat_completion_invoke(openai_client):
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 'What time is it?',
            }
        ],
    )
    assert "time" in chat_completion.choices[0].message.content

def test_chat_completion_stream(openai_client):
    chunks = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": 'How does photosynthesis work?'}],
        stream=True,
    )
    every_content = []
    for chunk in chunks:
        if chunk.choices and isinstance(chunk.choices[0].delta.content, str):
            every_content.append(chunk.choices[0].delta.content)

    stream_output = "".join(every_content)

    assert "light" in stream_output


# --- tests/test_functional/fastapi_chat_completion_openai/test_server_openai.py ---

def test_chat_completion_invoke(openai_client):
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 'Say "This is a test"',
            }
        ],
    )
    assert "This is a test" in chat_completion.choices[0].message.content

def test_chat_completion_stream(openai_client):
    chunks = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": 'Say "This is a test"'}],
        stream=True,
    )
    every_content = []
    for chunk in chunks:
        if chunk.choices and isinstance(chunk.choices[0].delta.content, str):
            every_content.append(chunk.choices[0].delta.content)

    stream_output = "".join(every_content)

    assert "This is a test" in stream_output


# --- tests/test_functional/fastapi_chat_completion_openai/test_server_openai_event_adapter.py ---

def test_chat_completion_invoke_custom_events(openai_client_custom_events):
    chat_completion = openai_client_custom_events.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 'Say "This is a test"',
            }
        ],
    )
    assert "This is a test" in chat_completion.choices[0].message.content

def test_chat_completion_stream_custom_events(openai_client_custom_events):
    chunks = openai_client_custom_events.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": 'Say "This is a test"'}],
        stream=True,
    )
    every_content = []
    for chunk in chunks:
        if chunk.choices and isinstance(chunk.choices[0].delta.content, str):
            every_content.append(chunk.choices[0].delta.content)

    stream_output = "".join(every_content)

    assert "This is a test" in stream_output


# --- tests/test_functional/fastapi_chat_completion_react_agent_openai/test_react_agent_server_openai.py ---

def test_chat_completion_invoke(openai_client):
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 'Say "This is a test"',
            }
        ],
    )
    assert "This is a test" in chat_completion.choices[0].message.content

def test_chat_completion_stream(openai_client):
    chunks = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": 'Say "This is a test"'}],
        stream=True,
    )
    every_content = []
    for chunk in chunks:
        if chunk.choices and isinstance(chunk.choices[0].delta.content, str):
            every_content.append(chunk.choices[0].delta.content)

    stream_output = "".join(every_content)

    assert "This is a test" in stream_output

def test_tool(openai_client):

    chat_completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 'Say "Magic number of 2"',
            }
        ],
    )
    assert "4" in chat_completion.choices[0].message.content


# --- tests/test_functional/injector/test_with_injector_assistant_server_openai.py ---

    def test_run_stream_starts_with_thread_run_created(
        self, openai_client: OpenAI, thread: Thread
    ):
        openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="gpt-4o-mini",
            assistant_id="any",
            temperature=0,
            stream=True,
        )

        openai_client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content="What is my favority fruit?"
        )

        stream_2 = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            temperature=0,
            model="gpt-4o-mini",
            assistant_id="any",
            stream=True,
        )

        events_2: List[AssistantStreamEvent] = []
        for event in stream_2:
            events_2.append(event)

        followup_response = assistant_stream_events_to_str_response(events_2)

        assert "banana" in followup_response


# --- tests/test_unit/assistant/adapter/test_on_chat_model_end_handler.py ---

    def test_message_not_existing_in_database_returns_no_events(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelEndHandler,
        some_thread_dto: ThreadRunsDto,
        some_run: Run,
    ):
        event = create_stream_output_event(run_id="a", event="on_chat_model_end")
        decoy.when(
            thread_message_repository.retreive_unique_by_run_id(
                run_id="a", thread_id=some_thread_dto.thread_id
            )
        ).then_return(None)

        result = instance.handle(event=event, dto=some_thread_dto, run=some_run)

        assert len(result) == 0

    def test_message_existing_in_database_is_completed_with_final_content(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelEndHandler,
        some_thread_dto: ThreadRunsDto,
        some_message: Message,
        some_run: Run,
    ):
        event = create_stream_output_event(
            run_id="a", event="on_chat_model_end", content="hello world!"
        )
        decoy.when(
            thread_message_repository.retreive_unique_by_run_id(
                run_id="a", thread_id=some_thread_dto.thread_id
            )
        ).then_return(some_message)
        decoy.when(thread_message_repository.update(matchers.Anything())).then_do(
            lambda message: message
        )

        result = instance.handle(event=event, dto=some_thread_dto, run=some_run)

        assert result[0].event == "thread.message.completed"
        assert result[0].data.content[0].text.value == "hello world!"

    def test_message_existing_in_database_is_completed_with_completed_status(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelEndHandler,
        some_thread_dto: ThreadRunsDto,
        some_message: Message,
        some_run: Run,
    ):
        event = create_stream_output_event(
            run_id="a", event="on_chat_model_end", content="hello world!"
        )
        decoy.when(
            thread_message_repository.retreive_unique_by_run_id(
                run_id="a", thread_id=some_thread_dto.thread_id
            )
        ).then_return(some_message)
        decoy.when(thread_message_repository.update(matchers.Anything())).then_do(
            lambda message: message
        )

        result = instance.handle(event=event, dto=some_thread_dto, run=some_run)

        assert result[0].data.status == "completed"
        decoy.verify(
            thread_message_repository.update(
                matchers.HasAttributes({"id": "1", "status": "completed"})
            )
        )


# --- tests/test_unit/assistant/adapter/test_on_chat_model_stream_handler.py ---

    def test_when_message_exist_delta_is_returned(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelStreamHandler,
        some_thread_dto: ThreadRunsDto,
        some_run: Run,
        some_message: Message,
    ):
        event = create_stream_chunk_event(
            run_id="a", event="on_chat_model_stream", content=" World!"
        )
        decoy.when(
            thread_message_repository.retreive_unique_by_run_id(
                run_id="a", thread_id="thread1"
            )
        ).then_return(some_message)

        result = instance.handle(event=event, dto=some_thread_dto, run=some_run)

        assert result[0].event == "thread.message.delta"
        assert result[0].data.delta.content[0].text.value == " World!"

    def test_message_delta_is_persisted(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelStreamHandler,
        some_thread_dto: ThreadRunsDto,
        some_run: Run,
        some_message: Message,
    ):
        event = create_stream_chunk_event(
            run_id="a", event="on_chat_model_stream", content=" World!"
        )
        decoy.when(
            thread_message_repository.retreive_unique_by_run_id(
                run_id="a", thread_id="thread1"
            )
        ).then_return(some_message)

        instance.handle(event=event, dto=some_thread_dto, run=some_run)

        decoy.verify(
            thread_message_repository.update(
                matchers.HasAttributes(
                    {
                        "id": "msg1",
                        "content": [
                            TextContentBlock(
                                text=Text(value=" World!", annotations=[]), type="text"
                            )
                        ],
                    }
                )
            )
        )

    def test_not_persisted_message_is_created_and_then_content_chunk_is_put_as_delta_event(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelStreamHandler,
        some_thread_dto: ThreadRunsDto,
        some_run: Run,
    ):
        event = create_stream_chunk_event(
            run_id="a", event="on_chat_model_stream", content="Hello"
        )

        decoy.when(
            thread_message_repository.retreive_message_id_by_run_id(
                run_id="a", thread_id="thread1"
            )
        ).then_return(None)
        created_message = create_message(
            id="1", thread_id="thread1", role="assistant", content=[]
        )
        decoy.when(
            thread_message_repository.create(
                thread_id="thread1",
                role="assistant",
                status="in_progress",
                run_id="a",
            )
        ).then_return(created_message)

        result = instance.handle(event=event, dto=some_thread_dto, run=some_run)

        assert len(result[0].data.content) == 0
        assert result[0].event == "thread.message.created"
        assert result[0].data == created_message
        assert result[1].event == "thread.message.delta"
        assert result[1].data.delta.content[0].text.value == "Hello"

    def test_event_without_content_returns_no_events(
        self,
        instance: OnChatModelStreamHandler,
        some_thread_dto: ThreadRunsDto,
        some_run: Run,
    ):
        event = create_stream_chunk_event(
            run_id="a", event="on_chat_model_stream", content=""
        )

        result = instance.handle(event=event, dto=some_thread_dto, run=some_run)

        assert len(result) == 0


# --- tests/test_unit/assistant/adapter/test_thread_run_event_handler.py ---

    def test_event_is_created_from_database(
        self,
        decoy: Decoy,
        run_repository: RunRepository,
        instance: ThreadRunEventHandler,
        some_run: Run,
    ):
        decoy.when(
            run_repository.create(
                assistant_id="assistant1",
                thread_id="thread1",
                model="some-model",
                status="in_progress",
                temperature=None,
            )
        ).then_return(some_run)

        result = instance.on_thread_run_start(
            assistant_id="assistant1", thread_id="thread1", model="some-model"
        )

        assert result.data == some_run

    def test_event_type(
        self,
        decoy: Decoy,
        run_repository: RunRepository,
        instance: ThreadRunEventHandler,
        some_run: Run,
    ):
        decoy.when(
            run_repository.create(
                assistant_id="assistant1",
                thread_id="thread1",
                model="some-model",
                status="in_progress",
                temperature=None,
            )
        ).then_return(some_run)

        result = instance.on_thread_run_start(
            assistant_id="assistant1", thread_id="thread1", model="some-model"
        )

        assert result.event == "thread.run.created"

    def test_event_type(
        self,
        instance: ThreadRunEventHandler,
        some_run: Run,
    ):
        result = instance.on_thread_run_completed(run=some_run)

        assert result.event == "thread.run.completed"

    def test_event_status_transition_to_completed(
        self,
        instance: ThreadRunEventHandler,
        some_run: Run,
    ):
        some_run.status = "in_progress"

        result = instance.on_thread_run_completed(run=some_run)

        assert result.data.status == "completed"

    def test_run_is_persisted(
        self,
        decoy: Decoy,
        run_repository: RunRepository,
        instance: ThreadRunEventHandler,
        some_run: Run,
    ):
        some_run.status = "in_progress"

        instance.on_thread_run_completed(run=some_run)

        decoy.verify(
            run_repository.update(
                matchers.HasAttributes({"id": "run1", "status": "completed"})
            ),
            times=1,
        )


# --- tests/test_unit/assistant/repository/test_in_memory_message_repository.py ---

    def test_created_contains_run_id(self, instance: InMemoryMessageRepository):
        created = instance.create(
            thread_id="A",
            role="user",
            content="",
            run_id="123",
        )

        retreived = instance.retreive(message_id=created.id, thread_id="A")

        assert retreived.run_id == "123"

    def test_message_is_retreivable_by_run_id(
        self, instance: InMemoryMessageRepository
    ):
        expected = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
            run_id="123",
        )
        instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_2],
            run_id="1234",
        )
        instance.create(
            thread_id="B",
            role="user",
            content=[some_text_content_1],
            run_id="123",
        )
        result = instance.retreive_unique_by_run_id(run_id="123", thread_id="A")

        assert result.id == expected.id
        assert result.content == expected.content

    def test_not_found_return_none(self, instance: InMemoryMessageRepository):

        result = instance.retreive_unique_by_run_id(run_id="123", thread_id="Z")

        assert result is None

