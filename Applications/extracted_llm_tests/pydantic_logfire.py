# pydantic/logfire
# 62 LLM-backed test functions across 81 test files
# Source: https://github.com/pydantic/logfire

# --- tests/test_cli.py ---

def test_main_module() -> None:
    """Test that logfire.__main__ is importable for coverage."""
    assert subprocess.run([sys.executable, '-m', 'logfire', '--help'], check=True).returncode == 0


# --- tests/otel_integrations/test_anthropic.py ---

def test_sync_messages(instrumented_client: anthropic.Anthropic, exporter: TestExporter) -> None:
    response = instrumented_client.messages.create(
        max_tokens=1000,
        model='claude-3-haiku-20240307',
        system='You are a helpful assistant.',
        messages=[{'role': 'user', 'content': 'What is four plus five?'}],
    )
    assert isinstance(response.content[0], TextBlock)
    assert response.content[0].text == 'Nine'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_sync_messages',
                    'code.lineno': 123,
                    'request_data': {
                        'max_tokens': 1000,
                        'system': 'You are a helpful assistant.',
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'claude-3-haiku-20240307',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-3-haiku-20240307',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'async': False,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-3-haiku-20240307'",
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'response_data': {
                        'message': {
                            'content': 'Nine',
                            'role': 'assistant',
                        },
                        'usage': IsPartialDict(
                            {
                                'cache_creation': None,
                                'input_tokens': 2,
                                'output_tokens': 3,
                                'cache_creation_input_tokens': None,
                                'cache_read_input_tokens': None,
                                'server_tool_use': None,
                                'service_tier': None,
                            }
                        ),
                    },
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'text', 'content': 'Nine'}],
                            'finish_reason': 'end_turn',
                        }
                    ],
                    'gen_ai.response.model': 'claude-3-haiku-20240307',
                    'gen_ai.response.id': 'test_id',
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.usage.output_tokens': 3,
                    'gen_ai.usage.raw': {'input_tokens': 2, 'output_tokens': 3},
                    'gen_ai.response.finish_reasons': ['end_turn'],
                    'operation.cost': 4.25e-06,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'usage': {
                                        'type': 'object',
                                        'title': 'Usage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.model': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'operation.cost': {},
                        },
                    },
                },
            }
        ]
    )

async def test_async_messages(instrumented_async_client: anthropic.AsyncAnthropic, exporter: TestExporter) -> None:
    response = await instrumented_async_client.messages.create(
        max_tokens=1000,
        model='claude-3-haiku-20240307',
        system='You are a helpful assistant.',
        messages=[{'role': 'user', 'content': 'What is four plus five?'}],
    )
    assert isinstance(response.content[0], TextBlock)
    assert response.content[0].text == 'Nine'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_async_messages',
                    'code.lineno': 123,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'claude-3-haiku-20240307',
                        'system': 'You are a helpful assistant.',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-3-haiku-20240307',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'async': True,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-3-haiku-20240307'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'response_data': {
                        'message': {'role': 'assistant', 'content': 'Nine'},
                        'usage': {
                            'cache_creation': None,
                            'cache_creation_input_tokens': None,
                            'cache_read_input_tokens': None,
                            'inference_geo': None,
                            'input_tokens': 2,
                            'output_tokens': 3,
                            'server_tool_use': None,
                            'service_tier': None,
                        },
                    },
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'text', 'content': 'Nine'}],
                            'finish_reason': 'end_turn',
                        }
                    ],
                    'gen_ai.response.model': 'claude-3-haiku-20240307',
                    'gen_ai.response.id': 'test_id',
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.usage.output_tokens': 3,
                    'gen_ai.usage.raw': {'input_tokens': 2, 'output_tokens': 3},
                    'gen_ai.response.finish_reasons': ['end_turn'],
                    'operation.cost': 4.25e-06,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'usage': {'type': 'object', 'title': 'Usage', 'x-python-datatype': 'PydanticModel'}
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.model': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'operation.cost': {},
                        },
                    },
                },
            }
        ]
    )

def test_sync_message_empty_response_chunk(instrumented_client: anthropic.Anthropic, exporter: TestExporter) -> None:
    response = instrumented_client.messages.create(
        max_tokens=1000,
        model='claude-3-haiku-20240307',
        system='empty response chunk',
        messages=[],
        stream=True,
    )
    combined = [chunk for chunk in response]
    assert combined == []
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_sync_message_empty_response_chunk',
                    'code.lineno': 123,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [],
                        'model': 'claude-3-haiku-20240307',
                        'stream': True,
                        'system': 'empty response chunk',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-3-haiku-20240307',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'empty response chunk'}],
                    'async': False,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-3-haiku-20240307'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'claude-3-haiku-20240307',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.level_num': 9,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [],
                        'model': 'claude-3-haiku-20240307',
                        'stream': True,
                        'system': 'empty response chunk',
                    },
                    'async': False,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_sync_message_empty_response_chunk',
                    'code.lineno': 123,
                    'logfire.msg': "streaming response from 'claude-3-haiku-20240307' took 1.00s",
                    'logfire.span_type': 'log',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-3-haiku-20240307',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'empty response chunk'}],
                    'logfire.tags': ('LLM',),
                    'duration': 1.0,
                    'response_data': {'combined_chunk_content': '', 'chunk_count': 0},
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'duration': {},
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                            'response_data': {'type': 'object'},
                        },
                    },
                    'gen_ai.response.model': 'claude-3-haiku-20240307',
                },
            },
        ]
    )

def test_sync_messages_stream(instrumented_client: anthropic.Anthropic, exporter: TestExporter) -> None:
    response = instrumented_client.messages.create(
        max_tokens=1000,
        model='claude-3-haiku-20240307',
        system='You are a helpful assistant.',
        messages=[{'role': 'user', 'content': 'What is four plus five?'}],
        stream=True,
    )
    with response as stream:
        combined = ''.join(
            chunk.delta.text  # type: ignore
            for chunk in stream
            if hasattr(chunk, 'delta') and isinstance(chunk.delta, TextDelta)  # type: ignore
        )
    assert combined == 'The answer is secret'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_sync_messages_stream',
                    'code.lineno': 123,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'claude-3-haiku-20240307',
                        'stream': True,
                        'system': 'You are a helpful assistant.',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-3-haiku-20240307',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'async': False,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-3-haiku-20240307'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'claude-3-haiku-20240307',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.level_num': 9,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'claude-3-haiku-20240307',
                        'stream': True,
                        'system': 'You are a helpful assistant.',
                    },
                    'async': False,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'code.filepath': 'test_anthropic.py',
                    'code.function': '<genexpr>',
                    'code.lineno': 123,
                    'logfire.msg': "streaming response from 'claude-3-haiku-20240307' took 1.00s",
                    'logfire.span_type': 'log',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-3-haiku-20240307',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'logfire.tags': ('LLM',),
                    'duration': 1.0,
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'text', 'content': 'The answer is secret'}],
                            'finish_reason': 'end_turn',
                        }
                    ],
                    'gen_ai.usage.input_tokens': 25,
                    'gen_ai.usage.output_tokens': 55,
                    'gen_ai.usage.raw': {'input_tokens': 25, 'output_tokens': 55},
                    'operation.cost': 7.5e-05,
                    'response_data': {'combined_chunk_content': 'The answer is secret', 'chunk_count': 2},
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'duration': {},
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                            'response_data': {'type': 'object'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'operation.cost': {},
                        },
                    },
                    'gen_ai.response.model': 'claude-3-haiku-20240307',
                },
            },
        ]
    )

async def test_async_messages_stream(
    instrumented_async_client: anthropic.AsyncAnthropic, exporter: TestExporter
) -> None:
    response = await instrumented_async_client.messages.create(
        max_tokens=1000,
        model='claude-3-haiku-20240307',
        system='You are a helpful assistant.',
        messages=[{'role': 'user', 'content': 'What is four plus five?'}],
        stream=True,
    )
    async with response as stream:
        chunk_content = [
            chunk.delta.text  # type: ignore
            async for chunk in stream
            if hasattr(chunk, 'delta') and isinstance(chunk.delta, TextDelta)  # type: ignore
        ]
        combined = ''.join(chunk_content)
    assert combined == 'The answer is secret'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_async_messages_stream',
                    'code.lineno': 123,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'claude-3-haiku-20240307',
                        'stream': True,
                        'system': 'You are a helpful assistant.',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-3-haiku-20240307',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'async': True,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-3-haiku-20240307'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'claude-3-haiku-20240307',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.level_num': 9,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'claude-3-haiku-20240307',
                        'stream': True,
                        'system': 'You are a helpful assistant.',
                    },
                    'async': True,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_async_messages_stream',
                    'code.lineno': 123,
                    'logfire.msg': "streaming response from 'claude-3-haiku-20240307' took 1.00s",
                    'logfire.span_type': 'log',
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-3-haiku-20240307',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'logfire.tags': ('LLM',),
                    'duration': 1.0,
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'text', 'content': 'The answer is secret'}],
                            'finish_reason': 'end_turn',
                        }
                    ],
                    'gen_ai.usage.input_tokens': 25,
                    'gen_ai.usage.output_tokens': 55,
                    'gen_ai.usage.raw': {'input_tokens': 25, 'output_tokens': 55},
                    'operation.cost': 7.5e-05,
                    'response_data': {'combined_chunk_content': 'The answer is secret', 'chunk_count': 2},
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'duration': {},
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                            'response_data': {'type': 'object'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'operation.cost': {},
                        },
                    },
                    'gen_ai.response.model': 'claude-3-haiku-20240307',
                },
            },
        ]
    )

def test_tool_messages(instrumented_client: anthropic.Anthropic, exporter: TestExporter):
    response = instrumented_client.messages.create(
        max_tokens=1000,
        model='claude-3-haiku-20240307',
        system='tool response',
        messages=[],
    )
    content = response.content[0]
    assert content.input == {'param': 'param'}  # type: ignore
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_tool_messages',
                    'code.lineno': 123,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [],
                        'model': 'claude-3-haiku-20240307',
                        'system': 'tool response',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-3-haiku-20240307',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'tool response'}],
                    'async': False,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-3-haiku-20240307'",
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'response_data': {
                        'message': {
                            'role': 'assistant',
                            'tool_calls': [
                                {'id': 'id', 'function': {'arguments': '{"input":{"param":"param"}}', 'name': 'tool'}}
                            ],
                        },
                        'usage': IsPartialDict(
                            {
                                'cache_creation': None,
                                'cache_creation_input_tokens': None,
                                'cache_read_input_tokens': None,
                                'input_tokens': 2,
                                'output_tokens': 3,
                                'server_tool_use': None,
                                'service_tier': None,
                            }
                        ),
                    },
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [
                                {'type': 'tool_call', 'id': 'id', 'name': 'tool', 'arguments': {'param': 'param'}}
                            ],
                            'finish_reason': 'tool_use',
                        }
                    ],
                    'gen_ai.response.model': 'claude-3-haiku-20240307',
                    'gen_ai.response.id': 'test_id',
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.usage.output_tokens': 3,
                    'gen_ai.usage.raw': {'input_tokens': 2, 'output_tokens': 3},
                    'gen_ai.response.finish_reasons': ['tool_use'],
                    'operation.cost': 4.25e-06,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'usage': {'type': 'object', 'title': 'Usage', 'x-python-datatype': 'PydanticModel'}
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.model': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'operation.cost': {},
                        },
                    },
                },
            }
        ]
    )

def test_messages_without_stop_reason(instrumented_client: anthropic.Anthropic, exporter: TestExporter) -> None:
    """Test response without stop_reason (e.g., interrupted)."""
    response = instrumented_client.messages.create(
        max_tokens=1000,
        model='claude-3-haiku-20240307',
        system='no stop reason',
        messages=[{'role': 'user', 'content': 'Hello'}],
    )
    assert isinstance(response.content[0], TextBlock)
    assert response.content[0].text == 'Partial'
    # Verify finish_reasons is not present when stop_reason is None
    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    assert spans == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_messages_without_stop_reason',
                    'code.lineno': 123,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [{'role': 'user', 'content': 'Hello'}],
                        'model': 'claude-3-haiku-20240307',
                        'system': 'no stop reason',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-3-haiku-20240307',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]}],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'no stop reason'}],
                    'async': False,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-3-haiku-20240307'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'response_data': {
                        'message': {'role': 'assistant', 'content': 'Partial'},
                        'usage': {
                            'cache_creation': None,
                            'cache_creation_input_tokens': None,
                            'cache_read_input_tokens': None,
                            'inference_geo': None,
                            'input_tokens': 2,
                            'output_tokens': 3,
                            'server_tool_use': None,
                            'service_tier': None,
                        },
                    },
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Partial'}]}
                    ],
                    'gen_ai.response.model': 'claude-3-haiku-20240307',
                    'gen_ai.response.id': 'test_id',
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.usage.output_tokens': 3,
                    'gen_ai.usage.raw': {'input_tokens': 2, 'output_tokens': 3},
                    'operation.cost': 4.25e-06,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'usage': {'type': 'object', 'title': 'Usage', 'x-python-datatype': 'PydanticModel'}
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.model': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'operation.cost': {},
                        },
                    },
                },
            }
        ]
    )

def test_unknown_method(instrumented_client: anthropic.Anthropic, exporter: TestExporter) -> None:
    response = instrumented_client.completions.create(max_tokens_to_sample=1000, model='claude-2.1', prompt='prompt')
    assert response.completion == 'completion'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Anthropic API call to {url!r}',
                'context': {'is_remote': False, 'span_id': 1, 'trace_id': 1},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'request_data': {'max_tokens_to_sample': 1000, 'model': 'claude-2.1', 'prompt': 'prompt'},
                    'url': '/v1/complete',
                    'async': False,
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.request.model': 'claude-2.1',
                    'logfire.msg_template': 'Anthropic API call to {url!r}',
                    'logfire.msg': "Anthropic API call to '/v1/complete'",
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_unknown_method',
                    'code.lineno': 123,
                    'gen_ai.response.model': 'claude-2.1',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'url': {},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.request.model': {},
                            'async': {},
                        },
                    },
                },
            }
        ]
    )

def test_request_parameters(instrumented_client: anthropic.Anthropic, exporter: TestExporter) -> None:
    """Test that all request parameters are extracted and added to span attributes."""
    tools: list[Any] = [
        {
            'name': 'get_weather',
            'description': 'Get the current weather',
            'input_schema': {
                'type': 'object',
                'properties': {'location': {'type': 'string'}},
                'required': ['location'],
            },
        }
    ]
    response = instrumented_client.messages.create(
        max_tokens=1000,
        model='claude-3-haiku-20240307',
        system='You are a helpful assistant.',
        messages=[{'role': 'user', 'content': 'What is four plus five?'}],
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        stop_sequences=['END', 'STOP'],
        tools=cast(Any, tools),
    )
    assert isinstance(response.content[0], TextBlock)
    assert response.content[0].text == 'Nine'

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    assert spans == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_request_parameters',
                    'code.lineno': 123,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'claude-3-haiku-20240307',
                        'stop_sequences': ['END', 'STOP'],
                        'system': 'You are a helpful assistant.',
                        'temperature': 0.7,
                        'tools': [
                            {
                                'name': 'get_weather',
                                'description': 'Get the current weather',
                                'input_schema': {
                                    'type': 'object',
                                    'properties': {'location': {'type': 'string'}},
                                    'required': ['location'],
                                },
                            }
                        ],
                        'top_k': 40,
                        'top_p': 0.9,
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-3-haiku-20240307',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.request.temperature': 0.7,
                    'gen_ai.request.top_p': 0.9,
                    'gen_ai.request.top_k': 40,
                    'gen_ai.request.stop_sequences': ['END', 'STOP'],
                    'gen_ai.tool.definitions': [
                        {
                            'name': 'get_weather',
                            'description': 'Get the current weather',
                            'input_schema': {
                                'type': 'object',
                                'properties': {'location': {'type': 'string'}},
                                'required': ['location'],
                            },
                        }
                    ],
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'async': False,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-3-haiku-20240307'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'response_data': {
                        'message': {'role': 'assistant', 'content': 'Nine'},
                        'usage': {
                            'cache_creation': None,
                            'cache_creation_input_tokens': None,
                            'cache_read_input_tokens': None,
                            'inference_geo': None,
                            'input_tokens': 2,
                            'output_tokens': 3,
                            'server_tool_use': None,
                            'service_tier': None,
                        },
                    },
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'text', 'content': 'Nine'}],
                            'finish_reason': 'end_turn',
                        }
                    ],
                    'gen_ai.response.model': 'claude-3-haiku-20240307',
                    'gen_ai.response.id': 'test_id',
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.usage.output_tokens': 3,
                    'gen_ai.usage.raw': {'input_tokens': 2, 'output_tokens': 3},
                    'gen_ai.response.finish_reasons': ['end_turn'],
                    'operation.cost': 4.25e-06,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.request.temperature': {},
                            'gen_ai.request.top_p': {},
                            'gen_ai.request.top_k': {},
                            'gen_ai.request.stop_sequences': {},
                            'gen_ai.tool.definitions': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'usage': {'type': 'object', 'title': 'Usage', 'x-python-datatype': 'PydanticModel'}
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.model': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'operation.cost': {},
                        },
                    },
                },
            }
        ]
    )

def test_sync_messages_version_latest(exporter: TestExporter) -> None:
    """Test that version='latest' uses semconv attributes with minimal request_data and no response_data."""
    client = anthropic.Anthropic(api_key='foobar')
    logfire.instrument_anthropic(client, version='latest')
    response = client.messages.create(
        max_tokens=1000,
        model='claude-sonnet-4-20250514',
        system='You are a helpful assistant.',
        messages=[{'role': 'user', 'content': 'What is four plus five?'}],
    )
    assert isinstance(response.content[0], TextBlock)
    assert response.content[0].text == 'Four plus five equals nine.'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_sync_messages_version_latest',
                    'code.lineno': 123,
                    'request_data': {'model': 'claude-sonnet-4-20250514'},
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-sonnet-4-20250514',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'async': False,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-sonnet-4-20250514'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'text', 'content': 'Four plus five equals nine.'}],
                            'finish_reason': 'end_turn',
                        }
                    ],
                    'gen_ai.response.model': 'claude-sonnet-4-20250514',
                    'gen_ai.response.id': IsStr(),
                    'gen_ai.usage.input_tokens': IsInt(),
                    'gen_ai.usage.output_tokens': IsInt(),
                    'gen_ai.usage.raw': {
                        'cache_creation': {'ephemeral_1h_input_tokens': 0, 'ephemeral_5m_input_tokens': 0},
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'inference_geo': 'not_available',
                        'input_tokens': 19,
                        'output_tokens': 9,
                        'service_tier': 'standard',
                    },
                    'gen_ai.response.finish_reasons': ['end_turn'],
                    'operation.cost': 0.000192,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.model': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'operation.cost': {},
                        },
                    },
                },
            }
        ]
    )

def test_sync_messages_version_v1_only(exporter: TestExporter) -> None:
    """Test that version=1 does not emit gen_ai.input.messages or gen_ai.output.messages."""
    client = anthropic.Anthropic(api_key='foobar')
    logfire.instrument_anthropic(client, version=1)
    response = client.messages.create(
        max_tokens=1000,
        model='claude-sonnet-4-20250514',
        system='You are a helpful assistant.',
        messages=[{'role': 'user', 'content': 'What is four plus five?'}],
    )
    assert isinstance(response.content[0], TextBlock)
    assert response.content[0].text == 'Four plus five equals nine.'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_sync_messages_version_v1_only',
                    'code.lineno': 123,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'claude-sonnet-4-20250514',
                        'system': 'You are a helpful assistant.',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-sonnet-4-20250514',
                    'gen_ai.request.max_tokens': 1000,
                    'async': False,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-sonnet-4-20250514'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'response_data': {
                        'message': {'role': 'assistant', 'content': 'Four plus five equals nine.'},
                        'usage': {
                            'cache_creation': {
                                'ephemeral_1h_input_tokens': IsInt(),
                                'ephemeral_5m_input_tokens': IsInt(),
                            },
                            'cache_creation_input_tokens': IsInt(),
                            'cache_read_input_tokens': IsInt(),
                            'inference_geo': IsStr(),
                            'input_tokens': IsInt(),
                            'output_tokens': IsInt(),
                            'server_tool_use': None,
                            'service_tier': IsStr(),
                        },
                    },
                    'gen_ai.response.model': 'claude-sonnet-4-20250514',
                    'gen_ai.response.id': IsStr(),
                    'gen_ai.usage.input_tokens': IsInt(),
                    'gen_ai.usage.output_tokens': IsInt(),
                    'gen_ai.usage.raw': {
                        'cache_creation': {'ephemeral_1h_input_tokens': 0, 'ephemeral_5m_input_tokens': 0},
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'inference_geo': 'not_available',
                        'input_tokens': 19,
                        'output_tokens': 9,
                        'service_tier': 'standard',
                    },
                    'gen_ai.response.finish_reasons': ['end_turn'],
                    'operation.cost': 0.000192,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'async': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'usage': {
                                        'type': 'object',
                                        'title': 'Usage',
                                        'x-python-datatype': 'PydanticModel',
                                        'properties': {
                                            'cache_creation': {
                                                'type': 'object',
                                                'title': 'CacheCreation',
                                                'x-python-datatype': 'PydanticModel',
                                            }
                                        },
                                    }
                                },
                            },
                            'gen_ai.response.model': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'operation.cost': {},
                        },
                    },
                },
            }
        ]
    )

def test_sync_messages_stream_version_latest(exporter: TestExporter) -> None:
    """Test that streaming with version='latest' emits semconv attributes without response_data."""
    client = anthropic.Anthropic(api_key='foobar')
    logfire.instrument_anthropic(client, version='latest')
    response = client.messages.create(
        max_tokens=1000,
        model='claude-sonnet-4-20250514',
        system='You are a helpful assistant.',
        messages=[{'role': 'user', 'content': 'What is four plus five?'}],
        stream=True,
    )
    with response as stream:
        combined = ''.join(
            chunk.delta.text  # type: ignore
            for chunk in stream
            if hasattr(chunk, 'delta') and isinstance(chunk.delta, TextDelta)  # type: ignore
        )
    assert combined == 'Four plus five equals nine.'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_sync_messages_stream_version_latest',
                    'code.lineno': 123,
                    'request_data': {'model': 'claude-sonnet-4-20250514'},
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-sonnet-4-20250514',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'async': False,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-sonnet-4-20250514'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'claude-sonnet-4-20250514',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'logfire.msg': "streaming response from 'claude-sonnet-4-20250514' took 1.00s",
                    'code.filepath': 'test_anthropic.py',
                    'code.function': '<genexpr>',
                    'code.lineno': 123,
                    'duration': 1.0,
                    'request_data': {'model': 'claude-sonnet-4-20250514'},
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-sonnet-4-20250514',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'async': False,
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'text', 'content': 'Four plus five equals nine.'}],
                            'finish_reason': 'end_turn',
                        }
                    ],
                    'gen_ai.usage.input_tokens': 19,
                    'gen_ai.usage.output_tokens': 9,
                    'gen_ai.usage.raw': {
                        'cache_creation': {'ephemeral_1h_input_tokens': 0, 'ephemeral_5m_input_tokens': 0},
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'inference_geo': 'not_available',
                        'input_tokens': 19,
                        'output_tokens': 9,
                        'service_tier': 'standard',
                    },
                    'operation.cost': 0.000192,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'duration': {},
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'operation.cost': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'claude-sonnet-4-20250514',
                },
            },
        ]
    )

def test_sync_messages_stream_version_v1_only(exporter: TestExporter) -> None:
    """Test that streaming with version=1 emits response_data without semconv message attributes."""
    client = anthropic.Anthropic(api_key='foobar')
    logfire.instrument_anthropic(client, version=1)
    response = client.messages.create(
        max_tokens=1000,
        model='claude-sonnet-4-20250514',
        system='You are a helpful assistant.',
        messages=[{'role': 'user', 'content': 'What is four plus five?'}],
        stream=True,
    )
    with response as stream:
        combined = ''.join(
            chunk.delta.text  # type: ignore
            for chunk in stream
            if hasattr(chunk, 'delta') and isinstance(chunk.delta, TextDelta)  # type: ignore
        )
    assert combined == 'Four plus five equals nine.'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_sync_messages_stream_version_v1_only',
                    'code.lineno': 123,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'claude-sonnet-4-20250514',
                        'stream': True,
                        'system': 'You are a helpful assistant.',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-sonnet-4-20250514',
                    'gen_ai.request.max_tokens': 1000,
                    'async': False,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-sonnet-4-20250514'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'async': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'claude-sonnet-4-20250514',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'logfire.msg': "streaming response from 'claude-sonnet-4-20250514' took 1.00s",
                    'code.filepath': 'test_anthropic.py',
                    'code.function': '<genexpr>',
                    'code.lineno': 123,
                    'duration': 1.0,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'claude-sonnet-4-20250514',
                        'stream': True,
                        'system': 'You are a helpful assistant.',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-sonnet-4-20250514',
                    'gen_ai.request.max_tokens': 1000,
                    'async': False,
                    'response_data': {'combined_chunk_content': 'Four plus five equals nine.', 'chunk_count': IsInt()},
                    'gen_ai.usage.input_tokens': 19,
                    'gen_ai.usage.output_tokens': 9,
                    'gen_ai.usage.raw': {
                        'cache_creation': {'ephemeral_1h_input_tokens': 0, 'ephemeral_5m_input_tokens': 0},
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'inference_geo': 'not_available',
                        'input_tokens': 19,
                        'output_tokens': 9,
                        'service_tier': 'standard',
                    },
                    'operation.cost': 0.000192,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'duration': {},
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'async': {},
                            'response_data': {'type': 'object'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'operation.cost': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'claude-sonnet-4-20250514',
                },
            },
        ]
    )

def test_sync_messages_beta_stream(exporter: TestExporter) -> None:
    client = anthropic.Anthropic(api_key='foobar')
    logfire.instrument_anthropic(client, version=['latest', 1])
    response = client.beta.messages.create(
        max_tokens=1000,
        model='claude-sonnet-4-20250514',
        system='You are a helpful assistant.',
        messages=[{'role': 'user', 'content': [{'text': 'What is four plus five?', 'type': 'text'}]}],
        stream=True,
    )
    with response as stream:
        combined = ''.join(
            chunk.delta.text  # type: ignore
            for chunk in stream
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text')  # type: ignore
        )
    assert combined == snapshot('Four plus five equals nine.')
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_sync_messages_beta_stream',
                    'code.lineno': 123,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [
                            {'role': 'user', 'content': [{'text': 'What is four plus five?', 'type': 'text'}]}
                        ],
                        'model': 'claude-sonnet-4-20250514',
                        'stream': True,
                        'system': 'You are a helpful assistant.',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-sonnet-4-20250514',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'async': False,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-sonnet-4-20250514'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'claude-sonnet-4-20250514',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'logfire.msg': "streaming response from 'claude-sonnet-4-20250514' took 1.00s",
                    'code.filepath': 'test_anthropic.py',
                    'code.function': '<genexpr>',
                    'code.lineno': 123,
                    'duration': 1.0,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [
                            {'role': 'user', 'content': [{'text': 'What is four plus five?', 'type': 'text'}]}
                        ],
                        'model': 'claude-sonnet-4-20250514',
                        'stream': True,
                        'system': 'You are a helpful assistant.',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-sonnet-4-20250514',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'async': False,
                    'response_data': {'combined_chunk_content': 'Four plus five equals nine.', 'chunk_count': 3},
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'text', 'content': 'Four plus five equals nine.'}],
                            'finish_reason': 'end_turn',
                        }
                    ],
                    'gen_ai.usage.input_tokens': 19,
                    'gen_ai.usage.output_tokens': 9,
                    'gen_ai.usage.raw': {
                        'cache_creation': {'ephemeral_1h_input_tokens': 0, 'ephemeral_5m_input_tokens': 0},
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'inference_geo': 'not_available',
                        'input_tokens': 19,
                        'output_tokens': 9,
                        'service_tier': 'standard',
                    },
                    'operation.cost': 0.000192,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'duration': {},
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                            'response_data': {'type': 'object'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'operation.cost': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'claude-sonnet-4-20250514',
                },
            },
        ]
    )

async def test_async_beta_messages(exporter: TestExporter) -> None:
    client = anthropic.AsyncAnthropic()
    logfire.instrument_anthropic(client, version=['latest', 1])
    response = await client.beta.messages.create(
        max_tokens=1000,
        model='claude-3-haiku-20240307',
        system='You are a helpful assistant.',
        messages=[{'role': 'user', 'content': 'What is four plus five?'}],
    )
    assert response.content[0].model_dump() == snapshot({'citations': None, 'text': '4 + 5 = 9.', 'type': 'text'})
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Message with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_anthropic.py',
                    'code.function': 'test_async_beta_messages',
                    'code.lineno': 123,
                    'request_data': {
                        'max_tokens': 1000,
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'claude-3-haiku-20240307',
                        'system': 'You are a helpful assistant.',
                    },
                    'gen_ai.system': 'anthropic',
                    'gen_ai.provider.name': 'anthropic',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'claude-3-haiku-20240307',
                    'gen_ai.request.max_tokens': 1000,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'You are a helpful assistant.'}],
                    'async': True,
                    'logfire.msg_template': 'Message with {request_data[model]!r}',
                    'logfire.msg': "Message with 'claude-3-haiku-20240307'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'response_data': {
                        'message': {'role': 'assistant', 'content': '4 + 5 = 9.'},
                        'usage': {
                            'cache_creation': {'ephemeral_1h_input_tokens': 0, 'ephemeral_5m_input_tokens': 0},
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'inference_geo': 'not_available',
                            'input_tokens': 19,
                            'iterations': None,
                            'output_tokens': 14,
                            'server_tool_use': None,
                            'service_tier': 'standard',
                            'speed': None,
                        },
                    },
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'text', 'content': '4 + 5 = 9.'}],
                            'finish_reason': 'end_turn',
                        }
                    ],
                    'gen_ai.response.model': 'claude-3-haiku-20240307',
                    'gen_ai.response.id': 'msg_01HJB23z1SCp7SjLxmybeqgF',
                    'gen_ai.usage.input_tokens': 19,
                    'gen_ai.usage.output_tokens': 14,
                    'gen_ai.usage.raw': {
                        'cache_creation': {'ephemeral_1h_input_tokens': 0, 'ephemeral_5m_input_tokens': 0},
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'inference_geo': 'not_available',
                        'input_tokens': 19,
                        'output_tokens': 14,
                        'service_tier': 'standard',
                    },
                    'gen_ai.response.finish_reasons': ['end_turn'],
                    'operation.cost': 2.225e-05,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'async': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'usage': {
                                        'type': 'object',
                                        'title': 'BetaUsage',
                                        'x-python-datatype': 'PydanticModel',
                                        'properties': {
                                            'cache_creation': {
                                                'type': 'object',
                                                'title': 'BetaCacheCreation',
                                                'x-python-datatype': 'PydanticModel',
                                            }
                                        },
                                    }
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.model': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                            'operation.cost': {},
                        },
                    },
                },
            }
        ]
    )


# --- tests/otel_integrations/test_langchain.py ---

def test_instrument_langchain(exporter: TestExporter) -> None:
    from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langchain_openai import ChatOpenAI

    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    model = ChatOpenAI(
        model='gpt-5',
        reasoning={'effort': 'medium', 'summary': 'concise'},
        base_url='https://gateway.pydantic.dev/proxy/openai/',
    )
    math_agent = create_agent(model, tools=[add])  # pyright: ignore [reportUnknownVariableType]

    result = math_agent.invoke(  # pyright: ignore
        {'messages': [{'role': 'user', 'content': "what's 123 + 456? think carefully and use the tool"}]}
    )

    assert result['messages'][-1].content == snapshot(
        [
            {
                'type': 'text',
                'text': '579',
                'annotations': [],
                'id': 'msg_033ba4b7d827c976006978a474036481a2bddedf312304869f',
            }
        ]
    )

    # Wait for langsmith OTel thread
    wait_for_all_tracers()

    # All spans that have messages should have some 'prefix' of this list, maybe with extra keys in each dict.
    message_events_minimum: list[dict[str, Any]] = [
        {
            'role': 'user',
            'content': "what's 123 + 456? think carefully and use the tool",
        },
        {
            'role': 'assistant',
            'content': [
                {'type': 'reasoning', 'content': '**Using tool to add numbers**'},
                {'type': 'reasoning', 'content': '**Executing addition and finalizing result**'},
            ],
            'tool_calls': [
                {
                    'id': 'call_XlgatTV1bBqLX1fOZTbu7cxO',
                    'function': {'arguments': {'a': 123, 'b': 456}, 'name': 'add'},
                    'type': 'function',
                }
            ],
        },
        {
            'role': 'tool',
            'content': '579.0',
            'name': 'add',
            'id': 'call_XlgatTV1bBqLX1fOZTbu7cxO',
        },
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': '579',
                    'annotations': [],
                    'id': 'msg_033ba4b7d827c976006978a474036481a2bddedf312304869f',
                }
            ],
        },
    ]

    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    for span in spans:
        for actual_event, expected_event in zip(
            span['attributes'].get('all_messages_events', []), message_events_minimum
        ):
            assert actual_event == IsPartialDict(expected_event)

        if span['name'] == 'ChatOpenAI':
            assert span['attributes']['gen_ai.usage.input_tokens'] > 0
            assert span['attributes']['gen_ai.request.model'] == snapshot('gpt-5')
            assert span['attributes']['gen_ai.response.model'] == snapshot('gpt-5')
            assert span['attributes']['gen_ai.system'] == 'openai'
            assert span['attributes']['gen_ai.provider.name'] == 'openai'
        else:
            assert 'gen_ai.usage.input_tokens' not in span['attributes']
            assert 'gen_ai.request.model' not in span['attributes']
            assert 'gen_ai.response.model' not in span['attributes']
            assert 'gen_ai.system' not in span['attributes']
            assert 'gen_ai.provider.name' not in span['attributes']

    assert [
        (span['name'], len(span['attributes'].get('all_messages_events', [])))
        for span in sorted(spans, key=lambda s: s['start_time'])
    ] == snapshot(
        [
            ('LangGraph', 4),  # Full conversation in outermost span
            # First request and response
            ('model', 2),
            ('ChatOpenAI', 2),
            ('tools', 0),
            ('add', 0),
            # Second request and response included, thus the whole conversation
            ('model', 4),
            ('ChatOpenAI', 4),
        ]
    )

    [span] = [s for s in spans if s['name'] == 'ChatOpenAI' and len(s['attributes']['all_messages_events']) == 4]
    assert span['attributes']['all_messages_events'] == snapshot(
        [
            {'content': "what's 123 + 456? think carefully and use the tool", 'role': 'user'},
            {
                'role': 'assistant',
                'content': [
                    {'type': 'reasoning', 'content': '**Using tool to add numbers**'},
                    {'type': 'reasoning', 'content': '**Executing addition and finalizing result**'},
                ],
                'tool_calls': [
                    {
                        'id': 'call_XlgatTV1bBqLX1fOZTbu7cxO',
                        'function': {'arguments': {'a': 123, 'b': 456}, 'name': 'add'},
                        'type': 'function',
                    }
                ],
                'invalid_tool_calls': [],
            },
            {
                'role': 'tool',
                'content': '579.0',
                'name': 'add',
                'id': 'call_XlgatTV1bBqLX1fOZTbu7cxO',
                'status': 'success',
            },
            {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': '579',
                        'annotations': [],
                        'id': 'msg_033ba4b7d827c976006978a474036481a2bddedf312304869f',
                    }
                ],
                'invalid_tool_calls': [],
            },
        ]
    )


# --- tests/otel_integrations/test_litellm.py ---

def test_litellm_instrumentation(exporter: TestExporter) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        import litellm

    logging.getLogger('LiteLLM').disabled = True

    logfire.instrument_litellm()

    def get_current_weather(location: str):
        """Get the current weather in a given location"""
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})

    messages = [{'role': 'user', 'content': "What's the weather like in San Francisco?"}]
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_current_weather',
                'description': 'Get the current weather in a given location',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'The city and state, e.g. San Francisco, CA',
                        },
                        'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                    },
                    'required': ['location'],
                },
            },
        }
    ]

    def completion() -> Any:
        model = 'gpt-4o-mini'
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            return litellm.completion(model=model, messages=messages, tools=tools)  # type: ignore

    response = completion()
    response_message = response.choices[0].message
    messages.append(response_message)

    [tool_call] = response_message.tool_calls
    function_name = tool_call.function.name
    assert function_name == get_current_weather.__name__
    function_args = json.loads(tool_call.function.arguments)
    function_response = get_current_weather(**function_args)
    messages.append(
        {
            'tool_call_id': tool_call.id,
            'role': 'tool',
            'name': function_name,
            'content': function_response,
        }
    )

    second_response = completion()
    assert second_response.choices[0].message.content == snapshot(
        'The current temperature in San Francisco is 72°F. If you need more specific weather details or a forecast, let me know!'
    )

    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'completion',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'logfire.span_type': 'span',
                    'logfire.msg': 'completion',
                    'llm.model_name': 'gpt-4o-mini',
                    'llm.provider': 'openai',
                    'llm.input_messages.0.message.role': 'user',
                    'llm.input_messages.0.message.content': "What's the weather like in San Francisco?",
                    'input.value': {
                        'messages': [{'role': 'user', 'content': "What's the weather like in San Francisco?"}]
                    },
                    'input.mime_type': 'application/json',
                    'llm.invocation_parameters': {
                        'model': 'gpt-4o-mini',
                        'tools': [
                            {
                                'type': 'function',
                                'function': {
                                    'name': 'get_current_weather',
                                    'description': 'Get the current weather in a given location',
                                    'parameters': {
                                        'type': 'object',
                                        'properties': {
                                            'location': {
                                                'type': 'string',
                                                'description': 'The city and state, e.g. San Francisco, CA',
                                            },
                                            'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                        },
                                        'required': ['location'],
                                    },
                                },
                            }
                        ],
                    },
                    'llm.tools.0.tool.json_schema': {
                        'type': 'function',
                        'function': {
                            'name': 'get_current_weather',
                            'description': 'Get the current weather in a given location',
                            'parameters': {
                                'type': 'object',
                                'properties': {
                                    'location': {
                                        'type': 'string',
                                        'description': 'The city and state, e.g. San Francisco, CA',
                                    },
                                    'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                },
                                'required': ['location'],
                            },
                        },
                    },
                    'output.value': {
                        'id': 'chatcmpl-Br2eczuAVPiovQVLOcoEi7qbHonyZ',
                        'created': 1751981286,
                        'model': 'gpt-4o-mini-2024-07-18',
                        'object': 'chat.completion',
                        'system_fingerprint': 'fp_34a54ae93c',
                        'choices': [
                            {
                                'finish_reason': 'tool_calls',
                                'index': 0,
                                'message': {
                                    'content': None,
                                    'role': 'assistant',
                                    'tool_calls': [
                                        {
                                            'function': {
                                                'arguments': '{"location":"San Francisco, CA"}',
                                                'name': 'get_current_weather',
                                            },
                                            'id': 'call_SWFIWhfCI6AeHuaV6EM1MRsJ',
                                            'type': 'function',
                                        }
                                    ],
                                    'function_call': None,
                                    'provider_specific_fields': {'refusal': None},
                                    'annotations': [],
                                },
                                'provider_specific_fields': {},
                            }
                        ],
                        'usage': {
                            'completion_tokens': 18,
                            'prompt_tokens': 80,
                            'total_tokens': 98,
                            'completion_tokens_details': IsPartialDict(),
                            'prompt_tokens_details': IsPartialDict(),
                        },
                        'service_tier': 'default',
                    },
                    'output.mime_type': 'application/json',
                    'llm.output_messages.0.message.role': 'assistant',
                    'llm.output_messages.0.message.tool_calls.0.tool_call.function.name': 'get_current_weather',
                    'llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments': {
                        'location': 'San Francisco, CA'
                    },
                    'llm.token_count.prompt': 80,
                    'llm.token_count.prompt_details.cache_read': 0,
                    'llm.token_count.prompt_details.audio': 0,
                    'llm.token_count.completion': 18,
                    'llm.token_count.completion_details.reasoning': 0,
                    'llm.token_count.completion_details.audio': 0,
                    'llm.token_count.total': 98,
                    'openinference.span.kind': 'LLM',
                    'request_data': {
                        'messages': [{'role': 'user', 'content': "What's the weather like in San Francisco?"}]
                    },
                    'response_data': {
                        'message': {
                            'content': None,
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'function': {
                                        'arguments': '{"location":"San Francisco, CA"}',
                                        'name': 'get_current_weather',
                                    },
                                    'id': 'call_SWFIWhfCI6AeHuaV6EM1MRsJ',
                                    'type': 'function',
                                }
                            ],
                            'function_call': None,
                            'provider_specific_fields': {'refusal': None},
                            'annotations': [],
                        }
                    },
                    'gen_ai.request.model': 'gpt-4o-mini',
                    'gen_ai.response.model': 'gpt-4o-mini-2024-07-18',
                    'gen_ai.usage.input_tokens': 80,
                    'gen_ai.usage.output_tokens': 18,
                    'gen_ai.system': 'openai',
                    'logfire.tags': ['LLM'],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {'request_data': {'type': 'object'}, 'response_data': {'type': 'object'}},
                    },
                },
            },
            {
                'name': 'completion',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 3000000000,
                'end_time': 4000000000,
                'attributes': {
                    'logfire.span_type': 'span',
                    'logfire.msg': 'completion',
                    'llm.model_name': 'gpt-4o-mini',
                    'llm.provider': 'openai',
                    'llm.input_messages.0.message.role': 'user',
                    'llm.input_messages.0.message.content': "What's the weather like in San Francisco?",
                    'llm.input_messages.1.message.role': 'assistant',
                    'llm.input_messages.1.message.tool_calls.0.tool_call.function.name': 'get_current_weather',
                    'llm.input_messages.1.message.tool_calls.0.tool_call.function.arguments': {
                        'location': 'San Francisco, CA'
                    },
                    'llm.input_messages.2.message.role': 'tool',
                    'llm.input_messages.2.message.content': {
                        'location': 'San Francisco',
                        'temperature': '72',
                        'unit': 'fahrenheit',
                    },
                    'input.value': {
                        'messages': [
                            {'role': 'user', 'content': "What's the weather like in San Francisco?"},
                            {
                                'content': None,
                                'role': 'assistant',
                                'tool_calls': [
                                    {
                                        'function': {
                                            'arguments': '{"location":"San Francisco, CA"}',
                                            'name': 'get_current_weather',
                                        },
                                        'id': 'call_SWFIWhfCI6AeHuaV6EM1MRsJ',
                                        'type': 'function',
                                    }
                                ],
                                'function_call': None,
                                'provider_specific_fields': {'refusal': None},
                                'annotations': [],
                            },
                            {
                                'tool_call_id': 'call_SWFIWhfCI6AeHuaV6EM1MRsJ',
                                'role': 'tool',
                                'name': 'get_current_weather',
                                'content': '{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}',
                            },
                        ]
                    },
                    'input.mime_type': 'application/json',
                    'llm.invocation_parameters': {
                        'model': 'gpt-4o-mini',
                        'tools': [
                            {
                                'type': 'function',
                                'function': {
                                    'name': 'get_current_weather',
                                    'description': 'Get the current weather in a given location',
                                    'parameters': {
                                        'type': 'object',
                                        'properties': {
                                            'location': {
                                                'type': 'string',
                                                'description': 'The city and state, e.g. San Francisco, CA',
                                            },
                                            'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                        },
                                        'required': ['location'],
                                    },
                                },
                            }
                        ],
                    },
                    'llm.tools.0.tool.json_schema': {
                        'type': 'function',
                        'function': {
                            'name': 'get_current_weather',
                            'description': 'Get the current weather in a given location',
                            'parameters': {
                                'type': 'object',
                                'properties': {
                                    'location': {
                                        'type': 'string',
                                        'description': 'The city and state, e.g. San Francisco, CA',
                                    },
                                    'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                },
                                'required': ['location'],
                            },
                        },
                    },
                    'output.value': 'The current temperature in San Francisco is 72°F. If you need more specific weather details or a forecast, let me know!',
                    'llm.output_messages.0.message.role': 'assistant',
                    'llm.output_messages.0.message.content': 'The current temperature in San Francisco is 72°F. If you need more specific weather details or a forecast, let me know!',
                    'llm.token_count.prompt': 62,
                    'llm.token_count.prompt_details.cache_read': 0,
                    'llm.token_count.prompt_details.audio': 0,
                    'llm.token_count.completion': 26,
                    'llm.token_count.completion_details.reasoning': 0,
                    'llm.token_count.completion_details.audio': 0,
                    'llm.token_count.total': 88,
                    'openinference.span.kind': 'LLM',
                    'request_data': {
                        'messages': [
                            {'role': 'user', 'content': "What's the weather like in San Francisco?"},
                            {
                                'content': None,
                                'role': 'assistant',
                                'tool_calls': [
                                    {
                                        'function': {
                                            'arguments': '{"location":"San Francisco, CA"}',
                                            'name': 'get_current_weather',
                                        },
                                        'id': 'call_SWFIWhfCI6AeHuaV6EM1MRsJ',
                                        'type': 'function',
                                    }
                                ],
                                'function_call': None,
                                'provider_specific_fields': {'refusal': None},
                                'annotations': [],
                            },
                            {
                                'tool_call_id': 'call_SWFIWhfCI6AeHuaV6EM1MRsJ',
                                'role': 'tool',
                                'name': 'get_current_weather',
                                'content': '{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}',
                            },
                        ]
                    },
                    'response_data': {
                        'message': {
                            'content': 'The current temperature in San Francisco is 72°F. If you need more specific weather details or a forecast, let me know!',
                            'role': 'assistant',
                        }
                    },
                    'gen_ai.request.model': 'gpt-4o-mini',
                    'gen_ai.response.model': 'gpt-4o-mini',
                    'gen_ai.usage.input_tokens': 62,
                    'gen_ai.usage.output_tokens': 26,
                    'gen_ai.system': 'openai',
                    'logfire.tags': ['LLM'],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {'request_data': {'type': 'object'}, 'response_data': {'type': 'object'}},
                    },
                },
            },
        ]
    )


# --- tests/otel_integrations/test_openai.py ---

def test_sync_chat_completions(instrumented_client: openai.Client, exporter: TestExporter) -> None:
    response = instrumented_client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is four plus five?'},
        ],
    )
    assert response.choices[0].message.content == 'Nine'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_completions',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': 'What is four plus five?'},
                        ],
                        'model': 'gpt-4',
                    },
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'You are a helpful assistant.'}]},
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]},
                    ],
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'gen_ai.system': 'openai',
                    'logfire.msg': "Chat Completion with 'gpt-4'",
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'gpt-4',
                    'operation.cost': 0.00012,
                    'gen_ai.response.id': 'test_id',
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.usage.output_tokens': 1,
                    'gen_ai.usage.raw': {'completion_tokens': 1, 'prompt_tokens': 2, 'total_tokens': 3},
                    'response_data': {
                        'message': {
                            'content': 'Nine',
                            'refusal': None,
                            'audio': None,
                            'annotations': None,
                            'role': 'assistant',
                            'function_call': None,
                            'tool_calls': None,
                        },
                        'usage': {
                            'completion_tokens': 1,
                            'prompt_tokens': 2,
                            'total_tokens': 3,
                            'completion_tokens_details': None,
                            'prompt_tokens_details': None,
                        },
                    },
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Nine'}], 'finish_reason': 'stop'}
                    ],
                    'gen_ai.response.finish_reasons': ['stop'],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system': {},
                            'async': {},
                            'gen_ai.response.model': {},
                            'operation.cost': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'object',
                                        'title': 'ChatCompletionMessage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                    'usage': {
                                        'type': 'object',
                                        'title': 'CompletionUsage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                        },
                    },
                },
            }
        ]
    )

def test_sync_chat_completions_with_all_request_params(
    instrumented_client: openai.Client, exporter: TestExporter
) -> None:
    """Test that all optional request parameters are extracted to span attributes."""
    response = instrumented_client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role': 'user', 'content': 'What is four plus five?'},
        ],
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        stop=['END', 'STOP'],
        seed=42,
        frequency_penalty=0.5,
        presence_penalty=0.3,
    )
    assert response.choices[0].message.content == 'Nine'
    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    assert spans == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_completions_with_all_request_params',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'gpt-4',
                        'frequency_penalty': 0.5,
                        'max_tokens': 100,
                        'presence_penalty': 0.3,
                        'seed': 42,
                        'stop': ['END', 'STOP'],
                        'temperature': 0.7,
                        'top_p': 0.9,
                    },
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.max_tokens': 100,
                    'gen_ai.request.temperature': 0.7,
                    'gen_ai.request.top_p': 0.9,
                    'gen_ai.request.stop_sequences': ['END', 'STOP'],
                    'gen_ai.request.seed': 42,
                    'gen_ai.request.frequency_penalty': 0.5,
                    'gen_ai.request.presence_penalty': 0.3,
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.system': 'openai',
                    'gen_ai.response.model': 'gpt-4',
                    'operation.cost': 0.00012,
                    'gen_ai.response.id': 'test_id',
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.usage.output_tokens': 1,
                    'gen_ai.usage.raw': {'completion_tokens': 1, 'prompt_tokens': 2, 'total_tokens': 3},
                    'response_data': {
                        'message': {
                            'content': 'Nine',
                            'refusal': None,
                            'role': 'assistant',
                            'annotations': None,
                            'audio': None,
                            'function_call': None,
                            'tool_calls': None,
                        },
                        'usage': {
                            'completion_tokens': 1,
                            'prompt_tokens': 2,
                            'total_tokens': 3,
                            'completion_tokens_details': None,
                            'prompt_tokens_details': None,
                        },
                    },
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Nine'}], 'finish_reason': 'stop'}
                    ],
                    'gen_ai.response.finish_reasons': ['stop'],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.max_tokens': {},
                            'gen_ai.request.temperature': {},
                            'gen_ai.request.top_p': {},
                            'gen_ai.request.stop_sequences': {},
                            'gen_ai.request.seed': {},
                            'gen_ai.request.frequency_penalty': {},
                            'gen_ai.request.presence_penalty': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                            'gen_ai.system': {},
                            'gen_ai.response.model': {},
                            'operation.cost': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'object',
                                        'title': 'ChatCompletionMessage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                    'usage': {
                                        'type': 'object',
                                        'title': 'CompletionUsage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                        },
                    },
                },
            }
        ]
    )

def test_sync_chat_completions_with_stop_string(instrumented_client: openai.Client, exporter: TestExporter) -> None:
    """Test that stop as a string is properly converted to JSON array."""
    response = instrumented_client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role': 'user', 'content': 'What is four plus five?'},
        ],
        stop='END',
    )
    assert response.choices[0].message.content == 'Nine'
    spans = exporter.exported_spans_as_dict(parse_json_attributes=True)
    assert spans == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_completions_with_stop_string',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [{'role': 'user', 'content': 'What is four plus five?'}],
                        'model': 'gpt-4',
                        'stop': 'END',
                    },
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.stop_sequences': ['END'],
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.system': 'openai',
                    'gen_ai.response.model': 'gpt-4',
                    'operation.cost': 0.00012,
                    'gen_ai.response.id': 'test_id',
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.usage.output_tokens': 1,
                    'gen_ai.usage.raw': {'completion_tokens': 1, 'prompt_tokens': 2, 'total_tokens': 3},
                    'response_data': {
                        'message': {
                            'content': 'Nine',
                            'refusal': None,
                            'role': 'assistant',
                            'annotations': None,
                            'audio': None,
                            'function_call': None,
                            'tool_calls': None,
                        },
                        'usage': {
                            'completion_tokens': 1,
                            'prompt_tokens': 2,
                            'total_tokens': 3,
                            'completion_tokens_details': None,
                            'prompt_tokens_details': None,
                        },
                    },
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Nine'}], 'finish_reason': 'stop'}
                    ],
                    'gen_ai.response.finish_reasons': ['stop'],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.stop_sequences': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                            'gen_ai.system': {},
                            'gen_ai.response.model': {},
                            'operation.cost': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'object',
                                        'title': 'ChatCompletionMessage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                    'usage': {
                                        'type': 'object',
                                        'title': 'CompletionUsage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                        },
                    },
                },
            }
        ]
    )

async def test_async_chat_completions(instrumented_async_client: openai.AsyncClient, exporter: TestExporter) -> None:
    response = await instrumented_async_client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is four plus five?'},
        ],
    )
    assert response.choices[0].message.content == 'Nine'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_async_chat_completions',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': 'What is four plus five?'},
                        ],
                        'model': 'gpt-4',
                    },
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'You are a helpful assistant.'}]},
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]},
                    ],
                    'async': True,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'gen_ai.system': 'openai',
                    'logfire.msg': "Chat Completion with 'gpt-4'",
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'gpt-4',
                    'operation.cost': 0.00012,
                    'gen_ai.response.id': 'test_id',
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.usage.output_tokens': 1,
                    'gen_ai.usage.raw': {'completion_tokens': 1, 'prompt_tokens': 2, 'total_tokens': 3},
                    'response_data': {
                        'message': {
                            'content': 'Nine',
                            'refusal': None,
                            'audio': None,
                            'annotations': None,
                            'role': 'assistant',
                            'function_call': None,
                            'tool_calls': None,
                        },
                        'usage': {
                            'completion_tokens': 1,
                            'prompt_tokens': 2,
                            'total_tokens': 3,
                            'completion_tokens_details': None,
                            'prompt_tokens_details': None,
                        },
                    },
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Nine'}], 'finish_reason': 'stop'}
                    ],
                    'gen_ai.response.finish_reasons': ['stop'],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system': {},
                            'async': {},
                            'gen_ai.response.model': {},
                            'operation.cost': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'object',
                                        'title': 'ChatCompletionMessage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                    'usage': {
                                        'type': 'object',
                                        'title': 'CompletionUsage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                        },
                    },
                },
            }
        ]
    )

def test_sync_chat_empty_response_chunk(instrumented_client: openai.Client, exporter: TestExporter) -> None:
    response = instrumented_client.chat.completions.create(
        model='gpt-4',
        messages=[{'role': 'system', 'content': 'empty response chunk'}],
        stream=True,
    )
    combined = [chunk for chunk in response]
    assert combined == [[]]
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_empty_response_chunk',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [{'role': 'system', 'content': 'empty response chunk'}],
                        'model': 'gpt-4',
                        'stream': True,
                    },
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'empty response chunk'}]}
                    ],
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'gpt-4',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.level_num': 9,
                    'request_data': {
                        'messages': [{'role': 'system', 'content': 'empty response chunk'}],
                        'model': 'gpt-4',
                        'stream': True,
                    },
                    'async': False,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_empty_response_chunk',
                    'code.lineno': 123,
                    'logfire.msg': "streaming response from 'gpt-4' took 1.00s",
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'logfire.span_type': 'log',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'empty response chunk'}]}
                    ],
                    'logfire.tags': ('LLM',),
                    'duration': 1.0,
                    'response_data': {'combined_chunk_content': '', 'chunk_count': 0},
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'async': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'duration': {},
                            'response_data': {'type': 'object'},
                        },
                    },
                    'gen_ai.response.model': 'gpt-4',
                },
            },
        ]
    )

def test_sync_chat_empty_response_choices(instrumented_client: openai.Client, exporter: TestExporter) -> None:
    response = instrumented_client.chat.completions.create(
        model='gpt-4',
        messages=[{'role': 'system', 'content': 'empty choices in response chunk'}],
        stream=True,
    )
    combined = [chunk for chunk in response]
    assert len(combined) == 1
    assert combined[0].choices == []
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_empty_response_choices',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [{'role': 'system', 'content': 'empty choices in response chunk'}],
                        'model': 'gpt-4',
                        'stream': True,
                    },
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'empty choices in response chunk'}]}
                    ],
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'gpt-4',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.level_num': 9,
                    'request_data': {
                        'messages': [{'role': 'system', 'content': 'empty choices in response chunk'}],
                        'model': 'gpt-4',
                        'stream': True,
                    },
                    'async': False,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_empty_response_choices',
                    'code.lineno': 123,
                    'logfire.msg': "streaming response from 'gpt-4' took 1.00s",
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'logfire.span_type': 'log',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'empty choices in response chunk'}]}
                    ],
                    'logfire.tags': ('LLM',),
                    'duration': 1.0,
                    'response_data': {'message': None, 'usage': None},
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'async': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'duration': {},
                            'response_data': {'type': 'object'},
                        },
                    },
                    'gen_ai.response.model': 'gpt-4',
                },
            },
        ]
    )

def test_sync_chat_tool_call_stream(instrumented_client: openai.Client, exporter: TestExporter) -> None:
    response = instrumented_client.chat.completions.create(
        model='gpt-4',
        messages=[{'role': 'system', 'content': 'streamed tool call'}],
        stream=True,
        stream_options={'include_usage': True},
        tool_choice={'type': 'function', 'function': {'name': 'get_current_weather'}},
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'get_current_weather',
                    'description': 'Get the current weather in a given location',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'location': {
                                'type': 'string',
                                'description': 'The city and state, e.g. San Francisco, CA',
                            },
                            'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                        },
                        'required': ['location'],
                    },
                },
            },
        ],
    )
    combined_arguments = ''.join(
        chunk.choices[0].delta.tool_calls[0].function.arguments
        for chunk in response
        if chunk.choices
        and chunk.choices[0].delta.tool_calls
        and chunk.choices[0].delta.tool_calls[0].function
        and chunk.choices[0].delta.tool_calls[0].function.arguments
    )
    assert combined_arguments == '{"location":"Boston"}'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_tool_call_stream',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [{'role': 'system', 'content': 'streamed tool call'}],
                        'model': 'gpt-4',
                        'stream': True,
                        'stream_options': {'include_usage': True},
                        'tool_choice': {'type': 'function', 'function': {'name': 'get_current_weather'}},
                        'tools': [
                            {
                                'type': 'function',
                                'function': {
                                    'name': 'get_current_weather',
                                    'description': 'Get the current weather in a given location',
                                    'parameters': {
                                        'type': 'object',
                                        'properties': {
                                            'location': {
                                                'type': 'string',
                                                'description': 'The city and state, e.g. San Francisco, CA',
                                            },
                                            'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                        },
                                        'required': ['location'],
                                    },
                                },
                            }
                        ],
                    },
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.tool.definitions': [
                        {
                            'type': 'function',
                            'function': {
                                'name': 'get_current_weather',
                                'description': 'Get the current weather in a given location',
                                'parameters': {
                                    'type': 'object',
                                    'properties': {
                                        'location': {
                                            'type': 'string',
                                            'description': 'The city and state, e.g. San Francisco, CA',
                                        },
                                        'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                    },
                                    'required': ['location'],
                                },
                            },
                        }
                    ],
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'streamed tool call'}]}
                    ],
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.tool.definitions': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'gpt-4',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'logfire.msg': "streaming response from 'gpt-4' took 1.00s",
                    'code.filepath': 'test_openai.py',
                    'code.function': '<genexpr>',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [{'role': 'system', 'content': 'streamed tool call'}],
                        'model': 'gpt-4',
                        'stream': True,
                        'stream_options': {'include_usage': True},
                        'tool_choice': {'type': 'function', 'function': {'name': 'get_current_weather'}},
                        'tools': [
                            {
                                'type': 'function',
                                'function': {
                                    'name': 'get_current_weather',
                                    'description': 'Get the current weather in a given location',
                                    'parameters': {
                                        'type': 'object',
                                        'properties': {
                                            'location': {
                                                'type': 'string',
                                                'description': 'The city and state, e.g. San Francisco, CA',
                                            },
                                            'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                        },
                                        'required': ['location'],
                                    },
                                },
                            }
                        ],
                    },
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'async': False,
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.tool.definitions': [
                        {
                            'type': 'function',
                            'function': {
                                'name': 'get_current_weather',
                                'description': 'Get the current weather in a given location',
                                'parameters': {
                                    'type': 'object',
                                    'properties': {
                                        'location': {
                                            'type': 'string',
                                            'description': 'The city and state, e.g. San Francisco, CA',
                                        },
                                        'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                    },
                                    'required': ['location'],
                                },
                            },
                        }
                    ],
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'streamed tool call'}]}
                    ],
                    'duration': 1.0,
                    'response_data': {
                        'message': {
                            'content': None,
                            'refusal': None,
                            'role': 'assistant',
                            'annotations': None,
                            'audio': None,
                            'function_call': None,
                            'tool_calls': [
                                {
                                    'id': '1',
                                    'function': {
                                        'arguments': '{"location":"Boston"}',
                                        'name': 'get_current_weather',
                                        'parsed_arguments': None,
                                    },
                                    'type': 'function',
                                    'index': 0,
                                }
                            ],
                            'parsed': None,
                        },
                        'usage': {
                            'completion_tokens': 1,
                            'prompt_tokens': 2,
                            'total_tokens': 3,
                            'completion_tokens_details': None,
                            'prompt_tokens_details': None,
                        },
                    },
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': '1',
                                    'name': 'get_current_weather',
                                    'arguments': {'location': 'Boston'},
                                }
                            ],
                            'finish_reason': 'stop',
                        }
                    ],
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.usage.output_tokens': 1,
                    'gen_ai.usage.raw': {'completion_tokens': 1, 'prompt_tokens': 2, 'total_tokens': 3},
                    'operation.cost': 0.00012,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'async': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.tool.definitions': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'duration': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'object',
                                        'title': 'ParsedChatCompletionMessage[object]',
                                        'x-python-datatype': 'PydanticModel',
                                        'properties': {
                                            'tool_calls': {
                                                'type': 'array',
                                                'items': {
                                                    'type': 'object',
                                                    'title': 'ParsedFunctionToolCall',
                                                    'x-python-datatype': 'PydanticModel',
                                                    'properties': {
                                                        'function': {
                                                            'type': 'object',
                                                            'title': 'ParsedFunction',
                                                            'x-python-datatype': 'PydanticModel',
                                                        }
                                                    },
                                                },
                                            }
                                        },
                                    },
                                    'usage': {
                                        'type': 'object',
                                        'title': 'CompletionUsage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'operation.cost': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'gpt-4',
                },
            },
        ]
    )

async def test_async_chat_tool_call_stream(
    instrumented_async_client: openai.AsyncClient, exporter: TestExporter
) -> None:
    response = await instrumented_async_client.chat.completions.create(
        model='gpt-4',
        messages=[{'role': 'system', 'content': 'streamed tool call'}],
        stream=True,
        stream_options={'include_usage': True},
        tool_choice={'type': 'function', 'function': {'name': 'get_current_weather'}},
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'get_current_weather',
                    'description': 'Get the current weather in a given location',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'location': {
                                'type': 'string',
                                'description': 'The city and state, e.g. San Francisco, CA',
                            },
                            'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                        },
                        'required': ['location'],
                    },
                },
            },
        ],
    )
    combined_arguments = ''.join(
        [
            chunk.choices[0].delta.tool_calls[0].function.arguments
            async for chunk in response
            if chunk.choices
            and chunk.choices[0].delta.tool_calls
            and chunk.choices[0].delta.tool_calls[0].function
            and chunk.choices[0].delta.tool_calls[0].function.arguments
        ]
    )
    assert combined_arguments == '{"location":"Boston"}'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_async_chat_tool_call_stream',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [{'role': 'system', 'content': 'streamed tool call'}],
                        'model': 'gpt-4',
                        'stream': True,
                        'stream_options': {'include_usage': True},
                        'tool_choice': {'type': 'function', 'function': {'name': 'get_current_weather'}},
                        'tools': [
                            {
                                'type': 'function',
                                'function': {
                                    'name': 'get_current_weather',
                                    'description': 'Get the current weather in a given location',
                                    'parameters': {
                                        'type': 'object',
                                        'properties': {
                                            'location': {
                                                'type': 'string',
                                                'description': 'The city and state, e.g. San Francisco, CA',
                                            },
                                            'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                        },
                                        'required': ['location'],
                                    },
                                },
                            }
                        ],
                    },
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.tool.definitions': [
                        {
                            'type': 'function',
                            'function': {
                                'name': 'get_current_weather',
                                'description': 'Get the current weather in a given location',
                                'parameters': {
                                    'type': 'object',
                                    'properties': {
                                        'location': {
                                            'type': 'string',
                                            'description': 'The city and state, e.g. San Francisco, CA',
                                        },
                                        'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                    },
                                    'required': ['location'],
                                },
                            },
                        }
                    ],
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'streamed tool call'}]}
                    ],
                    'async': True,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.tool.definitions': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'gpt-4',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'logfire.msg': "streaming response from 'gpt-4' took 1.00s",
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_async_chat_tool_call_stream',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [{'role': 'system', 'content': 'streamed tool call'}],
                        'model': 'gpt-4',
                        'stream': True,
                        'stream_options': {'include_usage': True},
                        'tool_choice': {'type': 'function', 'function': {'name': 'get_current_weather'}},
                        'tools': [
                            {
                                'type': 'function',
                                'function': {
                                    'name': 'get_current_weather',
                                    'description': 'Get the current weather in a given location',
                                    'parameters': {
                                        'type': 'object',
                                        'properties': {
                                            'location': {
                                                'type': 'string',
                                                'description': 'The city and state, e.g. San Francisco, CA',
                                            },
                                            'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                        },
                                        'required': ['location'],
                                    },
                                },
                            }
                        ],
                    },
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'async': True,
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.tool.definitions': [
                        {
                            'type': 'function',
                            'function': {
                                'name': 'get_current_weather',
                                'description': 'Get the current weather in a given location',
                                'parameters': {
                                    'type': 'object',
                                    'properties': {
                                        'location': {
                                            'type': 'string',
                                            'description': 'The city and state, e.g. San Francisco, CA',
                                        },
                                        'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                    },
                                    'required': ['location'],
                                },
                            },
                        }
                    ],
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'streamed tool call'}]}
                    ],
                    'duration': 1.0,
                    'response_data': {
                        'message': {
                            'content': None,
                            'refusal': None,
                            'role': 'assistant',
                            'annotations': None,
                            'audio': None,
                            'function_call': None,
                            'tool_calls': [
                                {
                                    'id': '1',
                                    'function': {
                                        'arguments': '{"location":"Boston"}',
                                        'name': 'get_current_weather',
                                        'parsed_arguments': None,
                                    },
                                    'type': 'function',
                                    'index': 0,
                                }
                            ],
                            'parsed': None,
                        },
                        'usage': {
                            'completion_tokens': 1,
                            'prompt_tokens': 2,
                            'total_tokens': 3,
                            'completion_tokens_details': None,
                            'prompt_tokens_details': None,
                        },
                    },
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': '1',
                                    'name': 'get_current_weather',
                                    'arguments': {'location': 'Boston'},
                                }
                            ],
                            'finish_reason': 'stop',
                        }
                    ],
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.usage.output_tokens': 1,
                    'gen_ai.usage.raw': {'completion_tokens': 1, 'prompt_tokens': 2, 'total_tokens': 3},
                    'operation.cost': 0.00012,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'async': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.tool.definitions': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'duration': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'object',
                                        'title': 'ParsedChatCompletionMessage[object]',
                                        'x-python-datatype': 'PydanticModel',
                                        'properties': {
                                            'tool_calls': {
                                                'type': 'array',
                                                'items': {
                                                    'type': 'object',
                                                    'title': 'ParsedFunctionToolCall',
                                                    'x-python-datatype': 'PydanticModel',
                                                    'properties': {
                                                        'function': {
                                                            'type': 'object',
                                                            'title': 'ParsedFunction',
                                                            'x-python-datatype': 'PydanticModel',
                                                        }
                                                    },
                                                },
                                            }
                                        },
                                    },
                                    'usage': {
                                        'type': 'object',
                                        'title': 'CompletionUsage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'operation.cost': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'gpt-4',
                },
            },
        ]
    )

def test_sync_chat_completions_stream(instrumented_client: openai.Client, exporter: TestExporter) -> None:
    response = instrumented_client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is four plus five?'},
        ],
        stream=True,
    )
    combined = ''.join(chunk.choices[0].delta.content for chunk in response if chunk.choices[0].delta.content)
    assert combined == 'The answer is secret'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_completions_stream',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': 'What is four plus five?'},
                        ],
                        'model': 'gpt-4',
                        'stream': True,
                    },
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'You are a helpful assistant.'}]},
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]},
                    ],
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'gpt-4',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.level_num': 9,
                    'request_data': {
                        'messages': [
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': 'What is four plus five?'},
                        ],
                        'model': 'gpt-4',
                        'stream': True,
                    },
                    'async': False,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'code.filepath': 'test_openai.py',
                    'code.function': '<genexpr>',
                    'code.lineno': 123,
                    'logfire.msg': "streaming response from 'gpt-4' took 1.00s",
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'logfire.span_type': 'log',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'You are a helpful assistant.'}]},
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]},
                    ],
                    'logfire.tags': ('LLM',),
                    'duration': 1.0,
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'The answer is secret'}]}
                    ],
                    'response_data': {
                        'message': {
                            'content': 'The answer is secret',
                            'refusal': None,
                            'role': 'assistant',
                            'annotations': None,
                            'audio': None,
                            'function_call': None,
                            'tool_calls': None,
                            'parsed': None,
                        },
                        'usage': None,
                    },
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'async': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'duration': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'object',
                                        'title': 'ParsedChatCompletionMessage[object]',
                                        'x-python-datatype': 'PydanticModel',
                                    }
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                        },
                    },
                    'gen_ai.response.model': 'gpt-4',
                },
            },
        ]
    )

async def test_async_chat_completions_stream(
    instrumented_async_client: openai.AsyncClient, exporter: TestExporter
) -> None:
    response = await instrumented_async_client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is four plus five?'},
        ],
        stream=True,
    )
    chunk_content = [chunk.choices[0].delta.content async for chunk in response if chunk.choices[0].delta.content]
    combined = ''.join(chunk_content)
    assert combined == 'The answer is secret'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_async_chat_completions_stream',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': 'What is four plus five?'},
                        ],
                        'model': 'gpt-4',
                        'stream': True,
                    },
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'You are a helpful assistant.'}]},
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]},
                    ],
                    'async': True,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'gpt-4',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.level_num': 9,
                    'request_data': {
                        'messages': [
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': 'What is four plus five?'},
                        ],
                        'model': 'gpt-4',
                        'stream': True,
                    },
                    'async': True,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_async_chat_completions_stream',
                    'code.lineno': 123,
                    'logfire.msg': "streaming response from 'gpt-4' took 1.00s",
                    'gen_ai.request.model': 'gpt-4',
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'logfire.span_type': 'log',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'You are a helpful assistant.'}]},
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]},
                    ],
                    'logfire.tags': ('LLM',),
                    'duration': 1.0,
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'The answer is secret'}]}
                    ],
                    'response_data': {
                        'message': {
                            'content': 'The answer is secret',
                            'refusal': None,
                            'role': 'assistant',
                            'annotations': None,
                            'audio': None,
                            'function_call': None,
                            'tool_calls': None,
                            'parsed': None,
                        },
                        'usage': None,
                    },
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'async': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'duration': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'object',
                                        'title': 'ParsedChatCompletionMessage[object]',
                                        'x-python-datatype': 'PydanticModel',
                                    }
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                        },
                    },
                    'gen_ai.response.model': 'gpt-4',
                },
            },
        ]
    )

def test_completions(instrumented_client: openai.Client, exporter: TestExporter) -> None:
    response = instrumented_client.completions.create(
        model='gpt-3.5-turbo-instruct',
        prompt='What is four plus five?',
    )
    assert response.choices[0].text == 'Nine'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_completions',
                    'code.lineno': 123,
                    'request_data': {'model': 'gpt-3.5-turbo-instruct', 'prompt': 'What is four plus five?'},
                    'gen_ai.provider.name': 'openai',
                    'async': False,
                    'gen_ai.operation.name': 'text_completion',
                    'logfire.msg_template': 'Completion with {request_data[model]!r}',
                    'logfire.msg': "Completion with 'gpt-3.5-turbo-instruct'",
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'gen_ai.system': 'openai',
                    'gen_ai.request.model': 'gpt-3.5-turbo-instruct',
                    'gen_ai.response.model': 'gpt-3.5-turbo-instruct',
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.response.id': 'test_id',
                    'gen_ai.usage.output_tokens': 1,
                    'operation.cost': 5e-06,
                    'gen_ai.usage.raw': {'completion_tokens': 1, 'prompt_tokens': 2, 'total_tokens': 3},
                    'response_data': {
                        'finish_reason': 'stop',
                        'text': 'Nine',
                        'usage': {
                            'completion_tokens': 1,
                            'prompt_tokens': 2,
                            'total_tokens': 3,
                            'completion_tokens_details': None,
                            'prompt_tokens_details': None,
                        },
                    },
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Nine'}], 'finish_reason': 'stop'}
                    ],
                    'gen_ai.response.finish_reasons': ['stop'],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'async': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.system': {},
                            'gen_ai.request.model': {},
                            'gen_ai.response.model': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.output_tokens': {},
                            'operation.cost': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'usage': {
                                        'type': 'object',
                                        'title': 'CompletionUsage',
                                        'x-python-datatype': 'PydanticModel',
                                    }
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                        },
                    },
                },
            }
        ]
    )

def test_sync_chat_completions_version_latest(exporter: TestExporter) -> None:
    """Test that version='latest' uses semconv attributes with minimal request_data and no response_data."""
    client = openai.Client(api_key='foobar')
    logfire.instrument_openai(client, version='latest')
    response = client.chat.completions.create(
        model='gpt-4.1',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is four plus five?'},
        ],
    )
    assert response.choices[0].message.content == 'Four plus five is nine.'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_completions_version_latest',
                    'code.lineno': 123,
                    'request_data': {'model': 'gpt-4.1'},
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'gpt-4.1',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'You are a helpful assistant.'}]},
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]},
                    ],
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4.1'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.system': 'openai',
                    'gen_ai.response.model': IsStr(),
                    'operation.cost': IsNumeric(),
                    'gen_ai.response.id': IsStr(),
                    'gen_ai.usage.input_tokens': IsInt(),
                    'gen_ai.usage.output_tokens': IsInt(),
                    'gen_ai.usage.raw': {
                        'completion_tokens': 6,
                        'prompt_tokens': 23,
                        'total_tokens': 29,
                        'completion_tokens_details': {
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                        'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0},
                    },
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'text', 'content': 'Four plus five is nine.'}],
                            'finish_reason': 'stop',
                        }
                    ],
                    'gen_ai.response.finish_reasons': ['stop'],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                            'gen_ai.system': {},
                            'gen_ai.response.model': {},
                            'operation.cost': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                        },
                    },
                },
            }
        ]
    )

def test_sync_chat_completions_version_v1_only(exporter: TestExporter) -> None:
    """Test that version=1 does not emit gen_ai.input.messages or gen_ai.output.messages."""
    client = openai.Client(api_key='foobar')
    logfire.instrument_openai(client, version=1)
    response = client.chat.completions.create(
        model='gpt-4.1',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is four plus five?'},
        ],
    )
    assert response.choices[0].message.content == 'Four plus five is nine.'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_completions_version_v1_only',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': 'What is four plus five?'},
                        ],
                        'model': 'gpt-4.1',
                    },
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'gpt-4.1',
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4.1'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.system': 'openai',
                    'gen_ai.response.model': IsStr(),
                    'operation.cost': IsNumeric(),
                    'gen_ai.response.id': IsStr(),
                    'gen_ai.usage.input_tokens': IsInt(),
                    'gen_ai.usage.output_tokens': IsInt(),
                    'gen_ai.usage.raw': {
                        'completion_tokens': 6,
                        'prompt_tokens': 23,
                        'total_tokens': 29,
                        'completion_tokens_details': {
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                        'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0},
                    },
                    'response_data': {
                        'message': {
                            'content': 'Four plus five is nine.',
                            'refusal': None,
                            'role': 'assistant',
                            'annotations': [],
                            'audio': None,
                            'function_call': None,
                            'tool_calls': None,
                        },
                        'usage': {
                            'completion_tokens': IsInt(),
                            'prompt_tokens': IsInt(),
                            'total_tokens': IsInt(),
                            'completion_tokens_details': {
                                'accepted_prediction_tokens': 0,
                                'audio_tokens': 0,
                                'reasoning_tokens': 0,
                                'rejected_prediction_tokens': 0,
                            },
                            'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0},
                        },
                    },
                    'gen_ai.response.finish_reasons': ['stop'],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'async': {},
                            'gen_ai.system': {},
                            'gen_ai.response.model': {},
                            'operation.cost': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'object',
                                        'title': 'ChatCompletionMessage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                    'usage': {
                                        'type': 'object',
                                        'title': 'CompletionUsage',
                                        'x-python-datatype': 'PydanticModel',
                                        'properties': {
                                            'completion_tokens_details': {
                                                'type': 'object',
                                                'title': 'CompletionTokensDetails',
                                                'x-python-datatype': 'PydanticModel',
                                            },
                                            'prompt_tokens_details': {
                                                'type': 'object',
                                                'title': 'PromptTokensDetails',
                                                'x-python-datatype': 'PydanticModel',
                                            },
                                        },
                                    },
                                },
                            },
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                        },
                    },
                },
            }
        ]
    )

def test_sync_chat_completions_stream_version_latest(exporter: TestExporter) -> None:
    """Test that streaming with version='latest' emits semconv attributes without response_data."""
    client = openai.Client(api_key='foobar')
    logfire.instrument_openai(client, version='latest')
    response = client.chat.completions.create(
        model='gpt-4.1',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is four plus five?'},
        ],
        stream=True,
    )
    combined = ''.join(chunk.choices[0].delta.content for chunk in response if chunk.choices[0].delta.content)
    assert combined == 'Four plus five is nine.'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_completions_stream_version_latest',
                    'code.lineno': 123,
                    'request_data': {'model': 'gpt-4.1'},
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'gpt-4.1',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'You are a helpful assistant.'}]},
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]},
                    ],
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4.1'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': IsStr(),
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'logfire.msg': "streaming response from 'gpt-4.1' took 1.00s",
                    'code.filepath': 'test_openai.py',
                    'code.function': '<genexpr>',
                    'code.lineno': 123,
                    'duration': 1.0,
                    'request_data': {'model': 'gpt-4.1'},
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'gpt-4.1',
                    'gen_ai.input.messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'You are a helpful assistant.'}]},
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]},
                    ],
                    'async': False,
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'text', 'content': 'Four plus five is nine.'}],
                            'finish_reason': 'stop',
                        }
                    ],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'duration': {},
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                            'gen_ai.output.messages': {'type': 'array'},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': IsStr(),
                },
            },
        ]
    )

def test_sync_chat_completions_stream_version_v1_only(exporter: TestExporter) -> None:
    """Test that streaming with version=1 emits response_data without semconv message attributes."""
    client = openai.Client(api_key='foobar')
    logfire.instrument_openai(client, version=1)
    response = client.chat.completions.create(
        model='gpt-4.1',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is four plus five?'},
        ],
        stream=True,
    )
    combined = ''.join(chunk.choices[0].delta.content for chunk in response if chunk.choices[0].delta.content)
    assert combined == 'Four plus five is nine.'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_sync_chat_completions_stream_version_v1_only',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': 'What is four plus five?'},
                        ],
                        'model': 'gpt-4.1',
                        'stream': True,
                    },
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'gpt-4.1',
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4.1'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'async': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': IsStr(),
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'logfire.msg': "streaming response from 'gpt-4.1' took 1.00s",
                    'code.filepath': 'test_openai.py',
                    'code.function': '<genexpr>',
                    'code.lineno': 123,
                    'duration': 1.0,
                    'request_data': {
                        'messages': [
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': 'What is four plus five?'},
                        ],
                        'model': 'gpt-4.1',
                        'stream': True,
                    },
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'gpt-4.1',
                    'async': False,
                    'response_data': {
                        'message': {
                            'content': 'Four plus five is nine.',
                            'refusal': None,
                            'role': 'assistant',
                            'annotations': None,
                            'audio': None,
                            'function_call': None,
                            'tool_calls': None,
                            'parsed': None,
                        },
                        'usage': None,
                    },
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'duration': {},
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'async': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'object',
                                        'title': 'ParsedChatCompletionMessage[object]',
                                        'x-python-datatype': 'PydanticModel',
                                    }
                                },
                            },
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': IsStr(),
                },
            },
        ]
    )

def test_responses_stream(exporter: TestExporter) -> None:
    client = openai.Client(api_key='foobar')
    logfire.instrument_openai(client, version=[1, 'latest'])
    with client.responses.stream(
        model='gpt-4.1',
        input='What is four plus five?',
    ) as stream:
        for _ in stream:
            pass

        final_response = stream.get_final_response()

    assert final_response.output_text == snapshot('Four plus five equals **nine**.')
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_responses_stream',
                    'code.lineno': 123,
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.system': 'openai',
                    'events': [
                        {'event.name': 'gen_ai.user.message', 'content': 'What is four plus five?', 'role': 'user'}
                    ],
                    'request_data': {'model': 'gpt-4.1', 'stream': True},
                    'gen_ai.request.model': 'gpt-4.1',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'async': False,
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.msg': "Responses API with 'gpt-4.1'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.provider.name': {},
                            'gen_ai.system': {},
                            'events': {'type': 'array'},
                            'request_data': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'gpt-4.1',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'logfire.msg': "streaming response from 'gpt-4.1' took 1.00s",
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_responses_stream',
                    'code.lineno': 123,
                    'request_data': {'model': 'gpt-4.1', 'stream': True},
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.system': 'openai',
                    'events': [
                        {'event.name': 'gen_ai.user.message', 'content': 'What is four plus five?', 'role': 'user'},
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'content': 'Four plus five equals **nine**.',
                            'role': 'assistant',
                        },
                    ],
                    'gen_ai.request.model': 'gpt-4.1',
                    'async': False,
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'duration': 1.0,
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Four plus five equals **nine**.'}]}
                    ],
                    'gen_ai.usage.input_tokens': 13,
                    'gen_ai.usage.output_tokens': 9,
                    'gen_ai.usage.raw': {
                        'input_tokens': 13,
                        'input_tokens_details': {'cached_tokens': 0},
                        'output_tokens': 9,
                        'output_tokens_details': {'reasoning_tokens': 0},
                        'total_tokens': 22,
                    },
                    'operation.cost': 9.8e-05,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'gen_ai.system': {},
                            'events': {'type': 'array'},
                            'gen_ai.request.model': {},
                            'async': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'duration': {},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'operation.cost': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'gpt-4.1',
                },
            },
        ]
    )

def test_responses_stream_error_propagates() -> None:
    """Test that streaming errors propagate and aren't masked by a secondary error.

    When streaming fails (e.g., API error), the original error should propagate.
    Previously, get_response_data() would raise a RuntimeError when no response.completed
    event was received, which would mask the original error. Now the original error propagates.
    """
    from opentelemetry.context import get_current

    from logfire._internal.integrations.llm_providers.llm_provider import record_streaming
    from logfire._internal.integrations.llm_providers.openai import OpenaiResponsesStreamState

    class StreamingError(Exception):
        pass

    span_data = {
        'events': [{'event.name': 'gen_ai.user.message', 'content': 'Hello', 'role': 'user'}],
        'request_data': {'model': 'gpt-4.1'},
    }

    with pytest.raises(StreamingError):
        with record_streaming(
            logfire.DEFAULT_LOGFIRE_INSTANCE,
            span_data,
            OpenaiResponsesStreamState,
            get_current(),
        ):
            raise StreamingError('API connection lost')

def test_completions_stream(instrumented_client: openai.Client, exporter: TestExporter) -> None:
    response = instrumented_client.completions.create(
        model='gpt-3.5-turbo-instruct',
        prompt='What is four plus five?',
        stream=True,
    )
    combined = ''.join(chunk.choices[0].text for chunk in response if chunk.choices[0].text)
    assert combined == 'The answer is Nine'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_completions_stream',
                    'code.lineno': 123,
                    'request_data': {
                        'model': 'gpt-3.5-turbo-instruct',
                        'prompt': 'What is four plus five?',
                        'stream': True,
                    },
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.request.model': 'gpt-3.5-turbo-instruct',
                    'gen_ai.operation.name': 'text_completion',
                    'async': False,
                    'logfire.msg_template': 'Completion with {request_data[model]!r}',
                    'logfire.msg': "Completion with 'gpt-3.5-turbo-instruct'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.operation.name': {},
                            'async': {},
                        },
                    },
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'gpt-3.5-turbo-instruct',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.level_num': 9,
                    'request_data': {
                        'model': 'gpt-3.5-turbo-instruct',
                        'prompt': 'What is four plus five?',
                        'stream': True,
                    },
                    'async': False,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'code.filepath': 'test_openai.py',
                    'code.function': '<genexpr>',
                    'code.lineno': 123,
                    'logfire.msg': "streaming response from 'gpt-3.5-turbo-instruct' took 1.00s",
                    'gen_ai.request.model': 'gpt-3.5-turbo-instruct',
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'logfire.span_type': 'log',
                    'gen_ai.operation.name': 'text_completion',
                    'logfire.tags': ('LLM',),
                    'duration': 1.0,
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'The answer is Nine'}]}
                    ],
                    'response_data': {'combined_chunk_content': 'The answer is Nine', 'chunk_count': 2},
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'async': {},
                            'gen_ai.operation.name': {},
                            'duration': {},
                            'response_data': {'type': 'object'},
                            'gen_ai.output.messages': {'type': 'array'},
                        },
                    },
                    'gen_ai.response.model': 'gpt-3.5-turbo-instruct',
                },
            },
        ]
    )

def test_images(instrumented_client: openai.Client, exporter: TestExporter) -> None:
    response = instrumented_client.images.generate(
        model='dall-e-3',
        prompt='A picture of a cat.',
    )
    assert response.data
    assert response.data[0].revised_prompt == 'revised prompt'
    assert response.data[0].url == 'https://example.com/image.jpg'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Image Generation with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_images',
                    'code.lineno': 123,
                    'request_data': {'prompt': 'A picture of a cat.', 'model': 'dall-e-3'},
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.request.model': 'dall-e-3',
                    'gen_ai.operation.name': 'image_generation',
                    'async': False,
                    'logfire.msg_template': 'Image Generation with {request_data[model]!r}',
                    'logfire.msg': "Image Generation with 'dall-e-3'",
                    'logfire.span_type': 'span',
                    'gen_ai.system': 'openai',
                    'logfire.tags': ('LLM',),
                    'response_data': {
                        'images': [
                            {
                                'b64_json': None,
                                'revised_prompt': 'revised prompt',
                                'url': 'https://example.com/image.jpg',
                            }
                        ]
                    },
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.system': {},
                            'async': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'images': {
                                        'type': 'array',
                                        'items': {
                                            'type': 'object',
                                            'title': 'Image',
                                            'x-python-datatype': 'PydanticModel',
                                        },
                                    }
                                },
                            },
                        },
                    },
                    'gen_ai.response.model': 'dall-e-3',
                },
            }
        ]
    )

def test_openai_suppressed(instrumented_client: openai.Client, exporter: TestExporter) -> None:
    with suppress_instrumentation():
        response = instrumented_client.completions.create(model='gpt-3.5-turbo-instruct', prompt='xxx')
    assert response.choices[0].text == 'Nine'
    assert (
        exporter.exported_spans_as_dict(
            parse_json_attributes=True,
        )
        == []
    )

async def test_async_openai_suppressed(instrumented_async_client: openai.AsyncClient, exporter: TestExporter) -> None:
    with suppress_instrumentation():
        response = await instrumented_async_client.completions.create(model='gpt-3.5-turbo-instruct', prompt='xxx')
    assert response.choices[0].text == 'Nine'
    assert (
        exporter.exported_spans_as_dict(
            parse_json_attributes=True,
        )
        == []
    )

def test_create_assistant(instrumented_client: openai.Client, exporter: TestExporter) -> None:
    assistant = instrumented_client.beta.assistants.create(  # pyright: ignore[reportDeprecated]
        name='Math Tutor',
        instructions='You are a personal math tutor. Write and run code to answer math questions.',
        tools=[{'type': 'code_interpreter'}],
        model='gpt-4o',
    )
    assert assistant.name == 'Math Tutor'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'OpenAI API call to {url!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'logfire.span_type': 'span',
                    'logfire.tags': ('LLM',),
                    'request_data': (
                        {
                            'model': 'gpt-4o',
                            'instructions': 'You are a personal math tutor. Write and run code to answer math questions.',
                            'name': 'Math Tutor',
                            'tools': [{'type': 'code_interpreter'}],
                        }
                    ),
                    'url': '/assistants',
                    'async': False,
                    'gen_ai.provider.name': 'openai',
                    'logfire.msg_template': 'OpenAI API call to {url!r}',
                    'gen_ai.tool.definitions': [{'type': 'code_interpreter'}],
                    'logfire.msg': "OpenAI API call to '/assistants'",
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_create_assistant',
                    'code.lineno': 123,
                    'gen_ai.request.model': 'gpt-4o',
                    'gen_ai.system': 'openai',
                    'gen_ai.response.model': 'gpt-4-turbo',
                    'gen_ai.response.id': 'asst_abc123',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'url': {},
                            'gen_ai.provider.name': {},
                            'async': {},
                            'gen_ai.tool.definitions': {},
                            'gen_ai.request.model': {},
                            'gen_ai.system': {},
                            'gen_ai.response.model': {},
                            'gen_ai.response.id': {},
                        },
                    },
                },
            }
        ]
    )

def test_responses_api(exporter: TestExporter) -> None:
    client = openai.Client(api_key='foobar')
    logfire.instrument_openai(client, version=[1, 'latest'])
    tools: Any = [
        {
            'type': 'function',
            'name': 'get_weather',
            'description': 'Get current temperature for a given location.',
            'parameters': {
                'type': 'object',
                'properties': {'location': {'type': 'string', 'description': 'City and country e.g. Bogotá, Colombia'}},
                'required': ['location'],
                'additionalProperties': False,
            },
        }
    ]

    input_messages: Any = [{'role': 'user', 'content': 'What is the weather like in Paris today?'}]
    response = client.responses.create(
        model='gpt-4.1', input=input_messages[0]['content'], tools=tools, instructions='Be nice'
    )
    tool_call: Any = response.output[0]
    input_messages.append(tool_call)
    input_messages.append({'type': 'function_call_output', 'call_id': tool_call.call_id, 'output': 'Rainy'})
    response2: Any = client.responses.create(model='gpt-4.1', input=input_messages)
    assert response2.output[0].content[0].text == snapshot(
        "The weather in Paris today is rainy. If you're planning to go out, don't forget an umbrella!"
    )
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_responses_api',
                    'code.lineno': 123,
                    'gen_ai.provider.name': 'openai',
                    'async': False,
                    'request_data': {'model': 'gpt-4.1', 'stream': False},
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.tool.definitions': [
                        {
                            'type': 'function',
                            'name': 'get_weather',
                            'description': 'Get current temperature for a given location.',
                            'parameters': {
                                'type': 'object',
                                'properties': {
                                    'location': {
                                        'type': 'string',
                                        'description': 'City and country e.g. Bogotá, Colombia',
                                    }
                                },
                                'required': ['location'],
                                'additionalProperties': False,
                            },
                        }
                    ],
                    'gen_ai.input.messages': [
                        {
                            'role': 'user',
                            'parts': [{'type': 'text', 'content': 'What is the weather like in Paris today?'}],
                        }
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be nice'}],
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.msg': "Responses API with 'gpt-4.1'",
                    'gen_ai.system': 'openai',
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.request.model': 'gpt-4.1',
                    'gen_ai.response.model': 'gpt-4.1-2025-04-14',
                    'events': [
                        {'event.name': 'gen_ai.system.message', 'content': 'Be nice', 'role': 'system'},
                        {
                            'event.name': 'gen_ai.user.message',
                            'content': 'What is the weather like in Paris today?',
                            'role': 'user',
                        },
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'id': 'call_uilZSE2qAuMA2NWct72DBwd6',
                                    'type': 'function',
                                    'function': {'name': 'get_weather', 'arguments': '{"location":"Paris, France"}'},
                                }
                            ],
                        },
                    ],
                    'gen_ai.response.id': 'resp_039e74dd66b112920068dfe10528b8819c82d1214897014964',
                    'gen_ai.usage.input_tokens': 65,
                    'gen_ai.usage.output_tokens': 17,
                    'gen_ai.usage.raw': {
                        'input_tokens': 65,
                        'input_tokens_details': {'cached_tokens': 0},
                        'output_tokens': 17,
                        'output_tokens_details': {'reasoning_tokens': 0},
                        'total_tokens': 82,
                    },
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': 'call_uilZSE2qAuMA2NWct72DBwd6',
                                    'name': 'get_weather',
                                    'arguments': {'location': 'Paris, France'},
                                }
                            ],
                        }
                    ],
                    'operation.cost': 0.000266,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.provider.name': {},
                            'events': {'type': 'array'},
                            'gen_ai.request.model': {},
                            'request_data': {'type': 'object'},
                            'gen_ai.operation.name': {},
                            'gen_ai.tool.definitions': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'gen_ai.system': {},
                            'async': {},
                            'gen_ai.response.model': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.output_tokens': {},
                            'operation.cost': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'gen_ai.output.messages': {'type': 'array'},
                        },
                    },
                },
            },
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 3000000000,
                'end_time': 4000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_responses_api',
                    'code.lineno': 123,
                    'gen_ai.provider.name': 'openai',
                    'async': False,
                    'request_data': {'model': 'gpt-4.1', 'stream': False},
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {
                            'role': 'user',
                            'parts': [{'type': 'text', 'content': 'What is the weather like in Paris today?'}],
                        },
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': 'call_uilZSE2qAuMA2NWct72DBwd6',
                                    'name': 'get_weather',
                                    'arguments': {'location': 'Paris, France'},
                                }
                            ],
                        },
                        {
                            'role': 'tool',
                            'parts': [
                                {
                                    'type': 'tool_call_response',
                                    'id': 'call_uilZSE2qAuMA2NWct72DBwd6',
                                    'response': 'Rainy',
                                }
                            ],
                        },
                    ],
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.msg': "Responses API with 'gpt-4.1'",
                    'logfire.tags': ('LLM',),
                    'gen_ai.system': 'openai',
                    'logfire.span_type': 'span',
                    'gen_ai.request.model': 'gpt-4.1',
                    'gen_ai.response.model': 'gpt-4.1-2025-04-14',
                    'gen_ai.usage.input_tokens': 43,
                    'gen_ai.response.id': 'resp_039e74dd66b112920068dfe10687b4819cb0bc63819abcde35',
                    'events': [
                        {
                            'event.name': 'gen_ai.user.message',
                            'content': 'What is the weather like in Paris today?',
                            'role': 'user',
                        },
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'id': 'call_uilZSE2qAuMA2NWct72DBwd6',
                                    'type': 'function',
                                    'function': {'name': 'get_weather', 'arguments': '{"location":"Paris, France"}'},
                                }
                            ],
                        },
                        {
                            'event.name': 'gen_ai.tool.message',
                            'role': 'tool',
                            'id': 'call_uilZSE2qAuMA2NWct72DBwd6',
                            'content': 'Rainy',
                            'name': 'get_weather',
                        },
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'content': "The weather in Paris today is rainy. If you're planning to go out, don't forget an umbrella!",
                            'role': 'assistant',
                        },
                    ],
                    'gen_ai.usage.output_tokens': 21,
                    'gen_ai.usage.raw': {
                        'input_tokens': 43,
                        'input_tokens_details': {'cached_tokens': 0},
                        'output_tokens': 21,
                        'output_tokens_details': {'reasoning_tokens': 0},
                        'total_tokens': 64,
                    },
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'text',
                                    'content': "The weather in Paris today is rainy. If you're planning to go out, don't forget an umbrella!",
                                }
                            ],
                        }
                    ],
                    'operation.cost': 0.000254,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.provider.name': {},
                            'events': {'type': 'array'},
                            'gen_ai.request.model': {},
                            'request_data': {'type': 'object'},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.system': {},
                            'async': {},
                            'gen_ai.response.model': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.output_tokens': {},
                            'operation.cost': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'gen_ai.output.messages': {'type': 'array'},
                        },
                    },
                },
            },
        ]
    )

def test_openrouter_streaming_reasoning(exporter: TestExporter) -> None:
    client = openai.Client(api_key='foobar', base_url='https://openrouter.ai/api/v1')
    logfire.instrument_openai(client, version=[1, 'latest'])

    response = client.chat.completions.create(
        model='google/gemini-2.5-flash',
        messages=[{'role': 'user', 'content': 'Hello, how are you? (This is a trick question)'}],
        stream=True,
        extra_body={'reasoning': {'effort': 'low'}},
    )

    for _ in response:
        ...

    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_openrouter_streaming_reasoning',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [{'role': 'user', 'content': 'Hello, how are you? (This is a trick question)'}],
                        'model': 'google/gemini-2.5-flash',
                        'stream': True,
                    },
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.request.model': 'google/gemini-2.5-flash',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {
                            'role': 'user',
                            'parts': [{'type': 'text', 'content': 'Hello, how are you? (This is a trick question)'}],
                        }
                    ],
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'google/gemini-2.5-flash'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'google/gemini-2.5-flash',
                },
            },
            {
                'name': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                'context': {'trace_id': 2, 'span_id': 3, 'is_remote': False},
                'parent': None,
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'streaming response from {request_data[model]!r} took {duration:.2f}s',
                    'logfire.msg': "streaming response from 'google/gemini-2.5-flash' took 1.00s",
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_openrouter_streaming_reasoning',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [{'role': 'user', 'content': 'Hello, how are you? (This is a trick question)'}],
                        'model': 'google/gemini-2.5-flash',
                        'stream': True,
                    },
                    'gen_ai.request.model': 'google/gemini-2.5-flash',
                    'gen_ai.system': 'openai',
                    'gen_ai.provider.name': 'openai',
                    'async': False,
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.input.messages': [
                        {
                            'role': 'user',
                            'parts': [{'type': 'text', 'content': 'Hello, how are you? (This is a trick question)'}],
                        }
                    ],
                    'duration': 1.0,
                    'response_data': {
                        'message': {
                            'content': """\
That's a clever way to put it! You're right, it is a bit of a trick question for an AI.

As a large language model, I don't experience emotions, have a physical body, or "feel" things in the human sense, so I can't really quantify "how" I am.

However, I am fully operational, my systems are running smoothly, and I'm ready to assist you!

So, while I can't genuinely answer it for myself, how are *you* doing today, and what can I help you with?\
""",
                            'refusal': None,
                            'role': 'assistant',
                            'annotations': None,
                            'audio': None,
                            'function_call': None,
                            'tool_calls': None,
                            'parsed': None,
                            'reasoning': """\
**Interpreting User Intent**

I'm zeroing in on the core of the query. The "how are you" is basic, but the "trick question" label is key. My focus is on decoding what the user *really* wants. I'm anticipating something beyond a simple pleasantry.


""",
                            'reasoning_details': [
                                {
                                    'type': 'reasoning.text',
                                    'text': """\
**Interpreting User Intent**

I'm zeroing in on the core of the query. The "how are you" is basic, but the "trick question" label is key. My focus is on decoding what the user *really* wants. I'm anticipating something beyond a simple pleasantry.


""",
                                    'provider': 'google-vertex',
                                }
                            ],
                        },
                        'usage': {
                            'completion_tokens': 1003,
                            'prompt_tokens': 13,
                            'total_tokens': 1016,
                            'completion_tokens_details': None,
                            'prompt_tokens_details': None,
                        },
                    },
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'text',
                                    'content': """\
That's a clever way to put it! You're right, it is a bit of a trick question for an AI.

As a large language model, I don't experience emotions, have a physical body, or "feel" things in the human sense, so I can't really quantify "how" I am.

However, I am fully operational, my systems are running smoothly, and I'm ready to assist you!

So, while I can't genuinely answer it for myself, how are *you* doing today, and what can I help you with?\
""",
                                }
                            ],
                            'finish_reason': 'stop',
                        }
                    ],
                    'gen_ai.usage.input_tokens': 13,
                    'gen_ai.usage.output_tokens': 1003,
                    'gen_ai.usage.raw': {'completion_tokens': 1003, 'prompt_tokens': 13, 'total_tokens': 1016},
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.request.model': {},
                            'gen_ai.system': {},
                            'gen_ai.provider.name': {},
                            'async': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'duration': {},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'object',
                                        'title': 'ParsedChatCompletionMessage[object]',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                    'usage': {
                                        'type': 'object',
                                        'title': 'CompletionUsage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                        },
                    },
                    'logfire.tags': ('LLM',),
                    'gen_ai.response.model': 'google/gemini-2.5-flash',
                },
            },
        ]
    )

def test_chat_completions_with_audio_input(exporter: TestExporter) -> None:
    import base64

    client = openai.Client(api_key='foobar')
    logfire.instrument_openai(client, version=[1, 'latest'])
    import struct
    import wave

    # Generate 0.2s of silence at 16kHz mono 16-bit — minimum for OpenAI
    buf = BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack('<' + 'h' * 3200, *([0] * 3200)))
    wav_data = buf.getvalue()
    response = client.chat.completions.create(
        model='gpt-4o-audio-preview',
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Respond with just the word "hello".'},
                    {
                        'type': 'input_audio',
                        'input_audio': {'data': base64.b64encode(wav_data).decode(), 'format': 'wav'},
                    },
                ],
            }
        ],
    )
    assert response.choices[0].message.content is not None
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Chat Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_chat_completions_with_audio_input',
                    'code.lineno': 123,
                    'request_data': {
                        'messages': [
                            {
                                'role': 'user',
                                'content': [
                                    {'type': 'text', 'text': 'Respond with just the word "hello".'},
                                    {
                                        'type': 'input_audio',
                                        'input_audio': {
                                            'data': IsStr(),
                                            'format': 'wav',
                                        },
                                    },
                                ],
                            }
                        ],
                        'model': 'gpt-4o-audio-preview',
                    },
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'gpt-4o-audio-preview',
                    'gen_ai.input.messages': [
                        {
                            'role': 'user',
                            'parts': [
                                {'type': 'text', 'content': 'Respond with just the word "hello".'},
                                {
                                    'type': 'blob',
                                    'content': 'UklGRiQZAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
                                    'modality': 'audio',
                                },
                            ],
                        }
                    ],
                    'async': False,
                    'logfire.msg_template': 'Chat Completion with {request_data[model]!r}',
                    'logfire.msg': "Chat Completion with 'gpt-4o-audio-preview'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.system': 'openai',
                    'gen_ai.response.model': 'gpt-4o-audio-preview-2025-06-03',
                    'operation.cost': 1.5e-05,
                    'gen_ai.response.id': 'chatcmpl-D5caSDR31gOd1Qpyucf0b5VvVy9zY',
                    'gen_ai.usage.input_tokens': 21,
                    'gen_ai.usage.output_tokens': 1,
                    'gen_ai.usage.raw': {
                        'completion_tokens': 1,
                        'prompt_tokens': 21,
                        'total_tokens': 22,
                        'completion_tokens_details': {
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'text_tokens': 1,
                        },
                        'prompt_tokens_details': {
                            'audio_tokens': 2,
                            'cached_tokens': 0,
                            'text_tokens': 19,
                            'image_tokens': 0,
                        },
                    },
                    'response_data': {
                        'message': {
                            'content': 'Hello',
                            'refusal': None,
                            'role': 'assistant',
                            'annotations': [],
                            'audio': None,
                            'function_call': None,
                            'tool_calls': None,
                        },
                        'usage': {
                            'completion_tokens': 1,
                            'prompt_tokens': 21,
                            'total_tokens': 22,
                            'completion_tokens_details': {
                                'accepted_prediction_tokens': 0,
                                'audio_tokens': 0,
                                'reasoning_tokens': 0,
                                'rejected_prediction_tokens': 0,
                                'text_tokens': 1,
                            },
                            'prompt_tokens_details': {
                                'audio_tokens': 2,
                                'cached_tokens': 0,
                                'text_tokens': 19,
                                'image_tokens': 0,
                            },
                        },
                    },
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hello'}], 'finish_reason': 'stop'}
                    ],
                    'gen_ai.response.finish_reasons': ['stop'],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                            'gen_ai.system': {},
                            'gen_ai.response.model': {},
                            'operation.cost': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'object',
                                        'title': 'ChatCompletionMessage',
                                        'x-python-datatype': 'PydanticModel',
                                    },
                                    'usage': {
                                        'type': 'object',
                                        'title': 'CompletionUsage',
                                        'x-python-datatype': 'PydanticModel',
                                        'properties': {
                                            'completion_tokens_details': {
                                                'type': 'object',
                                                'title': 'CompletionTokensDetails',
                                                'x-python-datatype': 'PydanticModel',
                                            },
                                            'prompt_tokens_details': {
                                                'type': 'object',
                                                'title': 'PromptTokensDetails',
                                                'x-python-datatype': 'PydanticModel',
                                            },
                                        },
                                    },
                                },
                            },
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                        },
                    },
                },
            }
        ]
    )

def test_completion_stream_state_version_latest_only() -> None:
    """Test OpenaiCompletionStreamState.get_attributes with version='latest'."""
    from logfire._internal.integrations.llm_providers.openai import (
        OpenaiCompletionStreamState,
        _versioned_stream_cls,  # pyright: ignore[reportPrivateUsage]
    )

    stream_cls = _versioned_stream_cls(OpenaiCompletionStreamState, frozenset({'latest'}))
    state = stream_cls()
    state._content = ['Hello', ' world']  # type: ignore[attr-defined]

    result = state.get_attributes({'gen_ai.request.model': 'gpt-3.5-turbo'})
    assert 'response_data' not in result
    assert result['gen_ai.output.messages'] == snapshot(
        [{'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hello world'}]}]
    )

def test_completion_stream_state_version_latest_empty_content() -> None:
    """Test OpenaiCompletionStreamState with latest version but no content."""
    from logfire._internal.integrations.llm_providers.openai import (
        OpenaiCompletionStreamState,
        _versioned_stream_cls,  # pyright: ignore[reportPrivateUsage]
    )

    stream_cls = _versioned_stream_cls(OpenaiCompletionStreamState, frozenset({'latest'}))
    state = stream_cls()

    result = state.get_attributes({})
    assert 'response_data' not in result
    assert 'gen_ai.output.messages' not in result

def test_completion_stream_state_v1_only() -> None:
    """Test OpenaiCompletionStreamState.get_attributes with version=1 only."""
    from logfire._internal.integrations.llm_providers.openai import (
        OpenaiCompletionStreamState,
        _versioned_stream_cls,  # pyright: ignore[reportPrivateUsage]
    )

    stream_cls = _versioned_stream_cls(OpenaiCompletionStreamState, frozenset({1}))
    state = stream_cls()
    state._content = ['Hello']  # type: ignore[attr-defined]

    result = state.get_attributes({})
    assert result['response_data'] == snapshot({'combined_chunk_content': 'Hello', 'chunk_count': 1})
    assert 'gen_ai.output.messages' not in result

def test_completions_version_v1_only(exporter: TestExporter) -> None:
    """Test text completions with version=1 only."""
    client = openai.Client(api_key='foobar')
    logfire.instrument_openai(client, version=1)
    response = client.completions.create(
        model='gpt-3.5-turbo-instruct',
        prompt='What is four plus five?',
        max_tokens=10,
    )
    assert response.choices[0].text == 'Nine'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_completions_version_v1_only',
                    'code.lineno': 123,
                    'request_data': {
                        'model': 'gpt-3.5-turbo-instruct',
                        'prompt': 'What is four plus five?',
                        'max_tokens': 10,
                    },
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'text_completion',
                    'gen_ai.request.model': 'gpt-3.5-turbo-instruct',
                    'gen_ai.request.max_tokens': 10,
                    'async': False,
                    'logfire.msg_template': 'Completion with {request_data[model]!r}',
                    'logfire.msg': "Completion with 'gpt-3.5-turbo-instruct'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.system': 'openai',
                    'gen_ai.response.model': IsStr(),
                    'operation.cost': IsNumeric(),
                    'gen_ai.response.id': IsStr(),
                    'gen_ai.usage.input_tokens': IsInt(),
                    'gen_ai.usage.output_tokens': IsInt(),
                    'gen_ai.usage.raw': {'completion_tokens': 1, 'prompt_tokens': 5, 'total_tokens': 6},
                    'response_data': {
                        'finish_reason': 'stop',
                        'text': 'Nine',
                        'usage': {
                            'completion_tokens': IsInt(),
                            'prompt_tokens': IsInt(),
                            'total_tokens': IsInt(),
                            'completion_tokens_details': None,
                            'prompt_tokens_details': None,
                        },
                    },
                    'gen_ai.response.finish_reasons': ['stop'],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'async': {},
                            'gen_ai.system': {},
                            'gen_ai.response.model': {},
                            'operation.cost': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'response_data': {
                                'type': 'object',
                                'properties': {
                                    'usage': {
                                        'type': 'object',
                                        'title': 'CompletionUsage',
                                        'x-python-datatype': 'PydanticModel',
                                    }
                                },
                            },
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                        },
                    },
                },
            }
        ]
    )

def test_completions_version_latest_only(exporter: TestExporter) -> None:
    """Test text completions with version='latest' only."""
    client = openai.Client(api_key='foobar')
    logfire.instrument_openai(client, version='latest')
    response = client.completions.create(
        model='gpt-3.5-turbo-instruct',
        prompt='What is four plus five?',
        max_tokens=10,
    )
    assert response.choices[0].text == 'Nine'
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Completion with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_completions_version_latest_only',
                    'code.lineno': 123,
                    'request_data': {'model': 'gpt-3.5-turbo-instruct'},
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'text_completion',
                    'gen_ai.request.model': 'gpt-3.5-turbo-instruct',
                    'gen_ai.request.max_tokens': 10,
                    'async': False,
                    'logfire.msg_template': 'Completion with {request_data[model]!r}',
                    'logfire.msg': "Completion with 'gpt-3.5-turbo-instruct'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.system': 'openai',
                    'gen_ai.response.model': IsStr(),
                    'operation.cost': IsNumeric(),
                    'gen_ai.response.id': IsStr(),
                    'gen_ai.usage.input_tokens': IsInt(),
                    'gen_ai.usage.output_tokens': IsInt(),
                    'gen_ai.usage.raw': {'completion_tokens': 1, 'prompt_tokens': 5, 'total_tokens': 6},
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Nine'}], 'finish_reason': 'stop'}
                    ],
                    'gen_ai.response.finish_reasons': ['stop'],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.request.max_tokens': {},
                            'async': {},
                            'gen_ai.system': {},
                            'gen_ai.response.model': {},
                            'operation.cost': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.response.finish_reasons': {'type': 'array'},
                        },
                    },
                },
            }
        ]
    )

def test_responses_api_version_v1_only(exporter: TestExporter) -> None:
    """Test responses API with version=1 only."""
    client = openai.Client(api_key='foobar')
    logfire.instrument_openai(client, version=1)
    response = client.responses.create(
        model='gpt-4.1',
        input='What is four plus five?',
    )
    assert response.output[0].content[0].text == 'Four plus five is nine.'  # type: ignore
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_responses_api_version_v1_only',
                    'code.lineno': 123,
                    'request_data': {'model': 'gpt-4.1', 'stream': False},
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'gpt-4.1',
                    'async': False,
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.msg': "Responses API with 'gpt-4.1'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.system': 'openai',
                    'gen_ai.response.model': IsStr(),
                    'operation.cost': IsNumeric(),
                    'gen_ai.response.id': IsStr(),
                    'gen_ai.usage.input_tokens': IsInt(),
                    'gen_ai.usage.output_tokens': IsInt(),
                    'gen_ai.usage.raw': {
                        'input_tokens': 12,
                        'input_tokens_details': {'cached_tokens': 0},
                        'output_tokens': 8,
                        'output_tokens_details': {'reasoning_tokens': 0},
                        'total_tokens': 20,
                    },
                    'events': [
                        {'event.name': 'gen_ai.user.message', 'content': 'What is four plus five?', 'role': 'user'},
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'content': 'Four plus five is nine.',
                            'role': 'assistant',
                        },
                    ],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'events': {'type': 'array'},
                            'async': {},
                            'gen_ai.system': {},
                            'gen_ai.response.model': {},
                            'operation.cost': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                        },
                    },
                },
            }
        ]
    )

def test_responses_api_version_latest_only(exporter: TestExporter) -> None:
    """Test responses API with version='latest' only."""
    client = openai.Client(api_key='foobar')
    logfire.instrument_openai(client, version='latest')
    response = client.responses.create(
        model='gpt-4.1',
        input='What is four plus five?',
    )
    assert response.output[0].content[0].text == 'Four plus five is nine.'  # type: ignore
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_responses_api_version_latest_only',
                    'code.lineno': 123,
                    'request_data': {'model': 'gpt-4.1', 'stream': False},
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': 'gpt-4.1',
                    'gen_ai.input.messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is four plus five?'}]}
                    ],
                    'async': False,
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.msg': "Responses API with 'gpt-4.1'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.system': 'openai',
                    'gen_ai.response.model': IsStr(),
                    'operation.cost': IsNumeric(),
                    'gen_ai.response.id': IsStr(),
                    'gen_ai.usage.input_tokens': IsInt(),
                    'gen_ai.usage.output_tokens': IsInt(),
                    'gen_ai.usage.raw': {
                        'input_tokens': 12,
                        'input_tokens_details': {'cached_tokens': 0},
                        'output_tokens': 8,
                        'output_tokens_details': {'reasoning_tokens': 0},
                        'total_tokens': 20,
                    },
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Four plus five is nine.'}]}
                    ],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'gen_ai.input.messages': {'type': 'array'},
                            'async': {},
                            'gen_ai.system': {},
                            'gen_ai.response.model': {},
                            'operation.cost': {},
                            'gen_ai.response.id': {},
                            'gen_ai.usage.input_tokens': {},
                            'gen_ai.usage.output_tokens': {},
                            'gen_ai.usage.raw': {'type': 'object'},
                            'gen_ai.output.messages': {'type': 'array'},
                        },
                    },
                },
            }
        ]
    )

def test_images_version_latest_only(exporter: TestExporter) -> None:
    """Test images API with version='latest' only."""
    client = openai.Client(api_key='foobar')
    logfire.instrument_openai(client, version='latest')
    response = client.images.generate(
        model='dall-e-2',
        prompt='A sunset',
        size='256x256',
    )
    assert response.data[0].url == 'https://example.com/image.png'  # type: ignore[union-attr]
    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Image Generation with {request_data[model]!r}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'code.filepath': 'test_openai.py',
                    'code.function': 'test_images_version_latest_only',
                    'code.lineno': 123,
                    'request_data': {'model': 'dall-e-2'},
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.operation.name': 'image_generation',
                    'gen_ai.request.model': 'dall-e-2',
                    'async': False,
                    'logfire.msg_template': 'Image Generation with {request_data[model]!r}',
                    'logfire.msg': "Image Generation with 'dall-e-2'",
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'gen_ai.system': 'openai',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'request_data': {'type': 'object'},
                            'gen_ai.provider.name': {},
                            'gen_ai.operation.name': {},
                            'gen_ai.request.model': {},
                            'async': {},
                            'gen_ai.system': {},
                        },
                    },
                    'gen_ai.response.model': IsStr(),
                },
            }
        ]
    )

def test_convert_chat_completions_with_list_content() -> None:
    """Test convert_chat_completions_to_semconv with list content parts."""
    from logfire._internal.integrations.llm_providers.openai import convert_chat_completions_to_semconv

    messages: list[dict[str, Any]] = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Describe this image'}]},
    ]
    assert convert_chat_completions_to_semconv(messages) == snapshot(
        [{'role': 'user', 'parts': [{'type': 'text', 'content': 'Describe this image'}]}]
    )


# --- tests/otel_integrations/test_openai_agents.py ---

def test_openai_agent_tracing(exporter: TestExporter):
    logfire.instrument_openai_agents()

    with logfire.span('logfire span 1'):
        assert get_current_trace() is None
        with trace('trace_name') as t:
            assert isinstance(t, LogfireTraceWrapper)
            assert get_current_trace() is t
            with logfire.span('logfire span 2'):
                assert get_current_span() is None
                with agent_span('agent_name') as s:
                    assert get_current_trace() is t
                    assert get_current_span() is s
                    assert isinstance(s, LogfireSpanWrapper)
                    logfire.info('Hi')
                assert get_current_span() is None
        assert get_current_trace() is None

    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Hi',
                'context': {'trace_id': 1, 'span_id': 9, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'Hi',
                    'logfire.msg': 'Hi',
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_openai_agent_tracing',
                    'code.lineno': 123,
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'start_time': 4000000000,
                'end_time': 6000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_openai_agent_tracing',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'name': 'agent_name',
                    'handoffs': 'null',
                    'tools': 'null',
                    'output_type': 'null',
                    'gen_ai.system': 'openai',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'name': {},
                            'handoffs': {'type': 'null'},
                            'tools': {'type': 'null'},
                            'output_type': {'type': 'null'},
                            'gen_ai.system': {},
                        },
                    },
                    'logfire.msg': "Agent run: 'agent_name'",
                },
            },
            {
                'name': 'logfire span 2',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 3000000000,
                'end_time': 7000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_openai_agent_tracing',
                    'code.lineno': 123,
                    'logfire.msg_template': 'logfire span 2',
                    'logfire.msg': 'logfire span 2',
                    'logfire.span_type': 'span',
                },
            },
            {
                'name': 'OpenAI Agents trace: {name}',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 8000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_openai_agent_tracing',
                    'code.lineno': 123,
                    'name': 'trace_name',
                    'agent_trace_id': IsStr(),
                    'metadata': 'null',
                    'tracing': 'null',
                    'group_id': 'null',
                    'logfire.msg_template': 'OpenAI Agents trace: {name}',
                    'logfire.span_type': 'span',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'name': {},
                            'agent_trace_id': {},
                            'group_id': {'type': 'null'},
                            'metadata': {'type': 'null'},
                            'tracing': {'type': 'null'},
                        },
                    },
                    'logfire.msg': 'OpenAI Agents trace: trace_name',
                },
            },
            {
                'name': 'logfire span 1',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 9000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_openai_agent_tracing',
                    'code.lineno': 123,
                    'logfire.msg_template': 'logfire span 1',
                    'logfire.msg': 'logfire span 1',
                    'logfire.span_type': 'span',
                },
            },
        ]
    )

def test_openai_agent_tracing_manual_start_end(exporter: TestExporter):
    logfire.instrument_openai_agents()

    with logfire.span('logfire span 1'):
        t = trace('trace_name')
        assert isinstance(t, LogfireTraceWrapper)
        assert not t.span_helper.span.is_recording()
        assert get_current_trace() is None
        t.start(mark_as_current=True)
        assert t.span_helper.span.is_recording()
        assert get_current_trace() is t
        with logfire.span('logfire span 2'):
            s = agent_span('agent_name')
            assert isinstance(s, LogfireSpanWrapper)
            assert get_current_span() is None
            s.start(mark_as_current=True)
            assert get_current_span() is s

            s2 = agent_span('agent_name2')
            assert isinstance(s2, LogfireSpanWrapper)
            assert get_current_span() is s
            s2.start()
            assert get_current_span() is s

            logfire.info('Hi')

            s2.finish(reset_current=True)
            assert get_current_span() is s
            s.finish(reset_current=True)
            assert get_current_span() is None

        assert get_current_trace() is t
        t.finish(reset_current=True)
        assert get_current_trace() is None

    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Hi',
                'context': {'trace_id': 1, 'span_id': 11, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'start_time': 6000000000,
                'end_time': 6000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'Hi',
                    'logfire.msg': 'Hi',
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_openai_agent_tracing_manual_start_end',
                    'code.lineno': 123,
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 9, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'start_time': 5000000000,
                'end_time': 7000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_openai_agent_tracing_manual_start_end',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'name': 'agent_name2',
                    'handoffs': 'null',
                    'tools': 'null',
                    'output_type': 'null',
                    'gen_ai.system': 'openai',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'name': {},
                            'handoffs': {'type': 'null'},
                            'tools': {'type': 'null'},
                            'output_type': {'type': 'null'},
                            'gen_ai.system': {},
                        },
                    },
                    'logfire.msg': "Agent run: 'agent_name2'",
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'start_time': 4000000000,
                'end_time': 8000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_openai_agent_tracing_manual_start_end',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'name': 'agent_name',
                    'handoffs': 'null',
                    'tools': 'null',
                    'output_type': 'null',
                    'gen_ai.system': 'openai',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'name': {},
                            'handoffs': {'type': 'null'},
                            'tools': {'type': 'null'},
                            'output_type': {'type': 'null'},
                            'gen_ai.system': {},
                        },
                    },
                    'logfire.msg': "Agent run: 'agent_name'",
                },
            },
            {
                'name': 'logfire span 2',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 3000000000,
                'end_time': 9000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_openai_agent_tracing_manual_start_end',
                    'code.lineno': 123,
                    'logfire.msg_template': 'logfire span 2',
                    'logfire.msg': 'logfire span 2',
                    'logfire.span_type': 'span',
                },
            },
            {
                'name': 'OpenAI Agents trace: {name}',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 10000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_openai_agent_tracing_manual_start_end',
                    'code.lineno': 123,
                    'name': 'trace_name',
                    'agent_trace_id': IsStr(),
                    'metadata': 'null',
                    'tracing': 'null',
                    'group_id': 'null',
                    'logfire.msg_template': 'OpenAI Agents trace: {name}',
                    'logfire.span_type': 'span',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'name': {},
                            'agent_trace_id': {},
                            'group_id': {'type': 'null'},
                            'metadata': {'type': 'null'},
                            'tracing': {'type': 'null'},
                        },
                    },
                    'logfire.msg': 'OpenAI Agents trace: trace_name',
                },
            },
            {
                'name': 'logfire span 1',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 11000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_openai_agent_tracing_manual_start_end',
                    'code.lineno': 123,
                    'logfire.msg_template': 'logfire span 1',
                    'logfire.msg': 'logfire span 1',
                    'logfire.span_type': 'span',
                },
            },
        ]
    )

def test_manual_parents(exporter: TestExporter):
    logfire.instrument_openai_agents()

    t = trace('my_trace', trace_id='trace_123')
    t.start()
    s = agent_span('my_span', parent=t)
    s.start()
    with custom_span('my_custom_span', parent=s):
        pass
    s.finish()
    t.finish()

    assert exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'Custom span: {name}',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 3000000000,
                'end_time': 4000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_manual_parents',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Custom span: {name}',
                    'logfire.span_type': 'span',
                    'name': 'my_custom_span',
                    'data': {},
                    'gen_ai.system': 'openai',
                    'logfire.msg': 'Custom span: my_custom_span',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {'name': {}, 'data': {'type': 'object'}, 'gen_ai.system': {}},
                    },
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 5000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_manual_parents',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'name': 'my_span',
                    'handoffs': 'null',
                    'tools': 'null',
                    'output_type': 'null',
                    'gen_ai.system': 'openai',
                    'logfire.msg': "Agent run: 'my_span'",
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'name': {},
                            'handoffs': {'type': 'null'},
                            'tools': {'type': 'null'},
                            'output_type': {'type': 'null'},
                            'gen_ai.system': {},
                        },
                    },
                },
            },
            {
                'name': 'OpenAI Agents trace: {name}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 6000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_manual_parents',
                    'code.lineno': 123,
                    'name': 'my_trace',
                    'group_id': 'null',
                    'metadata': 'null',
                    'tracing': 'null',
                    'logfire.msg_template': 'OpenAI Agents trace: {name}',
                    'logfire.msg': 'OpenAI Agents trace: my_trace',
                    'logfire.span_type': 'span',
                    'agent_trace_id': 'trace_123',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'name': {},
                            'agent_trace_id': {},
                            'group_id': {'type': 'null'},
                            'metadata': {'type': 'null'},
                            'tracing': {'type': 'null'},
                        },
                    },
                },
            },
        ]
    )

async def test_responses(exporter: TestExporter):
    logfire.instrument_openai_agents()

    @function_tool
    def random_number() -> int:
        return 4

    agent2 = Agent(name='agent2', instructions='Return double the number')
    agent1 = Agent(name='agent1', tools=[random_number], handoffs=[agent2])

    with logfire.instrument_openai():
        await Runner.run(agent1, input='Generate a random number then, hand off to agent2.')

    assert simplify_spans(exporter.exported_spans_as_dict(parse_json_attributes=True)) == snapshot(
        [
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 3000000000,
                'end_time': 4000000000,
                'attributes': {
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Responses API with 'gpt-4o'",
                    'response_id': 'resp_67ced68228748191b31ea5d9172a7b4b',
                    'gen_ai.request.model': 'gpt-4o',
                    'gen_ai.response.model': 'gpt-4o-2024-08-06',
                    'gen_ai.system': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'raw_input': [{'content': 'Generate a random number then, hand off to agent2.', 'role': 'user'}],
                    'events': [
                        {
                            'event.name': 'gen_ai.user.message',
                            'content': 'Generate a random number then, hand off to agent2.',
                            'role': 'user',
                        },
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'id': 'call_vwqy7HyGGnNht9NNfxMnnouY',
                                    'type': 'function',
                                    'function': {'name': 'random_number', 'arguments': '{}'},
                                }
                            ],
                        },
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'id': 'call_oEA0MnUXCwKevx8txteoopNL',
                                    'type': 'function',
                                    'function': {'name': 'transfer_to_agent2', 'arguments': '{}'},
                                }
                            ],
                        },
                    ],
                },
            },
            {
                'name': 'Function: {name}',
                'context': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 5000000000,
                'end_time': 6000000000,
                'attributes': {
                    'logfire.msg_template': 'Function: {name}',
                    'logfire.span_type': 'span',
                    'logfire.msg': 'Function: random_number',
                    'name': 'random_number',
                    'input': {},
                    'mcp_data': 'null',
                    'gen_ai.system': 'openai',
                    'output': 4,
                },
            },
            {
                'name': 'Handoff: {from_agent} → {to_agent}',
                'context': {'trace_id': 1, 'span_id': 9, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 7000000000,
                'end_time': 8000000000,
                'attributes': {
                    'logfire.msg_template': 'Handoff: {from_agent} → {to_agent}',
                    'logfire.span_type': 'span',
                    'logfire.msg': 'Handoff: agent1 → agent2',
                    'from_agent': 'agent1',
                    'gen_ai.system': 'openai',
                    'to_agent': 'agent2',
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 9000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_responses',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Agent run: 'agent1'",
                    'name': 'agent1',
                    'handoffs': ['agent2'],
                    'tools': ['random_number'],
                    'gen_ai.system': 'openai',
                    'output_type': 'str',
                },
            },
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 13, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 11, 'is_remote': False},
                'start_time': 11000000000,
                'end_time': 12000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_responses',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Responses API with 'gpt-4o'",
                    'response_id': 'resp_67ced68425f48191a5fb0c2b61cb27dd',
                    'gen_ai.request.model': 'gpt-4o',
                    'gen_ai.response.model': 'gpt-4o-2024-08-06',
                    'gen_ai.system': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'raw_input': [
                        {'content': 'Generate a random number then, hand off to agent2.', 'role': 'user'},
                        {
                            'arguments': '{}',
                            'call_id': 'call_vwqy7HyGGnNht9NNfxMnnouY',
                            'name': 'random_number',
                            'type': 'function_call',
                            'id': 'fc_67ced68352a48191aca3872f9376de86',
                            'status': 'completed',
                        },
                        {
                            'arguments': '{}',
                            'call_id': 'call_oEA0MnUXCwKevx8txteoopNL',
                            'name': 'transfer_to_agent2',
                            'type': 'function_call',
                            'id': 'fc_67ced683c8d88191b21be486e163e815',
                            'status': 'completed',
                        },
                        {'call_id': 'call_vwqy7HyGGnNht9NNfxMnnouY', 'output': '4', 'type': 'function_call_output'},
                        {
                            'call_id': 'call_oEA0MnUXCwKevx8txteoopNL',
                            'output': '{"assistant": "agent2"}',
                            'type': 'function_call_output',
                        },
                    ],
                    'events': [
                        {
                            'event.name': 'gen_ai.system.message',
                            'content': 'Return double the number',
                            'role': 'system',
                        },
                        {
                            'event.name': 'gen_ai.user.message',
                            'content': 'Generate a random number then, hand off to agent2.',
                            'role': 'user',
                        },
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'id': 'call_vwqy7HyGGnNht9NNfxMnnouY',
                                    'type': 'function',
                                    'function': {'name': 'random_number', 'arguments': '{}'},
                                }
                            ],
                        },
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'id': 'call_oEA0MnUXCwKevx8txteoopNL',
                                    'type': 'function',
                                    'function': {'name': 'transfer_to_agent2', 'arguments': '{}'},
                                }
                            ],
                        },
                        {
                            'event.name': 'gen_ai.tool.message',
                            'role': 'tool',
                            'id': 'call_vwqy7HyGGnNht9NNfxMnnouY',
                            'content': '4',
                            'name': 'random_number',
                        },
                        {
                            'event.name': 'gen_ai.tool.message',
                            'role': 'tool',
                            'id': 'call_oEA0MnUXCwKevx8txteoopNL',
                            'content': '{"assistant": "agent2"}',
                            'name': 'transfer_to_agent2',
                        },
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'content': "The random number generated is 4, and it's been handed off to agent2.",
                            'role': 'assistant',
                        },
                    ],
                    'gen_ai.usage.input_tokens': 89,
                    'gen_ai.usage.output_tokens': 18,
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 11, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 10000000000,
                'end_time': 13000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_responses',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Agent run: 'agent2'",
                    'name': 'agent2',
                    'handoffs': [],
                    'tools': [],
                    'gen_ai.system': 'openai',
                    'output_type': 'str',
                },
            },
            {
                'name': 'OpenAI Agents trace: {name}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 14000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_responses',
                    'code.lineno': 123,
                    'name': 'Agent workflow',
                    'group_id': 'null',
                    'metadata': 'null',
                    'tracing': 'null',
                    'logfire.msg_template': 'OpenAI Agents trace: {name}',
                    'logfire.msg': 'OpenAI Agents trace: Agent workflow',
                    'logfire.span_type': 'span',
                    'agent_trace_id': IsStr(),
                },
            },
        ]
    )

async def test_input_guardrails(exporter: TestExporter):
    logfire.instrument_openai_agents()

    @input_guardrail
    async def zero_guardrail(_context: Any, _agent: Agent[Any], inp: Any) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(output_info={'input': inp}, tripwire_triggered='0' in str(inp))

    agent = Agent[str](name='my_agent', input_guardrails=[zero_guardrail])

    await Runner.run(agent, '1+1?')
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, '0?')

    assert simplify_spans(exporter.exported_spans_as_dict(parse_json_attributes=True)) == snapshot(
        [
            {
                'name': 'Guardrail {name!r} {triggered=}',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 3000000000,
                'end_time': 4000000000,
                'attributes': {
                    'logfire.msg_template': 'Guardrail {name!r} {triggered=}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Guardrail 'zero_guardrail' triggered=False",
                    'name': 'zero_guardrail',
                    'gen_ai.system': 'openai',
                    'triggered': False,
                },
            },
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 5000000000,
                'end_time': 6000000000,
                'attributes': {
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Responses API with 'gpt-4o'",
                    'response_id': 'resp_67cee263c6e0819184efdc0fe2624cc8',
                    'gen_ai.request.model': 'gpt-4o',
                    'gen_ai.response.model': 'gpt-4o-2024-08-06',
                    'gen_ai.system': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'raw_input': [{'content': '1+1?', 'role': 'user'}],
                    'events': [
                        {'event.name': 'gen_ai.user.message', 'content': '1+1?', 'role': 'user'},
                        {'event.name': 'gen_ai.assistant.message', 'content': '1 + 1 equals 2.', 'role': 'assistant'},
                    ],
                    'gen_ai.usage.input_tokens': 29,
                    'gen_ai.usage.output_tokens': 9,
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 7000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_input_guardrails',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Agent run: 'my_agent'",
                    'name': 'my_agent',
                    'handoffs': [],
                    'tools': [],
                    'gen_ai.system': 'openai',
                    'output_type': 'str',
                },
            },
            {
                'name': 'OpenAI Agents trace: {name}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 8000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_input_guardrails',
                    'code.lineno': 123,
                    'name': 'Agent workflow',
                    'group_id': 'null',
                    'metadata': 'null',
                    'tracing': 'null',
                    'logfire.msg_template': 'OpenAI Agents trace: {name}',
                    'logfire.msg': 'OpenAI Agents trace: Agent workflow',
                    'logfire.span_type': 'span',
                    'agent_trace_id': IsStr(),
                },
            },
            {
                'name': 'Guardrail {name!r} {triggered=}',
                'context': {'trace_id': 2, 'span_id': 13, 'is_remote': False},
                'parent': {'trace_id': 2, 'span_id': 11, 'is_remote': False},
                'start_time': 11000000000,
                'end_time': 12000000000,
                'attributes': {
                    'logfire.msg_template': 'Guardrail {name!r} {triggered=}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Guardrail 'zero_guardrail' triggered=True",
                    'name': 'zero_guardrail',
                    'gen_ai.system': 'openai',
                    'triggered': True,
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 2, 'span_id': 11, 'is_remote': False},
                'parent': {'trace_id': 2, 'span_id': 9, 'is_remote': False},
                'start_time': 10000000000,
                'end_time': 13000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_input_guardrails',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'logfire.level_num': 17,
                    'logfire.msg': "Agent run: 'my_agent' failed: Guardrail tripwire triggered",
                    'name': 'my_agent',
                    'handoffs': [],
                    'tools': [],
                    'gen_ai.system': 'openai',
                    'output_type': 'str',
                    'error': {'message': 'Guardrail tripwire triggered', 'data': {'guardrail': 'zero_guardrail'}},
                },
            },
            {
                'name': 'OpenAI Agents trace: {name}',
                'context': {'trace_id': 2, 'span_id': 9, 'is_remote': False},
                'parent': None,
                'start_time': 9000000000,
                'end_time': 14000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_input_guardrails',
                    'code.lineno': 123,
                    'name': 'Agent workflow',
                    'group_id': 'null',
                    'metadata': 'null',
                    'tracing': 'null',
                    'logfire.msg_template': 'OpenAI Agents trace: {name}',
                    'logfire.msg': 'OpenAI Agents trace: Agent workflow',
                    'logfire.span_type': 'span',
                    'agent_trace_id': IsStr(),
                },
            },
        ]
    )

async def test_chat_completions(exporter: TestExporter):
    logfire.instrument_openai_agents()

    model = OpenAIChatCompletionsModel('gpt-4o', AsyncOpenAI())
    agent = Agent[str](name='my_agent', model=model)
    with logfire.instrument_openai():
        await Runner.run(agent, '1+1?')
    assert simplify_spans(exporter.exported_spans_as_dict(parse_json_attributes=True)) == snapshot(
        [
            {
                'name': 'Chat completion with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 3000000000,
                'end_time': 4000000000,
                'attributes': {
                    'logfire.msg_template': 'Chat completion with {gen_ai.request.model!r}',
                    'logfire.tags': ('LLM',),
                    'logfire.span_type': 'span',
                    'logfire.msg': "Chat completion with 'gpt-4o'",
                    'input': [{'role': 'user', 'content': '1+1?'}],
                    'output': [
                        {
                            'content': '1 + 1 = 2',
                            'refusal': None,
                            'role': 'assistant',
                            'audio': None,
                            'function_call': None,
                            'tool_calls': None,
                            'annotations': [],
                        }
                    ],
                    'model_config': IsPartialDict(),
                    'usage': {
                        'requests': 1,
                        'input_tokens': 11,
                        'output_tokens': 8,
                        'total_tokens': 19,
                        'input_tokens_details': {'cached_tokens': 0},
                        'output_tokens_details': {'reasoning_tokens': 0},
                    },
                    'gen_ai.system': 'openai',
                    'gen_ai.request.model': 'gpt-4o',
                    'gen_ai.response.model': 'gpt-4o',
                    'gen_ai.usage.input_tokens': 11,
                    'gen_ai.usage.output_tokens': 8,
                    'request_data': {
                        'messages': [
                            {'role': 'user', 'content': '1+1?'},
                            {
                                'content': '1 + 1 = 2',
                                'refusal': None,
                                'role': 'assistant',
                                'annotations': [],
                                'audio': None,
                                'function_call': None,
                                'tool_calls': None,
                            },
                        ],
                        'model': 'gpt-4o',
                    },
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 5000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_chat_completions',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Agent run: 'my_agent'",
                    'name': 'my_agent',
                    'handoffs': [],
                    'tools': [],
                    'gen_ai.system': 'openai',
                    'output_type': 'str',
                },
            },
            {
                'name': 'OpenAI Agents trace: {name}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 6000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_chat_completions',
                    'code.lineno': 123,
                    'name': 'Agent workflow',
                    'group_id': 'null',
                    'metadata': 'null',
                    'tracing': 'null',
                    'logfire.msg_template': 'OpenAI Agents trace: {name}',
                    'logfire.msg': 'OpenAI Agents trace: Agent workflow',
                    'logfire.span_type': 'span',
                    'agent_trace_id': IsStr(),
                },
            },
        ]
    )

async def test_responses_simple(exporter: TestExporter):
    logfire.instrument_openai_agents()

    agent1 = Agent(name='agent1')

    with trace('my_trace', trace_id='trace_123'):
        result = await Runner.run(agent1, input='2+2?')
        await Runner.run(agent1, input=result.to_input_list() + [{'role': 'user', 'content': '4?'}])

    assert simplify_spans(exporter.exported_spans_as_dict(parse_json_attributes=True)) == snapshot(
        [
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 3000000000,
                'end_time': 4000000000,
                'attributes': {
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Responses API with 'gpt-4o'",
                    'response_id': 'resp_01544eeb9f4d9c9100699336d86b9481a1aeace2b855734b8a',
                    'gen_ai.request.model': 'gpt-4o',
                    'gen_ai.response.model': 'gpt-4o-2024-08-06',
                    'gen_ai.system': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'raw_input': [{'content': '2+2?', 'role': 'user'}],
                    'events': [
                        {'event.name': 'gen_ai.user.message', 'content': '2+2?', 'role': 'user'},
                        {'event.name': 'gen_ai.assistant.message', 'content': '2 + 2 = 4', 'role': 'assistant'},
                    ],
                    'gen_ai.usage.input_tokens': 11,
                    'gen_ai.usage.output_tokens': 8,
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 5000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_responses_simple',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Agent run: 'agent1'",
                    'name': 'agent1',
                    'handoffs': [],
                    'tools': [],
                    'gen_ai.system': 'openai',
                    'output_type': 'str',
                },
            },
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 9, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'start_time': 7000000000,
                'end_time': 8000000000,
                'attributes': {
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Responses API with 'gpt-4o'",
                    'response_id': 'resp_01544eeb9f4d9c9100699336da82d081a1a2b74cadabbd9a07',
                    'gen_ai.request.model': 'gpt-4o',
                    'gen_ai.response.model': 'gpt-4o-2024-08-06',
                    'gen_ai.system': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'raw_input': [
                        {'content': '2+2?', 'role': 'user'},
                        {
                            'id': 'msg_01544eeb9f4d9c9100699336d96a4081a1ab96132d47d6fabe',
                            'content': [
                                {'annotations': [], 'text': '2 + 2 = 4', 'type': 'output_text', 'logprobs': []}
                            ],
                            'role': 'assistant',
                            'status': 'completed',
                            'type': 'message',
                        },
                        {'role': 'user', 'content': '4?'},
                    ],
                    'events': [
                        {'event.name': 'gen_ai.user.message', 'content': '2+2?', 'role': 'user'},
                        {'event.name': 'gen_ai.assistant.message', 'content': '2 + 2 = 4', 'role': 'assistant'},
                        {'event.name': 'gen_ai.user.message', 'content': '4?', 'role': 'user'},
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'content': "Yes, that's correct!",
                            'role': 'assistant',
                        },
                    ],
                    'gen_ai.usage.input_tokens': 28,
                    'gen_ai.usage.output_tokens': 6,
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 6000000000,
                'end_time': 9000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_responses_simple',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Agent run: 'agent1'",
                    'name': 'agent1',
                    'handoffs': [],
                    'tools': [],
                    'gen_ai.system': 'openai',
                    'output_type': 'str',
                },
            },
            {
                'name': 'OpenAI Agents trace: {name}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 10000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_responses_simple',
                    'code.lineno': 123,
                    'name': 'my_trace',
                    'group_id': 'null',
                    'metadata': 'null',
                    'tracing': 'null',
                    'logfire.msg_template': 'OpenAI Agents trace: {name}',
                    'logfire.msg': 'OpenAI Agents trace: my_trace',
                    'logfire.span_type': 'span',
                    'agent_trace_id': 'trace_123',
                },
            },
        ]
    )

async def test_file_search(exporter: TestExporter):
    logfire.instrument_openai_agents()

    agent = Agent(
        name='agent',
        tools=[FileSearchTool(max_num_results=1, vector_store_ids=['vs_67cd9e6afeb4819198cbffafab95d8ba'])],
    )

    with trace('my_trace', trace_id='trace_123'):
        result = await Runner.run(agent, 'Who made Logfire?')
        await Runner.run(agent, input=result.to_input_list() + [{'role': 'user', 'content': '2+2?'}])

    assert simplify_spans(exporter.exported_spans_as_dict(parse_json_attributes=True)) == snapshot(
        [
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 3000000000,
                'end_time': 4000000000,
                'attributes': {
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Responses API with 'gpt-4o'",
                    'response_id': 'resp_05cdfcc11457b63e0069933708a22881a08d070f84c840a8c8',
                    'gen_ai.request.model': 'gpt-4o',
                    'gen_ai.response.model': 'gpt-4o-2024-08-06',
                    'gen_ai.system': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'raw_input': [{'content': 'Who made Logfire?', 'role': 'user'}],
                    'events': [
                        {'event.name': 'gen_ai.user.message', 'content': 'Who made Logfire?', 'role': 'user'},
                        {
                            'event.name': 'gen_ai.unknown',
                            'role': 'assistant',
                            'content': """\
file_search_call

See JSON for details\
""",
                            'data': {
                                'id': 'fs_05cdfcc11457b63e0069933709b8ec81a0b339e16bec80b213',
                                'queries': ['Who made Logfire?'],
                                'status': 'completed',
                                'type': 'file_search_call',
                                'results': None,
                            },
                        },
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'content': 'Logfire is made by Pydantic.',
                            'role': 'assistant',
                        },
                    ],
                    'gen_ai.usage.input_tokens': 1144,
                    'gen_ai.usage.output_tokens': 38,
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 5000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_file_search',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Agent run: 'agent'",
                    'name': 'agent',
                    'handoffs': [],
                    'tools': ['file_search'],
                    'gen_ai.system': 'openai',
                    'output_type': 'str',
                },
            },
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 9, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'start_time': 7000000000,
                'end_time': 8000000000,
                'attributes': {
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Responses API with 'gpt-4o'",
                    'response_id': 'resp_05cdfcc11457b63e006993370c20d881a0ac8a9082d4d1b977',
                    'gen_ai.request.model': 'gpt-4o',
                    'gen_ai.response.model': 'gpt-4o-2024-08-06',
                    'gen_ai.system': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'raw_input': [
                        {'content': 'Who made Logfire?', 'role': 'user'},
                        {
                            'id': 'fs_05cdfcc11457b63e0069933709b8ec81a0b339e16bec80b213',
                            'queries': ['Who made Logfire?'],
                            'status': 'completed',
                            'type': 'file_search_call',
                            'results': None,
                        },
                        {
                            'id': 'msg_05cdfcc11457b63e006993370ae66881a0bddd68b7a653056f',
                            'content': [
                                {
                                    'annotations': [
                                        {
                                            'file_id': 'file-CmKZQn5qLRRgcAjS61GSqv',
                                            'index': 27,
                                            'type': 'file_citation',
                                            'filename': 'test.txt',
                                        }
                                    ],
                                    'text': 'Logfire is made by Pydantic.',
                                    'type': 'output_text',
                                    'logprobs': [],
                                }
                            ],
                            'role': 'assistant',
                            'status': 'completed',
                            'type': 'message',
                        },
                        {'role': 'user', 'content': '2+2?'},
                    ],
                    'events': [
                        {'event.name': 'gen_ai.user.message', 'content': 'Who made Logfire?', 'role': 'user'},
                        {
                            'event.name': 'gen_ai.unknown',
                            'role': 'unknown',
                            'content': """\
file_search_call

See JSON for details\
""",
                            'data': {
                                'id': 'fs_05cdfcc11457b63e0069933709b8ec81a0b339e16bec80b213',
                                'queries': ['Who made Logfire?'],
                                'status': 'completed',
                                'type': 'file_search_call',
                                'results': None,
                            },
                        },
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'content': 'Logfire is made by Pydantic.',
                            'role': 'assistant',
                        },
                        {'event.name': 'gen_ai.user.message', 'content': '2+2?', 'role': 'user'},
                        {'event.name': 'gen_ai.assistant.message', 'content': '2 + 2 equals 4.', 'role': 'assistant'},
                    ],
                    'gen_ai.usage.input_tokens': 862,
                    'gen_ai.usage.output_tokens': 10,
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 6000000000,
                'end_time': 9000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_file_search',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'logfire.msg': "Agent run: 'agent'",
                    'name': 'agent',
                    'handoffs': [],
                    'tools': ['file_search'],
                    'gen_ai.system': 'openai',
                    'output_type': 'str',
                },
            },
            {
                'name': 'OpenAI Agents trace: {name}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 10000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_file_search',
                    'code.lineno': 123,
                    'name': 'my_trace',
                    'group_id': 'null',
                    'metadata': 'null',
                    'tracing': 'null',
                    'logfire.msg_template': 'OpenAI Agents trace: {name}',
                    'logfire.msg': 'OpenAI Agents trace: my_trace',
                    'logfire.span_type': 'span',
                    'agent_trace_id': 'trace_123',
                },
            },
        ]
    )

async def test_function_tool_exception(exporter: TestExporter):
    logfire.instrument_openai_agents()

    @function_tool
    def tool():
        raise RuntimeError("Ouch, don't do that again!")

    agent = Agent(name='Start Agent', tools=[tool])
    await Runner.run(agent, input='Call the tool.')

    assert simplify_spans(exporter.exported_spans_as_dict(parse_json_attributes=True)) == snapshot(
        [
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 3000000000,
                'end_time': 4000000000,
                'attributes': {
                    'gen_ai.request.model': 'gpt-4o',
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.span_type': 'span',
                    'response_id': 'resp_67d17435ebcc8191b68300d26c22b0f90273f8a636c82b58',
                    'gen_ai.response.model': 'gpt-4o-2024-08-06',
                    'gen_ai.system': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'raw_input': [{'content': 'Call the tool.', 'role': 'user'}],
                    'events': [
                        {'event.name': 'gen_ai.user.message', 'content': 'Call the tool.', 'role': 'user'},
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'id': 'call_OpJ32C09GImFzxYLe01MiOOd',
                                    'type': 'function',
                                    'function': {'name': 'tool', 'arguments': '{}'},
                                }
                            ],
                        },
                    ],
                    'gen_ai.usage.input_tokens': 244,
                    'gen_ai.usage.output_tokens': 10,
                    'logfire.msg': "Responses API with 'gpt-4o'",
                },
            },
            {
                'name': 'Function: {name}',
                'context': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 5000000000,
                'end_time': 7000000000,
                'attributes': {
                    'logfire.msg_template': 'Function: {name}',
                    'logfire.span_type': 'span',
                    'logfire.level_num': 17,
                    'name': 'tool',
                    'input': {},
                    'output': "An error occurred while running the tool. Please try again. Error: Ouch, don't do that again!",
                    'mcp_data': 'null',
                    'gen_ai.system': 'openai',
                    'error': {
                        'message': 'Error running tool (non-fatal)',
                        'data': {'tool_name': 'tool', 'error': "Ouch, don't do that again!"},
                    },
                    'logfire.msg': 'Function: tool failed: Error running tool (non-fatal)',
                },
                'events': [
                    {
                        'name': 'exception',
                        'timestamp': 6000000000,
                        'attributes': {
                            'exception.type': 'RuntimeError',
                            'exception.message': "Ouch, don't do that again!",
                            'exception.stacktrace': "RuntimeError: Ouch, don't do that again!",
                            'exception.escaped': 'False',
                        },
                    }
                ],
            },
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 9, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'start_time': 8000000000,
                'end_time': 9000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_function_tool_exception',
                    'code.lineno': 123,
                    'gen_ai.request.model': 'gpt-4o',
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.span_type': 'span',
                    'response_id': 'resp_67d17436e29481919a2bd269518a8a3e0273f8a636c82b58',
                    'gen_ai.response.model': 'gpt-4o-2024-08-06',
                    'gen_ai.system': 'openai',
                    'gen_ai.operation.name': 'chat',
                    'raw_input': [
                        {'content': 'Call the tool.', 'role': 'user'},
                        {
                            'id': 'fc_67d1743683b4819192c2f0487f38fa280273f8a636c82b58',
                            'arguments': '{}',
                            'call_id': 'call_OpJ32C09GImFzxYLe01MiOOd',
                            'name': 'tool',
                            'type': 'function_call',
                            'status': 'completed',
                        },
                        {
                            'call_id': 'call_OpJ32C09GImFzxYLe01MiOOd',
                            'output': "An error occurred while running the tool. Please try again. Error: Ouch, don't do that again!",
                            'type': 'function_call_output',
                        },
                    ],
                    'events': [
                        {'event.name': 'gen_ai.user.message', 'content': 'Call the tool.', 'role': 'user'},
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'id': 'call_OpJ32C09GImFzxYLe01MiOOd',
                                    'type': 'function',
                                    'function': {'name': 'tool', 'arguments': '{}'},
                                }
                            ],
                        },
                        {
                            'event.name': 'gen_ai.tool.message',
                            'role': 'tool',
                            'id': 'call_OpJ32C09GImFzxYLe01MiOOd',
                            'content': "An error occurred while running the tool. Please try again. Error: Ouch, don't do that again!",
                            'name': 'tool',
                        },
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'content': 'It seems there was an error when trying to call the tool. If you need help with something specific, feel free to let me know!',
                            'role': 'assistant',
                        },
                    ],
                    'gen_ai.usage.input_tokens': 283,
                    'gen_ai.usage.output_tokens': 30,
                    'logfire.msg': "Responses API with 'gpt-4o'",
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 10000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_function_tool_exception',
                    'code.lineno': 123,
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'logfire.span_type': 'span',
                    'name': 'Start Agent',
                    'handoffs': [],
                    'tools': ['tool'],
                    'output_type': 'str',
                    'gen_ai.system': 'openai',
                    'logfire.msg': "Agent run: 'Start Agent'",
                },
            },
            {
                'name': 'OpenAI Agents trace: {name}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 11000000000,
                'attributes': {
                    'code.filepath': 'test_openai_agents.py',
                    'code.function': 'test_function_tool_exception',
                    'code.lineno': 123,
                    'name': 'Agent workflow',
                    'group_id': 'null',
                    'metadata': 'null',
                    'tracing': 'null',
                    'logfire.msg_template': 'OpenAI Agents trace: {name}',
                    'logfire.msg': 'OpenAI Agents trace: Agent workflow',
                    'logfire.span_type': 'span',
                    'agent_trace_id': IsStr(),
                },
            },
        ]
    )

async def test_voice_pipeline(exporter: TestExporter, vcr_allow_bytes: None):
    logfire.instrument_openai_agents()

    agent = Agent(name='Assistant')
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
    buffer = np.zeros(2400, dtype=np.int16)
    audio_input = AudioInput(buffer=buffer)
    result = await pipeline.run(audio_input)
    assert [{k: v for k, v in event.__dict__.items() if k != 'data'} async for event in result.stream()] == snapshot(
        [
            {'event': 'turn_started', 'type': 'voice_stream_event_lifecycle'},
            {'type': 'voice_stream_event_audio'},
            {'type': 'voice_stream_event_audio'},
            {'type': 'voice_stream_event_audio'},
            {'type': 'voice_stream_event_audio'},
            {'type': 'voice_stream_event_audio'},
            {'event': 'turn_ended', 'type': 'voice_stream_event_lifecycle'},
            {'event': 'session_ended', 'type': 'voice_stream_event_lifecycle'},
        ]
    )

    assert simplify_spans(exporter.exported_spans_as_dict(parse_json_attributes=True)) == snapshot(
        [
            {
                'name': 'Speech → Text with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'logfire.msg_template': 'Speech → Text with {gen_ai.request.model!r}',
                    'logfire.span_type': 'span',
                    'input': {'format': 'pcm'},
                    'output': 'あたし',
                    'gen_ai.request.model': 'gpt-4o-transcribe',
                    'gen_ai.system': 'openai',
                    'model_config': {'temperature': None, 'language': None, 'prompt': None},
                    'gen_ai.response.model': 'gpt-4o-transcribe',
                    'logfire.msg': "Speech → Text with 'gpt-4o-transcribe': あたし",
                },
            },
            {
                'name': 'Responses API with {gen_ai.request.model!r}',
                'context': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'start_time': 5000000000,
                'end_time': 6000000000,
                'attributes': {
                    'gen_ai.request.model': 'gpt-4o',
                    'response_id': 'resp_0f4c5a783ebc79bc00699336f0df488194bddca1fa9a623a2e',
                    'gen_ai.system': 'openai',
                    'gen_ai.response.model': 'gpt-4o-2024-08-06',
                    'gen_ai.operation.name': 'chat',
                    'raw_input': [{'role': 'user', 'content': 'あたし'}],
                    'events': [
                        {'event.name': 'gen_ai.user.message', 'content': 'あたし', 'role': 'user'},
                        {
                            'event.name': 'gen_ai.assistant.message',
                            'content': '「"あたし"」は日本語で「私」を意味する女性が多く使う一人称です。何か特定のことについてお話ししますか？',
                            'role': 'assistant',
                        },
                    ],
                    'gen_ai.usage.input_tokens': 10,
                    'gen_ai.usage.output_tokens': 41,
                    'logfire.msg_template': 'Responses API with {gen_ai.request.model!r}',
                    'logfire.msg': "Responses API with 'gpt-4o'",
                    'logfire.span_type': 'span',
                },
            },
            {
                'name': 'Agent run: {name!r}',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 4000000000,
                'end_time': 8000000000,
                'attributes': {
                    'logfire.msg_template': 'Agent run: {name!r}',
                    'name': 'Assistant',
                    'handoffs': [],
                    'tools': [],
                    'output_type': 'str',
                    'logfire.span_type': 'span',
                    'gen_ai.system': 'openai',
                    'logfire.msg': "Agent run: 'Assistant'",
                },
            },
            {
                'name': 'Text → Speech',
                'context': {'trace_id': 1, 'span_id': 11, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 9, 'is_remote': False},
                'start_time': 9000000000,
                'end_time': 10000000000,
                'attributes': {
                    'logfire.msg_template': 'Text → Speech',
                    'logfire.span_type': 'span',
                    'input': '「"あたし"」は日本語で「私」を意味する女性が多く使う一人称です。何か特定のことについてお話ししますか？',
                    'output': {'format': 'pcm'},
                    'model_config': {
                        'voice': None,
                        'instructions': 'You will receive partial sentences. Do not complete the sentence just read out the text.',
                        'speed': None,
                    },
                    'first_content_at': IsStr(),
                    'gen_ai.request.model': 'gpt-4o-mini-tts',
                    'gen_ai.response.model': 'gpt-4o-mini-tts',
                    'gen_ai.system': 'openai',
                    'logfire.msg': 'Text → Speech: 「"あたし"」は日本語で「私」を意味する女性が多く使う一人称です。何か特定のことについてお話ししますか？',
                },
            },
            {
                'name': 'Text → Speech group',
                'context': {'trace_id': 1, 'span_id': 9, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 7000000000,
                'end_time': 11000000000,
                'attributes': {
                    'logfire.msg_template': 'Text → Speech group',
                    'logfire.span_type': 'span',
                    'input': '「"あたし"」は日本語で「私」を意味する女性が多く使う一人称です。何か特定のことについてお話ししますか？',
                    'gen_ai.system': 'openai',
                    'logfire.msg': 'Text → Speech group: 「"あたし"」は日本語で「私」を意味する女性が多く使う一人称です。何か特定のことについてお話ししますか？',
                },
            },
            {
                'name': 'OpenAI Agents trace: {name}',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 12000000000,
                'attributes': {
                    'name': 'Voice Agent',
                    'metadata': 'null',
                    'tracing': 'null',
                    'logfire.msg_template': 'OpenAI Agents trace: {name}',
                    'logfire.span_type': 'span',
                    'logfire.msg': 'OpenAI Agents trace: Voice Agent',
                    'agent_trace_id': IsStr(),
                    'group_id': IsStr(),
                },
            },
        ]
    )

