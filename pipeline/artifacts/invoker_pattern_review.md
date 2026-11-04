# Invoker pattern review — keep/drop before the full run

Collision-prone bare-verb patterns (>=3 invokers). For each, sample call sites.
Mark KEEP (receiver is really an agent/model) or DROP (tool/util/unrelated).

## agno  `.run`   (2084 invokers / 10 repos)   [ ] KEEP  [ ] DROP
    current_agent.run(f'Current board state:\n{st.session_state.game_board.get_board_state()}\
    current_agent.run(f'Invalid move: {message}\n\nCurrent board state:\n{st.session_state.gam
    st.session_state.legal_team.run(combined_query)
    st.session_state.legal_team.run(f"Based on this previous analysis:    \n                  
    st.session_state.legal_team.run(f"Based on this previous analysis:\n                      
    st.session_state.legal_team.run(combined_query)

## agno  `.arun`   (937 invokers / 12 repos)   [ ] KEEP  [ ] DROP
    structured_output_agent.arun(prompt)
    st.session_state.agent.arun(user_input)
    travel_planner.arun(prompt)
    agent.arun(message)
    AgentAsJudgeEval(name=case.name, criteria=case.criteria, scoring_strategy='binary', model=
    case.agent.arun(input=case.input, images=images, audio=audio, stream=True, stream_events=T

## langchain  `.invoke`   (441 invokers / 24 repos)   [ ] KEEP  [ ] DROP
    template.invoke(original_params)
    model_with_tool.invoke([SystemMessage(content=prompt), *messages])
    model.invoke(messages)
    rag_chain.invoke({'context': docs, 'question': question})
    chain.invoke({'question': question, 'context': docs})
    model.invoke(msg)

## langchain_core  `.run`   (283 invokers / 15 repos)   [ ] KEEP  [ ] DROP
    mcp_tool.run(input_data)
    mcp_tool.run(input_data, config=config, **extra_kwargs)
    mcp_tool.run(input_data)
    mcp_tool.run(None)
    mcp_tool.run(input_data)
    mcp_tool.run('simple string input')

## pydantic_ai  `.run`   (226 invokers / 14 repos)   [ ] KEEP  [ ] DROP
    reviewer.run(review_input, model_settings=model_settings)
    designer.run(cleaned_prompt, model_settings=model_settings)
    builder.run(build_input, model_settings=model_settings)
    builder.run(repair_input, model_settings=model_settings)
    builder.run(repair_input, model_settings=model_settings)
    agent.run(user_prompt, deps=_build_dependencies(input), message_history=message_history)

## langchain_core  `.invoke`   (188 invokers / 21 repos)   [ ] KEEP  [ ] DROP
    component_tool.invoke(input={'expression': '10/2'})
    component_tool.invoke(input={'expression': '2+2'})
    component_tool.invoke(input={'expression': '3+3'})
    component_tool.invoke(input={'expression': '5+5'})
    component_tool.invoke(input={'expression': '4*4'})
    component_tool.invoke(input={'expression': '1+1'})

## agno  `.print_response`   (160 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    agent.print_response("Create a new file called 'report.txt' with high priority", stream=Tr
    agent.print_response('Restart the web server service', stream=True)
    agent.print_response(prompt, stream=False)
    agent.print_response(input='Hi, my Social Security Number is 123-45-6789. Can you help me 
    agent.print_response(input='Can you help me understand your return policy?')
    agent.print_response(input='Hi, my Social Security Number is 123-45-6789. Can you help me 

## agent_framework  `.run`   (154 invokers / 12 repos)   [ ] KEEP  [ ] DROP
    planner_agent.run(planner_prompt)
    critic_agent.run(critic_prompt)
    workflow.run(task)
    summarizer.run('\n\n'.join(sections))
    workflow.run(task)
    workflow.run("I'm a US citizen flying from Boston to Lisbon on 2026-09-12 and I need to kn

## langgraph  `.invoke`   (154 invokers / 21 repos)   [ ] KEEP  [ ] DROP
    graph.invoke({'count': 0}, config=cfg)
    graph.invoke({'count': 0}, config=cfg)
    graph.invoke(None, config=cfg)
    graph.invoke({'count': 0}, config=cfg)
    graph.invoke({'count': 0}, config={'configurable': {'thread_id': tid2}})
    graph.invoke({'count': 0}, config={'configurable': {'thread_id': tid1}})

## crewai  `.kickoff`   (153 invokers / 11 repos)   [ ] KEEP  [ ] DROP
    agent.kickoff(prompt)
    agent.kickoff(query)
    crew.kickoff()
    crew.kickoff()
    crew.kickoff()
    crew.kickoff()

## langchain_core  `.ainvoke`   (130 invokers / 15 repos)   [ ] KEEP  [ ] DROP
    tool.ainvoke.assert_called_once_with(input={'id': 789})
    tool.ainvoke.assert_called_once_with(input={'user_id': 456})
    tool.ainvoke.assert_called_once_with(input={'id': 123})
    tool.ainvoke.assert_called_once_with(input={'data': 'test'})
    tool.ainvoke.assert_called_once_with(input={'action': 'count'})
    tool.ainvoke.assert_called_once_with(input=complex_args)

## agno  `.aprint_response`   (124 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    case.agent.aprint_response(input=case.input, images=images, audio=audio, stream=True, sess
    agent.aprint_response('What is the capital of South Africa?', session_id='user1_session_1'
    agent.aprint_response('What is the capital of China?', session_id='user1_session_2', user_
    agent.aprint_response('What is the capital of France?', session_id='user1_session_3', user
    agent.aprint_response('What is the population of India?', session_id='user2_session_1', us
    agent.aprint_response('What is the currency of Japan?', session_id='user2_session_2', user

## langchain  `.ainvoke`   (100 invokers / 9 repos)   [ ] KEEP  [ ] DROP
    self._summary_model.ainvoke(prompt, config={'metadata': {'lc_source': 'summarization'}})
    model.ainvoke(prompt, config=self._get_runnable_config())
    agent.ainvoke({'messages': [HumanMessage(content='hi')]}, {'configurable': {'thread_id': '
    agent.ainvoke({'messages': [HumanMessage(content='hi')]}, {'configurable': {'thread_id': '
    graph.ainvoke({'messages': [HumanMessage(content='use the deferred calculator')]})
    graph.ainvoke({'messages': [HumanMessage(content='hi')]})

## langchain_openai  `.invoke`   (98 invokers / 14 repos)   [ ] KEEP  [ ] DROP
    llm.invoke('test')
    llm.invoke('test')
    llm.invoke('test')
    llm.invoke('test')
    research_agent.invoke({'messages': [HumanMessage(content=query)]})
    super().invoke(input, config, **kwargs)

## camel  `.step`   (93 invokers / 3 repos)   [ ] KEEP  [ ] DROP
    agent.step(message)
    self.step(input_message)
    self.step(input_message=conditions_and_quality_generation_msg)
    super().step(input_message)
    self.step(input_message=knowledge_graph_generation)
    self.step(input_message=user_message, response_format=MultiHopQA)

## langchain_community  `.invoke`   (93 invokers / 16 repos)   [ ] KEEP  [ ] DROP
    model.invoke(messages)
    model.invoke(messages)
    model.invoke(messages)
    rag.invoke(self.search_query, config={'callbacks': self.get_langchain_callbacks()})
    retriever.invoke(query)
    retriever.invoke(query)

## metagpt  `.run`   (91 invokers / 1 repos)   [ ] KEEP  [ ] DROP
    role1.run(requirement1, user_id='1')
    role.run(requirement, user_id)
    self.search_engine.run(query, max_results=max_results, as_string=False)
    self.search_engine.run(i, as_string=False)
    self.web_browser_engine.run(url, *urls)
    subprocess.run(cmd, check=check, cwd=cwd, env=env)

## langgraph  `.ainvoke`   (83 invokers / 13 repos)   [ ] KEEP  [ ] DROP
    agent.ainvoke(state, config=config)
    model.ainvoke([SystemMessage(content=system_instruction), HumanMessage(content=user_conten
    self.model.ainvoke([human], config=config)
    model.ainvoke.assert_awaited_once()
    supervisor.ainvoke({'messages': [HumanMessage(user_input)]}, config={'configurable': {'thr
    graph.ainvoke(ToolState(messages=[seed]), context=CLIContextSchema(offload_tool_call_id=to

## haystack  `.run`   (62 invokers / 10 repos)   [ ] KEEP  [ ] DROP
    agent.run(messages=messages, system_prompt=effective_prompt, **kwargs)
    asyncio.run(outer())
    toolset.run(agent, messages=[ChatMessage.from_user('hello')])
    toolset.run(agent, messages=[ChatMessage.from_user('What do you know about me?')])
    toolset.run(agent, messages=[ChatMessage.from_user('hello')])
    toolset.run(agent, messages=[ChatMessage.from_user('hi')], system_prompt='Custom prompt.')

## langchain_core  `.generate`   (56 invokers / 3 repos)   [ ] KEEP  [ ] DROP
    nanoid.generate(size=6)
    nanoid.generate(size=6)
    wrapper.generate(messages)
    wrapper.generate(messages)
    wrapper.generate(messages)
    wrapper.generate('not a list')

## langchain  `.run`   (52 invokers / 12 repos)   [ ] KEEP  [ ] DROP
    uvicorn.run('main:app', host='0.0.0.0', port=port, reload=True)
    super().run(query)
    super().run(query)
    search.run(query)
    search.run(query)
    routing_agent.run(question)

## pydantic_ai  `.run_stream`   (52 invokers / 6 repos)   [ ] KEEP  [ ] DROP
    adapter.run_stream(deps=None, on_complete=_on_complete)
    adapter.run_stream(deps=deps, on_complete=_on_complete)
    wrapped_agent.run_stream(user_prompt, model_settings=model_settings)
    simple_agent.run_stream('France')
    agent_with_tool.run_stream('Put my money on square eighteen', deps=18)
    adapter.run_stream(model_settings=model_settings)

## pydantic_ai  `.run_sync`   (46 invokers / 9 repos)   [ ] KEEP  [ ] DROP
    agent.run_sync('Where am I, and apply a filter to show only error spans.', deps=deps)
    agent.run_sync(user_msg_1, deps=Deps(tone='formal'))
    agent.run_sync('Now answer the same question one more time.', message_history=result1.all_
    agent_with_tool.run_sync('Put my money on square eighteen', deps=18)
    simple_agent.run_sync('France')
    simple_agent.run_sync('France')

## mem0  `.add`   (42 invokers / 19 repos)   [ ] KEEP  [ ] DROP
    memory.add.assert_called_once_with('hello', user_id='u', metadata={})
    memory.add.assert_called_once_with('hello', user_id='u', metadata={})
    self.client.add(messages_list, user_id=resolved_user_id, infer=self.infer)
    self._memory.add(messages, **kwargs)
    self._client.add(messages, **kwargs)
    deps.add(dep)

## mem0  `.search`   (42 invokers / 15 repos)   [ ] KEEP  [ ] DROP
    memory.search.assert_called_once_with(query='x', filters={'user_id': 'u'})
    self.client.search(query=query, user_id=resolved_user_id)
    self._memory.search(query, filters=filters, top_k=top_k)
    self._client.search(query, filters=filters, top_k=top_k, rerank=rerank)
    self._client.search(query, **kwargs)
    memory.search(query=question, user_id=user_id, limit=5)

## smolagents  `.run`   (40 invokers / 9 repos)   [ ] KEEP  [ ] DROP
    agent.run(q)
    agent.run(_DUMMY_INPUT)
    agent.run(_DUMMY_INPUT)
    agent.run(_DUMMY_INPUT)
    agent.run(task=task, additional_args={'val_sample_df': val_sample_df, 'task_analysis': sel
    agent.run(task=task, additional_args={'spark': self.spark, 'dataset_uri': self.dataset_uri

## langgraph  `.stream`   (40 invokers / 8 repos)   [ ] KEEP  [ ] DROP
    self.graph.stream(init_agent_state, **args)
    client.stream('hi', thread_id='t1')
    client.stream('hi', thread_id='t1', model_name='other-model')
    client.stream('do things', thread_id='t-parallel')
    client.stream('hi', thread_id='t-empty')
    client.stream('hi', thread_id='t-titles')

## langgraph  `.astream`   (35 invokers / 6 repos)   [ ] KEEP  [ ] DROP
    agent.astream(input_payload, config=stream_config, stream_mode=lg_modes, subgraphs=stream_
    agent.astream(input_payload, config=stream_config, stream_mode=single_mode)
    graph.astream({'count': 0}, config=thread_cfg)
    graph.astream({'messages': []}, stream_mode='values')
    parent_graph.astream({'messages': [classes['HumanMessage'](content='delegate')]}, config={
    agent.astream(Command(resume={'decisions': user_decisions}) if user_decisions else {'messa

## langchain_core  `.stream`   (27 invokers / 9 repos)   [ ] KEEP  [ ] DROP
    self._client.inference_pipelines.data.stream(inference_pipeline_id=self._inference_pipelin
    runnable.stream(inputs)
    runnable.stream(input_value)
    actual_agent._graph.graph.stream(agent._build_graph_input(input_text, actual_agent), confi
    self.agent._graph.graph.stream(self._build_graph_input(query), config=merged_config)
    self.agent._graph.graph.stream(self._build_graph_input(query), config=graph_config, stream

## langchain_openai  `.run`   (25 invokers / 8 repos)   [ ] KEEP  [ ] DROP
    self.wrapped_tool.run({})
    self.wrapped_tool.run(tool_input, *args[1:])
    self.wrapped_tool.run(kwargs)
    agent.run()
    browser_agent.run(max_steps=MAX_STEPS)
    asyncio.run(browser_agent.run(max_steps=MAX_STEPS))

## langchain_openai  `.ainvoke`   (25 invokers / 10 repos)   [ ] KEEP  [ ] DROP
    llm.ainvoke(messages)
    llm.ainvoke(messages)
    overall_chain.ainvoke({'title': 'Tragedy at sunset on the beach', 'era': 'Victorian Englan
    overall_chain.ainvoke({'title': 'Tragedy at sunset on the beach', 'era': 'Victorian Englan
    overall_chain.ainvoke({'title': 'Tragedy at sunset on the beach', 'era': 'Victorian Englan
    chain.ainvoke({'input': 'Tell me a joke about OpenTelemetry'})

## langchain_core  `.astream`   (22 invokers / 7 repos)   [ ] KEEP  [ ] DROP
    runnable.astream(inputs)
    agent.astream({'messages': [('user', task)]}, config={'configurable': {'thread_id': 'conte
    agent.astream({'messages': parent_messages, 'goal_criteria_request': {'request_id': 'reque
    self._chain.astream({self._transcript_key: text}, config={'configurable': {'session_id': s
    synopsis_chain.astream(test_prompts, config={'callbacks': [callback]})
    nvidia_client.astream('Hello!')

## mem0  `.update`   (21 invokers / 5 repos)   [ ] KEEP  [ ] DROP
    self._memory.update(memory_id, data=text)
    self._client.update(memory_id=memory_id, text=text)
    existing.update(data)
    self.memory_client.update(memory_id=actual_memory_id, metadata=metadata)
    self.memory_client.update(memory_id=actual_memory_id, metadata=metadata)
    context.update({'user_id': self.user_id})

## autogen_agentchat  `.run`   (21 invokers / 9 repos)   [ ] KEEP  [ ] DROP
    calc_agent.run(task=prompt)
    calc_agent.run(task=prompt)
    agent.run(task='1+1')
    agent.run(task=multi_modal_message)
    agent.run(task='1+1')
    guarded.run(task='BLOCKME')

## autogen_agentchat  `.run_stream`   (21 invokers / 8 repos)   [ ] KEEP  [ ] DROP
    agent.run_stream(task=user)
    agent.run_stream(task=task)
    assistant_agent.run_stream(task='What is the weather in New York?')
    team.run_stream(task='Increment the number 1 to 3.')
    agent.run_stream(task='What is the weather in New York?')
    team.run_stream(task='Count from 1 to 10, respond one at a time.')

## graphrag  `.search`   (19 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    re.search(pattern_int, record)
    re.search(pattern_bool, record)
    re.search('\\((.*)\\)', record)
    self.dataStore.search(fields, [], fltr, [], OrderByExpr(), 0, topn, idxnms, kb_ids)
    self.dataStore.search(['content_with_weight', 'entity_kwd', 'rank_flt'], [], filters, [mat
    self.dataStore.search(['entity_kwd', 'rank_flt'], [], filters, [], ordr, 0, N, idxnms, kb_

## autogen  `.initiate_chat`   (19 invokers / 6 repos)   [ ] KEEP  [ ] DROP
    executor.initiate_chat(recipient=assistant, message=message, max_turns=10)
    user_proxy_agent.initiate_chat(groupchat_manager, message=query)
    assistant.initiate_chat(user_proxy, message='foo')
    assistant.initiate_chat(user_proxy, message='foo')
    assistant.initiate_chat(user_proxy, message='foo')
    assistant.initiate_chat(user_proxy, message='How can I help you today?')

## langchain_core  `.arun`   (17 invokers / 4 repos)   [ ] KEEP  [ ] DROP
    mcp_tool.arun(input_data)
    mcp_tool.arun(input_data, config=config, **extra_kwargs)
    mcp_tool.arun(input_data)
    mcp_tool.arun(None)
    mcp_tool.arun(input_data)
    mcp_tool.arun('async string input')

## langchain  `.stream`   (16 invokers / 6 repos)   [ ] KEEP  [ ] DROP
    graph.stream(inputs)
    self._agent.stream(state, config=config, context=context, stream_mode=['values', 'messages
    self.stream(message, thread_id=thread_id, **kwargs)
    client.stream(prompt, thread_id=thread_id, subagent_enabled=True, thinking_enabled=False, 
    graph.stream({'messages': [HumanMessage(content='finish')]}, context=context, stream_mode=
    agent.stream({'messages': _messages()}, stream_mode='updates')

## dspy  `.forward`   (15 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    program.forward(inputs='test', outputs='test')
    program.forward(inputs='test', outputs='test', lm=mock_lm)
    program.forward(inputs='test input', outputs='test output')
    program.forward(inputs='test', outputs='test', lm=mock_lm)
    agent_lm.forward(prompt='test prompt')
    ensemble.forward(text1='Test 1', text2='Test 2')

## mem0  `.get_all`   (14 invokers / 9 repos)   [ ] KEEP  [ ] DROP
    memory.get_all.assert_called_once_with(filters={'user_id': 'u'})
    self.client.get_all(user_id=resolved_user_id)
    client.get_all(filters={'user_id': user_id})
    self.memory_client.get_all(filters=filters)
    self.memory_client.get_all(filters=filters)
    self.memory_client.get_all(filters={'AND': [{'user_id': self.default_user_id}, {'metadata'

## langchain_community  `.run`   (14 invokers / 7 repos)   [ ] KEEP  [ ] DROP
    wrapper.run(f'{self.input_value} (site:*)')
    client.run(run_id=run_id)
    client.run(run_id)
    self.db.run(self.query, fetch='cursor')
    self.db.run(self.query, include_columns=self.include_columns)
    tool.run({'query': self.input_value, 'params': self.search_params or {}, 'max_results': se

## langchain_community  `.predict`   (14 invokers / 1 repos)   [ ] KEEP  [ ] DROP
    self.client.predict(prompt_input, **kwargs)
    loaded_model.predict({'product': model_info.model_id})
    loaded_model.predict({'product': model_info.model_id})
    pyfunc_model.predict('MLflow')
    pyfunc_model.predict([{'product': 'MLflow'}] * 2)
    pyfunc_model.predict(question)

## camel  `.chat`   (13 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    self.agent.chat(*args, remote=remote, **kwargs)
    self._async_client.chat(messages=cohere_messages, model=self.model_type, **request_config)
    self._client.chat(messages=cohere_messages, model=self.model_type, **request_config)
    self._client.chat.complete(messages=mistral_messages, model=self.model_type, **request_con
    self._async_client.chat.create(messages=reka_messages, model=self.model_type, **self.model
    self._client.chat.create(messages=reka_messages, model=self.model_type, **self.model_confi

## langchain  `.astream`   (13 invokers / 4 repos)   [ ] KEEP  [ ] DROP
    agent.astream(state, config=run_config, context=context, stream_mode='values')
    streaming_agent.astream(stream_input, stream_mode=['messages', 'updates'], subgraphs=True,
    agent.astream(stream_input, stream_mode=['messages', 'updates', 'custom'], subgraphs=True,
    agent.astream(stream_input, stream_mode=['messages', 'updates', 'custom'], subgraphs=True,
    agent.astream(stream_input, stream_mode=['messages', 'updates'], subgraphs=True, config=co
    agent.astream(stream_input, stream_mode=['messages', 'updates'], subgraphs=True, config=co

## langchain_community  `.ainvoke`   (13 invokers / 4 repos)   [ ] KEEP  [ ] DROP
    structured_model.ainvoke(prompt_input, **kwargs)
    tool_model.ainvoke(prompt_input, **kwargs)
    self.client.ainvoke(prompt_input, **kwargs)
    runnable.ainvoke({'product': 'colorful socks'})
    runnable.ainvoke({'product': 'colorful socks'})
    runnable.ainvoke({'product': 'colorful socks'})

## astrbot  `.get_using_provider`   (13 invokers / 3 repos)   [ ] KEEP  [ ] DROP
    self.context.get_using_provider()
    self.context.get_using_provider()
    self.context.get_using_provider()
    self.context.get_using_provider(session_id)
    self.context.get_using_provider()
    self.context.get_using_provider()

## beeai_framework  `.run`   (12 invokers / 1 repos)   [ ] KEEP  [ ] DROP
    workflow.run(inputs=[AgentWorkflowInput(prompt='Provide a short history of the location.',
    main_agent.run(question, expected_output='Helpful and clear response.')
    main_agent.run(question, expected_output='Helpful and clear response.')
    llm.run([UserMessage(prompt)], stream=True, max_tokens=10)
    agent.run(prompt, max_retries_per_step=3, total_max_retries=10, max_iterations=20)
    agent.run(prompt, max_retries_per_step=3, total_max_retries=10, max_iterations=20)

## crewai  `.kickoff_async`   (11 invokers / 6 repos)   [ ] KEEP  [ ] DROP
    crew.kickoff_async()
    flow.kickoff_async({'answer': row['answer'], 'category': row['category']})
    flow.kickoff_async(cast(Any, task))
    opik_tracker.track(project_name=project_name, tags=['crewai'], name='Flow.kickoff_async', 
    self._original.kickoff_async(inputs)
    crew.kickoff_async()

## langchain_openai  `.stream`   (9 invokers / 3 repos)   [ ] KEEP  [ ] DROP
    runnable.stream({'product': 'colorful socks'})
    runnable.stream({'product': 'colorful socks'})
    runnable.stream({'product': 'colorful socks'})
    chain.stream({'input': 'Tell me a joke about OpenTelemetry'})
    chain.stream({'input': 'Tell me a joke about OpenTelemetry'})
    chain.stream({'input': 'Tell me a joke about OpenTelemetry'})

## semantic_kernel  `.invoke`   (9 invokers / 5 repos)   [ ] KEEP  [ ] DROP
    agent.invoke(history)
    kernel.invoke(chat_function, KernelArguments(user_input=user_input, chat_history=chat_hist
    kernel.invoke(function, KernelArguments(num1=5, num2=3))
    self._plan.invoke(**kwargs)
    self._kernel.invoke(function, **kwargs)
    self._kernel.invoke(self._kernel.plugins[plugin_name][function_name], **kwargs)

## haystack  `.run_async`   (9 invokers / 3 repos)   [ ] KEEP  [ ] DROP
    agent.run_async(messages=messages, system_prompt=effective_prompt, **kwargs)
    self.async_pipeline.run_async({'embedder': {'text': text}})
    self.async_pipeline.run_async(data={'llm': llm_input})
    rag_pipeline.run_async(data=data, include_outputs_from={'retriever', 'llm'})
    retriever.run_async(query='How many languages are spoken around the world today?')
    pipeline.run_async({'llm': {'prompt': question}})

## langchain  `.call`   (9 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    mock.call(content_sha256=hashlib.sha256(memory_content.encode('utf-8')).hexdigest())
    tool.call()
    tool.call(location='Munich')
    tool.call()
    tool.call()
    tool.call()

## langchain  `.astream_events`   (8 invokers / 3 repos)   [ ] KEEP  [ ] DROP
    agent.astream_events(input_dict, config={'callbacks': [AgentAsyncHandler(self.log), token_
    agent.astream_events({'messages': [HumanMessage(content='go')]}, version='v3')
    agent.astream_events({'messages': [HumanMessage(content='hi')]}, version='v3')
    agent.astream_events({'messages': [HumanMessage(content='go')]}, version='v3')
    agent.astream_events({'messages': [HumanMessage(content='go')]}, version='v3')
    agent.astream_events({'messages': [HumanMessage(content='go')]}, version='v3')

## crewai  `.kickoff_for_each`   (8 invokers / 3 repos)   [ ] KEEP  [ ] DROP
    crew.kickoff_for_each([{}])
    crew.kickoff_for_each(inputs=[{'input1': 'input1'}, {'input2': 'input2'}])
    crew.kickoff_for_each(inputs=inputs)
    crew.kickoff_for_each(inputs=[])
    crew.kickoff_for_each(inputs=inputs)
    crew.kickoff_for_each('invalid input')

## langchain_openai  `.call`   (8 invokers / 1 repos)   [ ] KEEP  [ ] DROP
    mock.call({'input': 'Say the word: Hi\n\nThis is the expect criteria for your final answer
    mock.call({'input': 'Say the word: Hi\n\nThis is the expect criteria for your final answer
    mock.call()
    mock.call('training_data.pkl')
    mock.call().load()
    mock.call()

## langchain_openai  `.astream`   (8 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    runnable.astream({'product': 'colorful socks'})
    runnable.astream({'product': 'colorful socks'})
    runnable.astream({'product': 'colorful socks'})
    chain.astream({'input': 'Tell me a joke about OpenTelemetry'})
    chain.astream({'input': 'Tell me a joke about OpenTelemetry'})
    chain.astream({'input': 'Tell me a joke about OpenTelemetry'})

## langchain_core  `.predict`   (6 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    model.predict(2, 5)
    model.predict(2, 5)
    loaded_model.predict({'messages': [{'role': 'user', 'content': 'hi'}], 'custom_inputs': {'
    loaded_model.predict({'messages': [{'role': 'user', 'content': 'hi'}]})
    sklearn_knn_model.predict(X)
    reloaded_model.predict(X)

## astrbot  `.text_chat`   (6 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    provider.text_chat(prompt=prompt, contexts=contexts, system_prompt=system_prompt, **kwargs
    retry_provider.text_chat(prompt=prompt, contexts=contexts, system_prompt=system_prompt, **
    fallback_provider.text_chat(prompt=prompt, contexts=contexts, system_prompt=system_prompt,
    fallback_provider.text_chat(prompt=prompt, contexts=contexts, system_prompt=system_prompt,
    fallback_provider.text_chat(prompt=prompt, contexts=contexts, system_prompt=system_prompt,
    old_provider.text_chat.assert_awaited_once()

## agent_framework  `.run_stream`   (5 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    agent.run_stream(stream_task)
    workflow.run_stream(initial_message)
    workflow.run_stream(task)
    workflow.run_stream(task)
    agent.run_stream(msgs, thread=thread)

## langchain_community  `.stream`   (5 invokers / 3 repos)   [ ] KEEP  [ ] DROP
    run_client.log().stream()
    runnable.stream(input={'product': 'colorful socks'}, config={'configurable': {'session_id'
    runnable.stream(input={'product': 'colorful socks'}, config={'configurable': {'session_id'
    runnable.stream(input={'product': 'colorful socks'}, config={'configurable': {'session_id'
    loaded_model.stream(msg)
    model.stream(input, config)

## agentops  `.record`   (4 invokers / 3 repos)   [ ] KEEP  [ ] DROP
    self._ao_client.record(LLMEvent(**event_params))
    self._ao_client.record(ToolEvent(name=event.tool.name, params=params))
    self._ao_client.record(ErrorEvent(details=str(err)))
    agentops.record(tool_event)
    agentops.record(tool_event)
    agentops.record(agentops.ErrorEvent(exception=e, trigger_event=tool_event))

## agentops  `.init`   (4 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    agentops.init(auto_start_session=False)
    agentops.init()
    self.init(dev_backend)
    self.init(backend)
    self.tracer.init()
    agentops.init(config.agentops_api_key, tags=['software_company'])

## langchain  `.predict`   (4 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    loaded_pyfunc_model.predict([{'product': 'MLflow'}])
    rf.predict([[2, 1]])
    llm.predict(**langchain_prompt_variables)
    llm.predict(**langchain_prompt_variables)
    llm.predict(**langchain_prompt_variables)
    llm.predict(**langchain_prompt_variables)

## autogen_core  `.publish_message`   (4 invokers / 1 repos)   [ ] KEEP  [ ] DROP
    self.publish_message(Message(content=val), DefaultTopicId())
    self.publish_message(Message(content=message.content), DefaultTopicId())
    self.publish_message(BookSections(sections=sections_with_images), topic_id=TopicId('BookGe
    self.publish_message(book_content, topic_id=TopicId('ImageGeneratorAgent', source=self.id.
    runtime.publish_message(StoryRequest(prompt=prompt), topic_id=TopicId('StoryGeneratorAgent
    self.publish_message(quote_request, topic_id=DefaultTopicId())

## crewai  `.kickoff_for_each_async`   (4 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    crew.kickoff_for_each_async(inputs=[{'input1': 'input1'}, {'input2': 'input2'}])
    crew.kickoff_for_each_async(inputs)
    crew.kickoff_for_each_async([])
    crew.kickoff_for_each_async(inputs=inputs)

## langchain  `.generate`   (4 invokers / 1 repos)   [ ] KEEP  [ ] DROP
    self.context_entity_recall_prompt.generate(llm=self.llm, data=StringIO(text=text), callbac
    self.extract_keyphrases_prompt.generate(data=StringIO(text=text), llm=self.llm, callbacks=
    self.answer_generation_prompt.generate(data=SummaryAndQuestions(questions=questions, summa
    self.question_generation_prompt.generate(data=GenerateQuestionsPromptInput(text=text, keyp

## langchain_cohere  `.invoke`   (4 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    chain.invoke(small_docs)
    chain.invoke(small_docs)
    chain.invoke(small_docs)
    chat.invoke(prompt)

## livekit  `.generate_reply`   (4 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    session.generate_reply(instructions=f"Greet {attendee['name'].split()[0]} and start the RS
    session.generate_reply(instructions=f'Say exactly: {greeting}')
    session.generate_reply(instructions="Greet the user as a knowledgeable assistant. Explain 
    session.generate_reply(instructions='Greet the user warmly as a travel assistant and ask h

## langgraph  `.batch`   (4 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    store.batch([PutOp(namespace, key, None) for key in to_delete])
    self._backing.batch([op])
    self._backing.batch([op for _, op in backing_ops])
    store.batch(ops)

## semantic_kernel  `.invoke_prompt`   (4 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    kernel.invoke_prompt('Is sushi the best food ever?')
    kernel.invoke_prompt('Hello?')
    kernel.invoke_prompt(prompt)
    self._kernel.invoke_prompt(prompt, **kwargs)
    kernel.invoke_prompt(prompt='Tell me a short joke about programming', arguments=KernelArgu

## autogen_core  `.send_message`   (3 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    runtime.send_message(Message(3), AgentId('checker', 'default'))
    runtime.send_message(Message('What is the result of 545.34567 * 34555.34'), assistant_id)
    runtime.send_message(UserMessage(content=msg, metadata={}), router_id)

## graphrag  `local_search`   (3 invokers / 1 repos)   [ ] KEEP  [ ] DROP
    local_search(config, entities=entities_combined, communities=communities_combined, communi
    api.local_search(config=config, entities=final_entities, communities=final_communities, co
    local_search(config=graphrag_config, entities=entities, communities=communities, community

## langchain  `.batch`   (3 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    super().batch(ops)
    self._original.batch(inputs, **kwargs)
    llm_chain.batch([{'number': 2}, {'number': 3}])
    llm_chain.batch(batch)

## langchain  `.acall`   (3 invokers / 1 repos)   [ ] KEEP  [ ] DROP
    function_tool.acall(2)
    function_tool.acall(1, 2)
    tool.acall()
    tool.acall()
    tool.acall()
    tool.acall(x=1)

## langchain_core  `.acall`   (3 invokers / 1 repos)   [ ] KEEP  [ ] DROP
    self.aclient.acall(**self.const_kwargs(messages, stream=False))
    self.aclient.acall(**self.const_kwargs(messages, stream=True))
    llm.aclient.acall(**llm.const_kwargs(messages, stream=True))

## langchain_anthropic  `.invoke`   (3 invokers / 3 repos)   [ ] KEEP  [ ] DROP
    super().invoke(input, config, **kwargs)
    self._chat.invoke([('system', system), ('human', user)])
    chat.invoke(messages)

## langchain  `.arun`   (3 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    task_tool.arun({'description': description, 'subagent_type': subagent_type, 'runtime': run
    langchain_tool.arun('1')
    langchain_tool.arun('1')
    langchain_tool2.arun({'x': 1, 'y': 2})

## graphrag  `basic_search`   (3 invokers / 1 repos)   [ ] KEEP  [ ] DROP
    basic_search(config, text_units=text_units_combined, query=query, callbacks=callbacks)
    api.basic_search(config=config, text_units=final_text_units, query=query, verbose=verbose)
    basic_search(config=graphrag_config, text_units=text_units, query=query)

## langgraph  `.abatch`   (3 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    store.abatch([PutOp(namespace, key, None) for key in to_delete])
    self._backing.abatch([op])
    self._backing.abatch([op for _, op in backing_ops])

## langchain_experimental  `.run`   (3 invokers / 1 repos)   [ ] KEEP  [ ] DROP
    python_repl.run(cleaned_code)
    tool.run(self.code)
    python_repl.run(code)

## metagpt  `.run_project`   (3 invokers / 1 repos)   [ ] KEEP  [ ] DROP
    company.run_project(idea)
    self.run_project(idea=idea, send_to=send_to)
    self.run_project(idea=idea, send_to=send_to)

## swarm  `.run`   (3 invokers / 2 repos)   [ ] KEEP  [ ] DROP
    client.run(agent, messages, context_variables, **kwargs)
    app.run(agent=self._get_first_task(), messages=[], context_variables=inputs, debug=agentst
    app.run(agent=self._get_first_task(), messages=[], context_variables=inputs, debug=agentst

