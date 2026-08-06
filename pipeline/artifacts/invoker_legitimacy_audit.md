# Direct-invoker legitimacy audit (by receiver variable)

For each bare-verb pattern: top receiver variables it matches. Agent/model receivers = KEEP; tool/util = DROP.

## agno `.run` (2084 inv / 10 repos)
receivers: agent(1318), team(313), workflow(307), asyncio(69), structured_output_agent(55), wf(32)

## agno `.arun` (937 inv / 12 repos)
receivers: agent(494), team(166), workflow(119), wf(19), reasoning_agent(17), async_test_agent(15)

## langchain `.invoke` (441 inv / 24 repos)
receivers: agent(212), read_file_tool(30), llm(24), grep_search_tool(24), graph(17), parent_agent(15)

## langchain_core `.run` (283 inv / 15 repos)
receivers: asyncio(216), commands(60), session(8), mcp_tool(6), subprocess(6), guarded(6)

## pydantic_ai `.run` (226 inv / 14 repos)
receivers: agent(118), asyncio(27), marvin(12), source(11), main_agent(9), orchestrator(6)

## langchain_core `.invoke` (188 inv / 21 repos)
receivers: tool(23), llm(23), invoker(18), chain(17), template(16), registered_bedrock_client(13)

## agno `.print_response` (160 inv / 2 repos)
receivers: agent(162), reasoning_agent(14), blocking_team(7), shopping_team(7), team(6), task_workflow(5)

## langgraph `.invoke` (154 inv / 21 repos)
receivers: graph(46), llm(40), agent(28), app(10), tracked_graph(7), model(5)

## agent_framework `.run` (154 inv / 12 repos)
receivers: workflow(53), agent(44), _agent(37), asyncio(15), analyst(5), agent1(4)

## crewai `.kickoff` (153 inv / 11 repos)
receivers: crew(108), flow(22), flow2(5), guarded(4), agent(3), test_crew(3)

## langchain_core `.ainvoke` (130 inv / 15 repos)
receivers: tool(24), llm(12), prompt_template(12), template(11), agent(9), wrapper(7)

## agno `.aprint_response` (124 inv / 2 repos)
receivers: agent(129), team(14), workflow(12), chat_agent(8), multi_language_team(6), async_agent(4)

## langchain `.ainvoke` (100 inv / 9 repos)
receivers: agent(23), graph(15), grep_search_tool(13), glob_search_tool(9), ls_tool(5), execute_tool(5)

## langchain_openai `.invoke` (98 inv / 14 repos)
receivers: chain(21), llm(20), model_with_tools(16), model(5), graph(3), overall_chain(3)

## camel `.step` (93 inv / 3 repos)
receivers: agent(28), self(23), web_agent(6), generator_agent(4), assistant_agent(4), coordinator_agent(4)

## langchain_community `.invoke` (93 inv / 16 repos)
receivers: loaded_model(24), model(23), chain(23), llm(11), retriever(7), runnable(6)

## metagpt `.run` (91 inv / 1 repos)
receivers: todo(21), subprocess(17), executor(14), role(11), action(7), search_engine(4)

## langgraph `.ainvoke` (83 inv / 13 repos)
receivers: graph(30), subgraph(15), agent(14), llm(8), chain(7), model(3)

## haystack `.run` (62 inv / 10 repos)
receivers: pipe(26), toolset(13), self(5), _generator(5), agent(3), retriever(2)

## langchain_core `.generate` (56 inv / 3 repos)
receivers: wrapper(11), self(4), nanoid(3), statement_generator_prompt(3), single_turn_prompt(3), multi_turn_prompt(3)

## langchain `.run` (52 inv / 12 repos)
receivers: asyncio(30), Runner(6), search(3), subprocess(3), agent_app(3), agent(2)

## pydantic_ai `.run_stream` (52 inv / 6 repos)
receivers: agent(31), test_agent(7), adapter(4), node(4), self(3), wrapped_agent(1)

## pydantic_ai `.run_sync` (46 inv / 9 repos)
receivers: agent(26), simple_agent(7), conn(6), asyncio(4), agent_with_tool(2), _original(1)

## mem0 `.add` (42 inv / 19 repos)
receivers: memory(12), memory_client(8), client(5), _client(4), _mem0(3), rag_client(2)

## mem0 `.search` (42 inv / 15 repos)
receivers: client(11), memory(5), _client(4), _mem0(4), memory_client(3), mem0_memory(3)

## langgraph `.stream` (40 inv / 8 repos)
receivers: client(30), app(7), agent(4), graph(2), chain(1), compiled(1)

## smolagents `.run` (40 inv / 9 repos)
receivers: agent(36), asyncio(2), runner(2), manager_agent(1), ctx(1), manager_code_agent(1)

## langgraph `.astream` (35 inv / 6 repos)
receivers: agent(18), graph(8), mock_tool_graph(2), parent_graph(1), final_graph(1), runnable(1)

## langchain_core `.stream` (27 inv / 9 repos)
receivers: node(8), client(7), graph(4), synopsis_chain(3), registered_bedrock_client(3), runnable(2)

## langchain_openai `.ainvoke` (25 inv / 10 repos)
receivers: llm(10), overall_chain(3), chain(3), workflow(2), model(2), guarded(1)

## langchain_openai `.run` (25 inv / 8 repos)
receivers: asyncio(21), wrapped_tool(3), browser_agent(3), commands(3), agent(2), chain(2)

## langchain_core `.astream` (22 inv / 7 repos)
receivers: registered_bedrock_client(4), chain(4), orch(4), agent(3), nvidia_client(2), llm(2)

## mem0 `.update` (21 inv / 5 repos)
receivers: context(3), payload(3), memory_client(2), event_data(2), memory_store(2), _memory(1)

## autogen_agentchat `.run` (21 inv / 9 repos)
receivers: agent(14), team(4), calc_agent(2), agent2(2), guarded(1), _agent1(1)

## autogen_agentchat `.run_stream` (21 inv / 8 repos)
receivers: agent(17), team(11), assistant_agent(1)

## autogen `.initiate_chat` (19 inv / 6 repos)
receivers: assistant(9), user_proxy(9), executor(1), user_proxy_agent(1), agent_1(1), user(1)

## graphrag `.search` (19 inv / 2 repos)
receivers: re(11), dataStore(4), file_pattern(3), document_collection(3), search_engine(1), action(1)

## langchain_core `.arun` (17 inv / 4 repos)
receivers: self(23), mcp_tool(6), tool(6), guarded_tool(2), original(2), _orig_tool(1)

## langchain `.stream` (16 inv / 6 repos)
receivers: agent(4), parent_agent(4), graph(2), _agent(1), self(1), client(1)

## dspy `.forward` (15 inv / 2 repos)
receivers: agent(6), program(4), detector(3), agent_lm(1), ensemble(1), resolver(1)

## mem0 `.get_all` (14 inv / 9 repos)
receivers: memory_client(10), client(3), memory(3), primary_memory(2), _mem0(2), memory_instance(1)

## langchain_community `.predict` (14 inv / 1 repos)
receivers: pyfunc_loaded_model(31), pyfunc_model(8), loaded_model(7), client(1), loaded_pyfunc_model(1)

## langchain_community `.run` (14 inv / 7 repos)
receivers: Runner(3), wrapper(2), client(2), db(2), tool(2), asyncio(2)

## langchain_community `.ainvoke` (13 inv / 4 repos)
receivers: runnable(6), model(4), chain(3), structured_model(1), tool_model(1), client(1)

## langchain `.astream` (13 inv / 4 repos)
receivers: agent(9), graph(3), streaming_agent(1), parent_agent(1), llm_chain(1), subagent(1)

## astrbot `.get_using_provider` (13 inv / 3 repos)
receivers: context(9)

## camel `.chat` (13 inv / 2 repos)
receivers: agent(2), _async_client(1), _client(1)

## beeai_framework `.run` (12 inv / 1 repos)
receivers: llm(4), agent(3), main_agent(2), workflow(1), tool(1), multiplication_workflow(1)

## crewai `.kickoff_async` (11 inv / 6 repos)
receivers: crew(5), flow(4), _original(1), self(1)

## langchain `.call` (9 inv / 2 repos)
receivers: tool(13), mock(1), vs_ret_tool(1)

## semantic_kernel `.invoke` (9 inv / 5 repos)
receivers: kernel(3), _kernel(2), agent(1), _plan(1), self(1), guarded(1)

## langchain_openai `.stream` (9 inv / 3 repos)
receivers: chain(5), runnable(3), agent(1)

## haystack `.run_async` (9 inv / 3 repos)
receivers: agent(2), async_pipeline(2), retriever(2), pipeline(2), rag_pipeline(1), pipe(1)

## crewai `.kickoff_for_each` (8 inv / 3 repos)
receivers: crew(8)

## langchain_openai `.call` (8 inv / 1 repos)
receivers: mock(8), llm(5)

## langchain_openai `.astream` (8 inv / 2 repos)
receivers: chain(6), runnable(3), client(2)

## langchain `.astream_events` (8 inv / 3 repos)
receivers: agent(7), agent_executor(1)

## astrbot `.text_chat` (6 inv / 2 repos)
receivers: fallback_provider(3), provider(2), retry_provider(1), llm_provider(1)

## langchain_core `.predict` (6 inv / 2 repos)
receivers: model(2), loaded_model(2), sklearn_knn_model(1), reloaded_model(1), nli_classifier(1)

## agent_framework `.run_stream` (5 inv / 2 repos)
receivers: workflow(3), agent(2)

## langchain_community `.stream` (5 inv / 3 repos)
receivers: model(4), runnable(3), loaded_model(1), retrieval_chain(1)

## langgraph `.batch` (4 inv / 2 repos)
receivers: store(2), _backing(2)

## langchain_cohere `.invoke` (4 inv / 2 repos)
receivers: chain(3), chat(1)

## livekit `.generate_reply` (4 inv / 2 repos)
receivers: session(4)

## semantic_kernel `.invoke_prompt` (4 inv / 2 repos)
receivers: kernel(4), _kernel(1)

## agentops `.record` (4 inv / 3 repos)
receivers: _ao_client(3), agentops(3)

## langchain `.generate` (4 inv / 1 repos)
receivers: context_entity_recall_prompt(1), extract_keyphrases_prompt(1), answer_generation_prompt(1), question_generation_prompt(1)

## crewai `.kickoff_for_each_async` (4 inv / 2 repos)
receivers: crew(4)

## langchain `.predict` (4 inv / 2 repos)
receivers: llm(4), loaded_pyfunc_model(1), rf(1)

## agentops `.init` (4 inv / 2 repos)
receivers: agentops(3), self(2), tracer(1)

## autogen_core `.publish_message` (4 inv / 1 repos)
receivers: self(8), runtime(1)

## langchain `.arun` (3 inv / 2 repos)
receivers: langchain_tool(2), task_tool(1), langchain_tool2(1)

## autogen_core `.send_message` (3 inv / 2 repos)
receivers: runtime(3)

## langchain_experimental `.run` (3 inv / 1 repos)
receivers: python_repl(2), tool(1)

## langchain_core `.acall` (3 inv / 1 repos)
receivers: aclient(3)

## metagpt `.run_project` (3 inv / 1 repos)
receivers: self(2), company(1)

## swarm `.run` (3 inv / 2 repos)
receivers: app(2), client(1)

## langgraph `.abatch` (3 inv / 2 repos)
receivers: _backing(2), store(1)

## langchain `.acall` (3 inv / 1 repos)
receivers: tool(7), function_tool(2)

## langchain_anthropic `.invoke` (3 inv / 3 repos)
receivers: _chat(1), chat(1)

## langchain `.batch` (3 inv / 2 repos)
receivers: llm_chain(2), _original(1)

## semantic_kernel `.get_response` (2 inv / 2 repos)
receivers: agent(2)

## langchain `.stream_events` (2 inv / 1 repos)
receivers: agent(2)

## langchain_core `.astream_events` (2 inv / 2 repos)
receivers: runnable(1), graph(1)

## langchain_community `.arun` (2 inv / 1 repos)
receivers: retriever_tool(2)

## langchain `.abatch` (2 inv / 2 repos)
receivers: llm_chain(1)

## langchain_core `.agenerate` (2 inv / 2 repos)
receivers: llm(2), langchain_model(1)

## langchain_core `.transform` (2 inv / 1 repos)
receivers: parser(2)

## langchain_community `.batch` (2 inv / 1 repos)
receivers: model(4), loaded_model(2), runnable(1)

## langchain_aws `.acall` (1 inv / 1 repos)
receivers: _create_graph_engine(1)

## langchain_community `.abatch` (1 inv / 1 repos)
receivers: model(3), loaded_model(1)

## langchain_aws `.invoke` (1 inv / 1 repos)
receivers: 

## autogen_agentchat `.on_messages_stream` (1 inv / 1 repos)
receivers: self(1)

## camel `.astep` (1 inv / 1 repos)
receivers: user_agent(1), assistant_agent(1)

## autogen `.generate_reply` (1 inv / 1 repos)
receivers: agent(1)

## autogen `.a_generate_reply` (1 inv / 1 repos)
receivers: agent(1)

## agentops `.end_session` (1 inv / 1 repos)
receivers: agentops(2)

## honcho `.create` (1 inv / 1 repos)
receivers: conclusions_scope(1)

## honcho `.chat` (1 inv / 1 repos)
receivers: ai_peer_obj(2), target_peer(1)

## lagent `.chat` (1 inv / 1 repos)
receivers: agent(1)

## langchain_core `.atransform` (1 inv / 1 repos)
receivers: parser(1)

## langchain_community `.generate` (1 inv / 1 repos)
receivers: instance(1)

## langchain_openai `.batch` (1 inv / 1 repos)
receivers: llm(1)

## langchain_openai `.astream_events` (1 inv / 1 repos)
receivers: runnable(1)

## langchain_openai `.abatch` (1 inv / 1 repos)
receivers: llm(1)

## langchain_openai `.agenerate` (1 inv / 1 repos)
receivers: self(1)

## langchain_google_genai `.invoke` (1 inv / 1 repos)
receivers: 

## langchain_experimental `.invoke` (1 inv / 1 repos)
receivers: agent_csv(1)

## langchain_core `.call` (1 inv / 1 repos)
receivers: aclient(1)

## langchain_core `.batch` (1 inv / 1 repos)
receivers: 

## langchain_community `.astream` (1 inv / 1 repos)
receivers: model(3), loaded_model(1)

## langgraph `.astream_events` (1 inv / 1 repos)
receivers: research_graph(1)

## langchain_openai `.predict` (1 inv / 1 repos)
receivers: model(1)

## langchain_together `.invoke` (1 inv / 1 repos)
receivers: chat(1)

## openlit `.init` (1 inv / 1 repos)
receivers: openlit(1)

## semantic_kernel `.get_chat_message_contents` (1 inv / 1 repos)
receivers: client(1)

## semantic_kernel `.get_chat_message_content` (1 inv / 1 repos)
receivers: chat_service(1)

## semantic_kernel `.invoke_async` (1 inv / 1 repos)
receivers: plan(1)

