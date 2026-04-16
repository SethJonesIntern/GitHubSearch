# swarmzero/swarmzero
# 11 LLM-backed test functions across 17 test files
# Source: https://github.com/swarmzero/swarmzero

# --- tests/chat/test_chat_manager.py ---

async def test_generate_response_with_generic_llm(agent, db_manager):
    chat_manager = ChatManager(agent, user_id="123", session_id="abc", agent_id="test_agent", swarm_id="test_swarm")
    user_message = ChatMessage(role=MessageRole.USER, content="Hello!")

    response = ""
    async for chunk in chat_manager.generate_response(db_manager, user_message, []):
        if isinstance(chunk, list):
            response += ''.join(chunk)
        elif chunk is not None:
            response += chunk
    assert response.split("END_OF_STREAM")[0] == "chat response"

    messages = await chat_manager.get_messages(db_manager)
    assert len(messages) == 2
    assert messages[0].content == "Hello!"
    assert messages[1].content == "chat response"

async def test_execute_task_success(multi_modal_agent):
    chat_manager = ChatManager(
        multi_modal_agent, user_id="123", session_id="abc", agent_id="test_agent", swarm_id="test_swarm"
    )

    result = ""
    async for chunk in chat_manager._execute_task("task_id_123", event_handler=None):
        if chunk is not None:
            result += chunk

    result = result.replace("END_OF_STREAM", "")
    assert result == "multimodal response"
    multi_modal_agent._arun_step.assert_called_once_with("task_id_123")


# --- tests/llms/test_llm.py ---

def test_openai_llm_initialization(tools, instruction, sdk_context):
    openai_llm = OpenAILLM(OpenAI(model="gpt-3.5-turbo"), tools, instruction, sdk_context=sdk_context)
    print(f"Agent: {openai_llm.agent}")
    print(f"Tools: {openai_llm.tools}")
    print(f"System Prompt: {openai_llm.system_prompt}")

    assert openai_llm.agent is not None
    assert isinstance(openai_llm.agent, AgentRunner)
    assert openai_llm.tools == tools
    assert instruction in openai_llm.system_prompt

def test_azureopenai_llm_initialization(tools, instruction, sdk_context):
    azureopenai = AzureOpenAILLM(
        AzureOpenAI(azure_deployment="gpt-3.5-turbo", azure_endpoint="https://YOUR_RESOURCE_NAME.openai.azure.com/"),
        tools,
        instruction,
        sdk_context=sdk_context,
    )
    print(f"Agent: {azureopenai.agent}")
    print(f"Tools: {azureopenai.tools}")
    print(f"System Prompt: {azureopenai.system_prompt}")

    assert azureopenai.agent is not None
    assert isinstance(azureopenai.agent, AgentRunner)
    assert azureopenai.tools == tools
    assert instruction in azureopenai.system_prompt

def test_openai_multimodal_llm_initialization(tools, instruction, sdk_context):
    openai_llm = OpenAIMultiModalLLM(OpenAIMultiModal(model="gpt-4"), tools, instruction, sdk_context=sdk_context)
    assert openai_llm.agent is not None
    assert isinstance(openai_llm.agent, llama_index.core.agent.runner.base.AgentRunner)
    assert openai_llm.tools == tools
    assert instruction in openai_llm.system_prompt

def test_claude_llm_initialization(tools, instruction, sdk_context):
    claude_llm = ClaudeLLM(Anthropic(model="claude-3-opus-20240229"), tools, instruction, sdk_context=sdk_context)
    assert claude_llm.agent is not None
    assert isinstance(claude_llm.agent, llama_index.core.agent.runner.base.AgentRunner)
    assert claude_llm.tools == tools
    assert instruction in claude_llm.system_prompt

def test_llama_llm_initialization(tools, instruction, sdk_context):
    llama_llm = OllamaLLM(Ollama(model="llama3"), tools, instruction, sdk_context=sdk_context)
    assert llama_llm.agent is not None
    assert isinstance(llama_llm.agent, llama_index.core.agent.runner.base.AgentRunner)
    assert llama_llm.tools == tools
    assert instruction in llama_llm.system_prompt

def test_mistral_llm_initialization(tools, instruction, sdk_context):
    mistral_llm = MistralLLM(
        MistralAI(model="mistral-large-latest", api_key="mistral_api_key"), tools, instruction, sdk_context=sdk_context
    )
    assert mistral_llm.agent is not None
    assert isinstance(mistral_llm.agent, llama_index.core.agent.runner.base.AgentRunner)
    assert mistral_llm.tools == tools
    assert instruction in mistral_llm.system_prompt

def test_nebius_llm_initialization(tools, instruction, sdk_context):
    nebius_llm = NebiuslLLM(Nebius(model="nebius-7b"), tools, instruction, sdk_context=sdk_context)
    assert nebius_llm.agent is not None
    assert isinstance(nebius_llm.agent, llama_index.core.agent.runner.base.AgentRunner)
    assert nebius_llm.tools == tools
    assert instruction in nebius_llm.system_prompt

def test_retrieval_openai_llm_initialization(empty_tools, instruction, tool_retriever, sdk_context):
    openai_llm = OpenAILLM(
        OpenAI(model="gpt-3.5-turbo"), empty_tools, instruction, tool_retriever=tool_retriever, sdk_context=sdk_context
    )
    assert openai_llm.agent is not None
    assert isinstance(openai_llm.agent, AgentRunner)
    assert openai_llm.tools == empty_tools
    assert instruction in openai_llm.system_prompt
    assert openai_llm.tool_retriever == tool_retriever

def test_retrieval_ollamallm_initialization(empty_tools, instruction, tool_retriever, sdk_context):
    ollamallm = OllamaLLM(
        Ollama(model="llama3"), empty_tools, instruction, tool_retriever=tool_retriever, sdk_context=sdk_context
    )
    assert ollamallm.agent is not None
    assert isinstance(ollamallm.agent, llama_index.core.agent.runner.base.AgentRunner)
    assert ollamallm.tools == empty_tools
    assert instruction in ollamallm.system_prompt
    assert ollamallm.tool_retriever == tool_retriever

