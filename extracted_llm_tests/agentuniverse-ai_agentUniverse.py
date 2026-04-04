# agentuniverse-ai/agentUniverse
# 13 test functions with real LLM calls
# Source: https://github.com/agentuniverse-ai/agentUniverse


# --- tests/test_agentuniverse/unit/agent/memory/test_memory.py ---

    def test_summarize_memory_1(self) -> None:
        langchain_memory = self.chat_memory.as_langchain()
        langchain_memory.memory_key = 'history'
        llm_chain = ConversationChain(llm=langchain_memory.llm,
                                      memory=langchain_memory)
        conversation = llm_chain.predict(input='Who am I？')
        print(conversation)

    def test_summarize_memory_2(self) -> None:
        langchain_memory = self.chat_memory.as_langchain()
        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"], template=template
        )
        llm_chain = LLMChain(llm=langchain_memory.llm, prompt=prompt,
                             memory=langchain_memory)
        conversation = llm_chain.predict(human_input='Who am I？')
        print(conversation)

    def test_truncate_memory_1(self) -> None:
        self.chat_memory.type = MemoryTypeEnum.SHORT_TERM
        langchain_memory = self.chat_memory.as_langchain()
        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"], template=template
        )
        llm_chain = LLMChain(llm=langchain_memory.llm, prompt=prompt,
                             memory=langchain_memory)
        conversation = llm_chain.predict(human_input='Who am I？')
        print(conversation)


# --- tests/test_agentuniverse/unit/llm/test_gemini_openai_style_llm.py ---

    def test_call(self) -> None:
        messages = [
            {
                "role": "user",
                "content": "hi, please introduce yourself",
            }
        ]
        output = self.llm.call(messages=messages, streaming=False)
        print(output.__str__())

    def test_acall(self) -> None:
        messages = [
            {
                "role": "user",
                "content": "hi, please introduce yourself",
            }
        ]
        output = asyncio.run(self.llm.acall(messages=messages, streaming=False))
        print(output.__str__())

    def test_call_stream(self):
        messages = [
            {
                "role": "user",
                "content": "hi, please introduce yourself",
            }
        ]
        for chunk in self.llm.call(messages=messages, streaming=True):
            print(chunk.text, end='')
        print()

    def test_acall_stream(self):
        messages = [
            {
                "role": "user",
                "content": "hi, please introduce yourself",
            }
        ]
        asyncio.run(self.call_stream(messages=messages))

    def test_as_langchain(self):
        langchain_llm = self.llm.as_langchain()
        llm_chain = ConversationChain(llm=langchain_llm)
        res = llm_chain.predict(input='hello')
        print(res)


# --- tests/test_agentuniverse/unit/llm/test_llm.py ---

    def test_call(self) -> None:
        messages = [
            {
                "role": "user",
                "content": "hi, please introduce yourself",
            }
        ]
        output = self.llm.call(messages=messages, streaming=False)
        print(output.__str__())

    def test_acall(self) -> None:
        messages = [
            {
                "role": "user",
                "content": "hi, please introduce yourself",
            }
        ]
        output = asyncio.run(self.llm.acall(messages=messages, streaming=False))
        print(output.__str__())

    def test_call_stream(self):
        messages = [
            {
                "role": "user",
                "content": "hi, please introduce yourself",
            }
        ]
        for chunk in self.llm.call(messages=messages, streaming=True):
            print(chunk.text, end='')
        print()

    def test_acall_stream(self):
        messages = [
            {
                "role": "user",
                "content": "hi, please introduce yourself",
            }
        ]
        asyncio.run(self.call_stream(messages=messages))

    def test_as_langchain(self):
        langchain_llm = self.llm.as_langchain()
        llm_chain = ConversationChain(llm=langchain_llm)
        res = llm_chain.predict(input='hello')
        print(res)

