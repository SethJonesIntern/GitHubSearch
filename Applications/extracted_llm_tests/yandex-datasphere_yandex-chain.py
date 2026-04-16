# yandex-datasphere/yandex-chain
# 2 LLM-backed test functions across 4 test files
# Source: https://github.com/yandex-datasphere/yandex-chain

# --- tests/test_ChatYandexGPT.py ---

    def test_create_from_file(self):
        YGPT = ChatYandexGPT(config="tests/config.json")
        res = YGPT([HumanMessage(content='Imagine no possessions...')])
        self.assertTrue(isinstance(res,AIMessage))
        self.assertGreater(len(res.content), 10)
        self.assertGreater(YGPT.totalTokens,0)
        self.assertGreater(YGPT.completionTokens,0)
        self.assertGreater(YGPT.inputTextTokens,0)

    def test_full_model(self):
        YGPT = ChatYandexGPT(config="tests/config.json",use_lite=False)
        res = YGPT([HumanMessage(content='Imagine no possessions...')])
        self.assertTrue(isinstance(res,AIMessage))
        self.assertGreater(len(res.content), 10)
        self.assertGreater(YGPT.totalTokens,0)
        self.assertGreater(YGPT.completionTokens,0)
        self.assertGreater(YGPT.inputTextTokens,0)

