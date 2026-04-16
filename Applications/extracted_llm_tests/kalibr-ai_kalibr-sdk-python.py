# kalibr-ai/kalibr-sdk-python
# 5 LLM-backed test functions across 18 test files
# Source: https://github.com/kalibr-ai/kalibr-sdk-python

# --- tests/test_instrumentation.py ---

    def test_openai_cost_calculation(self):
        """Test OpenAI cost calculation"""
        from kalibr.instrumentation.openai_instr import OpenAICostAdapter
        
        adapter = OpenAICostAdapter()
        
        # Test GPT-4o-mini pricing
        cost = adapter.calculate_cost(
            "gpt-4o-mini",
            {"prompt_tokens": 1000, "completion_tokens": 500}
        )
        
        # GPT-4o-mini: $0.15 input, $0.60 output per 1M tokens
        expected = (1000 / 1_000_000 * 0.15) + (500 / 1_000_000 * 0.60)
        assert abs(cost - expected) < 0.000001

    def test_google_cost_calculation(self):
        """Test Google cost calculation"""
        from kalibr.instrumentation.google_instr import GoogleCostAdapter
        
        adapter = GoogleCostAdapter()
        
        # Test Gemini 1.5 Flash pricing
        cost = adapter.calculate_cost(
            "gemini-1.5-flash",
            {"prompt_tokens": 1000, "completion_tokens": 500}
        )
        
        # Gemini 1.5 Flash: $0.075 input, $0.30 output per 1M tokens
        expected = (1000 / 1_000_000 * 0.075) + (500 / 1_000_000 * 0.30)
        assert abs(cost - expected) < 0.000001

    def test_openai_adapter_consistency(self):
        """Test that OpenAI instrumentation adapter matches pricing module"""
        from kalibr.instrumentation.openai_instr import OpenAICostAdapter
        from kalibr.pricing import compute_cost as pricing_compute_cost
        
        adapter = OpenAICostAdapter()
        
        # Test several models
        test_cases = [
            ("gpt-4o", 1000, 500),
            ("gpt-4", 2000, 1000),
            ("gpt-4o-mini", 5000, 2500),
        ]
        
        for model, input_tokens, output_tokens in test_cases:
            adapter_cost = adapter.calculate_cost(
                model, 
                {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}
            )
            pricing_cost = pricing_compute_cost("openai", model, input_tokens, output_tokens)
            assert adapter_cost == pricing_cost, f"Mismatch for {model}"

    def test_google_adapter_consistency(self):
        """Test that Google instrumentation adapter matches pricing module"""
        from kalibr.instrumentation.google_instr import GoogleCostAdapter
        from kalibr.pricing import compute_cost as pricing_compute_cost
        
        adapter = GoogleCostAdapter()
        
        # Test several models
        test_cases = [
            ("gemini-1.5-pro", 1000, 500),
            ("gemini-1.5-flash", 2000, 1000),
            ("gemini-pro", 5000, 2500),
        ]
        
        for model, input_tokens, output_tokens in test_cases:
            adapter_cost = adapter.calculate_cost(
                model,
                {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}
            )
            pricing_cost = pricing_compute_cost("google", model, input_tokens, output_tokens)
            assert adapter_cost == pricing_cost, f"Mismatch for {model}"

    def test_all_adapters_same_cost_for_same_model(self):
        """Test that using different adapters for same vendor produces same cost"""
        from kalibr.cost_adapter import OpenAICostAdapter as CoreOpenAIAdapter
        from kalibr.instrumentation.openai_instr import OpenAICostAdapter as InstrOpenAIAdapter
        
        core_adapter = CoreOpenAIAdapter()
        instr_adapter = InstrOpenAIAdapter()
        
        # Core adapter uses compute_cost(model, tokens_in, tokens_out)
        # Instrumentation adapter uses calculate_cost(model, usage_dict)
        core_cost = core_adapter.compute_cost("gpt-4o", 1000, 500)
        instr_cost = instr_adapter.calculate_cost(
            "gpt-4o",
            {"prompt_tokens": 1000, "completion_tokens": 500}
        )
        
        assert core_cost == instr_cost

