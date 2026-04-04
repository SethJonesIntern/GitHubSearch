# nottelabs/notte
# 2 test functions with real LLM calls
# Source: https://github.com/nottelabs/notte


# --- tests/llms/test_engine.py ---

    def test_inner_llm_completion_schema_roundtrip(self):
        """The full InnerLlmCompletion schema must have form_fill value properties after Gemini transform."""
        from notte_core.agent_types import AgentCompletion

        schema = AgentCompletion.InnerLlmCompletion.model_json_schema()
        result = fix_schema_for_gemini(schema)
        # Find form_fill in oneOf
        form_fill = None
        for item in result["properties"]["action"]["oneOf"]:
            if item.get("properties", {}).get("type", {}).get("const") == "form_fill":
                form_fill = item
                break
        assert form_fill is not None, "form_fill action not found in schema"
        value_schema = form_fill["properties"]["value"]
        assert "properties" in value_schema
        assert len(value_schema["properties"]) > 10

    def test_form_fill_action_schema_roundtrip(self):
        """The actual FormFillAction schema must survive Gemini transformation with proper type info."""
        from notte_core.actions.actions import FormFillAction

        schema = FormFillAction.model_json_schema()
        result = fix_schema_for_gemini(schema)
        value_schema = result["properties"]["value"]
        assert "properties" in value_schema
        assert "email" in value_schema["properties"]
        assert "username" in value_schema["properties"]
        assert "password" in value_schema["properties"]
        assert len(value_schema["properties"]) > 10

