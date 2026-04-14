# raga-ai-hub/RagaAI-Catalyst
# 2 test functions with real LLM calls
# Source: https://github.com/raga-ai-hub/RagaAI-Catalyst


# --- tests/test_catalyst/test_prompt_manager.py ---

def test_compile_prompt(prompt_manager):
    prompt = prompt_manager.get_prompt(prompt_name="test2", version="v2")
    compiled_prompt = prompt.compile(
    system1='What is chocolate?',
    system2 = "How it is made")
    def get_openai_response(prompt):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt
        )
        return response.choices[0].message.content
    get_openai_response(compiled_prompt)

def test_compile_prompt_no_modelname(prompt_manager):
    with pytest.raises(openai.BadRequestError,match="you must provide a model parameter"):

        prompt = prompt_manager.get_prompt(prompt_name="test2", version="v2")
        compiled_prompt = prompt.compile(
        system1='What is chocolate?',
        system2 = "How it is made")
        def get_openai_response(prompt):
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="",
                messages=prompt
            )
            return response.choices[0].message.content
        get_openai_response(compiled_prompt)

