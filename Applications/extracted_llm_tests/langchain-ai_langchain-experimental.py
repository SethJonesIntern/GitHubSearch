# langchain-ai/langchain-experimental
# 20 LLM-backed test functions across 34 test files
# Source: https://github.com/langchain-ai/langchain-experimental

# --- libs/experimental/tests/integration_tests/test_video_captioning.py ---

def test_video_captioning_hard() -> None:
    """Test input that is considered hard for this chain to process."""
    URL = """
    https://ia904700.us.archive.org/22/items/any-chibes/X2Download.com
    -FXX%20USA%20%C2%ABPromo%20Noon%20-%204A%20Every%20Day%EF%BF%BD%EF
    %BF%BD%C2%BB%20November%202021%EF%BF%BD%EF%BF%BD-%281080p60%29.mp4
    """
    chain = VideoCaptioningChain(  # type: ignore[call-arg]
        llm=ChatOpenAI(
            model="gpt-4",
            max_completion_tokens=4000,
        )
    )
    srt_content = chain.run(video_file_path=URL)

    assert (
        "mustache" in srt_content
        and "Any chives?" in srt_content
        and "How easy? A little tighter." in srt_content
        and "it's a little tight in" in srt_content
        and "every day" in srt_content
    )


# --- libs/experimental/tests/integration_tests/chains/test_cpal.py ---

    def test_against_pal_chain_doc(self) -> None:
        """
        Test CPAL chain against the first example in the PAL chain notebook doc:

        https://github.com/langchain-ai/langchain/blob/master/docs/extras/modules/chains/additional/pal.ipynb
        """

        narrative_input = (
            "Jan has three times the number of pets as Marcia."
            " Marcia has two more pets than Cindy."
            " If Cindy has four pets, how many total pets do the three have?"
        )

        llm = OpenAI(temperature=0, max_tokens=512)
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        answer = cpal_chain.run(narrative_input)

        """
        >>> story._outcome_table
             name                            code  value depends_on
        0   cindy                            pass    4.0         []
        1  marcia  marcia.value = cindy.value + 2    6.0    [cindy]
        2     jan    jan.value = marcia.value * 3   18.0   [marcia]

        """
        self.assertEqual(answer, 28.0)

    def test_hallucinating(self) -> None:
        """
        Test CPAL approach does not hallucinate when given
        an invalid entity in the question.

        The PAL chain would hallucinates here!
        """

        narrative_input = (
            "Jan has three times the number of pets as Marcia."
            "Marcia has two more pets than Cindy."
            "If Cindy has ten pets, how many pets does Barak have?"
        )
        llm = OpenAI(temperature=0, max_tokens=512)
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        with pytest.raises(Exception) as e_info:
            print(e_info)  # noqa: T201
            cpal_chain.run(narrative_input)

    def test_causal_mediator(self) -> None:
        """
        Test CPAL approach on causal mediator.
        """

        narrative_input = (
            "jan has three times the number of pets as marcia."
            "marcia has two more pets than cindy."
            "If marcia has ten pets, how many pets does jan have?"
        )
        llm = OpenAI(temperature=0, max_tokens=512)
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        answer = cpal_chain.run(narrative_input)
        self.assertEqual(answer, 30.0)

    def test_draw(self) -> None:
        """
        Test CPAL chain can draw its resulting DAG.
        """
        import os

        narrative_input = (
            "Jan has three times the number of pets as Marcia."
            "Marcia has two more pets than Cindy."
            "If Marcia has ten pets, how many pets does Jan have?"
        )
        llm = OpenAI(temperature=0, max_tokens=512)
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        cpal_chain.run(narrative_input)
        path = "graph.svg"
        cpal_chain.draw(path=path)
        self.assertTrue(os.path.exists(path))


# --- libs/experimental/tests/integration_tests/chains/test_pal.py ---

def test_math_prompt() -> None:
    """Test math prompt."""
    llm = OpenAI(temperature=0, max_tokens=512)
    pal_chain = PALChain.from_math_prompt(llm, timeout=None, allow_dangerous_code=False)
    question = (
        "Jan has three times the number of pets as Marcia. "
        "Marcia has two more pets than Cindy. "
        "If Cindy has four pets, how many total pets do the three have?"
    )
    output = pal_chain.run(question)
    assert output == "28"

def test_colored_object_prompt() -> None:
    """Test colored object prompt."""
    llm = OpenAI(temperature=0, max_tokens=512)
    pal_chain = PALChain.from_colored_object_prompt(
        llm, timeout=None, allow_dangerous_code=False
    )
    question = (
        "On the desk, you see two blue booklets, "
        "two purple booklets, and two yellow pairs of sunglasses. "
        "If I remove all the pairs of sunglasses from the desk, "
        "how many purple items remain on it?"
    )
    output = pal_chain.run(question)
    assert output == "2"


# --- libs/experimental/tests/integration_tests/chains/test_sql_database.py ---

def test_sql_database_run() -> None:
    """Test that commands can be run successfully and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison", user_company="Foo")
    with engine.connect() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine)
    db_chain = SQLDatabaseChain.from_llm(OpenAI(temperature=0), db)
    output = db_chain.run("What company does Harrison work at?")
    expected_output = " Harrison works at Foo."
    assert output == expected_output

def test_sql_database_run_update() -> None:
    """Test that update commands run successfully and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison", user_company="Foo")
    with engine.connect() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine)
    db_chain = SQLDatabaseChain.from_llm(OpenAI(temperature=0), db)
    output = db_chain.run("Update Harrison's workplace to Bar")
    expected_output = " Harrison's workplace has been updated to Bar."
    assert output == expected_output
    output = db_chain.run("What company does Harrison work at?")
    expected_output = " Harrison works at Bar."
    assert output == expected_output

def test_sql_database_sequential_chain_run() -> None:
    """Test that commands can be run successfully SEQUENTIALLY
    and returned in correct format."""
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison", user_company="Foo")
    with engine.connect() as conn:
        conn.execute(stmt)
    db = SQLDatabase(engine)
    db_chain = SQLDatabaseSequentialChain.from_llm(OpenAI(temperature=0), db)
    output = db_chain.run("What company does Harrison work at?")
    expected_output = " Harrison works at Foo."
    assert output == expected_output


# --- libs/experimental/tests/integration_tests/chains/test_synthetic_data_openai.py ---

def test_generate_synthetic(synthetic_data_generator: SyntheticDataGenerator) -> None:
    synthetic_results = synthetic_data_generator.generate(
        subject="medical_billing",
        extra="""the name must be chosen at random. Make it something you wouldn't 
        normally choose.""",
        runs=10,
    )
    assert len(synthetic_results) == 10
    for row in synthetic_results:
        assert isinstance(row, MedicalBilling)

async def test_agenerate_synthetic(
    synthetic_data_generator: SyntheticDataGenerator,
) -> None:
    synthetic_results = await synthetic_data_generator.agenerate(
        subject="medical_billing",
        extra="""the name must be chosen at random. Make it something you wouldn't 
        normally choose.""",
        runs=10,
    )
    assert len(synthetic_results) == 10
    for row in synthetic_results:
        assert isinstance(row, MedicalBilling)


# --- libs/experimental/tests/integration_tests/llms/test_anthropic_functions.py ---

    def test_default_chat_anthropic(self) -> None:
        base_model = AnthropicFunctions(model="claude-2")  # type: ignore[call-arg]
        self.assertIsInstance(base_model.model, ChatAnthropic)

        # bind functions
        model = base_model.bind(
            functions=[
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, "
                                "e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            ],
            function_call={"name": "get_current_weather"},
        )

        res = model.invoke("What's the weather in San Francisco?")

        function_call = res.additional_kwargs.get("function_call")
        assert function_call
        self.assertEqual(function_call.get("name"), "get_current_weather")
        self.assertEqual(
            function_call.get("arguments"),
            '{"location": "San Francisco, CA", "unit": "fahrenheit"}',
        )

    def test_bedrock_chat_anthropic(self) -> None:
        """
              const chatBedrock = new ChatBedrock({
          region: process.env.BEDROCK_AWS_REGION ?? "us-east-1",
          model: "anthropic.claude-v2",
          temperature: 0.1,
          credentials: {
            secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
            accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
          },
        });"""
        llm = BedrockChat(  # type: ignore[call-arg]
            model_id="anthropic.claude-v2",
            model_kwargs={"temperature": 0.1},
            region_name="us-east-1",
        )
        base_model = AnthropicFunctions(llm=llm)
        assert isinstance(base_model.model, BedrockChat)

        # bind functions
        model = base_model.bind(
            functions=[
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, "
                                "e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            ],
            function_call={"name": "get_current_weather"},
        )

        res = model.invoke("What's the weather in San Francisco?")

        function_call = res.additional_kwargs.get("function_call")
        assert function_call
        self.assertEqual(function_call.get("name"), "get_current_weather")
        self.assertEqual(
            function_call.get("arguments"),
            '{"location": "San Francisco, CA", "unit": "fahrenheit"}',
        )


# --- libs/experimental/tests/integration_tests/llms/test_ollama_functions.py ---

    def test_default_ollama_functions(self) -> None:
        base_model = OllamaFunctions(model="phi3", format="json")

        # bind functions
        model = base_model.bind_tools(
            tools=[
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, "
                                "e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            ],
            function_call={"name": "get_current_weather"},
        )

        res = model.invoke("What's the weather in San Francisco?")

        self.assertIsInstance(res, AIMessage)
        res = AIMessage(**res.__dict__)
        tool_calls = res.tool_calls
        assert tool_calls
        tool_call = tool_calls[0]
        assert tool_call
        self.assertEqual("get_current_weather", tool_call.get("name"))

    def test_ollama_functions_tools(self) -> None:
        base_model = OllamaFunctions(model="phi3", format="json")
        model = base_model.bind_tools(
            tools=[PubmedQueryRun(), DuckDuckGoSearchResults(max_results=2)]  # type: ignore[call-arg]
        )
        res = model.invoke("What causes lung cancer?")
        self.assertIsInstance(res, AIMessage)
        res = AIMessage(**res.__dict__)
        tool_calls = res.tool_calls
        assert tool_calls
        tool_call = tool_calls[0]
        assert tool_call
        self.assertEqual("pub_med", tool_call.get("name"))

    def test_default_ollama_functions_default_response(self) -> None:
        base_model = OllamaFunctions(model="phi3", format="json")

        # bind functions
        model = base_model.bind_tools(
            tools=[
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, "
                                "e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            ]
        )

        res = model.invoke("What is the capital of France?")

        self.assertIsInstance(res, AIMessage)
        res = AIMessage(**res.__dict__)
        tool_calls = res.tool_calls
        if len(tool_calls) > 0:
            tool_call = tool_calls[0]
            assert tool_call
            self.assertEqual("__conversational_response", tool_call.get("name"))

    def test_ollama_structured_output(self) -> None:
        model = OllamaFunctions(model="phi3")
        structured_llm = model.with_structured_output(Joke, include_raw=False)

        res = structured_llm.invoke("Tell me a joke about cats")
        assert isinstance(res, Joke)

    def test_ollama_structured_output_with_json(self) -> None:
        model = OllamaFunctions(model="phi3")
        joke_schema = convert_to_ollama_tool(Joke)
        structured_llm = model.with_structured_output(joke_schema, include_raw=False)

        res = structured_llm.invoke("Tell me a joke about cats")
        assert "setup" in res
        assert "punchline" in res

    def test_ollama_structured_output_raw(self) -> None:
        model = OllamaFunctions(model="phi3")
        structured_llm = model.with_structured_output(Joke, include_raw=True)

        res = structured_llm.invoke("Tell me a joke about cars")
        assert isinstance(res, dict)
        assert "raw" in res
        assert "parsed" in res
        assert isinstance(res["raw"], AIMessage)
        assert isinstance(res["parsed"], Joke)

