# ag2ai/ag2
# 97 LLM-backed test functions across 393 test files
# Source: https://github.com/ag2ai/ag2

# --- test/test_logging.py ---

def test_log_completion(response, expected_logged_response, db_connection):
    cur = db_connection.cursor()

    sample_completion = get_sample_chat_completion(response)
    autogen.runtime_logging.log_chat_completion(**sample_completion)

    query = """
        SELECT invocation_id, client_id, wrapper_id, request, response, is_cached,
            cost, start_time, source_name FROM chat_completions
    """

    for row in cur.execute(query):
        assert row["invocation_id"] == sample_completion["invocation_id"]
        assert row["client_id"] == sample_completion["client_id"]
        assert row["wrapper_id"] == sample_completion["wrapper_id"]
        assert json.loads(row["request"]) == sample_completion["request"]
        assert json.loads(row["response"]) == expected_logged_response
        assert row["is_cached"] == sample_completion["is_cached"]
        assert row["cost"] == sample_completion["cost"]
        assert row["start_time"] == sample_completion["start_time"]
        assert row["source_name"] == "TestAgent"

def test_log_function_use(db_connection):
    cur = db_connection.cursor()

    source = autogen.AssistantAgent(name="TestAgent", code_execution_config=False)
    func: Callable[[str, int], Any] = dummy_function
    args = {"foo": "bar"}
    returns = True

    autogen.runtime_logging.log_function_use(agent=source, function=func, args=args, returns=returns)

    query = """
        SELECT source_id, source_name, function_name, args, returns, timestamp
        FROM function_calls
    """

    for row in cur.execute(query):
        assert row["source_name"] == "TestAgent"
        assert row["args"] == json.dumps(args)
        assert row["returns"] == json.dumps(returns)

def test_log_new_agent(db_connection):
    from autogen import AssistantAgent

    cur = db_connection.cursor()
    agent_name = "some_assistant"
    config_list = [{"model": "gpt-4o", "api_key": "some_key"}]

    agent = AssistantAgent(agent_name, llm_config={"config_list": config_list})
    init_args = {"foo": "bar", "baz": {"other_key": "other_val"}, "a": None}

    autogen.runtime_logging.log_new_agent(agent, init_args)

    query = """
        SELECT session_id, name, class, init_args FROM agents
    """

    for row in cur.execute(query):
        assert row["session_id"] and str(uuid.UUID(row["session_id"], version=4)) == row["session_id"], (
            "session id is not valid uuid"
        )
        assert row["name"] == agent_name
        assert row["class"] == "AssistantAgent"
        assert row["init_args"] == json.dumps(init_args)

def test_log_oai_wrapper(db_connection):
    from autogen import OpenAIWrapper

    cur = db_connection.cursor()

    llm_config = {"config_list": [{"model": "gpt-4o", "api_key": "some_key", "base_url": "some url"}]}
    init_args = {"llm_config": llm_config, "base_config": {}}
    wrapper = OpenAIWrapper(**llm_config)

    autogen.runtime_logging.log_new_wrapper(wrapper, init_args)

    query = """
        SELECT session_id, init_args FROM oai_wrappers
    """

    for row in cur.execute(query):
        assert row["session_id"] and str(uuid.UUID(row["session_id"], version=4)) == row["session_id"], (
            "session id is not valid uuid"
        )
        saved_init_args = json.loads(row["init_args"])
        assert "config_list" in saved_init_args
        assert "api_key" not in saved_init_args["config_list"][0]
        assert "base_url" in saved_init_args["config_list"][0]
        assert "base_config" in saved_init_args

def test_event_print_default_logger_respects_end_and_flush() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("ag2.event.processor")
    old_handlers = logger.handlers[:]
    old_propagate = logger.propagate
    try:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        handler = EventStreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        event_print("structured", "output", sep="|", end="END", flush=True)

        assert stream.getvalue() == "structured|outputEND"
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        for handler in old_handlers:
            logger.addHandler(handler)
        logger.propagate = old_propagate


# --- test/agentchat/contrib/test_agent_builder.py ---

def test_build(builder: AgentBuilder, credentials_all: Credentials):
    building_task = (
        "Find a paper on arxiv by programming, and analyze its application in some domain. "
        "For example, find a recent paper about gpt-4 on arxiv "
        "and find its potential applications in software."
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        _, agent_config = builder.build(
            building_task=building_task,
            default_llm_config={"temperature": 0, "config_list": credentials_all.config_list},
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": f"{temp_dir}/test_agent_scripts",
                "timeout": 60,
                "use_docker": "python:3",
            },
        )
        _config_check(agent_config)

    # check number of agents
    assert len(agent_config["agent_configs"]) <= builder.max_agents

def test_build_from_library(builder: AgentBuilder, credentials_all: Credentials):
    building_task = (
        "Find a paper on arxiv by programming, and analyze its application in some domain. "
        "For example, find a recent paper about gpt-4 on arxiv "
        "and find its potential applications in software."
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        _, agent_config = builder.build_from_library(
            building_task=building_task,
            library_path_or_json=f"{here}/example_agent_builder_library.json",
            default_llm_config={"temperature": 0, "config_list": credentials_all.config_list},
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": f"{temp_dir}/test_agent_scripts",
                "timeout": 60,
                "use_docker": "python:3",
            },
        )
        _config_check(agent_config)

    # check number of agents
    assert len(agent_config["agent_configs"]) <= builder.max_agents

    builder.clear_all_agents()

    # test embedding similarity selection
    with tempfile.TemporaryDirectory() as temp_dir:
        _, agent_config = builder.build_from_library(
            building_task=building_task,
            library_path_or_json=f"{here}/example_agent_builder_library.json",
            default_llm_config={"temperature": 0, "config_list": credentials_all.config_list},
            embedding_model="all-mpnet-base-v2",
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": f"{temp_dir}/test_agent_scripts",
                "timeout": 60,
                "use_docker": "python:3",
            },
        )
        _config_check(agent_config)

    # check number of agents
    assert len(agent_config["agent_configs"]) <= builder.max_agents

def test_load(builder: AgentBuilder):
    with tempfile.TemporaryDirectory() as temp_dir:
        config_save_path = f"{here}/example_test_agent_builder_config.json"
        json.load(open(config_save_path))  # noqa: SIM115

        _, loaded_agent_configs = builder.load(
            config_save_path,
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": f"{temp_dir}/test_agent_scripts",
                "timeout": 60,
                "use_docker": "python:3",
            },
        )
        print(loaded_agent_configs)

        _config_check(loaded_agent_configs)

def test_clear_agent(builder: AgentBuilder):
    with tempfile.TemporaryDirectory() as temp_dir:
        config_save_path = f"{here}/example_test_agent_builder_config.json"
        builder.load(
            config_save_path,
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": f"{temp_dir}/test_agent_scripts",
                "timeout": 60,
                "use_docker": "python:3",
            },
        )
        builder.clear_all_agents()

        # check if the agent cleared
        assert len(builder.agent_procs_assign) == 0


# --- test/agentchat/contrib/test_captainagent.py ---

def test_captain_agent_from_scratch(credentials_all: Credentials):
    config_list = credentials_all.config_list
    llm_config = {
        "temperature": 0,
        "config_list": config_list,
    }
    nested_config = {
        "autobuild_init_config": {
            "config_file_or_env": os.path.join(KEY_LOC, OAI_CONFIG_LIST),
            "builder_model": "gpt-4o",
            "agent_model": "gpt-4o",
        },
        "autobuild_build_config": {
            "default_llm_config": {"temperature": 1, "top_p": 0.95, "max_tokens": 1500, "seed": 52},
            "code_execution_config": {"timeout": 300, "work_dir": "groupchat", "last_n_messages": 1},
            "coding": True,
        },
        "group_chat_config": {"max_round": 10},
        "group_chat_llm_config": llm_config.copy(),
    }
    captain_agent = CaptainAgent(
        name="captain_agent",
        llm_config=llm_config,
        code_execution_config={"use_docker": False, "work_dir": "groupchat"},
        nested_config=nested_config,
        agent_config_save_path=None,
    )
    captain_user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER")
    task = (
        "Find a paper on arxiv by programming, and analyze its application in some domain. "
        "For example, find a recent paper about gpt-4 on arxiv "
        "and find its potential applications in software."
    )

    result = captain_user_proxy.initiate_chat(captain_agent, message=task, max_turns=4)
    print(result)

def test_captain_agent_with_library(credentials_all: Credentials):
    config_list = credentials_all.config_list
    llm_config = {
        "temperature": 0,
        "config_list": config_list,
    }
    nested_config = {
        "autobuild_init_config": {
            "config_file_or_env": os.path.join(KEY_LOC, OAI_CONFIG_LIST),
            "builder_model": "gpt-4o",
            "agent_model": "gpt-4o",
        },
        "autobuild_build_config": {
            "default_llm_config": {"temperature": 1, "top_p": 0.95, "max_tokens": 1500, "seed": 52},
            "code_execution_config": {"timeout": 300, "work_dir": "groupchat", "last_n_messages": 1},
            "coding": True,
        },
        "autobuild_tool_config": {
            "retriever": "all-mpnet-base-v2",
        },
        "group_chat_config": {"max_round": 10},
        "group_chat_llm_config": llm_config.copy(),
    }
    captain_agent = CaptainAgent(
        name="captain_agent",
        llm_config=llm_config,
        code_execution_config={"use_docker": False, "work_dir": "groupchat"},
        nested_config=nested_config,
        agent_lib="example_test_captainagent.json",
        tool_lib="default",
        agent_config_save_path=None,
    )
    captain_user_proxy = UserProxyAgent(name="captain_user_proxy", human_input_mode="NEVER")
    task = (
        "Find a paper on arxiv by programming, and analyze its application in some domain. "
        "For example, find a recent paper about gpt-4 on arxiv "
        "and find its potential applications in software."
    )

    result = captain_user_proxy.initiate_chat(captain_agent, message=task, max_turns=4)
    print(result)


# --- test/agentchat/contrib/test_gpt_assistant.py ---

def test_gpt_assistant_chat_openai(
    provider: str, credentials_openai_mini: Credentials, credentials_azure: Credentials
) -> None:
    if provider == "openai":
        _test_gpt_assistant_chat(credentials_openai_mini)
    elif provider == "azure":
        _test_gpt_assistant_chat(credentials_azure)
    else:
        raise ValueError(f"Invalid provider: {provider}")

def test_get_assistant_instructions(
    provider: str, credentials_openai_mini: Credentials, credentials_azure: Credentials
) -> None:
    if provider == "openai":
        _test_get_assistant_instructions(credentials_openai_mini)
    elif provider == "azure":
        _test_get_assistant_instructions(credentials_azure)
    else:
        raise ValueError(f"Invalid provider: {provider}")

def test_gpt_assistant_instructions_overwrite(
    provider: str, credentials_openai_mini: Credentials, credentials_azure: Credentials
) -> None:
    if provider == "openai":
        _test_gpt_assistant_instructions_overwrite(credentials_openai_mini)
    elif provider == "azure":
        _test_gpt_assistant_instructions_overwrite(credentials_azure)
    else:
        raise ValueError(f"Invalid provider: {provider}")

def test_gpt_assistant_existing_no_instructions(credentials_openai_mini: Credentials) -> None:
    """Test function to check if the GPTAssistantAgent can retrieve instructions for an existing assistant
    even if the assistant was created with no instructions initially.
    """
    name = f"For_test_gpt_assistant_existing_no_instructions_{uuid.uuid4()}"
    instructions = "This is a test #1"

    assistant = GPTAssistantAgent(
        name,
        instructions=instructions,
        llm_config={
            "config_list": credentials_openai_mini.config_list,
        },
    )

    try:
        assistant_id = assistant.assistant_id

        # create a new assistant with the same ID but no instructions
        assistant = GPTAssistantAgent(
            name,
            llm_config={
                "config_list": credentials_openai_mini.config_list,
            },
            assistant_config={"assistant_id": assistant_id},
        )

        instruction_match = assistant.get_assistant_instructions() == instructions

    finally:
        assistant.delete_assistant()

    assert instruction_match is True

def test_get_assistant_files(credentials_openai_mini: Credentials) -> None:
    """Test function to create a new GPTAssistantAgent, set its instructions, retrieve the instructions,
    and assert that the retrieved instructions match the set instructions.
    """
    current_file_path = os.path.abspath(__file__)
    openai_client = OpenAIWrapper(config_list=credentials_openai_mini.config_list)._clients[0]._oai_client
    file = openai_client.files.create(file=open(current_file_path, "rb"), purpose="assistants")  # noqa: SIM115
    name = f"For_test_get_assistant_files_{uuid.uuid4()}"
    gpt_assistant_api_version = detect_gpt_assistant_api_version()

    # keep it to test older version of assistant config
    assistant = GPTAssistantAgent(
        name,
        instructions="This is a test",
        llm_config={
            "config_list": credentials_openai_mini.config_list,
            "tools": [{"type": "retrieval"}],
            "file_ids": [file.id],
        },
    )

    try:
        if gpt_assistant_api_version == "v1":
            files = assistant.openai_client.beta.assistants.files.list(assistant_id=assistant.assistant_id)
            retrieved_file_ids = [fild.id for fild in files]
        elif gpt_assistant_api_version == "v2":
            oas_assistant = assistant.openai_client.beta.assistants.retrieve(assistant_id=assistant.assistant_id)
            vectorstore_ids = oas_assistant.tool_resources.file_search.vector_store_ids
            retrieved_file_ids = []
            for vectorstore_id in vectorstore_ids:
                files = assistant.openai_client.vector_stores.files.list(vector_store_id=vectorstore_id)
                retrieved_file_ids.extend([fild.id for fild in files])
        expected_file_id = file.id
    finally:
        assistant.delete_assistant()
        openai_client.files.delete(file.id)

    assert expected_file_id in retrieved_file_ids

def test_assistant_retrieval(credentials_openai_mini: Credentials) -> None:
    """Test function to check if the GPTAssistantAgent can retrieve the same assistant"""
    name = f"For_test_assistant_retrieval_{uuid.uuid4()}"

    function_1_schema = {
        "name": "call_function_1",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "description": "This is a test function 1",
    }
    function_2_schema = {
        "name": "call_function_2",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "description": "This is a test function 2",
    }

    openai_client = OpenAIWrapper(config_list=credentials_openai_mini.config_list)._clients[0]._oai_client
    current_file_path = os.path.abspath(__file__)

    file_1 = openai_client.files.create(file=open(current_file_path, "rb"), purpose="assistants")  # noqa: SIM115
    file_2 = openai_client.files.create(file=open(current_file_path, "rb"), purpose="assistants")  # noqa: SIM115

    try:
        all_llm_config = {
            "config_list": credentials_openai_mini.config_list,
        }
        assistant_config = {
            "tools": [
                {"type": "function", "function": function_1_schema},
                {"type": "function", "function": function_2_schema},
                {"type": "retrieval"},
                {"type": "code_interpreter"},
            ],
            "file_ids": [file_1.id, file_2.id],
        }

        name = f"For_test_assistant_retrieval_{uuid.uuid4()}"

        assistant_first = GPTAssistantAgent(
            name,
            instructions="This is a test",
            llm_config=all_llm_config,
            assistant_config=assistant_config,
        )
        candidate_first = retrieve_assistants_by_name(assistant_first.openai_client, name)

        try:
            assistant_second = GPTAssistantAgent(
                name,
                instructions="This is a test",
                llm_config=all_llm_config,
                assistant_config=assistant_config,
            )
            candidate_second = retrieve_assistants_by_name(assistant_second.openai_client, name)

        finally:
            assistant_first.delete_assistant()
            with pytest.raises(openai.NotFoundError):
                assistant_second.delete_assistant()

    finally:
        openai_client.files.delete(file_1.id)
        openai_client.files.delete(file_2.id)

    assert candidate_first == candidate_second
    assert len(candidate_first) == 1

    candidates = retrieve_assistants_by_name(openai_client, name)
    assert len(candidates) == 0

def test_assistant_mismatch_retrieval(credentials_openai_mini: Credentials) -> None:
    """Test function to check if the GPTAssistantAgent can filter out the mismatch assistant"""
    name = f"For_test_assistant_retrieval_{uuid.uuid4()}"

    function_1_schema = {
        "name": "call_function_1",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "description": "This is a test function 1",
    }
    function_2_schema = {
        "name": "call_function_2",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "description": "This is a test function 2",
    }
    function_3_schema = {
        "name": "call_function_other",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "description": "This is a test function 3",
    }

    openai_client = OpenAIWrapper(config_list=credentials_openai_mini.config_list)._clients[0]._oai_client
    current_file_path = os.path.abspath(__file__)
    file_1 = openai_client.files.create(file=open(current_file_path, "rb"), purpose="assistants")  # noqa: SIM115
    file_2 = openai_client.files.create(file=open(current_file_path, "rb"), purpose="assistants")  # noqa: SIM115

    try:
        # keep it to test older version of assistant config
        all_llm_config = {
            "tools": [
                {"type": "function", "function": function_1_schema},
                {"type": "function", "function": function_2_schema},
                {"type": "file_search"},
                {"type": "code_interpreter"},
            ],
            "file_ids": [file_1.id, file_2.id],
            "config_list": credentials_openai_mini.config_list,
        }

        name = f"For_test_assistant_retrieval_{uuid.uuid4()}"

        assistant_first, assistant_instructions_mistaching = None, None
        try:
            assistant_first = GPTAssistantAgent(
                name,
                instructions="This is a test",
                llm_config=all_llm_config,
            )
            candidate_first = retrieve_assistants_by_name(assistant_first.openai_client, name)
            assert len(candidate_first) == 1

            # test instructions mismatch
            assistant_instructions_mistaching = GPTAssistantAgent(
                name,
                instructions="This is a test for mismatch instructions",
                llm_config=all_llm_config,
            )
            candidate_instructions_mistaching = retrieve_assistants_by_name(
                assistant_instructions_mistaching.openai_client, name
            )
            assert len(candidate_instructions_mistaching) == 2

            # test tools mismatch
            tools_mismatch_llm_config = {
                "tools": [
                    {"type": "code_interpreter"},
                    {"type": "file_search"},
                    {"type": "function", "function": function_3_schema},
                ],
                "file_ids": [file_2.id, file_1.id],
                "config_list": credentials_openai_mini.config_list,
            }
            assistant_tools_mistaching = GPTAssistantAgent(
                name,
                instructions="This is a test",
                llm_config=tools_mismatch_llm_config,
            )
            candidate_tools_mismatch = retrieve_assistants_by_name(assistant_tools_mistaching.openai_client, name)
            assert len(candidate_tools_mismatch) == 3

        finally:
            if assistant_first:
                assistant_first.delete_assistant()
            if assistant_instructions_mistaching:
                assistant_instructions_mistaching.delete_assistant()
            if assistant_tools_mistaching:
                assistant_tools_mistaching.delete_assistant()

    finally:
        openai_client.files.delete(file_1.id)
        openai_client.files.delete(file_2.id)

    candidates = retrieve_assistants_by_name(openai_client, name)
    assert len(candidates) == 0

def test_gpt_assistant_tools_overwrite(credentials_openai_mini: Credentials) -> None:
    """Test that the tools of a GPTAssistantAgent can be overwritten or not depending on the value of the
    `overwrite_tools` parameter when creating a new assistant with the same ID.

    Steps:
    1. Create a new GPTAssistantAgent with a set of tools.
    2. Get the ID of the assistant.
    3. Create a new GPTAssistantAgent with the same ID but different tools and `overwrite_tools=True`.
    4. Check that the tools of the assistant have been overwritten with the new ones.
    """
    original_tools = [
        {
            "type": "function",
            "function": {
                "name": "calculateTax",
                "description": "Calculate tax for a given amount",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number", "description": "The amount to calculate tax on"},
                        "tax_rate": {"type": "number", "description": "The tax rate to apply"},
                    },
                    "required": ["amount", "tax_rate"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convertCurrency",
                "description": "Convert currency from one type to another",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number", "description": "The amount to convert"},
                        "from_currency": {"type": "string", "description": "Currency type to convert from"},
                        "to_currency": {"type": "string", "description": "Currency type to convert to"},
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            },
        },
    ]

    new_tools = [
        {
            "type": "function",
            "function": {
                "name": "findRestaurant",
                "description": "Find a restaurant based on cuisine type and location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cuisine": {"type": "string", "description": "Type of cuisine"},
                        "location": {"type": "string", "description": "City or area for the restaurant search"},
                    },
                    "required": ["cuisine", "location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculateMortgage",
                "description": "Calculate monthly mortgage payments",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "principal": {"type": "number", "description": "The principal loan amount"},
                        "interest_rate": {"type": "number", "description": "Annual interest rate"},
                        "years": {"type": "integer", "description": "Number of years for the loan"},
                    },
                    "required": ["principal", "interest_rate", "years"],
                },
            },
        },
    ]

    name = f"For_test_gpt_assistant_tools_overwrite_{uuid.uuid4()}"

    # Create an assistant with original tools
    assistant_org = GPTAssistantAgent(
        name,
        llm_config={
            "config_list": credentials_openai_mini.config_list,
        },
        assistant_config={
            "tools": original_tools,
        },
    )

    assistant_id = assistant_org.assistant_id

    try:
        # Create a new assistant with new tools and overwrite_tools set to True
        assistant = GPTAssistantAgent(
            name,
            llm_config={
                "config_list": credentials_openai_mini.config_list,
            },
            assistant_config={
                "assistant_id": assistant_id,
                "tools": new_tools,
            },
            overwrite_tools=True,
        )

        # Add logic to retrieve the tools from the assistant and assert
        retrieved_tools = assistant.openai_assistant.tools
        retrieved_tools_name = [tool.function.name for tool in retrieved_tools]
    finally:
        assistant_org.delete_assistant()

    assert retrieved_tools_name == [tool["function"]["name"] for tool in new_tools]

def test_gpt_reflection_with_llm(credentials_openai_mini: Credentials) -> None:
    gpt_assistant = GPTAssistantAgent(
        name="assistant", llm_config={"config_list": credentials_openai_mini.config_list, "assistant_id": None}
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        code_execution_config=False,
        is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )
    result = user_proxy.initiate_chat(gpt_assistant, message="Write a Joke!", summary_method="reflection_with_llm")
    assert result is not None

    # use the assistant configuration
    agent_using_assistant_config = GPTAssistantAgent(
        name="assistant",
        llm_config={"config_list": credentials_openai_mini.config_list},
        assistant_config={"assistant_id": gpt_assistant.assistant_id},
    )
    result = user_proxy.initiate_chat(
        agent_using_assistant_config, message="Write a Joke!", summary_method="reflection_with_llm"
    )
    assert result is not None

def test_assistant_tool_and_function_role_messages(credentials_openai_mini: Credentials) -> None:
    """Tests that internally generated roles ('tool', 'function') are correctly mapped to
    OpenAI Assistant API-compatible role ('assistant') before sending to the OpenAI API
    to prevent BadRequestError when using GPTAssistantAgent with other tool-calling agents.

    See PR: Fix role mapping in GPTAssistantAgent for OpenAI API compatibility #46
    """
    name = f"For_test_gpt_assistant_special_roles_{uuid.uuid4()}"
    assistant = GPTAssistantAgent(
        name,
        llm_config={
            "config_list": credentials_openai_mini.config_list,
        },
    )

    try:
        # Test cases for different message role combinations
        test_cases = [
            # Case 1: Tool messages
            [
                {
                    "role": "user",
                    "content": "Hello, can you help me?",
                },
                {
                    "role": "tool",
                    "content": "Tool execution result: Success",
                },
                {
                    "role": "assistant",
                    "content": "I received the tool result.",
                },
            ],
            # Case 2: Function messages
            [
                {
                    "role": "user",
                    "content": "What's the weather?",
                },
                {
                    "role": "function",
                    "content": '{"temperature": 72, "condition": "sunny"}',
                },
                {
                    "role": "assistant",
                    "content": "The weather is sunny and 72 degrees.",
                },
            ],
        ]

        # Test each case
        for messages in test_cases:
            success, response = assistant._invoke_assistant(messages)

            # Verify response
            assert success is True
            assert isinstance(response, dict)
            assert "content" in response
            assert "role" in response
            assert response["role"] == "assistant"

    finally:
        assistant.delete_assistant()


# --- test/beta/config/anthropic/test_anthropic_usage.py ---

async def test_process_response_normalizes_usage():
    client = AnthropicClient(api_key="test", prompt_caching=False)
    usage = _make_usage(input_tokens=100, output_tokens=25)
    response = _make_response(usage)

    result = await client._process_response(response, _make_context())

    assert isinstance(result, ModelResponse)
    assert result.usage == Usage(prompt_tokens=100, completion_tokens=25)

async def test_process_response_includes_cache_creation_tokens():
    client = AnthropicClient(api_key="test", prompt_caching=False)
    usage = _make_usage(input_tokens=3, output_tokens=10, cache_creation_input_tokens=5058)
    response = _make_response(usage)

    result = await client._process_response(response, _make_context())

    assert result.usage == Usage(
        prompt_tokens=3,
        completion_tokens=10,
        cache_creation_input_tokens=5058,
    )
    assert result.usage.cache_read_input_tokens is None

async def test_process_response_includes_cache_read_tokens():
    client = AnthropicClient(api_key="test", prompt_caching=False)
    usage = _make_usage(input_tokens=3, output_tokens=14, cache_read_input_tokens=5043)
    response = _make_response(usage)

    result = await client._process_response(response, _make_context())

    assert result.usage == Usage(
        prompt_tokens=3,
        completion_tokens=14,
        cache_read_input_tokens=5043,
    )
    assert result.usage.cache_creation_input_tokens is None

async def test_process_response_no_usage():
    client = AnthropicClient(api_key="test", prompt_caching=False)
    response = _make_response(usage=None)
    response.usage = None

    result = await client._process_response(response, _make_context())

    assert result.usage == Usage(prompt_tokens=0, completion_tokens=0)


# --- test/beta/config/openai/test_openai_responses_usage.py ---

    def test_normalizes_input_output_keys(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
        result = normalize_responses_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )

    def test_lifts_cached_tokens(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=80),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
        result = normalize_responses_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=0,
        )

    def test_lifts_reasoning_tokens(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=10),
        )
        result = normalize_responses_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=10,
        )


# --- test/beta/config/openai/test_openai_usage.py ---

    def test_lifts_cached_tokens(self):
        usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=80),
        )
        result = normalize_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=80,
        )

    def test_no_details_no_cache_key(self):
        usage = CompletionUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60)
        result = normalize_usage(usage)
        assert result.cache_read_input_tokens is None

    def test_details_with_zero_cached_tokens(self):
        usage = CompletionUsage(
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
        )
        result = normalize_usage(usage)
        assert result.cache_read_input_tokens == 0

    def test_none_details(self):
        usage = CompletionUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60, prompt_tokens_details=None)
        result = normalize_usage(usage)
        assert result.cache_read_input_tokens is None


# --- test/oai/test_anthropic.py ---

def test_anthropic_llm_config_entry():
    anthropic_llm_config = AnthropicLLMConfigEntry(
        model="claude-sonnet-4-5",
        api_key="dummy_api_key",
        stream=False,
        temperature=1.0,
        max_tokens=100,
    )
    expected = {
        "api_type": "anthropic",
        "model": "claude-sonnet-4-5",
        "api_key": "dummy_api_key",
        "stream": False,
        "temperature": 1.0,
        "max_tokens": 100,
        "tags": [],
    }
    actual = anthropic_llm_config.model_dump()
    assert actual == expected

    assert LLMConfig(anthropic_llm_config).model_dump() == {
        "config_list": [expected],
    }

def test_initialization(anthropic_client):
    assert anthropic_client.api_key == "dummy_api_key", "`api_key` should be correctly set in the config"

def test_initialization_with_aws_credentials(anthropic_client_with_aws_credentials):
    assert anthropic_client_with_aws_credentials.aws_access_key == "dummy_access_key", (
        "`aws_access_key` should be correctly set in the config"
    )
    assert anthropic_client_with_aws_credentials.aws_secret_key == "dummy_secret_key", (
        "`aws_secret_key` should be correctly set in the config"
    )
    assert anthropic_client_with_aws_credentials.aws_session_token == "dummy_session_token", (
        "`aws_session_token` should be correctly set in the config"
    )
    assert anthropic_client_with_aws_credentials.aws_region == "us-west-2", (
        "`aws_region` should be correctly set in the config"
    )

def test_initialization_with_vertexai_credentials(anthropic_client_with_vertexai_credentials):
    assert anthropic_client_with_vertexai_credentials.gcp_project_id == "dummy_project_id", (
        "`gcp_project_id` should be correctly set in the config"
    )
    assert anthropic_client_with_vertexai_credentials.gcp_region == "us-west-2", (
        "`gcp_region` should be correctly set in the config"
    )
    assert anthropic_client_with_vertexai_credentials.gcp_auth_token == "dummy_auth_token", (
        "`gcp_auth_token` should be correctly set in the config"
    )

def test_load_config(anthropic_client):
    params = {
        "model": "claude-sonnet-4-5",
        "stream": False,
        "temperature": 1,
        "top_p": 0.8,
        "max_tokens": 100,
    }
    expected_params = {
        "model": "claude-sonnet-4-5",
        "stream": False,
        "temperature": 1,
        "timeout": None,
        "top_p": 0.8,
        "max_tokens": 100,
        "stop_sequences": None,
        "top_k": None,
        "tool_choice": None,
    }
    result = anthropic_client.load_config(params)
    assert result == expected_params, "Config should be correctly loaded"

def test_extract_json_response(anthropic_client):
    # Define test Pydantic model
    class Step(BaseModel):
        explanation: str
        output: str

    class MathReasoning(BaseModel):
        steps: list[Step]
        final_answer: str

    # Set up the response format
    anthropic_client._response_format = MathReasoning

    # Test case 1: JSON within tags - CORRECT
    tagged_response = Message(
        id="msg_123",
        content=[
            TextBlock(
                text="""<json_response>
            {
                "steps": [
                    {"explanation": "Step 1", "output": "8x = -30"},
                    {"explanation": "Step 2", "output": "x = -3.75"}
                ],
                "final_answer": "x = -3.75"
            }
            </json_response>""",
                type="text",
            )
        ],
        model="claude-sonnet-4-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage={"input_tokens": 10, "output_tokens": 25},
    )

    result = anthropic_client._extract_json_response(tagged_response)
    assert isinstance(result, MathReasoning)
    assert len(result.steps) == 2
    assert result.final_answer == "x = -3.75"

    # Test case 2: Plain JSON without tags - SHOULD STILL PASS
    plain_response = Message(
        id="msg_123",
        content=[
            TextBlock(
                text="""Here's the solution:
            {
                "steps": [
                    {"explanation": "Step 1", "output": "8x = -30"},
                    {"explanation": "Step 2", "output": "x = -3.75"}
                ],
                "final_answer": "x = -3.75"
            }""",
                type="text",
            )
        ],
        model="claude-sonnet-4-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage={"input_tokens": 10, "output_tokens": 25},
    )

    result = anthropic_client._extract_json_response(plain_response)
    assert isinstance(result, MathReasoning)
    assert len(result.steps) == 2
    assert result.final_answer == "x = -3.75"

    # Test case 3: Invalid JSON - RAISE ERROR
    invalid_response = Message(
        id="msg_123",
        content=[
            TextBlock(
                text="""<json_response>
            {
                "steps": [
                    {"explanation": "Step 1", "output": "8x = -30"},
                    {"explanation": "Missing closing brace"
                ],
                "final_answer": "x = -3.75"
            """,
                type="text",
            )
        ],
        model="claude-sonnet-4-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage={"input_tokens": 10, "output_tokens": 25},
    )

    with pytest.raises(
        ValueError, match="Failed to parse response as valid JSON matching the schema for Structured Output: "
    ):
        anthropic_client._extract_json_response(invalid_response)

    # Test case 4: No JSON content - RAISE ERROR
    no_json_response = Message(
        id="msg_123",
        content=[TextBlock(text="This response contains no JSON at all.", type="text")],
        model="claude-sonnet-4-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage={"input_tokens": 10, "output_tokens": 25},
    )

    with pytest.raises(ValueError, match="No valid JSON found in response for Structured Output."):
        anthropic_client._extract_json_response(no_json_response)

    # Test case 5: Plain JSON without tags, using ThinkingBlock - SHOULD STILL PASS
    plain_response = Message(
        id="msg_123",
        content=[
            ThinkingBlock(
                signature="json_response",
                thinking="""Here's the solution:
            {
                "steps": [
                    {"explanation": "Step 1", "output": "8x = -30"},
                    {"explanation": "Step 2", "output": "x = -3.75"}
                ],
                "final_answer": "x = -3.75"
            }""",
                type="thinking",
            )
        ],
        model="claude-3-5-sonnet-latest",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage={"input_tokens": 10, "output_tokens": 25},
    )

    result = anthropic_client._extract_json_response(plain_response)
    assert isinstance(result, MathReasoning)
    assert len(result.steps) == 2
    assert result.final_answer == "x = -3.75"

def test_convert_tools_to_functions(anthropic_client):
    tools = [
        {
            "type": "function",
            "function": {
                "description": "weather tool",
                "name": "weather_tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city_name": {"type": "string", "description": "city_name"},
                        "city_list": {
                            "$defs": {
                                "city_list_class": {
                                    "properties": {
                                        "item1": {"title": "Item1", "type": "string"},
                                        "item2": {"title": "Item2", "type": "string"},
                                    },
                                    "required": ["item1", "item2"],
                                    "title": "city_list_class",
                                    "type": "object",
                                }
                            },
                            "items": {"$ref": "#/$defs/city_list_class"},
                            "type": "array",
                            "description": "city_list",
                        },
                    },
                    "required": ["city_name", "city_list"],
                },
            },
        }
    ]
    expected = [
        {
            "description": "weather tool",
            "name": "weather_tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {"type": "string", "description": "city_name"},
                    "city_list": {
                        "$defs": {
                            "city_list_class": {
                                "properties": {
                                    "item1": {"title": "Item1", "type": "string"},
                                    "item2": {"title": "Item2", "type": "string"},
                                },
                                "required": ["item1", "item2"],
                                "title": "city_list_class",
                                "type": "object",
                            }
                        },
                        "items": {"$ref": "#/properties/city_list/$defs/city_list_class"},
                        "type": "array",
                        "description": "city_list",
                    },
                },
                "required": ["city_name", "city_list"],
            },
        }
    ]
    actual = anthropic_client.convert_tools_to_functions(tools=tools)
    assert actual == expected

def test_process_image_content_valid_data_url():
    from autogen.oai.anthropic import process_image_content

    content_item = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    processed = process_image_content(content_item)
    expected = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}
    assert processed == expected

def test_process_image_content_non_image_type():
    from autogen.oai.anthropic import process_image_content

    content_item = {"type": "text", "text": "Just text"}
    processed = process_image_content(content_item)
    assert processed == content_item

def test_process_message_content_string():
    from autogen.oai.anthropic import process_message_content

    message = {"content": "Hello"}
    processed = process_message_content(message)
    assert processed == "Hello"

def test_process_message_content_list():
    from autogen.oai.anthropic import process_message_content

    message = {
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
    }
    processed = process_message_content(message)
    expected = [
        {"type": "text", "text": "Hello"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
    ]
    assert processed == expected

def test_oai_messages_to_anthropic_messages():
    from autogen.oai.anthropic import oai_messages_to_anthropic_messages

    params = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "System text."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBB"}},
                ],
            },
        ]
    }
    processed = oai_messages_to_anthropic_messages(params)

    # The function should update the system message (in the params dict) by concatenating only its text parts.
    assert params.get("system") == "System text."

    # The processed messages list should include a user message with the image URL converted to a base64 image format.
    user_message = next((m for m in processed if m["role"] == "user"), None)
    expected_content = [
        {"type": "text", "text": "Hello"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "BBB"}},
    ]
    assert user_message is not None
    assert user_message["content"] == expected_content

def test_supports_native_structured_outputs():
    """Test model detection for native structured outputs (Approach 1)."""
    from autogen.oai.anthropic import supports_native_structured_outputs

    # Sonnet 4.5 models should be supported
    assert supports_native_structured_outputs("claude-sonnet-4-5")
    assert supports_native_structured_outputs("claude-3-5-sonnet-20241022")
    assert supports_native_structured_outputs("claude-3-7-sonnet-20250219")

    # Pattern matching for future Sonnet versions
    assert supports_native_structured_outputs("claude-3-5-sonnet-20260101")
    assert supports_native_structured_outputs("claude-3-7-sonnet-20260615")

    # Future Opus 4.x models should be supported
    assert supports_native_structured_outputs("claude-opus-4-1")
    assert supports_native_structured_outputs("claude-opus-4-5")

    # Older models should NOT be supported
    assert not supports_native_structured_outputs("claude-3-haiku-20240307")
    assert not supports_native_structured_outputs("claude-3-sonnet-20240229")
    assert not supports_native_structured_outputs("claude-3-opus-20240229")
    assert not supports_native_structured_outputs("claude-2.1")
    assert not supports_native_structured_outputs("claude-instant-1.2")

    # Haiku 4.5 should be supported
    assert supports_native_structured_outputs("claude-haiku-4-5")
    assert supports_native_structured_outputs("claude-haiku-4-5-20251001")

    # Older Haiku models should not be supported
    assert not supports_native_structured_outputs("claude-3-5-haiku-20241022")

def test_has_messages_parse_api():
    """Test SDK version detection for messages.parse() API."""
    from autogen.oai.anthropic import has_messages_parse_api

    # Should detect if current SDK has messages.parse()
    has_parse = has_messages_parse_api()

    # If we have anthropic SDK, it should be a boolean
    assert isinstance(has_parse, bool)

    # If True, verify we can import the stable API
    if has_parse:
        try:
            from anthropic.resources.messages import Messages

            assert hasattr(Messages, "parse"), "Stable API should have parse method"
        except ImportError:
            pytest.fail("has_messages_parse_api returned True but cannot import stable API")

def test_transform_schema_for_anthropic():
    """Test schema transformation for Anthropic compatibility."""
    from autogen.oai.anthropic import transform_schema_for_anthropic

    # Test basic schema transformation
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 50},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "score": {"type": "number"},
        },
        "required": ["name", "age"],
    }

    transformed = transform_schema_for_anthropic(input_schema)

    # Should remove unsupported constraints
    assert "minLength" not in transformed["properties"]["name"]
    assert "maxLength" not in transformed["properties"]["name"]
    assert "minimum" not in transformed["properties"]["age"]
    assert "maximum" not in transformed["properties"]["age"]

    # Should add additionalProperties: false if not present
    assert transformed["additionalProperties"] is False

    # Should preserve required fields and types
    assert transformed["required"] == ["name", "age"]
    assert transformed["properties"]["name"]["type"] == "string"
    assert transformed["properties"]["age"]["type"] == "integer"

def test_transform_schema_preserves_nested_structures():
    """Test that schema transformation preserves nested structures."""
    from autogen.oai.anthropic import transform_schema_for_anthropic

    input_schema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "minimum": 0},
                },
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                    },
                },
            },
        },
        "additionalProperties": True,
    }

    transformed = transform_schema_for_anthropic(input_schema)

    # Should preserve nested structure
    assert "data" in transformed["properties"]
    assert "value" in transformed["properties"]["data"]["properties"]

    # Should preserve arrays
    assert transformed["properties"]["items"]["type"] == "array"

    # Should preserve existing additionalProperties setting
    assert transformed["additionalProperties"] is True

def test_pydantic_model_vs_dict_schema(anthropic_client):
    """Test handling of both Pydantic models and dict schemas."""

    class TestModel(BaseModel):
        name: str
        value: int

    # Test with Pydantic model
    anthropic_client._response_format = TestModel
    schema_from_model = TestModel.model_json_schema() if anthropic_client._response_format else {}

    assert "properties" in schema_from_model
    assert "name" in schema_from_model["properties"]
    assert "value" in schema_from_model["properties"]

    # Test with dict schema
    dict_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "integer"},
        },
        "required": ["name", "value"],
    }
    anthropic_client._response_format = dict_schema

    assert anthropic_client._response_format == dict_schema

def test_real_native_structured_output_api_call():
    """Real API call test for native structured output with Claude Sonnet 4.5."""
    import os

    from pydantic import BaseModel

    # Define structured output schema
    class Step(BaseModel):
        explanation: str
        output: str

    class MathReasoning(BaseModel):
        steps: list[Step]
        final_answer: str

    # Create client with response format
    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Test with Claude Sonnet 4.5 (supports native structured outputs)
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Solve the equation: 2x + 5 = 15. Show your work step by step."}],
        "max_tokens": 1024,
        "response_format": MathReasoning,
    }

    # Make actual API call
    response = client.create(params)

    # Verify response structure
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None

    # Verify it's valid JSON and matches schema
    result = MathReasoning.model_validate_json(response.choices[0].message.content)

    # Verify mathematical correctness
    assert len(result.steps) > 0, "Should have at least one step"
    assert result.final_answer, "Should have a final answer"

    # The answer should be x = 5
    assert "5" in result.final_answer or "x = 5" in result.final_answer.lower()

    # Verify each step has required fields
    for step in result.steps:
        assert step.explanation, "Each step should have an explanation"
        assert step.output, "Each step should have output"

def test_real_json_mode_fallback_api_call():
    """Real API call test for JSON Mode fallback with older Claude model."""
    import os

    from pydantic import BaseModel

    # Define structured output schema
    class Step(BaseModel):
        explanation: str
        output: str

    class MathReasoning(BaseModel):
        steps: list[Step]
        final_answer: str

    # Create client
    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Test with Claude Haiku (does NOT support native structured outputs, should fallback to JSON Mode)
    params = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "Solve: 3x - 4 = 11. Show your work step by step."}],
        "max_tokens": 1024,
        "response_format": MathReasoning,
    }

    # Make actual API call - should use JSON Mode fallback
    response = client.create(params)

    # Verify response structure
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None

    # Verify it's valid JSON and matches schema
    result = MathReasoning.model_validate_json(response.choices[0].message.content)

    # Verify mathematical correctness
    assert len(result.steps) > 0, "JSON Mode should still produce steps"
    assert result.final_answer, "JSON Mode should have final answer"

    # The answer should be x = 5
    assert "5" in result.final_answer or "x = 5" in result.final_answer.lower()

def test_real_native_vs_json_mode_comparison():
    """Compare native structured output vs JSON Mode with same prompt."""
    import os

    from pydantic import BaseModel

    class AnalysisResult(BaseModel):
        summary: str
        key_points: list[str]
        conclusion: str

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    test_message = (
        "Analyze the benefits of structured outputs in AI systems. Provide a summary, key points, and conclusion."
    )

    # Test 1: Native structured output (Claude Sonnet 4.5)
    params_native = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": test_message}],
        "max_tokens": 1024,
        "response_format": AnalysisResult,
    }

    response_native = client.create(params_native)
    result_native = AnalysisResult.model_validate_json(response_native.choices[0].message.content)

    # Test 2: JSON Mode fallback (Haiku)
    params_json = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": test_message}],
        "max_tokens": 1024,
        "response_format": AnalysisResult,
    }

    response_json = client.create(params_json)
    result_json = AnalysisResult.model_validate_json(response_json.choices[0].message.content)

    # Both should produce valid structured outputs
    assert result_native.summary and result_native.key_points and result_native.conclusion
    assert result_json.summary and result_json.key_points and result_json.conclusion

    # Both should have at least some key points
    assert len(result_native.key_points) > 0
    assert len(result_json.key_points) > 0

def test_openai_func_to_anthropic_preserves_strict(anthropic_client):
    """Test that strict field is preserved during tool conversion."""
    from autogen.oai.anthropic import AnthropicClient

    # Tool with strict=True
    strict_tool = {
        "name": "calculate",
        "description": "Perform calculation",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "subtract"]},
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    }

    result = AnthropicClient.openai_func_to_anthropic(strict_tool)

    # Verify strict field is preserved
    assert "strict" in result
    assert result["strict"] is True

    # Verify input_schema conversion
    assert "input_schema" in result
    assert "parameters" not in result

    # Verify schema transformation was applied for strict tools
    # Should add additionalProperties: false (required by Anthropic for strict tools)
    assert result["input_schema"]["additionalProperties"] is False

    # Verify properties are still there
    assert "properties" in result["input_schema"]
    assert "operation" in result["input_schema"]["properties"]
    assert "a" in result["input_schema"]["properties"]
    assert "b" in result["input_schema"]["properties"]

    # Tool without strict field
    legacy_tool = {
        "name": "search",
        "description": "Search function",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }

    result_legacy = AnthropicClient.openai_func_to_anthropic(legacy_tool)

    # Verify strict field is not added if not present
    assert "strict" not in result_legacy

    # Legacy tools should not have schema transformation applied
    # (additionalProperties might not be set)
    assert result_legacy["input_schema"]["properties"]["query"]["type"] == "string"

def test_real_strict_tool_use_api_call():
    """Real API call test for strict tool use with type enforcement."""
    import json
    import os

    # Create client
    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Define strict tool with enum for operation
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Calculate 15 + 7 using the calculator tool"}],
        "max_tokens": 1024,
        "functions": [
            {
                "name": "calculate",
                "description": "Perform arithmetic calculation",
                "strict": True,  # Enable strict type validation
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The operation to perform",
                        },
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            }
        ],
    }

    # Make actual API call
    response = client.create(params)

    # Verify response structure
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0

    # Verify tool call was made
    message = response.choices[0].message
    assert message.tool_calls is not None, "Should have tool calls"
    assert len(message.tool_calls) > 0, "Should have at least one tool call"

    # Verify tool call structure
    tool_call = message.tool_calls[0]
    assert tool_call.function.name == "calculate"

    # Parse and verify arguments
    args = json.loads(tool_call.function.arguments)

    # With strict=True, these should be guaranteed to be correct types
    assert isinstance(args["a"], (int, float)), "Argument 'a' should be a number"
    assert isinstance(args["b"], (int, float)), "Argument 'b' should be a number"
    assert args["operation"] in ["add", "subtract", "multiply", "divide"], "Operation should be valid enum value"

    # Verify the calculation is correct
    assert args["operation"] == "add"
    assert args["a"] == 15
    assert args["b"] == 7

def test_real_strict_tool_type_enforcement():
    """Real API call test verifying strict mode enforces correct types."""
    import json
    import os

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Tool with multiple type constraints
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Book a flight for 2 passengers to New York, economy cabin"}],
        "max_tokens": 1024,
        "functions": [
            {
                "name": "book_flight",
                "description": "Book a flight",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "passengers": {"type": "integer", "description": "Number of passengers"},
                        "destination": {"type": "string", "description": "Destination city"},
                        "cabin_class": {
                            "type": "string",
                            "enum": ["economy", "business", "first"],
                            "description": "Cabin class",
                        },
                    },
                    "required": ["passengers", "destination", "cabin_class"],
                },
            }
        ],
    }

    response = client.create(params)

    # Verify tool call
    assert response.choices[0].message.tool_calls is not None

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    # Strict mode guarantees these types
    assert isinstance(args["passengers"], int), "passengers should be integer, not string '2'"
    assert args["passengers"] == 2

    assert isinstance(args["destination"], str)
    assert args["destination"].lower() == "new york"

    assert args["cabin_class"] in ["economy", "business", "first"], "cabin_class must match enum"

def test_real_combined_strict_tools_and_structured_output():
    """Real API call test combining strict tools with structured output."""
    import json
    import os

    from pydantic import BaseModel

    # Result schema
    class CalculationResult(BaseModel):
        problem: str
        steps: list[str]
        result: float
        verification: str

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Use both strict tools and structured output
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Calculate (10 + 5) * 2 and explain your work"}],
        "max_tokens": 1024,
        "response_format": CalculationResult,
        "functions": [
            {
                "name": "calculate",
                "description": "Perform calculation",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            }
        ],
    }

    response = client.create(params)

    # When both strict tools and structured output are configured with beta.messages.create,
    # Claude chooses which feature to use based on the prompt:
    # - Either makes tool calls (BetaToolUseBlock), OR
    # - Provides structured output (BetaTextBlock)
    # Both are processed via beta API with the structured-outputs-2025-11-13 header
    message = response.choices[0].message

    # Verify at least one content type is present
    has_tool_calls = message.tool_calls is not None and len(message.tool_calls) > 0
    has_structured_output = message.content and message.content.strip()

    assert has_tool_calls or has_structured_output, "Should have either tool calls OR structured output"

    # If tool calls are present, verify strict typing
    if has_tool_calls:
        tool_call = message.tool_calls[0]
        assert tool_call.function.name == "calculate", "Tool call should be for calculate function"
        args = json.loads(tool_call.function.arguments)
        assert isinstance(args["a"], (int, float)), "Argument 'a' should be a number"
        assert isinstance(args["b"], (int, float)), "Argument 'b' should be a number"
        assert args["operation"] in [
            "add",
            "subtract",
            "multiply",
            "divide",
        ], "Operation should be valid enum value"

    # If structured output is present, verify schema compliance
    if has_structured_output:
        result = CalculationResult.model_validate_json(message.content)
        assert result.problem, "Should have problem description"
        assert len(result.steps) > 0, "Should have calculation steps"
        assert isinstance(result.result, (int, float)), "Result should be a number"
        assert result.verification, "Should have verification"

def test_real_sdk_version_validation_on_strict_tools():
    """Test that SDK version is validated when using strict tools."""
    import os

    # This test verifies that the version check happens
    # If SDK is too old, it should raise ImportError

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 100,
        "functions": [
            {
                "name": "test_tool",
                "strict": True,
                "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
            }
        ],
    }

    # This should work if SDK >= 0.74.1, otherwise raise ImportError
    # We can't easily test the failure case without downgrading the SDK
    # So we just verify it doesn't raise with a compatible SDK
    try:
        response = client.create(params)
        # If we get here, SDK version is compatible
        assert response is not None
    except ImportError as e:
        # If SDK is too old, should get clear error message
        assert "anthropic>=0.74.1" in str(e)
        assert "Please upgrade" in str(e)

def test_real_extended_thinking_api_call():
    """Real API call test for extended thinking feature with ThinkingBlock."""
    import os

    # Create client
    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Test with a complex reasoning problem that benefits from extended thinking
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [
            {
                "role": "user",
                "content": """A farmer has 17 sheep. All but 9 die. How many sheep are left alive?
Think through this step by step, being careful about the wording.""",
            }
        ],
        "max_tokens": 8000,  # Must be greater than thinking.budget_tokens
        "thinking": {
            "type": "enabled",
            "budget_tokens": 3000,  # Budget for internal reasoning
        },
    }

    # Make API call with extended thinking enabled
    response = client.create(params)

    # Verify response structure
    assert response is not None
    assert response.choices is not None
    assert len(response.choices) > 0

    # Get message content
    message = response.choices[0].message
    assert message.content is not None

    content = message.content
    logger.info("\n=== Extended Thinking Response ===")
    logger.info(content)
    logger.info("=== End Response ===\n")

    # Verify both thinking and text content are present
    # The response should contain "[Thinking]" prefix when ThinkingBlock is present
    assert isinstance(content, str)
    assert len(content) > 0

    # Check if thinking was included (indicated by [Thinking] prefix)
    has_thinking = "[Thinking]" in content

    # Verify the answer is correct (9 sheep are left alive)
    assert "9" in content

    # If thinking was included, verify it's properly formatted
    if has_thinking:
        # Should have [Thinking] prefix followed by thinking content, then regular response
        assert content.startswith("[Thinking]")
        # Should have multiple parts (thinking + text)
        parts = content.split("\n\n", 1)
        assert len(parts) >= 1

    # Verify cost tracking includes thinking tokens if present
    assert response.cost is not None
    assert response.cost >= 0

    # Verify token usage
    assert response.usage is not None
    assert response.usage.total_tokens > 0
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0

def test_real_tools_with_structured_output_beta_api(credentials_anthropic_claude_sonnet, caplog):
    """Real API call test for tools + structured outputs using GA API.

    This test verifies that OpenAI tool format works with Anthropic's structured
    outputs API. Previously, the OpenAI wrapper format {"type": "function", ...}
    was rejected by the API with a 400 error.

    The key test is that combining tools (in OpenAI format) with response_format
    doesn't cause a 400 error, proving the tool format conversion works correctly.
    """
    import json
    import logging

    from pydantic import BaseModel

    # Capture logs to verify beta API usage
    caplog.set_level(logging.WARNING)

    # Define structured output schema
    class MathResult(BaseModel):
        steps: list[str]
        answer: int

    # Get API key from credentials
    api_key = credentials_anthropic_claude_sonnet.config_list[0]["api_key"]

    # Create client
    client = AnthropicClient(api_key=api_key)

    # Define tool in OpenAI format with "type": "function" wrapper
    # This is the format that was causing the 400 error before the fix
    tools = [
        {
            "type": "function",  # OpenAI wrapper format
            "function": {
                "name": "calculator",
                "description": "Perform basic math operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The operation to perform",
                        },
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["operation", "a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    # Test 1: Verify tools with structured output don't cause 400 error
    # This was the original bug - combining tools + response_format would fail
    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Use the calculator tool to compute 23 + 19"}],
        "max_tokens": 1024,
        "response_format": MathResult,  # Triggers beta API
        "tools": tools,  # OpenAI format with "type": "function"
    }

    # Make actual API call - this should succeed with the fix (no 400 error)
    response = client.create(params)

    # Verify response structure
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0

    message = response.choices[0].message

    # Verify tool call was made (Claude should use the tool)
    assert message.tool_calls is not None, "Should have tool calls"
    assert len(message.tool_calls) > 0, "Should have at least one tool call"

    # Verify tool call structure
    tool_call = message.tool_calls[0]
    assert hasattr(tool_call, "function"), "Tool call should have function attribute"
    assert tool_call.function.name == "calculator"

    # Parse and verify arguments - strict validation should work
    args = json.loads(tool_call.function.arguments)
    assert args["operation"] == "add"
    assert args["a"] == 23
    assert args["b"] == 19

    # Verify cost tracking
    assert response.cost is not None
    assert response.cost >= 0

    # VERIFY BETA API WAS ACTUALLY USED (not fallback to JSON mode)
    # Method 1: Check that no fallback warning was logged
    fallback_warnings = [
        record
        for record in caplog.records
        if "Falling back to JSON Mode" in record.message and record.levelname == "WARNING"
    ]
    assert len(fallback_warnings) == 0, (
        f"Beta API should not fall back to JSON mode. Found warnings: {[r.message for r in fallback_warnings]}"
    )

    # Method 2: Verify response characteristics indicate beta API usage
    # Beta API responses should have proper structured content
    assert response.choices[0].message.content is not None or response.choices[0].message.tool_calls is not None, (
        "Beta API should return either content or tool calls"
    )

    # Test 2: Verify structured output works after tool execution
    # Send tool result and request structured output using OpenAI format
    tool_result_params = {
        "model": "claude-sonnet-4-5",
        "messages": [
            {"role": "user", "content": "Calculate 15 + 7 and show your work"},
            {
                "role": "assistant",
                "content": "",  # Empty content when tool calls are present
                "tool_calls": [
                    {
                        "id": "call_test_123",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"operation": "add", "a": 15, "b": 7}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_test_123", "content": "22"},
        ],
        "max_tokens": 1024,
        "response_format": MathResult,
    }

    # Get response with structured output
    final_response = client.create(tool_result_params)

    # Verify structured output
    assert final_response.choices[0].message.content is not None
    content = final_response.choices[0].message.content

    # Parse and validate structured output
    result = MathResult.model_validate_json(content) if isinstance(content, str) else MathResult.model_validate(content)

    # Verify structured output has required fields
    assert result.steps, "Should have steps"
    assert result.answer == 22, "Should have correct answer"

def test_load_config_stream_enabled(anthropic_client):
    """Verify that stream=True flows through load_config without being forced to False."""
    params = {
        "model": "claude-sonnet-4-5",
        "stream": True,
        "temperature": 1,
        "max_tokens": 100,
    }
    result = anthropic_client.load_config(params)
    assert result["stream"] is True, "stream=True should be preserved in config"

def test_real_streaming_text():
    """Real API call test for basic streaming text response."""
    import os

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Say 'Hello streaming!' and nothing else."}],
        "max_tokens": 100,
        "stream": True,
    }

    response = client.create(params)

    # Verify response structure
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0

    content = response.choices[0].message.content
    assert content is not None
    assert len(content) > 0
    assert "hello" in content.lower() or "streaming" in content.lower()

    # Verify usage is tracked
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens > 0

    # Verify cost is calculated
    assert response.cost is not None
    assert response.cost >= 0

def test_real_streaming_with_tools():
    """Real API call test for streaming with tool calls."""
    import json
    import os

    client = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    params = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Calculate 10 + 5 using the calculator tool."}],
        "max_tokens": 1024,
        "stream": True,
        "functions": [
            {
                "name": "calculator",
                "description": "Perform basic arithmetic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            }
        ],
    }

    response = client.create(params)

    # Verify response
    assert response is not None
    message = response.choices[0].message

    # Should have tool calls
    assert message.tool_calls is not None, "Streaming should reconstruct tool calls"
    assert len(message.tool_calls) > 0

    tool_call = message.tool_calls[0]
    assert tool_call.function.name == "calculator"

    args = json.loads(tool_call.function.arguments)
    assert args["operation"] == "add"
    assert args["a"] == 10
    assert args["b"] == 5

    # Verify finish reason
    assert response.choices[0].finish_reason == "tool_calls"


# --- test/oai/test_client.py ---

def test_aoai_chat_completion(credentials_azure_gpt_4_1_mini: Credentials):
    """Updated to use gpt-4.1-mini (official replacement for gpt-35-turbo)"""
    config_list = credentials_azure_gpt_4_1_mini.config_list
    client = OpenAIWrapper(config_list=config_list)
    response = client.create(messages=[{"role": "user", "content": "2+2="}], cache_seed=None)
    print(response)
    print(client.extract_text_or_completion_object(response))

    # test dialect
    config = config_list[0]
    config["azure_deployment"] = config["model"]
    config["azure_endpoint"] = config.pop("base_url")
    client = OpenAIWrapper(**config)
    response = client.create(messages=[{"role": "user", "content": "2+2="}], cache_seed=None)
    print(response)
    print(client.extract_text_or_completion_object(response))

def test_fallback_kwargs():
    assert set(inspect.getfullargspec(OpenAI.__init__).kwonlyargs) == OPENAI_FALLBACK_KWARGS
    assert set(inspect.getfullargspec(AzureOpenAI.__init__).kwonlyargs) == AOPENAI_FALLBACK_KWARGS

def test_oai_tool_calling_extraction(credentials_openai_mini: Credentials):
    client = OpenAIWrapper(config_list=credentials_openai_mini.config_list)
    response = client.create(
        messages=[
            {
                "role": "user",
                "content": "What is the weather in San Francisco?",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "getCurrentWeather",
                    "description": "Get the weather in location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
                            "unit": {"type": "string", "enum": ["c", "f"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    )
    print(response)
    print(client.extract_text_or_completion_object(response))

def test_chat_completion(credentials_openai_mini: Credentials):
    client = OpenAIWrapper(config_list=credentials_openai_mini.config_list)
    response = client.create(messages=[{"role": "user", "content": "1+1="}])
    print(response)
    print(client.extract_text_or_completion_object(response))

def test_completion(credentials_azure_gpt_4_1_mini: Credentials):
    """Updated to use gpt-4.1-mini (gpt-3.5-turbo-instruct retired Nov 11, 2025)"""
    client = OpenAIWrapper(config_list=credentials_azure_gpt_4_1_mini.config_list)
    response = client.create(messages=[{"role": "user", "content": "1+1="}])
    print(response)
    print(client.extract_text_or_completion_object(response))

def test_cost(credentials_azure_gpt_4_1_mini: Credentials, cache_seed):
    """Updated to use gpt-4.1-mini (gpt-35-turbo-instruct retired Nov 11, 2025)"""
    client = OpenAIWrapper(config_list=credentials_azure_gpt_4_1_mini.config_list, cache_seed=cache_seed)
    response = client.create(messages=[{"role": "user", "content": "1+3="}])
    print(response.cost)

def test_customized_cost(credentials_azure_gpt_4_1_mini: Credentials):
    """Updated to use gpt-4.1-mini (gpt-35-turbo-instruct retired Nov 11, 2025)"""
    config_list = credentials_azure_gpt_4_1_mini.config_list
    for config in config_list:
        config.update({"price": [1000, 1000]})
    client = OpenAIWrapper(config_list=config_list, cache_seed=None)
    response = client.create(messages=[{"role": "user", "content": "1+3="}])
    assert response.cost >= 4, (
        f"Due to customized pricing, cost should be > 4. Message: {response.choices[0].message.content}"
    )

def test_usage_summary(credentials_azure_gpt_4_1_mini: Credentials):
    """Updated to use gpt-4.1-mini (gpt-35-turbo-instruct retired Nov 11, 2025)"""
    client = OpenAIWrapper(config_list=credentials_azure_gpt_4_1_mini.config_list)
    client.create(messages=[{"role": "user", "content": "1+3="}], cache_seed=None)

    # usage should be recorded
    assert client.actual_usage_summary["total_cost"] > 0, "total_cost should be greater than 0"
    assert client.total_usage_summary["total_cost"] > 0, "total_cost should be greater than 0"

    # check print
    client.print_usage_summary()

    # check clear
    client.clear_usage_summary()
    assert client.actual_usage_summary is None, "actual_usage_summary should be None"
    assert client.total_usage_summary is None, "total_usage_summary should be None"

def test_legacy_cache(credentials_openai_mini: Credentials):
    # Prompt to use for testing.
    prompt = "Write a 100 word summary on the topic of the history of human civilization."

    # Clear cache.
    if os.path.exists(LEGACY_CACHE_DIR):
        shutil.rmtree(LEGACY_CACHE_DIR)

    # Test default cache seed.
    client = OpenAIWrapper(config_list=credentials_openai_mini.config_list, cache_seed=LEGACY_DEFAULT_CACHE_SEED)
    start_time = time.time()
    cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_cold_cache = end_time - start_time

    start_time = time.time()
    warm_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_warm_cache = end_time - start_time
    assert cold_cache_response == warm_cache_response
    assert duration_with_warm_cache < duration_with_cold_cache
    assert os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(LEGACY_DEFAULT_CACHE_SEED)))

    # Test with cache seed set through constructor
    client = OpenAIWrapper(config_list=credentials_openai_mini.config_list, cache_seed=13)
    start_time = time.time()
    cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_cold_cache = end_time - start_time

    start_time = time.time()
    warm_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_warm_cache = end_time - start_time
    assert cold_cache_response == warm_cache_response
    assert duration_with_warm_cache < duration_with_cold_cache
    assert os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(13)))

    # Test with cache seed set through create method
    client = OpenAIWrapper(config_list=credentials_openai_mini.config_list)
    start_time = time.time()
    cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}], cache_seed=17)
    end_time = time.time()
    duration_with_cold_cache = end_time - start_time

    start_time = time.time()
    warm_cache_response = client.create(messages=[{"role": "user", "content": prompt}], cache_seed=17)
    end_time = time.time()
    duration_with_warm_cache = end_time - start_time
    assert cold_cache_response == warm_cache_response
    assert duration_with_warm_cache < duration_with_cold_cache
    assert os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(17)))

    # Test using a different cache seed through create method.
    start_time = time.time()
    cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}], cache_seed=21)
    end_time = time.time()
    duration_with_cold_cache = end_time - start_time
    assert duration_with_warm_cache < duration_with_cold_cache
    assert os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(21)))

def test_no_default_cache(credentials_openai_mini: Credentials):
    # Prompt to use for testing.
    prompt = "Write a 100 word summary on the topic of the history of human civilization."

    # Clear cache.
    if os.path.exists(LEGACY_CACHE_DIR):
        shutil.rmtree(LEGACY_CACHE_DIR)

    # Test default cache which is no cache
    client = OpenAIWrapper(config_list=credentials_openai_mini.config_list)
    start_time = time.time()
    no_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_no_cache = end_time - start_time

    # Legacy cache should not be used.
    assert not os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(LEGACY_DEFAULT_CACHE_SEED)))

    # Create cold cache
    client = OpenAIWrapper(config_list=credentials_openai_mini.config_list, cache_seed=LEGACY_DEFAULT_CACHE_SEED)
    start_time = time.time()
    cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_cold_cache = end_time - start_time

    # Create warm cache
    start_time = time.time()
    warm_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    duration_with_warm_cache = end_time - start_time

    # Test that warm cache is the same as cold cache.
    assert cold_cache_response == warm_cache_response
    assert no_cache_response != warm_cache_response

    # Test that warm cache is faster than cold cache and no cache.
    assert duration_with_warm_cache < duration_with_cold_cache
    assert duration_with_warm_cache < duration_with_no_cache

    # Test legacy cache is used.
    assert os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(LEGACY_DEFAULT_CACHE_SEED)))

def test_cache(credentials_openai_mini: Credentials):
    # Prompt to use for testing.
    prompt = "Write a 100 word summary on the topic of the history of artificial intelligence."

    # Clear cache.
    if os.path.exists(LEGACY_CACHE_DIR):
        shutil.rmtree(LEGACY_CACHE_DIR)
    cache_dir = ".cache_test"
    assert cache_dir != LEGACY_CACHE_DIR
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    # Test cache set through constructor.
    with Cache.disk(cache_seed=49, cache_path_root=cache_dir) as cache:
        client = OpenAIWrapper(config_list=credentials_openai_mini.config_list, cache=cache)
        start_time = time.time()
        cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
        end_time = time.time()
        duration_with_cold_cache = end_time - start_time

        start_time = time.time()
        warm_cache_response = client.create(messages=[{"role": "user", "content": prompt}])
        end_time = time.time()
        duration_with_warm_cache = end_time - start_time
        assert cold_cache_response == warm_cache_response
        assert duration_with_warm_cache < duration_with_cold_cache
        assert os.path.exists(os.path.join(cache_dir, str(49)))
        # Test legacy cache is not used.
        assert not os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(49)))
        assert not os.path.exists(os.path.join(cache_dir, str(LEGACY_DEFAULT_CACHE_SEED)))

    # Test cache set through method.
    client = OpenAIWrapper(config_list=credentials_openai_mini.config_list)
    with Cache.disk(cache_seed=312, cache_path_root=cache_dir) as cache:
        start_time = time.time()
        cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}], cache=cache)
        end_time = time.time()
        duration_with_cold_cache = end_time - start_time

        start_time = time.time()
        warm_cache_response = client.create(messages=[{"role": "user", "content": prompt}], cache=cache)
        end_time = time.time()
        duration_with_warm_cache = end_time - start_time
        assert cold_cache_response == warm_cache_response
        assert duration_with_warm_cache < duration_with_cold_cache
        assert os.path.exists(os.path.join(cache_dir, str(312)))
        # Test legacy cache is not used.
        assert not os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(312)))
        assert not os.path.exists(os.path.join(cache_dir, str(LEGACY_DEFAULT_CACHE_SEED)))

    # Test different cache seed.
    with Cache.disk(cache_seed=123, cache_path_root=cache_dir) as cache:
        start_time = time.time()
        cold_cache_response = client.create(messages=[{"role": "user", "content": prompt}], cache=cache)
        end_time = time.time()
        duration_with_cold_cache = end_time - start_time
        assert duration_with_warm_cache < duration_with_cold_cache
        # Test legacy cache is not used.
        assert not os.path.exists(os.path.join(LEGACY_CACHE_DIR, str(123)))
        assert not os.path.exists(os.path.join(cache_dir, str(LEGACY_DEFAULT_CACHE_SEED)))

def test_convert_system_role_to_user() -> None:
    messages = [
        {"content": "Your name is Jack and you are a comedian in a two-person comedy show.", "role": "system"},
        {"content": "Jack, tell me a joke.", "role": "user", "name": "user"},
    ]
    OpenAIClient._convert_system_role_to_user(messages)
    expected = [
        {"content": "Your name is Jack and you are a comedian in a two-person comedy show.", "role": "user"},
        {"content": "Jack, tell me a joke.", "role": "user", "name": "user"},
    ]
    assert messages == expected

def test_extra_headers_chat_completion(credentials_openai_mini: Credentials):
    """Test that extra_headers flows through to the API without error."""
    config_list = [
        {**config, "extra_headers": {"X-Custom-Test": "ag2-extra-headers"}}
        for config in credentials_openai_mini.config_list
    ]
    client = OpenAIWrapper(config_list=config_list)
    response = client.create(messages=[{"role": "user", "content": "1+1="}], cache_seed=None)
    print(response)
    print(client.extract_text_or_completion_object(response))

    def test_completion_o1_mini(self, o1_mini_client: OpenAIWrapper, messages: list[dict[str, str]]) -> None:
        self._test_completion(o1_mini_client, messages)

    def test_completion_o1(self, o1_client: OpenAIWrapper, messages: list[dict[str, str]]) -> None:
        self._test_completion(o1_client, messages)


# --- test/oai/test_client_stream.py ---

def test_completion_stream(credentials_azure_gpt_4_1_mini: Credentials) -> None:
    """Updated to use gpt-4.1-mini (gpt-35-turbo-instruct retired Nov 11, 2025)"""
    client = OpenAIWrapper(config_list=credentials_azure_gpt_4_1_mini.config_list)
    response = client.create(messages=[{"role": "user", "content": "1+1="}], stream=True)
    print(response)
    print(client.extract_text_or_completion_object(response))

def test_chat_completion_stream(credentials_openai_mini: Credentials) -> None:
    client = OpenAIWrapper(config_list=credentials_openai_mini.config_list)
    response = client.create(messages=[{"role": "user", "content": "1+1="}], stream=True)
    print(response)
    print(client.extract_text_or_completion_object(response))

def test__update_function_call_from_chunk() -> None:
    function_call_chunks = [
        ChoiceDeltaFunctionCall(arguments=None, name="get_current_weather"),
        ChoiceDeltaFunctionCall(arguments='{"', name=None),
        ChoiceDeltaFunctionCall(arguments="location", name=None),
        ChoiceDeltaFunctionCall(arguments='":"', name=None),
        ChoiceDeltaFunctionCall(arguments="San", name=None),
        ChoiceDeltaFunctionCall(arguments=" Francisco", name=None),
        ChoiceDeltaFunctionCall(arguments='"}', name=None),
    ]
    expected = {"name": "get_current_weather", "arguments": '{"location":"San Francisco"}'}

    full_function_call = None
    completion_tokens = 0
    for function_call_chunk in function_call_chunks:
        # print(f"{function_call_chunk=}")
        full_function_call, completion_tokens = OpenAIWrapper._update_function_call_from_chunk(
            function_call_chunk=function_call_chunk,
            full_function_call=full_function_call,
            completion_tokens=completion_tokens,
        )

    print(f"{full_function_call=}")
    print(f"{completion_tokens=}")

    assert full_function_call == expected
    assert completion_tokens == len(function_call_chunks)

    ChatCompletionMessage(role="assistant", function_call=full_function_call, content=None)

def test__update_tool_calls_from_chunk() -> None:
    tool_calls_chunks = [
        ChoiceDeltaToolCall(
            index=0,
            id="call_D2HOWGMekmkxXu9Ix3DUqJRv",
            function=ChoiceDeltaToolCallFunction(arguments="", name="get_current_weather"),
            type="function",
        ),
        ChoiceDeltaToolCall(
            index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='{"lo', name=None), type=None
        ),
        ChoiceDeltaToolCall(
            index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments="catio", name=None), type=None
        ),
        ChoiceDeltaToolCall(
            index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='n": "S', name=None), type=None
        ),
        ChoiceDeltaToolCall(
            index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments="an F", name=None), type=None
        ),
        ChoiceDeltaToolCall(
            index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments="ranci", name=None), type=None
        ),
        ChoiceDeltaToolCall(
            index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments="sco, C", name=None), type=None
        ),
        ChoiceDeltaToolCall(
            index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='A"}', name=None), type=None
        ),
        ChoiceDeltaToolCall(
            index=1,
            id="call_22HgJep4nowKU3UOr96xaLmd",
            function=ChoiceDeltaToolCallFunction(arguments="", name="get_current_weather"),
            type="function",
        ),
        ChoiceDeltaToolCall(
            index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments='{"lo', name=None), type=None
        ),
        ChoiceDeltaToolCall(
            index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments="catio", name=None), type=None
        ),
        ChoiceDeltaToolCall(
            index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments='n": "N', name=None), type=None
        ),
        ChoiceDeltaToolCall(
            index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments="ew Y", name=None), type=None
        ),
        ChoiceDeltaToolCall(
            index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments="ork, ", name=None), type=None
        ),
        ChoiceDeltaToolCall(
            index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments='NY"}', name=None), type=None
        ),
    ]

    full_tool_calls: list[dict[str, Any] | None] = [None, None]
    completion_tokens = 0
    for tool_calls_chunk in tool_calls_chunks:
        index = tool_calls_chunk.index
        full_tool_calls[index], completion_tokens = OpenAIWrapper._update_tool_calls_from_chunk(
            tool_calls_chunk=tool_calls_chunk,
            full_tool_call=full_tool_calls[index],
            completion_tokens=completion_tokens,
        )

    print(f"{full_tool_calls=}")
    print(f"{completion_tokens=}")

    ChatCompletionMessage(role="assistant", tool_calls=full_tool_calls, content=None)

def test__update_tool_calls_from_chunk_repeated_type() -> None:
    """Regression test for gh-2058: some providers send type='function' in
    every chunk, which caused it to be concatenated into
    'functionfunction...' instead of staying 'function'."""
    tool_calls_chunks = [
        ChoiceDeltaToolCall(
            index=0,
            id="call_abc123",
            function=ChoiceDeltaToolCallFunction(arguments="", name="my_tool"),
            type="function",
        ),
        ChoiceDeltaToolCall(
            index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='{"x"', name=None), type="function"
        ),
        ChoiceDeltaToolCall(
            index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments=": 1}", name=None), type="function"
        ),
    ]

    full_tool_call = None
    completion_tokens = 0
    for chunk in tool_calls_chunks:
        full_tool_call, completion_tokens = OpenAIWrapper._update_tool_calls_from_chunk(
            tool_calls_chunk=chunk,
            full_tool_call=full_tool_call,
            completion_tokens=completion_tokens,
        )

    assert full_tool_call["type"] == "function", f"type should stay 'function' but got '{full_tool_call['type']}'"
    assert full_tool_call["function"]["name"] == "my_tool"
    assert full_tool_call["function"]["arguments"] == '{"x": 1}'

def test_chat_functions_stream(credentials_openai_mini: Credentials) -> None:
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": ["location"],
            },
        },
    ]
    client = OpenAIWrapper(config_list=credentials_openai_mini.config_list)
    response = client.create(
        messages=[{"role": "user", "content": "What's the weather like today in San Francisco?"}],
        functions=functions,
        stream=True,
    )
    print(response)
    print(client.extract_text_or_completion_object(response))

def test_chat_tools_stream(credentials_openai_mini: Credentials) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
    ]
    client = OpenAIWrapper(config_list=credentials_openai_mini.config_list)
    response = client.create(
        messages=[{"role": "user", "content": "What's the weather like today in San Francisco?"}],
        tools=tools,
        stream=True,
    )
    # check response
    choices = response.choices
    assert isinstance(choices, list)
    assert len(choices) > 0

    choice = choices[0]
    assert choice.finish_reason == "tool_calls"

    message = choice.message
    tool_calls = message.tool_calls
    assert isinstance(tool_calls, list)
    assert len(tool_calls) > 0


# --- test/oai/test_oai_models.py ---

    def test_chat_completion_schema(self) -> None:
        local_schema = strip_descriptions(ChatCompletionLocal.model_json_schema())
        openai_schema = strip_descriptions(ChatCompletion.model_json_schema())
        assert local_schema == openai_schema

    def test_chat_completion_message_schema(self) -> None:
        local_schema = strip_descriptions(ChatCompletionMessageLocal.model_json_schema())
        openai_schema = strip_descriptions(ChatCompletionMessage.model_json_schema())
        assert local_schema == openai_schema

    def test_chat_completion_message_tool_call_schema(self) -> None:
        local_schema = strip_descriptions(ChatCompletionMessageFunctionToolCallLocal.model_json_schema())
        openai_schema = strip_descriptions(ChatCompletionMessageFunctionToolCall.model_json_schema())
        assert local_schema == openai_schema

    def test_completion_usage_schema(self) -> None:
        local_schema = strip_descriptions(CompletionUsageLocal.model_json_schema())
        openai_schema = strip_descriptions(CompletionUsage.model_json_schema())
        assert local_schema == openai_schema


# --- test/oai/test_openai_streaming_structured_output.py ---

    def test_adds_stream_options_when_streaming(self) -> None:
        """Verify stream_options with include_usage is added when stream=True."""
        params: dict[str, Any] = {"stream": True, "messages": []}
        OpenAIClient._add_streaming_usage_to_params(params)

        assert "stream_options" in params
        assert params["stream_options"]["include_usage"] is True

    def test_does_not_modify_when_not_streaming(self) -> None:
        """Verify params unchanged when stream=False."""
        params: dict[str, Any] = {"stream": False, "messages": []}
        OpenAIClient._add_streaming_usage_to_params(params)

        assert "stream_options" not in params

    def test_does_not_modify_when_stream_not_present(self) -> None:
        """Verify params unchanged when stream key is absent."""
        params: dict[str, Any] = {"messages": []}
        OpenAIClient._add_streaming_usage_to_params(params)

        assert "stream_options" not in params

    def test_preserves_existing_stream_options(self) -> None:
        """Verify existing stream_options are preserved."""
        params: dict[str, Any] = {
            "stream": True,
            "stream_options": {"some_other_option": "value"},
            "messages": [],
        }
        OpenAIClient._add_streaming_usage_to_params(params)

        assert params["stream_options"]["some_other_option"] == "value"
        assert params["stream_options"]["include_usage"] is True

    def test_does_not_override_existing_include_usage(self) -> None:
        """Verify existing include_usage is not overridden."""
        params: dict[str, Any] = {
            "stream": True,
            "stream_options": {"include_usage": False},
            "messages": [],
        }
        OpenAIClient._add_streaming_usage_to_params(params)

        # setdefault should not override existing value
        assert params["stream_options"]["include_usage"] is False

    def test_streaming_without_structured_output_captures_usage(self, credentials_openai_mini: Credentials) -> None:
        """Test that streaming without structured output correctly captures usage metrics.

        When stream=True without response_format, streaming should work normally
        and usage metrics should be captured from the final chunk.
        """
        config_list = credentials_openai_mini.config_list

        # Add stream=True without response_format
        for config in config_list:
            config["stream"] = True

        client = OpenAIWrapper(config_list=config_list, cache_seed=None)

        response = client.create(
            messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
        )

        # Verify response is valid
        assert response is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

        # Verify usage metrics were captured
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


# --- test/tools/experimental/reliable/test_reliable.py ---

    def test_bad_response(self, credentials_openai_mini: Credentials) -> None:
        # Skip this test in GitHub Actions due to SQLite database permission issues
        if os.getenv("GITHUB_ACTIONS") == "true":
            pytest.skip("Skipping ReliableTool test in GitHub Actions due to SQLite database permission issues")

        should_bad_response = True

        def generate_sub_questions_list(
            sub_questions: Annotated[list[str], "A list of sub-questions related to the main question."],
        ) -> list[str]:
            """
            Receives and returns a list of generated sub-questions.
            """
            nonlocal should_bad_response
            if should_bad_response:
                should_bad_response = False
                return []
            return sub_questions

        sub_question_tool = ReliableTool(
            name="SubQuestionGenerator",
            func_or_tool=generate_sub_questions_list,
            description="Reliably generates exactly 3 relevant sub-questions for a given main question.",
            runner_llm_config=credentials_openai_mini.llm_config,
            validator_llm_config=credentials_openai_mini.llm_config,
            system_message_addition_for_tool_calling=sub_question_runner_system_message_addition,
            system_message_addition_for_result_validation=sub_question_validator_system_message_addition,
            max_tool_invocations=5,
        )

        result: ToolExecutionDetails = sub_question_tool.run_and_get_details(
            task="How does photosynthesis work in plants?"
        )

        # Should fail once, then pass because of should_bad_response
        assert result.final_tool_context.attempt_count == 2
        assert not should_bad_response

    def test_error(self, credentials_openai_mini: Credentials) -> None:
        # Skip this test in GitHub Actions due to SQLite database permission issues
        if os.getenv("GITHUB_ACTIONS") == "true":
            pytest.skip("Skipping ReliableTool test in GitHub Actions due to SQLite database permission issues")

        should_error = True

        def generate_sub_questions_list(
            sub_questions: Annotated[list[str], "A list of sub-questions related to the main question."],
        ) -> list[str]:
            """
            Receives and returns a list of generated sub-questions.
            """
            nonlocal should_error
            if should_error:
                should_error = False
                raise Exception("Test Error")
            return sub_questions

        sub_question_tool = ReliableTool(
            name="SubQuestionGenerator",
            func_or_tool=generate_sub_questions_list,
            description="Reliably generates exactly 3 relevant sub-questions for a given main question.",
            runner_llm_config=credentials_openai_mini.llm_config,
            validator_llm_config=credentials_openai_mini.llm_config,
            system_message_addition_for_tool_calling=sub_question_runner_system_message_addition,
            system_message_addition_for_result_validation=sub_question_validator_system_message_addition,
            max_tool_invocations=5,
        )

        result: ToolExecutionDetails = sub_question_tool.run_and_get_details(
            task="How does photosynthesis work in plants?"
        )

        # Should fail once, then pass because of should_error
        assert result.final_tool_context.attempt_count == 2
        assert not should_error

    def test_return_tuple(self, credentials_openai_mini: Credentials) -> None:
        # Skip this test in GitHub Actions due to SQLite database permission issues
        if os.getenv("GITHUB_ACTIONS") == "true":
            pytest.skip("Skipping ReliableTool test in GitHub Actions due to SQLite database permission issues")

        should_error = True

        def generate_sub_questions_list(
            sub_questions: Annotated[list[str], "A list of sub-questions related to the main question."],
        ) -> tuple[list[str], str]:
            """
            Receives and returns a list of generated sub-questions.
            """
            nonlocal should_error
            if should_error:
                should_error = False
                raise Exception("Test Error")
            return sub_questions, "Sub Questions are:\n" + "\n".join(sub_questions)

        sub_question_tool = ReliableTool(
            name="SubQuestionGenerator",
            func_or_tool=generate_sub_questions_list,
            description="Reliably generates exactly 3 relevant sub-questions for a given main question.",
            runner_llm_config=credentials_openai_mini.llm_config,
            validator_llm_config=credentials_openai_mini.llm_config,
            system_message_addition_for_tool_calling=sub_question_runner_system_message_addition,
            system_message_addition_for_result_validation=sub_question_validator_system_message_addition,
            max_tool_invocations=5,
        )

        result: ToolExecutionDetails = sub_question_tool.run_and_get_details(
            task="How does photosynthesis work in plants?"
        )

        # Should fail once, then pass because of should_error
        assert result.final_tool_context.attempt_count == 2
        assert not should_error
        assert isinstance(result.final_tool_context.get_final_result_data(), list)
        assert isinstance(result.final_tool_context.get_final_result_str(), str)

