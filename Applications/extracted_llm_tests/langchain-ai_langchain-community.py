# langchain-ai/langchain-community
# 620 LLM-backed test functions across 733 test files
# Source: https://github.com/langchain-ai/langchain-community

# --- libs/community/tests/integration_tests/test_dalle.py ---

def test_call() -> None:
    """Test that call returns a URL in the output."""
    search = DallEAPIWrapper()
    output = search.run("volcano island")
    assert "https://oaidalleapi" in output


# --- libs/community/tests/integration_tests/test_long_context_reorder.py ---

def test_long_context_reorder() -> None:
    """Test Lost in the middle reordering get_relevant_docs."""
    texts = [
        "Basquetball is a great sport.",
        "Fly me to the moon is one of my favourite songs.",
        "The Celtics are my favourite team.",
        "This is a document about the Boston Celtics",
        "I simply love going to the movies",
        "The Boston Celtics won the game by 20 points",
        "This is just a random text.",
        "Elden Ring is one of the best games in the last 15 years.",
        "L. Kornet is one of the best Celtics players.",
        "Larry Bird was an iconic NBA player.",
    ]
    embeddings = OpenAIEmbeddings()
    retriever = InMemoryVectorStore.from_texts(
        texts, embedding=embeddings
    ).as_retriever(search_kwargs={"k": 10})
    reordering = LongContextReorder()
    docs = retriever.invoke("Tell me about the Celtics")
    actual = reordering.transform_documents(docs)

    # First 2 and Last 2 elements must contain the most relevant
    first_and_last = list(actual[:2]) + list(actual[-2:])
    assert len(actual) == 10
    assert texts[2] in [d.page_content for d in first_and_last]
    assert texts[3] in [d.page_content for d in first_and_last]
    assert texts[5] in [d.page_content for d in first_and_last]
    assert texts[8] in [d.page_content for d in first_and_last]


# --- libs/community/tests/integration_tests/test_pdf_pagesplitter.py ---

def test_pdf_pagesplitter() -> None:
    """Test splitting with page numbers included."""
    script_dir = os.path.dirname(__file__)
    loader = PyPDFLoader(os.path.join(script_dir, "examples/hello.pdf"))
    docs = loader.load()
    assert "page" in docs[0].metadata
    assert "source" in docs[0].metadata

    faiss_index = FAISS.from_documents(docs, OpenAIEmbeddings())
    docs = faiss_index.similarity_search("Complete this sentence: Hello", k=1)
    assert "Hello world" in docs[0].page_content


# --- libs/community/tests/integration_tests/agent/test_ainetwork_agent.py ---

def test_ainetwork_toolkit() -> None:
    def get(path: str, type: str = "value", default: Any = None) -> Any:
        ref = ain.db.ref(path)
        value = asyncio.run(
            {
                "value": ref.getValue,
                "rule": ref.getRule,
                "owner": ref.getOwner,
            }[type]()
        )
        return default if value is None else value

    def validate(path: str, template: Any, type: str = "value") -> bool:
        value = get(path, type)
        return Match.match(value, template)

    if not os.environ.get("AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY", None):
        from ain.account import Account

        account = Account.create()
        os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"] = account.private_key

    interface = authenticate(network="testnet")
    toolkit = AINetworkToolkit(network="testnet", interface=interface)
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    agent = initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
    )
    ain = interface
    self_address = ain.wallet.defaultAccount.address
    co_address = "0x6813Eb9362372EEF6200f3b1dbC3f819671cBA69"

    # Test creating an app
    UUID = uuid.UUID(
        int=(int(time.time() * 1000) << 64) | (uuid.uuid4().int & ((1 << 64) - 1))
    )
    app_name = f"_langchain_test__{str(UUID).replace('-', '_')}"
    agent.run(f"""Create app {app_name}""")
    validate(f"/manage_app/{app_name}/config", {"admin": {self_address: True}})
    validate(f"/apps/{app_name}/DB", None, "owner")

    # Test reading owner config
    agent.run(f"""Read owner config of /apps/{app_name}/DB .""")
    assert ...

    # Test granting owner config
    agent.run(
        f"""Grant owner authority to {co_address} for edit write rule permission of /apps/{app_name}/DB_co ."""  # noqa: E501
    )
    validate(
        f"/apps/{app_name}/DB_co",
        {
            ".owner": {
                "owners": {
                    co_address: {
                        "branch_owner": False,
                        "write_function": False,
                        "write_owner": False,
                        "write_rule": True,
                    }
                }
            }
        },
        "owner",
    )

    # Test reading owner config
    agent.run(f"""Read owner config of /apps/{app_name}/DB_co .""")
    assert ...

    # Test reading owner config
    agent.run(f"""Read owner config of /apps/{app_name}/DB .""")
    assert ...  # Check if owner {self_address} exists

    # Test reading a value
    agent.run(f"""Read value in /apps/{app_name}/DB""")
    assert ...  # empty

    # Test writing a value
    agent.run(f"""Write value {{1: 1904, 2: 43}} in /apps/{app_name}/DB""")
    validate(f"/apps/{app_name}/DB", {1: 1904, 2: 43})

    # Test reading a value
    agent.run(f"""Read value in /apps/{app_name}/DB""")
    assert ...  # check value

    # Test reading a rule
    agent.run(f"""Read write rule of app {app_name} .""")
    assert ...  # check rule that self_address exists

    # Test sending AIN
    self_balance = get(f"/accounts/{self_address}/balance", default=0)
    transaction_history = get(f"/transfer/{self_address}/{co_address}", default={})
    if self_balance < 1:
        try:
            with urllib.request.urlopen(
                f"http://faucet.ainetwork.ai/api/test/{self_address}/"
            ) as response:
                try_test = response.getcode()
        except HTTPError as e:
            try_test = e.getcode()
    else:
        try_test = 200

    if try_test == 200:
        agent.run(f"""Send 1 AIN to {co_address}""")
        transaction_update = get(f"/transfer/{self_address}/{co_address}", default={})
        assert any(
            transaction_update[key]["value"] == 1
            for key in transaction_update.keys() - transaction_history.keys()
        )


# --- libs/community/tests/integration_tests/agent/test_powerbi_agent.py ---

def test_daxquery() -> None:
    from azure.identity import DefaultAzureCredential

    DATASET_ID = get_from_env("", "POWERBI_DATASET_ID")
    TABLE_NAME = get_from_env("", "POWERBI_TABLE_NAME")
    NUM_ROWS = get_from_env("", "POWERBI_NUMROWS")

    fast_llm = ChatOpenAI(
        temperature=0.5, max_tokens=1000, model="gpt-3.5-turbo", verbose=True
    )
    smart_llm = ChatOpenAI(temperature=0, max_tokens=100, model="gpt-4", verbose=True)

    toolkit = PowerBIToolkit(
        powerbi=PowerBIDataset(
            dataset_id=DATASET_ID,
            table_names=[TABLE_NAME],
            credential=DefaultAzureCredential(),
        ),
        llm=smart_llm,
    )

    agent_executor = create_pbi_agent(llm=fast_llm, toolkit=toolkit, verbose=True)

    output = agent_executor.run(f"How many rows are in the table, {TABLE_NAME}")
    assert NUM_ROWS in output


# --- libs/community/tests/integration_tests/callbacks/test_langchain_tracer.py ---

def test_tracing_sequential() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    os.environ["LANGCHAIN_TRACING"] = "true"

    for q in questions[:3]:
        llm = OpenAI(temperature=0)
        tools = load_tools(["llm-math", "serpapi"], llm=llm)
        agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
        agent.run(q)

def test_tracing_session_env_var() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    os.environ["LANGCHAIN_TRACING"] = "true"
    os.environ["LANGCHAIN_SESSION"] = "my_session"

    llm = OpenAI(temperature=0)
    tools = load_tools(["llm-math", "serpapi"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    agent.run(questions[0])
    if "LANGCHAIN_SESSION" in os.environ:
        del os.environ["LANGCHAIN_SESSION"]

async def test_tracing_concurrent() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    os.environ["LANGCHAIN_TRACING"] = "true"
    aiosession = ClientSession()
    llm = OpenAI(temperature=0)
    async_tools = load_tools(["llm-math", "serpapi"], llm=llm, aiosession=aiosession)
    agent = initialize_agent(
        async_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    tasks = [agent.arun(q) for q in questions[:3]]
    await asyncio.gather(*tasks)
    await aiosession.close()

async def test_tracing_concurrent_bw_compat_environ() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    os.environ["LANGCHAIN_HANDLER"] = "langchain"
    if "LANGCHAIN_TRACING" in os.environ:
        del os.environ["LANGCHAIN_TRACING"]
    aiosession = ClientSession()
    llm = OpenAI(temperature=0)
    async_tools = load_tools(["llm-math", "serpapi"], llm=llm, aiosession=aiosession)
    agent = initialize_agent(
        async_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    tasks = [agent.arun(q) for q in questions[:3]]
    await asyncio.gather(*tasks)
    await aiosession.close()
    if "LANGCHAIN_HANDLER" in os.environ:
        del os.environ["LANGCHAIN_HANDLER"]

async def test_tracing_v2_environment_variable() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    aiosession = ClientSession()
    llm = OpenAI(temperature=0)
    async_tools = load_tools(["llm-math", "serpapi"], llm=llm, aiosession=aiosession)
    agent = initialize_agent(
        async_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    tasks = [agent.arun(q) for q in questions[:3]]
    await asyncio.gather(*tasks)
    await aiosession.close()

def test_tracing_v2_context_manager() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    llm = ChatOpenAI(temperature=0)
    tools = load_tools(["llm-math", "serpapi"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    if "LANGCHAIN_TRACING_V2" in os.environ:
        del os.environ["LANGCHAIN_TRACING_V2"]
    with tracing_v2_enabled():
        agent.run(questions[0])  # this should be traced

    agent.run(questions[0])  # this should not be traced

def test_tracing_v2_chain_with_tags() -> None:
    from langchain_classic.chains.constitutional_ai.base import ConstitutionalChain
    from langchain_classic.chains.constitutional_ai.models import (
        ConstitutionalPrinciple,
    )
    from langchain_classic.chains.llm import LLMChain

    llm = OpenAI(temperature=0)
    chain = ConstitutionalChain.from_llm(
        llm,
        chain=LLMChain.from_string(llm, "Q: {question} A:"),
        tags=["only-root"],
        constitutional_principles=[
            ConstitutionalPrinciple(
                critique_request="Tell if this answer is good.",
                revision_request="Give a better answer.",
            )
        ],
    )
    if "LANGCHAIN_TRACING_V2" in os.environ:
        del os.environ["LANGCHAIN_TRACING_V2"]
    with tracing_v2_enabled():
        chain.run("what is the meaning of life", tags=["a-tag"])

def test_tracing_v2_agent_with_metadata() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    llm = OpenAI(temperature=0)
    chat = ChatOpenAI(temperature=0)
    tools = load_tools(["llm-math", "serpapi"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    chat_agent = initialize_agent(
        tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    agent.run(questions[0], tags=["a-tag"], metadata={"a": "b", "c": "d"})
    chat_agent.run(questions[0], tags=["a-tag"], metadata={"a": "b", "c": "d"})

async def test_tracing_v2_async_agent_with_metadata() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    llm = OpenAI(temperature=0, metadata={"f": "g", "h": "i"})
    chat = ChatOpenAI(temperature=0, metadata={"f": "g", "h": "i"})
    async_tools = load_tools(["llm-math", "serpapi"], llm=llm)
    agent = initialize_agent(
        async_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    chat_agent = initialize_agent(
        async_tools,
        chat,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    await agent.arun(questions[0], tags=["a-tag"], metadata={"a": "b", "c": "d"})
    await chat_agent.arun(questions[0], tags=["a-tag"], metadata={"a": "b", "c": "d"})

def test_trace_as_group() -> None:
    from langchain_classic.chains.llm import LLMChain

    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    with trace_as_chain_group("my_group", inputs={"input": "cars"}) as group_manager:
        chain.run(product="cars", callbacks=group_manager)
        chain.run(product="computers", callbacks=group_manager)
        final_res = chain.run(product="toys", callbacks=group_manager)
        group_manager.on_chain_end({"output": final_res})

    with trace_as_chain_group("my_group_2", inputs={"input": "toys"}) as group_manager:
        final_res = chain.run(product="toys", callbacks=group_manager)
        group_manager.on_chain_end({"output": final_res})

def test_trace_as_group_with_env_set() -> None:
    from langchain_classic.chains.llm import LLMChain

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    with trace_as_chain_group(
        "my_group_env_set", inputs={"input": "cars"}
    ) as group_manager:
        chain.run(product="cars", callbacks=group_manager)
        chain.run(product="computers", callbacks=group_manager)
        final_res = chain.run(product="toys", callbacks=group_manager)
        group_manager.on_chain_end({"output": final_res})

    with trace_as_chain_group(
        "my_group_2_env_set", inputs={"input": "toys"}
    ) as group_manager:
        final_res = chain.run(product="toys", callbacks=group_manager)
        group_manager.on_chain_end({"output": final_res})

async def test_trace_as_group_async() -> None:
    from langchain_classic.chains.llm import LLMChain

    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    async with atrace_as_chain_group("my_async_group") as group_manager:
        await chain.arun(product="cars", callbacks=group_manager)
        await chain.arun(product="computers", callbacks=group_manager)
        await chain.arun(product="toys", callbacks=group_manager)

    async with atrace_as_chain_group(
        "my_async_group_2", inputs={"input": "toys"}
    ) as group_manager:
        res = await asyncio.gather(
            *[
                chain.arun(product="toys", callbacks=group_manager),
                chain.arun(product="computers", callbacks=group_manager),
                chain.arun(product="cars", callbacks=group_manager),
            ]
        )
        await group_manager.on_chain_end({"output": res})


# --- libs/community/tests/integration_tests/callbacks/test_openai_callback.py ---

async def test_openai_callback() -> None:
    llm = OpenAI(temperature=0)
    with get_openai_callback() as cb:
        llm.invoke("What is the square root of 4?")

    total_tokens = cb.total_tokens
    assert total_tokens > 0

    with get_openai_callback() as cb:
        llm.invoke("What is the square root of 4?")
        llm.invoke("What is the square root of 4?")

    assert cb.total_tokens == total_tokens * 2

    with get_openai_callback() as cb:
        await asyncio.gather(
            *[llm.agenerate(["What is the square root of 4?"]) for _ in range(3)]
        )

    assert cb.total_tokens == total_tokens * 3

    task = asyncio.create_task(llm.agenerate(["What is the square root of 4?"]))
    with get_openai_callback() as cb:
        await llm.agenerate(["What is the square root of 4?"])

    await task
    assert cb.total_tokens == total_tokens

def test_openai_callback_batch_llm() -> None:
    llm = OpenAI(temperature=0)
    with get_openai_callback() as cb:
        llm.generate(["What is the square root of 4?", "What is the square root of 4?"])

    assert cb.total_tokens > 0
    total_tokens = cb.total_tokens

    with get_openai_callback() as cb:
        llm.invoke("What is the square root of 4?")
        llm.invoke("What is the square root of 4?")

    assert cb.total_tokens == total_tokens

def test_openai_callback_agent() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    with get_openai_callback() as cb:
        agent.run(
            "Who is Olivia Wilde's boyfriend? "
            "What is his current age raised to the 0.23 power?"
        )
        print(f"Total Tokens: {cb.total_tokens}")  # noqa: T201
        print(f"Prompt Tokens: {cb.prompt_tokens}")  # noqa: T201
        print(f"Completion Tokens: {cb.completion_tokens}")  # noqa: T201
        print(f"Total Cost (USD): ${cb.total_cost}")  # noqa: T201


# --- libs/community/tests/integration_tests/callbacks/test_streamlit_callback.py ---

def test_streamlit_callback_agent() -> None:
    import streamlit as st
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    streamlit_callback = StreamlitCallbackHandler(st.container())

    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    agent.run(
        "Who is Olivia Wilde's boyfriend? "
        "What is his current age raised to the 0.23 power?",
        callbacks=[streamlit_callback],
    )


# --- libs/community/tests/integration_tests/callbacks/test_wandb_tracer.py ---

def test_tracing_sequential() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
    os.environ["WANDB_PROJECT"] = "langchain-tracing"

    for q in questions[:3]:
        llm = OpenAI(temperature=0)
        tools = load_tools(
            ["llm-math", "serpapi"],
            llm=llm,
        )
        agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
        agent.run(q)

def test_tracing_session_env_var() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

    llm = OpenAI(temperature=0)
    tools = load_tools(
        ["llm-math", "serpapi"],
        llm=llm,
    )
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    agent.run(questions[0])

async def test_tracing_concurrent() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
    aiosession = ClientSession()
    llm = OpenAI(temperature=0)
    async_tools = load_tools(
        ["llm-math", "serpapi"],
        llm=llm,
        aiosession=aiosession,
    )
    agent = initialize_agent(
        async_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    tasks = [agent.arun(q) for q in questions[:3]]
    await asyncio.gather(*tasks)
    await aiosession.close()

def test_tracing_context_manager() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    llm = OpenAI(temperature=0)
    tools = load_tools(
        ["llm-math", "serpapi"],
        llm=llm,
    )
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    if "LANGCHAIN_WANDB_TRACING" in os.environ:
        del os.environ["LANGCHAIN_WANDB_TRACING"]
    with wandb_tracing_enabled():
        agent.run(questions[0])  # this should be traced

    agent.run(questions[0])  # this should not be traced

async def test_tracing_context_manager_async() -> None:
    from langchain_classic.agents import AgentType, initialize_agent, load_tools

    llm = OpenAI(temperature=0)
    async_tools = load_tools(
        ["llm-math", "serpapi"],
        llm=llm,
    )
    agent = initialize_agent(
        async_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    if "LANGCHAIN_WANDB_TRACING" in os.environ:
        del os.environ["LANGCHAIN_TRACING"]

    # start a background task
    task = asyncio.create_task(agent.arun(questions[0]))  # this should not be traced
    with wandb_tracing_enabled():
        tasks = [agent.arun(q) for q in questions[1:4]]  # these should be traced
        await asyncio.gather(*tasks)

    await task


# --- libs/community/tests/integration_tests/chains/test_dalle_agent.py ---

def test_call() -> None:
    """Test that the agent runs and returns output."""
    llm = OpenAI(temperature=0.9)
    tools = load_tools(["dalle-image-generator"])

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    output = agent.run("Create an image of a volcano island")
    assert output is not None


# --- libs/community/tests/integration_tests/chains/test_graph_database.py ---

def test_cypher_generating_run() -> None:
    """Test that Cypher statement is correctly generated and executed."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password,
    )
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        "CREATE (a:Actor {name:'Bruce Willis'})"
        "-[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    chain = GraphCypherQAChain.from_llm(OpenAI(temperature=0), graph=graph)
    output = chain.run("Who played in Pulp Fiction?")
    expected_output = " Bruce Willis played in Pulp Fiction."
    assert output == expected_output

def test_cypher_top_k() -> None:
    """Test top_k parameter correctly limits the number of results in the context."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    TOP_K = 1

    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password,
    )
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        "CREATE (a:Actor {name:'Bruce Willis'})"
        "-[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})"
        "<-[:ACTED_IN]-(:Actor {name:'Foo'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    chain = GraphCypherQAChain.from_llm(
        OpenAI(temperature=0), graph=graph, return_direct=True, top_k=TOP_K
    )
    output = chain.run("Who played in Pulp Fiction?")
    assert len(output) == TOP_K

def test_cypher_return_direct() -> None:
    """Test that chain returns direct results."""
    url = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    assert url is not None
    assert username is not None
    assert password is not None

    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password,
    )
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        "CREATE (a:Actor {name:'Bruce Willis'})"
        "-[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    chain = GraphCypherQAChain.from_llm(
        OpenAI(temperature=0), graph=graph, return_direct=True
    )
    output = chain.run("Who played in Pulp Fiction?")
    expected_output = [{"a.name": "Bruce Willis"}]
    assert output == expected_output


# --- libs/community/tests/integration_tests/chains/test_ontotext_graphdb_qa.py ---

def test_chain(model_name: str, question: str) -> None:
    from langchain_openai import ChatOpenAI

    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/starwars",
        query_ontology="CONSTRUCT {?s ?p ?o} "
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        ChatOpenAI(temperature=0, model_name=model_name),
        graph=graph,
        verbose=True,
    )
    try:
        chain.invoke({chain.input_key: question})
    except ValueError:
        pass


# --- libs/community/tests/integration_tests/chains/test_react.py ---

def test_react() -> None:
    """Test functionality on a prompt."""
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")  # type: ignore[call-arg]
    react = ReActChain(llm=llm, docstore=Wikipedia())
    question = (
        "Author David Chanoff has collaborated with a U.S. Navy admiral "
        "who served as the ambassador to the United Kingdom under "
        "which President?"
    )
    output = react.run(question)
    assert output == "Bill Clinton"


# --- libs/community/tests/integration_tests/chains/test_retrieval_qa.py ---

def test_retrieval_qa_saving_loading(tmp_path: Path) -> None:
    """Test saving and loading."""
    loader = TextLoader("docs/extras/modules/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(texts, embeddings)
    qa = RetrievalQA.from_llm(llm=OpenAI(), retriever=docsearch.as_retriever())
    qa.run("What did the president say about Ketanji Brown Jackson?")

    file_path = tmp_path / "RetrievalQA_chain.yaml"
    qa.save(file_path=file_path)
    qa_loaded = load_chain(file_path, retriever=docsearch.as_retriever())

    assert qa_loaded == qa


# --- libs/community/tests/integration_tests/chains/test_self_ask_with_search.py ---

def test_self_ask_with_search() -> None:
    """Test functionality on a prompt."""
    question = "What is the hometown of the reigning men's U.S. Open champion?"
    chain = SelfAskWithSearchChain(
        llm=OpenAI(temperature=0),
        search_chain=SearchApiAPIWrapper(),
        input_key="q",
        output_key="a",
    )
    answer = chain.run(question)
    final_answer = answer.split("\n")[-1]
    assert final_answer == "Belgrade, Serbia"


# --- libs/community/tests/integration_tests/chat_models/test_azureml_endpoint.py ---

def test_llama_call() -> None:
    """Test valid call to Open Source Foundation Model."""
    chat = AzureMLChatOnlineEndpoint(
        content_formatter=CustomOpenAIChatContentFormatter()
    )
    response = chat.invoke([HumanMessage(content="Foo")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_temperature_kwargs() -> None:
    """Test that timeout kwarg works."""
    chat = AzureMLChatOnlineEndpoint(
        content_formatter=CustomOpenAIChatContentFormatter()
    )
    response = chat.invoke([HumanMessage(content="FOO")], temperature=0.8)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_message_history() -> None:
    """Test that multiple messages works."""
    chat = AzureMLChatOnlineEndpoint(
        content_formatter=CustomOpenAIChatContentFormatter()
    )
    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_multiple_messages() -> None:
    chat = AzureMLChatOnlineEndpoint(
        content_formatter=CustomOpenAIChatContentFormatter()
    )
    message = HumanMessage(content="Hi!")
    response = chat.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


# --- libs/community/tests/integration_tests/chat_models/test_baichuan.py ---

def test_chat_baichuan_default() -> None:
    chat = ChatBaichuan(streaming=True)  # type: ignore[call-arg]
    message = HumanMessage(content="请完整背诵将进酒，背诵5遍")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_baichuan_default_non_streaming() -> None:
    chat = ChatBaichuan()  # type: ignore[call-arg]
    message = HumanMessage(content="请完整背诵将进酒，背诵5遍")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_baichuan_turbo() -> None:
    chat = ChatBaichuan(model="Baichuan2-Turbo", streaming=True)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_baichuan_turbo_non_streaming() -> None:
    chat = ChatBaichuan(model="Baichuan2-Turbo")  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_baichuan_with_temperature() -> None:
    chat = ChatBaichuan(temperature=1.0)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_baichuan_with_kwargs() -> None:
    chat = ChatBaichuan()  # type: ignore[call-arg]
    message = HumanMessage(content="百川192K API是什么时候上线的？")
    response = chat.invoke(
        [message], temperature=0.88, top_p=0.7, with_search_enhance=True
    )
    print(response)  # noqa: T201
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

async def test_chat_baichuan_agenerate() -> None:
    chat = ChatBaichuan()  # type: ignore[call-arg]
    response = await chat.ainvoke("你好呀")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

async def test_chat_baichuan_astream() -> None:
    chat = ChatBaichuan()  # type: ignore[call-arg]
    async for chunk in chat.astream("今天天气如何？"):
        assert isinstance(chunk, AIMessage)
        assert isinstance(chunk.content, str)

def test_chat_baichuan_with_system_role() -> None:
    chat = ChatBaichuan()  # type: ignore[call-arg]
    messages = [
        ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
        ("human", "我喜欢编程。"),
    ]
    response = chat.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


# --- libs/community/tests/integration_tests/chat_models/test_baiduqianfan.py ---

def test_chat_qianfan_tool_result_to_model() -> None:
    """Test QianfanChatEndpoint invoke with tool_calling result."""
    messages = [
        HumanMessage("上海天气怎么样？"),
        AIMessage(
            content=" ",
            tool_calls=[
                ToolCall(
                    name="get_current_weather",
                    args={"location": "上海", "unit": "摄氏度"},
                    id="foo",
                    type="tool_call",
                ),
            ],
        ),
        ToolMessage(
            content="上海是晴天，25度左右。",
            tool_call_id="foo",
            name="get_current_weather",
        ),
    ]
    chat = QianfanChatEndpoint(model="ERNIE-3.5-8K")  # type: ignore[call-arg]
    llm_with_tool = chat.bind_tools([get_current_weather])
    response = llm_with_tool.invoke(messages)
    assert isinstance(response, AIMessage)
    print(response.content)  # noqa: T201


# --- libs/community/tests/integration_tests/chat_models/test_coze.py ---

def test_chat_coze_default() -> None:
    chat = ChatCoze(
        coze_api_base="https://api.coze.com",
        coze_api_key="pat_...",  # type: ignore[arg-type]
        bot_id="7....",
        user="123",
        conversation_id="",
        streaming=True,
    )
    message = HumanMessage(content="请完整背诵将进酒，背诵5遍")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_coze_default_non_streaming() -> None:
    chat = ChatCoze(
        coze_api_base="https://api.coze.com",
        coze_api_key="pat_...",  # type: ignore[arg-type]
        bot_id="7....",
        user="123",
        conversation_id="",
        streaming=False,
    )
    message = HumanMessage(content="请完整背诵将进酒，背诵5遍")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


# --- libs/community/tests/integration_tests/chat_models/test_dappier.py ---

def test_dappier_chat() -> None:
    """Test ChatDappierAI wrapper."""
    chat = ChatDappierAI(  # type: ignore[call-arg]
        dappier_endpoint="https://api.dappier.com/app/datamodelconversation",
        dappier_model="dm_01hpsxyfm2fwdt2zet9cg6fdxt",
    )
    message = HumanMessage(content="Who are you ?")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_dappier_generate() -> None:
    """Test generate method of Dappier AI."""
    chat = ChatDappierAI(  # type: ignore[call-arg]
        dappier_endpoint="https://api.dappier.com/app/datamodelconversation",
        dappier_model="dm_01hpsxyfm2fwdt2zet9cg6fdxt",
    )
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="Who won the last super bowl?")],
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy

async def test_dappier_agenerate() -> None:
    """Test async generation."""
    chat = ChatDappierAI(  # type: ignore[call-arg]
        dappier_endpoint="https://api.dappier.com/app/datamodelconversation",
        dappier_model="dm_01hpsxyfm2fwdt2zet9cg6fdxt",
    )
    message = HumanMessage(content="Who won the last super bowl?")
    result: LLMResult = await chat.agenerate([[message], [message]])
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content


# --- libs/community/tests/integration_tests/chat_models/test_deepinfra.py ---

def test_chat_deepinfra() -> None:
    """Test valid call to DeepInfra."""
    chat = ChatDeepInfra(
        max_tokens=10,
    )
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

async def test_async_chat_deepinfra() -> None:
    """Test async generation."""
    chat = ChatDeepInfra(
        max_tokens=10,
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    assert len(response.generations[0]) == 1
    generation = response.generations[0][0]
    assert isinstance(generation, ChatGeneration)
    assert isinstance(generation.text, str)
    assert generation.text == generation.message.content

def test_chat_deepinfra_bind_tools() -> None:
    class Foo(BaseModel):
        pass

    chat = ChatDeepInfra(
        max_tokens=10,
    )
    tools = [Foo]
    chat_with_tools = chat.bind_tools(tools)
    assert isinstance(chat_with_tools, RunnableBinding)
    chat_tools = chat_with_tools.tools
    assert chat_tools
    assert chat_tools == {
        "tools": [
            {
                "function": {
                    "description": "",
                    "name": "Foo",
                    "parameters": {"properties": {}, "type": "object"},
                },
                "type": "function",
            }
        ]
    }

def test_tool_use() -> None:
    llm = ChatDeepInfra(model="meta-llama/Meta-Llama-3-70B-Instruct", temperature=0)
    llm_with_tool = llm.bind_tools(tools=[GenerateMovieName], tool_choice=True)
    msgs: List = [
        HumanMessage(content="It should be a movie explaining humanity in 2133.")
    ]
    ai_msg = llm_with_tool.invoke(msgs)

    assert isinstance(ai_msg, AIMessage)
    assert isinstance(ai_msg.tool_calls, list)
    assert len(ai_msg.tool_calls) == 1
    tool_call = ai_msg.tool_calls[0]
    assert "args" in tool_call

    tool_msg = ToolMessage(
        content="Year 2133",
        tool_call_id=ai_msg.additional_kwargs["tool_calls"][0]["id"],
    )
    msgs.extend([ai_msg, tool_msg])
    llm_with_tool.invoke(msgs)


# --- libs/community/tests/integration_tests/chat_models/test_edenai.py ---

def test_chat_edenai() -> None:
    """Test ChatEdenAI wrapper."""
    chat = ChatEdenAI(  # type: ignore[call-arg]
        provider="openai", model="gpt-3.5-turbo", temperature=0, max_tokens=1000
    )
    message = HumanMessage(content="Who are you ?")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_edenai_generate() -> None:
    """Test generate method of edenai."""
    chat = ChatEdenAI(provider="google")  # type: ignore[call-arg]
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="What is the meaning of life?")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy

async def test_edenai_async_generate() -> None:
    """Test async generation."""
    chat = ChatEdenAI(provider="google", max_tokens=50)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    result: LLMResult = await chat.agenerate([[message], [message]])
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content

def test_edenai_streaming() -> None:
    """Test streaming EdenAI chat."""
    llm = ChatEdenAI(provider="openai", max_tokens=50)  # type: ignore[call-arg]

    for chunk in llm.stream("Generate a high fantasy story."):
        assert isinstance(chunk.content, str)

async def test_edenai_astream() -> None:
    """Test streaming from EdenAI."""
    llm = ChatEdenAI(provider="openai", max_tokens=50)  # type: ignore[call-arg]

    async for token in llm.astream("Generate a high fantasy story."):
        assert isinstance(token.content, str)


# --- libs/community/tests/integration_tests/chat_models/test_ernie.py ---

def test_chat_ernie_bot() -> None:
    chat = ErnieBotChat()
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_ernie_bot_with_model_name() -> None:
    chat = ErnieBotChat(model_name="ERNIE-Bot")
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_ernie_bot_with_temperature() -> None:
    chat = ErnieBotChat(model_name="ERNIE-Bot", temperature=1.0)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_ernie_bot_with_kwargs() -> None:
    chat = ErnieBotChat()
    message = HumanMessage(content="Hello")
    response = chat.invoke([message], temperature=0.88, top_p=0.7)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_wrong_temperature_1() -> None:
    chat = ErnieBotChat()
    message = HumanMessage(content="Hello")
    with pytest.raises(ValueError) as e:
        chat.invoke([message], temperature=1.2)
    assert "parameter check failed, temperature range is (0, 1.0]" in str(e)

def test_wrong_temperature_2() -> None:
    chat = ErnieBotChat()
    message = HumanMessage(content="Hello")
    with pytest.raises(ValueError) as e:
        chat.invoke([message], temperature=0)
    assert "parameter check failed, temperature range is (0, 1.0]" in str(e)


# --- libs/community/tests/integration_tests/chat_models/test_friendli.py ---

def test_friendli_invoke(friendli_chat: ChatFriendli) -> None:
    """Test invoke."""
    output = friendli_chat.invoke("What is generative AI?")
    assert isinstance(output, AIMessage)
    assert isinstance(output.content, str)

async def test_friendli_ainvoke(friendli_chat: ChatFriendli) -> None:
    """Test async invoke."""
    output = await friendli_chat.ainvoke("What is generative AI?")
    assert isinstance(output, AIMessage)
    assert isinstance(output.content, str)

def test_friendli_generate(friendli_chat: ChatFriendli) -> None:
    """Test generate."""
    message = HumanMessage(content="What is generative AI?")
    result = friendli_chat.generate([[message], [message]])
    assert isinstance(result, LLMResult)
    generations = result.generations
    assert len(generations) == 2
    for generation in generations:
        gen_ = generation[0]
        assert isinstance(gen_, Generation)
        text = gen_.text
        assert isinstance(text, str)
        generation_info = gen_.generation_info
        if generation_info is not None:
            assert "token" in generation_info

async def test_friendli_agenerate(friendli_chat: ChatFriendli) -> None:
    """Test async generate."""
    message = HumanMessage(content="What is generative AI?")
    result = await friendli_chat.agenerate([[message], [message]])
    assert isinstance(result, LLMResult)
    generations = result.generations
    assert len(generations) == 2
    for generation in generations:
        gen_ = generation[0]
        assert isinstance(gen_, Generation)
        text = gen_.text
        assert isinstance(text, str)
        generation_info = gen_.generation_info
        if generation_info is not None:
            assert "token" in generation_info

def test_friendli_stream(friendli_chat: ChatFriendli) -> None:
    """Test stream."""
    stream = friendli_chat.stream("Say hello world.")
    for chunk in stream:
        assert isinstance(chunk, AIMessage)
        assert isinstance(chunk.content, str)

async def test_friendli_astream(friendli_chat: ChatFriendli) -> None:
    """Test async stream."""
    stream = friendli_chat.astream("Say hello world.")
    async for chunk in stream:
        assert isinstance(chunk, AIMessage)
        assert isinstance(chunk.content, str)


# --- libs/community/tests/integration_tests/chat_models/test_google_palm.py ---

def test_chat_google_palm() -> None:
    """Test Google PaLM Chat API wrapper."""
    chat = ChatGooglePalm()  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_chat_google_palm_system_message() -> None:
    """Test Google PaLM Chat API wrapper with system message."""
    chat = ChatGooglePalm()  # type: ignore[call-arg]
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_chat_google_palm_generate() -> None:
    """Test Google PaLM Chat API wrapper with generate."""
    chat = ChatGooglePalm(n=2, temperature=1.0)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

def test_chat_google_palm_multiple_completions() -> None:
    """Test Google PaLM Chat API wrapper with multiple completions."""
    # The API de-dupes duplicate responses, so set temperature higher. This
    # could be a flakey test though...
    chat = ChatGooglePalm(n=5, temperature=1.0)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)

async def test_async_chat_google_palm() -> None:
    """Test async generation."""
    chat = ChatGooglePalm(n=2, temperature=1.0)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


# --- libs/community/tests/integration_tests/chat_models/test_gpt_router.py ---

def test_gpt_router_call() -> None:
    """Test valid call to GPTRouter."""
    anthropic_claude = GPTRouterModel(
        name="claude-instant-1.2", provider_name="anthropic"
    )
    chat = GPTRouter(models_priority_list=[anthropic_claude])
    message = HumanMessage(content="Hello World")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_gpt_router_call_incorrect_model() -> None:
    """Test invalid modelName"""
    anthropic_claude = GPTRouterModel(
        name="model_does_not_exist", provider_name="anthropic"
    )
    chat = GPTRouter(models_priority_list=[anthropic_claude])
    message = HumanMessage(content="Hello World")
    with pytest.raises(Exception):
        chat.invoke([message])

def test_gpt_router_generate() -> None:
    """Test generate method of GPTRouter."""
    anthropic_claude = GPTRouterModel(
        name="claude-instant-1.2", provider_name="anthropic"
    )
    chat = GPTRouter(models_priority_list=[anthropic_claude])
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="If (5 + x = 18), what is x?")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy

def test_gpt_router_streaming() -> None:
    """Test streaming tokens from GPTRouter."""
    anthropic_claude = GPTRouterModel(
        name="claude-instant-1.2", provider_name="anthropic"
    )
    chat = GPTRouter(models_priority_list=[anthropic_claude], streaming=True)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


# --- libs/community/tests/integration_tests/chat_models/test_hunyuan.py ---

def test_chat_hunyuan() -> None:
    chat = ChatHunyuan()
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.id is not None, "request_id is empty"
    assert uuid.UUID(response.id), "Invalid UUID"

def test_chat_hunyuan_with_temperature() -> None:
    chat = ChatHunyuan(temperature=0.6)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.id is not None, "request_id is empty"
    assert uuid.UUID(response.id), "Invalid UUID"

def test_chat_hunyuan_with_model_name() -> None:
    chat = ChatHunyuan(model="hunyuan-standard")
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.id is not None, "request_id is empty"
    assert uuid.UUID(response.id), "Invalid UUID"

def test_chat_hunyuan_with_stream() -> None:
    chat = ChatHunyuan(streaming=True)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.id is not None, "request_id is empty"
    assert uuid.UUID(response.id), "Invalid UUID"

def test_chat_hunyuan_with_prompt_template() -> None:
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant! Your name is {name}."
    )
    user_prompt = HumanMessagePromptTemplate.from_template("Question: {query}")
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    chat: RunnableSerializable[Any, Any] = (
        {"query": itemgetter("query"), "name": itemgetter("name")}
        | chat_prompt
        | ChatHunyuan()
    )
    response = chat.invoke({"query": "Hello", "name": "Tom"})
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.id is not None, "request_id is empty"
    assert uuid.UUID(response.id), "Invalid UUID"


# --- libs/community/tests/integration_tests/chat_models/test_jinachat.py ---

def test_jinachat_api_key_is_secret_string() -> None:
    llm = JinaChat(jinachat_api_key="secret-api-key")  # type: ignore[arg-type]
    assert isinstance(llm.jinachat_api_key, SecretStr)

def test_uses_actual_secret_value_from_secretstr() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    llm = JinaChat(jinachat_api_key="secret-api-key")  # type: ignore[arg-type]
    assert cast(SecretStr, llm.jinachat_api_key).get_secret_value() == "secret-api-key"

def test_jinachat() -> None:
    """Test JinaChat wrapper."""
    chat = JinaChat(max_tokens=10)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_jinachat_system_message() -> None:
    """Test JinaChat wrapper with system message."""
    chat = JinaChat(max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_jinachat_generate() -> None:
    """Test JinaChat wrapper with generate."""
    chat = JinaChat(max_tokens=10)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

async def test_async_jinachat() -> None:
    """Test async generation."""
    chat = JinaChat(max_tokens=102)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

def test_jinachat_extra_kwargs() -> None:
    """Test extra kwargs to chat openai."""
    # Check that foo is saved in extra_kwargs.
    llm = JinaChat(foo=3, max_tokens=10)  # type: ignore[call-arg]
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = JinaChat(foo=3, model_kwargs={"bar": 2})  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        JinaChat(foo=3, model_kwargs={"foo": 2})  # type: ignore[call-arg]

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        JinaChat(model_kwargs={"temperature": 0.2})


# --- libs/community/tests/integration_tests/chat_models/test_kinetica.py ---

    def test_generate(self) -> None:
        """Generate SQL from a chain."""
        kinetica_llm = ChatKinetica()  # type: ignore[call-arg]

        # create chain
        ctx_messages = kinetica_llm.load_messages_from_context(self.context_name)
        ctx_messages.append(("human", "{input}"))
        prompt_template = ChatPromptTemplate.from_messages(ctx_messages)
        chain = prompt_template | kinetica_llm

        resp_message = chain.invoke(
            {"input": "What are the female users ordered by username?"}
        )
        LOG.info(f"SQL Response: {resp_message.content}")
        assert isinstance(resp_message, AIMessage)

    def test_full_chain(self) -> None:
        """Generate SQL from a chain and execute the query."""
        kinetica_llm = ChatKinetica()  # type: ignore[call-arg]

        # create chain
        ctx_messages = kinetica_llm.load_messages_from_context(self.context_name)
        ctx_messages.append(("human", "{input}"))
        prompt_template = ChatPromptTemplate.from_messages(ctx_messages)
        chain = (
            prompt_template
            | kinetica_llm
            | KineticaSqlOutputParser(kdbc=kinetica_llm.kdbc)
        )
        sql_response: KineticaSqlResponse = chain.invoke(
            {"input": "What are the female users ordered by username?"}
        )

        assert isinstance(sql_response, KineticaSqlResponse)
        LOG.info(f"SQL Response: {sql_response.sql}")
        assert isinstance(sql_response.dataframe, pd.DataFrame)
        users = sql_response.dataframe["username"]
        assert users[0] == "alexander40"


# --- libs/community/tests/integration_tests/chat_models/test_konko.py ---

def test_konko_chat_test() -> None:
    """Evaluate basic ChatKonko functionality."""
    chat_instance = ChatKonko(max_tokens=10)
    msg = HumanMessage(content="Hi")
    chat_response = chat_instance.invoke([msg])
    assert isinstance(chat_response, BaseMessage)
    assert isinstance(chat_response.content, str)

def test_konko_chat_test_openai() -> None:
    """Evaluate basic ChatKonko functionality."""
    chat_instance = ChatKonko(max_tokens=10, model="meta-llama/llama-2-70b-chat")
    msg = HumanMessage(content="Hi")
    chat_response = chat_instance.invoke([msg])
    assert isinstance(chat_response, BaseMessage)
    assert isinstance(chat_response.content, str)

def test_konko_system_msg_test() -> None:
    """Evaluate ChatKonko's handling of system messages."""
    chat_instance = ChatKonko(max_tokens=10)
    sys_msg = SystemMessage(content="Initiate user chat.")
    user_msg = HumanMessage(content="Hi there")
    chat_response = chat_instance.invoke([sys_msg, user_msg])
    assert isinstance(chat_response, BaseMessage)
    assert isinstance(chat_response.content, str)

def test_konko_generation_test() -> None:
    """Check ChatKonko's generation ability."""
    chat_instance = ChatKonko(max_tokens=10, n=2)
    msg = HumanMessage(content="Hi")
    gen_response = chat_instance.generate([[msg], [msg]])
    assert isinstance(gen_response, LLMResult)
    assert len(gen_response.generations) == 2
    for gen_list in gen_response.generations:
        assert len(gen_list) == 2
        for gen in gen_list:
            assert isinstance(gen, ChatGeneration)
            assert isinstance(gen.text, str)
            assert gen.text == gen.message.content

def test_konko_multiple_outputs_test() -> None:
    """Test multiple completions with ChatKonko."""
    chat_instance = ChatKonko(max_tokens=10, n=5)
    msg = HumanMessage(content="Hi")
    gen_response = chat_instance._generate([msg])
    assert isinstance(gen_response, ChatResult)
    assert len(gen_response.generations) == 5
    for gen in gen_response.generations:
        assert isinstance(gen.message, BaseMessage)
        assert isinstance(gen.message.content, str)

def test_konko_llm_model_name_test() -> None:
    """Check if llm_output has model info."""
    chat_instance = ChatKonko(max_tokens=10)
    msg = HumanMessage(content="Hi")
    llm_data = chat_instance.generate([[msg]])
    assert llm_data.llm_output is not None
    assert llm_data.llm_output["model_name"] == chat_instance.model

def test_konko_streaming_model_name_test() -> None:
    """Check model info during streaming."""
    chat_instance = ChatKonko(max_tokens=10, streaming=True)
    msg = HumanMessage(content="Hi")
    llm_data = chat_instance.generate([[msg]])
    assert llm_data.llm_output is not None
    assert llm_data.llm_output["model_name"] == chat_instance.model

def test_konko_streaming_param_validation_test() -> None:
    """Ensure correct token callback during streaming."""
    with pytest.raises(ValueError):
        ChatKonko(
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )

def test_konko_token_streaming_test() -> None:
    """Check token streaming for ChatKonko."""
    chat_instance = ChatKonko(max_tokens=10)

    for token in chat_instance.stream("Just a test"):
        assert isinstance(token.content, str)


# --- libs/community/tests/integration_tests/chat_models/test_llamacpp.py ---

def test_structured_output() -> None:
    llm = ChatLlamaCpp(model_path="/path/to/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf")
    structured_llm = llm.with_structured_output(Joke)
    result = structured_llm.invoke("Tell me a short joke about cats.")
    assert isinstance(result, Joke)


# --- libs/community/tests/integration_tests/chat_models/test_llama_edge.py ---

def test_chat_wasm_service() -> None:
    """This test requires the port 8080 is not occupied."""

    # service url
    service_url = "https://b008-54-186-154-209.ngrok-free.app"

    # create wasm-chat service instance
    chat = LlamaEdgeChatService(service_url=service_url)

    # create message sequence
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]

    # chat with wasm-chat service
    response = chat.invoke(messages)

    # check response
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "Paris" in response.content

def test_chat_wasm_service_streaming() -> None:
    """This test requires the port 8080 is not occupied."""

    # service url
    service_url = "https://b008-54-186-154-209.ngrok-free.app"

    # create wasm-chat service instance
    chat = LlamaEdgeChatService(service_url=service_url, streaming=True)

    # create message sequence
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [
        user_message,
    ]

    output = ""
    for chunk in chat.stream(messages):
        print(chunk.content, end="", flush=True)  # noqa: T201
        output += chunk.content  # type: ignore[operator]

    assert "Paris" in output


# --- libs/community/tests/integration_tests/chat_models/test_minimax.py ---

def test_chat_minimax_not_group_id() -> None:
    if "MINIMAX_GROUP_ID" in os.environ:
        del os.environ["MINIMAX_GROUP_ID"]
    chat = MiniMaxChat()  # type: ignore[call-arg]
    response = chat.invoke("你好呀")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_minimax_with_stream() -> None:
    chat = MiniMaxChat()  # type: ignore[call-arg]
    for chunk in chat.stream("你好呀"):
        assert isinstance(chunk, AIMessage)
        assert isinstance(chunk.content, str)

def test_chat_minimax_with_tool() -> None:
    """Test MinimaxChat with bind tools."""
    chat = MiniMaxChat()  # type: ignore[call-arg]
    tools = [add, multiply]
    chat_with_tools = chat.bind_tools(tools)

    query = "What is 3 * 12?"
    messages = [HumanMessage(query)]
    ai_msg = chat_with_tools.invoke(messages)
    assert isinstance(ai_msg, AIMessage)
    assert isinstance(ai_msg.tool_calls, list)
    assert len(ai_msg.tool_calls) == 1
    tool_call = ai_msg.tool_calls[0]
    assert "args" in tool_call
    messages.append(ai_msg)  # type: ignore[arg-type]
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))  # type: ignore[arg-type]
    response = chat_with_tools.invoke(messages)
    assert isinstance(response, AIMessage)

def test_chat_minimax_with_structured_output() -> None:
    """Test MiniMaxChat with structured output."""
    llm = MiniMaxChat()  # type: ignore[call-arg]
    structured_llm = llm.with_structured_output(AnswerWithJustification)
    response = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    assert isinstance(response, AnswerWithJustification)

def test_chat_tongyi_with_structured_output_include_raw() -> None:
    """Test MiniMaxChat with structured output."""
    llm = MiniMaxChat()  # type: ignore[call-arg]
    structured_llm = llm.with_structured_output(
        AnswerWithJustification, include_raw=True
    )
    response = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    assert isinstance(response, dict)
    assert isinstance(response.get("raw"), AIMessage)
    assert isinstance(response.get("parsed"), AnswerWithJustification)


# --- libs/community/tests/integration_tests/chat_models/test_naver.py ---

def test_stream() -> None:
    """Test streaming tokens from ChatClovaX."""
    llm = ChatClovaX(include_ai_filters=True)

    for token in llm.stream("I'm Clova"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        if token.response_metadata:
            assert "input_length" in token.response_metadata
            assert "output_length" in token.response_metadata
            assert "stop_reason" in token.response_metadata
            assert "ai_filter" in token.response_metadata

async def test_astream() -> None:
    """Test streaming tokens from ChatClovaX."""
    llm = ChatClovaX(include_ai_filters=True)

    async for token in llm.astream("I'm Clova"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        if token.response_metadata:
            assert "input_length" in token.response_metadata
            assert "output_length" in token.response_metadata
            assert "stop_reason" in token.response_metadata
            assert "ai_filter" in token.response_metadata

async def test_ainvoke() -> None:
    """Test invoke tokens from ChatClovaX."""
    llm = ChatClovaX(include_ai_filters=True)

    result = await llm.ainvoke("I'm Clova", config={"tags": ["foo"]})
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    if result.response_metadata:
        assert "input_length" in result.response_metadata
        assert "output_length" in result.response_metadata
        assert "stop_reason" in result.response_metadata
        assert "ai_filter" in result.response_metadata

def test_invoke() -> None:
    """Test invoke tokens from ChatClovaX."""
    llm = ChatClovaX(include_ai_filters=True)

    result = llm.invoke("I'm Clova", config=dict(tags=["foo"]))
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    if result.response_metadata:
        assert "input_length" in result.response_metadata
        assert "output_length" in result.response_metadata
        assert "stop_reason" in result.response_metadata
        assert "ai_filter" in result.response_metadata

def test_stream_error_event() -> None:
    """Test streaming error event from ChatClovaX."""
    llm = ChatClovaX()
    prompt = "What is the best way to reduce my carbon footprint?"

    with pytest.raises(SSEError):
        for _ in llm.stream(prompt * 1000):
            pass

async def test_astream_error_event() -> None:
    """Test streaming error event from ChatClovaX."""
    llm = ChatClovaX()
    prompt = "What is the best way to reduce my carbon footprint?"

    with pytest.raises(SSEError):
        async for _ in llm.astream(prompt * 1000):
            pass


# --- libs/community/tests/integration_tests/chat_models/test_octoai.py ---

def test_chat_octoai() -> None:
    chat = ChatOctoAI()
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


# --- libs/community/tests/integration_tests/chat_models/test_outlines.py ---

def test_chat_outlines_inference(chat_model: ChatOutlines) -> None:
    """Test valid ChatOutlines inference."""
    messages = [HumanMessage(content="Say foo:")]
    output = chat_model.invoke(messages)
    assert isinstance(output, AIMessage)
    assert len(output.content) > 1

def test_chat_outlines_streaming(chat_model: ChatOutlines) -> None:
    """Test streaming tokens from ChatOutlines."""
    messages = [HumanMessage(content="How do you say 'hello' in Spanish?")]
    generator = chat_model.stream(messages)
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        assert isinstance(chunk, BaseMessageChunk)
        if isinstance(chunk.content, str):
            stream_results_string += chunk.content
        else:
            raise ValueError(
                f"Invalid content type, only str is supported, "
                f"got {type(chunk.content)}"
            )
    assert len(stream_results_string.strip()) > 1

def test_chat_outlines_regex(chat_model: ChatOutlines) -> None:
    """Test regex for generating a valid IP address"""
    ip_regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    chat_model.regex = ip_regex
    assert chat_model.regex == ip_regex

    messages = [HumanMessage(content="What is the IP address of Google's DNS server?")]
    output = chat_model.invoke(messages)

    assert isinstance(output, AIMessage)
    assert re.match(ip_regex, str(output.content)), (
        f"Generated output '{output.content}' is not a valid IP address"
    )

def test_chat_outlines_type_constraints(chat_model: ChatOutlines) -> None:
    """Test type constraints for generating an integer"""
    chat_model.type_constraints = int
    messages = [
        HumanMessage(
            content="What is the answer to life, the universe, and everything?"
        )
    ]
    output = chat_model.invoke(messages)
    assert isinstance(int(str(output.content)), int)

def test_chat_outlines_json(chat_model: ChatOutlines) -> None:
    """Test json for generating a valid JSON object"""

    class Person(BaseModel):
        name: str

    chat_model.json_schema = Person
    messages = [HumanMessage(content="Who are the main contributors to LangChain?")]
    output = chat_model.invoke(messages)
    person = Person.model_validate_json(str(output.content))
    assert isinstance(person, Person)

def test_chat_outlines_grammar(chat_model: ChatOutlines) -> None:
    """Test grammar for generating a valid arithmetic expression"""
    if chat_model.backend == "mlxlm":
        pytest.skip("MLX grammars not yet supported.")

    chat_model.grammar = """
        ?start: expression
        ?expression: term (("+" | "-") term)*
        ?term: factor (("*" | "/") factor)*
        ?factor: NUMBER | "-" factor | "(" expression ")"
        %import common.NUMBER
        %import common.WS
        %ignore WS
    """

    messages = [HumanMessage(content="Give me a complex arithmetic expression:")]
    output = chat_model.invoke(messages)

    # Validate the output is a non-empty string
    assert isinstance(output.content, str) and output.content.strip(), (
        "Output should be a non-empty string"
    )

    # Use a simple regex to check if the output contains basic arithmetic operations and numbers
    assert re.search(r"[\d\+\-\*/\(\)]+", output.content), (
        f"Generated output '{output.content}' does not appear to be a valid arithmetic expression"
    )

def test_chat_outlines_with_structured_output(chat_model: ChatOutlines) -> None:
    """Test that ChatOutlines can generate structured outputs"""

    class AnswerWithJustification(BaseModel):
        """An answer to the user question along with justification for the answer."""

        answer: str
        justification: str

    structured_chat_model = chat_model.with_structured_output(AnswerWithJustification)

    result = structured_chat_model.invoke(
        "What weighs more, a pound of bricks or a pound of feathers?"
    )

    assert isinstance(result, AnswerWithJustification)
    assert isinstance(result.answer, str)
    assert isinstance(result.justification, str)
    assert len(result.answer) > 0
    assert len(result.justification) > 0

    structured_chat_model_with_raw = chat_model.with_structured_output(
        AnswerWithJustification, include_raw=True
    )

    result_with_raw = structured_chat_model_with_raw.invoke(
        "What weighs more, a pound of bricks or a pound of feathers?"
    )

    assert isinstance(result_with_raw, dict)
    assert "raw" in result_with_raw
    assert "parsed" in result_with_raw
    assert "parsing_error" in result_with_raw
    assert isinstance(result_with_raw["raw"], BaseMessage)
    assert isinstance(result_with_raw["parsed"], AnswerWithJustification)
    assert result_with_raw["parsing_error"] is None


# --- libs/community/tests/integration_tests/chat_models/test_pai_eas_chat_endpoint.py ---

def test_pai_eas_call() -> None:
    chat = PaiEasChatEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),  # type: ignore[arg-type]
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),  # type: ignore[arg-type]
    )
    response = chat.invoke([HumanMessage(content="Say foo:")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = PaiEasChatEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),  # type: ignore[arg-type]
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),  # type: ignore[arg-type]
    )

    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = PaiEasChatEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),  # type: ignore[arg-type]
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),  # type: ignore[arg-type]
    )
    message = HumanMessage(content="Hi, how are you.")
    response = chat.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


# --- libs/community/tests/integration_tests/chat_models/test_premai.py ---

def test_chat_premai() -> None:
    """Test ChatPremAI wrapper."""
    chat = ChatPremAI(project_id=8)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_chat_prem_system_message() -> None:
    """Test ChatPremAI wrapper for system message"""
    chat = ChatPremAI(project_id=8)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_chat_prem_generate() -> None:
    """Test ChatPremAI wrapper with generate."""
    chat = ChatPremAI(project_id=8)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

async def test_prem_invoke(chat: ChatPremAI) -> None:
    """Tests chat completion with invoke"""
    result = chat.invoke("How is the weather in New York today?")
    assert isinstance(result.content, str)

def test_prem_streaming() -> None:
    """Test streaming tokens from Prem."""
    chat = ChatPremAI(project_id=8, streaming=True)

    for token in chat.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


# --- libs/community/tests/integration_tests/chat_models/test_promptlayer_openai.py ---

def test_promptlayer_chat_openai() -> None:
    """Test PromptLayerChatOpenAI wrapper."""
    chat = PromptLayerChatOpenAI(max_tokens=10)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_promptlayer_chat_openai_system_message() -> None:
    """Test PromptLayerChatOpenAI wrapper with system message."""
    chat = PromptLayerChatOpenAI(max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_promptlayer_chat_openai_generate() -> None:
    """Test PromptLayerChatOpenAI wrapper with generate."""
    chat = PromptLayerChatOpenAI(max_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

def test_promptlayer_chat_openai_multiple_completions() -> None:
    """Test PromptLayerChatOpenAI wrapper with multiple completions."""
    chat = PromptLayerChatOpenAI(max_tokens=10, n=5)
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)

def test_promptlayer_chat_openai_invalid_streaming_params() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with pytest.raises(ValueError):
        PromptLayerChatOpenAI(
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )

async def test_async_promptlayer_chat_openai() -> None:
    """Test async generation."""
    chat = PromptLayerChatOpenAI(max_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


# --- libs/community/tests/integration_tests/chat_models/test_qianfan_endpoint.py ---

def test_default_call() -> None:
    """Test default model.invoke(`ERNIE-Bot`) call."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_model() -> None:
    """Test model kwarg works."""
    chat = QianfanChatEndpoint(model="BLOOMZ-7B")  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_model_param() -> None:
    """Test model params works."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")], model="BLOOMZ-7B")
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_endpoint() -> None:
    """Test user custom model deployments like some open source models."""
    chat = QianfanChatEndpoint(endpoint="qianfan_bloomz_7b_compressed")  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_endpoint_param() -> None:
    """Test user custom model deployments like some open source models."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    response = chat.invoke(
        [HumanMessage(endpoint="qianfan_bloomz_7b_compressed", content="Hello")]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]

    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_chat_generate() -> None:
    """Tests chat generate works."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    response = chat.generate(
        [
            [
                HumanMessage(content="Hello."),
                AIMessage(content="Hello!"),
                HumanMessage(content="How are you doing?"),
            ]
        ]
    )
    assert isinstance(response, LLMResult)
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)

async def test_async_invoke() -> None:
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    res = await chat.ainvoke([HumanMessage(content="Hello")])
    assert isinstance(res, BaseMessage)
    assert res.content != ""

async def test_async_generate() -> None:
    """Tests chat agenerate works."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    response = await chat.agenerate(
        [
            [
                HumanMessage(content="Hello."),
                AIMessage(content="Hello!"),
                HumanMessage(content="How are you doing?"),
            ]
        ]
    )
    assert isinstance(response, LLMResult)
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)

async def test_async_stream() -> None:
    chat = QianfanChatEndpoint(streaming=True)  # type: ignore[call-arg]
    async for token in chat.astream(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="Who are you?"),
        ]
    ):
        assert isinstance(token, BaseMessageChunk)

def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    message = HumanMessage(content="Hi, how are you.")
    response = chat.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

def test_functions_call() -> None:
    chat = QianfanChatEndpoint(model="ERNIE-Bot")  # type: ignore[call-arg]

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessage(content="What's the temperature in Shanghai today?"),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "get_current_temperature",
                        "thoughts": "i will use get_current_temperature "
                        "to resolve the questions",
                        "arguments": '{"location":"Shanghai","unit":"centigrade"}',
                    }
                },
            ),
            FunctionMessage(
                name="get_current_weather",
                content='{"temperature": "25", \
                                "unit": "摄氏度", "description": "晴朗"}',
            ),
        ]
    )
    chain = prompt | chat.bind(functions=_FUNCTIONS)
    resp = chain.invoke({})
    assert isinstance(resp, AIMessage)


# --- libs/community/tests/integration_tests/chat_models/test_reka.py ---

def test_reka_call() -> None:
    """Test a simple call to Reka."""
    chat = ChatReka(model="reka-flash", verbose=True)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    logger.debug(f"Response content: {response.content}")

def test_reka_generate() -> None:
    """Test the generate method of Reka."""
    chat = ChatReka(model="reka-flash", verbose=True)
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="How many toes do dogs have?")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
        logger.debug(f"Generated response: {response.text}")
    assert chat_messages == messages_copy

def test_reka_streaming() -> None:
    """Test streaming tokens from Reka."""
    chat = ChatReka(model="reka-flash", streaming=True, verbose=True)
    message = HumanMessage(content="Tell me a story.")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    logger.debug(f"Streaming response content: {response.content}")

def test_reka_tool_usage_integration() -> None:
    """Test tool usage with Reka API integration."""
    # Initialize the ChatReka model with tools and verbose logging
    chat_reka = ChatReka(model="reka-flash", verbose=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_product_availability",
                "description": (
                    "Determine whether a product is currently in stock given "
                    "a product ID."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": (
                                "The unique product ID to check availability for"
                            ),
                        },
                    },
                    "required": ["product_id"],
                },
            },
        },
    ]
    chat_reka_with_tools = chat_reka.bind_tools(tools)

    # Start a conversation
    messages: List[BaseMessage] = [
        HumanMessage(content="Is product A12345 in stock right now?")
    ]

    # Get the initial response
    response = chat_reka_with_tools.invoke(messages)
    assert isinstance(response, AIMessage)
    logger.debug(f"Initial AI message: {response.content}")

    # Check if the model wants to use a tool
    if "tool_calls" in response.additional_kwargs:
        tool_calls = response.additional_kwargs["tool_calls"]
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            logger.debug(
                f"Tool call requested: {function_name} with arguments {arguments}"
            )

            # Simulate executing the tool
            tool_output = "AVAILABLE"

            tool_message = ToolMessage(
                content=tool_output, tool_call_id=tool_call["id"]
            )
            messages.append(response)
            messages.append(tool_message)

            final_response = chat_reka_with_tools.invoke(messages)
            assert isinstance(final_response, AIMessage)
            logger.debug(f"Final AI message: {final_response.content}")

            # Assert that the response message is non-empty
            assert final_response.content, "The final response content is empty."
    else:
        pytest.fail("The model did not request a tool.")

def test_reka_system_message() -> None:
    """Test Reka with system message."""
    chat = ChatReka(model="reka-flash", verbose=True)
    messages = [
        SystemMessage(content="You are a helpful AI that speaks like Shakespeare."),
        HumanMessage(content="Tell me about the weather today."),
    ]
    response = chat.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    logger.debug(f"Response with system message: {response.content}")

def test_reka_system_message_multi_turn() -> None:
    """Test multi-turn conversation with system message."""
    chat = ChatReka(model="reka-flash", verbose=True)
    messages = [
        SystemMessage(content="You are a math tutor who explains concepts simply."),
        HumanMessage(content="What is a prime number?"),
    ]

    # First turn
    response1 = chat.invoke(messages)
    assert isinstance(response1, AIMessage)
    messages.append(response1)

    # Second turn
    messages.append(HumanMessage(content="Can you give me an example?"))
    response2 = chat.invoke(messages)
    assert isinstance(response2, AIMessage)

    logger.debug(f"First response: {response1.content}")
    logger.debug(f"Second response: {response2.content}")


# --- libs/community/tests/integration_tests/chat_models/test_snowflake.py ---

def test_chat_snowflake_cortex(chat: ChatSnowflakeCortex) -> None:
    """Test ChatSnowflakeCortex."""
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_chat_snowflake_cortex_system_message(chat: ChatSnowflakeCortex) -> None:
    """Test ChatSnowflakeCortex for system message"""
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_chat_snowflake_cortex_generate(chat: ChatSnowflakeCortex) -> None:
    """Test ChatSnowflakeCortex with generate."""
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

def test_chat_snowflake_cortex_message_with_special_characters(
    chat: ChatSnowflakeCortex,
) -> None:
    """Test ChatSnowflakeCortex with messages containing special characters.

    Args:
        chat: The ChatSnowflakeCortex instance to test with.
    """
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Can you give me the weather in Tokyo?\n\n")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


# --- libs/community/tests/integration_tests/chat_models/test_sparkllm.py ---

def test_chat_spark_llm() -> None:
    chat = ChatSparkLLM()  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_spark_llm_streaming() -> None:
    chat = ChatSparkLLM(streaming=True)  # type: ignore[call-arg]
    for chunk in chat.stream("Hello!"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, str)

def test_chat_spark_llm_with_domain() -> None:
    chat = ChatSparkLLM(spark_llm_domain="generalv3")  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    print(response)  # noqa: T201
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_spark_llm_with_temperature() -> None:
    chat = ChatSparkLLM(temperature=0.9, top_k=2)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    print(response)  # noqa: T201
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_chat_spark_llm_streaming_with_stream_method() -> None:
    chat = ChatSparkLLM()  # type: ignore[call-arg]
    for chunk in chat.stream("Hello!"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, str)


# --- libs/community/tests/integration_tests/chat_models/test_tongyi.py ---

def test_default_call() -> None:
    """Test default model call."""
    chat = ChatTongyi()  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_model() -> None:
    """Test model kwarg works."""
    chat = ChatTongyi(model="qwen-plus")  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_vision_model() -> None:
    """Test model kwarg works."""
    chat = ChatTongyi(model="qwen-vl-max")  # type: ignore[call-arg]
    response = chat.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "image": "https://python.langchain.com/v0.1/assets/images/run_details-806f6581cd382d4887a5bc3e8ac62569.png"
                    },
                    {"text": "Summarize the image"},
                ]
            )
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, list)

def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = ChatTongyi()  # type: ignore[call-arg]

    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = ChatTongyi()  # type: ignore[call-arg]
    message = HumanMessage(content="Hi, how are you.")
    response = chat.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

def test_tool_use() -> None:
    llm = ChatTongyi(model="qwen-turbo", temperature=0)  # type: ignore[call-arg]
    llm_with_tool = llm.bind_tools(tools=[GenerateUsername])
    msgs: List = [
        HumanMessage(content="Sally has green hair, what would her username be?")
    ]
    ai_msg = llm_with_tool.invoke(msgs)
    # assert ai_msg is None
    # ai_msg.content = " "

    assert isinstance(ai_msg, AIMessage)
    assert isinstance(ai_msg.tool_calls, list)
    assert len(ai_msg.tool_calls) == 1
    tool_call = ai_msg.tool_calls[0]
    assert "args" in tool_call

    tool_msg = ToolMessage(
        content="sally_green_hair",
        tool_call_id=ai_msg.tool_calls[0]["id"],
        name=ai_msg.tool_calls[0]["name"],
    )
    msgs.extend([ai_msg, tool_msg])
    llm_with_tool.invoke(msgs)

    # Test streaming
    ai_messages = llm_with_tool.stream(msgs)
    first = True
    for message in ai_messages:
        if first:
            gathered = message
            first = False
        else:
            gathered = gathered + message  # type: ignore[assignment]
    assert isinstance(gathered, AIMessageChunk)

    streaming_tool_msg = ToolMessage(
        content="sally_green_hair",
        name=tool_call["name"],
        tool_call_id=tool_call["id"] if tool_call["id"] else " ",
    )
    msgs.extend([gathered, streaming_tool_msg])
    llm_with_tool.invoke(msgs)

def test_manual_tool_call_msg() -> None:
    """Test passing in manually construct tool call message."""
    llm = ChatTongyi(model="qwen-turbo", temperature=0)  # type: ignore[call-arg]
    llm_with_tool = llm.bind_tools(tools=[GenerateUsername])
    msgs: List = [
        HumanMessage(content="Sally has green hair, what would her username be?"),
        AIMessage(
            content=" ",
            tool_calls=[
                ToolCall(
                    name="GenerateUsername",
                    args={"name": "Sally", "hair_color": "green"},
                    id="foo",
                )
            ],
        ),
        ToolMessage(content="sally_green_hair", tool_call_id="foo"),
    ]
    output: AIMessage = cast(AIMessage, llm_with_tool.invoke(msgs))
    assert output.content
    # Should not have called the tool again.
    assert not output.tool_calls and not output.invalid_tool_calls

def test_chat_tongyi_with_structured_output() -> None:
    """Test ChatTongyi with structured output."""
    llm = ChatTongyi()  # type: ignore[call-arg]
    structured_llm = llm.with_structured_output(AnswerWithJustification)
    response = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    assert isinstance(response, AnswerWithJustification)

def test_chat_tongyi_with_structured_output_include_raw() -> None:
    """Test ChatTongyi with structured output."""
    llm = ChatTongyi()  # type: ignore[call-arg]
    structured_llm = llm.with_structured_output(
        AnswerWithJustification, include_raw=True
    )
    response = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    assert isinstance(response, dict)
    assert isinstance(response.get("raw"), AIMessage)
    assert isinstance(response.get("parsed"), AnswerWithJustification)


# --- libs/community/tests/integration_tests/chat_models/test_volcengine_maas.py ---

def test_default_call() -> None:
    """Test valid chat call to volc engine."""
    chat = VolcEngineMaasChat()
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = VolcEngineMaasChat()

    response = chat.invoke(
        [
            HumanMessage(content="Hello"),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = VolcEngineMaasChat()
    message = HumanMessage(content="Hi, how are you?")
    response = chat.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


# --- libs/community/tests/integration_tests/chat_models/test_yuan2.py ---

def test_chat_yuan2() -> None:
    """Test ChatYuan2 wrapper."""
    chat = ChatYuan2(  # type: ignore[call-arg]
        yuan2_api_key="EMPTY",
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
        streaming=False,
    )
    messages = [
        HumanMessage(content="Hello"),
    ]
    response = chat.invoke(messages)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_chat_yuan2_system_message() -> None:
    """Test ChatYuan2 wrapper with system message."""
    chat = ChatYuan2(  # type: ignore[call-arg]
        yuan2_api_key="EMPTY",
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
        streaming=False,
    )
    messages = [
        SystemMessage(content="You are an AI assistant."),
        HumanMessage(content="Hello"),
    ]
    response = chat.invoke(messages)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_chat_yuan2_generate() -> None:
    """Test ChatYuan2 wrapper with generate."""
    chat = ChatYuan2(  # type: ignore[call-arg]
        yuan2_api_key="EMPTY",
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
        streaming=False,
    )
    messages: List = [
        HumanMessage(content="Hello"),
    ]
    response = chat.generate([messages])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    assert response.llm_output
    generation = response.generations[0]
    for gen in generation:
        assert isinstance(gen, ChatGeneration)
        assert isinstance(gen.text, str)
        assert gen.text == gen.message.content

async def test_async_chat_yuan2() -> None:
    """Test async generation."""
    chat = ChatYuan2(  # type: ignore[call-arg]
        yuan2_api_key="EMPTY",
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
        streaming=False,
    )
    messages: List = [
        HumanMessage(content="Hello"),
    ]
    response = await chat.agenerate([messages])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    generations = response.generations[0]
    for generation in generations:
        assert isinstance(generation, ChatGeneration)
        assert isinstance(generation.text, str)
        assert generation.text == generation.message.content


# --- libs/community/tests/integration_tests/chat_models/test_zhipuai.py ---

def test_default_call() -> None:
    """Test default model call."""
    chat = ChatZhipuAI()
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_model() -> None:
    """Test model kwarg works."""
    chat = ChatZhipuAI(model="glm-4")
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = ChatZhipuAI()

    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = ChatZhipuAI()
    message = HumanMessage(content="Hi, how are you.")
    response = chat.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content

def test_tool_call() -> None:
    """Test tool calling by ChatZhipuAI"""
    chat = ChatZhipuAI(model="glm-4-long")
    tools = [add, multiply]
    chat_with_tools = chat.bind_tools(tools)

    query = "What is 3 * 12?"
    messages = [HumanMessage(query)]
    ai_msg = chat_with_tools.invoke(messages)
    assert isinstance(ai_msg, AIMessage)
    assert isinstance(ai_msg.tool_calls, list)
    assert len(ai_msg.tool_calls) == 1
    tool_call = ai_msg.tool_calls[0]
    assert "args" in tool_call
    messages.append(ai_msg)  # type: ignore[arg-type]
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))  # type: ignore[arg-type]
    response = chat_with_tools.invoke(messages)
    assert isinstance(response, AIMessage)


# --- libs/community/tests/integration_tests/document_loaders/test_blockchain.py ---

def test_get_all_10sec_timeout() -> None:
    start_time = time.time()

    contract_address = (
        "0x1a92f7381b9f03921564a437210bb9396471050c"  # Cool Cats contract address
    )

    with pytest.raises(RuntimeError):
        BlockchainDocumentLoader(
            contract_address=contract_address,
            blockchainType=BlockchainType.ETH_MAINNET,
            api_key=os.environ["ALCHEMY_API_KEY"],
            get_all_tokens=True,
            max_execution_time=10,
        ).load()

    end_time = time.time()

    print("Execution took ", end_time - start_time, " seconds")  # noqa: T201


# --- libs/community/tests/integration_tests/document_loaders/test_pdf.py ---

def test_amazontextract_loader(
    file_path: str,
    features: Union[Sequence[str], None],
    docs_length: int,
    create_client: bool,
) -> None:
    if create_client:
        import boto3

        textract_client = boto3.client("textract", region_name="us-east-2")
        loader = AmazonTextractPDFLoader(
            file_path, textract_features=features, client=textract_client
        )
    else:
        loader = AmazonTextractPDFLoader(file_path, textract_features=features)
    docs = loader.load()
    print(docs)  # noqa: T201

    assert len(docs) == docs_length

def test_amazontextract_loader_failures() -> None:
    # 2-page PDF local file system
    two_page_pdf = (
        Path(__file__).parent.parent / "examples/multi-page-forms-sample-2-page.pdf"
    )
    loader = AmazonTextractPDFLoader(two_page_pdf)
    with pytest.raises(ValueError):
        loader.load()


# --- libs/community/tests/integration_tests/embeddings/test_cloudflare_workersai.py ---

def test_cloudflare_workers_ai_embedding_documents() -> None:
    """Test Cloudflare Workers AI embeddings."""
    documents = ["foo bar", "foo bar", "foo bar"]

    responses.add(
        responses.POST,
        "https://api.cloudflare.com/client/v4/accounts/123/ai/run/@cf/baai/bge-base-en-v1.5",
        json={
            "result": {
                "shape": [3, 768],
                "data": [[0.0] * 768, [0.0] * 768, [0.0] * 768],
            },
            "success": "true",
            "errors": [],
            "messages": [],
        },
    )

    embeddings = CloudflareWorkersAIEmbeddings(account_id="123", api_token="abc")
    output = embeddings.embed_documents(documents)

    assert len(output) == 3
    assert len(output[0]) == 768

def test_cloudflare_workers_ai_embedding_query() -> None:
    """Test Cloudflare Workers AI embeddings."""

    responses.add(
        responses.POST,
        "https://api.cloudflare.com/client/v4/accounts/123/ai/run/@cf/baai/bge-base-en-v1.5",
        json={
            "result": {"shape": [1, 768], "data": [[0.0] * 768]},
            "success": "true",
            "errors": [],
            "messages": [],
        },
    )

    document = "foo bar"
    embeddings = CloudflareWorkersAIEmbeddings(account_id="123", api_token="abc")
    output = embeddings.embed_query(document)

    assert len(output) == 768


# --- libs/community/tests/integration_tests/embeddings/test_mosaicml.py ---

def test_mosaicml_embedding_endpoint() -> None:
    """Test MosaicML embeddings with a different endpoint"""
    documents = ["foo bar"]
    embedding = MosaicMLInstructorEmbeddings(
        endpoint_url=(
            "https://models.hosted-on.mosaicml.hosting/instructor-xl/v1/predict"
        )
    )
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


# --- libs/community/tests/integration_tests/graph_vectorstores/test_cassandra.py ---

    def test_gvs_traversal_search_sync(
        self,
        populated_graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        """Graph traversal search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        ts_response = g_store.traversal_search(query="[2, 10]", k=2, depth=2)
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        ts_labels = {doc.metadata["label"] for doc in ts_response}
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

        # verify the same works as a retriever
        retriever = g_store.as_retriever(
            search_type="traversal", search_kwargs={"k": 2, "depth": 2}
        )

        ts_labels = {doc.metadata["label"] for doc in retriever.invoke("[2, 10]")}
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

    async def test_gvs_traversal_search_async(
        self,
        populated_graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        """Graph traversal search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        ts_labels = set()
        async for doc in g_store.atraversal_search(query="[2, 10]", k=2, depth=2):
            ts_labels.add(doc.metadata["label"])
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

        # verify the same works as a retriever
        retriever = g_store.as_retriever(
            search_type="traversal", search_kwargs={"k": 2, "depth": 2}
        )

        ts_labels = {
            doc.metadata["label"] for doc in await retriever.ainvoke("[2, 10]")
        }
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}


# --- libs/community/tests/integration_tests/llms/test_ai21.py ---

def test_ai21_call() -> None:
    """Test valid call to ai21."""
    llm = AI21(maxTokens=10)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_ai21_call_experimental() -> None:
    """Test valid call to ai21 with an experimental model."""
    llm = AI21(maxTokens=10, model="j1-grande-instruct")
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_aleph_alpha.py ---

def test_aleph_alpha_call() -> None:
    """Test valid call to cohere."""
    llm = AlephAlpha(maximum_tokens=10)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_anthropic.py ---

def test_anthropic_call() -> None:
    """Test valid call to anthropic."""
    llm = Anthropic(model="claude-instant-1")  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_anthropic_streaming() -> None:
    """Test streaming tokens from anthropic."""
    llm = Anthropic(model="claude-instant-1")  # type: ignore[call-arg]
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)

async def test_anthropic_async_generate() -> None:
    """Test async generate."""
    llm = Anthropic()
    output = await llm.agenerate(["How many toes do dogs have?"])
    assert isinstance(output, LLMResult)


# --- libs/community/tests/integration_tests/llms/test_anyscale.py ---

def test_anyscale_call() -> None:
    """Test valid call to Anyscale."""
    llm = Anyscale()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_aviary.py ---

def test_aviary_call() -> None:
    """Test valid call to Anyscale."""
    llm = Aviary()
    output = llm.invoke("Say bar:")
    print(f"llm answer:\n{output}")  # noqa: T201
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_azureml_endpoint.py ---

def test_gpt2_call() -> None:
    """Test valid call to GPT2."""
    llm = AzureMLOnlineEndpoint(
        endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
        endpoint_url=os.getenv("OSS_ENDPOINT_URL"),  # type: ignore[arg-type]
        deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
        content_formatter=OSSContentFormatter(),
    )
    output = llm.invoke("Foo")
    assert isinstance(output, str)

def test_hf_call() -> None:
    """Test valid call to HuggingFace Foundation Model."""
    llm = AzureMLOnlineEndpoint(
        endpoint_api_key=os.getenv("HF_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
        endpoint_url=os.getenv("HF_ENDPOINT_URL"),  # type: ignore[arg-type]
        deployment_name=os.getenv("HF_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
        content_formatter=HFContentFormatter(),
    )
    output = llm.invoke("Foo")
    assert isinstance(output, str)

def test_dolly_call() -> None:
    """Test valid call to dolly-v2."""
    llm = AzureMLOnlineEndpoint(
        endpoint_api_key=os.getenv("DOLLY_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
        endpoint_url=os.getenv("DOLLY_ENDPOINT_URL"),  # type: ignore[arg-type]
        deployment_name=os.getenv("DOLLY_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
        content_formatter=DollyContentFormatter(),
    )
    output = llm.invoke("Foo")
    assert isinstance(output, str)

def test_custom_formatter() -> None:
    """Test ability to create a custom content formatter."""

    class CustomFormatter(ContentFormatterBase):
        content_type: str = "application/json"
        accepts: str = "application/json"

        def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:  # type: ignore[override]
            input_str = json.dumps(
                {
                    "inputs": [prompt],
                    "parameters": model_kwargs,
                    "options": {"use_cache": False, "wait_for_model": True},
                }
            )
            return input_str.encode("utf-8")

        def format_response_payload(self, output: bytes) -> str:  # type: ignore[override]
            response_json = json.loads(output)
            return response_json[0]["summary_text"]

    llm = AzureMLOnlineEndpoint(
        endpoint_api_key=os.getenv("BART_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
        endpoint_url=os.getenv("BART_ENDPOINT_URL"),  # type: ignore[arg-type]
        deployment_name=os.getenv("BART_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
        content_formatter=CustomFormatter(),
    )
    output = llm.invoke("Foo")
    assert isinstance(output, str)

def test_missing_content_formatter() -> None:
    """Test AzureML LLM without a content_formatter attribute"""
    with pytest.raises(AttributeError):
        llm = AzureMLOnlineEndpoint(
            endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
            endpoint_url=os.getenv("OSS_ENDPOINT_URL"),  # type: ignore[arg-type]
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
        )
        llm.invoke("Foo")

def test_invalid_request_format() -> None:
    """Test invalid request format."""

    class CustomContentFormatter(ContentFormatterBase):
        content_type: str = "application/json"
        accepts: str = "application/json"

        def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:  # type: ignore[override]
            input_str = json.dumps(
                {
                    "incorrect_input": {"input_string": [prompt]},
                    "parameters": model_kwargs,
                }
            )
            return str.encode(input_str)

        def format_response_payload(self, output: bytes) -> str:  # type: ignore[override]
            response_json = json.loads(output)
            return response_json[0]["0"]

    with pytest.raises(HTTPError):
        llm = AzureMLOnlineEndpoint(
            endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
            endpoint_url=os.getenv("OSS_ENDPOINT_URL"),  # type: ignore[arg-type]
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
            content_formatter=CustomContentFormatter(),
        )
        llm.invoke("Foo")

def test_incorrect_url(endpoint_url: str) -> None:
    """Testing AzureML Endpoint for an incorrect URL"""
    with pytest.raises(ValidationError):
        llm = AzureMLOnlineEndpoint(
            endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
            endpoint_url=endpoint_url,
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
            content_formatter=OSSContentFormatter(),
        )
        llm.invoke("Foo")

def test_incorrect_api_type() -> None:
    with pytest.raises(ValidationError):
        llm = AzureMLOnlineEndpoint(
            endpoint_api_key=os.getenv("OSS_ENDPOINT_API_KEY"),  # type: ignore[arg-type]
            endpoint_url=os.getenv("OSS_ENDPOINT_URL"),  # type: ignore[arg-type]
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
            endpoint_api_type="serverless",  # type: ignore[arg-type]
            content_formatter=OSSContentFormatter(),
        )
        llm.invoke("Foo")

def test_incorrect_key() -> None:
    """Testing AzureML Endpoint for incorrect key"""
    with pytest.raises(HTTPError):
        llm = AzureMLOnlineEndpoint(
            endpoint_api_key="incorrect-key",  # type: ignore[arg-type]
            endpoint_url=os.getenv("OSS_ENDPOINT_URL"),  # type: ignore[arg-type]
            deployment_name=os.getenv("OSS_DEPLOYMENT_NAME"),  # type: ignore[arg-type]
            content_formatter=OSSContentFormatter(),
        )
        llm.invoke("Foo")


# --- libs/community/tests/integration_tests/llms/test_azure_openai.py ---

def test_openai_call(llm: AzureOpenAI) -> None:
    """Test valid call to openai."""
    output = llm.invoke("Say something nice:")
    assert isinstance(output, str)

def test_openai_streaming(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    full_response = ""
    for token in generator:
        assert isinstance(token, str)
        full_response += token
    assert full_response

async def test_openai_astream(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)

async def test_openai_ainvoke(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)

def test_openai_invoke(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result, str)

def test_openai_multiple_prompts(llm: AzureOpenAI) -> None:
    """Test completion with multiple prompts."""
    output = llm.generate(["I'm Pickle Rick", "I'm Pickle Rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2

def test_openai_streaming_best_of_error() -> None:
    """Test validation for streaming fails if best_of is not 1."""
    with pytest.raises(ValueError):
        _get_llm(best_of=2, streaming=True)

def test_openai_streaming_n_error() -> None:
    """Test validation for streaming fails if n is not 1."""
    with pytest.raises(ValueError):
        _get_llm(n=2, streaming=True)

def test_openai_streaming_multiple_prompts_error() -> None:
    """Test validation for streaming fails if multiple prompts are given."""
    with pytest.raises(ValueError):
        _get_llm(streaming=True).generate(["I'm Pickle Rick", "I'm Pickle Rick"])

def test_openai_streaming_call() -> None:
    """Test valid call to openai."""
    llm = _get_llm(max_tokens=10, streaming=True)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

async def test_openai_async_generate() -> None:
    """Test async generation."""
    llm = _get_llm(max_tokens=10)
    output = await llm.agenerate(["Hello, how are you?"])
    assert isinstance(output, LLMResult)


# --- libs/community/tests/integration_tests/llms/test_baichuan.py ---

def test_call() -> None:
    """Test valid call to baichuan."""
    llm = BaichuanLLM()
    output = llm.invoke("Who won the second world war?")
    assert isinstance(output, str)

def test_generate() -> None:
    """Test valid call to baichuan."""
    llm = BaichuanLLM()
    output = llm.generate(["Who won the second world war?"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


# --- libs/community/tests/integration_tests/llms/test_banana.py ---

def test_banana_call() -> None:
    """Test valid call to BananaDev."""
    llm = Banana()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_bedrock.py ---

def test_claude_instant_v1(
    bedrock_runtime_client: "BaseClient", bedrock_models: dict
) -> None:
    try:
        llm = Bedrock(
            model_id="anthropic.claude-instant-v1",
            client=bedrock_runtime_client,
            model_kwargs={},
        )
        output = llm.invoke("Say something positive:")
        assert isinstance(output, str)
    except Exception as e:
        pytest.fail(f"can not instantiate claude-instant-v1: {e}", pytrace=False)

def test_amazon_bedrock_guardrails_no_intervention_for_valid_query(
    bedrock_runtime_client: "BaseClient", bedrock_models: dict
) -> None:
    try:
        llm = Bedrock(
            model_id="anthropic.claude-instant-v1",
            client=bedrock_runtime_client,
            model_kwargs={},
            guardrails={
                "id": GUARDRAILS_ID,
                "version": GUARDRAILS_VERSION,
                "trace": False,
            },
        )
        output = llm.invoke("Say something positive:")
        assert isinstance(output, str)
    except Exception as e:
        pytest.fail(f"can not instantiate claude-instant-v1: {e}", pytrace=False)

def test_amazon_bedrock_guardrails_intervention_for_invalid_query(
    bedrock_runtime_client: "BaseClient", bedrock_models: dict
) -> None:
    try:
        handler = BedrockAsyncCallbackHandler()
        llm = Bedrock(
            model_id="anthropic.claude-instant-v1",
            client=bedrock_runtime_client,
            model_kwargs={},
            guardrails={
                "id": GUARDRAILS_ID,
                "version": GUARDRAILS_VERSION,
                "trace": True,
            },
            callbacks=[handler],
        )
    except Exception as e:
        pytest.fail(f"can not instantiate claude-instant-v1: {e}", pytrace=False)
    else:
        llm.invoke(GUARDRAILS_TRIGGER)
        guardrails_intervened = handler.get_response()
        assert guardrails_intervened is True


# --- libs/community/tests/integration_tests/llms/test_bigdl_llm.py ---

def test_call(model_id: str) -> None:
    """Test valid call to bigdl-llm."""
    llm = BigdlLLM.from_model_id(
        model_id=model_id,
        model_kwargs={"temperature": 0, "max_length": 16, "trust_remote_code": True},
    )
    output = llm.invoke("Hello!")
    assert isinstance(output, str)

def test_generate(model_id: str) -> None:
    """Test valid call to bigdl-llm."""
    llm = BigdlLLM.from_model_id(
        model_id=model_id,
        model_kwargs={"temperature": 0, "max_length": 16, "trust_remote_code": True},
    )
    output = llm.generate(["Hello!"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


# --- libs/community/tests/integration_tests/llms/test_bittensor.py ---

def test_bittensor_call() -> None:
    """Test valid call to validator endpoint."""
    llm = NIBittensorLLM(system_prompt="Your task is to answer user prompt.")
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_cerebriumai.py ---

def test_cerebriumai_call() -> None:
    """Test valid call to cerebriumai."""
    llm = CerebriumAI(max_length=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_chatglm.py ---

def test_chatglm_call() -> None:
    """Test valid call to chatglm."""
    llm = ChatGLM()
    output = llm.invoke("北京和上海这两座城市有什么不同？")
    assert isinstance(output, str)

def test_chatglm_generate() -> None:
    """Test valid call to chatglm."""
    llm = ChatGLM()
    output = llm.generate(["who are you"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


# --- libs/community/tests/integration_tests/llms/test_clarifai.py ---

def test_clarifai_call() -> None:
    """Test valid call to clarifai."""
    llm = Clarifai(
        user_id="google-research",
        app_id="summarization",
        model_id="text-summarization-english-pegasus",
    )
    output = llm.invoke(
        "A chain is a serial assembly of connected pieces, called links, \
        typically made of metal, with an overall character similar to that\
        of a rope in that it is flexible and curved in compression but \
        linear, rigid, and load-bearing in tension. A chain may consist\
        of two or more links."
    )

    assert isinstance(output, str)
    assert llm._llm_type == "clarifai"
    assert llm.model_id == "text-summarization-english-pegasus"


# --- libs/community/tests/integration_tests/llms/test_cloudflare_workersai.py ---

def test_cloudflare_workersai_call() -> None:
    responses.add(
        responses.POST,
        "https://api.cloudflare.com/client/v4/accounts/my_account_id/ai/run/@cf/meta/llama-2-7b-chat-int8",
        json={"result": {"response": "4"}},
        status=200,
    )

    llm = CloudflareWorkersAI(
        account_id="my_account_id",
        api_token="my_api_token",
        model="@cf/meta/llama-2-7b-chat-int8",
    )
    output = llm.invoke("What is 2 + 2?")

    assert output == "4"

def test_cloudflare_workersai_stream() -> None:
    response_body = ['data: {"response": "Hello"}', "data: [DONE]"]
    responses.add(
        responses.POST,
        "https://api.cloudflare.com/client/v4/accounts/my_account_id/ai/run/@cf/meta/llama-2-7b-chat-int8",
        body="\n".join(response_body),
        status=200,
    )

    llm = CloudflareWorkersAI(
        account_id="my_account_id",
        api_token="my_api_token",
        model="@cf/meta/llama-2-7b-chat-int8",
        streaming=True,
    )

    outputs = []
    for chunk in llm.stream("Say Hello"):
        outputs.append(chunk)

    assert "".join(outputs) == "Hello"


# --- libs/community/tests/integration_tests/llms/test_cohere.py ---

def test_cohere_call() -> None:
    """Test valid call to cohere."""
    llm = Cohere(max_tokens=10)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_confident.py ---

def test_confident_deepeval() -> None:
    """Test valid call to Beam."""
    from deepeval.metrics.answer_relevancy import AnswerRelevancy

    from langchain_community.callbacks.confident_callback import DeepEvalCallbackHandler
    from langchain_community.llms import OpenAI

    answer_relevancy = AnswerRelevancy(minimum_score=0.3)
    deepeval_callback = DeepEvalCallbackHandler(
        implementation_name="exampleImplementation", metrics=[answer_relevancy]
    )
    llm = OpenAI(
        temperature=0,
        callbacks=[deepeval_callback],
        verbose=True,
        openai_api_key="<YOUR_API_KEY>",
    )
    llm.generate(
        [
            "What is the best evaluation tool out there? (no bias at all)",
        ]
    )
    assert answer_relevancy.is_successful(), "Answer not relevant"


# --- libs/community/tests/integration_tests/llms/test_deepinfra.py ---

def test_deepinfra_call() -> None:
    """Test valid call to DeepInfra."""
    llm = DeepInfra(model_id="meta-llama/Llama-2-7b-chat-hf")
    output = llm.invoke("What is 2 + 2?")
    assert isinstance(output, str)

async def test_deepinfra_acall() -> None:
    llm = DeepInfra(model_id="meta-llama/Llama-2-7b-chat-hf")
    output = await llm.ainvoke("What is 2 + 2?")
    assert llm._llm_type == "deepinfra"
    assert isinstance(output, str)

def test_deepinfra_stream() -> None:
    llm = DeepInfra(model_id="meta-llama/Llama-2-7b-chat-hf")
    num_chunks = 0
    for chunk in llm.stream("[INST] Hello [/INST] "):
        num_chunks += 1
    assert num_chunks > 0

async def test_deepinfra_astream() -> None:
    llm = DeepInfra(model_id="meta-llama/Llama-2-7b-chat-hf")
    num_chunks = 0
    async for chunk in llm.astream("[INST] Hello [/INST] "):
        num_chunks += 1
    assert num_chunks > 0


# --- libs/community/tests/integration_tests/llms/test_deepsparse.py ---

def test_deepsparse_call() -> None:
    """Test valid call to DeepSparse."""
    config = {"max_generated_tokens": 5, "use_deepsparse_cache": False}

    llm = DeepSparse(
        model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none",
        config=config,
    )

    output = llm.invoke("def ")
    assert isinstance(output, str)
    assert len(output) > 1
    assert output == "ids_to_names"


# --- libs/community/tests/integration_tests/llms/test_edenai.py ---

def test_edenai_call() -> None:
    """Test simple call to edenai."""
    llm = EdenAI(provider="openai", temperature=0.2, max_tokens=250)
    output = llm.invoke("Say foo:")

    assert llm._llm_type == "edenai"
    assert llm.feature == "text"
    assert llm.subfeature == "generation"
    assert isinstance(output, str)

async def test_edenai_acall() -> None:
    """Test simple call to edenai."""
    llm = EdenAI(provider="openai", temperature=0.2, max_tokens=250)
    output = await llm.agenerate(["Say foo:"])
    assert llm._llm_type == "edenai"
    assert llm.feature == "text"
    assert llm.subfeature == "generation"
    assert isinstance(output, str)

def test_edenai_call_with_old_params() -> None:
    """
    Test simple call to edenai with using `params`
    to pass optional parameters to api
    """
    llm = EdenAI(provider="openai", params={"temperature": 0.2, "max_tokens": 250})
    output = llm.invoke("Say foo:")

    assert llm._llm_type == "edenai"
    assert llm.feature == "text"
    assert llm.subfeature == "generation"
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_fireworks.py ---

def test_fireworks_call(llm: Fireworks) -> None:
    """Test valid call to fireworks."""
    output = llm.invoke("How is the weather in New York today?")
    assert isinstance(output, str)

def test_fireworks_invoke(llm: Fireworks) -> None:
    """Tests completion with invoke"""
    output = llm.invoke("How is the weather in New York today?", stop=[","])
    assert isinstance(output, str)
    assert output[-1] == ","

async def test_fireworks_ainvoke(llm: Fireworks) -> None:
    """Tests completion with invoke"""
    output = await llm.ainvoke("How is the weather in New York today?", stop=[","])
    assert isinstance(output, str)
    assert output[-1] == ","

def test_fireworks_multiple_prompts(
    llm: Fireworks,
) -> None:
    """Test completion with multiple prompts."""
    output = llm.generate(["How is the weather in New York today?", "I'm pickle rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2

def test_fireworks_streaming(llm: Fireworks) -> None:
    """Test stream completion."""
    generator = llm.stream("Who's the best quarterback in the NFL?")
    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)

def test_fireworks_streaming_stop_words(llm: Fireworks) -> None:
    """Test stream completion with stop words."""
    generator = llm.stream("Who's the best quarterback in the NFL?", stop=[","])
    assert isinstance(generator, Generator)

    last_token = ""
    for token in generator:
        last_token = token
        assert isinstance(token, str)
    assert last_token[-1] == ","

async def test_fireworks_streaming_async(llm: Fireworks) -> None:
    """Test stream completion."""

    last_token = ""
    async for token in llm.astream(
        "Who's the best quarterback in the NFL?", stop=[","]
    ):
        last_token = token
        assert isinstance(token, str)
    assert last_token[-1] == ","

async def test_fireworks_async_agenerate(llm: Fireworks) -> None:
    """Test async."""
    output = await llm.agenerate(["What is the best city to live in California?"])
    assert isinstance(output, LLMResult)

async def test_fireworks_multiple_prompts_async_agenerate(llm: Fireworks) -> None:
    output = await llm.agenerate(
        [
            "How is the weather in New York today?",
            "I'm pickle rick",
        ]
    )
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2


# --- libs/community/tests/integration_tests/llms/test_forefrontai.py ---

def test_forefrontai_call() -> None:
    """Test valid call to forefrontai."""
    llm = ForefrontAI(length=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_friendli.py ---

def test_friendli_invoke(friendli_llm: Friendli) -> None:
    """Test invoke."""
    output = friendli_llm.invoke("Say hello world.")
    assert isinstance(output, str)

async def test_friendli_ainvoke(friendli_llm: Friendli) -> None:
    """Test async invoke."""
    output = await friendli_llm.ainvoke("Say hello world.")
    assert isinstance(output, str)

def test_friendli_generate(friendli_llm: Friendli) -> None:
    """Test generate."""
    result = friendli_llm.generate(["Say hello world.", "Say bye world."])
    assert isinstance(result, LLMResult)
    generations = result.generations
    assert len(generations) == 2
    for generation in generations:
        gen_ = generation[0]
        assert isinstance(gen_, Generation)
        text = gen_.text
        assert isinstance(text, str)
        generation_info = gen_.generation_info
        if generation_info is not None:
            assert "token" in generation_info

async def test_friendli_agenerate(friendli_llm: Friendli) -> None:
    """Test async generate."""
    result = await friendli_llm.agenerate(["Say hello world.", "Say bye world."])
    assert isinstance(result, LLMResult)
    generations = result.generations
    assert len(generations) == 2
    for generation in generations:
        gen_ = generation[0]
        assert isinstance(gen_, Generation)
        text = gen_.text
        assert isinstance(text, str)
        generation_info = gen_.generation_info
        if generation_info is not None:
            assert "token" in generation_info

def test_friendli_stream(friendli_llm: Friendli) -> None:
    """Test stream."""
    stream = friendli_llm.stream("Say hello world.")
    for chunk in stream:
        assert isinstance(chunk, str)

async def test_friendli_astream(friendli_llm: Friendli) -> None:
    """Test async stream."""
    stream = friendli_llm.astream("Say hello world.")
    async for chunk in stream:
        assert isinstance(chunk, str)


# --- libs/community/tests/integration_tests/llms/test_google_palm.py ---

def test_google_generativeai_call(model_name: str) -> None:
    """Test valid call to Google GenerativeAI text API."""
    if model_name:
        llm = GooglePalm(max_output_tokens=10, model_name=model_name)  # type: ignore[call-arg]
    else:
        llm = GooglePalm(max_output_tokens=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
    assert llm._llm_type == "google_palm"
    if model_name and "gemini" in model_name:
        assert llm.client.model_name == "models/gemini-pro"
    else:
        assert llm.model_name == "models/text-bison-001"

def test_google_generativeai_generate(model_name: str) -> None:
    n = 1 if model_name == "gemini-pro" else 2
    if model_name:
        llm = GooglePalm(temperature=0.3, n=n, model_name=model_name)  # type: ignore[call-arg]
    else:
        llm = GooglePalm(temperature=0.3, n=n)  # type: ignore[call-arg]
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == n

async def test_google_generativeai_agenerate() -> None:
    llm = GooglePalm(temperature=0, model_name="gemini-pro")  # type: ignore[call-arg]
    output = await llm.agenerate(["Please say foo:"])
    assert isinstance(output, LLMResult)

def test_generativeai_stream() -> None:
    llm = GooglePalm(temperature=0, model_name="gemini-pro")  # type: ignore[call-arg]
    outputs = list(llm.stream("Please say foo:"))
    assert isinstance(outputs[0], str)


# --- libs/community/tests/integration_tests/llms/test_gooseai.py ---

def test_gooseai_call() -> None:
    """Test valid call to gooseai."""
    llm = GooseAI(max_tokens=10)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_gooseai_call_fairseq() -> None:
    """Test valid call to gooseai with fairseq model."""
    llm = GooseAI(model_name="fairseq-1-3b", max_tokens=10)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_gooseai_stop_valid() -> None:
    """Test gooseai stop logic on valid configuration."""
    query = "write an ordered list of five items"
    first_llm = GooseAI(stop="3", temperature=0)  # type: ignore[call-arg]
    first_output = first_llm.invoke(query)
    second_llm = GooseAI(temperature=0)
    second_output = second_llm.invoke(query, stop=["3"])
    # Because it stops on new lines, shouldn't return anything
    assert first_output == second_output


# --- libs/community/tests/integration_tests/llms/test_gpt4all.py ---

def test_gpt4all_inference() -> None:
    """Test valid gpt4all inference."""
    model_path = _download_model()
    llm = GPT4All(model=model_path)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_gradient_ai.py ---

def test_gradient_acall() -> None:
    """Test simple call to gradient.ai."""
    model = os.environ["GRADIENT_MODEL"]
    gradient_access_token = os.environ["GRADIENT_ACCESS_TOKEN"]
    gradient_workspace_id = os.environ["GRADIENT_WORKSPACE_ID"]
    llm = GradientLLM(
        model=model,
        gradient_access_token=gradient_access_token,
        gradient_workspace_id=gradient_workspace_id,
    )
    output = llm.invoke("Say hello:", temperature=0.2, max_tokens=250)

    assert llm._llm_type == "gradient"

    assert isinstance(output, str)
    assert len(output)

async def test_gradientai_acall() -> None:
    """Test async call to gradient.ai."""
    model = os.environ["GRADIENT_MODEL"]
    gradient_access_token = os.environ["GRADIENT_ACCESS_TOKEN"]
    gradient_workspace_id = os.environ["GRADIENT_WORKSPACE_ID"]
    llm = GradientLLM(
        model=model,
        gradient_access_token=gradient_access_token,
        gradient_workspace_id=gradient_workspace_id,
    )
    output = await llm.agenerate(["Say hello:"], temperature=0.2, max_tokens=250)
    assert llm._llm_type == "gradient"

    assert isinstance(output, str)
    assert len(output)


# --- libs/community/tests/integration_tests/llms/test_huggingface_endpoint.py ---

def test_huggingface_endpoint_call_error() -> None:
    """Test valid call to HuggingFace that errors."""
    llm = HuggingFaceEndpoint(endpoint_url="", model_kwargs={"max_new_tokens": -1})  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        llm.invoke("Say foo:")

def test_huggingface_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = HuggingFaceEndpoint(repo_id="gpt2", model_kwargs={"max_new_tokens": 10})  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    print(output)  # noqa: T201
    assert isinstance(output, str)

def test_huggingface_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text model."""
    llm = HuggingFaceEndpoint(repo_id="google/flan-t5-xl")  # type: ignore[call-arg]
    output = llm.invoke("The capital of New York is")
    assert output == "Albany"

def test_huggingface_summarization() -> None:
    """Test valid call to HuggingFace summarization model."""
    llm = HuggingFaceEndpoint(repo_id="facebook/bart-large-cnn")  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_huggingface_call_error() -> None:
    """Test valid call to HuggingFace that errors."""
    llm = HuggingFaceEndpoint(repo_id="gpt2", model_kwargs={"max_new_tokens": -1})  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        llm.invoke("Say foo:")

def test_invocation_params_stop_sequences() -> None:
    llm = HuggingFaceEndpoint()  # type: ignore[call-arg]
    assert llm._default_params["stop_sequences"] == []

    runtime_stop = None
    assert llm._invocation_params(runtime_stop)["stop_sequences"] == []
    assert llm._default_params["stop_sequences"] == []

    runtime_stop = ["stop"]
    assert llm._invocation_params(runtime_stop)["stop_sequences"] == ["stop"]
    assert llm._default_params["stop_sequences"] == []

    llm = HuggingFaceEndpoint(stop_sequences=["."])  # type: ignore[call-arg]
    runtime_stop = ["stop"]
    assert llm._invocation_params(runtime_stop)["stop_sequences"] == [".", "stop"]
    assert llm._default_params["stop_sequences"] == ["."]


# --- libs/community/tests/integration_tests/llms/test_huggingface_hub.py ---

def test_huggingface_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"max_new_tokens": 10})
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_huggingface_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text model."""
    llm = HuggingFaceHub(repo_id="google/flan-t5-xl")
    output = llm.invoke("The capital of New York is")
    assert output == "Albany"

def test_huggingface_summarization() -> None:
    """Test valid call to HuggingFace summarization model."""
    llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn")
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_huggingface_call_error() -> None:
    """Test valid call to HuggingFace that errors."""
    llm = HuggingFaceHub(model_kwargs={"max_new_tokens": -1})
    with pytest.raises(ValueError):
        llm.invoke("Say foo:")


# --- libs/community/tests/integration_tests/llms/test_huggingface_pipeline.py ---

def test_huggingface_pipeline_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2", task="text-generation", pipeline_kwargs={"max_new_tokens": 10}
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_huggingface_pipeline_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text generation model."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small", task="text2text-generation"
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_huggingface_pipeline_device_map() -> None:
    """Test pipelines specifying the device map parameter."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        device_map="auto",
        pipeline_kwargs={"max_new_tokens": 10},
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_init_with_pipeline() -> None:
    """Test initialization with a HF pipeline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_huggingface_pipeline_runtime_kwargs() -> None:
    """Test pipelines specifying the device map parameter."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
    )
    prompt = "Say foo:"
    output = llm.invoke(prompt, pipeline_kwargs={"max_new_tokens": 2})
    assert len(output) < 10

def test_huggingface_pipeline_text_generation_ov() -> None:
    """Test valid call to HuggingFace text generation model with openvino."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        backend="openvino",
        model_kwargs={"device": "CPU", "ov_config": ov_config},
        pipeline_kwargs={"max_new_tokens": 64},
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_huggingface_pipeline_text2text_generation_ov() -> None:
    """Test valid call to HuggingFace text2text generation model with openvino."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small",
        task="text2text-generation",
        backend="openvino",
        model_kwargs={"device": "CPU", "ov_config": ov_config},
        pipeline_kwargs={"max_new_tokens": 64},
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_huggingface_text_gen_inference.py ---

def test_invocation_params_stop_sequences() -> None:
    llm = HuggingFaceTextGenInference()
    assert llm._default_params["stop_sequences"] == []

    runtime_stop = None
    assert llm._invocation_params(runtime_stop)["stop_sequences"] == []
    assert llm._default_params["stop_sequences"] == []

    runtime_stop = ["stop"]
    assert llm._invocation_params(runtime_stop)["stop_sequences"] == ["stop"]
    assert llm._default_params["stop_sequences"] == []

    llm = HuggingFaceTextGenInference(stop_sequences=["."])
    runtime_stop = ["stop"]
    assert llm._invocation_params(runtime_stop)["stop_sequences"] == [".", "stop"]
    assert llm._default_params["stop_sequences"] == ["."]


# --- libs/community/tests/integration_tests/llms/test_ipex_llm.py ---

def test_call(model_id: str) -> None:
    """Test valid call."""
    llm = load_model(model_id)
    output = llm.invoke("Hello!")
    assert isinstance(output, str)

def test_asym_int4(model_id: str) -> None:
    """Test asym int4 data type."""
    llm = load_model_more_types(model_id=model_id, load_in_low_bit="asym_int4")
    output = llm.invoke("Hello!")
    assert isinstance(output, str)

def test_generate(model_id: str) -> None:
    """Test valid generate."""
    llm = load_model(model_id)
    output = llm.generate(["Hello!"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)

def test_save_load_lowbit(model_id: str) -> None:
    """Test save and load lowbit model."""
    saved_lowbit_path = "/tmp/saved_model"
    llm = load_model(model_id)
    llm.model.save_low_bit(saved_lowbit_path)
    del llm
    loaded_llm = IpexLLM.from_model_id_low_bit(
        model_id=saved_lowbit_path,
        tokenizer_id=model_id,
        model_kwargs={"temperature": 0, "max_length": 16, "trust_remote_code": True},
    )
    output = loaded_llm.invoke("Hello!")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_konko.py ---

def test_konko_call() -> None:
    """Test simple call to konko."""
    llm = Konko(
        model="mistralai/mistral-7b-v0.1",
        temperature=0.2,
        max_tokens=250,
    )
    output = llm.invoke("Say foo:")

    assert llm._llm_type == "konko"
    assert isinstance(output, str)

async def test_konko_acall() -> None:
    """Test simple call to konko."""
    llm = Konko(
        model="mistralai/mistral-7b-v0.1",
        temperature=0.2,
        max_tokens=250,
    )
    output = await llm.agenerate(["Say foo:"], stop=["bar"])

    assert llm._llm_type == "konko"
    output_text = output.generations[0][0].text
    assert isinstance(output_text, str)
    assert output_text.count("bar") <= 1


# --- libs/community/tests/integration_tests/llms/test_llamacpp.py ---

def test_llamacpp_inference() -> None:
    """Test valid llama.cpp inference."""
    model_path = get_model()
    llm = LlamaCpp(model_path=model_path)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
    assert len(output) > 1

def test_llamacpp_streaming() -> None:
    """Test streaming tokens from LlamaCpp."""
    model_path = get_model()
    llm = LlamaCpp(model_path=model_path, max_tokens=10)
    generator = llm.stream("Q: How do you say 'hello' in German? A:'", stop=["'"])
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        assert not isinstance(chunk, str)
        # Note that this matches the OpenAI format:
        assert isinstance(chunk["choices"][0]["text"], str)
        stream_results_string += chunk["choices"][0]["text"]
    assert len(stream_results_string.strip()) > 1


# --- libs/community/tests/integration_tests/llms/test_llamafile.py ---

def test_llamafile_call() -> None:
    llm = Llamafile()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_llamafile_streaming() -> None:
    llm = Llamafile(streaming=True)
    generator = llm.stream("Tell me about Roman dodecahedrons.")
    assert isinstance(generator, Generator)
    for token in generator:
        assert isinstance(token, str)


# --- libs/community/tests/integration_tests/llms/test_manifest.py ---

def test_manifest_wrapper() -> None:
    """Test manifest wrapper."""
    from manifest import Manifest

    manifest = Manifest(client_name="openai")
    llm = ManifestWrapper(client=manifest, llm_kwargs={"temperature": 0})
    output = llm.invoke("The capital of New York is:")
    assert output == "Albany"


# --- libs/community/tests/integration_tests/llms/test_minimax.py ---

def test_minimax_call() -> None:
    """Test valid call to minimax."""
    llm = Minimax(max_tokens=10)  # type: ignore[call-arg]
    output = llm.invoke("Hello world!")
    assert isinstance(output, str)

def test_minimax_call_successful() -> None:
    """Test valid call to minimax."""
    llm = Minimax()  # type: ignore[call-arg]
    output = llm.invoke(
        "A chain is a serial assembly of connected pieces, called links, \
        typically made of metal, with an overall character similar to that\
        of a rope in that it is flexible and curved in compression but \
        linear, rigid, and load-bearing in tension. A chain may consist\
        of two or more links."
    )
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_mlx_pipeline.py ---

def test_mlx_pipeline_text_generation() -> None:
    """Test valid call to MLX text generation model."""
    llm = MLXPipeline.from_model_id(
        model_id="mlx-community/quantized-gemma-2b",
        pipeline_kwargs={"max_tokens": 10},
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_init_with_model_and_tokenizer() -> None:
    """Test initialization with a HF pipeline."""
    from mlx_lm import load

    model, tokenizer = load("mlx-community/quantized-gemma-2b")
    llm = MLXPipeline(model=model, tokenizer=tokenizer)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_huggingface_pipeline_runtime_kwargs() -> None:
    """Test pipelines specifying the device map parameter."""
    llm = MLXPipeline.from_model_id(
        model_id="mlx-community/quantized-gemma-2b",
    )
    prompt = "Say foo:"
    output = llm.invoke(prompt, pipeline_kwargs={"max_tokens": 2})
    assert len(output) < 10

def test_mlx_pipeline_with_params() -> None:
    """Test valid call to MLX text generation model."""
    llm = MLXPipeline.from_model_id(
        model_id="mlx-community/quantized-gemma-2b",
        pipeline_kwargs={
            "max_tokens": 10,
            "temp": 0.8,
            "verbose": False,
            "repetition_penalty": 1.1,
            "repetition_context_size": 64,
            "top_p": 0.95,
        },
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_modal.py ---

def test_modal_call() -> None:
    """Test valid call to Modal."""
    llm = Modal()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_mosaicml.py ---

def test_mosaicml_llm_call() -> None:
    """Test valid call to MosaicML."""
    llm = MosaicML(model_kwargs={})
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_mosaicml_endpoint_change() -> None:
    """Test valid call to MosaicML."""
    new_url = "https://models.hosted-on.mosaicml.hosting/mpt-30b-instruct/v1/predict"
    llm = MosaicML(endpoint_url=new_url)
    assert llm.endpoint_url == new_url
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_mosaicml_extra_kwargs() -> None:
    llm = MosaicML(model_kwargs={"max_new_tokens": 1})
    assert llm.model_kwargs == {"max_new_tokens": 1}

    output = llm.invoke("Say foo:")

    assert isinstance(output, str)

    # should only generate one new token (which might be a new line or whitespace token)
    assert len(output.split()) <= 1

def test_instruct_prompt() -> None:
    """Test instruct prompt."""
    llm = MosaicML(inject_instruction_format=True, model_kwargs={"max_new_tokens": 10})
    instruction = "Repeat the word foo"
    prompt = llm._transform_prompt(instruction)
    expected_prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)
    assert prompt == expected_prompt
    output = llm.invoke(prompt)
    assert isinstance(output, str)

def test_retry_logic() -> None:
    """Tests that two queries (which would usually exceed the rate limit) works"""
    llm = MosaicML(inject_instruction_format=True, model_kwargs={"max_new_tokens": 10})
    instruction = "Repeat the word foo"
    prompt = llm._transform_prompt(instruction)
    expected_prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)
    assert prompt == expected_prompt
    output = llm.invoke(prompt)
    assert isinstance(output, str)
    output = llm.invoke(prompt)
    assert isinstance(output, str)

def test_short_retry_does_not_loop() -> None:
    """Tests that two queries with a short retry sleep does not infinite loop"""
    llm = MosaicML(
        inject_instruction_format=True,
        model_kwargs={"do_sample": False},
        retry_sleep=0.1,
    )
    instruction = "Repeat the word foo"
    prompt = llm._transform_prompt(instruction)
    expected_prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)
    assert prompt == expected_prompt

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Error raised by inference API: rate limit exceeded.\nResponse: You have "
            "reached maximum request limit.\n"
        ),
    ):
        for _ in range(10):
            output = llm.invoke(prompt)
            assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_nlpcloud.py ---

def test_nlpcloud_call() -> None:
    """Test valid call to nlpcloud."""
    llm = NLPCloud(max_length=10)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_octoai_endpoint.py ---

def test_octoai_endpoint_call() -> None:
    """Test valid call to OctoAI endpoint."""
    llm = OctoAIEndpoint()
    output = llm.invoke("Which state is Los Angeles in?")
    print(output)  # noqa: T201
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_opaqueprompts.py ---

def test_opaqueprompts() -> None:
    chain = PromptTemplate.from_template(prompt_template) | OpaquePrompts(llm=OpenAI())  # type: ignore[call-arg]
    output = chain.invoke(
        {
            "question": "Write a text message to remind John to do password reset \
                for his website through his email to stay secure."
        }
    )
    assert isinstance(output, str)

def test_opaqueprompts_functions() -> None:
    prompt = (PromptTemplate.from_template(prompt_template),)
    llm = OpenAI()
    pg_chain = (
        op.sanitize
        | RunnableParallel(
            secure_context=lambda x: x["secure_context"],
            response=(lambda x: x["sanitized_input"])  # type: ignore[operator]
            | prompt
            | llm
            | StrOutputParser(),
        )
        | (lambda x: op.desanitize(x["response"], x["secure_context"]))
    )

    pg_chain.invoke(
        {
            "question": "Write a text message to remind John to do password reset\
                 for his website through his email to stay secure.",
            "history": "",
        }
    )


# --- libs/community/tests/integration_tests/llms/test_openai.py ---

def test_openai_call() -> None:
    """Test valid call to openai."""
    llm = OpenAI()
    output = llm.invoke("Say something nice:")
    assert isinstance(output, str)

def test_openai_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    llm = OpenAI(max_tokens=10)
    llm_result = llm.generate(["Hello, how are you?"])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == llm.model_name

def test_openai_stop_valid() -> None:
    """Test openai stop logic on valid configuration."""
    query = "write an ordered list of five items"
    first_llm = OpenAI(stop="3", temperature=0)  # type: ignore[call-arg]
    first_output = first_llm.invoke(query)
    second_llm = OpenAI(temperature=0)
    second_output = second_llm.invoke(query, stop=["3"])
    # Because it stops on new lines, shouldn't return anything
    assert first_output == second_output

def test_openai_stop_error() -> None:
    """Test openai stop logic on bad configuration."""
    llm = OpenAI(stop="3", temperature=0)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        llm.invoke("write an ordered list of five items", stop=["\n"])

def test_openai_streaming() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)

async def test_openai_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)

async def test_openai_ainvoke() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)

def test_openai_invoke() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result, str)

def test_openai_multiple_prompts() -> None:
    """Test completion with multiple prompts."""
    llm = OpenAI(max_tokens=10)
    output = llm.generate(["I'm Pickle Rick", "I'm Pickle Rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2

def test_openai_streaming_best_of_error() -> None:
    """Test validation for streaming fails if best_of is not 1."""
    with pytest.raises(ValueError):
        OpenAI(best_of=2, streaming=True)

def test_openai_streaming_n_error() -> None:
    """Test validation for streaming fails if n is not 1."""
    with pytest.raises(ValueError):
        OpenAI(n=2, streaming=True)

def test_openai_streaming_multiple_prompts_error() -> None:
    """Test validation for streaming fails if multiple prompts are given."""
    with pytest.raises(ValueError):
        OpenAI(streaming=True).generate(["I'm Pickle Rick", "I'm Pickle Rick"])

def test_openai_streaming_call() -> None:
    """Test valid call to openai."""
    llm = OpenAI(max_tokens=10, streaming=True)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

async def test_openai_async_generate() -> None:
    """Test async generation."""
    llm = OpenAI(max_tokens=10)
    output = await llm.agenerate(["Hello, how are you?"])
    assert isinstance(output, LLMResult)


# --- libs/community/tests/integration_tests/llms/test_openllm.py ---

def test_openai_call() -> None:
    """Test valid call to openai."""
    llm = OpenLLM()
    output = llm.invoke("Say something nice:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_openlm.py ---

def test_openlm_call() -> None:
    """Test valid call to openlm."""
    llm = OpenLM(model_name="dolly-v2-7b", max_tokens=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_outlines.py ---

def test_outlines_inference(llm: Outlines) -> None:
    """Test valid outlines inference."""
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
    assert len(output) > 1

def test_outlines_streaming(llm: Outlines) -> None:
    """Test streaming tokens from Outlines."""
    generator = llm.stream("Q: How do you say 'hello' in Spanish?\n\nA:")
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        print(chunk)
        assert isinstance(chunk, str)
        stream_results_string += chunk
    print(stream_results_string)
    assert len(stream_results_string.strip()) > 1

def test_outlines_regex(llm: Outlines) -> None:
    """Test regex for generating a valid IP address"""
    ip_regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    llm.regex = ip_regex
    assert llm.regex == ip_regex

    output = llm.invoke("Q: What is the IP address of googles dns server?\n\nA: ")

    assert isinstance(output, str)

    assert re.match(ip_regex, output), (
        f"Generated output '{output}' is not a valid IP address"
    )

def test_outlines_type_constraints(llm: Outlines) -> None:
    """Test type constraints for generating an integer"""
    llm.type_constraints = int
    output = llm.invoke(
        "Q: What is the answer to life, the universe, and everything?\n\nA: "
    )
    assert int(output)

def test_outlines_json(llm: Outlines) -> None:
    """Test json for generating a valid JSON object"""

    class Person(BaseModel):
        name: str

    llm.json_schema = Person
    output = llm.invoke("Q: Who is the author of LangChain?\n\nA: ")
    person = Person.model_validate_json(output)
    assert isinstance(person, Person)

def test_outlines_grammar(llm: Outlines) -> None:
    """Test grammar for generating a valid arithmetic expression"""
    llm.grammar = """
        ?start: expression
        ?expression: term (("+" | "-") term)*
        ?term: factor (("*" | "/") factor)*
        ?factor: NUMBER | "-" factor | "(" expression ")"
        %import common.NUMBER
        %import common.WS
        %ignore WS
    """

    output = llm.invoke("Here is a complex arithmetic expression: ")

    # Validate the output is a non-empty string
    assert isinstance(output, str) and output.strip(), (
        "Output should be a non-empty string"
    )

    # Use a simple regex to check if the output contains basic arithmetic operations and numbers
    assert re.search(r"[\d\+\-\*/\(\)]+", output), (
        f"Generated output '{output}' does not appear to be a valid arithmetic expression"
    )


# --- libs/community/tests/integration_tests/llms/test_pai_eas_endpoint.py ---

def test_pai_eas_v1_call() -> None:
    """Test valid call to PAI-EAS Service."""
    llm = PaiEasEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),  # type: ignore[arg-type]
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),  # type: ignore[arg-type]
        version="1.0",
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_pai_eas_v2_call() -> None:
    llm = PaiEasEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),  # type: ignore[arg-type]
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),  # type: ignore[arg-type]
        version="2.0",
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_pai_eas_v1_streaming() -> None:
    """Test streaming call to PAI-EAS Service."""
    llm = PaiEasEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),  # type: ignore[arg-type]
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),  # type: ignore[arg-type]
        version="1.0",
    )
    generator = llm.stream("Q: How do you say 'hello' in German? A:'", stop=["."])
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        assert isinstance(chunk, str)
        stream_results_string = chunk
    assert len(stream_results_string.strip()) > 1

def test_pai_eas_v2_streaming() -> None:
    llm = PaiEasEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),  # type: ignore[arg-type]
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),  # type: ignore[arg-type]
        version="2.0",
    )
    generator = llm.stream("Q: How do you say 'hello' in German? A:'", stop=["."])
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        assert isinstance(chunk, str)
        stream_results_string = chunk
    assert len(stream_results_string.strip()) > 1


# --- libs/community/tests/integration_tests/llms/test_petals.py ---

def test_gooseai_call() -> None:
    """Test valid call to gooseai."""
    llm = Petals(max_new_tokens=10)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_pipelineai.py ---

def test_pipelineai_call() -> None:
    """Test valid call to Pipeline Cloud."""
    llm = PipelineAI()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_predictionguard.py ---

def test_predictionguard_invoke() -> None:
    """Test valid call to prediction guard."""
    llm = PredictionGuard(model="Hermes-3-Llama-3.1-8B")
    output = llm.invoke("Tell a joke.")
    assert isinstance(output, str)

def test_predictionguard_pii() -> None:
    llm = PredictionGuard(
        model="Hermes-3-Llama-3.1-8B",
        predictionguard_input={"pii": "block"},
        max_tokens=100,
        temperature=1.0,
    )

    messages = [
        "Hello, my name is John Doe and my SSN is 111-22-3333",
    ]

    with pytest.raises(ValueError, match=r"Could not make prediction. pii detected"):
        llm.invoke(messages)


# --- libs/community/tests/integration_tests/llms/test_promptlayer_openai.py ---

def test_promptlayer_openai_call() -> None:
    """Test valid call to promptlayer openai."""
    llm = PromptLayerOpenAI(max_tokens=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_promptlayer_openai_stop_valid() -> None:
    """Test promptlayer openai stop logic on valid configuration."""
    query = "write an ordered list of five items"
    first_llm = PromptLayerOpenAI(stop="3", temperature=0)  # type: ignore[call-arg]
    first_output = first_llm.invoke(query)
    second_llm = PromptLayerOpenAI(temperature=0)  # type: ignore[call-arg]
    second_output = second_llm.invoke(query, stop=["3"])
    # Because it stops on new lines, shouldn't return anything
    assert first_output == second_output

def test_promptlayer_openai_stop_error() -> None:
    """Test promptlayer openai stop logic on bad configuration."""
    llm = PromptLayerOpenAI(stop="3", temperature=0)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        llm.invoke("write an ordered list of five items", stop=["\n"])

def test_promptlayer_openai_streaming() -> None:
    """Test streaming tokens from promptalyer OpenAI."""
    llm = PromptLayerOpenAI(max_tokens=10)  # type: ignore[call-arg]
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token["choices"][0]["text"], str)


# --- libs/community/tests/integration_tests/llms/test_propmptlayer_openai_chat.py ---

def test_promptlayer_openai_chat_call() -> None:
    """Test valid call to promptlayer openai."""
    llm = PromptLayerOpenAIChat(max_tokens=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_promptlayer_openai_chat_stop_valid() -> None:
    """Test promptlayer openai stop logic on valid configuration."""
    query = "write an ordered list of five items"
    first_llm = PromptLayerOpenAIChat(stop="3", temperature=0)  # type: ignore[call-arg]
    first_output = first_llm.invoke(query)
    second_llm = PromptLayerOpenAIChat(temperature=0)  # type: ignore[call-arg]
    second_output = second_llm.invoke(query, stop=["3"])
    # Because it stops on new lines, shouldn't return anything
    assert first_output == second_output

def test_promptlayer_openai_chat_stop_error() -> None:
    """Test promptlayer openai stop logic on bad configuration."""
    llm = PromptLayerOpenAIChat(stop="3", temperature=0)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        llm.invoke("write an ordered list of five items", stop=["\n"])


# --- libs/community/tests/integration_tests/llms/test_qianfan_endpoint.py ---

def test_call() -> None:
    """Test valid call to qianfan."""
    llm = QianfanLLMEndpoint()
    output = llm.invoke("write a joke")
    assert isinstance(output, str)

def test_generate() -> None:
    """Test valid call to qianfan."""
    llm = QianfanLLMEndpoint()
    output = llm.generate(["write a joke"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)

def test_generate_stream() -> None:
    """Test valid call to qianfan."""
    llm = QianfanLLMEndpoint()
    output = llm.stream("write a joke")
    assert isinstance(output, Generator)

async def test_qianfan_aio() -> None:
    llm = QianfanLLMEndpoint(streaming=True)

    async for token in llm.astream("hi qianfan."):
        assert isinstance(token, str)

def test_rate_limit() -> None:
    llm = QianfanLLMEndpoint(model="ERNIE-Bot", init_kwargs={"query_per_second": 2})
    assert llm.client._client._rate_limiter._sync_limiter._query_per_second == 2
    output = llm.generate(["write a joke"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


# --- libs/community/tests/integration_tests/llms/test_replicate.py ---

def test_replicate_call() -> None:
    """Test simple non-streaming call to Replicate."""
    llm = Replicate(model=TEST_MODEL_HELLO)
    output = llm.invoke("What is LangChain")
    assert output
    assert isinstance(output, str)

def test_replicate_model_kwargs() -> None:
    """Test simple non-streaming call to Replicate."""
    llm = Replicate(  # type: ignore[call-arg]
        model=TEST_MODEL_LANG, model_kwargs={"max_new_tokens": 10, "temperature": 0.01}
    )
    long_output = llm.invoke("What is LangChain")
    llm = Replicate(  # type: ignore[call-arg]
        model=TEST_MODEL_LANG, model_kwargs={"max_new_tokens": 5, "temperature": 0.01}
    )
    short_output = llm.invoke("What is LangChain")
    assert len(short_output) < len(long_output)
    assert llm.model_kwargs == {"max_new_tokens": 5, "temperature": 0.01}


# --- libs/community/tests/integration_tests/llms/test_rwkv.py ---

def test_rwkv_inference() -> None:
    """Test valid gpt4all inference."""
    model_path = _download_model()
    llm = RWKV(model=model_path, tokens_path="20B_tokenizer.json", strategy="cpu fp32")
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_sambanova.py ---

def test_sambanova_cloud_call() -> None:
    """Test simple non-streaming call to sambastudio."""
    llm = SambaNovaCloud()
    output = llm.invoke("What is LangChain")
    assert output
    assert isinstance(output, str)

def test_sambastudio_call() -> None:
    """Test simple non-streaming call to sambastudio."""
    llm = SambaStudio()
    output = llm.invoke("What is LangChain")
    assert output
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_self_hosted_llm.py ---

def test_self_hosted_huggingface_pipeline_text_generation() -> None:
    """Test valid call to self-hosted HuggingFace text generation model."""
    gpu = get_remote_instance()
    llm = SelfHostedHuggingFaceLLM(
        model_id="gpt2",
        task="text-generation",
        model_kwargs={"n_positions": 1024},
        hardware=gpu,
        model_reqs=model_reqs,
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_self_hosted_huggingface_pipeline_text2text_generation() -> None:
    """Test valid call to self-hosted HuggingFace text2text generation model."""
    gpu = get_remote_instance()
    llm = SelfHostedHuggingFaceLLM(
        model_id="google/flan-t5-small",
        task="text2text-generation",
        hardware=gpu,
        model_reqs=model_reqs,
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_self_hosted_huggingface_pipeline_summarization() -> None:
    """Test valid call to self-hosted HuggingFace summarization model."""
    gpu = get_remote_instance()
    llm = SelfHostedHuggingFaceLLM(
        model_id="facebook/bart-large-cnn",
        task="summarization",
        hardware=gpu,
        model_reqs=model_reqs,
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_init_with_local_pipeline() -> None:
    """Test initialization with a self-hosted HF pipeline."""
    gpu = get_remote_instance()
    pipeline = load_pipeline()
    llm = SelfHostedPipeline.from_pipeline(
        pipeline=pipeline,
        hardware=gpu,
        model_reqs=model_reqs,
        inference_fn=inference_fn,
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_init_with_pipeline_path() -> None:
    """Test initialization with a self-hosted HF pipeline."""
    gpu = get_remote_instance()
    pipeline = load_pipeline()
    import runhouse as rh

    rh.blob(pickle.dumps(pipeline), path="models/pipeline.pkl").save().to(
        gpu, path="models"
    )
    llm = SelfHostedPipeline.from_pipeline(
        pipeline="models/pipeline.pkl",
        hardware=gpu,
        model_reqs=model_reqs,
        inference_fn=inference_fn,
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_init_with_pipeline_fn() -> None:
    """Test initialization with a self-hosted HF pipeline."""
    gpu = get_remote_instance()
    llm = SelfHostedPipeline(
        model_load_fn=load_pipeline,
        hardware=gpu,
        model_reqs=model_reqs,
        inference_fn=inference_fn,
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_sparkllm.py ---

def test_call() -> None:
    """Test valid call to sparkllm."""
    llm = SparkLLM()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_generate() -> None:
    """Test valid call to sparkllm."""
    llm = SparkLLM()
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)

def test_spark_llm_with_stream() -> None:
    """Test SparkLLM with stream."""
    llm = SparkLLM()
    for chunk in llm.stream("你好呀"):
        assert isinstance(chunk, str)


# --- libs/community/tests/integration_tests/llms/test_stochasticai.py ---

def test_stochasticai_call() -> None:
    """Test valid call to StochasticAI."""
    llm = StochasticAI()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_symblai_nebula.py ---

def test_symblai_nebula_call() -> None:
    """Test valid call to Nebula."""
    conversation = """Sam: Good morning, team! Let's keep this standup concise. 
    We'll go in the usual order: what you did yesterday, 
    what you plan to do today, and any blockers. Alex, kick us off.
Alex: Morning! Yesterday, I wrapped up the UI for the user dashboard. 
The new charts and widgets are now responsive. 
I also had a sync with the design team to ensure the final touchups are in 
line with the brand guidelines. Today, I'll start integrating the frontend with 
the new API endpoints Rhea was working on. 
The only blocker is waiting for some final API documentation, 
but I guess Rhea can update on that.
Rhea: Hey, all! Yep, about the API documentation - I completed the majority of
 the backend work for user data retrieval yesterday. 
 The endpoints are mostly set up, but I need to do a bit more testing today. 
 I'll finalize the API documentation by noon, so that should unblock Alex. 
 After that, I’ll be working on optimizing the database queries 
 for faster data fetching. No other blockers on my end.
Sam: Great, thanks Rhea. Do reach out if you need any testing assistance
 or if there are any hitches with the database. 
 Now, my update: Yesterday, I coordinated with the client to get clarity 
 on some feature requirements. Today, I'll be updating our project roadmap 
 and timelines based on their feedback. Additionally, I'll be sitting with 
 the QA team in the afternoon for preliminary testing. 
 Blocker: I might need both of you to be available for a quick call 
 in case the client wants to discuss the changes live.
Alex: Sounds good, Sam. Just let us know a little in advance for the call.
Rhea: Agreed. We can make time for that.
Sam: Perfect! Let's keep the momentum going. Reach out if there are any 
sudden issues or support needed. Have a productive day!
Alex: You too.
Rhea: Thanks, bye!"""
    llm = Nebula(nebula_api_key="<your_api_key>")  # type: ignore[arg-type]

    instruction = """Identify the main objectives mentioned in this 
conversation."""
    output = llm.invoke(f"{instruction}\n{conversation}")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_together.py ---

def test_together_call() -> None:
    """Test simple call to together."""
    llm = Together(
        model="togethercomputer/RedPajama-INCITE-7B-Base",
        temperature=0.2,
        max_tokens=250,
    )
    output = llm.invoke("Say foo:")

    assert llm._llm_type == "together"
    assert isinstance(output, str)

async def test_together_acall() -> None:
    """Test simple call to together."""
    llm = Together(
        model="togethercomputer/RedPajama-INCITE-7B-Base",
        temperature=0.2,
        max_tokens=250,
    )
    output = await llm.agenerate(["Say foo:"], stop=["bar"])

    assert llm._llm_type == "together"
    output_text = output.generations[0][0].text
    assert isinstance(output_text, str)
    assert output_text.count("bar") <= 1


# --- libs/community/tests/integration_tests/llms/test_tongyi.py ---

def test_tongyi_call() -> None:
    """Test valid call to tongyi."""
    llm = Tongyi()
    output = llm.invoke("who are you")
    assert isinstance(output, str)

def test_tongyi_generate() -> None:
    """Test valid call to tongyi."""
    llm = Tongyi()
    output = llm.generate(["who are you"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)

def test_tongyi_generate_stream() -> None:
    """Test valid call to tongyi."""
    llm = Tongyi(streaming=True)
    output = llm.generate(["who are you"])
    print(output)  # noqa: T201
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


# --- libs/community/tests/integration_tests/llms/test_vertexai.py ---

def test_vertex_call(model_name: str) -> None:
    llm = (
        VertexAI(model_name=model_name, temperature=0)
        if model_name
        else VertexAI(temperature=0.0)
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_vertex_generate() -> None:
    llm = VertexAI(temperature=0.3, n=2, model_name="text-bison@001")
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == 2

def test_vertex_generate_code() -> None:
    llm = VertexAI(temperature=0.3, n=2, model_name="code-bison@001")
    output = llm.generate(["generate a python method that says foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == 2

async def test_vertex_agenerate() -> None:
    llm = VertexAI(temperature=0)
    output = await llm.agenerate(["Please say foo:"])
    assert isinstance(output, LLMResult)

def test_vertex_stream(model_name: str) -> None:
    llm = (
        VertexAI(temperature=0, model_name=model_name)
        if model_name
        else VertexAI(temperature=0)
    )
    outputs = list(llm.stream("Please say foo:"))
    assert isinstance(outputs[0], str)

async def test_vertex_consistency() -> None:
    llm = VertexAI(temperature=0)
    output = llm.generate(["Please say foo:"])
    streaming_output = llm.generate(["Please say foo:"], stream=True)
    async_output = await llm.agenerate(["Please say foo:"])
    assert output.generations[0][0].text == streaming_output.generations[0][0].text
    assert output.generations[0][0].text == async_output.generations[0][0].text

def test_model_garden(
    endpoint_os_variable_name: str, result_arg: Optional[str]
) -> None:
    """In order to run this test, you should provide endpoint names.

    Example:
    export FALCON_ENDPOINT_ID=...
    export LLAMA_ENDPOINT_ID=...
    export PROJECT=...
    """
    endpoint_id = os.environ[endpoint_os_variable_name]
    project = os.environ["PROJECT"]
    location = "europe-west4"
    llm = VertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        result_arg=result_arg,
        location=location,
    )
    output = llm.invoke("What is the meaning of life?")
    assert isinstance(output, str)
    assert llm._llm_type == "vertexai_model_garden"

def test_model_garden_generate(
    endpoint_os_variable_name: str, result_arg: Optional[str]
) -> None:
    """In order to run this test, you should provide endpoint names.

    Example:
    export FALCON_ENDPOINT_ID=...
    export LLAMA_ENDPOINT_ID=...
    export PROJECT=...
    """
    endpoint_id = os.environ[endpoint_os_variable_name]
    project = os.environ["PROJECT"]
    location = "europe-west4"
    llm = VertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        result_arg=result_arg,
        location=location,
    )
    output = llm.generate(["What is the meaning of life?", "How much is 2+2"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 2

async def test_model_garden_agenerate(
    endpoint_os_variable_name: str, result_arg: Optional[str]
) -> None:
    endpoint_id = os.environ[endpoint_os_variable_name]
    project = os.environ["PROJECT"]
    location = "europe-west4"
    llm = VertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        result_arg=result_arg,
        location=location,
    )
    output = await llm.agenerate(["What is the meaning of life?", "How much is 2+2"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 2


# --- libs/community/tests/integration_tests/llms/test_volcengine_maas.py ---

def test_default_call() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLM()
    output = llm.invoke("tell me a joke")
    assert isinstance(output, str)

def test_generate() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLM()
    output = llm.generate(["tell me a joke"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)

def test_generate_stream() -> None:
    """Test valid call to volc engine."""
    llm = VolcEngineMaasLLM(streaming=True)
    output = llm.stream("tell me a joke")
    assert isinstance(output, Generator)


# --- libs/community/tests/integration_tests/llms/test_watsonxllm.py ---

def test_watsonxllm_call() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        apikey="***",
        project_id="***",
    )
    response = watsonxllm.invoke("What color sunflower is?")
    assert isinstance(response, str)


# --- libs/community/tests/integration_tests/llms/test_weight_only_quantization.py ---

def test_weight_only_quantization_with_config() -> None:
    """Test valid call to HuggingFace text2text model."""
    from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig

    conf = WeightOnlyQuantConfig(weight_dtype="nf4")
    llm = WeightOnlyQuantPipeline.from_model_id(
        model_id=model_id, task="text2text-generation", quantization_config=conf
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_weight_only_quantization_4bit() -> None:
    """Test valid call to HuggingFace text2text model."""
    llm = WeightOnlyQuantPipeline.from_model_id(
        model_id=model_id, task="text2text-generation", load_in_4bit=True
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_weight_only_quantization_8bit() -> None:
    """Test valid call to HuggingFace text2text model."""
    llm = WeightOnlyQuantPipeline.from_model_id(
        model_id=model_id, task="text2text-generation", load_in_8bit=True
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

def test_init_with_pipeline() -> None:
    """Test initialization with a HF pipeline."""
    from intel_extension_for_transformers.transformers import AutoModelForSeq2SeqLM
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, load_in_4bit=True, use_llm_runtime=False
    )
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = WeightOnlyQuantPipeline(pipeline=pipe)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/llms/test_xinference.py ---

def test_xinference_llm_(setup: Tuple[str, str]) -> None:
    from xinference.client import RESTfulClient

    endpoint, _ = setup

    client = RESTfulClient(endpoint)

    model_uid = client.launch_model(
        model_name="vicuna-v1.3", model_size_in_billions=7, quantization="q4_0"
    )

    llm = Xinference(server_url=endpoint, model_uid=model_uid)

    answer = llm.invoke("Q: What food can we try in the capital of France? A:")

    assert isinstance(answer, str)

    answer = llm.invoke(
        "Q: where can we visit in the capital of France? A:",
        generate_config={"max_tokens": 1024, "stream": True},
    )

    assert isinstance(answer, str)


# --- libs/community/tests/integration_tests/llms/test_yuan2.py ---

def test_yuan2_call_method() -> None:
    """Test valid call to Yuan2.0."""
    llm = Yuan2(
        infer_api="http://127.0.0.1:8000/yuan",
        max_tokens=1024,
        temp=1.0,
        top_p=0.9,
        use_history=False,
    )
    output = llm.invoke("写一段快速排序算法。")
    assert isinstance(output, str)

def test_yuan2_generate_method() -> None:
    """Test valid call to Yuan2.0 inference api."""
    llm = Yuan2(
        infer_api="http://127.0.0.1:8000/yuan",
        max_tokens=1024,
        temp=1.0,
        top_p=0.9,
        use_history=False,
    )
    output = llm.generate(["who are you?"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


# --- libs/community/tests/integration_tests/memory/test_xata.py ---

    def test_xata_chat_memory(self) -> None:
        message_history = XataChatMessageHistory(
            api_key=os.getenv("XATA_API_KEY", ""),
            db_url=os.getenv("XATA_DB_URL", ""),
            session_id="integration-test-session",
        )
        memory = ConversationBufferMemory(
            memory_key="baz", chat_memory=message_history, return_messages=True
        )
        # add some messages
        memory.chat_memory.add_ai_message("This is me, the AI")
        memory.chat_memory.add_user_message("This is me, the human")

        # get the message history from the memory store and turn it into a json
        messages = memory.chat_memory.messages
        messages_json = json.dumps([message_to_dict(msg) for msg in messages])

        assert "This is me, the AI" in messages_json
        assert "This is me, the human" in messages_json

        # remove the record from Redis, so the next test run won't pick it up
        memory.chat_memory.clear()


# --- libs/community/tests/integration_tests/prompts/test_ngram_overlap_example_selector.py ---

def test_ngram_overlap_score(selector: NGramOverlapExampleSelector) -> None:
    """Tests that ngram_overlap_score returns correct values."""
    selector.threshold = 1.0 + 1e-9
    none = ngram_overlap_score(["Spot can run."], ["My dog barks."])
    some = ngram_overlap_score(["Spot can run."], ["See Spot run."])
    complete = ngram_overlap_score(["Spot can run."], ["Spot can run."])

    check = [abs(none - 0.0) < 1e-9, 0.0 < some < 1.0, abs(complete - 1.0) < 1e-9]
    assert check == [True, True, True]


# --- libs/community/tests/integration_tests/retrievers/test_arxiv.py ---

def test_load_success(retriever: ArxivRetriever) -> None:
    docs = retriever.invoke("1605.08386")
    assert len(docs) == 1
    assert_docs(docs, all_meta=False)

def test_load_success_all_meta(retriever: ArxivRetriever) -> None:
    retriever.load_all_available_meta = True
    retriever.load_max_docs = 2
    docs = retriever.invoke("ChatGPT")
    assert len(docs) > 1
    assert_docs(docs, all_meta=True)

def test_load_success_init_args() -> None:
    retriever = ArxivRetriever(load_max_docs=1, load_all_available_meta=True)  # type: ignore[call-arg]
    docs = retriever.invoke("ChatGPT")
    assert len(docs) == 1
    assert_docs(docs, all_meta=True)

def test_load_no_result(retriever: ArxivRetriever) -> None:
    docs = retriever.invoke("1605.08386WWW")
    assert not docs


# --- libs/community/tests/integration_tests/retrievers/test_azure_ai_search.py ---

def test_azure_ai_search_invoke() -> None:
    """Test valid call to Azure AI Search.

    In order to run this test, you should provide
    a `service_name`, an 'index_name' and
    an azure search `api_key` or 'azure_ad_token'
    as arguments for the AzureAISearchRetriever in both tests.
    api_version, aiosession and topk_k are optional parameters.
    """
    retriever = AzureAISearchRetriever()

    documents = retriever.invoke("what is langchain?")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content

    retriever = AzureAISearchRetriever(top_k=1)
    documents = retriever.invoke("what is langchain?")
    assert len(documents) <= 1

async def test_azure_ai_search_ainvoke() -> None:
    """Test valid async call to Azure AI Search.

    In order to run this test, you should provide
    a `service_name`, an 'index_name' and
    an azure search `api_key` or 'azure_ad_token'
    as arguments for the AzureAISearchRetriever.
    """
    retriever = AzureAISearchRetriever()
    documents = await retriever.ainvoke("what is langchain?")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content

def test_azure_cognitive_search_invoke() -> None:
    """Test valid call to Azure Cognitive Search.

    This is to test backwards compatibility of the retriever
    """
    retriever = AzureCognitiveSearchRetriever()

    documents = retriever.invoke("what is langchain?")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content

    retriever = AzureCognitiveSearchRetriever(top_k=1)
    documents = retriever.invoke("what is langchain?")
    assert len(documents) <= 1

async def test_azure_cognitive_search_ainvoke() -> None:
    """Test valid async call to Azure Cognitive Search.

    This is to test backwards compatibility of the retriever
    """
    retriever = AzureCognitiveSearchRetriever()
    documents = await retriever.ainvoke("what is langchain?")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content


# --- libs/community/tests/integration_tests/retrievers/test_breebs.py ---

    def test_breeb_query(self) -> None:
        breeb_key = "Parivoyage"
        query = "What are the best churches to visit in Paris?"
        breeb_retriever = BreebsRetriever(breeb_key)
        documents: List[Document] = breeb_retriever.invoke(query)
        assert isinstance(documents, list), "Documents should be a list"
        for doc in documents:
            assert doc.page_content, "Document page_content should not be None"
            assert doc.metadata["source"], "Document metadata should contain 'source'"
            assert doc.metadata["score"] == 1, "Document score should be equal to 1"


# --- libs/community/tests/integration_tests/retrievers/test_contextual_compression.py ---

def test_contextual_compression_retriever_get_relevant_docs() -> None:
    """Test get_relevant_docs."""
    texts = [
        "This is a document about the Boston Celtics",
        "The Boston Celtics won the game by 20 points",
        "I simply love going to the movies",
    ]
    embeddings = OpenAIEmbeddings()
    base_compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
    base_retriever = FAISS.from_texts(texts, embedding=embeddings).as_retriever(
        search_kwargs={"k": len(texts)}
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=base_compressor, base_retriever=base_retriever
    )

    actual = retriever.invoke("Tell me about the Celtics")
    assert len(actual) == 2
    assert texts[-1] not in [d.page_content for d in actual]

async def test_acontextual_compression_retriever_get_relevant_docs() -> None:
    """Test get_relevant_docs."""
    texts = [
        "This is a document about the Boston Celtics",
        "The Boston Celtics won the game by 20 points",
        "I simply love going to the movies",
    ]
    embeddings = OpenAIEmbeddings()
    base_compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
    base_retriever = FAISS.from_texts(texts, embedding=embeddings).as_retriever(
        search_kwargs={"k": len(texts)}
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=base_compressor, base_retriever=base_retriever
    )

    actual = retriever.invoke("Tell me about the Celtics")
    assert len(actual) == 2
    assert texts[-1] not in [d.page_content for d in actual]


# --- libs/community/tests/integration_tests/retrievers/test_dria_index.py ---

def test_dria_retriever(dria_retriever: DriaRetriever) -> None:
    texts = [
        {
            "text": "Langchain",
            "metadata": {
                "source": "source#1",
                "document_id": "doc123",
                "content": "Langchain",
            },
        }
    ]
    dria_retriever.add_texts(texts)

    # Assuming invoke returns a list of Document instances
    docs = dria_retriever.invoke("Langchain")

    # Perform assertions
    assert len(docs) > 0, "Expected at least one document"
    doc = docs[0]
    assert isinstance(doc, Document), "Expected a Document instance"
    assert isinstance(doc.page_content, str), (
        "Expected document content type to be string"
    )
    assert isinstance(doc.metadata, dict), (
        "Expected document metadata content to be a dictionary"
    )


# --- libs/community/tests/integration_tests/retrievers/test_google_docai_warehoure_retriever.py ---

def test_google_documentai_warehoure_retriever() -> None:
    """In order to run this test, you should provide a project_id and user_ldap.

    Example:
    export USER_LDAP=...
    export PROJECT_NUMBER=...
    """
    project_number = os.environ["PROJECT_NUMBER"]
    user_ldap = os.environ["USER_LDAP"]
    docai_wh_retriever = GoogleDocumentAIWarehouseRetriever(
        project_number=project_number
    )
    documents = docai_wh_retriever.invoke(
        "What are Alphabet's Other Bets?", user_ldap=user_ldap
    )
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)


# --- libs/community/tests/integration_tests/retrievers/test_google_vertex_ai_search.py ---

def test_google_vertex_ai_search_invoke() -> None:
    """Test the invoke() method."""
    retriever = GoogleVertexAISearchRetriever()
    documents = retriever.invoke("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]

def test_google_vertex_ai_multiturnsearch_invoke() -> None:
    """Test the invoke() method."""
    retriever = GoogleVertexAIMultiTurnSearchRetriever()
    documents = retriever.invoke("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]

def test_google_vertex_ai_search_enterprise_search_deprecation() -> None:
    """Test the deprecation of GoogleCloudEnterpriseSearchRetriever."""
    with pytest.warns(
        DeprecationWarning,
        match="GoogleCloudEnterpriseSearchRetriever is deprecated, use GoogleVertexAISearchRetriever",  # noqa: E501
    ):
        retriever = GoogleCloudEnterpriseSearchRetriever()

    os.environ["SEARCH_ENGINE_ID"] = os.getenv("DATA_STORE_ID", "data_store_id")
    with pytest.warns(
        DeprecationWarning,
        match="The `search_engine_id` parameter is deprecated. Use `data_store_id` instead.",  # noqa: E501
    ):
        retriever = GoogleCloudEnterpriseSearchRetriever()

    # Check that mapped methods still work.
    documents = retriever.invoke("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]


# --- libs/community/tests/integration_tests/retrievers/test_kay.py ---

def test_kay_retriever() -> None:
    retriever = KayAiRetriever.create(
        dataset_id="company",
        data_types=["10-K", "10-Q", "8-K", "PressRelease"],
        num_contexts=3,
    )
    docs = retriever.invoke(
        "What were the biggest strategy changes and partnerships made by Roku in 2023?",
    )
    assert len(docs) == 3
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata
        assert len(list(doc.metadata.items())) > 0


# --- libs/community/tests/integration_tests/retrievers/test_merger_retriever.py ---

def test_merger_retriever_get_relevant_docs() -> None:
    """Test get_relevant_docs."""
    texts_group_a = [
        "This is a document about the Boston Celtics",
        "Fly me to the moon is one of my favourite songs."
        "I simply love going to the movies",
    ]
    texts_group_b = [
        "This is a document about the Poenix Suns",
        "The Boston Celtics won the game by 20 points",
        "Real stupidity beats artificial intelligence every time. TP",
    ]
    embeddings = OpenAIEmbeddings()
    retriever_a = InMemoryVectorStore.from_texts(
        texts_group_a, embedding=embeddings
    ).as_retriever(search_kwargs={"k": 1})
    retriever_b = InMemoryVectorStore.from_texts(
        texts_group_b, embedding=embeddings
    ).as_retriever(search_kwargs={"k": 1})

    # The Lord of the Retrievers.
    lotr = MergerRetriever(retrievers=[retriever_a, retriever_b])

    actual = lotr.invoke("Tell me about the Celtics")
    assert len(actual) == 2
    assert texts_group_a[0] in [d.page_content for d in actual]
    assert texts_group_b[1] in [d.page_content for d in actual]


# --- libs/community/tests/integration_tests/retrievers/test_pubmed.py ---

def test_load_success(retriever: PubMedRetriever) -> None:
    docs = retriever.invoke("chatgpt")
    assert len(docs) == 3
    assert_docs(docs)

def test_load_success_top_k_results(retriever: PubMedRetriever) -> None:
    retriever.top_k_results = 2
    docs = retriever.invoke("chatgpt")
    assert len(docs) == 2
    assert_docs(docs)

def test_load_no_result(retriever: PubMedRetriever) -> None:
    docs = retriever.invoke("1605.08386WWW")
    assert not docs


# --- libs/community/tests/integration_tests/retrievers/test_thirdai_neuraldb.py ---

def test_neuraldb_retriever_from_scratch(test_csv: str) -> None:
    retriever = NeuralDBRetriever.from_scratch()
    retriever.insert([test_csv])
    documents = retriever.invoke("column")
    assert_result_correctness(documents)

def test_neuraldb_retriever_from_checkpoint(test_csv: str) -> None:
    checkpoint = "thirdai-test-save.ndb"
    if os.path.exists(checkpoint):
        shutil.rmtree(checkpoint)
    try:
        retriever = NeuralDBRetriever.from_scratch()
        retriever.insert([test_csv])
        retriever.save(checkpoint)
        loaded_retriever = NeuralDBRetriever.from_checkpoint(checkpoint)
        documents = loaded_retriever.invoke("column")
        assert_result_correctness(documents)
    finally:
        if os.path.exists(checkpoint):
            shutil.rmtree(checkpoint)


# --- libs/community/tests/integration_tests/retrievers/test_weaviate_hybrid_search.py ---

    def test_invoke(self, weaviate_url: str) -> None:
        """Test end to end construction and MRR search."""
        from weaviate import Client

        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]

        client = Client(weaviate_url)

        retriever = WeaviateHybridSearchRetriever(
            client=client,
            index_name=f"LangChain_{uuid4().hex}",
            text_key="text",
            attributes=["page"],
        )
        for i, text in enumerate(texts):
            retriever.add_documents(
                [Document(page_content=text, metadata=metadatas[i])]
            )

        output = retriever.invoke("foo")
        assert output == [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="baz", metadata={"page": 2}),
            Document(page_content="bar", metadata={"page": 1}),
        ]

    def test_invoke_with_score(self, weaviate_url: str) -> None:
        """Test end to end construction and MRR search."""
        from weaviate import Client

        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]

        client = Client(weaviate_url)

        retriever = WeaviateHybridSearchRetriever(
            client=client,
            index_name=f"LangChain_{uuid4().hex}",
            text_key="text",
            attributes=["page"],
        )
        for i, text in enumerate(texts):
            retriever.add_documents(
                [Document(page_content=text, metadata=metadatas[i])]
            )

        output = retriever.invoke("foo", score=True)
        for doc in output:
            assert "_additional" in doc.metadata

    def test_invoke_with_filter(self, weaviate_url: str) -> None:
        """Test end to end construction and MRR search."""
        from weaviate import Client

        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]

        client = Client(weaviate_url)

        retriever = WeaviateHybridSearchRetriever(
            client=client,
            index_name=f"LangChain_{uuid4().hex}",
            text_key="text",
            attributes=["page"],
        )
        for i, text in enumerate(texts):
            retriever.add_documents(
                [Document(page_content=text, metadata=metadatas[i])]
            )

        where_filter = {"path": ["page"], "operator": "Equal", "valueNumber": 0}

        output = retriever.invoke("foo", where_filter=where_filter)
        assert output == [
            Document(page_content="foo", metadata={"page": 0}),
        ]

    def test_invoke_with_uuids(self, weaviate_url: str) -> None:
        """Test end to end construction and MRR search."""
        from weaviate import Client

        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        # Weaviate replaces the object if the UUID already exists
        uuids = [uuid.uuid5(uuid.NAMESPACE_DNS, "same-name") for text in texts]

        client = Client(weaviate_url)

        retriever = WeaviateHybridSearchRetriever(
            client=client,
            index_name=f"LangChain_{uuid4().hex}",
            text_key="text",
            attributes=["page"],
        )
        for i, text in enumerate(texts):
            retriever.add_documents(
                [Document(page_content=text, metadata=metadatas[i])], uuids=[uuids[i]]
            )

        output = retriever.invoke("foo")
        assert len(output) == 1


# --- libs/community/tests/integration_tests/retrievers/test_wikipedia.py ---

def test_load_success(retriever: WikipediaRetriever) -> None:
    docs = retriever.invoke("HUNTER X HUNTER")
    assert len(docs) > 1
    assert len(docs) <= 3
    assert_docs(docs, all_meta=False)

def test_load_success_all_meta(retriever: WikipediaRetriever) -> None:
    retriever.load_all_available_meta = True
    docs = retriever.invoke("HUNTER X HUNTER")
    assert len(docs) > 1
    assert len(docs) <= 3
    assert_docs(docs, all_meta=True)

def test_load_success_init_args() -> None:
    retriever = WikipediaRetriever(  # type: ignore[call-arg]
        lang="en", top_k_results=1, load_all_available_meta=True
    )
    docs = retriever.invoke("HUNTER X HUNTER")
    assert len(docs) == 1
    assert_docs(docs, all_meta=True)

def test_load_success_init_args_more() -> None:
    retriever = WikipediaRetriever(  # type: ignore[call-arg]
        lang="en", top_k_results=20, load_all_available_meta=False
    )
    docs = retriever.invoke("HUNTER X HUNTER")
    assert len(docs) == 20
    assert_docs(docs, all_meta=False)

def test_load_no_result(retriever: WikipediaRetriever) -> None:
    docs = retriever.invoke(
        "NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL"
    )
    assert not docs


# --- libs/community/tests/integration_tests/retrievers/test_you.py ---

    def test_invoke(self) -> None:
        retriever = YouRetriever()
        actual = retriever.invoke("test")

        assert len(actual) > 0


# --- libs/community/tests/integration_tests/retrievers/test_zep.py ---

def test_zep_retriever_invoke(
    zep_retriever: ZepRetriever, search_results: List[MemorySearchResult]
) -> None:
    documents: List[Document] = zep_retriever.invoke("My trip to Iceland")
    _test_documents(documents, search_results)

async def test_zep_retriever_ainvoke(
    zep_retriever: ZepRetriever, search_results: List[MemorySearchResult]
) -> None:
    documents: List[Document] = await zep_retriever.ainvoke("My trip to Iceland")
    _test_documents(documents, search_results)


# --- libs/community/tests/integration_tests/smith/evaluation/test_runner_utils.py ---

def test_chat_model(
    kv_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = ChatOpenAI(temperature=0)
    eval_config = RunEvalConfig(
        evaluators=[EvaluatorType.QA], custom_evaluators=[not_empty]
    )
    with pytest.raises(ValueError, match="Must specify reference_key"):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=llm,
            evaluation=eval_config,
            client=client,
        )
    eval_config = RunEvalConfig(
        evaluators=[EvaluatorType.QA],
        reference_key="some_output",
    )
    with pytest.raises(
        InputFormatError, match="Example inputs do not match language model"
    ):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=llm,
            evaluation=eval_config,
            client=client,
        )

    def input_mapper(d: dict) -> List[BaseMessage]:
        return [HumanMessage(content=d["some_input"])]

    run_on_dataset(
        client=client,
        dataset_name=kv_dataset_name,
        llm_or_chain_factory=input_mapper | llm,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)

def test_llm(kv_dataset_name: str, eval_project_name: str, client: Client) -> None:
    llm = OpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA])
    with pytest.raises(ValueError, match="Must specify reference_key"):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=llm,
            evaluation=eval_config,
            client=client,
        )
    eval_config = RunEvalConfig(
        evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA],
        reference_key="some_output",
    )
    with pytest.raises(InputFormatError, match="Example inputs"):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=llm,
            evaluation=eval_config,
            client=client,
        )

    def input_mapper(d: dict) -> str:
        return d["some_input"]

    run_on_dataset(
        client=client,
        dataset_name=kv_dataset_name,
        llm_or_chain_factory=input_mapper | llm,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)

def test_chain(kv_dataset_name: str, eval_project_name: str, client: Client) -> None:
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(llm, "The answer to the {question} is: ")
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(ValueError, match="Must specify reference_key"):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=lambda: chain,
            evaluation=eval_config,
            client=client,
        )
    eval_config = RunEvalConfig(
        evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA],
        reference_key="some_output",
    )
    with pytest.raises(InputFormatError, match="Example inputs"):
        run_on_dataset(
            dataset_name=kv_dataset_name,
            llm_or_chain_factory=lambda: chain,
            evaluation=eval_config,
            client=client,
        )

    eval_config = RunEvalConfig(
        custom_evaluators=[not_empty],
    )

    def right_input_mapper(d: dict) -> dict:
        return {"question": d["some_input"]}

    run_on_dataset(
        dataset_name=kv_dataset_name,
        llm_or_chain_factory=lambda: right_input_mapper | chain,
        client=client,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)

def test_chat_model_on_chat_dataset(
    chat_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = ChatOpenAI(temperature=0)
    eval_config = RunEvalConfig(custom_evaluators=[not_empty])
    run_on_dataset(
        dataset_name=chat_dataset_name,
        llm_or_chain_factory=llm,
        evaluation=eval_config,
        client=client,
        project_name=eval_project_name,
    )
    _check_all_feedback_passed(eval_project_name, client)

def test_llm_on_chat_dataset(
    chat_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = OpenAI(temperature=0)
    eval_config = RunEvalConfig(custom_evaluators=[not_empty])
    run_on_dataset(
        dataset_name=chat_dataset_name,
        llm_or_chain_factory=llm,
        client=client,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)

def test_chain_on_chat_dataset(chat_dataset_name: str, client: Client) -> None:
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(llm, "The answer to the {question} is: ")
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(
        ValueError, match="Cannot evaluate a chain on dataset with data_type=chat"
    ):
        run_on_dataset(
            dataset_name=chat_dataset_name,
            client=client,
            llm_or_chain_factory=lambda: chain,
            evaluation=eval_config,
        )

def test_chat_model_on_llm_dataset(
    llm_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = ChatOpenAI(temperature=0)
    eval_config = RunEvalConfig(custom_evaluators=[not_empty])
    run_on_dataset(
        client=client,
        dataset_name=llm_dataset_name,
        llm_or_chain_factory=llm,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)

def test_llm_on_llm_dataset(
    llm_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = OpenAI(temperature=0)
    eval_config = RunEvalConfig(custom_evaluators=[not_empty])
    run_on_dataset(
        client=client,
        dataset_name=llm_dataset_name,
        llm_or_chain_factory=llm,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)

def test_chain_on_llm_dataset(llm_dataset_name: str, client: Client) -> None:
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(llm, "The answer to the {question} is: ")
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    with pytest.raises(
        ValueError, match="Cannot evaluate a chain on dataset with data_type=llm"
    ):
        run_on_dataset(
            client=client,
            dataset_name=llm_dataset_name,
            llm_or_chain_factory=lambda: chain,
            evaluation=eval_config,
        )

def test_chat_model_on_kv_singleio_dataset(
    kv_singleio_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = ChatOpenAI(temperature=0)
    eval_config = RunEvalConfig(evaluators=[EvaluatorType.QA, EvaluatorType.CRITERIA])
    run_on_dataset(
        dataset_name=kv_singleio_dataset_name,
        llm_or_chain_factory=llm,
        evaluation=eval_config,
        client=client,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)

def test_llm_on_kv_singleio_dataset(
    kv_singleio_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = OpenAI(temperature=0)
    eval_config = RunEvalConfig(custom_evaluators=[not_empty])
    run_on_dataset(
        dataset_name=kv_singleio_dataset_name,
        llm_or_chain_factory=llm,
        client=client,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)

def test_chain_on_kv_singleio_dataset(
    kv_singleio_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(llm, "The answer to the {question} is: ")
    eval_config = RunEvalConfig(custom_evaluators=[not_empty])
    run_on_dataset(
        dataset_name=kv_singleio_dataset_name,
        llm_or_chain_factory=lambda: chain,
        client=client,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)

async def test_runnable_on_kv_singleio_dataset(
    kv_singleio_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    runnable = (
        ChatPromptTemplate.from_messages([("human", "{the wackiest input}")])
        | ChatOpenAI()
    )
    eval_config = RunEvalConfig(custom_evaluators=[not_empty])
    await arun_on_dataset(
        dataset_name=kv_singleio_dataset_name,
        llm_or_chain_factory=runnable,
        client=client,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)

async def test_arb_func_on_kv_singleio_dataset(
    kv_singleio_dataset_name: str, eval_project_name: str, client: Client
) -> None:
    runnable = (
        ChatPromptTemplate.from_messages([("human", "{the wackiest input}")])
        | ChatOpenAI()
    )

    def my_func(x: dict) -> str:
        content = runnable.invoke(x).content
        if isinstance(content, str):
            return content
        else:
            raise ValueError(
                f"Expected message with content type string, got {content}"
            )

    eval_config = RunEvalConfig(custom_evaluators=[not_empty])
    await arun_on_dataset(
        dataset_name=kv_singleio_dataset_name,
        llm_or_chain_factory=my_func,
        client=client,
        evaluation=eval_config,
        project_name=eval_project_name,
        tags=["shouldpass"],
    )
    _check_all_feedback_passed(eval_project_name, client)


# --- libs/community/tests/integration_tests/tools/test_yahoo_finance_news.py ---

def test_success() -> None:
    """Test that the tool runs successfully."""
    tool = YahooFinanceNewsTool()
    query = "AAPL"
    result = tool.run(query)
    assert result is not None
    assert f"Company ticker {query} not found." not in result

def test_failure_no_ticker() -> None:
    """Test that the tool fails."""
    tool = YahooFinanceNewsTool()
    query = ""
    result = tool.run(query)
    assert f"Company ticker {query} not found." in result

def test_failure_wrong_ticker() -> None:
    """Test that the tool fails."""
    tool = YahooFinanceNewsTool()
    query = "NOT_A_COMPANY"
    result = tool.run(query)
    assert f"Company ticker {query} not found." in result


# --- libs/community/tests/integration_tests/tools/connery/test_service.py ---

def test_run_action_with_no_iput() -> None:
    """Test for running Connery Action without input."""
    connery = ConneryService()
    # refreshPluginCache action from connery-io/connery-runner-administration plugin
    output = connery._run_action("CAF979E6D2FF4C8B946EEBAFCB3BA475")
    assert output is not None
    assert output == {}

def test_run_action_with_iput() -> None:
    """Test for running Connery Action with input."""
    connery = ConneryService()
    # summarizePublicWebpage action from connery-io/summarization-plugin plugin
    output = connery._run_action(
        "CA72DFB0AB4DF6C830B43E14B0782F70",
        {"publicWebpageUrl": "http://www.paulgraham.com/vb.html"},
    )
    assert output is not None
    assert output["summary"] is not None
    assert len(output["summary"]) > 0


# --- libs/community/tests/integration_tests/tools/edenai/test_audio_speech_to_text.py ---

def test_edenai_call() -> None:
    """Test simple call to edenai's speech to text endpoint."""
    speech2text = EdenAiSpeechToTextTool(providers=["amazon"])  # type: ignore[call-arg]

    output = speech2text.invoke(
        "https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-0.mp3"
    )

    assert speech2text.name == "edenai_speech_to_text"
    assert speech2text.feature == "audio"
    assert speech2text.subfeature == "speech_to_text_async"
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/tools/edenai/test_audio_text_to_speech.py ---

def test_edenai_call() -> None:
    """Test simple call to edenai's text to speech endpoint."""
    text2speech = EdenAiTextToSpeechTool(
        providers=["amazon"], language="en", voice="MALE"
    )

    output = text2speech.invoke("hello")
    parsed_url = urlparse(output)

    assert text2speech.name == "edenai_text_to_speech"
    assert text2speech.feature == "audio"
    assert text2speech.subfeature == "text_to_speech"
    assert isinstance(output, str)
    assert parsed_url.scheme in ["http", "https"]


# --- libs/community/tests/integration_tests/tools/edenai/test_image_explicitcontent.py ---

def test_edenai_call() -> None:
    """Test simple call to edenai's image moderation endpoint."""
    image_moderation = EdenAiExplicitImageTool(providers=["amazon"])

    output = image_moderation.invoke("https://static.javatpoint.com/images/objects.jpg")

    assert image_moderation.name == "edenai_image_explicit_content_detection"
    assert image_moderation.feature == "image"
    assert image_moderation.subfeature == "explicit_content"
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/tools/edenai/test_image_objectdetection.py ---

def test_edenai_call() -> None:
    """Test simple call to edenai's object detection endpoint."""
    object_detection = EdenAiObjectDetectionTool(providers=["google"])

    output = object_detection.invoke("https://static.javatpoint.com/images/objects.jpg")

    assert object_detection.name == "edenai_object_detection"
    assert object_detection.feature == "image"
    assert object_detection.subfeature == "object_detection"
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/tools/edenai/test_ocr_identityparser.py ---

def test_edenai_call() -> None:
    """Test simple call to edenai's identity parser endpoint."""
    id_parser = EdenAiParsingIDTool(providers=["amazon"], language="en")

    output = id_parser.invoke(
        "https://www.citizencard.com/images/citizencard-uk-id-card-2023.jpg"
    )

    assert id_parser.name == "edenai_identity_parsing"
    assert id_parser.feature == "ocr"
    assert id_parser.subfeature == "identity_parser"
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/tools/edenai/test_ocr_invoiceparser.py ---

def test_edenai_call() -> None:
    """Test simple call to edenai's invoice parser endpoint."""
    invoice_parser = EdenAiParsingInvoiceTool(providers=["amazon"], language="en")

    output = invoice_parser.invoke(
        "https://app.edenai.run/assets/img/data_1.72e3bdcc.png"
    )

    assert invoice_parser.name == "edenai_invoice_parsing"
    assert invoice_parser.feature == "ocr"
    assert invoice_parser.subfeature == "invoice_parser"
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/tools/edenai/test_text_moderation.py ---

def test_edenai_call() -> None:
    """Test simple call to edenai's text moderation endpoint."""

    text_moderation = EdenAiTextModerationTool(providers=["openai"], language="en")

    output = text_moderation.invoke("i hate you")

    assert text_moderation.name == "edenai_explicit_content_detection_text"
    assert text_moderation.feature == "text"
    assert text_moderation.subfeature == "moderation"
    assert isinstance(output, str)


# --- libs/community/tests/integration_tests/tools/zenguard/test_zenguard.py ---

def test_prompt_injection(zenguard_tool: ZenGuardTool) -> None:
    prompt = "Simple prompt injection test"
    detectors = [Detector.PROMPT_INJECTION]
    response = zenguard_tool.run({"detectors": detectors, "prompts": [prompt]})
    assert_successful_response_not_detected(response)

def test_pii(zenguard_tool: ZenGuardTool) -> None:
    prompt = "Simple PII test"
    detectors = [Detector.PII]
    response = zenguard_tool.run({"detectors": detectors, "prompts": [prompt]})
    assert_successful_response_not_detected(response)

def test_allowed_topics(zenguard_tool: ZenGuardTool) -> None:
    prompt = "Simple allowed topics test"
    detectors = [Detector.ALLOWED_TOPICS]
    response = zenguard_tool.run({"detectors": detectors, "prompts": [prompt]})
    assert_successful_response_not_detected(response)

def test_banned_topics(zenguard_tool: ZenGuardTool) -> None:
    prompt = "Simple banned topics test"
    detectors = [Detector.BANNED_TOPICS]
    response = zenguard_tool.run({"detectors": detectors, "prompts": [prompt]})
    assert_successful_response_not_detected(response)

def test_keywords(zenguard_tool: ZenGuardTool) -> None:
    prompt = "Simple keywords test"
    detectors = [Detector.KEYWORDS]
    response = zenguard_tool.run({"detectors": detectors, "prompts": [prompt]})
    assert_successful_response_not_detected(response)

def test_secrets(zenguard_tool: ZenGuardTool) -> None:
    prompt = "Simple secrets test"
    detectors = [Detector.SECRETS]
    response = zenguard_tool.run({"detectors": detectors, "prompts": [prompt]})
    assert_successful_response_not_detected(response)

def test_toxicity(zenguard_tool: ZenGuardTool) -> None:
    prompt = "Simple toxicity test"
    detectors = [Detector.TOXICITY]
    response = zenguard_tool.run({"detectors": detectors, "prompts": [prompt]})
    assert_successful_response_not_detected(response)

def test_all_detectors(zenguard_tool: ZenGuardTool) -> None:
    prompt = "Simple all detectors test"
    detectors = [
        Detector.ALLOWED_TOPICS,
        Detector.BANNED_TOPICS,
        Detector.KEYWORDS,
        Detector.PII,
        Detector.PROMPT_INJECTION,
        Detector.SECRETS,
        Detector.TOXICITY,
    ]
    response = zenguard_tool.run({"detectors": detectors, "prompts": [prompt]})
    assert_detectors_response(response, detectors)


# --- libs/community/tests/integration_tests/utilities/test_alpha_vantage.py ---

def test_run_method(api_wrapper: AlphaVantageAPIWrapper) -> None:
    """Test the run method for successful response."""
    response = api_wrapper.run("USD", "EUR")
    assert response is not None
    assert isinstance(response, dict)


# --- libs/community/tests/integration_tests/utilities/test_arxiv.py ---

def test_run_success_paper_name(api_client: ArxivAPIWrapper) -> None:
    """Test a query of paper name that returns the correct answer"""

    output = api_client.run("Heat-bath random walks with Markov bases")
    assert "Probability distributions for Markov chains based quantum walks" in output
    assert (
        "Transformations of random walks on groups via Markov stopping times" in output
    )
    assert (
        "Recurrence of Multidimensional Persistent Random Walks. Fourier and Series "
        "Criteria" in output
    )

def test_run_success_arxiv_identifier(api_client: ArxivAPIWrapper) -> None:
    """Test a query of an arxiv identifier returns the correct answer"""

    output = api_client.run("1605.08386v1")
    assert "Heat-bath random walks with Markov bases" in output

def test_run_success_multiple_arxiv_identifiers(api_client: ArxivAPIWrapper) -> None:
    """Test a query of multiple arxiv identifiers that returns the correct answer"""

    output = api_client.run("1605.08386v1 2212.00794v2 2308.07912")
    assert "Heat-bath random walks with Markov bases" in output
    assert "Scaling Language-Image Pre-training via Masking" in output
    assert (
        "Ultra-low mass PBHs in the early universe can explain the PTA signal" in output
    )

def test_run_returns_several_docs(api_client: ArxivAPIWrapper) -> None:
    """Test that returns several docs"""

    output = api_client.run("Caprice Stanley")
    assert "On Mixing Behavior of a Family of Random Walks" in output

def test_run_returns_no_result(api_client: ArxivAPIWrapper) -> None:
    """Test that gives no result."""

    output = api_client.run("1605.08386WWW")
    assert "No good Arxiv Result was found" == output

def test_load_arxiv_from_universal_entry() -> None:
    arxiv_tool = _load_arxiv_from_universal_entry()
    output = arxiv_tool.invoke("Caprice Stanley")
    assert "On Mixing Behavior of a Family of Random Walks" in output, (
        "failed to fetch a valid result"
    )

def test_load_arxiv_from_universal_entry_with_params() -> None:
    params = {
        "top_k_results": 1,
        "load_max_docs": 10,
        "load_all_available_meta": True,
    }
    arxiv_tool = _load_arxiv_from_universal_entry(**params)
    assert isinstance(arxiv_tool, ArxivQueryRun)
    wp = arxiv_tool.api_wrapper
    assert wp.top_k_results == 1, "failed to assert top_k_results"
    assert wp.load_max_docs == 10, "failed to assert load_max_docs"
    assert wp.load_all_available_meta is True, (
        "failed to assert load_all_available_meta"
    )


# --- libs/community/tests/integration_tests/utilities/test_bing_search.py ---

def test_call() -> None:
    """Test that call gives the correct answer."""
    search = BingSearchAPIWrapper()  # type: ignore[call-arg]
    output = search.run("Obama's first name")
    assert "Barack Hussein Obama" in output


# --- libs/community/tests/integration_tests/utilities/test_clickup.py ---

def test_folder_related(clickup_wrapper: ClickupAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    task_name = f"Test Folder - {time_str}"

    # Create Folder
    create_response = json.loads(
        clickup_wrapper.run(mode="create_folder", query=json.dumps({"name": task_name}))
    )
    assert create_response["name"] == task_name

def test_list_related(clickup_wrapper: ClickupAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    task_name = f"Test List - {time_str}"

    # Create List
    create_response = json.loads(
        clickup_wrapper.run(mode="create_list", query=json.dumps({"name": task_name}))
    )
    assert create_response["name"] == task_name

def test_task_related(clickup_wrapper: ClickupAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    task_name = f"Test Task - {time_str}"

    # Create task
    create_response = json.loads(
        clickup_wrapper.run(
            mode="create_task",
            query=json.dumps({"name": task_name, "description": "This is a Test"}),
        )
    )
    assert create_response["name"] == task_name

    # Get task
    task_id = create_response["id"]
    get_response = json.loads(
        clickup_wrapper.run(mode="get_task", query=json.dumps({"task_id": task_id}))
    )

    assert get_response["name"] == task_name

    # Update task
    new_name = f"{task_name} - New"
    clickup_wrapper.run(
        mode="update_task",
        query=json.dumps(
            {
                "task_id": task_id,
                "attribute_name": "name",
                "value": new_name,
            }
        ),
    )

    get_response_2 = json.loads(
        clickup_wrapper.run(mode="get_task", query=json.dumps({"task_id": task_id}))
    )
    assert get_response_2["name"] == new_name


# --- libs/community/tests/integration_tests/utilities/test_dataforseo_api.py ---

def test_search_call() -> None:
    search = DataForSeoAPIWrapper()
    output = search.run("pi value")
    assert "3.14159" in output

async def test_async_call() -> None:
    search = DataForSeoAPIWrapper()
    output = await search.arun("pi value")
    assert "3.14159" in output


# --- libs/community/tests/integration_tests/utilities/test_dataherald_api.py ---

def test_call() -> None:
    """Test that call gives the correct answer."""
    search = DataheraldAPIWrapper(db_connection_id="65fb766367dd22c99ce1a12d")
    output = search.run("How many employees are in the company?")
    assert "Answer: SELECT \n    COUNT(*) FROM \n    employees" in output


# --- libs/community/tests/integration_tests/utilities/test_duckduckdgo_search_api.py ---

def test_ddg_search_tool() -> None:
    keywords = "Bella Ciao"
    tool = DuckDuckGoSearchRun()
    result = tool.invoke(keywords)
    print(result)  # noqa: T201
    assert len(result.split()) > 20

def test_ddg_search_news_tool() -> None:
    keywords = "Tesla"
    tool = DuckDuckGoSearchResults(source="news")  # type: ignore[call-arg]
    result = tool.invoke(keywords)
    print(result)  # noqa: T201
    assert len(result.split()) > 20


# --- libs/community/tests/integration_tests/utilities/test_golden_query_api.py ---

def test_call() -> None:
    """Test that call gives the correct answer."""
    search = GoldenQueryAPIWrapper()
    output = json.loads(search.run("companies in nanotech"))
    assert len(output.get("results", [])) > 0


# --- libs/community/tests/integration_tests/utilities/test_googleserper_api.py ---

def test_search_call() -> None:
    """Test that call gives the correct answer from search."""
    search = GoogleSerperAPIWrapper()
    output = search.run("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output

def test_news_call() -> None:
    """Test that call gives the correct answer from news search."""
    search = GoogleSerperAPIWrapper(type="news")
    output = search.run("What's new with stock market?").lower()
    assert "stock" in output or "market" in output

async def test_async_call() -> None:
    """Test that call gives the correct answer."""
    search = GoogleSerperAPIWrapper()
    output = await search.arun("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output


# --- libs/community/tests/integration_tests/utilities/test_jira_api.py ---

def test_search() -> None:
    """Test for Searching issues on JIRA"""
    jql = "project = TP"
    jira = JiraAPIWrapper()
    output = jira.run("jql", jql)
    assert "issues" in output

def test_getprojects() -> None:
    """Test for getting projects on JIRA"""
    jira = JiraAPIWrapper()
    output = jira.run("get_projects", "")
    assert "projects" in output

def test_create_ticket() -> None:
    """Test the Create Ticket Call that Creates a Issue/Ticket on JIRA."""
    issue_string = (
        '{"summary": "Test Summary", "description": "Test Description",'
        ' "issuetype": {"name": "Bug"}, "project": {"key": "TP"}}'
    )
    jira = JiraAPIWrapper()
    output = jira.run("create_issue", issue_string)
    assert "id" in output
    assert "key" in output

def test_create_confluence_page() -> None:
    """Test for getting projects on JIRA"""
    jira = JiraAPIWrapper()
    create_page_dict = (
        '{"space": "ROC", "title":"This is the title",'
        '"body":"This is the body. You can use '
        '<strong>HTML tags</strong>!"}'
    )

    output = jira.run("create_page", create_page_dict)
    assert "type" in output
    assert "page" in output

def test_other() -> None:
    """Non-exhaustive test for accessing other JIRA API methods"""
    jira = JiraAPIWrapper()
    issue_create_dict = """
        {
            "function":"issue_create",
            "kwargs": {
                "fields": {
                    "summary": "Test Summary",
                    "description": "Test Description",
                    "issuetype": {"name": "Bug"},
                    "project": {"key": "TP"}
                }
            }
        }
    """
    output = jira.run("other", issue_create_dict)
    assert "id" in output
    assert "key" in output


# --- libs/community/tests/integration_tests/utilities/test_merriam_webster_api.py ---

def test_call(api_client: MerriamWebsterAPIWrapper) -> None:
    """Test that call gives correct answer."""
    output = api_client.run("LLM")
    assert "large language model" in output

def test_call_no_result(api_client: MerriamWebsterAPIWrapper) -> None:
    """Test that non-existent words return proper result."""
    output = api_client.run("NO_RESULT_NO_RESULT_NO_RESULT")
    assert "No Merriam-Webster definition was found for query" in output

def test_call_alternatives(api_client: MerriamWebsterAPIWrapper) -> None:
    """
    Test that non-existent queries that are close to an
    existing definition return proper result.
    """
    output = api_client.run("It's raining cats and dogs")
    assert "No Merriam-Webster definition was found for query" in output
    assert "You can try one of the following alternative queries" in output
    assert "raining cats and dogs" in output


# --- libs/community/tests/integration_tests/utilities/test_nasa.py ---

def test_media_search() -> None:
    """Test for NASA Image and Video Library media search"""
    nasa = NasaAPIWrapper()
    query = '{"q": "saturn", + "year_start": "2002", "year_end": "2010", "page": 2}'
    output = nasa.run("search_media", query)
    assert output is not None
    assert "collection" in output

def test_get_media_metadata_manifest() -> None:
    """Test for retrieving media metadata manifest from NASA Image and Video Library"""
    nasa = NasaAPIWrapper()
    output = nasa.run("get_media_metadata_manifest", "2022_0707_Recientemente")
    assert output is not None

def test_get_media_metadata_location() -> None:
    """Test for retrieving media metadata location from NASA Image and Video Library"""
    nasa = NasaAPIWrapper()
    output = nasa.run("get_media_metadata_location", "as11-40-5874")
    assert output is not None

def test_get_video_captions_location() -> None:
    """Test for retrieving video captions location from NASA Image and Video Library"""
    nasa = NasaAPIWrapper()
    output = nasa.run("get_video_captions_location", "172_ISS-Slosh.sr")
    assert output is not None


# --- libs/community/tests/integration_tests/utilities/test_openweathermap.py ---

def test_openweathermap_api_wrapper() -> None:
    """Test that OpenWeatherMapAPIWrapper returns correct data for London, GB."""

    weather = OpenWeatherMapAPIWrapper()
    weather_data = weather.run("London,GB")

    assert weather_data is not None
    assert "London" in weather_data
    assert "GB" in weather_data
    assert "Detailed status:" in weather_data
    assert "Wind speed:" in weather_data
    assert "direction:" in weather_data
    assert "Humidity:" in weather_data
    assert "Temperature:" in weather_data
    assert "Current:" in weather_data
    assert "High:" in weather_data
    assert "Low:" in weather_data
    assert "Feels like:" in weather_data
    assert "Rain:" in weather_data
    assert "Heat index:" in weather_data
    assert "Cloud cover:" in weather_data


# --- libs/community/tests/integration_tests/utilities/test_outline.py ---

def test_run_success(api_client: OutlineAPIWrapper) -> None:
    responses.add(
        responses.POST,
        api_client.outline_instance_url + api_client.outline_search_endpoint,  # type: ignore[operator]
        json=OUTLINE_SUCCESS_RESPONSE,
        status=200,
    )

    docs = api_client.run("Testing")
    assert_docs(docs, all_meta=False)

def test_run_success_all_meta(api_client: OutlineAPIWrapper) -> None:
    api_client.load_all_available_meta = True
    responses.add(
        responses.POST,
        api_client.outline_instance_url + api_client.outline_search_endpoint,  # type: ignore[operator]
        json=OUTLINE_SUCCESS_RESPONSE,
        status=200,
    )

    docs = api_client.run("Testing")
    assert_docs(docs, all_meta=True)

def test_run_no_result(api_client: OutlineAPIWrapper) -> None:
    responses.add(
        responses.POST,
        api_client.outline_instance_url + api_client.outline_search_endpoint,  # type: ignore[operator]
        json=OUTLINE_EMPTY_RESPONSE,
        status=200,
    )

    docs = api_client.run("No Result Test")
    assert not docs

def test_run_error(api_client: OutlineAPIWrapper) -> None:
    responses.add(
        responses.POST,
        api_client.outline_instance_url + api_client.outline_search_endpoint,  # type: ignore[operator]
        json=OUTLINE_ERROR_RESPONSE,
        status=401,
    )
    try:
        api_client.run("Testing")
    except Exception as e:
        assert "Outline API returned an error:" in str(e)


# --- libs/community/tests/integration_tests/utilities/test_passio_nutrition_ai.py ---

def test_call() -> None:
    """Test that call gives the correct answer."""
    api_key = get_from_env("", "NUTRITIONAI_SUBSCRIPTION_KEY")
    search = NutritionAIAPI(
        nutritionai_subscription_key=api_key, auth_=ManagedPassioLifeAuth(api_key)
    )
    output = search.run("Chicken tikka masala")
    assert output is not None
    assert "Chicken tikka masala" == output["results"][0]["displayName"]


# --- libs/community/tests/integration_tests/utilities/test_polygon.py ---

def test_get_last_quote() -> None:
    """Test for getting the last quote of a ticker from the Polygon API."""
    polygon = PolygonAPIWrapper()
    output = polygon.run("get_last_quote", "AAPL")
    assert output is not None


# --- libs/community/tests/integration_tests/utilities/test_powerbi_api.py ---

def test_daxquery() -> None:
    from azure.identity import DefaultAzureCredential

    DATASET_ID = get_from_env("", "POWERBI_DATASET_ID")
    TABLE_NAME = get_from_env("", "POWERBI_TABLE_NAME")
    NUM_ROWS = get_from_env("", "POWERBI_NUMROWS")

    powerbi = PowerBIDataset(
        dataset_id=DATASET_ID,
        table_names=[TABLE_NAME],
        credential=DefaultAzureCredential(),
    )

    output = powerbi.run(f'EVALUATE ROW("RowCount", COUNTROWS({TABLE_NAME}))')
    numrows = str(output["results"][0]["tables"][0]["rows"][0]["[RowCount]"])

    assert NUM_ROWS == numrows


# --- libs/community/tests/integration_tests/utilities/test_pubmed.py ---

def test_run_success(api_client: PubMedAPIWrapper) -> None:
    """Test that returns the correct answer"""

    search_string = (
        "Examining the Validity of ChatGPT in Identifying "
        "Relevant Nephrology Literature"
    )
    output = api_client.run(search_string)
    test_string = (
        "Examining the Validity of ChatGPT in Identifying "
        "Relevant Nephrology Literature: Findings and Implications"
    )
    assert test_string in output
    assert len(output) == api_client.doc_content_chars_max

def test_run_returns_no_result(api_client: PubMedAPIWrapper) -> None:
    """Test that gives no result."""

    output = api_client.run("1605.08386WWW")
    assert "No good PubMed Result was found" == output

def test_load_pupmed_from_universal_entry() -> None:
    pubmed_tool = _load_pubmed_from_universal_entry()
    search_string = (
        "Examining the Validity of ChatGPT in Identifying "
        "Relevant Nephrology Literature"
    )
    output = pubmed_tool.invoke(search_string)
    test_string = (
        "Examining the Validity of ChatGPT in Identifying "
        "Relevant Nephrology Literature: Findings and Implications"
    )
    assert test_string in output

def test_load_pupmed_from_universal_entry_with_params() -> None:
    params = {
        "top_k_results": 1,
    }
    pubmed_tool = _load_pubmed_from_universal_entry(**params)
    assert isinstance(pubmed_tool, PubmedQueryRun)
    wp = pubmed_tool.api_wrapper
    assert wp.top_k_results == 1, "failed to assert top_k_results"


# --- libs/community/tests/integration_tests/utilities/test_reddit_search_api.py ---

def test_run_empty_query(api_client: RedditSearchAPIWrapper) -> None:
    """Test that run gives the correct answer with empty query."""
    search = api_client.run(
        query="", sort="relevance", time_filter="all", subreddit="all", limit=5
    )
    assert search == "Searching r/all did not find any posts:"

def test_run_query(api_client: RedditSearchAPIWrapper) -> None:
    """Test that run gives the correct answer."""
    search = api_client.run(
        query="university",
        sort="relevance",
        time_filter="all",
        subreddit="funny",
        limit=5,
    )
    assert "University" in search


# --- libs/community/tests/integration_tests/utilities/test_searchapi.py ---

def test_call() -> None:
    """Test that call gives correct answer."""
    search = SearchApiAPIWrapper()
    output = search.run("What is the capital of Lithuania?")
    assert "Vilnius" in output

def test_scholar_call() -> None:
    """Test that call gives correct answer for scholar search."""
    search = SearchApiAPIWrapper(engine="google_scholar")
    output = search.run("large language models")
    assert "state of large language models and their applications" in output

def test_jobs_call() -> None:
    """Test that call gives correct answer for jobs search."""
    search = SearchApiAPIWrapper(engine="google_jobs")
    output = search.run("AI")
    assert "years of experience" in output

async def test_async_call() -> None:
    """Test that call gives the correct answer."""
    search = SearchApiAPIWrapper()
    output = await search.arun("What is Obama's full name?")
    assert "Barack Hussein Obama II" in output


# --- libs/community/tests/integration_tests/utilities/test_serpapi.py ---

def test_call() -> None:
    """Test that call gives the correct answer."""
    chain = SerpAPIWrapper()
    output = chain.run("What was Obama's first name?")
    assert output == "Barack Hussein Obama II"


# --- libs/community/tests/integration_tests/utilities/test_stackexchange.py ---

def test_call() -> None:
    """Test that call runs."""
    stackexchange = StackExchangeAPIWrapper()
    output = stackexchange.run("zsh: command not found: python")
    assert output != "hello"

def test_failure() -> None:
    """Test that call that doesn't run."""
    stackexchange = StackExchangeAPIWrapper()
    output = stackexchange.run("sjefbsmnf")
    assert output == "No relevant results found for 'sjefbsmnf' on Stack Overflow"

def test_success() -> None:
    """Test that call that doesn't run."""
    stackexchange = StackExchangeAPIWrapper()
    output = stackexchange.run("zsh: command not found: python")
    assert "zsh: command not found: python" in output


# --- libs/community/tests/integration_tests/utilities/test_steam_api.py ---

def test_get_game_details() -> None:
    """Test for getting game details on Steam"""
    steam = SteamWebAPIWrapper()
    output = steam.run("get_game_details", "Terraria")
    assert "id" in output
    assert "link" in output
    assert "detailed description" in output
    assert "supported languages" in output
    assert "price" in output

def test_get_recommended_games() -> None:
    """Test for getting recommended games on Steam"""
    steam = SteamWebAPIWrapper()
    output = steam.run("get_recommended_games", "76561198362745711")
    output = ast.literal_eval(output)
    assert len(output) == 5


# --- libs/community/tests/integration_tests/utilities/test_twilio.py ---

def test_call() -> None:
    """Test that call runs."""
    twilio = TwilioAPIWrapper()
    output = twilio.run("Message", "+16162904619")
    assert output


# --- libs/community/tests/integration_tests/utilities/test_wikipedia_api.py ---

def test_run_success(api_client: WikipediaAPIWrapper) -> None:
    output = api_client.run("HUNTER X HUNTER")
    assert "Yoshihiro Togashi" in output

def test_run_no_result(api_client: WikipediaAPIWrapper) -> None:
    output = api_client.run(
        "NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL"
    )
    assert "No good Wikipedia Search Result was found" == output


# --- libs/community/tests/integration_tests/utilities/test_wolfram_alpha_api.py ---

def test_call() -> None:
    """Test that call gives the correct answer."""
    search = WolframAlphaAPIWrapper()
    output = search.run("what is 2x+18=x+5?")
    assert "x = -13" in output


# --- libs/community/tests/integration_tests/vectorstores/test_aerospike.py ---

    def test_as_retriever(self, aerospike: Aerospike, admin_client: Any) -> None:
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
        )
        aerospike.add_texts(
            ["foo", "foo", "foo", "foo", "bar"],
            ids=["1", "2", "3", "4", "5"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking

        aerospike._index_name = index_name
        retriever = aerospike.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        results = retriever.invoke("foo")
        assert len(results) == 3
        assert all([d.page_content == "foo" for d in results])

    def test_as_retriever_distance_threshold(
        self, aerospike: Aerospike, admin_client: Any
    ) -> None:
        from aerospike_vector_search import types

        aerospike._distance_strategy = DistanceStrategy.COSINE
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            vector_distance_metric=types.VectorDistanceMetric.COSINE,
        )
        aerospike.add_texts(
            ["foo1", "foo2", "foo3", "bar4", "bar5", "bar6", "bar7", "bar8"],
            ids=["1", "2", "3", "4", "5", "6", "7", "8"],
            index_name=index_name,
            set_name=set_name,
        )  # blocking

        aerospike._index_name = index_name
        retriever = aerospike.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 9, "score_threshold": 0.90},
        )
        results = retriever.invoke("foo1")

        assert all([d.page_content.startswith("foo") for d in results])
        assert len(results) == 3

    def test_as_retriever_add_documents(
        self, aerospike: Aerospike, admin_client: Any
    ) -> None:
        from aerospike_vector_search import types

        aerospike._distance_strategy = DistanceStrategy.COSINE
        index_name = set_name = get_func_name()
        admin_client.index_create(
            namespace=TEST_NAMESPACE,
            sets=set_name,
            name=index_name,
            vector_field=VECTOR_KEY,
            dimensions=10,
            vector_distance_metric=types.VectorDistanceMetric.COSINE,
        )
        retriever = aerospike.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 9, "score_threshold": 0.90},
        )

        documents = [
            Document(
                page_content="foo1",
                metadata={
                    "a": 1,
                },
            ),
            Document(
                page_content="foo2",
                metadata={
                    "a": 2,
                },
            ),
            Document(
                page_content="foo3",
                metadata={
                    "a": 3,
                },
            ),
            Document(
                page_content="bar4",
                metadata={
                    "a": 4,
                },
            ),
            Document(
                page_content="bar5",
                metadata={
                    "a": 5,
                },
            ),
            Document(
                page_content="bar6",
                metadata={
                    "a": 6,
                },
            ),
            Document(
                page_content="bar7",
                metadata={
                    "a": 7,
                },
            ),
        ]
        retriever.add_documents(
            documents,
            ids=["1", "2", "3", "4", "5", "6", "7", "8"],
            index_name=index_name,
            set_name=set_name,
            wait_for_index=True,
        )

        aerospike._index_name = index_name
        results = retriever.invoke("foo1")

        assert all([d.page_content.startswith("foo") for d in results])
        assert len(results) == 3


# --- libs/community/tests/integration_tests/vectorstores/test_azure_cosmos_db.py ---

    def test_invalid_arguments_to_delete(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        with pytest.raises(ValueError) as exception_info:
            self.invoke_delete_with_no_args(azure_openai_embeddings, collection)
        assert str(exception_info.value) == "No document ids provided to delete."

    def test_no_arguments_to_delete_by_id(
        self, azure_openai_embeddings: AzureOpenAIEmbeddings, collection: Any
    ) -> None:
        with pytest.raises(Exception) as exception_info:
            self.invoke_delete_by_id_with_no_args(
                azure_openai_embeddings=azure_openai_embeddings, collection=collection
            )
        assert str(exception_info.value) == "No document id provided to delete."


# --- libs/community/tests/integration_tests/vectorstores/test_documentdb.py ---

    def test_invalid_arguments_to_delete(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        with pytest.raises(ValueError) as exception_info:
            self.invoke_delete_with_no_args(embedding_openai, collection)
        assert str(exception_info.value) == "No document ids provided to delete."

    def test_no_arguments_to_delete_by_id(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        with pytest.raises(Exception) as exception_info:
            self.invoke_delete_by_id_with_no_args(embedding_openai, collection)
        assert str(exception_info.value) == "No document id provided to delete."


# --- libs/community/tests/integration_tests/vectorstores/test_elasticsearch.py ---

    def test_similarity_search_with_approx_infer_instack(
        self, elasticsearch_connection: dict, index_name: str
    ) -> None:
        """test end to end with approx retrieval strategy and inference in-stack"""
        docsearch = ElasticsearchStore(
            index_name=index_name,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                query_model_id="sentence-transformers__all-minilm-l6-v2"
            ),
            query_field="text_field",
            vector_query_field="vector_query_field.predicted_value",
            **elasticsearch_connection,
        )

        # setting up the pipeline for inference
        docsearch.client.ingest.put_pipeline(
            id="test_pipeline",
            processors=[
                {
                    "inference": {
                        "model_id": "sentence-transformers__all-minilm-l6-v2",
                        "field_map": {"query_field": "text_field"},
                        "target_field": "vector_query_field",
                    }
                }
            ],
        )

        # creating a new index with the pipeline,
        # not relying on langchain to create the index
        docsearch.client.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "text_field": {"type": "text"},
                    "vector_query_field": {
                        "properties": {
                            "predicted_value": {
                                "type": "dense_vector",
                                "dims": 384,
                                "index": True,
                                "similarity": "l2_norm",
                            }
                        }
                    },
                }
            },
            settings={"index": {"default_pipeline": "test_pipeline"}},
        )

        # adding documents to the index
        texts = ["foo", "bar", "baz"]

        for i, text in enumerate(texts):
            docsearch.client.create(
                index=index_name,
                id=str(i),
                document={"text_field": text, "metadata": {}},
            )

        docsearch.client.indices.refresh(index=index_name)

        def assert_query(query_body: dict, query: str) -> dict:
            assert query_body == {
                "knn": {
                    "filter": [],
                    "field": "vector_query_field.predicted_value",
                    "k": 1,
                    "num_candidates": 50,
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": "sentence-transformers__all-minilm-l6-v2",
                            "model_text": "foo",
                        }
                    },
                }
            }
            return query_body

        output = docsearch.similarity_search("foo", k=1, custom_query=assert_query)
        assert output == [Document(page_content="foo")]

        output = docsearch.similarity_search("bar", k=1)
        assert output == [Document(page_content="bar")]


# --- libs/community/tests/integration_tests/vectorstores/test_jaguar.py ---

    def test_create(self) -> None:
        """
        Create a vector with vector index 'v' of dimension 10
        and 'v:text' to hold text and metadatas author and category
        """
        metadata_str = "author char(32), category char(16)"
        self.vectorstore.create(metadata_str, 1024)

        podstore = self.pod + "." + self.store
        js = self.vectorstore.run(f"desc {podstore}")
        jd = json.loads(js[0])
        assert podstore in jd["data"]


# --- libs/community/tests/integration_tests/vectorstores/test_vectara.py ---

def test_vectara_rag_with_reranking(vectara2: Vectara) -> None:
    """Test Vectara reranking."""

    query_str = "What is a transformer model?"

    # Note: we don't test rerank_multilingual_v1 as it's for Scale only

    # Test MMR
    summary_config = SummaryConfig(
        is_enabled=True,
        max_results=7,
        response_lang="eng",
        prompt_name=test_prompt_name,
    )
    rerank_config = RerankConfig(reranker="mmr", rerank_k=50, mmr_diversity_bias=0.2)
    config = VectaraQueryConfig(
        k=10,
        lambda_val=0.005,
        rerank_config=rerank_config,
        summary_config=summary_config,
    )

    rag1 = vectara2.as_rag(config)
    response1 = rag1.invoke(query_str)

    assert "transformer model" in response1["answer"].lower()

    # Test No reranking
    summary_config = SummaryConfig(
        is_enabled=True,
        max_results=7,
        response_lang="eng",
        prompt_name=test_prompt_name,
    )
    rerank_config = RerankConfig(reranker="None")
    config = VectaraQueryConfig(
        k=10,
        lambda_val=0.005,
        rerank_config=rerank_config,
        summary_config=summary_config,
    )
    rag2 = vectara2.as_rag(config)
    response2 = rag2.invoke(query_str)

    assert "transformer model" in response2["answer"].lower()

    # assert that the page content is different for the top 5 results
    # in each reranking
    n_results = 10
    response1_content = [x[0].page_content for x in response1["context"][:n_results]]
    response2_content = [x[0].page_content for x in response2["context"][:n_results]]
    assert response1_content != response2_content

def test_vectara_rerankers(vectara3: Vectara) -> None:
    # test Vectara multi-lingual reranker
    summary_config = SummaryConfig(is_enabled=True, max_results=7, response_lang="eng")
    rerank_config = RerankConfig(reranker="rerank_multilingual_v1", rerank_k=50)
    config = VectaraQueryConfig(
        k=10,
        lambda_val=0.005,
        rerank_config=rerank_config,
        summary_config=summary_config,
    )
    rag = vectara3.as_rag(config)
    output1 = rag.invoke("what is generative AI?")["answer"]
    assert len(output1) > 0

    # test Vectara udf reranker
    summary_config = SummaryConfig(is_enabled=True, max_results=7, response_lang="eng")
    rerank_config = RerankConfig(
        reranker="udf", rerank_k=50, user_function="get('$.score')"
    )
    config = VectaraQueryConfig(
        k=10,
        lambda_val=0.005,
        rerank_config=rerank_config,
        summary_config=summary_config,
    )
    rag = vectara3.as_rag(config)
    output1 = rag.invoke("what is generative AI?")["answer"]
    assert len(output1) > 0

    # test Vectara MMR reranker
    summary_config = SummaryConfig(is_enabled=True, max_results=7, response_lang="eng")
    rerank_config = RerankConfig(reranker="mmr", rerank_k=50, mmr_diversity_bias=0.2)
    config = VectaraQueryConfig(
        k=10,
        lambda_val=0.005,
        rerank_config=rerank_config,
        summary_config=summary_config,
    )
    rag = vectara3.as_rag(config)
    output1 = rag.invoke("what is generative AI?")["answer"]
    assert len(output1) > 0

    # test MMR directly with old mmr_config
    summary_config = SummaryConfig(is_enabled=True, max_results=7, response_lang="eng")
    mmr_config = MMRConfig(is_enabled=True, mmr_k=50, diversity_bias=0.2)
    config = VectaraQueryConfig(
        k=10, lambda_val=0.005, mmr_config=mmr_config, summary_config=summary_config
    )
    rag = vectara3.as_rag(config)
    output2 = rag.invoke("what is generative AI?")["answer"]
    assert len(output2) > 0

    # test reranking disabled - RerankConfig
    summary_config = SummaryConfig(is_enabled=True, max_results=7, response_lang="eng")
    rerank_config = RerankConfig(reranker="none")
    config = VectaraQueryConfig(
        k=10,
        lambda_val=0.005,
        rerank_config=rerank_config,
        summary_config=summary_config,
    )
    rag = vectara3.as_rag(config)
    output1 = rag.invoke("what is generative AI?")["answer"]
    assert len(output1) > 0

    # test with reranking disabled - MMRConfig
    summary_config = SummaryConfig(is_enabled=True, max_results=7, response_lang="eng")
    mmr_config = MMRConfig(is_enabled=False, mmr_k=50, diversity_bias=0.2)
    config = VectaraQueryConfig(
        k=10, lambda_val=0.005, mmr_config=mmr_config, summary_config=summary_config
    )
    rag = vectara3.as_rag(config)
    output2 = rag.invoke("what is generative AI?")["answer"]
    assert len(output2) > 0


# --- libs/community/tests/unit_tests/test_sql_database.py ---

def test_sql_database_run_fetch_all(db: SQLDatabase) -> None:
    """Verify running SQL expressions returning results as strings."""

    # Provision.
    stmt = insert(user).values(
        user_id=13, user_name="Harrison", user_bio="That is my Bio " * 24
    )
    db._execute(stmt)

    # Query and verify.
    command = "select user_id, user_name, user_bio from user where user_id = 13"
    partial_output = db.run(command)
    user_bio = "That is my Bio " * 19 + "That is my..."
    expected_partial_output = f"[(13, 'Harrison', '{user_bio}')]"
    assert partial_output == expected_partial_output

    full_output = db.run(command, include_columns=True)
    expected_full_output = (
        "[{'user_id': 13, 'user_name': 'Harrison', 'user_bio': '%s'}]" % user_bio
    )
    assert full_output == expected_full_output

def test_sql_database_run_fetch_result(db: SQLDatabase) -> None:
    """Verify running SQL expressions returning results as SQLAlchemy `Result` instances."""

    # Provision.
    stmt = insert(user).values(user_id=17, user_name="hwchase")
    db._execute(stmt)

    # Query and verify.
    command = "select user_id, user_name, user_bio from user where user_id = 17"
    result = db.run(command, fetch="cursor", include_columns=True)
    expected = [{"user_id": 17, "user_name": "hwchase", "user_bio": None}]
    assert isinstance(result, Result)
    assert result.mappings().fetchall() == expected

def test_sql_database_run_with_parameters(db: SQLDatabase) -> None:
    """Verify running SQL expressions with query parameters."""

    # Provision.
    stmt = insert(user).values(user_id=17, user_name="hwchase")
    db._execute(stmt)

    # Query and verify.
    command = "select user_id, user_name, user_bio from user where user_id = :user_id"
    full_output = db.run(command, parameters={"user_id": 17}, include_columns=True)
    expected_full_output = "[{'user_id': 17, 'user_name': 'hwchase', 'user_bio': None}]"
    assert full_output == expected_full_output

def test_sql_database_run_sqlalchemy_selectable(db: SQLDatabase) -> None:
    """Verify running SQL expressions using SQLAlchemy selectable."""

    # Provision.
    stmt = insert(user).values(user_id=17, user_name="hwchase")
    db._execute(stmt)

    # Query and verify.
    command = select(user).where(user.c.user_id == 17)
    full_output = db.run(command, include_columns=True)
    expected_full_output = "[{'user_id': 17, 'user_name': 'hwchase', 'user_bio': None}]"
    assert full_output == expected_full_output

def test_sql_database_run_update(db: SQLDatabase) -> None:
    """Test commands which return no rows return an empty string."""

    # Provision.
    stmt = insert(user).values(user_id=13, user_name="Harrison")
    db._execute(stmt)

    # Query and verify.
    command = "update user set user_name='Updated' where user_id = 13"
    output = db.run(command)
    expected_output = ""
    assert output == expected_output

def test_sql_database_schema_translate_map() -> None:
    """Verify using statement-specific execution options."""

    engine = sa.create_engine("sqlite:///:memory:")
    db = SQLDatabase(engine)

    # Define query using SQLAlchemy selectable.
    command = select(user).where(user.c.user_id == 17)

    # Define statement-specific execution options.
    execution_options = {"schema_translate_map": {None: "bar"}}

    # Verify the schema translation is applied.
    with pytest.raises(sa.exc.OperationalError) as ex:
        db.run(command, execution_options=execution_options, fetch="cursor")
    assert ex.match("no such table: bar.user")

def test_truncate_word() -> None:
    assert truncate_word("Hello World", length=5) == "He..."
    assert truncate_word("Hello World", length=0) == "Hello World"
    assert truncate_word("Hello World", length=-10) == "Hello World"
    assert truncate_word("Hello World", length=5, suffix="!!!") == "He!!!"
    assert truncate_word("Hello World", length=12, suffix="!!!") == "Hello World"


# --- libs/community/tests/unit_tests/test_sql_database_schema.py ---

def test_sql_database_run() -> None:
    """Test that commands can be run successfully and returned in correct format."""
    engine = create_engine("duckdb:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison")
    with engine.begin() as conn:
        conn.execute(stmt)

    db = SQLDatabase(engine, schema="schema_a")

    command = 'select user_name from "user" where user_id = 13'
    output = db.run(command)
    expected_output = "[('Harrison',)]"
    assert output == expected_output


# --- libs/community/tests/unit_tests/agents/test_tools.py ---

def test_tool_no_args_specified_assumes_str() -> None:
    """Older tools could assume *args and **kwargs were passed in."""

    def ambiguous_function(*args: Any, **kwargs: Any) -> str:
        """An ambiguously defined function."""
        return args[0]

    some_tool = Tool(
        name="chain_run",
        description="Run the chain",
        func=ambiguous_function,
    )
    expected_args = {"tool_input": {"type": "string"}}
    assert some_tool.args == expected_args
    assert some_tool.run("foobar") == "foobar"
    assert some_tool.run({"tool_input": "foobar"}) == "foobar"
    with pytest.raises(ToolException, match="Too many arguments to single-input tool"):
        some_tool.run({"tool_input": "foobar", "other_input": "bar"})


# --- libs/community/tests/unit_tests/callbacks/test_openai_info.py ---

def test_on_llm_end(handler: OpenAICallbackHandler) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
            "model_name": get_fields(BaseOpenAI)["model_name"].default,
        },
    )
    handler.on_llm_end(response)
    assert handler.successful_requests == 1
    assert handler.total_tokens == 3
    assert handler.prompt_tokens == 2
    assert handler.completion_tokens == 1
    assert handler.total_cost > 0

def test_on_llm_end_with_chat_generation(handler: OpenAICallbackHandler) -> None:
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    text="Hello, world!",
                    message=AIMessage(
                        content="Hello, world!",
                        usage_metadata={
                            "input_tokens": 2,
                            "output_tokens": 2,
                            "total_tokens": 4,
                            "input_token_details": {
                                "cache_read": 1,
                            },
                            "output_token_details": {
                                "reasoning": 1,
                            },
                        },
                    ),
                )
            ]
        ],
        llm_output={
            "model_name": get_fields(BaseOpenAI)["model_name"].default,
        },
    )
    handler.on_llm_end(response)
    assert handler.successful_requests == 1
    assert handler.total_tokens == 4
    assert handler.prompt_tokens == 2
    assert handler.prompt_tokens_cached == 1
    assert handler.completion_tokens == 2
    assert handler.reasoning_tokens == 1
    assert handler.total_cost > 0

def test_on_llm_end_custom_model(handler: OpenAICallbackHandler) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
            "model_name": "foo-bar",
        },
    )
    handler.on_llm_end(response)
    assert handler.total_cost == 0

def test_on_llm_end_finetuned_model(
    handler: OpenAICallbackHandler, model_name: str, expected_cost: float
) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "total_tokens": 2000,
            },
            "model_name": model_name,
        },
    )
    handler.on_llm_end(response)
    assert np.isclose(handler.total_cost, expected_cost)

def test_on_llm_end_azure_openai(
    handler: OpenAICallbackHandler, model_name: str, expected_cost: float
) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "total_tokens": 2000,
            },
            "model_name": model_name,
        },
    )
    handler.on_llm_end(response)
    assert math.isclose(handler.total_cost, expected_cost)

def test_on_llm_end_no_cost_invalid_model(
    handler: OpenAICallbackHandler, model_name: str
) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "total_tokens": 2000,
            },
            "model_name": model_name,
        },
    )
    handler.on_llm_end(response)
    assert handler.total_cost == 0


# --- libs/community/tests/unit_tests/callbacks/test_upstash_ratelimit_callback.py ---

def test_on_llm_end_with_token_limit(handler_with_both_limits: Any) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            }
        },
    )
    handler_with_both_limits.on_llm_end(response)
    handler_with_both_limits.token_ratelimit.limit.assert_called_once_with("user123", 2)

def test_on_llm_end_with_token_limit_include_output_tokens(
    token_ratelimit: Any,
) -> None:
    handler = UpstashRatelimitHandler(
        identifier="user123",
        token_ratelimit=token_ratelimit,
        request_ratelimit=None,
        include_output_tokens=True,
    )
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            }
        },
    )
    handler.on_llm_end(response)
    token_ratelimit.limit.assert_called_once_with("user123", 5)

def test_full_chain_with_both_limits(handler_with_both_limits: Any) -> None:
    handler_with_both_limits.on_chain_start(serialized={}, inputs={})
    handler_with_both_limits.on_chain_start(serialized={}, inputs={})

    assert handler_with_both_limits.request_ratelimit.limit.call_count == 1
    assert handler_with_both_limits.token_ratelimit.limit.call_count == 0
    assert handler_with_both_limits.token_ratelimit.get_remaining.call_count == 0

    handler_with_both_limits.on_llm_start(serialized={}, prompts=["test"])

    assert handler_with_both_limits.request_ratelimit.limit.call_count == 1
    assert handler_with_both_limits.token_ratelimit.limit.call_count == 0
    assert handler_with_both_limits.token_ratelimit.get_remaining.call_count == 1

    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            }
        },
    )
    handler_with_both_limits.on_llm_end(response)

    assert handler_with_both_limits.request_ratelimit.limit.call_count == 1
    assert handler_with_both_limits.token_ratelimit.limit.call_count == 1
    assert handler_with_both_limits.token_ratelimit.get_remaining.call_count == 1


# --- libs/community/tests/unit_tests/chains/test_api.py ---

def test_api_question() -> None:
    """Test simple question that needs API access."""
    with pytest.raises(ValueError):
        get_api_chain()
    with pytest.raises(ValueError):
        get_api_chain(limit_to_domains=tuple())

    # All domains allowed (not advised)
    api_chain = get_api_chain(limit_to_domains=None)
    data = get_test_api_data()
    assert api_chain.run(data["question"]) == data["api_summary"]

    # Use a domain that's allowed
    api_chain = get_api_chain(
        limit_to_domains=["https://thisapidoesntexist.com/api/notes?q=langchain"]
    )
    # Attempts to make a request against a domain that's not allowed
    assert api_chain.run(data["question"]) == data["api_summary"]

    # Use domains that are not valid
    api_chain = get_api_chain(limit_to_domains=["h", "*"])
    with pytest.raises(ValueError):
        # Attempts to make a request against a domain that's not allowed
        assert api_chain.run(data["question"]) == data["api_summary"]


# --- libs/community/tests/unit_tests/chat_models/test_llama_edge.py ---

def test_wasm_chat_without_service_url() -> None:
    chat = LlamaEdgeChatService()

    # create message sequence
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]

    with pytest.raises(ValueError) as e:
        chat.invoke(messages)

    assert "Error code: 503" in str(e)
    assert "reason: The IP address or port of the chat service is incorrect." in str(e)


# --- libs/community/tests/unit_tests/chat_models/test_naver.py ---

def test_stream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatClovaX(callbacks=[callback])
    for token in chat.stream("Hello"):
        assert callback.last_token == token.content

async def test_astream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatClovaX(callbacks=[callback])
    async for token in chat.astream("Hello"):
        assert callback.last_token == token.content


# --- libs/community/tests/unit_tests/chat_models/test_oci_data_science.py ---

def test_invoke_vllm(*args: Any) -> None:
    """Tests invoking vLLM endpoint."""
    llm = ChatOCIModelDeploymentVLLM(endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME)
    assert llm._headers().get("route") == CONST_COMPLETION_ROUTE
    output = llm.invoke(CONST_PROMPT)
    assert isinstance(output, AIMessage)
    assert output.content == CONST_COMPLETION

def test_invoke_tgi(*args: Any) -> None:
    """Tests invoking TGI endpoint using OpenAI Spec."""
    llm = ChatOCIModelDeploymentTGI(endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME)
    assert llm._headers().get("route") == CONST_COMPLETION_ROUTE
    output = llm.invoke(CONST_PROMPT)
    assert isinstance(output, AIMessage)
    assert output.content == CONST_COMPLETION

def test_stream_vllm(*args: Any) -> None:
    """Tests streaming with vLLM endpoint using OpenAI spec."""
    llm = ChatOCIModelDeploymentVLLM(
        endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME, streaming=True
    )
    assert llm._headers().get("route") == CONST_COMPLETION_ROUTE
    output = None
    count = 0
    for chunk in llm.stream(CONST_PROMPT):
        assert isinstance(chunk, AIMessageChunk)
        if output is None:
            output = chunk
        else:
            output += chunk
        count += 1
    assert count == 5 + 1  # + 1 additional chunk with chunk_position="last"
    assert output is not None
    if output is not None:
        assert str(output.content).strip() == CONST_COMPLETION


# --- libs/community/tests/unit_tests/chat_models/test_reka.py ---

def test_reka_streaming() -> None:
    llm = ChatReka(streaming=True)
    assert llm.streaming is True


# --- libs/community/tests/unit_tests/document_loaders/test_detect_encoding.py ---

def test_loader_detect_encoding_text() -> None:
    """Test text loader."""
    path = Path(__file__).parent.parent / "examples"
    files = path.glob("**/*.txt")
    loader = DirectoryLoader(str(path), glob="**/*.txt", loader_cls=TextLoader)
    loader_detect_encoding = DirectoryLoader(
        str(path),
        glob="**/*.txt",
        loader_kwargs={"autodetect_encoding": True},
        loader_cls=TextLoader,
    )

    with pytest.raises((UnicodeDecodeError, RuntimeError)):
        loader.load()

    docs = loader_detect_encoding.load()
    assert len(docs) == len(list(files))

def test_loader_detect_encoding_csv() -> None:
    """Test csv loader."""
    path = Path(__file__).parent.parent / "examples"
    files = path.glob("**/*.csv")

    # Count the number of lines.
    row_count = 0
    for file in files:
        encodings = detect_file_encodings(str(file))
        for encoding in encodings:
            try:
                row_count += sum(1 for line in open(file, encoding=encoding.encoding))
                break
            except UnicodeDecodeError:
                continue
        # CSVLoader uses DictReader, and one line per file is a header,
        # so subtract the number of files.
        row_count -= 1

    loader = DirectoryLoader(
        str(path),
        glob="**/*.csv",
        loader_cls=CSVLoader,
    )
    loader_detect_encoding = DirectoryLoader(
        str(path),
        glob="**/*.csv",
        loader_kwargs={"autodetect_encoding": True},
        loader_cls=CSVLoader,
    )

    with pytest.raises((UnicodeDecodeError, RuntimeError)):
        loader.load()

    docs = loader_detect_encoding.load()
    assert len(docs) == row_count


# --- libs/community/tests/unit_tests/document_loaders/parsers/language/test_lua.py ---

    def test_extract_functions_classes(self) -> None:
        segmenter = LuaSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = LuaSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)


# --- libs/community/tests/unit_tests/llms/test_oci_model_deployment_endpoint.py ---

def test_invoke_vllm(*args: Any) -> None:
    """Tests invoking vLLM endpoint."""
    llm = OCIModelDeploymentVLLM(endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME)
    assert llm._headers().get("route") == CONST_COMPLETION_ROUTE
    output = llm.invoke(CONST_PROMPT)
    assert output == CONST_COMPLETION

def test_stream_tgi(*args: Any) -> None:
    """Tests streaming with TGI endpoint using OpenAI spec."""
    llm = OCIModelDeploymentTGI(
        endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME, streaming=True
    )
    assert llm._headers().get("route") == CONST_COMPLETION_ROUTE
    output = ""
    count = 0
    for chunk in llm.stream(CONST_PROMPT):
        output += chunk
        count += 1
    assert count == 4
    assert output.strip() == CONST_COMPLETION

def test_generate_tgi(*args: Any) -> None:
    """Tests invoking TGI endpoint using TGI generate spec."""
    llm = OCIModelDeploymentTGI(
        endpoint=CONST_ENDPOINT, api="/generate", model=CONST_MODEL_NAME
    )
    assert llm._headers().get("route") == CONST_COMPLETION_ROUTE
    output = llm.invoke(CONST_PROMPT)
    assert output == CONST_COMPLETION


# --- libs/community/tests/unit_tests/retrievers/test_ensemble.py ---

def test_ensemble_retriever_get_relevant_docs() -> None:
    doc_list = [
        "I like apples",
        "I like oranges",
        "Apples and oranges are fruits",
    ]

    from langchain_community.retrievers import BM25Retriever

    dummy_retriever = BM25Retriever.from_texts(doc_list)
    dummy_retriever.k = 1

    ensemble_retriever = EnsembleRetriever(  # type: ignore[call-arg]
        retrievers=[dummy_retriever, dummy_retriever]
    )
    docs = ensemble_retriever.invoke("I like apples")
    assert len(docs) == 1


# --- libs/community/tests/unit_tests/tools/test_signatures.py ---

def test_all_subclasses_accept_run_manager(cls: Type[BaseTool]) -> None:
    """Test that tools defined in this repo accept a run manager argument."""
    # This wouldn't be necessary if the BaseTool had a strict API.
    if cls._run is not BaseTool._run:
        run_func = cls._run
        params = inspect.signature(run_func).parameters
        assert "run_manager" in params
        pattern = re.compile(r"(?!Async)CallbackManagerForToolRun")
        assert bool(re.search(pattern, str(params["run_manager"].annotation)))
        assert params["run_manager"].default is None

    if cls._arun is not BaseTool._arun:
        run_func = cls._arun
        params = inspect.signature(run_func).parameters
        assert "run_manager" in params
        assert "AsyncCallbackManagerForToolRun" in str(params["run_manager"].annotation)
        assert params["run_manager"].default is None


# --- libs/community/tests/unit_tests/tools/test_you.py ---

    def test_invoke_news(self) -> None:
        responses.add(
            responses.GET, f"{TEST_ENDPOINT}/news", json=NEWS_RESPONSE_RAW, status=200
        )

        query = "Test news text"
        you_tool = YouSearchTool(
            api_wrapper=YouSearchAPIWrapper(ydc_api_key="test", endpoint_type="news")
        )
        results = you_tool.invoke(query)
        expected_result = NEWS_RESPONSE_PARSED
        assert results == expected_result


# --- libs/community/tests/unit_tests/tools/test_zapier.py ---

def test_default_base_prompt() -> None:
    """Test that the default prompt is being inserted."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(
            zapier_nla_api_key="test", zapier_nla_oauth_access_token=""
        ),
    )

    # Test that the base prompt was successfully assigned to the default prompt
    assert tool.base_prompt == BASE_ZAPIER_TOOL_PROMPT
    assert tool.description == BASE_ZAPIER_TOOL_PROMPT.format(
        zapier_description="test",
        params=str(list({"test": "test"}.keys())),
    )

def test_custom_base_prompt() -> None:
    """Test that a custom prompt is being inserted."""
    base_prompt = "Test. {zapier_description} and {params}."
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        base_prompt=base_prompt,
        api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),  # type: ignore[call-arg]
    )

    # Test that the base prompt was successfully assigned to the default prompt
    assert tool.base_prompt == base_prompt
    assert tool.description == "Test. test and ['test']."

def test_custom_base_prompt_fail() -> None:
    """Test validating an invalid custom prompt."""
    base_prompt = "Test. {zapier_description}."
    with pytest.raises(ValueError):
        ZapierNLARunAction(
            action_id="test",
            zapier_description="test",
            params={"test": "test"},
            base_prompt=base_prompt,
            api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),  # type: ignore[call-arg]
        )

def test_format_headers_api_key() -> None:
    """Test that the action headers is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),  # type: ignore[call-arg]
    )
    headers = tool.api_wrapper._format_headers()
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"
    assert headers["X-API-Key"] == "test"

def test_format_headers_access_token() -> None:
    """Test that the action headers is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(zapier_nla_oauth_access_token="test"),  # type: ignore[call-arg]
    )
    headers = tool.api_wrapper._format_headers()
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"
    assert headers["Authorization"] == "Bearer test"

def test_create_action_payload() -> None:
    """Test that the action payload is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),  # type: ignore[call-arg]
    )

    payload = tool.api_wrapper._create_action_payload("some instructions")
    assert payload["instructions"] == "some instructions"
    assert payload.get("preview_only") is None

def test_create_action_payload_preview() -> None:
    """Test that the action payload with preview is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),  # type: ignore[call-arg]
    )

    payload = tool.api_wrapper._create_action_payload(
        "some instructions",
        preview_only=True,
    )
    assert payload["instructions"] == "some instructions"
    assert payload["preview_only"] is True

def test_create_action_payload_with_params() -> None:
    """Test that the action payload with params is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),  # type: ignore[call-arg]
    )

    payload = tool.api_wrapper._create_action_payload(
        "some instructions",
        {"test": "test"},
        preview_only=True,
    )
    assert payload["instructions"] == "some instructions"
    assert payload["preview_only"] is True
    assert payload["test"] == "test"


# --- libs/community/tests/unit_tests/tools/file_management/test_copy.py ---

def test_copy_file_with_root_dir() -> None:
    """Test the FileCopy tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = CopyFileTool(root_dir=temp_dir)
        source_file = Path(temp_dir) / "source.txt"
        destination_file = Path(temp_dir) / "destination.txt"
        source_file.write_text("Hello, world!")
        tool.run({"source_path": "source.txt", "destination_path": "destination.txt"})
        assert source_file.exists()
        assert destination_file.exists()
        assert source_file.read_text() == "Hello, world!"
        assert destination_file.read_text() == "Hello, world!"

def test_copy_file_errs_outside_root_dir() -> None:
    """Test the FileCopy tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = CopyFileTool(root_dir=temp_dir)
        result = tool.run(
            {
                "source_path": "../source.txt",
                "destination_path": "../destination.txt",
            }
        )
        assert result == INVALID_PATH_TEMPLATE.format(
            arg_name="source_path", value="../source.txt"
        )

def test_copy_file() -> None:
    """Test the FileCopy tool."""
    with TemporaryDirectory() as temp_dir:
        tool = CopyFileTool()
        source_file = Path(temp_dir) / "source.txt"
        destination_file = Path(temp_dir) / "destination.txt"
        source_file.write_text("Hello, world!")
        tool.run(
            {
                "source_path": str(source_file),
                "destination_path": str(destination_file),
            }
        )
        assert source_file.exists()
        assert destination_file.exists()
        assert source_file.read_text() == "Hello, world!"
        assert destination_file.read_text() == "Hello, world!"


# --- libs/community/tests/unit_tests/tools/file_management/test_file_search.py ---

def test_file_search_with_root_dir() -> None:
    """Test the FileSearch tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = FileSearchTool(root_dir=temp_dir)
        file_1 = Path(temp_dir) / "file1.txt"
        file_2 = Path(temp_dir) / "file2.log"
        file_1.write_text("File 1 content")
        file_2.write_text("File 2 content")
        matches = tool.run({"dir_path": ".", "pattern": "*.txt"}).split("\n")
        assert len(matches) == 1
        assert Path(matches[0]).name == "file1.txt"

def test_file_search_errs_outside_root_dir() -> None:
    """Test the FileSearch tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = FileSearchTool(root_dir=temp_dir)
        result = tool.run({"dir_path": "..", "pattern": "*.txt"})
        assert result == INVALID_PATH_TEMPLATE.format(arg_name="dir_path", value="..")

def test_file_search() -> None:
    """Test the FileSearch tool."""
    with TemporaryDirectory() as temp_dir:
        tool = FileSearchTool()
        file_1 = Path(temp_dir) / "file1.txt"
        file_2 = Path(temp_dir) / "file2.log"
        file_1.write_text("File 1 content")
        file_2.write_text("File 2 content")
        matches = tool.run({"dir_path": temp_dir, "pattern": "*.txt"}).split("\n")
        assert len(matches) == 1
        assert Path(matches[0]).name == "file1.txt"


# --- libs/community/tests/unit_tests/tools/file_management/test_list_dir.py ---

def test_list_directory_with_root_dir() -> None:
    """Test the DirectoryListing tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = ListDirectoryTool(root_dir=temp_dir)
        file_1 = Path(temp_dir) / "file1.txt"
        file_2 = Path(temp_dir) / "file2.txt"
        file_1.write_text("File 1 content")
        file_2.write_text("File 2 content")
        entries = tool.run({"dir_path": "."}).split("\n")
        assert set(entries) == {"file1.txt", "file2.txt"}

def test_list_directory_errs_outside_root_dir() -> None:
    """Test the DirectoryListing tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = ListDirectoryTool(root_dir=temp_dir)
        result = tool.run({"dir_path": ".."})
        assert result == INVALID_PATH_TEMPLATE.format(arg_name="dir_path", value="..")

def test_list_directory() -> None:
    """Test the DirectoryListing tool."""
    with TemporaryDirectory() as temp_dir:
        tool = ListDirectoryTool()
        file_1 = Path(temp_dir) / "file1.txt"
        file_2 = Path(temp_dir) / "file2.txt"
        file_1.write_text("File 1 content")
        file_2.write_text("File 2 content")
        entries = tool.run({"dir_path": temp_dir}).split("\n")
        assert set(entries) == {"file1.txt", "file2.txt"}


# --- libs/community/tests/unit_tests/tools/file_management/test_move.py ---

def test_move_file_with_root_dir() -> None:
    """Test the FileMove tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = MoveFileTool(root_dir=temp_dir)
        source_file = Path(temp_dir) / "source.txt"
        destination_file = Path(temp_dir) / "destination.txt"
        source_file.write_text("Hello, world!")
        tool.run({"source_path": "source.txt", "destination_path": "destination.txt"})
        assert not source_file.exists()
        assert destination_file.exists()
        assert destination_file.read_text() == "Hello, world!"

def test_move_file_errs_outside_root_dir() -> None:
    """Test the FileMove tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = MoveFileTool(root_dir=temp_dir)
        result = tool.run(
            {
                "source_path": "../source.txt",
                "destination_path": "../destination.txt",
            }
        )
        assert result == INVALID_PATH_TEMPLATE.format(
            arg_name="source_path", value="../source.txt"
        )

def test_move_file() -> None:
    """Test the FileMove tool."""
    with TemporaryDirectory() as temp_dir:
        tool = MoveFileTool()
        source_file = Path(temp_dir) / "source.txt"
        destination_file = Path(temp_dir) / "destination.txt"
        source_file.write_text("Hello, world!")
        tool.run(
            {
                "source_path": str(source_file),
                "destination_path": str(destination_file),
            }
        )
        assert not source_file.exists()
        assert destination_file.exists()
        assert destination_file.read_text() == "Hello, world!"


# --- libs/community/tests/unit_tests/tools/file_management/test_read.py ---

def test_read_file_with_root_dir() -> None:
    """Test the ReadFile tool."""
    with TemporaryDirectory() as temp_dir:
        with (Path(temp_dir) / "file.txt").open("w") as f:
            f.write("Hello, world!")
        tool = ReadFileTool(root_dir=temp_dir)
        result = tool.run("file.txt")
        assert result == "Hello, world!"
        # Check absolute files can still be passed if they lie within the root dir.
        result = tool.run(str(Path(temp_dir) / "file.txt"))
        assert result == "Hello, world!"

def test_read_file() -> None:
    """Test the ReadFile tool."""
    with TemporaryDirectory() as temp_dir:
        with (Path(temp_dir) / "file.txt").open("w") as f:
            f.write("Hello, world!")
        tool = ReadFileTool()
        result = tool.run(str(Path(temp_dir) / "file.txt"))
        assert result == "Hello, world!"


# --- libs/community/tests/unit_tests/tools/file_management/test_write.py ---

def test_write_file_with_root_dir() -> None:
    """Test the WriteFile tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = WriteFileTool(root_dir=temp_dir)
        tool.run({"file_path": "file.txt", "text": "Hello, world!"})
        assert (Path(temp_dir) / "file.txt").exists()
        assert (Path(temp_dir) / "file.txt").read_text() == "Hello, world!"

def test_write_file_errs_outside_root_dir() -> None:
    """Test the WriteFile tool when a root dir is specified."""
    with TemporaryDirectory() as temp_dir:
        tool = WriteFileTool(root_dir=temp_dir)
        result = tool.run({"file_path": "../file.txt", "text": "Hello, world!"})
        assert result == INVALID_PATH_TEMPLATE.format(
            arg_name="file_path", value="../file.txt"
        )

def test_write_file() -> None:
    """Test the WriteFile tool."""
    with TemporaryDirectory() as temp_dir:
        file_path = str(Path(temp_dir) / "file.txt")
        tool = WriteFileTool()
        tool.run({"file_path": file_path, "text": "Hello, world!"})
        assert (Path(temp_dir) / "file.txt").exists()
        assert (Path(temp_dir) / "file.txt").read_text() == "Hello, world!"

def test_write_file_in_subdir_of_root_dir() -> None:
    """Test the WriteFile tool when the path is a subdirectory of the root dir."""
    with TemporaryDirectory() as temp_dir:
        tool = WriteFileTool(root_dir=temp_dir)
        tool.run({"file_path": "a/b/file.txt", "text": "Hello, world!"})
        assert (Path(temp_dir) / "a/b/file.txt").exists()


# --- libs/community/tests/unit_tests/tools/shell/test_shell.py ---

def test_shell_tool_run() -> None:
    placeholder = PlaceholderProcess(output="hello")
    shell_tool = ShellTool(process=placeholder)
    result = shell_tool._run(commands=test_commands)
    assert result.strip() == "hello"

async def test_shell_tool_arun() -> None:
    placeholder = PlaceholderProcess(output="hello")
    shell_tool = ShellTool(process=placeholder)
    result = await shell_tool._arun(commands=test_commands)
    assert result.strip() == "hello"

def test_shell_tool_run_str() -> None:
    placeholder = PlaceholderProcess(output="hello")
    shell_tool = ShellTool(process=placeholder)
    result = shell_tool._run(commands="echo 'Hello, World!'")
    assert result.strip() == "hello"


# --- libs/community/tests/unit_tests/utilities/test_cassandra_database.py ---

    def test_run_query_invalid_fetch(self) -> None:
        with pytest.raises(ValueError):
            self.cassandra_db.run("SELECT * FROM table;", fetch="invalid")


# --- libs/community/tests/unit_tests/utilities/test_nvidia_riva_asr.py ---

def test_config(asr: RivaASR) -> None:
    """Verify the Riva config is properly assembled."""
    # pylint: disable-next=import-outside-toplevel
    import riva.client.proto.riva_asr_pb2 as rasr

    expected = rasr.StreamingRecognitionConfig(
        interim_results=True,
        config=rasr.RecognitionConfig(
            encoding=CONFIG["encoding"],
            sample_rate_hertz=CONFIG["sample_rate_hertz"],
            audio_channel_count=CONFIG["audio_channel_count"],
            max_alternatives=1,
            profanity_filter=CONFIG["profanity_filter"],
            enable_automatic_punctuation=CONFIG["enable_automatic_punctuation"],
            language_code=CONFIG["language_code"],
        ),
    )
    assert asr.config == expected


# --- libs/community/tests/unit_tests/utils/test_math.py ---

def test_cosine_similarity_top_k_and_score_threshold(
    X: List[List[float]], Y: List[List[float]]
) -> None:
    if importlib.util.find_spec("simsimd"):
        raise ValueError("test should be run without simsimd installed.")
    invoke_cosine_similarity_top_k_score_threshold(X, Y)

def test_cosine_similarity_top_k_and_score_threshold_with_simsimd(
    X: List[List[float]], Y: List[List[float]]
) -> None:
    # Same test, but ensuring simsimd is available in the project through the import.
    invoke_cosine_similarity_top_k_score_threshold(X, Y)


# --- libs/community/tests/unit_tests/vectorstores/redis/test_redis_schema.py ---

def test_hnsw_vector_field_defaults() -> None:
    """Test defaults for HNSWVectorField."""
    hnsw_vector_field_data = {
        "name": "example",
        "dims": 100,
        "algorithm": "HNSW",
    }

    hnsw_vector = HNSWVectorField(**hnsw_vector_field_data)  # type: ignore[arg-type]
    assert hnsw_vector.datatype == "FLOAT32"
    assert hnsw_vector.distance_metric == "COSINE"
    assert hnsw_vector.initial_cap is None
    assert hnsw_vector.m == 16
    assert hnsw_vector.ef_construction == 200
    assert hnsw_vector.ef_runtime == 10
    assert hnsw_vector.epsilon == 0.01

def test_hnsw_vector_field_optional_values() -> None:
    """Test optional values for HNSWVectorField."""
    hnsw_vector_field_data = {
        "name": "example",
        "dims": 100,
        "algorithm": "HNSW",
        "initial_cap": 2000,
        "m": 10,
        "ef_construction": 250,
        "ef_runtime": 15,
        "epsilon": 0.05,
    }
    hnsw_vector = HNSWVectorField(**hnsw_vector_field_data)  # type: ignore[arg-type]
    assert hnsw_vector.initial_cap == 2000
    assert hnsw_vector.m == 10
    assert hnsw_vector.ef_construction == 250
    assert hnsw_vector.ef_runtime == 15
    assert hnsw_vector.epsilon == 0.05

