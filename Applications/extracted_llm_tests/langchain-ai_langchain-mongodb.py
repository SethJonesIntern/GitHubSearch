# langchain-ai/langchain-mongodb
# 3 LLM-backed test functions across 45 test files
# Source: https://github.com/langchain-ai/langchain-mongodb

# --- libs/langchain-mongodb/tests/integration_tests/test_agent_toolkit.py ---

def test_toolkit_response(db):
    db_wrapper = MongoDBDatabase.from_connection_string(
        CONNECTION_STRING, database=DB_NAME
    )
    if "AZURE_OPENAI_ENDPOINT" in os.environ:
        llm = AzureChatOpenAI(model="gpt-5-mini", timeout=60, seed=12345)
    else:
        llm = ChatOpenAI(model="gpt-5-mini", timeout=60, seed=12345)

    toolkit = MongoDBDatabaseToolkit(db=db_wrapper, llm=llm)

    prompt = MONGODB_AGENT_SYSTEM_PROMPT.format(top_k=5)

    test_query = "Which country's customers spent the most?"
    agent = create_agent(llm, toolkit.get_tools(), system_prompt=prompt)
    agent.step_timeout = 60
    events = agent.stream(
        {"messages": [("user", test_query)]},
        stream_mode="values",
    )
    messages = []
    for event in events:
        messages.extend(event["messages"])
    assert "USA" in messages[-1].content, messages[-1].content
    db_wrapper.close()


# --- libs/langchain-mongodb/tests/integration_tests/test_retriever_selfquerying.py ---

def test_selfquerying(retriever, fictitious_movies):
    """Confirm that the retriever was initialized."""
    assert isinstance(retriever, SelfQueryRetriever)

    """This example specifies a single filter."""
    res_filter = retriever.invoke("I want to watch a movie rated higher than 8.5")
    assert isinstance(res_filter, list)
    assert isinstance(res_filter[0], Document)
    assert len(res_filter) == 1
    assert res_filter[0].metadata["title"] == "The Coda Paradox"

    """This example specifies a composite AND filter."""
    res_and = retriever.invoke(
        "Provide movies made after 2030 that are rated lower than 8"
    )
    assert isinstance(res_and, list)
    assert len(res_and) == 2
    assert set(film.metadata["title"] for film in res_and) == {
        "Manifesto Midnight",
        "Neon Tide",
    }

    """This example specifies a composite OR filter."""
    res_or = retriever.invoke("Provide movies made after 2030 or rated higher than 8.4")
    assert isinstance(res_or, list)
    assert len(res_or) == 4
    assert set(film.metadata["title"] for film in res_or) == {
        "Manifesto Midnight",
        "Neon Tide",
        "The Abyssal Crown",
        "The Coda Paradox",
    }

    """This one does not have a filter."""
    res_nofilter = retriever.invoke("Provide movies that take place underwater")
    assert len(res_nofilter) == len(fictitious_movies)

    """This example gives a limit."""
    res_limit = retriever.invoke("Provide 3 movies")
    assert len(res_limit) == 3

def test_selfquerying_autoembed(retriever, fictitious_movies):
    """Confirm that the retriever was initialized."""
    assert isinstance(retriever, SelfQueryRetriever)

    """This example specifies a single filter."""
    res_filter = retriever.invoke("I want to watch a movie rated higher than 8.5")
    assert isinstance(res_filter, list)
    assert isinstance(res_filter[0], Document)
    assert len(res_filter) == 1
    assert res_filter[0].metadata["title"] == "The Coda Paradox"

    """This example specifies a composite AND filter."""
    res_and = retriever.invoke(
        "Provide movies made after 2030 that are rated lower than 8"
    )
    assert isinstance(res_and, list)
    assert len(res_and) == 2
    assert set(film.metadata["title"] for film in res_and) == {
        "Manifesto Midnight",
        "Neon Tide",
    }

    """This example specifies a composite OR filter."""
    res_or = retriever.invoke("Provide movies made after 2030 or rated higher than 8.4")
    assert isinstance(res_or, list)
    assert len(res_or) == 4
    assert set(film.metadata["title"] for film in res_or) == {
        "Manifesto Midnight",
        "Neon Tide",
        "The Abyssal Crown",
        "The Coda Paradox",
    }

    """This one does not have a filter."""
    res_nofilter = retriever.invoke("Provide movies that take place underwater")
    assert len(res_nofilter) == len(fictitious_movies)

    """This example gives a limit."""
    res_limit = retriever.invoke("Provide 3 movies")
    assert len(res_limit) == 3

