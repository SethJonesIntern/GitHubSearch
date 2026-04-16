# langchain-ai/langchain-cohere
# 3 LLM-backed test functions across 28 test files
# Source: https://github.com/langchain-ai/langchain-cohere

# --- libs/cohere/tests/integration_tests/test_langgraph_agents.py ---

def test_langgraph_react_agent() -> None:
    from langgraph.prebuilt import create_react_agent  # type: ignore

    @tool
    def web_search(query: str) -> Union[int, str]:
        """Search the web to the answer to the question with a query search string.

        Args:
            query: The search query to surf the web with
        """
        if "obama" and "age" in query.lower():
            return 60
        if "president" in query:
            return "Barack Obama is the president of the USA"
        if "premier" in query:
            return "Chelsea won the premier league"
        return "The team called Fighter's Foxes won the champions league"

    @tool("python_interpeter_temp")
    def python_tool(code: str) -> str:
        """Executes python code and returns the result.
        The code runs in a static sandbox without interactive mode,
        so print output or save output to a file.

        Args:
            code: Python code to execute.
        """
        if "math.sqrt" in code:
            return "7.75"
        return "The code ran successfully"

    system_message = "You are a helpful assistant. Respond only in English."

    tools = [web_search, python_tool]
    model = ChatCohere(model=DEFAULT_MODEL)

    app = create_react_agent(model, tools, prompt=system_message)

    query = (
        "Find Barack Obama's age and use python tool to find the square root of his age"
    )

    messages = app.invoke({"messages": [("human", query)]})

    model_output = {
        "input": query,
        "output": messages["messages"][-1].content,
    }
    assert "7.7" in model_output.get("output", "").lower()

    message_history = messages["messages"]

    new_query = "who won the premier league"

    messages = app.invoke({"messages": message_history + [("human", new_query)]})
    final_answer = {
        "input": new_query,
        "output": messages["messages"][-1].content,
    }
    assert "chelsea" in final_answer.get("output", "").lower()

def test_langchain_tool_calling_agent() -> None:
    from langgraph.prebuilt import create_react_agent  # type: ignore

    @tool
    def magic_function(input: int) -> int:
        """Applies a magic function to an input.

        Args:
            input: Number to apply the magic function to.
        """
        return input + 2

    model = ChatCohere(model=DEFAULT_MODEL)
    app = create_react_agent(
        model, [magic_function], prompt="You are a helpful assistant"
    )

    query = "what is the value of magic_function(3)?"
    messages = app.invoke({"messages": [("human", query)]})
    assert "5" in messages["messages"][-1].content.lower()


# --- libs/cohere/tests/integration_tests/sql_agent/test_sql_agent.py ---

def test_sql_agent() -> None:
    db = SQLDatabase.from_uri(
        "sqlite:///tests/integration_tests/sql_agent/db/employees.db"
    )
    llm = ChatCohere(model="command-a-03-2025", temperature=0)
    agent_executor = create_sql_agent(
        llm, db=db, agent_type="tool-calling", verbose=True
    )
    resp = agent_executor.invoke({"input": "which employee has the highest salary?"})
    assert "output" in resp.keys()
    assert "jane doe" in resp.get("output", "").lower()

