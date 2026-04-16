# ag2ai/fastagency
# 2 LLM-backed test functions across 56 test files
# Source: https://github.com/ag2ai/fastagency

# --- tests/runtime/ag2/test_autogen.py ---

def test_simple(
    openai_gpt4o_mini_llm_config: LLMConfig, agent_class: type[ConversableAgent]
) -> None:
    wf = Workflow()

    @wf.register(
        name="simple_learning", description="Student and teacher learning chat"
    )
    def simple_workflow(ui: UI, params: dict[str, Any]) -> str:
        initial_message = "What is triangle inequality?"

        student_agent = ConversableAgent(
            name="Student_Agent",
            system_message="You are a student willing to learn.",
            llm_config=openai_gpt4o_mini_llm_config,
        )
        teacher_agent = agent_class(
            name="Teacher_Agent",
            system_message="You are a math teacher.",
            llm_config=openai_gpt4o_mini_llm_config,
        )

        response = student_agent.run(
            teacher_agent,
            message=initial_message,
            summary_method="reflection_with_llm",
            max_turns=3,
        )

        return ui.process(response)  # type: ignore[no-any-return]

    name = "simple_learning"

    ui = ConsoleUI().create_workflow_ui(workflow_uuid=uuid4().hex)

    ui.workflow_started(
        sender="workflow",
        recipient="User",
        name=name,
    )

    result = wf.run(
        name=name,
        ui=ui,
    )

    ui.workflow_completed(
        sender="workflow",
        recipient="User",
        result=result,
    )

    assert "triangle inequality" in result.lower()

def test_simple_async(
    openai_gpt4o_mini_llm_config: LLMConfig, agent_class: type[ConversableAgent]
) -> None:
    async def async_function() -> str:
        # Simulate some asynchronous work
        await asyncio.sleep(1)
        return "Async function completed!"

    wf = Workflow()

    @wf.register(  # type: ignore[type-var]
        name="simple_learning", description="Student and teacher learning chat"
    )
    async def simple_workflow(ui: UI, params: dict[str, Any]) -> str:
        initial_message = "What is triangle inequality?"

        student_agent = ConversableAgent(
            name="Student_Agent",
            system_message="You are a student willing to learn.",
            llm_config=openai_gpt4o_mini_llm_config,
        )
        teacher_agent = agent_class(
            name="Teacher_Agent",
            system_message="You are a math teacher.",
            llm_config=openai_gpt4o_mini_llm_config,
        )

        async_func_response = await async_function()
        assert async_func_response == "Async function completed!"

        response = await student_agent.a_run(
            teacher_agent,
            message=initial_message,
            summary_method="reflection_with_llm",
            max_turns=3,
        )

        return await ui.async_process(response)  # type: ignore[no-any-return]

    name = "simple_learning"

    ui = ConsoleUI().create_workflow_ui(workflow_uuid=uuid4().hex)

    ui.workflow_started(
        sender="workflow",
        recipient="User",
        name=name,
    )

    result = wf.run(
        name=name,
        ui=ui,
    )

    ui.workflow_completed(
        sender="workflow",
        recipient="User",
        result=result,
    )

    assert "triangle inequality" in result.lower()

