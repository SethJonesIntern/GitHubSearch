# ju-bezdek/langchain-decorators
# 10 LLM-backed test functions across 10 test files
# Source: https://github.com/ju-bezdek/langchain-decorators

# --- tests/test_chains.py ---

    def test_multi_step_workflow_sync(self, setup_real_llm):
        """Test multi-step workflow with function chaining - sync"""

        @llm_function
        def analyze_text(text: str) -> str:
            """Analyze text for sentiment and topics

            Args:
                text (str): Text to analyze
            """
            # Simple sentiment analysis simulation
            positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
            negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]

            text_lower = text.lower()
            sentiment = "neutral"

            if any(word in text_lower for word in positive_words):
                sentiment = "positive"
            elif any(word in text_lower for word in negative_words):
                sentiment = "negative"

            return f"Sentiment: {sentiment}, Text length: {len(text)} characters"

        @llm_function
        def generate_summary(analysis: str, original_text: str) -> str:
            """Generate a summary based on analysis

            Args:
                analysis (str): Text analysis results
                original_text (str): Original text
            """
            return f"Summary: Text analysis shows {analysis}. Original text: '{original_text[:50]}...'"

        @llm_prompt
        def text_workflow_manager(
            text_input: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """Manage text processing workflow for: {text_input}

            First analyze the text, then create a summary.
            """
            pass

        # Start workflow
        result = text_workflow_manager(
            text_input="This is an amazing product that exceeded my expectations!",
            functions=[analyze_text, generate_summary],
        )

        if result.is_function_call:
            analysis_result = result.execute()
            assert "positive" in analysis_result.lower()

    async def test_multi_step_workflow_async(self, setup_real_llm):
        """Test multi-step workflow with function chaining - async"""

        @llm_function
        def analyze_text(text: str) -> str:
            """Analyze text for sentiment and topics

            Args:
                text (str): Text to analyze
            """
            # Simple sentiment analysis simulation
            positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
            negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]

            text_lower = text.lower()
            sentiment = "neutral"

            if any(word in text_lower for word in positive_words):
                sentiment = "positive"
            elif any(word in text_lower for word in negative_words):
                sentiment = "negative"

            return f"Sentiment: {sentiment}, Text length: {len(text)} characters"

        @llm_function
        def generate_summary(analysis: str, original_text: str) -> str:
            """Generate a summary based on analysis

            Args:
                analysis (str): Text analysis results
                original_text (str): Original text
            """
            return f"Summary: Text analysis shows {analysis}. Original text: '{original_text[:50]}...'"

        @llm_prompt
        async def text_workflow_manager(
            text_input: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """Manage text processing workflow for: {text_input}

            First analyze the text, then create a summary.
            """
            pass

        # Start workflow
        result = await text_workflow_manager(
            text_input="This is an amazing product that exceeded my expectations!",
            functions=[analyze_text, generate_summary],
        )

        if result.is_function_call:
            analysis_result = result.execute()
            assert "positive" in analysis_result.lower()


# --- tests/test_common.py ---

    def test_global_settings_define_settings(self):
        """Test GlobalSettings.define_settings"""
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI()

        GlobalSettings.define_settings(
            default_llm=llm, verbose=True, default_streaming=True
        )

        settings = GlobalSettings.get_current_settings()
        assert settings.default_llm is llm
        assert settings.verbose == True
        assert settings.default_streaming == True

    def test_custom_prompt_types_inheritance(self):
        """Test creating custom prompt types through inheritance"""

        class CustomPromptTypes(PromptTypes):
            GPT4 = PromptTypeSettings(
                color=(
                    LogColors.PURPLE if hasattr(LogColors, "PURPLE") else LogColors.BLUE
                ),
                log_level="debug",
            )
            CREATIVE = PromptTypeSettings(
                color=LogColors.YELLOW, log_level="info", capture_stream=True
            )

        # Test that custom types are accessible
        assert hasattr(CustomPromptTypes, "GPT4")
        assert hasattr(CustomPromptTypes, "CREATIVE")
        assert hasattr(CustomPromptTypes, "UNDEFINED")  # Should inherit from parent

        # Test that custom types have correct properties
        assert CustomPromptTypes.GPT4.log_level == logging.DEBUG
        assert CustomPromptTypes.CREATIVE.capture_stream == True


# --- tests/test_integration.py ---

    async def test_async_streaming_chat_agent(self):
        """Test async streaming chat agent with real-time responses"""

        chat_history = []

        @llm_function
        async def remember_user_info(name: str, detail: str) -> str:
            """Remember information about the user

            Args:
                name (str): User's name
                detail (str): Information to remember
            """
            await asyncio.sleep(0.1)  # Simulate async operation
            timestamp = datetime.now().strftime("%H:%M")
            return f"Remembered about {name}: {detail} (saved at {timestamp})"

        @llm_function
        async def get_time() -> str:
            """Get current time"""
            await asyncio.sleep(0.05)
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        @llm_prompt(capture_stream=True)
        async def streaming_chat_agent(
            user_message: str, user_name: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """
            ```<prompt:system>
            You are a friendly chat assistant. Be conversational and helpful.
            Remember information about users when they share it.
            Keep responses concise and engaging.
            ```

            ```<prompt:user>
            {user_name}: {user_message}
            ```

            Available functions: {functions}
            """
            pass

        collected_tokens = []

        def token_collector(token: str):
            collected_tokens.append(token)

        # Test streaming response
        with StreamingContext(stream_to_stdout=False, callback=token_collector):
            result = await streaming_chat_agent(
                user_message="Hi, I'm Sarah and I love hiking in the mountains",
                user_name="Sarah",
                functions=[remember_user_info, get_time],
                function_call="none",
            )

        # Verify streaming worked
        assert len(collected_tokens) > 0

        if result.is_function_call:
            function_result = await result.execute_async()
            assert "Sarah" in function_result
            assert "hiking" in function_result.lower()

        # Test time function
        result2 = await streaming_chat_agent(
            user_message="What time is it?",
            user_name="Sarah",
            functions=[remember_user_info, get_time],
        )

        if result2.is_function_call and result2.function_name == "get_time":
            time_result = await result2.execute_async()
            assert "2024" in time_result or "2025" in time_result  # Current year

    def test_educational_tutor_system(self):
        """Test educational tutor with progress tracking"""

        student_progress = {
            "topics_covered": [],
            "quiz_scores": {},
            "current_level": "beginner",
        }

        quiz_questions = {
            "python_basics": [
                {
                    "question": "What keyword is used to define a function in Python?",
                    "answer": "def",
                },
                {"question": "How do you create a list in Python?", "answer": "[]"},
            ],
            "data_types": [
                {
                    "question": "What data type represents whole numbers?",
                    "answer": "int",
                },
                {"question": "What data type represents text?", "answer": "str"},
            ],
        }

        @llm_function
        def create_quiz(topic: str, difficulty: str = "beginner") -> str:
            """Create a quiz for a specific topic

            Args:
                topic (str): Topic for the quiz
                difficulty (str): Difficulty level (beginner, intermediate, advanced)
            """
            if topic.lower().replace(" ", "_") in quiz_questions:
                questions = quiz_questions[topic.lower().replace(" ", "_")]
                quiz_text = f"Quiz: {topic.title()} ({difficulty})\n"
                for i, q in enumerate(questions, 1):
                    quiz_text += f"{i}. {q['question']}\n"
                return quiz_text
            else:
                return f"No quiz available for topic '{topic}'"

        @llm_function
        def record_progress(topic: str, score: float) -> str:
            """Record student progress

            Args:
                topic (str): Topic studied
                score (float): Quiz score (0-100)
            """
            student_progress["topics_covered"].append(topic)
            student_progress["quiz_scores"][topic] = score

            # Update level based on average score
            avg_score = sum(student_progress["quiz_scores"].values()) / len(
                student_progress["quiz_scores"]
            )
            if avg_score >= 80:
                student_progress["current_level"] = "advanced"
            elif avg_score >= 60:
                student_progress["current_level"] = "intermediate"

            return f"Progress recorded for {topic}: {score}%. Current level: {student_progress['current_level']}"

        @llm_function
        def get_learning_path(current_topic: str) -> str:
            """Get suggested learning path

            Args:
                current_topic (str): Current topic being studied
            """
            learning_paths = {
                "python_basics": ["data_types", "control_structures", "functions"],
                "data_types": ["variables", "strings", "lists"],
                "control_structures": ["loops", "conditionals", "error_handling"],
            }

            next_topics = learning_paths.get(
                current_topic.lower().replace(" ", "_"), ["advanced_topics"]
            )
            return f"After {current_topic}, consider studying: {', '.join(next_topics)}"

        @llm_prompt
        def educational_tutor(
            student_request: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """
            ```<prompt:system>
            You are an educational tutor specializing in programming.
            Help students learn by providing quizzes, tracking progress, and suggesting learning paths.

            Be encouraging and adapt your teaching style to the student's level.
            Use available functions to create interactive learning experiences.
            ```

            ```<prompt:user>
            Student request: {student_request}
            ```

            Available functions: {functions}
            """
            pass

        # Test quiz creation
        result1 = educational_tutor(
            student_request="I want to practice Python basics with a quiz",
            functions=[create_quiz, record_progress, get_learning_path],
        )

        if result1.is_function_call and result1.function_name == "create_quiz":
            quiz_result = result1.execute()
            assert "Quiz:" in quiz_result
            assert "Python" in quiz_result or "python" in quiz_result

        # Test progress recording
        result2 = educational_tutor(
            student_request="I completed the Python basics quiz and got 85%",
            functions=[create_quiz, record_progress, get_learning_path],
        )

        if result2.is_function_call and result2.function_name == "record_progress":
            progress_result = result2.execute()
            assert "85" in progress_result
            assert "Progress recorded" in progress_result

        # Test learning path suggestion
        result3 = educational_tutor(
            student_request="What should I study next after Python basics?",
            functions=[create_quiz, record_progress, get_learning_path],
        )

        if result3.is_function_call and result3.function_name == "get_learning_path":
            path_result = result3.execute()
            assert "After" in path_result
            assert "consider studying" in path_result


# --- tests/test_prompt_decorator.py ---

    async def test_streaming_integration(self, setup_real_llm):
        """Test streaming integration with async prompt"""
        collected_tokens = []

        def token_collector(token: str):
            collected_tokens.append(token)

        @llm_prompt(capture_stream=True)
        async def streaming_writer(topic: str) -> str:
            """Write a short paragraph about {topic}."""
            pass

        # Test streaming response
        with StreamingContext(stream_to_stdout=False, callback=token_collector):
            result = await streaming_writer(topic="artificial intelligence")

        # Verify streaming worked
        assert len(collected_tokens) > 0
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_custom_prompt_types_sync(self, setup_real_llm):
        """Test custom prompt types and settings - sync"""
        from langchain_openai import ChatOpenAI

        custom_llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")

        custom_prompt_type = PromptTypeSettings(llm=custom_llm, capture_stream=False)

        @llm_prompt(prompt_type=custom_prompt_type)
        def creative_writing(prompt: str) -> str:
            """Write something creative about {prompt}. Be imaginative and artistic."""
            pass

        result = creative_writing(prompt="a magical forest")

        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should contain forest-related content
        assert any(
            word in result.lower() for word in ["forest", "tree", "magic", "wood"]
        )

    async def test_custom_prompt_types_async(self, setup_real_llm):
        """Test custom prompt types and settings - async"""
        from langchain_openai import ChatOpenAI

        custom_llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")

        custom_prompt_type = PromptTypeSettings(llm=custom_llm, capture_stream=False)

        @llm_prompt(prompt_type=custom_prompt_type)
        async def creative_writing(prompt: str) -> str:
            """Write something creative about {prompt}. Be imaginative and artistic."""
            pass

        result = await creative_writing(prompt="a magical forest")

        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should contain forest-related content
        assert any(
            word in result.lower() for word in ["forest", "tree", "magic", "wood"]
        )


# --- tests/test_tool_calling_example.py ---

def test_example():
    from code_examples.tool_calling import Agent

    input_output = [
        ("hi, whats my name", lambda res, messages: "John" in res),
        (
            "my cat died...how do you feel?",
            lambda res, messages: messages[-1].type == "tool",
        ),
        (
            "now use langchain to find me new one",
            lambda res, messages: messages[-1].type == "tool",
        ),
    ]
    agent = Agent(customer_name="John")
    for user_input, assert_func in input_output:
        res = agent.invoke(user_input=user_input)
        assert assert_func(res, agent.memory)

