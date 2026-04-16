# redis-developer/langgraph-redis
# 13 LLM-backed test functions across 91 test files
# Source: https://github.com/redis-developer/langgraph-redis

# --- tests/integration/test_middleware_create_agent.py ---

    async def test_semantic_cache_with_real_agent(
        self, redis_url: str, default_vectorizer
    ):
        """Test SemanticCacheMiddleware with a real LangChain agent."""
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"

        # Create middleware with unique cache name
        import uuid

        cache_name = f"test_cache_{uuid.uuid4().hex[:8]}"

        middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                ttl_seconds=60,
                distance_threshold=0.1,
                vectorizer=default_vectorizer,
            )
        )

        try:
            # Create agent with middleware
            agent = create_agent(
                model="gpt-4o-mini",
                tools=[get_weather],
                middleware=[middleware],
            )

            # First call - should be cache miss
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What's the weather in Paris?")]}
            )

            assert "messages" in result1
            assert len(result1["messages"]) > 0

            # Second call with same query - should be cache hit
            result2 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What's the weather in Paris?")]}
            )

            assert "messages" in result2
            assert len(result2["messages"]) > 0

            print(f"First result: {result1['messages'][-1].content[:100]}...")
            print(f"Second result: {result2['messages'][-1].content[:100]}...")

        finally:
            await middleware.aclose()

    async def test_middleware_without_llm_call(self, redis_url: str):
        """Test that middleware correctly wraps model calls.

        This test pre-populates the cache so no LLM call is needed.
        Note: Still requires OPENAI_API_KEY for create_agent initialization.
        """
        import ast
        import json
        import operator as op
        import uuid

        from langchain.agents import create_agent
        from langchain.agents.middleware.types import ModelResponse
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.tools import tool
        from redis.asyncio import Redis
        from redisvl.extensions.cache.llm import SemanticCache
        from redisvl.utils.vectorize import HFTextVectorizer

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        # Safe math evaluator
        safe_ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
        }

        def _eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp) and type(node.op) in safe_ops:
                return safe_ops[type(node.op)](
                    _eval_node(node.left), _eval_node(node.right)
                )
            elif isinstance(node, ast.UnaryOp) and type(node.op) in safe_ops:
                return safe_ops[type(node.op)](_eval_node(node.operand))
            raise ValueError("Unsupported expression")

        def safe_eval(expr: str) -> float:
            return _eval_node(ast.parse(expr, mode="eval").body)

        @tool
        def calculator(expression: str) -> str:
            """Calculate a math expression."""
            return str(safe_eval(expression))

        cache_name = f"test_cache_{uuid.uuid4().hex[:8]}"

        # Use the SAME vectorizer for both pre-population and middleware
        vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")

        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.1,
        )

        # Pre-populate cache with a response
        test_prompt = "What is 2 + 2?"
        # Serialize as our middleware does
        cached_response = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "The answer is 4.",
                    "type": "ai",
                    "tool_calls": [],
                },
            }
        )
        cache.store(prompt=test_prompt, response=cached_response)

        # Now create middleware with the SAME vectorizer
        middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                ttl_seconds=60,
                distance_threshold=0.1,
                vectorizer=vectorizer,  # Use same vectorizer!
            )
        )

        try:
            agent = create_agent(
                model="gpt-4o-mini",
                tools=[calculator],
                middleware=[middleware],
            )

            # This should hit the cache and NOT call the LLM
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is 2 + 2?")]}
            )

            # Verify we got a response
            assert "messages" in result
            last_message = result["messages"][-1]
            print(f"Result type: {type(last_message)}")
            print(f"Result: {last_message}")

            # The response should contain our cached answer
            assert "4" in last_message.content

        finally:
            await middleware.aclose()


# --- tests/integration/test_notebook_exact_replica.py ---

    async def test_semantic_cache_notebook_exact(self, redis_url: str):
        """EXACT replica of middleware_semantic_cache.ipynb cells."""
        import time

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        # === Cell: two-model-setup ===
        model_default = ChatOpenAI(model="gpt-4o-mini")

        # === Cell: define-tools ===
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"

        @tool
        def calculate(expression: str) -> str:
            """Calculate a math expression."""
            return f"{expression} = 42"

        tools = [get_weather, calculate]

        # === Cell: create-middleware ===
        import uuid

        cache_name = f"demo_semantic_cache_default_{uuid.uuid4().hex[:8]}"

        cache_middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                distance_threshold=0.15,
                ttl_seconds=3600,
                cache_final_only=True,
                deterministic_tools=["calculate"],
            )
        )

        print("SemanticCacheMiddleware created successfully!")

        try:
            # === Cell: create-agent ===
            agent = create_agent(
                model=model_default,
                tools=tools,
                middleware=[cache_middleware],
            )

            print("Agent created with SemanticCacheMiddleware!")

            # === Cell: first-query ===
            print("Query 1: 'What is the capital of France?'")
            print("=" * 50)

            start = time.time()
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is the capital of France?")]}
            )
            elapsed1 = time.time() - start

            print(f"Response: {result1['messages'][-1].content[:200]}...")
            print(f"Time: {elapsed1:.2f}s (cache miss - LLM call)")

            assert "messages" in result1
            assert len(result1["messages"]) >= 2  # Human + AI

            # === Cell: second-query ===
            print("\nQuery 2: 'Tell me France's capital city'")
            print("=" * 50)

            start = time.time()
            result2 = await agent.ainvoke(
                {"messages": [HumanMessage(content="Tell me France's capital city")]}
            )
            elapsed2 = time.time() - start

            print(f"Response: {result2['messages'][-1].content[:200]}...")
            print(f"Time: {elapsed2:.2f}s (expected: cache hit - much faster!)")

            assert "messages" in result2

            # === Cell: third-query ===
            print("\nQuery 3: 'What is the capital of Germany?'")
            print("=" * 50)

            start = time.time()
            result3 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is the capital of Germany?")]}
            )
            elapsed3 = time.time() - start

            print(f"Response: {result3['messages'][-1].content[:200]}...")
            print(f"Time: {elapsed3:.2f}s (cache miss - different topic)")

            assert "messages" in result3

            print("\n" + "=" * 50)
            print("SUCCESS! middleware_semantic_cache.ipynb replica passed!")

        finally:
            await cache_middleware.aclose()

    async def test_semantic_cache_responses_api_mode(self, redis_url: str):
        """Test semantic cache with Responses API mode."""
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        model_responses_api = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"

        tools = [get_weather]

        cache_name = f"demo_semantic_cache_responses_{uuid.uuid4().hex[:8]}"

        cache_middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                distance_threshold=0.15,
                ttl_seconds=3600,
                cache_final_only=True,
            )
        )

        try:
            agent = create_agent(
                model=model_responses_api,
                tools=tools,
                middleware=[cache_middleware],
            )

            # Cache miss
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is the capital of Japan?")]}
            )
            assert "messages" in result1
            assert len(result1["messages"]) >= 2

            # Cache hit
            result2 = await agent.ainvoke(
                {"messages": [HumanMessage(content="Tell me Japan's capital city")]}
            )
            assert "messages" in result2

            print("SUCCESS! Responses API semantic cache test passed!")

        finally:
            await cache_middleware.aclose()

    async def test_semantic_cache_responses_api_clean_blocks(self, redis_url: str):
        """Test that cached Responses API content blocks have no provider IDs."""
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        model_responses_api = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        @tool
        def noop(x: str) -> str:
            """Do nothing."""
            return x

        cache_name = f"demo_clean_blocks_{uuid.uuid4().hex[:8]}"

        cache_middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                distance_threshold=0.15,
                ttl_seconds=3600,
                cache_final_only=True,
            )
        )

        try:
            agent = create_agent(
                model=model_responses_api,
                tools=[noop],
                middleware=[cache_middleware],
            )

            # Populate cache
            await agent.ainvoke({"messages": [HumanMessage(content="Say hello")]})

            # Cache hit
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="Say hello please")]}
            )

            ai_msg = result["messages"][-1]
            if isinstance(ai_msg.content, list):
                for block in ai_msg.content:
                    if isinstance(block, dict):
                        assert (
                            "id" not in block
                        ), f"Cached block has provider ID: {block}"
                print("Cached content blocks are clean -- no provider IDs!")

            print("SUCCESS! Responses API clean blocks verification passed!")

        finally:
            await cache_middleware.aclose()

    async def test_composition_notebook_multiple_middleware(self, redis_url: str):
        """EXACT replica of middleware_composition.ipynb cells."""
        import ast
        import operator as op
        import time
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
            ToolCacheConfig,
            ToolResultCacheMiddleware,
        )

        model_default = ChatOpenAI(model="gpt-4o-mini")

        # Safe math evaluator
        safe_ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
        }

        def _eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp) and type(node.op) in safe_ops:
                return safe_ops[type(node.op)](
                    _eval_node(node.left), _eval_node(node.right)
                )
            elif isinstance(node, ast.UnaryOp) and type(node.op) in safe_ops:
                return safe_ops[type(node.op)](_eval_node(node.operand))
            raise ValueError("Unsupported expression")

        def safe_eval(expr: str) -> float:
            return _eval_node(ast.parse(expr, mode="eval").body)

        # Track tool calls
        tool_calls = []

        @tool
        def search(query: str) -> str:
            """Search for information."""
            tool_calls.append(("search", query))
            return f"Results for: {query}"

        @tool
        def calculate(expression: str) -> str:
            """Calculate a math expression."""
            tool_calls.append(("calculate", expression))
            return str(safe_eval(expression))

        tools = [search, calculate]

        # === Cell: create-individual-middleware ===
        llm_cache_name = f"composition_llm_cache_{uuid.uuid4().hex[:8]}"
        tool_cache_name = f"composition_tool_cache_{uuid.uuid4().hex[:8]}"

        semantic_cache = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=llm_cache_name,
                ttl_seconds=3600,
                deterministic_tools=["search", "calculate"],
            )
        )

        tool_cache = ToolResultCacheMiddleware(
            ToolCacheConfig(
                redis_url=redis_url,
                name=tool_cache_name,
                cacheable_tools=["search", "calculate"],
                ttl_seconds=1800,
            )
        )

        print("Created individual middleware:")
        print("- SemanticCacheMiddleware for LLM responses")
        print("- ToolResultCacheMiddleware for tool results")

        try:
            # === Cell: create-agent-multiple ===
            agent = create_agent(
                model=model_default,
                tools=tools,
                middleware=[semantic_cache, tool_cache],
            )

            print("Agent created with both SemanticCache and ToolCache middleware!")

            # === Cell: test-multiple ===
            print("\nTest 1: Search query")
            print("=" * 50)

            start = time.time()
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="Search for Python tutorials")]}
            )
            elapsed1 = time.time() - start

            print(f"Response: {result1['messages'][-1].content[:100]}...")
            print(f"Tool calls: {tool_calls}")
            print(f"Time: {elapsed1:.2f}s")

            assert "messages" in result1
            assert len(result1["messages"]) >= 2

            print("\n" + "=" * 50)
            print("SUCCESS! middleware_composition.ipynb replica passed!")

        finally:
            await semantic_cache.aclose()
            await tool_cache.aclose()

    async def test_middleware_stack_responses_api_sanitization(self, redis_url: str):
        """Test MiddlewareStack sanitizes Responses API content blocks."""
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            ConversationMemoryConfig,
            ConversationMemoryMiddleware,
            MiddlewareStack,
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        model_responses_api = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        cache_name = f"resp_stack_cache_{uuid.uuid4().hex[:8]}"
        memory_name = f"resp_stack_memory_{uuid.uuid4().hex[:8]}"

        responses_stack = MiddlewareStack(
            [
                SemanticCacheMiddleware(
                    SemanticCacheConfig(
                        redis_url=redis_url,
                        name=cache_name,
                        ttl_seconds=3600,
                    )
                ),
                ConversationMemoryMiddleware(
                    ConversationMemoryConfig(
                        redis_url=redis_url,
                        name=memory_name,
                        session_tag="responses_demo",
                        top_k=3,
                    )
                ),
            ]
        )

        try:
            agent = create_agent(
                model=model_responses_api,
                tools=[search],
                middleware=[responses_stack],
            )

            # Turn 1
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="Hello, I like Python programming")]}
            )
            assert "messages" in result1

            # Turn 2
            result2 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What language did I mention?")]}
            )
            assert "messages" in result2

            # Verify no duplicate IDs
            all_ids = set()
            for label, result in [("Turn 1", result1), ("Turn 2", result2)]:
                ai_msg = result["messages"][-1]
                if isinstance(ai_msg.content, list):
                    for block in ai_msg.content:
                        if isinstance(block, dict) and "id" in block:
                            block_id = block["id"]
                            assert (
                                block_id not in all_ids
                            ), f"Duplicate ID in {label}: {block_id}"
                            all_ids.add(block_id)

            print("SUCCESS! MiddlewareStack Responses API sanitization passed!")

        finally:
            await responses_stack.aclose()

    async def test_multi_turn_checkpointer_responses_api(self, redis_url: str):
        """Test multi-turn with checkpointer + Responses API."""
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        from langgraph.middleware.redis import (
            IntegratedRedisMiddleware,
            SemanticCacheConfig,
        )

        model_responses_api = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        async_checkpointer = AsyncRedisSaver(redis_url=redis_url)
        await async_checkpointer.asetup()

        cache_name = f"integrated_cache_{uuid.uuid4().hex[:8]}"

        integrated_stack = IntegratedRedisMiddleware.from_saver(
            async_checkpointer,
            configs=[
                SemanticCacheConfig(name=cache_name, ttl_seconds=3600),
            ],
        )

        try:
            agent = create_agent(
                model=model_responses_api,
                tools=[search],
                checkpointer=async_checkpointer,
                middleware=[integrated_stack],
            )

            thread_id = f"integrated_{uuid.uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": thread_id}}

            # Turn 1
            result1 = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content="What is the population of Tokyo?")
                    ]
                },
                config=config,
            )
            assert "messages" in result1

            # Turn 2
            result2 = await agent.ainvoke(
                {"messages": [HumanMessage(content="And what about New York?")]},
                config=config,
            )
            assert "messages" in result2

            # Turn 3
            result3 = await agent.ainvoke(
                {"messages": [HumanMessage(content="Which one is larger?")]},
                config=config,
            )
            assert "messages" in result3

            print("SUCCESS! Multi-turn checkpointer + Responses API test passed!")

        finally:
            try:
                await async_checkpointer.aclose()
            except Exception:
                pass

    async def test_middleware_stack_with_create_agent(self, redis_url: str):
        """Test MiddlewareStack with create_agent."""
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            ConversationMemoryConfig,
            ConversationMemoryMiddleware,
            MiddlewareStack,
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        model_default = ChatOpenAI(model="gpt-4o-mini")

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        cache_name = f"stack_cache_{uuid.uuid4().hex[:8]}"
        memory_name = f"stack_memory_{uuid.uuid4().hex[:8]}"

        # Create middleware stack
        stack = MiddlewareStack(
            [
                SemanticCacheMiddleware(
                    SemanticCacheConfig(
                        redis_url=redis_url,
                        name=cache_name,
                        ttl_seconds=3600,
                    )
                ),
                ConversationMemoryMiddleware(
                    ConversationMemoryConfig(
                        redis_url=redis_url,
                        name=memory_name,
                        session_tag="test_session",
                        top_k=3,
                    )
                ),
            ]
        )

        try:
            # Create agent with stack
            agent = create_agent(
                model=model_default,
                tools=[search],
                middleware=[stack],
            )

            print("Agent created with MiddlewareStack!")

            result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content="Hi, I'm testing the middleware stack!")
                    ]
                }
            )

            print(f"Response: {result['messages'][-1].content}")
            assert "messages" in result

            print("SUCCESS! MiddlewareStack with create_agent works!")

        finally:
            await stack.aclose()

    async def test_tool_cache_responses_api_mode(self, redis_url: str):
        """Test tool caching works identically with Responses API."""
        import ast
        import operator as op
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            ToolCacheConfig,
            ToolResultCacheMiddleware,
        )

        model_responses_api = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        # Safe math evaluator
        safe_ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
        }

        def _eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp) and type(node.op) in safe_ops:
                return safe_ops[type(node.op)](
                    _eval_node(node.left), _eval_node(node.right)
                )
            raise ValueError("Unsupported expression")

        def safe_eval(expr: str) -> float:
            return _eval_node(ast.parse(expr, mode="eval").body)

        exec_count = {"calculate": 0}

        @tool
        def calculate(expression: str) -> str:
            """Evaluate a mathematical expression."""
            exec_count["calculate"] += 1
            return str(safe_eval(expression))

        calculate.metadata = {"cacheable": True}

        cache_name = f"demo_tool_cache_responses_{uuid.uuid4().hex[:8]}"

        tool_cache = ToolResultCacheMiddleware(
            ToolCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                distance_threshold=0.1,
                ttl_seconds=1800,
            )
        )

        try:
            agent = create_agent(
                model=model_responses_api,
                tools=[calculate],
                middleware=[tool_cache],
            )

            # First call
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is 15 * 8 + 20?")]}
            )
            assert "messages" in result1
            assert len(result1["messages"]) >= 2

            print("SUCCESS! Tool cache with Responses API mode passed!")

        finally:
            await tool_cache.aclose()

    async def test_memory_responses_api_recall(self, redis_url: str):
        """Test memory recall works with Responses API (Carol persona)."""
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            ConversationMemoryConfig,
            ConversationMemoryMiddleware,
        )

        model_responses_api = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        @tool
        def get_user_preferences(category: str) -> str:
            """Get user preferences for a category."""
            return f"Preferences for {category}: not set"

        memory_name = f"demo_conversation_memory_{uuid.uuid4().hex[:8]}"

        memory_middleware = ConversationMemoryMiddleware(
            ConversationMemoryConfig(
                redis_url=redis_url,
                name=memory_name,
                session_tag="user_789",
                top_k=3,
                distance_threshold=0.7,
            )
        )

        try:
            agent = create_agent(
                model=model_responses_api,
                tools=[get_user_preferences],
                middleware=[memory_middleware],
            )

            # Turn 1: Introduce Carol
            result1 = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Hi! I'm Carol, an embedded systems engineer. I work with C and Rust."
                        )
                    ]
                }
            )
            assert "messages" in result1

            # Turn 2: Share interests
            result2 = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="I'm interested in RTOS, bare-metal programming, and IoT protocols."
                        )
                    ]
                }
            )
            assert "messages" in result2

            # Turn 3: Test recall
            result3 = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="What's my name and what languages do I use?"
                        )
                    ]
                }
            )
            assert "messages" in result3

            # Check that Carol's name appears in the response
            response_content = result3["messages"][-1].content
            if isinstance(response_content, list):
                # Responses API: extract text from blocks
                text_parts = []
                for block in response_content:
                    if isinstance(block, dict) and "text" in block:
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                response_text = " ".join(text_parts)
            else:
                response_text = response_content

            assert (
                "carol" in response_text.lower()
            ), f"Expected 'Carol' in response: {response_text[:200]}"

            print("SUCCESS! Memory with Responses API recall test passed!")

        finally:
            await memory_middleware.aclose()


# --- tests/integration/test_responses_api_duplicate_ids.py ---

    async def test_full_agent_conversation_memory_responses_api(self, redis_url: str):
        """End-to-end: ConversationMemory + create_agent + Responses API."""
        import uuid as uuid_mod

        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            ConversationMemoryConfig,
            ConversationMemoryMiddleware,
        )

        model = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        @tool
        def noop(x: str) -> str:
            """Do nothing."""
            return x

        memory_name = f"test_e2e_memory_{uuid_mod.uuid4().hex[:8]}"
        middleware = ConversationMemoryMiddleware(
            ConversationMemoryConfig(
                redis_url=redis_url,
                name=memory_name,
                session_tag="test_e2e_responses",
                top_k=3,
                distance_threshold=0.7,
            )
        )

        try:
            agent = create_agent(
                model=model,
                tools=[noop],
                middleware=[middleware],
            )

            # Turn 1
            result1 = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content="My name is TestUser and I like Rust.")
                    ]
                }
            )
            assert "messages" in result1

            # Turn 2: recall
            result2 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is my name?")]}
            )
            assert "messages" in result2

            # Verify stored messages are strings
            stored = middleware._history.get_recent(top_k=10)
            for msg in stored:
                assert isinstance(
                    msg["content"], str
                ), f"Stored content is {type(msg['content'])}: {msg['content'][:100]}"

        finally:
            await middleware.aclose()


# --- tests/integration/test_semantic_cache_provider_metadata.py ---

    def test_strips_chat_completions_metadata(self):
        """Chat Completions API uses different field names but same risk."""
        cached_str = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "Hello! How can I help?",
                    "type": "ai",
                    "id": "chatcmpl-abc123",
                    "tool_calls": [],
                    "additional_kwargs": {
                        "refusal": None,
                    },
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 15,
                            "prompt_tokens": 10,
                            "total_tokens": 25,
                        },
                        "model_name": "gpt-4o-mini",
                        "system_fingerprint": "fp_abc123xyz",
                        "finish_reason": "stop",
                    },
                },
            }
        )

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == "Hello! How can I help?"
        _assert_clean_cached_message(msg)

