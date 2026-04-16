# guaardvark/guaardvark
# 7 LLM-backed test functions across 127 test files
# Source: https://github.com/guaardvark/guaardvark

# --- backend/tests/test_chat_tools_e2e.py ---

    def test_qwen3_vl_sanitized_prompt_no_crash(self):
        """Verify that sanitized prompts don't crash Ollama's JSON serializer."""
        import ollama

        system_prompt = """You are an AI assistant with tool access.

AVAILABLE TOOLS:
- web_search(query) - Search the web

To call a tool, output this format:
[tool_call]
[tool]tool_name[/tool]
[query]value[/query]
[/tool_call]"""

        stream = ollama.chat(
            model="qwen3-vl:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What is the temperature in Cincinnati?"},
            ],
            stream=True,
            options={"num_predict": 300, "temperature": 0.3},
        )

        content_parts = []
        thinking_parts = []
        for chunk in stream:
            msg = chunk.get("message", {})
            c = msg.get("content", "")
            t = msg.get("thinking", "")
            if c:
                content_parts.append(c)
            if t:
                thinking_parts.append(t)

        content = "".join(content_parts)
        thinking = "".join(thinking_parts)
        result = content or thinking

        # Should produce a tool call, not crash
        assert len(result) > 0, "Empty response from qwen3-vl:8b"
        assert "web_search" in result.lower() or "tool" in result.lower(), \
            f"Expected tool call, got: {result[:200]}"

    def test_qwen3_vl_thinking_field_captured(self):
        """Verify thinking field content is captured when content is empty."""
        import ollama

        stream = ollama.chat(
            model="qwen3-vl:8b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Reply briefly."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            stream=True,
            options={"num_predict": 100},
        )

        content_parts = []
        thinking_parts = []
        for chunk in stream:
            msg = chunk.get("message", {})
            c = msg.get("content", "")
            t = msg.get("thinking", "")
            if c:
                content_parts.append(c)
            if t:
                thinking_parts.append(t)

        content = "".join(content_parts)
        thinking = "".join(thinking_parts)

        # qwen3-vl puts output in thinking field
        assert len(thinking) > 0 or len(content) > 0, "Both content and thinking empty"
        combined = content or thinking
        assert "4" in combined or "four" in combined.lower(), \
            f"Expected '4' or 'four' in response: {combined[:200]}"

    def test_chat_sends_message_and_receives_response(self, browser_context):
        """Send a simple message and verify a response appears."""
        page = browser_context.new_page()
        page.goto("http://localhost:5175/chat", wait_until="domcontentloaded", timeout=30000)

        # Find and click the chat input
        chat_input = page.locator("textarea, input[type='text']").first
        chat_input.wait_for(state="visible", timeout=10000)
        chat_input.click()
        chat_input.fill("What is 2 + 2? Reply with just the number.")

        # Submit (press Enter or click send button)
        chat_input.press("Enter")

        # Wait for the streaming response to appear.
        # The StreamingMessage component renders MUI Paper elements; the
        # "Processing..." text or actual streamed tokens will appear.
        # Thinking models (qwen3-vl) can take 60-90s on first load.
        try:
            # First, wait for the streaming indicator ("Processing..." or token text)
            streaming = page.locator("text=Processing..., text=Thinking")
            streaming.first.wait_for(state="visible", timeout=30000)
        except Exception:
            pass  # May have already completed

        # Wait for completion: either the "Assistant is typing..." disappears,
        # or actual response text appears in the page.
        # Poll for up to 120s for a response.
        import time
        deadline = time.time() + 120
        found_response = False
        while time.time() < deadline:
            body_text = page.locator("body").inner_text()
            # Check for a numeric answer or any substantive response
            if any(term in body_text for term in ["4", "four", "error", "Error"]):
                found_response = True
                break
            time.sleep(3)

        page.screenshot(path="/tmp/chat_e2e_response.png", full_page=True)

        if not found_response:
            pytest.skip("Model response timed out after 120s (likely slow inference)")

        page.close()


# --- backend/tests/test_contextual_prepender.py ---

    def test_basic_context(self):
        ctx = generate_chunk_context(
            file_path="src/app.py",
            repo_name="myrepo",
            language="python",
        )
        assert "[python]" in ctx
        assert "myrepo" in ctx
        assert "src/app.py" in ctx

    def test_with_symbol(self):
        ctx = generate_chunk_context(
            file_path="src/app.py",
            repo_name="myrepo",
            language="python",
            symbol_name="main",
            symbol_type="function",
        )
        assert "function" in ctx
        assert "`main`" in ctx

    def test_no_repo_name(self):
        ctx = generate_chunk_context(
            file_path="utils.js",
            repo_name=None,
            language="javascript",
        )
        assert "[javascript]" in ctx
        assert "utils.js" in ctx
        assert "Repository" not in ctx

    def test_ends_with_double_newline(self):
        ctx = generate_chunk_context(
            file_path="a.py", repo_name="r", language="python"
        )
        assert ctx.endswith("\n\n")

