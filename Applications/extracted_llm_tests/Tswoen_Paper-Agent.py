# Tswoen/Paper-Agent
# 1 LLM-backed test functions across 12 test files
# Source: https://github.com/Tswoen/Paper-Agent

# --- test/test_embedding_function.py ---

async def test_ark_chat_model():
    """测试 ArkChatModel 类"""
    import asyncio
    import os
    # Install SDK:  pip install 'volcengine-python-sdk[ark]'
    from volcenginesdkarkruntime import AsyncArk

    # 初始化Ark客户端
    client = AsyncArk(
        # The base URL for model invocation
        base_url="https://ark.cn-beijing.volces.com/api/v3", 
        # Get API Key：https://console.volcengine.com/ark/region:ark+cn-beijing/apikey 
        api_key=self.api_key, 
    )
    stream = await client.chat.completions.create(  
        # Replace with Model ID
        model = "doubao-seed-1-6-thinking-250715",
        messages=[
            {"role": "system", "content": "你是 AI 人工智能助手"},
            {"role": "user", "content": "常见的十字花科植物有哪些？"},
        ],
        stream=True
    )
    async for completion in stream:
        print(completion.choices[0].delta.content, end="")
    print()

