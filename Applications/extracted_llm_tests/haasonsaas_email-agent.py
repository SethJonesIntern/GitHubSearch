# haasonsaas/email-agent
# 3 LLM-backed test functions across 16 test files
# Source: https://github.com/haasonsaas/email-agent

# --- scripts/deprecated_migration_scripts/test_ai_analysis.py ---

async def test_openai_connection():
    """Test OpenAI API connection."""
    print("🔍 Testing OpenAI API connection...")
    
    if not settings.openai_api_key:
        print("❌ OpenAI API key not found in configuration")
        return False
    
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Simple test call
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, Email Agent!' if you can hear me."}
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ OpenAI API connected successfully!")
        print(f"   Model: {settings.openai_model}")
        print(f"   Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {str(e)}")
        return False

async def test_daily_brief_generation():
    """Test daily brief generation."""
    print("\n📰 Testing daily brief generation...")
    
    summarizer = SummarizerAgent()
    test_emails = create_test_emails()
    
    try:
        brief = await summarizer.generate_brief(test_emails, date.today())
        
        print(f"✅ Brief generated successfully!")
        print(f"📊 Total emails: {brief.total_emails}")
        print(f"📬 Unread emails: {brief.unread_emails}")
        print(f"📰 Headline: {brief.headline}")
        print(f"📝 Summary: {brief.summary}")
        print(f"📋 Action items: {len(brief.action_items)}")
        for item in brief.action_items:
            print(f"   - {item}")
        print(f"⏰ Deadlines: {len(brief.deadlines)}")
        for deadline in brief.deadlines:
            print(f"   - {deadline}")
        
    except Exception as e:
        print(f"❌ Brief generation failed: {str(e)}")

async def test_crew_integration():
    """Test CrewAI integration."""
    print("\n🤖 Testing CrewAI integration...")
    
    try:
        crew = EmailAgentCrew()
        await crew.initialize_crew({"verbose": False})
        
        print("✅ EmailAgentCrew initialized successfully")
        
        # Test email summarization task
        test_email = create_test_emails()[0]
        
        result = await crew.execute_task("summarize_email", email=test_email)
        print(f"✅ Email summarization task completed")
        print(f"   Summary: {result.get('summary', 'N/A')}")
        
        # Test brief generation task
        test_emails = create_test_emails()
        brief = await crew.execute_task("generate_brief", emails=test_emails, date=date.today())
        print(f"✅ Brief generation task completed")
        print(f"   Brief headline: {brief.headline}")
        
        await crew.shutdown()
        print("✅ CrewAI integration test completed")
        
    except Exception as e:
        print(f"❌ CrewAI integration failed: {str(e)}")

