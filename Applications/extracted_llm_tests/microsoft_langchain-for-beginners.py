# microsoft/langchain-for-beginners
# 1 LLM-backed test functions across 1 test files
# Source: https://github.com/microsoft/langchain-for-beginners

# --- scripts/test_setup.py ---

def test_setup():
    """Test AI provider connection and configuration."""
    print("🚀 Testing AI provider connection...\n")
    
    # Load environment variables
    load_dotenv()
    
    # Check if required variables are set
    if not os.getenv("AI_API_KEY"):
        print("❌ ERROR: AI_API_KEY not found in .env file")
        sys.exit(1)
    
    if not os.getenv("AI_ENDPOINT"):
        print("❌ ERROR: AI_ENDPOINT not found in .env file")
        sys.exit(1)
    
    try:
        model = ChatOpenAI(
            model=os.getenv("AI_MODEL", "gpt-5-mini"),
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )
        
        response = model.invoke("Say 'Setup successful!'")
        
        print("✅ SUCCESS! Your AI provider is working!")
        print(f"   Provider: {os.getenv('AI_ENDPOINT')}")
        print(f"   Model: {os.getenv('AI_MODEL', 'gpt-5-mini')}")
        print(f"\nModel response: {response.content}")
        print("\n🎉 You're ready to start the course!")
    except Exception as error:
        print(f"❌ ERROR: {str(error)}")
        print("\nTroubleshooting:")
        print("1. Check your AI_API_KEY in .env file")
        print("2. Verify the AI_ENDPOINT is correct")
        print("3. Ensure the AI_MODEL is valid for your provider")
        print("4. Verify the token/key has no extra spaces")
        sys.exit(1)

