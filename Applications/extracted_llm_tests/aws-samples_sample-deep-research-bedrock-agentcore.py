# aws-samples/sample-deep-research-bedrock-agentcore
# 1 LLM-backed test functions across 2 test files
# Source: https://github.com/aws-samples/sample-deep-research-bedrock-agentcore

# --- scripts/test_workflow.py ---

async def test_simple_topic():
    """Test with a simple topic"""

    # ============================================================================
    # CONFIGURATION - Modify these variables to customize the research
    # ============================================================================

    # Research topic
    TOPIC = "Python async programming basics"

    # Research type: "comprehensive", "academic", "basic_web", "advanced_web", "financial"
    RESEARCH_TYPE = "basic_web"

    # Research depth: "quick" (2x2), "balanced" (3x3), "deep" (5x3)
    RESEARCH_DEPTH = "quick"

    # LLM Model: "claude_sonnet", "claude_haiku", "nova_pro", "qwen3_mixed", "llama_maverick"
    LLM_MODEL = "llama_maverick"

    # Optional: Research context (background information, constraints, focus areas)
    RESEARCH_CONTEXT = ""  
    # Example: "Focus on enterprise applications. Exclude consumer products."

    # Optional: Reference materials (papers, articles to use as foundation)
    REFERENCE_MATERIALS = [
        # {
        #     "type": "url",
        #     "url": "https://aws.amazon.com/blogs/aws-cost-management/overview-of-the-cost-optimization-pillar/",
        #     "note": "AWS Cost Optimization Pillar Overview"
        # },
        # {
        #     "type": "url",
        #     "url": "https://cloud.google.com/architecture/framework/cost-optimization",
        #     "note": "Google Cloud Cost Optimization Framework"
        # },
        # {
        #     "type": "url",
        #     "url": "https://learn.microsoft.com/en-us/azure/well-architected/cost-optimization/",
        #     "note": "Azure Well-Architected Framework - Cost Optimization"
        # }
    ]

    # Set to a list or None to skip
    # Example:
    # REFERENCE_MATERIALS = [
    #     {
    #         "type": "arxiv",
    #         "id": "2401.12345",
    #         "note": "Research paper on cloud resource optimization"
    #     },
    #     {
    #         "type": "url",
    #         "url": "https://example.com/article",
    #         "note": "Industry perspective on FinOps practices"
    #     }
    # ]

    # ============================================================================

    print("\n" + "="*80)
    print(f"DIMENSIONAL RESEARCH TEST")
    print("="*80)
    print(f"\n📋 Topic: {TOPIC}")
    print(f"🔬 Research Type: {RESEARCH_TYPE}")
    print(f"📊 Research Depth: {RESEARCH_DEPTH}")
    print("="*80)

    # Create workflow
    app = create_workflow()

    # Create LangSmith configuration with increased recursion limit
    config = create_run_config(
        run_name="dimensional_research_test",
        tags=["test", RESEARCH_TYPE, RESEARCH_DEPTH],
        metadata={
            "research_type": RESEARCH_TYPE,
            "research_depth": RESEARCH_DEPTH,
            "environment": "test"
        }
    )
    # Set recursion limit for research agents
    # Allow reasonable exploration while preventing infinite loops
    config["recursion_limit"] = 35

    # Get research configuration and override depth
    research_config = get_research_config(RESEARCH_TYPE)
    research_config.research_depth = RESEARCH_DEPTH
    research_config.llm_model = LLM_MODEL

    # Reapply depth config
    from src.config.research_config import DEPTH_CONFIGS
    depth_config = DEPTH_CONFIGS[RESEARCH_DEPTH]
    research_config.target_dimensions = depth_config["dimensions"]
    research_config.target_aspects_per_dimension = depth_config["aspects_per_dimension"]
    research_config.arxiv_max_results = depth_config["arxiv_max_results"]
    research_config.web_search_max_results = depth_config["web_search_max_results"]

    # Add optional research context and reference materials
    research_config_dict = research_config.to_dict()
    if RESEARCH_CONTEXT:
        research_config_dict["research_context"] = RESEARCH_CONTEXT
        print(f"\n📝 Research Context Provided:")
        print(f"   {RESEARCH_CONTEXT[:200]}{'...' if len(RESEARCH_CONTEXT) > 200 else ''}")

    if REFERENCE_MATERIALS:
        research_config_dict["reference_materials"] = REFERENCE_MATERIALS
        print(f"\n📚 Reference Materials: {len(REFERENCE_MATERIALS)} item(s)")

    # Execute workflow
    start_time = time.time()

    # Generate unique session ID for test
    test_session_id = f"test_{uuid.uuid4().hex[:8]}"
    print(f"\n🔑 Test Session ID: {test_session_id}")

    result = await app.ainvoke(
        {
            "topic": TOPIC,
            "bff_session_id": test_session_id,  # Add session ID for initialize_session
            "research_config": research_config_dict,  # Pass config with optional fields
            "workflow_start_time": start_time
        },
        config=config
    )

    # Print results
    print("\n" + "="*80)
    print("RESEARCH RESULTS")
    print("="*80)

    print(f"\nTopic: {result.get('topic')}")
    print(f"\nDimensions ({len(result.get('dimensions', []))}):")
    for idx, dim in enumerate(result.get('dimensions', []), 1):
        print(f"  {idx}. {dim}")

    print(f"\nAspects by Dimension:")
    for dim, aspects in result.get('aspects_by_dimension', {}).items():
        print(f"\n  {dim}:")
        for aspect in aspects:
            print(f"    - {aspect}")

    print(f"\nResearch Findings:")
    for aspect_key, research in result.get('research_by_aspect', {}).items():
        print(f"\n  {aspect_key}:")
        if isinstance(research, dict):
            print(f"    Title: {research.get('title', 'N/A')}")
            print(f"    Content: {research.get('word_count', 0)} words")
            # Preview first 300 chars of content
            content = research.get('content', '')
            print(f"    Content Preview: {content[:300]}{'...' if len(content) > 300 else ''}")
        else:
            # Fallback for non-structured results
            print(f"    Length: {len(research)} characters")
            print(f"    Preview: {research[:200]}...")

    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f}s")

    return result

