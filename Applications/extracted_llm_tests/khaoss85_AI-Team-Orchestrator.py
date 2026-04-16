# khaoss85/AI-Team-Orchestrator
# 2 LLM-backed test functions across 160 test files
# Source: https://github.com/khaoss85/AI-Team-Orchestrator

# --- tests/integration/test_ai_driven_complete_e2e.py ---

async def test_complete_ai_driven_e2e():
    """Test completo end-to-end del sistema AI-driven"""
    
    print("🎯 AI-DRIVEN COMPLETE END-TO-END TEST")
    print("="*80)
    print("Testing complete flow from workspace creation to deliverable generation")
    print("Focus: AI-driven intent analysis producing real business data")
    print("="*80)
    
    try:
        # Step 1: Create workspace with specific data goal
        print("\n📋 STEP 1: Creating workspace with specific contact data goal")
        workspace_id = await create_test_workspace()
        print(f"✅ Created workspace: {workspace_id}")
        
        # Step 2: Test AI-driven Director team creation
        print("\n🧠 STEP 2: Testing AI-driven Director with intent analysis")
        proposal_result = await test_ai_director_enhancement(workspace_id)
        print(f"✅ Director created team with {len(proposal_result.get('agents', []))} agents")
        
        # Step 3: Approve team and create agents
        print("\n👥 STEP 3: Approving team and creating agents in database")
        agents_created = await approve_and_create_team(workspace_id, proposal_result)
        print(f"✅ Created {len(agents_created)} agents in database")
        
        # Step 4: Test AI-driven task execution
        print("\n🚀 STEP 4: Testing AI-driven task execution with intent analysis")
        execution_results = await test_ai_task_execution(workspace_id, agents_created)
        print(f"✅ Executed {len(execution_results)} tasks with AI classification")
        
        # Step 5: Validate AI-driven output analysis
        print("\n🔍 STEP 5: Validating AI-driven output specificity analysis")
        validation_results = await validate_output_specificity(execution_results)
        print(f"✅ AI validation completed for {len(validation_results)} outputs")
        
        # Step 6: Test deliverable creation with real business data
        print("\n📦 STEP 6: Testing deliverable creation with business data")
        deliverable_results = await test_deliverable_generation(workspace_id, execution_results)
        print(f"✅ Generated {len(deliverable_results)} deliverables")
        
        # Step 7: Final analysis - verify real business data
        print("\n📊 STEP 7: Final analysis - verify real vs methodology content")
        final_analysis = await analyze_final_business_value(deliverable_results, validation_results)
        
        # Results summary
        print("\n" + "="*80)
        print("🎉 AI-DRIVEN E2E TEST RESULTS")
        print("="*80)
        
        print(f"Workspace ID: {workspace_id}")
        print(f"Agents Created: {len(agents_created)}")
        print(f"Tasks Executed: {len(execution_results)}")
        print(f"AI Validations: {len(validation_results)}")
        print(f"Deliverables: {len(deliverable_results)}")
        
        # Key metrics
        specific_data_tasks = sum(1 for r in execution_results if r.get('classification', {}).get('output_specificity') == 'specific_data')
        ai_validations_passed = sum(1 for v in validation_results if v.get('contains_specific_data', False))
        business_ready_deliverables = sum(1 for d in deliverable_results if d.get('business_ready', False))
        
        print(f"\n📈 KEY METRICS:")
        print(f"   Specific Data Tasks: {specific_data_tasks}/{len(execution_results)}")
        print(f"   AI Validations Passed: {ai_validations_passed}/{len(validation_results)}")
        print(f"   Business-Ready Deliverables: {business_ready_deliverables}/{len(deliverable_results)}")
        
        # Success criteria
        success_rate = (ai_validations_passed / max(len(validation_results), 1)) * 100
        
        print(f"\n🎯 SUCCESS CRITERIA:")
        print(f"   AI-Driven Success Rate: {success_rate:.1f}%")
        print(f"   Real Business Data: {'✅ YES' if final_analysis['has_real_data'] else '❌ NO'}")
        print(f"   No Hard-Coded Logic: {'✅ YES' if final_analysis['ai_driven_only'] else '❌ NO'}")
        print(f"   Uses Existing Components: {'✅ YES' if final_analysis['no_new_silos'] else '❌ NO'}")
        
        overall_success = (
            success_rate >= 80 and 
            final_analysis['has_real_data'] and 
            final_analysis['ai_driven_only'] and
            final_analysis['no_new_silos']
        )
        
        if overall_success:
            print(f"\n🎉 OVERALL RESULT: ✅ SUCCESS - AI-Driven system working correctly!")
            print("   The system now produces real business data instead of methodologies")
            print("   All components are AI-driven without hard-coded keywords")
            print("   System uses existing components without creating new silos")
        else:
            print(f"\n❌ OVERALL RESULT: NEEDS IMPROVEMENT")
            print("   Some aspects of the AI-driven system need further refinement")
        
        return {
            'success': overall_success,
            'workspace_id': workspace_id,
            'metrics': {
                'success_rate': success_rate,
                'specific_data_tasks': specific_data_tasks,
                'ai_validations_passed': ai_validations_passed,
                'business_ready_deliverables': business_ready_deliverables
            },
            'analysis': final_analysis
        }
        
    except Exception as e:
        print(f"\n💥 E2E TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

async def test_ai_director_enhancement(workspace_id):
    """Step 2: Test AI-driven Director with intent analysis"""
    
    from ai_agents.director import DirectorAgent
    from models import DirectorTeamProposal, BudgetConstraint
    from services.ai_driven_director_enhancer import enhance_director_with_ai_intent
    
    # Create proposal request
    proposal_request = DirectorTeamProposal(
        workspace_id=workspace_id,
        workspace_goal='Genera lista contatti SaaS B2B qualificati con dati reali per outbound marketing',
        user_feedback='Ho bisogno di contatti reali con nomi, email e aziende specifiche - non strategie o metodologie',
        budget_constraint=BudgetConstraint(max_cost=100.0, currency='USD'),
        extracted_goals=[
            {
                'type': 'contact_extraction',
                'description': 'Lista 20 contatti SaaS marketing managers con email verificate',
                'metrics': {'contacts': 20, 'format': 'CSV', 'fields': ['nome', 'email', 'azienda', 'ruolo', 'linkedin']}
            }
        ]
    )
    
    # Test AI enhancement
    print("   🧠 Testing AI intent analysis...")
    ai_enhancement = await enhance_director_with_ai_intent(
        workspace_goal=proposal_request.workspace_goal,
        user_feedback=proposal_request.user_feedback,
        budget_constraint=proposal_request.budget_constraint.model_dump(),
        extracted_goals=proposal_request.extracted_goals
    )
    
    print(f"   ✅ AI enhancement completed")
    print(f"   🧠 Intent analysis: {len(ai_enhancement.get('intent_summary', '[]'))} intents detected")
    
    # Create team with enhanced feedback
    director = DirectorAgent()
    enhanced_request = proposal_request.model_copy()
    enhanced_request.user_feedback = ai_enhancement['enhanced_user_feedback']
    
    proposal = await director.create_team_proposal(enhanced_request)
    
    return {
        'proposal': proposal,
        'ai_enhancement': ai_enhancement,
        'agents': proposal.agents
    }

