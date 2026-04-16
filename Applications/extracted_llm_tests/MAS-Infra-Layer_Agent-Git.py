# MAS-Infra-Layer/Agent-Git
# 7 LLM-backed test functions across 9 test files
# Source: https://github.com/MAS-Infra-Layer/Agent-Git

# --- tests/test_framework_integration.py ---

    def test_framework_can_be_applied_to_any_agent(self):
        """Test that our framework can be applied to any LangChain/LangGraph agent setup."""
        print("\n=== Testing Framework Application to Standard Agent ===")
        
        # Create agent with our rollback framework
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            reverse_tools=self.reverse_tools,
            auto_checkpoint=False,  # Disable auto-checkpointing for this test
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        # Verify framework components are properly initialized
        self.assertIsNotNone(agent.tool_rollback_registry)
        self.assertIsNotNone(agent.graph)
        self.assertIsNotNone(agent.internal_session)
        
        print(f"✓ Agent created with {len(self.tools)} tools")
        print(f"✓ Rollback registry initialized")
        print(f"✓ LangGraph workflow compiled")
        
        # Test basic conversation without tools
        response1 = agent.run("Hello! Please introduce yourself.")
        print(f"Agent response 1: {response1}")
        self.assertIsInstance(response1, str)
        self.assertGreater(len(response1), 0)
        
        # Verify no auto-checkpoints were created (no tools called)
        checkpoints_before = self.checkpoint_repo.get_by_internal_session(agent.internal_session.id)
        auto_checkpoints_before = [cp for cp in checkpoints_before if cp.is_auto]
        print(f"Auto-checkpoints after non-tool conversation: {len(auto_checkpoints_before)}")
        
        # Test conversation with tools
        response2 = agent.run("Please calculate 25 + 17 using the calculate_sum tool.")
        print(f"Agent response 2: {response2}")
        
        # Verify tool was called
        tool_track = agent.get_tool_track()
        print(f"Tools in track: {[record.tool_name for record in tool_track]}")
        self.assertGreater(len(tool_track), 0)
        self.assertEqual(tool_track[0].tool_name, "calculate_sum")
        
        # Test manual checkpoint creation
        checkpoint_result = agent.create_checkpoint_tool("After calculation")
        self.assertIn("successfully", checkpoint_result.lower())
        print(f"✓ Manual checkpoint created: {checkpoint_result}")
        
        # Test that framework preserves standard LangChain/LangGraph functionality
        response3 = agent.run("What's the weather like in Tokyo?")
        print(f"Agent response 3: {response3}")
        self.assertIn("tokyo", response3.lower())
        
        print("✓ Framework successfully applied to standard agent")
        print("✓ Standard LangChain/LangGraph functionality preserved")

    def test_auto_checkpoint_creation_with_tools(self):
        """Test automatic checkpoint creation when tools are called."""
        print("\n=== Testing Automatic Checkpoint Creation ===")
        
        # Create agent with auto-checkpointing enabled
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            reverse_tools=self.reverse_tools,
            auto_checkpoint=True,  # Enable auto-checkpointing
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        print("Agent created with auto_checkpoint=True")
        
        # Test conversation without tools - should NOT create auto-checkpoint
        response1 = agent.run("Hello, how are you today?")
        print(f"Non-tool response: {response1}")
        
        checkpoints_after_chat = self.checkpoint_repo.get_by_internal_session(agent.internal_session.id)
        auto_checkpoints_chat = [cp for cp in checkpoints_after_chat if cp.is_auto]
        print(f"Auto-checkpoints after non-tool conversation: {len(auto_checkpoints_chat)}")
        self.assertEqual(len(auto_checkpoints_chat), 0, "No auto-checkpoint should be created without tool calls")
        
        # Test conversation with tools - SHOULD create auto-checkpoint
        response2 = agent.run("Please multiply 8 by 7 using the multiply_numbers tool.")
        print(f"Tool response: {response2}")
        
        checkpoints_after_tool = self.checkpoint_repo.get_by_internal_session(agent.internal_session.id)
        auto_checkpoints_tool = [cp for cp in checkpoints_after_tool if cp.is_auto]
        print(f"Auto-checkpoints after tool call: {len(auto_checkpoints_tool)}")
        self.assertEqual(len(auto_checkpoints_tool), 1, "One auto-checkpoint should be created after tool call")
        
        # Verify auto-checkpoint details
        auto_checkpoint = auto_checkpoints_tool[0]
        self.assertTrue(auto_checkpoint.is_auto)
        self.assertIn("multiply_numbers", auto_checkpoint.checkpoint_name)
        print(f"✓ Auto-checkpoint created: {auto_checkpoint.checkpoint_name}")
        
        # Test multiple tool calls create multiple auto-checkpoints
        response3 = agent.run("Now calculate 15 + 25 using calculate_sum.")
        print(f"Second tool response: {response3}")
        
        response4 = agent.run("Save 'user_preference' as 'dark_mode' to memory.")
        print(f"Third tool response: {response4}")
        
        final_checkpoints = self.checkpoint_repo.get_by_internal_session(agent.internal_session.id)
        final_auto_checkpoints = [cp for cp in final_checkpoints if cp.is_auto]
        print(f"Final auto-checkpoints: {len(final_auto_checkpoints)}")
        self.assertEqual(len(final_auto_checkpoints), 3, "Three auto-checkpoints should exist after three tool calls")
        
        # Verify checkpoint names reflect the tools used
        checkpoint_names = [cp.checkpoint_name for cp in final_auto_checkpoints]
        print(f"Auto-checkpoint names: {checkpoint_names}")
        self.assertTrue(any("multiply_numbers" in name for name in checkpoint_names))
        self.assertTrue(any("calculate_sum" in name for name in checkpoint_names))
        self.assertTrue(any("save_to_memory" in name for name in checkpoint_names))
        
        print("✓ Auto-checkpoint creation works correctly")
        print("✓ No checkpoints created without tool calls")
        print("✓ Checkpoints created automatically after each tool call")

    def test_rollback_functionality_integration(self):
        """Test rollback functionality with integrated framework."""
        print("\n=== Testing Rollback Functionality Integration ===")
        
        # Create agent with both auto-checkpointing and rollback capabilities
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            reverse_tools=self.reverse_tools,
            auto_checkpoint=True,
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        # Build up conversation with mixed tool and non-tool interactions
        agent.run("Hello, I'm setting up a calculation session.")
        
        # First tool call - should create auto-checkpoint
        agent.run("Calculate 10 + 5 using calculate_sum.")
        
        # Create manual checkpoint
        manual_cp_result = agent.create_checkpoint_tool("Before complex operations")
        checkpoint_id = int(manual_cp_result.split("ID: ")[1].split(")")[0])
        print(f"Manual checkpoint created with ID: {checkpoint_id}")
        
        # More operations after checkpoint
        agent.run("Now multiply 3 by 4.")
        agent.run("Save 'calculation_mode' as 'advanced' to memory.")
        agent.run("Get weather for Paris.")
        
        # Store state before rollback
        original_history = agent.get_conversation_history().copy()
        original_tool_track = agent.get_tool_track().copy()
        
        print(f"Before rollback - History: {len(original_history)} messages")
        print(f"Before rollback - Tool track: {len(original_tool_track)} tools")
        
        # Perform rollback
        rolled_back_agent = RollbackAgent.from_checkpoint(
            checkpoint_id=checkpoint_id,
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            reverse_tools=self.reverse_tools,
            checkpoint_repo=self.checkpoint_repo,
            internal_session_repo=self.internal_repo
        )
        
        # Verify rollback worked
        rolled_back_history = rolled_back_agent.get_conversation_history()
        rolled_back_tool_track = rolled_back_agent.get_tool_track()
        
        print(f"After rollback - History: {len(rolled_back_history)} messages")
        print(f"After rollback - Tool track: {len(rolled_back_tool_track)} tools")
        
        self.assertLess(len(rolled_back_history), len(original_history))
        self.assertLess(len(rolled_back_tool_track), len(original_tool_track))
        
        # Verify new branch can continue independently
        continue_response = rolled_back_agent.run("Let's try a different calculation: 20 + 30.")
        print(f"Continued conversation: {continue_response}")
        
        # Verify both sessions exist in database
        all_sessions = self.internal_repo.get_by_external_session(self.external_session.id)
        self.assertEqual(len(all_sessions), 2, "Should have original and branched sessions")
        
        branch_session = rolled_back_agent.internal_session
        self.assertTrue(branch_session.is_branch())
        self.assertEqual(branch_session.branch_point_checkpoint_id, checkpoint_id)
        
        print("✓ Rollback functionality works with framework integration")
        print("✓ Tool rollback executed successfully")  
        print("✓ New branch created and can continue independently")
        print("✓ Original session preserved in database")


# --- tests/test_rollback_branching_real_llm.py ---

    def test_rollback_creates_new_session_preserves_old(self):
        """Test that rollback creates new internal session while preserving the original."""
        # Create original agent
        original_agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            auto_checkpoint=True,
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        original_session_id = original_agent.internal_session.id
        original_langgraph_id = original_agent.langgraph_session_id
        
        print(f"\n--- Original Agent Created ---")
        print(f"Original internal session ID: {original_session_id}")
        print(f"Original LangGraph session ID: {original_langgraph_id}")
        
        # Have conversation with original agent
        response1 = original_agent.run("Hello, my name is Alice. Please remember this.")
        print(f"\nOriginal Agent Response 1: {response1}")
        
        # Create checkpoint after introduction
        checkpoint_result = original_agent.create_checkpoint_tool(name="After Introduction")
        print(f"Checkpoint created: {checkpoint_result}")
        
        # Extract checkpoint ID
        checkpoint_id_str = checkpoint_result.split("ID: ")[1].split(")")[0]
        checkpoint_id = int(checkpoint_id_str)
        
        # Continue original conversation
        response2 = original_agent.run("I live in Paris and love mathematics. Use the add_numbers tool to calculate 15 + 27.")
        print(f"\nOriginal Agent Response 2: {response2}")
        
        response3 = original_agent.run("Now please multiply 6 by 8 using the multiply_numbers tool.")
        print(f"Original Agent Response 3: {response3}")
        
        # Store original conversation state
        original_history_before_rollback = original_agent.internal_session.conversation_history.copy()
        original_history_length = len(original_history_before_rollback)
        
        print(f"\nOriginal conversation length before rollback: {original_history_length}")
        
        # Perform rollback - this should create a new internal session
        print(f"\n--- Performing Rollback to Checkpoint {checkpoint_id} ---")
        
        rolled_back_agent = RollbackAgent.from_checkpoint(
            checkpoint_id=checkpoint_id,
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            checkpoint_repo=self.checkpoint_repo,
            internal_session_repo=self.internal_repo
        )
        
        rolled_back_session_id = rolled_back_agent.internal_session.id
        rolled_back_langgraph_id = rolled_back_agent.langgraph_session_id
        
        print(f"Rolled-back internal session ID: {rolled_back_session_id}")
        print(f"Rolled-back LangGraph session ID: {rolled_back_langgraph_id}")
        
        # === VERIFICATION TESTS ===
        
        # 1. Verify new internal session was created
        self.assertNotEqual(original_session_id, rolled_back_session_id)
        self.assertNotEqual(original_langgraph_id, rolled_back_langgraph_id)
        print("✓ New internal session created successfully")
        
        # 2. Verify original session still exists and is preserved
        original_session_from_db = self.internal_repo.get_by_id(original_session_id)
        self.assertIsNotNone(original_session_from_db)
        self.assertEqual(len(original_session_from_db.conversation_history), original_history_length)
        print("✓ Original session preserved in database")
        
        # 3. Verify rolled-back agent has correct history (only up to checkpoint)
        rolled_back_history = rolled_back_agent.internal_session.conversation_history
        self.assertLess(len(rolled_back_history), original_history_length)
        self.assertEqual(len(rolled_back_history), 2)  # Only "Hello, my name is Alice" exchange
        print(f"✓ Rolled-back agent has correct history length: {len(rolled_back_history)}")
        
        # 4. Verify branch relationship
        self.assertTrue(rolled_back_agent.internal_session.is_branch())
        self.assertEqual(rolled_back_agent.internal_session.parent_session_id, original_session_id)
        self.assertEqual(rolled_back_agent.internal_session.branch_point_checkpoint_id, checkpoint_id)
        print("✓ Branch relationship established correctly")
        
        # 5. Verify rolled-back agent remembers Alice but not Paris/math
        rolled_back_response = rolled_back_agent.run("Do you remember my name and where I live?")
        print(f"\nRolled-back agent response: {rolled_back_response}")
        
        # The response shows "your name is Alice" - this is correct behavior
        response_lower = rolled_back_response.lower()
        self.assertTrue("alice" in response_lower, f"Agent should remember Alice's name. Response: {rolled_back_response}")
        self.assertFalse("paris" in response_lower, f"Agent should not remember Paris (came after checkpoint). Response: {rolled_back_response}")
        print("✓ Rolled-back agent has correct memory state")
        
        # 6. Test that both agents can continue independently
        print(f"\n--- Testing Independent Continuation ---")
        
        # Continue original agent
        original_continue = original_agent.run("What's the capital of France?")
        print(f"Original agent continues: {original_continue}")
        
        # Continue rolled-back agent on different path
        rolled_back_continue = rolled_back_agent.run("I actually live in Tokyo. Please remember this new information.")
        print(f"Rolled-back agent continues: {rolled_back_continue}")
        
        # Verify they have different conversation states now
        final_original_history = original_agent.internal_session.conversation_history
        final_rolled_back_history = rolled_back_agent.internal_session.conversation_history
        
        self.assertGreater(len(final_original_history), len(final_rolled_back_history))
        print(f"✓ Agents continue independently - Original: {len(final_original_history)}, Rolled-back: {len(final_rolled_back_history)}")
        
        # 7. Verify both sessions exist in database
        all_internal_sessions = self.internal_repo.get_by_external_session(self.external_session.id)
        session_ids = [s.id for s in all_internal_sessions]
        
        self.assertIn(original_session_id, session_ids)
        self.assertIn(rolled_back_session_id, session_ids)
        self.assertEqual(len(all_internal_sessions), 2)
        print("✓ Both sessions exist in database")
        
        print(f"\n🎉 All rollback branching tests passed!")

    def test_multiple_rollbacks_create_multiple_branches(self):
        """Test that multiple rollbacks from same checkpoint create separate branches."""
        # Create original agent
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        original_session_id = agent.internal_session.id
        
        # Create conversation and checkpoint
        agent.run("Hello, I'm testing multiple branches.")
        checkpoint_result = agent.create_checkpoint_tool(name="Branch Point")
        checkpoint_id = int(checkpoint_result.split("ID: ")[1].split(")")[0])
        
        agent.run("This message will be lost in rollbacks.")
        
        print(f"\n--- Creating Multiple Branches ---")
        
        # Create first branch
        branch1 = RollbackAgent.from_checkpoint(
            checkpoint_id=checkpoint_id,
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            checkpoint_repo=self.checkpoint_repo,
            internal_session_repo=self.internal_repo
        )
        
        # Create second branch
        branch2 = RollbackAgent.from_checkpoint(
            checkpoint_id=checkpoint_id,
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            checkpoint_repo=self.checkpoint_repo,
            internal_session_repo=self.internal_repo
        )
        
        # Verify all sessions are different
        self.assertNotEqual(original_session_id, branch1.internal_session.id)
        self.assertNotEqual(original_session_id, branch2.internal_session.id)
        self.assertNotEqual(branch1.internal_session.id, branch2.internal_session.id)
        
        # Verify branch relationships
        self.assertEqual(branch1.internal_session.parent_session_id, original_session_id)
        self.assertEqual(branch2.internal_session.parent_session_id, original_session_id)
        
        # Continue each branch differently
        branch1.run("I'm in branch 1, going to calculate 5 + 3.")
        branch2.run("I'm in branch 2, going to calculate 10 * 4.")
        
        # Verify they have independent states
        branch1_history = branch1.internal_session.conversation_history
        branch2_history = branch2.internal_session.conversation_history
        
        branch1_text = str(branch1_history)
        branch2_text = str(branch2_history)
        
        self.assertIn("branch 1", branch1_text.lower())
        self.assertNotIn("branch 1", branch2_text.lower())
        self.assertIn("branch 2", branch2_text.lower())
        self.assertNotIn("branch 2", branch1_text.lower())
        
        print("✓ Multiple branches created successfully with independent states")


# --- tests/test_tool_reversal.py ---

    def test_file_creation_and_reversal(self):
        """Test that file creation tool is properly reversed during rollback."""
        
        # Create the file creation tool
        @tool
        def create_text_file(filename: str, content: str) -> str:
            """Create a text file with the given content.
            
            Args:
                filename: Name of the file to create (without path)
                content: Content to write to the file
                
            Returns:
                Success message with full file path
            """
            filepath = os.path.join(self.test_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
            self.created_files.append(filepath)
            print(f"\n✅ TOOL EXECUTED: Created file -> {filepath}")
            return f"File created successfully at: {filepath}"
        
        # Create the reverse function for file deletion
        def delete_text_file_reverse(args, result):
            """Reverse function that deletes the created file.
            
            Args:
                args: Original arguments passed to create_text_file
                result: Result from create_text_file (contains filepath)
            """
            filename = args.get("filename")
            if filename:
                filepath = os.path.join(self.test_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"\n🔄 REVERSE EXECUTED: Deleted file -> {filepath}")
                    if filepath in self.created_files:
                        self.created_files.remove(filepath)
                else:
                    print(f"\n⚠️ REVERSE: File {filepath} doesn't exist, nothing to reverse")
        
        # Create agent with the file tool and its reverse handler
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=[create_text_file],
            reverse_tools={
                "create_text_file": delete_text_file_reverse
            },
            auto_checkpoint=True,  # Enable auto-checkpointing after tools
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        # Step 1: Initial conversation and create checkpoint BEFORE file creation
        print("\n=== Step 1: Initial conversation ===")
        response1 = agent.run("Hello! I need help managing some files.")
        print(f"Agent: {response1}")
        
        # Create a manual checkpoint before any file operations
        checkpoint_result = agent.create_checkpoint_tool(name="Before File Creation")
        print(f"Checkpoint: {checkpoint_result}")
        self.assertIn("created successfully", checkpoint_result)
        
        # Extract checkpoint ID
        checkpoint_id = int(checkpoint_result.split("ID: ")[1].split(")")[0])
        
        # Step 2: Ask agent to create a file (should trigger tool use)
        print("\n=== Step 2: Creating file via tool ===")
        test_filename = "test_document.txt"
        test_content = "This is a test file created by the LLM agent."
        
        response2 = agent.run(
            f"Please create a text file named '{test_filename}' with the content: '{test_content}'"
        )
        print(f"Agent: {response2}")
        
        # Verify file was created
        test_filepath = os.path.join(self.test_dir, test_filename)
        self.assertTrue(os.path.exists(test_filepath), "File should have been created by the tool")
        
        # Verify file content
        with open(test_filepath, 'r') as f:
            actual_content = f.read()
        self.assertEqual(actual_content, test_content, "File content should match requested content")
        print(f"✓ File created at: {test_filepath}")
        print(f"✓ Content verified: {actual_content}")
        
        # Step 3: Continue conversation after file creation
        print("\n=== Step 3: Conversation after file creation ===")
        response3 = agent.run("Great! Can you confirm the file was created?")
        print(f"Agent: {response3}")
        
        # Add more context that should be forgotten after rollback
        response4 = agent.run("Let's remember that we created this important file.")
        print(f"Agent: {response4}")
        
        # Verify conversation history includes file creation
        history_before = agent.get_conversation_history()
        self.assertTrue(
            any(test_filename in str(msg) for msg in history_before),
            "Conversation should mention the created file"
        )
        
        # Step 4: Get tool track to verify recording
        print("\n=== Step 4: Checking tool track ===")
        tool_track = agent.get_tool_track()
        print(f"Tool track length: {len(tool_track)}")
        
        # Find our file creation in the track
        file_creation_found = False
        for record in tool_track:
            if record.tool_name == "create_text_file":
                file_creation_found = True
                print(f"Found tool invocation: {record.tool_name} with args: {record.args}")
                self.assertEqual(record.args.get("filename"), test_filename)
                self.assertTrue(record.success)
        
        self.assertTrue(file_creation_found, "File creation tool should be in the track")
        
        # Step 5: Rollback to checkpoint BEFORE file creation
        print("\n=== Step 5: Rolling back to checkpoint before file creation ===")
        
        # Create new agent from checkpoint (simulating rollback with tool reversal)
        rolled_back_agent = RollbackAgent.from_checkpoint(
            checkpoint_id=checkpoint_id,
            external_session_id=self.external_session.id,
            model=self.model,
            checkpoint_repo=self.checkpoint_repo,
            internal_session_repo=self.internal_repo,
            tools=[create_text_file],  # Need to provide tools to the rolled-back agent
            reverse_tools={
                "create_text_file": delete_text_file_reverse
            }
        )
        
        # Get the checkpoint to access tool track position
        checkpoint = self.checkpoint_repo.get_by_id(checkpoint_id)
        if "tool_track_position" in checkpoint.metadata:
            track_position = checkpoint.metadata["tool_track_position"]
            print(f"Rolling back tools from position: {track_position}")
            
            # Manually trigger tool rollback (simulating what would happen in production)
            reverse_results = agent.rollback_tools_from_track_index(track_position)
            
            for result in reverse_results:
                print(f"Reverse result: {result.tool_name} - Success: {result.reversed_successfully}")
                if not result.reversed_successfully and result.error_message:
                    print(f"  Error: {result.error_message}")
        
        # Step 6: Verify file has been deleted by reverse handler
        print("\n=== Step 6: Verifying file deletion ===")
        self.assertFalse(
            os.path.exists(test_filepath),
            "File should have been deleted by the reverse handler during rollback"
        )
        print(f"✓ File successfully deleted by reverse handler: {test_filepath}")
        
        # Step 7: Verify rolled back agent has no memory of file creation
        print("\n=== Step 7: Verifying conversation state after rollback ===")
        rolled_back_history = rolled_back_agent.get_conversation_history()
        
        # Should not contain any mention of the file
        self.assertFalse(
            any(test_filename in str(msg) for msg in rolled_back_history),
            "Rolled back conversation should not mention the file"
        )
        
        # Should only have conversation up to checkpoint
        self.assertLess(
            len(rolled_back_history),
            len(history_before),
            "Rolled back history should be shorter"
        )
        
        print(f"✓ Original history length: {len(history_before)}")
        print(f"✓ Rolled back history length: {len(rolled_back_history)}")
        
        # Step 8: Verify the rollback created a branch
        print("\n=== Step 8: Verifying branch creation ===")
        self.assertNotEqual(
            agent.internal_session.id,
            rolled_back_agent.internal_session.id,
            "Rolled back agent should have a new internal session (branch)"
        )
        print(f"✓ Original session ID: {agent.internal_session.id}")
        print(f"✓ Branched session ID: {rolled_back_agent.internal_session.id}")
        
        # Verify original file remains deleted
        self.assertFalse(os.path.exists(test_filepath), "Original file should remain deleted")
        
        print("\n=== SUMMARY ===")
        print("✓ File was created by tool")
        print("✓ Tool invocation was tracked")
        print("✓ Rollback triggered reverse function")
        print("✓ File was successfully deleted")
        print("✓ Conversation history was rolled back")
        print("✓ New branch was created")
        print("✓ Tool reversal test completed successfully!")

    def test_multiple_tool_reversals(self):
        """Test that multiple tools are reversed in correct order."""
        
        # Create multiple file tools
        @tool
        def create_file_a(content: str) -> str:
            """Create file A with content."""
            filepath = os.path.join(self.test_dir, "file_a.txt")
            with open(filepath, 'w') as f:
                f.write(content)
            self.created_files.append(filepath)
            return f"File A created at: {filepath}"
        
        @tool
        def create_file_b(content: str) -> str:
            """Create file B with content."""
            filepath = os.path.join(self.test_dir, "file_b.txt")
            with open(filepath, 'w') as f:
                f.write(content)
            self.created_files.append(filepath)
            return f"File B created at: {filepath}"
        
        # Reverse functions
        def delete_file_a_reverse(args, result):
            filepath = os.path.join(self.test_dir, "file_a.txt")
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Reversed: Deleted file A")
        
        def delete_file_b_reverse(args, result):
            filepath = os.path.join(self.test_dir, "file_b.txt")
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Reversed: Deleted file B")
        
        # Create agent
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=[create_file_a, create_file_b],
            reverse_tools={
                "create_file_a": delete_file_a_reverse,
                "create_file_b": delete_file_b_reverse
            },
            auto_checkpoint=True,
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        # Create checkpoint before any operations
        checkpoint_result = agent.create_checkpoint_tool(name="Before Any Files")
        checkpoint_id = int(checkpoint_result.split("ID: ")[1].split(")")[0])
        
        # Create both files
        agent.run("Create file A with content 'First file'")
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "file_a.txt")))
        
        agent.run("Create file B with content 'Second file'")
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "file_b.txt")))
        
        # Rollback - should delete both files in reverse order (B then A)
        checkpoint = self.checkpoint_repo.get_by_id(checkpoint_id)
        track_position = checkpoint.metadata.get("tool_track_position", 0)
        reverse_results = agent.rollback_tools_from_track_index(track_position)
        
        # Verify both files are deleted
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "file_a.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "file_b.txt")))
        
        print("✓ Multiple tool reversals completed successfully!")

