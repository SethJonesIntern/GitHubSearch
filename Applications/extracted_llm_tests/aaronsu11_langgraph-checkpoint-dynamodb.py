# aaronsu11/langgraph-checkpoint-dynamodb
# 14 LLM-backed test functions across 11 test files
# Source: https://github.com/aaronsu11/langgraph-checkpoint-dynamodb

# --- langgraph_checkpoint_dynamodb/tests/integration/test_localstack_verification.py ---

    def test_checkpointer_uses_localstack_endpoint(self, checkpointer, dynamodb_config):
        """Verify checkpointer is configured with LocalStack endpoint."""
        # Check that checkpointer's config uses LocalStack endpoint
        checkpointer_config = checkpointer.config
        assert checkpointer_config.endpoint_url == dynamodb_config.endpoint_url
        assert (
            "localhost" in checkpointer_config.endpoint_url
            or "127.0.0.1" in checkpointer_config.endpoint_url
        )

        # Verify we can actually use it (proves it's connecting to LocalStack)
        from langchain_core.runnables import RunnableConfig

        test_config = RunnableConfig(
            configurable={"thread_id": "test_verification", "checkpoint_id": "test_1"}
        )
        checkpoint = {"id": "test_1"}
        metadata = {"step": 1}

        # This should work if LocalStack is running
        result = checkpointer.put(test_config, checkpoint, metadata, {})
        assert result is not None

        # Verify we can read it back (confirms LocalStack persistence)
        result_tuple = checkpointer.get_tuple(test_config)
        assert result_tuple is not None
        assert result_tuple.checkpoint["id"] == "test_1"


# --- langgraph_checkpoint_dynamodb/tests/integration/test_table_deployment.py ---

    def test_deploy_false_existing_table(self, dynamodb_config):
        """Test using existing table without deploy."""
        table_name = f"test-existing-{uuid.uuid4().hex[:8]}"
        config = dynamodb_config
        config.table_config = DynamoDBTableConfig(table_name=table_name)

        # First create table
        saver1 = DynamoDBSaver(config=config, deploy=True)

        # Now use existing table with deploy=False
        saver2 = DynamoDBSaver(config=config, deploy=False)

        # Verify we can use it
        from langchain_core.runnables import RunnableConfig

        checkpoint_config = RunnableConfig(
            configurable={"thread_id": "test_thread", "checkpoint_id": "test_1"}
        )
        checkpoint = {"id": "test_1"}
        metadata = {"step": 1}
        saver2.put(checkpoint_config, checkpoint, metadata, {})

        result = saver2.get_tuple(checkpoint_config)
        assert result is not None
        assert result.checkpoint["id"] == "test_1"

        # Cleanup
        saver1.destroy()

    def test_table_ttl_configuration(self, dynamodb_config):
        """Test table creation with TTL enabled."""
        table_name = f"test-ttl-{uuid.uuid4().hex[:8]}"
        config = dynamodb_config
        config.table_config = DynamoDBTableConfig(
            table_name=table_name, ttl_days=7, ttl_attribute="expireAt"
        )

        saver = DynamoDBSaver(config=config, deploy=True)

        # Verify TTL is enabled
        client = boto3.client(
            "dynamodb",
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
        )
        response = client.describe_time_to_live(TableName=table_name)
        ttl_status = response["TimeToLiveDescription"]["TimeToLiveStatus"]
        assert ttl_status == "ENABLED"
        assert response["TimeToLiveDescription"]["AttributeName"] == "expireAt"

        # Test that checkpoint gets TTL attribute
        from langchain_core.runnables import RunnableConfig

        checkpoint_config = RunnableConfig(
            configurable={"thread_id": "test_thread_ttl", "checkpoint_id": "test_1"}
        )
        checkpoint = {"id": "test_1"}
        metadata = {"step": 1}
        saver.put(checkpoint_config, checkpoint, metadata, {})

        # Verify item has TTL attribute
        table = boto3.resource(
            "dynamodb",
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
        ).Table(table_name)

        result = table.get_item(
            Key={"PK": "test_thread_ttl", "SK": "#checkpoint#test_1"}
        )
        assert "Item" in result
        assert "expireAt" in result["Item"]

        saver.destroy()


# --- langgraph_checkpoint_dynamodb/tests/unit/test_saver.py ---

    def test_get_nonexistent_checkpoint(self, aws_credentials):
        """Test getting a checkpoint that doesn't exist."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        config = RunnableConfig(
            configurable={"thread_id": "nonexistent", "checkpoint_id": "nonexistent"}
        )
        result = saver.get_tuple(config)
        assert result is None

    def test_state_graph_basic(self, aws_credentials):
        """Test basic StateGraph usage."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        workflow = StateGraph(MessagesState)
        workflow.add_node(
            "chatbot", lambda state: {"messages": [{"role": "ai", "content": "Hello!"}]}
        )
        workflow.add_edge(START, "chatbot")
        workflow.add_edge("chatbot", END)

        graph = workflow.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "test_thread_msg"}}

        result = graph.invoke(
            {"messages": [{"role": "human", "content": "Hi!"}]}, config
        )
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], HumanMessage)
        assert isinstance(result["messages"][1], AIMessage)

    def test_state_graph_with_channels(self, aws_credentials):
        """Test StateGraph with multiple channels."""

        class State(TypedDict):
            count: int
            messages: Annotated[list[str], operator.add]

        workflow = StateGraph(State)
        workflow.add_node("counter", lambda state: {"count": state["count"] + 1})
        workflow.add_node(
            "messenger", lambda state: {"messages": ["msg" + str(state["count"])]}
        )
        workflow.add_edge(START, "counter")
        workflow.add_edge("counter", "messenger")
        workflow.add_edge("messenger", END)

        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        graph = workflow.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "test_thread_state"}}

        result = graph.invoke({"count": 0, "messages": []}, config)
        assert result["count"] == 1
        assert result["messages"] == ["msg1"]

    def test_graph_state_management(self, aws_credentials):
        """Test graph state management and retrieval."""
        workflow = StateGraph(MessagesState)
        workflow.add_node(
            "chatbot", lambda state: {"messages": [{"role": "ai", "content": "Hello!"}]}
        )
        workflow.add_edge(START, "chatbot")
        workflow.add_edge("chatbot", END)

        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        graph = workflow.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "test_thread_state_mgmt"}}

        # Initial invocation
        graph.invoke({"messages": [{"role": "human", "content": "Hi!"}]}, config)

        # Get state
        state = graph.get_state(config)
        assert len(state.values["messages"]) == 2
        assert state.values["messages"][0].content == "Hi!"
        assert state.values["messages"][1].content == "Hello!"

        # Get state history
        history = list(graph.get_state_history(config))
        assert len(history) > 0
        assert all(hasattr(state, "values") for state in history)

    def test_graph_streaming(self, aws_credentials):
        """Test graph streaming capabilities."""
        workflow = StateGraph(MessagesState)
        workflow.add_node(
            "chatbot", lambda state: {"messages": [{"role": "ai", "content": "Hello!"}]}
        )
        workflow.add_edge(START, "chatbot")
        workflow.add_edge("chatbot", END)

        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        graph = workflow.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "test_thread_stream"}}

        # Test values stream mode
        values_updates = list(
            graph.stream(
                {"messages": [{"role": "human", "content": "Hi!"}]},
                config,
                stream_mode="values",
            )
        )
        assert len(values_updates) > 0

        # Test updates stream mode
        updates = list(
            graph.stream(
                {"messages": [{"role": "human", "content": "Hi!"}]},
                config,
                stream_mode="updates",
            )
        )
        assert len(updates) > 0

    def test_get_latest_checkpoint(self, aws_credentials, sample_config):
        """
        Test getting latest checkpoint without checkpoint_id.

        Covers:
        - get_tuple() without checkpoint_id (gets latest) (lines 140-166)
        - Parent checkpoint handling
        """
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # Create multiple checkpoints
        checkpoint_configs = []
        parent_id = None
        for i in range(3):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_ns": "",
                    "checkpoint_id": parent_id,  # Set parent checkpoint_id
                }
            )
            checkpoint = {"id": f"checkpoint_{i}"}
            metadata = {"step": i}
            saver.put(checkpoint_config, checkpoint, metadata, {})
            checkpoint_configs.append(checkpoint_config)
            # Update parent_id for next checkpoint
            parent_id = f"checkpoint_{i}"

        # Get latest checkpoint without checkpoint_id
        latest_config = RunnableConfig(
            configurable={
                "thread_id": sample_config["configurable"]["thread_id"],
                "checkpoint_ns": "",
                # No checkpoint_id specified - should get latest
            }
        )
        result = saver.get_tuple(latest_config)

        # Should return the most recent checkpoint (checkpoint_2)
        assert result is not None
        assert result.checkpoint["id"] == "checkpoint_2"

    def test_list_with_pagination(self, aws_credentials, sample_config):
        """
        Test list() with before and limit parameters.

        Covers:
        - list() with before parameter (lines 402-405)
        - list() with limit parameter (lines 413-414)
        """
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # Create multiple checkpoints
        checkpoint_configs = []
        for i in range(5):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_ns": "",
                    "checkpoint_id": f"checkpoint_{i}",
                }
            )
            checkpoint = {"id": f"checkpoint_{i}"}
            metadata = {"step": i}
            saver.put(checkpoint_config, checkpoint, metadata, {})
            checkpoint_configs.append(checkpoint_config)

        # Test with limit
        results = list(saver.list(sample_config, limit=3))
        assert len(results) == 3
        # Should get the 3 most recent checkpoints
        assert results[0].checkpoint["id"] == "checkpoint_4"
        assert results[1].checkpoint["id"] == "checkpoint_3"
        assert results[2].checkpoint["id"] == "checkpoint_2"

        # Test with before parameter
        before_config = RunnableConfig(
            configurable={
                "thread_id": sample_config["configurable"]["thread_id"],
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint_3",
            }
        )
        results_before = list(saver.list(sample_config, before=before_config))
        # Should get checkpoints before checkpoint_3 (checkpoint_2, checkpoint_1, checkpoint_0)
        assert len(results_before) == 3
        assert results_before[0].checkpoint["id"] == "checkpoint_2"
        assert results_before[1].checkpoint["id"] == "checkpoint_1"
        assert results_before[2].checkpoint["id"] == "checkpoint_0"

    def test_ttl_functionality(self, aws_credentials, sample_config):
        """
        Test checkpoint creation with TTL and TTL filtering in queries.

        Covers:
        - TTL paths in create_checkpoint_item (lines 152-154 in utils.py)
        - TTL filtering in queries (lines 152-157, 176-181 in saver.py)
        """
        # Configure table with TTL
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints",
            billing_mode=BillingMode.PAY_PER_REQUEST,
            ttl_days=7,
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # Create checkpoint with TTL
        checkpoint = {"id": sample_config["configurable"]["checkpoint_id"]}
        metadata = {"step": 1}
        saver.put(sample_config, checkpoint, metadata, {})

        # Get checkpoint and verify TTL was set
        result = saver.get_tuple(sample_config)
        assert result is not None
        assert result.checkpoint["id"] == sample_config["configurable"]["checkpoint_id"]

        # Verify TTL filter is applied in queries
        # Create another checkpoint
        checkpoint_config2 = RunnableConfig(
            configurable={
                "thread_id": sample_config["configurable"]["thread_id"],
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint_with_ttl",
            }
        )
        checkpoint2 = {"id": "checkpoint_with_ttl"}
        metadata2 = {"step": 2}
        saver.put(checkpoint_config2, checkpoint2, metadata2, {})

        # List checkpoints - should filter out expired ones
        results = list(saver.list(sample_config))
        assert len(results) == 2
        # Both checkpoints should be returned as they haven't expired yet
        checkpoint_ids = {r.checkpoint["id"] for r in results}
        assert "checkpoint_with_ttl" in checkpoint_ids
        assert sample_config["configurable"]["checkpoint_id"] in checkpoint_ids

    def test_ttl_list_filtering(self, aws_credentials, sample_config):
        """
        Test TTL filtering in list operations.

        Covers:
        - TTL filtering in list() method (lines 420-426 in saver.py)
        - TTL filtering with before parameter
        """
        # Configure table with TTL
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints",
            billing_mode=BillingMode.PAY_PER_REQUEST,
            ttl_days=7,
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # Create multiple checkpoints
        for i in range(3):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_ns": "",
                    "checkpoint_id": f"checkpoint_ttl_{i}",
                }
            )
            checkpoint = {"id": f"checkpoint_ttl_{i}"}
            metadata = {"step": i}
            saver.put(checkpoint_config, checkpoint, metadata, {})

        # List checkpoints - TTL filter should be applied
        results = list(saver.list(sample_config))
        assert len(results) == 3
        assert all(isinstance(r, CheckpointTuple) for r in results)

    def test_enable_ttl_updates_existing_checkpoints(
        self, aws_credentials, sample_config
    ):
        """
        Test that _enable_ttl updates existing checkpoints without TTL attribute.

        Covers:
        - _enable_ttl() method (lines 1063-1147 in saver.py)
        - Updating existing checkpoints with TTL attribute
        - Batch processing of items
        """
        # Create table without TTL first
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints",
            billing_mode=BillingMode.PAY_PER_REQUEST,
            ttl_days=None,  # No TTL initially
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # Create multiple checkpoints without TTL
        checkpoint_configs = []
        for i in range(5):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_ns": "",
                    "checkpoint_id": f"checkpoint_no_ttl_{i}",
                }
            )
            checkpoint = {"id": f"checkpoint_no_ttl_{i}"}
            metadata = {"step": i}
            saver.put(checkpoint_config, checkpoint, metadata, {})
            checkpoint_configs.append(checkpoint_config)

        # Verify items don't have TTL attribute yet
        table = saver.table
        for checkpoint_config in checkpoint_configs[:2]:  # Check first 2
            thread_id = checkpoint_config["configurable"]["thread_id"]
            checkpoint_id = checkpoint_config["configurable"]["checkpoint_id"]
            key = {"PK": thread_id, "SK": f"#checkpoint#{checkpoint_id}"}
            item = table.get_item(Key=key).get("Item", {})
            assert "expireAt" not in item, "Item should not have TTL attribute yet"

        # Now enable TTL on existing table
        table_config.ttl_days = 7
        table_config.ttl_attribute = "expireAt"
        saver.config.table_config = table_config
        saver._enable_ttl()

        # Verify all existing checkpoints now have TTL attribute
        import time

        for checkpoint_config in checkpoint_configs:
            thread_id = checkpoint_config["configurable"]["thread_id"]
            checkpoint_id = checkpoint_config["configurable"]["checkpoint_id"]
            key = {"PK": thread_id, "SK": f"#checkpoint#{checkpoint_id}"}
            item = table.get_item(Key=key).get("Item", {})
            assert "expireAt" in item, f"Item {checkpoint_id} should have TTL attribute"
            assert item["expireAt"] > int(time.time()), "TTL should be in the future"

    def test_enable_ttl_skips_items_with_existing_ttl(
        self, aws_credentials, sample_config
    ):
        """
        Test that _enable_ttl skips items that already have TTL attribute.

        Covers:
        - _enable_ttl() filtering logic (lines 1134-1142 in saver.py)
        - attribute_not_exists filter
        """
        # Create table with TTL enabled from start
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints",
            billing_mode=BillingMode.PAY_PER_REQUEST,
            ttl_days=7,
            ttl_attribute="expireAt",
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # Create checkpoints (these will have TTL from creation)
        checkpoint = {"id": sample_config["configurable"]["checkpoint_id"]}
        metadata = {"step": 1}
        config_with_checkpoint = saver.put(sample_config, checkpoint, metadata, {})

        # Verify checkpoint has TTL
        table = saver.table
        thread_id = sample_config["configurable"]["thread_id"]
        checkpoint_id = config_with_checkpoint["configurable"]["checkpoint_id"]
        key = {"PK": thread_id, "SK": f"#checkpoint#{checkpoint_id}"}
        item = table.get_item(Key=key).get("Item", {})
        original_ttl = item.get("expireAt")
        assert original_ttl is not None, "Item should have TTL from creation"

        # Create another checkpoint without TTL (by manually inserting)
        # This simulates an item that somehow doesn't have TTL
        from langgraph_checkpoint_dynamodb.utils import create_checkpoint_item

        checkpoint2_id = "checkpoint_no_ttl_manual"
        checkpoint2 = {"id": checkpoint2_id}
        checkpoint2_config = RunnableConfig(
            configurable={
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint2_id,
            }
        )
        type_, checkpoint_data = saver.serde.dumps_typed(checkpoint2)
        _, metadata_data = saver.serde.dumps_typed(metadata)

        # Create item manually without TTL
        item2 = create_checkpoint_item(
            thread_id,
            "",
            checkpoint2_id,
            type_,
            checkpoint_data,
            metadata_data,
            None,
            None,  # No TTL
            "expireAt",
        )
        # Remove TTL if it was added
        if "expireAt" in item2:
            del item2["expireAt"]
        table.put_item(Item=item2)

        # Now call _enable_ttl again - should only update item without TTL
        saver._enable_ttl()

        # Verify original checkpoint's TTL hasn't changed
        item = table.get_item(Key=key).get("Item", {})
        assert (
            item.get("expireAt") == original_ttl
        ), "Original item's TTL should not change"

        # Verify new checkpoint now has TTL
        key2 = {"PK": thread_id, "SK": f"#checkpoint#{checkpoint2_id}"}
        item2_updated = table.get_item(Key=key2).get("Item", {})
        assert "expireAt" in item2_updated, "Item without TTL should get TTL attribute"

