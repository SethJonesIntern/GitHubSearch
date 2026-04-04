# griptape-ai/griptape
# 4 test functions with real LLM calls
# Source: https://github.com/griptape-ai/griptape


# --- tests/unit/drivers/memory/conversation/test_dynamodb_conversation_memory_driver.py ---

    def test_store(self):
        session = boto3.Session(region_name=self.AWS_REGION)
        dynamodb = session.resource("dynamodb")
        table = dynamodb.Table(self.DYNAMODB_TABLE_NAME)
        memory_driver = AmazonDynamoDbConversationMemoryDriver(
            session=session,
            table_name=self.DYNAMODB_TABLE_NAME,
            partition_key=self.DYNAMODB_PARTITION_KEY,
            value_attribute_key=self.VALUE_ATTRIBUTE_KEY,
            partition_key_value=self.PARTITION_KEY_VALUE,
        )
        memory = ConversationMemory(conversation_memory_driver=memory_driver)
        pipeline = Pipeline(conversation_memory=memory)

        pipeline.add_task(PromptTask("test"))

        response = table.get_item(TableName=self.DYNAMODB_TABLE_NAME, Key={"entryId": "bar"})
        assert "Item" not in response

        pipeline.run()

        response = table.get_item(TableName=self.DYNAMODB_TABLE_NAME, Key={"entryId": "bar"})
        assert "Item" in response

    def test_store_with_sort_key(self):
        session = boto3.Session(region_name=self.AWS_REGION)
        dynamodb = session.resource("dynamodb")
        table = dynamodb.Table(self.DYNAMODB_COMPOSITE_TABLE_NAME)
        memory_driver = AmazonDynamoDbConversationMemoryDriver(
            session=session,
            table_name=self.DYNAMODB_COMPOSITE_TABLE_NAME,
            partition_key=self.DYNAMODB_PARTITION_KEY,
            value_attribute_key=self.VALUE_ATTRIBUTE_KEY,
            partition_key_value=self.PARTITION_KEY_VALUE,
            sort_key=self.DYNAMODB_SORT_KEY,
            sort_key_value=self.SORT_KEY_VALUE,
        )
        memory = ConversationMemory(conversation_memory_driver=memory_driver)
        pipeline = Pipeline(conversation_memory=memory)

        pipeline.add_task(PromptTask("test"))

        response = table.get_item(
            TableName=self.DYNAMODB_COMPOSITE_TABLE_NAME, Key={"entryId": "bar", "sortKey": "baz"}
        )
        assert "Item" not in response

        pipeline.run()

        response = table.get_item(
            TableName=self.DYNAMODB_COMPOSITE_TABLE_NAME, Key={"entryId": "bar", "sortKey": "baz"}
        )
        assert "Item" in response

    def test_load(self):
        memory_driver = AmazonDynamoDbConversationMemoryDriver(
            session=boto3.Session(region_name=self.AWS_REGION),
            table_name=self.DYNAMODB_TABLE_NAME,
            partition_key=self.DYNAMODB_PARTITION_KEY,
            value_attribute_key=self.VALUE_ATTRIBUTE_KEY,
            partition_key_value=self.PARTITION_KEY_VALUE,
        )
        memory = ConversationMemory(conversation_memory_driver=memory_driver, meta={"foo": "bar"})
        pipeline = Pipeline(conversation_memory=memory)

        pipeline.add_task(PromptTask("test"))

        pipeline.run()
        pipeline.run()

        runs, metadata = memory_driver.load()

        assert len(runs) == 2
        assert metadata == {"foo": "bar"}

    def test_load_with_sort_key(self):
        memory_driver = AmazonDynamoDbConversationMemoryDriver(
            session=boto3.Session(region_name=self.AWS_REGION),
            table_name=self.DYNAMODB_COMPOSITE_TABLE_NAME,
            partition_key=self.DYNAMODB_PARTITION_KEY,
            value_attribute_key=self.VALUE_ATTRIBUTE_KEY,
            partition_key_value=self.PARTITION_KEY_VALUE,
            sort_key=self.DYNAMODB_SORT_KEY,
            sort_key_value=self.SORT_KEY_VALUE,
        )
        memory = ConversationMemory(conversation_memory_driver=memory_driver, meta={"foo": "bar"})
        pipeline = Pipeline(conversation_memory=memory)

        pipeline.add_task(PromptTask("test"))

        pipeline.run()
        pipeline.run()

        runs, metadata = memory_driver.load()

        assert len(runs) == 2
        assert metadata == {"foo": "bar"}

