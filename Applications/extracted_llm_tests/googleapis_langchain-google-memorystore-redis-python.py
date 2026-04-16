# googleapis/langchain-google-memorystore-redis-python
# 1 LLM-backed test functions across 4 test files
# Source: https://github.com/googleapis/langchain-google-memorystore-redis-python

# --- tests/test_memorystore_redis_vectorstore.py ---

def test_vector_store_init_hnsw_index(client):
    index_name = str(uuid.uuid4())

    index_config = HNSWConfig(
        name=index_name,
        distance_strategy=DistanceStrategy.COSINE,
        vector_size=128,
        m=1,
        ef_construction=2,
        ef_runtime=3,
    )

    assert not check_index_exists(client, index_name, index_config)
    RedisVectorStore.init_index(client=client, index_config=index_config)
    assert check_index_exists(client, index_name, index_config)
    RedisVectorStore.drop_index(client=client, index_name=index_name)
    assert not check_index_exists(client, index_name, index_config)
    client.flushall()

