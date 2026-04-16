# langchain-ai/langchain-pinecone
# 1 LLM-backed test functions across 10 test files
# Source: https://github.com/langchain-ai/langchain-pinecone

# --- libs/pinecone/tests/integration_tests/test_vectorstores.py ---

    def test_from_texts_with_metadatas_benchmark(
        self,
        pool_threads: int,
        batch_size: int,
        embeddings_chunk_size: int,
        data_multiplier: int,
        documents: List[Document],
        embedding_openai: OpenAIEmbeddings,
    ) -> None:
        """Test end to end construction and search."""

        texts = [document.page_content for document in documents] * data_multiplier
        uuids = [uuid.uuid4().hex for _ in range(len(texts))]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embedding_openai,
            ids=uuids,
            metadatas=metadatas,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
            pool_threads=pool_threads,
            batch_size=batch_size,
            embeddings_chunk_size=embeddings_chunk_size,
        )

        query = "What did the president say about Ketanji Brown Jackson"
        _ = docsearch.similarity_search(query, k=1, namespace=NAMESPACE_NAME)

