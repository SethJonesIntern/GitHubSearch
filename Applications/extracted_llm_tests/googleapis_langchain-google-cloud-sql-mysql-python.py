# googleapis/langchain-google-cloud-sql-mysql-python
# 10 LLM-backed test functions across 9 test files
# Source: https://github.com/googleapis/langchain-google-cloud-sql-mysql-python

# --- tests/integration/test_mysql_vectorstore.py ---

    def test_add_texts(self, engine, vs):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs.add_texts(texts, ids=ids)
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 3

        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs.add_texts(texts, metadatas, ids)
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 6
        engine._execute(f"TRUNCATE TABLE `{DEFAULT_TABLE}`")

    def test_add_texts_edge_cases(self, engine, vs):
        texts = ["Taylor's", '"Swift"', "best-friend"]
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs.add_texts(texts, ids=ids)
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 3
        engine._execute(f"TRUNCATE TABLE `{DEFAULT_TABLE}`")

    def test_add_docs(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs.add_documents(docs, ids=ids)
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 3
        engine._execute(f"TRUNCATE TABLE `{DEFAULT_TABLE}`")

    def test_add_embedding(self, engine, vs):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs._add_embeddings(texts, embeddings, metadatas, ids)
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 3
        engine._execute(f"TRUNCATE TABLE `{DEFAULT_TABLE}`")

    def test_add_texts_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs_custom.add_texts(texts, ids=ids)
        results = engine._fetch(f"SELECT * FROM `{CUSTOM_TABLE}`")
        content = [result["mycontent"] for result in results]
        assert len(results) == 3
        assert "foo" in content
        assert "bar" in content
        assert "baz" in content
        assert results[0]["myembedding"]
        assert results[0]["page"] is None
        assert results[0]["source"] is None

        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs_custom.add_texts(texts, metadatas, ids)
        results = engine._fetch(f"SELECT * FROM `{CUSTOM_TABLE}`")
        assert len(results) == 6
        engine._execute(f"TRUNCATE TABLE `{CUSTOM_TABLE}`")

    def test_add_docs_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "google.com"},
            )
            for i in range(len(texts))
        ]
        vs_custom.add_documents(docs, ids=ids)

        results = engine._fetch(f"SELECT * FROM `{CUSTOM_TABLE}`")
        content = [result["mycontent"] for result in results]
        assert len(results) == 3
        assert "foo" in content
        assert "bar" in content
        assert "baz" in content
        assert results[0]["myembedding"]
        pages = [result["page"] for result in results]
        assert "0" in pages
        assert "1" in pages
        assert "2" in pages
        assert results[0]["source"] == "google.com"
        engine._execute(f"TRUNCATE TABLE `{CUSTOM_TABLE}`")

    def test_add_embedding_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs_custom._add_embeddings(texts, embeddings, metadatas, ids)
        results = engine._fetch(f"SELECT * FROM `{CUSTOM_TABLE}`")
        assert len(results) == 3
        engine._execute(f"TRUNCATE TABLE `{CUSTOM_TABLE}`")


# --- tests/integration/test_mysql_vectorstore_from_methods.py ---

    def test_from_texts(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        MySQLVectorStore.from_texts(
            texts,
            embeddings_service,
            engine,
            DEFAULT_TABLE,
            metadatas=metadatas,
            ids=ids,
        )
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 3
        engine._execute(f"TRUNCATE TABLE `{DEFAULT_TABLE}`")

    def test_from_docs(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        MySQLVectorStore.from_documents(
            docs,
            embeddings_service,
            engine,
            DEFAULT_TABLE,
            ids=ids,
        )
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 3
        engine._execute(f"TRUNCATE TABLE `{DEFAULT_TABLE}`")

    def test_from_docs_custom(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "google.com"},
            )
            for i in range(len(texts))
        ]
        MySQLVectorStore.from_documents(
            docs,
            embeddings_service,
            engine,
            CUSTOM_TABLE,
            ids=ids,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
        )

        results = engine._fetch(f"SELECT * FROM `{CUSTOM_TABLE}`")
        content = [result["mycontent"] for result in results]
        assert len(results) == 3
        assert "foo" in content
        assert "bar" in content
        assert "baz" in content
        assert results[0]["myembedding"]
        pages = [result["page"] for result in results]
        assert "0" in pages
        assert "1" in pages
        assert "2" in pages
        assert results[0]["source"] == "google.com"
        engine._execute(f"TRUNCATE TABLE `{CUSTOM_TABLE}`")

