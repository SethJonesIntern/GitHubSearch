# googleapis/langchain-google-spanner-python
# 7 LLM-backed test functions across 7 test files
# Source: https://github.com/googleapis/langchain-google-spanner-python

# --- tests/integration/test_spanner_graph_qa.py ---

    def test_spanner_graph_qa_chain_1(self, chain):
        question = "Where does Elias Thorne's sibling live?"
        response = chain.invoke("query=" + question)
        print(response)

        answer = response["result"]
        assert (
            get_evaluator().evaluate_strings(
                prediction=answer,
                reference="Elias Thorne's sibling lives in Capital City.\n",
            )["score"]
            < 0.1
        )

    def test_spanner_graph_qa_chain_no_answer(self, chain):
        question = "Where does Sarah's sibling live?"
        response = chain.invoke("query=" + question)
        print(response)

        answer = response["result"]
        assert (
            get_evaluator().evaluate_strings(
                prediction=answer,
                reference="I don't know the answer.\n",
            )["score"]
            < 0.1
        )


# --- tests/integration/test_spanner_graph_retriever.py ---

    def test_spanner_graph_gql_retriever(self, setup_db_load_data):
        graph, suffix = setup_db_load_data
        retriever = SpannerGraphTextToGQLRetriever.from_params(
            graph_store=graph,
            llm=get_llm(),
        )
        response = retriever.invoke("Where does Elias Thorne's sibling live?")

        assert len(response) == 1
        assert "Capital City" in response[0].page_content

    def test_spanner_graph_semantic_gql_retriever(self, setup_db_load_data):
        graph, suffix = setup_db_load_data
        suffix = "_" + suffix
        retriever = SpannerGraphTextToGQLRetriever.from_params(
            graph_store=graph,
            llm=get_llm(),
            embedding_service=get_embedding(),
        )
        retriever.add_example(
            "Where does Sam Smith live?",
            """
        GRAPH QAGraph
        MATCH (n:Person{suffix} {{name: "Sam Smith"}})-[:LivesIn]->(l:Location{suffix})
        RETURN l.id AS location_id
    """.format(
                suffix=suffix
            ),
        )
        retriever.add_example(
            "Where does Sam Smith's sibling live?",
            """
        GRAPH QAGraph
        MATCH (n:Person{suffix} {{name: "Sam Smith"}})-[:Sibling]->(m:Person{suffix})-[:LivesIn]->(l:Location{suffix})
        RETURN l.id AS location_id
    """.format(
                suffix=suffix
            ),
        )
        response = retriever.invoke("Where does Elias Thorne's sibling live?")
        assert response == [
            Document(metadata={}, page_content='{"location_id": "Capital City"}')
        ]

    def test_spanner_graph_vector_node_retriever(self, setup_db_load_data):
        graph, suffix = setup_db_load_data
        suffix = "_" + suffix
        retriever = SpannerGraphVectorContextRetriever.from_params(
            graph_store=graph,
            embedding_service=get_embedding(),
            label_expr="Person{}".format(suffix),
            return_properties_list=["name"],
            embeddings_column="desc_embedding",
            top_k=1,
            k=1,
        )
        response = retriever.invoke("Who lives in desert?")
        assert len(response) == 1
        assert "Elias Thorne" in response[0].page_content

    def test_spanner_graph_vector_node_retriever_2(self, setup_db_load_data):
        graph, suffix = setup_db_load_data
        suffix = "_" + suffix
        retriever = SpannerGraphVectorContextRetriever.from_params(
            graph_store=graph,
            embedding_service=get_embedding(),
            label_expr="Person{}".format(suffix),
            expand_by_hops=1,
            embeddings_column="desc_embedding",
            top_k=1,
            k=10,
        )
        response = retriever.invoke(
            "What do you know about the person who lives in desert?"
        )
        assert len(response) == 4
        assert "Elias Thorne" in response[0].page_content

    def test_spanner_graph_vector_node_retriever_0_hops(self, setup_db_load_data):
        graph, suffix = setup_db_load_data
        suffix = "_" + suffix
        retriever = SpannerGraphVectorContextRetriever.from_params(
            graph_store=graph,
            embedding_service=get_embedding(),
            label_expr="Person{}".format(suffix),
            expand_by_hops=0,
            embeddings_column="desc_embedding",
            top_k=1,
            k=10,
        )
        response = retriever.invoke(
            "What do you know about the person who lives in desert?"
        )
        assert len(response) == 1
        assert "Elias Thorne" in response[0].page_content

