# dynamiq-ai/dynamiq
# 40 test functions with real LLM calls
# Source: https://github.com/dynamiq-ai/dynamiq


# --- tests/integration/nodes/embedders/test_bedrock_embedders.py ---

def test_text_embedder_missing_input(bedrock_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = bedrock_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_text_embedder_empty_input(bedrock_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = bedrock_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_missing_input(bedrock_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = bedrock_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_empty_document_list(bedrock_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = bedrock_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value

def test_document_embedder_empty_content(bedrock_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = bedrock_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


# --- tests/integration/nodes/embedders/test_cohere_embedders.py ---

def test_text_embedder_missing_input(cohere_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = cohere_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_text_embedder_empty_input(cohere_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = cohere_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_missing_input(cohere_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = cohere_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_empty_document_list(cohere_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = cohere_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value

def test_document_embedder_empty_content(cohere_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = cohere_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


# --- tests/integration/nodes/embedders/test_gemini_embedders.py ---

def test_text_embedder_missing_input(gemini_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = gemini_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_text_embedder_empty_input(gemini_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = gemini_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_missing_input(gemini_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = gemini_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_empty_document_list(gemini_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = gemini_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value

def test_document_embedder_empty_content(gemini_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = gemini_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


# --- tests/integration/nodes/embedders/test_huggingface_embedders.py ---

def test_text_embedder_missing_input(huggingface_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = huggingface_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_text_embedder_empty_input(huggingface_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = huggingface_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_missing_input(huggingface_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = huggingface_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_empty_document_list(huggingface_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = huggingface_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value

def test_document_embedder_empty_content(huggingface_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = huggingface_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


# --- tests/integration/nodes/embedders/test_mistral_embedders.py ---

def test_text_embedder_missing_input(mistral_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = mistral_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_text_embedder_empty_input(mistral_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = mistral_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_missing_input(mistral_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = mistral_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_empty_document_list(mistral_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = mistral_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value

def test_document_embedder_empty_content(mistral_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = mistral_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


# --- tests/integration/nodes/embedders/test_openai_embedders.py ---

def test_text_embedder_missing_input(openai_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = openai_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_text_embedder_empty_input(openai_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = openai_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_missing_input(openai_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = openai_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_empty_document_list(openai_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = openai_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value

def test_document_embedder_empty_content(openai_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = openai_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


# --- tests/integration/nodes/embedders/test_vertexai_embedders.py ---

def test_text_embedder_missing_input(vertexai_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = vertexai_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_text_embedder_empty_input(vertexai_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = vertexai_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_missing_input(vertexai_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = vertexai_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_empty_document_list(vertexai_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = vertexai_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value

def test_document_embedder_empty_content(vertexai_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = vertexai_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


# --- tests/integration/nodes/embedders/test_watsonx_embedders.py ---

def test_text_embedder_missing_input(watsonx_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = watsonx_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_text_embedder_empty_input(watsonx_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = watsonx_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_missing_input(watsonx_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = watsonx_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)

def test_document_embedder_empty_document_list(watsonx_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = watsonx_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value

def test_document_embedder_empty_content(watsonx_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = watsonx_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)

