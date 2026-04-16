# pprados/langchain-rag
# 4 LLM-backed test functions across 7 test files
# Source: https://github.com/pprados/langchain-rag

# --- tests/unit_tests/document_transformers/test_transformers_tools.py ---

def test_generate_questions_transform_documents() -> None:
    doc1 = Document(page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, and 
    quantities and their changes. These topics are represented in modern mathematics 
    with the major subdisciplines of number theory, algebra, geometry, and analysis, 
    respectively. 
    """)
    doc2 = Document(page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """)
    llm = init_llm(
        {
            0: "\n\n"
            "1. What are the major subdisciplines of mathematics?\n"
            "2. What topics are represented in modern mathematics?\n"
            "3. What are the topics of mathematics that include numbers, "
            "formulas and related structures, shapes and the spaces in "
            "which they are contained, and quantities and their changes?\n",
            1: "\n\n"
            "1. What are the origins of mathematics?\n"
            "2. What are some of the mathematical methods and notation used "
            "in the past?\n"
            "3. How has the history of mathematics evolved over time?\n",
        }
    )
    transformer = GenerateQuestionsTransformer.from_llm(llm=llm)
    result = transformer.transform_documents([doc1, doc2])
    assert len(result) == 6

def test_generate_questions_lazy_transform_documents() -> None:
    doc1 = Document(page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, 
    formulas and related structures, shapes and the spaces in which they are 
    contained, and quantities and their changes. These topics are represented 
    in modern mathematics with the major subdisciplines of number theory, algebra, 
    geometry, and analysis, respectively. 
    """)
    doc2 = Document(page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """)
    llm = init_llm(
        {
            0: "\n\n"
            "1. What are the major subdisciplines of mathematics?\n"
            "2. What topics are represented in modern mathematics?\n"
            "3. What are the topics of mathematics that include numbers, "
            "formulas and related structures, shapes and the spaces in "
            "which they are contained, and quantities and their changes?\n",
            1: "\n\n"
            "1. What are the origins of mathematics?\n"
            "2. What are some of the mathematical methods and notation used in the "
            "past?\n"
            "3. How has the history of mathematics evolved over time?\n",
        }
    )
    transformer = GenerateQuestionsTransformer.from_llm(llm=llm)
    result = list(transformer.lazy_transform_documents(iter([doc1, doc2])))
    assert len(result) == 6

async def test_generate_questions_atransform_documents() -> None:
    doc1 = Document(page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, and 
    quantities and their changes. These topics are represented in modern mathematics 
    with the major subdisciplines of number theory, algebra, geometry, and analysis, 
    respectively. 
    """)
    doc2 = Document(page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """)
    llm = init_llm(
        {
            0: "\n\n"
            "1. What are the major subdisciplines of mathematics?\n"
            "2. What topics are represented in modern mathematics?\n"
            "3. What are the topics of mathematics that include numbers, "
            "formulas and related structures, shapes and the spaces in "
            "which they are contained, and quantities and their changes?\n",
            1: "\n\n"
            "1. What are the origins of mathematics?\n"
            "2. What are some of the mathematical methods and notation "
            "used in the past?\n"
            "3. How has the history of mathematics evolved over time?\n",
        }
    )
    transformer = GenerateQuestionsTransformer.from_llm(llm=llm)
    result = await transformer.atransform_documents([doc1, doc2])
    assert len(result) == 6

async def test_generate_questions_alazy_transform_documents() -> None:
    doc1 = Document(page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, and 
    quantities and their changes. These topics are represented in modern mathematics 
    with the major subdisciplines of number theory, algebra, geometry, and analysis, 
    respectively. 
    """)
    doc2 = Document(page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """)
    llm = init_llm(
        {
            0: "\n\n"
            "1. What are the major subdisciplines of mathematics?\n"
            "2. What topics are represented in modern mathematics?\n"
            "3. What are the topics of mathematics that include numbers, "
            "formulas and related structures, shapes and the spaces in "
            "which they are contained, and quantities and their changes?\n",
            1: "\n\n"
            "1. What are the origins of mathematics?\n"
            "2. What are some of the mathematical methods and notation used in the "
            "past?\n"
            "3. How has the history of mathematics evolved over time?\n",
        }
    )
    transformer = GenerateQuestionsTransformer.from_llm(llm=llm)
    result = [
        doc async for doc in transformer.alazy_transform_documents(iter([doc1, doc2]))
    ]
    assert len(result) == 6

