# enoch3712/ExtractThinker
# 4 LLM-backed test functions across 27 test files
# Source: https://github.com/enoch3712/ExtractThinker

# --- tests/test_extractor.py ---

def test_forbidden_strategy_with_token_limit():
    test_file_path = os.path.join(os.getcwd(), "tests", "test_images", "eu_tax_chart.png")
    tesseract_path = os.getenv("TESSERACT_PATH")

    llm = LLM(get_lite_model(), token_limit=10)

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    extractor.load_llm(llm)

    # Should raise ExtractThinkerError due to FORBIDDEN strategy
    with pytest.raises(ExtractThinkerError, match="Incomplete output received and FORBIDDEN strategy is set"):
        extractor.extract(
            test_file_path,
            ReportContract,
            vision=False,
            content="RULE: Give me all the pages content",
            completion_strategy=CompletionStrategy.FORBIDDEN
        )

def test_pagination_handler():
    test_file_path = os.path.join(os.getcwd(), "tests", "files", "Regional_GDP_per_capita_2018_2.pdf")

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderDocling())
    extractor.load_llm(get_big_model())

    # Create and run both extractions in parallel
    async def run_parallel_extractions():
        result_1, result_2 = await asyncio.gather(
            extract_async(extractor, test_file_path, vision=True, completion_strategy=CompletionStrategy.PAGINATE),
            extract_async(extractor, test_file_path, vision=True, completion_strategy=CompletionStrategy.FORBIDDEN)
        )
        return result_1, result_2

    # Run the async extraction and get the results as instances of OptionalEUData
    results = asyncio.run(run_parallel_extractions())
    result_1, result_2 = results

    # Compare top-level EU data
    assert result_1.eu_total_gdp_million_27 == result_2.eu_total_gdp_million_27
    assert result_1.eu_total_gdp_million_28 == result_2.eu_total_gdp_million_28

    # Compare country count
    assert len(result_1.countries) == len(result_2.countries)

def test_pagination_handler_optional():
    test_file_path = os.path.join(os.getcwd(), "tests", "files", "Regional_GDP_per_capita_2018_2.pdf")

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderDocling())
    extractor.load_llm(get_big_model())

    async def extract_async_optional(extractor, file_path, vision, completion_strategy):
        return extractor.extract(
            file_path,
            EUDataOptional,
            vision=vision,
            completion_strategy=completion_strategy
        )
    
    result = asyncio.run(extract_async_optional(extractor, test_file_path, vision=True, completion_strategy=CompletionStrategy.PAGINATE))

    assert len(result.countries) == 6

def test_concatenation_handler():
    test_file_path = os.path.join(os.getcwd(), "tests", "test_images", "eu_tax_chart.png")
    tesseract_path = os.getenv("TESSERACT_PATH")
    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    llm_first = LLM(get_big_model(), token_limit=4096)
    extractor.load_llm(llm_first)

    result_1: ReportContract = extractor.extract(
        test_file_path,
        ReportContract,
        vision=True,
        completion_strategy=CompletionStrategy.CONCATENATE
    )

    second_extractor = Extractor()
    second_extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    second_extractor.load_llm(get_big_model())

    result_2: ReportContract = second_extractor.extract(
        test_file_path,
        ReportContract,
        vision=True,
        completion_strategy=CompletionStrategy.FORBIDDEN
    )

    assert semantically_similar(
        result_1.title,
        result_2.title,
        threshold=0.8
    ), "Titles are not semantically similar enough (threshold: 90%)"

    assert result_1.pages[0].number == result_2.pages[0].number
    assert semantically_similar(
        result_1.pages[0].content, 
        result_2.pages[0].content,
        threshold=0.8
    ), "Page contents are not semantically similar enough (threshold: 90%)"

