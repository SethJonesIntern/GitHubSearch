# danny-avila/rag_api
# 3 LLM-backed test functions across 14 test files
# Source: https://github.com/danny-avila/rag_api

# --- tests/utils/test_lazy_load.py ---

def test_unstructured_lazy_load_no_memory_benefit(tmp_path, filename, content_type):
    """Unstructured-based loaders internally load the full file regardless of
    lazy_load() vs load(). Verify lazy_load() doc count matches load()."""
    file_path = tmp_path / filename
    ext = os.path.splitext(filename)[1]
    _UNSTRUCTURED_CREATORS[ext](str(file_path))

    def factory():
        loader, _, _ = get_loader(filename, content_type, str(file_path))
        return loader

    load_docs, _ = _measure_load(factory)
    texts, _ = _measure_lazy_load_streaming(factory)

    assert len(load_docs) == len(texts)

    def test_pdf_streaming_lazy_load_peak_memory(self, pdf_path):
        """Streaming lazy_load() should use <= peak memory vs load()."""

        def factory():
            return SafePyPDFLoader(pdf_path, extract_images=False)

        load_docs, peak_load = _measure_load(factory)
        texts, peak_lazy_stream = _measure_lazy_load_streaming(factory)

        assert len(load_docs) == self.NUM_PAGES
        assert len(texts) == self.NUM_PAGES

        # Streaming should not use MORE memory than eager (allow 10% noise)
        assert peak_lazy_stream <= peak_load * 1.10, (
            f"streaming peak ({peak_lazy_stream:,} B) exceeded "
            f"load() peak ({peak_load:,} B) by >10%"
        )

    def test_csv_streaming_lazy_load_peak_memory(self, csv_path):
        """Streaming lazy_load() should use significantly less peak memory
        than load() for CSVs with many rows."""

        def factory():
            from langchain_community.document_loaders import CSVLoader

            return CSVLoader(csv_path)

        load_docs, peak_load = _measure_load(factory)
        texts, peak_lazy_stream = _measure_lazy_load_streaming(factory)

        assert len(load_docs) == self.NUM_ROWS
        assert len(texts) == self.NUM_ROWS

        # CSV streaming should use meaningfully less memory
        assert peak_lazy_stream <= peak_load * 1.10, (
            f"streaming peak ({peak_lazy_stream:,} B) exceeded "
            f"load() peak ({peak_load:,} B) by >10%"
        )

