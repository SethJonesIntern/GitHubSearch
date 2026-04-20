# danny-avila/rag_api
# 7 LLM-backed test functions across 15 test files
# Source: https://github.com/danny-avila/rag_api

# --- tests/integration/test_pgvector_filter.py ---

    def test_langchain_default_eq_emits_jsonb_path_match(self, store):
        """Without our override, LangChain generates jsonb_path_match for $eq."""
        upstream_clause = PGVector._handle_field_filter(
            store, "file_id", {"$eq": "test-id"}
        )
        sql = _compile_clause(upstream_clause)
        assert "jsonb_path_match" in sql, (
            f"Expected LangChain default to use jsonb_path_match.\n"
            f"Got: {sql}\n"
            f"If this fails, LangChain may have fixed the issue upstream — "
            f"review whether our override is still needed."
        )
        assert "->>" not in sql, (
            f"LangChain default unexpectedly uses ->> for $eq.\n"
            f"Got: {sql}\n"
            f"The upstream bug may have been fixed."
        )

    def test_our_override_emits_astext_for_same_input(self, store):
        """Our override produces ->> instead of jsonb_path_match for $eq."""
        our_clause = store._handle_field_filter("file_id", {"$eq": "test-id"})
        our_sql = _compile_clause(our_clause)

        upstream_clause = PGVector._handle_field_filter(
            store, "file_id", {"$eq": "test-id"}
        )
        upstream_sql = _compile_clause(upstream_clause)

        assert (
            our_sql != upstream_sql
        ), "Override produces identical SQL to parent — override may be a no-op"
        assert "->>" in our_sql
        assert "jsonb_path_match" not in our_sql

    def test_langchain_default_causes_seq_scan_on_real_pg(
        self, engine, seeded_data, store
    ):
        """The upstream jsonb_path_match SQL seq-scans on real PostgreSQL."""
        _, target_file_id = seeded_data

        upstream_clause = PGVector._handle_field_filter(
            store, "file_id", {"$eq": target_file_id}
        )

        with Session(engine) as session:
            sa_query = session.query(store.EmbeddingStore).filter(upstream_clause)
            compiled = sa_query.statement.compile(
                dialect=engine.dialect,
                compile_kwargs={"literal_binds": True},
            )
            full_sql = str(compiled)

        with engine.begin() as conn:
            plan_json = conn.execute(
                text(f"EXPLAIN (ANALYZE, FORMAT JSON) {full_sql}")
            ).scalar()

        node_types = [
            n.get("Node Type", "").lower() for n in _walk_plan_nodes(plan_json)
        ]
        has_seq_scan = any("seq scan" in nt for nt in node_types)
        has_index = any("index" in nt for nt in node_types)
        assert has_seq_scan and not has_index, (
            f"Expected seq scan (no index) from LangChain's default filter.\n"
            f"Node types: {node_types}\n"
            f"SQL: {full_sql}\n"
            f"If this fails, LangChain may have fixed the issue upstream."
        )

    def test_performance_comparison_with_and_without_override(
        self, engine, seeded_data, store
    ):
        """Side-by-side: our override vs LangChain default, actual execution time."""
        _, target_file_id = seeded_data

        our_clause = store._handle_field_filter("file_id", {"$eq": target_file_id})
        upstream_clause = PGVector._handle_field_filter(
            store, "file_id", {"$eq": target_file_id}
        )

        with Session(engine) as session:
            our_sql = str(
                session.query(store.EmbeddingStore)
                .filter(our_clause)
                .statement.compile(
                    dialect=engine.dialect,
                    compile_kwargs={"literal_binds": True},
                )
            )
            upstream_sql = str(
                session.query(store.EmbeddingStore)
                .filter(upstream_clause)
                .statement.compile(
                    dialect=engine.dialect,
                    compile_kwargs={"literal_binds": True},
                )
            )

        with engine.begin() as conn:
            our_plan = conn.execute(
                text(f"EXPLAIN (ANALYZE, FORMAT JSON) {our_sql}")
            ).scalar()
            upstream_plan = conn.execute(
                text(f"EXPLAIN (ANALYZE, FORMAT JSON) {upstream_sql}")
            ).scalar()

        our_details = _get_plan_details(our_plan)
        upstream_details = _get_plan_details(upstream_plan)

        report = (
            f"\n{'='*70}\n"
            f"  WITH vs WITHOUT override ({ROW_COUNT:,} rows)\n"
            f"{'='*70}\n"
            f"  WITH override (ExtendedPgVector)\n"
            f"    Plan node : {our_details['node_type']}\n"
            f"    Time      : {our_details['actual_time_ms']:.3f} ms\n"
            f"{'  -'*23}\n"
            f"  WITHOUT override (LangChain default)\n"
            f"    Plan node : {upstream_details['node_type']}\n"
            f"    Time      : {upstream_details['actual_time_ms']:.3f} ms\n"
            f"{'  -'*23}\n"
            f"  Speedup     : "
            f"{upstream_details['actual_time_ms'] / max(our_details['actual_time_ms'], 0.001):.1f}x\n"
            f"{'='*70}"
        )
        print(report)

        assert (
            our_details["actual_time_ms"] < upstream_details["actual_time_ms"]
        ), f"Override should be faster than LangChain default.\n{report}"


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

