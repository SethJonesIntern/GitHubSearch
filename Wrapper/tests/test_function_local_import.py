"""Function-local imports — names bound inside a function body must still
resolve during transitive iteration."""


def test_deep_llm_is_direct(invokers):
    assert "test_repo.function_local_import.deep_llm.do_deep_llm" in invokers


def test_orchestrate_is_transitive(invokers):
    qname = "test_repo.function_local_import.orchestrator.orchestrate"
    assert qname in invokers, (
        "orchestrate uses a function-local import to bind do_deep_llm; "
        "without fi.local_names the call would fail to resolve."
    )
    assert "calls test_repo.function_local_import.deep_llm.do_deep_llm" in invokers[qname]
