"""Direct invokers: function body contains a pattern match."""


def test_do_llm_is_direct(invokers):
    qname = "test_repo.direct.main.do_llm"
    assert qname in invokers
    assert invokers[qname].startswith("matches"), invokers[qname]


def test_make_helper_is_not_invoker(invokers):
    """A function that doesn't call anything LLM-related should not appear."""
    assert "test_repo.direct.main.make_helper" not in invokers
