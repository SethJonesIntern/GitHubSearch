"""One-hop wrapper: function that calls a direct invoker."""


def test_wrapper_is_direct(invokers):
    assert "test_repo.one_hop.wrapper.do_llm" in invokers


def test_caller_is_transitive(invokers):
    qname = "test_repo.one_hop.caller.entry"
    assert qname in invokers
    assert "calls test_repo.one_hop.wrapper.do_llm" in invokers[qname]
