"""Per-file name disambiguation.

Both collisions/wrapper_a.py and collisions/wrapper_b.py define a top-level
function named `wrapper`.  Files importing each one must resolve to their
specific qualified name; only the caller of the LLM-invoking variant should
be flagged.
"""


def test_wrapper_a_is_invoker(invokers):
    assert "test_repo.collisions.wrapper_a.wrapper" in invokers


def test_wrapper_b_is_not_invoker(invokers):
    assert "test_repo.collisions.wrapper_b.wrapper" not in invokers


def test_use_a_caller_is_transitive(invokers):
    qname = "test_repo.collisions.use_a.caller_a"
    assert qname in invokers
    assert "calls test_repo.collisions.wrapper_a.wrapper" in invokers[qname]


def test_use_b_caller_is_not_invoker(invokers):
    """caller_b imports wrapper from wrapper_b (non-invoker); must not be flagged."""
    assert "test_repo.collisions.use_b.caller_b" not in invokers
