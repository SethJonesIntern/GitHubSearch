"""Multi-hop chain: a -> b -> c -> d.  The fixed-point loop must reach
all four through successive iterations.
"""


def test_d_is_direct(invokers):
    assert "test_repo.multi_hop.d.step_d" in invokers


def test_c_calls_d(invokers):
    qname = "test_repo.multi_hop.c.step_c"
    assert qname in invokers
    assert "calls test_repo.multi_hop.d.step_d" in invokers[qname]


def test_b_calls_c(invokers):
    qname = "test_repo.multi_hop.b.step_b"
    assert qname in invokers
    assert "calls test_repo.multi_hop.c.step_c" in invokers[qname]


def test_a_calls_b(invokers):
    qname = "test_repo.multi_hop.a.step_a"
    assert qname in invokers
    assert "calls test_repo.multi_hop.b.step_b" in invokers[qname]
