"""Top of the multi-hop chain: should be flagged after 3 iterations."""
from test_repo.multi_hop.b import step_b


def step_a():
    return step_b()
