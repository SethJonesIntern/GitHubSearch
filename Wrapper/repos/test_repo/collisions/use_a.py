"""Imports wrapper from wrapper_a — caller_a SHOULD be flagged transitive."""
from test_repo.collisions.wrapper_a import wrapper


def caller_a():
    return wrapper()
