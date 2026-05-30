"""Imports wrapper from wrapper_b — caller_b should NOT be an invoker.

This is the disambiguation test: same function name 'wrapper' in both
collisions/use_a.py and collisions/use_b.py, but the name maps in each
file resolve to different qualified names.
"""
from test_repo.collisions.wrapper_b import wrapper


def caller_b():
    return wrapper()
