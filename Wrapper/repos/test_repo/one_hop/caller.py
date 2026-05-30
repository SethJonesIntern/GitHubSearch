"""Calls do_llm from wrapper.py — should be flagged transitively."""
from test_repo.one_hop.wrapper import do_llm


def entry():
    return do_llm()
