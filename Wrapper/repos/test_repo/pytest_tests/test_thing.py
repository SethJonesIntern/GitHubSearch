"""Fixture for find_llm_tests filtering.

This file's basename starts with `test_` (matches pytest discovery), and
contains both a test function (test_invokes_llm) and a non-test helper
(helper_function) that also touches an LLM call.  find_llm_tests should
flag only the test_ function.
"""
from langchain.dummy import Anything


def test_invokes_llm():
    obj = make()
    return obj.invoke({"input": "hi"})


def helper_function():
    """An LLM-touching helper that is NOT a pytest test — find_llm_tests
    should leave it out even though transitive_invokers picks it up."""
    obj = make()
    return obj.invoke({})


def make():
    return None
