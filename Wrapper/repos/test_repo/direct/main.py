"""Scenario: a function whose body directly contains a .invoke call."""
from langchain.dummy import Anything


def do_llm():
    obj = make_helper()
    return obj.invoke({"input": "hi"})


def make_helper():
    return None
