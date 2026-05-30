"""Classmethod calling cls.from_config which is the actual invoker.

Tests the cls.X -> enclosing_class.X resolution branch.
"""
from langchain.dummy import Anything


class AgentClient:
    @classmethod
    def make_default(cls):
        return cls.from_config({})

    @classmethod
    def from_config(cls, config):
        obj = make()
        return obj.invoke({"input": "x"})


def make():
    return None
