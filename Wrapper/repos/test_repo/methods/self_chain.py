"""Method that calls self.helper; helper is the actual LLM invoker.

Tests the self.X -> enclosing_class.X resolution branch in iterate_once.
"""
from langchain.dummy import Anything


class AgentClient:
    def public_run(self, msg):
        return self.helper(msg)

    def helper(self, msg):
        obj = make()
        return obj.invoke({"input": msg})


def make():
    return None
