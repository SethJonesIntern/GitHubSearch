"""End of the one-hop chain: a direct invoker."""
from langchain.dummy import Anything


def do_llm():
    obj = make_helper()
    return obj.invoke({"input": "hi"})


def make_helper():
    return None
