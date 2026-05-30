"""Bottom of the multi-hop chain: direct invoker."""
from langchain.dummy import Anything


def step_d():
    obj = make()
    return obj.invoke({})


def make():
    return None
