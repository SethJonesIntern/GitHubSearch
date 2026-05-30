"""Direct invoker imported lazily by orchestrator.py via a function-local import."""
from langchain.dummy import Anything


def do_deep_llm(req):
    obj = make()
    return obj.invoke({"input": req})


def make():
    return None
