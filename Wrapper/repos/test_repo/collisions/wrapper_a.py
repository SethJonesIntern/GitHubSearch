"""A function named 'wrapper' that IS a direct invoker."""
from langchain.dummy import Anything


def wrapper():
    obj = make()
    return obj.invoke({})


def make():
    return None
