"""Top-level call of ClassName.method() — tests the ClassName.X
resolution branch via the file's name_map.
"""
from langchain.dummy import Anything


class Helper:
    @staticmethod
    def do_work():
        obj = make()
        return obj.invoke({"input": "x"})


def entry():
    return Helper.do_work()


def make():
    return None
