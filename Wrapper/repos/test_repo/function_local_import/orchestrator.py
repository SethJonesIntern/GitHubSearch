"""Wrapper function that imports its invoker LOCALLY (inside the function body).

Tests that fi.local_names captures function-local imports so that callers
of names bound this way still resolve during iteration.
"""


def orchestrate(req):
    from test_repo.function_local_import.deep_llm import do_deep_llm
    return do_deep_llm(req)
