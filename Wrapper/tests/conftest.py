"""Pytest fixtures.

Runs the full transitive-invokers pipeline against repos/test_repo/ once per
session and exposes the resulting {qname -> reason} dict as the `invokers`
fixture.  Tests in this directory assert on the presence of specific qnames
in that dict — never on counts, so adding new fixtures elsewhere doesn't
break unrelated tests.
"""
import sys
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WRAPPER))

from transitive_invokers import index_repo, seed_invokers, build_call_graph, transitive_closure  # noqa: E402


@pytest.fixture(scope="session")
def invokers():
    test_repo = WRAPPER / "repos" / "test_repo"
    repo_root = WRAPPER / "repos"
    functions, contexts = index_repo(test_repo, repo_root)
    seeds = seed_invokers(functions, contexts)
    call_graph = build_call_graph(test_repo, repo_root)
    return transitive_closure(seeds, call_graph)
