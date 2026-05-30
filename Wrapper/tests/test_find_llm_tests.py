"""Tests for find_llm_tests.

Unit tests on the convention-check helpers, plus an integration assertion
that the existing invokers fixture flags the synthetic pytest test fixture
but find_llm_tests-style filtering would keep only the test_ function.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from find_llm_tests import is_test_file, is_test_function


# ── unit tests on the convention checks ──────────────────────────────────────


def test_is_test_file_with_test_prefix():
    assert is_test_file("foo/test_bar.py")
    assert is_test_file("test_bar.py")


def test_is_test_file_with_test_suffix():
    assert is_test_file("foo/bar_test.py")


def test_is_not_test_file():
    assert not is_test_file("foo/bar.py")
    # conftest.py contains 'test' but doesn't match the patterns
    assert not is_test_file("conftest.py")
    # 'testing.py' doesn't start with 'test_' or end with '_test.py'
    assert not is_test_file("foo/testing.py")


def test_is_test_function_function_form():
    assert is_test_function("pkg.mod.test_foo")


def test_is_test_function_method_form():
    """A method named test_X on a Test class still ends in test_X."""
    assert is_test_function("pkg.mod.TestY.test_method")


def test_is_not_test_function():
    assert not is_test_function("pkg.mod.foo")
    # 'testing' is not prefixed by 'test_'
    assert not is_test_function("pkg.mod.testing")


# ── integration: fixture-based ───────────────────────────────────────────────


def test_synthetic_pytest_test_is_flagged(invokers):
    """The test function in pytest_tests/test_thing.py is in the invoker set."""
    qname = "test_repo.pytest_tests.test_thing.test_invokes_llm"
    assert qname in invokers
    assert is_test_function(qname)


def test_non_test_helper_is_invoker_but_not_a_test(invokers):
    """The non-test helper in the same file is still an LLM-touching invoker,
    but find_llm_tests would correctly filter it out by name."""
    qname = "test_repo.pytest_tests.test_thing.helper_function"
    assert qname in invokers, "helper_function calls .invoke so transitive_invokers must flag it"
    assert not is_test_function(qname), "but its name doesn't match pytest conventions"
