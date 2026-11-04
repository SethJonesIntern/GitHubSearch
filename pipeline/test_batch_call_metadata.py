"""Tests for the Stage 5/7 batch driver (no network; uses the bundled fixture
and synthetic temp repos).

Run from repo root:  python -m pytest pipeline/test_batch_call_metadata.py -q
"""
from pathlib import Path

from pipeline import batch_call_metadata as b

FIXTURE = Path(__file__).resolve().parent.parent / "Wrapper" / "repos" / "test_repo"


# ── pure helpers ──────────────────────────────────────────────────────────────

def test_calls_from_metadata_collapses_by_call_id():
    meta = [
        {"call_id": "f::1::0", "file": "f", "enclosing_qname": "m.fn", "framework": "langchain",
         "pattern": ".invoke", "callable": "c.invoke", "call_source": "c.invoke(a, b)",
         "call_line": 1, "call_col": 0, "is_await": False, "arg_count": 2},
        {"call_id": "f::1::0", "file": "f", "enclosing_qname": "m.fn", "framework": "langchain",
         "pattern": ".invoke", "callable": "c.invoke", "call_source": "c.invoke(a, b)",
         "call_line": 1, "call_col": 0, "is_await": False, "arg_count": 2},
        {"call_id": "f::9::4", "file": "f", "enclosing_qname": "m.fn", "framework": "langchain",
         "pattern": ".invoke", "callable": "d.invoke", "call_source": "d.invoke()",
         "call_line": 9, "call_col": 4, "is_await": False, "arg_count": 0},
    ]
    calls = b._calls_from_metadata("o/r", meta)
    assert len(calls) == 2                      # two distinct call_ids
    assert all(c["repo"] == "o/r" for c in calls)
    assert set(calls[0]) == set(b.CALL_FIELDS)


def test_invokers_rows_and_tests_subset():
    class FI:
        def __init__(self, fp, line): self.file_path, self.line = fp, line
    functions = {
        "pkg.test_x.test_a": FI("pkg/test_x.py", 3),      # pytest test (direct)
        "pkg.helpers.do_thing": FI("pkg/helpers.py", 8),   # invoker, not a test
        "pkg.app.test_b": FI("pkg/app.py", 5),             # test_ name, not a test file
    }
    invokers = {
        "pkg.test_x.test_a": "matches '.invoke' from langchain",
        "pkg.helpers.do_thing": "matches '.invoke' from langchain",
        "pkg.app.test_b": "calls pkg.helpers.do_thing",
    }
    rows = b._invokers_rows("o/r", invokers, functions)
    assert len(rows) == 3                        # all invokers, not just tests
    assert {r["kind"] for r in rows} == {"direct", "transitive"}
    assert set(rows[0]) == set(b.INVOKER_FIELDS)

    tests = b._tests_among(rows)
    assert len(tests) == 1                        # only the real pytest test file+name
    assert tests[0]["qname"] == "pkg.test_x.test_a"
    assert tests[0]["kind"] == "direct"


# ── integration against the bundled fixture (LLM only) ────────────────────────

def test_process_repo_fixture_llm():
    res = b.process_repo("test/fixture", FIXTURE)
    assert len(res["llm_invokers"]) > 0         # the invoker search result
    assert len(res["llm_calls"]) > 0
    assert len(res["call_metadata"]) > 0
    assert len(res["eval_calls"]) == 0          # fixture imports no eval frameworks
    # invokers are a superset of (or equal to) the tests among them
    assert len(res["llm_invokers"]) >= len(res["llm_tests"])
    # schemas
    assert set(res["llm_invokers"][0]) == set(b.INVOKER_FIELDS)
    assert set(res["llm_calls"][0]) == set(b.CALL_FIELDS)
    assert set(res["call_metadata"][0]) == set(b.METADATA_FIELDS)
    # repo tagging + framework attribution
    assert all(r["repo"] == "test/fixture" for r in res["llm_invokers"])
    assert all(r["framework"] == "langchain" for r in res["call_metadata"])


# ── integration against a synthetic eval repo ─────────────────────────────────

def test_process_repo_eval(tmp_path):
    pkg = tmp_path / "evalapp" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "run.py").write_text(
        "from deepeval import assert_test, evaluate\n"
        "def check(case, metric):\n"
        "    evaluate([case], [metric])\n"
        "    assert_test(case, [metric])\n"
    )
    res = b.process_repo("demo/evalapp", tmp_path / "evalapp")
    frameworks = {c["framework"] for c in res["eval_calls"]}
    callables = {c["callable"] for c in res["eval_calls"]}
    assert frameworks == {"deepeval"}
    assert "evaluate" in callables and "assert_test" in callables
    assert len(res["llm_calls"]) == 0           # deepeval is not an LLM framework
    assert set(res["eval_call_metadata"][0]) == set(b.METADATA_FIELDS)
