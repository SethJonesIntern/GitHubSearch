"""Tests for the eval frequency table (no IO).

Run from repo root:  python -m pytest pipeline/test_eval_frequency.py -q
"""
from pipeline import eval_frequency as ef


def _call(repo, framework):
    return {"repo": repo, "framework": framework}


def test_counts_repos_and_sites():
    rows = [
        _call("a/x", "deepeval"), _call("a/x", "deepeval"),   # same repo, 2 sites
        _call("b/y", "deepeval"),                              # another repo
        _call("b/y", "ragas"),
    ]
    table = ef.build_frequency(rows)
    by_fw = {r["eval_framework"]: r for r in table}
    assert by_fw["deepeval"]["repos_with_calls"] == 2
    assert by_fw["deepeval"]["total_call_sites"] == 3
    assert by_fw["ragas"]["repos_with_calls"] == 1
    assert by_fw["ragas"]["total_call_sites"] == 1


def test_sorted_by_repos_desc():
    rows = [_call("a/x", "ragas"), _call("a/x", "deepeval"), _call("b/y", "deepeval")]
    table = ef.build_frequency(rows)
    assert [r["eval_framework"] for r in table] == ["deepeval", "ragas"]  # deepeval 2 > ragas 1


def test_pct_of_apps():
    rows = [_call("a/x", "deepeval"), _call("b/y", "deepeval")]
    table = ef.build_frequency(rows, total_apps=8)
    assert table[0]["pct_of_apps"] == 25.0   # 2 of 8

def test_pct_blank_without_total():
    table = ef.build_frequency([_call("a/x", "deepeval")], total_apps=None)
    assert table[0]["pct_of_apps"] == ""

def test_empty_input():
    assert ef.build_frequency([]) == []

def test_schema():
    table = ef.build_frequency([_call("a/x", "deepeval")], total_apps=1)
    assert set(table[0]) == set(ef.FIELDS)
