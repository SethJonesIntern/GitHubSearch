"""Tests for Stage 3 framework frequency aggregation (no IO).

Run from repo root:  python -m pytest Applications/test_framework_distribution.py -q
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import framework_distribution as fd


def test_rows_union_and_values():
    repo = Counter({"langchain": 100, "crewai": 40, "obscure": 5})
    cand = Counter({"langchain": 30, "crewai": 12})
    kept = Counter({"langchain": 10, "crewai": 4})
    rows = fd.build_frequency_rows(repo, cand, kept, kept_total=12)
    by = {r["framework"]: r for r in rows}
    assert set(by) == {"langchain", "crewai", "obscure"}      # union of all views
    assert by["langchain"]["repos_imported"] == 100
    assert by["langchain"]["repos_candidate"] == 30
    assert by["langchain"]["repos_kept"] == 10
    assert by["obscure"]["repos_kept"] == 0                    # imported but never kept


def test_sorted_by_kept_then_imported():
    repo = Counter({"a": 5, "b": 50})
    kept = Counter({"a": 3, "b": 3})        # tie on kept -> imported breaks it
    rows = fd.build_frequency_rows(repo, Counter(), kept, kept_total=6)
    assert [r["framework"] for r in rows] == ["b", "a"]


def test_pct_of_kept():
    rows = fd.build_frequency_rows(Counter(), Counter(), Counter({"x": 5}), kept_total=20)
    assert rows[0]["pct_of_kept"] == 25.0

def test_pct_blank_without_total():
    rows = fd.build_frequency_rows(Counter({"x": 1}), Counter(), Counter(), kept_total=0)
    assert rows[0]["pct_of_kept"] == ""

def test_schema():
    rows = fd.build_frequency_rows(Counter({"x": 1}), Counter(), Counter({"x": 1}), 1)
    assert set(rows[0]) == set(fd.FIELDS)

def test_empty():
    assert fd.build_frequency_rows(Counter(), Counter(), Counter(), 0) == []
