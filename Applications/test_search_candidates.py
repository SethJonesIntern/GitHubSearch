"""Unit tests for Stage 2 application search (no network).

Run from repo root:  python -m pytest Applications/test_search_candidates.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))       # import search_candidates
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import pipeline

import search_candidates as sc


# ── candidate decision: spec thresholds + sequential attribution ──────────────

def test_all_pass_is_candidate():
    assert sc.evaluate_candidate(60, 5, 4.0, 3) == (True, None)

def test_drop_lifetime_first():
    # fails everything; charged to lifetime (the first check)
    assert sc.evaluate_candidate(10, 0, 0.0, 0) == (False, "lifetime")

def test_drop_contributors():
    assert sc.evaluate_candidate(60, 1, 4.0, 3) == (False, "contributors")

def test_drop_commit_freq():
    assert sc.evaluate_candidate(60, 5, 1.99, 3) == (False, "commit_freq")

def test_drop_no_tests():
    assert sc.evaluate_candidate(60, 5, 4.0, 0) == (False, "no_tests")

def test_commit_freq_at_least_2_is_kept():
    # spec says "at least 2 commits a month" -> exactly 2 must pass
    assert sc.evaluate_candidate(60, 5, 2.0, 1) == (True, None)

def test_lifetime_exactly_30_kept():
    assert sc.evaluate_candidate(30, 2, 2.0, 1) == (True, None)

def test_contributors_exactly_2_kept():
    assert sc.evaluate_candidate(40, 2, 2.0, 1) == (True, None)

def test_none_signals_treated_as_zero():
    assert sc.evaluate_candidate(None, None, None, None) == (False, "lifetime")


# ── search-time filters (incl. Python primary-language requirement) ───────────

from datetime import datetime
_CUTOFF = datetime.fromisoformat("2025-04-14T00:00:00+00:00")

def _repo(**over):
    base = {
        "fork": False, "archived": False, "disabled": False, "language": "Python",
        "stargazers_count": 50, "pushed_at": "2026-01-01T00:00:00Z",
    }
    base.update(over)
    return base

def test_search_filters_happy_path():
    assert sc.passes_search_filters(_repo(), _CUTOFF) == (True, None)

def test_search_filters_non_python_repo_dropped():
    # the matched import was in a Python file, but the repo's primary language is not
    assert sc.passes_search_filters(_repo(language="TypeScript"), _CUTOFF) == (False, "not_python")

def test_search_filters_missing_language_dropped():
    assert sc.passes_search_filters(_repo(language=None), _CUTOFF) == (False, "not_python")

def test_search_filters_jupyter_dropped():
    # NOTE: notebook-primary repos are currently excluded (primary != "Python")
    assert sc.passes_search_filters(_repo(language="Jupyter Notebook"), _CUTOFF) == (False, "not_python")

def test_search_filters_fork_dropped():
    assert sc.passes_search_filters(_repo(fork=True), _CUTOFF) == (False, "fork_archived_disabled")

def test_search_filters_low_stars_dropped():
    assert sc.passes_search_filters(_repo(stargazers_count=9), _CUTOFF) == (False, "stars")

def test_search_filters_stale_dropped():
    assert sc.passes_search_filters(_repo(pushed_at="2024-01-01T00:00:00Z"), _CUTOFF) == (False, "stale")


# ── commits-per-month math ────────────────────────────────────────────────────

def test_commits_per_month_basic():
    # 30 commits over 60 days (~2 months) -> ~15/mo
    assert sc.commits_per_month_of(30, 60) == 15.0

def test_commits_per_month_min_one_month_floor():
    # very short lifetime is floored to 1 month so freq isn't overstated
    assert sc.commits_per_month_of(10, 5) == 10.0

def test_commits_per_month_no_lifetime():
    assert sc.commits_per_month_of(10, None) is None
    assert sc.commits_per_month_of(10, 0) is None


# ── import-pattern derivation from Stage 1 (replaces the curated dict) ─────────

def test_import_patterns_forms():
    assert sc.import_patterns("crewai") == [
        "from crewai import", "from crewai.", "import crewai",
    ]

def test_build_import_index_maps_names_to_frameworks():
    frameworks = [
        {"full_name": "langchain-ai/langchain", "import_names": ["langchain"]},
        {"full_name": "crewAIInc/crewAI", "import_names": ["crewai", "crewai_tools"]},
    ]
    idx = sc.build_import_index(frameworks)
    assert idx["langchain"] == ["langchain-ai/langchain"]
    assert idx["crewai"] == ["crewAIInc/crewAI"]
    assert idx["crewai_tools"] == ["crewAIInc/crewAI"]

def test_build_import_index_collision_maps_to_multiple():
    # two frameworks shipping the same generic import name -> both recorded
    frameworks = [
        {"full_name": "a/one", "import_names": ["agents"]},
        {"full_name": "b/two", "import_names": ["agents"]},
    ]
    assert sc.build_import_index(frameworks)["agents"] == ["a/one", "b/two"]

def test_build_import_index_skips_frameworks_without_import_names():
    frameworks = [{"full_name": "x/y", "import_names": []}]
    assert sc.build_import_index(frameworks) == {}


# ── row builders ──────────────────────────────────────────────────────────────

def _details():
    return {
        "full_name": "acme/app", "html_url": "h", "clone_url": "c", "default_branch": "main",
        "description": "d", "homepage": "hp", "owner": {"login": "acme", "type": "Organization"},
        "stargazers_count": 120, "forks_count": 9, "watchers_count": 120,
        "subscribers_count": 15, "network_count": 9, "open_issues_count": 3, "size": 500,
        "language": "Python", "topics": ["agents"], "license": {"spdx_id": "MIT"},
        "fork": False, "archived": False, "disabled": False,
        "visibility": "public", "is_template": False, "allow_forking": True,
        "has_issues": True, "has_projects": False, "has_wiki": True, "has_pages": False,
        "has_discussions": True, "has_downloads": True,
        "created_at": "2024-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "pushed_at": "2026-02-01T00:00:00Z",
    }

def test_metadata_row_schema_and_values():
    row = sc.build_metadata_row(_details(), ["langchain"], 365, 12, 400, 33.3, 5, True,
                                True, None)
    assert set(row) == set(sc.METADATA_FIELDS)
    assert row["owner_type"] == "Organization"
    assert row["subscribers_count"] == 15
    assert row["has_ci"] is True
    assert row["has_discussions"] is True
    assert row["is_candidate"] is True
    assert row["drop_reason"] == ""

def test_metadata_row_drop_reason_recorded():
    row = sc.build_metadata_row(_details(), ["langchain"], 10, 1, 0, 0.5, 0, False,
                                False, "lifetime")
    assert row["is_candidate"] is False
    assert row["drop_reason"] == "lifetime"

def test_candidate_row_schema():
    row = sc.build_candidate_row(_details(), ["langchain", "langchain"], 365, 12, 400, 33.3)
    assert set(row) == set(sc.CANDIDATE_FIELDS)
    assert row["matched_frameworks"] == "langchain"   # deduped
    assert row["license"] == "MIT"
