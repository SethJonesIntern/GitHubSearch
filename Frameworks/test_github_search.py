"""Unit tests for Stage 1 framework search (no network).

Run from repo root:  python -m pytest Frameworks/test_github_search.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # import GithubSearch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import pipeline

import GithubSearch as gs


# ── query building reflects the spec qualifiers ───────────────────────────────

def test_build_search_queries_applies_star_and_language():
    queries = gs.build_search_queries()
    assert len(queries) == len(gs.SEARCH_PHRASES)
    for phrase, q in zip(gs.SEARCH_PHRASES, queries):
        assert q.startswith(phrase)
        assert "stars:>=1000" in q
        assert "language:Python" in q


# ── filter conditions (the heart of the spec) ─────────────────────────────────

def _row(**over):
    base = {"archived": False, "contributors_count": 5, "test_file_count": 3, "stars": 1500}
    base.update(over)
    return base

def test_passes_filters_happy_path():
    assert gs.passes_filters(_row()) is True

def test_archived_repo_dropped():
    assert gs.passes_filters(_row(archived=True)) is False

def test_single_contributor_dropped():
    assert gs.passes_filters(_row(contributors_count=1)) is False

def test_exactly_two_contributors_kept():
    assert gs.passes_filters(_row(contributors_count=2)) is True

def test_no_test_files_dropped():
    assert gs.passes_filters(_row(test_file_count=0)) is False

def test_one_test_file_kept():
    assert gs.passes_filters(_row(test_file_count=1)) is True

def test_none_values_treated_as_zero():
    assert gs.passes_filters(_row(contributors_count=None)) is False
    assert gs.passes_filters(_row(test_file_count=None)) is False


# ── filter funnel (per-step drop counts) ──────────────────────────────────────

def test_apply_filters_funnel_counts():
    rows = [
        _row(),                                  # kept
        _row(archived=True),                     # dropped: archived
        _row(contributors_count=1),              # dropped: contributors
        _row(test_file_count=0),                 # dropped: no tests
        _row(contributors_count=7, test_file_count=2),  # kept
    ]
    kept, stats = gs.apply_filters(rows)
    assert len(kept) == 2
    assert stats == {
        "enriched": 5,
        "dropped_archived": 1,
        "dropped_contributors": 1,
        "dropped_no_tests": 1,
        "kept": 2,
    }

def test_apply_filters_sequential_attribution():
    # Archived AND single-contributor AND no tests — charged only to archived
    # (the first failing condition), so the funnel sums cleanly.
    rows = [_row(archived=True, contributors_count=1, test_file_count=0)]
    _, stats = gs.apply_filters(rows)
    assert stats["dropped_archived"] == 1
    assert stats["dropped_contributors"] == 0
    assert stats["dropped_no_tests"] == 0

def test_apply_filters_counts_sum_to_input():
    rows = [_row(), _row(archived=True), _row(test_file_count=0), _row(contributors_count=0)]
    _, stats = gs.apply_filters(rows)
    dropped = stats["dropped_archived"] + stats["dropped_contributors"] + stats["dropped_no_tests"]
    assert stats["kept"] + dropped == stats["enriched"] == len(rows)


# ── Link-header contributor counting ──────────────────────────────────────────

class _FakeResp:
    def __init__(self, status_code=200, link="", payload=None):
        self.status_code = status_code
        self.headers = {"Link": link} if link else {}
        self._payload = payload if payload is not None else []
    def json(self):
        return self._payload

def test_contributor_count_from_link_header(monkeypatch):
    link = '<https://api.github.com/...&page=2>; rel="next", ' \
           '<https://api.github.com/...&page=37>; rel="last"'
    monkeypatch.setattr(gs, "github_get", lambda *a, **k: _FakeResp(link=link))
    assert gs.get_contributor_count("o", "r") == 37

def test_contributor_count_single_page_fallback(monkeypatch):
    monkeypatch.setattr(gs, "github_get", lambda *a, **k: _FakeResp(payload=[{"login": "a"}]))
    assert gs.get_contributor_count("o", "r") == 1

def test_contributor_count_404(monkeypatch):
    monkeypatch.setattr(gs, "github_get", lambda *a, **k: None)
    assert gs.get_contributor_count("o", "r") == 0


# ── enrichment mapping (network helpers stubbed) ──────────────────────────────

def _tree(*paths, kind="blob"):
    return [{"type": kind, "path": p} for p in paths]


# ── import-name derivation ────────────────────────────────────────────────────

def test_import_names_flat_package():
    tree = _tree("langchain/__init__.py", "langchain/chains/__init__.py", "README.md")
    assert gs.derive_import_names(tree) == ["langchain"]

def test_import_names_src_layout():
    tree = _tree("src/pydantic_ai/__init__.py", "src/pydantic_ai/agent.py")
    assert gs.derive_import_names(tree) == ["pydantic_ai"]

def test_import_names_monorepo_root_not_subpackage():
    # the import name is 'langchain', not 'libs' or the nested subpackage
    tree = _tree(
        "libs/langchain/langchain/__init__.py",
        "libs/langchain/langchain/agents/__init__.py",
    )
    assert gs.derive_import_names(tree) == ["langchain"]

def test_import_names_excludes_tests_and_docs():
    tree = _tree("tests/__init__.py", "docs/__init__.py", "mypkg/__init__.py")
    assert gs.derive_import_names(tree) == ["mypkg"]

def test_import_names_multiple_top_level_packages():
    tree = _tree("foo/__init__.py", "bar/__init__.py")
    assert gs.derive_import_names(tree) == ["bar", "foo"]

def test_import_names_none_when_no_packages():
    assert gs.derive_import_names(_tree("main.py", "setup.py")) == []

def test_import_names_excludes_packages_under_examples_dir():
    # example/sample packages nested under examples/ are not the project's import
    tree = _tree("realpkg/__init__.py", "examples/demo_app/__init__.py")
    assert gs.derive_import_names(tree) == ["realpkg"]

def test_import_names_excludes_non_identifier_template_dirs():
    # cookiecutter template dirs like '{{folder_name}}' aren't valid imports
    tree = _tree("crewai/__init__.py", "{{folder_name}}/__init__.py")
    assert gs.derive_import_names(tree) == ["crewai"]


def test_enrich_repo_maps_fields(monkeypatch):
    monkeypatch.setattr(gs, "get_contributor_count", lambda o, r: 9)
    monkeypatch.setattr(gs, "get_default_branch_commit_date", lambda o, r, b: "2026-01-01T00:00:00Z")
    monkeypatch.setattr(gs, "get_tree", lambda o, r, b: _tree("widget/__init__.py", "widget/core.py"))
    monkeypatch.setattr(gs, "get_test_metrics", lambda o, r, b, t=None: (4, 11, True))
    # full repo payload adds subscribers/network/flags not present on the search item
    monkeypatch.setattr(gs, "get_repo_details", lambda o, r: {
        "subscribers_count": 42, "network_count": 100, "watchers_count": 2000,
        "owner": {"login": "acme", "type": "Organization"},
        "has_discussions": True, "visibility": "public",
    })

    item = {
        "full_name": "acme/widget",
        "html_url": "https://github.com/acme/widget",
        "description": "a framework",
        "homepage": "https://acme.dev",
        "stargazers_count": 2000,
        "forks_count": 100,
        "language": "Python",
        "topics": ["agents", "llm"],
        "open_issues_count": 7,
        "size": 1234,
        "default_branch": "main",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "pushed_at": "2026-01-02T00:00:00Z",
        "archived": False,
        "disabled": False,
        "fork": False,
        "license": {"spdx_id": "MIT"},
        "clone_url": "https://github.com/acme/widget.git",
    }
    row = gs.enrich_repo(item, "AI agent framework stars:>=1000 language:Python")

    assert row["full_name"] == "acme/widget"
    assert row["stars"] == 2000
    assert row["topics"] == "agents,llm"
    assert row["license"] == "MIT"
    assert row["contributors_count"] == 9
    assert row["test_file_count"] == 4
    assert row["test_function_count"] == 11
    # richer metadata from the merged full payload
    assert row["subscribers_count"] == 42
    assert row["network_count"] == 100
    assert row["owner_type"] == "Organization"
    assert row["has_discussions"] is True
    assert row["has_ci"] is True
    assert row["import_names"] == "widget"
    assert set(row) == set(gs.FIELDNAMES)
    assert gs.passes_filters(row) is True
