"""Tests for the Stage-6 slicer additions: recursive discovery + the
llm_invokers_all.csv function filter. None of these need Joern."""
from __future__ import annotations

import csv
from pathlib import Path

from pipeline.per_variable_pdg_slicer import (
    _allowed_lines_for_file,
    _norm_path_parts,
    _parts_suffix_match,
    _select_source_files,
    function_filter_from_rows,
    load_function_filter,
)


# ── path/suffix helpers ─────────────────────────────────────────────────────────


def test_norm_path_parts_normalizes_separators_and_dots():
    assert _norm_path_parts("foo\\bar/baz.py") == ("foo", "bar", "baz.py")
    assert _norm_path_parts("./a/./b.py") == ("a", "b.py")
    assert _norm_path_parts("") == ()


def test_parts_suffix_match():
    longer = ("foo_bar", "pkg", "mod.py")
    assert _parts_suffix_match(longer, ("pkg", "mod.py"))
    assert _parts_suffix_match(("pkg", "mod.py"), longer)
    assert _parts_suffix_match(longer, longer)
    assert not _parts_suffix_match(longer, ("other", "mod.py"))
    assert not _parts_suffix_match((), ("mod.py",))


# ── invoker CSV loading ─────────────────────────────────────────────────────────


def _write_invokers_csv(path: Path, rows: list[dict]) -> None:
    fields = ["repo", "qname", "file", "line", "reason", "kind"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_load_function_filter_reads_file_and_line(tmp_path):
    csv_path = tmp_path / "llm_invokers_all.csv"
    _write_invokers_csv(csv_path, [
        {"repo": "o/r", "qname": "r.pkg.mod.f", "file": "r/pkg/mod.py",
         "line": "10", "reason": "matches '.invoke'", "kind": "direct"},
        {"repo": "o/r", "qname": "r.pkg.mod.g", "file": "r/pkg/mod.py",
         "line": "25", "reason": "calls r.pkg.mod.f", "kind": "transitive"},
    ])
    flt = load_function_filter(csv_path)
    assert flt == {
        (("r", "pkg", "mod.py"), 10),
        (("r", "pkg", "mod.py"), 25),
    }


def test_load_function_filter_scopes_by_repo(tmp_path):
    csv_path = tmp_path / "llm_invokers_all.csv"
    _write_invokers_csv(csv_path, [
        {"repo": "o/keep", "qname": "k.f", "file": "keep/a.py",
         "line": "3", "reason": "matches", "kind": "direct"},
        {"repo": "o/drop", "qname": "d.f", "file": "drop/a.py",
         "line": "7", "reason": "matches", "kind": "direct"},
    ])
    flt = load_function_filter(csv_path, repo="o/keep")
    assert flt == {(("keep", "a.py"), 3)}


def test_load_function_filter_skips_bad_rows(tmp_path):
    csv_path = tmp_path / "llm_invokers_all.csv"
    _write_invokers_csv(csv_path, [
        {"repo": "o/r", "qname": "r.f", "file": "", "line": "10",
         "reason": "x", "kind": "direct"},
        {"repo": "o/r", "qname": "r.g", "file": "r/a.py", "line": "",
         "reason": "x", "kind": "direct"},
        {"repo": "o/r", "qname": "r.h", "file": "r/a.py", "line": "notint",
         "reason": "x", "kind": "direct"},
        {"repo": "o/r", "qname": "r.i", "file": "r/a.py", "line": "5",
         "reason": "x", "kind": "direct"},
    ])
    assert load_function_filter(csv_path) == {(("r", "a.py"), 5)}


# ── in-memory filter from invoker rows (the folded-in driver path) ──────────────


def test_function_filter_from_rows_matches_csv_loader():
    rows = [
        {"repo": "o/r", "qname": "r.pkg.mod.f", "file": "r/pkg/mod.py", "line": 10},
        {"repo": "o/r", "qname": "r.pkg.mod.g", "file": "r/pkg/mod.py", "line": "25"},
        {"repo": "o/r", "qname": "bad", "file": "", "line": 5},
        {"repo": "o/r", "qname": "bad2", "file": "r/a.py", "line": None},
    ]
    assert function_filter_from_rows(rows) == {
        (("r", "pkg", "mod.py"), 10),
        (("r", "pkg", "mod.py"), 25),
    }


# ── per-file allowed-lines (suffix match tolerates the repo-slug prefix) ─────────


def test_allowed_lines_none_when_no_filter():
    assert _allowed_lines_for_file(None, Path("/x/pkg/mod.py"), Path("/x")) is None


def test_allowed_lines_matches_through_slug_prefix():
    # CSV path carries the clone-slug segment ("foo_bar"); the slicer sees the
    # path relative to the checkout root (no slug). Suffix match bridges them.
    flt = {(("foo_bar", "pkg", "mod.py"), 10), (("foo_bar", "pkg", "mod.py"), 42)}
    base = Path("/work/foo_bar")
    src = base / "pkg" / "mod.py"
    assert _allowed_lines_for_file(flt, src, base) == {10, 42}


def test_allowed_lines_empty_for_unlisted_file():
    flt = {(("foo_bar", "pkg", "mod.py"), 10)}
    base = Path("/work/foo_bar")
    other = base / "pkg" / "other.py"
    assert _allowed_lines_for_file(flt, other, base) == set()


# ── recursive discovery + skip dirs ─────────────────────────────────────────────


def test_select_source_files_recursive_and_skips_junk(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 1\n", encoding="utf-8")
    for junk in (".git", "__pycache__", "node_modules"):
        (tmp_path / junk).mkdir()
        (tmp_path / junk / "c.py").write_text("z = 1\n", encoding="utf-8")

    flat = _select_source_files(tmp_path, None, recursive=False)
    assert {p.name for p in flat} == {"a.py"}

    recursive = _select_source_files(tmp_path, None, recursive=True)
    assert {p.relative_to(tmp_path).as_posix() for p in recursive} == {"a.py", "sub/b.py"}
