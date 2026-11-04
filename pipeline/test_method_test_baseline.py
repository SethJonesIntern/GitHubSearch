"""Semantic checks for method denominators, nested calls, and missing run data."""
import ast
import contextlib
import csv
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.method_test_baseline import compare_repo, counts, definitions, main, write_csv


def mapping(name, line, end, test="test_target", rule="exact"):
    return {"repo": "o/r", "test_file": "o_r/tests/test_app.py", "test_method": test,
            "test_line": "1", "focal_file": "o_r/app.py", "focal_method": name,
            "focal_line": str(line), "focal_line_end": str(end), "method_rule": rule}


class BaselineTests(unittest.TestCase):
    def test_methods_are_denominator_and_tests_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "app.py").write_text(
                "def ask():\n    return llm.invoke()\ndef plain():\n    return 1\n")
            (root / "tests").mkdir()
            (root / "tests/test_app.py").write_text("def helper():\n    return 1\n")
            rows, audit = compare_repo("o/r", "o_r", root,
                [{"file": "o_r/app.py", "call_line": 2}],
                [mapping("ask", 1, 2), mapping("ask", 1, 2),
                 mapping("ask", 1, 2, test="test_second", rule="fuzzy")])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["mapped_tests"], 2)
            self.assertEqual(counts(rows)["llm_pct"], 100)
            self.assertEqual(counts(rows)["other_pct"], 0)
            self.assertEqual(audit["unmatched_production_focals"], 0)

    def test_nested_call_is_not_assigned_to_outer_method(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "app.py").write_text(
                "def outer():\n    async def inner():\n        return llm.invoke()\n    return inner\n")
            rows, _ = compare_repo("o/r", "o_r", root,
                                  [{"file": "o_r/app.py", "call_line": 3}], [])
            self.assertEqual([(r["qname"], r["group"]) for r in rows],
                             [("outer", "other"), ("outer.inner", "llm")])

    def test_snapshot_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "app.py").write_text("def changed():\n    return 1\n")
            _, audit = compare_repo("o/r", "o_r", root,
                [{"file": "o_r/missing.py", "call_line": 2}], [mapping("ask", 1, 2)])
            self.assertEqual(audit["unmatched_production_calls"], 1)
            self.assertEqual(audit["unmatched_production_focals"], 1)

    def test_no_synthetic_constructor(self):
        found = definitions(ast.parse("class A:\n    async def f(self):\n        pass\n"))
        self.assertEqual([m["qname"] for m in found], ["A.f"])

    def test_full_report_excludes_failed_run_instead_of_counting_it_as_untested(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, repos, output = root / "artifacts", root / "repos", root / "out"
            artifacts.mkdir()
            for slug in ("o_r", "o_failed", "o_zero"):
                clone = repos / slug
                clone.mkdir(parents=True)
                (clone / "app.py").write_text(
                    "def ask():\n    return llm.invoke()\ndef plain():\n    return 1\n")
            rows = [mapping("ask", 1, 2)]
            write_csv(artifacts / "pym2t_focal_tests.csv", rows, list(rows[0]))
            runs = [{"repo": repo, "slug": repo.replace("/", "_"), "status": status,
                     "focal_methods": focal, "cli_mismatch": 0}
                    for repo, status, focal in (("o/r", "ok", 1),
                                               ("o/failed", "crash", 0),
                                               ("o/zero", "no_tests", 0))]
            write_csv(artifacts / "pym2t_runs.csv", runs, list(runs[0]))
            fake = types.ModuleType("Applications.make_figures")
            fake.POP = {r["repo"] for r in runs}
            fake.CALLS = types.SimpleNamespace(to_dict=lambda _: [
                {"repo": r["repo"], "file": r["slug"] + "/app.py", "call_line": 2}
                for r in runs])
            with patch.dict("sys.modules", {"Applications.make_figures": fake}), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = main(["--artifacts-dir", str(artifacts), "--repos-dir", str(repos),
                               "--output-dir", str(output)])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(result, 0)
            self.assertEqual(summary["included_repos"], 2)
            self.assertEqual(summary["excluded_repos"], 1)
            self.assertEqual(summary["pooled_paired_repos"]["llm"]["pct"], 50)
            self.assertEqual(summary["pooled_paired_repos"]["other"]["methods"], 2)
            with (output / "methods.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)


if __name__ == "__main__":
    unittest.main()
