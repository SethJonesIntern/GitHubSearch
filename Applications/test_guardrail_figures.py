"""Synthetic checks for input consistency and refusal to overwrite outputs."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from Applications.guardrail_figures import check_baseline, inventory, main, write_csv


class FigureSafetyTests(unittest.TestCase):
    def test_existing_directory_is_refused_without_changing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "existing.csv"
            sentinel.write_bytes(b"important research data\n")
            with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
                main(["--output-dir", temp])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(sentinel.read_bytes(), b"important research data\n")
            self.assertEqual(list(Path(temp).iterdir()), [sentinel])

    def test_existing_table_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "table.csv"
            path.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_csv(path, [{"a": 1}], ["a"])
            self.assertEqual(path.read_text(), "preserve")

    def test_source_is_parsed_without_executing_top_level_code(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "registry.py"
            source.write_text("""
raise RuntimeError('Do not execute this file')
FRAMEWORK_CALLS = {'langchain': ['Agent', '.invoke'], 'openai': ['.create']}
_LANGCHAIN_INTEGRATIONS = ['langchain_extra']
_KEEP_BARE_NAMES = set()
_IN_SCOPE_EXPLICIT = {'openai'}
_IN_SCOPE_PREFIXES = ('langchain',)
""", encoding="utf-8")
            grouping = Path(temp) / "groups.py"
            grouping.write_text("PREFIX_GROUPS={'langchain':'langchain'}\nEXTRA_MEMBERS={}\n")
            rows, group, _ = inventory(source, grouping)
            self.assertEqual(group("langchain_extra"), "langchain")
            self.assertEqual([r["pattern"] for r in rows if r["package"] == "langchain"], [".invoke"])
            self.assertEqual(sum(r["observed_study_scope"] for r in rows), 2)

    def test_inconsistent_summary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "totals disagree"):
            check_baseline({"included_repos": 2, "excluded_repos": 1, "selected_repos": 4}, [])

    def test_duplicate_repositories_are_rejected(self):
        summary = {"included_repos": 2, "excluded_repos": 0, "selected_repos": 2}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            check_baseline(summary, [{"repo": "a/b"}, {"repo": "a/b"}])

    def test_stale_percentages_are_rejected_even_when_counts_agree(self):
        groups = {"llm": {"methods": 2, "with_mapped_test": 1, "with_exact_test": 1,
                          "pct": 50, "exact_pct": 50},
                  "other": {"methods": 4, "with_mapped_test": 1, "with_exact_test": 0,
                            "pct": 25, "exact_pct": 0}}
        summary = {"included_repos": 1, "excluded_repos": 0, "selected_repos": 1,
                   "paired_repos": 1, "pooled_all_complete_repos": groups,
                   "pooled_paired_repos": groups, "macro_paired_pct": {"llm": 50, "other": 25}}
        rows = [{"repo": "a/b", "llm_methods": "2", "llm_mapped": "1",
                 "llm_exact_mapped": "1", "other_methods": "4", "other_mapped": "1",
                 "other_exact_mapped": "0"}]
        self.assertEqual(len(check_baseline(summary, rows)), 1)
        groups["llm"]["pct"] = 60
        with self.assertRaisesRegex(ValueError, "percentage mismatch"):
            check_baseline(summary, rows)


if __name__ == "__main__":
    unittest.main()
