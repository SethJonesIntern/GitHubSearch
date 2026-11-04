"""Independent acceptance tests for specs/direct-invoker-test-flow.spec.md.

Synthetic source and saved rows only. The tester does not execute this suite.
"""
import contextlib
import csv
import hashlib
import json
import sys
import types
from unittest import mock
import io
from pathlib import Path
import tempfile
import unittest

from pipeline import direct_invoker_test_flow as flow


EVIDENCE = "saved_pyan_edge_local_return_to_passfail_candidate"
CAVEAT = "Saved pyan target plus syntactic call match; dispatch and execution are unverified."


def row(repo="acme/app", qname="acme_app.app.f", file="acme_app/app.py",
        line=1, reason="matches framework", kind="direct"):
    return dict(repo=repo, qname=qname, file=file, line=line,
                reason=reason, kind=kind)


class DirectInvokerTestFlowTests(unittest.TestCase):
    def test_edges_join_exact_repo_target_deduplicate_and_exclude_self(self):
        invokers = [row(), row(repo="other/app")]
        candidate = row(qname="acme_app.tests.test_f", file="acme_app/tests.py",
                        line=7, reason="calls acme_app.app.f", kind="transitive")
        edges, diagnostics = flow.select_test_edges(invokers, [
            candidate, dict(candidate),
            row(reason="calls acme_app.app.f", kind="transitive"),
            row(repo="absent/app", qname="test_f", file="tests.py", line=9,
                reason="calls acme_app.app.f", kind="transitive"),
            row(qname="test_direct", file="tests.py", line=10,
                reason="matches framework", kind="direct"),
            row(qname="test_wrong_module", file="tests.py", line=11,
                reason="calls wrong.module.f", kind="transitive"),
        ])
        self.assertEqual(edges, [{
            "repo": "acme/app", "invoker_file": "acme_app/app.py",
            "invoker_line": 1, "invoker_qname": "acme_app.app.f",
            "test_file": "acme_app/tests.py", "test_line": 7,
            "test_qname": "acme_app.tests.test_f",
            "reason": "calls acme_app.app.f",
        }])
        self.assertIsInstance(diagnostics, list)

        invokers = [row(), row(file="acme_app/alternate.py", line=20)]
        edges, diagnostics = flow.select_test_edges(invokers, [
            row(qname="test_f", file="tests.py", line=4,
                reason="calls acme_app.app.f", kind="transitive"),
            row(qname="test_wrapper", file="tests.py", line=8,
                reason="calls acme_app.app.wrapper", kind="transitive"),
        ])
        self.assertEqual(edges, [])
        self.assertTrue(any(d["kind"] == "ambiguous_target" for d in diagnostics))
        # The deferred diagnostic's kind is intentionally not prescribed by the spec.
        self.assertTrue(any(d["file"] == "tests.py" and d["line"] == 8
                            for d in diagnostics))

    def test_plain_assert_needs_no_pytest_import_or_focal_eligibility(self):
        result = flow.analyze_saved_test(
            "def test_f():\n    r = f()\n    assert len(r) > 0\n", 1,
            "acme_app-name.app.f")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["checks"]), 1)
        check = result["checks"][0]
        self.assertEqual((check["call_line"], check["check_line"],
                          check["failure_line"], check["check_kind"]),
                         (2, 3, None, "assert"))
        self.assertEqual(check["evidence"], EVIDENCE)
        self.assertEqual(check["trace_lines"], [2, 3])
        self.assertEqual(check["variables"], ["r"])
        self.assertIn(CAVEAT, result["limitations"])

    def test_async_unittest_and_exact_definition_line(self):
        source = ("class First:\n"
                  "    async def test_f(self):\n"
                  "        r = await f()\n"
                  "        self.assertEqual(r, 'x')\n"
                  "class Second:\n"
                  "    async def test_f(self):\n"
                  "        r = await f()\n"
                  "        self.assertEqual(r, 'y')\n")
        first = flow.analyze_saved_test(source, 2, "acme_app.app.f")
        second = flow.analyze_saved_test(source, 6, "acme_app.app.f")
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        self.assertEqual([(c["call_line"], c["check_line"])
                          for c in first["checks"]], [(3, 4)])
        self.assertEqual([(c["call_line"], c["check_line"])
                          for c in second["checks"]], [(7, 8)])
        self.assertEqual(first["checks"][0]["evidence"], EVIDENCE)

    def test_local_alias_and_derived_subscript_reach_assertion(self):
        source = ("def test_f():\n"
                  "    r = f()\n"
                  "    alias = r\n"
                  "    value = alias['text']\n"
                  "    assert value.startswith('ok')\n")
        result = flow.analyze_saved_test(source, 1, "acme_app.app.f")
        self.assertEqual(result["status"], "ok")
        self.assertEqual([(c["call_line"], c["check_line"], c["evidence"])
                          for c in result["checks"]], [(2, 5, EVIDENCE)])

        cases = {
            "overwrite": "r = f()\n    r = 'replacement'\n    assert r",
            "receiver": "r = service.f()\n    assert service",
            "message": "r = f()\n    assert healthy, r",
            "argument": "r = f(request)\n    assert request",
            "terminal_walrus": "r = f()\n    return (x := 0)\n    assert r",
        }
        for name, body in cases.items():
            with self.subTest(name=name):
                result = flow.analyze_saved_test("def test_f():\n    " + body + "\n",
                                                 1, "acme_app.app.f")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["checks"], [])

    def test_resolved_immediate_controlled_failures_and_unrelated_predicate(self):
        cases = [
            ("", "raise AssertionError('bad')"),
            ("import pytest as pt\n", "pt.fail('bad')"),
            ("", "self.fail('bad')"),
        ]
        for prefix, failure in cases:
            with self.subTest(failure=failure):
                source = (prefix + "def test_f(self):\n    r = f()\n"
                          "    if not r:\n        " + failure + "\n")
                line = 2 if prefix else 1
                result = flow.analyze_saved_test(source, line, "acme_app.app.f")
                self.assertEqual(result["status"], "ok")
                self.assertEqual([(c["check_kind"], c["check_line"], c["failure_line"])
                                  for c in result["checks"]],
                                 [("controlled_failure", line + 2, line + 3)])
                self.assertEqual(result["checks"][0]["evidence"], EVIDENCE)
        source = ("def test_f():\n    r = f()\n    if unrelated:\n"
                  "        raise AssertionError('bad')\n")
        result = flow.analyze_saved_test(source, 1, "acme_app.app.f")
        self.assertEqual(result["checks"], [])

    def test_call_selection_abstains_on_ambiguity_and_accepts_explicit_import_alias(self):
        cases = [
            ("def test_f():\n    a = f()\n    b = other.f()\n    assert a\n",
             "ambiguous_call"),
            ("def test_f():\n    r = unrelated()\n    assert r\n", "no_matching_call"),
            ("def test_f():\n    def inner():\n        return f()\n    assert True\n",
             "no_matching_call"),
        ]
        for source, status in cases:
            with self.subTest(status=status, source=source):
                result = flow.analyze_saved_test(source, 1, "acme_app.app.f")
                self.assertEqual(result["status"], status)
                self.assertEqual(result["checks"], [])
        result = flow.analyze_saved_test(
            "from acme_app.app import f as invoke\ndef test_f():\n"
            "    r = invoke()\n    assert r\n", 2, "acme_app.app.f")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["checks"]), 1)

    def test_compound_reassignment_arbitrary_validator_and_caught_raise_abstain(self):
        cases = [
            "r = f()\n    if condition:\n        r = replacement\n    assert r",
            "r = f()\n    assert validate(r)",
            "r = f()\n    try:\n        if not r:\n"
            "            raise ValueError('bad')\n    except ValueError:\n        pass",
        ]
        for body in cases:
            with self.subTest(body=body):
                result = flow.analyze_saved_test("def test_f():\n    " + body + "\n",
                                                 1, "acme_app.app.f")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["checks"], [])

    @staticmethod
    def _write_rows(path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["repo", "qname", "file", "line", "reason", "kind"])
            writer.writeheader()
            writer.writerows(rows)

    def test_loader_uses_canonical_filter_without_focal_inputs(self):
        import pandas as pd
        from pipeline import paths
        import Applications
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "invokers.csv"
            self._write_rows(raw, [row(), row(), row(repo="outside/pop"),
                row(qname="wrapper", line=2, kind="transitive"),
                row(qname="sdk", line=3, reason="raw sdk"),
                row(qname="removed", line=4, reason="removed pattern")])
            seen = []
            def canonical_filter(frame):
                self.assertIsInstance(frame, pd.DataFrame)
                seen.append(True)
                return frame[frame["reason"] != "removed pattern"]
            fake = types.ModuleType("Applications.make_figures")
            fake.POP = {"acme/app"}
            fake.RAW_SDKS = {"openai"}
            fake.AZ = types.SimpleNamespace(
                drop_removed_patterns_invokers=canonical_filter,
                fw_from_reason=lambda reason: "openai" if reason == "raw sdk" else "framework")
            original_open = __import__("builtins").open
            def checked_open(file, *args, **kwargs):
                if isinstance(file, (str, Path)):
                    self.assertNotIn("pym2t", str(file).lower())
                    self.assertNotIn("method_test_baseline", str(file).lower())
                return original_open(file, *args, **kwargs)
            with mock.patch.object(paths, "LLM_INVOKERS_CSV", raw), \
                    mock.patch.dict(sys.modules, {"Applications.make_figures": fake}), \
                    mock.patch.object(Applications, "make_figures", fake, create=True), \
                    mock.patch("builtins.open", side_effect=checked_open):
                result = flow.load_direct_invokers()
            self.assertEqual(result, [row()])
            self.assertTrue(seen)

    def test_cli_retains_source_gaps_hashes_and_direct_test_frame_gap(self):
        from pipeline import paths
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            repos = root / "repos"
            clone = repos / "acme_app"
            clone.mkdir(parents=True)
            (clone / "app.py").write_text("def f():\n    return 'x'\n\n\n\n\n\ndef g():\n    return 'g'\n\n\ndef h():\n    return 'h'\n", encoding="utf-8")
            (clone / "tests.py").write_text(
                "def test_f():\n    r = f()\n    assert r\n    assert len(r) > 0\n", encoding="utf-8")
            (clone / "parse.py").write_text("def broken(:\n", encoding="utf-8")
            (clone / "empty.py").write_text("def renamed_test():\n    r = h()\n    assert r\n", encoding="utf-8")
            canonical = [row(), row(qname="acme_app.tests.helper", file="acme_app/tests/helpers.py"),
                row(repo="missing/clone", qname="missing_clone.app.f", file="missing_clone/app.py"),
                row(qname="acme_app.app.g", line=8), row(qname="acme_app.app.h", line=12)]
            tests = [row(qname="acme_app.tests.test_f", file="acme_app/tests.py", reason="calls acme_app.app.f", kind="transitive"),
                row(repo="missing/clone", qname="missing_clone.tests.test_f", file="missing_clone/tests.py",
                    reason="calls missing_clone.app.f", kind="transitive"),
                row(qname="acme_app.parse.test_g", file="acme_app/parse.py", reason="calls acme_app.app.g", kind="transitive"),
                row(qname="acme_app.empty.test_h", file="acme_app/empty.py", reason="calls acme_app.app.h", kind="transitive"),
                row(qname="acme_app.direct.test_direct", file="acme_app/direct.py", reason="matches framework")]
            tests.append(dict(tests[0]))
            inv_csv, test_csv = artifacts / "invokers.csv", artifacts / "tests.csv"
            self._write_rows(inv_csv, canonical)
            self._write_rows(test_csv, tests)
            original = {p: p.read_bytes() for p in (inv_csv, test_csv)}
            with mock.patch.object(paths, "ARTIFACTS_DIR", artifacts), \
                    mock.patch.object(paths, "LLM_INVOKERS_CSV", inv_csv), \
                    mock.patch.object(paths, "LLM_TESTS_CSV", test_csv), \
                    mock.patch.object(paths, "REPOS_DIR", repos), \
                    mock.patch.object(flow, "load_direct_invokers", return_value=canonical):
                output = root / "fresh"
                self.assertEqual(flow.main(["--repos-dir", str(repos), "--output-dir", str(output)]), 0)
                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["invokers"]["denominator"], 5)
                self.assertEqual(summary["invokers"]["with_saved_direct_test_edge"], 4)
                self.assertEqual(summary["invokers"]["with_result_check_candidate"], 1)
                self.assertEqual(summary["test_edges"]["selected"], 4)
                self.assertEqual(summary["test_edges"]["with_result_check_candidate"], 1)
                self.assertEqual(summary["test_edges"]["direct_test_rows_without_target_edge"], 1)
                self.assertEqual(summary["checks"]["total"], 2)
                self.assertEqual(sum(s["denominator"] for s in summary["by_stratum"].values()), 5)
                self.assertEqual(sum(s["with_result_check_candidate"]
                                     for s in summary["by_stratum"].values()), 1)
                with (output / "invokers.csv").open(newline="", encoding="utf-8") as stream:
                    retained = list(csv.DictReader(stream))
                self.assertEqual(len(retained), 5)
                self.assertEqual(sum(int(r["has_result_check_candidate"]) for r in retained), 1)
                with (output / "diagnostics.csv").open(newline="", encoding="utf-8") as stream:
                    self.assertTrue(list(csv.DictReader(stream)))
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                for path, content in original.items():
                    digest = hashlib.sha256(content).hexdigest()
                    self.assertTrue(any(entry["sha256_before"] == digest and entry["sha256_after"] == digest
                                        for entry in manifest["inputs"].values()))
                    self.assertEqual(path.read_bytes(), content)
                empty = root / "empty_subset"
                self.assertEqual(flow.main(["--output-dir", str(empty), "--repo", "absent/repo"]), 0)
                zero = json.loads((empty / "summary.json").read_text(encoding="utf-8"))
                self.assertTrue(zero["subset"])
                self.assertEqual(zero["invokers"]["denominator"], 0)
                self.assertEqual(zero["checks"]["total"], 0)

    def test_existing_output_is_refused_and_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep this", encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = flow.main(["--output-dir", str(output)])
            self.assertNotEqual(result, 0)
            self.assertIn(str(output), stdout.getvalue() + stderr.getvalue())
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep this")
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["keep.txt"])


if __name__ == "__main__":
    unittest.main()
