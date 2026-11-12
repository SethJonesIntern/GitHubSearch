"""Focused discovery-queue tests; all disk fixtures are fresh temporary files."""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import guardrail_test_review_queue as subject


def saved_call(**updates):
    row = dict(call_id="acme_app/tests/test_quality.py::3::8", repo="acme/app",
               rel_path="tests/test_quality.py", code_kind="test", call_line=3,
               framework="langchain", pattern=".invoke", enclosing_qname="test_quality")
    row.update(updates)
    return row


def source_result(call=None, **updates):
    call = call or saved_call()
    row = dict(call_id=call["call_id"], source_test_context=["test_named_function:test_quality"],
               test_identity=dict(repo="acme/app", test_file="tests/test_quality.py",
                                  test_method="test_quality", test_line=2, test_line_end=5),
               original_source_lines=["from rich.panel import Panel as Display", "def test_quality():",
                                      "    r = llm.invoke(x)", "    Display(r.grade)", "    assert r.grade"],
               source_sha256="a" * 64, checks=[], limitations=["unknown callback"],
               routes=[dict(call_id=call["call_id"], sink_line=5,
                            route_kind="uncertain_flow_assertion_candidate",
                            sink_source="assert r.grade", trace_lines=[3, 4, 5],
                            uncertainty_sources=["unknown callback:4"])])
    row.update(updates)
    return row


class GuardrailTestReviewQueueTests(unittest.TestCase):
    def test_unmapped_uncertain_assertion_is_pending_and_inputs_are_unchanged(self):
        calls, results = [saved_call()], [source_result()]
        before = copy.deepcopy((calls, results))
        candidates = subject.build_review_candidates(calls, results)
        self.assertEqual(len(candidates), 1)
        row = candidates[0]
        self.assertEqual(row["review_status"], "pending")
        self.assertEqual(row["evidence_origin"], "uncertain_route")
        self.assertIsNone(row["has_saved_exact_test_identity_mapping"])
        self.assertEqual(row["trace_lines"], [3, 4, 5])
        self.assertEqual(row["original_trace"][-1], {"line": 5, "text": "    assert r.grade"})
        self.assertEqual(row["source_sha256"], "a" * 64)
        self.assertEqual((calls, results), before)
        self.assertEqual(subject.build_review_candidates(calls, results, []),
                         [dict(row, has_saved_exact_test_identity_mapping=False)])

    def test_mapping_requires_exact_test_identity_and_never_assigns_producer(self):
        identity = source_result()["test_identity"]
        mismatches = [dict(identity, test_file="other/test_quality.py"),
                      dict(identity, test_line=99), dict(identity, repo="other/app"),
                      dict(identity, test_method="test_other")]
        for mapping in mismatches:
            with self.subTest(mapping=mapping):
                row = subject.build_review_candidates([saved_call()], [source_result()], [mapping])[0]
                self.assertIs(row["has_saved_exact_test_identity_mapping"], False)
        exact = dict(identity, test_file="acme_app\\tests\\test_quality.py",
                     method_key="incorrect_producer", focal_method="wrong_focal_method")
        row = subject.build_review_candidates([saved_call()], [source_result()], [exact])[0]
        self.assertIs(row["has_saved_exact_test_identity_mapping"], True)
        self.assertEqual(row["review_status"], "pending")
        self.assertNotIn("producer_method_id", row)
        self.assertNotIn("method_key", row)
        self.assertNotIn("oracle_type", row)
        self.assertEqual(row["call_id"], saved_call()["call_id"])
        self.assertEqual(row["attribution"], "protected_production_method_not_assigned")

    def test_source_test_context_retains_saved_production_label(self):
        call = saved_call(code_kind="production")
        result = source_result(call)
        candidates = subject.build_review_candidates([call], [result], [])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["code_kind"], "production")
        self.assertTrue(candidates[0]["source_test_context"])
        self.assertEqual(subject.build_review_candidates(
            [call], [dict(result, source_test_context=[], test_identity=None)], []), [])

    def test_trace_numbers_must_be_original_source_line_numbers(self):
        for trace in [[0], [6], [-1], [True], [2.5]]:
            result = source_result()
            result["routes"][0]["trace_lines"] = trace
            with self.subTest(trace=trace), self.assertRaisesRegex(ValueError, "original trace bounds"):
                subject.build_review_candidates([saved_call()], [result])

    def test_cli_writes_fresh_json_and_preserves_supplied_csv_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repos = root / "repos"
            source = repos / "acme_app" / "tests" / "test_quality.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(("from rich.panel import Panel as Display\r\n"
                                "def test_quality():\r\n    r = llm.invoke(x)\r\n"
                                "    Display(r.grade)\r\n    assert r.grade\r\n").encode())
            calls = root / "calls.csv"
            with calls.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(saved_call()))
                writer.writeheader()
                writer.writerow(saved_call(code_kind="production"))
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [calls, source]}
            output = root / "pending.json"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = subject.main(["--calls-csv", str(calls), "--repos-dir", str(repos),
                                       "--output", str(output)])
            self.assertEqual(status, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["review_status"], "pending")
            self.assertEqual(len(payload["candidates"]), 1)
            candidate = payload["candidates"][0]
            self.assertIsNone(candidate["has_saved_exact_test_identity_mapping"])
            self.assertEqual(candidate["test_identity"]["test_line"], 2)
            self.assertEqual(candidate["source_sha256"], before[source])
            self.assertEqual(candidate["code_kind"], "production")
            self.assertEqual(candidate["original_trace"][-1]["text"], "    assert r.grade")
            self.assertEqual({p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}, before)
            consumed = {row["path"]: row for row in payload["consumed_inputs"]}
            for path, digest in before.items():
                self.assertEqual(consumed[str(path.absolute())]["sha256_before"], digest)
                self.assertEqual(consumed[str(path.absolute())]["sha256_after"], digest)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["calls.csv", "pending.json", "repos"])

    def test_cli_refuses_existing_or_non_json_output_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing.json"
            existing.write_bytes(b"keep these exact bytes")
            for output in [existing, root / "new.csv"]:
                with self.subTest(output=output), \
                        patch.object(subject, "_read_csv", side_effect=AssertionError("must refuse before inputs")), \
                        patch.object(subject.detector, "_hash", side_effect=AssertionError("must refuse before hashes")), \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(subject.main([
                        "--calls-csv", str(root / "missing.csv"), "--repos-dir", str(root / "missing"),
                        "--mapped-edges-csv", str(root / "missing-mappings.csv"),
                        "--output", str(output)]), 1)
            self.assertEqual(existing.read_bytes(), b"keep these exact bytes")
            self.assertFalse((root / "new.csv").exists())


if __name__ == "__main__":
    unittest.main()
