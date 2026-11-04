"""Synthetic contract tests for specs/response-guardrail-slices.spec.md.

Written independently of the implementation. No corpus or model execution.
"""
import contextlib
import csv
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from pipeline import response_guardrail_slices as subject


CALL_HEADERS = "call_id repo repo_slug rel_path code_kind framework pattern call_line enclosing_qname repo_sliced in_slice".split()
HIT_HEADERS = "slice_id call_id call_repo repo rel_path code_kind call_line variable role parses".split()
INDEX_HEADERS = "slice_id repo rel_path code_kind method variable parses".split()


def call(line=2, col=8, **changes):
    row = dict(call_id=f"acme_app/app.py::{line}::{col}", repo="acme/app",
               repo_slug="acme_app", rel_path="app.py", code_kind="production",
               framework="langchain", pattern=".invoke", call_line=line,
               call_col=col, enclosing_qname="acme_app.app.f", repo_sliced="1", in_slice="1")
    row.update(changes)
    return row


def write_csv(path, headers, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fixture(root, calls=None, slices=True):
    artifacts, repos = root / "artifacts", root / "repos"
    artifacts.mkdir()
    (repos / "acme_app").mkdir(parents=True)
    rows = [call()] if calls is None else calls
    write_csv(artifacts / "slice_calls_by_call.csv", CALL_HEADERS, rows)
    index = dict(slice_id=7, repo="acme_app", rel_path="app.py", code_kind="production",
                 method="f", variable="r", parses=1)
    hit = dict(index, call_id=call()["call_id"], call_repo="acme/app", call_line=2, role="binding")
    write_csv(artifacts / "slice_index.csv", INDEX_HEADERS, [index] if slices else [])
    write_csv(artifacts / "slice_call_hits.csv", HIT_HEADERS, [hit] if slices else [])
    write_csv(artifacts / "slice_pdg_lines.csv.gz", ["slice_id", "line"],
              [{"slice_id": 7, "line": 2}, {"slice_id": 7, "line": 3}] if slices else [])
    (repos / "acme_app" / "app.py").write_text(
        "def f():\n    r = llm.invoke(x)\n    assert r is not None\n", encoding="utf-8")
    return artifacts, repos


def run_cli(artifacts, repos, output, rows, extra=()):
    with patch.object(subject, "load_calls", return_value=rows), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return subject.main(["--artifacts-dir", str(artifacts), "--repos-dir", str(repos),
                             "--output-dir", str(output), *extra])


class ResponseGuardrailContract(unittest.TestCase):
    def test_assertions_follow_response_origins_and_ignore_constant_conditions(self):
        positive = "def f():\n    r = llm.invoke(x)\n    assert r is not None\n"
        result = subject.analyze_source(positive, [call()])[0]
        self.assertEqual((result["status"], result["output_form"], result["response_variables"]),
                         ("ok", "assigned", ["r"]))
        self.assertEqual(len(result["checks"]), 1)
        check = result["checks"][0]
        self.assertEqual({key: check[key] for key in
                          ("call_id", "call_line", "sink_line", "check_kind", "mechanism", "intervention", "resolution", "trace_lines")},
                         dict(call_id="acme_app/app.py::2::8", call_line=2, sink_line=3,
                              check_kind="assertion", mechanism="deterministic_assert", intervention="assertion_failure",
                              resolution="source_local", trace_lines=[2, 3]))
        for tail in ["assert llm", "assert x", "r = 'fixed'\n    assert r", "assert True, r",
                     "assert True or r", "assert r or True", "assert False and r", "assert r and False",
                     "assert [r]", "assert {'value': r}", "if r or True:\n        raise ValueError()",
                     "if False:\n        assert r"]:
            with self.subTest(tail=tail):
                row = subject.analyze_source("def f():\n    r = llm.invoke(x)\n    " + tail + "\n", [call()])[0]
                self.assertEqual(row["checks"], [])

        source = "def f():\n    r = llm.invoke(x)\n    s = llm.invoke(y)\n    assert r and s\n"
        seeds = [call(), call(3, 8)]
        rows = subject.analyze_source(source, [seeds[1], seeds[0], seeds[0]])
        self.assertEqual([r["call_id"] for r in rows],
                         ["acme_app/app.py::2::8", "acme_app/app.py::3::8"])
        self.assertEqual([[c["call_id"] for c in r["checks"]] for r in rows],
                         [["acme_app/app.py::2::8"], ["acme_app/app.py::3::8"]])

    def test_exact_ast_positions_chains_scopes_and_retained_failures(self):
        source = "r = llm.invoke(\n    x\n).strip()\nassert r\n"
        row = subject.analyze_source(source, [call(1, 4)])[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["checks"][0]["sink_line"], 4)
        source = "async def f():\n    r = await llm.invoke(\n        x\n    )\n    assert r\n"
        row = subject.analyze_source(source, [call(2, 14)])[0]
        self.assertEqual((row["status"], row["checks"][0]["trace_lines"]), ("ok", [2, 5]))
        self.assertEqual(subject.analyze_source("r = other(x)\n", [call(1, 4)])[0]["status"], "seed_missing")
        self.assertEqual(subject.analyze_source("def f(:\n", [call()])[0]["status"], "parse_error")
        source = "def f():\n    r = llm.invoke(x)\n    def inner():\n        assert r\n    return r\n    assert r\n"
        self.assertEqual(subject.analyze_source(source, [call()])[0]["checks"], [])
        for source, expected in [("def f():\n    return llm.invoke(x)\n", "returned_to_caller"),
                                 ("def f():\n    llm.invoke(x)\n", "discarded")]:
            with self.subTest(expected=expected):
                seed = call(2, 11 if expected == "returned_to_caller" else 4)
                self.assertEqual(subject.analyze_source(source, [seed])[0]["output_form"], expected)

    def test_schema_ports_parser_routes_and_unknown_validators(self):
        source = ("import json\nimport jsonschema as js\nfrom pydantic import BaseModel, TypeAdapter\n"
                  "class Shape(BaseModel):\n    value: str\n"
                  "def f():\n    r = llm.invoke(x)\n    parsed = json.loads(r)\n"
                  "    unknown.validate(r)\n    js.validate(r, {})\n    js.validate({}, r)\n"
                  "    Shape.model_validate(r)\n    adapter = TypeAdapter(Shape)\n"
                  "    adapter.validate_python(r)\n    assert parsed\n")
        row = subject.analyze_source(source, [call(7, 8)])[0]
        self.assertEqual(sorted((c["sink_line"], c["check_kind"]) for c in row["checks"]),
                         [(10, "schema_validation"), (12, "schema_validation"),
                          (14, "schema_validation"), (15, "assertion")])
        self.assertEqual(sorted(r["route_kind"] for r in row["routes"]), ["parser_only", "unresolved_validator"])
        for check in row["checks"]:
            if check["check_kind"] == "schema_validation":
                self.assertEqual((check["mechanism"], check["intervention"], check["resolution"]),
                                 ("schema_validator", "validator_exception_contract", "source_import"))

        prefix = ("from openevals.llm import create_llm_as_judge as make_judge\n"
                  "def f():\n    judge = make_judge(prompt='rubric', model='provider:model')\n"
                  "    r = llm.invoke(x)\n")
        for statement, expected in [
                ("judge(outputs=r)", ["evaluation_only"]),
                ("feedback = judge(outputs=r)", ["evaluation_flow_unassessed"]),
                ("judge(inputs=r, outputs='fixed')", []),
                ("judge = other\n    judge(outputs=r)", [])]:
            with self.subTest(judge_statement=statement):
                row = subject.analyze_source(prefix + "    " + statement + "\n", [call(4, 8)])[0]
                self.assertEqual(row["checks"], [])
                self.assertEqual([route["route_kind"] for route in row["routes"]
                                  if route["route_kind"].startswith("evaluation_")], expected)
                if expected:
                    self.assertEqual(row["routes"][0]["sink_line"], 5)

    def test_rejection_early_exit_and_loop_retry_have_distinct_interventions(self):
        for body, expected in [("if not r:\n        raise ValueError()", [("rejection_branch", "raise")]),
                               ("if not r:\n        return None", [("early_exit_candidate", "replacement_return")]),
                               ("if r:\n        return r", []), ("if r:\n        print(r)", [])]:
            with self.subTest(body=body):
                row = subject.analyze_source("def f():\n    r = llm.invoke(x)\n    " + body + "\n", [call()])[0]
                self.assertEqual([(c["check_kind"], c["intervention"]) for c in row["checks"]], expected)
        source = "def f():\n    for x in prompts:\n        r = llm.invoke(x)\n        if not r:\n            continue\n"
        row = subject.analyze_source(source, [call(3, 12)])[0]
        self.assertEqual([(c["check_kind"], c["intervention"]) for c in row["checks"]], [("retry_branch", "retry_continue")])

    def test_blocks_find_checks_but_uncertain_writes_abstain(self):
        for block in ["try:", "with context():", "for item in items:"]:
            source = "def f():\n    " + block + "\n        r = llm.invoke(x)\n        assert r\n"
            if block == "try:":
                source += "    except Exception:\n        pass\n"
            with self.subTest(block=block):
                row = subject.analyze_source(source, [call(3, 12)])[0]
                self.assertEqual(len(row["checks"]), 1)
        source = "def f():\n    r = llm.invoke(x)\n    if flag:\n        r = 'fixed'\n    assert r\n"
        row = subject.analyze_source(source, [call()])[0]
        self.assertEqual(row["checks"], [])
        self.assertTrue(row["limitations"], "An uncertain merge must retain explicit unassessed evidence")

        abstentions = [
            "def f():\n    r = llm.invoke(x)\n    for item in items:\n        if not r:\n            continue\n",
            "def f():\n    r = llm.invoke(x)\n    if flag:\n        r.clear()\n    assert r\n",
            "def f():\n    r = llm.invoke(x)\n    try:\n        risky()\n    except Exception:\n        r.clear()\n    assert r\n",
            "def f():\n    r = llm.invoke(x)\n    if True:\n        return None\n    assert r\n",
            "def f():\n    r = llm.invoke(x)\n    len = lambda _: True\n    assert len(r)\n",
        ]
        for source in abstentions:
            with self.subTest(abstention=source):
                self.assertEqual(subject.analyze_source(source, [call()])[0]["checks"], [])
        source = ("def f():\n    for prompt in prompts:\n        r = llm.invoke(prompt)\n"
                  "        for item in items:\n            if not r:\n                continue\n")
        self.assertEqual(subject.analyze_source(source, [call(3, 12)])[0]["checks"], [])
        source = "len = lambda _: True\nr = llm.invoke(x)\nassert len(r)\n"
        self.assertEqual(subject.analyze_source(source, [call(2, 4)])[0]["checks"], [])

    def test_slice_roles_and_duplicate_order_do_not_inflate_corroboration(self):
        check = dict(call_id=call()["call_id"], call_line=2, sink_line=3, response_variables=["r"])
        binding = dict(slice_id=7, call_id=call()["call_id"], variable="r", role="binding", parses=0)
        receiver = dict(binding, slice_id=8, variable="llm", role="receiver")
        for hits in itertools.permutations([binding, receiver, dict(binding)]):
            result = subject.corroborate_slices([check], list(hits), {7: {2, 3}, 8: {2, 3}})[0]
            self.assertEqual(result["joern_status"], "same_response_slice")
            self.assertEqual(result["slice_ids"], [7, 8])
            self.assertEqual(result["shared_slice_ids"], [7, 8])
            self.assertEqual(result["response_slice_ids"], [7])
            self.assertTrue(result["slice_parse_flags"])
        for hit in [receiver, dict(binding, role="argument")]:
            result = subject.corroborate_slices([check], [hit], {7: {2, 3}, 8: {2, 3}})[0]
            self.assertEqual((result["joern_status"], result["response_slice_ids"]), ("shared_slice_unresolved_role", []))

    def test_same_line_missing_and_nonjoint_slices_remain_source_only(self):
        check = dict(call_id=call()["call_id"], call_line=2, sink_line=3, response_variables=["r"])
        hit = dict(slice_id=7, call_id=call()["call_id"], variable="r", role="binding", parses=1)
        for checks, hits, pdg, status in [([dict(check, sink_line=2)], [hit], {7: {2}}, "same_line_source_only"),
                                          ([check], [hit], {7: {2}}, "no_joint_slice"),
                                          ([check], [], {}, "no_saved_slice")]:
            result = subject.corroborate_slices(checks, hits, pdg)[0]
            self.assertEqual((result["joern_status"], result["response_slice_ids"]), (status, []))

    def test_cli_four_table_fixture_counts_unique_calls_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = call(1, 4, call_id="acme_app/missing.py::1::4", rel_path="missing.py", code_kind="test")
            artifacts, repos = fixture(root, [call(), missing])
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*artifacts.iterdir(), repos / "acme_app" / "app.py"]}
            output = root / "result"
            self.assertEqual(run_cli(artifacts, repos, output, [call(), call(), missing]), 0)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual({k: summary["calls"][k] for k in ["denominator", "with_source_check", "with_response_slice_check", "with_runtime_intervention"]},
                             dict(denominator=2, with_source_check=1, with_response_slice_check=1, with_runtime_intervention=1))
            self.assertEqual(summary["checks"]["total"], 1)
            self.assertEqual(summary["by_code_kind"]["test"]["denominator"], 1)
            for name in ["calls.csv", "checks.csv", "routes.csv", "diagnostics.csv"]:
                with (output / name).open(encoding="utf-8-sig", newline="") as stream:
                    self.assertTrue(csv.DictReader(stream).fieldnames)
            self.assertEqual({p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}, before)
            manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            self.assertEqual(set(manifest["inputs"]),
                             {"slice_calls_by_call.csv", "slice_call_hits.csv", "slice_index.csv", "slice_pdg_lines.csv.gz"})
            for name, recorded in manifest["inputs"].items():
                self.assertEqual(recorded["sha256_before"], before[artifacts / name])
                self.assertEqual(recorded["sha256_after"], before[artifacts / name])
            self.assertEqual(list(manifest["inspected_sources"].values()),
                             [before[repos / "acme_app" / "app.py"]])
            generated = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in output.iterdir() if p.name != "manifest.json" and p.is_file()}
            self.assertEqual(manifest["outputs"], generated)
            self.assertTrue(manifest["revision"])
            self.assertTrue(manifest["python"])
            self.assertIn("options", manifest)
            self.assertTrue((output / "REPORT.md").read_text(encoding="utf-8"))

    def test_cli_existing_output_refuses_before_loading_and_empty_subset_keeps_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, repos = fixture(root, slices=False)
            existing = root / "existing"
            existing.mkdir()
            sentinel = existing / "keep.txt"
            sentinel.write_text("untouched", encoding="utf-8")
            captured = io.StringIO()
            with patch.object(subject, "load_calls", side_effect=AssertionError("loader must not run")), contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                try:
                    result = subject.main(["--output-dir", str(existing)])
                except SystemExit as error:
                    result = error.code
            self.assertNotEqual(result, 0)
            self.assertIn("exists", captured.getvalue().lower())
            self.assertIn(str(existing), captured.getvalue())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")
            output = root / "empty"
            self.assertEqual(run_cli(artifacts, repos, output, [call()], ["--repo", "other/repo"]), 0)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertIs(summary["subset"], True)
            self.assertEqual(summary["calls"]["denominator"], 0)
            self.assertEqual(summary["checks"]["total"], 0)
            for name in ["calls.csv", "checks.csv", "routes.csv", "diagnostics.csv"]:
                with (output / name).open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertTrue(reader.fieldnames)
                    self.assertEqual(list(reader), [])

    def test_canonical_loader_uses_fake_mf_snapshot_and_original_strata(self):
        import pandas as pd
        fake_parent = types.ModuleType("Applications")
        fake_parent.__path__ = []
        fake_mf = types.ModuleType("Applications.make_figures")
        fake_parent.make_figures = fake_mf
        saved = [call(code_kind="test_support"), call(4, 12, code_kind="test")]
        fake_mf.CALLS = pd.DataFrame(saved)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, _ = fixture(root, saved)
            # MF itself is fake; these files isolate its documented saved-input
            # hash boundary without accessing the user's actual canonical tables.
            write_csv(artifacts / "llm_calls_all.csv", CALL_HEADERS, saved)
            write_csv(artifacts / "application_audit.csv", ["repo", "in_scope"],
                      [{"repo": "acme/app", "in_scope": ""}])
            progress = artifacts / ".batch_progress.json"
            progress.write_text("{}", encoding="utf-8")
            with patch.dict(sys.modules, {"Applications": fake_parent, "Applications.make_figures": fake_mf}), \
                    patch("pipeline.paths.ARTIFACTS_DIR", artifacts), \
                    patch("pipeline.paths.BATCH_PROGRESS_JSON", progress, create=True):
                rows = subject.load_calls(artifacts)
                self.assertEqual(sorted((r["call_id"], r["call_line"], r["call_col"], r["code_kind"]) for r in rows),
                                 [("acme_app/app.py::2::8", 2, 8, "test_support"), ("acme_app/app.py::4::12", 4, 12, "test")])
                write_csv(artifacts / "slice_calls_by_call.csv", CALL_HEADERS, [saved[0]])
                with self.assertRaises(Exception) as raised:
                    subject.load_calls(artifacts)
                self.assertTrue(str(raised.exception), "Snapshot mismatch must explain the failure")


if __name__ == "__main__":
    unittest.main()
