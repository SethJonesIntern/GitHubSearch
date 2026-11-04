"""Focused independent contract tests with synthetic fixtures only."""
import contextlib
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.invoker_result_checks import analyze_local_checks, main, normalized_relative


class PathTests(unittest.TestCase):
    def test_normalizes_exact_slug_without_suffix_matching(self):
        self.assertEqual(normalized_relative('acme_app\\pkg\\a.py', 'acme_app'), 'pkg/a.py')
        self.assertEqual(normalized_relative('other/acme_app/a.py', 'acme_app'), 'other/acme_app/a.py')

    def test_rejects_unsafe_paths(self):
        for path in ('../a.py', 'C:/tmp/a.py', ''):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, '^Unsafe source path$'):
                    normalized_relative(path, 'acme_app')


class LocalChecksTests(unittest.TestCase):
    def test_result_assignment_has_literal_local_evidence(self):
        result = analyze_local_checks('def f():\n r = agent.invoke(x)\n assert r is not None\n', 1, {2})
        self.assertEqual(result, {
            'status': 'ok', 'detail': '', 'limitations': [],
            'checks': [{'call_line': 2, 'check_line': 3, 'check_kind': 'assert',
                        'check_source': 'assert r is not None', 'trace_lines': [2, 3],
                        'variables': ['r'], 'evidence': 'syntactic_local_result_to_assert_candidate'}],
        })

    def test_alias_and_builtin_value_keep_the_local_trace(self):
        result = analyze_local_checks(
            'def f():\n r = agent.invoke(x)\n s = r.content\n assert len(s) > 0\n', 1, {2})
        self.assertEqual(len(result['checks']), 1)
        self.assertEqual(result['checks'][0]['trace_lines'], [2, 3, 4])
        self.assertIn('Builtin names may be shadowed; calls are not resolved.', result['limitations'])

    def test_receiver_and_assertion_messages_are_not_result_checks(self):
        result = analyze_local_checks(
            'def f(self):\n r = agent.invoke(x)\n assert agent is not None\n assert True, r\n self.assertEqual(1, 1, msg=r)\n',
            1, {2})
        self.assertEqual(result['checks'], [])

    def test_unittest_value_assertion_is_a_candidate(self):
        result = analyze_local_checks(
            'def f(self):\n r = agent.invoke(x)\n self.assertEqual(r, "yes")\n', 1, {2})
        self.assertEqual(len(result['checks']), 1)
        self.assertEqual(result['checks'][0]['check_kind'], 'unittest_assert')
        self.assertEqual(result['checks'][0]['check_line'], 3)

    def test_predicate_association_comes_from_receiver(self):
        result = analyze_local_checks(
            'def f():\n r = agent.invoke(x)\n assert r.content.startswith("yes")\n assert "fixed".startswith(r)\n',
            1, {2})
        self.assertEqual(len(result['checks']), 1)
        self.assertEqual(result['checks'][0]['check_line'], 3)
        self.assertTrue(result['limitations'])

    def test_unsupported_walrus_cannot_leave_stale_or_unreachable_checks(self):
        result = analyze_local_checks(
            'def f():\n r = agent.invoke(x)\n s = (r := 0)\n assert r\n', 1, {2})
        self.assertEqual(result['checks'], [])
        self.assertTrue(result['limitations'])
        unreachable = analyze_local_checks(
            'def f():\n r = agent.invoke(x)\n return (other := 0)\n assert r\n', 1, {2})
        self.assertEqual(unreachable['checks'], [])

    def test_unsupported_assignment_to_other_name_preserves_result(self):
        result = analyze_local_checks(
            'def f():\n r = agent.invoke(x)\n other = (s := 0)\n assert r\n', 1, {2})
        self.assertEqual(len(result['checks']), 1)
        self.assertEqual(result['checks'][0]['check_line'], 4)
        self.assertTrue(result['limitations'])

    def test_unsupported_match_capture_invalidates_bound_result(self):
        result = analyze_local_checks(
            'def f():\n r = agent.invoke(x)\n match value:\n  case {"item": r}:\n   pass\n assert r\n',
            1, {2})
        self.assertEqual(result['checks'], [])
        self.assertTrue(result['limitations'])

    def test_nested_definition_bindings_invalidate_result(self):
        for definition in (' def r():\n  pass\n', ' class r:\n  pass\n'):
            with self.subTest(definition=definition):
                result = analyze_local_checks('def f():\n r = agent.invoke(x)\n' + definition + ' assert r\n', 1, {2})
                self.assertEqual(result['checks'], [])

HEADERS = {
    'method_test_baseline/methods.csv': 'repo,file,qname,line,line_end,group,mapped_tests,exact_mapped_tests',
    'method_test_baseline/by_repo.csv': 'repo,llm_methods,llm_mapped,llm_exact_mapped',
    'pym2t_focal_tests.csv': 'repo,test_file,test_method,test_line,test_line_end,focal_file,file_rule,focal_method,focal_line,focal_line_end,method_rule',
    'pym2t_focal_calls.csv': 'repo,call_id,file,enclosing_qname,framework,call_line',
    'slice_index.csv': 'slice_id,repo,rel_path,code_kind,method,variable,parses',
    'slice_call_hits.csv': 'slice_id,repo,rel_path,code_kind,method,variable,call_id,call_repo,call_code_kind,call_line',
    'slice_pdg_lines.csv.gz': 'slice_id,line',
}


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / 'artifacts'
        self.repos = self.root / 'repos'
        self.repos.mkdir()
        self.rows = {name: [] for name in HEADERS}
        self.rows['method_test_baseline/methods.csv'] = [['acme/app', 'app.py', 'f', 1, 3, 'llm', 0, 0]]
        self.rows['method_test_baseline/by_repo.csv'] = [['acme/app', 1, 0, 0]]
        self.rows['pym2t_focal_calls.csv'] = [['acme/app', 'acme_app/app.py::2::8', 'acme_app/app.py', 'acme_app.app.f', 'langchain', 2]]
        self.rows['slice_index.csv'] = [[7, 'acme_app', 'app.py', 'production', 'f', 'r', 1]]
        self.rows['slice_call_hits.csv'] = [[7, 'acme_app', 'app.py', 'production', 'f', 'r', 'acme_app/app.py::2::8', 'acme/app', 'production', 2]]
        self.rows['slice_pdg_lines.csv.gz'] = [[7, 2], [7, 3]]
        self.summary = {'pooled_all_complete_repos': {'llm': {'methods': 1, 'with_mapped_test': 0, 'with_exact_test': 0}}}
        self.source('app.py', 'def f():\n    r = agent.invoke("hello")\n    assert r is not None\n')

    def source(self, relative, content):
        path = self.repos / 'acme_app' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

    def save(self):
        for name, header in HEADERS.items():
            path = self.artifacts / name
            path.parent.mkdir(parents=True, exist_ok=True)
            opener = gzip.open if name.endswith('.gz') else open
            with opener(path, 'wt', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(header.split(','))
                writer.writerows(self.rows[name])
        (self.artifacts / 'method_test_baseline/summary.json').write_text(json.dumps(self.summary), encoding='utf-8')

    def run_cli(self, output='result', extra=()):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            code = main(['--artifacts-dir', str(self.artifacts), '--repos-dir', str(self.repos),
                         '--output-dir', str(self.root / output), *extra])
        return code, captured.getvalue()

    def read_csv(self, name, output='result'):
        with (self.root / output / name).open(newline='', encoding='utf-8-sig') as stream:
            return list(csv.DictReader(stream))

    def read_json(self, name, output='result'):
        return json.loads((self.root / output / name).read_text(encoding='utf-8'))

    def add_mapping(self, line=10, rule='exact', file_rule='import'):
        self.rows['method_test_baseline/methods.csv'][0][-2:] = [1, 1 if rule == 'exact' else 0]
        self.rows['method_test_baseline/by_repo.csv'][0][-2:] = [1, 1 if rule == 'exact' else 0]
        self.summary['pooled_all_complete_repos']['llm'].update(with_mapped_test=1, with_exact_test=1 if rule == 'exact' else 0)
        self.rows['pym2t_focal_tests.csv'].append(['acme/app', 'acme_app/test_app.py', 'test_x', line, line + 2,
                                                 'acme_app/app.py', file_rule, 'f', 1, 3, rule])
        self.source('test_app.py', '\n' * 9 + 'def test_x():\n    result = f()\n    assert result\n' + '\n' * 7 +
                    'def test_x():\n    result = f()\n    assert result\n')

    def test_shared_slice_literal_fixture(self):
        self.save()
        code, message = self.run_cli()
        self.assertEqual(code, 0, message)
        checks = self.read_csv('checks.csv')
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]['side'], 'production')
        self.assertEqual(checks[0]['call_line'], '2')
        self.assertEqual(checks[0]['check_line'], '3')
        self.assertEqual(checks[0]['joern_status'], 'same_slice_candidate')
        self.assertEqual(json.loads(checks[0]['slice_ids']), [7])
        self.assertEqual(json.loads(checks[0]['slice_variables']), ['r'])
        self.assertEqual(self.read_json('summary.json')['methods']['with_shared_slice_candidate'], 1)

    def test_slice_without_check_preserves_ast_candidate(self):
        self.rows['slice_pdg_lines.csv.gz'] = [[7, 2]]
        self.save()
        code, message = self.run_cli()
        self.assertEqual(code, 0, message)
        self.assertEqual(self.read_csv('checks.csv')[0]['joern_status'], 'not_in_same_saved_slice')
        summary = self.read_json('summary.json')
        self.assertEqual(summary['methods']['denominator'], 1)
        self.assertEqual(summary['checks']['production'], 1)
        self.assertEqual(summary['methods']['with_shared_slice_candidate'], 0)

    def test_duplicate_links_deduplicate_but_distinct_definitions_remain(self):
        self.add_mapping(10)
        self.add_mapping(20)
        self.rows['pym2t_focal_tests.csv'] *= 2
        self.save()
        code, message = self.run_cli()
        self.assertEqual(code, 0, message)
        links = self.read_csv('test_links.csv')
        self.assertEqual(sorted(row['test_line'] for row in links), ['10', '20'])
        self.assertEqual({row['callee_resolution'] for row in links}, {'name_only'})
        summary = self.read_json('summary.json')
        self.assertEqual(summary['methods']['denominator'], 1)
        self.assertEqual(summary['methods']['with_test_candidate'], 1)
        self.assertEqual(summary['checks']['test'], 2)

    def test_file_association_uses_same_population_without_focal_method(self):
        self.rows['pym2t_focal_tests.csv'] = [
            ['acme/app', 'test_app.py', 'test_x', 10, 12, 'app.py', 'import', '', '', '', '']]
        self.save()
        code, message = self.run_cli()
        self.assertEqual(code, 0, message)
        row = self.read_csv('methods.csv')[0]
        self.assertEqual(row['mapped_test'], '0')
        self.assertEqual(row['file_associated'], '1')
        self.assertEqual(row['import_file_associated'], '1')
        summary = self.read_json('summary.json')['methods']
        self.assertEqual(summary['denominator'], 1)
        self.assertEqual(summary['with_mapped_test'], 0)
        self.assertEqual(summary['with_file_association'], 1)

    def test_missing_source_keeps_denominator_unassessed(self):
        (self.repos / 'acme_app/app.py').unlink()
        self.save()
        code, message = self.run_cli()
        self.assertEqual(code, 0, message)
        self.assertIn('source_missing', [row['kind'] for row in self.read_csv('diagnostics.csv')])
        summary = self.read_json('summary.json')
        self.assertEqual(summary['methods']['denominator'], 1)
        self.assertEqual(summary['assessment']['production_source_ok'], 0)

    def test_manifest_hashes_and_original_inputs_are_preserved(self):
        self.save()
        before = {path.relative_to(self.artifacts).as_posix(): path.read_bytes()
                  for path in self.artifacts.rglob('*') if path.is_file()}
        original_source = (self.repos / 'acme_app/app.py').read_bytes()
        code, message = self.run_cli()
        self.assertEqual(code, 0, message)
        inputs = self.read_json('manifest.json')['inputs']
        self.assertEqual(set(inputs), set(HEADERS) | {'method_test_baseline/summary.json'})
        for name, content in before.items():
            expected = hashlib.sha256(content).hexdigest()
            self.assertEqual(inputs[name]['sha256_before'], expected)
            self.assertEqual(inputs[name]['sha256_after'], expected)
            self.assertEqual((self.artifacts / name).read_bytes(), content)
        self.assertEqual((self.repos / 'acme_app/app.py').read_bytes(), original_source)

    def test_existing_output_directory_is_never_modified(self):
        output = self.root / 'existing'
        output.mkdir()
        sentinel = output / 'keep.txt'
        sentinel.write_bytes(b'existing output\x00')
        code, message = self.run_cli(output='existing')
        self.assertNotEqual(code, 0)
        self.assertIn('exists', message.lower())
        self.assertEqual(sentinel.read_bytes(), b'existing output\x00')
        self.assertEqual(sorted(path.name for path in output.iterdir()), ['keep.txt'])

    def test_missing_syntactic_callee_retains_mapping(self):
        self.add_mapping()
        self.source('test_app.py', '\n' * 9 + 'def test_x():\n    result = another()\n    assert result\n')
        self.save()
        code, message = self.run_cli()
        self.assertEqual(code, 0, message)
        links = self.read_csv('test_links.csv')
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]['matched_call_count'], '0')
        self.assertEqual(links[0]['candidate_count'], '0')
        self.assertTrue(self.read_csv('diagnostics.csv'))
        self.assertEqual(self.read_json('summary.json')['methods']['with_mapped_test'], 1)

    def test_one_line_production_definition_retains_saved_body_call(self):
        self.source('app.py', 'def f(): r = agent.invoke("hello"); assert r is not None\n')
        self.rows['method_test_baseline/methods.csv'][0][4] = 1
        self.rows['pym2t_focal_calls.csv'] = [
            ['acme/app', 'acme_app/app.py::1::13', 'acme_app/app.py', 'acme_app.app.f', 'langchain', 1]]
        self.rows['slice_index.csv'] = []
        self.rows['slice_call_hits.csv'] = []
        self.rows['slice_pdg_lines.csv.gz'] = []
        self.save()
        code, message = self.run_cli()
        self.assertEqual(code, 0, message)
        checks = self.read_csv('checks.csv')
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]['call_line'], '1')
        self.assertEqual(checks[0]['check_line'], '1')
        self.assertEqual(checks[0]['joern_status'], 'no_saved_slice')
        self.assertEqual(self.read_json('summary.json')['methods']['with_production_candidate'], 1)


if __name__ == '__main__':
    unittest.main()
