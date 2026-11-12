import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import {
  parseCsv, normalizeRel, methodKey, buildWorklists, aggregateAnnotations, runAudit,
} from './guardrail_prevalence_audit.mjs';

const repo = 'acme/app';
const aId = `${repo}::app.py::1`;
const bId = `${repo}::app.py::8`;
const testId = `${repo}::tests/test_app.py::test_produce::1`;
const callId = 'acme_app/app.py::3::8';
const digest = text => createHash('sha256').update(text).digest('hex');
const csv = (headers, rows) => [headers, ...rows.map(row => headers.map(h => row[h] ?? ''))]
  .map(values => values.map(v => `"${String(v).replaceAll('"', '""')}"`).join(','))
  .join('\r\n') + '\r\n';

function fixture() {
  const methods = [
    { producer_method_id: aId, repo, rel_path: 'app.py', qname: 'produce', line: '1', line_end: '6',
      mapped_tests: '1', exact_mapped_tests: '1', runtime_simple: 'yes', test_complex: 'yes' },
    { producer_method_id: bId, repo, rel_path: 'app.py', qname: 'other', line: '8', line_end: '12',
      mapped_tests: '0', exact_mapped_tests: '0' },
  ];
  const mappings = [{ repo, test_file: 'acme_app/tests/test_app.py', test_method: 'test_produce',
    test_line: '1', test_line_end: '5', focal_file: 'acme_app/app.py', focal_method: 'produce',
    focal_line: '1', focal_line_end: '6', file_rule: 'import', method_rule: 'exact', method_key: aId }];
  const calls = [{ call_id: callId, repo, rel_path: 'app.py', code_kind: 'production', call_line: '3' }];
  const checks = [{ call_id: callId, repo, rel_path: 'app.py', code_kind: 'production', call_line: '3',
    sink_line: '4', check_kind: 'assertion', mechanism: 'deterministic_assert',
    intervention: 'assertion_failure', sink_source: 'assert r', joern_status: 'same_response_slice',
    slice_ids: '[7,8]', response_slice_ids: '[7]' }];
  return { methods, mappings, calls, checks };
}

function annotation(overrides = {}) {
  return { guardrail_id: 'guard-1', arm: 'runtime', producer_method_id: aId, test_id: '',
    output_call_id: callId, decision: 'guardrail', oracle_type: 'simple',
    checked_property: 'response is present', failure_action: 'assertion_failure',
    source_association: 'verified', reviewer: 'independent synthetic reviewer',
    rationale: 'Output binding is asserted before the return.',
    source_evidence: JSON.stringify(['output', 'checker', 'enforcement'].map((role, i) => ({
      rel_path: 'app.py', line: 3 + i, line_end: 3 + i, sha256: 'a'.repeat(64), role,
    }))), ...overrides };
}

const independent = overrides => ({ repo, invoker_file: 'acme_app/app.py', invoker_line: '1',
  test_file: 'acme_app/tests/test_app.py', test_line: '1',
  test_qname: 'acme_app.tests.test_app.test_produce', call_line: '3', check_line: '4',
  failure_line: '', check_kind: 'assert', evidence: 'saved target plus source-local candidate',
  call_source: 'r = produce()', check_source: 'assert r', ...overrides });

test('CSV supports BOM, quoted Unicode, escaped quotes, and multiline evidence', () => {
  const rows = parseCsv('\ufeffname,note\r\n"Seth","first, line\nsecond ""quoted"" line ü"\r\n');
  assert.deepEqual(rows, [{ name: 'Seth', note: 'first, line\nsecond "quoted" line ü' }]);
  assert.deepEqual(parseCsv('name,note\n'), []);
  assert.throws(() => parseCsv('a,b\n"unterminated,x'));
  assert.throws(() => parseCsv('a,b\n1,2,3\n'));
});

test('source identities normalize canonical repo prefixes but reject unsafe paths', () => {
  assert.equal(normalizeRel('acme_app\\src\\app.py', repo), 'src/app.py');
  assert.equal(normalizeRel('src/ü space.py', repo), 'src/ü space.py');
  assert.equal(methodKey(repo, 'acme_app/app.py', '1'), aId);
  for (const file of ['../secret.py', 'src/../../secret.py', 'C:/secret.py', '/secret.py',
    '\\\\server\\share\\x.py', 'src/file.py:stream', 'src/./../x.py']) {
    assert.throws(() => normalizeRel(file, repo), undefined, file);
  }
});

test('worklists preserve mappings without candidates and only join exact saved target identities', () => {
  const f = fixture();
  const lists = buildWorklists({ ...f, independentChecks: [independent(), independent(),
    independent({ test_line: '2' }), independent({ invoker_line: '2' })] });
  assert.equal(lists.runtime.length, 1);
  assert.equal(lists.runtime[0].producer_method_id, aId);
  assert.equal(lists.runtime[0].review_status, 'unassessed');
  assert.equal(lists.runtime[0].joern_status, 'same_response_slice');
  assert.ok(lists.testSuite.some(row => row.producer_method_id === aId && row.test_id === testId));
  assert.ok(lists.diagnostics.length >= 2, 'Unjoined saved target/test identities remain diagnostic.');
  const emptyCandidates = buildWorklists({ ...f, checks: [], independentChecks: [] });
  assert.equal(emptyCandidates.testSuite.length, 1);
  assert.equal(emptyCandidates.testSuite[0].test_id, testId);
  const reordered = buildWorklists({ ...f, methods: [...f.methods].reverse(),
    calls: [...f.calls, ...f.calls], checks: [...f.checks, ...f.checks], independentChecks: [independent()] });
  assert.deepEqual(lists.runtime.map(r => r.candidate_id).sort(), reordered.runtime.map(r => r.candidate_id).sort());
});

test('runtime attribution chooses the innermost method and rejects identity conflicts', () => {
  const f = fixture();
  const innerId = `${repo}::app.py::2`;
  const inner = { ...f.methods[0], producer_method_id: innerId, qname: 'produce.inner', line: '2', line_end: '5',
    mapped_tests: '0', exact_mapped_tests: '0' };
  const lists = buildWorklists({ ...f, methods: [...f.methods, inner], mappings: [] });
  assert.equal(lists.runtime[0].producer_method_id, innerId);
  const noRuntime = buildWorklists({ ...f, calls: [{ ...f.calls[0], code_kind: 'test' }],
    checks: [{ ...f.checks[0], code_kind: 'test' }] });
  assert.equal(noRuntime.runtime.length, 0);
  assert.throws(() => buildWorklists({ ...f, calls: [f.calls[0], { ...f.calls[0], call_line: '4' }] }));
  assert.throws(() => buildWorklists({ ...f, checks: [{ ...f.checks[0], repo: 'other/repo' }] }));
  assert.throws(() => buildWorklists({ ...f, mappings: [{ ...f.mappings[0], focal_line_end: '5' }] }));
  assert.throws(() => buildWorklists({ ...f, mappings: [], methods: [f.methods[0],
    { ...inner, qname: 'overlapping', line_end: '7' }] }), /ambiguous|span/i);
});

test('positive annotations must preserve target/test identity and substantive evidence', () => {
  const f = fixture();
  const testGuard = annotation({ arm: 'test_suite', test_id: testId });
  assert.equal(aggregateAnnotations({ ...f, annotations: [testGuard] }).verified.length, 1);
  for (const bad of [
    { arm: 'test_suite', test_id: testId.replace('::1', '::2') },
    { arm: 'runtime', test_id: testId },
    { producer_method_id: bId },
    { source_association: 'unresolved' }, { checked_property: '' }, { reviewer: '' },
    { failure_action: '' }, { rationale: '' }, { source_evidence: '[]' },
    { source_evidence: JSON.stringify([{ rel_path: 'app.py', line: 3, line_end: 3,
      sha256: 'a'.repeat(64), role: 'output' }]) },
    { oracle_type: 'not-a-type' },
  ]) assert.throws(() => aggregateAnnotations({ ...f, annotations: [annotation(bad)] }), undefined, JSON.stringify(bad));
  const out = aggregateAnnotations({ ...f, annotations: [annotation(), annotation()] });
  assert.equal(out.verified.length, 1);
  assert.throws(() => aggregateAnnotations({ ...f, annotations: [annotation(), annotation({ oracle_type: 'complex' })] }));
});

test('shared denominator and empty-review missingness do not turn placeholders into measured guardrails', () => {
  const f = fixture();
  const result = aggregateAnnotations({ ...f, annotations: [] });
  assert.equal(result.summary.denominator, 2);
  assert.equal(result.summary.mapped_methods, 1);
  assert.equal(result.summary.mapped_tests, 1);
  assert.equal(result.summary.mapping_pct, 50);
  assert.equal(result.summary.review_status, 'source_review_only_not_population_validation');
  for (const arm of ['runtime', 'test_suite']) {
    const s = result.summary.arms[arm];
    assert.equal(s.denominator, 2);
    assert.equal(s.identified_automated_methods, 0);
    assert.equal(s.category_counts.no_verified_evidence, 2);
    assert.equal(s.unreviewed_methods, 2);
    assert.deepEqual(s.composition_pct, {
      simple_detected_only: null, complex_detected_only: null, both_detected: null,
    });
    for (const m of result.methods) {
      assert.equal(Number(m[`${arm}_has_simple`]), 0);
      assert.equal(Number(m[`${arm}_has_complex`]), 0);
      assert.equal(Number(m[`${arm}_unreviewed`]), 1);
    }
  }
  const empty = aggregateAnnotations({ methods: [], mappings: [], calls: [], annotations: [] });
  assert.equal(empty.summary.denominator, 0);
  assert.equal(empty.summary.mapping_pct, null);
  assert.equal(empty.summary.arms.runtime.category_prevalence_pct.no_verified_evidence, null);
});

test('simple/complex overlap counts methods once; auxiliary, unknown, human, and rejected evidence stay distinct', () => {
  const f = fixture();
  const extra = (line, name) => ({ ...f.methods[1], producer_method_id: `${repo}::app.py::${line}`,
    line: String(line), line_end: String(line + 2), qname: name });
  const methods = [...f.methods, extra(14, 'opaque'), extra(18, 'manual')];
  const annotations = [
    annotation(), annotation(), annotation({ guardrail_id: 'semantic', oracle_type: 'complex' }),
    annotation({ guardrail_id: 'judge-schema', oracle_type: 'simple', auxiliary: '1' }),
    annotation({ guardrail_id: 'unresolved', producer_method_id: bId, output_call_id: '',
      decision: 'unresolved', oracle_type: 'unknown', source_association: 'unresolved', source_evidence: '[]' }),
    annotation({ guardrail_id: 'shell', producer_method_id: bId, output_call_id: '',
      decision: 'not_guardrail', oracle_type: 'unknown', source_association: 'unresolved', source_evidence: '[]' }),
    annotation({ guardrail_id: 'opaque', producer_method_id: methods[2].producer_method_id,
      output_call_id: '', oracle_type: 'unknown' }),
    annotation({ guardrail_id: 'manual', producer_method_id: methods[3].producer_method_id,
      output_call_id: '', oracle_type: 'human_only' }),
    annotation({ guardrail_id: 'test', arm: 'test_suite', test_id: testId }),
  ];
  const result = aggregateAnnotations({ ...f, methods, annotations });
  const s = result.summary.arms.runtime;
  assert.equal(s.denominator, 4);
  assert.deepEqual(s.category_counts, {
    simple_detected_only: 0, complex_detected_only: 0, both_detected: 1,
    unknown_type_only: 1, human_only: 1, no_verified_evidence: 1,
  });
  assert.equal(s.identified_automated_methods, 1);
  assert.equal(s.any_simple_methods, 1);
  assert.equal(s.any_complex_methods, 1);
  assert.equal(s.any_simple_prevalence_pct, 25);
  assert.equal(s.any_complex_prevalence_pct, 25);
  assert.equal(s.composition_pct.both_detected, 100);
  assert.equal(s.unknown_type_methods, 1);
  assert.equal(s.human_methods, 1);
  assert.equal(s.unresolved_methods, 1);
  assert.equal(s.rejected_annotations, 1);
  assert.equal(s.unresolved_annotations, 1);
  assert.equal(s.auxiliary_annotations, 1);
  assert.equal(result.summary.arms.test_suite.denominator, 4);
  assert.equal(result.summary.arms.test_suite.category_counts.simple_detected_only, 1);
  assert.equal(result.summary.arms.test_suite.unreviewed_methods, 3);
  const a = result.methods.find(m => m.producer_method_id === aId);
  assert.equal(Number(a.runtime_verified_count), 2);
  assert.equal(Number(a.runtime_auxiliary_count), 1);
  assert.equal(a.runtime_category, 'both_detected');
});

async function diskFixture(t, annotations = []) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'guardrail-audit-test-'));
  t.after(async () => {
    const target = path.resolve(dir);
    assert.ok(target.startsWith(path.resolve(os.tmpdir()) + path.sep));
    await fs.rm(target, { recursive: true, force: true });
  });
  const frameDir = path.join(dir, 'frame');
  const reposDir = path.join(dir, 'repos');
  await fs.mkdir(frameDir);
  await fs.mkdir(path.join(reposDir, 'acme_app', 'tests'), { recursive: true });
  const source = 'def produce():\n    prompt = "hi"\n    r = llm.invoke(prompt)\n    assert r\n    return r\n\n\ndef other():\n    r = llm.invoke("other")\n    return r\n\n\n';
  await fs.writeFile(path.join(reposDir, 'acme_app', 'app.py'), source);
  await fs.writeFile(path.join(reposDir, 'acme_app', 'tests', 'test_app.py'),
    'def test_produce():\n    setup = 1\n    r = produce()\n    assert r\n    return\n');
  const f = fixture();
  await fs.writeFile(path.join(frameDir, 'method_assessment.csv'), csv(Object.keys(f.methods[0]), f.methods));
  await fs.writeFile(path.join(frameDir, 'mapped_test_edges.csv'), csv(Object.keys(f.mappings[0]), f.mappings));
  await fs.writeFile(path.join(frameDir, 'readiness.json'), JSON.stringify({ frame: 'Synthetic complete-case frame', baseline_repos_excluded: 55 }));
  await fs.writeFile(path.join(frameDir, 'codebook.json'), JSON.stringify({ status: 'Independent validation pending', current_percentages: { runtime_simple: 99 } }));
  const callsCsv = path.join(dir, 'calls.csv');
  const checksCsv = path.join(dir, 'checks.csv');
  const annotationsCsv = path.join(dir, 'annotations.csv');
  await fs.writeFile(callsCsv, csv(Object.keys(f.calls[0]), f.calls));
  await fs.writeFile(checksCsv, csv(Object.keys(f.checks[0]), f.checks));
  const rows = annotations.map(a => ({ ...a, source_evidence: JSON.stringify(JSON.parse(a.source_evidence)
    .map(e => ({ ...e, sha256: e.sha256 === 'a'.repeat(64) ? digest(source) : e.sha256 }))) }));
  await fs.writeFile(annotationsCsv, csv(Object.keys(annotation()), rows));
  return { frameDir, reposDir, callsCsv, checksCsv, annotationsCsv,
    outputDir: path.join(dir, 'out'), source, dir };
}

test('disk audit verifies source hashes and provenance, writes fresh headered artifacts, and preserves inputs', async t => {
  const options = await diskFixture(t, [annotation({ rationale: 'Original source: ü, "quoted"\nsecond line' })]);
  const before = digest(await fs.readFile(options.callsCsv));
  await runAudit(options);
  assert.equal(digest(await fs.readFile(options.callsCsv)), before);
  const names = await fs.readdir(options.outputDir);
  for (const name of ['runtime_candidates.csv', 'test_suite_candidates.csv', 'methods.csv',
    'verified_guardrails.csv', 'rejected_candidates.csv', 'unresolved_candidates.csv',
    'diagnostics.csv', 'summary.json', 'REPORT.md', 'manifest.json']) assert.ok(names.includes(name), name);
  const verified = parseCsv(await fs.readFile(path.join(options.outputDir, 'verified_guardrails.csv'), 'utf8'));
  assert.equal(verified.length, 1);
  assert.equal(verified[0].rationale, 'Original source: ü, "quoted"\nsecond line');
  const manifest = await fs.readFile(path.join(options.outputDir, 'manifest.json'), 'utf8');
  assert.ok(manifest.includes(before), 'Consumed-input hashes are recorded.');
  assert.ok(manifest.includes(digest(options.source)), 'Inspected original-source hash is recorded.');
  for (const name of names.filter(n => n !== 'manifest.json')) {
    assert.ok(manifest.includes(digest(await fs.readFile(path.join(options.outputDir, name)))), name);
  }
  await assert.rejects(runAudit({ ...options, frameDir: path.join(options.dir, 'does-not-exist') }), /exists/i);
});

test('disk audit refuses stale evidence, unsafe paths, and out-of-bounds evidence lines', async t => {
  const badHash = await diskFixture(t, [annotation({ source_evidence: JSON.stringify(['output', 'checker', 'enforcement'].map(role => ({
    rel_path: 'app.py', line: 3, line_end: 3, sha256: 'b'.repeat(64), role,
  }))) })]);
  await assert.rejects(runAudit(badHash), /hash|sha256|changed|mismatch/i);
  const bounds = await diskFixture(t, [annotation({ source_evidence: JSON.stringify(['output', 'checker', 'enforcement'].map(role => ({
    rel_path: 'app.py', line: 300, line_end: 301, sha256: 'a'.repeat(64), role,
  }))) })]);
  await assert.rejects(runAudit(bounds), /line|bound/i);
  const unsafe = await diskFixture(t, [annotation({ source_evidence: JSON.stringify(['output', 'checker', 'enforcement'].map(role => ({
    rel_path: '../secret.py', line: 1, line_end: 1, sha256: 'a'.repeat(64), role,
  }))) })]);
  await assert.rejects(runAudit(unsafe), /unsafe|path|travers|escape/i);
  await t.test('source symlinks cannot substitute for clone files', async st => {
    const linked = await diskFixture(st, [annotation()]);
    const app = path.join(linked.reposDir, 'acme_app', 'app.py');
    await fs.writeFile(path.join(linked.reposDir, 'acme_app', 'real-app.py'), linked.source);
    await fs.unlink(app);
    try {
      await fs.symlink('real-app.py', app, 'file');
    } catch (error) {
      if (['EPERM', 'EACCES', 'ENOTSUP'].includes(error.code)) {
        st.skip(`Host does not allow synthetic source symlinks: ${error.code}`);
        return;
      }
      throw error;
    }
    await assert.rejects(runAudit(linked), /link|unsafe|path/i);
  });
});

test('a no-annotation disk run emits review-pending results with the entire synthetic frame', async t => {
  const options = await diskFixture(t);
  const withoutAnnotationFile = { ...options, annotationsCsv: null };
  const summary = await runAudit(withoutAnnotationFile);
  assert.equal(summary.denominator, 2);
  assert.equal(summary.arms.runtime.unreviewed_methods, 2);
  assert.equal(summary.arms.test_suite.unreviewed_methods, 2);
  assert.equal(summary.arms.runtime.composition_pct.simple_detected_only, null);
  assert.equal(parseCsv(await fs.readFile(path.join(options.outputDir, 'verified_guardrails.csv'), 'utf8')).length, 0);
  const report = await fs.readFile(path.join(options.outputDir, 'REPORT.md'), 'utf8');
  assert.match(report, /pending|unreviewed|not.*(?:population|complete)|incomplete/i);
  assert.doesNotMatch(report, /0% (?:of .* )?have guardrails/i);
  assert.match(report, /55/);
});

test('CLI help succeeds without inputs and unknown flags fail', () => {
  const file = path.resolve('pipeline/guardrail_prevalence_audit.mjs');
  const help = spawnSync(process.execPath, [file, '--help'], { encoding: 'utf8' });
  assert.equal(help.status, 0, help.stderr);
  assert.match(help.stdout, /frame-dir|usage/i);
  const bad = spawnSync(process.execPath, [file, '--invented-option'], { encoding: 'utf8' });
  assert.notEqual(bad.status, 0);
});
