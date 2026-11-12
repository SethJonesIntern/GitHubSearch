/** Read-only aggregation of source-reviewed guardrail annotations. */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const METHOD_FIELDS = 'producer_method_id repo rel_path qname line line_end mapped_tests exact_mapped_tests'.split(' ');
const MAPPING_FIELDS = 'repo test_file test_method test_line test_line_end focal_file focal_method focal_line focal_line_end file_rule method_rule method_key'.split(' ');
const CALL_FIELDS = 'call_id repo rel_path code_kind call_line'.split(' ');
const CHECK_FIELDS = 'call_id repo rel_path code_kind call_line sink_line check_kind mechanism intervention sink_source joern_status slice_ids response_slice_ids'.split(' ');
const INDEPENDENT_FIELDS = 'repo invoker_file invoker_line test_file test_line test_qname call_line check_line failure_line check_kind evidence call_source check_source'.split(' ');
const ANNOTATION_FIELDS = 'guardrail_id arm producer_method_id test_id output_call_id decision oracle_type checked_property failure_action source_association source_evidence reviewer rationale'.split(' ');
const ARMS = ['test_suite', 'runtime'];
const CATEGORIES = ['simple_detected_only', 'complex_detected_only', 'both_detected', 'unknown_type_only', 'human_only', 'no_verified_evidence'];
const LIMITATIONS = [
  'Source-reviewed identified guardrail methods in the supplied preliminary complete-case frame; not confirmed population prevalence.',
  'Unreviewed or no-verified-evidence methods do not establish absence of guardrails.',
  'Candidates, focal correspondence and saved slice membership do not establish guardrails or execution coverage.',
  'Purposive source review is not independent human validation or a probability sample; no sampling confidence intervals are claimed.',
  'Simple and complex findings can have different discovery sensitivity; composition describes only source-reviewed identified methods.',
  'Saved slice membership is corroboration, not a verified dependence path; original-source evidence is required.',
];

function fail(message) { throw new Error(message); }
function integer(value, label, minimum = 1) {
  if (!/^\d+$/.test(String(value)) || !Number.isSafeInteger(Number(value)) || Number(value) < minimum) fail(`${label}: expected integer >= ${minimum}`);
  return Number(value);
}
function stable(value) {
  if (Array.isArray(value)) return `[${value.map(stable).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${stable(value[k])}`).join(',')}}`;
  return JSON.stringify(value);
}
function digest(value) { return crypto.createHash('sha256').update(value).digest('hex'); }
function required(row, fields, label) {
  if (!row || fields.some(k => !Object.hasOwn(row, k) || row[k] === undefined || row[k] === null)) fail(`${label}: missing required columns ${fields.join(',')}`);
}
function rowsRequired(rows, fields, label) {
  if (!Array.isArray(rows)) fail(`${label}: expected row array`);
  rows.forEach(r => required(r, fields, label));
}
function unique(rows, key, label) {
  const result = new Map();
  for (const row of rows) {
    const id = key(row);
    if (result.has(id) && stable(result.get(id)) !== stable(row)) fail(`${label}: conflicting identity ${id}`);
    result.set(id, row);
  }
  return [...result.values()].sort((a, b) => key(a).localeCompare(key(b)));
}

export function parseCsv(text) {
  text = String(text).replace(/^\uFEFF/, '');
  if (!text.trim()) return [];
  const records = [];
  let row = [], value = '', quoted = false, closed = false;
  const field = () => { row.push(value); value = ''; closed = false; };
  const record = () => { field(); records.push(row); row = []; };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { value += '"'; i++; }
        else { quoted = false; closed = true; }
      } else value += c;
    } else if (c === '"') {
      if (value || closed) fail('CSV: unexpected quote in unquoted field');
      quoted = true;
    } else if (c === ',') field();
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      record();
    } else {
      if (closed) fail('CSV: unexpected characters after closing quote');
      value += c;
    }
  }
  if (quoted) fail('CSV: unterminated quoted field');
  if (value || row.length || closed) record();
  const headers = records.shift();
  if (!headers || headers.some(h => !h) || new Set(headers).size !== headers.length) fail('CSV: invalid or duplicate headers');
  return records.map((a, i) => {
    if (a.length !== headers.length) fail(`CSV: row ${i + 2} has ${a.length} fields, expected ${headers.length}`);
    return Object.fromEntries(headers.map((h, j) => [h, a[j]]));
  });
}

function repoSlug(repo) {
  if (typeof repo !== 'string' || !/^[^\\/:\s]+\/[^\\/:\s]+$/.test(repo) || repo.split('/').some(p => p === '.' || p === '..')) fail(`unsafe repo: ${repo}`);
  return repo.replace('/', '_');
}
export function normalizeRel(file, repo) {
  const slug = repoSlug(repo);
  if (typeof file !== 'string' || !file || file.includes('\0')) fail(`unsafe path: ${file}`);
  let rel = file.replaceAll('\\', '/');
  if (rel.startsWith('/') || /^[A-Za-z]:/.test(rel) || rel.includes(':')) fail(`unsafe path: ${file}`);
  if (rel.startsWith(`${slug}/`)) rel = rel.slice(slug.length + 1);
  const parts = rel.split('/');
  if (parts.some(p => !p || p === '.' || p === '..' || /[. ]$/.test(p))) fail(`unsafe path: ${file}`);
  return rel;
}
export function methodKey(repo, relPath, line) {
  return `${repo}::${normalizeRel(relPath, repo)}::${integer(line, 'method line')}`;
}
function testId(row) {
  return `${row.repo}::${normalizeRel(row.test_file, row.repo)}::${row.test_method}::${integer(row.test_line, 'test line')}`;
}

function identities({ methods, mappings, calls }) {
  rowsRequired(methods, METHOD_FIELDS, 'methods');
  rowsRequired(mappings, MAPPING_FIELDS, 'mappings');
  rowsRequired(calls, CALL_FIELDS, 'calls');
  const ms = unique(methods, r => r.producer_method_id, 'method');
  const byId = new Map(), byFile = new Map();
  for (const m of ms) {
    const start = integer(m.line, 'method start'), end = integer(m.line_end, 'method end');
    integer(m.mapped_tests, 'mapped_tests', 0); integer(m.exact_mapped_tests, 'exact_mapped_tests', 0);
    if (end < start || !m.qname || m.producer_method_id !== methodKey(m.repo, m.rel_path, start)) fail(`method identity mismatch: ${m.producer_method_id}`);
    byId.set(m.producer_method_id, m);
    const k = `${m.repo}::${normalizeRel(m.rel_path, m.repo)}`;
    if (!byFile.has(k)) byFile.set(k, []);
    byFile.get(k).push(m);
  }
  const maps = unique(mappings, r => `${r.method_key}|${testId(r)}`, 'mapping');
  const edgeByKey = new Map();
  for (const r of maps) {
    const m = byId.get(r.method_key);
    const start = integer(r.focal_line, 'focal line'), end = integer(r.focal_line_end, 'focal end');
    const tStart = integer(r.test_line, 'test line'), tEnd = integer(r.test_line_end, 'test end');
    if (tEnd < tStart || !m || r.method_key !== methodKey(r.repo, r.focal_file, start) || Number(m.line_end) !== end || m.qname.split('.').at(-1) !== r.focal_method) fail(`mapping does not match frame method: ${r.method_key}`);
    edgeByKey.set(`${r.method_key}|${testId(r)}`, r);
  }
  const cs = unique(calls, r => r.call_id, 'call');
  const callById = new Map(), callMethod = new Map();
  for (const c of cs) {
    const line = integer(c.call_line, 'call line');
    const rel = normalizeRel(c.rel_path, c.repo);
    const matches = (byFile.get(`${c.repo}::${rel}`) || []).filter(m => Number(m.line) <= line && Number(m.line_end) >= line).sort((a, b) => (Number(a.line_end) - Number(a.line)) - (Number(b.line_end) - Number(b.line)));
    if (matches.length > 1 && Number(matches[0].line_end) - Number(matches[0].line) === Number(matches[1].line_end) - Number(matches[1].line)) fail(`ambiguous containing method: ${c.call_id}`);
    callById.set(c.call_id, c);
    if (matches.length && c.code_kind === 'production') callMethod.set(c.call_id, matches[0].producer_method_id);
  }
  return { methods: ms, mappings: maps, calls: cs, byId, edgeByKey, callById, callMethod };
}

export function buildWorklists({ methods, mappings, calls, checks, independentChecks = [] }) {
  const id = identities({ methods, mappings, calls });
  rowsRequired(checks, CHECK_FIELDS, 'checks');
  rowsRequired(independentChecks, INDEPENDENT_FIELDS, 'independentChecks');
  const diagnostics = [], runtime = [], testSuite = [];
  for (const c of id.calls) if (c.code_kind === 'production' && !id.callMethod.has(c.call_id)) diagnostics.push({ kind: 'call_outside_frame', call_id: c.call_id, detail: `${c.repo}/${c.rel_path}:${c.call_line}` });
  for (const row of checks) {
    const call = id.callById.get(row.call_id);
    if (!call || row.repo !== call.repo || normalizeRel(row.rel_path, row.repo) !== normalizeRel(call.rel_path, call.repo) || Number(row.call_line) !== Number(call.call_line) || row.code_kind !== call.code_kind) fail(`check/call identity conflict: ${row.call_id}`);
    integer(row.sink_line, 'sink line');
    if (call.code_kind !== 'production') continue;
    const methodId = id.callMethod.get(row.call_id);
    if (!methodId) { diagnostics.push({ kind: 'runtime_check_outside_frame', call_id: row.call_id, detail: String(row.sink_line) }); continue; }
    const candidateId = digest(stable([methodId, row.call_id, row.sink_line, row.check_kind, row.sink_source]));
    runtime.push({ ...row, producer_method_id: methodId, qname: id.byId.get(methodId).qname, candidate_id: candidateId, review_status: 'unassessed' });
  }
  const independents = new Map();
  for (const row of independentChecks) {
    const mid = methodKey(row.repo, row.invoker_file, row.invoker_line);
    const tid = `${row.repo}::${normalizeRel(row.test_file, row.repo)}::${row.test_qname.split('.').at(-1)}::${integer(row.test_line, 'independent test line')}`;
    const key = `${mid}|${tid}`;
    if (!id.edgeByKey.has(key)) { diagnostics.push({ kind: 'independent_check_unjoined', call_id: '', detail: key }); continue; }
    integer(row.call_line, 'independent target call'); integer(row.check_line, 'independent check');
    if (!independents.has(key)) independents.set(key, []);
    independents.get(key).push(row);
  }
  for (const row of id.mappings) {
    const tid = testId(row), key = `${row.method_key}|${tid}`;
    const candidates = independents.get(key) || [];
    const base = { producer_method_id: row.method_key, test_id: tid, repo: row.repo, test_file: row.test_file, test_method: row.test_method, test_line: row.test_line, focal_file: row.focal_file, focal_line: row.focal_line, focal_method: row.focal_method, file_rule: row.file_rule, method_rule: row.method_rule, review_status: 'unassessed' };
    if (!candidates.length) testSuite.push({ ...base, candidate_id: digest(key), candidate_kind: 'mapped_test_unassessed' });
    else for (const c of candidates) testSuite.push({ ...base, ...c, test_id: tid, producer_method_id: row.method_key, candidate_id: digest(stable([key, c.call_line, c.check_line, c.check_kind, c.check_source])), candidate_kind: 'independent_return_check_candidate' });
  }
  return {
    runtime: unique(runtime, r => r.candidate_id, 'runtime candidate'),
    testSuite: unique(testSuite, r => r.candidate_id, 'test candidate'),
    diagnostics: unique(diagnostics, r => stable(r), 'diagnostic'),
  };
}

function evidence(row) {
  let entries;
  try { entries = typeof row.source_evidence === 'string' ? JSON.parse(row.source_evidence || '[]') : row.source_evidence; }
  catch { fail(`invalid source_evidence JSON: ${row.guardrail_id}`); }
  if (!Array.isArray(entries)) fail(`source_evidence must be array: ${row.guardrail_id}`);
  for (const e of entries) {
    required(e, ['rel_path', 'line', 'line_end', 'sha256', 'role'], 'source evidence');
    normalizeRel(e.rel_path, row.producer_method_id.split('::')[0]);
    if (integer(e.line_end, 'evidence end') < integer(e.line, 'evidence line') || !/^[a-fA-F0-9]{64}$/.test(e.sha256) || !['output', 'checker', 'enforcement', 'acceptance', 'support'].includes(e.role)) fail(`invalid source evidence: ${row.guardrail_id}`);
  }
  if (row.decision === 'guardrail' && ['output', 'checker', 'enforcement'].some(role => !entries.some(e => e.role === role))) fail(`guardrail needs output/checker/enforcement evidence: ${row.guardrail_id}`);
  return entries;
}
const percentage = (n, d) => d ? Number((100 * n / d).toFixed(6)) : null;
const tally = (rows, field) => Object.fromEntries([...new Set(rows.map(r => r[field]))].sort().map(v => [v, rows.filter(r => r[field] === v).length]));

export function aggregateAnnotations({ methods, mappings, calls, annotations }) {
  const id = identities({ methods, mappings, calls });
  rowsRequired(annotations, ANNOTATION_FIELDS, 'annotations');
  const anns = unique(annotations, r => `${r.arm}|${r.producer_method_id}|${r.guardrail_id}`, 'annotation');
  for (const a of anns) {
    if (!a.guardrail_id || !ARMS.includes(a.arm) || !['guardrail', 'not_guardrail', 'unresolved'].includes(a.decision) || !['simple', 'complex', 'human_only', 'unknown'].includes(a.oracle_type) || !['verified', 'unresolved'].includes(a.source_association) || !['', '0', '1', undefined].includes(a.auxiliary)) fail(`invalid annotation enums: ${a.guardrail_id}`);
    if (!id.byId.has(a.producer_method_id)) fail(`annotation method outside frame: ${a.producer_method_id}`);
    if (a.arm === 'runtime' && a.test_id) fail(`runtime annotation must have blank test_id: ${a.guardrail_id}`);
    if (a.arm === 'test_suite' && !id.edgeByKey.has(`${a.producer_method_id}|${a.test_id}`)) fail(`annotation test is not mapped to producer: ${a.guardrail_id}`);
    if (a.output_call_id && id.callMethod.get(a.output_call_id) !== a.producer_method_id) fail(`annotation output call mismatch: ${a.guardrail_id}`);
    if (a.decision === 'guardrail' && (a.source_association !== 'verified' || ['checked_property', 'failure_action', 'reviewer', 'rationale'].some(k => !String(a[k]).trim()))) fail(`incomplete verified guardrail: ${a.guardrail_id}`);
    evidence(a);
  }
  const verified = anns.filter(a => a.decision === 'guardrail' && a.auxiliary !== '1');
  const rejected = anns.filter(a => a.decision === 'not_guardrail');
  const unresolved = anns.filter(a => a.decision === 'unresolved');
  const results = id.methods.map(m => ({ ...m }));
  const summary = { schema_version: 'guardrail-prevalence-audit-v1', denominator: results.length, mapped_methods: new Set(id.mappings.map(r => r.method_key)).size, mapped_tests: new Set(id.mappings.map(testId)).size, review_status: 'source_review_only_not_population_validation', annotation_review_status: anns.length ? 'partial_source_review' : 'review_pending', correspondence: { methods_with_mapped_test: new Set(id.mappings.map(r => r.method_key)).size, mapped_test_edges: id.mappings.length }, arms: {}, limitations: LIMITATIONS };
  summary.correspondence.identified_method_pct = percentage(summary.correspondence.methods_with_mapped_test, results.length);
  summary.mapping_pct = summary.correspondence.identified_method_pct;
  for (const arm of ARMS) {
    const counts = Object.fromEntries(CATEGORIES.map(k => [k, 0]));
    let unreviewed = 0, hasUnknown = 0, hasHuman = 0, unresolvedMethods = 0;
    for (const m of results) {
      const rows = anns.filter(a => a.arm === arm && a.producer_method_id === m.producer_method_id);
      const positive = verified.filter(a => a.arm === arm && a.producer_method_id === m.producer_method_id);
      const types = new Set(positive.map(a => a.oracle_type));
      const s = types.has('simple'), c = types.has('complex'), u = types.has('unknown'), h = types.has('human_only');
      const category = s && c ? 'both_detected' : s ? 'simple_detected_only' : c ? 'complex_detected_only' : u ? 'unknown_type_only' : h ? 'human_only' : 'no_verified_evidence';
      counts[category]++;
      if (!rows.length) unreviewed++;
      if (u) hasUnknown++;
      if (h) hasHuman++;
      if (rows.some(a => a.decision === 'unresolved')) unresolvedMethods++;
      Object.assign(m, { [`${arm}_category`]: category, [`${arm}_has_simple`]: Number(s), [`${arm}_has_complex`]: Number(c), [`${arm}_has_unknown_type`]: Number(u), [`${arm}_has_human`]: Number(h), [`${arm}_has_unresolved`]: Number(rows.some(a => a.decision === 'unresolved')), [`${arm}_annotation_count`]: rows.length, [`${arm}_unreviewed`]: Number(!rows.length), [`${arm}_verified_count`]: positive.length, [`${arm}_verified_annotation_count`]: positive.length, [`${arm}_rejected_count`]: rows.filter(a => a.decision === 'not_guardrail').length, [`${arm}_unresolved_count`]: rows.filter(a => a.decision === 'unresolved').length, [`${arm}_auxiliary_count`]: rows.filter(a => a.auxiliary === '1').length });
    }
    const g = counts.simple_detected_only + counts.complex_detected_only + counts.both_detected;
    const positive = verified.filter(a => a.arm === arm);
    summary.arms[arm] = { denominator: results.length, categories: counts, identified_prevalence_pct: Object.fromEntries(CATEGORIES.map(k => [k, percentage(counts[k], results.length)])), verified_automated_methods: g, identified_automated_method_pct: percentage(g, results.length), any_simple_methods: counts.simple_detected_only + counts.both_detected, any_complex_methods: counts.complex_detected_only + counts.both_detected, any_simple_pct: percentage(counts.simple_detected_only + counts.both_detected, results.length), any_complex_pct: percentage(counts.complex_detected_only + counts.both_detected, results.length), overlap_note: 'Any-simple and any-complex overlap at both_detected methods.', composition_pct_among_identified_automated_methods: Object.fromEntries(CATEGORIES.slice(0, 3).map(k => [k, percentage(counts[k], g)])), unreviewed_methods: unreviewed, unresolved_methods: unresolvedMethods, methods_with_unknown_type: hasUnknown, methods_with_human_guard: hasHuman, verified_annotation_count: positive.length, rejected_annotation_count: rejected.filter(a => a.arm === arm).length, unresolved_annotation_count: unresolved.filter(a => a.arm === arm).length, auxiliary_annotation_count: anns.filter(a => a.arm === arm && a.auxiliary === '1').length, checked_properties_annotation_units: tally(positive, 'checked_property'), failure_actions_annotation_units: tally(positive, 'failure_action'), reviewers: [...new Set(anns.filter(a => a.arm === arm).map(a => a.reviewer).filter(Boolean))].sort(), independent_validation_status: 'Not established by this aggregation; source-review annotations are not independent validation evidence.' };
    const armSummary = summary.arms[arm];
    Object.assign(armSummary, { category_counts: armSummary.categories, category_prevalence_pct: armSummary.identified_prevalence_pct, identified_automated_methods: g, composition_pct: armSummary.composition_pct_among_identified_automated_methods, any_simple_prevalence_pct: armSummary.any_simple_pct, any_complex_prevalence_pct: armSummary.any_complex_pct, unknown_type_methods: hasUnknown, human_methods: hasHuman, verified_annotations: armSummary.verified_annotation_count, rejected_annotations: armSummary.rejected_annotation_count, unresolved_annotations: armSummary.unresolved_annotation_count, auxiliary_annotations: armSummary.auxiliary_annotation_count });
  }
  return { methods: results, summary, verified, rejected, unresolved };
}

function csvText(rows, minimumFields) {
  const fields = [...new Set([...minimumFields, ...rows.flatMap(r => Object.keys(r))])];
  const encode = v => `"${String(v == null ? '' : typeof v === 'object' ? stable(v) : v).replaceAll('"', '""')}"`;
  return [fields, ...rows.map(r => fields.map(f => r[f]))].map(r => r.map(encode).join(',')).join('\n') + '\n';
}
async function safeSource(reposDir, repo, rel) {
  const root = path.resolve(reposDir), target = path.resolve(root, repoSlug(repo), normalizeRel(rel, repo));
  const checkInside = (base, value) => { const r = path.relative(base, value); return r !== '..' && !r.startsWith(`..${path.sep}`) && !path.isAbsolute(r); };
  if (!checkInside(root, target)) fail(`source path escapes repos: ${rel}`);
  let current = root;
  // Check ancestors as well: an apparently safe root may itself be under a junction.
  for (let p = root; ; p = path.dirname(p)) {
    const stat = await fs.lstat(p);
    if (stat.isSymbolicLink()) fail(`source root link/junction rejected: ${p}`);
    if (path.dirname(p) === p) break;
  }
  for (const part of path.relative(root, target).split(path.sep)) {
    current = path.join(current, part);
    const stat = await fs.lstat(current);
    if (stat.isSymbolicLink()) fail(`source link/junction rejected: ${current}`);
  }
  const realRoot = await fs.realpath(root), realTarget = await fs.realpath(target);
  if (!checkInside(realRoot, realTarget)) fail(`resolved source escapes repos: ${rel}`);
  return target;
}
async function exists(file) { try { await fs.lstat(file); return true; } catch (e) { if (e.code === 'ENOENT') return false; throw e; } }

export async function runAudit({ frameDir, callsCsv, checksCsv, independentChecksCsv = null, annotationsCsv = null, reposDir, outputDir }) {
  if (!outputDir) fail('outputDir is required');
  if (await exists(outputDir)) fail(`output directory exists: ${outputDir}`);
  if (![frameDir, callsCsv, checksCsv, reposDir].every(Boolean)) fail('frameDir, callsCsv, checksCsv and reposDir are required');
  const inputs = new Map();
  async function read(file, fields = null) {
    const name = path.resolve(file), bytes = await fs.readFile(name);
    inputs.set(name, { sha256_before: digest(bytes) });
    if (fields) {
      const text = bytes.toString('utf8'), rows = parseCsv(text);
      // Header-only tables still require their schema.
      const headerLine = text.split(/\r?\n/, 1)[0];
      const headers = parseCsv(`${headerLine}\n${headerLine}\n`)[0];
      required(headers, fields, name); rowsRequired(rows, fields, name);
      return rows;
    }
    return JSON.parse(bytes.toString('utf8').replace(/^\uFEFF/, ''));
  }
  const methods = await read(path.join(frameDir, 'method_assessment.csv'), METHOD_FIELDS);
  const mappings = await read(path.join(frameDir, 'mapped_test_edges.csv'), MAPPING_FIELDS);
  const readiness = await read(path.join(frameDir, 'readiness.json'));
  const codebook = await read(path.join(frameDir, 'codebook.json'));
  const calls = await read(callsCsv, CALL_FIELDS), checks = await read(checksCsv, CHECK_FIELDS);
  const independentChecks = independentChecksCsv ? await read(independentChecksCsv, INDEPENDENT_FIELDS) : [];
  const annotations = annotationsCsv ? await read(annotationsCsv, ANNOTATION_FIELDS) : [];
  const work = buildWorklists({ methods, mappings, calls, checks, independentChecks });
  const result = aggregateAnnotations({ methods, mappings, calls, annotations });
  const sources = new Map();
  for (const row of unique(annotations, a => `${a.arm}|${a.producer_method_id}|${a.guardrail_id}`, 'annotation')) {
    const repo = row.producer_method_id.split('::')[0];
    for (const e of evidence(row)) {
      const file = await safeSource(reposDir, repo, e.rel_path);
      if (!sources.has(file)) {
        const bytes = await fs.readFile(file);
        sources.set(file, { sha256_before: digest(bytes), line_count: bytes.toString('utf8').replace(/\r\n/g, '\n').split('\n').length - Number(bytes.at(-1) === 10) });
      }
      const s = sources.get(file);
      if (s.sha256_before.toLowerCase() !== e.sha256.toLowerCase()) fail(`source hash mismatch: ${file}`);
      if (Number(e.line_end) > s.line_count) fail(`source evidence line out of bounds: ${file}:${e.line_end}`);
    }
  }
  const summary = { ...result.summary, frame_context: { readiness, codebook_status: codebook.status, codebook_limits: codebook.validation || codebook.limits || null }, candidates: { runtime_rows: work.runtime.length, runtime_same_response_slice_rows: work.runtime.filter(r => r.joern_status === 'same_response_slice').length, runtime_source_only_or_weaker_rows: work.runtime.filter(r => r.joern_status !== 'same_response_slice').length, runtime_weak_return_continue_rows: work.runtime.filter(r => ['early_exit_candidate', 'retry_branch'].includes(r.check_kind)).length, test_suite_rows: work.testSuite.length, test_suite_mapped_edges: new Set(work.testSuite.map(r => `${r.producer_method_id}|${r.test_id}`)).size, test_suite_independent_check_rows: work.testSuite.filter(r => r.candidate_kind === 'independent_return_check_candidate').length }, source_files_verified: sources.size };
  const report = [
    '# Source-reviewed identified guardrail methods', '',
    `Shared denominator: ${summary.denominator} supplied production LLM-containing methods. Review status: ${summary.review_status}.`, '',
    summary.annotation_review_status === 'review_pending' ? '**Review pending: no annotations were supplied. Zero identified counts do not mean 0% true guardrail prevalence.**' : '**These are source-reviewed identified rates in this frame, not completed population prevalence.**', '',
    `Focal correspondence: ${summary.correspondence.methods_with_mapped_test}/${summary.denominator} methods (${summary.correspondence.identified_method_pct ?? 'unavailable'}%). Correspondence does not establish output checking.`, '',
    '| Arm | Identified automated guardrail methods | Identified method rate | Simple only | Complex only | Both | Unreviewed methods |',
    '|---|---:|---:|---:|---:|---:|---:|',
    ...ARMS.map(arm => { const s = summary.arms[arm]; return `| ${arm} | ${s.verified_automated_methods}/${s.denominator} | ${s.identified_automated_method_pct ?? 'unavailable'}% | ${s.categories.simple_detected_only} | ${s.categories.complex_detected_only} | ${s.categories.both_detected} | ${s.unreviewed_methods} |`; }), '',
    '## Contents and unresolved evidence', '',
    ...ARMS.flatMap(arm => { const s = summary.arms[arm]; return [`${arm}: checked properties (annotation units): ${JSON.stringify(s.checked_properties_annotation_units)}.`, '', `Failure actions (annotation units): ${JSON.stringify(s.failure_actions_annotation_units)}.`, '', `Unknown-type methods: ${s.methods_with_unknown_type}; human-guard methods: ${s.methods_with_human_guard}; unresolved annotations: ${s.unresolved_annotation_count}; rejected candidates: ${s.rejected_annotation_count}; auxiliary annotations: ${s.auxiliary_annotation_count}.`, '']; }),
    '## Frame and review limitations', '',
    `Frame context: ${JSON.stringify(readiness)}.`, '', `Codebook status: ${codebook.status || 'not supplied'}.`, '',
    ...LIMITATIONS.map(s => `- ${s}`), '',
    'Independent human/detection validation is not established by these annotations. Source byte hashes and line bounds are verified, not the reviewer’s semantic judgments.', '',
    'Evidence references (up to ten source-reviewed rows):', '',
    ...result.verified.slice(0, 10).map(a => `- ${a.arm}: ${a.producer_method_id}; ${a.checked_property}; ${a.failure_action}; reviewer ${a.reviewer}; source_evidence ${a.source_evidence}`), '',
  ].join('\n');
  const outputBodies = {
    'runtime_candidates.csv': csvText(work.runtime, ['candidate_id', 'producer_method_id', ...CHECK_FIELDS, 'review_status']),
    'test_suite_candidates.csv': csvText(work.testSuite, ['candidate_id', 'producer_method_id', 'test_id', 'candidate_kind', 'review_status']),
    'methods.csv': csvText(result.methods, METHOD_FIELDS),
    'verified_guardrails.csv': csvText(result.verified, ANNOTATION_FIELDS),
    'rejected_candidates.csv': csvText(result.rejected, ANNOTATION_FIELDS),
    'unresolved_candidates.csv': csvText(result.unresolved, ANNOTATION_FIELDS),
    'diagnostics.csv': csvText(work.diagnostics, ['kind', 'call_id', 'detail']),
    'summary.json': JSON.stringify(summary, null, 2) + '\n',
    'REPORT.md': report,
  };
  for (const [file, state] of [...inputs, ...sources]) {
    state.sha256_after = digest(await fs.readFile(file));
    if (state.sha256_after !== state.sha256_before) fail(`consumed input/source changed: ${file}`);
  }
  const manifest = { node: process.version, script_sha256: digest(await fs.readFile(fileURLToPath(import.meta.url))), options: { frameDir, callsCsv, checksCsv, independentChecksCsv, annotationsCsv, reposDir, outputDir }, inputs: Object.fromEntries(inputs), inspected_sources: Object.fromEntries(sources), outputs: Object.fromEntries(Object.entries(outputBodies).map(([name, body]) => [name, digest(body)])) };
  // mkdir without recursive is an atomic final refusal if another process created it.
  await fs.mkdir(outputDir);
  for (const [name, body] of Object.entries(outputBodies)) await fs.writeFile(path.join(outputDir, name), body, { flag: 'wx' });
  await fs.writeFile(path.join(outputDir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
  return summary;
}

export async function main(argv = process.argv.slice(2)) {
  const names = { '--frame-dir': 'frameDir', '--calls-csv': 'callsCsv', '--checks-csv': 'checksCsv', '--repos-dir': 'reposDir', '--output-dir': 'outputDir', '--independent-checks-csv': 'independentChecksCsv', '--annotations-csv': 'annotationsCsv' };
  try {
    if (argv.length === 1 && ['--help', '-h'].includes(argv[0])) { console.log('Usage: node pipeline/guardrail_prevalence_audit.mjs --frame-dir PATH --calls-csv PATH --checks-csv PATH --repos-dir PATH --output-dir NEW_PATH [--independent-checks-csv PATH] [--annotations-csv PATH]'); return 0; }
    const options = {};
    for (let i = 0; i < argv.length; i += 2) {
      if (!names[argv[i]] || !argv[i + 1] || argv[i + 1].startsWith('--') || Object.hasOwn(options, names[argv[i]])) fail(`unknown, repeated or incomplete argument: ${argv[i]}`);
      options[names[argv[i]]] = argv[i + 1];
    }
    const summary = await runAudit(options);
    console.log(JSON.stringify({ denominator: summary.denominator, review_status: summary.review_status, arms: summary.arms }, null, 2));
    return 0;
  } catch (e) { console.error(`Guardrail audit: ${e.message}`); return 1; }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) process.exitCode = await main();
