"""Build blind evidence packets for runtime call-site review. Read-only over study data."""
import ast, csv, gzip, hashlib, json, os, sys, collections
csv.field_size_limit(10**9)
ROOT = r'C:/Users/Seth/Documents/papers/GitHubSearch'
ART = ROOT + '/pipeline/artifacts'
RG = ART + '/response_guardrail_slices'
OUT = sys.argv[1]
def rd(p): return list(csv.DictReader(open(p, encoding='utf-8', newline='')))
calls = {c['call_id']: c for c in rd(RG + '/2026-11-04-v1/calls.csv')}
checks = rd(RG + '/2026-11-04-v1/checks.csv')
routes = rd(RG + '/2026-11-04-v1/routes.csv')
man = json.load(open(RG + '/2026-11-04-v1/manifest.json', encoding='utf-8'))
nov4_hash = {os.path.normcase(os.path.normpath(k)): v for k, v in man['inspected_sources'].items()}
P = {k for k, c in calls.items() if c['code_kind'] == 'production'}
EXP = {'assertion', 'rejection_branch', 'schema_validation'}
by_call = collections.defaultdict(list)
for r in checks:
    if r['call_id'] in P: by_call[r['call_id']].append(r)
def tier(cid):
    rows = by_call[cid]
    if any(r['check_kind'] in EXP and r['joern_status'] == 'same_response_slice' for r in rows): return 'explicit_slice_supported'
    if any(r['check_kind'] in EXP for r in rows): return 'explicit_not_slice_supported'
    return 'weaker_only'
uniform = sorted(by_call)
ann = [r for r in rd(RG + '/2026-11-12-prevalence-source-audit-v1/annotations.csv') if r['arm'] == 'runtime']
purposive = set()
for r in ann:
    if r['output_call_id'] and r['output_call_id'] not in by_call: purposive.add(r['output_call_id'])
qm = 'ericjiang18_Agent-Q-Mix/agent_baseline/autogen/python/packages/autogen-agentchat/src/autogen_agentchat/agents/_code_executor_agent.py'
purposive |= {qm + '::808::31', qm + '::821::33'}
purposive = sorted(purposive)
sites = [(c, 'uniform', tier(c)) for c in uniform] + [(c, 'purposive', 'purposive_only') for c in purposive]
need_slices = set()
hits = collections.defaultdict(list)
ids = {s[0] for s in sites}
for h in csv.DictReader(open(ART + '/slice_call_hits.csv', encoding='utf-8', newline='')):
    if h['call_id'] in ids:
        hits[h['call_id']].append({k: h[k] for k in ('slice_id', 'method', 'variable', 'relation', 'role', 'n_pdg', 'parses')})
        need_slices.add(h['slice_id'])
for r in checks:
    if r['call_id'] in ids:
        for f in ('slice_ids', 'shared_slice_ids', 'response_slice_ids'):
            try: need_slices |= {str(x) for x in json.loads(r[f] or '[]')}
            except Exception: pass
pdg = collections.defaultdict(list)
with gzip.open(ART + '/slice_pdg_lines.csv.gz', 'rt', encoding='utf-8') as fh:
    next(fh)
    for line in fh:
        s, l = line.rstrip('\n').split(',')
        if s in need_slices: pdg[s].append(int(l))
def enclosing(tree, line):
    best = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= line <= (n.end_lineno or n.lineno):
            if best is None or n.lineno >= best.lineno: best = n
    return best
os.makedirs(OUT + '/packets', exist_ok=True)
index = []
for i, (cid, stratum, tr) in enumerate(sites, 1):
    c = calls[cid]
    repo_dir = cid.split('/', 1)[0]
    path = os.path.join(ROOT, 'pipeline', 'repos', repo_dir, *c['rel_path'].split('/'))
    data = open(path, 'rb').read()
    sha = hashlib.sha256(data).hexdigest()
    n4 = nov4_hash.get(os.path.normcase(os.path.normpath(path)))
    text = data.decode('utf-8', errors='replace')
    lines = text.splitlines()
    line = int(c['call_line'])
    try:
        node = enclosing(ast.parse(text), line)
    except SyntaxError:
        node = None
    lo, hi = (node.lineno, node.end_lineno) if node else (max(1, line - 40), min(len(lines), line + 60))
    if hi - lo > 220: lo, hi = max(lo, line - 80), min(hi, line + 140)
    want = set(range(lo, hi + 1))
    for r in by_call.get(cid, []):
        s = int(r['sink_line']); want |= set(range(max(1, s - 8), min(len(lines), s + 8) + 1))
    excerpt = '\n'.join(f'{n:5d}| {lines[n-1]}' for n in sorted(want) if 1 <= n <= len(lines))
    pk = dict(site_no=i, call_id=cid, discovery_stratum=stratum, nov4_tier=tr, repo=c['repo'], repo_dir=repo_dir,
              rel_path=c['rel_path'], abs_path=path, saved_code_kind=c['code_kind'], framework=c['framework'],
              pattern=c['pattern'], call_line=line, call_col=c['call_col'], output_form=c['output_form'],
              enclosing_qname=c['enclosing_qname'], source_test_context=json.loads(c['source_test_context'] or '[]'),
              file_sha256_now=sha, file_sha256_nov4=n4, hash_matches_nov4=(sha == n4) if n4 else None,
              enclosing_def_lines=[lo, hi],
              saved_checks=[{k: r[k] for k in ('check_kind', 'mechanism', 'intervention', 'sink_line', 'sink_source', 'joern_status', 'response_variables', 'trace_lines', 'shared_slice_ids', 'response_slice_ids')} for r in by_call.get(cid, [])],
              saved_routes=[{k: r[k] for k in ('route_kind', 'sink_line', 'sink_source')} for r in routes if r['call_id'] == cid],
              saved_slice_hits=hits.get(cid, []),
              saved_slice_pdg_lines={s: sorted(pdg[s]) for s in sorted({h['slice_id'] for h in hits.get(cid, [])} | {str(x) for r in by_call.get(cid, []) for x in json.loads(r['response_slice_ids'] or '[]')}, key=int)},
              source_excerpt=excerpt)
    fn = f'{OUT}/packets/site_{i:03d}.json'
    json.dump(pk, open(fn, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    index.append(dict(site_no=i, call_id=cid, stratum=stratum, tier=tr, repo=c['repo'], hash_matches_nov4=pk['hash_matches_nov4']))
json.dump(index, open(OUT + '/index.json', 'w', encoding='utf-8'), indent=1)
print(len(index), collections.Counter((x['stratum'], x['tier']) for x in index), collections.Counter(str(x['hash_matches_nov4']) for x in index))
