"""Runtime call-site guardrail reconciliation (2026-11-12).

DETERMINISTIC: frozen-set reconstruction, joins on call IDs, counts, percentages, hashes.
ASSISTANT CLASSIFICATIONS (inputs, not produced here): reviews/batch_*.json (blind source review)
and adjudication.json (primary-session adjudication). No pyan artifact is read.

Usage: python -B consolidate.py <recon_dir> <out_dir>   (refuses an existing out_dir)
"""
import csv, collections, hashlib, json, os, sys, glob, datetime, platform, shutil
csv.field_size_limit(10**9)
ROOT = 'C:/Users/Seth/Documents/papers/GitHubSearch'
RG = ROOT + '/pipeline/artifacts/response_guardrail_slices'
REC, OUT = sys.argv[1], sys.argv[2]
if os.path.exists(OUT):
    sys.exit(f'refusing to write: {OUT} exists')

def rd(p): return list(csv.DictReader(open(p, encoding='utf-8', newline='')))
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
def pct(a, b): return round(100.0 * a / b, 6) if b else None

INPUTS = {
    'nov4_calls': RG + '/2026-11-04-v1/calls.csv',
    'nov4_checks': RG + '/2026-11-04-v1/checks.csv',
    'nov4_routes': RG + '/2026-11-04-v1/routes.csv',
    'nov4_manifest': RG + '/2026-11-04-v1/manifest.json',
    'nov4_candidate_tiers': RG + '/2026-11-04-candidate-tiers.csv',
    'nov4_review_md': RG + '/2026-11-04-review.md',
    'nov4_source_review': RG + '/2026-11-04-source-review.csv',
    'nov12_runtime_annotations': RG + '/2026-11-12-prevalence-source-audit-v1/annotations.csv',
    'nov12_runtime_strict': RG + '/2026-11-12-prevalence-source-audit-v1/runtime_strict_annotations.csv',
    'nov12_runtime_extra': RG + '/2026-11-12-prevalence-source-audit-v1/runtime_extra_annotations.csv',
    'nov12_runtime_supplemental': RG + '/2026-11-12-prevalence-source-audit-v1/runtime_supplemental_annotations.csv',
    'nov12_adjudication': RG + '/2026-11-12-prevalence-source-audit-v1/adjudication.json',
    'nov12_verified_guardrails': RG + '/2026-11-12-method-prevalence-v1/verified_guardrails.csv',
    'nov12_method_summary': RG + '/2026-11-12-method-prevalence-v1/summary.json',
    'saved_slice_call_hits': ROOT + '/pipeline/artifacts/slice_call_hits.csv',
    'saved_slice_pdg_lines': ROOT + '/pipeline/artifacts/slice_pdg_lines.csv.gz',
}
REVIEW_FILES = sorted(glob.glob(REC + '/reviews/batch_*.json'))
METHOD_FILES = {'protocol': REC + '/REVIEW_PROTOCOL.md', 'adjudication': REC + '/adjudication.json',
                'packet_builder': REC + '/build_packets.py', 'consolidator': os.path.abspath(__file__),
                'batches': REC + '/batches.json', 'packet_index': REC + '/index.json'}
hash_before = {k: sha(v) for k, v in INPUTS.items()}

calls = {c['call_id']: c for c in rd(INPUTS['nov4_calls'])}
checks = rd(INPUTS['nov4_checks'])
P = {k for k, c in calls.items() if c['code_kind'] == 'production'}
EXP = {'assertion', 'rejection_branch', 'schema_validation'}
TC = {k for k in P if json.loads(calls[k]['source_test_context'] or '[]')}

# ================= A. deterministic frozen reconstruction =================
by_call = collections.defaultdict(list)
for r in checks:
    if r['call_id'] in P: by_call[r['call_id']].append(r)
S106 = {c for c, rows in by_call.items() if any(r['check_kind'] in EXP and r['joern_status'] == 'same_response_slice' for r in rows)}
S175 = {c for c, rows in by_call.items() if any(r['check_kind'] in EXP for r in rows)}
S205 = set(by_call)
mech = {c: sorted({r['check_kind'] for r in by_call[c] if r['check_kind'] in EXP and r['joern_status'] == 'same_response_slice'}) for c in S106}
assert all(len(v) == 1 for v in mech.values())
prod_row = next(r for r in rd(INPUTS['nov4_candidate_tiers']) if r['code_kind'] == 'production')
recon = dict(call_sites=len(P), all_check_candidates=len(S205), explicit_check_candidates=len(S175),
             explicit_slice_supported=len(S106), explicit_outside_obvious_test_context=len(S175 - TC),
             explicit_slice_outside_obvious_test_context=len(S106 - TC), weaker_only_candidates=len(S205 - S175))
recon_match = {k: str(v) == prod_row[k] for k, v in recon.items()}
assert all(recon_match.values()), recon_match
mech_counts = dict(collections.Counter(v[0] for v in mech.values()))
assert mech_counts == {'assertion': 47, 'rejection_branch': 52, 'schema_validation': 7}
mech_by_check_rows = dict(collections.Counter(r['check_kind'] for c in S106 for r in by_call[c] if r['check_kind'] in EXP and r['joern_status'] == 'same_response_slice'))

# ================= reviews + adjudication =================
index = {x['site_no']: x for x in json.load(open(REC + '/index.json'))}
reviews = {}
for f in REVIEW_FILES:
    for o in json.load(open(f, encoding='utf-8')): reviews[o['site_no']] = o
assert set(reviews) == set(index)
for n, o in reviews.items(): assert o['call_id'] == index[n]['call_id'], n
adj = json.load(open(METHOD_FILES['adjudication'], encoding='utf-8'))
for n, patch in adj['site_overrides'].items():
    o = reviews[int(n)]
    before = {k: o.get(k) for k in patch if k not in ('guard_patches', 'adjudication_note')}
    for gi, gp in patch.get('guard_patches', {}).items():
        before[f'guard_{gi}'] = {k: o['guards'][int(gi)].get(k) for k in gp}
        o['guards'][int(gi)].update(gp)
    o.update({k: v for k, v in patch.items() if k not in ('guard_patches', 'adjudication_note')})
    o['adjudication'] = dict(note=patch['adjudication_note'], blind_values=before)
for n, fl in adj.get('site_flags', {}).items():
    reviews[int(n)].setdefault('adjudication_flags', []).extend(fl)
site_of = {o['call_id']: n for n, o in reviews.items()}
EXT = adj['external_producer_identity']

def q(g):
    return g.get('qualifies') == 'yes' and g.get('before_acceptance') == 'yes' and not g.get('auxiliary_judge_plumbing')

def identity(pid):
    if pid not in calls:
        return dict(in_frozen=False, saved_code_kind='', context='unresolved', invocation='unresolved', flags=[], source='none')
    if pid in site_of:
        o = reviews[site_of[pid]]
        return dict(in_frozen=True, saved_code_kind=calls[pid]['code_kind'], context=o['actual_context'],
                    invocation=o['invocation'], flags=sorted(set(o.get('context_flags', []) + o.get('adjudication_flags', []))), source='own_site_review')
    if pid in EXT:
        e = EXT[pid]
        return dict(in_frozen=True, saved_code_kind=calls[pid]['code_kind'], context=e['actual_context'],
                    invocation=e['invocation'], flags=e['context_flags'], source='adjudicator_identity_check')
    return dict(in_frozen=True, saved_code_kind=calls[pid]['code_kind'], context='unreviewed', invocation='unreviewed', flags=[], source='none')

# ================= producer aggregation =================
prod = collections.defaultdict(lambda: dict(oracles=set(), unknown=0, sites=set(), strata=set(), guards=[]))
unres_targets = collections.defaultdict(set)
for n, o in sorted(reviews.items()):
    st = index[n]['stratum']
    for gi, g in enumerate(o.get('guards', [])):
        pid = (g.get('protected_producer_call_id') or '').strip()
        if q(g):
            d = prod[pid]
            (d['oracles'].add(g['oracle_type']) if g.get('oracle_type') in ('simple', 'complex') else None)
            if g.get('oracle_type') not in ('simple', 'complex'): d['unknown'] += 1
            d['sites'].add(n); d['strata'].add(st); d['guards'].append((n, gi, g))
        elif g.get('qualifies') == 'unresolved' and pid:
            unres_targets[pid].add((n, st))

def cls(oracles):
    return {frozenset({'simple'}): 'simple_only', frozenset({'complex'}): 'complex_only',
            frozenset({'simple', 'complex'}): 'both'}.get(frozenset(oracles), 'unresolved_oracle')

def eligible(ident):
    return ident['in_frozen'] and ident['saved_code_kind'] == 'production' and ident['context'] == 'production' and ident['invocation'] == 'genuine_in_scope_llm'

producers = {}
for pid, d in prod.items():
    ident = identity(pid)
    gs = [g for _, _, g in d['guards']]
    def any_flag(key): return 'yes' if any(g.get(key) == 'yes' for g in gs) else ('no' if gs else '')
    producers[pid] = dict(
        producer_call_id=pid, guard_class=cls(d['oracles']), eligible_production_llm=eligible(ident),
        identity_source=ident['source'], reviewed_context=ident['context'], invocation=ident['invocation'],
        context_flags=';'.join(ident['flags']), saved_code_kind=ident['saved_code_kind'],
        deterministic_test_context_flag=pid in TC, in_frozen_106=pid in S106, in_uniform_205=pid in S205,
        discovered_via=';'.join(sorted(d['strata'])), discovered_at_sites=';'.join(map(str, sorted(d['sites']))),
        n_qualifying_guards=len(gs),
        simple_properties=' | '.join(sorted({g['checked_property'] for g in gs if g['oracle_type'] == 'simple'})),
        complex_properties=' | '.join(sorted({g['checked_property'] for g in gs if g['oracle_type'] == 'complex'})),
        joern_same_slice=any(str(g.get('joern_support', '')).startswith('call_and_check') for g in gs),
        retry_budget_bypass=any_flag('retry_budget_bypass'), fail_open=any_flag('fail_open'),
        configuration_dependent=any_flag('configuration_dependent'),
        assert_optimization_sensitive_only=all(g.get('assert_optimization_sensitive') for g in gs) if gs else False)

def numerator(strata, exclude_tc=False):
    return [p for p in producers.values() if p['eligible_production_llm']
            and set(p['discovered_via'].split(';')) & strata and not (exclude_tc and p['deterministic_test_context_flag'])]

def unresolved_candidates(strata, confirmed, exclude_tc=False):
    out = set()
    for n, o in reviews.items():
        if index[n]['stratum'] in strata and o['site_decision'] == 'unresolved':
            cid = o['call_id']
            if o['actual_context'] in ('test', 'test_support') or o['invocation'] == 'false_positive_non_llm': continue
            out.add(cid)
    for pid, hits in unres_targets.items():
        if any(st in strata for _, st in hits):
            ident = identity(pid)
            if ident['context'] in ('test', 'test_support') or ident['invocation'] == 'false_positive_non_llm': continue
            out.add(pid)
    out = {c for c in out if c in P and c not in confirmed and not (exclude_tc and c in TC)}
    return sorted(out)

def composition(sel):
    c = collections.Counter(p['guard_class'] for p in sel); n = len(sel)
    return {k: dict(count=c.get(k, 0), pct_of_numerator=pct(c.get(k, 0), n)) for k in ('simple_only', 'complex_only', 'both', 'unresolved_oracle')}

def pie_block(label, sel, denom, unres):
    n, u = len(sel), len(unres)
    return dict(
        label=label, unit='unique frozen call_id (production-labelled direct framework call site)', denominator=denom,
        pie1_three_slice=dict(detected_source_confirmed=dict(count=n, pct=pct(n, denom)),
                              unresolved_candidate=dict(count=u, pct=pct(u, denom)),
                              no_detected=dict(count=denom - n - u, pct=pct(denom - n - u, denom))),
        pie1_two_slice=dict(detected_source_confirmed=dict(count=n, pct=pct(n, denom)),
                            no_confirmed_detection=dict(count=denom - n, pct=pct(denom - n, denom))),
        pie2_composition_of_detected=composition(sel),
        detected_flags=dict(vendored_framework_copy=sum('vendored_framework_copy' in p['context_flags'] for p in sel),
                            example_tutorial_demo=sum(any(f in p['context_flags'] for f in ('example_or_tutorial', 'demo_or_sample_app')) for p in sel),
                            joern_same_saved_slice=sum(p['joern_same_slice'] for p in sel),
                            retry_budget_bypass=sum(p['retry_budget_bypass'] == 'yes' for p in sel),
                            fail_open=sum(p['fail_open'] == 'yes' for p in sel),
                            configuration_dependent=sum(p['configuration_dependent'] == 'yes' for p in sel),
                            assert_only_optimization_sensitive=sum(p['assert_optimization_sensitive_only'] for p in sel)),
        unresolved_call_ids=unres, detected_call_ids=sorted(p['producer_call_id'] for p in sel))

U, ALL = {'uniform'}, {'uniform', 'purposive'}
prim = numerator(U); prim_ids = {p['producer_call_id'] for p in prim}
prim_tc = numerator(U, True)
purp = numerator(ALL); purp_ids = {p['producer_call_id'] for p in purp}
pies = {
    'recommended_primary': pie_block(
        'Uniform discovery: every production-labelled site with a Nov 4 static check candidate (205) was source-reviewed; '
        'numerator = distinct protected producer call sites with >=1 source-confirmed runtime guardrail, eligible as genuine '
        'framework LLM calls in production context. Denominator = all 10,922 production-labelled detected call sites, NOT '
        'corrected for false-positive invocations or mislabelled test code (no consistent correction exists).',
        prim, len(P), unresolved_candidates(U, prim_ids)),
    'sensitivity_test_context_both_sides': pie_block(
        'Recommended primary with the deterministic Nov 4 source_test_context flag removed from numerator AND denominator.',
        prim_tc, len(P - TC), unresolved_candidates(U, {p['producer_call_id'] for p in prim_tc}, True)),
    'sensitivity_plus_purposive': pie_block(
        'Uniform discovery plus purposive semantic-search additions (2026-11-12 supplemental catalog). Discovery is unequal: '
        'the purposive search targeted LLM-judge loops, so complex share is not comparable to uniform discovery.',
        purp, len(P), unresolved_candidates(ALL, purp_ids)),
}
historical = dict(label='Historical Nov 4 static candidate tier (unchanged): explicit check candidate with matching saved response slice.',
                  numerator=len(S106), denominator=len(P), pct=pct(len(S106), len(P)),
                  mechanism_counts=mech_counts,
                  not_comparable_to='source-confirmed guardrails; mechanism counts are syntax/API categories, not simple/complex classes')

# ================= E1. frozen 106 reconciliation =================
prior = collections.defaultdict(list)
ann_rows = [r for r in rd(INPUTS['nov12_runtime_annotations']) if r['arm'] == 'runtime']
QM = 'ericjiang18_Agent-Q-Mix/agent_baseline/autogen/python/packages/autogen-agentchat/src/autogen_agentchat/agents/_code_executor_agent.py'
BLANK_RESOLVE = {'runtime-extra-044': [QM + '::808::31', QM + '::821::33']}
for r in ann_rows:
    for pid in (BLANK_RESOLVE.get(r['guardrail_id']) or [r['output_call_id']]):
        prior[pid].append(r)
strict_ids = {r['guardrail_id'] for r in rd(INPUTS['nov12_runtime_strict'])}
supp_ids = {r['guardrail_id'] for r in rd(INPUTS['nov12_runtime_supplemental'])}

def nonq_reasons(o):
    return ';'.join(sorted({g.get('nonqualifying_reason') for g in o.get('guards', []) if g.get('qualifies') != 'yes' and g.get('nonqualifying_reason')}))

def frozen_outcome(cid, o):
    if o['invocation'] == 'false_positive_non_llm': return 'rejected_false_positive_invocation'
    if o['actual_context'] in ('test', 'test_support'): return 'rejected_test_or_test_support_context'
    if cid in prim_ids or cid in purp_ids:
        return 'confirmed_guardrail_on_this_call'
    if o['site_decision'] == 'guard_attributed_to_other_producer_only': return 'judge_call_guard_attributed_to_earlier_generation'
    if o['site_decision'] == 'unresolved' or o['invocation'] == 'unresolved': return 'unresolved'
    return 'rejected_nonqualifying_check'

frozen_rows = []
for cid in sorted(S106):
    n = site_of[cid]; o = reviews[n]; c = calls[cid]
    attributed = sorted({g['protected_producer_call_id'] for g in o['guards'] if q(g) and g['protected_producer_call_id'] != cid})
    p = producers.get(cid)
    pri = prior.get(cid, [])
    frozen_rows.append(dict(
        site_no=n, call_id=cid, repo=c['repo'], rel_path=c['rel_path'], call_line=c['call_line'], framework=c['framework'], pattern=c['pattern'],
        file_sha256=json.load(open(f"{REC}/packets/site_{n:03d}.json", encoding='utf-8'))['file_sha256_now'],
        nov4_mechanism=mech[cid][0], nov4_sink_lines=';'.join(sorted({r['sink_line'] for r in by_call[cid] if r['check_kind'] in EXP and r['joern_status'] == 'same_response_slice'}, key=int)),
        nov4_response_slice_ids=';'.join(sorted({str(x) for r in by_call[cid] for x in json.loads(r['response_slice_ids'] or '[]')}, key=int)),
        nov4_test_context_flag=cid in TC,
        reviewed_context=o['actual_context'], context_flags=';'.join(o.get('context_flags', [])),
        invocation=o['invocation'], site_role=o.get('site_role', ''),
        outcome=frozen_outcome(cid, o),
        guard_class_on_this_call=(p['guard_class'] if p and p['eligible_production_llm'] else ''),
        guards_attributed_to_other_producers=';'.join(attributed),
        nonqualifying_reasons=nonq_reasons(o),
        adjudicated=bool(o.get('adjudication')), adjudication_flags=';'.join(o.get('adjudication_flags', [])),
        nov12_annotation=';'.join(sorted({r['guardrail_id'] + '=' + r['decision'] + ':' + r['oracle_type'] + (':aux' if r['auxiliary'] == '1' else '') for r in pri})) or 'none (outside 7,642-method frame)',
        rationale=o.get('rationale', '') + ((' | ADJUDICATION: ' + o['adjudication']['note']) if o.get('adjudication') else '')))

# ================= per-site audit (all reviewed sites + external producers) =================
audit_rows = []
for n in sorted(reviews):
    o = reviews[n]; cid = o['call_id']; c = calls[cid]; pk = json.load(open(f"{REC}/packets/site_{n:03d}.json", encoding='utf-8'))
    p = producers.get(cid)
    audit_rows.append(dict(
        site_no=n, call_id=cid, repo=c['repo'], rel_path=c['rel_path'], call_line=c['call_line'], framework=c['framework'], pattern=c['pattern'],
        saved_code_kind=c['code_kind'], file_sha256=pk['file_sha256_now'], hash_matches_nov4=pk['hash_matches_nov4'],
        discovery_stratum=index[n]['stratum'], nov4_tier=index[n]['tier'], in_frozen_106=cid in S106,
        nov4_mechanism=mech[cid][0] if cid in S106 else ';'.join(sorted({r['check_kind'] for r in by_call.get(cid, [])})),
        nov4_test_context_flag=cid in TC,
        reviewed_context=o['actual_context'], context_flags=';'.join(sorted(set(o.get('context_flags', []) + o.get('adjudication_flags', [])))),
        invocation=o['invocation'], invocation_evidence=o.get('invocation_evidence', ''), site_role=o.get('site_role', ''),
        site_decision=o['site_decision'],
        protected_by_confirmed_guardrail=bool(p and p['eligible_production_llm']),
        guard_class=(p['guard_class'] if p and p['eligible_production_llm'] else ''),
        counted_in_primary=cid in prim_ids, counted_in_plus_purposive=cid in purp_ids,
        guards_attributed_to_other_producers=';'.join(sorted({g['protected_producer_call_id'] for g in o['guards'] if q(g) and g['protected_producer_call_id'] != cid})),
        nonqualifying_reasons=nonq_reasons(o),
        nov12_annotation=';'.join(sorted({r['decision'] + ':' + r['oracle_type'] + (':aux' if r['auxiliary'] == '1' else '') for r in prior.get(cid, [])})),
        adjudicated=bool(o.get('adjudication')),
        rationale=o.get('rationale', '') + ((' | ADJUDICATION: ' + o['adjudication']['note']) if o.get('adjudication') else '')))
for pid, e in EXT.items():
    c = calls[pid]; p = producers.get(pid)
    path = os.path.join(ROOT, 'pipeline', 'repos', pid.split('/', 1)[0], *c['rel_path'].split('/'))
    audit_rows.append(dict(
        site_no='', call_id=pid, repo=c['repo'], rel_path=c['rel_path'], call_line=c['call_line'], framework=c['framework'], pattern=c['pattern'],
        saved_code_kind=c['code_kind'], file_sha256=sha(path), hash_matches_nov4='', discovery_stratum='attributed_producer_not_a_candidate',
        nov4_tier='', in_frozen_106=pid in S106, nov4_mechanism='', nov4_test_context_flag=pid in TC,
        reviewed_context=e['actual_context'], context_flags=';'.join(e['context_flags']), invocation=e['invocation'], invocation_evidence=e['evidence'],
        site_role='generation', site_decision='protected_producer_identified_via_other_site',
        protected_by_confirmed_guardrail=bool(p and p['eligible_production_llm']), guard_class=(p['guard_class'] if p else ''),
        counted_in_primary=pid in prim_ids, counted_in_plus_purposive=pid in purp_ids, guards_attributed_to_other_producers='',
        nonqualifying_reasons='', nov12_annotation=';'.join(sorted({r['decision'] + ':' + r['oracle_type'] for r in prior.get(pid, [])})),
        adjudicated=True, rationale='Identity checked by adjudicator in original source; guard found at site(s) ' + (p['discovered_at_sites'] if p else '')))
# file hash for externals vs Nov 4
man = json.load(open(INPUTS['nov4_manifest'], encoding='utf-8'))
n4 = {os.path.normcase(os.path.normpath(k)): v for k, v in man['inspected_sources'].items()}
for r in audit_rows:
    if r['hash_matches_nov4'] == '':
        path = os.path.join(ROOT, 'pipeline', 'repos', r['call_id'].split('/', 1)[0], *r['rel_path'].split('/'))
        h = n4.get(os.path.normcase(os.path.normpath(path)))
        r['hash_matches_nov4'] = (h == r['file_sha256']) if h else 'not_in_nov4_inspected_sources'

# ================= guard-level evidence =================
guard_rows = []
for n in sorted(reviews):
    o = reviews[n]
    for gi, g in enumerate(o['guards']):
        guard_rows.append(dict(
            site_no=n, guard_index=gi, site_call_id=o['call_id'], protected_producer_call_id=g.get('protected_producer_call_id', ''),
            counts_toward_producer=q(g), qualifies=g.get('qualifies'), before_acceptance=g.get('before_acceptance'),
            auxiliary_judge_plumbing=g.get('auxiliary_judge_plumbing'), oracle_type=g.get('oracle_type'),
            protected_output=g.get('protected_output', ''), checked_property=g.get('checked_property', ''), evaluator=g.get('evaluator', ''),
            verdict_location=g.get('verdict_location', ''), check_location=g.get('check_location', ''), enforcement_location=g.get('enforcement_location', ''),
            failure_consequence=g.get('failure_consequence', ''), nonqualifying_reason=g.get('nonqualifying_reason', ''),
            joern_support=g.get('joern_support', ''), joern_slice_ids=';'.join(map(str, g.get('joern_slice_ids', []) or [])),
            retry_budget_bypass=g.get('retry_budget_bypass', ''), fail_open=g.get('fail_open', ''),
            configuration_dependent=g.get('configuration_dependent', ''), assert_optimization_sensitive=g.get('assert_optimization_sensitive', ''),
            evidence_lines=json.dumps(g.get('evidence_lines', []), ensure_ascii=False), adjudicated=bool(o.get('adjudication'))))

# ================= C. crosswalk with 2026-11-12 runtime annotations =================
xw = []
for r in ann_rows:
    pids = BLANK_RESOLVE.get(r['guardrail_id']) or [r['output_call_id']]
    for pid in pids:
        p = producers.get(pid)
        src = 'strict(frozen106-in-frame)' if r['guardrail_id'] in strict_ids else ('supplemental(purposive)' if r['guardrail_id'] in supp_ids else 'extra(other Nov4 candidates in frame)')
        new_status = ('confirmed:' + p['guard_class'] if p and p['eligible_production_llm'] else
                      ('ineligible_producer' if p else 'no_confirmed_guard_on_this_producer'))
        prior_status = r['decision'] + (':' + r['oracle_type'] if r['decision'] == 'guardrail' else '') + (':aux' if r['auxiliary'] == '1' else '')
        agree = (r['decision'] == 'guardrail' and r['auxiliary'] != '1') == bool(p and p['eligible_production_llm'])
        xw.append(dict(nov12_guardrail_id=r['guardrail_id'], nov12_source=src, nov12_producer_method_id=r['producer_method_id'],
                       nov12_output_call_id=r['output_call_id'], resolved_producer_call_id=pid,
                       blank_call_id_resolved=bool(BLANK_RESOLVE.get(r['guardrail_id'])),
                       in_frozen_106=pid in S106, in_uniform_205=pid in S205,
                       nov12_status=prior_status, nov12_checked_property=r['checked_property'],
                       new_status=new_status, new_discovered_via=p['discovered_via'] if p else '',
                       guardrail_presence_agrees=agree,
                       class_agrees=(agree and (r['decision'] != 'guardrail' or r['auxiliary'] == '1' or
                                     (p and p['guard_class'] in ({'simple': 'simple_only', 'complex': 'complex_only'}[r['oracle_type']], 'both'))))))
# verified-method view: 86 methods -> producers
ver = [r for r in rd(INPUTS['nov12_verified_guardrails']) if r['arm'] == 'runtime']
ver_prod = set()
for r in ver: ver_prod |= set(BLANK_RESOLVE.get(r['guardrail_id']) or [r['output_call_id']])
crosswalk_summary = dict(
    nov12_runtime_annotation_rows=len(ann_rows), nov12_verified_runtime_annotations=len(ver),
    nov12_verified_methods=len({r['producer_method_id'] for r in ver}),
    nov12_verified_producer_call_ids=len(ver_prod),
    nov12_verified_producers_in_frozen_106=len(ver_prod & S106),
    nov12_verified_producers_in_uniform_205_not_106=len((ver_prod & S205) - S106),
    nov12_verified_producers_outside_205=len(ver_prod - S205),
    new_primary_producers=len(prim_ids), new_primary_in_106=len(prim_ids & S106),
    new_primary_in_205_not_106=len((prim_ids & S205) - S106), new_primary_attributed_outside_205=len(prim_ids - S205),
    nov12_verified_still_confirmed=len(ver_prod & purp_ids), nov12_verified_not_confirmed=sorted(ver_prod - purp_ids),
    new_confirmed_not_nov12_verified=sorted(purp_ids - ver_prod),
    presence_agreement_rows=collections.Counter(r['guardrail_presence_agrees'] for r in xw))

# ================= frozen-set summary counts =================
F = frozen_rows
frozen_summary = dict(
    total=len(F),
    outcome=dict(collections.Counter(r['outcome'] for r in F)),
    outcome_by_mechanism={m: dict(collections.Counter(r['outcome'] for r in F if r['nov4_mechanism'] == m)) for m in mech_counts},
    reviewed_context=dict(collections.Counter(r['reviewed_context'] for r in F)),
    invocation=dict(collections.Counter(r['invocation'] for r in F)),
    nov4_test_context_flag_x_reviewed_context={str(k): dict(collections.Counter(r['reviewed_context'] for r in F if r['nov4_test_context_flag'] == k)) for k in (True, False)},
    confirmed_class=dict(collections.Counter(r['guard_class_on_this_call'] for r in F if r['outcome'] == 'confirmed_guardrail_on_this_call')),
    in_nov12_frame=sum(not r['nov12_annotation'].startswith('none') for r in F),
    context_flags=dict(collections.Counter(f for r in F for f in r['context_flags'].split(';') if f)))
uniform_summary = dict(
    reviewed_uniform_sites=sum(index[n]['stratum'] == 'uniform' for n in reviews),
    by_tier={t: dict(collections.Counter(r['site_decision'] for r in audit_rows if r['nov4_tier'] == t)) for t in ('explicit_slice_supported', 'explicit_not_slice_supported', 'weaker_only')},
    invocation=dict(collections.Counter(r['invocation'] for r in audit_rows if r['discovery_stratum'] == 'uniform')),
    context=dict(collections.Counter(r['reviewed_context'] for r in audit_rows if r['discovery_stratum'] == 'uniform')))

# ================= write =================
os.makedirs(OUT)
def wcsv(name, rows):
    with open(os.path.join(OUT, name), 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL); w.writeheader(); w.writerows(rows)
wcsv('frozen_106_reconciliation.csv', frozen_rows)
wcsv('callsite_audit.csv', audit_rows)
wcsv('guard_evidence.csv', guard_rows)
wcsv('protected_producers.csv', sorted(producers.values(), key=lambda p: p['producer_call_id']))
wcsv('crosswalk_nov12_runtime.csv', xw)
os.makedirs(os.path.join(OUT, 'assistant_review'))
for f in REVIEW_FILES: shutil.copy2(f, os.path.join(OUT, 'assistant_review', os.path.basename(f)))
for k in ('protocol', 'adjudication', 'batches', 'packet_index'): shutil.copy2(METHOD_FILES[k], os.path.join(OUT, 'assistant_review', os.path.basename(METHOD_FILES[k])))
os.makedirs(os.path.join(OUT, 'scripts'))
for k in ('packet_builder', 'consolidator'): shutil.copy2(METHOD_FILES[k], os.path.join(OUT, 'scripts', os.path.basename(METHOD_FILES[k])))

counts = dict(
    schema='runtime-callsite-reconciliation-v1', date='2026-11-12',
    status='PRELIMINARY. Assistant source classifications; no independent human validation. Not a corrected population prevalence.',
    unit='unique frozen call_id: detected direct framework LLM call site, production-labelled (Nov 4 calls.csv)',
    historical_frozen=historical, frozen_reconstruction=dict(tier_counts=recon, matches_saved_tier_table=recon_match,
        mechanism_counts_one_per_site=mech_counts, mechanism_check_rows=mech_by_check_rows),
    frozen_106_reconciliation=frozen_summary, uniform_candidate_review=uniform_summary,
    presentation=pies, crosswalk_nov12=crosswalk_summary,
    pyan_used=False, executed_applications_models_joern_or_producers=False)
json.dump(counts, open(os.path.join(OUT, 'counts.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=str)

hash_after = {k: sha(v) for k, v in INPUTS.items()}
source_hashes = {r['call_id'].split('::')[0]: r['file_sha256'] for r in audit_rows}
manifest = dict(
    created=datetime.datetime.now().isoformat(timespec='seconds'), python=platform.python_version(),
    git_revision=os.popen(f'git -C "{ROOT}" rev-parse HEAD').read().strip(),
    inputs={k: dict(path=os.path.relpath(v, ROOT).replace('\\', '/'), sha256_before=hash_before[k], sha256_after=hash_after[k], unchanged=hash_before[k] == hash_after[k]) for k, v in INPUTS.items()},
    assistant_classification_inputs={os.path.basename(f): sha(f) for f in REVIEW_FILES} | {k: sha(v) for k, v in METHOD_FILES.items()},
    original_source_files=source_hashes,
    outputs={f: sha(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT)) if os.path.isfile(os.path.join(OUT, f))},
    provenance=dict(
        deterministic=['frozen tier reconstruction from Nov 4 calls.csv/checks.csv', 'joins on call_id', 'test-context flag (Nov 4 source_test_context)',
                       'counts, percentages, file hashes', 'Joern saved-slice PDG line lookup for review packets'],
        assistant_classified=['actual context', 'invocation genuineness', 'site role (generation/judge)', 'protected producer attribution',
                              'qualifies/before_acceptance', 'simple/complex oracle', 'retry/fail-open/config flags'],
        pyan='NOT USED. No pyan call graph, invoker closure, transitive reachability or pyan-derived test relationship was read. '
             'Note: the Stage-6 Joern slicer chose which functions to slice from the invoker closure; saved Joern slices are used only as co-membership corroboration.',
        not_executed='No application, model, discovery, batch analysis, focal mapping, pyMethods2Test or Joern producer was run.'),
    rules=dict(
        frozen_106='production-labelled call_id with >=1 Nov 4 check row of kind assertion|rejection_branch|schema_validation and joern_status == same_response_slice; one mechanism per site (no site has two).',
        denominator='all 10,922 production-labelled call_ids in Nov 4 calls.csv (= make_figures CALLS 23,990 restricted to code_kind production).',
        uniform_discovery='all 205 production-labelled call_ids with any Nov 4 check row (explicit 175 + weaker 30) were source-reviewed.',
        qualifying_guard='qualifies == yes AND before_acceptance == yes AND not auxiliary_judge_plumbing.',
        numerator='distinct protected_producer_call_id of qualifying guards found while reviewing sites of the named discovery strata; producer must be in frozen inventory, saved production, reviewed context production and genuine in-scope LLM invocation.',
        judge_attribution='a judge verdict that triggers intervention protects the generation it judges; credited to that producer call_id; a schema/type check solely on the judge verdict is auxiliary and not counted.',
        classes='simple_only / complex_only / both = oracle types of qualifying guards on the same producer; unresolved_oracle if only unknown-oracle qualifying guards.',
        unresolved_slice='production-labelled sites not in the numerator whose review decision is unresolved or which are targeted by a qualifies=unresolved guard, excluding reviewed false positives and test contexts.',
        denominator_not_corrected='Reviewer findings (false positives, test code, vendored copies) exist only for reviewed sites; applying them to the numerator alone is not a corrected prevalence. Only the deterministic test-context flag is applied to both sides (sensitivity).'))
json.dump(manifest, open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=str)
assert all(v['unchanged'] for v in manifest['inputs'].values())
print(json.dumps({k: (v['pie1_two_slice'], v['pie2_composition_of_detected'], v['pie1_three_slice']['unresolved_candidate']) for k, v in pies.items()}, indent=1, default=str))
print(json.dumps(frozen_summary, indent=1, default=str))
print(json.dumps(crosswalk_summary, indent=1, default=str))
print(json.dumps(uniform_summary, indent=1, default=str))
