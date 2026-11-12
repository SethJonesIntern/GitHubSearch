"""Discover pending test-output check reviews from frozen calls and original source.

This queue does not assign focal methods, classify oracles, or establish guards.
Only the explicitly supplied snapshot is used; application code is never executed.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tokenize

from pipeline import response_guardrail_slices as detector


def _rel(value: str, repo: str) -> str:
    value = value.replace('\\', '/')
    prefix = repo.replace('/', '_') + '/'
    return value[len(prefix):] if value.startswith(prefix) else value


def _test_key(row: dict) -> tuple:
    return (row['repo'], _rel(row['test_file'], row['repo']),
            row['test_method'], int(row['test_line']))


def build_review_candidates(calls: list[dict], results: list[dict],
                            mapping_rows: list[dict] | None = None) -> list[dict]:
    """Pure join; enriched results carry source lines/hash and lexical test identity.

    ``original_source_lines`` is a list of original lines without line endings.
    ``test_identity`` is None for helpers outside a lexical test function. A saved
    identity match says nothing about focal-method accuracy or producer identity.
    """
    by_call = {c['call_id']: c for c in calls}
    mapped = {_test_key(r) for r in mapping_rows or []}
    candidates = {}
    for result in results:
        call = by_call[result['call_id']]
        context = result.get('source_test_context', [])
        if call['code_kind'] not in {'test', 'test_support'} and not context:
            continue
        identity = result.get('test_identity')
        lines = result.get('original_source_lines', [])
        selected = [('check', r) for r in result.get('checks', [])
                    if r.get('check_kind') in {'assertion', 'rejection_branch'}]
        selected += [('uncertain_route', r) for r in result.get('routes', [])
                     if r.get('route_kind', '').startswith('uncertain_flow_')
                     and r['route_kind'].endswith('_candidate')]
        for origin, evidence in selected:
            trace = sorted(set(evidence.get('trace_lines', [])))
            if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 or n > len(lines)
                   for n in trace):
                raise ValueError(f"invalid original trace bounds: {call['call_id']}")
            payload = dict(
                call_id=call['call_id'], repo=call['repo'], rel_path=call['rel_path'],
                call_line=int(call['call_line']), code_kind=call['code_kind'],
                framework=call.get('framework', ''), pattern=call['pattern'],
                enclosing_qname=call.get('enclosing_qname', ''),
                source_test_context=context, test_identity=identity,
                has_saved_exact_test_identity_mapping=(None if mapping_rows is None else
                    bool(identity and _test_key(identity) in mapped)),
                review_status='pending', evidence_origin=origin,
                source_sha256=result.get('source_sha256', ''),
                trace_lines=trace,
                original_trace=[{'line': n, 'text': lines[n - 1]} for n in trace],
                evidence=dict(evidence), limitations=result.get('limitations', []),
                attribution='protected_production_method_not_assigned',
            )
            key = json.dumps([call['call_id'], origin, evidence], sort_keys=True)
            payload['candidate_id'] = hashlib.sha256(key.encode()).hexdigest()
            candidates[key] = payload
    return sorted(candidates.values(), key=lambda r: (r['repo'], r['rel_path'],
                  r['call_line'], r['evidence'].get('sink_line', 0), r['candidate_id']))


def _read_csv(path: Path, required: set[str], snapshots: dict) -> list[dict]:
    raw = path.read_bytes()
    snapshots[str(path.absolute())] = hashlib.sha256(raw).hexdigest()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    if not reader.fieldnames or not required <= set(reader.fieldnames):
        raise ValueError(f'{path}: missing headers {sorted(required)}')
    rows = list(reader)
    if any(None in r or any(r.get(k) is None for k in required) for r in rows):
        raise ValueError(f'{path}: malformed CSV')
    return rows


def _identity(tree: ast.AST, call: dict) -> dict | None:
    line = int(call['call_line'])
    tests = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name.startswith('test') and n.lineno <= line <= n.end_lineno]
    if not tests:
        return None
    node = max(tests, key=lambda n: n.lineno)
    return dict(repo=call['repo'], test_file=_rel(call['rel_path'], call['repo']),
                test_method=node.name, test_line=node.lineno, test_line_end=node.end_lineno)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calls-csv', required=True, type=Path)
    parser.add_argument('--repos-dir', required=True, type=Path)
    parser.add_argument('--mapped-edges-csv', type=Path)
    parser.add_argument('--repo', action='append', default=[])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    # Refuse before reading any input, including source or optional mappings.
    if args.output.exists() or args.output.is_symlink():
        print(f'output exists: {args.output}', file=sys.stderr)
        return 1
    if args.output.suffix.lower() != '.json':
        print('output must be a fresh .json file', file=sys.stderr)
        return 1
    snapshots, diagnostics, results = {}, [], []
    try:
        for module in (Path(__file__), Path(detector.__file__)):
            snapshots[str(module.absolute())] = detector._hash(module)
        calls = _read_csv(args.calls_csv,
                          {'call_id', 'repo', 'rel_path', 'code_kind', 'pattern', 'call_line'},
                          snapshots)
        calls = detector._unique_calls(calls)
        if args.repo:
            calls = [c for c in calls if c['repo'] in set(args.repo)]
        mappings = (None if args.mapped_edges_csv is None else
                    _read_csv(args.mapped_edges_csv,
                              {'repo', 'test_file', 'test_method', 'test_line'}, snapshots))
        grouped = defaultdict(list)
        for call in calls:
            # Derive only the established clone directory name, not study scope.
            call.setdefault('repo_slug', call['repo'].replace('/', '_'))
            try:
                grouped[detector._safe_source(args.repos_dir, call)].append(call)
            except detector.InputError as exc:
                diagnostics.append(dict(call_id=call['call_id'], status='unsafe_source', detail=str(exc)))
        for source_path, group in sorted(grouped.items(), key=lambda p: str(p[0])):
            try:
                raw = source_path.read_bytes()
                sha = hashlib.sha256(raw).hexdigest()
                snapshots[str(source_path.absolute())] = sha
                encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
                source = raw.decode(encoding)
                tree = ast.parse(source)
            except (OSError, UnicodeError, SyntaxError, LookupError, ValueError, RecursionError) as exc:
                diagnostics.extend(dict(call_id=c['call_id'], status='source_unavailable_or_unparseable',
                                        detail=str(exc)) for c in group)
                continue
            source_lines = source.splitlines()
            by_id = {c['call_id']: c for c in group}
            for result in detector.analyze_source(source, group):
                call = by_id[result['call_id']]
                result.update(source_sha256=sha, original_source_lines=source_lines,
                              test_identity=_identity(tree, call))
                results.append(result)
                if result['status'] != 'ok' or result.get('limitations'):
                    diagnostics.append(dict(call_id=call['call_id'], status=result['status'],
                                            limitations=result.get('limitations', [])))
        candidates = build_review_candidates(calls, results, mappings)
        # Revalidate components, even if a replacement link exposes identical bytes.
        for source_path, group in grouped.items():
            for call in group:
                if detector._safe_source(args.repos_dir, call) != source_path:
                    raise ValueError(f'source path changed: {source_path}; no queue written')
        hashes = []
        for name, before in sorted(snapshots.items()):
            after = detector._hash(name)
            if after != before:
                raise ValueError(f'consumed input changed: {name}; no queue written')
            hashes.append(dict(path=name, sha256_before=before, sha256_after=after))
        payload = dict(schema_version='guardrail-test-review-queue-v1', review_status='pending',
                       selected_saved_calls=len(calls), candidates=candidates, diagnostics=diagnostics,
                       consumed_inputs=hashes,
                       source_path_policy='response_guardrail_slices._safe_source; reject traversal, links and junctions',
                       limits=['Discovery only; no oracle classification or protected-producer mapping.',
                               'Frozen call identities only; absent calls/tests remain undiscovered.',
                               'Exact test-identity match does not validate saved focal methods.',
                               'Uncertain flow remains a pending lead, never a confirmed guardrail.',
                               'Source test context can differ from saved production labels; labels and scope are unchanged.'])
        # Exclusive create also protects against a competing writer after preflight.
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        return 0
    except (OSError, ValueError, detector.InputError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
