"""Bounded response-check candidates from original source and saved slice lines.

No models, applications, validators or graph tools are executed. Slice membership
corroborates joint source lines, not a saved dependence path. Negative results
are deliberately incomplete; unsupported flow is retained in diagnostics.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass

from pipeline import paths

CALL_HEADERS = 'call_id repo repo_slug rel_path code_kind framework pattern call_line enclosing_qname repo_sliced in_slice'.split()
HIT_HEADERS = 'slice_id call_id call_repo repo rel_path code_kind call_line variable role parses'.split()
INDEX_HEADERS = 'slice_id repo rel_path code_kind method variable parses'.split()
CALL_OUTPUT = 'call_id repo rel_path code_kind framework pattern call_line call_col source_status output_form response_variables source_check_count slice_supported_check_count route_count'.split()
CHECK_OUTPUT = 'call_id repo rel_path code_kind call_line sink_line check_kind mechanism intervention resolution sink_source response_variables trace_lines joern_status slice_ids shared_slice_ids response_slice_ids slice_variables slice_parse_flags'.split()
ROUTE_OUTPUT = 'call_id repo rel_path code_kind call_line sink_line route_kind sink_source response_variables'.split()
DIAGNOSTIC_OUTPUT = 'call_id repo rel_path kind detail'.split()
_CANONICAL_INPUT_HASHES = {}


class InputError(ValueError):
    """Saved inputs cannot safely describe this analysis."""


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _rows(path, required):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(required) <= set(reader.fieldnames):
            raise InputError(f'{path}: missing required headers {required}; check Git LFS content')
        for row in reader:
            if None in row or any(row.get(key) is None for key in required):
                raise InputError(f'{path}: malformed CSV row {reader.line_num}')
            yield row


def _integer(value, key, minimum=0):
    if isinstance(value, bool) or not re.fullmatch(r'\d+', str(value)):
        raise InputError(f'{key}: expected integer, got {value!r}')
    result = int(value)
    if result < minimum:
        raise InputError(f'{key}: must be >= {minimum}')
    return result


def _call(row):
    row = dict(row)
    try:
        _, line, col = row['call_id'].rsplit('::', 2)
    except (KeyError, ValueError) as exc:
        raise InputError(f'malformed call_id: {row.get("call_id")!r}') from exc
    line = _integer(line, row['call_id'], 1)
    col = _integer(col, row['call_id'])
    if _integer(row['call_line'], 'call_line', 1) != line:
        raise InputError(f'{row["call_id"]}: call_line disagrees with ID')
    if 'call_col' in row and _integer(row['call_col'], 'call_col') != col:
        raise InputError(f'{row["call_id"]}: call_col disagrees with ID')
    row.update(call_line=line, call_col=col)
    return row


def _unique_calls(rows):
    unique = {}
    for row in rows:
        row = _call(row)
        key = row['call_id']
        if key in unique and unique[key] != row:
            raise InputError(f'conflicting call identity: {key}')
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def load_calls(artifacts_dir: Path | None = None) -> list[dict]:
    """Read saved scope, refusing a snapshot different from canonical MF.CALLS."""
    root = Path(artifacts_dir) if artifacts_dir is not None else paths.ARTIFACTS_DIR
    saved_path = root / 'slice_calls_by_call.csv'
    cached = sys.modules.get('Applications.make_figures')
    synthetic = cached is not None and not getattr(cached, '__file__', None)
    consumed = [saved_path]
    if not synthetic:
        consumed += [paths.ARTIFACTS_DIR / 'llm_calls_all.csv',
                     paths.ARTIFACTS_DIR / 'application_audit.csv', paths.BATCH_PROGRESS_JSON]
    before = {str(p.absolute()): _hash(p) for p in consumed}
    rows = _unique_calls(_rows(saved_path, CALL_HEADERS))
    try:
        from Applications import make_figures as MF
    except (ImportError, ModuleNotFoundError) as exc:
        raise InputError(f'canonical scope dependency unavailable: {exc}; restore existing environment, do not regenerate inputs') from exc
    canonical = {(str(r['call_id']), str(r['repo'])) for r in MF.CALLS.to_dict('records')}
    saved = {(r['call_id'], r['repo']) for r in rows}
    if saved != canonical:
        raise InputError(f'{saved_path}: snapshot mismatch against MF.CALLS (saved={len(saved)}, canonical={len(canonical)})')
    for name, digest in before.items():
        if _hash(name) != digest:
            raise InputError(f'changed consumed input: {name}')
    _CANONICAL_INPUT_HASHES.clear()
    _CANONICAL_INPUT_HASHES.update(before)
    return rows


@dataclass(frozen=True)
class _Flow:
    variables: frozenset = frozenset()
    lines: frozenset = frozenset()

    def __bool__(self):
        return bool(self.lines)


def _join(*flows):
    return _Flow(frozenset().union(*(f.variables for f in flows)),
                 frozenset().union(*(f.lines for f in flows)))


def _terminal(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ''


def _literal_truth(node):
    """Only obvious truth constants; no evaluation of application expressions."""
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        if any(isinstance(n, ast.Starred) for n in node.elts):
            return None
        return bool(node.elts)
    if isinstance(node, ast.Dict):
        return True if any(k is not None for k in node.keys) else (False if not node.keys else None)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _literal_truth(node.operand)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_literal_truth(n) for n in node.values]
        if isinstance(node.op, ast.Or):
            return True if True in values else (False if all(v is False for v in values) else None)
        return False if False in values else (True if all(v is True for v in values) else None)
    return None


def _names_written(nodes):
    names = set()
    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name):
                names.add(root.id)
        if isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split('.')[0] for a in node.names)
        if isinstance(node, ast.MatchAs) and node.name:
            names.add(node.name)
        if isinstance(node, ast.MatchStar) and node.name:
            names.add(node.name)
        if isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        for child in ast.iter_child_nodes(node):
            visit(child)
    for node in nodes:
        visit(node)
    return names


class _Local:
    def __init__(self, source, tree, parents, seed, call, scope):
        self.source, self.tree, self.parents = source, tree, parents
        self.seed, self.call, self.scope = seed, call, scope
        self.env, self.symbols = {}, {}
        self.checks, self.routes, self.limitations = [], [], set()
        self.output_form = 'unassessed'
        self.response_variables = set()
        self.seen = set()
        self.seed_seen = False
        self.invalidated_events = []

    def limit(self, message):
        self.limitations.add(message)

    def symbol(self, node):
        if isinstance(node, ast.Name):
            return self.symbols.get(node.id, '')
        if isinstance(node, ast.Attribute):
            base = self.symbol(node.value)
            return base + '.' + node.attr if base else ''
        return ''

    def sink(self, node, flow, kind, intervention='', resolution='source_local', route=False):
        if not flow:
            return
        key = (route, node.lineno, node.col_offset, kind)
        if key in self.seen:
            return
        self.seen.add(key)
        row = dict(call_id=self.call['call_id'], call_line=self.call['call_line'],
                   sink_line=node.lineno, sink_source=ast.get_source_segment(self.source, node) or '',
                   response_variables=sorted(flow.variables), trace_lines=sorted(flow.lines | {node.lineno}))
        if route:
            row['route_kind'] = kind
            self.routes.append(row)
        else:
            row.update(check_kind=kind, mechanism=('deterministic_assert' if kind == 'assertion' else
                       'schema_validator' if kind == 'schema_validation' else 'response_predicate'),
                       intervention=intervention, resolution=resolution)
            self.checks.append(row)

    def invalidate(self, nodes, reason):
        mutated = set()
        for outer in nodes:
            for node in ast.walk(outer):
                if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    root = node.value
                    while isinstance(root, (ast.Attribute, ast.Subscript)):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in self.env:
                        mutated.update(self.env[root.id].variables)
        for name, flow in list(self.env.items()):
            if flow.variables & mutated:
                self.env.pop(name, None)
                self.invalidated_events.append(name)
                self.limit('unsupported_assignment_or_mutation')
        for name in _names_written(nodes):
            if name in self.env:
                self.limit(reason)
            self.env.pop(name, None)
            self.symbols.pop(name, None)

    def bind(self, target, flow, line):
        if isinstance(target, ast.Name):
            self.env.pop(target.id, None)
            self.symbols.pop(target.id, None)
            if flow:
                tagged = _Flow(flow.variables | {target.id}, flow.lines | {line})
                self.env[target.id] = tagged
                self.response_variables.update(tagged.variables)
        else:
            self.invalidate([target], 'unsupported_assignment_or_mutation')
            if flow:
                self.limit('unsupported_assignment_or_mutation')

    def expr(self, node):
        if node is None:
            return _Flow()
        if node is self.seed:
            self.seed_seen = True
            return _Flow(lines=frozenset({node.lineno}))
        if isinstance(node, ast.Name):
            return self.env.get(node.id, _Flow())
        if isinstance(node, ast.Constant):
            return _Flow()
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            self.limit('evaluation_flow_unassessed' if isinstance(node, ast.Lambda) else 'unsupported_comprehension_scope')
            return _Flow()
        if isinstance(node, ast.NamedExpr):
            self.expr(node.value)
            self.invalidate([node.target], 'unsupported_assignment_expression')
            self.limit('unsupported_assignment_expression')
            return _Flow()
        if isinstance(node, ast.Await):
            return self.expr(node.value)
        if isinstance(node, ast.Attribute):
            flow = self.expr(node.value)
            if flow:
                self.limit('receiver_dispatch_unresolved')
            return flow
        if isinstance(node, ast.Subscript):
            flow = self.expr(node.value)
            self.expr(node.slice)
            return flow
        if isinstance(node, ast.Compare):
            return _join(self.expr(node.left), *(self.expr(n) for n in node.comparators))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.expr(node.operand)
        if isinstance(node, ast.BoolOp):
            flows = [self.expr(n) for n in node.values]
            return _Flow() if _literal_truth(node) is not None else _join(*flows)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            # Containers can carry values but their own truth is not value validation.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.expr(child)
            self.limit('container_response_flow_unassessed')
            return _Flow()
        if isinstance(node, ast.Call):
            return self.call_expr(node)
        flows = [self.expr(n) for n in ast.iter_child_nodes(node) if isinstance(n, ast.expr)]
        if any(flows) or any(n is self.seed for n in ast.walk(node)):
            self.limit('unsupported_expression:' + type(node).__name__)
        return _Flow()

    def predicate(self, node):
        flow = self.expr(node)
        return flow if _literal_truth(node) is None else _Flow()

    def call_expr(self, node):
        name = _terminal(node.func)
        symbol = self.symbol(node.func)
        receiver = self.expr(node.func.value) if isinstance(node.func, ast.Attribute) else _Flow()
        args = [self.expr(n) for n in node.args]
        keywords = {k.arg: self.expr(k.value) for k in node.keywords if k.arg is not None}
        combined = _join(receiver, *args, *keywords.values())
        def port(index, keyword):
            return args[index] if len(args) > index else keywords.get(keyword, _Flow())
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
            operand_counts = {'assertEqual': 2, 'assertTrue': 1, 'assertIsNotNone': 1,
                              'assertIn': 2, 'assertIsInstance': 2, 'assertGreaterEqual': 2}
            if name in operand_counts:
                count = operand_counts[name]
                operand_names = {'assertEqual': ('first', 'second'), 'assertTrue': ('expr',),
                                 'assertIsNotNone': ('obj',), 'assertIn': ('member', 'container'),
                                 'assertIsInstance': ('obj', 'cls'), 'assertGreaterEqual': ('a', 'b')}[name]
                flow = _join(*(port(i, k) for i, k in enumerate(operand_names[:count])))
                self.sink(node, flow, 'assertion', 'assertion_failure')
                self.limit('unittest_receiver_dispatch_unresolved')
                return _Flow()
        schema = symbol == 'jsonschema.validate'
        model = name == 'model_validate' and isinstance(node.func, ast.Attribute) and self.symbol(node.func.value) == 'known_model'
        adapter = name == 'validate_python' and isinstance(node.func, ast.Attribute) and (
            self.symbol(node.func.value) == 'known_adapter' or
            isinstance(node.func.value, ast.Call) and self.symbol(node.func.value.func) == 'pydantic.TypeAdapter')
        if self.symbol(node.func) == 'known_judge':
            flow = keywords.get('outputs', _Flow())
            parent = self.parents.get(node)
            while isinstance(parent, ast.Await):
                parent = self.parents.get(parent)
            kind = 'evaluation_only' if isinstance(parent, ast.Expr) else 'evaluation_flow_unassessed'
            self.sink(node, flow, kind, route=True)
            if flow:
                self.limit('evaluation_flow_unassessed')
            return _Flow()
        if schema or model or adapter:
            flow = port(0, 'instance' if schema else 'obj' if model else 'object')
            # Pydantic publicly accepts positional data; common keyword spelling is obj/object.
            self.sink(node, flow, 'schema_validation', 'validator_exception_contract', 'source_import')
            return flow
        if symbol == 'json.loads':
            flow = port(0, 's')
            self.sink(node, flow, 'parser_only', route=True)
            return flow
        if name in {'validate', 'model_validate', 'validate_python', 'parse_obj', 'validate_json'}:
            self.sink(node, combined, 'unresolved_validator', route=True)
            if combined:
                self.limit('validator_dispatch_unresolved')
            return _Flow()
        if (name in {'evaluate', 'aevaluate', 'evaluate_strings', 'evaluate_run', 'evaluate_response'} or
                name.startswith('evaluate_')) and combined:
            self.sink(node, combined, 'evaluation_flow_unassessed', route=True)
            self.limit('evaluation_flow_unassessed')
            return _Flow()
        if isinstance(node.func, ast.Name) and name in {'len', 'str', 'bool', 'isinstance'} and name not in self.shadowed_builtins:
            self.limit('builtin_dispatch_unresolved')
            # isinstance type/config operands are not the value under inspection.
            return port(0, 'obj')
        if isinstance(node.func, ast.Attribute) and name in {'lower', 'upper', 'strip', 'startswith', 'endswith'}:
            if receiver:
                self.limit('receiver_dispatch_unresolved')
            return receiver
        if combined:
            self.limit('unknown_transform_or_callback:' + (name or '<dynamic>'))
            # Unknown helpers can mutate arguments or receivers: do not keep stale aliases.
            touched = set(combined.variables)
            for key, flow in list(self.env.items()):
                if flow.variables & touched:
                    self.env.pop(key, None)
                    self.invalidated_events.append(key)
        if any(isinstance(n, ast.Lambda) for n in node.args):
            self.limit('evaluation_flow_unassessed')
        return _Flow()

    def import_stmt(self, node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                key = alias.asname or alias.name.split('.')[0]
                self.env.pop(key, None)
                self.symbols[key] = alias.name if alias.asname else alias.name.split('.')[0]
        else:
            for alias in node.names:
                key = alias.asname or alias.name
                self.env.pop(key, None)
                self.symbols[key] = (node.module + '.' + alias.name) if node.module and node.level == 0 else ''

    def branch(self, statements, loop_depth):
        before_env, before_symbols = self.env.copy(), self.symbols.copy()
        start = len(self.invalidated_events)
        terminal = self.block(statements, loop_depth)
        self.env, self.symbols = before_env, before_symbols
        for name in self.invalidated_events[start:]:
            self.env.pop(name, None)
        return terminal

    def block(self, statements, loop_depth=0):
        for node in statements:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.import_stmt(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.env.pop(node.name, None)
                self.symbols.pop(node.name, None)
                if isinstance(node, ast.ClassDef) and any(self.symbol(b) == 'pydantic.BaseModel' for b in node.bases):
                    self.symbols[node.name] = 'known_model'
                if isinstance(node, ast.ClassDef) and any(_terminal(b) in {'TestCase', 'IsolatedAsyncioTestCase'} for b in node.bases):
                    self.limit('original_source_TestCase_evidence')
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                flow = self.expr(value)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    factory = self.symbol(value.func) if isinstance(value, ast.Call) else ''
                    self.bind(target, flow, node.lineno)
                    if isinstance(target, ast.Name) and factory == 'pydantic.TypeAdapter':
                        self.symbols[target.id] = 'known_adapter'
                    elif isinstance(target, ast.Name) and factory == 'openevals.llm.create_llm_as_judge':
                        self.symbols[target.id] = 'known_judge'
                    elif isinstance(value, ast.Name) and self.symbol(value):
                        self.limit('schema_or_factory_alias_unassessed')
                if flow and value is not None and any(n is self.seed for n in ast.walk(value)):
                    self.output_form = 'assigned'
            elif isinstance(node, ast.AugAssign):
                self.expr(node.value)
                self.invalidate([node.target], 'unsupported_augmented_assignment')
            elif isinstance(node, ast.Delete):
                self.invalidate(node.targets, 'deleted_binding')
            elif isinstance(node, ast.Assert):
                flow = self.predicate(node.test)
                self.sink(node, flow, 'assertion', 'assertion_failure')
            elif isinstance(node, ast.Expr):
                self.expr(node.value)
                if node.value is self.seed or isinstance(node.value, ast.Await) and node.value.value is self.seed:
                    self.output_form = 'discarded'
            elif isinstance(node, ast.Return):
                self.expr(node.value)
                if node.value is self.seed or isinstance(node.value, ast.Await) and node.value.value is self.seed:
                    self.output_form = 'returned_to_caller'
                    self.limit('caller_flow_unassessed')
                return True
            elif isinstance(node, ast.Raise):
                self.expr(node.exc)
                return True
            elif isinstance(node, (ast.Break, ast.Continue)):
                return True
            elif isinstance(node, ast.If):
                flow = self.predicate(node.test)
                truth = _literal_truth(node.test)
                terminal_paths = []
                for body, active in ((node.body, truth is not False), (node.orelse, truth is not True)):
                    if not active:
                        continue
                    if body and flow:
                        first = body[0]
                        if isinstance(first, ast.Raise):
                            self.sink(node, flow, 'rejection_branch', 'raise')
                        elif isinstance(first, ast.Return) and self.empty_return(first.value):
                            self.sink(node, flow, 'early_exit_candidate', 'replacement_return')
                        elif isinstance(first, ast.Continue) and loop_depth and self.same_seed_loop(node):
                            self.sink(node, flow, 'retry_branch', 'retry_continue')
                    terminal_paths.append(bool(self.branch(body, loop_depth)))
                if terminal_paths and all(terminal_paths):
                    return True
                if truth is None:
                    self.invalidate(node.body + node.orelse, 'unsupported_merge')
                    self.limit('unsupported_merge')
                else:
                    # A single literal path is safe, but already collected its sinks above.
                    self.invalidate(node.body if truth else node.orelse, 'unsupported_merge')
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                if isinstance(node, ast.While):
                    self.expr(node.test)
                    if _literal_truth(node.test) is False:
                        self.branch(node.orelse, loop_depth)
                        continue
                else:
                    self.expr(node.iter)
                saved_env, saved_symbols = self.env.copy(), self.symbols.copy()
                invalidation_start = len(self.invalidated_events)
                self.invalidate(node.body + ([node.target] if hasattr(node, 'target') else []), 'unsupported_merge')
                self.branch(node.body, loop_depth + 1)
                self.env, self.symbols = saved_env, saved_symbols
                for name in self.invalidated_events[invalidation_start:]:
                    self.env.pop(name, None)
                self.branch(node.orelse, loop_depth)
                self.invalidate([node], 'unsupported_merge')
                self.limit('unsupported_merge')
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    self.expr(item.context_expr)
                    if item.optional_vars:
                        self.bind(item.optional_vars, _Flow(), node.lineno)
                self.branch(node.body, loop_depth)
                self.invalidate(node.body, 'unsupported_merge')
                self.limit('unsupported_with_exit_flow')
            elif isinstance(node, (ast.Try, ast.TryStar)):
                self.branch(node.body, loop_depth)
                for handler in node.handlers:
                    saved = self.env.copy(), self.symbols.copy()
                    invalidation_start = len(self.invalidated_events)
                    self.invalidate(node.body, 'unsupported_merge')
                    if handler.name:
                        self.env.pop(handler.name, None)
                        self.symbols.pop(handler.name, None)
                    self.branch(handler.body, loop_depth)
                    self.env, self.symbols = saved
                    for name in self.invalidated_events[invalidation_start:]:
                        self.env.pop(name, None)
                self.invalidate([node], 'unsupported_merge')
                self.branch(node.orelse, loop_depth)
                self.branch(node.finalbody, loop_depth)
                self.limit('unsupported_merge')
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                for name in node.names:
                    self.env.pop(name, None)
                    self.symbols.pop(name, None)
                self.limit('nonlocal_binding_unassessed')
            elif isinstance(node, ast.Pass):
                pass
            else:
                self.invalidate([node], 'unsupported_statement:' + type(node).__name__)
                self.limit('unsupported_statement:' + type(node).__name__)

    def same_seed_loop(self, node):
        ancestors = set()
        cursor = self.seed
        while cursor in self.parents:
            cursor = self.parents[cursor]
            if isinstance(cursor, (ast.For, ast.AsyncFor, ast.While)):
                ancestors.add(cursor)
            if cursor is self.scope:
                break
        cursor = node
        while cursor in self.parents:
            cursor = self.parents[cursor]
            if isinstance(cursor, (ast.For, ast.AsyncFor, ast.While)):
                return cursor in ancestors and self.seed_seen
            if cursor is self.scope:
                break
        return False

    @staticmethod
    def empty_return(node):
        if node is None:
            return True
        if isinstance(node, ast.Constant):
            return node.value is None or node.value is False or node.value == ''
        return (isinstance(node, (ast.List, ast.Tuple, ast.Set)) and not node.elts or
                isinstance(node, ast.Dict) and not node.keys)


def analyze_source(source: str, calls: list[dict]) -> list[dict]:
    calls = _unique_calls(calls)
    results = []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return [dict(call_id=c['call_id'], status='parse_error', output_form='unassessed',
                     response_variables=[], checks=[], routes=[], limitations=[f'parse_error:{exc}']) for c in calls]
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    call_index = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_index[(node.lineno, node.col_offset, _terminal(node.func))].append(node)
    for call in calls:
        result = dict(call_id=call['call_id'], status='ok', output_form='unassessed',
                      response_variables=[], checks=[], routes=[], limitations=[])
        terminal = re.findall(r'[A-Za-z_]\w*', str(call['pattern']))
        matches = call_index.get((call['call_line'], call['call_col'], terminal[-1] if terminal else ''), [])
        if len(matches) != 1:
            result['status'] = 'ambiguous_seed' if len(matches) > 1 else 'seed_missing'
            result['limitations'] = [result['status']]
            results.append(result)
            continue
        seed = matches[0]
        scope = parents.get(seed)
        while scope is not None and not isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                                           ast.ClassDef, ast.Lambda, ast.ListComp, ast.SetComp,
                                                           ast.DictComp, ast.GeneratorExp)):
            scope = parents.get(scope)
        if isinstance(scope, (ast.ClassDef, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            result['status'] = 'unsupported_scope'
            result['limitations'] = ['unsupported_execution_scope']
            results.append(result)
            continue
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cursor = seed
            while cursor in parents and parents[cursor] is not scope:
                cursor = parents[cursor]
            if cursor not in scope.body:
                result['status'] = 'unsupported_scope'
                result['limitations'] = ['function_definition_time_expression_unassessed']
                results.append(result)
                continue
        context = []
        cursor = seed
        while cursor in parents:
            cursor = parents[cursor]
            if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)) and cursor.name.startswith('test'):
                context.append('test_named_function:' + cursor.name)
            if isinstance(cursor, ast.ClassDef) and any(_terminal(b) in {'TestCase', 'IsolatedAsyncioTestCase'} for b in cursor.bases):
                context.append('TestCase_class:' + cursor.name)
        result['source_test_context'] = sorted(context)
        worker = _Local(source, tree, parents, seed, call, scope)
        # Only prior module-level imports/classes resolve global validation APIs.
        if not isinstance(scope, ast.Module):
            for statement in tree.body:
                if statement.lineno >= scope.lineno:
                    break
                if isinstance(statement, (ast.Import, ast.ImportFrom)):
                    worker.import_stmt(statement)
                elif isinstance(statement, ast.ClassDef):
                    if any(worker.symbol(base) == 'pydantic.BaseModel' for base in statement.bases):
                        worker.symbols[statement.name] = 'known_model'
                else:
                    worker.invalidate([statement], 'global_symbol_resolution_unassessed')
            local_names = _names_written(scope.body)
            local_names.update(a.arg for a in scope.args.posonlyargs + scope.args.args + scope.args.kwonlyargs)
            local_names.update(a.arg for a in (scope.args.vararg, scope.args.kwarg) if a is not None)
            for name in local_names:
                worker.symbols.pop(name, None)
            # Local builtins are unsupported rather than presumed pure.
            worker.shadowed_builtins = (local_names | _names_written(tree.body)) & {'len', 'str', 'bool', 'isinstance'}
        else:
            worker.shadowed_builtins = _names_written(tree.body) & {'len', 'str', 'bool', 'isinstance'}
        for evidence in context:
            worker.limit('original_source_test_context:' + evidence)
        if any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr) for n in ast.walk(tree)):
            worker.limit('runnable_chain_flow_unassessed')
        worker.block(scope.body)
        if not worker.seed_seen:
            worker.limit('seed_not_reached_in_supported_local_flow')
        if worker.output_form == 'unassessed' and worker.seed_seen:
            worker.output_form = 'nested_expression'
            worker.limit('nested_expression_flow_bounded')
        result.update(output_form=worker.output_form, response_variables=sorted(worker.response_variables),
                      checks=sorted(worker.checks, key=lambda r: (r['sink_line'], r['check_kind'])),
                      routes=sorted(worker.routes, key=lambda r: (r['sink_line'], r['route_kind'])),
                      limitations=sorted(worker.limitations))
        results.append(result)
    return results


def corroborate_slices(checks: list[dict], hits: list[dict],
                       pdg_lines: dict[int, set[int]]) -> list[dict]:
    """Joint line membership only; never a verified DDG/CDG path."""
    by_call = defaultdict(dict)
    for hit in hits:
        item = dict(hit)
        item['slice_id'] = _integer(item['slice_id'], 'slice_id')
        key = (item['slice_id'], str(item['variable']), str(item['role']), str(item['parses']))
        by_call[item['call_id']][key] = item
    output = []
    for check in checks:
        row = dict(check)
        relevant = list(by_call.get(row['call_id'], {}).values())
        ids = sorted({h['slice_id'] for h in relevant})
        call_line, sink_line = int(row['call_line']), int(row['sink_line'])
        shared = {sid for sid in ids if {call_line, sink_line} <= pdg_lines.get(sid, set())}
        variables = set(row.get('response_variables', []))
        response = {h['slice_id'] for h in relevant if h['slice_id'] in shared and
                    h['role'] == 'binding' and h['variable'] in variables}
        if call_line == sink_line:
            status = 'same_line_source_only'
            response = set()
        elif response:
            status = 'same_response_slice'
        elif shared:
            status = 'shared_slice_unresolved_role'
        elif ids:
            status = 'no_joint_slice'
        else:
            status = 'no_saved_slice'
        row.update(joern_status=status, slice_ids=ids, shared_slice_ids=sorted(shared),
                   response_slice_ids=sorted(response),
                   slice_variables=sorted({str(h['variable']) for h in relevant}),
                   slice_parse_flags=[{'slice_id': sid, 'parses': sorted({str(h['parses']) for h in relevant if h['slice_id'] == sid})} for sid in ids])
        output.append(row)
    return output


def _safe_source(repos_dir, row):
    root = Path(repos_dir).absolute()
    slug, rel = str(row['repo_slug']), str(row['rel_path'])
    for label, value in (('repo_slug', slug), ('rel_path', rel)):
        posix = PurePosixPath(value.replace('\\', '/'))
        win = PureWindowsPath(value)
        if (not value or '\x00' in value or win.drive or win.root or posix.is_absolute()
                or '..' in posix.parts or ':' in value or any(part.endswith(('.', ' ')) for part in posix.parts)):
            raise InputError(f'unsafe source {label}: {value!r}')
    if len(PurePosixPath(slug.replace('\\', '/')).parts) != 1:
        raise InputError(f'unsafe repo_slug: {slug!r}')
    if slug != str(row.get('repo', '')).replace('/', '_'):
        raise InputError(f'noncanonical repo_slug {slug!r} for {row.get("repo")!r}')
    # Reject links/junctions on every existing component, including the root itself.
    candidate = root / slug / Path(rel.replace('\\', '/'))
    for component in [root, *root.parents]:
        if component.is_symlink() or getattr(component, 'is_junction', lambda: False)():
            raise InputError(f'unsafe source root link/junction: {component}')
    current = root
    for part in (slug, *PurePosixPath(rel.replace('\\', '/')).parts):
        current = current / part
        if current.is_symlink() or getattr(current, 'is_junction', lambda: False)():
            raise InputError(f'unsafe source link/junction: {current}')
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise InputError(f'unsafe source escape: {candidate}')
    return candidate


def _saved_slices(root, calls):
    """Stream large tables; only selected calls' hit/index/line rows are retained."""
    wanted = {c['call_id']: c for c in calls}
    hits, unique = [], {}
    for raw in _rows(root / 'slice_call_hits.csv', HIT_HEADERS):
        if raw['call_id'] not in wanted:
            continue
        hit = dict(raw)
        hit['slice_id'] = _integer(hit['slice_id'], 'slice_call_hits.csv slice_id')
        hit['call_line'] = _integer(hit['call_line'], 'slice_call_hits.csv call_line', 1)
        call = wanted[hit['call_id']]
        for field, expected in [('call_repo', call['repo']), ('repo', call['repo_slug']),
                                ('rel_path', call['rel_path']), ('code_kind', call['code_kind']),
                                ('call_line', call['call_line'])]:
            if hit[field] != expected:
                raise InputError(f'slice_call_hits.csv {hit["call_id"]}: conflicting {field}')
        key = (hit['call_id'], hit['slice_id'])
        if key in unique and unique[key] != hit:
            raise InputError(f'slice_call_hits.csv: conflicting hit identity {key}')
        unique[key] = hit
    hits = [unique[key] for key in sorted(unique)]
    wanted_ids = {h['slice_id'] for h in hits}
    index = {}
    for raw in _rows(root / 'slice_index.csv', INDEX_HEADERS):
        sid = _integer(raw['slice_id'], 'slice_index.csv slice_id')
        if sid not in wanted_ids:
            continue
        item = dict(raw, slice_id=sid)
        if sid in index and index[sid] != item:
            raise InputError(f'slice_index.csv: conflicting slice identity {sid}')
        index[sid] = item
    for hit in hits:
        sid = hit['slice_id']
        if sid not in index:
            raise InputError(f'slice_index.csv: missing slice_id {sid}')
        for key in ('repo', 'rel_path', 'code_kind', 'variable', 'parses'):
            if hit[key] != index[sid][key]:
                raise InputError(f'slice_id {sid}: index/hit disagree on {key}')
    lines = defaultdict(set)
    for raw in _rows(root / 'slice_pdg_lines.csv.gz', ['slice_id', 'line']):
        sid = _integer(raw['slice_id'], 'slice_pdg_lines.csv.gz slice_id')
        if sid in wanted_ids:
            lines[sid].add(_integer(raw['line'], f'slice_id {sid} line', 1))
    return hits, lines


def _write_csv(path, fields, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (list, dict)) else int(value)
                             if isinstance(value, bool) else value for key, value in row.items()})


def _write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def _summary(calls, checks, routes, results, subset, hits):
    by_id = {row['call_id']: row for row in calls}
    check_ids = {r['call_id'] for r in checks}
    strong_ids = {r['call_id'] for r in checks if r['joern_status'] == 'same_response_slice'}
    validator_ids = {r['call_id'] for r in checks if r['check_kind'] == 'schema_validation'}
    interventions = {'assertion_failure', 'raise', 'retry_continue', 'replacement_return'}
    runtime_ids = {r['call_id'] for r in checks if by_id[r['call_id']]['code_kind'] == 'production' and r['intervention'] in interventions}
    parser_ids = {r['call_id'] for r in routes if r['route_kind'] == 'parser_only'}
    eval_ids = {r['call_id'] for r in routes if r['route_kind'] == 'evaluation_only'}
    named = {'with_source_check': check_ids, 'with_response_slice_check': strong_ids,
             'with_validator_route': validator_ids, 'with_runtime_intervention': runtime_ids,
             'with_parser_only_route': parser_ids, 'with_evaluation_only_route': eval_ids}
    def count(ids):
        return dict(denominator=len(ids), **{name: len(ids & values) for name, values in named.items()})
    all_ids = set(by_id)
    strata = {kind: count({c['call_id'] for c in calls if c['code_kind'] == kind}) for kind in ('production', 'test', 'test_support')}
    totals = count(all_ids)
    def rates(counts):
        n = counts['denominator']
        return {key: {'denominator': n, 'value': value / n if n else None}
                for key, value in counts.items() if key.startswith('with_')}
    methods = {}
    for kind in strata:
        selected = [c for c in calls if c['code_kind'] == kind]
        def keys(ids):
            return {(c['repo'], c['rel_path'], c.get('enclosing_qname', '')) for c in selected
                    if c['call_id'] in ids and c.get('enclosing_qname', '')}
        methods[kind] = {'with_saved_enclosing_name': len(keys(all_ids)),
                         'with_source_check': len(keys(check_ids)),
                         'with_response_slice_check': len(keys(strong_ids)),
                         'calls_without_enclosing_name': sum(not c.get('enclosing_qname') for c in selected)}
    test_context = {c['call_id'] for c in calls if c.get('source_test_context')}
    mislabeled = test_context & {c['call_id'] for c in calls if c['code_kind'] == 'production'}
    limitations = sorted({item for result in results for item in result['limitations']} | {
        'Static invocation-site outputs, not runtime responses or validated guardrail prevalence.',
        'Joint saved slice lines are not verified DDG/CDG paths; full per-edge paths were not persisted.',
        'Bounded local source flow is incomplete; negative detections do not imply absent guardrails.',
        'Validator routes do not establish observed failure handling or protected/accepted output.',
        'Original code_kind strata are retained even where original source has obvious test context.',
        'Builtin and receiver dispatch can remain unresolved; candidates require manual precision review.'})
    return dict(subset=bool(subset), calls=totals, checks={'total': len(checks)},
                by_code_kind=strata, rates={'calls': rates(totals), 'by_code_kind': {k: rates(v) for k, v in strata.items()}},
                source_statuses=dict(sorted(Counter(c['source_status'] for c in calls).items())),
                joern_statuses=dict(sorted(Counter(r['joern_status'] for r in checks).items())),
                mechanisms=dict(sorted(Counter(r['mechanism'] for r in checks).items())),
                interventions=dict(sorted(Counter(r['intervention'] for r in checks).items())),
                runtime_intervention_calls_by_kind={kind: len({r['call_id'] for r in checks if r['intervention'] == kind and r['call_id'] in runtime_ids}) for kind in sorted(interventions)},
                methods_by_code_kind=methods,
                production_labelled_test_context={'calls': len(mislabeled), 'with_source_check': len(mislabeled & check_ids),
                    'with_response_slice_check': len(mislabeled & strong_ids),
                    'runtime_intervention_candidates_without_obvious_test_context': len(runtime_ids - mislabeled)},
                saved_slices={'calls_with_hit': len({h['call_id'] for h in hits}),
                              'selected_hit_rows': len(hits), 'distinct_slices': len({h['slice_id'] for h in hits})},
                source_assessable_calls=sum(c['source_status'] == 'ok' and c['output_form'] != 'unassessed' for c in calls),
                limitations=limitations)


def _report(summary, calls, checks, routes):
    n = summary['calls']['denominator']
    def fraction(num, denominator=n):
        return f'{num}/{denominator}' + (f' ({100 * num / denominator:.2f}%)' if denominator else ' (rate unavailable)')
    lines = ['# Response-check candidates', '',
             f"Source-check candidates: {fraction(summary['calls']['with_source_check'])} static invocation sites. "
             f"Response-slice-supported candidates: {fraction(summary['calls']['with_response_slice_check'])}.", '',
             'These are static invocation-site outputs, not the number of runtime responses, validated guardrail prevalence, or verified dependence paths.', '',
             '| Saved code kind | Invocation sites | Source-check sites | Response-slice-supported sites | Validator-route sites | Runtime-intervention candidate sites |',
             '|---|---:|---:|---:|---:|---:|']
    for kind, counts in summary['by_code_kind'].items():
        d = counts['denominator']
        lines.append('| ' + kind + ' | ' + str(d) + ' | ' + ' | '.join(fraction(counts[key], d) for key in
            ('with_source_check', 'with_response_slice_check', 'with_validator_route', 'with_runtime_intervention')) + ' |')
    lines += ['', f"Saved hit slices are available for {fraction(summary['saved_slices']['calls_with_hit'])}; "
              f"{summary['saved_slices']['distinct_slices']} distinct relevant slices. "
              f"Original-source assessable outputs: {fraction(summary['source_assessable_calls'])}.",
              f"Source-only check sites (no strong slice-supported check): {summary['calls']['with_source_check'] - summary['calls']['with_response_slice_check']}.", '',
              'Mechanism counts (check rows): ' + json.dumps(summary['mechanisms'], sort_keys=True) + '.',
              'Intervention counts (check rows): ' + json.dumps(summary['interventions'], sort_keys=True) + '.',
              'Production intervention counts (unique call IDs per kind): ' + json.dumps(summary['runtime_intervention_calls_by_kind'], sort_keys=True) + '.', '',
              'Validation routes measure source-resolved data routing, separately from observed failure handling. '
              'Early replacement-return candidates are weaker than assertion/raise candidates; retry-continue does not prove another invocation executes. '
              'Test assertions and test support are separate from production runtime candidates.', '',
              'Canonical production labels with obvious test context (sensitivity): ' + json.dumps(summary['production_labelled_test_context'], sort_keys=True) + '.',
              'Canonical code_kind is unchanged. Unconventional tests.py files can be misclassified; these flags use the innermost function/class source context.', '',
              'Saved enclosing-method counts (exclude blank names, no new method-eligibility rule): ' + json.dumps(summary['methods_by_code_kind'], sort_keys=True) + '.', '',
              '## Unresolved evidence', '',
              'Source statuses: ' + json.dumps(summary['source_statuses'], sort_keys=True) + '.',
              'Joern statuses: ' + json.dumps(summary['joern_statuses'], sort_keys=True) + '.',
              f'Nonqualifying parser/evaluation/unresolved routes: {len(routes)} rows.', '',
              *['- ' + value for value in summary['limitations']], '',
              '## Deterministic review sample', '',
              'Manual review must establish precision and the property being checked before any guardrail-prevalence claim.', '']
    rows = sorted(checks, key=lambda r: (r['call_id'], r['sink_line'], r['check_kind']))[:10]
    if not rows:
        lines.append('No source check candidates in this selection.')
    for row in rows:
        excerpt = row['sink_source'].replace('\n', ' ')[:220]
        lines.append(f"- {row['call_id']} → line {row['sink_line']}: {row['check_kind']} / {row['joern_status']}; {excerpt}")
    return '\n'.join(lines) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--artifacts-dir', type=Path)
    parser.add_argument('--repos-dir', type=Path)
    parser.add_argument('--repo', action='append', nargs='+', default=[])
    args = parser.parse_args(argv)
    output = args.output_dir.absolute()
    # The refusal must happen before even a patched loader or saved-table read.
    if output.exists() or output.is_symlink():
        print(f'output exists: {output}', file=sys.stderr)
        return 1
    root = args.artifacts_dir if args.artifacts_dir is not None else paths.ARTIFACTS_DIR
    repos = args.repos_dir if args.repos_dir is not None else paths.REPOS_DIR
    saved_names = ['slice_calls_by_call.csv', 'slice_call_hits.csv', 'slice_index.csv', 'slice_pdg_lines.csv.gz']
    try:
        before = {str((root / name).absolute()): _hash(root / name) for name in saved_names}
        _CANONICAL_INPUT_HASHES.clear()
        selected = _unique_calls(load_calls(root))
        before.update(_CANONICAL_INPUT_HASHES)
        filters = {name for group in args.repo for name in group}
        if filters:
            selected = [row for row in selected if row['repo'] in filters]
        hits, pdg = _saved_slices(root, selected)
        diagnostics, results, source_hashes = [], [], {}
        grouped = defaultdict(list)
        for call in selected:
            for key in CALL_HEADERS:
                if key not in call:
                    raise InputError(f'{call["call_id"]}: missing {key}')
            if call['code_kind'] not in {'production', 'test', 'test_support'}:
                raise InputError(f'{call["call_id"]}: unknown code_kind {call["code_kind"]!r}')
            try:
                source_path = _safe_source(repos, call)
            except InputError as exc:
                results.append(dict(call_id=call['call_id'], status='unsafe_source', output_form='unassessed',
                                    response_variables=[], checks=[], routes=[], limitations=[str(exc)]))
                continue
            grouped[source_path].append(call)
        for source_path in sorted(grouped, key=str):
            group = grouped[source_path]
            try:
                before_source = _hash(source_path)
                raw = source_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != before_source:
                    raise InputError(f'changed inspected source: {source_path}')
                source_hashes[str(source_path.absolute())] = before_source
                encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
                source = raw.decode(encoding)
            except FileNotFoundError:
                results.extend(dict(call_id=c['call_id'], status='missing_source', output_form='unassessed',
                                    response_variables=[], checks=[], routes=[], limitations=[f'missing_source:{source_path}']) for c in group)
                continue
            except (UnicodeError, SyntaxError, LookupError) as exc:
                results.extend(dict(call_id=c['call_id'], status='parse_error', output_form='unassessed',
                                    response_variables=[], checks=[], routes=[], limitations=[f'source_decode_error:{exc}']) for c in group)
                continue
            results.extend(analyze_source(source, group))
        by_id = {r['call_id']: r for r in results}
        call_map = {c['call_id']: c for c in selected}
        checks = corroborate_slices([row for result in results for row in result['checks']], hits, pdg)
        checks.sort(key=lambda r: (r['call_id'], r['sink_line'], r['check_kind']))
        routes = sorted([row for result in results for row in result['routes']], key=lambda r: (r['call_id'], r['sink_line'], r['route_kind']))
        for row in checks + routes:
            row.update({key: call_map[row['call_id']][key] for key in ('repo', 'rel_path', 'code_kind')})
        calls_out = []
        checks_by_call, routes_by_call = defaultdict(list), defaultdict(list)
        for row in checks:
            checks_by_call[row['call_id']].append(row)
        for row in routes:
            routes_by_call[row['call_id']].append(row)
        for call in selected:
            result = by_id[call['call_id']]
            rows = checks_by_call[call['call_id']]
            calls_out.append(dict(call, source_status=result['status'], output_form=result['output_form'],
                                  response_variables=result['response_variables'], source_check_count=len(rows),
                                  slice_supported_check_count=sum(r['joern_status'] == 'same_response_slice' for r in rows),
                                  route_count=len(routes_by_call[call['call_id']]),
                                  source_test_context=result.get('source_test_context', [])))
            for detail in result['limitations']:
                diagnostics.append(dict(call_id=call['call_id'], repo=call['repo'], rel_path=call['rel_path'],
                                        kind=detail.split(':', 1)[0], detail=detail))
        summary = _summary(calls_out, checks, routes, results, bool(args.repo), hits)
        after = {name: _hash(name) for name in before}
        for name, digest in before.items():
            if after[name] != digest:
                raise InputError(f'changed consumed input: {name}')
        for name, digest in source_hashes.items():
            # Recheck the source path policy as well as bytes to reject replacement links.
            group = grouped[Path(name)]
            _safe_source(repos, group[0])
            if _hash(name) != digest:
                raise InputError(f'changed inspected source: {name}')
        try:
            revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent.parent,
                                      check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            revision = 'unavailable'
        # Exclusive creation is the final race-safe refusal; never overwrite files.
        output.mkdir(parents=True, exist_ok=False)
        _write_csv(output / 'calls.csv', CALL_OUTPUT + ['enclosing_qname', 'source_test_context'], calls_out)
        _write_csv(output / 'checks.csv', CHECK_OUTPUT, checks)
        _write_csv(output / 'routes.csv', ROUTE_OUTPUT, routes)
        _write_csv(output / 'diagnostics.csv', DIAGNOSTIC_OUTPUT, sorted(diagnostics, key=lambda r: (r['call_id'], r['kind'], r['detail'])))
        _write_json(output / 'summary.json', summary)
        with (output / 'REPORT.md').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(_report(summary, calls_out, checks, routes))
        output_names = ['calls.csv', 'checks.csv', 'routes.csv', 'diagnostics.csv', 'summary.json', 'REPORT.md']
        def input_name(name):
            path = Path(name)
            if path in {(root / saved).absolute() for saved in saved_names}:
                return path.relative_to(root.absolute()).as_posix()
            try:
                return path.relative_to(paths.REPO_ROOT).as_posix()
            except ValueError:
                return path.as_posix()
        manifest = dict(inputs={input_name(name): {'sha256_before': before[name], 'sha256_after': after[name]} for name in sorted(before)},
                        inspected_sources=dict(sorted(source_hashes.items())),
                        revision=revision, module_sha256=_hash(Path(__file__)), python=sys.version,
                        options={'artifacts_dir': str(root.absolute()), 'repos_dir': str(repos.absolute()),
                                 'output_dir': str(output), 'repos': sorted(filters)},
                        outputs={name: _hash(output / name) for name in output_names})
        # Inputs may change while writing outputs; a success manifest requires final equality.
        for name, digest in manifest['inspected_sources'].items():
            _safe_source(repos, grouped[Path(name)][0])
            if _hash(name) != digest:
                raise InputError(f'changed inspected source: {name}')
        for name, digest in before.items():
            if _hash(name) != digest:
                raise InputError(f'changed consumed input: {name}')
        _write_json(output / 'manifest.json', manifest)
        return 0
    except (OSError, InputError, csv.Error, UnicodeError, KeyError, TypeError, ValueError) as exc:
        print(f'response guardrail analysis failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
