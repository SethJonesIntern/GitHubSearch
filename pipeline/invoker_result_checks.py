"""Read-only static result/check candidates from saved study artifacts.

Run: py -3.14 -B -m pipeline.invoker_result_checks --output-dir NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import ntpath
from pathlib import Path
import subprocess
import sys
import tokenize
from collections import Counter, OrderedDict, defaultdict

VALUE_ASSERTS = {
    "assertEqual": 2, "assertNotEqual": 2, "assertTrue": 1, "assertFalse": 1,
    "assertIs": 2, "assertIsNot": 2, "assertIsNone": 1, "assertIsNotNone": 1,
    "assertIn": 2, "assertNotIn": 2, "assertIsInstance": 2, "assertNotIsInstance": 2,
    "assertGreater": 2, "assertGreaterEqual": 2, "assertLess": 2, "assertLessEqual": 2,
    "assertAlmostEqual": 2, "assertNotAlmostEqual": 2, "assertRegex": 2,
    "assertNotRegex": 2, "assertCountEqual": 2,
}
LIMITS = [
    "Static candidates do not establish execution or correctness.",
    "Test calls match final names only; dispatch, mocks and execution remain unresolved.",
    "Only supported straight-line local return-value expressions and value assertions are assessed.",
    "Saved Joern membership is corroboration only, not verified dependence; scaffolding is not evidence.",
    "Production and test chains are separate; no cross-method value flow is established.",
    "Absence of detection is not absence of checking; missing/invalid source remains unassessed.",
]


def normalized_relative(value: str, slug: str) -> str:
    value = value.replace("\\", "/")
    if not value or value.startswith("/") or ntpath.splitdrive(value)[0] or ".." in value.split("/"):
        raise ValueError("Unsafe source path")
    prefix = slug.replace("\\", "/") + "/"
    if value.startswith(prefix):
        value = value[len(prefix):]
    parts = [p for p in value.split("/") if p and p != "."]
    if not parts:
        raise ValueError("Unsafe source path")
    return "/".join(parts)


def _local_walk(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _local_walk(child)


def _function(tree, line):
    return next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.lineno == line), None)


def _calls(function):
    return [n for stmt in function.body for n in _local_walk(stmt) if isinstance(n, ast.Call)]


def _bound_names(node):
    """Names potentially rebound in this scope, excluding nested scope bodies."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    names = set()
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        names.add(node.id)
    elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
        names.add(node.name)
    elif isinstance(node, ast.MatchMapping) and node.rest:
        names.add(node.rest)
    elif isinstance(node, ast.ExceptHandler) and node.name:
        names.add(node.name)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    for child in ast.iter_child_nodes(node):
        names.update(_bound_names(child))
    return names


def _final_name(call):
    return call.func.attr if isinstance(call.func, ast.Attribute) else (
        call.func.id if isinstance(call.func, ast.Name) else "")


def analyze_local_checks(source: str, definition_line: int, seed_call_lines: set[int]) -> dict:
    result = dict(status="ok", detail="", checks=[], limitations=[])
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError) as exc:
        result.update(status="parse_error", detail=str(exc))
        return result
    function = _function(tree, definition_line)
    if function is None:
        result.update(status="definition_missing", detail=f"No function definition at {definition_line}")
        return result
    by_line = defaultdict(list)
    for call in _calls(function):
        by_line[call.lineno].append(call)
    if any(len(by_line[line]) > 1 for line in seed_call_lines):
        result.update(status="ambiguous_seed", detail="Multiple calls begin on a seed line")
        return result
    if not seed_call_lines:
        return result
    for line in sorted(seed_call_lines):
        if not by_line[line]:
            result["limitations"].append(f"No call begins on seed line {line}")
    seeds = {id(by_line[line][0]): line for line in seed_call_lines if by_line[line]}
    env = {}

    def merge(parts):
        combined = {}
        for part in parts:
            for seed, (trace, names) in part.items():
                lines_out, names_out = combined.setdefault(seed, (set(), set()))
                lines_out.update(trace)
                names_out.update(names)
        return combined

    def expr(node):
        if node is None:
            return {}
        if isinstance(node, ast.Name):
            return env.get(node.id, {})
        if isinstance(node, ast.Call):
            if id(node) in seeds:
                line = seeds[id(node)]
                return {line: ({line}, set())}
            if isinstance(node.func, ast.Name) and node.func.id in {"len", "isinstance", "bool", "str"}:
                result["limitations"].append("Builtin names may be shadowed; calls are not resolved.")
                if (not node.args or isinstance(node.args[0], ast.Starred) or node.keywords
                        or len(node.args) != (2 if node.func.id == "isinstance" else 1)):
                    result["limitations"].append(f"Unsupported builtin signature at line {node.lineno}")
                    return {}
                return expr(node.args[0])
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"startswith", "endswith", "__contains__"}:
                result["limitations"].append("Predicate receiver types and dynamic method implementations are unresolved.")
                return expr(node.func.value)
            result["limitations"].append(f"Unsupported call-derived expression at line {node.lineno}")
            return {}
        if isinstance(node, (ast.Lambda, ast.NamedExpr, ast.ListComp, ast.SetComp,
                             ast.DictComp, ast.GeneratorExp, ast.Yield, ast.YieldFrom)):
            result["limitations"].append(f"Unsupported expression at line {node.lineno}")
            return {}
        if isinstance(node, (ast.Constant, ast.Load, ast.operator, ast.unaryop,
                             ast.boolop, ast.cmpop, ast.expr_context)):
            return {}
        if isinstance(node, (ast.Await, ast.Attribute, ast.Subscript, ast.BinOp,
                             ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Tuple, ast.List,
                             ast.Set, ast.Dict, ast.JoinedStr, ast.FormattedValue, ast.Slice)):
            return merge(expr(child) for child in ast.iter_child_nodes(node))
        result["limitations"].append(f"Unsupported expression at line {getattr(node, 'lineno', '?')}")
        return {}

    def record(node, kind, values):
        for seed, (trace, names) in sorted(merge(expr(v) for v in values).items()):
            result["checks"].append(dict(call_line=seed, check_line=node.lineno,
                check_kind=kind, check_source=ast.get_source_segment(source, node) or "",
                trace_lines=sorted(trace | {node.lineno}), variables=sorted(names),
                evidence="syntactic_local_result_to_assert_candidate"))

    for stmt in function.body:
        if any(isinstance(node, ast.NamedExpr) for node in _local_walk(stmt)):
            # Evaluation order within unsupported assignment expressions is not
            # modeled; abstain for the whole statement after forgetting writes.
            for name in _bound_names(stmt):
                env.pop(name, None)
            result["limitations"].append(f"Unsupported NamedExpr statement at line {stmt.lineno}")
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                break
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            env.pop(stmt.name, None)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            associated = expr(stmt.value)
            for target in targets:
                if isinstance(target, ast.Name):
                    env[target.id] = {seed: (lines | {stmt.lineno}, names | {target.id})
                                      for seed, (lines, names) in associated.items()}
                else:
                    for node in ast.walk(target):
                        if isinstance(node, ast.Name):
                            env.pop(node.id, None)
                    result["limitations"].append(f"Unsupported assignment target at line {stmt.lineno}")
        elif isinstance(stmt, ast.Assert):
            record(stmt, "assert", [stmt.test])
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            func = call.func
            if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and func.value.id == "self" and func.attr in VALUE_ASSERTS):
                values = list(call.args[:VALUE_ASSERTS[func.attr]])
                operands = {"first", "second", "expr", "obj", "member", "container", "cls",
                            "text", "regex", "expected_regex"}
                values.extend(k.value for k in call.keywords if k.arg in operands)
                record(call, "unittest_assert", values)
            else:
                result["limitations"].append(f"Unassessed expression call at line {stmt.lineno}")
        elif isinstance(stmt, (ast.Pass, ast.Return, ast.Raise, ast.Break, ast.Continue)):
            if not isinstance(stmt, ast.Pass):
                break
        else:
            for name in _bound_names(stmt):
                env.pop(name, None)
            result["limitations"].append(f"Unsupported statement/flow {type(stmt).__name__} at line {stmt.lineno}")
    result["limitations"] = sorted(set(result["limitations"]))
    return result


def _rows(path, required):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for number, row in enumerate(reader, 2):
            if None in row or any(row[k] is None for k in required):
                raise ValueError(f"{path}:{number}: malformed row")
            yield row


def _integer(row, key, table, minimum=1):
    try:
        value = int(row[key])
        if value < minimum:
            raise ValueError()
        return value
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{table}: invalid {key} for {row.get('repo', row.get('slice_id', '?'))}") from exc


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True)
                             if isinstance(v, (list, dict)) else v for k, v in row.items()})


class _Sources:
    def __init__(self, root, diagnostic):
        self.root = root.resolve()
        self.cache = OrderedDict()
        self.hashes = {}
        self.diagnostic = diagnostic

    def get(self, repo, file):
        identity = (repo, file)
        if identity in self.cache:
            self.cache.move_to_end(identity)
            return self.cache[identity]
        slug = repo.replace("/", "_")
        rel = normalized_relative(file, slug)
        path = self.root / slug / rel
        current = self.root
        for part in (slug, *Path(rel).parts):
            current = current / part
            if current.is_symlink():
                raise ValueError(f"Unsafe source path: symlink {current}")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError(f"Unsafe source path: {path}")
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if str(path) in self.hashes and self.hashes[str(path)] != digest:
                raise ValueError(f"changed-input: source {path}")
            self.hashes[str(path)] = digest
            encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
            source = raw.decode(encoding)
            tree = ast.parse(source, filename=str(path))
            value = ("ok", source, tree)
        except FileNotFoundError:
            value = ("source_missing", "", None)
            self.diagnostic("source_missing", repo, file, "", str(path))
        except (SyntaxError, UnicodeError, LookupError) as exc:
            value = ("parse_error", "", None)
            self.diagnostic("parse_error", repo, file, "", str(exc))
        self.cache[identity] = value
        while len(self.cache) > 16 or sum(len(v[1]) for v in self.cache.values()) > 8 * 1024 * 1024:
            self.cache.popitem(last=False)
        return value


METHOD_FIELDS = ["repo", "file", "qname", "line", "line_end", "mapped_test", "mapped_tests",
    "exact_mapped_tests", "joined_mapped_tests", "joined_exact_mapped_tests", "file_associated",
    "import_file_associated", "source_status", "saved_calls", "saved_slices", "test_assessed_links",
    "production_source_ok", "test_candidate_count", "production_candidate_count", "shared_slice_candidate_count"]
LINK_FIELDS = ["repo", "file", "qname", "line", "line_end", "test_file", "test_method", "test_line",
    "test_line_end", "method_rule", "file_rule", "source_status", "matched_call_count", "candidate_count",
    "callee_resolution", "analysis_status", "limitations"]
CHECK_FIELDS = ["side", "repo", "file", "qname", "line", "line_end", "test_file", "test_method", "test_line",
    "test_line_end", "method_rule", "file_rule", "call_id", "call_line", "call_source", "check_line",
    "check_kind", "check_source", "trace_lines", "variables", "evidence", "callee_resolution", "joern_status",
    "slice_ids", "shared_slice_ids", "slice_variables", "slice_parse_flags", "limitations"]


def _run(args, output):
    names = ["method_test_baseline/methods.csv", "method_test_baseline/summary.json",
        "method_test_baseline/by_repo.csv", "pym2t_focal_tests.csv", "pym2t_focal_calls.csv",
        "slice_index.csv", "slice_call_hits.csv", "slice_pdg_lines.csv.gz"]
    inputs = {name: args.artifacts_dir / name for name in names}
    for path in inputs.values():
        if not path.is_file():
            raise ValueError(f"Missing input: {path}")
    before = {name: _sha(path) for name, path in inputs.items()}
    diagnostics = []

    def diagnostic(kind, repo="", file="", line="", detail=""):
        diagnostics.append(dict(kind=kind, repo=repo, file=file, line=line, detail=detail))

    methods, seen = {}, {}
    all_counts, mapped_counts, exact_counts = Counter(), Counter(), Counter()
    filters = set(args.repo or [])

    def valid_repo(repo):
        return len(repo.split("/")) == 2 and all(p and p not in (".", "..") and "\\" not in p
                                                    and ":" not in p for p in repo.split("/"))
    for repo in filters:
        if not valid_repo(repo):
            raise ValueError(f"Unsafe repo filter: {repo}")
    for row in _rows(inputs[names[0]], ["repo", "file", "qname", "line", "line_end", "group", "mapped_tests", "exact_mapped_tests"]):
        if row["group"] != "llm":
            continue
        repo = row["repo"]
        if not valid_repo(repo):
            raise ValueError(f"methods.csv: unsafe repo {repo}")
        file = normalized_relative(row["file"], repo.replace("/", "_"))
        line, end = (_integer(row, col, names[0]) for col in ("line", "line_end"))
        nmap, nexact = (_integer(row, col, names[0], 0) for col in ("mapped_tests", "exact_mapped_tests"))
        if end < line or not row["qname"] or nexact > nmap:
            raise ValueError(f"methods.csv: inconsistent baseline identity {(repo, file, line)}")
        key, identity = (repo, file, line), (row["qname"], end, nmap, nexact)
        if key in seen:
            if seen[key] != identity:
                raise ValueError(f"methods.csv: conflicting baseline identity {key}")
            continue
        seen[key] = identity
        all_counts[repo] += 1
        mapped_counts[repo] += nmap > 0
        exact_counts[repo] += nexact > 0
        if filters and repo not in filters:
            continue
        methods[key] = dict(repo=repo, file=file, qname=row["qname"], line=line, line_end=end,
            mapped_test=int(nmap > 0), mapped_tests=nmap, exact_mapped_tests=nexact, joined_mapped_tests=0,
            joined_exact_mapped_tests=0, file_associated=0, import_file_associated=0, source_status="not_assessed",
            saved_calls=0, saved_slices=0, test_assessed_links=0, production_source_ok=0,
            test_candidate_count=0, production_candidate_count=0, shared_slice_candidate_count=0)
    saved_summary = json.loads(inputs[names[1]].read_text(encoding="utf-8-sig"))
    try:
        saved = saved_summary["pooled_all_complete_repos"]["llm"]
        expected = (saved["methods"], saved["with_mapped_test"], saved["with_exact_test"])
    except (KeyError, TypeError) as exc:
        raise ValueError("method_test_baseline/summary.json: missing pooled_all_complete_repos.llm counts") from exc
    measured = (sum(all_counts.values()), sum(mapped_counts.values()), sum(exact_counts.values()))
    if measured != expected:
        raise ValueError(f"summary.json: baseline reconciliation failed: {measured} != {expected}")
    repo_rows = {}
    for row in _rows(inputs[names[2]], ["repo", "llm_methods", "llm_mapped", "llm_exact_mapped"]):
        repo = row["repo"]
        values = tuple(_integer(row, col, names[2], 0) for col in ("llm_methods", "llm_mapped", "llm_exact_mapped"))
        if repo in repo_rows and repo_rows[repo] != values:
            raise ValueError(f"by_repo.csv: conflicting repo {repo}")
        repo_rows[repo] = values
    for repo in set(all_counts) | set(repo_rows):
        if repo_rows.get(repo) != (all_counts[repo], mapped_counts[repo], exact_counts[repo]):
            raise ValueError(f"by_repo.csv: baseline reconciliation failed for {repo}")
    by_file = defaultdict(list)
    for key in methods:
        by_file[key[:2]].append(key)
    selected_repos = {k[0] for k in methods}
    links = {}
    for row in _rows(inputs[names[3]], ["repo", "focal_file", "focal_method", "focal_line", "focal_line_end", "test_file", "test_method", "test_line", "test_line_end", "method_rule", "file_rule"]):
        repo = row["repo"]
        if repo not in selected_repos or not row["focal_file"]:
            continue
        focal_file = normalized_relative(row["focal_file"], repo.replace("/", "_"))
        for key in by_file.get((repo, focal_file), []):
            methods[key]["file_associated"] = 1
            if row["file_rule"] == "import":
                methods[key]["import_file_associated"] = 1
        if not row["focal_method"]:
            continue
        key = (repo, focal_file, _integer(row, "focal_line", names[3]))
        if key not in methods:
            # Mappings to other methods in this file are outside our population.
            continue
        method = methods[key]
        if row["focal_method"] != method["qname"].split(".")[-1] or _integer(row, "focal_line_end", names[3]) != method["line_end"]:
            diagnostic("join_disagreement", repo, focal_file, key[2], "Focal method name/end disagrees with baseline")
            continue
        test_file = normalized_relative(row["test_file"], repo.replace("/", "_"))
        test_line, test_end = (_integer(row, col, names[3]) for col in ("test_line", "test_line_end"))
        if test_end < test_line or row["method_rule"] not in ("exact", "fuzzy"):
            raise ValueError(f"pym2t_focal_tests.csv: inconsistent test identity {repo} {test_file}:{test_line}")
        identity = (*key, test_file, test_line)
        link = {k: method[k] for k in ("repo", "file", "qname", "line", "line_end")}
        link.update(test_file=test_file, test_method=row["test_method"], test_line=test_line,
            test_line_end=test_end, method_rule=row["method_rule"], file_rule=row["file_rule"],
            source_status="not_assessed", matched_call_count=0, candidate_count=0,
            callee_resolution="name_only", analysis_status="not_assessed", limitations=[])
        if identity in links and links[identity] != link:
            raise ValueError(f"pym2t_focal_tests.csv: conflicting test link {identity}")
        links[identity] = link
    links_by_method = defaultdict(list)
    for identity, link in links.items():
        links_by_method[identity[:3]].append(link)
    for key, method in methods.items():
        relevant = links_by_method[key]
        method["joined_mapped_tests"] = len(relevant)
        method["joined_exact_mapped_tests"] = sum(v["method_rule"] == "exact" for v in relevant)
        if (len(relevant), method["joined_exact_mapped_tests"]) != seen[key][2:]:
            diagnostic("mapping_reconciliation", key[0], key[1], key[2],
                f"baseline mapped/exact={seen[key][2:]}; joined={(len(relevant), method['joined_exact_mapped_tests'])}")
    calls, call_owners = defaultdict(dict), {}
    for row in _rows(inputs[names[4]], ["repo", "file", "call_id", "call_line", "enclosing_qname", "framework"]):
        repo = row["repo"]
        if repo not in selected_repos:
            continue
        file = normalized_relative(row["file"], repo.replace("/", "_"))
        line = _integer(row, "call_line", names[4])
        candidates = [k for k in by_file.get((repo, file), []) if k[2] <= line <= methods[k]["line_end"]]
        if not candidates:
            continue
        key = min(candidates, key=lambda k: (methods[k]["line_end"] - k[2], -k[2]))
        cid = row["call_id"]
        if not cid:
            raise ValueError(f"pym2t_focal_calls.csv: blank call_id {key}")
        if cid in call_owners and call_owners[cid] != (key, line):
            raise ValueError(f"pym2t_focal_calls.csv: conflicting call_id {cid}")
        call_owners[cid] = (key, line)
        calls[key][cid] = line
    hits = defaultdict(set)
    for row in _rows(inputs[names[6]], ["slice_id", "repo", "rel_path", "code_kind", "method", "variable", "call_id", "call_repo", "call_code_kind", "call_line"]):
        cid = row["call_id"]
        if cid in call_owners:
            key, line = call_owners[cid]
            if (_integer(row, "call_line", names[6]) != line or row["call_repo"] != key[0]
                    or row["repo"] != key[0].replace("/", "_")
                    or normalized_relative(row["rel_path"], row["repo"]) != key[1]):
                raise ValueError(f"slice_call_hits.csv: conflicting call identity {cid}")
            hits[cid].add(_integer(row, "slice_id", names[6]))
    wanted = set().union(*hits.values()) if hits else set()
    slices = {}
    for row in _rows(inputs[names[5]], ["slice_id", "repo", "rel_path", "code_kind", "method", "variable", "parses"]):
        sid = _integer(row, "slice_id", names[5])
        if sid not in wanted:
            continue
        data = (row["repo"], row["rel_path"], row["variable"], row["parses"])
        if sid in slices and slices[sid] != data:
            raise ValueError(f"slice_index.csv: conflicting slice_id {sid}")
        slices[sid] = data
    for cid in hits:
        key, _ = call_owners[cid]
        for sid in sorted(hits[cid]):
            if sid not in slices:
                diagnostic("slice_join_missing", key[0], key[1], key[2], f"Missing slice_index entry {sid}")
                continue
            slug, rel, _, _ = slices[sid]
            if slug != key[0].replace("/", "_") or normalized_relative(rel, slug) != key[1]:
                raise ValueError(f"slice_index.csv: slice/call source disagreement {sid} {cid}")
    sources = _Sources(args.repos_dir, diagnostic)
    checks = []

    def analyze(key, side, identity, seeds, source, tree):
        line = identity["test_line"] if side == "test" else key[2]
        result = analyze_local_checks(source, line, seeds)
        if result["status"] != "ok":
            diagnostic(result["status"], key[0], identity.get("test_file", key[1]), line, result["detail"])
        for detail in result["limitations"]:
            diagnostic("unsupported", key[0], identity.get("test_file", key[1]), line, detail)
        function = _function(tree, line)
        call_nodes = {n.lineno: n for n in _calls(function)} if function else {}
        for check in result["checks"]:
            base = {k: methods[key][k] for k in ("repo", "file", "qname", "line", "line_end")}
            base.update({k: identity[k] for k in ("test_file", "test_method", "test_line", "test_line_end", "method_rule", "file_rule") if k in identity})
            base.update(check, side=side, callee_resolution="name_only" if side == "test" else "saved_framework_call",
                call_source=ast.get_source_segment(source, call_nodes[check["call_line"]]) or "",
                call_id="", joern_status="not_assessed", slice_ids=[], shared_slice_ids=[], slice_variables=[], slice_parse_flags=[],
                limitations=result["limitations"])
            if side == "production":
                for cid, saved_line in sorted(calls[key].items()):
                    if saved_line == check["call_line"]:
                        checks.append(dict(base, call_id=cid))
            else:
                checks.append(base)
        return result

    for key, method in sorted(methods.items()):
        status, source, tree = sources.get(key[0], key[1])
        method["source_status"] = status
        method["saved_calls"] = len(calls[key])
        method["saved_slices"] = len({sid for cid in calls[key] for sid in hits[cid] if sid in slices})
        if status == "ok":
            function = _function(tree, key[2])
            if function is None or function.end_lineno != method["line_end"] or function.name != method["qname"].split(".")[-1]:
                method["source_status"] = "definition_missing" if function is None else "source_disagreement"
                diagnostic(method["source_status"], key[0], key[1], key[2], "Original definition/range disagrees with baseline")
            else:
                body_calls = _calls(function)
                for cid, saved_line in list(calls[key].items()):
                    positions = [node for node in body_calls if node.lineno == saved_line]
                    # Canonical call IDs carry an original AST column. Use it
                    # when present to separate one-line bodies from defaults.
                    parts = cid.rsplit("::", 2)
                    if len(parts) == 3 and parts[-1].isdigit() and parts[-2].isdigit():
                        positions = [node for node in positions if node.col_offset == int(parts[-1])]
                    if not positions:
                        diagnostic("call_outside_body", key[0], key[1], saved_line,
                                   f"Saved call is not in exact original function body: {cid}")
                        del calls[key][cid]
                method["saved_calls"] = len(calls[key])
                method["saved_slices"] = len({sid for cid in calls[key] for sid in hits[cid] if sid in slices})
                method["production_source_ok"] = 1
                analyze(key, "production", {}, set(calls[key].values()), source, tree)
    for identity, link in sorted(links.items()):
        key = identity[:3]
        status, source, tree = sources.get(key[0], link["test_file"])
        link["source_status"] = status
        if status != "ok":
            continue
        function = _function(tree, link["test_line"])
        if function is None or function.name != link["test_method"].split(".")[-1] or function.end_lineno != link["test_line_end"]:
            link["source_status"] = "definition_missing" if function is None else "source_disagreement"
            diagnostic(link["source_status"], key[0], link["test_file"], link["test_line"], "Original test definition/range disagrees with mapping")
            continue
        matching = [n for n in _calls(function) if _final_name(n) == methods[key]["qname"].split(".")[-1]]
        link["matched_call_count"] = len(matching)
        methods[key]["test_assessed_links"] += 1
        if not matching:
            diagnostic("no_matching_call", key[0], link["test_file"], link["test_line"], "No name-only call to mapped method")
        result = analyze(key, "test", link, {n.lineno for n in matching}, source, tree)
        link["analysis_status"] = result["status"]
        link["limitations"] = result["limitations"]
        link["candidate_count"] = len(result["checks"])
    needed = defaultdict(set)
    for check in checks:
        if check["side"] == "production":
            for sid in hits[check["call_id"]]:
                if sid in slices:
                    needed[sid].update((check["call_line"], check["check_line"]))
    memberships = defaultdict(set)
    for row in _rows(inputs[names[7]], ["slice_id", "line"]):
        sid = _integer(row, "slice_id", names[7])
        if sid in needed:
            line = _integer(row, "line", names[7])
            if line in needed[sid]:
                memberships[sid].add(line)
    for check in checks:
        key = (check["repo"], check["file"], check["line"])
        method = methods[key]
        if check["side"] == "test":
            method["test_candidate_count"] += 1
        else:
            method["production_candidate_count"] += 1
            ids = sorted(sid for sid in hits[check["call_id"]] if sid in slices)
            shared = [sid for sid in ids if {check["call_line"], check["check_line"]} <= memberships[sid]]
            check["joern_status"] = "same_slice_candidate" if shared else "not_in_same_saved_slice" if ids else "no_saved_slice"
            check["slice_ids"] = ids
            check["shared_slice_ids"] = shared
            check["slice_variables"] = [slices[sid][2] for sid in ids]
            check["slice_parse_flags"] = [slices[sid][3] for sid in ids]
            method["shared_slice_candidate_count"] += bool(shared)
    checks.sort(key=lambda r: (r["repo"], r["file"], r["line"], r["side"], r.get("test_file", ""), r.get("test_line", 0), r["call_line"], r["check_line"], r["call_id"]))
    diagnostic_keys = {json.dumps(r, sort_keys=True, ensure_ascii=False): r for r in diagnostics}
    diagnostics = [diagnostic_keys[k] for k in sorted(diagnostic_keys)]
    n = len(methods)

    def metric(count):
        return dict(numerator=count, denominator=n, rate=count / n if n else None,
                    denominator_name="selected_llm_baseline_methods")
    values = list(methods.values())
    method_totals = dict(denominator=n,
        with_mapped_test=sum(m["mapped_test"] for m in values),
        with_exact_mapped_test=sum(m["exact_mapped_tests"] > 0 for m in values),
        with_file_association=sum(m["file_associated"] for m in values),
        with_import_file_association=sum(m["import_file_associated"] for m in values),
        with_test_candidate=sum(m["test_candidate_count"] > 0 for m in values),
        with_production_candidate=sum(m["production_candidate_count"] > 0 for m in values),
        with_shared_slice_candidate=sum(m["shared_slice_candidate_count"] > 0 for m in values))
    exact_candidates = {(c["repo"], c["file"], c["line"]) for c in checks if c["side"] == "test" and c["method_rule"] == "exact"}
    check_totals = dict(test=sum(c["side"] == "test" for c in checks),
        production=sum(c["side"] == "production" for c in checks),
        production_same_slice=sum(c["joern_status"] == "same_slice_candidate" for c in checks))
    summary = dict(population="unique group=llm saved baseline methods", subset=bool(filters), subset_repos=sorted(filters),
        label="subset" if filters else "full saved baseline", full_baseline_methods=measured[0],
        methods=method_totals, checks=check_totals,
        rates={k: metric(v) for k, v in method_totals.items() if k != "denominator"},
        assessment=dict(production_source_ok=sum(m["production_source_ok"] for m in values),
                        test_links=len(links), test_source_ok_links=sum(l["source_status"] == "ok" for l in links.values())),
        exact_only_sensitivity=dict(mapped_methods=metric(method_totals["with_exact_mapped_test"]),
                                    candidate_methods=metric(len(exact_candidates))),
        candidate_rows=len(checks), missingness=dict(Counter(d["kind"] for d in diagnostics)),
        source_statuses=dict(Counter(m["source_status"] for m in values)), limitations=LIMITS)
    _write_csv(output / "methods.csv", [methods[k] for k in sorted(methods)], METHOD_FIELDS)
    _write_csv(output / "test_links.csv", [links[k] for k in sorted(links)], LINK_FIELDS)
    _write_csv(output / "checks.csv", checks, CHECK_FIELDS)
    _write_csv(output / "diagnostics.csv", diagnostics, ["repo", "file", "line", "kind", "detail"])
    _write_json(output / "summary.json", summary)
    report = ["# Static result-to-assertion candidates", "", f"Population: {summary['label']}; {n} unique saved LLM baseline methods.", ""]
    for label, count in method_totals.items():
        if label != "denominator":
            report.append(f"- {label}: {count}/{n} methods" + (f" ({count / n:.2%})." if n else " (rate unavailable)."))
    report.extend(["", f"Candidate rows: {len(checks)}; distinct test links: {len(links)}.", "",
        "## Assessment limits", "", *[f"- {text}" for text in LIMITS], "",
        "## Deterministic positive sample", ""])
    for row in checks[:5]:
        report.append(f"- {row['side']} candidate: {row['repo']} {row.get('test_file') or row['file']} call {row['call_line']} -> check {row['check_line']}; {row['joern_status']}; `{row['check_source'].replace(chr(10), ' ')}`")
    if not checks:
        report.append("No positive candidates detected in assessed inputs.")
    report.extend(["", "## Missingness and disagreement examples", ""])
    for row in diagnostics[:10]:
        report.append(f"- {row['kind']}: {row['repo']} {row['file']}:{row['line']}: {row['detail']}")
    if not diagnostics:
        report.append("No diagnostics recorded.")
    report.extend(["", "## Provenance", "",
        "manifest.json records Python/revision/options, input hashes before/after, inspected original source hashes, and all output hashes except its own. Sources use their declared Python encoding. Source caching is bounded to 16 entries and 8 MiB of text; saved tables are streamed. Lists in CSV are JSON arrays. slice_ids lists every relevant saved call-hit slice; shared_slice_ids lists only those containing both original call and check lines. slice_variables and slice_parse_flags correspond positionally to slice_ids."])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    after = {name: _sha(path) for name, path in inputs.items()}
    changed = [name for name in names if before[name] != after[name]]
    changed.extend(path for path, digest in sources.hashes.items() if not Path(path).is_file() or _sha(Path(path)) != digest)
    if changed:
        diagnostics.append(dict(kind="changed_input", detail=", ".join(changed)))
        _write_csv(output / "diagnostics.csv", diagnostics, ["repo", "file", "line", "kind", "detail"])
        raise ValueError("changed-input: " + ", ".join(changed))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    _write_json(output / "manifest.json", dict(python=sys.version, revision=revision,
        options=dict(artifacts_dir=str(args.artifacts_dir.resolve()), repos_dir=str(args.repos_dir.resolve()),
                     output_dir=str(output.resolve()), repo=sorted(filters)),
        inputs={name: dict(sha256_before=before[name], sha256_after=after[name]) for name in names},
        inspected_sources=sources.hashes,
        output_hashes={p.name: _sha(p) for p in sorted(output.iterdir()) if p.is_file()}))


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=root / "pipeline" / "artifacts")
    parser.add_argument("--repos-dir", type=Path, default=root / "pipeline" / "repos")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", action="append", help="Repeatable OWNER/NAME subset filter")
    args = parser.parse_args(argv)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Output exists: {args.output_dir}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Output error: {exc}", file=sys.stderr)
        return 1
    try:
        _run(args, args.output_dir)
    except (ValueError, OSError, csv.Error, KeyError, TypeError) as exc:
        print(f"Input/output error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
