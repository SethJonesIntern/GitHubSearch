"""Compare production methods with/without direct observed LLM calls.

Run on the machine holding Stage-5 clones and the full pyMethods2Test output:
    py -3.14 -m pipeline.method_test_baseline

Reads existing snapshots only; never clones, runs Joern, or reruns focal mapping.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import statistics
import tokenize
from collections import defaultdict
from pathlib import Path

from pipeline import paths
from pipeline.slice_index import code_kind

GROUPS = ("llm", "other")
GOOD_RUNS = {"ok", "no_tests", "no_focal_methods"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".tox",
             "build", "dist", "site-packages", ".mypy_cache", ".pytest_cache"}


def normalized(value):
    return value.replace("\\", "/")


def relative(value, slug):
    value = normalized(value)
    prefix = slug + "/"
    if value.startswith(prefix):
        return value[len(prefix):]
    marker = "/" + prefix
    if marker in value:
        return value.rsplit(marker, 1)[1]
    return value


def definitions(tree):
    """All real sync/async defs, including nested defs; no synthetic constructors."""
    found = []

    def visit(node, parents):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = ".".join((*parents, child.name))
                if not isinstance(child, ast.ClassDef):
                    found.append({"name": child.name, "qname": name,
                                  "line": child.lineno, "end": child.end_lineno,
                                  "body_start": child.body[0].lineno})
                visit(child, (*parents, child.name))
            else:
                visit(child, parents)

    visit(tree, ())
    return found


def containing(methods, line):
    # Decorators/defaults are not part of the function body. Nested function calls
    # belong only to the innermost body, never automatically to its outer def.
    hits = [m for m in methods if m["body_start"] <= line <= m["end"]]
    return min(hits, key=lambda m: m["end"] - m["line"]) if hits else None


def inventory(root):
    methods, errors = {}, []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                         and not (Path(directory) / d).is_symlink())
        for filename in sorted(files):
            if not filename.endswith(".py"):
                continue
            source = Path(directory) / filename
            if source.is_symlink():
                continue
            rel = source.relative_to(root).as_posix()
            # Production only, using the same classification as the slice analysis.
            if code_kind(rel, "") != "production":
                continue
            try:
                with tokenize.open(source) as handle:
                    tree = ast.parse(handle.read(), filename=rel)
                methods[rel] = definitions(tree)
            except (SyntaxError, UnicodeError, OSError, LookupError) as exc:
                errors.append(f"{rel}: {exc}")
    return methods, errors


def compare_repo(repo, slug, root, calls, mappings):
    files, errors = inventory(root)
    by_key = {(file, m["line"]): m for file, defs in files.items() for m in defs}
    llm, tests, exact = set(), defaultdict(set), defaultdict(set)
    unmatched_calls = unmatched_focals = 0
    for call in calls:
        file = relative(call["file"], slug)
        if code_kind(file, "") != "production":
            continue
        method = containing(files.get(file, []), int(call["call_line"]))
        if method is None:
            unmatched_calls += 1
        else:
            llm.add((file, method["line"]))
    for row in mappings:
        if not row["focal_method"]:
            continue
        file = relative(row["focal_file"], slug)
        if code_kind(file, "") != "production":
            continue
        key = (file, int(row["focal_line"]))
        method = by_key.get(key)
        if (method is None or method["name"] != row["focal_method"]
                or method["end"] != int(row["focal_line_end"])):
            unmatched_focals += 1
            continue
        test = (normalized(row["test_file"]), row["test_method"], row["test_line"])
        tests[key].add(test)
        if row["method_rule"] == "exact":
            exact[key].add(test)
    rows = []
    for (file, line), method in sorted(by_key.items()):
        key = file, line
        rows.append({"repo": repo, "file": file, "qname": method["qname"],
                     "line": line, "line_end": method["end"],
                     "group": "llm" if key in llm else "other",
                     "mapped_tests": len(tests[key]), "exact_mapped_tests": len(exact[key])})
    diagnostics = {"parse_errors": len(errors), "error_examples": errors[:5],
                   "unmatched_production_calls": unmatched_calls,
                   "unmatched_production_focals": unmatched_focals}
    return rows, diagnostics


def rate(numerator, denominator):
    return round(100 * numerator / denominator, 6) if denominator else None


def counts(rows):
    out = {}
    for group in GROUPS:
        selected = [r for r in rows if r["group"] == group]
        n = len(selected)
        mapped = sum(r["mapped_tests"] > 0 for r in selected)
        exact = sum(r["exact_mapped_tests"] > 0 for r in selected)
        out.update({f"{group}_methods": n, f"{group}_mapped": mapped,
                    f"{group}_pct": rate(mapped, n), f"{group}_exact_mapped": exact,
                    f"{group}_exact_pct": rate(exact, n)})
    out["difference_pp"] = (round(out["llm_pct"] - out["other_pct"], 6)
                            if out["llm_methods"] and out["other_methods"]
                            else None)
    return out


def summarize(repos):
    out = {}
    for group in GROUPS:
        n = sum(r[f"{group}_methods"] for r in repos)
        mapped = sum(r[f"{group}_mapped"] for r in repos)
        exact = sum(r[f"{group}_exact_mapped"] for r in repos)
        out[group] = {"methods": n, "with_mapped_test": mapped,
                      "pct": rate(mapped, n), "with_exact_test": exact,
                      "exact_pct": rate(exact, n)}
    return out


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=paths.ARTIFACTS_DIR)
    parser.add_argument("--repos-dir", type=Path, default=paths.REPOS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Smoke run; not a population estimate")
    args = parser.parse_args(argv)
    artifacts = args.artifacts_dir
    required = [artifacts / name for name in
                ("pym2t_focal_tests.csv", "pym2t_runs.csv")]
    missing = [str(p) for p in required if not p.is_file()]
    if not args.repos_dir.is_dir():
        missing.append(str(args.repos_dir))
    if missing:
        parser.error("Missing inputs; run on the PC holding the full artifacts: "
                     + ", ".join(missing))
    # Same authoritative observed scope as the focal-mapping driver. Import lazily
    # so --help and missing-input diagnostics need no analysis dependencies.
    from Applications.make_figures import POP, CALLS
    calls = defaultdict(list)
    for row in CALLS.to_dict("records"):
        calls[row["repo"]].append(row)
    runs = {}
    with required[1].open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["repo"] in runs:
                parser.error(f"Duplicate run record: {row['repo']}")
            runs[row["repo"]] = row
    selected = sorted(POP)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        selected = selected[:args.limit]
    selected_set = set(selected)
    mappings = defaultdict(list)
    with required[0].open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["repo"] in selected_set and row["focal_method"]:
                mappings[row["repo"]].append(row)
    output = args.output_dir or artifacts / "method_test_baseline"
    output.mkdir(parents=True, exist_ok=True)
    per_repo, diagnostics = [], []
    fields = ["repo", "file", "qname", "line", "line_end", "group",
              "mapped_tests", "exact_mapped_tests"]
    with (output / "methods.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for repo in selected:
            run = runs.get(repo, {})
            slug = run.get("slug") or repo.replace("/", "_")
            root = args.repos_dir / slug
            detail = {"repo": repo, "status": "included"}
            if not root.is_dir():
                detail.update(status="missing_clone")
            elif run.get("status") not in GOOD_RUNS:
                detail.update(status="incomplete_mapping_run", run_status=run.get("status"))
            elif int(run.get("cli_mismatch") or 0):
                detail.update(status="cli_mismatch")
            elif len(mappings[repo]) != int(run.get("focal_methods") or 0):
                detail.update(status="inconsistent_mapping_rows")
            else:
                rows, audit = compare_repo(repo, slug, root, calls[repo], mappings[repo])
                detail.update(audit)
                if any(audit[k] for k in ("parse_errors", "unmatched_production_calls",
                                          "unmatched_production_focals")):
                    detail["status"] = "inventory_or_snapshot_mismatch"
                elif not rows:
                    detail["status"] = "no_production_methods"
                else:
                    writer.writerows(rows)
                    per_repo.append({"repo": repo, **counts(rows)})
            diagnostics.append(detail)
            print(f"{len(diagnostics)}/{len(selected)} {repo}: {detail['status']}", flush=True)
    paired = [r for r in per_repo if r["llm_methods"] and r["other_methods"]]
    summary = {
        "scope": "Observed POP/CALLS; production functions and methods; direct calls only",
        "selected_repos": len(selected), "included_repos": len(per_repo),
        "excluded_repos": len(selected) - len(per_repo), "smoke_run": args.limit is not None,
        "pooled_all_complete_repos": summarize(per_repo),
        "paired_repos": len(paired), "pooled_paired_repos": summarize(paired),
        "macro_paired_pct": {g: statistics.mean(r[f"{g}_pct"] for r in paired)
                              if paired else None for g in GROUPS},
        "mean_paired_difference_pp": statistics.mean(r["difference_pp"] for r in paired)
                                     if paired else None,
        "median_paired_difference_pp": statistics.median(r["difference_pp"] for r in paired)
                                       if paired else None,
        "limitations": ["Test correspondence is heuristic, not execution coverage.",
                        "Other means no detected direct in-scope framework call; may include transitive invokers or missed calls.",
                        "Whole repositories with source parse failures or unjoined production calls/focals are excluded.",
                        "Snapshot validation checks focal name/start/end, not source hashes.",
                        "Production classification follows slice_index.code_kind; unconventional test locations may remain."],
        "diagnostics": diagnostics,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "by_repo.csv", per_repo,
              ["repo", *counts([])])
    print(json.dumps({k: v for k, v in summary.items() if k != "diagnostics"}, indent=2))
    print(f"Outputs: {output}")
    return 0 if per_repo else 1


if __name__ == "__main__":
    raise SystemExit(main())
