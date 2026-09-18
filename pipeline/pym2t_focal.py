"""Stage 8 — focal mapping with pyMethods2Test, over the 644 observed applications.

Focal mapping comes from the published tool (Abdelmadjid & Dyer, MSR 2025; vendored,
unmodified, in `pipeline/third_party/pymethods2test/`), not from `focal_map.py`.
Decision and rationale: HANDOFF_2026-09-15.md.

Per repo, their two scripts run as CLIs, exactly as their README shows:

    test_identification.py --outfile <slug>.json --outtests <slug>.tests.json <slug>
    find_focals.py <slug>.json <slug>.tests.json -outfile <slug>.focal.json

Their CLI output keeps only tests that got a focal METHOD, keyed by method name
(duplicates across classes in one file overwrite). So the per-test table is built by
replaying their loop with their own imported functions, which additionally records
  * tests that got a focal FILE but no focal method, and
  * which rule produced the method: `exact` (test name ends with a called name) or
    `fuzzy` (`fuzz.ratio > 50` fallback).
Every repo's replay is compared against its CLI focal.json; any disagreement is counted
and printed, so the replay can never silently drift from the tool.

Three joins, all written to ARTIFACTS_DIR:
  pym2t_focal_tests.csv  every test method the tool identified, with its focal file/method
  pym2t_nd_tests.csv     our ND tests (observed scope, direct + transitive) -> focal status
  pym2t_focal_calls.csv  the 23,990 framework call sites -> does a test's focal method
                         (by line range in the focal file) contain it?
  pym2t_runs.csv         one row per repo: tool exit status, counts, parse errors

Scope is imported, never restated: `make_figures.POP` / `.CALLS` (644 apps, 23,990
calls) and `focal_map.observed_tests` (the 9,700 direct ND tests).

    py -3.14 -m pipeline.pym2t_focal              # all 644, reusing any finished repo
    py -3.14 -m pipeline.pym2t_focal --rerun      # re-run the tool on every repo
    py -3.14 -m pipeline.pym2t_focal --limit 10   # smoke run

READ-ONLY over existing artifacts; the raw per-repo JSON lives in ARTIFACTS_DIR/pym2t/
(gitignored — regenerable, and several hundred MB).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pipeline.paths import ARTIFACTS_DIR, REPOS_DIR

_ROOT = Path(__file__).resolve().parent.parent
TOOL_DIR = Path(__file__).resolve().parent / "third_party" / "pymethods2test"
RAW_DIR = ARTIFACTS_DIR / "pym2t"

TESTS_CSV = ARTIFACTS_DIR / "pym2t_focal_tests.csv"
ND_CSV = ARTIFACTS_DIR / "pym2t_nd_tests.csv"
CALLS_CSV = ARTIFACTS_DIR / "pym2t_focal_calls.csv"
RUNS_CSV = ARTIFACTS_DIR / "pym2t_runs.csv"

TIMEOUT_S = 1800

TEST_FIELDS = ["repo", "test_file", "test_method", "test_line", "test_line_end",
               "test_framework", "n_called", "focal_file", "file_rule", "focal_class",
               "focal_method", "focal_owner", "focal_line", "focal_line_end",
               "method_rule"]
RUN_FIELDS = ["repo", "slug", "status", "seconds", "parse_errors", "test_files",
              "test_methods", "focal_files", "focal_methods", "cli_mismatch", "detail"]


def _find_focals():
    """Their find_focals.py as a module — imported, not copied."""
    spec = importlib.util.spec_from_file_location("pym2t_find_focals",
                                                  TOOL_DIR / "find_focals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_backend() -> None:
    from fuzzywuzzy import fuzz
    backend = fuzz.SequenceMatcher.__module__
    if backend != "fuzzywuzzy.StringMatcher":
        sys.exit(f"fuzzywuzzy is using {backend!r}, not the Levenshtein C backend — "
                 "fuzz.ratio values would differ from the tool's. "
                 "py -3.14 -m pip install Levenshtein")


# ── per repo ──────────────────────────────────────────────────────────────────

def _norm(path: str) -> str:
    return path.replace("\\", "/")


def _run_tool(slug: str) -> dict:
    """Both CLIs on one clone. cwd=REPOS_DIR and a bare slug make every path the tool
    emits `<slug>/<rel>` — the same shape as the `file` column of our artifacts."""
    impl, tests, focal = (RAW_DIR / f"{slug}{ext}"
                          for ext in (".json", ".tests.json", ".focal.json"))
    for p in (impl, tests, focal):
        p.unlink(missing_ok=True)
    t0 = time.time()
    rec = {"status": "", "parse_errors": 0, "detail": ""}
    steps = (
        [sys.executable, str(TOOL_DIR / "test_identification.py"),
         "--outfile", str(impl), "--outtests", str(tests), slug],
        [sys.executable, str(TOOL_DIR / "find_focals.py"),
         str(impl), str(tests), "-outfile", str(focal)],
    )
    # UTF-8 mode: on Windows a piped stdout is cp1252, and find_focals.py's progress
    # print() of a non-ASCII test name raises UnicodeEncodeError and kills the repo.
    # That is our platform, not the tool — it runs as it would on Linux. (Source files
    # are unaffected: test_identification opens them with an explicit encoding='utf-8',
    # so a non-UTF-8 .py file still crashes its repo, as it would anywhere.)
    env = {**os.environ, "PYTHONUTF8": "1"}
    for i, cmd in enumerate(steps):
        try:
            proc = subprocess.run(cmd, cwd=REPOS_DIR, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=TIMEOUT_S,
                                  env=env)
        except subprocess.TimeoutExpired:
            rec.update(status="timeout" if i == 0 else "focal_timeout")
            break
        err = proc.stderr
        if i == 0:
            rec["parse_errors"] = err.count("Error parsing file")
        if proc.returncode == 0:
            continue
        if "No test methods found" in err:
            rec["status"] = "no_tests"
        elif "No focal methods found" in err:
            rec["status"] = "no_focal_methods"
        else:
            tail = [l for l in err.strip().splitlines() if l.strip()][-3:]
            rec.update(status="crash" if i == 0 else "focal_crash",
                       detail=" | ".join(tail)[:500])
        break
    rec["status"] = rec["status"] or "ok"
    rec["seconds"] = round(time.time() - t0, 1)
    (RAW_DIR / f"{slug}.run.json").write_text(json.dumps(rec), encoding="utf-8")
    return rec


def _owner(file_data: dict, name: str) -> str:
    """Which dict find_focal_method resolved `name` from — same lookup order as theirs."""
    if "__global__" in file_data and name in file_data["__global__"]:
        return "__global__"
    for k, v in file_data.items():
        if k != "__modulename__" and name in v:
            return k
    return ""


def _file_rule(all_data: dict, test_file: dict, focal_file: str) -> str:
    """How find_focal_file got its answer. Their implementation index holds EVERY .py
    file, test files included, so the basename fallback can always match a test file to
    itself — the tool never returns "no focal file". Only `import` is real evidence."""
    imports = [v for v in test_file["test_imports"].values()]
    imports += [".".join(v.split(".")[:-1]) for v in imports if "." in v]
    if any(v["__modulename__"] in imports for v in all_data.values()):
        return "import"
    if _norm(focal_file).endswith("/" + _norm(test_file["file_path"])):
        return "self"
    return "basename"


def process_repo(job):
    repo, slug, rerun = job
    run_file = RAW_DIR / f"{slug}.run.json"
    if rerun or not run_file.exists():
        rec = _run_tool(slug)
    else:
        rec = json.loads(run_file.read_text(encoding="utf-8"))

    run = {"repo": repo, "slug": slug, "test_files": 0, "test_methods": 0,
           "focal_files": 0, "focal_methods": 0, "cli_mismatch": 0, **rec}
    impl, tests = RAW_DIR / f"{slug}.json", RAW_DIR / f"{slug}.tests.json"
    if not (impl.exists() and tests.exists()):
        return run, []

    ff = _find_focals()
    all_data = json.loads(impl.read_text(encoding="utf-8"))
    all_tests = json.loads(tests.read_text(encoding="utf-8"))

    rows, replay = [], {}
    for test_file in all_tests:
        focal_file = ff.find_focal_file(all_data, test_file["file_path"],
                                        test_file["test_imports"])
        run["test_files"] += 1
        run["focal_files"] += bool(focal_file)
        tf = f"{slug}/{_norm(test_file['file_path'])}"
        file_rule = _file_rule(all_data, test_file, focal_file) if focal_file else ""
        for m in test_file["test_methods"]:
            run["test_methods"] += 1
            row = {"repo": repo, "test_file": tf, "test_method": m["method_name"],
                   "test_line": m["line"], "test_line_end": m["line_end"],
                   "test_framework": test_file["test_framework"],
                   "n_called": len(m["called_methods"]),
                   "focal_file": _norm(focal_file) if focal_file else "", "file_rule": file_rule,
                   "focal_class": "", "focal_method": "", "focal_owner": "",
                   "focal_line": "", "focal_line_end": "", "method_rule": ""}
            if focal_file and m["called_methods"]:
                fm = ff.find_focal_method(all_data[focal_file], m["method_name"],
                                          m["called_methods"])
                if fm:
                    run["focal_methods"] += 1
                    exact = any(m["method_name"].endswith(c.split(".")[-1])
                                for c in m["called_methods"])
                    row.update(
                        focal_class=ff.find_focal_class(all_data[focal_file],
                                                        m["called_methods"]) or "",
                        focal_method=fm["name"],
                        focal_owner=_owner(all_data[focal_file], fm["name"]),
                        focal_line=fm["line"], focal_line_end=fm["line_end"],
                        method_rule="exact" if exact else "fuzzy")
                    # their CLI: keyed by method name, later matches overwrite
                    replay[(test_file["file_path"], m["method_name"])] = (
                        focal_file, fm["name"], fm["line"])
            rows.append(row)

    # agreement with the tool's own output
    focal = RAW_DIR / f"{slug}.focal.json"
    cli = {}
    if focal.exists():
        for tfile, d in json.loads(focal.read_text(encoding="utf-8")).items():
            for name, v in d["methods"].items():
                cli[(tfile, name)] = (d["focal_file"], v["focal_method"]["name"],
                                      v["focal_method"]["line"])
    run["cli_mismatch"] = len(set(cli.items()) ^ set(replay.items()))
    return run, rows


# ── joins ─────────────────────────────────────────────────────────────────────

def _scope():
    for extra in (_ROOT / "Applications", _ROOT / "Wrapper"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    import make_figures as MF                        # noqa: PLC0415
    from pipeline.focal_map import observed_tests    # noqa: PLC0415
    nd, steps = observed_tests(kind="all", scoped=True)
    return MF.POP, MF.CALLS, nd, steps


def join_nd(nd, tests):
    by_exact = {(r["test_file"], r["test_method"], int(r["test_line"])): r for r in tests}
    by_name = defaultdict(list)
    for r in tests:
        by_name[(r["test_file"], r["test_method"])].append(r)

    out = []
    for repo, qname, file, line, kind in zip(nd["repo"], nd["qname"], nd["file"],
                                             nd["line"], nd["kind"]):
        name = qname.rsplit(".", 1)[-1]
        hit, how = by_exact.get((file, name, int(line))), "file+name+line"
        if hit is None:
            cands = by_name.get((file, name), [])
            hit, how = (cands[0], "file+name") if len(cands) == 1 else (None, "")
        if hit is None:
            status = "not_identified"
        elif not hit["focal_file"]:
            status = "no_focal_file"
        elif not hit["focal_method"]:
            status = "focal_file_only"
        else:
            status = f"focal_method_{hit['method_rule']}"
        out.append({"repo": repo, "qname": qname, "file": file, "line": line,
                    "kind": kind, "status": status, "join": how,
                    **{k: (hit or {}).get(k, "") for k in
                       ("focal_file", "file_rule", "focal_class", "focal_method", "focal_owner",
                        "focal_line", "focal_line_end", "method_rule")}})
    return out


def join_calls(calls, tests, nd_rows):
    """A call site is 'covered' by a test when it lies inside that test's focal method
    (focal file equal, call line within [focal_line, focal_line_end])."""
    nd_kinds = defaultdict(set)
    for r in nd_rows:
        if r["join"]:
            nd_kinds[(r["file"], r["qname"].rsplit(".", 1)[-1], int(r["line"]))].add(r["kind"])

    ranges = defaultdict(list)       # focal_file -> [(start, end, test_key)]
    per_file = Counter()
    for r in tests:
        if r["focal_file"]:
            per_file[r["focal_file"]] += 1
        if r["focal_method"]:
            key = (r["test_file"], r["test_method"], int(r["test_line"]))
            ranges[r["focal_file"]].append((int(r["focal_line"]), int(r["focal_line_end"]), key))

    out = []
    for c in calls.itertuples(index=False):
        n_any = n_direct = n_nd = 0
        methods = set()
        for start, end, key in ranges.get(c.file, ()):
            if start <= int(c.call_line) <= end:
                n_any += 1
                methods.add(start)
                kinds = nd_kinds.get(key, set())
                n_nd += bool(kinds)
                n_direct += "direct" in kinds
        out.append({"repo": c.repo, "call_id": c.call_id, "file": c.file,
                    "enclosing_qname": c.enclosing_qname, "framework": c.framework,
                    "call_line": c.call_line,
                    "tests_focal_file": per_file.get(c.file, 0),
                    "tests_focal_method": n_any, "nd_tests_focal_method": n_nd,
                    "nd_direct_tests_focal_method": n_direct,
                    "focal_methods_containing": len(methods)})
    return out


def _write(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _pct(n, d):
    return f"{n:>7,} / {d:<7,} {100 * n / d:5.1f}%" if d else f"{n:>7,} / 0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", action="store_true", help="re-run the tool on every repo")
    ap.add_argument("--limit", type=int, help="only the first N repos (smoke run)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = ap.parse_args()

    check_backend()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pop, calls, nd, steps = _scope()
    repos = sorted(pop)[: args.limit] if args.limit else sorted(pop)
    if args.limit:
        calls = calls[calls["repo"].isin(repos)]
        nd = nd[nd["repo"].isin(repos)]
    print(f"# scope: {len(repos)} apps, {len(calls):,} call sites, {len(nd):,} ND test rows")
    for label, n in steps:
        print(f"#   {label:35s} {n:9,}")

    runs, tests = [], []
    jobs = [(r, r.replace("/", "_"), args.rerun) for r in repos]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_repo, j) for j in jobs]
        for done, fut in enumerate(as_completed(futures), 1):
            run, rows = fut.result()
            runs.append(run)
            tests.extend(rows)
            if done % 25 == 0 or done == len(jobs):
                print(f"#   {done}/{len(jobs)} repos", file=sys.stderr, flush=True)

    runs.sort(key=lambda r: r["repo"])
    tests.sort(key=lambda r: (r["test_file"], int(r["test_line"])))
    nd_rows = join_nd(nd, tests)
    call_rows = join_calls(calls, tests, nd_rows)

    _write(RUNS_CSV, runs, RUN_FIELDS)
    _write(TESTS_CSV, tests, TEST_FIELDS)
    _write(ND_CSV, nd_rows, list(nd_rows[0]) if nd_rows else ["repo"])
    _write(CALLS_CSV, call_rows, list(call_rows[0]) if call_rows else ["repo"])

    # ── report ──
    st = Counter(r["status"] for r in runs)
    print(f"\n# REPOS ({len(runs)})")
    for k, n in st.most_common():
        print(f"#   {k:20s} {n:5d}")
    print(f"#   files the tool failed to parse: {sum(r['parse_errors'] for r in runs):,}")
    bad = [r for r in runs if r["cli_mismatch"]]
    print(f"#   replay vs CLI focal.json mismatches: {len(bad)} repos")
    for r in bad[:5]:
        print(f"#     {r['repo']}: {r['cli_mismatch']}")
    for r in [r for r in runs if "crash" in r["status"] or "timeout" in r["status"]][:8]:
        print(f"#     {r['status']:12s} {r['repo']}: {r['detail'][:160]}")

    n = len(tests)
    print(f"\n# ALL TESTS THE TOOL IDENTIFIED ({n:,}, in {sum(r['test_methods'] > 0 for r in runs)} repos)")
    print(f"#   focal file     {_pct(sum(bool(r['focal_file']) for r in tests), n)}")
    fr = Counter(r["file_rule"] for r in tests if r["file_rule"])
    for k in ("import", "basename", "self"):
        print(f"#     {k:12s} {_pct(fr[k], n)}")
    fm = Counter(r["method_rule"] for r in tests if r["method_rule"])
    print(f"#   focal method   {_pct(sum(fm.values()), n)}")
    for k in ("exact", "fuzzy"):
        print(f"#     {k:12s} {_pct(fm[k], n)}")

    for kind in ("direct", "transitive"):
        rows = [r for r in nd_rows if r["kind"] == kind]
        if not rows:
            continue
        c = Counter(r["status"] for r in rows)
        print(f"\n# ND TESTS, {kind} ({len(rows):,} in {len({r['repo'] for r in rows})} apps)")
        for k in ("not_identified", "no_focal_file", "focal_file_only",
                  "focal_method_exact", "focal_method_fuzzy"):
            print(f"#   {k:20s} {_pct(c[k], len(rows))}")
        hit = [r for r in rows if r["focal_method"]]
        print(f"#   => focal file      {_pct(sum(bool(r['focal_file']) for r in rows), len(rows))}")
        print(f"#      ... via import  {_pct(sum(r['file_rule'] == 'import' for r in rows), len(rows))}")
        print(f"#   => focal method    {_pct(len(hit), len(rows))}"
              f"   ({len({r['repo'] for r in hit})} apps)")
        print(f"#   joined by name only (line differed): {sum(r['join'] == 'file+name' for r in rows):,}")

    nc = len(call_rows)
    print(f"\n# FRAMEWORK CALL SITES ({nc:,} in {len({r['repo'] for r in call_rows})} apps)")
    for label, key in (("in some test's focal file", "tests_focal_file"),
                       ("in some test's focal method", "tests_focal_method"),
                       ("  ... of an ND test (any)", "nd_tests_focal_method"),
                       ("  ... of a direct ND test", "nd_direct_tests_focal_method")):
        hit = [r for r in call_rows if r[key]]
        print(f"#   {label:30s} {_pct(len(hit), nc)}   ({len({r['repo'] for r in hit})} apps)")

    print(f"\n# wrote {TESTS_CSV.name}, {ND_CSV.name}, {CALLS_CSV.name}, {RUNS_CSV.name}")


if __name__ == "__main__":
    main()
