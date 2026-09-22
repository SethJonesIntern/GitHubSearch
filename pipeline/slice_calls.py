#!/usr/bin/env python
"""Stage 6b: join the slice index to the framework call inventory.

Step 0 (``pipeline.slice_index``) made the slices queryable. This asks the
question the slices exist to answer: **which per-variable PDG slices contain a
framework LLM call site, and on which side of it does the variable sit?**

A slice is a variable's dependence closure, so a call site appearing in a
slice's ``pdg_lines`` means that variable is genuinely data- or
control-dependent on (or depended upon by) the LLM call.

Two columns then say what that variable IS at the call. ``relation`` is
positional, against the slicing criterion -- which is the variable's LAST USE,
because slice_repo runs the slicer in ``bidirectional`` mode:

  call_before_last_use   the variable is still live after the call
  call_at_last_use       the call is the variable's final appearance
  call_after_last_use    the value reaches the call past its own last direct use

``relation`` alone cannot separate prompt from response -- a prompt argument's
last use IS the call line -- so ``role`` reads the call text instead:

  receiver   the object called on: the agent/LLM/client that carries model and
             sampling configuration
  argument   named in the call: prompt, messages, params
  binding    absent from the call text: the assignment target or a value
             derived downstream of it

Scope is the paper's observed scope, imported from ``make_figures`` rather than
restated: framework call sites only, raw provider SDKs excluded, false-positive
tiers and removed patterns dropped.

    py -3.14 -m pipeline.slice_calls
    py -3.14 -m pipeline.slice_calls --all-repos   # do not restrict to POP
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "Applications", _ROOT / "Wrapper"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import paths  # noqa: E402
from pipeline.slice_index import (  # noqa: E402
    SLICE_INDEX_CSV, SLICE_PDG_LINES_CSV_GZ,
)

SLICE_CALL_HITS_CSV = paths.ARTIFACTS_DIR / "slice_call_hits.csv"      # 1 row / (slice, call site)
SLICE_CALLS_BY_CALL_CSV = paths.ARTIFACTS_DIR / "slice_calls_by_call.csv"  # 1 row / observed call site
SLICE_CALLS_SUMMARY_JSON = paths.ARTIFACTS_DIR / "slice_calls_summary.json"

HIT_FIELDS = [
    "slice_id", "repo", "rel_path", "is_test_path", "method", "variable",
    "is_duplicate", "parses", "n_pdg", "n_criterion", "criterion_first", "criterion_last",
    "criterion_exact",   # n_criterion == 1, so the direction call is exact rather than a span test
    "call_id", "call_repo", "framework", "pattern", "call_line", "enclosing_qname",
    "relation",          # call_before_last_use | call_at_last_use | call_after_last_use
    "role",              # receiver | argument | binding  (what the variable is at the call)
]

CALL_FIELDS = [
    "call_id", "repo", "repo_slug", "rel_path", "framework", "pattern", "call_line",
    "enclosing_qname", "repo_sliced", "in_slice",
    "n_slices", "n_slices_nondup", "n_variables",
    "n_before", "n_at", "n_after",
    "n_receiver", "n_argument", "n_binding", "variables_sample",
]


def load_calls(all_repos: bool) -> Tuple[List[Dict[str, Any]], int]:
    """Observed-scope framework call sites, via make_figures so the scope
    definition lives in exactly one place (CLAUDE.md: import, do not restate)."""
    import make_figures as MF  # noqa: WPS433 - heavy import, deliberately lazy

    calls = MF.CALLS
    if not all_repos:
        calls = calls[calls["repo"].isin(MF.POP)]
    rows = calls.to_dict("records")
    return rows, len(MF.POP)


def _split_file(f: str) -> Tuple[str, str]:
    """'<repo_slug>/<rel/path.py>' -> (slug, rel_path)."""
    norm = str(f).replace("\\", "/")
    slug, _, rel = norm.partition("/")
    return slug, rel


def _relation(call_line: int, c_first: str, c_last: str) -> str:
    """Where the call sits relative to the slicing criterion.

    The criterion is the variable's LAST USE: slice_repo runs the slicer in
    ``bidirectional`` mode, whose backward half seeds at last-use
    (per_variable_pdg_slicer._criterion_lines_for_variable, called with
    "last-use" at line 948). That is why every hit has n_criterion == 1, and
    why these buckets are named after last-use rather than after a
    prompt/response reading -- a prompt argument's last use IS the call line,
    so it lands in call_at_last_use, not in a "feeds" bucket. Use the
    companion role column to separate those.
    """
    if not c_first:
        return "no_criterion"
    lo, hi = int(c_first), int(c_last)
    if call_line < lo:
        return "call_before_last_use"    # variable still live after the call: it carries what came back
    if call_line > hi:
        return "call_after_last_use"     # value reaches the call past its own last direct use
    return "call_at_last_use"            # the call is the variable's final appearance


def _role(variable: str, callable_: str, call_source: str) -> str:
    """What the variable IS at the call, read off the call text.

    receiver  the object the call is made on -- the agent/LLM/client, which is
              where model and sampling configuration lives
    argument  named in the call -- prompt, messages, params
    binding   not in the call text at all: the assignment target or a
              downstream value
    """
    if not variable:
        return ""
    v = re.escape(variable)
    if re.match(r"^\s*" + v + r"\s*[.\[(]", callable_ or ""):
        return "receiver"
    if re.search(r"(?<![\w.])" + v + r"\b", call_source or ""):
        return "argument"
    return "binding"


def _iter_lines(fh) -> Iterator[Tuple[int, int]]:
    r = csv.reader(fh)
    next(r, None)   # header
    for row in r:
        if row:
            yield int(row[0]), int(row[1])


def build(all_repos: bool, quiet: bool) -> Dict[str, Any]:
    started = time.time()
    call_rows, n_pop = load_calls(all_repos)

    # (slug, rel_path, line) -> [call row]; several call sites can share a line.
    by_line: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    call_meta: Dict[str, Dict[str, Any]] = {}
    for c in call_rows:
        slug, rel = _split_file(c["file"])
        key = (slug, rel, int(c["call_line"]))
        by_line[key].append(c)
        call_meta[c["call_id"]] = {"slug": slug, "rel": rel, "row": c}
    slugs_with_calls = {k[0] for k in by_line}

    # Which of those slugs were actually sliced at all (Stage 6 coverage).
    sliced_slugs = {e.name for e in os.scandir(paths.SLICES_DIR) if e.is_dir()}

    per_call: Dict[str, Dict[str, Any]] = {
        cid: {"n_slices": 0, "n_slices_nondup": 0, "variables": set(),
              "n_before": 0, "n_at": 0, "n_after": 0,
              "n_receiver": 0, "n_argument": 0, "n_binding": 0}
        for cid in call_meta
    }

    n_index_rows = 0
    n_slices_with_hit = 0
    n_slices_with_hit_nondup = 0
    n_hits = 0
    rel_counter: Counter = Counter()
    rel_counter_nondup: Counter = Counter()
    role_counter: Counter = Counter()
    role_counter_nondup: Counter = Counter()
    rel_role_counter: Counter = Counter()
    crit_counter: Counter = Counter()
    hit_repos: set = set()
    test_hits = 0
    unparseable_hits = 0

    hits_fh = SLICE_CALL_HITS_CSV.open("w", encoding="utf-8", newline="")
    try:
        hits_w = csv.writer(hits_fh)
        hits_w.writerow(HIT_FIELDS)

        # slice_index.csv and slice_pdg_lines.csv.gz are both emitted in
        # slice_id order, so this is a streaming merge join -- neither file is
        # ever held in memory.
        with gzip.open(SLICE_PDG_LINES_CSV_GZ, "rt", encoding="utf-8", newline="") as lfh, \
                SLICE_INDEX_CSV.open("r", encoding="utf-8", newline="") as ifh:
            lines_it = _iter_lines(lfh)
            pending: Optional[Tuple[int, int]] = next(lines_it, None)
            idx = csv.DictReader(ifh)

            for row in idx:
                n_index_rows += 1
                sid = int(row["slice_id"])

                # Collect this slice's pdg lines from the sorted line stream.
                lines: List[int] = []
                while pending is not None and pending[0] == sid:
                    lines.append(pending[1])
                    pending = next(lines_it, None)
                if pending is not None and pending[0] < sid:   # never expected
                    while pending is not None and pending[0] < sid:
                        pending = next(lines_it, None)

                slug = row["repo"]
                if slug not in slugs_with_calls or not lines:
                    continue
                rel = row["rel_path"]
                is_dup = row["is_duplicate"] == "1"
                c_first, c_last = row["criterion_first"], row["criterion_last"]
                n_crit = int(row["n_criterion"] or 0)
                crit_exact = 1 if n_crit == 1 else 0

                slice_hit = False
                for ln in lines:
                    for c in by_line.get((slug, rel, ln), ()):  # type: ignore[arg-type]
                        rel_kind = _relation(ln, c_first, c_last)
                        role = _role(row["variable"], str(c.get("callable") or ""),
                                     str(c.get("call_source") or ""))
                        hits_w.writerow([
                            sid, slug, rel, row["is_test_path"], row["method"], row["variable"],
                            row["is_duplicate"], row["parses"], row["n_pdg"],
                            n_crit, c_first, c_last, crit_exact,
                            c["call_id"], c["repo"], c["framework"], c["pattern"], ln,
                            c.get("enclosing_qname", ""), rel_kind, role,
                        ])
                        n_hits += 1
                        slice_hit = True
                        rel_counter[rel_kind] += 1
                        role_counter[role] += 1
                        rel_role_counter[(rel_kind, role)] += 1
                        if not is_dup:
                            rel_counter_nondup[rel_kind] += 1
                            role_counter_nondup[role] += 1
                        crit_counter[crit_exact] += 1
                        pc = per_call[c["call_id"]]
                        pc["n_slices"] += 1
                        if not is_dup:
                            pc["n_slices_nondup"] += 1
                        pc["variables"].add(row["variable"])
                        pc["n_before"] += rel_kind == "call_before_last_use"
                        pc["n_at"] += rel_kind == "call_at_last_use"
                        pc["n_after"] += rel_kind == "call_after_last_use"
                        pc["n_receiver"] += role == "receiver"
                        pc["n_argument"] += role == "argument"
                        pc["n_binding"] += role == "binding"

                if slice_hit:
                    n_slices_with_hit += 1
                    hit_repos.add(slug)
                    if not is_dup:
                        n_slices_with_hit_nondup += 1
                    if row["is_test_path"] == "1":
                        test_hits += 1
                    if row["parses"] == "0":
                        unparseable_hits += 1

                if not quiet and n_index_rows % 500000 == 0:
                    print("  {:>9,} index rows, {:,} hits".format(n_index_rows, n_hits),
                          file=sys.stderr, flush=True)
    finally:
        hits_fh.close()

    # Per-call-site table: every observed call site, whether or not a slice hit it.
    fw_total: Counter = Counter()
    fw_in_slice: Counter = Counter()
    with SLICE_CALLS_BY_CALL_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CALL_FIELDS)
        for cid, meta in call_meta.items():
            c = meta["row"]
            pc = per_call[cid]
            in_slice = 1 if pc["n_slices"] else 0
            fw_total[c["framework"]] += 1
            if in_slice:
                fw_in_slice[c["framework"]] += 1
            variables = sorted(pc["variables"])
            w.writerow([
                cid, c["repo"], meta["slug"], meta["rel"], c["framework"], c["pattern"],
                c["call_line"], c.get("enclosing_qname", ""),
                1 if meta["slug"] in sliced_slugs else 0, in_slice,
                pc["n_slices"], pc["n_slices_nondup"], len(variables),
                pc["n_before"], pc["n_at"], pc["n_after"],
                pc["n_receiver"], pc["n_argument"], pc["n_binding"],
                "|".join(variables[:8]),
            ])

    n_calls = len(call_meta)
    n_in_slice = sum(1 for pc in per_call.values() if pc["n_slices"])
    calls_in_sliced_repos = sum(1 for m in call_meta.values() if m["slug"] in sliced_slugs)

    def _pct(n: int, d: int) -> float:
        return round(100.0 * n / d, 2) if d else 0.0

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 1),
        "scope": "all_repos" if all_repos else "observed (make_figures.POP/CALLS)",
        "observed_repos": n_pop,
        "observed_call_sites": n_calls,
        "call_sites_in_a_sliced_repo": calls_in_sliced_repos,
        "pct_call_sites_in_a_sliced_repo": _pct(calls_in_sliced_repos, n_calls),
        "call_sites_inside_at_least_one_slice": n_in_slice,
        "pct_call_sites_inside_a_slice": _pct(n_in_slice, n_calls),
        "pct_call_sites_inside_a_slice_of_sliced": _pct(n_in_slice, calls_in_sliced_repos),
        "index_rows_scanned": n_index_rows,
        "slices_touching_a_call": n_slices_with_hit,
        "slices_touching_a_call_nondup": n_slices_with_hit_nondup,
        "pct_slices_touching_a_call": _pct(n_slices_with_hit, n_index_rows),
        "slice_call_pairs": n_hits,
        "repos_with_a_hit": len(hit_repos),
        "hits_in_test_paths": test_hits,
        "hits_unparseable_slice": unparseable_hits,
        "relation": dict(rel_counter),
        "role": dict(role_counter),
        "role_nondup": dict(role_counter_nondup),
        "pct_role": {k: _pct(v, n_hits) for k, v in role_counter.items()},
        "relation_x_role": {"{}|{}".format(a, b): v for (a, b), v in sorted(rel_role_counter.items())},
        "relation_nondup": dict(rel_counter_nondup),
        "pct_relation": {k: _pct(v, n_hits) for k, v in rel_counter.items()},
        "criterion_exact_hits": crit_counter.get(1, 0),
        "criterion_span_hits": crit_counter.get(0, 0),
        "pct_criterion_exact": _pct(crit_counter.get(1, 0), n_hits),
        "by_framework": {
            fw: {"call_sites": fw_total[fw], "in_slice": fw_in_slice.get(fw, 0),
                 "pct": _pct(fw_in_slice.get(fw, 0), fw_total[fw])}
            for fw in sorted(fw_total, key=lambda k: -fw_total[k])
        },
        "outputs": {
            "slice_call_hits_csv": str(SLICE_CALL_HITS_CSV),
            "slice_calls_by_call_csv": str(SLICE_CALLS_BY_CALL_CSV),
        },
    }
    SLICE_CALLS_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-repos", action="store_true",
                    help="do not restrict to the observed population")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not SLICE_INDEX_CSV.exists():
        print("no slice index; run: py -3.14 -m pipeline.slice_index", file=sys.stderr)
        return 2

    s = build(args.all_repos, args.quiet)
    print("\n{:,} observed call sites, {:,} in a sliced repo".format(
        s["observed_call_sites"], s["call_sites_in_a_sliced_repo"]))
    print("  inside >=1 slice : {:,} ({}% of observed, {}% of sliced)".format(
        s["call_sites_inside_at_least_one_slice"], s["pct_call_sites_inside_a_slice"],
        s["pct_call_sites_inside_a_slice_of_sliced"]))
    print("  slices touching a call: {:,} of {:,} ({}%)".format(
        s["slices_touching_a_call"], s["index_rows_scanned"], s["pct_slices_touching_a_call"]))
    print("  slice-call pairs      : {:,}".format(s["slice_call_pairs"]))
    print("  relation              : {}".format(s["relation"]))
    print("  role                  : {}".format(s["role"]))
    print("  wrote {}".format(SLICE_CALLS_SUMMARY_JSON.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
