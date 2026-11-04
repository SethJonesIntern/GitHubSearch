"""Builds the targeted re-run queues in pipeline/next_runs/.

Each file is a curated `--input` for `batch_call_metadata`: the same schema as
`applications_slim.csv` (so every other script can read it too) plus a `reason`
column saying why the repo is queued. Regenerate any time — these are derived from
the audit sheet and the progress file, never hand-edited.

    reclone.csv   never analysed: the clone failed, or its checkout is gone. Nothing
                  downstream has data for these repos at all.
    reslice.csv   analysed, but Stage 6 (Joern) failed — call/invoker data is fine,
                  the per-variable slices are missing.
    regraph.csv   analysed, but pyan produced no usable call graph, so the repo has
                  direct invokers only and NO transitive closure. These are the rows
                  that make every transitive number an undercount; they need the
                  `ast.parse` pre-filter or a no-timer re-run, not just a re-clone.

Read `pipeline/prepare_rerun.py` before running any of them: re-running a repo that
is still in `.batch_progress.json` is a silent no-op, and re-running one whose rows
are still in the artifacts duplicates them.

    py -3.14 -m pipeline.make_next_runs
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import paths

OUT_DIR = paths.PIPELINE_DIR / "next_runs"
AUDIT_CSV = paths.ARTIFACTS_DIR / "application_audit.csv"
SLIM_CSV = paths.ARTIFACTS_DIR / "applications_slim.csv"

csv.field_size_limit(10 ** 9)


def write_queue(name: str, rows: list[dict], fields: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.relative_to(_ROOT)}  ({len(rows)} repos)")


def main() -> None:
    slim = {r["full_name"]: r for r in csv.DictReader(SLIM_CSV.open(encoding="utf-8"))}
    audit = list(csv.DictReader(AUDIT_CSV.open(encoding="utf-8")))
    fields = list(next(iter(slim.values())).keys()) + ["reason"]

    def rows_for(pred, reason) -> list[dict]:
        out = []
        for a in audit:
            if not pred(a):
                continue
            base = slim.get(a["full_name"])
            if not base:
                continue
            r = dict(base)
            r["reason"] = reason(a) if callable(reason) else reason
            out.append(r)
        return out

    on_disk = {e.name for e in os.scandir(paths.REPOS_DIR)} if paths.REPOS_DIR.is_dir() else set()

    def needs_clone(a):
        return a["clone_failed"] == "1" or a["clone_slug"] not in on_disk

    def clone_reason(a):
        if a["clone_failed"] == "1" and a["clone_slug"] not in on_disk:
            return "clone failed; no checkout on disk"
        if a["clone_failed"] == "1":
            return "clone recorded as failed, but a checkout exists — never analysed"
        return "checkout missing from pipeline/repos"

    print("next-run queues:")
    write_queue("reclone.csv", rows_for(needs_clone, clone_reason), fields)
    write_queue("reslice.csv",
                rows_for(lambda a: a["slice_status"] == "failed",
                         lambda a: f"joern slice failed: {a['slice_error'] or 'unknown'}"),
                fields)
    write_queue("regraph.csv",
                rows_for(lambda a: a["processed"] == "1" and a["graph_usable"] != "True",
                         lambda a: (f"pyan produced no usable graph (cg_source="
                                    f"{a['cg_source'] or 'none'}, "
                                    f"{a['invokers_direct']} direct invokers, "
                                    f"0 transitive)")),
                fields)

    # rerun.csv -- the queue to ACTUALLY run. reslice and regraph overlap on 11 repos
    # (and prepare_rerun is per-queue), so running them as two queues either processes
    # those 11 twice or, in the other order, lets --resume silently skip them and leaves
    # them unsliced. One merged queue makes each repo run exactly once with no ordering
    # to get wrong. reclone is disjoint from both but folded in so a single run clears
    # everything; it is listed FIRST so the 20 cheap "no data at all" repos land before
    # the expensive ones.
    #
    # Repos with no .py file at all are dropped: a Python static-analysis pass can never
    # produce a graph for them, so re-running only reproduces the same empty result.
    # (They keep their audit row and their in_scope value -- this is a run-queue filter,
    # not a population cut.)
    def py_count(a) -> int:
        d = paths.REPOS_DIR / a["clone_slug"]
        if not d.is_dir():
            return -1          # not cloned yet: reclone will fetch it, keep it
        return sum(1 for _ in d.rglob("*.py"))

    def needs_rerun(a):
        return (needs_clone(a)
                or a["slice_status"] == "failed"
                or (a["processed"] == "1" and a["graph_usable"] != "True"))

    def rerun_reason(a):
        if needs_clone(a):
            return clone_reason(a)
        why = []
        if a["processed"] == "1" and a["graph_usable"] != "True":
            # Carry the exclusion count: 0 means pyan never completed even a FIRST pass,
            # which the ast.parse pre-filter may not help and which a no-timer re-run can
            # leave running for hours. Non-zero means it was time-boxed mid-retry.
            why.append(f"no usable graph (cg_source={a['cg_source'] or 'none'}, "
                       f"{a['graph_excluded_files'] or 0} files excluded, "
                       f"{a['invokers_direct']} direct invokers, 0 transitive)")
        if a["slice_status"] == "failed":
            why.append(f"joern slice failed: {a['slice_error'] or 'unknown'}")
        return "; ".join(why)

    merged, skipped = [], []
    for a in audit:
        if not needs_rerun(a) or a["full_name"] not in slim:
            continue
        # ...but only for repos whose checkout is COMPLETE. A reclone repo's checkout
        # is the failed/partial one, so its .py count says nothing about what a fresh
        # clone will contain (kagenti/agent-examples has 0 .py on disk and is alive on
        # GitHub). Never drop a repo we are about to re-clone.
        if not needs_clone(a) and py_count(a) == 0:
            skipped.append(a["full_name"])
            continue
        r = dict(slim[a["full_name"]])
        r["reason"] = rerun_reason(a)
        merged.append((0 if needs_clone(a) else 1, r))
    merged.sort(key=lambda t: t[0])
    write_queue("rerun.csv", [r for _, r in merged], fields)
    if skipped:
        print(f"  NOTE rerun.csv: dropped {len(skipped)} repo(s) with no .py file at all "
              f"(a graph is impossible, not a pyan failure): {', '.join(skipped)}")

    prog = json.loads(paths.BATCH_PROGRESS_JSON.read_text(encoding="utf-8")) \
        if paths.BATCH_PROGRESS_JSON.exists() else {}
    done = set(prog.get("processed", []))
    for name in ("reclone.csv", "reslice.csv", "regraph.csv", "rerun.csv"):
        q = list(csv.DictReader((OUT_DIR / name).open(encoding="utf-8")))
        blocked = sum(1 for r in q if r["full_name"] in done)
        if blocked:
            print(f"  NOTE {name}: {blocked}/{len(q)} are still in .batch_progress.json "
                  f"— run prepare_rerun.py or --resume will skip them")


if __name__ == "__main__":
    main()
