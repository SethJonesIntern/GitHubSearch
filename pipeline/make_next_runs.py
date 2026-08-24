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

    prog = json.loads(paths.BATCH_PROGRESS_JSON.read_text(encoding="utf-8")) \
        if paths.BATCH_PROGRESS_JSON.exists() else {}
    done = set(prog.get("processed", []))
    for name in ("reclone.csv", "reslice.csv", "regraph.csv"):
        q = list(csv.DictReader((OUT_DIR / name).open(encoding="utf-8")))
        blocked = sum(1 for r in q if r["full_name"] in done)
        if blocked:
            print(f"  NOTE {name}: {blocked}/{len(q)} are still in .batch_progress.json "
                  f"— run prepare_rerun.py or --resume will skip them")


if __name__ == "__main__":
    main()
