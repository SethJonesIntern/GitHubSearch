"""Makes a queue in pipeline/next_runs/ actually re-runnable.

`batch_call_metadata --resume` skips any repo already in `.batch_progress.json`, and
its writers append without deduping. So re-running a repo that was processed before
is either a silent no-op (with --resume) or a duplicate-row corruption (without it),
and running WITHOUT --resume deletes every artifact and starts the 1,035-repo run
over. This script is the missing middle: it removes the queued repos from the
progress file and lifts their existing rows out of every artifact, so the re-run
writes them back exactly once.

Nothing is discarded. The lifted rows are written to
`artifacts/_rerun_backup_<timestamp>/` — only the removed rows, not a copy of the
189 MB originals — so a queue can be restored by appending them back.

    py -3.14 -m pipeline.prepare_rerun pipeline/next_runs/regraph.csv          # dry run
    py -3.14 -m pipeline.prepare_rerun pipeline/next_runs/regraph.csv --apply

Then, and only then:

    py -3.14 -m pipeline.batch_call_metadata --input pipeline/next_runs/regraph.csv --resume ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import paths

csv.field_size_limit(10 ** 9)

# Every per-repo artifact Stage 5/7 appends to. call_graph_health is included because
# a stale health row would otherwise survive a graph repair and keep reporting the
# old cg_source.
ARTIFACTS = [
    paths.LLM_INVOKERS_CSV, paths.LLM_CALLS_CSV, paths.CALL_METADATA_CSV,
    paths.LLM_TESTS_CSV, paths.EVAL_INVOKERS_CSV, paths.EVAL_CALLS_CSV,
    paths.EVAL_CALL_METADATA_CSV, paths.CALL_GRAPH_HEALTH_CSV,
]


def targets(queue: Path) -> set[str]:
    with queue.open(encoding="utf-8") as f:
        return {r["full_name"] for r in csv.DictReader(f) if r.get("full_name")}


def strip_rows(path: Path, repos: set[str], backup_dir: Path, apply: bool) -> tuple[int, int]:
    """-> (rows removed, rows kept). Streams, so a 189 MB artifact never loads."""
    if not path.exists() or path.stat().st_size == 0:
        return 0, 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    removed_path = backup_dir / path.name
    kept = removed = 0
    with path.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        fields = reader.fieldnames
        if not fields or "repo" not in fields:
            return 0, 0
        out = tmp.open("w", newline="", encoding="utf-8") if apply else None
        bak = removed_path.open("w", newline="", encoding="utf-8") if apply else None
        try:
            if apply:
                w = csv.DictWriter(out, fieldnames=fields, quoting=csv.QUOTE_ALL)
                w.writeheader()
                b = csv.DictWriter(bak, fieldnames=fields, quoting=csv.QUOTE_ALL)
                b.writeheader()
            for row in reader:
                if row["repo"] in repos:
                    removed += 1
                    if apply:
                        b.writerow(row)
                else:
                    kept += 1
                    if apply:
                        w.writerow(row)
        finally:
            if out:
                out.close()
            if bak:
                bak.close()
    if apply:
        if removed:
            # os.replace, NOT shutil.move: on Windows os.rename refuses an existing
            # destination, so shutil.move falls back to copy2 -- which rewrites the
            # 189 MB original in place and leaves it TRUNCATED if interrupted.
            # os.replace is atomic here and does not copy.
            os.replace(tmp, path)
        else:
            tmp.unlink(missing_ok=True)
            removed_path.unlink(missing_ok=True)
    return removed, kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue", type=Path, help="a CSV in pipeline/next_runs/")
    ap.add_argument("--apply", action="store_true",
                    help="actually modify the artifacts (default: report only)")
    args = ap.parse_args()

    repos = targets(args.queue)
    print(f"# queue {args.queue.name}: {len(repos)} repos")

    prog = json.loads(paths.BATCH_PROGRESS_JSON.read_text(encoding="utf-8")) \
        if paths.BATCH_PROGRESS_JSON.exists() else {"processed": []}
    in_progress = [r for r in prog.get("processed", []) if r in repos]
    print(f"#   in .batch_progress.json['processed'] : {len(in_progress)}"
          f"   <- these would be SILENTLY SKIPPED by --resume")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = paths.ARTIFACTS_DIR / f"_rerun_backup_{stamp}"
    if args.apply:
        backup.mkdir(parents=True, exist_ok=True)

    total = 0
    for path in ARTIFACTS:
        removed, kept = strip_rows(path, repos, backup, args.apply)
        total += removed
        if removed:
            print(f"#   {path.name:<28} {removed:>8} rows "
                  f"{'lifted' if args.apply else 'would be lifted'} ({kept} kept)")
    print(f"#   {'total':<28} {total:>8} rows")

    if args.apply:
        prog["processed"] = [r for r in prog.get("processed", []) if r not in repos]
        for key in ("failed", "slice_failed"):
            if key in prog:
                prog[key] = [e for e in prog[key] if e.get("repo") not in repos]
        paths.BATCH_PROGRESS_JSON.write_text(json.dumps(prog), encoding="utf-8")
        print(f"\n# applied. removed rows saved to {backup.relative_to(_ROOT)}")
        print(f"# progress['processed'] now {len(prog['processed'])} repos")
        print(f"# next: py -3.14 -m pipeline.batch_call_metadata --input {args.queue} --resume ...")
    else:
        print("\n# DRY RUN — nothing changed. Re-run with --apply to make the queue runnable.")


if __name__ == "__main__":
    main()
