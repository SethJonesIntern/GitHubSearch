"""Stage 3 — framework frequency table.

How many applications import which framework, across three views (broadest to
narrowest), all keyed by the importable package name Stage 2 searched:

  1. Imported  — distinct repos importing each name at search time, before the
                 quality filters (progress["framework_repo_counts"]). Popularity.
  2. Candidate — repos that passed the search-time filters (progress["candidates"]).
  3. Kept      — repos that survived enrichment and made applications.csv
                 (matched_frameworks column).

A repo can import several frameworks, so per-framework counts need not sum to the
population. Prints the three tables and writes a durable framework_frequency.csv.

Run any time — including while search_candidates.py is still going, since it
reads the resumable progress file.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths  # noqa: E402

PROGRESS_FILE = paths.SEARCH_PROGRESS_JSON
APPLICATIONS_CSV = paths.APPLICATIONS_CSV
FREQUENCY_CSV = paths.FRAMEWORK_FREQUENCY_CSV

FIELDS = ["framework", "repos_imported", "repos_candidate", "repos_kept", "pct_of_kept"]


def _load_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return {}
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _kept_framework_counts() -> "tuple[Counter, int]":
    counts: Counter = Counter()
    total = 0
    if not APPLICATIONS_CSV.exists():
        return counts, total
    with open(APPLICATIONS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            fws = [s.strip() for s in (row.get("matched_frameworks") or "").split(",") if s.strip()]
            for fw in set(fws):
                counts[fw] += 1
    return counts, total


def _candidate_framework_counts(progress: dict) -> "tuple[Counter, int]":
    counts: Counter = Counter()
    candidates = progress.get("candidates", {})
    for saved in candidates.values():
        for fw in set(saved.get("frameworks", [])):
            counts[fw] += 1
    return counts, len(candidates)


def build_frequency_rows(repo_counts: Counter, candidate_counts: Counter,
                         kept_counts: Counter, kept_total: int) -> List[dict]:
    """One row per framework (union of all three views), sorted by kept then
    imported. Pure — drives both the CSV and tests."""
    names = set(repo_counts) | set(candidate_counts) | set(kept_counts)
    rows = []
    for name in names:
        kept = kept_counts.get(name, 0)
        rows.append({
            "framework": name,
            "repos_imported": repo_counts.get(name, 0),
            "repos_candidate": candidate_counts.get(name, 0),
            "repos_kept": kept,
            "pct_of_kept": round(100.0 * kept / kept_total, 1) if kept_total else "",
        })
    rows.sort(key=lambda r: (-r["repos_kept"], -r["repos_imported"], r["framework"]))
    return rows


def _print_table(title: str, counts: Counter, denom: int, extra: dict = None) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    if denom == 0 and not counts:
        print("  (no data yet)")
        return
    width = max((len(k) for k in counts), default=10)
    header = f"  {'framework':<{width}}  {'count':>7}"
    if denom:
        header += f"  {'pct':>7}"
    if extra is not None:
        header += f"  {'files':>7}"
    print(header)
    for fw, n in counts.most_common():
        line = f"  {fw:<{width}}  {n:>7}"
        if denom:
            line += f"  {100.0 * n / denom:>6.1f}%"
        if extra is not None:
            line += f"  {extra.get(fw, 0):>7}"
        print(line)


def main() -> None:
    paths.ensure_dirs()
    progress = _load_progress()

    repo_counts = Counter(progress.get("framework_repo_counts", {}))
    file_matches = progress.get("framework_file_matches", {})
    imported_total = sum(repo_counts.values())
    completed = len(progress.get("completed_search_terms", []))
    print(f"Import names searched: {completed}")

    cand_counts, cand_total = _candidate_framework_counts(progress)
    kept_counts, kept_total = _kept_framework_counts()

    if repo_counts:
        _print_table("1. Imported — distinct repos per import name (pre-filter popularity)",
                     repo_counts, imported_total, extra=file_matches)
    if cand_total:
        _print_table("2. Candidate — repos passing search-time filters", cand_counts, cand_total)
    if kept_total:
        _print_table("3. Kept — applications in applications.csv", kept_counts, kept_total)

    # Durable artifact (all three views, one row per framework).
    rows = build_frequency_rows(repo_counts, cand_counts, kept_counts, kept_total)
    with open(FREQUENCY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    if not (repo_counts or cand_total or kept_total):
        print("\nNo results yet. Run Stage 2 (search_candidates.py) first.")
    print(f"\nWrote {FREQUENCY_CSV} ({len(rows)} frameworks)")


if __name__ == "__main__":
    main()
