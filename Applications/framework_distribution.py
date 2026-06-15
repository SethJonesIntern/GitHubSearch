"""Report the distribution of which agent frameworks are imported, and at
what percentage, across the repos found by search_candidates.py.

Three views, from broadest to narrowest:

  1. Imported  — distinct repos containing each framework's import pattern,
                 captured at search time before quality filters
                 (progress["framework_repo_counts"]). Best "popularity" signal.
  2. Candidate — repos that passed the search-time filters (stars / push date)
                 and became candidates (progress["candidates"]).
  3. Kept      — repos that survived full enrichment and made the final CSV
                 (matched_frameworks column).

A repo can import several frameworks, so per-framework percentages are of the
relevant repo population and need not sum to 100%.

Run any time — including while search_candidates.py is still going, since it
reads the resumable progress file.
"""
import csv
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE = os.path.join(HERE, ".search_progress.json")
OUTPUT_CSV = os.path.join(HERE, "application_candidates_v2.csv")


def _load_progress() -> dict:
    if not os.path.exists(PROGRESS_FILE):
        return {}
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _kept_framework_counts() -> "tuple[Counter, int]":
    counts: Counter = Counter()
    total = 0
    if not os.path.exists(OUTPUT_CSV):
        return counts, total
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
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
    if denom:
        print(f"\n  population (repos) = {denom}; percentages are of this population "
              f"(a repo may import multiple frameworks, so columns can exceed 100%).")


def main() -> None:
    progress = _load_progress()

    repo_counts = Counter(progress.get("framework_repo_counts", {}))
    file_matches = progress.get("framework_file_matches", {})
    imported_total = sum(repo_counts.values())  # repo-import incidences, not unique repos
    completed = len(progress.get("completed_search_terms", []))

    print(f"Search terms completed: {completed}")

    if repo_counts:
        # Percentage here is share of total import incidences across frameworks,
        # i.e. "of all framework-imports we saw, what fraction were X".
        _print_table(
            "1. Imported — distinct repos per framework (pre-filter popularity)",
            repo_counts, imported_total, extra=file_matches,
        )

    cand_counts, cand_total = _candidate_framework_counts(progress)
    if cand_total:
        _print_table(
            "2. Candidate — repos passing search-time filters",
            cand_counts, cand_total,
        )

    kept_counts, kept_total = _kept_framework_counts()
    if kept_total:
        _print_table(
            "3. Kept — repos in final CSV",
            kept_counts, kept_total,
        )

    if not (repo_counts or cand_total or kept_total):
        print("\nNo results yet. Run search_candidates.py first.")


if __name__ == "__main__":
    main()
