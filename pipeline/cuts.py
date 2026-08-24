"""The population cuts, in one place, so every count drops the same repos.

A cut is decided once — in `pipeline/audit_apps.py` (`CUT`), or by hand in the
`in_scope` column of `application_audit.csv` — and lands here as the single set every
consumer reads. That matters because a cut has to move BOTH sides of a ratio: the
repo leaves the denominator and its rows leave the numerator. A script that reads
`applications_slim.csv` on its own silently keeps counting cut repos, which is how
`keep_frequency.csv` came to report 51 haystack apps when only 24 call deepset
Haystack (EXCLUSIONS.md §10).

Readers: `Applications/analyze.py` (folds this into `EXCLUDED`),
`Applications/keep_frequency.py`, `Applications/plot_coverage.py`.

Nothing is deleted anywhere — cut repos keep their rows in every CSV, carrying
`in_scope=0` and the reason in `notes`. This module only decides what gets COUNTED.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import paths  # noqa: E402

AUDIT_CSV = paths.ARTIFACTS_DIR / "application_audit.csv"

csv.field_size_limit(10 ** 9)


def _by_disposition(value: str) -> dict[str, str]:
    if not AUDIT_CSV.exists():
        return {}
    with AUDIT_CSV.open(newline="", encoding="utf-8") as fh:
        return {r["full_name"]: (r.get("notes") or "")
                for r in csv.DictReader(fh)
                if (r.get("in_scope") or "").strip() == value}


def cut_repos() -> dict[str, str]:
    """in_scope=0 — not an LLM application at all (collision tokens, junk matches).
    Leaves EVERY count, including the coverage denominator: it was never a real app."""
    return _by_disposition("0")


def uncovered_repos() -> dict[str, str]:
    """in_scope=uncovered — a real LLM application built on a framework outside the
    top-20, which we deliberately do not measure because the top-20 already covers
    ~90% of the population.

    Leaves the ANALYZED statistics (they'd dilute every prevalence number with repos
    whose framework we never match) but STAYS in the coverage denominator — quantifying
    the unmeasured tail is the entire purpose of the 827/918 = 90.1% figure, which
    collapses to a meaningless 100% if these are removed from it."""
    return _by_disposition("uncovered")


def excluded_from_stats() -> dict[str, str]:
    """Everything that must not appear in an analyzed statistic: junk AND the
    deliberately-unmeasured tail. This is the set `analyze.py` drops."""
    return {**cut_repos(), **uncovered_repos()}


def drop_cut(rows, key: str = "full_name"):
    """Filter dict rows for the COVERAGE view: only true non-apps are removed, so the
    uncovered tail still counts toward the denominator it exists to measure."""
    cut = cut_repos()
    return [r for r in rows if r.get(key) not in cut]
