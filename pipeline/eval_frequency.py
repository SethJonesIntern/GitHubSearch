"""Stage 7 — eval frequency table.

Derived purely from the eval-call piggyback (eval_calls_all.csv, produced by the
batch invoker search with the EVAL_CALLS seed dict). Reports, per evaluator
framework, how many applications actually *call* it and how many call sites
there are — the "called" view, no dependency-declaration check needed.

  eval_framework, repos_with_calls, total_call_sites, pct_of_apps

  repos_with_calls  distinct applications with >= 1 eval call site
  total_call_sites  eval call sites across all applications
  pct_of_apps       repos_with_calls as a % of all Stage 2 applications
                    (blank if applications.csv isn't available)

Writes pipeline/artifacts/eval_frequency.csv.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from typing import List, Optional

from pipeline import paths

FIELDS = ["eval_framework", "repos_with_calls", "total_call_sites", "pct_of_apps"]


def build_frequency(eval_call_rows: List[dict], total_apps: Optional[int] = None) -> List[dict]:
    """Aggregate eval call rows (one per call site) into per-framework counts."""
    agg: dict = defaultdict(lambda: {"repos": set(), "sites": 0})
    for r in eval_call_rows:
        fw = r.get("framework")
        if not fw:
            continue
        agg[fw]["repos"].add(r.get("repo"))
        agg[fw]["sites"] += 1

    rows = []
    for fw, d in agg.items():
        n = len(d["repos"])
        rows.append({
            "eval_framework": fw,
            "repos_with_calls": n,
            "total_call_sites": d["sites"],
            "pct_of_apps": round(100.0 * n / total_apps, 1) if total_apps else "",
        })
    rows.sort(key=lambda r: (-r["repos_with_calls"], r["eval_framework"]))
    return rows


def _read_rows(path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_applications() -> Optional[int]:
    """Distinct applications from Stage 2, for the percentage denominator."""
    rows = _read_rows(paths.APPLICATIONS_CSV)
    if not rows:
        return None
    return len({r.get("full_name") for r in rows if r.get("full_name")})


def main() -> None:
    paths.ensure_dirs()

    eval_rows = _read_rows(paths.EVAL_CALLS_CSV)
    if not paths.EVAL_CALLS_CSV.exists():
        raise FileNotFoundError(
            f"{paths.EVAL_CALLS_CSV} not found — run the analysis stage "
            f"(pipeline.batch_call_metadata) first.")

    total_apps = count_applications()
    table = build_frequency(eval_rows, total_apps)

    with open(paths.EVAL_FREQUENCY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(table)

    print(f"Eval frequency (apps with eval calls; {len(eval_rows)} call sites total"
          f"{f'; {total_apps} apps' if total_apps else ''})")
    if not table:
        print("  (no eval calls found)")
    else:
        width = max(len(r["eval_framework"]) for r in table)
        for r in table:
            pct = f"  {r['pct_of_apps']:>5}%" if r["pct_of_apps"] != "" else ""
            print(f"  {r['eval_framework']:<{width}}  repos={r['repos_with_calls']:>4}  "
                  f"sites={r['total_call_sites']:>5}{pct}")
    print(f"\nWrote {paths.EVAL_FREQUENCY_CSV}")


if __name__ == "__main__":
    main()
