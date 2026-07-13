"""Repeatable research cross-tabs over the Stage-5 outputs.

Loads the analysis CSVs and produces the standard conclusions:
  - population + LLM-usage prevalence
  - non-deterministic tests (headline): totals, per-framework, direct vs transitive
  - determinism knobs (temperature/seed/model set? literal or variable?)
  - call characteristics (per framework, sync vs async)
  - eval-framework usage

Prints a report and writes per-question CSVs under artifacts/analysis/.
Run: py -3.14 Applications/analyze.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Wrapper"))
from pipeline import paths  # noqa: E402
from pipeline import engines as E  # noqa: E402  (FRAMEWORK_CALLS)

A = paths.ARTIFACTS_DIR
OUT = A / "analysis"

# (framework, pattern) pairs still valid under the CURRENT FrameworkDict. Rows in
# already-generated CSVs matched by a since-removed pattern (e.g. agno's `Agent`
# constructor) are dropped here, so the pilot numbers reflect the fixed patterns
# without re-running Stage 5. Once a run is done with the fixed dict this is a no-op.
VALID = {(fw, p) for fw, pats in E.FRAMEWORK_CALLS.items() for p in pats}

# call kwargs whose value governs (non-)determinism
KNOBS = ["temperature", "top_p", "top_k", "seed", "max_tokens",
         "frequency_penalty", "presence_penalty", "model"]


def load(name: str) -> pd.DataFrame:
    p = A / name
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    return pd.DataFrame()


def fw_from_reason(s) -> str | None:
    m = re.search(r"from (\S+)\s*$", str(s))
    return m.group(1) if m else None


def truthy(series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def drop_removed_patterns_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose (framework, pattern) is still in FRAMEWORK_CALLS."""
    if df.empty or "pattern" not in df.columns:
        return df
    mask = [(f, p) in VALID for f, p in zip(df["framework"], df["pattern"])]
    return df[mask]


def drop_removed_patterns_invokers(df: pd.DataFrame) -> pd.DataFrame:
    """Same, for invoker/test rows whose pattern lives in the `reason` string
    ('matches '<pattern>' from <framework>'). Only DIRECT rows carry a pattern;
    transitive rows reach a seed via the call graph and are kept as-is (they'll be
    recomputed exactly on the next full run with the fixed patterns)."""
    if df.empty or "reason" not in df.columns:
        return df
    fw = df["reason"].str.extract(r"from (\S+)\s*$")[0]
    pat = df["reason"].str.extract(r"matches '([^']+)'")[0]
    valid = pd.Series([(f, p) in VALID for f, p in zip(fw, pat)], index=df.index)
    is_direct = df["kind"].eq("direct") if "kind" in df.columns else pd.Series(True, index=df.index)
    return df[(~is_direct) | valid]


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save(df: pd.DataFrame, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


def main():
    apps = load("applications_slim.csv")
    invokers = drop_removed_patterns_invokers(load("llm_invokers_all.csv"))
    tests = drop_removed_patterns_invokers(load("llm_tests_all.csv"))
    calls = drop_removed_patterns_calls(load("llm_calls_all.csv"))
    meta = drop_removed_patterns_calls(load("call_metadata_all.csv"))
    ev_calls = load("eval_calls_all.csv")
    ev_inv = load("eval_invokers_all.csv")

    prog = json.loads(paths.BATCH_PROGRESS_JSON.read_text()) \
        if paths.BATCH_PROGRESS_JSON.exists() else {}
    analyzed = set(prog.get("processed", []))
    n_analyzed = len(analyzed) or (invokers["repo"].nunique() if not invokers.empty else 0)

    # ── population + prevalence ───────────────────────────────────────────────
    section("POPULATION & LLM-USAGE PREVALENCE")
    print(f"applications (slimmed population) : {len(apps)}")
    print(f"repos analyzed (Stage 5 processed): {n_analyzed}")
    if n_analyzed:
        def repos_with(df):
            return df["repo"].nunique() if not df.empty else 0
        for label, df in [(">=1 LLM call site", calls), (">=1 LLM invoker", invokers),
                          (">=1 LLM test", tests), (">=1 eval call", ev_calls)]:
            r = repos_with(df)
            print(f"  repos with {label:<18}: {r:>5}  ({100 * r / n_analyzed:.0f}% of analyzed)")

    # ── non-deterministic tests (headline) ────────────────────────────────────
    section("NON-DETERMINISTIC TESTS  (tests that invoke a real LLM)")
    if tests.empty:
        print("no llm_tests rows yet.")
    else:
        uniq = tests.drop_duplicates(["repo", "qname"])
        print(f"distinct non-deterministic tests : {len(uniq)}")
        print(f"repos containing them            : {uniq['repo'].nunique()}")
        print(f"direct vs transitive             : "
              f"{(tests['kind'] == 'direct').sum()} direct / "
              f"{(tests['kind'] == 'transitive').sum()} transitive")
        by_repo = (uniq.groupby("repo").size().sort_values(ascending=False)
                   .rename("nd_tests").reset_index())
        save(by_repo, "nd_tests_by_repo.csv")
        print("\ntop repos by non-deterministic tests:")
        print(by_repo.head(10).to_string(index=False))
        tests = tests.assign(framework=tests["reason"].map(fw_from_reason))
        by_fw = (tests.drop_duplicates(["repo", "qname", "framework"])
                 .groupby("framework").size().sort_values(ascending=False)
                 .rename("nd_tests").reset_index())
        save(by_fw, "nd_tests_by_framework.csv")
        print("\nnon-deterministic tests by framework (a test can count for >1):")
        print(by_fw.head(12).to_string(index=False))

    # ── determinism knobs ─────────────────────────────────────────────────────
    section("DETERMINISM KNOBS  (are calls pinned to deterministic settings?)")
    if meta.empty:
        print("no call_metadata rows yet.")
    else:
        total_calls = meta["call_id"].nunique()
        print(f"distinct LLM calls with argument metadata: {total_calls}")
        rows = []
        for k in KNOBS:
            sub = meta[meta["arg_keyword"] == k]
            cw = sub["call_id"].nunique()
            lit = sub[truthy(sub["arg_is_literal"])]["call_id"].nunique()
            rows.append({"kwarg": k, "calls_setting_it": cw,
                         "pct_of_calls": round(100 * cw / total_calls, 1) if total_calls else 0,
                         "literal_value": lit, "variable_value": cw - lit})
        knobs = pd.DataFrame(rows)
        save(knobs, "determinism_knobs.csv")
        print(knobs.to_string(index=False))
        print("(literal_value = hard-coded in the call; variable_value = passed via a variable)")

    # ── call characteristics ──────────────────────────────────────────────────
    section("LLM CALL CHARACTERISTICS")
    if calls.empty:
        print("no llm_calls rows yet.")
    else:
        by_fw = (calls.groupby("framework")["call_id"].nunique()
                 .sort_values(ascending=False).rename("calls").reset_index())
        save(by_fw, "calls_by_framework.csv")
        print("LLM calls by framework:")
        print(by_fw.head(15).to_string(index=False))
        if "is_await" in calls.columns:
            aw = truthy(calls["is_await"]).sum()
            print(f"\nawait (async) calls: {aw} / {len(calls)} ({100 * aw / len(calls):.0f}%)")

    # ── eval usage ────────────────────────────────────────────────────────────
    section("EVALUATION-FRAMEWORK USAGE")
    if ev_calls.empty and ev_inv.empty:
        print("no eval usage detected.")
    else:
        if n_analyzed and not ev_calls.empty:
            r = ev_calls["repo"].nunique()
            print(f"repos calling an evaluator: {r} ({100 * r / n_analyzed:.0f}% of analyzed)")
        if not ev_calls.empty:
            by_fw = (ev_calls.groupby("framework")["call_id"].nunique()
                     .sort_values(ascending=False).rename("eval_calls").reset_index())
            save(by_fw, "eval_calls_by_framework.csv")
            print(by_fw.head(12).to_string(index=False))

    print(f"\nCross-tab CSVs written to {OUT}")


if __name__ == "__main__":
    main()
