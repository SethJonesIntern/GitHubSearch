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
from collections import Counter
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Wrapper"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import keep_frequency as kf  # noqa: E402  (category() — the one grouping table)
from pipeline import paths  # noqa: E402
from pipeline import cuts  # noqa: E402
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

# Repos dropped from the study population as a code-quality filter: too many source
# files fail to parse (real Python SyntaxErrors, per the scope-vs-syntax audit) for the
# codebase to count as a maintained application. Criterion: >=10 unparseable files —
# which cleanly isolates these two outliers (next-worst repo has 9). Counts: atom 31,
# Hands-On-AI-Engineering 56. Filtered here (not deleted from the CSVs) so the criterion
# stays on the record and the raw data is preserved.
QUALITY_EXCLUDED = {
    "rush86999/atom": "31 unparseable files",
    "Sumanth077/Hands-On-AI-Engineering": "56 unparseable files",
}

# Not an LLM application: Stage-2 pulled these in by an import-name collision, but Stage
# 5 found 0 LLM invokers / 0 LLM calls in a healthy (or, for sunnypilot, unreadable)
# graph — the matched framework token doesn't reflect real usage. Dropped from the
# population as false positives, separate from the code-quality filter above.
NOT_LLM_APP = {
    "sunnypilot/sunnypilot": "0 invokers; agno name-collision (driver-assistance app)",
}


# From the audit sheet. Both dispositions leave every number in THIS report — it must
# be in scale with what is actually analyzed — but they differ downstream: `uncovered`
# repos remain in the coverage denominator (see pipeline/cuts.py), cut repos do not.
AUDIT_CUT = cuts.cut_repos()
AUDIT_UNCOVERED = cuts.uncovered_repos()
EXCLUDED = {**QUALITY_EXCLUDED, **NOT_LLM_APP, **AUDIT_CUT, **AUDIT_UNCOVERED}


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


def drop_fp(df: pd.DataFrame) -> pd.DataFrame:
    """Drop calls flagged as false positives (fp_tier non-empty) — the receiver/syntax
    collisions from false_positives.classify_fp (EXCLUSIONS.md §6). Centralized here so
    every count is on clean data by default; the raw rows remain in the CSV for audit.
    No-op on data that predates the fp_tier column."""
    if df.empty or "fp_tier" not in df.columns:
        return df
    return df[df["fp_tier"].fillna("").astype(str).str.strip() == ""]


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


# Raw SDK packages: real LLM calls, but not a framework. Split out so a framework
# ranking isn't topped by "people call the OpenAI SDK directly" (SPRINT_HANDOFF §7.3).
RAW_SDKS = {"openai", "anthropic"}


def group_of(name):
    """Fold a fragmented package family into its one framework — `langchain_core`,
    `langchain_openai`, `langchain_anthropic` ... are all langchain. Same table the
    app ranking and the coverage figure use, so all three agree.

    Missing stays missing: transitive invoker rows carry no pattern, so their
    framework is NaN and groupby must keep dropping them rather than collecting them
    under a literal "nan"."""
    if pd.isna(name):
        return name
    return kf.category(str(name))


def kind_of(name: str) -> str:
    return "raw SDK" if name in RAW_SDKS else "framework"


def by_import_name(df: pd.DataFrame, keys: list, label: str) -> pd.DataFrame:
    """The ungrouped per-package view, kept alongside every grouped table so the
    family breakdown is never lost."""
    return (df.drop_duplicates(keys + ["framework"]).groupby("framework").size()
            .sort_values(ascending=False).rename(label).reset_index())


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save(df: pd.DataFrame, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


def main():
    apps = load("applications_slim.csv")
    invokers = drop_removed_patterns_invokers(load("llm_invokers_all.csv"))
    tests = drop_removed_patterns_invokers(load("llm_tests_all.csv"))
    calls = drop_fp(drop_removed_patterns_calls(load("llm_calls_all.csv")))
    meta = drop_fp(drop_removed_patterns_calls(load("call_metadata_all.csv")))
    ev_calls = load("eval_calls_all.csv")
    ev_inv = load("eval_invokers_all.csv")

    # Drop code-quality-excluded repos from every frame before any counting.
    def _drop_excluded(df, col):
        if df.empty or col not in df.columns:
            return df
        return df[~df[col].isin(EXCLUDED)]
    apps = _drop_excluded(apps, "full_name")
    invokers, tests = _drop_excluded(invokers, "repo"), _drop_excluded(tests, "repo")
    calls, meta = _drop_excluded(calls, "repo"), _drop_excluded(meta, "repo")
    ev_calls, ev_inv = _drop_excluded(ev_calls, "repo"), _drop_excluded(ev_inv, "repo")

    prog = json.loads(paths.BATCH_PROGRESS_JSON.read_text()) \
        if paths.BATCH_PROGRESS_JSON.exists() else {}
    analyzed = set(prog.get("processed", [])) - set(EXCLUDED)
    n_analyzed = len(analyzed) or (invokers["repo"].nunique() if not invokers.empty else 0)
    if QUALITY_EXCLUDED:
        print(f"[code-quality filter, >=10 unparseable files] dropped "
              f"{len(QUALITY_EXCLUDED)}: " +
              ", ".join(f"{r} ({why})" for r, why in QUALITY_EXCLUDED.items()))
    if NOT_LLM_APP:
        print(f"[false positive, not an LLM app] dropped {len(NOT_LLM_APP)}: " +
              ", ".join(f"{r} ({why})" for r, why in NOT_LLM_APP.items()))
    if AUDIT_CUT:
        by_reason = {}
        for repo, why in AUDIT_CUT.items():
            by_reason.setdefault(why, []).append(repo)
        print(f"[audit sheet, in_scope=0, not an LLM app] dropped {len(AUDIT_CUT)}:")
        for why, repos in sorted(by_reason.items()):
            print(f"    {len(repos):>3}  {why}")
    if AUDIT_UNCOVERED:
        # One line per distinct framework COMBINATION would be dozens of rows; the
        # useful summary is which unmeasured frameworks the tail actually runs on.
        tail = re.compile(r"imports only ([^—]+?) —")
        names = Counter(n.strip() for why in AUDIT_UNCOVERED.values()
                        for m in tail.findall(why) for n in m.split(","))
        print(f"[audit sheet, in_scope=uncovered] dropped {len(AUDIT_UNCOVERED)} real LLM "
              f"apps built only on frameworks/SDKs outside the top-20:")
        print("    " + ", ".join(f"{n} {c}" for n, c in names.most_common(10)))

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

    # ── call-graph health (can we trust the transitive numbers?) ──────────────
    section("CALL-GRAPH HEALTH  (is low transitive real, or a pyan artifact?)")
    health = _drop_excluded(load("call_graph_health.csv"), "repo")
    if health.empty:
        print("no call_graph_health.csv yet — populates on the next Stage-5 run.")
    else:
        n = len(health)
        usable = truthy(health["graph_usable"])
        print(f"repos with a usable (non-empty) call graph: {int(usable.sum())}/{n} "
              f"({100 * usable.mean():.0f}%)")
        print("cg_source:", {k: int(v) for k, v in health["cg_source"].value_counts().items()})
        if "excluded_files" in health.columns:
            resilient = health[health["cg_source"] == "pyan_resilient"]
            if not resilient.empty:
                print(f"pyan_resilient repos: {len(resilient)} "
                      f"(recovered by dropping {int(resilient['excluded_files'].sum())} "
                      f"bad files total; median {resilient['excluded_files'].median():.0f}/repo)")
        print(f"median graph coverage (nodes/functions): "
              f"{health['graph_coverage_pct'].median():.0f}%")
        print("\nLLM invokers split by graph health — transitive on empty graphs is an artifact:")
        for label, sub in [("usable-graph repos", health[usable]),
                           ("empty-graph repos ", health[~usable])]:
            d = int(sub["llm_direct_invokers"].sum())
            t = int(sub["llm_transitive_invokers"].sum())
            print(f"   {label}: {d} direct / {t} transitive")

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
        save(by_import_name(tests, ["repo", "qname"], "nd_tests"),
             "nd_tests_by_import_name.csv")
        # Grouped: a test reaching langchain_core AND langchain_openai is ONE langchain
        # test, so the dedupe key uses the group, not the package.
        tests = tests.assign(framework=tests["framework"].map(group_of))
        by_fw = (tests.drop_duplicates(["repo", "qname", "framework"])
                 .groupby("framework").size().sort_values(ascending=False)
                 .rename("nd_tests").reset_index())
        by_fw["kind"] = by_fw["framework"].map(kind_of)
        save(by_fw, "nd_tests_by_framework.csv")
        print("\nnon-deterministic tests by framework, grouped "
              "(a test can count for >1 framework):")
        print(by_fw.head(15).to_string(index=False))

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
        raw = (calls.groupby("framework")["call_id"].nunique()
               .sort_values(ascending=False).rename("calls").reset_index())
        save(raw, "calls_by_import_name.csv")
        by_fw = (calls.assign(framework=calls["framework"].map(group_of))
                 .groupby("framework")["call_id"].nunique()
                 .sort_values(ascending=False).rename("calls").reset_index())
        by_fw["kind"] = by_fw["framework"].map(kind_of)
        by_fw["pct_of_calls"] = (100 * by_fw["calls"] / by_fw["calls"].sum()).round(1)
        save(by_fw, "calls_by_framework.csv")
        print("LLM calls by framework, grouped "
              f"({len(raw)} import names -> {len(by_fw)} frameworks):")
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
            # (1) application prevalence: what % of analyzed apps import each eval
            #     framework, plus an ANY row for apps using any evaluator at all.
            prev = (ev_calls.groupby("framework")["repo"].nunique()
                    .sort_values(ascending=False).rename("apps_using").reset_index())
            prev["pct_of_analyzed_apps"] = (100 * prev["apps_using"] / n_analyzed).round(1) \
                if n_analyzed else 0
            any_apps = ev_calls["repo"].nunique()
            prev = pd.concat([prev, pd.DataFrame([{
                "framework": "ANY", "apps_using": any_apps,
                "pct_of_analyzed_apps": round(100 * any_apps / n_analyzed, 1) if n_analyzed else 0,
            }])], ignore_index=True)
            save(prev, "eval_app_prevalence.csv")
            print("\napplication prevalence (of "
                  f"{n_analyzed} analyzed apps, % importing each evaluator):")
            print(prev.to_string(index=False))

            # (2) call composition: of all eval calls, what % is each eval framework.
            by_fw = (ev_calls.groupby("framework")["call_id"].nunique()
                     .sort_values(ascending=False).rename("eval_calls").reset_index())
            total = by_fw["eval_calls"].sum()
            by_fw["pct_of_eval_calls"] = (100 * by_fw["eval_calls"] / total).round(1) \
                if total else 0
            save(by_fw, "eval_calls_by_framework.csv")
            print("\neval-call composition (share of all eval calls):")
            print(by_fw.head(12).to_string(index=False))

    print(f"\nCross-tab CSVs written to {OUT}")


if __name__ == "__main__":
    main()
