"""Eight research questions, each answered by one figure and one table.

    Q1  Do the mined applications actually call an LLM, and do their tests?
    Q2  Which frameworks do these applications really import?
    Q3  Where do the LLM calls go?
    Q4  Are LLM calls pinned to deterministic settings?
    Q5  How many determinism parameters does a single call set?
    Q6  How many tests reach a live LLM?
    Q7  Can the static analysis be trusted?
    Q8  Do these projects use an LLM evaluation framework?

Every question emits three things into `pipeline/artifacts/figures/`:

    Qn_<slug>.png   the chart — question as title, answer as subtitle, n= as note
    Qn_<slug>.csv   the table twin, the same numbers in full precision
    FIGURES.md      the index: question, one-sentence answer, both filenames

Numbers come through `analyze.py`'s own loaders and filters (false positives
dropped, since-removed patterns dropped, cut/uncovered repos dropped) and share
`analyze.py`'s definition of the analyzed set, so no figure can disagree with
the report or with another figure.

Two deliberate departures from `analyze.py`'s printed report:

  * **Non-deterministic tests are reported DIRECT ONLY.** The transitive count is
    ~92% of the total, inflates with call-graph size, and is absent entirely for
    repos with no usable graph — so it measures the graph as much as the code.
    Q7 is the figure that justifies this. See CLAUDE.md, "Known-bad things".
  * **Q2 counts imports found in the cloned source** (`frameworks_imported` in the
    audit sheet), not `matched_frameworks`. The latter is the GitHub *code-search*
    token that surfaced the repo; 139 repos never import what they matched. This
    also sidesteps the `clai -> pydantic_ai` mapping that CLAUDE.md flags as wrong.

Run: py -3.14 Applications/make_figures.py
     py -3.14 Applications/make_figures.py Q4 Q6      # just those two
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "Applications", _ROOT / "Wrapper"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

import analyze  # noqa: E402
import figstyle as F  # noqa: E402
import keep_frequency as kf  # noqa: E402
from FrameworkDict import IN_SCOPE_FRAMEWORKS  # noqa: E402
from pipeline import paths  # noqa: E402

csv.field_size_limit(10 ** 9)

OUT = paths.ARTIFACTS_DIR / "figures"
AUDIT = paths.ARTIFACTS_DIR / "application_audit.csv"

# Which artifact each question reads. Used only for the staleness check below.
BIG_INVOKER_CSVS = ["llm_tests_all.csv", "llm_invokers_all.csv"]


# ── shared data, loaded once ─────────────────────────────────────────────────
class Data:
    """Lazy accessors over the Stage-5 artifacts, each already run through
    analyze.py's filters. Loading is deferred because two of these files are
    ~90 MB and ~190 MB and most questions need neither."""

    def __init__(self):
        self._cache = {}

    def _once(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def analyzed(self) -> set[str]:
        """The denominator every prevalence figure divides by — exactly the set
        analyze.py calls `analyzed`: Stage-5 processed, minus cut repos, minus
        the deliberately-unmeasured tail."""
        def build():
            prog = json.loads(paths.BATCH_PROGRESS_JSON.read_text(encoding="utf-8")) \
                if paths.BATCH_PROGRESS_JSON.exists() else {}
            return set(prog.get("processed", [])) - set(analyze.EXCLUDED)
        return self._once("analyzed", build)

    def _frame(self, name, cleaner, **read_kw):
        """Every frame is restricted to `self.analyzed` — the SAME set every
        prevalence figure divides by.

        Dropping only `analyze.EXCLUDED` is not enough. `prepare_rerun` lifts a
        repo's rows out of the artifacts and removes it from `.batch_progress`
        while it is re-queued, but the artifacts are rewritten at different
        times (two of them are Git-LFS files written by whichever machine ran
        the batch). A repo can therefore be absent from the denominator while
        its stale rows still sit in some numerator — which is exactly how Q1
        once reported more repos with an LLM-calling function than repos with
        an LLM call site. Intersecting here makes numerator <= denominator by
        construction, for every figure.
        """
        def build():
            path = paths.ARTIFACTS_DIR / name
            if not path.exists() or path.stat().st_size == 0:
                return pd.DataFrame()
            df = cleaner(pd.read_csv(path, **read_kw))
            return df[df["repo"].isin(self.analyzed)] if "repo" in df else df
        return self._once(name + str(read_kw.get("usecols", "")), build)

    @property
    def calls(self):
        return self._frame("llm_calls_all.csv",
                           lambda d: analyze.drop_fp(analyze.drop_removed_patterns_calls(d)))

    @property
    def meta(self):
        return self._frame("call_metadata_all.csv",
                           lambda d: analyze.drop_fp(analyze.drop_removed_patterns_calls(d)))

    @property
    def tests(self):
        return self._frame("llm_tests_all.csv", analyze.drop_removed_patterns_invokers)

    # llm_invokers_all.csv is deliberately NOT read here. "Repos with a function
    # that calls an LLM" and "repos with an LLM call site" are the same fact —
    # a call site is inside a function — and measured 0 repos apart on the call
    # side. Showing both invited the reader to treat filter differences between
    # the two files as a finding. Q1 asks the question once.

    @property
    def ev_calls(self):
        return self._frame("eval_calls_all.csv", lambda d: d)

    @property
    def health(self):
        return self._frame("call_graph_health.csv", lambda d: d)

    @property
    def audit(self) -> list[dict]:
        def build():
            with AUDIT.open(newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
        return self._once("audit", build)


D = Data()


def direct(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the analyzer saw itself, as opposed to rows reached through the pyan
    call graph. Everything reported as a test count in this figure set is direct."""
    return df[df["kind"] == "direct"] if not df.empty else df


def pct(n, total) -> float:
    return 100 * n / total if total else 0.0


def top_n(pairs, n, other_label="other"):
    """Keep the n largest, fold the rest into one honest 'other' row. Never
    silently truncate — the tail keeps its total and its member count."""
    pairs = sorted(pairs, key=lambda kv: -kv[1])
    if len(pairs) <= n:
        return pairs, 0, 0
    head, tail = pairs[:n], pairs[n:]
    return head + [(f"{other_label} ({len(tail)})", sum(v for _, v in tail))], \
        len(tail), sum(v for _, v in tail)


# ── the questions ────────────────────────────────────────────────────────────
def q1():
    stem = "Q1_llm_usage"
    n = len(D.analyzed)
    calls, tests, ev = D.calls, D.tests, D.ev_calls

    measures = [
        ("Calls an LLM somewhere in its source", calls["repo"].nunique() if not calls.empty else 0),
        ("Has a TEST that calls an LLM", direct(tests)["repo"].nunique() if not tests.empty else 0),
        ("Uses an LLM evaluation framework", ev["repo"].nunique() if not ev.empty else 0),
    ]

    # Why bar 1 is not ~100%. Every repo here is in scope, so a missing call site
    # is a limit of DETECTION, not evidence the app has no LLM in it. Split the
    # shortfall so the figure cannot be read as "19% of these are not LLM apps".
    zero = D.analyzed - set(calls["repo"].unique())
    audit = {r["full_name"]: r for r in D.audit}
    no_fw = [r for r in zero
             if not (audit.get(r, {}).get("frameworks_imported") or "").strip()]
    def _evidence(col):
        return sum(1 for r in no_fw
                   if (audit.get(r, {}).get(col) or "0") not in ("", "0"))
    http_n, cli_n = _evidence("http_llm_files"), _evidence("cli_llm_files")
    labels = [m for m, _ in measures]
    vals = [pct(v, n) for _, v in measures]
    # the middle bar is the study's subject; the other two are context for it
    colors = [F.EMPHASIS if i == 1 else F.CONTEXT for i in range(len(measures))]

    fig, ax = F.figure(
        "Q1.  Do these applications actually call an LLM — and do their tests?",
        f"Yes to the first, mostly no to the second. {vals[0]:.0f}% contain an LLM call\n"
        f"site, but only {vals[1]:.0f}% have a test that reaches one.",
        f"n = {n:,} analyzed applications; test counts are DIRECT invocations only (see Q6, Q7).\n"
        f"Bar 1 is a floor on DETECTION, not on LLM use: of the {len(zero)} with no call site, "
        f"{len(no_fw)} import\nno framework this pipeline matches ({http_n} call a provider over raw HTTP, "
        f"{cli_n} shell out to an\nLLM CLI, and raw Gemini is not yet detected); the other "
        f"{len(zero) - len(no_fw)} import one that yielded no match.",
        plot_height_in=0.52 * len(measures) + 0.35)
    y = range(len(measures))
    bars = ax.barh(list(y), vals, height=0.58, color=colors, linewidth=0)
    ax.set_yticks(list(y), labels, fontsize=9.5, color=F.SECOND)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25%", "50%", "75%", "100%"])
    for i, ((_, raw), v) in enumerate(zip(measures, vals)):
        ax.text(v + 1.4, i, f"{v:.0f}%   ({raw:,} repos)", va="center", ha="left",
                fontsize=9, color=F.INK if i == 1 else F.MUTED,
                fontweight="bold" if i == 1 else "normal")
    F.frame(ax, axis="x")
    F.round_ends(ax, bars, horizontal=True)
    F.table(OUT, stem, ["measure", "repos", "pct_of_analyzed"],
            [[m, v, f"{pct(v, n):.1f}"] for m, v in measures]
            + [["analyzed applications (denominator)", n, "100.0"]])
    return stem, fig, (
        "Q1. Do these applications actually call an LLM, and do their tests?",
        f"{vals[0]:.0f}% contain an LLM call site; only {vals[1]:.0f}% have a test that "
        f"reaches one (n = {n:,}).")


def q2():
    stem = "Q2_frameworks_imported"
    live = [r for r in D.audit if r["full_name"] in D.analyzed]
    groups, members = Counter(), {}
    for r in live:
        names = {x.strip() for x in (r.get("frameworks_imported") or "").split(",") if x.strip()}
        for g in {kf.category(x) for x in names}:
            groups[g] += 1
        for x in names:
            members.setdefault(kf.category(x), set()).add(x)
    measured = {g: any(m in IN_SCOPE_FRAMEWORKS for m in ms) for g, ms in members.items()}

    rows = groups.most_common(15)
    n = len(live)
    labels = [g for g, _ in rows]
    vals = [pct(v, n) for _, v in rows]
    colors = [F.SERIES[0] if measured.get(g) else F.SERIES[1] for g in labels]
    n_unmeasured = sum(1 for g in groups if not measured.get(g))

    fig, ax = F.figure(
        "Q2.  Which frameworks do these applications really import?",
        f"LangChain dominates at {vals[0]:.0f}% of applications, and the raw OpenAI SDK is\n"
        f"second — most of this population talks to a provider directly as well.",
        f"n = {n:,} analyzed applications; top 15 of {len(groups)} import families. Counted from\n"
        f"imports in the cloned source, NOT the GitHub search token. {n_unmeasured} families fall\n"
        f"outside the analysed set. Raw Gemini (google.generativeai) is not yet detected.",
        plot_height_in=0.34 * len(rows) + 0.55, legend_rows=1)
    y = range(len(rows))
    bars = ax.barh(list(y), vals, height=0.62, color=colors, linewidth=0)
    ax.set_yticks(list(y), labels, fontsize=9.5, color=F.SECOND)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.24)
    for i, ((_, raw), v) in enumerate(zip(rows, vals)):
        ax.text(v + max(vals) * 0.015, i, f"{v:.0f}%  ({raw})", va="center", ha="left",
                fontsize=8.5, color=F.MUTED)
    F.frame(ax, axis="x")
    F.legend(fig, ["analysed by this study", "outside the analysed set"], F.SERIES[:2])
    F.round_ends(ax, bars, horizontal=True)
    F.table(OUT, stem, ["framework_family", "applications", "pct_of_analyzed", "analysed_by_study",
                        "member_import_names"],
            [[g, v, f"{pct(v, n):.1f}", int(bool(measured.get(g))),
              " ".join(sorted(members.get(g, [])))] for g, v in groups.most_common()])
    return stem, fig, (
        "Q2. Which frameworks do these applications really import?",
        f"{labels[0]} leads at {vals[0]:.0f}% of {n:,} applications, with the raw OpenAI SDK "
        f"close behind; measured from imports in the cloned source.")


def q3():
    stem = "Q3_calls_by_framework"
    calls = D.calls
    grouped = (calls.assign(framework=calls["framework"].map(analyze.group_of))
               .groupby("framework")["call_id"].nunique())
    total = int(calls["call_id"].nunique())
    rows, n_tail, _ = top_n(list(grouped.items()), 12, "other frameworks")
    labels = [k for k, _ in rows]
    vals = [pct(v, total) for _, v in rows]
    colors = [F.SERIES[1] if analyze.kind_of(k) == "raw SDK" else F.SERIES[0] for k in labels]
    sdk = sum(v for k, v in grouped.items() if analyze.kind_of(k) == "raw SDK")

    fig, ax = F.figure(
        "Q3.  Where do the LLM calls go?",
        f"Through frameworks, mostly — but {pct(sdk, total):.0f}% of call sites bypass\n"
        f"them and hit a provider SDK directly.",
        f"n = {total:,} distinct LLM call sites across {calls['repo'].nunique():,} repositories.\n"
        f"Package families are grouped (langchain_openai -> langchain); tail folded into 'other'.",
        plot_height_in=0.34 * len(rows) + 0.55, legend_rows=1)
    y = range(len(rows))
    bars = ax.barh(list(y), vals, height=0.62, color=colors, linewidth=0)
    ax.set_yticks(list(y), labels, fontsize=9.5, color=F.SECOND)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.22)
    for i, ((_, raw), v) in enumerate(zip(rows, vals)):
        ax.text(v + max(vals) * 0.015, i, f"{v:.0f}%  ({raw:,})", va="center", ha="left",
                fontsize=8.5, color=F.MUTED)
    F.frame(ax, axis="x")
    F.legend(fig, ["agent / LLM framework", "raw provider SDK"], F.SERIES[:2])
    F.round_ends(ax, bars, horizontal=True)
    F.table(OUT, stem, ["framework", "calls", "pct_of_calls", "kind"],
            [[k, v, f"{pct(v, total):.1f}", analyze.kind_of(k)]
             for k, v in grouped.sort_values(ascending=False).items()])
    return stem, fig, (
        "Q3. Where do the LLM calls go?",
        f"{pct(sdk, total):.0f}% of {total:,} call sites bypass every framework and call a "
        f"provider SDK directly.")


def q4():
    stem = "Q4_determinism_knobs"
    meta = D.meta
    total = int(meta["call_id"].nunique())
    rows = []
    for k in analyze.KNOBS:
        sub = meta[meta["arg_keyword"] == k]
        cw = sub["call_id"].nunique()
        lit = sub[analyze.truthy(sub["arg_is_literal"])]["call_id"].nunique()
        rows.append((k, cw, lit, cw - lit))
    rows.sort(key=lambda r: -r[1])

    labels = [r[0] for r in rows]
    lit_pct = [pct(r[2], total) for r in rows]
    var_pct = [pct(r[3], total) for r in rows]
    span = max(l + v for l, v in zip(lit_pct, var_pct))
    seed = next(r for r in rows if r[0] == "seed")
    temp = next(r for r in rows if r[0] == "temperature")

    fig, ax = F.figure(
        "Q4.  Are LLM calls pinned to deterministic settings?",
        f"Almost never. temperature is set on {pct(temp[1], total):.1f}% of call sites and\n"
        f"seed — the only parameter that actually pins output — on {pct(seed[1], total):.1f}%.",
        f"n = {total:,} distinct LLM call sites. A bar is the share of call sites passing that\n"
        f"keyword at all; the split is whether the value is a literal or comes from a variable.",
        plot_height_in=0.40 * len(rows) + 0.55, legend_rows=1)
    y = list(range(len(rows)))
    gap = span * 0.005                     # ~2px surface gap between the segments
    # Painted back-to-front rather than stacked left-to-right: the full-length bar
    # carries the rounded data end, then the inner segment is over-painted flat on
    # top of it. Stacking the segments instead would round the JOIN as well, and
    # two curved edges meeting there pinch the bar into an arrowhead.
    totals = [l + gap + v for l, v in zip(lit_pct, var_pct)]
    b_total = ax.barh(y, totals, height=0.58, color=F.SERIES[1], linewidth=0, zorder=2)
    ax.barh(y, [l + gap for l in lit_pct], height=0.58, color=F.SURFACE,
            linewidth=0, zorder=3)
    ax.barh(y, lit_pct, height=0.58, color=F.SERIES[0], linewidth=0, zorder=4)
    ax.set_yticks(y, labels, fontsize=9.5, color=F.SECOND)
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.28)
    ax.set_xticks([0, 5, 10, 15], ["0", "5%", "10%", "15%"])
    for i, r in enumerate(rows):
        ax.text(lit_pct[i] + var_pct[i] + span * 0.02, i,
                f"{pct(r[1], total):.1f}%  ({r[1]:,})", va="center", ha="left",
                fontsize=8.5, color=F.INK if r[0] == "seed" else F.MUTED,
                fontweight="bold" if r[0] == "seed" else "normal")
    F.frame(ax, axis="x")
    F.legend(fig, ["hard-coded value", "passed via a variable"], F.SERIES[:2])
    F.round_ends(ax, list(b_total), horizontal=True)
    F.table(OUT, stem, ["kwarg", "calls_setting_it", "pct_of_calls", "literal_value",
                        "variable_value"],
            [[r[0], r[1], f"{pct(r[1], total):.1f}", r[2], r[3]] for r in rows]
            + [["(total distinct calls)", total, "100.0", "", ""]])
    return stem, fig, (
        "Q4. Are LLM calls pinned to deterministic settings?",
        f"Almost never — temperature on {pct(temp[1], total):.1f}% of {total:,} call sites, "
        f"seed on {pct(seed[1], total):.1f}%.")


def q5():
    stem = "Q5_knobs_per_call"
    meta = D.meta
    all_calls = meta["call_id"].drop_duplicates()
    per = (meta[meta["arg_keyword"].isin(set(analyze.KNOBS))]
           .groupby("call_id")["arg_keyword"].nunique()
           .reindex(all_calls).fillna(0).astype(int))
    dist = Counter(per)
    total = len(all_calls)
    ks = sorted(dist)
    vals = [pct(dist[k], total) for k in ks]

    fig, ax = F.figure(
        "Q5.  How many determinism parameters does a single call set?",
        f"None, in {vals[0]:.0f}% of cases. The call is left entirely on provider defaults.",
        f"n = {total:,} distinct LLM call sites across {meta['repo'].nunique():,} repositories.\n"
        f"Parameters counted: {', '.join(analyze.KNOBS)}.",
        plot_height_in=2.9)
    fig.subplots_adjust(left=0.11)
    bars = ax.bar(ks, vals, width=0.68, linewidth=0,
                  color=[F.EMPHASIS if k == 0 else F.CONTEXT for k in ks])
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_xlim(-0.6, max(ks) + 0.6)
    ax.set_xticks(ks)
    ax.set_yticks([0, 20, 40, 60, 80], ["0", "20%", "40%", "60%", "80%"])
    for k, v in zip(ks, vals):
        ax.text(k, v + max(vals) * 0.022, f"{v:.1f}%" if v >= 0.05 else f"{v:.2f}%",
                ha="center", va="bottom", fontsize=9,
                color=F.INK if k == 0 else F.MUTED,
                fontweight="bold" if k == 0 else "normal")
    ax.set_xlabel("number of determinism parameters set in the call", fontsize=9.5,
                  color=F.SECOND, labelpad=8)
    F.frame(ax, axis="y")
    F.round_ends(ax, bars)
    F.table(OUT, stem, ["knobs_set", "calls", "pct_of_calls"],
            [[k, dist[k], f"{pct(dist[k], total):.2f}"] for k in ks])
    return stem, fig, (
        "Q5. How many determinism parameters does a single call set?",
        f"{vals[0]:.0f}% of {total:,} call sites set none at all.")


def q6():
    stem = "Q6_nd_tests"
    tests = direct(D.tests)
    uniq = tests.drop_duplicates(["repo", "qname"])
    n_tests, n_repos = len(uniq), uniq["repo"].nunique()
    n_analyzed = len(D.analyzed)

    fw = tests.assign(framework=tests["reason"].map(analyze.fw_from_reason))
    fw = fw.assign(framework=fw["framework"].map(analyze.group_of))
    grouped = (fw.drop_duplicates(["repo", "qname", "framework"])
               .groupby("framework").size().sort_values(ascending=False))
    rows, _, _ = top_n(list(grouped.items()), 12, "other frameworks")
    labels = [k for k, _ in rows]
    vals = [v for _, v in rows]

    fig, ax = F.figure(
        "Q6.  How many tests reach a live LLM?",
        f"{n_tests:,} distinct tests across {n_repos} repositories "
        f"({pct(n_repos, n_analyzed):.0f}% of the corpus)\ninvoke a real model, so their outcome "
        f"is not reproducible.",
        f"DIRECT invocations only — a test whose own body calls the model. Tests that reach one\n"
        f"through the call graph are excluded: that count inflates with graph size and is absent\n"
        f"for repos with no usable graph (Q7). A test can count for more than one framework.",
        plot_height_in=0.34 * len(rows) + 0.55)
    y = range(len(rows))
    bars = ax.barh(list(y), vals, height=0.62, linewidth=0,
                   color=[F.EMPHASIS if i == 0 else F.CONTEXT for i in range(len(rows))])
    ax.set_yticks(list(y), labels, fontsize=9.5, color=F.SECOND)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.20)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, i, f"{v:,}", va="center", ha="left",
                fontsize=8.5, color=F.MUTED)
    F.frame(ax, axis="x")
    F.round_ends(ax, bars, horizontal=True)
    F.table(OUT, stem, ["framework", "direct_nd_tests"],
            [[k, v] for k, v in grouped.items()]
            + [["(distinct tests, deduplicated across frameworks)", n_tests],
               ["(repositories containing them)", n_repos]])
    return stem, fig, (
        "Q6. How many tests reach a live LLM?",
        f"{n_tests:,} distinct tests in {n_repos} repositories "
        f"({pct(n_repos, n_analyzed):.0f}% of the corpus), counting direct invocations only.")


def q7():
    stem = "Q7_graph_health"
    health = D.health
    health = health[health["repo"].isin(D.analyzed)]
    n = len(health)
    order = [("pyan", "complete call graph", F.SERIES[0]),
             ("pyan_resilient", "graph built after dropping unparseable files", F.SERIES[1]),
             ("none", "no usable call graph", F.SERIES[2])]
    counts = health["cg_source"].value_counts().to_dict()
    vals = [pct(counts.get(k, 0), n) for k, _, _ in order]
    usable = pct(counts.get("pyan", 0) + counts.get("pyan_resilient", 0), n)
    none_n = counts.get("none", 0)

    fig, ax = F.figure(
        "Q7.  Can the static analysis be trusted?",
        f"For direct findings, yes — those come from the source itself. For anything\n"
        f"reached through the call graph, only for the {usable:.0f}% of repositories that have one.",
        f"n = {n:,} analyzed applications. {none_n} have no usable call graph, so their transitive\n"
        f"count is structurally zero rather than genuinely zero — which is why every test number\n"
        f"in this set (Q1, Q6) reports direct invocations only.",
        plot_height_in=1.05, legend_rows=3)
    left = 0.0
    gap = 0.35
    bars = []
    for (key, _, color), v in zip(order, vals):
        bars.append(ax.barh([0], [v], left=[left], height=0.42, color=color, linewidth=0)[0])
        # inside the segment where it fits, above it where it doesn't — slot 3
        # (aqua) is below 3:1 on this surface, so its label must be visible ink
        if v > 8:
            ax.text(left + v / 2, 0, f"{v:.0f}%", va="center", ha="center",
                    fontsize=10.5, color="#ffffff", fontweight="bold")
        else:
            ax.text(left + v / 2, 0.26, f"{v:.0f}%", va="bottom", ha="center",
                    fontsize=9.5, color=F.INK, fontweight="bold")
        left += v + gap
    ax.set_xlim(0, 100 + gap * 2)
    ax.set_ylim(-0.62, 0.62)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25%", "50%", "75%", "100%"])
    F.frame(ax, axis="x")
    ax.xaxis.grid(False)
    ax.spines["left"].set_visible(False)   # no categories on y — a baseline here
    #                                        reads as a stray rule, not an axis
    F.legend(fig, [lbl for _, lbl, _ in order], [c for _, _, c in order], ncol=1)
    F.round_ends(ax, bars, horizontal=True)
    F.table(OUT, stem, ["cg_source", "meaning", "repos", "pct_of_analyzed"],
            [[k, lbl, counts.get(k, 0), f"{pct(counts.get(k, 0), n):.1f}"]
             for k, lbl, _ in order]
            + [["(analyzed applications)", "denominator", n, "100.0"]])
    return stem, fig, (
        "Q7. Can the static analysis be trusted?",
        f"{usable:.0f}% of {n:,} applications have a usable call graph; the {none_n} without one "
        f"are why test counts are reported direct-only.")


def q8():
    stem = "Q8_eval_adoption"
    ev = D.ev_calls
    n = len(D.analyzed)
    per = ev.groupby("framework")["repo"].nunique().sort_values(ascending=False)
    any_apps = ev["repo"].nunique()
    rows = list(per.items()) + [("ANY evaluator", any_apps)]
    labels = [k for k, _ in rows]
    vals = [pct(v, n) for _, v in rows]
    colors = [F.CONTEXT] * (len(rows) - 1) + [F.EMPHASIS]

    fig, ax = F.figure(
        "Q8.  Do these projects use an LLM evaluation framework?",
        f"Hardly any do. {pct(any_apps, n):.1f}% of applications call an evaluator at all —\n"
        f"the tooling built to cope with non-deterministic output is barely adopted.",
        f"n = {n:,} analyzed applications. Evaluators detected: "
        f"{', '.join(per.index)}.",
        plot_height_in=0.42 * len(rows) + 0.35)
    y = range(len(rows))
    bars = ax.barh(list(y), vals, height=0.58, color=colors, linewidth=0)
    ax.set_yticks(list(y), labels, fontsize=9.5, color=F.SECOND)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.30)
    for i, ((_, raw), v) in enumerate(zip(rows, vals)):
        ax.text(v + max(vals) * 0.02, i, f"{v:.1f}%  ({raw} repos)", va="center", ha="left",
                fontsize=8.5, color=F.INK if i == len(rows) - 1 else F.MUTED,
                fontweight="bold" if i == len(rows) - 1 else "normal")
    F.frame(ax, axis="x")
    F.round_ends(ax, bars, horizontal=True)
    F.table(OUT, stem, ["evaluator", "apps_using", "pct_of_analyzed"],
            [[k, v, f"{pct(v, n):.1f}"] for k, v in rows])
    return stem, fig, (
        "Q8. Do these projects use an LLM evaluation framework?",
        f"Only {pct(any_apps, n):.1f}% of {n:,} applications call one.")


QUESTIONS = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4,
             "Q5": q5, "Q6": q6, "Q7": q7, "Q8": q8}


# ── driver ───────────────────────────────────────────────────────────────────
def staleness_warning() -> str | None:
    """The two invoker/test CSVs are Git-LFS tracked and are written by the batch
    driver on whichever machine ran it. If they predate the call artifacts, the
    test figures (Q1, Q6) describe an older corpus than the call figures (Q3-Q5)
    and the set is internally inconsistent. Say so rather than emit it silently."""
    ref = paths.ARTIFACTS_DIR / "call_metadata_all.csv"
    if not ref.exists():
        return None
    stale = [n for n in BIG_INVOKER_CSVS
             if (paths.ARTIFACTS_DIR / n).exists()
             and (paths.ARTIFACTS_DIR / n).stat().st_mtime < ref.stat().st_mtime - 3600]
    if not stale:
        return None
    return ("PROVISIONAL: " + ", ".join(stale) + " predate call_metadata_all.csv by "
            ">1h, so the test-based figures (Q1, Q6) and the call-based figures "
            "(Q3, Q4, Q5) may describe different corpus states. Re-run once the "
            "batch driver has finished and the artifacts are in sync.")


def main(argv):
    wanted = [a.upper() for a in argv[1:]] or list(QUESTIONS)
    unknown = [w for w in wanted if w not in QUESTIONS]
    if unknown:
        sys.exit(f"unknown question(s): {', '.join(unknown)}. "
                 f"Choose from {', '.join(QUESTIONS)}.")

    warn = staleness_warning()
    if warn:
        print(f"\n!! {warn}\n")

    index = []
    for key in wanted:
        stem, fig, (question, answer) = QUESTIONS[key]()
        png = F.save(fig, OUT, stem)
        print(f"{key}  {png.name:<32} + {stem}.csv")
        print(f"    {answer}")
        index.append((key, question, answer, stem))

    if len(wanted) == len(QUESTIONS):
        lines = ["# Findings — one question, one figure, one table", ""]
        if warn:
            lines += [f"> **{warn}**", ""]
        lines += ["Regenerate with `py -3.14 Applications/make_figures.py`.", ""]
        for key, question, answer, stem in index:
            lines += [f"## {question}", "",
                      f"**{answer}**", "",
                      f"![{key}](figures/{stem}.png)", "",
                      f"Table: [`{stem}.csv`](figures/{stem}.csv)", ""]
        (OUT / "FIGURES.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"\nwrote {OUT / 'FIGURES.md'}")


if __name__ == "__main__":
    main(sys.argv)
