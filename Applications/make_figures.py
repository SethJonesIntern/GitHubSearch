"""The figure set: one question, one figure, one table twin.

Three questions, asked of the observed population — applications in which a
FRAMEWORK LLM call site was detected. Raw provider-SDK calls (openai/anthropic)
are excluded from both the population test and every count; they remain in the
artifacts, they are simply not the subject.

  F1  Do these applications use an LLM evaluation framework?
  F2  Do LLM calls set the parameters that control non-determinism?
  F3  Which frameworks do these applications actually call?

Run: py -3.14 Applications/make_figures.py [F1 F2 ...]
Writes Fn_<slug>.png + Fn_<slug>.csv + FIGURES.md into artifacts/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "Applications", _ROOT / "Wrapper"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

import analyze as AZ  # noqa: E402
import figstyle as S  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.eval_calls import EVAL_CALLS  # noqa: E402

OUT = paths.ARTIFACTS_DIR / "figures"
RAW_SDKS = {"openai", "anthropic"}


# ── the observed population ──────────────────────────────────────────────────

def population():
    """(repos, framework-call rows). A repo is observed iff we detected at least
    one framework call site in it — the study's scoping boundary."""
    prog = json.loads(paths.BATCH_PROGRESS_JSON.read_text(encoding="utf-8")) \
        if paths.BATCH_PROGRESS_JSON.exists() else {}
    analysed = set(prog.get("processed", [])) - set(AZ.EXCLUDED)
    calls = AZ.drop_fp(AZ.drop_removed_patterns_calls(
        pd.read_csv(paths.ARTIFACTS_DIR / "llm_calls_all.csv")))
    calls = calls[calls["repo"].isin(analysed) & ~calls["framework"].isin(RAW_SDKS)]
    return set(calls["repo"]), calls


POP, CALLS = population()
N = len(POP)


def _label_h(ax, bars, values, fmt):
    """Direct value labels at the end of horizontal bars — every figure ships
    them, so nothing depends on reading a length against a gridline."""
    span = ax.get_xlim()[1]
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + span * 0.015, bar.get_y() + bar.get_height() / 2,
                fmt(v), va="center", ha="left", fontsize=9, color=S.SECOND)


# ── F1 ───────────────────────────────────────────────────────────────────────

def f1():
    ev = pd.read_csv(paths.ARTIFACTS_DIR / "eval_calls_all.csv")
    ev = ev[ev["repo"].isin(POP)]
    per = {k: 0 for k in EVAL_CALLS}
    for fw, sub in ev.groupby("framework"):
        per[fw] = sub["repo"].nunique()
    users = ev["repo"].nunique()

    rows = [("no evaluation framework", N - users)]
    rows += sorted(per.items(), key=lambda kv: -kv[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [S.CONTEXT] + [S.EMPHASIS] * (len(rows) - 1)

    fig, ax = S.figure("Do these applications use an LLM evaluation framework?",
                       plot_height_in=0.42 * len(rows) + 0.35)
    y = range(len(rows))
    bars = ax.barh(list(y), vals, color=colors, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.5, color=S.INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_xlabel("applications (n = %d)" % N, fontsize=9, color=S.SECOND, labelpad=8)
    S.frame(ax, axis="x")
    # No round_ends here: the bars span 0-606, and the rounding radius notches
    # the short ones into a pinched shape. Square reads cleanly at every length.
    _label_h(ax, bars, vals, lambda v: "%s  (%.1f%%)" % (format(v, ","), 100 * v / N))

    S.table(OUT, "F1_eval_adoption", ["group", "applications", "pct_of_population"],
            [[l, v, round(100 * v / N, 1)] for l, v in rows])
    # Every claim in the caption is computed, never typed: it must move with the data.
    top2 = sorted(per, key=lambda k: -per[k])[:2]
    by_top2 = ev[ev["framework"].isin(top2)]["repo"].nunique()
    unused = [k for k in EVAL_CALLS if per[k] == 0]
    answer = ("%d of %d (%.1f%%) call one. %s and %s account for %d of them%s."
              % (users, N, 100 * users / N, top2[0], top2[1], by_top2,
                 "; %s appear%s in none" % (" and ".join(unused),
                                            "" if len(unused) > 1 else "s")
                 if unused else ""))
    return ("F1_eval_adoption", fig,
            ("Do these applications use an LLM evaluation framework?", answer))


# ── F2 ───────────────────────────────────────────────────────────────────────

def f2():
    meta = AZ.drop_fp(AZ.drop_removed_patterns_calls(
        pd.read_csv(paths.ARTIFACTS_DIR / "call_metadata_all.csv")))
    meta = meta[meta["repo"].isin(POP) & ~meta["framework"].isin(RAW_SDKS)]
    ids = meta["call_id"].drop_duplicates()
    total = len(ids)

    rows = []
    for k in AZ.KNOBS:
        sub = meta[meta["arg_keyword"] == k]
        c = sub["call_id"].nunique()
        lit = sub[AZ.truthy(sub["arg_is_literal"])]["call_id"].nunique() if c else 0
        rows.append((k, c, lit, c - lit))
    rows.sort(key=lambda r: -r[1])

    # how many knobs a single call sets — the table twin carries this
    per_call = (meta[meta["arg_keyword"].isin(set(AZ.KNOBS))]
                .groupby("call_id")["arg_keyword"].nunique()
                .reindex(ids).fillna(0).astype(int))
    none = int((per_call == 0).sum())

    fig, ax = S.figure("Do LLM calls set the parameters that control non-determinism?",
                       plot_height_in=0.40 * len(rows) + 0.35, legend_rows=1)
    y = range(len(rows))
    lit = [r[2] for r in rows]
    var = [r[3] for r in rows]
    ax.barh(list(y), lit, color=S.EMPHASIS, height=0.6)
    b2 = ax.barh(list(y), var, left=lit, color=S.CONTEXT, height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=S.INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(r[1] for r in rows), 1) * 1.30)
    ax.set_xlabel("call sites setting the parameter (of %s)" % format(total, ","),
                  fontsize=9, color=S.SECOND, labelpad=8)
    S.frame(ax, axis="x")
    # Stacked: anchor each label to the END of the whole bar, not to the segment
    # b2 happens to be — get_width() on a stacked segment is the segment only.
    span = ax.get_xlim()[1]
    for bar, r in zip(b2, rows):
        ax.text(r[1] + span * 0.015, bar.get_y() + bar.get_height() / 2,
                "%s  (%.1f%%)" % (format(r[1], ","), 100 * r[1] / total),
                va="center", ha="left", fontsize=9, color=S.SECOND)
    S.legend(fig, ["literal value", "passed as a variable"], [S.EMPHASIS, S.CONTEXT])

    S.table(OUT, "F2_determinism_parameters",
            ["parameter", "call_sites", "pct_of_calls", "literal", "variable"],
            [[k, c, round(100 * c / total, 2), l, v] for k, c, l, v in rows]
            + [["(calls setting no parameter at all)", none,
                round(100 * none / total, 1), "", ""]])
    # Computed, never typed (the old text called `model` the most often set; it was 2nd).
    top = rows[0]
    seed = next(r for r in rows if r[0] == "seed")
    answer = ("%s of %s call sites (%.1f%%) set none of the eight. The most often set is "
              "`%s` (%s call sites, %s of them passed as a variable); `seed` is set at %s."
              % (format(none, ","), format(total, ","), 100 * none / total,
                 top[0], format(top[1], ","), format(top[3], ","),
                 "no call site" if seed[1] == 0 else "%d call sites" % seed[1]))
    return ("F2_determinism_parameters", fig,
            ("Do LLM calls set the parameters that control non-determinism?", answer))


# ── F3 ───────────────────────────────────────────────────────────────────────

def f3():
    # Every framework we detected a call to, not a top-N slice: the tail is small
    # and cutting it hides how concentrated the corpus is.
    g = CALLS.assign(fw=CALLS["framework"].map(AZ.group_of))
    t = (g.groupby("fw").agg(apps=("repo", "nunique"),
                             sites=("call_id", "nunique"))
         .sort_values("apps", ascending=False))
    head = t

    fig, ax = S.figure("Which frameworks do these applications actually call?",
                       plot_height_in=0.40 * len(head) + 0.35)
    y = range(len(head))
    bars = ax.barh(list(y), head["apps"], color=S.EMPHASIS, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(head.index, fontsize=9.5, color=S.INK)
    ax.invert_yaxis()
    ax.set_xlim(0, head["apps"].max() * 1.34)
    ax.set_xlabel("applications calling the framework (n = %d)" % N,
                  fontsize=9, color=S.SECOND, labelpad=8)
    S.frame(ax, axis="x")
    span = ax.get_xlim()[1]
    for bar, (_, r) in zip(bars, head.iterrows()):
        ax.text(bar.get_width() + span * 0.015, bar.get_y() + bar.get_height() / 2,
                "%d  (%.0f%%)   ·   %s call sites"
                % (r.apps, 100 * r.apps / N, format(r.sites, ",")),
                va="center", ha="left", fontsize=8.6, color=S.SECOND)

    S.table(OUT, "F3_frameworks_called",
            ["framework", "applications", "pct_of_population", "call_sites",
             "call_sites_per_app"],
            [[i, r.apps, round(100 * r.apps / N, 1), r.sites,
              round(r.sites / r.apps, 1)] for i, r in t.iterrows()])
    answer = ("%s leads at %.0f%% of %d applications. Measured from detected call sites, "
              "not from the search token, so a framework is credited only where it is "
              "actually invoked."
              % (head.index[0], 100 * head["apps"].iloc[0] / N, N))
    return ("F3_frameworks_called", fig,
            ("Which frameworks do these applications actually call?", answer))


# ── F4 ───────────────────────────────────────────────────────────────────────

def f4():
    """Is the code that calls the LLM ever what a test is written against?

    Asked of EVERY test pyMethods2Test identified, not only of the tests that call
    an LLM themselves — starting from those would drop the commonest shape, a test
    that calls repo code which in turn calls the model. A test counts when its focal
    method (pyMethods2Test's answer to "what is under test") contains a framework
    call site; every such method is a direct invoker in llm_invokers_all.csv.
    """
    fc = pd.read_csv(paths.ARTIFACTS_DIR / "pym2t_focal_calls.csv")
    t = pd.read_csv(paths.ARTIFACTS_DIR / "pym2t_focal_tests.csv", low_memory=False)
    lines = fc.groupby("file")["call_line"].apply(list).to_dict()

    resolved = t[t["focal_method"].notna()].copy()
    resolved["llm"] = [any(s <= ln <= e for ln in lines.get(f, ()))
                       for f, s, e in zip(resolved["focal_file"], resolved["focal_line"],
                                          resolved["focal_line_end"])]
    hit = resolved[resolved["llm"]]

    total = len(fc)
    in_file = fc["tests_focal_method"].gt(0) | fc["tests_focal_file"].gt(0)
    in_method = fc["tests_focal_method"].gt(0)
    # (label, value, is_a_share_of_total). The tests bar counts tests, not call sites,
    # so it carries no percentage — 558/23,990 would divide tests by call sites.
    rows = [
        ("LLM call sites", total, True),
        ("in a tested file", int(in_file.sum()), True),
        ("tests on LLM methods", len(hit), False),
        ("call sites tested", int(in_method.sum()), True),
    ]
    colors = [S.CONTEXT, S.CONTEXT, S.CONTEXT, S.EMPHASIS]

    fig, ax = S.figure("How many of the LLM calls are actually tested?",
                       plot_height_in=0.46 * len(rows) + 0.35)
    y = range(len(rows))
    vals = [r[1] for r in rows]
    bars = ax.barh(list(y), vals, color=colors, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=S.INK)
    ax.invert_yaxis()
    # Linear, deliberately: a log axis would read 296 as a peer of 23,990.
    ax.set_xlim(0, max(vals) * 1.26)
    ax.set_xlabel("framework LLM call sites (n = %s in %d applications)"
                  % (format(total, ","), fc["repo"].nunique()),
                  fontsize=9, color=S.SECOND, labelpad=8)
    S.frame(ax, axis="x")
    span = ax.get_xlim()[1]
    for bar, (_, v, share) in zip(bars, rows):
        text = ("%s  (%.1f%%)" % (format(v, ","), 100 * v / total) if share
                else "%s tests" % format(v, ","))
        ax.text(bar.get_width() + span * 0.015,
                bar.get_y() + bar.get_height() / 2, text,
                va="center", ha="left", fontsize=9, color=S.SECOND)

    apps = fc.loc[in_method, "repo"].nunique()
    S.table(OUT, "F4_focal_mapping", ["step", "unit", "count", "pct_of_call_sites",
                                      "applications"],
            [["framework LLM call sites", "call sites", total, 100.0,
              fc["repo"].nunique()],
             ["in a file some test targets", "call sites", rows[1][1],
              round(100 * rows[1][1] / total, 1), fc.loc[in_file, "repo"].nunique()],
             ["tests whose focal method calls an LLM", "tests", len(hit), "",
              hit["repo"].nunique()],
             ["distinct focal methods those tests target", "methods",
              len(hit.drop_duplicates(["focal_file", "focal_line"])), "", apps],
             ["call sites inside those methods", "call sites", rows[3][1],
              round(100 * rows[3][1] / total, 2), apps]])
    answer = ("%d of %s framework call sites (%.1f%%, in %d of %d applications) sit "
              "inside a method some test is written against — %d tests over %d distinct "
              "methods. The other %.1f%% of call sites are targeted by no test."
              % (rows[3][1], format(total, ","), 100 * rows[3][1] / total, apps,
                 fc["repo"].nunique(), len(hit),
                 len(hit.drop_duplicates(["focal_file", "focal_line"])),
                 100 * (total - rows[3][1]) / total))
    return ("F4_focal_mapping", fig,
            ("How many of the LLM calls are actually tested?", answer))


# ── F5 ───────────────────────────────────────────────────────────────────────

def f5():
    """Two mapping rates side by side, each a share of its own population:
    how often pyMethods2Test can say what a test targets at all, and how often an
    LLM-calling method is what some test targets. The second is the coverage
    question; the first is the ceiling the tool puts on it."""
    t = pd.read_csv(paths.ARTIFACTS_DIR / "pym2t_focal_tests.csv", low_memory=False,
                    usecols=["focal_file", "focal_method", "focal_line"])
    fm = t[t["focal_method"].notna()]
    focal = {(f, int(ln)) for f, ln in zip(fm["focal_file"], fm["focal_line"])}

    inv = pd.read_csv(paths.LLM_INVOKERS_CSV,
                      usecols=["repo", "file", "line", "reason", "kind"])
    inv = AZ.drop_removed_patterns_invokers(inv)
    fw = inv["reason"].str.extract(r"from (\S+)\s*$")[0]
    d = inv[inv["kind"].eq("direct") & inv["repo"].isin(POP) & ~fw.isin(RAW_SDKS)] \
        .drop_duplicates(["file", "line"])
    covered = sum((f, int(ln)) in focal for f, ln in zip(d["file"], d["line"]))

    rows = [("tests with a focal method", len(fm), len(t), "tests"),
            ("LLM-calling methods a test targets", covered, len(d), "methods")]
    fig, ax = S.figure("Does focal mapping reach the code that calls the LLM?",
                       plot_height_in=0.52 * len(rows) + 0.45)
    y = range(len(rows))
    pcts = [100 * n / tot for _, n, tot, _ in rows]
    bars = ax.barh(list(y), pcts, color=[S.CONTEXT, S.EMPHASIS], height=0.55)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=S.INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(pcts) * 1.55)
    ax.set_xlabel("share of its own population (%)", fontsize=9, color=S.SECOND,
                  labelpad=8)
    S.frame(ax, axis="x")
    span = ax.get_xlim()[1]
    for bar, (_, n, tot, unit), p in zip(bars, rows, pcts):
        ax.text(bar.get_width() + span * 0.015, bar.get_y() + bar.get_height() / 2,
                "%.1f%%   ·   %s of %s %s" % (p, format(n, ","), format(tot, ","), unit),
                va="center", ha="left", fontsize=9, color=S.SECOND)

    S.table(OUT, "F5_mapping_rates", ["population", "unit", "mapped", "total", "pct"],
            [[lab, unit, n, tot, round(100 * n / tot, 2)] for lab, n, tot, unit in rows])
    answer = ("pyMethods2Test resolves a focal method for %.1f%% of tests (%s of %s). "
              "Of the %s methods that make a framework LLM call, %.2f%% (%d, in %d "
              "applications) are what a test targets — an order of magnitude below the "
              "tool's own hit rate, so the gap is coverage, not only tooling."
              % (pcts[0], format(len(fm), ","), format(len(t), ","), format(len(d), ","),
                 pcts[1], covered,
                 d.assign(c=[(f, int(ln)) in focal for f, ln in zip(d["file"], d["line"])])
                 .query("c")["repo"].nunique()))
    return ("F5_mapping_rates", fig,
            ("Does focal mapping reach the code that calls the LLM?", answer))


FIGURES = {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5}


def main(argv):
    wanted = [a.upper() for a in argv[1:]] or list(FIGURES)
    bad = [w for w in wanted if w not in FIGURES]
    if bad:
        sys.exit("unknown figure(s): %s. Choose from %s."
                 % (", ".join(bad), ", ".join(FIGURES)))
    entries = []
    for key in wanted:
        stem, fig, (question, answer) = FIGURES[key]()
        png = S.save(fig, OUT, stem)
        entries.append((key, stem, question, answer))
        print("%s  %s  +  %s.csv" % (key, png.name, stem))

    if len(wanted) == len(FIGURES):
        lines = ["# Findings — one question, one figure, one table",
                 "",
                 "Observed population: **%d applications** in which a framework LLM "
                 "call site was detected. Raw provider-SDK calls are excluded." % N,
                 "",
                 "Regenerate with `py -3.14 Applications/make_figures.py`.", ""]
        for key, stem, question, answer in entries:
            lines += ["## %s. %s" % (key, question), "",
                      "**%s**" % answer, "",
                      "![%s](figures/%s.png)" % (key, stem), "",
                      "Table: [`%s.csv`](figures/%s.csv)" % (stem, stem), ""]
        (OUT / "FIGURES.md").write_text("\n".join(lines), encoding="utf-8")
        print("\nWrote %s" % (OUT / "FIGURES.md"))


if __name__ == "__main__":
    main(sys.argv)
