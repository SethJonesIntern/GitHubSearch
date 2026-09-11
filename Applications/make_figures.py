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
    answer = ("%d of %d (%.1f%%) call one. Two tools account for all but one of them; "
              "giskard and phoenix appear in none."
              % (users, N, 100 * users / N))
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
    answer = ("%s of %s call sites (%.1f%%) set none of the eight. `seed` is set at no "
              "call site in the corpus, and `model` — the most often set — is usually "
              "passed as a variable rather than pinned."
              % (format(none, ","), format(total, ","), 100 * none / total))
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


FIGURES = {"F1": f1, "F2": f2, "F3": f3}


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
