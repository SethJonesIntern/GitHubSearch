"""The population waterfall: every drop, in pipeline order, with no double-counting.

Answers the question "we started at 6,446 candidates and we are quoting 677 — where
did the rest go, and which of those losses were decisions rather than accidents?"

A repo leaves at the FIRST step that removes it and is never counted again, so the
`dropped` column sums exactly to the difference between the first and last rows.
That is the whole point: the same repo is a `clai` collision AND a clone failure AND
an uncovered-tail app, and counting it three times is how the ladder stops adding up.

Ordering is pipeline order — discovery, then run mechanics, then scope decisions —
so a row's position tells you whether it is a bug (fix it), machinery (wait for it),
or a recorded judgement (read EXCLUSIONS.md).

Every number is derived from the artifacts; nothing here is a constant copied out of
a document, so the table stays true as the sheet is edited and the run progresses.

Run: py -3.14 -m pipeline.waterfall
     py -3.14 -m pipeline.waterfall --md      # markdown, for pasting into a doc
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "Applications", _ROOT / "Wrapper"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

import analyze  # noqa: E402
import slim_applications as sa  # noqa: E402
from pipeline import paths  # noqa: E402

csv.field_size_limit(10 ** 9)
A = paths.ARTIFACTS_DIR

# in_scope=0 is one column carrying many different decisions. The reason lives in
# `notes`, written by hand when the cut was made, so the buckets key off the prefix
# each pass used. Anything unmatched surfaces as "other" rather than being hidden.
NOTE_BUCKETS = [
    ("framework, not an application", "framework / library, not an application (EXCL 11, 12a)"),
    ("framework source copy", "framework source copy (EXCL 11)"),
    ("platform/builder", "platform or builder, not an application (EXCL 12b)"),
    ("not an LLM app: `clai`", "`clai` collision token (EXCL 9)"),
    ("not an LLM app: django-haystack", "django-haystack collision (EXCL 10)"),
    ("not an LLM app: Project Haystack", "Project Haystack collision (EXCL 10)"),
]

DONE, BUG, TEMP, DESIGN, QUEUE = "done", "BUG - fixable", "temporary", "by design", "see queue"


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["full_name"]: r for r in csv.DictReader(fh)}


def build():
    slim = _rows(A / "applications_slim.csv")
    audit = _rows(A / "application_audit.csv")
    prog = json.loads(paths.BATCH_PROGRESS_JSON.read_text(encoding="utf-8"))
    processed = set(prog.get("processed", []))
    # `failed` holds {repo, stage} records, not bare names
    failed = {f["repo"] if isinstance(f, dict) else f for f in prog.get("failed", [])}
    rerun_csv = _ROOT / "pipeline" / "next_runs" / "rerun.csv"
    rerun = set(_rows(rerun_csv)) if rerun_csv.exists() else set()

    calls = analyze.drop_fp(analyze.drop_removed_patterns_calls(
        pd.read_csv(A / "llm_calls_all.csv")))
    have_calls = set(calls["repo"].unique())

    candidates = set(_rows(A / "applications.csv"))
    self_repos = set(sa.SELF_REPOS) & candidates

    def note_bucket(repo):
        t = (audit.get(repo, {}).get("notes") or "").strip()
        # §13 cuts are keyed on the section marker rather than a prefix: they share one
        # criterion but each row names its own token, so no common prefix exists.
        if "EXCLUSIONS §13" in t or "EXCLUSIONS §13" in t:
            return "collision token, denominator-side cut (EXCL 13)"
        if "EXCLUSIONS §14" in t or "collision-token cut (EXCLUSIONS §14)" in t:
            return "collision token, Queue A review (EXCL 14)"
        for prefix, label in NOTE_BUCKETS:
            if t.startswith(prefix):
                return label
        return "other in_scope=0 (no matching notes prefix)"

    live = set(slim)
    ladder = OrderedDict()

    def drop(label, members, status):
        nonlocal live
        hit = live & set(members)
        live -= hit
        if hit or status in (BUG, TEMP):
            ladder[label] = (len(hit), len(live), status)

    # run mechanics — not scope decisions
    drop("clone failed (Windows illegal-filename bug, HANDOFF 2)", failed, BUG)
    lifted = rerun - processed
    drop("lifted out for the in-flight re-run", lifted, TEMP)
    drop("never processed (other)", set(slim) - processed, "check")

    # hand exclusions carried in analyze.py itself
    drop("code quality: >=10 unparseable files (EXCL 1)", analyze.QUALITY_EXCLUDED, DONE)
    drop("not an LLM app: sunnypilot agno collision (EXCL 1)", analyze.NOT_LLM_APP, DONE)

    # in_scope=0, split by the reason recorded in `notes`
    cut0 = {r for r in slim if (audit.get(r, {}).get("in_scope") or "").strip() == "0"}
    for label, _ in Counter(note_bucket(r) for r in (live & cut0)).most_common():
        drop(label, {r for r in live & cut0 if note_bucket(r) == label}, DONE)

    # the deliberately-unmeasured tail: out of the analyzed stats, IN the coverage
    # denominator (pipeline/cuts.py) — measuring it is the point of the 90.1% figure
    drop("uncovered: real LLM app on a framework outside the top-20 (EXCL 9)",
         {r for r in slim if (audit.get(r, {}).get("in_scope") or "").strip() == "uncovered"},
         DESIGN)

    analyzed = set(live)
    drop("in scope, but zero LLM call sites found", analyzed - have_calls, QUEUE)

    return dict(
        candidates=len(candidates), self_repos=len(self_repos), population=len(slim),
        ladder=ladder, analyzed=analyzed, with_calls=live, audit=audit, slim=slim,
        lifted=lifted, rerun=rerun, have_calls=have_calls,
    )


def queues(d):
    """The 161 that are in scope with no call site, split by what we know about them."""
    zero = d["analyzed"] - d["with_calls"]
    audit, slim = d["audit"], d["slim"]
    no_import = {r for r in zero
                 if not (audit.get(r, {}).get("frameworks_imported") or "").strip()}
    flagged = {r for r in no_import
               if (slim.get(r, {}).get("real_ai_app") or "").strip() == "0"}
    return OrderedDict([
        ("A. no framework import, token already flagged junk", flagged),
        ("B. no framework import, nothing flagged", no_import - flagged),
        ("C. imports a framework, no call site found", zero - no_import),
    ])


def report(d, md=False):
    n_an, n_calls = len(d["analyzed"]), len(d["with_calls"])
    bar, W = ("|", 0) if md else ("", 64)

    def row(label, dropped, remaining, status="", indent=True):
        pre = "  " if indent else ""
        if md:
            print(f"| {pre}{label} | {dropped or ''} | {remaining or ''} | {status} |")
        else:
            print(f"{pre + label:<{W}}{str(dropped):>9}{str(remaining):>11}  {status}")

    if md:
        print("| step | dropped | remaining | status |")
        print("|---|---:|---:|---|")
    else:
        print("=" * 104)
        print("POPULATION WATERFALL - a repo leaves at the FIRST step that drops it")
        print("=" * 104)
        print(f"{'step':<{W}}{'dropped':>9}{'remaining':>11}  status")
        print("-" * 104)

    row("GitHub code-search candidates", "", d["candidates"], indent=False)
    row("no trustworthy framework name in the match",
        d["candidates"] - d["population"] - d["self_repos"],
        d["population"] + d["self_repos"], DONE)
    row("framework / eval self-repositories (EXCL 7)", d["self_repos"], d["population"], DONE)
    row("**POPULATION**" if md else "POPULATION", "", d["population"], indent=False)
    for label, (dropped, remaining, status) in d["ladder"].items():
        if label.startswith("in scope, but zero"):
            row("**ANALYZED** (denominator for every prevalence figure)" if md
                else "ANALYZED (denominator for every prevalence figure)", "", n_an, indent=False)
        row(label, dropped, remaining, status)
    row("**REPOS WITH AT LEAST ONE LLM CALL SITE**" if md
        else "REPOS WITH AT LEAST ONE LLM CALL SITE", "", n_calls, indent=False)
    if not md:
        print("=" * 104)

    in_scope_lifted = {r for r in d["lifted"]
                       if (d["audit"].get(r, {}).get("in_scope") or "").strip()
                       not in ("0", "uncovered")}
    print(f"\nRE-RUN: {len(d['rerun'])} repos queued, {len(d['lifted'])} currently lifted out of "
          f".batch_progress,\n  {len(in_scope_lifted)} of them in scope. The denominator returns to "
          f"~{n_an + len(in_scope_lifted)} when the run finishes.\n  This is machinery, not a scope "
          f"decision - do not read it as a result.")

    print(f"\n{'PENDING REVIEW - still counted, no decision recorded':=^104}" if not md
          else "\n**Pending review**\n")
    if md:
        print("| queue | repos | denominator | prevalence becomes |")
        print("|---|---:|---:|---:|")
    # Cumulative: each queue's denominator assumes every queue above it was also cut,
    # because that is the order the work would actually be done in. Reporting each in
    # isolation would let three "+3 point" rows read as if they stacked to +9.
    cut_so_far = 0
    for label, members in queues(d).items():
        cut_so_far += len(members)
        den = n_an - cut_so_far
        line = (f"| {label} | {len(members)} | {den} | {100 * n_calls / den:.1f}% |" if md
                else f"  {label:<58}{len(members):>6}{den:>7}   {100 * n_calls / den:>5.1f}%")
        print(line)
    if not md:
        print("\n  Queues are CUMULATIVE - each denominator assumes the ones above it were cut.")
        print("  C is not a cut queue: dropping every zero-call repo makes the figure 100% by")
        print("  construction. Most of C should stay; it is a detector-scope question.")


def main(argv):
    report(build(), md="--md" in argv)


if __name__ == "__main__":
    main(sys.argv)
