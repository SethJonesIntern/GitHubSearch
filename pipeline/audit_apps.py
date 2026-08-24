"""The audit sheet — one row per analyzed application, one column per quality question.

Starts from `applications_analyzed.csv` (the Stage-5 run set) and fills in what the
existing artifacts already know: how much each repo produced (calls / invokers / ND
tests), whether its pyan call graph was usable, and whether Joern sliced it. The
judgement columns (is it really a framework? why are there no invokers?) start EMPTY
and get filled in by later passes, one question at a time.

Re-running never clobbers a filled column: any value already in the sheet — hand
entered or written by a later pass — is carried forward.

    py -3.14 -m pipeline.audit_apps          # build / refresh the sheet
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "Wrapper"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import paths  # noqa: E402
from FrameworkDict import FRAMEWORK_CALLS  # noqa: E402
from pipeline.eval_calls import EVAL_CALLS  # noqa: E402

AUDIT_CSV = paths.ARTIFACTS_DIR / "application_audit.csv"
INPUT_CSV = paths.ARTIFACTS_DIR / "applications_slim.csv"

csv.field_size_limit(10 ** 9)

# Columns this script computes. Everything else in FIELDS is left for a later pass.
COMPUTED = {
    "full_name", "clone_slug", "matched_frameworks", "stars", "size_kb",
    "contributors", "pushed_at", "description", "real_ai_app", "analyzed_scope",
    "processed", "clone_failed", "llm_calls", "llm_calls_raw", "fp_dropped_calls",
    "invokers_direct", "invokers_transitive", "nd_tests", "nd_tests_direct",
    "nd_tests_transitive", "eval_calls", "total_functions", "pct_functions_direct",
    "cg_source", "graph_usable", "graph_excluded_files", "graph_coverage_pct",
    "slice_status", "slice_error", "zero_invoker",
}

FIELDS = [
    # ── identity ──────────────────────────────────────────────────────────────
    "full_name", "clone_slug", "matched_frameworks", "stars", "size_kb",
    "contributors", "pushed_at", "description",
    # ── which scope bucket the 2026-08-11 filter put it in ────────────────────
    "real_ai_app", "analyzed_scope",
    # ── did Stage 5 run at all ────────────────────────────────────────────────
    "processed", "clone_failed",
    # ── what it produced ──────────────────────────────────────────────────────
    "llm_calls", "llm_calls_raw", "fp_dropped_calls",
    "invokers_direct", "invokers_transitive",
    "nd_tests", "nd_tests_direct", "nd_tests_transitive", "eval_calls",
    "total_functions", "pct_functions_direct",
    # ── pyan call-graph health ────────────────────────────────────────────────
    "cg_source", "graph_usable", "graph_excluded_files", "graph_coverage_pct",
    # ── Joern slicing ─────────────────────────────────────────────────────────
    "slice_status", "slice_error",
    # ── what the code actually imports (audit_imports.py), not the search token ─
    "frameworks_imported",
    # ── to fill in later: is this a framework, not an application? ────────────
    "framework_suspect", "framework_evidence",
    # ── to fill in later: if it has no invokers, why? ─────────────────────────
    "zero_invoker", "zero_invoker_reason", "imports_matched_fw",
    "py_files", "ipynb_files", "test_files", "http_llm_files", "cli_llm_files",
    # ── the verdict ───────────────────────────────────────────────────────────
    "in_scope", "notes",
]

# ── population cuts ───────────────────────────────────────────────────────────
# Repos removed from the study, each with the evidence that settled it. They stay in
# the sheet carrying in_scope=0 so every denominator remains recoverable — this
# project filters, it never deletes (SPRINT_HANDOFF §10). Log each cut in
# EXCLUSIONS.md with the same reason string.
#
# `haystack` collides three ways: deepset Haystack (the LLM framework we study),
# django-haystack (Django search indexing) and Project Haystack (building-automation
# data). Only the first is in scope. All 20 repos below matched the token `haystack`
# and nothing else, produced 0 invokers and 0 LLM calls, and were confirmed by
# reading their imports — `from haystack import indexes` / `haystack.forms` is
# django-haystack, not `haystack.components`.
_DJANGO_HAYSTACK = """
tendenci/tendenci GrandComicsDatabase/gcd-django elixir-luxembourg/daisy
liangliangyy/DjangoBlog ulgens/drf-haystack City-of-Helsinki/linkedevents
wechange-eg/cosinnus-core bcgov/aries-vcr ae-utbm/sith Nabla-NTNU/nablaweb
Ajapaik/ajapaik-web macports/macports-webapp Parisson/Telemeta
widelands/widelands-website reviewboard/reviewboard batiste/django-page-cms
""".split()
_PROJECT_HAYSTACK = """
ChristianTremblay/pyhaystack dhrumilp15/haystackfs BrickSchema/py-brickschema
rick-jennings/phable
""".split()

# `clai` is a collision token, NOT pydantic-ai's CLI: as a GitHub search term it matched
# the substring inside `claim` / `claiming`. The repos below matched it (and no real
# framework), make 0 LLM calls, import no LLM package, and show no out-of-process
# evidence — they are SDKs, web apps and research code.
_CLAI_COLLISION = """
binance/binance-connector-python StellarCN/py-stellar-base huaweicloud/huaweicloud-sdk-python-v3
dajiaji/python-cwt vertex-protocol/vertex-python-sdk ProzorroUKR/openprocurement.api
rapidpro/rapidpro canonical/maas ClairMeta/ClairMeta zincware/ZnDraw
rai-opensource/spot_wrapper perrette/papers SiliconEinstein/Gaia
MontrealAI/agialpha-first-real-loop jpietek/PenguinBurner
WenJinfeng/FaaSLight SkBlaz/py3plex
""".split()

CUT = {r: "not an LLM app: django-haystack (Django search), not deepset Haystack"
       for r in _DJANGO_HAYSTACK}
CUT.update({r: "not an LLM app: Project Haystack (building automation), not deepset Haystack"
            for r in _PROJECT_HAYSTACK})
CUT.update({r: "not an LLM app: `clai` collision token (claim/claiming); 0 calls, no LLM import"
            for r in _CLAI_COLLISION})

# in_scope=uncovered — a real LLM application on a framework outside the top-20. Not a
# cut: it leaves the analyzed statistics but stays in the coverage denominator, which
# exists to measure exactly this tail. Both below import litellm, which is a real LLM
# SDK we chose not to write patterns for, so their 0 calls is expected, not a finding.
UNCOVERED = {
    "infinitywings/rka": "real LLM app on litellm — outside the top-20, not measured",
    "HolobiomicsLab/Perspicacite-AI": "real LLM app on litellm — outside the top-20, not measured",
}

VALID_PATTERNS = {(fw, p) for fw, pats in FRAMEWORK_CALLS.items() for p in pats}
VALID_EVAL_PATTERNS = {(fw, p) for fw, pats in EVAL_CALLS.items() for p in pats}
_REASON_RE = re.compile(r"matches '([^']+)' from (\S+)\s*$")


def reason_valid(reason: str) -> bool:
    """A direct invoker row only counts if the pattern behind it still exists in
    FrameworkDict — same rule Applications/analyze.py applies, so the sheet and the
    report can't disagree."""
    m = _REASON_RE.search(reason or "")
    return bool(m) and (m.group(2), m.group(1)) in VALID_PATTERNS


def count_calls(path: Path, valid=None):
    """-> (kept call_ids, raw call_ids, fp-dropped rows) per repo. Kept = the
    analyze.py view: false-positive tiers dropped, since-removed patterns dropped.

    `valid` is the (framework, pattern) set a row must still match. It defaults to the
    LLM dict; the EVAL pass MUST pass its own, because eval frameworks are not keys in
    FRAMEWORK_CALLS and testing them against it silently drops every eval call."""
    if valid is None:
        valid = VALID_PATTERNS
    kept, raw, fp = defaultdict(set), defaultdict(set), Counter()
    if not path.exists() or path.stat().st_size == 0:
        return kept, raw, fp
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            repo = r["repo"]
            raw[repo].add(r["call_id"])
            if (r.get("fp_tier") or "").strip():
                fp[repo] += 1
            elif (r.get("framework"), r.get("pattern")) in valid:
                kept[repo].add(r["call_id"])
    return kept, raw, fp


def count_invokers(path: Path):
    """-> {repo: {kind: {qnames}}} over distinct function names."""
    seen = defaultdict(lambda: defaultdict(set))
    if not path.exists() or path.stat().st_size == 0:
        return seen
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            kind = r.get("kind") or "direct"
            if kind == "direct" and not reason_valid(r.get("reason", "")):
                continue
            seen[r["repo"]][kind].add(r["qname"])
    return seen


def slice_status_map(progress: dict):
    """Joern outcome per repo. The recorded failure is a full Joern log — far too
    long for a cell — so it is reduced to its signature."""
    sigs = ["OutOfMemoryError", "StackOverflowError", "TimeoutError", "timed out",
            "WinError", "No such file", "joern-parse failed", "MemoryError"]
    out = {}
    for e in progress.get("slice_failed", []):
        err = (e.get("error") or "").replace("\n", " ")
        out[e["repo"]] = ("failed", next((s for s in sigs if s in err), err[:80]))
    return out


def main() -> None:
    progress = json.loads(paths.BATCH_PROGRESS_JSON.read_text(encoding="utf-8")) \
        if paths.BATCH_PROGRESS_JSON.exists() else {}
    processed = set(progress.get("processed", []))
    clone_failed = {f["repo"] for f in progress.get("failed", [])
                    if f.get("stage") == "clone"}
    sliced = slice_status_map(progress)
    have_slice = {e.name for e in os.scandir(paths.SLICES_DIR)
                  if e.is_dir()} if paths.SLICES_DIR.is_dir() else set()

    print("# reading Stage-5 artifacts...")
    calls, calls_raw, calls_fp = count_calls(paths.LLM_CALLS_CSV)
    ev_calls, _, _ = count_calls(paths.EVAL_CALLS_CSV, VALID_EVAL_PATTERNS)
    invokers = count_invokers(paths.LLM_INVOKERS_CSV)
    tests = count_invokers(paths.LLM_TESTS_CSV)

    health = {}
    if paths.CALL_GRAPH_HEALTH_CSV.exists():
        with paths.CALL_GRAPH_HEALTH_CSV.open(newline="", encoding="utf-8") as fh:
            health = {r["repo"]: r for r in csv.DictReader(fh)}

    prior = {}
    if AUDIT_CSV.exists():                     # carry forward everything already filled
        with AUDIT_CSV.open(newline="", encoding="utf-8") as fh:
            prior = {r["full_name"]: r for r in csv.DictReader(fh)}

    with INPUT_CSV.open(newline="", encoding="utf-8") as fh:
        apps = list(csv.DictReader(fh))

    rows = []
    for app in apps:
        repo = app["full_name"]
        h = health.get(repo, {})
        direct = len(invokers.get(repo, {}).get("direct", ()))
        trans = len(invokers.get(repo, {}).get("transitive", ()))
        t_direct = len(tests.get(repo, {}).get("direct", ()))
        t_trans = len(tests.get(repo, {}).get("transitive", ()))
        total_fns = int(h.get("total_functions") or 0)
        slug = repo.replace("/", "_")

        status, error = sliced.get(repo, ("", ""))
        if not status:
            status = ("ok" if slug in have_slice else
                      "absent" if repo in processed else "not_processed")

        row = {f: "" for f in FIELDS}
        row.update({
            "full_name": repo,
            "clone_slug": slug,
            "matched_frameworks": app.get("matched_frameworks", ""),
            "stars": app.get("stars", ""),
            "size_kb": app.get("size_kb", ""),
            "contributors": app.get("contributors", ""),
            "pushed_at": (app.get("pushed_at") or "")[:10],
            "description": (app.get("description") or "").replace("\n", " ")[:200],
            "real_ai_app": app.get("real_ai_app", ""),
            "analyzed_scope": app.get("analyzed", ""),
            "processed": "1" if repo in processed else "0",
            "clone_failed": "1" if repo in clone_failed else "0",
            "llm_calls": len(calls.get(repo, ())),
            "llm_calls_raw": len(calls_raw.get(repo, ())),
            "fp_dropped_calls": calls_fp.get(repo, 0),
            "invokers_direct": direct,
            "invokers_transitive": trans,
            "nd_tests": t_direct + t_trans,
            "nd_tests_direct": t_direct,
            "nd_tests_transitive": t_trans,
            "eval_calls": len(ev_calls.get(repo, ())),
            "total_functions": total_fns,
            "pct_functions_direct": round(100 * direct / total_fns, 2) if total_fns else "",
            "cg_source": h.get("cg_source", ""),
            "graph_usable": h.get("graph_usable", ""),
            "graph_excluded_files": h.get("excluded_files", ""),
            "graph_coverage_pct": h.get("graph_coverage_pct", ""),
            "slice_status": status,
            "slice_error": error,
            "zero_invoker": "1" if (direct + trans) == 0 else "0",
        })
        # A later pass (or a human) owns every non-computed column: keep what's there.
        for f in FIELDS:
            if f not in COMPUTED and prior.get(repo, {}).get(f):
                row[f] = prior[repo][f]
        if repo in CUT:
            row["in_scope"], row["notes"] = "0", CUT[repo]
        elif repo in UNCOVERED:
            row["in_scope"], row["notes"] = "uncovered", UNCOVERED[repo]
        rows.append(row)

    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    ran = [r for r in rows if r["processed"] == "1"]
    print(f"# wrote {AUDIT_CSV}  ({n} rows, {len(FIELDS)} columns)")
    print(f"\n  processed by Stage 5      {len(ran):>5} / {n}")
    print(f"  clone failed              {sum(r['clone_failed'] == '1' for r in rows):>5}")
    print(f"  zero invokers             {sum(r['zero_invoker'] == '1' for r in ran):>5}"
          f"  ({100 * sum(r['zero_invoker'] == '1' for r in ran) / max(len(ran), 1):.1f}% of processed)")
    print("\n  cg_source:", dict(Counter(r["cg_source"] or "-" for r in ran).most_common()))
    print("  slice_status:", dict(Counter(r["slice_status"] for r in ran).most_common()))
    for label, value, effect in [
            ("cut (in_scope=0)", "0", "out of every count, coverage denominator included"),
            ("uncovered", "uncovered", "out of analyzed stats, still in the coverage denominator")]:
        sub = [r for r in rows if r["in_scope"] == value]
        print(f"\n  {label:<24}{len(sub):>5}  {effect}")
        for reason in sorted({r["notes"] for r in sub}):
            print(f"    {sum(1 for r in sub if r['notes'] == reason):>4}  {reason}")
    empty = [f for f in FIELDS if f not in COMPUTED]
    filled = {f: sum(1 for r in rows if r[f]) for f in empty}
    print("\n  columns still to fill:", ", ".join(f"{f} ({filled[f]} filled)" for f in empty))


if __name__ == "__main__":
    main()
