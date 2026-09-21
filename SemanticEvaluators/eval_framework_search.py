"""Eval-framework search - GitHub issue #3, "Find List Of Missing Eval Frameworks".

**Discovery only.** This script produces a *candidate* list of LLM-evaluation
frameworks and a funnel showing what each filter cost. It adopts nothing, changes
no statistic, and is independent of every other artifact in this repo: it does not
read `EVAL_CALLS`, does not touch `application_audit.csv`, and re-runs no stage.
Whether anything here becomes an evaluator is a hand decision made afterwards.

Two phases:

  --search-only   Free tier. 8 fixed phrases through GitHub repository search with
                  **no `stars:` and no `language:` qualifier** - both are post-hoc
                  funnel steps, and baking either into the query would make its
                  waterfall bar unmeasurable. Writes eval_framework_candidates.csv
                  (every deduped hit, unfiltered) plus the waterfall.

  --enrich        Paid tier. Full Stage-1 enrichment over the rows marked
                  `confirmed=1`, writing eval_frameworks.csv with the SAME
                  fieldnames as frameworks.csv so the two tables are row-comparable.

The Stage-1 helpers are reused by import from Frameworks/GithubSearch.py, never
copied: github_get, search_repositories, get_repo_details, enrich_repo,
derive_import_names, get_contributor_count, get_test_metrics, apply_filters,
FIELDNAMES, write_csv.

The `confirmed` column is written EMPTY and is never inferred. Auto-populating it
from a keyword or topic rule would be the script answering the question the search
is asking. The final waterfall bar renders as pending until the column is filled.

Run (repo root):
    py -3.14 -m SemanticEvaluators.eval_framework_search --search-only
    py -3.14 -m SemanticEvaluators.eval_framework_search --enrich
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
# Frameworks/ MUST come before Applications/: both ship a module named
# `GithubSearch`, and the Stage-1 helpers reused here are the Frameworks/ ones.
for _p in (_ROOT / "Applications", _ROOT, _ROOT / "Frameworks"):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

import GithubSearch as gs  # noqa: E402
from pipeline import paths  # noqa: E402


# ── search spec (fixed — do not add or reword) ────────────────────────────────
SEARCH_PHRASES = [
    "LLM evaluation framework",
    "LLM eval framework",
    "evaluating LLM applications",
    "LLM testing framework",
    "LLM-as-a-judge",
    "LLM benchmarking framework",
    "LLM observability evaluation",
    "guardrails evaluation LLM",
]

# ── filter thresholds ─────────────────────────────────────────────────────────
# Stars is deliberately the FIRST costed filter: it is the one under interrogation.
# Everything above the enrichment boundary is free; everything below costs ~4 API
# calls per surviving repo.
MIN_STARS = 10
LANGUAGE = "Python"
MIN_CONTRIBUTORS = 2
MIN_TEST_FILES = 1

PER_PAGE = 100
MAX_PAGES = 10
RESULT_CEILING = PER_PAGE * MAX_PAGES   # GitHub's hard 1000-result search cap
# The search endpoint allows 30 requests/min. ~88 requests total, so the limit is
# reached mid-run; gs.github_get sleeps until reset when it sees it, which is what
# makes the run take ~3 min. This pause between phrases just keeps that gentler.
SEARCH_PACE_SEC = 2.0

# Phase A cannot compute `already_seeded` from import names — those come from
# derive_import_names, which needs the repo tree, i.e. an API call, i.e.
# enrichment. So Phase A matches full_name against this hand map; Phase B
# recomputes it properly from import_names. Keys are lowercase: GitHub owner
# casing varies and a case miss would silently break the sanity check.
SEEDED_EVALUATOR_REPOS = {
    "confident-ai/deepeval": "deepeval",
    "explodinggradients/ragas": "ragas",
    "giskard-ai/giskard": "giskard",
    "comet-ml/opik": "opik",
    "arize-ai/phoenix": "phoenix",
}
SEEDED_IMPORT_NAMES = set(SEEDED_EVALUATOR_REPOS.values())

CANDIDATES_CSV = paths.ARTIFACTS_DIR / "eval_framework_candidates.csv"
FIGURES_DIR = paths.ARTIFACTS_DIR / "figures"
FIGURE_STEM = "eval_search_waterfall"

CANDIDATE_FIELDNAMES = [
    "full_name", "description", "stars", "language", "topics",
    "archived", "fork", "pushed_at",
    "matched_queries", "already_seeded", "in_frameworks_csv", "confirmed",
]


# ── phase A: search ───────────────────────────────────────────────────────────

def build_search_queries() -> List[str]:
    """One query per phrase, verbatim.

    No `stars:` and no `language:` qualifier. Both are applied post-hoc in the
    funnel so their cost is visible; a qualifier in the query would drop repos
    silently, before anything could count them.
    """
    return list(SEARCH_PHRASES)


def is_truncated(total_count: Optional[int]) -> bool:
    """True when GitHub reports more matches than it will ever hand back.

    Past 1000 results the endpoint stops paging, so what we retrieved is a
    relevance-ranked prefix, not the phrase's full match set.
    """
    return (total_count or 0) > RESULT_CEILING


def preflight(queries: Sequence[str], pace: float = SEARCH_PACE_SEC) -> List[dict]:
    """Report each phrase's `total_count` before paging anything (8 requests).

    A wildly broad phrase gets cut here, by hand, rather than after 80 requests.
    """
    rows = []
    for query in queries:
        resp = gs.github_get(f"{gs.API_BASE}/search/repositories",
                             params={"q": query, "per_page": 1})
        total = (resp.json().get("total_count") if resp is not None else None) or 0
        rows.append({"query": query, "total_count": total,
                     "truncated": is_truncated(total)})
        time.sleep(pace)
    return rows


def dedupe(results: Sequence[Tuple[str, Sequence[dict]]]) -> Dict[str, dict]:
    """[(query, items)] -> {full_name: {"item": ..., "matched_queries": [...]}}

    First sighting owns the payload; EVERY phrase that surfaced a repo is kept,
    in first-seen order and without repeats, so a repo found by five phrases can
    be told apart from one found by a single narrow phrase.
    """
    out: Dict[str, dict] = {}
    for query, items in results:
        for item in items:
            full_name = item.get("full_name")
            if not full_name:
                continue
            record = out.get(full_name)
            if record is None:
                out[full_name] = {"item": item, "matched_queries": [query]}
            elif query not in record["matched_queries"]:
                record["matched_queries"].append(query)
    return out


# ── the free (pre-enrichment) filters ─────────────────────────────────────────
# Sequential attribution: a repo is charged to the FIRST condition it fails, so
# the drop counts form a clean funnel that sums back to the input.

def _stars(item: dict) -> int:
    return item.get("stargazers_count") or 0


def _excluded_kind(item: dict) -> bool:
    return bool(item.get("fork") or item.get("archived") or item.get("disabled"))


def passes_free_filters(item: dict) -> bool:
    """The three filters that cost nothing: kind, stars, language."""
    return (not _excluded_kind(item)
            and _stars(item) >= MIN_STARS
            and (item.get("language") or "") == LANGUAGE)


def apply_free_filters(items: Sequence[dict]) -> Tuple[List[dict], Dict[str, int]]:
    stats = {
        "deduped": len(items),
        "dropped_fork_archived": 0,
        "dropped_stars": 0,
        "dropped_language": 0,
        "kept_free": 0,
    }
    kept: List[dict] = []
    for item in items:
        if _excluded_kind(item):
            stats["dropped_fork_archived"] += 1
        elif _stars(item) < MIN_STARS:
            stats["dropped_stars"] += 1
        elif (item.get("language") or "") != LANGUAGE:
            stats["dropped_language"] += 1
        else:
            stats["kept_free"] += 1
            kept.append(item)
    return kept, stats


# ── candidate rows ────────────────────────────────────────────────────────────

def already_seeded(full_name: str) -> bool:
    """Phase A: full_name against the hand map, case-insensitively."""
    return (full_name or "").lower() in SEEDED_EVALUATOR_REPOS


def already_seeded_from_import_names(import_names: str) -> bool:
    """Phase B: the proper computation, once derive_import_names has run."""
    names = {n.strip() for n in (import_names or "").split(";") if n.strip()}
    return bool(names & SEEDED_IMPORT_NAMES)


def load_frameworks_full_names(path: Optional[Path] = None) -> set:
    """Lowercased full_names already in frameworks.csv (Stage-1 agentic frameworks).

    Informational only — it populates a column, it never filters anything.
    """
    path = path or paths.FRAMEWORKS_CSV
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r.get("full_name") or "").lower()
                for r in csv.DictReader(fh) if r.get("full_name")}


def candidate_row(item: dict, matched_queries: Sequence[str],
                  framework_names: set) -> dict:
    full_name = item.get("full_name") or ""
    return {
        "full_name": full_name,
        "description": item.get("description") or "",
        "stars": _stars(item),
        "language": item.get("language") or "",
        "topics": ",".join(item.get("topics") or []),
        "archived": bool(item.get("archived")),
        "fork": bool(item.get("fork")),
        "pushed_at": item.get("pushed_at") or "",
        "matched_queries": " | ".join(matched_queries),
        "already_seeded": "1" if already_seeded(full_name) else "0",
        "in_frameworks_csv": "1" if full_name.lower() in framework_names else "0",
        "confirmed": "",   # hand-filled. Never inferred. See the module docstring.
    }


def write_candidates(rows: Sequence[dict], out_path: Path) -> None:
    """Every deduped hit, unfiltered, stars-descending.

    The funnel is post-hoc, so the file keeps the rows the filters reject too —
    that is what makes `archived`, `fork`, `stars` and `language` worth columns,
    and what lets the `already_seeded` sanity check see a seeded evaluator even
    if some filter would have dropped it.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_FIELDNAMES,
                                quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["stars"] or 0, reverse=True))


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def candidate_passes_free_filters(row: dict) -> bool:
    """The free filters re-applied to a candidate CSV row (Phase B selection).

    `disabled` is not among the spec'd candidate columns, so it cannot be
    re-checked here; Phase A charged it at search time from the live payload.
    """
    if _truthy(row.get("archived")) or _truthy(row.get("fork")):
        return False
    try:
        stars = int(row.get("stars") or 0)
    except ValueError:
        stars = 0
    if stars < MIN_STARS:
        return False
    return (row.get("language") or "") == LANGUAGE


def read_candidates(path: Optional[Path] = None) -> List[dict]:
    path = path or CANDIDATES_CSV
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def confirmed_summary(rows: Sequence[dict]) -> Optional[int]:
    """Count of hand-confirmed eval frameworks, or None while the column is blank.

    None means "pending", not "zero" — the difference the last waterfall bar draws.
    """
    if not any((r.get("confirmed") or "").strip() for r in rows):
        return None
    return sum(1 for r in rows
               if (r.get("confirmed") or "").strip() == "1"
               and candidate_passes_free_filters(r))


# ── the waterfall ─────────────────────────────────────────────────────────────

def waterfall_rows(stats: Dict[str, object]) -> List[dict]:
    """The funnel as bars: [{step, survivors, dropped, status}].

    Everything below the enrichment boundary (contributors, test files, confirmed)
    is `pending` until the data exists — Phase B runs after triage, so a Phase-A
    figure genuinely cannot know those numbers. Once a step is pending, every step
    after it is too. Nothing here is fabricated to fill the chart.
    """
    rows: List[dict] = []
    retrieved = int(stats.get("retrieved") or 0)
    rows.append({"step": "search hits retrieved", "survivors": retrieved,
                 "dropped": None, "status": "final"})

    survivors = int(stats.get("deduped") or 0)
    rows.append({"step": "deduped across phrases", "survivors": survivors,
                 "dropped": retrieved - survivors, "status": "final"})

    sequence = [
        ("not fork / archived / disabled", "dropped_fork_archived"),
        (f"stars >= {MIN_STARS}", "dropped_stars"),
        (f"language == {LANGUAGE}", "dropped_language"),
        (f"contributors >= {MIN_CONTRIBUTORS}", "dropped_contributors"),
        (f"test files >= {MIN_TEST_FILES}", "dropped_no_tests"),
    ]
    pending = False
    for label, key in sequence:
        if pending or stats.get(key) is None:
            pending = True
            rows.append({"step": label, "survivors": None, "dropped": None,
                         "status": "pending"})
            continue
        dropped = int(stats[key])
        survivors -= dropped
        rows.append({"step": label, "survivors": survivors, "dropped": dropped,
                     "status": "final"})

    confirmed = stats.get("confirmed")
    if pending or confirmed is None:
        rows.append({"step": "confirmed eval framework (hand triage)",
                     "survivors": None, "dropped": None, "status": "pending"})
    else:
        rows.append({"step": "confirmed eval framework (hand triage)",
                     "survivors": int(confirmed),
                     "dropped": survivors - int(confirmed), "status": "final"})
    return rows


BOUNDARY_AFTER = f"language == {LANGUAGE}"   # the enrichment boundary sits here


def render_waterfall(stats: Dict[str, object], out_dir: Optional[Path] = None) -> Path:
    import figstyle as S   # here, so the module imports without matplotlib present

    out_dir = out_dir or FIGURES_DIR
    rows = waterfall_rows(stats)
    S.table(out_dir, FIGURE_STEM, ["step", "survivors", "dropped", "status"],
            [[r["step"],
              "" if r["survivors"] is None else r["survivors"],
              "" if r["dropped"] is None else r["dropped"],
              r["status"]] for r in rows])

    fig, ax = S.figure(
        "How many eval-framework candidates survive each filter?",
        plot_height_in=0.40 * len(rows) + 0.5)

    y = list(range(len(rows)))
    finals = [r for r in rows if r["status"] == "final"]
    last_final = finals[-1]["step"] if finals else None
    widths, colors = [], []
    for r in rows:
        widths.append(r["survivors"] or 0)
        colors.append(S.EMPHASIS if r["step"] == last_final else S.CONTEXT)

    bars = ax.barh(y, widths, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([r["step"] for r in rows], fontsize=9)
    ax.invert_yaxis()
    span = max(widths + [1])
    ax.set_xlim(0, span * 1.18)
    ax.set_xlabel("repositories", fontsize=9, color=S.SECOND)
    S.frame(ax, axis="x")
    S.round_ends(ax, bars, horizontal=True)

    for idx, r in enumerate(rows):
        if r["status"] == "pending":
            ax.text(span * 0.012, idx, "pending — needs --enrich",
                    va="center", fontsize=8.5, color=S.MUTED, style="italic")
        else:
            label = f"{r['survivors']:,}"
            if r["dropped"]:
                label += f"   (-{r['dropped']:,})"
            ax.text(r["survivors"] + span * 0.012, idx, label,
                    va="center", fontsize=9, color=S.INK)

    for idx, r in enumerate(rows):
        if r["step"] == BOUNDARY_AFTER:
            ax.axhline(idx + 0.5, color=S.BASELINE, linewidth=0.9,
                       linestyle=(0, (4, 3)))
            ax.text(ax.get_xlim()[1], idx + 0.5, " enrichment boundary ",
                    ha="right", va="bottom", fontsize=8, color=S.MUTED)
            break

    return S.save(fig, out_dir, FIGURE_STEM)


# ── stats file ────────────────────────────────────────────────────────────────

def load_stats() -> Dict[str, object]:
    path = paths.EVAL_FRAMEWORKS_FILTER_STATS_JSON
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_stats(stats: Dict[str, object]) -> None:
    paths.EVAL_FRAMEWORKS_FILTER_STATS_JSON.write_text(
        json.dumps(stats, indent=2), encoding="utf-8")


# ── phases ────────────────────────────────────────────────────────────────────

def run_search(max_pages: int = MAX_PAGES, out_path: Optional[Path] = None) -> dict:
    out_path = out_path or CANDIDATES_CSV
    queries = build_search_queries()

    print("Pre-flight - reported total_count per phrase (nothing paged yet):")
    pre = preflight(queries)
    width = max(len(r["query"]) for r in pre)
    for r in pre:
        flag = "   TRUNCATED (>1000: relevance-ranked prefix only)" if r["truncated"] else ""
        print(f"  {r['query']:<{width}}  {r['total_count']:>8,}{flag}")
    print()

    results: List[Tuple[str, List[dict]]] = []
    retrieved = 0
    for r in pre:
        items = gs.search_repositories(r["query"], max_pages=max_pages)
        retrieved += len(items)
        results.append((r["query"], items))
        print(f"  retrieved {len(items):>4} / {r['total_count']:,} reported "
              f"for [{r['query']}]" + ("   (truncated)" if r["truncated"] else ""))
        time.sleep(SEARCH_PACE_SEC)
    print()

    deduped = dedupe(results)
    survivors, funnel = apply_free_filters([v["item"] for v in deduped.values()])

    framework_names = load_frameworks_full_names()
    rows = [candidate_row(v["item"], v["matched_queries"], framework_names)
            for v in deduped.values()]
    write_candidates(rows, out_path)

    stats = {
        "phase": "search",
        "phrases": len(queries),
        "per_query": [{"query": r["query"], "total_count": r["total_count"],
                       "retrieved": len(items), "truncated": r["truncated"]}
                      for r, (_, items) in zip(pre, results)],
        "retrieved": retrieved,
        **funnel,
        "confirmed": None,
    }
    merged = {**load_stats(), **stats}
    save_stats(merged)

    seeded_found = sorted(r["full_name"] for r in rows if r["already_seeded"] == "1")
    print("Funnel (free tier):")
    print(f"  search hits retrieved   : {retrieved:,}")
    print(f"  deduped                 : {funnel['deduped']:,}")
    print(f"  dropped (fork/archived) : {funnel['dropped_fork_archived']:,}")
    print(f"  dropped (stars<{MIN_STARS})      : {funnel['dropped_stars']:,}")
    print(f"  dropped (not {LANGUAGE})    : {funnel['dropped_language']:,}")
    print(f"  survivors               : {funnel['kept_free']:,}")
    print(f"\nSanity check - seeded evaluators surfaced: {len(seeded_found)}/"
          f"{len(SEEDED_EVALUATOR_REPOS)}")
    for name in seeded_found:
        print(f"    found   {name}")
    missing = set(SEEDED_EVALUATOR_REPOS) - {n.lower() for n in seeded_found}
    for name in sorted(missing):
        print(f"    MISSING {name}   <- the phrase set has a hole here")
    print(f"\nWrote {len(rows)} candidate rows to {out_path}")
    print(f"  ({len(survivors)} pass the free filters; `confirmed` is blank - "
          f"fill it by hand, then run --enrich)")
    print(f"Wrote {render_waterfall(merged)}")
    return merged


def run_enrich(limit: Optional[int] = None,
               candidates_path: Optional[Path] = None,
               out_path: Optional[Path] = None) -> dict:
    out_path = out_path or paths.EVAL_FRAMEWORKS_CSV
    rows = read_candidates(candidates_path)
    confirmed = confirmed_summary(rows)
    if confirmed is None:
        print("`confirmed` is blank in every row - nothing has been triaged yet.\n"
              "Fill the column with 1/0 by hand, then re-run --enrich.")
        return load_stats()

    targets = [r for r in rows
               if (r.get("confirmed") or "").strip() == "1"
               and candidate_passes_free_filters(r)]
    if limit is not None:
        targets = targets[:limit]
    print(f"Enriching {len(targets)} triaged survivors "
          f"(~4 API calls each, ~{len(targets) * 4} requests)")

    enriched: List[dict] = []
    for i, row in enumerate(targets, start=1):
        full_name = row["full_name"]
        owner, repo = full_name.split("/", 1)
        print(f"[{i}/{len(targets)}] Enriching {full_name}")
        try:
            # get_repo_details first: enrich_repo reads default_branch off the item
            # it is handed, and the candidate CSV does not carry one.
            details = gs.get_repo_details(owner, repo)
            if not details:
                print(f"  skipped {full_name}: repo details unavailable")
                continue
            item = dict(details, full_name=full_name)
            enriched.append(gs.enrich_repo(item, row.get("matched_queries", "")))
            time.sleep(0.2)
        except Exception as e:      # noqa: BLE001 — one bad repo must not end the run
            print(f"  skipped {full_name}: {e}")

    kept, gs_stats = gs.apply_filters(enriched)
    gs.write_csv(kept, out_path)

    seeded = [r["full_name"] for r in kept
              if already_seeded_from_import_names(r.get("import_names", ""))]

    stats = {**load_stats(), "phase": "enrich", "enriched": gs_stats["enriched"],
             "dropped_contributors": gs_stats["dropped_contributors"],
             "dropped_no_tests": gs_stats["dropped_no_tests"],
             "kept_enriched": gs_stats["kept"], "confirmed": confirmed,
             "already_seeded_by_import_name": sorted(seeded)}
    save_stats(stats)

    print("\nFunnel (enrichment tier):")
    print(f"  enriched                       : {gs_stats['enriched']:,}")
    print(f"  dropped (contributors<{MIN_CONTRIBUTORS})        : "
          f"{gs_stats['dropped_contributors']:,}")
    print(f"  dropped (test files<{MIN_TEST_FILES})          : "
          f"{gs_stats['dropped_no_tests']:,}")
    print(f"  kept                           : {gs_stats['kept']:,}")
    print(f"  already seeded (by import name): {len(seeded)}")
    print(f"\nWrote {len(kept)} rows to {out_path}")
    print(f"Wrote {render_waterfall(stats)}")
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--search-only", action="store_true",
                      help="Phase A: free search + candidate CSV + waterfall")
    mode.add_argument("--enrich", action="store_true",
                      help="Phase B: Stage-1 enrichment over triaged survivors")
    mode.add_argument("--figure-only", action="store_true",
                      help="Redraw the waterfall from the saved stats + candidate CSV")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES,
                        help=f"Search pages per phrase (default {MAX_PAGES}, "
                             f"GitHub's {RESULT_CEILING}-result ceiling)")
    parser.add_argument("--limit", type=int, default=None,
                        help="--enrich: enrich at most N repos (quick test runs)")
    parser.add_argument("--candidates", type=Path, default=None,
                        help=f"Candidate CSV path (default {CANDIDATES_CSV})")
    parser.add_argument("--out", type=Path, default=None,
                        help="--enrich: output CSV "
                             "(default pipeline/artifacts/eval_frameworks.csv)")
    args = parser.parse_args(argv)

    paths.ensure_dirs()

    if args.search_only:
        run_search(max_pages=args.max_pages, out_path=args.candidates)
    elif args.enrich:
        run_enrich(limit=args.limit, candidates_path=args.candidates,
                   out_path=args.out)
    else:
        stats = load_stats()
        path = args.candidates or CANDIDATES_CSV
        if path.exists():
            stats["confirmed"] = confirmed_summary(read_candidates(path))
        print(f"Wrote {render_waterfall(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
