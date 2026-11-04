"""Fills `frameworks_imported` and sets `in_scope` from what each repo ACTUALLY imports.

Scope has been decided by the Stage-2 search token up to now, and that token is a
GitHub code-search hit, not a parsed import — which is how django-haystack apps came
to count as Haystack apps and 48 `claim`/`claiming` repos came to count as pydantic_ai
(EXCLUSIONS.md §9, §10). This pass replaces the guess with the evidence: it reads every
clone and records which known framework, eval tool or provider SDK it imports.

The rule, applied to imports rather than tokens:

  imports a TOP-20 framework          -> in scope; measured; in_scope left as-is
  imports only tail frameworks / SDKs -> in_scope=uncovered. A real LLM application,
                                         but on a framework we deliberately do not
                                         measure because the top-20 already covers
                                         ~90% of the population. Leaves the analyzed
                                         statistics, stays in the coverage denominator.
  imports nothing LLM-related         -> left alone; it is a cut candidate, but the
                                         absence of an import is not by itself proof,
                                         so no verdict is written here.

Repos already carrying in_scope=0 are never touched — a cut decision outranks this.

    py -3.14 -m pipeline.audit_imports --scan     # walk the clones (slow, cached)
    py -3.14 -m pipeline.audit_imports            # apply the rule from the cache
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "Wrapper", _ROOT / "Applications"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import paths  # noqa: E402
from pipeline.eval_calls import EVAL_CALLS  # noqa: E402
from FrameworkDict import FRAMEWORK_CALLS, IN_SCOPE_FRAMEWORKS  # noqa: E402
import keep_frequency as kf  # noqa: E402  (category() — the one grouping table)

AUDIT_CSV = paths.ARTIFACTS_DIR / "application_audit.csv"
IMPORTS_CSV = paths.ARTIFACTS_DIR / "audit_framework_imports.csv"

csv.field_size_limit(10 ** 9)

# Provider SDKs: a direct line to a model, not an agent framework. Importing one makes
# a repo an LLM application; it does not make it an application of a framework we
# study. Deliberately excludes names too generic to attribute (`google`, `transformers`,
# `huggingface_hub` — local models and unrelated cloud APIs share them).
SDK_NAMES = {
    "openai", "anthropic", "litellm", "cohere", "mistralai", "groq", "together",
    "ollama", "replicate", "fireworks", "dashscope", "zhipuai", "vertexai",
    "google_generativeai", "boto3_bedrock", "agents",
}

# Every framework/eval package we can recognise at all — the Stage-1 discovery record,
# which is much wider than the set we measure.
KNOWN = set(FRAMEWORK_CALLS) | set(EVAL_CALLS) | SDK_NAMES

# TOP-20 = what we measure, minus the raw SDKs. Grouped, so `langchain_core` and
# `langchain_openai` both resolve to the one framework the ranking counts.
TOP20_GROUPS = {kf.category(n) for n in IN_SCOPE_FRAMEWORKS} - {kf.category(n) for n in SDK_NAMES}

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".tox", "build",
    "dist", "site-packages", ".mypy_cache", ".pytest_cache", ".ruff_cache", "vendor",
    "third_party", "3rdparty", ".idea", ".vscode", "eggs", ".eggs", ".next",
}
MAX_FILE_BYTES = 1_500_000

# One alternation over the known packages only, rather than matching every import line
# in the corpus and testing each name for membership. The regex engine rejects the
# ~99% of import lines that name something we don't care about far faster than Python
# can, which is the difference between a 15-minute and a 90-minute pass.
IMPORT_RE = re.compile(
    r"^[ \t]*(?:from|import)[ \t]+("
    + "|".join(sorted(map(re.escape, KNOWN), key=len, reverse=True))
    + r")(?=[ \t.,;)\\]|$)", re.M | re.I)


def scan(slug: str):
    """-> (slug, {known package: files importing it}). One walk, never raises."""
    hits = Counter()
    root = paths.REPOS_DIR / slug
    if not root.is_dir():
        return slug, hits
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(p) > MAX_FILE_BYTES:
                        continue
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    continue
                for name in {m.lower() for m in IMPORT_RE.findall(text)}:
                    hits[name] += 1
    except Exception:                    # noqa: BLE001 — one bad clone, not a dead run
        pass
    return slug, hits


def run_scan(workers: int, limit=None) -> None:
    with AUDIT_CSV.open(newline="", encoding="utf-8") as fh:
        audit = list(csv.DictReader(fh))
    slugs = [r["clone_slug"] for r in audit][:limit] if limit else [r["clone_slug"] for r in audit]
    print(f"# scanning {len(slugs)} clones for {len(KNOWN)} known packages, "
          f"{workers} workers")
    start = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(scan, s) for s in slugs]
        for done, fut in enumerate(as_completed(futures), 1):
            slug, hits = fut.result()
            rows.append({"clone_slug": slug,
                         "imports": ";".join(f"{k}:{v}" for k, v in hits.most_common())})
            if done % 100 == 0 or done == len(slugs):
                rate = done / max(time.time() - start, 1e-9)
                print(f"#   {done}/{len(slugs)}  ({rate:.1f}/s, "
                      f"{(len(slugs) - done) / max(rate, 1e-9) / 60:.1f} min left)",
                      file=sys.stderr)
    with IMPORTS_CSV.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=["clone_slug", "imports"])
        w.writeheader()
        w.writerows(rows)
    print(f"# wrote {IMPORTS_CSV}  ({time.time() - start:.0f}s)")


def load_imports() -> dict[str, list[str]]:
    if not IMPORTS_CSV.exists():
        sys.exit(f"{IMPORTS_CSV.name} not found — run with --scan first.")
    out = {}
    with IMPORTS_CSV.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["clone_slug"]] = [p.split(":")[0]
                                    for p in (r["imports"] or "").split(";") if p]
    return out


def apply() -> None:
    imports = load_imports()
    with AUDIT_CSV.open(newline="", encoding="utf-8") as fh:
        audit = list(csv.DictReader(fh))
        fields = list(audit[0].keys())

    changed = Counter()
    for row in audit:
        names = imports.get(row["clone_slug"], [])
        row["frameworks_imported"] = ", ".join(names)
        if row["in_scope"] == "0":
            changed["left as cut"] += 1
            continue                      # a cut outranks anything decided here
        groups = {kf.category(n) for n in names}
        if groups & TOP20_GROUPS:
            if row["in_scope"] == "uncovered":
                row["in_scope"] = row["notes"] = ""    # evidence overrules a token guess
                changed["uncovered -> in scope (imports a top-20 framework)"] += 1
            else:
                changed["in scope"] += 1
        elif names:
            tail = sorted(groups - TOP20_GROUPS)
            row["in_scope"] = "uncovered"
            row["notes"] = ("real LLM app, but imports only "
                            f"{', '.join(tail[:4])} — outside the top-20, not measured")
            changed["-> uncovered"] += 1
        else:
            changed["no LLM import found (left undecided)"] += 1

    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(audit)

    print(f"# {AUDIT_CSV.name} updated from imports, not tokens\n")
    for k, v in changed.most_common():
        print(f"  {k:<48}{v:>5}")
    unc = [r for r in audit if r["in_scope"] == "uncovered"]
    print(f"\n  in_scope=uncovered now {len(unc)} rows. Most common tail frameworks:")
    tail = Counter(g for r in unc for g in
                   ({kf.category(n) for n in (r["frameworks_imported"] or "").split(", ") if n}
                    - TOP20_GROUPS))
    for name, n in tail.most_common(15):
        print(f"    {name:<28}{n:>5}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true", help="walk the clones (slow, cached)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = ap.parse_args()
    if args.scan:
        run_scan(args.workers, args.limit)
        if args.limit:
            return
    apply()


if __name__ == "__main__":
    main()
