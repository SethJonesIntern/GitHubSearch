"""Holistic false-positive audit over llm_calls_all.csv — the standing FP-discovery tool.

FPs cluster by RECEIVER, so this profiles the receiver expressions that matched each
(framework, pattern). Run it after any Stage-5 run to (a) size known FP tiers and
(b) DISCOVER new ones by scanning receiver profiles.

    py -3.14 Applications/audit_fp.py [path/to/llm_calls_all.csv]

Emits to <artifacts>/analysis/:
  pattern_audit.csv  — per (framework,pattern): volume, receiver diversity, stdlib-root
                       hits, non-terminal hits, confident-FP %, suspicion score, top roots.
  receiver_pivot.csv — (framework,pattern,receiver_root,calls) for drill-down.

HOW TO FIND MORE FP CATEGORIES (the workflow):
  1. Sort pattern_audit.csv by pct_confident_fp AND by distinct_receiver_roots (desc).
  2. On high-diversity patterns, scan top_receiver_roots for NON-MODEL nouns
     (tool, sandbox, driver, template, retriever, parser, mock, asyncio, session, db...).
  3. Confirm a suspect receiver in receiver_pivot.csv, then add it as an exclusion tier
     in EXCLUSIONS.md. Keep model receivers (model_with_tools/llm_with_tools/chain/*_model).

Suspicion signals (objective, no intent-guessing):
  stdlib_root  — receiver root is a stdlib/util module (asyncio/subprocess/os/re/mock...)
  nonterminal  — matched method isn't the terminal call segment (accessed, not called)
  diversity    — # distinct receiver roots (a collision-prone verb sprays across many)
"""
import csv, re, sys, collections
from pathlib import Path

CALLS = Path(sys.argv[1] if len(sys.argv) > 1 else
             "pipeline/artifacts/llm_calls_all.csv")
OUTDIR = CALLS.parent / "analysis"
OUTDIR.mkdir(exist_ok=True)

STDLIB = {"asyncio", "subprocess", "os", "re", "sys", "functools", "itertools",
          "logging", "pytest", "unittest", "threading", "multiprocessing", "json",
          "time", "contextlib", "mock", "Mock", "MagicMock", "AsyncMock", "patch"}


def receiver_root(callable_text: str) -> str:
    """Leftmost identifier of the call expression: 'self' for self.x.run, 're' for re.search."""
    m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", callable_text.strip())
    return m.group(0) if m else "<expr>"


def is_terminal(callable_text: str, pattern: str) -> bool:
    """The matched method is the one actually invoked iff it is the terminal segment of the
    call expression. Uses endswith (NOT 'appears with a trailing dot') so a legit self.run.run
    is kept while agent.arun.assert_called_once is dropped."""
    if pattern.startswith("."):
        method = pattern[1:]
        return callable_text.endswith("." + method) or callable_text == method
    return callable_text == pattern or callable_text.endswith("." + pattern)


def main() -> None:
    per = collections.defaultdict(lambda: {
        "n": 0, "roots": collections.Counter(), "stdlib": 0, "nonterm": 0})
    pivot = collections.Counter()

    with CALLS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fw, pat, call = row["framework"], row["pattern"], row["callable"]
            root = receiver_root(call)
            rec = per[(fw, pat)]
            rec["n"] += 1
            rec["roots"][root] += 1
            if root in STDLIB:
                rec["stdlib"] += 1
            if not is_terminal(call, pat):
                rec["nonterm"] += 1
            pivot[(fw, pat, root)] += 1

    rows = []
    for (fw, pat), rec in per.items():
        n = rec["n"]
        suspect = rec["stdlib"] + rec["nonterm"]
        diversity = len(rec["roots"])
        top = "; ".join(f"{r}:{c}" for r, c in rec["roots"].most_common(5))
        rows.append({
            "framework": fw, "pattern": pat, "calls": n,
            "distinct_receiver_roots": diversity,
            "stdlib_root_hits": rec["stdlib"], "nonterminal_hits": rec["nonterm"],
            "confident_fp": suspect, "pct_confident_fp": round(100 * suspect / n, 1),
            "suspicion_score": round(suspect + diversity / 2, 1),
            "top_receiver_roots": top,
        })
    rows.sort(key=lambda r: (r["suspicion_score"], r["calls"]), reverse=True)

    with (OUTDIR / "pattern_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with (OUTDIR / "receiver_pivot.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["framework", "pattern", "receiver_root", "calls"])
        for (fw, pat, root), c in sorted(pivot.items(), key=lambda kv: -kv[1]):
            w.writerow([fw, pat, root, c])

    print(f"{'framework':<18}{'pattern':<14}{'calls':>6}{'roots':>6}{'FP':>5}{'FP%':>6}  top receivers")
    print("-" * 100)
    for r in rows[:25]:
        print(f"{r['framework']:<18}{r['pattern']:<14}{r['calls']:>6}{r['distinct_receiver_roots']:>6}"
              f"{r['confident_fp']:>5}{r['pct_confident_fp']:>6}  {r['top_receiver_roots'][:52]}")
    print(f"\nwrote {OUTDIR/'pattern_audit.csv'} and {OUTDIR/'receiver_pivot.csv'}")


if __name__ == "__main__":
    main()
