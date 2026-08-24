"""Fills the audit sheet's `zero_invoker_reason` / `zero_invoker_evidence` columns.

140-odd analyzed repos produced no LLM invoker at all. "0 invokers" is not one
finding — it is at least five, and they have opposite consequences for the study:

  a repo that never imports the framework Stage 2 said it imports  -> drop it
  a repo that calls a model over raw HTTP or a CLI subprocess      -> a real app we
                                                                      cannot see
  a plugin for a framework, not an application                     -> out of scope
  a repo that imports the framework but calls nothing we match     -> OUR gap

So each zero-invoker repo gets a reason from an ordered cascade, and the raw counts
behind it in `zero_invoker_evidence`, so any rung can be re-judged without a re-scan.

Only the zero-invoker rows are scanned (a few seconds), and only those two columns
are written.

    py -3.14 -m pipeline.audit_zero_invokers
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import paths  # noqa: E402

AUDIT_CSV = paths.ARTIFACTS_DIR / "application_audit.csv"

csv.field_size_limit(10 ** 9)

# In the study but deliberately not measured (EXCLUSIONS.md §2): agentops is an
# observability layer, omnigent runs the model out of process. 0 invocations is the
# intended result for both, never a bug.
EXEMPT_FRAMEWORKS = {"agentops", "omnigent"}

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".tox", "build",
    "dist", "site-packages", ".mypy_cache", ".pytest_cache", ".ruff_cache", "vendor",
    "third_party", "3rdparty", ".idea", ".vscode", "eggs", ".eggs", ".next",
}
MAX_FILE_BYTES = 1_500_000

IMPORT_RE = re.compile(r"^[ \t]*(?:from|import)[ \t]+([A-Za-z_]\w*)", re.M)

# A model called over the wire — invisible to AST method-matching.
HTTP_LLM_RE = re.compile(
    r"api\.openai\.com|openai\.azure\.com|api\.anthropic\.com|api\.groq\.com"
    r"|generativelanguage\.googleapis\.com|openrouter\.ai|api\.mistral\.ai"
    r"|api\.together\.xyz|api\.deepseek\.com|dashscope\.aliyuncs\.com|api\.cohere\."
    r"|localhost:11434|127\.0\.0\.1:11434"
    r"|/v1/chat/completions|/v1/messages|/api/chat\b|/api/generate\b")

# A model driven by shelling out (the omnigent pattern).
CLI_LLM_RE = re.compile(r"\b(claude|ollama|gemini|codex|aider|llama[-_.]cpp)\b")

PLUGIN_RE = re.compile(r"(?:^|[-_])plugin(?:s)?(?:[-_]|$)|_plugin_", re.I)


def scan(job):
    """-> (repo, facts). One walk of one clone; never raises."""
    repo, slug = job
    f = {"clone_present": 0, "py_files": 0, "ipynb_files": 0, "test_files": 0,
         "http_files": 0, "cli_files": 0, "imports": set()}
    root = paths.REPOS_DIR / slug
    if not root.is_dir():
        return repo, f
    f["clone_present"] = 1
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn.endswith(".ipynb"):
                    f["ipynb_files"] += 1
                    continue
                if not fn.endswith(".py"):
                    continue
                f["py_files"] += 1
                if fn.startswith("test_") or fn.endswith("_test.py"):
                    f["test_files"] += 1
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(p) > MAX_FILE_BYTES:
                        continue
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    continue
                f["imports"].update(m.lower() for m in IMPORT_RE.findall(text))
                if HTTP_LLM_RE.search(text):
                    f["http_files"] += 1
                if "subprocess" in text and CLI_LLM_RE.search(text):
                    f["cli_files"] += 1
    except Exception:                       # noqa: BLE001 — one bad clone, not a dead run
        pass
    return repo, f


def imported_frameworks(matched: list[str], imports: set[str]) -> list[str]:
    """Which matched frameworks the clone ACTUALLY imports. A family counts through
    its subpackages (`langchain` <- `langchain_openai`), which is how it is used."""
    return [n for n in matched
            if n.lower() in imports or any(i.startswith(n.lower() + "_") for i in imports)]


def classify(row: dict, facts: dict, hits: list[str]) -> str:
    """Ordered cascade over ONE question — is the matched framework actually used? —
    so the column partitions the zero-invoker set instead of overlapping.

    Out-of-process evidence (raw HTTP to a model endpoint, a CLI subprocess) is
    deliberately NOT a rung here. It coexists with every reason: a repo can both fail
    to import the framework Stage 2 matched AND call a model over HTTP. It lives in
    `http_llm_files` / `cli_llm_files` instead, so it can be crossed with any reason.
    """
    matched = [m.strip() for m in (row["matched_frameworks"] or "").split(",") if m.strip()]
    if row["clone_failed"] == "1" or row["processed"] == "0":
        return "clone_failed"
    if not facts["clone_present"]:
        return "clone_deleted"              # analyzed, but the checkout is gone now
    if facts["py_files"] == 0:
        return "no_python_files"
    if facts["ipynb_files"] > 0 and facts["py_files"] < 5:
        return "notebooks_only"             # our AST pass never opens .ipynb
    if matched and not hits:
        return "framework_never_imported"   # Stage-2 code-search false positive
    if matched and set(matched) <= EXEMPT_FRAMEWORKS:
        return "exempt_framework_only"
    if PLUGIN_RE.search(row["full_name"].split("/")[-1]):
        return "plugin"                     # a plugin for a framework, not an app
    if hits:
        return "imports_fw_no_call_site"    # OUR gap: pattern, or non-LLM use of it
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = ap.parse_args()

    with AUDIT_CSV.open(newline="", encoding="utf-8") as fh:
        audit = list(csv.DictReader(fh))
        fields = list(audit[0].keys())

    targets = [r for r in audit if r["zero_invoker"] == "1"]
    print(f"# {len(targets)} zero-invoker repos to explain")

    facts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(scan, (r["full_name"], r["clone_slug"])) for r in targets]
        for done, fut in enumerate(as_completed(futures), 1):
            repo, f = fut.result()
            facts[repo] = f
            if done % 25 == 0 or done == len(targets):
                print(f"#   {done}/{len(targets)} clones", file=sys.stderr)

    for row in audit:
        if row["zero_invoker"] != "1":
            continue
        f = facts[row["full_name"]]
        matched = [m.strip() for m in (row["matched_frameworks"] or "").split(",")
                   if m.strip()]
        hits = imported_frameworks(matched, f["imports"])
        row["zero_invoker_reason"] = classify(row, f, hits)
        row["imports_matched_fw"] = ", ".join(hits)
        row["py_files"] = f["py_files"]
        row["ipynb_files"] = f["ipynb_files"]
        row["test_files"] = f["test_files"]
        row["http_llm_files"] = f["http_files"]
        row["cli_llm_files"] = f["cli_files"]

    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(audit)

    counts = Counter(r["zero_invoker_reason"] for r in targets)
    print(f"\n# {AUDIT_CSV.name}: zero_invoker_reason filled for {len(targets)} rows\n")
    print(f"  {'reason':<26}{'repos':>6}{'of zero-inv':>13}{'of population':>15}"
          f"{'+http':>7}{'+cli':>6}")
    for reason, n in counts.most_common():
        sub = [r for r in targets if r["zero_invoker_reason"] == reason]
        http = sum(1 for r in sub if int(r["http_llm_files"] or 0) > 0)
        cli = sum(1 for r in sub if int(r["cli_llm_files"] or 0) > 0)
        print(f"  {reason:<26}{n:>6}{100 * n / len(targets):>12.1f}%"
              f"{100 * n / len(audit):>14.1f}%{http:>7}{cli:>6}")
    print("\n  +http / +cli = repos in that bucket that ALSO show out-of-process LLM\n"
          "  evidence (a model called over raw HTTP or via a CLI subprocess), which is\n"
          "  invisible to AST method-matching regardless of why the reason fired.")


if __name__ == "__main__":
    main()
