"""Fills the audit sheet's `framework_suspect` / `framework_evidence` columns.

The question: is a repo with a large call/test count actually a FRAMEWORK (or a
framework's own integration/instrumentation package) rather than an application
built on one? The decisive evidence available to us is our own corpus: a library is
imported by other repos, an application is imported by nobody.

Three signals, all recorded in `framework_evidence` so any of them can be re-judged:

  imported_by=N   how many OTHER cloned repos import a package this repo publishes.
                  This is the strong one. Package names come from the clone's own
                  packaging metadata (pyproject/setup.py `name=`) and its top-level
                  `__init__.py` packages; generic directory names (`app`, `core`,
                  `utils`, ...) are excluded because they identify nothing.
  stage1_framework  the repo is on the Stage-1 discovered-framework list.
  owner_matches   the repo's OWNER is the framework it matched (langchain-ai/…
                  matching langchain), i.e. it is first-party framework code.

Two passes, so the expensive one runs once:

    py -3.14 -m pipeline.audit_framework_check --scan     # walk clones -> hits cache
    py -3.14 -m pipeline.audit_framework_check            # fill the sheet from cache

`--scan --limit N` scans only N clones, for timing. The scan cache is
`audit_import_hits.csv`; the sheet is only ever updated in these two columns.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import tomllib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "Wrapper"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import paths  # noqa: E402
from pipeline.eval_calls import EVAL_CALLS  # noqa: E402
from FrameworkDict import FRAMEWORK_CALLS  # noqa: E402

AUDIT_CSV = paths.ARTIFACTS_DIR / "application_audit.csv"
HITS_CSV = paths.ARTIFACTS_DIR / "audit_import_hits.csv"

csv.field_size_limit(10 ** 9)

# A repo imported by at least this many OTHER cloned repos is behaving as a library
# inside our own corpus, not as a leaf application.
SUSPECT_IMPORTERS = 2

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".tox", "build",
    "dist", "site-packages", ".mypy_cache", ".pytest_cache", ".ruff_cache", "vendor",
    "third_party", "3rdparty", ".idea", ".vscode", "eggs", ".eggs", ".next",
}
MAX_FILE_BYTES = 1_500_000

# Directory conventions, not distributable packages: if repo B contains `agents/`,
# `import agents` in repo C says nothing about repo B.
GENERIC_NAMES = {
    "src", "app", "apps", "api", "core", "utils", "util", "tests", "test", "main",
    "config", "configs", "common", "lib", "libs", "tools", "agent", "agents",
    "models", "model", "scripts", "script", "examples", "example", "data", "server",
    "client", "backend", "frontend", "base", "settings", "database", "db", "schemas",
    "schema", "services", "service", "routes", "ui", "web", "cli", "types", "docs",
    "constants", "helpers", "handlers", "plugins", "plugin", "modules", "module",
    "shared", "internal", "pkg", "python", "setup", "demo", "demos", "notebooks",
    "workflows", "workflow", "prompts", "chains", "graph", "graphs", "memory",
    "middleware", "adapters", "providers", "integrations", "evaluation", "eval",
    "benchmark", "benchmarks", "training", "train", "inference", "pipeline",
    "pipelines", "project", "package", "source", "code", "bot", "bots", "assets",
    "chatbot", "llm", "llms", "rag", "chat", "gui", "frontend", "static", "templates",
}

DIST_NAME_RE = re.compile(r"""^\s*name\s*[=:]\s*["']([A-Za-z0-9_.\-]+)["']""", re.M)
SETUP_NAME_RE = re.compile(r"""\bname\s*=\s*["']([A-Za-z0-9_.\-]+)["']""")

# Third-party packages every corpus repo imports. A repo that merely CONTAINS a
# directory of one of these names does not publish it, and counting its importers
# would attribute the whole ecosystem's usage to one application.
THIRD_PARTY = set(FRAMEWORK_CALLS) | set(EVAL_CALLS) | {
    "mcp", "datasets", "transformers", "channels", "openai", "anthropic", "litellm",
    "torch", "numpy", "pandas", "fastapi", "django", "flask", "pydantic", "requests",
    "httpx", "boto3", "redis", "celery", "sqlalchemy", "streamlit", "gradio",
    "chromadb", "qdrant_client", "weaviate", "pinecone", "kuzu", "neo4j", "duckdb",
    "ollama", "tiktoken", "sentence_transformers", "wandb", "mlflow", "openlit",
}


# ── the names each repo publishes ─────────────────────────────────────────────

def declared_names(slug: str) -> list[str]:
    """The package name a repo DECLARES in its packaging metadata, normalised to
    import form. Declaration is the point: a top-level directory called `mcp` or
    `datasets` proves only that the repo has a directory by that name, whereas
    `[project] name` is the author asserting what this repo publishes."""
    root = paths.REPOS_DIR / slug
    names: list[str] = []

    p = root / "pyproject.toml"
    if p.is_file():
        try:
            data = tomllib.loads(p.read_text(encoding="utf-8", errors="replace"))
            for path in (("project", "name"), ("tool", "poetry", "name")):
                node = data
                for key in path:
                    node = node.get(key) if isinstance(node, dict) else None
                if isinstance(node, str):
                    names.append(node)
        except (OSError, ValueError, TypeError):
            pass

    p = root / "setup.py"
    if p.is_file():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            call = text.find("setup(")
            if call != -1:
                m = SETUP_NAME_RE.search(text, call, call + 3000)
                if m:
                    names.append(m.group(1))
        except OSError:
            pass

    p = root / "setup.cfg"
    if p.is_file():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            meta = text.find("[metadata]")
            if meta != -1:
                m = DIST_NAME_RE.search(text, meta)
                if m:
                    names.append(m.group(1))
        except OSError:
            pass

    out = []
    for raw in names:
        n = raw.strip().replace("-", "_").lower()
        if n and n not in out and len(n) >= 3 and n not in GENERIC_NAMES \
                and n not in THIRD_PARTY and importable(root, n):
            out.append(n)
    return out


def importable(root: Path, name: str) -> bool:
    """The declared name must exist in the tree as something another repo could
    actually import — `name/__init__.py`, `src/name/__init__.py`, or `name.py`.
    Without this a declared-but-absent name (a renamed or namespace package) would
    collect importers that belong to a different project."""
    for base in (root, root / "src"):
        if (base / name / "__init__.py").is_file() or (base / f"{name}.py").is_file():
            return True
    return False


# ── the scan ──────────────────────────────────────────────────────────────────

def scan_clone(job):
    """-> (slug, {name: n_files}) for the candidate names imported by this clone."""
    slug, pattern = job
    hits = Counter()
    rx = re.compile(pattern, re.M)
    root = paths.REPOS_DIR / slug
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
                for name in set(rx.findall(text)):
                    hits[name.lower()] += 1
    except Exception:                       # noqa: BLE001 — a bad clone must not stop the scan
        pass
    return slug, hits


def claimed_names(audit) -> dict[str, list[str]]:
    """name -> the audited repos declaring it. A name two repos both claim identifies
    neither, so `fill` refuses to count its importers."""
    out = defaultdict(list)
    for row in audit:
        for n in declared_names(row["clone_slug"]):
            out[n].append(row["full_name"])
    return out


def build_pattern(names) -> str:
    """One alternation over every candidate name, anchored to an import statement so
    a mere mention in a string or comment doesn't count."""
    alt = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
    return rf"^[ \t]*(?:from|import)[ \t]+({alt})(?=[ \t.,;)\\]|$)"


def run_scan(limit=None, workers=8) -> None:
    with AUDIT_CSV.open(newline="", encoding="utf-8") as fh:
        audit = list(csv.DictReader(fh))

    print(f"# deriving package names for {len(audit)} audited repos...")
    owner_of = claimed_names(audit)
    print(f"#   {len(owner_of)} distinct candidate package names")

    slugs = sorted(e.name for e in os.scandir(paths.REPOS_DIR) if e.is_dir())
    if limit:
        slugs = slugs[:limit]
    pattern = build_pattern(owner_of)
    print(f"# scanning {len(slugs)} clones with {workers} workers...")

    start = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(scan_clone, (s, pattern)) for s in slugs]
        for done, fut in enumerate(as_completed(futures), 1):
            slug, hits = fut.result()
            rows.extend({"importer_slug": slug, "name": n, "files": c}
                        for n, c in hits.items())
            if done % 50 == 0 or done == len(slugs):
                rate = done / max(time.time() - start, 1e-9)
                print(f"#   {done}/{len(slugs)} clones  "
                      f"({rate:.1f}/s, {(len(slugs) - done) / max(rate, 1e-9) / 60:.1f} min left)",
                      file=sys.stderr)

    with HITS_CSV.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=["importer_slug", "name", "files"])
        w.writeheader()
        w.writerows(rows)
    print(f"# wrote {HITS_CSV}  ({len(rows)} hits, {time.time() - start:.0f}s)")


# ── filling the sheet ─────────────────────────────────────────────────────────

def stage1_frameworks() -> set[str]:
    out = set()
    if paths.FRAMEWORKS_CSV.exists():
        with paths.FRAMEWORKS_CSV.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if (r.get("full_name") or "").strip():
                    out.add(r["full_name"].strip())
    return out


def name_relates_to_repo(name: str, repo: str) -> bool:
    """Does the declared package name plausibly belong to THIS repo?

    A fork keeps its upstream's packaging metadata, so `haohervchb/GooseLLM` declares
    `name = "vllm"` and would otherwise collect every vLLM importer in the corpus as
    evidence that GooseLLM is a framework. Requiring the name and the repo to share a
    word catches that without hand-listing forks: `getsentry/sentry-python` ->
    `sentry_sdk` shares `sentry`, `DataDog/dd-trace-py` -> `ddtrace` shares `trace`,
    but `GooseLLM` -> `vllm` shares nothing.
    """
    flat = re.sub(r"[^a-z0-9]", "", repo.lower())
    n = name.replace("_", "").lower()
    if n in flat:
        return True
    return any(len(tok) >= 3 and tok in n
               for tok in re.split(r"[^a-z0-9]+", repo.lower()) if tok)


def owner_matches_framework(repo: str, matched: str) -> str:
    """`langchain-ai/langchain-google` matching `langchain` is first-party framework
    code, not an application built on the framework."""
    owner = repo.split("/")[0].replace("-", "").replace("_", "").lower()
    for name in (m.strip() for m in matched.split(",")):
        if not name:
            continue
        base = name.split("_")[0].lower()
        if len(base) >= 4 and (base in owner or owner.startswith(base)):
            return name
    return ""


def fill() -> None:
    if not HITS_CSV.exists():
        sys.exit(f"{HITS_CSV.name} not found — run with --scan first.")
    with AUDIT_CSV.open(newline="", encoding="utf-8") as fh:
        audit = list(csv.DictReader(fh))
        fields = audit[0].keys()

    slug_to_repo = {r["clone_slug"]: r["full_name"] for r in audit}
    importers: dict[str, set[str]] = defaultdict(set)
    with HITS_CSV.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            importers[r["name"]].add(slug_to_repo.get(r["importer_slug"],
                                                      r["importer_slug"]))

    stage1 = stage1_frameworks()
    claimed = claimed_names(audit)
    changed = 0
    for row in audit:
        repo = row["full_name"]
        best_name, best, examples, ambiguous, unrelated = "", set(), [], [], []
        for name in declared_names(row["clone_slug"]):
            if len(claimed.get(name, ())) > 1:
                ambiguous.append(name)      # two repos declare it; it names neither
                continue
            if not name_relates_to_repo(name, repo):
                unrelated.append(name)      # inherited from an upstream project
                continue
            others = importers.get(name, set()) - {repo, row["clone_slug"]}
            if len(others) > len(best):
                best_name, best = name, others
                examples = sorted(others)[:3]

        evidence = []
        if best:
            evidence.append(f"imported_by={len(best)} as `{best_name}` "
                            f"(e.g. {', '.join(examples)})")
        if ambiguous:
            evidence.append(f"ambiguous_name={','.join(ambiguous)}")
        if unrelated:
            evidence.append(f"upstream_name?={','.join(unrelated)}")
        if repo in stage1:
            evidence.append("stage1_framework")
        owner_hit = owner_matches_framework(repo, row["matched_frameworks"])
        if owner_hit:
            evidence.append(f"owner_matches={owner_hit}")

        suspect = "1" if (len(best) >= SUSPECT_IMPORTERS or repo in stage1
                          or owner_hit) else "0"
        if row["framework_suspect"] != suspect or row["framework_evidence"] != "; ".join(evidence):
            changed += 1
        row["framework_suspect"] = suspect
        row["framework_evidence"] = "; ".join(evidence)

    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=list(fields), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(audit)

    suspects = [r for r in audit if r["framework_suspect"] == "1"]
    hv = [r for r in suspects if int(r["nd_tests"] or 0) >= 1000
          or int(r["llm_calls"] or 0) >= 200]
    print(f"# updated {changed} rows in {AUDIT_CSV.name}")
    print(f"#   framework_suspect=1 : {len(suspects)} / {len(audit)} "
          f"({100 * len(suspects) / len(audit):.1f}%)")
    print(f"#   of those, high-volume: {len(hv)}")
    print(f"\n{'repo':<45}{'calls':>7}{'ndtests':>9}  evidence")
    for r in sorted(suspects, key=lambda r: -int(r["nd_tests"] or 0))[:25]:
        print(f"{r['full_name'][:44]:<45}{r['llm_calls']:>7}{r['nd_tests']:>9}  "
              f"{r['framework_evidence'][:70]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true", help="walk the clones (slow, cached)")
    ap.add_argument("--limit", type=int, help="scan only N clones (timing run)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = ap.parse_args()
    if args.scan:
        run_scan(args.limit, args.workers)
        if args.limit:
            return          # a partial cache would understate every count
    fill()


if __name__ == "__main__":
    main()
