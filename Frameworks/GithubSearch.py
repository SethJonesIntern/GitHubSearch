"""Stage 1 — Framework search.

Mines GitHub for candidate agent-framework repositories using the same approach
as the prior work by Mehedi Hasan: a set of natural-language search phrases run
through GitHub repository search, then a fixed set of quality filters.

Filter conditions (per spec):
  * Contributor count >= 2   (well-maintained projects)
  * Star count >= 1000       (popularity proxy)
  * Language: Python
  * Number of test files >= 1 (testing-practices focus)
  * Not archived

Beyond the filters we capture as much repo metadata as the API cheaply gives, so
downstream stages have a rich record. Output: pipeline/artifacts/frameworks.csv.
"""
import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Make the repo-root `pipeline` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths  # noqa: E402

load_dotenv()  # repo-root / CWD .env
load_dotenv(Path(__file__).resolve().parent / ".env")  # Frameworks/.env (token lives here)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

API_BASE = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

RAW_HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# ── search spec ───────────────────────────────────────────────────────────────
# Natural-language phrases mined in the prior work. The popularity/language
# qualifiers are spec'd filter conditions, kept in one place and appended to
# every phrase so they're applied at search time (cheap) as well.
SEARCH_PHRASES = [
    "AI agent framework",
    "LLM-based agent framework",
    "LLM agent library",
    "multi-agent orchestration framework",
    "LLM powered agents framework",
]
MIN_STARS = 1000
LANGUAGE = "Python"

# ── filter thresholds (per spec) ──────────────────────────────────────────────
MIN_CONTRIBUTORS = 2
MIN_TEST_FILES = 1

PER_PAGE = 100
MAX_PAGES_PER_QUERY = 3
TEST_FILE_RE = re.compile(r"(^|/)test_[^/]+\.py$")
LAST_PAGE_RE = re.compile(r'[?&]page=(\d+)>;\s*rel="last"')


def build_search_queries() -> List[str]:
    """One GitHub search query per phrase, with the spec'd star/language qualifiers."""
    qualifier = f"stars:>={MIN_STARS} language:{LANGUAGE}"
    return [f"{phrase} {qualifier}" for phrase in SEARCH_PHRASES]


def github_get(url: str, params: Optional[dict] = None, allow_404: bool = False,
               max_retries: int = 3) -> Optional[requests.Response]:
    """GET with rate-limit waiting, network-error retry, and 5xx retry.

    Returns None for 404/409/451 when allow_404 is set; otherwise raises on
    persistent client/server errors.
    """
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait = 2 ** attempt * 5
                print(f"Request error ({e}). Retrying in {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            raise
        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = resp.headers.get("X-RateLimit-Reset")
            if reset:
                sleep_for = max(int(reset) - int(time.time()) + 2, 2)
                print(f"Rate limit hit. Sleeping {sleep_for}s...")
                time.sleep(sleep_for)
                continue
        if allow_404 and resp.status_code in (404, 409, 451):
            return None
        if resp.status_code >= 500 and attempt < max_retries:
            wait = 2 ** attempt * 5
            print(f"Server error {resp.status_code}. Retrying in {wait}s... ({attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    return None


def _last_page_from_link(link_header: str) -> Optional[int]:
    m = LAST_PAGE_RE.search(link_header or "")
    return int(m.group(1)) if m else None


def search_repositories(query: str, max_pages: int = MAX_PAGES_PER_QUERY) -> List[dict]:
    results = []
    for page in range(1, max_pages + 1):
        params = {"q": query, "per_page": PER_PAGE, "page": page}
        resp = github_get(f"{API_BASE}/search/repositories", params=params)
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        results.extend(items)
        print(f"Query page {page}: got {len(items)} repos for [{query}]")
        if len(items) < PER_PAGE:
            break
    return results


def get_contributor_count(owner: str, repo: str) -> int:
    """Number of contributors (anonymous included). One call via the Link-header
    last-page trick; falls back to counting a single page."""
    resp = github_get(
        f"{API_BASE}/repos/{owner}/{repo}/contributors",
        params={"per_page": 1, "anon": "true"},
        allow_404=True,
    )
    if resp is None or resp.status_code == 204:
        return 0
    last = _last_page_from_link(resp.headers.get("Link", ""))
    if last is not None:
        return last
    data = resp.json()
    return len(data) if isinstance(data, list) else 0


def get_repo_details(owner: str, repo: str) -> Optional[dict]:
    """Full repo payload. The /search/repositories item omits subscribers_count
    and network_count (and a few flags), which only the per-repo GET returns."""
    resp = github_get(f"{API_BASE}/repos/{owner}/{repo}", allow_404=True)
    return resp.json() if resp is not None else None


def get_default_branch_commit_date(owner: str, repo: str, default_branch: str) -> Optional[str]:
    resp = github_get(
        f"{API_BASE}/repos/{owner}/{repo}/commits",
        params={"sha": default_branch, "per_page": 1, "page": 1},
        allow_404=True,
    )
    if resp is None:
        return None
    data = resp.json()
    if not data:
        return None
    try:
        return data[0]["commit"]["committer"]["date"]
    except (KeyError, IndexError, TypeError):
        return None


def get_tree(owner: str, repo: str, default_branch: str) -> list:
    """The recursive git tree (list of blob/tree entries). Empty on failure.
    Fetched once and reused for test metrics, CI detection, and import names."""
    resp = github_get(
        f"{API_BASE}/repos/{owner}/{repo}/git/trees/{default_branch}",
        params={"recursive": "1"},
        allow_404=True,
    )
    if resp is None:
        return []
    return resp.json().get("tree", [])


# Directories whose __init__.py shouldn't count as the project's import name.
_NON_PACKAGE_DIRS = {
    "tests", "test", "testing", "docs", "doc", "examples", "example",
    "scripts", "benchmarks", "benchmark", "samples", "sample", "build",
}


def derive_import_names(tree: list) -> list:
    """Derive the repo's importable top-level package name(s) from its tree.

    The import name is the directory you'd `import` — i.e. a package root: a
    directory that has an __init__.py while its parent does not. This handles
    src/ layouts and monorepos (e.g. libs/langchain/langchain/ -> 'langchain')
    where the repo/PyPI name doesn't match the import name. Test/doc/example
    packages are excluded. Returns a sorted, de-duped list (a repo may ship
    several top-level packages)."""
    init_dirs = set()
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if path == "__init__.py":
            init_dirs.add("")
        elif path.endswith("/__init__.py"):
            init_dirs.add(path[: -len("/__init__.py")])

    roots = []
    for d in init_dirs:
        parent = d.rsplit("/", 1)[0] if "/" in d else ""
        if parent in init_dirs:
            continue  # d is a subpackage, not a root
        segments = d.split("/") if d else []
        # Drop packages living under a test/docs/examples/etc. directory
        # (e.g. examples/foo/__init__.py is a sample, not the project's package).
        if any(seg.lower() in _NON_PACKAGE_DIRS for seg in segments):
            continue
        name = segments[-1] if segments else ""
        # A real import name is a valid identifier — filters cookiecutter
        # template dirs like '{{folder_name}}'.
        if not name or not name.isidentifier():
            continue
        roots.append(name)
    return sorted(set(roots))


def get_test_metrics(owner: str, repo: str, default_branch: str,
                     tree: Optional[list] = None) -> Tuple[int, int, bool]:
    """Count test_*.py files, AST-count test_* functions within them, and detect
    CI (a .github/workflows/ entry). Reuses a pre-fetched `tree` when given.
    Best-effort: any file that fails to fetch/parse is skipped."""
    if tree is None:
        tree = get_tree(owner, repo, default_branch)

    test_files = [
        item["path"] for item in tree
        if item.get("type") == "blob" and TEST_FILE_RE.search(item.get("path", ""))
    ]
    has_ci = any(item.get("path", "").startswith(".github/workflows/") for item in tree)

    test_function_count = 0
    for path in test_files:
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{path}"
        try:
            r = requests.get(raw_url, headers=RAW_HEADERS, timeout=15)
            if r.status_code == 200:
                parsed = ast.parse(r.text)
                test_function_count += sum(
                    1 for node in ast.walk(parsed)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test_")
                )
        except Exception:
            pass
        time.sleep(0.05)

    return len(test_files), test_function_count, has_ci


def enrich_repo(repo_item: dict, matched_query: str) -> dict:
    full_name = repo_item["full_name"]
    owner, repo = full_name.split("/", 1)
    default_branch = repo_item.get("default_branch", "main")

    # Merge the search item with the full repo payload (adds subscribers_count /
    # network_count / extra flags). details overrides the search item.
    details = get_repo_details(owner, repo) or {}
    data = {**repo_item, **details}
    owner_obj = data.get("owner") or {}

    tree = get_tree(owner, repo, default_branch)
    import_names = derive_import_names(tree)
    contributor_count = get_contributor_count(owner, repo)
    latest_commit_date = get_default_branch_commit_date(owner, repo, default_branch)
    test_file_count, test_function_count, has_ci = get_test_metrics(
        owner, repo, default_branch, tree)

    return {
        "full_name": full_name,
        "html_url": data.get("html_url"),
        "description": data.get("description"),
        "matched_query": matched_query,
        "homepage": data.get("homepage"),
        "owner_login": owner_obj.get("login"),
        "owner_type": owner_obj.get("type"),
        "stars": data.get("stargazers_count"),
        "forks": data.get("forks_count"),
        "watchers": data.get("watchers_count"),
        "subscribers_count": data.get("subscribers_count"),
        "network_count": data.get("network_count"),
        "language": data.get("language"),
        "import_names": ";".join(import_names),
        "topics": ",".join(data.get("topics", [])) if data.get("topics") else "",
        "open_issues": data.get("open_issues_count"),
        "size_kb": data.get("size"),
        "default_branch": default_branch,
        "visibility": data.get("visibility"),
        "is_template": data.get("is_template"),
        "allow_forking": data.get("allow_forking"),
        "has_issues": data.get("has_issues"),
        "has_projects": data.get("has_projects"),
        "has_wiki": data.get("has_wiki"),
        "has_pages": data.get("has_pages"),
        "has_discussions": data.get("has_discussions"),
        "has_downloads": data.get("has_downloads"),
        "has_ci": has_ci,
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "pushed_at": data.get("pushed_at"),
        "latest_default_branch_commit_date": latest_commit_date,
        "archived": data.get("archived"),
        "disabled": data.get("disabled"),
        "fork": data.get("fork"),
        "license": (data.get("license") or {}).get("spdx_id"),
        "contributors_count": contributor_count,
        "test_file_count": test_file_count,
        "test_function_count": test_function_count,
        "clone_url": data.get("clone_url"),
    }


def passes_filters(row: dict) -> bool:
    """Apply the spec'd filter conditions to an enriched row.

      * not archived
      * contributors >= 2
      * test files >= 1

    (Star floor and language are enforced at search time by the query.)
    """
    if row.get("archived"):
        return False
    if (row.get("contributors_count") or 0) < MIN_CONTRIBUTORS:
        return False
    if (row.get("test_file_count") or 0) < MIN_TEST_FILES:
        return False
    return True


def apply_filters(rows: List[dict]) -> Tuple[List[dict], Dict[str, int]]:
    """Run the spec filters and record the funnel — how many repos are dropped
    at each step. Attribution is sequential (archived → contributors → tests):
    a repo is charged to the first condition it fails, so the counts form a
    clean funnel that sums to the input total.
    """
    stats = {
        "enriched": len(rows),
        "dropped_archived": 0,
        "dropped_contributors": 0,
        "dropped_no_tests": 0,
        "kept": 0,
    }
    kept: List[dict] = []
    for row in rows:
        if row.get("archived"):
            stats["dropped_archived"] += 1
        elif (row.get("contributors_count") or 0) < MIN_CONTRIBUTORS:
            stats["dropped_contributors"] += 1
        elif (row.get("test_file_count") or 0) < MIN_TEST_FILES:
            stats["dropped_no_tests"] += 1
        else:
            stats["kept"] += 1
            kept.append(row)
    return kept, stats


FIELDNAMES = [
    "full_name", "html_url", "description", "matched_query", "homepage",
    "owner_login", "owner_type",
    "stars", "forks", "watchers", "subscribers_count", "network_count",
    "language", "import_names", "topics", "open_issues", "size_kb",
    "default_branch", "visibility", "is_template", "allow_forking",
    "has_issues", "has_projects", "has_wiki", "has_pages", "has_discussions",
    "has_downloads", "has_ci",
    "created_at", "updated_at", "pushed_at", "latest_default_branch_commit_date",
    "archived", "disabled", "fork", "license", "contributors_count",
    "test_file_count", "test_function_count", "clone_url",
]


def write_csv(rows: List[dict], out_path: Path) -> None:
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["stars"] or 0), reverse=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="Enrich at most N unique repos (for quick test runs)")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_PER_QUERY,
                        help=f"Search result pages per query (default {MAX_PAGES_PER_QUERY})")
    parser.add_argument("--out", type=Path, default=paths.FRAMEWORKS_CSV,
                        help="Output CSV path (default: pipeline/artifacts/frameworks.csv)")
    args = parser.parse_args()

    paths.ensure_dirs()

    deduped: Dict[str, Tuple[dict, List[str]]] = {}
    for query in build_search_queries():
        repos = search_repositories(query, max_pages=args.max_pages)
        for item in repos:
            full_name = item["full_name"]
            if full_name not in deduped:
                deduped[full_name] = (item, [query])
            else:
                deduped[full_name][1].append(query)

    print(f"Unique repos before enrichment: {len(deduped)}")

    items = list(deduped.items())
    if args.limit is not None:
        items = items[:args.limit]
        print(f"--limit set: enriching only {len(items)} repos")

    enriched_rows = []
    for i, (full_name, (item, queries)) in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] Enriching {full_name}")
        try:
            row = enrich_repo(item, " | ".join(sorted(set(queries))))
            enriched_rows.append(row)
            time.sleep(0.2)
        except requests.HTTPError as e:
            print(f"Skipping {full_name} due to HTTP error: {e}")
        except Exception as e:
            print(f"Skipping {full_name} due to unexpected error: {e}")

    filtered, stats = apply_filters(enriched_rows)
    stats = {"unique_before_enrichment": len(deduped), **stats}

    write_csv(filtered, args.out)

    stats_path = args.out.with_name("frameworks_filter_stats.json") \
        if args.out != paths.FRAMEWORKS_CSV else paths.FRAMEWORKS_FILTER_STATS_JSON
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nFilter funnel:")
    print(f"  unique before enrichment : {stats['unique_before_enrichment']}")
    print(f"  enriched                 : {stats['enriched']}")
    print(f"  dropped (archived)       : {stats['dropped_archived']}")
    print(f"  dropped (contributors<{MIN_CONTRIBUTORS}) : {stats['dropped_contributors']}")
    print(f"  dropped (test files<{MIN_TEST_FILES})   : {stats['dropped_no_tests']}")
    print(f"  kept                     : {stats['kept']}")
    print(f"\nWrote {len(filtered)} rows to {args.out}")
    print(f"Wrote filter stats to {stats_path}")


if __name__ == "__main__":
    main()
