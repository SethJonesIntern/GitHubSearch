import csv
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

API_BASE = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

RAW_HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Path to frameworks CSV so we can exclude framework repos from results
FRAMEWORKS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Frameworks", "github_agent_framework_candidates.csv"
)

# Python package names of the agent frameworks we want to find applications for
FRAMEWORK_SEARCH_TERMS = [
    "langchain",
    "langgraph",
    "autogen",
    "crewai",
    "pydantic-ai",
    "metagpt",
    "camel-ai",
    "agency-swarm",
    "griptape",
    "agentops",
    "openai-agents",
    "adalflow",
    "swarms",
    "parlant",
    "praisonai",
    "dynamiq",
    "openai-swarm",
    "superagi",
    "agent-zero",
    "ragaai-catalyst",
    "pentestgpt",
    "ten-framework",
    "livekit-agents",
    "agent-squad",
    "lavague",
    "superduper",
    "giskard",
    "ii-agent",
    "beeai-framework",
    "cheshire-cat",
    "solace-agent-mesh",
    "openlit",
    "nextpy",
    "llmstack",
    "lagent",
    "agentuniverse",
    "notte",
    "demogpt",
    "pentestagent",
    "redamon",
    "honcho",
    "uagents",
    "openakita",
    "patchwork",
    "agent-protocol",
    "npcpy",
    "infiagent",
    "any-agent",
    "sage",
]

OUTPUT_CSV = "github_agent_application_candidates.csv"
PER_PAGE = 100
MAX_PAGES_PER_QUERY = 3
MIN_STARS = 10
# Only consider repos pushed within the last year — filters out abandoned projects
# for free at search time (no API cost) and drastically shrinks the enrichment set.
PUSHED_AFTER = "2025-04-14"


def load_framework_repos() -> set:
    """Load framework repo full_names to exclude from application results."""
    exclusions = set()
    if os.path.exists(FRAMEWORKS_CSV):
        with open(FRAMEWORKS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exclusions.add(row["full_name"].lower())
    return exclusions


def github_get(url: str, params: Optional[dict] = None) -> requests.Response:
    while True:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = resp.headers.get("X-RateLimit-Reset")
            if reset:
                sleep_for = max(int(reset) - int(time.time()) + 2, 2)
                print(f"Rate limit hit. Sleeping {sleep_for}s...")
                time.sleep(sleep_for)
                continue
        resp.raise_for_status()
        return resp


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
        print(f"  Page {page}: got {len(items)} repos")
        if len(items) < PER_PAGE:
            break
    return results


def get_contributor_count(owner: str, repo: str) -> Optional[int]:
    # Single page is enough — we only filter on >= 2 contributors
    params = {"per_page": 100, "page": 1, "anon": "true"}
    resp = github_get(f"{API_BASE}/repos/{owner}/{repo}/contributors", params=params)
    data = resp.json()
    if not isinstance(data, list):
        return 0
    return len(data)


def get_total_commits(owner: str, repo: str, branch: str) -> int:
    """Get total commit count using the Link header pagination trick."""
    params = {"sha": branch, "per_page": 1}
    resp = github_get(f"{API_BASE}/repos/{owner}/{repo}/commits", params=params)
    link = resp.headers.get("Link", "")
    match = re.search(r'page=(\d+)>; rel="last"', link)
    if match:
        return int(match.group(1))
    data = resp.json()
    return len(data) if isinstance(data, list) else 0


def get_test_file_count(owner: str, repo: str, default_branch: str) -> int:
    """Count test files from the tree without downloading them."""
    resp = github_get(
        f"{API_BASE}/repos/{owner}/{repo}/git/trees/{default_branch}",
        params={"recursive": "1"},
    )
    data = resp.json()

    return sum(
        1 for item in data.get("tree", [])
        if item["type"] == "blob"
        and re.search(r"(^|/)test_[^/]+\.py$", item["path"])
    )


def enrich_repo(repo_item: dict, matched_frameworks: List[str]) -> Optional[dict]:
    """Staged enrichment — short-circuits when a filter fails, skipping
    remaining API calls. Returns None if the repo fails any filter."""
    full_name = repo_item["full_name"]
    owner, repo = full_name.split("/", 1)
    default_branch = repo_item.get("default_branch", "main")

    # Stage 0 (free): compute lifetime_days from search data using pushed_at
    # as latest-commit proxy. pushed_at tracks the last push to any branch,
    # close enough for filtering and saves 1 API call per repo.
    latest_commit_date = repo_item.get("pushed_at")
    lifetime_days = None
    if latest_commit_date and repo_item.get("created_at"):
        try:
            created = datetime.fromisoformat(repo_item["created_at"].replace("Z", "+00:00"))
            latest = datetime.fromisoformat(latest_commit_date.replace("Z", "+00:00"))
            lifetime_days = (latest - created).days
        except Exception:
            pass
    if (lifetime_days or 0) < 30:
        return None

    # Stage 1: total commits (1 API call). Filter on commits_per_month >= 2.
    total_commits = get_total_commits(owner, repo, default_branch)
    commits_per_month = None
    if lifetime_days and lifetime_days > 0:
        commits_per_month = round(total_commits / (lifetime_days / 30.44), 2)
    if (commits_per_month or 0) < 2:
        return None

    # Stage 2: contributors (1 API call). Filter on >= 2.
    contributor_count = get_contributor_count(owner, repo)
    if (contributor_count or 0) < 2:
        return None

    # Stage 3: test files (1 API call, potentially large response). Filter on >= 2.
    test_file_count = get_test_file_count(owner, repo, default_branch)
    if test_file_count < 2:
        return None

    return {
        "full_name": full_name,
        "html_url": repo_item.get("html_url"),
        "description": repo_item.get("description"),
        "matched_frameworks": ", ".join(sorted(set(matched_frameworks))),
        "stars": repo_item.get("stargazers_count"),
        "forks": repo_item.get("forks_count"),
        "language": repo_item.get("language"),
        "topics": ",".join(repo_item.get("topics", [])) if repo_item.get("topics") else "",
        "open_issues": repo_item.get("open_issues_count"),
        "size_kb": repo_item.get("size"),
        "default_branch": default_branch,
        "created_at": repo_item.get("created_at"),
        "updated_at": repo_item.get("updated_at"),
        "pushed_at": repo_item.get("pushed_at"),
        "latest_default_branch_commit_date": latest_commit_date,
        "fork": repo_item.get("fork"),
        "archived": repo_item.get("archived"),
        "disabled": repo_item.get("disabled"),
        "license": (repo_item.get("license") or {}).get("spdx_id"),
        "contributors_count": contributor_count,
        "total_commits": total_commits,
        "lifetime_days": lifetime_days,
        "commits_per_month": commits_per_month,
        "test_file_count": test_file_count,
        "test_function_count": None,
        "clone_url": repo_item.get("clone_url"),
    }


def main():
    framework_repos = load_framework_repos()
    print(f"Loaded {len(framework_repos)} framework repos to exclude")

    deduped: Dict[str, Tuple[dict, List[str]]] = {}

    for term in FRAMEWORK_SEARCH_TERMS:
        query = f'"{term}" language:Python stars:>{MIN_STARS} pushed:>{PUSHED_AFTER}'
        print(f"Searching: {query}")
        repos = search_repositories(query)
        for item in repos:
            full_name = item["full_name"]
            if full_name.lower() in framework_repos:
                continue
            if item.get("fork"):
                continue
            if full_name not in deduped:
                deduped[full_name] = (item, [term])
            else:
                deduped[full_name][1].append(term)
        time.sleep(1)

    # Early filter: skip repos that can't pass filters using search result data
    pre_filtered = {}
    for full_name, (item, frameworks) in deduped.items():
        if item.get("fork") or item.get("archived") or item.get("disabled"):
            continue
        pre_filtered[full_name] = (item, frameworks)

    print(f"\nUnique non-framework, non-fork repos: {len(deduped)}")
    print(f"After pre-filter (not fork/archived/disabled): {len(pre_filtered)}")

    fieldnames = [
        "full_name", "html_url", "description", "matched_frameworks",
        "stars", "forks", "language", "topics", "open_issues",
        "size_kb", "default_branch", "created_at", "updated_at", "pushed_at",
        "latest_default_branch_commit_date", "fork", "archived", "disabled",
        "license", "contributors_count", "total_commits", "lifetime_days",
        "commits_per_month", "test_file_count", "test_function_count", "clone_url",
    ]

    enriched_rows = []
    done = 0
    FLUSH_EVERY = 50

    def _enrich(item, frameworks):
        return enrich_repo(item, frameworks)

    def flush_csv(rows: List[dict]) -> None:
        """Rewrite OUTPUT_CSV with current rows, sorted by stars desc."""
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["stars"] or 0), reverse=True))

    # Write empty file with header upfront so it exists immediately.
    flush_csv([])

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {
            executor.submit(_enrich, item, fw): fn
            for fn, (item, fw) in pre_filtered.items()
        }
        for future in as_completed(futures):
            full_name = futures[future]
            done += 1
            try:
                row = future.result()
                if row is not None:
                    enriched_rows.append(row)
            except requests.HTTPError as e:
                print(f"  Skipping {full_name}: {e}")
            except Exception as e:
                print(f"  Skipping {full_name}: {e}")

            if done % FLUSH_EVERY == 0:
                flush_csv(enriched_rows)
                print(f"  Processed {done}/{len(pre_filtered)} ({len(enriched_rows)} kept) — CSV updated")

    # Final write to catch the tail end since last flush.
    flush_csv(enriched_rows)
    print(f"\nWrote {len(enriched_rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
