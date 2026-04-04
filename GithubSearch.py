import os
import re
import time
import csv
import requests
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

API_BASE = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

SEARCH_QUERIES = [
    "AI agent framework stars:>999 language:Python",
    "LLM-based agent framework stars:>999 language:Python",
    "LLM agent library stars:>999 language:Python",
    "multi-agent orchestration framework stars:>999 language:Python",
    "LLM powered agents framework stars:>999 language:Python",
]

OUTPUT_CSV = "github_agent_framework_candidates.csv"
PER_PAGE = 100
MAX_PAGES_PER_QUERY = 3


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


def search_repositories(query: str, max_pages: int = 3) -> List[dict]:
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


def get_contributor_count(owner: str, repo: str) -> Optional[int]:
    count = 0
    page = 1
    while True:
        params = {"per_page": 100, "page": page, "anon": "true"}
        resp = github_get(f"{API_BASE}/repos/{owner}/{repo}/contributors", params=params)
        data = resp.json()
        if not isinstance(data, list) or not data:
            break
        count += len(data)
        if len(data) < 100:
            break
        page += 1
    return count


def get_default_branch_commit_date(owner: str, repo: str, default_branch: str) -> Optional[str]:
    params = {"sha": default_branch, "per_page": 1, "page": 1}
    resp = github_get(f"{API_BASE}/repos/{owner}/{repo}/commits", params=params)
    data = resp.json()
    if not data:
        return None
    try:
        return data[0]["commit"]["committer"]["date"]
    except Exception:
        return None


def get_test_file_count(owner: str, repo: str, default_branch: str) -> int:
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


def enrich_repo(repo_item: dict, matched_query: str) -> dict:
    full_name = repo_item["full_name"]
    owner, repo = full_name.split("/", 1)
    default_branch = repo_item.get("default_branch", "main")

    contributor_count = get_contributor_count(owner, repo)
    latest_commit_date = get_default_branch_commit_date(owner, repo, default_branch)
    test_file_count = get_test_file_count(owner, repo, default_branch)

    return {
        "full_name": full_name,
        "html_url": repo_item.get("html_url"),
        "description": repo_item.get("description"),
        "matched_query": matched_query,
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
        "archived": repo_item.get("archived"),
        "disabled": repo_item.get("disabled"),
        "license": (repo_item.get("license") or {}).get("spdx_id"),
        "contributors_count": contributor_count,
        "test_file_count": test_file_count,
        "clone_url": repo_item.get("clone_url"),
    }


def main():
    deduped: Dict[str, Tuple[dict, List[str]]] = {}

    for query in SEARCH_QUERIES:
        repos = search_repositories(query, max_pages=MAX_PAGES_PER_QUERY)
        for item in repos:
            full_name = item["full_name"]
            if full_name not in deduped:
                deduped[full_name] = (item, [query])
            else:
                deduped[full_name][1].append(query)

    print(f"Unique repos before enrichment: {len(deduped)}")

    enriched_rows = []
    for i, (full_name, (item, queries)) in enumerate(deduped.items(), start=1):
        print(f"[{i}/{len(deduped)}] Enriching {full_name}")
        try:
            row = enrich_repo(item, " | ".join(sorted(set(queries))))
            enriched_rows.append(row)
            time.sleep(0.2)
        except requests.HTTPError as e:
            print(f"Skipping {full_name} due to HTTP error: {e}")
        except Exception as e:
            print(f"Skipping {full_name} due to unexpected error: {e}")

    filtered = [
        r for r in enriched_rows
        if not r["archived"]
        and not r["disabled"]
        and (r["contributors_count"] is None or r["contributors_count"] >= 2)
        and (r["test_file_count"] >= 1)
    ]

    fieldnames = [
        "full_name", "html_url", "description", "matched_query",
        "stars", "forks", "language", "topics", "open_issues",
        "size_kb", "default_branch", "created_at", "updated_at", "pushed_at",
        "latest_default_branch_commit_date", "archived", "disabled",
        "license", "contributors_count", "test_file_count", "clone_url",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(filtered, key=lambda r: (r["stars"] or 0), reverse=True))

    print(f"Wrote {len(filtered)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
