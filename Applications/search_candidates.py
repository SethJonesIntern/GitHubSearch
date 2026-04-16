"""Phase 1: search GitHub for candidate application repos and write a list.

No expensive enrichment — only the search API + cheap filters we can do from
the search response alone. Output feeds analyze_tests.py.
"""
import csv
import os
import time
from datetime import datetime
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

FRAMEWORKS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Frameworks", "github_agent_framework_candidates.csv"
)

FRAMEWORK_SEARCH_TERMS = [
    "langchain", "langgraph", "autogen", "crewai", "pydantic-ai", "metagpt",
    "camel-ai", "agency-swarm", "griptape", "agentops", "openai-agents",
    "adalflow", "swarms", "parlant", "praisonai", "dynamiq", "openai-swarm",
    "superagi", "agent-zero", "ragaai-catalyst", "pentestgpt", "ten-framework",
    "livekit-agents", "agent-squad", "lavague", "superduper", "giskard",
    "ii-agent", "beeai-framework", "cheshire-cat", "solace-agent-mesh",
    "openlit", "nextpy", "llmstack", "lagent", "agentuniverse", "notte",
    "demogpt", "pentestagent", "redamon", "honcho", "uagents", "openakita",
    "patchwork", "agent-protocol", "npcpy", "infiagent", "any-agent", "sage",
]

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "application_candidates.csv")
PER_PAGE = 100
MAX_PAGES_PER_QUERY = 3
MIN_STARS = 10
PUSHED_AFTER = "2025-04-14"
MIN_LIFETIME_DAYS = 30


def load_framework_repos() -> set:
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
        items = resp.json().get("items", [])
        if not items:
            break
        results.extend(items)
        print(f"  Page {page}: got {len(items)} repos")
        if len(items) < PER_PAGE:
            break
    return results


def compute_lifetime_days(created_at: Optional[str], pushed_at: Optional[str]) -> Optional[int]:
    if not (created_at and pushed_at):
        return None
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
        return (pushed - created).days
    except Exception:
        return None


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
            if item.get("fork") or item.get("archived") or item.get("disabled"):
                continue
            if full_name not in deduped:
                deduped[full_name] = (item, [term])
            else:
                deduped[full_name][1].append(term)
        time.sleep(1)

    print(f"\nUnique non-framework candidates: {len(deduped)}")

    rows = []
    for full_name, (item, frameworks) in deduped.items():
        lifetime_days = compute_lifetime_days(item.get("created_at"), item.get("pushed_at"))
        if (lifetime_days or 0) < MIN_LIFETIME_DAYS:
            continue
        rows.append({
            "full_name": full_name,
            "html_url": item.get("html_url"),
            "clone_url": item.get("clone_url"),
            "default_branch": item.get("default_branch", "main"),
            "description": item.get("description"),
            "matched_frameworks": ", ".join(sorted(set(frameworks))),
            "stars": item.get("stargazers_count"),
            "forks": item.get("forks_count"),
            "language": item.get("language"),
            "topics": ",".join(item.get("topics", [])) if item.get("topics") else "",
            "open_issues": item.get("open_issues_count"),
            "size_kb": item.get("size"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "pushed_at": item.get("pushed_at"),
            "license": (item.get("license") or {}).get("spdx_id"),
            "lifetime_days": lifetime_days,
        })

    rows.sort(key=lambda r: (r["stars"] or 0), reverse=True)

    fieldnames = list(rows[0].keys()) if rows else [
        "full_name", "html_url", "clone_url", "default_branch", "description",
        "matched_frameworks", "stars", "forks", "language", "topics", "open_issues",
        "size_kb", "created_at", "updated_at", "pushed_at", "license", "lifetime_days",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} candidates to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
