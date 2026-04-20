"""Phase 1: search GitHub for candidate application repos and write a list.

No expensive enrichment — only the search API + cheap filters we can do from
the search response alone. Output feeds analyze_tests.py.
"""
import argparse
import csv
import json
import os
import re
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

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "application_candidates_v2.csv")
PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".search_progress.json")
PER_PAGE = 100
MAX_PAGES_PER_QUERY = 3
MIN_STARS = 10
PUSHED_AFTER = "2025-04-14"
MIN_LIFETIME_DAYS = 30
MIN_CONTRIBUTORS = 2
MIN_COMMITS_PER_MONTH = 2  # strictly greater than this

TEST_FILE_RE = re.compile(r"(^|/)test_[^/]+\.py$")
LAST_PAGE_RE = re.compile(r'[?&]page=(\d+)>;\s*rel="last"')


def load_framework_repos() -> set:
    exclusions = set()
    if os.path.exists(FRAMEWORKS_CSV):
        with open(FRAMEWORKS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exclusions.add(row["full_name"].lower())
    return exclusions


def github_get(url: str, params: Optional[dict] = None, allow_404: bool = False,
               max_retries: int = 3) -> Optional[requests.Response]:
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


def count_contributors(owner: str, repo: str) -> int:
    """Return the number of contributors (anonymous included). Cheap: one call."""
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
    items = resp.json()
    return len(items) if isinstance(items, list) else 0


def count_commits(owner: str, repo: str, branch: Optional[str] = None) -> int:
    """Return the total commit count on the default branch."""
    params = {"per_page": 1}
    if branch:
        params["sha"] = branch
    resp = github_get(
        f"{API_BASE}/repos/{owner}/{repo}/commits",
        params=params,
        allow_404=True,
    )
    if resp is None:
        return 0
    last = _last_page_from_link(resp.headers.get("Link", ""))
    if last is not None:
        return last
    items = resp.json()
    return len(items) if isinstance(items, list) else 0


def has_test_file(owner: str, repo: str, branch: str) -> bool:
    resp = github_get(
        f"{API_BASE}/repos/{owner}/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
        allow_404=True,
    )
    if resp is None:
        return False
    tree = resp.json().get("tree", [])
    return any(
        item.get("type") == "blob" and TEST_FILE_RE.search(item.get("path", ""))
        for item in tree
    )


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


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "completed_search_terms": [],
        "processed_repos": [],
        "candidates": {},
        "stats": {"kept": 0, "dropped_lifetime": 0, "dropped_contributors": 0,
                  "dropped_commit_freq": 0, "dropped_no_tests": 0},
    }


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def load_existing_rows() -> List[dict]:
    if not os.path.exists(OUTPUT_CSV):
        return []
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


FIELDNAMES = [
    "full_name", "html_url", "clone_url", "default_branch", "description",
    "matched_frameworks", "stars", "forks", "language", "topics", "open_issues",
    "size_kb", "created_at", "updated_at", "pushed_at", "license", "lifetime_days",
    "contributors", "total_commits", "commits_per_month",
]


def append_row_to_csv(row: dict):
    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous progress instead of starting fresh")
    args = parser.parse_args()

    framework_repos = load_framework_repos()
    print(f"Loaded {len(framework_repos)} framework repos to exclude")

    if args.resume:
        progress = load_progress()
        completed_terms = set(progress["completed_search_terms"])
        processed_repos = set(progress["processed_repos"])
        existing_rows = load_existing_rows()
        processed_repos |= {row["full_name"] for row in existing_rows}
        if processed_repos:
            print(f"Resuming: {len(processed_repos)} repos already processed, skipping them")
        if completed_terms:
            print(f"Resuming: {len(completed_terms)}/{len(FRAMEWORK_SEARCH_TERMS)} search terms already completed")
    else:
        # Fresh start — wipe old progress and CSV
        for f in (PROGRESS_FILE, OUTPUT_CSV):
            if os.path.exists(f):
                os.remove(f)
        progress = load_progress()
        completed_terms = set()
        processed_repos = set()
        print("Starting fresh run")

    # --- Phase 1: Search (resumable per search term) ---
    candidates: Dict[str, Tuple[dict, List[str]]] = {}

    # Reload candidates saved from prior runs (only relevant with --resume)
    for full_name, saved in progress["candidates"].items():
        candidates[full_name] = (saved["item"], saved["frameworks"])

    for term in FRAMEWORK_SEARCH_TERMS:
        if term in completed_terms:
            continue
        query = f'"{term}" language:Python stars:>{MIN_STARS} pushed:>{PUSHED_AFTER}'
        print(f"Searching: {query}")
        repos = search_repositories(query)
        for item in repos:
            full_name = item["full_name"]
            if full_name.lower() in framework_repos:
                continue
            if item.get("fork") or item.get("archived") or item.get("disabled"):
                continue
            if full_name not in candidates:
                candidates[full_name] = (item, [term])
            else:
                candidates[full_name][1].append(term)

        # Mark this search term as done and persist candidates so far
        progress["completed_search_terms"].append(term)
        progress["candidates"] = {
            fn: {"item": item, "frameworks": fws}
            for fn, (item, fws) in candidates.items()
        }
        save_progress(progress)
        print(f"  Progress saved ({len(progress['completed_search_terms'])}/{len(FRAMEWORK_SEARCH_TERMS)} terms)")
        time.sleep(1)

    print(f"\nUnique non-framework candidates: {len(candidates)}")

    # --- Phase 2: Enrichment (resumable per repo) ---
    new_rows = 0
    dropped = {"lifetime": 0, "contributors": 0, "commit_freq": 0, "no_tests": 0, "skipped": 0}
    total = len(candidates)

    for idx, (full_name, (item, frameworks)) in enumerate(candidates.items(), 1):
        if full_name in processed_repos:
            dropped["skipped"] += 1
            continue

        owner, repo = full_name.split("/", 1)
        branch = item.get("default_branch") or "main"

        lifetime_days = compute_lifetime_days(item.get("created_at"), item.get("pushed_at"))
        if (lifetime_days or 0) < MIN_LIFETIME_DAYS:
            dropped["lifetime"] += 1
            progress["processed_repos"].append(full_name)
            progress["stats"]["dropped_lifetime"] += 1
            save_progress(progress)
            continue

        contributors = count_contributors(owner, repo)
        if contributors < MIN_CONTRIBUTORS:
            dropped["contributors"] += 1
            print(f"  [{idx}/{total}] {full_name}: drop (contributors={contributors})")
            progress["processed_repos"].append(full_name)
            progress["stats"]["dropped_contributors"] += 1
            save_progress(progress)
            continue

        total_commits = count_commits(owner, repo, branch)
        months = max(lifetime_days / 30.0, 1.0)
        commits_per_month = total_commits / months
        if commits_per_month <= MIN_COMMITS_PER_MONTH:
            dropped["commit_freq"] += 1
            print(f"  [{idx}/{total}] {full_name}: drop (commits/mo={commits_per_month:.2f})")
            progress["processed_repos"].append(full_name)
            progress["stats"]["dropped_commit_freq"] += 1
            save_progress(progress)
            continue

        if not has_test_file(owner, repo, branch):
            dropped["no_tests"] += 1
            print(f"  [{idx}/{total}] {full_name}: drop (no test files)")
            progress["processed_repos"].append(full_name)
            progress["stats"]["dropped_no_tests"] += 1
            save_progress(progress)
            continue

        print(f"  [{idx}/{total}] {full_name}: KEEP (contribs={contributors}, "
              f"commits/mo={commits_per_month:.2f})")
        row = {
            "full_name": full_name,
            "html_url": item.get("html_url"),
            "clone_url": item.get("clone_url"),
            "default_branch": branch,
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
            "contributors": contributors,
            "total_commits": total_commits,
            "commits_per_month": round(commits_per_month, 2),
        }
        append_row_to_csv(row)
        progress["processed_repos"].append(full_name)
        progress["stats"]["kept"] += 1
        save_progress(progress)
        new_rows += 1

    existing_count = len(load_existing_rows())
    print(f"\nNew rows added: {new_rows}. Total in CSV: {existing_count}.")
    print(f"Skipped (already processed): {dropped['skipped']}")
    print(f"Dropped this run: "
          f"lifetime={dropped['lifetime']}, "
          f"contributors={dropped['contributors']}, "
          f"commit_freq={dropped['commit_freq']}, "
          f"no_tests={dropped['no_tests']}")
    print(f"\nCumulative stats from progress file: {json.dumps(progress['stats'])}")
    print(f"\nResults in {OUTPUT_CSV}")
    print(f"To start fresh, delete {PROGRESS_FILE} and {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
