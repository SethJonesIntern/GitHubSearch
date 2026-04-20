"""For repos listed in no_deps_found.csv, use the GitHub API tree endpoint
to find dependency files anywhere in the repo, then download and check
them for semantic eval frameworks.

Updates .dep_check_progress.json and rewrites both output CSVs.
"""
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).parent
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

API_BASE = "https://api.github.com"
API_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    API_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

PROGRESS_FILE = HERE / ".dep_check_progress.json"
NO_DEPS_CSV = HERE / "no_deps_found.csv"
OUT_CSV = HERE / "semantic_evaluator_repos.csv"
RAW_BASE = "https://raw.githubusercontent.com"

CANDIDATE_CSVS = [
    ("Applications", HERE.parent / "Applications" / "application_candidates_v2.csv"),
    ("Frameworks", HERE.parent / "Frameworks" / "github_agent_framework_candidates.csv"),
]

DEP_FILE_RE = re.compile(
    r"(^|/)(requirements[^/]*\.txt|pyproject\.toml|setup\.py|setup\.cfg)$",
    re.IGNORECASE,
)

SEMANTIC_EVAL_PACKAGES = {
    "giskard": re.compile(r"(?:^|[\s,\"\'\[])giskard(?:\s|[>=<!\[,\]\"\']|$)", re.IGNORECASE | re.MULTILINE),
    "deepeval": re.compile(r"(?:^|[\s,\"\'\[])deepeval(?:\s|[>=<!\[,\]\"\']|$)", re.IGNORECASE | re.MULTILINE),
    "opik": re.compile(r"(?:^|[\s,\"\'\[])opik(?:\s|[>=<!\[,\]\"\']|$)", re.IGNORECASE | re.MULTILINE),
    "ragas": re.compile(r"(?:^|[\s,\"\'\[])ragas(?:\s|[>=<!\[,\]\"\']|$)", re.IGNORECASE | re.MULTILINE),
    "promptfoo": re.compile(r"(?:^|[\s,\"\'\[])promptfoo(?:\s|[>=<!\[,\]\"\']|$)", re.IGNORECASE | re.MULTILINE),
    "phoenix": re.compile(r"(?:^|[\s,\"\'\[])arize-phoenix(?:\s|[>=<!\[,\]\"\']|$)", re.IGNORECASE | re.MULTILINE),
}


def github_api_get(url: str, params: Optional[dict] = None,
                   max_retries: int = 3) -> Optional[requests.Response]:
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=API_HEADERS, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt * 5)
                continue
            raise
        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = resp.headers.get("X-RateLimit-Reset")
            if reset:
                sleep_for = max(int(reset) - int(time.time()) + 2, 2)
                print(f"  Rate limit hit. Sleeping {sleep_for}s...")
                time.sleep(sleep_for)
                continue
        if resp.status_code in (404, 409, 451):
            return None
        if resp.status_code >= 500 and attempt < max_retries:
            time.sleep(2 ** attempt * 5)
            continue
        resp.raise_for_status()
        return resp
    return None


def fetch_raw_file(full_name: str, branch: str, filepath: str) -> Optional[str]:
    url = f"{RAW_BASE}/{full_name}/{branch}/{filepath}"
    try:
        resp = requests.get(url, timeout=15)
    except requests.exceptions.RequestException:
        return None
    if resp.status_code == 200:
        return resp.text
    return None


def load_branch_lookup() -> Dict[str, str]:
    lookup = {}
    for _, csv_path in CANDIDATE_CSVS:
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lookup[row["full_name"]] = row.get("default_branch", "main")
    return lookup


def main() -> None:
    if not NO_DEPS_CSV.exists():
        print(f"No {NO_DEPS_CSV} found. Run find_semantic_eval_tests.py first.")
        return

    with open(NO_DEPS_CSV, "r", encoding="utf-8") as f:
        no_dep_repos = [(row["full_name"], row["category"]) for row in csv.DictReader(f)]

    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        progress = json.load(f)
    processed = progress["processed"]

    branches = load_branch_lookup()

    # Skip already deep-checked repos
    to_check = [(name, cat) for name, cat in no_dep_repos
                if not processed.get(name, {}).get("deep_checked")]
    print(f"{len(to_check)} repos to deep-check ({len(no_dep_repos) - len(to_check)} already done)")

    for i, (full_name, category) in enumerate(to_check, 1):
        branch = branches.get(full_name, "main")
        owner, repo = full_name.split("/", 1)

        print(f"[{i}/{len(to_check)}] {full_name}...", end=" ", flush=True)

        resp = github_api_get(
            f"{API_BASE}/repos/{owner}/{repo}/git/trees/{branch}",
            params={"recursive": "1"},
        )
        if resp is None:
            print("tree fetch failed")
            processed[full_name]["deep_checked"] = True
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f, indent=2)
            continue

        tree = resp.json().get("tree", [])
        dep_paths = [item["path"] for item in tree
                     if item.get("type") == "blob" and DEP_FILE_RE.search(item["path"])]

        if not dep_paths:
            print("no dep files anywhere")
            processed[full_name]["deep_checked"] = True
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f, indent=2)
            continue

        matches: Dict[str, List[str]] = {}
        for dep_path in dep_paths:
            content = fetch_raw_file(full_name, branch, dep_path)
            if content is None:
                continue
            for fw, pat in SEMANTIC_EVAL_PACKAGES.items():
                if pat.search(content):
                    matches.setdefault(fw, []).append(dep_path)

        if matches:
            fw_list = ",".join(sorted(matches.keys()))
            dep_list = ",".join(sorted({p for paths in matches.values() for p in paths}))
            print(f"FOUND: {fw_list}")
            processed[full_name] = {
                "category": category,
                "frameworks": fw_list,
                "dep_files": dep_list,
                "deep_checked": True,
            }
        else:
            print(f"{len(dep_paths)} dep files, no frameworks")
            processed[full_name] = {
                "category": category,
                "frameworks": "",
                "dep_files": ",".join(dep_paths),
                "deep_checked": True,
            }

        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    # Rewrite both output CSVs
    hit_rows = []
    still_no_deps = []
    for name, info in processed.items():
        if info["frameworks"]:
            hit_rows.append({
                "category": info["category"],
                "full_name": name,
                "frameworks": info["frameworks"],
                "dep_files": info.get("dep_files", ""),
            })
        elif info.get("no_deps") and not info.get("dep_files"):
            still_no_deps.append({"category": info["category"], "full_name": name})

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "full_name", "frameworks", "dep_files"],
                                quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sorted(hit_rows, key=lambda r: (r["category"], r["full_name"])))

    with open(NO_DEPS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "full_name"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sorted(still_no_deps, key=lambda r: (r["category"], r["full_name"])))

    print(f"\n{len(hit_rows)} repos with semantic eval deps -> {OUT_CSV}")
    print(f"{len(still_no_deps)} repos with no dep files anywhere -> {NO_DEPS_CSV}")


if __name__ == "__main__":
    main()
