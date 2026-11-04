"""Semantic-evaluation dependency check (STANDALONE / OPTIONAL — not in the
pipeline orchestrator).

The pipeline's eval signal is the "called" view from the invoker-search
piggyback (EVAL_CALLS -> eval_calls_all.csv -> eval_frequency.csv). This script
is the coarser "declared" view: which Stage 2 application repos *list* a
semantic-evaluator (giskard / deepeval / opik / ragas / promptfoo / arize-phoenix)
in their root dependency files. Keep it around only as a cross-check (e.g.
"declares deepeval but no call detected" -> possible EVAL_CALLS pattern gap);
run it by hand when needed.

Downloads root-level dependency files directly from raw.githubusercontent.com.
Repos where no dependency file is found are flagged for manual review.

Outputs (under pipeline/artifacts/):
  semantic_evaluator_repos.csv
  no_deps_found.csv   (repos with no root-level dep files)
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

# Make the repo-root `pipeline` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths  # noqa: E402

load_dotenv()
load_dotenv(paths.REPO_ROOT / "Frameworks" / ".env")  # token (raises raw rate limit)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
RAW_HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Stage 2 applications are the subject of the eval check.
CANDIDATE_CSVS = [("Applications", paths.APPLICATIONS_CSV)]

OUT_CSV = paths.SEMANTIC_EVALUATOR_REPOS_CSV
NO_DEPS_CSV = paths.NO_DEPS_CSV
PROGRESS_FILE = paths.DEP_CHECK_PROGRESS_JSON

ROOT_DEP_FILES = [
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "requirements_dev.txt",
    "requirements_test.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
]

SEMANTIC_EVAL_PACKAGES = {
    "giskard": re.compile(r"(?:^|[\s,\"\'\[])giskard(?:\s|[>=<!\[,\]\"\']|$)", re.IGNORECASE | re.MULTILINE),
    "deepeval": re.compile(r"(?:^|[\s,\"\'\[])deepeval(?:\s|[>=<!\[,\]\"\']|$)", re.IGNORECASE | re.MULTILINE),
    "opik": re.compile(r"(?:^|[\s,\"\'\[])opik(?:\s|[>=<!\[,\]\"\']|$)", re.IGNORECASE | re.MULTILINE),
    "ragas": re.compile(r"(?:^|[\s,\"\'\[])ragas(?:\s|[>=<!\[,\]\"\']|$)", re.IGNORECASE | re.MULTILINE),
    "promptfoo": re.compile(r"(?:^|[\s,\"\'\[])promptfoo(?:\s|[>=<!\[,\]\"\']|$)", re.IGNORECASE | re.MULTILINE),
    "phoenix": re.compile(r"(?:^|[\s,\"\'\[])arize-phoenix(?:\s|[>=<!\[,\]\"\']|$)", re.IGNORECASE | re.MULTILINE),
}

RAW_BASE = "https://raw.githubusercontent.com"


def fetch_raw_file(full_name: str, branch: str, filename: str) -> Optional[str]:
    url = f"{RAW_BASE}/{full_name}/{branch}/{filename}"
    try:
        resp = requests.get(url, headers=RAW_HEADERS, timeout=15)
    except requests.exceptions.RequestException:
        return None
    if resp.status_code == 200:
        return resp.text
    return None


def check_repo(full_name: str, branch: str) -> tuple:
    """Returns (matches dict, list of dep files found).

    matches: {framework_name: [dep_file, ...]}
    """
    matches: Dict[str, List[str]] = {}
    found_files: List[str] = []

    for filename in ROOT_DEP_FILES:
        content = fetch_raw_file(full_name, branch, filename)
        if content is None:
            continue
        found_files.append(filename)
        for fw_name, pattern in SEMANTIC_EVAL_PACKAGES.items():
            if pattern.search(content):
                matches.setdefault(fw_name, []).append(filename)

    return matches, found_files


def load_candidates() -> List[dict]:
    repos = []
    for category, csv_path in CANDIDATE_CSVS:
        if not csv_path.exists():
            print(f"Skipping missing CSV: {csv_path}")
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                repos.append({
                    "category": category,
                    "full_name": row["full_name"],
                    "default_branch": row.get("default_branch", "main"),
                })
    return repos


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": {}}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="Check at most N repos (smoke runs)")
    args = parser.parse_args()

    paths.ensure_dirs()
    candidates = load_candidates()
    if args.limit is not None:
        candidates = candidates[:args.limit]
    print(f"Loaded {len(candidates)} candidate repos")

    progress = load_progress()
    processed = progress["processed"]

    hit_rows: List[dict] = []
    no_dep_rows: List[dict] = []

    for i, cand in enumerate(candidates, 1):
        full_name = cand["full_name"]
        category = cand["category"]
        branch = cand["default_branch"]

        if full_name in processed:
            cached = processed[full_name]
            if cached.get("no_deps"):
                no_dep_rows.append({"category": cached["category"], "full_name": full_name})
            elif cached["frameworks"]:
                hit_rows.append({
                    "category": cached["category"],
                    "full_name": full_name,
                    "frameworks": cached["frameworks"],
                    "dep_files": cached["dep_files"],
                })
            continue

        print(f"[{i}/{len(candidates)}] {full_name}...", end=" ", flush=True)
        matches, found_files = check_repo(full_name, branch)

        if not found_files:
            print("NO DEP FILES")
            no_dep_rows.append({"category": category, "full_name": full_name})
            processed[full_name] = {"category": category, "frameworks": "", "dep_files": "", "no_deps": True}
        elif matches:
            fw_list = ",".join(sorted(matches.keys()))
            dep_list = ",".join(sorted({p for paths in matches.values() for p in paths}))
            print(f"FOUND: {fw_list}")
            hit_rows.append({
                "category": category,
                "full_name": full_name,
                "frameworks": fw_list,
                "dep_files": dep_list,
            })
            processed[full_name] = {"category": category, "frameworks": fw_list, "dep_files": dep_list}
        else:
            print("none")
            processed[full_name] = {"category": category, "frameworks": "", "dep_files": ""}

        save_progress(progress)

    # Write hits
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "full_name", "frameworks", "dep_files"],
                                quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sorted(hit_rows, key=lambda r: (r["category"], r["full_name"])))

    # Write no-deps-found for manual review
    with open(NO_DEPS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "full_name"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sorted(no_dep_rows, key=lambda r: (r["category"], r["full_name"])))

    print(f"\n{len(hit_rows)} repos with semantic eval deps -> {OUT_CSV}")
    print(f"{len(no_dep_rows)} repos with no dep files found -> {NO_DEPS_CSV}")
    for r in sorted(hit_rows, key=lambda r: r["full_name"]):
        print(f"  {r['full_name']}: {r['frameworks']}")


if __name__ == "__main__":
    main()
