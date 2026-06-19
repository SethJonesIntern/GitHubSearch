"""Stage 2 — Application search.

Same GitHub-search method as the prior work by Mehedi Hasan: for each agent
framework we know an import-statement pattern; we search GitHub *code* for that
exact substring so a repo only matches if it genuinely imports the framework,
dedupe matches to repos, then apply quality filters.

Filter conditions (per spec):
  * Language: Python
  * Not forked / archived / disabled
  * Stars >= 10
  * Pushed after 2025-04-14 (recent activity)
  * Contributor count >= 2 (well-maintained)
  * Lifetime > ~1 month (created->pushed >= 30 days; not a one-off)
  * Commit frequency >= 2 commits/month
  * Test files >= 1

Outputs (under pipeline/artifacts/):
  applications.csv               kept candidates (the downstream work list)
  application_metadata.csv       rich metadata for every enriched repo + is_candidate
  applications_filter_stats.json the filter funnel (per-step drop counts)
  .search_progress.json          resumable progress
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Make the repo-root `pipeline` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths  # noqa: E402

load_dotenv()  # repo-root / CWD .env
load_dotenv(paths.REPO_ROOT / "Frameworks" / ".env")  # token lives here
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

API_BASE = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

OUTPUT_CSV = paths.APPLICATIONS_CSV
METADATA_CSV = paths.APPLICATION_METADATA_CSV
STATS_JSON = paths.APPLICATIONS_FILTER_STATS_JSON
PROGRESS_FILE = paths.SEARCH_PROGRESS_JSON
FRAMEWORKS_CSV = paths.FRAMEWORKS_CSV

PER_PAGE = 100
# GitHub code search caps at 1000 results and is rate limited to ~30 req/min,
# much stricter than the normal REST limit — hence the extra sleeps.
CODE_SEARCH_MAX_PAGES = 10
CODE_SEARCH_SLEEP_SECONDS = 2.5

# ── filter thresholds (per spec) ──────────────────────────────────────────────
MIN_STARS = 10
PUSHED_AFTER = "2025-04-14"
MIN_LIFETIME_DAYS = 30
MIN_CONTRIBUTORS = 2
MIN_COMMITS_PER_MONTH = 2   # "at least 2 commits a month" -> >= 2
MIN_TEST_FILES = 1

TEST_FILE_RE = re.compile(r"(^|/)test_[^/]+\.py$")
LAST_PAGE_RE = re.compile(r'[?&]page=(\d+)>;\s*rel="last"')


# ── framework list (Stage 1) loading + import-pattern derivation ──────────────
#
# Stage 2 derives what to search from Stage 1's output: each framework repo in
# frameworks.csv carries an `import_names` column (the importable top-level
# package names, read from the repo's __init__.py structure — see Stage 1
# derive_import_names). We turn each import name into code-search patterns, so
# there's no separate hand-maintained import-pattern list to drift.


def load_frameworks() -> List[dict]:
    """Stage 1 frameworks as [{full_name, import_names: [...]}, ...].
    Errors if the frameworks CSV is missing — Stage 1 must run first."""
    if not FRAMEWORKS_CSV.exists():
        raise FileNotFoundError(
            f"{FRAMEWORKS_CSV} not found — run Stage 1 (Frameworks/GithubSearch.py) first.")
    out = []
    with open(FRAMEWORKS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full_name = row.get("full_name")
            if not full_name:
                continue
            names = [n.strip() for n in (row.get("import_names") or "").split(";") if n.strip()]
            out.append({"full_name": full_name, "import_names": names})
    return out


def import_patterns(name: str) -> List[str]:
    """Code-search substrings for a single importable package name. Covers the
    `from X import ...`, `from X.sub import ...`, and `import X` usages."""
    return [f"from {name} import", f"from {name}.", f"import {name}"]


def build_import_index(frameworks: List[dict]) -> Dict[str, List[str]]:
    """Map each importable name -> the framework repo(s) that ship it. The name
    is Stage 2's search unit; the value records which Stage 1 framework(s) it
    belongs to (usually one, but a collision could map to several)."""
    index: Dict[str, set] = {}
    for fw in frameworks:
        for name in fw["import_names"]:
            index.setdefault(name, set()).add(fw["full_name"])
    return {name: sorted(fws) for name, fws in index.items()}


# ── HTTP ──────────────────────────────────────────────────────────────────────


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
    params = {"per_page": 1}
    if branch:
        params["sha"] = branch
    resp = github_get(f"{API_BASE}/repos/{owner}/{repo}/commits", params=params, allow_404=True)
    if resp is None:
        return 0
    last = _last_page_from_link(resp.headers.get("Link", ""))
    if last is not None:
        return last
    items = resp.json()
    return len(items) if isinstance(items, list) else 0


def tree_metrics(owner: str, repo: str, branch: str) -> Tuple[int, bool]:
    """From one tree fetch: number of test_*.py files and whether CI exists
    (a .github/workflows/ entry). (0, False) on failure."""
    resp = github_get(
        f"{API_BASE}/repos/{owner}/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
        allow_404=True,
    )
    if resp is None:
        return 0, False
    tree = resp.json().get("tree", [])
    test_file_count = sum(
        1 for item in tree
        if item.get("type") == "blob" and TEST_FILE_RE.search(item.get("path", ""))
    )
    has_ci = any(item.get("path", "").startswith(".github/workflows/") for item in tree)
    return test_file_count, has_ci


def search_code(query: str, max_pages: int = CODE_SEARCH_MAX_PAGES) -> List[dict]:
    results = []
    for page in range(1, max_pages + 1):
        params = {"q": query, "per_page": PER_PAGE, "page": page}
        resp = github_get(f"{API_BASE}/search/code", params=params)
        items = resp.json().get("items", [])
        if not items:
            break
        results.extend(items)
        print(f"  Page {page}: got {len(items)} code matches")
        if len(items) < PER_PAGE:
            break
        time.sleep(CODE_SEARCH_SLEEP_SECONDS)
    return results


def get_repo_details(owner: str, repo: str) -> Optional[dict]:
    resp = github_get(f"{API_BASE}/repos/{owner}/{repo}", allow_404=True)
    return resp.json() if resp is not None else None


def compute_lifetime_days(created_at: Optional[str], pushed_at: Optional[str]) -> Optional[int]:
    if not (created_at and pushed_at):
        return None
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
        return (pushed - created).days
    except Exception:
        return None


# ── candidate decision (pure, testable) ───────────────────────────────────────


def passes_search_filters(details: dict, pushed_after_dt: datetime) -> Tuple[bool, Optional[str]]:
    """Search-time filters applied to the full repo payload, before the costlier
    enrichment. Per spec: Python primary language, not fork/archived/disabled,
    stars >= MIN_STARS, pushed after the cutoff. Returns (ok, reason_if_dropped).

    Note: code search matched a Python *file*; this additionally requires the
    repo's *primary* language to be Python, which the code-search qualifier alone
    does not guarantee.
    """
    if details.get("fork") or details.get("archived") or details.get("disabled"):
        return False, "fork_archived_disabled"
    if (details.get("language") or "") != "Python":
        return False, "not_python"
    if (details.get("stargazers_count") or 0) < MIN_STARS:
        return False, "stars"
    pushed_at = details.get("pushed_at")
    try:
        pushed_dt = datetime.fromisoformat(pushed_at.replace("Z", "+00:00")) if pushed_at else None
    except Exception:
        pushed_dt = None
    if pushed_dt is None or pushed_dt <= pushed_after_dt:
        return False, "stale"
    return True, None


def evaluate_candidate(lifetime_days: Optional[int], contributors: Optional[int],
                       commits_per_month: Optional[float],
                       test_file_count: Optional[int]) -> Tuple[bool, Optional[str]]:
    """Apply the spec quality filters to computed signals. Sequential
    attribution (lifetime -> contributors -> commit_freq -> tests): returns the
    first condition that fails, or (True, None) when all pass."""
    if (lifetime_days or 0) < MIN_LIFETIME_DAYS:
        return False, "lifetime"
    if (contributors or 0) < MIN_CONTRIBUTORS:
        return False, "contributors"
    if (commits_per_month or 0) < MIN_COMMITS_PER_MONTH:
        return False, "commit_freq"
    if (test_file_count or 0) < MIN_TEST_FILES:
        return False, "no_tests"
    return True, None


def commits_per_month_of(total_commits: int, lifetime_days: Optional[int]) -> Optional[float]:
    if not lifetime_days:
        return None
    months = max(lifetime_days / 30.0, 1.0)
    return round(total_commits / months, 2)


# ── progress / IO ─────────────────────────────────────────────────────────────


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "completed_search_terms": [],
        "processed_repos": [],
        "candidates": {},
        "framework_repo_counts": {},
        "framework_file_matches": {},
        "stats": {"kept": 0, "dropped_lifetime": 0, "dropped_contributors": 0,
                  "dropped_commit_freq": 0, "dropped_no_tests": 0, "enriched": 0},
    }


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def load_existing_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


CANDIDATE_FIELDS = [
    "full_name", "html_url", "clone_url", "default_branch", "description",
    "matched_frameworks", "stars", "forks", "language", "topics", "open_issues",
    "size_kb", "created_at", "updated_at", "pushed_at", "license", "lifetime_days",
    "contributors", "total_commits", "commits_per_month",
]

METADATA_FIELDS = [
    "full_name", "html_url", "clone_url", "default_branch", "description", "homepage",
    "owner_login", "owner_type", "matched_frameworks",
    "stars", "forks", "watchers", "subscribers_count", "network_count",
    "open_issues", "size_kb", "language", "topics", "license",
    "visibility", "is_template", "allow_forking",
    "has_issues", "has_projects", "has_wiki", "has_pages", "has_discussions",
    "has_downloads", "has_ci",
    "fork", "archived", "disabled", "created_at", "updated_at", "pushed_at",
    "lifetime_days", "contributors", "total_commits", "commits_per_month",
    "test_file_count", "is_candidate", "drop_reason",
]


def append_row(path: Path, fieldnames: List[str], row: dict):
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def ensure_csv_header(path: Path, fieldnames: List[str]):
    """Create the CSV with just its header if it doesn't exist yet, so the file
    always exists for downstream stages even when zero rows are written."""
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL).writeheader()


def build_candidate_row(item: dict, frameworks: List[str], lifetime_days, contributors,
                        total_commits, commits_per_month) -> dict:
    return {
        "full_name": item.get("full_name"),
        "html_url": item.get("html_url"),
        "clone_url": item.get("clone_url"),
        "default_branch": item.get("default_branch") or "main",
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
        "commits_per_month": commits_per_month,
    }


def build_metadata_row(item: dict, frameworks: List[str], lifetime_days, contributors,
                       total_commits, commits_per_month, test_file_count, has_ci,
                       is_candidate: bool, drop_reason: Optional[str]) -> dict:
    owner = item.get("owner") or {}
    return {
        "full_name": item.get("full_name"),
        "html_url": item.get("html_url"),
        "clone_url": item.get("clone_url"),
        "default_branch": item.get("default_branch") or "main",
        "description": item.get("description"),
        "homepage": item.get("homepage"),
        "owner_login": owner.get("login"),
        "owner_type": owner.get("type"),
        "matched_frameworks": ", ".join(sorted(set(frameworks))),
        "stars": item.get("stargazers_count"),
        "forks": item.get("forks_count"),
        "watchers": item.get("watchers_count"),
        "subscribers_count": item.get("subscribers_count"),
        "network_count": item.get("network_count"),
        "open_issues": item.get("open_issues_count"),
        "size_kb": item.get("size"),
        "language": item.get("language"),
        "topics": ",".join(item.get("topics", [])) if item.get("topics") else "",
        "license": (item.get("license") or {}).get("spdx_id"),
        "visibility": item.get("visibility"),
        "is_template": item.get("is_template"),
        "allow_forking": item.get("allow_forking"),
        "has_issues": item.get("has_issues"),
        "has_projects": item.get("has_projects"),
        "has_wiki": item.get("has_wiki"),
        "has_pages": item.get("has_pages"),
        "has_discussions": item.get("has_discussions"),
        "has_downloads": item.get("has_downloads"),
        "has_ci": has_ci,
        "fork": item.get("fork"),
        "archived": item.get("archived"),
        "disabled": item.get("disabled"),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "pushed_at": item.get("pushed_at"),
        "lifetime_days": lifetime_days,
        "contributors": contributors,
        "total_commits": total_commits,
        "commits_per_month": commits_per_month,
        "test_file_count": test_file_count,
        "is_candidate": is_candidate,
        "drop_reason": drop_reason or "",
    }


def write_filter_stats(progress: dict, n_candidates: int):
    stats = progress["stats"]
    funnel = {
        "candidates_after_search_filters": n_candidates,
        "enriched": stats.get("enriched", 0),
        "dropped_lifetime": stats.get("dropped_lifetime", 0),
        "dropped_contributors": stats.get("dropped_contributors", 0),
        "dropped_commit_freq": stats.get("dropped_commit_freq", 0),
        "dropped_no_tests": stats.get("dropped_no_tests", 0),
        "kept": stats.get("kept", 0),
    }
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(funnel, f, indent=2)
    return funnel


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous progress instead of starting fresh")
    parser.add_argument("--max-terms", type=int, default=None,
                        help="Search only the first N import names (smoke runs)")
    parser.add_argument("--code-pages", type=int, default=None,
                        help="Cap code-search pages per pattern (smoke runs)")
    parser.add_argument("--max-repos", type=int, default=None,
                        help="Enrich at most N candidates this run (smoke runs)")
    args = parser.parse_args()

    code_pages = args.code_pages or CODE_SEARCH_MAX_PAGES

    paths.ensure_dirs()

    # Derive what to search from Stage 1's output: each framework's importable
    # package name(s) become the search units; their repos are also excluded.
    frameworks = load_frameworks()
    framework_repos = {fw["full_name"].lower() for fw in frameworks}
    import_index = build_import_index(frameworks)   # import name -> [framework full_names]
    search_names = sorted(import_index)
    if args.max_terms:
        search_names = search_names[:args.max_terms]
    print(f"Loaded {len(frameworks)} Stage 1 frameworks -> "
          f"{len(import_index)} importable names to search "
          f"(searching {len(search_names)})")

    if args.resume:
        progress = load_progress()
        completed_terms = set(progress["completed_search_terms"])
        processed_repos = set(progress["processed_repos"])
        processed_repos |= {row["full_name"] for row in load_existing_rows(METADATA_CSV)}
        if processed_repos:
            print(f"Resuming: {len(processed_repos)} repos already processed")
        if completed_terms:
            print(f"Resuming: {len(completed_terms)} import names already searched")
    else:
        for f in (PROGRESS_FILE, OUTPUT_CSV, METADATA_CSV, STATS_JSON):
            if f.exists():
                f.unlink()
        progress = load_progress()
        completed_terms = set()
        processed_repos = set()
        print("Starting fresh run")

    # Always create the output CSVs (header-only) up front, so downstream stages
    # find applications.csv even if zero candidates are kept this run.
    ensure_csv_header(OUTPUT_CSV, CANDIDATE_FIELDS)
    ensure_csv_header(METADATA_CSV, METADATA_FIELDS)

    # --- Phase 1: code search (resumable per search term) ---
    candidates: Dict[str, Tuple[dict, List[str]]] = {}
    for full_name, saved in progress["candidates"].items():
        candidates[full_name] = (saved["item"], saved["frameworks"])

    pushed_after_dt = datetime.fromisoformat(f"{PUSHED_AFTER}T00:00:00+00:00")

    for name in search_names:
        if name in completed_terms:
            continue

        name_repos: set = set()
        name_file_matches = 0
        for pattern in import_patterns(name):
            query = f'"{pattern}" language:Python'
            print(f"Searching code: {query}")
            code_items = search_code(query, max_pages=code_pages)
            print(f"  {len(code_items)} matching files")
            name_file_matches += len(code_items)

            for code_item in code_items:
                repo_stub = code_item.get("repository") or {}
                full_name = repo_stub.get("full_name")
                if not full_name:
                    continue
                if full_name.lower() not in framework_repos:
                    name_repos.add(full_name)
                if full_name.lower() in framework_repos:
                    continue
                if full_name in candidates:
                    if name not in candidates[full_name][1]:
                        candidates[full_name][1].append(name)
                    continue

                owner, repo = full_name.split("/", 1)
                details = get_repo_details(owner, repo)
                if details is None:
                    continue
                ok, _reason = passes_search_filters(details, pushed_after_dt)
                if not ok:
                    continue

                candidates[full_name] = (details, [name])
            time.sleep(CODE_SEARCH_SLEEP_SECONDS)

        # Popularity signal per import name: distinct non-framework repos importing it.
        progress.setdefault("framework_repo_counts", {})[name] = len(name_repos)
        progress.setdefault("framework_file_matches", {})[name] = name_file_matches
        progress["completed_search_terms"].append(name)
        progress["candidates"] = {
            fn: {"item": item, "frameworks": fws} for fn, (item, fws) in candidates.items()
        }
        save_progress(progress)
        print(f"  Progress saved ({len(progress['completed_search_terms'])}/{len(search_names)} names)")

    print(f"\nUnique non-framework candidates: {len(candidates)}")

    # --- Phase 2: enrichment (resumable per repo) ---
    # Compute all signals for every enriched repo (no short-circuit) so the
    # metadata file is complete; the candidate decision is then a pure call.
    reason_to_stat = {
        "lifetime": "dropped_lifetime",
        "contributors": "dropped_contributors",
        "commit_freq": "dropped_commit_freq",
        "no_tests": "dropped_no_tests",
    }
    new_candidates = 0
    enriched_this_run = 0
    total = len(candidates)

    for idx, (full_name, (item, frameworks)) in enumerate(candidates.items(), 1):
        if full_name in processed_repos:
            continue
        if args.max_repos is not None and enriched_this_run >= args.max_repos:
            print(f"  --max-repos {args.max_repos} reached; stopping enrichment")
            break
        enriched_this_run += 1
        owner, repo = full_name.split("/", 1)
        branch = item.get("default_branch") or "main"

        lifetime_days = compute_lifetime_days(item.get("created_at"), item.get("pushed_at"))
        contributors = count_contributors(owner, repo)
        total_commits = count_commits(owner, repo, branch)
        commits_per_month = commits_per_month_of(total_commits, lifetime_days)
        test_file_count, has_ci = tree_metrics(owner, repo, branch)

        is_candidate, drop_reason = evaluate_candidate(
            lifetime_days, contributors, commits_per_month, test_file_count)

        # Always write the rich metadata row.
        append_row(METADATA_CSV, METADATA_FIELDS, build_metadata_row(
            item, frameworks, lifetime_days, contributors, total_commits,
            commits_per_month, test_file_count, has_ci, is_candidate, drop_reason))

        progress["stats"]["enriched"] += 1
        if is_candidate:
            append_row(OUTPUT_CSV, CANDIDATE_FIELDS, build_candidate_row(
                item, frameworks, lifetime_days, contributors, total_commits, commits_per_month))
            progress["stats"]["kept"] += 1
            new_candidates += 1
            print(f"  [{idx}/{total}] {full_name}: KEEP "
                  f"(contribs={contributors}, commits/mo={commits_per_month})")
        else:
            progress["stats"][reason_to_stat[drop_reason]] += 1
            print(f"  [{idx}/{total}] {full_name}: drop ({drop_reason})")

        progress["processed_repos"].append(full_name)
        save_progress(progress)

    funnel = write_filter_stats(progress, len(candidates))
    kept_total = len(load_existing_rows(OUTPUT_CSV))
    print("\nFilter funnel:")
    for k, v in funnel.items():
        print(f"  {k:<32}: {v}")
    print(f"\nNew candidates this run: {new_candidates}. Total in {OUTPUT_CSV.name}: {kept_total}.")
    print(f"Metadata rows in {METADATA_CSV.name}: {len(load_existing_rows(METADATA_CSV))}")
    print(f"Wrote filter stats to {STATS_JSON}")


if __name__ == "__main__":
    main()
