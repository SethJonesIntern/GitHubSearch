"""Repo metrics from a local git clone instead of the GitHub REST API.

The Stage-2 enrichment signals contributors / commit count / test-file count / CI
were each a REST call per repo, burning the 5000/hr core-API budget. This computes
them from a **blobless** clone (`--filter=blob:none --no-checkout`): git fetches the
full commit graph and tree objects but no file *contents*, so it is fast and small,
and `git rev-list` / `log` / `ls-tree` answer everything offline — no API, no rate
limit. Only stars / language / fork still need the REST repo-details call.
"""
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Optional

import git

TEST_FILE_RE = re.compile(r"(^|/)test_[^/]+\.py$")


def _rmtree(path: Path):
    """Windows-safe rmtree: git packs are read-only, so clear the bit and retry."""
    def onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass
    shutil.rmtree(path, onerror=onerror)


def measure_repo(clone_url: str) -> Optional[dict]:
    """Blobless-clone `clone_url` and return
    {contributors, total_commits, test_file_count, has_ci}, or None on failure.
    The clone is always removed before returning."""
    tmp = Path(tempfile.mkdtemp(prefix="ghmetrics_"))
    repo = None
    try:
        repo = git.Repo.clone_from(
            clone_url, tmp,
            multi_options=["--filter=blob:none", "--no-checkout", "--single-branch"],
        )
        g = repo.git
        total_commits = int(g.rev_list("--count", "HEAD"))
        authors = g.log("--format=%aE", "HEAD").splitlines()
        contributors = len({a.strip().lower() for a in authors if a.strip()})
        paths = g.ls_tree("-r", "--name-only", "HEAD").splitlines()
        test_file_count = sum(1 for p in paths if TEST_FILE_RE.search(p))
        has_ci = any(p.startswith(".github/workflows/") for p in paths)
        return {
            "contributors": contributors,
            "total_commits": total_commits,
            "test_file_count": test_file_count,
            "has_ci": has_ci,
        }
    except Exception:
        return None
    finally:
        if repo is not None:
            repo.close()
        _rmtree(tmp)
