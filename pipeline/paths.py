"""Canonical filesystem paths for the whole pipeline.

Every stage script — both the new drivers in this package and the existing
scripts in Frameworks/, Applications/, Wrapper/, SemanticEvaluators/ — should
import its input/output locations from here so there is exactly one place that
knows where artifacts live. Repointing in Stage 0 is then a matter of replacing
each script's local path constants with the names below.

Paths resolve from this file's location, so they are stable no matter what the
current working directory is when a script runs.
"""
from __future__ import annotations

from pathlib import Path

# ── base directories ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
ARTIFACTS_DIR = PIPELINE_DIR / "artifacts"   # all generated CSV/metadata
REPOS_DIR = PIPELINE_DIR / "repos"           # cloned application checkouts

# ── Stage 1: framework search ───────────────────────────────────────────────────
FRAMEWORKS_CSV = ARTIFACTS_DIR / "frameworks.csv"
FRAMEWORKS_FILTER_STATS_JSON = ARTIFACTS_DIR / "frameworks_filter_stats.json"

# ── Stage 2: application search ─────────────────────────────────────────────────
APPLICATIONS_CSV = ARTIFACTS_DIR / "applications.csv"
APPLICATION_METADATA_CSV = ARTIFACTS_DIR / "application_metadata.csv"
APPLICATIONS_FILTER_STATS_JSON = ARTIFACTS_DIR / "applications_filter_stats.json"
SEARCH_PROGRESS_JSON = ARTIFACTS_DIR / ".search_progress.json"

# ── Stage 3: framework frequency ────────────────────────────────────────────────
FRAMEWORK_FREQUENCY_CSV = ARTIFACTS_DIR / "framework_frequency.csv"

# ── Stage 5: batch invoker + LLM-call extraction ────────────────────────────────
LLM_INVOKERS_CSV = ARTIFACTS_DIR / "llm_invokers_all.csv"   # all direct+transitive invokers
LLM_CALLS_CSV = ARTIFACTS_DIR / "llm_calls_all.csv"
CALL_METADATA_CSV = ARTIFACTS_DIR / "call_metadata_all.csv"
LLM_TESTS_CSV = ARTIFACTS_DIR / "llm_tests_all.csv"
BATCH_PROGRESS_JSON = ARTIFACTS_DIR / ".batch_progress.json"

# ── Stage 7: semantic evaluation ────────────────────────────────────────────────
EVAL_INVOKERS_CSV = ARTIFACTS_DIR / "eval_invokers_all.csv"   # all direct+transitive eval invokers
EVAL_CALLS_CSV = ARTIFACTS_DIR / "eval_calls_all.csv"
EVAL_CALL_METADATA_CSV = ARTIFACTS_DIR / "eval_call_metadata_all.csv"  # → JOERN
EVAL_FREQUENCY_CSV = ARTIFACTS_DIR / "eval_frequency.csv"
SEMANTIC_EVALUATOR_REPOS_CSV = ARTIFACTS_DIR / "semantic_evaluator_repos.csv"
NO_DEPS_CSV = ARTIFACTS_DIR / "no_deps_found.csv"
DEP_CHECK_PROGRESS_JSON = ARTIFACTS_DIR / ".dep_check_progress.json"


def ensure_dirs() -> None:
    """Create artifacts/ and repos/ if they don't exist. Safe to call repeatedly."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
