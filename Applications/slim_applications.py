"""Post-hoc slimming of Stage 2 output.

Stage 1's derive_import_names over-extracted import names (internal subpackages,
and PEP-420 namespace packages mis-split to their last segment), so Stage 2 matched
many candidates on names that don't identify a framework — generic words (app, api,
core...), infra/tool libs, and provider SDKs misattributed to a vendoring framework
(e.g. `openai` credited to livekit/agents). See pipeline/artifacts/name_classification.csv.

This does NOT re-run the search. Since every one of the 539 names (good and bad) was
already searched, the existing results already contain every repo matching any GOOD
name; we simply keep candidates whose matched_frameworks intersects a trustworthy
KEEP set, and rewrite matched_frameworks to that intersection.

Reads : applications.csv, application_metadata.csv, name_classification.csv
Writes: applications_slim.csv, application_metadata_slim.csv, slim_report.json
Non-destructive: originals are untouched. Idempotent: safe to re-run after the
Stage 2 enrichment completes.
"""
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "Wrapper"))
from pipeline import paths  # noqa: E402
from pipeline.eval_calls import EVAL_CALLS  # noqa: E402
from FrameworkDict import FRAMEWORK_CALLS, IN_SCOPE_FRAMEWORKS  # noqa: E402

CLASSIFICATION_CSV = paths.ARTIFACTS_DIR / "name_classification.csv"

# REVIEW-bucket names promoted to KEEP: real frameworks that were misattributed to a
# vendoring framework and so didn't match their own source repo name.
EXPLICIT_KEEP = {
    "smolagents", "graphrag", "clai",
    # rescued: real framework imports dropped by over-aggressive auto rules
    "sia", "ten_runtime",
    # blind-spot frameworks recovered via targeted re-search (correct import names
    # that Stage 1 never extracted). agent-zero/redamon are apps with no importable
    # package, so they remain unrecoverable by import-pattern search.
    "lavague", "solace_agent_mesh", "assetopsbench_mcp", "assetopsbench",
}

# Auto-KEEP names demoted to DROP. `agent`/`agents` matched their source repo name
# (langchain-ai/langgraph has an `agent/` subdir; livekit/agents, openai-agents-python
# are named `agents`) so identity-match accepted them — but they're generic tokens
# (`agent` isn't any framework's real import; `agents` is the OpenAI Agents SDK but is
# dominated by repos with their own local `agents/` package), so they carry heavy
# false positives.
# `mcp` is the Model Context Protocol SDK — cross-cutting protocol infrastructure that
# many frameworks import, not a framework itself (same category as the dropped provider
# SDKs), so it inflates the table without identifying a framework.
EXPLICIT_DROP = {"agent", "agents", "mcp"}


def framework_self_repos() -> set:
    """Canonical repos of every framework / eval tool we analyze. A framework's OWN
    source repo is not an application built on it — its self-imports, tests and examples
    inflate every metric (agno's own test suite alone = >12k "ND tests"). Dropped from
    the population so Stage 5 never clones or call-graphs them (~50-70 large repos).
    Base = frameworks.csv full_names (the Stage-1 discovery record); supplemented with
    the top-20 frameworks + eval tools added after that snapshot. NOTE: only frameworks
    in our dicts are here — langflow/litellm/marimo are NOT discovered frameworks, so
    they remain applications."""
    repos: set[str] = set()
    if paths.FRAMEWORKS_CSV.exists():
        with open(paths.FRAMEWORKS_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fn = (row.get("full_name") or "").strip()
                if fn:
                    repos.add(fn)
    repos |= {
        "agno-agi/agno", "mem0ai/mem0", "stanfordnlp/dspy", "deepset-ai/haystack",
        "huggingface/smolagents", "microsoft/graphrag", "microsoft/semantic-kernel",
        # eval tools (EVAL_CALLS)
        "confident-ai/deepeval", "explodinggradients/ragas", "vibrantlabsai/ragas",
        "Giskard-AI/giskard", "comet-ml/opik", "Arize-ai/phoenix",
    }
    return repos


SELF_REPOS = framework_self_repos()


# ── analysis scope ────────────────────────────────────────────────────────────
# A KEEP name is trustworthy (it identifies *something*), but that alone doesn't
# make its repo an AI application, nor one we can measure. Two further tests,
# both derived from the dicts rather than hand-listed — see COVERAGE_ANALYSIS.md
# and EXCLUSIONS.md §9.
#
# ALIASES: companion / submodule packages of a framework we DO analyze. Their apps
# really are that framework's apps, but the package name isn't a FRAMEWORK_CALLS
# key, so without the rollup they'd look unmeasurable and be dropped. NOTE the
# Stage-5 detector needs the same table at its own import-matching choke point
# (Wrapper/transitive_invokers.index_repo) — this one only decides what we RUN.
# `clai` is deliberately NOT here: it is a junk collision token (binance-connector,
# py-stellar-base, huaweicloud-sdk...), NOT pydantic-ai's CLI as once assumed.
ALIASES = {
    "agent_framework_foundry": "agent_framework",
    "agent_framework_openai": "agent_framework",
    "agent_framework_foundry_hosting": "agent_framework",
    "crewai_tools": "crewai",
}

# REAL_AI  — is this an AI application at all? (imports a framework/eval tool we
#            discovered and have patterns for). Junk tokens (`clai`), non-LLM
#            langchain utilities (`langchain_text_splitters`/`_chroma`/`_qdrant`/
#            `_tests`) and the `omnigent` phantom are not keys, so they fail this.
# ANALYZED — do we actually measure it? (imports an in-scope top-20 framework or
#            an eval tool). Real AI apps on out-of-scope long-tail frameworks
#            (metagpt, lagent, honcho...) fail this but stay REAL_AI: they are
#            "known uncovered" and belong in the denominator, not in the run.
REAL_AI = set(FRAMEWORK_CALLS) | set(EVAL_CALLS)
ANALYZED = set(IN_SCOPE_FRAMEWORKS) | set(EVAL_CALLS)

SCOPE_FLAGS = ["real_ai_app", "analyzed"]


def resolve(names) -> set:
    """Matched import names with companion packages rolled up to their parent."""
    return {ALIASES.get(n, n) for n in names}


def load_keep_set() -> set:
    """KEEP = auto_bucket==KEEP from the classification, plus EXPLICIT_KEEP.
    Everything else (DROP, REVIEW_provider incl. LLM SDKs, and the REVIEW tail)
    is dropped."""
    if not CLASSIFICATION_CSV.exists():
        sys.exit(f"{CLASSIFICATION_CSV} not found — regenerate the name classification first.")
    keep = set(EXPLICIT_KEEP)
    with open(CLASSIFICATION_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["auto_bucket"] == "KEEP":
                keep.add(row["import_name"])
    return keep - EXPLICIT_DROP


def matched_names(row: dict) -> list:
    return [n.strip() for n in (row.get("matched_frameworks") or "").split(",") if n.strip()]


def _write_rows(path: Path, fields: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


def slim_csv(src: Path, dst: Path, keep: set, analyzed_dst: Path = None) -> tuple:
    """Keep rows whose matched_frameworks intersects `keep`; rewrite that column to
    the intersection (sorted), and tag each row with the two scope flags.

    Nothing is dropped for scope: junk and out-of-scope rows stay in `dst` carrying
    real_ai_app=0 / analyzed=0, so every denominator (all candidates / real AI apps /
    analyzed apps) is recoverable from one file. `analyzed_dst`, when given, gets the
    analyzed subset only — that is the batch run input, so we never clone a repo we
    can't measure. Returns (in_rows, out_rows, real_ai_rows, analyzed_rows)."""
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    if not rows:
        return 0, 0, 0, 0
    fields = [k for k in rows[0].keys() if k not in SCOPE_FLAGS] + SCOPE_FLAGS
    out, analyzed_rows = [], []
    dropped_self = n_real = 0
    for r in rows:
        if (r.get("full_name") or "").strip() in SELF_REPOS:
            dropped_self += 1          # a framework/eval tool's own repo — not an app
            continue
        kept = sorted(n for n in matched_names(r) if n in keep)
        if not kept:
            continue
        r = dict(r)
        r["matched_frameworks"] = ", ".join(kept)
        resolved = resolve(kept)
        r["real_ai_app"] = "1" if resolved & REAL_AI else "0"
        r["analyzed"] = "1" if resolved & ANALYZED else "0"
        out.append(r)
        n_real += r["real_ai_app"] == "1"
        if r["analyzed"] == "1":
            analyzed_rows.append(r)
    if dropped_self:
        print(f"  [{dst.name}] dropped {dropped_self} framework/eval self-repositories")
    _write_rows(dst, fields, out)
    if analyzed_dst is not None:
        _write_rows(analyzed_dst, fields, analyzed_rows)
    return len(rows), len(out), n_real, len(analyzed_rows)


def blind_spot(keep: set) -> list:
    """Frameworks whose EVERY extracted import name was dropped — their genuine users
    were only searchable via junk names, so the salvage cannot recover them."""
    lost = []
    with open(paths.FRAMEWORKS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row.get("full_name")
            names = [n.strip() for n in (row.get("import_names") or "").split(";") if n.strip()]
            if names and not any(n in keep for n in names):
                lost.append({"framework": fn, "import_names": names})
    return lost


def main():
    keep = load_keep_set()
    analyzed_csv = paths.ARTIFACTS_DIR / "applications_analyzed.csv"
    a_in, a_out, a_real, a_run = slim_csv(
        paths.APPLICATIONS_CSV, paths.ARTIFACTS_DIR / "applications_slim.csv",
        keep, analyzed_dst=analyzed_csv)
    m_in, m_out, _, _ = slim_csv(paths.APPLICATION_METADATA_CSV,
                                 paths.ARTIFACTS_DIR / "application_metadata_slim.csv", keep)
    lost = blind_spot(keep)
    report = {
        "keep_names": len(keep),
        "applications": {"before": a_in, "after": a_out,
                         "real_ai_apps": a_real, "analyzed": a_run},
        "coverage_pct": round(100 * a_run / a_real, 1) if a_real else 0,
        "metadata": {"before": m_in, "after": m_out},
        "blind_spot_frameworks": len(lost),
        "blind_spot": lost,
    }
    with open(paths.ARTIFACTS_DIR / "slim_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"KEEP names: {len(keep)}")
    print(f"applications.csv : {a_in} -> {a_out}  ({a_out/a_in*100:.0f}% kept)" if a_in else "applications.csv empty")
    print(f"metadata.csv     : {m_in} -> {m_out}")
    print(f"  real AI apps (denominator) : {a_real}")
    print(f"  analyzed     (run set)     : {a_run} -> {analyzed_csv.name}")
    print(f"  coverage                   : {report['coverage_pct']}%")
    print(f"blind-spot frameworks (no usable import name): {len(lost)}")
    for e in lost:
        print(f"  {e['framework']}: {', '.join(e['import_names'])}")


if __name__ == "__main__":
    main()
