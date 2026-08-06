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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths  # noqa: E402

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


def slim_csv(src: Path, dst: Path, keep: set) -> tuple:
    """Keep rows whose matched_frameworks intersects `keep`; rewrite that column to
    the intersection (sorted). Returns (in_rows, out_rows)."""
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    if not rows:
        return 0, 0
    fields = list(rows[0].keys())
    out = []
    for r in rows:
        kept = sorted(n for n in matched_names(r) if n in keep)
        if kept:
            r = dict(r)
            r["matched_frameworks"] = ", ".join(kept)
            out.append(r)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(out)
    return len(rows), len(out)


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
    a_in, a_out = slim_csv(paths.APPLICATIONS_CSV,
                           paths.ARTIFACTS_DIR / "applications_slim.csv", keep)
    m_in, m_out = slim_csv(paths.APPLICATION_METADATA_CSV,
                           paths.ARTIFACTS_DIR / "application_metadata_slim.csv", keep)
    lost = blind_spot(keep)
    report = {
        "keep_names": len(keep),
        "applications": {"before": a_in, "after": a_out},
        "metadata": {"before": m_in, "after": m_out},
        "blind_spot_frameworks": len(lost),
        "blind_spot": lost,
    }
    with open(paths.ARTIFACTS_DIR / "slim_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"KEEP names: {len(keep)}")
    print(f"applications.csv : {a_in} -> {a_out}  ({a_out/a_in*100:.0f}% kept)" if a_in else "applications.csv empty")
    print(f"metadata.csv     : {m_in} -> {m_out}")
    print(f"blind-spot frameworks (no usable import name): {len(lost)}")
    for e in lost:
        print(f"  {e['framework']}: {', '.join(e['import_names'])}")


if __name__ == "__main__":
    main()
