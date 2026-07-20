"""Select one representative application per top-N framework for a pilot run.

Picks the highest-starred slimmed application matching each of the top-N ecosystem
categories (grouped the same way as keep_frequency.py). Greedy in rank order so
each framework gets a DISTINCT repo: if a framework's top pick is already taken by
a higher-ranked framework, the next-highest-starred match is used instead.

Reads : keep_frequency.csv (ranking), applications_slim.csv (candidates)
Writes: pilot_applications.csv  (same schema as applications_slim.csv, so it can be
        fed to `python -m pipeline.batch_call_metadata --input <this file>`)
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths  # noqa: E402
import importlib.util
_spec = importlib.util.spec_from_file_location("kf", Path(__file__).with_name("keep_frequency.py"))
kf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(kf)

FREQ_CSV = paths.ARTIFACTS_DIR / "keep_frequency.csv"
SLIM_CSV = paths.ARTIFACTS_DIR / "applications_slim.csv"
FRAMEWORKS_CSV = paths.FRAMEWORKS_CSV
OUT_CSV = paths.ARTIFACTS_DIR / "pilot_applications.csv"

# Short/common-word framework tokens prone to false positives (a big unrelated repo
# with its own local `cat`/`camel`/... module matches the string). A pick under one
# of these is only trusted when the same repo ALSO matches a distinctive framework.
NOISY = {"cat", "clai", "camel", "subagents", "agui", "omnigent",
         "honcho", "sia", "swarm", "notte"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def stars(row: dict) -> int:
    try:
        return int(row.get("stars") or 0)
    except ValueError:
        return 0


def matched(row: dict) -> list:
    return [n.strip() for n in (row.get("matched_frameworks") or "").split(",") if n.strip()]


def categories(row: dict) -> set:
    return {kf.category(n) for n in matched(row)}


def is_own_repo(row: dict, fw: str, framework_repos: set) -> bool:
    """The framework's own repo, not an application using it: either it's in the
    Stage-1 framework list, or its name equals a matched package that maps to fw
    (e.g. huggingface/smolagents matched 'smolagents')."""
    if row["full_name"].lower() in framework_repos:
        return True
    short = norm(row["full_name"].split("/")[-1])
    return any(norm(n) == short for n in matched(row) if kf.category(n) == fw)


def is_credible(row: dict, fw: str) -> bool:
    """For a noisy-token framework, require the repo to also match a distinctive
    (non-noisy) framework, so lone-noisy-token false positives are dropped."""
    if fw not in NOISY:
        return True
    return any(c not in NOISY for c in categories(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20, help="number of top frameworks to sample")
    ap.add_argument("--per-framework", type=int, default=1,
                    help="how many top repos to take per framework (distinct across all)")
    args = ap.parse_args()

    if not (FREQ_CSV.exists() and SLIM_CSV.exists()):
        sys.exit("run keep_frequency.py / slim_applications.py first.")
    ranked = [r["framework"] for r in csv.DictReader(open(FREQ_CSV, encoding="utf-8"))][:args.top]

    framework_repos = {r["full_name"].lower()
                       for r in csv.DictReader(open(FRAMEWORKS_CSV, encoding="utf-8"))
                       if r.get("full_name")} if FRAMEWORKS_CSV.exists() else set()

    with open(SLIM_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    chosen, chosen_names, missing = [], set(), []
    for fw in ranked:
        cands = sorted((r for r in rows
                        if fw in categories(r) and r["full_name"] not in chosen_names
                        and r.get("clone_url")
                        and not is_own_repo(r, fw, framework_repos)
                        and is_credible(r, fw)),
                       key=stars, reverse=True)
        picks = cands[:args.per_framework]
        if not picks:
            missing.append(fw)
            continue
        for pick in picks:
            chosen.append((fw, pick))
            chosen_names.add(pick["full_name"])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for _, pick in chosen:
            w.writerow(pick)

    print(f"selected {len(chosen)} repos (up to {args.per_framework} per framework) "
          f"-> {OUT_CSV.name}\n")
    print(f"{'framework':<22}{'stars':>8}  repo")
    last = None
    for fw, pick in chosen:
        label = fw if fw != last else ""
        print(f"{label:<22}{stars(pick):>8}  {pick['full_name']}")
        last = fw
    if missing:
        print(f"\nno distinct repo available for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
