"""Framework-frequency table over the trustworthy (KEEP) import names.

For each KEEP import name, count how many slimmed applications matched it (a repo
matching N names contributes to N rows). Built from applications_slim.csv, so it
reflects the post-slim, de-polluted candidate set.

Writes pipeline/artifacts/keep_frequency.csv (import_name, applications, pct_of_apps),
sorted by application count.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths  # noqa: E402

SLIM_CSV = paths.ARTIFACTS_DIR / "applications_slim.csv"
OUT_CSV = paths.ARTIFACTS_DIR / "keep_frequency.csv"


def main():
    if not SLIM_CSV.exists():
        sys.exit(f"{SLIM_CSV} not found — run Applications/slim_applications.py first.")
    rows = list(csv.DictReader(open(SLIM_CSV, encoding="utf-8")))
    total_apps = len(rows)
    counts = Counter()
    for r in rows:
        for n in (r.get("matched_frameworks") or "").split(","):
            n = n.strip()
            if n:
                counts[n] += 1

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["import_name", "applications", "pct_of_apps"])
        for name, c in counts.most_common():
            w.writerow([name, c, f"{100 * c / total_apps:.1f}" if total_apps else "0"])

    print(f"total slimmed applications: {total_apps}")
    print(f"distinct KEEP names with >=1 app: {len(counts)}")
    print(f"wrote {OUT_CSV}\n")
    print(f"{'applications':>12}  import_name")
    for name, c in counts.most_common(30):
        print(f"{c:>12}  {name}")


if __name__ == "__main__":
    main()
