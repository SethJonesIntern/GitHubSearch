import csv

INPUT_CSV = "github_agent_framework_candidates.csv"
OUTPUT_CSV = "agent_framework_table.csv"

with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

table = []
for r in rows:
    try:
        stars_k = round(int(r["stars"]) / 1000, 1)
    except (ValueError, TypeError):
        stars_k = ""
    table.append({
        "AI agent framework": r["full_name"],
        "# Stars (k)": stars_k,
        "# Contributors": r["contributors_count"],
        "# Test files": r["test_file_count"],
        "# Test functions": r["test_function_count"],
    })

table.sort(key=lambda r: float(r["# Stars (k)"]) if r["# Stars (k)"] != "" else 0, reverse=True)

fieldnames = ["AI agent framework", "# Stars (k)", "# Contributors", "# Test files", "# Test functions"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(table)

print(f"Wrote {len(table)} rows to {OUTPUT_CSV}")
