import ast
import csv
import os
import re
import time
from collections import defaultdict

import requests
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
RAW_HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

INPUT_CSV = "llm_test_functions.csv"
OUTPUT_DIR = "extracted_llm_tests"


def fetch_raw(owner, repo, branch, path):
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    try:
        resp = requests.get(url, headers=RAW_HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def get_default_branch(owner, repo):
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    resp = requests.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json().get("default_branch", "main")


def extract_function_source(source, func_name):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return "\n".join(lines[node.lineno - 1: node.end_lineno])
    return None


def main():
    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["framework"], row["file"])].append(row["test_function"])

    branches = {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    framework_functions = defaultdict(list)

    for (framework, filepath), func_names in grouped.items():
        owner, repo = framework.split("/", 1)

        if framework not in branches:
            try:
                branches[framework] = get_default_branch(owner, repo)
            except Exception:
                branches[framework] = "main"
            time.sleep(0.1)

        branch = branches[framework]
        print(f"Fetching {framework}/{filepath}...")
        source = fetch_raw(owner, repo, branch, filepath)
        time.sleep(0.05)

        if not source:
            print(f"  Could not fetch, skipping")
            continue

        for func_name in func_names:
            func_src = extract_function_source(source, func_name)
            if func_src:
                framework_functions[framework].append({
                    "file": filepath,
                    "name": func_name,
                    "source": func_src,
                })

    for framework, functions in sorted(framework_functions.items()):
        safe_name = framework.replace("/", "_")
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.py")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# {framework}\n")
            f.write(f"# {len(functions)} test functions with real LLM calls\n")
            f.write(f"# Source: https://github.com/{framework}\n\n")

            current_file = None
            for func in functions:
                if func["file"] != current_file:
                    current_file = func["file"]
                    f.write(f"\n# --- {current_file} ---\n\n")
                f.write(func["source"])
                f.write("\n\n")

        print(f"Wrote {len(functions)} functions to {output_path}")

    print(f"\nDone. {sum(len(v) for v in framework_functions.values())} functions across {len(framework_functions)} frameworks in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
