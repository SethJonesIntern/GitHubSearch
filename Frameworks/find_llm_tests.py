import ast
import csv
import re
import time
import requests
from dotenv import load_dotenv
from typing import Optional
import os

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
RAW_HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

INPUT_CSV = "github_agent_framework_candidates.csv"
OUTPUT_CSV = "llm_test_functions.csv"

LLM_MODULES = {
    "openai", "anthropic", "litellm", "cohere", "mistralai",
    "google.generativeai", "vertexai", "boto3", "langchain",
    "langchain_openai", "langchain_anthropic", "langchain_community",
    "llamaindex", "llama_index", "ollama", "groq", "together",
    "huggingface_hub", "transformers",
}

LLM_CALL_PATTERNS = re.compile(
    r"(chat\.completions\.create|messages\.create|generate|invoke|"
    r"complete|completion|predict|run|stream|achat|ainvoke|agenerate|"
    r"apredict|acomplete|astream)",
    re.IGNORECASE,
)

MOCK_PATTERNS = re.compile(
    r"mock|patch|fake|stub|fixture|monkeypatch|MagicMock|AsyncMock",
    re.IGNORECASE,
)


def get_imports(tree: ast.Module) -> set:
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    return imported


def has_real_llm_call(func_node: ast.FunctionDef, source: str, file_imports: set) -> bool:
    if not file_imports & LLM_MODULES:
        return False

    func_lines = source.splitlines()[func_node.lineno - 1: func_node.end_lineno]
    func_src = "\n".join(func_lines)

    if MOCK_PATTERNS.search(func_src):
        return False

    for node in ast.walk(func_node):
        if isinstance(node, (ast.Call, ast.Attribute)):
            segment = ast.unparse(node) if hasattr(ast, "unparse") else ""
            if LLM_CALL_PATTERNS.search(segment):
                return True

    return False


def fetch_raw(owner: str, repo: str, branch: str, path: str) -> Optional[str]:
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    try:
        resp = requests.get(url, headers=RAW_HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def get_test_files(owner: str, repo: str, branch: str) -> list:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}"
    resp = requests.get(url, headers=headers, params={"recursive": "1"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [
        item["path"] for item in data.get("tree", [])
        if item["type"] == "blob"
        and re.search(r"(^|/)test_[^/]+\.py$", item["path"])
    ]


def main():
    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        repos = list(reader)

    results = []

    for repo_row in repos:
        full_name = repo_row["full_name"]
        branch = repo_row["default_branch"] or "main"
        owner, repo = full_name.split("/", 1)

        print(f"Scanning {full_name}...")

        try:
            test_files = get_test_files(owner, repo, branch)
        except Exception as e:
            print(f"  Could not fetch tree: {e}")
            continue

        for path in test_files:
            source = fetch_raw(owner, repo, branch, path)
            time.sleep(0.05)
            if not source:
                continue

            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            file_imports = get_imports(tree)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("test_"):
                        continue
                    if has_real_llm_call(node, source, file_imports):
                        results.append({
                            "framework": full_name,
                            "file": path,
                            "test_function": node.name,
                        })
                        print(f"  [FOUND] {path}::{node.name}")

        time.sleep(0.2)

    fieldnames = ["framework", "file", "test_function"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["framework"]))

    print(f"\nFound {len(results)} test functions with real LLM calls across {len({r['framework'] for r in results})} frameworks.")
    print(f"Wrote to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
