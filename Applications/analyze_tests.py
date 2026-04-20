"""Phase 2: clone each candidate repo, parse its test files locally, delete clone.

Reads application_candidates.csv (produced by search_candidates.py),
writes application_tests.csv. Resumable — skips repos already in the output.
"""
import ast
import csv
import os
import re
import shutil
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).parent
INPUT_CSV = HERE / "application_candidates_v2.csv"
OUTPUT_CSV = HERE / "application_tests.csv"
LLM_TESTS_CSV = HERE / "llm_test_functions.csv"
CACHE_DIR = HERE / "repo_cache"
EXTRACTED_DIR = HERE / "extracted_llm_tests"

MAX_WORKERS = 4
CLONE_TIMEOUT = 300
FLUSH_EVERY = 10
TEST_FILE_RE = re.compile(r"(^|/)test_[^/]+\.py$")

# A test is flagged as "LLM-backed" (non-deterministic) when its file imports
# any of these — LLM SDKs plus eval frameworks that typically wrap real models.
LLM_MODULES = {
    # LLM SDKs / clients
    "openai", "anthropic", "litellm", "cohere", "mistralai",
    "google.generativeai", "vertexai", "boto3", "langchain",
    "langchain_openai", "langchain_anthropic", "langchain_community",
    "llamaindex", "llama_index", "ollama", "groq", "together",
    "huggingface_hub", "transformers",
    # LLM evaluation frameworks
    "opik", "deepeval", "giskard", "promptfoo", "phoenix", "arize_phoenix", "ragas",
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


def is_real_llm_test(func_node, func_src: str, file_imports: set) -> bool:
    if not file_imports & LLM_MODULES:
        return False
    if MOCK_PATTERNS.search(func_src):
        return False
    for node in ast.walk(func_node):
        if isinstance(node, (ast.Call, ast.Attribute)):
            try:
                segment = ast.unparse(node)
            except Exception:
                segment = ""
            if LLM_CALL_PATTERNS.search(segment):
                return True
    return False


def remove_readonly(func, path, _):
    """Windows: git pack files are read-only, strip the flag then retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=remove_readonly)


def shallow_clone(clone_url: str, dest: Path) -> bool:
    safe_rmtree(dest)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", clone_url, str(dest)],
            timeout=CLONE_TIMEOUT,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.TimeoutExpired:
        print(f"  Clone timeout: {clone_url}")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else ""
        print(f"  Clone failed: {clone_url} — {stderr.strip()[:200]}")
    safe_rmtree(dest)
    return False


def find_test_files(repo_dir: Path) -> List[Path]:
    test_files = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", ".venv", "venv", "__pycache__")]
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), repo_dir).replace(os.sep, "/")
            if TEST_FILE_RE.search(rel):
                test_files.append(Path(root) / name)
    return test_files


def parse_test_file(path: Path) -> Tuple[int, int, List[dict]]:
    """Return (total_test_count, llm_test_count, llm_test_records).

    llm_test_records is a list of {"name", "source"} for tests that look
    like real (non-mocked) LLM calls.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return 0, 0, []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0, 0, []

    file_imports = get_imports(tree)
    lines = source.splitlines()
    total = 0
    llm_records: List[dict] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            total += 1
            end = getattr(node, "end_lineno", None) or node.lineno
            func_src = "\n".join(lines[node.lineno - 1: end])
            if is_real_llm_test(node, func_src, file_imports):
                llm_records.append({"name": node.name, "source": func_src})

    return total, len(llm_records), llm_records


def analyze_repo(row: dict) -> Tuple[dict, List[dict]]:
    """Return (repo_summary_row, list_of_llm_test_rows)."""
    full_name = row["full_name"]
    clone_url = row["clone_url"]
    safe_name = full_name.replace("/", "_")
    repo_dir = CACHE_DIR / safe_name

    if not shallow_clone(clone_url, repo_dir):
        return (
            {**row, "test_file_count": 0, "test_function_count": 0,
             "llm_test_count": 0, "clone_status": "failed"},
            [],
        )

    try:
        test_files = find_test_files(repo_dir)
        total_funcs = 0
        llm_total = 0
        extracted_sections: List[str] = []
        llm_rows: List[dict] = []

        for tf in test_files:
            rel_path = tf.relative_to(repo_dir).as_posix()
            file_total, file_llm, llm_records = parse_test_file(tf)
            total_funcs += file_total
            llm_total += file_llm
            if llm_records:
                extracted_sections.append(f"\n# --- {rel_path} ---\n\n")
                for rec in llm_records:
                    extracted_sections.append(rec["source"] + "\n\n")
                    llm_rows.append({
                        "repo": full_name,
                        "file": rel_path,
                        "test_function": rec["name"],
                    })

        if extracted_sections:
            EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
            out_path = EXTRACTED_DIR / f"{safe_name}.py"
            header = (
                f"# {full_name}\n"
                f"# {llm_total} LLM-backed test functions across {len(test_files)} test files\n"
                f"# Source: {row.get('html_url')}\n"
            )
            out_path.write_text(header + "".join(extracted_sections), encoding="utf-8")

        return (
            {
                **row,
                "test_file_count": len(test_files),
                "test_function_count": total_funcs,
                "llm_test_count": llm_total,
                "clone_status": "ok",
            },
            llm_rows,
        )
    finally:
        safe_rmtree(repo_dir)


def load_done_full_names() -> set:
    if not OUTPUT_CSV.exists():
        return set()
    with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
        return {row["full_name"] for row in csv.DictReader(f)}


def write_repo_output(rows: List[dict]) -> None:
    if not rows:
        return
    seen = dict()
    for row in rows:
        for k in row:
            if k not in seen:
                seen[k] = None
    fieldnames = list(seen)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (int(r.get("stars") or 0)), reverse=True))


def write_llm_tests(rows: List[dict]) -> None:
    fieldnames = ["repo", "file", "test_function"]
    with open(LLM_TESTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["repo"], r["file"], r["test_function"])))


def load_existing_llm_tests() -> List[dict]:
    if not LLM_TESTS_CSV.exists():
        return []
    with open(LLM_TESTS_CSV, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))

    done = load_done_full_names()
    existing_repo_rows = []
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
            existing_repo_rows = list(csv.DictReader(f))
    existing_llm_rows = load_existing_llm_tests()

    todo = [row for row in candidates if row["full_name"] not in done]
    print(f"{len(candidates)} candidates, {len(done)} already done, {len(todo)} remaining")

    repo_rows = list(existing_repo_rows)
    llm_rows = list(existing_llm_rows)
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_repo, row): row["full_name"] for row in todo}
        for future in as_completed(futures):
            full_name = futures[future]
            processed += 1
            try:
                repo_row, new_llm_rows = future.result()
                repo_rows.append(repo_row)
                llm_rows.extend(new_llm_rows)
                status = repo_row.get("clone_status")
                tfc = repo_row.get("test_file_count")
                tfn = repo_row.get("test_function_count")
                llm = repo_row.get("llm_test_count")
                print(f"  [{processed}/{len(todo)}] {full_name}: {status}, "
                      f"{tfc} test files, {tfn} tests, {llm} LLM-backed")
            except Exception as e:
                print(f"  [{processed}/{len(todo)}] {full_name}: ERROR {e}")

            if processed % FLUSH_EVERY == 0:
                write_repo_output(repo_rows)
                write_llm_tests(llm_rows)
                print(f"  ---- CSVs flushed ({len(repo_rows)} repos, {len(llm_rows)} LLM tests) ----")

    write_repo_output(repo_rows)
    write_llm_tests(llm_rows)
    safe_rmtree(CACHE_DIR)
    print(f"\nDone. {len(repo_rows)} repos in {OUTPUT_CSV}, "
          f"{len(llm_rows)} LLM-backed tests in {LLM_TESTS_CSV}")


if __name__ == "__main__":
    main()
