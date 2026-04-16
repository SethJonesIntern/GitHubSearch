"""Scan the already-extracted LLM test files under
Applications/extracted_llm_tests/ and Frameworks/extracted_llm_tests/
and keep only test functions whose body references a semantic
evaluator framework (Giskard, DeepEval, Opik, RAGAs, Phoenix,
Promptfoo).

Outputs:
  SemanticEvaluators/extracted_tests/<repo>.py
  SemanticEvaluators/semantic_evaluator_tests.csv
"""
import ast
import csv
import re
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).parent
SOURCE_DIRS = [
    HERE.parent / "Applications" / "extracted_llm_tests",
    HERE.parent / "Frameworks" / "extracted_llm_tests",
]
OUT_DIR = HERE / "extracted_tests"
OUT_CSV = HERE / "semantic_evaluator_tests.csv"

SEMANTIC_EVAL_FRAMEWORKS = {
    "giskard": re.compile(r"\bgiskard\b", re.IGNORECASE),
    "deepeval": re.compile(r"\bdeepeval\b", re.IGNORECASE),
    "opik": re.compile(r"\bopik\b", re.IGNORECASE),
    "ragas": re.compile(r"\bragas\b", re.IGNORECASE),
    "promptfoo": re.compile(r"\bpromptfoo\b", re.IGNORECASE),
    "phoenix": re.compile(r"\b(arize_phoenix|arize\.phoenix|phoenix\.evals|phoenix\.experiments)\b", re.IGNORECASE),
}

SECTION_RE = re.compile(r"^# --- (.+?) ---\s*$", re.MULTILINE)


def split_sections(text: str) -> List[Tuple[str, str]]:
    """Split an extracted_llm_tests file into (relative_path, body) sections."""
    matches = list(SECTION_RE.finditer(text))
    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        rel_path = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((rel_path, text[start:end]))
    return sections


def iter_test_funcs(section_src: str):
    """Yield (func_name, func_source) for each top-level test_ function in a section.

    Sections are concatenations of individual extracted function sources, so we
    fall back to a regex split if ast.parse chokes (e.g. async defs at module
    top-level with surrounding blank lines parse fine, but stray snippets may
    not)."""
    try:
        tree = ast.parse(section_src)
    except SyntaxError:
        yield from _regex_split_funcs(section_src)
        return

    lines = section_src.splitlines()
    found_any = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            found_any = True
            end = getattr(node, "end_lineno", None) or node.lineno
            yield node.name, "\n".join(lines[node.lineno - 1: end])

    if not found_any:
        yield from _regex_split_funcs(section_src)


FUNC_HEADER_RE = re.compile(
    r"^(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(",
    re.MULTILINE,
)


def _regex_split_funcs(section_src: str):
    headers = list(FUNC_HEADER_RE.finditer(section_src))
    for i, m in enumerate(headers):
        name = m.group(1)
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(section_src)
        yield name, section_src[start:end]


def frameworks_in(text: str) -> List[str]:
    return [name for name, rx in SEMANTIC_EVAL_FRAMEWORKS.items() if rx.search(text)]


def scan_file(path: Path):
    """Return (repo_slug, list of matching {file, test_function, frameworks, source})."""
    text = path.read_text(encoding="utf-8", errors="replace")
    repo_slug = path.stem
    hits = []
    for rel_path, section_src in split_sections(text):
        for func_name, func_src in iter_test_funcs(section_src):
            fws = frameworks_in(func_src)
            if fws:
                hits.append({
                    "file": rel_path,
                    "test_function": func_name,
                    "frameworks": ",".join(fws),
                    "source": func_src.rstrip() + "\n",
                })
    return repo_slug, hits


def write_repo_output(repo_slug: str, source_path: Path, hits: List[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header_lines = [f"# {repo_slug.replace('_', '/', 1)}"]
    header_lines.append(f"# {len(hits)} semantic-evaluator test functions")
    header_lines.append(f"# Source extract: {source_path.as_posix()}")
    header_lines.append("")
    body: List[str] = []
    last_file = None
    for h in hits:
        if h["file"] != last_file:
            body.append(f"\n# --- {h['file']}  [{h['frameworks']}] ---\n\n")
            last_file = h["file"]
        body.append(h["source"] + "\n")
    out_path = OUT_DIR / f"{repo_slug}.py"
    out_path.write_text("\n".join(header_lines) + "".join(body), encoding="utf-8")


def main() -> None:
    all_rows: List[dict] = []
    per_repo_counts: List[Tuple[str, int]] = []

    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            print(f"Skipping missing source dir: {src_dir}")
            continue
        category = src_dir.parent.name
        for py_file in sorted(src_dir.glob("*.py")):
            repo_slug, hits = scan_file(py_file)
            if not hits:
                continue
            write_repo_output(repo_slug, py_file, hits)
            per_repo_counts.append((f"{category}/{repo_slug}", len(hits)))
            for h in hits:
                all_rows.append({
                    "category": category,
                    "repo": repo_slug.replace("_", "/", 1),
                    "file": h["file"],
                    "test_function": h["test_function"],
                    "frameworks": h["frameworks"],
                })

    fieldnames = ["category", "repo", "file", "test_function", "frameworks"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda r: (r["category"], r["repo"], r["file"], r["test_function"])))

    print(f"Wrote {len(all_rows)} tests from {len(per_repo_counts)} repos to {OUT_CSV}")
    for name, n in sorted(per_repo_counts, key=lambda x: -x[1]):
        print(f"  {n:4d}  {name}")


if __name__ == "__main__":
    main()
