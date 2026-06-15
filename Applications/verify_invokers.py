"""Second pass: for the frameworks whose invocation method we're unsure of,
mine each repo's README + example files for the method calls users actually
make, so we can set real seeds in Wrapper/FrameworkDict.py.

Core REST only (readme + trees + raw example files) — no code search.
"""
import base64
import os
import re
from collections import Counter

import requests
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Frameworks", ".env"))
TOKEN = os.getenv("GITHUB_TOKEN")
H = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
RAW = {}
if TOKEN:
    H["Authorization"] = f"Bearer {TOKEN}"
    RAW["Authorization"] = f"Bearer {TOKEN}"
API = "https://api.github.com"

REPOS = {
    "parlant": "emcie-co/parlant",
    "nextpy": "dot-agent/nextpy",
    "ten": "TEN-framework/ten-framework",
    "ii_agent": "Intelligent-Internet/ii-agent",
    "solace_agent_mesh": "SolaceLabs/solace-agent-mesh",
    "cheshire_cat": "cheshire-cat-ai/core",
    "lagent": "InternLM/lagent",
}

METHOD_RE = re.compile(r"\.([a-zA-Z_]\w*)\s*\(")
AWAIT_RE = re.compile(r"await\s+[\w.]+\.([a-zA-Z_]\w*)\s*\(")
EXAMPLE_DIRS = ("example", "examples", "cookbook", "quickstart", "docs", "demo")
# noise methods to ignore — not framework invocations
IGNORE = {"format", "join", "split", "append", "get", "print", "len", "open",
          "read", "write", "load", "dump", "dumps", "loads", "add", "update",
          "items", "keys", "values", "str", "int", "range", "list", "dict",
          "getenv", "environ", "sleep", "path", "len", "set", "strip"}


def get(url, **params):
    return requests.get(url, headers=H, params=params or None, timeout=30)


def readme_text(full):
    r = get(f"{API}/repos/{full}/readme")
    if r.status_code != 200:
        return ""
    try:
        return base64.b64decode(r.json().get("content", "")).decode("utf-8", "replace")
    except Exception:
        return ""


def example_files(full, branch, limit=3):
    r = get(f"{API}/repos/{full}/git/trees/{branch}", recursive="1")
    if r.status_code != 200:
        return []
    paths = [
        n["path"] for n in r.json().get("tree", [])
        if n.get("type") == "blob" and n["path"].endswith(".py")
        and any(f"/{d}/" in f"/{n['path']}" or n["path"].startswith(f"{d}/") for d in EXAMPLE_DIRS)
        and "test" not in n["path"].lower()
    ]
    # prefer short/top-level example files
    paths.sort(key=lambda p: (p.count("/"), len(p)))
    return paths[:limit]


def raw_file(full, branch, path):
    url = f"https://raw.githubusercontent.com/{full}/{branch}/{path}"
    r = requests.get(url, headers=RAW, timeout=20)
    return r.text if r.status_code == 200 else ""


def analyze(text):
    methods = Counter()
    awaited = Counter()
    for m in METHOD_RE.finditer(text):
        name = m.group(1)
        if name not in IGNORE and not name.startswith("_"):
            methods[name] += 1
    for m in AWAIT_RE.finditer(text):
        awaited[m.group(1)] += 1
    return methods, awaited


def usage_lines(text, names, maxn=6):
    out = []
    for line in text.splitlines():
        s = line.strip()
        if 8 < len(s) < 100 and any(f".{n}(" in s for n in names):
            if s not in out:
                out.append(s)
        if len(out) >= maxn:
            break
    return out


def main():
    for fw, full in REPOS.items():
        info = get(f"{API}/repos/{full}")
        branch = info.json().get("default_branch", "main") if info.status_code == 200 else "main"
        rm = readme_text(full)
        m1, a1 = analyze(rm)

        ex_paths = example_files(full, branch)
        ex_text = ""
        for p in ex_paths:
            ex_text += "\n" + raw_file(full, branch, p)
        m2, a2 = analyze(ex_text)

        combined = m1 + m2
        awaited = a1 + a2
        print("=" * 72)
        print(f"{fw}  ({full})  branch={branch}")
        print(f"  example files scanned: {ex_paths or '(none)'}")
        print(f"  top method calls (readme+examples): "
              f"{[f'{n}:{c}' for n, c in combined.most_common(12)] or '(none)'}")
        if awaited:
            print(f"  awaited calls          : {[f'{n}:{c}' for n, c in awaited.most_common(8)]}")
        samples = usage_lines(rm + ex_text, [n for n, _ in combined.most_common(8)])
        if samples:
            print(f"  sample usage lines:")
            for s in samples:
                print(f"      {s}")
    print("=" * 72)


if __name__ == "__main__":
    main()
