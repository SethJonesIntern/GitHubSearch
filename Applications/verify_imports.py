"""One-off: for the import patterns we're unsure about, inspect each repo's
actual structure to find the real importable package and how the README
imports it. Uses only core REST (trees + contents + readme), NOT code search,
so it won't eat the code-search rate budget the main run needs.
"""
import base64
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Frameworks", ".env"))
TOKEN = os.getenv("GITHUB_TOKEN")
H = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
if TOKEN:
    H["Authorization"] = f"Bearer {TOKEN}"
API = "https://api.github.com"

# framework -> (repo full_name, current pattern in our code)
UNCERTAIN = {
    "agent-zero": ("agent0ai/agent-zero", "from agent_zero import"),
    "ten-framework": ("TEN-framework/ten-framework", "from ten_framework import"),
    "pentestgpt": ("GreyDGL/PentestGPT", "from pentestgpt import"),
    "pentestagent": ("GH05TCREW/pentestagent", "from pentestagent import"),
    "ii-agent": ("Intelligent-Internet/ii-agent", "from ii_agent import"),
    "llmstack": ("trypromptly/LLMStack", "from llmstack import"),
    "redamon": ("samugit83/redamon", "from redamon import"),
    "openakita": ("openakita/openakita", "from openakita import"),
    "infiagent": ("polyuiislab/infiAgent", "from infiagent import"),
    "solace-agent-mesh": ("SolaceLabs/solace-agent-mesh", "from solace_agent_mesh import"),
    "notte": ("nottelabs/notte", "from notte import"),
    "honcho": ("plastic-labs/honcho", "from honcho import"),
    "patchwork": ("patched-codes/patchwork", "from patchwork import"),
    "sage": ("ZHangZHengEric/Sage", "from sage import"),
    "npcpy": ("NPC-Worldwide/npcpy", "from npcpy import"),
    "parlant": ("emcie-co/parlant", "from parlant import"),
    "openai-swarm": ("openai/swarm", "from swarm import"),
}

IMPORT_RE = re.compile(r"^\s*(?:from\s+[\w.]+\s+import\s+.+|import\s+[\w.]+.*)$")


def get(url, **params):
    r = requests.get(url, headers=H, params=params or None, timeout=30)
    return r


def repo_info(full):
    r = get(f"{API}/repos/{full}")
    if r.status_code != 200:
        return None
    return r.json()


def find_packages(full, branch):
    """Return top-level importable package dirs (those with __init__.py at
    depth 1, or under src/). These are the real import roots."""
    r = get(f"{API}/repos/{full}/git/trees/{branch}", recursive="1")
    if r.status_code != 200:
        return [], False
    data = r.json()
    truncated = data.get("truncated", False)
    pkgs = set()
    for node in data.get("tree", []):
        if node.get("type") != "blob":
            continue
        p = node.get("path", "")
        # depth-1 package:  pkg/__init__.py
        m = re.match(r"^([^/]+)/__init__\.py$", p)
        if m:
            pkgs.add(m.group(1))
        # src layout:  src/pkg/__init__.py
        m = re.match(r"^src/([^/]+)/__init__\.py$", p)
        if m:
            pkgs.add(m.group(1))
    return sorted(pkgs), truncated


def readme_imports(full):
    r = get(f"{API}/repos/{full}/readme")
    if r.status_code != 200:
        return []
    content = r.json().get("content", "")
    try:
        text = base64.b64decode(content).decode("utf-8", "replace")
    except Exception:
        return []
    hits = []
    for line in text.splitlines():
        s = line.strip()
        if IMPORT_RE.match(s) and len(s) < 100:
            hits.append(s)
    # dedupe, keep order
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out[:12]


def main():
    for fw, (full, current) in UNCERTAIN.items():
        info = repo_info(full)
        print("=" * 70)
        if info is None:
            print(f"{fw:18} {full}  -> REPO NOT FOUND (renamed/moved?)")
            continue
        branch = info.get("default_branch", "main")
        pkgs, trunc = find_packages(full, branch)
        imports = readme_imports(full)
        print(f"{fw}  ({full})  branch={branch}")
        print(f"  current pattern : {current}")
        print(f"  top-level pkgs  : {pkgs if pkgs else '(none — likely an app, not an importable library)'}"
              + ("  [tree truncated]" if trunc else ""))
        if imports:
            print(f"  README imports  :")
            for i in imports:
                print(f"      {i}")
        else:
            print(f"  README imports  : (none found)")
    print("=" * 70)


if __name__ == "__main__":
    main()
