"""Search a local or remote repo for LLM SDK imports and call patterns."""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from astWrappers import Match, scan_repo

LLM_IMPORTS = {
    # LLM SDKs
    "openai",
    "anthropic",
    # Agent frameworks (from Frameworks/agent_framework_table.csv)
    "langchain",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_community",
    "langchain_core",
    "langgraph",
    "metagpt",
    "autogen",
    "autogen_core",
    "autogen_agentchat",
    "crewai",
    "swarm",
    "agents",                  # openai-agents-python
    "parlant",
    "superagi",
    "camel",
    "ragaai_catalyst",
    "pydantic_ai",
    "ten",
    "livekit",
    "agent_framework",
    "agent_squad",
    "praisonai",
    "lavague",
    "swarms",
    "agentops",
    "superduper",
    "giskard",
    "agency_swarm",
    "adalflow",
    "ii_agent",
    "beeai_framework",
    "cheshire_cat",
    "solace_agent_mesh",
    "griptape",
    "openlit",
    "nextpy",
    "llmstack",
    "lagent",
    "agentuniverse",
    "notte",
    "redamon",
    "honcho",
    "uagents",
    "patchwork",
    "agent_protocol",
    "npcpy",
    "infiagent",
    "any_agent",
    "sage",
    "dynamiq",
}

LLM_CALLS = [
    # Direct SDK calls
    "chat.completions.create",
    "messages.create",
    # Agent framework entrypoints (generic — noisy until paired with import filter)
    ".invoke",
    ".ainvoke",
    ".stream",
    ".astream",
    ".batch",
    ".abatch",
    ".run",
    ".arun",
    ".predict",
    ".apredict",
    ".kickoff",
    ".kickoff_async",
    ".initiate_chat",
    ".generate_reply",
    ".step",
    ".chat",
    ".achat",
    "BaseToolOutput",
    "LLMChain"
]

CLONE_TIMEOUT_SEC = 300
HITS_DIR = Path(__file__).parent / "hits"
MAX_HIT_FILE_BYTES = 100_000  # 100 KB — skip large files when copying out


def _on_rm_error(func, path, _):
    """Windows: git pack files are read-only; clear the flag and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def shallow_clone(url: str, dest: Path) -> bool:
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
            check=True,
            timeout=CLONE_TIMEOUT_SEC,
            capture_output=True,
        )
        return True
    except subprocess.TimeoutExpired:
        print(f"clone timed out: {url}", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode(errors="ignore").strip()
        print(f"clone failed: {url} — {stderr[:200]}", file=sys.stderr)
    return False


def repo_slug(target: str) -> str:
    """Derive a safe folder name from a URL, owner/repo slug, or local path."""
    if not target.startswith(("http://", "https://", "git@")):
        p = Path(target)
        if p.is_dir():
            return p.resolve().name
    name = target.rstrip("/")
    if name.endswith(".git"):
        name = name[:-4]
    for sep in ("://", "github.com/", "git@github.com:"):
        if sep in name:
            name = name.split(sep, 1)[1]
    return name.replace("/", "_").replace("\\", "_")


def save_hits(repo: Path, matches: list[Match], dest: Path, max_bytes: int) -> int:
    """Copy each file containing an import hit (under max_bytes) into dest/.
    Preserves the file's path inside the repo. Returns the count copied."""
    files_with_import = {m.file for m in matches if m.kind == "import"}
    saved = 0
    for rel in sorted(files_with_import):
        src = repo / rel
        try:
            if src.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        saved += 1
    return saved


def report(matches: list[Match]) -> None:
    by_file: dict[str, list[Match]] = {}
    for m in matches:
        by_file.setdefault(m.file, []).append(m)

    for file, hits in sorted(by_file.items()):
        print(f"\n{file}")
        for m in hits:
            loc = f"L{m.line}" if m.line else "  -"
            print(f"  {loc}  {m.kind:6}  {m.text}")

    print(f"\n{len(matches)} matches across {len(by_file)} files")


def scan_and_save(repo: Path, target: str) -> None:
    matches = scan_repo(repo, LLM_IMPORTS, LLM_CALLS)
    report(matches)
    dest = HITS_DIR / repo_slug(target)
    saved = save_hits(repo, matches, dest, MAX_HIT_FILE_BYTES)
    if saved:
        print(f"Saved {saved} files to {dest}")


def main(target: str) -> None:
    if not target.startswith(("http://", "https://", "git@")):
        local = Path(target)
        if local.is_dir():
            scan_and_save(local, target)
            return

    tmp = Path(tempfile.mkdtemp(prefix="wrapper-clone-"))
    try:
        print(f"cloning {target} → {tmp}")
        if not shallow_clone(target, tmp):
            sys.exit(1)
        scan_and_save(tmp, target)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, onerror=_on_rm_error)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python FindLLMWrapper.py <local_path | owner/repo | git_url>")
    main(sys.argv[1])
