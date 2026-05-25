"""Search a local or remote repo for LLM SDK imports and call patterns."""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from astWrappers import Match, scan_repo

# Maps each framework's top-level import name to the call patterns specific to
# that framework.  When a file imports a framework, the scanner searches only
# for that framework's patterns — reducing noise from generic names like .run.
FRAMEWORK_CALLS: dict[str, list[str]] = {
    # ── Direct LLM SDKs ────────────────────────────────────────────────────────
    "openai": [
        "chat.completions.create",
        "responses.create",
        "completions.create",
        "embeddings.create",
    ],
    "anthropic": [
        "messages.create",
        "messages.stream",
        "messages.count_tokens",
    ],

    # ── LangChain family ───────────────────────────────────────────────────────
    "langchain": [
        ".invoke",
        ".ainvoke",
        ".stream",
        ".astream",
        ".batch",
        ".abatch",
        "LLMChain",
        "ConversationChain",
        "AgentExecutor",
    ],
    "langchain_openai": [
        ".invoke",
        ".ainvoke",
        ".stream",
        ".astream",
        "ChatOpenAI",
        "OpenAI",
        "AzureChatOpenAI",
    ],
    "langchain_anthropic": [
        ".invoke",
        ".ainvoke",
        ".stream",
        ".astream",
        "ChatAnthropic",
    ],
    "langchain_community": [
        ".invoke",
        ".ainvoke",
        ".stream",
        ".astream",
    ],
    "langchain_core": [
        ".invoke",
        ".ainvoke",
        ".stream",
        ".astream",
    ],
    "langgraph": [
        ".invoke",
        ".ainvoke",
        ".stream",
        ".astream",
        "StateGraph",
        "CompiledGraph",
        "MessageGraph",
    ],

    # ── AutoGen family ─────────────────────────────────────────────────────────
    "autogen": [
        ".initiate_chat",
        ".generate_reply",
        ".a_initiate_chat",
    ],
    "autogen_core": [
        ".send_message",
        ".publish_message",
        ".register",
    ],
    "autogen_agentchat": [
        ".initiate_chat",
        ".generate_reply",
        "RoundRobinGroupChat",
        "SelectorGroupChat",
    ],

    # ── CrewAI ─────────────────────────────────────────────────────────────────
    "crewai": [
        ".kickoff",
        ".kickoff_async",
        ".kickoff_for_each",
        "Crew",
        "Agent",
        "Task",
    ],

    # ── OpenAI Swarm ───────────────────────────────────────────────────────────
    "swarm": [
        ".run",
        "Swarm",
    ],

    # ── OpenAI Agents SDK ──────────────────────────────────────────────────────
    "agents": [
        "Runner.run",
        "Runner.run_sync",
        "Runner.stream",
        "Agent",
        "handoff",
    ],

    # ── PydanticAI ─────────────────────────────────────────────────────────────
    "pydantic_ai": [
        ".run",
        ".run_sync",
        ".run_stream",
        "Agent",
    ],

    # ── MetaGPT ────────────────────────────────────────────────────────────────
    "metagpt": [
        ".run",
        ".arun",
        "Team",
        "Role",
        "Message",
    ],

    # ── CAMEL ──────────────────────────────────────────────────────────────────
    "camel": [
        ".step",
        ".chat",
        ".achat",
        "ChatAgent",
        "RolePlaying",
    ],

    # ── Griptape ───────────────────────────────────────────────────────────────
    "griptape": [
        ".run",
        "Pipeline",
        "Workflow",
        "Agent",
        "PromptTask",
    ],

    # ── AdalFlow ───────────────────────────────────────────────────────────────
    "adalflow": [
        ".call",
        ".acall",
        ".forward",
        "Generator",
        "Runner",
    ],

    # ── Agency Swarm ───────────────────────────────────────────────────────────
    "agency_swarm": [
        ".run_demo",
        ".initiate_chat",
        "Agency",
        "Agent",
    ],

    # ── Swarms ─────────────────────────────────────────────────────────────────
    "swarms": [
        ".run",
        ".arun",
        "Swarm",
        "Agent",
        "SequentialWorkflow",
    ],

    # ── Parlant ────────────────────────────────────────────────────────────────
    "parlant": [
        ".run",
        ".arun",
        "Agent",
        "Session",
    ],

    # ── Dynamiq ────────────────────────────────────────────────────────────────
    "dynamiq": [
        ".run",
        ".arun",
        "Workflow",
        "Agent",
    ],

    # ── LiveKit Agents ─────────────────────────────────────────────────────────
    "livekit": [
        ".run",
        ".arun",
        "WorkerOptions",
        "JobProcess",
    ],

    # ── TEN Framework ──────────────────────────────────────────────────────────
    "ten": [
        ".run",
        ".start",
        "TenEnv",
        "Extension",
    ],

    # ── BeeAI Framework ────────────────────────────────────────────────────────
    "beeai_framework": [
        ".run",
        ".stream",
        "BeeAgent",
        "ReActAgent",
    ],

    # ── PraisonAI ──────────────────────────────────────────────────────────────
    "praisonai": [
        ".start",
        ".run",
        "PraisonAI",
        "Agent",
    ],

    # ── SuperAGI ───────────────────────────────────────────────────────────────
    "superagi": [
        ".run",
        ".execute_next_action",
        "SuperAgi",
    ],

    # ── RagaAI Catalyst ────────────────────────────────────────────────────────
    "ragaai_catalyst": [
        ".run",
        ".evaluate",
        "RagaAICatalyst",
        "Tracer",
    ],

    # ── AgentUniverse ──────────────────────────────────────────────────────────
    "agentuniverse": [
        ".run",
        ".arun",
        "AgentManager",
    ],

    # ── Agent Squad (AWS) ──────────────────────────────────────────────────────
    "agent_squad": [
        ".route_request",
        ".process_request",
        "Orchestrator",
    ],

    # ── AgentOps ───────────────────────────────────────────────────────────────
    "agentops": [
        ".init",
        ".start_session",
        ".end_session",
        ".record",
    ],

    # ── OpenLIT ────────────────────────────────────────────────────────────────
    "openlit": [
        ".init",
        ".trace",
    ],

    # ── Giskard ────────────────────────────────────────────────────────────────
    "giskard": [
        ".scan",
        ".evaluate",
        "Model",
        "Dataset",
    ],

    # ── SuperDuper ─────────────────────────────────────────────────────────────
    "superduper": [
        ".predict",
        ".fit",
        ".apply",
    ],

    # ── ii-agent ───────────────────────────────────────────────────────────────
    "ii_agent": [
        ".run",
        ".execute",
        "IIAgent",
    ],

    # ── LaVague ────────────────────────────────────────────────────────────────
    "lavague": [
        ".run",
        ".execute",
        "WebAgent",
        "ActionEngine",
    ],

    # ── Cheshire Cat ───────────────────────────────────────────────────────────
    "cheshire_cat": [
        ".run",
        ".send",
        "CatClient",
    ],

    # ── Solace Agent Mesh ──────────────────────────────────────────────────────
    "solace_agent_mesh": [
        ".run",
        ".publish",
        "SolaceAgentMesh",
    ],

    # ── Misc remaining frameworks ──────────────────────────────────────────────
    "lagent": [".run", ".step", "ActionExecutor"],
    "patchwork": [".run", "PatchFlow"],
    "npcpy": [".run", ".chat", "NPC"],
    "any_agent": [".run", "AnyAgent"],
    "sage": [".run", ".query", "Sage"],
    "honcho": [".create", ".get", "Honcho"],
    "uagents": [".run", ".send", "Agent", "Bureau"],
    "agent_protocol": [".run", ".step", "Agent"],
    "infiagent": [".run", "InfiAgent"],
    "notte": [".run", "Notte"],
    "redamon": [".run"],
    "agent_framework": [".run", ".execute"],
    "llmstack": [".run", "LLMStack"],
    "nextpy": [".run", "App"],
}


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
            print(f"  {loc}  {m.kind:6}  [{m.framework}]  {m.text}")

    print(f"\n{len(matches)} matches across {len(by_file)} files")


def scan_and_save(repo: Path, target: str) -> None:
    matches = scan_repo(repo, FRAMEWORK_CALLS)
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
