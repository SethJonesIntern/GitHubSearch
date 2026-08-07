"""Framework-agnostic false-positive filter for matched invocation calls.

A pattern match inside a framework-importing file can be a *collision*: the same method
name on a non-model object (`asyncio.run`, `read_file_tool.invoke`, a Mock assertion).
`classify_fp(call_text, pattern)` flags such matches by receiver identity and call syntax,
returning a tier label (or None if the call is a genuine invocation). Callers TAG the call
with this label (the `fp_tier` column) and exclude tagged calls from counts — nothing is
deleted, so the filter stays auditable/reversible.

Mirrors EXCLUSIONS.md §6. Implemented here: tiers 1-5.

Design:
  - Tier 2 (terminal-segment) is universal + purely syntactic; applied FIRST, works on any
    receiver including non-identifier `<expr>` ones.
  - Tier 1 (stdlib) tests the receiver ROOT (leftmost id) — stdlib is module-rooted.
  - Tier 3 (tool exec) tests the IMMEDIATE receiver name (segment before the method) — a
    tool can be an attribute (`self.read_file_tool.invoke`).
  - If the receiver root is a non-identifier expression (`<expr>`), receiver-based tiers
    DEFAULT-KEEP (guardrail) — e.g. `(agent or fallback()).step` is a real invocation.
"""
from __future__ import annotations

import re

# ── Tier 1 — stdlib / util receiver roots (module-rooted collisions). EXCLUSIONS §6 t1 ──
_STDLIB_UTIL_ROOTS: frozenset[str] = frozenset({
    "asyncio", "subprocess", "os", "re", "sys", "functools", "itertools",
    "logging", "threading", "multiprocessing", "json", "time", "contextlib",
    "mock", "Mock", "MagicMock", "AsyncMock", "patch", "mocker",
    "nanoid", "uuid",                       # deterministic ID libs (nanoid.generate)
})

# ── Tier 3 — tool/sandbox/driver/executor receivers = tool execution. EXCLUSIONS §6 t3 ──
# Only on these invocation methods (a bare verb shared with real model calls).
_TOOL_METHODS: frozenset[str] = frozenset({
    ".invoke", ".ainvoke", ".run", ".arun", ".stream", ".astream", ".call", ".acall",
})
_TOOL_EXACT: frozenset[str] = frozenset({
    "tool", "tools", "toolset", "sandbox", "driver", "executor",
})


def _is_tool_receiver(name: str) -> bool:
    # Carve-out: a MODEL bound with tools (`model_with_tools`, `llm_with_tools`) IS a real
    # LLM call — must be checked before the `_tools` suffix rule below.
    if name.endswith("_with_tools"):
        return False
    return (name in _TOOL_EXACT
            or name.endswith("_tool") or name.endswith("_tools")
            or name.endswith("_executor") or name.endswith("_sandbox"))


# ── Tier 4 — non-model LangChain Runnable receivers. EXCLUSIONS §6 t4 ──
# Every LCEL component implements .invoke, but templates format, retrievers retrieve,
# parsers parse — none call a model. The real model call is a SEPARATE chain/model site.
_RUNNABLE_METHODS: frozenset[str] = frozenset({".invoke", ".ainvoke"})


def _is_model_receiver(name: str) -> bool:
    """Keep-list: receivers that ARE a model/chain (a full chain includes the model)."""
    return (name in {"chain", "model", "llm", "chat"}
            or name.endswith("_chain") or name.endswith("_model")
            or name.endswith("_llm") or name.endswith("_with_tools"))


def _is_nonmodel_runnable(name: str) -> bool:
    if _is_model_receiver(name):                      # carve-out wins (prompt_model, etc.)
        return False
    return ("template" in name
            or name == "prompt" or name.endswith("_prompt")
            or "retriever" in name
            or name == "parser" or name.endswith("_parser")
            or "passthrough" in name.lower()
            or name.endswith("splitter") or name == "embedder" or name.endswith("_embedder"))


# ── Tier 5 — non-model infrastructure receivers (storage/cache/DB). EXCLUSIONS §6 t5 ──
# Generic verbs (create/batch/run_sync) on storage/cache/DB objects, not model calls.
# A targeted per-verb blocklist (so model clients `client`/`model_client` are untouched).
def _is_infra_receiver(pattern: str, name: str) -> bool:
    if pattern == ".create":
        return name in {"zep", "cache_client", "Resource"} or name.endswith("_cache")
    if pattern in (".batch", ".abatch"):
        return name in {"store", "collection"}
    if pattern == ".run_sync":
        return name == "conn"
    return False


# ── receiver / syntax helpers ────────────────────────────────────────────────────────
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _method_of(pattern: str) -> str:
    """The method token a pattern matches: '.run' -> 'run', 'chat.completions.create' and
    'Runner.run' -> themselves (dotted/Class.method patterns match as a whole)."""
    return pattern[1:] if pattern.startswith(".") else pattern


def is_terminal(call_text: str, pattern: str) -> bool:
    """True iff the matched method is the INVOKED (terminal) segment of the call expression
    — i.e. `call_text` ends with it. `endswith` (not "appears with a trailing dot") so a
    legit `self.run.run` is kept while `agent.arun.assert_called_once` is dropped."""
    method = _method_of(pattern)
    return call_text == method or call_text.endswith("." + method)


def receiver_root(call_text: str) -> str | None:
    """Leftmost identifier: 'asyncio' for asyncio.run, 'self' for self.x.run. None when the
    expression doesn't start with an identifier (`<expr>`: `(await ...)`, `[0].run`, ...)."""
    m = _IDENT.match(call_text.strip())
    return m.group(0) if m else None


def receiver_name(call_text: str, pattern: str) -> str | None:
    """The immediate receiver: last identifier segment before the (terminal) matched method.
    'self.read_file_tool.invoke' -> 'read_file_tool'; 'tool.invoke' -> 'tool'."""
    method = _method_of(pattern)
    recv = call_text[: -(len(method) + 1)] if call_text.endswith("." + method) else call_text
    ids = _IDENT.findall(recv)
    return ids[-1] if ids else None


# ── the classifier ───────────────────────────────────────────────────────────────────
def classify_fp(call_text: str, pattern: str) -> str | None:
    """Return the FP tier label for a matched (call_text, pattern), or None if it's a
    genuine invocation. `call_text` is the unparsed callable (e.g. 'agent.run')."""
    call_text = call_text.strip()

    # Tier 2 — universal, syntactic: the matched method must be the invoked terminal segment.
    if not is_terminal(call_text, pattern):
        return "tier2_nonterminal"

    root = receiver_root(call_text)
    if root is None:
        return None                       # <expr>: default-keep guardrail

    # Tier 1 — stdlib/util receiver root.
    if root in _STDLIB_UTIL_ROOTS:
        return "tier1_stdlib"

    # Tiers 3-5 key off the IMMEDIATE receiver name (segment before the method).
    rname = receiver_name(call_text, pattern)
    if rname is None:
        return None

    # Tier 3 — tool/sandbox/driver execution, on invocation methods.
    if pattern in _TOOL_METHODS and _is_tool_receiver(rname):
        return "tier3_tool_exec"

    # Tier 4 — non-model LangChain Runnable (template/retriever/parser), on invoke/ainvoke.
    if pattern in _RUNNABLE_METHODS and _is_nonmodel_runnable(rname):
        return "tier4_nonmodel_runnable"

    # Tier 5 — non-model infrastructure receiver on a generic verb.
    if _is_infra_receiver(pattern, rname):
        return "tier5_infra"

    return None
