"""Import-name aliases: a companion package activates its parent's patterns.

Detection gates each file on `file_imports(tree) & framework_keys`. A file doing
`from agent_framework_foundry import ...` (the Microsoft Agent Framework) used to
intersect empty, so NO patterns ran and its real invocations were never tested —
a false 0 on a repo we cloned and parsed. See EXCLUSIONS.md §9.
"""
import sys
from pathlib import Path

WRAPPER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WRAPPER))

from FrameworkDict import FRAMEWORK_CALLS, resolve_framework_imports  # noqa: E402
from transitive_invokers import index_repo, seed_invokers  # noqa: E402

KEYS = set(FRAMEWORK_CALLS)


def test_alias_resolves_to_parent():
    assert resolve_framework_imports({"agent_framework_foundry"}, KEYS) == {"agent_framework"}
    assert resolve_framework_imports({"crewai_tools"}, KEYS) == {"crewai"}


def test_plain_names_pass_through_and_non_frameworks_drop():
    assert resolve_framework_imports({"crewai", "os", "json"}, KEYS) == {"crewai"}


def test_clai_is_not_an_alias():
    """`clai` is a junk collision token, NOT pydantic-ai's CLI (EXCLUSIONS.md §9).
    Aliasing it would credit ~42 non-AI repos to pydantic_ai."""
    assert resolve_framework_imports({"clai"}, KEYS) == set()


def test_langchain_non_llm_utilities_are_not_aliases():
    """Importing a text splitter or vector store is not evidence of a model call."""
    assert resolve_framework_imports({"langchain_text_splitters", "langchain_chroma"},
                                     KEYS) == set()


def test_alias_ignored_when_parent_not_in_the_active_dict():
    """A pass driven by another pattern dict (EVAL_CALLS) must not gain crewai."""
    assert resolve_framework_imports({"crewai_tools"}, {"deepeval", "ragas"}) == set()


def test_companion_import_seeds_a_real_invoker(tmp_path):
    """End-to-end: the call that used to be walked past is now seeded."""
    pkg = tmp_path / "aliased"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "main.py").write_text(
        "from agent_framework_foundry import FoundryChatClient\n"
        "\n"
        "def ask(messages):\n"
        "    client = FoundryChatClient()\n"
        "    return client.run(messages)\n",
        encoding="utf-8",
    )
    functions, contexts = index_repo(tmp_path, tmp_path)
    seeds = seed_invokers(functions, contexts)
    assert any(q.endswith("main.ask") for q in seeds), seeds
