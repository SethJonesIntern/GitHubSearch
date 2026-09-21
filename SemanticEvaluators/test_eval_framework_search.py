"""Unit tests for the eval-framework search (no network).

Run from repo root:
    py -3.14 -m pytest SemanticEvaluators/test_eval_framework_search.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # the module itself
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # pipeline

import eval_framework_search as efs
import GithubSearch as gs


# ── query construction: the qualifiers must NOT be in the query ───────────────
# This is the spec's load-bearing constraint. A `stars:` or `language:` qualifier
# baked into the query would drop repos at search time, where nothing can count
# them, and the corresponding waterfall bar would be unmeasurable.

def test_queries_are_the_phrases_verbatim():
    queries = efs.build_search_queries()
    assert queries == list(efs.SEARCH_PHRASES)
    assert len(queries) == 8


def test_no_stars_or_language_qualifier_emitted():
    for query in efs.build_search_queries():
        assert "stars:" not in query
        assert "language:" not in query


def test_phrase_set_is_the_spec_set():
    assert efs.SEARCH_PHRASES == [
        "LLM evaluation framework",
        "LLM eval framework",
        "evaluating LLM applications",
        "LLM testing framework",
        "LLM-as-a-judge",
        "LLM benchmarking framework",
        "LLM observability evaluation",
        "guardrails evaluation LLM",
    ]


# ── truncation flag ───────────────────────────────────────────────────────────

def test_truncated_above_the_ceiling():
    assert efs.RESULT_CEILING == 1000
    assert efs.is_truncated(1001) is True
    assert efs.is_truncated(50_000) is True


def test_not_truncated_at_or_below_the_ceiling():
    assert efs.is_truncated(1000) is False
    assert efs.is_truncated(999) is False
    assert efs.is_truncated(0) is False
    assert efs.is_truncated(None) is False


# ── dedupe across phrases ─────────────────────────────────────────────────────

def _item(full_name, **over):
    base = {"full_name": full_name, "stargazers_count": 100, "language": "Python",
            "fork": False, "archived": False, "disabled": False,
            "description": "", "topics": [], "pushed_at": "2026-01-01T00:00:00Z"}
    base.update(over)
    return base


def test_dedupe_collapses_repeats():
    out = efs.dedupe([("q1", [_item("a/one"), _item("b/two")]),
                      ("q2", [_item("a/one")])])
    assert set(out) == {"a/one", "b/two"}


def test_dedupe_keeps_every_matching_query():
    out = efs.dedupe([("q1", [_item("a/one")]),
                      ("q2", [_item("a/one")]),
                      ("q3", [_item("a/one")])])
    assert out["a/one"]["matched_queries"] == ["q1", "q2", "q3"]


def test_dedupe_does_not_repeat_a_query():
    out = efs.dedupe([("q1", [_item("a/one"), _item("a/one")])])
    assert out["a/one"]["matched_queries"] == ["q1"]


def test_dedupe_first_sighting_owns_the_payload():
    out = efs.dedupe([("q1", [_item("a/one", stargazers_count=10)]),
                      ("q2", [_item("a/one", stargazers_count=999)])])
    assert out["a/one"]["item"]["stargazers_count"] == 10


def test_dedupe_skips_items_without_a_full_name():
    out = efs.dedupe([("q1", [{"stargazers_count": 5}, _item("a/one")])])
    assert list(out) == ["a/one"]


# ── free-filter predicates at their boundaries ────────────────────────────────

def test_stars_boundary_nine_ten_eleven():
    assert efs.MIN_STARS == 10
    assert efs.passes_free_filters(_item("a/x", stargazers_count=9)) is False
    assert efs.passes_free_filters(_item("a/x", stargazers_count=10)) is True
    assert efs.passes_free_filters(_item("a/x", stargazers_count=11)) is True


def test_missing_star_count_treated_as_zero():
    assert efs.passes_free_filters(_item("a/x", stargazers_count=None)) is False


def test_non_python_dropped():
    assert efs.passes_free_filters(_item("a/x", language="TypeScript")) is False
    assert efs.passes_free_filters(_item("a/x", language=None)) is False


def test_fork_archived_disabled_dropped():
    assert efs.passes_free_filters(_item("a/x", fork=True)) is False
    assert efs.passes_free_filters(_item("a/x", archived=True)) is False
    assert efs.passes_free_filters(_item("a/x", disabled=True)) is False


# ── enrichment-tier boundaries (the thresholds gs.apply_filters enforces) ─────
# Phase B delegates to gs.apply_filters, so these pin the thresholds this module
# advertises to the ones actually applied.

def _enriched(**over):
    base = {"archived": False, "contributors_count": 5, "test_file_count": 3}
    base.update(over)
    return base


def test_contributor_boundary_one_two():
    assert efs.MIN_CONTRIBUTORS == gs.MIN_CONTRIBUTORS
    assert gs.passes_filters(_enriched(contributors_count=1)) is False
    assert gs.passes_filters(_enriched(contributors_count=2)) is True


def test_test_file_boundary_zero_one():
    assert efs.MIN_TEST_FILES == gs.MIN_TEST_FILES
    assert gs.passes_filters(_enriched(test_file_count=0)) is False
    assert gs.passes_filters(_enriched(test_file_count=1)) is True


# ── sequential attribution ────────────────────────────────────────────────────

def test_funnel_counts_sum_to_input():
    items = [_item("a/1"), _item("a/2", fork=True), _item("a/3", stargazers_count=2),
             _item("a/4", language="Go"), _item("a/5")]
    _, stats = efs.apply_free_filters(items)
    assert stats["deduped"] == len(items)
    assert (stats["dropped_fork_archived"] + stats["dropped_stars"]
            + stats["dropped_language"] + stats["kept_free"]) == len(items)


def test_first_failing_criterion_owns_the_drop():
    # a fork that ALSO has too few stars and the wrong language: charged to fork
    items = [_item("a/1", fork=True, stargazers_count=0, language="Go")]
    _, stats = efs.apply_free_filters(items)
    assert stats["dropped_fork_archived"] == 1
    assert stats["dropped_stars"] == 0
    assert stats["dropped_language"] == 0


def test_stars_owns_the_drop_over_language():
    # too few stars AND not Python: stars comes first in the funnel
    items = [_item("a/1", stargazers_count=3, language="Rust")]
    _, stats = efs.apply_free_filters(items)
    assert stats["dropped_stars"] == 1
    assert stats["dropped_language"] == 0


def test_survivors_match_the_predicate():
    items = [_item("a/1"), _item("a/2", fork=True), _item("a/3", stargazers_count=2)]
    kept, stats = efs.apply_free_filters(items)
    assert [i["full_name"] for i in kept] == ["a/1"]
    assert stats["kept_free"] == 1


# ── candidate rows ────────────────────────────────────────────────────────────

def test_confirmed_column_is_written_empty():
    row = efs.candidate_row(_item("a/x"), ["q1"], set())
    assert row["confirmed"] == ""


def test_candidate_row_columns_are_the_spec_set():
    row = efs.candidate_row(_item("a/x"), ["q1"], set())
    assert list(row) == efs.CANDIDATE_FIELDNAMES


def test_matched_queries_joined():
    row = efs.candidate_row(_item("a/x"), ["q1", "q2"], set())
    assert row["matched_queries"] == "q1 | q2"


def test_already_seeded_is_case_insensitive():
    assert efs.already_seeded("Arize-ai/phoenix") is True
    assert efs.already_seeded("ARIZE-AI/PHOENIX") is True
    assert efs.already_seeded("Giskard-AI/giskard") is True
    assert efs.already_seeded("someone/not-an-evaluator") is False


def test_already_seeded_flag_on_the_row():
    assert efs.candidate_row(_item("comet-ml/opik"), [], set())["already_seeded"] == "1"
    assert efs.candidate_row(_item("a/x"), [], set())["already_seeded"] == "0"


def test_in_frameworks_csv_flag():
    names = {"someone/langchain"}
    assert efs.candidate_row(_item("SomeOne/LangChain"), [], names)["in_frameworks_csv"] == "1"
    assert efs.candidate_row(_item("a/x"), [], names)["in_frameworks_csv"] == "0"


def test_already_seeded_from_import_names():
    assert efs.already_seeded_from_import_names("ragas;something") is True
    assert efs.already_seeded_from_import_names("mypkg;other") is False
    assert efs.already_seeded_from_import_names("") is False


# ── candidate-row re-filtering (Phase B selection) ────────────────────────────

def _crow(**over):
    base = {"full_name": "a/x", "stars": "42", "language": "Python",
            "archived": "False", "fork": "False", "confirmed": ""}
    base.update(over)
    return base


def test_candidate_row_free_filters_at_the_star_boundary():
    assert efs.candidate_passes_free_filters(_crow(stars="9")) is False
    assert efs.candidate_passes_free_filters(_crow(stars="10")) is True


def test_candidate_row_free_filters_reject_fork_and_archived():
    assert efs.candidate_passes_free_filters(_crow(fork="True")) is False
    assert efs.candidate_passes_free_filters(_crow(archived="True")) is False


def test_candidate_row_free_filters_reject_non_python():
    assert efs.candidate_passes_free_filters(_crow(language="Go")) is False


def test_candidate_row_free_filters_tolerate_junk_stars():
    assert efs.candidate_passes_free_filters(_crow(stars="")) is False


# ── confirmed summary: blank means pending, not zero ──────────────────────────

def test_confirmed_summary_none_while_blank():
    assert efs.confirmed_summary([_crow(), _crow()]) is None


def test_confirmed_summary_counts_ones():
    rows = [_crow(confirmed="1"), _crow(confirmed="0"), _crow(confirmed="1")]
    assert efs.confirmed_summary(rows) == 2


def test_confirmed_summary_zero_is_not_pending():
    assert efs.confirmed_summary([_crow(confirmed="0")]) == 0


def test_confirmed_summary_ignores_rows_the_free_filters_dropped():
    rows = [_crow(confirmed="1"), _crow(confirmed="1", stars="2")]
    assert efs.confirmed_summary(rows) == 1


# ── waterfall: pending below the enrichment boundary ──────────────────────────

PHASE_A = {"retrieved": 500, "deduped": 400, "dropped_fork_archived": 40,
           "dropped_stars": 200, "dropped_language": 100, "confirmed": None}


def test_waterfall_free_steps_are_final_after_phase_a():
    rows = {r["step"]: r for r in efs.waterfall_rows(PHASE_A)}
    assert rows["search hits retrieved"]["survivors"] == 500
    assert rows["deduped across phrases"]["survivors"] == 400
    assert rows["not fork / archived / disabled"]["survivors"] == 360
    assert rows["stars >= 10"]["survivors"] == 160
    assert rows["language == Python"]["survivors"] == 60


def test_waterfall_below_boundary_is_pending_after_phase_a():
    rows = {r["step"]: r for r in efs.waterfall_rows(PHASE_A)}
    for step in ("contributors >= 2", "test files >= 1",
                 "confirmed eval framework (hand triage)"):
        assert rows[step]["status"] == "pending"
        assert rows[step]["survivors"] is None


def test_waterfall_survivors_chain_matches_the_drops():
    rows = [r for r in efs.waterfall_rows(PHASE_A) if r["status"] == "final"]
    for prev, cur in zip(rows[1:], rows[2:]):
        assert cur["survivors"] == prev["survivors"] - cur["dropped"]


def test_waterfall_pending_propagates_forward():
    stats = {**PHASE_A, "dropped_contributors": 10}   # test-file count still missing
    rows = {r["step"]: r for r in efs.waterfall_rows(stats)}
    assert rows["contributors >= 2"]["status"] == "final"
    assert rows["contributors >= 2"]["survivors"] == 50
    assert rows["test files >= 1"]["status"] == "pending"
    assert rows["confirmed eval framework (hand triage)"]["status"] == "pending"


def test_waterfall_all_final_once_everything_exists():
    stats = {**PHASE_A, "dropped_contributors": 10, "dropped_no_tests": 5,
             "confirmed": 20}
    rows = efs.waterfall_rows(stats)
    assert all(r["status"] == "final" for r in rows)
    assert rows[-1]["survivors"] == 20
    assert rows[-1]["dropped"] == 45 - 20


def test_waterfall_confirmed_zero_is_drawn_not_pending():
    stats = {**PHASE_A, "dropped_contributors": 0, "dropped_no_tests": 0,
             "confirmed": 0}
    assert efs.waterfall_rows(stats)[-1]["status"] == "final"
