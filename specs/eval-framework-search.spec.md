# Eval-framework discovery search (GitHub issue #3)

Source of truth: `SemanticEvaluators/EVAL_SEARCH_SPEC.md` (written by the repo owner).
This file translates that prose into a testable contract. Where the two appear to
disagree, the source document wins and this file is wrong — report it.

## Purpose

Produce a *candidate* list of LLM-evaluation frameworks that `EVAL_CALLS` may be missing,
plus a funnel showing what each filter cost, so Seth can judge by hand whether the
measured any-eval floor (40 of 753 apps, 5.3%) is real or an artifact of a five-name seed
list. This is **discovery only**: the script adopts nothing, infers nothing about whether
a candidate is genuinely an evaluator, and moves no statistic in the study. It runs eight
fixed natural-language phrases through GitHub repository search with **no `stars:` and no
`language:` qualifier** (both are post-hoc funnel steps, so their cost is measurable),
dedupes across phrases, writes every hit to a candidate CSV with an empty hand-triage
column, and renders a waterfall whose below-the-enrichment-boundary bars draw as
*pending* until a later, separate enrichment phase supplies them.

## Deliverables

| # | artifact | written by |
|---|---|---|
| 1 | `SemanticEvaluators/eval_framework_search.py` | coder |
| 2 | `SemanticEvaluators/test_eval_framework_search.py` | tester |
| 3 | `pipeline/paths.py`: `EVAL_FRAMEWORKS_CSV`, `EVAL_FRAMEWORKS_FILTER_STATS_JSON` | **already present** (lines 57–61); do not re-add |
| 4 | `pipeline/artifacts/figures/eval_search_waterfall.png` + `.csv` | run output |

Run commands (repo root, `py -3.14` always):

```
py -3.14 -m SemanticEvaluators.eval_framework_search --search-only
py -3.14 -m SemanticEvaluators.eval_framework_search --enrich
py -3.14 -m SemanticEvaluators.eval_framework_search --figure-only
```

---

## Public API

Module path: `SemanticEvaluators.eval_framework_search`.
`SemanticEvaluators/` has no `__init__.py`; it is a namespace package. Both the coder and
the tester assume **cwd = repo root** (`C:\Users\Seth\Documents\papers\GitHubSearch`).

The tester imports exactly this way:

```python
from SemanticEvaluators import eval_framework_search as efs
```

Importing the module must not perform network I/O, must not read
`pipeline/artifacts/*.csv`, and must not import `matplotlib`.

### Reuse contract (mandatory)

The module obtains its Stage-1 helpers by import from `Frameworks/GithubSearch.py` and
**never copies them**. It makes `Frameworks/`, the repo root and `Applications/`
importable by mutating `sys.path` at module scope, with `Frameworks/` inserted **last**
(therefore searched first) because `Applications/` also ships a module named
`GithubSearch` and the Stage-1 helpers are the `Frameworks/` ones.

```python
import GithubSearch as gs          # Frameworks/GithubSearch.py
from pipeline import paths
```

Helpers used, all via the `gs.` prefix so a test can monkeypatch them on one object:
`gs.API_BASE`, `gs.github_get`, `gs.search_repositories`, `gs.get_repo_details`,
`gs.enrich_repo`, `gs.derive_import_names`, `gs.get_contributor_count`,
`gs.get_test_metrics`, `gs.apply_filters`, `gs.FIELDNAMES`, `gs.write_csv`.

`figstyle` (from `Applications/figstyle.py`) is imported **inside** `render_waterfall`,
not at module scope, so the module imports on a machine without matplotlib.

### Module-level constants

```python
SEARCH_PHRASES: list[str] = [
    "LLM evaluation framework",
    "LLM eval framework",
    "evaluating LLM applications",
    "LLM testing framework",
    "LLM-as-a-judge",
    "LLM benchmarking framework",
    "LLM observability evaluation",
    "guardrails evaluation LLM",
]                                    # fixed, verbatim, in this order. 8 entries.

MIN_STARS: int = 10
LANGUAGE: str = "Python"
MIN_CONTRIBUTORS: int = 2
MIN_TEST_FILES: int = 1

PER_PAGE: int = 100
MAX_PAGES: int = 10
RESULT_CEILING: int = PER_PAGE * MAX_PAGES      # 1000 — GitHub's hard search cap
SEARCH_PACE_SEC: float = 2.0

SEEDED_EVALUATOR_REPOS: dict[str, str] = {      # keys LOWERCASE full_name
    "confident-ai/deepeval":      "deepeval",
    "explodinggradients/ragas":   "ragas",
    "giskard-ai/giskard":         "giskard",
    "comet-ml/opik":              "opik",
    "arize-ai/phoenix":           "phoenix",
}
SEEDED_IMPORT_NAMES: set[str] = set(SEEDED_EVALUATOR_REPOS.values())

CANDIDATES_CSV: pathlib.Path = paths.ARTIFACTS_DIR / "eval_framework_candidates.csv"
FIGURES_DIR:    pathlib.Path = paths.ARTIFACTS_DIR / "figures"
FIGURE_STEM:    str = "eval_search_waterfall"

CANDIDATE_FIELDNAMES: list[str] = [
    "full_name", "description", "stars", "language", "topics",
    "archived", "fork", "pushed_at",
    "matched_queries", "already_seeded", "in_frameworks_csv", "confirmed",
]                                    # 12 columns, this exact order

BOUNDARY_AFTER: str = f"language == {LANGUAGE}"   # == "language == Python"
STEP_LABELS: list[str] = [           # the 8 waterfall steps, this exact order
    "search hits retrieved",
    "deduped across phrases",
    "not fork / archived / disabled",
    f"stars >= {MIN_STARS}",                     # "stars >= 10"
    f"language == {LANGUAGE}",                   # "language == Python"
    f"contributors >= {MIN_CONTRIBUTORS}",       # "contributors >= 2"
    f"test files >= {MIN_TEST_FILES}",           # "test files >= 1"
    "confirmed eval framework (hand triage)",
]
```

`MIN_STARS`, `LANGUAGE`, `MIN_CONTRIBUTORS`, `MIN_TEST_FILES` are local to this module and
are **not** imported from `GithubSearch` — Stage 1's `MIN_STARS` is 1000 and its language
qualifier is baked into its query, which is exactly what this search must not do.

### Phase A — search

```python
def build_search_queries() -> list[str]: ...
```
Returns a **new list** equal to `SEARCH_PHRASES`, element-for-element, in order. No
qualifier of any kind is appended. `build_search_queries() is not SEARCH_PHRASES`.

```python
def is_truncated(total_count: int | None) -> bool: ...
```
`True` iff `(total_count or 0) > RESULT_CEILING`.

```python
def preflight(queries: Sequence[str], pace: float = SEARCH_PACE_SEC) -> list[dict]: ...
```
One `GET {gs.API_BASE}/search/repositories` per query with
`params={"q": query, "per_page": 1}` via `gs.github_get`, in input order, sleeping `pace`
seconds after each. Returns one dict per query, in input order:

```python
{"query": str, "total_count": int, "truncated": bool}
```

`total_count` is `(resp.json().get("total_count") or 0)`; `0` when `gs.github_get` returns
`None`. `truncated` is `is_truncated(total_count)`. No paging happens here.

```python
def dedupe(results: Sequence[tuple[str, Sequence[dict]]]) -> dict[str, dict]: ...
```
Input is `[(query, items), ...]` in phrase order. Returns an insertion-ordered dict keyed
by `full_name`, values:

```python
{"item": dict, "matched_queries": list[str]}
```

- First sighting owns `item`; later sightings never replace it.
- `matched_queries` holds every distinct query that surfaced the repo, in first-seen
  order, no repeats (a repo appearing twice under one query adds that query once).
- An item with a falsy/absent `full_name` is skipped entirely.

```python
def passes_free_filters(item: dict) -> bool: ...
def apply_free_filters(items: Sequence[dict]) -> tuple[list[dict], dict[str, int]]: ...
```

The three pre-enrichment filters, over **raw GitHub search-item payloads**:

1. kind: `not (item.get("fork") or item.get("archived") or item.get("disabled"))`
2. stars: `(item.get("stargazers_count") or 0) >= MIN_STARS`
3. language: `(item.get("language") or "") == LANGUAGE` (exact, case-sensitive)

`apply_free_filters` returns `(kept, stats)` where `kept` preserves input order and
`stats` has **exactly these five keys, in this order**:

```python
{"deduped": int,                 # == len(items)
 "dropped_fork_archived": int,
 "dropped_stars": int,
 "dropped_language": int,
 "kept_free": int}
```

Attribution is **sequential**: a repo is charged to the first condition it fails and to no
other. `passes_free_filters(item)` is `True` iff `apply_free_filters([item])[1]["kept_free"] == 1`.

### Candidate rows

```python
def already_seeded(full_name: str) -> bool: ...
```
Phase A only. `(full_name or "").lower() in SEEDED_EVALUATOR_REPOS`. Case-insensitive.
Never computed from import names in Phase A.

```python
def already_seeded_from_import_names(import_names: str) -> bool: ...
```
Phase B. `import_names` is the `";"`-joined string `enrich_repo` writes. Splits on `";"`,
strips each, drops empties, returns whether the set intersects `SEEDED_IMPORT_NAMES`.
Comparison is exact and case-sensitive (import names are).

```python
def load_frameworks_full_names(path: Path | None = None) -> set[str]: ...
```
Default `paths.FRAMEWORKS_CSV`. Returns the lowercased non-empty `full_name` column.
Returns `set()` when the file does not exist. Informational only — it populates a column
and never filters anything.

```python
def candidate_row(item: dict, matched_queries: Sequence[str],
                  framework_names: set[str]) -> dict: ...
```
Returns a dict whose keys are **exactly `CANDIDATE_FIELDNAMES`**:

| key | value |
|---|---|
| `full_name` | `item.get("full_name") or ""` |
| `description` | `item.get("description") or ""` |
| `stars` | `int`, `item.get("stargazers_count") or 0` |
| `language` | `item.get("language") or ""` |
| `topics` | `",".join(item.get("topics") or [])` — comma, no space |
| `archived` | `bool(item.get("archived"))` |
| `fork` | `bool(item.get("fork"))` |
| `pushed_at` | `item.get("pushed_at") or ""` |
| `matched_queries` | `" | ".join(matched_queries)` — space-pipe-space, given order |
| `already_seeded` | `"1"` if `already_seeded(full_name)` else `"0"` |
| `in_frameworks_csv` | `"1"` if `full_name.lower() in framework_names` else `"0"` |
| `confirmed` | `""` — **always**, unconditionally, never inferred |

```python
def write_candidates(rows: Sequence[dict], out_path: Path) -> None: ...
```
`csv.DictWriter(fieldnames=CANDIDATE_FIELDNAMES, quoting=csv.QUOTE_ALL)`, header row
first, `newline=""`, `encoding="utf-8"`. Creates `out_path.parent` if missing.
Rows sorted by `stars` **descending**, stable (ties keep caller order, i.e. first-seen
dedupe order). `archived`/`fork` land in the file as the literal text `True` / `False`
(Python `str(bool)`), quoted like every other cell.

**Which rows:** *every* deduped hit, unfiltered. The funnel is post-hoc, so the file keeps
the rows the free filters reject — that is what makes `archived`, `fork`, `stars` and
`language` worth columns, and what lets the `already_seeded` sanity check see a seeded
evaluator even if a filter would have dropped it.

### Triage readback

```python
def read_candidates(path: Path | None = None) -> list[dict]: ...
```
Default `CANDIDATES_CSV`. `list(csv.DictReader(...))`, `encoding="utf-8"`. Every value is
a `str`. Raises `FileNotFoundError` if the file is absent.

```python
def candidate_passes_free_filters(row: dict) -> bool: ...
```
The free filters re-applied to a **candidate CSV row** (all-string values):

- `False` if `row["archived"]` or `row["fork"]` is truthy-as-text, where truthy-as-text
  means `str(v).strip().lower() in {"1", "true", "yes"}`.
- `False` if `int(row["stars"])` `< MIN_STARS`; a non-integer or missing `stars` is
  treated as `0`.
- else `(row.get("language") or "") == LANGUAGE`.

`disabled` is not a candidate column and therefore cannot be re-checked here; Phase A
charged it at search time from the live payload.

```python
def confirmed_summary(rows: Sequence[dict]) -> int | None: ...
```
- `None` — meaning **pending, not zero** — when no row has a non-blank `confirmed`
  (whitespace-only counts as blank).
- Otherwise the count of rows where `(row.get("confirmed") or "").strip() == "1"` **and**
  `candidate_passes_free_filters(row)`.

Only the literal `"1"` is a yes. `"0"`, `""`, `"yes"`, `"y"`, `"true"` are **not**
confirmations (the asymmetry with truthy-as-text above is deliberate: the free-filter
columns are machine-written, `confirmed` is hand-written and must be unambiguous).

### Waterfall

```python
def waterfall_rows(stats: dict) -> list[dict]: ...
```

Returns **exactly 8 rows**, one per `STEP_LABELS` entry, in that order, for any input dict
including `{}`. Each row:

```python
{"step": str, "survivors": int | None, "dropped": int | None, "status": "final" | "pending"}
```

A **pending row is `{"step": <label>, "survivors": None, "dropped": None, "status": "pending"}`** —
`None`, never `0`, never a guess.

Construction:

| row | step | survivors | dropped |
|---|---|---|---|
| 0 | `search hits retrieved` | `stats["retrieved"]` | `None` (always, even when final) |
| 1 | `deduped across phrases` | `stats["deduped"]` | `retrieved - deduped` |
| 2 | `not fork / archived / disabled` | previous survivors − `stats["dropped_fork_archived"]` | `stats["dropped_fork_archived"]` |
| 3 | `stars >= 10` | previous − `stats["dropped_stars"]` | `stats["dropped_stars"]` |
| 4 | `language == Python` | previous − `stats["dropped_language"]` | `stats["dropped_language"]` |
| 5 | `contributors >= 2` | previous − `stats["dropped_contributors"]` | `stats["dropped_contributors"]` |
| 6 | `test files >= 1` | previous − `stats["dropped_no_tests"]` | `stats["dropped_no_tests"]` |
| 7 | `confirmed eval framework (hand triage)` | `stats["confirmed"]` | previous survivors − `stats["confirmed"]` |

**Pending rule, and it is sticky:** a row is `pending` iff its driving key is absent from
`stats` or is `None`, **or any earlier row is pending**. Once pending, every later row is
pending regardless of what `stats` holds. Row 0's driving key is `retrieved`; row 1's is
`deduped`; rows 2–6 are their `dropped_*` key; row 7's is `confirmed`.

Rows 5, 6 and 7 are the below-the-boundary bars. After Phase A alone they are always
pending, because Phase A writes `None` for all four enrichment-tier keys (see *Stats file*).

```python
def render_waterfall(stats: dict, out_dir: Path | None = None) -> Path: ...
```
Default `out_dir` is `FIGURES_DIR`. Writes both:

- `<out_dir>/eval_search_waterfall.csv` via `figstyle.table`, header
  `["step", "survivors", "dropped", "status"]`, one row per `waterfall_rows` entry in
  order, with `None` rendered as the **empty string** `""`.
- `<out_dir>/eval_search_waterfall.png` via `figstyle.figure` / `frame` / `round_ends` /
  `save`, using `FIGURE_STEM`.

Returns the `Path` to the **`.png`** (`figstyle.save`'s return value).

Figure rules (visual, not unit-tested — assert on the backing CSV instead):
horizontal bars, question header `"How many eval-framework candidates survive each
filter?"`, a pending row draws zero-width with the italic muted label
`pending — needs --enrich` and **no number**, a final row is labelled
`"{survivors:,}"` plus `"   (-{dropped:,})"` when `dropped` is truthy, the last `final`
row is `figstyle.EMPHASIS` and the rest `figstyle.CONTEXT`, and a dashed
`figstyle.BASELINE` rule sits immediately below the `BOUNDARY_AFTER` row labelled
`enrichment boundary`.

This figure is **not** wired into `Applications/make_figures.py`.

### Stats file

```python
def load_stats() -> dict: ...      # {} when the file does not exist
def save_stats(stats: dict) -> None: ...
```
File: `paths.EVAL_FRAMEWORKS_FILTER_STATS_JSON`
(`pipeline/artifacts/eval_frameworks_filter_stats.json`), `json.dumps(indent=2)`,
`encoding="utf-8"`.

**Phase A writes exactly these keys** (nothing merged in from a previous run — see
Decided rule D1):

```python
{"phase": "search",
 "phrases": int,                   # len(build_search_queries()) == 8
 "per_query": [ {"query": str, "total_count": int,
                 "retrieved": int, "truncated": bool}, ... ],   # phrase order
 "retrieved": int,                 # sum of per_query[].retrieved
 "deduped": int,
 "dropped_fork_archived": int,
 "dropped_stars": int,
 "dropped_language": int,
 "kept_free": int,
 "dropped_contributors": None,     # enrichment tier — explicitly cleared
 "dropped_no_tests": None,
 "kept_enriched": None,
 "confirmed": None}
```

**Phase B reloads that file and adds/overwrites:**

```python
{"phase": "enrich",
 "enriched": int,                  # gs.apply_filters stats["enriched"]
 "dropped_contributors": int,
 "dropped_no_tests": int,
 "kept_enriched": int,             # gs.apply_filters stats["kept"]
 "confirmed": int,                 # confirmed_summary(...), non-None by then
 "already_seeded_by_import_name": list[str]}   # sorted full_names
```

Phase A's free-tier keys are preserved through Phase B untouched.

### Phases

```python
def run_search(max_pages: int = MAX_PAGES, out_path: Path | None = None) -> dict: ...
```
Default `out_path` is `CANDIDATES_CSV`. Order of operations:

1. `preflight(build_search_queries())` — 8 requests — and print a per-phrase table of
   `total_count`, flagging truncated phrases, **before paging anything**.
2. For each phrase in order, `gs.search_repositories(query, max_pages=max_pages)`, sleep
   `SEARCH_PACE_SEC`, print `retrieved N / total_count reported` and the truncation flag.
3. `dedupe`, then `apply_free_filters` over `[v["item"] for v in deduped.values()]`.
4. `load_frameworks_full_names()`, then one `candidate_row` per deduped entry, then
   `write_candidates`.
5. `save_stats(...)` with the Phase-A key set above.
6. Print the free-tier funnel, then the sanity check: how many of the five seeded
   evaluators surfaced, naming each found repo and each `MISSING` one.
7. `render_waterfall(stats)`.

Returns the saved stats dict.

```python
def run_enrich(limit: int | None = None,
               candidates_path: Path | None = None,
               out_path: Path | None = None) -> dict: ...
```
Defaults: `CANDIDATES_CSV`, `paths.EVAL_FRAMEWORKS_CSV`.

1. `read_candidates(candidates_path)`; `confirmed_summary(rows)`.
2. If that is `None`: print that nothing has been triaged, **make zero API calls, write no
   CSV, write no stats**, and return `load_stats()` unchanged.
3. Targets = rows with `confirmed.strip() == "1"` **and** `candidate_passes_free_filters`,
   in file order; `[:limit]` when `limit is not None`.
4. Per target: `gs.get_repo_details(owner, repo)`; if falsy, print a skip and continue.
   Otherwise `gs.enrich_repo(dict(details, full_name=full_name), row["matched_queries"])`,
   then `time.sleep(0.2)`. Any exception is caught, printed as a skip, and the loop
   continues — one bad repo must not end the run.
   (`get_repo_details` is called first because `enrich_repo` reads `default_branch` off the
   item it is handed and the candidate CSV carries no such column.)
5. `kept, gs_stats = gs.apply_filters(enriched)`; `gs.write_csv(kept, out_path)` — so
   `eval_frameworks.csv` has **exactly `gs.FIELDNAMES`**, making it row-comparable with
   `frameworks.csv`.
6. `save_stats` with the Phase-B keys; print the enrichment funnel; `render_waterfall`.

Returns the saved stats dict.

```python
def main(argv: Sequence[str] | None = None) -> int: ...
```
`argparse`, with a **required, mutually exclusive** group:

| flag | action |
|---|---|
| `--search-only` | `run_search(max_pages=args.max_pages, out_path=args.candidates)` |
| `--enrich` | `run_enrich(limit=args.limit, candidates_path=args.candidates, out_path=args.out)` |
| `--figure-only` | `load_stats()`; if the candidates file exists, set `stats["confirmed"] = confirmed_summary(read_candidates(path))`; `render_waterfall(stats)` |

Other options: `--max-pages int` (default `MAX_PAGES`), `--limit int` (default `None`,
`--enrich` only), `--candidates Path` (default `None` → `CANDIDATES_CSV`),
`--out Path` (default `None` → `paths.EVAL_FRAMEWORKS_CSV`).

Calls `paths.ensure_dirs()` before dispatching. Returns `0`.
`if __name__ == "__main__": raise SystemExit(main())`.

---

## Behavior

| # | input | expected output | notes |
|---|---|---|---|
| 1 | `build_search_queries()` | `["LLM evaluation framework", "LLM eval framework", "evaluating LLM applications", "LLM testing framework", "LLM-as-a-judge", "LLM benchmarking framework", "LLM observability evaluation", "guardrails evaluation LLM"]` | 8 items, this order |
| 2 | `any("stars:" in q or "language:" in q for q in build_search_queries())` | `False` | the core constraint |
| 3 | `is_truncated(999)` / `is_truncated(1000)` / `is_truncated(1001)` / `is_truncated(0)` / `is_truncated(None)` | `False` / `False` / `True` / `False` / `False` | ceiling is 1000 inclusive |
| 4 | `preflight(["a","b"], pace=0)` with `gs.github_get` stubbed to return payloads `{"total_count": 5}`, `{"total_count": 2500}` | `[{"query":"a","total_count":5,"truncated":False}, {"query":"b","total_count":2500,"truncated":True}]` | 2 calls, `per_page=1` |
| 5 | `preflight(["a"], pace=0)` with `gs.github_get` → `None` | `[{"query":"a","total_count":0,"truncated":False}]` | |
| 6 | `dedupe([("q1",[A,B]), ("q2",[B,C])])`, `A={"full_name":"o/a"}` etc. | keys `["o/a","o/b","o/c"]` in that order; `out["o/b"]["matched_queries"] == ["q1","q2"]` | first sighting owns `item` |
| 7 | `dedupe([("q1",[B,B])])` | `out["o/b"]["matched_queries"] == ["q1"]` | no repeats within a query |
| 8 | `dedupe([("q1",[{"description":"x"}])])` | `{}` | no `full_name` → skipped |
| 9 | `apply_free_filters([])` | `([], {"deduped":0,"dropped_fork_archived":0,"dropped_stars":0,"dropped_language":0,"kept_free":0})` | |
| 10 | one item `{"full_name":"o/r","stargazers_count":10,"language":"Python"}` | kept; `kept_free == 1` | stars boundary, inclusive |
| 11 | same with `stargazers_count=9` | dropped; `dropped_stars == 1` | |
| 12 | same with `stargazers_count=11` | kept | |
| 13 | `{"fork":True,"stargazers_count":0,"language":"Go"}` | `dropped_fork_archived == 1`, `dropped_stars == 0`, `dropped_language == 0` | fails three; charged to the earliest |
| 14 | `{"stargazers_count":3,"language":"Go"}` | `dropped_stars == 1`, `dropped_language == 0` | fails two; charged to stars |
| 15 | `{"archived":True,...}` / `{"disabled":True,...}` | `dropped_fork_archived` each | all three kinds share one bar |
| 16 | `{"stargazers_count":50,"language":"python"}` | `dropped_language == 1` | exact, case-sensitive |
| 17 | `already_seeded("Confident-AI/DeepEval")` / `("comet-ml/opik")` / `("openai/evals")` / `("")` | `True` / `True` / `False` / `False` | case-insensitive |
| 18 | `already_seeded_from_import_names("ragas;foo")` / `("deep_eval")` / `("")` / `(None)` | `True` / `False` / `False` / `False` | exact names only |
| 19 | `load_frameworks_full_names(Path("nope.csv"))` | `set()` | missing file is not an error |
| 20 | `candidate_row({"full_name":"comet-ml/opik","stargazers_count":7,"language":"Python","topics":["llm","eval"],"description":None,"pushed_at":"2026-01-02T03:04:05Z"}, ["q1","q2"], {"langchain-ai/langchain"})` | `{"full_name":"comet-ml/opik","description":"","stars":7,"language":"Python","topics":"llm,eval","archived":False,"fork":False,"pushed_at":"2026-01-02T03:04:05Z","matched_queries":"q1 | q2","already_seeded":"1","in_frameworks_csv":"0","confirmed":""}` | exact dict |
| 21 | `candidate_row(item, ["q"], names)` for **any** item | `row["confirmed"] == ""` | never inferred |
| 22 | `write_candidates` with stars `[5, 50, 5]` (`a`,`b`,`c`) | file order `b, a, c` | stars desc, stable ties |
| 23 | `write_candidates` output header line | `"full_name","description","stars","language","topics","archived","fork","pushed_at","matched_queries","already_seeded","in_frameworks_csv","confirmed"` | `QUOTE_ALL` |
| 24 | a rejected item (`stars=0`) passed to `write_candidates` | present in the file | candidate CSV is unfiltered |
| 25 | `candidate_passes_free_filters({"archived":"False","fork":"False","stars":"10","language":"Python"})` | `True` | the round trip from row 23 |
| 26 | same with `stars:"9"` / `archived:"True"` / `fork:"true"` / `language:"Go"` / `stars:""` | `False` each | `""` → 0 |
| 27 | `confirmed_summary([{"confirmed":""},{"confirmed":"  "}])` | `None` | pending, not zero |
| 28 | `confirmed_summary([{"confirmed":"0",...}])` | `0` | triage happened, nothing confirmed |
| 29 | `confirmed_summary([{"confirmed":"1","archived":"False","fork":"False","stars":"20","language":"Python"}, {"confirmed":"1","archived":"False","fork":"False","stars":"2","language":"Python"}])` | `1` | a confirmed row still has to pass the free filters |
| 30 | `confirmed_summary([{"confirmed":"yes",...}])` | `0` | only the literal `"1"` confirms |
| 31 | `waterfall_rows({})` | 8 rows, all `{"step":<label>,"survivors":None,"dropped":None,"status":"pending"}` | `{}` fabricates nothing |
| 32 | `waterfall_rows({"retrieved":800,"deduped":600,"dropped_fork_archived":100,"dropped_stars":300,"dropped_language":150,"dropped_contributors":None,"dropped_no_tests":None,"kept_enriched":None,"confirmed":None})` | rows 0–4 `final` with survivors `800, 600, 500, 200, 50` and dropped `None, 200, 100, 300, 150`; rows 5–7 `pending` | the shape after Phase A |
| 33 | the row-32 stats plus `{"dropped_contributors":10,"dropped_no_tests":5,"confirmed":20}` | rows 0–7 all `final`; survivors `800,600,500,200,50,40,35,20`; row 7 `dropped == 15` | the shape after Phase B |
| 34 | row-32 stats but `dropped_contributors` present and `dropped_no_tests` absent, `confirmed=20` | row 5 `final`, rows 6 **and 7** `pending` | pending is sticky |
| 35 | `[r["step"] for r in waterfall_rows(anything)]` | `STEP_LABELS` | order is fixed, never data-dependent |
| 36 | `render_waterfall(stats, tmpdir)` for the row-32 stats | returns `tmpdir/"eval_search_waterfall.png"`; `tmpdir/"eval_search_waterfall.csv"` has header `step,survivors,dropped,status` and 8 rows; the 3 pending rows read `<label>,,,pending` | `None` → `""` |
| 37 | `run_enrich` on a candidates CSV whose `confirmed` is blank everywhere | prints the "nothing triaged" message; `gs.get_repo_details` / `gs.enrich_repo` / `gs.write_csv` never called; no stats written | guard before any spend |
| 38 | `run_search` end to end with `gs.github_get` + `gs.search_repositories` stubbed and `SEARCH_PACE_SEC` patched to 0 | candidate CSV row count == number of unique `full_name`s across all stubbed items; stats `retrieved` == total stubbed items; stats `confirmed is None` | |
| 39 | `main([])` | `SystemExit` with code `2` | the mode group is required |
| 40 | `main(["--search-only","--enrich"])` | `SystemExit` with code `2` | mutually exclusive |

## Errors

| condition | exception type | message contract |
|---|---|---|
| `read_candidates(p)` with `p` absent | `FileNotFoundError` | propagates from `open`; not caught |
| `main([])`, or two mode flags, or an unknown flag | `SystemExit(2)` | argparse's own text; do not assert on it |
| `--limit` / `--max-pages` given a non-integer | `SystemExit(2)` | argparse `type=int` |
| GitHub 5xx / network error inside `gs.github_get` | whatever `gs.github_get` raises after its retries (`requests.HTTPError`, `requests.RequestException`) | not re-wrapped; retry/back-off is Stage 1's job, not re-implemented here |
| GitHub secondary rate limit (403, `X-RateLimit-Remaining: 0`) | none — `gs.github_get` sleeps until reset | the reason the run takes ~3 min |
| `gs.get_repo_details` returns `None`/`{}` for a target in `run_enrich` | none | prints `  skipped <full_name>: repo details unavailable`, continues |
| any exception while enriching one target in `run_enrich` | none | caught broadly, printed as `  skipped <full_name>: <e>`, loop continues |
| `gs.github_get` returns `None` in `preflight` | none | `total_count` becomes `0` |
| `paths.FRAMEWORKS_CSV` missing | none | `load_frameworks_full_names` → `set()`; `in_frameworks_csv` is `"0"` everywhere |
| `stars` cell in a candidate row is non-numeric | none | `candidate_passes_free_filters` reads it as `0` → `False` |
| `matplotlib` not installed | `ImportError` at `render_waterfall` call time only | module import itself must still succeed |

## Boundaries

| case | answer |
|---|---|
| empty | `apply_free_filters([])` → all-zero stats, `[]`. `dedupe([])` → `{}`. `waterfall_rows({})` → 8 pending rows. `write_candidates([], p)` → a header-only file. `confirmed_summary([])` → `None` (no non-blank cell exists). Test all five. |
| zero | `stargazers_count == 0` → `dropped_stars`. `total_count == 0` → not truncated. `confirmed_summary` → `0` is a real answer and must not be confused with `None`. `dropped == 0` on a final waterfall row prints no `(-n)` suffix. Test. |
| negative | GitHub never returns negative stars. `stars == -1` is still `< MIN_STARS` → dropped; a pure-function test may assert this. A negative `--limit` slices to `[]`; **undefined, do not test**. |
| max | `total_count > 1000` sets `truncated`; the retrieved count is capped at `PER_PAGE * MAX_PAGES == 1000` per phrase and the output must say *truncated, relevance-ranked prefix*, never imply full coverage. Test `is_truncated` at 999/1000/1001 and that `per_query[i]["truncated"]` reaches the stats file. |
| unicode | Non-ASCII descriptions, topics and repo names pass through verbatim; every file is opened `encoding="utf-8"`. Test one round trip with a description like `"评估 framework — ünicode"` through `write_candidates` → `read_candidates`. |
| null | `description`/`language`/`pushed_at`/`full_name` = `None` → `""`. `topics` = `None` → `""`. `stargazers_count` = `None` → `0`. `archived`/`fork`/`disabled` = `None` → falsy. A `stats` key that is `None` → pending, identical to absent. Test each. |
| duplicate | One CSV row per `full_name`; `matched_queries` accumulates in first-seen order with no repeats; the first sighting's payload wins. Test a repo surfaced by 3 phrases and a repo listed twice by one phrase. |
| unordered | `waterfall_rows` output order is `STEP_LABELS` regardless of `stats` key order — test with a shuffled dict. Candidate CSV order is stars-descending, stable; `run_enrich` targets follow file order. `dedupe` preserves first-seen order. |
| case | `already_seeded` and `in_frameworks_csv` compare lowercased; `language` and `already_seeded_from_import_names` compare exactly; `confirmed` compares to the literal `"1"`; truthy-as-text (`archived`/`fork`) is case-insensitive over `{"1","true","yes"}`. Test one of each. |

## Invariants

1. No string returned by `build_search_queries()` contains `stars:` or `language:`, and the
   list equals `SEARCH_PHRASES` exactly. Holds for all inputs — there are none.
2. For every `items`: `stats["deduped"] == stats["dropped_fork_archived"] +
   stats["dropped_stars"] + stats["dropped_language"] + stats["kept_free"]`, and
   `len(kept) == stats["kept_free"]`. Property-test over random item lists.
3. Sequential attribution: for every item, at most one `dropped_*` counter increments.
4. `passes_free_filters(item) == (apply_free_filters([item])[1]["kept_free"] == 1)`.
5. `candidate_row(...)["confirmed"] == ""` for every input. No code path anywhere writes a
   non-empty `confirmed`.
6. `set(candidate_row(...).keys()) == set(CANDIDATE_FIELDNAMES)` and the CSV header equals
   `CANDIDATE_FIELDNAMES` in order.
7. The candidate CSV has exactly one data row per key of the `dedupe` result — no filtering.
8. `len(waterfall_rows(s)) == 8` and `[r["step"] for r in ...] == STEP_LABELS`, for every
   dict `s` including `{}`.
9. In `waterfall_rows`, `status == "pending"` ⇔ `survivors is None` ⇔ `dropped is None`,
   except row 0 whose `dropped` is `None` even when `final`.
10. Pending is monotone: if row *i* is pending, so is every row *j > i*.
11. For final rows 2–6: `survivors[i] == survivors[i-1] - dropped[i]`, and every
    `dropped >= 0`, so survivors are non-increasing down the funnel.
12. `eval_frameworks.csv` (Phase B) has exactly `gs.FIELDNAMES` as its header — the same
    columns, in the same order, as `frameworks.csv`.
13. Phase A performs `8 + (≤ 8 × max_pages)` GitHub requests and **zero** per-repo calls
    (`get_repo_details`, `get_tree`, `get_contributor_count`, `get_test_metrics` are never
    reached). Test by asserting the stubs' call counts.
14. `run_enrich` makes zero API calls while `confirmed_summary(...) is None`.
15. The module never writes to `pipeline/eval_calls.py`, `pipeline/artifacts/application_audit.csv`,
    `frameworks.csv`, or any `analysis/*.csv`; it opens `FRAMEWORKS_CSV` read-only. Its only
    writes are `eval_framework_candidates.csv`, `eval_frameworks.csv`,
    `eval_frameworks_filter_stats.json`, and the two `figures/eval_search_waterfall.*` files.

## Non-goals

- **Do not** build or test any rule that populates `confirmed`. Keyword, topic, description
  and README heuristics are all forbidden: that would be the script answering the question
  the study is asking. `confirmed` is hand-filled by Seth.
- **Do not** edit `pipeline/eval_calls.py` or `EVAL_CALLS`, re-run any pipeline stage, touch
  `in_scope` in `application_audit.csv`, or move any existing statistic.
- **Do not** wire the waterfall into `Applications/make_figures.py` — that reads `cuts.py`
  and produces the paper's F1–F5. This is not a paper figure yet.
- **Do not** copy any `GithubSearch.py` helper into this module, and do not re-implement
  retry, rate-limit or enrichment logic.
- In-house eval modules shipped by agentic frameworks (`langchain.evaluation`,
  `llama_index.evaluation`, …) are a **separate issue**. Out of scope here.
- No test may touch the network. Stub `gs.github_get`, `gs.search_repositories`,
  `gs.get_repo_details`, `gs.enrich_repo`, `gs.apply_filters`, `gs.write_csv` on the `gs`
  object; patch `SEARCH_PACE_SEC`/`pace=0` and `time.sleep` so no test is slow.
- Do not assert on PNG pixels, matplotlib artists, colours, or exact stdout formatting.
  Assert on `waterfall_rows` and on the backing `.csv`. A test may assert that a phrase and
  its `total_count` both appear somewhere in captured stdout; nothing finer.
- Do not add phrases, reword phrases, or make the phrase list configurable.
- No live acceptance run in the suite. Seth runs the search himself.

## Open questions

Each is stated above as a decided rule so coder and tester cannot diverge. Flagged here for
Seth to confirm; if any is overruled, the rule above changes and the tests change with it.

- **D1 — stats merge on re-run.** The source document says below-boundary bars "draw as
  pending until the data exists" and must never be fabricated. A naive
  `{**load_stats(), **new}` merge in Phase A would leave `dropped_contributors`,
  `dropped_no_tests`, `kept_enriched` and `confirmed` from a *previous* enrichment in the
  file, and the new waterfall would draw them as final against a different candidate set.
  **Decided:** Phase A writes those four keys as explicit `None`, clearing the enrichment
  tier. Phase B merges onto Phase A's keys as normal.
- **D2 — pending vs zero at the top of the funnel.** Undefined in the source document.
  **Decided:** `retrieved`/`deduped` absent or `None` → those rows are `pending` too, not a
  `0`-wide `final` bar. `waterfall_rows({})` is therefore 8 pending rows. Rationale: a `0`
  labelled final before any search has run is fabrication.
- **D3 — candidate CSV path.** The source document names two `paths.py` entries, neither of
  which is the candidate file. **Decided:** `eval_framework_candidates.csv` is a module-level
  constant `CANDIDATES_CSV = paths.ARTIFACTS_DIR / "eval_framework_candidates.csv"`, not a
  third `paths.py` entry, since the document is explicit that there are exactly two.
- **D4 — `confirmed` truthiness.** **Decided:** only the literal `"1"` (after `.strip()`)
  confirms; `"yes"`/`"true"`/`"y"` do not, while the machine-written `archived`/`fork`
  columns do accept `{"1","true","yes"}` case-insensitively. The asymmetry is deliberate but
  is a hand-edited column, so Seth should know the exact accepted token.
- **D5 — confirmed rows must also pass the free filters.** A hand `1` on a 3-star repo is
  not promoted. **Decided:** `confirmed_summary` and the `run_enrich` target set both
  intersect `confirmed == "1"` with `candidate_passes_free_filters`, so the last waterfall
  bar stays a subset of the bar above it (invariant 11).
- **D6 — `disabled` is not a candidate column.** The source document lists the candidate
  columns and `disabled` is not among them, yet the kind filter tests it. **Decided:**
  Phase A charges `disabled` from the live search payload; `candidate_passes_free_filters`
  (which sees only the CSV) cannot re-check it. A repo disabled between Phase A and Phase B
  is not re-detected until `enrich_repo`.
- **D7 — `--figure-only`.** Not named in the source document, which nonetheless requires
  "render the figure at any point". **Decided:** a third mode flag that redraws from the
  saved stats plus the current `confirmed` column, making no API call.
- **D8 — tie-break in the candidate CSV.** Undefined. **Decided:** stable sort on stars
  descending, so ties keep first-seen dedupe order (which is phrase order).
- **D9 — per-phrase page budget.** `MAX_PAGES = 10` × `PER_PAGE = 100` reaches GitHub's
  1000-result ceiling, giving ~88 requests. `--max-pages` exists for cheap manual runs;
  the default is the documented one.
- **Unverified (assumed, not measured):** that the 8 phrases surface all five seeded
  evaluators. The document makes this the first-run sanity check, and `already_seeded` in
  the printed summary is the one-glance test. A miss means the phrase set has a hole — a
  finding to report, not a bug to patch by adding a phrase.
