# GitHubSearch

A mining tool for identifying and analyzing AI agent frameworks and their downstream applications on GitHub. Discovers repositories matching domain-specific queries, filters them by metadata criteria, and analyzes their test suites for non-deterministic LLM calls.

## Overview

This project supports a research study on testing practices for LLM-based agents. It runs two parallel mining pipelines:

- **`Frameworks/`** — discovers agent *frameworks* (libraries people build agents with)
- **`Applications/`** — discovers *applications built on* those frameworks

Both pipelines share the same core steps:

1. **Repository discovery** via the GitHub Search API
2. **Metadata filtering** (stars, activity, fork/archive status, etc.)
3. **Test function extraction** using AST parsing
4. **LLM call detection** — identifying test functions that make real (non-mocked) calls to LLM providers or are orchestrated by LLM evaluation frameworks

The applications pipeline additionally clones each candidate locally and analyzes its filesystem directly, avoiding GitHub's per-file API rate limits.

## Setup

### Requirements

- Python 3.8+
- GitHub Personal Access Token
- `git` on PATH (applications pipeline only)

### Installation

```bash
pip install requests python-dotenv
```

### Configuration

Create a `.env` file in the project root:

```
GITHUB_TOKEN=<your_github_personal_access_token>
```

Generate a token at **GitHub > Settings > Developer settings > Personal access tokens** with `public_repo` scope.

---

## Frameworks Pipeline

Mines agent *frameworks* (e.g. LangChain, AutoGen, CrewAI).

### Search Queries

```
"AI AND agent AND framework"
"LLM-based AND agent AND framework"
"LLM AND agent AND library"
"multi-agent orchestration framework"
"LLM powered agents AND framework"
```

### Filter Criteria

| Filter Condition     | Threshold |
|----------------------|-----------|
| Star count           | >= 1000   |
| Language             | Python    |
| Contributor count    | >= 2      |
| Number of test files | >= 1      |
| Archived/Disabled    | Excluded  |

### Usage

```bash
cd Frameworks

# 1. Mine and enrich repositories (API-based enrichment)
python GithubSearch.py

# 2. Generate a condensed summary table
python reformat_csv.py

# 3. Detect real LLM calls in test functions
python find_llm_tests.py

# 4. Extract the source code of those tests
python extract_llm_tests.py
```

### Outputs

| Path                                        | Description                                                 |
|---------------------------------------------|-------------------------------------------------------------|
| `Frameworks/github_agent_framework_candidates.csv` | Full metadata for all qualifying framework repos     |
| `Frameworks/agent_framework_table.csv`      | Summary table sorted by star count                          |
| `Frameworks/llm_test_functions.csv`         | Test functions containing real LLM calls                    |
| `Frameworks/extracted_llm_tests/`           | Source code of each LLM test function, grouped by framework |

---

## Applications Pipeline

Mines *applications* built with the frameworks above. Uses the framework list as an exclusion set so the same repo never shows up in both.

### Phase 1 — Search (API-only)

```bash
cd Applications
python search_candidates.py
```

Searches GitHub once per framework keyword, dedupes, and applies every filter that can be decided from the search response alone.

### Phase 2 — Local clone + analysis (no per-file API calls)

```bash
cd Applications
python analyze_tests.py
```

For each candidate: shallow-clones the repo, walks the filesystem for test files, AST-parses each, runs the LLM-call detector, extracts matching sources, then deletes the clone. Resumable — skips repos already present in the output CSV.

### Search Query

For each of 48 framework keywords:

```
"<keyword>" language:Python stars:>10 pushed:>2025-04-14
```

### Filter Criteria

| Filter Condition                            | Threshold / Rule                                       |
|---------------------------------------------|--------------------------------------------------------|
| Language                                    | Python                                                 |
| Star count                                  | >= 10                                                  |
| Pushed since                                | 2025-04-14 (last year)                                 |
| Lifetime (`pushed_at` − `created_at`)       | >= 30 days                                             |
| Fork / Archived / Disabled                  | Excluded                                               |
| Appears in framework candidates list        | Excluded                                               |
| Contains at least one `test_*.py` file      | Required for the repo to yield any phase-2 output      |

### LLM-Call Detection (AST)

A test is marked **LLM-backed (non-deterministic)** when all three hold:

1. The enclosing file imports at least one of the tracked packages:
   - **SDKs / clients**: OpenAI, Anthropic, LiteLLM, Cohere, Mistral, Google Generative AI, Vertex AI, Boto3, LangChain (+ partner packages), LlamaIndex, Ollama, Groq, Together, HuggingFace Hub, Transformers
   - **LLM evaluation frameworks**: Opik, DeepEval, Giskard, Promptfoo, Phoenix (Arize), RAGAs
2. The function body does not match any mock pattern (`mock`, `patch`, `fake`, `stub`, `fixture`, `monkeypatch`, `MagicMock`, `AsyncMock`).
3. The function body contains a call whose unparsed form matches an LLM-invocation pattern: `chat.completions.create`, `messages.create`, `generate`, `invoke`, `complete`, `completion`, `predict`, `run`, `stream`, and their async variants (`a*`).

### Outputs

| Path                                        | Description                                                   |
|---------------------------------------------|---------------------------------------------------------------|
| `Applications/application_candidates.csv`   | Phase-1 candidate list (one row per repo)                     |
| `Applications/application_tests.csv`        | Phase-2 per-repo summary including `test_file_count`, `test_function_count`, `llm_test_count`, `clone_status` |
| `Applications/llm_test_functions.csv`       | One row per LLM-backed test: `repo, file, test_function`      |
| `Applications/extracted_llm_tests/`         | Source code of each LLM test function, one file per repo      |

---

## License

MIT
