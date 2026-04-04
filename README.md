# GitHubSearch

A mining tool for identifying and analyzing AI agent frameworks on GitHub. Searches for repositories matching specific keywords, filters them by metadata criteria, and analyzes their test suites for non-deterministic LLM calls.

## Overview

This project supports a research study on testing practices in LLM-based agent frameworks. It automates the process of:

1. **Repository discovery** via the GitHub Search API using domain-specific queries
2. **Metadata filtering** (stars, contributors, language, test presence)
3. **Test function extraction** using AST parsing
4. **LLM call detection** to identify test functions that make real (non-mocked) calls to LLM providers

## Search Queries

```
"AI AND agent AND framework"
"LLM-based AND agent AND framework"
"LLM AND agent AND library"
"multi-agent orchestration framework"
"LLM powered agents AND framework"
```

## Filter Criteria

| Filter Condition        | Threshold       |
|-------------------------|-----------------|
| Star count              | >= 1000         |
| Language                | Python          |
| Contributor count       | >= 2            |
| Number of test files    | >= 1            |
| Archived/Disabled       | Excluded        |

## Setup

### Requirements

- Python 3.8+
- GitHub Personal Access Token

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

## Usage

### 1. Mine and enrich repositories

```bash
python GithubSearch.py
```

Outputs `github_agent_framework_candidates.csv` with full repository metadata including star count, contributor count, test file count, and test function count.

### 2. Generate summary table

```bash
python reformat_csv.py
```

Outputs `agent_framework_table.csv` with a condensed view: framework name, stars, contributors, test files, and test functions.

### 3. Detect real LLM calls in test functions

```bash
python find_llm_tests.py
```

Outputs `llm_test_functions.csv` listing every test function that makes a real (non-mocked) call to a supported LLM provider.

#### Supported LLM Providers

OpenAI, Anthropic, Cohere, Mistral, Groq, Together, LiteLLM, Ollama, Google Generative AI, Vertex AI, HuggingFace, LangChain, LlamaIndex

### 4. Extract LLM test function source code

```bash
python extract_llm_tests.py
```

Fetches the full source code of every test function identified in step 3 and saves them to `extracted_llm_tests/`, one `.py` file per framework. Each file contains the raw test function bodies grouped by source file.

## Output

| Path | Description |
|------|-------------|
| `github_agent_framework_candidates.csv` | Full metadata for all qualifying repositories |
| `agent_framework_table.csv` | Summary table sorted by star count |
| `llm_test_functions.csv` | Test functions containing real LLM calls |
| `extracted_llm_tests/` | Source code of each LLM test function, grouped by framework |

## License

MIT
