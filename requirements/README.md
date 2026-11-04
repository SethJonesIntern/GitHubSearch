# Dependency roles

| File | Role | Validation status |
|---|---|---|
| `requirements/ci.txt` | Synthetic core-test runner | Used by Linux/Windows CI on Python 3.14. No token, Joern, clones or LFS data needed. |
| `requirements/figures.txt` | New figure builder, reading saved evidence | Matplotlib version used for the local render; run manifest records the actual Python/Matplotlib versions. New figures can render without the full research environment. |
| `requirements/analysis.in` | Explicit top-level research dependencies, derived from source imports | **Draft declaration, not a tested lockfile.** Must be checked against the working data-PC environment before recommending a fresh full-analysis install. |
| Root `requirements.txt` | Preserved historical Python 3.9 freeze with a later pyan adjustment | Not the CI lockfile or a validated modern installation recipe. Retained for provenance. |
| `Wrapper/requirements.txt` | Saved Wrapper environment package versions | Preserve as an environment snapshot. Its pytest pin differs from CI; this does not establish a single full-project environment. |
| `pipeline/third_party/pymethods2test/requirements.txt` | Published upstream dependency specification | Preserve unmodified with attribution. See CLAUDE.md for the documented modern Levenshtein wheel substitution. |
| `Wrapper/package*.json` / `node_modules` | Separate historical npm dependency | Not Python pytest. Candidate for purpose/dependency review, not deletion. |

Use a dedicated virtual environment for new tests/figures. Do not install into or
upgrade the existing data-PC environment as cleanup. Installing a test runner is
separate from running an analysis producer. Example dependency choices:

```powershell
python -m pip install -r requirements/ci.txt
# Only when creating new figures from saved results:
python -m pip install -r requirements/figures.txt
```

The full analysis targets Python 3.14 (pyan3 2.6.0); Joern and Git LFS are external
tools. A figure-only render was validated separately on Python 3.12.14. A future
analysis lock should come from an inspected working PC environment, retain upstream
tool attribution, and record platform-specific differences without overwriting old
environment snapshots.
