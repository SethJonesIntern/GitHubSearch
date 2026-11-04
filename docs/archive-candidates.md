# Candidates for a future archive review

**Status: candidate only. No code or historical artifact is deleted, moved, disabled,
or automatically scheduled for removal.** The owner requires another thorough check
before any deletion. A marker does not establish that a file is unused.

| Candidate | Why review it | Known dependency / required follow-up |
|---|---|---|
| `Applications/GithubSearch.py` | Older application-search implementation; current orchestrator selects `Applications/search_candidates.py` | Search all imports, scripts, docs and data-PC commands; compare selection behavior and saved result lineage. |
| `Frameworks/reformat_csv.py` | Older CSV reformatter; historical notes call it superseded | Top-level file IO: do not import to inspect. Determine whether any historical table requires this exact transformation. |
| `Frameworks/find_llm_tests.py`, `Frameworks/extract_llm_tests.py` | Older framework test discovery/extraction; overlap with current AST and focal workflows | Identify extracted-test artifacts, producer revisions and any callable helpers. These scripts may still support historical reproduction. |
| Custom matching engine in `pipeline/focal_map.py` | Mapping replaced by upstream pyMethods2Test | **Do not archive the whole module:** `observed_tests()` is still imported by current analyses, including the focal driver. Extract/migrate dependencies only in a separate reviewed change. |
| `Wrapper/package.json`, `Wrapper/package-lock.json`, tracked `Wrapper/node_modules/` | npm package named pytest is separate from the Python CI runner; no use found in inspected Python/docs references | Establish original purpose and inspect all callers. Retain files until reviewed; do not run npm install or remove dependencies as part of this task. |
| Historical determinism/knob charts and `Applications/plot_knob_count.py` | Outside the new main guardrail figure story | Preserve source and original outputs. An appendix/reproduction role may remain. |
| Stale current-state prose in `PIPELINE.md` | Predates complete runs and new guardrail framing | Mark historical; preserve architecture details that still describe working code. |

## Required evidence before a later archive or deletion

1. Search tracked references, CLI entry points, imports, dynamic path loading and tests.
2. Review data-PC scripts and notebooks, ignored files and manual reproduction commands.
3. Establish which saved results depend on each implementation and its commit revision.
4. Confirm a replacement preserves needed behavior using synthetic fixtures, not a corpus rerun.
5. Review a concrete path-by-path proposal with the owner. Archiving/moving and deletion
   are separate decisions; neither is approved by the current candidate register.

Invocation patterns, scope filters, provenance records, vendored mapper/license,
clones, slices and historical raw outputs are not disposable cleanup targets.
