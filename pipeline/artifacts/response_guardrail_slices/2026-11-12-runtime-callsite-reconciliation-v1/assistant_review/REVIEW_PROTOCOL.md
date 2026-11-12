# Runtime call-site guardrail review protocol (assistant source review, 2026-11-12)

You are one of several blind reviewers. Your labels are ASSISTANT CLASSIFICATIONS, not human validation.
Read-only: never edit, move or delete any file outside your own output file. Never run applications,
models, Joern, pyan, pipeline producers or tests. You may read files and run read-only `grep`, `rg --no-ignore`,
`sed -n` or `cat` scoped to ONE repo directory under
`C:/Users/Seth/Documents/papers/GitHubSearch/pipeline/repos/<repo_dir>/`
(the Grep tool silently skips that gitignored tree; use Bash). Never grep the whole `pipeline/` tree (too large).
Do NOT use pyan call-graph or invoker artifacts (`llm_invokers_all.csv`, anything `transitive`, call graphs) as evidence.
Caller/helper tracing is done by reading original source only.

## Inputs

Each packet `packets/site_NNN.json` holds: the frozen call identity (`call_id` = `<repo_dir>/<rel_path>::<line>::<col>`),
saved production label (`saved_code_kind`), the framework pattern that matched, an original source excerpt with line
numbers (file hash verified equal to the Nov 4 snapshot), the Nov 4 detector's saved check rows (`saved_checks`: kind,
sink line/source, joern_status), saved routes, saved Joern slice hits and the ORIGINAL PDG line numbers of those slices
(`saved_slice_pdg_lines`). Read the full original file (`abs_path`) and bounded callers/helpers when the excerpt is
insufficient.

To find the frozen call_id of a different producer call, run
`grep -F "<repo_dir>/<rel_path>::" C:/Users/Seth/Documents/papers/GitHubSearch/pipeline/artifacts/response_guardrail_slices/2026-11-04-v1/calls.csv | cut -c1-250`.
If the producer is not in that inventory, write `not_in_frozen_inventory:<rel_path>:<line>`.

## Definitions

A qualifying runtime guardrail is: protected LLM output or proposed action -> acceptability check -> verdict/predicate ->
failure consequence that intervenes BEFORE the protected output/action is accepted, returned as success, stored,
executed or otherwise consumed (reject/raise, stop, retry/regenerate, rewrite, suppress, fallback/replace, request
correction). Logging, parsing alone, reporting a score, slice membership or a library name alone does not qualify.
An ordinary workflow branch on the output's content (e.g. "if the model said no screenshot is needed, return") is NOT
an acceptability check unless it rejects or handles an invalid or unacceptable output.

- **simple**: explicit rule-based oracle (type, presence/non-empty, equality, regex, schema/Pydantic validation, range,
  membership, deterministic domain constraint). Simple is about oracle kind, not code size.
- **complex**: a learned/model-based evaluator (LLM judge, learned scorer) judges meaning, quality, relevance,
  groundedness or safety, and its verdict actually controls the intervention. Trace evaluator input, backend, verdict
  and use. Names like critic/validate are hints only.
- **unknown**: backend, configuration, provenance or consequence cannot be established.
- Syntax does not decide class: `assert judge.grade` can be complex; `if not isinstance(x, str): raise` is simple.
- **Auxiliary plumbing**: a type/schema/presence check whose ONLY subject is a judge's own verdict object does NOT
  protect the judge call as a generation, and is not an independent simple guard on the original generation. In that
  case the judge's verdict-driven intervention is a COMPLEX guard attributed to the EARLIER GENERATION's producer call,
  and the schema check is marked auxiliary.
- If this site's LLM call is itself a judge/evaluator, say so (`site_role: judge_evaluator`) and attribute any guard
  to the generation it judges.
- **Both**: independent simple and complex checks on the same protected output.
- Python `assert` can be disabled with -O: flag `assert_optimization_sensitive`.
- Record retry-budget bypass (loop accepts the output after N failed checks), fail-open (an exception in the checker
  leads to acceptance), and configuration dependence (guard active only when a flag/setting is on).

## Context and invocation

- `actual_context`: production | test | test_support | unresolved. A production-labelled function named `test_*` that
  is an HTTP health/connectivity endpoint is still production code, but its check is a connectivity probe; say so.
  Add `context_flags` as applicable: example_or_tutorial, demo_or_sample_app, vendored_framework_copy (a copy of a
  framework's own source inside the app repo), benchmark_or_eval_harness, duplicate_copy_of_other_file.
- `invocation`: genuine_in_scope_llm (the matched call really invokes an LLM/agent through the agentic framework) |
  false_positive_non_llm (pattern matched a non-LLM operation, e.g. DB/HTTP/blockchain, or `.run` of a non-LLM
  component) | unresolved. Explain what the receiver is.

## Output

Write ONE JSON file (an array, one object per assigned site, in site order) to the path given in your task.
Each object:

```
{
 "site_no": int, "call_id": str,
 "actual_context": "...", "context_flags": [...], "context_evidence": "short, with line refs",
 "invocation": "...", "invocation_evidence": "short",
 "site_role": "generation|judge_evaluator|other|unresolved",
 "guards": [ {
   "protected_producer_call_id": "frozen call_id or not_in_frozen_inventory:...",
   "protected_output": "what output/action is protected",
   "checked_property": "...",
   "oracle_type": "simple|complex|unknown",
   "evaluator": "rule text, or judge description (backend if complex)",
   "verdict_location": "rel_path:line", "check_location": "rel_path:line", "enforcement_location": "rel_path:line",
   "failure_consequence": "...",
   "before_acceptance": "yes|no|unresolved",
   "auxiliary_judge_plumbing": true,
   "qualifies": "yes|no|unresolved",
   "nonqualifying_reason": "" or one of: not_llm_output | false_positive_invocation | test_or_test_support_context |
       workflow_branch_not_acceptability | parse_only | logging_or_reporting_only | after_acceptance |
       auxiliary_judge_plumbing | connectivity_probe | no_consequence | other:<text>,
   "joern_support": "call_and_check_in_same_saved_slice_original_lines | partial | none",
   "joern_slice_ids": [],
   "retry_budget_bypass": "yes|no|n/a|unresolved", "fail_open": "yes|no|n/a|unresolved",
   "configuration_dependent": "yes|no|unresolved", "assert_optimization_sensitive": false,
   "evidence_lines": [{"rel_path": "...", "line": 0, "line_end": 0, "role": "output|check|verdict|enforcement|support"}]
 } ],
 "site_decision": "guardrail_protects_this_site | guard_attributed_to_other_producer_only | nonqualifying | unresolved",
 "rationale": "2-4 sentences"
}
```

Include every check you examined in `guards` (including non-qualifying ones from `saved_checks`), one entry per
distinct check. A site whose invocation is false_positive_non_llm, or whose context is test/test_support, cannot have a
qualifying guard on its own output (use the matching nonqualifying_reason). `site_decision` = guardrail_protects_this_site
iff at least one guard with qualifies=yes has protected_producer_call_id equal to this site's call_id. Be conservative:
if evidence is insufficient, use unresolved.

This review is blind. Do not open or search for earlier annotation files (`*annotations*`, `verified_guardrails*`,
`rejected_*`, `unresolved_*`, `*candidates*` under 2026-11-11 or 2026-11-12 directories, AGENTS.md guardrail sections).
