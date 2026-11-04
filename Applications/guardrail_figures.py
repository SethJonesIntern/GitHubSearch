"""New guardrail figures from saved inputs only, in a NEW output directory.

No research modules are imported or executed. No discovery, cloning, focal mapping,
Joern, or existing artifact generation runs. Existing output directories are refused.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import platform
import statistics
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "pipeline/artifacts"
BLUE, TEAL, INK, GREY = "#2864b7", "#157f79", "#172b46", "#586779"


def literals(path, names):
    """Read literal declarations from source text, never import the source module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id in names:
            found[target.id] = ast.literal_eval(node.value)
    if set(found) != set(names):
        raise ValueError(f"Missing expected literal declarations in {path}")
    return found


def inventory(registry_path, grouping_path):
    """Mirror ONLY the registry's two documented literal transformations.

    Package aliases share LangChain patterns; construction-only CapWords are
    filtered where method patterns exist. DSPy receiver binding is structural and
    is not represented by this regex inventory. Source hashes record provenance.
    """
    data = literals(registry_path, {"FRAMEWORK_CALLS", "_LANGCHAIN_INTEGRATIONS",
                    "_KEEP_BARE_NAMES", "_IN_SCOPE_EXPLICIT", "_IN_SCOPE_PREFIXES"})
    groups = literals(grouping_path, {"PREFIX_GROUPS", "EXTRA_MEMBERS"})
    declared = data["FRAMEWORK_CALLS"]
    effective = {k: list(v) for k, v in declared.items()}
    for package in data["_LANGCHAIN_INTEGRATIONS"]:
        effective.setdefault(package, list(effective["langchain"]))
    for package, patterns in effective.items():
        if package not in data["_KEEP_BARE_NAMES"] and any(p.startswith(".") for p in patterns):
            effective[package] = [p for p in patterns
                                  if p.startswith(".") or "." in p or p[:1].islower()]

    def group(package):
        if package in groups["EXTRA_MEMBERS"]:
            return groups["EXTRA_MEMBERS"][package]
        for prefix, label in groups["PREFIX_GROUPS"].items():
            if package == prefix or package.startswith(prefix + "_"):
                return label
        return package

    rows = []
    for package, patterns in sorted(effective.items()):
        scoped = package in data["_IN_SCOPE_EXPLICIT"] or any(
            package == p or package.startswith(p + "_") for p in data["_IN_SCOPE_PREFIXES"])
        for pattern in sorted(set(patterns)):
            rows.append({"package": package, "ecosystem": group(package), "pattern": pattern,
                         "detection_scope": int(scoped),
                         "observed_study_scope": int(scoped and package not in {"openai", "anthropic"})})
    return rows, group, {"declared_package_keys": len(declared),
                         "effective_package_keys": len(effective)}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def check_baseline(summary, repos):
    """Reject stale/mixed baseline inputs before generating a chart."""
    if summary["included_repos"] + summary["excluded_repos"] != summary["selected_repos"]:
        raise ValueError("Baseline eligibility totals disagree")
    if len({r["repo"] for r in repos}) != len(repos):
        raise ValueError("Duplicate repositories in baseline table")
    if len(repos) != summary["included_repos"]:
        raise ValueError("Baseline table and summary disagree on included repositories")
    paired = [r for r in repos if int(r["llm_methods"]) and int(r["other_methods"])]
    if len(paired) != summary["paired_repos"]:
        raise ValueError("Baseline paired repository counts disagree")
    for label, selected in (("pooled_all_complete_repos", repos), ("pooled_paired_repos", paired)):
        for group in ("llm", "other"):
            saved = summary[label][group]
            for column, field in (("methods", "methods"), ("mapped", "with_mapped_test"),
                                  ("exact_mapped", "with_exact_test")):
                if sum(int(r[f"{group}_{column}"]) for r in selected) != saved[field]:
                    raise ValueError(f"Baseline count mismatch: {label}/{group}/{field}")
            if not 0 <= saved["with_exact_test"] <= saved["with_mapped_test"] <= saved["methods"]:
                raise ValueError("Invalid method-count ordering")
            for numerator, pct in (("with_mapped_test", "pct"), ("with_exact_test", "exact_pct")):
                expected = 100 * saved[numerator] / saved["methods"]
                if abs(expected - saved[pct]) > 0.00001:
                    raise ValueError("Baseline percentage mismatch")
    for group in ("llm", "other"):
        mean = statistics.mean(100 * int(r[f"{group}_mapped"]) / int(r[f"{group}_methods"])
                               for r in paired)
        if abs(mean - summary["macro_paired_pct"][group]) > 0.00001:
            raise ValueError("Baseline repository mean mismatch")
    return paired


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_figures(output, summary, by_repo, calls, registry, group, registry_counts, plt):
    """All writes target the freshly created output directory."""
    entries = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": INK, "axes.labelcolor": INK,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})

    def canvas(title, subtitle, height=5.1):
        fig = plt.figure(figsize=(11.8, height), facecolor="white")
        fig.text(.065, .95, title, fontsize=17, weight="bold", va="top")
        fig.text(.065, .88, subtitle, fontsize=10, color=GREY, va="top")
        return fig

    def save(fig, stem, question, status, caption, table):
        for extension in ("png", "svg"):
            with (output / f"{stem}.{extension}").open("xb") as handle:
                fig.savefig(handle, format=extension, dpi=190)
        plt.close(fig)
        if table:
            write_csv(output / f"{stem}.csv", table, list(table[0]))
        entries.append({"figure": stem, "question": question, "status": status, "caption": caption})

    def frame(ax):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#d5dde7")
        ax.tick_params(length=0, pad=8)
        ax.xaxis.grid(True, color="#e8edf3")
        ax.set_axisbelow(True)

    # G1: the inventory is a contribution, not a count of validated APIs.
    patterns, packages, apps = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in registry:
        if row["observed_study_scope"]:
            patterns[row["ecosystem"]].add(row["pattern"])
            packages[row["ecosystem"]].add(row["package"])
    observed_ecosystems = {group(row["framework"]) for row in calls}
    for row in calls:
        apps[group(row["framework"])].add(row["repo"])
    if set(apps) - set(patterns):
        raise ValueError("Observed framework missing from current inventory scope")
    names = sorted(patterns, key=lambda n: (-len(patterns[n]), n))
    table = [{"ecosystem": n, "distinct_patterns": len(patterns[n]),
              "package_keys": len(packages[n]), "observed_applications": len(apps[n])} for n in names]
    fig = canvas("The invocation inventory covers framework call patterns",
                 "Contribution 1: distinct pattern strings per ecosystem, within the current study scope", 8.3)
    ax = fig.add_axes([.22, .15, .57, .65])
    ax.barh(names, [len(patterns[n]) for n in names], color=BLUE, height=.62)
    ax.invert_yaxis()
    frame(ax)
    maximum = max(map(len, patterns.values()))
    ax.set_xlim(0, maximum * 1.18)
    ax.set_xlabel("Distinct invocation pattern strings (not validated API counts)")
    for i, name in enumerate(names):
        ax.text(len(patterns[name]) + maximum * .02, i, str(len(patterns[name])), va="center")
        ax.text(1.08, i, f"{len(packages[name])} keys / {len(apps[name])} apps",
                transform=ax.get_yaxis_transform(), va="center", color=GREY, fontsize=10)
    fig.text(.83, .81, "Package keys / observed apps", fontsize=9, color=GREY)
    fig.text(.065, .045, textwrap.fill(
        f"{len(observed_ecosystems)} ecosystems have observed calls across {len({r['repo'] for r in calls}):,} apps. "
        "Packages can share patterns. Raw SDKs are excluded. Generic matches require precision review. "
        "DSPy receiver-binding logic supplements this pattern list.", 145), fontsize=9, color=GREY)
    save(fig, "G1_invocation_inventory", "What invocation patterns does the inventory cover?",
         "Existing inventory; precision/completeness validation ongoing",
         "Read directly from source literals and the documented LangChain-alias and bare-name filters, without executing the registry. "
         "Patterns are neither independent APIs nor evidence of guardrails. Full entries are in invocation_inventory.csv.", table)

    # G2: do not mix framework, app, method and call counts into one funnel.
    d = summary["diagnostics"]
    exclusions = [{"reason": "Mapping run incomplete", "applications": sum(r["status"] == "incomplete_mapping_run" for r in d)},
                  {"reason": "Source parsing failed", "applications": sum(r.get("parse_errors", 0) > 0 for r in d)},
                  {"reason": "Focal definitions unmatched", "applications": sum(r.get("unmatched_production_focals", 0) > 0 for r in d)}]
    fig = canvas(f"The paired comparison uses {summary['paired_repos']} of the {summary['selected_repos']} observed applications",
                 "Methods evidence: baseline eligibility and analysis-quality exclusions", 5.8)
    ax = fig.add_axes([.29, .40, .49, .37])
    labels = ["Observed applications", "Complete baseline outputs", "Both production-method groups"]
    values = [summary["selected_repos"], summary["included_repos"], summary["paired_repos"]]
    ax.barh(labels, values, color=["#a4b5cd", BLUE, TEAL], height=.58)
    ax.invert_yaxis(); frame(ax); ax.set_xlim(0, max(values) * 1.14)
    for i, value in enumerate(values):
        ax.text(value + 8, i, f"{value:,}", va="center", weight="bold")
    ax.set_xlabel("Applications")
    fig.text(.065, .24, f"{summary['excluded_repos']} excluded from the baseline: " + "; ".join(
        f"{r['applications']} {r['reason'].lower()}" for r in exclusions), fontsize=11)
    fig.text(.065, .16, f"{summary['included_repos'] - summary['paired_repos']} retained apps lack an eligible LLM-containing production method.", fontsize=11)
    fig.text(.065, .055, "Whole-repository exclusions are conservative. Parsing failures include templates/examples and need review.", fontsize=9, color=GREY)
    save(fig, "G2_baseline_eligibility", "Which applications contribute to the method comparison?", "Measured; exclusions under review",
         "Complete-case analysis. Repository exclusion is not evidence of missing tests. Pairing requires both method groups.",
         [{"stage": label, "applications": value} for label, value in zip(labels, values)])
    write_csv(output / "baseline_exclusions.csv", exclusions, ["reason", "applications"])

    # G3: same method numerator/denominator, with both pooling conventions explicit.
    paired = summary["pooled_paired_repos"]
    fig = canvas("LLM-containing methods have fewer identified test correspondences",
                 f"Preliminary test-presence baseline: {summary['paired_repos']} repositories with both production-method groups", 5.5)
    tables = []
    for idx, (title, vals) in enumerate((
        ("Pooled methods", [paired[g]["pct"] for g in ("llm", "other")]),
        ("Equal-weight repository means", [summary["macro_paired_pct"][g] for g in ("llm", "other")]))):
        ax = fig.add_axes([.24 + idx * .39, .30, .30, .39])
        ax.barh(["LLM-containing", "Other"], vals, color=[BLUE, TEAL], height=.5)
        ax.invert_yaxis(); frame(ax); ax.set_xlim(0, 6)
        ax.set_title(title, loc="left", fontsize=12, pad=16)
        ax.set_xlabel("Methods with ≥1 mapped test (%)")
        if idx:
            ax.set_yticklabels([])
        for i, (g, value) in enumerate(zip(("llm", "other"), vals)):
            ax.text(value + .12, i, f"{value:.2f}%", va="center", weight="bold")
            tables.append({"aggregation": title, "group": g, "pct": value,
                           "mapped_methods": paired[g]["with_mapped_test"] if not idx else "",
                           "total_methods": paired[g]["methods"] if not idx else ""})
    fig.text(.065, .17, "Pooled: " + "; ".join(f"{paired[g]['with_mapped_test']:,}/{paired[g]['methods']:,} {label}"
             for g, label in (("llm", "LLM-containing"), ("other", "other methods"))), fontsize=11)
    fig.text(.065, .105, f"Mean within-repository difference: {summary['mean_paired_difference_pp']:.2f} percentage points (LLM minus other).", fontsize=10, color=GREY)
    fig.text(.065, .045, "Heuristic correspondence only. These rates establish neither correctness checks nor execution coverage.", fontsize=9, color=GREY)
    save(fig, "G3_test_correspondence", "How often do LLM-containing production methods have a corresponding test?",
         "Measured correspondence; precision/exclusion review pending",
         "Both panels use method-based rates in the same paired repositories. Equal-weight means average each repository's rate. "
         "The comparison is descriptive and does not establish a causal or statistically significant gap.", tables)

    # G4: exact match is a naming heuristic, not manually verified ground truth.
    fig = canvas("The comparison depends on the focal-mapping rule",
                 "Sensitivity analysis in the same paired repositories and production-method denominators", 5.5)
    ax = fig.add_axes([.25, .27, .60, .48])
    table = []
    for offset, g, color in ((-.17, "llm", BLUE), (.17, "other", TEAL)):
        vals = [paired[g]["pct"], paired[g]["exact_pct"]]
        ax.barh([0 + offset, 1 + offset], vals, color=color, height=.29,
                label="LLM-containing" if g == "llm" else "Other")
        for y, value in zip([offset, 1 + offset], vals):
            ax.text(value + .10, y, f"{value:.2f}%", va="center", fontsize=11)
        for rule, field, value in zip(("Exact + fuzzy", "Exact only"), ("with_mapped_test", "with_exact_test"), vals):
            table.append({"mapping_rule": rule, "group": g, "mapped_methods": paired[g][field],
                          "total_methods": paired[g]["methods"], "pct": value})
    ax.set_yticks([0, 1], ["Exact + fuzzy", "Exact only"])
    ax.invert_yaxis(); frame(ax); ax.set_xlim(0, 6)
    ax.set_xlabel("Production methods with ≥1 mapped test (%)")
    ax.legend(frameon=False, loc="upper right")
    fig.text(.065, .095, "Exact-only restricts the tool's naming heuristic. It does not establish mapping precision or a lower bound on true testing.", fontsize=9, color=GREY)
    save(fig, "G4_mapping_sensitivity", "How sensitive are correspondence rates to fuzzy matching?", "Measured diagnostic",
         "All identified tests are eligible. Exact-only means at least one exact naming-rule match, not a manually validated test.", table)

    # G5/G6: conceptual figures carry status in title and contain no invented prevalence.
    rqs = [
        {"rq": "RQ1", "question": "How often do methods have associated\ncorrectness checks in tests or at runtime?",
         "evidence": "Test correspondence measured\nGuardrail counts pending", "role": "Primary prevalence result"},
        {"rq": "RQ2", "question": "How does test correspondence compare\nwith other production methods?",
         "evidence": "Initial baseline available\nMapping validation pending", "role": "Supporting comparison"},
        {"rq": "RQ3", "question": "What properties do checks evaluate,\nand which mechanisms do they use?",
         "evidence": "Content inspection pending\nCategories are provisional", "role": "Exploratory characterization"},
    ]
    fig = canvas("Proposed research questions and current evidence",
                 "Study design only: no runtime or correctness-guardrail prevalence results yet", 6.2)
    for i, row in enumerate(rqs):
        y = .72 - i * .22
        fig.text(.065, y, row["rq"], fontsize=19, weight="bold", color=BLUE, va="top")
        fig.text(.16, y, row["question"], fontsize=13, va="top", linespacing=1.5)
        fig.text(.70, y, row["evidence"], fontsize=11, color=GREY, va="top", linespacing=1.5)
        fig.text(.16, y - .11, row["role"], fontsize=10, color=TEAL)
        fig.add_artist(plt.Line2D([.065, .94], [y - .16, y - .16], transform=fig.transFigure, color="#dce4ed"))
    fig.text(.065, .045, "Common unit: eligible production methods containing direct observed framework LLM invocations. RQ2 adds a separate comparison group.", fontsize=9, color=GREY)
    save(fig, "G5_proposed_research_questions", "Which questions can the current evidence answer?", "PROPOSED study design, not empirical results",
         "The inventory and reproducible pipeline support all questions. Guardrail overlap (test only/runtime only/both) "
         "needs common method denominators and a separate unassessed category. Proposed RQ wording remains open for discussion.", rqs)

    dimensions = [
        {"dimension": "When", "examples": "Test suite\nRuntime", "interpretation": "Runtime checks must intervene before\nacceptance or action to count as guardrails."},
        {"dimension": "What", "examples": "Availability / structure / expected content\nBehavioral constraints / semantic correctness", "interpretation": "The property checked determines\nthe claim the check can support."},
        {"dimension": "How", "examples": "Deterministic rules or assertions\nMetrics / model-based judgments", "interpretation": "An assert can enforce a judge-score threshold.\nLibrary use does not determine check complexity."},
        {"dimension": "Response", "examples": "Fail test / reject / retry / correct\nEscalate for review", "interpretation": "Record the action on failure.\nLogging alone establishes observation."},
    ]
    fig = canvas("Proposed dimensions for characterizing correctness checks",
                 "Candidate coding scheme for manual inspection, not a validated taxonomy or observed distribution", 6.6)
    for i, row in enumerate(dimensions):
        y = .73 - i * .165
        fig.text(.065, y, row["dimension"], fontsize=14, weight="bold", color=BLUE, va="top")
        fig.text(.22, y, row["examples"], fontsize=11, linespacing=1.55, va="top")
        fig.text(.65, y, row["interpretation"], fontsize=10, color=GREY, linespacing=1.55, va="top")
        fig.add_artist(plt.Line2D([.065, .94], [y - .12, y - .12], transform=fig.transFigure, color="#dce4ed"))
    fig.text(.065, .045, "Dimensions can overlap. Validate definitions and coding reliability before reporting category frequencies.", fontsize=9, color=GREY)
    save(fig, "G6_candidate_check_dimensions", "What might distinguish kinds of correctness checks?", "PROPOSED categories, not observed findings",
         "These dimensions separate checked properties from implementation mechanisms. Simple versus complex assert "
         "is not a mutually exclusive taxonomy. Examples are illustrative and have not been identified in the corpus by this figure builder.", dimensions)
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Must not already exist; no overwrite option")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Refusing existing output directory: {output}")
    inputs = [ARTIFACTS / "method_test_baseline/summary.json",
              ARTIFACTS / "method_test_baseline/by_repo.csv",
              ARTIFACTS / "pym2t_focal_calls.csv", ROOT / "Wrapper/FrameworkDict.py",
              ROOT / "Applications/keep_frequency.py"]
    before = {str(p.relative_to(ROOT)): fingerprint(p) for p in inputs}
    summary = json.loads(inputs[0].read_text(encoding="utf-8"))
    by_repo, calls = read_csv(inputs[1]), read_csv(inputs[2])
    check_baseline(summary, by_repo)
    if len({r["repo"] for r in calls}) != summary["selected_repos"]:
        raise ValueError("Observed call scope differs from baseline scope")
    if len({r["call_id"] for r in calls}) != len(calls):
        raise ValueError("Duplicate call IDs")
    registry, group, registry_counts = inventory(inputs[3], inputs[4])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=False)
    entries = create_figures(output, summary, by_repo, calls, registry, group, registry_counts, plt)
    write_csv(output / "invocation_inventory.csv", registry, list(registry[0]))
    after = {str(p.relative_to(ROOT)): fingerprint(p) for p in inputs}
    if before != after:
        raise RuntimeError("An input changed during rendering; review this run before use")
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "builder": str(Path(__file__).relative_to(ROOT)),
                "builder_sha256": fingerprint(Path(__file__)),
                "python": platform.python_version(), "matplotlib": matplotlib.__version__,
                "inputs_sha256": before, "input_hashes_unchanged": True,
                "registry": registry_counts, "figures": entries}
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    index = ["# Guardrail study figures", "", "New figures from saved evidence. Historical figures are unchanged.",
             "G1–G4 summarize existing evidence. G5–G6 are explicitly proposed, not findings.",
             "See manifest.json for input hashes, versions, statuses and captions.", ""]
    for entry in entries:
        index += [f"## {entry['figure']}", "", f"**{entry['question']}**", "",
                  f"Status: {entry['status']}", "", f"![{entry['figure']}]({entry['figure']}.png)",
                  "", entry["caption"], ""]
    with (output / "FIGURES.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(index))
    print(f"Created {len(entries)} PNG/SVG figures and table twins in {output}")
    print("All input hashes unchanged. No existing research modules executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
