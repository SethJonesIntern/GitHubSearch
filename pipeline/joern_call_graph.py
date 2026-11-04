"""Joern-CPG call graph — a resilient fallback for the (rare) repos where pyan's
graph is still empty after the exclude-and-retry pass (huge repos with many bad
files that blow the time budget).

pyan is preferred: it resolves dynamic dispatch / decorators / inheritance better
(validated 2-5x denser transitive on complex repos). Joern's Python frontend is
per-file resilient but resolves less, so its transitive counts are a LOWER BOUND —
used only when pyan gives up.

This module (a) builds a CPG for a repo, (b) exports its method-to-method call
graph, and (c) runs the transitive closure from our AST invoker seeds over it. As a
CLI it also writes human-readable methods.tsv / edges.tsv so the graph is inspectable.

  python -m pipeline.joern_call_graph --repo pipeline/repos/<slug> --out <dir>
"""
import argparse
import csv
import re
import subprocess
import sys
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths  # noqa: E402
from pipeline.create_project_codenet_cpgs import resolve_joern_parse_executable, run_joern_parse  # noqa: E402
from pipeline.slice_repo import _hide_oversized, _restore_hidden, DEFAULT_JAVA_HEAP_SIZES  # noqa: E402
from pipeline.create_project_codenet_cpgs import parse_java_heap_sizes  # noqa: E402
from pipeline.per_variable_pdg_slicer import DEFAULT_JOERN  # noqa: E402

# Emit one line per method (M) and one per resolved call edge (E). b64 avoids tab/newline
# trouble in names/paths.
_EXPORT = r'''
import io.shiftleft.semanticcpg.language._
import java.util.Base64
import java.nio.charset.StandardCharsets
def b64(s: String) = Base64.getEncoder.encodeToString(Option(s).getOrElse("").getBytes(StandardCharsets.UTF_8))
importCpg("__CPG__")
val w = new java.io.PrintWriter("__OUT__", "UTF-8")
cpg.method.foreach { m =>
  w.println("M\t"+m.id+"\t"+b64(m.name)+"\t"+b64(m.fullName)+"\t"+b64(m.filename)+"\t"+m.lineNumber.getOrElse(-1))
}
cpg.call.foreach { c => c.callee.foreach { ce => w.println("E\t"+c.method.id+"\t"+ce.id) } }
w.close()
println("JOERN_EXPORT_DONE")
'''


def _b64d(s: str) -> str:
    import base64
    try:
        return base64.b64decode(s).decode("utf-8", "replace")
    except Exception:
        return s


def _base(p: str) -> str:
    return Path(str(p).replace("\\", "/")).name


def export_call_graph(cpg_path: Path, joern: str, timeout: int = 900):
    """Run the export query on a built CPG. Returns (methods, rev_edges, by_key):
    methods[id] = (name, fullname, basefile, line); rev_edges[callee] = {callers};
    by_key[(basefile, name)] = [(id, line)] for seed alignment."""
    with tempfile.TemporaryDirectory(prefix="joern_cg_") as tmp:
        out = Path(tmp) / "cg.tsv"
        sc = Path(tmp) / "export.sc"
        sc.write_text(_EXPORT.replace("__CPG__", str(cpg_path).replace("\\", "/"))
                             .replace("__OUT__", str(out).replace("\\", "/")), encoding="utf-8")
        subprocess.run([joern, "--script", str(sc)], capture_output=True, text=True, timeout=timeout)
        methods, rev, by_key = {}, defaultdict(set), defaultdict(list)
        if not out.exists():
            return methods, rev, by_key
        for line in out.open(encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "M" and len(parts) == 6:
                mid, name, full, fn = parts[1], _b64d(parts[2]), _b64d(parts[3]), _b64d(parts[4])
                ln = int(parts[5]) if parts[5].lstrip("-").isdigit() else -1
                methods[mid] = (name, full, _base(fn), ln)
                by_key[(_base(fn), name)].append((mid, ln))
            elif parts[0] == "E" and len(parts) == 3:
                rev[parts[2]].add(parts[1])
        return methods, rev, by_key


def transitive_from_seeds(seed_rows, methods, rev, by_key):
    """Align our AST invoker seeds (rows with 'file','line','qname') to Joern method
    ids, then BFS backward (callers) to get every method transitively reaching a seed.
    Returns (transitive_fullnames, aligned_count)."""
    seed_ids, aligned = set(), 0
    for r in seed_rows:
        name = r["qname"].split(".")[-1]
        cands = by_key.get((_base(r["file"]), name), [])
        if not cands:
            continue
        try:
            sl = int(r["line"])
        except (ValueError, TypeError):
            sl = -1
        seed_ids.add(min(cands, key=lambda c: abs(c[1] - sl))[0])
        aligned += 1
    seen, q = set(seed_ids), deque(seed_ids)
    while q:
        m = q.popleft()
        for caller in rev.get(m, ()):
            if caller not in seen:
                seen.add(caller); q.append(caller)
    return {methods[m][1] for m in (seen - seed_ids) if m in methods}, aligned


def build_cpg(repo_dir: Path, joern_parse: str, cpg_path: Path,
              heaps: str = DEFAULT_JAVA_HEAP_SIZES, timeout: int = 900):
    jp = resolve_joern_parse_executable(joern_parse)
    hidden = _hide_oversized(repo_dir)
    try:
        run_joern_parse(joern_parse=jp, chunk_dir=repo_dir, cpg_path=cpg_path, timeout=timeout,
                        java_stack_sizes=[None], java_heap_sizes=parse_java_heap_sizes(heaps),
                        java_opts="")
    finally:
        _restore_hidden(hidden)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True, help="a cloned repo checkout")
    ap.add_argument("--out", type=Path, required=True, help="dir for inspectable methods/edges TSVs")
    ap.add_argument("--joern-parse", default=str(paths.REPO_ROOT).replace("\\", "/") and
                    "C:/Users/Seth/joern_install/joern-cli")
    ap.add_argument("--joern", default=DEFAULT_JOERN)
    ap.add_argument("--repo-name", default=None, help="full_name to pull invoker seeds from llm_invokers_all.csv")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="joern_cpg_") as tmp:
        cpg = Path(tmp) / "repo.cpg"
        print(f"building CPG for {args.repo.name} ...", flush=True)
        build_cpg(args.repo, args.joern_parse, cpg)
        print("exporting call graph ...", flush=True)
        methods, rev, by_key = export_call_graph(cpg, args.joern)

    intra = {mid for mid, (_n, _f, base, ln) in methods.items() if base and ln > 0}
    n_edges = sum(len(v) for v in rev.values())
    # write inspectable TSVs
    with (args.out / "methods.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["method_id", "name", "file", "line", "full_name"])
        for mid, (name, full, base, ln) in sorted(methods.items(), key=lambda x: (x[1][2], x[1][3])):
            w.writerow([mid, name, base, ln, full])
    with (args.out / "edges.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["caller", "callee"])
        for callee, callers in rev.items():
            for caller in callers:
                if caller in methods and callee in methods:
                    w.writerow([methods[caller][1], methods[callee][1]])
    print(f"\nmethods: {len(methods)} ({len(intra)} intra-repo)   resolved call edges: {n_edges}")
    print(f"wrote {args.out/'methods.tsv'} and {args.out/'edges.tsv'}")

    if args.repo_name:
        seeds = [r for r in csv.DictReader(open(paths.LLM_INVOKERS_CSV, encoding="utf-8"))
                 if r["repo"] == args.repo_name and r["kind"] == "direct"]
        trans, aligned = transitive_from_seeds(seeds, methods, rev, by_key)
        print(f"\nseeds: {len(seeds)} direct ({aligned} aligned to Joern methods)  "
              f"-> Joern transitive invokers: {len(trans)}")


if __name__ == "__main__":
    main()
