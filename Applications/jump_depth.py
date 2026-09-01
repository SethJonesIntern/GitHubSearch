"""How many calls separate an invoker from the function that actually hits a model?

`llm_invokers_all.csv` tags every row `direct` or `transitive` but records no depth.
It does record the BFS-tree parent: a transitive row's `reason` is `calls <qname>`.
`transitive_invokers.transitive_closure` is a multi-source BFS over the reversed call
graph — every direct invoker seeded at once, FIFO deque, `if caller not in invokers`
guard — so each node is written exactly once, on first discovery. First discovery in a
FIFO BFS from a multi-source frontier is the MINIMUM distance to the nearest seed, so
walking the parent chain recovers the true hop count. Direct invokers are depth 0.

No re-run and no clone tree needed; this is pure post-processing of the artifact.

## The orphan problem, and why the answer is a bounded range

`batch_call_metadata._invokers_rows` skips closure members that pyan resolved but our
AST pass never indexed:

    fi = functions.get(qname)
    if fi is None:
        continue        # no file/line -> no row

Their children ARE written, and point at a parent that has no row. Measured on the
current artifact: 20,876 such links (5.6% of transitive rows), and because each break
orphans its whole subtree, 21.6% of nodes cannot be resolved by chain-walking alone.

So the depth is reported two ways and the truth lies between them:

  strict      orphans dropped. Biased DEEP-ward: an orphan is by construction at least
              as deep as the broken link above it, so excluding them removes mostly
              far nodes and OVERSTATES the 3+ share.
  lower_bound every missing parent is treated as depth 0 — the most generous possible
              assumption, since a real parent is at depth >= 0. Covers 100% of nodes
              and makes every reported depth a strict floor.

Quote the lower bound. It is the defensible number.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "Applications", _ROOT / "Wrapper"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

import analyze  # noqa: E402
from pipeline import paths  # noqa: E402

BANDS = ["0", "1", "2", "3+"]


def analyzed_repos() -> set[str]:
    prog = json.loads(paths.BATCH_PROGRESS_JSON.read_text(encoding="utf-8")) \
        if paths.BATCH_PROGRESS_JSON.exists() else {}
    return set(prog.get("processed", [])) - set(analyze.EXCLUDED)


def load(source: str = "llm_invokers_all.csv") -> pd.DataFrame:
    """One row per (repo, qname), restricted to the analyzed set and run through
    analyze.py's pattern filter so this agrees with every other figure."""
    df = analyze.drop_removed_patterns_invokers(
        pd.read_csv(paths.ARTIFACTS_DIR / source,
                    usecols=["repo", "qname", "reason", "kind"]))
    return df[df["repo"].isin(analyzed_repos())].drop_duplicates(["repo", "qname"])


def depths(df: pd.DataFrame, lower_bound: bool = True) -> tuple[dict, int]:
    """{(repo, qname) -> hops from the nearest direct invoker}, and the number of
    nodes left unresolved (always 0 when lower_bound=True)."""
    parent, kind = {}, {}
    for repo, qname, reason, k in df.itertuples(index=False):
        key = (repo, qname)
        kind[key] = k
        if k == "transitive" and isinstance(reason, str) and reason.startswith("calls "):
            parent[key] = (repo, reason[6:].strip())

    depth: dict[tuple, int] = {}
    for key in kind:
        chain, cur = [], key
        while True:
            if cur in depth:
                base = depth[cur]
                break
            if kind.get(cur) == "direct":
                base = 0
                break
            if cur not in parent:               # parent has no CSV row (see docstring)
                base = 0 if lower_bound else None
                break
            if cur in chain:                    # defensive; a BFS tree is acyclic
                base = None
                break
            chain.append(cur)
            cur = parent[cur]
        if base is None:
            continue
        for i, node in enumerate(reversed(chain)):
            depth[node] = base + i + 1
        depth.setdefault(cur, base)

    # `cur` may be a MISSING parent key, which owns no row; letting it through would
    # invent depth-0 nodes and push the resolved count above the row count.
    depth = {k: v for k, v in depth.items() if k in kind}
    return depth, len(kind) - len(depth)


def depth_and_root(df: pd.DataFrame) -> tuple[dict, dict]:
    """{(repo, qname) -> hops}, {(repo, qname) -> the depth-0 ancestor it reaches}.

    A transitive row's `reason` is `calls <qname>` — it names the callee, not a
    framework. So a 1-jump test cannot say which framework it exercises; only the
    direct invoker at the end of its chain can. Walking to that root recovers it,
    which is what lets a by-framework chart show both definitions.

    Lower-bound semantics as in `depths()`: a chain that hits an unindexed parent
    stops there and is rooted on the last node it could reach.
    """
    parent, kind = {}, {}
    for repo, qname, reason, k in df.itertuples(index=False):
        key = (repo, qname)
        kind[key] = k
        if k == "transitive" and isinstance(reason, str) and reason.startswith("calls "):
            parent[key] = (repo, reason[6:].strip())

    depth, root = {}, {}
    for key in kind:
        chain, cur = [], key
        while True:
            if cur in depth:
                base, r = depth[cur], root[cur]
                break
            if kind.get(cur) == "direct" or cur not in parent or cur in chain:
                base, r = 0, cur
                break
            chain.append(cur)
            cur = parent[cur]
        for i, node in enumerate(reversed(chain)):
            depth[node], root[node] = base + i + 1, r
        depth.setdefault(cur, base)
        root.setdefault(cur, r)
    keep = set(kind)
    return ({k: v for k, v in depth.items() if k in keep},
            {k: v for k, v in root.items() if k in keep})


def banded(depth: dict) -> Counter:
    """Collapse to the reporting bands 0 / 1 / 2 / 3+."""
    out = Counter()
    for d in depth.values():
        out["3+" if d >= 3 else str(d)] += 1
    return out


def summary(source: str = "llm_invokers_all.csv") -> dict:
    """Depth is ALWAYS computed over the full invoker closure, then restricted.

    `llm_tests_all.csv` is `_tests_among(llm_invoker_rows)` — a filtered subset of the
    same rows, keeping only pytest functions. A test's BFS parent is usually a plain
    helper, not another test, so resolving chains inside the tests file alone finds
    almost no parents and reports nearly everything as one hop. That is an artifact of
    the filtering, not a measurement. The parent chain only exists in the unfiltered
    invoker graph, so build depth there and select the test qnames afterwards.
    """
    graph = load("llm_invokers_all.csv")
    lo, _ = depths(graph, lower_bound=True)
    st, orphans = depths(graph, lower_bound=False)

    if source != "llm_invokers_all.csv":
        keep = {(r, q) for r, q in load(source)[["repo", "qname"]].itertuples(index=False)}
        lo = {k: v for k, v in lo.items() if k in keep}
        st = {k: v for k, v in st.items() if k in keep}
        orphans = len(keep) - len(st)
        rows = len(keep)
    else:
        rows = len(graph)

    raw = Counter(lo.values())
    return {
        "source": source, "rows": rows,
        "lower": banded(lo), "strict": banded(st),
        "orphans": orphans, "tail": raw, "max": max(raw) if raw else 0,
        "n_lower": len(lo), "n_strict": len(st),
    }


def main():
    for src in ("llm_invokers_all.csv", "llm_tests_all.csv"):
        s = summary(src)
        n = s["n_lower"]
        print(f"\n{src}  —  {s['rows']:,} distinct (repo, qname) in the analyzed set")
        print(f"  strict resolves {s['n_strict']:,} ({100*s['n_strict']/n:.1f}%), "
              f"{s['orphans']:,} orphaned; lower bound resolves all {n:,}")
        print(f"  {'jumps':>6}{'lower bound':>14}{'share':>9}{'strict':>12}{'share':>9}")
        for b in BANDS:
            lo_, st_ = s["lower"][b], s["strict"][b]
            print(f"  {b:>6}{lo_:>14,}{100*lo_/n:>8.1f}%{st_:>12,}"
                  f"{100*st_/s['n_strict']:>8.1f}%")
        print(f"  deepest chain: {s['max']} jumps")


if __name__ == "__main__":
    main()
