#!/usr/bin/env python3
"""Create one Joern CPG per ProjectCodeNet_1k_chunks directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


CHUNK_RE = re.compile(r"^(?P<start>\d+)k-(?P<end>\d+)k$")


def slurm_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def default_slurm_nodes() -> int:
    return slurm_int("SLURM_NNODES", slurm_int("SLURM_JOB_NUM_NODES", 1))


def default_cpus_per_task() -> int:
    return slurm_int("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)


def parse_chunk_index(path: Path) -> int | None:
    match = CHUNK_RE.match(path.name)
    if match:
        return int(match.group("start"))
    return None


def cpg_name_for_chunk(chunk_name: str) -> str:
    return f"{chunk_name.replace('-', '_')}.cpg"


def completed_cpg(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def cpg_sidecar_path(cpg_path: Path) -> Path:
    return cpg_path.with_suffix(cpg_path.suffix + ".manifest.json")


def count_python_files(chunk_dir: Path) -> int:
    return sum(1 for _ in chunk_dir.glob("*.py"))


def validate_chunk_dir(chunk_dir: Path, files_per_dir: int, final_chunk_index: int) -> dict:
    if not chunk_dir.is_dir():
        raise RuntimeError(f"Missing chunk directory: {chunk_dir}")

    chunk_index = parse_chunk_index(chunk_dir)
    if chunk_index is None:
        raise RuntimeError(f"Invalid chunk directory name: {chunk_dir.name}")

    py_count = count_python_files(chunk_dir)
    if py_count <= 0:
        raise RuntimeError(f"Chunk {chunk_dir} contains no .py files")
    if py_count != files_per_dir and chunk_index != final_chunk_index:
        raise RuntimeError(
            f"Chunk {chunk_dir} has {py_count} .py files, expected {files_per_dir}. "
            "Only the final available chunk is allowed to be partial."
        )
    if py_count > files_per_dir:
        raise RuntimeError(
            f"Chunk {chunk_dir} has {py_count} .py files, expected at most {files_per_dir}"
        )

    global_index_start = chunk_index * files_per_dir
    global_index_end_exclusive = global_index_start + py_count
    return {
        "chunk_name": chunk_dir.name,
        "chunk_dir": str(chunk_dir),
        "program_count": py_count,
        "global_index_start": global_index_start,
        "global_index_end_exclusive": global_index_end_exclusive,
    }


def completed_cpg_for_chunk(cpg_path: Path, chunk_dir: Path, chunk_manifest: dict) -> bool:
    if not completed_cpg(cpg_path):
        return False
    sidecar = load_json(cpg_sidecar_path(cpg_path))
    if sidecar is None:
        return False
    return (
        sidecar.get("status") == "created"
        and sidecar.get("chunk_name") == chunk_dir.name
        and sidecar.get("chunk_dir") == str(chunk_dir)
        and sidecar.get("program_count") == chunk_manifest.get("program_count")
        and sidecar.get("global_index_start") == chunk_manifest.get("global_index_start")
        and sidecar.get("global_index_end_exclusive")
        == chunk_manifest.get("global_index_end_exclusive")
    )


def _executable_variants(p: Path):
    """On Windows, prefer the .bat/.cmd/.exe wrapper over the extension-less Unix
    launcher script — running the latter via subprocess raises
    'WinError 193: %1 is not a valid Win32 application'."""
    if os.name == "nt" and not p.suffix:
        for ext in (".bat", ".cmd", ".exe"):
            yield p.with_suffix(ext)
    yield p


def resolve_joern_parse_executable(value: str) -> Path:
    """Resolve joern-parse from a binary name, binary path, joern-cli dir, or install root."""
    raw = Path(value).expanduser()
    candidates = []

    if raw.stem == "joern-parse" or raw.suffix:
        candidates.append(raw)
    if raw.is_dir():
        candidates.extend([raw / "joern-parse", raw / "joern-cli" / "joern-parse"])
    else:
        candidates.append(raw / "joern-cli" / "joern-parse")

    found_value = shutil.which(value)
    if found_value:
        candidates.append(Path(found_value))
    found_default = shutil.which("joern-parse")
    if found_default:
        candidates.append(Path(found_default))

    seen = set()
    for candidate in candidates:
        for cand in _executable_variants(candidate):
            cand = cand.resolve() if cand.exists() else cand
            if cand in seen:
                continue
            seen.add(cand)
            if cand.is_file():
                return cand

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find joern-parse. Pass --joern-parse as the binary path, "
        "the joern-cli directory, or the Joern install root. "
        f"Checked: {checked}"
    )


def _format_joern_failure(detail: str, max_chars: int = 4000) -> str:
    detail = detail.strip()
    if len(detail) <= max_chars:
        return detail
    # Keep BOTH ends: the head has the startup context, but the fatal exception
    # (OutOfMemoryError / StackOverflowError / stack trace) is at the TAIL — which
    # a head-only truncation would discard, leaving failures undiagnosable.
    head = max_chars // 3
    tail = max_chars - head
    return (f"{detail[:head]}\n... <truncated {len(detail) - max_chars} chars> ...\n"
            f"{detail[-tail:]}")


def _java_env(stack_size: str | None, heap_size: str | None,
              extra_java_opts: str) -> dict[str, str]:
    env = os.environ.copy()
    opts = []
    if heap_size:
        opts.append(f"-Xmx{heap_size}")
    if stack_size:
        opts.append(f"-Xss{stack_size}")
    if extra_java_opts.strip():
        opts.append(extra_java_opts.strip())
    if not opts:
        return env

    extra = " ".join(opts)
    for key in ("JAVA_OPTS", "_JAVA_OPTIONS"):
        existing = env.get(key, "").strip()
        env[key] = f"{existing} {extra}".strip()
    return env


def run_joern_parse_once(
    *,
    joern_parse: Path,
    chunk_dir: Path,
    cpg_path: Path,
    timeout: int,
    java_stack_size: str | None,
    java_opts: str,
    java_heap_size: str | None = None,
) -> float:
    if cpg_path.exists():
        cpg_path.unlink()

    start = time.monotonic()
    cmd = [
        str(joern_parse),
        str(chunk_dir),
        "--language",
        "PYTHONSRC",
        "-o",
        str(cpg_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_java_env(java_stack_size, java_heap_size, java_opts),
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        if cpg_path.exists():
            cpg_path.unlink()
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        detail = _format_joern_failure((stderr or stdout or "").strip())
        raise RuntimeError(
            f"joern-parse timed out after {timeout} seconds for {chunk_dir} "
            f"(elapsed {elapsed:.3f}s). Partial CPG removed: {cpg_path}. {detail}"
        ) from exc

    elapsed = time.monotonic() - start
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if cpg_path.exists():
            cpg_path.unlink()
        raise RuntimeError(f"joern-parse failed for {chunk_dir}: {_format_joern_failure(detail)}")
    if not completed_cpg(cpg_path):
        if cpg_path.exists():
            cpg_path.unlink()
        raise RuntimeError(f"joern-parse finished but did not create {cpg_path}")
    return elapsed


def run_joern_parse(
    *,
    joern_parse: Path,
    chunk_dir: Path,
    cpg_path: Path,
    timeout: int,
    java_stack_sizes: list[str | None],
    java_opts: str,
    java_heap_sizes: list[str | None] | None = None,
) -> tuple[float, str | None, int]:
    """Parse a repo into a CPG, escalating JVM memory only as failures demand it:
    a larger HEAP (-Xmx) on OutOfMemoryError and a larger STACK (-Xss) on
    StackOverflowError. Most repos succeed on the first (smallest) settings; only
    the pathologically large ones ever climb the ladders — and each joern process
    is short-lived, so the bigger memory is transient, never held across the run."""
    heaps = list(java_heap_sizes) if java_heap_sizes else [None]
    stacks = list(java_stack_sizes) if java_stack_sizes else [None]
    last_error: Exception | None = None
    hi = si = 0
    attempt = 0
    while True:
        attempt += 1
        heap, stack = heaps[hi], stacks[si]
        print(
            f"  joern-parse attempt {attempt} "
            f"(heap {heap or 'default'}, stack {stack or 'default'})",
            flush=True,
        )
        try:
            elapsed = run_joern_parse_once(
                joern_parse=joern_parse,
                chunk_dir=chunk_dir,
                cpg_path=cpg_path,
                timeout=timeout,
                java_stack_size=stack,
                java_opts=java_opts,
                java_heap_size=heap,
            )
            return elapsed, stack, attempt
        except RuntimeError as exc:
            last_error = exc
            message = str(exc)
            if "OutOfMemoryError" in message and hi < len(heaps) - 1:
                hi += 1
                print(f"  OutOfMemoryError; retrying with a larger heap ({heaps[hi]})",
                      flush=True)
                continue
            if "StackOverflowError" in message and si < len(stacks) - 1:
                si += 1
                print(f"  StackOverflowError; retrying with a larger stack ({stacks[si]})",
                      flush=True)
                continue
            break

    if cpg_path.exists():
        cpg_path.unlink()
    raise RuntimeError(str(last_error) if last_error is not None else f"joern-parse failed for {chunk_dir}")


def select_chunk_dirs(
    input_dir: Path,
    start_dir_index: int,
    numofdirs: int | None,
) -> list[Path]:
    indexed_chunks = {
        index: path
        for path in input_dir.iterdir()
        if path.is_dir() and (index := parse_chunk_index(path)) is not None
    }
    if numofdirs is None:
        return [
            path
            for index, path in sorted(indexed_chunks.items())
            if index >= start_dir_index
        ]

    range_end_index = start_dir_index + numofdirs
    missing = [
        index
        for index in range(start_dir_index, range_end_index)
        if index not in indexed_chunks
    ]
    if missing:
        preview = ", ".join(str(item) for item in missing[:10])
        raise SystemExit(
            f"Missing {len(missing)} chunk directories in requested range. "
            f"First missing chunk index(es): {preview}"
        )
    return [indexed_chunks[index] for index in range(start_dir_index, range_end_index)]


def final_available_chunk_index(input_dir: Path) -> int:
    indexes = [
        index
        for path in input_dir.iterdir()
        if path.is_dir() and (index := parse_chunk_index(path)) is not None
    ]
    if not indexes:
        raise SystemExit(f"No chunk directories found under {input_dir}")
    return max(indexes)


def parse_java_stack_sizes(value: str) -> list[str | None]:
    java_stack_sizes = [
        None if item.strip().lower() in {"", "default", "none"} else item.strip()
        for item in value.split(",")
    ]
    return java_stack_sizes or [None]


def parse_java_heap_sizes(value: str) -> list[str | None]:
    """Parse a comma list like '8g,12g,16g' into the -Xmx escalation ladder."""
    heaps = [
        None if item.strip().lower() in {"", "default", "none"} else item.strip()
        for item in value.split(",")
    ]
    return heaps or [None]


def shard_chunk_dirs(chunk_dirs: list[Path], shard_id: int, num_shards: int) -> list[Path]:
    if shard_id < 0 or shard_id >= num_shards:
        raise SystemExit(f"Invalid shard id {shard_id}; expected 0 <= shard_id < {num_shards}")
    return [
        chunk_dir
        for position, chunk_dir in enumerate(chunk_dirs)
        if position % num_shards == shard_id
    ]


def process_chunk_dirs(
    *,
    args: argparse.Namespace,
    joern_parse: Path,
    chunk_dirs: list[Path],
    final_chunk_index: int,
    manifest_path: Path,
    worker_label: str,
) -> dict:
    total_started_at = time.monotonic()
    java_stack_sizes = parse_java_stack_sizes(args.java_stack_sizes)

    print(f"{worker_label} Input chunks: {args.input_dir}", flush=True)
    print(f"{worker_label} Output CPGs: {args.output_dir}", flush=True)
    print(f"{worker_label} Joern parser: {joern_parse}", flush=True)
    print(f"{worker_label} Selected chunk directories: {len(chunk_dirs)}", flush=True)
    print(f"{worker_label} Final available chunk index: {final_chunk_index}", flush=True)

    summaries = []
    timed_runs = []
    for position, chunk_dir in enumerate(chunk_dirs, start=1):
        chunk_manifest = validate_chunk_dir(chunk_dir, args.files_per_dir, final_chunk_index)
        cpg_path = args.output_dir / cpg_name_for_chunk(chunk_dir.name)
        if completed_cpg_for_chunk(cpg_path, chunk_dir, chunk_manifest) and not args.overwrite:
            print(f"{worker_label} [{position}/{len(chunk_dirs)}] Skipping existing {cpg_path}", flush=True)
            summaries.append(
                {
                    "chunk": chunk_dir.name,
                    "cpg": str(cpg_path),
                    "status": "skipped",
                    "seconds": 0.0,
                    **chunk_manifest,
                }
            )
            continue

        print(
            f"{worker_label} [{position}/{len(chunk_dirs)}] Parsing {chunk_dir} "
            f"({chunk_manifest['program_count']} files) -> {cpg_path}",
            flush=True,
        )
        elapsed, java_stack_size, attempts = run_joern_parse(
            joern_parse=joern_parse,
            chunk_dir=chunk_dir,
            cpg_path=cpg_path,
            timeout=args.timeout,
            java_stack_sizes=java_stack_sizes,
            java_opts=args.java_opts,
        )
        sidecar = {
            "status": "created",
            **chunk_manifest,
            "cpg": str(cpg_path),
            "seconds": round(elapsed, 3),
            "size_bytes": cpg_path.stat().st_size,
            "java_stack_size": java_stack_size or "default",
            "attempts": attempts,
            "worker_label": worker_label,
        }
        cpg_sidecar_path(cpg_path).write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        timed_runs.append(elapsed)
        average = sum(timed_runs) / len(timed_runs)
        remaining = len(chunk_dirs) - position
        print(
            f"{worker_label} Created {cpg_path} in {elapsed / 60:.2f} minutes; "
            f"average {average / 60:.2f} min/CPG; ETA {remaining * average / 60:.1f} min",
            flush=True,
        )
        summaries.append(
            {
                "chunk": chunk_dir.name,
                "cpg": str(cpg_path),
                "status": "created",
                "seconds": round(elapsed, 3),
                "size_bytes": cpg_path.stat().st_size,
                "java_stack_size": java_stack_size or "default",
                "attempts": attempts,
                **chunk_manifest,
            }
        )

    range_end_index = None if args.numofdirs is None else args.start_dir_index + args.numofdirs
    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "joern_parse": str(joern_parse),
        "start_dir_index": args.start_dir_index,
        "numofdirs": args.numofdirs,
        "range_end_index_exclusive": range_end_index,
        "files_per_dir": args.files_per_dir,
        "java_stack_sizes": args.java_stack_sizes,
        "java_opts": args.java_opts,
        "chunk_count": len(chunk_dirs),
        "total_seconds": round(time.monotonic() - total_started_at, 3),
        "worker_label": worker_label,
        "cpgs": summaries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{worker_label} Wrote manifest: {manifest_path}", flush=True)
    print(f"{worker_label} EXPERIMENT_SECONDS create_project_codenet_cpgs={manifest['total_seconds']:.3f}", flush=True)
    return manifest


def srun_supports_overlap(srun: str) -> bool:
    try:
        result = subprocess.run(
            [srun, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return "--overlap" in result.stdout


def combine_worker_manifests(output_dir: Path, nodes: int) -> None:
    cpgs = []
    manifests = []
    for shard_id in range(nodes):
        manifest_path = output_dir / f"manifest_worker_{shard_id:04d}.json"
        manifest = load_json(manifest_path)
        if manifest is None:
            raise SystemExit(f"Missing or invalid worker manifest: {manifest_path}")
        manifests.append(manifest)
        cpgs.extend(manifest.get("cpgs", []))

    cpgs.sort(key=lambda item: item.get("global_index_start", 10**18))
    total_seconds = max((item.get("total_seconds", 0.0) for item in manifests), default=0.0)
    combined = {
        **{key: manifests[0].get(key) for key in manifests[0] if key != "cpgs"},
        "distributed_nodes": nodes,
        "worker_manifests": [str(output_dir / f"manifest_worker_{shard_id:04d}.json") for shard_id in range(nodes)],
        "chunk_count": len(cpgs),
        "worker_chunk_counts": [manifest.get("chunk_count", 0) for manifest in manifests],
        "total_seconds": total_seconds,
        "cpgs": cpgs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(f"[driver] Wrote combined manifest: {manifest_path}", flush=True)


def launch_distributed_workers(args: argparse.Namespace) -> int:
    srun = shutil.which("srun")
    if not srun:
        raise SystemExit("srun not found; use --force-local or run inside a Slurm allocation.")

    overlap_args = ["--overlap"] if srun_supports_overlap(srun) else []
    command = [
        srun,
        *overlap_args,
        "--nodes",
        str(args.nodes),
        "--ntasks",
        str(args.nodes),
        "--ntasks-per-node",
        "1",
        "--cpus-per-task",
        str(args.cpus_per_task),
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(args.output_dir),
        "--joern-parse",
        args.joern_parse,
        "--timeout",
        str(args.timeout),
        "--start-dir-index",
        str(args.start_dir_index),
        "--files-per-dir",
        str(args.files_per_dir),
        "--java-stack-sizes",
        args.java_stack_sizes,
        f"--java-opts={args.java_opts}",
        "--num-shards",
        str(args.nodes),
    ]
    if args.numofdirs is not None:
        command.extend(["--numofdirs", str(args.numofdirs)])
    if args.overwrite:
        command.append("--overwrite")

    print("[driver] " + " ".join(command), flush=True)
    return subprocess.run(command).returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Joern CPGs from ProjectCodeNet_1k_chunks directories."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("ProjectCodeNet_1k_chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("CPGs_ProjectCodeNet"))
    parser.add_argument("--joern-parse", default=shutil.which("joern-parse") or "joern-parse")
    parser.add_argument("--timeout", type=int, default=28800)
    parser.add_argument(
        "--start-dir-index",
        type=int,
        default=0,
        help="Chunk index to start from. 1285 means start at 1285k-1286k.",
    )
    parser.add_argument(
        "--numofdirs",
        type=int,
        default=None,
        help="Process this many chunk directories starting at --start-dir-index. Omit to run to the end.",
    )
    parser.add_argument("--files-per-dir", type=int, default=1000)
    parser.add_argument(
        "--java-stack-sizes",
        default="default,32m,64m,128m,256m,512m",
        help=(
            "Comma-separated JVM stack sizes to try when joern-parse fails with StackOverflowError. "
            "Use 'default' for no explicit -Xss."
        ),
    )
    parser.add_argument(
        "--java-opts",
        default="",
        help="Extra JVM options appended to JAVA_OPTS and _JAVA_OPTIONS for joern-parse.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=default_slurm_nodes(),
        help="Slurm nodes to use. In driver mode, one Joern worker is launched per node.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=default_cpus_per_task(),
        help="CPUs allocated to each Joern worker task.",
    )
    parser.add_argument(
        "--force-local",
        action="store_true",
        help="Run sequentially in the current process even when --nodes is greater than 1.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing CPG files.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    if args.start_dir_index < 0:
        raise SystemExit("--start-dir-index must be non-negative")
    if args.numofdirs is not None and args.numofdirs <= 0:
        raise SystemExit("--numofdirs must be positive when provided")
    if args.files_per_dir <= 0:
        raise SystemExit("--files-per-dir must be positive")
    if args.nodes <= 0:
        raise SystemExit("--nodes must be positive")
    if args.cpus_per_task <= 0:
        raise SystemExit("--cpus-per-task must be positive")

    if args.nodes > 1 and not args.force_local and not args.worker:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        returncode = launch_distributed_workers(args)
        if returncode != 0:
            raise SystemExit(returncode)
        combine_worker_manifests(args.output_dir, args.nodes)
        return

    joern_parse = resolve_joern_parse_executable(args.joern_parse)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dirs = select_chunk_dirs(args.input_dir, args.start_dir_index, args.numofdirs)
    if not chunk_dirs:
        raise SystemExit(f"No chunk directories found under {args.input_dir}")
    final_chunk_index = final_available_chunk_index(args.input_dir)
    if args.worker:
        shard_id = args.shard_id
        if shard_id is None:
            shard_id = slurm_int("SLURM_PROCID", 0)
        chunk_dirs = shard_chunk_dirs(chunk_dirs, shard_id, args.num_shards)
        manifest_path = args.output_dir / f"manifest_worker_{shard_id:04d}.json"
        worker_label = f"[worker {shard_id}/{args.num_shards}]"
    else:
        manifest_path = args.output_dir / "manifest.json"
        worker_label = "[local]"

    process_chunk_dirs(
        args=args,
        joern_parse=joern_parse,
        chunk_dirs=chunk_dirs,
        final_chunk_index=final_chunk_index,
        manifest_path=manifest_path,
        worker_label=worker_label,
    )


if __name__ == "__main__":
    main()
