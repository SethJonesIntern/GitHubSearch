"""Delete cloned application checkouts to reclaim disk.

Stage 5 run with --keep-clones leaves each analyzed repo under pipeline/repos/.
This removes those clones, freeing the disk while KEEPING every analysis output
in artifacts/ (the CSVs and the per-repo slices/). With --cpg it also sweeps any
persisted repo.cpg files (left by --keep-cpg) without touching the slice JSON.

  python -m pipeline.clean_clones             # delete all clones, report freed space
  python -m pipeline.clean_clones --dry-run   # show what would be deleted, delete nothing
  python -m pipeline.clean_clones --cpg       # also delete persisted *.cpg files
"""
import argparse
import os
import stat
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths  # noqa: E402


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _rmtree(path: Path):
    """Windows-safe: git pack files are read-only, so clear the bit and retry."""
    def onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass
    shutil.rmtree(path, onerror=onerror)


def _fmt(nbytes: int) -> str:
    gb = nbytes / 1024 ** 3
    return f"{gb:.2f} GB" if gb >= 1 else f"{nbytes / 1024 ** 2:.1f} MB"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be deleted without deleting anything")
    ap.add_argument("--cpg", action="store_true",
                    help="Also delete persisted *.cpg files under slices/ (keeps the JSON)")
    args = ap.parse_args()

    clones = [d for d in paths.REPOS_DIR.iterdir() if d.is_dir()] \
        if paths.REPOS_DIR.exists() else []
    cpgs = list(paths.SLICES_DIR.rglob("*.cpg")) if (args.cpg and paths.SLICES_DIR.exists()) else []

    clone_bytes = sum(_dir_size(d) for d in clones)
    cpg_bytes = sum(f.stat().st_size for f in cpgs if f.exists())
    total = clone_bytes + cpg_bytes

    verb = "Would remove" if args.dry_run else "Removing"
    print(f"{verb} {len(clones)} clones from {paths.REPOS_DIR}  ({_fmt(clone_bytes)})")
    if args.cpg:
        print(f"{verb} {len(cpgs)} .cpg files from {paths.SLICES_DIR}  ({_fmt(cpg_bytes)})")
    if not clones and not cpgs:
        print("Nothing to clean.")
        return

    if args.dry_run:
        print(f"\nDry run — nothing deleted. Would free {_fmt(total)}.")
        return

    for d in clones:
        _rmtree(d)
    for f in cpgs:
        try:
            f.unlink()
        except OSError:
            pass
    print(f"\nFreed {_fmt(total)}. Analysis outputs in {paths.ARTIFACTS_DIR} are untouched.")


if __name__ == "__main__":
    main()
