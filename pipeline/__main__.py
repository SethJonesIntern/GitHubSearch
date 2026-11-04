"""Allow `python -m pipeline [run] ...` as the single entry point.

`run` is the default (and only) subcommand, so both of these work:
    python -m pipeline run --smoke
    python -m pipeline --smoke
"""
import sys

from pipeline.run import main

if __name__ == "__main__":
    # Accept an optional leading "run" verb for readability; strip it so the
    # rest of the args go straight to the runner's parser.
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        del sys.argv[1]
    main()
