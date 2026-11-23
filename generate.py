"""Kept for compatibility with v0.1: same as `synthetic-slices generate ...` (see `synthetic-slices generate --help`).
    python generate.py --n 1000 --size 256 --out tiles256
Since v0.2 the defaults are the v0.2 recipe (folds, cracks, crushed regions...) and the v0.2 generator; `--recipe v0.1 --weights v0.1`
gives the v0.1 behaviour."""
import sys
from synthetic_slices.cli import main

if __name__ == "__main__":
    main(["generate"] + sys.argv[1:])
