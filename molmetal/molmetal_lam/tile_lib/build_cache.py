"""build_cache — Pre-compute the SMARTS-diverse tile library cache.

Running this module as ``python -m molmetal_lam.tile_lib.build_cache``
forces the joblib-backed cache for :func:`FRAGMENT_LIBRARY_200_TILES`
to be populated on disk.  Subsequent MCTS / benchmark runs then skip
the ~270 s RDKit sanitization step on cold start.

Usage
-----
::

    uv run python -m molmetal_lam.tile_lib.build_cache
    # or, with a custom cache dir:
    MOLMETAL_TILE_LIB_CACHE=/tmp/tile_cache \
        uv run python -m molmetal_lam.tile_lib.build_cache

Output
------
Three lines on stdout (machine-readable, ``key=value``):

    tile_count=<int>           # number of validated tiles persisted
    wall_seconds_first_call=<float>   # cold-call wall time (s)
    wall_seconds_cached_call=<float>  # warm-call wall time (s)

Exit code is 0 on success, non-zero on RDKit failure.
"""

from __future__ import annotations

import sys
import time
from typing import List

from molmetal_lam.tile_lib.library import (
    FRAGMENT_LIBRARY_200_TILES,
    TILE_LIB_CACHE_DIR,
    clear_tile_lib_cache,
)
from molmetal_lam.tile_lib.tile import Tile


def _build_timed() -> tuple[List[Tile], float]:
    """Call :func:`FRAGMENT_LIBRARY_200_TILES` and return (tiles, seconds)."""
    t0 = time.perf_counter()
    tiles = FRAGMENT_LIBRARY_200_TILES()
    return tiles, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv or "-f" in argv

    print(f"# cache_dir={TILE_LIB_CACHE_DIR}", flush=True)

    if force:
        print("# --force: clearing existing cache before warm-up", flush=True)
        clear_tile_lib_cache()

    # First call (cold): build, sanitize, persist.
    tiles_first, t_first = _build_timed()
    tile_count = len(tiles_first)
    print(f"tile_count={tile_count}", flush=True)
    print(f"wall_seconds_first_call={t_first:.3f}", flush=True)

    # Second call (warm): load from disk.
    tiles_second, t_cached = _build_timed()
    if len(tiles_second) != tile_count:
        print(
            f"# ERROR: cache returned {len(tiles_second)} tiles, "
            f"expected {tile_count}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(f"wall_seconds_cached_call={t_cached:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
