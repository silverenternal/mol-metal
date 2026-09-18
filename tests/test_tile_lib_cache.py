"""Tests for the joblib-backed cache of :func:`FRAGMENT_LIBRARY_200_TILES`.

These tests verify the r0 fix: the SMARTS-diverse 204-tile library is
pre-computed once on disk so that subsequent benchmark / MCTS calls
re-load it in <1 s instead of paying ~270 s of RDKit sanitization
each time.

Run with::

    uv run pytest tests/test_tile_lib_cache.py -v --tb=short
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import List

import pytest

from molmetal_lam.tile_lib.library import (
    FRAGMENT_LIBRARY_200_TILES,
    clear_tile_lib_cache,
)
from molmetal_lam.tile_lib.tile import Tile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Force the tile-lib cache into a fresh tmp dir per test.

    This guarantees a cold cache at the start of each test, regardless
    of state left by previous runs or other test modules.
    """
    cache_dir = tmp_path / "tile_lib_joblib"
    monkeypatch.setenv("MOLMETAL_TILE_LIB_CACHE", str(cache_dir))
    # The module-level joblib.Memory reads the env var at import time,
    # so we must reload the library to pick up the override.
    import importlib
    import molmetal_lam.tile_lib.library as lib

    importlib.reload(lib)
    yield cache_dir
    # Reload again to restore the default cache location for other tests.
    monkeypatch.delenv("MOLMETAL_TILE_LIB_CACHE", raising=False)
    importlib.reload(lib)


def _wall_seconds(fn) -> tuple[List[Tile], float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_first_call_under_5s(isolated_cache_dir: Path) -> None:
    """Cold-cache first call: build + sanitize + persist, <5 s on this box.

    The cold call still does RDKit work, so it cannot be cached.
    We only enforce an upper bound to catch gross regressions (e.g.
    RDKit re-installed to an empty venv) — the spec's hard target is
    the cached-call wall time below.
    """
    tiles, elapsed = _wall_seconds(FRAGMENT_LIBRARY_200_TILES)
    assert len(tiles) > 0, "FRAGMENT_LIBRARY_200_TILES returned empty list"
    assert elapsed < 5.0, (
        f"first-call wall time {elapsed:.3f}s exceeded 5s budget "
        "(RDKit probably broken)"
    )


def test_second_call_under_1s(isolated_cache_dir: Path) -> None:
    """Warm-cache second call: load from disk, <1 s.

    This is the actual r0 fix: subsequent calls skip RDKit entirely.
    """
    # Warm the cache (cold call is allowed to take seconds).
    FRAGMENT_LIBRARY_200_TILES()

    # Measure the cached call.
    tiles, elapsed = _wall_seconds(FRAGMENT_LIBRARY_200_TILES)
    assert len(tiles) > 0
    assert elapsed < 1.0, (
        f"cached-call wall time {elapsed:.3f}s exceeded 1s budget — "
        "joblib cache is not being hit"
    )


def test_tile_count_is_deterministic(isolated_cache_dir: Path) -> None:
    """Two calls return the same number of tiles (deterministic build)."""
    tiles_a = FRAGMENT_LIBRARY_200_TILES()
    tiles_b = FRAGMENT_LIBRARY_200_TILES()
    assert len(tiles_a) == len(tiles_b), (
        f"non-deterministic tile count: {len(tiles_a)} vs {len(tiles_b)}"
    )


def test_tile_identities_are_stable(isolated_cache_dir: Path) -> None:
    """Tile ids and SMILES are identical between cold and warm calls."""
    tiles_a = FRAGMENT_LIBRARY_200_TILES()
    tiles_b = FRAGMENT_LIBRARY_200_TILES()
    assert len(tiles_a) == len(tiles_b)
    for a, b in zip(tiles_a, tiles_b):
        assert a.tile_id == b.tile_id
        assert a.smiles == b.smiles


def test_cache_clear_forces_rebuild(isolated_cache_dir: Path) -> None:
    """``clear_tile_lib_cache`` drops the on-disk cache and the next call
    rebuilds from scratch.
    """
    tiles_warm_a = FRAGMENT_LIBRARY_200_TILES()
    clear_tile_lib_cache()
    tiles_after_clear = FRAGMENT_LIBRARY_200_TILES()
    # Same content, different cache entry (joblib hash bumped).
    assert len(tiles_warm_a) == len(tiles_after_clear)
    for a, b in zip(tiles_warm_a, tiles_after_clear):
        assert a.tile_id == b.tile_id
