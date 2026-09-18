"""build_tile_library — Public entry point for the Phase-0 click tile library.

This module is a thin orchestration layer over :mod:`click_tiles` and
:mod:`fragment_pool`. It exposes :func:`build_tile_library` that
returns the canonical 12-tile Phase-0 library (backward-compatible) or
the extended 200+ tile Phase-1 library when ``include_fragments=True``.

Phase 0 (current — default):
    Returns the 12 hand-curated standard tiles (4 azides + 4 alkynes +
    4 partners). These are the smallest set that exercises every
    reaction class implemented in :mod:`molmetal_lam.reactions`.

Phase 1 (opt-in via ``include_fragments=True``):
    A larger SMARTS-diverse fragment library drawn from a hardcoded
    ChEMBL reactive-handle / ZINC click-chemistry subset (see
    :mod:`fragment_pool`). The interface of :func:`build_tile_library`
    is backward-compatible — only an opt-in kwarg is added.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import joblib

from molmetal_lam.tile_lib.click_tiles import (
    ALKYNE_TILES,
    AZIDE_TILES,
    PARTNER_TILES,
    STANDARD_12_TILES,
    STANDARD_14_TILES,
    THIOL_TILES,
    build_all_click_tiles,
)
from molmetal_lam.tile_lib.fragment_pool import fragments_from_chembl_reactive
from molmetal_lam.tile_lib.property_tests import assert_library_well_formed
from molmetal_lam.tile_lib.tile import Tile

# ---------------------------------------------------------------------------
# Disk cache for FRAGMENT_LIBRARY_200_TILES().
#
# Rationale: building the SMARTS-diverse 204-tile library invokes RDKit
# sanitization + ETKDGv3 embedding for ~200 SMILES, which dominates wall
# time (measured ~270s on RX 7800 XT at branching=1020).  Building once
# and persisting to disk via joblib.Memory drops subsequent calls to <1s.
#
# Cache location
# --------------
# Default: ``molmetal/molmetal_lam/tile_lib/.cache/joblib``
# (sibling to this file so the cache ships with the source tree and
#  is auto-invalidated by git when the source changes — only the
#  on-disk joblib hash matters).
#
# Override via the ``MOLMETAL_TILE_LIB_CACHE`` env var (useful for CI
# scratch dirs or sandboxed test runs).
#
# Invalidation
# ------------
# joblib.Memory uses a content hash of the cached callable's bytecode
# AND the function source.  Any edit to :func:`_build_fragment_library_impl`
# or to the underlying helpers it transitively calls from this module
# therefore invalidates the cache automatically.
# ---------------------------------------------------------------------------

_CACHE_DIR_ENV = "MOLMETAL_TILE_LIB_CACHE"
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "joblib"


def _resolve_cache_dir() -> str:
    """Return the cache directory, honoring ``MOLMETAL_TILE_LIB_CACHE``."""
    override = os.environ.get(_CACHE_DIR_ENV)
    cache_dir = Path(override) if override else _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


_JOBLIB_MEMORY = joblib.Memory(location=_resolve_cache_dir(), verbose=0)


def _clear_cache_dir() -> None:
    """Erase the disk cache (test helper — see ``test_tile_lib_cache.py``)."""
    _JOBLIB_MEMORY.clear(warn=False)

__all__ = [
    "build_tile_library",
    "build_tile_dict",
    "l6_library_metrics",
    "FRAGMENT_LIBRARY_200_TILES",
    "TILE_LIB_CACHE_DIR",
    "clear_tile_lib_cache",
]


# Public handle for tests and the build_cache.py CLI helper.
TILE_LIB_CACHE_DIR: str = _resolve_cache_dir()


def clear_tile_lib_cache() -> None:
    """Erase the on-disk cache backing :func:`FRAGMENT_LIBRARY_200_TILES`.

    Exposed for tests that need a guaranteed cold-cache state.  Most
    callers should rely on joblib's automatic hash-based invalidation.
    """
    _clear_cache_dir()


# L6 library-side instrumentation hook (govern_review_L4_L6.md).
_L6_LIB_COUNTERS: dict = {"library_build_calls": 0, "tiles_emitted": 0}


def l6_library_metrics() -> dict:
    """Return a snapshot of L6 library-level instrumentation."""
    return dict(_L6_LIB_COUNTERS)


def build_tile_library(
    n_max: int = 12,
    *,
    include_thiol: bool = False,
    include_fragments: bool = False,
) -> List[Tile]:
    """Return the click tile library.

    Parameters
    ----------
    n_max : int, optional
        Maximum number of tiles to return.
        - ``n_max=12`` (default) returns the Phase-0 12-tile library.
        - ``n_max=14`` with ``include_thiol=True`` returns the 14-tile
          extended Phase-0 library (Close-Loop #3).
        - ``n_max=200`` (or larger) with ``include_fragments=True``
          returns the Phase-1 200+ tile library.
    include_thiol : bool, optional
        If True, the 2 dedicated thiol tiles are appended (max 14).
        Default False preserves backward compatibility.
    include_fragments : bool, optional
        If True, the Phase-1 SMARTS-diverse fragment pool (200+ tiles)
        replaces the 12-tile library. Default False preserves backward
        compatibility with all callers expecting exactly 12 tiles.

    Returns
    -------
    List[Tile]
        List of :class:`Tile` instances (fresh copy).
    """
    if n_max <= 0:
        return []

    if include_fragments:
        tiles = fragments_from_chembl_reactive()
    else:
        tiles = build_all_click_tiles(embed_3d=True)
        if not include_thiol:
            tiles = tiles[:12]

    _L6_LIB_COUNTERS["library_build_calls"] += 1
    _L6_LIB_COUNTERS["tiles_emitted"] += len(tiles[:n_max])
    return list(tiles[:n_max])


def build_tile_dict(
    n_max: int = 12,
    *,
    include_thiol: bool = False,
    include_fragments: bool = False,
) -> Dict[str, Tile]:
    """Build the tile library as a ``{tile_id: Tile}`` dict.

    Useful when the proof-search algorithm wants O(1) lookup by tile id.

    Parameters
    ----------
    n_max : int, optional
        Maximum number of tiles to include (default 12).
    include_thiol : bool, optional
        If True, the 2 dedicated thiol tiles are included (max 14).
        Default False for backward compatibility.
    include_fragments : bool, optional
        If True, the Phase-1 200+ fragment library is returned instead
        of the 12-tile library. Default False.

    Returns
    -------
    Dict[str, Tile]
        Mapping ``tile_id -> Tile``.
    """
    return {
        t.tile_id: t
        for t in build_tile_library(
            n_max=n_max,
            include_thiol=include_thiol,
            include_fragments=include_fragments,
        )
    }


# Convenience: expose the four group slices as module-level constants
# so other modules can do ``from molmetal_lam.tile_lib.library import
# AZIDES, ALKYNES, PARTNERS, THIOLS``.
AZIDES = AZIDE_TILES()
ALKYNES = ALKYNE_TILES()
PARTNERS = PARTNER_TILES()
THIOLS = THIOL_TILES()
STANDARD_12 = STANDARD_12_TILES()
STANDARD_14 = STANDARD_14_TILES()


# Phase-1 fragment library — opt-in. Returns the validated 200+ tile pool.
@_JOBLIB_MEMORY.cache(ignore=[])
def _build_fragment_library_impl() -> List[Tile]:
    """Uncached worker for :func:`FRAGMENT_LIBRARY_200_TILES`.

    Wrapped by :data:`_JOBLIB_MEMORY.cache` so the validated library is
    persisted to disk on the first call and reloaded on every subsequent
    call (no RDKit work, <1s even on a cold disk cache).
    """
    return list(fragments_from_chembl_reactive())


def FRAGMENT_LIBRARY_200_TILES() -> List[Tile]:  # noqa: N802 — caps for API
    """Return the Phase-1 SMARTS-diverse fragment library (200+ tiles).

    Backed by :func:`fragments_from_chembl_reactive` (ChEMBL reactive
    handles + ZINC click-chemistry subset, see :mod:`fragment_pool`).
    Each tile has been validated for RDKit parse, ETKDGv3 embed
    success, MW in [40, 300] and logP in [-2, 5].

    The first call invokes RDKit sanitization + ETKDGv3 embedding for
    ~200 SMILES and persists the validated tiles to disk via
    :class:`joblib.Memory`.  Subsequent calls (including those from
    other processes) load the cached list in <1s.  The cache is
    invalidated automatically when this module's source changes.

    On every call (cold *or* cached) the returned list is run through
    :func:`assert_library_well_formed` so that any malformed SMILES
    introduced by edits to the hardcoded fragment pools is filtered
    out before reaching the MCTS — see
    :mod:`molmetal_lam.tile_lib.property_tests` for the invariant
    definitions.

    Returns
    -------
    List[Tile]
        Validated fragment tiles (typically 200+ depending on filter
        outcomes; see :func:`l6_fragment_pool_metrics`).
    """
    # joblib returns the cached value (a list); wrap to defend against
    # accidental in-place mutation by callers.  The well-formedness
    # check is intentionally cheap (no RDKit embedding) so it adds
    # negligible overhead even on every call.
    raw = list(_build_fragment_library_impl())
    return assert_library_well_formed(raw)
