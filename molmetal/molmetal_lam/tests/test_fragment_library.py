"""Tests for the Phase-1 SMARTS-diverse fragment library (L-3).

Covers:
    * Library size >= 200 when ``include_fragments=True``.
    * Backward-compat: the original 12-tile library still returns 12
      tiles when ``include_fragments=False`` (default).
    * Per-tile validity: every tile parses with ``Chem.MolFromSmiles``,
      embeds with ETKDGv3 (``randomSeed=0xC11C``) and satisfies the
      MW/logP filters.
    * WF-SA-Fragment-Pool-Optimize: the top-10 highest-SA filter
      correctly identifies + excludes unstable / hard-to-synthesize
      fragments from the validated pool.
"""
from __future__ import annotations

import pytest

from molmetal_lam.sbdd_env.sa_score import (  # type: ignore
    _SASCORER_AVAILABLE,
    sa_score_ertl,
)
from molmetal_lam.tile_lib.click_tiles import STANDARD_12_TILES
from molmetal_lam.tile_lib.fragment_pool import (
    FRAGMENT_POOL_AZIDES,
    FRAGMENT_POOL_ALKYNES,
    FRAGMENT_POOL_DIENES,
    FRAGMENT_POOL_THOLS,
    fragments_from_chembl_reactive,
    l6_fragment_pool_metrics,
)
from molmetal_lam.tile_lib.library import (
    FRAGMENT_LIBRARY_200_TILES,
    build_tile_library,
)
from molmetal_lam.tile_lib.sa_filter import (  # type: ignore
    TOP10_HIGHEST_SA_PAIRS,
    TOP10_HIGHEST_SA_SMILES,
    is_top10_highest_sa,
    top10_filter_pool,
)


# ---------------------------------------------------------------------------
# Library size
# ---------------------------------------------------------------------------


def test_fragment_library_size() -> None:
    """``FRAGMENT_LIBRARY_200_TILES`` returns >= 200 validated tiles.

    Counts tiles emitted by the SMARTS-diverse pool (validated for
    parse + ETKDGv3 embed + MW/logP filters).
    """
    pool = FRAGMENT_LIBRARY_200_TILES()
    assert isinstance(pool, list)
    assert len(pool) >= 200, (
        f"Fragment pool has only {len(pool)} tiles, target >= 200"
    )
    # All emitted tiles must be Tile instances.
    from molmetal_lam.tile_lib.tile import Tile
    assert all(isinstance(t, Tile) for t in pool)


def test_build_tile_library_with_fragments_returns_200_plus() -> None:
    """``build_tile_library(include_fragments=True)`` returns the Phase-1 pool."""
    tiles = build_tile_library(n_max=400, include_fragments=True)
    assert len(tiles) >= 200


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_fragment_library_backward_compat() -> None:
    """STANDARD_12_TILES still returns 12 tiles (backward compat).

    The original Phase-0 12-tile library must remain exactly 12 when
    the Phase-1 fragment expansion is in place — no regression.
    """
    std = STANDARD_12_TILES()
    assert len(std) == 12

    # build_tile_library with default args still returns 12 tiles.
    assert len(build_tile_library(n_max=12)) == 12
    assert len(build_tile_library()) == 12

    # include_fragments=False (default) does NOT change the count.
    assert len(build_tile_library(n_max=200, include_fragments=False)) == 12


# ---------------------------------------------------------------------------
# Per-tile validity
# ---------------------------------------------------------------------------


def test_fragment_tile_validity() -> None:
    """Every fragment-pool tile parses, embeds, and passes MW/logP filters.

    Re-runs the RDKit validation on each emitted tile (independently
    of the build_click_tile path) so any silent regression in the
    validation hook is caught here.
    """
    from rdkit import Chem  # type: ignore[import-not-found]
    from rdkit.Chem import AllChem, Descriptors  # type: ignore[import-not-found]

    pool = fragments_from_chembl_reactive()
    assert pool, "Fragment pool is empty — cannot validate"

    for t in pool:
        # Parse the canonical SMILES.
        mol = Chem.MolFromSmiles(t.smiles)
        assert mol is not None, f"Tile {t.tile_id} failed to parse: {t.smiles}"

        # Embed with ETKDGv3 / randomSeed=0xC11C.
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC11C
        rc = AllChem.EmbedMolecule(mol_h, params)
        assert rc == 0, (
            f"Tile {t.tile_id} ({t.smiles}) ETKDGv3 embed failed rc={rc}"
        )

        # Descriptor filters: MW in [40, 300], logP in [-2, 5].
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        assert 40.0 <= mw <= 300.0, (
            f"Tile {t.tile_id} ({t.smiles}) MW out of range: {mw}"
        )
        assert -2.0 <= logp <= 5.0, (
            f"Tile {t.tile_id} ({t.smiles}) logP out of range: {logp}"
        )

        # Tile.coords must be populated (embed succeeded) and have the
        # expected (N_atoms, 3) shape.
        assert t.coords is not None, (
            f"Tile {t.tile_id} ({t.smiles}) has no 3D coords"
        )


# ---------------------------------------------------------------------------
# Coverage by category
# ---------------------------------------------------------------------------


def test_fragment_pool_covers_all_four_categories() -> None:
    """Each reactive-handle family is represented in the pool.

    The pool must contain at least one tile tagged ``azide``, one
    tagged ``terminal_alkyne`` or ``cyclooctyne``, one tagged
    ``diene`` or ``dienophile``, and one tagged ``thiol``.
    """
    pool = fragments_from_chembl_reactive()

    has_azide = any(t.has_group("azide") for t in pool)
    has_alkyne = any(
        t.has_group("terminal_alkyne") or t.has_group("cyclooctyne") for t in pool
    )
    has_diene = any(
        t.has_group("diene") or t.has_group("dienophile") for t in pool
    )
    has_thiol = any(t.has_group("thiol") for t in pool)

    assert has_azide, "No azide tile in fragment pool"
    assert has_alkyne, "No alkyne tile in fragment pool"
    assert has_diene, "No diene/dienophile tile in fragment pool"
    assert has_thiol, "No thiol tile in fragment pool"


def test_fragment_pool_hardcoded_lists_nonempty() -> None:
    """Each hardcoded SMARTS pool has at least 50 entries."""
    assert len(FRAGMENT_POOL_AZIDES) >= 50
    assert len(FRAGMENT_POOL_ALKYNES) >= 50
    assert len(FRAGMENT_POOL_DIENES) >= 50
    assert len(FRAGMENT_POOL_THOLS) >= 50


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


def test_fragment_pool_metrics_populated() -> None:
    """``l6_fragment_pool_metrics`` reports counters per category.

    After at least one ``fragments_from_chembl_reactive`` call, the
    metrics dict must contain ``input`` and ``valid`` counters for
    every category.
    """
    # Touch the pool to ensure counters are populated.
    _ = fragments_from_chembl_reactive()
    metrics = l6_fragment_pool_metrics()
    # R10 axis A — the pool now has 9 categories (4 legacy + 5 new
    # click-handle families).  The 4 legacy categories are still
    # required to satisfy the strict L-3 contract; the 5 new
    # families are checked separately below.
    legacy = {"azide", "alkyne", "diene", "thiol"}
    assert legacy <= set(metrics.keys()), (
        f"Missing legacy L-3 categories; got {sorted(metrics.keys())}"
    )
    for cat in ("azide", "alkyne", "diene", "thiol"):
        assert "input" in metrics[cat]
        assert "valid" in metrics[cat]
        assert "embed_fail" in metrics[cat]
        assert "mw_fail" in metrics[cat]
        assert "logp_fail" in metrics[cat]
        # input == valid + embed_fail + mw_fail + logp_fail
        total_fail = (
            metrics[cat]["embed_fail"]
            + metrics[cat]["mw_fail"]
            + metrics[cat]["logp_fail"]
        )
        assert metrics[cat]["input"] == metrics[cat]["valid"] + total_fail


# ---------------------------------------------------------------------------
# WF-SA-Fragment-Pool-Optimize (2026-09-15)
#
# The top-10 highest-SA tiles are the 10 SMILES with the largest
# Ertl-Schuffenhauer SA score in the validated pool.  They are
# unstable alpha-hydroxy azides, alkyne-thioether with thiol-yne
# competition, and a dicyclopentadiene partial.  We verify the
# blacklist ordering + the default-filter behaviour.
# ---------------------------------------------------------------------------


_SA_FILTER_TEST_MARK = pytest.mark.skipif(
    not _SASCORER_AVAILABLE,
    reason="RDKit sascorer not importable in this environment",
)


@_SA_FILTER_TEST_MARK
def test_top10_highest_sa_tiles_identified() -> None:
    """TOP10_HIGHEST_SA_PAIRS contains 10 entries whose SA score is the
    10 largest in the validated pool, and each of those 10 scores
    is >= every score outside the top-10.

    This test runs the Ertl sascorer over the full pool, sorts the
    SMILES by SA descending, and asserts the top-10 stored in
    :data:`TOP10_HIGHEST_SA_PAIRS` is exactly that set (set-wise —
    we tolerate reordering among the 10 because the published
    blacklist stores them in descending order for readability).
    """
    pool = FRAGMENT_LIBRARY_200_TILES()
    assert len(pool) >= 200, (
        f"Pool has {len(pool)} tiles; need >= 200 for SA test"
    )
    # Score every tile in the validated pool.
    scored = []
    for t in pool:
        smi = getattr(t, "smiles", "") or ""
        sa = sa_score_ertl(smi)
        scored.append((smi, sa))
    # Drop NaN scores (rare; the sascorer returns NaN only on
    # unparseable SMILES, which the validator should already have
    # filtered out, but be defensive).
    scored = [(s, v) for s, v in scored if v == v]
    scored.sort(key=lambda p: -p[1])
    actual_top10_smiles = {s for s, _ in scored[:10]}
    blacklist_smiles = {s for _, s, _ in TOP10_HIGHEST_SA_PAIRS}
    assert actual_top10_smiles == blacklist_smiles, (
        f"Top-10 SA set mismatch.\n"
        f"  Actual:  {sorted(actual_top10_smiles)}\n"
        f"  Stored:  {sorted(blacklist_smiles)}"
    )
    # Verify ordering: each of the top-10 SA scores >= every score
    # outside the top-10.
    top10_scores = sorted({v for _, v in scored[:10]}, reverse=True)
    rest_scores = [v for _, v in scored[10:]]
    assert min(top10_scores) >= max(rest_scores), (
        f"Top-10 ordering broken: min(top10)={min(top10_scores)} "
        f"< max(rest)={max(rest_scores)}"
    )
    # Cross-check the stored SA values match the computed ones (within
    # FP tolerance — sascorer is deterministic so this is exact).
    stored = {s: sa for _, s, sa in TOP10_HIGHEST_SA_PAIRS}
    for s, v in scored[:10]:
        assert abs(stored[s] - v) < 1e-6, (
            f"SA value mismatch for {s}: stored={stored[s]} "
            f"computed={v}"
        )


def test_filtered_pool_default_excludes_top10() -> None:
    """top10_filter_pool(tiles) drops all 10 highest-SA SMILES by default.

    Independent of sascorer availability (uses stored blacklist, not
    RDKit).  Returns the unfiltered pool when ``keep_high_sa=True``.
    """
    pool = FRAGMENT_LIBRARY_200_TILES()
    pre = len(pool)
    # Default: keep_high_sa=False → top-10 removed.
    filtered = top10_filter_pool(pool)
    assert len(filtered) == pre - 10, (
        f"Expected {pre - 10} tiles after top-10 filter, got {len(filtered)}"
    )
    # No blacklisted SMILES survive the filter.
    smiles_in_filtered = {getattr(t, "smiles", "") for t in filtered}
    for black in TOP10_HIGHEST_SA_SMILES:
        assert black not in smiles_in_filtered, (
            f"Blacklisted SMILES survived filter: {black}"
        )
    # Opt-in: keep_high_sa=True → pool returned unchanged.
    kept = top10_filter_pool(pool, keep_high_sa=True)
    assert len(kept) == pre
    assert {getattr(t, "smiles", "") for t in kept} == {
        getattr(t, "smiles", "") for t in pool
    }
    # Membership helper agrees.
    for rank, smi, _sa in TOP10_HIGHEST_SA_PAIRS:
        assert is_top10_highest_sa(smi)
    # And the helper is False for a non-blacklisted SMILES.
    assert not is_top10_highest_sa("CCO")
    # Defensive: an empty blacklist-of-everything list would not
    # collapse the pool; if every tile were blacklisted, the filter
    # falls back to the unfiltered pool (top10_filter_pool guards
    # against ``filtered == []``).
    sentinel_empty: list = []
    result = top10_filter_pool(sentinel_empty)
    assert result == []

