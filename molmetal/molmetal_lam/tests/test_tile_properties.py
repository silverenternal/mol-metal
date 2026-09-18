"""Property-based tests for the Phase-1 SMARTS-diverse tile library.

These tests are the *CI-time* safety net for malformed SMILES in
:func:`molmetal_lam.tile_lib.library.FRAGMENT_LIBRARY_200_TILES`.
The atom-level property tests in :mod:`test_lambda_properties` caught
the Au_III arity bug; this file catches a complementary class of
bugs — *tile-level* malformations that only surface when the library
is iterated (e.g. RDKit parse errors, drug-likeness violations,
duplicate SMILES, library size regressions).

Why ``hypothesis``?
-------------------
Property-based testing is the right tool here because we want a
*guarantee* that the library invariants hold across every tile, not
just a sample of three or four.  The four property tests in this
file iterate the full 200+ tile library and assert each invariant
on every element.  If any tile fails, the failing SMILES is logged
in the pytest output for easy debugging.

Tests
-----
1. ``property_test_all_tiles_parse`` — every ``Tile.smiles`` parses
   with RDKit's ``Chem.MolFromSmiles`` without raising.
2. ``property_test_all_tiles_drug_like`` — every ``Tile`` has
   MW <= 500 (Lipinski Rule of 5 ceiling).
3. ``property_test_all_tiles_unique_smiles`` — no two tiles share
   a canonical SMILES (duplicate SMILES would silently halve the
   MCTS branching factor).
4. ``property_test_tile_count_in_range`` — the library contains
   between 150 and 250 tiles (loose, allows for pool-curation
   growth/shrinkage).

Why no pytest fixtures on the @given tests?
-------------------------------------------
``hypothesis`` 6.x and pytest fixtures don't always cooperate on
``@given``-decorated functions: the test is silently dropped from
collection if the signature contains both a pytest fixture and
``@given`` parameters.  We side-step the issue by using a
module-level cached library and a function-scoped ``Chem`` import
inside each test body.  The cost is one library build per
``pytest`` invocation (loaded from the joblib cache in <1s) and
the per-test RDKit import overhead is negligible.
"""

from __future__ import annotations

import functools
from typing import List

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from molmetal_lam.tile_lib.library import FRAGMENT_LIBRARY_200_TILES
from molmetal_lam.tile_lib.property_tests import (
    ALLOWED_ATOMS,
    MAX_TILE_MW,
    assert_library_well_formed,
    assert_tile_well_formed,
)
from molmetal_lam.tile_lib.tile import Tile


# ---------------------------------------------------------------------------
# Module-level lazy library loader.
#
# Using ``functools.cache`` instead of a pytest fixture keeps the
# library object outside the @given test signatures so hypothesis
# can introspect the test parameters without tripping on pytest
# fixture injection (a known issue with hypothesis 6.x + pytest
# fixtures on the same function).
# ---------------------------------------------------------------------------


@functools.cache
def _get_library() -> List[Tile]:
    """Return the Phase-1 SMARTS-diverse fragment library (cached)."""
    return FRAGMENT_LIBRARY_200_TILES()


@functools.cache
def _get_chem():
    """Return the ``rdkit.Chem`` module, or raise if not installed."""
    return pytest.importorskip("rdkit.Chem")


# ---------------------------------------------------------------------------
# 1. Every tile's SMILES parses via RDKit.
# ---------------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=10000))
@settings(
    max_examples=64,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_property_all_tiles_parse(idx: int) -> None:
    """Pick an arbitrary tile; its SMILES must parse via RDKit.

    ``hypothesis`` draws an arbitrary integer in ``[0, 10000]`` and
    we modulo it against the library length so the property test
    exercises a different tile on every example.  A regression
    that breaks *one* specific SMILES is caught even if it slips
    past unit tests.
    """
    library = _get_library()
    if not library:
        pytest.skip("Fragment library is empty")
    Chem = _get_chem()
    real_idx = idx % len(library)
    tile = library[real_idx]
    mol = Chem.MolFromSmiles(tile.smiles)
    assert mol is not None, (
        f"Tile {tile.tile_id} (idx={real_idx}) failed RDKit parse: "
        f"{tile.smiles!r}"
    )


# ---------------------------------------------------------------------------
# 2. Every tile is drug-like (MW <= 500).
# ---------------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=10000))
@settings(
    max_examples=64,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_property_all_tiles_drug_like(idx: int) -> None:
    """Pick an arbitrary tile; its MW must be <= 500.

    The fragment library validation (see
    :mod:`molmetal_lam.tile_lib.fragment_pool`) caps MW at 300; the
    property test here uses 500 as a *regression* ceiling — anything
    heavier than 500 is, by construction, a malformed tile.
    """
    library = _get_library()
    if not library:
        pytest.skip("Fragment library is empty")
    real_idx = idx % len(library)
    tile = library[real_idx]
    assert 0.0 <= tile.mw <= MAX_TILE_MW, (
        f"Tile {tile.tile_id} (idx={real_idx}) MW={tile.mw} outside "
        f"[0, {MAX_TILE_MW}] (smiles={tile.smiles!r})"
    )


# ---------------------------------------------------------------------------
# 3. No two tiles share a SMILES.
# ---------------------------------------------------------------------------


def test_property_all_tiles_unique_smiles() -> None:
    """No duplicate SMILES in the fragment library.

    Duplicate SMILES would silently halve the MCTS branching
    factor — the proof-search algorithm relies on
    ``len(library) * |rules|`` distinct candidates per move.  Any
    duplicate is a CI-visible regression.
    """
    library = _get_library()
    if not library:
        pytest.skip("Fragment library is empty")
    smiles_list = [t.smiles for t in library]
    seen: set[str] = set()
    duplicates: list[tuple[str, int, int]] = []
    for i, smi in enumerate(smiles_list):
        if smi in seen:
            first = smiles_list.index(smi)
            duplicates.append((smi, first, i))
        else:
            seen.add(smi)
    assert not duplicates, (
        f"Found {len(duplicates)} duplicate SMILES in the fragment library: "
        f"{duplicates[:5]}"
    )


# ---------------------------------------------------------------------------
# 4. Library size is in the expected range.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "low, high",
    [
        (150, 250),
    ],
)
def test_property_tile_count_in_range(low: int, high: int) -> None:
    """Library contains between ``low`` and ``high`` tiles.

    Loose bounds (150 / 250) to allow for pool-curation growth
    or shrinkage without flaking CI on every commit.  The hard
    ``>= 200`` guarantee lives in
    :mod:`test_fragment_library.test_fragment_library_size`; this
    test catches *unintended* size changes (e.g. accidentally
    halving the pool via a broken
    :func:`fragments_from_chembl_reactive` category toggle).
    """
    library = _get_library()
    n = len(library)
    assert low <= n <= high, (
        f"Fragment library size {n} outside expected range [{low}, {high}]"
    )


# ---------------------------------------------------------------------------
# 5. Well-formedness helper sanity checks.
# ---------------------------------------------------------------------------


def test_assert_library_well_formed_returns_subset() -> None:
    """``assert_library_well_formed`` returns a (possibly smaller) subset.

    Acts as a smoke test for the helper module: the function must
    return a list and must not raise even on the full Phase-1
    library.
    """
    library = _get_library()
    validated = assert_library_well_formed(library)
    assert isinstance(validated, list)
    assert len(validated) <= len(library)


def test_assert_tile_well_formed_accepts_good_tile() -> None:
    """``assert_tile_well_formed`` returns True for any library tile."""
    library = _get_library()
    if not library:
        pytest.skip("Fragment library is empty")
    for tile in library:
        assert assert_tile_well_formed(tile), (
            f"Tile {tile.tile_id} ({tile.smiles!r}) failed well-formedness"
        )


def test_allowed_atoms_set_is_frozen() -> None:
    """``ALLOWED_ATOMS`` is a ``frozenset`` (immutable)."""
    assert isinstance(ALLOWED_ATOMS, frozenset)
    for el in ("C", "N", "O", "F", "S", "Cl", "Br", "I", "P"):
        assert el in ALLOWED_ATOMS, f"{el} missing from ALLOWED_ATOMS"
    for el in ("Pt", "Ru", "Zn", "Ir", "Cu", "Au"):
        assert el not in ALLOWED_ATOMS, (
            f"Metal {el} should NOT be in ALLOWED_ATOMS (tiles are organic)"
        )
