"""Tests for the tile_lib module (Tile dataclass + 12 click tiles)."""
from __future__ import annotations

import dataclasses

import pytest

from molmetal_lam.tile_lib.tile import Tile
from molmetal_lam.tile_lib.click_tiles import (
    ALKYNE_TILES,
    AZIDE_TILES,
    PARTNER_TILES,
    STANDARD_12_TILES,
    build_all_click_tiles,
)
from molmetal_lam.tile_lib.library import (
    ALKYNES,
    AZIDES,
    PARTNERS,
    STANDARD_12,
    build_tile_dict,
    build_tile_library,
)


# ---------------------------------------------------------------------------
# Smoke / imports
# ---------------------------------------------------------------------------


def test_imports() -> None:
    """All public symbols are importable."""
    assert Tile is not None
    assert callable(build_tile_library)
    assert callable(build_tile_dict)
    assert isinstance(STANDARD_12, list)


# ---------------------------------------------------------------------------
# Tile dataclass
# ---------------------------------------------------------------------------


def test_tile_dataclass_creation() -> None:
    """A Tile instance is constructible from the minimum required fields."""
    t = Tile(smiles="CCO", functional_groups=["alcohol"])
    assert t.smiles == "CCO"
    assert t.functional_groups == ["alcohol"]
    # Auto-generated id is non-empty
    assert isinstance(t.tile_id, str) and len(t.tile_id) > 0
    # Default physico-chemical descriptors
    assert t.mw == 0.0
    assert t.logp == 0.0
    assert t.tpsa == 0.0
    assert t.sas_score == 1.0
    # coords defaults to None
    assert t.coords is None
    # dataclass with frozen=True
    assert dataclasses.is_dataclass(Tile)
    assert Tile.__dataclass_params__.frozen is True


def test_tile_has_group() -> None:
    """``Tile.has_group`` returns True iff the tag is in ``functional_groups``."""
    azide_tile = Tile(
        smiles="CCN=[N+]=[N-]",
        functional_groups=["azide"],
    )
    assert azide_tile.has_group("azide") is True
    assert azide_tile.has_group("terminal_alkyne") is False

    # Empty tag list -> no group is present
    empty_tile = Tile(smiles="CCO", functional_groups=[])
    assert empty_tile.has_group("anything") is False


# ---------------------------------------------------------------------------
# Click-tile library
# ---------------------------------------------------------------------------


def test_click_tiles_count() -> None:
    """``build_tile_library`` returns exactly 12 tiles for Phase 0."""
    tiles = build_tile_library(n_max=12)
    assert len(tiles) == 12
    # And the eager constants agree.
    assert len(STANDARD_12) == 12
    assert len(STANDARD_12_TILES()) == 12


def test_azide_count() -> None:
    """At least 4 tiles carry the ``'azide'`` tag."""
    azides = AZIDE_TILES()
    assert len(azides) >= 4
    assert all(t.has_group("azide") for t in azides)
    # And the AZIDES slice from library.py is consistent.
    assert len(AZIDES) >= 4
    assert all(t.has_group("azide") for t in AZIDES)


def test_alkyne_count() -> None:
    """At least 4 tiles carry ``'terminal_alkyne'`` or ``'cyclooctyne'``."""
    alkynes = ALKYNE_TILES()
    assert len(alkynes) >= 4
    for t in alkynes:
        ok = t.has_group("terminal_alkyne") or t.has_group("cyclooctyne")
        assert ok, f"Tile {t} has neither terminal_alkyne nor cyclooctyne"
    # And the ALKYNES slice from library.py agrees.
    assert len(ALKYNES) >= 4


# ---------------------------------------------------------------------------
# Sanity: build_tile_dict + slice consistency
# ---------------------------------------------------------------------------


def test_build_tile_dict_round_trip() -> None:
    """``build_tile_dict`` returns one entry per ``tile_id``."""
    tiles = build_tile_library(n_max=12)
    d = build_tile_dict(n_max=12)
    assert len(d) == 12
    # All 12 tile_ids appear in the dict
    for t in tiles:
        assert t.tile_id in d
        # And the looked-up tile has the same canonical SMILES + tags.
        looked = d[t.tile_id]
        assert looked.smiles == t.smiles
        assert looked.functional_groups == t.functional_groups


def test_partner_tiles_have_expected_tags() -> None:
    """Partner tiles cover the remaining reaction classes (Diels-Alder,
    Staudinger phosphine, thiol-ene maleimide)."""
    partners = PARTNER_TILES()
    assert len(partners) == 4
    expected_tags = {"phosphine", "diene", "dienophile", "maleimide"}
    seen_tags = set()
    for t in partners:
        for tag in expected_tags:
            if t.has_group(tag):
                seen_tags.add(tag)
    assert seen_tags == expected_tags


def test_full_library_no_duplicate_ids() -> None:
    """All 12 tiles have distinct tile_ids."""
    tiles = build_tile_library(n_max=12)
    ids = [t.tile_id for t in tiles]
    assert len(set(ids)) == 12
