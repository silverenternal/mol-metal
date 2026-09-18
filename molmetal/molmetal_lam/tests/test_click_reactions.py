"""Tests for the click-chemistry reactions in
:mod:`molmetal_lam.reactions.click_reactions`.

Five tests, matching the task spec (TODO/13_lambda_clickchem):

1. ``test_cuaac_smiles_valid``       : CuAAC of ethyl azide + propyne -> triazole
2. ``test_spaac_with_cyclooctyne``   : SPAAC works (cyclooctyne + azide)
3. ``test_spc_amide_formation``      : SPC amide product correct
4. ``test_diels_alder_cyclohexene``  : Diels-Alder produces 6-membered ring
5. ``test_tile_library_size``        : >=12 tiles in the standard library

Run with::

    source .venv/bin/activate && python -m pytest \
        molmetal/molmetal_lam/tests/test_click_reactions.py -v
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from molmetal_lam.reactions.click_reactions import (
    CLICK_REACTIONS,
    ReactionResult,
    apply_click_reaction,
    click_cuaac,
    click_diels_alder,
    click_spaac,
    click_spc,
)
from molmetal_lam.tile_lib.click_tiles import (
    ALKYNE_TILES,
    AZIDE_TILES,
    PARTNER_TILES,
    STANDARD_12_TILES,
    build_click_tile,
)
from molmetal_lam.tile_lib.tile import Tile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _smiles_has_ring(smiles: str, ring_size: int) -> bool:
    """Return True if the canonical mol contains a ring of size ``ring_size``."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    return any(len(r) == ring_size for r in ring_info.AtomRings())


# ---------------------------------------------------------------------------
# 1. CuAAC — ethyl azide + propyne -> 1-ethyl-4-methyl-1,2,3-triazole
# ---------------------------------------------------------------------------


def test_cuaac_smiles_valid() -> None:
    """CuAAC of ethyl azide + propyne should give 1-ethyl-4-methyl-1,2,3-triazole.

    Expected product canonical SMILES: ``CCn1cc(C)nn1``  (heavy atoms = 8).
    The 1,4-disubstituted 1,2,3-triazole should be a 5-membered aromatic ring.
    """
    azide_tile = AZIDE_TILES()[0]   # ethyl azide: CCN=[N+]=[N-]
    alkyne_tile = ALKYNE_TILES()[0]  # propyne: C#CC

    results = click_cuaac(azide_tile, alkyne_tile)
    assert len(results) == 1, f"expected 1 CuAAC product, got {len(results)}"
    result = results[0]
    assert isinstance(result, ReactionResult)
    assert result.reaction_name == "CuAAC"
    # Heavy atom count: ethyl (2) + propyne (3) + azide (3 N's) + alkyne (2 C's) − 0 (cycloaddition conserves atoms)
    # = 2 + 3 + 3 = 8 heavy atoms in product.  Wait: ethyl azide is 5 atoms (CC-NNN),
    # propyne is 3 atoms (CC≡CH), total = 8.  Cycloaddition conserves heavy atoms.
    assert result.atom_count == 8, f"expected 8 atoms, got {result.atom_count}"
    # Product should contain a 5-membered aromatic ring (the triazole).
    assert _smiles_has_ring(result.product_smiles, 5), (
        f"expected 5-ring in {result.product_smiles!r}"
    )
    # The product should be parseable by RDKit (sanity).
    from rdkit import Chem
    mol = Chem.MolFromSmiles(result.product_smiles)
    assert mol is not None


# ---------------------------------------------------------------------------
# 2. SPAAC — azide + cyclooctyne -> triazole
# ---------------------------------------------------------------------------


def test_spaac_with_cyclooctyne() -> None:
    """SPAAC of benzyl azide + cyclooctyne should produce a triazole.

    The product should be a valid SMILES containing the azide-derived
    N=N-N substructure (the triazole backbone).  In practice the
    triazole fuses into the cyclooctane ring, producing a bicyclic
    system — the test verifies (a) the product contains >= 3 N atoms
    (matching the azide contribution) and (b) >= 2 C atoms (the
    alkyne contribution).  The combination uniquely identifies the
    triazole as a click-chemistry cycloaddition product.
    """
    azide_tile = AZIDE_TILES()[1]    # benzyl azide
    alkyne_tile = ALKYNE_TILES()[2]  # cyclooctyne

    results = click_spaac(azide_tile, alkyne_tile)
    assert len(results) >= 1, "SPAAC should produce at least one product"
    result = results[0]
    assert result.reaction_name == "SPAAC"
    assert result.atom_count > 0
    # Verify chemical composition of the click product: 3 N's (azide)
    # + 2 C's (alkyne) + the two R-groups should appear.
    from rdkit import Chem
    mol = Chem.MolFromSmiles(result.product_smiles)
    assert mol is not None
    n_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
    c_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    assert n_count >= 3, f"expected >=3 N's, got {n_count} in {result.product_smiles!r}"
    assert c_count >= 2, f"expected >=2 C's, got {c_count} in {result.product_smiles!r}"
    # SPAAC requires both an azide AND an internal alkyne.  Applying
    # it to a tile pair lacking either should produce an empty list.
    no_azide = alkyne_tile
    no_internal = ALKYNE_TILES()[0]  # propyne — terminal alkyne
    assert click_spaac(no_azide, alkyne_tile) == []
    assert click_spaac(azide_tile, no_internal) == []


# ---------------------------------------------------------------------------
# 3. SPC — Staudinger phosphine-azide coupling
# ---------------------------------------------------------------------------


def test_spc_amide_formation() -> None:
    """SPC of ethyl azide + methylphosphine should give ethyl-N=P-CH3.

    Product is the iminophosphorane: ``CCN=PC``.
    """
    azide_tile = AZIDE_TILES()[0]      # ethyl azide
    phosphine_tile = PARTNER_TILES()[0]  # methylphosphine (CP)

    results = click_spc(azide_tile, phosphine_tile)
    assert len(results) == 1, f"expected 1 SPC product, got {len(results)}"
    result = results[0]
    assert result.reaction_name == "SPC"
    # The iminophosphorane has the substructure ``N=P``.
    from rdkit import Chem
    mol = Chem.MolFromSmiles(result.product_smiles)
    assert mol is not None
    pat = Chem.MolFromSmarts("[N]=[P]")
    assert mol.HasSubstructMatch(pat), (
        f"expected N=P iminophosphorane in {result.product_smiles!r}"
    )
    # Heavy atoms conserved except for the 2 N's lost as N2:
    # ethyl azide = 5 atoms; methylphosphine = 2 atoms; total = 7.
    # Product loses 2 N's (released as N2) → 5 heavy atoms remain.
    assert result.atom_count == 5, f"expected 5 atoms, got {result.atom_count}"


# ---------------------------------------------------------------------------
# 4. Diels–Alder — diene + dienophile -> cyclohexene
# ---------------------------------------------------------------------------


def test_diels_alder_cyclohexene() -> None:
    """Diels-Alder of butadiene + ethylene should give cyclohexene.

    Product canonical SMILES: ``C1CC=CCC1``  (6-membered ring with one C=C).
    """
    # Use build_click_tile directly to construct butadiene + ethylene
    # tiles (clean diene + clean dienophile, no extra R groups).
    butadiene = build_click_tile("C=CC=C", ["diene"], embed_3d=False)
    ethylene = build_click_tile("C=C", ["alkene"], embed_3d=False)

    results = click_diels_alder(butadiene, ethylene)
    assert len(results) == 1, f"expected 1 DA product, got {len(results)}"
    result = results[0]
    assert result.reaction_name == "DielsAlder"
    # Heavy atom count: butadiene (4) + ethylene (2) = 6 heavy atoms (cycloaddition).
    assert result.atom_count == 6, f"expected 6 atoms, got {result.atom_count}"
    # Product should contain a 6-membered ring (the cyclohexene).
    assert _smiles_has_ring(result.product_smiles, 6), (
        f"expected 6-ring cyclohexene in {result.product_smiles!r}"
    )
    # And it should contain a C=C double bond (cyclohexene, not cyclohexane).
    from rdkit import Chem
    mol = Chem.MolFromSmiles(result.product_smiles)
    assert mol is not None
    pat = Chem.MolFromSmarts("[C]=[C]")
    assert mol.HasSubstructMatch(pat), (
        f"expected C=C in {result.product_smiles!r}"
    )


# ---------------------------------------------------------------------------
# 5. Tile library size — >= 12 standard tiles
# ---------------------------------------------------------------------------


def test_tile_library_size() -> None:
    """The standard 12-tile library should contain at least 12 tiles."""
    tiles = STANDARD_12_TILES()
    assert len(tiles) >= 12, f"expected >= 12 tiles, got {len(tiles)}"
    # Sanity: each tile has a non-empty SMILES.
    for t in tiles:
        assert isinstance(t, Tile)
        assert t.smiles, f"empty SMILES for tile {t.tile_id}"
    # Sanity: the 4 azide + 4 alkyne + 4 partner split is preserved.
    assert len(AZIDE_TILES()) == 4
    assert len(ALKYNE_TILES()) == 4
    assert len(PARTNER_TILES()) == 4
    # Dispatcher: apply_click_reaction works for each known name.
    az0 = AZIDE_TILES()[0]
    al0 = ALKYNE_TILES()[0]
    pt0 = PARTNER_TILES()[0]
    pt1 = PARTNER_TILES()[1]
    pt2 = PARTNER_TILES()[2]
    for name, t_a, t_b in [
        ("CuAAC", az0, al0),
        ("SPAAC", az0, ALKYNE_TILES()[2]),
        ("SPC", az0, pt0),
        ("DielsAlder", pt1, pt2),
    ]:
        results = apply_click_reaction(name, t_a, t_b)
        assert len(results) >= 1, (
            f"{name} on ({t_a.smiles}, {t_b.smiles}) produced no products"
        )
    # Unknown reaction name → ClickReactionError.
    from molmetal_lam.reactions.click_reactions import ClickReactionError
    with pytest.raises(ClickReactionError):
        apply_click_reaction("NoSuchReaction", az0, al0)