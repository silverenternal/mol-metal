"""Smoke tests for the MLC Atom layer (atoms/combinators.py)."""
from __future__ import annotations

import pytest

from molmetal_lam.atoms.combinators import (
    Atom,
    METAL_ATOMS,
    PRIMITIVE_ATOMS,
    assert_well_formed,
    from_smiles,
    make_ligand,
    sanity_check,
)


def test_sanity_check_all_true() -> None:
    report = sanity_check()
    assert all(report.values()), f"sanity_check failed: {report}"


def test_assert_well_formed_does_not_raise() -> None:
    assert_well_formed()


def test_primitive_atoms_have_expected_arity() -> None:
    assert PRIMITIVE_ATOMS["H"].arity == 1
    assert PRIMITIVE_ATOMS["C"].arity == 4
    assert PRIMITIVE_ATOMS["N"].arity == 4   # valence 3 + 1 lone pair
    assert PRIMITIVE_ATOMS["O"].arity == 4   # valence 2 + 2 lone pairs
    assert PRIMITIVE_ATOMS["F"].arity == 4   # valence 1 + 3 lone pairs
    assert PRIMITIVE_ATOMS["Cl"].arity == 4


def test_metal_atoms_have_expected_arity() -> None:
    assert METAL_ATOMS["Pt_II"].arity == 4
    assert METAL_ATOMS["Ru_II"].arity == 6
    assert METAL_ATOMS["Zn_II"].arity == 4
    assert METAL_ATOMS["Ir_III"].arity == 6
    assert METAL_ATOMS["Cu_II"].arity == 4
    assert METAL_ATOMS["Au_III"].arity == 4   # valence 2 + 2 lone pairs; square-planar


def test_metal_geometries() -> None:
    assert METAL_ATOMS["Pt_II"].geometry == "square_planar"
    assert METAL_ATOMS["Ru_II"].geometry == "octahedral"
    assert METAL_ATOMS["Zn_II"].geometry == "tetrahedral"


def test_is_saturated() -> None:
    pt = METAL_ATOMS["Pt_II"]
    assert pt.is_saturated(4)
    assert not pt.is_saturated(3)
    assert not pt.is_saturated(0)


def test_with_one_less_free_site_returns_new_atom() -> None:
    pt = METAL_ATOMS["Pt_II"]
    pt2 = pt.with_one_less_free_site()
    assert pt is not pt2
    assert pt.symbol == pt2.symbol
    assert pt.arity == pt2.arity  # intrinsic property preserved


def test_atom_is_hashable_and_frozen() -> None:
    pt = METAL_ATOMS["Pt_II"]
    # Frozen dataclass → should be hashable
    assert hash(pt) == hash(pt)
    # Setting an attribute on a frozen dataclass raises FrozenInstanceError
    with pytest.raises(Exception):
        pt.symbol = "X"  # type: ignore[misc]


def test_from_smiles_simple() -> None:
    atoms = from_smiles("CCO")
    assert [a.symbol for a in atoms] == ["C", "C", "O"]
    assert atoms[0].arity == 4


def test_from_smiles_includes_metal() -> None:
    atoms = from_smiles("N.N.Cl.Cl.[Pt]")
    assert atoms[-1].is_metal
    assert atoms[-1].arity == 4


def test_from_smiles_invalid_raises() -> None:
    with pytest.raises(ValueError):
        from_smiles("not-a-smiles-@@")


def test_make_ligand_nh3_is_saturated() -> None:
    nh3 = make_ligand("NH3")
    # NH3 is modelled as a saturated nitrogen (3 H bonds already
    # formed) acting as a *value* for dative bonding.
    assert nh3.valence == 0
    assert nh3.symbol == "NH3"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))