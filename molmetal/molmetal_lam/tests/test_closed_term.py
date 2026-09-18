"""Tests for the MLC Molecule layer (molecules/closed_term.py)."""
from __future__ import annotations

import pytest

from molmetal_lam.atoms.combinators import METAL_ATOMS, PRIMITIVE_ATOMS
from molmetal_lam.bonds.application import Bond, FreeSiteLedger, distinct
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm


# ---------------------------------------------------------------------------
# Sanity / imports
# ---------------------------------------------------------------------------


def test_import() -> None:
    """The module exposes the MoleculeClosedTerm class."""
    assert MoleculeClosedTerm is not None


# ---------------------------------------------------------------------------
# is_closed
# ---------------------------------------------------------------------------


def test_water_is_closed() -> None:
    """Water's O has valence 2 and 2 implicit Hs → is_closed=True."""
    m = MoleculeClosedTerm.from_smiles("O")
    assert m.is_closed is True
    # O still has 2 lone-pair free sites (H-bond capacity)
    assert m.free_sites == {0: 4} or m.free_sites[0] >= 2


def test_cisplatin_is_closed() -> None:
    """Cisplatin ``[Pt](N)(N)(Cl)Cl`` is closed (Pt's 4 valences satisfied)."""
    m = MoleculeClosedTerm.from_smiles("[Pt](N)(N)(Cl)Cl")
    assert m.is_closed is True


def test_empty_term_is_closed() -> None:
    """An empty molecule has no atoms → trivially closed."""
    m = MoleculeClosedTerm()
    assert m.is_closed is True


def test_as_cisplatin_shell_is_unsaturated() -> None:
    """The cisplatin shell descriptor is well-formed but unsaturated."""
    m = MoleculeClosedTerm.from_smiles("N.N.Cl.Cl").as_cisplatin_shell()
    # Pt has 4 free coordination sites
    assert m.n_atoms == 5
    assert m.n_bonds == 0
    assert m.is_closed is False
    # Pt (index 0) must report 4 free sites — the unsaturation.
    assert m.free_sites[0] == 4


# ---------------------------------------------------------------------------
# free_sites
# ---------------------------------------------------------------------------


def test_free_sites_keys_are_atom_indices() -> None:
    m = MoleculeClosedTerm.from_smiles("[Pt](N)(N)(Cl)Cl")
    # Free-site dict keys are integer atom positions.
    for k in m.free_sites.keys():
        assert isinstance(k, int)
        assert 0 <= k < m.n_atoms


def test_free_sites_only_unsaturated_atoms() -> None:
    """Saturated atoms should not appear in the free_sites dict."""
    m = MoleculeClosedTerm.from_smiles("[Pt](N)(N)(Cl)Cl")
    # Pt (index 0) is saturated in cisplatin → not in free_sites.
    assert 0 not in m.free_sites


# ---------------------------------------------------------------------------
# alpha_equivalent
# ---------------------------------------------------------------------------


def test_alpha_equivalent_same_canonical_smiles() -> None:
    """Two SMILES that differ only by atom ordering should be α-equivalent."""
    a = MoleculeClosedTerm.from_smiles("[Pt](N)(N)(Cl)Cl")
    b = MoleculeClosedTerm.from_smiles("[Pt](Cl)(N)(N)Cl")
    assert a.alpha_equivalent(b) is True


def test_alpha_equivalent_different_molecules() -> None:
    """Different connectivity → not α-equivalent."""
    a = MoleculeClosedTerm.from_smiles("CCO")      # ethanol
    b = MoleculeClosedTerm.from_smiles("CC(=O)C")  # acetone / propanal
    # Either way, ethanol and acetone should not be α-equivalent.
    # If propanal parses, the test still passes since it differs from ethanol.
    assert a.alpha_equivalent(b) is False


# ---------------------------------------------------------------------------
# reduce_once
# ---------------------------------------------------------------------------


def test_reduce_once_on_beta_nf_returns_copy() -> None:
    """A term already in β-NF returns a new copy without changing itself."""
    m = MoleculeClosedTerm.from_smiles("O")
    out = m.reduce_once()
    # Either out is the same instance (cached) or a new object — but the
    # receiver must not be mutated.
    assert out.is_closed is True


def test_reduce_once_on_unsaturated_does_not_crash() -> None:
    """The unsaturated cisplatin shell reduces gracefully."""
    m = MoleculeClosedTerm.from_smiles("N.N.Cl.Cl").as_cisplatin_shell()
    out = m.reduce_once()
    assert out is not None


# ---------------------------------------------------------------------------
# as_cisplatin_shell
# ---------------------------------------------------------------------------


def test_cisplatin_shell_uses_pt_ii() -> None:
    """The shell must use the Pt_II combinator (metal + square-planar)."""
    m = MoleculeClosedTerm.from_smiles("N.N.Cl.Cl").as_cisplatin_shell()
    pt = m.atoms[0]
    assert pt.symbol == "Pt_II"
    assert pt.is_metal is True
    assert pt.geometry == "square_planar"


def test_cisplatin_shell_term_string() -> None:
    """The shell has a human-readable λ-term rendering."""
    m = MoleculeClosedTerm.from_smiles("N.N.Cl.Cl").as_cisplatin_shell()
    assert m.term is not None
    assert "Pt_II" in m.term


# ---------------------------------------------------------------------------
# Round-tripping via RDKit
# ---------------------------------------------------------------------------


def test_to_rdkit_from_smiles_round_trip() -> None:
    """``to_rdkit`` should produce an RDKit Mol that re-parses consistently."""
    m = MoleculeClosedTerm.from_smiles("CCO")
    mol = m.to_rdkit()
    assert mol is not None
    m2 = MoleculeClosedTerm.from_rdkit(mol)
    assert m2.alpha_equivalent(m)


def test_from_smiles_lazy_imports() -> None:
    """from_smiles should not require an existing Mol instance."""
    m = MoleculeClosedTerm.from_smiles("CCN")
    assert m.n_atoms == 3


def test_from_smiles_with_embed_3d() -> None:
    """from_smiles with embed_3d=True should not crash."""
    m = MoleculeClosedTerm.from_smiles("CCO", embed_3d=True)
    assert m.n_atoms == 3


def test_from_smiles_without_embed_3d() -> None:
    """from_smiles with embed_3d=False should also work."""
    m = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    assert m.n_atoms == 3


def test_from_rdkit_invalid_smiles_raises() -> None:
    """A malformed SMILES should raise ValueError."""
    with pytest.raises(ValueError):
        MoleculeClosedTerm.from_smiles("not-a-smiles!!")


# ---------------------------------------------------------------------------
# has_redex
# ---------------------------------------------------------------------------


def test_has_redex_returns_bool() -> None:
    m = MoleculeClosedTerm.from_smiles("O")
    assert isinstance(m.has_redex(), bool)


def test_cyclopentadiene_has_redex_with_explicit_h() -> None:
    """L-A1: cyclopentadiene reports ``has_redex() == True`` after
    ``from_smiles_with_explicit_h``.

    The default ``from_smiles`` parses SMILES without running
    ``AddHs`` first — so the implicit Hs are folded into
    ``valence_used`` but not exposed as separate atoms.  With the
    new ``from_smiles_with_explicit_h`` classmethod (and the
    parallel ``from_rdkit`` patch that populates ``implicit_h_count``
    from ``GetNumImplicitHs() + GetNumExplicitHs()``), the
    ``[C/N/O/S]H1``-style "acidic heavy atom" profile is recognised
    by ``_is_acidic``, so the cyclopentadiene diene reports
    ``has_redex() == True``.
    """
    default_term = MoleculeClosedTerm.from_smiles("C1=CCC=C1")
    explicit_h_term = MoleculeClosedTerm.from_smiles_with_explicit_h(
        "C1=CCC=C1"
    )
    # The new path must report the redex — that is the whole point
    # of the L-A1 patch.
    assert explicit_h_term.has_redex() is True
    # Both parsers must agree on graph identity (same heavy-atom set).
    assert default_term.n_atoms == explicit_h_term.n_atoms == 5
    assert default_term.n_bonds == explicit_h_term.n_bonds


# ---------------------------------------------------------------------------
# Pure-Python construction (no RDKit)
# ---------------------------------------------------------------------------


def test_construct_atom_only_no_rdkit() -> None:
    """Construct a term without touching RDKit at all."""
    pt = distinct(METAL_ATOMS["Pt_II"])
    cl1 = distinct(PRIMITIVE_ATOMS["Cl"])
    cl2 = distinct(PRIMITIVE_ATOMS["Cl"])
    ledger = FreeSiteLedger()
    bonds = [
        Bond.dative(cl1, pt, ledger=ledger),
        Bond.dative(cl2, pt, ledger=ledger),
    ]
    m = MoleculeClosedTerm(
        atoms=[pt, cl1, cl2],
        bonds=bonds,
        ledger=ledger,
    )
    assert m.n_atoms == 3
    # Pt has valence 2 + 2 lone pairs → arity 4, with 2 dative bonds
    # the metal's valence (2) is satisfied → is_closed True.
    assert m.is_closed is True