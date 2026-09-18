"""Tests for the MLC Bond layer (bonds-as-application).

Covers the four application rules (covalent / dative / aromatic /
hydrogen), the dispatcher, the validity predicate, and the beta-reduction
into a molecule-like dict.
"""

from __future__ import annotations

import pytest

from molmetal_lam.atoms.combinators import METAL_ATOMS, PRIMITIVE_ATOMS, make_ligand
from molmetal_lam.bonds.application import (
    AROMATIC,
    COVALENT,
    DATIVE,
    HYDROGEN,
    Bond,
    BondError,
    FreeSiteLedger,
    assemble,
    bond,
    can_bond,
    cisplatin,
    distinct,
    free_sites,
    is_valid,
    resolve_atom,
    saturate,
)


@pytest.fixture()
def led() -> FreeSiteLedger:
    """A private ledger so library singletons are never mutated."""
    return FreeSiteLedger()


def C(): return distinct(PRIMITIVE_ATOMS["C"])
def H(): return distinct(PRIMITIVE_ATOMS["H"])
def O(): return distinct(PRIMITIVE_ATOMS["O"])
def N(): return distinct(PRIMITIVE_ATOMS["N"])
def Cl(): return distinct(PRIMITIVE_ATOMS["Cl"])
def Pt(): return distinct(METAL_ATOMS["Pt_II"])


# ---------------------------------------------------------------- covalent

def test_covalent_decrements_both_atoms(led):
    c, h = C(), H()
    b = Bond.covalent(c, h, ledger=led)
    assert b.is_covalent and not b.is_dative
    assert free_sites(c, led) == 3
    assert free_sites(h, led) == 0
    assert is_valid(b)


def test_double_bond_is_nested_application(led):
    c, o = C(), O()
    b = Bond.covalent(c, o, order=2, ledger=led)
    assert b.order == 2 and b.effective_order == 2.0
    assert free_sites(c, led) == 2
    assert free_sites(o, led) == 2  # O arity 4 (valence 2 + 2 lone pairs)
    assert is_valid(b)


def test_covalent_refuses_oversaturation(led):
    h, c = H(), C()
    Bond.covalent(h, c, ledger=led)
    with pytest.raises(BondError):
        Bond.covalent(h, C(), ledger=led)


def test_no_self_application(led):
    c = C()
    with pytest.raises(BondError):
        Bond.covalent(c, c, ledger=led)


# ------------------------------------------------------------------ dative

def test_dative_is_curried_metal_keeps_free_sites(led):
    """Pt(NH3) must keep 3 free sites — the defining MLC currying claim."""
    pt, nh3 = Pt(), make_ligand("NH3")
    b = Bond.dative(nh3, pt, ledger=led)
    assert b.is_dative and b.kind == DATIVE
    assert b.acceptor is pt and b.donor is nh3
    assert free_sites(pt, led) == 3        # curried: arity 4 -> 3
    assert free_sites(nh3, led) == 0       # ligand collapsed to a value
    assert is_valid(b)


def test_dative_auto_orients_metal_as_acceptor(led):
    """Bond.dative(metal, ligand) is accepted and swapped internally."""
    pt, nh3 = Pt(), make_ligand("NH3")
    b = Bond.dative(pt, nh3, ledger=led)
    assert b.acceptor is pt and b.donor is nh3
    assert free_sites(pt, led) == 3


def test_dative_none_resolves_to_anonymous_donor(led):
    """A lookup miss (METAL_ATOMS.get('N') -> None) still forms a bond."""
    pt = Pt()
    b = Bond.dative(pt, None, ledger=led)
    assert b.is_dative and free_sites(pt, led) == 3


def test_dative_refuses_acceptor_without_lone_pairs(led):
    with pytest.raises(BondError):
        Bond.dative(make_ligand("NH3"), C(), ledger=led)  # C has 0 lone pairs


def test_dative_refuses_saturated_metal(led):
    pt = Pt()
    for _ in range(4):
        Bond.dative(make_ligand("NH3"), pt, ledger=led)
    assert free_sites(pt, led) == 0
    with pytest.raises(BondError):
        Bond.dative(make_ligand("NH3"), pt, ledger=led)


def test_coordinate_alias(led):
    pt = Pt()
    b = Bond.coordinate(pt, make_ligand("NH3"), ledger=led)
    assert b.acceptor is pt and free_sites(pt, led) == 3


# ---------------------------------------------------------------- aromatic

def test_benzene_gives_six_aromatic_bonds(led):
    ring = [C() for _ in range(6)]
    bonds = Bond.aromatic(ring, ledger=led)
    assert len(bonds) == 6
    assert all(b.kind == AROMATIC and b.effective_order == 1.5 for b in bonds)
    # every ring atom participates in exactly two ring bonds
    assert all(free_sites(a, led) == 2 for a in ring)
    assert all(is_valid(b) for b in bonds)


def test_pyridine_ring(led):
    ring = [C() for _ in range(5)] + [N()]
    bonds = Bond.aromatic(ring, ledger=led)
    assert len(bonds) == 6


def test_aromatic_requires_distinct_objects(led):
    c = C()
    with pytest.raises(BondError):
        Bond.aromatic([c] * 6, ledger=led)


def test_aromatic_requires_three_atoms(led):
    with pytest.raises(BondError):
        Bond.aromatic([C(), C()], ledger=led)


# ---------------------------------------------------------------- hydrogen

def test_hydrogen_requires_saturated_partners(led):
    n, o = N(), O()
    with pytest.raises(BondError):
        Bond.hydrogen(n, o, ledger=led)
    saturate(n, led)
    saturate(o, led)
    b = Bond.hydrogen(n, o, ledger=led)
    assert b.kind == HYDROGEN and b.order == 0 and b.effective_order == 0.0
    assert is_valid(b)
    # no free sites were consumed (both were already values)
    assert free_sites(n, led) == 0 and free_sites(o, led) == 0


# -------------------------------------------------------------- dispatcher

def test_dispatcher_picks_dative_for_metal(led):
    b = bond(Pt(), make_ligand("NH3"), ledger=led)
    assert b.kind == DATIVE


def test_dispatcher_picks_covalent_for_two_organics(led):
    b = bond(C(), H(), ledger=led)
    assert b.kind == COVALENT


def test_dispatcher_picks_hydrogen_for_two_values(led):
    n, o = N(), O()
    saturate(n, led)
    saturate(o, led)
    assert bond(n, o, ledger=led).kind == HYDROGEN


def test_dispatcher_raises_when_nothing_fits(led):
    h1, h2 = H(), H()
    Bond.covalent(h1, C(), ledger=led)          # h1 now saturated
    with pytest.raises(BondError):
        bond(h1, h2, ledger=led)                 # saturated + unsaturated H


# ------------------------------------------------------------------ can_bond

def test_can_bond_is_non_mutating(led):
    c, h = C(), H()
    assert can_bond(c, h, ledger=led)
    assert free_sites(c, led) == 4 and free_sites(h, led) == 1


def test_can_bond_rejects_impossible_order(led):
    assert not can_bond(C(), H(), order=2, ledger=led)


def test_can_bond_unknown_kind(led):
    with pytest.raises(BondError):
        can_bond(C(), H(), kind="ionic", ledger=led)


# ------------------------------------------------------------- beta-reduce

def test_apply_returns_molecule_dict(led):
    pt, nh3 = Pt(), make_ligand("NH3")
    mol = Bond.dative(nh3, pt, ledger=led).apply(led)
    assert set(mol) >= {"atoms", "bonds", "free_sites_per_atom", "is_closed"}
    assert len(mol["atoms"]) == 2 and len(mol["bonds"]) == 1
    assert sum(mol["free_sites_per_atom"].values()) == 3
    assert mol["is_closed"] is False       # Pt still expects 3 arguments
    assert mol["valid"] is True


def test_assemble_water_is_closed_term(led):
    o, h1, h2 = O(), H(), H()
    bonds = [Bond.covalent(o, h1, ledger=led), Bond.covalent(o, h2, ledger=led)]
    mol = assemble(bonds, ledger=led)
    assert len(mol["atoms"]) == 3
    # O keeps its 2 lone pairs as free dative/H-bond sites
    assert mol["open_sites"] == 2
    saturate(o, led)
    assert assemble(bonds, ledger=led)["is_closed"] is True


def test_cisplatin_is_closed_beta_normal_form():
    mol = cisplatin()
    assert mol["is_closed"] is True
    assert mol["valid"] is True
    assert len(mol["bonds"]) == 4
    assert all(b.is_dative for b in mol["bonds"])
    assert mol["term"] == "((((Pt_II NH3) NH3) Cl) Cl)"


# ---------------------------------------------------------------- is_valid

def test_is_valid_rejects_handmade_bond_without_snapshot():
    assert not is_valid(Bond(atom_a=C(), atom_b=H()))
    assert not is_valid("not a bond")


def test_is_valid_rejects_unknown_kind():
    b = Bond(atom_a=C(), atom_b=H(), kind="ionic",
             free_a_before=4, free_b_before=1)
    assert not is_valid(b)


# ---------------------------------------------------------------- resolution

def test_resolve_atom_accepts_symbols_and_none():
    assert resolve_atom("C").symbol == "C"
    assert resolve_atom("Pt_II").is_metal
    assert resolve_atom(None).symbol == "NH3"
    with pytest.raises(BondError):
        resolve_atom(3.14)


def test_distinct_is_value_equal_but_object_distinct():
    a = PRIMITIVE_ATOMS["C"]
    b = distinct(a)
    assert a == b and a is not b
