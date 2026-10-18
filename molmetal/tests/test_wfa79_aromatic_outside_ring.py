"""Wf-A79 — aromatic-outside-ring demotion at decode time.

The L3 decode A/B arm (molmetal/reports/cfm_l3_decode_enabled_ab/
control/arm_control.json) recorded 57/64 clouds failing with
AtomKekulizeException / KekulizeException -- 89.1% of the arm -- and a
decode_rate of 0.109.  The message names the cause: "non-ring atom N
marked aromatic".  These tests pin the repair and its no-op cases.
"""
from __future__ import annotations

from rdkit import Chem

from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    _demote_aromatic_outside_rings,
)


def _aromatic_bond(rw, i, j):
    rw.AddBond(i, j, Chem.BondType.AROMATIC)
    rw.GetBondBetweenAtoms(i, j).SetIsAromatic(True)


def _benzene():
    rw = Chem.RWMol()
    for _ in range(6):
        a = Chem.Atom(6)
        a.SetIsAromatic(True)
        rw.AddAtom(a)
    for i in range(6):
        _aromatic_bond(rw, i, (i + 1) % 6)
    return rw.GetMol()


def _aromatic_chain(n=3):
    rw = Chem.RWMol()
    for _ in range(n):
        a = Chem.Atom(6)
        a.SetIsAromatic(True)
        rw.AddAtom(a)
    for i in range(n - 1):
        _aromatic_bond(rw, i, i + 1)
    return rw.GetMol()


def _n_aromatic_bonds(mol):
    return sum(
        1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC
    )


def test_benzene_is_unchanged():
    mol = _benzene()
    before = _n_aromatic_bonds(mol)
    assert before == 6
    _demote_aromatic_outside_rings(mol)
    assert _n_aromatic_bonds(mol) == 6


def test_aromatic_chain_is_demoted_and_sanitizes():
    mol = _aromatic_chain(3)
    assert _n_aromatic_bonds(mol) == 2
    _demote_aromatic_outside_rings(mol)
    assert _n_aromatic_bonds(mol) == 0
    # The whole point: the molecule survives sanitisation instead of
    # being lost to the kekulizer.
    Chem.SanitizeMol(mol)
    assert Chem.MolToSmiles(mol) == "CCC"


def test_non_aromatic_mol_is_unchanged():
    rw = Chem.RWMol()
    for _ in range(3):
        rw.AddAtom(Chem.Atom(6))
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    rw.AddBond(1, 2, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    _demote_aromatic_outside_rings(mol)
    assert all(
        b.GetBondType() == Chem.BondType.SINGLE for b in mol.GetBonds()
    )


def test_non_ring_atom_flags_are_cleared():
    mol = _aromatic_chain(3)
    assert all(a.GetIsAromatic() for a in mol.GetAtoms())
    _demote_aromatic_outside_rings(mol)
    assert not any(a.GetIsAromatic() for a in mol.GetAtoms())


def test_ring_bonds_survive_a_dangling_aromatic_bond():
    # Benzene plus one aromatic bond hanging off atom 0 (to a 7th atom).
    rw = Chem.RWMol(_benzene())
    a = Chem.Atom(6)
    a.SetIsAromatic(True)
    rw.AddAtom(a)
    _aromatic_bond(rw, 0, 6)
    mol = rw.GetMol()
    assert _n_aromatic_bonds(mol) == 7
    _demote_aromatic_outside_rings(mol)
    assert _n_aromatic_bonds(mol) == 6  # the ring is untouched
    assert mol.GetBondBetweenAtoms(0, 6).GetBondType() == Chem.BondType.SINGLE
