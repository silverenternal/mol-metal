# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 (T83) — the kekulize repair chain was a no-op, and it was silent.

WHY THIS EXISTS
---------------
``_assemble_mol`` called ``Chem.SanitizeMol(mol)`` and, in the ``except``
branch, handed the SAME ``mol`` to ``_demote_all_aromatic``.  But RDKit's
kekulizer rewrites every aromatic bond to SINGLE in place *before* it raises:
the repair found nothing left to demote, re-sanitised an already-saturated
graph, and recorded ``repair="kekulize_demoted_all"`` as if it had chosen to
drop the aromaticity.

MEASURED on the corpus's own bond classes, with the head replaced by the
answer (``metrics/by_metric/wfa113_t83_truth_roundtrip.json``): 63.7% of the
six-metal cohort took that path, so every aromatic ring was destroyed on
complexes the model had reproduced exactly.  Against the corpus's reference
SMILES the repaired molecules matched the aromatic-ring count on 43.1%, and
before the fix that number was 0 for the same 764 molecules.

Two rules make the repair real:

* snapshot the molecule BEFORE the first sanitize, and run every attempt on
  ``Chem.Mol(snapshot)`` — a failed ``SanitizeMol`` mutates its argument;
* resolve a partially-aromatic ring AS A RING (promote when at least half its
  bonds are already aromatic, demote otherwise), because a delocalised ring's
  C-N / C-O / C-S bonds sit near 1.2 and fall out of the aromatic band while
  its C-C bonds sit near 1.4 and stay in it.

RED CHECK: monkeypatch ``_resolve_partial_rings`` to the pre-T83 rule (demote
every partial ring) and ``test_pyridine_ring_keeps_its_aromaticity`` fails —
the decode comes back as a saturated ring.
"""

from __future__ import annotations

import unittest
from unittest import mock

from rdkit import Chem

from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    _aromatic_components,
    _demote_bond_set,
    _promote_bond_set,
    _resolve_partial_rings,
    _ring_bond_indices,
    _try_sanitize,
)


def _ring(zs, orders):
    """An UNSANITISED ring; ``orders`` is one entry per ring bond."""
    rw = Chem.RWMol()
    for z in zs:
        rw.AddAtom(Chem.Atom(int(z)))
    n = len(zs)
    for i, order in enumerate(orders):
        rw.AddBond(i, (i + 1) % n, order)
    return rw.GetMol()


S, D, A = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.AROMATIC


class FailedSanitizeDestroysTheMolecule(unittest.TestCase):
    """The property the pre-T83 repair chain did not know about."""

    def test_bonds_become_single_but_atom_flags_survive(self) -> None:
        mol = _ring([6] * 5, [A] * 5)
        self.assertIsNotNone(_try_sanitize(mol))
        self.assertTrue(
            all(b.GetBondType() == S for b in mol.GetBonds()),
            "RDKit rewrites aromatic bonds to SINGLE before raising",
        )
        self.assertTrue(
            all(a.GetIsAromatic() for a in mol.GetAtoms()),
            "the atom aromatic flags are NOT cleared, which is why a repair "
            "that inspects the mol after the failure sees no aromatic bonds "
            "but twelve aromatic atoms and gives up",
        )

    def test_a_copy_is_isolated(self) -> None:
        mol = _ring([6] * 5, [A] * 5)
        self.assertIsNotNone(_try_sanitize(Chem.Mol(mol)))
        self.assertTrue(all(b.GetBondType() == A for b in mol.GetBonds()))

    def test_success_returns_none(self) -> None:
        self.assertIsNone(_try_sanitize(_ring([6] * 6, [A] * 6)))


class ResolvePartialRings(unittest.TestCase):
    def test_pyridine_is_promoted_not_demoted(self) -> None:
        # C-N bonds sit near BO 1.2 (SINGLE) and C-C near 1.4 (AROMATIC), so
        # a pyridine ring arrives as 4 aromatic + 2 single.
        mol = _ring([7, 6, 6, 6, 6, 6], [S, A, A, A, A, S])
        self.assertEqual(_resolve_partial_rings(mol), 1)
        self.assertTrue(all(b.GetBondType() == A for b in mol.GetBonds()))

    def test_a_mostly_saturated_ring_is_demoted(self) -> None:
        mol = _ring([6] * 6, [A, A, S, S, S, S])
        self.assertEqual(_resolve_partial_rings(mol), 1)
        self.assertTrue(all(b.GetBondType() == S for b in mol.GetBonds()))

    def test_a_fully_aromatic_ring_is_left_alone(self) -> None:
        mol = _ring([6] * 6, [A] * 6)
        self.assertEqual(_resolve_partial_rings(mol), 0)

    def test_a_saturated_ring_is_left_alone(self) -> None:
        mol = _ring([6] * 6, [S] * 6)
        self.assertEqual(_resolve_partial_rings(mol), 0)


class AromaticComponents(unittest.TestCase):
    def test_fused_bonds_form_one_component(self) -> None:
        # naphthalene-shaped: two six-rings sharing an edge, all aromatic.
        rw = Chem.RWMol()
        for _ in range(10):
            rw.AddAtom(Chem.Atom(6))
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
                 (4, 6), (6, 7), (7, 8), (8, 9), (9, 3)]
        for a, b in edges:
            rw.AddBond(a, b, A)
        comps = _aromatic_components(rw.GetMol())
        self.assertEqual(len(comps), 1)
        self.assertEqual(len(comps[0]), len(edges))

    def test_disjoint_rings_are_separate_components(self) -> None:
        rw = Chem.RWMol()
        for _ in range(12):
            rw.AddAtom(Chem.Atom(6))
        for base in (0, 6):
            for i in range(6):
                rw.AddBond(base + i, base + (i + 1) % 6, A)
        self.assertEqual(len(_aromatic_components(rw.GetMol())), 2)


class RingBondIndices(unittest.TestCase):
    def test_one_entry_per_ring_with_every_bond(self) -> None:
        mol = _ring([6] * 6, [S] * 6)
        rings = _ring_bond_indices(mol)
        self.assertEqual(len(rings), 1)
        self.assertEqual(len(rings[0]), 6)
        self.assertEqual(len(set(rings[0])), 6)


class EndToEndRepair(unittest.TestCase):
    """Drive ``_assemble_mol`` the way ``decode`` does."""

    def _assemble(self, mol):
        import torch

        from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
            YuelBondBondDecoder,
        )
        from molmetal.models.bond_head import AtomCloud

        zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
        orders = []
        for b in mol.GetBonds():
            order = {S: 1, D: 2, A: 4}.get(b.GetBondType(), 1)
            orders.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), order))
        cloud = AtomCloud(
            positions=torch.zeros((len(zs), 3), dtype=torch.float32),
            atomic_numbers=torch.tensor(zs, dtype=torch.long),
        )
        return YuelBondBondDecoder()._assemble_mol(cloud, orders, [])

    def test_pyridine_ring_keeps_its_aromaticity(self) -> None:
        mol = _ring([7, 6, 6, 6, 6, 6], [S, A, A, A, A, S])
        res = self._assemble(mol)
        self.assertTrue(res.sanitized)
        self.assertEqual(res.repair, "kekulize_resolved_partial_rings")
        self.assertTrue(any(c.islower() for c in res.smiles), res.smiles)

    def test_pyridine_fails_under_the_pre_t83_rule(self) -> None:
        mol = _ring([7, 6, 6, 6, 6, 6], [S, A, A, A, A, S])

        def demote_only(m):
            n = 0
            for idxs in _ring_bond_indices(m):
                types = [m.GetBondWithIdx(i).GetBondType() for i in idxs]
                n_arom = sum(1 for t in types if t == A)
                if 0 < n_arom < len(idxs):
                    _demote_bond_set(m, idxs)
                    n += 1
            return n

        with mock.patch(
            "molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond."
            "_resolve_partial_rings",
            side_effect=demote_only,
        ):
            res = self._assemble(mol)
        self.assertTrue(res.sanitized)
        self.assertFalse(any(c.islower() for c in res.smiles), res.smiles)

    def test_a_bad_ring_does_not_destroy_the_good_one(self) -> None:
        # A benzene plus an all-carbon 5-ring that cannot be kekulized.  The
        # pre-T83 chain dropped both; the ring-local repair keeps the benzene.
        rw = Chem.RWMol()
        for _ in range(11):
            rw.AddAtom(Chem.Atom(6))
        for i in range(6):
            rw.AddBond(i, (i + 1) % 6, A)
        for i in range(5):
            rw.AddBond(6 + i, 6 + (i + 1) % 5, A)
        res = self._assemble(rw.GetMol())
        self.assertTrue(res.sanitized)
        self.assertTrue(any(c.islower() for c in res.smiles), res.smiles)


class PromoteBondSet(unittest.TestCase):
    def test_promotion_flags_both_endpoints(self) -> None:
        mol = _ring([6] * 6, [S, A, A, A, A, S])
        _promote_bond_set(mol, [b.GetIdx() for b in mol.GetBonds()])
        for atom in mol.GetAtoms():
            self.assertTrue(atom.GetIsAromatic())
        for bond in mol.GetBonds():
            self.assertTrue(bond.GetIsAromatic())


if __name__ == "__main__":
    unittest.main()
