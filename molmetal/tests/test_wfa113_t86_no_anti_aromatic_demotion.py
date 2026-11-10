# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 (T86) — the anti-aromatic (4n) demotion was removed.

WHY THIS EXISTS
---------------
Wf-A55 added ``_detect_and_demote_anti_aromatic`` to the selector: whenever a
``ring_size % 4 == 0`` all-aromatic ring closed, every other bond in it was
rewritten to ``BOND_SINGLE`` and a ``[wfA55-anti-aromatic] HONEST-NEGATIVE``
line was printed.  The premise was that RDKit cannot kekulize a 4n ring.

MEASURED on RDKit 2026.03.6 (this file's ``test_rdkit_kekulize_table``):

===========================  ==========================================
ring size, all aromatic      ``Chem.SanitizeMol``
===========================  ==========================================
4, 6, 8, 10, 12              succeeds
3, 5, 7, 9                   ``KekulizeException``
===========================  ==========================================

The rule therefore demoted exactly the rings RDKit handles and never touched
the odd rings that actually fail.  Two further problems: it mutated
``bond_orders`` in place — that list is returned to the caller, so the decode
artifact reported bond classes the head had not predicted — and it did not
credit ``remaining[]`` for the 0.5 it gave back per demoted bond, so a later
real bond could be refused.

The six-metal cohort contains 0 complexes with a 4n all-aromatic ring in its
true bond graph (``metrics/by_metric/wfa113_t83_truth_roundtrip.json``), so
the rule only ever fired on a head hallucination.

RED CHECK: re-add the demotion call to ``_select_bonds`` and
``test_a_four_membered_aromatic_ring_survives_assembly`` fails — the ring
comes back as cyclobutane instead of cyclobutadiene.
"""

from __future__ import annotations

import unittest

from rdkit import Chem

import molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond as M


def _all_aromatic_ring(n_atoms: int):
    rw = Chem.RWMol()
    for _ in range(n_atoms):
        rw.AddAtom(Chem.Atom(6))
    for i in range(n_atoms):
        rw.AddBond(i, (i + 1) % n_atoms, Chem.BondType.AROMATIC)
    return rw.GetMol()


class TheRuleIsGone(unittest.TestCase):
    def test_the_demotion_helpers_are_removed(self) -> None:
        self.assertFalse(hasattr(M, "_detect_and_demote_anti_aromatic"))
        self.assertFalse(hasattr(M, "_is_anti_aromatic"))


class RdkitKekulizeTable(unittest.TestCase):
    """The measurement the removal rests on."""

    def test_rdkit_kekulize_table(self) -> None:
        succeeds = []
        fails = []
        for n in (3, 4, 5, 6, 7, 8, 9, 10, 12):
            mol = _all_aromatic_ring(n)
            try:
                Chem.SanitizeMol(mol)
                succeeds.append(n)
            except Exception:
                fails.append(n)
        self.assertEqual(succeeds, [4, 6, 8, 10, 12])
        self.assertEqual(fails, [3, 5, 7, 9])


class AssemblyKeepsTheRing(unittest.TestCase):
    def _assemble(self, mol):
        import torch

        from molmetal.models.bond_head import AtomCloud

        zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
        orders = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), 4) for b in mol.GetBonds()
        ]
        cloud = AtomCloud(
            positions=torch.zeros((len(zs), 3), dtype=torch.float32),
            atomic_numbers=torch.tensor(zs, dtype=torch.long),
        )
        return M.YuelBondBondDecoder()._assemble_mol(cloud, orders, [])

    def test_a_four_membered_aromatic_ring_survives_assembly(self) -> None:
        res = self._assemble(_all_aromatic_ring(4))
        self.assertTrue(res.sanitized)
        self.assertEqual(res.repair, "")
        self.assertEqual(Chem.MolToSmiles(res.mol), "C1=CC=C1")

    def test_an_eight_membered_aromatic_ring_survives_assembly(self) -> None:
        res = self._assemble(_all_aromatic_ring(8))
        self.assertTrue(res.sanitized)
        self.assertEqual(res.repair, "")
        self.assertEqual(Chem.MolToSmiles(res.mol), "C1=CC=CC=CC=C1")

    def test_a_six_membered_ring_is_the_control(self) -> None:
        res = self._assemble(_all_aromatic_ring(6))
        self.assertTrue(res.sanitized)
        self.assertEqual(res.repair, "")
        self.assertEqual(Chem.MolToSmiles(res.mol), "c1ccccc1")

    def test_the_returned_bond_orders_are_what_was_predicted(self) -> None:
        mol = _all_aromatic_ring(4)
        res = self._assemble(mol)
        self.assertTrue(all(o == 4 for _i, _j, o in res.bond_orders))


if __name__ == "__main__":
    unittest.main()
