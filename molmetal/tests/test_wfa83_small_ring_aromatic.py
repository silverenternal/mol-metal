"""Wf-A83 (2026-10-18) — an aromatic bond in a 3-ring cannot be kekulized.

OPT-2's ring audit measured 3-rings at 89.8% of generated rings against 0.5%
in the real corpus, and the decoder loses exactly those molecules:
`KekulizeException: Can't kekulize mol. Unkekulized atoms: 0 1 2`.  Wf-A79
demotes aromatic bonds that lie in NO ring; this pass demotes aromatic bonds
whose both endpoints lie in a ring smaller than 5.  It is a decode-side repair,
not a ring-size fix.
"""

import unittest

from rdkit import Chem

from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    _demote_small_ring_aromatic,
)


def _aromatic_ring(smiles: str):
    """Build an UNSANITISED mol whose ring bonds are all flagged aromatic."""
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles, sanitize=False))
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.AROMATIC)
        bond.SetIsAromatic(True)
    for atom in mol.GetAtoms():
        atom.SetIsAromatic(True)
    return mol


class DemoteSmallRing(unittest.TestCase):
    def test_three_ring_becomes_sanitizable(self) -> None:
        mol = _aromatic_ring("C1CC1")
        with self.assertRaises(Exception):
            Chem.SanitizeMol(Chem.Mol(mol))
        out = _demote_small_ring_aromatic(mol)
        Chem.SanitizeMol(out)  # must not raise
        self.assertFalse(
            any(b.GetBondType() == Chem.BondType.AROMATIC for b in out.GetBonds())
        )

    def test_benzene_is_untouched(self) -> None:
        mol = _aromatic_ring("c1ccccc1")
        out = _demote_small_ring_aromatic(mol)
        self.assertTrue(
            all(b.GetBondType() == Chem.BondType.AROMATIC for b in out.GetBonds())
        )
        Chem.SanitizeMol(out)

    def test_fused_system_keeps_the_six_ring(self) -> None:
        # A 3-ring fused to a 6-ring: only the three-ring bonds may be demoted.
        mol = _aromatic_ring("C1CC1c2ccccc2")
        out = _demote_small_ring_aromatic(mol)
        six = [
            b
            for b in out.GetBonds()
            if b.IsInRing() and b.GetBeginAtomIdx() >= 3 and b.GetEndAtomIdx() >= 3
        ]
        self.assertTrue(six)
        self.assertTrue(
            all(b.GetBondType() == Chem.BondType.AROMATIC for b in six)
        )

    def test_acyclic_is_a_noop(self) -> None:
        mol = _aromatic_ring("CCCC")
        out = _demote_small_ring_aromatic(mol)
        self.assertEqual(
            [b.GetBondType() for b in out.GetBonds()],
            [b.GetBondType() for b in mol.GetBonds()],
        )

    def test_four_ring_is_also_demoted(self) -> None:
        mol = _aromatic_ring("C1CCC1")
        out = _demote_small_ring_aromatic(mol)
        Chem.SanitizeMol(out)
        self.assertFalse(
            any(b.GetBondType() == Chem.BondType.AROMATIC for b in out.GetBonds())
        )

    def test_min_size_is_configurable(self) -> None:
        mol = _aromatic_ring("c1ccccc1")
        out = _demote_small_ring_aromatic(mol, min_size=7)
        self.assertFalse(
            any(b.GetBondType() == Chem.BondType.AROMATIC for b in out.GetBonds())
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
