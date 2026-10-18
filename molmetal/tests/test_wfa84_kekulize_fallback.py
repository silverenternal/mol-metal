# SPDX-License-Identifier: Apache-2.0
"""Wf-A84 (2026-10-19) — keep the molecule, drop the aromaticity.

WHY THIS EXISTS
---------------
The Wf-A83 post-fix smoke (/tmp/wfa83_smoke, MEASURED) still reported
`gen_validity=0.500 (8/16)` and every failure was
`sanitize_failed:KekulizeException`.  Wf-A79 (aromatic outside any ring) and
Wf-A83 (aromatic inside a <5 ring) each repair one class of aromatic bond
locally; what remains is a ring the kekulizer refuses AS A WHOLE — no local
edit can fix it, because RDKit needs a Kekule assignment for the entire
molecule or none.

The decoder's last resort is therefore structural, not local: clear every
aromatic flag and rewrite every aromatic bond as SINGLE.  The molecule
survives; its rings become saturated.  That is a downgrade, so the result
carries `repair="kekulize_demoted_all"` and the trainer counts it in
`ok_repaired:` rather than `ok`.

RED CHECK: delete the `_demote_all_aromatic` call from the SanitizeMol
`except` branch and `test_five_ring_all_aromatic_becomes_valid` fails with
KekulizeException.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any, List, Optional

from rdkit import Chem

from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    _demote_all_aromatic,
)

REPO = Path(__file__).resolve().parent.parent.parent
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _all_aromatic_ring(n_atoms: int):
    """An UNSANITISED all-carbon ring whose every bond is flagged aromatic."""
    rw = Chem.RWMol()
    for _ in range(n_atoms):
        rw.AddAtom(Chem.Atom(6))
    for i in range(n_atoms):
        rw.AddBond(i, (i + 1) % n_atoms, Chem.BondType.AROMATIC)
        rw.GetBondBetweenAtoms(i, (i + 1) % n_atoms).SetIsAromatic(True)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(True)
    return rw.GetMol()


class DemoteAllAromatic(unittest.TestCase):
    def test_five_ring_all_aromatic_becomes_valid(self) -> None:
        # 5 aromatic carbons cannot be kekulized: the repair is what keeps
        # the molecule instead of losing it to KekulizeException.
        mol = _all_aromatic_ring(5)
        with self.assertRaises(Exception):
            Chem.SanitizeMol(Chem.Mol(mol))
        out = _demote_all_aromatic(mol)
        Chem.SanitizeMol(out)  # must not raise
        self.assertFalse(
            any(b.GetBondType() == Chem.BondType.AROMATIC for b in out.GetBonds())
        )
        self.assertEqual(Chem.MolToSmiles(out), "C1CCCC1")

    def test_no_aromatic_bond_is_a_noop(self) -> None:
        mol = Chem.RWMol(Chem.MolFromSmiles("CCO", sanitize=False)).GetMol()
        before = [b.GetBondType() for b in mol.GetBonds()]
        out = _demote_all_aromatic(mol)
        self.assertEqual([b.GetBondType() for b in out.GetBonds()], before)

    def test_every_aromatic_flag_is_cleared(self) -> None:
        mol = _all_aromatic_ring(6)
        out = _demote_all_aromatic(mol)
        self.assertFalse(any(a.GetIsAromatic() for a in out.GetAtoms()))
        self.assertFalse(any(b.GetIsAromatic() for b in out.GetBonds()))

    def test_repaired_molecule_has_no_double_bonds(self) -> None:
        # The downgrade is explicit: what was aromatic is now saturated.
        out = _demote_all_aromatic(_all_aromatic_ring(6))
        Chem.SanitizeMol(out)
        self.assertFalse(
            any(b.GetBondType() == Chem.BondType.DOUBLE for b in out.GetBonds())
        )


class DecodedMolRepairField(unittest.TestCase):
    def test_default_is_empty(self) -> None:
        from molmetal.models.bond_head import DecodedMol

        self.assertEqual(DecodedMol().repair, "")


def _trainer():
    if "_wfa84_trainer" in sys.modules:
        return sys.modules["_wfa84_trainer"]
    spec = importlib.util.spec_from_file_location("_wfa84_trainer", str(TRAINER))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_wfa84_trainer"] = m
    spec.loader.exec_module(m)
    return m


class _Decoded:
    def __init__(self, smiles: str, repair: str = "", sanitized: bool = True):
        self.smiles = smiles
        self.sanitized = sanitized
        self.error = ""
        self.repair = repair


class _Result:
    def __init__(self, decoded):
        self.decoded = decoded


class _Decoder:
    def __init__(self, decoded):
        self.decoded = decoded

    def decode(self, cloud):
        return _Result(self.decoded)


class _Cloud:
    def __init__(self):
        import torch

        self.atomic_numbers = torch.tensor([6, 6, 6, 6], dtype=torch.long)


class BucketAccounting(unittest.TestCase):
    """A repaired success must not be reported as a clean one."""

    def _counts(self, decoded) -> dict:
        t = _trainer()
        counts: dict = {}
        t.decode_clouds_and_classify(
            _Decoder(decoded), [_Cloud()], [], None, [], [], None, counts
        )
        return counts

    def test_clean_decode_lands_in_ok(self) -> None:
        self.assertEqual(self._counts(_Decoded("CCO")), {"ok": 1})

    def test_repaired_decode_lands_in_its_own_bucket(self) -> None:
        counts = self._counts(_Decoded("C1CCCC1", repair="kekulize_demoted_all"))
        self.assertEqual(counts, {"ok_repaired:kekulize_demoted_all": 1})
        self.assertNotIn("ok", counts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
