# SPDX-License-Identifier: Apache-2.0
"""Wf-A85 (2026-10-19) — ring SIZE is reported by both lines, one definition.

OPT-2 (metrics/by_metric/cfm_opt2_ring_content.json) decided the generation
question on the ring-size profile: 5/6-membered rings are 1.85% generated vs
96.92% real, 3-membered 89.8% vs 0.5%.  That number came from a one-off script
over one CSV, so neither a training run nor the property panel could report it.

molmetal/eval/ring_profile.py is now the single implementation; the trainer
(decode_chemistry -> summarise_decode_chemistry) and the panel (build_panel)
both call it.  These tests pin the definition: SSSR, shares over RINGS.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from molmetal.eval.ring_profile import (
    ring_profile,
    ring_sizes_from_mol,
    summarise_ring_sizes,
)

REPO = Path(__file__).resolve().parent.parent.parent
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _trainer():
    if "_wfa85_trainer" in sys.modules:
        return sys.modules["_wfa85_trainer"]
    spec = importlib.util.spec_from_file_location("_wfa85_trainer", str(TRAINER))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_wfa85_trainer"] = m
    spec.loader.exec_module(m)
    return m


class RingProfile(unittest.TestCase):
    def test_three_ring(self) -> None:
        out = ring_profile(["C1CC1"])
        self.assertEqual(out["ring_size_distribution"], {"3": 1})
        self.assertEqual(out["ring_3_share"], 1.0)
        self.assertEqual(out["ring_5_6_share"], 0.0)
        self.assertEqual(out["rings_per_mol_mean"], 1.0)

    def test_benzene_is_a_six_ring(self) -> None:
        out = ring_profile(["c1ccccc1"])
        self.assertEqual(out["ring_5_6_share"], 1.0)
        self.assertEqual(out["ring_3_share"], 0.0)

    def test_acyclic_has_no_rings_and_zero_shares(self) -> None:
        out = ring_profile(["CCCC"])
        self.assertEqual(out["ring_size_distribution"], {})
        self.assertEqual(out["rings_per_mol_mean"], 0.0)
        self.assertEqual(out["ring_5_6_share"], 0.0)
        self.assertEqual(out["n_molecules"], 1)

    def test_unparseable_is_counted_not_silently_ring_free(self) -> None:
        out = ring_profile(["not-a-smiles", "C1CC1"])
        self.assertEqual(out["n_unparsed"], 1)
        self.assertEqual(out["n_molecules"], 1)
        self.assertEqual(out["ring_3_share"], 1.0)

    def test_share_denominator_is_rings_not_molecules(self) -> None:
        # one molecule, two rings: 3+6 -> 5/6 share 0.5, not 1.0
        out = ring_profile(["C1CC1c2ccccc2"])
        self.assertEqual(out["rings_per_mol_mean"], 2.0)
        self.assertEqual(out["ring_5_6_share"], 0.5)
        self.assertEqual(out["ring_3_share"], 0.5)

    def test_fused_system_sizes(self) -> None:
        # naphthalene: SSSR gives two 6-rings
        out = ring_profile(["c1ccc2ccccc2c1"])
        self.assertEqual(out["ring_size_distribution"], {"6": 2})

    def test_summarise_matches_mol_helper(self) -> None:
        from rdkit import Chem

        mol = Chem.MolFromSmiles("C1CC1")
        self.assertEqual(ring_sizes_from_mol(mol), [3])
        # 2 of 3 rings are 5/6-membered: the denominator is RINGS.
        self.assertAlmostEqual(
            summarise_ring_sizes([[3], [6, 6]])["ring_5_6_share"], 2.0 / 3.0
        )


class TrainerPlumb(unittest.TestCase):
    """The per-epoch chemistry block must carry the same numbers."""

    def test_summary_has_the_ring_block(self) -> None:
        t = _trainer()
        rows = [t.decode_chemistry("C1CC1"), t.decode_chemistry("c1ccccc1")]
        out = t.summarise_decode_chemistry(rows)
        self.assertEqual(out["ring_size_distribution"], {"3": 1, "6": 1})
        self.assertEqual(out["rings_per_mol_mean"], 1.0)
        self.assertEqual(out["ring_5_6_share"], 0.5)
        self.assertEqual(out["ring_3_share"], 0.5)

    def test_empty_rows_still_have_the_keys(self) -> None:
        t = _trainer()
        out = t.summarise_decode_chemistry([])
        for k in (
            "ring_size_distribution",
            "rings_per_mol_mean",
            "ring_5_6_share",
            "ring_3_share",
        ):
            self.assertIn(k, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
