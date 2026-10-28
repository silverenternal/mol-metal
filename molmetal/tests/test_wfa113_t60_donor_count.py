# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T60/T61 — the donor counter read the SMILES text, not the molecule.

``count_donors`` scanned for ``[A-Z][a-z]?``.  SMILES writes aromatic nitrogen
as lowercase ``n``, so 1,10-phenanthroline counted ZERO donors; the ligand was
dropped and the reconstruction padded the empty ligand list with six waters.
21.4% of the corpus went through that path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molmetal.data.metal_smiles import count_donors, has_coordinating_ligand

_REPO = Path(__file__).resolve().parents[2]


def test_aromatic_nitrogen_is_a_donor():
    """1,10-phenanthroline: two aromatic n, no uppercase N anywhere."""
    assert count_donors("c1cnc2c(c1)ccc1cccnc12") == 2


def test_a_bipyridine_counts_two():
    assert count_donors("c1ccc(-c2ccccn2)nc1") == 2


def test_pyrrole_nitrogen_is_not_a_donor():
    """The lone pair is in the ring, so [nH] does not coordinate."""
    assert count_donors("c1cc[nH]c1") == 0


def test_aromatic_oxygen_and_sulfur_are_donors():
    assert count_donors("c1ccoc1") == 1
    assert count_donors("c1ccsc1") == 1


def test_a_cationic_nitrogen_is_not_a_donor():
    assert count_donors("[NH4+]") == 0
    assert count_donors("c1cc[nH+]cc1") == 0


def test_anionic_halide_is_not_a_donor():
    assert count_donors("[Cl-]") == 0
    assert count_donors("[Br-]") == 0


def test_uppercase_donors_still_count():
    """The T60 fix must not lose what the old scanner got right."""
    assert count_donors("NCCN") == 2
    assert count_donors("N") == 1
    assert count_donors("O") == 1
    assert count_donors("[NH3]") == 1


def test_an_arene_complex_has_no_coordinating_ligand():
    """T61 — the eta6-arene Ru rows: donors are carbons, which we do not model."""
    assert not has_coordinating_ligand("Cc1c(C)c(C)[c-](C)c1C.Cc1ccccc1", "Ru", 3)


def test_a_polypyridyl_complex_has_one():
    assert has_coordinating_ligand("c1cnc2c(c1)ccc1cccnc12", "Ru", 3)


def test_the_trainer_rejects_a_ligand_less_row():
    """T61 — a row with no donor must not come back as [OH2]x6.[Ru].

    ``reconstruct_multi_component`` still pads (that is its documented
    behaviour for an under-filled coordination sphere), so the rejection has
    to happen at the trainer boundary.
    """
    import importlib.util
    import sys

    if "_t60_trainer" not in sys.modules:
        sys.path.insert(0, str(_REPO / "molmetal" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "_t60_trainer",
            str(_REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_t60_trainer"] = mod
        spec.loader.exec_module(mod)
    tr = sys.modules["_t60_trainer"]

    cloud, status = tr.build_cloud_from_csv_row(
        "Cc1c(C)c(C)[c-](C)c1C.Cc1ccccc1", "Ru", 3, metal_placement="bonded"
    )
    assert cloud is None
    assert status == "no_coordinating_ligand"


def test_the_recorded_measurement_still_holds():
    path = _REPO / "metrics" / "by_metric" / "wfa113_t60_donor_count.json"
    if not path.exists():
        pytest.skip("probe artifact not present")
    d = json.loads(path.read_text())
    before, after = d["before"]["rows"], d["after"]["rows"]
    assert before["rows_with_zero_donors"] > 5000
    assert after["rows_with_zero_donors"] < before["rows_with_zero_donors"] / 10
    assert after["water_padding_per_row_mean"] < before["water_padding_per_row_mean"]
    cb, ca = d["before"]["cohort"], d["after"]["cohort"]
    assert ca["water_atom_fraction"] < cb["water_atom_fraction"] / 2
    assert ca["clouds_at_least_90pct_water"] == 0
    assert ca["unique_smiles"] > cb["unique_smiles"]
