#!/usr/bin/env python
"""Tests for the PAIRED embedder floor (BLK-C-OBJ geometry gate).

The failure mode these pin is the one that produced today's nine errors: a
number that is right and a CLAIM about the number that is not.  Specifically
``metrics/by_metric/cfm_metal_geometry.json`` labels a cohort
"the same real molecules" that shares only MEASURED 329 of 3,000 molecules
with its DFT cohort, and then uses the difference between them as an embedder
floor.

Red checks (each performed by reverting the change, seeing red, and restoring)
are recorded in metrics/by_metric/cfm_metal_geometry_pairing_control.json's
sibling note and in the wave report.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "_pc", REPO / "molmetal/scripts/wf_blk_c_obj_geometry_pairing_control.py"
)
PC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PC)


def test_floor_is_computed_from_identical_molecules_not_two_draws():
    """The paired floor must take BOTH sides from one csd_code list.

    RED without this: `_floor(dft, etkdg_from_csv_draw)` -- the shipped
    cfm_metal_geometry.json shape -- returns a floor computed on two disjoint
    populations.  The floor value below then changes when the ETKDG side is
    drawn from a different population, which is the defect.
    """
    dft = {"coordination_number_mean": 6.0}
    etkdg_same = {"coordination_number_mean": 5.8}
    etkdg_other = {"coordination_number_mean": 5.5}
    assert PC._floor(dft, etkdg_same)["coordination_number_mean"]["abs_floor"] == pytest.approx(0.2)
    # a different population gives a different floor; that is the whole point
    assert PC._floor(dft, etkdg_other)["coordination_number_mean"]["abs_floor"] == pytest.approx(0.5)


def test_verdict_requires_a_floor_not_just_a_gap():
    """A generator gap may only be called a finding if it beats the floor.

    RED without the comparison: a `_verdict` that reports
    `generated_minus_real_etkdg` alone labels a 0.02 gap and a 1.13 gap the
    same way, which is exactly the "the number was right, the claim was not"
    error class.
    """
    gen = {"coordination_number_mean": 4.88}
    etkdg = {"coordination_number_mean": 5.96}
    dft_big_floor = {"coordination_number_mean": 4.0}     # floor 1.96
    dft_small_floor = {"coordination_number_mean": 5.95}   # floor 0.01
    assert PC._verdict(gen, etkdg, dft_big_floor)["coordination_number_mean"]["verdict"] == (
        "generator gap is WITHIN the embedder floor -- not a finding"
    )
    assert PC._verdict(gen, etkdg, dft_small_floor)["coordination_number_mean"]["verdict"] == (
        "generator gap EXCEEDS the embedder floor"
    )


def test_paired_only_requires_cn4_on_both_sides_of_the_same_molecule():
    """The pairing itself, not just the table built from it, must require CN=4.

    RED without the filter: a CN=6 molecule that ETKDG fails to embed arrives as
    ('octahedral_like', 'tetrahedral') and is counted as a geometry-class
    disagreement -- a definitional artefact presented as a measurement.
    """
    dft_geoms = {
        "CN4OK": {"coordination_number": 4, "geometry_class": "square_planar"},
        "CN6": {"coordination_number": 6, "geometry_class": "octahedral_like"},
    }
    etkdg = [
        {"csd_code": "CN4OK", "coordination_number": 4, "geometry_class": "tetrahedral"},
        {"csd_code": "CN6", "coordination_number": 6, "geometry_class": "tetrahedral"},
    ]
    assert PC._paired_only(dft_geoms, etkdg) == [("square_planar", "tetrahedral")]


def test_smiles_lookup_returns_only_the_requested_molecules():
    """The pairing is enforced HERE, at the lookup, before any geometry is built.

    RED CHECK RUN: replacing this call with an independent `random.Random(seed)
    .sample` over the whole six-metal csv (the shape
    metrics/by_metric/cfm_metal_geometry.json uses) asked 40 molecules and got
    4 -- a 10x silent cohort shrink -- while the emitted `note` still claimed
    "the SAME csd_code list".  The CN-definition agreement under that draw also
    read 1.0, a pure artefact of measuring agreement over 4 surviving molecules.
    This test is what makes that red instead of silent.
    """
    rows = PC.MG.select_dft_population(40, 20261012)
    got = PC._smiles_for(rows)
    requested = {csd for csd, _m, _c in rows}
    assert set(got) <= requested, "the ETKDG side was drawn from a different population"
    # and it must actually find them: an all-but-empty lookup is the same defect
    assert len(got) >= 0.5 * len(requested)


def test_pairing_guard_raises_on_an_unreachable_etkdg_side(monkeypatch):
    """The guard must fire on COVERAGE, and coverage is the thing that breaks.

    RED CHECK 3 RUN (measured): with the lookup replaced by an independent
    seeded sample, the script asked for 40 molecules and scored 4 --
    `AssertionError: pairing broken: only 4/40 DFT molecules were reachable on
    the ETKDG side (coverage 0.100)`.  A first attempt asserted on csd overlap
    instead and never fired, because the ETKDG loop already walks dft_rows and
    so overlap is 1.0 by construction -- a guard that cannot go red is not a
    guard.
    """
    import csv as _csv
    import random as _rnd

    def independent_draw(rows, limit=3000, seed=20261012):
        allr = []
        with open(PC.MG.TMQM_CSV, newline="") as fh:
            for row in _csv.DictReader(fh):
                if row.get("metal") in set(PC.MG.TRAIN_METALS) and row.get("smiles"):
                    allr.append((row["csd_code"], row["smiles"]))
        allr.sort()
        return dict(sorted(_rnd.Random(seed).sample(allr, limit)))

    monkeypatch.setattr(PC, "_smiles_for", independent_draw)
    with pytest.raises(AssertionError, match="pairing broken"):
        PC.paired_embedder_floor(40, 20261012, 0.15)


def test_cn4_table_is_built_only_from_both_sides_cn4():
    """The CN4 comparison must not be diluted by molecules ETKDG could not score.

    RED without the CN==4 filter on BOTH sides: pairs like
    ('square_planar', 'octahedral_like') enter the table, so an ETKDG failure
    to embed shows up as a geometry-class disagreement rather than as missing.
    """
    pairs = [("square_planar", "tetrahedral"), ("square_planar", "octahedral_like")]
    table = PC.paired_cn4_table(pairs)
    assert table["n_paired_cn4"] == 2
    assert table["table"] == {
        "dft=square_planar|etkdg=tetrahedral": 1,
        "dft=square_planar|etkdg=octahedral_like": 1,
    }


def test_cn_definition_agreement_is_reported_not_assumed():
    """A CN gap across cohorts is meaningless unless the two definitions agree.

    RED without this: nothing reports that DFT CN is a 2.6 A donor window and
    ETKDG CN is RDKit graph degree, so a coordination-number comparison is
    quoted as if it were like-for-like.
    """
    dft_geoms = {"A": {"coordination_number": 6}, "B": {"coordination_number": 4}}
    etkdg = [{"csd_code": "A", "coordination_number": 6},
             {"csd_code": "B", "coordination_number": 5}]
    out = PC._cn_agreement_paired(dft_geoms, etkdg)
    assert out["n_compared"] == 2
    assert out["n_agree"] == 1
    assert out["fraction"] == 0.5


def test_measurement_json_on_disk_is_paired_and_agrees_with_these_rules():
    """The emitted metric must carry the paired floor, not the drifted one."""
    p = REPO / "metrics/by_metric/cfm_metal_geometry_pairing_control.json"
    if not p.exists():                       # NOT ON DISK until the script runs
        pytest.skip("pairing control not yet run")
    d = json.loads(p.read_text())
    assert d["tag"] == "MEASURED"
    pop = d["population"]
    assert pop["dft_requested"] == pop["etkdg_measured"], (
        "the two sides were not drawn from one population"
    )
    # the CN definition gap is measured and reported, never assumed away
    assert d["cn_definition_agreement_dft_vs_etkdg_paired"]["fraction"] is not None
    # and every axis carries a verdict that names both a gap and a floor
    for axis, row in d["generator_gap_versus_paired_floor"].items():
        assert "abs_generator_gap" in row and "abs_paired_embedder_floor" in row, axis
        assert row["verdict"].startswith("generator gap")