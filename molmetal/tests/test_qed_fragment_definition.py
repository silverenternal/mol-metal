# SPDX-License-Identifier: Apache-2.0
"""m5 QED must be quoted on single-fragment molecules, with its denominator.

WHY
---
`cfm_property_panel.py` computed QED on the whole parsed SMILES. RDKit accepts
multi-fragment SMILES, so on a fragmented cohort that number scores a BAG OF
FRAGMENTS rather than a molecule. Measured on 776 decoded CFM molecules
(85.1% multi-fragment):

    QED on the whole SMILES        0.1952   <- what used to be reported
    QED on the largest fragment    0.2718
    QED on single-fragment only    0.1246   <- the number to quote

The old figure was flattering by ~57%. Taking the largest fragment is NOT the
fix either: single-fragment molecules here score LOWER still (0.1246), because
they are small connected fragments and QED penalises small molecules. Both
readings are wrong in different directions, so the panel now carries the raw
per-molecule facts and the aggregate reports the single-fragment-only view WITH
its denominator.

Note the corrected number is worse than the contaminated one. That is the
point: a metric fix is allowed to make the project look worse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = REPO_ROOT / "molmetal" / "scripts"


@pytest.fixture(scope="module")
def panel():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import cfm_property_panel as mod

    return mod


#: one connected drug-ish molecule, and one that is explicitly two pieces
SINGLE = "CC(=O)Nc1ccc(O)cc1"
MULTI = "CC(=O)Nc1ccc(O)cc1.CCCCCC"


def test_panel_reports_single_fragment_qed_with_its_denominator(panel):
    rep = panel.build_panel([SINGLE, MULTI], include_validity=True)
    agg = rep["aggregate"]
    for key in ("qed_single_fragment", "single_fragment_rate", "n_single_fragment",
                "multifragment_rate", "qed_largest_fragment_all",
                "qed_contamination_note"):
        assert key in agg, f"aggregate lost {key}"
    assert agg["n_single_fragment"] == 1
    assert agg["n_input"] == 2
    assert agg["single_fragment_rate"] == pytest.approx(0.5)


def test_per_molecule_records_carry_the_raw_facts(panel):
    rep = panel.build_panel([SINGLE, MULTI])
    single, multi = rep["per_molecule"]
    assert single["n_frags"] == 1
    assert multi["n_frags"] == 2
    assert single["largest_frag_frac"] == pytest.approx(1.0)
    assert 0.0 < multi["largest_frag_frac"] < 1.0
    assert single["qed_largest_fragment"] == pytest.approx(
        single["qed"], rel=1e-6
    ), "a single fragment must score the same either way"


def test_the_old_whole_smiles_field_is_still_labelled_contaminated(panel):
    """Keep it for continuity, but it must never be quotable alone."""
    agg = panel.build_panel([SINGLE, MULTI]).get("aggregate") or {}
    rep = panel.build_panel([SINGLE, MULTI])
    note = rep["aggregate"]["qed_contamination_note"]
    assert "multi-fragment" in note and "qed_single_fragment" in note
    assert "do not quote" in note.lower()


def test_single_fragment_view_can_differ_from_the_whole_view(panel):
    """The two must be genuinely distinct numbers, not aliases."""
    rep = panel.build_panel([SINGLE, MULTI])
    agg = rep["aggregate"]
    assert agg["qed"]["mean"] != agg["qed_single_fragment"]["mean"], (
        "if these are equal the fix is not doing anything on a cohort that "
        "contains multi-fragment molecules"
    )


def test_a_fully_fragmented_cohort_reports_an_empty_single_fragment_view(panel):
    """n=0 must not silently become 0.0 mean and read like a real score."""
    rep = panel.build_panel([MULTI, MULTI])
    agg = rep["aggregate"]
    assert agg["n_single_fragment"] == 0
    assert agg["single_fragment_rate"] == 0.0
    # _percentiles over an empty list must not invent a mean.
    assert not agg["qed_single_fragment"].get("mean")
