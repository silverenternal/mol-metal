# SPDX-License-Identifier: Apache-2.0
"""m7's canonical metric file must measure diversity, not the trainer's plumbing.

WHY THIS FILE EXISTS
--------------------
``metrics/by_metric/m7_diversity.json`` used to declare::

    "unit": "CLI flag count wired-in (count 0-6)"

for a metric named *m7 DIVERSITY*. A count of wired-in argparse flags is a
statement about the trainer's plumbing; it is not a property of any generated
molecule. The BLK-C6 title nonetheless quoted 0.9159 / 0.9215 / 0.8674 as
m7 values, produced by a throwaway script that was never kept, on a
2048-bit / self-pairs-excluded protocol that is not the MOSES reference.

A metric file whose name and measurement disagree is worse than no file,
because downstream readers treat the file as the definition.

WHAT THIS PINS
--------------
1. The unit is a diversity score, not a flag count.
2. There is a MEASURED value, sourced to a CSV that exists on disk.
3. Any run named in the file has its source CSV present — this is what
   caught the five unreproducible IntDiv1 rows in the side file.
4. The finite-sample caveat is stated, because IntDiv1 at n=37 cannot be
   compared to MOSES at n=1.9M, and a reader who skips that note will
   misread 0.869 as "above reference".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
M7_JSON = REPO_ROOT / "metrics" / "by_metric" / "m7_diversity.json"


@pytest.fixture(scope="module")
def m7():
    return json.loads(M7_JSON.read_text(encoding="utf-8"))


def test_the_unit_is_a_diversity_score_not_a_flag_count(m7):
    """A 'count of flags' is not what DIVERSITY means."""
    unit = m7["unit"].lower()
    assert "flag count" not in unit, (
        "m7 is the diversity metric; its unit must be a diversity score. A "
        "wired-flag count describes the trainer, not the generated cohort."
    )
    assert "intdiv" in unit


def test_flag_count_is_not_the_headline_value(m7):
    """The old file's headline slot held the flag count. Catch a regression."""
    value = m7["aggregate"]["best_measured"]
    assert isinstance(value, float), "the headline must be a measured float"
    assert 0.0 < value < 1.0, f"{value} is not an IntDiv1-scale score"
    assert value != float(m7["diversity_term_wiring"]["n_flags_wired_in"])


def test_every_named_run_has_its_csv_on_disk(m7):
    """The five IntDiv1 rows in the side file named runs with no CSV.

    This is the test that would have caught them: a row is only evidence if
    the artifact it was computed from is still there.
    """
    missing = [
        row["run"]
        for row in m7["measured_rows"]
        if not (REPO_ROOT / row["source_csv"]).is_file()
    ]
    assert not missing, (
        f"these rows cite a source CSV that is not on disk, so the numbers "
        f"cannot be reproduced or checked: {missing}"
    )


def test_the_measured_value_recomputes_from_its_own_csv(m7):
    """Recompute the headline from the cited CSV.

    Guards against the number drifting away from the artifact it claims to
    come from — the exact failure that let 0.9159/0.9215/0.8674 stand for a
    month.
    """
    import sys

    script_dir = REPO_ROOT / "molmetal" / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import cfm_property_panel as panel

    row = next(
        r
        for r in m7["measured_rows"]
        if r["run"] == m7["aggregate"]["best_measured_run"]
    )
    smiles = panel.read_smiles_csv(REPO_ROOT / row["source_csv"], last_epoch=True)
    out = panel.internal_diversity(smiles)
    assert out["intdiv1"] == pytest.approx(row["intdiv1"], abs=1e-9)
    assert out["intdiv1"] == pytest.approx(m7["aggregate"]["best_measured"], abs=1e-9)


def test_the_finite_sample_caveat_is_stated(m7):
    """These cohorts are n=37-47; MOSES' 0.857 is n=1.9M.

    Without this note a reader sees 0.869 > 0.857 and concludes the cohort is
    above the reference — when at matched n it is actually BELOW the training
    distribution. The caveat is what makes the number readable.
    """
    fs = m7["FINITE_SAMPLE_WARNING"]
    assert "n-dependent" in fs["finding"]
    assert fs["n_matched_comparison"]["delta"] < 0, (
        "the n-matched comparison is the load-bearing claim; if the sign "
        "flips, this test should fail and the caveat needs rewriting"
    )


def test_no_target_is_asserted_at_the_current_cohort_size(m7):
    """Claiming a target we cannot measure decides the result in advance."""
    assert m7["target"]["value"] is None, (
        "no m7 target may be asserted until a >=1000-molecule cohort exists; "
        "see restore_condition"
    )
    assert "1000" in m7["restore_condition"]


def test_the_trainer_wiring_survives_the_redefinition(m7):
    """The flags stay wired and default-off; the metric just stops reporting them."""
    w = m7["diversity_term_wiring"]
    assert w["n_flags_wired_in"] == w["n_flags_total"] == 6
    assert w["n_flags_default_off"] == 6
