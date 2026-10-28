# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 / T42 — bond_ce is not a class-weight artefact, and T42 is refuted.

THE HYPOTHESIS
--------------
``bond_ce`` used to be read as a transient: 15.92 at epoch 0, 1.59 by epoch 6.
The class weights are renormalised on ``sum_c p_c w_c == 1``, so the weighted CE
has the same EXPECTED value as the unweighted one only when the prediction
matches the prior.  The natural guess was that the early value is the
denominator, and T42's done_when asked for the unweighted CE to stay flat
(|ep0 - ep6| < 0.2) while the weighted one falls.

THE RESULT
----------
Both are large at epoch 0.  MEASURED on ``configs/wfa108j_budget36_2026_10_25``
(CPU, 7 epochs):

    ep0  bond_ce 13.5135  unweighted 16.6028  ratio 0.814
    ep6  bond_ce  1.4705  unweighted  0.5394  ratio 2.726

The unweighted CE is LARGER at ep0 and its ep0->ep6 difference is 16.06, not
< 0.2.  Removing the weights makes the number worse, so the denominator is
exonerated.  T42's kill clause fired and pointed at the pair head's output-layer
initialisation, which is T52 (see ``test_wfa113_t52_bond_head_init.py``).

WHAT THIS FILE CHECKS
---------------------
1. Both CEs are recorded in the same epoch row, on the real runs on disk.
2. The weighted/unweighted ratio crosses 1 exactly where the head learns the
   base rate, which is what the ratio is a diagnostic OF.
3. The lr-0 control does not fall, so the decline is the optimiser.
4. The artifact's quoted numbers match the run files.

WHAT THIS FILE DOES NOT CLAIM
-----------------------------
It does not claim bond_ce is now healthy.  It claims the early value is not the
weight denominator; the actual cause is the initialisation, measured separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "metrics" / "by_metric" / "wfa113_t42_bond_ce_decomposition.json"


@pytest.fixture(scope="module")
def art():
    return json.loads(ART.read_text())


def _run(art, name):
    for r in art["measurements"]["runs"]:
        if r["run"].startswith(name):
            return r
    raise KeyError(name)


def test_artifact_exists_and_is_measured(art):
    assert art["tag"] == "MEASURED"
    assert art["artifact_for"].endswith("T42")


def test_every_quoted_row_matches_the_run_file(art):
    for run in art["measurements"]["runs"]:
        path = REPO / run["artifact"]
        assert path.exists(), path
        rows = json.loads(path.read_text())["per_epoch_metrics"]
        for quoted, real in zip(run["rows"], rows):
            assert quoted["epoch"] == real["epoch"]
            assert quoted["bond_ce"] == pytest.approx(real["bond_ce_loss"], abs=5e-5)
            assert quoted["bond_ce_unweighted"] == pytest.approx(
                real["bond_ce_unweighted"], abs=5e-5
            )


def test_unweighted_ce_is_not_flat(art):
    """The done_when gate: |ep0 - ep6| < 0.2.  It fails, and that is the point."""
    run = _run(art, "wfa108j budget36, lr 1e-3")
    rows = run["rows"]
    diff = abs(rows[0]["bond_ce_unweighted"] - rows[-1]["bond_ce_unweighted"])
    assert diff > 15.0
    assert run["ep0_minus_ep6_unweighted"] == pytest.approx(diff, abs=5e-4)


def test_unweighted_is_larger_at_epoch_zero(art):
    """The refutation: dropping the weights makes ep0 worse, not better."""
    for run in art["measurements"]["runs"]:
        ep0 = run["rows"][0]
        assert ep0["bond_ce_unweighted"] > ep0["bond_ce"]
        assert ep0["ratio"] < 1.0


def test_ratio_crosses_one_only_after_the_head_learns_the_base_rate(art):
    run = _run(art, "wfa108j budget36, lr 1e-3")
    rows = run["rows"]
    before = [r for r in rows if r["epoch"] < run["crossover_epoch"]]
    after = [r for r in rows if r["epoch"] > run["crossover_epoch"]]
    assert before and after
    assert all(r["ratio"] < 1.0 for r in before)
    assert all(r["ratio"] > 1.0 for r in after)


def test_lr_zero_control_does_not_fall(art):
    run = _run(art, "wfa108j budget36, lr 0")
    rows = run["rows"]
    assert rows[1]["bond_ce"] >= rows[0]["bond_ce"] - 0.5


def test_verdict_is_a_refutation(art):
    assert art["verdict"]["result"] == "REFUTED"
    assert art["verdict"]["done_when_status"].startswith("VIOLATED")


def test_class_statistics_are_internally_consistent(art):
    st = art["measurements"]["class_statistics"]
    assert sum(st["counts"]) == 11266
    # sum_c p_c w_c == 1 is the invariant the whole weight scheme rests on.
    dot = sum(p * w for p, w in zip(st["prior"], st["weights_at_power_0.5"]))
    assert dot == pytest.approx(1.0, abs=1e-2)


def test_the_sibling_t52_artifact_exists(art):
    """The kill clause has to point at something real."""
    assert "wfa113_t52_bond_head_init.json" in art["verdict"]["kill_clause_outcome"]
    assert (REPO / "metrics" / "by_metric" / "wfa113_t52_bond_head_init.json").exists()
