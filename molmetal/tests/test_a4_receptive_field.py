# SPDX-License-Identifier: Apache-2.0
"""Ring size is separable at 1 hop -- which REFUTES claim A4 in cfm_todo.json.

THE DEFECT THIS PINS
--------------------
On 2026-10-13 a structural claim was added to cfm_todo.json as a load-bearing
premise of the ARCH programme (claim A4): ring size is UNREPRESENTABLE in
``YuelBondDecoderHead`` because the head is purely per-edge and ring closure
is a property of a CYCLE rather than of any single edge. It was used as the
structural argument against spending the single authorised bond-head retrain.

It was inferred by reading the class body -- ``ring`` / ``cycle`` / ``RingInfo``
do not appear in ``YuelBondDecoderHead`` -- and that inference is wrong.
"No explicit cycle operator" does not imply "cycles cannot be represented": a
message-passing network over a radius graph can encode membership implicitly.

MEASURED, 657 rings across 120 real tmQM clouds (the repo's own
``load_tmqm_clouds``, cutoff 3.5 A, the head's default; the trained
checkpoint's args confirm ``n_layers=3``):

    ring   n_rings   mean 1-hop within-ring neighbours   sd
       3      120                    2.00             0.00
       4       14                    3.00             0.00
       5      165                    3.96             0.22
       6      335                    4.97             0.17

    3-ring vs 4-ring  exact-value overlap   0/120 =  0.0%

A 1-hop statistic separates 3-rings from 4-rings PERFECTLY, and the head
runs 3 message-passing layers. So its receptive field comfortably covers the
thing A4 said it could not see.

WHAT THE TEST PINS, AND WHY EACH DIRECTION MATTERS
--------------------------------------------------
A refutation this consequential needs both directions. A test that only
asserts the numbers reproduces would still pass if the measurement silently
stopped measuring ring membership -- which is exactly the failure mode this
project has hit repeatedly (a loader returning a different shape, a truth
block absent, a silent skip). So:

1. ``test_three_and_four_rings_do_not_overlap`` -- the load-bearing result.
2. ``test_measurement_is_not_vacuous`` -- guards the OTHER direction: if the
   ring extraction silently returned nothing, or the statistic collapsed to a
   constant, the separation test would trivially "pass". These fail closed.
3. ``test_truth_bonds_are_an_edge_index_not_an_edge_list`` -- pins the shape
   assumption. tmQM truth bonds arrive as an ``(2, E)`` edge-index; treating
   that as an ``(E, 2)`` list yields edges ``(i, j) = (0, 8), (0, 9), ...``
   which is a *different graph* that still produces rings, so the bug is
   silent. It was hit once while writing this and produced 0 cycles.
4. ``test_head_actually_has_three_layers`` -- the receptive-field claim is
   only load-bearing if the depth is what the checkpoint says.

THE LOAD-BEARING MEASUREMENT IS THE DISTANCE PATTERN, NOT THE 1-HOP COUNT
------------------------------------------------------------------------
An earlier version of this file pinned a "within-ring 1-hop neighbour count"
statistic and called that the refutation. It is a weak instrument: 94.2
percent of rings are cliques in the radius graph, so for those the count is
n-1 by construction -- i.e. it just re-derives the ring size rather than
showing the head can see it. The decisive measurement therefore uses what
the head ACTUALLY receives:

    A plain LogisticRegression on the sorted pairwise distances of each ring
    -- no message passing, no learned features -- classifies ring size at
    1.000 +/- 0.000 (5-fold CV) and 1.000 on a held-out 30 percent, with
    per-class recall 1.000 for sizes 3, 4, 5 and 6, over 634 real rings.

A linear model on raw distances solving the task makes it a LOWER bound on
what the head's 3 message-passing layers could represent.

A SECOND BUG THIS FILE NOW GUARDS
---------------------------------
The clique check originally read ``(sub > 0).all()`` over a matrix whose
diagonal is zeroed, so it was False for every ring and the script reported
"0 percent of rings are cliques" as a MEASUREMENT. It was structurally
pinned at zero. The mutation pass caught it: replacing the cutoff graph with
the full distance matrix should have driven the fraction to 1.0 and left it
at 0.0 instead. ``test_clique_fraction_is_measured_not_structurally_pinned``
is the regression guard.

RED CHECKS (each run and observed to fail)
------------------------------------------
* Build the radius graph over the FULL distance matrix instead of the
  cutoff graph, so every pair is a neighbour -> every ring becomes a clique
  and all sizes collapse to n-1 -> the clique guard fails.
* Revert the off-diagonal fix in the clique check -> the fraction returns to
  exactly 0.0 -> ``test_clique_fraction_is_measured_not_structurally_pinned``
  FAILS. (This is the bug that shipped once.)
* Drop the ``.T`` transpose in the truth-bond unpacking -> cycles silently
  vanish -> ``test_measurement_is_not_vacuous`` FAILS.
* Shuffle the ring-size labels in the capacity test -> accuracy collapses to
  chance -> the capacity assertion FAILS.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT = (
    PROJECT_ROOT / "molmetal" / "scripts" / "wf_a4_receptive_field_2026_10_13.py"
)
METRIC = PROJECT_ROOT / "metrics" / "by_metric" / "cfm_a4_receptive_field.json"

_RESULT = None


def _measured():
    """Run the measurement once per session (it reads real tmQM geometry)."""
    global _RESULT
    if _RESULT is None:
        mod = importlib.import_module(
            "molmetal.scripts.wf_a4_receptive_field_2026_10_13"
        )
        _RESULT = mod.measure(n_clouds=40)
    return _RESULT


# ---------------------------------------------------------------------------
# 1. The load-bearing result
# ---------------------------------------------------------------------------


def test_three_and_four_rings_do_not_overlap():
    """A 1-hop statistic separates 3-rings from 4-rings perfectly.

    This is the refutation of A4. If it ever becomes true that they DO
    overlap, ring size is NOT separable at 1 hop and A4's weaker form is
    reinstated -- so this test failing is not a nuisance, it is information.
    """
    res = _measured()
    per = res["per_ring_size"]
    if "3" not in per or "4" not in per:
        pytest.skip("no 4-rings in this small cohort")
    assert per["3"]["mean_within_ring_1hop_neighbours"] == pytest.approx(2.0, abs=0.05)
    assert per["4"]["mean_within_ring_1hop_neighbours"] == pytest.approx(3.0, abs=0.05)
    overlap = res["pairwise_overlap_exact_values"]["3_vs_4"]
    assert overlap == 0.0, (
        f"3-rings and 4-rings now overlap at {overlap:.1%}. Ring size is no "
        "longer separable at 1 hop, so cfm_todo.json::A4's weaker form -- "
        "'the head must LEARN ring size from local geometry rather than read "
        "it off' -- becomes the only defensible claim again."
    )


def test_ring_size_tracks_the_neighbour_count_monotonically():
    """4.97 for 6-rings vs 2.00 for 3-rings is not a coincidence of one pair."""
    per = _measured()["per_ring_size"]
    seq = [
        (int(k), v["mean_within_ring_1hop_neighbours"])
        for k, v in sorted(per.items(), key=lambda kv: int(kv[0]))
        if int(k) <= 6
    ]
    assert len(seq) >= 3, f"need >=3 ring sizes to test monotonicity, got {seq}"
    for (k1, v1), (k2, v2) in zip(seq, seq[1:]):
        assert v2 >= v1 - 0.1, (
            f"mean 1-hop within-ring neighbours fell from {v1:.2f} at ring "
            f"size {k1} to {v2:.2f} at size {k2}; the statistic is supposed "
            "to grow with ring size"
        )


# ---------------------------------------------------------------------------
# 2. The measurement must not be vacuous
# ---------------------------------------------------------------------------


def test_measurement_is_not_vacuous():
    """Guards the direction a separation test can silently fail in.

    If ring extraction returned nothing, or the neighbour counts collapsed to
    a single constant, test 1 would still 'pass' -- an empty set has no
    overlap. So assert the measurement actually produced the structure it
    claims to have measured.
    """
    res = _measured()
    assert res["setup"]["n_rings_measured"] > 0, (
        "zero rings measured -- the separation test is vacuous, not passing"
    )
    values = [
        v["mean_within_ring_1hop_neighbours"] for v in res["per_ring_size"].values()
    ]
    assert len(values) >= 3, f"too few ring sizes to mean anything: {values}"
    assert len(set(round(v, 3) for v in values)) >= 3, (
        f"the statistic collapsed to {len(set(values))} distinct values "
        f"{values}; ring size is not being distinguished at all"
    )
    assert 2 in [round(v) for v in values], (
        f"no ring size shows the 2-neighbour signature of a 3-ring: {values}"
    )


def test_clique_fraction_is_measured_not_structurally_pinned():
    """Guards a bug this file already shipped once.

    The first version of the measurement counted cliques with
    ``(sub > 0).all()`` over a matrix whose diagonal is zeroed by
    construction. That is False for EVERY ring, so the count was pinned at
    zero and the script reported "0 percent of rings are cliques" as though
    it had been measured. The true value is 94.2 percent.

    A guard that cannot fire is not a guard. This asserts the count is a
    plausible interior fraction, so a future off-diagonal bug cannot hide
    behind a structurally-zero result again. The red check is to revert the
    off-diagonal fix -- the fraction drops to exactly 0.0 and this fails.
    """
    res = _measured()
    frac = res["clique_fraction"]
    assert 0.0 < frac < 1.0, (
        f"clique_fraction is {frac:.4f}. Either the off-diagonal mask is "
        "broken (structurally 0, the bug this test exists for) or the radius "
        "graph changed and the number needs re-deriving."
    )
    assert res["n_rings_that_are_cliques"] > 0, (
        "zero cliques reported -- check that the clique test excludes the "
        "zeroed diagonal before believing any figure derived from it"
    )


def test_ring_size_is_recoverable_from_the_distance_pattern_alone():
    """The decisive refutation: a LINEAR model on raw distances solves it.

    No message passing, no learned features -- just the sorted pairwise
    distances of a ring, which is a strict subset of what the head's
    message-passing layers receive. Accuracy at or near 1.0 means ring size
    is trivially recoverable, so A4's "unrepresentable" cannot hold.

    This is a capacity result: it says the head COULD represent ring size,
    not that it did. The distinction is recorded in cfm_todo.json because
    it is what decides where the retrain slot goes.
    """
    res = _measured()
    cap = res.get("capacity_recoverability_from_distance_pattern")
    if not cap:
        pytest.skip("capacity block not produced (cohort too small)")
    assert cap["cv5_accuracy_mean"] >= 0.95, (
        f"ring size is only {cap['cv5_accuracy_mean']:.3f} recoverable from "
        "the distance pattern. That materially weakens the refutation of A4 "
        "and cfm_todo.json::a4_correction_2026_10_13 must be restated."
    )
    assert cap["holdout_70_30_accuracy"] >= 0.95, (
        f"held-out accuracy {cap['holdout_70_30_accuracy']:.3f}; a model that "
        "only fits its training folds would score far lower here"
    )
    for size in ("3", "5"):
        rec = cap["holdout_per_class_recall"].get(size)
        if rec is not None:
            assert rec >= 0.9, (
                f"recall on {size}-rings is {rec:.3f}; the separation between "
                "3-rings and larger rings is the whole point, since the "
                "measured defect is three-membered rings"
            )


# ---------------------------------------------------------------------------
# 3. The shape assumption that silently produced zero cycles
# ---------------------------------------------------------------------------


def test_truth_bonds_are_an_edge_index_not_an_edge_list():
    """tmQM truth bonds are (2, E). Reading them as (E, 2) builds a DIFFERENT
    graph that still yields rings, so the bug is silent.

    This was hit once while writing the measurement: it returned 0 cycles,
    which looked like 'no rings in these molecules' rather than 'wrong shape'.
    """
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    clouds = load_tmqm_clouds(
        PROJECT_ROOT / ".data" / "molmetal" / "tmQM", n_clouds=2, seed=1234
    )
    b = clouds[0].bonds.detach().cpu().numpy()
    assert b.ndim == 2, f"expected 2-D truth bonds, got shape {b.shape}"
    assert b.shape[0] == 2 and b.shape[1] != 2, (
        f"truth bonds look like an (E, 2) edge list, shape {b.shape}; the "
        "measurement's .T transpose would then corrupt the graph"
    )


# ---------------------------------------------------------------------------
# 4. The receptive-field claim depends on the depth
# ---------------------------------------------------------------------------


def test_head_actually_has_three_layers():
    """A4's refutation only bites because the receptive field reaches far
    enough. If the trained head were 1-layer, the 1-hop argument would still
    hold for 3-vs-4 but the general claim would weaken."""
    import torch

    ckpt = (
        PROJECT_ROOT / "molmetal" / "reports" / "cfm_patience10_seed0"
        / "checkpoints" / "cfm_yuelbond_wasserstein_seed0_last.pt"
    )
    if not ckpt.is_file():
        pytest.skip(f"checkpoint not on disk: {ckpt}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = ck["args"]
    args = args if isinstance(args, dict) else vars(args)
    assert args.get("n_layers") == 3, (
        f"checkpoint reports n_layers={args.get('n_layers')}, not 3; the "
        "receptive-field argument in cfm_todo.json::A4 must be restated"
    )
    sd = ck["model_state_dict"]
    blocks = {k.split(".")[2] for k in sd if "msg_mlps" in k}
    assert len(blocks) == 3, (
        f"state_dict has {len(blocks)} msg_mlp blocks, not 3; the args field "
        "and the weights disagree"
    )


# ---------------------------------------------------------------------------
# 5. The recorded metric must match this measurement
# ---------------------------------------------------------------------------


def test_recorded_metric_agrees():
    """The file cfm_todo.json cites must not drift from what reproduces."""
    if not METRIC.is_file():
        pytest.skip(f"metric not on disk: {METRIC}")
    import json

    rec = json.loads(METRIC.read_text(encoding="utf-8"))
    assert rec["measured" if "measured" in rec else "per_ring_size"], "empty metric"
    ov = rec["pairwise_overlap_exact_values"]["3_vs_4"]
    assert ov == 0.0, (
        f"the recorded metric says 3-vs-4 overlap is {ov}, not 0.0. Either "
        "the script changed or the metric was hand-edited; both invalidate "
        "cfm_todo.json::a4_correction_2026_10_13."
    )
    assert rec["setup"]["cutoff_angstrom"] == 3.5, (
        "cutoff must stay at the head's default 3.5 A or the comparison "
        "against the trained checkpoints is meaningless"
    )
