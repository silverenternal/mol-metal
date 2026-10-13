# SPDX-License-Identifier: Apache-2.0
"""The bond head is not the ring defect -- and this pins that measurement.

THE CLAIM BEING PINNED
----------------------
``cfm_todo.json`` claim A4 said ring size is UNREPRESENTABLE in the per-edge
bond head, and used that to argue against spending the single authorised
bond-head retrain on ring closure. A4 was refuted on capacity (a linear model
on raw distances classifies ring size at 1.000 held-out,
``metrics/by_metric/cfm_a4_receptive_field.json``).

Capacity is not behaviour, though, and the behaviour result is the one that
actually settles where the retrain slot goes. Measured on CLEAN tmQM geometry
-- fully in-distribution for a head trained on clean CSD structures -- the
trained head's ring-size distribution is CLOSER to the real corpus than the
truth bonds are:

    5/6-membered   TRUTH 76.10%   HEAD 82.62%
    3-membered     TRUTH 18.26%   HEAD 12.79%
    total rings    TRUTH 657      HEAD 305

So the 89.81% three-membered rate in GENERATED output cannot be blamed on a
head that, fed clean geometry, gets ring size right. The defect lives
upstream, in the geometry the velocity field supplies.

THE FAILURE MODE THIS FILE GUARDS
---------------------------------
A wrong number here would be expensive, because it is the measurement that
redirects the retrain slot. Two specific ways it could go wrong, both of
which produce plausible-looking output rather than an exception:

1. **Wrong Z table.** The checkpoint's ``atom_encoder`` is
   ``Linear(128, 18)``; the default Pt-only table is width 13. Loading with
   the wrong table either raises (good) or -- if the code falls back to
   ``strict=False`` -- silently produces a confident head with garbage
   weights, and every number below becomes fiction while still printing.
   ``test_checkpoint_loads_strictly_with_the_all6_table`` pins strict loading
   and the width.
2. **Truth-bond orientation.** tmQM bonds arrive as an ``(2, E)`` edge-index.
   Reading that as an ``(E, 2)`` list builds a DIFFERENT graph that still
   yields rings, so the bug is silent -- it was hit twice while writing these
   scripts. ``test_truth_bonds_are_an_edge_index`` pins the shape.

RED CHECKS (each run and observed to fail)
------------------------------------------
* Randomise the head's weights after loading -> the predicted rings collapse
  to near zero and ``test_head_rings_are_nonempty_and_plausible`` FAILS.
* Count rings from a graph built with the transposed-but-not-transposed bond
  array (drop the ``.T``) -> ``test_truth_bonds_are_an_edge_index`` FAILS.
* Assert the head's 3-membered fraction is WORSE than truth's (inverting the
  finding) -> ``test_head_is_not_worse_than_truth_on_ring_size`` FAILS.
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
    PROJECT_ROOT / "molmetal" / "scripts"
    / "wf_head_ring_size_behaviour_2026_10_13.py"
)
METRIC = (
    PROJECT_ROOT / "metrics" / "by_metric" / "cfm_head_ring_size_behaviour.json"
)

_RESULT = None


def _measured():
    global _RESULT
    if _RESULT is None:
        mod = importlib.import_module(
            "molmetal.scripts.wf_head_ring_size_behaviour_2026_10_13"
        )
        _RESULT = mod.measure(n_clouds=40)
    return _RESULT


# ---------------------------------------------------------------------------
# 1. The checkpoint must load strictly, or every number here is fiction
# ---------------------------------------------------------------------------


def test_checkpoint_loads_strictly_with_the_all6_table():
    """The Z table is a model dimension, and the wrong one fails silently."""
    mod = importlib.import_module(
        "molmetal.scripts.wf_head_ring_size_behaviour_2026_10_13"
    )
    model, ck, args = mod.load_model()
    import torch

    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        _YUELBOND_Z_TABLE_ALL6,
    )

    assert len(_YUELBOND_Z_TABLE_ALL6) == 18, "all6 table width changed"
    enc = model.head.atom_encoder
    assert enc.in_features == len(_YUELBOND_Z_TABLE_ALL6), (
        f"atom_encoder expects {enc.in_features} inputs, but the all6 table is "
        f"{len(_YUELBOND_Z_TABLE_ALL6)} wide. Loading would have failed -- or, "
        "worse, silently succeeded with the wrong vocabulary."
    )
    assert model.head.n_layers == 3, (
        f"head depth is {model.head.n_layers}, not 3; the receptive-field "
        "argument in cfm_todo.json::A4 must be restated"
    )


def test_truth_bonds_are_an_edge_index():
    """tmQM truth bonds are (2, E). Reading them as (E, 2) is a silent bug."""
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    clouds = load_tmqm_clouds(
        PROJECT_ROOT / ".data" / "molmetal" / "tmQM", n_clouds=2, seed=1234
    )
    b = clouds[0].bonds.detach().cpu().numpy()
    assert b.ndim == 2 and b.shape[0] == 2 and b.shape[1] != 2, (
        f"truth bonds have shape {b.shape}; the measurement's .T transpose "
        "would corrupt the graph while still producing rings"
    )


# ---------------------------------------------------------------------------
# 2. The measurement must not be vacuous
# ---------------------------------------------------------------------------


def test_head_rings_are_nonempty_and_plausible():
    """Guards the direction where this measurement could silently report 0.

    If the head predicted nothing, or the graph construction broke, the
    fraction comparison below would divide by a near-zero total and read as
    a finding. Fail closed instead.
    """
    res = _measured()
    # These live under "setup", not at top level. An earlier version of this
    # test read them from the top level and raised KeyError -- which pytest
    # reports as an ERROR, so it would have looked like a passing guard to
    # anyone skimming for "no failures".
    s = res["setup"]
    assert s["n_clouds_used"] > 0, "no clouds processed"
    assert s["n_candidate_pairs"] > 0, "the head emitted no candidate pairs"
    assert s["n_predicted_bonds"] > 0, "head predicted zero bonds"
    assert res["n_truth_rings"] > 0, "truth side produced no rings"
    assert res["n_head_rings"] > 0, (
        "the head predicted no rings at all -- the comparison below would be "
        "vacuous rather than a finding"
    )


# ---------------------------------------------------------------------------
# 3. The finding itself
# ---------------------------------------------------------------------------


def test_head_is_not_worse_than_truth_on_ring_size():
    """The load-bearing result: on CLEAN geometry the head is not the defect.

    Red check: this asserts the head's 3-membered fraction is no worse than
    truth's. Inverting it (asserting the head IS worse) makes the test fail,
    which is the point -- the whole retrain-slot redirection rests on this
    direction holding.
    """
    res = _measured()
    head3 = res["frac_3_membered"]["head"]
    truth3 = res["frac_3_membered"]["truth"]
    head56 = res["frac_5_or_6"]["head"]
    truth56 = res["frac_5_or_6"]["truth"]

    assert head3 <= truth3, (
        f"head 3-membered {head3:.2%} is now WORSE than truth {truth3:.2%} on "
        "clean geometry. If that holds, the head IS a ring-closure defect and "
        "the retrain slot argument changes; cfm_todo.json::A4's weaker form "
        "must be reinstated."
    )
    assert head56 >= truth56, (
        f"head 5/6-membered {head56:.2%} is now BELOW truth {truth56:.2%} on "
        "clean geometry -- same consequence as above."
    )


def test_the_head_still_underproduces_rings_in_total():
    """The honest remainder, pinned so it cannot be quietly dropped.

    A correct ring-size DISTRIBUTION is not a correct ring-size COUNT: the
    head predicts far fewer rings than the truth bonds contain. This is why
    the result rules out 'the head cannot do ring closure' WITHOUT ruling out
    the separate macro-F1 deficit.
    """
    res = _measured()
    assert res["n_head_rings"] < res["n_truth_rings"], (
        f"head now predicts {res['n_head_rings']} rings against "
        f"{res['n_truth_rings']} in truth. If it matches, the head is not "
        "dropping bonds, and the 'it under-counts' caveat in "
        "cfm_todo.json::cfm_head_ring_size_behaviour no longer holds."
    )


# ---------------------------------------------------------------------------
# 4. The recorded metric must match
# ---------------------------------------------------------------------------


def test_recorded_metric_matches_this_measurement():
    if not METRIC.is_file():
        pytest.skip(f"metric not on disk: {METRIC}")
    import json

    rec = json.loads(METRIC.read_text(encoding="utf-8"))
    assert rec["setup"]["state_dict_loaded"].startswith("strict=True")
    assert rec["frac_3_membered"]["head"] <= rec["frac_3_membered"]["truth"], (
        "the recorded metric contradicts the finding in its own verdict text"
    )
    assert "not the ring defect" in rec["verdict"].lower()
