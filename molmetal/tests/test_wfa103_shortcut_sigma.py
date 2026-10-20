# SPDX-License-Identifier: Apache-2.0
"""Wf-A103 (2026-10-20) - the ShortcutDistillLoss perturbation scale is a knob.

Wf-A102 made the shortcut self-consistency term live by building the second
view, but the perturbation size was a hard-coded 0.01 * randn_like(x_t).
That constant was never calibrated, so "0.01 A is too small" was an inference.

Wf-A103 makes it a flag (--shortcut-distill-perturb-sigma, default 0.01 so
every existing run stays bit-exact) and measures the curve:
metrics/by_metric/wfa103_shortcut_sigma_sweep.json.

The measurement lives in the JSON; these tests only pin the wire.  A test that
asserted a particular magnitude would be asserting a property of the toy
cohort's velocity field, not of the code.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
SWEEP = REPO / "molmetal" / "scripts" / "wfa103_shortcut_sigma_sweep.py"


def _parse():
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    argv = sys.argv
    sys.argv = ["trainer", "--config", os.devnull]
    try:
        return T.parse_args()
    finally:
        sys.argv = argv


def test_the_default_is_the_value_that_was_hard_coded():
    """0.01 must stay the default, or every Wf-A102-era run changes silently."""
    assert float(_parse().shortcut_distill_perturb_sigma) == 0.01


def test_the_function_default_matches_the_flag_default():
    """parse_args and train_one_epoch must agree, or a direct caller diverges."""
    import inspect

    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    sig = inspect.signature(T.train_one_epoch)
    assert "shortcut_distill_perturb_sigma" in sig.parameters
    assert float(sig.parameters["shortcut_distill_perturb_sigma"].default) == 0.01


def test_the_call_site_forwards_the_flag():
    """The flag must reach train_one_epoch; a flag nobody forwards is inert."""
    src = TRAINER.read_text()
    i = src.index("shortcut_distill_perturb_sigma=")
    # the value expression spans a line break, so match the dest, not a literal
    assert 'getattr(' in src[i : i + 200]
    assert '"shortcut_distill_perturb_sigma"' in src[i : i + 200]


def test_the_constant_is_no_longer_hard_coded():
    """The old body was 0.01 * torch.randn_like(x_t).  Pin its absence."""
    src = TRAINER.read_text()
    assert "0.01 * torch.randn_like(x_t)" not in src
    assert "float(shortcut_distill_perturb_sigma) * torch.randn_like(x_t)" in src


def test_the_sweep_script_exists_and_records_the_not_established_clause():
    """The artefact must carry its own limit, not just numbers."""
    assert SWEEP.exists()
    src = SWEEP.read_text()
    assert "not_established" in src
    assert "GPU A/B" in src


def test_the_sigma2_normalisation_recovers_a_constant_field_statistic():
    """Pin the algebra: for term = C * sigma^2 the derived column is C."""
    from molmetal.scripts.wfa103_shortcut_sigma_sweep import derive_sigma_rows

    C = 1.5e-4
    rows = [
        {"sigma": s, "shortcut_distill_class_term": C * s * s}
        for s in (0.01, 0.1, 1.0)
    ]
    summary = derive_sigma_rows(rows)
    assert summary["n"] == 3
    assert summary["spread_over_min"] == 0.0
    for r in rows:
        assert abs(r["normalized_by_sigma2"] - C) < 1e-18


def test_the_measured_artefact_is_sigma_independent():
    """The sweep is only a calibration if the normalised column is flat.

    A large spread would mean the field is not smooth at some rung, and the
    "pick a sigma" conclusion would not follow.  10% is generous; the
    measured spread is 7.6% over a 100x ladder.
    """
    import json

    art = REPO / "metrics" / "by_metric" / "wfa103_shortcut_sigma_sweep.json"
    if not art.exists():
        pytest.skip("sweep artefact not generated in this checkout")
    d = json.loads(art.read_text(encoding="utf-8"))
    s = d["sigma2_normalised"]
    assert s["n"] >= 3
    assert s["spread_over_min"] < 0.10
    # the sigma ladder must span at least two decades, or "flat" is trivial
    sig = [r["sigma"] for r in d["rows"]]
    assert max(sig) / min(sig) >= 100


def test_the_epoch_ladder_is_recorded():
    """The sigma sweep runs on an untrained field; the ladder is the control."""
    import json

    art = REPO / "metrics" / "by_metric" / "wfa103_shortcut_sigma_sweep.json"
    if not art.exists():
        pytest.skip("sweep artefact not generated in this checkout")
    d = json.loads(art.read_text(encoding="utf-8"))
    rows = d["epoch_ladder"]["rows"]
    assert [r["epochs"] for r in rows] == [1, 2, 4, 8]
    # bond_ce must actually move, or the ladder proves nothing
    assert rows[-1]["bond_ce"] < 0.5 * rows[0]["bond_ce"]
