"""Wf-A80 — per-class calibrated thresholds, training -> inference.

The trainer calibrates one threshold per bond class (Wf-A70) and reports
bond_per_class_test_macro_f1 0.457 vs a pooled 0.219 on the Wf-A75a
runs.  Until this wave none of it reached the decoder, which decided
every pair with one global threshold on P(bonded).  These tests pin the
new decision rule, its failure modes, and the loader that reads the
trainer's own metrics file.
"""
from __future__ import annotations

import json

import pytest
import torch

from molmetal.eval.bond_threshold_calibration import (
    load_per_class_thresholds,
    thresholds_from_metrics,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import bond_decision

# BOND_LABELS order: none, single, double, triple, aromatic
THRESHOLDS = [0.30, 0.20, 0.05, 0.15, 0.25]


def _logits(probs):
    return torch.log(torch.tensor([probs], dtype=torch.float32))


def test_per_class_finds_a_bond_the_global_threshold_misses():
    # p_bonded = 0.40 < 0.5 -> global threshold says NO_BOND.
    # p_single = 0.30 >= t_single 0.20 -> per-class says SINGLE.
    logits = _logits([0.60, 0.30, 0.10, 0.0, 0.0])
    assert bond_decision(logits, "threshold", 0.5)[0] == 0
    assert bond_decision(logits, "per_class", 0.5, None, THRESHOLDS)[0] == 1


def test_per_class_can_change_the_class_not_just_existence():
    # Global: p_bonded 0.55 >= 0.5, argmax over non-zero -> SINGLE (0.30).
    # Per-class: score_single 0.10 vs score_double 0.20 -> DOUBLE.
    logits = _logits([0.45, 0.30, 0.25, 0.0, 0.0])
    assert bond_decision(logits, "threshold", 0.5)[0] == 1
    assert bond_decision(logits, "per_class", 0.5, None, THRESHOLDS)[0] == 2


def test_per_class_returns_no_bond_when_no_class_clears_its_bar():
    logits = _logits([0.90, 0.04, 0.04, 0.01, 0.01])
    assert bond_decision(logits, "per_class", 0.5, None, THRESHOLDS)[0] == 0


def test_class_zero_threshold_is_not_a_veto():
    # t_0 = 0.99 must NOT suppress a pair whose class clears its own bar.
    thresholds = [0.99, 0.20, 0.05, 0.15, 0.25]
    logits = _logits([0.60, 0.30, 0.10, 0.0, 0.0])
    assert bond_decision(logits, "per_class", 0.5, None, thresholds)[0] == 1


def test_missing_thresholds_raise():
    logits = _logits([0.60, 0.30, 0.10, 0.0, 0.0])
    with pytest.raises(ValueError, match="per_class_thresholds"):
        bond_decision(logits, "per_class", 0.5)


def test_wrong_length_raises():
    logits = _logits([0.60, 0.30, 0.10, 0.0, 0.0])
    with pytest.raises(ValueError, match="entries"):
        bond_decision(logits, "per_class", 0.5, None, [0.2, 0.2])


def test_thresholds_from_metrics_reads_the_last_epoch():
    payload = {
        "per_epoch_metrics": [
            {"epoch": 0, "bond_class_1_best_threshold": 0.9},
            {
                "epoch": 1,
                "bond_class_0_best_threshold": 0.3,
                "bond_class_1_best_threshold": 0.2,
                "bond_class_2_best_threshold": 0.1,
                "bond_class_3_best_threshold": 0.15,
                "bond_class_4_best_threshold": 0.25,
            },
        ]
    }
    assert thresholds_from_metrics(payload) == [0.3, 0.2, 0.1, 0.15, 0.25]
    assert thresholds_from_metrics(payload, epoch=1) == [0.3, 0.2, 0.1, 0.15, 0.25]


def test_thresholds_from_metrics_raises_without_calibration():
    with pytest.raises(ValueError, match="bond_class_"):
        thresholds_from_metrics({"per_epoch_metrics": [{"epoch": 0}]})


def test_thresholds_from_metrics_falls_back_per_missing_class():
    payload = {"epoch": 0, "bond_class_1_best_threshold": 0.2}
    assert thresholds_from_metrics(payload) == [0.5, 0.2, 0.5, 0.5, 0.5]


def test_loader_reads_a_real_run_artifact():
    vals = load_per_class_thresholds(
        "results/cfm_5metal_wfa75a_w000/per_epoch_metrics.json"
    )
    assert len(vals) == 5
    assert all(0.0 <= v <= 1.0 for v in vals)


# ---------------------------------------------------------------------------
# Decoder wiring (the part that makes the training->inference hand-off real)
# ---------------------------------------------------------------------------
def _square_planar_pt():
    from molmetal.models.bond_head import AtomCloud

    return AtomCloud(
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],   # Pt
                [2.05, 0.0, 0.0],  # N
                [-2.05, 0.0, 0.0], # N
                [0.0, 2.30, 0.0],  # Cl
                [0.0, -2.30, 0.0], # Cl
            ],
            dtype=torch.float32,
        ),
        atomic_numbers=torch.tensor([78, 7, 7, 17, 17], dtype=torch.long),
    )


def test_decoder_accepts_per_class_mode_with_thresholds():
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(
        hidden_dim=16, n_layers=1, decision="per_class",
        per_class_thresholds=THRESHOLDS,
    )
    assert decoder.per_class_thresholds == THRESHOLDS
    result = decoder.decode(_square_planar_pt())
    assert result.decoded.n_atoms == 5


def test_decoder_per_class_without_thresholds_fails_loudly():
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=16, n_layers=1, decision="per_class")
    assert decoder.per_class_thresholds is None
    with pytest.raises(ValueError, match="per_class_thresholds"):
        decoder.decode(_square_planar_pt())


def test_decoder_rejects_unknown_decision_at_construction():
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    with pytest.raises(ValueError, match="unknown bond decision rule"):
        YuelBondBondDecoder(hidden_dim=16, n_layers=1, decision="perclass")

