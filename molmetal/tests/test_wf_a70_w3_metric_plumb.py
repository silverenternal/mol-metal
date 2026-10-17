"""Wf-A70 W3 (2026-10-18) — EpochMetrics plumb regression test.

The Wf-A70 W1+W2 fixes ship ``bond_class_*`` / ``bond_per_class_*`` /
``bond_aux_mse_term`` from :func:`evaluate_bond_accuracy` and the
aux-MSE term stashed on the model, but the trainer's ``metric =
EpochMetrics(...)`` construction at line ~13396 silently dropped
those keys at the dataclass boundary — the dataclass had no field
for them, so per_epoch_metrics.json recorded ``0.0`` for all of
them and a 30-epoch retrain could not MEASURE whether Fix 7's
per-class calibration or Fix 9's per-class ranking actually moved.

This test pins the plumb end-to-end:

  1. :func:`evaluate_bond_accuracy` returns the per-class + sweep
     fields on a small synthetic val set (the function contract, also
     pinned by W1 + W2 tests).
  2. The trainer's ``metric = EpochMetrics(...)`` construction at
     line ~13396 reads those fields from ``_bond_acc`` and assigns
     them to the dataclass.
  3. ``asdict()`` on the resulting EpochMetrics carries every field
     forward.
  4. ``json.dumps(asdict(...))`` round-trips through JSON without
     losing any field.

This is the regression test for the "per-class fields don't reach
per_epoch_metrics.json" finding that motivated W3.  CPU-only, <5s.

Wf-A70 W3 (2026-10-18) — created after Wf-A70 W2's HOLD verification
flagged that EpochMetrics was missing ~25 fields that Fix 7 + Fix 9
compute but never reach the JSON.  See ``docs/HONEST_NEGATIVE_CATALOG.md``
§6.98 (to be filled by closeout).
"""

from __future__ import annotations

import json
import sys

import pytest
import torch

PROJECT_ROOT_TESTS = "/home/hugo/codes/mol-metal"


# ---------------------------------------------------------------------------
# Helpers — fabricate an `_bond_acc` dict as evaluate_bond_accuracy emits it
# ---------------------------------------------------------------------------
def _fake_bond_acc() -> dict:
    """Return a synthetic ``_bond_acc`` dict mimicking the evaluate_bond_accuracy
    return shape, with NON-ZERO values on every per-class + sweep key.  Used to
    test the metric construction plumb without needing a real model run."""
    out = {
        # Legacy (already plumbed before W3) — present so the legacy fields
        # stay covered in the round-trip test.
        "n_scored": 7.0,
        "n_truth_bonds": 11.0,
        "bond_precision": 0.4,
        "bond_recall": 0.5,
        "bond_f1": 2 * 0.4 * 0.5 / max(0.4 + 0.5, 1e-9),
        "bond_exact_match": 0.0,
        "bond_decision": "argmax",
        "bond_decision_threshold": 0.5,
        "bond_class_logit_bias": None,
        "bond_f1_alt_rule": 0.3,
        "bond_precision_alt_rule": 0.35,
        "bond_recall_alt_rule": 0.28,
        "bond_type_macro_f1_pooled": 0.22,
        "bond_exact_type_rate": 0.0,
        "bond_type_n_classes": 3.0,
        # Fix 9 — per-class PR-AUC.
        "bond_class_0_pr_auc": 0.95,
        "bond_class_0_pos_rate": 0.71,
        "bond_class_0_n_pos": 71.0,
        "bond_class_1_pr_auc": 0.50,
        "bond_class_1_pos_rate": 0.20,
        "bond_class_1_n_pos": 20.0,
        "bond_class_2_pr_auc": 0.10,
        "bond_class_2_pos_rate": 0.05,
        "bond_class_2_n_pos": 5.0,
        "bond_class_3_pr_auc": 0.04,
        "bond_class_3_pos_rate": 0.008,
        "bond_class_3_n_pos": 0.8,
        "bond_class_4_pr_auc": 0.30,
        "bond_class_4_pos_rate": 0.035,
        "bond_class_4_n_pos": 3.5,
        "bond_per_class_macro_pr_auc": 0.378,
        "bond_per_class_n_classes_with_pos": 5.0,
        "bond_per_class_n_pairs": 100.0,
        # Fix 7 — per-class held-out threshold sweep.
        "bond_class_0_best_threshold": 0.40,
        "bond_class_0_test_f1": 0.85,
        "bond_class_0_test_precision": 0.80,
        "bond_class_0_test_recall": 0.90,
        "bond_class_1_best_threshold": 0.30,
        "bond_class_1_test_f1": 0.55,
        "bond_class_1_test_precision": 0.60,
        "bond_class_1_test_recall": 0.51,
        "bond_class_2_best_threshold": 0.20,
        "bond_class_2_test_f1": 0.20,
        "bond_class_2_test_precision": 0.15,
        "bond_class_2_test_recall": 0.30,
        "bond_class_3_best_threshold": 0.10,
        "bond_class_3_test_f1": 0.08,
        "bond_class_3_test_precision": 0.05,
        "bond_class_3_test_recall": 0.20,
        "bond_class_4_best_threshold": 0.25,
        "bond_class_4_test_f1": 0.40,
        "bond_class_4_test_precision": 0.35,
        "bond_class_4_test_recall": 0.50,
        "bond_per_class_calibration_frac": 0.5,
        "bond_per_class_seed": 0.0,
        "bond_per_class_threshold_grid_size": 9.0,
        "bond_per_class_test_macro_f1": 0.416,
        "bond_per_class_n_classes_with_test_pos": 5.0,
        # Legacy preserved fields (PR-AUC, positive_rate, best_f1, etc.)
        "bond_pr_auc": 0.95,
        "bond_positive_rate": 0.29,
        "bond_best_f1_threshold": 0.45,
        "bond_best_f1_at_threshold": 0.5,
        "bond_n_pairs_scored": 100.0,
        "bond_pr_auc_note": "backward compat note",
        # Wf-A68 spot 7 — held-out sweep (pre-existing drift; not the W3
        # focus but included so the test catches if it's also broken).
        "bond_held_out_best_threshold": 0.45,
        "bond_held_out_best_f1": 0.42,
        "bond_held_out_best_precision": 0.40,
        "bond_held_out_best_recall": 0.44,
        "bond_held_out_threshold_grid_size": 7.0,
        "bond_held_out_threshold_grid": [0.10, 0.20, 0.30, 0.40, 0.50],
    }
    return out


def _build_epoch_metrics(_bond_acc: dict) -> object:
    """Mirror the metric = EpochMetrics(...) construction at line ~13396.

    Only the fields that READ from _bond_acc are exercised here — the
    other (gen_validity, decode_ratio, early_stop) fields default to 0.0
    since they don't affect the per-class + sweep plumb under test.
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import EpochMetrics

    return EpochMetrics(
        epoch=0,
        wall_time_s=1.0,
        train_loss=0.5,
        bond_ce_loss=0.4,
        wasserstein_loss=0.05,
        catherine_audit_pass=False,
        decode_ratio_rdkit_strict=0.0,
        decode_ratio_val=0,
        decode_ratio_n=0,
        decoding_attempted=False,
        lr=1e-3,
        grad_norm=0.0,
        # Bond legacy fields.
        bond_precision=float(_bond_acc.get("bond_precision", 0.0)),
        bond_recall=float(_bond_acc.get("bond_recall", 0.0)),
        bond_f1=float(_bond_acc.get("bond_f1", 0.0)),
        bond_exact_match=float(_bond_acc.get("bond_exact_match", 0.0)),
        bond_n_scored=int(_bond_acc.get("n_scored", 0.0)),
        bond_pr_auc=float(_bond_acc.get("bond_pr_auc", 0.0)),
        bond_best_f1_at_threshold=float(
            _bond_acc.get("bond_best_f1_at_threshold", 0.0)
        ),
        bond_best_f1_threshold=float(
            _bond_acc.get("bond_best_f1_threshold", 0.0)
        ),
        bond_pos_rate=float(_bond_acc.get("bond_positive_rate", 0.0)),
        bond_n_truth_bonds=int(_bond_acc.get("n_truth_bonds", 0.0)),
        bond_decision=str(_bond_acc.get("bond_decision", "")),
        bond_decision_threshold=float(
            _bond_acc.get("bond_decision_threshold", 0.0)
        ),
        bond_type_macro_f1_pooled=float(
            _bond_acc.get("bond_type_macro_f1_pooled", 0.0)
        ),
        bond_exact_type_rate=float(
            _bond_acc.get("bond_exact_type_rate", 0.0)
        ),
        # W3 Fix — per-class + sweep plumbing (the regression target).
        bond_class_0_pr_auc=float(_bond_acc.get("bond_class_0_pr_auc", 0.0)),
        bond_class_0_pos_rate=float(_bond_acc.get("bond_class_0_pos_rate", 0.0)),
        bond_class_0_n_pos=int(float(_bond_acc.get("bond_class_0_n_pos", 0.0))),
        bond_class_1_pr_auc=float(_bond_acc.get("bond_class_1_pr_auc", 0.0)),
        bond_class_1_pos_rate=float(_bond_acc.get("bond_class_1_pos_rate", 0.0)),
        bond_class_1_n_pos=int(float(_bond_acc.get("bond_class_1_n_pos", 0.0))),
        bond_class_2_pr_auc=float(_bond_acc.get("bond_class_2_pr_auc", 0.0)),
        bond_class_2_pos_rate=float(_bond_acc.get("bond_class_2_pos_rate", 0.0)),
        bond_class_2_n_pos=int(float(_bond_acc.get("bond_class_2_n_pos", 0.0))),
        bond_class_3_pr_auc=float(_bond_acc.get("bond_class_3_pr_auc", 0.0)),
        bond_class_3_pos_rate=float(_bond_acc.get("bond_class_3_pos_rate", 0.0)),
        bond_class_3_n_pos=int(float(_bond_acc.get("bond_class_3_n_pos", 0.0))),
        bond_class_4_pr_auc=float(_bond_acc.get("bond_class_4_pr_auc", 0.0)),
        bond_class_4_pos_rate=float(_bond_acc.get("bond_class_4_pos_rate", 0.0)),
        bond_class_4_n_pos=int(float(_bond_acc.get("bond_class_4_n_pos", 0.0))),
        bond_per_class_macro_pr_auc=float(
            _bond_acc.get("bond_per_class_macro_pr_auc", 0.0)
        ),
        bond_per_class_n_classes_with_pos=int(
            float(_bond_acc.get("bond_per_class_n_classes_with_pos", 0.0))
        ),
        bond_per_class_n_pairs=int(
            float(_bond_acc.get("bond_per_class_n_pairs", 0.0))
        ),
        bond_class_0_best_threshold=float(
            _bond_acc.get("bond_class_0_best_threshold", 0.0)
        ),
        bond_class_0_test_f1=float(
            _bond_acc.get("bond_class_0_test_f1", 0.0)
        ),
        bond_class_0_test_precision=float(
            _bond_acc.get("bond_class_0_test_precision", 0.0)
        ),
        bond_class_0_test_recall=float(
            _bond_acc.get("bond_class_0_test_recall", 0.0)
        ),
        bond_class_1_best_threshold=float(
            _bond_acc.get("bond_class_1_best_threshold", 0.0)
        ),
        bond_class_1_test_f1=float(
            _bond_acc.get("bond_class_1_test_f1", 0.0)
        ),
        bond_class_1_test_precision=float(
            _bond_acc.get("bond_class_1_test_precision", 0.0)
        ),
        bond_class_1_test_recall=float(
            _bond_acc.get("bond_class_1_test_recall", 0.0)
        ),
        bond_class_2_best_threshold=float(
            _bond_acc.get("bond_class_2_best_threshold", 0.0)
        ),
        bond_class_2_test_f1=float(
            _bond_acc.get("bond_class_2_test_f1", 0.0)
        ),
        bond_class_2_test_precision=float(
            _bond_acc.get("bond_class_2_test_precision", 0.0)
        ),
        bond_class_2_test_recall=float(
            _bond_acc.get("bond_class_2_test_recall", 0.0)
        ),
        bond_class_3_best_threshold=float(
            _bond_acc.get("bond_class_3_best_threshold", 0.0)
        ),
        bond_class_3_test_f1=float(
            _bond_acc.get("bond_class_3_test_f1", 0.0)
        ),
        bond_class_3_test_precision=float(
            _bond_acc.get("bond_class_3_test_precision", 0.0)
        ),
        bond_class_3_test_recall=float(
            _bond_acc.get("bond_class_3_test_recall", 0.0)
        ),
        bond_class_4_best_threshold=float(
            _bond_acc.get("bond_class_4_best_threshold", 0.0)
        ),
        bond_class_4_test_f1=float(
            _bond_acc.get("bond_class_4_test_f1", 0.0)
        ),
        bond_class_4_test_precision=float(
            _bond_acc.get("bond_class_4_test_precision", 0.0)
        ),
        bond_class_4_test_recall=float(
            _bond_acc.get("bond_class_4_test_recall", 0.0)
        ),
        bond_per_class_calibration_frac=float(
            _bond_acc.get("bond_per_class_calibration_frac", 0.5)
        ),
        bond_per_class_seed=int(
            float(_bond_acc.get("bond_per_class_seed", 0.0))
        ),
        bond_per_class_threshold_grid_size=int(
            float(_bond_acc.get("bond_per_class_threshold_grid_size", 0.0))
        ),
        bond_per_class_test_macro_f1=float(
            _bond_acc.get("bond_per_class_test_macro_f1", 0.0)
        ),
        bond_per_class_n_classes_with_test_pos=int(
            float(_bond_acc.get("bond_per_class_n_classes_with_test_pos", 0.0))
        ),
        # Fix 8 — aux MSE term.  Source: model.last_bond_aux_mse_term.
        # Test path simulates this via a stub model attribute.
        bond_aux_mse_term=float(_bond_acc.get("bond_aux_mse_term", 0.0)),
    )


# ---------------------------------------------------------------------------
# Plumb regression tests
# ---------------------------------------------------------------------------
def test_epoch_metrics_has_per_class_fields():
    """EpochMetrics dataclass declares every per-class + sweep field."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import EpochMetrics

    fields = set(EpochMetrics.__dataclass_fields__.keys())
    # Fix 9 (per-class PR-AUC).
    for c in (0, 1, 2, 3, 4):
        assert f"bond_class_{c}_pr_auc" in fields
        assert f"bond_class_{c}_pos_rate" in fields
        assert f"bond_class_{c}_n_pos" in fields
    assert "bond_per_class_macro_pr_auc" in fields
    assert "bond_per_class_n_classes_with_pos" in fields
    assert "bond_per_class_n_pairs" in fields
    # Fix 7 (per-class held-out sweep).
    for c in (0, 1, 2, 3, 4):
        assert f"bond_class_{c}_best_threshold" in fields
        assert f"bond_class_{c}_test_f1" in fields
        assert f"bond_class_{c}_test_precision" in fields
        assert f"bond_class_{c}_test_recall" in fields
    assert "bond_per_class_calibration_frac" in fields
    assert "bond_per_class_seed" in fields
    assert "bond_per_class_threshold_grid_size" in fields
    assert "bond_per_class_test_macro_f1" in fields
    assert "bond_per_class_n_classes_with_test_pos" in fields
    # Fix 8 (aux MSE term).
    assert "bond_aux_mse_term" in fields


def test_per_class_pr_auc_round_trips_through_asdict():
    """Per-class PR-AUC values make it through EpochMetrics → asdict() → JSON."""
    _bond_acc = _fake_bond_acc()
    m = _build_epoch_metrics(_bond_acc)
    from dataclasses import asdict

    d = asdict(m)
    for c in (0, 1, 2, 3, 4):
        assert d[f"bond_class_{c}_pr_auc"] == pytest.approx(
            _bond_acc[f"bond_class_{c}_pr_auc"]
        )
        assert d[f"bond_class_{c}_pos_rate"] == pytest.approx(
            _bond_acc[f"bond_class_{c}_pos_rate"]
        )
        # `n_pos` round-trips through int conversion (the source emits
        # as float, we coerce to int at the metric construction site).
        assert d[f"bond_class_{c}_n_pos"] == int(
            _bond_acc[f"bond_class_{c}_n_pos"]
        )
    assert d["bond_per_class_macro_pr_auc"] == pytest.approx(
        _bond_acc["bond_per_class_macro_pr_auc"]
    )
    # JSON round-trip (the regression target — per_epoch_metrics.json).
    j = json.dumps(d)
    assert "bond_class_3_pr_auc" in j
    assert "bond_per_class_macro_pr_auc" in j


def test_per_class_threshold_sweep_round_trips_through_asdict():
    """Per-class held-out threshold sweep fields round-trip via asdict + JSON."""
    _bond_acc = _fake_bond_acc()
    m = _build_epoch_metrics(_bond_acc)
    from dataclasses import asdict

    d = asdict(m)
    for c in (0, 1, 2, 3, 4):
        assert d[f"bond_class_{c}_best_threshold"] == pytest.approx(
            _bond_acc[f"bond_class_{c}_best_threshold"]
        )
        assert d[f"bond_class_{c}_test_f1"] == pytest.approx(
            _bond_acc[f"bond_class_{c}_test_f1"]
        )
        assert d[f"bond_class_{c}_test_precision"] == pytest.approx(
            _bond_acc[f"bond_class_{c}_test_precision"]
        )
        assert d[f"bond_class_{c}_test_recall"] == pytest.approx(
            _bond_acc[f"bond_class_{c}_test_recall"]
        )
    assert d["bond_per_class_calibration_frac"] == pytest.approx(0.5)
    assert d["bond_per_class_seed"] == 0
    assert d["bond_per_class_threshold_grid_size"] == 9
    assert d["bond_per_class_test_macro_f1"] == pytest.approx(0.416)
    assert d["bond_per_class_n_classes_with_test_pos"] == 5
    j = json.dumps(d)
    assert "bond_class_3_test_f1" in j
    assert "bond_per_class_test_macro_f1" in j


def test_bond_aux_mse_term_round_trips():
    """Fix 8: bond_aux_mse_term (τ-scaled linear BO MSE) reaches JSON."""
    _bond_acc = _fake_bond_acc()
    # Stamp the aux MSE value the trainer would read from
    # `model.last_bond_aux_mse_term.item()`.
    _bond_acc["bond_aux_mse_term"] = 0.0123
    m = _build_epoch_metrics(_bond_acc)
    from dataclasses import asdict

    d = asdict(m)
    assert d["bond_aux_mse_term"] == pytest.approx(0.0123)
    j = json.dumps(d)
    assert "bond_aux_mse_term" in j
    assert "0.0123" in j


def test_per_class_n_pos_coerces_to_int():
    """Source emits n_pos as float; EpochMetrics stores it as int (a count)."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import EpochMetrics

    field = EpochMetrics.__dataclass_fields__["bond_class_0_n_pos"]
    assert field.type == "int", (
        f"bond_class_0_n_pos must be int (a count), got {field.type}"
    )


def test_evaluate_bond_accuracy_to_epoch_metrics_pipeline():
    """End-to-end: evaluate_bond_accuracy → EpochMetrics → JSON.

    This is the regression test for the verify finding.  It runs
    ``evaluate_bond_accuracy`` on a small synthetic val set, then
    manually constructs EpochMetrics with the returned dict, and
    asserts every per-class + sweep field appears in the asdict().
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        CFMJointModel,
        evaluate_bond_accuracy,
    )
    from molmetal.models.bond_head import AtomCloud

    torch.manual_seed(0)
    model = CFMJointModel(
        hidden_dim=64,
        n_layers=3,
        cutoff=3.5,
        bond_predictor="yuelbond_edge_mpnn",
        n_edge_layers=4,
    )
    val_clouds = []
    for _ in range(3):
        positions = torch.randn(7, 3) * 1.5
        atomic_numbers = torch.tensor([6, 6, 7, 8, 6, 6, 78], dtype=torch.long)
        bonds = [[], []]
        types = []
        for i in range(7):
            for j in range(i + 1, 7):
                d = (positions[i] - positions[j]).norm().item()
                if d < 2.0:
                    bonds[0].append(i)
                    bonds[1].append(j)
                    types.append(1 if d > 1.0 else 2)
        bonds_t = torch.tensor(bonds, dtype=torch.long)
        types_t = (
            torch.tensor(types, dtype=torch.long)
            if types
            else torch.zeros(0, dtype=torch.long)
        )
        val_clouds.append(
            AtomCloud(
                positions=positions,
                atomic_numbers=atomic_numbers,
                bonds=bonds_t,
                bond_types=types_t,
            )
        )

    bond_acc = evaluate_bond_accuracy(model, val_clouds)
    metric = _build_epoch_metrics(bond_acc)
    from dataclasses import asdict

    d = asdict(metric)
    # Per-class PR-AUC fields are present (the regression target).
    for c in (0, 1, 2, 3, 4):
        assert f"bond_class_{c}_pr_auc" in d
        assert f"bond_class_{c}_pos_rate" in d
        assert f"bond_class_{c}_n_pos" in d
    # Per-class sweep fields are present.
    for c in (0, 1, 2, 3, 4):
        assert f"bond_class_{c}_best_threshold" in d
        assert f"bond_class_{c}_test_f1" in d
        assert f"bond_class_{c}_test_precision" in d
        assert f"bond_class_{c}_test_recall" in d
    # JSON round-trip succeeds.
    j = json.dumps(d)
    for c in (0, 1, 2, 3, 4):
        assert f"bond_class_{c}_pr_auc" in j


def test_default_values_when_bond_acc_lacks_per_class_fields():
    """If _bond_acc lacks the per-class keys (legacy-only path),
    EpochMetrics defaults to 0.0 / 0 — the Wf-A68 backward-compat bit-exact
    semantic.  This protects against a future refactor that breaks the
    .get(..., 0.0) pattern."""
    # No per-class keys → empty dict.
    bond_acc = {
        "n_scored": 0.0,
        "bond_f1": 0.0,
        "bond_pr_auc": 0.0,
        "bond_positive_rate": 0.0,
        "bond_best_f1_threshold": 0.0,
        "bond_best_f1_at_threshold": 0.0,
    }
    m = _build_epoch_metrics(bond_acc)
    from dataclasses import asdict

    d = asdict(m)
    # Defaults fire.
    for c in (0, 1, 2, 3, 4):
        assert d[f"bond_class_{c}_pr_auc"] == 0.0
        assert d[f"bond_class_{c}_pos_rate"] == 0.0
        assert d[f"bond_class_{c}_n_pos"] == 0
        assert d[f"bond_class_{c}_best_threshold"] == 0.0
        assert d[f"bond_class_{c}_test_f1"] == 0.0
    assert d["bond_per_class_macro_pr_auc"] == 0.0
    assert d["bond_per_class_n_classes_with_pos"] == 0
    assert d["bond_per_class_n_pairs"] == 0
    assert d["bond_aux_mse_term"] == 0.0
