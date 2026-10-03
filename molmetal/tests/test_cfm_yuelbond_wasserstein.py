"""Pytest: Wf-124 Phase 2 — YuelBond + Wasserstein + Catherine training script.

Fifteen tests covering the Wf-124 P2 architectural wiring:

1. ``test_script_imports_clean`` — the training script imports without
   error (no missing names, no circular imports).
2. ``test_cfm_joint_model_forward_returns_pair_shapes`` — running the
   joint model on a cisplatin-like cloud returns logits of the right
   shape + per-atom velocity predictions.
3. ``test_cfm_joint_model_handles_variable_atom_count`` — model is
   robust to clouds of 3-9 atoms (small synthetic variations).
4. ``test_apply_closure_bounded_mask_preserves_shape`` — the closure
   bias mask returns same shape as input logits.
5. ``test_apply_closure_bounded_mask_prefers_single_bond`` — when
   the bias dict favours CuAAC (which maps to SINGLE), the BOND_SINGLE
   column gets the highest boost.
6. ``test_build_reference_positions_is_canonical_square_planar`` —
   the reference positions are 1 Pt + 4 donors at 2.05 Å.
7. ``test_build_training_cohort_returns_expected_count`` — building
   a training cohort of N clouds returns exactly N clouds.
8. ``test_evaluate_decode_ratio_smoke_on_initial_model`` — running
   the baseline (untrained) model produces a valid n_ok in [0, n_val].
9. ``test_closure_bounded_decoder_weight_swap_restores`` — the
   decoder's last-layer weights are restored after the temporary
   closure-bounded swap (no leakage).
10. ``test_per_epoch_metrics_dataclass_round_trip`` — the per-epoch
    metric dataclass serialises to/from JSON cleanly.
11. ``test_wasserstein_decay_check_short_window_passes`` — a
    monotonic-decreasing short history passes the decay check.
12. ``test_pocket_specific_closure_bias_returns_finite_dict`` — the
    per-pocket bias is finite and sums to ~1.0 (or 0.0 if pocket is
    unknown).
13. ``test_evaluate_decode_ratio_does_not_mutate_model_params`` —
    running the evaluator does not modify the head's weights in
    place (only via the explicit restore).
14. ``test_train_one_epoch_returns_five_floats`` — train_one_epoch
    returns a 5-tuple of finite floats on the synthetic cohort.
15. ``test_target_decode_ratio_set_via_args`` — the script's
    ``--target-decode-ratio`` CLI flag is honoured at the parser
    level (≥ 0.0, ≤ 1.0).

All tests are **CPU-only** and use synthetic Pt-click clouds (so the
synthetic-cohort overhead is < 1 s).  No GPU / no external evaluators
required.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _import_module_safe():
    """Import the training script as a module without running main()."""
    import importlib.util

    script = PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    spec = importlib.util.spec_from_file_location(
        "train_cfm_yuelbond_wasserstein",
        str(script),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except SystemExit:
        # ``argparse`` might call ``sys.exit`` if ``main()`` runs
        # accidentally — handle just in case.
        pass
    return mod


def _cisplatin_like_cloud() -> "AtomCloud":
    from molmetal.models.bond_head import AtomCloud

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.05, 0.0, 0.0],
            [-2.05, 0.0, 0.0],
            [0.0, 2.05, 0.0],
            [0.0, -2.05, 0.0],
        ],
        dtype=torch.float32,
    )
    atomic_numbers = torch.tensor([78, 17, 17, 7, 7], dtype=torch.long)
    return AtomCloud(positions=positions, atomic_numbers=atomic_numbers)


# ---------------------------------------------------------------------------
# Test 1: clean import
# ---------------------------------------------------------------------------
def test_script_imports_clean() -> None:
    """The training script module imports without syntax/import errors."""
    mod = _import_module_safe()
    assert hasattr(mod, "CFMJointModel")
    assert hasattr(mod, "apply_closure_bounded_mask")
    assert hasattr(mod, "evaluate_decode_ratio")
    assert hasattr(mod, "train_one_epoch")
    assert hasattr(mod, "build_training_cohort")
    assert hasattr(mod, "build_reference_positions")
    assert hasattr(mod, "parse_args")
    assert hasattr(mod, "EpochMetrics")
    assert hasattr(mod, "main")


# ---------------------------------------------------------------------------
# Test 2: joint model forward returns expected shapes
# ---------------------------------------------------------------------------
def test_cfm_joint_model_forward_returns_pair_shapes() -> None:
    mod = _import_module_safe()
    model = mod.CFMJointModel(hidden_dim=32, n_layers=2)
    cloud = _cisplatin_like_cloud()
    logits, velocities = model(cloud)
    # logits: (E, NUM_BOND_CLASSES); E = upper-tri pairs within cutoff
    assert logits.dim() == 2
    assert logits.shape[-1] == 5  # NUM_BOND_CLASSES
    # velocities: (N_atoms, 3)
    assert velocities.dim() == 2
    assert velocities.shape[0] == 5
    assert velocities.shape[1] == 3


# ---------------------------------------------------------------------------
# Test 3: model handles variable atom counts
# ---------------------------------------------------------------------------
def test_cfm_joint_model_handles_variable_atom_count() -> None:
    mod = _import_module_safe()
    from molmetal.models.bond_head import AtomCloud

    model = mod.CFMJointModel(hidden_dim=16, n_layers=2)
    for n_atoms in (3, 5, 7, 9):
        positions = torch.randn(n_atoms, 3)
        atomic_numbers = torch.ones(n_atoms, dtype=torch.long) * 6  # all C
        # Inject Pt at slot 0 for n_atoms >= 5
        if n_atoms >= 5:
            atomic_numbers[0] = 78
        cloud = AtomCloud(positions=positions, atomic_numbers=atomic_numbers)
        logits, velocities = model(cloud)
        assert logits.dim() == 2
        assert logits.shape[-1] == 5
        assert velocities.shape == (n_atoms, 3)


# ---------------------------------------------------------------------------
# Test 4: closure-bounded mask preserves shape
# ---------------------------------------------------------------------------
def test_apply_closure_bounded_mask_preserves_shape() -> None:
    mod = _import_module_safe()
    torch.manual_seed(0)
    logits = torch.randn(7, 5)  # E=7 pairs, 5 classes
    bias = {
        "CuAAC": 0.30,
        "SPAAC": 0.10,
        "Suzuki": 0.20,
        "AmideCoupling": 0.30,
        "ThiolEne": 0.10,
    }
    masked = mod.apply_closure_bounded_mask(
        logits,
        bias,
        prefer_dative=True,
    )
    assert masked.shape == logits.shape
    assert not torch.allclose(masked, logits)


# ---------------------------------------------------------------------------
# Test 5: closure-bounded mask prefers SINGLE bond
# ---------------------------------------------------------------------------
def test_apply_closure_bounded_mask_prefers_single_bond() -> None:
    mod = _import_module_safe()
    from molmetal.models.bond_head import BOND_SINGLE

    torch.manual_seed(0)
    # All-zero logits, then the mask should boost BOND_SINGLE the most.
    logits = torch.zeros(3, 5)
    bias = {
        "CuAAC": 0.30,
        "SPAAC": 0.10,
        "Suzuki": 0.20,
        "AmideCoupling": 0.30,
        "ThiolEne": 0.10,
    }
    masked = mod.apply_closure_bounded_mask(
        logits,
        bias,
        prefer_dative=True,
    )
    # The SINGLE column should have the highest mean boost.
    mean_per_class = masked.mean(dim=0)
    argmax_class = int(mean_per_class.argmax())
    assert (
        argmax_class == BOND_SINGLE
    ), f"bias should prefer SINGLE (idx {BOND_SINGLE}), got {argmax_class}"


# ---------------------------------------------------------------------------
# Test 6: reference positions are canonical Pt-square-planar
# ---------------------------------------------------------------------------
def test_build_reference_positions_is_canonical_square_planar() -> None:
    mod = _import_module_safe()
    ref = mod.build_reference_positions()
    assert ref.shape == (5, 3)
    # Pt at origin.
    assert torch.allclose(ref[0], torch.zeros(3), atol=1e-6)
    # 4 donors at ±2.05 along x or y.
    expected_distances = {
        2.05,  # |ref[1][0]| = 2.05
        2.05,  # |ref[2][0]| = 2.05
        2.05,  # |ref[3][1]| = 2.05
        2.05,  # |ref[4][1]| = 2.05
    }
    actual = sorted([round(float(torch.norm(ref[i]).item()), 3) for i in range(1, 5)])
    assert set(actual) == {2.05}, f"per-atom distances should all be 2.05; got {actual}"


# ---------------------------------------------------------------------------
# Test 7: training cohort builder returns N clouds
# ---------------------------------------------------------------------------
def test_build_training_cohort_returns_expected_count() -> None:
    mod = _import_module_safe()
    n = 16
    clouds = mod.build_training_cohort(n, seed=0, jitter=0.20)
    assert len(clouds) == n
    # Each cloud is a 5-atom Pt-click cloud by default.
    for c in clouds:
        assert c.positions.shape == (5, 3)
        assert c.atomic_numbers.tolist() == [78, 17, 17, 7, 7]


# ---------------------------------------------------------------------------
# Test 8: evaluate_decode_ratio smoke on initial model
# ---------------------------------------------------------------------------
def test_evaluate_decode_ratio_smoke_on_initial_model() -> None:
    mod = _import_module_safe()
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    model = mod.CFMJointModel(hidden_dim=16, n_layers=2)
    decoder = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=16,
        n_layers=2,
    )
    val_clouds = mod.build_synthetic_pt_click_clouds(
        n_clouds=4,
        jitter=0.20,
        seed=99,
    )
    # Move to CPU; this test is CPU-only.
    n_ok, n_total = mod.evaluate_decode_ratio(
        model=model,
        decoder=decoder,
        val_clouds=val_clouds,
        device=torch.device("cpu"),
        closure_bounded=False,  # smoke; no bias
    )
    assert n_total == 4
    assert 0 <= n_ok <= n_total


# ---------------------------------------------------------------------------
# Test 9: closure-bounded decoder weight swap restores
# ---------------------------------------------------------------------------
def test_closure_bounded_decoder_weight_swap_restores() -> None:
    """The closure-bounded swap in evaluate_decode_ratio must restore
    the decoder's head weights after each call (no leakage across
    iterations)."""
    mod = _import_module_safe()
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    model = mod.CFMJointModel(hidden_dim=16, n_layers=2)
    decoder = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=16,
        n_layers=2,
    )
    val_clouds = mod.build_synthetic_pt_click_clouds(
        n_clouds=4,
        jitter=0.20,
        seed=99,
    )
    # Snapshot pre-call bias from the last pair-head layer.
    pair_head_obj = (
        decoder._decoder.head.pair_head
        if hasattr(decoder._decoder.head, "pair_head")
        else decoder._decoder.head.edge_mlp
    )
    last_layer = pair_head_obj[-1]
    pre = last_layer.bias.data.clone()
    bias = {
        "CuAAC": 0.30,
        "SPAAC": 0.10,
        "Suzuki": 0.20,
        "AmideCoupling": 0.30,
        "ThiolEne": 0.10,
    }
    mod.evaluate_decode_ratio(
        model=model,
        decoder=decoder,
        val_clouds=val_clouds,
        device=torch.device("cpu"),
        closure_bounded=True,
        closure_bias=bias,
    )
    post = last_layer.bias.data
    assert torch.allclose(
        pre, post
    ), "closure-bounded weight swap should restore head bias"


# ---------------------------------------------------------------------------
# Test 10: EpochMetrics dataclass JSON round-trip
# ---------------------------------------------------------------------------
def test_per_epoch_metrics_dataclass_round_trip() -> None:
    mod = _import_module_safe()
    from dataclasses import asdict

    m = mod.EpochMetrics(
        epoch=3,
        wall_time_s=1.234,
        train_loss=0.42,
        bond_ce_loss=0.37,
        wasserstein_loss=0.05,
        catherine_audit_pass=True,
        decode_ratio_rdkit_strict=0.5,
        decode_ratio_val=4,
        decode_ratio_n=8,
        decoding_attempted=True,
        lr=3.0e-4,
        grad_norm=1.2,
    )
    d = asdict(m)
    j = json.dumps(d)
    assert "epoch" in j
    assert "catherine_audit_pass" in j
    assert "decode_ratio_rdkit_strict" in j


# ---------------------------------------------------------------------------
# Test 11: Wasserstein decay check on short monotonic history
# ---------------------------------------------------------------------------
def test_wasserstein_decay_check_short_window_passes() -> None:
    mod = _import_module_safe()
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        wasserstein_decay_check,
    )

    history = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    assert wasserstein_decay_check(history, window=5, tol=1e-3) is True


# ---------------------------------------------------------------------------
# Test 12: pocket-specific closure bias returns finite dict
# ---------------------------------------------------------------------------
def test_pocket_specific_closure_bias_returns_finite_dict() -> None:
    mod = _import_module_safe()
    from molmetal.molmetal_lam.reward.pocket_closure_bias import (
        pocket_specific_closure_bias,
    )

    verdict = pocket_specific_closure_bias("1h36")
    assert verdict.is_known_pocket
    assert verdict.bias_sum == pytest.approx(1.0, abs=1e-9)
    assert all(
        math.isfinite(v) for v in verdict.bias_dict.values()
    ), "bias_dict values should all be finite"


# ---------------------------------------------------------------------------
# Test 13: evaluate_decode_ratio does not mutate model params
# ---------------------------------------------------------------------------
def test_evaluate_decode_ratio_does_not_mutate_model_params() -> None:
    mod = _import_module_safe()
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    torch.manual_seed(0)
    model = mod.CFMJointModel(hidden_dim=16, n_layers=2)
    decoder = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=16,
        n_layers=2,
    )
    # Snapshot all params.
    pre = {n: p.detach().clone() for n, p in model.named_parameters()}
    val_clouds = mod.build_synthetic_pt_click_clouds(
        n_clouds=4,
        jitter=0.20,
        seed=99,
    )
    bias = {
        "CuAAC": 0.30,
        "SPAAC": 0.10,
        "Suzuki": 0.20,
        "AmideCoupling": 0.30,
        "ThiolEne": 0.10,
    }
    mod.evaluate_decode_ratio(
        model=model,
        decoder=decoder,
        val_clouds=val_clouds,
        device=torch.device("cpu"),
        closure_bounded=True,
        closure_bias=bias,
    )
    for n, p in model.named_parameters():
        assert torch.allclose(
            pre[n], p.detach()
        ), f"parameter {n} should not be mutated by evaluate_decode_ratio"


# ---------------------------------------------------------------------------
# Test 14: train_one_epoch returns a 5-tuple of finite floats
# ---------------------------------------------------------------------------
def test_train_one_epoch_returns_five_floats() -> None:
    mod = _import_module_safe()
    import torch.optim as optim
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    model = mod.CFMJointModel(hidden_dim=16, n_layers=2)
    decoder = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=16,
        n_layers=2,
    )
    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    train_clouds = mod.build_training_cohort(n_train=4, seed=0, jitter=0.10)
    out = mod.train_one_epoch(
        model=model,
        decoder=decoder,
        optimiser=optimiser,
        train_clouds=train_clouds,
        device=torch.device("cpu"),
        batch_size=4,
        wasserstein_weight=0.1,
        closure_bounded=True,
        closure_bias={
            "CuAAC": 0.30,
            "SPAAC": 0.10,
            "Suzuki": 0.20,
            "AmideCoupling": 0.30,
            "ThiolEne": 0.10,
        },
        reference_positions=mod.build_reference_positions(),
    )
    assert len(out) == 5
    for v in out:
        assert isinstance(v, float)
        assert math.isfinite(v), f"value {v} should be finite"


# ---------------------------------------------------------------------------
# Test 15: target decode_ratio CLI flag is honoured
# ---------------------------------------------------------------------------
def test_target_decode_ratio_set_via_args() -> None:
    mod = _import_module_safe()
    # Use argparse directly to verify the CLI wires the target.
    import argparse

    parser = argparse.ArgumentParser()
    # Re-use the training script's parser via a small shim:
    import sys as _sys

    saved_argv = _sys.argv
    try:
        _sys.argv = [
            "train_cfm_yuelbond_wasserstein.py",
            "--target-decode-ratio",
            "0.85",
            "--epochs",
            "5",
        ]
        args = mod.parse_args()
    finally:
        _sys.argv = saved_argv
    assert args.target_decode_ratio == 0.85
    assert 0.0 <= args.target_decode_ratio <= 1.0
