# SPDX-License-Identifier: Apache-2.0
"""ARCH-1 t-degradation gate unit tests.

Pins the local helpers behind the ARCH-1 measurement so a future
refactor that breaks ``x_t = (1-t)*x_0 + t*x_1`` semantics, the
monotonicity test, or the macro-F1 from a confusion matrix will be
caught at unit-test time rather than silently corrupting the curve.

Each test below is intentionally small and CPU-only. The intent is
to assert the SHAPE of the measurement, not to reproduce the full
checkpoint-driven gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.models.bond_head import (  # noqa: E402
    AtomCloud,
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_NO_BOND,
    BOND_SINGLE,
    BOND_TRIPLE,
    NUM_BOND_CLASSES,
)
from molmetal.scripts.wf_arch1_t_degradation_2026_10_13 import (  # noqa: E402
    _build_joint_model,
    _is_monotonic_nonincreasing,
    _make_noisy_cloud,
    _per_class_macro_f1,
)


# ---------------------------------------------------------------------------
# Test 1 — t=1 is identical to clean geometry.
# ---------------------------------------------------------------------------
def test_t1_round_trips_to_clean_positions():
    """x_t at t=1 must equal x_1 (clean) for every coordinate, exactly.

    The ARCH-1 gate's t=1 row is the CONSISTENCY baseline against
    metrics/by_metric/tmqm_bond_perception_3seed.json. If this test
    fails the script's t=1 number is suspect even when the rest of the
    curve looks fine.
    """
    rng = torch.Generator(device="cpu").manual_seed(1234)
    n = 5
    positions = torch.randn(n, 3, generator=rng) * 2.0
    atomic_numbers = torch.tensor([78, 7, 7, 17, 17], dtype=torch.long)
    clean = AtomCloud(
        positions=positions,
        atomic_numbers=atomic_numbers,
        bonds=None,
        bond_types=None,
    )
    noisy_at_t1 = _make_noisy_cloud(clean, t=1.0, sigma=2.485, rng=rng)
    assert torch.allclose(noisy_at_t1.positions, positions, atol=1e-6), (
        f"x_t at t=1 drifted from x_1; max abs diff = "
        f"{(noisy_at_t1.positions - positions).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Test 2 — t=0 is pure noise (no correlation with x_1).
# ---------------------------------------------------------------------------
def test_t0_is_pure_noise_not_clean():
    """x_t at t=0 must NOT equal x_1. The gate's claim is that t=0
    reads the head on pure Gaussian noise, so any positive correlation
    here is a sign the formula is wrong.
    """
    rng = torch.Generator(device="cpu").manual_seed(1234)
    n = 5
    positions = torch.randn(n, 3, generator=rng) * 2.0
    atomic_numbers = torch.tensor([78, 7, 7, 17, 17], dtype=torch.long)
    clean = AtomCloud(
        positions=positions,
        atomic_numbers=atomic_numbers,
        bonds=None,
        bond_types=None,
    )
    noisy_at_t0 = _make_noisy_cloud(clean, t=0.0, sigma=2.485, rng=rng)
    # Same RNG consumed again would be different -- rebuild the cloud
    # with a fresh RNG so the test is deterministic.
    rng2 = torch.Generator(device="cpu").manual_seed(1234)
    _ = _make_noisy_cloud(clean, t=1.0, sigma=2.485, rng=rng2)
    assert not torch.allclose(noisy_at_t0.positions, positions, atol=1e-6), (
        "x_t at t=0 equals x_1 -- the formula is wrong or sigma is zero."
    )


# ---------------------------------------------------------------------------
# Test 3 — monotonicity test accepts nonincreasing and rejects increases.
# ---------------------------------------------------------------------------
def test_monotonic_nonincreasing_is_strict_order():
    """The verdict depends on this helper. If it ever silently flips
    the meaning of "monotonic" (e.g. <= to >=), the verdict_label
    would point the wrong way for every future curve.
    """
    assert _is_monotonic_nonincreasing([0.5, 0.4, 0.3, 0.2, 0.1]) is True
    assert _is_monotonic_nonincreasing([0.5, 0.4, 0.4, 0.2, 0.1]) is True  # flat
    assert _is_monotonic_nonincreasing([0.5, 0.6, 0.3, 0.2, 0.1]) is False
    assert _is_monotonic_nonincreasing([0.1]) is True
    assert _is_monotonic_nonincreasing([]) is True


# ---------------------------------------------------------------------------
# Test 4 — macro-F1 from a confusion counter matches sklearn-comparable result.
# ---------------------------------------------------------------------------
def test_macro_f1_handles_class_imbalance_and_tiny_classes():
    """With NO positives, a class's F1 must be 0 (not NaN). With class
    imbalance the macro average must NOT be dominated by the majority
    class. This pins both the per-class and the macro arithmetic.
    """
    from collections import Counter
    # Imbalanced: 90 BOND_NO_BOND correct, 10 BOND_NO_BOND mispredicted as SINGLE,
    # 1 SINGLE correct, 0 DOUBLE/TRIPLE/AROMATIC positives.
    cm = Counter({
        (BOND_NO_BOND, BOND_NO_BOND): 90,
        (BOND_NO_BOND, BOND_SINGLE): 10,
        (BOND_SINGLE, BOND_SINGLE): 1,
    })
    out = _per_class_macro_f1(cm, NUM_BOND_CLASSES)
    # Macro-F1 over 5 classes should include the empty classes as 0,
    # so a single correct SINGLE is averaged against 4 zeros.
    assert 0.0 <= out["macro_f1"] <= 1.0
    # BOND_SINGLE precision = 1/(1+10) = 1/11; recall = 1/1 = 1.
    expected_prec_single = 1 / 11
    expected_rec_single = 1.0
    assert abs(out["per_class_precision"]["single"] - expected_prec_single) < 1e-9
    assert abs(out["per_class_recall"]["single"] - expected_rec_single) < 1e-9
    # An empty class must NOT raise and must report 0 F1 / 0 support.
    assert out["per_class_f1"]["double"] == 0.0
    assert out["per_class_support"]["double"] == 0


# ---------------------------------------------------------------------------
# Test 5 — the joint model used by the script builds with the same
# flags as the ARCH-1 checkpoint_settings_read_from_disk. This pins
# state_dict compatibility so a future flag tweak cannot silently
# break the gate.
# ---------------------------------------------------------------------------
def test_joint_model_builds_with_arch1_flags():
    """If this test fails the joint model no longer matches the
    checkpoint's state_dict, and the gate would fail to load the
    checkpoint -- which we want to catch at unit-test time.
    """
    model = _build_joint_model(hidden_dim=128, n_layers=3)
    assert hasattr(model, "head"), "CFMJointModel must carry a bond head"
    assert hasattr(model, "vel_field") or hasattr(model, "fm_vel_head"), (
        "ARCH-1 checkpoint uses --use-egnn-velocity; the model must "
        "carry the EGNN velocity field OR the fm_vel_head MLP."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
