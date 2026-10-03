"""Pytest — TODO-24 CFM M5 schedulers (CPU skeleton).

Covers the three utilities in
``molmetal.adapters.flow_matching_lipman.m5_schedulers``:

- :class:`AdaptiveTimestepScheduler`  (M5.1)
- :class:`LossAnnealer`                 (M5.2)
- :class:`VarianceReducedEstimator`     (M5.3)

Twenty tests; each pins one behavioural contract so a future refactor
of the schedulers can't silently break the M5 invariants without a
failing test.

Reference: TODO-24 M5 spec — "Adaptive timestep + schedulers CPU
skeleton".  All three classes are CPU-only by design (see module
docstring); no GPU, no external evaluators, no fixtures beyond
``torch.manual_seed``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman.m5_schedulers import (
    AdaptiveTimestepScheduler,
    LossAnnealer,
    VarianceReducedEstimator,
)


# ===========================================================================
# M5.1 — AdaptiveTimestepScheduler
# ===========================================================================
def test_adaptive_step_min_max():
    """Returned step is always in ``[min_step, max_step]`` across regimes."""
    sched = AdaptiveTimestepScheduler(min_step=1e-3, max_step=1e-1)
    # Zero curvature → max step.
    x_zero = torch.zeros(2, 3)
    dx_zero = torch.zeros(2, 3)
    dt_zero = sched.step_size(0.5, x_zero, dx_zero)
    assert dt_zero == 1e-1, f"zero curvature must yield max_step=0.1; got {dt_zero}"
    # High curvature (dx >> x) → min step.
    sched2 = AdaptiveTimestepScheduler(min_step=1e-3, max_step=1e-1)
    x_small = torch.full((2, 3), 0.1)
    dx_huge = torch.full((2, 3), 100.0)
    dt_huge = sched2.step_size(0.5, x_small, dx_huge)
    assert dt_huge == 1e-3, f"high curvature must yield min_step=0.001; got {dt_huge}"
    # Mid curvature → mid step.
    sched3 = AdaptiveTimestepScheduler(min_step=1e-3, max_step=1e-1)
    x_mid = torch.ones(2, 3)
    dx_mid = torch.ones(2, 3)
    dt_mid = sched3.step_size(0.5, x_mid, dx_mid)
    assert 1e-3 <= dt_mid <= 1e-1, f"mid step out of range: {dt_mid}"


def test_adaptive_step_zero_curvature():
    """Zero velocity (zero curvature) → max step (legacy fixed-step upper bound)."""
    sched = AdaptiveTimestepScheduler(min_step=1e-4, max_step=5e-2)
    x = torch.randn(4, 3)
    dx = torch.zeros_like(x)
    dt = sched.step_size(0.0, x, dx)
    assert dt == 5e-2, f"zero curvature must return max_step=0.05; got {dt}"


def test_adaptive_step_zero_position_safe():
    """Zero position (x=0) does NOT divide by zero; returns max_step."""
    sched = AdaptiveTimestepScheduler(min_step=1e-3, max_step=1e-1)
    x = torch.zeros(2, 3)
    dx = torch.randn(2, 3)
    dt = sched.step_size(0.5, x, dx)
    # With x=0 and dx ≠ 0 the curvature is huge → factor → 0 → dt → min_step.
    assert 1e-3 <= dt <= 1e-1, f"zero position must stay in range; got {dt}"


def test_adaptive_step_high_curvature_yields_min():
    """When curvature >> target, the factor shrinks dt toward min_step."""
    sched = AdaptiveTimestepScheduler(
        min_step=1e-3, max_step=1e-1, target_curvature=1.0
    )
    # x=ones, dx=1000*ones → curvature ~ 1000 >> target=1 → dt → min_step.
    x = torch.ones(1, 1)
    dx = torch.full((1, 1), 1000.0)
    dt = sched.step_size(0.5, x, dx)
    assert dt == 1e-3, f"high curvature must yield min_step; got {dt}"


def test_adaptive_step_target_curvature_yields_midpoint():
    """At curvature == target, factor = 0.5 and dt = max_step/2."""
    max_step = 0.2
    sched = AdaptiveTimestepScheduler(
        min_step=1e-4, max_step=max_step, target_curvature=5.0
    )
    # x = ones (||x|| = sqrt(3)); dx = 5*ones (||dx|| = 5*sqrt(3)) → curvature = 5.
    x = torch.ones(1, 3)
    dx = torch.full((1, 3), 5.0)
    dt = sched.step_size(0.0, x, dx)
    expected = max_step * 0.5  # factor = 1/(1 + 5/5) = 0.5
    assert abs(dt - expected) < 1e-6, (
        f"At target curvature, dt must equal max_step/2 = {expected}; got {dt}"
    )


def test_adaptive_step_records_diagnostics():
    """After ``step_size`` the diagnostics are populated."""
    sched = AdaptiveTimestepScheduler()
    assert sched.last_curvature == 0.0
    assert sched.last_step == sched.max_step
    sched.step_size(0.5, torch.ones(2, 3), torch.ones(2, 3))
    assert sched.last_curvature > 0.0
    assert sched.last_step > 0.0
    sched.reset()
    assert sched.last_curvature == 0.0


def test_adaptive_step_rejects_invalid_constructor():
    """Constructor rejects negative / zero / inverted bounds."""
    with pytest.raises(ValueError, match="min_step must be positive"):
        AdaptiveTimestepScheduler(min_step=0.0)
    with pytest.raises(ValueError, match="min_step must be positive"):
        AdaptiveTimestepScheduler(min_step=-1.0)
    with pytest.raises(ValueError, match="max_step must be positive"):
        AdaptiveTimestepScheduler(max_step=0.0)
    with pytest.raises(ValueError, match="must be <="):
        AdaptiveTimestepScheduler(min_step=0.5, max_step=0.1)
    with pytest.raises(ValueError, match="target_curvature must be positive"):
        AdaptiveTimestepScheduler(target_curvature=0.0)


def test_adaptive_step_repr_contains_params():
    """``__repr__`` lists the constructor params (debuggability)."""
    s = AdaptiveTimestepScheduler(min_step=1e-3, max_step=2e-2, target_curvature=0.5)
    r = repr(s)
    assert "min_step=0.001" in r
    assert "max_step=0.02" in r
    assert "target_curvature=0.5" in r


# ===========================================================================
# M5.2 — LossAnnealer
# ===========================================================================
def test_loss_annealer_warmup():
    """During warmup (step < warmup_steps), weight = 0."""
    annealer = LossAnnealer(total_steps=1000, warmup_frac=0.1)
    # warmup_steps = 100.  Steps [0, 99] inclusive → weight = 0.
    for step in [0, 1, 50, 99]:
        w = annealer.weight(step)
        assert w == 0.0, f"step={step} during warmup must yield weight=0; got {w}"


def test_loss_annealer_post_warmup():
    """After warmup, weight is in [0, 1] and monotone non-decreasing."""
    annealer = LossAnnealer(total_steps=1000, warmup_frac=0.1)
    last = -1.0
    for step in range(100, 1000, 50):
        w = annealer.weight(step)
        assert 0.0 <= w <= 1.0, f"weight out of range at step={step}: {w}"
        assert w >= last, f"weight must be non-decreasing; step={step} w={w} < last={last}"
        last = w


def test_loss_annealer_full_weight_at_end():
    """At step >= total_steps, weight = 1.0 (post-ramp, full weight)."""
    annealer = LossAnnealer(total_steps=1000, warmup_frac=0.1)
    assert annealer.weight(1000) == 1.0
    assert annealer.weight(2000) == 1.0
    assert annealer.weight(10000) == 1.0


def test_loss_annealer_linear_ramp_midpoint():
    """At the midpoint of the post-warmup ramp, weight = 0.5."""
    annealer = LossAnnealer(total_steps=1000, warmup_frac=0.2)
    # warmup_steps = 200; ramp_span = 800; midpoint of ramp = 200 + 400 = 600.
    w = annealer.weight(600)
    assert abs(w - 0.5) < 1e-9, f"midpoint of ramp must be 0.5; got {w}"


def test_loss_annealer_zero_warmup_linear_ramp_full_at_end():
    """warmup_frac=0 → linear ramp from 0 at step 0 to 1 at total_steps.

    Companion to ``test_loss_annealer_full_weight_at_end``: with no
    warmup, the schedule is a pure linear ramp starting at step 0.
    The ramp reaches ``weight=1.0`` exactly at ``step == total_steps``
    (not before).
    """
    annealer = LossAnnealer(total_steps=1000, warmup_frac=0.0)
    assert annealer.weight(0) == 0.0, (
        "step=0 with warmup_frac=0 must be at the ramp start (weight=0)"
    )
    assert abs(annealer.weight(500) - 0.5) < 1e-9, (
        f"step=500 (midpoint of ramp) must be 0.5; got {annealer.weight(500)}"
    )
    assert annealer.weight(999) < 1.0, "step=999 < total_steps=1000; ramp not done"
    assert annealer.weight(1000) == 1.0, "step=total_steps must be weight=1.0"
    assert annealer.weight(2000) == 1.0


def test_loss_annealer_clamps_negative_step():
    """Negative step → weight = 0.0 (defensive clamp)."""
    annealer = LossAnnealer(total_steps=1000, warmup_frac=0.1)
    assert annealer.weight(-1) == 0.0
    assert annealer.weight(-1000) == 0.0


def test_loss_annealer_rejects_invalid_constructor():
    """Constructor rejects bad total_steps / warmup_frac."""
    with pytest.raises(ValueError, match="total_steps must be positive"):
        LossAnnealer(total_steps=0)
    with pytest.raises(ValueError, match="total_steps must be positive"):
        LossAnnealer(total_steps=-100)
    with pytest.raises(ValueError, match="warmup_frac must be in"):
        LossAnnealer(total_steps=100, warmup_frac=-0.1)
    with pytest.raises(ValueError, match="warmup_frac must be in"):
        LossAnnealer(total_steps=100, warmup_frac=1.0)
    with pytest.raises(ValueError, match="warmup_frac must be in"):
        LossAnnealer(total_steps=100, warmup_frac=1.5)


def test_loss_annealer_repr_contains_params():
    """``__repr__`` lists the schedule params (debuggability)."""
    a = LossAnnealer(total_steps=2000, warmup_frac=0.05)
    r = repr(a)
    assert "total_steps=2000" in r
    assert "warmup_frac=0.05" in r
    assert "warmup_steps=100" in r


# ===========================================================================
# M5.3 — VarianceReducedEstimator
# ===========================================================================
def test_variance_reduced_zero_advantage():
    """When advantage is constant (zero variance), output equals -baseline."""
    est = VarianceReducedEstimator(baseline_decay=0.95, init_baseline=0.5)
    advantages = torch.zeros(8)
    out = est.forward(advantages)
    # Output = adv - baseline = 0 - 0.5 = -0.5 (broadcast).
    assert torch.allclose(out, torch.full((8,), -0.5))
    assert out.shape == (8,)


def test_variance_reduced_baseline_update():
    """Baseline EMA-updates toward the advantage mean over many calls."""
    est = VarianceReducedEstimator(baseline_decay=0.9)
    # First call with advantage mean = 1.0 → baseline hard-set to 1.0.
    out0 = est.forward(torch.full((4,), 1.0))
    assert abs(est.baseline - 1.0) < 1e-6
    # Second call with advantage mean = 0.0 → baseline drifts toward 0.0.
    est.forward(torch.full((4,), 0.0))
    assert 0.0 < est.baseline < 1.0, (
        f"Baseline must drift from 1.0 toward 0.0; got {est.baseline}"
    )
    # Many calls with advantage mean = 0.0 → baseline converges near 0.0.
    for _ in range(50):
        est.forward(torch.full((4,), 0.0))
    assert abs(est.baseline) < 0.05, (
        f"Baseline must converge to 0.0 after 50 calls; got {est.baseline}"
    )


def test_variance_reduced_zero_output_when_advantage_equals_baseline():
    """When advantage equals the baseline, output is exactly zero."""
    est = VarianceReducedEstimator(baseline_decay=0.5, init_baseline=0.3)
    advantages = torch.full((4,), 0.3)
    out = est.forward(advantages)
    assert torch.allclose(out, torch.zeros(4)), (
        f"adv=baseline must yield zero output; got {out}"
    )


def test_variance_reduced_first_call_hard_sets_baseline():
    """First call without ``init_baseline`` hard-sets baseline to adv mean."""
    est = VarianceReducedEstimator(baseline_decay=0.5)
    assert est.baseline == 0.0
    assert not est._initialized
    est.forward(torch.full((4,), 0.7))
    # The first-call hard-set goes through a float() cast → allow tiny
    # FP drift from the .item() roundtrip (PyTorch float32 → Python
    # float → assignment).
    assert abs(est.baseline - 0.7) < 1e-6, (
        f"baseline must be hard-set to 0.7 on first call; got {est.baseline}"
    )
    assert est._initialized


def test_variance_reduced_per_sample_baseline():
    """When caller passes a per-sample ``baseline``, output uses it directly."""
    est = VarianceReducedEstimator(baseline_decay=0.5)
    adv = torch.tensor([1.0, 2.0, 3.0, 4.0])
    baseline = torch.tensor([0.5, 0.5, 0.5, 0.5])
    out = est.forward(adv, baseline=baseline)
    expected = torch.tensor([0.5, 1.5, 2.5, 3.5])
    assert torch.allclose(out, expected)
    # The stored baseline is still EMA-updated toward the advantage mean
    # (1.0 + 2.0 + 3.0 + 4.0) / 4 = 2.5.
    assert abs(est.baseline - 2.5) < 1e-6


def test_variance_reduced_ema_decay_rate():
    """With decay=1.0, baseline tracks the latest advantage mean exactly."""
    est = VarianceReducedEstimator(baseline_decay=1.0)
    est.forward(torch.full((4,), 0.5))  # baseline → 0.5
    assert abs(est.baseline - 0.5) < 1e-6
    est.forward(torch.full((4,), 0.9))  # baseline → 0.9 (full override)
    assert abs(est.baseline - 0.9) < 1e-6


def test_variance_reduced_n_calls_increments():
    """Each forward call increments ``n_calls`` (diagnostic counter)."""
    est = VarianceReducedEstimator()
    assert est.n_calls == 0
    for k in range(1, 6):
        est.forward(torch.zeros(2))
        assert est.n_calls == k


def test_variance_reduced_reset_clears_state():
    """``reset()`` returns baseline + counter to the initial state."""
    est = VarianceReducedEstimator(baseline_decay=0.9)
    est.forward(torch.full((4,), 0.8))
    assert est.n_calls > 0
    assert est.baseline > 0.0
    est.reset()
    assert est.n_calls == 0
    assert est.baseline == 0.0
    assert not est._initialized


def test_variance_reduced_rejects_invalid_decay():
    """Constructor rejects out-of-range ``baseline_decay``."""
    with pytest.raises(ValueError, match="baseline_decay must be in"):
        VarianceReducedEstimator(baseline_decay=0.0)
    with pytest.raises(ValueError, match="baseline_decay must be in"):
        VarianceReducedEstimator(baseline_decay=-0.5)
    with pytest.raises(ValueError, match="baseline_decay must be in"):
        VarianceReducedEstimator(baseline_decay=1.5)


def test_variance_reduced_repr_contains_params():
    """``__repr__`` lists the constructor params (debuggability)."""
    est = VarianceReducedEstimator(baseline_decay=0.8)
    est.forward(torch.zeros(2))
    r = repr(est)
    assert "baseline_decay=0.8" in r
    assert "n_calls=1" in r
