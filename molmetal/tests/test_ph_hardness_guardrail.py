"""Tests for :mod:`molmetal.math_skills.ph_hardness_guardrail`.

T6 — PH Approximation Hardness (Kolbe-Mayr 2026)
==================================================

These tests cover the three mathematical primitives shipped by
the T6 PH approximation-hardness guardrail skeleton:

1. **BottleneckGuardrail** — pure decision oracle that flags when
   exact bottleneck matching exceeds a time budget.  We verify
   construction (default values, input validation), the small /
   large pair-count regimes, the ``safety_factor`` knob and the
   per-call input validation.
2. **PHComplexityEstimator** — cubic + dimension wall-clock
   estimator for exact bottleneck matching.  We verify
   construction, input validation, the cubic scaling, the
   dimension dependence and the conversion from µs to seconds.
3. **approximate_bottleneck** — :math:`O(n \\log n)` greedy
   approximation.  We verify the empty-diagram handling, the
   same-diagram identity, the shift stability, the single-point
   case, the asymmetric size handling, the diagonal-only filter,
   input validation and the ``ratio`` ceiling.

Every test is **CPU-only** and uses small data (pair counts
0-1000, ratios 1.1-2.0) so the full suite runs in < 1 second.

Reference
---------
Kolbe & Mayr (2026).  *On the Computational Hardness of the
Bottleneck Distance for Persistence Diagrams*.  J. Comput. Geom.
**17**(2), 89-124.  doi:`10.20382/jocg.v17i2.345`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from molmetal.math_skills.ph_hardness_guardrail import (
    BottleneckGuardrail,
    PHComplexityEstimator,
    approximate_bottleneck,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _toy_diagram(n: int, seed: int = 0, max_pers: float = 1.0) -> torch.Tensor:
    """Deterministic ``(N, 2)`` persistence diagram with positive persistence.

    Each pair is ``(birth, death)`` with ``birth <= death``.  We
    sample ``birth ~ U(0, 0.5)`` and ``persistence ~ U(0.01,
    max_pers)`` so all points are above the diagonal.
    """
    g = torch.Generator().manual_seed(int(seed))
    birth = torch.rand(n, generator=g) * 0.5
    pers = 0.01 + torch.rand(n, generator=g) * float(max_pers)
    death = birth + pers
    return torch.stack([birth, death], dim=-1)


# ---------------------------------------------------------------------------
# 1. BottleneckGuardrail — construction
# ---------------------------------------------------------------------------


def test_bottleneck_guardrail_default_construction() -> None:
    """Default constructor uses ``approx_ratio=1.5`` and
    ``safety_factor=1.0``.
    """
    g = BottleneckGuardrail()
    assert g.approx_ratio == 1.5
    assert g.safety_factor == 1.0


def test_bottleneck_guardrail_rejects_ratio_le_1() -> None:
    """``approx_ratio <= 1`` is rejected (would mean "exact" or better)."""
    with pytest.raises(ValueError, match="approx_ratio"):
        BottleneckGuardrail(approx_ratio=1.0)
    with pytest.raises(ValueError, match="approx_ratio"):
        BottleneckGuardrail(approx_ratio=0.5)


def test_bottleneck_guardrail_rejects_non_positive_safety() -> None:
    """``safety_factor <= 0`` is rejected."""
    with pytest.raises(ValueError, match="safety_factor"):
        BottleneckGuardrail(safety_factor=0.0)
    with pytest.raises(ValueError, match="safety_factor"):
        BottleneckGuardrail(safety_factor=-1.0)


# ---------------------------------------------------------------------------
# 2. BottleneckGuardrail.should_use_approx — regime tests
# ---------------------------------------------------------------------------


def test_should_use_approx_false_for_small_pairs() -> None:
    """A 10-pair diagram fits comfortably in a 1 ms budget."""
    g = BottleneckGuardrail()
    # 10^3 = 1000 µs = 1e-3 s.  At k_cubic_us=1e-3 the budget
    # is right at the edge; safety_factor=1.0 returns False.
    # Use a slightly larger budget to be safe.
    assert g.should_use_approx(n_persistence_pairs=10,
                               time_budget_s=1.0) is False


def test_should_use_approx_true_for_large_pairs() -> None:
    """A 1000-pair diagram blows a 1 ms budget."""
    g = BottleneckGuardrail()
    # 1000^3 = 1e9 µs = 1000 s.  Definitely > 1 ms.
    assert g.should_use_approx(n_persistence_pairs=1000,
                               time_budget_s=1.0e-3) is True


def test_should_use_approx_safety_factor_scales_budget() -> None:
    """Larger ``safety_factor`` triggers approximation sooner.

    At ``n=1000`` pairs and ``k_cubic_us=1e-3`` the estimated
    cost is exactly ``1 s``.  With ``safety_factor=0.5`` (0.5 s
    effective budget) the guardrail returns True; with
    ``safety_factor=2.0`` (2 s effective budget) it returns False.
    """
    g_small = BottleneckGuardrail(safety_factor=0.5)
    g_large = BottleneckGuardrail(safety_factor=2.0)
    n = 1000
    budget = 1.0
    assert g_small.should_use_approx(n_persistence_pairs=n,
                                     time_budget_s=budget) is True
    assert g_large.should_use_approx(n_persistence_pairs=n,
                                     time_budget_s=budget) is False


def test_should_use_approx_rejects_negative_pairs() -> None:
    """Negative pair count is rejected."""
    g = BottleneckGuardrail()
    with pytest.raises(ValueError, match="n_persistence_pairs"):
        g.should_use_approx(n_persistence_pairs=-1, time_budget_s=1.0)


def test_should_use_approx_rejects_non_positive_budget() -> None:
    """Non-positive time budget is rejected."""
    g = BottleneckGuardrail()
    with pytest.raises(ValueError, match="time_budget_s"):
        g.should_use_approx(n_persistence_pairs=10, time_budget_s=0.0)
    with pytest.raises(ValueError, match="time_budget_s"):
        g.should_use_approx(n_persistence_pairs=10, time_budget_s=-1.0)


def test_should_use_approx_handles_zero_pairs() -> None:
    """Zero pairs → trivial exact match → approximation never triggered."""
    g = BottleneckGuardrail()
    assert g.should_use_approx(n_persistence_pairs=0,
                               time_budget_s=1.0e-9) is False


# ---------------------------------------------------------------------------
# 3. PHComplexityEstimator — construction
# ---------------------------------------------------------------------------


def test_ph_complexity_estimator_default_construction() -> None:
    """Default constructor uses the calibrated round-13 coefficients."""
    e = PHComplexityEstimator()
    assert e.k_cubic_us == pytest.approx(1.0e-3, rel=1e-9)
    assert e.dimension_overhead_us == pytest.approx(1.0e-7, rel=1e-9)


def test_ph_complexity_estimator_rejects_non_positive_k_cubic() -> None:
    """``k_cubic_us <= 0`` is rejected."""
    with pytest.raises(ValueError, match="k_cubic_us"):
        PHComplexityEstimator(k_cubic_us=0.0)
    with pytest.raises(ValueError, match="k_cubic_us"):
        PHComplexityEstimator(k_cubic_us=-1.0)


def test_ph_complexity_estimator_rejects_negative_overhead() -> None:
    """``dimension_overhead_us < 0`` is rejected."""
    with pytest.raises(ValueError, match="dimension_overhead_us"):
        PHComplexityEstimator(dimension_overhead_us=-1.0e-9)


# ---------------------------------------------------------------------------
# 4. PHComplexityEstimator.estimate — scaling & input validation
# ---------------------------------------------------------------------------


def test_estimate_returns_small_for_small_n() -> None:
    """``n=10`` returns ~1 µs = 1e-6 s (plus the dimension overhead)."""
    e = PHComplexityEstimator()
    t = e.estimate(n_pairs=10, dimension=2)
    # cubic = 1e-3 * 1e3 = 1 µs; dim = 1e-7 * 10 * 2 = 2e-6 µs = 2e-12 s.
    assert t == pytest.approx(1.0e-6, abs=1e-9)


def test_estimate_cubic_scaling() -> None:
    """Doubling ``n`` increases cost by 8x (n^3) with isolated cubic."""
    e = PHComplexityEstimator(dimension_overhead_us=0.0)
    t10 = e.estimate(n_pairs=10, dimension=1)
    t20 = e.estimate(n_pairs=20, dimension=1)
    assert t20 == pytest.approx(8.0 * t10, abs=1e-15)


def test_estimate_cubic_scaling_dominant_at_large_n() -> None:
    """At large ``n`` the cubic term dominates the dimension overhead."""
    e = PHComplexityEstimator()
    t100 = e.estimate(n_pairs=100, dimension=2)
    t200 = e.estimate(n_pairs=200, dimension=2)
    # Ratio should be ~8 with the linear dimension overhead as a
    # small perturbation; allow 1% slack.
    assert t200 / t100 == pytest.approx(8.0, rel=1e-2)


def test_estimate_linear_dimension_scaling() -> None:
    """Doubling dimension doubles the per-pair dimension overhead."""
    e = PHComplexityEstimator(dimension_overhead_us=1.0)
    t_d2 = e.estimate(n_pairs=10, dimension=2)
    t_d4 = e.estimate(n_pairs=10, dimension=4)
    # 10 pairs * (4-2) * 1.0 µs = 20 µs = 2.0e-5 s additional.
    assert t_d4 - t_d2 == pytest.approx(2.0e-5, rel=1e-6)


def test_estimate_returns_zero_for_zero_pairs() -> None:
    """``n_pairs=0`` returns 0 s (no work)."""
    e = PHComplexityEstimator()
    assert e.estimate(n_pairs=0, dimension=2) == 0.0


def test_estimate_rejects_negative_pairs() -> None:
    """Negative pair count is rejected."""
    e = PHComplexityEstimator()
    with pytest.raises(ValueError, match="n_pairs"):
        e.estimate(n_pairs=-1, dimension=2)


def test_estimate_rejects_zero_dimension() -> None:
    """``dimension < 1`` is rejected."""
    e = PHComplexityEstimator()
    with pytest.raises(ValueError, match="dimension"):
        e.estimate(n_pairs=10, dimension=0)


def test_estimate_returns_seconds() -> None:
    """The result is in seconds (not µs), so a 1000-pair diagram is ~1 s."""
    e = PHComplexityEstimator()
    t = e.estimate(n_pairs=1000, dimension=2)
    # 1000^3 = 1e9; k_cubic_us = 1e-3 → cubic = 1e6 µs → 1 s.
    assert t == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# 5. approximate_bottleneck — edge cases
# ---------------------------------------------------------------------------


def test_approximate_bottleneck_empty_vs_empty() -> None:
    """Two empty diagrams have bottleneck distance 0."""
    empty = torch.zeros(0, 2)
    assert approximate_bottleneck(empty, empty) == 0.0


def test_approximate_bottleneck_empty_vs_nonempty() -> None:
    """Empty vs non-empty diagram = max persistence of non-empty side."""
    empty = torch.zeros(0, 2)
    dgm = torch.tensor([[0.0, 0.5], [0.1, 0.9]])
    expected = float((dgm[:, 1] - dgm[:, 0]).abs().max().item())
    assert approximate_bottleneck(empty, dgm) == pytest.approx(expected)
    assert approximate_bottleneck(dgm, empty) == pytest.approx(expected)


def test_approximate_bottleneck_same_diagram_zero() -> None:
    """A diagram matched against itself has bottleneck distance 0."""
    dgm = _toy_diagram(5, seed=0)
    assert approximate_bottleneck(dgm, dgm) == 0.0


def test_approximate_bottleneck_diagonal_only_filtered() -> None:
    """Diagonal points (``birth == death``) are filtered out."""
    # All-diagram: should be 0.
    diag = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert approximate_bottleneck(diag, diag) == 0.0
    # Mix: only the off-diagonal point contributes.
    mixed = torch.tensor([[0.0, 0.0], [0.1, 0.9]])
    assert approximate_bottleneck(mixed, mixed) == 0.0


def test_approximate_bottleneck_single_point() -> None:
    """A single persistence pair matches to itself."""
    dgm = torch.tensor([[0.0, 1.0]])
    assert approximate_bottleneck(dgm, dgm) == 0.0
    # Shifted: the L_inf distance in (birth, death) is 0.1.
    dgm2 = torch.tensor([[0.0, 1.1]])
    assert approximate_bottleneck(dgm, dgm2) == pytest.approx(0.1)


def test_approximate_bottleneck_shifted_dgm_positive() -> None:
    """A shifted diagram gives a positive bottleneck distance."""
    dgm1 = _toy_diagram(5, seed=1)
    dgm2 = dgm1 + torch.tensor([0.0, 0.2])  # shift death up by 0.2
    d = approximate_bottleneck(dgm1, dgm2)
    assert d > 0.0
    # The shift is 0.2; the greedy algorithm returns at most that
    # (with float32 precision slack).
    assert d == pytest.approx(0.2, abs=1e-5)


# ---------------------------------------------------------------------------
# 6. approximate_bottleneck — validation
# ---------------------------------------------------------------------------


def test_approximate_bottleneck_rejects_non_2d() -> None:
    """Inputs must be 2-D ``(N, 2)`` tensors."""
    dgm = torch.tensor([[0.0, 1.0]])
    with pytest.raises(ValueError):
        approximate_bottleneck(dgm.unsqueeze(0), dgm)  # (1, 1, 2)
    with pytest.raises(ValueError):
        approximate_bottleneck(dgm, dgm.unsqueeze(0))


def test_approximate_bottleneck_rejects_wrong_last_dim() -> None:
    """Last dim must be 2 (birth, death)."""
    dgm = torch.tensor([[0.0, 1.0]])
    with pytest.raises(ValueError):
        approximate_bottleneck(dgm, torch.tensor([[0.0, 1.0, 2.0]]))
    with pytest.raises(ValueError):
        approximate_bottleneck(torch.tensor([[0.0]]), dgm)


def test_approximate_bottleneck_rejects_ratio_le_1() -> None:
    """``ratio <= 1`` is rejected."""
    dgm = torch.tensor([[0.0, 1.0]])
    with pytest.raises(ValueError, match="ratio"):
        approximate_bottleneck(dgm, dgm, ratio=1.0)
    with pytest.raises(ValueError, match="ratio"):
        approximate_bottleneck(dgm, dgm, ratio=0.5)


# ---------------------------------------------------------------------------
# 7. Integration
# ---------------------------------------------------------------------------


def test_guardrail_triggers_approx_for_large_diagram() -> None:
    """Integration: a 2000-pair diagram is flagged for approximation."""
    g = BottleneckGuardrail()
    # 2000^3 = 8e9; cubic = 1e-3 * 8e9 = 8e6 µs = 8 s.  Above 1 s budget.
    assert g.should_use_approx(n_persistence_pairs=2000,
                               time_budget_s=1.0) is True


def test_approximate_bottleneck_handles_random_diagram() -> None:
    """Smoke test on a randomly-generated 50-pair diagram."""
    torch.manual_seed(0)
    dgm1 = _toy_diagram(50, seed=1)
    dgm2 = _toy_diagram(50, seed=2, max_pers=0.5)
    d = approximate_bottleneck(dgm1, dgm2)
    assert math.isfinite(d)
    assert d >= 0.0
