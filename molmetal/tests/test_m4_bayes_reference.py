"""M4's reference: the shared slope was a restriction, not the Bayes bound.

``gate_m4`` judged the field against ``_bayes_ratio``, which fits ONE slope
``v ~ a * x_t`` shared by all three coordinates.  That is the optimal linear
predictor only when the data variance is isotropic.  MetalCytoToxDB's is not
(per-axis variance 16.95 / 2.67 / 1.60 A^2), so the three optimal slopes have
opposite signs and the shared one is ~7x too small at t=0.5 — which is the
factor M4 was failing by.

These tests pin the two properties that make ``_bayes_ratio_per_axis`` a
correction rather than a redefinition:

1. on isotropic data the two agree, so nothing about the gate's meaning
   changed where the old reference was valid;
2. on anisotropic data they separate by a large factor, and the closed form
   matches a numerically fitted per-axis predictor.
"""

from __future__ import annotations

import math

import pytest
import torch

from molmetal.scripts.cfm_module_gates import (
    _bayes_ratio,
    _bayes_ratio_per_axis,
)

SIGMA = 2.485


def _isotropic(n: int = 3000, std: float = 2.0, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, generator=g) * std


def _anisotropic(n: int = 6000, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 3, generator=g)
    return z * torch.tensor([4.117, 1.634, 1.264])


@pytest.mark.parametrize("t", [0.05, 0.25, 0.50, 0.75, 0.95])
def test_on_isotropic_data_the_two_references_agree(t):
    """The correction must be a no-op where the old reference was valid.

    If this fails, ``_bayes_ratio_per_axis`` is a different quantity and the
    retraction of the old number would be unjustified.
    """
    x1 = _isotropic()
    shared = _bayes_ratio(x1, SIGMA, t, n_draws=48, seed=3)
    per_axis = _bayes_ratio_per_axis(x1, SIGMA, t)
    assert per_axis == pytest.approx(shared, rel=0.02), (
        f"t={t}: per-axis {per_axis:.4f} vs shared {shared:.4f} on isotropic "
        "data — the two must coincide when there is no anisotropy to lose"
    )


def test_on_anisotropic_data_the_shared_slope_is_much_too_small():
    """The failure M4 was scoring against, as an assertion.

    On the real per-axis spread the shared slope under-estimates the
    achievable magnitude by several times at the midpoint.  A gate that
    compares the field to it fails a field that is doing exactly what the
    data supports.
    """
    x1 = _anisotropic()
    shared = _bayes_ratio(x1, SIGMA, 0.50, n_draws=48, seed=3)
    per_axis = _bayes_ratio_per_axis(x1, SIGMA, 0.50)
    assert per_axis / shared > 3.0, (
        f"per-axis {per_axis:.4f} vs shared {shared:.4f} — expected the "
        "shared slope to collapse at the midpoint of an anisotropic cohort"
    )


def test_closed_form_matches_a_fitted_per_axis_predictor():
    """The analytic ratio must equal what a least-squares fit actually achieves."""
    x1 = _anisotropic()
    t = 0.50
    g = torch.Generator().manual_seed(7)
    n_draws = 48
    xs, ys = [], []
    for _ in range(n_draws):
        x0 = torch.randn(x1.shape, generator=g) * SIGMA
        xs.append((1.0 - t) * x0 + t * x1)
        ys.append(x1 - x0)
    xt = torch.cat(xs, 0).double()
    v = torch.cat(ys, 0).double()

    # Per-axis least squares with an intercept.
    a = torch.cat([torch.ones(xt.shape[0], 1, dtype=torch.float64), xt], dim=1)
    coef = torch.linalg.lstsq(a, v).solution
    pred = a @ coef
    fitted = float(pred.norm() / v.norm())

    closed = _bayes_ratio_per_axis(x1, SIGMA, t)
    assert fitted == pytest.approx(closed, rel=0.05), (
        f"closed form {closed:.4f} vs fitted {fitted:.4f}"
    )


def test_the_ratio_is_one_when_the_target_is_fully_recoverable():
    """``t = 0``: the field sees ``x0`` and ``v = x1 - x0`` has no noise left
    in the ``x0`` direction.  The ratio must be ``|x0| / |v|`` exactly, which
    is what the closed form returns.
    """
    x1 = _anisotropic()
    s2 = x1.double().var(dim=0, unbiased=False)
    sig2 = SIGMA ** 2
    expected = math.sqrt(3.0 * sig2 / float(3.0 * sig2 + s2.sum()))
    assert _bayes_ratio_per_axis(x1, SIGMA, 0.0) == pytest.approx(expected, rel=1e-6)


def test_a_zero_variance_axis_does_not_divide_by_zero():
    """A cohort with a degenerate axis must return a finite ratio, not inf/nan."""
    x1 = _anisotropic().clone()
    x1[:, 2] = 0.0
    r = _bayes_ratio_per_axis(x1, SIGMA, 0.50)
    assert math.isfinite(r) and 0.0 <= r <= 1.0
