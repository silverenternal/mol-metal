# SPDX-License-Identifier: Apache-2.0
"""Wf-A107h — the radial ceiling used to judge the velocity field's capacity.

WHY THIS EXISTS
---------------
``EGNNVelocityField.forward`` composes

    vel = tanh(vel_head(h)) * vel_scale * (x - centroid) + last_v

so the dominant term can only move an atom along the ray from the cloud
centroid.  ``wfa107h_velocity_capacity.py`` compares the loss the field
actually reaches with the best loss ANY radial-only field could reach on the
same batch.  If the helper is wrong, that comparison is meaningless, so the
three geometric cases are pinned here: a purely radial target must be fitted
exactly, a purely tangential one must gain nothing, and the ceiling must never
beat the ``v = 0`` baseline.

RED CHECK: return ``0.0`` from ``_radial_best_mse`` and
``test_tangential_target_gains_nothing`` fails; project onto the target
instead of onto the ray and ``test_radial_target_is_fitted_exactly`` fails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "molmetal" / "scripts" / "wfa107h_velocity_capacity.py"


def _mod():
    if "_wfa107h_cap" in sys.modules:
        return sys.modules["_wfa107h_cap"]
    spec = importlib.util.spec_from_file_location("_wfa107h_cap", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_wfa107h_cap"] = m
    spec.loader.exec_module(m)
    return m


def _cloud(n: int = 6, seed: int = 0) -> torch.Tensor:
    """A cloud with ZERO centroid, so `rel = x - centroid` is just `x`."""
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(n, 3, generator=g)
    return p - p.mean(dim=0, keepdim=True)


def _split(xt: torch.Tensor, tgt: torch.Tensor):
    """(x0, x1) whose t=0.5 interpolant is `xt` and whose target is `tgt`."""
    return xt - 0.5 * tgt, xt + 0.5 * tgt


def test_radial_target_is_fitted_exactly():
    """target parallel to the ray from the centroid -> the ceiling is 0."""
    t = _mod()
    xt = _cloud()
    x0, x1 = _split(xt, 0.5 * xt)
    assert t._radial_best_mse(x0, x1, [xt.shape[0]]) == pytest.approx(0.0, abs=1e-6)


def test_tangential_target_gains_nothing():
    """A target orthogonal to every ray cannot be reduced by a radial field."""
    t = _mod()
    xt = _cloud(seed=3)
    unit = xt / xt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    z = torch.tensor([0.0, 0.0, 1.0]).expand_as(unit)
    perp = torch.cross(unit, z, dim=-1)
    bad = perp.norm(dim=-1) < 1e-3
    perp[bad] = torch.tensor([1.0, 0.0, 0.0])
    perp = perp / perp.norm(dim=-1, keepdim=True)
    x0, x1 = _split(xt, 0.4 * perp)
    ceiling = t._radial_best_mse(x0, x1, [xt.shape[0]])
    baseline = float(F.mse_loss(torch.zeros_like(x1 - x0), x1 - x0))
    assert ceiling == pytest.approx(baseline, rel=1e-4)


def test_ceiling_never_exceeds_the_zero_baseline():
    t = _mod()
    for seed in (0, 1, 2):
        x0 = _cloud(seed=seed)
        g = torch.Generator().manual_seed(seed + 100)
        x1 = torch.randn(x0.shape, generator=g)
        sizes = [x0.shape[1]]
        baseline = float(F.mse_loss(torch.zeros_like(x1 - x0), x1 - x0))
        assert t._radial_best_mse(x0, x1, sizes) <= baseline + 1e-6


def test_multi_cloud_sizes_are_honoured():
    """Two clouds must be centred independently, not over the concatenation."""
    t = _mod()
    a = _cloud(4, seed=7)
    b = _cloud(4, seed=8) + torch.tensor([50.0, 0.0, 0.0])
    xt = torch.cat([a, b], dim=0)
    # target radial about EACH cloud's own centroid
    tgt = torch.cat([0.5 * a, 0.5 * b - torch.tensor([25.0, 0.0, 0.0])], dim=0)
    x0, x1 = _split(xt, tgt)
    two = t._radial_best_mse(x0, x1, [4, 4])
    one = t._radial_best_mse(x0, x1, [8])
    assert two == pytest.approx(0.0, abs=1e-5)
    assert one > two
