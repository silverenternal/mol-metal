# SPDX-License-Identifier: Apache-2.0
"""Wf-A108j — the closed-form output-head initialiser and the magnitude switch.

WHY THESE TESTS EXIST
---------------------
``cfm_module_gates`` M4 measured the epoch-4 vector_channels checkpoint at
``||v|| / ||v_target|| = 0.021``: the field emitted 2% of the magnitude the
target asked for, and ``fm_term`` (12.17) was worse than predicting zero
(about 9.4).  The mechanism was two multiplying magnitude controls, both
driven to zero: a zero-init gate and a trainable per-channel ``norm_gain``
that had fallen 1.0 -> 0.0017.

The fix adds two opt-in surfaces.  Both must be inert at their defaults,
because every existing checkpoint has to rebuild and run bit-exactly.  That
inertness is the first thing tested here; the second is that the fix actually
does what it claims on a fresh field.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from molmetal.adapters.flow_matching_lipman import EGNNVelocityField  # noqa: E402
from molmetal.scripts.train_cfm_yuelbond_wasserstein import (  # noqa: E402
    CFMJointModel,
)


def _field(**kw) -> EGNNVelocityField:
    torch.manual_seed(0)
    return EGNNVelocityField(hidden_dim=32, n_layers=2, max_atomic_number=100, **kw)


def _model(**kw) -> CFMJointModel:
    """The real entry point for the initialiser: it lives on the joint model
    because the padded-batch loop needs ``cutoff`` and the cloud sizes."""
    torch.manual_seed(0)
    m = CFMJointModel(
        hidden_dim=32,
        n_layers=2,
        fm_objective=True,
        use_egnn_velocity=True,
        velocity_head="vector_channels",
        **kw,
    )
    return m.eval()


def _batch(n_clouds: int = 4, n_atoms: int = 9, sigma: float = 2.4):
    g = torch.Generator().manual_seed(7)
    x1 = torch.randn(n_clouds, n_atoms, 3, generator=g) * 1.6
    x0 = torch.randn(n_clouds, n_atoms, 3, generator=g) * sigma
    z = torch.randint(1, 20, (n_clouds, n_atoms), generator=g)
    z[:, 0] = 78  # a metal centre, so metal_mask has something to see
    return x0, x1, z


def _edges(x: torch.Tensor, z: torch.Tensor, cutoff: float = 3.5):
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _block_diag_radius_edges,
    )

    mask = z > 0
    return _block_diag_radius_edges(x, mask, cutoff)


# ---------------------------------------------------------------------------
# 1. defaults are inert
# ---------------------------------------------------------------------------
def test_default_build_has_no_bias_and_no_t_gate():
    vf = _field(velocity_head="vector_channels")
    assert vf.vc_head.channel_gate.bias is None
    assert vf.vc_head.t_gate is None
    assert isinstance(vf.vc_layers[0].norm_gain, torch.nn.Parameter)
    assert vf.vector_channel_norm_gain == "learnable"
    assert vf.velocity_head_init == "zero"


def test_default_state_dict_keys_unchanged():
    """The pre-Wf-A108j key set, exactly.  A new key here would break every
    checkpoint's ``load_state_dict(strict=True)``."""
    vf = _field(velocity_head="vector_channels")
    keys = {k for k in vf.state_dict() if k.startswith("vc_")}
    assert keys == {
        "vc_layers.0.norm_gain",
        "vc_layers.0.channel_mix.weight",
        "vc_layers.0.gate_head.weight",
        "vc_layers.0.gate_head.bias",
        "vc_layers.0.phi_head.weight",
        "vc_layers.0.phi_head.bias",
        "vc_layers.0.msg_scalar_head.weight",
        "vc_layers.0.msg_scalar_head.bias",
        "vc_layers.0.edge_mlp.0.weight",
        "vc_layers.0.edge_mlp.0.bias",
        "vc_layers.0.edge_mlp.2.weight",
        "vc_layers.0.edge_mlp.2.bias",
        "vc_layers.0.update_mlp.0.weight",
        "vc_layers.0.update_mlp.0.bias",
        "vc_layers.0.update_mlp.2.weight",
        "vc_layers.0.update_mlp.2.bias",
        "vc_layers.1.norm_gain",
        "vc_layers.1.channel_mix.weight",
        "vc_layers.1.gate_head.weight",
        "vc_layers.1.gate_head.bias",
        "vc_layers.1.phi_head.weight",
        "vc_layers.1.phi_head.bias",
        "vc_layers.1.msg_scalar_head.weight",
        "vc_layers.1.msg_scalar_head.bias",
        "vc_layers.1.edge_mlp.0.weight",
        "vc_layers.1.edge_mlp.0.bias",
        "vc_layers.1.edge_mlp.2.weight",
        "vc_layers.1.edge_mlp.2.bias",
        "vc_layers.1.update_mlp.0.weight",
        "vc_layers.1.update_mlp.0.bias",
        "vc_layers.1.update_mlp.2.weight",
        "vc_layers.1.update_mlp.2.bias",
        "vc_head.channel_gate.weight",
    }


def test_default_forward_is_deterministic_and_zero_at_init():
    """A zero-init gate must still emit exactly zero: that contract is what
    makes the legacy behaviour reproducible, and it is also the reason the
    field needs 125 steps before it emits anything at all."""
    vf = _field(velocity_head="vector_channels").eval()
    x0, x1, z = _batch()
    xt = 0.5 * x0 + 0.5 * x1
    ei, em = _edges(xt, z)
    t = torch.full((z.shape[0],), 0.5)
    with torch.no_grad():
        out = vf(xt, z, ei, t, edge_mask=em)["vel"]
    assert float(out.abs().max()) == 0.0


def test_lsq_init_is_refused_at_default():
    vf = _field(velocity_head="vector_channels").eval()
    x0, x1, z = _batch()
    rep = vf.fit_output_head_lsq(
        (0.5 * x0 + 0.5 * x1),
        z,
        *_edges(0.5 * x0 + 0.5 * x1, z),
        torch.full((z.shape[0],), 0.5),
        x1 - x0,
    )
    assert rep["w"] is None
    assert "velocity_head_init" in str(rep["reason"])


# ---------------------------------------------------------------------------
# 2. the switches do what they say
# ---------------------------------------------------------------------------
def test_fixed_gain_removes_the_parameter_but_keeps_the_key():
    vf = _field(velocity_head="vector_channels", vector_channel_norm_gain="fixed")
    assert not isinstance(vf.vc_layers[0].norm_gain, torch.nn.Parameter)
    assert "vc_layers.0.norm_gain" in vf.state_dict()
    assert float(vf.vc_layers[0].norm_gain.mean()) == 1.0
    names = {n for n, _ in vf.named_parameters()}
    assert "vc_layers.0.norm_gain" not in names


def test_lsq_build_adds_bias_and_t_gate_only_when_asked():
    vf = _field(
        velocity_head="vector_channels",
        velocity_head_init="lsq",
        t_embedding_dim=16,
    )
    assert vf.vc_head.channel_gate.bias is not None
    assert vf.vc_head.t_gate is not None
    assert vf.vc_head.t_gate.weight.shape == (8, 16)
    # zero-init: the extra surface must not change the initial function
    x0, x1, z = _batch()
    xt = 0.5 * x0 + 0.5 * x1
    ei, em = _edges(xt, z)
    with torch.no_grad():
        out = vf(xt, z, ei, torch.full((z.shape[0],), 0.5), edge_mask=em)["vel"]
    assert float(out.abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 3. the initialiser improves the initial field
# ---------------------------------------------------------------------------
def _lsq_field():
    m = _model(
        vector_channel_norm_gain="fixed",
        velocity_head_init="lsq",
        t_embedding_dim=16,
    )
    x0, x1, z = _batch()
    sizes = [int(z.shape[1])] * int(z.shape[0])
    rep = m.init_velocity_head_lsq(x0.reshape(-1, 3), x1.reshape(-1, 3), z.reshape(-1), sizes, n_knots=5)
    assert rep["applied"] is True
    return m, rep, x0, x1, z


def test_lsq_init_beats_the_zero_baseline_at_every_knot():
    _, rep, x0, x1, _ = _lsq_field()
    for row in rep["knots"]:
        assert row["mse"] < row["mse_zero"], row


def test_lsq_init_sets_a_magnitude_that_is_no_longer_a_few_percent():
    """M4's finding was ``ratio = 0.021`` at EVERY t — the field could not emit
    magnitude even where the target is nearly deterministic.  At ``t = 0.02``
    the conditional mean is a large fraction of the target, so the initialiser
    has to land in that range or it is not addressing the measurement.

    This deliberately does not assert a floor at every knot: with ``x0`` and
    ``x1`` at different scales the conditional mean passes through zero around
    ``t = sigma0^2 / (sigma0^2 + sigma1^2)``, and a fit that emitted magnitude
    there would be the opposite failure.  The knot-by-knot shape is pinned by
    ``test_the_midpoint_is_shrunk_because_nothing_is_predictable_there``.
    """
    _, rep, _, _, _ = _lsq_field()
    rows = {round(row["t"], 2): row["ratio"] for row in rep["knots"]}
    assert max(rows.values()) > 0.2, rows
    assert rows[0.02] > 0.2, rows


def test_lsq_init_makes_the_field_depend_on_t():
    """M6's failure mode was a field flat in ``t`` (2.3% relative change across
    the whole sweep).  The gate is fitted as a linear function of the
    sinusoidal embedding, so the endpoints must now differ by more than 10%."""
    m, _, x0, x1, z = _lsq_field()
    vf = m.vel_field
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _block_diag_radius_edges,
    )

    x1_flat = x1.reshape(-1, 3)
    x0_flat = x0.reshape(-1, 3)
    z_flat = z.reshape(-1)
    mask = (z_flat > 0).unsqueeze(0)
    outs = []
    for tv in (0.02, 0.98):
        xt = ((1.0 - tv) * x0_flat + tv * x1_flat).unsqueeze(0)
        ei, em = _block_diag_radius_edges(xt, mask, m.cutoff)
        with torch.no_grad():
            outs.append(
                vf(xt, z_flat.unsqueeze(0), ei, torch.full((1,), tv), edge_mask=em)[
                    "vel"
                ]
            )
    a, b = outs
    rel = float((a - b).norm()) / max(0.5 * (float(a.norm()) + float(b.norm())), 1e-12)
    assert rel > 0.1, rel


def test_lsq_init_leaves_the_head_parameters_finite():
    m, rep, _, _, _ = _lsq_field()
    for name, p in m.vel_field.vc_head.named_parameters():
        assert torch.isfinite(p).all(), name
    assert rep["fit"]["fit_residual_rms"] < rep["fit"]["gate_weight_rms"] * 2.0


# ---------------------------------------------------------------------------
# 4. the fit is an estimate of the conditional mean, not of the draw
# ---------------------------------------------------------------------------
def _equal_variance_field(sigma: float = 1.6):
    """Equal variances on both ends of the interpolant.

    That is the regime M4 caught: with ``x0`` and ``x1`` at the same scale the
    conditional mean of ``x1 - x0`` given ``x_t`` vanishes at ``t = 0.5``, so
    any magnitude the fit emits there is fitted noise.  The metalcytotox
    cohort is close to this (prior 2.485 against a data std of 1.572).
    """
    m = _model(
        vector_channel_norm_gain="fixed",
        velocity_head_init="lsq",
        t_embedding_dim=16,
    )
    x0, x1, z = _batch(n_clouds=8, n_atoms=12, sigma=sigma)
    rep = m.init_velocity_head_lsq(
        x0.reshape(-1, 3),
        x1.reshape(-1, 3),
        z.reshape(-1),
        [12] * 8,
        n_knots=5,
    )
    assert rep["applied"] is True
    return rep


def test_the_ridge_is_selected_and_reported_per_knot():
    """The defect was an effectively unregularised solve (ridge 1e-6 * scale).
    Selection has to be visible in the report, or a caller cannot tell a
    regularised fit from the old one."""
    rep = _equal_variance_field()
    for row in rep["knots"]:
        assert "ridge_relative" in row, row
        assert row["gcv"] == row["gcv"], row  # not NaN
    assert max(row["ridge_relative"] for row in rep["knots"]) > 1e-6, [
        row["ridge_relative"] for row in rep["knots"]
    ]


def test_the_midpoint_is_shrunk_because_nothing_is_predictable_there():
    """At ``t = 0.5`` with matched variances the conditional mean is zero, so a
    fit that emits magnitude there is emitting noise.  The unregularised fit
    measured 5.12x the achievable magnitude on the real batch; the selected
    ridge has to hold the in-sample ratio well below the ends."""
    rep = _equal_variance_field()
    mid = [r for r in rep["knots"] if abs(r["t"] - 0.5) < 0.06][0]
    ends = [r["ratio"] for r in rep["knots"] if abs(r["t"] - 0.5) >= 0.06]
    assert mid["ratio"] < 0.5 * max(ends), (mid, ends)
