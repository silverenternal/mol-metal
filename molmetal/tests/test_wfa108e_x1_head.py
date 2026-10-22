# SPDX-License-Identifier: Apache-2.0
"""Wf-A108e — the x1 (absolute-position) prediction target.

Why the target exists: Wf-A108d MEASURED that the velocity-trained field is 64%
worse than predicting zero at ``t = 0.05`` (MSE 14.781 against 9.008), while a
one-parameter Gaussian posterior mean reaches 3.605.  The conditional mean of
``x1 - x0`` has to go from ``-x_t`` at ``t -> 0`` to ``+x_t`` at ``t -> 1``, and
the field never reverses (endpoint cosine +0.736).  In x1 space the low-t target
is ``x1`` itself, i.e. the data mean, which does not require a reversal.

These tests pin what makes the mode safe to enable:

* the default (``velocity``) path publishes no new key and registers no new
  parameter, so every existing checkpoint still loads;
* ``x1_hat`` really is ``x_t + u`` and ``vel`` really is ``u / max(1-t, floor)``;
* the conversion is SE(3)-equivariant, and the floor bounds the sampler step at
  ``t -> 1``;
* asking for ``x1`` from a velocity field (or the reverse) fails loudly instead
  of silently training the wrong operand.
"""

from __future__ import annotations

import pytest
import torch

from molmetal.adapters.flow_matching_lipman import EGNNVelocityField


def _graph(b: int = 2, n: int = 7, scale: float = 2.0, seed: int = 0):
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _block_diag_radius_edges,
    )

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, n, 3, generator=g) * scale
    row = [78, 6, 6, 7, 7, 8, 8, 6, 6, 15, 1, 1]
    z = torch.tensor([row[:n] if i == 0 else [44] + row[1:n] for i in range(b)])
    mask = torch.ones(b, n, dtype=torch.bool)
    edge_index, edge_mask = _block_diag_radius_edges(x, mask, 6.0)
    return x, z, edge_index, edge_mask


def _field(**kw) -> EGNNVelocityField:
    f = EGNNVelocityField(
        hidden_dim=16,
        n_layers=2,
        velocity_head="vector_channels",
        vector_channels=4,
        **kw,
    )
    # A fresh channel gate is small but non-zero; scale it up so the assertions
    # below are about the conversion, not about floating-point noise.
    torch.manual_seed(0)
    with torch.no_grad():
        torch.nn.init.normal_(f.vc_head.channel_gate.weight, std=0.5)
    f.eval()
    return f


# ---------------------------------------------------------------------------
# 1. The default path is untouched
# ---------------------------------------------------------------------------
def test_default_target_publishes_no_x1_key():
    x, z, edge_index, edge_mask = _graph()
    t = torch.full((x.shape[0],), 0.5)
    f = _field()
    assert f.prediction_target == "velocity"
    with torch.no_grad():
        out = f(x, z, edge_index, t, edge_mask=edge_mask)
    assert "x1_hat" not in out, sorted(out)
    assert set(out) == {"vel", "atom_logits", "h"}


def test_default_target_registers_no_extra_parameter():
    a = EGNNVelocityField(
        hidden_dim=16, n_layers=2, velocity_head="vector_channels",
        vector_channels=4,
    )
    b = EGNNVelocityField(
        hidden_dim=16, n_layers=2, velocity_head="vector_channels",
        vector_channels=4, prediction_target="x1",
    )
    assert list(a.state_dict()) == list(b.state_dict())


# ---------------------------------------------------------------------------
# 2. The conversion itself
# ---------------------------------------------------------------------------
def test_x1_hat_is_x_t_plus_the_displacement():
    x, z, edge_index, edge_mask = _graph()
    t = torch.full((x.shape[0],), 0.25)
    f = _field(prediction_target="x1")
    with torch.no_grad():
        out = f(x, z, edge_index, t, edge_mask=edge_mask)
    u = out["vel"] * (1.0 - 0.25)  # invert the division
    assert torch.allclose(out["x1_hat"], x + u, atol=1e-6)


def test_velocity_is_the_displacement_over_one_minus_t():
    x, z, edge_index, edge_mask = _graph()
    f = _field(prediction_target="x1")
    f_v = _field(prediction_target="velocity")
    for tv in (0.1, 0.5, 0.9):
        t = torch.full((x.shape[0],), tv)
        with torch.no_grad():
            out_x1 = f(x, z, edge_index, t, edge_mask=edge_mask)
            out_v = f_v(x, z, edge_index, t, edge_mask=edge_mask)
        # Both fields share the same weights, so the x1 mode's velocity must be
        # exactly the velocity mode's output divided by (1 - t).
        assert torch.allclose(
            out_x1["vel"], out_v["vel"] / (1.0 - tv), atol=1e-5
        )


def test_denominator_floor_bounds_the_step_at_large_t():
    x, z, edge_index, edge_mask = _graph()
    f = _field(prediction_target="x1", x1_denominator_floor=0.05)
    t = torch.full((x.shape[0],), 0.999)
    with torch.no_grad():
        out = f(x, z, edge_index, t, edge_mask=edge_mask)
    u = out["x1_hat"] - x
    assert torch.allclose(out["vel"], u / 0.05, atol=1e-5)
    # Without the floor this would be u / 0.001, a 50x larger step.
    assert float(out["vel"].abs().max()) < 50.0 * float(u.abs().max())


# ---------------------------------------------------------------------------
# 3. Equivariance
# ---------------------------------------------------------------------------
def test_x1_hat_is_se3_equivariant():
    x, z, edge_index, edge_mask = _graph()
    t = torch.full((x.shape[0],), 0.3)
    f = _field(prediction_target="x1")

    g = torch.Generator().manual_seed(3)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([0.7, -1.3, 2.1])

    with torch.no_grad():
        base = f(x, z, edge_index, t, edge_mask=edge_mask)["x1_hat"]
        rot = f(x @ q.T + shift, z, edge_index, t, edge_mask=edge_mask)[
            "x1_hat"
        ]
    assert torch.allclose(rot, base @ q.T + shift, atol=2e-5)


# ---------------------------------------------------------------------------
# 4. Misuse fails loudly
# ---------------------------------------------------------------------------
def test_fm_x1_refuses_a_velocity_field(monkeypatch):
    """The guard has to fire: returning ``x_t + velocity`` would train a
    quantity nobody asked for, and the loss would look plausible while doing
    it."""
    import sys

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        build_joint_model,
        parse_args,
    )

    monkeypatch.setattr(sys, "argv", ["train_cfm_yuelbond_wasserstein.py"])
    ns = parse_args()
    ns.hidden_dim = 16
    ns.n_layers = 2
    ns.use_fm_objective = True
    ns.use_egnn_velocity = True
    ns.velocity_head = "vector_channels"
    ns.vector_channels = 4
    ns.prediction_target = "velocity"
    ns.use_tmqm_init = False
    model = build_joint_model(ns)
    x = torch.randn(7, 3)
    z = torch.tensor([78, 6, 6, 7, 7, 8, 8])
    with pytest.raises(RuntimeError, match="prediction-target x1"):
        model.fm_x1(x, torch.full((7,), 0.5), z, cloud_sizes=[7])


def test_invalid_prediction_target_is_rejected():
    with pytest.raises(ValueError, match="prediction_target"):
        EGNNVelocityField(hidden_dim=8, n_layers=1, prediction_target="x0")


def test_invalid_denominator_floor_is_rejected():
    for bad in (0.0, -0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="x1_denominator_floor"):
            EGNNVelocityField(
                hidden_dim=8, n_layers=1, prediction_target="x1",
                x1_denominator_floor=bad,
            )
