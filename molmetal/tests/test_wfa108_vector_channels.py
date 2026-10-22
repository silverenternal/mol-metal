# SPDX-License-Identifier: Apache-2.0
"""Wf-A108 — vector-channel coordinate head.

The head exists because Wf-A107h MEASURED that the radial head cannot fit one
8-cloud batch (MSE 10.529 against a 10.652 ``v = 0`` baseline and a 5.783
radial-only ceiling).  These tests pin the four properties that claim rests on:

* the default ``radial`` head is untouched (no new parameters, same formula);
* the new head is SE(3)-equivariant;
* the new head can represent a *tangential* target that the radial head
  provably cannot;
* the metal-frame channels are actually metal-relative.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from molmetal.adapters.flow_matching_lipman import EGNNVelocityField
from molmetal.adapters.flow_matching_lipman.vector_channels import (
    METAL_ATOMIC_NUMBERS,
    VectorChannelLayer,
    build_metal_mask,
    initial_vector_channels,
)


def _graph(b: int = 2, n: int = 7, scale: float = 2.0, seed: int = 0):
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _block_diag_radius_edges,
    )

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, n, 3, generator=g) * scale
    # Pt / Ru centre + a light-atom shell; pad the rest with carbon so the
    # tensor is (b, n) for any n.
    row = [78, 6, 6, 7, 7, 8, 8, 6, 6, 15, 1, 1]
    z = torch.tensor([row[:n] if i == 0 else [44] + row[1:n] for i in range(b)])
    mask = torch.ones(b, n, dtype=torch.bool)
    edge_index, edge_mask = _block_diag_radius_edges(x, mask, 6.0)
    return x, z, edge_index, edge_mask, mask


def _random_rotation(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


# ---------------------------------------------------------------------------
# 1. The default path is untouched
# ---------------------------------------------------------------------------
def test_default_head_registers_no_vector_channel_parameters():
    f = EGNNVelocityField(hidden_dim=16, n_layers=2)
    keys = list(f.state_dict())
    assert not any(k.startswith("vc_") for k in keys), keys
    assert len(f.vc_layers) == 0
    assert f.velocity_head == "radial"
    assert isinstance(f.vel_head, torch.nn.Linear)


def test_default_head_matches_the_legacy_formula():
    """``vel = tanh(vel_head(h)) * vel_scale * (x - centroid) + last_v``."""
    x, z, edge_index, edge_mask, _ = _graph()
    b, n = x.shape[:2]
    t = torch.full((b,), 0.5)
    f = EGNNVelocityField(hidden_dim=16, n_layers=2)
    f.eval()
    with torch.no_grad():
        # Make both terms non-zero: a random gate and a random last layer.
        torch.nn.init.normal_(f.vel_head.weight, std=0.3)
        torch.nn.init.normal_(f.layers[-1].msg_vector_head.weight, std=0.3)

        t_per_atom = f.time_mlp(t.unsqueeze(-1)).unsqueeze(1).expand(b, n, -1)
        h = f.atom_embed(z) + t_per_atom
        last_v = torch.zeros_like(x)
        for layer in f.layers:
            h, last_v = layer(
                h, x, edge_index, cond_per_node=t_per_atom, edge_mask=edge_mask
            )
        expected = (
            torch.tanh(f.vel_head(h)) * f.vel_scale
            * (x - x.mean(dim=1, keepdim=True))
            + last_v
        )
        out = f(x, z, edge_index, t, edge_mask=edge_mask)
    assert float(expected.abs().max()) > 1e-3, "vacuous formula test"
    assert torch.allclose(out["vel"], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. The new head is SE(3)-equivariant
# ---------------------------------------------------------------------------
def test_vector_head_is_se3_equivariant():
    x, z, edge_index, edge_mask, mask = _graph(b=3, n=9, seed=3)
    f = EGNNVelocityField(
        hidden_dim=16, n_layers=2, velocity_head="vector_channels",
        vector_channels=8, metal_frame_channels=True,
    )
    f.eval()
    torch.nn.init.normal_(f.vc_head.channel_gate.weight, std=0.5)

    q = _random_rotation()
    shift = torch.tensor([3.0, -2.0, 1.5])
    x2 = x @ q.T + shift
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _block_diag_radius_edges,
    )

    ei2, em2 = _block_diag_radius_edges(x2, mask, 6.0)
    with torch.no_grad():
        v1 = f(x, z, edge_index, torch.full((3,), 0.5), edge_mask=edge_mask)["vel"]
        v2 = f(x2, z, ei2, torch.full((3,), 0.5), edge_mask=em2)["vel"]
    assert float(v1.norm(dim=-1).mean()) > 1e-3, "vacuous equivariance test"
    assert torch.allclose(v2, v1 @ q.T, atol=2e-5)


def test_vector_head_starts_at_exactly_zero_velocity():
    x, z, edge_index, edge_mask, _ = _graph()
    f = EGNNVelocityField(
        hidden_dim=16, n_layers=2, velocity_head="vector_channels"
    )
    f.eval()
    with torch.no_grad():
        out = f(x, z, edge_index, torch.full((2,), 0.5), edge_mask=edge_mask)
    assert float(out["vel"].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 3. Expressivity: a tangential target the radial head cannot reach
# ---------------------------------------------------------------------------
def _radial_best_mse(x0: torch.Tensor, x1: torch.Tensor, n: int) -> float:
    xt = 0.5 * x0 + 0.5 * x1
    tgt = x1 - x0
    rel = xt - xt.mean(dim=0, keepdim=True)
    unit = rel / rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    coeff = (tgt * unit).sum(dim=-1, keepdim=True)
    return float(F.mse_loss(coeff * unit, tgt))


def _tangential_pair(n: int = 8, seed: int = 5):
    """A cloud whose target velocity is orthogonal to every radial ray."""
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(n, 3, generator=g)
    xt = p - p.mean(dim=0, keepdim=True)
    unit = xt / xt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ref = torch.tensor([0.0, 0.0, 1.0]).expand_as(unit)
    perp = torch.cross(unit, ref, dim=-1)
    bad = perp.norm(dim=-1) < 1e-3
    perp[bad] = torch.tensor([1.0, 0.0, 0.0])
    perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    tgt = 0.8 * perp
    return xt - 0.5 * tgt, xt + 0.5 * tgt


def test_radial_ceiling_is_positive_on_a_tangential_target():
    x0, x1 = _tangential_pair()
    baseline = float(F.mse_loss(torch.zeros_like(x1 - x0), x1 - x0))
    ceiling = _radial_best_mse(x0, x1, x0.shape[0])
    assert ceiling == pytest.approx(baseline, rel=1e-4)


def test_vector_head_fits_a_tangential_target():
    """The gate Wf-A107h failed: reach well below the radial ceiling."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _block_diag_radius_edges,
    )

    x0, x1 = _tangential_pair(n=8, seed=5)
    tgt = x1 - x0
    xt = 0.5 * x0 + 0.5 * x1
    x = xt.unsqueeze(0)
    z = torch.tensor([[6] * 8])
    mask = torch.ones(1, 8, dtype=torch.bool)
    edge_index, edge_mask = _block_diag_radius_edges(x, mask, 6.0)

    baseline = float(F.mse_loss(torch.zeros_like(tgt), tgt))
    ceiling = _radial_best_mse(x0, x1, 8)
    assert ceiling > 0.5 * baseline, "test fixture is not tangential enough"

    torch.manual_seed(0)
    f = EGNNVelocityField(
        hidden_dim=32, n_layers=2, velocity_head="vector_channels",
        vector_channels=8,
    )
    f.train()
    opt = torch.optim.AdamW(f.parameters(), lr=5e-3)
    for _ in range(200):
        out = f(x, z, edge_index, torch.full((1,), 0.5), edge_mask=edge_mask)
        loss = F.mse_loss(out["vel"][0], tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = float(
            F.mse_loss(
                f(x, z, edge_index, torch.full((1,), 0.5), edge_mask=edge_mask)[
                    "vel"
                ][0],
                tgt,
            )
        )
    assert final < 0.25 * ceiling, (
        f"final={final:.4f} ceiling={ceiling:.4f} baseline={baseline:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Metal-frame channels
# ---------------------------------------------------------------------------
def test_build_metal_mask_marks_platinum_only():
    z = torch.tensor([[78, 6, 6, 44, 7, 15]])
    mask = build_metal_mask(z)
    assert mask.tolist() == [[True, False, False, True, False, False]]
    assert 78 in METAL_ATOMIC_NUMBERS and 6 not in METAL_ATOMIC_NUMBERS


def test_metal_frame_channel_is_metal_relative():
    pos = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [9.0, 9.0, 9.0]]]
    )
    z = torch.tensor([[78, 7, 7, 6]])
    mm = build_metal_mask(z)
    vec = initial_vector_channels(
        pos, 3, metal_mask=mm, metal_frame_channels=True
    )
    # Channel 1 is x - x_metal (the metal is at the origin here).
    assert torch.allclose(vec[0, :, 1, :], pos[0])
    # Channel 2 is x - donor_centroid.  Donors are the two N at 2.0 A;
    # their centroid is (1, 1, 0), so the carbon at (9,9,9) sits at
    # (8, 8, 9) and the metal itself at (-1, -1, 0).
    assert torch.allclose(vec[0, 0, 2, :], torch.tensor([-1.0, -1.0, 0.0]))
    assert torch.allclose(vec[0, 3, 2, :], torch.tensor([8.0, 8.0, 9.0]))


def test_metal_frame_channels_are_zero_without_a_metal():
    pos = torch.randn(1, 5, 3)
    z = torch.tensor([[6, 6, 7, 8, 1]])
    vec = initial_vector_channels(
        pos, 3, metal_mask=build_metal_mask(z), metal_frame_channels=True
    )
    assert float(vec[0, :, 1:, :].abs().max()) == 0.0


def test_padded_atoms_keep_zero_channels():
    layer = VectorChannelLayer(hidden_dim=8, n_channels=4)
    b, n, h = 1, 5, 8
    h_node = torch.randn(b, n, h)
    pos = torch.randn(b, n, 3)
    vec = torch.randn(b, n, 4, 3)
    vec[:, 3:, :, :] = 0.0  # atoms 3,4 are padding
    edge_index = torch.tensor([[[0, 1, 2, 3, 4], [1, 2, 0, 0, 0]]])
    edge_mask = torch.tensor([[True, True, True, False, False]])
    h_out, v_out = layer(
        h_node, pos, vec, edge_index, edge_mask=edge_mask
    )
    assert float(v_out.detach()[:, 3:, :, :].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 5. Argument validation
# ---------------------------------------------------------------------------
def test_unknown_velocity_head_raises():
    with pytest.raises(ValueError, match="velocity_head"):
        EGNNVelocityField(hidden_dim=8, n_layers=1, velocity_head="magic")


def test_vector_channel_count_is_respected():
    f = EGNNVelocityField(
        hidden_dim=8, n_layers=1, velocity_head="vector_channels",
        vector_channels=5,
    )
    assert f.vc_layers[0].n_channels == 5
    assert f.vc_head.channel_gate.out_features == 5
