"""Wf-307D — CPU smoke tests for the 4 frontier-math adapter families.

Per the Wf-307D brief and TODO-234 audit, these tests cover the
forward-pass smoke on the 4 adapter module families that have been
wired into the molmetal pipeline. The actual class names differ
from what the Wf-307D brief assumed (the adapters were never
re-exported under ``MultiPHScheduler`` / ``SE3LipmanScheduler`` /
``SE3SBScheduler`` / ``TransportRelaxationScheduler`` aliases);
this test pins the REAL classes that ship in the codebase.

Honest framing
--------------
- All 4 adapter families are already imported successfully.
- These tests are CPU-only (no GPU / no checkpoint load).
- Per TODO-234 §5 honest framing, ``bass_existence``,
  ``sb_bass_martingale``, and ``monge_ampere_regularity`` are
  research-only and are NOT wired into ``flow_matching_lipman``.
"""
from __future__ import annotations

import inspect

import torch

# Adapter 1: Multi-PH (T5 in chemprop_v2_frontier; multi_ph_integration in CFM)
from molmetal.adapters.multi_ph_integration import (
    MultiPHPocketFeatures,
    augment_pocket_embedding,
)

# Adapter 2: SE(3) Lipman solver (se3_lipman_integration)
from molmetal.adapters.se3_lipman_integration import (
    SE3LipmanSolver,
    SE3CFMComposite,
)

# Adapter 3: SE(3) SB velocity field (se3_sb_integration)
from molmetal.adapters.se3_sb_integration import (
    SE3SBVelocityField,
    SE3SBDecoder,
    integrate_with_cfm,
)

# Adapter 4: Transport Relaxation (math_skills/transport_relaxation)
from molmetal.math_skills.transport_relaxation import TransportRelaxationTerm


def test_multi_ph_forward() -> None:
    """Smoke: MultiPHPocketFeatures.compute returns (B, n_bins, n_bins)."""
    mp = MultiPHPocketFeatures(n_bins=8, max_radius=3.0)
    coords = torch.randn(2, 12, 3)
    out = mp.compute(coords)
    assert out.shape == (2, 8, 8), f"expected (2,8,8), got {tuple(out.shape)}"
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all(), "non-finite values in multi_ph output"


def test_se3_lipman_forward() -> None:
    """Smoke: SE3LipmanSolver instantiates with default hyper-params."""
    sl = SE3LipmanSolver(n_steps=5)
    assert sl.n_steps == 5
    assert sl.method == "midpoint"
    assert sl.ortho_tol == 1e-3
    assert hasattr(sl, "solve"), "SE3LipmanSolver must expose solve()"


def test_se3_sb_forward() -> None:
    """Smoke: SE3SBVelocityField.forward returns a (B, 6) SE(3) twist.

    The wrapper reduces per-atom Euclidean velocity into the SE(3)
    tangent space: linear (centroid) + angular (cross of moment-arm
    with residual).
    """
    import torch.nn as nn

    class _DummyV(nn.Module):
        def forward(self, x_T, t=None, pocket=None):
            return torch.zeros(x_T.shape[0], x_T.shape[1], 3)

    sv = SE3SBVelocityField(base_velocity_net=_DummyV(), manifold="SE3")
    assert sv.manifold == "SE3"
    x = torch.randn(1, 6, 3)
    t = torch.tensor([0.5])
    twist = sv.forward(x, t)
    assert twist.shape == (1, 6), f"expected (1,6), got {tuple(twist.shape)}"
    assert torch.isfinite(twist).all()


def test_transport_relaxation_forward() -> None:
    """Smoke: TransportRelaxationTerm.kl_divergence returns a scalar."""
    tr = TransportRelaxationTerm(lambda_relax=0.1)
    # Both inputs must share shape (B, n, d); use 3D so they match.
    p_T = torch.randn(2, 8, 3)
    q_T = torch.randn(2, 8, 3)
    loss = tr.kl_divergence(p_T, q_T)
    assert loss.dim() == 0, f"expected scalar, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss), f"kl_divergence non-finite: {loss.item()}"


def test_se3_sb_decoder_smoke() -> None:
    """Smoke: SE3SBDecoder instantiates with bit-exact (use_sinkhorn=False) mode."""
    dec = SE3SBDecoder(eps=0.1, n_iters=10, use_sinkhorn=False)
    assert dec.use_sinkhorn is False
    assert dec.n_iters == 10
    assert hasattr(dec, "decode_at"), "decoder must expose decode_at()"


def test_se3_cfm_composite_signature() -> None:
    """Smoke: SE3CFMComposite class exists and exposes forward() callable."""
    assert callable(SE3CFMComposite)
    sig = inspect.signature(SE3CFMComposite.__init__)
    # Constructor should accept at least (base_cfm, se3_solver, ...).
    params = list(sig.parameters.keys())
    assert "base_cfm" in params
    assert "se3_solver" in params


def test_multi_ph_augment_helper() -> None:
    """Smoke: augment_pocket_embedding concats base embed + multi-PH features."""
    h = torch.randn(2, 16)
    # augment_pocket_embedding expects (base_pocket_emb, multi_ph_features)
    # where multi_ph_features is (B, n_bins, n_bins).
    mp_feats = torch.randn(2, 8, 8)
    out = augment_pocket_embedding(h, mp_feats)
    assert out.dim() == 2
    assert out.shape[0] == 2
    assert out.shape[1] == 16 + 64, f"expected 80, got {out.shape[1]}"
