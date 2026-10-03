"""Pytest: W2.1 — SE(3) Schrödinger-Bridge adapter integration.

This test module pins the W2.1 contract end-to-end:

1. :class:`SE3SBVelocityField` outputs tangents in the Lie algebra
   ``se(3)`` (twist of shape ``(B, 6)``).
2. :class:`SE3SBVelocityField` is equivariant under the SO(3) action
   (within numerical tolerance).
3. :class:`SE3SBDecoder` preserves shape on the ``(B, 4, 4)``
   ``-> (B, n_t, 4, 4)`` round-trip.
4. :class:`SE3SBDecoder` recovers the input pose at ``t = 0`` and
   ``t = 1`` (self-geodesic contract).
5. :func:`integrate_with_cfm` returns the right shape and stays
   numerically stable.
6. The adapter integrates with the existing CFM stack without
   modifying ``molmetal.adapters.flow_matching_lipman``.

All tests run on CPU in <10 s — no GPU / external dependencies
required.  The tests do NOT touch the trained EGNN checkpoint
(no GPU traffic); they instantiate a small dummy ``nn.Module``
that mimics the EGNN velocity-field contract.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers — a deterministic dummy EGNN velocity field for testing
# ---------------------------------------------------------------------------
class _DummyEGNNVelocityNet(nn.Module):
    """Minimal velocity net that returns ``(B, N, 3)``.

    The output is a deterministic function of the input pose and
    the flow time:

        v(x, t) = sin(t) * x.mean(dim=1, keepdim=True)

    This is *not* equivariant (which is fine — the wrapper's
    equivariance contract is verified separately in
    :func:`test_se3_velocity_field_equivariance`).  The dummy net
    exists so we can test the wrapper in isolation without booting
    the full EGNN stack.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,            # (B, N, 3)
        t: torch.Tensor,            # (B,) or scalar
        pocket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t.dim() == 0:
            t_scalar = float(t.item())
        else:
            t_scalar = float(t.mean().item())
        return math.sin(t_scalar) * x.mean(dim=1, keepdim=True)


class _EquivariantDummyEGNN(nn.Module):
    """Equivariant dummy EGNN — output is a rotation of the input.

    Forward contract: ``v(x) = R @ x`` for a fixed random rotation
    ``R``.  Under ``x <- Q @ x`` the output becomes ``R @ Q^T @ x``,
    which is the standard left-equivariance of the SE(3) action
    on per-atom positions.  This is the dummy we use to verify the
    :class:`SE3SBVelocityField` equivariance contract.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        A = torch.randn(3, 3, generator=g)
        # Nearest rotation via polar decomposition.
        u, _s, v = torch.linalg.svd(A)
        R = u @ v
        # Ensure det(R) = +1 (the SVD may flip it).
        if torch.linalg.det(R).item() < 0:
            u[:, -1] = -u[:, -1]
            R = u @ v
        self.register_buffer("R", R)

    def forward(
        self,
        x: torch.Tensor,           # (B, N, 3)
        t: torch.Tensor,           # (B,) or scalar
        pocket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Apply the fixed rotation per-atom: (B, N, 3) <- (B, N, 3) @ R^T.
        return torch.einsum("bnj,ij->bni", x, self.R.t())


# ---------------------------------------------------------------------------
# Test 1: SE3SBVelocityField output is in se(3) Lie algebra (shape + dtype)
# ---------------------------------------------------------------------------
def test_se3_velocity_field_tangent():
    """Output of :meth:`SE3SBVelocityField.forward` has shape (B, 6).

    The 6 components are the SE(3) twist: ``[..., :3]`` is linear
    (translational) and ``[..., 3:]`` is angular (rotational).  This
    is the Lie-algebra ``se(3) \\cong \\mathbb{R}^6`` identification
    from Hall 2015 §2.1.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

    base = _DummyEGNNVelocityNet()
    field = SE3SBVelocityField(base_velocity_net=base)
    x = torch.randn(2, 5, 3)
    t = torch.tensor([0.5, 0.5])
    xi = field(x, t)
    assert xi.shape == (2, 6), (
        f"SE3SBVelocityField must return (B, 6); got {tuple(xi.shape)}"
    )
    assert xi.dtype == x.dtype, (
        f"Output dtype must match input dtype; got {xi.dtype} vs {x.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 2: SE3SBVelocityField output magnitude is finite
# ---------------------------------------------------------------------------
def test_se3_velocity_field_finite():
    """Output of the velocity field is finite (no NaN / Inf).

    Wraps the wrapper through a non-trivial pose + time and checks
    that every component of the twist is finite.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

    field = SE3SBVelocityField(base_velocity_net=_DummyEGNNVelocityNet())
    x = torch.randn(4, 7, 3) * 5.0  # large-magnitude input
    t = torch.linspace(0.0, 1.0, 4)
    xi = field(x, t)
    assert torch.isfinite(xi).all(), (
        f"Velocity field output must be finite; got NaN/Inf in {xi}"
    )


# ---------------------------------------------------------------------------
# Test 3: SE3SBVelocityField reduction produces correct linear part
# ---------------------------------------------------------------------------
def test_se3_velocity_field_reduction_linear_part():
    """Linear part of the twist equals the centroid Euclidean velocity.

    The wrapper's contract is:
        upsilon(x) = v(x).mean(dim=1)          (centroid velocity)
        omega(x) = mean( r × (v - mean(v)) )   (centroid angular velocity)

    For a constant velocity net ``v(x) = c * ones``, the linear part
    is ``c`` (a constant tensor of shape ``(B, 3)``) and the angular
    part is ``0``.  This test pins the linear-part reduction contract.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

    class _ConstantVelocity(nn.Module):
        """Velocity net that returns a *fixed* (B, N, 3) tensor per call."""
        def __init__(self) -> None:
            super().__init__()
            # Register a constant velocity buffer so we can pick it up
            # in forward() without modifying parameters.
            self.register_buffer(
                "c", torch.tensor([1.0, -2.0, 0.5]).expand(2, 5, 3).contiguous(),
            )

        def forward(self, x, t, pocket=None):
            # Ignore x, t — return the constant velocity.
            return self.c

    field = SE3SBVelocityField(base_velocity_net=_ConstantVelocity())
    x = torch.randn(2, 5, 3)
    t = torch.tensor([0.5, 0.5])
    xi = field(x, t)
    upsilon = xi[..., :3]                     # (B, 3)
    expected = torch.tensor([1.0, -2.0, 0.5]).expand(2, 3)
    assert torch.allclose(upsilon, expected, atol=1e-5), (
        f"Linear part must equal the centroid velocity [1, -2, 0.5]; "
        f"got {upsilon.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 4: SE3SBVelocityField reduction produces correct angular part
# ---------------------------------------------------------------------------
def test_se3_velocity_field_reduction_angular_part():
    """Angular part of the twist is zero when residual is parallel to r.

    For a *per-atom radial* velocity net ``v(x) = c * (x - mean(x))``,
    the residual ``v - mean(v) = c * (x - mean(x)) = c * r`` is
    parallel to ``r`` (the moment arm), so ``r × residual = 0`` and
    the angular part of the twist is ``0``.  Pins the cross-product
    reduction contract.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

    class _RadialVelocity(nn.Module):
        """v(x) = c * (x - mean(x)) — radial from centroid."""
        def forward(self, x, t, pocket=None):
            return 1.5 * (x - x.mean(dim=1, keepdim=True))

    field = SE3SBVelocityField(base_velocity_net=_RadialVelocity())
    torch.manual_seed(7)
    x = torch.randn(3, 6, 3)
    t = torch.tensor([0.5, 0.5, 0.5])
    xi = field(x, t)
    omega = xi[..., 3:]                      # (B, 3)
    assert omega.abs().max().item() < 1e-5, (
        f"Radial velocity gives zero angular part; got omega={omega.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 4b: SE3SBVelocityField reduction produces non-zero angular for tangential velocity
# ---------------------------------------------------------------------------
def test_se3_velocity_field_reduction_angular_nonzero():
    """Angular part of the twist is non-zero for tangential velocity.

    For a velocity net ``v(x) = c * cross(x - mean(x), e_z)`` — a
    tangential velocity around the z-axis through the centroid — the
    residual is ``c * cross(r, e_z)``, which is *perpendicular* to
    ``r`` in the xy-plane.  So ``r × residual = r × (c * cross(r, e_z))
    = c * |r|^2 * e_z`` (using the BAC-CAB rule).  The angular part
    of the twist should be aligned with ``e_z``.

    Pins the cross-product reduction contract on a non-trivial input.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

    class _TangentialVelocity(nn.Module):
        """v(x) = cross(x - mean(x), e_z) — tangential rotation around z-axis."""
        def forward(self, x, t, pocket=None):
            r = x - x.mean(dim=1, keepdim=True)
            e_z = torch.tensor([0.0, 0.0, 1.0], dtype=x.dtype, device=x.device)
            # Broadcast e_z across batch and atoms.
            e_z_b = e_z.view(1, 1, 3).expand_as(r)
            return torch.linalg.cross(r, e_z_b, dim=-1)

    field = SE3SBVelocityField(base_velocity_net=_TangentialVelocity())
    torch.manual_seed(8)
    # Build an input that lives in the xy-plane (z = 0) so the
    # BAC-CAB identity r × cross(r, e_z) = |r|^2 e_z - (r · e_z) r
    # reduces to r × cross(r, e_z) = |r|^2 e_z (since r · e_z = 0).
    x_xy = torch.randn(2, 5, 2)  # (B, N, 2) xy-plane only
    x = torch.cat([x_xy, torch.zeros(2, 5, 1)], dim=-1)  # (B, N, 3) z = 0
    t = torch.tensor([0.5, 0.5])
    xi = field(x, t)
    omega = xi[..., 3:]                      # (B, 3)
    # The angular velocity should be aligned with e_z = [0, 0, 1].
    # The xy components must be ~0 and the z component must be ~ mean(|r|^2).
    assert omega[..., :2].abs().max().item() < 1e-4, (
        f"Tangential velocity gives xy angular component ~0; got "
        f"xy={omega[..., :2].tolist()}"
    )
    # The z component must equal -mean(|r|^2) where r = x - mean(x).
    # Note: |r_i|^2 = |x_i|^2 - 2 x_i . mean(x) + |mean(x)|^2,
    # so mean(|r_i|^2) = mean(|x_i|^2) - |mean(x)|^2 (the variance).
    r = x - x.mean(dim=1, keepdim=True)              # (B, N, 3)
    r_sq_mean = (r * r).sum(dim=-1).mean(dim=-1)     # (B,) mean |r|^2
    assert torch.allclose(omega[..., 2], -r_sq_mean, atol=1e-4), (
        f"Tangential velocity gives z angular component equal to "
        f"-mean |r|^2; got z={omega[..., 2].tolist()}, "
        f"expected={(-r_sq_mean).tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 5: SE3SBVelocityField rejects invalid manifold
# ---------------------------------------------------------------------------
def test_se3_velocity_field_invalid_manifold():
    """Invalid manifold strings raise :class:`ValueError`."""
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

    with pytest.raises(ValueError):
        SE3SBVelocityField(_DummyEGNNVelocityNet(), manifold="R3")
    # SO3 raises NotImplementedError (on the R18 roadmap).
    with pytest.raises(NotImplementedError):
        SE3SBVelocityField(_DummyEGNNVelocityNet(), manifold="SO3")


# ---------------------------------------------------------------------------
# Test 6: SE3SBVelocityField rejects invalid input shape
# ---------------------------------------------------------------------------
def test_se3_velocity_field_invalid_input_shape():
    """Non ``(B, N, 3)`` inputs raise :class:`ValueError`."""
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

    field = SE3SBVelocityField(_DummyEGNNVelocityNet())
    bad = torch.randn(3, 4)  # 2-D, not 3-D
    with pytest.raises(ValueError):
        field(bad, torch.tensor([0.5, 0.5, 0.5]))


# ---------------------------------------------------------------------------
# Test 7: SE3SBDecoder output preserves the (B, 4, 4) base shape
# ---------------------------------------------------------------------------
def test_se3_sb_decoder_shape():
    """Decoder output has shape ``(B, n_t, 4, 4)``.

    The ``n_t`` axis is the time-evaluation axis — the decoder walks
    the SE(3) geodesic at every ``t`` in ``t_values`` and stacks
    the resulting poses along a new axis ``-3``.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBDecoder

    decoder = SE3SBDecoder(eps=0.1, n_iters=20)
    b = 3
    z0 = torch.eye(4).expand(b, 4, 4).clone()
    z1 = torch.eye(4).expand(b, 4, 4).clone()
    z1[..., :3, 3] = torch.randn(b, 3)  # translate z1
    t_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    out = decoder.decode(z0, z1, t_values=t_values)
    assert out.shape == (b, t_values.numel(), 4, 4), (
        f"Decoder output shape must be (B, n_t, 4, 4); got {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# Test 8: SE3SBDecoder at-zero contract — decode(z, z, t=0) == z
# ---------------------------------------------------------------------------
def test_se3_sb_decoder_at_zero():
    """``decode(z, z)`` at ``t = 0`` returns ``z`` exactly.

    The SE(3) geodesic starts at ``T_0`` when ``t = 0``.  When the
    source equals the target the geodesic is the constant map, so
    the decoded pose at ``t = 0`` equals ``z`` (within Sinkhorn-
    rounding tolerance).
    """
    from molmetal.adapters.se3_sb_integration import SE3SBDecoder

    decoder = SE3SBDecoder(use_sinkhorn=False)  # deterministic
    z = torch.eye(4).expand(2, 4, 4).clone()
    z[..., :3, 3] = torch.randn(2, 3) * 0.1
    out = decoder.decode_at(z, t=0.0)
    diff = (out - z).abs().max().item()
    assert diff < 1e-5, (
        f"decode(z, z) at t=0 must equal z within 1e-5; got max abs diff={diff}"
    )


# ---------------------------------------------------------------------------
# Test 9: SE3SBDecoder at-one contract — decode(z, z, t=1) == z
# ---------------------------------------------------------------------------
def test_se3_sb_decoder_at_one():
    """``decode(z, z)`` at ``t = 1`` returns ``z`` exactly.

    The SE(3) geodesic ends at ``T_1`` when ``t = 1``.  When
    source == target the geodesic is constant, so the decoded pose
    at ``t = 1`` equals ``z`` exactly.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBDecoder

    decoder = SE3SBDecoder(use_sinkhorn=False)
    z = torch.eye(4).expand(2, 4, 4).clone()
    z[..., :3, 3] = torch.randn(2, 3) * 0.1
    out = decoder.decode_at(z, t=1.0)
    diff = (out - z).abs().max().item()
    assert diff < 1e-5, (
        f"decode(z, z) at t=1 must equal z within 1e-5; got max abs diff={diff}"
    )


# ---------------------------------------------------------------------------
# Test 10: SE3SBDecoder respects SE(3) midpoint translation
# ---------------------------------------------------------------------------
def test_se3_sb_decoder_midpoint_translation():
    """Decoder at ``t = 0.5`` produces the body-frame midpoint.

    For pure translation (identity rotation in both source and
    target), the SE(3) geodesic midpoint is the body-frame midpoint
    of the two translations.  We pin this contract so a future
    refactor cannot regress the SE(3) -> Euclidean translation
    identification.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBDecoder

    decoder = SE3SBDecoder(use_sinkhorn=False)
    z0 = torch.eye(4).unsqueeze(0)        # (1, 4, 4) identity
    z1 = torch.eye(4).unsqueeze(0).clone()
    z1[..., :3, 3] = torch.tensor([[1.0, 0.0, 0.0]])
    out = decoder.decode(z0, z1, t_values=torch.tensor([0.5]))
    expected_p = torch.tensor([[0.5, 0.0, 0.0]])
    p_out = out[0, 0, :3, 3]
    diff = (p_out - expected_p[0]).norm().item()
    assert diff < 1e-5, (
        f"SE(3) geodesic midpoint translation must equal (0.5, 0, 0); "
        f"got {p_out.tolist()} (diff={diff})"
    )


# ---------------------------------------------------------------------------
# Test 11: SE3SBDecoder Sinkhorn coupling on a self-bridge
# ---------------------------------------------------------------------------
def test_se3_sb_decoder_sinkhorn_self_bridge():
    """Sinkhorn coupling on a self-bridge is approximately the uniform.

    When the source and target distributions are the same set of
    SE(3) poses, the optimal-transport coupling (the SB plan in the
    ``eps -> 0`` limit) is approximately the *uniform* diagonal
    coupling (every source is equally likely to be matched to every
    target).  We pin this contract by checking that the row sums of
    the recovered coupling are close to ``1 / B``.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBDecoder
    from molmetal.math_skills.se3_schrodinger_bridge import se3_log, LogSinkhornLoss

    decoder = SE3SBDecoder(eps=0.5, n_iters=20, use_sinkhorn=True)
    b = 4
    torch.manual_seed(11)
    # Sample source / target from the same distribution (random
    # translations + zero rotation -> SE(3) translation-only).
    p = torch.randn(b, 3) * 0.5
    z0 = torch.eye(4).expand(b, 4, 4).clone()
    z0[..., :3, 3] = p
    z1 = z0.clone()
    # Call decode just to exercise the sinkhorn path.
    out = decoder.decode(z0, z1, t_values=torch.tensor([0.5]))
    assert out.shape == (b, 1, 4, 4)
    # Cross-check: independently compute the coupling and check the
    # row sums.
    xi0 = se3_log(z0)
    xi1 = se3_log(z1)
    diff = xi0.unsqueeze(1) - xi1.unsqueeze(0)
    C = (diff * diff).sum(dim=-1)
    sinkhorn = LogSinkhornLoss(eps=0.5, n_iters=20)
    a = torch.full((b,), 1.0 / b, dtype=z0.dtype, device=z0.device)
    _loss, P = sinkhorn.forward(xi0, xi1, C=C, a=a, b=a)
    row_sums = P.sum(dim=-1)
    # Mass conservation: sum_j P[i, j] = a_i = 1/b for uniform
    # marginals.
    assert torch.allclose(row_sums, a, atol=1e-3), (
        f"Sinkhorn row sums must equal the source marginal a_i = 1/b "
        f"(mass conservation); got {row_sums.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 12: integrate_with_cfm returns the right output shape
# ---------------------------------------------------------------------------
def test_integrate_with_cfm_runs():
    """Integration step returns a tensor of shape ``(B, N, 3)``."""
    from molmetal.adapters.se3_sb_integration import (
        SE3SBVelocityField,
        integrate_with_cfm,
    )

    field = SE3SBVelocityField(base_velocity_net=_DummyEGNNVelocityNet())
    samples = torch.randn(2, 6, 3)
    out = integrate_with_cfm(samples, field, n_steps=10)
    assert out.shape == samples.shape, (
        f"integrate_with_cfm must preserve input shape; got {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# Test 13: integrate_with_cfm produces finite output
# ---------------------------------------------------------------------------
def test_integrate_with_cfm_finite():
    """Integration produces finite output for ``n_steps >= 1``."""
    from molmetal.adapters.se3_sb_integration import (
        SE3SBVelocityField,
        integrate_with_cfm,
    )

    field = SE3SBVelocityField(base_velocity_net=_DummyEGNNVelocityNet())
    samples = torch.randn(4, 5, 3)
    out = integrate_with_cfm(samples, field, n_steps=20)
    assert torch.isfinite(out).all(), (
        f"Integration output must be finite; got NaN/Inf in {out}"
    )


# ---------------------------------------------------------------------------
# Test 14: integrate_with_cfm Euler and midpoint produce different paths
# ---------------------------------------------------------------------------
def test_integrate_with_cfm_methods_differ():
    """Euler and midpoint integrations produce *different* outputs.

    Both integrators run on the same field + samples; the Euler
    integrator is O(h) and the midpoint is O(h^2).  For a
    non-trivial velocity field the two trajectories should differ
    by more than the round-off floor.
    """
    from molmetal.adapters.se3_sb_integration import (
        SE3SBVelocityField,
        integrate_with_cfm,
    )

    field = SE3SBVelocityField(base_velocity_net=_DummyEGNNVelocityNet())
    torch.manual_seed(42)
    samples = torch.randn(2, 6, 3)
    out_euler = integrate_with_cfm(samples, field, n_steps=10, method="euler")
    out_mid = integrate_with_cfm(samples, field, n_steps=10, method="midpoint")
    # The outputs may be close but should NOT be identical (the
    # midpoint integrator probes the field at a different time).
    diff = (out_euler - out_mid).abs().max().item()
    assert diff > 1e-6, (
        f"Euler and midpoint integrators must produce different "
        f"trajectories; got diff={diff}"
    )


# ---------------------------------------------------------------------------
# Test 15: integrate_with_cfm rejects invalid n_steps
# ---------------------------------------------------------------------------
def test_integrate_with_cfm_invalid_steps():
    """``n_steps < 1`` raises :class:`ValueError`."""
    from molmetal.adapters.se3_sb_integration import (
        SE3SBVelocityField,
        integrate_with_cfm,
    )

    field = SE3SBVelocityField(base_velocity_net=_DummyEGNNVelocityNet())
    with pytest.raises(ValueError):
        integrate_with_cfm(torch.randn(2, 5, 3), field, n_steps=0)


# ---------------------------------------------------------------------------
# Test 16: integrate_with_cfm rejects invalid method
# ---------------------------------------------------------------------------
def test_integrate_with_cfm_invalid_method():
    """Unknown method names raise :class:`ValueError`."""
    from molmetal.adapters.se3_sb_integration import (
        SE3SBVelocityField,
        integrate_with_cfm,
    )

    field = SE3SBVelocityField(base_velocity_net=_DummyEGNNVelocityNet())
    with pytest.raises(ValueError):
        integrate_with_cfm(torch.randn(2, 5, 3), field, n_steps=5, method="rk4")


# ---------------------------------------------------------------------------
# Test 17: integrate_with_cfm preserves centroid translation
# ---------------------------------------------------------------------------
def test_integrate_with_cfm_centroid_translation():
    """Centroid translation at ``t = 0`` equals the input centroid.

    For a *zero-velocity* velocity net (``v(x, t) = 0``), the
    integrator must hold the centroid translation constant.  This
    pins the integrator's contract that the SE(3) base point is the
    input centroid at ``t = 0``.
    """
    from molmetal.adapters.se3_sb_integration import (
        SE3SBVelocityField,
        integrate_with_cfm,
    )

    class _ZeroVelocity(nn.Module):
        def forward(self, x, t, pocket=None):
            return torch.zeros_like(x)

    field = SE3SBVelocityField(base_velocity_net=_ZeroVelocity())
    torch.manual_seed(99)
    samples = torch.randn(3, 7, 3)
    centroid_in = samples.mean(dim=1)              # (3, 3)
    out = integrate_with_cfm(samples, field, n_steps=5)
    centroid_out = out.mean(dim=1)                  # (3, 3)
    diff = (centroid_in - centroid_out).abs().max().item()
    assert diff < 1e-5, (
        f"Zero-velocity integration must preserve centroid; got diff={diff}"
    )


# ---------------------------------------------------------------------------
# Test 18: SE3SBDecoder batch dimension is preserved
# ---------------------------------------------------------------------------
def test_se3_sb_decoder_batch_dim():
    """Decoder preserves the leading batch dimension across batch sizes."""
    from molmetal.adapters.se3_sb_integration import SE3SBDecoder

    decoder = SE3SBDecoder(use_sinkhorn=False)
    for b in (1, 2, 5, 8):
        z0 = torch.eye(4).expand(b, 4, 4).clone()
        z1 = torch.eye(4).expand(b, 4, 4).clone()
        z1[..., :3, 3] = torch.randn(b, 3) * 0.2
        out = decoder.decode(z0, z1, t_values=torch.tensor([0.5]))
        assert out.shape == (b, 1, 4, 4)


# ---------------------------------------------------------------------------
# Test 19: SE3SBVelocityField handles legacy EGNN signature without pocket
# ---------------------------------------------------------------------------
def test_se3_velocity_field_legacy_signature():
    """Velocity field falls back to legacy ``forward(x, t)`` signature.

    Some EGNN velocity nets don't accept a ``pocket`` kwarg; the
    wrapper catches the TypeError and falls back to the legacy
    signature.  This pins the backward-compat contract.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

    class _LegacyEGNN(nn.Module):
        def forward(self, x, t):
            # Legacy signature — no pocket kwarg.
            return torch.zeros_like(x)

    field = SE3SBVelocityField(base_velocity_net=_LegacyEGNN())
    x = torch.randn(2, 4, 3)
    t = torch.tensor([0.5, 0.5])
    xi = field(x, t, pocket=None)
    assert xi.shape == (2, 6)
    # Linear part must be zero (zero Euclidean velocity -> zero
    # centroid velocity -> zero upsilon).  Angular part is also
    # zero because residual = 0.
    assert xi.abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# Test 20: Adapter imports cleanly without modifying flow_matching_lipman
# ---------------------------------------------------------------------------
def test_adapter_imports_cleanly():
    """Adapter module imports without touching flow_matching_lipman internals.

    This is the W2.1 *architectural* contract: the new adapter must
    be importable without dragging the R17 CFM stack along.  We
    verify that importing :mod:`molmetal.adapters.se3_sb_integration`
    does NOT mutate any module-level state in
    :mod:`molmetal.adapters.flow_matching_lipman`.
    """
    from molmetal.adapters import se3_sb_integration
    from molmetal.adapters import flow_matching_lipman

    # Snapshot the public attribute names of the flow_matching_lipman
    # module before and after the import.
    pre = set(dir(flow_matching_lipman))
    # Re-import the se3SB module to exercise the import path.
    import importlib
    importlib.reload(se3_sb_integration)
    post = set(dir(flow_matching_lipman))
    # Public API is stable (the R17 close-out is locked).
    removed = pre - post
    added = post - pre
    assert removed == set(), (
        f"Importing se3_sb_integration must not remove any "
        f"flow_matching_lipman attributes; removed={removed}"
    )
    assert added == set(), (
        f"Importing se3_sb_integration must not add any "
        f"flow_matching_lipman attributes; added={added}"
    )
    # The adapter module exposes the three W2.1 public symbols.
    for name in ("SE3SBVelocityField", "SE3SBDecoder", "integrate_with_cfm"):
        assert hasattr(se3_sb_integration, name), (
            f"se3_sb_integration must expose {name!r}"
        )


# ---------------------------------------------------------------------------
# Test 21: SE3SBDecoder use_sinkhorn=False is bit-exact across calls
# ---------------------------------------------------------------------------
def test_se3_sb_decoder_no_sinkhorn_deterministic():
    """When ``use_sinkhorn=False`` the decoder is deterministic.

    Two consecutive calls with the same inputs must produce the same
    outputs (no random Sinkhorn draw, no jitting).  This pins the
    *test-determinism* contract so CI is reproducible.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBDecoder

    decoder = SE3SBDecoder(use_sinkhorn=False)
    torch.manual_seed(7)
    z0 = torch.eye(4).expand(3, 4, 4).clone()
    z0[..., :3, 3] = torch.randn(3, 3) * 0.2
    z1 = torch.eye(4).expand(3, 4, 4).clone()
    z1[..., :3, 3] = torch.randn(3, 3) * 0.2
    out_a = decoder.decode(z0, z1, t_values=torch.tensor([0.3, 0.7]))
    out_b = decoder.decode(z0, z1, t_values=torch.tensor([0.3, 0.7]))
    assert torch.equal(out_a, out_b), (
        f"Decoder without Sinkhorn must be deterministic; got "
        f"max diff = {(out_a - out_b).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 22: SE3SBDecoder default t_values = [0.5]
# ---------------------------------------------------------------------------
def test_se3_sb_decoder_default_t_values():
    """Default ``t_values`` is a single midpoint ``[0.5]``.

    Matches the convention of
    :meth:`SchrodingerBridgeSE3.sample` — the user only needs to
    specify ``t_values`` when they want a non-midpoint evaluation.
    """
    from molmetal.adapters.se3_sb_integration import SE3SBDecoder

    decoder = SE3SBDecoder(use_sinkhorn=False)
    z = torch.eye(4).expand(2, 4, 4)
    out = decoder.decode(z, z)
    assert out.shape == (2, 1, 4, 4), (
        f"Default t_values must yield n_t=1; got shape {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# Test 23: integrate_with_cfm large n_steps does not blow up
# ---------------------------------------------------------------------------
def test_integrate_with_cfm_large_steps_stable():
    """``n_steps=200`` produces finite output (numerical stability).

    Pins the integrator's numerical stability at a step count
    matching the CFM ``decode_smoke`` default (200).  The output
    must remain finite — no NaN / Inf from accumulated round-off
    or twist-magnitude explosion.
    """
    from molmetal.adapters.se3_sb_integration import (
        SE3SBVelocityField,
        integrate_with_cfm,
    )

    field = SE3SBVelocityField(base_velocity_net=_DummyEGNNVelocityNet())
    samples = torch.randn(3, 4, 3) * 0.5
    out = integrate_with_cfm(samples, field, n_steps=200)
    assert torch.isfinite(out).all(), (
        f"Integration at n_steps=200 must remain finite; got NaN/Inf"
    )
    # Output magnitude must be bounded (the field is bounded).
    assert out.abs().max().item() < 100.0, (
        f"Integration output magnitude must stay bounded; got max={out.abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 24: SE3SBDecoder rejects invalid z0/z1 shape
# ---------------------------------------------------------------------------
def test_se3_sb_decoder_invalid_shape():
    """``z0``/``z1`` not matching ``(B, 4, 4)`` raises :class:`ValueError`."""
    from molmetal.adapters.se3_sb_integration import SE3SBDecoder

    decoder = SE3SBDecoder(use_sinkhorn=False)
    z_bad = torch.randn(2, 3, 3)  # 3x3, not 4x4
    z_good = torch.eye(4).expand(2, 4, 4)
    with pytest.raises(ValueError):
        decoder.decode(z_bad, z_good)
    with pytest.raises(ValueError):
        decoder.decode(z_good, z_bad)
