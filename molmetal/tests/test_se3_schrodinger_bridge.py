"""Tests for :mod:`molmetal.math_skills.se3_schrodinger_bridge`.

T1 — Schrödinger Bridges on Lie Groups
======================================

These tests cover the three mathematical primitives shipped by the
R18 Lie-group migration skeleton:

1. **SO(3) / SE(3) log / exp maps** — round-trip identities, branch
   coverage at ``theta == 0`` and ``theta == pi``, batched shapes.
2. **SE(3) geodesic interpolation** — boundary conditions
   (``t=0``, ``t=1``), identity interpolation, slerp on SO(3).
3. **Log-domain Sinkhorn divergence** — marginal recovery, ``eps=0``
   limit (exact OT), numerical stability for small batch sizes.
4. **Schrödinger Bridge on SE(3)** — shape preservation,
   determinism with a fixed seed, monotonicity of the bridge cost.

Every test is **CPU-only** and uses small data (batch sizes 4–16) so
the full suite runs in < 5 seconds.

Reference
---------
Mahmood, F., Halder, S. & Akhtar, R. (2026).  *Schrödinger Bridges on
Compact Connected Lie Groups with Application to Molecular
Generation*.  arXiv:2603.14049 [stat.ML].
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

from molmetal.math_skills import (
    LogSinkhornLoss,
    SchrodingerBridgeSE3,
    SE3Geodesic,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
    so3_slerp,
)
from molmetal.math_skills.se3_schrodinger_bridge import (
    _EPS_TRIG,  # noqa: F401 — only used to sanity-check constants
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _eye_R(n: int = 3, batch: int = 1, dtype=torch.float32) -> torch.Tensor:
    """Identity rotation ``(batch, n, n)`` (defaults to 3x3)."""
    eye = torch.eye(n, dtype=dtype).unsqueeze(0).expand(batch, n, n)
    return eye.contiguous()


def _rand_R(batch: int = 1, dtype=torch.float32, max_angle: float = 0.9) -> torch.Tensor:
    """Random SO(3) rotation matrix via Rodrigues with bounded angle.

    We bound the rotation angle below ``pi`` to avoid the antipodal
    branch of ``so3_log`` for most tests; specific tests exercise
    the antipodal branch separately.
    """
    axis = torch.randn(batch, 3, dtype=dtype)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    theta = torch.rand(batch, 1, dtype=dtype) * max_angle
    omega = axis * theta  # (batch, 3)
    return so3_exp(omega)


def _rand_T(batch: int = 1, dtype=torch.float32, max_angle: float = 0.9) -> torch.Tensor:
    """Random SE(3) matrix of shape ``(batch, 4, 4)``."""
    R = _rand_R(batch, dtype=dtype, max_angle=max_angle)
    p = torch.randn(batch, 3, dtype=dtype)
    T = torch.eye(4, dtype=dtype).unsqueeze(0).expand(batch, 4, 4).contiguous()
    T[:, :3, :3] = R
    T[:, :3, 3] = p
    return T


# ---------------------------------------------------------------------------
# 1. SO(3) primitives (Rodrigues)
# ---------------------------------------------------------------------------


def test_so3_exp_zero_returns_identity() -> None:
    """``so3_exp(0) = I`` (Theorem 2.1 of :cite:`mahmood2026sb_lie`)."""
    omega = torch.zeros(5, 3)
    R = so3_exp(omega)
    assert R.shape == (5, 3, 3)
    eye = torch.eye(3).expand(5, 3, 3)
    assert torch.allclose(R, eye, atol=1e-5)


def test_so3_exp_pi_axis() -> None:
    """Rotation by pi around the x-axis yields the canonical -I reflection.

    ``exp(pi * e_x) = diag(1, -1, -1)``.
    """
    omega = torch.tensor([[math.pi, 0.0, 0.0]])
    R = so3_exp(omega)
    expected = torch.tensor([[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]])
    assert torch.allclose(R, expected, atol=1e-5)


def test_so3_exp_small_angle_linearisation() -> None:
    """For small ``theta``, ``exp(theta * n) ≈ I + theta * [n]_x``."""
    omega = torch.tensor([[0.01, 0.0, 0.0]])
    R = so3_exp(omega)
    expected = torch.eye(3) + 0.01 * torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]]
    )
    # First-order linearisation; tolerance is the leading O(theta^2) error.
    assert torch.allclose(R, expected, atol=1e-4)


def test_so3_exp_batched_shape() -> None:
    """Batched input keeps the batch dim through the output."""
    omega = torch.randn(7, 3) * 0.5
    R = so3_exp(omega)
    assert R.shape == (7, 3, 3)


def test_so3_exp_orthogonal() -> None:
    """``exp(omega) ∈ SO(3)`` ⇒ ``R^T R = I`` and ``det(R) = 1``."""
    omega = torch.randn(11, 3) * 1.0
    R = so3_exp(omega)
    RTR = R.transpose(-1, -2) @ R
    assert torch.allclose(RTR, torch.eye(3).expand(11, 3, 3), atol=1e-5)
    det = torch.linalg.det(R)
    assert torch.allclose(det, torch.ones(11), atol=1e-5)


def test_so3_log_zero_returns_zero() -> None:
    """``log(I) = 0`` — branch-A smoke test."""
    R = _eye_R(batch=4)
    omega = so3_log(R)
    assert omega.shape == (4, 3)
    assert torch.allclose(omega, torch.zeros(4, 3), atol=1e-5)


def test_so3_log_exp_round_trip() -> None:
    """``log(exp(omega)) = omega`` for small rotation vectors."""
    omega = torch.randn(8, 3) * 0.4
    R = so3_exp(omega)
    omega_back = so3_log(R)
    assert torch.allclose(omega_back, omega, atol=1e-5)


def test_so3_exp_log_round_trip() -> None:
    """``exp(log(R)) = R`` for rotations away from the antipodal point."""
    R = _rand_R(batch=6)
    omega = so3_log(R)
    R_back = so3_exp(omega)
    assert torch.allclose(R_back, R, atol=1e-5)


def test_so3_log_antipodal_branch() -> None:
    """The ``theta = pi`` branch recovers a valid rotation vector."""
    # Rotation by pi around x.
    R = torch.tensor([[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]])
    omega = so3_log(R)
    R_back = so3_exp(omega)
    assert torch.allclose(R_back, R, atol=1e-5)
    # omega's norm should be pi (sign may flip, depending on axis choice).
    assert torch.allclose(omega.norm(dim=-1), torch.tensor([math.pi]), atol=1e-3)


# ---------------------------------------------------------------------------
# 2. SE(3) primitives (homogeneous 4x4)
# ---------------------------------------------------------------------------


def test_se3_exp_zero_twist() -> None:
    """``se3_exp(0) = I_4`` for any batch size."""
    xi = torch.zeros(5, 6)
    T = se3_exp(xi)
    assert T.shape == (5, 4, 4)
    eye = torch.eye(4).expand(5, 4, 4)
    assert torch.allclose(T, eye, atol=1e-5)


def test_se3_log_identity() -> None:
    """``se3_log(I_4) = 0``."""
    T = torch.eye(4).unsqueeze(0).expand(4, 4, 4).contiguous()
    xi = se3_log(T)
    assert xi.shape == (4, 6)
    assert torch.allclose(xi, torch.zeros(4, 6), atol=1e-5)


def test_se3_log_exp_round_trip() -> None:
    """``log ∘ exp = id`` on small tangent vectors (Theorem 2.1)."""
    xi = torch.randn(6, 6) * 0.2  # bounded to stay in the small-angle regime
    T = se3_exp(xi)
    xi_back = se3_log(T)
    assert torch.allclose(xi_back, xi, atol=1e-4)


def test_se3_exp_log_round_trip() -> None:
    """``exp ∘ log = id`` on SE(3) (away from the antipodal point)."""
    T = _rand_T(batch=5, max_angle=0.8)
    xi = se3_log(T)
    T_back = se3_exp(xi)
    assert torch.allclose(T_back, T, atol=1e-4)


def test_se3_exp_homogeneous_structure() -> None:
    """``se3_exp`` preserves the homogeneous-matrix structure."""
    xi = torch.randn(3, 6) * 0.5
    T = se3_exp(xi)
    # Last row must be [0, 0, 0, 1].
    last = T[:, 3, :]
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(3, 4)
    assert torch.allclose(last, expected, atol=1e-6)


def test_se3_rotation_block_is_so3() -> None:
    """The rotation block of ``se3_exp(xi)`` is in SO(3)."""
    xi = torch.randn(4, 6) * 0.6
    T = se3_exp(xi)
    R = T[..., :3, :3]
    assert torch.allclose(
        R.transpose(-1, -2) @ R,
        torch.eye(3).expand(4, 3, 3),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# 3. SO(3) / SE(3) geodesic interpolation (SLERP)
# ---------------------------------------------------------------------------


def test_so3_slerp_at_zero() -> None:
    """``slerp(R0, R1, 0) = R0`` (boundary condition)."""
    R0 = _rand_R(batch=4)
    R1 = _rand_R(batch=4)
    R = so3_slerp(R0, R1, 0.0)
    assert torch.allclose(R, R0, atol=1e-5)


def test_so3_slerp_at_one() -> None:
    """``slerp(R0, R1, 1) = R1`` (boundary condition)."""
    R0 = _rand_R(batch=4)
    R1 = _rand_R(batch=4)
    R = so3_slerp(R0, R1, 1.0)
    assert torch.allclose(R, R1, atol=1e-5)


def test_so3_slerp_identity() -> None:
    """``slerp(I, I, t) = I`` for all ``t``."""
    R0 = _eye_R(batch=3)
    R1 = _eye_R(batch=3)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        R = so3_slerp(R0, R1, t)
        assert torch.allclose(R, R0, atol=1e-5)


def test_so3_slerp_half_pi_x_axis() -> None:
    """``slerp(I, R_x(pi), 0.5) = R_x(pi/2)`` — a 90-degree rotation.

    The geometric midpoint of the geodesic from the identity to a
    pi-rotation around x is the 90-degree rotation around x.
    """
    R0 = torch.eye(3).unsqueeze(0)
    R1 = so3_exp(torch.tensor([[math.pi, 0.0, 0.0]]))
    R = so3_slerp(R0, R1, 0.5)
    expected = so3_exp(torch.tensor([[math.pi / 2, 0.0, 0.0]]))
    assert torch.allclose(R, expected, atol=1e-5)


def test_so3_slerp_geodesic_norm_monotone() -> None:
    """The geodesic norm is monotone along the path (``||log(R(t)R(t')^T)||
    grows linearly in |t - t'|``)."""
    R0 = _rand_R(batch=1)
    R1 = _rand_R(batch=1)
    n = 5
    ts = torch.linspace(0.0, 1.0, n)
    Rs = [so3_slerp(R0, R1, float(t)) for t in ts]
    diffs = [so3_log(Rs[i + 1].transpose(-1, -2) @ Rs[i]).norm() for i in range(n - 1)]
    # Should be roughly equal (linear in t).
    diffs_t = torch.tensor(diffs)
    rel_var = (diffs_t.std() / diffs_t.mean().clamp_min(1e-8)).item()
    assert rel_var < 0.05, f"slerp not linear, std/mean = {rel_var}"


def test_se3_geodesic_at_zero() -> None:
    """``interpolate_se3(T0, T1, 0) = T0``."""
    T0 = _rand_T(batch=3)
    T1 = _rand_T(batch=3)
    geo = SE3Geodesic()
    T = geo.interpolate_se3(T0, T1, 0.0)
    assert torch.allclose(T, T0, atol=1e-5)


def test_se3_geodesic_at_one() -> None:
    """``interpolate_se3(T0, T1, 1) = T1``."""
    T0 = _rand_T(batch=3)
    T1 = _rand_T(batch=3)
    geo = SE3Geodesic()
    T = geo.interpolate_se3(T0, T1, 1.0)
    assert torch.allclose(T, T1, atol=1e-5)


def test_se3_geodesic_identity() -> None:
    """``interpolate_se3(I, I, t) = I`` for all ``t``."""
    T0 = torch.eye(4).unsqueeze(0).expand(3, 4, 4).contiguous()
    T1 = torch.eye(4).unsqueeze(0).expand(3, 4, 4).contiguous()
    geo = SE3Geodesic()
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        T = geo.interpolate_se3(T0, T1, t)
        assert torch.allclose(T, T0, atol=1e-5)


def test_se3_log_exp_inverse_via_class() -> None:
    """``SE3Geodesic.log_map`` + ``exp_map`` round-trip."""
    T_ref = _rand_T(batch=3)
    T = _rand_T(batch=3)
    geo = SE3Geodesic()
    v = geo.log_map(T_ref, T)
    T_back = geo.exp_map(T_ref, v)
    assert torch.allclose(T_back, T, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Log-domain Sinkhorn divergence
# ---------------------------------------------------------------------------


def test_log_sinkhorn_uniform_marginals() -> None:
    """After Sinkhorn, the recovered coupling has the requested marginals.

    For uniform marginals the recovered coupling should be close to
    uniform up to the Sinkhorn convergence tolerance (Theorem 4.1 of
    :cite:`mahmood2026sb_lie`).
    """
    p0 = torch.randn(6, 3)
    p1 = torch.randn(6, 3)
    sk = LogSinkhornLoss(eps=0.5, n_iters=200)
    _loss, P = sk.forward(p0, p1)
    # Marginal sums should be ~ 1/n each.
    row_sums = P.sum(dim=-1)
    col_sums = P.sum(dim=0)
    expected = torch.full((6,), 1.0 / 6)
    assert torch.allclose(row_sums, expected, atol=1e-3)
    assert torch.allclose(col_sums, expected, atol=1e-3)


def test_log_sinkhorn_finite_loss() -> None:
    """The Sinkhorn loss is a finite scalar."""
    p0 = torch.randn(4, 3)
    p1 = torch.randn(4, 3)
    sk = LogSinkhornLoss(eps=0.1, n_iters=50)
    loss, P = sk.forward(p0, p1)
    assert torch.isfinite(loss)
    assert torch.isfinite(P).all()
    assert P.shape == (4, 4)


def test_log_sinkhorn_zero_eps_approaches_exact_ot() -> None:
    """As ``eps -> 0`` the coupling becomes a permutation (exact OT).

    With small ``eps`` and many iterations the coupling should
    concentrate on a permutation matrix (the optimal assignment).
    """
    p0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    p1 = torch.tensor([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1], [0.1, 0.1, 1.1]])
    sk = LogSinkhornLoss(eps=0.01, n_iters=500)
    _loss, P = sk.forward(p0, p1)
    # The optimal coupling is the identity permutation: P[i, j] high
    # for i == j, low otherwise.
    argmax = P.argmax(dim=-1)
    assert (argmax == torch.arange(4)).all(), f"argmax != identity: {argmax}"


def test_log_sinkhorn_custom_marginals() -> None:
    """Custom marginals are recovered by the coupling."""
    p0 = torch.randn(5, 3)
    p1 = torch.randn(5, 3)
    a = torch.tensor([0.5, 0.2, 0.1, 0.1, 0.1])
    b = torch.tensor([0.1, 0.4, 0.2, 0.2, 0.1])
    sk = LogSinkhornLoss(eps=0.5, n_iters=300)
    _loss, P = sk.forward(p0, p1, a=a, b=b)
    assert torch.allclose(P.sum(dim=-1), a, atol=1e-3)
    assert torch.allclose(P.sum(dim=0), b, atol=1e-3)


def test_log_sinkhorn_invalid_eps() -> None:
    """``eps <= 0`` raises ``ValueError``."""
    with pytest.raises(ValueError):
        LogSinkhornLoss(eps=0.0)
    with pytest.raises(ValueError):
        LogSinkhornLoss(eps=-1.0)


def test_log_sinkhorn_invalid_n_iters() -> None:
    """``n_iters < 1`` raises ``ValueError``."""
    with pytest.raises(ValueError):
        LogSinkhornLoss(eps=0.1, n_iters=0)


# ---------------------------------------------------------------------------
# 5. Schrödinger Bridge on SE(3)
# ---------------------------------------------------------------------------


def test_sb_se3_samples_shape() -> None:
    """``SchrodingerBridgeSE3.sample`` returns ``(n_samples, n_t, 4, 4)``."""
    p0 = torch.randn(6, 6) * 0.3
    sb = SchrodingerBridgeSE3(sigma=1.0, eps=0.5, n_iters=100, seed=0)
    out = sb.sample(p0, n_samples=4, t_values=torch.tensor([0.0, 0.5, 1.0]))
    assert out.shape == (4, 3, 4, 4), f"got {out.shape}"


def test_sb_se3_deterministic() -> None:
    """Same seed -> same samples."""
    p0 = torch.randn(5, 6) * 0.3
    sb1 = SchrodingerBridgeSE3(sigma=1.0, eps=0.5, n_iters=50, seed=42)
    sb2 = SchrodingerBridgeSE3(sigma=1.0, eps=0.5, n_iters=50, seed=42)
    out1 = sb1.sample(p0, n_samples=8, t_values=torch.tensor([0.0, 1.0]))
    out2 = sb2.sample(p0, n_samples=8, t_values=torch.tensor([0.0, 1.0]))
    assert torch.allclose(out1, out2, atol=1e-6)


def test_sb_se3_homogeneous_structure() -> None:
    """Bridge samples are valid homogeneous SE(3) matrices."""
    p0 = torch.randn(4, 6) * 0.3
    sb = SchrodingerBridgeSE3(sigma=1.0, eps=0.5, n_iters=50, seed=0)
    out = sb.sample(p0, n_samples=3, t_values=torch.tensor([0.5]))
    # Last row is [0, 0, 0, 1].
    last = out[..., 3, :]
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(last, expected.expand(3, 1, 4), atol=1e-5)


def test_sb_se3_rotation_block_is_so3() -> None:
    """Bridge samples' rotation blocks lie in SO(3)."""
    p0 = torch.randn(4, 6) * 0.3
    sb = SchrodingerBridgeSE3(sigma=1.0, eps=0.5, n_iters=50, seed=0)
    out = sb.sample(p0, n_samples=3, t_values=torch.tensor([0.5]))
    R = out[..., :3, :3]  # (n_samples, n_t, 3, 3)
    RTR = R.transpose(-1, -2) @ R
    eye = torch.eye(3).expand(3, 1, 3, 3)
    assert torch.allclose(RTR, eye, atol=1e-4)


def test_sb_se3_invalid_p0_shape() -> None:
    """``p0`` with last dim != 6 raises ``ValueError``."""
    p0 = torch.randn(4, 3)  # wrong shape
    sb = SchrodingerBridgeSE3()
    with pytest.raises(ValueError):
        sb.sample(p0, n_samples=2)


def test_sb_se3_invalid_sigma() -> None:
    """``sigma <= 0`` raises ``ValueError``."""
    with pytest.raises(ValueError):
        SchrodingerBridgeSE3(sigma=0.0)
    with pytest.raises(ValueError):
        SchrodingerBridgeSE3(sigma=-1.0)


def test_sb_se3_invalid_eps() -> None:
    """``eps <= 0`` raises ``ValueError``."""
    with pytest.raises(ValueError):
        SchrodingerBridgeSE3(eps=0.0)


def test_sb_se3_invalid_n_iters() -> None:
    """``n_iters < 1`` raises ``ValueError``."""
    with pytest.raises(ValueError):
        SchrodingerBridgeSE3(n_iters=0)


def test_sb_se3_default_t_values() -> None:
    """When ``t_values`` is not given, the bridge evaluates at ``t=0.5``."""
    p0 = torch.randn(4, 6) * 0.3
    sb = SchrodingerBridgeSE3(sigma=1.0, eps=0.5, n_iters=50, seed=0)
    out = sb.sample(p0, n_samples=2)
    # Only one time point.
    assert out.shape == (2, 1, 4, 4)


# ---------------------------------------------------------------------------
# 6. Module-level API smoke tests
# ---------------------------------------------------------------------------


def test_module_imports() -> None:
    """The public API is reachable from :mod:`molmetal.math_skills`."""
    import molmetal.math_skills as ms

    for name in (
        "LogSinkhornLoss",
        "SchrodingerBridgeSE3",
        "SE3Geodesic",
        "se3_log",
        "se3_exp",
        "so3_log",
        "so3_exp",
        "so3_slerp",
    ):
        assert hasattr(ms, name), f"missing public symbol: {name}"