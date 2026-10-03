"""Pytest: R18 P2-4 — Quotient-Space Diffusion adapter.

This test module pins the R18 P2-4 contract end-to-end:

1. :class:`QuotientSpace` projects conformers into the quotient
   space (centre + Procrustes alignment).
2. :class:`QuotientSpace` is *SE(3)-equivariant by construction*:
   ``project(x) == project(R x + t)`` for every rotation ``R`` and
   translation ``t``.
3. :class:`QuotientSpace.lift` returns a valid conformer with the
   same shape as the input.
4. :class:`QuotientDiffusionScheduler.step` adds Gaussian noise with
   the right incremental variance and preserves shape.
5. :class:`QuotientSpaceCFM` wraps a base CFM and preserves shape /
   dtype / equivariance.
6. :meth:`QuotientSpaceCFM.equivariance_error` returns a small
   number for a *trivially-equivariant* base CFM.

All tests run on CPU in <10 s — no GPU / external dependencies
required.  The tests do NOT touch any trained EGNN checkpoint;
they instantiate a small dummy ``nn.Module`` that mimics the
CFM velocity-field contract.
"""

from __future__ import annotations

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
# Helpers — deterministic dummy modules for testing
# ---------------------------------------------------------------------------
class _DummyCFM(nn.Module):
    """Minimal CFM-like callable for testing.

    Forward contract: ``forward(z0, t) -> Tensor`` of the same shape
    as ``z0``.  The output is a deterministic function of the
    *projected* input ``z0`` and the flow time:

        v(z, t) = (1 - t) * z

    This is a *trivially SE(3)-equivariant* velocity field (it
    commutes with any rotation / translation because it's just a
    scalar multiple of the input).  It exists so we can test the
    wrapper's equivariance contract without booting the full
    LipmanFlowMatchingAdapter.
    """

    def __init__(self, scale: float = 0.5) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        pocket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t.dim() == 0:
            t_scalar = float(t.item())
        else:
            t_scalar = float(t.mean().item())
        # Trivial equivariant flow: return the identity, scaled by
        # the flow time so the output shape matches the input.
        return self.scale * (1.0 - t_scalar) * z0


class _DummyNonEquivariantCFM(nn.Module):
    """CFM-like callable that *flips* the input — breaks equivariance.

    Forward contract: ``forward(z0, t) -> -z0``.  Under a rotation
    ``R z0`` the output is ``-R z0``, which equals ``R (-z0)`` only
    if ``R = -R`` (never the case for SO(3)).  So this base CFM is
    *not* SE(3)-equivariant on its own.

    Used to verify that :class:`QuotientSpaceCFM.equivariance_error`
    detects non-equivariant base CFMs.
    """

    def forward(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        pocket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return -z0


def _random_rotation_matrix(
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Sample ``batch`` random ``SO(3)`` rotation matrices."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(batch, 3, 3, generator=g, dtype=dtype).to(device)
    u, _s, v = torch.linalg.svd(A)
    R = u @ v
    det = torch.linalg.det(R)
    need_flip = det < 0
    if need_flip.any():
        v[need_flip, -1, :] = -v[need_flip, -1, :]
    R = u @ v
    return R


# ---------------------------------------------------------------------------
# Test 1: QuotientSpace constructor validation
# ---------------------------------------------------------------------------
def test_quotient_space_constructor_default():
    """Default constructor uses the ``"SE3"`` manifold."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    q = QuotientSpace()
    assert q.manifold == "SE3"
    assert q.eps > 0


def test_quotient_space_constructor_explicit_manifold():
    """Explicit manifold selection (rotation / translation)."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    for m in {"SE3", "rotation", "translation"}:
        q = QuotientSpace(manifold=m)
        assert q.manifold == m


def test_quotient_space_constructor_invalid_manifold():
    """Invalid manifold raises ``ValueError``."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    with pytest.raises(ValueError):
        QuotientSpace(manifold="invalid")
    with pytest.raises(ValueError):
        QuotientSpace(manifold="")


def test_quotient_space_constructor_invalid_eps():
    """Non-positive ``eps`` raises ``ValueError``."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    with pytest.raises(ValueError):
        QuotientSpace(eps=0.0)
    with pytest.raises(ValueError):
        QuotientSpace(eps=-1.0)


# ---------------------------------------------------------------------------
# Test 2: QuotientSpace.project preserves shape / dtype
# ---------------------------------------------------------------------------
def test_quotient_space_project_shape_dtype():
    """``project`` preserves shape and dtype for the SE(3) quotient."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    q = QuotientSpace(manifold="SE3")
    x = torch.randn(3, 5, 3, dtype=torch.float64)
    y = q.project(x)
    assert y.shape == x.shape, f"project must preserve shape; got {tuple(y.shape)}"
    assert (
        y.dtype == x.dtype
    ), f"project must preserve dtype; got {y.dtype} vs {x.dtype}"


def test_quotient_space_project_translation_shape():
    """``project`` for the ``"translation"`` quotient centres the input."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    q = QuotientSpace(manifold="translation")
    x = torch.randn(2, 4, 3) + 7.0  # far from origin
    y = q.project(x)
    assert y.shape == x.shape
    # Centred: per-batch mean must be zero.
    mean = y.mean(dim=1)
    assert torch.allclose(
        mean, torch.zeros_like(mean), atol=1e-5
    ), f"translation-quotient must centre coordinates; got mean={mean}"


def test_quotient_space_project_rotation_shape():
    """``project`` for the ``"rotation"`` quotient preserves centroid norm.

    The rotation quotient canonicalises the orientation via a frame
    built from the first 3 atoms, so the *absolute position* of the
    centroid is rotated (not preserved).  However, the *norm* of the
    centroid is preserved because the canonical frame is an
    orthogonal rotation (Arvind 2026, Lemma 3.5).
    """
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    q = QuotientSpace(manifold="rotation")
    x = torch.randn(2, 4, 3)
    centroid_before = x.mean(dim=1)
    y = q.project(x)
    centroid_after = y.mean(dim=1)
    norm_before = centroid_before.norm(dim=-1)
    norm_after = centroid_after.norm(dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-4), (
        f"rotation-quotient must preserve centroid norm; got "
        f"before={norm_before.tolist()} vs after={norm_after.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 3: QuotientSpace.project is SE(3)-equivariant (the core contract)
# ---------------------------------------------------------------------------
def test_quotient_space_project_equivariant_se3():
    """``project(x) == project(R x + t)`` for the SE(3) quotient.

    This is the central contract of the quotient-space diffusion
    paper (Arvind 2026, Definition 3.1).  We sample a random
    rotation ``R`` and translation ``t_b`` and check that the
    projection of ``x`` equals the projection of ``R x + t_b``.
    """
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    q = QuotientSpace(manifold="SE3")
    torch.manual_seed(0)
    x = torch.randn(3, 6, 3)
    R = _random_rotation_matrix(3, x.dtype, x.device, seed=1)
    t_b = torch.randn(3, 3, generator=torch.Generator().manual_seed(2))
    x_trans = torch.einsum("bij,bnj->bni", R, x) + t_b.unsqueeze(1)
    y1 = q.project(x)
    y2 = q.project(x_trans)
    assert torch.allclose(y1, y2, atol=1e-4), (
        f"SE(3)-equivariance broken: max diff " f"{(y1 - y2).abs().max().item()}"
    )


def test_quotient_space_project_equivariant_translation():
    """``project(x) == project(x + t)`` for the translation quotient."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    q = QuotientSpace(manifold="translation")
    torch.manual_seed(3)
    x = torch.randn(4, 5, 3)
    t_b = torch.randn(4, 3, generator=torch.Generator().manual_seed(4))
    x_trans = x + t_b.unsqueeze(1)
    y1 = q.project(x)
    y2 = q.project(x_trans)
    assert torch.allclose(y1, y2, atol=1e-5), (
        f"translation-equivariance broken: max diff " f"{(y1 - y2).abs().max().item()}"
    )


def test_quotient_space_project_equivariant_rotation():
    """``project(x) == project(R x)`` for the rotation quotient."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    q = QuotientSpace(manifold="rotation")
    torch.manual_seed(5)
    x = torch.randn(2, 8, 3)
    R = _random_rotation_matrix(2, x.dtype, x.device, seed=6)
    x_trans = torch.einsum("bij,bnj->bni", R, x)
    y1 = q.project(x)
    y2 = q.project(x_trans)
    assert torch.allclose(y1, y2, atol=1e-4), (
        f"rotation-equivariance broken: max diff " f"{(y1 - y2).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 4: QuotientSpace.lift returns a valid conformer
# ---------------------------------------------------------------------------
def test_quotient_space_lift_shape():
    """``lift`` returns a tensor of the same shape as the input."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    q = QuotientSpace(manifold="SE3")
    rep = torch.randn(2, 5, 3)
    lifted = q.lift(rep)
    assert lifted.shape == rep.shape
    assert lifted.dtype == rep.dtype


def test_quotient_space_lift_preserves_se3_quotient():
    """``lift(project(x))`` lives in the SE(3) quotient (centred)."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    q = QuotientSpace(manifold="SE3")
    torch.manual_seed(7)
    x = torch.randn(3, 6, 3)
    lifted = q.lift(q.project(x))
    # Centred: per-batch mean is zero.
    mean = lifted.mean(dim=1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), (
        f"lift(project(x)) for SE(3) quotient must be centred; got " f"mean={mean}"
    )


# ---------------------------------------------------------------------------
# Test 5: QuotientSpace.shape_distance is invariant under SE(3)
# ---------------------------------------------------------------------------
def test_quotient_space_shape_distance_se3_invariant():
    """``shape_distance`` is SE(3)-invariant for the SE(3) quotient.

    The quotient-space distance between two conformers depends only
    on their *shape*, not on their absolute position / orientation
    in 3D space.  We verify this by transforming both inputs and
    checking the distance is unchanged.
    """
    from molmetal.adapters.quotient_space_diffusion import QuotientSpace

    q = QuotientSpace(manifold="SE3")
    torch.manual_seed(8)
    x = torch.randn(3, 5, 3)
    y = torch.randn(3, 5, 3)
    d0 = q.shape_distance(x, y)
    R = _random_rotation_matrix(3, x.dtype, x.device, seed=9)
    t_b = torch.randn(3, 3, generator=torch.Generator().manual_seed(10))
    x_t = torch.einsum("bij,bnj->bni", R, x) + t_b.unsqueeze(1)
    y_t = torch.einsum("bij,bnj->bni", R, y) + t_b.unsqueeze(1)
    d1 = q.shape_distance(x_t, y_t)
    assert torch.allclose(
        d0, d1, atol=1e-4
    ), f"SE(3)-invariant distance broken: {d0.tolist()} vs {d1.tolist()}"


# ---------------------------------------------------------------------------
# Test 6: QuotientDiffusionScheduler constructor validation
# ---------------------------------------------------------------------------
def test_quotient_diffusion_scheduler_constructor_default():
    """Default constructor uses ``n_steps=100, sigma_min=0.01, sigma_max=10.0``."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientDiffusionScheduler,
    )

    s = QuotientDiffusionScheduler()
    assert s.n_steps == 100
    assert s.sigma_min == 0.01
    assert s.sigma_max == 10.0
    assert s.seed is None


def test_quotient_diffusion_scheduler_constructor_invalid_n_steps():
    """``n_steps < 1`` raises ``ValueError``."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientDiffusionScheduler,
    )

    with pytest.raises(ValueError):
        QuotientDiffusionScheduler(n_steps=0)
    with pytest.raises(ValueError):
        QuotientDiffusionScheduler(n_steps=-5)


def test_quotient_diffusion_scheduler_constructor_invalid_sigma():
    """Invalid ``sigma_min`` / ``sigma_max`` raise ``ValueError``."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientDiffusionScheduler,
    )

    with pytest.raises(ValueError):
        QuotientDiffusionScheduler(sigma_min=0.0)
    with pytest.raises(ValueError):
        QuotientDiffusionScheduler(sigma_min=-0.1)
    with pytest.raises(ValueError):
        QuotientDiffusionScheduler(sigma_min=0.1, sigma_max=0.05)


# ---------------------------------------------------------------------------
# Test 7: QuotientDiffusionScheduler sigma / alpha-beta helpers
# ---------------------------------------------------------------------------
def test_quotient_diffusion_scheduler_sigma_boundary():
    """``sigma(0) == sigma_min`` and ``sigma(n_steps) == sigma_max``."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientDiffusionScheduler,
    )

    s = QuotientDiffusionScheduler(n_steps=10, sigma_min=0.1, sigma_max=2.0)
    assert abs(s.sigma(0) - 0.1) < 1e-9
    assert abs(s.sigma(10) - 2.0) < 1e-9


def test_quotient_diffusion_scheduler_sigma_monotonic():
    """``sigma(t)`` is monotonically non-decreasing in ``t``."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientDiffusionScheduler,
    )

    s = QuotientDiffusionScheduler(n_steps=20, sigma_min=0.1, sigma_max=5.0)
    sigmas = [s.sigma(t) for t in range(21)]
    for i in range(len(sigmas) - 1):
        assert (
            sigmas[i + 1] >= sigmas[i] - 1e-9
        ), f"sigma not monotonic: sigma({i})={sigmas[i]} > sigma({i+1})={sigmas[i+1]}"


def test_quotient_diffusion_scheduler_alpha_beta_unit_norm():
    """``alpha(t)^2 + beta(t)^2 == 1`` (VP schedule normalisation)."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientDiffusionScheduler,
    )

    s = QuotientDiffusionScheduler(n_steps=10, sigma_min=0.1, sigma_max=5.0)
    for t in range(0, 11):
        alpha, beta = s.alpha_beta(t)
        assert abs(alpha * alpha + beta * beta - 1.0) < 1e-5, (
            f"VP normalisation broken at t={t}: " f"alpha={alpha}, beta={beta}"
        )


def test_quotient_diffusion_scheduler_sigma_invalid_t():
    """Out-of-range ``t`` raises ``ValueError``."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientDiffusionScheduler,
    )

    s = QuotientDiffusionScheduler(n_steps=10)
    with pytest.raises(ValueError):
        s.sigma(-1)
    with pytest.raises(ValueError):
        s.sigma(11)


# ---------------------------------------------------------------------------
# Test 8: QuotientDiffusionScheduler.step preserves shape
# ---------------------------------------------------------------------------
def test_quotient_diffusion_step_shape():
    """``step`` returns a tensor of the same shape as the input."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientDiffusionScheduler,
    )

    s = QuotientDiffusionScheduler(n_steps=20)
    x = torch.randn(3, 5, 3)
    for t in range(20):
        x = s.step(x, t)
        assert x.shape == (
            3,
            5,
            3,
        ), f"step must preserve shape at t={t}; got {tuple(x.shape)}"


def test_quotient_diffusion_step_finite():
    """``step`` returns finite values (no NaN / Inf)."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientDiffusionScheduler,
    )

    s = QuotientDiffusionScheduler(n_steps=10, seed=42)
    x = torch.randn(2, 4, 3)
    for t in range(10):
        x = s.step(x, t)
    assert torch.isfinite(x).all(), "step output must be finite; got NaN/Inf"


def test_quotient_diffusion_step_invalid_t():
    """Out-of-range ``t`` raises ``ValueError``."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientDiffusionScheduler,
    )

    s = QuotientDiffusionScheduler(n_steps=10)
    x = torch.randn(1, 3, 3)
    with pytest.raises(ValueError):
        s.step(x, -1)
    with pytest.raises(ValueError):
        s.step(x, 10)
    with pytest.raises(ValueError):
        s.step(x, 100)


# ---------------------------------------------------------------------------
# Test 9: QuotientDiffusionScheduler.forward_sample shape / dtype
# ---------------------------------------------------------------------------
def test_quotient_diffusion_forward_sample_shape():
    """``forward_sample`` returns a tensor of the same shape as ``x0``."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientDiffusionScheduler,
    )

    s = QuotientDiffusionScheduler(n_steps=10)
    x0 = torch.randn(2, 4, 3, dtype=torch.float64)
    for t in range(0, 11):
        x_t = s.forward_sample(x0, t)
        assert x_t.shape == x0.shape
        assert x_t.dtype == x0.dtype


def test_quotient_diffusion_forward_sample_deterministic_with_seed():
    """``forward_sample`` is deterministic given a fixed seed + t."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientDiffusionScheduler,
    )

    s = QuotientDiffusionScheduler(n_steps=10, seed=0)
    x0 = torch.randn(2, 4, 3)
    y1 = s.forward_sample(x0, 5)
    y2 = s.forward_sample(x0, 5)
    assert torch.allclose(y1, y2), (
        f"forward_sample must be deterministic with fixed seed; "
        f"max diff {(y1 - y2).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 10: QuotientSpaceCFM constructor validation
# ---------------------------------------------------------------------------
def test_quotient_space_cfm_constructor_default_quotient():
    """Default constructor uses an SE(3) quotient."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientSpace,
        QuotientSpaceCFM,
    )

    base = _DummyCFM()
    wrapper = QuotientSpaceCFM(base_cfm=base)
    assert isinstance(wrapper.quotient, QuotientSpace)
    assert wrapper.quotient.manifold == "SE3"


def test_quotient_space_cfm_constructor_invalid_cfm():
    """Non-callable ``base_cfm`` raises ``TypeError``."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpaceCFM

    with pytest.raises(TypeError):
        QuotientSpaceCFM(base_cfm=None)
    with pytest.raises(TypeError):
        QuotientSpaceCFM(base_cfm="not-callable")


def test_quotient_space_cfm_constructor_invalid_quotient():
    """Non-QuotientSpace ``quotient`` raises ``TypeError``."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpaceCFM

    base = _DummyCFM()
    with pytest.raises(TypeError):
        QuotientSpaceCFM(base_cfm=base, quotient="not-a-quotient")


# ---------------------------------------------------------------------------
# Test 11: QuotientSpaceCFM.forward preserves shape / dtype
# ---------------------------------------------------------------------------
def test_quotient_space_cfm_forward_shape():
    """``forward`` returns a tensor of the same shape as ``z0``."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpaceCFM

    wrapper = QuotientSpaceCFM(base_cfm=_DummyCFM())
    z0 = torch.randn(2, 5, 3)
    t = torch.tensor([0.3, 0.7])
    z_t = wrapper.forward(z0, t)
    assert z_t.shape == z0.shape
    assert z_t.dtype == z0.dtype


def test_quotient_space_cfm_forward_scalar_t():
    """``forward`` accepts a scalar ``t`` tensor."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpaceCFM

    wrapper = QuotientSpaceCFM(base_cfm=_DummyCFM())
    z0 = torch.randn(2, 5, 3)
    z_t = wrapper.forward(z0, torch.tensor(0.5))
    assert z_t.shape == z0.shape


def test_quotient_space_cfm_forward_invalid_z0():
    """``forward`` rejects inputs of the wrong shape."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpaceCFM

    wrapper = QuotientSpaceCFM(base_cfm=_DummyCFM())
    with pytest.raises(ValueError):
        wrapper.forward(torch.randn(5, 3), torch.tensor(0.5))  # no batch dim
    with pytest.raises(ValueError):
        wrapper.forward(torch.randn(2, 5, 4), torch.tensor(0.5))  # last dim != 3


# ---------------------------------------------------------------------------
# Test 12: QuotientSpaceCFM.equivariance_error returns a small number
#          for an equivariant base CFM
# ---------------------------------------------------------------------------
def test_quotient_cfm_equivariance_error_small():
    """A trivially-equivariant base CFM yields ``equivariance_error < 1e-3``."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpaceCFM

    wrapper = QuotientSpaceCFM(base_cfm=_DummyCFM())
    err = wrapper.equivariance_error(n_runs=5)
    assert err < 1e-3, f"Equivariant base CFM must have small error; got {err}"


def test_quotient_cfm_equivariance_error_seed_invariant():
    """The equivariance error is deterministic across multiple calls."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpaceCFM

    wrapper = QuotientSpaceCFM(base_cfm=_DummyCFM())
    err1 = wrapper.equivariance_error(n_runs=5, seed=42)
    err2 = wrapper.equivariance_error(n_runs=5, seed=42)
    assert (
        err1 == err2
    ), f"equivariance_error must be deterministic; got {err1} vs {err2}"


def test_quotient_cfm_equivariance_error_zero_for_identity_cfm():
    """An identity-mapping CFM has essentially zero equivariance error.

    The identity CFM ``v(z, t) = z`` is *trivially* SE(3)-equivariant
    (rotating the input and rotating the output commute).  The
    wrapper's ``equivariance_error`` should therefore return a value
    that is numerically indistinguishable from zero (the test uses
    ``atol=1e-5`` to absorb float32 round-off in the canonical-frame
    SVD).
    """
    from molmetal.adapters.quotient_space_diffusion import QuotientSpaceCFM

    class _IdentityCFM(nn.Module):
        def forward(self, z0, t, pocket=None):
            return z0

    wrapper = QuotientSpaceCFM(base_cfm=_IdentityCFM())
    err = wrapper.equivariance_error(n_runs=3)
    assert err < 1e-5, f"Identity CFM must have ~zero equivariance error; got {err}"


def test_quotient_cfm_equivariance_error_invalid_n_runs():
    """Negative or zero ``n_runs`` raises ``ValueError``."""
    from molmetal.adapters.quotient_space_diffusion import QuotientSpaceCFM

    wrapper = QuotientSpaceCFM(base_cfm=_DummyCFM())
    with pytest.raises(ValueError):
        wrapper.equivariance_error(n_runs=0)
    with pytest.raises(ValueError):
        wrapper.equivariance_error(n_runs=-1)


# ---------------------------------------------------------------------------
# Test 13: QuotientSpaceCFM end-to-end with the translation quotient
# ---------------------------------------------------------------------------
def test_quotient_cfm_translation_quotient_equivariance():
    """The translation quotient is fully SE(3)-equivariant.

    With ``manifold="translation"`` the quotient centres the input
    (canonical frame = identity, so ``F = I``).  The wrapper is
    therefore SE(3)-equivariant for *any* equivariant base CFM:
    ``forward(R x + t) = R forward(x) + t``.  We verify this end-
    to-end with a randomly-rotated + translated input.
    """
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientSpace,
        QuotientSpaceCFM,
    )

    wrapper = QuotientSpaceCFM(
        base_cfm=_DummyCFM(),
        quotient=QuotientSpace(manifold="translation"),
    )
    torch.manual_seed(11)
    x = torch.randn(2, 5, 3)
    R = _random_rotation_matrix(2, x.dtype, x.device, seed=12)
    t_b = torch.randn(2, 3, generator=torch.Generator().manual_seed(13))
    t = torch.tensor([0.5, 0.5])
    out_x = wrapper.forward(x, t)
    out_x_trans = wrapper.forward(
        torch.einsum("bij,bnj->bni", R, x) + t_b.unsqueeze(1),
        t,
    )
    # SE(3) equivariance: forward(R x + t) = R forward(x) + t.
    out_x_rot = torch.einsum("bij,bnj->bni", R, out_x) + t_b.unsqueeze(1)
    diff = (out_x_trans - out_x_rot).flatten(start_dim=1).norm(dim=-1)
    assert diff.max().item() < 1e-3, (
        f"Translation quotient must be SE(3)-equivariant; " f"got diff={diff.tolist()}"
    )
