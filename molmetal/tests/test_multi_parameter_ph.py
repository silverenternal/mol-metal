"""Tests for :mod:`molmetal.math_skills.multi_parameter_ph`.

T5 — Multi-Parameter Persistent Homology (Kim & Jung 2026)
=============================================================

These tests cover the three mathematical primitives shipped by the
T5 multi-parameter persistent homology skeleton:

1. **TwoParameterFiltration** — 2-parameter Vietoris-Rips filtration.
   We verify construction (input validation, default values), the
   ``compute`` shape and finiteness contracts, SE(3) invariance,
   single-atom degeneracy and the auto-promotion of ``(B, N, 3)``
   coords with the carbon default.
2. **MultiPHKernel** — multi-PH kernel SVM head.  We verify the
   three kernel families (``sliced_wasserstein``, ``rbf``,
   ``cosine``), the unit-diagonal property, the symmetry of the
   Gram matrix, the input validation and the
   ``B_x != B_y`` asymmetric ``kernel_matrix`` path.
3. **MultiPHSummarizer** — multi-PH summary replacing the
   single-parameter Betti curve.  We verify construction, the
   multi-channel output shape, the per-channel finiteness, the
   ``multi_betti_curve`` and ``marginal_betti`` helpers, and the
   backward-compatibility path on ``(B, n_bins, n_bins)`` inputs.

Every test is **CPU-only** and uses small data (batch sizes 1–6,
``n_bins`` 4–16, point clouds with 6–10 atoms) so the full suite
runs in < 5 seconds.

Reference
---------
Kim, M.-J. & Jung, S.-H. (2026).  *Fourier-Accelerated Computation
of Multi-Parameter Persistent Homology for Molecular Shape
Analysis*.  Adv. Appl. Math. 174:103023.
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
    MultiPHKernel,
    MultiPHSummarizer,
    TwoParameterDiagram,
    TwoParameterFiltration,
    sliced_wasserstein_kernel,
    two_param_filtration,
)
from molmetal.math_skills.multi_parameter_ph import (
    _EPS_LOG,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _toy_clouds(
    b: int = 2, n: int = 8, seed: int = 0, with_z: bool = True,
) -> torch.Tensor:
    """Deterministic ``(B, N, 4)`` or ``(B, N, 3)`` point clouds."""
    g = torch.Generator().manual_seed(int(seed))
    coords = torch.randn(b, n, 3, generator=g) * 2.0
    if with_z:
        atom_z = torch.randint(1, 18, (b, n), generator=g).float()
        return torch.cat([coords, atom_z.unsqueeze(-1)], dim=-1)
    return coords


def _toy_feature_maps(
    b: int = 2, n_bins: int = 8, seed: int = 0,
) -> torch.Tensor:
    """Deterministic ``(B, n_bins, n_bins)`` feature maps."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.rand(b, n_bins, n_bins, generator=g) * 5.0


def _rotation_matrix_z(angle: float) -> torch.Tensor:
    """A rotation about the ``z`` axis by ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


# ---------------------------------------------------------------------------
# 1. TwoParameterFiltration — construction & invariants
# ---------------------------------------------------------------------------


def test_two_param_filtration_default_construction() -> None:
    """Default constructor accepts the documented defaults."""
    filt = TwoParameterFiltration()
    assert filt.max_radius == 5.0
    assert filt.max_coord == 4.0
    assert filt.n_bins == 32


def test_two_param_filtration_rejects_zero_max_radius() -> None:
    """``__init__`` rejects non-positive ``max_radius``."""
    with pytest.raises(ValueError, match="max_radius"):
        TwoParameterFiltration(max_radius=0.0)
    with pytest.raises(ValueError, match="max_radius"):
        TwoParameterFiltration(max_radius=-1.0)


def test_two_param_filtration_rejects_zero_max_coord() -> None:
    """``__init__`` rejects non-positive ``max_coord``."""
    with pytest.raises(ValueError, match="max_coord"):
        TwoParameterFiltration(max_coord=0.0)
    with pytest.raises(ValueError, match="max_coord"):
        TwoParameterFiltration(max_coord=-2.0)


def test_two_param_filtration_rejects_too_few_bins() -> None:
    """``__init__`` rejects ``n_bins < 2``."""
    with pytest.raises(ValueError, match="n_bins"):
        TwoParameterFiltration(n_bins=1)
    with pytest.raises(ValueError, match="n_bins"):
        TwoParameterFiltration(n_bins=0)


def test_two_param_filtration_compute_shape() -> None:
    """``compute`` returns a tensor of shape ``(B, n_bins, n_bins)``."""
    filt = TwoParameterFiltration(max_radius=5.0, max_coord=4.0, n_bins=16)
    clouds = _toy_clouds(b=3, n=8, seed=0)
    out = filt.compute(clouds)
    assert out.shape == (3, 16, 16), (
        f"expected (3, 16, 16), got {tuple(out.shape)}"
    )


def test_two_param_filtration_compute_finite_non_negative() -> None:
    """The ``compute`` output is finite and non-negative (it's a count)."""
    filt = TwoParameterFiltration(n_bins=12)
    clouds = _toy_clouds(b=4, n=10, seed=2)
    out = filt.compute(clouds)
    assert torch.isfinite(out).all(), "feature map contains NaN/Inf"
    assert (out >= 0).all(), "feature map must be non-negative"


def test_two_param_filtration_compute_3d_auto_promotes_carbon() -> None:
    """A ``(B, N, 3)`` cloud is auto-promoted with ``z = 6`` (carbon)."""
    filt = TwoParameterFiltration(n_bins=10)
    coords3 = _toy_clouds(b=2, n=6, seed=4, with_z=False)
    out3 = filt.compute(coords3)
    # Equivalent construction with explicit carbon atom types must
    # match the auto-promoted output bit-for-bit.
    coords4 = torch.cat(
        [coords3, torch.full((2, 6, 1), 6.0)], dim=-1,
    )
    out4 = filt.compute(coords4)
    torch.testing.assert_close(out3, out4, rtol=1e-6, atol=1e-6)


def test_two_param_filtration_rotation_invariant() -> None:
    """A global rotation of the pocket does NOT change the feature map."""
    filt = TwoParameterFiltration(n_bins=12)
    clouds = _toy_clouds(b=2, n=8, seed=5)
    R = _rotation_matrix_z(math.pi / 4)
    rotated_coords = clouds[..., :3] @ R.T
    clouds_rot = torch.cat(
        [rotated_coords, clouds[..., 3:4]], dim=-1,
    )
    out = filt.compute(clouds)
    out_rot = filt.compute(clouds_rot)
    torch.testing.assert_close(out, out_rot, rtol=1e-5, atol=1e-5)


def test_two_param_filtration_translation_invariant() -> None:
    """A global translation of the pocket does NOT change the feature map."""
    filt = TwoParameterFiltration(n_bins=12)
    clouds = _toy_clouds(b=2, n=8, seed=7)
    shift = torch.tensor([10.0, -3.0, 7.0])
    coords_shift = clouds[..., :3] + shift
    clouds_shift = torch.cat(
        [coords_shift, clouds[..., 3:4]], dim=-1,
    )
    out = filt.compute(clouds)
    out_shift = filt.compute(clouds_shift)
    torch.testing.assert_close(out, out_shift, rtol=1e-5, atol=1e-5)


def test_two_param_filtration_single_atom_returns_zero() -> None:
    """A single-atom pocket (no edges) returns the all-zero feature map."""
    filt = TwoParameterFiltration(n_bins=8)
    clouds = _toy_clouds(b=3, n=1, seed=0)
    out = filt.compute(clouds)
    assert out.shape == (3, 8, 8)
    assert torch.allclose(out, torch.zeros_like(out))


def test_two_param_filtration_rejects_wrong_clouds_shape() -> None:
    """``compute`` rejects ``clouds`` with the wrong rank / last-dim."""
    filt = TwoParameterFiltration(n_bins=4)
    # 2-D rank: rejected.
    with pytest.raises(ValueError):
        filt.compute(torch.randn(2, 5))
    # Last dim != 3, 4: rejected.
    with pytest.raises(ValueError):
        filt.compute(torch.randn(2, 5, 5))


def test_two_param_filtration_grid_helpers() -> None:
    """``r_centers`` / ``d_centers`` have the right shape and bounds."""
    filt = TwoParameterFiltration(max_radius=4.0, max_coord=2.0, n_bins=10)
    rc = filt.r_centers()
    dc = filt.d_centers()
    assert rc.shape == (10,) and dc.shape == (10,)
    assert rc[0] > 0.0 and rc[-1] < 4.0
    assert dc[0] > 0.0 and dc[-1] < 2.0


def test_two_param_filtration_compute_with_atom_types_equivalent() -> None:
    """``compute_with_atom_types`` agrees with the function API."""
    filt = TwoParameterFiltration(n_bins=10)
    clouds = _toy_clouds(b=2, n=6, seed=3)
    coords = clouds[..., :3]
    atom_types = clouds[..., 3].long()
    out_class = filt.compute_with_atom_types(coords, atom_types)
    out_func = two_param_filtration(
        coords, atom_types,
        max_radius=5.0, max_coord=4.0, n_bins=10,
    )["features"]
    torch.testing.assert_close(out_class, out_func, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. MultiPHKernel — kernel SVM head
# ---------------------------------------------------------------------------


def test_multi_ph_kernel_default_construction() -> None:
    """Default constructor uses ``sliced_wasserstein`` + sigma=1.0."""
    kp = MultiPHKernel()
    assert kp.kernel == "sliced_wasserstein"
    assert kp.sigma == 1.0
    assert kp.n_projections == 8


def test_multi_ph_kernel_rejects_unknown_kernel() -> None:
    """An unknown kernel name raises ``ValueError``."""
    with pytest.raises(ValueError, match="kernel"):
        MultiPHKernel(kernel="laplace")


def test_multi_ph_kernel_rejects_non_positive_sigma() -> None:
    """A non-positive ``sigma`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="sigma"):
        MultiPHKernel(sigma=0.0)
    with pytest.raises(ValueError, match="sigma"):
        MultiPHKernel(sigma=-1.0)


def test_multi_ph_kernel_rejects_zero_projections() -> None:
    """``n_projections = 0`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="n_projections"):
        MultiPHKernel(n_projections=0)


def test_multi_ph_kernel_matrix_shape_sliced_wasserstein() -> None:
    """``kernel_matrix`` returns ``(B_x, B_y)`` with ``sliced_wasserstein``."""
    kp = MultiPHKernel(
        kernel="sliced_wasserstein", sigma=2.0, n_projections=4, seed=0,
    )
    x = _toy_feature_maps(b=3, n_bins=8, seed=1)
    y = _toy_feature_maps(b=5, n_bins=8, seed=2)
    K = kp.kernel_matrix(x, y)
    assert K.shape == (3, 5), f"expected (3, 5), got {tuple(K.shape)}"


def test_multi_ph_kernel_matrix_shape_rbf() -> None:
    """``kernel_matrix`` returns ``(B_x, B_y)`` with ``rbf``."""
    kp = MultiPHKernel(kernel="rbf", sigma=2.0)
    x = _toy_feature_maps(b=3, n_bins=6, seed=1)
    y = _toy_feature_maps(b=4, n_bins=6, seed=2)
    K = kp.kernel_matrix(x, y)
    assert K.shape == (3, 4)


def test_multi_ph_kernel_matrix_shape_cosine() -> None:
    """``kernel_matrix`` returns ``(B_x, B_y)`` with ``cosine``."""
    kp = MultiPHKernel(kernel="cosine")
    x = _toy_feature_maps(b=3, n_bins=6, seed=1)
    y = _toy_feature_maps(b=2, n_bins=6, seed=2)
    K = kp.kernel_matrix(x, y)
    assert K.shape == (3, 2)


def test_multi_ph_kernel_gram_diagonal_is_one() -> None:
    """``kernel_matrix(X, X)`` has unit diagonal for all three kernels."""
    x = _toy_feature_maps(b=4, n_bins=6, seed=3)
    for kernel in ("sliced_wasserstein", "rbf", "cosine"):
        kp = MultiPHKernel(
            kernel=kernel, sigma=2.0, n_projections=4, seed=0,
        )
        K = kp.kernel_matrix(x)
        diag = torch.diagonal(K)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4), (
            f"{kernel}: diagonal must be 1, got {diag.tolist()}"
        )


def test_multi_ph_kernel_gram_is_symmetric() -> None:
    """The Gram matrix ``kernel_matrix(X, X)`` is symmetric."""
    x = _toy_feature_maps(b=4, n_bins=8, seed=3)
    for kernel in ("sliced_wasserstein", "rbf", "cosine"):
        kp = MultiPHKernel(
            kernel=kernel, sigma=2.0, n_projections=4, seed=0,
        )
        K = kp.kernel_matrix(x)
        assert torch.allclose(K, K.T, atol=1e-5), (
            f"{kernel}: Gram matrix not symmetric"
        )


def test_multi_ph_kernel_asymmetric_bx_by_shape() -> None:
    """``kernel_matrix`` accepts ``B_x != B_y`` and returns the right shape."""
    kp = MultiPHKernel(kernel="rbf", sigma=1.0)
    x = _toy_feature_maps(b=1, n_bins=6, seed=1)
    y = _toy_feature_maps(b=10, n_bins=6, seed=2)
    K = kp.kernel_matrix(x, y)
    assert K.shape == (1, 10)
    # The kernel matrix must still be finite.
    assert torch.isfinite(K).all()


def test_multi_ph_kernel_y_none_equals_gram() -> None:
    """Passing ``y=None`` is equivalent to ``kernel_matrix(x, x)``."""
    kp = MultiPHKernel(kernel="rbf", sigma=1.5)
    x = _toy_feature_maps(b=3, n_bins=6, seed=4)
    K1 = kp.kernel_matrix(x)
    K2 = kp.kernel_matrix(x, x)
    torch.testing.assert_close(K1, K2, rtol=1e-6, atol=1e-6)


def test_multi_ph_kernel_rbf_matches_rbf_formula() -> None:
    """The RBF kernel value equals ``exp(-||x - y||² / σ²)`` on flat maps."""
    kp = MultiPHKernel(kernel="rbf", sigma=2.0)
    x = _toy_feature_maps(b=2, n_bins=4, seed=5)
    y = _toy_feature_maps(b=1, n_bins=4, seed=6)
    K = kp.kernel_matrix(x, y)
    # Manual cross-check: row 0, col 0.
    xf = x[0].reshape(-1)
    yf = y[0].reshape(-1)
    expected = torch.exp(
        -((xf - yf) ** 2).sum() / (2.0 * 2.0 + 1e-12)
    )
    assert torch.allclose(K[0, 0], expected, atol=1e-4), (
        f"expected {expected.item()}; got {K[0, 0].item()}"
    )


def test_multi_ph_kernel_cosine_in_unit_interval() -> None:
    """Cosine kernel values lie in ``[0, 1]`` for any pair."""
    kp = MultiPHKernel(kernel="cosine")
    x = _toy_feature_maps(b=4, n_bins=6, seed=7)
    K = kp.kernel_matrix(x)
    assert (K >= -1e-5).all() and (K <= 1.0 + 1e-5).all()


def test_multi_ph_kernel_rejects_wrong_input_rank() -> None:
    """``kernel_matrix`` rejects ``x`` / ``y`` with the wrong rank."""
    kp = MultiPHKernel(kernel="rbf", sigma=1.0)
    with pytest.raises(ValueError):
        kp.kernel_matrix(torch.randn(2, 3, 4, 5))  # 4-D rank


# ---------------------------------------------------------------------------
# 3. MultiPHSummarizer — multi-PH summary
# ---------------------------------------------------------------------------


def test_multi_ph_summarizer_default_construction() -> None:
    """Default constructor uses ``n_bins = 32``."""
    summ = MultiPHSummarizer()
    assert summ.n_bins == 32


def test_multi_ph_summarizer_rejects_too_few_bins() -> None:
    """``__init__`` rejects ``n_bins < 2``."""
    with pytest.raises(ValueError, match="n_bins"):
        MultiPHSummarizer(n_bins=1)


def test_multi_ph_summarizer_summarize_shape() -> None:
    """``summarize`` returns ``(B, 4, n_bins, n_bins)``."""
    summ = MultiPHSummarizer(n_bins=10)
    diagrams = _toy_feature_maps(b=3, n_bins=10, seed=1)
    out = summ.summarize([d for d in diagrams])
    assert out.shape == (3, 4, 10, 10), (
        f"expected (3, 4, 10, 10), got {tuple(out.shape)}"
    )


def test_multi_ph_summarizer_accepts_batched_tensor_input() -> None:
    """``summarize`` accepts a ``(B, n_bins, n_bins)`` tensor input."""
    summ = MultiPHSummarizer(n_bins=8)
    diagrams = _toy_feature_maps(b=4, n_bins=8, seed=2)
    out = summ.summarize(diagrams)
    assert out.shape == (4, 4, 8, 8)


def test_multi_ph_summarizer_accepts_single_diagram() -> None:
    """``summarize`` accepts a single ``(n_bins, n_bins)`` diagram."""
    summ = MultiPHSummarizer(n_bins=6)
    diag = _toy_feature_maps(b=1, n_bins=6, seed=3)[0]
    out = summ.summarize([diag])
    assert out.shape == (1, 4, 6, 6)


def test_multi_ph_summarizer_output_is_finite() -> None:
    """All four summary channels are finite."""
    summ = MultiPHSummarizer(n_bins=8)
    diagrams = _toy_feature_maps(b=4, n_bins=8, seed=4)
    out = summ.summarize(diagrams)
    assert torch.isfinite(out).all()


def test_multi_ph_summarizer_multi_betti_curve_shape() -> None:
    """``multi_betti_curve`` returns ``(B, n_bins, n_bins)`` cumulative map."""
    summ = MultiPHSummarizer(n_bins=10)
    diagrams = _toy_feature_maps(b=3, n_bins=10, seed=5)
    mb = summ.multi_betti_curve(diagrams)
    assert mb.shape == (3, 10, 10)


def test_multi_ph_summarizer_multi_betti_curve_is_monotone() -> None:
    """The cumulative multi-Betti curve is monotone non-decreasing in
    both axes."""
    summ = MultiPHSummarizer(n_bins=8)
    diagrams = _toy_feature_maps(b=2, n_bins=8, seed=6)
    mb = summ.multi_betti_curve(diagrams)
    # Check axis-wise: column j+1 >= column j, row i+1 >= row i.
    diff_r = mb[..., :, 1:] - mb[..., :, :-1]
    diff_d = mb[..., 1:, :] - mb[..., :-1, :]
    assert (diff_r >= -1e-6).all(), "multi-Betti curve must increase along r"
    assert (diff_d >= -1e-6).all(), "multi-Betti curve must increase along d"


def test_multi_ph_summarizer_marginal_betti_shape() -> None:
    """``marginal_betti`` returns ``(B, 2, n_bins)``."""
    summ = MultiPHSummarizer(n_bins=10)
    diagrams = _toy_feature_maps(b=3, n_bins=10, seed=7)
    mb = summ.marginal_betti(diagrams)
    assert mb.shape == (3, 2, 10)


def test_multi_ph_summarizer_marginal_betti_equals_sum() -> None:
    """``marginal_betti`` is exactly the per-axis sum of the count grid."""
    summ = MultiPHSummarizer(n_bins=6)
    diagrams = _toy_feature_maps(b=2, n_bins=6, seed=8)
    out = summ.summarize(diagrams)
    marg = summ.marginal_betti(diagrams)
    # Channel 0 is the raw count grid (the persistence surface).
    expected_r = out[:, 0].sum(dim=-1)
    expected_d = out[:, 0].sum(dim=-2)
    torch.testing.assert_close(marg[:, 0], expected_r, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(marg[:, 1], expected_d, rtol=1e-6, atol=1e-6)


def test_multi_ph_summarizer_channel1_equals_cumsum_of_channel0() -> None:
    """Channel 1 of ``summarize`` is the cumulative sum of channel 0
    along both axes."""
    summ = MultiPHSummarizer(n_bins=8)
    diagrams = _toy_feature_maps(b=2, n_bins=8, seed=9)
    out = summ.summarize(diagrams)
    # Cumsum along r (dim=-2) first, then along d (dim=-1).
    expected = torch.cumsum(
        torch.cumsum(out[:, 0], dim=-2), dim=-1,
    )
    torch.testing.assert_close(out[:, 1], expected, rtol=1e-6, atol=1e-6)


def test_multi_ph_summarizer_channel3_zero_mean() -> None:
    """Channel 3 of ``summarize`` is centred to zero mean along rows
    and columns (the persistence surface minus its row+column mean)."""
    summ = MultiPHSummarizer(n_bins=8)
    diagrams = _toy_feature_maps(b=2, n_bins=8, seed=10)
    out = summ.summarize(diagrams)
    # Channel 3 must satisfy: row_mean + col_mean == raw count.
    row_mean = out[:, 3].mean(dim=-1, keepdim=True)
    col_mean = out[:, 3].mean(dim=-2, keepdim=True)
    # The mean is zero by construction (we subtract half the
    # row/col mean so each row/column mean vanishes).
    assert torch.allclose(row_mean, torch.zeros_like(row_mean), atol=1e-4)
    assert torch.allclose(col_mean, torch.zeros_like(col_mean), atol=1e-4)


def test_multi_ph_summarizer_rejects_empty_diagrams() -> None:
    """An empty sequence raises ``ValueError``."""
    summ = MultiPHSummarizer(n_bins=4)
    with pytest.raises(ValueError, match="empty"):
        summ.summarize([])


def test_multi_ph_summarizer_rejects_wrong_diagram_shape() -> None:
    """A diagram with the wrong ``n_bins`` raises ``ValueError``."""
    summ = MultiPHSummarizer(n_bins=8)
    bad = torch.rand(6, 6)  # wrong n_bins
    with pytest.raises(ValueError):
        summ.summarize([bad])


def test_multi_ph_summarizer_rejects_tensor_wrong_rank() -> None:
    """A tensor input with rank > 3 raises ``ValueError``."""
    summ = MultiPHSummarizer(n_bins=8)
    with pytest.raises(ValueError):
        summ.summarize(torch.rand(2, 3, 8, 8))


# ---------------------------------------------------------------------------
# 4. Integration — class-based pipeline matches the function API
# ---------------------------------------------------------------------------


def test_two_param_filtration_then_summarizer_end_to_end() -> None:
    """End-to-end: ``TwoParameterFiltration.compute`` -> ``MultiPHSummarizer``."""
    filt = TwoParameterFiltration(n_bins=12)
    summ = MultiPHSummarizer(n_bins=12)
    clouds = _toy_clouds(b=3, n=8, seed=11)
    feats = filt.compute(clouds)  # (3, 12, 12)
    out = summ.summarize(feats)  # (3, 4, 12, 12)
    assert out.shape == (3, 4, 12, 12)
    assert torch.isfinite(out).all()


def test_two_param_filtration_then_kernel_matrix_end_to_end() -> None:
    """End-to-end: ``TwoParameterFiltration.compute`` -> ``MultiPHKernel``."""
    filt = TwoParameterFiltration(n_bins=10)
    kp = MultiPHKernel(kernel="rbf", sigma=2.0)
    clouds = _toy_clouds(b=4, n=8, seed=12)
    feats = filt.compute(clouds)  # (4, 10, 10)
    K = kp.kernel_matrix(feats)
    assert K.shape == (4, 4)
    # Unit diagonal and symmetry are the cross-check.
    assert torch.allclose(torch.diagonal(K), torch.ones(4), atol=1e-5)
    assert torch.allclose(K, K.T, atol=1e-5)


def test_multi_ph_kernel_x3_vs_x4_diagonal_unchanged() -> None:
    """Computing ``K(X, X)`` on three diagrams vs four must give the
    same top-left ``3 × 3`` block."""
    kp = MultiPHKernel(kernel="rbf", sigma=2.0)
    x3 = _toy_feature_maps(b=3, n_bins=6, seed=13)
    x4 = _toy_feature_maps(b=4, n_bins=6, seed=14)
    K3 = kp.kernel_matrix(x3, x3)
    K4 = kp.kernel_matrix(x4, x4)
    # Use a slightly looser tolerance — the kernel values are
    # ``exp(-||x-y||² / σ²)`` and ``1.5e-5`` is the cumulative
    # fp32 round-off on the largest norm-squared value.
    torch.testing.assert_close(K3, K4[:3, :3], rtol=1e-4, atol=1e-5)
