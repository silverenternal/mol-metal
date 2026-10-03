"""Tests for :mod:`molmetal.math_skills.bergman_kernel`.

T9 — Bergman Kernel on Bounded Symmetric Domains (麻小南)
==========================================================

These tests cover the three mathematical skeleton primitives shipped
by the Bergman-kernel module:

1. **BoundedSymmetricDomain** — the type II_d Siegel disk with the
   Hermitian positive embedding :math:`z(o) = o o^T / (1 + \\|o\\|^2)`.
   Verifies construction (default and custom :math:`d`), the
   dimension properties (real, complex, rank, multiplicity), the
   membership check ``in_domain``, the embedding ``point``
   (symmetry, spectral norm, shape), and the random sampler
   ``random_point``.

2. **BergmanKernel** — the analytic Bergman kernel
   :math:`K(z, w) = \\det(I_d - z w^*)^{-(d+1)}` plus the Nyström
   eigenfunction expansion.  Verifies construction, validation of
   arguments, the kernel axioms (Hermitian symmetry, base value
   :math:`K(0, 0) = 1`, positive-definiteness, self-similarity
   :math:`K(z, z) \\ge 1`), the ``eigenfunctions`` accessor, the
   ``features`` Nyström map, and the ``anchor_norms`` diagnostic.

3. **BergmanKernelEmbedding** — ``nn.Module`` that maps a d-electron
   occupation vector to a ready-to-use feature vector.  Verifies
   construction, output shape, differentiability (gradient flows
   through the embedding), finiteness of outputs, and consistency
   with the lower-level :class:`BergmanKernel`.

Every test is **CPU-only** and uses small data (dimension :math:`d`
in ``[1, 5]``, hidden dim in ``[4, 16]``, batch size in
``[1, 8]``) so the full suite runs in < 1 second.

Reference
---------
Ma, X. (麻小南) (2025).  *Bergman Kernels on Bounded Symmetric
Domains and Applications to Representation Theory*.
Cited as ``ma_xiaonan2025bergman_kernel`` in ``paper/refs.bib``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from molmetal.math_skills import (
    BergmanKernel,
    BergmanKernelEmbedding,
    BoundedSymmetricDomain,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _occupancy(batch: int, d: int, scale: float = 0.5, seed: int = 0) -> torch.Tensor:
    """Sample ``batch`` occupancy vectors of dimension ``d``."""
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(batch, d, generator=g)


# ---------------------------------------------------------------------------
# 1. BoundedSymmetricDomain — construction and dimension properties
# ---------------------------------------------------------------------------


def test_bsd_default_construction() -> None:
    """Default construction: dimension = 5 (d-orbital occupancy)."""
    dom = BoundedSymmetricDomain()
    assert dom.d == 5


def test_bsd_custom_dimension() -> None:
    """Custom dimension is stored."""
    dom = BoundedSymmetricDomain(dimension=3)
    assert dom.d == 3


def test_bsd_rejects_zero_dimension() -> None:
    """Dimension 0 is rejected at construction time."""
    with pytest.raises(ValueError, match="dimension"):
        BoundedSymmetricDomain(dimension=0)


def test_bsd_rejects_negative_dimension() -> None:
    """Negative dimension is rejected at construction time."""
    with pytest.raises(ValueError, match="dimension"):
        BoundedSymmetricDomain(dimension=-2)


def test_bsd_domain_dim_property() -> None:
    """``domain_dim = d(d+1)`` is the real dimension of R_II(d)."""
    for d in (1, 2, 3, 5, 7):
        dom = BoundedSymmetricDomain(dimension=d)
        assert dom.domain_dim == d * (d + 1)


def test_bsd_complex_dim_property() -> None:
    """``complex_dim = d(d+1)/2`` is the complex dimension of R_II(d)."""
    for d in (1, 2, 3, 5, 7):
        dom = BoundedSymmetricDomain(dimension=d)
        assert dom.complex_dim == d * (d + 1) // 2


def test_bsd_rank_property() -> None:
    """``rank = d`` for type II_d."""
    for d in (1, 2, 5):
        dom = BoundedSymmetricDomain(dimension=d)
        assert dom.rank == d


def test_bsd_multiplicity_property() -> None:
    """``multiplicity = d + 1`` — the exponent in K(z, w)."""
    for d in (1, 2, 3, 5):
        dom = BoundedSymmetricDomain(dimension=d)
        assert dom.multiplicity == d + 1


# ---------------------------------------------------------------------------
# 2. BoundedSymmetricDomain — in_domain check
# ---------------------------------------------------------------------------


def test_bsd_in_domain_zeros() -> None:
    """The zero matrix is in the domain."""
    dom = BoundedSymmetricDomain(dimension=5)
    z = torch.zeros(3, 5, 5)
    mask = dom.in_domain(z)
    assert mask.shape == (3,)
    assert bool(mask.all())


def test_bsd_in_domain_random_points() -> None:
    """Random Hermitian-positive embeddings lie strictly inside the disk."""
    dom = BoundedSymmetricDomain(dimension=5)
    occ = _occupancy(batch=8, d=5)
    z = dom.point(occ)
    mask = dom.in_domain(z)
    assert bool(mask.all())


def test_bsd_in_domain_rejects_boundary_and_outside() -> None:
    """A matrix with spectral norm >= 1 is flagged as out-of-domain."""
    dom = BoundedSymmetricDomain(dimension=3)
    # 1.5 * I has spectral norm 1.5 — outside the disk.
    z = 1.5 * torch.eye(3).unsqueeze(0)  # (1, 3, 3)
    mask = dom.in_domain(z)
    assert bool((~mask).all())


# ---------------------------------------------------------------------------
# 3. BoundedSymmetricDomain — point embedding
# ---------------------------------------------------------------------------


def test_bsd_point_zero_occupancy_gives_zero() -> None:
    """``point(o=0) == 0`` — the Hermitian positive embed maps 0 to 0."""
    dom = BoundedSymmetricDomain(dimension=5)
    o = torch.zeros(3, 5)
    z = dom.point(o)
    assert torch.allclose(z, torch.zeros_like(z), atol=1e-7)


def test_bsd_point_is_symmetric() -> None:
    """``point(o)`` is exactly symmetric (Z = Z^T)."""
    dom = BoundedSymmetricDomain(dimension=4)
    o = _occupancy(batch=6, d=4)
    z = dom.point(o)
    assert torch.allclose(z, z.transpose(-1, -2), atol=1e-7)


def test_bsd_point_is_positive_semidefinite() -> None:
    """``point(o)`` has non-negative eigenvalues (rank-1 PSD)."""
    dom = BoundedSymmetricDomain(dimension=5)
    o = _occupancy(batch=4, d=5)
    z = dom.point(o)
    eigvals = torch.linalg.eigvalsh(z)
    assert (eigvals >= -1e-6).all()


def test_bsd_point_spectral_norm_less_than_one() -> None:
    """The spectral norm of ``point(o)`` is < 1 for any real ``o``."""
    dom = BoundedSymmetricDomain(dimension=5)
    # Try various magnitudes, including large ones.
    for scale in (0.1, 1.0, 5.0, 100.0):
        o = scale * torch.randn(8, 5)
        z = dom.point(o)
        s = torch.linalg.svdvals(z)
        assert (s.max(dim=-1).values < 1.0).all()


def test_bsd_point_broadcasts_over_batch() -> None:
    """``point`` accepts arbitrary leading batch dims."""
    dom = BoundedSymmetricDomain(dimension=5)
    # 3-D batched occupancy.
    o = torch.randn(2, 3, 5)
    z = dom.point(o)
    assert z.shape == (2, 3, 5, 5)
    assert torch.allclose(z, z.transpose(-1, -2), atol=1e-7)


def test_bsd_random_point_in_domain() -> None:
    """``random_point`` samples points strictly inside the domain."""
    dom = BoundedSymmetricDomain(dimension=5)
    g = torch.Generator().manual_seed(42)
    z = dom.random_point((10,), generator=g)
    assert z.shape == (10, 5, 5)
    assert bool(dom.in_domain(z).all())


# ---------------------------------------------------------------------------
# 4. BergmanKernel — construction and validation
# ---------------------------------------------------------------------------


def test_bergman_kernel_default_construction() -> None:
    """``BergmanKernel(d=5, n_eigenfunctions=64)`` constructs without error."""
    kern = BergmanKernel(dimension=5, n_eigenfunctions=64)
    assert kern.d == 5
    assert kern.n_eigenfunctions == 64
    assert kern.multiplicity == 6


def test_bergman_kernel_buffers_have_correct_shape() -> None:
    """Anchor buffer has shape ``(n_eigenfunctions, d, d)``."""
    kern = BergmanKernel(dimension=4, n_eigenfunctions=8)
    assert kern._anchors.shape == (8, 4, 4)


def test_bergman_kernel_anchors_are_in_domain() -> None:
    """The Nyström anchors are interior points of R_II(d)."""
    kern = BergmanKernel(dimension=5, n_eigenfunctions=16)
    dom = BoundedSymmetricDomain(dimension=5)
    assert bool(dom.in_domain(kern._anchors).all())


def test_bergman_kernel_rejects_zero_dimension() -> None:
    """Dimension 0 is rejected at construction time."""
    with pytest.raises(ValueError, match="dimension"):
        BergmanKernel(dimension=0, n_eigenfunctions=4)


def test_bergman_kernel_rejects_zero_n_eigenfunctions() -> None:
    """``n_eigenfunctions = 0`` is rejected at construction time."""
    with pytest.raises(ValueError, match="n_eigenfunctions"):
        BergmanKernel(dimension=5, n_eigenfunctions=0)


def test_bergman_kernel_is_seed_deterministic() -> None:
    """Two constructions with the same seed produce identical anchors."""
    kern1 = BergmanKernel(dimension=3, n_eigenfunctions=4, seed=7)
    kern2 = BergmanKernel(dimension=3, n_eigenfunctions=4, seed=7)
    assert torch.allclose(kern1._anchors, kern2._anchors)


# ---------------------------------------------------------------------------
# 5. BergmanKernel — kernel axioms
# ---------------------------------------------------------------------------


def test_bergman_kernel_at_origin_is_one() -> None:
    """``K(0, 0) = 1`` — the Bergman kernel normalises at the origin."""
    kern = BergmanKernel(dimension=5, n_eigenfunctions=4)
    z = torch.zeros(1, 5, 5)
    K = kern.forward(z, z)
    assert K.shape == (1,)
    assert torch.allclose(K, torch.ones_like(K), atol=1e-5)


def test_bergman_kernel_zero_argument_is_one() -> None:
    """``K(0, w) = 1`` and ``K(z, 0) = 1`` for any in-domain ``z``, ``w``."""
    kern = BergmanKernel(dimension=4, n_eigenfunctions=4)
    dom = BoundedSymmetricDomain(dimension=4)
    occ = _occupancy(batch=3, d=4)
    z = dom.point(occ)  # (3, 4, 4)
    z0 = torch.zeros(1, 4, 4)  # (1, 4, 4)
    K_0w = kern.forward(z0, z)  # (1, 3) — K(0, w) = 1
    K_z0 = kern.forward(z, z0)  # (3, 1) — K(z, 0) = 1
    assert torch.allclose(K_0w, torch.ones_like(K_0w), atol=1e-5)
    assert torch.allclose(K_z0, torch.ones_like(K_z0), atol=1e-5)


def test_bergman_kernel_self_similarity_at_least_one() -> None:
    """``K(z, z) >= 1`` for any in-domain ``z`` (unit-norm zonal pol.)."""
    kern = BergmanKernel(dimension=5, n_eigenfunctions=4)
    dom = BoundedSymmetricDomain(dimension=5)
    occ = _occupancy(batch=8, d=5)
    z = dom.point(occ)
    K_zz = kern.forward(z, z)
    assert (K_zz >= 1.0 - 1e-5).all()


def test_bergman_kernel_strictly_positive_in_domain() -> None:
    """``K(z, w) > 0`` for any pair of in-domain points."""
    kern = BergmanKernel(dimension=5, n_eigenfunctions=4)
    dom = BoundedSymmetricDomain(dimension=5)
    occ = _occupancy(batch=6, d=5)
    z = dom.point(occ)
    # Cross-kernel matrix K(z_i, z_j) for i, j in [0, 6).
    z_e = z.unsqueeze(-3)  # (6, 1, 5, 5)
    z_t = z.unsqueeze(0)  # (1, 6, 5, 5)
    K = kern.forward(z_e, z_t)  # (6, 6)
    assert (K > 0.0).all()
    assert torch.isfinite(K).all()


def test_bergman_kernel_is_symmetric_real() -> None:
    """``K(z, w) == K(w, z)`` for real symmetric ``z``, ``w``."""
    kern = BergmanKernel(dimension=4, n_eigenfunctions=4)
    dom = BoundedSymmetricDomain(dimension=4)
    occ = _occupancy(batch=5, d=4)
    z = dom.point(occ)
    z_e = z.unsqueeze(-3)
    z_t = z.unsqueeze(0)
    K_zw = kern.forward(z_e, z_t)  # (5, 5)
    K_wz = kern.forward(z_t, z_e)  # (5, 5)
    assert torch.allclose(K_zw, K_wz, atol=1e-5)


def test_bergman_kernel_positive_definite_gram_matrix() -> None:
    """The Gram matrix ``G_{ij} = K(z_i, z_j)`` is positive semi-definite."""
    kern = BergmanKernel(dimension=4, n_eigenfunctions=4)
    dom = BoundedSymmetricDomain(dimension=4)
    occ = _occupancy(batch=6, d=4)
    z = dom.point(occ)
    z_e = z.unsqueeze(-3)
    z_t = z.unsqueeze(0)
    G = kern.forward(z_e, z_t)  # (6, 6)
    # Add a small jitter so eigvalsh is robust even when the kernel is
    # rank-deficient (Nyström truncation).
    G_jit = G + 1e-6 * torch.eye(6)
    eigvals = torch.linalg.eigvalsh(G_jit)
    assert (eigvals >= -1e-5).all()


# ---------------------------------------------------------------------------
# 6. BergmanKernel — eigenfunctions and Nyström features
# ---------------------------------------------------------------------------


def test_bergman_kernel_eigenfunctions_count() -> None:
    """``eigenfunctions()`` returns exactly ``n_eigenfunctions`` tensors."""
    kern = BergmanKernel(dimension=4, n_eigenfunctions=12)
    efs = kern.eigenfunctions()
    assert isinstance(efs, list)
    assert len(efs) == 12


def test_bergman_kernel_eigenfunctions_shape() -> None:
    """Each eigenfunction anchor has shape ``(d, d)``."""
    kern = BergmanKernel(dimension=5, n_eigenfunctions=8)
    for ef in kern.eigenfunctions():
        assert ef.shape == (5, 5)


def test_bergman_kernel_eigenfunctions_are_in_domain() -> None:
    """Each Nyström anchor lies in R_II(d)."""
    kern = BergmanKernel(dimension=5, n_eigenfunctions=8)
    dom = BoundedSymmetricDomain(dimension=5)
    anchors = torch.stack(kern.eigenfunctions(), dim=0)  # (N, d, d)
    assert bool(dom.in_domain(anchors).all())


def test_bergman_kernel_eigenfunctions_are_clones() -> None:
    """``eigenfunctions()`` returns fresh tensors (not aliased views)."""
    kern = BergmanKernel(dimension=4, n_eigenfunctions=4)
    efs = kern.eigenfunctions()
    # Mutating a returned tensor must NOT change the buffer.
    efs[0].zero_()
    assert kern._anchors[0].abs().sum() > 0.0


def test_bergman_kernel_features_shape() -> None:
    """``features(z)`` returns shape ``(..., n_eigenfunctions)``."""
    kern = BergmanKernel(dimension=5, n_eigenfunctions=8)
    dom = BoundedSymmetricDomain(dimension=5)
    occ = _occupancy(batch=4, d=5)
    z = dom.point(occ)
    feats = kern.features(z)
    assert feats.shape == (4, 8)


def test_bergman_kernel_features_at_origin_are_one() -> None:
    """``features(0) = (K(0, w_i)) = (1, ..., 1)`` for any anchors."""
    kern = BergmanKernel(dimension=5, n_eigenfunctions=6)
    z0 = torch.zeros(2, 5, 5)
    feats = kern.features(z0)
    assert torch.allclose(feats, torch.ones_like(feats), atol=1e-5)


def test_bergman_kernel_features_strictly_positive() -> None:
    """Nyström features are strictly positive for in-domain inputs."""
    kern = BergmanKernel(dimension=5, n_eigenfunctions=8)
    dom = BoundedSymmetricDomain(dimension=5)
    occ = _occupancy(batch=6, d=5)
    z = dom.point(occ)
    feats = kern.features(z)
    assert (feats > 0.0).all()
    assert torch.isfinite(feats).all()


def test_bergman_kernel_anchor_norms_at_least_one() -> None:
    """``K(w_i, w_i) >= 1`` for each anchor (zonal polynomial unit norm)."""
    kern = BergmanKernel(dimension=5, n_eigenfunctions=8)
    norms = kern.anchor_norms()
    assert norms.shape == (8,)
    assert (norms >= 1.0 - 1e-5).all()


# ---------------------------------------------------------------------------
# 7. BergmanKernelEmbedding — end-to-end module
# ---------------------------------------------------------------------------


def test_bergman_embedding_default_construction() -> None:
    """Default construction: hidden_dim=64, dimension=5."""
    emb = BergmanKernelEmbedding()
    assert emb.hidden_dim == 64
    assert emb.dimension == 5
    assert emb.domain.d == 5
    assert emb.kernel.d == 5
    assert emb.kernel.n_eigenfunctions == 64


def test_bergman_embedding_forward_shape() -> None:
    """``forward(occupation)`` returns shape ``(..., hidden_dim)``."""
    emb = BergmanKernelEmbedding(hidden_dim=12, dimension=5)
    occ = _occupancy(batch=8, d=5)
    feats = emb(occ)
    assert feats.shape == (8, 12)


def test_bergman_embedding_2d_batch_input() -> None:
    """``forward`` accepts arbitrary leading batch dims."""
    emb = BergmanKernelEmbedding(hidden_dim=4, dimension=5)
    occ = torch.randn(2, 3, 5)
    feats = emb(occ)
    assert feats.shape == (2, 3, 4)


def test_bergman_embedding_gradients_flow() -> None:
    """Gradients flow end-to-end through the Bergman-kernel feature map."""
    emb = BergmanKernelEmbedding(hidden_dim=4, dimension=5)
    occ = _occupancy(batch=3, d=5, scale=0.5).requires_grad_(True)
    feats = emb(occ)  # (3, 4)
    # A simple scalar loss: sum of features.
    loss = feats.sum()
    loss.backward()
    assert occ.grad is not None
    assert torch.isfinite(occ.grad).all()
    assert (occ.grad.abs() > 0.0).all()


def test_bergman_embedding_finite_outputs() -> None:
    """All outputs are finite for non-saturating occupancy inputs."""
    emb = BergmanKernelEmbedding(hidden_dim=8, dimension=5)
    occ = _occupancy(batch=16, d=5, scale=0.7)
    feats = emb(occ)
    assert torch.isfinite(feats).all()


def test_bergman_embedding_strictly_positive_features() -> None:
    """The embedding features are strictly positive (K > 0 on R_II)."""
    emb = BergmanKernelEmbedding(hidden_dim=8, dimension=5)
    occ = _occupancy(batch=4, d=5)
    feats = emb(occ)
    assert (feats > 0.0).all()


def test_bergman_embedding_consistent_with_kernel() -> None:
    """Embedding output matches ``kernel.features(domain.point(occ))``."""
    emb = BergmanKernelEmbedding(hidden_dim=8, dimension=5)
    occ = _occupancy(batch=3, d=5)
    feats = emb(occ)
    # Recompute via the underlying primitives.
    z = emb.domain.point(occ)
    feats_ref = emb.kernel.features(z)
    assert torch.allclose(feats, feats_ref, atol=1e-6)


def test_bergman_embedding_zero_input_gives_ones() -> None:
    """``embedding(0) = (1, 1, ..., 1)`` — the kernel normalises at 0."""
    emb = BergmanKernelEmbedding(hidden_dim=8, dimension=5)
    occ = torch.zeros(2, 5)
    feats = emb(occ)
    assert torch.allclose(feats, torch.ones_like(feats), atol=1e-5)


def test_bergman_embedding_rejects_zero_hidden_dim() -> None:
    """``hidden_dim = 0`` is rejected at construction time."""
    with pytest.raises(ValueError, match="hidden_dim"):
        BergmanKernelEmbedding(hidden_dim=0, dimension=5)


def test_bergman_embedding_rejects_zero_dimension() -> None:
    """``dimension = 0`` is rejected at construction time."""
    with pytest.raises(ValueError, match="dimension"):
        BergmanKernelEmbedding(hidden_dim=4, dimension=0)


def test_bergman_embedding_state_dict_round_trip() -> None:
    """The embedding can be saved and reloaded via ``state_dict``."""
    emb = BergmanKernelEmbedding(hidden_dim=4, dimension=3, seed=1)
    feats_before = emb(_occupancy(batch=2, d=3, seed=99))
    state = emb.state_dict()
    emb2 = BergmanKernelEmbedding(hidden_dim=4, dimension=3, seed=0)
    emb2.load_state_dict(state)
    feats_after = emb2(_occupancy(batch=2, d=3, seed=99))
    assert torch.allclose(feats_before, feats_after, atol=1e-6)


# ---------------------------------------------------------------------------
# 8. Standalone — module does not import from molmetal.adapters or lam
# ---------------------------------------------------------------------------


def test_bergman_kernel_module_is_standalone() -> None:
    """Module source has no ``import molmetal.adapters`` /
    ``import molmetal.molmetal_lam`` statements (standalone skeleton contract)."""
    src_path = (
        Path(__file__).resolve().parent.parent / "math_skills" / "bergman_kernel.py"
    )
    source = src_path.read_text()
    # Check for actual import statements, not docstring mentions.
    forbidden_imports = [
        "import molmetal.adapters",
        "from molmetal.adapters",
        "import molmetal.molmetal_lam",
        "from molmetal.molmetal_lam",
    ]
    for token in forbidden_imports:
        assert (
            token not in source
        ), f"bergman_kernel.py must not {token!r}; found in source."
    # The only allowed third-party dependencies are torch + torch.nn.
    assert "import torch" in source
    assert "from torch import Tensor" in source or "torch.Tensor" in source
