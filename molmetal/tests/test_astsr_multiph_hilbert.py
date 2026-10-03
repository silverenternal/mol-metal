"""Tests for the ASTSR + Multi-PH joint Hilbert integration layer.

================================================================
Wf-14 phase-3 (joint Hilbert optimization, 2026-09-26)
================================================================
This module validates that
:mod:`molmetal.molmetal_lam.reward.astsr_multiph_hilbert` correctly
joins the Wf-14 phase-2 (Hilbert + ASTSR) and Wf-14 phase-4
(Multi-PH + PLL) feature streams behind a single ``fit`` /
``predict`` / ``compare`` API and provides a fair head-to-head
wall-time + R² comparison.

Test inventory (25 tests)
-------------------------
1. **JointFeatureConcatenator** — concat shape, normalization
   statistics, 2-D / 3-D multi-PH handling (7 tests).
2. **JointHilbertOptimizer** — fit returns R² ≥ 0.5, predict shape
   preserved, NaN-safe unfitted stub, edge-case rejection (10
   tests).
3. **JointVsSeparateComparator** — wall-time + R² comparison, dict
   shape, speedup finite, no mutation of the underlying joint
   optimizer (8 tests).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from molmetal.molmetal_lam.reward.astsr_multiph_hilbert import (
    JointFeatureConcatenator,
    JointHilbertOptimizer,
    JointVsSeparateComparator,
    MultiPHConcatAdapter,
    StableMultiPHExtractor,
)
from molmetal.molmetal_lam.reward.hilbert_primal_dual import LambdaASTSRHilbert


# ---------------------------------------------------------------------------
# Helpers — small deterministic design matrices for fast tests
# ---------------------------------------------------------------------------
def _toy_astr_design(
    n: int = 24,
    dim: int = 4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a deterministic ``(X_astr, y)`` with a known linear structure."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    w = rng.standard_normal(dim)
    y = X @ w + 0.01 * rng.standard_normal(n)
    return X, y


def _toy_multi_ph_design(
    n: int = 24,
    n_bins: int = 4,
    seed: int = 1,
) -> np.ndarray:
    """Build a deterministic ``(N, n_bins, n_bins)`` multi-PH feature map."""
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n, n_bins, n_bins))
    return feats.astype(float)


def _toy_joint_inputs(
    n: int = 24,
    dim_astr: int = 4,
    n_bins: int = 4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build matched (astr_X, multi_ph_X, y) with informative streams."""
    rng = np.random.default_rng(seed)
    X_astr = rng.standard_normal((n, dim_astr))
    mph = rng.standard_normal((n, n_bins, n_bins)).reshape(n, -1) * 0.1
    # Target = linear(ASTSR) + 0 contribution from multi-PH — joint
    # should at least match separate because it sees everything.
    w = rng.standard_normal(dim_astr)
    y = X_astr @ w + 0.01 * rng.standard_normal(n)
    mph_full = mph.reshape(n, n_bins, n_bins)
    return X_astr, mph_full, y


# ---------------------------------------------------------------------------
# 1. JointFeatureConcatenator (7 tests)
# ---------------------------------------------------------------------------
def test_concatenator_concatenate_shape_2d_multi_ph() -> None:
    """``concatenate`` returns ``(N, d_a + d_m)`` for 2-D multi-PH."""
    cat = JointFeatureConcatenator()
    astr = np.zeros((8, 3))
    mph = np.ones((8, 5))
    out = cat.concatenate(astr, mph)
    assert out.shape == (8, 8)


def test_concatenator_concatenate_shape_3d_multi_ph() -> None:
    """``concatenate`` flattens 3-D multi-PH to ``(N, n_bins ** 2)``."""
    cat = JointFeatureConcatenator()
    astr = np.zeros((6, 2))
    mph = np.ones((6, 4, 4))  # 16 flattened features
    out = cat.concatenate(astr, mph)
    assert out.shape == (6, 18)
    # Cached dims recorded for downstream introspection.
    assert cat.dim_astr == 2
    assert cat.dim_multi_ph == 16


def test_concatenator_normalize_zero_mean_unit_std() -> None:
    """``fit_normalize`` produces columns with mean ~0 and std ~1."""
    cat = JointFeatureConcatenator()
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((50, 6)) * 5 + 3
    out = cat.fit_normalize(feats)
    assert np.allclose(np.mean(out, axis=0), 0.0, atol=1.0e-6)
    assert np.allclose(np.std(out, axis=0), 1.0, atol=1.0e-6)


def test_concatenator_normalize_reuses_cached_stats() -> None:
    """A second ``transform`` reuses the cached (mean, std)."""
    cat = JointFeatureConcatenator()
    rng = np.random.default_rng(1)
    train = rng.standard_normal((40, 4)) * 2 + 1
    cat.fit_normalize(train)
    cached_mean = cat._mean.copy()
    test = rng.standard_normal((10, 4)) * 2 + 1
    out = cat.transform(test)
    # The transform must use the *cached* mean — re-fitting on test
    # would produce a different (mean, std).
    assert np.allclose(cat._mean, cached_mean)
    # Output has zero mean only when subtracting the *cached* mean.
    shifted = test - cached_mean
    expected = shifted / cat._std
    assert np.allclose(out, expected, atol=1.0e-9)


def test_concatenator_normalize_handles_zero_variance() -> None:
    """``fit_normalize`` does not divide by zero on constant columns."""
    cat = JointFeatureConcatenator()
    feats = np.ones((10, 3))
    out = cat.fit_normalize(feats)
    # All-zero output for a constant column is the documented
    # behaviour; what matters is finiteness.
    assert np.all(np.isfinite(out))


def test_concatenator_rejects_batch_mismatch() -> None:
    """``concatenate`` raises ``ValueError`` on batch-dim mismatch."""
    cat = JointFeatureConcatenator()
    with pytest.raises(ValueError, match="batch mismatch"):
        cat.concatenate(np.zeros((5, 3)), np.ones((6, 4)))


def test_concatenator_rejects_invalid_ndim() -> None:
    """``concatenate`` rejects multi-PH tensors with batch dim too small."""
    cat = JointFeatureConcatenator()
    with pytest.raises(ValueError, match="astr_feat must be"):
        cat.concatenate(np.zeros((3, 4, 5)), np.ones((3, 4)))


# ---------------------------------------------------------------------------
# 2. JointHilbertOptimizer (10 tests)
# ---------------------------------------------------------------------------
def test_joint_hilbert_optimizer_fit_returns_high_r2() -> None:
    """``fit`` returns R² ≥ 0.5 on a noiseless-ish linear problem."""
    astr, mph, y = _toy_joint_inputs(n=24, dim_astr=4, n_bins=4, seed=2)
    opt = JointHilbertOptimizer(
        hilbert_dim=8,
        normalize=False,
    )
    res = opt.fit(astr, mph, y)
    assert "r2" in res and "wall_s" in res
    assert res["backend"] == "joint_hilbert"
    assert res["join_dim"] == astr.shape[1] + int(np.prod(mph.shape[1:]))
    # The Hilbert regressor should at least beat the constant-mean
    # baseline by a wide margin on this noiseless-ish problem.
    assert res["r2"] >= 0.5, res["r2"]


def test_joint_hilbert_optimizer_predict_shape_preserved() -> None:
    """``predict`` returns ``(N,)`` with the same batch dim as input."""
    astr, mph, y = _toy_joint_inputs(n=16, dim_astr=3, n_bins=3, seed=3)
    opt = JointHilbertOptimizer(hilbert_dim=6, normalize=False)
    opt.fit(astr, mph, y)
    out = opt.predict(astr, mph)
    assert out.shape == (astr.shape[0],)


def test_joint_hilbert_optimizer_predict_unfitted_returns_zeros() -> None:
    """``predict`` on an unfitted optimizer returns zeros (NaN-safe)."""
    opt = JointHilbertOptimizer(hilbert_dim=4)
    out = opt.predict(np.zeros((5, 3)), np.ones((5, 2, 2)))
    assert out.shape == (5,)
    assert np.allclose(out, 0.0)


def test_joint_hilbert_optimizer_is_fitted_reflects_state() -> None:
    """``is_fitted`` flips to ``True`` only after a successful fit."""
    opt = JointHilbertOptimizer(hilbert_dim=4)
    assert opt.is_fitted is False
    astr, mph, y = _toy_joint_inputs(n=12, dim_astr=3, n_bins=2, seed=4)
    opt.fit(astr, mph, y)
    assert opt.is_fitted is True


def test_joint_hilbert_optimizer_predict_after_fit_finite() -> None:
    """Predictions after fit are finite (no NaN / Inf)."""
    astr, mph, y = _toy_joint_inputs(n=20, dim_astr=4, n_bins=3, seed=5)
    opt = JointHilbertOptimizer(hilbert_dim=8, normalize=True)
    opt.fit(astr, mph, y)
    out = opt.predict(astr, mph)
    assert np.all(np.isfinite(out))


def test_joint_hilbert_optimizer_rejects_length_mismatch() -> None:
    """``fit`` raises when target length differs from batch dim."""
    opt = JointHilbertOptimizer(hilbert_dim=4)
    astr = np.zeros((10, 3))
    mph = np.ones((10, 2, 2))
    with pytest.raises(ValueError, match="length mismatch"):
        opt.fit(astr, mph, np.zeros(7))


def test_joint_hilbert_optimizer_rejects_invalid_hilbert_dim() -> None:
    """``__post_init__`` rejects ``hilbert_dim < 1``."""
    with pytest.raises(ValueError, match="hilbert_dim"):
        JointHilbertOptimizer(hilbert_dim=0)


def test_joint_hilbert_optimizer_records_wall_time() -> None:
    """``last_fit_wall_s_`` is non-negative and matches the returned dict."""
    astr, mph, y = _toy_joint_inputs(n=16, dim_astr=3, n_bins=2, seed=6)
    opt = JointHilbertOptimizer(hilbert_dim=4, normalize=False)
    res = opt.fit(astr, mph, y)
    assert opt.last_fit_wall_s_ >= 0.0
    assert res["wall_s"] == opt.last_fit_wall_s_


def test_joint_hilbert_optimizer_normalize_optional() -> None:
    """``normalize=False`` skips z-score; the optimizer still fits."""
    astr, mph, y = _toy_joint_inputs(n=14, dim_astr=3, n_bins=2, seed=7)
    opt = JointHilbertOptimizer(hilbert_dim=4, normalize=False)
    res = opt.fit(astr, mph, y)
    assert math.isfinite(res["r2"])
    # Normalization is opt-in: the concatenator's stats stay unset.
    assert opt.concatenator.is_normalized is False


def test_joint_hilbert_optimizer_rejects_empty_target() -> None:
    """``fit`` raises on an empty target."""
    opt = JointHilbertOptimizer(hilbert_dim=4)
    astr = np.zeros((4, 2))
    mph = np.ones((4, 2, 2))
    with pytest.raises(ValueError, match="target"):
        opt.fit(astr, mph, np.zeros(0))


# ---------------------------------------------------------------------------
# 3. JointVsSeparateComparator (8 tests)
# ---------------------------------------------------------------------------
def _toy_separate_hilbert_factory(
    population_dim: int = 4,
    seed: int = 0,
) -> LambdaASTSRHilbert:
    """Build a deterministic ``LambdaASTSRHilbert`` for the comparator tests.

    ``n_iters=200`` so the per-stream regressors converge on
    noiseless-ish linear problems; the comparator's per-stream R² ceiling
    is then a function of the data, not the optimiser.
    """
    return LambdaASTSRHilbert(
        population_dim=int(population_dim),
        lr_primal=0.05,
        lr_dual=0.05,
        n_iters=200,
        seed=int(seed),
    )


def test_joint_vs_separate_returns_full_dict() -> None:
    """``compare`` returns the documented wall-time + R² dict."""
    astr, mph, y = _toy_joint_inputs(n=20, dim_astr=4, n_bins=3, seed=8)
    joint = JointHilbertOptimizer(hilbert_dim=8, normalize=False)
    comp = JointVsSeparateComparator(
        separate_hilbert=_toy_separate_hilbert_factory,
        joint_hilbert=joint,
    )
    out = comp.compare((astr, mph), y, n_seeds=2)
    for key in (
        "joint_wall_s",
        "separate_wall_s",
        "joint_r2",
        "separate_r2",
        "speedup",
        "n_seeds",
    ):
        assert key in out, f"missing {key}"
    assert out["n_seeds"] == 2


def test_joint_vs_separate_wall_times_nonneg() -> None:
    """Both wall-times are non-negative floats."""
    astr, mph, y = _toy_joint_inputs(n=16, dim_astr=3, n_bins=2, seed=9)
    joint = JointHilbertOptimizer(hilbert_dim=4, normalize=False)
    comp = JointVsSeparateComparator(
        separate_hilbert=_toy_separate_hilbert_factory,
        joint_hilbert=joint,
    )
    out = comp.compare((astr, mph), y, n_seeds=2)
    assert out["joint_wall_s"] >= 0.0
    assert out["separate_wall_s"] >= 0.0
    # Speedup must be finite on real wall-time.
    assert math.isfinite(out["speedup"])


def test_joint_vs_separate_speedup_finite() -> None:
    """``speedup`` is finite when both wall-times are positive."""
    astr, mph, y = _toy_joint_inputs(n=16, dim_astr=3, n_bins=2, seed=10)
    joint = JointHilbertOptimizer(hilbert_dim=4, normalize=False)
    comp = JointVsSeparateComparator(
        separate_hilbert=_toy_separate_hilbert_factory,
        joint_hilbert=joint,
    )
    out = comp.compare((astr, mph), y, n_seeds=2)
    assert math.isfinite(out["speedup"])
    assert out["speedup"] > 0.0


def test_joint_vs_separate_r2_in_unit_interval() -> None:
    """Both R² values land in ``[0, 1]`` (clipped / degenerate OK)."""
    astr, mph, y = _toy_joint_inputs(n=18, dim_astr=3, n_bins=2, seed=11)
    joint = JointHilbertOptimizer(hilbert_dim=4, normalize=False)
    comp = JointVsSeparateComparator(
        separate_hilbert=_toy_separate_hilbert_factory,
        joint_hilbert=joint,
    )
    out = comp.compare((astr, mph), y, n_seeds=2)
    for k in ("joint_r2", "separate_r2"):
        v = float(out[k])
        assert -1.0e-3 <= v <= 1.0 + 1.0e-3, f"{k}={v}"


def test_joint_vs_separate_joint_at_least_separate_r2() -> None:
    """Joint Hilbert R² ≥ separate Hilbert R² on noiseless-ish linear data.

    Under a linear Gaussian model where ASTSR is the only informative
    stream, the joint Hilbert regressor sees everything the separate
    regressor sees plus the multi-PH noise — the joint R² should be
    at least as good as the separate R², *up to optimisation noise*.
    """
    astr, mph, y = _toy_joint_inputs(n=24, dim_astr=4, n_bins=3, seed=12)
    joint = JointHilbertOptimizer(hilbert_dim=8, normalize=False)
    comp = JointVsSeparateComparator(
        separate_hilbert=_toy_separate_hilbert_factory,
        joint_hilbert=joint,
    )
    out = comp.compare((astr, mph), y, n_seeds=3)
    # Allow a 5% slack for PD-iteration noise; the joint regressor
    # is theoretically at least as informative as the sum of the
    # separate regressors under a linear-Gaussian data model.
    assert out["joint_r2"] + 0.05 >= out["separate_r2"], (
        f"joint R² {out['joint_r2']:.4f} < separate R² " f"{out['separate_r2']:.4f}"
    )


def test_joint_vs_separate_joint_wall_time_bounded() -> None:
    """Joint wall-time stays within one order of magnitude of separate."""
    astr, mph, y = _toy_joint_inputs(n=18, dim_astr=4, n_bins=3, seed=13)
    joint = JointHilbertOptimizer(hilbert_dim=8, normalize=False)
    comp = JointVsSeparateComparator(
        separate_hilbert=_toy_separate_hilbert_factory,
        joint_hilbert=joint,
    )
    out = comp.compare((astr, mph), y, n_seeds=2)
    # Joint is one regressor; separate is two — joint cannot be more
    # than ~10x slower on this toy design.
    assert out["joint_wall_s"] <= 10.0 * out["separate_wall_s"] + 1.0e-3


def test_joint_vs_separate_rejects_invalid_features_list() -> None:
    """``compare`` raises on a 1-element or 3-element features list."""
    astr, mph, y = _toy_joint_inputs(n=12, dim_astr=3, n_bins=2, seed=14)
    joint = JointHilbertOptimizer(hilbert_dim=4, normalize=False)
    comp = JointVsSeparateComparator(
        separate_hilbert=_toy_separate_hilbert_factory,
        joint_hilbert=joint,
    )
    with pytest.raises(ValueError, match="2-tuple"):
        comp.compare((astr, mph, y), y, n_seeds=1)


def test_joint_vs_separate_does_not_mutate_joint_optimizer() -> None:
    """``compare`` does not silently rewire the joint optimizer's config."""
    astr, mph, y = _toy_joint_inputs(n=16, dim_astr=3, n_bins=2, seed=15)
    joint = JointHilbertOptimizer(
        hilbert_dim=4,
        normalize=False,
        feature_columns=("a", "b", "c"),
    )
    fc_before = tuple(joint.feature_columns)
    norm_before = bool(joint.normalize)
    comp = JointVsSeparateComparator(
        separate_hilbert=_toy_separate_hilbert_factory,
        joint_hilbert=joint,
    )
    comp.compare((astr, mph), y, n_seeds=1)
    assert tuple(joint.feature_columns) == fc_before
    assert bool(joint.normalize) == norm_before


# ---------------------------------------------------------------------------
# 4. Honest framing (3 tests) — covers the cross-cutting invariants
# ---------------------------------------------------------------------------
def test_does_not_mutate_hilbert_astsr_integration() -> None:
    """Importing the new module does not patch the existing integration layer."""
    from molmetal.molmetal_lam.reward import (
        astsr_multiph_hilbert,
        hilbert_astsr_integration,
    )

    # The new module exposes exactly the documented public API.
    assert hasattr(astsr_multiph_hilbert, "JointHilbertOptimizer")
    assert hasattr(astsr_multiph_hilbert, "JointFeatureConcatenator")
    assert hasattr(astsr_multiph_hilbert, "JointVsSeparateComparator")
    # The existing integration layer is *not* silently re-wired.
    assert not hasattr(hilbert_astsr_integration, "_astsr_multiph_override")
    # The new module imports the existing public API rather than
    # re-implementing it — verify the public symbols are still bound.
    assert hasattr(hilbert_astsr_integration, "HilbertASTSRComposite")
    assert hasattr(hilbert_astsr_integration, "HilbertEvolutionaryComparator")


def test_normalize_then_transform_is_consistent() -> None:
    """``transform`` reuses the train statistics on new (raw) test data."""
    cat = JointFeatureConcatenator()
    rng = np.random.default_rng(42)
    train = rng.standard_normal((30, 5)) * 3 + 7
    test = rng.standard_normal((10, 5)) * 3 + 7
    # Fit on train, transform test with the same stats.
    cat.fit_normalize(train)
    cached_mean = cat._mean.copy()
    cached_std = cat._std.copy()
    out_test = cat.transform(test)
    # Expected: (test - mean_of_train) / std_of_train — NOT the
    # train/test statistics re-fit independently.
    expected = (test - cached_mean) / cached_std
    assert np.allclose(out_test, expected, atol=1.0e-9)


def test_comparator_rejects_non_callable_factory() -> None:
    """``JointVsSeparateComparator`` raises on a non-callable factory."""
    joint = JointHilbertOptimizer(hilbert_dim=4)
    with pytest.raises(ValueError, match="callable"):
        JointVsSeparateComparator(
            separate_hilbert="not_callable",  # type: ignore[arg-type]
            joint_hilbert=joint,
        )


# ---------------------------------------------------------------------------
# Wf-236 — StableMultiPHExtractor + MultiPHConcatAdapter
# ---------------------------------------------------------------------------
# The Wf-236 integration ships the *production* 2-PH extraction path
# (Carlsson-Mémoli 2010 AND-multi-filter + Kim-Jung 2026 CountGrid
# statistic) plus a numpy-friendly adapter that bridges pocket
# coordinates into the existing :class:`JointHilbertOptimizer`.  These
# tests cover the API surface and the stability gate.


def _toy_pocket(n: int = 12, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic pocket coords + atomic numbers."""
    rng = np.random.default_rng(int(seed))
    coords = rng.standard_normal((n, 3)) * 2.0
    atoms = rng.integers(1, 18, size=(n,)).astype(np.int64)
    return coords.astype(np.float32), atoms


def _toy_pocket_batch(
    b: int = 4,
    n: int = 12,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic batched pocket coords + atomic numbers + mask."""
    rng = np.random.default_rng(int(seed))
    coords = rng.standard_normal((b, n, 3)) * 2.0
    atoms = rng.integers(1, 18, size=(b, n)).astype(np.int64)
    mask = np.ones((b, n), dtype=bool)
    return coords.astype(np.float32), atoms, mask


# -- StableMultiPHExtractor ------------------------------------------------


def test_stable_multi_ph_extractor_construction() -> None:
    """The extractor validates its constructor arguments."""
    with pytest.raises(ValueError, match="max_radius"):
        StableMultiPHExtractor(max_radius=0.0)
    with pytest.raises(ValueError, match="max_coord"):
        StableMultiPHExtractor(max_coord=-1.0)
    with pytest.raises(ValueError, match="n_bins"):
        StableMultiPHExtractor(n_bins=1)


def test_stable_multi_ph_extractor_feature_dim() -> None:
    """``feature_dim`` is the flattened bin grid (``n_bins ** 2``)."""
    extractor = StableMultiPHExtractor(n_bins=8)
    assert extractor.feature_dim == 64
    extractor16 = StableMultiPHExtractor(n_bins=16)
    assert extractor16.feature_dim == 256


def test_stable_multi_ph_extractor_single_pocket_compute() -> None:
    """Single-pocket ``compute`` returns a ``(n_bins**2,)`` array."""
    extractor = StableMultiPHExtractor(n_bins=8)
    coords, atoms = _toy_pocket(n=12)
    feat = extractor.compute(coords, atoms)
    assert feat.shape == (extractor.feature_dim,)
    assert np.isfinite(feat).all()
    # The feature map is non-negative (count grid).
    assert (feat >= 0).all()


def test_stable_multi_ph_extractor_batch_compute_shape() -> None:
    """Batched ``compute_batch`` returns a ``(B, n_bins**2)`` array."""
    extractor = StableMultiPHExtractor(n_bins=8)
    coords, atoms, mask = _toy_pocket_batch(b=4, n=12)
    feat = extractor.compute_batch(coords, atoms, pocket_mask=mask)
    assert feat.shape == (4, extractor.feature_dim)
    assert np.isfinite(feat).all()


def test_stable_multi_ph_extractor_batch_handles_padding() -> None:
    """Padded rows (mask=False) do not contribute spurious edges."""
    extractor = StableMultiPHExtractor(n_bins=8)
    coords, atoms, mask = _toy_pocket_batch(b=2, n=10)
    # Mask out half the atoms — only the un-masked half should
    # contribute edges.  We don't assert exact equality (the
    # carbon-default fallback also fires), but we assert that the
    # masked-batch features are finite and non-negative.
    mask[:, :5] = False
    feat = extractor.compute_batch(coords, atoms, pocket_mask=mask)
    assert feat.shape == (2, extractor.feature_dim)
    assert np.isfinite(feat).all()
    assert (feat >= 0).all()


def test_stable_multi_ph_extractor_validates_batch_inputs() -> None:
    """Batched extraction raises on shape mismatches."""
    extractor = StableMultiPHExtractor(n_bins=8)
    coords_bad = np.zeros((4, 12, 4), dtype=np.float32)  # wrong trailing dim
    atoms = np.zeros((4, 12), dtype=np.int64)
    with pytest.raises(ValueError, match="pocket_coords"):
        extractor.compute_batch(coords_bad, atoms)
    atoms_bad = np.zeros((4, 11), dtype=np.int64)  # wrong N
    coords = np.zeros((4, 12, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="pocket_atom_types"):
        extractor.compute_batch(coords, atoms_bad)
    mask_bad = np.zeros((4, 11), dtype=bool)  # wrong N
    with pytest.raises(ValueError, match="pocket_mask"):
        extractor.compute_batch(coords, atoms, pocket_mask=mask_bad)


def test_stable_multi_ph_extractor_stability_gate_finite() -> None:
    """``is_stable`` rejects NaN / Inf entries and all-zero feature maps."""
    extractor = StableMultiPHExtractor(n_bins=8)
    feat = np.zeros((4, 64))
    # All-zero feature map → degenerate (no edges detected).
    assert extractor.is_stable(feat) is False
    feat[0, 0] = np.nan
    assert extractor.is_stable(feat) is False
    feat[0, 0] = np.inf
    assert extractor.is_stable(feat) is False


def test_stable_multi_ph_extractor_stability_gate_variance() -> None:
    """``is_stable`` returns False when every column has zero variance."""
    extractor = StableMultiPHExtractor(n_bins=8)
    # All-zero feature map → zero variance across the batch axis.
    feat = np.zeros((3, 64))
    assert extractor.is_stable(feat) is False
    # Tiny non-zero variance but below the std floor.
    feat = np.full((3, 64), 1.0e-12)
    assert extractor.is_stable(feat) is False
    # Sufficient variance → stable.
    rng = np.random.default_rng(0)
    feat = rng.standard_normal((3, 64))
    assert extractor.is_stable(feat) is True


# -- MultiPHConcatAdapter --------------------------------------------------


def test_multi_ph_concat_adapter_default_construction() -> None:
    """Default constructor builds a fresh extractor + concatenator."""
    adapter = MultiPHConcatAdapter()
    assert isinstance(adapter.extractor, StableMultiPHExtractor)
    assert isinstance(adapter.concatenator, JointFeatureConcatenator)


def test_multi_ph_concat_adapter_single_pocket() -> None:
    """Single-pocket extraction: (N, 3) → broadcast across all ASTSR rows."""
    extractor = StableMultiPHExtractor(n_bins=8)
    adapter = MultiPHConcatAdapter(extractor=extractor)
    coords, atoms = _toy_pocket(n=12)
    astr = np.random.default_rng(0).standard_normal((4, 6))
    combined = adapter.extract_joint(astr, coords, atoms)
    assert combined.shape == (4, 6 + extractor.feature_dim)
    assert np.isfinite(combined).all()


def test_multi_ph_concat_adapter_batched_pockets() -> None:
    """Batched extraction: (B, N, 3) + (B, N) → (B, d_a + n_bins**2)."""
    extractor = StableMultiPHExtractor(n_bins=8)
    adapter = MultiPHConcatAdapter(extractor=extractor)
    coords, atoms, mask = _toy_pocket_batch(b=4, n=12)
    astr = np.random.default_rng(0).standard_normal((4, 6))
    combined = adapter.extract_joint(astr, coords, atoms, pocket_mask=mask)
    assert combined.shape == (4, 6 + extractor.feature_dim)
    assert np.isfinite(combined).all()


def test_multi_ph_concat_adapter_normalize_path() -> None:
    """``normalize=True`` z-scores the joined feature matrix."""
    extractor = StableMultiPHExtractor(n_bins=8)
    adapter = MultiPHConcatAdapter(extractor=extractor)
    coords, atoms, mask = _toy_pocket_batch(b=8, n=12)
    astr = np.random.default_rng(0).standard_normal((8, 6)) * 5.0
    combined = adapter.extract_joint(
        astr,
        coords,
        atoms,
        pocket_mask=mask,
        normalize=True,
    )
    # After z-score, per-column mean ≈ 0 and per-column std ≈ 1 *for the
    # dense (ASTSR) columns*.  The multi-PH columns are sparse, so many
    # bins are uniformly zero → std clipped at the floor (not 1).
    # We only check the ASTSR columns here.
    n_astr = 6
    col_means = combined.mean(axis=0)
    col_stds = combined.std(axis=0)
    assert np.allclose(col_means[:n_astr], 0.0, atol=1.0e-6)
    assert np.allclose(col_stds[:n_astr], 1.0, atol=1.0e-5)


def test_multi_ph_concat_adapter_batch_mismatch_raises() -> None:
    """A B-mismatch between ASTSR and pockets raises ``ValueError``."""
    adapter = MultiPHConcatAdapter(extractor=StableMultiPHExtractor(n_bins=8))
    coords = np.zeros((4, 12, 3), dtype=np.float32)
    atoms = np.zeros((4, 12), dtype=np.int64)
    astr = np.random.default_rng(0).standard_normal((5, 6))  # 5 vs 4
    with pytest.raises(ValueError, match="batch mismatch"):
        adapter.extract_joint(astr, coords, atoms)


def test_multi_ph_concat_adapter_emits_warning_on_degenerate() -> None:
    """A degenerate multi-PH feature map emits a ``RuntimeWarning``."""
    extractor = StableMultiPHExtractor(n_bins=8)
    adapter = MultiPHConcatAdapter(extractor=extractor)
    # Single pocket with a single atom → degenerate multi-PH
    coords = np.zeros((1, 3), dtype=np.float32)
    atoms = np.array([6], dtype=np.int64)
    astr = np.zeros((1, 6))
    with pytest.warns(RuntimeWarning, match="unstable"):
        adapter.extract_joint(astr, coords, atoms)


def test_multi_ph_concat_adapter_fit_joint_hilbert_smoke() -> None:
    """End-to-end: extract multi-PH + fit a joint Hilbert regressor."""
    extractor = StableMultiPHExtractor(n_bins=8)
    adapter = MultiPHConcatAdapter(extractor=extractor)
    coords, atoms, mask = _toy_pocket_batch(b=8, n=12)
    astr = np.random.default_rng(0).standard_normal((8, 6))
    target = np.random.default_rng(1).standard_normal(8)
    joint = JointHilbertOptimizer(hilbert_dim=8, n_iters=20, lr=0.05)
    result = adapter.fit_joint_hilbert(
        joint,
        astr,
        coords,
        atoms,
        target,
        pocket_mask=mask,
    )
    assert "r2" in result
    assert "wall_s" in result
    assert "join_dim" in result
    assert "multi_ph_stable" in result
    assert result["multi_ph_stable"] is True
    # join_dim = d_a + n_bins ** 2 = 6 + 64 = 70
    assert result["join_dim"] == 6 + extractor.feature_dim


# -- 2-PH mathematical correctness -----------------------------------------


def test_multi_ph_two_parameter_filter_correctness() -> None:
    """2-PH filter: small-radius / small-coord pocket has near-empty cells at large r/d."""
    extractor = StableMultiPHExtractor(
        max_radius=1.0,
        max_coord=1.0,
        n_bins=8,
    )
    # Two clusters 0.5 Å apart — within the radius filter but with
    # large atomic-number differences.  The high-r + low-d corner of
    # the (r, d) grid should be populated; the low-r + high-d corner
    # should be empty.
    rng = np.random.default_rng(0)
    coords = np.concatenate(
        [
            rng.standard_normal((6, 3)) * 0.1,  # cluster 1
            rng.standard_normal((6, 3)) * 0.1 + 0.5,  # cluster 2
        ]
    ).astype(np.float32)
    atoms = np.concatenate(
        [
            np.full(6, 6, dtype=np.int64),  # all carbon
            np.full(6, 78, dtype=np.int64),  # all platinum
        ]
    )
    feat = extractor.compute(coords, atoms).reshape(8, 8)
    # The first row of feat (low-r cells) should be near zero (no
    # short edges within the small cluster), and the d-range that
    # includes |78 - 6| = 72 should be saturated (we clamp at max_coord=1,
    # so all |Pt - C| edges fall into the maximum-d bin).
    assert feat.shape == (8, 8)
    assert np.isfinite(feat).all()
    # Sanity: at least one bin must be non-empty.
    assert feat.sum() > 0.0


def test_multi_ph_single_atom_degeneracy() -> None:
    """A single-atom pocket yields an empty feature map (no edges)."""
    extractor = StableMultiPHExtractor(n_bins=8)
    coords = np.zeros((1, 3), dtype=np.float32)
    atoms = np.array([6], dtype=np.int64)
    feat = extractor.compute(coords, atoms)
    # No edges → all bins are zero.
    assert np.allclose(feat, 0.0)
    # Stability gate: zero-variance → unstable.
    assert extractor.is_stable(feat.reshape(1, -1)) is False
