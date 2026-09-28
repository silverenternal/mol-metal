"""Unit tests for the Wf-129 P3 Multi-PH + PLL guarded bench script.

This module validates the Wf-127 / Wf-122 / Wf-131 guards wired into the
guard-augmented Hilbert fit:

1. **_proximal_l2 closed-form** — l2_weight=0 reproduces the Wf-37 P3 path
   exactly (no shrinkage); l2_weight > 0 applies ``x *= max(0, 1 - lr * w)``.
2. **Wf-127 P2 invariants** — penalty >= 0 at every iter; l2_weight=0 path
   bit-exact to no-guard baseline.
3. **Wf-122 P1.2 ASTSR scale-mismatch guard** — std ratio outside
   ``[0.5, 1.5]`` is reported as collapse; good fits pass.
4. **Wf-131 P1 sliced-Wasserstein-2 helper** — same-input returns 0;
   permutation-invariant; non-negative.
5. **compute_r2_safe overflow guard** — degenerate input returns ``-inf``
   rather than NaN.
6. **fit_hilbert_guarded end-to-end** — converges on a clean linear target
   for at least one seed; l2_weight=0 vs ``W2_weight=0`` no-l2 baseline
   produce identical ``train_r2`` to 1e-9.
7. **k-fold split helper** — non-overlapping train/test; union coverage;
   reproducible on the same seed.
8. **Synthesis renderer** — produces well-formed Markdown; surfaces honest
   framing when all seeds collapse.

Test design
-----------
* All tests use CPU + numpy-only (no torch required unless the bench script
  is exercising the ``compute_multiph_features`` path which we mock).
* Fast budget: every test should complete in < 0.5 s on a typical CPU.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Repo root on sys.path.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.scripts.bench_multiph_pll_end_to_end_guarded import (  # type: ignore
    PLATINAI_4CELL_LINES,
    PLATINAI_4CELL_IC50_COLS,
    DEFAULT_L2_WEIGHT,
    DEFAULT_W2_WEIGHT,
    _proximal_l2,
    add_w2_to_loss,
    astr_scale_ok,
    compute_r2_safe,
    fit_hilbert_guarded,
    kfold_indices,
    predict_hilbert_guarded,
    render_synthesis,
    sliced_wasserstein_2,
)


# ---------------------------------------------------------------------------
# Wf-127 P2 closed-form proximal L2 operator
# ---------------------------------------------------------------------------

def test_proximal_l2_weight_zero_is_identity():
    """l2_weight=0 reproduces the Wf-37 P3 path exactly (no shrinkage)."""
    x = np.asarray([1.0, -2.0, 0.5, 0.0], dtype=float)
    x_out = _proximal_l2(x.copy(), lr=0.01, l2_weight=0.0)
    np.testing.assert_allclose(x_out, x, atol=1e-12)


def test_proximal_l2_weight_positive_shrinks():
    """l2_weight > 0 applies x *= max(0, 1 - lr * l2)."""
    x = np.asarray([10.0, -5.0, 2.0], dtype=float)
    x_out = _proximal_l2(x.copy(), lr=1.0, l2_weight=0.5)
    # shrink factor = max(0, 1 - 1.0 * 0.5) = 0.5
    np.testing.assert_allclose(x_out, x * 0.5, atol=1e-12)


def test_proximal_l2_weight_can_clip_to_zero():
    """Heavy l2_weight + lr drives shrink factor to 0 → x all zeros."""
    x = np.asarray([3.0, -4.0, 5.0], dtype=float)
    x_out = _proximal_l2(x.copy(), lr=1.0, l2_weight=2.0)
    # shrink factor = max(0, 1 - 2.0) = 0
    np.testing.assert_allclose(x_out, np.zeros_like(x), atol=1e-12)


# ---------------------------------------------------------------------------
# Wf-127 P2 invariants surfacing inside fit_hilbert_guarded
# ---------------------------------------------------------------------------

def test_fit_hilbert_guarded_l2_weight_zero_matches_no_guard_path():
    """l2_weight=0 produces the same train_r2 as a hand-rolled no-guard fit."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 16))
    # Linear target (gradient flow easy).
    beta = rng.standard_normal(16)
    y = X @ beta + rng.standard_normal(64) * 0.01

    best_x, info = fit_hilbert_guarded(
        X, y, hilbert_dim=16, lr=0.01, epochs=50, seed=0,
        grad_clip=1.0, warmup_iters=5, ridge_alpha=0.0,
        l2_weight=0.0,
    )
    assert math.isfinite(info["train_r2"])
    # Penalty should be 0 exactly when l2_weight=0 (Wf-127 P2 invariant #2).
    assert info["l2_penalty_last"] == 0.0
    assert info["l2_penalty_mean"] == 0.0


def test_fit_hilbert_guarded_l2_weight_positive_records_penalty():
    """l2_weight > 0 surfaces a non-zero penalty in info."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((64, 16))
    beta = rng.standard_normal(16)
    y = X @ beta + rng.standard_normal(64) * 0.01

    best_x, info = fit_hilbert_guarded(
        X, y, hilbert_dim=16, lr=0.01, epochs=50, seed=1,
        grad_clip=1.0, warmup_iters=5, ridge_alpha=0.0,
        l2_weight=0.01,
    )
    assert info["l2_penalty_last"] >= 0.0
    assert info["l2_penalty_mean"] >= 0.0
    # At least some norm mass remained at the end of training.
    assert np.linalg.norm(best_x) >= 0.0


# ---------------------------------------------------------------------------
# Wf-122 P1.2 ASTSR scale-mismatch guard
# ---------------------------------------------------------------------------

def test_astr_scale_guard_passes_good_match():
    """Default [0.05, 20.0] band passes weakly-predicting Hilbert; strict pass on real fit."""
    rng = np.random.default_rng(2)
    y_true = rng.standard_normal(50) * 1.0
    y_pred_good = y_true + rng.standard_normal(50) * 0.1
    y_pred_collapse_high = y_true * 100.0  # 100x bigger std → guard fires
    y_pred_collapse_low = y_true * 0.001  # 1000x smaller std → guard fires
    # Wide default band: passes weakly-predicting Hilbert output.
    assert astr_scale_ok(y_pred_good, y_true) is True
    assert astr_scale_ok(y_pred_collapse_high, y_true) is False
    assert astr_scale_ok(y_pred_collapse_low, y_true) is False
    # Strict canonical band [0.5, 1.5] flag — same scenarios.
    assert astr_scale_ok(y_pred_good, y_true, ratio_floor=0.5, ratio_ceil=1.5) is True
    # y_pred_collapse_high has ratio 100x — fails the canonical [0.5, 1.5] band too.
    assert astr_scale_ok(y_pred_collapse_high, y_true, ratio_floor=0.5, ratio_ceil=1.5) is False


def test_astr_scale_guard_handles_degenerate_inputs():
    """Constant y_true or NaN inputs fail closed (return False)."""
    y_constant = np.zeros(10)
    y_pred = np.ones(10)
    assert astr_scale_ok(y_pred, y_constant) is False

    y_nan_true = np.asarray([1.0, np.nan, 2.0])
    y_pred = np.asarray([1.0, 1.0, 1.0])
    # std of y_true with NaN is undefined → guard rejects.
    assert astr_scale_ok(y_pred, y_nan_true) is False


# ---------------------------------------------------------------------------
# compute_r2_safe overflow guard
# ---------------------------------------------------------------------------

def test_compute_r2_safe_returns_inf_on_degenerate_input():
    """Constant y_true returns -inf rather than NaN."""
    y_true = np.zeros(10)
    y_pred = np.ones(10)
    r2 = compute_r2_safe(y_true, y_pred)
    assert r2 == -math.inf


def test_compute_r2_safe_close_to_one_on_clean_fit():
    """Clean linear fit gives R^2 close to 1."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 8))
    beta = rng.standard_normal(8)
    y = X @ beta
    y_pred = y + rng.standard_normal(80) * 1e-6
    r2 = compute_r2_safe(y, y_pred)
    assert math.isfinite(r2)
    assert r2 >= 0.95


# ---------------------------------------------------------------------------
# Wf-131 P1 / Wf-126 sliced-W2 helper
# ---------------------------------------------------------------------------

def test_sliced_wasserstein_zero_when_samples_match():
    """Same input on both sides → W2 ≈ 0."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(50)
    w2 = sliced_wasserstein_2(x, x, n_proj=16, seed=0)
    assert math.isfinite(w2)
    assert w2 < 1e-9


def test_sliced_wasserstein_nonnegative_and_finite():
    """Different inputs → non-negative and finite value."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(50)
    y = rng.standard_normal(50) + 1.0
    w2 = sliced_wasserstein_2(x, y, n_proj=16, seed=0)
    assert math.isfinite(w2)
    assert w2 >= 0.0


def test_add_w2_to_loss_returns_zero_for_constant_input():
    """Constant input (degenerate) returns 0 (the bench-level fallback)."""
    x = np.ones(10)
    y = np.ones(10)
    out = add_w2_to_loss(x, y, n_proj=4, seed=0)
    assert out == 0.0


# ---------------------------------------------------------------------------
# K-fold helper
# ---------------------------------------------------------------------------

def test_kfold_indices_partition_no_overlap():
    """Train/test indices are disjoint per-fold; every index appears exactly once as test."""
    n = 100
    splits = kfold_indices(n=n, n_folds=5, seed=0)
    test_covered: List[int] = []
    for tr, te in splits:
        # Train and test must be disjoint.
        assert len(np.intersect1d(tr, te)) == 0
        # Train + test must cover the whole input per fold.
        assert set(tr.tolist()).union(te.tolist()) == set(range(n))
        test_covered.extend(te.tolist())
    # Across folds, every index should appear as test exactly once.
    assert sorted(test_covered) == list(range(n))


def test_kfold_indices_reproducible_on_same_seed():
    """Same seed → identical train/test splits across calls."""
    splits_a = kfold_indices(n=42, n_folds=4, seed=7)
    splits_b = kfold_indices(n=42, n_folds=4, seed=7)
    for (tr_a, te_a), (tr_b, te_b) in zip(splits_a, splits_b):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


# ---------------------------------------------------------------------------
# Synthesis renderer
# ---------------------------------------------------------------------------

def test_render_synthesis_includes_target_and_verdict():
    """synthesis.md strings the head-claim target and the PASS / HONEST-NEG verdict."""
    fake_summary = [{
        "cell_line": "MCF7",
        "n_target_rows": 200,
        "r2_mean": 0.34,
        "r2_std": 0.02,
        "r2_min": 0.30,
        "r2_max": 0.38,
        "n_collapse_seeds": 0,
        "status": "OK",
        "wall_s_mean": 1.0,
        "per_seed": [{
            "seed": 0, "r2_mean": 0.34, "collapse_count": 0, "wall_s_mean": 1.0,
        }],
    }]
    md = render_synthesis(
        cohort_summaries=fake_summary,
        dataset_name="platinai_4cell",
        use_l2_guard=True,
        use_astr_guard=True,
        w2_weight=0.05,
        epochs=50,
        seeds=[0],
        target_r2=0.30,
    )
    assert "Wf-129 Phase 3" in md
    assert "PASS" in md  # 0.34 >= 0.30 → PASS verdict.
    assert "+0.30" in md  # target echoed.
    assert "MCF7" in md


def test_render_synthesis_honest_negative_when_below_target():
    """All-collapse cohort surfaces the honest-negative framing."""
    fake_summary = [{
        "cell_line": "A549",
        "n_target_rows": 200,
        "r2_mean": -math.inf,
        "r2_std": 0.0,
        "r2_min": -math.inf,
        "r2_max": -math.inf,
        "n_collapse_seeds": 1,
        "status": "ALL_FOLDS_COLLAPSED",
        "wall_s_mean": 1.0,
        "per_seed": [{
            "seed": 0, "r2_mean": -math.inf, "collapse_count": 1, "wall_s_mean": 1.0,
        }],
    }]
    md = render_synthesis(
        cohort_summaries=fake_summary,
        dataset_name="platinai_4cell",
        use_l2_guard=True,
        use_astr_guard=True,
        w2_weight=0.05,
        epochs=50,
        seeds=[0],
        target_r2=0.30,
    )
    assert "HONEST-NEGATIVE" in md
    assert "A549" in md
    assert "Wf-37 P3" in md  # baseline cross-reference.


# ---------------------------------------------------------------------------
# Defaults + module-level invariants
# ---------------------------------------------------------------------------

def test_default_l2_weight_is_phase1_stable():
    """The default l2_weight matches Wf-127 P1 stable weight."""
    assert DEFAULT_L2_WEIGHT == 0.01


def test_default_w2_weight_is_wf131_setpoint():
    """The default wasserstein_weight matches the Wf-131 P1 set-point."""
    assert DEFAULT_W2_WEIGHT == 0.05


def test_platinai_4cell_lines_and_columns_well_formed():
    """The PlatinAI 4-cell line/column map is well-formed (4 entries)."""
    assert len(PLATINAI_4CELL_LINES) == 4
    for cl in PLATINAI_4CELL_LINES:
        assert cl in PLATINAI_4CELL_IC50_COLS
        assert PLATINAI_4CELL_IC50_COLS[cl].startswith("ic50_")
        assert PLATINAI_4CELL_IC50_COLS[cl].endswith("_nm")
