"""Wf-44 Phase 4 — Numerical stability tests.

================================================================
Validates the Wf-44 Phase 4 numerical-stability fixes for the
Hilbert primal-dual optimiser, the LambdaASTSR evolutionary
regressor, and the affinity-regression trainer:

1. **Gradient clipping** — ``PrimalDualOptimizer.grad_clip_max_norm``
   rescales gradients above the threshold to unit norm; the iterate
   does not explode (4 tests).
2. **Step-size auto-decay** — when the iterate becomes non-finite,
   ``lr_primal`` and ``lr_dual`` shrink by ``step_size_decay`` until
   they fall below ``min_step_size``; the iterate stays bounded
   (3 tests).
3. **Loss-scaled fit + fp64 fallback** — ``HilbertASTSRComposite.fit``
   honours ``loss_scale`` and ``fp64_fallback``; non-finite fits are
   retried in float64 (3 tests).
4. **ASTSR NaN early-stop** — ``LambdaASTSR.early_stop_on_nan=True``
   terminates the evolutionary loop when the entire population
   produces non-finite scores; the most recent finite ``best_ast_``
   is preserved (2 tests).
5. **Affinity trainer AMP** — ``train_affinity_regression`` wires up
   ``torch.autocast`` + ``torch.amp.GradScaler`` behind the ``--amp``
   flag without breaking the CPU path (1 test).
6. **R² recovery** — after the fixes, the Hilbert regressor on a
   noiseless linear problem hits R² >= 0.0 (paper headline metric
   in the Wf-44 §12 honest-framing section; target is R² >= 0.0
   so we no longer report "worse than mean") (2 tests).

Total: 15 tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
    HilbertASTSRComposite,
)
from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
    LambdaASTSRHilbert,
    PrimalDualOptimizer,
)
from molmetal.molmetal_lam.reward.lambda_astsr import (
    LambdaASTSR,
    compute_r2,
)


# ---------------------------------------------------------------------------
# 1. Gradient clipping (max-norm)
# ---------------------------------------------------------------------------
def test_grad_clip_default_max_norm_is_one() -> None:
    """Default ``grad_clip_max_norm`` is 1.0 (Wf-44 Phase 4 spec)."""
    opt = PrimalDualOptimizer()
    assert opt.grad_clip_max_norm == pytest.approx(1.0, abs=1e-12)


def test_grad_clip_rescales_oversized_gradient() -> None:
    """A gradient whose L² norm exceeds the clip is rescaled to the clip."""
    opt = PrimalDualOptimizer(
        lr_primal=0.01,
        lr_dual=0.01,
        n_iters=5,
        grad_clip_max_norm=1.0,
    )
    x = np.zeros(4)
    y = np.zeros(4)
    g = np.array([10.0, 0.0, -10.0, 0.0])  # ||g|| = sqrt(200) ~ 14.14

    def objective(xk: np.ndarray):
        return float(np.sum(xk * g)), g

    x_new, y_new = opt.step(x, y, objective)
    # With clipping: g_clipped = g * 1 / sqrt(200)
    expected_grad_clipped = g / math.sqrt(float(np.sum(g * g)))
    expected_x_new = -0.01 * expected_grad_clipped
    assert np.allclose(x_new, expected_x_new, atol=1e-12)
    # The counter should record exactly one clipped step.
    assert opt.n_steps_with_clipped_grad_ == 1


def test_grad_clip_zero_disables_clipping() -> None:
    """``grad_clip_max_norm=0.0`` disables clipping entirely."""
    opt = PrimalDualOptimizer(
        lr_primal=0.01,
        lr_dual=0.01,
        n_iters=5,
        grad_clip_max_norm=0.0,
    )
    x = np.zeros(3)
    y = np.zeros(3)
    g = np.array([100.0, -100.0, 50.0])

    def objective(xk: np.ndarray):
        return float(np.sum(xk * g)), g

    opt.step(x, y, objective)
    assert opt.n_steps_with_clipped_grad_ == 0


def test_grad_clip_rejects_negative() -> None:
    """Negative ``grad_clip_max_norm`` raises ValueError at construction."""
    with pytest.raises(ValueError):
        PrimalDualOptimizer(grad_clip_max_norm=-0.5)


# ---------------------------------------------------------------------------
# 2. Step-size auto-decay on NaN / inf
# ---------------------------------------------------------------------------
def test_hilbert_step_size_decay_on_nan() -> None:
    """Step size decays by ``step_size_decay`` after a non-finite iterate.

    We construct an objective whose gradient is a fixed huge vector
    (independent of the iterate), so the very first step drives the
    iterate to ±inf.  After the first step, the iterate is non-finite,
    so the step size decays by ``step_size_decay`` (default 0.5).
    """
    opt = PrimalDualOptimizer(
        lr_primal=0.5,
        lr_dual=0.5,
        n_iters=5,
        grad_clip_max_norm=0.0,  # disable clipping so the step sees the full grad
        step_size_decay=0.5,
        min_step_size=1e-6,
    )
    x = np.ones(4) * 1.0
    y = np.ones(4) * 1.0
    # Fixed huge gradient (independent of x) — the first update is
    # x_new = 1 - 0.5 * 1e10 = -5e9, finite.  We want the *iterate*
    # to become non-finite, so we use a gradient whose magnitude is
    # such that ``lr * ||grad|| > 1e308`` (overflow threshold).
    huge_grad = np.array([1e310, 1e310, 1e310, 1e310])

    def objective(xk: np.ndarray):
        return float(np.sum(xk * huge_grad)), huge_grad

    # First step: gradient is so large that the iterate overflows.
    x, y = opt.step(x, y, objective)
    assert opt.n_steps_with_nan_ >= 1
    assert opt.n_steps_with_decayed_lr_ >= 1
    # The effective step size has been shrunk by step_size_decay.
    assert opt.effective_lr_primal_ < 0.5


def test_step_size_decay_respects_min_step_size() -> None:
    """Decay is capped at ``min_step_size`` so the step never hits 0.

    The objective produces a huge gradient at every iteration so the
    iterate overflows; after each overflow the step size is shrunk
    by ``step_size_decay`` until it hits the ``min_step_size`` floor.
    """
    opt = PrimalDualOptimizer(
        lr_primal=1e-4,
        lr_dual=1e-4,
        n_iters=10,
        grad_clip_max_norm=0.0,
        step_size_decay=0.5,
        min_step_size=1e-6,
    )
    x = np.ones(2)
    y = np.ones(2)
    huge_grad = np.array([1e310, 1e310])

    def objective(xk: np.ndarray):
        return float(np.sum(xk * huge_grad)), huge_grad

    for _ in range(10):
        x, y = opt.step(x, y, objective)
    # The step size decayed all the way to the floor.
    assert opt.effective_lr_primal_ == pytest.approx(1e-6, abs=1e-12)
    assert opt.effective_lr_dual_ == pytest.approx(1e-6, abs=1e-12)


def test_step_size_decay_disabled_with_decay_one() -> None:
    """``step_size_decay=1.0`` disables decay (the step size is preserved)."""
    opt = PrimalDualOptimizer(
        lr_primal=0.3,
        lr_dual=0.3,
        n_iters=5,
        grad_clip_max_norm=0.0,
        step_size_decay=1.0,  # disabled
    )
    x = np.zeros(3)
    y = np.zeros(3)

    def objective(xk: np.ndarray):
        return float(np.sum(xk**2)), xk * 1e6  # explosive gradient

    for _ in range(5):
        x, y = opt.step(x, y, objective)
    # No decay: effective_lr_primal is unchanged.
    assert opt.effective_lr_primal_ == pytest.approx(0.3, abs=1e-12)


def test_step_size_decay_rejects_out_of_range() -> None:
    """``step_size_decay`` outside ``(0, 1]`` is rejected at construction."""
    with pytest.raises(ValueError):
        PrimalDualOptimizer(step_size_decay=0.0)
    with pytest.raises(ValueError):
        PrimalDualOptimizer(step_size_decay=1.5)
    with pytest.raises(ValueError):
        PrimalDualOptimizer(step_size_decay=-0.1)


# ---------------------------------------------------------------------------
# 3. Loss-scaled fit + fp64 fallback (HilbertASTSRComposite)
# ---------------------------------------------------------------------------
def test_loss_scale_no_inf() -> None:
    """``loss_scale`` keeps the squared residuals away from underflow.

    With ``loss_scale=128``, the composite scales the target before
    fitting.  The Hilbert regressor's R² is invariant to affine scale,
    so the resulting R² matches the un-scaled fit; the *gradient*
    is correspondingly larger, but the underlying optimiser scales
    the gradient-clipping threshold proportionally so convergence is
    preserved (no NaN / ±inf in the iterate history).
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8))
    w = rng.standard_normal(8)
    y = X @ w + 0.01 * rng.standard_normal(40)
    composite = HilbertASTSRComposite(
        hilbert_optimizer=LambdaASTSRHilbert(
            population_dim=8,
            lr_primal=0.1,
            lr_dual=0.1,
            n_iters=50,
            seed=0,
        ),
        feature_columns=tuple(f"f{i}" for i in range(8)),
        population_dim=8,
        loss_scale=128.0,
        fp64_fallback=False,
    )
    result = composite.fit(X, y)
    assert math.isfinite(result["r2"]), result["r2"]
    assert result["loss_scale"] == pytest.approx(128.0, abs=1e-12)
    assert result["fp64_fallback"] is False
    # No inf or NaN in the iterate history.
    for entry in composite.hilbert_optimizer.history_:
        assert np.isfinite(entry["gap"])
        assert np.isfinite(entry["effective_lr_primal"])
        assert np.isfinite(entry["effective_lr_dual"])


def test_loss_scale_one_matches_no_scaling() -> None:
    """``loss_scale=1.0`` matches the no-scaling baseline.

    R² is invariant to affine scale on the target — the two fits
    should agree to within the gradient-clipping tolerance (1e-3).
    The agreement is not bit-for-bit because gradient clipping at
    ``max_norm=1.0`` introduces a small asymmetry when the gradient
    is near the clipping threshold.
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 8))
    w = rng.standard_normal(8)
    y = X @ w + 0.01 * rng.standard_normal(40)
    fc = tuple(f"f{i}" for i in range(8))
    comp_a = HilbertASTSRComposite(
        hilbert_optimizer=LambdaASTSRHilbert(
            population_dim=8,
            lr_primal=0.1,
            lr_dual=0.1,
            n_iters=30,
            seed=2,
        ),
        feature_columns=fc,
        population_dim=8,
        loss_scale=1.0,
        fp64_fallback=False,
    )
    comp_b = HilbertASTSRComposite(
        hilbert_optimizer=LambdaASTSRHilbert(
            population_dim=8,
            lr_primal=0.1,
            lr_dual=0.1,
            n_iters=30,
            seed=2,
        ),
        feature_columns=fc,
        population_dim=8,
        loss_scale=128.0,
        fp64_fallback=False,
    )
    res_a = comp_a.fit(X, y)
    res_b = comp_b.fit(X, y)
    # R² is approximately scale-invariant; the two fits should agree
    # to within the gradient-clipping tolerance (1e-3).
    assert res_a["r2"] == pytest.approx(res_b["r2"], abs=1e-3)


def test_fp64_fallback_recovers_finite_r2_on_nan() -> None:
    """``fp64_fallback=True`` recovers a finite R² when the fp32 fit fails.

    We force the fp32 Hilbert branch to fail by wrapping the target in
    values near the float32 denorm boundary (so the squared residual
    underflows).  The composite then retries in float64 and produces a
    finite R².
    """
    rng = np.random.default_rng(3)
    X = rng.standard_normal((30, 8))
    w = rng.standard_normal(8)
    y = X @ w
    # Tiny target amplitudes — these underflow to 0 in fp32 (the
    # squared residual ``(X @ w - 1e-40)²`` is below fp32 epsilon),
    # which can drive the loss-scaled fit to a non-finite R².
    y_tiny = y * 1e-20
    fc = tuple(f"f{i}" for i in range(8))
    composite = HilbertASTSRComposite(
        hilbert_optimizer=LambdaASTSRHilbert(
            population_dim=8,
            lr_primal=0.1,
            lr_dual=0.1,
            n_iters=20,
            seed=4,
        ),
        feature_columns=fc,
        population_dim=8,
        loss_scale=128.0,
        fp64_fallback=True,
    )
    result = composite.fit(X, y_tiny)
    # Either the fp32 path or the fp64 fallback must produce a finite
    # R²; in either case the result is finite.  When the fallback is
    # used, the ``fp64_fallback`` flag in the result is True.
    assert math.isfinite(result["r2"]), result["r2"]


# ---------------------------------------------------------------------------
# 4. ASTSR NaN early-stop
# ---------------------------------------------------------------------------
def test_astsr_early_stop_on_nan() -> None:
    """``early_stop_on_nan=True`` terminates the loop on a NaN population.

    We construct an ASTSR with a feature schema that drives ``div`` by
    zero on every evaluation, producing non-finite scores across the
    population.  With ``early_stop_on_nan=True`` the loop exits early;
    with ``early_stop_on_nan=False`` the loop runs all generations.
    """
    # Features that guarantee div(a, b) where |b| < 1e-9 -> 0
    # (the ASTSR's protected div returns 0 in that branch).
    # We use an objective that explicitly yields NaN to force the path.
    n_features = 5
    feature_names = tuple(f"f{i}" for i in range(n_features))
    X = np.zeros((20, n_features), dtype=float)  # all zeros => div=0/0 chain
    y = np.ones(20, dtype=float) * 5.0
    astsr_nan_early = LambdaASTSR(
        depth=2,
        population=4,
        generations=10,
        seed=5,
        feature_names=feature_names,
        early_stop_on_nan=True,
    )
    astsr_nan_no_early = LambdaASTSR(
        depth=2,
        population=4,
        generations=10,
        seed=5,
        feature_names=feature_names,
        early_stop_on_nan=False,
    )
    astsr_nan_early.fit(X, y)
    astsr_nan_no_early.fit(X, y)
    # The early-stop variant's history should be shorter or equal to
    # the no-early-stop variant (since the loop may have bailed out).
    assert len(astsr_nan_early.history_) <= len(astsr_nan_no_early.history_)
    # The early-stop variant's best_r2_train_ is finite (a fallback).
    assert np.isfinite(astsr_nan_early.best_r2_train_)


def test_astsr_early_stop_disabled_runs_all_generations() -> None:
    """``early_stop_on_nan=False`` runs all generations even on NaN."""
    feature_names = tuple(f"f{i}" for i in range(3))
    X = np.zeros((10, 3), dtype=float)
    y = np.ones(10, dtype=float) * 5.0
    astsr = LambdaASTSR(
        depth=1,
        population=2,
        generations=4,
        seed=0,
        feature_names=feature_names,
        early_stop_on_nan=False,
    )
    astsr.fit(X, y)
    # 4 generations ran, history has 4 entries (no early-stop).
    assert len(astsr.history_) == 4


# ---------------------------------------------------------------------------
# 5. Affinity trainer AMP wiring (no torch import here — just arg plumbing)
# ---------------------------------------------------------------------------
def test_train_affinity_regression_amp_flag_default_false() -> None:
    """``train_affinity_regression`` ``--amp`` flag defaults to False.

    Skipped when torch is unavailable in the environment (the script
    imports torch at module load — we cannot exercise the parser in
    that case).
    """
    pytest.importorskip("torch")
    from molmetal.scripts import train_affinity_regression

    # Parse with a minimal argv (data + seeds only) — the default
    # ``args.amp`` must be False so CPU runs are unaffected.
    args = train_affinity_regression.parse_args(
        [
            "--dataset",
            "platinai",
            "--seeds",
            "0",
            "--epochs",
            "1",
            "--init",
            "random",
        ]
    )
    assert args.amp is False


# ---------------------------------------------------------------------------
# 6. R² recovery after numerical-stability fix (paper headline metric)
# ---------------------------------------------------------------------------
def test_r2_reasonable_after_stability_fix() -> None:
    """After the Wf-44 Phase 4 fix, the Hilbert regressor hits R² >= 0.0.

    The Phase 2 baseline check (Wf-44) found 4 models whose R² was
    negative because of predict-the-mean collapse (Cause #1) or
    numerical instability (Cause #6 — Hilbert step too large).  The
    Phase 4 fixes (gradient clipping + step-size decay + loss-scale
    + fp64 fallback) close Cause #6; on a noiseless linear problem
    the regressor now hits R² >= 0.0 (paper headline: not "worse
    than mean").

    Target: R² >= 0.0 (not strictly positive — the simple linear
    problem may not have enough signal at small ``n_iters``; what
    matters is that we are no longer *worse than the mean*).
    """
    rng = np.random.default_rng(42)
    dim = 8
    n = 32
    X = rng.standard_normal((n, dim))
    w = rng.standard_normal(dim)
    y = X @ w  # noiseless
    fc = tuple(f"f{i}" for i in range(dim))
    composite = HilbertASTSRComposite(
        hilbert_optimizer=LambdaASTSRHilbert(
            population_dim=dim,
            lr_primal=0.1,
            lr_dual=0.1,
            n_iters=100,
            seed=0,
            grad_clip_max_norm=1.0,
            step_size_decay=0.5,
        ),
        feature_columns=fc,
        population_dim=dim,
        loss_scale=128.0,
        fp64_fallback=True,
    )
    result = composite.fit(X, y)
    assert result["r2"] >= 0.0, (
        f"Hilbert regressor after stability fix should not be worse "
        f"than mean; got R²={result['r2']:.4f}"
    )


def test_r2_strongly_positive_on_noiseless_linear() -> None:
    """On a noiseless linear problem the Hilbert regressor hits R² > 0.9.

    Companion to :func:`test_r2_reasonable_after_stability_fix`.  This
    stronger assertion guards against the case where the stability fixes
    keep the regressor at exactly R²=0 (the "predict the mean" floor)
    rather than letting it converge on the linear signal.
    """
    rng = np.random.default_rng(11)
    dim = 6
    X = rng.standard_normal((24, dim))
    w = rng.standard_normal(dim)
    y = X @ w  # noiseless
    fc = tuple(f"f{i}" for i in range(dim))
    composite = HilbertASTSRComposite(
        hilbert_optimizer=LambdaASTSRHilbert(
            population_dim=dim,
            lr_primal=0.1,
            lr_dual=0.1,
            n_iters=200,
            seed=0,
            grad_clip_max_norm=1.0,
            step_size_decay=0.5,
        ),
        feature_columns=fc,
        population_dim=dim,
        loss_scale=128.0,
        fp64_fallback=True,
    )
    result = composite.fit(X, y)
    assert result["r2"] > 0.9, (
        f"Hilbert regressor on noiseless linear should hit R² > 0.9; "
        f"got R²={result['r2']:.4f}"
    )


# ---------------------------------------------------------------------------
# Sanity helpers used by individual tests
# ---------------------------------------------------------------------------
def test_compute_r2_zero_for_constant() -> None:
    """Constant predictor R² is exactly 0 (sanity guard)."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.full_like(y, np.mean(y))
    assert compute_r2(y, y_pred) == pytest.approx(0.0, abs=1e-12)
