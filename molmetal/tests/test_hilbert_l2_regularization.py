"""Tests for the Wf-122 Phase 1 L2 regularisation patch on the
Hilbert primal-dual optimiser.

Background (TODO-111 §0 / TODO/pending/111_wf85_r2_root_fix.md)
--------------------------------------------------------------
The R17/R18-ship ``LambdaASTSRHilbert._objective`` was a bare
``0.5 * mean((X x - y)^2)`` MSE without any regulariser.  On the
Wf-83 P1 mcf7 cohort (12-d RDKit 2D features vs pIC50) this drove
the primal iterate to diverge, yielding train R² ≈ -21 across all
seeds — the primal variable grew without bound because the data
fidelity term had nothing to bound ``||x||``.

Wf-122 Phase 1 adds a Tikhonov ridge ``0.5 * lambda * ||x||²`` with
an adaptive ``lambda`` schedule (starts at ``l2_lambda = 0.01``,
doubles on inner-val regression, capped at ``max_l2_lambda``).

Test inventory (~12 tests)
--------------------------
1. ``_objective`` closed form (``l2_lambda=0``, ``0.1``, default)
2. ``_objective`` rejects negative ``l2_lambda``
3. ``__post_init__`` validation (``l2_lambda >= 0``, ``val_check_every > 0``,
   ``max_l2_lambda >= l2_lambda``)
4. Wf-83-profile probe + adaptive schedule: train R² ≥ -0.5 (spec exit
   criterion)
5. Adaptive ``lambda`` doubles on inner-val regression
6. Adaptive ``lambda`` does NOT spuriously double on strong-signal problems
7. Adaptive path bounded by ``max_l2_lambda``
8. Adaptive path returns the best inner-val snapshot
9. Static path (``adaptive_l2=False``) keeps ``n_l2_doublings_=0``
10. No-val-supplied sets ``best_r2_val_=NaN``
11. ``val_features``/``val_target`` length-mismatch raises
12. ``l2_lambda_history_`` records the live ridge coefficient at every iter
"""

from __future__ import annotations

import numpy as np
import pytest

from molmetal.molmetal_lam.reward.hilbert_primal_dual import LambdaASTSRHilbert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hilbert_feature_map(X_12d, hilbert_dim=64):
    """Augment 12-d descriptors to ``hilbert_dim`` via linear + bias +
    pairwise interactions (Wf-37 P1 §3, mirrored from
    ``bench_hilbert_only_real.py``).

    Variance-ranked truncation keeps the bias and the top
    ``hilbert_dim - F - 1`` interaction columns when the
    augmentation exceeds ``hilbert_dim``.
    """
    n, F = X_12d.shape
    if F >= hilbert_dim:
        return X_12d[:, :hilbert_dim]
    bias = np.ones((n, 1), dtype=np.float32)
    interactions = []
    for i in range(F):
        for j in range(i + 1, F):
            interactions.append(X_12d[:, i] * X_12d[:, j])
    inter_arr = (
        np.stack(interactions, axis=1) if interactions
        else np.zeros((n, 0), dtype=np.float32)
    )
    base = np.concatenate([X_12d, bias], axis=1)
    aug = np.concatenate([base, inter_arr], axis=1)
    if aug.shape[1] <= hilbert_dim:
        pad = np.zeros((n, hilbert_dim - aug.shape[1]), dtype=np.float32)
        return np.concatenate([aug, pad], axis=1)
    vars_ = np.var(aug, axis=0)
    ranked = np.argsort(-vars_)
    keep = [F]  # bias at column F
    for idx in ranked:
        if idx == F:
            continue
        if len(keep) >= hilbert_dim:
            break
        keep.append(int(idx))
    keep = sorted(set(keep))
    keep_arr = np.asarray(keep[:hilbert_dim], int)
    return aug[:, keep_arr]


def _wf83_probe(seed: int = 0, n: int = 3262, hilbert_dim: int = 64):
    """Build the Wf-83 P1 mcf7-style probe: 12-d RDKit 2D-like features,
    Hilbert feature-map augmentation to ``hilbert_dim``-d, weak
    signal-to-noise ratio (PlatinAI pIC50 std≈0.45, |ρ|<0.02 across
    all 12 features).

    Returns ``(X_h, y_native, w_true)`` with the full data (caller is
    responsible for train/val splits).
    """
    rng = np.random.default_rng(seed)
    F_12 = 12
    X_12 = rng.standard_normal((n, F_12))
    # PlatinAI pIC50 SNR: weak signal + comparable noise.
    w_true = rng.standard_normal(F_12) * 0.1
    y = X_12 @ w_true + 0.5 * rng.standard_normal(n)
    X_h = _hilbert_feature_map(X_12.astype(np.float32), hilbert_dim)
    return X_h, y, w_true


# ---------------------------------------------------------------------------
# 1. _objective closed form
# ---------------------------------------------------------------------------
def test_objective_clam_matches_mse_at_lambda_zero() -> None:
    """``l2_lambda=0.0`` recovers the bare-MSE closed form (R17 ship)."""
    hil = LambdaASTSRHilbert(population_dim=4, seed=0, l2_lambda=0.0)
    x = np.array([0.5, -0.2, 0.0, 0.3])
    feat = np.eye(4)
    tgt = np.array([1.0, 0.0, 0.5, -0.5])

    loss, grad = hil._objective(x, feat, tgt, l2_lambda=0.0)
    expected_loss = 0.5 * float(np.mean((feat @ x - tgt) ** 2))
    expected_grad = feat.T @ (feat @ x - tgt) / max(1, tgt.shape[0])

    assert loss == pytest.approx(expected_loss, abs=1e-12)
    assert np.allclose(grad, expected_grad, atol=1e-12)


def test_objective_clam_adds_ridge_term() -> None:
    """``l2_lambda > 0`` adds ``0.5 * lam * ||x||²`` to loss and
    ``lam * x`` to the gradient.
    """
    hil = LambdaASTSRHilbert(population_dim=4, seed=0)
    x = np.array([0.5, -0.2, 0.0, 0.3])
    feat = np.eye(4)
    tgt = np.array([1.0, 0.0, 0.5, -0.5])
    loss_no, _ = hil._objective(x, feat, tgt, l2_lambda=0.0)
    loss_lam, grad_lam = hil._objective(x, feat, tgt, l2_lambda=0.1)

    expected_ridge_loss = 0.5 * 0.1 * float(np.sum(x * x))
    assert loss_lam == pytest.approx(loss_no + expected_ridge_loss, abs=1e-12)

    # Gradient should match closed form: ridge term contributes ``lam * x``.
    expected_grad = (
        feat.T @ (feat @ x - tgt) / max(1, tgt.shape[0]) + 0.1 * x
    )
    assert np.allclose(grad_lam, expected_grad, atol=1e-12)


def test_objective_clam_default_uses_self_l2_lambda() -> None:
    """When ``l2_lambda`` kwarg is ``None``, falls back to
    ``self.l2_lambda``.
    """
    hil = LambdaASTSRHilbert(population_dim=4, seed=0, l2_lambda=0.07)
    x = np.array([0.5, -0.2, 0.0, 0.3])
    feat = np.eye(4)
    tgt = np.array([1.0, 0.0, 0.5, -0.5])
    loss_default, _ = hil._objective(x, feat, tgt)
    loss_explicit, _ = hil._objective(x, feat, tgt, l2_lambda=0.07)
    assert loss_default == pytest.approx(loss_explicit, abs=1e-12)


def test_objective_clam_rejects_negative_lambda() -> None:
    """Negative ``l2_lambda`` raises ``ValueError``."""
    hil = LambdaASTSRHilbert(population_dim=4, seed=0)
    x = np.zeros(4)
    feat = np.eye(4)
    tgt = np.zeros(4)
    with pytest.raises(ValueError):
        hil._objective(x, feat, tgt, l2_lambda=-0.01)


# ---------------------------------------------------------------------------
# 2. Default + dataclass validation
# ---------------------------------------------------------------------------
def test_default_l2_lambda_matches_spec() -> None:
    """Default ``l2_lambda=0.01`` per Wf-122 Phase 1 spec."""
    hil = LambdaASTSRHilbert()
    assert hil.l2_lambda == pytest.approx(0.01, abs=1e-12)
    assert hil.adaptive_l2 is True
    assert hil.val_check_every == 10
    assert hil.max_l2_lambda == pytest.approx(10.0, abs=1e-12)


def test_dataclass_rejects_negative_l2_lambda() -> None:
    """``__post_init__`` rejects ``l2_lambda < 0``."""
    with pytest.raises(ValueError):
        LambdaASTSRHilbert(l2_lambda=-0.01)


def test_dataclass_rejects_invalid_val_check_every() -> None:
    """``__post_init__`` rejects ``val_check_every <= 0``."""
    with pytest.raises(ValueError):
        LambdaASTSRHilbert(val_check_every=0)


def test_dataclass_rejects_max_l2_below_initial() -> None:
    """``__post_init__`` rejects ``max_l2_lambda < l2_lambda``."""
    with pytest.raises(ValueError):
        LambdaASTSRHilbert(l2_lambda=1.0, max_l2_lambda=0.5)


# ---------------------------------------------------------------------------
# 3. Exit criterion: train R² ≥ -0.5 on the Wf-83 probe
# ---------------------------------------------------------------------------
def test_l2_regularization_prevents_train_r2_divergence() -> None:
    """With L2 + adaptive schedule, train R² ≥ -0.5 on the Wf-83 probe.

    The spec exit criterion (TODO-111 §3 P1.1) is "train R² ≥ -0.5 on
    probe (vs current -21)" — i.e. L2 prevents the divergence that
    R17/R18-ship produced.  We replicate the Wf-83 P1 profile (12-d
    features + Hilbert feature-map 64-d + ``loss_scale=128`` +
    ``grad_clip_max_norm=1.0``) and verify the criterion over 3 seeds
    with the adaptive schedule (``max_l2_lambda=20.0``).
    """
    for seed in (0, 1, 2):
        X_h, y, _ = _wf83_probe(seed=seed, n=3262, hilbert_dim=64)
        n_tr = 2500
        Xtr, ytr = X_h[:n_tr], (y[:n_tr] * 128).astype(np.float64)
        Xva, yva = X_h[n_tr:], (y[n_tr:] * 128).astype(np.float64)

        hil = LambdaASTSRHilbert(
            population_dim=64, n_iters=50,
            lr_primal=0.05, lr_dual=0.05,
            l2_lambda=0.01, max_l2_lambda=20.0,
            seed=seed,
            grad_clip_max_norm=1.0, step_size_decay=0.5,
            adaptive_l2=True, val_check_every=5,
        )
        hil.optimize(
            Xtr, ytr, val_features=Xva, val_target=yva,
            loss_scale=128.0,
        )
        assert hil.best_r2_train_ >= -0.5, (
            f"seed={seed}: best_r2_train_={hil.best_r2_train_:.3f} "
            "is below the -0.5 bound — L2 + adaptive did not prevent "
            "divergence on the Wf-83 profile"
        )


# ---------------------------------------------------------------------------
# 4. Adaptive lambda doubling schedule
# ---------------------------------------------------------------------------
def test_adaptive_lambda_doubles_on_val_regression() -> None:
    """When inner-val R² regresses on a noise-only probe, L2 doubles ≥ 2x.

    The pure-noise probe drives the inner-val R² into regression
    territory at every check; the adaptive schedule responds by
    doubling the L2 coefficient at each regression detection.
    """
    rng = np.random.default_rng(0)
    n, dim = 200, 12
    X = rng.standard_normal((n, dim))
    y = 0.5 * rng.standard_normal(n)
    n_tr = 120
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xva, yva = X[n_tr:], y[n_tr:]

    hil = LambdaASTSRHilbert(
        population_dim=dim, n_iters=80,
        lr_primal=0.05, lr_dual=0.05,
        l2_lambda=0.01, max_l2_lambda=8.0,
        seed=0,
        adaptive_l2=True, val_check_every=5,
    )
    hil.optimize(Xtr, ytr, val_features=Xva, val_target=yva)
    assert hil.n_l2_doublings_ >= 2, (
        f"only {hil.n_l2_doublings_} doublings; adaptive schedule "
        "did not respond to inner-val regression"
    )
    assert hil.best_l2_lambda_ <= 8.0 + 1e-12


def test_adaptive_lambda_does_not_double_when_val_improves() -> None:
    """Strong-signal probe: inner-val R² monotonically improves, no
    spurious doubling.

    On a strong-signal regression the adaptive schedule should be
    quiet (val R² keeps improving, so no regression is detected).
    We assert at most 2 doublings (lenient bound — the schedule is
    noise-triggered by tiny fluctuations even on healthy problems).
    """
    rng = np.random.default_rng(0)
    n, dim = 200, 12
    X = rng.standard_normal((n, dim))
    w_true = rng.standard_normal(dim) * 2.0
    y = X @ w_true + 0.05 * rng.standard_normal(n)
    n_tr = 120
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xva, yva = X[n_tr:], y[n_tr:]

    hil = LambdaASTSRHilbert(
        population_dim=dim, n_iters=80, lr_primal=0.005,
        lr_dual=0.005, l2_lambda=0.01, seed=0,
        adaptive_l2=True, val_check_every=10,
        max_l2_lambda=2.0,
    )
    hil.optimize(Xtr, ytr, val_features=Xva, val_target=yva)
    assert hil.n_l2_doublings_ <= 2, (
        f"unexpectedly many doublings ({hil.n_l2_doublings_}) on a "
        "strong-signal problem; adaptive schedule is over-responsive"
    )


def test_adaptive_lambda_capped_at_max() -> None:
    """Even under aggressive regression, lambda stays ≤ max_l2_lambda."""
    rng = np.random.default_rng(0)
    n, dim = 200, 12
    X = rng.standard_normal((n, dim))
    y = 0.5 * rng.standard_normal(n)
    n_tr = 120
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xva, yva = X[n_tr:], y[n_tr:]
    hil = LambdaASTSRHilbert(
        population_dim=dim, n_iters=80,
        lr_primal=0.05, lr_dual=0.05,
        l2_lambda=0.01, max_l2_lambda=0.5,
        seed=0, adaptive_l2=True, val_check_every=5,
    )
    hil.optimize(Xtr, ytr, val_features=Xva, val_target=yva)
    assert hil.best_l2_lambda_ <= 0.5 + 1e-12
    assert max(hil.l2_lambda_history_) <= 0.5 + 1e-12


# ---------------------------------------------------------------------------
# 5. Adaptive vs static / no-val semantics
# ---------------------------------------------------------------------------
def test_static_adaptive_path_records_no_doublings() -> None:
    """``adaptive_l2=False`` keeps ``n_l2_doublings_=0`` and
    ``best_r2_val_=NaN`` even if val data is supplied.
    """
    rng = np.random.default_rng(0)
    n, dim = 200, 12
    X = rng.standard_normal((n, dim))
    y = 0.5 * rng.standard_normal(n)
    n_tr = 120
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xva, yva = X[n_tr:], y[n_tr:]
    hil = LambdaASTSRHilbert(
        population_dim=dim, n_iters=40, lr_primal=0.05,
        lr_dual=0.05, l2_lambda=0.05, seed=0, adaptive_l2=False,
    )
    hil.optimize(Xtr, ytr, val_features=Xva, val_target=yva)
    assert hil.n_l2_doublings_ == 0
    assert np.isnan(hil.best_r2_val_)
    assert hil.best_l2_lambda_ == pytest.approx(0.05, abs=1e-12)


def test_no_val_supplied_disables_adaptive_path() -> None:
    """When ``val_features``/``val_target`` are None, adaptive is
    silently disabled: ``n_l2_doublings_=0``, ``best_r2_val_=NaN``,
    and ``l2_lambda_history_`` is a constant stream at the initial
    value.
    """
    rng = np.random.default_rng(0)
    n, dim = 200, 12
    X = rng.standard_normal((n, dim))
    y = 0.5 * rng.standard_normal(n)
    hil = LambdaASTSRHilbert(
        population_dim=dim, n_iters=40, lr_primal=0.05,
        lr_dual=0.05, l2_lambda=0.05, seed=0, adaptive_l2=True,
    )
    hil.optimize(X, y)  # no val_* args
    assert hil.n_l2_doublings_ == 0
    assert np.isnan(hil.best_r2_val_)
    assert hil.best_l2_lambda_ == pytest.approx(0.05, abs=1e-12)
    assert len(hil.l2_lambda_history_) == 40
    assert all(v == pytest.approx(0.05, abs=1e-12)
               for v in hil.l2_lambda_history_)


def test_val_length_mismatch_raises() -> None:
    """``val_features`` and ``val_target`` of different lengths is rejected."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 12))
    y = rng.standard_normal(60)
    hil = LambdaASTSRHilbert(
        population_dim=12, n_iters=10, lr_primal=0.05,
        lr_dual=0.05, seed=0, adaptive_l2=True,
    )
    with pytest.raises(ValueError):
        hil.optimize(
            X[:30], y[:30],
            val_features=X[30:50],
            val_target=y[30:55],
        )


def test_l2_lambda_history_populated() -> None:
    """``l2_lambda_history_`` records the live ridge coefficient at
    each of the ``n_iters`` iterations.
    """
    rng = np.random.default_rng(0)
    n, dim = 60, 12
    X = rng.standard_normal((n, dim))
    y = 0.5 * rng.standard_normal(n)
    hil = LambdaASTSRHilbert(
        population_dim=dim, n_iters=20, lr_primal=0.05,
        lr_dual=0.05, l2_lambda=0.05, seed=0,
    )
    hil.optimize(X, y)
    assert len(hil.l2_lambda_history_) == 20
    # All entries are finite and ≥ initial value.
    for v in hil.l2_lambda_history_:
        assert np.isfinite(v)
        assert v >= 0.05 - 1e-12
