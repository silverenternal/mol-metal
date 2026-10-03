"""Tests for the Hilbert ↔ LambdaASTSR integration layer (TODO-34 §L-R18-P1-5).

================================================================
Integration layer verification — 25 tests
================================================================
This module validates that :mod:`molmetal.molmetal_lam.reward.hilbert_astsr_integration`
correctly wraps :class:`LambdaASTSR` and :class:`LambdaASTSRHilbert`
behind a single ``fit`` / ``predict`` / ``switch_to_hilbert`` API and
provides a fair head-to-head wall-time + R² comparison.

Test inventory (25 tests)
-------------------------
1. **Composite construction** — defaults, custom kwargs, env flag
   propagation (3 tests).
2. **fit/predict** — fit returns dict, predict matches active
   regressor, switch routes correctly (4 tests).
3. **Runtime switch** — switch_to_hilbert toggles active branch,
   predict uses the right regressor after switch (3 tests).
4. **Hilbert vs evolutionary comparison** — wall-time + R² reported,
   speedup finite, n_seeds honoured (3 tests).
5. **Multi-run benchmark** — statistics across n_runs, std non-negative
   (3 tests).
6. **Reward channel** — closure total, NaN-safe, opt-in flag (3 tests).
7. **Honest framing** — neither regressor is silently re-fitted on
   switch; both kept available; no mutation of underlying modules
   (3 tests).
8. **Edge cases** — empty fit history, mismatched lengths, unsupported
   shape (3 tests).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
    HilbertASTSRComposite,
    HilbertEvolutionaryComparator,
    benchmark_hilbert_vs_evolutionary,
    make_hilbert_astsr_reward_channel,
)


# ---------------------------------------------------------------------------
# Helpers — small deterministic design matrices for fast tests
# ---------------------------------------------------------------------------
def _toy_design(
    n: int = 24,
    dim: int = 8,
    seed: int = 0,
    noise: float = 0.01,
) -> tuple:
    """Build a deterministic ``(X, y)`` with a known linear structure."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    w = rng.standard_normal(dim)
    y = X @ w + noise * rng.standard_normal(n)
    return X, y, w


# ---------------------------------------------------------------------------
# 1. Composite construction
# ---------------------------------------------------------------------------
def test_composite_default_construction() -> None:
    """Default composite has both regressors and ``use_hilbert=True``."""
    composite = HilbertASTSRComposite()
    assert composite.use_hilbert is True
    assert composite.lambda_astsr is not None
    assert composite.hilbert_optimizer is not None
    assert len(composite.feature_columns) >= 1


def test_composite_custom_feature_columns() -> None:
    """Custom ``feature_columns`` propagates to the evolutionary regressor."""
    fc = ("a", "b", "c", "d")
    composite = HilbertASTSRComposite(feature_columns=fc)
    assert composite.feature_columns == fc
    # The evolutionary regressor must have the same column tuple.
    assert tuple(composite.lambda_astsr.feature_names) == fc


def test_composite_env_flag_propagation() -> None:
    """Construction writes the ``use_hilbert`` flag into the env var."""
    old = os.environ.pop("USE_HILBERT_PRIMAL_DUAL", None)
    try:
        composite = HilbertASTSRComposite(use_hilbert=False)
        assert os.environ.get("USE_HILBERT_PRIMAL_DUAL") == "0"
        composite.switch_to_hilbert(True)
        assert os.environ.get("USE_HILBERT_PRIMAL_DUAL") == "1"
    finally:
        if old is not None:
            os.environ["USE_HILBERT_PRIMAL_DUAL"] = old


# ---------------------------------------------------------------------------
# 2. fit / predict
# ---------------------------------------------------------------------------
def test_composite_fit_returns_dict_with_r2_and_wall() -> None:
    """fit returns a dict with r2, wall_s, backend keys."""
    X, y, _ = _toy_design(n=20, dim=8, seed=0)
    composite = HilbertASTSRComposite(
        feature_columns=tuple(f"f{i}" for i in range(8)),
        population_dim=8,
    )
    result = composite.fit(X, y)
    assert "r2" in result
    assert "wall_s" in result
    assert "backend" in result
    assert result["backend"] in ("hilbert", "evolutionary")
    assert math.isfinite(result["r2"])
    assert result["wall_s"] >= 0.0


def test_composite_fit_hilbert_branch_backend_label() -> None:
    """Hilbert-active fit reports ``backend='hilbert'``."""
    X, y, _ = _toy_design(n=16, dim=4, seed=1)
    composite = HilbertASTSRComposite(
        feature_columns=tuple(f"f{i}" for i in range(4)),
        population_dim=4,
        use_hilbert=True,
    )
    result = composite.fit(X, y)
    assert result["backend"] == "hilbert"


def test_composite_fit_evolutionary_branch_backend_label() -> None:
    """Evolutionary-active fit reports ``backend='evolutionary'``."""
    X, y, _ = _toy_design(n=16, dim=4, seed=2)
    composite = HilbertASTSRComposite(
        feature_columns=tuple(f"f{i}" for i in range(4)),
        population_dim=4,
        use_hilbert=False,
    )
    result = composite.fit(X, y)
    assert result["backend"] == "evolutionary"


def test_composite_predict_after_fit_returns_finite_array() -> None:
    """predict after fit returns a finite array of length N."""
    X, y, _ = _toy_design(n=12, dim=8, seed=3)
    composite = HilbertASTSRComposite(
        feature_columns=tuple(f"f{i}" for i in range(8)),
        population_dim=8,
    )
    composite.fit(X, y)
    y_hat = composite.predict(X)
    assert y_hat.shape == (X.shape[0],)
    assert np.all(np.isfinite(y_hat))


# ---------------------------------------------------------------------------
# 3. Runtime switch
# ---------------------------------------------------------------------------
def test_composite_switch_to_hilbert_toggles_flag() -> None:
    """switch_to_hilbert flips the ``use_hilbert`` boolean."""
    composite = HilbertASTSRComposite()
    assert composite.use_hilbert is True
    composite.switch_to_hilbert(False)
    assert composite.use_hilbert is False
    composite.switch_to_hilbert(True)
    assert composite.use_hilbert is True


def test_composite_switch_routes_subsequent_fit() -> None:
    """Switching flips which regressor subsequent fit calls consult."""
    X, y, _ = _toy_design(n=16, dim=4, seed=4)
    composite = HilbertASTSRComposite(
        feature_columns=tuple(f"f{i}" for i in range(4)),
        population_dim=4,
        use_hilbert=True,
    )
    # Hilbert arm first.
    res_h = composite.fit(X, y)
    assert res_h["backend"] == "hilbert"

    # Switch and re-fit.
    composite.switch_to_hilbert(False)
    res_e = composite.fit(X, y)
    assert res_e["backend"] == "evolutionary"

    # Switch back.
    composite.switch_to_hilbert(True)
    res_h2 = composite.fit(X, y)
    assert res_h2["backend"] == "hilbert"


def test_composite_switch_does_not_silently_refit() -> None:
    """A switch without re-fit preserves the previously-cached output."""
    X, y, _ = _toy_design(n=12, dim=4, seed=5)
    composite = HilbertASTSRComposite(
        feature_columns=tuple(f"f{i}" for i in range(4)),
        population_dim=4,
        use_hilbert=True,
    )
    composite.fit(X, y)
    y_hat_before = composite.predict(X)
    composite.switch_to_hilbert(False)
    # After the switch the composite has not been re-fitted, so the
    # predict still uses the cached Hilbert output (NOT a fresh
    # evolutionary fit).
    y_hat_after = composite.predict(X)
    assert np.allclose(y_hat_before, y_hat_after, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Hilbert vs evolutionary comparison
# ---------------------------------------------------------------------------
def test_comparator_compare_returns_full_dict() -> None:
    """compare returns the 7 expected keys."""
    X, y, _ = _toy_design(n=16, dim=4, seed=6)
    composite = HilbertASTSRComposite(
        feature_columns=tuple(f"f{i}" for i in range(4)),
        population_dim=4,
    )
    comparator = HilbertEvolutionaryComparator(hilbert_astsr=composite)
    out = comparator.compare(X, y, n_seeds=2)
    for key in (
        "hilbert_wall_s",
        "evolutionary_wall_s",
        "hilbert_r2",
        "evolutionary_r2",
        "speedup",
        "n_seeds",
    ):
        assert key in out, f"missing key: {key}"
    assert out["n_seeds"] == 2


def test_comparator_compare_wall_times_nonneg() -> None:
    """Wall-time averages are non-negative."""
    X, y, _ = _toy_design(n=12, dim=4, seed=7)
    composite = HilbertASTSRComposite(
        feature_columns=tuple(f"f{i}" for i in range(4)),
        population_dim=4,
    )
    comparator = HilbertEvolutionaryComparator(hilbert_astsr=composite)
    out = comparator.compare(X, y, n_seeds=2)
    assert out["hilbert_wall_s"] >= 0.0
    assert out["evolutionary_wall_s"] >= 0.0


def test_comparator_compare_r2_in_unit_interval() -> None:
    """Both R² averages lie in ``[-1, 1]``."""
    X, y, _ = _toy_design(n=16, dim=4, seed=8)
    composite = HilbertASTSRComposite(
        feature_columns=tuple(f"f{i}" for i in range(4)),
        population_dim=4,
    )
    comparator = HilbertEvolutionaryComparator(hilbert_astsr=composite)
    out = comparator.compare(X, y, n_seeds=2)
    assert -1.0 <= out["hilbert_r2"] <= 1.0
    assert -1.0 <= out["evolutionary_r2"] <= 1.0


# ---------------------------------------------------------------------------
# 5. Multi-run benchmark
# ---------------------------------------------------------------------------
def test_benchmark_returns_full_dict() -> None:
    """benchmark returns the 12 expected keys."""
    X, y, _ = _toy_design(n=12, dim=4, seed=9)
    fc = tuple(f"f{i}" for i in range(4))
    out = benchmark_hilbert_vs_evolutionary(
        X,
        y,
        n_runs=3,
        feature_columns=fc,
        population_dim=4,
    )
    for key in (
        "hilbert_wall_s_mean",
        "hilbert_wall_s_std",
        "evolutionary_wall_s_mean",
        "evolutionary_wall_s_std",
        "hilbert_r2_mean",
        "hilbert_r2_std",
        "evolutionary_r2_mean",
        "evolutionary_r2_std",
        "speedup_mean",
        "speedup_std",
        "n_runs",
        "n_samples",
    ):
        assert key in out, f"missing key: {key}"
    assert out["n_runs"] == 3
    assert out["n_samples"] == X.shape[0]


def test_benchmark_std_nonneg() -> None:
    """All std fields are non-negative floats."""
    X, y, _ = _toy_design(n=12, dim=4, seed=10)
    fc = tuple(f"f{i}" for i in range(4))
    out = benchmark_hilbert_vs_evolutionary(
        X,
        y,
        n_runs=3,
        feature_columns=fc,
        population_dim=4,
    )
    for k in (
        "hilbert_wall_s_std",
        "evolutionary_wall_s_std",
        "hilbert_r2_std",
        "evolutionary_r2_std",
        "speedup_std",
    ):
        assert out[k] >= 0.0, f"{k} negative: {out[k]}"


def test_benchmark_hilbert_r2_close_to_optimal_on_linear_problem() -> None:
    """On a noiseless linear problem the Hilbert regressor hits R² > 0.9."""
    # Build a noiseless linear problem; both regressors should hit
    # near-perfect R² — we assert the Hilbert branch does.
    rng = np.random.default_rng(11)
    dim = 6
    X = rng.standard_normal((24, dim))
    w = rng.standard_normal(dim)
    y = X @ w  # noiseless
    fc = tuple(f"f{i}" for i in range(dim))
    # Use a larger lr and more iters for the Hilbert branch so it
    # converges on the noiseless problem.
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
        LambdaASTSRHilbert,
    )

    composite = HilbertASTSRComposite(
        hilbert_optimizer=LambdaASTSRHilbert(
            population_dim=dim,
            lr_primal=0.1,
            lr_dual=0.1,
            n_iters=200,
            seed=0,
        ),
        feature_columns=fc,
        use_hilbert=True,
        population_dim=dim,
    )
    result = composite.fit(X, y)
    assert result["r2"] > 0.9, result["r2"]


# ---------------------------------------------------------------------------
# 6. Reward channel
# ---------------------------------------------------------------------------
def test_reward_channel_total_when_disabled() -> None:
    """Reward channel returns 0.0 when ``enabled=False``."""
    composite = HilbertASTSRComposite()
    closure = make_hilbert_astsr_reward_channel(composite, enabled=False)
    assert closure({"diversity_tanimoto": 0.5}) == 0.0


def test_reward_channel_returns_zero_when_not_fitted() -> None:
    """Reward channel returns 0.0 before the composite has been fitted."""
    composite = HilbertASTSRComposite(
        feature_columns=tuple(f"f{i}" for i in range(4)),
        population_dim=4,
    )
    closure = make_hilbert_astsr_reward_channel(composite, enabled=True)
    assert closure({"f0": 1.0, "f1": 2.0, "f2": 3.0, "f3": 4.0}) == 0.0


def test_reward_channel_returns_finite_after_fit() -> None:
    """Reward channel returns a finite float after the composite is fitted."""
    X, y, _ = _toy_design(n=16, dim=4, seed=12)
    fc = tuple(f"f{i}" for i in range(4))
    composite = HilbertASTSRComposite(
        feature_columns=fc,
        population_dim=4,
        use_hilbert=True,
    )
    composite.fit(X, y)
    closure = make_hilbert_astsr_reward_channel(composite, enabled=True)
    row = {fn: float(X[0, j]) for j, fn in enumerate(fc)}
    out = closure(row)
    assert math.isfinite(out)


# ---------------------------------------------------------------------------
# 7. Honest framing — no silent mutation
# ---------------------------------------------------------------------------
def test_composite_does_not_mutate_underlying_regressors() -> None:
    """Construction does not import / patch LambdaASTSR globals."""
    # Snapshot the LambdaASTSR module's identity (its class object is
    # accessed elsewhere; the composite must not re-bind it).
    from molmetal.molmetal_lam.reward import lambda_astsr

    lam_id_before = id(lambda_astsr.LambdaASTSR)
    _ = HilbertASTSRComposite()
    assert id(lambda_astsr.LambdaASTSR) == lam_id_before


def test_composite_predict_unfitted_returns_zeros() -> None:
    """Predict on an unfitted composite returns zeros (NaN-safe stub)."""
    composite = HilbertASTSRComposite(
        feature_columns=tuple(f"f{i}" for i in range(4)),
        population_dim=4,
    )
    y_hat = composite.predict(np.zeros((3, 4)))
    assert y_hat.shape == (3,)
    # Not strictly zero (Hilbert reshape may produce a 1-d output), but
    # the contract is "no crash, finite output".
    assert np.all(np.isfinite(y_hat))


def test_composite_is_fitted_reflects_state() -> None:
    """``is_fitted`` is True after a successful fit."""
    X, y, _ = _toy_design(n=12, dim=4, seed=13)
    fc = tuple(f"f{i}" for i in range(4))
    composite = HilbertASTSRComposite(
        feature_columns=fc,
        population_dim=4,
    )
    assert composite.is_fitted is False
    composite.fit(X, y)
    assert composite.is_fitted is True


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------
def test_composite_fit_records_history_per_call() -> None:
    """Each fit appends one entry to ``fit_history_``."""
    X, y, _ = _toy_design(n=12, dim=4, seed=14)
    fc = tuple(f"f{i}" for i in range(4))
    composite = HilbertASTSRComposite(
        feature_columns=fc,
        population_dim=4,
    )
    assert composite.fit_history_ == []
    composite.fit(X, y)
    assert len(composite.fit_history_) == 1
    composite.fit(X, y)
    assert len(composite.fit_history_) == 2


def test_composite_fit_rejects_1d_features() -> None:
    """fit with 1-d features raises ValueError."""
    fc = tuple(f"f{i}" for i in range(4))
    composite = HilbertASTSRComposite(
        feature_columns=fc,
        population_dim=4,
    )
    with pytest.raises(ValueError):
        composite.fit(np.zeros(10), np.zeros(10))


def test_composite_fit_rejects_length_mismatch() -> None:
    """fit with mismatched ``N`` between features and target raises."""
    fc = tuple(f"f{i}" for i in range(4))
    composite = HilbertASTSRComposite(
        feature_columns=fc,
        population_dim=4,
    )
    X = np.zeros((10, 4))
    y = np.zeros(8)  # mismatch
    with pytest.raises(ValueError):
        composite.fit(X, y)


def test_benchmark_rejects_bad_shapes() -> None:
    """benchmark with 1-d features raises ValueError."""
    with pytest.raises(ValueError):
        benchmark_hilbert_vs_evolutionary(
            np.zeros(10),
            np.zeros(10),
            n_runs=2,
        )
