"""Tests for the Lambda AST Symbolic Regression (Deflexpressor-style) module.

This module validates the L-R17-B3 deliverable — the swap of F5
(linear+hinge) for a Deflexpressor-style λ-AST symbolic regressor in
the lambda-model coupling adapter.

Test design
-----------
1. ``test_random_ast_depth_bound`` — every random AST respects ``max_depth``.
2. ``test_evaluate_on_synthetic_row`` — evaluation is finite & deterministic.
3. ``test_evaluate_matrix_shape`` — ``evaluate_matrix`` returns ``(N,)``.
4. ``test_compute_r2_perfect`` — R²=1.0 when prediction equals target.
5. ``test_compute_r2_constant`` — R²=0.0 when prediction is the mean.
6. ``test_fit_deterministic`` — same seed → same best R².
7. ``test_fit_toy_r2_above_threshold`` — the verify gate: on a 4-feature
   toy problem with structure ``y ≈ 2*x0 + 1.5*x1 - x2``, an ASTSR fit
   with 100 generations / 64 population must hit R² > 0.5.
8. ``test_30_cell_holdout_r2_above_f5_baseline`` — on the real 30-cell
   hold-out from the metallo drug pool, ASTSR R² must exceed the F5
   baseline (R² >= 0.62).
9. ``test_lambda_string_round_trip`` — the Deflexpressor λ-string is
   syntactically valid (starts with ``λ``).
10. ``test_swap_f5_for_astsr`` — the F5 → ASTSR swap mutates
    :mod:`molmetal_lam.reward.learned_shaping` correctly.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from molmetal.molmetal_lam.reward.lambda_astsr import (
    DEFAULT_FEATURE_COLUMNS,
    LambdaAST,
    LambdaASTSR,
    compute_r2,
    crossover_ast,
    default_feature_columns,
    legacy_f5_feature_columns,
    make_lambda_astsr_reward_channel,
    mutate_ast,
    random_ast,
    swap_f5_for_astsr,
)


# ---------------------------------------------------------------------------
# 1. Random AST construction
# ---------------------------------------------------------------------------
def test_random_ast_depth_bound() -> None:
    """``random_ast`` never produces an AST deeper than ``max_depth``."""
    rng = random.Random(0)
    for d in (1, 2, 3, 4, 5):
        for _ in range(40):
            ast = random_ast(
                rng,
                ("f0", "f1", "f2"),
                max_depth=d,
                p_var=0.45,
            )
            assert ast.depth <= d, f"depth {ast.depth} exceeds max_depth {d}"


def test_random_ast_uses_only_provided_features() -> None:
    """Random AST leaves reference only the supplied feature names."""
    rng = random.Random(1)
    features = ("alpha", "beta")
    for _ in range(20):
        ast = random_ast(rng, features, max_depth=3, p_var=0.7)
        used = ast.used_features()
        assert used.issubset(
            set(features)
        ), f"AST references features outside pool: {used - set(features)}"


def test_default_feature_columns() -> None:
    """Legacy F5 5-feature set is reachable via the backward-compat shim.

    Wf-49 Phase 2: ``default_feature_columns()`` now defaults to the
    **F5_RDKIT_2D** 12-d RDKit 2D set.  The legacy F5 5-feature set
    is preserved at ``DEFAULT_FEATURE_COLUMNS`` and reachable via
    ``default_feature_columns(feature_set="f5_legacy")`` or
    ``legacy_f5_feature_columns()``.  This test pins down the F5
    backward-compat contract.
    """
    cols = legacy_f5_feature_columns()
    assert isinstance(cols, tuple)
    assert cols == DEFAULT_FEATURE_COLUMNS
    assert "diversity_tanimoto" in cols
    assert "sa_mean_norm" in cols
    # Same answer through the ``feature_set=`` flag.
    cols_via_flag = default_feature_columns(feature_set="f5_legacy")
    assert cols_via_flag == DEFAULT_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 2. Evaluation
# ---------------------------------------------------------------------------
def test_evaluate_on_synthetic_row() -> None:
    """A manually constructed AST evaluates to the expected scalar."""
    # AST: add(const 2.0, var f0)  →  2.0 + f0
    ops = (
        {"kind": "const", "value": 2.0},
        {"kind": "var", "name": "f0"},
        {"kind": "op", "op": "add", "left": 0, "right": 1},
    )
    ast = LambdaAST(lam_node=None, ops=ops, depth=2, n_leaves=2)
    val = ast.evaluate({"f0": 3.5})
    assert val == pytest.approx(5.5, abs=1e-9)


def test_evaluate_is_nan_safe() -> None:
    """NaN inputs are mapped to 0.0; no propagation."""
    ops = (
        {"kind": "var", "name": "f0"},
        {"kind": "const", "value": 1.0},
        {"kind": "op", "op": "add", "left": 0, "right": 1},
    )
    ast = LambdaAST(lam_node=None, ops=ops, depth=2, n_leaves=2)
    val = ast.evaluate({"f0": float("nan")})
    assert val == pytest.approx(1.0, abs=1e-9)
    val = ast.evaluate({})
    assert val == pytest.approx(1.0, abs=1e-9)


def test_evaluate_matrix_shape() -> None:
    """evaluate_matrix returns (N,) finite float array."""
    rng = random.Random(0)
    ast = random_ast(rng, ("a", "b"), max_depth=3)
    X = np.array([[0.1, 0.2], [0.5, 0.6], [0.9, 0.1]])
    out = ast.evaluate_matrix(X, ("a", "b"))
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 3. R² helpers
# ---------------------------------------------------------------------------
def test_compute_r2_perfect() -> None:
    """R² = 1.0 when y_pred == y_true."""
    y = np.array([0.1, 0.5, 0.9, 1.3])
    assert compute_r2(y, y) == pytest.approx(1.0, abs=1e-9)


def test_compute_r2_constant_prediction() -> None:
    """R² = 0.0 when prediction is the mean of y."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y, np.mean(y))
    assert compute_r2(y, y_pred) == pytest.approx(0.0, abs=1e-9)


def test_compute_r2_zero_variance() -> None:
    """R² = 0.0 (degenerate) when y_true is constant."""
    y = np.full(10, 0.5)
    y_pred = np.linspace(0.0, 1.0, 10)
    assert compute_r2(y, y_pred) == 0.0


def test_compute_r2_nan_safe() -> None:
    """R² is 0.0 when either array contains NaN."""
    y = np.array([1.0, 2.0, np.nan, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_r2(y, y_pred) == 0.0


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------
def test_fit_deterministic_same_seed() -> None:
    """Two fits with the same seed yield the same best R²."""
    rng = np.random.default_rng(0)
    N, F = 40, 4
    X = rng.uniform(0, 1, (N, F))
    y = 2.0 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2]
    a = LambdaASTSR(
        depth=3,
        population=32,
        generations=20,
        seed=7,
        feature_names=("a", "b", "c", "d"),
    )
    a.fit(X, y)
    b = LambdaASTSR(
        depth=3,
        population=32,
        generations=20,
        seed=7,
        feature_names=("a", "b", "c", "d"),
    )
    b.fit(X, y)
    assert a.best_r2_train_ == pytest.approx(b.best_r2_train_, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. The verify-gate tests
# ---------------------------------------------------------------------------
def test_fit_toy_r2_above_threshold() -> None:
    """Verify gate 1: ASTSR must hit R² > 0.5 on a 4-feature toy problem.

    The toy target is ``y ≈ 2*x0 + 1.5*x1 - x2`` (a sparse linear
    function of 3 of the 4 features).  This is intentionally easy for
    the ASTSR — the verify threshold of 0.5 confirms the evolutionary
    search converges on a useful hypothesis within 100 generations.
    """
    rng = np.random.default_rng(42)
    N = 80
    F = 4
    X = rng.uniform(0.0, 1.0, (N, F))
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.0 * X[:, 2] + 0.1
    y += rng.normal(0.0, 0.02, N)  # small noise

    astsr = LambdaASTSR(
        depth=4,
        population=64,
        generations=100,
        seed=42,
        feature_names=("f0", "f1", "f2", "f3"),
    )
    astsr.fit(X, y)
    y_pred = astsr.predict(X)
    r2 = compute_r2(y, y_pred)
    assert r2 > 0.5, (
        f"ASTSR failed to reach R² > 0.5 on the toy 4-feature problem; "
        f"got R²={r2:.4f}"
    )


def test_30_cell_holdout_r2_above_f5_baseline() -> None:
    """Verify gate 2: on the metallo drug pool 30-cell hold-out, ASTSR R²
    must exceed the F5 baseline (R² >= 0.62).

    The 30/33 split is deterministic (seed=42) and matches the
    production split used by ``molmetal_lam.reward.symbolic_regression``.
    The reward target is the canonical 3-feature composite
    ``r = sa_mean_norm + diversity_tanimoto + metal_compliance_rate``,
    which the ASTSR can recover exactly (the target is literally a sum
    of three of its available features, so R²=1.0 is achievable).
    """
    from molmetal.molmetal_lam.reward.symbolic_regression import (
        build_reward_target,
        load_measured_cells,
    )

    df = load_measured_cells()
    df["reward"] = build_reward_target(df)
    df = df.dropna(subset=["reward"]).reset_index(drop=True)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    train_idx, test_idx = perm[:33], perm[33:]
    assert len(test_idx) >= 30, "30-cell hold-out contract violated"

    features = (
        "diversity_tanimoto",
        "diversity_homotype",
        "sa_mean_norm",
        "metal_compliance_rate",
    )
    X_train = df.iloc[train_idx][list(features)].fillna(0.0).to_numpy(dtype=float)
    X_test = df.iloc[test_idx][list(features)].fillna(0.0).to_numpy(dtype=float)
    y_train = df.iloc[train_idx]["reward"].to_numpy(dtype=float)
    y_test = df.iloc[test_idx]["reward"].to_numpy(dtype=float)

    astsr = LambdaASTSR(
        depth=4,
        population=64,
        generations=50,
        seed=42,
        feature_names=features,
    )
    astsr.fit(X_train, y_train)
    y_pred = astsr.predict(X_test)
    r2 = compute_r2(y_test, y_pred)
    assert r2 >= 0.62, (
        f"ASTSR failed to exceed the F5 baseline (R² ≥ 0.62) on the "
        f"30-cell hold-out; got R²={r2:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. Mutation / crossover sanity
# ---------------------------------------------------------------------------
def test_mutate_ast_preserves_validity() -> None:
    """Mutation always returns an AST whose evaluate is finite."""
    rng = random.Random(2)
    ast = random_ast(rng, ("a", "b", "c"), max_depth=4)
    for _ in range(30):
        m = mutate_ast(ast, rng, ("a", "b", "c"), max_depth=4)
        # Either we kept the original (mutation refused) or we got a
        # new tree that still evaluates without raising.
        v = m.evaluate({"a": 0.5, "b": 0.3, "c": 0.9})
        assert np.isfinite(v)


def test_crossover_ast_preserves_validity() -> None:
    """Crossover always returns an AST whose evaluate is finite."""
    rng = random.Random(3)
    a = random_ast(rng, ("x", "y"), max_depth=3)
    b = random_ast(rng, ("x", "y"), max_depth=3)
    for _ in range(30):
        c = crossover_ast(a, b, rng)
        v = c.evaluate({"x": 0.5, "y": 0.5})
        assert np.isfinite(v)


# ---------------------------------------------------------------------------
# 7. Lambda string round-trip (Deflexpressor form)
# ---------------------------------------------------------------------------
def test_lambda_string_round_trip() -> None:
    """The Deflexpressor λ-string starts with the binder character."""
    rng = random.Random(0)
    ast = random_ast(rng, ("f", "g"), max_depth=3)
    s = ast.to_lambda_string()
    assert s.startswith(
        "λ"
    ), f"λ-string must start with the binder character; got {s[:20]}"
    # The string must reference at least one feature or constant.
    assert any(c.isalnum() for c in s)


def test_lisp_print_is_readable() -> None:
    """The LISP print contains the operation names."""
    rng = random.Random(0)
    ast = random_ast(rng, ("a", "b"), max_depth=2)
    s = ast.to_lisp()
    # At depth 2 we expect at least one operator symbol.
    assert "(" in s and ")" in s


# ---------------------------------------------------------------------------
# 8. Reward channel + swap integration
# ---------------------------------------------------------------------------
def test_make_lambda_astsr_reward_channel_disabled() -> None:
    """The reward channel returns 0.0 when the env gate is OFF."""
    os.environ["LAMBDA_ASTSR_ENABLED"] = "0"
    ch = make_lambda_astsr_reward_channel(enabled=False)
    assert ch({"sa_mean_norm": 0.5}) == 0.0


def test_make_lambda_astsr_reward_channel_enabled() -> None:
    """The reward channel returns the ASTSR prediction when enabled."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (30, 3))
    y = X[:, 0] + X[:, 1]
    astsr = LambdaASTSR(
        depth=3,
        population=32,
        generations=10,
        seed=0,
        feature_names=("a", "b", "c"),
    )
    astsr.fit(X, y)
    ch = make_lambda_astsr_reward_channel(astsr=astsr, enabled=True)
    val = ch({"a": 0.7, "b": 0.3, "c": 0.0})
    # Value is finite and in a plausible range (target ~ 1.0).
    assert np.isfinite(val)


def test_swap_f5_for_astsr_mutates_module(monkeypatch) -> None:
    """``swap_f5_for_astsr`` replaces ``load_best_formula`` in learned_shaping."""
    from molmetal import molmetal_lam  # noqa: F401 - triggers package imports

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (40, 3))
    y = X[:, 0] + 0.5 * X[:, 1]
    astsr = LambdaASTSR(
        depth=3,
        population=16,
        generations=5,
        seed=0,
        feature_names=("a", "b", "c"),
    )
    astsr.fit(X, y)

    from molmetal.molmetal_lam.reward import learned_shaping as ls_module

    # Snapshot the original load_best_formula so we can restore it.
    original_load = ls_module.load_best_formula
    try:
        swap_f5_for_astsr(ls_module, astsr)
        # Now load_best_formula should return the ASTSR formula.
        new_formula = ls_module.load_best_formula()
        assert new_formula.family == "ASTSR"
        assert "Yu/Yang/Ren/Zhao 2026" in new_formula.lit_anchor
    finally:
        ls_module.load_best_formula = original_load
