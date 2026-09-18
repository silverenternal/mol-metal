"""Tests for the ``lam_chem`` module: AST + HeuristicRegressor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from molmetal.molmetal_lam.lam_chem import (
    HeuristicRegressor,
    LamAbs,
    LamApp,
    LamVar,
    apply,
    lam,
    var,
)


# ---------------------------------------------------------------------------
# AST: pretty-printing
# ---------------------------------------------------------------------------
def test_lamvar_to_string() -> None:
    """LamVar('x') -> 'x' (just the bare identifier)."""
    assert LamVar("x").to_string() == "x"


def test_lamabs_to_string() -> None:
    """LamAbs(LamVar('x'), LamVar('x')) -> 'λx.x'."""
    assert LamAbs(LamVar("x"), LamVar("x")).to_string() == "λx.x"


def test_lamapp_to_string() -> None:
    """LamApp(LamVar('f'), LamVar('x')) -> '(f x)'."""
    assert LamApp(LamVar("f"), LamVar("x")).to_string() == "(f x)"


# ---------------------------------------------------------------------------
# AST: substitution and β-reduction
# ---------------------------------------------------------------------------
def test_subst_simple() -> None:
    """Substituting ``y`` for free ``x`` in ``λx.x`` keeps ``λx.x``.

    Under capture-avoiding substitution the *binder* shadows the
    replacement: occurrences of ``x`` inside ``λx.…`` are bound, not
    free. So ``λx.x`` is unchanged. The substitution is observable
    when ``x`` appears *outside* the binder, which the second assertion
    verifies.
    """
    identity = LamAbs(LamVar("x"), LamVar("x"))
    one = LamVar("1")
    result = identity.subst("x", one)
    # Binder shadows; substitution leaves the term alone.
    assert result == identity
    assert result.to_string() == "λx.x"

    # However substituting ``y`` into ``λx.(x y)`` for free ``y``
    # does rewrite the body.
    term = lam(var("x"), apply(var("x"), var("y")))
    rewritten = term.subst("y", var("z"))
    assert rewritten == lam(var("x"), apply(var("x"), var("z")))


def test_beta_reduce_one_step() -> None:
    """(λx.x) y  →β  y  (the identity function applied to ``y``)."""
    term = apply(lam(var("x"), var("x")), var("y"))
    reduced = term.beta_reduce()
    assert reduced is not None
    assert reduced == var("y")


# ---------------------------------------------------------------------------
# HeuristicRegressor
# ---------------------------------------------------------------------------
def test_heuristic_regressor_runs() -> None:
    """Fit a near-linear synthetic surface and check predict is sensible.

    We use ``y = 2*x0 + x1 + 0.05 * noise`` — close to linear so the
    :class:`~sklearn.linear_model.Ridge` fallback scores well. PySR
    (when available) will discover the exact closed form. We allow up
    to ``max_error=1.5`` (RMSE) which is comfortably above the noise
    floor but rules out a degenerate model.
    """
    rng = np.random.default_rng(42)
    n_train = 100
    X_train = rng.standard_normal((n_train, 3))
    y_train = 2.0 * X_train[:, 0] + X_train[:, 1] + 0.05 * rng.standard_normal(n_train)

    reg = HeuristicRegressor(niterations=10, random_state=0)
    reg.fit(X_train, y_train)
    assert reg.backend_ in {"pysr", "sklearn-linear", "sklearn-rf"}

    X_test = rng.standard_normal((20, 3))
    y_pred = reg.predict(X_test)

    # Build the true values for comparison.
    y_true = 2.0 * X_test[:, 0] + X_test[:, 1]
    rmse = math.sqrt(float(np.mean((y_pred - y_true) ** 2)))
    assert rmse < 1.5, f"RMSE too high: {rmse:.3f}"

    # equation() must always return a non-empty string.
    assert isinstance(reg.equation(), str)
    assert reg.equation() != "<unfitted>"


# ---------------------------------------------------------------------------
# Bonus: capture-avoiding substitution (not in the spec but exercised by
# the LamApp beta-reduction tests above).
# ---------------------------------------------------------------------------
def test_capture_avoiding_subst() -> None:
    """Substituting ``λy.w`` for ``x`` in ``λy.(x y)`` must rename the
    *inner* ``y`` to avoid capture by the outer binder.

    The classic problem: ``(λy.(x y))[x := λy.w]`` naïvely yields
    ``λy.((λy.w) y)`` — the inner ``y`` becomes bound by the outer
    ``λy`` (variable shadowing / capture). Capture-avoiding substitution
    must rename the *inner* binder to a fresh name ``__y{N}``.
    """
    target = lam(var("y"), apply(var("x"), var("y")))
    replacement = lam(var("y"), var("w"))
    rewritten = target.subst("x", replacement)
    assert isinstance(rewritten, LamAbs)
    # Outer binder is unchanged.
    assert rewritten.var.name == "y"
    # Body is now (λ__y1.w) y; the inner LamAbs binder is renamed.
    body = rewritten.body
    assert isinstance(body, LamApp)
    inner = body.func
    assert isinstance(inner, LamAbs)
    assert inner.var.name != "y"
    assert inner.var.name.startswith("__y")
    # Free variables of the resulting term: only {w} should remain free.
    assert rewritten.free_vars() == frozenset({"w"})


# ---------------------------------------------------------------------------
# Bonus: full β-NF on the classic Church numeral identity.
# ---------------------------------------------------------------------------
def test_to_normal_form_church_identity() -> None:
    """(λx.x) a b  →β  a b  (multi-step β-NF)."""
    term = apply(apply(lam(var("x"), var("x")), var("a")), var("b"))
    nf = term.to_normal_form()
    assert nf == apply(var("a"), var("b"))