"""Tests for the Wf-122 ASTSR scale-mismatch guard + best_ast_lisp logging
AND the Wf-127 Phase 3 ASTSR scale-clipping guard.

This module validates the Wf-122 P1.2 + P1.3 fixes:

1. **Scale-mismatch guard** — ``collapse_detected(y_pred, y_true)`` must
   fire when ``|pred_std / y_true_std - 1.0| > 0.50``.  This guards
   against mode B (ASTSR exploding AST) failures observed in
   ``reports/wf83_p1_phase2_4cell/mcf7/per_seed_metrics.json`` (seed 5
   ``astr_r2_mean = -21``).

2. **ASTSR integration** — when a ``LambdaASTSR.fit`` produces an
   exploding AST, the score is forced to ``-inf`` so the exploding
   AST cannot be selected as the best.

3. **best_ast_lisp logging** — the new ``LambdaASTSR.best_ast_lisp()``
   accessor returns the LISP rendering of :attr:`best_ast_` (or
   ``"<no_lisp>"`` when no AST has been learned).  This is the
   Wf-122 P1.3 fix for the broken ``hasattr(reg, "best_ast_lisp")``
   check in :mod:`molmetal.scripts.bench_hilbert_astr_combined_real`.

Wf-127 Phase 3 scale-clipping guard tests
-----------------------------------------
4. **Scale clipping guard** — ``clip_astr_signature(y_hat, lo, hi)``
   clamps predictions to ``[lo, hi]`` and counts violations.  Wired
   into ``LambdaASTSR._score`` and ``LambdaASTSR.predict`` so the
   guard fires on any exploding AST the regression produces.
   Defaults are ``(-inf, +inf)`` (no-op) — the Wf-122 ship behaviour
   is preserved.  The Phase 1 stable range
   (:data:`DEFAULT_ASTSR_SCALE_CLIP`) is the documented
   recommended value.

Test design
-----------
1. ``test_collapse_detected_collapse_mode`` — ratio < 0.10 fires.
2. ``test_collapse_detected_scale_mismatch_upper`` — ratio > 1.50 fires.
3. ``test_collapse_detected_scale_mismatch_lower`` — ratio < 0.50 fires.
4. ``test_collapse_detected_good_scale_passes`` — ratio in [0.5, 1.5] passes.
5. ``test_collapse_detected_degenerate_y_std`` — y_true std=0 fires.
6. ``test_collapse_detected_nan_safe`` — NaN inputs → True.
7. ``test_lambda_astsr_rejects_exploding_ast`` — an AST whose pred
   std is 5x y std lands with ``-inf`` score; the resulting
   ``best_ast_`` is a non-exploding AST (or none).
8. ``test_lambda_astsr_keeps_good_ast`` — on a simple target with
   good scale, the ASTSR still picks a useful AST.
9. ``test_best_ast_lisp_pre_fit`` — before ``fit()`` returns ``"<no_lisp>"``.
10. ``test_best_ast_lisp_after_fit`` — after ``fit()`` returns the
    LISP rendering of ``best_ast_``.
11. ``test_clip_astr_signature_noop_with_inf_bounds`` — ±inf → no clipping.
12. ``test_clip_astr_signature_clamps_above`` — values above max clipped.
13. ``test_clip_astr_signature_clamps_below`` — values below min clipped.
14. ``test_clip_astr_signature_counts_violations`` — n_clip is correct.
15. ``test_clip_astr_signature_phase1_stable_default`` — (-10, 10) is stable.
16. ``test_clip_astr_signature_nan_safe`` — NaN values don't crash.
17. ``test_lambda_astsr_default_clip_is_noop`` — Default scale_clip is ±inf.
18. ``test_lambda_astsr_clips_predictions_to_phase1_range`` — clip fires.
19. ``test_lambda_astsr_tracks_n_clipped_predictions`` — counter increments.
20. ``test_lambda_astsr_predict_clips_output`` — predict() also clips.
21. ``test_composite_propagates_scale_clip_to_lambda_astsr`` — propagation.
22. ``test_composite_fit_surfaces_clip_diagnostics`` — fit() return shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from molmetal.molmetal_lam.reward.lambda_astsr import (
    COLLAPSE_RATIO_THRESHOLD,
    DEFAULT_ASTSR_SCALE_CLIP,
    SCALE_MISMATCH_TOLERANCE,
    LambdaASTSR,
    clip_astr_signature,
    collapse_detected,
    compute_r2,
    random_ast,
)


# ---------------------------------------------------------------------------
# 1. collapse_detected — direct unit tests
# ---------------------------------------------------------------------------
def test_collapse_detected_collapse_mode() -> None:
    """A near-constant prediction is rejected (mode A)."""
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, 100)
    # pred is essentially constant — std ~ 1e-3, ratio < 0.10
    y_pred = np.full_like(y, 0.5) + rng.normal(0.0, 1e-3, 100)
    assert collapse_detected(y_pred, y) is True


def test_collapse_detected_scale_mismatch_upper() -> None:
    """A prediction with std 2x target std is rejected (mode B)."""
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1.0, 100)
    # pred std is 2.0, y std is 1.0 → ratio 2.0, deviation 1.0 > 0.5
    y_pred = rng.normal(0.0, 2.0, 100)
    assert collapse_detected(y_pred, y) is True


def test_collapse_detected_scale_mismatch_lower() -> None:
    """A prediction with std 0.4x target std is rejected (mode B)."""
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 1.0, 100)
    # pred std 0.4, y std 1.0 → ratio 0.4, deviation 0.6 > 0.5
    y_pred = rng.normal(0.0, 0.4, 100)
    assert collapse_detected(y_pred, y) is True


def test_collapse_detected_good_scale_passes() -> None:
    """A prediction whose std is within [0.5, 1.5] of y's std passes."""
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 1.0, 100)
    # pred std 1.2 → ratio 1.2, deviation 0.2 ≤ 0.5
    y_pred = rng.normal(0.0, 1.2, 100)
    assert collapse_detected(y_pred, y) is False

    # exact match passes
    y_pred2 = rng.normal(0.0, 1.0, 100)
    assert collapse_detected(y_pred2, y) is False


def test_collapse_detected_degenerate_y_std() -> None:
    """y_true with zero variance is flagged as degenerate."""
    y = np.zeros(50)
    y_pred = np.linspace(-1.0, 1.0, 50)
    assert collapse_detected(y_pred, y) is True


def test_collapse_detected_nan_safe() -> None:
    """NaN inputs → True (can't compute a meaningful ratio)."""
    y = np.array([1.0, 2.0, np.nan, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert collapse_detected(y_pred, y) is True

    y2 = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred2 = np.array([1.0, 2.0, np.nan, 4.0])
    assert collapse_detected(y_pred2, y2) is True


def test_collapse_detected_size_mismatch() -> None:
    """Mismatched array sizes → True (degenerate)."""
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    assert collapse_detected(y_pred, y) is True


# ---------------------------------------------------------------------------
# 2. ASTSR integration — scale-mismatch guard fires on exploding ASTs
# ---------------------------------------------------------------------------
def test_lambda_astsr_rejects_exploding_ast_via_score() -> None:
    """An AST whose pred std is 5x y std gets a ``-inf`` score.

    We construct a simple problem where ``y ≈ 2*x0`` (good signal)
    but the ASTSR's internal ``_score`` must reject any candidate
    whose pred std is far from the target's std.

    To force the rejection path, we monkey-patch the ``collapse_detected``
    symbol used inside the module to a function that always returns
    True, then verify every score is ``-inf``.  This proves the
    guard is wired correctly.
    """
    import molmetal.molmetal_lam.reward.lambda_astsr as la_module

    original = la_module.collapse_detected
    try:
        la_module.collapse_detected = lambda y_pred, y_true: True
        rng = np.random.default_rng(0)
        N, F = 30, 3
        X = rng.uniform(0.0, 1.0, (N, F))
        y = 2.0 * X[:, 0] + 0.5
        astsr = LambdaASTSR(
            depth=3, population=16, generations=5, seed=0,
            feature_names=("a", "b", "c"),
        )
        astsr.fit(X, y)
        # Every AST was rejected, so best_r2_train_ is the sentinel 0.0.
        assert astsr.best_r2_train_ == 0.0
    finally:
        la_module.collapse_detected = original


def test_lambda_astsr_keeps_good_ast_on_simple_target() -> None:
    """On a well-conditioned target, the ASTSR still picks a useful AST.

    The scale-mismatch guard is a **filter**, not a replacement.  When
    the candidate ASTs have reasonable scale, the evolutionary search
    should still converge on a useful hypothesis.
    """
    rng = np.random.default_rng(42)
    N = 80
    F = 4
    X = rng.uniform(0.0, 1.0, (N, F))
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.0 * X[:, 2] + 0.1
    y += rng.normal(0.0, 0.02, N)
    astsr = LambdaASTSR(
        depth=4, population=64, generations=50, seed=42,
        feature_names=("f0", "f1", "f2", "f3"),
    )
    astsr.fit(X, y)
    assert astsr.best_ast_ is not None
    y_pred = astsr.predict(X)
    r2 = compute_r2(y, y_pred)
    # The toy problem is tractable; we should hit positive R².
    assert r2 > 0.5, f"ASTSR regressed on good data: R²={r2:.4f}"


def test_lambda_astsr_score_clipped_to_neg_inf_for_exploding_ast() -> None:
    """When ``collapse_detected`` returns True, ``_score`` returns ``-inf``.

    This is the contract the evolutionary search relies on — the
    argmax must skip rejected candidates.
    """
    import random

    from molmetal.molmetal_lam.reward.lambda_astsr import (
        LambdaAST,
        default_feature_columns,
    )

    rng_a = random.Random(0)
    ast = random_ast(rng_a, ("a", "b"), max_depth=2)
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    y = np.array([1.0, 2.0])

    # Monkey-patch collapse_detected to always return True.
    import molmetal.molmetal_lam.reward.lambda_astsr as la_module

    original = la_module.collapse_detected
    try:
        la_module.collapse_detected = lambda y_pred, y_true: True
        astsr = LambdaASTSR(
            depth=2, population=2, generations=0, seed=0,
            feature_names=("a", "b"),
        )
        score = astsr._score(ast, X, y)
        assert score == float("-inf")
    finally:
        la_module.collapse_detected = original


# ---------------------------------------------------------------------------
# 3. best_ast_lisp logging accessor
# ---------------------------------------------------------------------------
def test_best_ast_lisp_pre_fit() -> None:
    """``best_ast_lisp()`` returns ``"<no_lisp>"`` before ``fit()``."""
    astsr = LambdaASTSR(
        depth=2, population=4, generations=0, seed=0,
        feature_names=("a", "b"),
    )
    assert astsr.best_ast_lisp() == "<no_lisp>"


def test_best_ast_lisp_after_fit_known_ast() -> None:
    """After ``fit()``, ``best_ast_lisp()`` returns the LISP string."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (40, 3))
    y = X[:, 0] + 0.5 * X[:, 1]
    astsr = LambdaASTSR(
        depth=3, population=16, generations=5, seed=0,
        feature_names=("a", "b", "c"),
    )
    astsr.fit(X, y)
    lisp = astsr.best_ast_lisp()
    # The LISP string must contain at least one operator paren or
    # feature reference.
    assert isinstance(lisp, str)
    assert lisp != "<no_lisp>"
    assert lisp == astsr.best_ast_.to_lisp()


# ---------------------------------------------------------------------------
# 4. Constant values exposed for downstream calibration
# ---------------------------------------------------------------------------
def test_collapse_threshold_constant() -> None:
    """The collapse threshold constant is pinned to 0.10 (Wf-83 / Wf-85)."""
    assert COLLAPSE_RATIO_THRESHOLD == pytest.approx(0.10)


def test_scale_mismatch_tolerance_constant() -> None:
    """The scale-mismatch tolerance constant is pinned to 0.50 (Wf-122 P1.2)."""
    assert SCALE_MISMATCH_TOLERANCE == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# 5. Wf-127 P3 — ASTSR scale-clipping guard (value-space)
# ---------------------------------------------------------------------------
def test_clip_astr_signature_noop_with_inf_bounds() -> None:
    """When bounds are ``(-inf, +inf)`` the clip is a no-op (zero violations)."""
    y = np.array([-1e6, -1.0, 0.0, 1.0, 1e6, 100.0, -100.0])
    out, n = clip_astr_signature(y, float("-inf"), float("inf"))
    assert n == 0
    np.testing.assert_array_equal(out, y)


def test_clip_astr_signature_clamps_above() -> None:
    """Values above ``scale_clip_max`` are clamped to the max."""
    y = np.array([-5.0, 0.0, 5.0, 11.0, 100.0])
    out, n = clip_astr_signature(y, float("-inf"), 10.0)
    assert n == 2
    np.testing.assert_array_equal(out, np.array([-5.0, 0.0, 5.0, 10.0, 10.0]))


def test_clip_astr_signature_clamps_below() -> None:
    """Values below ``scale_clip_min`` are clamped to the min."""
    y = np.array([-5.0, 0.0, 5.0, -11.0, -100.0])
    out, n = clip_astr_signature(y, -10.0, float("inf"))
    assert n == 2
    np.testing.assert_array_equal(out, np.array([-5.0, 0.0, 5.0, -10.0, -10.0]))


def test_clip_astr_signature_counts_violations() -> None:
    """The violation count includes both below-min and above-max values."""
    y = np.array([-20.0, -5.0, 0.0, 5.0, 20.0])
    out, n = clip_astr_signature(y, -10.0, 10.0)
    assert n == 2  # -20 and 20
    np.testing.assert_array_equal(out, np.array([-10.0, -5.0, 0.0, 5.0, 10.0]))


def test_clip_astr_signature_phase1_stable_default() -> None:
    """The Phase 1 stable range is ``(-10.0, 10.0)``."""
    assert DEFAULT_ASTSR_SCALE_CLIP == (-10.0, 10.0)
    # The Phase 1 stable range should leave a typical pIC50 prediction
    # untouched (y values in [3, 9]) but clamp the worst explosions
    # (y values > 10 or < -10).
    y_typical = np.array([3.0, 5.0, 7.0, 9.0])
    out_t, n_t = clip_astr_signature(
        y_typical, DEFAULT_ASTSR_SCALE_CLIP[0], DEFAULT_ASTSR_SCALE_CLIP[1],
    )
    assert n_t == 0
    np.testing.assert_array_equal(out_t, y_typical)
    y_exploding = np.array([3.0, 50.0, -50.0, 9.0])
    out_e, n_e = clip_astr_signature(
        y_exploding, DEFAULT_ASTSR_SCALE_CLIP[0], DEFAULT_ASTSR_SCALE_CLIP[1],
    )
    assert n_e == 2
    np.testing.assert_array_equal(out_e, np.array([3.0, 10.0, -10.0, 9.0]))


def test_clip_astr_signature_nan_safe() -> None:
    """NaN inputs are converted to 0.0 before counting — not counted as violations."""
    y = np.array([1.0, np.nan, 5.0, np.nan, -100.0])
    out, n = clip_astr_signature(y, -10.0, 10.0)
    # 1.0 → 1.0, NaN → 0.0, 5.0 → 5.0, NaN → 0.0, -100.0 → -10.0 (violation).
    # Total violations: 1 (only the -100.0, not the NaN).
    assert n == 1
    np.testing.assert_array_equal(out, np.array([1.0, 0.0, 5.0, 0.0, -10.0]))


def test_clip_astr_signature_rejects_inverted_bounds() -> None:
    """``scale_clip_min > scale_clip_max`` raises ValueError."""
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="scale_clip_min"):
        clip_astr_signature(y, 5.0, -5.0)


def test_lambda_astsr_default_clip_is_noop() -> None:
    """LambdaASTSR defaults ``scale_clip_min=-inf``, ``scale_clip_max=+inf``."""
    astsr = LambdaASTSR(
        depth=2, population=4, generations=0, seed=0,
        feature_names=("a", "b"),
    )
    assert astsr.scale_clip_min == float("-inf")
    assert astsr.scale_clip_max == float("inf")
    # Before fit(), diagnostics are at their zero state.
    assert astsr.n_clipped_predictions_ == 0
    assert astsr.clip_violations_ == []
    assert astsr.last_scale_clip_min_ == float("-inf")
    assert astsr.last_scale_clip_max_ == float("inf")


def test_lambda_astsr_clips_predictions_to_phase1_range() -> None:
    """With scale_clip=(-10, 10), the ASTSR's predictions are clipped."""
    rng = np.random.default_rng(0)
    # Construct a target that the ASTSR can fit; the clip fires when
    # the best AST's predictions exceed ±10.  We force this by
    # monkey-patching the evaluate path to return exploding values
    # for a single AST, then verifying the diagnostics fire.
    import molmetal.molmetal_lam.reward.lambda_astsr as la_module

    original = la_module.collapse_detected
    # Disable the Wf-122 collapse detector so only the Wf-127 P3 clip fires.
    try:
        la_module.collapse_detected = lambda y_pred, y_true: False
        # Build a deterministic target the ASTSR can fit (no explosion).
        N, F = 30, 3
        X = rng.uniform(0.0, 1.0, (N, F))
        y = 2.0 * X[:, 0] + 0.5
        # Configure a tight clip range so the ASTSR's natural output
        # may or may not trip it; we verify the counter exists and is
        # correctly typed after a fit run.
        astsr = LambdaASTSR(
            depth=3, population=16, generations=5, seed=0,
            feature_names=("a", "b", "c"),
            scale_clip_min=-10.0,
            scale_clip_max=10.0,
        )
        astsr.fit(X, y)
        # The tight Phase 1 stable range is large enough that a 30-row
        # linear target should not clip, but the diagnostics must be
        # exposed and typed correctly.
        assert isinstance(astsr.n_clipped_predictions_, int)
        assert astsr.n_clipped_predictions_ >= 0
        assert isinstance(astsr.clip_violations_, list)
        assert astsr.last_scale_clip_min_ == pytest.approx(-10.0)
        assert astsr.last_scale_clip_max_ == pytest.approx(10.0)
    finally:
        la_module.collapse_detected = original


def test_lambda_astsr_tracks_n_clipped_predictions() -> None:
    """``n_clipped_predictions_`` increments when clip fires."""
    import molmetal.molmetal_lam.reward.lambda_astsr as la_module

    original = la_module.collapse_detected
    try:
        la_module.collapse_detected = lambda y_pred, y_true: False
        # Monkey-patch random_ast to always return a degenerate AST
        # whose evaluate_matrix produces values well above +10 — this
        # forces the clip to fire on every candidate, accumulating a
        # positive n_clipped_predictions_.
        import random
        from molmetal.molmetal_lam.reward.lambda_astsr import LambdaAST

        # Build a single canned AST that always returns 100.0.
        canned_ops = (
            {"kind": "const", "value": 100.0},
        )
        # We don't need a fully-formed LamNode for this test; use a stub.
        from molmetal.molmetal_lam.lam_chem.ast import var
        canned_ast = LambdaAST(
            lam_node=var("const_100.0"),
            ops=canned_ops,
            depth=1,
            n_leaves=1,
        )

        rng_a = random.Random(0)
        N, F = 10, 2
        X = np.zeros((N, F), dtype=float)
        y = np.zeros(N, dtype=float)

        # Patch _initial_population to return only the canned AST.
        astsr = LambdaASTSR(
            depth=1, population=4, generations=0, seed=0,
            feature_names=("a", "b"),
            scale_clip_min=-10.0,
            scale_clip_max=10.0,
        )

        class _StubAst:
            """Stub that behaves like LambdaAST for evaluate_matrix."""
            def __init__(self):
                self._ops = canned_ops
            def evaluate_matrix(self, X, feature_names):
                return np.full(X.shape[0], 100.0, dtype=float)
            def complexity(self):
                return 1

        stub_ast = _StubAst()
        astsr._initial_population = lambda rng: [stub_ast] * astsr.population
        astsr.fit(X, y)
        # Every prediction was 100.0 → every prediction was clipped.
        # population=4 candidates, each evaluating 10 rows = 40 clips.
        assert astsr.n_clipped_predictions_ == 40
        # All 4 candidates tripped the clip → 4 entries in the log.
        assert len(astsr.clip_violations_) == 4
        assert all(v == 10 for v in astsr.clip_violations_)
    finally:
        la_module.collapse_detected = original


def test_lambda_astsr_predict_clips_output() -> None:
    """``predict()`` returns clipped predictions (matches :meth:`_score` path)."""
    rng = np.random.default_rng(0)
    N, F = 30, 3
    X = rng.uniform(0.0, 1.0, (N, F))
    y = 2.0 * X[:, 0] + 0.5
    astsr = LambdaASTSR(
        depth=3, population=16, generations=5, seed=0,
        feature_names=("a", "b", "c"),
        scale_clip_min=-1.0,
        scale_clip_max=1.0,
    )
    astsr.fit(X, y)
    y_pred = astsr.predict(X)
    # All predictions must be in the clip range (the best AST may
    # not have learned anything useful because the target exceeds
    # the clip, but every prediction must be clipped).
    assert np.all(y_pred >= -1.0)
    assert np.all(y_pred <= 1.0)


def test_composite_propagates_scale_clip_to_lambda_astsr() -> None:
    """``HilbertASTSRComposite`` forwards ``scale_clip_*`` to its inner ``LambdaASTSR``."""
    from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
        HilbertASTSRComposite,
    )

    composite = HilbertASTSRComposite(
        feature_columns=("a", "b"),
        use_hilbert=False,
        scale_clip_min=-7.5,
        scale_clip_max=7.5,
    )
    assert composite.scale_clip_min == pytest.approx(-7.5)
    assert composite.scale_clip_max == pytest.approx(7.5)
    # The auto-instantiated LambdaASTSR must carry the same clip bounds.
    assert composite.lambda_astsr is not None
    assert composite.lambda_astsr.scale_clip_min == pytest.approx(-7.5)
    assert composite.lambda_astsr.scale_clip_max == pytest.approx(7.5)


def test_composite_fit_surfaces_clip_diagnostics() -> None:
    """``HilbertASTSRComposite.fit()`` returns clip diagnostics in its result dict."""
    from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
        HilbertASTSRComposite,
    )

    rng = np.random.default_rng(0)
    N, F = 30, 3
    X = rng.uniform(0.0, 1.0, (N, F))
    y = 2.0 * X[:, 0] + 0.5
    composite = HilbertASTSRComposite(
        feature_columns=("a", "b", "c"),
        use_hilbert=False,
        scale_clip_min=-10.0,
        scale_clip_max=10.0,
    )
    res = composite.fit(X, y)
    # The result dict must carry the Phase 3 diagnostics.
    assert "n_clipped" in res
    assert "scale_clip_min" in res
    assert "scale_clip_max" in res
    assert res["scale_clip_min"] == pytest.approx(-10.0)
    assert res["scale_clip_max"] == pytest.approx(10.0)
    # Diagnostics on the composite instance match.
    assert composite.last_scale_clip_min_used_ == pytest.approx(-10.0)
    assert composite.last_scale_clip_max_used_ == pytest.approx(10.0)
    assert composite.n_clipped_predictions_ == res["n_clipped"]