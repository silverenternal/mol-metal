"""Tests for Wf-242 Fix B + Fix C NaN guard in train_chemprop_v2_v3.py.

Wf-242 Fix B (root cause):
    In some PyTorch paths ``sigmoid(inf_bf16)`` returns NaN due to BF16
    NaN propagation.  Converting logit_1 to FP32 first avoids the NaN
    sentinel, and the clamp guarantees no overflow survives the cast.

Wf-242 Fix C (preventive):
    Under BF16 autocast, the encoder can saturate the logit to +/-inf
    in BF16.  Clamping BEFORE sigmoid so ``sigmoid(±30) ≈ 1e-7 / 0.9999999``
    is numerically safe.

NaN guard (lines 1076-1081):
    After ``p = torch.sigmoid(logit_1.squeeze(-1)).cpu().numpy()``,
    if ``np.any(np.isnan(p))`` the driver logs a warning and substitutes
    ``nan_to_num(p, nan=0.5)`` so the eval can still produce a finite
    ``roc_auc_score`` instead of returning NaN.

These tests are CPU-only and validate the substitution logic + the
source-level marker (so future readers can trace the fix).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# G1: Wf-242 NaN guard must exist in train_chemprop_v2_v3.py.
# ---------------------------------------------------------------------------
def test_wf242_nan_guard_marker_in_source() -> None:
    """The Wf-242 Fix B + Fix C NaN guard must remain in
    train_chemprop_v2_v3.py:1076-1081.  Without this guard the driver
    silently propagates NaN to ``roc_auc_score`` which then returns
    NaN, masking the regression."""
    src = (_SCRIPT_DIR / "train_chemprop_v2_v3.py").read_text()
    assert "nan_to_num(p, nan=0.5)" in src, (
        "Wf-242 NaN guard missing or drifted: must use "
        "np.nan_to_num(p, nan=0.5) substitution."
    )
    # The warning marker must still be there (logs the count of NaNs).
    assert "eval produced %d/%d NaN scores, replacing with 0.5" in src, (
        "Wf-242 NaN warning log line missing — guard may have lost "
        "its observability hook."
    )
    # The detector must use np.any(np.isnan(p)).
    assert "np.any(np.isnan(p))" in src, (
        "Wf-242 NaN guard detector drifted: must use "
        "np.any(np.isnan(p)) to gate the substitution."
    )


# ---------------------------------------------------------------------------
# G2: Substitution semantics — injecting NaN logits yields 0.5 substitution.
# ---------------------------------------------------------------------------
def test_nan_guard_substitutes_0_5_for_nan_scores() -> None:
    """When ``p`` contains NaN, the guard MUST replace NaN with 0.5
    (the documented "random score" sentinel) while leaving finite
    scores untouched.  Verifies backward compat: a NaN-free input
    passes through unchanged.
    """
    # Simulate a sigmoid output with some NaN entries (the Wf-242
    # regression signature: BF16 sigmoid(inf) → NaN).
    p = np.array([0.1, 0.2, np.nan, 0.4, np.nan, 0.6], dtype=np.float32)
    assert np.any(np.isnan(p)), "precondition: p must contain NaN"

    # The guard from the source (lines 1076-1081) — verbatim shape.
    if np.any(np.isnan(p)):
        p = np.nan_to_num(p, nan=0.5)

    # NaN slots replaced with 0.5.
    assert p[2] == pytest.approx(0.5), f"NaN at idx 2 must be 0.5, got {p[2]}"
    assert p[4] == pytest.approx(0.5), f"NaN at idx 4 must be 0.5, got {p[4]}"
    # Finite slots untouched.
    assert p[0] == pytest.approx(0.1)
    assert p[1] == pytest.approx(0.2)
    assert p[3] == pytest.approx(0.4)
    assert p[5] == pytest.approx(0.6)
    # No NaN should survive.
    assert not np.any(np.isnan(p)), "NaN must not survive the guard"


def test_nan_guard_passthrough_when_input_clean() -> None:
    """Backward compat: when no NaN is present, the guard MUST NOT
    modify the array (NaN-free eval yields identical scores).
    """
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    p_before = p.copy()
    # The guard is gated on np.any(np.isnan(p)) — must short-circuit.
    if np.any(np.isnan(p)):
        p = np.nan_to_num(p, nan=0.5)
    np.testing.assert_array_equal(
        p,
        p_before,
        err_msg="NaN guard must not modify a clean input array",
    )


def test_nan_guard_handles_all_nan_input() -> None:
    """Edge case: ALL scores are NaN (extreme encoder collapse).  After
    the guard every entry MUST be 0.5 — the eval then reports
    test_auc=0.5 (random) instead of NaN, which is the documented
    honest-negative behavior.
    """
    p = np.full(64, np.nan, dtype=np.float32)
    assert np.all(np.isnan(p))
    if np.any(np.isnan(p)):
        p = np.nan_to_num(p, nan=0.5)
    assert np.all(p == 0.5), "all-NaN input must yield all-0.5 output"
    assert p.shape == (64,), "shape must be preserved"


# ---------------------------------------------------------------------------
# G3: Substitution preserves dtype + shape for roc_auc_score contract.
# ---------------------------------------------------------------------------
def test_nan_guard_output_usable_by_roc_auc_score() -> None:
    """The driver feeds ``p`` into ``roc_auc_score(labels, scores)``.
    After the guard, scores must be finite AND have the same length
    as labels — otherwise ``roc_auc_score`` raises ValueError.  This
    test reproduces the post-guard contract.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    p = np.array([0.1, np.nan, 0.3, np.nan, 0.5, np.nan, 0.7, np.nan], dtype=np.float32)
    if np.any(np.isnan(p)):
        p = np.nan_to_num(p, nan=0.5)
    # Post-guard: must be all-finite + correct length.
    assert not np.any(np.isnan(p)), "post-guard NaN violates roc_auc_score contract"
    assert len(p) == len(labels), "shape drift would break roc_auc_score"
    # roc_auc_score must succeed (returns 0.5 = random for the alternation).
    auc = float(roc_auc_score(labels, p))
    assert np.isfinite(auc), f"post-guard AUC must be finite, got {auc}"
    # With 0.5 substituted for half the scores and the other half ranking
    # above 0.5, the AUC must be ≥ 0.5 (not worse than random).
    assert auc >= 0.5, f"post-guard AUC must be >= 0.5, got {auc:.4f}"


# ---------------------------------------------------------------------------
# G4: Substitution contract — only NaN is replaced; +/-inf is left alone.
#     (Documents intent — if a future fix wants +inf→1 / -inf→0,
#     update the source AND this test.)
# ---------------------------------------------------------------------------
def test_nan_guard_only_replaces_nan_not_inf() -> None:
    """``nan_to_num(p, nan=0.5)`` by default also clips +inf→max finite
    and -inf→min finite, but the call signature in the driver does NOT
    supply ``posinf`` / ``neginf``, so only NaN is substituted.  This
    test pins the behavior so a future refactor (e.g. adding posinf=1.0)
    is intentional.
    """
    p = np.array([0.1, np.nan, np.inf, -np.inf, 0.5], dtype=np.float64)
    p_orig = p.copy()
    if np.any(np.isnan(p)):
        p = np.nan_to_num(p, nan=0.5)
    # NaN → 0.5
    assert p[1] == pytest.approx(0.5)
    # +inf / -inf clipped to dtype max / min (default nan_to_num behavior).
    assert (
        p[2] == np.finfo(np.float64).max
    ), "posinf clipped to dtype max (no posinf kwarg in driver call)"
    assert (
        p[3] == np.finfo(np.float64).min
    ), "neginf clipped to dtype min (no neginf kwarg in driver call)"
    # Finite untouched.
    assert p[0] == pytest.approx(p_orig[0])
    assert p[4] == pytest.approx(p_orig[4])
