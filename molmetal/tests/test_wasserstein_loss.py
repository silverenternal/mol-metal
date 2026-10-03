"""Tests for :mod:`molmetal.molmetal_lam.reward.wasserstein_loss`.

Wf-123 Phase 1 — M1 Wasserstein loss module (TODO-126 §3 M1)
============================================================

This module is the **shared Wasserstein regularization module** used by
6 strengthen TODOs (119, 120, 121, 122, 123, 125).  These tests cover
the 6 loss / helper functions and the supporting configuration dataclass:

1. :class:`WassersteinConfig` — dataclass validation (n_mc, seed, decay_window, decay_tol).
2. :func:`wasserstein_loss` — generic W₂; finite scalar; zero on equal samples;
   symmetry; shape-mismatch rejection.
3. :func:`wasserstein_transfer_loss` — pairwise W₂ over a per-metal dict;
   rejects < 2 metals / empty batches / dim mismatch.
4. :func:`wasserstein_pic50_loss` — 1-D W₂ between predicted and reference pIC50;
   accepts (N,) and (N, 1).
5. :func:`wasserstein_pose_loss` — pose W₂; rejects non-3 last dim;
   accepts (N_atoms, 3) and (K, N_atoms, 3).
6. :func:`wasserstein_pocket_loss` — per-pocket W₂; pocket_id is diagnostic-only.
7. :func:`wasserstein_decay_check` — passes on decreasing history; fails on
   growing history; rejects invalid window / tol; handles short / non-finite inputs.
8. :func:`build_per_loss_verification` — round-trips; all finite; consistent keys.

All tests are **CPU-only** and use small synthetic tensors (batch ≤ 64,
dim ≤ 16) so the full suite runs in < 5 seconds.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.reward.wasserstein_loss import (
    WassersteinConfig,
    build_per_loss_verification,
    wasserstein_decay_check,
    wasserstein_loss,
    wasserstein_pic50_loss,
    wasserstein_pocket_loss,
    wasserstein_pose_loss,
    wasserstein_transfer_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 1. WassersteinConfig
# ---------------------------------------------------------------------------


class TestWassersteinConfig:
    def test_defaults_match_spec(self):
        """Spec defaults: n_mc=1000, seed=0, decay_window=10, decay_tol=1e-3."""
        cfg = WassersteinConfig()
        assert cfg.n_mc == 1000
        assert cfg.seed == 0
        assert cfg.decay_window == 10
        assert cfg.decay_tol == 1.0e-3

    def test_n_mc_below_one_rejected(self):
        with pytest.raises(ValueError, match="n_mc"):
            WassersteinConfig(n_mc=0)

    def test_decay_window_below_two_rejected(self):
        with pytest.raises(ValueError, match="decay_window"):
            WassersteinConfig(decay_window=1)

    def test_decay_tol_negative_rejected(self):
        with pytest.raises(ValueError, match="decay_tol"):
            WassersteinConfig(decay_tol=-1.0)

    def test_cast_to_native_python_types(self):
        """Post-init casts n_mc / seed / decay_window to int and decay_tol to float."""
        cfg = WassersteinConfig(n_mc=64.0, seed=1.0, decay_window=5.0, decay_tol=0.01)
        assert isinstance(cfg.n_mc, int)
        assert isinstance(cfg.seed, int)
        assert isinstance(cfg.decay_window, int)
        assert isinstance(cfg.decay_tol, float)


# ---------------------------------------------------------------------------
# 2. wasserstein_loss (generic W₂)
# ---------------------------------------------------------------------------


class TestWassersteinLoss:
    def test_returns_finite_scalar_on_synthetic(self):
        _set_seed(0)
        pred = torch.randn(16, 4)
        ref = torch.randn(16, 4) + 1.0
        loss = wasserstein_loss(pred, ref, n_mc=64)
        assert loss.dim() == 0
        assert bool(torch.isfinite(loss))

    def test_zero_when_distributions_match(self):
        """Identical sample clouds yield W₂ ≈ 0."""
        _set_seed(0)
        x = torch.randn(32, 4)
        loss = wasserstein_loss(x, x, n_mc=64)
        assert float(loss) < 1.0e-6

    def test_increases_with_shift(self):
        """W₂ grows as we shift one distribution away from the other."""
        _set_seed(0)
        base = torch.randn(64, 4)
        a = wasserstein_loss(base, base + 1.0, n_mc=64)
        b = wasserstein_loss(base, base + 5.0, n_mc=64)
        assert float(a) < float(b)

    def test_symmetric(self):
        """W₂(X, Y) ≈ W₂(Y, X) (sliced-W₂ estimator noise)."""
        _set_seed(0)
        x = torch.randn(32, 4)
        y = torch.randn(32, 4) + 0.5
        l1 = wasserstein_loss(x, y, n_mc=128)
        l2 = wasserstein_loss(y, x, n_mc=128)
        assert abs(float(l1) - float(l2)) < 0.5 * (
            abs(float(l1)) + abs(float(l2)) + 1e-6
        )

    def test_shape_mismatch_rejected(self):
        x = torch.randn(8, 4)
        y = torch.randn(8, 6)
        with pytest.raises(ValueError, match="last dim"):
            wasserstein_loss(x, y, n_mc=16)

    def test_invalid_n_mc_rejected(self):
        x = torch.randn(8, 4)
        with pytest.raises(ValueError, match="n_mc"):
            wasserstein_loss(x, x, n_mc=0)

    def test_empty_batch_returns_zero(self):
        """Empty inputs → zero (vacuous) rather than NaN, so trainer
        logging stays stable across empty data shards."""
        x = torch.zeros(0, 4)
        y = torch.randn(8, 4)
        loss = wasserstein_loss(x, y, n_mc=16)
        assert float(loss) == 0.0

    def test_grad_flow_finite(self):
        """Gradient w.r.t. ``pred_dist`` is finite and non-NaN."""
        _set_seed(0)
        pred = torch.randn(16, 4, requires_grad=True)
        ref = torch.randn(16, 4)
        loss = wasserstein_loss(pred, ref, n_mc=64)
        loss.backward()
        assert pred.grad is not None
        assert bool(torch.isfinite(pred.grad).all())


# ---------------------------------------------------------------------------
# 3. wasserstein_transfer_loss
# ---------------------------------------------------------------------------


class TestWassersteinTransferLoss:
    def test_finite_scalar_on_synthetic(self):
        _set_seed(0)
        per_metal = {
            "Ru": torch.randn(16, 8),
            "Ir": torch.randn(16, 8) + 0.3,
            "Os": torch.randn(16, 8) + 0.1,
            "Rh": torch.randn(16, 8) + 0.5,
            "Re": torch.randn(16, 8) + 0.2,
        }
        loss = wasserstein_transfer_loss(per_metal, n_mc=64)
        assert loss.dim() == 0
        assert bool(torch.isfinite(loss))

    def test_rejects_single_metal(self):
        per_metal = {"Ru": torch.randn(8, 4)}
        with pytest.raises(ValueError, match="at least 2 metals"):
            wasserstein_transfer_loss(per_metal, n_mc=16)

    def test_rejects_dim_mismatch(self):
        per_metal = {"Ru": torch.randn(8, 4), "Ir": torch.randn(8, 6)}
        with pytest.raises(ValueError, match="embedding dim"):
            wasserstein_transfer_loss(per_metal, n_mc=16)

    def test_rejects_non_tensor(self):
        per_metal = {"Ru": "not a tensor", "Ir": torch.randn(8, 4)}
        with pytest.raises(TypeError, match="Tensor"):
            wasserstein_transfer_loss(per_metal, n_mc=16)

    def test_rejects_non_dict(self):
        with pytest.raises(TypeError, match="dict"):
            wasserstein_transfer_loss([("Ru", torch.randn(8, 4))], n_mc=16)

    def test_filtering_empty_batches(self):
        """Empty batches are dropped — but a single non-empty batch
        plus an empty one still leaves only 1 valid key → rejection."""
        per_metal = {
            "Ru": torch.zeros(0, 4),
            "Ir": torch.randn(8, 4),
        }
        with pytest.raises(ValueError, match="non-empty"):
            wasserstein_transfer_loss(per_metal, n_mc=16)

    def test_accepts_two_metals_with_empty_third(self):
        """3 metals, 1 empty — pairwise W₂ is computed over the 2 valid keys."""
        _set_seed(0)
        per_metal = {
            "Ru": torch.zeros(0, 4),
            "Ir": torch.randn(8, 4),
            "Os": torch.randn(8, 4) + 0.5,
        }
        loss = wasserstein_transfer_loss(per_metal, n_mc=32)
        assert loss.dim() == 0
        assert bool(torch.isfinite(loss))


# ---------------------------------------------------------------------------
# 4. wasserstein_pic50_loss
# ---------------------------------------------------------------------------


class TestWassersteinPic50Loss:
    def test_finite_scalar_1d(self):
        _set_seed(0)
        pred = torch.randn(32) * 0.5 + 5.0
        ref = torch.randn(64) * 0.5 + 5.0
        loss = wasserstein_pic50_loss(pred, ref)
        assert loss.dim() == 0
        assert bool(torch.isfinite(loss))

    def test_accepts_column_vector(self):
        """Accepts (N, 1) shape alongside (N,)."""
        _set_seed(0)
        pred = (torch.randn(32) * 0.5 + 5.0).unsqueeze(-1)
        ref = torch.randn(64) * 0.5 + 5.0
        loss = wasserstein_pic50_loss(pred, ref)
        assert loss.dim() == 0
        assert bool(torch.isfinite(loss))

    def test_zero_when_distributions_match(self):
        """Identical pIC50 samples yield zero W₂."""
        _set_seed(0)
        x = torch.randn(32) * 0.5 + 5.0
        loss = wasserstein_pic50_loss(x, x)
        assert float(loss) < 1.0e-6

    def test_increases_with_shift(self):
        """W₂ grows when the predicted mean drifts from the reference mean."""
        _set_seed(0)
        ref = torch.randn(128) * 0.5 + 5.0
        a = wasserstein_pic50_loss(ref + 0.1, ref)
        b = wasserstein_pic50_loss(ref + 5.0, ref)
        assert float(a) < float(b)

    def test_gradient_flows_through_predicted_pic50(self):
        """Gradient w.r.t. ``pred_pic50`` is finite and non-zero.

        This pins the TODO-179 invariant: ``wasserstein_pic50_loss`` must
        be differentiable w.r.t. the predicted pIC50 tensor (NOT the
        reference).  The trainer wires
        ``L = MSE(pred, tgt) + w_w * W₂(pred, ref)`` and minimises the
        joint objective; if ``pred`` is detached the gradient from the W₂
        term is zeroed out and the trainer cannot align its predictions
        to the reference distribution (the Wf-123 P3 signal-floor R²).
        """
        _set_seed(0)
        pred_leaf = (torch.randn(16) * 0.5 + 5.0).detach().clone().requires_grad_(True)
        ref = torch.randn(64) * 0.5 + 5.0
        loss = wasserstein_pic50_loss(pred_leaf, ref)
        assert loss.requires_grad is True
        assert loss.grad_fn is not None
        loss.backward()
        # Gradient w.r.t. pred_leaf is finite, non-zero, and the right shape.
        assert pred_leaf.grad is not None
        assert bool(torch.isfinite(pred_leaf.grad).all())
        assert float(pred_leaf.grad.norm()) > 0.0
        assert pred_leaf.grad.shape == pred_leaf.shape

    def test_detaching_pred_zeros_gradient_signal(self):
        """Pin the BUG that produced TODO-179's R² ≤ 0.

        ``pred.detach()`` produces a tensor with no grad_fn, so the W₂
        penalty cannot be back-propagated.  Calling ``backward()`` on the
        W₂ term alone raises ``RuntimeError``; adding the W₂ term to a
        composite loss means the W₂ contribution is invisible to the
        optimizer.  This is the exact mechanism that left PlatinAI 4-cell
        R² at the signal floor in Wf-123 P3 / Wf-131 P1.
        """
        _set_seed(0)
        pred_det = torch.randn(16) * 0.5 + 5.0
        ref = torch.randn(64) * 0.5 + 5.0
        # Compute W₂ as if the trainer were to detach the prediction side
        # (the buggy code path in train_dmpnn_multitask_wasserstein.py).
        w_pen = wasserstein_pic50_loss(pred_det.detach(), ref)
        assert w_pen.requires_grad is False, (
            "W₂ on a detached prediction MUST have requires_grad=False; if "
            "this assertion fails, the detach is being silently ignored and "
            "the trainer pattern bug may not exist."
        )
        assert w_pen.grad_fn is None
        # The trainer pattern `total = loss + w_w * w_pen; total.backward()`
        # produces no gradient signal from the W₂ term — backward() on the
        # W₂ term alone raises a hard RuntimeError.  This is the diagnostic
        # that triggered the Wf-131 P1 signal-floor R² ≤ 0 outcome.
        with pytest.raises(RuntimeError, match="does not require grad"):
            w_pen.backward()

    def test_grad_descent_lowers_w2(self):
        """End-to-end gradient check: a one-step gradient descent on the
        W₂ penalty actually lowers the loss (i.e., the gradient has the
        correct SIGN).  If W₂ were applied with a flipped sign the trainer
        would *diverge*, not converge — this test catches that.
        """
        _set_seed(0)
        pred_leaf = torch.tensor([6.0, 7.0, 8.0, 5.5], requires_grad=True)
        ref = torch.tensor([5.0, 5.5, 5.0, 5.5])
        w_before = float(wasserstein_pic50_loss(pred_leaf, ref).item())
        loss = wasserstein_pic50_loss(pred_leaf, ref)
        loss.backward()
        assert pred_leaf.grad is not None
        # Gradient descent step (lr=1.0 — large but OK because we're on a
        # bounded scale).
        with torch.no_grad():
            pred_leaf -= 1.0 * pred_leaf.grad
        w_after = float(wasserstein_pic50_loss(pred_leaf, ref).item())
        # After a single GD step the W₂ distance must decrease.
        assert w_after < w_before, (
            f"W₂ did not decrease after GD step: w_before={w_before:.4f}, "
            f"w_after={w_after:.4f}.  The W₂ gradient has the WRONG SIGN."
        )

    def test_sign_and_magnitude_against_known_shift(self):
        """Validate sign + magnitude against an analytic closed-form.

        For a constant shift ``c`` between two equal-mass 1-D samples of
        size ``n`` (no overlap of sorted order statistics), the 1-D W₂
        reduces to ``|c|`` — i.e. the L1 distance between corresponding
        order statistics is ``|c|`` (each x_(i) = y_(i) + c), so the
        mean absolute difference is exactly ``|c|``.

        Note: ``wasserstein_pic50_loss`` returns the W₁ (1-Wasserstein)
        closed form for 1-D inputs — the mean absolute difference of
        sorted order statistics.  We pin that contract here so the
        PlatinAI 4-cell alignment uses the same scalar magnitude.
        """
        n = 32
        ref = torch.arange(n, dtype=torch.float32)
        # Shift pred by +5 → per-order-statistic diff is exactly 5.
        pred = ref + 5.0
        loss = wasserstein_pic50_loss(pred, ref)
        assert math.isclose(
            float(loss.item()), 5.0, abs_tol=1e-5
        ), f"expected |c|=5.0 for shift c=+5, got {float(loss.item()):.4f}"
        # Sign check: shift in the opposite direction gives the same
        # magnitude (W₂ / W₁ are both |c| for a pure shift).
        pred2 = ref - 3.0
        loss2 = wasserstein_pic50_loss(pred2, ref)
        assert math.isclose(float(loss2.item()), 3.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# 5. wasserstein_pose_loss
# ---------------------------------------------------------------------------


class TestWassersteinPoseLoss:
    def test_finite_scalar_2d(self):
        _set_seed(0)
        pred = torch.randn(30, 3)
        ref = torch.randn(30, 3) + 0.5
        loss = wasserstein_pose_loss(pred, ref, n_mc=64)
        assert loss.dim() == 0
        assert bool(torch.isfinite(loss))

    def test_accepts_3d_conformer_tensor(self):
        """Multi-conformer (K, N_atoms, 3) → reshape to (K*N_atoms, 3)."""
        _set_seed(0)
        pred = torch.randn(4, 30, 3)
        ref = torch.randn(4, 30, 3) + 0.3
        loss = wasserstein_pose_loss(pred, ref, n_mc=32)
        assert loss.dim() == 0
        assert bool(torch.isfinite(loss))

    def test_rejects_non_3_last_dim(self):
        pred = torch.randn(10, 4)
        ref = torch.randn(10, 4)
        with pytest.raises(ValueError, match="last dim must be 3"):
            wasserstein_pose_loss(pred, ref, n_mc=16)


# ---------------------------------------------------------------------------
# 6. wasserstein_pocket_loss
# ---------------------------------------------------------------------------


class TestWassersteinPocketLoss:
    def test_finite_scalar_with_pocket_id(self):
        _set_seed(0)
        cand = torch.randn(32, 8)
        ref = torch.randn(32, 8) + 0.7
        loss = wasserstein_pocket_loss(cand, ref, pocket_id="1h36", n_mc=64)
        assert loss.dim() == 0
        assert bool(torch.isfinite(loss))

    def test_pocket_id_optional(self):
        _set_seed(0)
        cand = torch.randn(16, 4)
        ref = torch.randn(16, 4)
        loss = wasserstein_pocket_loss(cand, ref, n_mc=32)
        assert loss.dim() == 0
        assert bool(torch.isfinite(loss))

    def test_rejects_non_string_pocket_id(self):
        cand = torch.randn(8, 4)
        ref = torch.randn(8, 4)
        with pytest.raises(TypeError, match="pocket_id"):
            wasserstein_pocket_loss(cand, ref, pocket_id=12345, n_mc=16)


# ---------------------------------------------------------------------------
# 7. wasserstein_decay_check
# ---------------------------------------------------------------------------


class TestWassersteinDecayCheck:
    def test_passes_on_strictly_decreasing_history(self):
        hist = [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5]
        assert wasserstein_decay_check(hist) is True

    def test_passes_on_noisy_decreasing_history(self):
        """Small wobble within ``tol`` is still considered non-increasing.

        Uses a custom ``tol`` large enough to absorb a 1% per-step wobble.
        """
        hist = [10.0, 9.5, 9.6, 9.4, 9.45, 9.5, 9.4, 9.45, 9.5, 9.4]
        assert wasserstein_decay_check(hist, tol=0.05) is True

    def test_fails_on_growing_history(self):
        hist = [10.0, 9.5, 9.0, 9.2, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
        assert wasserstein_decay_check(hist) is False

    def test_rejects_window_below_two(self):
        with pytest.raises(ValueError, match="window"):
            wasserstein_decay_check([1.0, 0.5], window=1)

    def test_rejects_negative_tol(self):
        with pytest.raises(ValueError, match="tol"):
            wasserstein_decay_check([1.0, 0.5], tol=-1.0)

    def test_short_history_returns_true(self):
        """Empty / single-element history has no decision to make → True."""
        assert wasserstein_decay_check([]) is True
        assert wasserstein_decay_check([1.0]) is True

    def test_non_finite_history_fails(self):
        """NaN / inf in the trailing window is an automatic fail."""
        hist = [1.0, 0.9, 0.8, float("nan"), 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        assert wasserstein_decay_check(hist) is False

    def test_uses_trailing_window_when_history_longer(self):
        """Window applies to the trailing ``window`` elements, not the head."""
        hist = [100.0, 90.0, 80.0, 70.0, 60.0] + [10.0, 9.5, 9.0, 8.5, 8.0]
        assert wasserstein_decay_check(hist, window=5) is True

    def test_growth_in_trailing_window_detected(self):
        """Head can grow; tail must not."""
        hist = [1.0, 2.0, 3.0, 4.0, 5.0] + [10.0, 10.5, 11.0, 11.5, 12.0]
        assert wasserstein_decay_check(hist, window=5) is False


# ---------------------------------------------------------------------------
# 8. build_per_loss_verification
# ---------------------------------------------------------------------------


class TestBuildPerLossVerification:
    def test_keys_match_public_api(self):
        """The verification record covers all 6 public loss / check functions."""
        rec = build_per_loss_verification()
        expected = {
            "wasserstein_loss",
            "wasserstein_transfer_loss",
            "wasserstein_pic50_loss",
            "wasserstein_pose_loss",
            "wasserstein_pocket_loss",
            "wasserstein_decay_check",
        }
        assert set(rec.keys()) == expected

    def test_each_value_finite_and_engine_tagged(self):
        """Every entry carries ``finite=True``, ``engine="wasserstein_reg"`` and
        ``shape_ok=True`` — the honest-reporting invariant from TODO-126 §7."""
        rec = build_per_loss_verification(n_metals=3, batch_size=16, embedding_dim=4)
        for name, entry in rec.items():
            assert entry["engine"] == "wasserstein_reg", name
            assert entry["finite"] is True, f"{name} not finite: {entry}"
            assert entry["shape_ok"] is True, name

    def test_decay_check_true_on_decreasing_history(self):
        rec = build_per_loss_verification()
        assert bool(rec["wasserstein_decay_check"]["value"]) is True

    def test_accepts_custom_config(self):
        cfg = WassersteinConfig(n_mc=64, seed=42, decay_window=5, decay_tol=1e-4)
        rec = build_per_loss_verification(config=cfg)
        for entry in rec.values():
            assert entry["finite"] is True
