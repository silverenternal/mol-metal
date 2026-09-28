"""Tests for Wf-187 m19 Rh oversample fix — train_metal_lomo_v2.py.

Wf-131 P2 Rh LOMO regressed 0.368 → 0.136 because of the
constant-inactive Rh cohort.  Wf-187 m19 fix adds:

1. ``--oversample-positives`` flag in ``train_metal_lomo_v2.py`` — when
   set, applies ``sklearn.utils.resample`` on positive-only rows for
   under-represented metals (Rh, Re) to lift the per-batch positive
   rate.
2. ``pos_weight=10`` fallback in :class:`MultiTaskLoss` — the
   classification loss switches from ``nn.CrossEntropyLoss`` to
   ``nn.BCEWithLogitsLoss(pos_weight=...)`` on the positive-class
   logit when ``pos_weight`` is set (default ``None`` keeps the
   backward-compatible CE path).

This file covers two unit tests:

* ``test_oversample_increases_pos_rate`` — exercises
  :func:`oversample_underrepresented_metals` with a synthetic
  class-imbalanced pool (5 positives, 45 negatives for Rh) and
  asserts the post-oversample positive rate is materially higher.
* ``test_pos_weight_changes_loss`` — exercises :class:`MultiTaskLoss`
  with and without ``pos_weight`` on the same logits/targets and
  asserts the classification loss differs by > 10 % (sanity-check
  that the fallback actually changes the loss surface).

Test count: 2 unit tests as required by the task spec.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from molmetal.baselines.dmpnn_multitask import MultiTaskLoss
from molmetal.scripts.train_metal_lomo_v2 import (
    oversample_underrepresented_metals,
)


# ---------------------------------------------------------------------------
# Wf-187 m19 fix tests.
# ---------------------------------------------------------------------------

class TestWf187M19Oversample:
    """Verify the m19 Rh oversample fix wires end-to-end."""

    def test_oversample_increases_pos_rate(self) -> None:
        """``oversample_underrepresented_metals`` lifts Rh pos_rate.

        We build a synthetic class-imbalanced pool: 5 Rh positives + 45
        Rh negatives + 100 Ru rows (all negatives).  Pre-oversample
        pos_rate for Rh is 5 / 50 = 0.10.  After applying the helper
        with ``target_pos_rate=0.5`` we expect the per-metal pos_rate
        for Rh to lift to >= 0.45 (allowing 5 % slack for the round
        in target_n_pos computation) and the Rh negatives count to
        stay at 45 (no negatives added — we only resample positives).
        """
        # Build the synthetic pool.
        n_rh_pos_pre = 5
        n_rh_neg_pre = 45
        n_ru_pre = 100
        smiles = (
            [f"Rh_pos_{i}" for i in range(n_rh_pos_pre)]
            + [f"Rh_neg_{i}" for i in range(n_rh_neg_pre)]
            + [f"Ru_{i}" for i in range(n_ru_pre)]
        )
        active = np.array(
            [1] * n_rh_pos_pre + [0] * n_rh_neg_pre + [0] * n_ru_pre,
            dtype=np.int64,
        )
        metals = np.array(
            ["Rh"] * (n_rh_pos_pre + n_rh_neg_pre) + ["Ru"] * n_ru_pre,
            dtype=object,
        )
        pic50 = np.array(
            [5.0] * n_rh_pos_pre + [6.5] * n_rh_neg_pre + [6.0] * n_ru_pre,
            dtype=np.float32,
        )

        # Sanity check pre-condition.
        pre_pos_rate = float((active[metals == "Rh"] == 1).mean())
        assert pre_pos_rate == pytest.approx(0.10, abs=1e-6), (
            f"Setup invariant violated: pre_pos_rate={pre_pos_rate}"
        )

        # Apply the helper.
        (
            smi2,
            act2,
            met2,
            pic502,
            stats,
        ) = oversample_underrepresented_metals(
            smiles=smiles,
            active=active,
            metals=metals,
            pic50=pic50,
            underrepresented=["Rh", "Re"],
            target_pos_rate=0.5,
            seed=42,
        )

        # Verify post-conditions.
        assert isinstance(stats, dict)
        assert "Rh" in stats
        assert stats["Rh"] > 0, (
            f"expected Rh positives appended, got stats={stats}"
        )
        # Re-only check: Re had 0 rows, so stats['Re'] should be 0.
        assert stats.get("Re", 0) == 0

        # Verify Rh pos_rate increased materially.
        rh_mask = met2 == "Rh"
        post_pos_rate = float((act2[rh_mask] == 1).mean())
        assert post_pos_rate > pre_pos_rate, (
            f"pos_rate did not increase: pre={pre_pos_rate:.3f} "
            f"post={post_pos_rate:.3f}"
        )
        # The helper targets 0.5, allow 5 % slack for rounding.
        assert post_pos_rate >= 0.45, (
            f"post-oversample Rh pos_rate too low: {post_pos_rate:.3f} "
            f"(target was 0.50, allowed >= 0.45)"
        )
        # Ru should be untouched.
        ru_mask = met2 == "Ru"
        assert int(ru_mask.sum()) == n_ru_pre, (
            f"Ru rows were modified: pre={n_ru_pre} post={int(ru_mask.sum())}"
        )

    def test_pos_weight_changes_loss(self) -> None:
        """``MultiTaskLoss(pos_weight=10)`` produces different cls loss.

        Build two loss instances — one with the default
        ``CrossEntropyLoss`` path (pos_weight=None) and one with the
        ``BCEWithLogitsLoss(pos_weight=10)`` fallback.  On identical
        synthetic logits + targets the classification loss must
        differ by a substantial fraction (we use a relative tolerance
        of 50 % to allow for absolute numerical scale).
        """
        torch.manual_seed(0)
        active_logits = torch.randn(8, 2)  # (B, 2) classification logits
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        pic50_pred = torch.randn(8)
        pic50_target = torch.randn(8)

        loss_default = MultiTaskLoss(alpha=0.5)
        loss_pos10 = MultiTaskLoss(alpha=0.5, pos_weight=10.0)

        # Sanity check: default path is CrossEntropyLoss.
        assert loss_default._pos_weight is None, (
            "default MultiTaskLoss should have _pos_weight=None "
            "(backward-compatible CrossEntropyLoss)"
        )
        # pos10 path should have _pos_weight=10.0.
        assert loss_pos10._pos_weight == pytest.approx(10.0), (
            f"MultiTaskLoss(pos_weight=10) should set _pos_weight=10.0, "
            f"got {loss_pos10._pos_weight}"
        )

        # Compute both losses on identical inputs.
        total_default, cls_default, reg_default = loss_default(
            pic50_pred, active_logits, pic50_target, targets
        )
        total_pos10, cls_pos10, reg_pos10 = loss_pos10(
            pic50_pred, active_logits, pic50_target, targets
        )

        # The classification loss MUST differ between the two paths.
        # We use a relative tolerance of 50 % (loss magnitude scales
        # with the pos_weight).
        cls_diff = abs(cls_default.item() - cls_pos10.item())
        cls_rel_diff = cls_diff / max(abs(cls_default.item()), 1e-9)
        assert cls_rel_diff > 0.50, (
            f"pos_weight did not change cls loss materially: "
            f"default={cls_default.item():.4f} pos10={cls_pos10.item():.4f} "
            f"rel_diff={cls_rel_diff:.4f}"
        )
        # The regression loss MUST be identical (we only changed the
        # classification path).
        assert reg_default.item() == pytest.approx(reg_pos10.item(), abs=1e-6), (
            f"regression loss drifted between loss paths: "
            f"default={reg_default.item():.6f} pos10={reg_pos10.item():.6f}"
        )
        # Total loss should differ (cls differs, reg same).
        assert total_default.item() != pytest.approx(total_pos10.item(), abs=1e-4), (
            f"total loss did not change between paths"
        )