# SPDX-License-Identifier: Apache-2.0
"""ShortcutDistillLoss — the class contract, kept after its trainer wire-in was retired.

History, because it is easy to re-create this defect:

* Wf-A102 found the trainer always passed `x_prime=None`, so the class's
  self-consistency branch short-circuited to zero at every weight, and wired a
  second view.
* Wf-A103 calibrated the perturbation sigma (the term is exactly quadratic in
  it, sigma^2-normalised to E[||grad v||^2] = 1.52e-04).
* Wf-A105 then MEASURED that the tensor never entered the loss at all: Wf-A34-1
  had removed this duplicate wrapper from the sum, and
  `--shortcut-distill-term-weight` 1e-9 and 500 gave a bit-identical epoch.
* Wf-A106 deleted the wrapper and the flag.  The class itself stays in
  `molmetal.loss` because it is tested here and is the reference implementation
  for the KL / self-consistency branches.

These tests cover the CLASS.  They deliberately do not assert anything about
the trainer, because the trainer no longer uses it.
"""
from __future__ import annotations

import pytest
import torch


def test_none_second_view_is_exactly_zero():
    """The branch needs a second view; without one it is an exact zero."""
    from molmetal.loss.distillation_losses import ShortcutDistillLoss

    student = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
    student = student.requires_grad_(True)
    dead = ShortcutDistillLoss(weight=0.05, enabled=True).compute(
        student_logits=student, x_prime=None
    )
    assert float(dead) == 0.0


def test_a_real_second_view_is_live_and_differentiable():
    from molmetal.loss.distillation_losses import ShortcutDistillLoss

    student = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
    student = student.requires_grad_(True)
    second = student.detach() + 0.5
    live = ShortcutDistillLoss(weight=0.05, enabled=True).compute(
        student_logits=student, x_prime=second
    )
    assert float(live.detach()) > 0.0
    assert live.requires_grad


def test_identical_views_are_zero():
    """The correct value for a self-consistency penalty, and the reason a
    0.01 A perturbation reads ~1e-08."""
    from molmetal.loss.distillation_losses import ShortcutDistillLoss

    student = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
    student = student.requires_grad_(True)
    same = ShortcutDistillLoss(weight=0.05, enabled=True).compute(
        student_logits=student, x_prime=student.clone()
    )
    assert float(same) == pytest.approx(0.0, abs=1e-12)


def test_a_disabled_weight_short_circuits_before_any_forward():
    from molmetal.loss.distillation_losses import ShortcutDistillLoss

    out = ShortcutDistillLoss(weight=0.0, enabled=False).compute(
        student_logits=torch.randn(2, 3), x_prime=torch.randn(2, 3)
    )
    assert float(out) == 0.0


def test_the_trainer_no_longer_uses_the_class():
    """Wf-A106 — if this comes back, it must come back WITH a loss-sum entry."""
    from pathlib import Path

    trainer = (
        Path(__file__).resolve().parents[2]
        / "molmetal"
        / "scripts"
        / "train_cfm_yuelbond_wasserstein.py"
    )
    src = trainer.read_text()
    assert "ShortcutDistillLoss(" not in src, (
        "the wrapper is back in the trainer. Before Wf-A34-1 it was a duplicate "
        "of the inline --shortcut-distill path and was not in the loss sum; if "
        "it is re-added, add it to the sum and to EXPECTED_LIVE in "
        "test_cfm_loss_gradient_liveness.py, or do not add it at all (§6.149)."
    )
