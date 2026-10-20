# SPDX-License-Identifier: Apache-2.0
"""Wf-A102 (2026-10-20) — ShortcutDistillLoss could never be non-zero.

Its self-consistency branch compares the model's output against a SECOND view
(``x_prime``).  The trainer passed ``x_prime=None`` unconditionally, so
``compute`` short-circuited to ``torch.zeros(())`` at every weight:

    shortcut_distill_class_term   value 0.0   requires_grad False   (weight 0.05)
    shortcut_distill_loss (old)   value 1.897 requires_grad True

The weight read ON while the term trained nothing.  Wf-A102 builds the second
view (a second forward on a perturbed input) whenever the weight is > 0, and
moves the default 0.05 -> 0.0 so that no existing run changes.

The term is LIVE now, but MEASURED at 1.24e-08 at weight 0.05: a 0.01 A input
perturbation barely moves the velocity field, so the penalty is three orders of
magnitude below observability.  That is recorded, not hidden — see
HONEST_NEGATIVE_CATALOG.md §6.146.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _parse():
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    argv = sys.argv
    sys.argv = ["trainer", "--config", os.devnull]
    try:
        return T.parse_args()
    finally:
        sys.argv = argv


def test_default_is_off_because_the_old_default_was_inert():
    """0.05 was the shipped value and produced exactly zero.

    Moving the default to 0.0 changes no run; it stops the weight claiming to
    be ON.  Pinned against the real argparse declaration.
    """
    assert float(_parse().shortcut_distill_term_weight) == 0.0


def test_the_second_view_is_built_only_when_the_weight_is_positive():
    """The trainer must not pay an extra forward at the shipped default."""
    src = TRAINER.read_text()
    assert "_sd_second_view" in src
    assert "x_prime=_sd_second_view" in src
    # the guard must test the weight, not just truthiness of a flag
    guard = src[src.index("_sd_second_view = None") : src.index("x_prime=_sd_second_view")]
    assert "shortcut_distill_term_weight" in guard
    assert "> 0.0" in guard


def test_the_class_is_live_when_given_a_second_view():
    """Behaviour of the class itself: None -> 0, a real second view -> non-zero."""
    from molmetal.loss.distillation_losses import ShortcutDistillLoss

    student = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
    # requires_grad, because in the trainer this tensor IS the model output —
    # without it the term cannot carry a gradient and the liveness guard would
    # be measuring the test's own construction.
    student = student.requires_grad_(True)
    second = student.detach() + 0.5

    dead = ShortcutDistillLoss(weight=0.05, enabled=True).compute(
        student_logits=student, x_prime=None
    )
    live = ShortcutDistillLoss(weight=0.05, enabled=True).compute(
        student_logits=student, x_prime=second
    )
    assert float(dead) == 0.0
    assert float(live) > 0.0
    assert live.requires_grad

    # identical views -> zero, which is the correct value for a self-consistency
    # penalty (and the reason a 0.01 A perturbation reads ~1e-08)
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
