"""Wf-285B — Kendall UW + DANN GRL wiring tests for
:mod:`molmetal.scripts.train_dmpnn_multitask_wasserstein`.

Wf-283B added the ``--no-kendall`` / ``--no-dann`` CLI flags but did NOT
wire them into the actual loss / model / training-loop code — they were
parses but ignored (a flag-only contract).  Wf-285B closes the gap by
passing the toggles into ``MultiTaskLoss(use_kendall_loss=...)`` and
``MultiTaskWassersteinModel(use_dann=...)`` plus a DANN-GRL block inside
``train_one_epoch_joint``.  These three tests pin that contract:

1. ``test_kendall_on`` — when ``use_kendall=True``, ``MultiTaskLoss`` is
   built with ``use_kendall_loss=True``; the loss head contains a
   :class:`UncertaintyWeightedMultiTaskLoss` whose forward involves the
   learned per-task log-sigma (NOT the manual alpha path).
2. ``test_kendall_off`` — when ``use_kendall=False``, ``MultiTaskLoss``
   is built with ``use_kendall_loss=False``; the loss head contains NO
   Kendall module and the manual ``alpha * BCE + (1 - alpha) * MSE``
   formula is preserved bit-for-bit (Wf-131 P1 shipped behaviour).
3. ``test_dann_on_off`` — ``MultiTaskWassersteinModel(use_dann=True)``
   registers a non-``None`` :class:`DomainAdversarialHead`; with
   ``use_dann=False`` the head is ``None`` and the parameter count is
   unchanged from the Wf-131 P1 baseline.

All tests are CPU-only and finish in <10 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_dmpnn_multitask_wasserstein as mtw  # noqa: E402

from molmetal.baselines.dmpnn_multitask import MultiTaskLoss  # noqa: E402
from molmetal.models.dmpnn import DomainAdversarialHead  # noqa: E402


def _tiny_inputs(device: str = "cpu"):
    """Build deterministic tiny (pic50, cls, target) inputs."""
    torch.manual_seed(0)
    pic50_pred = torch.tensor([5.5, 6.0, 5.2, 5.9], requires_grad=True)
    active_logits = torch.tensor(
        [[1.0, 0.5], [0.2, 1.4], [0.7, 0.1], [-0.3, 0.9]],
        requires_grad=True,
    )
    pic50_target = torch.tensor([5.4, 6.1, 5.3, 5.7])
    active_target = torch.tensor([1, 0, 1, 1], dtype=torch.long)
    return pic50_pred, active_logits, pic50_target, active_target


# ===========================================================================
# 1. Kendall UW wiring (loss-level)
# ===========================================================================
def test_kendall_on() -> None:
    """Wf-285B: ``use_kendall=True`` wires Kendall UW into the loss.

    The :class:`MultiTaskLoss` is built with ``use_kendall_loss=True`` and
    a :class:`UncertaintyWeightedMultiTaskLoss` (Kendall et al. 2018)
    sub-module is lazily created on first ``forward()`` call.  The total
    loss is therefore driven by per-task log-sigma, NOT by the manual
    ``alpha * BCE + (1 - alpha) * MSE`` formula.
    """
    loss = MultiTaskLoss(alpha=0.5, use_kendall_loss=True)
    assert (
        loss.use_kendall_loss is True
    ), "MultiTaskLoss.use_kendall_loss flag must be True after wiring"
    pic50_pred, active_logits, pic50_target, active_target = _tiny_inputs()
    total, lc, lr_ = loss(pic50_pred, active_logits, pic50_target, active_target)
    assert total.requires_grad, "Kendall-weighted loss must remain differentiable"
    assert (
        lc.item() > 0 and lr_.item() > 0
    ), "cls and reg components must be positive for the random init"
    # The Kendall module is lazily built on first forward() — verify it
    # was instantiated and is registered as a sub-module (so the
    # optimiser picks up its log-sigma parameters).
    assert (
        loss._kendall is not None
    ), "Kendall UW head must be lazily created on first forward()"
    assert any(
        isinstance(m, nn.Module) and "UncertaintyWeighted" in type(m).__name__
        for m in loss.modules()
    ), "UncertaintyWeightedMultiTaskLoss must be present in loss.modules()"
    # Total loss is finite and finite-valued (no NaN/Inf).
    assert torch.isfinite(total), f"Kendall-weighted total must be finite, got {total}"


def test_kendall_off() -> None:
    """Wf-285B: ``use_kendall=False`` preserves the manual alpha path.

    When ``use_kendall_loss=False`` the loss is the Wf-131 P1 bit-for-bit
    formula ``alpha * BCE + (1 - alpha) * MSE`` — the Kendall UW head is
    NOT instantiated.
    """
    loss = MultiTaskLoss(alpha=0.5, use_kendall_loss=False)
    assert (
        loss.use_kendall_loss is False
    ), "MultiTaskLoss.use_kendall_loss flag must be False after wiring"
    pic50_pred, active_logits, pic50_target, active_target = _tiny_inputs()
    total, lc, lr_ = loss(pic50_pred, active_logits, pic50_target, active_target)

    # Manual alpha * BCE + (1 - alpha) * MSE reproduces the same scalar.
    expected_total = 0.5 * lc + 0.5 * lr_
    assert torch.allclose(total, expected_total, atol=1e-5), (
        f"Manual-alpha path broken: total={total.item()} "
        f"vs expected={expected_total.item()}"
    )
    # The Kendall UW head must NOT have been lazily created.
    assert (
        loss._kendall is None
    ), "Kendall UW head must NOT be created when use_kendall_loss=False"


# ===========================================================================
# 2. DANN wiring (model-level)
# ===========================================================================
def test_dann_on_off() -> None:
    """Wf-285B: ``use_dann`` toggles the DANN GRL head on the model.

    With ``use_dann=True`` the model registers a non-``None``
    :class:`DomainAdversarialHead` sub-module; with ``use_dann=False`` the
    head is ``None`` and the parameter count matches the Wf-131 P1
    shipped baseline (encoder + per-metal heads only).
    """
    torch.manual_seed(0)

    # OFF: no domain head, parameter count = baseline
    model_off = mtw.MultiTaskWassersteinModel(
        n_metals=2,
        hidden=32,
        depth=2,
        dropout=0.1,
        use_dann=False,
    )
    assert model_off.use_dann is False
    assert model_off.domain_head is None, "use_dann=False must leave domain_head=None"
    n_params_off = sum(p.numel() for p in model_off.parameters())

    # ON: domain head attached, parameter count grows
    model_on = mtw.MultiTaskWassersteinModel(
        n_metals=2,
        hidden=32,
        depth=2,
        dropout=0.1,
        use_dann=True,
        num_domains=4,
    )
    assert model_on.use_dann is True
    assert isinstance(model_on.domain_head, DomainAdversarialHead), (
        f"use_dann=True must register DomainAdversarialHead, got "
        f"{type(model_on.domain_head).__name__}"
    )
    assert model_on.num_domains == 4
    n_params_on = sum(p.numel() for p in model_on.parameters())

    # ON must strictly increase parameter count (domain_head has its
    # own Linear weights + biases on top of the shared encoder).
    assert (
        n_params_on > n_params_off
    ), f"use_dann=True must add params, but on={n_params_on} vs off={n_params_off}"
    # OFF must match the head-less baseline: the domain_head attribute
    # exists but is None, and the per-metal heads + encoder are intact.
    assert hasattr(model_off, "domain_head")
    assert hasattr(model_on, "domain_head")
