"""Kendall uncertainty-weighted multi-task loss (drop-in re-export of mtl_uncertainty).

This module re-exports the verbatim ``UncertaintyWeightedMultiTaskLoss``
implementation from ``mtl_uncertainty_loss`` (MIT licence, originally by
``elna4os`` https://github.com/elna4os/mtl_uncertainty_loss).  The
algorithm is the homoscedastic-uncertainty multi-task loss from:

    Kendall, A., Gal, Y., & Cipolla, R. (2018). "Multi-Task Learning Using
    Uncertainty to Weigh Losses for Scene Geometry and Semantics". CVPR 2018.
    arXiv:1705.07115.

Mathematical formulation (per-task scalar weight is the noise precision
``1 / sigma^2`` with a learnable log-sigma):

    L_kendall(losses) = sum_k ( loss_k / (2 * sigma_k^2) + log(sigma_k) )

Where ``sigma_k = exp(log_sigma_k)`` and ``log_sigma_k`` is a learnable
per-task scalar parameter initialised to 0 (i.e. ``sigma_k = 1``).  Gradients
flow into both the upstream task losses and ``log_sigmas``, so the optimizer
jointly learns the task weights and the model parameters.

Why we keep a thin local shim instead of importing upstream
-----------------------------------------------------------
TODO-143 §2.1 directs us to "drop-in" the loss so Mol-Metal owns the code
rather than depending on the upstream repository.  The implementation is
~30 lines and dependency-free (only ``torch``), so the cost of duplication
is negligible and we are insulated from upstream breakage, licence
changes, or disappearance of the elna4os repo.

This file is regenerated from ``mtl_uncertainty/mtl_uncertainty_loss/loss.py``
and adds only one defensive improvement over the upstream source: a
positive-int validation guard on ``num_tasks`` so that
``num_tasks <= 0`` or a non-int value is rejected at construction time
instead of failing later inside ``torch.zeros(num_tasks)``.  Behaviour is
otherwise byte-identical to the upstream ``mtl_uncertainty_loss`` package
(same ``log_sigmas`` parameter, same forward formula, same
``get_sigmas()`` helper, same exported class name).

API contract (kept stable for backward compat)
---------------------------------------------
::

    from molmetal.molmetal_lam.sbdd_env.uncertainty_loss import (
        UncertaintyWeightedMultiTaskLoss,
    )

    loss_fn = UncertaintyWeightedMultiTaskLoss(num_tasks=5)
    total = loss_fn([loss_a, loss_b, loss_c, loss_d, loss_e])
    sigmas = loss_fn.get_sigmas()  # shape [num_tasks], detached

Usage in Mol-Metal
------------------
* ``V4MultiTaskLoss`` (``molmetal/models/v4_multitask_dmpnn.py``) uses this
  head when ``use_kendall_loss=True`` — for the 5-way metal set (Ru / Ir /
  Rh / Re / Os) + coordination-geometry auxiliary head (``num_tasks=6``),
  or expanded to ``num_tasks=45`` for the 9 cell-line cross-product.
* ``DMPNNMultitaskTrainer`` (``molmetal/baselines/dmpnn_multitask.py``)
  wires the same head into the baseline D-MPNN multi-task trainer.

Verification
------------
Verified ``mtl_uncertainty/mtl_uncertainty_loss/loss.py`` against this
file's prior verbatim port on 2026-09-27 — they were already byte-equivalent
except for the upstream's looser ``num_tasks`` validation (now hardened
here).  Regression test
``molmetal/tests/test_uncertainty_loss.py::test_api_matches_mtl_uncertainty``
exercises the contract.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """Kendall et al. 2018 (CVPR, arXiv:1705.07115) homoscedastic-uncertainty loss.

    Re-exported from ``mtl_uncertainty`` (MIT, originally elna4os) with a
    single defensive addition: ``num_tasks`` must be a positive ``int`` —
    non-int or non-positive values raise ``ValueError`` at construction.

    The module holds a single learnable ``log_sigmas`` parameter of shape
    ``(num_tasks,)``.  During the forward pass each per-task loss is
    scaled by ``1 / (2 * sigma_k^2)`` and offset by ``log(sigma_k)``;
    ``sigma_k = exp(log_sigma_k)`` so ``log_sigmas`` initialised to 0
    means all tasks start with ``sigma = 1`` (equal weighting).

    Args:
        num_tasks: number of tasks being aggregated (must match the
            length of the losses sequence passed to ``forward``).

    Shape:
        * Input:  ``Sequence[Tensor]`` of length ``num_tasks``; each
          tensor is a 0-D scalar loss.
        * Output: 0-D scalar tensor — sum across all tasks.
    """

    def __init__(self, num_tasks: int):
        super().__init__()
        if not isinstance(num_tasks, int) or num_tasks < 1:
            raise ValueError(
                f"num_tasks must be a positive int, got {num_tasks!r}"
            )
        self.num_tasks = int(num_tasks)
        # log(sigma) initialised to 0  <=>  sigma = 1, so the initial
        # weighting is uniform.  Registered as a Parameter so the
        # optimizer can update it like any other learnable weight.
        self.log_sigmas = nn.Parameter(torch.zeros(self.num_tasks))

    def forward(self, losses: Sequence[torch.Tensor]) -> torch.Tensor:
        """Aggregate per-task losses into a single Kendall-weighted loss.

        Args:
            losses: a sequence of length ``self.num_tasks`` whose entries
                are 0-D scalar tensors.  Passing a sequence of the wrong
                length raises ``ValueError``.

        Returns:
            A 0-D scalar tensor — the sum over all tasks of
            ``loss_k / (2 * sigma_k^2) + log(sigma_k)``.  Gradients flow
            back through both the task losses and ``log_sigmas``.
        """
        if len(losses) != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} losses, got {len(losses)}"
            )
        task_losses = torch.stack(losses)
        sigmas = self.log_sigmas.exp()
        return (task_losses / (2 * sigmas ** 2) + self.log_sigmas).sum()

    def get_sigmas(self) -> torch.Tensor:
        """Return the current per-task sigma estimates (detached).

        Returns ``exp(log_sigmas).detach()`` — a 1-D tensor of shape
        ``(num_tasks,)``.  These are the homoscedastic uncertainty
        estimates learned by the optimizer; larger sigma means the
        task is being down-weighted.
        """
        return self.log_sigmas.exp().detach()


__all__ = ["UncertaintyWeightedMultiTaskLoss"]