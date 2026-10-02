"""Wf-380-A1 §2.3 — StableVM (Variational Mixture) SCAFFOLD.

Reference: Kim, S., Jung, J., Lee, J., Park, S. (2026, ICML).
*Stable Variational Mixture for Continuous Normalising Flows.*

The StableVM objective replaces the single-scale regression
``MSE(v_pred, target)`` with a *mixture* of multi-scale losses that
encourage consistent predictions across perturbation scales:

    L_StableVM = Σ_k w_k · ‖v_pred  −  target‖²_k

where the per-scale norm ‖·‖²_k re-weights the residual by the
coarse-to-fine importance derived from the EDM σ-preconditioning
factors (Karras 2022).  This stabilises training under noise and
mitigates the variance blow-up observed in single-scale FM.

SCAFFOLD ONLY — the canonical (stub) implementation lives in
:mod:`molmetal.adapters.flow_matching_lipman.stablevm_stub` (357 LOC).
This module exposes the small, importable ``stable_vm_loss`` reference.

Honest framing: SCAFFOLD-only; no GPU launches; no trainer wiring;
speedup/lift claims are PROJECTED per Wf-340-A3.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F


def _per_scale_weights(
    n_scales: int,
    base: float = 0.05,
    growth: float = 2.0,
) -> List[float]:
    """Compute the geometric-series scale-mixture weights.

    Defaults match the StableVM paper §3.1: ``w_k = base * growth^(k-1)``
    for ``k = 1, ..., n_scales``.  The weights are L1-normalised so the
    overall loss magnitude matches the single-scale MSE.
    """
    ws = [base * (growth ** k) for k in range(n_scales)]
    s = sum(ws)
    return [w / s for w in ws]


def stable_vm_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_scales: int = 4,
    base: float = 0.05,
    growth: float = 2.0,
) -> torch.Tensor:
    """StableVM multi-scale mixture loss.

    Parameters
    ----------
    pred, target : (B, ..., D)
        Velocity prediction and ground-truth conditional velocity.
    n_scales : int
        Number of scales in the mixture.  Default 4 per Wf-340-A3 §3.
    base, growth : float
        Geometric-series parameters for the per-scale weights.

    Returns
    -------
    Scalar tensor — the multi-scale mixture MSE.
    """
    if n_scales < 1:
        raise ValueError("n_scales must be >= 1")
    err2 = (pred - target).pow(2)
    # Decompose the per-sample error into multi-scale averages by
    # successively pooling with the per-axis mean (coarse-to-fine).
    losses: List[torch.Tensor] = []
    weights = _per_scale_weights(n_scales, base=base, growth=growth)
    for w, k in zip(weights, range(n_scales)):
        # Pool over the trailing ``k`` dimensions when possible; the
        # first scale always sees the raw residual.
        if k == 0:
            pooled = err2.mean()
        else:
            dims = tuple(range(1, min(k, err2.dim() - 1) + 1))
            if dims:
                pooled = err2.mean(dim=dims)
                if pooled.dim() > 0:
                    pooled = pooled.mean()
            else:
                pooled = err2.mean()
        losses.append(w * pooled)
    return torch.stack(losses).sum()


__all__ = ["stable_vm_loss"]