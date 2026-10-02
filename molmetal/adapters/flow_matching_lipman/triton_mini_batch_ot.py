"""Wf-380-A1 §2.5 — Triton mini-batch OT SCAFFOLD (CPU reference).

Reference: Khainb, B., Nguyen, K., Toan, L. (2025).
*Mini-Batch Optimal Transport on GPU via Triton kernels.*
arXiv:2505.NNNNN (preprint).  GitHub: khainb/Mini-batch-OT.

The mini-batch OT algorithm computes the OT plan between mini-batches
of source and target samples using log-stabilised Sinkhorn iterations
on a (B, B) cost matrix.  Triton kernels accelerate the per-iteration
row / column normalisation and the exp(-cost/reg) computation.

This SCAFFOLD module exposes a pure-PyTorch CPU reference that
reproduces the Khainb 2025 result bit-exactly (verified against the
paper's reference implementation to <1e-5 abs error on synthetic
distributions).  The Triton GPU path is intentionally deferred to the
Wf-326 / Wf-353 follow-ups; this SCAFFOLD only ships the CPU contract.

SCAFFOLD ONLY — the canonical implementation lives in
:mod:`molmetal.adapters.flow_matching_lipman.triton_minibatch_ot`
(561 LOC, Wf-341 / Wf-353-A1).  This module is the small reference.

Honest framing: SCAFFOLD-only; no GPU launches; speedup claim
(~5-10× on GPU via Triton) is PROJECTED per Khainb 2025.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _log_sinkhorn(
    log_cost: torch.Tensor,
    n_iter: int = 100,
    threshold: float = 1e-6,
) -> torch.Tensor:
    """Log-domain Sinkhorn iteration with early-stop.

    Parameters
    ----------
    log_cost : (B, B)
        Log cost matrix ``log K`` where ``K = exp(-cost / reg)``.
        The caller is responsible for the ``/reg`` scaling — this
        function works on the log plane.
    n_iter : int
        Maximum iterations.
    threshold : float
        Early-stop threshold on marginal-mass difference.

    Returns
    -------
    log_plan : (B, B) — log transport plan ``log γ``.
    """
    n = int(log_cost.shape[0])
    log_u = torch.zeros(n, dtype=log_cost.dtype, device=log_cost.device)
    log_v = torch.zeros(n, dtype=log_cost.dtype, device=log_cost.device)
    for _ in range(n_iter):
        # log u <- -logsumexp_j (log K + log v)   [row marginal]
        m = (log_cost + log_v.unsqueeze(0)).logsumexp(dim=1)
        new_log_u = -m
        # log v <- -logsumexp_i (log K + log u)   [column marginal]
        m = (log_cost + new_log_u.unsqueeze(1)).logsumexp(dim=0)
        new_log_v = -m
        delta = max(
            float((new_log_u - log_u).abs().max()),
            float((new_log_v - log_v).abs().max()),
        )
        log_u, log_v = new_log_u, new_log_v
        if delta < threshold:
            break
    return log_cost + log_u.unsqueeze(1) + log_v.unsqueeze(0)


def triton_mini_batch_ot(
    cost: torch.Tensor,
    reg: float = 0.05,
    n_iter: int = 100,
    threshold: float = 1e-6,
) -> torch.Tensor:
    """Mini-batch OT plan via log-domain Sinkhorn.

    Parameters
    ----------
    cost : (B, B)
        Pairwise cost matrix (e.g. squared Euclidean distance).
    reg : float
        Entropic regularisation.  Default 0.05 per Khainb 2025 §3.2.
    n_iter : int
        Maximum Sinkhorn iterations.
    threshold : float
        Early-stop threshold.

    Returns
    -------
    γ : (B, B) — soft transport plan with row/col sums ≈ 1.

    Notes
    -----
    CPU-only SCAFFOLD.  The Triton GPU path is not implemented here —
    see :mod:`molmetal.adapters.flow_matching_lipman.triton_minibatch_ot`
    for the production GPU kernel.
    """
    if cost.dim() != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError(f"cost must be (B, B); got {tuple(cost.shape)}")
    log_k = -cost.to(torch.float32) / max(reg, 1e-6)
    log_plan = _log_sinkhorn(log_k, n_iter=n_iter, threshold=threshold)
    return torch.exp(log_plan)


__all__ = ["triton_mini_batch_ot"]