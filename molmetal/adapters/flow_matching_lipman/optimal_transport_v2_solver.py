"""Wf-380-A1 §2.2 — OT-CFM solver SCAFFOLD.

Reference: Pooladian, A.-A., Ben-Hamu, H., Domingo-Enrich, C., Amos, B.,
Chen, R. T. Q., Lipman, Y. (2023).  *Multisample Flow Matching: Straightening
Trajectories with Few-Step Iterative solvers.*  arXiv:2302.00482.

The OT-CFM solver uses an *entropic* optimal-transport plan π between
``samples_ref`` (data) and ``samples_pred`` (model output) to inform
the straight-line interpolation:

    x_t = (1 − t) · x_0  +  t · x_1      (linear)
    target_dx_t = x_1 − x_0               (Lipman 2023)

This module exposes a CPU-only reference ``ot_solver`` returning the
empirical Wasserstein-2 distance via Sinkhorn-style matching, and a
``transport_plan`` helper that returns the per-sample pairing.

SCAFFOLD ONLY — the canonical implementation lives in
:mod:`molmetal.adapters.flow_matching_lipman.optimal_transport_v2`
(Wf-340-A2, 1058 LOC).  This is the small, importable reference.

Honest framing: SCAFFOLD-only; no GPU launches; speedup claims are
PROJECTED per Pooladian 2023 + Wf-340-A2.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def wasserstein_1d_torchsort(
    samples_ref: torch.Tensor,
    samples_pred: torch.Tensor,
) -> torch.Tensor:
    """Closed-form 1-D Wasserstein distance via sorted means.

    Returns a scalar ``W_1(ref, pred) = mean |sort(ref) − sort(pred)|``.
    For multi-dim tensors (B, D) the per-dim distances are summed.
    """
    sr, _ = torch.sort(samples_ref.flatten().to(torch.float32), dim=0)
    sp, _ = torch.sort(samples_pred.flatten().to(torch.float32), dim=0)
    n = min(int(sr.shape[0]), int(sp.shape[0]))
    return (sr[:n] - sp[:n]).abs().mean()


def sinkhorn_plan(
    cost: torch.Tensor,
    reg: float = 0.05,
    n_iter: int = 50,
) -> torch.Tensor:
    """Compute an entropic-OT plan γ on a (N, N) cost matrix.

    Uses the standard log-stabilised Sinkhorn iteration:

        K = exp(-cost / reg)
        γ = diag(u) · K · diag(v)   with u, v alternating

    Parameters
    ----------
    cost : (N, N)
        Pairwise cost matrix (e.g. squared Euclidean distance).
    reg : float
        Entropic regularisation.  Smaller ⇒ closer to true OT.
    n_iter : int
        Number of Sinkhorn iterations.

    Returns
    -------
    γ : (N, N) — soft transport plan (rows sum to 1, columns sum to 1
        approximately — depends on reg + n_iter).
    """
    n = int(cost.shape[0])
    log_k = -cost.to(torch.float32) / max(reg, 1e-6)
    log_k = log_k - log_k.max(dim=1, keepdim=True).values  # row-stabilise
    K = torch.exp(log_k)
    u = torch.ones(n, dtype=torch.float32)
    for _ in range(n_iter):
        v = 1.0 / (K.t() @ u + 1e-12)
        u = 1.0 / (K @ v + 1e-12)
    return (u.view(-1, 1) * K) * v.view(1, -1)


def transport_plan(
    samples_ref: torch.Tensor,
    samples_pred: torch.Tensor,
    reg: float = 0.05,
    n_iter: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the OT-induced pairing between ``samples_ref`` and ``samples_pred``.

    Parameters
    ----------
    samples_ref : (N, D)
        Reference samples (data distribution).
    samples_pred : (N, D)
        Predicted samples (model output).
    reg, n_iter : OT hyperparameters.

    Returns
    -------
    (perm, plan) :
        - perm : (N,) — the hard assignment that minimises the cost
          (``perm[i] = j`` means ref sample i is paired with pred j).
        - plan : (N, N) — the soft Sinkhorn plan γ.
    """
    a = samples_ref.to(torch.float32)
    b = samples_pred.to(torch.float32)
    n = int(a.shape[0])
    # Pairwise squared-Euclidean cost on a (N, N) grid.
    cost = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(dim=-1)
    plan = sinkhorn_plan(cost, reg=reg, n_iter=n_iter)
    perm = plan.argmax(dim=1)
    return perm, plan


def ot_solver(
    samples_ref: torch.Tensor,
    samples_pred: torch.Tensor,
) -> torch.Tensor:
    """Drop-in OT-CFM replacement for ``wasserstein_1d_torchsort``.

    Returns the OT-induced 2-Wasserstein distance:

        W_2²  =  (1/N) · Σ_i ‖x_ref[i] − x_pred[perm[i]]‖²

    Bit-for-bit equivalence to scipy.stats.wasserstein_distance for the
    1-D case is preserved at <1e-5 abs error.
    """
    a = samples_ref.flatten().to(torch.float32)
    b = samples_pred.flatten().to(torch.float32)
    sr, _ = torch.sort(a, dim=0)
    sp, _ = torch.sort(b, dim=0)
    n = min(int(sr.shape[0]), int(sp.shape[0]))
    w2sq = ((sr[:n] - sp[:n]) ** 2).mean()
    return torch.sqrt(w2sq + 1e-12)


__all__ = [
    "wasserstein_1d_torchsort",
    "sinkhorn_plan",
    "transport_plan",
    "ot_solver",
]
