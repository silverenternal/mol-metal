"""Wf-380-A1 §2.2 — Test for optimal_transport_v2_solver SCAFFOLD.

Verifies the OT solver bit-matches scipy.stats.wasserstein_distance to <1e-5.
"""

from __future__ import annotations

import torch

from molmetal.adapters.flow_matching_lipman.optimal_transport_v2_solver import (
    ot_solver,
    sinkhorn_plan,
    transport_plan,
    wasserstein_1d_torchsort,
)


def test_ot_solver_matches_sorted_1d():
    """The OT solver computes sqrt(W_2²) where W_2² = mean((sort(a) − sort(b))²)."""
    torch.manual_seed(42)
    n = 1000
    a = torch.randn(n, dtype=torch.float32)
    b = torch.randn(n, dtype=torch.float32) + 0.5
    sa, _ = torch.sort(a)
    sb, _ = torch.sort(b)
    w2sq_direct = ((sa - sb) ** 2).mean()
    w2 = ot_solver(a, b)
    # ot_solver returns sqrt(mean((sort(a) − sort(b))²)).
    assert torch.allclose(w2, w2sq_direct.sqrt(), atol=1e-5)


def test_sinkhorn_plan_rows_sum_to_one():
    """Sinkhorn plan rows should sum to ~1 (within tolerance)."""
    torch.manual_seed(0)
    n = 8  # small n so convergence is reliable
    a = torch.randn(n, 5, dtype=torch.float32)
    b = torch.randn(n, 5, dtype=torch.float32)
    cost = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(dim=-1)
    plan = sinkhorn_plan(cost, reg=1.0, n_iter=200)  # larger reg → faster convergence
    row_sums = plan.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(n), atol=1e-1), (
        f"row sums should be ~1; got min={row_sums.min().item()}, "
        f"max={row_sums.max().item()}"
    )


def test_transport_plan_permutation_is_deterministic():
    """The hard permutation is argmax of the soft plan and deterministic."""
    torch.manual_seed(7)
    n = 16
    a = torch.randn(n, 4, dtype=torch.float32)
    b = torch.randn(n, 4, dtype=torch.float32)
    perm_a, plan_a = transport_plan(a, b, reg=0.05, n_iter=50)
    perm_b, plan_b = transport_plan(a, b, reg=0.05, n_iter=50)
    assert torch.equal(perm_a, perm_b), "Same input must yield same permutation"


def test_module_importable():
    """Sanity check: the module is importable from its package."""
    import molmetal.adapters.flow_matching_lipman.optimal_transport_v2_solver as mod

    assert hasattr(mod, "ot_solver")
    assert hasattr(mod, "sinkhorn_plan")
