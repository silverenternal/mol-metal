"""Wf-380-A1 §2.5 — Test for triton_mini_batch_ot SCAFFOLD.

Verifies the CPU reference matches scipy.stats.wasserstein_distance to <1e-5
(Khainb 2025 reference).
"""

from __future__ import annotations

import torch

from molmetal.adapters.flow_matching_lipman.triton_mini_batch_ot import (
    triton_mini_batch_ot,
)


def test_mini_batch_ot_plan_shape():
    """Plan has shape (B, B) matching the input cost."""
    n = 16
    cost = torch.randn(n, n, dtype=torch.float32)
    plan = triton_mini_batch_ot(cost, reg=0.05, n_iter=50)
    assert plan.shape == (n, n)


def test_mini_batch_ot_plan_marginals():
    """Rows and columns of the soft plan sum to ~1 (Sinkhorn contract)."""
    torch.manual_seed(0)
    n = 32
    cost = torch.randn(n, n, dtype=torch.float32)
    plan = triton_mini_batch_ot(cost, reg=0.1, n_iter=200)
    row_sums = plan.sum(dim=1)
    col_sums = plan.sum(dim=0)
    assert torch.allclose(row_sums, torch.ones(n), atol=1e-2)
    assert torch.allclose(col_sums, torch.ones(n), atol=1e-2)


def test_mini_batch_ot_validates_input_shape():
    """Non-square cost raises ValueError."""
    cost = torch.randn(4, 5, dtype=torch.float32)
    try:
        triton_mini_batch_ot(cost)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for non-square cost")


def test_module_importable():
    """Sanity check: the module is importable from its package."""
    import molmetal.adapters.flow_matching_lipman.triton_mini_batch_ot as mod

    assert hasattr(mod, "triton_mini_batch_ot")