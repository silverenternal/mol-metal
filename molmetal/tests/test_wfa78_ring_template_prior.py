"""Wf-A78 / OPT-5 — tests for the training-time ring-template prior.

The prior exists because OPT-2 measured the defect as ring SIZE
(3-rings 89.8% generated vs 0.5% real; 5/6-rings 1.85% vs 96.92%).
These tests pin the three properties the trainer relies on:

  1. default-off is bit-exact (exact zero, no graph),
  2. a correct soft graph scores ~0,
  3. a wrong ring size scores > 0 and produces a gradient.
"""
from __future__ import annotations

import pytest
import torch

from molmetal.loss.ring_template import (
    RingTemplatePrior,
    RingTemplatePriorConfig,
    _cycle_counts,
)


def _bond_logits(pairs, n_pairs, high=3.0, low=-3.0):
    """Logits with class 1 (SINGLE) high on the listed pairs, NO_BOND elsewhere."""
    logits = torch.zeros((n_pairs, 5))
    logits[:, 0] = high
    for idx in pairs:
        logits[idx, 0] = low
        logits[idx, 1] = high
    return logits


def _hexagon():
    """6-ring truth: atoms 0-1-2-3-4-5-0."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_zero_weight_is_exact_zero_and_graphless():
    prior = RingTemplatePrior(weight=0.0)
    logits = torch.randn(6, 5, requires_grad=True)
    pairs = _hexagon()
    out = prior.compute(logits, pairs, n_atoms=6, truth_bonds=pairs)
    assert out.item() == 0.0
    assert out.grad_fn is None  # nothing was computed


def test_perfect_prediction_scores_near_zero():
    pairs = _hexagon()
    logits = _bond_logits(range(6), 6)
    out = RingTemplatePrior(weight=1.0).compute(
        logits, pairs, n_atoms=6, truth_bonds=pairs
    )
    assert out.item() < 1e-3


def test_wrong_ring_size_scores_positive():
    # Truth is a 6-ring; the prediction closes a 3-ring instead.
    truth = _hexagon()
    pred_pairs = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    logits = _bond_logits(range(3), 3)
    out = RingTemplatePrior(weight=0.1).compute(
        logits, pred_pairs, n_atoms=6, truth_bonds=truth
    )
    assert out.item() > 0.05  # both the missing 6-count and the extra 3-count


def test_acyclic_truth_is_skipped():
    # No ring profile to match -> the prior must not invent one.
    truth = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    pairs = truth
    logits = _bond_logits(range(3), 3)
    out = RingTemplatePrior(weight=1.0).compute(
        logits, pairs, n_atoms=4, truth_bonds=truth
    )
    assert out.item() == 0.0


def test_gradient_reaches_bond_logits():
    # Same 6-ring topology as the truth, but one edge is only half-open,
    # so the 6-share sits strictly between 0 and 1 and the prior has a
    # slope.  (A pure triangle is a boundary case: its 6-share is exactly
    # 0 for every bond probability, so the correct gradient there IS 0.)
    truth = _hexagon()
    logits = _bond_logits(range(5), 6).requires_grad_(True)  # edge 5 stays weak
    out = RingTemplatePrior(weight=0.1).compute(
        logits, truth, n_atoms=6, truth_bonds=truth
    )
    assert out.item() > 0.0
    out.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_tuple_pair_index_accepted():
    pairs = _hexagon()
    logits = _bond_logits(range(6), 6)
    out_tuple = RingTemplatePrior(weight=1.0).compute(
        logits, (pairs[0], pairs[1]), n_atoms=6, truth_bonds=pairs
    )
    out_tensor = RingTemplatePrior(weight=1.0).compute(
        logits, pairs, n_atoms=6, truth_bonds=pairs
    )
    assert torch.allclose(out_tuple, out_tensor)


def test_empty_inputs_return_zero():
    prior = RingTemplatePrior(weight=1.0)
    assert prior.compute(None, torch.zeros((2, 0), dtype=torch.long), 6).item() == 0.0
    assert prior.compute(
        torch.zeros((0, 5)), torch.zeros((2, 0), dtype=torch.long), 6,
        truth_bonds=_hexagon(),
    ).item() == 0.0


def test_batch_mean_matches_manual_mean():
    pairs = _hexagon()
    logits = _bond_logits(range(6), 6)
    prior = RingTemplatePrior(weight=1.0)
    one = prior.compute(logits, pairs, n_atoms=6, truth_bonds=pairs)
    batch = prior.compute_batch(
        [(logits, pairs, 6, pairs), (logits, pairs, 6, pairs)],
        device=torch.device("cpu"),
    )
    assert torch.allclose(batch, one, atol=1e-6)


def test_spectrum_counts_each_cycle_size_exactly():
    # One triangle -> 1 on size 3; one hexagon -> 1 on size 6.
    tri = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    c3 = _cycle_counts(torch.ones(3), tri, (3, 4, 5, 6))
    assert abs(float(c3[3]) - 1.0) < 1e-5
    assert float(c3[6]) < 1e-6

    hexa = _hexagon()
    c6 = _cycle_counts(torch.ones(6), hexa, (3, 4, 5, 6))
    assert abs(float(c6[6]) - 1.0) < 1e-5
    assert float(c6[3]) < 1e-6


def test_acyclic_graph_has_zero_counts():
    path = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    counts = _cycle_counts(torch.ones(3), path, (3, 4, 5, 6))
    assert all(float(v) < 1e-6 for v in counts.values())
