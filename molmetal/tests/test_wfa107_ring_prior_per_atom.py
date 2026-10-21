"""Wf-A107 — the per-atom ring prior and the gradient-vanishing defect.

Wf-A96 measured the shipped prior (``formulation="count"``) at weight 0.01:
rings/mol moved 0.0 -> 7.05 (real 4.19) while the 3-ring share stayed at
0.44 (real 0.005) and the 5/6 share at 0.19 (real 0.97).  The prior was
firing hardest at the cold start and had almost no gradient left once the
model over-produced triangles.

The cause is structural, not a tuning problem: ``log1p(c)`` plus a /E
denominator make d(loss)/dp shrink as c grows.  ``per_atom`` normalises by
the atom count and drops ``log1p``, so the gradient does not vanish in the
over-production regime.  These tests pin:

  1. default ``count`` is bit-exact (the shipped configs are unchanged),
  2. ``per_atom`` is zero on a correct prediction,
  3. ``per_atom`` has a NON-vanishing gradient where ``count`` does not,
  4. the asymmetry (excess 3/4, deficit 5/6) is real,
  5. an unknown formulation is refused.
"""
from __future__ import annotations

import pytest
import torch

from molmetal.loss.ring_template import (
    RingTemplatePrior,
    RingTemplatePriorConfig,
    _cycle_counts,
)

_HEXAGON = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]


def _truth_hexagon() -> torch.Tensor:
    return torch.tensor(_HEXAGON, dtype=torch.long).t().contiguous()


def _logits_for(pairs, n_pairs: int, p_bond: float = 0.9) -> torch.Tensor:
    """Logits with ``p_bond`` on ``pairs`` and 0.35 elsewhere.

    ``p_bond = 1 - softmax[0]``, so with the other four classes at 0 the
    class-0 logit is ``log(4 * (1 - p) / p)``.
    """

    def _a(p: float) -> float:
        return float(torch.log(torch.tensor(4.0 * (1.0 - p) / p)))

    logits = torch.zeros((n_pairs, 5))
    logits[:, 0] = _a(0.35)
    for idx in pairs:
        logits[idx, 0] = _a(p_bond)
    return logits


def _pairs_from_edges(edges, n_atoms: int) -> torch.Tensor:
    """Full candidate pair list, with ``edges`` marked as bonded logits."""
    i, j = torch.triu_indices(n_atoms, n_atoms, offset=1)
    pairs = torch.stack([i, j])
    lookup = {tuple(sorted(e)): k for k, e in enumerate(zip(i.tolist(), j.tolist()))}
    return pairs, [lookup[tuple(sorted(e))] for e in edges]


def _grad_norm(prior: RingTemplatePrior, logits, pairs, truth, n_atoms: int) -> float:
    logits = logits.clone().requires_grad_(True)
    out = prior.compute(logits, pairs, n_atoms=n_atoms, truth_bonds=truth)
    out.backward()
    return float(logits.grad.norm())


def test_default_formulation_is_count_and_bit_exact():
    cfg = RingTemplatePriorConfig()
    assert cfg.formulation == "count"
    pairs, bonded = _pairs_from_edges(_HEXAGON, 6)
    logits = _logits_for(bonded, pairs.shape[1], p_bond=0.9)
    truth = _truth_hexagon()
    default = RingTemplatePrior(weight=0.1).compute(
        logits, pairs, n_atoms=6, truth_bonds=truth
    )
    explicit = RingTemplatePrior(
        weight=0.1, config=RingTemplatePriorConfig(formulation="count")
    ).compute(logits, pairs, n_atoms=6, truth_bonds=truth)
    assert torch.equal(default, explicit)


def test_zero_weight_is_graphless_in_per_atom_mode():
    prior = RingTemplatePrior(
        weight=0.0, config=RingTemplatePriorConfig(formulation="per_atom")
    )
    pairs, bonded = _pairs_from_edges(_HEXAGON, 6)
    logits = _logits_for(bonded, pairs.shape[1], p_bond=0.9)
    out = prior.compute(logits, pairs, n_atoms=6, truth_bonds=_truth_hexagon())
    assert out.item() == 0.0
    assert out.grad_fn is None


def test_per_atom_correct_prediction_is_zero():
    pairs, bonded = _pairs_from_edges(_HEXAGON, 6)
    logits = _logits_for(bonded, pairs.shape[1], p_bond=0.9)
    prior = RingTemplatePrior(
        weight=1.0, config=RingTemplatePriorConfig(formulation="per_atom")
    )
    out = prior.compute(logits, pairs, n_atoms=6, truth_bonds=_truth_hexagon())
    assert out.item() < 1e-3


def test_per_atom_penalises_excess_triangles():
    # K6: every pair bonded -> 20 triangles, truth is a single 6-ring.
    n = 6
    i, j = torch.triu_indices(n, n, offset=1)
    pairs = torch.stack([i, j])
    logits = _logits_for(range(pairs.shape[1]), pairs.shape[1], p_bond=0.9)
    counts = _cycle_counts(torch.ones(pairs.shape[1]), pairs, (3,))
    assert float(counts[3]) == pytest.approx(20.0, abs=1e-4)
    prior = RingTemplatePrior(
        weight=1.0, config=RingTemplatePriorConfig(formulation="per_atom")
    )
    out = prior.compute(logits, pairs, n_atoms=n, truth_bonds=_truth_hexagon())
    assert out.item() > 0.5


def test_per_atom_gradient_does_not_vanish_under_overproduction():
    """The regression this work package exists for.

    On a graph that has closed many triangles, the shipped count form's
    gradient is ~1e-2 while the per-atom form's is O(10).  Assert a >20x
    ratio so the test pins the mechanism, not a particular number.
    """
    n = 10
    i, j = torch.triu_indices(n, n, offset=1)
    pairs = torch.stack([i, j])
    logits = _logits_for(range(pairs.shape[1]), pairs.shape[1], p_bond=0.9)
    truth = _truth_hexagon()
    g_count = _grad_norm(
        RingTemplatePrior(weight=1.0), logits, pairs, truth, n_atoms=6
    )
    g_per_atom = _grad_norm(
        RingTemplatePrior(
            weight=1.0, config=RingTemplatePriorConfig(formulation="per_atom")
        ),
        logits,
        pairs,
        truth,
        n_atoms=6,
    )
    assert g_count > 0.0
    assert g_per_atom > 20.0 * g_count


def test_per_atom_promotes_a_missing_six_ring():
    # No hard ring yet.  Only the six hexagon edges are plausible
    # (p=0.35); every other pair sits at p=0.01 so a 6-walk that uses one
    # of them is negligible.  The gradient must therefore point at the
    # hexagon edges, and it must raise their p_bond.
    pairs, _ = _pairs_from_edges(_HEXAGON, 6)
    # Every pair at p=0.01, then the six hexagon edges at p=0.35.
    logits = _logits_for(range(pairs.shape[1]), pairs.shape[1], p_bond=0.01)
    lookup = {
        tuple(sorted(e)): k
        for k, e in enumerate(zip(pairs[0].tolist(), pairs[1].tolist()))
    }
    ring = [lookup[tuple(sorted(e))] for e in _HEXAGON]
    for k in ring:
        logits[k, 0] = float(torch.log(torch.tensor(4.0 * 0.65 / 0.35)))
    truth = _truth_hexagon()
    prior = RingTemplatePrior(
        weight=1.0, config=RingTemplatePriorConfig(formulation="per_atom")
    )
    logits = logits.clone().requires_grad_(True)
    out = prior.compute(logits, pairs, n_atoms=6, truth_bonds=truth)
    assert out.item() > 0.0
    out.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    other = [k for k in range(pairs.shape[1]) if k not in set(ring)]
    # d(loss)/d(logit0) > 0 means "lower the NO_BOND logit" = raise p_bond.
    assert logits.grad[ring, 0].mean() > 0.0
    assert logits.grad[ring, 0].abs().mean() > 10.0 * logits.grad[other, 0].abs().mean()


def test_per_atom_does_not_penalise_extra_six_rings():
    """Deficit-only for 5/6: overshooting the 6-count is not punished."""
    pairs, bonded = _pairs_from_edges(_HEXAGON, 6)
    # Truth = one hexagon; prediction = the same hexagon plus a second
    # disjoint 6-ring on atoms 6..11 -> 6-count 2 vs truth 1.
    second = [(6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 6)]
    pairs, bonded = _pairs_from_edges(_HEXAGON + second, 12)
    logits = _logits_for(bonded, pairs.shape[1], p_bond=0.9)
    truth = _truth_hexagon()
    prior = RingTemplatePrior(
        weight=1.0, config=RingTemplatePriorConfig(formulation="per_atom")
    )
    out = prior.compute(logits, pairs, n_atoms=12, truth_bonds=truth)
    # Only the 5/6 deficit terms can contribute; both are satisfied.
    assert out.item() < 1e-3


def test_unknown_formulation_is_refused():
    with pytest.raises(ValueError, match="formulation"):
        RingTemplatePriorConfig(formulation="share")


def test_soft_six_ring_survives_the_triangle_correction():
    """Wf-A107 — the old correction zeroed the 6-count on a soft graph.

    A hexagon at p=0.35 plus weak (p=0.01) edges that close partial
    triangles made ``raw[6] - 0.5 * raw[3]`` negative, and the clamp then
    returned an exact 0.  The correction must use the SQUARED triangle
    product, so the hexagon's 6-count stays positive.
    """
    pairs, _ = _pairs_from_edges(_HEXAGON, 6)
    lookup = {
        tuple(sorted(e)): k
        for k, e in enumerate(zip(pairs[0].tolist(), pairs[1].tolist()))
    }
    ring = [lookup[tuple(sorted(e))] for e in _HEXAGON]
    p = torch.full((pairs.shape[1],), 0.01)
    p[ring] = 0.35
    counts = _cycle_counts(p, pairs, (3, 6), n_atoms=6)
    assert float(counts[3]) > 0.0          # partial triangles exist ...
    assert float(counts[6]) > 1e-4         # ... but the 6-ring survives
    # The corrected value is the hexagon product minus a negligible
    # triangle-squared term, i.e. just over 0.35**6.
    assert float(counts[6]) == pytest.approx(0.35 ** 6, rel=0.05)


def test_triangle_correction_keeps_the_six_ring_gradient():
    """Wf-A107 — the old correction also killed the 6-ring GRADIENT.

    Same hexagon, but the chord 0-2 sits at the same probability (0.35) as
    the ring, which is the realistic case: in a dense predicted graph most
    ring edges also sit in a partial triangle.  ``0.5 * raw[3]`` then
    exceeds the hexagon's own 6-count and the clamp returns exact 0, so
    ``d(count6)/d(hex-edge logit) == 0`` -- the term cannot fire.  With the
    squared correction the count equals the hexagon product and the
    gradient is non-zero.
    """
    pairs, _ = _pairs_from_edges(_HEXAGON + [(0, 2)], 6)
    lookup = {
        tuple(sorted(e)): k
        for k, e in enumerate(zip(pairs[0].tolist(), pairs[1].tolist()))
    }
    ring = [lookup[tuple(sorted(e))] for e in _HEXAGON]
    p = torch.zeros(pairs.shape[1])
    p[ring] = 0.35
    p[lookup[(0, 2)]] = 0.35

    p_grad = p.clone().requires_grad_(True)
    six = _cycle_counts(p_grad, pairs, (6,), n_atoms=6)[6]
    six.backward()
    # edge (3,4) is in the hexagon but NOT in the triangle 0-1-2.
    g_hex = float(p_grad.grad[lookup[(3, 4)]])

    assert float(six) == pytest.approx(0.35 ** 6, rel=1e-4)
    assert abs(g_hex) > 1e-4
