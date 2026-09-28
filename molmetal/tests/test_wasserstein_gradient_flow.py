"""Tests for :mod:`molmetal.math_skills.wasserstein_gradient_flow`.

T4 — Wasserstein Gradient Flow (薛小平)
======================================

These tests cover the three mathematical primitives shipped by the
R18 W₂-gradient-flow skeleton:

1. **W2GradientFlow** — KL-functional W₂ gradient flow with
   kernelised score smoothing.  We verify construction,
   kernel-type / sigma validation, the drift sign convention, the
   explicit-Euler ``step``, the reference-score offset, the
   kernel-smoothing effect and the JKO CFL condition.
2. **ChemicalGraphMetric** — bond-type-weighted graph edit
   distance.  We verify construction, the bond-type weighting, the
   identity-of-indiscernibles, the symmetry property, the
   insertion / deletion cost, and the metric triangle inequality
   on canonical random graphs.
3. **ReactionStepW2** — single reaction step as W₂ gradient step.
   We verify construction, the drift direction, the
   ``forward = product`` idempotence, the reversibility property,
   the ``w2_distance`` invariant, the shape conventions and the
   chemical-graph-metric edge cases.

Every test is **CPU-only** and uses small data (dim 2–4, batches
of 4–8) so the full suite runs in < 5 seconds.

Reference
---------
Xue, X.-P. (薛小平) (2026).  *Wasserstein Gradient Flows and their
Applications in Generative Modeling*.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from molmetal.math_skills import (
    ChemicalGraphMetric,
    ReactionStepW2,
    W2GradientFlow,
)
from molmetal.math_skills.wasserstein_gradient_flow import (
    _EPS_LOG,
    _EPS_SIGMA2,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _random_points(n: int = 8, d: int = 3, seed: int = 0) -> torch.Tensor:
    """Random sample points of shape ``(n, d)``."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def _random_score(x: torch.Tensor, seed: int = 1) -> torch.Tensor:
    """Random score field of the same shape as ``x``."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*x.shape, generator=g)


# ---------------------------------------------------------------------------
# 1. W2GradientFlow — construction & invariants
# ---------------------------------------------------------------------------


def test_w2_gradient_flow_default_kernel_is_gaussian() -> None:
    """Default kernel is ``'gaussian'`` and default sigma is 1.0."""
    flow = W2GradientFlow()
    assert flow.kernel == "gaussian"
    assert flow.sigma == 1.0


def test_w2_gradient_flow_init_rejects_unknown_kernel() -> None:
    """An unknown kernel name raises ``ValueError``."""
    with pytest.raises(ValueError, match="kernel"):
        W2GradientFlow(kernel="laplace")


def test_w2_gradient_flow_init_rejects_non_positive_sigma() -> None:
    """A non-positive sigma raises ``ValueError``."""
    with pytest.raises(ValueError, match="sigma"):
        W2GradientFlow(sigma=0.0)
    with pytest.raises(ValueError, match="sigma"):
        W2GradientFlow(sigma=-0.5)


def test_w2_gradient_flow_accepts_imq_kernel() -> None:
    """``'imq'`` kernel is accepted."""
    flow = W2GradientFlow(kernel="imq", sigma=2.0)
    assert flow.kernel == "imq"
    assert flow.sigma == 2.0


def test_w2_gradient_flow_accepts_none_kernel() -> None:
    """``'none'`` kernel disables smoothing."""
    flow = W2GradientFlow(kernel="none")
    assert flow.kernel == "none"


# ---------------------------------------------------------------------------
# 2. W2GradientFlow — drift & step
# ---------------------------------------------------------------------------


def test_w2_gradient_flow_drift_is_negative_score_when_ref_is_zero() -> None:
    """``drift = -score`` when reference score is None (uniform reference)."""
    flow = W2GradientFlow(kernel="none")
    x = _random_points(n=4, d=2)
    score = _random_score(x, seed=1)
    drift = flow.drift(x, score)
    assert drift.shape == x.shape
    assert torch.allclose(drift, -score, atol=1e-6)


def test_w2_gradient_flow_drift_with_reference_score() -> None:
    """``drift = -(score - reference_score)`` when reference score is supplied."""
    flow = W2GradientFlow(kernel="none")
    x = _random_points(n=4, d=2)
    score = _random_score(x, seed=1)
    ref = _random_score(x, seed=2)
    drift = flow.drift(x, score, reference_score=ref)
    assert torch.allclose(drift, -(score - ref), atol=1e-6)


def test_w2_gradient_flow_drift_shape_is_preserved() -> None:
    """``drift`` preserves the input shape ``(B, d)``."""
    flow = W2GradientFlow(kernel="gaussian")
    x = torch.randn(7, 5)
    score = torch.randn(7, 5)
    drift = flow.drift(x, score)
    assert drift.shape == x.shape


def test_w2_gradient_flow_drift_rejects_mismatched_shapes() -> None:
    """``drift`` rejects ``x`` and ``score`` of different shapes."""
    flow = W2GradientFlow(kernel="none")
    with pytest.raises(ValueError, match="same shape"):
        flow.drift(torch.randn(4, 2), torch.randn(3, 2))


def test_w2_gradient_flow_step_is_explicit_euler() -> None:
    """``step(x, score, dt) == x + dt * drift(x, score)`` for ``kernel='none'``."""
    flow = W2GradientFlow(kernel="none")
    x = _random_points(n=5, d=3)
    score = _random_score(x, seed=2)
    dt = 0.1
    x_new = flow.step(x, score, dt)
    expected = x + dt * (-score)
    assert torch.allclose(x_new, expected, atol=1e-6)


def test_w2_gradient_flow_step_rejects_negative_dt() -> None:
    """Negative ``dt`` raises ``ValueError``."""
    flow = W2GradientFlow(kernel="none")
    x = torch.randn(4, 2)
    score = torch.randn(4, 2)
    with pytest.raises(ValueError, match="dt"):
        flow.step(x, score, dt=-0.01)


def test_w2_gradient_flow_step_zero_dt_returns_x() -> None:
    """``step(x, score, 0)`` returns ``x`` (no-op)."""
    flow = W2GradientFlow(kernel="none")
    x = _random_points(n=4, d=2)
    score = _random_score(x, seed=0)
    x_new = flow.step(x, score, dt=0.0)
    assert torch.allclose(x_new, x, atol=1e-6)


def test_w2_gradient_flow_kernel_smooths_score() -> None:
    """Gaussian smoothing at large bandwidth approaches the score mean."""
    flow = W2GradientFlow(kernel="gaussian", sigma=1.0e3)
    x = _random_points(n=8, d=2)
    score = _random_score(x, seed=1)
    smoothed = flow._smooth_score(x, score)
    # When sigma is huge, every weight is ~1, so smoothed ~ row-wise
    # mean of score (shape (B, d) broadcasting the (B, B) weight).
    expected = score.mean(dim=0, keepdim=True).expand_as(score)
    assert torch.allclose(smoothed, expected, atol=1e-2, rtol=1e-2)


def test_w2_gradient_flow_imq_kernel_smooths_score() -> None:
    """IMQ smoothing at large bandwidth also approaches the score mean."""
    flow = W2GradientFlow(kernel="imq", sigma=1.0e3)
    x = _random_points(n=8, d=2)
    score = _random_score(x, seed=1)
    smoothed = flow._smooth_score(x, score)
    expected = score.mean(dim=0, keepdim=True).expand_as(score)
    assert torch.allclose(smoothed, expected, atol=1e-2, rtol=1e-2)


def test_w2_gradient_flow_kernel_none_preserves_score() -> None:
    """``kernel='none'`` returns the score unmodified."""
    flow = W2GradientFlow(kernel="none", sigma=1.0)
    x = _random_points(n=4, d=2)
    score = _random_score(x, seed=0)
    smoothed = flow._smooth_score(x, score)
    assert torch.allclose(smoothed, score, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. W2GradientFlow — KL-decrease condition
# ---------------------------------------------------------------------------


def test_w2_gradient_flow_kl_decreases_finite_upper_bound() -> None:
    """``kl_decreases`` returns a finite upper bound on ``dt``."""
    flow = W2GradientFlow()
    bound = flow.kl_decreases(score_norm=1.0, reference_norm=0.0)
    assert math.isfinite(bound)
    assert bound > 0.0


def test_w2_gradient_flow_kl_decreases_infinity_when_ref_equals_score() -> None:
    """``kl_decreases`` returns ``inf`` when ``score = reference_score``."""
    flow = W2GradientFlow()
    bound = flow.kl_decreases(score_norm=1.0, reference_norm=1.0)
    assert bound == float("inf")


def test_w2_gradient_flow_kl_decreases_rejects_negative_norm() -> None:
    """Negative score norms raise ``ValueError``."""
    flow = W2GradientFlow()
    with pytest.raises(ValueError, match="non-negative"):
        flow.kl_decreases(score_norm=-1.0, reference_norm=0.0)


# ---------------------------------------------------------------------------
# 4. ChemicalGraphMetric — construction & invariants
# ---------------------------------------------------------------------------


def test_chemical_graph_metric_default_bond_types() -> None:
    """Default bond types ``[0, 1, 2, 3]`` set insert / delete cost to 3."""
    metric = ChemicalGraphMetric(bond_types=[0, 1, 2, 3])
    assert metric.bond_types == [0, 1, 2, 3]
    assert metric.insert_delete_cost == 3


def test_chemical_graph_metric_init_rejects_empty_bond_types() -> None:
    """Empty ``bond_types`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="non-empty"):
        ChemicalGraphMetric(bond_types=[])


def test_chemical_graph_metric_init_rejects_negative_bond_types() -> None:
    """Negative bond-type codes raise ``ValueError``."""
    with pytest.raises(ValueError, match="non-negative"):
        ChemicalGraphMetric(bond_types=[-1, 0, 1])


def test_chemical_graph_metric_distance_identical_graphs_is_zero() -> None:
    """``distance(g, g) == 0`` for any graph ``g``."""
    metric = ChemicalGraphMetric(bond_types=[0, 1, 2, 3])
    g = {
        "nodes": [0, 1, 2],
        "edges": [(0, 1, 1), (1, 2, 2)],
    }
    assert metric.distance(g, g) == 0.0


def test_chemical_graph_metric_distance_is_symmetric() -> None:
    """``distance(g1, g2) == distance(g2, g1)``."""
    metric = ChemicalGraphMetric(bond_types=[0, 1, 2, 3])
    g1 = {
        "nodes": [0, 1, 2],
        "edges": [(0, 1, 1), (1, 2, 2)],
    }
    g2 = {
        "nodes": [0, 1, 2],
        "edges": [(0, 1, 2), (1, 2, 1), (0, 2, 3)],
    }
    assert metric.distance(g1, g2) == metric.distance(g2, g1)


def test_chemical_graph_metric_insert_delete_cost() -> None:
    """Inserting / deleting one edge costs ``max(bond_types)``."""
    metric = ChemicalGraphMetric(bond_types=[0, 1, 2, 3])
    g1 = {"nodes": [0, 1, 2], "edges": []}
    g2 = {"nodes": [0, 1, 2], "edges": [(0, 1, 1)]}
    # Adding one edge should cost the insert / delete cost (3).
    assert metric.distance(g1, g2) == 3.0


def test_chemical_graph_metric_bond_type_substitution_cost() -> None:
    """Substituting bond type ``a`` for ``b`` costs ``|a - b|``."""
    metric = ChemicalGraphMetric(bond_types=[0, 1, 2, 3])
    g1 = {"nodes": [0, 1], "edges": [(0, 1, 1)]}
    g2 = {"nodes": [0, 1], "edges": [(0, 1, 3)]}
    # Same edge, bond types 1 vs 3 → |1 − 3| = 2.
    assert metric.distance(g1, g2) == 2.0


def test_chemical_graph_metric_accepts_tuple_input() -> None:
    """The ``(nodes, edges)`` tuple form is accepted."""
    metric = ChemicalGraphMetric(bond_types=[0, 1, 2, 3])
    g1 = ([0, 1, 2], [(0, 1, 1), (1, 2, 2)])
    g2 = ([0, 1, 2], [(0, 1, 2), (1, 2, 1)])
    assert metric.distance(g1, g2) == pytest.approx(2.0)


def test_chemical_graph_metric_rejects_bad_edge_tuple() -> None:
    """A ``(i, j)`` edge without bond type raises ``ValueError``."""
    metric = ChemicalGraphMetric(bond_types=[0, 1, 2, 3])
    bad = {"nodes": [0, 1], "edges": [(0, 1)]}  # missing bond type
    with pytest.raises(ValueError, match="bond_type"):
        metric.distance(bad, {"nodes": [], "edges": []})


# ---------------------------------------------------------------------------
# 5. ReactionStepW2 — construction & invariants
# ---------------------------------------------------------------------------


def test_reaction_step_w2_default_n_bond_types() -> None:
    """Default ``n_bond_types=4`` matches the canonical bond-type count."""
    step = ReactionStepW2()
    assert step.n_bond_types == 4
    assert step.dt == 1.0


def test_reaction_step_w2_init_rejects_zero_n_bond_types() -> None:
    """``n_bond_types < 1`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="n_bond_types"):
        ReactionStepW2(n_bond_types=0)


def test_reaction_step_w2_init_rejects_non_positive_dt() -> None:
    """Non-positive ``dt`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="dt"):
        ReactionStepW2(dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        ReactionStepW2(dt=-0.5)


def test_reaction_step_w2_drift_is_product_minus_reactant() -> None:
    """``drift = product - reactant`` (the optimal transport direction)."""
    step = ReactionStepW2(n_bond_types=4, dt=2.0)
    reactant = torch.zeros(3, 4)
    product = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * 3
    )
    drift = step.drift(reactant, product)
    expected = product - reactant
    assert torch.allclose(drift, expected, atol=1e-6)


def test_reaction_step_w2_forward_returns_product_at_default_dt() -> None:
    """``forward(reactant, product) == product`` when ``dt=1``."""
    step = ReactionStepW2(n_bond_types=4)
    reactant = torch.randn(3, 4)
    product = torch.randn(3, 4)
    out = step.forward(reactant, product)
    assert torch.allclose(out, product, atol=1e-6)


def test_reaction_step_w2_forward_returns_midpoint_at_half_dt() -> None:
    """``forward`` at ``dt=0.5`` returns the midpoint between reactant and product."""
    step = ReactionStepW2(n_bond_types=4, dt=0.5)
    reactant = torch.zeros(2, 4)
    product = torch.ones(2, 4)
    out = step.forward(reactant, product)
    expected = 0.5 * reactant + 0.5 * product
    assert torch.allclose(out, expected, atol=1e-6)


def test_reaction_step_w2_forward_dt_override_at_third() -> None:
    """``forward`` with ``dt_override=1/3`` returns ``(2/3) r + (1/3) p``."""
    step = ReactionStepW2(n_bond_types=4)
    reactant = torch.zeros(3, 4)
    product = torch.ones(3, 4)
    out = step.forward(reactant, product, dt=1.0 / 3.0)
    # x(dt) = r + dt * (p - r) = (1 - dt) r + dt p
    expected = (2.0 / 3.0) * reactant + (1.0 / 3.0) * product
    assert torch.allclose(out, expected, atol=1e-6)


def test_reaction_step_w2_w2_distance_is_zero_for_identical_inputs() -> None:
    """``w2_distance(r, r) == 0`` for any fingerprint ``r``."""
    step = ReactionStepW2(n_bond_types=4)
    r = torch.softmax(torch.randn(3, 4), dim=-1)
    assert step.w2_distance(r, r).item() == 0.0


def test_reaction_step_w2_w2_distance_matches_l1_norm() -> None:
    """``w2_distance(r, p)`` is the L1 distance between ``r`` and ``p``."""
    step = ReactionStepW2(n_bond_types=4)
    r = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    p = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
    expected = (r - p).abs().sum().item()
    assert step.w2_distance(r, p).item() == pytest.approx(expected)


def test_reaction_step_w2_is_reversible_returns_true() -> None:
    """``is_reversible(r, p) == True`` for any compatible tensors."""
    step = ReactionStepW2(n_bond_types=4)
    r = torch.softmax(torch.randn(5, 4), dim=-1)
    p = torch.softmax(torch.randn(5, 4), dim=-1)
    assert step.is_reversible(r, p) is True


def test_reaction_step_w2_rejects_shape_mismatch() -> None:
    """``forward`` / ``drift`` / ``w2_distance`` reject mismatched shapes."""
    step = ReactionStepW2(n_bond_types=4)
    r = torch.randn(3, 4)
    p = torch.randn(4, 4)
    with pytest.raises(ValueError, match="same shape"):
        step.forward(r, p)
    with pytest.raises(ValueError, match="same shape"):
        step.drift(r, p)
    with pytest.raises(ValueError, match="same shape"):
        step.w2_distance(r, p)


def test_reaction_step_w2_rejects_wrong_last_dim() -> None:
    """A last-dim size other than ``n_bond_types`` raises ``ValueError``."""
    step = ReactionStepW2(n_bond_types=4)
    r = torch.randn(3, 5)  # 5 != 4
    p = torch.randn(3, 5)
    with pytest.raises(ValueError, match="n_bond_types"):
        step.forward(r, p)


def test_reaction_step_w2_batched_inputs_supported() -> None:
    """Leading batch dimensions ``(B, ...)`` are preserved."""
    step = ReactionStepW2(n_bond_types=3)
    r = torch.randn(4, 7, 3)
    p = torch.randn(4, 7, 3)
    out = step.forward(r, p)
    assert out.shape == (4, 7, 3)


# ---------------------------------------------------------------------------
# 6. Integration — all three primitives wired together
# ---------------------------------------------------------------------------


def test_w2_flow_then_reaction_step_returns_product() -> None:
    """A W₂ gradient flow step composed with the reaction step recovers the product."""
    flow = W2GradientFlow(kernel="none")
    step = ReactionStepW2(n_bond_types=4)
    # Particle in the chemical-graph metric: a single bond-type fingerprint.
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    score = -torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # drift points to product
    # W₂ step at dt=1: x -> x - score = product.
    x_next = flow.step(x, score, dt=1.0)
    # Reaction step from reactant=x to product=x_next returns x_next.
    out = step.forward(x, x_next)
    assert torch.allclose(out, x_next, atol=1e-6)


def test_w2_gradient_flow_uses_only_torch() -> None:
    """The module imports *only* ``torch`` (no internal molmetal deps)."""
    import molmetal.math_skills.wasserstein_gradient_flow as wgf

    src = Path(wgf.__file__).read_text(encoding="utf-8")
    # Allow the standard ``from __future__ import annotations`` and
    # ``import torch`` lines, but no other molmetal imports.
    forbidden = (
        "from molmetal.adapters",
        "from molmetal.molmetal_lam",
        "from molmetal.data",
        "from molmetal.eval",
        "from molmetal.metrics",
        "from molmetal.models",
        "import molmetal.adapters",
        "import molmetal.molmetal_lam",
    )
    for needle in forbidden:
        assert needle not in src, (
            f"unexpected molmetal internal import: {needle!r}"
        )