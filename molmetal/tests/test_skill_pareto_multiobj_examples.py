"""Regression test — verifies the pareto-multiobj SKILL.md API references.

Background
----------
Wf-A44-J19-pilot (2026-10-15): the docs-vs-code-drift guard pattern from
Wf-A43-fixup-3 is being extended to 3 pilot skills before scaling to the
remaining 26 skills. This file guards `mol-metal-pareto-multiobj`.

The pareto-multiobj SKILL.md §10.2 "Existing Pareto code (Block B
foundation)" claims `molmetal/molmetal_lam/search_alg/pareto.py` ships
"440 LOC NSGA-II primitives: ``dominates``, ``non_dominated_set``,
``pareto_front``, ``hypervolume``, ``_crowding_distance``,
``rank_population``".

On-disk reality (verified 2026-10-15):

  - All 6 names are exposed on the module.
  - ``dominates(a, b)`` is the canonical Deb 2002 NSGA-II Def. 1
    (maximise-all convention; ``a`` dominates ``b`` iff
    ``a[i] >= b[i] for all i`` AND ``a[i] > b[i]`` for some i).
  - ``hypervolume(front, ref)`` is in [0, 1]^d; reference point must
    dominate every point in ``front`` (else ``math.inf`` returned).
  - ``rank_population(pop, weights=None)`` returns indices sorted by
    (Pareto rank ascending, crowding distance descending).

This test pins the §3.2 worked example (the 3-drug A/B/C dominance
table) and the 6 top-level names.
"""

from __future__ import annotations

import math as _math

import pytest

from molmetal.molmetal_lam.search_alg.pareto import (
    dominates,
    non_dominated_set,
    pareto_front,
    hypervolume,
    rank_population,
    _crowding_distance,
)


# === §3.2 worked example — 3-drug dominance table ===

def test_dominates_a_over_c():
    """SKILL.md §3.2: A=[0.7,0.5,0.2] dominates C=[0.3,0.5,0.2]
    (m18 + ADMET maximise, toxicity minimise).
    """
    A = [0.7, 0.5, 0.2]
    C = [0.3, 0.5, 0.2]
    assert dominates(A, C) is True
    assert dominates(C, A) is False


def test_dominates_a_and_b_incomparable():
    """SKILL.md §3.2: A vs B — A higher m18, B higher ADMET, toxicity tied.
    Neither dominates the other.
    """
    A = [0.7, 0.5, 0.2]
    B = [0.5, 0.7, 0.2]
    assert dominates(A, B) is False
    assert dominates(B, A) is False


def test_non_dominated_set_abc_returns_ab():
    """SKILL.md §3.2: rank-0 front on {A, B, C} = {A, B} (C dominated
    by both A and B). The function returns the *indices* of the rank-0
    points.
    """
    A = [0.7, 0.5, 0.2]
    B = [0.5, 0.7, 0.2]
    C = [0.3, 0.5, 0.2]
    front = non_dominated_set([A, B, C])
    # Sort for stable comparison
    assert sorted(front) == [0, 1], f"rank-0 must be {{A, B}} = indices [0, 1]; got {front}"


# === hypervolume ===

def test_hypervolume_single_point():
    """Single-point front: HV = product of (ref_i - point_i) over i."""
    front = [[0.0, 0.0]]
    ref = [1.0, 1.0]
    hv = hypervolume(front, ref)
    assert hv == pytest.approx(1.0, abs=1e-9)


def test_hypervolume_two_point_dominated():
    """If one point dominates the other, HV = product of (ref - dom_point)
    only (the dominated point adds zero new region).
    """
    front = [[0.0, 0.0], [0.5, 0.5]]
    ref = [1.0, 1.0]
    hv = hypervolume(front, ref)
    # dom = [0.5, 0.5] → region = (1-0.5)*(1-0.5) = 0.25
    assert hv == pytest.approx(0.25, abs=1e-9)


def test_hypervolume_unbounded_returns_inf():
    """SKILL.md §3.2 + docs: HV is inf if ref is componentwise < a
    front point (the dominated region extends unboundedly).
    """
    front = [[2.0, 2.0]]
    ref = [1.0, 1.0]
    assert hypervolume(front, ref) == _math.inf


def test_hypervolume_3d_basic():
    """3-objective HV — the realistic R22-P3 dimension (per SKILL.md §2)."""
    front = [[0.0, 0.0, 0.0]]
    ref = [1.0, 1.0, 1.0]
    hv = hypervolume(front, ref)
    assert hv == pytest.approx(1.0, abs=1e-9)


# === pareto_front — iterative rank extraction ===

def test_pareto_front_returns_rank_zero_only():
    """pareto_front(pop) returns the rank-0 POINTS only (NOT iterative layers).

    This is a convenience wrapper over ``non_dominated_set`` that swaps
    indices for points. To get iterative layers, use ``pareto_front``
    iteratively until the population is empty (NOT what this function does).
    """
    A = [0.7, 0.5, 0.2]
    B = [0.5, 0.7, 0.2]
    C = [0.3, 0.5, 0.2]
    front = pareto_front([A, B, C])
    # front = rank-0 = {A, B}
    assert len(front) == 2, f"rank-0 front must have 2 elements; got {len(front)}"
    front_set = {tuple(p) for p in front}
    assert front_set == {(0.7, 0.5, 0.2), (0.5, 0.7, 0.2)}, (
        f"rank-0 must be {{A, B}}; got {front}"
    )
    # C must NOT be in the rank-0 front (it is rank-1, dominated by both)
    assert (0.3, 0.5, 0.2) not in front_set


# === rank_population ===

def test_rank_population_returns_indices():
    """rank_population returns indices sorted by (rank ascending, crowding desc)."""
    A = [0.7, 0.5, 0.2]
    B = [0.5, 0.7, 0.2]
    C = [0.3, 0.5, 0.2]
    indices = rank_population([A, B, C])
    # rank 0 = {A, B} (sorted first); rank 1 = {C}
    assert len(indices) == 3
    # C must come last (rank 1 < rank 0 in selection order? — actually
    # rank_population returns indices in selection order: lower rank first,
    # so C should be at index 2).
    assert indices[-1] == 2, (
        f"rank-1 point C (index 2) must come last in selection order; got {indices}"
    )


def test_rank_population_with_weights():
    """weights arg enables the weighted-sum scalar-collapse tie-break."""
    A = [0.7, 0.5, 0.2]
    B = [0.5, 0.7, 0.2]
    C = [0.3, 0.5, 0.2]
    indices = rank_population([A, B, C], weights=[1.0, 1.0, 1.0])
    assert len(indices) == 3


# === _crowding_distance ===

def test_crowding_distance_exposed():
    """SKILL.md §10.2 claims ``_crowding_distance`` is exposed."""
    assert callable(_crowding_distance)


def test_crowding_distance_two_point_front():
    """Two-point front: each gets +inf crowding distance (boundary).

    Signature: _crowding_distance(front_idx, pop) — takes the indices
    of the front-layer points and the full population, returns the
    crowding distance per front-slot.
    """
    pop = [[0.0, 0.0], [1.0, 1.0]]
    cd = _crowding_distance([0, 1], pop)
    assert len(cd) == 2
    # Boundary points get +inf
    assert cd[0] == _math.inf
    assert cd[1] == _math.inf


def test_crowding_distance_three_point_front():
    """Three-point front: middle point gets finite crowding distance;
    boundary points get +inf.
    """
    pop = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    cd = _crowding_distance([0, 1, 2], pop)
    assert len(cd) == 3
    assert cd[0] == _math.inf
    assert cd[2] == _math.inf
    # Middle is finite
    assert _math.isfinite(cd[1])


# === Negative: stale names should NOT silently come back ===

def test_no_three_objective_helper_at_top_level():
    """SKILL.md §3.1 sometimes refers to a ``truncate_to_3d`` helper
    in the cross-skill audit. On disk, pareto.py is dimension-agnostic;
    only ``dominates`` / ``hypervolume`` / ``non_dominated_set`` /
    ``pareto_front`` / ``rank_population`` / ``_crowding_distance`` are
    exposed.
    """
    import molmetal.molmetal_lam.search_alg.pareto as p
    for stale in ("truncate_to_3d", "nsga3_pareto", "nsga3_select",
                  "mcts_nsga3_bridge"):
        assert not hasattr(p, stale), (
            f"stale name `{stale}` on pareto.py — these belong to "
            f"Wf-77 P3.2/P3.3 nsga3_pareto.py / mcts_nsga3_bridge.py, "
            f"NOT to the 440-LOC NSGA-II primitives block"
        )


def test_no_pareto_rank_top_k_flag_in_pareto_module():
    """SKILL.md §2 + §R20 cross-ref: ``pareto_rank_top_k`` is on
    ``proof_search.ParetoRankerConfig``, NOT on pareto.py. Make sure
    no one has accidentally re-exported it here.
    """
    import molmetal.molmetal_lam.search_alg.pareto as p
    assert not hasattr(p, "pareto_rank_top_k"), (
        "`pareto_rank_top_k` belongs to proof_search.ParetoRankerConfig, "
        "not to pareto.py"
    )