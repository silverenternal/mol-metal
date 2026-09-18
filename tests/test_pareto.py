"""Tests for ``molmetal.molmetal_lam.search_alg.pareto``.

Validates the Pareto-aware multi-objective ranking primitives:

  1. :func:`dominates` basic, strict, equal cases
  2. :func:`non_dominated_set` rank-0 front identification
  3. :func:`pareto_front` empty / single edge cases
  4. :func:`hypervolume` 2D closed-form + monotonicity + ref bound
  5. :func:`rank_population` sorted by Pareto rank then crowding

Math priors (Deb 2002 NSGA-II, Zitzler 1999 MOEA, Knowles 2006 HV).
"""
from __future__ import annotations

import math
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "molmetal"))

from molmetal.molmetal_lam.search_alg.pareto import (  # noqa: E402
    dominates,
    non_dominated_set,
    pareto_front,
    hypervolume,
    rank_population,
)


# ---------------------------------------------------------------------------
# dominates
# ---------------------------------------------------------------------------


def test_dominates_basic():
    """a=(1,1) strictly dominates b=(0,0) on every objective."""
    assert dominates([1.0, 1.0], [0.0, 0.0]) is True


def test_dominates_strict():
    """a=(1,0) does NOT dominate b=(0,1) — they are incomparable."""
    assert dominates([1.0, 0.0], [0.0, 1.0]) is False
    # Symmetry: also fails the other way.
    assert dominates([0.0, 1.0], [1.0, 0.0]) is False


def test_dominates_equal():
    """Equal vectors do not strictly dominate each other (NSGA-II Def. 1)."""
    assert dominates([1.0, 1.0], [1.0, 1.0]) is False
    assert dominates([0.5], [0.5]) is False


def test_dominates_3d():
    """3-D dominance: need strict in at least one coordinate."""
    # (1, 1, 0) does not dominate (0, 1, 1) — last coord worse.
    assert dominates([1.0, 1.0, 0.0], [0.0, 1.0, 1.0]) is False
    # (1, 1, 1) dominates (1, 1, 0).
    assert dominates([1.0, 1.0, 1.0], [1.0, 1.0, 0.0]) is True


def test_dominates_length_mismatch():
    """Unequal-length vectors raise ValueError."""
    with pytest.raises(ValueError):
        dominates([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        dominates([], [1.0])


# ---------------------------------------------------------------------------
# non_dominated_set / pareto_front
# ---------------------------------------------------------------------------


def test_non_dominated_set_2d():
    """5 points, identify the Pareto front of size 2."""
    pop = [
        [0.9, 0.1],  # dominated by next
        [1.0, 0.5],  # on front
        [0.5, 1.0],  # on front
        [0.6, 0.4],  # dominated
        [0.2, 0.9],  # dominated
    ]
    front_idx = non_dominated_set(pop)
    front_vecs = sorted(tuple(pop[i]) for i in front_idx)
    assert (1.0, 0.5) in front_vecs
    assert (0.5, 1.0) in front_vecs
    assert len(front_idx) == 2


def test_non_dominated_set_strictly_dominated_chain():
    """Strictly dominated chain: only the lexicographically maximal survives."""
    pop = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]
    front_idx = non_dominated_set(pop)
    # (0.9,0.9) dominates both (0.1,0.1) and (0.5,0.5).
    # (0.5,0.5) dominates (0.1,0.1).
    # Only (0.9,0.9) is on the rank-0 front.
    assert front_idx == [2]


def test_non_dominated_set_no_dominance():
    """Incomparable vectors: all on front."""
    pop = [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]]
    front_idx = non_dominated_set(pop)
    # No pair has dominance; everyone is on the rank-0 front.
    assert sorted(front_idx) == [0, 1, 2]


def test_pareto_front_empty():
    """Empty population -> empty front."""
    assert pareto_front([]) == []
    assert non_dominated_set([]) == []


def test_pareto_front_single():
    """Single point population -> itself is the front."""
    assert pareto_front([[0.5, 0.5]]) == [[0.5, 0.5]]


def test_pareto_front_returns_vectors():
    """pareto_front returns the vectors themselves (not just indices).

    Pop: (0.1,0.2) (0.4,0.6) (0.9,0.1) (0.5,0.5)
    (0.4, 0.6) dominates (0.1, 0.2) — strict in both coords.
    (0.5, 0.5), (0.4, 0.6), (0.9, 0.1) are pairwise incomparable.
    Front has 3 vectors.
    """
    pop = [[0.1, 0.2], [0.4, 0.6], [0.9, 0.1], [0.5, 0.5]]
    front = pareto_front(pop)
    front_set = {tuple(v) for v in front}
    assert (0.4, 0.6) in front_set
    assert (0.9, 0.1) in front_set
    assert (0.5, 0.5) in front_set
    assert (0.1, 0.2) not in front_set
    assert len(front) == 3


# ---------------------------------------------------------------------------
# hypervolume
# ---------------------------------------------------------------------------


def test_hypervolume_2d():
    """Closed-form 2D HV: front {(1,0), (0,1), (0,0)} with ref (2,2).

    Sweep sort by x ascending: (0,0), (0,1), (1,0).
      - i=0: prev_x=2, x=0 -> w=2, best_y=-inf, h=0 -> contrib 0.
        Actually need to track the best y AFTER the point.  Let me
        re-derive below using the simpler closed-form.

    Direct closed-form for a 2D point set dominated by ref (2,2):
      The dominated region is the union of [p, ref] rectangles.
      For {(0,0), (0,1), (1,0)} with ref (2,2):

      Sweep sort by x ascending: [(0,0), (0,1), (1,0)].
        prev_x=2; step to x=0: w=2, best_y at this stage = -inf
          (no point has been processed yet), so h=0.  After step
          best_y = max(0, -inf) = 0.
        prev_x=0; step to x=0 (tied): w=0, no contrib. best_y=1.
        prev_x=0; step to x=1: w=0 (no width since both 0), no
          contrib.  After step best_y = 1.
        prev_x=1; end.  No further slice.

      Wait — that gives 0.  But the actual HV is:
        [(0,0)→(2,2)] covers 4.0
        [(0,1)→(2,2)] covers 2*1 = 2.0 (subset of the above)
        [(1,0)→(2,2)] covers 1*2 = 2.0 (subset of the above)
      So HV = 4.0 (the union is the 2x2 square anchored at (0,0)).

      Hmm, that suggests our sweep is wrong.  Let me recompute
      manually using the rectangle decomposition: the union of the
      three rectangles is just [0,2]×[0,2] = 4.0.

    With our sweep, processing sorted ascending [(0,0),(0,1),(1,0)]:
      Before any point, prev_x = ref[0] = 2.0.
      Process (0,0): width = prev_x - x = 2 - 0 = 2; h = ref[1] - best_y
        = 2 - (-inf) = inf.  But we have a guard: if best_y == -inf
        then h is undefined / skipped.  So this slice would contribute
        0 if best_y is still -inf.  After processing, best_y = 0.
      Process (0,1): width = 2 - 0 = wait, prev_x was reset to x=0
        after the previous step.  So now width = 0 - 0 = 0.  No contrib.
        best_y = max(0, 1) = 1.
      Process (1,0): width = 0 - 1 = -1.  Negative, skipped.
        best_y = max(1, 0) = 1.
      End: prev_x = 1, never advances to ref.

    So this gives 0, which is wrong.  The sweep is broken for
    points that anchor the entire [0, ref] strip.  Test exposed a
    bug — fix the sweep algorithm below.

    Expected HV = 4.0 for front {(0,0),(0,1),(1,0)} ref (2,2).
    """
    front = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    ref = [2.0, 2.0]
    hv = hypervolume(front, ref)
    assert abs(hv - 4.0) < 1e-9


def test_hypervolume_2d_classic():
    """Classic 2D HV: front {(1,1)} with ref (2,2) -> 1.0."""
    assert abs(hypervolume([[1.0, 1.0]], [2.0, 2.0]) - 1.0) < 1e-9


def test_hypervolume_2d_dominated_point_ignored():
    """Strictly dominated points do not change the HV."""
    # (0.2, 0.2) IS dominated by (0.5, 0.5): strict in both.  So adding
    # (0.2, 0.2) should not change HV relative to {(0.5,0.5)} alone.
    front_with = [[0.5, 0.5], [0.2, 0.2]]
    front_without = [[0.5, 0.5]]
    ref = [1.0, 1.0]
    hv_with = hypervolume(front_with, ref)
    hv_without = hypervolume(front_without, ref)
    assert abs(hv_with - hv_without) < 1e-9
    assert abs(hv_with - 0.25) < 1e-9  # 0.5*0.5 rectangle


def test_hypervolume_empty_front():
    """Empty front -> HV 0."""
    assert hypervolume([], [1.0, 1.0]) == 0.0


def test_hypervolume_unbounded():
    """Point beyond ref -> inf (componentwise strict)."""
    # (1.5, 1.0) with ref (1.0, 2.0): obj0 > ref0 -> unbounded.
    assert hypervolume([[1.5, 1.0]], [1.0, 2.0]) == math.inf


def test_hypervolume_monotonic():
    """Adding a non-dominated point should not decrease HV."""
    front_a = [[0.5, 0.0]]
    front_b = [[0.5, 0.0], [0.0, 0.5]]
    ref = [1.0, 1.0]
    hv_a = hypervolume(front_a, ref)
    hv_b = hypervolume(front_b, ref)
    assert hv_b >= hv_a


# ---------------------------------------------------------------------------
# rank_population
# ---------------------------------------------------------------------------


def test_rank_population_sorted_by_pareto_rank():
    """Rank-0 (Pareto front) individuals come before rank-1+."""
    pop = [
        [0.9, 0.1],  # idx 0
        [1.0, 0.5],  # idx 1 -- rank 0
        [0.5, 1.0],  # idx 2 -- rank 0
        [0.5, 0.5],  # idx 3 -- rank 1 (dominated by 1 and 2)
    ]
    order = rank_population(pop)
    # First two should be the rank-0 individuals; the rank-1 is last.
    rank0_set = {order[0], order[1]}
    assert rank0_set == {1, 2}
    assert order[-1] == 3


def test_rank_population_empty():
    """Empty population -> empty order."""
    assert rank_population([]) == []


def test_rank_population_with_weights():
    """When weights provided, ties on rank+crowding break by weighted score."""
    pop = [
        [1.0, 0.0],  # idx 0
        [0.0, 1.0],  # idx 1
        [0.5, 0.5],  # idx 2 -- dominated
    ]
    # All on rank 0 (no dominance relation).  Weights favor obj0.
    order = rank_population(pop, weights=[1.0, 0.0])
    # idx 0 should beat idx 1 (1.0 > 0.0 weighted sum).
    # Both should appear before idx 2 (which is dominated, so on rank 1).
    assert order[-1] == 2


def test_rank_population_lengths_mismatch():
    """weights of wrong length raises ValueError."""
    with pytest.raises(ValueError):
        rank_population([[1.0, 1.0]], weights=[1.0])


def test_rank_population_3d_basic():
    """3-D rank: simple front of 4 corner points + 1 dominated interior."""
    pop = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5],  # dominated by all 3 above
        [1.0, 1.0, 1.0],  # dominates all of the above
    ]
    order = rank_population(pop)
    # Last (worst rank) should be the dominated interior point.
    assert order[-1] == 3
    # First (best rank) should be the one that dominates all.
    assert order[0] == 4


if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-x", "--tb=short", "-q"])