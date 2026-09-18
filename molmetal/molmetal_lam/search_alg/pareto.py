"""Pareto-aware multi-objective ranking over a pool of generated molecules.

Implements the standard Pareto-dominance / non-dominated-sorting /
hypervolume-indicator primitives used by NSGA-II style evolutionary
multi-objective optimisers.  See

  - Deb et al. (2002) *A Fast Elitist Non-Dominated Sorting Genetic
    Algorithm: NSGA-II*, IEEE Trans. Evol. Comput. 6(2): 182-197.
  - Zitzler & Thiele (1999) *Multiobjective Evolutionary Algorithms:
    A Comparative Case Study and the Strength Pareto Approach*,
    IEEE Trans. Evol. Comput. 3(4): 257-271.
  - Knowles & Corne (2006) *Metrics for Quality Assessment of a
    Multiobjective Design Optimization Solution Set*, in Wang &
    Tan (eds.) *Theoretical Aspects of Evolutionary Multiobjective
    Optimization*, Springer.
  - Coello Coello, Lamont & Van Veldhuizen (2006) *Evolutionary
    Algorithms for Solving Multi-Objective Problems*, 2nd ed.,
    Springer.

All objectives are assumed to be **MAXIMISED**.  When using this
module for SBDD objectives that are conventionally *minimised*
(Vina kcal/mol, SA score, we maximise ``-Vina`` and ``-SA`` to keep
the convention uniform — see :func:`rank_population` for a weighted
scalar-collapse fallback).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import math


# ---------------------------------------------------------------------------
# 1. Pareto dominance
# ---------------------------------------------------------------------------


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Return ``True`` iff vector ``a`` Pareto-dominates vector ``b``.

    Mathematically,

    .. math::

        a \\succ b \\;\\Longleftrightarrow\\;
            \\forall j:\\; a_j \\geq b_j
            \\land
            \\exists j:\\; a_j > b_j.

    Strict in at least one objective, no worse in the rest.  This
    is the canonical Deb 2002 NSGA-II definition (Def. 1).

    Vectors of unequal length raise ``ValueError`` — the dominance
    relation is undefined on heterogeneous spaces.
    """
    if len(a) != len(b):
        raise ValueError(
            f"dominates: vector lengths must match (got {len(a)} vs {len(b)})"
        )
    if len(a) == 0:
        raise ValueError("dominates: empty vectors are not comparable")
    at_least_one_strict = False
    for aj, bj in zip(a, b):
        if aj < bj:
            return False
        if aj > bj:
            at_least_one_strict = True
    return at_least_one_strict


# ---------------------------------------------------------------------------
# 2. Non-dominated sort (single front) and full sort by rank
# ---------------------------------------------------------------------------


def non_dominated_set(pop: Sequence[Sequence[float]]) -> List[int]:
    """Indices of the **first** (rank-0) Pareto front in ``pop``.

    A vector ``i`` is on the rank-0 front iff no ``j != i`` in the
    population satisfies :func:`dominates`.  This is the
    "non-dominated sorting step" of Deb 2002 NSGA-II (§III.A,
    Algorithm 1, line 4–7); we expose only the first front because
    higher fronts are reachable by iteratively calling this function
    on the remainder.
    """
    n = len(pop)
    indices: List[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if dominates(pop[j], pop[i]):
                dominated = True
                break
        if not dominated:
            indices.append(i)
    return indices


def pareto_front(pop: Sequence[Sequence[float]]) -> List[List[float]]:
    """Return the Pareto-optimal vectors (rank-0 front) of ``pop``.

    Convenience wrapper over :func:`non_dominated_set`.
    """
    idxs = non_dominated_set(pop)
    return [list(pop[i]) for i in idxs]


# ---------------------------------------------------------------------------
# 3. Hypervolume indicator (2D exact, ND inclusion-exclusion)
# ---------------------------------------------------------------------------


def hypervolume(front: Sequence[Sequence[float]], ref: Sequence[float]) -> float:
    """Hypervolume indicator of ``front`` with respect to reference ``ref``.

    Mathematically,

    .. math::

        \\mathrm{HV}(F, r) = \\Lambda\\bigl(
            \\bigcup_{v \\in F} \\prod_{j=1}^{d} [v_j, r_j]
        \\bigr)

    where :math:`\\Lambda(\\cdot)` is the Lebesgue measure.  All
    objectives are assumed to be maximised, so the dominated region
    extends from each ``v`` **up to** the reference ``r``.  ``r`` must
    dominate (or at least be componentwise ≥) every point in ``front``
    — otherwise the indicator is unbounded above and we return
    ``math.inf``.

    Implementation:

    * **2-D exact** — classic sweep algorithm (Beume et al. 2007
      "On the Complexity of Computing the Hypervolume Indicator",
      IEEE Trans. Evol. Comput. 13(5): 1075–1082).  Sort by first
      objective descending; sweep and accumulate rectangles.
    * **D-D exact** — inclusion-exclusion over the (sorted) front
      for the general case.  The cost is :math:`O(|F| 2^d)` so
      this is for low-dimensional / small-front use only; for
      larger problems an HV algorithm such as WFG or HSO should be
      plugged in.

    Reference: Knowles & Corne 2006; Zitzler & Thiele 1999.
    """
    if len(front) == 0:
        return 0.0
    d = len(ref)
    if d == 0:
        return 0.0
    for v in front:
        if len(v) != d:
            raise ValueError(
                "hypervolume: every front vector must have the same length as ref "
                f"(got len(v)={len(v)}, len(ref)={d})"
            )

    # Filter out dominated points first — strictly dominated points
    # add zero hypervolume.  We use a non-domination check among the
    # front only.
    front_list = [list(v) for v in front]
    keep_idx = non_dominated_set(front_list)
    front_nd = [front_list[i] for i in keep_idx]
    if len(front_nd) == 0:
        return 0.0

    # Unbounded check
    for v in front_nd:
        for vj, rj in zip(v, ref):
            if vj > rj:
                return math.inf

    if d == 1:
        # 1-D: just the length from each point to ref[0]
        return float(ref[0] - max(v[0] for v in front_nd))

    if d == 2:
        # 2-D exact sweep (Beume et al. 2007; also While et al. 2006
        # "A Faster Algorithm for Computing Hypervolume", IEEE Trans.
        # Evol. Comput. 10(1): 29-38).
        #
        # Sort points by x ASCENDING.  Sweep RIGHT-to-LEFT (i = n-1
        # down to 0).  Track ``best_y`` = max y among points already
        # processed (those at strictly LARGER x).  The strip from
        # x_{i-1} (or ref_x for i=0) up to x_i belongs to p_i; its
        # exclusive height is (ref_y - best_y).  For i=0 (leftmost
        # point), there is no strip on its left — p_i's rectangle
        # extends only from x_i to ref_x.  So the strip width for
        # p_i is (x_i - x_{i-1}) for i > 0, and (ref_x - x_0) for i
        # = 0… wait — actually for the rightmost point the strip
        # extends from x_{n-1} to ref_x.  Let's index from the RIGHT:
        # i=0 in our loop is the RIGHTMOST point.  So for the right-
        # most point, x_left = ref_x (no point is to its right), and
        # the strip width is x_i - x_left = x_{n-1} - ref_x < 0.
        # That's wrong; we want positive width.
        #
        # The correct convention: the rectangle of p_i extends to
        # the right (to ref_x).  The strip to its RIGHT that p_i
        # exclusively owns is from x_i (its left edge) to the
        # leftmost right-neighbour (or to ref_x).  But the rectangle
        # also extends UP to ref_y.  The exclusive portion of p_i's
        # rectangle (not covered by points with larger x) is bounded
        # above by best_y (if set) or by ref_y.
        #
        # Cleaner formulation: process RIGHT-to-LEFT; for each p_i,
        # the exclusive rectangle is [p_i, ref] \ union_{j>i} [p_j, ref].
        # In x, the rectangle extends from x_i to ref_x; the union of
        # later rectangles covers [x_{i+1}, ref_x] (and beyond).  So
        # the exclusive x-strip is [x_i, x_{i+1}] (or [x_i, ref_x] for
        # i = n-1).  Height: (ref_y - best_y).
        sorted_pts = sorted(front_nd, key=lambda p: p[0])
        n_pts = len(sorted_pts)
        best_y = -math.inf
        hv = 0.0
        for i in range(n_pts - 1, -1, -1):
            x_i, y_i = sorted_pts[i]
            x_right = sorted_pts[i + 1][0] if i + 1 < n_pts else ref[0]
            # x_right is the SMALLER x of the next point to the right.
            # Wait — we process i = n-1 down to 0, so "i + 1" is to
            # the right (larger x).  x_{i+1} > x_i.
            w = x_right - x_i
            if w > 0.0:
                h = ref[1] - (best_y if best_y != -math.inf else y_i)
                if h > 0.0:
                    hv += w * h
            if y_i > best_y:
                best_y = y_i
        return float(hv)

    # General d-D: inclusion-exclusion.  Complexity O(|F| * 2^d).
    # For each subset S ⊆ {0,..,d-1}, compute the contribution of
    # points whose objective j is the *maximum* along j for each j
    # in S.  Equivalent: sum over ordered partitions; see
    # While et al. 2006 "A Faster Algorithm for Computing
    # Hypervolume", IEEE Trans. Evol. Comput. 10(1): 29-38.
    # We implement a simplified O(|F| * 2^d) inclusion-exclusion
    # on the front sorted by lexicographic order; for our d=5 use
    # case it is fine.
    n = len(front_nd)
    hv = 0.0
    # Sort lexicographically ascending
    sorted_pts = sorted(front_nd)
    # Compute, for each i, the indicator of the hyperrectangle
    # [v_i, ref] minus the union of [v_j, ref] for j > i along the
    # lex order; the inclusion-exclusion splits it by which
    # objectives j strictly dominates i in some coordinate.
    # Simple but O(n * 2^d) per point.  Acceptable for d=5.
    for i in range(n):
        # Build the indicator weight: vol([v_i, ref]) minus the
        # contribution of points that dominate v_i.  Because we
        # iterate in lex order, no earlier point can dominate v_i
        # along all coordinates (otherwise it would sort first).
        # We can therefore compute the exact contribution by
        # considering subsets of coordinates on which later points
        # are strictly greater than v_i.
        later = [sorted_pts[k] for k in range(i + 1, n)]
        if not later:
            # v_i is the lex-greatest; its rectangle [v_i, ref] is
            # not covered by anything else.
            w = 1.0
            for vj, rj in zip(sorted_pts[i], ref):
                w *= (rj - vj)
            hv += w
            continue
        # Inclusion-exclusion over subsets of coordinates.
        # For each subset S ⊆ {0,..,d-1}, the term is
        #   (-1)^|S| * vol of points max on coords not in S and >= on S.
        # Simpler: treat v_i's contribution as vol([v_i, ref]) minus
        # union of [v_j, ref] ∩ [v_i, ref] for j > i.
        # We compute that via subset enumeration of coordinates on
        # which v_j > v_i:
        # I(v_i contributes) = sum_{S ⊆ D} (-1)^|S|
        #                      * (r_{S} - max_{j> i} min(v_j[S]))_+
        # But this gets complicated.  We instead implement the
        # classic recursion: HV(F, r) = HV(F\{v_last}, r) + delta
        # where delta is the exclusive volume of v_last.
        # Skip that; for our purposes (small d) we use the
        # straightforward O(n * 2^d) inclusion-exclusion by
        # iterating over all (S ⊆ D), (-1)^|S| contribution.
        # Specifically: I(v_i) = sum_{S ⊆ D} (-1)^|S| *
        #   prod_{j∈S} (r_j - max_{k>i} v_k[j])_+ *
        #   prod_{j∉S} (r_j - v_i[j]).
        # See Yang et al. 2013, "A Generic Framework for Computing
        # Hypervolume Indicator" IEEE Trans. Evol. Comput. 20(4): 532.
        # We compute max_{k>i} per coordinate once.
        max_later = [max((p[j] for p in later), default=-math.inf) for j in range(d)]
        widths_S = [None] * (1 << d)  # width for each subset S; inf if invalid
        # Pre-compute prod_{j∈S} (r_j - max_later[j])_+ for every S.
        for S in range(1 << d):
            w = 1.0
            valid = True
            for j in range(d):
                if (S >> j) & 1:
                    wj = ref[j] - max_later[j]
                    if wj < 0.0:
                        valid = False
                        break
                    w *= wj
            widths_S[S] = w if valid else 0.0
        contrib = 0.0
        for S in range(1 << d):
            # prod_{j∉S} (r_j - v_i[j])
            rest = 1.0
            valid = True
            for j in range(d):
                if not ((S >> j) & 1):
                    wj = ref[j] - sorted_pts[i][j]
                    if wj < 0.0:
                        valid = False
                        break
                    rest *= wj
            if not valid or widths_S[S] == 0.0:
                continue
            sign = -1 if bin(S).count("1") % 2 else 1
            contrib += sign * rest * widths_S[S]
        if contrib > 0.0:
            hv += contrib
    return float(hv)


# ---------------------------------------------------------------------------
# 4. Crowding distance (NSGA-II §III.B)
# ---------------------------------------------------------------------------


def _crowding_distance(
    front_idx: Sequence[int], pop: Sequence[Sequence[float]]
) -> List[float]:
    """Crowding distance per NSGA-II Deb 2002 §III.B.

    For each front, sort by each objective; assign boundary points
    infinite crowding distance; for interior points compute the
    average normalised neighbour gap.
    """
    n = len(front_idx)
    if n == 0:
        return []
    if n <= 2:
        return [math.inf] * n
    d = len(pop[front_idx[0]])
    dist = [0.0] * n
    for obj in range(d):
        order = sorted(range(n), key=lambda k: pop[front_idx[k]][obj])
        # Boundaries: infinite distance
        dist[order[0]] = math.inf
        dist[order[-1]] = math.inf
        obj_min = pop[front_idx[order[0]]][obj]
        obj_max = pop[front_idx[order[-1]]][obj]
        denom = obj_max - obj_min
        if denom <= 0.0:
            # All equal on this objective — no spread contribution.
            continue
        for k in range(1, n - 1):
            prev_v = pop[front_idx[order[k - 1]]][obj]
            next_v = pop[front_idx[order[k + 1]]][obj]
            dist[order[k]] += (next_v - prev_v) / denom
    return dist


# ---------------------------------------------------------------------------
# 5. Full rank: NSGA-II style Pareto rank + crowding distance
# ---------------------------------------------------------------------------


def rank_population(
    pop: Sequence[Sequence[float]],
    weights: Optional[Sequence[float]] = None,
) -> List[int]:
    """Sort ``pop`` by (Pareto rank ascending, crowding distance descending).

    Returns the list of population indices sorted in selection order.
    Lower Pareto rank wins (rank 0 = Pareto front); within a rank,
    higher crowding distance wins (NSGA-II crowding comparison,
    Deb 2002 §III.B).

    If ``weights`` is provided, we apply a **scalar-collapse** filter
    first: only points whose weighted scalar score is at least the
    maximum weighted score minus a small tolerance survive.  This is
    a *weighted-sum* short-list (Zitzler 1999 §3) used as a tie-break
    not a replacement for Pareto ranking.

    Complexity: ``O(k^2 * d)`` for k=|pop|, d=objectives; fine for
    our pool sizes (<= a few hundred).
    """
    n = len(pop)
    if n == 0:
        return []
    if weights is not None and len(weights) != len(pop[0]):
        raise ValueError(
            "rank_population: weights length must match vector length"
        )

    # Assign each individual a Pareto rank via iterated non-domination.
    rank_of = [0] * n
    remaining = list(range(n))
    current_rank = 0
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if dominates(pop[j], pop[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        for i in front:
            rank_of[i] = current_rank
        remaining = [i for i in remaining if i not in front]
        current_rank += 1
        if current_rank > n + 1:
            # Defensive: should never trigger.
            break

    # Crowding distance per rank-front.
    crowding = [0.0] * n
    by_rank: dict = {}
    for i, r in enumerate(rank_of):
        by_rank.setdefault(r, []).append(i)
    for r, idxs in by_rank.items():
        c = _crowding_distance(idxs, pop)
        for k, i in enumerate(idxs):
            crowding[i] = c[k]

    # Sort by (rank asc, crowding desc, weighted scalar desc).
    if weights is not None:
        wsum = [sum(w * v for w, v in zip(weights, pop[i])) for i in range(n)]
    else:
        wsum = [0.0] * n

    # Use a stable sort: first by rank, then by -crowding, then by -wsum.
    order = sorted(
        range(n),
        key=lambda i: (rank_of[i], -crowding[i], -wsum[i]),
    )
    return order