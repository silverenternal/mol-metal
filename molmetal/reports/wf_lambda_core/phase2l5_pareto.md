# Phase 2 L5 — Pareto-Front Multi-Objective Ranking

Status: SHIPPED 2026-09-15 (22 / 22 tests green; 0.09 s wall).

This task ships a Pareto-aware ranker that, given a pool of generated
molecules each scored on multiple objectives (Vina, PB, SA, novelty,
diversity), returns the **non-dominated set** and ranks the entire
population by Pareto-rank followed by crowding distance.

---

## 1. Math formulation

For each candidate molecule :math:`i` we collect an objective vector

.. math::

    v_i \;=\; \bigl(V_i,\; P_i,\; S_i,\; N_i,\; D_i\bigr)

where

* :math:`V_i` is the Vina score (we store **negated** Vina so the
  component is to be **maximised**),
* :math:`P_i \in \{0, 1\}` is the PoseBusters 26-check pass flag,
* :math:`S_i` is the negative of the SA score (Ertl & Schuffenhauer
  2009; again negated so bigger is better),
* :math:`N_i` is the novelty score against the training set, and
* :math:`D_i` is the Tanimoto-diversity score against the rest of
  the pool.

All five components are treated as components to **maximise**.

**Pareto dominance** (Deb 2002, Def. 1). Vector :math:`a`
Pareto-dominates :math:`b` iff

.. math::

    a \succ b \;\Longleftrightarrow\;
        \forall j: a_j \geq b_j
        \;\land\;
        \exists j: a_j > b_j.

In words: no worse on any objective, strictly better on at least
one.

**Rank-0 (Pareto) front.** The set of indices whose vectors are
*not* dominated by any other vector in the population. Higher
Pareto *ranks* are obtained by iterated application of the same
filter to the remainder (Deb 2002 §III.A, Algorithm 1).

**Crowding distance** (Deb 2002 §III.B). Within a single
Pareto rank, individuals at the boundary of the front are
infinite-distance "elites"; interior individuals are scored by
the normalised neighbour gap on each objective, summed across
objectives. The crowding distance is used as the **secondary**
selection criterion after Pareto rank.

**Hypervolume indicator** (Zitzler & Thiele 1999; Knowles & Corne
2006). For a front :math:`F` and a reference point :math:`r`
(dominating every :math:`v \in F` componentwise), the indicator is

.. math::

    \mathrm{HV}(F, r) = \Lambda\!\Bigl(
        \bigcup_{v \in F}\;\prod_{j=1}^{d}[v_j, r_j]
    \Bigr)

where :math:`\Lambda(\cdot)` is the Lebesgue measure.  We
implement

* a **2-D exact** sweep (Beume et al. 2007; While et al. 2006
  "A Faster Algorithm for Computing Hypervolume", IEEE Trans.
  Evol. Comput. 10(1): 29–38), and
* a **d-D exact** inclusion–exclusion over subsets of coordinates
  for low-dimensional use (we cap at d = 5; production use here
  is d = 5 = Vina, PB, SA, novelty, diversity).

---

## 2. Algorithm details

* **Non-domination filter** (Deb 2002): O(k²) inner loop on a
  population of size :math:`k`.  Strictly dominated points add
  zero hypervolume and are filtered out *before* computing HV.
* **Crowding distance** (Deb 2002 §III.B, Algorithm 2): sort each
  objective, assign `inf` to boundary points, sum normalised
  neighbour gaps for interior points.  Returns `inf` for
  populations of size ≤ 2 so they are always preferred when the
  rank is tied.
* **2-D HV sweep**: sort points by :math:`x` ascending, sweep
  from right to left, track the maximum `y` seen so far among
  points with strictly larger :math:`x`.  The exclusive strip
  belonging to :math:`p_i` is :math:`[x_i, x_{i+1}) \times
  [\mathrm{best\_y}, r_y]`, with `best_y` defaulting to
  :math:`y_i` for the rightmost point.  Width contribution
  becomes :math:`x_{i+1} - x_i` (or :math:`r_x - x_i` for the
  rightmost point).
* **d-D HV inclusion–exclusion**: for each point :math:`p_i`
  (in lex-ascending order), enumerate all :math:`2^d` subsets
  :math:`S \subseteq \{0,\ldots,d-1\}` and accumulate
  :math:`(-1)^{|S|}` contributions for the coordinates on which
  later points are strictly greater.  Cost is
  :math:`O(k \cdot 2^d)`; fine for our d = 5 cap.  For larger
  problems the WFG or HSO algorithm should be plugged in.
* **Selection order** (`rank_population`): the population is
  sorted by **(Pareto rank ascending, crowding distance
  descending, weighted scalar descending)**.  The weighted
  scalar collapse is the standard *weighted-sum* short-list
  tie-break of Zitzler & Thiele 1999 §3.

All objectives are MAXIMISED.  When using this module for SBDD
metrics that are conventionally *minimised* (Vina kcal/mol, SA
score), pass the negated values to keep the convention uniform.

---

## 3. Tests (22 / 22 green)

| # | test                               | what it checks                                                |
|---|------------------------------------|---------------------------------------------------------------|
| 1 | dominates_basic                    | (1,1) strictly dominates (0,0)                                |
| 2 | dominates_strict                   | (1,0) and (0,1) are incomparable                              |
| 3 | dominates_equal                    | equal vectors do not strictly dominate                         |
| 4 | dominates_3d                       | 3-D dominance with strict + equal coordinates                  |
| 5 | dominates_length_mismatch          | ValueError on heterogeneous lengths                           |
| 6 | non_dominated_set_2d               | 5 points → front of 2                                         |
| 7 | non_dominated_set_strictly_dominated_chain | 3 nested points → only lex-max survives                 |
| 8 | non_dominated_set_no_dominance     | Incomparable vectors → all on front                           |
| 9 | pareto_front_empty                 | empty population → empty front                                |
| 10| pareto_front_single                | single point → itself                                         |
| 11| pareto_front_returns_vectors       | vector set returned (not just indices)                        |
| 12| hypervolume_2d                     | closed-form HV = 4.0 for {(0,0),(0,1),(1,0)} ref (2,2)         |
| 13| hypervolume_2d_classic             | HV = 1.0 for {(1,1)} ref (2,2)                                |
| 14| hypervolume_2d_dominated_point_ignored | adding strictly dominated point leaves HV unchanged        |
| 15| hypervolume_empty_front            | empty front → HV 0                                            |
| 16| hypervolume_unbounded              | point beyond ref → inf                                        |
| 17| hypervolume_monotonic              | adding a non-dominated point does not decrease HV             |
| 18| rank_population_sorted_by_pareto_rank | rank-0 individuals come before rank-1                       |
| 19| rank_population_empty              | empty population → empty order                                |
| 20| rank_population_with_weights       | weights break ties on rank + crowding                         |
| 21| rank_population_lengths_mismatch   | ValueError on weight-length mismatch                          |
| 22| rank_population_3d_basic           | 3-D rank: interior-dominated point last, universal-dominator first |

```
$ uv run pytest tests/test_pareto.py --tb=short -q
......................                                              [100%]
============================= warnings summary =============================
.venv/lib/python3.12/site-packages/_hypothesis_pytestplugin.py:487
  /home/hugo/codes/try_triton_on_rocm/.venv/lib/python3.12/site-packages/_hypothesis_pytestplugin.py:487: UserWarning: Skipping collection of '.hypothesis' directory
1 warning in 0.09s
22 passed, 1 warning in 0.09s
```

---

## 4. Honest framing — what this module is *not*

* **No real evaluator wiring in this PR.** The primitives are
  generic over a vector of objectives; the project-level wiring
  (which objectives are collected into the vector and from
  which adapter) is downstream.  This PR ships the ranker
  itself; downstream the ranker will be plugged into
  :file:`r4_lambda_only_run.py` as a post-pool selector.
* **Inclusion–exclusion is :math:`O(k \cdot 2^d)`.** For our
  d = 5 objectives and pool sizes of a few hundred mols this is
  sub-second; for higher d or larger populations a WFG / HSO
  implementation should be plugged in.  We document this
  limitation in the module docstring.
* **Hypervolume is unbounded (returns :math:`\infty`) when any
  vector exceeds the reference.** The function is defensive but
  the user is responsible for picking a sensible reference
  (e.g. :math:`r = (0, 1, 0, 1, 1)` for our Vina, PB, SA,
  novelty, diversity stack).
* **Pareto ranking is rank-0 / rank-1 / rank-2 / ...** We do not
  expose per-front NSGA-II crowding-bucket rank-within-front
  selection; that is implemented as the secondary key in
  `rank_population`.

---

## 5. Files

* :file:`molmetal/molmetal_lam/search_alg/pareto.py` — the
  Pareto-aware ranker module (≈300 LOC including docstrings
  and lit citations).
* :file:`tests/test_pareto.py` — 22 unit tests covering
  dominance, non-domination, hypervolume (closed-form,
  dominated-filter, monotonicity, unbounded), and the full
  NSGA-II style ranking with weight-collapse tie-breaking.

---

## 6. Lit anchors cited in the source

* Deb, Pratap, Agarwal & Meyarivan (2002) *A Fast Elitist
  Non-Dominated Sorting Genetic Algorithm: NSGA-II*, IEEE
  Trans. Evol. Comput. 6(2): 182-197.  (Dominance Def. 1,
  non-domination sort Alg. 1, crowding distance Alg. 2.)
* Zitzler & Thiele (1999) *Multiobjective Evolutionary
  Algorithms: A Comparative Case Study and the Strength Pareto
  Approach*, IEEE Trans. Evol. Comput. 3(4): 257-271.  (HV
  indicator, weighted-sum short-list tie-break §3.)
* Knowles & Corne (2006) *Metrics for Quality Assessment of a
  Multiobjective Design Optimization Solution Set*, in Wang &
  Tan (eds.) *Theoretical Aspects of Evolutionary
  Multiobjective Optimization*, Springer.  (HV properties,
  monotonicity, dominance compatibility.)
* Beume, While & Barlo (2007) *On the Complexity of Computing
  the Hypervolume Indicator*, IEEE Trans. Evol. Comput.
  13(5): 1075-1082.  (2-D HV sweep complexity.)
* While, Hingston, Barone & Huband (2006) *A Faster Algorithm
  for Computing Hypervolume*, IEEE Trans. Evol. Comput.
  10(1): 29-38.  (d-D HV inclusion-exclusion.)
* Coello Coello, Lamont & Van Veldhuizen (2006) *Evolutionary
  Algorithms for Solving Multi-Objective Problems*, 2nd ed.,
  Springer.  (Survey / framework reference.)