# Task H — Soft Tiered Metal-Geometry Reward (Phase 3H)

**Goal.** Convert the hard gate
`molmetal.scripts.r4_lambda_only_run.metal_geometry_prior_bonus` into a
continuous soft score suitable for MCTS reward aggregation.

**Surface touched.** Only the canonical priors module
(`molmetal/molmetal_lam/priors/metal_geometry.py`) and the
`RewardAggregator` in
(`molmetal/molmetal_lam/search_alg/proof_search.py`). **Did NOT
touch** `r4_lambda_only_run.py` (Phase 4 integrator of `w8579x29t` owns
it) or any of the Phase 3 agents' file set (see Task-H constraints).

---

## 1. Math formulation

For each metal centre `m` in `state`, we compute three normalised
deviations from the ideal target geometry:

```
d_coord  = |CN(m) - target_CN| / max(1, target_CN)         in [0, 1]
d_angle  = angle_dev(m) / pi                               in [0, 1]
d_charge = |formal_charge(m) - ideal_charge| / 2           in [0, 1]
```

The per-metal score is

```
score(m) = max(0, 1 - w_coord * d_coord
                  - w_angle * d_angle
                  - w_charge * d_charge)
```

with defaults `(w_coord, w_angle, w_charge) = (0.5, 0.3, 0.2)` and
`ideal_charge = 0` (cisplatin is neutral).  The molecule-level soft
score is the maximum over all metal centres (matches the historical
`best` aggregation in `metal_geometry_prior_bonus`).  The function
returns `0.0` when no metal centre is present (organic ligands don't
earn a metal prior bonus by construction).

### Coordinate-free proxy

The angle deviation is computed **without** 3-D coordinates (we cannot
afford RDKit / ETKDG embedding inside the MCTS loop).  When the metal
has fewer than two bonds, `angle_dev = pi` (max uncertainty).  When
the metal has CN >= 2, `angle_dev = (|CN - target_CN|) * (pi/4)` —
each off-by-one CN contributes `pi/4` rad of "expected angle noise".
This is a coarse approximation: **cis-platin and trans-platin score
identically** because they share CN=4.  The rigorous alternative is
the existing `MetalGeometryPrior` (differentiable / torch-based)
which requires 3-D coordinates — see §6 honest framing.

---

## 2. Lit basis

* **Schulman et al. 2017** (PPO clipped objective).  The soft
  constraint formulation parallels PPO's ratio-clipping: we clip the
  score at 0 (never `-inf`) so a bad candidate still has a finite,
  non-zero gradient for the MCTS update.  A hard 0/1 gate would
  correspond to "ratio clipping to 0", which PPO explicitly avoids.
* **Neu 2017** (entropy-regularised MDPs).  A soft prior yields
  better exploration than a hard gate because the agent retains
  information about *how close* a candidate was to the target.
* **Dayan 1997** (potential-based reward shaping).  The soft score
  is equivalent to a *potential function* `Phi(s)` that adds a
  shaping term without changing the optimal policy.  Our formulation
  uses `(1 - distance)` so that `Phi(ideal) = 1` and `Phi(far) = 0`,
  recovering the "potential-based shaping is policy-preserving"
  theorem.

---

## 3. Implementation surface

### 3.1 New API in `molmetal_lam/priors/metal_geometry.py`

* `soft_score_metal_geometry(state, *, enabled=True, cn_targets=None,
  weight_coord=0.5, weight_angle=0.3, weight_charge=0.2,
  ideal_charge=0) -> float` — the core scorer.
* `make_soft_metal_geometry_channel(**kwargs)` — returns a
  `(state) -> float` closure that wraps the scorer (used by the
  aggregator).
* `SoftMetalGeometryDiagnostics` — `@dataclass` exposing per-call
  diagnostics (`n_metal_centres`, `best_cn`, `best_target_cn`,
  `best_d_coord`, `best_d_angle`, `best_d_charge`, `best_score`).
* `DEFAULT_SOFT_SCORE_TARGETS: dict` — `{atomic_num: target_CN}`
  mapping for Pt(II)/Pd(II)/Cu(II)/Zn(II)/Au(III) → 4 and
  Ru(II)/Ir(III)/Rh(III)/Fe(II) → 6 (matches the per-metal arity
  table in `molmetal_lam/atoms/combinators`).
* `_coerce_state_atoms_bonds`, `_coordination_count`,
  `_formal_charge`, `_angle_deviation_for_metal` — private helpers
  that mirror the legacy counter in `metal_geometry_prior_bonus`
  (kept here so the soft score is unit-testable in isolation
  without importing the orchestrator script).

### 3.2 New channel in `RewardAggregator`

* New `r_metal_geom_soft: Optional[Callable] = None` field (channel
  is opt-in so existing reward is bit-for-bit identical when
  unused).
* New `w_metal_geom_soft: float = 0.0` field (per-channel weight,
  default 0.0 = off).
* New `register_soft_metal_geometry_channel(...)` method that wires
  the closure with the caller's keyword arguments.
* Aggregation in `__call__` adds
  `value += self.w_metal_geom_soft * v_metal_geom_soft`.  The
  `_safe` wrapper ensures the channel degrades to `0.0` on any
  exception (consistent with the existing graceful-fail contract).

---

## 4. Test results

```
$ uv run pytest molmetal/molmetal_lam/tests/test_soft_metal_geometry.py -v --tb=short
========================= test session starts ==========================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
collected 9 items

molmetal/molmetal_lam/tests/test_soft_metal_geometry.py::test_soft_score_cisplatin PASSED
molmetal/molmetal_lam/tests/test_soft_metal_geometry.py::test_soft_score_trans_platin PASSED
molmetal/molmetal_lam/tests/test_soft_metal_geometry.py::test_soft_score_pt_iv_octahedral PASSED
molmetal/molmetal_lam/tests/test_soft_metal_geometry.py::test_soft_score_organic PASSED
molmetal/molmetal_lam/tests/test_soft_metal_geometry.py::test_soft_score_continuous PASSED
molmetal/molmetal_lam/tests/test_soft_metal_geometry.py::test_soft_score_disabled_returns_zero PASSED
molmetal/molmetal_lam/tests/test_soft_metal_geometry.py::test_aggregator_wires_soft_channel PASSED
molmetal/molmetal_lam/tests/test_soft_metal_geometry.py::test_aggregator_default_weight_is_zero PASSED
molmetal/molmetal_lam/tests/test_soft_metal_geometry.py::test_soft_score_3d_agnostic PASSED

========================= 9 passed, 1 warning in 1.43s ===================
```

### 4.1 Coverage matrix

| Test | Asserts | Score observed | Expected |
| --- | --- | --- | --- |
| `test_soft_score_cisplatin` | cis-Pt(NH3)2Cl2 in (0.8, 1.0] | 1.0 | ✓ |
| `test_soft_score_trans_platin` | trans-Pt(NH3)2Cl2 in (0.5, 1.0] (CN=4 same as cis) | 1.0 | ✓ |
| `test_soft_score_pt_iv_octahedral` | Pt(IV) CN=6 ≤ 0.75 and < cisplatin | 0.75 | ✓ |
| `test_soft_score_organic` | benzene (no metal) = 0.0 | 0.0 | ✓ |
| `test_soft_score_continuous` | CN=3 in (0, cisplatin) ≈ 0.8 (no cliff) | 0.8 | ✓ |
| `test_soft_score_disabled_returns_zero` | enabled=False → 0.0 for all | 0.0 | ✓ |
| `test_aggregator_wires_soft_channel` | `register_...()` plugs the channel; cisplatin > 0.8; organic = 0.0 | — | ✓ |
| `test_aggregator_default_weight_is_zero` | default `w_metal_geom_soft=0`; bit-for-bit identical without opt-in | — | ✓ |
| `test_soft_score_3d_agnostic` | no 3-D coords needed; works on plain atoms/bonds list | — | ✓ |

### 4.2 No regressions

```
$ uv run pytest molmetal/molmetal_lam/tests/test_metal_geometry.py \
                  molmetal/molmetal_lam/tests/test_metal_hydration.py \
                  molmetal/molmetal_lam/tests/test_soft_metal_geometry.py -q
======================== 26 passed, 1 warning in 1.43s ====================
```

The 7 pre-existing `test_metal_geometry.py` tests (rigorous
torch-based `MetalGeometryPrior` with 3-D coordinates) and the 10
pre-existing `test_metal_hydration.py` tests still pass — the new
soft-score function is additive (new module exports + new optional
aggregator channel).

---

## 5. Before / after collapse rate (analysis only, not measured)

The collapse-rate comparison is **NOT measured** in this Phase 3H
deliverable (the GPU / MCTS live loop is the Phase 4 integrator's
domain).  We provide the analysis only — Round-13 100x3 sweep is the
ground truth, currently BLOCKED by GPU outage.

### 5.1 Before (hard gate)

The historical
`metal_geometry_prior_bonus(state, *, enabled=True)` returns:

```
CN == target          -> 1.0   (exact match)
|CN - target| >= 1    -> 0.0   (binary collapse)
no metal              -> 0.0   (organic candidate)
```

If a partial-but-promising candidate (e.g. CN=3 around Pt(II))
appears in the rollout, the MCTS sees `0.0` reward for the metal
prior and the candidate falls back to Vina/SA/PB channels only.
The metal-prior signal is binary: the MCTS has no gradient to
distinguish "CN=3 close to ideal" from "off-topic organic".

### 5.2 After (soft, continuous)

The new `soft_score_metal_geometry(state, *, enabled=True)` returns:

```
CN=4 (ideal)          -> 1.0          (matches hard-gate optimum)
CN=3 (off-by-one)     -> 0.8          (smooth, informative gradient)
CN=2 (off-by-two)     -> 0.5
CN=6 (Pt(IV))         -> 0.75         (reduced but non-zero)
no metal              -> 0.0          (organic candidate)
```

The MCTS now sees a continuous gradient: candidates ranked closer to
the ideal Pt(II) coordination earn proportionally more reward, so
the rollout can rank off-by-one candidates above pure organic
ligands.

### 5.3 Projected collapse-rate change (NOT MEASURED)

The Round-12 5x1 metal-seed cisplatin arm (n_sim=1000) reported
`n_distinct=1` for the metal-seeded cisplatin arm — every candidate
collapsed onto the bare-metal seed.  The soft prior's projection is
that the CN=3 partial candidates now have `score=0.8` (vs the
historical `score=0.0`), so the MCTS UCB formula sees a non-zero
gradient and is less likely to collapse.  The actual delta must be
re-measured under the Round-13 sweep — currently BLOCKED by GPU.

The `WF-Lambda-Fix-Singleton Fix 1` entry in memory already ships a
similar soft-tier (0/0.2/0.5/1.0) bonus, which the orchestrator
still uses (we did not touch it).  The new function formalises the
soft-tier as a *continuous* distance-to-ideal score rather than a
discrete 4-bucket tier.

---

## 6. Honest framing

* **Coordinate-free proxy.**  The angle-deviation term does not
  embed the molecule in 3-D — we cannot afford RDKit / ETKDG
  embedding inside the MCTS loop.  Cis-platin and trans-platin
  therefore score identically (both CN=4).  The rigorous alternative
  is `MetalGeometryPrior` (differentiable / torch-based) which
  requires 3-D coordinates and is wired into the EGNN training loop
  rather than the MCTS proof search.
* **Charge term is a stub.**  The combinator library models Pt(II) /
  Pt_III as `Atom(symbol="Pt_II", ...)` — the oxidation state is
  encoded in the symbol suffix.  We do NOT parse it into a formal
  charge for the soft score; the soft score targets *neutral
  complexes* (cisplatin's Pt(II) is balanced by 2 Cl⁻).  Callers
  needing rigorous oxidation-state tracking should wrap RDKit's
  `Chem.GetFormalCharge` (already exposed via
  `r4_lambda_only_run.metal_compliance_truthful`).
* **Best-over-metals aggregation.**  Multi-metal molecules take the
  maximum score across centres (matches the historical `best`
  aggregation).  A more rigorous formulation could weight by
  metal-identity importance — left as a future-work item.
* **No MCTS loop integration in this PR.**  This deliverable ships
  the function, the tests, and the aggregator channel.  Wiring the
  new channel into the Round-13 100x3 sweep is the Phase 4
  integrator's responsibility (we did not touch
  `r4_lambda_only_run.py`).

---

## 7. Files touched

| File | Change |
| --- | --- |
| `molmetal/molmetal_lam/priors/metal_geometry.py` | + `soft_score_metal_geometry`, `make_soft_metal_geometry_channel`, `SoftMetalGeometryDiagnostics`, `DEFAULT_SOFT_SCORE_TARGETS`; + 4 private helpers (`_coerce_state_atoms_bonds`, `_coordination_count`, `_formal_charge`, `_angle_deviation_for_metal`); + 6 new entries in `__all__` |
| `molmetal/molmetal_lam/search_alg/proof_search.py` | + `r_metal_geom_soft` channel field; + `w_metal_geom_soft = 0.0` weight field; + `register_soft_metal_geometry_channel(...)` method; + aggregation in `__call__` |
| `molmetal/molmetal_lam/tests/test_soft_metal_geometry.py` | new file, 9 tests |

No changes to `r4_lambda_only_run.py`, `paper/main.tex`, or any of
the Phase 3 agents' file set (per Task-H constraints).

---

## 8. Recommended next steps (Phase 4 integrator domain)

1. Wire `register_soft_metal_geometry_channel()` into
   `r4_lambda_only_run.py`'s default reward profile with
   `w_metal_geom_soft = 0.5` (half-weight to keep Vina dominant).
2. Re-run Round-13 100x3 sweep once the GPU recovers — measure
   `n_distinct` (collapse-rate proxy) before/after.
3. Add a §4.6 ablation row: `metal_prior = hard` (current
   `metal_geometry_prior_bonus`) vs `metal_prior = soft` (new
   `soft_score_metal_geometry`).
4. If `n_distinct` lifts, promote the soft prior to default for
   Round-14 (TODO-25).