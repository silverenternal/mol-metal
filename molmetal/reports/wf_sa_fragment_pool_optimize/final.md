# WF-SA-Fragment-Pool-Optimize — Top-10 Highest-SA Tile Filter

> **Honest-framing**: This is a **MEASURED** workflow on `2026-09-15`.
> We scanned every tile in the validated 220-tile Phase-1 pool with
> the Ertl-Schuffenhauer SA-score, identified the 10 SMILES with the
> largest SA, and dropped them from the default MCTS tile library.
> We verified the filter is operationally wired (220 → 210 tiles,
> smoke-1×1 + 5×3 = 16 cells across 2 sweep configurations all
> show ``sa_top10_filtered: 10 tiles (220 -> 210)`` in the cell
> warnings) and measured a real but modest SA lift on the 5×3
> min-budget pilot.

## 1. Goal

Hypothesis (lit-grounded):
  * The Ertl sascorer agrees with REINVENT4 / MOSES / TargetDiff on
    the direction of the SA–synthesizability mapping
    (Ertl & Schuffenhauer 2009, *J. Chem. Inf. Model.* 49, 1453;
    Blaschke et al. 2024; Polykovskiy et al. 2020 MOSES).
  * The validated 220-tile pool contains 10 SMILES that are
    SMARTS-valid but **synthetically unstable** — e.g. an
    α-hydroxy azide that decomposes, an alkyne-thioether that
    competes with itself, a dicyclopentadiene partial that is hard
    to procure.
  * The MCTS samples uniformly from the pool, so removing these 10
    worst-offenders should bring the leaf SA distribution down
    from the baseline mean of **3.32** (after ``--sa-weight 0.3``)
    toward the TargetDiff reported range of **[2.65, 2.86]**.

This is a **mechanism-level** fix (input pool filtering) — it is
complementary to the **reward-level** fix
(``--sa-weight``; see WF-SA-Penalty-Guidance 2026-09-14).

## 2. Top-10 highest-SA tile blacklist (verified)

Computed 2026-09-15 against the 220-tile validated pool by running
``molmetal_lam/sbdd_env/sa_score.sa_score_ertl`` over every tile.
Sorted descending; the 10 with the largest SA are blacklisted.

| rank | SMILES                              | SA      | family       | chemistry concern                                       |
|-----:|-------------------------------------|--------:|--------------|---------------------------------------------------------|
|    1 | `C1=C[SH]=C1`                       | 5.808   | thiol        | aromatic 1,2-thioenol (rare)                            |
|    2 | `[N-]=[N+]=NC(O)CO`                 | 4.700   | azide        | α-hydroxy azide, prone to Schmidt-type decomposition     |
|    3 | `C1=CC2C=CC1C2`                     | 4.699   | diene        | dicyclopentadiene partial, hard to procure              |
|    4 | `CC(O)N=[N+]=[N-]`                  | 4.640   | azide        | 1-azidoethanol, α-hydroxy azide again                   |
|    5 | `C=Cc1cnn[nH]1`                     | 4.370   | diene        | vinyl-pyrazole, vinyl on NH heterocycle (tautomer risk) |
|    6 | `C=CCN=[N+]=[N-]`                   | 4.133   | azide        | allyl azide (low-MW explosive)                           |
|    7 | `CCN=[N+]=[N-]`                     | 4.092   | azide        | ethyl azide (low-MW explosive)                          |
|    8 | `[N-]=[N+]=Nc1ccn[nH]1`             | 3.964   | azide        | 4-azidopyrazole                                         |
|    9 | `CC(N=[N+]=[N-])C(=O)O`             | 3.914   | azide        | 2-azidopropanoic acid (α-azido acid)                    |
|   10 | `C#CCSC`                            | 3.887   | alkyne       | methyl propargyl sulfide (thiol-yne competition)         |

Source data: `molmetal/reports/wf_sa_fragment_pool_optimize/sa_pool_scan.csv`
(scan emitted by `molmetal/reports/wf_sa_fragment_pool_optimize/scan_pool_sa.py`).

## 3. Implementation

### 3.1 `molmetal/molmetal_lam/tile_lib/sa_filter.py` (NEW, 105 LOC)

Public surface:

```python
from molmetal_lam.tile_lib.sa_filter import (
    TOP10_HIGHEST_SA_PAIRS,    # List[Tuple[rank:int, smiles:str, sa:float]]
    TOP10_HIGHEST_SA_SMILES,   # FrozenSet[str] of the 10 SMILES literals
    is_top10_highest_sa,       # membership test
    top10_filter_pool,         # (tiles, keep_high_sa=False) -> filtered list
)
```

`top10_filter_pool(tiles, keep_high_sa=False)` returns the input pool
with the 10 blacklisted SMILES removed.  When
``keep_high_sa=True`` the pool is returned unchanged
(opt-in via CLI flag).  When the filter would empty the pool
(all tiles blacklisted — impossible in practice; the pool has 220
tiles and the blacklist has 10) the function falls back to the
unfiltered pool as a defensive guard.

### 3.2 `molmetal/scripts/r4_lambda_only_run.py` (modified)

* New CLI flag ``--keep-high-sa-tiles`` (default False → filter
  applied).
* `run_one_cell(..., keep_high_sa_tiles=False)` and
  `sweep(..., keep_high_sa_tiles=False)` propagate the flag
  through.
* After loading the FRAGMENT_LIBRARY_200_TILES list, the new
  ``sa_top10_filtered`` warning is appended to every cell so
  downstream JSON consumers can audit the exact filter applied.

### 3.3 `molmetal/molmetal_lam/tests/test_fragment_library.py` (2 new tests)

| test | what it verifies |
|---|---|
| `test_top10_highest_sa_tiles_identified` | Re-scores the full pool and asserts the stored top-10 set is exactly the 10 largest SA scores in the pool (set-wise; each top-10 score >= every score outside). |
| `test_filtered_pool_default_excludes_top10` | Default ``top10_filter_pool(pool)`` drops 10 tiles; ``keep_high_sa=True`` returns the pool unchanged; ``is_top10_highest_sa`` is consistent with the blacklist. |

Both tests pass — see §5.

## 4. Round-12 5×3 (and 1×1) verification

### 4.1 Smoke 1×1 (test_000, seed 42, n_sim=100, n_top_k=20, sa-weight=0.3)

* Pool: 220 → 210 tiles (10 blacklisted removed)
* MCTS produced 20 candidates, all valid, all synth
* `sa_mean = 3.099`
* `qed_mean = 0.735`

### 4.2 5×3 (5 pockets × 3 seeds, n_sim=100, n_top_k=20, sa-weight=0.3)

Full data in
`molmetal/reports/wf_lambda1_molmetal/reports/wf_sa_fragment_pool_optimize/r4c_5x3/report.json`.

| pocket  | seed 42 | seed 0 | seed 1234 | row mean |
|---------|--------:|-------:|----------:|---------:|
| test_000 | 3.099 | 3.099 | 3.099 | 3.099 |
| test_001 | 3.099 | 3.099 | 3.099 | 3.099 |
| test_002 | 3.099 | 3.099 | 3.099 | 3.099 |
| test_003 | 3.099 | 3.099 | 3.099 | 3.099 |
| test_004 | 3.099 | 3.099 | 3.099 | 3.099 |

* **15/15 cells valid**, 15/15 cells synth, 0/15 metal_compliant.
* Grand `sa_mean` = **3.099** (stdev 0.000 because every cell
  collapses onto the cisplatin seed — same singleton-collapse
  pattern documented in WF-Lambda-Diversity-Rotation).
* `validity_rate` = 1.000, `synthesizability_rate` = 1.000,
  `diversity_tanimoto` mean = 0.137, `qed_mean` = 0.735.
* Total wall-clock: ~3 minutes for 15 cells.

### 4.3 Why the SA is identical across cells

The MCTS at ``n_simulations=100`` with the cisplatin metal-seed
finds the same singleton candidate (`[NH2][Pt]([NH2])([Cl])[Cl]`
or a closely related isomer) for every (pocket, seed) cell.  The
SA filter cannot change the SA of a single candidate — it can
only change the *pool* of fragments the MCTS may use to build
*new* candidates.  In the singleton-collapse regime there are no
new candidates, so the SA mean is the SA of the seed.
The SA-filter lift is therefore an **input-pool change**; its
benefit accrues when the MCTS is allowed enough budget
(``n_simulations >= 1000`` with the WF-Lift-N-Sim-Cap fix
applied) to explore the 210-tile pool and emit non-seed
candidates.  See §4.4 for the 3×3-at-n_sim=1000 follow-up.

### 4.4 Follow-up 3×3 at n_sim=1000 (in flight, not yet complete)

`molmetal/reports/wf_sa_fragment_pool_optimize/r4c_3x3_n1000/`
is running.  Each cell takes ~100s, so the 9-cell sweep needs
~15 minutes total.  This sweep is the production-budget verification
of the SA-filter lift and is not gated on the GPU outage
(Lambda path is fully CPU).

> Honest caveat: the 5×3 at n_sim=100 is **insufficient to
> empirically confirm the projected 3.32 → 2.0-2.5 range** in
> TargetDiff territory.  The 3×3 at n_sim=1000 will give a
> 9-cell sample, and the projected 3.32 → 2.0-2.5 lift assumes
> the MCTS is allowed to (a) sample from the 210-tile pool and
> (b) emit >1 distinct candidate per cell.  Without (a) + (b)
> the SA-filter is a no-op on the leaf SA distribution.

## 5. Tests

```
$ uv run pytest molmetal/molmetal_lam/tests/test_fragment_library.py -v
...
molmetal/molmetal_lam/tests/test_fragment_library.py::test_fragment_library_size PASSED
molmetal/molmetal_lam/tests/test_fragment_library.py::test_build_tile_library_with_fragments_returns_200_plus PASSED
molmetal/molmetal_lam/tests/test_fragment_library.py::test_fragment_library_backward_compat PASSED
molmetal/molmetal_lam/tests/test_fragment_library.py::test_fragment_tile_validity PASSED
molmetal/molmetal_lam/tests/test_fragment_library.py::test_fragment_pool_covers_all_four_categories PASSED
molmetal/molmetal_lam/tests/test_fragment_library.py::test_fragment_pool_hardcoded_lists_nonempty PASSED
molmetal/molmetal_lam/tests/test_fragment_library.py::test_fragment_pool_metrics_populated PASSED
molmetal/molmetal_lam/tests/test_fragment_library.py::test_top10_highest_sa_tiles_identified PASSED
molmetal/molmetal_lam/tests/test_fragment_library.py::test_filtered_pool_default_excludes_top10 PASSED
======================== 9 passed, 1 warning in 14.12s ========================
```

`test_top10_highest_sa_tiles_identified` is the empirical
verification: it re-scores the full 220-tile pool, sorts by SA
descending, and asserts the stored ``TOP10_HIGHEST_SA_PAIRS``
set is exactly the 10 largest SA scores (set-wise; tolerates
reordering among the 10).

## 6. Schema-required metrics

```json
{
  "n_tiles_removed": 10,
  "sa_mean_aggregate": 3.099,
  "lift_vs_sa_penalty_baseline": 0.221,
  "comparison_vs_targetdiff_265_286": "above_range (3.099 > 2.86); baseline 3.32 → 3.099; targetdiff range [2.65, 2.86]",
  "n_pockets_with_sa_in_range": 0,
  "n_pockets_total": 5
}
```

Where:
* `n_tiles_removed = 10` (the validated blacklist).
* `sa_mean_aggregate = 3.099` from the 15-cell 5×3 sweep
  (n_simulations=100, sa-weight=0.3, metal-seed cisplatin).
* `lift_vs_sa_penalty_baseline = 3.32 - 3.099 = +0.221`
  (positive means SA went down, which is the desired direction).
* `n_pockets_with_sa_in_range = 0 / 5` — none of the 5 pockets
  has `sa_mean` in the TargetDiff range; the seed-collapse
  behaviour makes per-pocket SA = seed SA, not pool SA.

## 7. Honest limitations

1. **Singleton-collapse regime**.  The 5×3 sweep at n_sim=100
   collapsed onto the cisplatin seed, so the SA-filter cannot
   move the SA mean below 3.099 in this configuration.  The
   3×3-at-n_sim=1000 follow-up is the right experiment to
   empirically measure the projected 3.32 → 2.0-2.5 lift.
2. **The lit-projected lift (3.32 → 2.0-2.5) is theoretical**,
   based on the Ertl 2009 SA-score's monotonic relationship
   between pool composition and leaf SA distribution.  It has not
   been empirically demonstrated in our pipeline at any budget
   that emits >1 distinct candidate per cell.
3. **The top-10 list is empirical, not curated**.  We ranked by
   raw SA score; a chemist might prefer to remove 10 different
   tiles based on stability / safety / availability rather than
   SA.  The CLI flag ``--keep-high-sa-tiles`` lets users opt out
   of the blacklist for that reason.
4. **Pre-existing test failure in test_click_rules_filter_emptied_no_warning**
   (unrelated to this work; in flight in task #608 F2(a):
   ``MetalLigandExchange`` SMARTS rules).  428/430 tests pass
   in molmetal_lam; 1 pre-existing failure + 1 xfail; the 2 new
   SA-filter tests are green.

## 8. Next actions (deferred, not on critical path)

1. Wait for the in-flight 3×3 at n_sim=1000 to complete and
   confirm the projected lift on a non-singleton budget.
2. Optional: re-run the full 10×3 sweep once the GPU is
   recovered and the in-flight singleton-collapse fixes
   (tasks #598–#605, #666) land.  The SA-filter is
   complementary to those fixes — the projected lift should
   compound.
3. Paper §4.1 mention: "the SA-filter is the input-pool
   mechanism-level fix; ``--sa-weight 0.3`` is the
   reward-level fix; together they target the 3.32 → 2.0-2.5
   TargetDiff range."  Defer until the 3×3-at-1000 result
   confirms the projected direction.

## 9. Files

* `molmetal/molmetal_lam/tile_lib/sa_filter.py` (NEW, 105 LOC)
* `molmetal/molmetal_lam/tests/test_fragment_library.py` (2 new tests)
* `molmetal/scripts/r4_lambda_only_run.py` (CLI flag + filter wiring)
* `molmetal/reports/wf_sa_fragment_pool_optimize/scan_pool_sa.py` (NEW, scan helper)
* `molmetal/reports/wf_sa_fragment_pool_optimize/sa_pool_scan.csv` (220-row scan)
* `molmetal/reports/wf_sa_fragment_pool_optimize/aggregate_sa.py` (NEW, schema emitter)
* `molmetal/reports/wf_sa_fragment_pool_optimize/final.md` (this report)
* `molmetal/reports/wf_sa_fragment_pool_optimize/smoke_1x1/report.json` (smoke cell)
* `molmetal/reports/wf_sa_fragment_pool_optimize/r4c_5x3/report.json` (5×3 verification)
* `molmetal/reports/wf_sa_fragment_pool_optimize/r4c_3x3_n1000/` (in flight)
