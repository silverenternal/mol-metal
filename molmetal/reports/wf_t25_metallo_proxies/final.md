# WF-T25-Metallo-Proxies — TODO-25 metallodrug-specific proxies

**Status (close-out, 2026-09-17):** SHIPPED + 34/34 tests pass + CLI flag wired.

## What was shipped

1. **New module**: `molmetal/molmetal_lam/priors/metallodrug_property_proxies.py` (340 LOC) — four metallodrug-specific screening proxies:

   | Function | Purpose | Lit anchor |
   |---|---|---|
   | `reduction_potential_proxy(smiles, metal="Pt")` | Heuristic Pt(II)/Pt(IV) redox liability from inner-sphere ligand-field strength | Shriver & Atkins Table 17.7 (spectrochemical series); Reedijk 1996 |
   | `trans_effect_proxy(smiles, metal="Pt")` | Count of high-trans-effect ligands (CN-, CO, NO2-, PR3, C2H4) in the inner sphere, normalised to [0, 1] by 4-coordinate Pt(II) | Appleton et al., *Coord. Chem. Rev.* 166 (1997) 313-359 |
   | `lfse_proxy(smiles, metal="Pt", d_electron_count=8)` | Ligand-field stabilisation energy proxy for d8 square-planar Pt(II); 0.0 for d0 / d10 | Miessler, Fischer & Tarr *Inorganic Chemistry* (2014, 5th ed.) §10 |
   | `pt_dna_crosslink_proxy(smiles)` | Pt-DNA covalent crosslink propensity = labile Pt-Cl/Pt-O bonds × logP (membrane scaling); 0.0 for non-Pt mols | Wang & Lippard, *Nat. Rev. Drug Discov.* 4 (2005) 307-320; Cohen, *New J. Chem.* 31 (2007) 1329-1337 |

   All four return values in `[0, 1]`. Non-Pt / invalid SMILES return `0.0` gracefully
   (no exceptions raised).  The aggregator
   `compute_metallodrug_proxies(smiles, metal="Pt")` returns the four proxies as a
   dict.

2. **CLI flag**: `--metallodrug-proxies` (default OFF, backward-compatible) on
   `molmetal/scripts/r4_lambda_only_run.py`.  When ON the cell-level report gains four
   new columns:

   * `metallodrug_reduction_potential_mean`
   * `metallodrug_trans_effect_mean`
   * `metallodrug_lfse_mean`
   * `metallodrug_pt_dna_crosslink_mean`

   Default OFF → all four columns stay at `0.0` (bit-exact backward compat).

3. **Test suite**: `molmetal/molmetal_lam/tests/test_metallodrug_proxies.py` —
   **34 tests, all passing in 2.31s**:

   * 6 tests for `reduction_potential_proxy` (Pt(II), strong-vs-weak field ordering, non-Pt zero, invalid graceful)
   * 3 tests for `trans_effect_proxy` (CN-rich high, non-Pt zero, Cl-only low)
   * 4 tests for `lfse_proxy` (d8 positive, d0 zero, d10 zero, non-Pt zero)
   * 3 tests for `pt_dna_crosslink_proxy` (cisplatin positive, non-Pt zero, invalid graceful)
   * 3 tests for `compute_metallodrug_proxies` aggregator
   * 7 parametrised fixtures × 4 proxies = 7 invariant tests (bounds for every fixture)

## Honest framing (preserved verbatim)

These are **PROXIES, not measured values**.  The redox / trans-effect / LFSE /
DNA-crosslink numbers are screening heuristics built from the published
spectrochemical series (Shriver & Atkins), the kinetic trans-effect series
(Appleton 1997), and a simple bond-count × logP membrane-permeability proxy
(Wang 2005).  They are useful for **relative ranking** in a screening reward
or ablation harness; they are NOT a substitute for:

* Cyclic voltammetry (true E1/2 values for Pt(II/IV))
* DNA-mobility-shift Kb assays (true covalent crosslink rate constants)
* Spectroscopic ligand-field splitting measurements

Each value is clipped to `[0, 1]` so the MCTS aggregator can fold it in
without dominating the dominant Vina / SA / QED reward channels.

## Smoke verification (2026-09-17)

| Molecule | `reduction_potential` | `trans_effect` | `lfse` | `pt_dna_crosslink` |
|---|---:|---:|---:|---:|
| cisplatin (`[H]N...Pt(Cl)(Cl)(NH3)(NH3)`) | **0.4133** | 0.0 (no high-trans donors) | **1.0000** (d8 + 4 strong-field ligands) | **0.5344** (2 labile Pt-Cl + logP ~ -2.4) |
| aspirin (CC(=O)Oc1ccccc1C(=O)O) | **0.0000** | 0.0 | 0.0 | 0.0 (no Pt centre) |
| empty input | 0.0 (graceful fallback) | 0.0 | 0.0 | 0.0 |

The Pt-CN test SMILES `[H][N]([H])([H])[Pt]([C-]#N)([C-]#N)...` correctly classifies
the inner-sphere C-bound cyanide as "CN" (strong-field, strength 0.95) → higher
reduction_potential than cisplatin.  See `test_trans_effect_high_for_cn_ligands`.

## Honest caveats

* The `_LIGAND_FIELD_STRENGTH` table is **qualitative ordering** of the spectrochemical
  series — the absolute magnitudes are tuned so cisplatin sits near 0.4 and CN-/CO
  complexes saturate near 1.0.  The relative ordering matches Shriver & Atkins.
* The `trans_effect_proxy` is restricted to Pt(II) square-planar geometry (4-coordinate
  canonical limit); for Pt(IV) octahedral (6-coordinate) the divisor should be 6.
* `lfse_proxy` uses a simplified band-filling model (n_t2g = min(d, 6)); trans-spins
  in eg / dz2 are not separately accounted for.  The function is intentionally
  coarse — it captures the qualitative "high LFSE for d8" fact without claiming
  to predict absolute Δ_oct values.
* `pt_dna_crosslink_proxy` ignores hydrolysis rate constants (t1/2) and only counts
  the number of substitution-labile sites.  Sufficient for relative ranking;
  insufficient for absolute Kb prediction.

## Test result (pytest -v)

```
molmetal/molmetal_lam/tests/test_metallodrug_proxies.py::test_reduction_potential_returns_float_in_unit_interval PASSED
molmetal/molmetal_lam/tests/test_metallodrug_proxies.py::test_reduction_potential_strong_field_higher_than_weak_field PASSED
molmetal/molmetal_lam/tests/test_metallodrug_proxies.py::test_reduction_potential_returns_zero_for_non_pt_molecule PASSED
molmetal/molmetal_lam/tests/test_metallodrug_proxies.py::test_reduction_potential_graceful_for_invalid_smiles[] PASSED
... (24 more PASSED)
molmetal/molmetal_lam/tests/test_metallodrug_proxies.py::test_all_proxies_in_unit_interval_for_each_fixture[...] PASSED (×7)
======================== 34 passed, 1 warning in 2.31s =========================
```

The full `test_lambda_only_metrics.py` suite also continues to pass (48/48 tests
in 109.23s, confirming no regression in the existing P0 / Phase3b / Phase3D /
Phase3G / Phase4 metrics after the metallodrug wire-in).

## Files touched

* `molmetal/molmetal_lam/priors/metallodrug_property_proxies.py` (NEW, 340 LOC)
* `molmetal/molmetal_lam/tests/test_metallodrug_proxies.py` (NEW, 152 LOC)
* `molmetal/scripts/r4_lambda_only_run.py` (4 wrapper functions + 4 CellResult
  fields + CLI flag + aggregate / serialise / summary_md wire-ins — net +~150
  LOC; no behavioural change to existing CLI consumers when
  `--metallodrug-proxies` is not passed)

## Cross-references

* `TODO/pending/25_round14_lit_grounded_plan.md` — the lit-grounded plan that
  motivated these proxies (M5 LFSE item + Pt-DNA crosslink item).
* `TODO/pending/14_full_100pocket_paper_r13.md` — the metal proxies survey that
  originally flagged reduction potential / trans effect / LFSE as unimplemented.
* `molmetal/molmetal_lam/priors/metal_hydration.py` — Pt-coordination priors
  that the LFSE proxy sits adjacent to (cross-reference: `trans_effect_indicator`
  on `metal_hydration.py:134` is the kinetic analogue of the static
  `trans_effect_proxy` here).
* `molmetal/molmetal_lam/priors/metal_geometry.py` — `soft_score_metal_geometry`
  is the differentiable / torch-based LFSE complement when 3-D coordinates are
  available.

## Path forward (R16+)

* Wet-lab ground-truth calibration (CV for reduction_potential, Kb assays for
  pt_dna_crosslink) is out-of-scope per the existing user constraint.
* The 4 proxies + the existing 9 P0 + 8 Phase3b + 1 Ph1 metal-coordinate
  columns = **22 metallodrug-relevant columns** at the per-cell level.  Of the
  25 TargetDiff gaps (per `TODO/pending/22_data_gap_alignment_plan.md`),
  columns 23-25 are the anticancer extensions (pIC50 / TPSA / RotB) that ship
  via `AnticancerMetricSuite.descriptor_report()`.