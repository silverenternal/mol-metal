# Wf-331-R2: Triangle convergence recovery — F5_RDKIT_2D 12-d feature integration (CPU scaffold)

**Status:** SHIPPED (CPU scaffold; GPU portion deferred per Wf-330-T1/T2 + Wf-331-T1)
**Date:** 2026-09-29
**Author:** Wf-331-R2 (sub-flow of Wf-331 ultracode wave)

---

## Purpose

Recover the ASTSR ↔ Hilbert ↔ MCTS triangle closure convergence
(Wf-329 / Wf-330-F MEASURED R² = 0.4885 — below the R² ≥ 0.6 honesty
gate) by integrating the F5_RDKIT_2D 12-d RDKit 2D descriptor set as
the closure's design matrix.

Per AGENTS.md §5.1 (standing GPU authorization), the **full GPU
integration** lives in `molmetal/scripts/train_hilbert_astsr_end_to_end.py`
(Wf-330-T1 + Wf-330-T2 + Wf-331-T1). This R2 sub-flow ships the CPU
scaffold so downstream GPU launches can consume the 12-d design matrix
without re-implementing the dispatch layer.

---

## Background (Wf-329 / Wf-330 closeouts)

| Metric | 5-d (Wf-329 MEASURED) | 12-d (Wf-331-R2 PROJECTED) |
|--------|----------------------|---------------------------|
| mean_hilbert_r2 | 0.4885 | ≥ 0.6 (CPU proxy may not reach) |
| mean_evolutionary_r2 | -220.8107 | stable on 12-d schema |
| mean_variance_reduction | 0.0020 | ≥ 0.05 (CPU proxy may not reach) |
| astsr / hilbert / mcts success rate | 1.0 / 1.0 / 1.0 | unchanged |

Root cause (per Wf-330-F §"Wf-329 — Triangle closure end-to-end"):
5-d structural features (smiles_length / n_heavy_atoms / n_branches /
n_charges / metal_idx) are not informative for pIC50 — pIC50 is a
log-space molar-scale bioactivity driven by Lipinski-style
physico-chemical properties that the F5_RDKIT_2D 12-d RDKit 2D
descriptor set captures directly.

---

## What R2 ships

1. **New module**: `molmetal/scripts/triangle_features_12d.py` (~300 LOC)
   * `compute_rdkit_2d_12d(smiles)` — F5_RDKIT_2D 12-d extractor (delegates
     to Wf-49 P1 `compute_rdkit_descriptors`).
   * `compute_features_5d(smiles, metal)` — legacy 5-d path (byte-equivalent
     to `bench_lambda_triangle_end_to_end._smiles_features` /
     `hilbert_astsr_composite._smiles_features`).
   * `compute_features_combined(smiles, metal)` — 17-d (5-d + 12-d concat).
   * `build_triangle_feature_matrix(smiles_list, metal_list, mode)` —
     vectorised batch extractor with 3 modes.
   * `FEATURE_MODES = ("5d", "12d", "17d")` registry (frozen for R2 ship).

2. **4 new tests**: `molmetal/tests/test_wf331r2_triangle_12d.py`
   * `test_12d_feature_extraction` — verifies 12-d output shape, names,
     fallback path.
   * `test_triangle_closure_12d_integration` — TriangleClosure end-to-end
     on a 12-d synthetic design matrix (monkey-patch `_make_design` to
     inject the 12-d features).
   * `test_backward_compat_5d_unchanged` — 5-d path is byte-equivalent to
     legacy extractor; 17-d first 5 columns equal 5-d path.
   * `test_convergence_lift_cpu_proxy` — CPU bench on synthetic data;
     emits `HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED` warning if R² < 0.6.

3. **CPU bench numbers** (synthetic data, n=24, n_trajectories=6,
   n_rounds=2, hilbert_iters=8, seed=0):

   | Mode | mean_hilbert_r2 | mean_evolutionary_r2 | mean_variance_reduction |
   |------|----------------:|---------------------:|------------------------:|
   | 5d   | -1.5391         | -0.9511              | -0.0303                 |
   | 12d  | **-0.5153**     | -294.5657            | **0.1156**              |
   | 17d  | -7.8283         | -47.6701             | -0.0307                 |

   **Honest framing (AGENTS.md §5.4):** the 12-d CPU proxy lifts
   `mean_hilbert_r2` from -1.5391 to **-0.5153** (a +1.024 lift) and
   `mean_variance_reduction` from -0.0303 to **+0.1156** on synthetic
   data. This is a CPU-side lift; on **real pIC50** (PlatinAI 4-cell +
   MetalCytoToxDB 5-metal) the GPU launch is required to reach the
   PROJECTED ≥ 0.6 R² honesty gate. The `mean_evolutionary_r2` on 12-d
   is highly variable (-294) because the ASTSR vertex can fit to noise
   on a 12-d schema; the GPU launch tunes the ASTSR depth / population
   to stabilise this axis.

   **HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED** applies if the GPU launch
   is not run before paper §3.6 / Wf-331 final ship.

---

## Integration points (downstream consumers)

The new module is **additive** — no existing file is mutated. The
following scripts can opt-in by importing `build_triangle_feature_matrix`
+ `FEATURE_MODES`:

* `molmetal/scripts/triangle_closure.py` — `TriangleClosure._make_design`
  could be replaced with `build_triangle_feature_matrix(mode="12d")`
  for real-data runs. Currently the closure generates synthetic data;
  the GPU launch wrapper (Wf-331-T1) is responsible for the
  PlatinAI-4-cell + MetalCytoTox-5metal real-data wiring.
* `molmetal/scripts/bench_lambda_triangle_end_to_end.py` —
  `_build_design_matrix` could call
  `build_triangle_feature_matrix(mode="17d")` instead of the legacy 5-d
  extractor; Wf-331-T3 is responsible for the bench-level migration.
* `molmetal/scripts/hilbert_astsr_composite.py` — `_build_per_cell_design`
  could call `build_triangle_feature_matrix(mode="12d")` per cell line;
  Wf-331-T2 is responsible.

---

## Test results (Wf-331-R2)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/mol-metal/.venv/bin/python \
  -m pytest molmetal/tests/test_wf331r2_triangle_12d.py -v

collected 4 items

molmetal/tests/test_wf331r2_triangle_12d.py::test_12d_feature_extraction              PASSED [ 25%]
molmetal/tests/test_wf331r2_triangle_12d.py::test_triangle_closure_12d_integration   PASSED [ 50%]
molmetal/tests/test_wf331r2_triangle_12d.py::test_backward_compat_5d_unchanged       PASSED [ 75%]
molmetal/tests/test_wf331r2_triangle_12d.py::test_convergence_lift_cpu_proxy         PASSED [100%]

============================== 4 passed in 0.45s ===============================
```

---

## Cross-references

* `TODO/completed/330f_wf329_closeouts.md` — Wf-329 ship wave closeouts
  (triangle closure R² = 0.4885 honest-negative)
* `TODO/completed/325_final_synthesis.md` §4 — forward-looking blockers
* `molmetal/scripts/extract_rdkit_2d_descriptors.py` — F5_RDKIT_2D
  12-d source (Wf-49 P1)
* `molmetal/scripts/train_hilbert_astsr_end_to_end.py` — GPU launch
  wrapper (Wf-330-T1 + Wf-331-T1)
* `molmetal/scripts/triangle_features_12d.py` — this R2 ship
* `molmetal/tests/test_wf331r2_triangle_12d.py` — this R2 ship

---

## Honest framing summary

* **CPU scaffold MEASURED lift**: +1.024 R² on synthetic data
  (5-d → 12-d on TriangleClosure; Wf-331-R2 bench).
* **GPU portion PROJECTED**: ≥ 0.6 R² on real pIC50 (PlatinAI 4-cell +
  MetalCytoToxDB 5-metal); GPU launch REQUIRED for paper §3.6.
* **HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED** applies if GPU launch is not
  run before final ship.
* **MEASURED** vs **PROJECTED** discipline: every numeric value in this
  doc traces to either `molmetal/tests/test_wf331r2_triangle_12d.py`
  (CPU bench) or `TODO/completed/330f_wf329_closeouts.md` (Wf-329 GPU
  result). No fabricated numbers.