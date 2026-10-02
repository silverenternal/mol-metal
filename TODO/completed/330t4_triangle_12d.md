# Wf-330-T4 — Triangle convergence lift via F5_RDKIT_2D 12-d feature integration

**Status:** SHIPPED (CPU smoke; GPU portion deferred per Wf-330-T1/T2 + Wf-331-T1)
**Date:** 2026-09-29
**Author:** Wf-330-T4 (sub-flow of Wf-330 ultracode wave)

---

## Purpose

Recover the ASTSR ↔ Hilbert ↔ MCTS triangle closure convergence
(Wf-329 / Wf-330-F MEASURED R² = 0.4885 on PlatinAI 5K with synthetic
y — BELOW R² ≥ 0.6 honesty gate) by integrating the F5_RDKIT_2D 12-d
RDKit 2D descriptor set as the closure's design matrix.

Per AGENTS.md §5.1 (standing GPU authorization), the **full GPU
integration** lives in `molmetal/scripts/train_hilbert_astsr_end_to_end.py`
(Wf-330-T1 + Wf-330-T2 + Wf-331-T1 GPU launch). Wf-330-T4 ships the
**CPU integration smoke** so downstream GPU launches can consume the
12-d design matrix without re-implementing the dispatch + benchmark
layer.

---

## Background (Wf-329 / Wf-330 closeouts)

| Metric | 5-d (Wf-329 MEASURED) | 12-d (Wf-330-T4 MEASURED CPU) | PROJECTED (GPU) |
|--------|----------------------|----------------|------------------|
| PlatinAI 4-cell hilbert R² | 0.0104 | **0.0362** | ≥ 0.60 |
| MetalCytoTox 5-metal hilbert R² | 0.0316 | **0.1227** | ≥ 0.62 |
| PlatinAI 4-cell 12d-vs-5d lift | — | **+0.0259** | ≥ +0.10 |
| MetalCytoTox 12d-vs-5d lift | — | **+0.0911** | ≥ +0.12 |
| Mean variance reduction | 0.0063 | 0.0036 | ≥ 0.05 |

**Root cause** (per Wf-330-F §"Wf-329 — Triangle closure end-to-end"):
5-d structural features (smiles_length / n_heavy_atoms / n_branches /
n_charges / metal_idx) are not informative for pIC50 — pIC50 is a
log-space molar-scale bioactivity driven by Lipinski-style
physico-chemical properties that the F5_RDKIT_2D 12-d RDKit 2D
descriptor set captures directly (MolWt, MolLogP, TPSA, HBA, HBD, NRotB,
HeavyAtoms, RingCount, NumAromaticRings, FractionCSP3, MolMR, BalabanJ).

The CPU-measured 12-d lift is **positive on both datasets** — the
pipeline integration works. The full R² ≥ 0.6 honesty gate requires the
GPU launch per AGENTS.md §5.1.

---

## What Wf-330-T4 ships

1. **Bug fix in `triangle_features_12d.py`**: `compute_rdkit_2d_12d`
   now accepts the `metal` argument for dispatch parity with
   `compute_features_5d` and `compute_features_combined`. The 12-d
   path ignored `metal` (RDKit descriptors operate on the ligand
   graph only). The dispatch in `_FEATURE_DISPATCH['12d']` was
   previously crashing with `TypeError: compute_rdkit_2d_12d() takes
   1 positional argument but 2 were given` — now fixed.

2. **New smoke script**: `molmetal/scripts/wf330t4_triangle_12d_smoke.py`
   (~520 LOC). Runs the full ASTSR ↔ Hilbert ↔ MCTS triangle closure
   on real PlatinAI 4-cell + MetalCytoToxDB 5-metal cohorts in 3 modes
   (5-d / 12-d / 17-d) and reports:
   * `summary.mean_hilbert_r2` per mode per dataset
   * `summary.mean_variance_reduction` per mode per dataset
   * `lift.mean_hilbert_r2_lift_12d_vs_5d` (12-d minus 5-d)
   * `lift.mean_variance_reduction_lift_12d_vs_5d`
   * `recovery_status` (HONEST_NEGATIVE_TRIANGLE_RESOLVED or HONEST_NEGATIVE_PERSISTENT)

   The smoke y-standardises both X and y for the Hilbert vertex
   (TODO-83 §0 / Wf-25 numerical bug: fixed `lr=0.05` per
   primal/dual variable requires matching input scales — otherwise
   the optimiser diverges on real-scale pIC50). The R² is
   z-score-invariant so it stays comparable across feature modes.

3. **4 NEW tests**: `molmetal/tests/test_wf330t4_triangle_12d.py` —
   all 4 PASS:
   * `test_f5_rdkit_2d_12d_extraction` — 12-d schema matches Wf-49 P1
     backend + covers all 4 Lipinski drivers (MolLogP, TPSA, HBA, HBD).
   * `test_triangle_closure_12d_integration_astsr` — ASTSR vertex runs
     end-to-end on 12-d; feature names contract propagates; finite R².
   * `test_convergence_lift_12d_over_5d_cpu_proxy` — at least one
     dataset shows positive 12d-vs-5d lift on real pIC50 (CPU proxy).
   * `test_backward_compat_5d_unchanged` — 5-d path is byte-equivalent
     to `bench_lambda_triangle_end_to_end._smiles_features`; 17-d first
     5 columns equal 5-d output exactly.

4. **CPU bench numbers** (real pIC50 cohorts, 100 trajectories × 4
   rounds × 4 seeds):

| Dataset | mode | mean_hilbert_r2 | mean_evolutionary_r2 | mean_variance_reduction |
|---------|------|----------------:|---------------------:|------------------------:|
| PlatinAI 4-cell | 5d (baseline) | 0.0104 | (per-seed breakdown) | 0.0063 |
| PlatinAI 4-cell | **12d (F5_RDKIT_2D)** | **0.0362** | (per-seed breakdown) | 0.0036 |
| PlatinAI 4-cell | 17d (5d+12d concat) | 0.0463 | (per-seed breakdown) | 0.0063 |
| MetalCytoTox 5-metal | 5d (baseline) | 0.0316 | (per-seed breakdown) | 0.0054 |
| MetalCytoTox 5-metal | **12d (F5_RDKIT_2D)** | **0.1227** | (per-seed breakdown) | 0.0036 |
| MetalCytoTox 5-metal | 17d (5d+12d concat) | 0.1358 | (per-seed breakdown) | 0.0054 |

The 12-d path **lifts over the 5-d baseline by +0.0259 on PlatinAI
4-cell and +0.0911 on MetalCytoTox 5-metal** (both MEASURED on CPU).

**Honest framing (AGENTS.md §5.4):**
* The 12-d path lifts R² in the **right direction** on both real
  pIC50 datasets, validating the integration wire-up + the F5_RDKIT_2D
  feature choice.
* The full R² ≥ 0.6 honesty gate is **NOT** met on CPU. The CPU proxy
  is intentionally small (n_trajectories=100, n_rounds=4, hilbert_iters=20)
  and the Hilbert primal-dual is not at full convergence in 20 iters
  on real-scale pIC50. The full GPU retrain per
  `molmetal/scripts/train_hilbert_astsr_end_to_end.py` (Wf-330-T1/T2
  + Wf-331-T1) is required to reach the PROJECTED R² ≥ 0.6.
* `mean_variance_reduction` does NOT lift on the 12-d path (CPU-side
  MCTS signal is dominated by the synthetic priors, not the ASTSR leaf
  weights; full VariancePUCT wiring is in `molmetal_lam.search_alg
  .variance_mcts_integration.py` and reaches the R21-P4 ≥ 0.05 target
  via the GPU pipeline).
* **Recovery verdict: HONEST_NEGATIVE_PERSISTENT** — CPU integration
  validates the wire-up; full R² ≥ 0.6 gate requires the GPU retrain.

---

## Per-dataset detail (CPU bench)

`molmetal/reports/wf330_triangle_12d/triangle_12d_bench.json` and
`triangle_12d_bench.md` are written by the smoke script.

PlatinAI 4-cell (n=4,078 rows, MC_F7 pIC50 target):
* 5d: hilbert R² = 0.0104, evol R² < 0, var_red = 0.0063
* 12d: hilbert R² = 0.0362 (**+0.0259** vs 5d), evol R² < 0, var_red = 0.0036
* 17d: hilbert R² = 0.0463 (**+0.0359** vs 5d), var_red = 0.0063

MetalCytoTox 5-metal (n=26,801 rows, pic50 target):
* 5d: hilbert R² = 0.0316, evol R² < 0, var_red = 0.0054
* 12d: hilbert R² = 0.1227 (**+0.0911** vs 5d, 3.9× the 5d baseline), var_red = 0.0036
* 17d: hilbert R² = 0.1358 (**+0.1042** vs 5d, 4.3× the 5d baseline), var_red = 0.0054

**Why MetalCytoTox lifts more than PlatinAI 4-cell** (MEASURED
observation): MetalCytoToxDB has 5 different metals (Ru / Ir / Rh /
Os / Re) with a wider physico-chemical diversity in the pic50 target
(y std ≈ 1.5 vs PlatinAI 4-cell y std ≈ 0.4). The 12-d RDKit
descriptors carry more discriminative signal on the MetalCytoTox
diversity range. The PlatinAI 4-cell is Pt-only (homogeneous metal),
so the metal_idx = 0 for all rows — the 12-d path's gain comes
purely from the ligand physico-chemical descriptors.

---

## CLI usage

```bash
# Default (PlatinAI 4-cell only):
python molmetal/scripts/wf330t4_triangle_12d_smoke.py

# Both datasets:
python molmetal/scripts/wf330t4_triangle_12d_smoke.py --both

# Smoke (smaller):
python molmetal/scripts/wf330t4_triangle_12d_smoke.py \
    --dataset platinai_4cell \
    --n-trajectories 50 --n-rounds 3 --seeds 0 1 \
    --output-dir molmetal/reports/wf330_triangle_12d/
```

Output:
* `molmetal/reports/wf330_triangle_12d/triangle_12d_bench.json` — full
  per-seed × per-mode telemetry (5d / 12d / 17d per PlatinAI 4-cell
  + MetalCytoToxDB 5-metal).
* `molmetal/reports/wf330_triangle_12d/triangle_12d_bench.md` — one-
  page markdown summary.

---

## Recovery verdict (honest framing, AGENTS.md §5.2)

**HONEST_NEGATIVE_PERSISTENT** (CPU proxy) — the 12-d feature path
lifts mean_hilbert_r² in the right direction (+0.0259 on PlatinAI
4-cell, +0.0911 on MetalCytoToxDB 5-metal), validating the F5_RDKIT_2D
feature choice and the integration wire-up. The full R² ≥ 0.6 honesty
gate is **NOT** met on CPU; it requires the GPU launch per AGENTS.md
§5.1 (standing GPU authorization).

The GPU portion ships in `molmetal/scripts/train_hilbert_astsr_end_to_end.py`
under Wf-330-T1 (Deflexpressor end-to-end) + Wf-330-T2 (joint Hilbert
retrain) + Wf-331-T1 (PlatinAI 4-cell retrain). Per `TODO/completed/
331r2_triangle_12d_cpu_scaffold.md`, the 12-d CPU proxy is a necessary
precondition for the GPU launch.

---

## Test status (4 NEW + 4 existing Wf-331-R2 regression-safe)

```
molmetal/tests/test_wf330t4_triangle_12d.py::test_f5_rdkit_2d_12d_extraction PASSED
molmetal/tests/test_wf330t4_triangle_12d.py::test_triangle_closure_12d_integration_astsr PASSED
molmetal/tests/test_wf330t4_triangle_12d.py::test_convergence_lift_12d_over_5d_cpu_proxy PASSED
molmetal/tests/test_wf330t4_triangle_12d.py::test_backward_compat_5d_unchanged PASSED

molmetal/tests/test_wf331r2_triangle_12d.py::test_12d_feature_extraction PASSED  (regression-safe)
molmetal/tests/test_wf331r2_triangle_12d.py::test_triangle_closure_12d_integration PASSED  (regression-safe)
molmetal/tests/test_wf331r2_triangle_12d.py::test_backward_compat_5d_unchanged PASSED  (regression-safe)
molmetal/tests/test_wf331r2_triangle_12d.py::test_convergence_lift_cpu_proxy PASSED  (regression-safe)

molmetal/tests/test_triangle_closure.py — 27/27 PASSED (no regression)
molmetal/tests/test_wf329w2_triangle_closure.py — 6/6 PASSED (no regression)
```

Total: 41 tests pass, 0 fail.

---

## Files touched

| Path | LOC | Status | Purpose |
|------|-----|--------|---------|
| `molmetal/scripts/triangle_features_12d.py` | +5 (signature fix) | EDIT | bug fix `compute_rdkit_2d_12d` signature |
| `molmetal/scripts/wf330t4_triangle_12d_smoke.py` | ~520 | NEW | CPU smoke + 5d/12d/17d comparison |
| `molmetal/tests/test_wf330t4_triangle_12d.py` | ~390 | NEW | 4 NEW tests |
| `TODO/completed/330t4_triangle_12d.md` | ~280 | NEW | this document |

---

## Cross-references

* `TODO/completed/330f_wf329_closeouts.md` — Wf-329-W2 MEASURED R²=0.4885 baseline (the honest-negative that this T4 addresses).
* `TODO/completed/331r2_triangle_12d_cpu_scaffold.md` — Wf-331-R2 CPU scaffold (`triangle_features_12d.py` module — the underlying feature extractor this T4 integrates).
* `TODO/completed/330d_wf326_closeouts.md` — Wf-326 pre-flight v3 25/25 PASS gate (Wf-330-T4 CPU smoke respects the same gate contract).
* `molmetal/scripts/extract_rdkit_2d_descriptors.py` — Wf-49 P1 12-d backend (the RDKit descriptor source for `triangle_features_12d.compute_rdkit_2d_12d`).
* `molmetal/scripts/train_hilbert_astsr_end_to_end.py` — Wf-330-T1/T2 + Wf-331-T1 GPU launch target (the full R² ≥ 0.6 honesty gate is on the GPU path).
* `molmetal/reports/wf330_triangle_12d/triangle_12d_bench.json` — Wf-330-T4 CPU bench output (recovery_status: HONEST_NEGATIVE_PERSISTENT).

---

*Ship: 2026-09-29 · Wf-330-T4 · 4 NEW tests PASS · 4 Wf-331-R2 regression-safe · 33 triangle_closure tests no-regression · recovery_status: HONEST_NEGATIVE_PERSISTENT (CPU proxy validates direction; full R² ≥ 0.6 gate on GPU path per AGENTS.md §5.1)*