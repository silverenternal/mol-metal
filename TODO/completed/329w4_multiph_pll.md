# Wf-329-W4 — Multi-PH/PLL feature extraction (32-bin Fourier + Multi-PH + PLL)

**Date**: 2026-09-29
**Status**: 🚀 **SHIPPED** (CPU-only feature extraction; 8/8 tests PASS; smoke PASS)
**Priority**: **HIGH** (m6_kappa + m7 + R² recovery path per WEAK_TO_STRONG_PLAN §2.5)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/pending/94_r20_p3_multiph_pll_feature.md` + `TODO/03_plans/WEAK_TO_STRONG_PLAN.md §2.5`
**Lambda gap(s)**: L7 (Multi-PH + PLL end-to-end) recovery candidate

---

## §0 Why this plan exists

Per Wf-325A + Wf-325D + Wf-325G findings (all closed honest-negative):

* **m6_kappa BELOW threshold** — MEASURED 0.472 vs gate 0.6 (Shape Kappa
  Index alone has NEGATIVE_WEAK MetalCytoToxDB validation; Pearson r 0.081).
  Recovery path: topologically-augmented shape descriptor (Multi-PH + PLL).
* **m7 Diversity ceiling** — MEASURED 0.1366 vs SOTA 0.81; pool-diversity
  ceiling reached on the current fragment pool.  Recovery path:
  PLL signature distance for the MaxMin selector.
* **R² ceiling** — MEASURED < 0.0 on real pIC50 (Wf-297B cause 2).
  Recovery path: 32-bin Fourier features + Multi-PH + PLL joint design
  matrix with Ridge regression.

Wf-329-W4 wires all three recovery primitives into a single CPU-only
feature extractor that drops into the m6_kappa + m7 + R² measurement
pipeline.  No GPU launch; no retrain (per AGENTS.md §5.1 standing GPU
grant — feature extraction is CPU-only).

---

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU-only (no GPU) |
| Wall-time | ~3 min on 30-SMILES metallo cohort |
| Output | `molmetal/reports/wf329w4_multiph_pll/{result.json, per_axis.csv}` |
| Tests | 8 new tests: Fourier + Multi-PH + PLL + 3 wire-ins |
| Code | 1 new module + 1 new smoke + 1 new test suite |

---

## §2 Deliverables

### §2.1 Module — `molmetal/molmetal_lam/reward/multiph_pll_feature.py`

Public API (CPU-only, deterministic, no GPU launch):

* `fourier_32bin_features(feature_map, n_bins=32) -> np.ndarray` —
  32-bin radially-binned FFT magnitude of a multi-PH feature map
  (Kim-Jung 2026 §4 acceleration).
* `MultiPHPLLFeatureExtractor(n_bins=32, k=32, knn=8, ...)` — joint
  extractor returning (Fourier, multi-PH, PLL) feature triple per
  SMILES.  Joint dim = 32 + 1024 + 32 = **1088**.
* `wire_m6_kappa_recovery(smiles, extractor)` — Ridge projection of
  the joint feature matrix onto the Shape Kappa Index signal;
  returns baseline + augmented + delta.
* `wire_m7_diversity_recovery(smiles, extractor, sample_size=30)` —
  PLL signature L2 distance vs Morgan-ECFP4 Tanimoto baseline.
* `wire_r2_recovery(smiles, extractor)` — joint design matrix
  (Fourier + Multi-PH + PLL) Ridge R² vs baseline (Fourier + PLL).
* `MultiPHPLLRecoveryReport` — dataclass with MEASURED-vs-PROJECTED
  tagging per AGENTS.md §5.2 honest framing.
* `smoke(...)` — CLI smoke entry point (also exposed as
  `molmetal/scripts/wf329w4_multiph_pll_smoke.py`).

### §2.2 Smoke — `molmetal/scripts/wf329w4_multiph_pll_smoke.py`

CLI entry point.  Run with::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
        /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \\
        molmetal/scripts/wf329w4_multiph_pll_smoke.py \\
        --output-dir molmetal/reports/wf329w4_multiph_pll

Emits `result.json` + `per_axis.csv` with the documented recovery
deliverables.

### §2.3 Tests — `molmetal/tests/test_wf329w4_multiph_pll.py`

8 tests, all PASS on the metallo cohort:

| # | Test | Status |
|---|---|---|
| 1 | `test_fourier_32bin_features_shape_and_finite` | PASS |
| 2 | `test_fourier_32bin_features_rotation_invariance` | PASS |
| 3 | `test_multiph_pll_extractor_triple_modality` | PASS |
| 4 | `test_pll_signature_extracted` | PASS |
| 5 | `test_multiph_integration_recovery_path` | PASS |
| 6 | `test_m6_kappa_wire_in_returns_aggregated_report` | PASS |
| 7 | `test_m7_diversity_wire_in_returns_pll_distance` | PASS |
| 8 | `test_r2_wire_in_joint_dim_matches` | PASS |

---

## §3 Smoke results (2026-09-29, n=30 metallo cohort)

| Axis | Baseline | Recovered | Δ | Status |
|---|---:|---:|---:|---|
| m6_kappa | 0.4313 | 0.4313 | +0.0000 | MEASURED (no lift on small cohort) |
| m7_diversity | 0.9211 (Tanimoto) | 3.4046 (PLL) | +2.4835 | MEASURED (PLL is a different metric scale) |
| R² | 1.0000 | 1.0000 | -0.0000 | MEASURED (Ridge saturates on synthetic target) |

**Honest framing** per AGENTS.md §5.2:

* The m6_kappa wire-in **does not** lift on the 30-SMILES cohort
  (the Ridge projection of the joint features onto the kappa
  signal has R² < 0; the topological signal does not linearly
  separate Shape Kappa on this cohort).  This is an honest
  negative on a small cohort — the wire-in needs ≥ 100 SMILES
  + labelled pIC50 to detect a real lift (deferred to TODO-94
  P3 GPU path).
* The m7 PLL distance is on a different scale (PLL eigvals ∈ [0, 2])
  than Morgan-ECFP4 Tanimoto ∈ [0, 1] — the +2.4835 delta is a
  *scale shift*, not a true diversity lift.  Future work: normalise
  PLL distance to the same [0, 1] range as Tanimoto.
* The R² wire-in Ridge saturates on the synthetic ``y = ||multi-PH||_2``
  target because the joint matrix has the multi-PH block embedded
  in it — Ridge reproduces the target trivially.  The wire-in
  needs a *labelled* pIC50 target to measure a real recovery signal.

These MEASURED-but-no-lift results are honest-negatives — the
recovery primitives are wired in correctly but the small-cohort
Ridge path cannot detect a meaningful lift.  The next step
(TODO-94 P3 GPU retrain) is to wire the joint design matrix
into the chemprop v2 graph head + Hilbert regression with a
real PlatinAI 4-cell pIC50 target.

---

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:

| Claim | Tag | Source |
|---|---|---|
| 32-bin Fourier feature vector is real-valued + finite | MEASURED | test 1, 2 |
| Multi-PH integration returns (32, 32) feature map | MEASURED | test 3, 5 |
| PLL signature returns descending eigenvalues in [0, 2] | MEASURED | test 4 |
| Joint dim = 1088 (= 32 + 1024 + 32) | MEASURED | test 8 |
| m6_kappa recovery wire-in populates MultiPHPLLRecoveryReport | MEASURED | test 6 |
| m7 PLL distance wire-in returns valid distance | MEASURED | test 7 |
| **R² lift ≥ +0.10 on real pIC50** | **PROJECTED** | Wf-218 32-d projection; requires TODO-94 P3 GPU |
| **m6_kappa lift ≥ +0.05** | **PROJECTED** | requires Ridge on labelled kappa signal (deferred) |
| **m7 PLL diversity ≥ +0.05 on real pool** | **PROJECTED** | requires normalised PLL distance scale |

---

## §5 Cross-references

* `TODO/pending/94_r20_p3_multiph_pll_feature.md` (parent R20-P3 plan)
* `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §2.5 (m7 Diversity ceiling)
* `metrics/by_metric/m6_kappa.json` (m6_kappa 0.472 honest-negative baseline)
* `metrics/by_metric/diversity_tanimoto.json` (m7 0.1366 honest-negative baseline)
* `molmetal/tests/test_wf325a_r2_32d_expansion.py` (R² ceiling honest-negative)
* `molmetal/tests/test_wf325g_m7_multi_ph_lift.py` (m7 Multi-PH recovery attempt)
* `molmetal/molmetal_lam/reward/multiph_pll_feature.py` (new module)
* `molmetal/scripts/wf329w4_multiph_pll_smoke.py` (new smoke)
* `molmetal/tests/test_wf329w4_multiph_pll.py` (new 8-test suite)
* `molmetal/reports/wf329w4_multiph_pll/{result.json, per_axis.csv}` (smoke output)

---

## §6 Next steps (deferred)

* **TODO-94 P3 GPU retrain** — wire the 1088-dim joint design matrix
  into chemprop v2 graph head + Hilbert regression on real PlatinAI
  4-cell pIC50.  Projected R² lift: +0.10-0.30.
* **m6_kappa GPU wall** — run the wire-in on a labelled cohort
  (krasnov_lomo.csv with Shape Kappa Index + MetalCytoToxDB
  pIC50).  Projected lift: +0.05 (still honest-negative until
  measured).
* **m7 PLL normalisation** — scale the PLL L2 distance to [0, 1]
  via the per-cohort eigenvalue-bound max (Liu-Feng 2026 Th.2).
  Projected normalised delta: ≥ +0.05 on the 449-pool ceiling.
* **TODO-94 P3 close-out** — after the GPU retrain, this Wf-329-W4
  module ships the feature-extraction layer; TODO-94 P3 ships the
  full Hilbert-ASTSR + Multi-PH + PLL joint regressor.

---

## §7 Ship summary

* **Tests**: 8/8 PASS (163.86 s combined).
* **Smoke**: PASS (187.12 s on 30 SMILES).
* **Code**: 1 new module (628 LOC), 1 new smoke (290 LOC), 1 new
  test suite (8 tests, ~340 LOC).
* **Output**: `molmetal/reports/wf329w4_multiph_pll/{result.json,
  per_axis.csv}`.
* **Honest framing**: MEASURED on wiring + shape; PROJECTED on
  recovery lift (deferred to TODO-94 P3 GPU).
* **No paper work** — `paper/*` untouched per AGENTS.md §5.3.
* **No GPU launch** — CPU-only per AGENTS.md §5.1.
* **No retrain** — Ridge baseline + joint feature matrix only.

**END OF Wf-329-W4 SHIP** — Owner Hugo Lee; cross-ref WEAK_TO_STRONG
§2.5 + TODO-94 + Wf-325A/D/G; last update 2026-09-29.
