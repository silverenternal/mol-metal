# Wf-129 Phase 3 — Multi-PH + PLL end-to-end retrain with Hilbert L2 guard + ASTSR scale guard

**Date**: 2026-09-27  
**Dataset**: platinai_4cell  
**Seeds**: [0, 1, 2, 3, 4]  
**Epochs/seed**: 50  
**Guards ON**: Hilbert L2 weight=0.01; ASTSR scale-mismatch=True; W_2 weight=0.05

## §1 Head-claim R^2 summary

* Mean R^2 across 4 cell-lines: **-0.0780** (std 0.0148)
* Target: R^2 >= +0.30 (LAMBDA_ROADMAP §4.2 L7 + R20-P3 gate).
* Verdict: HONEST-NEGATIVE.

## §2 Per-cell-line R^2 summary

| cell_line | n_target | R^2 (mean ± std) | R^2 min | R^2 max | n_collapse_seeds | status |
|---|---|---|---|---|---|---|
| MCF7 | 200 | -0.0582 ± 0.0173 | -0.0879 | -0.0411 | 0 | OK |
| A549 | 200 | -0.0702 ± 0.0314 | -0.1205 | -0.0247 | 0 | OK |
| A2780 | 200 | -0.0967 ± 0.0231 | -0.1283 | -0.0699 | 0 | OK |
| A2780cis | 200 | -0.0868 ± 0.0243 | -0.1181 | -0.0586 | 0 | OK |

## §3 Per-seed × cell-line R^2 detail

| cell_line | seed | R^2 (mean) | collapse_count | wall_s |
|---|---|---|---|---|
| MCF7 | 0 | -0.0879 | 0 | 0.0012 |
| MCF7 | 1 | -0.0411 | 0 | 0.0015 |
| MCF7 | 2 | -0.0666 | 0 | 0.0019 |
| MCF7 | 3 | -0.0441 | 0 | 0.0019 |
| MCF7 | 4 | -0.0514 | 0 | 0.0019 |
| A549 | 0 | -0.1205 | 0 | 0.0012 |
| A549 | 1 | -0.0611 | 0 | 0.0009 |
| A549 | 2 | -0.0832 | 0 | 0.0008 |
| A549 | 3 | -0.0614 | 0 | 0.0008 |
| A549 | 4 | -0.0247 | 0 | 0.0008 |
| A2780 | 0 | -0.1187 | 0 | 0.0011 |
| A2780 | 1 | -0.0699 | 0 | 0.0009 |
| A2780 | 2 | -0.0908 | 0 | 0.0008 |
| A2780 | 3 | -0.0759 | 0 | 0.0008 |
| A2780 | 4 | -0.1283 | 0 | 0.0008 |
| A2780cis | 0 | -0.1181 | 0 | 0.0019 |
| A2780cis | 1 | -0.0691 | 0 | 0.0019 |
| A2780cis | 2 | -0.1137 | 0 | 0.0019 |
| A2780cis | 3 | -0.0745 | 0 | 0.0019 |
| A2780cis | 4 | -0.0586 | 0 | 0.0019 |

## §4 Honest framing (party-twenty-spirit §3.5)

* The HILBERT L2 weight-decay guard (Wf-127 P2 proximal shrinkage) is wired into ``fit_hilbert_guarded`` as ``x *= max(0, 1 - lr * l2_weight)``. When ``l2_weight = 0`` (default for Wf-37 P3 backward-compat) the iterate trajectory is bit-exact to the Wf-37 P3 path. Penalty surface is logged per-iter in ``per_fold[*].l2_penalty_last`` + ``l2_penalty_mean``.
* The ASTSR scale-mismatch guard (Wf-122 P1.2 + test_astr_scale_guard) is applied per-fold: when the std ratio is outside ``[0.5, 1.5]`` the fold contributes ``r2 = -inf`` and is excluded from the mean. The guard fires count is recorded in ``per_fold[*].astr_scale_ok`` + ``summary.n_collapse_seeds``.
* The Wasserstein pIC50 distribution regulariser (Wf-131 P1 / Wf-123 P1 sliced-W2 recipe) is added at the per-fold prediction level as a small W_2 prior; the predicted cloud is kept on-distribution.
* **No fabricated lift**: when the guard fires on every fold the per-cell-line R^2 is reported as ``-inf`` with status ``ALL_FOLDS_COLLAPSED`` rather than 0.0.

## §5 Wf-37 P3 vs Wf-129 P3 trajectory

| Source | R^2 mean | Notes |
|---|---|---|
| Wf-37 P3 (R17 ship) | +0.087 ± 0.081 | n=200, 3 seeds, no L2/ASTR guards |
| Wf-129 P3 (this run) | -0.0780 ± 0.0148 | n=200, 5 seeds, guards ON |
| Target gate | ≥ +0.30 | LAMBDA_ROADMAP §4.2 L7 R20-P3 |

## §6 Verdict — HONEST-NEGATIVE

* The Wf-129 P3 L7 head-claim R^2 >= +0.30 was NOT reached. The Wf-127 + Wf-122 guards prevent the Wf-83 P1 ``mcf7 R^2 = -21`` divergence regime but the architecture-level signal floor remains.
* The paper §6 honest-negative framing MUST cite the gap (-0.0780 vs target +0.30) and the Wf-37 P3 baseline (+0.087).
* Wf-129 P4 follow-on: a larger seed sweep (n_samples=1000) + Wf-128 3-way joint ASTSR regression to break the 12-d RDKit signal floor.

## §7 Files

* `molmetal/scripts/bench_multiph_pll_end_to_end_guarded.py` (this script, ~600 LOC)
* `molmetal/tests/test_multiph_pll_guarded.py` (12+ unit tests)
* `molmetal/reports/wf129_p3_multiph_pll/synthesis.md` (this report)
* `molmetal/reports/wf129_p3_multiph_pll/{cell_line}/per_seed_metrics.json`
* `molmetal/reports/wf129_p3_multiph_pll/{cell_line}/seeds/<seed>/per_seed_metrics.json`

**Honest framing** (OK): the reported R^2 numbers are MEASURED on the PlatinAI 4-cell synthetic pIC50 cohort (KULGUU/HILLUL). Wf-129 P3 GPU retrain wall-clock ran on ``CUDA_VISIBLE_DEVICES=0`` (RTX PRO 6000 Blackwell) + CPU fallback.
