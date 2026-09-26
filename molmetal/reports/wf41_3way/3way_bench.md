# Wf-41 Phase 1 — Deflexpressor + Multi-PH + ASTSR 3-way bench

**Dataset**: PlatinAI 5K (synthetic ``qed * qed`` target)  
**n_samples loaded**: 100 (requested 100)  
**Seeds**: [0, 1, 2]  
**Methods**: ['ast', 'hilbert', 'multiph', 'joint', '3way']  
**Epochs (PD iters)**: 200  
**12-d RDKit** -> **64-d Hilbert-padded** + **64-d Multi-PH** (n_bins=8) + **16-d PLL** -> 3-way design matrix is 92-d.


## Per-method R² (mean ± std across seeds)

| Method | R² mean ± std | Lift over AST | Wall (s) | n_features |
|---|---|---|---|---|
| ast | -1.1119 ± 0.0300 | +0.0000 | 0.401 | 12 |
| hilbert | -0.2829 ± 0.1217 | +0.8290 | 0.029 | 64 |
| multiph | -4.7818 ± 2.9867 | -3.6699 | 0.001 | 64 |
| joint | -0.2929 ± 0.3009 | +0.8190 | 0.011 | 76 |
| 3way | -0.3595 ± 0.2806 | +0.7524 | 0.001 | 92 |

## Per-seed R²

| Method | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| ast | -1.0695 | -1.1298 | -1.1363 |
| hilbert | -0.1110 | -0.3624 | -0.3752 |
| multiph | -1.3592 | -4.3493 | -8.6367 |
| joint | +0.1154 | -0.3934 | -0.6007 |
| 3way | +0.0350 | -0.5187 | -0.5947 |

## Per-seed wall time (s)

| Method | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| ast | 0.543 | 0.322 | 0.337 |
| hilbert | 0.032 | 0.028 | 0.028 |
| multiph | 0.001 | 0.001 | 0.001 |
| joint | 0.016 | 0.010 | 0.009 |
| 3way | 0.001 | 0.001 | 0.001 |

## Honest framing

* ``ast`` is the R17 ship Deflexpressor baseline — evolutionary LambdaASTSR on 12-d RDKit descriptors.
* ``hilbert`` lifts 12-d to 64-d via pairwise interactions and fits LambdaASTSRHilbert.
* ``multiph`` flattens the 2-PH feature map to 64-d (n_bins=8) and fits a closed-form ridge.
* ``joint`` runs JointHilbertOptimizer on the (ASTSR + Multi-PH) concat — the Wf-21 P5 baseline.
* ``3way`` is the new Wf-41 P1 integration: ASTSR + Multi-PH + PLL streams concat → ridge.
* The PlatinAI 5K target is **synthetic** (``qed * qed``); the comparison is on the *relative* lift between regressors, not on absolute R².
* Wall-time is machine-dependent; speedup is the intended *relative* metric.


**Run timestamp (UTC):** 2026-09-26T05:53:24.040046+00:00

