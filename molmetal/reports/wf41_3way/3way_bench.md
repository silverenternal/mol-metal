# Wf-41 Phase 1 — Deflexpressor + Multi-PH + ASTSR 3-way bench

**Dataset**: PlatinAI 5K (synthetic ``qed * qed`` target)  
**n_samples loaded**: 80 (requested 80)  
**Seeds**: [0, 1]  
**Methods**: ['ast', 'hilbert', 'multiph', 'joint', '3way']  
**Epochs (PD iters)**: 200  
**12-d RDKit** -> **64-d Hilbert-padded** + **64-d Multi-PH** (n_bins=8) + **16-d PLL** -> 3-way design matrix is 92-d.


## Per-method R² (mean ± std across seeds)

| Method | R² mean ± std | Lift over AST | Wall (s) | n_features |
|---|---|---|---|---|
| ast | -1.1717 ± 0.0678 | +0.0000 | 0.277 | 12 |
| hilbert | -0.1423 ± 0.1760 | +1.0295 | 0.030 | 64 |
| multiph | -1.8704 ± 0.4402 | -0.6986 | 0.001 | 64 |
| joint | -0.3155 ± 0.0974 | +0.8562 | 0.012 | 76 |
| 3way | -0.3215 ± 0.1150 | +0.8502 | 0.001 | 92 |

## Per-seed R²

| Method | seed 0 | seed 1 |
|---|---|---|
| ast | -1.1039 | -1.2396 |
| hilbert | -0.3182 | +0.0337 |
| multiph | -1.4301 | -2.3106 |
| joint | -0.2182 | -0.4129 |
| 3way | -0.2066 | -0.4365 |

## Per-seed wall time (s)

| Method | seed 0 | seed 1 |
|---|---|---|
| ast | 0.281 | 0.273 |
| hilbert | 0.031 | 0.028 |
| multiph | 0.001 | 0.001 |
| joint | 0.015 | 0.009 |
| 3way | 0.001 | 0.001 |

## Honest framing

* ``ast`` is the R17 ship Deflexpressor baseline — evolutionary LambdaASTSR on 12-d RDKit descriptors.
* ``hilbert`` lifts 12-d to 64-d via pairwise interactions and fits LambdaASTSRHilbert.
* ``multiph`` flattens the 2-PH feature map to 64-d (n_bins=8) and fits a closed-form ridge.
* ``joint`` runs JointHilbertOptimizer on the (ASTSR + Multi-PH) concat — the Wf-21 P5 baseline.
* ``3way`` is the new Wf-41 P1 integration: ASTSR + Multi-PH + PLL streams concat → ridge.
* The PlatinAI 5K target is **synthetic** (``qed * qed``); the comparison is on the *relative* lift between regressors, not on absolute R².
* Wall-time is machine-dependent; speedup is the intended *relative* metric.


**Run timestamp (UTC):** 2026-09-26T05:44:32.653335+00:00

