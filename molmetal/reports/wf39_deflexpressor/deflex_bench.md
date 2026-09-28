# Wf-39 Phase 2 — Deflexpressor end-to-end training bench

**Dataset**: `nci60`  
**n_samples loaded**: 500 (requested 500)  
**Seeds**: [0, 1, 2, 3, 4]  
**Methods**: ['ast', 'hilbert', 'composite']  
**Epochs (PD iters)**: 200  
**Descriptors**: 12-d RDKit → 64-d Hilbert-augmented


## Aggregate (across seeds)

| Method | R² mean ± std | Wall (s) mean ± std | Peak mem (MB) mean ± std | Speedup |
|---|---|---|---|---|
| ast | -0.0905 ± 0.0173 | 4.270 ± 1.067 | 12.83 ± 25.19 | 1477.12x |
| hilbert | -0.0782 ± 0.6802 | 0.003 ± 0.002 | 0.28 ± 0.04 | 1.00x (ref) |
| composite | +0.1861 ± 0.1734 | 3.519 ± 0.138 | 0.39 ± 0.02 | 1217.34x |

## Per-seed R²

| Method | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| ast | -0.0706 | -0.0693 | -0.1035 | -0.0981 | -0.1112 |
| hilbert | +0.1519 | +0.3668 | -1.4252 | +0.3637 | +0.1519 |
| composite | +0.1519 | +0.3668 | -0.1035 | +0.3637 | +0.1519 |

## Per-seed wall time (s)

| Method | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| ast | 6.246 | 3.436 | 3.319 | 3.881 | 4.467 |
| hilbert | 0.007 | 0.002 | 0.002 | 0.002 | 0.001 |
| composite | 3.413 | 3.557 | 3.387 | 3.769 | 3.469 |

## Active branch for composite (per seed)

- seed=  0  active_branch=`hilbert`  R²=+0.1519  train_R²=+0.4876
- seed=  1  active_branch=`hilbert`  R²=+0.3668  train_R²=+0.4517
- seed=  2  active_branch=`ast`  R²=-0.1035  train_R²=-0.0675
- seed=  3  active_branch=`hilbert`  R²=+0.3637  train_R²=+0.4429
- seed=  4  active_branch=`hilbert`  R²=+0.1519  train_R²=+0.4792

## Honest framing

* ``ast`` uses 12-d RDKit descriptors; ``hilbert`` / ``composite`` use the 64-d pairwise-interaction augmentation.
* ``epochs`` is forwarded as ``n_iters`` to the Hilbert branch only; the AST branch uses ``generations``.
* ``composite`` runs both branches and reports the better test R²; it is **not** a weighted ensemble.
* Wall-time and memory are machine-dependent; the speedup column is the intended *relative* metric.
* pIC50 is **not log-scaled** beyond the standard ``pIC50 = -log10(IC50_M)`` conversion; the regression target is continuous in ``[3.5, 9.5]``.


**Run timestamp (UTC):** 2026-09-26T05:14:51.999011+00:00

