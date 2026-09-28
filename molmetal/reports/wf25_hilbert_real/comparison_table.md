# Wf-25 Phase 1 — Hilbert-ASTSR real end-to-end evaluation

**Dataset:** PlatinAI 5K + MetalCytoToxDB Ru subset (n=500)
**Folds:** 5  |  **Population dim:** 8
**Features:** mol_wt, mol_logp, tpsa, hba, hbd, rot_bonds, ring_count, frac_sp3

**Parse stats:** parsed=500, failed=0, target_rows=500

## Side-by-side comparison

| Backend | R² mean ± std | Wall time (s) | Speedup |
|---|---|---|---|
| Hilbert (composite) | -0.3527 ± 0.0735 | 0.0022 | 1278.49x |
| Evolutionary (LambdaASTSR) | -0.0157 ± 0.0102 | 2.7558 | 1.00x (ref) |

## Per-fold detail

| Fold | n_train | n_test | Hilbert R² | Hilbert wall (s) | Evol R² | Evol wall (s) |
|---|---|---|---|---|---|---|
| 0 | 400 | 100 | -0.2493 | 0.0022 | -0.0033 | 3.0722 |
| 1 | 400 | 100 | -0.3084 | 0.0021 | -0.0284 | 2.6257 |
| 2 | 400 | 100 | -0.3889 | 0.0021 | -0.0141 | 2.9080 |
| 3 | 400 | 100 | -0.3499 | 0.0022 | -0.0062 | 2.5780 |
| 4 | 400 | 100 | -0.4670 | 0.0021 | -0.0264 | 2.5949 |

## Honest framing

* Features are 8 RDKit descriptors, not the F5 schema (which requires a Pareto-front computation unavailable here).
* Activity is binary; the target is ``activity + N(0, 0.05)`` — the regressor sees a noisy continuous signal, R² reflects how much of that binary signal the ASTSR-class regressor recovers.
* Wall-time is machine-dependent; the speedup column is the intended *relative* metric.
