# Metric 19 — Multi-metal leave-one-metal-out (LOMO) transfer AUC

**Date:** 2026-09-25  
**Metals:** Ru, Ir, Os, Rh, Re  
**Seeds:** [0, 1, 2]  
**Model:** Multi-task D-MPNN (BCE activity + MSE pIC50, alpha=0.5)
**Epochs:** 30, lr=0.001, batch=64

## Goal

Train on **all rows from 4 of the 5 precious metals** in the LOMO subset, evaluate on the **held-out metal** as an OOD transfer test.  Repeat for every metal × seed combination and report the **mean ± std across the 15 runs**.

## Per-fold results

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5706 | 0.0032 | 3 |
| Ir | 0.6176 | 0.0068 | 3 |
| Os | 0.6113 | 0.0235 | 3 |
| Rh | 0.3677 | 0.0184 | 3 |
| Re | 0.5059 | 0.1281 | 3 |

## Aggregate

- **Mean LOMO test AUC** = **0.5346** ± 0.1096  (n=15 runs across 5 folds)
- Gate threshold (TODO/METRICS_TARGETS.md §2 / Metric 19): mean ≥ 0.70
- **Verdict:** BELOW GATE — mean LOMO AUC < 0.70 (transfer is weak)

## Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1442 | 3668 | 0.263 | 0.6923 | 0.5721 | 249.0 |
| Ru | 1 | 1442 | 3668 | 0.263 | 0.7278 | 0.5662 | 196.6 |
| Ru | 2 | 1442 | 3668 | 0.263 | 0.7208 | 0.5736 | 166.8 |
| Ir | 0 | 3626 | 1242 | 0.448 | 0.7068 | 0.6120 | 418.8 |
| Ir | 1 | 3626 | 1242 | 0.448 | 0.7272 | 0.6137 | 338.1 |
| Ir | 2 | 3626 | 1242 | 0.448 | 0.7446 | 0.6271 | 271.2 |
| Os | 0 | 4665 | 87 | 0.437 | 0.7528 | 0.6445 | 414.6 |
| Os | 1 | 4665 | 87 | 0.437 | 0.7809 | 0.5967 | 402.4 |
| Os | 2 | 4665 | 87 | 0.437 | 0.7595 | 0.5929 | 477.7 |
| Rh | 0 | 4646 | 108 | 0.324 | 0.7150 | 0.3724 | 362.8 |
| Rh | 1 | 4646 | 108 | 0.324 | 0.7831 | 0.3875 | 552.8 |
| Rh | 2 | 4646 | 108 | 0.324 | 0.7314 | 0.3432 | 507.0 |
| Re | 0 | 4594 | 166 | 0.163 | 0.6797 | 0.3609 | 431.3 |
| Re | 1 | 4594 | 166 | 0.163 | 0.7634 | 0.4843 | 351.9 |
| Re | 2 | 4594 | 166 | 0.163 | 0.7509 | 0.6725 | 429.9 |

## Honest caveats

* The smallest folds (Os/Re/Rh) have few test rows (≤166) so per-metal AUC is
  noisy; the aggregate mean is the most reliable number.
* We do **not** include the metal token as an input feature — the model can only
  transfer if the chemistry learned on the other 4 metals generalises to the
  held-out metal's ligands.  This is a *strict* OOD by metal identity.
* pIC50 is regressed jointly with activity (alpha=0.5) as in Metric 1 for
  comparability.  Splitting strategy here is **metal**, not scaffold, so the
  scaffold split is *within* the train pool only.
