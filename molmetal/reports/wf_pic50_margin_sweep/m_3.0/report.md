# WF-Extra-1 — D-MPNN retrain (HeLa/48h/dark, censor-aware)

Cohort: **1451 formulations / 715 scaffolds** (censored: 251, exact: 1200).

Loss: `MSE(exact) + hinge-margin(censored, margin=3.0)`.

| Seed | n_train/val/test | Test RMSE | Test Pearson r | n_exact_test | n_censored_test | bound_aware_accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 1173/152/126 | 0.7759 | -0.0459 | 108 | 18 | 1.000 |
| 0 | 1150/162/139 | 0.6768 | 0.3900 | 106 | 33 | 0.970 |
| 1234 | 1181/138/132 | 0.7191 | 0.2151 | 116 | 16 | 1.000 |

Aggregate Pearson r: **0.1864 ± 0.1791** (target band 0.5–0.7, historical D-MPNN 0.407, ridge ref 0.5723).

## Honest framing

* **MEASURED**: cohort sizes, scaffold counts, censoring fractions, GPU forward/backward.
* **PROJECTED** (not measured in this run): final Pearson r for a 50-epoch full sweep — smoke run only validates GPU wiring, loss convergence, no NaN.

