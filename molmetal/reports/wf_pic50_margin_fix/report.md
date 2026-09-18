# WF-Extra-1 — D-MPNN retrain (HeLa/48h/dark, censor-aware)

Cohort: **1451 formulations / 715 scaffolds** (censored: 251, exact: 1200).

Loss: `MSE(exact) + hinge-margin(censored, margin=1.0)`.

| Seed | n_train/val/test | Test RMSE | Test Pearson r | n_exact_test | n_censored_test | bound_aware_accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 1173/152/126 | 0.7497 | -0.0491 | 108 | 18 | 1.000 |
| 0 | 1150/162/139 | 0.6380 | 0.4841 | 106 | 33 | 1.000 |
| 1234 | 1181/138/132 | 0.7274 | 0.1090 | 116 | 16 | 1.000 |

Aggregate Pearson r: **0.1814 ± 0.2236** (target band 0.5–0.7, historical D-MPNN 0.407, ridge ref 0.5723).

## Honest framing

* **MEASURED**: cohort sizes, scaffold counts, censoring fractions, GPU forward/backward.
* **PROJECTED** (not measured in this run): final Pearson r for a 50-epoch full sweep — smoke run only validates GPU wiring, loss convergence, no NaN.

