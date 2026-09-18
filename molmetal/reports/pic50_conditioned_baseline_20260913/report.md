# Conditioned Ru pIC50 baseline

HeLa, 48h, dark; 702 formulations / 383 scaffolds.

| Seed | Train/val/test rows | Ridge RMSE | Mean baseline RMSE | Pearson |
|---|---|---:|---:|---:|
| 42 | 544/64/94 | 0.5972 | 0.6489 | 0.4774 |
| 0 | 527/70/105 | 0.6116 | 0.7601 | 0.6055 |
| 1234 | 551/71/80 | 0.4150 | 0.5274 | 0.6341 |

No comparison to old pooled/censored random-split metrics: targets and splits differ.
Scaffold proportions are group counts, not guaranteed row proportions; no test-driven rebalancing.
Known assay context is limited to available metadata; cross-study measurement differences remain.
This is one Ru/cell/time cohort; other metals, cell lines, temporal or publication holdouts remain unvalidated.
