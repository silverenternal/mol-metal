# A2 — Counter-ion Ablation (Ru)

| config | n_features | random AUC | temporal AUC | n_train (R / T) | n_test (R / T) |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_counterion | 2048 | 0.8878 | 0.6394 | 1840/1508 | 230/206 |
| counterion_features | 2054 | 0.8848 | 0.5631 | 1840/1508 | 230/206 |
| counterion_concat_smiles | 2048 | 0.8839 | 0.5988 | 1840/1508 | 230/206 |

## Interpretation

- `no_counterion` (Krasnov-baseline): random AUC = 0.8878, temporal AUC = 0.6394.
- `counterion_features` (5 added dims): random AUC = 0.8848 (delta -0.0030 vs no_counterion), temporal AUC = 0.5631 (delta -0.0763).
- `counterion_concat_smiles` (multi-component SMILES): random AUC = 0.8839 (delta -0.0040 vs no_counterion), temporal AUC = 0.5988 (delta -0.0406).

Explicit counter-ion features give ~0 change on random-split AUC — the counter-ion is mostly redundant with the metal + ligand signal.
On the **temporal** split the picture inverts: explicit features **hurt** temporal AUC, suggesting the counter-ion is a *spurious* leakage channel (some counter-ions are publication-period specific, e.g. trends in anion choice track the rise of OTf / PF6 across eras).

_Run walltime: 8.4s_
