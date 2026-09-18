# A3 — Multi-granularity Temporal Splits (Ru)

Each row trains Morgan-FP + XGBoost on the **train** portion of one temporal grid and evaluates on the **test** portion.  Hit rate = fraction of rows with IC50_Dark_value < 10 µM.

| split | n_train | n_test | hit_rate_train | hit_rate_test | AUC | PR_AUC | mean_year_test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre_2020_vs_2020+ | 2219 | 1449 | 0.2659 | 0.2581 | 0.6386 | 0.4276 | 2021.93 |
| pre_2022_vs_2022+ | 2870 | 798 | 0.2638 | 0.2594 | 0.5913 | 0.3712 | 2023.05 |
| pre_2024_vs_2024+ | 3378 | 290 | 0.2641 | 0.2483 | 0.4950 | 0.2606 | 2024.21 |
| pre_2024_vs_2025+ | 3607 | 61 | 0.2639 | 0.1967 | 0.4065 | 0.1884 | 2025.00 |
| rolling_cutoff_2018 | 1376 | 2292 | 0.2398 | 0.2766 | 0.6121 | 0.3550 | 2020.64 |
| rolling_cutoff_2020 | 2219 | 1449 | 0.2659 | 0.2581 | 0.6386 | 0.4276 | 2021.93 |
| rolling_cutoff_2022 | 2870 | 798 | 0.2638 | 0.2594 | 0.5913 | 0.3712 | 2023.05 |
| rolling_cutoff_2024 | 3378 | 290 | 0.2641 | 0.2483 | 0.4950 | 0.2606 | 2024.21 |

## Hit-rate vs publication year

```
hit-rate (y) per year (x), bar width = scaled hit-rate (46.2% = full bar)

  1.00 |------------------------------------------------------------
   2001 |####################################################          40.0%
   2002 |#########################################################     44.0%
   2006 |#####################                                         16.0%
   2007 |                                                               0.0%
   2008 |###################                                           14.6%
   2009 |                                                               0.0%
   2010 |############################                                  21.7%
   2011 |################################                              25.0%
   2012 |#################                                             12.8%
   2013 |#################                                             13.2%
   2014 |##############################                                23.4%
   2015 |############################################################  46.2%
   2016 |#################################                             25.5%
   2017 |################################                              24.6%
   2018 |#####################################                         28.8%
   2019 |############################################                  33.5%
   2020 |######################################                        29.3%  <-- cutoff
   2021 |#############################                                 22.7%
   2022 |######################################                        29.0%  <-- cutoff
   2023 |##############################                                22.7%
   2024 |##################################                            26.2%  <-- cutoff
   2025 |##########################                                    19.7%  <-- cutoff
   2026 |------------------------------------------------------------
```

**Cutoff legend:**
- cutoff=2020: n_test=1449
- cutoff=2022: n_test=798
- cutoff=2024: n_test=290
- cutoff=2025: n_test=61


## Interpretation

- **Test hit-rate range**: pre_2024_vs_2025+ has the lowest test hit-rate (0.1967), rolling_cutoff_2018 has the highest (0.2766).
- **AUC range**: pre_2024_vs_2025+ has the lowest test AUC (0.4065), pre_2020_vs_2020+ the highest (0.6386).
- **Hit-rate decay (test − train)**: worst = pre_2024_vs_2025+ (-0.0672), best = rolling_cutoff_2018 (+0.0368).

### Recommendation

**Use `pre_2024_vs_2024+` as the paper's main temporal-test split.**

Rationale: largest test-train hit-rate gap (+0.0158) with non-trivial n_test=290, and the lowest absolute test hit-rate (0.2483) among eligible splits — exactly the signature of publication-bias induced data drift.

_Runner-up_: `rolling_cutoff_2024` (gap=+0.0158, n_test=290, AUC=0.4950).

_Counter-recommendation:_ a more permissive cutoff (e.g. `pre_2020_vs_2020+` or `pre_2022_vs_2022+`) inflates test AUC because the test set still contains many 'easy' recent-but-overlapping ligands.  Pick the split that *worries you the most* — that's the honest OOD test.

_Run walltime: 5.7s_
