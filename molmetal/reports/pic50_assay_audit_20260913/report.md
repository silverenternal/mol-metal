# Ru pIC50 assay audit

| Quantity | Count |
|---|---:|
| all_rows | 26801 |
| ru_rows | 19135 |
| ru_positive_numeric_dark_rows | 19135 |
| ru_valid_structure_positive_rows | 19135 |
| ru_unique_structures | 4833 |
| ru_numeric_but_censored_rows | 3864 |
| legacy_first_per_molecule_censored | 857 |
| legacy_sample2000_censored | 356 |
| molecules_measured_in_multiple_cell_lines | 3984 |
| molecules_measured_at_multiple_exposures | 395 |
| molecules_with_multiple_counterions | 96 |
| molecules_with_multiple_oxidation_states | 1 |
| molecules_with_pic50_span_gt1 | 711 |
| first_last_label_change_gt0_5 | 1142 |
| uncensored_condition_known_rows | 15091 |

## Largest uncensored cell-line/exposure cohorts
| Cell line | Hours | Rows | Unique structures | DOIs |
|---|---:|---:|---:|---:|
| HeLa | 48.0 | 717 | 695 | 232 |
| A549 | 48.0 | 652 | 634 | 215 |
| MCF-7 | 48.0 | 562 | 550 | 200 |
| HepG2 | 48.0 | 404 | 383 | 134 |
| HeLa | 24.0 | 366 | 361 | 106 |
| A2780 | 72.0 | 369 | 351 | 107 |
| MCF-7 | 72.0 | 356 | 339 | 109 |
| A549 | 24.0 | 324 | 314 | 104 |
| MCF-7 | 24.0 | 290 | 289 | 104 |
| MDA-MB-231 | 48.0 | 292 | 288 | 84 |

## Interpretation
The old first-row-per-SMILES target can depend on CSV ordering and assay context. Censored bounds cannot be treated as exact IC50 regression labels.

Censor detection uses explicit comparison symbols in the raw dark-assay field; ambiguous free text still needs separate curation.
Compound grouping mirrors legacy canonical ligand SMILES, not a claim that counterion/oxidation/assay context is interchangeable.
Within-compound variation includes different cell lines/exposures and is not labelled replicate measurement noise.
Eligible rows are an audit export, not a train/test split; do not randomly split repeated compounds or tune cohorts using test performance.
This does not retroactively change any trained checkpoint or metric. Retraining needs a new protocol and new output prefix.
