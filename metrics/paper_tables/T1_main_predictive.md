# Main predictive metrics

Generated 2026-10-19T11:52:15Z from `metrics/by_metric/*.json`.

| Metric | Unit | Value | Source | Tag |
|---|---|---|---|---|
| Canonical R2, stack (5-seed mean) | R2 | 0.5828 | msrs_seed_stability.json:/splits/canonical_grouped/arms/stack_foldwise/r2/mean | MEASURED |
| Canonical R2, stack (5-seed SD) | R2 | 0.0102 | msrs_seed_stability.json:/splits/canonical_grouped/arms/stack_foldwise/r2/std | MEASURED |
| Canonical R2, LightGBM tree (5-seed mean) | R2 | 0.5555 | msrs_seed_stability.json:/splits/canonical_grouped/arms/lgb_tree/r2/mean | MEASURED |
| Canonical R2, D-MPNN alone (5-seed mean) | R2 | 0.5032 | msrs_seed_stability.json:/splits/canonical_grouped/arms/gnn_metal_state/r2/mean | MEASURED |
| R2, ligand protocol | R2 | 0.5556 | msrs_generalization.json:/splits/ligand/arms/ligand+both/r2 | MEASURED |
| R2, DOI protocol | R2 | 0.3321 | msrs_generalization.json:/splits/doi/arms/ligand+both/r2 | MEASURED |
| R2, scaffold protocol | R2 | 0.4503 | msrs_generalization.json:/splits/scaffold/arms/ligand+both/r2 | MEASURED |
| R2, year protocol | R2 | 0.3739 | msrs_generalization.json:/splits/year/arms/ligand+both/r2 | MEASURED |
| dR2, DOI minus ligand | dR2 | 0.2235 | msrs_generalization.json:/delta_r2_doi_vs_ligand | MEASURED |
| Delta sign accuracy, DOI at |d|>=0.2091 | accuracy | 0.6193 | msrs_delta_metrics.json:/protocols/doi/thresholds/0.2091/sign_accuracy | MEASURED |
| MAE(delta), DOI | log10 IC50 | 0.5089 | msrs_delta_metrics.json:/protocols/doi/thresholds/0.2091/mae_delta | MEASURED |
| Decision regret, DOI full coverage | log10 IC50 | 0.1168 | msrs_decision_curve.json:/protocols/doi/full_coverage/model/mean_regret | MEASURED |
| Top-1 accuracy, DOI full coverage | accuracy | 0.5600 | msrs_decision_curve.json:/protocols/doi/full_coverage/model/top1_accuracy | MEASURED |
| LOMO AUC (5-fold x 5-seed) | ROC-AUC | 0.5575 | metal_lomo_auc_wasserstein.json:/value | MEASURED |
