# Corpus and split contamination

Generated 2026-10-19T11:45:22Z from `metrics/by_metric/*.json`.

Cross-fold rate = share of test rows whose key value also appears in that fold's training rows.

| Metric | Value | Source | Tag |
|---|---|---|---|
| Rows in source corpus | 43268 | msrs_data_card.json:/corpus/n_rows_total | MEASURED |
| Potency rows | 26801 | msrs_data_card.json:/corpus/potency_rows | MEASURED |
| Model rows after filter | 21404 | msrs_data_card.json:/corpus/model_rows | MEASURED |
| Distinct metals | 5 | msrs_data_card.json:/corpus/n_metals | MEASURED |
| Distinct ligands | 5861 | msrs_data_card.json:/corpus/n_ligands | MEASURED |
| Distinct cell lines | 711 | msrs_data_card.json:/corpus/n_cell_lines | MEASURED |
| Distinct DOIs | 1784 | msrs_data_card.json:/corpus/n_dois | MEASURED |
| Rows with metal in SMILES | 0 | msrs_data_card.json:/corpus/metal_in_smiles_rows | MEASURED |
| Right-censored potency rows dropped | 5015 | msrs_data_card.json:/corpus/potency_censoring_counts/right | MEASURED |
| Repeated keys | 304 | msrs_data_card.json:/duplicates/n_keys_repeated | MEASURED |
| Rows in repeated keys | 663 | msrs_data_card.json:/duplicates/n_rows_in_repeated_keys | MEASURED |
| ligand: ligand seen in train | 0.00% | msrs_data_card.json:/protocols/ligand/ligand_cross_fold/rate | MEASURED |
| ligand: DOI seen in train | 89.43% | msrs_data_card.json:/protocols/ligand/doi_cross_fold/rate | MEASURED |
| ligand: cell line seen in train | 98.40% | msrs_data_card.json:/protocols/ligand/cell_cross_fold/rate | MEASURED |
| doi: ligand seen in train | 15.43% | msrs_data_card.json:/protocols/doi/ligand_cross_fold/rate | MEASURED |
| doi: DOI seen in train | 0.00% | msrs_data_card.json:/protocols/doi/doi_cross_fold/rate | MEASURED |
| doi: cell line seen in train | 96.04% | msrs_data_card.json:/protocols/doi/cell_cross_fold/rate | MEASURED |
| scaffold: ligand seen in train | 0.00% | msrs_data_card.json:/protocols/scaffold/ligand_cross_fold/rate | MEASURED |
| scaffold: DOI seen in train | 65.58% | msrs_data_card.json:/protocols/scaffold/doi_cross_fold/rate | MEASURED |
| scaffold: cell line seen in train | 97.76% | msrs_data_card.json:/protocols/scaffold/cell_cross_fold/rate | MEASURED |
| year: ligand seen in train | 14.80% | msrs_data_card.json:/protocols/year/ligand_cross_fold/rate | MEASURED |
| year: DOI seen in train | 0.00% | msrs_data_card.json:/protocols/year/doi_cross_fold/rate | MEASURED |
| year: cell line seen in train | 97.41% | msrs_data_card.json:/protocols/year/cell_cross_fold/rate | MEASURED |
