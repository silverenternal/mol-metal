# Four-protocol generalization matrix (arm ligand+both)

Generated 2026-10-19T11:45:22Z from `metrics/by_metric/*.json`.

Each protocol holds out a different unit: ligand / DOI / Bemis-Murcko scaffold / publication year.

| Metric | Groups | Rows scored | R2 | 95% CI | dR2 vs ligand | Source | Tag |
|---|---|---|---|---|---|---|---|
| ligand | 5861 | 21404 | 0.5556 | [0.5358, 0.5744] | 0.0000 | msrs_generalization.json:/splits/ligand/n_groups; msrs_generalization.json:/splits/ligand/n_rows_evaluated; msrs_generalization.json:/splits/ligand/arms/ligand+both/r2; msrs_generalization.json:/splits/ligand/arms/ligand+both/bootstrap_r2/ci95; msrs_generalization.json:/splits/ligand/arms/ligand+both/r2 - /splits/ligand/arms/ligand+both/r2 | MEASURED |
| doi | 1784 | 21404 | 0.3321 | [0.2806, 0.3820] | -0.2235 | msrs_generalization.json:/splits/doi/n_groups; msrs_generalization.json:/splits/doi/n_rows_evaluated; msrs_generalization.json:/splits/doi/arms/ligand+both/r2; msrs_generalization.json:/splits/doi/arms/ligand+both/bootstrap_r2/ci95; msrs_generalization.json:/splits/doi/arms/ligand+both/r2 - /splits/ligand/arms/ligand+both/r2 | MEASURED |
| scaffold | 2636 | 21404 | 0.4503 | [0.4102, 0.4884] | -0.1053 | msrs_generalization.json:/splits/scaffold/n_groups; msrs_generalization.json:/splits/scaffold/n_rows_evaluated; msrs_generalization.json:/splits/scaffold/arms/ligand+both/r2; msrs_generalization.json:/splits/scaffold/arms/ligand+both/bootstrap_r2/ci95; msrs_generalization.json:/splits/scaffold/arms/ligand+both/r2 - /splits/ligand/arms/ligand+both/r2 | MEASURED |
| year | 16 | 6499 | 0.3739 | [0.3230, 0.4175] | -0.1817 | msrs_generalization.json:/splits/year/n_groups; msrs_generalization.json:/splits/year/n_rows_evaluated; msrs_generalization.json:/splits/year/arms/ligand+both/r2; msrs_generalization.json:/splits/year/arms/ligand+both/bootstrap_r2/ci95; msrs_generalization.json:/splits/year/arms/ligand+both/r2 - /splits/ligand/arms/ligand+both/r2 | MEASURED |
