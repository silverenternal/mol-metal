# Conformal calibration (alpha=0.10, nominal 90%)

Generated 2026-10-19T11:45:22Z from `metrics/by_metric/*.json`.

Coverage alone is guaranteed by the conformal construction; width and PIT are the informative columns.

| Metric | Coverage | Median width | Winkler | PIT KS | Slope | R2(fit) | Bias | Source | Tag |
|---|---|---|---|---|---|---|---|---|---|
| doi / point calibration | - | - | - | - | 0.9256 | 0.3363 | -0.0347 | msrs_calibration.json:/protocols/doi/point_calibration/slope; msrs_calibration.json:/protocols/doi/point_calibration/r2_fit; msrs_calibration.json:/protocols/doi/point_calibration/mean_bias | MEASURED |
| doi / global | 0.9004 | 2.0052 | 2.7362 | 0.0012 | - | - | - | msrs_calibration.json:/protocols/doi/variants/global/coverage; msrs_calibration.json:/protocols/doi/variants/global/median_width; msrs_calibration.json:/protocols/doi/variants/global/winkler; msrs_calibration.json:/protocols/doi/variants/global/pit/ks_stat | MEASURED |
| doi / metal | 0.9002 | 2.0231 | 2.7328 | 0.0019 | - | - | - | msrs_calibration.json:/protocols/doi/variants/metal/coverage; msrs_calibration.json:/protocols/doi/variants/metal/median_width; msrs_calibration.json:/protocols/doi/variants/metal/winkler; msrs_calibration.json:/protocols/doi/variants/metal/pit/ks_stat | MEASURED |
| doi / difficulty | 0.8789 | 1.9195 | 2.8587 | 0.0439 | - | - | - | msrs_calibration.json:/protocols/doi/variants/difficulty/coverage; msrs_calibration.json:/protocols/doi/variants/difficulty/median_width; msrs_calibration.json:/protocols/doi/variants/difficulty/winkler; msrs_calibration.json:/protocols/doi/variants/difficulty/pit/ks_stat | MEASURED |
| ligand / point calibration | - | - | - | - | 1.0740 | 0.5583 | 0.0042 | msrs_calibration.json:/protocols/ligand/point_calibration/slope; msrs_calibration.json:/protocols/ligand/point_calibration/r2_fit; msrs_calibration.json:/protocols/ligand/point_calibration/mean_bias | MEASURED |
| ligand / global | 0.9003 | 1.6515 | 2.2437 | 0.0012 | - | - | - | msrs_calibration.json:/protocols/ligand/variants/global/coverage; msrs_calibration.json:/protocols/ligand/variants/global/median_width; msrs_calibration.json:/protocols/ligand/variants/global/winkler; msrs_calibration.json:/protocols/ligand/variants/global/pit/ks_stat | MEASURED |
| ligand / metal | 0.9003 | 1.6278 | 2.2405 | 0.0019 | - | - | - | msrs_calibration.json:/protocols/ligand/variants/metal/coverage; msrs_calibration.json:/protocols/ligand/variants/metal/median_width; msrs_calibration.json:/protocols/ligand/variants/metal/winkler; msrs_calibration.json:/protocols/ligand/variants/metal/pit/ks_stat | MEASURED |
| ligand / difficulty | 0.8835 | 1.6389 | 2.3525 | 0.0455 | - | - | - | msrs_calibration.json:/protocols/ligand/variants/difficulty/coverage; msrs_calibration.json:/protocols/ligand/variants/difficulty/median_width; msrs_calibration.json:/protocols/ligand/variants/difficulty/winkler; msrs_calibration.json:/protocols/ligand/variants/difficulty/pit/ks_stat | MEASURED |
