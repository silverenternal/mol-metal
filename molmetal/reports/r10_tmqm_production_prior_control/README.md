# Real tmQM production prior energy control

real DFT geometry coordinate-repair control; production prior energy; not production sampler or Vina evidence

Same 8 actual Pt CN4 structures and 3 seeds as the original explicit-edge primitive control, with all atoms and explicit Wiberg donor endpoints retained. The production MetalGeometryPrior now consumes those endpoints and applies its documented 0.4–5 Å donor-distance sanity bounds. Fifty SGD steps optimize restraint to the identical noisy start plus the prior. DFT targets are evaluation-only; all negative pairs remain.

| Independent recovery metric | Mean on − off | Sample SD | Improved / 24 |
|---|---:|---:|---:|
| all_atom_coordinate_rmse_A_no_alignment | -0.00162455 | 0.00247933 | 20 / 24 |
| donor_vector_rmse_A | -0.07132249 | 0.08150823 | 18 / 24 |
| coordination_plane_rms_A | -0.01602116 | 0.04845491 | 20 / 24 |
| donor_angle_mae_to_dft_degrees | -4.37478250 | 4.55572979 | 19 / 24 |
| bo_edge_length_mae_to_dft_A | -0.00277935 | 0.00696489 | 16 / 24 |

This is production energy-function coordinate repair, separate from the untrained actual-sampler integration test. It is not trained metal-complex generation or evidence of improved Vina. The angular function is only a soft local constraint, not a complete geometric validator; distance cutoffs can deactivate distant declared donors. CN4 alone does not establish oxidation state. There are 8 molecular identities with repeated seeds, not 24 independent molecules. A valid receptor-paired Pt scoring setup remains unavailable.

Full coordinates, source/runtime hashes and every pair: [report.json](report.json).
