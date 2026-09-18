# Real tmQM geometry primitive control

real DFT geometry coordinate-repair control; explicit-edge primitive; not production sampler or Vina evidence

Eight Pt CN4 complexes from actual tmQM DFT structures, with four explicit Wiberg neighbors >= 0.3, were independently perturbed at 0.25 Å using seeds 42, 0, 1234. Prior off/on start from identical full coordinates. Fifty GPU SGD steps minimize a restraint to the noisy input plus an angular primitive with 90°/180° targets. DFT coordinates are used only for evaluating recovery, not as optimization targets. All atoms are retained.

| Independent metric | Mean paired delta (on − off) | Sample SD | Improved / 24 |
|---|---:|---:|---:|
| all_atom_coordinate_rmse_A_no_alignment | -0.00238971 | 0.00208704 | 22 / 24 |
| donor_vector_rmse_A | -0.09760685 | 0.06839821 | 23 / 24 |
| coordination_plane_rms_A | -0.03022627 | 0.03444814 | 21 / 24 |
| donor_angle_mae_to_dft_degrees | -6.07727575 | 4.68785677 | 21 / 24 |
| bo_edge_length_mae_to_dft_A | -0.00462960 | 0.00647699 | 17 / 24 |

The 24 pairs contain repeated seeds of eight structures, not 24 independent molecular identities; no significance test or generalization claim is made. Negative cases remain in report.json alongside full coordinates. The angular diagnostic was evaluated before the final SGD update; independent coordinate metrics and saved structures use the final coordinates.

This validates only the explicit-edge GeometryPenalty coordinate repair primitive. It does not validate MetalGeometryPrior production sampling, molecular generation, oxidation state (CN4 is not proof of Pt(II)), or affinity. No receptor-paired Pt scoring setup is validated locally; no Vina improvement is claimed.

Artifacts: [report.json](report.json), reference and paired XYZ files. The JSON records dataset and runtime hashes and all sample identities.
