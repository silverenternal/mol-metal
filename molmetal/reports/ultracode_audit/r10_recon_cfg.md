# R10 CFG end-to-end harness recon

Honest-framing recon of the R10 real-CrossDocked CFG control. All measurements below were read directly from on-disk artifacts; no new runs were launched. Anything inferred beyond that is flagged PROJECTED.

## Recon summary

The R10 CFG end-to-end harness `molmetal/scripts/r10_cfg_real_crossdocked.py` is on disk and runnable. Its CLI surface exposes `--output-dir`, `--train-steps`, `--n-train`, `--ode-steps`, `--lr`, `--hidden-dim`, `--n-layers`, `--n-samples`, `--budget-seconds`, `--gpu-binary` (required), and `--trace-library`. The seeds sweep is NOT a CLI flag — the three seeds `[42, 0, 1234]` and two CFG scales `[1.0, 2.0]` are hardcoded as a tuple in the `protocol` payload (line 167) and as the loop iterator `for seed in (42, 0, 1234)` (line 190), so a fresh seed requires editing the source. The CFG 1h36 micro-ablation sibling `molmetal/scripts/r10_cfg_ablation_1h36.py` does expose `--seed`, `--cfg-scales`, `--n-mols`, `--train-steps`, `--n-steps`, `--exhaustiveness`, `--device`, `--output-prefix`, `--skip-dock` (this is the wiring-validation harness, not the real-data harness).

The real-data harness has been run for FOUR configurations (MEASURED from report.json/README.md artifacts):

| Run dir | Train pairs / updates / seed scope | Requested | Finite | Decoded |
|---|---:|---:|---:|---:|
| `r10_cfg_real_crossdocked/` (original teacher-forced atom head) | 8 / 200 | 96 | 64 | 0 |
| `r10_cfg_real_crossdocked_masked_atoms/` (masked atom head) | 8 / 200 | 96 | 64 | 0 |
| `r10_cfg_real_crossdocked_train32_2000/` (larger training, old v1 spatial readout) | 32 / 2000 | 96 | 96 | 0 |
| `r10_cfg_real_crossdocked_v2_train32_2000/` (larger training, v2 equivariant readout) | 32 / 2000 | 96 | 96 | 0 |

The "96 -> 64 -> 0" path the user is asking about corresponds to the 8-pair 200-step controls (original teacher-forced and masked-atom variants): 96 samples requested per run (3 seeds x 2 pockets x 2 CFG x 8 samples), 64 samples with finite coordinates (the remaining 32 went nonfinite due to the v1 invariant-H-to-3 velocity projection bug that produced unbounded superlinear ODE growth — see `r10_equivariant_velocity_repair/`), and 0 connected sanitized decoded graphs across every configuration to date. The 32-pair/2000-step runs (v1 and v2) recovered finiteness (96/96) but still produce 0 decoded graphs; v2 produces 92 `disconnected_distance_graph` + 4 `atom_outside_training_vocabulary`, v1 produces 90 + 6.

Seeds used in MEASURED runs: `[42, 0, 1234]` for every configuration above (hardcoded loop). All three seeds have already been exercised in every on-disk run — there is no single next-seed value pending; the natural next move for a seed-sweep is to extend the iterable (e.g., add `7, 2024, 31415`) which requires modifying the source tuple at line 167 and line 190 of the harness, plus deciding whether to consume the new seed through the same single-checkpoint protocol or to introduce a seed-list CLI flag. The v2 spatial-repair report (`r10_equivariant_velocity_repair/`) and the model_real_cfg_prior_spatial_repairs_20260913.md note that the relevant independent variable is no longer seed but architecture (v1 vs v2) and training budget (8/200 vs 32/2000), since seed-variance is dwarfed by both.

Tests touching CFG end-to-end decode: `molmetal/tests/test_atom_training_contract.py` imports `decode_distance_graph` from the harness (a unit-level test of the decoder function, not a full pipeline run). No other test under `molmetal/tests/` or `molmetal/molmetal_lam/tests/` references `r10_cfg` end-to-end; the closely related CFG unit tests live at `molmetal/tests/test_egnn_velocity_cfg.py` (axis C, 5 tests, asserts the `v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)` math and bit-exact recovery at `cfg_scale==1.0`).

Failure denominators are honest: aggregate from `r10_cfg_real_crossdocked_train32_2000/report.json` is `n_requested_planned=96, n_requested=96, n_raw_generated=96, n_decoded=0, n_docked=0, n_pb_pass_docked=0` and identical in `r10_cfg_real_crossdocked_v2_train32_2000/report.json`. Both reports are timestamped 2026-09-13T21:48–21:51 UTC and carry full runtime/checkpoint/source sha256 hashes plus per-cell status counters; nothing in the JSON aggregate is marked as a success.

Recommendation: do NOT spend cycles on a fresh seed sweep until the decoder path produces a nonzero decoded-graph rate. The bottleneck identified in `model_real_cfg_prior_spatial_repairs_20260913.md` is the distance-connectivity decoder (92/96 disconnected in v2) and the unconstrained atom head (4/96 outside C/N/O/F vocabulary). A seed-sweep at this stage measures the same zero. PROJECTED next actions, in priority: (1) replace distance-connectivity decoder with a learned bond-order head trained jointly with coordinates; (2) add output vocabulary mask to constrain generated atoms to {6,7,8,9}; (3) extend `r10_cfg_real_crossdocked.py` with `--seeds` (nargs="+") so seed sweeps are CLI-driven and no longer source-edit; (4) only then re-run the seed sweep on the v2 architecture to characterise seed-variance of an actually-validating model.

## Path list

Harness:
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py` (real-data CFG E2E harness, seeds hardcoded lines 167+190)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_ablation_1h36.py` (1h36 micro-ablation sibling; exposes `--seed` and `--cfg-scales`)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_integrator_diagnostic.py` (replay harness, reads `--source` JSON)

Reports (MEASURED):
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_crossdocked/report.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_crossdocked/README.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_crossdocked_masked_atoms/report.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_crossdocked_train32_2000/report.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_crossdocked_train32_2000/README.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_crossdocked_v2_train32_2000/report.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_crossdocked_v2_train32_2000/README.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_integrator_diagnostic.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_integrator_v2_diagnostic.json`

Synthesis report (MEASURED failure-path description):
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/model_real_cfg_prior_spatial_repairs_20260913.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_equivariant_velocity_repair/` (v1 vs v2 spatial counterexamples)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/round10_final.md` (axis C micro-bench summary, 1h36 only)

Tests touching CFG E2E decode:
- `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_atom_training_contract.py` (imports `decode_distance_graph` from the harness; unit-level decoder test)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_egnn_velocity_cfg.py` (CFG math unit tests; 5 tests; not E2E)

CLI seed-sweep flag:
- NOT exposed in `r10_cfg_real_crossdocked.py` (hardcoded tuple at line 167 / loop at line 190). Sibling `r10_cfg_ablation_1h36.py` does expose `--seed` (single, default 0) and `--cfg-scales` (nargs="+", default 1.0 1.5 2.0 3.0).
