# scripts/_wrappers_index.md — canonical wrapper catalog

**Last updated:** 2026-10-02 (Fri 02) 07:08 CST via Wf-A2 wf-A2 ADR docs-only sync

**Total on disk:** 23 launch_wf*.sh; CANONICAL=8, SCAFFOLD=15.

**Quickstart** — `bash scripts/launch_wf<NNN>_<purpose>.sh --dry-run` (requires `USER_OK=1` per Wf-294B gate pattern).

## Real wrappers by target trainer

### `train_cfm_yuelbond_wasserstein.py` (4 wrappers)

- **launch_wf299_cfm_main.sh** (701 lines) — see `scripts/launch_wf299_cfm_main.sh`
- **launch_wf310f_frontier_math_gpu.sh** (329 lines) — see `scripts/launch_wf310f_frontier_math_gpu.sh`
- **launch_wf380_scaffold_loss.sh** (252 lines) — see `scripts/launch_wf380_scaffold_loss.sh`
- **launch_wf394a1_lambda_cfm_patch.sh** (340 lines) — see `scripts/launch_wf394a1_lambda_cfm_patch.sh`

### `train_chemprop_v2_v3.py` (1 wrappers)

- **launch_wf310e_r20_p4_bindingdb_1.07M.sh** (519 lines) — see `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh`

### `train_dmpnn_multitask_wasserstein.py` (4 wrappers)

- **launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh** (473 lines) — see `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`
- **launch_wf388_a1_metal_coord_retrain.sh** (384 lines) — see `scripts/launch_wf388_a1_metal_coord_retrain.sh`
- **launch_wf390a1_m2_r2_path_a.sh** (429 lines) — see `scripts/launch_wf390a1_m2_r2_path_a.sh`
- **launch_wf409a3_metal_coord_retrain.sh** (287 lines) — see `scripts/launch_wf409a3_metal_coord_retrain.sh`

### `train_metal_lomo_v2.py` (1 wrappers)

- **launch_wf310e_wf135_phase2_gpu.sh** (244 lines) — see `scripts/launch_wf310e_wf135_phase2_gpu.sh`

### `train_metal_lomo_wasserstein.py` (3 wrappers)

- **launch_wf125_p1_controlled_pair.sh** (405 lines) — see `scripts/launch_wf125_p1_controlled_pair.sh`
- **launch_wf131_p3_deferred_runs.sh** (322 lines) — see `scripts/launch_wf131_p3_deferred_runs.sh`
- **launch_wf131_p3_pt_4cells.sh** (510 lines) — see `scripts/launch_wf131_p3_pt_4cells.sh`

### `wf146_chemprop_lomo.py` (1 wrappers)

- **launch_wf146_deferred_runs.sh** (575 lines) — see `scripts/launch_wf146_deferred_runs.sh`

### `wf185_chemprop_v2_drugood_full.py` (1 wrappers)

- **launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh** (509 lines) — see `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh`

## Duplicate groups (same target trainer)

- `train_metal_lomo_wasserstein.py`: 3 wrappers — launch_wf125_p1_controlled_pair.sh, launch_wf131_p3_deferred_runs.sh, launch_wf131_p3_pt_4cells.sh
- `train_cfm_yuelbond_wasserstein.py`: 4 wrappers — launch_wf299_cfm_main.sh, launch_wf310f_frontier_math_gpu.sh, launch_wf380_scaffold_loss.sh, launch_wf394a1_lambda_cfm_patch.sh
- `train_dmpnn_multitask_wasserstein.py`: 4 wrappers — launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh, launch_wf388_a1_metal_coord_retrain.sh, launch_wf390a1_m2_r2_path_a.sh, launch_wf409a3_metal_coord_retrain.sh

## Notes

- Wrappers without explicit `TRAINER=` macro reference `TRAINER="${DOCK_PANEL:-molmetal/scripts/...}"` style — see `_WRAPPER_TEMPLATE.sh` for canonical surface.
- DRY_RUN.log + plan.json emission per Wf-363-A1 schema (`_dry_run_log_helper.sh` is the shared source).
- USER_OK gate per Wf-294B + Wf-355-B1 + Wf-405-A4 + Wf-366-A1 pattern (canonical surface).
- Pre-flight v3 `pre_flight_check.py --pipeline all` BEFORE any GPU work (per Wf-415-A3 + Wf-427-B1 cascade closures).
- `PY=/home/hugo/codes/mol-metal/.venv/bin/python` HARDCODED (per §5.9 + AGENTS.md memory rule "Standing GPU grant").
