# BLK-I1: launcher silent death fix (2026-10-09)

**Status**: SHIPPED (per §5.5 filesystem-truth evidence below)
**Scope**: per `todo.json` execution_order step 1; prerequisite for BLK-D1 + BLK-C1 retest.
**AGENTS.md compliance**: §5.4 (no wet-lab) + §5.5 (filesystem-truth) + §5.6 (atomic_write_file)
+ §5.8 (≤2 parallel agents; direct Edit/Write preferred) + §5.9 (single-venv).

## Root cause (per BLK-I1)

`scripts/launch_*.sh` wrappers lacked `setsid`/`nohup` for the trainer
invocation. When the parent bash exited (e.g. user closes the tmux pane, or
the wrapper is killed by SIGTERM during tmux session cleanup), the trainer
process group is killed by SIGHUP — even though the trainer is running on
the GPU. The log freezes mid-stream with no traceback and `metrics.json`
never appears.

Three concrete ABORTED events trace to this pattern (per `todo.json`):
- Wf-FIX-3LINES bace (seed 2/5)
- Wf-FIX-3LINES CFM bond head (epoch 1/20)
- Wf-R16-A NCI-60 multi-day sequential

## Fix design

For each `scripts/launch_*.sh` that actually launches a long-running trainer,
the trainer invocation is now wrapped in `setsid nohup ... &` so the trainer
detaches from the parent process group and survives parent shell exit.

Two patch families applied:

1. **Custom high-fidelity patch** for the 2 BLK-I1-directly-affected launchers
   (`launch_wf_fix3lines_bace.sh` + `launch_wf_fix3lines_cfm.sh`). These get
   a complete fix that ALSO redirects the wrapper's stdout/stderr to
   `logs/<script>.log` via `exec >"$LOG" 2>&1`, plus a `wait $PID` so the
   wrapper still captures the trainer's exit code. This is the pattern the
   Wf-FIX-3LINES retest needs.

2. **Helper-patch (minimal inline wrap)** for 21 other launchers that already
   have their own logging (`| tee` / `exec >LOG`) and a structured exit-code
   capture (`PIPESTATUS[0]` / `|| true`). The helper prepends `setsid nohup `
   to the trainer invocation line only, leaving all existing logging and
   exit-code capture untouched.

The helper itself is `scripts/_patch_launcher_silent_death.py` (one-off
tool, idempotent on re-run). All writes go through
`molmetal.scripts.wf386a3_atomic_write.atomic_write_file` (per §5.6), so
the launcher mode (+x) is preserved.

## Inventory of patched launchers (26 of 36)

| Patched by | Count | Launchers |
|---|---:|---|
| Pre-existing (setsid/nohup already wired) | 3 | `launch_all_training.sh`, `launch_chemprop_v3_p14_post_cfm_round7.sh`, `launch_item4_chain_post_wf131.sh` |
| Custom (BLK-I1 direct, with redirect+wait) | 2 | `launch_wf_fix3lines_bace.sh`, `launch_wf_fix3lines_cfm.sh` |
| Helper (setsid+nohup on trainer line) | 21 | `launch_cfm_round7_f5f6_retrain.sh`, `launch_cfm_tmqm_all6_retrain.sh`, `launch_dmpnn_clean5_lomo.sh`, `launch_dmpnn_clean5v3_loso.sh`, `launch_dmpnn_clean5v4_loso_d1d3.sh`, `launch_variance_puct_full.sh`, `launch_wf119_p1_4_chemprop_v3_lomo.sh`, `launch_wf124_p3_g5.sh`, `launch_wf124_p3_g6.sh`, `launch_wf125_p1_controlled_pair.sh`, `launch_wf131_p3_pt_4cells.sh`, `launch_wf146_deferred_runs.sh`, `launch_wf299_cfm_main.sh`, `launch_wf308c_todo220_gpu.sh`, `launch_wf310b_g6_final.sh`, `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh`, `launch_wf310e_r20_p4_bindingdb_1.07M.sh`, `launch_wf310f_frontier_math_gpu.sh`, `launch_wf390a1_m2_r2_path_a.sh`, `launch_wf394a1_lambda_cfm_patch.sh`, `launch_wf446_cfm_tmqm_large_cohort.sh` |
| **Total** | **26** | |

## Inventory of deferred launchers (10 of 36)

10 launchers still lack `setsid`/`nohup` and are documented below:

| Reason | Count | Launchers | Required refactor (deferred scope) |
|---|---:|---|---|
| `if CUDA_VISIBLE_DEVICES=0 "${PY}" "${TRAINER}" \` pattern | 4 | `launch_wf131_p3_deferred_runs.sh`, `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`, `launch_wf310e_wf135_phase2_gpu.sh`, `launch_wf310e_wf219_production_gpu.sh` | Refactor `if-then` to detached background + `wait $? -eq 0` (scope > BLK-I1 minimal patch) |
| Dry-run / readiness-audit only (no trainer invocation) | 6 | `launch_variance_puct_smoke.sh`, `launch_wf333_vina_4step_gpu_bench.sh`, `launch_wf380_scaffold_loss.sh`, `launch_wf388_a1_metal_coord_retrain.sh`, `launch_wf394_lambda_cfm.sh`, `launch_wf409a3_metal_coord_retrain.sh` | No fix needed; these scripts run `pre_flight_check` / scaffold probes only |

## §5.5 filesystem-truth evidence

All 26 patched launchers + the 1 helper script on disk (timestamps 2026-10-09
unless noted; pre-existing items at 2026-10-08):

```
2026-10-08 06:01:21  scripts/launch_all_training.sh
2026-10-08 08:59:37  scripts/launch_item4_chain_post_wf131.sh
2026-10-08 09:43:08  scripts/launch_chemprop_v3_p14_post_cfm_round7.sh
2026-10-09 09:32:26  scripts/launch_wf_fix3lines_bace.sh           md5=7128c460bcec050dbda2e8c2a458a0f8
2026-10-09 09:32:39  scripts/launch_wf_fix3lines_cfm.sh            md5=d08fa78d34a599bc9f95d1994e287807
2026-10-09 09:33:52  scripts/launch_cfm_round7_f5f6_retrain.sh
2026-10-09 09:33:52  scripts/launch_cfm_tmqm_all6_retrain.sh
2026-10-09 09:33:52  scripts/launch_dmpnn_clean5_lomo.sh
2026-10-09 09:33:52  scripts/launch_dmpnn_clean5v3_loso.sh
2026-10-09 09:33:52  scripts/launch_dmpnn_clean5v4_loso_d1d3.sh
2026-10-09 09:33:52  scripts/launch_wf124_p3_g5.sh
2026-10-09 09:33:52  scripts/launch_wf124_p3_g6.sh
2026-10-09 09:33:52  scripts/launch_wf125_p1_controlled_pair.sh
2026-10-09 09:33:52  scripts/launch_wf131_p3_pt_4cells.sh
2026-10-09 09:33:52  scripts/launch_wf310b_g6_final.sh
2026-10-09 09:33:52  scripts/launch_wf390a1_m2_r2_path_a.sh
2026-10-09 09:34:37  scripts/launch_variance_puct_full.sh
2026-10-09 09:34:37  scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh
2026-10-09 09:34:37  scripts/launch_wf146_deferred_runs.sh
2026-10-09 09:34:37  scripts/launch_wf299_cfm_main.sh
2026-10-09 09:34:37  scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh
2026-10-09 09:34:37  scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh
2026-10-09 09:34:37  scripts/launch_wf310f_frontier_math_gpu.sh
2026-10-09 09:34:37  scripts/launch_wf394a1_lambda_cfm_patch.sh
2026-10-09 09:34:37  scripts/launch_wf446_cfm_tmqm_large_cohort.sh
2026-10-09 09:34:49  scripts/launch_wf308c_todo220_gpu.sh
2026-10-09 09:35:XX  scripts/_patch_launcher_silent_death.py      md5=22b5327d36ce2287ed1f6834d9856699
```

All 36 launchers pass `bash -n` syntax check (0/36 fail).

## §5.7 50-step CPU smoke (smoke gate for BLK-I1 is N/A — infrastructure, not trainer)

Per AGENTS.md §5.7, the §5.7 smoke gate is required for trainer FLAG WIRE-INS,
not for wrapper-script patches. The actual validation of BLK-I1 is the
Wf-FIX-3LINES bace 5-seed retest (BLK-D1 = execution_order step 2) — see
the `done_when` clause: "5/5 bace seeds complete without silent death". That
retest is USER_OK-gated per §5.1 and is the next item in the execution_order.

## §5.9 single-venv rule verification

`grep -L "\.venv/bin/python" scripts/launch_*.sh` returns 3 launchers:
`launch_chemprop_v3_p14_post_cfm_round7.sh`, `launch_all_training.sh`,
`launch_item4_chain_post_wf131.sh`. These 3 are pre-existing (timestamps
2026-10-08) and were not touched by this wave. The 23 launchers we
patched all use `.venv/bin/python` directly or via the `${PY}` variable
that points to `.venv/bin/python`. **BLK-I1 does not include venv audit**;
the venv audit is the §5.9 gate (per AGENTS.md, _wrapper_audit.py
+ wf437a1_single_venv_audit.py) and is out of scope here.

## Honest-negative framing (per §5.2)

- 4 of 36 launchers remain un-patched (`if-then` pattern). These are
  still at risk of trainer silent death in the `if`-branch path. The
  4-launcher refactor is a future wave item; BLK-I1 retest is unaffected
  (those 4 are not in the Wf-FIX-3LINES retest path).
- 6 of 36 launchers are dry-run / readiness-audit (no trainer
  invocation). They are not at risk of silent death and need no fix.
- The 23 launchers we patched have the trainer detached from the
  parent process group. We have NOT actually launched one of the
  patched launchers end-to-end and observed survival — that is
  BLK-D1's job (per the `done_when` clause).

## Cross-references

- `todo.json` execution_order step 1 (BLK-I1)
- `AGENTS.md` §5.5 (filesystem-truth) + §5.6 (atomic write) + §5.7 (smoke) + §5.8 (token budget) + §5.9 (single-venv)
- `memory/wrapper-pipefail-sigpipe-silent-death.md` (the original SIGPIPE-on-`tee` lesson that this fix complements)
- `TODO/completed/metric_fix_3lines_2026_10_09.md` §6 (the 2-of-3 ABORTED root-cause analysis that motivated BLK-I1)
- `TODO/INDEX.md` §22 (will be added in the final closeout wave per execution_order step 19)
