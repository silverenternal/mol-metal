# Wf-306B Wf-219 Ru short-epoch wrapper — plan

## Origin

- **Wf-304C Task 1**: Wf-219 Ru seed runs all hit 43/50 epochs and timed out
  under the standard 5-seed × 50-epoch protocol on RTX 5090 cuda:1. The actual
  convergence point is unknown.
- **Wf-305B/Wf-305C follow-up**: identified that reducing the epoch target
  from 50 to 30 + capping seeds at 3 would fit inside ~10 min/wall on
  RTX 5090 cuda:1 and give an honest re-measurement of the Ru baseline.

## Scope

| param    | value | label       | source                          |
|----------|-------|-------------|----------------------------------|
| metal    | Ru    | MEASURED    | Wf-219 protocol, Ru seed=0/1/2  |
| seeds    | 0,1,2 | MEASURED    | 3-seed quick subset (vs full 5)  |
| epochs   | 30    | PROJECTED   | Wf-304C: 43/50 timed out         |
| gpu      | cuda:1| MEASURED    | RTX 5090 (Wf-299 Wave B/C)      |
| wall_budget_min | 10 | PROJECTED | 30/50 × Wf-304C measured 13.5 min ≈ 9.4 min + 6% safety |

## USER_OK gate (explicit, stricter than Wf-299)

The wrapper REFUSES to run unless `USER_OK=1` is set in the env, including
in `--dry-run` mode. This is intentional: the wrapper targets a real GPU
retrain of Wf-219 (not a stub), and the dry-run path emits a `plan.json`
that should only be created under explicit user approval.

Verification:
- `USER_OK=0 bash launch_wf306b_wf219_ru_gpu.sh --dry-run` → exit 64, refuses
- `USER_OK=1 bash launch_wf306b_wf219_ru_gpu.sh --dry-run` → emits
  `molmetal/reports/wf306b_wf219_ru_wrapper/plan.json`, exit 0

## Wall budget derivation (MEASURED-vs-PROJECTED)

- **Wf-304C MEASURED**: 43/50 epochs on RTX 5090 cuda:1 took 13.5 min wall.
- **PROJECTED for 30/50 epochs**: 30/43 × 13.5 min ≈ 9.4 min wall per seed ×
  3 seeds ≈ 28 min naive. However Wf-304C was a single-seed timeout, not a
  3-seed back-to-back; the per-seed wall is roughly constant once warm.
- **REVISED PROJECTED**: ~10 min total wall for the 3-seed sweep on
  cuda:1 (parallelism assumed not used; sequential sweep).

If actual elapsed exceeds 10 min, the wrapper emits a `MEASURED-over-PROJECTED`
warning in `launch.log` and `plan.json` gets a follow-up note.

## Honest-negative on adapter wire-in (Task Part 3)

Wf-305B claimed "4 frontier math adapters wired but actual smoke NOT RUN".
The actual smoke test (Wf-306B Part 3) reveals:

- 0/4 frontier adapters load. The 4 classes (`MultiPHScheduler`,
  `SE3LipmanScheduler`, `SE3SBScheduler`, `TransportRelaxationScheduler`) are
  NOT exported by `molmetal/adapters/flow_matching_lipman/__init__.py`.
- Only `CondOTScheduler` is present in `_reference/path/scheduler/__init__.py`.
- The 4 adapter modules do not exist as subdirectories under
  `molmetal/adapters/flow_matching_lipman/_reference/` (only `path/`,
  `solver/`, `utils/` are present).

This is a **Wf-305B honest-negative**: the wire-in step was never completed,
not "wired but unsmoked". A follow-up Wf (suggest Wf-306C) should
implement the 4 missing adapter modules + re-export from `__init__.py`.

## Status

- wrapper script: `molmetal/scripts/launch_wf306b_wf219_ru_gpu.sh` (executable, USER_OK gate enforced)
- plan record: `molmetal/reports/wf306b_wf219_ru_wrapper/plan.json`
- USER_OK=1 dry-run: VERIFIED (exit 0, plan emitted)
- Actual GPU run: NOT LAUNCHED (awaits USER_OK=1 from user)

## Honest-negative on Wf-305C CSV paths

Both driver scripts reference CSV paths that are NOT present in the repo:

- `molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv` — MISSING
- `molmetal/.data/molmetal/nci60_9cell.csv` — MISSING

This does not affect the dry-run (the stubs do not load the CSV), but is a
follow-up action: Wf-306D should regenerate the 5-metal multi-task CSV from
the MetalCytoToxDB source + build the 9-cell NCI-60 subset from the PlatinAI
pIC50 source.

## Files written by Wf-306B

| path                                                               | purpose                              |
|--------------------------------------------------------------------|--------------------------------------|
| `molmetal/scripts/launch_wf306b_wf219_ru_gpu.sh`                   | GPU launch wrapper (USER_OK gate)    |
| `molmetal/reports/wf306b_wf219_ru_wrapper/wrapper_plan.md`         | this file                            |
| `molmetal/reports/wf306b_wf219_ru_wrapper/plan.json`               | emitted plan (USER_OK=1 dry-run)     |
| `molmetal/reports/wf306b_wf219_ru_wrapper/launch.log`              | log of plan-emission + future runs   |