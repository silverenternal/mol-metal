# Wf-310C Part 4 — TODO-220 Driver 1+2 full GPU launch plan

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent
**Status**: 🚧 WRAPPER READY — USER_OK-gated, awaiting user opt-in
**Wrapper path**: `molmetal/scripts/launch_wf310c_todo220_gpu.sh`

---

## §1 Purpose

Wf-310C Part 4 ships the final USER_OK-gated GPU wrapper for the TODO-220
temporal-split + NCI-60 9-cell drivers. This wrapper is the Wf-310C sibling
of `launch_wf308c_todo220_gpu.sh` (Wf-308C Part 3) — same USER_OK-file gate
convention, but additionally runs the Wf-259 `pre_flight_check.py` first per
the Wf-310C spec, so the env is verified before the 3.9 hr GPU sweep starts.

---

## §2 Scope

| Item | Value |
|---|---|
| Stage A trainer | `molmetal/scripts/wf219_temporal_split_driver.py` (Wf-308C Part 1) |
| Stage A CSV | `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (26633 rows) |
| Stage A scope | 5 metals (Ru/Ir/Os/Rh/Re) × 5 seeds × 50 epochs × hidden=300 depth=3 (chemprop v2 prod) |
| Stage A train/test cutoff | 2023-01-01 / 2024-01-01 (Krasnov 2026 exact protocol) |
| Stage A wall | ~2.9 hr PROJECTED on RTX 5090 cuda:1 |
| Stage B trainer | `molmetal/scripts/nci60_9cell_chemprop_driver.py` (Wf-308C Part 2) |
| Stage B CSV | `molmetal/data/nci60_precious_9cell.csv` (7823 rows × 9 cells) |
| Stage B scope | 9 cells × 3 seeds × 50 epochs × hidden=128 depth=3 (TODO-130 G3 prod) |
| Stage B wall | ~1.0 hr PROJECTED on RTX 5090 cuda:1 |
| Pre-flight | `pre_flight_check.py --pipeline chemprop_v2_v3` (Wf-259) |
| Total wall | ~3.9 hr GPU (Stage A 2.9 hr + Stage B 1.0 hr) |

---

## §3 USER_OK gate (per kill_then_adapt)

| Step | Action |
|---|---|
| 1 | `touch /tmp/USER_OK_WF310C_TODO220` (user opt-in marker) |
| 2 | `bash molmetal/scripts/launch_wf310c_todo220_gpu.sh` (actual GPU run) |

Without the marker file the wrapper exits 2 (USER_OK missing) and prints
the honest framing banner.

---

## §4 CPU smoke status (Wf-310C Parts 1+2 already verified 2026-09-29)

| Part | Driver | Result |
|---|---|---|
| 1 | `wf219_temporal_split_driver.py --execute` (1-seed 1-epoch hidden=64 cpu) | **5/5 ok runs** in 0.42 min, mean_auc=0.6810 |
| 2 | `nci60_9cell_chemprop_driver.py --execute` (1-seed 1-epoch hidden=64 cpu) | **9/9 ok runs** in 0.09 min, mean_auc=0.5617 |

Both drivers' `--execute` paths are verified working on CPU. The GPU run is
the MEASURED-vs-PROJECTED promotion step that requires user opt-in.

---

## §5 Exit codes

| Code | Meaning |
|---:|---|
| 0 | GPU run completed (all stages ok) |
| 2 | USER_OK missing (gated; no run started) |
| 3 | Pre-flight check failed (env / GPU / CSV / pre_flight_check.py FAILed) |
| 4 | Stage A failed (Stage B not attempted) |
| 5 | Stage B failed (Stage A results still saved) |

---

## §6 Outputs

- `molmetal/reports/wf310c_todo220_launch/pre_flight.json` (Wf-259 chemprop_v2_v3 pre-flight JSON)
- `molmetal/reports/wf310c_todo220_launch/plan.json` (plan + honest framing)
- `molmetal/reports/wf310c_temporal_split/sweep_summary.json` (Stage A summary)
- `molmetal/reports/wf310c_nci60_9cell/sweep_summary.json` (Stage B summary)
- `molmetal/reports/wf310c_temporal_split/run.log` (Stage A stdout/stderr)
- `molmetal/reports/wf310c_nci60_9cell/run.log` (Stage B stdout/stderr)

---

## §7 Cross-references

- `TODO/pending/220_wf219_temporal_split_nci60_driver.md` (parent spec)
- `molmetal/scripts/launch_wf308c_todo220_gpu.sh` (Wf-308C sibling — same scope, no pre_flight_check.py)
- `molmetal/scripts/wf219_temporal_split_driver.py` (Wf-308C Part 1 driver)
- `molmetal/scripts/nci60_9cell_chemprop_driver.py` (Wf-308C Part 2 driver)
- `molmetal/scripts/pre_flight_check.py` (Wf-259 — chemprop_v2_v3 + CFM env verifier)

---

## §8 Honest framing disclaimer (party-twenty-spirit §3.5)

Wf-310C Part 4 ships the **USER_OK-gated final wrapper** for the TODO-220
GPU sweep. CPU smokes (Parts 1+2) confirmed both `--execute` paths work on
CPU at 1-seed 1-epoch scale. The 5-seed × 50-epoch GPU sweep is the production
run that lifts the smoke numbers into the Krasnov 2026-aligned AUC range.
3.9 hr wall on RTX 5090 cuda:1; USER_OK gate enforced.