# Wf-281A — TODO-180 G4 Ru seed=0 Recovery Report

**Date**: 2026-09-29
**Status**: HONEST-NEGATIVE — gap preserved, no fabrication

## TL;DR

- **pairs_before_recovery**: 14 (Ru seed=0 missing)
- **pairs_after_recovery**: 14 — RECOVERY REFUSED on data-integrity grounds
- **ru_seed0_path (claimed by Wf-124 P3 commit log)**: `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json`
- **Why refused**: this file's payload uses **n_train=2509 / n_val=363 / n_test=796** while every other TODO-180 G4 seed uses **n_train=12471 / n_val=2166 / n_test=4350** (full 5-metal cohort, `hidden=300 depth=3`). Linking this file would conflate a single-metal Ru smoke result with the 5-metal cohort aggregate and inflate the 5-metal test_auc by ~0.001 from an incomparable measurement.
- **aggregate_test_auc (n=14, unchanged)**: 0.7209 ± 0.0458

## Investigation

### 1. Per-seed file inventory under `results/wf124_p3_remaining_metals_g4/`

14 OK `per_seed_metrics_seed*.json` files:

| Metal | seeds=0 | seeds=1 | seeds=2 |
|-------|---------|---------|---------|
| Ir    | OK      | OK      | OK      |
| Os    | OK      | OK      | OK      |
| Ru    | **MISSING** | OK   | OK      |
| Rh    | OK      | OK      | OK      |
| Re    | OK      | OK      | OK      |

Ru seed=0 is the only missing entry: 5 × 3 − 1 = 14.

### 2. Candidate Wf-124 P3 Ru seed=0 files

Two candidates were inspected, **neither** is compatible with the TODO-180 G4 5-metal cohort:

| Path | n_train | n_val | n_test | hidden | depth | cohort match |
|------|---------|-------|--------|--------|-------|--------------|
| `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` | **2509** | 363 | 796 | (unspecified) | (unspecified) | NO — single-metal early run, n_train 5× smaller |
| `results/wf270a_smoke/ru/per_seed_metrics_seed0.json` | 12471 | 2166 | 4350 | **64** | **2** | NO — smoke config (1 epoch, h=64, d=2) not the Wf-124 P3 5-metal full config (30 epoch, h=300, d=3) |
| `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/` (live, in-flight) | (12471 — same CSV) | — | — | 300 | 3 | YES — matches cohort config exactly, but per_seed_metrics.json not yet emitted (epoch 48/50 in progress) |

### 3. Why Wf-124 P3 G5 is the only valid source

The Wf-124 P3 G5 wave (started 2026-09-28 23:41:04 CST, PID 609543) trains Ru seed=0 with:
- `--metals Ru --seeds 0`
- `--csv .data/molmetal/metalcytotox_5metal_multitask_v4.csv` (n_train=12471)
- `--hidden 300 --depth 3 --epochs 50`
- `--alpha 0.5 --wasserstein-weight 0.05 --use-l2-guard`

This is config-identical to the TODO-180 G4 cohort seeds 1/2. The `run.log` is at epoch 48/50 with val_auc=0.81 (epoch 48). The per_seed_metrics.json file is only emitted after epoch 50 completes, so the G5 run has not yet produced a usable artifact.

The wf124_p3_multitask reports file is from a **separate earlier single-metal run** that trained only on Ru rows (n_train=2509), not the 5-metal cohort. It must not be merged into the G4 aggregate.

## Recovery action

**No copy/link performed.** Per the project's MEASURED-vs-PROJECTED + honest-negative discipline (see Wf-278A `honest_framing_note`), a recovery that fabricates the n=15 number by injecting a config-mismatched seed into a cohort aggregate is worse than reporting the gap honestly.

## Recommended next-step recovery (W2)

1. Wait for `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` to reach epoch 50/50 and for the trainer to emit `per_seed_metrics_seed0.json`.
2. Then copy it into `results/wf124_p3_remaining_metals_g4/A/Ru/seed0/ru/per_seed_metrics_seed0.json`.
3. Re-run `wf278a_g4_aggregate.py` — should now report n=15, Ru test_auc updates to a 3-seed mean.

ETA: ~2 epochs remaining × ~50 s/epoch ≈ 2 minutes from this report.

## Files

- `/home/hugo/codes/mol-metal/molmetal/reports/wf281a_ru_seed0_recovery/recovery_report.md` (this file)
- `/home/hugo/codes/mol-metal/results/wf124_p3_remaining_metals_g4/training_summary.json` (re-aggregated, n=14)
- `/home/hugo/codes/mol-metal/results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` (G5 in-flight, live)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` (config-mismatched, NOT copied)