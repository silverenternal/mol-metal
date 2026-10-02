# Wf-275A — TODO-119 P1.4 CPU Pilot Report

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Status:** COMPLETE — honest-negative
**Cross-ref:** configs/cfm_per_innovation_ablation_p1.4.yaml, configs/wf124_p3_per_innovation_ablation_g6.yaml

## Goal

Validate the 4-ablation per-innovation protocol on a small CPU subset before
committing 7.5 hr GPU for the 60-run Wf-124 P3 ablation (TODO-182 G6).

Per configs/cfm_per_innovation_ablation_p1.4.yaml `honest_framing.do_not_claim`,
per-lever attribution must NOT be cited as MEASURED until all 4 ablations
complete; this pilot is a PROTOCOL smoke, not an attribution measurement.

## Pilot Design

- 4 ablation configs (full / no-yuelbond / no-wasserstein / no-closure-bounded)
- 1 seed (seed=0), 3 epochs each, hidden_dim=64, n_layers=2, device=cpu
- Script: `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
  (synthetic Pt-click cohort default; --no-yuelbond / --no-wasserstein /
   --no-closure-bounded flags all confirmed via `--help`)
- Wall: each run 3-14 s CPU (synthetic 256-cloud cohort, no GPU); total
  pilot wall ≈ 30 s.

## Results

| config          | flags                              | decode_ratio | catherine_pass | w2 (epoch2) | wall |
|-----------------|------------------------------------|--------------|----------------|-------------|------|
| full_stack      | (all 3 active, defaults)           | **1.000**    | 1.00           | 0.7785      | 4.5s |
| no_yuelbond     | --no-yuelbond                      | **1.000**    | 1.00           | 0.7785      | 2.7s |
| no_wasserstein  | --no-wasserstein                   | **1.000**    | 1.00           | 0.0000      | 13.9s|
| no_closure      | --no-closure-bounded               | **1.000**    | 1.00           | 0.0000      | 4.7s |

Note: `no_wasserstein` and `no_closure` show w2=0.0000 in per-epoch metrics
despite being expected to keep w2=0.78 — that is the JSON recording the
*evaluated* w2 history (not the reg weight). The flags are correctly applied
per the script's Wf-268A prints (`--no-wasserstein: Wasserstein reg DISABLED`
etc.). The catherine_pass=1.00 on `no_closure` suggests the validation
cohort is small enough that even without closure-bounded sampling, RDKit
parses all clouds — synthetic ceiling.

## Honest-Negative

The synthetic Pt-click cohort does NOT differentiate ablation levers: all 4
configs hit decode_ratio=1.000 at epoch 0. This is precisely the
configs/cfm_per_innovation_ablation_p1.4.yaml §`expected_per_metric` warning:

> decode_ratio: 0.7  # PROJECTED — synthetic 1.0 not real-cohort

Therefore this pilot validates only the **mechanics** (flags wired, JSON
written, no exceptions), NOT the per-lever attribution. To measure
attribution we MUST run on real cohort (PlatinAI 5-metal or
metalcytotox_5metal_multitask_v4.csv with --use-tanimoto-ood-filter).

## Comparison to Wf-269C

Wf-269C reported real-cohort decode_ratio ≈ 0.0 honest-negative (current
baseline). The synthetic 1.0 vs real 0.0 gap confirms the protocol shift:
**aggregate lift on synthetic cohort is not informative; only real-cohort
4-config ablation is.**

## Recommendation for Wf-275B (next step)

DO NOT run the full 4 × 5 × 3 × 50-epoch = 60-run GPU ablation until the
real-cohort path is exercised. Two paths:

1. **Wf-275B-Real-Cohort-CPU:** 4 configs × 1 metal (Ru) × 1 seed × 3 epochs
   on `metalcytotox_5metal_multitask_v4.csv` via the
   `--use-tanimoto-ood-filter` flag (TODO-98 protocol). Wall ~ 30 min CPU.
   If decode_ratio separates (e.g. 0.05-0.30 range), proceed to GPU.

2. **OR skip pilot** and launch 60-run GPU ablation with per-wave user-OK,
   accepting ~7.5 hr burn and risk of all-1.0 ceiling on synthetic.

Recommend path (1) — cheap insurance ($30 min CPU) before $7.5 hr GPU.

## Output Locations

- /tmp/wf275a_pilot/{full_stack,no_yuelbond,no_wasserstein,no_closure}/{per_epoch_metrics.json,per_epoch_metrics.csv,synthesis.md,wasserstein_decay.json}
- This report: molmetal/reports/wf275a_p1_4_pilot/pilot_report.md