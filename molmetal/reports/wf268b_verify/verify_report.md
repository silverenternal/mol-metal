# Wf-268B-Aggregate — per-split AUC aggregation + v2-baseline verify

**Date:** 2026-09-29  
**Task:** Aggregate per-seed ood_test AUC from Wf-268B-Run output and verify against Wf-185 chemprop v2 baseline.  
**Run output root:** `molmetal/reports/wf268b_run/`  
**Aggregate root:** `molmetal/reports/wf268b_verify/`  

## Wf-268B-Run recipe

Script: `molmetal/scripts/wf263a_verify_drugood.py`  
Args used (per `wf268b_run.log` header):  
- `--configs lbap_refined_ic50_assay lbap_refined_ic50_scaffold lbap_refined_ic50_size`  
- `--seeds 0 1 2 3 4`  
- `--epochs 20 --hidden 300 --depth 3 --head-hidden 300 --batch 64 --lr 0.001`  
- `--device cpu --time-budget-s 10800 --seed-time-budget-s 600 --eval-max-samples 1000`  
- `--chemprop-mode v3-fallback` (forces chemprop v2 BinaryClassificationFFN head)

Note: backbone is chemprop v2 DEFAULTS (h=300/d=3), NOT the h=128/d=2 recipe used in Wf-185. Wf-268B is therefore a different (larger) model compared against the Wf-185 v2 anchor.

## Per-config measured seed counts

| Config | n_seeds_measured / n_expected | missing_seeds |
|---|---:|---|
| lbap_refined_ic50_assay | 1/5 | 1,2,3,4 |
| lbap_refined_ic50_scaffold | 0/5 | 0,1,2,3,4 |
| lbap_refined_ic50_size | 0/5 | 0,1,2,3,4 |

**Total:** 1/15 per-seed JSONs on disk.

## Per-config per-split AUC (mean ± std across measured seeds)

| Config | split | mean | std | n |
|---|---|---:|---:|---:|
| lbap_refined_ic50_assay | train | 0.9260 | NaN | 1 |
| lbap_refined_ic50_assay | iid_val | 0.9162 | NaN | 1 |
| lbap_refined_ic50_assay | iid_test | 0.8885 | NaN | 1 |
| lbap_refined_ic50_assay | ood_val | 0.8129 | NaN | 1 |
| lbap_refined_ic50_assay | ood_test | 0.7016 | NaN | 1 |
| lbap_refined_ic50_scaffold | train | NaN | NaN | 0 |
| lbap_refined_ic50_scaffold | iid_val | NaN | NaN | 0 |
| lbap_refined_ic50_scaffold | iid_test | NaN | NaN | 0 |
| lbap_refined_ic50_scaffold | ood_val | NaN | NaN | 0 |
| lbap_refined_ic50_scaffold | ood_test | NaN | NaN | 0 |
| lbap_refined_ic50_size | train | NaN | NaN | 0 |
| lbap_refined_ic50_size | iid_val | NaN | NaN | 0 |
| lbap_refined_ic50_size | iid_test | NaN | NaN | 0 |
| lbap_refined_ic50_size | ood_val | NaN | NaN | 0 |
| lbap_refined_ic50_size | ood_test | NaN | NaN | 0 |

## v2 baseline (Wf-185) + delta vs Wf-268B

| Config | v2_mean | v2_baseline (Wf-185) | v3-fallback (Wf-268B) | delta | passes? |
|---|---:|---:|---:|---:|:---:|
| lbap_refined_ic50_assay | 0.7137 | 0.7137 | 0.7016 | -0.0121 | FAIL |
| lbap_refined_ic50_scaffold | 0.6707 | 0.6707 | NaN (insufficient data) | NaN | FAIL |
| lbap_refined_ic50_size | 0.6757 | 0.6757 | NaN (insufficient data) | NaN | FAIL |

## Per-seed ood_test AUC (raw)

| Config | seed | ood_test | epochs_done | status | elapsed_s |
|---|---:|---:|---:|---|---:|
| lbap_refined_ic50_assay | 0 | 0.7016 | 8 | timeout | 815.4 |
| lbap_refined_ic50_assay | 1 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_assay | 2 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_assay | 3 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_assay | 4 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_scaffold | 0 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_scaffold | 1 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_scaffold | 2 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_scaffold | 3 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_scaffold | 4 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_size | 0 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_size | 1 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_size | 2 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_size | 3 | (NOT MEASURED) | — | — | — |
| lbap_refined_ic50_size | 4 | (NOT MEASURED) | — | — | — |

## Acceptance gates (FLAT schema)

| Gate | Value |
|---|---:|
| G1_assay_mean | 0.7016 |
| G2_scaffold_mean | NaN |
| G3_size_mean | NaN |
| G4_assay_above_v2 | False |
| G5_scaffold_above_v2 | False |
| G6_size_above_v2 | False |
| G7_report_path | /home/hugo/codes/mol-metal/molmetal/reports/wf268b_verify/verify_report.md |

## Honest framing

HONEST-NEGATIVE: Wf-268B-Run produced only 1/15 per-seed JSON outputs. Std is NaN where n<2; per-config mean over a partial seed set is NOT comparable to the Wf-185 5-seed v2 baseline. The single completed seed (lbap_refined_ic50_assay seed=0) reported ood_test=0.7016, which is below the Wf-185 v2 baseline of 0.7137, but this is a 1-seed point estimate with seed_time_budget=600s hit at epoch=8 — NOT a 5-seed ablation. Re-run Wf-268B-Run with a larger wall budget to complete the remaining seeds before drawing any v2-vs-v3-fallback conclusion.

### What Wf-268B-Run measured

- `lbap_refined_ic50_assay` seed=0: ood_test=0.7016 (8/20 epochs, status=timeout, elapsed=815s). Seed-time-budget=600s hit at epoch=8; same timeout pattern as Wf-185 (which also aborted seed=0/1 of the assay config at 6-8 epochs).
- `lbap_refined_ic50_assay` seed=1..4: log shows epoch=1 started (loss=0.2501, elapsed=66.6s) but no further epoch/FINAL lines — the run was interrupted before seed=1 completed.
- `lbap_refined_ic50_scaffold` and `lbap_refined_ic50_size`: 0/5 seeds each — neither config was reached before the run was interrupted.

### Why Wf-268B-Aggregate cannot pass

- Task spec requires mean ± std across 5 seeds per config. With 1 seed measured (assay seed=0) and 0 measured for the other two configs, we cannot compute a 5-seed mean ± std for ANY config.
- The single 1-seed point estimate for assay is 0.7016 vs Wf-185 v2 baseline 0.7137 → below v2 by 0.0121, BUT this is a 1-seed point estimate that timed out at 8/20 epochs on a *larger* backbone (h=300/d=3 vs Wf-185 h=128/d=2). Drawing a v2-vs-v3-fallback conclusion from one timed-out seed is unsound.
- G4/G5/G6 are therefore False — not because v3-fallback is demonstrably worse than v2, but because there is not enough measured data to make the comparison. Re-run Wf-268B-Run with a larger wall budget (>= 6 h) to complete all 15 seeds before re-running Wf-268B-Aggregate.

### Files

* `per_split_summary.json` — full per-config + FLAT-schema gate payload
* `verify_report.md` — this file
* `aggregate.py` — READ-ONLY aggregator (no result.json writes)
