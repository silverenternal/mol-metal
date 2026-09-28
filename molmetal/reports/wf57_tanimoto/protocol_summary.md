# Wf-57 Phase 1 — Tanimoto<0.4 OOD protocol (tmQM d-block)

Tanimoto OOD protocol applied to the **tmQM d-block subset** (2244 real mononuclear transition-metal complexes across 30 metals — Balcells & Skjelstad 2020).  Per-cluster train/test split + per-cluster Tanimoto matrix + aggregate OOD claim.  Fills the Wf-57 P1 verification target **>= 80% of (train, test) cross-pairs with Tanimoto < 0.40** (Pocket2Mol/TargetDiff convention).

## Run metadata

**Date (UTC)**: 2026-09-26T08:19:39Z  
**Tool**: `bench_ood_tanimoto_protocol.py`  
**Pilot ID**: `wf57_tanimoto_ood`  
**Train ratio**: 0.800  
**Tanimoto threshold**: 0.400  
**Verification target**: 0.800  
**FP radius / n_bits**: 2 / 2048  
**tmQM CSV**: `/home/hugo/codes/mol-metal/molmetal/data/tmqm_dblock_subset.csv`  
**Max samples / cluster**: 100  
**Total wall**: 37.73 s  

## Aggregate OOD claim

| Metric | Value |
|--------|-------|
| N clusters | 30 |
| N samples (total) | 2244 |
| N (train, test) cross-pairs | 32256 |
| N pairs below threshold | 32185 |
| fraction_below_threshold | 0.9978 |
| mean Tanimoto (cross-side) | 0.0973 |
| median Tanimoto (cross-side) | 0.0896 |
| max Tanimoto (cross-side) | 1.0000 |
| Verification (0.80) | **PASS** |

## Per-cluster summary

| Cluster | N total | N train | N test | train_ratio | OOD rate | train-test max Tan. | train-test mean Tan. |
|---------|---------|---------|--------|------------|----------|---------------------|----------------------|
| Ag | 81 | 65 | 16 | 0.802 | 0.938 | 0.531 | 0.101 |
| Au | 100 | 80 | 20 | 0.800 | 0.850 | 0.537 | 0.117 |
| Cd | 85 | 68 | 17 | 0.800 | 0.824 | 0.469 | 0.100 |
| Co | 100 | 80 | 20 | 0.800 | 0.900 | 0.449 | 0.089 |
| Cr | 100 | 80 | 20 | 0.800 | 0.900 | 0.515 | 0.111 |
| Cu | 100 | 80 | 20 | 0.800 | 0.800 | 0.720 | 0.095 |
| Fe | 100 | 80 | 20 | 0.800 | 0.900 | 0.473 | 0.093 |
| Hf | 20 | 16 | 4 | 0.800 | 1.000 | 0.227 | 0.087 |
| Hg | 61 | 49 | 12 | 0.803 | 0.917 | 0.423 | 0.084 |
| Ir | 100 | 80 | 20 | 0.800 | 0.800 | 0.508 | 0.103 |
| La | 3 | 2 | 1 | 0.667 | 1.000 | 0.086 | 0.043 |
| Mn | 100 | 80 | 20 | 0.800 | 0.850 | 0.596 | 0.088 |
| Mo | 100 | 80 | 20 | 0.800 | 0.800 | 0.727 | 0.087 |
| Nb | 19 | 15 | 4 | 0.789 | 1.000 | 0.356 | 0.081 |
| Ni | 100 | 80 | 20 | 0.800 | 0.950 | 0.432 | 0.087 |
| Os | 55 | 44 | 11 | 0.800 | 1.000 | 0.303 | 0.096 |
| Pd | 100 | 80 | 20 | 0.800 | 1.000 | 0.319 | 0.107 |
| Pt | 100 | 80 | 20 | 0.800 | 0.950 | 0.541 | 0.091 |
| Re | 100 | 80 | 20 | 0.800 | 0.950 | 0.635 | 0.100 |
| Rh | 100 | 80 | 20 | 0.800 | 0.900 | 0.462 | 0.094 |
| Ru | 100 | 80 | 20 | 0.800 | 0.850 | 0.521 | 0.103 |
| Sc | 20 | 16 | 4 | 0.800 | 1.000 | 0.368 | 0.125 |
| Ta | 30 | 24 | 6 | 0.800 | 1.000 | 0.352 | 0.093 |
| Tc | 15 | 12 | 3 | 0.800 | 0.667 | 0.515 | 0.094 |
| Ti | 100 | 80 | 20 | 0.800 | 0.950 | 0.535 | 0.086 |
| V | 72 | 58 | 14 | 0.806 | 0.929 | 1.000 | 0.107 |
| W | 100 | 80 | 20 | 0.800 | 0.900 | 0.410 | 0.108 |
| Y | 30 | 24 | 6 | 0.800 | 0.667 | 0.513 | 0.099 |
| Zn | 100 | 80 | 20 | 0.800 | 0.950 | 0.429 | 0.094 |
| Zr | 53 | 42 | 11 | 0.792 | 0.909 | 0.435 | 0.089 |

## Honest framing

- **CPU-only.**  No CUDA / Triton / CFM retraining.
- **Deterministic.**  Per-cluster split is hash-stable on CSD code; re-running with the same flags reproduces every per-sample Tanimoto.
- **Real chemistry.**  tmQM d-block (Wf-28 ship) — 5,000 real mononuclear transition-metal complexes, 30 d-block metals.
- **Per-cluster (per-metal) split.**  Stratification key is `metal_center`.  Each cluster is split 80/20 (default); the `test_test_max_tanimoto` column flags clusters whose test samples are chemically similar (i.e. clusters where the OOD claim is weakest).
- **Verification target.**  Wf-57 P1 verification: **>= 80%** of (train, test) cross-pairs across all clusters must have Tanimoto < 0.40.  The aggregate row above is the single number to check.
- **Honest-negative framing.**  Clusters with `test_test_max_tanimoto >= 0.4` are reported as not OOD — not silently dropped.  The CSV row for those clusters preserves the max value for downstream audit.
- **Bridge to Wf-41 P2 / Wf-44 P1-P2.**  This script is the bench layer that proves the tmQM d-block split is a **genuine OOD** split — the same protocol is what Wf-41 P2 used to validate the PlatinAI 5K + MetalCytoToxDB cross-domain OOD claim (`ood_closure.json` 0.5581 plat / 0.0300 metalcyt).
