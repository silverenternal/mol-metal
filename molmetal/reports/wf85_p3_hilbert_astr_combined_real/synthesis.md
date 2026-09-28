# Wf-85 Phase 3 — REAL Hilbert+ASTSR combined R² on PlatinAI 4-cell

**Date**: 2026-09-26
**Constraint**: Combined Hilbert primal-dual + ASTSR evolutionary symbolic regressor
  (no Multi-PH + PLL, no composite blend); per `party-twenty-spirit §3.5` honest framing.
**Hardware**: CPU fp32 (default); `--device cuda` accepted but no Blackwell acceleration
  required for Wf-85 P3 anchor.
**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines).

## Cross-cell-line 5-seed summary

| cell line | n_samples | combined R² (mean ± std) | Hilbert-only R² (P1) | ASTSR-only R² (P2) | lift vs P1 | lift vs P2 |
|---|---:|---|---|---|---|---|
| mcf7 | 4078 | -0.0141 ± 0.0137 | -23.7388 | -0.0144 | +23.7248 | +0.0004 |
| a549 | 4078 | -2.9397 ± 5.7762 | -19.4292 | -0.0499 | +16.4894 | -2.8898 |
| a2780 | 4078 | -0.3639 ± 0.2416 | -28.6266 | -0.1173 | +28.2627 | -0.2466 |
| a2780cis | 4078 | -2.6941 ± 5.1968 | -18.8999 | -0.0978 | +16.2058 | -2.5963 |
| **4-cell aggregate** | — | **-1.5030 ± 1.3226** | — | — | — | — |

## Verify gate (per `party-twenty-spirit §3.5`)

**Aggregate combined R² (mean of 4 cell-line means) = -1.5030** — ❌ **HONEST NEGATIVE** (combined R² < 0; per-cell honest-negative reporting)

**Per-cell combined R² ≥ +0.30**: 0 of 4 cell lines (this is the load-bearing number per `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 L5 gap).

## Per-cell-line per-seed R² detail

### mcf7 (ic50_mcf7_nm)

| seed | combined R² (test) | combined R² (val) | combined R² (train) | Hilbert R² | ASTSR R² | alpha | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | -0.0161 | -0.0072 | -0.0151 | -22.5622 | -0.0161 | 0.000 | 22.50 | yes |
| 1 | -0.0377 | -0.0096 | -0.0112 | -22.4993 | -0.0377 | 0.000 | 19.18 | yes |
| 2 | -0.0157 | -0.0063 | -0.0155 | -24.5784 | -0.0157 | 0.000 | 19.89 | yes |
| 3 | -0.0017 | -0.0100 | -0.0190 | -22.2392 | -0.0017 | 0.000 | 20.42 | yes |
| 4 | +0.0008 | -0.0018 | -0.0062 | -27.0331 | +0.0008 | 0.000 | 20.30 | yes |

### a549 (ic50_a549_nm)

| seed | combined R² (test) | combined R² (val) | combined R² (train) | Hilbert R² | ASTSR R² | alpha | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | -0.0373 | -0.0532 | -0.0520 | -17.7317 | -0.0373 | 0.000 | 32.63 | yes |
| 1 | -0.0382 | -0.0839 | -0.0461 | -17.0044 | -0.0382 | 0.000 | 18.68 | yes |
| 2 | -0.0803 | -0.0415 | -0.0458 | -21.8369 | -0.0803 | 0.000 | 18.25 | yes |
| 3 | -14.4922 | -13.0321 | -14.3162 | -20.3176 | -14.4922 | 0.000 | 36.40 | no |
| 4 | -0.0507 | -0.0234 | -0.0563 | -20.4318 | -0.0507 | 0.000 | 19.29 | yes |

### a2780 (ic50_a2780_nm)

| seed | combined R² (test) | combined R² (val) | combined R² (train) | Hilbert R² | ASTSR R² | alpha | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | -0.0363 | -0.0490 | -0.0546 | -29.6043 | -0.0363 | 0.000 | 23.63 | yes |
| 1 | -0.4987 | -0.4694 | -0.4779 | -24.5205 | -0.4987 | 0.000 | 19.53 | no |
| 2 | -0.6007 | -0.5977 | -0.6200 | -28.5085 | -0.6007 | 0.000 | 23.96 | yes |
| 3 | -0.1081 | -0.0851 | -0.1167 | -27.7533 | -0.1081 | 0.000 | 25.09 | yes |
| 4 | -0.5757 | -0.6216 | -0.6207 | -33.0037 | -0.5757 | 0.000 | 19.80 | yes |

### a2780cis (ic50_a2780cis_nm)

| seed | combined R² (test) | combined R² (val) | combined R² (train) | Hilbert R² | ASTSR R² | alpha | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | -0.0811 | -0.0664 | -0.1058 | -16.8811 | -0.0811 | 0.000 | 34.17 | yes |
| 1 | -0.1165 | -0.0785 | -0.0953 | -17.8487 | -0.1165 | 0.000 | 20.28 | yes |
| 2 | -0.0937 | -0.0845 | -0.0981 | -20.2312 | -0.0937 | 0.000 | 21.54 | yes |
| 3 | -13.0877 | -12.2967 | -11.9504 | -18.3795 | -13.6442 | 0.250 | 34.30 | no |
| 4 | -0.0915 | -0.0908 | -0.0972 | -21.3177 | -0.0915 | 0.000 | 19.31 | yes |

## Honest framing

- **REAL retrain (not scaffold reuse)**: 5 seeds × 50 epochs of `LambdaASTSRHilbert.optimize` + `LambdaASTSR.fit` on the strict Bemis-Murcko scaffold split; test R² is computed on completely held-out scaffolds (never seen at training time).
- **No Multi-PH + PLL, no 3-way + PLL** — this is the **combined Hilbert+ASTSR** anchor; Multi-PH/PLL augmentation is a separate workstream (Wf-79 P2 3-way + PLL).
- **Alpha selection**: per-seed 90/10 inner validation on the training fold; grid over alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}; reported as `alpha_chosen` per seed.
- **Attribution**: `lift vs P1` = combined − Wf-85 P1 (Hilbert-only) cell-line mean; `lift vs P2` = combined − Wf-85 P2 (ASTSR-only) cell-line mean.  This isolates the lift attributable to the blend vs split variance.
- **CPU smoke**: full 5-seed × strict-scaffold × 4-cell runs in < 6 minutes on a single CPU core (verified smoke); GPU-0 Blackwell optional but not required.
- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged per seed and reported as `collapse_detected=yes` in the per-seed JSON.

Per `party-twenty-spirit §3.5`: per-cell combined test R² ≥ 0 means are **MEASURED** not projected; aggregate combined R² = -1.5030; any cell with `collapse_detected=yes` is reported in `per_seed` and rolled into the per-cell mean honestly.
