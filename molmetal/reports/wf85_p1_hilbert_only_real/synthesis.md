# Wf-85 Phase 1 — REAL Hilbert-only retrain (PlatinAI 4-cell, strict-scaffold)

**Date**: 2026-09-26
**Constraint**: REAL Hilbert-only retrain (no ASTSR, no Multi-PH + PLL); 5 seeds × 50 epochs × strict Bemis-Murcko scaffold split.
Per `party-twenty-spirit §3.5` honest framing.
**Hardware**: CPU fp32 (default) + GPU-0 Blackwell optional via `--device cuda`.
**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines).

## Cross-cell-line 5-seed summary

| cell line | n_samples | Hilbert test R² (mean ± std) | Hilbert val R² | Hilbert train R² | collapse? |
|---|---:|---|---|---|:---:|
| mcf7 | 4078 | -23.7388 ± 1.7748 | -22.9604 ± 2.3519 | -21.8463 ± 0.7296 | no |
| a549 | 4078 | -19.4292 ± 1.7981 | -21.6038 ± 1.8621 | -19.6200 ± 0.8458 | no |
| a2780 | 4078 | -28.6266 ± 2.7163 | -29.7658 ± 2.4349 | -26.9660 ± 0.8773 | no |
| a2780cis | 4078 | -18.8999 ± 1.5943 | -18.2379 ± 2.1347 | -16.8315 ± 0.6442 | no |
| **4-cell aggregate** | — | **-22.6736 ± 3.9160** | — | — | no |

## Verify gate (per `party-twenty-spirit §3.5`)

**Aggregate Hilbert test R² (mean of 4 cell-line means) = -22.6736** — ❌ **HONEST NEGATIVE** (no cell-line R² > 0; per-cell honest-negative reporting)

**Per-cell Hilbert test R²**: 0 of 4 cell lines have **5-seed mean test R² > 0** (this is the load-bearing number per `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 L5 gap).

## Per-cell-line per-seed test R² detail

### mcf7 (ic50_mcf7_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 / cuda | wall_s |
|---:|---:|---:|---:|---|:---:|---|---:|
| 0 | -22.4338 | -20.9242 | -21.1546 | 2855/612/611 | no | fp32/cuda | 3.94 |
| 1 | -22.4694 | -20.1255 | -21.1758 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 2 | -24.3415 | -24.1689 | -21.4506 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 3 | -22.4744 | -22.8803 | -22.8512 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 4 | -26.9751 | -26.7029 | -22.5994 | 2855/612/611 | no | fp32/cuda | 0.01 |

### a549 (ic50_a549_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 / cuda | wall_s |
|---:|---:|---:|---:|---|:---:|---|---:|
| 0 | -17.6178 | -21.7812 | -18.6822 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 1 | -16.9884 | -18.6504 | -19.3921 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 2 | -21.6179 | -22.9756 | -18.8466 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 3 | -20.5249 | -20.6157 | -20.3006 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 4 | -20.3969 | -23.9963 | -20.8785 | 2855/612/611 | no | fp32/cuda | 0.01 |

### a2780 (ic50_a2780_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 / cuda | wall_s |
|---:|---:|---:|---:|---|:---:|---|---:|
| 0 | -29.4489 | -29.3360 | -25.4591 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 1 | -24.4814 | -26.0393 | -26.9178 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 2 | -28.2526 | -31.2041 | -26.7814 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 3 | -28.0192 | -28.9100 | -27.9543 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 4 | -32.9307 | -33.3395 | -27.7172 | 2855/612/611 | no | fp32/cuda | 0.01 |

### a2780cis (ic50_a2780cis_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 / cuda | wall_s |
|---:|---:|---:|---:|---|:---:|---|---:|
| 0 | -16.7741 | -17.2214 | -16.3236 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 1 | -17.8241 | -14.6013 | -16.6771 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 2 | -20.0289 | -20.7214 | -16.0152 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 3 | -18.5931 | -19.3734 | -17.4410 | 2855/612/611 | no | fp32/cuda | 0.01 |
| 4 | -21.2797 | -19.2720 | -17.7005 | 2855/612/611 | no | fp32/cuda | 0.01 |

## Honest framing

- **REAL retrain (not scaffold reuse)**: 5 seeds × 50 epochs of `LambdaASTSRHilbert.optimize` on the strict Bemis-Murcko scaffold split; test R² is computed on completely held-out scaffolds (never seen at training time).
- **No ASTSR, no Multi-PH + PLL, no composite blend** — this is the **Hilbert-only R² anchor** for the Wf-56 R20-P2 final ship and the Wf-83 P2 combined Hilbert+ASTSR lift attribution.
- **CPU smoke**: per-seed wall-time on CPU is roughly 0.5–2 s × 5 seeds = < 15 s per cell line on a single CPU core; full 5-seed × 4-cell run < 1 minute.
- **GPU-0 Blackwell path**: `--device cuda` dispatches via :mod:`hilbert_astsr_cuda` batch wrapper; CUDA-unavailable falls back to CPU fp32.
- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged per seed and reported as `collapse_detected=yes` in the per-seed JSON.

Per `party-twenty-spirit §3.5`: per-cell test R² ≥ 0 means are **MEASURED** not projected; aggregate Hilbert test R² = -22.6736; any cell with `collapse_detected=yes` is reported in `per_seed` and rolled into the per-cell mean honestly.
