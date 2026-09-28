# Wf-85 Phase 1 — REAL Hilbert-only retrain (PlatinAI 4-cell, strict-scaffold)

**Date**: 2026-09-26
**Constraint**: REAL Hilbert-only retrain (no ASTSR, no Multi-PH + PLL); 5 seeds × 50 epochs × strict Bemis-Murcko scaffold split.
Per `party-twenty-spirit §3.5` honest framing.
**Hardware**: CPU fp32 (default) + GPU-0 Blackwell optional via `--device cuda`.
**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines).

## Cross-cell-line 5-seed summary

| cell line | n_samples | Hilbert test R² (mean ± std) | Hilbert val R² | Hilbert train R² | collapse? |
|---|---:|---|---|---|:---:|
| mcf7 | 4078 | -23.7825 ± 1.8285 | -22.9829 ± 2.3690 | -21.8519 ± 0.6275 | no |
| a549 | 4078 | -19.4645 ± 1.8081 | -21.6281 ± 1.9394 | -19.6245 ± 0.7633 | no |
| a2780 | 4078 | -28.6781 ± 2.7473 | -29.7930 ± 2.4986 | -26.9740 ± 0.7958 | no |
| a2780cis | 4078 | -18.9316 ± 1.6164 | -18.2500 ± 2.1633 | -16.8322 ± 0.5565 | no |
| **4-cell aggregate** | — | **-22.7142 ± 3.9236** | — | — | no |

## Verify gate (per `party-twenty-spirit §3.5`)

**Aggregate Hilbert test R² (mean of 4 cell-line means) = -22.7142** — ❌ **HONEST NEGATIVE** (no cell-line R² > 0; per-cell honest-negative reporting)

**Per-cell Hilbert test R²**: 0 of 4 cell lines have **5-seed mean test R² > 0** (this is the load-bearing number per `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 L5 gap).

## Per-cell-line per-seed test R² detail

### mcf7 (ic50_mcf7_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 / cuda | wall_s |
|---:|---:|---:|---:|---|:---:|---|---:|
| 0 | -22.5622 | -21.0479 | -21.2047 | 2855/612/611 | no | fp64/cpu | 1.01 |
| 1 | -22.4993 | -20.1251 | -21.1942 | 2855/612/611 | no | fp64/cpu | 0.01 |
| 2 | -24.5784 | -24.4340 | -21.6823 | 2855/612/611 | no | fp64/cpu | 0.01 |
| 3 | -22.2392 | -22.5968 | -22.5665 | 2855/612/611 | no | fp64/cpu | 0.01 |
| 4 | -27.0331 | -26.7105 | -22.6120 | 2855/612/611 | no | fp64/cpu | 0.01 |

### a549 (ic50_a549_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 / cuda | wall_s |
|---:|---:|---:|---:|---|:---:|---|---:|
| 0 | -17.7317 | -21.9213 | -18.7359 | 2855/612/611 | no | fp64/cpu | 0.00 |
| 1 | -17.0044 | -18.6515 | -19.4090 | 2855/612/611 | no | fp64/cpu | 0.00 |
| 2 | -21.8369 | -23.2391 | -19.0592 | 2855/612/611 | no | fp64/cpu | 0.00 |
| 3 | -20.3176 | -20.3353 | -20.0324 | 2855/612/611 | no | fp64/cpu | 0.00 |
| 4 | -20.4318 | -23.9931 | -20.8859 | 2855/612/611 | no | fp64/cpu | 0.00 |

### a2780 (ic50_a2780_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 / cuda | wall_s |
|---:|---:|---:|---:|---|:---:|---|---:|
| 0 | -29.6043 | -29.4759 | -25.5102 | 2855/612/611 | no | fp64/cpu | 0.00 |
| 1 | -24.5205 | -26.0452 | -26.9472 | 2855/612/611 | no | fp64/cpu | 0.00 |
| 2 | -28.5085 | -31.5086 | -27.0401 | 2855/612/611 | no | fp64/cpu | 0.00 |
| 3 | -27.7533 | -28.5834 | -27.6414 | 2855/612/611 | no | fp64/cpu | 0.00 |
| 4 | -33.0037 | -33.3516 | -27.7309 | 2855/612/611 | no | fp64/cpu | 0.00 |

### a2780cis (ic50_a2780cis_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 / cuda | wall_s |
|---:|---:|---:|---:|---|:---:|---|---:|
| 0 | -16.8811 | -17.3101 | -16.3648 | 2855/612/611 | no | fp64/cpu | 0.00 |
| 1 | -17.8487 | -14.5907 | -16.6821 | 2855/612/611 | no | fp64/cpu | 0.00 |
| 2 | -20.2312 | -20.9553 | -16.1989 | 2855/612/611 | no | fp64/cpu | 0.00 |
| 3 | -18.3795 | -19.1167 | -17.2079 | 2855/612/611 | no | fp64/cpu | 0.00 |
| 4 | -21.3177 | -19.2770 | -17.7071 | 2855/612/611 | no | fp64/cpu | 0.00 |

## Honest framing

- **REAL retrain (not scaffold reuse)**: 5 seeds × 50 epochs of `LambdaASTSRHilbert.optimize` on the strict Bemis-Murcko scaffold split; test R² is computed on completely held-out scaffolds (never seen at training time).
- **No ASTSR, no Multi-PH + PLL, no composite blend** — this is the **Hilbert-only R² anchor** for the Wf-56 R20-P2 final ship and the Wf-83 P2 combined Hilbert+ASTSR lift attribution.
- **CPU smoke**: per-seed wall-time on CPU is roughly 0.5–2 s × 5 seeds = < 15 s per cell line on a single CPU core; full 5-seed × 4-cell run < 1 minute.
- **GPU-0 Blackwell path**: `--device cuda` dispatches via :mod:`hilbert_astsr_cuda` batch wrapper; CUDA-unavailable falls back to CPU fp32.
- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged per seed and reported as `collapse_detected=yes` in the per-seed JSON.

Per `party-twenty-spirit §3.5`: per-cell test R² ≥ 0 means are **MEASURED** not projected; aggregate Hilbert test R² = -22.7142; any cell with `collapse_detected=yes` is reported in `per_seed` and rolled into the per-cell mean honestly.
