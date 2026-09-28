# Wf-85 Phase 2 — REAL ASTSR-only retrain (PlatinAI 4-cell, strict-scaffold)

**Date**: 2026-09-26
**Constraint**: REAL ASTSR-only retrain (no Hilbert, no Multi-PH + PLL); 5 seeds × 50 epochs (= 50 AST generations) × strict Bemis-Murcko scaffold split.
Per `party-twenty-spirit §3.5` honest framing.
**Feature set**: `rdkit_2d` (F5_RDKIT_2D 12-d RDKit 2D by default).
**Hardware**: CPU fp32 (default) + GPU-0 Blackwell optional via `--device cuda` (LambdaASTSR has no CUDA wrapper today — `--device cuda` is accepted for symmetry with Wf-85 P1 and delegates to the same CPU fp32 ASTSR path; `used_cuda=False` is recorded per seed).
**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines).

## Cross-cell-line 5-seed summary

| cell line | n_samples | ASTSR test R² (mean ± std) | ASTSR val R² | ASTSR train R² | collapse? |
|---|---:|---|---|---|:---:|
| mcf7 | 4078 | -0.0144 ± 0.0133 | -0.0100 ± 0.0037 | -0.0157 ± 0.0026 | yes |
| a549 | 4078 | -0.0499 ± 0.0159 | -0.0570 ± 0.0236 | -0.0490 ± 0.0044 | yes |
| a2780 | 4078 | -0.1173 ± 0.0362 | -0.1054 ± 0.0229 | -0.0938 ± 0.0244 | yes |
| a2780cis | 4078 | -0.0978 ± 0.0123 | -0.0847 ± 0.0123 | -0.0976 ± 0.0046 | yes |
| **4-cell aggregate** | — | **-0.0699 ± 0.0403** | — | — | yes |

## Verify gate (per `party-twenty-spirit §3.5`)

**Aggregate ASTSR test R² (mean of 4 cell-line means) = -0.0699** --- **HONEST NEGATIVE** (per-cell honest-negative reporting when no cell line R² > 0).

**Per-cell ASTSR test R²**: 0 of 4 cell lines have **5-seed mean test R² > 0** (this is the load-bearing number per `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 L5 gap).

## Per-cell-line per-seed test R² detail

### mcf7 (ic50_mcf7_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | cuda | wall_s |
|---:|---:|---:|---:|---|:---:|:---:|---:|
| 0 | -0.0161 | -0.0072 | -0.0151 | 2855/612/611 | yes | cpu | 12.47 |
| 1 | -0.0377 | -0.0096 | -0.0112 | 2855/612/611 | yes | cpu | 11.43 |
| 2 | -0.0157 | -0.0063 | -0.0155 | 2855/612/611 | yes | cpu | 9.42 |
| 3 | -0.0017 | -0.0100 | -0.0189 | 2855/612/611 | yes | cpu | 11.34 |
| 4 | -0.0011 | -0.0168 | -0.0176 | 2855/612/611 | yes | cpu | 9.39 |

### a549 (ic50_a549_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | cuda | wall_s |
|---:|---:|---:|---:|---|:---:|:---:|---:|
| 0 | -0.0373 | -0.0532 | -0.0520 | 2855/612/611 | yes | cpu | 9.67 |
| 1 | -0.0382 | -0.0839 | -0.0461 | 2855/612/611 | yes | cpu | 11.52 |
| 2 | -0.0803 | -0.0415 | -0.0458 | 2855/612/611 | yes | cpu | 9.84 |
| 3 | -0.0432 | -0.0830 | -0.0449 | 2855/612/611 | yes | cpu | 11.56 |
| 4 | -0.0507 | -0.0234 | -0.0563 | 2855/612/611 | yes | cpu | 10.44 |

### a2780 (ic50_a2780_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | cuda | wall_s |
|---:|---:|---:|---:|---|:---:|:---:|---:|
| 0 | -0.1509 | -0.1283 | -0.0996 | 2855/612/611 | yes | cpu | 9.56 |
| 1 | -0.1111 | -0.0908 | -0.1146 | 2855/612/611 | yes | cpu | 10.03 |
| 2 | -0.1161 | -0.1082 | -0.1095 | 2855/612/611 | yes | cpu | 9.30 |
| 3 | -0.0541 | -0.0698 | -0.0466 | 2855/612/611 | yes | cpu | 11.28 |
| 4 | -0.1544 | -0.1299 | -0.0986 | 2855/612/611 | yes | cpu | 11.22 |

### a2780cis (ic50_a2780cis_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | cuda | wall_s |
|---:|---:|---:|---:|---|:---:|:---:|---:|
| 0 | -0.0811 | -0.0664 | -0.1058 | 2855/612/611 | yes | cpu | 9.58 |
| 1 | -0.1165 | -0.0785 | -0.0953 | 2855/612/611 | yes | cpu | 10.67 |
| 2 | -0.0937 | -0.0845 | -0.0981 | 2855/612/611 | yes | cpu | 9.71 |
| 3 | -0.1063 | -0.1032 | -0.0916 | 2855/612/611 | yes | cpu | 13.08 |
| 4 | -0.0915 | -0.0908 | -0.0972 | 2855/612/611 | yes | cpu | 12.48 |

## Best AST (lisp form, post-fit per seed)

### mcf7

- seed 0: `(div (add c_2.0 c_0.5) c_0.5)`
- seed 1: `(add c_2.0 (sub c_2.0 c_-1.0))`
- seed 2: `(add (add c_1.0 (max c_2.0 c_0.5)) c_2.0)`
- seed 3: `(max (add (max c_1.0 c_-1.0) (div c_2.0 c_0.5)) (mul c_1.0 v_RingCount))`
- seed 4: `(add (add c_2.0 c_2.0) c_1.0)`

### a549

- seed 0: `(div (add c_2.0 c_0.5) c_0.5)`
- seed 1: `(add c_2.0 (sub c_2.0 c_-1.0))`
- seed 2: `(add (add c_1.0 (max c_2.0 c_0.5)) c_2.0)`
- seed 3: `(max (add (max c_1.0 c_-1.0) (div c_2.0 c_0.5)) (mul c_1.0 v_RingCount))`
- seed 4: `(add (add c_2.0 c_2.0) c_1.0)`

### a2780

- seed 0: `(add (div (add c_2.0 c_0.5) c_0.5) c_0.5)`
- seed 1: `(add (add c_2.0 (sub c_2.0 c_-1.0)) c_0.5)`
- seed 2: `(add (add c_1.0 (add c_2.0 c_0.5)) c_2.0)`
- seed 3: `(add (add (max c_1.0 c_0.0) (div c_2.0 c_0.5)) (mul c_0.5 c_0.5))`
- seed 4: `(add (add (add c_2.0 c_2.0) (min c_2.0 c_1.0)) c_0.5)`

### a2780cis

- seed 0: `(div (add c_2.0 c_0.5) c_0.5)`
- seed 1: `(add c_2.0 (sub c_2.0 c_-1.0))`
- seed 2: `(add (add c_1.0 (max c_2.0 c_0.5)) c_2.0)`
- seed 3: `(max (add (max c_1.0 c_-1.0) (div c_2.0 c_0.5)) (mul c_1.0 c_0.5))`
- seed 4: `(add (add c_2.0 c_2.0) c_1.0)`

## Honest framing

- **REAL retrain (not scaffold reuse)**: 5 seeds × 50 generations of :class:`LambdaASTSR` evolutionary search on the strict Bemis-Murcko scaffold split; test R² is computed on completely held-out scaffolds (never seen at training time).
- **No Hilbert, no Multi-PH + PLL, no composite blend** --- this is the **ASTSR-only R² anchor** that pairs with Wf-85 P1 Hilbert-only (aggregate = -22.67, see `wf85_p1_hilbert_only_real/synthesis.md`) for any future Wf-56 R20-P2 composite Hilbert+ASTSR lift attribution.
- **Feature set**: F5_RDKIT_2D 12-d RDKit 2D descriptors (Wf-49 Phase 2 ship). The legacy F5 5-feature set is intentionally NOT used here because Wf-49 §Cause 2 (CRITICAL) confirmed that on PlatinAI 4-cell every F5 feature has `|Pearson r vs pIC50_mcf7| < 0.02` (LassoLarsIC zeroed 4/5).
- **CPU smoke**: per-seed wall-time on CPU is roughly 5-30 s × 5 seeds × 4 cell lines = ~2-10 min total on a single CPU core; full 5-seed × 4-cell run < 15 min on a modern 8-core CPU.
- **GPU-0 Blackwell path**: `--device cuda` dispatches via the same CPU fp32 + NumPy path (LambdaASTSR has no CUDA wrapper today); CUDA-unavailable falls back to CPU fp32.  `used_cuda=False` is recorded in every per-seed JSON.
- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged per seed and reported as `collapse_detected=yes` in the per-seed JSON.

Per `party-twenty-spirit §3.5`: per-cell test R² ≥ 0 means are **MEASURED** not projected; aggregate ASTSR test R² = -0.0699; any cell with `collapse_detected=yes` is reported in `per_seed` and rolled into the per-cell mean honestly.
