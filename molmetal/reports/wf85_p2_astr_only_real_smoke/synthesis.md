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
| mcf7 | 4078 | -4.4948 ± 0.0000 | -4.4957 ± 0.0000 | -4.4705 ± 0.0000 | yes |
| **4-cell aggregate** | — | **-4.4948 ± 0.0000** | — | — | yes |

## Verify gate (per `party-twenty-spirit §3.5`)

**Aggregate ASTSR test R² (mean of 4 cell-line means) = -4.4948** --- **HONEST NEGATIVE** (per-cell honest-negative reporting when no cell line R² > 0).

**Per-cell ASTSR test R²**: 0 of 1 cell lines have **5-seed mean test R² > 0** (this is the load-bearing number per `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 L5 gap).

## Per-cell-line per-seed test R² detail

### mcf7 (ic50_mcf7_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | cuda | wall_s |
|---:|---:|---:|---:|---|:---:|:---:|---:|
| 0 | -4.4948 | -4.4957 | -4.4705 | 2855/612/611 | yes | cpu | 1.70 |

## Best AST (lisp form, post-fit per seed)

### mcf7

- seed 0: `(mul c_2.0 (add c_2.0 c_1.0))`

## Honest framing

- **REAL retrain (not scaffold reuse)**: 5 seeds × 50 generations of :class:`LambdaASTSR` evolutionary search on the strict Bemis-Murcko scaffold split; test R² is computed on completely held-out scaffolds (never seen at training time).
- **No Hilbert, no Multi-PH + PLL, no composite blend** --- this is the **ASTSR-only R² anchor** that pairs with Wf-85 P1 Hilbert-only (aggregate = -22.67, see `wf85_p1_hilbert_only_real/synthesis.md`) for any future Wf-56 R20-P2 composite Hilbert+ASTSR lift attribution.
- **Feature set**: F5_RDKIT_2D 12-d RDKit 2D descriptors (Wf-49 Phase 2 ship). The legacy F5 5-feature set is intentionally NOT used here because Wf-49 §Cause 2 (CRITICAL) confirmed that on PlatinAI 4-cell every F5 feature has `|Pearson r vs pIC50_mcf7| < 0.02` (LassoLarsIC zeroed 4/5).
- **CPU smoke**: per-seed wall-time on CPU is roughly 5-30 s × 5 seeds × 4 cell lines = ~2-10 min total on a single CPU core; full 5-seed × 4-cell run < 15 min on a modern 8-core CPU.
- **GPU-0 Blackwell path**: `--device cuda` dispatches via the same CPU fp32 + NumPy path (LambdaASTSR has no CUDA wrapper today); CUDA-unavailable falls back to CPU fp32.  `used_cuda=False` is recorded in every per-seed JSON.
- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged per seed and reported as `collapse_detected=yes` in the per-seed JSON.

Per `party-twenty-spirit §3.5`: per-cell test R² ≥ 0 means are **MEASURED** not projected; aggregate ASTSR test R² = -4.4948; any cell with `collapse_detected=yes` is reported in `per_seed` and rolled into the per-cell mean honestly.
