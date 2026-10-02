# Wf-348-B3 — Hilbert R² on R20-P3 features (L5 gate verification)

**Date**: 2026-09-29
**Constraint**: Real Hilbert regression on R20-P3 features (1024-d Multi-PH 32-bin Fourier ⊕ 128-d PLL) 5 seeds × 50 epochs × strict-scaffold on PlatinAI 4-cell.
Per `party-twenty-spirit §3.5` honest framing.
**Hardware**: CPU fp32 (default) + optional fp64 fallback.
**Inputs**: `molmetal/data/platinai_4cell.csv` (500 rows × 4 cell lines).

## Cross-cell-line 5-seed summary

| cell line | n_samples | Hilbert test R² (mean ± std) | Hilbert val R² | Hilbert train R² | collapse? |
|---|---:|---|---|---|:---:|
| mcf7 | 500 | -1.4713 ± 0.4920 | -1.4072 ± 0.3120 | -0.9389 ± 0.0295 | no |
| a549 | 500 | -0.8868 ± 0.2440 | -0.9051 ± 0.2281 | -0.7277 ± 0.0299 | no |
| a2780 | 500 | -1.7658 ± 0.2025 | -2.0340 ± 0.5395 | -1.6156 ± 0.1184 | no |
| a2780cis | 500 | -0.8100 ± 0.1560 | -0.7590 ± 0.1802 | -0.5872 ± 0.0313 | no |
| **4-cell aggregate** | — | **-1.2335 ± 0.3998** | — | — | no |

| **4-cell joint (seed=0 test split)** | — | **-0.8067** | — | — | — |

## Verdict (honesty gate)

**Hilbert test R² per-cohort aggregate (mean of 4 cell-line means) = -1.2335**
**Hilbert test R² joint (across 4 cell-lines, seed=0 test split) = -0.8067**
**Cells with R² ≥ 0.6: 0 / 4**
**Cells with R² > 0: 0 / 4**

**Gate status: PRESERVED** — aggregate R² -1.2335 < 0.6; joint R² -0.8067 < 0.6
**Verdict label**: HONEST_NEGATIVE_PARTIAL_STRUCTURAL

## Per-cell-line per-seed test R² detail

### mcf7 (ic50_mcf7_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 | wall_s |
|---:|---:|---:|---:|---|:---:|---|---:|
| 0 | -1.2598 | -1.1289 | -0.9836 | 350/75/75 | no | fp32 | 0.00 |
| 1 | -2.3441 | -1.8442 | -0.9190 | 350/75/75 | no | fp32 | 0.00 |
| 2 | -1.2782 | -1.1000 | -0.9457 | 350/75/75 | no | fp32 | 0.00 |
| 3 | -0.8787 | -1.2438 | -0.9498 | 350/75/75 | no | fp32 | 0.00 |
| 4 | -1.5955 | -1.7192 | -0.8966 | 350/75/75 | no | fp32 | 0.00 |

### a549 (ic50_a549_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 | wall_s |
|---:|---:|---:|---:|---|:---:|---|---:|
| 0 | -0.5106 | -1.3086 | -0.7617 | 350/75/75 | no | fp32 | 0.00 |
| 1 | -0.9224 | -0.8668 | -0.7499 | 350/75/75 | no | fp32 | 0.00 |
| 2 | -1.2470 | -0.9685 | -0.6794 | 350/75/75 | no | fp32 | 0.00 |
| 3 | -0.7654 | -0.6954 | -0.7087 | 350/75/75 | no | fp32 | 0.00 |
| 4 | -0.9886 | -0.6864 | -0.7388 | 350/75/75 | no | fp32 | 0.00 |

### a2780 (ic50_a2780_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 | wall_s |
|---:|---:|---:|---:|---|:---:|---|---:|
| 0 | -1.9225 | -2.7539 | -1.5782 | 350/75/75 | no | fp32 | 0.00 |
| 1 | -1.3880 | -1.3841 | -1.8500 | 350/75/75 | no | fp32 | 0.01 |
| 2 | -1.9184 | -1.9447 | -1.5321 | 350/75/75 | no | fp32 | 0.00 |
| 3 | -1.7223 | -1.5403 | -1.5729 | 350/75/75 | no | fp32 | 0.00 |
| 4 | -1.8775 | -2.5469 | -1.5449 | 350/75/75 | no | fp32 | 0.01 |

### a2780cis (ic50_a2780cis_nm)

| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 | wall_s |
|---:|---:|---:|---:|---|:---:|---|---:|
| 0 | -0.8487 | -1.0083 | -0.5992 | 350/75/75 | no | fp32 | 0.00 |
| 1 | -0.8811 | -0.6108 | -0.6188 | 350/75/75 | no | fp32 | 0.00 |
| 2 | -0.5707 | -0.6812 | -0.5909 | 350/75/75 | no | fp32 | 0.00 |
| 3 | -1.0321 | -0.9371 | -0.5273 | 350/75/75 | no | fp32 | 0.00 |
| 4 | -0.7174 | -0.5577 | -0.5998 | 350/75/75 | no | fp32 | 0.00 |

## Honest framing

- **Real Hilbert regression on R20-P3 features** = Multi-PH 32-bin Fourier + 128-d PLL.
- **Per `party-twenty-spirit §3.5`**: MEASURED R² reported honestly; no PROJECTED values.
- **CPU-only evaluation**; fp64 fallback on numerical failure.
- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged per seed.
- **Sample limit**: `--n-samples 500` default for wall-time budget; actual cohort used: 500 molecules present in all 4 cell-lines.

**Aggregate Hilbert test R² (mean of 4 cell-line means) = -1.2335; joint R² (across 4 cell-lines, seed=0 test split) = -0.8067; gate status = PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL).**
