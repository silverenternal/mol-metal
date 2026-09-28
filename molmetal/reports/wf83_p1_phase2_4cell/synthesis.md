# Wf-83 Phase 1 — Full Hilbert+ASTSR combined R² on PlatinAI 4-cell

**Date**: 2026-09-26
**Constraint**: Combined Hilbert primal-dual + ASTSR evolutionary symbolic regressor
  (no Multi-PH + PLL); per `party-twenty-spirit §3.5` honest framing.
**Hardware**: CPU fp32 (default); `--device cuda` accepted but no Blackwell acceleration
  required for Wf-83 P1 production anchor.
**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines).

## Cross-cell-line summary (10-seed × 5-fold CV)

| cell line | n_samples | combined R² (mean ± std) | Hilbert-only R² | ASTSR-only R² |
|---|---:|---|---|---|
| mcf7 | 4078 | -5.8613 ± 5.9979 | -25.7868 ± 0.2509 | -6.1100 ± 6.2624 |
| a549 | 4078 | -6.1722 ± 6.6239 | -23.0323 ± 0.2317 | -7.0742 ± 8.5087 |
| a2780 | 4078 | -9.2445 ± 7.1959 | -31.7342 ± 0.2569 | -9.3351 ± 7.5702 |
| a2780cis | 4078 | -5.5725 ± 5.4746 | -20.1063 ± 0.1986 | -6.3356 ± 7.0541 |
| **4-cell aggregate** | — | **-6.7126 ± 1.4771** | — | — |

## Verify gate (vs Wf-77 Phase 3 anchor ≥ +0.30)

**Aggregate combined R² (mean of 4 cell-line means) = -6.7126** — ❌ **FAIL** (below both anchor + stretch)

## Per-cell-line per-seed R² detail

### mcf7 (ic50_mcf7_nm)

| seed | combined R² (overall) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0138 | -25.7368 | -0.0138 | 0.000 | 11.33 |
| 1 | -6.6488 | -25.7222 | -8.1642 | 0.100 | 10.24 |
| 2 | -5.5278 | -25.4237 | -5.5278 | 0.000 | 10.36 |
| 3 | -3.2098 | -25.2912 | -3.2098 | 0.000 | 7.56 |
| 4 | -1.5237 | -26.0496 | -1.5237 | 0.000 | 8.43 |
| 5 | -20.0554 | -25.6462 | -21.0190 | 0.250 | 9.07 |
| 6 | -0.0138 | -25.4771 | -0.0138 | 0.000 | 8.12 |
| 7 | -1.5237 | -25.7896 | -1.5237 | 0.000 | 8.28 |
| 8 | -12.0262 | -26.0745 | -12.0262 | 0.000 | 8.12 |
| 9 | -7.8758 | -25.7772 | -7.8758 | 0.000 | 9.97 |

### a549 (ic50_a549_nm)

| seed | combined R² (overall) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0496 | -22.9773 | -0.0496 | 0.000 | 8.30 |
| 1 | -18.4704 | -22.9959 | -26.8263 | 0.550 | 10.81 |
| 2 | -4.0547 | -22.7197 | -4.0547 | 0.000 | 10.12 |
| 3 | -3.4662 | -22.6130 | -3.4415 | 0.050 | 8.93 |
| 4 | -0.8019 | -23.2703 | -0.8019 | 0.000 | 9.94 |
| 5 | -17.4047 | -22.9234 | -18.0618 | 0.250 | 11.05 |
| 6 | -0.0496 | -22.7321 | -0.0496 | 0.000 | 9.71 |
| 7 | -0.8019 | -23.0452 | -0.8019 | 0.000 | 9.05 |
| 8 | -9.8081 | -23.3357 | -9.8081 | 0.000 | 8.79 |
| 9 | -6.5966 | -23.0096 | -6.5966 | 0.000 | 9.80 |

### a2780 (ic50_a2780_nm)

| seed | combined R² (overall) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.6137 | -31.6582 | -0.6137 | 0.000 | 10.04 |
| 1 | -8.2563 | -31.7105 | -7.6319 | 0.050 | 7.98 |
| 2 | -9.0863 | -31.3166 | -9.0863 | 0.000 | 8.53 |
| 3 | -3.6225 | -31.1964 | -3.6225 | 0.000 | 8.00 |
| 4 | -3.6057 | -32.0464 | -3.6057 | 0.000 | 7.97 |
| 5 | -25.9817 | -31.6387 | -27.5136 | 0.400 | 8.96 |
| 6 | -9.0863 | -31.3927 | -9.0863 | 0.000 | 11.20 |
| 7 | -3.6057 | -31.6961 | -3.6057 | 0.000 | 8.42 |
| 8 | -17.0556 | -32.0567 | -17.0556 | 0.000 | 8.33 |
| 9 | -11.2491 | -31.7044 | -11.2491 | 0.000 | 8.12 |

### a2780cis (ic50_a2780cis_nm)

| seed | combined R² (overall) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0953 | -20.0750 | -0.0953 | 0.000 | 7.91 |
| 1 | -15.5693 | -20.0649 | -22.6457 | 0.550 | 8.31 |
| 2 | -3.3116 | -19.8029 | -3.3116 | 0.000 | 8.46 |
| 3 | -3.3116 | -19.7275 | -3.3116 | 0.000 | 7.74 |
| 4 | -3.3116 | -20.3414 | -3.3116 | 0.000 | 9.14 |
| 5 | -15.0459 | -20.0360 | -15.5892 | 0.250 | 10.72 |
| 6 | -0.0953 | -19.8068 | -0.0953 | 0.000 | 8.07 |
| 7 | -0.5708 | -20.1034 | -0.5708 | 0.000 | 8.19 |
| 8 | -8.3177 | -20.3301 | -8.3177 | 0.000 | 8.03 |
| 9 | -5.9046 | -20.0595 | -5.9046 | 0.000 | 8.59 |

## Honest framing

- **MEASURED today**: 10 seeds × 5-fold × 4 cell lines on PlatinAI 4-cell
  real pIC50 data; Hilbert+ASTSR convex blend with per-fold alpha selection.
- **Alpha selection**: per-fold 90/10 inner validation; grid over
  alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}; reported as `alpha_chosen` per fold.
- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged in `per_fold`.
- **No Multi-PH + PLL**: this is the **combined Hilbert+ASTSR** anchor;
  Multi-PH/PLL augmentation is a separate workstream (Wf-79 P2 3-way + PLL).
- **CPU smoke**: full 10-seed × 5-fold × 4-cell runs in < 6 minutes on a
  single CPU core (verified smoke); GPU-0 Blackwell optional but not required.

Per `party-twenty-spirit §3.5`: aggregate combined R² = -6.7126; this is the **MEASURED** Wf-83 Phase 1 production anchor.
