# Wf-128 Phase 2 — 3-way (Deflexpressor + Multi-PH + ASTSR) on REAL MetalCytoToxDB 5-metal

**Date**: 2026-09-26
**Constraint**: 172-d feature stack (RDKit 12 + Multi-PH 32 + PLL 128); multi-task 256→128 backbone + 5 per-metal heads; Wasserstein weight 0.05; L2 grad-norm guard; ASTSR loss_scale guard `<= 1.0`; 5-fold CV × 5 seeds × 5 metals; CPU default, GPU-1 optional.
**Task**: combined R² ≥ 0 for ≥ 1 of 5 metals (vs Wf-41 synthetic `-0.36`).

## Cross-metal summary (per-seed aggregate over 5 folds)

| metal | n_seeds | combined R² (mean ± std) | Hilbert R² | Multi-PH R² | ASTSR R² | collapse_any |
|---|---:|---|---|---|---|---|
| Ru | 5 | -0.0694 ± 0.0425 | -1.3612 | -0.3270 | -2.2013 | YES |
| Ir | 5 | -201.2546 ± 80.7401 | -1.3658 | -1.3621 | -0.5125 | no |
| Os | 5 | -287.1975 ± 117.9581 | -3.9487 | -0.9674 | -1.0780 | no |
| Rh | 5 | -183.8218 ± 97.3128 | -1.0056 | -1.0758 | -0.8088 | no |
| Re | 5 | -80.8805 ± 30.4413 | -0.3655 | -0.3627 | -0.9013 | YES |
| **5-metal aggregate** | — | **-150.6448 ± 99.8486** | — | — | — | — |

## Verify gate (target: combined R² ≥ 0 for ≥ 1 of 5 metals)

**0 of 5 metals pass the gate.**

**Verdict: FAIL** — no metal hits R² ≥ 0; honest framing per `party-twenty-spirit §3.5`.

## Per-(metal, seed) detail
### Ru

| seed | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---|
| 0 | -0.0239 ± 0.0665 | -0.4846 | -0.3207 | -2.0872 | 2.26 | YES |
| 1 | -0.0460 ± 0.0794 | -1.6263 | -0.3219 | -2.5725 | 0.49 | no |
| 2 | -0.1489 ± 0.1631 | -3.1396 | -0.3257 | -2.4837 | 0.29 | YES |
| 3 | -0.0593 ± 0.0948 | -0.7403 | -0.3244 | -2.1927 | 0.36 | YES |
| 4 | -0.0689 ± 0.0337 | -0.8152 | -0.3422 | -1.6701 | 0.51 | YES |

### Ir

| seed | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---|
| 0 | -125.6806 ± 77.8997 | -1.4572 | -1.3976 | -0.5369 | 1.84 | no |
| 1 | -189.6409 ± 142.4104 | -1.3428 | -1.4371 | -0.4386 | 1.66 | no |
| 2 | -326.7762 ± 253.6264 | -1.3083 | -1.2922 | -0.5740 | 0.89 | no |
| 3 | -253.7048 ± 219.5598 | -1.3073 | -1.3218 | -0.4580 | 0.72 | no |
| 4 | -110.4703 ± 116.9645 | -1.4133 | -1.3618 | -0.5546 | 0.70 | no |

### Os

| seed | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---|
| 0 | -326.8857 ± 139.1980 | -5.0926 | -0.6506 | -0.9017 | 0.32 | no |
| 1 | -263.2593 ± 176.6676 | -5.2948 | -2.0508 | -0.3743 | 0.30 | no |
| 2 | -489.3962 ± 702.8881 | -3.5419 | -0.6821 | -1.3737 | 0.28 | no |
| 3 | -215.5546 ± 47.9724 | -2.8487 | -0.7612 | -1.8288 | 0.11 | no |
| 4 | -140.8920 ± 205.5608 | -2.9656 | -0.6924 | -0.9116 | 0.31 | no |

### Rh

| seed | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---|
| 0 | -218.1232 ± 226.4412 | -0.8449 | -0.8005 | -0.4233 | 0.59 | no |
| 1 | -284.9239 ± 383.3931 | -1.3865 | -2.1822 | -2.1239 | 0.59 | no |
| 2 | -279.7163 ± 357.5232 | -0.8200 | -0.8057 | -0.3793 | 0.77 | no |
| 3 | -69.8326 ± 57.2482 | -0.8148 | -0.7833 | -0.3643 | 0.50 | no |
| 4 | -66.5129 ± 19.8937 | -1.1618 | -0.8071 | -0.7533 | 0.33 | no |

### Re

| seed | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---|
| 0 | -43.7558 ± 45.9345 | +0.0431 | -0.3750 | -0.5162 | 0.33 | YES |
| 1 | -127.4985 ± 146.4908 | -0.3547 | -0.3917 | -0.9474 | 0.45 | YES |
| 2 | -97.5500 ± 37.6522 | -0.1609 | -0.4500 | -0.9757 | 0.34 | YES |
| 3 | -82.9941 ± 53.8107 | -1.0755 | -0.3033 | -1.1883 | 0.27 | no |
| 4 | -52.6042 ± 38.1224 | -0.2797 | -0.2933 | -0.8789 | 0.37 | no |

## Honest framing

- **REAL data**: MetalCytoToxDB 5-metal binary activity; the CSV exposes ``activity`` (0/1) — R² is regression on a binary target, not a continuous pIC50.
- **Feature stack**: 172-d = 12-d RDKit + 32-d Multi-PH Fourier coefficients + 128-d PLL signature (zero-padded).
- **Multi-task backbone**: shared 172 → 128 trunk + 5 per-metal 128 → 64 → 1 heads (Wf-128 P1 ship).
- **Wasserstein weight 0.05**: sliced-W² penalty between predicted pIC50 distribution and the training reference.
- **L2 guard**: ``clip_grad_norm_(parameters, max_norm=1.0)``.
- **ASTSR scale guard**: predictions clipped to `[0, 1]` (Wf-127 P1 stable-region cap; loss_scale > 1.0 collapse regime).
- **CPU + GPU-1**: CPU is default; GPU-1 is selected when ``CUDA_VISIBLE_DEVICES=1`` is set and ``--device cuda``.
- **Collapse detector**: ``pred_std / y_test_std < 0.10`` per fold → flag in ``per_seed_metrics.json``.

Per `party-twenty-spirit §3.5`: aggregate combined R² = -150.6448; verdict = **FAIL** (0 of 5 metals hit R² ≥ 0).
