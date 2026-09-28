# Wf-127 Phase 1 — Hilbert+ASTSR parameter-sweep diagnostic

**Date**: 2026-09-27
**Constraint**: CPU-only diagnostic; honest framing per `party-twenty-spirit §3.5`.
**Why this exists**: Wf-83 Phase 1 reported `combined R² = -6.71` on PlatinAI 4-cell (real pIC50) below the Wf-77 Phase 3 anchor `+0.30` on synthetic. Wf-127 P1 sweeps the (L2 weight, ASTSR loss_scale, init mode) axes to identify the **stable region** on synthetic — i.e. the parameter setting where the Hilbert primal-dual + ASTSR combination does not collapse.

## 1 Sweep axes

| axis | values |
|---|---|
| `l2_lambda` | [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0] |
| `astr_loss_scale` | [0.1, 1.0, 10.0, 100.0] |
| `init_mode` | ['zeros', 'random_normal', 'xavier'] |

Stable-region gate: held-out `R² ≥ 0.0` on synthetic (snr=1.0, n=800, dim=12) for ALL seeds.

## 2 L2 weight sweep

| `l2_lambda` | test R² (mean ± std) | stable? |
|---:|---:|:---:|
| 0e+00 | +0.4888 ± 0.0196 | YES |
| 1e-04 | +0.4888 ± 0.0196 | YES |
| 1e-03 | +0.4888 ± 0.0196 | YES |
| 1e-02 | +0.4887 ± 0.0199 | YES |
| 1e-01 | +0.4860 ± 0.0221 | YES |
| 1e+00 | +0.3832 ± 0.0237 | YES |

## 3 ASTSR loss_scale sweep

| `loss_scale` | test R² (mean ± std) | stable? |
|---:|---:|:---:|
| 0.1 | +0.3309 ± 0.0222 | YES |
| 1 | +0.3309 ± 0.0222 | YES |
| 10 | -4.5768 ± 0.7286 | no |
| 100 | -828.0711 ± 100.0330 | no |

## 4 Init mode sweep

| `init_mode` | test R² (mean ± std) | stable? |
|---:|---:|:---:|
| `zeros` | +0.4888 ± 0.0203 | YES |
| `random_normal` | +0.4887 ± 0.0199 | YES |
| `xavier` | +0.4716 ± 0.0122 | YES |

## 5 Stable region (synthetic R² ≥ 0)

- L2 stable: [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
- ASTSR loss_scale stable: [0.1, 1.0]
- Init mode stable: ['zeros', 'random_normal', 'xavier']

## 7 Honest framing (party-twenty-spirit §3.5)

- **MEASURED today**: all sweep cells are MEASURED on synthetic data (n=800, dim=12, snr=1.0).
- **Stable region**: defined as `R² ≥ 0` on synthetic; this does NOT guarantee PlatinAI 4-cell lift.
- **No fabricated lift**: the diagnostic reports PlatinAI R² only when `--cell-line` is passed; default run is synthetic-only.
- **PROJECTED follow-ons**: Wf-122 Phase 2 (adaptive L2 with finer doubling) + Wf-127 Phase 2 (joint sweep + per-fold validation) + Wf-56 R20-P2 (GPU retrain on Blackwell sm_120).
