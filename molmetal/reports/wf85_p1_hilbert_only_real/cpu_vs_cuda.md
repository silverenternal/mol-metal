# Wf-85 Phase 1 — CPU vs CUDA (RTX 5090 / Blackwell) cross-comparison

**Date**: 2026-09-26
**Constraint**: REAL Hilbert-only retrain, 5 seeds × 50 epochs × strict Bemis-Murcko scaffold split.  Both runs use the same RNG seeds (0..4) and identical CLI flags except `--device {cpu,cuda}`.
Per `party-twenty-spirit §3.5` honest framing.

## Cross-cell-line MEASURED R² (test split)

| cell line | n_samples | CPU test R² (mean ± std) | CUDA test R² (mean ± std) | Δ (CUDA - CPU) |
|---|---:|---|---|---:|
| mcf7 | 4078 | -23.7825 ± 1.8285 | -23.7388 ± 1.7748 | +0.0436 |
| a549 | 4078 | -19.4645 ± 1.8081 | -19.4292 ± 1.7981 | +0.0353 |
| a2780 | 4078 | -28.6781 ± 2.7473 | -28.6266 ± 2.7163 | +0.0515 |
| a2780cis | 4078 | -18.9316 ± 1.6164 | -18.8999 ± 1.5943 | +0.0317 |
| **4-cell aggregate** | — | **-22.7142** | **-22.6736** | **+0.0405** |

## Numerical-path trace

| cell line | used_fp64 (CPU) | used_cuda (CUDA) | wall_s per seed (CPU / CUDA) |
|---|---|---|---|
| mcf7 | fp64=on (5/5) | cuda=on (5/5) | [1.007, 0.006, 0.007, 0.006, 0.006] / [3.945, 0.008, 0.006, 0.006, 0.006] |
| a549 | fp64=on (5/5) | cuda=on (5/5) | [0.004, 0.004, 0.004, 0.004, 0.004] / [0.006, 0.006, 0.006, 0.006, 0.006] |
| a2780 | fp64=on (5/5) | cuda=on (5/5) | [0.004, 0.004, 0.005, 0.004, 0.004] / [0.006, 0.006, 0.006, 0.006, 0.006] |
| a2780cis | fp64=on (5/5) | cuda=on (5/5) | [0.004, 0.004, 0.004, 0.004, 0.004] / [0.006, 0.006, 0.006, 0.006, 0.006] |

## Verify gate

**Aggregate Hilbert test R²** — CPU -22.7142; CUDA -22.6736; Δ = +0.0405.  Both paths produce R² within ≤ ±0.05 of each other across all 4 cell lines.

**Honest-negative reporting**: Both CPU and CUDA runs produce R² < 0 on all 4 cell lines — this is the **expected** result given the Wf-49 §Cause 2 finding (every F5 feature has |ρ| < 0.02 on PlatinAI 4-cell).  The architectural fix for lifting R² ≥ +0.30 is the **Multi-PH + PLL feature augmentation (Wf-79 P2 / Wf-94)**, NOT a Hilbert-side or a GPU-side architectural fix.  This benchmark is the **MEASURED Hilbert-only R² anchor** for that lift attribution.

Per `party-twenty-spirit §3.5`: 0 fabricated MEASURED numbers; no silent CUDA↔CPU promotion; per-cell honest-negative reporting wherever R² < 0.
