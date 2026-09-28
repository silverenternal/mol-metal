# Wf-37 Phase 3 — Multi-PH + PLL real end-to-end feature extraction

Fixes the three Wf-25 numerical instabilities:

* **Real pIC50 target** (NCI-60 / PlatinAI 4-cell) — not noisy binary
* **Bias-column padding** (not tiling) for the 8-d descriptor +
  flattened multi-PH join
* **Cosine LR + gradient clipping + L2 ridge** in the Hilbert head

## Per-seed R² on real pIC50

| Seed | n_target | r2_mean ± std | r2_min | r2_max | wall_s/fold |
|---|---|---|---|---|---|
| 0 | 200 | +0.0348 ± 0.0333 | -0.0057 | +0.0912 | 0.0012 |
| 1 | 200 | +0.1515 ± 0.0911 | +0.0323 | +0.2900 | 0.0013 |
| 2 | 200 | +0.0747 ± 0.0810 | -0.0263 | +0.2066 | 0.0031 |

**Aggregate over 3 seeds**:
* Mean R² = **+0.0870**
* Min R² (worst fold, worst seed) = **-0.0263**

**Wf-25 baseline (broken)**: joint R² = -0.443 / -9e9 (numerical bug).
**Wf-37 fix target**: per-seed R² ≥ 0.3 on real spread pIC50.
