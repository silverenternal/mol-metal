# Wf-25 Phase 4 — 5-seed × 3-method ASTSR benchmark (PlatinAI 5K)

**Dataset:** PlatinAI 5K + MetalCytoToxDB Ru subset (n=500)
**Seeds:** [0, 1, 2, 3, 4]  |  **Folds per seed:** 5  |  **Join dim:** 264 (astr=8 + mph=256)
**Multi-PH bins:** 16  |  **Hilbert iters:** 200  |  **Hilbert lr:** 0.05
**ASTSR depth:** 4  |  **population:** 32  |  **generations:** 20

**Parse stats:** parsed=500, failed=0, target_rows=500, mph_parsed=500, mph_failed=0

## Aggregate (5-seed × 5-fold = 25 records per method)

| Method | R² mean ± std | Wall (s) mean ± std | Peak mem (MB) | Speedup |
|---|---|---|---|---|
| astsr | -0.0990 ± 0.0882 | 3.620 ± 0.563 | 0.91 ± 1.47 | 134.16x |
| hilbert | -0.4427 ± 0.1070 | 0.027 ± 0.000 | 0.08 ± 0.03 | 1.00x (ref) |
| joint | -9006210749.3297 ± 18445777365.2588 | 0.042 ± 0.000 | 5.09 ± 0.02 | 1.54x |

## Per-seed R² (mean across folds)

| Method | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| astsr | -0.0421 | -0.1257 | -0.1585 | -0.0424 | -0.1262 |
| hilbert | -0.4554 | -0.4333 | -0.4474 | -0.4341 | -0.4434 |
| joint | -10123063209.6739 | -8549158642.1335 | -9412192943.1962 | -8457787546.7317 | -8488851404.9135 |

## Per-seed wall time (mean across folds)

| Method | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| astsr | 3.612s | 4.481s | 3.689s | 3.082s | 3.238s |
| hilbert | 0.027s | 0.027s | 0.027s | 0.027s | 0.027s |
| joint | 0.042s | 0.042s | 0.042s | 0.041s | 0.041s |

## Honest framing

* ASTSR features are 8 RDKit descriptors — the F5 schema is
  unavailable offline (requires a Pareto-front computation).
* The three backends operate on different feature sets:
  ASTSR + Hilbert share the 8-d RDKit stream; Joint adds a 256-d
  multi-PH stream (16×16 bins flattened) concatenated to 264-d
  before fitting.
* Wall-time and memory are machine-dependent; the speedup column
  is the intended *relative* metric against the fastest backend
  (hilbert).
* `seed` controls the k-fold partition only; the per-backend
  internal RNG is reset independently for each (seed, fold).
* Activity is binary (0/1); the target is `activity + N(0, 0.05)`
  — R² measures how much of that binary signal each backend
  recovers from descriptors alone.

**Run timestamp (UTC):** 2026-09-25T21:04:27.453514+00:00

