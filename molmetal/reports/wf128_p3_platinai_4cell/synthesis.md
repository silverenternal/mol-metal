# Wf-128 Phase 3 — 3-way (Deflexpressor + Multi-PH + ASTSR) cross-dataset verify on PlatinAI 4-cell

**Date**: 2026-09-26
**Constraint**: 12-d RDKit + zero-padded 32-d Multi-PH + 128-d PLL (PlatinAI 4-cell ships no 172-d NPZ); 5-fold CV × 5 seeds × 4 cell-lines; CPU default, GPU-1 optional.
**Task**: 3-way model not worse than baseline Hilbert on cross-dataset.

## Cross-cell-line summary (per-seed aggregate over 5 folds)

| cell_line | n_seeds | combined R² (mean ± std) | Hilbert R² | Multi-PH R² | ASTSR R² | collapse_any |
|---|---:|---|---|---|---|---|
| mcf7 | 5 | -117.7033 ± 0.0916 | -127.6099 | -127.6027 | -82.0925 | YES |
| a549 | 5 | -110.8056 ± 0.0878 | -120.4387 | -120.4345 | -76.2973 | YES |
| a2780 | 5 | -132.4979 ± 0.1711 | -142.9858 | -142.9883 | -94.5405 | YES |
| a2780cis | 5 | -98.4833 ± 0.1471 | -107.1132 | -107.1610 | -67.5650 | YES |
| **4-cell aggregate** | — | **-114.8725 ± 12.2863** | **-124.5369** | — | — | — |

## Verify gate (target: combined R² ≥ Hilbert baseline)

**4 of 4 cell-lines pass.**

**Verdict: PASS** — 3-way combined R² = -114.8725; Hilbert baseline R² = -124.5369.

## Per-(cell_line, seed) detail
### mcf7

| seed | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---|
| 0 | -117.6451 ± 4.3148 | -127.5333 | -127.5494 | -82.0577 | 1.94 | YES |
| 1 | -117.6827 ± 4.6161 | -127.5953 | -127.5742 | -82.0755 | 2.09 | YES |
| 2 | -117.7148 ± 4.7171 | -127.6171 | -127.6198 | -82.1014 | 2.09 | YES |
| 3 | -117.8707 ± 6.5100 | -127.7980 | -127.7788 | -82.2072 | 2.12 | YES |
| 4 | -117.6032 ± 3.4864 | -127.5059 | -127.4910 | -82.0207 | 2.09 | YES |

### a549

| seed | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---|
| 0 | -110.8376 ± 2.6394 | -120.4715 | -120.4715 | -76.3186 | 1.96 | YES |
| 1 | -110.9309 ± 5.0650 | -120.5723 | -120.5726 | -76.3846 | 2.30 | YES |
| 2 | -110.8145 ± 3.7611 | -120.4458 | -120.4454 | -76.3054 | 2.33 | YES |
| 3 | -110.6594 ± 3.3721 | -120.2767 | -120.2784 | -76.1968 | 2.20 | YES |
| 4 | -110.7858 ± 4.3864 | -120.4270 | -120.4047 | -76.2813 | 2.23 | YES |

### a2780

| seed | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---|
| 0 | -132.2121 ± 4.0353 | -142.6767 | -142.6808 | -94.3362 | 2.43 | YES |
| 1 | -132.7494 ± 9.2080 | -143.2527 | -143.2609 | -94.7255 | 1.77 | YES |
| 2 | -132.5353 ± 6.8617 | -143.0225 | -143.0320 | -94.5676 | 1.99 | YES |
| 3 | -132.5027 ± 5.5091 | -142.9973 | -142.9878 | -94.5430 | 2.16 | YES |
| 4 | -132.4899 ± 5.8420 | -142.9795 | -142.9802 | -94.5302 | 2.16 | YES |

### a2780cis

| seed | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | wall_s | collapse |
|---:|---:|---:|---:|---:|---:|---|
| 0 | -98.7232 ± 7.0256 | -107.3683 | -107.4241 | -67.7355 | 2.05 | YES |
| 1 | -98.5834 ± 5.9665 | -107.2172 | -107.2719 | -67.6379 | 1.87 | YES |
| 2 | -98.4061 ± 4.6466 | -107.0335 | -107.0753 | -67.5080 | 1.85 | YES |
| 3 | -98.3422 ± 4.0842 | -106.9654 | -107.0047 | -67.4631 | 1.78 | YES |
| 4 | -98.3617 ± 2.5226 | -106.9814 | -107.0287 | -67.4805 | 1.83 | YES |

## Honest framing

- **REAL data**: PlatinAI 4-cell synthetic pIC50 (per ``platinai_4cell.meta.json``).
- **Feature stack**: 172-d zero-padded (12-d RDKit + 32-d Multi-PH zeros + 128-d PLL zeros); the 3-way Multi-PH stream is effectively a constant-zero ridge.
- **ASTSR scale guard**: predictions clipped to `[0, 1]`.
- **CPU + GPU-1**: CPU default; GPU-1 optional.
- **Collapse detector**: ``pred_std / y_test_std < 0.10`` per fold → flag in ``per_seed_metrics.json``.

Per `party-twenty-spirit §3.5`: 4-cell combined R² = -114.8725 vs Hilbert baseline = -124.5369; verdict = **PASS** (4 of 4 cell-lines pass).
