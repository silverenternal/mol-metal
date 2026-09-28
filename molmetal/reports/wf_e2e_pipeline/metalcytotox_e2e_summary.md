# R18 end-to-end pipeline benchmark on MetalCytoToxDB

**Date**: 2026-09-25T17:37:50Z  
**Dataset**: `metalcytotox`  
**Pipeline**: 5-step R18 orchestrator (32 ship-modules chained)  
**N samples per metal** (target): 500  
**Metals**: Ru, Ir, Rh, Re, Os  

## Per-metal summary

| Metal | N samples | Wall-time (s) | Success rate | Mean sample wall (s) |
|-------|-----------|---------------|---------------|---------------------|
| Ru | 500 | 38.01 | 100.00% | 0.0760 |
| Ir | 500 | 43.77 | 100.00% | 0.0875 |
| Rh | 500 | 36.78 | 99.80% | 0.0736 |
| Re | 500 | 19.99 | 100.00% | 0.0400 |
| Os | 500 | 42.42 | 97.60% | 0.0848 |

**Total wall-time**: 181.21 s across all 5 metals (2500 samples).

**Peak memory delta**: 0.00 MB

## Per-step × per-metal total wall-time (seconds)

| Step | Ru | Ir | Rh | Re | Os | **Sum** |
|---|---|---|---|---|---|---|
| step1_closure (MLC closure BFS) | 0.0019 | 0.0010 | 0.0011 | 0.0010 | 0.0011 | **0.0061** |
| step2_se3 (SE(3) SB + Lipman ODE) | 14.8903 | 15.8696 | 16.0214 | 8.0147 | 15.3067 | **70.1028** |
| step3_multiph (Multi-PH + PLL) | 20.0170 | 24.6632 | 17.6215 | 10.0346 | 24.0747 | **96.4110** |
| step4_mcts (Variance-controlled PUCT) | 0.0247 | 0.0295 | 0.0259 | 0.0211 | 0.0262 | **0.1275** |
| step5_hilbert (Hilbert-ASTSR reward) | 2.9150 | 3.0390 | 3.0278 | 1.8535 | 2.9172 | **13.7526** |

## Per-step × per-metal mean wall-time per sample (seconds)

| Step | Ru | Ir | Rh | Re | Os | **Sum** |
|---|---|---|---|---|---|---|
| step1_closure | 0.000004 | 0.000002 | 0.000002 | 0.000002 | 0.000002 |  |
| step2_se3 | 0.029781 | 0.031739 | 0.032043 | 0.016029 | 0.030613 |  |
| step3_multiph | 0.040034 | 0.049326 | 0.035243 | 0.020069 | 0.048149 |  |
| step4_mcts | 0.000049 | 0.000059 | 0.000052 | 0.000042 | 0.000052 |  |
| step5_hilbert | 0.005830 | 0.006078 | 0.006056 | 0.003707 | 0.005834 |  |

## Per-step × per-metal status counts

### step1_closure

| Metal | ok | skipped | error | total |
|-------|----|---------|-------|-------|
| Ru | 500 | 0 | 0 | 500 |
| Ir | 500 | 0 | 0 | 500 |
| Rh | 500 | 0 | 0 | 500 |
| Re | 500 | 0 | 0 | 500 |
| Os | 500 | 0 | 0 | 500 |

### step2_se3

| Metal | ok | skipped | error | total |
|-------|----|---------|-------|-------|
| Ru | 500 | 0 | 0 | 500 |
| Ir | 500 | 0 | 0 | 500 |
| Rh | 500 | 0 | 0 | 500 |
| Re | 500 | 0 | 0 | 500 |
| Os | 500 | 0 | 0 | 500 |

### step3_multiph

| Metal | ok | skipped | error | total |
|-------|----|---------|-------|-------|
| Ru | 500 | 0 | 0 | 500 |
| Ir | 500 | 0 | 0 | 500 |
| Rh | 499 | 0 | 1 | 500 |
| Re | 500 | 0 | 0 | 500 |
| Os | 488 | 0 | 12 | 500 |

### step4_mcts

| Metal | ok | skipped | error | total |
|-------|----|---------|-------|-------|
| Ru | 500 | 0 | 0 | 500 |
| Ir | 500 | 0 | 0 | 500 |
| Rh | 500 | 0 | 0 | 500 |
| Re | 500 | 0 | 0 | 500 |
| Os | 500 | 0 | 0 | 500 |

### step5_hilbert

| Metal | ok | skipped | error | total |
|-------|----|---------|-------|-------|
| Ru | 500 | 0 | 0 | 500 |
| Ir | 500 | 0 | 0 | 500 |
| Rh | 500 | 0 | 0 | 500 |
| Re | 500 | 0 | 0 | 500 |
| Os | 500 | 0 | 0 | 500 |

## How to re-run

```bash
source env.sh && uv run --no-sync python molmetal/scripts/benchmark_metalcytotox_e2e.py --n-samples-per-metal 500
```
