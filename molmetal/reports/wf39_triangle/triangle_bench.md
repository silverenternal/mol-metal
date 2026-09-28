# Wf-39 L4 triangle closure end-to-end benchmark

n_seeds = 5, n_trajectories = 100, n_rounds = 5, n_arms = 4, hilbert_iters = 20

PlatinAI rows loaded: 5000

## Per-stage wall time (mean ± std, s)

| stage | mean wall s | std wall s |
|-------|------------:|-----------:|
| ASTSR | 0.0350 | 0.0014 |
| Hilbert | 0.0012 | 0.0001 |
| MCTS | 0.0011 | 0.0001 |

## Per-stage success rate (across seeds)

| stage | success rate |
|-------|-------------:|
| ASTSR | 1.0000 |
| Hilbert | 1.0000 |
| MCTS | 1.0000 |

## Regression metrics (mean ± std)

| metric | mean | std |
|--------|-----:|----:|
| Hilbert R² | 0.5397 | 0.0676 |
| evolutionary R² | 0.0198 | 0.0413 |
| variance reduction | 0.0020 | 0.0046 |

## Per-seed detail

| seed | total wall | hilbert R² | evol R² | var red |
|-----:|-----------:|-----------:|--------:|--------:|
| 0 | 0.200 | 0.5964 | -0.0000 | 0.0058 |
| 1 | 0.180 | 0.6249 | 0.0994 | 0.0039 |
| 2 | 0.190 | 0.5510 | 0.0174 | 0.0050 |
| 3 | 0.178 | 0.4782 | -0.0178 | 0.0025 |
| 4 | 0.190 | 0.4479 | -0.0000 | -0.0069 |
