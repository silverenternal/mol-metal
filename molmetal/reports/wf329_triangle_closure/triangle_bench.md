# Wf-39 L4 triangle closure end-to-end benchmark

n_seeds = 5, n_trajectories = 100, n_rounds = 5, n_arms = 4, hilbert_iters = 20

PlatinAI rows loaded: 5000

## Per-stage wall time (mean ± std, s)

| stage | mean wall s | std wall s |
|-------|------------:|-----------:|
| ASTSR | 0.0413 | 0.0177 |
| Hilbert | 0.0031 | 0.0010 |
| MCTS | 0.0015 | 0.0004 |

## Per-stage success rate (across seeds)

| stage | success rate |
|-------|-------------:|
| ASTSR | 1.0000 |
| Hilbert | 1.0000 |
| MCTS | 1.0000 |

## Regression metrics (mean ± std)

| metric | mean | std |
|--------|-----:|----:|
| Hilbert R² | 0.4885 | 0.0594 |
| evolutionary R² | -220.8107 | 376.0193 |
| variance reduction | 0.0020 | 0.0046 |

## Per-seed detail

| seed | total wall | hilbert R² | evol R² | var red |
|-----:|-----------:|-----------:|--------:|--------:|
| 0 | 0.220 | 0.5436 | -2.7341 | 0.0058 |
| 1 | 0.219 | 0.5096 | 0.1738 | 0.0039 |
| 2 | 0.375 | 0.5439 | -9.1524 | 0.0050 |
| 3 | 0.270 | 0.4575 | -125.4308 | 0.0025 |
| 4 | 0.078 | 0.3878 | -966.9102 | -0.0069 |
