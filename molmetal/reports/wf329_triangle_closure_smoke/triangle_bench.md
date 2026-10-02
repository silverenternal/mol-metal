# Wf-39 L4 triangle closure end-to-end benchmark

n_seeds = 2, n_trajectories = 10, n_rounds = 2, n_arms = 4, hilbert_iters = 20

PlatinAI rows loaded: 5000

## Per-stage wall time (mean ± std, s)

| stage | mean wall s | std wall s |
|-------|------------:|-----------:|
| ASTSR | 0.0569 | 0.0041 |
| Hilbert | 0.0101 | 0.0079 |
| MCTS | 0.0006 | 0.0004 |

## Per-stage success rate (across seeds)

| stage | success rate |
|-------|-------------:|
| ASTSR | 1.0000 |
| Hilbert | 1.0000 |
| MCTS | 1.0000 |

## Regression metrics (mean ± std)

| metric | mean | std |
|--------|-----:|----:|
| Hilbert R² | 0.6165 | 0.0030 |
| evolutionary R² | -5.7023 | 5.9295 |
| variance reduction | -0.0165 | 0.0075 |

## Per-seed detail

| seed | total wall | hilbert R² | evol R² | var red |
|-----:|-----------:|-----------:|--------:|--------:|
| 0 | 0.161 | 0.6195 | -11.6318 | -0.0239 |
| 1 | 0.111 | 0.6136 | 0.2272 | -0.0090 |
