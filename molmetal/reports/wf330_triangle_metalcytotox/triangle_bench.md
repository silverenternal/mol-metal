# Wf-330-T2 triangle closure on MetalCytoToxDB 5-metal

dataset = `metalcytotox_5metal`, feature_mode = `f5_rdkit_2d`, n_seeds = 5, n_trajectories = 100, n_rounds = 5, n_arms = 4, hilbert_iters = 20

Rows loaded: 26632

## MEASURED vs PROJECTED (Wf-330-T2 ship ledger)

- PROJECTED cohort mean `mean_hilbert_r2` = 0.6220
- MEASURED cohort mean `mean_hilbert_r2` = 0.4156
- Verdict = `MEASURED=BELOW_TARGET (PROJECTED=0.622)`

## Per-stage wall time (mean ± std, s)

| stage | mean wall s | std wall s |
|-------|------------:|-----------:|
| ASTSR | 0.0319 | 0.0060 |
| Hilbert | 0.0018 | 0.0002 |
| MCTS | 0.0010 | 0.0001 |

## Per-stage success rate (across seeds)

| stage | success rate |
|-------|-------------:|
| ASTSR | 1.0000 |
| Hilbert | 1.0000 |
| MCTS | 1.0000 |

## Regression metrics (mean ± std)

| metric | mean | std |
|--------|-----:|----:|
| Hilbert R² | 0.4054 | 0.0808 |
| evolutionary R² | -4835.4804 | 9669.2788 |
| variance reduction | 0.0014 | 0.0028 |

## Per-seed detail

| seed | total wall | hilbert R² | evol R² | var red |
|-----:|-----------:|-----------:|--------:|--------:|
| 0 | 0.145 | 0.4073 | -0.6242 | 0.0032 |
| 1 | 0.217 | 0.3851 | -2.0188 | 0.0003 |
| 2 | 0.190 | 0.4774 | -0.5308 | 0.0048 |
| 3 | 0.182 | 0.4918 | -0.1900 | 0.0019 |
| 4 | 0.138 | 0.2654 | -24174.0380 | -0.0033 |

## Per-metal breakdown (Wf-330-T2)

| metal | n_rows | hilbert R² MEASURED | PROJECTED | evol R² | var red | verdict |
|-------|------:|-------------------:|---------:|--------:|--------:|---------|
| Ru | 18987 | 0.4626 | 0.6500 | -0.3406 | 0.0032 | MEASURED=BELOW_TARGET |
| Ir | 4546 | 0.3790 | 0.6300 | -0.5251 | 0.0032 | MEASURED=BELOW_TARGET |
| Rh | 1118 | 0.4309 | 0.6200 | -0.1792 | 0.0032 | MEASURED=BELOW_TARGET |
| Os | 1113 | 0.4526 | 0.6100 | -0.3330 | 0.0032 | MEASURED=BELOW_TARGET |
| Re | 868 | 0.3530 | 0.6000 | -0.3767 | 0.0030 | MEASURED=BELOW_TARGET |
