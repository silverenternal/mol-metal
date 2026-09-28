# Wf-MCTS-Real-Bench — benchmark_log

## Setup

- PlatinAI CSV: `/home/hugo/codes/mol-metal/molmetal/data/platinai_5k.csv`
- n_simulations = 2
- max_depth     = 2
- top_k         = 5
- RNG seed      = 0
- Tile library  : STANDARD_12_TILES (canonical Phase-0 12 tiles)
- Reaction rules: REACTION_RULES (3 samples searched)

## Per-strategy results

| strategy      | wall-time (s) | mean/sample (s) | max/sample (s) | success_rate | total_cands | n_distinct (mean) | errors |
|---------------|--------------:|----------------:|---------------:|-------------:|------------:|------------------:|-------:|
| base          |        0.136 |         0.0453 |        0.0517 |      0.0000 |          0 |           0.0000 |     0 |
| variance_puct |        0.136 |         0.0452 |        0.0512 |      0.0000 |          0 |           0.0000 |     0 |
| variance_mcts |        0.134 |         0.0448 |        0.0511 |      0.0000 |          0 |           0.0000 |     0 |

## Wall-time ratios (variance / base)

- variance_puct / base = **0.998**
- variance_mcts / base = **0.989**

## Interpretation

- variance_puct ran the same number of candidates in 0.2% less wall-time than the base.
- variance_mcts ran in 1.1% less wall-time than the base.

- Total candidates: base = 0, variance_puct = 0, variance_mcts = 0.
- Success rates: base = 0.000, variance_puct = 0.000, variance_mcts = 0.000.

## Reproduce

```
uv run --no-sync python molmetal/scripts/benchmark_mcts_real.py \
    --n-samples 3 \
    --n-simulations 2 \
    --max-depth 2 \
    --top-k 5 \
    --seed 0
```
