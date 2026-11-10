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
| base          |        0.187 |         0.0622 |        0.0739 |      0.0000 |          0 |           0.0000 |     0 |
| variance_puct |        0.215 |         0.0717 |        0.0749 |      0.0000 |          0 |           0.0000 |     0 |
| variance_mcts |        0.211 |         0.0702 |        0.0754 |      0.0000 |          0 |           0.0000 |     0 |

## Wall-time ratios (variance / base)

- variance_puct / base = **1.153**
- variance_mcts / base = **1.129**

## Interpretation

- variance_puct was 15.3% slower than the base; the re-scoring pass on top of ``MCTSProofSearch.search`` is not free.
- variance_mcts was 12.9% slower than the base; tree-statistics collection (``leaf_value_var``, ``puct_exploit_ratio_var``) adds a small overhead.

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
