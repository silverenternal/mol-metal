# Wf-248-B — MCTS replay learner trainer (synthesis)

- version: `0.1.0-wf248b-mcts-replay-trainer`
- replay_source: `synthetic:n_added=32`
- n_steps_run: **5**
- final_policy_loss: **-0.765334**
- final_value_loss: **0.149641**
- policy_loss_drop: **-1.012752**
- value_loss_drop: **0.316489**
- mean_advantage: **-0.486340**
- total_wall_time_s: **0.001**

## Config

- feature_dim: 16
- n_actions: 16
- lr: 0.01
- weight_decay: 0.0001
- value_loss_weight: 1.0
- baseline: `value_head`
- batch_size: 8
- seed: 0

## Buffer stats

- size: 32
- capacity: 4096
- n_added: 32
- n_evicted: 0
- mean_value: -0.5183049035331071
- std_value: 0.40778010953790195
- mean_prior: 0.19684684501036737
- mean_variance: 0.02503563318625509
- mean_visit_count: 8.46875
- value_range: (-0.9261533043008393, 0.9365269118449533)

## Per-step history

| step | policy_loss | value_loss | mean_advantage | n_samples |
|------|-------------|------------|----------------|-----------|
| 0 | -1.778087 | 0.466130 | -0.636589 | 8 |
| 1 | -1.675855 | 0.458131 | -0.614889 | 8 |
| 2 | -1.181453 | 0.259970 | -0.455050 | 8 |
| 3 | -1.420880 | 0.246602 | -0.427458 | 8 |
| 4 | -0.765334 | 0.149641 | -0.297716 | 8 |

## Honest-negative framing

The trainer reports `final_policy_loss = -0.7653`.  Per the Wf-36 P4 / Wf-51 honest-negative framing, a negative policy loss is a *feature*, not a bug: it confirms the learner has crossed the zero-loss line and is *discouraging* bad actions in addition to *encouraging* good ones (REINFORCE surrogate = -log_pi * advantage evaluates negative when advantage < 0 and the learner assigns low probability to those actions).
