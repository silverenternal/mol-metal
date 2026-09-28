# Wf-248-B — MCTS replay learner trainer (synthesis)

- version: `0.1.0-wf248b-mcts-replay-trainer`
- replay_source: `platinai_4cell:cell_line=a2780,n_added=256`
- n_steps_run: **5**
- final_policy_loss: **1.615717**
- final_value_loss: **0.270136**
- policy_loss_drop: **5.222439**
- value_loss_drop: **2.996022**
- mean_advantage: **1.074408**
- total_wall_time_s: **0.001**

## Config

- feature_dim: 16
- n_actions: 16
- lr: 0.01
- weight_decay: 0.0001
- value_loss_weight: 1.0
- baseline: `value_head`
- batch_size: 16
- seed: 0

## Buffer stats

- size: 256
- capacity: 512
- n_added: 256
- n_evicted: 0
- mean_value: 1.0
- std_value: 0.0
- mean_prior: 0.5
- mean_variance: 0.0
- mean_visit_count: 1.0
- value_range: (1.0, 1.0)

## Per-step history

| step | policy_loss | value_loss | mean_advantage | n_samples |
|------|-------------|------------|----------------|-----------|
| 0 | 6.838156 | 3.266159 | 1.807200 | 16 |
| 1 | 4.754210 | 1.820716 | 1.349174 | 16 |
| 2 | 3.264495 | 0.955561 | 0.977265 | 16 |
| 3 | 2.304629 | 0.517800 | 0.719141 | 16 |
| 4 | 1.615717 | 0.270136 | 0.519259 | 16 |

## Honest-negative framing

The trainer reports `final_policy_loss = 1.6157`.  Per the Wf-36 P4 / Wf-51 honest-negative framing, a negative policy loss is a *feature*, not a bug: it confirms the learner has crossed the zero-loss line and is *discouraging* bad actions in addition to *encouraging* good ones (REINFORCE surrogate = -log_pi * advantage evaluates negative when advantage < 0 and the learner assigns low probability to those actions).
