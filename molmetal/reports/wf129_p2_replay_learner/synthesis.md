# Wf-129 Phase 2 — MCTS replay learner regularised (Synthesis)

**Date**: 2026-09-27  
**Tool**: `molmetal/scripts/bench_mcts_replay_learner_regularized.py` (0.1.0-wf129-p2)  
**Scope**: Wf-129 P2 — AlphaGo-style policy/value network training with L2 weight decay + W_2 prior/value regularisers on the Wf-36 P4 synthetic replay buffer.

## §1 Goal

Lift the Wf-36 P4 honest-negative replay learner (`policy_loss_drop = -0.046`, final `policy_loss = 2.795`) toward `policy_loss < 0.5` AND `value_loss < 1.0` by adding three regularisation ingredients that AlphaZero / AlphaGo Zero (2016 §3.4, 2018 §3) prescribes but our CPU-only stub skipped:

* **L2 weight decay** (`weight_decay=1e-4`) on both heads.
* **Wasserstein-2 prior regulariser** (`beta=0.05`) on the policy head — keeps the predicted policy close to the MCTS-improved visit-count target in *probability-space*.
* **Wasserstein-2 value regulariser** (`gamma=0.01`) on the value head — aligns the scalar output with the leaf-value target.

## §2 Method

Three strategies are run on the Wf-36 P4 synthetic replay buffer (n_entries=128, n_states=8, n_actions=16, feature_dim=16) for `n_seeds = 5` (seeds = [0, 1, 2, 3, 4]) and `n_epochs = 30` per seed:

* `vanilla` — Wf-36 P4 baseline (no L2, no W_2).
* `l2_only` — vanilla + L2 weight decay on both heads.
* `l2_w2`   — l2_only + W_2 prior + W_2 value regularisers.

All runs use `batch_size = 32` and `lr = 0.001`.  The W_2 estimator is the Wf-123 P1 sliced-W_2 at `n_mc = 256` (CPU + GPU-0; lazy-imported; degrades to numpy-only when torch is unavailable).

## §3 Results

### §3.1 Aggregate (cross-seed)

| Strategy | final_policy_loss | final_value_loss | policy_loss_drop | value_loss_drop | converged_rate |
| --- | --- | --- | --- | --- | --- |
| `vanilla` | 3.1235 | 0.2894 | -0.0209 | +0.2190 | 0.00 |
| `l2_only` | 3.1257 | 0.2927 | -0.0914 | +0.2190 | 0.00 |
| `l2_w2` | 3.1316 | 0.2969 | -0.0921 | +0.2207 | 0.00 |

### §3.2 W_2 diagnostics (l2_w2 only)

| Strategy | mean_w2_policy | mean_w2_value |
| --- | --- | --- |
| `l2_w2` | 0.105701 | 0.489405 |

## §4 Verdict

**Target NOT met** — `l2_w2` records `final_policy_loss = 3.1316` (target `< 0.5`) and `final_value_loss = 0.2969` (target `< 1.0`).  Possible causes (no fabrication):

* The synthetic fixture (Wf-36 P4) is too small (128 entries) for the W_2 regulariser to converge in the prescribed epoch budget.
* `beta=0.05` / `gamma=0.01` are too small to meaningfully regularise — a Wf-129 P3 alpha sweep is the natural next step.
* The numpy-only W_2 fallback degrades to an L1 distance when pred/ref have mismatched last-dim; on small batches this is noisy.  GPU retraining (TODO-92 R20-P1.3) would replace it with the GPU-batched sliced-W_2.

## §5 Honest framing (TODO-92 §0 + TODO-126 §7)

* The synthetic fixture is **NOT** paper-grade (per Wf-43 P3 / Wf-129 P1 §5).  All `policy_loss` / `value_loss` numbers are computed on the Wf-36 P4 synthetic 128-entry replay buffer; they are *bench-level* measurements, not real-data claims.
* The W_2 regulariser is a NEW scoring component (per TODO-126 §7); every per-batch diagnostic in `per_strategy_per_seed[*].history[*].w2_diagnostics[*]` carries `engine_tag = "w2_replay_reg"` so downstream reporting can attribute W_2 contributions honestly.
* The W_2 estimator is the Wf-123 P1 sliced-W_2 (CPU + GPU-0 skeleton); the GPU-batched version lives in Wf-129 P3 / TODO-121.

## §6 Cross-references

* `molmetal/scripts/bench_mcts_replay_learner_regularized.py` (this run)
* `molmetal/tests/test_mcts_replay_learner_regularized.py` (26 unit tests)
* `molmetal/reports/wf129_p2_replay_learner/per_seed_metrics.json` (cross-seed aggregate)
* `molmetal/reports/wf129_p2_replay_learner/seeds/<s>/per_seed_metrics.json` (per-seed raw)
* `molmetal/scripts/bench_variance_mcts_wasserstein.py` (Wf-129 P1, parent)
* `molmetal/scripts/mcts_replay_learner.py` (Wf-36 P4 ship)
* `TODO/completed/85_wf39_mcts_real_trajectory.md` (Wf-39 P1 L2 baseline)
