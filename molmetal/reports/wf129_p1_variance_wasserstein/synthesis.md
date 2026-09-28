# Wf-129 Phase 1 — Variance MCTS x Wasserstein (Synthesis)

**Date**: 2026-09-27  
**Tool**: `molmetal/scripts/bench_variance_mcts_wasserstein.py` (0.1.0-wf129-p1)  
**Scope**: Wf-129 P1 — variance_mcts <-> proof_search integration with Wf-123 P1 Wasserstein regulariser.

## §1 Goal

Lift variance_reduction of the variance-controlled MCTS (Wf-11 / Wf-36 / Wf-50 chain) from the Wf-39 P1 honest-negative baseline (`variance_reduction = -0.185`) toward `>= -0.05` by mixing the Wf-123 P1 Wasserstein (W_2) regulariser (sliced-W_2 estimator, n_mc=256) into the per-arm prior at the MCTS layer.

## §2 Method

Three strategies are run on the Wf-36 P2 mock chemistry (identity concatenation rule, 4 tile SMILES, hash-derived scorer) for `n_seeds = 1` (seeds = [0]) and `n_trajectories = 5` per seed:

* `vanilla` — `variance_mcts_integration.VarianceMCTSWrapper` with no W_2 hook (== Wf-39 P1 baseline).
* `w2_reg`  — `w2_reg` adds `alpha * wasserstein_loss(pred, ref)` to the per-arm prior at trajectory start.
* `w2_reg_decay` — `w2_reg` + JKO CFL decay check gate (`wasserstein_decay_check` on a 5-step history).

The W_2 reference distribution is `prior` (uniform over the `action_count` arms) by default.  All runs are CPU + (opt) GPU-0 — torch is lazy-imported inside the hook so the bench still runs when only numpy/mock chemistry is available.

## §3 Results

### §3.1 Aggregate (cross-seed)

| Strategy | mean_var | best_score | success_rate |
| --- | --- | --- | --- |
| `vanilla` | 0.000000 | 0.0000 | 0.000 |
| `w2_reg` | 0.000000 | 0.0000 | 0.000 |
| `w2_reg_decay` | 0.000000 | 0.0000 | 0.000 |

### §3.2 Variance reduction vs vanilla

| Strategy | Variance reduction (per-seed mean +/- std) | Target met |
| --- | --- | --- |
| `w2_reg` | +1.0000 +/- 0.0000 | NO |
| `w2_reg_decay` | +1.0000 +/- 0.0000 | NO |

## §4 Verdict

**Bench did NOT execute** — the baseline (`vanilla`) recorded `mean_var = 0.0` and `success_rate = 0.0`, meaning the MCTS layer never produced candidates.  Most likely the bench ran on a torch-less host (the lazy-import fallback in ``bench_variance_mcts_wasserstein.py`` short-circuits to no-op when ``proof_search`` cannot be imported).  The `1.0` / `+1.0` numbers above are the artifact of `1 - 0 / 1e-12 == 1` and are **NOT** a variance-reduction measurement.  Re-run on a host with torch installed to obtain a real ``variance_reduction``.

## §5 Honest framing (TODO-92 §0 + TODO-126 §7)

* The mock chemistry is **not** paper-grade (per Wf-43 P3).  All `variance_reduction` numbers are relative A/B on the same mock chemistry; they are NOT real-data claims.
* The W_2 regulariser is a NEW scoring component (per TODO-126 §7); every per-trajectory diagnostic in `per_strategy_per_seed[*].w2_diagnostics[*]` carries `engine_tag = "wasserstein_reg"` so downstream reporting can attribute Wasserstein contributions honestly.
* The Wf-123 P1 Wasserstein loss is a CPU-only skeleton; the GPU-batched W_2 lives in Wf-129 P2 / TODO-121.

## §6 Cross-references

* `molmetal/scripts/bench_variance_mcts_wasserstein.py` (this run)
* `molmetal/tests/test_variance_mcts_wasserstein.py` (12 unit tests)
* `molmetal/reports/wf129_p1_variance_wasserstein/per_seed_metrics.json` (cross-seed aggregate)
* `molmetal/reports/wf129_p1_variance_wasserstein/seeds/<s>/per_seed_metrics.json` (per-seed raw)
* `TODO/pending/92_r20_p1_variance_puct_lazy_rdkit.md` §3 (parent plan, R20-P1)
* `TODO/pending/126_innovation_strengthen_roadmap.md` §3 (M1 Wasserstein spec)
* `TODO/completed/85_wf39_mcts_real_trajectory.md` (Wf-39 P1)
