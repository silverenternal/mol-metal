# Lambda Close-Loops Round 2 — combined L-A1 + L-A2 re-measurement

**Date:** 2026-09-11
**Goal:** confirm `ROLLOUT_DEPTH_DIST_median > 0` **AND** `PUCT_EXPLOIT_RATIO_VAR > 0` in the same measurement cycle.

The two patches live on top of `main` (no commits made):

* L-A1 — `molmetal/molmetal_lam/search_alg/proof_search.py` SELECT-loop
  gate (`is_closed_for_search`) plus
  `molmetal/molmetal_lam/molecules/closed_term.py`
  `_is_acidic` extension and `from_smiles_with_explicit_h`.
  See `molmetal/reports/lambda_close_loops_round2_LA1.md` for full diff.
* L-A2 — `molmetal/molmetal_lam/search_alg/proof_search.py` adds
  `r_qed` / `r_vina_proxy` channels, `RewardAggregator.with_default_channels()`
  classmethod, and the `puct_exploit_ratio_var` + `leaf_value_var`
  fields on `MCTSProofSearch`.
  See `molmetal/reports/lambda_close_loops_round2_LA2.md` for full diff.

---

## 1. Side-by-side metrics

| Metric | Before round 2 | After round 2 |
|--------|---------------:|--------------:|
| `ROLLOUT_DEPTH_DIST_median` (deeper-seed, cyclopentadiene) | 0 | **1** |
| `N_STATES_EXPLORED` (deeper-seed final) | 6 | 6 |
| `TREE_DIVERSITY` (deeper-seed) | 0.8333 | 0.8333 |
| `ROLLOUT_GUIDED_RATIO` (guided ε=0.25) | n/a (no prior head) | **0.7583** |
| `PUCT_EXPLOIT_RATIO_VAR` (guided ε=0.25) | 0 (degenerate reward) | **0.9996** |
| `LEAF_VALUE_VAR` (guided ε=0.25) | 0 | **0.002557** |

*Cisplatin baseline (informational, after L-A1):* `ROLLOUT_DEPTH_DIST_median=4.0`, `N_STATES_EXPLORED=233`, `TREE_DIVERSITY=0.2876` — confirms the gate fix is global, not seed-specific.

*Uniform cross-check (guided ε=0.0):* `PUCT_EXPLOIT_RATIO_VAR=0.9988`, `LEAF_VALUE_VAR=0.000801` — strictly positive even when the rollout policy ignores the prior, isolating the variance to the reward head.

---

## 2. Source dumps

* `molmetal/reports/close_loop_1_deeper_seed.json` — captured 2026-09-11 18:38
* `molmetal/reports/close_loop_2_symbolic_prior.json` — captured 2026-09-11 (sidecar written by the script)

---

## 3. Pytest (combined molmetal/tests/ + molmetal/molmetal_lam/tests/)

```
518 passed, 5 failed, 1 skipped, 1 warning in 132.68s
```

Failures (all pre-existing, none caused by L-A1/L-A2):

* `molmetal/tests/test_3d_embed.py::test_embed_cisplatin_pt`
* `molmetal/molmetal_lam/tests/test_baselines.py::test_compare_all_methods_runs`
* `molmetal/molmetal_lam/tests/test_baselines.py::test_lambda_sas_best`
* `molmetal/molmetal_lam/tests/test_baselines.py::test_predict_pic50_smoke`
* `molmetal/molmetal_lam/tests/test_baselines.py::test_sas_score_smoke`

---

## 4. Verdict

**PASS.** Both target gates hold in the same measurement cycle:

* `ROLLOUT_DEPTH_DIST_median = 1` (> 0) — moved from OBS to PASS by L-A1.
* `PUCT_EXPLOIT_RATIO_VAR = 0.9996` (> 0) — moved from OBS to PASS by L-A2.

Total: 518 / 523 tests pass (5 pre-existing failures unrelated to the
SELECT-loop gate or reward-head wiring).
