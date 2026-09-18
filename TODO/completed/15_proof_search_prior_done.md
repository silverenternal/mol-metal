# Switch proof_search.py:153 heuristic() to RewardAggregator-based prior

**Status:** pending
**Priority:** high
**Effort:** 1d
**Owner:** (unset)
**Depends on:** 01 (closed_loop rewire)
**Blockers:** none
**Created:** 2026-09-12

## Goal
Replace the constant 0.5 heuristic prior at `proof_search.py:153` with a
RewardAggregator-based prior so MCTS leaf expansion weights candidates by
their expected multi-objective reward (Vina affinity + PoseBusters pass +
AiZynth route availability + pic50 + qed + sa) instead of the hand-coded
constant. The channels already exist (proof_search.py:519-587) — this task
just rewires the default prior in `ProofSearch` instantiation.

## File(s) to edit
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py  (line 153 `heuristic()` and ProofSearch constructor)

## Success criterion
`ProofSearch` instantiation no longer calls `heuristic(features)` for leaf
expansion prior; leaf value is r_vina + r_posebusters + r_pic50 + r_sa
weighted aggregate from `RewardAggregator` with per-channel 0.0 fallback;
test_proof_search.py grows by ≥3 cases asserting non-constant leaf values.

## Related reports
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_round3_combined_report.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/audit_lambda_upper_bound.md