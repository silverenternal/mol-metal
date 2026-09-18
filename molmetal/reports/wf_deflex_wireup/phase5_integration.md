# WF-Deflex Wire-up Phase 5 — End-to-End integration smoke

## Summary

Bring up the full Deflex wire-up chain (F5 + PocketMacro v2 +
learned_prior argmax) and verify the three adapters co-exist without
crashing.  This is a "no crash" integration test — it does NOT run a
full MCTS rollout (which requires rdkit + chemistry fixtures out of
scope for this 90-min wire-up).

## Test results

* `uv run pytest tests/test_deflex_wireup_phase5_integration.py -v` →
  **3 passed**.
* `uv run pytest tests/test_deflex_wireup_*.py -v` → **26 passed** in
  1.40s total wall.

## What the integration test verifies

1. `test_full_chain_no_crash` — builds all three adapters
   (`RewardAggregator` + `PocketMacroInference` + stub
   `LearnedPolicyPrior`) and confirms they coexist without exception.
2. `test_aggregator_f5_call_with_pocket_macro_embedding` — the
   `__call__` math for an aggregator with F5 + SA + 32-d PMI embedding
   matches the analytical expression
   `w_sa * (1 - (sa-1)/9) + w_f5 * (2.5836 - 2.5149*sa)`.
3. `test_search_kwargs_accepted` — the new `search()` kwargs
   (`use_pocket_macro`, `pocket_macro_target_name`,
   `use_learned_prior_argmax`) are present in the signature with the
   correct defaults (False / None / False) so existing callers stay
   bit-for-bit compatible.

## Wall budget

* Wire-up: ~25 min (4 phases + tests).
* Tests: ~1.4 s wall.
* Total: well within the 90 min budget.

## Honest framing

The integration test does NOT execute the production caller
`r4_lambda_only_run.py --use-learned-shaping --use-pocket-macro
--use-learned-prior`.  The required flags do not exist on
`r4_lambda_only_run.py` yet (this workflow owns `proof_search.py`
only, per the task constraints).  The downstream caller work is owned
by Workflow 2 and is out of scope for this 90-min wire-up.

What IS verified:
* F5 contribution fires on `__call__` when the flag is on.
* PocketMacro v2 loads and produces correct 32-d embeddings.
* LearnedPolicyPrior argmax rule fires at the 0.75 floor.
* All three coexist without exception.

What is NOT verified (next step, owned by Workflow 2):
* End-to-end MCTS rollout on cisplatin with all 3 flags ON.
* Per-pocket diversity lift measurement.
* Reward-channel contribution breakdown.