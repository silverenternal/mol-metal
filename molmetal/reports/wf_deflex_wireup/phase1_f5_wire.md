# WF-Deflex Wire-up Phase 1 — F5 Symbolic Reward Shaping Wire

## Summary

Wire the F5 symbolic-reward adapter (`learned_shaping.py`) into
`RewardAggregator.__call__` so the F5 formula contributes to the
aggregate reward on top of the existing channels.  Default OFF (env-gated
+ flag-gated), bit-for-bit legacy behaviour when unused.

## Files touched

* `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py`
  — new fields `use_learned_shaping: bool = False` and
  `w_learned_shaping: float = 1.0`; new method
  `register_learned_shaping_channel()`; new helper
  `_learned_shaping_active()` and `_learned_shaping_contribution(sa)`;
  new contribution line added to `__call__`:
  `value += self._learned_shaping_contribution(v_sa_raw)`.

* `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_deflex_wireup_phase1_f5.py`
  — 6 unit tests covering: default-off, flag-on, env-var-on, __call__
  integration, no-channels baseline, and custom-formula override.

## Contract

* When `use_learned_shaping=False` AND `LEARNED_SHAPING_ENABLED` env-var
  is unset → `_learned_shaping_contribution()` returns `0.0`.  Existing
  reward is bit-for-bit identical to the legacy behaviour.
* When the flag is True OR the env-var is set → contribution is
  `w_learned_shaping * (2.5836 - 2.5149 * sa_score)`.  Computed lazily
  via `LearnedShaping()` so the production import surface stays lean
  when F5 is unused.
* Custom formulas can be passed via
  `register_learned_shaping_channel(formula=...)` for callers that have
  re-fit on a new cohort via
  `learned_shaping.fit_from_metrics()`.

## Honest framing

The F5 formula was recovered from n=33 MEASURED Round-12 path-(a) cells
with LassoLarsIC + BIC.  In-sample R^2 = 1.0, LOO R^2 = -0.0635 — close
to random. The wire-up is INTERPRETIVE (it identifies sa_norm as the
lone surviving signal), not predictive.  Production pilots should log
the F5 contribution alongside the dominant channels and NOT trust the
F5 lift as a measurement without re-fit validation on the target
cohort.

## Verification

* `uv run pytest tests/test_deflex_wireup_phase1_f5.py -v` → **6 passed**
* Full chain: `uv run pytest tests/test_deflex_wireup_*.py -v` →
  **26 passed** (no regression in phase 2/3/4/5 tests).