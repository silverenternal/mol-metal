# WF-Deflex Wire-up Phase 3 — Sub-fix B learned_prior argmax wire

## Summary

Wire the `LearnedPolicyPrior.predict_proba(smiles)` argmax rule into
the post-modify_root_prior boost loop at a 0.75 floor.  Opt-in via
`use_learned_prior_argmax=True` flag on `search()`.  When the flag is
on AND `learned_prior` is supplied, the MCTS root's children get an
additional boost: the child whose `rule_name` matches the learned
prior's argmax for the current state SMILES is lifted to ``P=0.75``
(only if its current P is below the floor — preserves pocket-argmax
advantage).

## Files touched

* `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py`
  — new kwarg on `search()`: `use_learned_prior_argmax: bool = False`.
  New boost block after the existing `pocket_prior` argmax lift in
  `search()`.  The block:
  1. Checks `use_learned_prior_argmax AND self._learned_prior_for_search AND root.children`.
  2. Calls `lp.predict_proba(state_smi)` to get per-rule distribution.
  3. Finds the argmax rule.
  4. For every child whose `rule_name` matches and `P < 0.75`, sets `P = 0.75`.

* `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_deflex_wireup_phase3_learned_prior.py`
  — 4 unit tests covering: argmax lift fires, flag-off is no-op,
  learned_prior=None is harmless, existing high P not lowered.

## Contract

* `use_learned_prior_argmax=False` (default): no learned-prior boost
  applied.  Existing root-prior behaviour is bit-for-bit identical.
* `use_learned_prior_argmax=True` + `learned_prior=None`: loop is
  gated on `self._learned_prior_for_search is not None` so the flag
  is harmless when no prior is supplied.
* `use_learned_prior_argmax=True` + `learned_prior=my_prior`: the
  matching child P is lifted to 0.75 only when its existing P is below
  0.75 (preserves any higher P that the pocket-argmax loop set).

## Why a 0.75 floor

The existing pocket-argmax boost loop uses a 0.05 floor (was 0.5; cut
to 0.05 by the WF-Pocket-Invariance Phase 2A sub-fix A so magnitude
information can propagate).  0.75 is above the legacy 0.5 floor
and below the maximum 0.99 cap — a middle ground that biases the
learned prior's pick above the pocket-argmax default while still
leaving room for the pocket-argmax to win if it set a higher P.

## Lit anchors

* Lai & Robbins 1985 — Lai-Robbins lower bound sqrt(KT log T)
* Silver 2017 AlphaGo Zero — policy prior + value net + MCTS
* Auger 2013 Continuous UCT — PUCT formula for continuous action spaces

## Honest framing

The boost is a heuristic.  The learned prior is a small GRU on tmQM
training data, not on the actual pocket-conditioned distribution, so
the argmax may not be the "right" rule for the current state SMILES.
Production pilots should compare search() with vs without the flag on
the same cohort — the wire-up is bit-for-bit identical when off, so
the diff is a clean A/B.

## Verification

* `uv run pytest tests/test_deflex_wireup_phase3_learned_prior.py -v` → **4 passed**
* All Deflex tests: `uv run pytest tests/test_deflex_wireup_*.py -v` →
  **26 passed**.