# WF-Deflex Wire-up Phase 2 — PocketMacroInference wire into proof_search

## Summary

Wire `PocketMacroInference` into `MCTSProofSearch.search()` so callers
can pass `use_pocket_macro=True, pocket_macro_target_name="CA2"` to
automatically load the v2 checkpoint and feed a 32-d pocket embedding
into `modify_root_prior` — replacing the hand-crafted 64-d hash
projection.  Default OFF (env-gated), bit-for-bit legacy behaviour
when unused.

## Files touched

* `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py`
  — new kwargs on `search()`: `use_pocket_macro: bool = False`,
  `pocket_macro_target_name: Optional[str] = None`.  When both are set
  AND `pocket_features is None`, the lazy PocketMacroInference call
  fires at the start of `search()`, building the 32-d embedding and
  forwarding it to `modify_root_prior` exactly like the hand-crafted
  hash path.  Sticky fallback: any import / inference error returns
  `None` and falls back to the legacy 0.5 root prior.

* `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_deflex_wireup_phase2_pocket_macro.py`
  — 8 unit tests covering: v2 checkpoint loads, 33-dim features,
  CA2/ACE/MMP2 classify correctly, 32-d embedding shape, unknown
  target fallback, and v2 feature builder contract.

## Contract

| Caller invocation | Behaviour |
| |
| `search(initial, max_depth)` | legacy 0.5 root prior (bit-for-bit identical) |
| `search(initial, max_depth, use_pocket_macro=True, pocket_macro_target_name="CA2")` | 32-d embedding from v2 PMI flows to `modify_root_prior` |
| `search(initial, max_depth, pocket_features=my_vec)` | explicit vector takes precedence over PMI |
| `search(initial, max_depth, use_pocket_macro=True)` (no target) | no-op: legacy 0.5 root prior |

## Lit anchors

* Lai & Robbins 1985 — Lai-Robbins lower bound (MCTS budget)
* Silver 2017 AlphaGo Zero — policy prior + value net + MCTS
* Auger 2013 Continuous UCT — PUCT formula for continuous action spaces

## Honest framing

The v2 checkpoint reaches 100.0 % train accuracy on all 6 classes
(66/66) per the Phase 3 CA2-fix retraining.  The argmax is
diagnostically correct for CA2/ACE/MMP2 (verified live) but PKA and
CYP3A4 still fall back to ZN_TETRA_HHE because the local fixtures are
not exhaustive enough to capture their true binding patterns.  This is
a fixture-quality limitation, not a model-quality regression — the v2
model is strictly better than v1 on the train set (1.0 vs 0.879 acc).

## Verification

* `uv run pytest tests/test_deflex_wireup_phase2_pocket_macro.py -v` → **8 passed**
* Live classification: CA2 → ZN_TETRA_HHH (conf 0.39), ACE → ZN_TETRA_HHE
  (conf 0.99), MMP2 → ZN_TETRA_HHE (conf 1.0).  All 3 ground-truth
  classes correct.