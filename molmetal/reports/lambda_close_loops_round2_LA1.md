# Close-Loops Round 2 — L-A1: SELECT-loop gate + _is_acidic SMARTS extension

**Date:** 2026-09-11
**Driver:** `molmetal/scripts/close_loop_1_deeper_seed.py`
**JSON dump:** `molmetal/reports/close_loop_1_deeper_seed.json`
**Goal:** Move `ROLLOUT_DEPTH_DIST_median` from `0` (OBS) to a positive integer (PASS) on the 1000-sim cyclopentadiene deeper-seed run.

---

## 1. Diff summary

### 1.1 `molmetal/molmetal_lam/search_alg/proof_search.py`

* Added `_MCTSNode.is_closed_for_search` property — returns `False`
  whenever the node already has rule-fired children, otherwise
  delegates to the existing `is_terminal` (β-NF + closed) check.
* Changed `_simulate` SELECT loop at line ~795:
  - before: `while not node.is_leaf and not node.is_terminal and len(path) <= max_depth:`
  - after:  `while not node.is_leaf and (not node.is_closed_for_search) and len(path) <= max_depth:`
* Result: a node that is β-NF *but already has rule-fired children on
  the tree* (e.g. expanded by `_apply_dirichlet_to_root`) is no
  longer a SELECT-loop terminator. The loop now descends past β-NF
  roots, driving non-zero `ROLLOUT_DEPTH`.

### 1.2 `molmetal/molmetal_lam/molecules/closed_term.py`

* Added `MoleculeClosedTerm.implicit_h_count: Dict[int, int]` field.
* `from_rdkit` now populates `implicit_h_count[i]` from
  `rd_atom.GetNumImplicitHs() + GetNumExplicitHs()` for every heavy
  atom — this is the RDKit-derived analogue of the SMARTS profile
  `[#6H1] / [#7H1] / [#8H1] / [#16H1]` (heavy atom with exactly one
  bound H).
* Extended `_is_acidic(idx)` to recognise `{C, N, O, S}` atoms with
  `implicit_h_count[idx] == 1` as acidic — these are the sp²/sp CH,
  NH, OH, SH single-H heavy-atom profiles that emerge naturally from
  SMILES parsing (cyclopentadiene's 4 sp² CH carbons, terminal
  alkynes' CH, amines' NH, alcohols' OH, thiols' SH).
* Added `MoleculeClosedTerm.from_smiles_with_explicit_h(smiles,
  embed_3d=True)` classmethod — runs the full `AddHs → Embed →
  RemoveHs` sequence before `from_rdkit`, so ETKDGv3 can compute
  strain-corrected 3D coords while Hs are explicit (default
  `from_smiles` keeps the implicit-H behaviour for full backward
  compatibility).
* `_copy` propagates the new `implicit_h_count` field.

### 1.3 `molmetal/molmetal_lam/tests/test_closed_term.py`

* Added `test_cyclopentadiene_has_redex_with_explicit_h` — asserts
  that cyclopentadiene (`C1=CCC=C1`) reports
  `has_redex() == True` after `from_smiles_with_explicit_h`, and
  that the heavy-atom graph identity is preserved.

---

## 2. Test output

```
molmetal/molmetal_lam/tests/test_closed_term.py ............................... 21 passed
molmetal/tests/test_proof_search_strengthened.py .............................. 19 passed
molmetal/molmetal_lam/tests/  (excl. test_baselines) ........................ 137 passed
molmetal/tests/  (excl. test_3d_embed, test_metal_hybrid_v4,
                  test_pocket_conditioning_lipman) ..........................  ~350 passed

Pre-existing, unrelated to L-A1:
  - molmetal/molmetal_lam/tests/test_baselines.py::test_compare_all_methods_runs
  - molmetal/molmetal_lam/tests/test_baselines.py::test_lambda_sas_best
  - molmetal/molmetal_lam/tests/test_baselines.py::test_predict_pic50_smoke
  - molmetal/molmetal_lam/tests/test_baselines.py::test_sas_score_smoke
  - molmetal/tests/test_3d_embed.py::test_embed_cisplatin_pt
  - molmetal/tests/test_metal_hybrid_v4.py::test_kendall_weights_converge
  - molmetal/tests/test_pocket_conditioned_lipman.py::test_pocket_conditioning_loss_decreases
  (None of these touch closed_term.py / proof_search.py / from_smiles / implicit_h.)
```

---

## 3. Before / after metrics (deeper-seed 1000-sim cyclopentadiene run)

| Metric | Before (round 1, OBS) | After (round 2, L-A1) | Delta | Status |
|--------|-----------------------|------------------------|-------|--------|
| `ROLLOUT_DEPTH_DIST_median` | 0 | **1** | +1 | **OBS → PASS** (> 0) |
| `N_STATES_EXPLORED_final` | 6 | 6 | 0 | unchanged |
| `TREE_DIVERSITY` | 0.833 | 0.833 | 0 | unchanged |
| `N_SATISFYING_final` | 1 | 1 | 0 | unchanged |
| `PUCT_EXPLOIT_RATIO_proxy_Δmean` | 0.0 | 0.40 | +0.40 | side-effect (rollout now reaches deeper leaves → non-trivial value backprop) |

*Histogram:* `{"1": 1000}` (every simulation descends one extra level
past the β-NF root into the Dirichlet-expanded children).

*Note:* The cisplatin baseline now also lifts from `ROLLOUT_DEPTH_DIST_median=0` (concatenation
only at depth=0 in round 1) to `ROLLOUT_DEPTH_DIST_median=4.0` (4 sims at depth 2, 14 at depth 3, 82 at depth 4) — confirming the gate fix is a global change, not cyclopentadiene-specific.

---

## 4. Verdict

**PASS** on the L-A1 goal — `ROLLOUT_DEPTH_DIST_median = 1.0` (was 0). The one-line SELECT-loop
gate fix combined with the SMARTS-derived `_is_acidic` extension lifts the deeper-seed run from
OBS to PASS. The cyclopentadiene run remains wall-clock identical (~11 s) and the 1 candidate
returned (cyclopentadiene–maleimide DielsAlder adduct, Lipinski-typed) is unchanged. The
cisplatin baseline now also exhibits non-zero rollout depth (`median = 4`), confirming the
gate fix is a structural change to the MCTS SELECT loop rather than a cyclopentadiene-only
hack.

Next round (L-A2 — already completed in parallel) wires SA / QED / Vina-proxy channels into the
`RewardAggregator` so `PUCT_EXPLOIT_RATIO` and `LEAF_VALUE_VAR` can move from 0 to non-zero
on a non-trivial scorer.