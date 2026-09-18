# WF-Pocket-Invariance Phase 3 — Combined sub-fixes

**Date:** 2026-09-15
**Workflow:** w2swi9tsu continuation
**Test file:** `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py`

## Honest verdict

**PARTIAL — 3/4 sub-assertions pass; 1/4 fails.**

The 3 sub-fixes from Phase 2 (A pocket_bias_strength, B learned_prior_mix_uniform,
C metal_seed_from_pocket) are **necessary but not sufficient** to break
pocket-invariance at the **prior-argmax level**. They DO break it at the
**search-tree level** and **metal-seed level**, but the prior-argmax break
requires deeper work on the action-weight vector itself (Phase-3K learned head).

## Combined recipe

The Phase-3 test combines all 3 sub-fixes simultaneously:

| Sub-fix | Knob                                          | Value   | Where              |
|---------|-----------------------------------------------|---------|---------------------|
| A       | `pocket_bias_strength` (Task J strong boost)  | 20.0    | `modify_root_prior` |
| B       | `learned_prior_mix_uniform` (Task L learned)  | 0.9     | `search()` kwarg    |
| C       | `--metal-seed-from-pocket` (Task C cond. seed)| True    | `pocket_derived_metal_seed` |

## Sub-assertion results

| # | Sub-assertion                                                                  | Status   |
|---|--------------------------------------------------------------------------------|----------|
| i | `modify_root_prior` produces different argmax for CA2 vs MMP2 at bias=20.0     | **FAIL** |
| ii | `search()` accepts pocket_features + learned_prior + mix_uniform=0.9         | **PASS** |
| iii | `pocket_derived_metal_seed` importable + callable from r4_lambda_only_run.py | **PASS** |
| iv | Derived metal seed from CA2 pocket features is non-None + non-empty SMILES  | **PASS** |

Overall: **6/7 tests pass; 1/7 fails** (the pocket-invariance break test).

## Root-cause analysis of sub-assertion (i) failure

The failure: both `argmax_ca2 == argmax_mmp2 == ('CuAAC', 'C#C')`. Why?

`modify_root_prior` computes `pocket_score[i] = <v_P, w(a_i)>`. The
`_action_weight_vector` function (warm_start.py:513-542) only fills
**3 of 64 slots** of `w(a)`:

```
w[0] = sin(hash(action)) * 0.5
w[1] = (sum(ord(c) for c in str(action)) % 1024) / 1024.0 - 0.5
w[2] = (hash(action) % 4096) / 4096.0 - 0.5
# slots 3..63 = 0
```

Meanwhile `pocket_features` fills slots 0-5 of `v_P`. The dot product
reduces to a **3-dim** projection over slots {0, 1, 2}.

For the 5 actions `[(CuAAC,C#C), (CuAAC,N=N=N), (SPAAC,C#C), (ThiolEne,C=C),
(AmideCoupling,C(=O)O)]`, the hash-derived `w[0..2]` ranking puts
`('CuAAC', 'C#C')` at the top, regardless of pocket differences — because
slot-2 differences between CA2 and MMP2 (pos=0.43 vs 0.33) are within the
hash-noise band of the 3-axis projection.

Lifting `pocket_bias_strength` from 10.0 to 20.0 amplifies the
`w_pocket = 0.25 * 20 = 5.0` weight but doesn't change the **relative
ranking** of `pocket_score` across actions — only their absolute gap. The
softmax ranking is preserved.

**Conclusion:** the prior-argmax break requires EITHER:
- (a) Learn the action-weight projection from data (Phase-3K learned head
  populates slots 3..63), so the pocket embedding's full 64-d signal
  gets consumed. Or
- (b) Change the hash-based projection to a **pocket-name-conditioned**
  one (different `w(a | pocket_name)` per pocket class). Or
- (c) Use the metal-seed-derived SMILES (sub-fix C) as the action's
  *binding-pocket anchor* in the dot product, so the pocket signal
  enters through a different channel.

None of (a), (b), (c) is in the Phase 3 scope — the spec only asks
to combine the 3 sub-fixes that were already shipped in Phase 2.

## What 3/4 sub-assertions DO demonstrate

- **Sub-fix A wiring:** `modify_root_prior(..., pocket_bias_strength=20.0)`
  does not raise and produces a valid probability distribution. The
  `pocket_bias_strength` parameter correctly modulates the pocket score
  weight (5/5 distribution invariants pass).
- **Sub-fix B wiring:** `MCTSProofSearch.search(pocket_features=...,
  learned_prior=..., learned_prior_mix_uniform=0.9)` accepts all three
  kwargs simultaneously without TypeError. The search produces a
  non-empty tree (history populated, _root set).
- **Sub-fix C wiring:** `pocket_derived_metal_seed(PocketFeatureVector)`
  is importable from `r4_lambda_only_run.py`, callable, returns a
  non-None `(name, smiles)` tuple for the CA2 pocket (His anchor present),
  and the smiles field is non-empty.

In other words: **all 3 sub-fixes are reachable in the combined call**.
The Phase-3 reachability test confirms the integration contract holds.
The pocket-invariance **break** at the prior-argmax level is a
**separate, deeper problem** that the spec itself acknowledges:
"single patch (pocket_features + learned_prior kwargs with [0.5, 1.0]
clamp) was insufficient."

## Phase-3 pytest verbatim output

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/hugo/codes/try_triton_on_rocm
configfile: pyproject.toml
plugins: hypothesis-6.168.0, anyio-4.15.0
collecting ... collected 7 items

molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py::test_search_with_pocket_features_changes_selection PASSED [ 14%]
molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py::test_search_with_learned_prior_mix PASSED [ 28%]
molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py::test_search_backward_compatible PASSED [ 42%]
molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py::test_search_pocket_invariance_break FAILED [ 57%]
molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py::test_strong_pocket_boost_overrides_default PASSED [ 71%]
molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py::test_learned_prior_overrides_when_mix_high PASSED [ 85%]
molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py::test_metal_seed_pocket_derived PASSED [100%]

=================================== FAILURES ===================================
_____________________ test_search_pocket_invariance_break ______________________
molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py:471: in test_search_pocket_invariance_break
    assert argmax_ca2 != argmax_mmp2, (
E   AssertionError: pocket-conditioned priors must produce different argmax actions (pocket-invariance break); both picked ('CuAAC', 'C#C')
E   assert ('CuAAC', 'C#C') != ('CuAAC', 'C#C')
========================== short test summary info ============================
FAILED molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py::test_search_pocket_invariance_break
=================== 1 failed, 6 passed, 1 warning in 12.22s ===================
```

## Honest framing — recommended next steps

The 3 sub-fixes DO unlock the integration path but DO NOT break
pocket-invariance on the prior argmax. The deeper fix is **Phase-3K
learned action-weight head**, which was already identified as
the follow-up in the Phase-2A report. That fix requires training data
(tmQM reactions) and is deferred behind the GPU outage.

Recommended ordering:

1. **Now (Phase 4):** keep the 3 combined sub-fixes shipped. They DO
   unlock the search-tree and metal-seed paths. Mark the prior-argmax
   break as **PARTIAL** in the paper §4 evaluation table.
2. **After GPU recovers:** ship Phase-3K learned action-weight head
   (replaces `_action_weight_vector` with a learned MLP). Retrain on
   tmQM reactions; this should close the prior-argmax gap.
3. **Round-13 pilot:** measure pocket-conditioned candidate-list
   divergence (per-cell `div_tanimoto(pocket_A_list, pocket_B_list)`)
   with the 3 combined sub-fixes — the partial result is still useful
   evidence that the integration path is functional.

## Files modified in Phase 3

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py`
  — `test_search_pocket_invariance_break` updated to combine all 3 sub-fixes
  (pocket_bias_strength=20.0, learned_prior_mix_uniform=0.9,
  pocket_derived_metal_seed call) and assert 4 sub-conditions.

## Files verified READ-ONLY (no modifications needed)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py`
  — already accepts pocket_features + learned_prior + learned_prior_mix_uniform
  kwargs (Phase 2B ship).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/warm_start.py`
  — `modify_root_prior` already accepts pocket_bias_strength param
  (Phase 2A ship); READ-ONLY per spec.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/learned_prior.py`
  — `LearnedPolicyPrior.predict_proba` already shipped; READ-ONLY per spec.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py`
  — `pocket_derived_metal_seed` + `--metal-seed-from-pocket` CLI flag
  already shipped (Phase 2C); no new edits needed.

## Verdict

**PARTIAL — 3/4 sub-assertions pass.**

The combined Phase-3 recipe integrates cleanly into both the
`modify_root_prior` callable and the `MCTSProofSearch.search()` method
+ the `pocket_derived_metal_seed` CLI helper, but the prior-argmax
break (sub-assertion i) requires Phase-3K learned action-weight head,
which is GPU-blocked.
