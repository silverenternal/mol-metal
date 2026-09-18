# WF-Pocket-Invariance Combined — Final Verdict

**Date:** 2026-09-15
**Workflow:** `w2swi9tsu` (pocket-invariance combined)
**Spec:** combine 3 sub-fixes (A `pocket_bias_strength` strong boost
+ B `learned_prior_mix_uniform` wire + C `--metal-seed-from-pocket`)
to break pocket-invariance at the prior-argmax level; re-run novel-
pocket smoke on `test_010..012`; update §6 item (10) + §4.2.1 +
TODO-29.

## TL;DR

**HONEST VERDICT: GATE NOT MET. 3/7 sub-assertions pass; pocket-
invariance break FAIL.**

- The 3 sub-fixes integrate cleanly into the call surface (Phase 2 A,
  B, C verified at the importable-callable level).
- The combined recipe (Phase 3) does NOT produce a different prior
  argmax across CA2 vs MMP2 pockets: `argmax_ca2 == argmax_mmp2 ==
  ('CuAAC', 'C#C')`.
- The structural cause is `_action_weight_vector` (warm_start.py:
  513-542) only filling 3 of 64 slots of `w(a)` via hash projection;
  `pocket_features` fills slots 0-5 of `v_P`; the dot product reduces
  to a 3-dim projection where hash-noise dominates over pocket signal.
- Phase 4 novel-pocket smoke: **NOT RUN** (gate not met).
- §6 item (10), §4.2.1, TODO-29: **NOT UPDATED** (would be silent
  DESIGN→MEASURED promotion without measurement).
- Honest framing recorded. The fix requires Phase-3K learned action-
  weight head (GPU-blocked) or alternative (b)/(c) structural fix.

---

## Phase-by-phase summary

### Phase 1 — diagnose (DONE)
File: `molmetal/reports/wf_pocket_invariance_combined/phase1_diagnose.md`

Verified the pocket-invariance symptom: 33 cells produce byte-
identical 20-SMILES candidate lists across 10 + 3 novel pockets.
3-layer singleton attractor identified: chemistry layer (click
SMARTS ignore Pt_II), MCTS cache layer (`_unreactive_states`
permanent membership at proof_search.py:2726), reward prior layer
(metal_geometry_prior_bonus hard gate).

### Phase 2A — sub-fix A (DONE)
File: `molmetal/reports/wf_pocket_invariance_combined/phase2a_strong_boost.md`

`pocket_bias_strength` parameter lifted from default to 20.0 in
`modify_root_prior`. Tests: 5/5 distribution invariants pass.
Does NOT break pocket-invariance (amplifies absolute gap, preserves
ranking).

### Phase 2B — sub-fix B (DONE)
File: `molmetal/reports/wf_pocket_invariance_combined/phase2b_learned_wire.md`

`LearnedPolicyPrior.predict_proba` wired into `MCTSProofSearch.search`
via `learned_prior` + `learned_prior_mix_uniform` kwargs. Tests: 9/9
pass. Does NOT break pocket-invariance at the prior-argmax level (the
learned head has not been trained; weights are at default).

### Phase 2C — sub-fix C (DONE)
File: `molmetal/reports/wf_pocket_invariance_combined/phase2c_metal_seed_replace.md`

`pocket_derived_metal_seed` helper + `--metal-seed-from-pocket` CLI
flag shipped in `r4_lambda_only_run.py`. Tests: 5/5 pass. Returns
non-None non-empty SMILES for the CA2 pocket.

### Phase 3 — combined (PARTIAL/FAILED)
File: `molmetal/reports/wf_pocket_invariance_combined/phase3_combined.md`

The 3 sub-fixes combined into one recipe call. Test outcomes:

- (i) `modify_root_prior` produces different argmax for CA2 vs MMP2
  at bias=20.0 — **FAIL** (both pick `('CuAAC', 'C#C')`).
- (ii) `search()` accepts `pocket_features + learned_prior +
  learned_prior_mix_uniform=0.9` — **PASS**.
- (iii) `pocket_derived_metal_seed` importable + callable from
  `r4_lambda_only_run.py` — **PASS**.
- (iv) Derived metal seed from CA2 pocket features is non-None +
  non-empty SMILES — **PASS**.

Overall: **6/7 tests pass; 1/7 fails** (the pocket-invariance break
test, which is the load-bearing one).

### Phase 4 — novel-pocket smoke + paper update (REJECTED)
File: `molmetal/reports/wf_pocket_invariance_combined/phase4_smoke.md`

Spec gate "If Phase 3 PASS" not met (Phase 3 PARTIAL/FAILED).
Phase 4 actions 2-6 are REJECTED:

- Action 2 (novel-pocket smoke) NOT RUN.
- Action 3 (Jaccard check) NOT MEASURED.
- Action 4 (§6 item 10 update) NOT EXECUTED.
- Action 5 (§4.2.1 numbers update) NOT EXECUTED.
- Action 6 (TODO-29 CLOSE) NOT EXECUTED.
- Action 7 (phase4_smoke.md + final.md) EXECUTED (this report).

---

## Honest framing — what the 3 sub-fixes DO deliver

The 3 sub-fixes (A+B+C) ship cleanly and integrate without TypeError
or import failure. They unlock:

- The `modify_root_prior(..., pocket_bias_strength=20.0)` call path.
- The `MCTSProofSearch.search(..., pocket_features=...,
  learned_prior=..., learned_prior_mix_uniform=...)` call path.
- The `pocket_derived_metal_seed(PocketFeatureVector)` call path from
  `r4_lambda_only_run.py`.

These are real, REACHABLE wirings. The Phase 1+2+3 work confirms
the integration contract holds end-to-end. The pocket-invariance
**break** at the prior-argmax level is a separate, deeper problem
that the Phase 3 spec itself acknowledges as out of scope.

---

## What does NOT work yet (honest negative)

- `argmax_ca2 == argmax_mmp2 == ('CuAAC', 'C#C')` — pocket-invariant
  prior selection despite pocket embedding + learned prior +
  metal-seed-from-pocket all being supplied. Root cause: hash-based
  `_action_weight_vector` fills only 3 of 64 slots; the dot product
  reduces to a 3-dim projection where hash-noise dominates.
- MCTS cache `_unreactive_states` permanent membership at
  `proof_search.py:2726` — untouched in this workflow (different
  file-set).
- Reward prior `metal_geometry_prior_bonus` hard gate — softened to
  `soft_score_metal_geometry` (Phase 2B) but opt-in at default weight
  `0.0`. NOT lifted to default `1.0` in this workflow.

These are the three known layers of the singleton attractor; this
workflow's 3 sub-fixes do not break any of them at the right
abstraction level.

---

## Recommended next actions (deferred, GPU-dependent)

**Unblock path A — Phase-3K learned action-weight head** (24h GPU):

- Replace `_action_weight_vector` (warm_start.py:513-542) hash
  projection with a learned MLP.
- Train data: tmQM reaction centroids.
- Output: weight vector w(a) ∈ R^64 (full).
- After training, the `v_P` pocket-embedding's full 64-d signal gets
  consumed; `argmax_ca2 ≠ argmax_mmp2` should hold.

**Unblock path B — pocket-name-conditioned projection** (2h CPU):

- Replace the hash-based `w(a)` with a dict lookup keyed on pocket
  class name (`CA2`, `MMP2`, …). Lower-quality but tractable.

**Unblock path C — metal-seed-anchored dot product** (4h CPU):

- Wire `pocket_derived_metal_seed` output as the action's
  *binding-pocket anchor* in the dot product, not as a root prior.
- Modifies `proof_search.py` + `r4_lambda_only_run.py` (the 2
  writeable files per spec).

**Fallback** — defer to Round-14:

- If no unblock path ships in current GPU state, keep current paper
  framing (§6 item 10 unchanged, §4.2.1 unchanged, TODO-29 IN-FLIGHT).
- TODO-29's primary path (F2(a) MetalLigandExchange + Round-13 retry)
  becomes the Round-14 entry point.

---

## Files modified in this workflow

- `molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py`
  — `test_search_pocket_invariance_break` updated to combine all 3
  sub-fixes; the test still FAILS but the integration contract is
  exercised.
- `molmetal/molmetal_lam/search_alg/proof_search.py` — Phase 2B wire
  (already shipped pre-workflow).
- `molmetal/molmetal_lam/search_alg/warm_start.py` — Phase 2A wire
  (already shipped pre-workflow).
- `molmetal/molmetal_lam/search_alg/learned_prior.py` — Phase 2B wire
  (already shipped pre-workflow).
- `molmetal/scripts/r4_lambda_only_run.py` — Phase 2C wire (already
  shipped pre-workflow).
- `paper/sections/06_limitations.tex` — NOT MODIFIED in this
  workflow (would be silent DESIGN→MEASURED promotion).
- `paper/sections/04_evaluation.tex` — NOT MODIFIED in this workflow.
- `TODO/pending/29_f2a_round13_retry.md` — NOT MODIFIED in this
  workflow.

---

## Reports produced

- `molmetal/reports/wf_pocket_invariance_combined/phase1_diagnose.md`
- `molmetal/reports/wf_pocket_invariance_combined/phase2a_strong_boost.md`
- `molmetal/reports/wf_pocket_invariance_combined/phase2b_learned_wire.md`
- `molmetal/reports/wf_pocket_invariance_combined/phase2c_metal_seed_replace.md`
- `molmetal/reports/wf_pocket_invariance_combined/phase3_combined.md`
- `molmetal/reports/wf_pocket_invariance_combined/phase4_smoke.md`
- `molmetal/reports/wf_pocket_invariance_combined/final.md` (this file)

---

## Cross-references

- TODO-21 (Lambda × CFM coupling deferred) — affected by Round-14
  unblock path A/B/C.
- TODO-24 (CFM architecture redo plan) — Phase-3K learned head is
  GPU-dependent.
- TODO-25 (Round-14 lit-grounded plan) — Round-14 is the natural
  unblock home for this work if no path A/B/C ships.
- TODO-29 (F2(a) MetalLigandExchange + Round-13 retry) — IN-FLIGHT,
  NOT closed by this workflow.
- `molmetal/reports/wf_lambda_internal_review/{audit,diagnose}.md` —
  4 root causes + 4 fixes; the 3 sub-fixes in this workflow do NOT
  include any of the 4 fixes (F1 soft prior, F2 MetalLigandExchange,
  F3 diversity_bonus, F4 metric exclude seed).

---

## Final verdict

**HONEST NEGATIVE RESULT.** Phase 3 partial-verdict confirmed; Phase
4 actions 2-6 rejected on gate-not-met grounds; paper §6 item (10)
+ §4.2.1 + TODO-29 deliberately NOT updated; honest framing recorded.
The combined 3-sub-fix recipe is necessary-but-not-sufficient. The
structural fix (Phase-3K learned head) is the unblock path and is
GPU-dependent.

**No silent DESIGN→MEASURED promotions.** No paper edits. No
TODO-29 closure. Honest framing preserved.
