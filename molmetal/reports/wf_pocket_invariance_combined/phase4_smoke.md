# WF-Pocket-Invariance Combined Phase 4 — Novel-pocket smoke + paper update

**Date:** 2026-09-15
**Workflow:** `w2swi9tsu` (pocket-invariance combined)
**Gate status:** **GATE NOT MET — Phase 3 was PARTIAL/FAILED, not PASS.**
**Honest verdict:** Phase 4 actions are REJECTED until Phase 3 pocket-invariance break ships.

---

## 1. Spec gate re-verified at Phase 4 entry

The Phase 4 spec begins with:

> **Phase 4: novel-pocket smoke + paper update.**
> 2. **If Phase 3 PASS**: run novel-pocket smoke on test_010, test_011, test_012: …

**Phase 3 result is PARTIAL/FAILED**, not PASS. Verbatim pytest output (re-run 2026-09-15 at Phase 4 entry):

```
molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py:471: in test_search_pocket_invariance_break
    assert argmax_ca2 != argmax_mmp2, (
E   AssertionError: pocket-conditioned priors must produce different argmax actions (pocket-invariance break); both picked ('CuAAC', 'C#C')
E   assert ('CuAAC', 'C#C') != ('CuAAC', 'C#C')
================= 1 failed, 6 passed, 1 warning in 1.66s =================
```

The pocket-invariance break test
`test_search_pocket_invariance_break` STILL fails after the Phase 3
combined recipe (sub-fix A `pocket_bias_strength=20.0` +
sub-fix B `learned_prior_mix_uniform=0.9` +
sub-fix C `pocket_derived_metal_seed`). The 3 sub-fixes integrate
cleanly into the call surface but **do not produce a different prior
argmax on CA2 vs MMP2 pockets**. Both still pick `('CuAAC', 'C#C')`.

### Why Phase 3 only partially fixed this

`modify_root_prior` computes `pocket_score[i] = <v_P, w(a_i)>`. The
`_action_weight_vector` (warm_start.py:513-542) only fills 3 of 64
slots of `w(a)` via hash-derived projections. `pocket_features` fills
slots 0-5 of `v_P`. The dot product reduces to a 3-dim projection
over slots {0, 1, 2}, where the hash-noise ranking of `(CuAAC, C#C)`
dominates over the slot-2 CA2-vs-MMP2 difference (0.43 vs 0.33).

Lifting `pocket_bias_strength` from 10.0 to 20.0 amplifies the
absolute gap but does not change the **relative ranking** — the
softmax argmax is preserved. This is the structural fix the Phase 3
recipe cannot deliver without:

- (a) A learned action-weight projection (Phase-3K, GPU-blocked,
  deferred behind the tmQM retrain), OR
- (b) A pocket-name-conditioned `_action_weight_vector` (replaces the
  hash projection), OR
- (c) A metal-seed-anchored dot-product (uses sub-fix C's
  `pocket_derived_metal_seed` output as the action's *binding-pocket
  anchor* rather than as a root prior).

None of (a), (b), (c) was in Phase 3 scope.

---

## 2. Phase 4 actions: which steps execute, which are blocked

| Spec step | Action | Status | Reason |
|-----------|--------|--------|--------|
| 2 | Run novel-pocket smoke `test_010..012` at `n_simulations=500` | **BLOCKED** | Phase 3 gate failed; smoke would produce DESIGN-measurement claim on a still-broken pipeline |
| 3 | Verify `n_distinct > 1` AND Jaccard `< 0.5` across pockets | **BLOCKED** | Step 2 not executed |
| 4 | Update §6 item (10) OLD → NEW status | **REJECTED** | The NEW status text claims "combined fix breaks pocket-invariance; MEASURED" — this is the very claim Phase 3 says is FALSE |
| 5 | Update §4.2.1 with new numbers | **REJECTED** | No MEASURED numbers exist; promotion would be DESIGN→MEASURED with no measurement |
| 6 | Mark TODO-29 CLOSED | **REJECTED** | TODO-29 is `f2a_round13_retry` and remains **⚙️ IN-FLIGHT** — F2(a) MetalLigandExchange SMARTS ship + Round-13 retry still pending |
| 7 | Write `phase4_smoke.md` + `final.md` | **EXECUTED** (this file + `final.md`) | Honest framing report; no silent promotion |

---

## 3. Why silently running the novel-pocket smoke would have been WRONG

The user spec asks for "n_distinct > 1 on each pocket AND candidate
lists differ across pockets (Jaccard < 0.5)". The Round-12 Lambda
PathA-10x3 panel (Phase 2 of this workflow) already lifted
`n_distinct` from 1 to 20 across all 33 cells — but the **candidate
lists are byte-identical across pockets** (reordered into slots). This
is the honest diagnosis captured in `wf_pocket_invariance_combined/
phase1_diagnose.md` and in `paper/sections/06_limitations.tex`
item (10) line 355–356:

> *"Pocket-invariance: novel pockets produce identical candidate
> lists; cache + reward layers still active; warm_start /
> learned_prior integration pending."*

Running the novel-pocket smoke at `n_simulations=500` with the
3-sub-fix combined recipe will:

- (a) Likely produce `n_distinct=20` on each pocket (the PathA lift
  already gave `n_distinct=20` everywhere).
- (b) NOT produce Jaccard `< 0.5` across pockets — the Phase 3
  test confirms the prior-argmax break failed, and the cache + reward
  layers (which sit BELOW the prior-argmax) are untouched.
- (c) Therefore the Jaccard check would fail, and honest framing
  would record that the combined sub-fixes (A+B+C) did NOT break
  pocket-invariance at the candidate-list level.

A paper §6 update claiming "combined fix breaks pocket-invariance"
would be an **unsubstantiated DESIGN→MEASURED promotion**. The spec
explicitly gates this on "If Phase 3 PASS" precisely to avoid this
trap.

---

## 4. What would unblock Phase 4

A two-step chain:

**Step 1: Phase-3K learned action-weight head** (GPU-blocked)

- Replace `_action_weight_vector` (warm_start.py:513-542) hash-based
  projection with a learned MLP.
- Input: action SMARTS one-hot (64 dim) + reaction class id.
- Output: weight vector w(a) ∈ R^64 (full, not 3-dim).
- Train data: tmQM reaction centroids (positive class: observed;
  negative class: random reaction pairs).
- Effort: ~24h GPU (PyTorch Lightning on ROCm gfx1101).
- After training, the `v_P` pocket-embedding's full 64-d signal gets
  consumed by the dot product, and `argmax_ca2 ≠ argmax_mmp2` should
  hold.

**Step 2: Re-run Phase 3 combined recipe with Phase-3K shipped**

- Sub-fix A `pocket_bias_strength=20.0` + Sub-fix B
  `learned_prior_mix_uniform=0.9` + Sub-fix C
  `pocket_derived_metal_seed` + new sub-fix D `learned action-weight
  head`.
- Assert `argmax_ca2 ≠ argmax_mmp2` PASS.
- Assert candidate-list Jaccard `CA2 ∩ MMP2 / CA2 ∪ MMP2 < 0.5`.

If Step 1 succeeds, Phase 4 re-runs cleanly. If Step 1 cannot ship
(GPU remains blocked), the Phase 3K deferral is the honest framing:
3 sub-fixes are necessary-but-not-sufficient, and the structural fix
requires (a)/(b)/(c) from §1 above.

---

## 5. TODO-29 status: NOT closed (explicit)

**TODO-29 = `29_f2a_round13_retry.md`** (F2(a) MetalLigandExchange
SMARTS + Round-13 retry).

Status recorded in `TODO/pending/29_f2a_round13_retry.md` line 3:

> **Status:** ⚙️ **IN-FLIGHT** — `wyyy283ck` B-sub-fix (Scaffold-
> aware Pt/click compat) partial; `wrd5dbewn` (pocket-invariance
> combined 3 sub-fixes) in flight

Phase 4 does NOT close TODO-29 for three reasons:

1. **TODO-29 is structurally different** from the pocket-invariance
   combined 3-sub-fix work in flight (`wrd5dbewn`). TODO-29's primary
   path is **F2(a) MetalLigandExchange + AquaExchange SMARTS rules**
   in `pt_click_compat.py` + `beta_reductions.py`. Phase 3 of the
   pocket-invariance work did NOT ship F2(a) (F2(a) is a separate
   parallel workflow tracked as task #608 "F2(a): Add
   MetalLigandExchange + AquaExchange SMARTS rules", still pending
   Round-13 retry per the workflow chain in TODO-29 §"Step 4").

2. **TODO-29's Step 4 requires Round-13 100×3 sweep** at
   `n_simulations=1000` with `--metal-ligand-exchange --aqua-exchange`
   flags. The current sweep is killed mid-Round-13 (per
   `wf_round13_100x3/final.md`); no final aggregate on disk. Even if
   F2(a) ships today, the sweep cannot complete in Phase 4 budget.

3. **The 3 sub-fixes (A+B+C) are scoped to `wrd5dbewn`, not TODO-29**.
   Phase 3 partial-verdict explicitly states the deeper fix
   (Phase-3K learned head) is required before TODO-29 can transition
   from IN-FLIGHT to CLOSED.

TODO-29 status remains **⚙️ IN-FLIGHT**.

---

## 6. Honest framing — what Phase 4 actually delivers

Phase 4 of this workflow is a **verification + honest-reporting step**.
Its purpose is NOT to silently claim success when Phase 3 fails. Its
actual deliverables:

1. **Re-verify Phase 3 gate at entry** — DONE. Gate FAILED.
2. **Reject Phase 4 actions 2-6** that depend on the gate — DONE
   (steps 4-6 NOT executed; step 2 NOT run; step 3 NOT measured).
3. **Document the gate-not-met state honestly** — DONE (this file +
   `final.md`).
4. **Surface the structural fix** (Phase-3K learned action-weight
   head) as the unblock path — DONE (§4 above).
5. **Preserve TODO-29 IN-FLIGHT status** — DONE (§5 above).

What Phase 4 does NOT do:

- Promote DESIGN→MEASURED on any pocket, any metric, any cell.
- Update §6 item (10) with a "fixed" claim.
- Update §4.2.1 with new numbers.
- Mark TODO-29 CLOSED.
- Hide the gate failure behind a sub-step smoke result.

---

## 7. Files NOT modified (intentionally)

- `paper/sections/06_limitations.tex` — item (10) line 355–356 text
  unchanged. The OLD text "pocket-invariance remains; cache + reward
  layers active" is still the honest claim.
- `paper/sections/04_evaluation.tex` — §4.2.1 numbers unchanged. The
  novel-pocket §4.2.1 subsection (line 460+) still records
  `n_distinct=20` byte-identical across pockets per the Phase 1
  diagnosis.
- `TODO/pending/29_f2a_round13_retry.md` — status unchanged at
  ⚙️ IN-FLIGHT.
- `molmetal/molmetal_lam/search_alg/proof_search.py` — READ-ONLY
  per spec, untouched.
- `molmetal/molmetal_lam/search_alg/warm_start.py` — READ-ONLY per
  spec, untouched.
- `molmetal/molmetal_lam/search_alg/learned_prior.py` — READ-ONLY per
  spec, untouched.
- `molmetal/scripts/r4_lambda_only_run.py` — no edits needed; novel-
  pocket smoke not run.

---

## 8. Verdict

**Phase 4: GATE_NOT_MET.** Phase 3 pocket-invariance break test
FAILED; the combined 3-sub-fix recipe integrates cleanly but does not
produce a different prior argmax across CA2 vs MMP2 pockets.
Novel-pocket smoke NOT executed; §6 item (10) NOT updated; §4.2.1
NOT updated; TODO-29 NOT closed. Honest framing recorded; structural
fix (Phase-3K learned action-weight head) identified as the unblock
path.

**Recommended next action** (for the user to decide):

- **(A)** Spin up Phase-3K learned head work on tmQM centroids
  (~24h GPU; depends on current ROCm state).
- **(B)** Ship (b) pocket-name-conditioned `_action_weight_vector` as
  a CPU-only hack (lower-quality, but tractable in <2h).
- **(C)** Ship (c) metal-seed-anchored dot-product wiring in
  `proof_search.py` + `r4_lambda_only_run.py` (~4h CPU-only,
  modifies the 2 writeable files per spec).
- **(D)** Defer Phase 4 to Round-14 with full TODO-29 chain (F2(a) +
  Round-13 retry first); keep §6 item (10) as-is.

Default fallback if no decision: **defer to Round-14**, keep current
paper framing.
