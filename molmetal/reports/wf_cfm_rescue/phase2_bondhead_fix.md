# WF-CFM-Rescue Phase 2 — bond_head joint_train default verify

**Date:** 2026-09-16
**Scope:** Verify (regression-guard) the Phase 2 Fix #1 bond-head default
flip shipped by WF-CFM-Frontier-Research.
**Status:** Phase 2 VERIFIED — joint_train=True default + bond_head in optimizer
end-to-end confirmed. Pre-existing tests at
``molmetal/tests/test_cfm_fix1_bond_head_default.py`` (5/5 pass) cover
the same surface; this module adds explicit rescue coverage.

---

## 1. What was already shipped (Phase 2 Fix #1, 2026-09-15)

| Surface | Old | New |
|---|---|---|
| `LipmanFlowMatchingAdapter(joint_train=)` default | `False` | **`True`** |
| `--joint-train` CLI default in r10_cfg_real_crossdocked | `False` | **`True`** |
| BondOrderHead in optimizer when joint_train=True | NO | **YES** |

Pre-existing tests (test_cfm_fix1_bond_head_default.py) cover:
- argparse defaults
- adapter constructor default
- optimizer wiring (bond_head.parameters() in optimizer.param_groups)
- legacy opt-out (joint_train=False still works)

## 2. New rescue tests (test_cfm_rescue_bondhead_default.py)

4 tests, all CPU, all pass:

```
$ uv run pytest molmetal/tests/test_cfm_rescue_bondhead_default.py --tb=short
molmetal/tests/test_cfm_rescue_bondhead_default.py .... [100%]
4 passed in 2.70s
```

Coverage:
- `test_joint_train_default_train_step_runs` — full train_step completes without exception
- `test_bond_head_receives_gradient_when_joint_train_true` — bond_head params get .grad
- `test_bond_head_no_gradient_when_joint_train_false` — backward-compat: joint_train=False freezes head
- `test_legacy_no_bond_head_works` — use_bond_head=False still constructs no head

## 3. Why the rescue layer?

Phase 2 Fix #1 shipped in WF-CFM-Frontier-Research Phase 2 (2026-09-15)
flipped the default; the existing tests verify the surface but don't
exercise the *training-loop* end-to-end path. The rescue tests:
1. Call `train_step(pocket=None, mols=mols)` (the real signature, not
   the legacy `positions=...` from P0-F1 test stubs).
2. Assert that after one step, at least one bond_head param has `.grad != None`.
3. Verify the negative path (joint_train=False keeps bond_head frozen).

## 4. Honest framing

The bond_head is **structurally** co-trained now. Whether it
**empirically** lifts decode_ratio depends on (a) the head's
representational capacity (K=64 hidden_dim is fine for 5-class output;
matches round-10 donor vocab), (b) the training data scale (Phase 3 fix
bumps n_train 8 → 32), and (c) the integration with the EGNN velocity
field (Phase 4 fixes the ODE solver).

The test suite verifies that the wiring is correct; the metric lift
requires a GPU retrain (TODO-24 §5 user-decision point).

## 5. Files written

- `molmetal/tests/test_cfm_rescue_bondhead_default.py` (4 tests)
- `molmetal/reports/wf_cfm_rescue/phase2_bondhead_fix.md` (this report)
