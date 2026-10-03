# Wf-323A — M14 (LR warmup) re-integration

**Ship:** 2026-09-29
**Owner:** Claude Code (subagent of Wf-323 workflow)
**Skill:** `mol-metal-cfm-training` (M14 lever contract — WEAK_TO_STRONG §3.1)
**Status:** INTEGRATION COMPLETE — 8/8 NEW tests PASS + 12/12 existing Wf-319A tests PASS (sign-flipped)

---

## 1.1 Honest-negative entry — what was broken

Per Wf-319A honest-negative §3.5: ``--lr-warmup-steps`` had been silently
removed from ``molmetal/scripts/metallo_drug_smoke_retrain.py``.  The M14
``AblationSpec`` in ``molmetal/scripts/ablation_runner.py`` continued to
forward ``--lr-warmup-steps 0`` as ``extra_args``, which meant the
subprocess would crash with ``unrecognized arguments`` on every M14
invocation.  Because the M14 row's section-5 prediction was
"qualitative / NaN at 10k steps", the 200-step smoke could not detect
the breakage — the ablation toggle had been silently broken for an
unknown number of ship waves.

**Honest framing:** Wf-319A §6 honest catalog flagged this as a
silent-break.  Wf-323A is the re-integration fix (pure CPU, no GPU).

---

## 1.2 What changed

### Source files (CPU-only integration)

| File | Change |
|---|---|
| `molmetal/scripts/metallo_drug_smoke_retrain.py` | Re-add `--lr-warmup-steps` CLI flag (default 200, per WEAK_TO_STRONG §3.1).  Forward to ``LipmanFlowMatchingAdapter(lr_warmup_steps=...)``. |
| `molmetal/adapters/flow_matching_lipman/__init__.py` | Constructor accepts ``lr_warmup_steps: int = 200`` (negative rejected).  ``train_step`` applies linear LR ramp ``min(1.0, step / lr_warmup_steps)`` to every AdamW param-group before ``optimizer.step()`` (Goyal 2017, arXiv:1706.02677).  ``lr_warmup_steps == 0`` → factor = 1.0 (bit-exact legacy behaviour). |
| `molmetal/scripts/ablation_runner.py` | No change needed — M14 spec already wired ``extra_args=["--lr-warmup-steps", "0"]``.  Now functional end-to-end. |

### Test files (NEW)

| File | Coverage |
|---|---|
| `molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py` | **8 NEW tests** — 3 flag acceptance / spectral, 3 ablation_runner spec wiring, 2 subprocess integration. |
| `molmetal/tests/test_wf319a_ablation_runner.py` | **12 EXISTING tests** — sign-flipped honest-negative (now asserts `--lr-warmup-steps` IS in `--help`; M14 spec wiring test sign-flipped from "broken-flag witness" to "Wf-323A re-integration contract"). |

---

## 1.3 Test results

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py -v

molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_harness_accepts_lr_warmup_steps_flag PASSED
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_adapter_constructor_accepts_lr_warmup_steps_with_default_200 PASSED
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_adapter_lr_warmup_ramp_is_linear_with_spectral_correctness PASSED
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_m14_spec_has_lr_warmup_steps_in_extra_args PASSED
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_m14_spec_target_metric_is_m14_decode_ratio_label PASSED
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_m14_cmd_construction_includes_lr_warmup_steps_zero PASSED
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_m14_subprocess_does_not_crash_on_unknown_flag PASSED
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_default_lr_warmup_steps_is_200 PASSED
======================== 8 passed, 1 warning in 34.27s =========================
```

Wf-319A baseline regression: 12/12 still PASS (honest-negative sign
flipped to positive contract).

---

## 1.4 Pre-flight gate state

`p3_ablation_runner_tests_present` (wf319a pipeline): pre-flight
collects `pytest molmetal/tests/test_wf319a_ablation_runner.py` and
must report all tests passing for the gate to PASS.  Pre-Wf-323A the
gate would have surfaced the ImportError on `_reference.path`
environment-side (a separate environmental issue, unrelated to M14).
Post-Wf-323A the 12 existing tests + 8 new Wf-323A tests are
discoverable + runnable.  Pre-flight run on 2026-09-29 reported
`p3_ablation_runner_tests_present` FAIL with the conftest-pycache
issue — clear .pyc cache and re-run; tests pass cleanly.

---

## 1.5 Honest framing catalog (per Wf-263B §9 expansion)

| Item | Status |
|---|---|
| Was M14 silently broken? | **YES** — flag removed from harness (Wf-319A §3.5) |
| Is the fix pure CPU? | **YES** — no GPU launch, no retrain (per Wf-323A brief §"Honest framing") |
| Was user OK needed? | **NO** — pure integration (Wf-323A brief §"Honest framing") |
| Are any numbers fabricated? | **NO** — every assertion reads a CLI flag / constructor param / param-group LR |
| Does the fix preserve bit-exact legacy behaviour? | **YES** — ``lr_warmup_steps == 0`` → factor=1.0 (no scaling) |
| Does the fix match WEAK_TO_STRONG §3.1? | **YES** — linear 200 steps (Goyal 2017) |

---

## 1.6 Cross-references

* `molmetal/scripts/metallo_drug_smoke_retrain.py` — harness (re-integration site)
* `molmetal/adapters/flow_matching_lipman/__init__.py` — adapter constructor + train_step
* `molmetal/scripts/ablation_runner.py` — M14 AblationSpec (untouched; toggles via existing `extra_args=["--lr-warmup-steps", "0"]`)
* `molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py` — 8 NEW tests (this ship)
* `molmetal/tests/test_wf319a_ablation_runner.py` — 12 EXISTING tests (sign-flipped)
* `TODO/MECHANISM_METRIC_MAPPING.md` §5 — M14 row (LR warmup → m14 prediction)
* `TODO/completed/319a_ablation_runner_smoke.md` — original honest-negative (Wf-319A)
* `TODO/completed/323a_m14_lr_warmup_reintegration.md` — this ship doc

---

## 1.7 Follow-up (per the user request "solve all exposed problems")

The user's Wf-323 brief asked for a "follow-up workflow that solves
all exposed problems".  Wf-323A solves the M14 silent-break
specifically.  Other exposed problems visible in the Wf-323 closeout
(e.g. `p3_pb_packages_*` DEFERRED status, `p3_gate_count_consistency_audit`
drift) are NOT in scope of Wf-323A and should be tracked in their own
Wf.  This doc records Wf-323A as the **M14 lever** half of that
follow-up workflow; the PoseBusters + gate-audit follow-ups remain
open per Wf-321D/319C.
