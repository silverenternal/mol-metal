# Wf-345A2 — `p3_wf343c4_yuelbond_decoder_env` env-aware SKIP contract

**Status:** SHIPPED (CPU-only)
**Date:** 2026-09-29
**Workflow:** Wf-345 ultracode (user directive "覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长")
**Sub-flow:** Wf-345-A2 (gate env-aware SKIP contract)
**Predecessors:** Wf-343C4 (gate-builder; 4 NEW pre-flight gates added per `TODO/completed/343c4_integration.md`)
**Tests:** 3 NEW in `molmetal/tests/test_wf345a2_yuelbond_env.py` + 1 UPDATED in `molmetal/tests/test_wf343c4_integration.py`

---

## §1 Problem statement

Per `wx63q8fwe` ultracode review: the pre-flight gate
`p3_wf343c4_yuelbond_decoder_registered` at
`molmetal/scripts/pre_flight_check.py:3072` (Wf-343C4) fails on the
bare host Python (``/usr/bin/python``) because the canonical
``cfm_decoder_yuelbond.py`` ``import torch`` /
``import torch.nn as nn`` at module top-level (line 59-60) and the
host Python lacks torch. The Wf-272I structural check assumed
``kanzi_venv`` is the host Python.

MEASURED on the host:

```text
$ /usr/bin/python -c "import torch"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

The pre-flight gate's `exec_module()` then raised a generic
``Exception`` (caught by the Wf-343C4 ``except Exception`` clause)
and the gate returned ``FAIL`` with ``decoder module import failed:
ModuleNotFoundError: No module named 'torch'``. This is an env gap, not
a regression — the on-disk module is intact, syntactically loadable,
and structurally correct.

The fix (Wf-345A2):
  1. **Rename** ``p3_wf343c4_yuelbond_decoder_registered`` →
     ``p3_wf343c4_yuelbond_decoder_env`` (more accurate: the gate
     verifies the decoder-in-env contract, not just "registered").
  2. **Wrap** ``(ModuleNotFoundError, ImportError)`` in a try/except
     that emits SKIP rather than FAIL, naming the missing dep + the
     expected runtime per AGENTS.md §5.1 standing GPU grant.

---

## §2 Honest framing (party-twenty-spirit §3.5)

| Claim | Verdict |
|---|---|
| Gate FAILs on host Python without torch | **MEASURED** — confirmed via ``/usr/bin/python -c 'import torch'`` → ModuleNotFoundError |
| Gate FAILs in any real version | **MEASURED** — confirmed on pre-fix `pre_flight_check.py:3072` via `pre_flight_check.py --pipeline all` |
| env-aware SKIP branch is the correct contract | **PROJECTED** (per AGENTS.md §5.2 honest framing + party-twenty-spirit §3.5) |
| Gate rename improves accuracy | **PROJECTED** — name now reflects the env-dependency contract |
| No GPU launches performed | **MEASURED** — CPU-only docs + code |
| Wall | **PROJECTED 15-25 min** (CPU-only docs + 3 NEW tests + 1 EDIT on gate function + 1 EDIT on gate registration + 1 EDIT on the existing test's reference + 1 EDIT on the test_wf343c4_integration.py's gate-name assertion) |

---

## §3 Changes

### §3.1 `molmetal/scripts/pre_flight_check.py`

Two EDITS:

**EDIT 1: Rename gate function + add env-aware SKIP contract.**

```text
- def gate_wf343c4_yuelbond_decoder_registered(
+ def gate_wf343c4_yuelbond_decoder_env(
```

The new gate function:
  * Has identical PASS / FAIL contracts for the
    torch-enabled case (preserves the Wf-343C4 regression test).
  * Catches ``(ModuleNotFoundError, ImportError)`` separately from
    the catch-all ``except Exception`` and emits SKIP rather than
    FAIL when the decoder's top-level ``import torch`` cannot be
    resolved.
  * The SKIP evidence names the missing dep + the expected runtime
    per AGENTS.md §5.1:
    - either ``kanzi_venv`` (the canonical CFM-retrain venv)
    - or ``/home/hugo/codes/CadButEaas/.venv-vlm/bin/python``
      (the observed torch-enabled runtime).
  * On a torch-enabled Python, the canonical decoder loads and the
    Wf-272I / Wf-269C structural check runs as before — emits PASS
    when ``YuelBondDecoder`` is exported, FAIL when the module is
    missing / empty / doesn't export the class.

**EDIT 2: Rename gate registration string.**

```text
- "p3_wf343c4_yuelbond_decoder_registered"
- lambda: gate_wf343c4_yuelbond_decoder_registered(
-     "p3_wf343c4_yuelbond_decoder_registered",
+ "p3_wf343c4_yuelbond_decoder_env"
+ lambda: gate_wf343c4_yuelbond_decoder_env(
+     "p3_wf343c4_yuelbond_decoder_env",
```

The total gate count (`EXPECTED_GATE_COUNTS["all"]`) is preserved
at the pre-Wf-345A2 baseline (68 = 62 pre-Wf-343C4 baseline + 4 NEW
Wf-343C4 gates + 2 NEW Wf-343C2/C3 per-wave + data-quality gates).
No count drift from the rename.

### §3.2 `molmetal/tests/test_wf343c4_integration.py`

Two EDITS (rename propagation + closure-bracketed PASS-or-SKIP
contract):

**EDIT 1: Update `test_gate_wf343c4_yuelbond_decoder_registered_on_canonical` →
`test_gate_wf343c4_yuelbond_decoder_env_on_canonical`.**

The existing test's PASS branch now asserts PASS-or-SKIP (rather
than just PASS) because:
  * On a torch-enabled runtime: PASS (regression test for
    Wf-272I / Wf-269C structural check).
  * On a torch-less host Python: SKIP (the Wf-345A2 env gap
    contract).

The stub-FAIL branch (stub without YuelBondDecoder class) is
preserved unchanged — it confirms the gate's FAIL detection works
as designed (a future commit that removes the YuelBond class would
still be caught at the pre-flight gate layer).

**EDIT 2: Update the gate-name assertion in
`test_gate_wf343c4_wave_budget_remaining_nonneg_cross_script` →
`p3_wf343c4_yuelbond_decoder_env`** (rename propagated to the
Wf-321D audit invariant assertion).

### §3.3 `molmetal/tests/test_wf345a2_yuelbond_env.py` (NEW)

3 NEW tests covering the Wf-345A2 contract:

1. **`test_gate_yuelbond_decoder_env_pass_when_torch_available`** —
   Writes a tmp_path stub that defines ``YuelBondDecoder`` class
   WITHOUT importing torch (no torch dep). Asserts the gate returns
   ``status == "PASS"`` with ``YuelBondDecoder`` in the evidence
   string.

2. **`test_gate_yuelbond_decoder_env_skip_when_torch_missing`** —
   Writes a tmp_path stub that raises
   ``ModuleNotFoundError("No module named 'torch'")`` at module
   level — the exact exception the canonical decoder would raise on
   a host Python without torch. Asserts the gate returns
   ``status == "SKIP"`` (NOT FAIL) with ``"torch"`` in the evidence
   + a pointer to the expected runtime (kanzi_venv or venv-vlm).

   Why this approach (and not mocking torch in sys.modules): popping
   torch from ``sys.modules`` leaves its C extensions in an
   inconsistent state — subsequent tests segfault on
   ``torch.cuda.is_available()`` (MEASURED during Wf-345A2
   development). The exception-as-stub approach exercises the same
   gate behavior, without any C-state corruption.

3. **`test_gate_yuelbond_decoder_env_gate_name_propagates`** —
   Verifies the rename is fully propagated:
     * ``gate_wf343c4_yuelbond_decoder_env`` exposed on
       pre_flight_check module.
     * Old ``gate_wf343c4_yuelbond_decoder_registered`` NOT exposed.
     * ``"p3_wf343c4_yuelbond_decoder_env"`` registered in
       ``build_gates("all")``.
     * Old ``"p3_wf343c4_yuelbond_decoder_registered"`` NOT
       registered in any pipeline.
     * Total ``EXPECTED_GATE_COUNTS["all"]`` unchanged at 68.
     * The runner for the renamed gate is registered + callable.

### §3.4 Documentation updates

  * The gate function's docstring was updated to document the
    Wf-345A2 env-aware SKIP contract (the ``module-level torch
    dep`` rationale + the expected runtime per AGENTS.md §5.1).
  * The existing `molmetal/scripts/pre_flight_check.py:2794`
    comment listing the 4 NEW Wf-343C4 gate names was updated to
    use the new name ``p3_wf343c4_yuelbond_decoder_env``.

---

## §4 Test results

```text
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf345a2_yuelbond_env.py -v
...
molmetal/tests/test_wf345a2_yuelbond_env.py::test_gate_yuelbond_decoder_env_pass_when_torch_available PASSED [ 33%]
molmetal/tests/test_wf345a2_yuelbond_env.py::test_gate_yuelbond_decoder_env_skip_when_torch_missing PASSED [ 66%]
molmetal/tests/test_wf345a2_yuelbond_env.py::test_gate_yuelbond_decoder_env_gate_name_propagates PASSED [100%]
========================= 3 passed, 1 warning in 0.36s =========================
```

```text
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf345a2_yuelbond_env.py molmetal/tests/test_wf343c4_integration.py -v
...
molmetal/tests/test_wf345a2_yuelbond_env.py::test_gate_yuelbond_decoder_env_pass_when_torch_available PASSED [ 14%]
molmetal/tests/test_wf345a2_yuelbond_env.py::test_gate_yuelbond_decoder_env_skip_when_torch_missing PASSED [ 28%]
molmetal/tests/test_wf345a2_yuelbond_env.py::test_gate_yuelbond_decoder_env_gate_name_propagates PASSED [ 42%]
molmetal/tests/test_wf343c4_integration.py::test_gate_wf343c4_hyperparams_config_complete_pass_on_fixture PASSED [ 57%]
molmetal/tests/test_wf343c4_integration.py::test_gate_wf343c4_yuelbond_decoder_env_on_canonical PASSED [ 71%]
molmetal/tests/test_wf343c4_integration.py::test_gate_wf343c4_anchor_freshness_24h_pass_on_fresh_fixture PASSED [ 85%]
molmetal/tests/test_wf343c4_integration.py::test_gate_wf343c4_wave_budget_remaining_nonneg_cross_script PASSED [100%]
========================= 7 passed, 1 warning in 9.95s =========================
```

**3 NEW tests PASS** + **4 EXISTING tests PASS** (no regression on
the Wf-343C4 integration suite). The env-aware SKIP contract is
verified end-to-end without any torch state corruption.

---

## §5 Cross-references

* `molmetal/scripts/pre_flight_check.py:3072` — the renamed +
  env-aware gate function ``gate_wf343c4_yuelbond_decoder_env``.
* `molmetal/scripts/pre_flight_check.py:4718-4721` — the renamed
  gate registration in ``build_gates("all")`` under the new name.
* `molmetal/tests/test_wf345a2_yuelbond_env.py` — the 3 NEW tests
  for the env-aware SKIP contract.
* `molmetal/tests/test_wf343c4_integration.py` — the 4 EXISTING
  tests (1 updated to use the renamed function + handle the
  PASS-or-SKIP contract).
* `TODO/completed/343c4_integration.md` — Wf-343C4 ship surface
  (predecessor to Wf-345A2; the original gate that this Wf-345A2
  makes env-aware).
* AGENTS.md §5.1 (standing GPU grant) — the env-aware SKIP evidence
  names the expected torch-enabled runtime per this directive.

---

## §6 Wall + tests accounting

| Item | Value |
|---|---|
| Wall (CPU-only) | PROJECTED 15-25 min; ACTUAL ~10 min (1 Edit on gate function + 1 Edit on gate registration + 1 Edit on the existing test reference + 1 Edit on the existing test's gate-name assertion + 1 NEW test file + 1 NEW closeout doc) |
| Tests | 3 NEW PASS + 4 EXISTING PASS (no regression) |
| GPU launches | 0 (CPU-only verification per §5.1 + AGENTS.md standing directives) |
| Pre-flight gate count delta | 0 (rename only; EXPECTED_GATE_COUNTS["all"] preserved at 68) |
| Honest-negative caveats | 0 (all 3 NEW tests PASS cleanly without env manipulation that could corrupt torch C-state) |

---

## §7 Remaining work / follow-ups

None for Wf-345A2 itself. The gate now emits SKIP (not FAIL) on the
host Python without torch; the operator can resolve the env gap by
running the pre-flight under the expected runtime per AGENTS.md §5.1
(``kanzi_venv`` or ``/home/hugo/codes/CadButEaas/.venv-vlm/bin/python``).

Historical references to the old gate name in
`TODO/completed/{343c2_per_wave_check.md,343d2.md,343d3.md,343c4_integration.md}`
+ `TODO/INDEX.md` §22.23 + AGENTS.md §6 are intentional (they
document past Wf-343C4 work under its original name). Future
docs-only sync passes (e.g., Wf-345-SYN) can update those if needed,
but per party-twenty-spirit §3.5 honest framing, historical
documentation should not be retroactively rewritten when it changes
the meaning of past actions.

---

**END OF Wf-345A2 SHIP SURFACE** — Owner Hugo Lee; cross-ref
`party-twenty-spirit` §3.5 + AGENTS.md §5.1 (standing GPU grant) +
AGENTS.md §5.2 (honest framing); last update 2026-09-29.