# Wf-346-A4 — Wave Budget Gate Registration Verification (CPU-only)

**Date:** 2026-09-29
**Wave:** Wf-346-A4 (Wf-346 ultracode residual wave; A4 sub-flow)
**Status:** SHIPPED (verification + 2 NEW tests)
**Tag discipline:** party-twenty-spirit §3.5 (MEASURED vs PROJECTED)

---

## §1 Background

Per the 2026-09-29 ultracode user directive ("基于刚才的洞察，可以启动
ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长"),
Wf-346-A4 audits the registration of
``gate_wf343c4_wave_budget_remaining_nonneg`` in the
``build_gates('all')`` dispatcher. The Wf-343D1 finding hypothesised
that this gate might exhibit the same registration drift as the
hyperparams / yuelbond / anchor-freshness gates (Wf-346-A1..A3).

This Wf-346-A4 sub-flow produces the **MEASURED verification** +
**2 NEW tests** that pin the registration invariant.

---

## §2 Verification result (MEASURED)

**Source:** `molmetal/tests/test_wf346a4_wave_budget_registered.py`
(2 NEW tests, both PASS — wall 0.24 s).

| # | Test | MEASURED status |
|---|---|---|
| 1 | `test_wf343c4_wave_budget_gate_registered_in_dispatcher` | PASS |
| 2 | `test_build_gates_all_returns_66_with_four_wf343c4_gates` | PASS |

**Direct introspection** (`pre_flight_check.build_gates('all')`):

| Field | Value | MEASURED |
|---|---|---|
| `len(build_gates('all'))` | **66** | yes |
| `EXPECTED_GATE_COUNTS['all']` | **66** | yes |
| `MATCH` (audit invariant) | **True** | yes |
| `wf343c4_gates_count` | **4** (a + b + c + d) | yes |
| `p3_wf343c4_wave_budget_remaining_nonneg` in `all` | **True** | yes |
| `p3_wf343c4_wave_budget_remaining_nonneg` in `frontier` | **False** | yes |
| `len(build_gates('frontier'))` | **25** (unchanged) | yes |

**Honest framing:** the gate **IS already registered** at
`pre_flight_check.py:4731-4739` (per the Wf-343C4 ship summary at
`TODO/completed/343c4_integration.md`). The Wf-343D1 hypothesis of
"registration drift for wave budget gate" was **refuted by direct
introspection**. **No Edit** was applied to `pre_flight_check.py`
in this sub-flow — the gate registration was correct as of Wf-343C4
ship.

---

## §3 Pre-flight v3 gate count audit (Wf-321D invariant preserved)

Per Wf-321D `gate_gate_count_consistency_audit` (TODO/completed/321d_preflight_gate_audit.md),
the canonical 'all' pipeline count baseline is 66 (= 62 Wf-327-SYN
baseline + 4 NEW Wf-343C4 gates). This Wf-346-A4 sub-flow confirms
the invariant:

- `EXPECTED_GATE_COUNTS["all"] == 66` (line 2817-2822 of
  `pre_flight_check.py`)
- `build_gates("all")` returns exactly 66 runners (matches)
- All 4 Wf-343C4 gates present:
  - `p3_wf343c4_hyperparams_config_complete` (a — Wf-346-A1)
  - `p3_wf343c4_yuelbond_decoder_env` (b — Wf-346-A2)
  - `p3_wf343c4_anchor_freshness_24h` (c — Wf-346-A3)
  - `p3_wf343c4_wave_budget_remaining_nonneg` (d — Wf-346-A4,
    THIS sub-flow)
- Frontier pipeline count unchanged at 25 (the 4 NEW Wf-343C4 gates
  are 'all'-only by Wf-343C4 §3.4 design contract; pre-retrain
  internal checks are NOT paper-grade claims).

---

## §4 Why no Edit was needed

The Wf-346-A4 task brief suggested "registration drift" mirroring
Wf-346-A1..A3. Per MEASURED introspection (and the 2 NEW tests
above), the Wf-343C4 ship correctly registered all 4 NEW gates in
`build_gates('all')` at the Wf-343C4 integration close-out
(`TODO/completed/343c4_integration.md`). The audit invariant
(`EXPECTED_GATE_COUNTS["all"] == 66`) has been preserved since the
Wf-343C4 ship.

**Conclusion:** the gate is already registered. This sub-flow is a
**MEASURED verification + test pinning** — not a registration fix.

---

## §5 Wall budget (PROJECTED vs MEASURED)

| Phase | PROJECTED (per task brief) | MEASURED |
|---|---|---|
| Syntax check | < 30 s | **< 1 s** |
| Test design + write | 5-10 min | **~3 min** |
| Test execution | < 30 s | **0.24 s** (2 PASS) |
| Closeout doc | 5-10 min | **~2 min** |
| **Total** | **10-15 min CPU-only** | **~5 min** |

---

## §6 Honest framing (party-twenty-spirit §3.5)

- All numeric values trace to MEASURED introspection of
  `molmetal/scripts/pre_flight_check.py` (imported + `build_gates('all')`
  invoked) + the 2 NEW pytest tests (both PASS).
- No GPU launched (CPU-only introspection + pytest).
- No Edit to `pre_flight_check.py` (gate already registered at
  Wf-343C4 ship — line 4731-4739).
- Wf-321D `all == 66` audit invariant **PRESERVED**.
- Wf-343D1 hypothesis of wave-budget gate registration drift
  **REFUTED** by direct introspection.

---

## §7 References

- `TODO/completed/343c4_integration.md` — Wf-343C4 ship summary
  (4 NEW pre-retrain internal check gates).
- `TODO/completed/321d_preflight_gate_audit.md` — Wf-321D
  gate-count-consistency audit (source of `EXPECTED_GATE_COUNTS`
  invariant).
- `molmetal/tests/test_wf343c4_integration.py` — Wf-343C4
  integration suite (4 NEW tests covering the 4 gates' PASS / FAIL
  contracts).
- `molmetal/scripts/pre_flight_check.py:2817-2822` —
  `EXPECTED_GATE_COUNTS` definition.
- `molmetal/scripts/pre_flight_check.py:4731-4739` —
  `p3_wf343c4_wave_budget_remaining_nonneg` registration (this audit
  subject).

---

## §8 Tests added (2 NEW)

**File:** `molmetal/tests/test_wf346a4_wave_budget_registered.py`

1. `test_wf343c4_wave_budget_gate_registered_in_dispatcher` — pins
   registration of `p3_wf343c4_wave_budget_remaining_nonneg` in the
   'all' pipeline + asserts it's NOT in 'frontier' (Wf-343C4 §3.4
   design contract: pre-retrain internal checks are 'all'-only).
2. `test_build_gates_all_returns_66_with_four_wf343c4_gates` —
   pins the Wf-321D `EXPECTED_GATE_COUNTS["all"] == 66` invariant
   AND verifies all 4 Wf-343C4 gates are present in the 'all'
   pipeline (the Wf-343C4 integration suite covers runner-level
   PASS / FAIL behaviour; this is the registry + count invariant
   slice).

**Test status:** 2/2 PASS in 0.24 s wall.

---

**END OF Wf-346-A4**