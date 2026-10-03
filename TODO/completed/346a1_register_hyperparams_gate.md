# Wf-346-A1 — Register `gate_wf343c4_hyperparams_config_complete` in `build_gates(all)` dispatcher + sync audit doc per Wf-321D §5.3 bump protocol

**Date:** 2026-09-29
**Author:** Wf-346-A1 (computed task) verification
**Workflow scope:** per the 2026-09-29 user ultracode directive *"基于刚才的洞察，
可以启动ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长"*
**Outcome:** **REGISTRATION ALREADY IN PLACE at Wf-346-A1 timestamp** — see
§2 Verification below. Wf-346-A1 wrote the 2-test regression-guard surface
AND fixed the downstream audit-doc drift surfaced by the Wf-327F11
regression-guard test (TestAuditDocSync FAIL at 62 vs 66).

---

## §1 Background (per the Wf-346-A1 task brief)

Per **Wf-343D1 finding (2026-09-29 07:51:24 UTC, wall 112.35s)**,
:func:`molmetal.scripts.pre_flight_check.build_gates` had a registration
drift:

  * `EXPECTED_GATE_COUNTS["all"] == 66` at `pre_flight_check.py:2818` (bumped
    by 4 to make room for the 4 NEW Wf-343C4 pre-retrain internal-check gates)
  * `build_gates("all")` returned only 62 runners at Wf-343D1 timestamp (the
    4 NEW gates were added to `EXPECTED_GATE_COUNTS` but NOT registered in
    the `all` pipeline's runner list)

Per **Wf-345-A1 follow-up work**, the gate LOGIC at
`pre_flight_check.py:2940-3069` was modified — but, because the gates were
never registered in the dispatcher, the entire Wf-345-A1 logic-fix effort
was moot (none of the 4 functions were ever CALLED).

Wf-346-A1 was opened to register the gate(s) and write the ADMEASURABLE
regression-guard surface so the registration cannot silently regress in
the future.

---

## §2 Verification (MEASURED at Wf-346-A1 timestamp 2026-09-29)

### §2.1 Direct introspection

```python
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
import sys; sys.path.insert(0, 'molmetal/scripts')
from pre_flight_check import build_gates, EXPECTED_GATE_COUNTS
for p in ('all', 'frontier', 'chemprop_v2_v3', 'cfm'):
    runners = build_gates(p); expected = EXPECTED_GATE_COUNTS[p]
    delta = len(runners) - expected
    names = [r[0] for r in runners]
    wf343c4 = [n for n in names if 'wf343c4' in n]
    print(f'{p:20s} expected={expected:3d} actual={len(runners):3d} delta={delta:+d} wf343c4={len(wf343c4)}')"

all                  expected= 66 actual= 66 delta=+0 wf343c4=4
frontier             expected= 25 actual= 25 delta=+0 wf343c4=0
chemprop_v2_v3       expected= 13 actual= 13 delta=+0 wf343c4=0
cfm                  expected=  6 actual=  6 delta=+0 wf343c4=0
```

**Verdict:** All 4 Wf-343C4 gates are registered in `build_gates("all")`; the
62-vs-66 mismatch that Wf-343D1 surfaced at 07:51:24 UTC has already been
resolved (likely by a Wf-345 / Wf-345-A1 / Wf-345-SYN follow-up commit that
registered the 4 gates between Wf-343D1 timestamp and Wf-346-A1 timestamp).

### §2.2 Per-gate registration map (MEASURED)

The 4 NEW Wf-343C4 gates are registered at the following lines in
`molmetal/scripts/pre_flight_check.py`:

| Gate name | Line | Function | Status at Wf-346-A1 |
|---|---|---|---|
| `p3_wf343c4_hyperparams_config_complete` | 4780-4795 | `gate_wf343c4_hyperparams_config_complete` (`pre_flight_check.py:2940`) | FAIL (logic) |
| `p3_wf343c4_yuelbond_decoder_env` | 4796-4803 | `gate_wf343c4_yuelbond_decoder_env` (`pre_flight_check.py:3072`) | PASS |
| `p3_wf343c4_anchor_freshness_24h` | 4804-4809 | `gate_wf343c4_anchor_freshness_24h` (`pre_flight_check.py:3215`) | FAIL (logic) |
| `p3_wf343c4_wave_budget_remaining_nonneg` | 4810-4818 | `gate_wf343c4_wave_budget_remaining_nonneg` (`pre_flight_check.py:3337`) | FAIL (logic) |

Note: the FAIL statuses above are LOGIC failures (the Wf-345-A1 follow-up
work to refresh stale anchors + complete the hyperparams contract check),
NOT registration failures. The registration contract is in place; the
logic-fix work is Wf-345 Phase 3 follow-up and is NOT in scope for
Wf-346-A1.

---

## §3 What Wf-346-A1 added (the regression-guard surface)

### §3.1 `molmetal/tests/test_wf346a1_hyperparams_registered.py` (NEW, 282 LOC)

Two NEW pytest tests, ADDITIVE only (no pre-existing test / gate / runner /
path modified):

1. **`TestBuildGatesAllRegistered::test_build_gates_all_returns_63_or_more_after_registration`** —
   pin the post-registration floor at 63 (one above the 62 pre-registration
   baseline per Wf-343D1). At Wf-346-A1 timestamp the actual count is 66,
   matching `EXPECTED_GATE_COUNTS["all"]`. The `>= 63` (rather than `== 66`)
   bound leaves headroom for future ADDITIVE gates — the exact-equality
   invariant is enforced separately by
   `TestExpectedGateCountsSync::test_expected_gate_counts_match_actual`
   in `molmetal/tests/test_wf327f11_gate_count_consistency.py`.

2. **`TestWf343C4GateInAllPipeline::test_gate_wf343c4_hyperparams_config_complete_in_all_pipeline`** —
   pin the SPECIFIC gate name `p3_wf343c4_hyperparams_config_complete` in
   the `all` pipeline dispatcher. Catches a future regression that REMOVES
   just that one gate (the broader 4-gate drift is covered by test 1).
   The runner is also invoked to assert GateResult contract shape
   (`status` / `priority` / `pipeline` / `evidence` / `wall_sec`) and that
   `result.status ∈ {PASS, FAIL, SKIP}` — wrapping the invocation in
   `try/except` so transient Wf-345-A1 logic-fix FAILs do not mask the
   registration contract.

### §3.2 Audit doc drift fix (per Wf-321D §5.3 bump protocol)

Running the existing Wf-327F11 regression-guard tests surfaced a
docs-vs-code drift: the audit doc
`TODO/completed/321d_preflight_gate_audit.md` still claimed 62 gates
(`§2.1 \`all\` pipeline — 62 gates`, §6 summary table 62 (`all`),
62/62 n_measured_only), but the code constant was bumped to 66 when the
4 Wf-343C4 gates were registered in the dispatcher.

Per Wf-321D §5.3 bump protocol, the audit doc must be updated
synchronously with `EXPECTED_GATE_COUNTS`. Wf-346-A1 fixed the drift:

| Location (pre-fix) | Old | New |
|---|---:|---:|
| §2.1 header (line 47) | `all` pipeline — 62 gates | `all` pipeline — 66 gates |
| §2.1 P3 row (line 55) | P3 = 36 | P3 = 40 (+ 4 Wf-343C4) |
| §2.1 Total row (line 56) | **Total 62** | **Total 66** |
| §5.2 code block (line 243) | `"all": 62` | `"all": 66` |
| §6 summary table (line 273) | 62 (`all`) | 66 (`all`) |
| §6 n_measured_only (line 277) | 62 / 62 | 66 / 66 |

### §3.3 Test results (MEASURED)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
  molmetal/tests/test_wf327f11_gate_count_consistency.py \
  molmetal/tests/test_wf346a1_hyperparams_registered.py -v

collected 5 items
molmetal/tests/test_wf327f11_gate_count_consistency.py::TestExpectedGateCountsSync::test_expected_gate_counts_match_actual PASSED [ 20%]
molmetal/tests/test_wf327f11_gate_count_consistency.py::TestAuditDocSync::test_audit_doc_table_count_matches_constant PASSED [ 40%]
molmetal/tests/test_wf327f11_gate_count_consistency.py::TestAuditGateFlip::test_audit_gate_passes_for_all_pipeline PASSED [ 60%]
molmetal/tests/test_wf346a1_hyperparams_registered.py::TestBuildGatesAllRegistered::test_build_gates_all_returns_63_or_more_after_registration PASSED [ 80%]
molmetal/tests/test_wf346a1_hyperparams_registered.py::TestWf343C4GateInAllPipeline::test_gate_wf343c4_hyperparams_config_complete_in_all_pipeline PASSED [100%]

=== 5 passed, 1 warning in 0.50s ===
```

The previously-FAILing
`TestAuditDocSync::test_audit_doc_table_count_matches_constant`
(now PASSes after the audit doc sync) was the immediate
"residual problem" the user surfaced via the ultracode directive;
this is the §3.2 fix landed in Wf-346-A1.

---

## §4 Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

### §4.1 What was NOT done

* **NO Edit to `pre_flight_check.py`** — the dispatcher at lines 4780-4818
  already contains all 4 Wf-343C4 gate registrations. The Wf-346-A1 task
  brief's literal "register the gate" work was moot at Wf-346-A1
  timestamp.
* **NO change to gate LOGIC** — Wf-345-A1 logic-fix work is in flight
  (3 of 4 Wf-343C4 gates currently FAIL on logic; this is downstream of
  Wf-346-A1 scope).
* **NO change to `EXPECTED_GATE_COUNTS`** — the constant was already at
  66 at Wf-346-A1 timestamp.

### §4.2 What was DONE (the honest shipment)

* Wrote the regression-guard surface (2 NEW tests, 282 LOC) that pins the
  registration contract so a future Edit cannot silently regress to the
  Wf-343-D1 4-gate drift state.
* Fixed the docs-vs-code drift in `TODO/completed/321d_preflight_gate_audit.md`
  per Wf-321D §5.3 bump protocol (6 string replacements; the audit doc
  now correctly reflects 66 gates).
* Verified the current registration is correct (66/66 match between
  constant and live runners; all 4 Wf-343C4 gates present).
* Documented the current registration map (§2.2) so downstream
  agents investigating gate FAILs have a quick reference.

### §4.3 Standing-rule compliance

* **AGENTS.md §1 MEASURED-vs-PROJECTED:** every numeric is build_gates-
  derived (test output + direct introspection).
* **AGENTS.md §5.2 honest framing:** no metric inflation; the FAIL statuses
  on 3 of 4 Wf-343C4 gates are LOGIC failures (downstream Wf-345-A1
  scope), not registration failures.
* **party-twenty-spirit §3.5:** every test is additive (no Wf-343C4 /
  Wf-321D / Wf-327F11 test removed or modified).
* **AGENTS.md §5.3 no paper work:** no `paper/*` touches.
* **AGENTS.md §5.4 Tier-2 markers:** no wet-lab planning; Wf-346-A1 is
  in-silico CPU-only work.

---

## §5 Wall budget (MEASURED)

| Phase | Wall |
|---|---|
| Build / Read / Verify | ~3 min |
| Write 2 NEW tests | ~5 min |
| Run 2 NEW tests | 0.31 s |
| Write TODO doc | ~3 min |
| **Total Wf-346-A1 wall** | **~12 min CPU-only** |

PROJECTED vs MEASURED: the Wf-346-A1 task brief PROJECTED 15-25 min CPU-only.
MEASURED was ~12 min (lower because the registration was already in place;
the projected Edit phase was moot).

---

## §6 What is NEXT (downstream of Wf-346-A1)

* **Wf-345 Phase 3 follow-up**: fix the LOGIC on the 3 FAIL Wf-343C4
  gates (hyperparams_config_complete, anchor_freshness_24h,
  wave_budget_remaining_nonneg). Once fixed, `build_gates("all")` will
  return 66 runners all PASSING (the user's ultracode directive end state).
* **Wf-321D §5.3 bump protocol**: any future ADDITIVE gate to the `all`
  pipeline must update BOTH `EXPECTED_GATE_COUNTS["all"]` AND the
  audit doc `TODO/completed/321d_preflight_gate_audit.md` together.
  The Wf-346-A1 regression-guard tests will catch any drift in
  `EXPECTED_GATE_COUNTS`.

---

**END OF Wf-346-A1** — owner: Wf-346-A1 computed-task; cross-ref
`TODO/completed/343c4_integration.md` + `TODO/completed/321d_preflight_gate_audit.md`
+ `TODO/completed/327f11_gate_count_consistency.md` + `TODO/completed/345a3_anchor_audit.md`.
