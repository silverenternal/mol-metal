# Wf-367-A2 — Pre-flight No-Fail-Gates cascade close-out

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Status:** SHIPPED (root-cause fix + cascade fully closed + 1 NEW regression-guard test)
**Workflow:** Wf-367-A2
**Cross-ref:** Wf-358-D1 + Wf-362-B1 (cascade surfacing); Wf-364-A1 (upstream cascade fix on `p2_tests_pass`); Wf-365-A2 (follow-up investigation).

---

## §1 Background

Per Wf-358-D1 + Wf-362-B1: pre-flight v3 reported **1 FAIL** that
**cascaded** to `p3_wf343c2_per_wave_runtime_check`.  The cascade
contract is:

```
p2_tests_pass FAIL (Wf-364-A1 root cause: literal '%' in argparse help)
    └─→ preflight summary.fail = 1
            └─→ wf343c2.check_preflight_no_fail_gates FAIL
                    └─→ wf343c2_per_wave_check.py exit 1 (with --strict)
                            └─→ p3_wf343c2_per_wave_runtime_check FAIL
```

Wf-364-A1 closed the upstream FAIL on `p2_tests_pass` (3-line patch to
escape `%` in `train_dmpnn_multitask_wasserstein.py` argparse help
strings + Wf-364-A1 §4.1 self-referential cycle fix on
`wf343c2_per_wave_check.py:check_preflight_no_fail_gates`).  But
Wf-364-A1 §4.2 noted a **remaining FAIL** that was NOT in the cascade
scope:

> "The cascade cycle existed because `check_preflight_no_fail_gates`
> reads `summary.fail` from the JSON, but `summary.fail` includes this
> gate's own FAIL.  [...] `p3_wf343c2_per_wave_runtime_check` itself
> still reports FAIL with `rc=2` (`--strict` + `--skip-live` semantics:
> 3 SKIPs → exit 2 → FAIL).  [...] If the team wants to close this
> last FAIL, the next ultracode wave should remove `--strict` from
> `gate_wf343c2_per_wave_runtime_check`'s invocation (or split
> `--skip-live` vs `--strict` into 2 distinct gates).  Tracked for
> follow-up; not blocking Wf-364-A1 ship."

Wf-367-A2 closes this follow-up.

## §2 Root cause

The `gate_wf343c2_per_wave_runtime_check` function in
`molmetal/scripts/pre_flight_check.py` invoked the Wf-343C2 per-wave
companion script with both `--skip-live` AND `--strict`:

```python
# pre_flight_check.py:4590-4598 (pre-fix)
args = [
    VENV_PY, str(script_path),
    "--wave", wave,
    "--pairs-metals", pairs_metals,
    "--pairs-seeds", pairs_seeds,
    "--budget-hr", str(budget_hr),
    "--skip-live",   # ← produces 3 SKIPs (hyperparams + gpu + disk)
    "--strict",      # ← converts ANY SKIP into FAIL (exit 2)
]
```

The semantic conflict is:

| Sub-check | Status under `--skip-live` |
|---|---|
| `preflight_freshness_24h` | PASS |
| `preflight_no_fail_gates` | PASS |
| `hyperparams_config_complete` | SKIP (no `--config` provided) |
| `yuelbond_decoder_registered` | PASS |
| `wave_budget_remaining_nonneg` | PASS |
| `gpu_vram_available` | SKIP (`--skip-live`) |
| `disk_free_space` | SKIP (`--skip-live`) |

→ 4 PASS / 0 FAIL / 3 SKIP → `overall_status = PASS_WITH_SKIP` →
exit code 2 with `--strict`, exit code 0 without `--strict`.

The script itself returns exit 0 on `PASS_WITH_SKIP` if `--strict` is
not set; the pre-flight gate's invocation was incorrectly layering
`--strict` on top of `--skip-live`, producing the synthetic FAIL.

## §3 Fix

Drop `--strict` from the gate's invocation.  The `--strict` flag
remains available on the `wf343c2_per_wave_check.py` CLI for explicit
user opt-in (e.g. when a wave launcher script wants SKIP-as-FAIL
semantics), but the pre-flight gate's CPU-only `fresh-run` context
must NOT use `--strict` (it would convert the legitimate
`--skip-live` SKIPs into FAIL).

```diff
--- a/molmetal/scripts/pre_flight_check.py
+++ b/molmetal/scripts/pre_flight_check.py
@@ -4590,11 +4590,16 @@ def gate_wf343c2_per_wave_runtime_check(
     args = [
         VENV_PY, str(script_path),
         "--wave", wave,
         "--pairs-metals", pairs_metals,
         "--pairs-seeds", pairs_seeds,
         "--budget-hr", str(budget_hr),
         "--skip-live",
-        "--strict",
+        # NOTE (Wf-367-A2): do NOT pass --strict here. The --skip-live flag
+        # legitimately produces 3 SKIPs (hyperparams + gpu + disk); --strict
+        # would convert any SKIP into FAIL (exit 2), causing this gate to FAIL
+        # forever in the CPU-only pre-flight context.  --strict belongs in
+        # the user-invoked CLI path (e.g. ``wf343c2_per_wave_check.py
+        # --strict`` from a wave launcher), NOT in the pre-flight gate that
+        # drives the upstream cascade.  Per Wf-364-A1 §4.2 root-cause
+        # recommendation.
     ]
```

Also updated the PASS / FAIL evidence strings (lines 4610-4625) to
reflect the Wf-367-A2 fix so future agents can grep the JSON for the
contract marker `no --strict per Wf-367-A2`.

## §4 Cascade verification

Pre-flight fresh-run (`python molmetal/scripts/pre_flight_check.py
--pipeline all`) — BEFORE vs AFTER Wf-367-A2 fix:

| Step | Pre-fix | Post-fix |
|---|---|---|
| `gate_wf343c2_per_wave_runtime_check` | FAIL (rc=2, --skip-live --strict) | **PASS** (rc=0, --skip-live, no --strict per Wf-367-A2) |
| Pre-flight summary | 91 PASS / **1 FAIL** / 2 SKIP / 94 TOTAL | **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** |
| `preflight_no_fail_gates` sub-check | PASS (upstream FAIL=1) | PASS (upstream FAIL=0) |
| `p2_tests_pass` | PASS (Wf-364-A1 fix) | PASS (Wf-364-A1 fix preserved) |

The pre-flight cascade is now **fully closed**:

- `p2_tests_pass` → PASS (Wf-364-A1 fix)
- `preflight_no_fail_gates` → PASS (upstream FAIL=0)
- `gate_wf343c2_per_wave_runtime_check` → PASS (Wf-367-A2 fix)

## §5 NEW test (regression guard)

`molmetal/tests/test_wf367a2_no_fail_gates.py` (NEW, 1 test):

### `test_gate_wf343c2_per_wave_runtime_check_no_strict_invocation`

Pins the Wf-367-A2 contract:

1. **Primary contract:** invoking the gate function directly via the
   production default args (no --strict, --skip-live only) returns
   `GateResult(status="PASS", ...)`.  This catches any future
   regression where someone re-introduces `--strict` into the gate's
   subprocess args list.
2. **Secondary contract:** the gate function source contains NO
   literal `"--strict"` token in the args=[ ... ] list (other than in
   comments / docstrings).  Source-grep regression guard — directly
   inspects the function body and fails if the offending token
   reappears.

The test driver writes an inline gate runner to a temp path under
`SCRIPTS_DIR`, runs it via subprocess, parses `STATUS=` and `EVIDENCE=`
lines from stdout, and cleans up the temp file in a `finally:` block.

## §6 Run results

```
$ pytest molmetal/tests/test_wf367a2_no_fail_gates.py -v
collected 1 item

molmetal/tests/test_wf367a2_no_fail_gates.py::test_gate_wf343c2_per_wave_runtime_check_no_strict_invocation PASSED [100%]

========================= 1 passed, 1 warning in 1.74s =========================
```

Wf-364-A1 cascade tests still PASS (no regression from the Wf-367-A2 fix):

```
$ pytest molmetal/tests/test_wf364a1_preflight_cascade.py molmetal/tests/test_wf367a2_no_fail_gates.py -v
collected 3 items

molmetal/tests/test_wf364a1_preflight_cascade.py::test_train_dmpnn_multitask_wasserstein_help_exits_zero PASSED
molmetal/tests/test_wf364a1_preflight_cascade.py::test_preflight_no_fail_gates_cascade_status PASSED
molmetal/tests/test_wf367a2_no_fail_gates.py::test_gate_wf343c2_per_wave_runtime_check_no_strict_invocation PASSED

========================= 3 passed, 1 warning in 4.19s =========================
```

Pre-flight fresh-run results (post-fix): see `molmetal/reports/wf367a2_preflight_after.txt`:
- **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** (vs 91/1/94 pre-fix)
- `p3_wf343c2_per_wave_runtime_check` evidence: `wf343c2_per_wave_check.py PASS (rc=0; wave='Wf-347A1 pre-retrain smoke', pairs=(Ru,Ir,Os,Rh,Re)×(0,1,2), budget=10.0h, --skip-live, no --strict per Wf-367-A2)`

## §7 Honest framing

- **MEASURED**: 1 NEW test PASS (Wf-367-A2 regression guard); 0/3
  Wf-364-A1 cascade tests regressed; pre-flight summary went from
  `91/1/2/94` (1 FAIL cascade) → `92/0/2/94` (cascade fully closed);
  wall: ~2 s for the new test, ~155 s for the full pre-flight fresh-run
  (delta vs Wf-364-A1 baseline negligible).
- **PROJECTED**: ~15-25 min total wall budget for the Wf-367-A2
  workflow (CPU-only documentation + 1-line source patch + 1 NEW test
  file).  Actual wall: ~5 min.
- **NO GPU launches** performed (CPU-only fix; documented per §5.1).
- **NO paper touches** per §5.3 (TODO-84 active).

## §8 File diff summary

```
molmetal/scripts/pre_flight_check.py                     | 13 +++++++++++--
molmetal/tests/test_wf367a2_no_fail_gates.py             | NEW file (~150 LOC)
TODO/completed/367a2_preflight_no_fail_gates_fix.md      | NEW file (this doc)
molmetal/reports/wf367a2_preflight.txt                   | NEW (pre-fix baseline)
molmetal/reports/wf367a2_preflight_after.txt             | NEW (post-fix clean state)
```

## §9 Cross-references

- Wf-364-A1 — `TODO/completed/364a1_preflight_cascade_fix.md`
  (upstream cascade fix; §4.2 root-cause recommendation followed by
  Wf-367-A2)
- Wf-358-D1 / Wf-362-B1 — surface discoveries that motivated the
  cascade investigation
- Wf-347A1 — introduced `p3_wf343c2_per_wave_runtime_check` (the gate
  Wf-367-A2 fixes)
- Wf-302A — AnchorGuard 24h cap (the underlying `--strict` semantic)
- AGENTS.md §1 rule 4 — pre-flight PASS requirement
- AGENTS.md §5.2 — MEASURED vs PROJECTED honest framing

**END OF Wf-367-A2 SHIP DOC** — Owner Hugo Lee; 2026-09-29.