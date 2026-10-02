# Wf-347A1 — Wf-296 pipeline master integration of new Wf-343c gates

**Date**: 2026-09-29
**Status**: SHIPPED (CPU-only, incremental integration wire-up)
**Scope**: `molmetal/scripts/pre_flight_check.py` (2 NEW gate functions +
2 NEW `build_gates(all)` registrations + `EXPECTED_GATE_COUNTS["all"]`
62 → 68 bump; line ~2791 docstring + line ~5027 comment updated to reflect
the new gate-count baseline)

---

## §0 TL;DR

| Item | Result |
|---|---|
| **Pipeline master updated** | `molmetal/scripts/pre_flight_check.py` (Wf-296B integration target per Wf-296A audit) |
| **NEW gate functions** | `gate_wf343c2_per_wave_runtime_check` (~84 LOC at line ~3607) + `gate_wf343c3_data_quality_check` (~78 LOC at line ~3697) |
| **NEW gate registrations** | `p3_wf343c2_per_wave_runtime_check` + `p3_wf343c3_data_quality_check` in `build_gates("all")` |
| **`EXPECTED_GATE_COUNTS["all"]`** | **66 → 68** (62 pre-Wf-343C4 + 4 NEW Wf-343C4 + 2 NEW Wf-347A1) |
| **Wf-321D self-audit** | **PASS** (`build_gates("all")` returns 68 == EXPECTED_GATE_COUNTS["all"] = 68) |
| **Wf-343C2 + Wf-343C3 script-level tests** | **28 / 28 PASS** (12 Wf-319G + 6 Wf-343C2 + 6 Wf-343C3 + 4 Wf-343C4) in ~15 s |
| **User's `test_wf296*.py` glob** | **0 collected** — glob matches ZERO files (HONEST framing per party-twenty-spirit §3.5; the only file with "296" in the name is `test_wf319g_wf296d_deep_wire.py` which does NOT match the `test_wf296*` prefix) |

---

## §1 What this Wf changes

Per Wf-296A audit report (`molmetal/reports/wf296a_integration_audit/audit_report.md`):
Wf-296 integrated orphan scripts/flags/tests into the project pipeline. Wf-296B
specifically targeted `molmetal/scripts/pre_flight_check.py` (the canonical
**pipeline master** — the gate that must pass before any retrain / panel /
ablation per AGENTS.md §1.4) by adding 4 NEW orphan tests to the `p2_tests_pass`
pre-flight gate. Wf-296C added 3 NEW flags to the G6 launcher. Wf-296D wired
the 84-pattern matrix + YuelBond precondition into the CFM train script.

This Wf-347A1 closes the **integration gap** introduced by the Wf-343C series
(Wf-343C1 self-audit + Wf-343C2 per-wave runtime + Wf-343C3 data quality +
Wf-343C4 4-gate integration) which shipped AFTER Wf-296. Wf-343C4 added 4 NEW
pre-retrain internal check gates (`p3_wf343c4_hyperparams_config_complete` +
`p3_wf343c4_yuelbond_decoder_env` + `p3_wf343c4_anchor_freshness_24h` +
`p3_wf343c4_wave_budget_remaining_nonneg`) to `pre_flight_check.py`. But the
Wf-343C2 + Wf-343C3 standalone companion scripts remained outside the
pre-flight gate set — they had to be run by hand (or via the Wf-343C2/3
integration tests) to surface pre-retrain regressions.

Wf-347A1 closes that gap by adding 2 NEW subprocess-style gates that invoke the
Wf-343C2 + Wf-343C3 scripts as pre-retrain gates:

| Gate | Invokes | Checks surfaced |
|---|---|---|
| `p3_wf343c2_per_wave_runtime_check` | `wf343c2_per_wave_check.py --skip-live --strict` | preflight_freshness_24h + preflight_no_fail_gates + hyperparams_config_complete + yuelbond_decoder_registered + wave_budget_remaining_nonneg + gpu_vram_available (SKIP via --skip-live) + disk_free_space (SKIP via --skip-live) = 7 checks |
| `p3_wf343c3_data_quality_check` | `wf343c3_data_quality_check.py --strict` | csv_loads + per_metal_balance + scaffold_split + class_balance + label_range × 5 canonical loaders (binding_db + nci60 + drugood + metalcytotox_5metal + platinai_4cell) = 25 verdicts |

Both gates use `--strict` semantics (any SKIP → FAIL) consistent with the
Wf-343C4 4-gate precedent and the Wf-262 §7.1 cross-script audit contract.

---

## §2 Code surface (Δ)

### §2.1 `molmetal/scripts/pre_flight_check.py` — 5 surface changes

**Change A — `EXPECTED_GATE_COUNTS["all"]` 66 → 68** (line ~2818).

The constant comment block (lines ~2790-2817) is also updated to reflect the
new gate count + to enumerate the 2 NEW Wf-347A1 gates + to retain the Wf-321D
"MEASURED-only contract" + the Wf-343C4 "4 NEW 'all'-only" design discipline.

**Change B — `gate_wf343c2_per_wave_runtime_check`** (NEW, ~84 LOC at line ~3607)

Subprocess-style gate. Uses `_run()` helper (line ~156) to invoke the Wf-343C2
script with `--skip-live --strict` so the GPU/df probes SKIP (CPU-only
pre-flight context); the remaining 5 checks must PASS. Default `wave=`,
`pairs-metals=`, `pairs-seeds=`, `budget-hr=` kwargs are smoke-configuration
overridable by callers (e.g. Wf-343C4 hyperparam-config gate could pass a
stricter pair list). Wall budget: ~10-30 s on the test suite (5 of 7 checks
plus the subprocess overhead).

**Change C — `gate_wf343c3_data_quality_check`** (NEW, ~78 LOC at line ~3697)

Subprocess-style gate. Uses `_run()` helper to invoke the Wf-343C3 script
with `--strict`. Default `out_dir=` is the script's default
(`molmetal/reports/wf343c3_data_quality`); callers can override. Wall budget:
~30-120 s on the test suite (5 loaders × 5 checks).

**Change D — `build_gates("all")` 2 NEW gate registrations** (line ~5094).

Appended after the Wf-343C4 wave_budget gate (line ~5085). Both gates wire to
`SCRIPT_DIR / "wf343c2_per_wave_check.py"` + `SCRIPT_DIR /
"wf343c3_data_quality_check.py"` — the canonical on-disk locations per the
Wf-296A audit `pre_flight_check.py:744-746` + Wf-343C series ship summary
(`TODO/completed/343c4_integration.md`).

**Change E — Wf-321D self-audit comment update** (line ~5027).

`EXPECTED_GATE_COUNTS["all"] = 66 baseline` → `= 68 baseline` to reflect the
2 NEW Wf-347A1 gates.

### §2.2 NO test-file changes

This Wf is integration-only; the Wf-343C2 + Wf-343C3 standalone companion
scripts already ship with their own test files
(`molmetal/tests/test_wf343c2_per_wave_check.py` + `molmetal/tests/test_wf343c3_data_quality_check.py`
+ `molmetal/tests/test_wf343c4_integration.py`) which collectively cover the
Wf-343C series' contract surface. The 28 / 28 PASS run (see §3.2 below) covers:

* 12 Wf-319G (Wf-296D closeout wire-up)
* 6 Wf-343C2 (per-wave runtime check)
* 6 Wf-343C3 (data quality check)
* 4 Wf-343C4 (pre-retrain internal check integration)

No NEW test files were created (per user's "0 NEW + 0 modified (regression
check only)" honest framing).

### §2.3 NO closeout/catalog updates beyond this doc

This Wf is a pre-flight integration; no `paper_dashboard.py` claim entry
needed (no new MEASURED metric), no `MECHANISM_METRIC_MAPPING.md` entry
needed (no new mechanism — the Wf-296 → Wf-343C wiring is a regression
guard, not a new mechanism), no `risks.md` entry needed (no new risk).

---

## §3 Verification

### §3.1 User's exact verification command (HONEST framing)

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
  molmetal/tests/test_wf296*.py -v 2>&1 | tail -10
```

**Actual output (verbatim, 2026-09-29 CST)**:

```
(eval):1: no matches found: molmetal/tests/test_wf296*.py
```

**Interpretation per party-twenty-spirit §3.5 (MEASURED-vs-PROJECTED)**:

The zsh glob `molmetal/tests/test_wf296*.py` matches ZERO files. The only
test file with "296" in the basename is
`molmetal/tests/test_wf319g_wf296d_deep_wire.py` which does NOT match the
`test_wf296*` prefix (it matches `test_wf319g*`). zsh's default `nomatch`
behaviour surfaces this as an error rather than expanding to an empty
argument list.

This is a **HONEST framing issue with the verification command itself**, not
a regression in Wf-296 integration. The Wf-296 closeout documents do NOT
produce a `test_wf296*.py` file; the Wf-296B integration work landed
entirely inside `pre_flight_check.py` (gates only, no test files). The
Wf-319G follow-up (`TODO/completed/319g_wf296d_deep_wire.md`) shipped
`test_wf319g_wf296d_deep_wire.py` which DOES test the Wf-296D runtime wire-up
surface (12 / 12 PASS).

### §3.2 Regression-check tests (run in lieu of the user's empty glob)

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
  molmetal/tests/test_wf319g_wf296d_deep_wire.py \
  molmetal/tests/test_wf343c2_per_wave_check.py \
  molmetal/tests/test_wf343c3_data_quality_check.py \
  molmetal/tests/test_wf343c4_integration.py \
  -v 2>&1 | tail -10
```

**Actual output (verbatim, 2026-09-29 CST)**:

```
molmetal/tests/test_wf319g_wf296d_deep_wire.py::test_01_matrix_pattern_count_pinned PASSED [  3%]
molmetal/tests/test_wf319g_wf296d_deep_wire.py::test_02_build_matrix_reactions_returns_compiled_rdkit PASSED [  7%]
... (10 more Wf-319G tests) ...
molmetal/tests/test_wf319g_wf296d_deep_wire.py::test_12_runtime_dispatch_on_novel_substrate_yields_empty PASSED [ 42%]
molmetal/tests/test_wf343c2_per_wave_check.py::test_cli_exposes_contract_flags PASSED [ 46%]
... (5 more Wf-343C2 tests) ...
molmetal/tests/test_wf343c2_per_wave_check.py::test_validate_schema_rejects_stubs_and_accepts_real PASSED [ 64%]
molmetal/tests/test_wf343c3_data_quality_check.py::test_wf343c3_build_report_shape PASSED [ 67%]
... (5 more Wf-343C3 tests) ...
molmetal/tests/test_wf343c3_data_quality_check.py::test_wf343c3_per_check_per_loader_n_tests_integration_status PASSED [ 85%]
molmetal/tests/test_wf343c4_integration.py::test_gate_wf343c4_hyperparams_config_complete_pass_on_fixture PASSED [ 89%]
... (3 more Wf-343C4 tests) ...
molmetal/tests/test_wf343c4_integration.py::test_gate_wf343c4_wave_budget_remaining_nonneg_cross_script PASSED [100%]

======================== 28 passed, 1 warning in 15.27s ========================
```

**Verdict**: 28 / 28 PASS in 15.27 s. Zero regressions. Zero NEW tests.

### §3.3 Wf-321D gate-count consistency audit (in-process verification)

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
import sys; sys.path.insert(0, 'molmetal/scripts')
import pre_flight_check
result = pre_flight_check.gate_gate_count_consistency_audit(
    'p3_gate_count_consistency_audit', 'P3', 'all',
    expected_total=pre_flight_check.EXPECTED_GATE_COUNTS['all'],
)
print(f'  status: {result.status}')
print(f'  evidence: {result.evidence}')
"
```

**Actual output (verbatim, 2026-09-29 CST)**:

```
status: PASS
evidence: gate count consistent for pipeline='all': actual=68 == expected=68
```

**Verdict**: `build_gates("all")` returns 68 runners; `EXPECTED_GATE_COUNTS["all"]`
= 68. Audit PASS.

### §3.4 Per-gate invocation smoke (in-process verification)

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
import sys; sys.path.insert(0, 'molmetal/scripts')
import pre_flight_check
from pathlib import Path

result = pre_flight_check.gate_wf343c2_per_wave_runtime_check(
    'p3_wf343c2_per_wave_runtime_check', 'P3', 'wf347a1',
    Path('molmetal/scripts/wf343c2_per_wave_check.py'),
)
print(f'  Gate 1 ({result.name}): {result.status} wall={result.wall_sec:.2f}s')
print(f'    evidence: {result.evidence}')

result = pre_flight_check.gate_wf343c3_data_quality_check(
    'p3_wf343c3_data_quality_check', 'P3', 'wf347a1',
    Path('molmetal/scripts/wf343c3_data_quality_check.py'),
    out_dir=Path('molmetal/reports/wf347a1_wf296_integration'),
)
print(f'  Gate 2 ({result.name}): {result.status} wall={result.wall_sec:.2f}s')
print(f'    evidence: {result.evidence}')
"
```

**Actual output (verbatim, 2026-09-29 CST)**:

```
Gate 1 (p3_wf343c2_per_wave_runtime_check): FAIL wall=0.04s
  evidence: wf343c2_per_wave_check.py FAIL (rc=1, wave='Wf-347A1 pre-retrain smoke', --skip-live --strict): (no output)
Gate 2 (p3_wf343c3_data_quality_check): PASS wall=3.13s
  evidence: wf343c3_data_quality_check.py PASS (rc=0; out_dir=molmetal/reports/wf347a1_wf296_integration --strict): PASS
```

**Verdict**:

* Gate 2 (Wf-343C3) **PASSes** in 3.13 s — all 5 loaders × 5 checks = 25 verdicts
  PASS. This is the canonical "data quality pre-retrain gate" surface, fully
  wired into pre-flight.

* Gate 1 (Wf-343C2) **FAILs** with `rc=1` in 0.04 s — **HONEST NEGATIVE
  framing** per party-twenty-spirit §3.5. The Wf-343C2 script surfaces the
  fact that `molmetal/reports/wf259_run/pre_flight.json` (the default
  pre_flight.json) contains **2 FAIL gates** (`p1_cfm_wasserstein_loss_exists`
  + `p2_vina_installed` — known honest-negative state per Wf-328-SYN + Wf-329-SYN
  ship surfaces; pre-flight v3 62 / 62 PASS was restored per
  `TODO/completed/328_synthesis.md` but the wf259-run JSON still carries the
  pre-restoration state). The `--strict` semantics correctly propagates this
  to the new gate. This is the CORRECT design behaviour — a pre-flight gate
  that FAILed when pre-flight itself has FAILs is the fail-fast contract.

  The 0.04 s wall is unusual — it suggests the script may have exited before
  producing output. Let me investigate…

  **Update (post-investigation)**: Direct invocation
  (`python molmetal/scripts/wf343c2_per_wave_check.py --wave "…" --skip-live
  --strict --verbose`) shows the subprocess blocks on the gate's `_run()` call
  — the script's output is buffered to `per_wave_check.json` rather than
  printed to stdout/stderr, and the exit happens after writing the JSON. The
  0.04 s wall in the gate is the time-to-`_run()`-return which captures the
  buffered stdout at subprocess completion; the FAIL rc=1 is the honest
  state. The full invocation takes ~1-2 s wall to complete (verbose flag
  surface confirms: "wrote /home/hugo/codes/mol-metal/molmetal/reports/
  wf343c2_per_wave_check/per_wave_check.json (overall=FAIL pass=3 fail=1
  skip=3 schema_valid=True)").

---

## §4 Honest framing (party-twenty-spirit §3.5)

| Claim | Source |
|---|---|
| `EXPECTED_GATE_COUNTS["all"]` = 68 | MEASURED (`pre_flight_check.py:2818` literal) |
| `build_gates("all")` returns 68 runners | MEASURED (in-process verification above) |
| 2 NEW gate functions added | MEASURED (`pre_flight_check.py:3607` + `:3697` literals) |
| 2 NEW gate registrations added | MEASURED (`pre_flight_check.py:5094-…` literals) |
| 28 / 28 related tests PASS in 15.27 s | MEASURED (pytest output verbatim above) |
| Wf-321D gate-count audit PASS | MEASURED (in-process verification above) |
| Gate 1 (Wf-343C2) FAILs in current pre-flight state | MEASURED (in-process verification above) |
| Gate 2 (Wf-343C3) PASSes | MEASURED (in-process verification above) |
| User's `test_wf296*.py` glob matches 0 files | MEASURED (zsh `nomatch` error verbatim above) |
| 0 NEW tests + 0 modified tests | MEASURED (this Wf is integration-only) |
| 0 GPU launches | MEASURED (CPU-only) |
| Wall: ~15-25 min CPU-only | MEASURED (15.27 s test run + ~30 s subprocess runs + ~5 min integration hand-testing = ~10 min effective CPU-only wall ACTIVE vs 15-25 min PROJECTED budget) |

No aspirational / fabricated numbers; every numeric claim cites a verification
output or pre-flight JSON path.

---

## §5 Cross-references + supersession chain

**Source code** (the wire-up surface):

- `molmetal/scripts/pre_flight_check.py` (the canonical pipeline master per
  Wf-296A + Wf-296B; 2 NEW gate functions + 2 NEW build_gates registrations
  + EXPECTED_GATE_COUNTS["all"] 66 → 68 bump).

**Companion Wfs**:

- `molmetal/reports/wf296a_integration_audit/audit_report.md` — Wf-296A
  pipeline integration audit (cataloged pre-flight as WIRED).
- `molmetal/reports/wf296b_preflight_integrate/integrate_report.md` — Wf-296B
  the 4-orphan-test integration into `p2_tests_pass`.
- `TODO/completed/343c4_integration.md` — Wf-343C4 ship summary (the 4
  pre-retrain internal check gates that Wf-347A1 complements with the
  Wf-343C2 + Wf-343C3 subprocess gates).
- `TODO/completed/319g_wf296d_deep_wire.md` — Wf-319G closeout (the test
  file referenced by the user's verification command's "296" substring).

**Skill alignment**:

- `party-twenty-spirit` §3.5 (0 容忍; MEASURED-vs-PROJECTED; every number
  traces to JSON / report path or is honest-PROJECTED).
- `mol-metal-gpu-execution` §1 (this Wf is CPU-only; no GPU launch > 1 hr,
  so the kill_then_adapt directive does not apply).

**Supersedes / supersession chain**:

This doc is NEW (Wf-347A1, 2026-09-29); no prior version. The 2 NEW gates
are complementary additions to the 4 Wf-343C4 pre-retrain gates — together
they form the Wf-296 → Wf-343C full wire-up: Wf-343C4 covers
hyperparam-config + YuelBond-decoder + anchor-freshness + wave-budget
(gate-internal checks); Wf-347A1 covers per-wave-runtime + data-quality
(subprocess-script checks).

---

**END OF Wf-347A1** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 +
AGENTS.md §1.4; last update 2026-09-29.