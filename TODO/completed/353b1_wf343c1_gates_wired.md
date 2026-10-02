# Wf-353-B1 — Wire Wf-343c1's 22 auto-verifiable gates into pre_flight_check.py

**Date:** 2026-09-29
**Workflow:** Wf-353-B1 (ultracode residual — wire Wf-343c1's un-wired gates)
**Status:** SHIPPED
**Branch:** master
**Commits:** (this work-tree session; not yet committed)

## §1 Summary

Wf-353-B1 wires 22 of Wf-343c1's 26 pre-retrain gates INDIVIDUALLY into the
`build_gates('all')` pipeline as 22 `p3_wf353b1_<gate_name>` runners. The
4 §5.11-§5.14 post-retrain SKIP gates remain surfaced via the standalone
`molmetal/scripts/wf343c1_pre_retrain_check.py` script (per the
honest-framing convention — they're filled in AFTER the retrain wave, so
a SKIP-as-FAIL pattern would be misleading).

| Surface | Pre-Wf-353-B1 | Post-Wf-353-B1 |
|---|---:|---:|
| `build_gates('all')` length | 71 | 93 (= 71 + 22 NEW) |
| `EXPECTED_GATE_COUNTS['all']` | 71 | 93 |
| Wf-343c1 gates wired | 0 | 22 (auto-verifiable subset) |
| Wf-343c1 post-retrain SKIP gates wired | 0 | 0 (per honest-framing) |

## §2 What was wired

Per Wf-343c1 §1.1 + §1.3 + §2.1 + §3.1 + §8.15 (the auto-verifiable
subset; the 4 §5.11-§5.14 post-retrain SKIP gates are intentionally NOT
wired individually):

| Gate family | Count | Naming prefix | Surface |
|---|---:|---|---|
| §1.1.1-§1.1.10 code-level grep / help | 10 | `p3_wf353b1_code_1_1_<N>_<descriptor>` | code-level invariants (indices aliasing, Kendall log_sigma, DANN per_metal, NaN guard, grad clip, W₂ epoch-avg, ablation flags, enhanced metal head, click-aug, master modules) |
| §1.3.1-§1.3.3 ablation flag coverage | 3 | `p3_wf353b1_ablation_1_3_<N>_<descriptor>` | flag presence on chemprop v3 driver |
| §2.1.1-§2.1.7 data-level file existence / min-lines | 7 | `p3_wf353b1_data_2_1_<N>_<descriptor>` | MetalCytoToxDB / DrugOOD / NCI-60 / PlatinAI / BindingDB (honest-negative SKIP) / click yields |
| §3.1.5 test-level | 1 | `p3_wf353b1_test_3_1_5_nan_guard_test_present` | NaN guard test file on disk |
| §8.15 honest-framing catalog | 1 | `p3_wf353b1_honest_framing_8_15_catalog_entry_count` | parent doc has ≥15 §8 entries |
| **Total** | **22** | — | — |

The 22 runners are appended to the `'all'` pipeline via a list
comprehension over the Wf-343c1 gate-builder helpers
(`_wf343c1._code_level_gates()`, `_ablation_flag_gates()`,
`_data_level_gates()`, `_test_level_gates()`,
`_honest_framing_gates()`). The standalone script's `_GateResult`
objects are adapted to the pre-flight schema via the new helper
`_run_wf343c1_gate` (1-to-1 field mapping; no transformation needed).

## §3 Design contract

Per the Wf-343C4 + Wf-347A1 + Wf-350-A3 + Wf-350-A1 §3.4 design contract,
the 22 NEW gates are:

* **'all'-only** (NOT in `'frontier'` / `'chemprop_v2_v3'` / `'cfm'`).
* **Pre-retrain internal check metrics**, NOT paper-grade claims.
* **CPU-only** (the Wf-343c1 gate primitives — `gate_grep`,
  `gate_help`, `gate_file_exists`, `gate_min_lines`,
  `gate_section8_entry_count` — are all in-process; the only
  subprocess hop is `gate_help` which invokes
  `{py} {script} --help` with `timeout=30.0`).
* **Backward-compat**: does NOT modify any pre-existing gate / runner /
  path; only APPENDS 22 runners to `'all'` pipeline.

## §4 What was NOT wired (and why)

The 4 §5.11-§5.14 post-retrain SKIP gates are INTENTIONALLY NOT wired
individually. Per the Wf-343c1 docstring §"Gate primitives" + honest-framing
mandate (AGENTS.md §5.2 + party-twenty-spirit §3.5):

* **§5.11 `post_retrain_5_11_per_seed_std`** — requires post-retrain
  per-(metal, seed) std evidence; always SKIP pre-retrain.
* **§5.12 `post_retrain_5_12_os_auc_non_degenerate`** — requires
  Os AUC measurement; always SKIP pre-retrain.
* **§5.13 `post_retrain_5_13_ru_w2_epoch_avg`** — requires Ru
  `w2_epoch_avg_history` measurement; always SKIP pre-retrain.
* **§5.14 `post_retrain_5_14_no_nan_auc`** — requires 5 metals × 5 seeds
  non-NaN `test_auc`; always SKIP pre-retrain.

These gates remain surfaced via the standalone
`molmetal/scripts/wf343c1_pre_retrain_check.py` script (which is
invoked from `molmetal/tests/test_wf343c1_pre_retrain_check.py` for
full verification including the 4 SKIP gates). Surfacing them in
`build_gates('all')` as always-SKIP runners would dilute the pre-flight
PASS/FAIL signal — the §3.4 design contract reserves `'all'` for
gates that produce actual verdicts on the pre-retrain host state.

This matches the Wf-347A1 honest-framing convention: the
`gate_wf343c2_per_wave_runtime_check` + `gate_wf343c3_data_quality_check`
wrapper gates surface 7 + 5 checks respectively as ONE pre-flight
runner each (with the wrapper reporting aggregate PASS/FAIL); the
per-check granularity is preserved in the wrapper's evidence string
(via the standalone script's JSON output). The 22 Wf-353-B1 gates
take this further — each gate IS its own runner (because the
auto-verifiable subset is purely per-gate check, with no shared
subprocess state).

## §5 Files modified

| File | Change | LOC delta |
|---|---|---:|
| `molmetal/scripts/pre_flight_check.py` | (a) Added `_wf343c1` import block (top of file, ~37 LOC); (b) Added `_run_wf343c1_gate` helper function (~62 LOC); (c) Added 22-gate list comprehension in `build_gates('all')` (~70 LOC); (d) Updated `EXPECTED_GATE_COUNTS['all']` from 71 to 93 (now >= 93 with Wf-350-C1 addition). | +169 LOC |
| `molmetal/tests/test_wf353b1_wf343c1_gates_wired.py` | NEW — 4 integration tests (see §6 below). | +314 LOC NEW |
| `TODO/completed/353b1_wf343c1_gates_wired.md` | NEW — this closeout document. | +300 LOC NEW |

## §6 Test surface (4 NEW tests)

`molmetal/tests/test_wf353b1_wf343c1_gates_wired.py` — 4 NEW tests
(CPU-only, ~64 s wall on host):

| Test | Contract |
|---|---|
| `TestGateWiring::test_22_new_gates_registered` | All 22 expected `p3_wf353b1_*` runners are registered in `build_gates('all')`; the 4 §5.11-§5.14 post-retrain SKIP gates are INTENTIONALLY NOT wired individually; total count is exactly 22 (no duplicates, no extras). |
| `TestGateWiring::test_build_gates_all_returns_93` | `EXPECTED_GATE_COUNTS['all']` >= 93 (the Wf-353-B1 contribution); `len(build_gates('all'))` >= 93; the Wf-321D `gate_count_consistency_audit` PASSes with Wf-345-B3 auto-compute mode. |
| `TestGateWiring::test_per_gate_cli_hook_works` | Each of the 22 NEW runners is individually invokable + returns a canonical `GateResult` (status ∈ {PASS, FAIL, SKIP}, evidence non-empty string, wall_sec >= 0). |
| `TestGateWiring::test_preflight_fresh_run_passes_22_new_gates` | On a fresh-run pre-flight, all 22 NEW gates produce non-FAIL verdicts (21 PASS + 1 honest-negative SKIP for §2.1.6 BindingDB). FAIL count = 0 (regression detector). |

**Test result:** 4/4 PASS (CPU-only, ~64 s wall).

## §7 Backward-compat verification

Pre-existing pre-flight tests that exercise the gate registry:

| Test file | Status |
|---|---|
| `molmetal/tests/test_wf345b3_dynamic_count.py` (3 tests) | ALL PASS (Wf-345-B3 dynamic-count auto-compute still works; `EXPECTED_GATE_COUNTS` update flows through). |
| `molmetal/tests/test_wf343c4_integration.py` (4 tests) | ALL PASS (the 4 Wf-343C4 gates are still registered; per_wave_checklist partition still picks them up; 'frontier' pipeline count unchanged at 25). |

Total backward-compat: 7/7 PASS.

## §8 Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

* **MEASURED-vs-PROJECTED**: The 22 NEW gate names + their
  PASS/SKIP/FAIL verdicts are MEASURED via the in-process
  `_wf343c1.<gate_name>()` runner invocations. Wall time
  (`wall_sec`) is MEASURED on the host. `EXPECTED_GATE_COUNTS['all']`
  update from 71 to 93 is MEASURED in the test (the test asserts
  the constant >= 93 + matches the actual count).
* **Zero fabricated numbers**: all 22 gate names are verbatim from
  the Wf-343c1 standalone script's `_code_level_gates()` /
  `_ablation_flag_gates()` / `_data_level_gates()` /
  `_test_level_gates()` / `_honest_framing_gates()` builders.
* **Projected ceiling**: zero GPU launches performed; CPU-only fix.
  Wall budget on test suite: ~64 s (was ~0 s before; +64 s for the
  22 NEW gates × ~3 s avg per gate including the 6 `gate_help`
  subprocess calls at 30 s timeout but <1 s actual).

## §9 Forward-looking items

For Wf-354+ workflows that may want to extend this surface:

* The 4 §5.11-§5.14 post-retrain SKIP gates could be wired as
  always-SKIP runners IF a future workflow needs them in the
  per_wave_checklist P3 partition. Per §4 above, this is
  intentionally deferred — current `gate_count_consistency_audit`
  passes with 22-wired + 4-unwired pattern.
* If `wf343c1_pre_retrain_check.py` grows new gate families
  (e.g. §1.4 / §4 / §6 / §7 / §10 follow-on gates), the
  `_wf343c1.<builder>()` helpers can be appended to the
  list comprehension in `build_gates('all')` and the test
  file's `EXPECTED_NEW_GATE_NAMES` tuple updated.

## §10 Cross-references

* `molmetal/scripts/pre_flight_check.py` — the integration surface
  (import block at line ~67; `_run_wf343c1_gate` helper at line ~4499;
  22-gate list comprehension in `build_gates` at line ~5958).
* `molmetal/scripts/wf343c1_pre_retrain_check.py` — the standalone
  script (CPU-only, 893 LOC, 26-gate surface: 10 code + 3 ablation
  + 7 data + 1 test + 4 post_retrain + 1 honest_framing).
* `molmetal/tests/test_wf343c1_pre_retrain_check.py` — the standalone
  script's 8-test surface (preserves the 4 §5.11-§5.14 SKIP gate
  visibility).
* `molmetal/tests/test_wf343c4_integration.py` — the Wf-343C4
  integration tests (4 NEW tests for the 4 Wf-343C4 gates — the
  Wf-353-B1 surface follows the same per-gate registration pattern).
* `molmetal/tests/test_wf345b3_dynamic_count.py` — the Wf-345-B3
  dynamic-count auto-compute tests (3 tests — pin the invariant
  that `expected_total=None` auto-computes from `build_gates()`,
  which makes the `EXPECTED_GATE_COUNTS['all']` update from 71 to
  93 a no-op maintenance trap).

**END OF Wf-353-B1 SHIP SUMMARY**
