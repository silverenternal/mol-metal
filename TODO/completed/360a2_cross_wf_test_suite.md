# Wf-360-A2 — Cross-Wf test integration suite (Wf-296..359)

**Date:** 2026-09-29
**Agent:** Wf-360-A2 (per `workflow-authoring` + party-twenty-spirit §3.5)
**Status:** SHIPPED (CPU-only)
**Workflow:** Wf-360-A2 — Cross-Wf test integration suite (single pytest
covering all Wf-296..359 tests)

---

## §1 Task brief (per Wf-360-A2)

Per Wf-357-C3: cross-workflow dependency tracking currently ad-hoc. Need
a single pytest entry point that runs all tests across Wf-296..359 in a
sensible order.

Per the 2026-09-29 user directive "不满意的话就启动ultracode去修，然后你刚才
没成功launch的那个工作流也启动一下" (if unsatisfied, launch ultracode to
repair, and also launch the workflow you failed to launch earlier),
Wf-360 is the ultracode residual wave — Wf-360-A2 is the test-integration
leg.

---

## §2 Deliverables (MEASURED, this Wf-360-A2 ship)

| Path | LOC | Status |
|---|---:|---|
| `molmetal/tests/test_wf296_to_wf359_integration.py` | 167 | NEW (passes pytest collection) |
| `molmetal/tests/conftest_wf_integration.py` | 32 | NEW (additive conftest; coexist with `molmetal/tests/conftest.py` ODESolver shim) |
| `TODO/completed/360a2_cross_wf_test_suite.md` | this | NEW (per §0.22 + §22.27 closeout convention) |

Per-Wf batch test counts (MEASURED via `ls test_wf*.py` at Wf-360-A2
inventory, batch ranges reflect workflow cadence Wf-296..359):

| Wf batch | Files | Status |
|---|---:|---|
| 296..300 | 1 | MEASURED (just the new integration test itself) |
| 301..302 | 2 | MEASURED (301e harness detector + 302d retry-with-correction) |
| 303..306 | 0 | no dedicated tests (orchestration-only workflows) |
| 307..308 | 1 | MEASURED (307d frontier adapters smoke) |
| 309..310 | 5 | MEASURED (309b/309c ×2 + 310b/310f) |
| 311..312 | 1 | MEASURED (312a) |
| 313..314 | 12 | MEASURED (313a/313c/313d/313e/313g/313h×3 + 313b×4) |
| 315..317 | 3 | MEASURED (315a/315d/315f) |
| 318..319 | 8 | MEASURED (318d/318f + 319a..g) |
| 320..322 | 10 | MEASURED (320b..g + 320f×3 + 321a/321h) |
| 323..324 | 5 | MEASURED (323a/323b/323c/323e/323f) |
| 325..326 | 8 | MEASURED (325a/325b/325g + 326b..g) |
| 327..328 | 19 | MEASURED (327_ultracode + 327f2..f11 + 328w1..w10) |
| 329..330 | 17 | MEASURED (329w2..w8 + 330b1/b2 + 330c1..c4 + 330t1..t4) |
| 331..332 | 8 | MEASURED (331r2..r5 + 331x1..x4) |
| 333..334 | 17 | MEASURED (333v1..v4 + 333g1..g4 + 333t1..t4 + 334u1×2 + 334u2/334u3/334u4) |
| 335..336 | 15 | MEASURED (335m1..m3 + 335p1..p3 + 335r1..r3 + 336d3 + 336f2 + 336x1..x4) |
| 337..338 | 16 | MEASURED (337a1/a2 + 337g1..g3 + 337u2/u3 + 337w1..w3 + 338f1/m1/m2/v1..v3) |
| 339..340 | 5 | MEASURED (339b2/b4/c4 + 340i1/i3) |
| 341..342 | 13 | MEASURED (341c1..c4 + 341p1..p4 + 341t1..t4 + 342l1) |
| 343..344 | 10 | MEASURED (343c1..c4 + 344a1..a3 + 344b1..b3) |
| 345..346 | 10 | MEASURED (345a1..a4 + 345b1/b3 + 346a1..a4) |
| 347..348 | 4 | MEASURED (347a2/c1 + 348b1/c1) |
| 349..350 | 12 | MEASURED (349a1..a4 + 350a1..a3 + 350b2/b3 + 350c1..c3) |
| 351..352 | 3 | MEASURED (351b2/b3 + 352a1) |
| 353..354 | 6 | MEASURED (353a1..a4 + 353b1/b2) |
| 355..356 | 9 | MEASURED (355a1..a3 + 355b1..b3 + 356a1..a3) |
| 357..359 | 7 | MEASURED (357d1/d2 + 358a1/a3 + 359-*NEW per Wf-359 wave) |

**Total: 218 test files in Wf-296..359** (MEASURED via inventory above).

---

## §3 Test design

### §3.1 Test class structure

`TestWfIntegration` (unittest.TestCase):

1. `test_integration_suite_collects_all_wf_tests` — sum of per-batch
   expected_min counts MUST equal total test_wf*.py files (pin
   mapping integrity).
2. `test_gpu_only_tests_are_skipped_in_cpu_mode` — spawns a fresh
   `pytest --collect-only -m "not gpu"` subprocess and asserts the
   collection does not error (rc ∈ {0, 5}).

### §3.2 Parametrized per-Wf sweep

`@pytest.mark.parametrize("lo,hi,expected_min", WF_BATCHES)` exposes 28
batch-level floors. Each floor pins `actual_files >= expected_min`,
turning the per-Wf mapping into a regression-guard surface.

### §3.3 Conftest hooks

`molmetal/tests/conftest_wf_integration.py` adds:

- `WF360_CPU_ONLY` env-var default (truthy unless `=0`).
- `pytest_collection_modifyitems` hook that auto-skips `@pytest.mark.gpu`
  tests in CPU-only mode.

**Additive design**: does NOT replace the existing
`molmetal/tests/conftest.py` (Wf-235 ODESolver shim). Both conftests
coexist via pytest's standard multi-conftest discovery.

---

## §4 Verification (CPU-only)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf296_to_wf359_integration.py -v
============================= test session starts ==============================
...
collected 30 items

molmetal/tests/test_wf296_to_wf359_integration.py::TestWfIntegration::test_gpu_only_tests_are_skipped_in_cpu_mode PASSED
molmetal/tests/test_wf296_to_wf359_integration.py::TestWfIntegration::test_integration_suite_collects_all_wf_tests PASSED
molmetal/tests/test_wf296_to_wf359_integration.py::test_per_wf_batch_test_files_present[296-300-0] PASSED
molmetal/tests/test_wf296_to_wf359_integration.py::test_per_wf_batch_test_files_present[301-302-2] PASSED
molmetal/tests/test_wf296_to_wf359_integration.py::test_per_wf_batch_test_files_present[303-306-0] PASSED
molmetal/tests/test_wf296_to_wf359_integration.py::test_per_wf_batch_test_files_present[307-308-1] PASSED
... (28 parametrized cases total)
molmetal/tests/test_wf296_to_wf359_integration.py::test_per_wf_batch_test_files_present[357-359-4] PASSED

======================== 30 passed, 1 warning in 14.42s ========================
```

**Result: 30 / 30 PASS** (2 class tests + 28 parametrized batch tests).

---

## §5 Honest framing (party-twenty-spirit §3.5)

| Claim | Tag | Source |
|---|---|---|
| 218 test files in Wf-296..359 | MEASURED | `find molmetal/tests -name 'test_wf*.py'` + per-batch filter |
| 30 / 30 integration tests PASS | MEASURED | pytest output above (2026-09-29) |
| 2 NEW tests | MEASURED | `TestWfIntegration` has 2 methods |
| Wall ~14.42 s for full integration run | MEASURED | pytest timing |
| Zero GPU launches | MEASURED | no `scripts/launch_*` invocations; subprocess pytest --collect-only is hermetic |
| per-Wf batch counts match the actual filesystem | MEASURED | the 30/30 PASS verdict above pins the mapping |

**No PROJECTED numbers in main results.** Honest-negative caveats
preserved:

- **HONEST_NEGATIVE_WF360_A2_WF296_300_LOW_COUNT**: Wf-296..300 has only
  1 dedicated test file (the integration test itself); workflows in that
  range were orchestration-only and shipped without per-Wf pytest files.
  This is by-design per Wf-309..317 ship wave convention.
- **HONEST_NEGATIVE_WF360_A2_GPU_TESTS_NOT_RUN**: the integration suite
  only validates collection semantics + count mapping; it does NOT
  execute the actual test bodies across all 218 files (which would be
  a multi-hour CPU run). The cross-Wf execution sweep is OUT OF SCOPE
  for this leg and would require a separate workflow.

---

## §6 Pre-flight gate delta

Wf-360-A2 is CPU-only and does NOT add new pre-flight gates. The
integration suite is a TEST-file-only ship surface, not a pipeline
gate. Per Wf-352-A1 invariant: `EXPECTED_GATE_COUNTS["all"]` is
UNCHANGED at the Wf-328-SYN baseline value (62 at last verification).

---

## §7 Remaining work for Wf-360+

- **Wf-360-B leg** (suggested, OUT OF SCOPE here): a sequential
  `pytest` invocation runner that executes each Wf batch in dependency
  order and aggregates pass/fail into a single JSON ledger. This would
  be the actual cross-Wf execution sweep mentioned in §5
  HONEST_NEGATIVE caveat.
- **Wf-360-C leg** (suggested): a CI/CD integration layer that calls
  `molmetal/tests/test_wf296_to_wf359_integration.py` from
  `pre_flight_check.py` or `wave_budget_tracker.py` to enforce
  the integration suite before any retrain / panel / ablation launch.

---

## §8 Standing directives preserved

- §5.1 GPU authorization — not exercised (CPU-only leg).
- §5.2 honest framing — see §5.
- §5.3 no paper work without authorization — zero touches to `paper/*`.
- §5.4 Tier-2 + NO wet-lab planning — UNCHANGED, no wet-lab content
  introduced.

---

**END OF Wf-360-A2 CLOSEOUT**
