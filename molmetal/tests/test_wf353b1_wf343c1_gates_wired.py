"""Wf-353-B1 — wire Wf-343c1's 26 pre-retrain gates into pre_flight_check.py.

==============================================================================
Wf-353-B1 — integration test layer for the Wf-343c1 → pre_flight_check
wiring.  Per Wf-343c1 finding: 26 pre-retrain gates are defined in
``molmetal/scripts/wf343c1_pre_retrain_check.py`` (auto-verifiable
subset of §1.1 + §1.3 + §2.1 + §3.1 + §5.11-§5.14 + §8.15).  Wf-353-B1
wires 22 of those 26 gates (the auto-verifiable subset, excluding the
4 §5.11-§5.14 post-retrain SKIP gates) INDIVIDUALLY into the
``build_gates('all')`` pipeline as 22 ``p3_wf353b1_<gate_name>``
runners.

The 4 post-retrain §5.11-§5.14 gates remain SKIP-only and are surfaced
via the standalone ``wf343c1_pre_retrain_check.py`` script (which is
verified by ``molmetal/tests/test_wf343c1_pre_retrain_check.py``).  This
matches the honest-framing convention used by Wf-347A1 (per-gate SKIP
gates are surfaced via the standalone script, NOT duplicated as
always-SKIP runners in pre_flight_check).

This module adds **4 NEW tests** (CPU-only, < 5 s total):

  1. ``TestGateWiring::test_22_new_gates_registered`` — pin the
     contract that exactly 22 ``p3_wf353b1_*`` runners are registered
     in ``build_gates('all')`` (10 code + 3 ablation + 7 data + 1
     test + 1 honest_framing; the 4 §5.11-§5.14 post-retrain SKIP
     gates are intentionally NOT wired individually).

  2. ``TestGateWiring::test_build_gates_all_returns_93`` — pin the
     backward-compat invariant: ``len(build_gates('all'))`` returns
     93 (= 71 pre-Wf-353-B1 baseline + 22 NEW Wf-343c1 gates).  The
     ``EXPECTED_GATE_COUNTS['all']`` constant is updated to 93 in
     lockstep; the Wf-345-B3 dynamic-count auto-compute also
     satisfies this invariant.

  3. ``TestGateWiring::test_per_gate_cli_hook_works`` — pin the
     contract that every one of the 22 NEW gate runners is
     individually invokable + returns a canonical ``GateResult``
     (status ∈ {PASS, FAIL, SKIP}, evidence non-empty, wall_sec >= 0).

  4. ``TestGateWiring::test_preflight_fresh_run_passes_22_new_gates`` —
     pin the contract that on a fresh-run pre-flight, all 22 NEW
     Wf-343c1 gates produce non-FAIL verdicts (PASS or honest-negative
     SKIP per the §2.1.6 BindingDB convention).  FAIL is not allowed
     for any of the 22 wired gates on the canonical repo state.

Honest framing (per AGENTS.md §1 + §5.2 + party-twenty-spirit §3.5):

* The 22 NEW gates are ADDITIVE only — they do NOT modify any
  pre-existing gate / runner / path.
* The Wf-321D gate-count-consistency audit invariant (``all == 93``
  = 71 pre-Wf-353-B1 baseline + 22 NEW) is preserved because the
  ``EXPECTED_GATE_COUNTS['all']`` constant is updated from 71 to 93
  in lockstep with the Wf-353-B1 ship (also auto-computed per
  Wf-345-B3).
* The 4 §5.11-§5.14 post-retrain gates are intentionally NOT
  wired individually — they remain SKIP per the honest-framing
  convention and are surfaced via the standalone
  ``wf343c1_pre_retrain_check.py`` script.
* CPU-only fix; runs in < 5 s.
==============================================================================
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest


# ---------------------------------------------------------------------------
# Paths / module fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
PREFLIGHT_SCRIPT = SCRIPTS_DIR / "pre_flight_check.py"
WF343C1_SCRIPT = SCRIPTS_DIR / "wf343c1_pre_retrain_check.py"
VENV_PY = "/home/hugo/codes/mol-metal/.venv/bin/python"

# The 22 NEW Wf-353-B1 gate names registered in build_gates('all').
# Per the Wf-343c1 surface: 10 code-level (§1.1.1-§1.1.10) +
# 3 ablation-flag (§1.3.1-§1.3.3) + 7 data-level (§2.1.1-§2.1.7) +
# 1 test-level (§3.1.5) + 1 honest-framing (§8.15) = 22 auto-verifiable
# gates.  The 4 §5.11-§5.14 post-retrain SKIP gates are NOT included.
EXPECTED_NEW_GATE_NAMES: Tuple[str, ...] = (
    # §1.1.1 — §1.1.10: code-level grep / help gates (10)
    "p3_wf353b1_code_1_1_1_indices_aliasing_fix",
    "p3_wf353b1_code_1_1_2_kendall_log_sigma",
    "p3_wf353b1_code_1_1_3_dann_per_metal",
    "p3_wf353b1_code_1_1_4_nan_to_num_guard",
    "p3_wf353b1_code_1_1_5_grad_clip_max_norm",
    "p3_wf353b1_code_1_1_6_w2_epoch_avg_history",
    "p3_wf353b1_code_1_1_7_ablation_flags_present",
    "p3_wf353b1_code_1_1_8_enhanced_metal_head",
    "p3_wf353b1_code_1_1_9_click_aug_flag",
    "p3_wf353b1_code_1_1_10_master_modules_flags",
    # §1.3.1 — §1.3.3: ablation flag coverage gates (3)
    "p3_wf353b1_ablation_1_3_1_flags_present",
    "p3_wf353b1_ablation_1_3_2_single_metal_flag",
    "p3_wf353b1_ablation_1_3_3_n_seeds_min_flag",
    # §2.1.1 — §2.1.7: data-level file existence / min-lines gates (7)
    "p3_wf353b1_data_2_1_1_metalcytotox_5metal_csv",
    "p3_wf353b1_data_2_1_2_metalcytotox_raw_csv",
    "p3_wf353b1_data_2_1_3_drugood_dir",
    "p3_wf353b1_data_2_1_4_nci60_dir",
    "p3_wf353b1_data_2_1_5_platinai_4cell_dir",
    "p3_wf353b1_data_2_1_6_bindingdb_external_honest_negative",
    "p3_wf353b1_data_2_1_7_click_yields_csv",
    # §3.1.5: NaN guard test file present (1)
    "p3_wf353b1_test_3_1_5_nan_guard_test_present",
    # §8.15: honest framing catalog entry-count gate (1)
    "p3_wf353b1_honest_framing_8_15_catalog_entry_count",
)

# The 4 §5.11-§5.14 post-retrain SKIP gate names that are INTENTIONALLY
# NOT wired individually (per honest-framing convention; surfaced via
# the standalone wf343c1_pre_retrain_check.py script).
EXPECTED_UNWIRED_POST_RETRAIN_GATE_NAMES: Tuple[str, ...] = (
    "p3_wf353b1_post_retrain_5_11_per_seed_std",
    "p3_wf353b1_post_retrain_5_12_os_auc_non_degenerate",
    "p3_wf353b1_post_retrain_5_13_ru_w2_epoch_avg",
    "p3_wf353b1_post_retrain_5_14_no_nan_auc",
)

# Insert repo root into sys.path so `from molmetal.scripts.pre_flight_check`
# resolves without requiring the test runner to set PYTHONPATH itself.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_preflight_module():
    """Import ``molmetal.scripts.pre_flight_check`` via :mod:`importlib`.

    Returns the loaded module object so tests can call the gate
    functions + ``build_gates()`` directly. This mirrors the
    :func:`_load_preflight_module` helper in
    ``test_wf343c4_integration.py``.
    """
    spec = importlib.util.spec_from_file_location(
        "pre_flight_check_under_test",
        PREFLIGHT_SCRIPT,
    )
    assert (
        spec is not None and spec.loader is not None
    ), f"pre_flight_check.py has no import spec ({PREFLIGHT_SCRIPT})"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pre_flight_check_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def preflight():
    """Module-level fixture exposing the pre_flight_check module."""
    return _load_preflight_module()


# ---------------------------------------------------------------------------
# 1 — 22 NEW gates registered in build_gates('all')
# ---------------------------------------------------------------------------


class TestGateWiring:
    """Wf-353-B1 — 22 Wf-343c1 gates wired into build_gates('all')."""

    def test_22_new_gates_registered(self, preflight) -> None:
        """Pin: exactly 22 ``p3_wf353b1_*`` runners are registered in
        ``build_gates('all')`` (10 code + 3 ablation + 7 data + 1 test
        + 1 honest_framing).

        Verifies the Wf-353-B1 contract:

          * All 22 expected NEW gate names are present in
            ``build_gates('all')`` runners.
          * The 4 post-retrain §5.11-§5.14 gates are INTENTIONALLY
            NOT wired individually (per honest-framing convention;
            surfaced via the standalone wf343c1_pre_retrain_check.py
            script).
          * No duplicate gate names (each name is unique).
        """
        all_runners = preflight.build_gates("all")
        all_names = [n for n, _ in all_runners]

        # All 22 expected NEW gates registered.
        for new_name in EXPECTED_NEW_GATE_NAMES:
            assert new_name in all_names, (
                f"{new_name} NOT registered in build_gates('all'); "
                f"Wf-353-B1 wiring drifted.  All runners: "
                f"{[n for n in all_names if 'wf353b1' in n]}"
            )

        # 4 post-retrain SKIP gates INTENTIONALLY NOT wired individually.
        for unwired in EXPECTED_UNWIRED_POST_RETRAIN_GATE_NAMES:
            assert unwired not in all_names, (
                f"{unwired} unexpectedly wired into build_gates('all'); "
                f"Wf-353-B1 design contract violated (post-retrain gates "
                f"should remain SKIP-only via standalone script)."
            )

        # Total count: exactly 22 wired (no duplicates, no extras).
        wf353b1_names = [n for n in all_names if n.startswith("p3_wf353b1_")]
        assert len(wf353b1_names) == 22, (
            f"expected 22 p3_wf353b1_* runners, got {len(wf353b1_names)}: "
            f"{wf353b1_names}"
        )

        # Uniqueness: no duplicate names (defensive against Wf-343c1
        # source-side renames that might collide).
        assert len(set(wf353b1_names)) == len(wf353b1_names), (
            f"duplicate p3_wf353b1_* gate names detected: "
            f"{[n for n in wf353b1_names if wf353b1_names.count(n) > 1]}"
        )

    def test_build_gates_all_returns_93(self, preflight) -> None:
        """Pin: ``build_gates('all')`` returns at least 93 runners
        (71 pre-Wf-353-B1 baseline + 22 NEW Wf-343c1 gates) AND
        ``EXPECTED_GATE_COUNTS['all']`` is updated to >= 93 in lockstep.

        Verifies the Wf-321D gate-count-consistency audit invariant:

          * ``len(build_gates('all'))`` >= 93 (the Wf-353-B1 contract
            is additive; subsequent workflows like Wf-350-C1 may add
            more gates on top).
          * ``EXPECTED_GATE_COUNTS['all']`` >= 93 (updated in lockstep
            with the Wf-353-B1 ship).
          * ``len(build_gates('all'))`` == ``EXPECTED_GATE_COUNTS['all']``
            (Wf-321D audit invariant).
          * The Wf-345-B3 dynamic auto-compute satisfies this invariant
            (i.e. ``actual == expected`` for the 'all' pipeline).

        The hard equality ``== 93`` is intentionally relaxed to ``>= 93``
        to accommodate subsequent workflow additions (e.g. Wf-350-C1
        added 1 GPU env-probe gate to bring the total to 94).  The
        additive-invariant contract is the load-bearing claim.
        """
        all_runners = preflight.build_gates("all")
        actual_total = len(all_runners)

        # The hardcoded constant must reflect the new baseline (>= 93).
        expected_total = preflight.EXPECTED_GATE_COUNTS["all"]
        assert expected_total >= 93, (
            f"EXPECTED_GATE_COUNTS['all'] must be >= 93 (71 pre-Wf-353-B1 "
            f"baseline + 22 NEW Wf-343c1 gates); got {expected_total}"
        )
        # Confirm the Wf-353-B1 contribution is reflected: actual must
        # also be >= 93 (i.e. the 22 NEW gates are wired, not phantom).
        assert actual_total >= 93, (
            f"build_gates('all') must be >= 93 (= 71 baseline + 22 NEW "
            f"Wf-343c1 gates); got {actual_total}.  Wf-353-B1 wiring "
            f"may have been reverted or partially broken."
        )

        # The hardcoded constant must match the actual count (Wf-321D audit).
        assert actual_total == expected_total, (
            f"build_gates('all') count drift: actual={actual_total} "
            f"vs expected={expected_total} (Wf-321D audit invariant)"
        )

        # Sanity: the audit gate itself PASSes (Wf-345-B3 auto-compute mode).
        audit_runner = next(
            (r for n, r in all_runners if n == "p3_gate_count_consistency_audit"),
            None,
        )
        assert (
            audit_runner is not None
        ), "Wf-321D gate_count_consistency_audit NOT in build_gates('all')"
        audit_result = audit_runner()
        assert audit_result.status == "PASS", (
            f"Wf-321D gate_count_audit FAILed with status="
            f"{audit_result.status}: {audit_result.evidence}"
        )
        assert (
            f"actual={expected_total} == expected={expected_total}"
            in audit_result.evidence
        ), (
            f"audit evidence must show actual={expected_total} == "
            f"expected={expected_total}; got {audit_result.evidence!r}"
        )
        assert "(auto-computed via Wf-345-B3)" in audit_result.evidence, (
            f"audit evidence must include Wf-345-B3 marker; got "
            f"{audit_result.evidence!r}"
        )

    def test_per_gate_cli_hook_works(self, preflight) -> None:
        """Pin: each of the 22 NEW gate runners is individually invokable
        and returns a canonical ``GateResult`` (status ∈ {PASS, FAIL, SKIP},
        evidence non-empty, wall_sec >= 0).

        Verifies the per-gate registration contract — every runner is
        a callable that produces a well-formed result.  This pins the
        Wf-353-B1 design contract that the 22 gates are wired as
        INDIVIDUAL runners (one per gate) so downstream consumers can
        inspect each gate's verdict independently (vs a single wrapper
        gate that emits one aggregate verdict).
        """
        all_runners = preflight.build_gates("all")
        runner_by_name: Dict[str, any] = {
            n: r for n, r in all_runners if n.startswith("p3_wf353b1_")
        }
        assert (
            len(runner_by_name) == 22
        ), f"expected 22 p3_wf353b1_* runners, got {len(runner_by_name)}"

        ALLOWED_STATUSES = {"PASS", "FAIL", "SKIP"}

        for gate_name in EXPECTED_NEW_GATE_NAMES:
            assert (
                gate_name in runner_by_name
            ), f"{gate_name} missing from build_gates('all') runner map"
            runner = runner_by_name[gate_name]
            assert callable(
                runner
            ), f"{gate_name} runner is not callable; got {type(runner)}"

            # Invoke the runner; it must return a GateResult (not raise).
            result = runner()

            # Canonical GateResult schema (mirrors pre_flight_check.GateResult).
            assert (
                result is not None
            ), f"{gate_name} returned None instead of GateResult"
            assert hasattr(
                result, "status"
            ), f"{gate_name} result missing 'status' attribute"
            assert hasattr(
                result, "evidence"
            ), f"{gate_name} result missing 'evidence' attribute"
            assert hasattr(
                result, "wall_sec"
            ), f"{gate_name} result missing 'wall_sec' attribute"
            assert hasattr(
                result, "name"
            ), f"{gate_name} result missing 'name' attribute"
            assert hasattr(
                result, "priority"
            ), f"{gate_name} result missing 'priority' attribute"
            assert hasattr(
                result, "pipeline"
            ), f"{gate_name} result missing 'pipeline' attribute"

            # Status must be one of the 3 canonical values.
            assert result.status in ALLOWED_STATUSES, (
                f"{gate_name} status must be in {ALLOWED_STATUSES}; "
                f"got {result.status!r}"
            )

            # Evidence must be a non-empty string.
            assert isinstance(result.evidence, str), (
                f"{gate_name} evidence must be a str; got " f"{type(result.evidence)}"
            )
            assert (
                len(result.evidence) > 0
            ), f"{gate_name} evidence must be non-empty; got empty string"

            # Wall time must be a non-negative float.
            assert isinstance(result.wall_sec, (int, float)), (
                f"{gate_name} wall_sec must be numeric; got " f"{type(result.wall_sec)}"
            )
            assert (
                result.wall_sec >= 0.0
            ), f"{gate_name} wall_sec must be >= 0; got {result.wall_sec}"

    def test_preflight_fresh_run_passes_22_new_gates(self, preflight) -> None:
        """Pin: on a fresh-run pre-flight, all 22 NEW Wf-343c1 gates
        produce non-FAIL verdicts (PASS or honest-negative SKIP per the
        §2.1.6 BindingDB convention).

        This test runs all 22 NEW gate runners IN-PROCESS via
        ``build_gates('all')`` (no subprocess hop, < 2 s on the host)
        and asserts that none return FAIL on the canonical repo state.

        Verifies the Wf-353-B1 contract:

          * All 21 expected PASS verdicts (10 code + 3 ablation +
            6 data + 1 test + 1 honest_framing).
          * The 1 expected honest-negative SKIP verdict
            (§2.1.6 BindingDB external — NOT used in retrain).
          * 0 FAIL verdicts (any FAIL means the wiring broke a
            pre-existing pass contract; that's a regression).
        """
        all_runners = preflight.build_gates("all")
        runner_by_name: Dict[str, any] = {
            n: r for n, r in all_runners if n.startswith("p3_wf353b1_")
        }
        assert (
            len(runner_by_name) == 22
        ), f"expected 22 p3_wf353b1_* runners; got {len(runner_by_name)}"

        # Expected verdicts per the Wf-343c1 design contract:
        #   * 21 PASS  (10 code + 3 ablation + 6 data + 1 test + 1 honest_framing)
        #   *  1 SKIP  (§2.1.6 BindingDB external — honest-negative)
        #   *  0 FAIL  (any FAIL = regression on canonical repo state)
        expected_pass_names: Tuple[str, ...] = tuple(
            n
            for n in EXPECTED_NEW_GATE_NAMES
            if n != "p3_wf353b1_data_2_1_6_bindingdb_external_honest_negative"
        )
        expected_skip_names: Tuple[str, ...] = (
            "p3_wf353b1_data_2_1_6_bindingdb_external_honest_negative",
        )

        results: Dict[str, str] = {}
        evidences: Dict[str, str] = {}
        for gate_name in EXPECTED_NEW_GATE_NAMES:
            runner = runner_by_name[gate_name]
            result = runner()
            results[gate_name] = result.status
            evidences[gate_name] = result.evidence

        # 21 PASS verdicts expected.
        actual_pass_names = tuple(n for n, s in results.items() if s == "PASS")
        assert set(actual_pass_names) == set(expected_pass_names), (
            f"PASS set drift: expected {set(expected_pass_names)}, "
            f"got {set(actual_pass_names)}.  Full results: {results}"
        )

        # 1 SKIP verdict expected (BindingDB honest-negative).
        actual_skip_names = tuple(n for n, s in results.items() if s == "SKIP")
        assert set(actual_skip_names) == set(expected_skip_names), (
            f"SKIP set drift: expected {set(expected_skip_names)}, "
            f"got {set(actual_skip_names)}.  Full results: {results}"
        )

        # 0 FAIL verdicts allowed (regression detector).
        actual_fail_names = tuple(n for n, s in results.items() if s == "FAIL")
        assert len(actual_fail_names) == 0, (
            f"REGRESSION: {len(actual_fail_names)} of 22 NEW Wf-343c1 "
            f"gates FAILed on the canonical repo state: "
            f"{actual_fail_names}.  "
            f"Fail evidences: {[evidences[n] for n in actual_fail_names]}.  "
            f"Wf-353-B1 wiring broke a pre-existing pass contract."
        )

        # Cross-check: the BindingDB SKIP evidence must reference the
        # honest-negative framing (not just generic skip text).
        bindingdb_evidence = evidences[
            "p3_wf353b1_data_2_1_6_bindingdb_external_honest_negative"
        ]
        assert (
            "BindingDB" in bindingdb_evidence
            and "external" in bindingdb_evidence.lower()
        ), (
            f"BindingDB SKIP evidence must mention BindingDB + external; "
            f"got {bindingdb_evidence!r}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
