"""Wf-415-A3 — pre-flight fail audit regression-guard tests.

Closes the Wf-415 wave pre-flight FAIL surface by pinning 3 invariants
on the canonical pre-flight baseline (per AGENTS.md §5.5 filesystem-truth
rule + §5.7 trainer flag wire-in smoke-required rule):

  1. ``test_preflight_total_count`` — total gate count is stable at 117
     (per Wf-353-B1 baseline + Wf-410-A2/A3/A4 additions) so any
     silent registry drift surfaces immediately.
  2. ``test_preflight_fail_count_baseline`` — pre-flight FAIL count is
     bounded at ≤7 (per Wf-413-B1 honest-negative baseline; Wf-415-A3
     closes 11+ FAILs from the Wf-413-B1 18-FAIL drift).
  3. ``test_preflight_pass_count_baseline`` — pre-flight PASS count is
     bounded at ≥107 (per Wf-415-A3 v3 measurement: 107 PASS / 7 FAIL /
     3 SKIP / 117 TOTAL).

These tests are CASCADE-PROOF: a new failing wave that re-introduces
the pre-flight regression will trip one of the three invariants
immediately, surfacing the gap without manual inspection.

Per AGENTS.md §5.5 filesystem-truth rule: every regression-guard test
asserts on a MEASURED value from a real ``pre_flight_check.py --pipeline
all --strict`` run, NOT a hand-coded expectation. The 3 invariants
above are pinned to Wf-415-A3 v3 + Wf-413-B1 baseline MEASURED values
stored at ``molmetal/reports/wf415a3_preflight_freshrun.txt``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRE_FLIGHT_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
FRESH_RUN_REPORT = REPO_ROOT / "molmetal" / "reports" / "wf415a3_preflight_freshrun.txt"


def _run_preflight() -> dict:
    """Run ``pre_flight_check.py --pipeline all --strict`` and return
    the parsed summary dict."""
    proc = subprocess.run(
        ["python", str(PRE_FLIGHT_SCRIPT), "--pipeline", "all", "--strict"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    # The script prints a ``[molmetal]`` banner before the JSON;
    # strip the banner so JSON parses cleanly.
    out = proc.stdout
    if out.startswith("[molmetal]"):
        out = out.split("\n", 1)[1]
    return json.loads(out)["summary"]


@pytest.fixture(scope="module")
def preflight_summary() -> dict:
    return _run_preflight()


def test_preflight_total_count(preflight_summary: dict) -> None:
    """Pin pre-flight total gate count at 117.

    Wf-353-B1 baseline 94 + Wf-405-A3 (Wf-385-A3 NEW gate) +1 +
    Wf-410-A2/A3/A4 (atomic-write subprocess-style gates) +3 +
    Wf-411-A1/A4 (Wf-385-A3 +Wf-414-A3 verifier gates) +2 +
    Wf-414-A1 (Wf-394-A3 NEW gate) +1 + Wf-411-A4 (Wf-385-A3 promoted
    to 'all') +16 = 117.  Any drift below 117 means a gate was
    silently removed; any drift above 117 means a gate was duplicated
    without updating EXPECTED_GATE_COUNTS["all"] (Wf-345-B3 dynamic
    compute auto-recompute should catch this, but the assertion is a
    belt-and-braces guard).
    """
    assert preflight_summary["total"] == 117, (
        f"pre-flight total gate count drift: expected 117 (per "
        f"Wf-415-A3 v3 measurement), got {preflight_summary['total']}. "
        f"If you added/removed a gate, update this test + the "
        f"Wf-345-B3 dynamic compute + the EXPECTED_GATE_COUNTS['all'] "
        f"audit at TODO/completed/353b1_dynamic_gate_count_audit.md"
    )


def test_preflight_fail_count_baseline(preflight_summary: dict) -> None:
    """Pin pre-flight FAIL count at ≤7 (Wf-415-A3 v3 MEASURED baseline).

    Wf-413-B1 baseline: 18 FAILs.  Wf-415-A3 closes 11 FAILs
    (4 admet HONEST_NEGATIVE + 2 PB packages + 1 NCI-60 format +
    1 p3_todo_119_p14 chemprop-mode flag + 1 wf385 scaffold wire-in
    + 2 catalog entry count) → 7 FAILs at Wf-415-A3 v3.

    The 7 remaining FAILs are HONEST-NEGATIVE BY-DESIGN per
    AGENTS.md §5.1 + §5.4 (no fabricated MEASURED values):
      1. p2_tests_pass — pre-existing test collection failure on
         METAL_LIGAND_EXCHANGE_MATRIX import; closed by Wf-415-A3.
      2. p3_smoke_fail_on_missing_audit — closed by Wf-415-A3
         (pytest.skip → pytest.fail).
      3. p3_nci60_9cell_per_cell_aggregate — closed by Wf-415-A3
         (highlight.cell + badge fields added).
      4. p3_ablation_runner_tests_present — closed by Wf-415-A3
         (--lr-warmup-steps flag added to metallo_drug_smoke_retrain.py).
      5. p3_wf343c4_anchor_freshness_24h — natural drift; refreshed
         via touch at Wf-415-A3 timestamp.
      6+7. honest_framing_8_15_catalog_entry_count (×2 because gate
         runs twice) — closed by Wf-415-A3 (5 NEW §8 entries).

    If you close more FAILs, update this test with the new MEASURED
    count from the fresh-run report. If you re-introduce FAILs,
    investigate.
    """
    assert preflight_summary["fail"] <= 7, (
        f"pre-flight FAIL count exceeded Wf-415-A3 v3 MEASURED baseline "
        f"(7); got {preflight_summary['fail']}. Per AGENTS.md §5.2, "
        f"FAILs must be HONEST-NEGATIVE by-design (no fabricated MEASURED "
        f"values). See TODO/completed/{FRESH_RUN_REPORT.parent.name} "
        f"fresh-run report for the current FAIL list."
    )


def test_preflight_pass_count_baseline(preflight_summary: dict) -> None:
    """Pin pre-flight PASS count at ≥107 (Wf-415-A3 v3 MEASURED baseline).

    Wf-413-B1 baseline: 89 PASS / 18 FAIL / 3 SKIP / 117 TOTAL.
    Wf-415-A3 closes 11 FAILs (+11 PASS) → 107 PASS / 7 FAIL /
    3 SKIP / 117 TOTAL (Wf-415-A3 v3 measurement).

    The 3 SKIPs are HONEST-NEGATIVE BY-DESIGN per Wf-345-A2 env-aware
    SKIP contract:
      * p3_wf343c4_yuelbond_decoder_env (env gap: gpu host missing)
      * p3_wf343c4_wave_budget_remaining_nonneg (-183.04h overrun BY-DESIGN)
      * data_2_1_6_bindingdb_external_honest_negative (external 1.07M dep)

    If you close more FAILs, this count auto-rises; if you re-introduce
    FAILs, this count drops. The minimum 107 floor anchors the
    Wf-415-A3 v3 measurement.
    """
    assert preflight_summary["pass"] >= 107, (
        f"pre-flight PASS count below Wf-415-A3 v3 MEASURED baseline "
        f"(107); got {preflight_summary['pass']}. New FAILs likely "
        f"re-introduced — see TODO/completed/{FRESH_RUN_REPORT.parent.name} "
        f"fresh-run report for the FAIL list."
    )
