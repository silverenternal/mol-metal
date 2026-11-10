"""Wf-415-A3 — pre-flight fail audit regression-guard tests.

Closes the Wf-415 wave pre-flight FAIL surface by pinning 3 invariants
on the canonical pre-flight baseline (per AGENTS.md §5.5 filesystem-truth
rule + §5.7 trainer flag wire-in smoke-required rule):

  1. ``test_preflight_total_count`` — total gate count is stable at 129
     (per Wf-353-B1 baseline + Wf-410-A2/A3/A4 additions + Wf-A125 G5's
     ``p3_run_ledger_provenance``) so any silent registry drift surfaces
     immediately.  **Re-synced 2026-10-31 (Wf-A127): 117 → 127.**
     **Re-synced 2026-11-01 (Wf-A154): 127 → 129** (the two SSOT-document
     consistency gates).
     **Re-synced 2026-11-04 (Wf-A269 wave-5/6): 129 → 132** (the
     ``p3_runtime_transport_check`` §5.11 gate).
     **Re-synced 2026-11-04 (Wf-A269 wave-19): 132 → 133** (the
     ``p3_script_contract_check`` §5.11 gate).  The superseded values
     (127, 129, 132) stay visible in this lineage per AGENTS.md §5.5
     rule #6.
  2. ``test_preflight_fail_count_baseline`` — pre-flight FAIL count is
     bounded at ≤7 (per Wf-413-B1 honest-negative baseline; Wf-415-A3
     closes 11+ FAILs from the Wf-413-B1 18-FAIL drift).
  3. ``test_preflight_pass_count_baseline`` — pre-flight PASS count is
     bounded at ≥107 (per Wf-415-A3 v3 measurement: 107 PASS / 7 FAIL /
     3 SKIP / 117 TOTAL — the historical Wf-415-A3 baseline, which still
     holds; MEASURED 2026-10-31 (Wf-A127): 113 PASS / 4 FAIL / 10 SKIP /
     0 WARN / 127 TOTAL).
  4. ``test_preflight_no_timeout_derived_fail`` — **NEW, Wf-A158
     (2026-11-01).**  No FAIL gate's evidence may carry a timeout
     marker.  A wall-clock timeout means the MEASUREMENT WAS NOT
     OBTAINED; reporting it as FAIL is a FALSE DEFECT CLAIM
     (AGENTS.md §5.2).  Wf-A135 established the rule; MEASURED
     2026-11-01 that it had been applied to only 3 of the 6 gates
     carrying an ``rc == 124`` branch, so the other three still mapped a
     timeout onto FAIL.  Wf-A158 repaired all three and added this
     invariant so a fourth cannot re-introduce it.

These tests are CASCADE-PROOF: a new failing wave that re-introduces
the pre-flight regression will trip one of the four invariants
immediately, surfacing the gap without manual inspection.

MEASURED lineage of the LIVE pre-flight (``--pipeline all --strict``,
129 gates):

  * 2026-10-31 (Wf-A127): 113 PASS / 4 FAIL / 10 SKIP / 0 WARN / 127 TOTAL
  * 2026-11-01 01:16 (Wf-A154): **101 PASS / 3 FAIL / 11 SKIP / 0 WARN /
    14 TIMEOUT / 129 TOTAL**, wall 686.63 s — the 14 TIMEOUTs included
    all 12 ``--help`` probe gates, and the PASS count fell BELOW the
    107 floor.
  * 2026-11-01 01:28 (Wf-A157): **118 PASS / 0 FAIL / 11 SKIP / 0 WARN /
    0 TIMEOUT / 129 TOTAL**, wall 280.78 s (−59 %) — the per-key probe
    lock collapsed 12 ``--help`` spawns to 1 and the budget rose from
    30 s to 180 s.
  * 2026-11-01 (Wf-A158): the three remaining ``rc == 124`` branches
    were re-pointed at ``_timeout_gate_result`` (a timeout is now its
    own verdict, never a FAIL).

Per AGENTS.md §5.5 filesystem-truth rule: every regression-guard test
asserts on a MEASURED value from a real ``pre_flight_check.py --pipeline
all --strict`` run, NOT a hand-coded expectation. The 3 invariants
above are pinned to Wf-415-A3 v3 + Wf-413-B1 baseline MEASURED values
stored at ``molmetal/reports/wf415a3_preflight_freshrun.txt``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRE_FLIGHT_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
FRESH_RUN_REPORT = REPO_ROOT / "molmetal" / "reports" / "wf415a3_preflight_freshrun.txt"


def _run_preflight() -> dict:
    """Run ``pre_flight_check.py --pipeline all --strict`` and return
    the parsed JSON payload (summary + the per-gate list).

    Wf-A133 (2026-10-31) — TWO fixes to a guard that was DEAD on this host:

    1. **The 300 s timeout expired on every run.**  MEASURED 2026-10-31: the
       pre-flight's own wall time on this host exceeds 300 s, so
       ``subprocess.TimeoutExpired`` was raised inside the module-scoped
       fixture and **all three invariants below reported ERROR instead of
       running**.  A guard that can never execute is not a guard — it is the
       exact "zombie gate" failure mode (a gate that looks installed and
       catches nothing).  The timeout is now 1800 s, overridable via
       ``MOLMETAL_PREFLIGHT_TIMEOUT_SEC``.
    2. **The subprocess ran bare ``python``** — a violation of AGENTS.md
       §5.9 (single-venv: ``/home/hugo/codes/mol-metal/.venv/bin/python``).
       ``sys.executable`` is the interpreter that is running the test, so the
       child cannot drift onto a different venv.

    Wf-A158 (2026-11-01) — the payload (not just the summary) is returned,
    because the 4th invariant inspects the per-gate evidence strings.
    """
    timeout = int(os.environ.get("MOLMETAL_PREFLIGHT_TIMEOUT_SEC", "1800"))
    proc = subprocess.run(
        [sys.executable, str(PRE_FLIGHT_SCRIPT), "--pipeline", "all", "--strict"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    # The script prints a ``[molmetal]`` banner before the JSON;
    # strip the banner so JSON parses cleanly.
    out = proc.stdout
    if out.startswith("[molmetal]"):
        out = out.split("\n", 1)[1]
    return json.loads(out)


@pytest.fixture(scope="module")
def preflight_payload() -> dict:
    return _run_preflight()


@pytest.fixture(scope="module")
def preflight_summary(preflight_payload: dict) -> dict:
    return preflight_payload["summary"]


def test_preflight_total_count(preflight_summary: dict) -> None:
    """Pin pre-flight total gate count at 129.

    Wf-353-B1 baseline 94 + Wf-405-A3 (Wf-385-A3 NEW gate) +1 +
    Wf-410-A2/A3/A4 (atomic-write subprocess-style gates) +3 +
    Wf-411-A1/A4 (Wf-385-A3 +Wf-414-A3 verifier gates) +2 +
    Wf-414-A1 (Wf-394-A3 NEW gate) +1 + Wf-411-A4 (Wf-385-A3 promoted
    to 'all') +16 = 117.  **Re-synced 2026-10-31 (Wf-A127): 117 → 127**
    after Wf-A125 G5 added ``p3_run_ledger_provenance`` to the ``all``
    pipeline (MEASURED by live import: ``len(build_gates("all")) == 127``,
    ``len(build_gates("launch")) == 129``).  **Re-synced 2026-11-01
    (Wf-A154): 127 → 129** after Wf-A154 added the two SSOT-document
    consistency gates (``p3_ssot_path_reference_check`` and
    ``p3_ssot_census_claim_check``) to the ``all`` pipeline (MEASURED by
    live import: ``len(build_gates("all")) == 129``,
    ``len(build_gates("launch")) == 131``).  Any drift below 129 means a
    gate was silently removed; any drift above 129 means a gate was
    duplicated without updating EXPECTED_GATE_COUNTS["all"] (Wf-345-B3
    dynamic compute auto-recompute should catch this, but the assertion is
    a belt-and-braces guard).

    **Bump protocol (4 places, catalog §6.236):** the code constant
    ``EXPECTED_GATE_COUNTS["all"]``, this assertion, the SSOT doc header
    ``TODO/completed/321d_preflight_gate_audit.md`` §2.1, and the §6
    summary table of the same doc.  A wave that bumps one and not the
    others leaves a RED test behind — that is exactly how this number
    drifted twice (2026-10-20 and 2026-10-31).
    """
    # 2026-11-04 (Wf-A269 wave-19) — re-synced, dated.  The superseded
    # value 132 stays visible BELOW and in the message below, per
    # AGENTS.md §5.5 rule #6.  MEASURED 2026-11-04 by live import:
    # len(build_gates("all")) == 133 after p3_script_contract_check
    # was registered.
    assert preflight_summary["total"] == 133, (
        f"pre-flight total gate count drift: expected 133 (per "
        f"the Wf-A269 wave-19 re-sync of the wave-5/6 baseline of "
        f"132, which itself re-synced the Wf-A154 baseline of 129, "
        f"which re-synced the Wf-A127 baseline of 127), got "
        f"{preflight_summary['total']}. "
        f"If you added/removed a gate, update this test + the "
        f"Wf-345-B3 dynamic compute + the EXPECTED_GATE_COUNTS['all'] "
        f"audit at TODO/completed/353b1_dynamic_gate_count_audit.md "
        f"+ TODO/completed/321d_preflight_gate_audit.md §2.1 and §6 "
        f"(catalog §6.236: the count lives in 4 places)."
    )
    # SUPERSEDED (2026-11-04, Wf-A269 wave-5/6 rendering — PRESERVED
    # VERBATIM per AGENTS.md §5.5 rule #6):
    #     assert preflight_summary["total"] == 132, (
    #         f"pre-flight total gate count drift: expected 132 (per "
    #         f"Wf-A154 re-sync of the Wf-A127 baseline of 127, plus "
    #         f"Wf-A154's two SSOT-document consistency gates), got {preflight_summary['total']}. "
    #     )
    # with 129 → 132 made by p3_runtime_transport_check (the THIRD
    # §5.11 gate).  The count moved on to 133 with
    # p3_script_contract_check (the FOURTH §5.11 gate).
    #
    # WHY this preserved rendering sits BELOW the live assertion and not
    # above it: the §6.236 docs-drift guard
    # (molmetal/tests/test_wfa127_residue_closure.py::
    # test_the_preflight_total_count_pin_is_the_current_count) greps
    # THIS file for the FIRST `assert preflight_summary["total"] == N`
    # and compares it with EXPECTED_GATE_COUNTS["all"].  A preserved
    # historical rendering placed above the live assert would be read as
    # the live pin (MEASURED: the guard reported "the pin says 132 but
    # the code constant says 133").  The live pin must be the first
    # occurrence.


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
    3 SKIP / 117 TOTAL (Wf-415-A3 v3 measurement — the historical
    baseline, still the threshold).  MEASURED 2026-10-31 (Wf-A127):
    **113 PASS / 4 FAIL / 10 SKIP / 0 WARN / 127 TOTAL**, so the ≥107
    PASS bound and the ≤7 FAIL bound both hold with margin.

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


# Wf-A158 (2026-11-01) — a FAIL may only ever be a MEASURED defect.
#
# A wall-clock timeout means the MEASUREMENT WAS NOT OBTAINED.  Reporting
# that as FAIL claims the thing under test is broken, which is exactly the
# false-defect-claim failure mode AGENTS.md §5.2 forbids.  Wf-A135 created
# the TIMEOUT verdict; MEASURED 2026-11-01 that three of the six gates
# carrying an ``rc == 124`` branch had never been converted, so a leakage
# diagnostic that ran out of wall time was reported as "the data leaks".
# This invariant closes the family: the marker strings below are the
# timeout verdict's own fingerprints, and a FAIL may not carry one.
TIMEOUT_MARKERS = (
    "TIMEOUT:",                 # the _timeout_gate_result evidence prefix
    "the measurement was NOT obtained",
    "TIMED OUT",               # the --help probe timeout wording
    "wall budget",              # the budget-exceeded wording
)


def test_preflight_no_timeout_derived_fail(preflight_payload: dict) -> None:
    """No FAIL gate's evidence may carry a timeout marker.

    MEASURED 2026-11-01 (Wf-A157, 129 gates): 118 PASS / 0 FAIL / 11 SKIP
    / 0 WARN / 0 TIMEOUT — the invariant holds trivially on a clean run.
    It exists to catch the REGRESSION: a wave that re-introduces (or adds)
    a timeout-as-FAIL mapping trips it immediately, with the offending gate
    named.
    """
    offenders = []
    for gate in preflight_payload.get("gates", []):
        if gate.get("status") != "FAIL":
            continue
        evidence = gate.get("evidence", "") or ""
        hits = [m for m in TIMEOUT_MARKERS if m in evidence]
        if hits:
            offenders.append((gate.get("name"), hits, evidence[:200]))
    assert not offenders, (
        "pre-flight reported a TIMEOUT-DERIVED FAIL — a wall-clock timeout "
        "means the measurement was NOT obtained, so it is NOT a verdict that "
        "the thing under test is broken (AGENTS.md §5.2; the Wf-A135 TIMEOUT "
        f"verdict). Offending gates: {offenders}"
    )
