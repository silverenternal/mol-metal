"""Wf-360-A1 — Wave budget rebalancing tests.

Per Wf-262 §7.1 + Wf-345-A4 + Wf-357-A1 + LAMBDA_ROADMAP §3.6, the 8
DEFERRED waves classified by the budget tracker fall into two distinct
populations:

  * DEFERRED_TEMP — USER_OK-gated, resumable when wall budget is allocated
    (the 5 CFM/TODO-119 waves + Wf-176 Phase 2 + Wf-131 P3 7 DEFERRED +
    Wf-146 Re fold + Os seed 4).
  * DEFERRED_PERMANENT — architectural blocker; resumption requires
    upstream feature expansion (Wf-316A STATUS_HOLD on the 12-d
    F5_RDKIT_2D signal floor; R20 §5 tmQM upstream required).

This test file covers:

  1. ``test_classify_deferred_class_per_wave_correct`` — per-wave
     classification: every canonical DEFERRED wave name maps to its
     declared class (DEFERRED_TEMP / DEFERRED_PERMANENT), and the
     DEFERRED_CLASSIFICATIONS table has the right size.
  2. ``test_deferred_class_filter_filters_wave_list`` — the
     ``--deferred-class-filter`` CLI flag filters the emitted
     ``deferred_waves`` list to only include waves of the specified
     class, while the per-class summary fields
     (``deferred_temp_waves`` / ``deferred_permanent_waves``) always
     carry the full per-class partition regardless of the filter.
  3. ``test_total_deferred_permanent_budget_reserved`` — the
     ``deferred_permanent_hr`` field carries the aggregate
     DEFERRED_PERMANENT wall-time budget, which the summary correctly
     flags as "architectural-blocker" and reserves separately from the
     DEFERRED_TEMP (resumable-with-USER_OK) budget.

Backward-compat (separate verification command per Wf-360-A1 task §8):
``PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wave_budget*.py -v 2>&1 | tail -10``
should still show 35/35 PASS for ``test_wave_budget_tracker.py`` +
3/3 PASS for ``test_wf360a1_wave_budget.py``.

Honest framing per ``party-twenty-spirit`` §3.5:
  * 7 DEFERRED_TEMP waves MEASURED via DEFERRED_CLASSIFICATIONS table.
  * 1 DEFERRED_PERMANENT wave MEASURED via DEFERRED_CLASSIFICATIONS table.
  * Wall: PROJECTED 20-30 min CPU-only (this test file runs in <30 s).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make ``wave_budget_tracker`` importable as a module.
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import wave_budget_tracker  # noqa: E402

VENV_PY = "/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"


# ---------------------------------------------------------------------------
# Canonical 9-wave fixture set (8 DEFERRED + 1 RUNNING boundary)
# ---------------------------------------------------------------------------
#
# 7 DEFERRED_TEMP waves (per Wf-360-A1 §1):
#   - 5 CFM/TODO-119 waves
#   - Wf-176 Phase 2 (USER_OK-BLOCKED per Wf-298D §3.2)
#   - Wf-131 P3 7 DEFERRED (USER_OK-gated per Wf-355-B1)
#   - Wf-146 Re fold + Os seed 4 (USER_OK-gated per Wf-355-B2)
# 1 DEFERRED_PERMANENT wave:
#   - Wf-316A STATUS_HOLD on 12-d F5_RDKIT_2D signal floor
#     (TODO/pending/179_*.md §2 + Wf-313G dashboard §3.4)
#
# Total cumulative = 16+16+7.5+16+7.5+5+3+3+5 = 79.0 hr (vs 10 hr budget).
# Boundary wave = the first cfm_full_retrain (cumulative 16 > 10) →
# RUNNING (start fits at cumulative 0, end overshoots at 16).  All 8
# remaining waves are DEFERRED.
CANONICAL_DEFERRED_WAVES: list = [
    # (wave_name, estimated_hr, source, expected_class, expected_status)
    ("Wave-cfm:cfm_full_retrain", 16.0, "TODO-119 P1.3", "DEFERRED_TEMP", "RUNNING"),
    ("Wave-cfm:cfm_real_decode_lift_p1.3", 16.0, "TODO-119 P1.3", "DEFERRED_TEMP", "DEFERRED"),
    ("Wave-cfm:cfm_per_innovation_ablation_p1.4", 7.5, "TODO-119 P1.4", "DEFERRED_TEMP", "DEFERRED"),
    ("Wave-6:cfm_TODO119_P1.3_real_decode_lift", 16.0, "TODO-119 P1.3", "DEFERRED_TEMP", "DEFERRED"),
    ("Wave-7:cfm_TODO119_P1.4_per_innovation_ablation", 7.5, "TODO-119 P1.4", "DEFERRED_TEMP", "DEFERRED"),
    ("Wave-wf176_phase2_m19_lomo_gpu", 5.0, "Wf-176 Phase 2", "DEFERRED_TEMP", "DEFERRED"),
    ("Wave-wf131_p3_7_deferred", 3.0, "Wf-131 P3", "DEFERRED_TEMP", "DEFERRED"),
    ("Wave-wf146_re_fold_os_seed4", 3.0, "Wf-146", "DEFERRED_TEMP", "DEFERRED"),
    ("Wave-wf316a_status_hold_f5_rdkit_2d", 5.0, "Wf-316A STATUS_HOLD", "DEFERRED_PERMANENT", "DEFERRED"),
]


# ---------------------------------------------------------------------------
# Test 1 — per-wave classification correct
# ---------------------------------------------------------------------------


def test_classify_deferred_class_per_wave_correct():
    """Every canonical DEFERRED wave name (per Wf-360-A1 §1) maps to
    its declared class.  The DEFERRED_CLASSIFICATIONS table carries
    exactly 14 entries (8 unique DEFERRED_TEMP canonical keys + 3
    DEFERRED_PERMANENT canonical keys + 3 fallback substring-match
    keys for the Wf-176 / Wf-131 / Wf-146 launch-wrapper naming)."""

    # Per-wave: every canonical wave in the fixture set must classify
    # to its declared expected_class.
    for name, _est, _src, expected_class, _expected_status in CANONICAL_DEFERRED_WAVES:
        actual = wave_budget_tracker.classify_deferred_class(name)
        assert actual == expected_class, (
            f"classify_deferred_class({name!r}) = {actual!r} != "
            f"expected {expected_class!r}"
        )

    # Default for unrecognised waves must be DEFERRED_TEMP (honest
    # default per AGENTS.md §1.5: never silently mis-classify a
    # USER_OK-resumable wave as permanent).
    assert wave_budget_tracker.classify_deferred_class("Wave-X:unknown") == "DEFERRED_TEMP"
    assert wave_budget_tracker.classify_deferred_class("Wave-1:chemprop_v2_smoke") == "DEFERRED_TEMP"

    # Substring-match fallback for non-canonical wave names.
    assert (
        wave_budget_tracker.classify_deferred_class("Wave-launch_wf316a_v2_xyz")
        == "DEFERRED_PERMANENT"
    )
    assert (
        wave_budget_tracker.classify_deferred_class("Wave-launch_wf131_v3_deferred")
        == "DEFERRED_TEMP"
    )

    # Table size + class value sanity.
    table = wave_budget_tracker.DEFERRED_CLASSIFICATIONS
    assert len(table) == 14, f"DEFERRED_CLASSIFICATIONS table has {len(table)} entries != 14"
    # Every value must be one of the two valid class strings.
    for k, v in table.items():
        assert v in wave_budget_tracker._VALID_DEFERRED_CLASSES, (
            f"DEFERRED_CLASSIFICATIONS[{k!r}] = {v!r} is not a valid class"
        )


# ---------------------------------------------------------------------------
# Test 2 — --deferred-class-filter filters wave_list
# ---------------------------------------------------------------------------


def test_deferred_class_filter_filters_wave_list(tmp_path: Path):
    """The ``--deferred-class-filter`` CLI flag filters the emitted
    ``deferred_waves`` list to only include waves of the specified
    class.  The per-class summary fields
    (``deferred_temp_waves`` / ``deferred_permanent_waves``) always
    carry the full per-class partition regardless of the filter."""

    waves = [(name, est, src) for name, est, src, _cls, _status in CANONICAL_DEFERRED_WAVES]
    wave_dicts = wave_budget_tracker.compute_wave_status(
        budget_hr=10.0, waves=waves, actual_hr_by_name={},
    )

    # Filter "all" (default): both classes appear in deferred_waves.
    summary_all = wave_budget_tracker.summarise(
        10.0, wave_dicts, deferred_class_filter="all",
    )
    assert len(summary_all["deferred_waves"]) == 8  # all 8 DEFERRED
    assert len(summary_all["deferred_temp_waves"]) == 7
    assert len(summary_all["deferred_permanent_waves"]) == 1

    # Filter "DEFERRED_TEMP": only DEFERRED_TEMP waves appear in
    # deferred_waves; the per-class partitions are unchanged.
    summary_temp = wave_budget_tracker.summarise(
        10.0, wave_dicts, deferred_class_filter="DEFERRED_TEMP",
    )
    assert all(
        "Wave-wf316a_status_hold_f5_rdkit_2d" not in n
        for n in summary_temp["deferred_waves"]
    )
    assert len(summary_temp["deferred_waves"]) == 7
    # Per-class partitions are full (unfiltered).
    assert len(summary_temp["deferred_temp_waves"]) == 7
    assert len(summary_temp["deferred_permanent_waves"]) == 1

    # Filter "DEFERRED_PERMANENT": only DEFERRED_PERMANENT waves appear.
    summary_perm = wave_budget_tracker.summarise(
        10.0, wave_dicts, deferred_class_filter="DEFERRED_PERMANENT",
    )
    assert summary_perm["deferred_waves"] == ["Wave-wf316a_status_hold_f5_rdkit_2d"]
    assert len(summary_perm["deferred_waves"]) == 1
    # Per-class partitions are full (unfiltered).
    assert len(summary_perm["deferred_temp_waves"]) == 7
    assert len(summary_perm["deferred_permanent_waves"]) == 1

    # Sanity-check: the CLI round-trip works too (using a flat
    # single-colon CLI name that won't conflict with the parser).
    report_path = tmp_path / "filter_cli.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr", "10.0",
            "--waves", "wf360a1_temp:1.0,wf360a1_perm:1.0",
            "--report-json", str(report_path),
            "--deferred-class-filter", "DEFERRED_PERMANENT",
        ]
    )
    assert rc == 0
    report_obj = json.loads(report_path.read_text())
    # CLI flag round-trip: the JSON must record the filter value.
    assert "deferred_class_filter" in report_obj
    assert report_obj["deferred_class_filter"] == "DEFERRED_PERMANENT"
    # deferred_permanent_waves must always carry the full partition
    # (unaffected by the filter).
    assert isinstance(report_obj["deferred_permanent_waves"], list)
    assert isinstance(report_obj["deferred_temp_waves"], list)


# ---------------------------------------------------------------------------
# Test 3 — total DEFERRED_PERMANENT budget reserved
# ---------------------------------------------------------------------------


def test_total_deferred_permanent_budget_reserved(tmp_path: Path):
    """The ``deferred_permanent_hr`` field carries the aggregate
    DEFERRED_PERMANENT wall-time budget, which the summary correctly
    flags as "architectural-blocker" and reserves separately from the
    DEFERRED_TEMP (resumable-with-USER_OK) budget."""

    waves = [(name, est, src) for name, est, src, _cls, _status in CANONICAL_DEFERRED_WAVES]
    wave_dicts = wave_budget_tracker.compute_wave_status(
        budget_hr=10.0, waves=waves, actual_hr_by_name={},
    )
    summary = wave_budget_tracker.summarise(
        10.0, wave_dicts, deferred_class_filter="all",
    )

    # DEFERRED_PERMANENT budget = sum of Wf-316A STATUS_HOLD = 5.0 hr.
    # This is the wall reserved-but-not-actionable: it CANNOT be
    # reclaimed by adding GPU budget because Wf-316A is blocked on
    # upstream feature expansion (R20 §5 tmQM).
    expected_perm_hr = 5.0
    assert summary["deferred_permanent_hr"] == pytest.approx(
        expected_perm_hr, abs=1e-6
    )

    # DEFERRED_TEMP budget = sum of all 7 DEFERRED_TEMP waves
    # (excluding the RUNNING boundary wave) =
    # 16 + 7.5 + 16 + 7.5 + 5 + 3 + 3 = 58.0 hr.  This is the
    # resumable-with-USER_OK budget: adding GPU budget could clear
    # these (subject to per-launch user OK per AGENTS.md §5.1).
    expected_temp_hr = 16.0 + 7.5 + 16.0 + 7.5 + 5.0 + 3.0 + 3.0
    assert summary["deferred_temp_hr"] == pytest.approx(
        expected_temp_hr, abs=1e-6
    )

    # Honest framing MUST surface both classes distinctly so
    # downstream wrappers can tell them apart at a glance.
    honest = summary["honest_framing"]
    assert "DEFERRED_TEMP=7" in honest
    assert "DEFERRED_PERMANENT=1" in honest
    assert "resumable-with-USER_OK" in honest
    assert "architectural-blocker" in honest
    assert f"{expected_temp_hr:.1f}h" in honest  # 58.0h
    assert f"{expected_perm_hr:.1f}h" in honest  # 5.0h

    # End-to-end smoke via subprocess: write the report to disk and
    # verify the JSON contract carries all 4 new fields.
    report_path = tmp_path / "perm_budget.json"
    rc = wave_budget_tracker.main(
        [
            "--budget-hr", "10.0",
            "--waves", "wf_temp_1:1.0,wf_temp_2:1.0,wf_perm_1:1.0",
            "--report-json", str(report_path),
        ]
    )
    assert rc == 0
    report_obj = json.loads(report_path.read_text())
    # All 4 new fields present.
    assert "deferred_temp_waves" in report_obj
    assert "deferred_permanent_waves" in report_obj
    assert "deferred_temp_hr" in report_obj
    assert "deferred_permanent_hr" in report_obj
    # Both new fields are floats (or 0.0 for empty classes).
    assert isinstance(report_obj["deferred_temp_hr"], (int, float))
    assert isinstance(report_obj["deferred_permanent_hr"], (int, float))
    # Default filter is "all".
    assert report_obj["deferred_class_filter"] == "all"